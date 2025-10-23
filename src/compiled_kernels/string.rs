use arrow_array::{cast::AsArray, Array, BinaryArray, BooleanArray, Datum};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use itertools::Itertools;
use memchr::memmem::Finder;

use crate::{
    compiled_kernels::dsl::{DSLKernel, KernelOutputType},
    logical_nulls, ArrowKernelError, Kernel,
};

#[derive(Clone, Copy, Debug)]
enum LikeSeq {
    Wildcard,
    AnyChar,
    Literal(u8),
}

impl LikeSeq {
    fn is_wildcard(&self) -> bool {
        matches!(self, LikeSeq::Wildcard)
    }

    fn is_any(&self) -> bool {
        matches!(self, LikeSeq::AnyChar)
    }

    fn into_literal(self) -> u8 {
        match self {
            LikeSeq::Wildcard => unreachable!("cannot convert wildcard to literal"),
            LikeSeq::AnyChar => unreachable!("cannot convert any char to literal"),
            LikeSeq::Literal(c) => c,
        }
    }
}

fn masked_compare(str: &[u8], pattern: &[u8], mask: &[u32]) -> bool {
    for idx in mask {
        if str[*idx as usize] != pattern[*idx as usize] {
            return false;
        }
    }
    true
}

/// Fast LIKE matcher over raw bytes using a greedy wildcard with checkpoint
/// backtracking. Inspired by Kirk Krauss' algorithm for '*'/'?' globbing.
fn match_like(s: &[u8], pat: &[LikeSeq]) -> bool {
    use LikeSeq::*;

    let mut i = 0usize; // index in s
    let mut j = 0usize; // index in pat

    // Last seen wildcard checkpoint:
    // - star_j: index of the wildcard in pat
    // - star_i: the position in s that wildcard currently covers up to (exclusive).
    let mut star_j: Option<usize> = None;
    let mut star_i: usize = 0;

    while i < s.len() {
        if j < pat.len() {
            match pat[j] {
                Literal(b) if s[i] == b => {
                    i += 1;
                    j += 1;
                    continue;
                }
                AnyChar => {
                    i += 1;
                    j += 1;
                    continue;
                }
                Wildcard => {
                    // If the wildcard is the last token, it matches the rest
                    if j + 1 == pat.len() {
                        return true;
                    }
                    // Record checkpoint: wildcard can expand later if needed.
                    star_j = Some(j);
                    star_i = i;
                    j += 1; // try to match the token after the wildcard
                    continue;
                }
                _ => { /* fall through to mismatch handling */ }
            }
        }

        // Mismatch: if we saw a previous wildcard, expand it to eat one more byte.
        if let Some(sj) = star_j {
            star_i += 1; // make wildcard consume one more byte
            if star_i > s.len() {
                return false; // nothing left to consume
            }
            i = star_i; // retry matching after the wildcard
            j = sj + 1; // right after the wildcard in the pattern
        } else {
            return false; // no wildcard to save us
        }
    }

    // We've consumed all of s. Remaining pattern must be all wildcards.
    while j < pat.len() && matches!(pat[j], Wildcard) {
        j += 1;
    }
    j == pat.len()
}

fn filter_bytes<F: Fn(&[u8]) -> bool>(
    arr: &dyn Array,
    f: F,
) -> Result<BooleanArray, ArrowKernelError> {
    let mut builder = BooleanBufferBuilder::new(arr.len());
    if arr.is_nullable() {
        let mut last_idx = 0;
        for (idx, bytes) in crate::arrow_interface::iter::iter_nonnull_bytes(arr)?.indexed() {
            builder.append_n(idx - last_idx, false);
            last_idx = idx + 1;
            builder.append(f(bytes));
        }
        for _ in last_idx..arr.len() {
            builder.append(false);
        }
    } else {
        for bytes in crate::arrow_interface::iter::iter_nonnull_bytes(arr)? {
            builder.append(f(bytes));
        }
    }
    Ok(BooleanArray::new(builder.finish(), None))
}

/// Compile a string pattern into a function that can be used to filter arrays.
/// This has the same semantics as `arrow_interface::cmp::like`. Reusing the
/// same compiled pattern is more efficient than compiling it each time.
///
/// ```
/// use arrow_array::StringArray;
/// use arrow_compile_compute::compile_string_like;
/// let data = StringArray::from(vec!["hello", "world", "hello world"]);
/// let like = compile_string_like(b"hell%", b'\\').unwrap();
/// let result = like(&data).unwrap();
/// let values: Vec<bool> = result.iter().map(|x| x.unwrap()).collect();
/// assert_eq!(values, vec![true, false, true]);
/// ```
pub fn compile_string_like(
    like_pattern: &[u8],
    escape: u8,
) -> Result<Box<dyn Fn(&dyn Array) -> Result<BooleanArray, ArrowKernelError>>, ArrowKernelError> {
    let mut seq = Vec::new();
    let mut i = 0;
    while i < like_pattern.len() {
        match like_pattern[i] {
            b'%' => {
                if !seq
                    .last()
                    .map(|x: &LikeSeq| x.is_wildcard())
                    .unwrap_or(false)
                {
                    seq.push(LikeSeq::Wildcard);
                }
                i += 1;
            }
            b'_' => seq.push(LikeSeq::AnyChar),
            x if x == escape => {
                if i + 1 < like_pattern.len() {
                    seq.push(LikeSeq::Literal(like_pattern[i + 1]));
                    i += 2;
                } else {
                    return Err(ArrowKernelError::UnsupportedArguments(
                        "LIKE pattern ended with escape character".to_string(),
                    ));
                }
            }
            x => {
                seq.push(LikeSeq::Literal(x));
                i += 1;
            }
        }
    }

    let num_wildcards = seq.iter().filter(|seq| seq.is_wildcard()).count();
    let has_masked = seq.iter().any(|seq| seq.is_any());

    Ok(match (num_wildcards, has_masked) {
        (0, false) => {
            // exact match
            let pattern = seq.into_iter().map(|seq| seq.into_literal()).collect_vec();
            let pattern = BinaryArray::new_scalar(pattern);
            Box::new(move |arr| crate::arrow_interface::cmp::eq(&arr, &pattern))
        }
        (0, true) => {
            // masked match
            let mut mask = Vec::new();
            let seq = seq
                .iter()
                .enumerate()
                .map(|(idx, i)| match i {
                    LikeSeq::Wildcard => unreachable!(),
                    LikeSeq::AnyChar => {
                        mask.push(idx as u32);
                        b'_'
                    }
                    LikeSeq::Literal(l) => *l,
                })
                .collect_vec();

            Box::new(move |arr| filter_bytes(arr, |bytes| masked_compare(bytes, &seq, &mask)))
        }
        (1, false) => {
            // single wildcard, no any chars
            if seq[0].is_wildcard() {
                // suffix
                let pattern = seq
                    .into_iter()
                    .skip(1)
                    .map(|i| i.into_literal())
                    .collect_vec();
                let pattern = BinaryArray::new_scalar(pattern);
                println!("pattern: {:?}", pattern);
                Box::new(move |arr| crate::arrow_interface::cmp::ends_with(arr, &pattern))
            } else if seq.last().unwrap().is_wildcard() {
                // prefix
                let seq_len = seq.len();
                let pattern = seq
                    .into_iter()
                    .take(seq_len - 1)
                    .map(|i| i.into_literal())
                    .collect_vec();

                let pattern = BinaryArray::new_scalar(pattern);
                Box::new(move |arr| crate::arrow_interface::cmp::starts_with(arr, &pattern))
            } else {
                // prefix and suffix
                let wildcard_idx = seq.iter().position(|c| c.is_wildcard()).unwrap();
                let prefix = seq[0..wildcard_idx]
                    .iter()
                    .copied()
                    .map(|i| i.into_literal())
                    .collect_vec();
                let suffix = seq[wildcard_idx + 1..]
                    .iter()
                    .copied()
                    .map(|i| i.into_literal())
                    .collect_vec();
                let min_length = prefix.len() + suffix.len();

                Box::new(move |arr| {
                    filter_bytes(arr, |b| {
                        b.len() >= min_length && b.starts_with(&prefix) && b.ends_with(&suffix)
                    })
                })
            }
        }
        (1, true) => {
            // single wildcard with any chars
            if seq[0].is_wildcard() {
                // suffix
                let mut mask = Vec::new();
                let pattern = seq
                    .iter()
                    .skip(1)
                    .enumerate()
                    .map(|(idx, i)| match i {
                        LikeSeq::Wildcard => unreachable!(),
                        LikeSeq::AnyChar => {
                            mask.push(idx as u32);
                            b'_'
                        }
                        LikeSeq::Literal(l) => *l,
                    })
                    .collect_vec();
                let l = pattern.len();

                Box::new(move |arr| {
                    filter_bytes(arr, |b| {
                        b.len() > l && masked_compare(&b[b.len() - l..], &pattern, &mask)
                    })
                })
            } else if seq.last().unwrap().is_wildcard() {
                // prefix
                let mut mask = Vec::new();
                let pattern = seq
                    .iter()
                    .take(seq.len() - 1)
                    .enumerate()
                    .map(|(idx, i)| match i {
                        LikeSeq::Wildcard => unreachable!(),
                        LikeSeq::AnyChar => {
                            mask.push(idx as u32);
                            b'_'
                        }
                        LikeSeq::Literal(l) => *l,
                    })
                    .collect_vec();
                let l = pattern.len();

                Box::new(move |arr| {
                    filter_bytes(arr, |b| {
                        b.len() > l && masked_compare(&b[..l], &pattern, &mask)
                    })
                })
            } else {
                // prefix and suffix
                let wildcard_idx = seq.iter().position(|c| c.is_wildcard()).unwrap();
                let mut prefix_mask = Vec::new();
                let mut suffix_mask = Vec::new();
                let prefix = seq[..wildcard_idx]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, i)| match i {
                        LikeSeq::Wildcard => unreachable!(),
                        LikeSeq::AnyChar => {
                            prefix_mask.push(idx as u32);
                            b'_'
                        }
                        LikeSeq::Literal(l) => l,
                    })
                    .collect_vec();
                let suffix = seq[wildcard_idx + 1..]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, i)| match i {
                        LikeSeq::Wildcard => unreachable!(),
                        LikeSeq::AnyChar => {
                            suffix_mask.push(idx as u32);
                            b'_'
                        }
                        LikeSeq::Literal(l) => l,
                    })
                    .collect_vec();
                let min_length = prefix.len() + suffix.len();

                Box::new(move |arr| {
                    filter_bytes(arr, |b| {
                        b.len() >= min_length
                            && masked_compare(&b[..prefix.len()], &prefix, &prefix_mask)
                            && masked_compare(&b[b.len() - suffix.len()..], &suffix, &suffix_mask)
                    })
                })
            }
        }
        (2, false) => {
            // two wildcards no anychars
            let wc1 = seq.iter().position(|i| i.is_wildcard()).unwrap();
            let wc2 = seq.iter().rposition(|i| i.is_wildcard()).unwrap();

            let prefix = seq[..wc1].iter().map(|i| i.into_literal()).collect_vec();
            let infix = seq[wc1 + 1..wc2]
                .iter()
                .map(|i| i.into_literal())
                .collect_vec();
            let suffix = seq[wc2 + 1..]
                .iter()
                .map(|i| i.into_literal())
                .collect_vec();

            let finder = Finder::new(&infix).into_owned();
            let min_len = prefix.len() + infix.len() + suffix.len();

            Box::new(move |arr| {
                filter_bytes(arr, |b| {
                    b.len() >= min_len
                        && b[..prefix.len()] == prefix
                        && b[b.len() - suffix.len()..] == suffix
                        && finder
                            .find(&b[prefix.len()..b.len() - suffix.len()])
                            .is_some()
                })
            })
        }
        _ => {
            let min_len = seq.iter().filter(|i| !i.is_wildcard()).count();
            Box::new(move |arr| filter_bytes(arr, |b| b.len() >= min_len && match_like(b, &seq)))
        }
    })
}

pub fn string_contains(data: &dyn Array, pattern: &[u8]) -> Result<BooleanArray, ArrowKernelError> {
    let finder = Finder::new(pattern);
    filter_bytes(data, |b| finder.find(b).is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StringKernelType {
    StartsWith,
    EndsWith,
}

pub struct StringStartEndKernel(DSLKernel);
unsafe impl Sync for StringStartEndKernel {}
unsafe impl Send for StringStartEndKernel {}

impl Kernel for StringStartEndKernel {
    type Key = (DataType, DataType, bool, StringKernelType);

    type Input<'a>
        = (&'a dyn Array, &'a dyn Datum)
    where
        Self: 'a;

    type Params = StringKernelType;

    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (needle, is_scalar) = inp.1.get();
        if is_scalar && needle.is_null(0) {
            return Ok(BooleanArray::from(vec![false; inp.0.len()]));
        }

        let mut res = self.0.call(&[&inp.0, inp.1])?.as_boolean().clone();
        if let Some(nulls) = logical_nulls(inp.0)? {
            let b1 = nulls.inner();
            let b2 = res.values();
            res = BooleanArray::from(b1 & b2);
        }
        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, needle) = inp;

        Ok(StringStartEndKernel(
            DSLKernel::compile(&[arr, *needle], |ctx| {
                let arr = ctx.get_input(0)?;
                let needle = ctx.get_input(1)?;
                ctx.iter_over(vec![arr, needle])
                    .map(|i| match params {
                        StringKernelType::StartsWith => vec![i[0].starts_with(&i[1])],
                        StringKernelType::EndsWith => vec![i[0].ends_with(&i[1])],
                    })
                    .collect(KernelOutputType::Boolean)
            })
            .map_err(ArrowKernelError::DSLError)?,
        ))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let haystack_type = i.0.data_type().clone();
        let (needle_data, is_scalar) = i.1.get();
        Ok((
            haystack_type,
            needle_data.data_type().clone(),
            is_scalar,
            *p,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{StringKernelType, StringStartEndKernel};
    use crate::compiled_kernels::{string::compile_string_like, string_contains, Kernel};
    use arrow_array::{BooleanArray, Scalar, StringArray, StringViewArray};
    use itertools::Itertools;

    #[test]
    fn test_string_starts_with_kernel() {
        let source = StringArray::from(vec!["foobar", "barfoo", "foobaz"]);
        let needle = Scalar::new(StringArray::from(vec!["foo"]));
        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::StartsWith)
                .unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![true, false, true]));
    }

    #[test]
    fn test_string_ends_with_kernel() {
        let source = StringArray::from(vec!["foobar", "barfoo", "bazfoo"]);
        let needle = Scalar::new(StringArray::from(vec!["foo"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::EndsWith).unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![false, true, true]));
    }

    #[test]
    fn test_string_starts_with_kernel_nulls() {
        let source = StringArray::from(vec![
            Some("prefix-value"),
            None,
            Some("other"),
            Some("prefix"),
        ]);
        let needle = Scalar::new(StringArray::from(vec!["prefix"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::StartsWith)
                .unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![true, false, false, true]));
    }

    #[test]
    fn test_string_view_ends_with_kernel() {
        let source = StringViewArray::from(vec!["alpha", "beta-suffix", "suffix"]);
        let needle = Scalar::new(StringArray::from(vec!["suffix"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::EndsWith).unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![false, true, true]));
    }

    #[test]
    fn test_string_contains() {
        let data = StringArray::from(vec!["hello", "world", "hello world"]);
        let pattern = b"llo";

        let result = string_contains(&data, pattern).unwrap();

        assert_eq!(result, BooleanArray::from(vec![true, false, true]));
    }

    #[test]
    fn test_string_like() {
        let data = StringArray::from(vec!["hello", "world", "hello world"]);

        let prefix = compile_string_like(b"hell%", b'\\').unwrap();
        let result = prefix(&data)
            .unwrap()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(result, vec![true, false, true]);

        let suffix = compile_string_like(b"%world", b'\\').unwrap();
        let result = suffix(&data)
            .unwrap()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(result, vec![false, true, true]);

        let infix = compile_string_like(b"%hello%", b'\\').unwrap();
        let result = infix(&data)
            .unwrap()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(result, vec![true, false, true]);

        let pands = compile_string_like(b"h%d", b'\\').unwrap();
        let result = pands(&data)
            .unwrap()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(result, vec![false, false, true]);

        let arb = compile_string_like(b"h%o%o%", b'\\').unwrap();
        let result = arb(&data).unwrap().iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(result, vec![false, false, true]);
    }
}
