use arrow_array::{cast::AsArray, Array, BooleanArray, Datum};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use itertools::Itertools;
use memchr::memmem::Finder;

use crate::{
    compiled_kernels::dsl::{DSLKernel, KernelOutputType},
    logical_nulls, ArrowKernelError, Kernel,
};

enum LikeSeq {
    Wildcard,
    AnyChar,
    Literal(u8),
}

impl LikeSeq {
    fn is_wildcard(&self) -> bool {
        matches!(self, LikeSeq::Wildcard)
    }

    fn into_literal_or_any(self) -> LiteralOrAny {
        match self {
            LikeSeq::Wildcard => unreachable!("cannot convert wildcard to literal or any"),
            LikeSeq::AnyChar => LiteralOrAny::AnyChar,
            LikeSeq::Literal(c) => LiteralOrAny::Literal(c),
        }
    }
}

enum LiteralOrAny {
    Literal(u8),
    AnyChar,
}

impl LiteralOrAny {
    fn into_like_seq(self) -> LikeSeq {
        match self {
            LiteralOrAny::Literal(x) => LikeSeq::Literal(x),
            LiteralOrAny::AnyChar => LikeSeq::AnyChar,
        }
    }
}

enum LiteralOrAnySeq {
    OnlyLiteral(Vec<u8>),
    Mixed(Vec<LiteralOrAny>),
}

impl LiteralOrAnySeq {
    fn len(&self) -> usize {
        match self {
            LiteralOrAnySeq::OnlyLiteral(literals) => literals.len(),
            LiteralOrAnySeq::Mixed(literals) => literals.len(),
        }
    }

    fn check_match(&self, haystack: &[u8]) -> bool {
        if haystack.len() != self.len() {
            return false;
        }

        match self {
            LiteralOrAnySeq::OnlyLiteral(v) => haystack == v,
            LiteralOrAnySeq::Mixed(items) => {
                haystack.iter().zip(items.iter()).all(|(h, p)| match p {
                    LiteralOrAny::Literal(l) => *h == *l,
                    LiteralOrAny::AnyChar => true,
                })
            }
        }
    }
}

impl From<Vec<LiteralOrAny>> for LiteralOrAnySeq {
    fn from(literal_or_any: Vec<LiteralOrAny>) -> Self {
        if literal_or_any
            .iter()
            .all(|l| matches!(l, LiteralOrAny::Literal(_)))
        {
            LiteralOrAnySeq::OnlyLiteral(
                literal_or_any
                    .into_iter()
                    .map(|l| match l {
                        LiteralOrAny::Literal(c) => c,
                        _ => unreachable!(),
                    })
                    .collect(),
            )
        } else {
            LiteralOrAnySeq::Mixed(literal_or_any)
        }
    }
}

enum LikeStrategy<'a> {
    General(Vec<LikeSeq>),
    Exact(LiteralOrAnySeq),
    Prefix(LiteralOrAnySeq),
    Infix(Finder<'a>),
    Suffix(LiteralOrAnySeq),
    PrefixSuffix(LiteralOrAnySeq, LiteralOrAnySeq),
    Special {
        prefix: LiteralOrAnySeq,
        infix: Finder<'a>,
        suffix: LiteralOrAnySeq,
    },
}

impl<'a> LikeStrategy<'a> {
    fn minimum_length(&self) -> usize {
        match self {
            LikeStrategy::General(like_seqs) => {
                like_seqs.iter().filter(|p| !p.is_wildcard()).count()
            }
            LikeStrategy::Exact(v) => v.len(),
            LikeStrategy::Prefix(v) => v.len(),
            LikeStrategy::Infix(finder) => finder.needle().len(),
            LikeStrategy::Suffix(v) => v.len(),
            LikeStrategy::PrefixSuffix(p, s) => p.len() + s.len(),
            LikeStrategy::Special {
                prefix,
                infix,
                suffix,
            } => prefix.len() + infix.needle().len() + suffix.len(),
        }
    }

    fn match_string(&self, string: &[u8]) -> bool {
        if string.len() < self.minimum_length() {
            return false;
        }

        match self {
            LikeStrategy::General(like_seqs) => todo!(),
            LikeStrategy::Exact(l) => l.check_match(string),
            LikeStrategy::Prefix(l) => l.check_match(&string[0..l.len()]),
            LikeStrategy::Infix(f) => f.find(string).is_some(),
            LikeStrategy::Suffix(l) => l.check_match(&string[string.len() - l.len()..]),
            LikeStrategy::PrefixSuffix(p, s) => {
                if !p.check_match(&string[0..p.len()]) {
                    return false;
                }
                s.check_match(&string[string.len() - s.len()..])
            }
            LikeStrategy::Special {
                prefix,
                infix,
                suffix,
            } => {
                if !prefix.check_match(&string[0..prefix.len()]) {
                    return false;
                }

                if !suffix.check_match(&string[string.len() - suffix.len()..]) {
                    return false;
                }

                let middle = &string[prefix.len()..string.len() - suffix.len()];
                infix.find(middle).is_some()
            }
        }
    }
}

fn compile_string_like(like_pattern: &[u8], escape: u8) -> Result<LikeStrategy, ArrowKernelError> {
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

    Ok(match num_wildcards {
        0 => LikeStrategy::Exact(
            seq.into_iter()
                .map(|x| x.into_literal_or_any())
                .collect_vec()
                .into(),
        ),
        1 => {
            if seq[0].is_wildcard() {
                // suffix scan
                LikeStrategy::Suffix(
                    seq.into_iter()
                        .skip(1)
                        .map(|x| x.into_literal_or_any())
                        .collect_vec()
                        .into(),
                )
            } else if seq.last().unwrap().is_wildcard() {
                // prefix scan
                seq.pop().unwrap();
                LikeStrategy::Prefix(
                    seq.into_iter()
                        .map(|x| x.into_literal_or_any())
                        .collect_vec()
                        .into(),
                )
            } else {
                // prefix and suffix scan
                let mut seq = seq.into_iter();
                let mut prefix = Vec::new();
                while let Some(v) = seq.next() {
                    if v.is_wildcard() {
                        break;
                    }
                    prefix.push(v.into_literal_or_any());
                }

                let suffix = seq.map(|v| v.into_literal_or_any()).collect_vec();

                LikeStrategy::PrefixSuffix(prefix.into(), suffix.into())
            }
        }
        2 => {
            // prefix, infix, and suffix scan
            let mut seq = seq.into_iter();
            let mut prefix = Vec::new();
            while let Some(v) = seq.next() {
                if v.is_wildcard() {
                    break;
                }
                prefix.push(v.into_literal_or_any());
            }

            let mut infix = Vec::new();
            while let Some(v) = seq.next_back() {
                if v.is_wildcard() {
                    break;
                }
                infix.push(v.into_literal_or_any());
            }

            let suffix = seq.map(|v| v.into_literal_or_any()).collect_vec();
            let infix: LiteralOrAnySeq = infix.into();

            match infix {
                LiteralOrAnySeq::OnlyLiteral(items) => {
                    let finder = Finder::new(&items).into_owned();
                    LikeStrategy::Special {
                        prefix: prefix.into(),
                        infix: finder,
                        suffix: suffix.into(),
                    }
                }
                LiteralOrAnySeq::Mixed(items) => {
                    LikeStrategy::General(items.into_iter().map(|x| x.into_like_seq()).collect())
                }
            }
        }
        _ => LikeStrategy::General(seq),
    })
}

pub fn string_contains(data: &dyn Array, pattern: &[u8]) -> Result<BooleanArray, ArrowKernelError> {
    let finder = Finder::new(pattern);

    if data.null_count() == 0 {
        let mut builder = BooleanBufferBuilder::new(data.len());
        for bytes in crate::arrow_interface::iter::iter_nonnull_bytes(data)? {
            builder.append(finder.find(bytes).is_some());
        }
        return Ok(BooleanArray::from(builder.finish()));
    }

    let mut last_position = 0;
    let mut builder = BooleanBufferBuilder::new(data.len());
    for (idx, bytes) in crate::arrow_interface::iter::iter_nonnull_bytes(data)?.indexed() {
        if idx > last_position {
            builder.append_n(idx - last_position, false);
        }
        builder.append(finder.find(bytes).is_some());
        last_position = idx + 1;
    }

    if last_position < data.len() {
        builder.append_n(data.len() - last_position, false);
    }

    Ok(BooleanArray::from(builder.finish()))
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
    use crate::compiled_kernels::Kernel;
    use arrow_array::{BooleanArray, Scalar, StringArray, StringViewArray};

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
}
