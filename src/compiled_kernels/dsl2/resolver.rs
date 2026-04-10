use std::{
    collections::{BTreeMap, HashMap},
    fmt::Error,
};

use inkwell::llvm_sys::orc2::LLVMOrcSymbolPredicate;
use itertools::Itertools;
use lincomdb::{LinComDB, QueryError};
use num_rational::{Ratio, Rational, Rational32};

#[derive(Debug, Clone, PartialEq)]
pub enum SizeTerm {
    Term(String),
    Add(Box<SizeTerm>, Box<SizeTerm>),
    AtLeast(Box<SizeTerm>),
}

impl SizeTerm {
    pub fn parse(input: &str) -> Result<Self, String> {
        let input = input.trim_ascii();
        if input.is_empty() {
            return Err("input ended unexpectedly".to_string());
        }
        if input.starts_with("<=") {
            return Ok(SizeTerm::AtLeast(Box::new(SizeTerm::parse(&input[2..])?)));
        }

        match input.split_once("+") {
            Some((p, s)) => Ok(SizeTerm::Add(
                Box::new(SizeTerm::parse(p)?),
                Box::new(SizeTerm::parse(s)?),
            )),
            None => {
                if !input.chars().all(|c| c.is_alphabetic()) {
                    return Err(format!("invalid term {}", input));
                }
                return Ok(SizeTerm::Term(input.trim_ascii().to_string()));
            }
        }
    }

    pub fn resolve(&self, m: &HashMap<&str, usize>) -> Result<ResolveResult, String> {
        match self {
            SizeTerm::Term(n) => m
                .get(n.as_str())
                .map(|s| ResolveResult::Exact(*s))
                .ok_or_else(|| format!("unbound term: {}", n)),
            SizeTerm::Add(s1, s2) => {
                let s1 = s1
                    .resolve(m)?
                    .as_exact()
                    .ok_or_else(|| "sum was not exact".to_string())?;
                let s2 = s2
                    .resolve(m)?
                    .as_exact()
                    .ok_or_else(|| "sum was not exact".to_string())?;
                Ok(ResolveResult::Exact(s1 + s2))
            }
            SizeTerm::AtLeast(size_term) => {
                let s = size_term
                    .resolve(m)?
                    .as_exact()
                    .ok_or_else(|| "sum was not exact".to_string())?;
                Ok(ResolveResult::AtLeast(s))
            }
        }
    }

    pub fn terms(&self) -> Vec<&str> {
        let mut v = Vec::new();
        match self {
            SizeTerm::Term(s) => v.push(s.as_str()),
            SizeTerm::Add(s1, s2) => {
                v.extend(s1.terms());
                v.extend(s2.terms());
            }
            SizeTerm::AtLeast(s) => {
                v.extend(s.terms());
            }
        };
        v
    }

    pub fn to_vec(&self, st: &BTreeMap<String, usize>, v: &mut [i32]) {
        match self {
            SizeTerm::Term(s) => {
                let idx = st[s];
                v[idx] += 1;
            }
            SizeTerm::Add(s1, s2) => {
                s1.to_vec(st, v);
                s2.to_vec(st, v);
            }
            SizeTerm::AtLeast(s) => {
                s.to_vec(st, v);
            }
        }
    }

    pub fn is_exact(&self) -> bool {
        match self {
            SizeTerm::Term(_) => true,
            SizeTerm::Add(s1, s2) => s1.is_exact() && s2.is_exact(),
            SizeTerm::AtLeast(_) => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveResult {
    Exact(usize),
    AtLeast(usize),
    Unknown,
}

impl ResolveResult {
    pub fn as_exact(&self) -> Option<usize> {
        match self {
            ResolveResult::Exact(s) => Some(*s),
            _ => None,
        }
    }
}

enum Solution {
    Known(Vec<Ratio<i128>>),
    Unknown,
}

pub struct Resolver {
    st: BTreeMap<String, usize>,
    solutions: Vec<Solution>,
    is_input_exact: Vec<bool>,
    is_output_exact: Vec<bool>,
}

impl Resolver {
    pub fn new(terms: Vec<SizeTerm>, outputs: Vec<SizeTerm>) -> Result<Self, String> {
        // first, create an index for each term
        let mut st = BTreeMap::new();

        terms.iter().flat_map(|t| t.terms()).for_each(|s| {
            let len = st.len();
            st.entry(s.to_string()).or_insert(len);
        });
        outputs.iter().flat_map(|t| t.terms()).for_each(|s| {
            let len = st.len();
            st.entry(s.to_string()).or_insert(len);
        });

        let db = LinComDB::from_rows(terms.iter().map(|t| {
            let mut slice = vec![0; st.len()];
            t.to_vec(&st, &mut slice);
            slice
        }))
        .unwrap();

        let mut solutions = Vec::new();
        for out in outputs.iter() {
            let mut slice = vec![0; st.len()];
            out.to_vec(&st, &mut slice);
            solutions.push(match db.query(slice) {
                Ok(s) => Solution::Known(s),
                Err(QueryError::NotInRowSpace) => {
                    return Err(format!("unable to resolve output: {:?}", out));
                }
                Err(QueryError::EmptyDatabase) => Solution::Unknown,
                Err(e) => {
                    panic!("unexpected error: {:?}", e)
                }
            });
        }

        let is_input_exact = terms.iter().map(|x| x.is_exact()).collect_vec();
        let is_output_exact = outputs.iter().map(|x| x.is_exact()).collect_vec();

        Ok(Self {
            st,
            solutions,
            is_input_exact,
            is_output_exact,
        })
    }

    pub fn resolve(&self, known: &[usize]) -> Result<Vec<ResolveResult>, String> {
        let mut result_sizes: Vec<ResolveResult> = Vec::new();

        for (idx, (soln, is_exact)) in self
            .solutions
            .iter()
            .zip(self.is_output_exact.iter())
            .enumerate()
        {
            let mut is_exact = *is_exact;
            match soln {
                Solution::Known(ratios) => {
                    let sum = ratios
                        .iter()
                        .zip(known.iter())
                        .map(|(x, y)| Ratio::<i128>::from(*y as i128) * *x)
                        .fold(Ratio::<i128>::from(0), |acc, v| acc + v);

                    if ratios
                        .iter()
                        .enumerate()
                        .any(|(idx, r)| r != &0.into() && !self.is_input_exact[idx])
                    {
                        is_exact = false;
                    }

                    if sum.is_integer() {
                        let sum = sum.to_integer() as usize;
                        if is_exact {
                            result_sizes.push(ResolveResult::Exact(sum));
                        } else {
                            result_sizes.push(ResolveResult::AtLeast(sum));
                        }
                    } else {
                        return Err(format!("non-integer size for output {} ({})", idx, sum));
                    }
                }
                Solution::Unknown => result_sizes.push(ResolveResult::Unknown),
            }
        }

        Ok(result_sizes)
    }
}

#[cfg(test)]
mod tests {
    use crate::compiled_kernels::dsl2::resolver::{ResolveResult, Resolver, SizeTerm};

    #[test]
    fn test_compile() {
        let input_terms = vec![
            SizeTerm::parse("a + a").unwrap(),
            SizeTerm::parse("b + c").unwrap(),
            SizeTerm::parse("c").unwrap(),
        ];

        let output_terms = vec![
            SizeTerm::parse("a").unwrap(),
            SizeTerm::parse("b").unwrap(),
            SizeTerm::parse("<= a + b").unwrap(),
        ];

        assert!(Resolver::new(input_terms, output_terms).is_ok());
    }

    #[test]
    fn test_parse() {
        assert_eq!(
            SizeTerm::parse("foo"),
            Ok(SizeTerm::Term("foo".to_string())),
        );
        assert_eq!(
            SizeTerm::parse("foo+bar"),
            Ok(SizeTerm::Add(
                Box::new(SizeTerm::Term("foo".to_string())),
                Box::new(SizeTerm::Term("bar".to_string())),
            )),
        );
        assert_eq!(
            SizeTerm::parse("<= foo   +   bar"),
            Ok(SizeTerm::AtLeast(Box::new(SizeTerm::Add(
                Box::new(SizeTerm::Term("foo".to_string())),
                Box::new(SizeTerm::Term("bar".to_string())),
            )))),
        );
    }

    #[test]
    fn test_resolve() {
        let terms = vec![
            SizeTerm::Add(
                Box::new(SizeTerm::Term("a".to_string())),
                Box::new(SizeTerm::Term("b".to_string())),
            ),
            SizeTerm::Add(
                Box::new(SizeTerm::Term("a".to_string())),
                Box::new(SizeTerm::Term("c".to_string())),
            ),
            SizeTerm::Add(
                Box::new(SizeTerm::Term("b".to_string())),
                Box::new(SizeTerm::Term("b".to_string())),
            ),
        ];

        let outputs = vec![
            SizeTerm::Term("a".to_string()),
            SizeTerm::Term("b".to_string()),
            SizeTerm::Term("c".to_string()),
        ];

        let resolver = Resolver::new(terms, outputs).unwrap();
        assert_eq!(
            resolver.resolve(&[3, 6, 4]).unwrap(),
            vec![
                ResolveResult::Exact(1),
                ResolveResult::Exact(2),
                ResolveResult::Exact(5)
            ]
        );
    }
}
