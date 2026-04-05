use std::{
    collections::{BTreeMap, HashMap},
    fmt::Error,
};

use itertools::Itertools;
use lincomdb::LinComDB;
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveResult {
    Exact(usize),
    AtLeast(usize),
}

impl ResolveResult {
    pub fn as_exact(&self) -> Option<usize> {
        match self {
            ResolveResult::Exact(s) => Some(*s),
            ResolveResult::AtLeast(_) => None,
        }
    }
}

pub struct Resolver {
    st: BTreeMap<String, usize>,
    solutions: Vec<Vec<Ratio<i128>>>,
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
            solutions.push(db.query(slice).unwrap());
        }

        Ok(Self { st, solutions })
    }

    pub fn resolve(&self, known: &[usize]) -> Result<Vec<usize>, String> {
        let result_sizes: Vec<usize> = self
            .solutions
            .iter()
            .map(|soln| {
                known
                    .iter()
                    .zip(soln.iter())
                    .map(|(i, c)| Ratio::<i128>::from(*i as i128) * *c)
                    .map(|s| {
                        s.is_integer()
                            .then(|| s.to_integer() as i64)
                            .ok_or_else(|| format!("got non-integer final size {}", s))
                    })
                    .try_fold(0_i64, |acc, i| i.map(|i| acc + i))
                    .and_then(|i| {
                        if i < 0 {
                            Err(format!("negative output size {}", i))
                        } else {
                            Ok(i as usize)
                        }
                    })
            })
            .try_collect()?;

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
        assert_eq!(resolver.resolve(&[3, 6, 4]).unwrap(), vec![1, 2, 5]);
    }
}
