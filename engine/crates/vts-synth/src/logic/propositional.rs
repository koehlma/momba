//! Symbolic propositional logic feature guards.

use std::{fmt::Write, str::FromStr};

mod parser;

/// A symbolic propositional logic formula usable as a guard.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Formula<F> {
    Atom(F),
    And(Vec<Self>),
    Or(Vec<Self>),
    Xor(Vec<Self>),
    Not(Box<Self>),
    True,
    False,
}

impl<F> Formula<F> {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) => Self::False,
            (_, Self::False) => Self::False,
            (left, Self::True) => left,
            (Self::True, right) => right,
            (Self::And(mut left), Self::And(right)) => {
                left.extend(right.into_iter());
                Self::And(left)
            }
            (Self::And(mut left), right) => {
                left.push(right);
                Self::And(left)
            }
            (left, Self::And(mut right)) => {
                right.insert(0, left);
                Self::And(right)
            }
            (left, right) => Self::And(vec![left, right]),
        }
    }

    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, Self::True) => Self::False,
            (_, Self::True) => Self::True,
            (Self::True, _) => Self::True,
            (left, Self::False) => left,
            (Self::False, right) => right,
            (Self::Or(mut left), Self::Or(right)) => {
                left.extend(right.into_iter());
                Self::Or(left)
            }
            (Self::Or(mut left), right) => {
                left.push(right);
                Self::Or(left)
            }
            (left, Self::Or(mut right)) => {
                right.insert(0, left);
                Self::Or(right)
            }
            (left, right) => Self::Or(vec![left, right]),
        }
    }

    pub fn xor(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) => Self::True,
            (_, Self::True) => Self::True,
            (left, Self::False) => left,
            (Self::False, right) => right,
            (Self::Xor(mut left), Self::Xor(right)) => {
                left.extend(right.into_iter());
                Self::Xor(left)
            }
            (Self::Xor(mut left), right) => {
                left.push(right);
                Self::Xor(left)
            }
            (left, Self::Xor(mut right)) => {
                right.insert(0, left);
                Self::Xor(right)
            }
            (left, right) => Self::Xor(vec![left, right]),
        }
    }

    pub fn not(self) -> Self {
        match self {
            Formula::True => Self::False,
            Formula::False => Self::True,
            Formula::Not(operand) => *operand,
            guard => Self::Not(Box::new(guard)),
        }
    }

    pub fn traverse(&self) -> impl Iterator<Item = &Self> {
        let mut stack = vec![self];
        std::iter::from_fn(move || {
            stack.pop().map(|top| {
                match top {
                    Formula::And(children) | Formula::Or(children) => stack.extend(children),
                    Formula::Not(child) => stack.push(child.as_ref()),
                    _ => {
                        // Nothing to do!
                    }
                };
                top
            })
        })
    }
}

impl<F: std::fmt::Display> std::fmt::Display for Formula<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Formula::Atom(feature) => feature.fmt(f),
            Formula::And(operands) => {
                f.write_char('(')?;
                let mut first = true;
                for operand in operands {
                    if !first {
                        f.write_str(" && ")?;
                    }
                    operand.fmt(f)?;
                    first = false;
                }
                f.write_char(')')
            }
            Formula::Or(operands) => {
                f.write_char('(')?;
                let mut first = true;
                for operand in operands {
                    if !first {
                        f.write_str(" || ")?;
                    }
                    operand.fmt(f)?;
                    first = false;
                }
                f.write_char(')')
            }
            Formula::Xor(operands) => {
                f.write_char('(')?;
                let mut first = true;
                for operand in operands {
                    if !first {
                        f.write_str(" xor ")?;
                    }
                    operand.fmt(f)?;
                    first = false;
                }
                f.write_char(')')
            }
            Formula::Not(operand) => {
                f.write_char('!')?;
                operand.fmt(f)
            }
            Formula::True => f.write_str("true"),
            Formula::False => f.write_str("false"),
        }
    }
}

impl FromStr for Formula<String> {
    type Err = Vec<parser::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (formula, errors) = parser::parse(s);
        formula.ok_or(errors)
    }
}
