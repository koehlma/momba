//! Verdict domains.

use std::{fmt, hash::Hash, str::FromStr};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::lattice::{HasTop, Join, Meet, PartiallyOrdered, Poset, Set};

/// Error parsing a value from a string.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParseError(String);

/// Klenee's [three-valued truth domain](https://en.wikipedia.org/wiki/Three-valued_logic).
///
/// ```plain
///   Unknown
///    /   \
/// True   False
/// ```
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bool3 {
    /// Unknown truth value.
    Unknown,
    /// True.
    True,
    /// False.
    False,
}

impl Bool3 {
    /// Returns `true` if and only if the truth value is [`Bool3::Unknown`].
    pub fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Returns `true` if and only if the truth value is [`Bool3::True`].
    pub fn is_true(self) -> bool {
        matches!(self, Self::True)
    }

    /// Returns `true` if and only if the truth value is [`Bool3::False`].
    pub fn is_false(self) -> bool {
        matches!(self, Self::False)
    }

    /// Negates the value.
    pub fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            _ => self,
        }
    }

    /// Computes the conjunction of two values.
    pub fn and(self, other: Self) -> Self {
        use Bool3::*;
        match (self, other) {
            (_, False) | (False, _) => False,
            (True, True) => True,
            _ => Unknown,
        }
    }

    /// Computes the disjunction of two values.
    pub fn or(self, other: Self) -> Self {
        use Bool3::*;
        match (self, other) {
            (_, True) | (True, _) => True,
            (False, False) => False,
            _ => Unknown,
        }
    }
}

impl fmt::Display for Bool3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl FromStr for Bool3 {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Bool3::*;
        match s {
            "U" | "-" | "Unknown" | "unknown" | "UNKNOWN" => Ok(Unknown),
            "T" | "1" | "True" | "true" | "TRUE" => Ok(True),
            "F" | "0" | "False" | "false" | "FALSE" => Ok(False),
            _ => Err(ParseError(format!("`{s}` is not a valid Bool3"))),
        }
    }
}

impl PartiallyOrdered for Bool3 {
    #[inline]
    fn is_less_than(&self, other: &Self) -> bool {
        use Bool3::*;
        match (self, other) {
            (Unknown, _) => false,
            (_, Unknown) => true,
            _ => false,
        }
    }
}

impl Join for Bool3 {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        use Bool3::*;
        match (self, other) {
            (True, True) => True,
            (False, False) => False,
            _ => Unknown,
        }
    }
}

// impl HasTop for Bool3 {
//     #[inline(always)]
//     fn top() -> Self
//     where
//         Self: Sized,
//     {
//         Self::Unknown
//     }

//     #[inline(always)]
//     fn is_top(&self) -> bool {
//         self.is_unknown()
//     }
// }

/// Belnap's [four-valued truth domain](https://en.wikipedia.org/wiki/Four-valued_logic).
///
/// ```plain
///     Both
///    /    \
/// True    False
///    \    /
///     None
/// ```
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bool4 {
    /// Truth value is *true or false*.
    Both,
    /// True.
    True,
    /// False.
    False,
    /// Truth value is *neither true nor false*.
    None,
}

impl Bool4 {
    /// Returns `true` if and only if the truth value is [`Bool4::Both`].
    pub fn is_both(self) -> bool {
        matches!(self, Self::Both)
    }

    /// Returns `true` if and only if the truth value is [`Bool4::True`].
    pub fn is_true(self) -> bool {
        matches!(self, Self::True)
    }

    /// Returns `true` if and only if the truth value is [`Bool4::Both`] or
    /// [`Bool4::True`].
    pub fn maybe_true(self) -> bool {
        self.is_both() || self.is_true()
    }

    /// Returns `true` if and only if the truth value is [`Bool4::False`].
    pub fn is_false(self) -> bool {
        matches!(self, Self::False)
    }

    /// Returns `true` if and only if the truth value is [`Bool4::Both`] or
    /// [`Bool4::False`].
    pub fn maybe_false(self) -> bool {
        self.is_both() || self.is_false()
    }

    /// Returns `true` if and only if the truth value is [`Bool4::None`].
    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    /// Negates the value.
    pub fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            _ => self,
        }
    }

    /// Computes the conjunction of two values.
    pub fn and(self, other: Self) -> Self {
        use Bool4::*;
        match (self, other) {
            (_, False) | (False, _) => False,
            (None, Both) | (Both, None) => False,
            (None, None) => None,
            (True, right) => right,
            (left, True) => left,
            _ => Both,
        }
    }

    /// Computes the disjunction of two values.
    pub fn or(self, other: Self) -> Self {
        use Bool4::*;
        match (self, other) {
            (_, True) | (True, _) => True,
            (None, Both) | (Both, None) => True,
            (Both, Both) => Both,
            (False, right) => right,
            (left, False) => left,
            _ => None,
        }
    }
}

impl From<Bool3> for Bool4 {
    fn from(value: Bool3) -> Self {
        match value {
            Bool3::Unknown => Self::Both,
            Bool3::True => Self::True,
            Bool3::False => Self::False,
        }
    }
}

impl fmt::Display for Bool4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl FromStr for Bool4 {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Bool4::*;
        match s {
            // We allow unknown for compatibility with `Bool3`.
            "U" | "-" | "Unknown" | "unknown" | "UNKNOWN" => Ok(Both),
            "T" | "1" | "True" | "true" | "TRUE" => Ok(True),
            "F" | "0" | "False" | "false" | "FALSE" => Ok(False),
            "B" | "Both" | "both" | "BOTH" => Ok(None),
            "N" | "None" | "none" | "NONE" => Ok(None),
            _ => Err(ParseError(format!("`{s}` is not a valid Bool3"))),
        }
    }
}

impl PartiallyOrdered for Bool4 {
    fn is_less_than(&self, other: &Self) -> bool {
        use Bool4::*;
        match (self, other) {
            (Both, _) => false,
            (_, Both) => true,
            (_, None) => false,
            (None, _) => true,
            _ => false,
        }
    }
}

impl Meet for Bool4 {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        use Bool4::*;
        match (self, other) {
            (None, _) | (_, None) => None,
            (True, Both) | (Both, True) | (True, True) => True,
            (False, Both) | (Both, False) | (False, False) => False,
            (True, False) | (False, True) => None,
            _ => Both,
        }
    }
}

impl Join for Bool4 {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        use Bool4::*;
        match (self, other) {
            (Both, _) | (_, Both) => Both,
            (True, None) | (None, True) | (True, True) => True,
            (False, None) | (None, False) | (False, False) => False,
            (True, False) | (False, True) => Both,
            _ => None,
        }
    }
}

// impl HasTop for Bool4 {
//     fn top() -> Self
//     where
//         Self: Sized,
//     {
//         Self::Both
//     }

//     #[inline(always)]
//     fn is_top(&self) -> bool {
//         self.is_both()
//     }
// }

// impl HasBottom for Bool4 {
//     fn bottom() -> Self
//     where
//         Self: Sized,
//     {
//         Self::None
//     }

//     fn is_bottom(&self) -> bool {
//         self.is_none()
//     }
// }

pub struct PowerSet<T: Clone + Hash + Eq> {
    empty_set: Set<T>,
}

impl<T: Clone + Hash + Eq> PowerSet<T> {
    pub fn new() -> Self {
        Self {
            empty_set: Set::new(),
        }
    }
    pub fn singleton(&self, value: T) -> Set<T> {
        let mut set = self.empty_set.clone();
        set.insert(value);
        set
    }
}

impl<T: Clone + Hash + Eq> Poset for PowerSet<T> {
    type Element = Set<T>;
}

impl<T: Clone + Hash + Eq> HasTop for PowerSet<T> {
    fn top(&self) -> Self::Element {
        self.empty_set.clone()
    }

    fn is_top(&self, element: &Self::Element) -> bool {
        element.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_order {
        ($value:expr, [$($lt:expr),*], [$($nlt:expr),*]) => {
            assert!(!$value.is_less_than($value));
            $(assert!($value.is_less_than($lt));)*
            $(assert!(!$value.is_less_than($nlt));)*
        };
    }

    macro_rules! assert_table_binary {
        ([$($values:expr),*], $func:ident, $table:expr) => {{
            let table = $table;
            for (i, left) in [$($values),*].into_iter().enumerate() {
                for (j, right) in [$($values),*].into_iter().enumerate() {
                    assert_eq!(
                        left.$func(right),
                        table[i][j],
                        "{}({left:?}, {right:?}) is {:?}, expected {:?}",
                        stringify!($func),
                        left.$func(right),
                        table[i][j]
                    );
                }
            }
        }};
    }

    #[test]
    fn test_bool3_order() {
        use Bool3::*;
        assert_order!(&Unknown, [], [&True, &False]);
        assert_order!(&True, [&Unknown], [&False]);
        assert_order!(&False, [&Unknown], [&True]);
    }

    #[test]
    fn test_bool3_not() {
        use Bool3::*;
        assert_eq!(Unknown.not(), Unknown);
        assert_eq!(True.not(), False);
        assert_eq!(False.not(), True);
    }

    #[test]
    fn test_bool3_and() {
        use Bool3::*;
        assert_table_binary!(
            [Unknown, True, False],
            and,
            [
                [Unknown, Unknown, False],
                [Unknown, True, False],
                [False, False, False]
            ]
        );
    }

    #[test]
    fn test_bool3_join() {
        use Bool3::*;
        assert_table_binary!(
            [Unknown, True, False],
            join_move,
            [
                [Unknown, Unknown, Unknown],
                [Unknown, True, Unknown],
                [Unknown, Unknown, False]
            ]
        );
    }

    #[test]
    fn test_bool3_or() {
        use Bool3::*;
        assert_table_binary!(
            [Unknown, True, False],
            or,
            [
                [Unknown, True, Unknown],
                [True, True, True],
                [Unknown, True, False]
            ]
        );
    }

    #[test]
    fn test_bool4_order() {
        use Bool4::*;
        assert_order!(&Both, [], [&None, &True, &False]);
        assert_order!(&True, [&Both], [&None, &False]);
        assert_order!(&False, [&Both], [&None, &True]);
        assert_order!(&None, [&Both, &True, &False], []);
    }

    #[test]
    fn test_bool4_not() {
        use Bool4::*;
        assert_eq!(Both.not(), Both);
        assert_eq!(True.not(), False);
        assert_eq!(False.not(), True);
        assert_eq!(None.not(), None);
    }

    #[test]
    fn test_bool4_and() {
        use Bool4::*;
        assert_table_binary!(
            [Both, True, False, None],
            and,
            [
                [Both, Both, False, False],
                [Both, True, False, None],
                [False, False, False, False],
                [False, None, False, None]
            ]
        );
    }

    #[test]
    fn test_bool4_or() {
        use Bool4::*;
        assert_table_binary!(
            [Both, True, False, None],
            or,
            [
                [Both, True, Both, True],
                [True, True, True, True],
                [Both, True, False, None],
                [True, True, None, None]
            ]
        );
    }

    #[test]
    fn test_bool4_join() {
        use Bool4::*;
        assert_table_binary!(
            [Both, True, False, None],
            join_move,
            [
                [Both, Both, Both, Both],
                [Both, True, Both, True],
                [Both, Both, False, False],
                [Both, True, False, None],
            ]
        );
    }
    #[test]
    fn test_bool4_meet() {
        use Bool4::*;
        assert_table_binary!(
            [Both, True, False, None],
            meet_move,
            [
                [Both, True, False, None],
                [True, True, None, None],
                [False, None, False, None],
                [None, None, None, None],
            ]
        );
    }
}
