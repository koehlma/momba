use crate::constants::*;

/// Represents a bound.
pub trait Bound: Clone {
    /// The type of the bounding constant.
    type Constant: Constant;

    /// The maximal value of the bounding constant if it exists.
    fn max_constant() -> Option<Self::Constant>;
    /// The minimal value of the bounding constant if it exists.
    fn min_constant() -> Option<Self::Constant>;

    /// Returns the *unbounded bound*, i.e., `< ∞`.
    fn unbounded() -> Self;

    /// Returns the bound for `≤ 0`.
    fn le_zero() -> Self;
    /// Returns the bound for `< 0`.
    fn lt_zero() -> Self;

    /// Constructs a new bound.
    fn new(is_strict: bool, constant: Self::Constant) -> Self;

    /// Constructs a new bound `≤ constant`.
    fn new_le(constant: Self::Constant) -> Self;
    /// Constructs a new bound `< constant`.
    fn new_lt(constant: Self::Constant) -> Self;

    /// Returns whether the bound is strict.
    fn is_strict(&self) -> bool;
    /// Returns whether the bound is the *unbounded bound*.
    fn is_unbounded(&self) -> bool;

    /// Retrieves the constant associated with the bound.
    ///
    /// If the bound is the *unbounded bound*, [None] will be returned.
    fn constant(&self) -> Option<Self::Constant>;

    /// Constructs a new bound by adding the constants of both bounds.
    ///
    /// Returns [None] if adding the constants will lead to an overflow.
    fn add(&self, other: &Self) -> Option<Self>;

    /// Returns whether `self` is a tighter bound than `other`.
    fn is_tighter_than(&self, other: &Self) -> bool;
}

macro_rules! int_bound_impl {
    ($int_type:ty) => {
        /// Implementation of [Bound] for primitive integers.
        ///
        /// Encodes strictness and boundedness within individual bits.
        impl Bound for $int_type {
            type Constant = $int_type;

            #[inline(always)]
            fn max_constant() -> Option<Self::Constant> {
                Some((<$int_type>::MAX >> 1) - 1)
            }
            #[inline(always)]
            fn min_constant() -> Option<Self::Constant> {
                Some(<$int_type>::MIN >> 1)
            }

            #[inline(always)]
            fn unbounded() -> Self {
                <$int_type>::MAX & !1
            }

            #[inline(always)]
            fn le_zero() -> Self {
                1
            }
            #[inline(always)]
            fn lt_zero() -> Self {
                0
            }

            #[inline(always)]
            fn new(is_strict: bool, constant: Self::Constant) -> Self {
                if is_strict {
                    Self::new_lt(constant)
                } else {
                    Self::new_le(constant)
                }
            }

            #[inline(always)]
            fn new_le(constant: Self::Constant) -> Self {
                constant << 1 | 1
            }
            #[inline(always)]
            fn new_lt(constant: Self::Constant) -> Self {
                constant << 1
            }

            #[inline(always)]
            fn is_strict(&self) -> bool {
                self & 1 == 0
            }
            #[inline(always)]
            fn is_unbounded(&self) -> bool {
                *self == Self::unbounded()
            }

            #[inline(always)]
            fn constant(&self) -> Option<Self::Constant> {
                if *self == Self::unbounded() {
                    None
                } else {
                    Some(self >> 1)
                }
            }

            #[inline(always)]
            fn add(&self, other: &Self) -> Option<Self> {
                if (*self == Self::unbounded() || *other == Self::unbounded()) {
                    Some(Self::unbounded())
                } else {
                    let constant = (self >> 1).checked_add(other >> 1)?;
                    if Self::min_constant()? <= constant && constant <= Self::max_constant()? {
                        Some((constant << 1) | (self & 1 | other & 1))
                    } else {
                        None
                    }
                }
            }

            #[inline(always)]
            fn is_tighter_than(&self, other: &Self) -> bool {
                self < other
            }
        }
    };
}

int_bound_impl!(i8);
int_bound_impl!(i16);
int_bound_impl!(i32);
int_bound_impl!(i64);
int_bound_impl!(i128);

/// A bound for a generic [Constant].
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct ConstantBound<C: Constant> {
    /// The constant associated with the bound.
    ///
    /// Is [None] in case the bound is the *unbounded bound*.
    constant: Option<C>,
    /// Indicates whether the bound is strict or not.
    is_strict: bool,
}

impl<C: Constant> Bound for ConstantBound<C> {
    type Constant = C;

    #[inline(always)]
    fn max_constant() -> Option<Self::Constant> {
        C::max_value()
    }
    #[inline(always)]
    fn min_constant() -> Option<Self::Constant> {
        C::min_value()
    }

    #[inline(always)]
    fn unbounded() -> Self {
        ConstantBound {
            constant: None,
            is_strict: true,
        }
    }

    #[inline(always)]
    fn le_zero() -> Self {
        ConstantBound {
            constant: Some(C::zero()),
            is_strict: false,
        }
    }
    #[inline(always)]
    fn lt_zero() -> Self {
        ConstantBound {
            constant: Some(C::zero()),
            is_strict: true,
        }
    }

    #[inline(always)]
    fn new(is_strict: bool, constant: Self::Constant) -> Self {
        ConstantBound {
            constant: Some(constant),
            is_strict,
        }
    }

    #[inline(always)]
    fn new_le(constant: Self::Constant) -> Self {
        ConstantBound {
            constant: Some(constant),
            is_strict: false,
        }
    }
    #[inline(always)]
    fn new_lt(constant: Self::Constant) -> Self {
        ConstantBound {
            constant: Some(constant),
            is_strict: true,
        }
    }

    #[inline(always)]
    fn is_strict(&self) -> bool {
        self.is_strict
    }
    #[inline(always)]
    fn is_unbounded(&self) -> bool {
        self.constant.is_none()
    }

    #[inline(always)]
    fn constant(&self) -> Option<Self::Constant> {
        self.constant.clone()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Option<Self> {
        match &self.constant {
            Some(left) => match &other.constant {
                Some(right) => Some(ConstantBound {
                    constant: Some(left.checked_add(&right)?),
                    is_strict: self.is_strict || other.is_strict,
                }),
                None => Some(Self::unbounded()),
            },
            None => Some(Self::unbounded()),
        }
    }

    #[inline(always)]
    fn is_tighter_than(&self, other: &Self) -> bool {
        match &self.constant {
            Some(left) => match &other.constant {
                Some(right) => {
                    left < right || (left == right && self.is_strict && !other.is_strict)
                }
                None => true,
            },
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! basic_constant_test {
        ($bound_type:ty, $constant:expr) => {{
            let le = <$bound_type>::new_le($constant.clone());
            assert!(!le.is_strict());
            assert!(!le.is_unbounded());
            assert_eq!(le.constant(), Some($constant.clone()));
            let lt = <$bound_type>::new_lt($constant.clone());
            assert!(lt.is_strict());
            assert!(!lt.is_unbounded());
            assert_eq!(lt.constant(), Some($constant.clone()));
            assert!(lt.is_tighter_than(&le));
            assert!(!le.is_tighter_than(&lt));
        }};
    }

    macro_rules! bound_test {
        ($bound_type:ty, $test_name:ident, $constants:expr) => {
            #[test]
            fn $test_name() {
                let le_zero = <$bound_type>::le_zero();
                assert!(!le_zero.is_strict());
                assert_eq!(
                    le_zero.constant(),
                    Some(<$bound_type as Bound>::Constant::zero())
                );

                let lt_zero = <$bound_type>::lt_zero();
                assert!(lt_zero.is_strict());
                assert_eq!(
                    lt_zero.constant(),
                    Some(<$bound_type as Bound>::Constant::zero())
                );

                let unbounded = <$bound_type>::unbounded();
                assert!(unbounded.is_strict());
                assert_eq!(unbounded.constant(), None);

                assert!(le_zero.is_tighter_than(&unbounded));
                assert!(lt_zero.is_tighter_than(&unbounded));
                assert!(lt_zero.is_tighter_than(&le_zero));
                assert!(!le_zero.is_tighter_than(&lt_zero));
                assert!(!unbounded.is_tighter_than(&le_zero));
                assert!(!unbounded.is_tighter_than(&lt_zero));

                if let Some(constant) = <$bound_type as Bound>::min_constant() {
                    basic_constant_test!($bound_type, constant);
                }
                if let Some(constant) = <$bound_type as Bound>::max_constant() {
                    basic_constant_test!($bound_type, constant);
                }

                let mut previous: Option<<$bound_type as Bound>::Constant> = None;
                for constant in $constants.iter() {
                    basic_constant_test!($bound_type, constant);
                    if let Some(other) = previous {
                        let other_le = <$bound_type as Bound>::new_le(other.clone());
                        let this_le = <$bound_type as Bound>::new_le(constant.clone());
                        assert!(other_le.is_tighter_than(&this_le));
                    }
                    previous = Some(constant.clone());
                }
            }
        };
    }

    bound_test!(i8, test_bound_i8_native, [-13, 42]);
    bound_test!(i16, test_bound_i16_native, [-13, 42]);
    bound_test!(i32, test_bound_i32_native, [-13, 42]);
    bound_test!(i64, test_bound_i64_native, [-13, 42]);
    bound_test!(i128, test_bound_i128_native, [-13, 42]);

    bound_test!(ConstantBound<i8>, test_bound_i8_wrapped, [-13, 42]);
    bound_test!(ConstantBound<i16>, test_bound_i16_wrapped, [-13, 42]);
    bound_test!(ConstantBound<i32>, test_bound_i32_wrapped, [-13, 42]);
    bound_test!(ConstantBound<i64>, test_bound_i64_wrapped, [-13, 42]);
    bound_test!(ConstantBound<i128>, test_bound_i128_wrapped, [-13, 42]);

    #[cfg(feature = "float")]
    mod float {
        use ordered_float::NotNan;

        use super::*;

        bound_test!(
            ConstantBound<NotNan<f32>>,
            test_bound_f32,
            [
                ordered_float::NotNan::new(-13.0).unwrap(),
                ordered_float::NotNan::new(42.0).unwrap()
            ]
        );
        bound_test!(
            ConstantBound<NotNan<f64>>,
            test_bound_f64,
            [
                ordered_float::NotNan::new(-13.0).unwrap(),
                ordered_float::NotNan::new(42.0).unwrap()
            ]
        );
    }
}
