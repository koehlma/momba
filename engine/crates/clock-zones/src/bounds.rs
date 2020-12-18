use crate::constants::*;

pub trait Bound: Clone {
    type Constant: Constant;

    fn max_constant() -> Option<Self::Constant>;
    fn min_constant() -> Option<Self::Constant>;

    fn unbounded() -> Self;

    fn le_zero() -> Self;
    fn lt_zero() -> Self;

    fn new_le(constant: Self::Constant) -> Self;
    fn new_lt(constant: Self::Constant) -> Self;

    fn is_strict(&self) -> bool;
    fn is_unbounded(&self) -> bool;

    fn get_constant(&self) -> Option<Self::Constant>;

    fn add(&self, other: &Self) -> Option<Self>;

    fn is_tighter_than(&self, other: &Self) -> bool;
}

macro_rules! int_bound_impl {
    ($int_type:ty) => {
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
            fn get_constant(&self) -> Option<Self::Constant> {
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

#[derive(Clone, PartialEq, Debug)]
pub struct ConstantBound<C: Constant> {
    constant: Option<C>,
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
    fn get_constant(&self) -> Option<Self::Constant> {
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
            assert_eq!(le.get_constant(), Some($constant.clone()));
            let lt = <$bound_type>::new_lt($constant.clone());
            assert!(lt.is_strict());
            assert!(!lt.is_unbounded());
            assert_eq!(lt.get_constant(), Some($constant.clone()));
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
                    le_zero.get_constant(),
                    Some(<$bound_type as Bound>::Constant::zero())
                );

                let lt_zero = <$bound_type>::lt_zero();
                assert!(lt_zero.is_strict());
                assert_eq!(
                    lt_zero.get_constant(),
                    Some(<$bound_type as Bound>::Constant::zero())
                );

                let unbounded = <$bound_type>::unbounded();
                assert!(unbounded.is_strict());
                assert_eq!(unbounded.get_constant(), None);

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

    bound_test!(ConstantBound<f32>, test_bound_f32, [-13.0, 42.0]);
    bound_test!(ConstantBound<f64>, test_bound_f64, [-13.0, 42.0]);

    #[cfg(feature = "bigint")]
    mod bigint {
        use std::rc::Rc;

        use num_bigint::BigInt;

        use super::*;

        bound_test!(
            ConstantBound::<Rc<BigInt>>,
            test_bound_bigint,
            [Rc::new(BigInt::from(-13)), Rc::new(BigInt::from(42))]
        );
    }

    #[cfg(feature = "rational")]
    mod rational {
        use std::rc::Rc;
        use std::str::FromStr;

        use num_rational::BigRational;

        use super::*;

        bound_test!(
            ConstantBound::<Rc<BigRational>>,
            test_bound_bigint,
            [
                Rc::new(BigRational::from_str("-13/2").unwrap()),
                Rc::new(BigRational::from_str("1337/3").unwrap())
            ]
        );
    }
}
