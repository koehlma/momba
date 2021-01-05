pub trait Constant: Clone + PartialEq + PartialOrd {
    fn max_value() -> Option<Self>;
    fn min_value() -> Option<Self>;

    fn zero() -> Self;

    fn checked_add(&self, other: &Self) -> Option<Self>;
    fn checked_neg(&self) -> Option<Self>;
}

macro_rules! int_constant_impl {
    ($int_type:ty) => {
        impl Constant for $int_type {
            #[inline(always)]
            fn max_value() -> Option<Self> {
                Some(<$int_type>::MAX)
            }
            #[inline(always)]
            fn min_value() -> Option<Self> {
                Some(<$int_type>::MIN)
            }

            #[inline(always)]
            fn zero() -> Self {
                0
            }

            #[inline(always)]
            fn checked_add(&self, other: &Self) -> Option<Self> {
                <$int_type>::checked_add(*self, *other)
            }
            #[inline(always)]
            fn checked_neg(&self) -> Option<Self> {
                <$int_type>::checked_neg(*self)
            }
        }
    };
}

int_constant_impl!(i8);
int_constant_impl!(i16);
int_constant_impl!(i32);
int_constant_impl!(i64);
int_constant_impl!(i128);

macro_rules! float_constant_impl {
    ($float_type:ty) => {
        impl Constant for ordered_float::NotNan<$float_type> {
            #[inline(always)]
            fn max_value() -> Option<Self> {
                Some(ordered_float::NotNan::new(<$float_type>::MAX).unwrap())
            }
            #[inline(always)]
            fn min_value() -> Option<Self> {
                Some(ordered_float::NotNan::new(<$float_type>::MIN).unwrap())
            }

            #[inline(always)]
            fn zero() -> Self {
                ordered_float::NotNan::new(0.0).unwrap()
            }

            #[inline(always)]
            fn checked_add(&self, other: &Self) -> Option<Self> {
                let result = *self + *other;
                if result.is_infinite() {
                    None
                } else {
                    Some(result)
                }
            }
            #[inline(always)]
            fn checked_neg(&self) -> Option<Self> {
                Some(-*self)
            }
        }
    };
}

float_constant_impl!(f32);
float_constant_impl!(f64);

#[cfg(feature = "bigint")]
pub mod bigint {
    use std::borrow::Borrow;
    use std::rc::Rc;

    use num_bigint::BigInt;
    use num_traits::Zero;

    use super::*;

    impl Constant for Rc<BigInt> {
        #[inline(always)]
        fn max_value() -> Option<Self> {
            None
        }
        #[inline(always)]
        fn min_value() -> Option<Self> {
            None
        }

        #[inline(always)]
        fn zero() -> Self {
            Rc::new(BigInt::zero())
        }

        #[inline(always)]
        fn checked_add(&self, other: &Self) -> Option<Self> {
            let left: &BigInt = self.borrow();
            let right: &BigInt = other.borrow();
            Some(Rc::new(left + right))
        }
        #[inline(always)]
        fn checked_neg(&self) -> Option<Self> {
            let value: &BigInt = self.borrow();
            Some(Rc::new(-value))
        }
    }
}

#[cfg(feature = "rational")]
pub mod rational {
    use std::borrow::Borrow;
    use std::rc::Rc;

    use num_rational::BigRational;
    use num_traits::Zero;

    use super::*;

    impl Constant for Rc<BigRational> {
        #[inline(always)]
        fn max_value() -> Option<Self> {
            None
        }
        #[inline(always)]
        fn min_value() -> Option<Self> {
            None
        }

        #[inline(always)]
        fn zero() -> Self {
            Rc::new(BigRational::zero())
        }

        #[inline(always)]
        fn checked_add(&self, other: &Self) -> Option<Self> {
            let left: &BigRational = self.borrow();
            let right: &BigRational = other.borrow();
            Some(Rc::new(left + right))
        }
        #[inline(always)]
        fn checked_neg(&self) -> Option<Self> {
            let value: &BigRational = self.borrow();
            Some(Rc::new(-value))
        }
    }
}
