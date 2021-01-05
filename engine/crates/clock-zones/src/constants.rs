/// Represents a constant.
pub trait Constant: Clone + PartialEq + PartialOrd {
    /// The maximal value by which the difference can be bounded.
    fn max_value() -> Option<Self>;
    /// The minimal value by which the difference can be bounded.
    fn min_value() -> Option<Self>;

    /// The constant being equivalent to `0`.
    fn zero() -> Self;

    /// Overflow checked addition of two constants.
    fn checked_add(&self, other: &Self) -> Option<Self>;
    /// Overflow checked negation of a constant.
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

#[cfg(feature = "float")]
mod float {
    use ordered_float::NotNan;

    use super::*;

    macro_rules! float_constant_impl {
        ($float_type:ty) => {
            impl Constant for NotNan<$float_type> {
                #[inline(always)]
                fn max_value() -> Option<Self> {
                    Some(NotNan::new(<$float_type>::MAX).unwrap())
                }
                #[inline(always)]
                fn min_value() -> Option<Self> {
                    Some(NotNan::new(<$float_type>::MIN).unwrap())
                }

                #[inline(always)]
                fn zero() -> Self {
                    NotNan::new(0.0).unwrap()
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
}

#[cfg(feature = "float")]
pub use float::*;
