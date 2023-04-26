//! Types for dealing with memory-related units such as bits, bytes, and words.

use std::fmt::Debug;
use std::hash::Hash;

use std::ops::Range;

/// A memory unit like [`NumBits`], [`NumBytes`], or [`NumWords`].
pub trait MemUnit:
    'static
    + Debug
    + Copy
    + Eq
    + Ord
    + Hash
    + std::ops::Add<Self, Output = Self>
    + std::ops::Mul<usize, Output = Self>
    + From<usize>
    + Into<usize>
{
    /// Offsets the value by the given offset.
    fn offset_by(self, offset: Self) -> Self;
    /// Aligns the value to the given alignment.
    fn align_to(self, align: Self) -> Self;
}

/// Auxiliary macro for defining [`NumBits`], [`NumBytes`], and [`NumWords`].
macro_rules! define_mem_unit {
    (
        $(#[doc = $doc:tt])*
        $vis:vis struct $name:ident
    ) => {
        $(#[doc = $doc])*
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(usize);

        impl MemUnit for $name {
            fn offset_by(self, offset: Self) -> Self {
                self + offset
            }

            fn align_to(self, align: Self) -> Self {
                Self(align.0 * ((self.0 + align.0 - 1) / align.0))
            }
        }

        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }

        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl std::ops::Add for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl std::ops::AddAssign for $name {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl std::ops::SubAssign for $name {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl std::ops::Mul<usize> for $name {
            type Output = Self;

            fn mul(self, rhs: usize) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl std::ops::Mul<$name> for usize {
            type Output = $name;

            fn mul(self, rhs: $name) -> Self::Output {
                rhs * self
            }
        }
    };
}

define_mem_unit! {
    /// A number of bits.
    pub struct NumBits
}

impl NumBits {
    /// Converts the number of bits to a number of bytes (rounding up).
    pub fn to_bytes(self) -> NumBytes {
        NumBytes((self.0 + 7) / 8)
    }

    /// Converts the number of bits to a number of words (rounding up).
    pub fn to_words(self) -> NumWords {
        NumWords((self.0 + 63) / 64)
    }
}

impl From<NumBytes> for NumBits {
    #[inline]
    fn from(value: NumBytes) -> Self {
        NumBits(value.0 * 8)
    }
}

define_mem_unit! {
    /// A number of bytes.
    pub struct NumBytes
}

define_mem_unit! {
    /// A number of words.
    pub struct NumWords
}

pub trait RangeExt {
    type Size: MemUnit;

    fn offset_by(&self, offset: Self::Size) -> Self;
}

impl<S: MemUnit> RangeExt for Range<S> {
    type Size = S;

    fn offset_by(&self, offset: Self::Size) -> Self {
        self.start.offset_by(offset)..self.end.offset_by(offset)
    }
}

/// A range of bits.
pub type BitRange = Range<NumBits>;

/// A range of bytes.
pub type ByteRange = Range<NumBytes>;

/// A range of words.
pub type WordRange = Range<NumWords>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align() {
        assert_eq!(NumBits::from(1).align_to(8.into()), NumBits::from(8));
        assert_eq!(NumBits::from(3).align_to(16.into()), NumBits::from(16));
        assert_eq!(NumBits::from(16).align_to(16.into()), NumBits::from(16));
        assert_eq!(NumBits::from(17).align_to(16.into()), NumBits::from(32));
    }
}
