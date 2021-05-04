use std::convert::TryFrom;

/// Represents either the constant *zero clock* or a *clock variable*.
#[derive(Clone, Eq, PartialEq, Hash, Copy, Debug)]
#[repr(transparent)]
pub struct Clock(pub(crate) usize);

impl Clock {
    /// The constant zero clock.
    pub const ZERO: Clock = Clock(0);

    /// Constructs a [Variable] with the given number.
    pub fn variable(number: usize) -> Variable {
        Variable(number + 1)
    }

    /// Creates a clock from the given index.
    #[inline(always)]
    pub(crate) fn from_index(index: usize) -> Self {
        Clock(index)
    }
}

/// Represents a *clock variable*.
#[derive(Clone, Eq, PartialEq, Hash, Copy, Debug)]
#[repr(transparent)]
pub struct Variable(pub(crate) usize);

impl Variable {
    /// Retrieves the number of the variable.
    pub fn number(self) -> usize {
        self.0 - 1
    }
}

impl From<Variable> for Clock {
    fn from(variable: Variable) -> Self {
        Clock(variable.0)
    }
}

impl TryFrom<Clock> for Variable {
    type Error = ();

    fn try_from(value: Clock) -> Result<Self, Self::Error> {
        if value.0 != 0 {
            Ok(Variable(value.0))
        } else {
            Err(())
        }
    }
}

pub trait ClockToIndex {
    /// Converts the clock into an index.
    fn into_index(self) -> usize;
}

impl ClockToIndex for Clock {
    #[inline(always)]
    fn into_index(self) -> usize {
        self.0
    }
}

impl ClockToIndex for Variable {
    #[inline(always)]
    fn into_index(self) -> usize {
        self.0
    }
}

/// Either a [Clock] or a [Variable].
pub trait AnyClock: Copy + ClockToIndex {
    /// Converts the [AnyClock] into a [Clock].
    #[inline(always)]
    fn as_clock(&self) -> Clock {
        Clock(self.into_index())
    }

    /// Checks whether the clock is the constant zero clock.
    fn is_zero(&self) -> bool {
        self.into_index() == 0
    }

    /// Checks whether the clock is a clock variable.
    fn is_variable(&self) -> bool {
        self.into_index() != 0
    }
}

impl AnyClock for Clock {}

impl AnyClock for Variable {}
