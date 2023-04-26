//! Infrastructure for working with _values_.

use momba_model::values::Value;

pub mod layout;
pub mod memory;
pub mod types;
pub mod units;

/// An untyped 64-bit _value_.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Word {
    /// The underlying raw [`u64`].
    raw: u64,
}

impl Word {
    /// Constructs a [`Word`] from its raw [`u64`] representation.
    #[inline]
    pub fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Returns the underlying raw [`u64`] representation.
    #[inline]
    pub fn to_raw(self) -> u64 {
        self.raw
    }

    #[inline]
    pub fn coerce_int_to_float(self) -> Self {
        (i64::from_word(self) as f64).into_word()
    }
}

/// Trait for converting values into [`Word`].
pub trait IntoWord {
    /// Converts `self` into `Word`.
    fn into_word(self) -> Word;
}

/// Trait for converting values from [`Word`].
pub trait FromWord {
    fn from_word(word: Word) -> Self;
}

impl IntoWord for Word {
    fn into_word(self) -> Word {
        self
    }
}

impl IntoWord for Value {
    fn into_word(self) -> Word {
        match self {
            Value::Int(value) => value.into_word(),
            Value::Float(value) => value.into_word(),
            Value::Bool(value) => value.into_word(),
        }
    }
}

impl IntoWord for i64 {
    fn into_word(self) -> Word {
        Word::from_raw(self as u64)
    }
}

impl IntoWord for f64 {
    fn into_word(self) -> Word {
        Word::from_raw(self.to_bits())
    }
}

impl IntoWord for bool {
    fn into_word(self) -> Word {
        Word::from_raw(self as u64)
    }
}

impl From<Word> for i64 {
    #[inline]
    fn from(value: Word) -> Self {
        value.to_raw() as i64
    }
}

impl FromWord for i64 {
    fn from_word(word: Word) -> Self {
        word.to_raw() as i64
    }
}

impl FromWord for f64 {
    fn from_word(word: Word) -> Self {
        f64::from_bits(word.to_raw())
    }
}

impl FromWord for bool {
    fn from_word(word: Word) -> Self {
        word.to_raw() != 0
    }
}

impl FromWord for usize {
    fn from_word(word: Word) -> Self {
        (word.to_raw() as i64) as usize
    }
}

/// The four _memory regions_ where values can be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemRegion {
    /// The _static region_ for static values computed during compile time.
    Static = 0b00,
    /// The _state region_ for state-specific values stored efficiently in a bit vector.
    State = 0b01,
    /// The _scratch region_ for temporary values constructed during evaluation.
    Scratch = 0b10,
    /// The _stack region_ for temporary values constructed on the evaluation stack.
    Stack = 0b11,
}

impl MemRegion {
    /// Turns a numeric region tag into the respective [`MemRegion`].
    ///
    /// # Panics
    ///
    /// Panics in case the tag is invalid.
    #[inline]
    pub(crate) fn from_tag(tag: u8) -> Self {
        match tag {
            0b00 => Self::Static,
            0b01 => Self::State,
            0b10 => Self::Scratch,
            0b11 => Self::Stack,
            _ => panic!("Invalid `MemArea` tag `{tag}`."),
        }
    }

    /// Returns the numeric tag of the [`MemRegion`].
    #[inline]
    pub(crate) fn to_tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            MemRegion::Static => "static",
            MemRegion::State => "state",
            MemRegion::Scratch => "scratch",
            MemRegion::Stack => "stack",
        }
    }
}

/// A pointer to a memory location of the VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pointer(Word);

impl Pointer {
    const OFFSET_BITS: u64 = 30;
    const AREA_BITS: u64 = 2;

    const TAG_SHIFT: u64 = Self::OFFSET_BITS + Self::AREA_BITS;
    const AREA_SHIFT: u64 = Self::OFFSET_BITS;

    const MAX_AREA: u64 = (1 << Self::AREA_BITS) - 1;

    pub(crate) const MAX_OFFSET: u64 = (1 << Self::OFFSET_BITS) - 1;
    pub(crate) const MAX_TAG: u32 = u32::MAX;

    #[inline]
    pub fn new(area: MemRegion, offset: usize, tag: u32) -> Self {
        debug_assert!(
            offset as u64 <= Self::MAX_OFFSET,
            "Offset must not exceed 30 bits."
        );
        debug_assert!(
            area.to_tag() as u64 <= Self::MAX_AREA,
            "Area tag must not exceed 2 bits."
        );
        Self(Word::from_raw(
            ((tag as u64) << Self::TAG_SHIFT)
                | ((area.to_tag() as u64) << Self::AREA_SHIFT)
                | (offset as u64),
        ))
    }

    #[inline]
    pub fn invalid() -> Self {
        Self::new(MemRegion::Static, Self::MAX_OFFSET as usize, Self::MAX_TAG)
    }

    #[inline]
    pub fn area(&self) -> MemRegion {
        MemRegion::from_tag(((self.0.raw >> Self::OFFSET_BITS) & Self::MAX_AREA) as u8)
    }

    #[inline]
    pub fn addr(&self) -> usize {
        (self.0.raw & Self::MAX_OFFSET) as usize
    }

    #[inline]
    pub fn tag(&self) -> u32 {
        (self.0.to_raw() >> Self::TAG_SHIFT) as u32
    }
}

impl From<Pointer> for Word {
    #[inline]
    fn from(value: Pointer) -> Self {
        value.0
    }
}

impl From<Word> for Pointer {
    #[inline]
    fn from(value: Word) -> Self {
        Pointer(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slice(Pointer);

impl Slice {
    pub fn pointer(&self) -> &Pointer {
        &self.0
    }
}

impl From<Slice> for Word {
    #[inline]
    fn from(value: Slice) -> Self {
        value.0.into()
    }
}

impl From<Word> for Slice {
    #[inline]
    fn from(value: Word) -> Self {
        Slice(Pointer(value))
    }
}
