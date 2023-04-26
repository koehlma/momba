//! A continuous chunk of memory storing words.

use std::marker::PhantomData;

use crate::values::{
    layout::{Addr, ValueLayout},
    types::IntTy,
    units::NumWords,
    Word,
};

use super::{Load, Region, Store};

/// A continuous chunk of memory storing words.
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct WordSlice<L> {
    _phantom_layout: PhantomData<fn(&L)>,
    slice: [Word],
}

impl<L> WordSlice<L> {
    /// Constructs a word store from the given word slice.
    #[inline]
    pub fn from_slice(slice: &[Word]) -> &Self {
        // SAFETY: `WordStore` and `[Word]` have the same memory layout.
        unsafe { &*(slice as *const _ as *const Self) }
    }

    /// Constructs a mutable word store from the given mutable word slice.
    #[inline]
    pub fn from_slice_mut(slice: &mut [Word]) -> &mut Self {
        // SAFETY: `WordStore` and `[Word]` have the same memory layout.
        unsafe { &mut *(slice as *mut _ as *mut Self) }
    }
}

impl<L> Region for WordSlice<L>
where
    L: ValueLayout<MemUnit = NumWords>,
{
    type ValueLayout = L;
}

impl<L> Load for WordSlice<L>
where
    L: ValueLayout<MemUnit = NumWords>,
{
    #[inline(always)]
    fn load_bool(&self, addr: Addr<Self::ValueLayout>) -> Word {
        self.slice[usize::from(addr)]
    }

    #[inline(always)]
    fn load_signed_int(&self, addr: Addr<Self::ValueLayout>, _: &IntTy) -> Word {
        self.slice[usize::from(addr)]
    }

    #[inline(always)]
    fn load_unsigned_int(&self, addr: Addr<Self::ValueLayout>, _: &IntTy) -> Word {
        self.slice[usize::from(addr)]
    }

    #[inline(always)]
    fn load_float32(&self, addr: Addr<Self::ValueLayout>) -> Word {
        self.slice[usize::from(addr)]
    }

    #[inline(always)]
    fn load_float64(&self, addr: Addr<Self::ValueLayout>) -> Word {
        self.slice[usize::from(addr)]
    }
}

impl<L> Store for WordSlice<L>
where
    L: ValueLayout<MemUnit = NumWords>,
{
    #[inline(always)]
    fn store_bool(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        self.slice[usize::from(addr)] = value;
    }

    #[inline(always)]
    fn store_signed_int(&mut self, addr: Addr<Self::ValueLayout>, _: &IntTy, value: Word) {
        self.slice[usize::from(addr)] = value;
    }

    #[inline(always)]
    fn store_unsigned_int(&mut self, addr: Addr<Self::ValueLayout>, _: &IntTy, value: Word) {
        self.slice[usize::from(addr)] = value;
    }

    #[inline(always)]
    fn store_float32(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        self.slice[usize::from(addr)] = value;
    }

    #[inline(always)]
    fn store_float64(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        self.slice[usize::from(addr)] = value;
    }
}
