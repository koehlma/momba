//! A continuous chunk of memory for storing bits.

use std::marker::PhantomData;

use bitvec::field::BitField;

use crate::values::{
    layout::{Addr, Layout, ValueLayout},
    types::IntTy,
    units::{MemUnit, NumBits},
    FromWord, IntoWord, Word,
};

use super::{Allocate, Load, Region, Store};

/// A continuous chunk of memory for storing bits.
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BitSlice<L> {
    _phantom_layout: PhantomData<fn(&L)>,
    slice: bitvec::slice::BitSlice<u8>,
}

impl<L> BitSlice<L> {
    /// Constructs a bit store from the given slice.
    #[inline]
    pub fn from_slice(slice: &[u8]) -> &Self {
        let slice = bitvec::slice::BitSlice::<u8>::from_slice(slice);
        // SAFETY: `BitStore<L, B>` and `BitSlice<B>` have the same memory layout.
        unsafe { &*(slice as *const _ as *const Self) }
    }

    /// Constructs a mutable bit store from the given mutable slice.
    #[inline]
    pub fn from_slice_mut(slice: &mut [u8]) -> &mut Self {
        let slice = bitvec::slice::BitSlice::<u8>::from_slice_mut(slice);
        // SAFETY: `BitStore<L, B>` and `BitSlice<B>` have the same memory layout.
        unsafe { &mut *(slice as *mut _ as *mut Self) }
    }
}

impl<L> Region for BitSlice<L>
where
    L: ValueLayout<MemUnit = NumBits>,
{
    type ValueLayout = L;
}

impl<L> Load for BitSlice<L>
where
    L: ValueLayout<MemUnit = NumBits>,
{
    #[inline(always)]
    fn load_bool(&self, addr: Addr<Self::ValueLayout>) -> Word {
        self.slice[usize::from(addr)].into_word()
    }

    #[inline(always)]
    fn load_signed_int(&self, addr: Addr<Self::ValueLayout>, ty: &IntTy) -> Word {
        let start = usize::from(addr);
        let end = start + usize::from(ty.bits);
        (self.slice[start..end].load::<i64>() + ty.offset).into_word()
    }

    #[inline(always)]
    fn load_unsigned_int(&self, addr: Addr<Self::ValueLayout>, ty: &IntTy) -> Word {
        let start = usize::from(addr);
        let end = start + usize::from(ty.bits);
        ((self.slice[start..end].load::<u64>() as i64) + ty.offset).into_word()
    }

    #[inline(always)]
    fn load_float32(&self, addr: Addr<Self::ValueLayout>) -> Word {
        let start = usize::from(addr);
        let end = start + 32;
        (f32::from_bits(self.slice[start..end].load::<u32>()) as f64).into_word()
    }

    #[inline(always)]
    fn load_float64(&self, addr: Addr<Self::ValueLayout>) -> Word {
        let start = usize::from(addr);
        let end = start + 64;
        Word::from_raw(self.slice[start..end].load::<u64>())
    }
}

impl<L> Store for BitSlice<L>
where
    L: ValueLayout<MemUnit = NumBits>,
{
    #[inline(always)]
    fn store_bool(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        self.slice.set(usize::from(addr), bool::from_word(value));
    }

    #[inline(always)]
    fn store_signed_int(&mut self, addr: Addr<Self::ValueLayout>, ty: &IntTy, value: Word) {
        let start = usize::from(addr);
        let end = start + usize::from(ty.bits);
        self.slice[start..end].store(value.to_raw());
    }

    #[inline(always)]
    fn store_unsigned_int(&mut self, addr: Addr<Self::ValueLayout>, ty: &IntTy, value: Word) {
        let start = usize::from(addr);
        let end = start + usize::from(ty.bits);
        self.slice[start..end].store(value.to_raw());
    }

    #[inline(always)]
    fn store_float32(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        let start = usize::from(addr);
        let end = start + 32;
        self.slice[start..end].store((f64::from_word(value) as f32).to_bits());
    }

    #[inline(always)]
    fn store_float64(&mut self, addr: Addr<Self::ValueLayout>, value: Word) {
        let start = usize::from(addr);
        let end = start + 64;
        self.slice[start..end].store(value.to_raw());
    }
}

#[derive(Debug, Clone)]
pub struct BitVec<L> {
    data: Vec<u8>,
    head: NumBits,
    _phantom_layout: PhantomData<fn(&L)>,
}

impl<L> BitVec<L> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            head: 0.into(),
            _phantom_layout: PhantomData,
        }
    }
}

impl<L> Region for BitVec<L>
where
    L: ValueLayout<MemUnit = NumBits>,
{
    type ValueLayout = L;
}

impl<L> Allocate for BitVec<L>
where
    L: ValueLayout<MemUnit = NumBits>,
{
    fn allocate(&mut self, layout: Layout<Self::ValueLayout>) -> Addr<Self::ValueLayout> {
        let addr = self.head.align_to(layout.align);
        self.head = addr.offset_by(layout.size);
        let length = (usize::from(self.head) + 7) / 8;
        if self.data.len() < length {
            self.data.reserve(length - self.data.len());
            while self.data.len() < length {
                self.data.push(0)
            }
        }
        addr
    }
}
