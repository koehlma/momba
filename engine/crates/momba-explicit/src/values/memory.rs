use super::{
    layout::{Addr, Layout, ValueLayout},
    types::{IntTy, ValueTy, ValueTyKind},
    Word,
};

pub mod bits;
pub mod words;

pub trait Region {
    type ValueLayout: ValueLayout;
}

pub struct StateField<M: Memory> {
    pub ty: ValueTy,
    pub addr: Addr<<M::StateRegion as Region>::ValueLayout>,
}

pub trait Memory {
    type StateRegion: Region + Load;
}

// impl<M: Memory> Memory for &M {
//     type ValueLayout = M::ValueLayout;
// }

pub trait Load: Region {
    fn load_bool(&self, addr: Addr<Self::ValueLayout>) -> Word;

    fn load_signed_int(&self, addr: Addr<Self::ValueLayout>, ty: &IntTy) -> Word;
    fn load_unsigned_int(&self, addr: Addr<Self::ValueLayout>, ty: &IntTy) -> Word;

    fn load_float32(&self, addr: Addr<Self::ValueLayout>) -> Word;
    fn load_float64(&self, addr: Addr<Self::ValueLayout>) -> Word;
}

// impl<L, T: Load<L>> Load<L> for &T {
//     fn load_bool(&self, addr: Addr<L>) -> Word {
//         (**self).load_bool(addr)
//     }

//     fn load_signed_int(&self, addr: Addr<L>, ty: &IntTy) -> Word {
//         (**self).load_signed_int(addr, ty)
//     }

//     fn load_unsigned_int(&self, addr: Addr<L>, ty: &IntTy) -> Word {
//         (**self).load_unsigned_int(addr, ty)
//     }

//     fn load_float32(&self, addr: Addr<Self::ValueLayout>) -> Word {
//         (**self).load_float32(addr)
//     }

//     fn load_float64(&self, addr: Addr<Self::ValueLayout>) -> Word {
//         (**self).load_float64(addr)
//     }
// }

pub trait Store: Region {
    fn store_bool(&mut self, addr: Addr<Self::ValueLayout>, value: Word);

    fn store_signed_int(&mut self, addr: Addr<Self::ValueLayout>, ty: &IntTy, value: Word);
    fn store_unsigned_int(&mut self, addr: Addr<Self::ValueLayout>, ty: &IntTy, value: Word);

    fn store_float32(&mut self, addr: Addr<Self::ValueLayout>, value: Word);
    fn store_float64(&mut self, addr: Addr<Self::ValueLayout>, value: Word);

    fn store(&mut self, addr: Addr<Self::ValueLayout>, ty: &ValueTy, value: Word) {
        match ty.kind() {
            ValueTyKind::Bool => self.store_bool(addr, value),
            ValueTyKind::SignedInt(ty) => self.store_signed_int(addr, ty, value),
            ValueTyKind::UnsignedInt(ty) => self.store_unsigned_int(addr, ty, value),
            ValueTyKind::Float32 => self.store_float32(addr, value),
            ValueTyKind::Float64 => self.store_float64(addr, value),
            _ => todo!(),
        }
    }
}

pub trait Allocate: Region {
    fn allocate(&mut self, layout: Layout<Self::ValueLayout>) -> Addr<Self::ValueLayout>;
}
