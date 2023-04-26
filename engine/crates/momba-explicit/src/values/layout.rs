//! Layouts describe how values are stored in memory.

use std::fmt;

use crate::datatypes::idxvec::{new_idx_type, IdxVec};

use super::{
    types::{ArrayTy, ValueTy, ValueTyKind},
    units::{MemUnit, NumBits, NumBytes, NumWords},
};

/// A layout for storing values in memory.
pub trait ValueLayout {
    /// The size type of the layout.
    type MemUnit: MemUnit;

    /// The size of the given value type.
    fn size_of(ty: &ValueTy) -> Self::MemUnit;

    /// The alignment of the given value type.
    fn align_of(ty: &ValueTy) -> Self::MemUnit;
}

/// An address of a value.
pub type Addr<L> = <L as ValueLayout>::MemUnit;

/// The size of a value.
pub type Size<L> = <L as ValueLayout>::MemUnit;

/// The alignment of a value.
pub type Align<L> = <L as ValueLayout>::MemUnit;

/// A layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout<L: ValueLayout> {
    pub size: Size<L>,
    pub align: Align<L>,
}

impl<L: ValueLayout> Layout<L> {
    pub fn of_type(ty: &ValueTy) -> Self {
        Self {
            size: L::size_of(ty),
            align: L::size_of(ty),
        }
    }
}

/// Computes the size of an array type with proper element alignment.
fn array_size<L: ValueLayout>(ty: &ArrayTy) -> L::MemUnit {
    L::size_of(&ty.element_ty).align_to(L::align_of(&ty.element_ty)) * ty.length
}

/// Computes the alignment of an array type.
fn array_align<L: ValueLayout>(ty: &ArrayTy) -> L::MemUnit {
    L::align_of(&ty.element_ty)
}

/// An extremely space-efficient layout for storing values in a dense bit vector.
pub struct DenseBitLayout;

impl ValueLayout for DenseBitLayout {
    type MemUnit = NumBits;

    fn size_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(ty) => ty.bits,
            ValueTyKind::UnsignedInt(ty) => ty.bits,
            ValueTyKind::Float32 => 32.into(),
            ValueTyKind::Float64 => 64.into(),
            ValueTyKind::Pointer(_) => 64.into(),
            ValueTyKind::Slice(_) => 64.into(),
            ValueTyKind::Array(ty) => array_size::<Self>(ty),
            ValueTyKind::Void => 0.into(),
        }
    }

    fn align_of(_: &ValueTy) -> Self::MemUnit {
        1.into()
    }
}

/// A very space-efficient layout for storing values in a bit vector with some alignment.
pub struct AlignedBitLayout;

impl ValueLayout for AlignedBitLayout {
    type MemUnit = NumBits;

    fn size_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(ty) => ty.bits,
            ValueTyKind::UnsignedInt(ty) => ty.bits,
            ValueTyKind::Float32 => 32.into(),
            ValueTyKind::Float64 => 64.into(),
            ValueTyKind::Pointer(_) => 64.into(),
            ValueTyKind::Slice(_) => 64.into(),
            ValueTyKind::Array(ty) => array_size::<Self>(ty),
            ValueTyKind::Void => 0.into(),
        }
    }

    fn align_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(_) => 8.into(),
            ValueTyKind::UnsignedInt(_) => 8.into(),
            ValueTyKind::Float32 => 8.into(),
            ValueTyKind::Float64 => 8.into(),
            ValueTyKind::Pointer(_) => 8.into(),
            ValueTyKind::Slice(_) => 8.into(),
            ValueTyKind::Array(ty) => array_align::<Self>(ty),
            ValueTyKind::Void => 1.into(),
        }
    }
}

/// A rather space-efficient layout for storing values in a dense byte vector.
pub struct DenseByteLayout;

impl ValueLayout for DenseByteLayout {
    type MemUnit = NumBytes;

    fn size_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(ty) => ty.bits.to_bytes(),
            ValueTyKind::UnsignedInt(ty) => ty.bits.to_bytes(),
            ValueTyKind::Float32 => 4.into(),
            ValueTyKind::Float64 => 8.into(),
            ValueTyKind::Pointer(_) => 8.into(),
            ValueTyKind::Slice(_) => 8.into(),
            ValueTyKind::Array(ty) => array_size::<Self>(ty),
            ValueTyKind::Void => 0.into(),
        }
    }

    fn align_of(_: &ValueTy) -> Self::MemUnit {
        1.into()
    }
}

/// A somewhat space-efficient layout for storing values in a byte vector with some alignment.
pub struct AlignedByteLayout;

impl ValueLayout for AlignedByteLayout {
    type MemUnit = NumBytes;

    fn size_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(_) => 8.into(),
            ValueTyKind::UnsignedInt(_) => 8.into(),
            ValueTyKind::Float32 => 4.into(),
            ValueTyKind::Float64 => 8.into(),
            ValueTyKind::Pointer(_) => 8.into(),
            ValueTyKind::Slice(_) => 8.into(),
            ValueTyKind::Array(ty) => array_size::<Self>(ty),
            ValueTyKind::Void => 0.into(),
        }
    }

    fn align_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(_) => 8.into(),
            ValueTyKind::UnsignedInt(_) => 8.into(),
            ValueTyKind::Float32 => 4.into(),
            ValueTyKind::Float64 => 8.into(),
            ValueTyKind::Pointer(_) => 8.into(),
            ValueTyKind::Slice(_) => 8.into(),
            ValueTyKind::Array(ty) => array_align::<Self>(ty),
            ValueTyKind::Void => 1.into(),
        }
    }
}

/// A layout for storing values in a word vector (absolutely not space efficient).
pub struct WordLayout;

impl ValueLayout for WordLayout {
    type MemUnit = NumWords;

    fn size_of(ty: &ValueTy) -> Self::MemUnit {
        match ty.kind() {
            ValueTyKind::Bool => 1.into(),
            ValueTyKind::SignedInt(_) => 1.into(),
            ValueTyKind::UnsignedInt(_) => 1.into(),
            ValueTyKind::Float32 => 1.into(),
            ValueTyKind::Float64 => 1.into(),
            ValueTyKind::Pointer(_) => 1.into(),
            ValueTyKind::Slice(_) => 1.into(),
            ValueTyKind::Array(ty) => array_size::<Self>(ty),
            ValueTyKind::Void => 0.into(),
        }
    }

    fn align_of(_: &ValueTy) -> Self::MemUnit {
        1.into()
    }
}

/// A _field_ of a [`StructLayout`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    /// The value type of the field.
    ty: ValueTy,
    /// An optional name of the field (mostly useful for debugging).
    name: Option<String>,
}

impl Field {
    /// Creates a new field of the given type.
    pub fn new(ty: ValueTy) -> Self {
        Self { ty, name: None }
    }

    /// Sets the name of the field.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The type of the field.
    pub fn ty(&self) -> &ValueTy {
        &self.ty
    }

    /// The optional name of the field.
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(String::as_str)
    }
}

new_idx_type! {
    /// Uniquely identifies a field of a [`StructLayout`].
    pub FieldIdx(usize)
}

/// A layout of a structure with fields.
#[derive(Debug, Clone)]
pub struct StructLayout {
    pub(crate) fields: IdxVec<FieldIdx, Field>,
}

impl StructLayout {
    /// Creates a new structure layout.
    pub fn new() -> Self {
        Self {
            fields: IdxVec::new(),
        }
    }

    /// Adds a field to the layout and returns its index.
    pub fn add_field(&mut self, field: Field) -> FieldIdx {
        let idx = FieldIdx::from(self.fields.len());
        self.fields.push(field);
        idx
    }

    /// Computes the offsets of the fields given a value layout.
    pub fn field_offsets<L: ValueLayout>(&self) -> FieldOffsets<L> {
        let mut offsets = IdxVec::new();
        let mut size = L::MemUnit::from(0);
        for field in self.fields.iter() {
            size = size.align_to(L::align_of(&field.ty));
            offsets.push(size);
            size = size.offset_by(L::size_of(&field.ty));
        }
        FieldOffsets { offsets }
    }

    /// Computes the size of the structure given a value layout.
    pub fn size<L: ValueLayout>(&self) -> L::MemUnit {
        let mut size = L::MemUnit::from(0);
        for field in self.fields.iter() {
            size = size
                .align_to(L::align_of(&field.ty))
                .offset_by(L::size_of(&field.ty));
        }
        size
    }

    /// Computes the alignment of the structure given a value layout.
    pub fn align<L: ValueLayout>(&self) -> L::MemUnit {
        let mut align = L::MemUnit::from(1);
        for field in self.fields.iter() {
            align = align.align_to(L::align_of(&field.ty))
        }
        align
    }
}

impl std::ops::Index<FieldIdx> for StructLayout {
    type Output = Field;

    fn index(&self, index: FieldIdx) -> &Self::Output {
        &self.fields[index]
    }
}
impl fmt::Display for StructLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{ ")?;
        for (idx, field) in self.fields.iter().enumerate() {
            if idx > 0 {
                f.write_str(", ")?;
            }
            f.write_fmt(format_args!(
                "{}: {}",
                field
                    .name
                    .as_ref()
                    .map(String::as_str)
                    .unwrap_or("<unnamed>"),
                field.ty()
            ))?;
        }
        f.write_str(" }")
    }
}

/// Computed offsets of the fields of a [`StructLayout`].
#[derive(Debug, Clone)]
pub struct FieldOffsets<L: ValueLayout> {
    offsets: IdxVec<FieldIdx, Addr<L>>,
}

impl<L: ValueLayout> std::ops::Index<FieldIdx> for FieldOffsets<L> {
    type Output = Addr<L>;

    fn index(&self, index: FieldIdx) -> &Self::Output {
        &self.offsets[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::values::types::{IntTy, TypeCtx};

    use super::*;

    #[test]
    fn test_layouts() {
        let tcx = TypeCtx::new();

        let int12 = tcx.value_unsigned_int(IntTy::new(12.into()));

        assert_eq!(DenseBitLayout::size_of(&int12), 12.into());
        assert_eq!(AlignedBitLayout::size_of(&int12), 12.into());
        assert_eq!(WordLayout::size_of(&int12), 1.into());

        assert_eq!(DenseBitLayout::align_of(&int12), 1.into());
        assert_eq!(AlignedBitLayout::align_of(&int12), 8.into());
        assert_eq!(WordLayout::align_of(&int12), 1.into());

        let int12_array8 = tcx.value_array(ArrayTy::new(int12.clone(), 9));

        assert_eq!(DenseBitLayout::size_of(&int12_array8), (12 * 9).into());
        assert_eq!(AlignedBitLayout::size_of(&int12_array8), (16 * 9).into());
        assert_eq!(WordLayout::size_of(&int12_array8), 9.into());

        assert_eq!(DenseBitLayout::align_of(&int12_array8), 1.into());
        assert_eq!(AlignedBitLayout::align_of(&int12_array8), 8.into());
        assert_eq!(WordLayout::align_of(&int12_array8), 1.into());

        let mut struct_layout = StructLayout::new();

        let field_x = Field::new(tcx.value_bool()).with_name("x");
        let x = struct_layout.add_field(field_x);
        let field_y = Field::new(int12.clone()).with_name("y");
        let y = struct_layout.add_field(field_y);
        let field_z = Field::new(int12_array8.clone()).with_name("z");
        let z = struct_layout.add_field(field_z);

        let offsets = struct_layout.field_offsets::<DenseBitLayout>();
        assert_eq!(
            struct_layout.size::<DenseBitLayout>(),
            (1 + 12 + 12 * 9).into()
        );
        assert_eq!(offsets[x], 0.into());
        assert_eq!(offsets[y], (0 + 1).into());
        assert_eq!(offsets[z], (0 + 1 + 12).into());

        let offsets = struct_layout.field_offsets::<AlignedBitLayout>();
        assert_eq!(
            struct_layout.size::<AlignedBitLayout>(),
            (8 + 16 + 16 * 9).into()
        );
        assert_eq!(offsets[x], 0.into());
        assert_eq!(offsets[y], (0 + 8).into());
        assert_eq!(offsets[z], (0 + 8 + 16).into());

        let offsets = struct_layout.field_offsets::<WordLayout>();
        assert_eq!(struct_layout.size::<WordLayout>(), (1 + 1 + 9).into());
        assert_eq!(offsets[x], 0.into());
        assert_eq!(offsets[y], 1.into());
        assert_eq!(offsets[z], 2.into());
    }
}
