//! Implementation of the type system.

use std::{fmt, hash::Hash, sync::Arc};

use hashbrown::HashSet;
use momba_model::{types::Type, values::Value};
use parking_lot::RwLock;
use thiserror::Error;

use super::{
    units::{NumBits, NumWords},
    MemRegion,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WordTyKind {
    /// The boolean type.
    Bool,
    /// The signed integer type.
    Int,
    /// The floating-point number type.
    Float,
    /// The zero-sized type indicating the absence of a meaningful value.
    Void,
    /// A pointer to a value of the given type.
    Pointer(PointerTy),
    /// A slice of elements of the given type.
    Slice(SliceTy),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerTy {
    value_ty: ValueTy,
    region: Option<MemRegion>,
}

impl fmt::Display for PointerTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("&")?;
        if let Some(region) = &self.region {
            f.write_fmt(format_args!("'{}", region.name()))?;
        }
        self.value_ty.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SliceTy {
    element_ty: ValueTy,
    region: Option<MemRegion>,
    length: Option<usize>,
}

impl fmt::Display for SliceTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("&")?;
        if let Some(region) = &self.region {
            f.write_fmt(format_args!("'{}", region.name()))?;
        }
        f.write_str("[")?;
        self.element_ty.fmt(f)?;
        if let Some(length) = self.length {
            f.write_fmt(format_args!("; {}", length))?;
        }
        f.write_str("]")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntTy {
    pub(crate) bits: NumBits,
    pub(crate) offset: i64,
}

impl IntTy {
    pub fn new(bits: NumBits) -> Self {
        Self { bits, offset: 0 }
    }

    pub fn with_offset(mut self, offset: i64) -> Self {
        self.offset = offset;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayTy {
    pub(crate) element_ty: ValueTy,
    pub(crate) length: usize,
}

impl ArrayTy {
    pub fn new(element_ty: ValueTy, length: usize) -> Self {
        Self { element_ty, length }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WordTy(Arc<WordTyKind>);

impl WordTy {
    pub fn kind(&self) -> &WordTyKind {
        &*self.0
    }

    /// Checks whether the type is the boolean type.
    pub fn is_bool(&self) -> bool {
        matches!(self.kind(), WordTyKind::Bool)
    }

    /// Checks whether the type is the integer type.
    pub fn is_int(&self) -> bool {
        matches!(self.kind(), WordTyKind::Int)
    }
}

impl fmt::Display for WordTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            WordTyKind::Bool => f.write_str("bool"),
            WordTyKind::Int => f.write_str("int"),
            WordTyKind::Float => f.write_str("float"),
            WordTyKind::Void => f.write_str("bool"),
            WordTyKind::Pointer(_) => todo!(),
            WordTyKind::Slice(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueTyKind {
    /// The boolean type.
    Bool,
    /// A signed integer type with a specific bit width and offset.
    SignedInt(IntTy),
    /// An unsigned integer type with a specific bit width and offset.
    UnsignedInt(IntTy),
    /// The 32-bit floating-point number type.
    Float32,
    /// The 64-bit floating point number type.
    Float64,
    /// A pointer to a value of a specific type.
    Pointer(PointerTy),
    /// A slice of elements of a specific element type.
    Slice(SliceTy),
    /// An array of elements of a specific element type.
    Array(ArrayTy),
    /// The zero-sized type.
    Void,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueTy(Arc<ValueTyKind>);

impl ValueTy {
    pub fn kind(&self) -> &ValueTyKind {
        &*self.0
    }
}

impl fmt::Display for ValueTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            ValueTyKind::Bool => f.write_str("bool"),
            ValueTyKind::SignedInt(ty) => {
                f.write_fmt(format_args!("i{}{:+}", usize::from(ty.bits), ty.offset))
            }
            ValueTyKind::UnsignedInt(ty) => {
                f.write_fmt(format_args!("u{}{:+}", usize::from(ty.bits), ty.offset))
            }
            ValueTyKind::Float32 => f.write_str("f32"),
            ValueTyKind::Float64 => f.write_str("f64"),
            ValueTyKind::Slice(ty) => ty.fmt(f),
            ValueTyKind::Array(ty) => {
                f.write_fmt(format_args!("[{}; {}]", ty.element_ty, ty.length))
            }
            ValueTyKind::Pointer(ty) => ty.fmt(f),
            ValueTyKind::Void => f.write_str("zst"),
        }
    }
}

/// A type error.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct TypeError(String);

/// Auxiliary macro for formatting a [`TypeError`].
macro_rules! fmt_type_error {
    ($($args:tt)*) => {
        TypeError(format!($($args)*))
    };
}

/// Auxiliary macro for returning a formatted [`TypeError`].
macro_rules! return_type_error {
    ($($args:tt)*) => {
        return Err(fmt_type_error!($($args)*))
    }
}

/// A simple [`Arc`] based interner for types.
#[derive(Debug)]
struct ArcInterner<V>(RwLock<HashSet<Arc<V>>>);

impl<V> ArcInterner<V> {
    /// Constructs a new interner.
    fn new() -> Self {
        Self(RwLock::default())
    }

    /// Interns the provided value.
    fn intern(&self, value: V) -> Arc<V>
    where
        V: Eq + Hash,
    {
        let values = self.0.read();
        if let Some(value) = values.get(&value) {
            value.clone()
        } else {
            drop(values);
            let mut values = self.0.write();
            if let Some(value) = values.get(&value) {
                value.clone()
            } else {
                let value = Arc::new(value);
                values.insert(value.clone());
                value
            }
        }
    }
}

impl<V> Default for ArcInterner<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A type context for constructing types.
#[derive(Debug, Default)]
pub struct TypeCtx {
    word_types: ArcInterner<WordTyKind>,
    value_types: ArcInterner<ValueTyKind>,
}

impl TypeCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn word_bool(&self) -> WordTy {
        self.intern_word_ty(WordTyKind::Bool)
    }

    pub fn word_int(&self) -> WordTy {
        self.intern_word_ty(WordTyKind::Int)
    }

    pub fn word_float(&self) -> WordTy {
        self.intern_word_ty(WordTyKind::Float)
    }

    pub fn value_bool(&self) -> ValueTy {
        self.intern_value_ty(ValueTyKind::Bool)
    }

    pub fn value_bounded_int(
        &self,
        lower_bound: Option<i64>,
        upper_bound: Option<i64>,
    ) -> Result<ValueTy, TypeError> {
        let lower_bound = lower_bound.unwrap_or(i64::MIN);
        let upper_bound = upper_bound.unwrap_or(i64::MAX);
        if upper_bound < lower_bound {
            return_type_error!("Upper bound on integer type must not be smaller than lower bound.");
        }
        if lower_bound < 0 {
            // Use a signed integer type.
            let magnitude = lower_bound.abs().max(upper_bound);
            let bits = (64 - magnitude.leading_zeros()) + 1;
            Ok(self.value_signed_int(IntTy::new((bits as usize).into())))
        } else {
            // Use an unsigned integer type with an offset.
            let offset = -lower_bound;
            let bits = 64 - (upper_bound - lower_bound).leading_zeros();
            Ok(self.value_unsigned_int(IntTy::new((bits as usize).into()).with_offset(offset)))
        }
    }

    pub fn value_signed_int(&self, ty: IntTy) -> ValueTy {
        assert!(
            ty.bits <= NumBits::from(64),
            "Signed integer type cannot have more than 64 bits."
        );
        self.intern_value_ty(ValueTyKind::SignedInt(ty))
    }

    pub fn value_unsigned_int(&self, ty: IntTy) -> ValueTy {
        assert!(
            // When representing integers, words are always 64-bit signed
            // integers. Hence, one bit is lost.
            ty.bits <= NumBits::from(63),
            "Unsigned integer type cannot have more than 63 bits."
        );
        self.intern_value_ty(ValueTyKind::UnsignedInt(ty))
    }

    pub fn value_array(&self, ty: ArrayTy) -> ValueTy {
        self.intern_value_ty(ValueTyKind::Array(ty))
    }

    pub fn value_float32(&self) -> ValueTy {
        self.intern_value_ty(ValueTyKind::Float32)
    }

    pub fn value_float64(&self) -> ValueTy {
        self.intern_value_ty(ValueTyKind::Float64)
    }

    pub fn intern_word_ty(&self, kind: WordTyKind) -> WordTy {
        WordTy(self.word_types.intern(kind))
    }

    pub fn intern_value_ty(&self, kind: ValueTyKind) -> ValueTy {
        ValueTy(self.value_types.intern(kind))
    }

    pub fn word_slice(&self, ty: SliceTy) -> WordTy {
        self.intern_word_ty(WordTyKind::Slice(ty))
    }

    pub fn word_pointer(&self, ty: PointerTy) -> WordTy {
        self.intern_word_ty(WordTyKind::Pointer(ty))
    }

    pub fn word_void(&self) -> WordTy {
        self.intern_word_ty(WordTyKind::Void)
    }

    /// Returns the common type to which both types coerce.
    pub fn common_coercion(&self, this: &WordTy, that: &WordTy) -> Result<WordTy, TypeError> {
        if this == that {
            return Ok(this.clone());
        }
        Ok(match (this.kind(), that.kind()) {
            (WordTyKind::Int, WordTyKind::Float) => self.word_float(),
            (WordTyKind::Float, WordTyKind::Int) => self.word_float(),
            _ => return_type_error!(
                "There is no common type to which `{}` and `{}` coerce.",
                this,
                that
            ),
        })
    }

    pub(crate) fn type_to_expr_type(&self, ty: &Type) -> Result<WordTy, TypeError> {
        Ok(match ty {
            Type::Int(_) => self.word_int(),
            Type::Real(_) => self.word_float(),
            Type::Bool => self.word_bool(),
            Type::Array(typ) => todo!(),
            Type::Clock => return_type_error!("Clocks are not supported."),
        })
    }

    /// Returns the type of the value.
    pub fn type_of_value(&self, value: &Value) -> WordTy {
        match value {
            Value::Int(_) => self.word_int(),
            Value::Float(_) => self.word_float(),
            Value::Bool(_) => self.word_bool(),
        }
    }

    /// The word type resulting when loading a value of the given type.
    pub fn loaded_value_ty(&self, ty: &ValueTy) -> Result<WordTy, TypeError> {
        macro_rules! invalid_type {
            () => {
                return_type_error!("Type `{}` does not fit into a word.", ty)
            };
        }
        Ok(match ty.kind() {
            ValueTyKind::Bool => self.word_bool(),
            ValueTyKind::SignedInt(_) => self.word_int(),
            ValueTyKind::UnsignedInt(_) => self.word_int(),
            ValueTyKind::Float32 => self.word_float(),
            ValueTyKind::Float64 => self.word_float(),
            ValueTyKind::Slice(ty) => self.word_slice(ty.clone()),
            ValueTyKind::Array(_) => invalid_type!(),
            ValueTyKind::Pointer(ty) => self.word_pointer(ty.clone()),
            ValueTyKind::Void => self.word_void(),
        })
    }
}

pub trait HasType {
    fn value_ty(ctx: &TypeCtx) -> ValueTy;
    fn word_ty(ctx: &TypeCtx) -> WordTy;
}

impl HasType for bool {
    fn value_ty(ctx: &TypeCtx) -> ValueTy {
        ctx.value_bool()
    }

    fn word_ty(ctx: &TypeCtx) -> WordTy {
        ctx.word_bool()
    }
}

impl HasType for i64 {
    fn value_ty(ctx: &TypeCtx) -> ValueTy {
        ctx.value_signed_int(IntTy::new(64.into()))
    }

    fn word_ty(ctx: &TypeCtx) -> WordTy {
        ctx.word_int()
    }
}

impl HasType for f64 {
    fn value_ty(ctx: &TypeCtx) -> ValueTy {
        ctx.value_float64()
    }

    fn word_ty(ctx: &TypeCtx) -> WordTy {
        ctx.word_float()
    }
}
