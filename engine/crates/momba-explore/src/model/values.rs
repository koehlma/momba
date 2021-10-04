//! Data structures for representing values.

use std::cmp;

use std::convert::TryInto;

use serde::{Deserialize, Serialize};

use ordered_float::NotNan;

use super::types::*;

use self::Value::*;

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(untagged)]
#[repr(u8)]
pub enum Value {
    Int64(i64),
    Float64(NotNan<f64>),
    Bool(bool),
    Vector(Vec<Value>),
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl TryInto<bool> for Value {
    type Error = String;

    fn try_into(self) -> Result<bool, Self::Error> {
        match self {
            Value::Bool(value) => Ok(value),
            _ => Err(format!("Unable to convert {:?} to boolean.", self)),
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int64(value)
    }
}

impl TryInto<i64> for Value {
    type Error = String;

    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Value::Int64(value) => Ok(value),
            _ => Err(format!("Unable to convert {:?} to integer.", self)),
        }
    }
}

impl TryInto<f64> for Value {
    type Error = String;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            Value::Float64(value) => Ok(value.into_inner()),
            _ => Err(format!("Unable to convert {:?} to float.", self)),
        }
    }
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::Int64(_) => Type::Int64,
            Value::Float64(_) => Type::Float64,
            Value::Bool(_) => Type::Bool,
            Value::Vector(elements) => Type::Vector {
                element_type: Box::new(
                    // All elements are required to have the same type, hence,
                    // we just take the first element of the vector.
                    elements
                        .first()
                        .map(|element| element.get_type())
                        .unwrap_or(Type::Unknown),
                ),
            },
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int64(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Value::Float64(_))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Value::Int64(_) | Value::Float64(_))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, Value::Vector(_))
    }

    pub fn unwrap_bool(&self) -> bool {
        match self {
            Value::Bool(value) => *value,
            _ => panic!("Value {:?} is not a Bool.", self),
        }
    }

    pub fn unwrap_int64(&self) -> i64 {
        match self {
            Value::Int64(value) => *value,
            _ => panic!("Value {:?} is not an Int64.", self),
        }
    }

    pub fn unwrap_float64(&self) -> NotNan<f64> {
        match self {
            Value::Float64(value) => *value,
            Value::Int64(value) => NotNan::new(*value as f64).unwrap(),
            _ => panic!("Value {:?} is not a Float64.", self),
        }
    }

    pub fn unwrap_vector(&self) -> &Vec<Value> {
        match self {
            Value::Vector(vector) => vector,
            _ => panic!("Value {:?} is not a Vector.", self),
        }
    }

    #[inline(always)]
    pub fn apply_not(self) -> Value {
        match self {
            Bool(operand) => Bool(!operand),
            operand => panic!("Invalid operand in expression (! {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_floor(self) -> Value {
        match self {
            Float64(operand) => Int64(operand.floor() as i64),
            Int64(operand) => Int64(operand),
            operand => panic!("Invalid operand in expression (floor {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_ceil(self) -> Value {
        match self {
            Float64(operand) => Int64(operand.ceil() as i64),
            operand => panic!("Invalid operand in expression (ceil {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_abs(self) -> Value {
        match self {
            Float64(operand) => Float64(NotNan::new(operand.abs()).unwrap()),
            Int64(operand) => Int64(operand.abs()),
            operand => panic!("Invalid operand in expression (abs {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_sgn(self) -> Value {
        match self {
            Float64(operand) => Float64(NotNan::new(operand.signum()).unwrap()),
            Int64(operand) => Int64(operand.signum()),
            operand => panic!("Invalid operand in expression (sgn {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_trc(self) -> Value {
        match self {
            Float64(operand) => Int64(operand.trunc() as i64),
            operand => panic!("Invalid operand in expression (trc {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_minus(self) -> Value {
        match self {
            Int64(operand) => Int64(-operand),
            Float64(operand) => Float64(-operand),
            operand => panic!("Invalid operand in expression (- {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_add(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(left + right),
            (Float64(left), Float64(right)) => Float64(left + right),
            (Int64(left), Float64(right)) => Float64(NotNan::new(left as f64).unwrap() + right),
            (Float64(left), Int64(right)) => Float64(left + NotNan::new(right as f64).unwrap()),
            (left, right) => panic!("Invalid operands in expression ({:?} + {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_sub(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(left - right),
            (Float64(left), Float64(right)) => Float64(left - right),
            (Int64(left), Float64(right)) => Float64(NotNan::new(left as f64).unwrap() - right),
            (Float64(left), Int64(right)) => Float64(left - NotNan::new(right as f64).unwrap()),
            (left, right) => panic!("Invalid operands in expression ({:?} - {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_mul(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(left * right),
            (Float64(left), Float64(right)) => Float64(left * right),
            (Int64(left), Float64(right)) => Float64(NotNan::new(left as f64).unwrap() * right),
            (Float64(left), Int64(right)) => Float64(left * NotNan::new(right as f64).unwrap()),
            (left, right) => panic!("Invalid operands in expression ({:?} * {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_floor_div(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(left.div_euclid(right)),
            (Float64(left), Float64(right)) => Int64((left / right).floor() as i64),
            (Int64(left), Float64(right)) => {
                Int64((NotNan::new(left as f64).unwrap() / right).floor() as i64)
            }
            (Float64(left), Int64(right)) => {
                Int64((left / (NotNan::new(right as f64).unwrap()).floor()).into_inner() as i64)
            }
            (left, right) => panic!(
                "Invalid operands in expression ({:?} // {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_real_div(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => {
                Float64(NotNan::new((left as f64) / (right as f64)).unwrap())
            }
            (Float64(left), Float64(right)) => Float64(left / right),
            (left, right) => panic!("Invalid operands in expression ({:?} / {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_mod(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(left.rem_euclid(right)),
            (Float64(left), Float64(right)) => Float64(left % right),
            (left, right) => panic!("Invalid operands in expression ({:?} % {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_pow(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => {
                Float64(NotNan::new((left as f64).powf(right as f64)).unwrap())
            }
            (Float64(left), Float64(right)) => {
                Float64(NotNan::new(left.powf(right.into())).unwrap())
            }
            (left, right) => panic!(
                "Invalid operands in expression ({:?} ** {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_log(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => {
                Float64(NotNan::new((left as f64).log(right as f64)).unwrap())
            }
            (Float64(left), Float64(right)) => {
                Float64(NotNan::new(left.log(right.into())).unwrap())
            }
            (left, right) => panic!(
                "Invalid operands in expression ({:?} log {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_min(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(cmp::min(left, right)),
            (Float64(left), Float64(right)) => Float64(cmp::min(left, right)),
            (left, right) => panic!(
                "Invalid operands in expression ({:?} min {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_max(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Int64(cmp::max(left, right)),
            (Float64(left), Float64(right)) => Float64(cmp::max(left, right)),
            (left, right) => panic!(
                "Invalid operands in expression ({:?} max {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_cmp_eq(self, other: Value) -> Value {
        Value::Bool(self == other)
    }

    #[inline(always)]
    pub fn apply_cmp_ne(self, other: Value) -> Value {
        Value::Bool(self != other)
    }

    #[inline(always)]
    pub fn apply_cmp_lt(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Bool(left < right),
            (Float64(left), Float64(right)) => Bool(left < right),
            (Int64(left), Float64(right)) => Bool((left as f64) < right.into_inner()),
            (Float64(left), Int64(right)) => Bool(left.into_inner() < (right as f64)),
            (left, right) => panic!("Invalid operands in expression ({:?} < {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_cmp_le(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Bool(left <= right),
            (Float64(left), Float64(right)) => Bool(left <= right),
            (Int64(left), Float64(right)) => Bool((left as f64) <= right.into_inner()),
            (Float64(left), Int64(right)) => Bool(left.into_inner() <= (right as f64)),
            (left, right) => panic!(
                "Invalid operands in expression ({:?} <= {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_cmp_ge(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Bool(left >= right),
            (Float64(left), Float64(right)) => Bool(left >= right),
            (Int64(left), Float64(right)) => Bool((left as f64) >= right.into_inner()),
            (Float64(left), Int64(right)) => Bool(left.into_inner() >= (right as f64)),
            (left, right) => panic!(
                "Invalid operands in expression ({:?} >= {:?}).",
                left, right
            ),
        }
    }

    #[inline(always)]
    pub fn apply_cmp_gt(self, other: Value) -> Value {
        match (self, other) {
            (Int64(left), Int64(right)) => Bool(left > right),
            (Float64(left), Float64(right)) => Bool(left > right),
            (Int64(left), Float64(right)) => Bool((left as f64) > right.into_inner()),
            (Float64(left), Int64(right)) => Bool(left.into_inner() > (right as f64)),
            (left, right) => panic!("Invalid operands in expression ({:?} > {:?}).", left, right),
        }
    }

    #[inline(always)]
    pub fn apply_sin(self) -> Value {
        match self {
            Float64(operand) => Float64(operand.sin().try_into().unwrap()),
            operand => panic!("Invalid operand in expression (sin {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_cos(self) -> Value {
        match self {
            Float64(operand) => Float64(operand.cos().try_into().unwrap()),
            operand => panic!("Invalid operand in expression (sin {:?}).", operand),
        }
    }

    #[inline(always)]
    pub fn apply_tan(self) -> Value {
        match self {
            Float64(operand) => Float64(operand.tan().try_into().unwrap()),
            operand => panic!("Invalid operand in expression (sin {:?}).", operand),
        }
    }
}
