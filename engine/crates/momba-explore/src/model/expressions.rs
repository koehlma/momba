//! Data structures for representing expressions.

use serde::{Deserialize, Serialize};

use super::values::*;

/// Represents an expression.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Expression {
    Name(NameExpression),
    Constant(ConstantExpression),
    Unary(UnaryExpression),
    Binary(BinaryExpression),
    Boolean(BooleanExpression),
    Comparison(ComparisonExpression),
    Conditional(ConditionalExpression),
    Trigonometric(TrigonometricExpression),
    Index(IndexExpression),
    Comprehension(ComprehensionExpression),
    Vector(VectorExpression),
}

/// Represents a name expression.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct NameExpression {
    pub identifier: String,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ConstantExpression {
    pub value: Value,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UnaryOperator {
    Not,
    Minus,
    Floor,
    Ceil,
    Abs,
    Sgn,
    Trc,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
}

/// Operators for binary expressions.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    FloorDiv,
    RealDiv,
    Mod,
    Pow,
    Log,
    Min,
    Max,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

/// Operators for boolean expressions.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BooleanOperator {
    And,
    Or,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct BooleanExpression {
    pub operator: BooleanOperator,
    pub operands: Vec<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComparisonOperator {
    Eq,
    Ne,
    Lt,
    Le,
    Ge,
    Gt,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ComparisonExpression {
    pub operator: ComparisonOperator,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ConditionalExpression {
    pub condition: Box<Expression>,
    pub consequence: Box<Expression>,
    pub alternative: Box<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
    Cot,
    Sec,
    Csc,
    ArcSin,
    ArcCos,
    ArcTan,
    ArcCot,
    ArcSec,
    ArcCsc,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct TrigonometricExpression {
    pub function: TrigonometricFunction,
    pub operand: Box<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct IndexExpression {
    pub vector: Box<Expression>,
    pub index: Box<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ComprehensionExpression {
    pub variable: String,
    pub length: Box<Expression>,
    pub element: Box<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct VectorExpression {
    pub elements: Vec<Expression>,
}
