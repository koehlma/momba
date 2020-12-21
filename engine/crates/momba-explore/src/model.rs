//! Momba's intermediate representation for PTAs and MDPs.
//!
//! This module define the structure of *Momba's Intermediate Representation* (MombaIR).
//! The structure is defined directly in Rust using [Serde](https://serde.rs) and `derive`.
//! As a result, any format supported by Serde can be used to store MombaIR models.
//! Usually, however, MombaIR models are stored using the JSON format.
//!
//! MombaIR has been inspired by the [JANI](https://jani-spec.org) model interchange format.
//! In comparison to JANI, it gives up some convenient higher-level modeling features in favor of
//! simplicity and being more low-level.
//! MombaIR is not intended to be used directly for modeling.
//! Instead a higher-level modeling formalism such as JANI should be used which is then
//! translated to MombaIR.

use std::collections::{HashMap, HashSet};

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::values::*;

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

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Network {
    pub declarations: Declarations,
    pub automata: IndexMap<String, Automaton>,
    pub links: Vec<Link>,
    pub initial_states: Vec<State>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Declarations {
    pub global_variables: IndexMap<String, Type>,
    pub transient_variables: HashMap<String, Expression>,
    pub clock_variables: IndexSet<String>,
    pub action_types: IndexMap<String, Vec<Type>>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Automaton {
    pub locations: IndexMap<String, Location>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Location {
    pub invariant: HashSet<ClockConstraint>,
    pub edges: Vec<Edge>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ClockConstraint {
    pub left: Clock,
    pub right: Clock,
    pub bound: Bound,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Clock {
    Zero,
    Variable(ClockVariable),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ClockVariable {
    pub identifier: String,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct Bound {
    pub is_strict: bool,
    pub constant: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Edge {
    pub action: Action,
    pub guard: Guard,
    pub destinations: Vec<Destination>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Guard {
    pub boolean_condition: Expression,
    pub clock_constraints: HashSet<ClockConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Action {
    Internal,
    Pattern(ActionPattern),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct ActionPattern {
    pub name: String,
    pub arguments: Vec<PatternArgument>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "direction")]
pub enum PatternArgument {
    Write(WriteArgument),
    Read(ReadArgument),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct WriteArgument {
    pub value: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct ReadArgument {
    pub identifier: String,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Destination {
    pub location: String,
    pub probability: Expression,
    pub assignments: Vec<Assignment>,
    pub reset: HashSet<Clock>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct State {
    pub values: HashMap<String, Value>,
    pub locations: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Link {
    pub slots: IndexSet<String>,
    pub vector: IndexMap<String, LinkPattern>,
    pub result: LinkResult,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct LinkPattern {
    pub name: String,
    pub arguments: Vec<String>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum LinkResult {
    Internal,
    Pattern(LinkPattern),
}
