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
    Unary(Box<UnaryExpression>),
    Binary(Box<BinaryExpression>),
    Boolean(Box<BooleanExpression>),
    Comparison(Box<ComparisonExpression>),
    Trigonometric(Box<TrigonometricExpression>),
    Index(Box<IndexExpression>),
    Comprehension(Box<ComprehensionExpression>),
    Vector(Box<VectorExpression>),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct NameExpression {
    pub(crate) identifier: String,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ConstantExpression {
    pub(crate) value: Value,
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
    pub(crate) operator: UnaryOperator,
    pub(crate) operand: Expression,
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
    pub(crate) operator: BinaryOperator,
    pub(crate) left: Expression,
    pub(crate) right: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BooleanOperator {
    And,
    Or,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct BooleanExpression {
    pub(crate) operator: BooleanOperator,
    pub(crate) operands: Vec<Expression>,
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
    pub(crate) operator: ComparisonOperator,
    pub(crate) left: Expression,
    pub(crate) right: Expression,
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
    pub(crate) function: TrigonometricFunction,
    pub(crate) operand: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct IndexExpression {
    pub(crate) vector: Expression,
    pub(crate) index: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct ComprehensionExpression {
    pub(crate) variable: String,
    pub(crate) length: Expression,
    pub(crate) element: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct VectorExpression {
    pub(crate) elements: Vec<Expression>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Network {
    pub(crate) variables: IndexMap<String, Type>,
    pub(crate) clocks: IndexSet<String>,
    pub(crate) actions: IndexMap<String, Vec<Type>>,
    pub(crate) automata: IndexMap<String, Automaton>,
    pub(crate) initial: Vec<State>,
    pub(crate) links: Vec<Link>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Automaton {
    pub(crate) locations: IndexMap<String, Location>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ClockConstraint {
    // TODO: implement clock constraints
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Location {
    pub(crate) invariant: HashSet<ClockConstraint>,
    pub(crate) edges: Vec<Edge>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Edge {
    pub(crate) action: Action,
    pub(crate) guard: Expression,
    pub(crate) destinations: Vec<Destination>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Action {
    Internal,
    Pattern(ActionPattern),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct ActionPattern {
    pub(crate) name: String,
    pub(crate) arguments: Vec<PatternArgument>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "direction")]
pub enum PatternArgument {
    Write { value: Expression },
    Read { identifier: String },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Destination {
    pub(crate) location: String,
    pub(crate) probability: Expression,
    pub(crate) assignments: Vec<Assignment>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Assignment {
    pub(crate) target: Expression,
    pub(crate) value: Expression,
    pub(crate) index: usize,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct State {
    pub(crate) values: HashMap<String, Value>,
    pub(crate) locations: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Link {
    pub(crate) slots: IndexSet<String>,
    pub(crate) vector: IndexMap<String, LinkPattern>,
    pub(crate) result: LinkResult,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct LinkPattern {
    pub(crate) name: String,
    pub(crate) arguments: Vec<String>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum LinkResult {
    Internal,
    Pattern(LinkPattern),
}
