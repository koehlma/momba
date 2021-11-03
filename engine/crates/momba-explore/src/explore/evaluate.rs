use std::cmp::max;
use std::convert::TryInto;

use indexmap::{IndexSet, IndexMap};

use super::model::*;

/// Represents an evaluation environment.
#[derive(Debug)]
pub struct Environment<'r, const BANKS: usize> {
    pub(crate) banks: [&'r [Value]; BANKS],
}

/// Represents a register address.
#[derive(Debug)]
pub struct RegisterAddress {
    bank: usize,
    register: usize,
}

impl<'r, const BANKS: usize> Environment<'r, BANKS> {
    /// Creates a new evaluation environment from the given register banks.
    pub fn new(banks: [&'r [Value]; BANKS]) -> Self {
        Environment { banks }
    }

    /// Returns the value stored at the given address.
    pub fn get_value(&self, address: &RegisterAddress) -> &Value {
        &self.banks[address.bank][address.register]
    }
}

/// Represents a compiled expression.
pub struct CompiledExpression<const BANKS: usize> {
    closure: Box<dyn Send + Sync + Fn(&Environment<BANKS>, &mut Vec<Value>) -> Value>,
    stack_depth: usize,
}

impl<const BANKS: usize> CompiledExpression<BANKS> {
    fn new(
        closure: impl 'static + Send + Sync + Fn(&Environment<BANKS>, &mut Vec<Value>) -> Value,
        stack_depth: usize,
    ) -> Self {
        CompiledExpression {
            closure: Box::new(closure),
            stack_depth,
        }
    }

    fn evaluate_with_stack(&self, env: &Environment<BANKS>, stack: &mut Vec<Value>) -> Value {
        (self.closure)(env, stack)
    }

    pub fn evaluate(&self, env: &Environment<BANKS>) -> Value {
        self.evaluate_with_stack(env, &mut Vec::with_capacity(self.stack_depth))
    }
}

/// Represents an assignment target.
pub struct Target<'e> {
    store: &'e mut [Value],
    index: usize,
}

impl<'e> Target<'e> {
    pub fn store(&mut self, value: Value) {
        self.store[self.index] = value
    }

    pub fn resolve(self) -> &'e mut Value {
        &mut self.store[self.index]
    }
}

/// Represents a compiled assignment target.
pub struct CompiledTargetExpression<const BANKS: usize> {
    closure: Box<
        dyn Send
            + Sync
            + for<'e> Fn(&'e mut [Value], &Environment<BANKS>, &mut Vec<Value>) -> Target<'e>,
    >,
    stack_depth: usize,
}

impl<const BANKS: usize> CompiledTargetExpression<BANKS> {
    fn new(
        closure: impl 'static
            + Send
            + Sync
            + for<'e> Fn(&'e mut [Value], &Environment<BANKS>, &mut Vec<Value>) -> Target<'e>,
        stack_depth: usize,
    ) -> Self {
        CompiledTargetExpression {
            closure: Box::new(closure),
            stack_depth,
        }
    }

    fn evaluate_with_stack<'e>(
        &self,
        targets: &'e mut [Value],
        env: &Environment<BANKS>,
        stack: &mut Vec<Value>,
    ) -> Target<'e> {
        (self.closure)(targets, env, stack)
    }

    pub fn evaluate<'e>(&self, targets: &'e mut [Value], env: &Environment<BANKS>) -> Target<'e> {
        self.evaluate_with_stack(targets, env, &mut Vec::with_capacity(self.stack_depth))
    }
}

pub trait CompileBackend<C> {
    fn compile_name(&self, identifier: &str) -> C;

    fn compile_with_context(&self, expression: &Expression, ctx: &mut CompileContext) -> C;

    fn compile(&self, expression: &Expression) -> C;
}

#[derive(Clone)]
pub struct CompileContext {
    max_stack_depth: usize,
    stack_variables: IndexSet<String>,
}

impl CompileContext {
    fn new() -> Self {
        CompileContext {
            max_stack_depth: 0,
            stack_variables: IndexSet::new(),
        }
    }

    fn push_stack_variable(&mut self, identifier: String) {
        self.stack_variables.insert(identifier);
        if self.stack_variables.len() > self.max_stack_depth {
            self.max_stack_depth = self.stack_variables.len()
        }
    }

    fn pop_stack_variable(&mut self) {
        self.stack_variables.pop();
    }

    fn get_stack_index(&self, identifier: &str) -> Option<usize> {
        self.stack_variables.get_index_of(identifier)
    }
}

pub struct Scope<const BANKS: usize> {
    banks: [IndexMap<String, usize>; BANKS],
}

impl<const BANKS: usize> Scope<BANKS> {
    pub fn get_address(&self, identifier: &str) -> Option<RegisterAddress> {
        self.banks
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(bank, identifiers)| {
                identifiers.get(identifier).map(|register| RegisterAddress {
                    bank,
                    register: *register,
                })
            })
            .next()
    }

    fn compile_with_context(
        &self,
        expression: &Expression,
        ctx: &mut CompileContext,
    ) -> CompiledExpression<BANKS> {
        macro_rules! compile {
            ($expr:expr) => {
                self.compile_with_context($expr, ctx)
            };
            ($expr:expr; push stack $var:expr) => {{
                ctx.push_stack_variable($var.into());
                let compiled = self.compile_with_context($expr, ctx);
                ctx.pop_stack_variable();
                compiled
            }};
        }

        macro_rules! evaluate {
            ($expr:expr, $env:expr, $stack:expr) => {
                $expr.evaluate_with_stack($env, $stack)
            };
            ($expr:expr, $env:expr, $stack:expr; push stack $val:expr) => {{
                $stack.push($val.into());
                let result = $expr.evaluate_with_stack($env, $stack);
                $stack.pop();
                result
            }};
        }

        macro_rules! construct {
            ($closure:expr) => {
                CompiledExpression::new($closure, ctx.max_stack_depth)
            };
        }

        match expression {
            Expression::Name(NameExpression { identifier }) => {
                ctx.get_stack_index(identifier).map_or_else(
                    || {
                        let address = self
                            .get_address(identifier)
                            .expect(&format!("invalid identifier `{}`", identifier));
                        construct!(move |env, _| env.get_value(&address).clone())
                    },
                    |index| construct!(move |_, stack| stack[index].clone()),
                )
            }
            Expression::Constant(ConstantExpression { value }) => {
                let value = value.clone();
                construct!(move |_, _| value.clone())
            }
            Expression::Unary(UnaryExpression { operator, operand }) => {
                let operand = compile!(operand);

                macro_rules! compile_unary {
                    ($function:ident) => {
                        construct!(move |env, stack| evaluate!(operand, env, stack).$function())
                    };
                }

                match operator {
                    UnaryOperator::Not => compile_unary!(apply_not),
                    UnaryOperator::Minus => compile_unary!(apply_minus),
                    UnaryOperator::Floor => compile_unary!(apply_floor),
                    UnaryOperator::Ceil => compile_unary!(apply_ceil),
                    UnaryOperator::Abs => compile_unary!(apply_abs),
                    UnaryOperator::Sgn => compile_unary!(apply_sgn),
                    UnaryOperator::Trc => compile_unary!(apply_trc),
                }
            }
            Expression::Binary(BinaryExpression {
                operator,
                left,
                right,
            }) => {
                let left = compile!(left);
                let right = compile!(right);

                macro_rules! compile_binary {
                    ($function:ident) => {
                        construct!(move |env, stack| evaluate!(left, env, stack)
                            .$function(evaluate!(right, env, stack)))
                    };
                }

                match operator {
                    BinaryOperator::Add => compile_binary!(apply_add),
                    BinaryOperator::Sub => compile_binary!(apply_sub),
                    BinaryOperator::Mul => compile_binary!(apply_mul),
                    BinaryOperator::FloorDiv => compile_binary!(apply_floor_div),
                    BinaryOperator::RealDiv => compile_binary!(apply_real_div),
                    BinaryOperator::Mod => compile_binary!(apply_mod),
                    BinaryOperator::Log => compile_binary!(apply_log),
                    BinaryOperator::Pow => compile_binary!(apply_pow),
                    BinaryOperator::Min => compile_binary!(apply_min),
                    BinaryOperator::Max => compile_binary!(apply_max),
                }
            }
            Expression::Comparison(ComparisonExpression {
                operator,
                left,
                right,
            }) => {
                let left = compile!(left);
                let right = compile!(right);

                macro_rules! compile_comparison {
                    ($function:ident) => {
                        construct!(move |env, stack| evaluate!(left, env, stack)
                            .$function(evaluate!(right, env, stack)))
                    };
                }

                match operator {
                    ComparisonOperator::Eq => compile_comparison!(apply_cmp_eq),
                    ComparisonOperator::Ne => compile_comparison!(apply_cmp_ne),
                    ComparisonOperator::Lt => compile_comparison!(apply_cmp_lt),
                    ComparisonOperator::Le => compile_comparison!(apply_cmp_le),
                    ComparisonOperator::Ge => compile_comparison!(apply_cmp_ge),
                    ComparisonOperator::Gt => compile_comparison!(apply_cmp_gt),
                }
            }
            Expression::Boolean(BooleanExpression { operator, operands }) => {
                let operands: Box<[_]> = operands.iter().map(|operand| compile!(operand)).collect();

                macro_rules! compile_boolean {
                    ($function:ident) => {
                        construct!(move |env, stack| {
                            operands
                                .iter()
                                .$function(|operand| {
                                    evaluate!(operand, env, stack).try_into().unwrap()
                                })
                                .into()
                        })
                    };
                }

                match operator {
                    BooleanOperator::And => compile_boolean!(all),
                    BooleanOperator::Or => compile_boolean!(any),
                }
            }
            Expression::Comprehension(ComprehensionExpression {
                variable,
                length,
                element,
            }) => {
                let length = compile!(length);
                let element = compile!(element; push stack variable);

                construct!(move |env, stack| {
                    let length = evaluate!(length, env, stack).try_into().unwrap();
                    Value::Vector(
                        (0..length)
                            .map(|index| evaluate!(element, env, stack; push stack index))
                            .collect(),
                    )
                })
            }
            Expression::Conditional(ConditionalExpression {
                condition,
                consequence,
                alternative,
            }) => {
                let condition = compile!(condition);
                let consequence = compile!(consequence);
                let alternative = compile!(alternative);

                construct!(move |env, stack| {
                    if evaluate!(condition, env, stack).try_into().unwrap() {
                        evaluate!(consequence, env, stack)
                    } else {
                        evaluate!(alternative, env, stack)
                    }
                })
            }
            Expression::Vector(VectorExpression { elements }) => {
                let elements: Vec<_> = elements.iter().map(|element| compile!(element)).collect();

                construct!(move |env, stack| {
                    Value::Vector(
                        elements
                            .iter()
                            .map(|element| evaluate!(element, env, stack))
                            .collect(),
                    )
                })
            }
            Expression::Trigonometric(TrigonometricExpression { function, operand }) => {
                let operand = compile!(operand);

                macro_rules! compile_trigonometric {
                    ($function:ident) => {
                        construct!(move |env, stack| evaluate!(operand, env, stack).$function())
                    };
                }

                match function {
                    TrigonometricFunction::Sin => compile_trigonometric!(apply_sin),
                    TrigonometricFunction::Cos => compile_trigonometric!(apply_cos),
                    TrigonometricFunction::Tan => compile_trigonometric!(apply_tan),
                    _ => panic!("trigonometric function {:?} not implemented", function),
                }
            }
            Expression::Index(IndexExpression { vector, index }) => {
                let vector = self.compile(vector);
                let index = self.compile(index);
                construct!(move |env, stack| {
                    evaluate!(vector, env, stack).unwrap_vector()[evaluate!(index, env, stack).unwrap_int64() as usize].clone()
                })
            }
            _ => panic!("not implemented {:?}", expression),
        }
    }

    pub fn compile(&self, expression: &Expression) -> CompiledExpression<BANKS> {
        self.compile_with_context(expression, &mut CompileContext::new())
    }

    pub fn compile_target(&self, expression: &Expression) -> CompiledTargetExpression<BANKS> {
        match expression {
            Expression::Name(NameExpression { identifier }) => {
                let address = self.get_address(identifier).unwrap();
                // Is the identifier a global variable?
                let index = address.register;
                CompiledTargetExpression::new(
                    move |targets, _, _| match &mut targets[address.bank] {
                        Value::Vector(vector) => Target {
                            store: vector,
                            index: index,
                        },
                        _ => panic!("Expected vector got."),
                    },
                    0,
                )
            }
            Expression::Index(IndexExpression { vector, index }) => {
                let vector = self.compile_target(vector);
                let index = self.compile(index);
                let stack_depth = max(vector.stack_depth, index.stack_depth);
                CompiledTargetExpression::new(
                    move |targets, env, stack| {
                        let index = index.evaluate_with_stack(env, stack);
                        let vector = vector.evaluate_with_stack(targets, env, stack);
                        match (vector.resolve(), index) {
                            (Value::Vector(vector), Value::Int64(index)) => Target {
                                store: vector,
                                index: index as usize,
                            },
                            tuple => {
                                panic!("Unable to construct assignment target from {:?}.", tuple)
                            }
                        }
                    },
                    stack_depth,
                )
            }
            _ => panic!("Unable to compile target from expression {:?}.", expression),
        }
    }
}

impl Network {
    pub fn global_scope(&self) -> Scope<2> {
        Scope {
            banks: [
                self.declarations
                    .global_variables
                    .keys()
                    .enumerate()
                    .map(|(index, identifier)| (identifier.clone(), index))
                    .collect(),
                self.declarations
                    .transient_variables
                    .keys()
                    .enumerate()
                    .map(|(index, identifier)| (identifier.clone(), index))
                    .collect(),
            ],
        }
    }

    pub fn transient_scope(&self) -> Scope<1> {
        Scope {
            banks: [self
                .declarations
                .global_variables
                .keys()
                .enumerate()
                .map(|(index, identifier)| (identifier.clone(), index))
                .collect()],
        }
    }
}

impl Edge {
    pub fn edge_scope(&self, network: &Network, edge: &Edge) -> Scope<3> {
        let global_scope = network.global_scope();
        Scope {
            banks: [
                global_scope.banks[0].clone(),
                global_scope.banks[1].clone(),
                match &edge.pattern {
                    ActionPattern::Silent => IndexMap::new(),
                    ActionPattern::Labeled(labeled) => labeled
                        .arguments
                        .iter()
                        .enumerate()
                        .filter_map(|(index, argument)| match argument {
                            PatternArgument::Read { identifier } => {
                                Some((identifier.clone(), index))
                            }
                            _ => None,
                        })
                        .collect(),
                },
            ],
        }
    }
}
