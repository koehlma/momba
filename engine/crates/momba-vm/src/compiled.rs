use std::collections::HashMap;

use ordered_float::NotNan;

use itertools::Itertools;

use crate::model::*;
use crate::types::*;
use crate::values::*;

#[derive(Debug)]
pub struct CompilationContext {
    stack_depth: usize,
    stack_positions: HashMap<String, usize>,
}

impl CompilationContext {
    fn get_stack_offset(&self, name: &str) -> usize {
        self.stack_positions[name] - self.stack_depth
    }
}

#[derive(Debug)]
pub struct EvaluationContext {
    values: Box<[Value]>,
}

pub struct LValue<'c> {
    storage: &'c mut [Value],
    index: usize,
}

impl<'c> LValue<'c> {
    pub fn store(&mut self, value: Value) {
        self.storage[self.index] = value
    }

    pub fn resolve(self) -> &'c mut Value {
        &mut self.storage[self.index]
    }
}

pub struct CompiledTarget<'c> {
    closure: Box<dyn 'c + Fn(&mut EvaluationContext) -> LValue>,
    expression: Expression,
}

impl<'c> CompiledTarget<'c> {
    pub fn new(
        closure: impl 'c + Fn(&mut EvaluationContext) -> LValue,
        expression: Expression,
    ) -> Self {
        CompiledTarget {
            closure: Box::new(closure),
            expression: expression,
        }
    }

    pub fn evaluate<'e>(&self, ctx: &'e mut EvaluationContext) -> LValue<'e> {
        (self.closure)(ctx)
    }
}

impl Expression {
    pub fn compile_target<'c>(&self, network: &Network) -> CompiledTarget<'c> {
        match self {
            Expression::Name(NameExpression { identifier }) => {
                let index = network
                    .variables
                    .get_index_of(identifier)
                    .expect(identifier);
                CompiledTarget::new(
                    move |ctx| LValue {
                        storage: &mut ctx.values,
                        index: index,
                    },
                    self.clone(),
                )
            }
            Expression::Index(expression) => {
                let index = expression.index.compile(network);
                let vector = expression.vector.compile_target(network);
                CompiledTarget::new(
                    move |ctx| {
                        let index = index.evaluate(ctx);
                        let vector = vector.evaluate(ctx);
                        match (vector.resolve(), index) {
                            (Value::Vector(vector), Value::Int64(index)) => LValue {
                                storage: vector,
                                index: index as usize,
                            },
                            _ => panic!("error"),
                        }
                    },
                    self.clone(),
                )
            }
            _ => panic!("compile not implemented for expression"),
        }
    }
}

pub struct CompiledExpression<'c> {
    closure: Box<dyn 'c + Fn(&EvaluationContext) -> Value>,
    expression: Expression,
}

impl<'c> CompiledExpression<'c> {
    pub fn new(closure: impl 'c + Fn(&EvaluationContext) -> Value, expression: Expression) -> Self {
        CompiledExpression {
            closure: Box::new(closure),
            expression: expression,
        }
    }

    pub fn evaluate(&self, ctx: &EvaluationContext) -> Value {
        let result = (self.closure)(ctx);
        //println!("{:?} => {:?}", self.expression, result);
        result
    }
}

impl Expression {
    pub fn compile<'c>(&self, network: &Network) -> CompiledExpression<'c> {
        match self {
            Expression::Name(NameExpression { identifier }) => {
                let index = network
                    .variables
                    .get_index_of(identifier)
                    .expect(identifier);
                CompiledExpression::new(move |ctx| ctx.values[index].clone(), self.clone())
            }
            Expression::Constant(ConstantExpression { value }) => {
                let value = value.clone();
                CompiledExpression::new(move |_| value.clone(), self.clone())
            }
            Expression::Unary(expression) => {
                let operand = expression.operand.compile(network);

                macro_rules! compile_unary {
                    ($function:ident) => {
                        CompiledExpression::new(
                            move |ctx| operand.evaluate(ctx).$function(),
                            self.clone(),
                        )
                    };
                }

                match expression.operator {
                    UnaryOperator::Not => compile_unary!(apply_not),
                    UnaryOperator::Minus => compile_unary!(apply_minus),
                    UnaryOperator::Floor => compile_unary!(apply_floor),
                    UnaryOperator::Ceil => compile_unary!(apply_ceil),
                    UnaryOperator::Abs => compile_unary!(apply_abs),
                    UnaryOperator::Sgn => compile_unary!(apply_sgn),
                    UnaryOperator::Trc => compile_unary!(apply_trc),
                }
            }
            Expression::Binary(expression) => {
                let left = expression.left.compile(network);
                let right = expression.right.compile(network);

                macro_rules! compile_binary {
                    ($function:ident) => {
                        CompiledExpression::new(
                            move |ctx| left.evaluate(ctx).$function(right.evaluate(ctx)),
                            self.clone(),
                        )
                    };
                }

                match expression.operator {
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
            Expression::Comparison(expression) => {
                let left = expression.left.compile(network);
                let right = expression.right.compile(network);

                macro_rules! compile_comparison {
                    ($function:ident) => {
                        CompiledExpression::new(
                            move |ctx| left.evaluate(ctx).$function(right.evaluate(ctx)),
                            self.clone(),
                        )
                    };
                }

                match expression.operator {
                    ComparisonOperator::Eq => compile_comparison!(apply_cmp_eq),
                    ComparisonOperator::Ne => compile_comparison!(apply_cmp_ne),
                    ComparisonOperator::Lt => compile_comparison!(apply_cmp_lt),
                    ComparisonOperator::Le => compile_comparison!(apply_cmp_le),
                    ComparisonOperator::Ge => compile_comparison!(apply_cmp_ge),
                    ComparisonOperator::Gt => compile_comparison!(apply_cmp_gt),
                }
            }
            Expression::Boolean(expression) => {
                let operands: Box<[CompiledExpression<'c>]> = expression
                    .operands
                    .iter()
                    .map(|operand| operand.compile(network))
                    .collect();

                macro_rules! compile_comparison {
                    ($function:ident) => {
                        CompiledExpression::new(
                            move |ctx| {
                                Value::Bool(
                                    operands
                                        .iter()
                                        .$function(|operand| operand.evaluate(ctx).unwrap_bool()),
                                )
                            },
                            self.clone(),
                        )
                    };
                }

                match expression.operator {
                    BooleanOperator::And => compile_comparison!(all),
                    BooleanOperator::Or => compile_comparison!(any),
                }
            }

            _ => panic!("Compilation not implemented for {:?}.", self),
        }
    }
}

pub struct CompiledAssignment<'c> {
    pub(crate) target: CompiledTarget<'c>,
    pub(crate) value: CompiledExpression<'c>,
    pub(crate) index: usize,
}

impl Assignment {
    pub fn compile<'c>(&self, network: &Network) -> CompiledAssignment<'c> {
        CompiledAssignment {
            target: self.target.compile_target(network),
            value: self.value.compile(network),
            index: self.index,
        }
    }
}

pub struct CompiledEdge<'c> {
    pub(crate) guard: CompiledExpression<'c>,
    pub(crate) action: CompiledAction<'c>,
    pub(crate) destinations: Box<[CompiledDestination<'c>]>,
}

pub struct CompiledDestination<'c> {
    pub(crate) location: usize,
    pub(crate) probability: CompiledExpression<'c>,
    pub(crate) assignments: Box<[CompiledAssignment<'c>]>,
}

pub struct CompiledLocation<'c> {
    pub(crate) edges: Box<[CompiledEdge<'c>]>,
}

pub struct CompiledAutomaton<'c> {
    pub(crate) locations: Box<[CompiledLocation<'c>]>,
}

impl Automaton {
    pub fn compile<'c>(&self, network: &Network) -> CompiledAutomaton<'_> {
        CompiledAutomaton {
            locations: self
                .locations
                .values()
                .map(|location| CompiledLocation {
                    edges: location
                        .edges
                        .iter()
                        .map(|edge| CompiledEdge {
                            guard: edge.guard.compile(network),
                            action: edge.action.compile(network),
                            destinations: edge
                                .destinations
                                .iter()
                                .map(|destination| CompiledDestination {
                                    location: self
                                        .locations
                                        .get_index_of(&destination.location)
                                        .unwrap(),
                                    probability: destination.probability.compile(network),
                                    assignments: destination
                                        .assignments
                                        .iter()
                                        .map(|assignment| CompiledAssignment {
                                            target: assignment.target.compile_target(network),
                                            value: assignment.value.compile(network),
                                            index: assignment.index,
                                        })
                                        .collect(),
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct CompiledState<'c, 's> {
    pub(crate) network: &'s CompiledNetwork<'c>,
    pub(crate) values: Box<[Value]>,
    pub(crate) locations: Box<[usize]>,
}

#[derive(Clone)]
pub struct InstanceTransition<'c, 't> {
    automaton: usize,
    edge: &'t CompiledEdge<'c>,
    arguments: Box<[Value]>,
}

pub struct NetworkTransition<'c, 't> {
    instance_transitions: Vec<InstanceTransition<'c, 't>>,
    result_action: ResultAction,
}

pub enum ResultAction {
    Internal,
    Visible(usize, Box<[Value]>),
}

pub struct ComputedDestination<'c, 's> {
    pub(crate) probability: Value,
    pub(crate) state: CompiledState<'c, 's>,
}

impl<'c, 's> CompiledState<'c, 's> {
    pub fn execute<'t>(
        &self,
        transition: NetworkTransition<'c, 't>,
    ) -> Vec<ComputedDestination<'c, '_>> {
        let destinations = transition
            .instance_transitions
            .iter()
            .map(|transition| {
                transition
                    .edge
                    .destinations
                    .iter()
                    .map(move |destination| (transition, destination))
            })
            .multi_cartesian_product();
        let mut result = Vec::new();
        let eval_ctx = EvaluationContext {
            values: self.values.clone(),
        };
        for x in destinations {
            let mut ctx = EvaluationContext {
                values: self.values.clone(),
            };
            let mut locations = self.locations.clone();
            let mut probability = Value::Float64(NotNan::new(1.0).unwrap());
            for (transition, destination) in x {
                probability = probability.apply_mul(destination.probability.evaluate(&ctx));
                for assignment in destination.assignments.iter() {
                    let value = assignment.value.evaluate(&eval_ctx);
                    assignment.target.evaluate(&mut ctx).store(value);
                }
                locations[transition.automaton] = destination.location;
            }
            result.push(ComputedDestination {
                probability: probability,
                state: CompiledState {
                    network: self.network,
                    values: ctx.values,
                    locations: locations,
                },
            });
        }

        result
    }

    pub fn transitions(&self) -> Vec<NetworkTransition<'c, '_>> {
        let ctx = EvaluationContext {
            values: self.values.clone(),
        };

        let locations = self
            .locations
            .iter()
            .zip(self.network.automata.iter())
            .map(|(location_index, automaton)| &automaton.locations[*location_index]);

        let mut transitions = Vec::new();

        let mut automata_active_edges = Vec::new();

        for (index, location) in locations.enumerate() {
            let mut active_edges: Vec<Vec<&CompiledEdge>> =
                Vec::with_capacity(self.network.network.actions.len());

            for _ in 0..self.network.network.actions.len() {
                active_edges.push(Vec::new());
            }

            for edge in location.edges.iter() {
                if edge.guard.evaluate(&ctx).unwrap_bool() {
                    match &edge.action {
                        CompiledAction::Internal => transitions.push(NetworkTransition {
                            instance_transitions: vec![InstanceTransition {
                                automaton: index,
                                arguments: Box::new([]),
                                edge: edge,
                            }],
                            result_action: ResultAction::Internal,
                        }),
                        CompiledAction::Pattern(pattern) => active_edges[pattern.action].push(edge),
                    }
                }
            }
            automata_active_edges.push(active_edges);
        }

        for link in &self.network.network.links {
            let instance_transitions = link
                .vector
                .iter()
                .map(|(name, link_pattern)| {
                    let automaton_index = self.network.network.automata.get_index_of(name).unwrap();
                    let action_index = self
                        .network
                        .network
                        .actions
                        .get_index_of(&link_pattern.name)
                        .unwrap();
                    automata_active_edges[automaton_index][action_index]
                        .iter()
                        .map(move |active_edge| InstanceTransition {
                            automaton: automaton_index,
                            edge: active_edge,
                            arguments: Box::new([]),
                        })
                })
                .multi_cartesian_product();
            for x in instance_transitions {
                transitions.push(NetworkTransition {
                    instance_transitions: x,
                    result_action: match &link.result {
                        LinkResult::Internal => ResultAction::Internal,
                        LinkResult::Pattern(pattern) => ResultAction::Visible(
                            self.network
                                .network
                                .actions
                                .get_index_of(&pattern.name)
                                .unwrap(),
                            Box::new([]),
                        ),
                    },
                })
            }
        }

        transitions
    }
}

pub struct CompiledActionPattern<'c> {
    pub(crate) action: usize,
    pub(crate) write: Box<[Option<CompiledExpression<'c>>]>,
}

impl<'c> CompiledActionPattern<'c> {
    pub fn apply(&self, ctx: &EvaluationContext) -> Box<[Option<Value>]> {
        self.write
            .iter()
            .map(|expr| expr.as_ref().map(|expr| expr.evaluate(ctx)))
            .collect()
    }
}

pub enum CompiledAction<'c> {
    Internal,
    Pattern(CompiledActionPattern<'c>),
}

impl Action {
    pub fn compile<'c>(&self, network: &Network) -> CompiledAction {
        match self {
            Action::Internal => CompiledAction::Internal,
            Action::Pattern(pattern) => CompiledAction::Pattern(pattern.compile(network)),
        }
    }
}

impl ActionPattern {
    pub fn compile<'c>(&self, network: &Network) -> CompiledActionPattern<'c> {
        let parameter_types = network.actions.get(&self.name).unwrap();
        if parameter_types.len() != self.arguments.len() {
            panic!("Arity of action pattern does not match arity of action type.")
        }
        CompiledActionPattern {
            action: network.actions.get_index_of(&self.name).unwrap(),
            write: self
                .arguments
                .iter()
                .map(|argument| match argument {
                    PatternArgument::Write { value } => Some(value.compile(network)),
                    PatternArgument::Read { identifier: _ } => None,
                })
                .collect(),
        }
    }
}

pub struct CompiledNetwork<'c> {
    pub(crate) network: &'c Network,
    pub(crate) automata: Box<[CompiledAutomaton<'c>]>,
}

impl<'c> CompiledNetwork<'c> {
    pub fn initial_states(&self) -> Box<[CompiledState<'c, '_>]> {
        self.network
            .initial
            .iter()
            .map(|state| CompiledState {
                network: self,
                values: self
                    .network
                    .variables
                    .keys()
                    .map(|identifier| state.values.get(identifier).unwrap().clone())
                    .collect(),
                locations: self
                    .network
                    .automata
                    .iter()
                    .map(|(name, automaton)| {
                        automaton
                            .locations
                            .get_index_of(state.locations.get(name).unwrap())
                            .unwrap()
                    })
                    .collect(),
            })
            .collect()
    }
}

impl Network {
    pub fn compile(&self) -> CompiledNetwork {
        CompiledNetwork {
            network: self,
            automata: self
                .automata
                .values()
                .map(|automaton| automaton.compile(self))
                .collect(),
        }
    }
}