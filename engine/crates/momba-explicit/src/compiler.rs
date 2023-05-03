//! Compiler for transforming a model into an efficiently executable representation.

use std::{hash::Hash, iter};

use crate::{
    compiler::{compiled::CompiledLinkPattern, scope::Scope},
    datatypes::idxvec::{new_idx_type, Idx, IdxVec},
    params::Params,
    values::{
        layout::{DenseBitLayout, Field, FieldIdx, FieldOffsets, StructLayout},
        memory::{bits::BitSlice, Store},
        types::{HasType, TypeCtx, TypeError, ValueTy, ValueTyKind, WordTy},
        FromWord, IntoWord, Word,
    },
};

use momba_model::{
    actions::{ActionLabel, ActionPattern},
    automata::{Automaton, AutomatonName, Edge, LocationName},
    expressions::{ConstantExpression, Expression},
    models::Model,
    types::Type,
    values::Value,
    variables::VariableDeclaration,
};

use parking_lot::Mutex;
use thiserror::Error;

use self::{
    assignments::compile_assignment,
    compiled::{
        ActionIdx, CompiledDestination, CompiledEdge, CompiledInstance, CompiledLink,
        CompiledLinks, CompiledLocation, CompiledModel, InstanceIdx, LocationIdx,
        TransientVariableIdx,
    },
    expressions::{compile_expression, Constant},
    scope::ScopeItem,
};

pub mod assignments;
pub mod compiled;
pub mod expressions;
pub mod functions;
pub mod scope;

/// Compiles a model with the provided parameters and options.
pub fn compile_model(
    model: &Model,
    params: &Params,
    options: &Options,
) -> CompileResult<CompiledModel> {
    Ctx::new(model, params, options).compile()
}

/// Compilation options.
#[derive(Debug, Clone)]
pub struct Options {}

impl Options {
    /// Creates [`Options`] with the default options.
    pub fn new() -> Self {
        Self {}
    }

    // /// Sets the type used for storing reals.
    // pub fn with_real_type(mut self, ty: ValueTy) -> Self {
    //     self.real_type = ty;
    //     self
    // }
}

/// The result type used by the compiler.
pub type CompileResult<T> = Result<T, CompileError>;

/// A compilation error.
#[derive(Debug, Error, Clone)]
#[error("{0}")]
pub struct CompileError(String);

impl CompileError {
    /// Creates a new compilation error.
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<TypeError> for CompileError {
    fn from(value: TypeError) -> Self {
        Self(format!("Type Error: {}", value))
    }
}

/// Auxiliary macro for construction and returning a compilation error.
macro_rules! return_error {
    ($msg:literal $($args:tt)*) => {
        return Err(crate::compiler::CompileError::new(format!($msg $($args)*)))
    };
}

pub(crate) use return_error;

pub(crate) fn clone_result<T>(result: &CompileResult<T>) -> CompileResult<&T> {
    match result {
        Ok(value) => Ok(value),
        Err(err) => Err(err.clone()),
    }
}

/// A cached value of a compilation query.
///
/// Works like a `OnceCell` but panics on recursive use.
pub struct Cached<T>(Mutex<CachedState<T>>);

impl<T> Cached<T> {
    pub fn get_or_init<F: FnOnce() -> T>(&self, fun: F) -> &T {
        let mut state = self.0.lock();
        match &*state {
            CachedState::Value(value) => {
                // SAFETY: We never overwrite the value.
                unsafe { &*(value as *const T) }
            }
            CachedState::Empty => {
                *state = CachedState::Computing;
                drop(state);
                let value = fun();
                let mut state = self.0.lock();
                *state = CachedState::Value(value);
                match &*state {
                    CachedState::Value(value) => {
                        // SAFETY: We never overwrite the value.
                        unsafe { &*(value as *const T) }
                    }
                    _ => {
                        unreachable!("We just set the value.");
                    }
                }
            }
            CachedState::Computing => {
                panic!("Recursive queries are not allowed.");
            }
        }
    }

    pub fn set(&self, value: T) {
        let mut state = self.0.lock();
        match &*state {
            CachedState::Empty => *state = CachedState::Value(value),
            CachedState::Computing => panic!("Value is currently being computed."),
            CachedState::Value(_) => panic!("Value has already been set."),
        }
    }

    pub fn into_inner(self) -> T {
        match self.0.into_inner() {
            CachedState::Empty | CachedState::Computing => panic!("Cache has not been populated."),
            CachedState::Value(value) => value,
        }
    }
}

impl<T> Default for Cached<T> {
    fn default() -> Self {
        Self(Mutex::new(CachedState::Empty))
    }
}

/// The state of a cached value.
enum CachedState<T> {
    Empty,
    Computing,
    Value(T),
}

/// Query cache of the root compilation context.
#[derive(Default)]
struct CtxQueryCache {
    constants: Cached<CompileResult<CompiledConstants>>,
    variables: Cached<CompileResult<CompiledVariables>>,
    instances: Cached<IdxVec<InstanceIdx, CompiledInstance>>,
}

pub(crate) struct CompiledConstants {
    scope: Scope,
    values: IdxVec<ConstantIdx, Cached<CompileResult<Constant>>>,
}

pub type StateLayout = DenseBitLayout;

pub struct CompiledVariables {
    pub(crate) global_scope: Scope,
    pub(crate) instance_scopes: IdxVec<InstanceIdx, Scope>,
    pub(crate) instance_locations: IdxVec<InstanceIdx, FieldIdx>,
    pub state_layout: StructLayout,
    pub(crate) state_offsets: FieldOffsets<StateLayout>,
    pub initial_values: IdxVec<FieldIdx, Word>,
    pub initial_state: Vec<u8>,
}

/// The root compilation context.
pub(crate) struct Ctx<'cx> {
    model: &'cx Model,
    options: &'cx Options,
    params: &'cx Params,
    pub(crate) tcx: TypeCtx,
    cache: CtxQueryCache,
}

impl<'cx> Ctx<'cx> {
    fn new(model: &'cx Model, params: &'cx Params, options: &'cx Options) -> Self {
        Self {
            model,
            options,
            params,
            tcx: TypeCtx::new(),
            cache: CtxQueryCache::default(),
        }
    }

    pub(crate) fn query_constants(&self) -> CompileResult<&CompiledConstants> {
        clone_result(self.cache.constants.get_or_init(|| {
            let values: IdxVec<_, _> = iter::repeat_with(|| Cached::default())
                .take(self.model.constants.len())
                .collect::<Vec<_>>()
                .into();
            let mut scope = Scope::new();
            for (idx, decl) in self.model.constants.iter().enumerate() {
                let idx = ConstantIdx(idx);
                scope.insert(decl.name.clone(), ScopeItem::constant(idx));
                if let Some(value) = self.params.get(&decl.name) {
                    values[idx].set(Ok(match (&decl.typ, value) {
                        (Type::Int(_), Value::Int(value)) => {
                            Constant::from_parts(*value, self.tcx.word_int())
                        }
                        (Type::Real(_), Value::Int(value)) => {
                            Constant::from_parts(*value as f64, self.tcx.word_float())
                        }
                        (Type::Real(_), Value::Float(value)) => {
                            Constant::from_parts(*value, self.tcx.word_float())
                        }
                        (Type::Bool, Value::Bool(value)) => {
                            Constant::from_parts(*value, self.tcx.word_bool())
                        }
                        _ => {
                            return_error!(
                                "Invalid constant value {value:?} for constant of type {:?}.",
                                &decl.typ
                            )
                        }
                    }))
                }
            }
            Ok(CompiledConstants { scope, values })
        }))
    }

    pub(crate) fn query_constant_scope(&self) -> CompileResult<&Scope> {
        self.query_constants().map(|constants| &constants.scope)
    }

    pub(crate) fn query_constant_value(&self, constant: ConstantIdx) -> CompileResult<Constant> {
        let constants = self.query_constants()?;
        constants.values[constant]
            .get_or_init(|| {
                let decl = &self.model.constants[constant.as_usize()];
                match &decl.default {
                    Some(default) => {
                        let ty = self.tcx.type_to_expr_type(&decl.typ)?;
                        self.eval_const_expr(default, &ty)
                    }
                    None => {
                        return_error!("No value for constant.");
                    }
                }
            })
            .clone()
    }

    pub(crate) fn compute_word_ty(&self, ty: &Type) -> CompileResult<WordTy> {
        Ok(match ty {
            Type::Int(_) => self.tcx.word_int(),
            Type::Real(_) => self.tcx.word_float(),
            Type::Bool => self.tcx.word_bool(),
            Type::Clock => {
                return_error!("Clock variables are not supported yet.");
            }
            Type::Array(typ) => {
                return_error!("Arrays are not supported yet.");
            }
        })
    }

    pub(crate) fn compute_variable_type(
        &self,
        decl: &VariableDeclaration,
    ) -> CompileResult<ValueTy> {
        Ok(match &decl.typ {
            Type::Int(typ) => {
                let lower_bound = self.eval_bound(typ.lower_bound.as_ref())?;
                let upper_bound = self.eval_bound(typ.upper_bound.as_ref())?;
                self.tcx.value_bounded_int(lower_bound, upper_bound)?
            }
            Type::Real(_) => self.tcx.value_float64(),
            Type::Bool => self.tcx.value_bool(),
            Type::Clock => {
                return_error!("Clock variables are not supported yet.");
            }
            Type::Array(_) => {
                return_error!("Arrays are not supported yet.");
            }
        })
    }

    pub fn eval_const_expr(&self, expr: &Expression, ty: &WordTy) -> CompileResult<Constant> {
        compile_expression(&self, self.query_constant_scope()?, expr)?
            .coerce_to(&self.tcx, ty)?
            .evaluate_const()
            .ok_or_else(|| CompileError::new("Expression does not have a constant value!"))
    }

    pub fn eval_bound<T: HasType + FromWord>(
        &self,
        bound: Option<&Expression>,
    ) -> CompileResult<Option<T>> {
        Ok(bound
            .map(|bound| self.eval_const_expr(&bound, &T::word_ty(&self.tcx)))
            .transpose()?
            .map(|constant| T::from_word(constant.value())))
    }

    pub fn query_variables(&self) -> CompileResult<&CompiledVariables> {
        clone_result(self.cache.variables.get_or_init(|| {
            let mut state_layout = StructLayout::new();
            let mut global_scope = self.query_constant_scope()?.create_child();

            let mut initial_values = Vec::new();

            for (idx, decl) in self.model.globals.iter().enumerate() {
                if decl.transient.unwrap_or(false) {
                    return_error!("Transient variables are not supported!");
                }
                let ty = self.compute_variable_type(decl)?;

                let Some(initial) = &decl.default else {
                    return_error!("Variables must have initial values");
                };

                initial_values.push(
                    self.eval_const_expr(initial, &self.tcx.loaded_value_ty(&ty)?)?
                        .value(),
                );

                let field_idx = state_layout
                    .add_field(Field::new(ty.clone()).with_name(format!("globals.{}", decl.name)));
                global_scope.insert(decl.name.clone(), ScopeItem::StateVariable(field_idx));
            }

            let mut instance_locations = Vec::new();

            for (idx, instance) in self.model.instances.iter().enumerate() {
                let automaton = self
                    .model
                    .automata
                    .iter()
                    .find(|automaton| automaton.name == instance.automaton)
                    .unwrap();

                let loc_value_ty = self
                    .tcx
                    .value_bounded_int(Some(0), Some((automaton.locations.len() - 1) as i64))?;

                let mut initial_location = None;
                for (idx, location) in automaton.locations.iter().enumerate() {
                    if location.initial {
                        initial_location = Some(idx);
                        break;
                    }
                }

                let Some(initial_location) = initial_location else {
                    return_error!("Instance has no initial location");
                };

                initial_values.push((initial_location as i64).into_word());

                instance_locations.push(
                    state_layout.add_field(
                        Field::new(loc_value_ty).with_name(format!("locations[{}]", idx)),
                    ),
                );
            }

            let mut instance_scopes = Vec::new();

            for (idx, instance) in self.model.instances.iter().enumerate() {
                let automaton = self
                    .model
                    .automata
                    .iter()
                    .find(|automaton| automaton.name == instance.automaton)
                    .unwrap();

                let mut instance_scope = global_scope.create_child();
                for decl in automaton.locals.iter() {
                    if decl.transient.unwrap_or(false) {
                        return_error!("Transient variables are not supported!");
                    }
                    let ty = self.compute_variable_type(decl)?;
                    let field_idx = state_layout.add_field(
                        Field::new(ty.clone()).with_name(format!("locals[{}].{}", idx, decl.name)),
                    );
                    let Some(initial) = &decl.default else {
                        return_error!("Variables must have initial values");
                    };

                    initial_values.push(
                        self.eval_const_expr(initial, &self.tcx.loaded_value_ty(&ty)?)?
                            .value(),
                    );

                    instance_scope.insert(decl.name.clone(), ScopeItem::StateVariable(field_idx))
                }
                instance_scopes.push(instance_scope);
            }

            let state_offsets = state_layout.field_offsets::<StateLayout>();

            let state_size = state_layout.size::<StateLayout>().to_bytes();

            let mut initial_state = vec![0u8; usize::from(state_size)];

            let mut_initial_state = BitSlice::<StateLayout>::from_slice_mut(&mut initial_state);

            for ((field_idx, field), value) in state_layout
                .fields
                .indexed_iter()
                .zip(initial_values.iter())
            {
                mut_initial_state.store(state_offsets[field_idx], field.ty(), *value)
            }

            Ok(CompiledVariables {
                global_scope,
                instance_scopes: instance_scopes.into(),
                instance_locations: instance_locations.into(),
                state_layout,
                state_offsets,
                initial_values: initial_values.into(),
                initial_state,
            })
        }))
    }

    fn compile_instances(&self) -> CompileResult<IdxVec<InstanceIdx, CompiledInstance>> {
        let mut instances = Vec::new();

        let variables = self.query_variables()?;

        let default_probability = Expression::Constant(ConstantExpression {
            value: Value::Float(1.0),
        });
        let default_guard = Expression::Constant(ConstantExpression {
            value: Value::Bool(true),
        });

        for (idx, instance) in self.model.instances.iter().enumerate() {
            let instance_idx = InstanceIdx::from(idx);
            let instance_scope = &variables.instance_scopes[instance_idx];

            let automaton = self.get_automaton(&instance.automaton)?;

            println!(
                "Compile Instance ({}): {:?}",
                instance_idx.as_usize(),
                automaton.name
            );

            let mut locations = IdxVec::new();
            let mut edges = IdxVec::new();
            let mut destinations = IdxVec::new();
            let mut assignments = IdxVec::new();

            for location in automaton.locations.iter() {
                //let location_idx = LocationIdx::from(idx);
                let edges_start = edges.next_idx();
                for (original_idx, edge) in iter_outgoing_edges(automaton, &location.name) {
                    let edge_idx = edges.next_idx();
                    let destinations_start = destinations.next_idx();
                    for destination in edge.destinations.iter() {
                        let destination_idx = destinations.next_idx();
                        let target = get_location_idx(automaton, &destination.target)?;

                        let assignments_start = assignments.next_idx();
                        for assignment in destination.assignments.iter() {
                            assignments.push(compile_assignment(
                                self,
                                instance_scope,
                                assignment,
                                instance_idx,
                                edge_idx,
                                destination_idx,
                            )?);
                        }
                        let assignments_end = assignments.next_idx();

                        destinations.push(CompiledDestination {
                            idx: destination_idx,
                            target,
                            probability: compile_expression(
                                self,
                                instance_scope,
                                destination
                                    .probability
                                    .as_ref()
                                    .unwrap_or(&default_probability),
                            )?
                            .cast(&self.tcx)?,
                            assignments: assignments_start..assignments_end,
                        })
                    }
                    let destinations_end = destinations.next_idx();
                    let action = match &edge.action {
                        ActionPattern::Silent => None,
                        ActionPattern::Labeled(labeled) => {
                            Some(self.get_action_idx(&labeled.label)?)
                        }
                    };
                    edges.push(CompiledEdge {
                        original_idx,
                        action,
                        guard: compile_expression(
                            self,
                            instance_scope,
                            edge.guard.as_ref().unwrap_or(&default_guard),
                        )?
                        .cast(&self.tcx)?,
                        destinations: destinations_start..destinations_end,
                    })
                }
                let edges_end = edges.next_idx();
                locations.push(CompiledLocation {
                    name: location.name.clone(),
                    edges: edges_start..edges_end,
                })
            }

            let location_field_idx = variables.instance_locations[instance_idx];
            let location_field = &variables.state_layout[location_field_idx];
            let location_addr = variables.state_offsets[location_field_idx];

            let ValueTyKind::UnsignedInt(int_ty) = location_field.ty().kind() else {
                panic!("State location type should be unsigned int!")
            };

            instances.push(CompiledInstance {
                automaton: automaton.name.clone(),
                locations,
                edges,
                destinations,
                location_field: (location_addr, int_ty.clone()),
                assignments,
            });
        }
        Ok(instances.into())
    }

    fn compile_links(&self) -> CompileResult<CompiledLinks> {
        let mut patterns = IdxVec::new();
        let mut links = Vec::new();

        for link in &self.model.links {
            let patterns_start = patterns.next_idx();

            for (instance_idx, pattern) in &link.vector {
                let action = self.get_action_idx(&pattern.label)?;
                assert!(pattern.arguments.is_empty());
                patterns.push(CompiledLinkPattern {
                    instance: instance_idx.as_usize().into(),
                    action,
                })
            }

            links.push(CompiledLink {
                patterns: patterns_start..patterns.next_idx(),
            })
        }

        Ok(CompiledLinks { links, patterns })
    }

    fn compile(self) -> CompileResult<CompiledModel> {
        let _ = self.query_variables()?;
        let instances = self.compile_instances()?;
        let links = self.compile_links()?;
        Ok(CompiledModel {
            links,
            instances,
            variables: self.cache.variables.into_inner().unwrap(),
        })
    }

    fn get_automaton(&self, name: &AutomatonName) -> CompileResult<&'cx Automaton> {
        self.model
            .automata
            .iter()
            .find(|automaton| automaton.name == *name)
            .ok_or_else(|| CompileError::new(format!("Unable to find automaton `{:?}`.", name)))
    }

    fn get_action_idx(&self, label: &ActionLabel) -> CompileResult<ActionIdx> {
        self.model
            .actions
            .iter()
            .position(|decl| decl.label == *label)
            .ok_or_else(|| CompileError::new(format!("Unable to find action `{:?}`.", label)))
            .map(Into::into)
    }
}

fn get_location_idx(automaton: &Automaton, name: &LocationName) -> CompileResult<LocationIdx> {
    automaton
        .locations
        .iter()
        .position(|automaton| automaton.name == *name)
        .ok_or_else(|| CompileError::new(format!("Unable to find automaton `{:?}`.", name)))
        .map(Into::into)
}

fn iter_outgoing_edges<'model: 'iter, 'location: 'iter, 'iter>(
    automaton: &'model Automaton,
    location: &'location LocationName,
) -> impl 'iter + Iterator<Item = (usize, &'model Edge)> {
    automaton
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.source == *location)
}

new_idx_type! {
    /// Uniquely identifies a constant in a model.
    pub ConstantIdx(usize)
}
