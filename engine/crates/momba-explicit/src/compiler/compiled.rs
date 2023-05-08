//! Compiled model representation.

use std::{fmt::Write, ops::Range};

use momba_model::{
    actions::ActionLabel,
    automata::{AutomatonName, LocationName},
};

use crate::{
    datatypes::idxvec::{new_idx_type, Idx, IdxVec},
    values::{
        layout::Addr,
        memory::{bits::BitSlice, Load, Store},
        types::{IntTy, ValueTyKind},
        FromWord, IntoWord,
    },
    vm::evaluate::Env,
};

use super::{
    assignments::{AssignmentGroup, AssignmentIdx, CompiledAssignment},
    expressions::CompiledExpression,
    CompiledTransientAssignment, CompiledVariables, StateLayout,
};

new_idx_type! {
    /// Uniquely identifies an automaton instance of a compiled model.
    pub InstanceIdx(u16)
}

new_idx_type! {
    /// Uniquely identifies an edge of a compiled instance.
    pub EdgeIdx(u16)
}

new_idx_type! {
    /// Uniquely identifies a location of a compiled instance.
    pub LocationIdx(u32)
}

new_idx_type! {
    /// Uniquely identifies a destination of an edge of a compiled instance.
    pub DestinationIdx(u32)
}

new_idx_type! {
    pub ActionIdx(u16)
}

new_idx_type! {
    pub TransientVariableIdx(u16)
}

/// A compiled model.
pub struct CompiledModel {
    // pub(crate) actions: IdxVec<ActionIdx, CompiledAction>,
    /// The compiled automaton instances of the model.
    pub(crate) instances: CompiledInstances,
    /// The layout of states.
    pub variables: CompiledVariables,
    pub links: CompiledLinks,
}

impl CompiledModel {
    pub fn print_state(&self, state: &BitSlice<StateLayout>) {
        println!("{}", self.fmt_state(state));
    }

    pub fn fmt_state(&self, state: &BitSlice<StateLayout>) -> String {
        let mut buffer = String::new();
        for (field_idx, field) in self.variables.state_layout.fields.indexed_iter() {
            if field_idx.as_usize() > 0 {
                buffer.push_str(", ");
            }
            let addr = self.variables.state_offsets[field_idx];
            buffer
                .write_fmt(format_args!("{} = ", field.name().unwrap()))
                .unwrap();
            match field.ty().kind() {
                ValueTyKind::Bool => buffer
                    .write_fmt(format_args!("{}", bool::from_word(state.load_bool(addr))))
                    .unwrap(),
                ValueTyKind::SignedInt(ty) => buffer
                    .write_fmt(format_args!(
                        "{}",
                        i64::from_word(state.load_signed_int(addr, ty))
                    ))
                    .unwrap(),
                ValueTyKind::UnsignedInt(ty) => buffer
                    .write_fmt(format_args!(
                        "{}",
                        i64::from_word(state.load_unsigned_int(addr, ty))
                    ))
                    .unwrap(),
                ValueTyKind::Float32 => todo!(),
                ValueTyKind::Float64 => todo!(),
                ValueTyKind::Pointer(_) => todo!(),
                ValueTyKind::Slice(_) => todo!(),
                ValueTyKind::Array(_) => todo!(),
                ValueTyKind::Void => todo!(),
            }
        }
        buffer
    }
}

new_idx_type! {
    pub LinkPatternIdx(u16)
}

pub struct CompiledLinks {
    pub links: Vec<CompiledLink>,
    pub patterns: IdxVec<LinkPatternIdx, CompiledLinkPattern>,
}

pub struct CompiledLink {
    pub patterns: Range<LinkPatternIdx>,
}

#[derive(Debug)]
pub struct CompiledLinkPattern {
    pub instance: InstanceIdx,
    pub action: ActionIdx,
}

pub struct CompiledAction {
    pub label: ActionLabel,
}

/// A compiled automaton instance.
pub struct CompiledInstance {
    pub(crate) automaton: AutomatonName,
    /// The locations of the automaton.
    pub(crate) locations: IdxVec<LocationIdx, CompiledLocation>,
    /// The edges of the automaton.
    pub(crate) edges: IdxVec<EdgeIdx, CompiledEdge>,
    /// The destinations of the edges of the automaton.
    pub(crate) destinations: IdxVec<DestinationIdx, CompiledDestination>,
    /// The assignments of the destinations of the edges of the automaton.
    pub(crate) assignments: IdxVec<AssignmentIdx, CompiledAssignment>,
    /// The state field where the location of the automaton is stored.
    pub(crate) location_field: (Addr<StateLayout>, IntTy),
}

pub struct CompiledInstances {
    pub assignment_groups: Vec<AssignmentGroup>,
    pub instances: IdxVec<InstanceIdx, CompiledInstance>,
}

impl CompiledInstance {
    /// The edges which go out ouf the given location.
    #[inline(always)]
    pub fn edges(&self, loc: LocationIdx) -> &[CompiledEdge] {
        &self.edges[self.locations[loc].edges.clone()]
    }

    /// The destinations of the given edge.
    #[inline(always)]
    pub fn destinations(&self, edge: &CompiledEdge) -> &[CompiledDestination] {
        &self.destinations[edge.destinations.clone()]
    }

    /// Loads the current location from the given memory.
    #[inline(always)]
    pub fn load_location(&self, mem: &BitSlice<StateLayout>) -> LocationIdx {
        (i64::from_word(mem.load_unsigned_int(self.location_field.0, &self.location_field.1))
            as usize)
            .into()
    }

    /// Loads the current location from the given memory.
    #[inline(always)]
    pub fn store_location(&self, mem: &mut BitSlice<StateLayout>, idx: LocationIdx) {
        mem.store_unsigned_int(
            self.location_field.0,
            &self.location_field.1,
            (idx.as_usize() as i64).into_word(),
        );
    }

    /// Calls the provided callback for each enabled edge.
    #[inline(always)]
    pub fn foreach_enabled_edge<C: FnMut(EdgeIdx)>(&self, env: &mut Env, mut callback: C) {
        let loc = self.load_location(&env.state);
        let edges_range = self.locations[loc].edges.clone();
        let edges = &self.edges[edges_range.clone()];
        for (idx, edge) in edges.iter().enumerate() {
            if edge.guard.evaluate(env) {
                callback((edges_range.start.as_usize() + idx).into());
            }
        }
    }
}

/// A compiled location of an automaton instance.
pub struct CompiledLocation {
    pub name: LocationName,
    /// The outgoing edges of the location.
    pub(crate) edges: Range<EdgeIdx>,
    pub(crate) transient_assignments: Vec<CompiledTransientAssignment>,
}

/// A compiled edge.
pub struct CompiledEdge {
    pub original_idx: usize,
    pub(crate) action: Option<ActionIdx>,
    /// The compiled guard of the edge.
    pub(crate) guard: CompiledExpression<bool>,
    /// The destinations of the edge.
    pub(crate) destinations: Range<DestinationIdx>,
}

/// A compiled destination.
pub struct CompiledDestination {
    pub(crate) idx: DestinationIdx,
    /// The target location of the destination.
    pub(crate) target: LocationIdx,
    /// The probability of the destination.
    pub(crate) probability: CompiledExpression<f64>,
    /// The assignments of the destination.
    pub(crate) assignments: Range<AssignmentIdx>,
}
