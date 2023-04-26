use std::ops::Range;

use crate::{
    compiler::compiled::{CompiledModel, DestinationIdx, EdgeIdx, InstanceIdx},
    datatypes::idxvec::{new_idx_type, IdxVec},
};

pub mod buffers;
pub mod store;

pub struct Engine {
    model: CompiledModel,
}

impl Engine {
    pub fn initial_states(&self) {
        todo!()
    }

    pub fn transitions(&self, state: &State) {
        todo!()
    }

    pub fn destinations(&self, state: &State, transition: &Transition) {
        todo!()
    }

    pub fn successor(&self, state: &State, transition: &Transition, destination: &Destination) {
        todo!()
    }
}

type State = ();

pub struct Transition {
    items: Range<TransitionItemIdx>,
}

new_idx_type!(pub TransitionItemIdx(u16));

pub struct TransitionItem {
    instance: InstanceIdx,
    edge: EdgeIdx,
}

pub struct TransitionBuffer {
    items: IdxVec<TransitionItemIdx, TransitionItem>,
}

impl TransitionBuffer {
    pub fn items(&self, transition: &Transition) -> &[TransitionItem] {
        &self.items[transition.items.clone()]
    }
}

pub struct DestinationItem {
    instance: InstanceIdx,
    destination: DestinationIdx,
}

new_idx_type!(pub DestinationItemIdx(u16));

pub struct Destination {
    probability: f64,
    items: Range<DestinationItemIdx>,
}

pub struct TransitionsBuffer {}
