//! Buffers for the exploration engine.

use std::ops::Range;

use crate::{
    compiler::compiled::{EdgeIdx, InstanceIdx},
    datatypes::idxvec::IdxVec,
};

pub struct EnabledEdgesBuffer {
    instance_ranges: IdxVec<InstanceIdx, Range<usize>>,
    enabled_edges: Vec<EdgeIdx>,
}

impl EnabledEdgesBuffer {
    pub fn enabled_edges(&self, instance: InstanceIdx) -> &[EdgeIdx] {
        &self.enabled_edges[self.instance_ranges[instance].clone()]
    }

    pub fn clear(&mut self) {
        self.instance_ranges.clear();
        self.enabled_edges.clear();
    }

    // pub fn fill(&mut self, model: &CompiledModel, env: &mut Env) {
    //     self.clear();

    //     // self.instance_ranges
    //     //     .extend(model.instances.iter().map(|instance| {
    //     //         let start = self.enabled_edges.len();
    //     //         instance.foreach_enabled_edge(env, |edge| self.enabled_edges.push(edge));
    //     //         let end = self.enabled_edges.len();
    //     //         start..end
    //     //     }))
    //     todo!()
    // }
}
