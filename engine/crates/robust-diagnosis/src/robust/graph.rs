//! Implementation of the *diagnosis search graph*.

use std::rc::{Rc, Weak};

use hashbrown::{HashMap, HashSet};

use super::Observer;

type Explorer = momba_explore::Explorer<momba_explore::time::Float64Zone>;
type State = momba_explore::State<momba_explore::time::Float64Zone>;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct HistoryItem {
    pub observation: usize,
    pub clock: clock_zones::Variable,
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DiagnosisState {
    pub state: State,
    pub faults: im::HashSet<momba_explore::Action>,
    pub history: Vec<HistoryItem>,
}

pub struct Graph<'d> {
    pub observer: &'d Observer,
    pub explorer: &'d Explorer,
    pub observables: &'d HashSet<usize>,
    pub successors: HashMap<Rc<DiagnosisState>, im::HashSet<Rc<DiagnosisState>>>,
}

impl<'d> Graph<'d> {
    pub fn new(
        observer: &'d Observer,
        explorer: &'d Explorer,
        observables: &'d HashSet<usize>,
    ) -> Self {
        Self {
            observer,
            explorer,
            observables,
            successors: hashbrown::HashMap::new(),
        }
    }

    pub fn explore(&mut self, state: Rc<DiagnosisState>) -> &im::HashSet<Rc<DiagnosisState>> {
        match self.successors.get(&state) {
            Some(result) => result,
            None => {
                todo!()
            }
        }
    }
}
