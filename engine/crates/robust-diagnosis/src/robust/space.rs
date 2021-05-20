use std::{
    ops::Deref,
    sync::{Arc, Weak},
};

use hashbrown::HashMap;

use momba_explore::time;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct State<T: time::Time>(Arc<momba_explore::State<T>>);

impl<T: time::Time> Deref for State<T> {
    type Target = momba_explore::State<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Space<T: time::Time> {
    explorer: momba_explore::Explorer<T>,
    initial_states: Vec<State<T>>,
    successors: HashMap<State<T>, Vec<Weak<momba_explore::State<T>>>>,
}

impl<T: time::Time> Space<T> {
    pub fn new(explorer: momba_explore::Explorer<T>) {}
}
