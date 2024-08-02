//! Output in [Graphviz's](https://graphviz.org) `dot` format.

use std::{
    fs,
    hash::Hash,
    io::{self, Write},
    path::Path,
};

use hashbrown::HashMap;

use crate::ts::{traits::*, StateId, Transition, Ts};

pub fn dump_ts<S, L, W>(
    ts: &Ts<S, L>,
    title: &str,
    mut out: W,
    state_label: impl Fn(StateId, &S) -> String,
    transition_attrs: impl Fn(&Transition<L>) -> TransitionAttributes,
) -> Result<(), io::Error>
where
    S: Clone + Eq + Hash,
    L: Clone + Eq + Hash,
    W: Write,
{
    writeln!(out, "digraph Lts {{")?;
    writeln!(out, "  label=\"{}\"", title)?;
    let mut state_ids = HashMap::new();
    for state in ts.states() {
        let state_id = format!("s{}", state_ids.len());
        let is_initial = ts.is_initial(&state);
        let shape = if is_initial {
            ",shape=rect"
        } else {
            ",shape=ellipse"
        };
        writeln!(
            out,
            "  {}[label=\"{}\" {}]",
            state_id,
            state_label(state, ts.get_label(&state)),
            shape
        )?;
        state_ids.insert(state, state_id);
    }
    for transition in ts.transitions() {
        let source_id = state_ids.get(&transition.source).unwrap();
        let target_id = state_ids.get(&transition.target).unwrap();
        writeln!(
            out,
            "  {} -> {} [{}]",
            source_id,
            target_id,
            transition_attrs(transition).to_string()
        )?;
    }
    // for source in ts.states() {
    //     for target in ts.predecessors(&source) {
    //         let source_id = state_ids.get(&source).unwrap();
    //         let target_id = state_ids.get(&target).unwrap();
    //         writeln!(out, "  {} -> {} [color=red]", source_id, target_id,)?;
    //     }
    // }
    writeln!(out, "}}")?;
    Ok(())
}

pub struct TransitionAttributes(HashMap<String, String>);

impl TransitionAttributes {
    pub fn new(label: &str) -> Self {
        Self(HashMap::from([(
            "label".to_owned(),
            format!("\"{}\"", label),
        )]))
    }

    pub fn with_color(mut self, color: String) -> Self {
        self.0.insert("color".to_owned(), color);
        self
    }

    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|(key, value)| format!("{}={}", key, value))
            .collect::<Vec<_>>()
            .join(",")
    }
}

pub fn write_to_file<S, L>(
    lts: &Ts<S, L>,
    title: &str,
    path: &Path,
    state_label: impl Fn(StateId, &S) -> String,
    transition_attrs: impl Fn(&Transition<L>) -> TransitionAttributes,
) -> Result<(), io::Error>
where
    S: Clone + Eq + Hash,
    L: Clone + Eq + Hash,
{
    let out = fs::File::create(path)?;
    dump_ts(
        lts,
        title,
        io::BufWriter::new(out),
        state_label,
        transition_attrs,
    )
}
