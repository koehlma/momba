//! General algorithms for transition systems.

use super::{traits::*, transposed::Transposed};

pub mod bfs;
pub mod dfs;

/// Computes the _strongly-connected components_ (SCCs) of a TS.
///
/// Implements [Kosaraju's algorithm](https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm).
pub fn scc<TS: TsRef + dfs::SupportsDfs + States + Predecessors>(
    ts: TS,
) -> impl Iterator<Item = Vec<TS::StateId>> {
    let mut stack = Vec::new();

    let mut dfs = dfs::Dfs::empty(ts);
    for state in ts.states() {
        if dfs.push(state) {
            for state in dfs.iter_post_order() {
                stack.push(state);
            }
        }
    }

    let mut dfs = dfs::Dfs::empty(Transposed::new(ts));

    std::iter::from_fn(move || {
        while let Some(state) = stack.pop() {
            if dfs.push(state.clone()) {
                let mut scc = vec![state];
                for state in dfs.iter_pre_order() {
                    scc.push(state);
                }
                return Some(scc);
            }
        }
        None
    })
}
