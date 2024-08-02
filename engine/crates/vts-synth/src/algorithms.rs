//! Algorithms for constructing and transforming belief transition systems.

use std::{collections::VecDeque, fmt::Debug, hash::Hash};

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use tracing::{info, span, trace, warn, Level};

use crate::{
    lattice::{HasBottom, Join, Meet, MeetSemiLattice},
    ts::{
        algorithms::dfs::Dfs,
        traits::{
            BaseTs, InitialStates, MakeDenseStateMap, MakeSparseStateSet, StateMap, StateSet,
            States, Transitions,
        },
        types::{Vats, Vts, VtsState},
        SparseStateSet, StateId, Transition, Ts, TsBuilder,
    },
};

pub mod minimize;

/// Constructs an _annotation tracking_ VTS from a _verdict-annotated_ TS.
pub fn annotation_tracking<S, A, V>(
    domain: &V,
    vats: &Vats<S, A, V::Element>,
    constraint: &V::Element,
    do_bottom_elimination: bool,
) -> Vts<StateId, V::Element, A>
where
    S: Clone + Eq + Hash,
    A: Clone + Eq + Hash,
    V: MeetSemiLattice + HasBottom,
    V::Element: Clone + Hash + Eq + Meet,
{
    let _span = span!(Level::INFO, "annotation_tracking").entered();

    // Start with an empty VTS and add the initial states to it.
    let mut builder = TsBuilder::new();
    for vats_state in vats.initial_states() {
        let id = builder.insert_state(VtsState::new(vats_state, constraint.clone()));
        builder.mark_initial(id);
    }

    let mut queue = builder.initial_states().collect::<VecDeque<_>>();
    let mut iteration = 0;
    while let Some(source_id) = queue.pop_front() {
        let vats_state = builder.get_label(&source_id).control;
        iteration += 1;
        if iteration % 20_000 == 0 {
            info!(
                "VTS has {} states. {} states in waiting.",
                builder.num_states(),
                queue.len()
            );
        }
        for transition in vats.outgoing(&vats_state) {
            let target_belief = builder
                .get_label(&source_id)
                .verdict
                .meet(&transition.action().guard);

            // Bottom elimination (should not be done for diagnosis).
            if domain.is_bottom(&target_belief) && do_bottom_elimination {
                continue;
            }

            let target = VtsState::new(transition.target(), target_belief);

            let target_id = match builder.lookup_state(&target) {
                Some(id) => id,
                None => {
                    let id = builder.insert_state(target);
                    queue.push_back(id);
                    id
                }
            };

            builder.insert_transition(source_id, transition.action().action.clone(), target_id);
        }
    }

    builder.build()
}

/// Refines the verdicts of a VTS by looking ahead.
pub fn lookahead_refinement<Q, B, L>(original: &Vts<Q, B, L>) -> Vts<Q, B, L>
where
    Q: Clone + Eq + Hash,
    L: Clone + Eq + Hash,
    B: Clone + Hash + Eq + Join,
{
    let mut refined_verdicts =
        original.create_state_labeling(|state| original.get_label(&state).verdict.clone());

    let mut iteration = 0;
    let mut changed = true;
    while changed {
        info!("Starting iteration {iteration}.");
        changed = false;
        let mut visited = 0;
        for state in Dfs::new(&original).iter_post_order() {
            let mut verdicts = Vec::new();
            for transition in original.outgoing(&state) {
                verdicts.push(&refined_verdicts[transition.target()]);
            }

            let verdict = verdicts
                .iter()
                .fold(None, |acc: Option<B>, belief| {
                    Some(
                        acc.map(|acc| acc.join(belief))
                            .unwrap_or_else(|| (*belief).clone()),
                    )
                })
                .unwrap_or_else(|| original.get_label(&state).verdict.clone());

            if refined_verdicts[state] != verdict {
                refined_verdicts[state] = verdict;
                changed = true;
            }
            visited += 1;
            if visited % 100_000 == 0 {
                info!(
                    "Processed {} out of {} states ({} %).",
                    visited,
                    original.num_states(),
                    (visited * 100) / original.num_states(),
                )
            }
        }
        iteration += 1;
    }
    info!("Building refined VTS.");
    original
        .map_states(|id, state| VtsState::new(state.control.clone(), refined_verdicts[id].clone()))
}

/// Determinizes the provided LTS using the usual power set construction.
pub fn determinize<S, A>(original: &Ts<S, A>) -> Ts<SparseStateSet, A>
where
    S: Clone + Eq + Hash,
    A: Clone + Eq + Hash,
{
    let mut builder = TsBuilder::new();

    // 1️⃣ Create the initial state.
    let mut initial_state_set = original.make_sparse_state_set();
    for state in original.initial_states() {
        initial_state_set.insert(state);
    }
    let initial_state = builder.insert_state(initial_state_set);
    builder.mark_initial(initial_state);

    // 2️⃣ Incrementally construct the reachable fragment of the determinized TS.
    let mut stack = vec![initial_state];
    while let Some(source_det) = stack.pop() {
        let source_set = builder.get_label(&source_det);
        // Collect the outgoing transitions for each label.
        let mut outgoing = HashMap::new();
        for source in source_set.iter() {
            for transition in original.outgoing(&source) {
                outgoing
                    .entry(transition.action().clone())
                    .or_insert_with(|| original.make_sparse_state_set())
                    .insert(transition.target());
            }
        }
        // Build the transitions.
        for (act, target_set) in outgoing {
            let target_det = builder.lookup_state(&target_set).unwrap_or_else(|| {
                let target_det = builder.insert_state(target_set);
                stack.push(target_det);
                target_det
            });
            builder.insert_transition(source_det, act, target_det);
        }
    }

    builder.build()
}

/// Turns a TS labeled with sets of states of a VTS into a VTS by joining the verdicts.
pub fn join_verdicts<Q, A, V>(
    vts: &Vts<Q, V, A>,
    ts: &Ts<SparseStateSet, A>,
    bottom: &V,
) -> Vts<SparseStateSet, V, A>
where
    Q: Clone + Eq + Hash,
    A: Clone + Eq + Hash,
    V: Clone + Hash + Eq + Join,
{
    ts.map_states(|_, set| {
        VtsState::new(set.clone(), {
            set.iter()
                .map(|id| vts.get_label(&id))
                .fold(None, |verdict: Option<V>, state| {
                    Some(
                        verdict
                            .map(|acc| acc.join(&state.verdict))
                            .unwrap_or_else(|| state.verdict.clone()),
                    )
                })
                .unwrap_or_else(|| bottom.clone())
        })
    })
}

// /// Uniquely identifies an equivalence class of a partition.
// struct EquivCls(usize);

// pub struct Partition {
//     num_classes: usize,
// }

pub struct Minimize<'cx, Q, V, A> {
    vts: &'cx Vts<Q, V, A>,
    actions: &'cx [A],
    language_insensitive: bool,
    eliminate_self_loops: bool,
}

impl<'cx, Q, V, A> Minimize<'cx, Q, V, A>
where
    V: Clone + Hash + Eq,
    A: Debug + Clone + Hash + Eq,
{
    pub fn new(vts: &'cx Vts<Q, V, A>, actions: &'cx [A]) -> Self {
        Self {
            vts,
            actions,
            language_insensitive: false,
            eliminate_self_loops: false,
        }
    }

    pub fn with_language_insensitive(mut self, language_insensitive: bool) -> Self {
        self.language_insensitive = language_insensitive;
        self.eliminate_self_loops = language_insensitive;
        self
    }

    pub fn with_eliminate_self_loops(mut self, eliminate_self_loops: bool) -> Self {
        self.eliminate_self_loops = eliminate_self_loops;
        self
    }

    /// Minimizes the given `vts` using a variant of Hopcroft's algorithm[^1].
    ///
    /// If `language_insensitive` is set, the language of the original VTS is not preserved.
    ///
    /// [^1]: John Hopcroft. [An `n log n` Algorithm for Minimizing States in Finite Automata](https://doi.org/10.1016/B978-0-12-417750-5.50022-1). 1971.
    pub fn run(self) -> Vts<SparseStateSet, V, A> {
        let _span = span!(Level::INFO, "minimize").entered();

        info!(
            "Minimizing VTS with {} states and {} transitions.",
            self.vts.num_states(),
            self.vts.num_transitions(),
        );

        if self.eliminate_self_loops && !self.language_insensitive {
            warn!("Removing self loops will change the accepted language.");
        }

        let empty_set = im::HashSet::new();

        // 1️⃣ Partition the states based on the verdicts they yield.
        let mut partition = HashMap::new();
        for state_id in self.vts.states() {
            let verdict = self.vts.get_label(&state_id).verdict.clone();
            partition
                .entry(verdict)
                .or_insert_with(|| empty_set.clone())
                .insert(state_id);
        }
        let mut partition = partition.into_values().collect::<HashSet<_>>();
        debug_assert_eq!(
            partition.iter().map(|cls| cls.len()).sum::<usize>(),
            self.vts.num_states(),
            "No states of the original VTS must get lost."
        );
        info!(
            "Initial partition has {} equivalence classes.",
            partition.len()
        );
        info!("Refining equivalence classes.");

        // 2️⃣ Refine the partition according to Hopcroft's algorithm.
        let mut work_set = partition.iter().cloned().collect::<IndexSet<_>>();
        let mut iteration = 0;
        while let Some(splitter_cls) = work_set.pop() {
            iteration += 1;
            if iteration % 10 == 0 {
                info!(
                    "Partition has {} equivalence classes, {} splitters are waiting.",
                    partition.len(),
                    work_set.len()
                );
            }
            for action in self.actions {
                let splitter = Splitter::new(splitter_cls.clone(), action);
                for cls in partition.clone() {
                    let Some((inside, outside)) =
                        split(self.vts, &cls, &splitter, self.language_insensitive)
                    else {
                        // The class is not split by the splitter. Continue!
                        continue;
                    };
                    // The class must be split. Remove it from the partition and update the work
                    // set based on the returned split.
                    partition.remove(&cls);
                    // for action in self.actions {
                    // let splitter = Splitter::new(cls.clone(), action);
                    if work_set.swap_remove(&splitter_cls) {
                        work_set.insert(inside.clone());
                        work_set.insert(outside.clone());
                    } else {
                        if inside.len() < outside.len() {
                            work_set.insert(inside.clone());
                        } else {
                            work_set.insert(outside.clone());
                        }
                    }
                    // }
                    partition.insert(inside);
                    partition.insert(outside);
                }
            }
        }
        info!("Found {} equivalence classes.", partition.len());
        assert_eq!(
            partition.iter().map(|cls| cls.len()).sum::<usize>(),
            self.vts.num_states(),
            "No states of the original VTS must get lost."
        );

        // 3️⃣ Build a lookup tables for equivalence classes.
        let mut state_to_cls = self.vts.make_dense_state_map();
        let mut cls_to_verdict = Vec::new();
        let mut cls_to_set = Vec::new();
        for (cls_idx, cls) in partition.iter().enumerate() {
            let mut cls_set = self.vts.make_sparse_state_set();
            for state in cls {
                cls_set.insert(*state);
                state_to_cls.insert(*state, cls_idx);
                // TODO: Check whether there was no class for the state before.
                // debug_assert!(inserted.is_none(), "A state be in a single class.");
            }
            cls_to_set.push(cls_set);
            let cls_verdict = self
                .vts
                .get_label(
                    cls.iter()
                        .next()
                        .expect("Equivalence classes must not be empty"),
                )
                .verdict
                .clone();
            cls_to_verdict.push(cls_verdict);
        }
        // TODO: Check whether any states got lost.
        // assert_eq!(
        //     state_to_cls.len(),
        //     vts.num_states(),
        //     "No states of the original VTS must get lost."
        // );

        // 3️⃣ Build the minimized VTS using the computed classes.
        info!("Constructing minimized VTS using the computed classes.");
        let mut builder = TsBuilder::new();
        // Add the states to the new VTS.
        let cls_states = partition
            .iter()
            .enumerate()
            .map(|(cls_idx, cls)| {
                let cls_set = cls_to_set[cls_idx].clone();
                let verdict = cls_to_verdict[cls_idx].clone();
                let cls_state = builder.insert_state(VtsState::new(cls_set, verdict));
                for state in cls {
                    if self.vts.is_initial(state) {
                        builder.mark_initial(cls_state);
                        break;
                    }
                }
                cls_state
            })
            .collect::<Vec<_>>();
        // Add the transitions to the new VTS.
        for (source_cls_idx, source) in cls_states.iter().enumerate() {
            for state in cls_to_set[source_cls_idx].iter() {
                for transition in self.vts.outgoing(&state) {
                    let target_cls_idx = *state_to_cls
                        .get(&transition.target())
                        .expect("Every state must belong to a class.");
                    if !self.eliminate_self_loops || source_cls_idx != target_cls_idx {
                        let target = cls_states[target_cls_idx];
                        builder.insert_transition(*source, transition.action().clone(), target);
                    }
                }
            }
        }

        builder.build()
    }
}

/// A splitter for minimization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Splitter<'act, A> {
    cls: im::HashSet<StateId>,
    action: &'act A,
}

impl<'act, A> Splitter<'act, A> {
    /// Constructs a new [`Splitter`].
    pub fn new(cls: im::HashSet<StateId>, action: &'act A) -> Self {
        Self { cls, action }
    }
}

/// Tries to split an equivalence class `cls` with a [`Splitter`].
fn split<Q, V, A>(
    vts: &Vts<Q, V, A>,
    cls: &im::HashSet<StateId>,
    splitter: &Splitter<'_, A>,
    language_insensitive: bool,
) -> Option<(im::HashSet<StateId>, im::HashSet<StateId>)>
where
    A: Eq,
{
    #[derive(Debug, Clone, Copy)]
    enum Hit {
        /// Transition hits inside of the splitter's class.
        Inside,
        /// Transition hits outside of the splitter's class.
        Outside,
        /// There is no transition with the splitter's action.
        Disabled,
    }

    // 1️⃣ Determine a `Hit` for each of the states of the class.
    let default_hit = if language_insensitive {
        Hit::Disabled
    } else {
        Hit::Outside
    };
    let mut hits = Vec::new();
    let mut hits_inside = 0;
    let mut hits_outside = 0;
    for state in cls {
        let mut hit = default_hit;
        for transition in vts.outgoing(state) {
            if transition.action() == splitter.action {
                if splitter.cls.contains(&transition.target()) {
                    hit = Hit::Inside;
                } else {
                    hit = Hit::Outside;
                }
                // The TS is assumed to be deterministic, hence, there is no other
                // transition with the action of the splitter. Stop searching!
                break;
            }
        }
        hits.push(hit);
        match hit {
            Hit::Inside => hits_inside += 1,
            Hit::Outside => hits_outside += 1,
            Hit::Disabled => { /* ignore */ }
        };
    }
    debug_assert_eq!(
        hits.len(),
        cls.len(),
        "There must be a `Hit` for each state."
    );
    trace!(
        inside = hits_inside,
        outside = hits_outside,
        none = cls.len() - hits_inside - hits_outside,
    );

    // 2️⃣ Decide whether to split based on the hit counters.
    if hits_inside > 0 && hits_outside > 0 {
        // The class needs to be split as some transitions hit inside and some
        // outside of the splitter's class. We use the previously collected hits
        // to decide which states go where.
        let mut inside = cls.new_from();
        let mut outside = cls.new_from();
        cls.iter().zip(hits).for_each(|(state, hit)| {
            match hit {
                Hit::Inside => inside.insert(*state),
                Hit::Outside | Hit::Disabled => outside.insert(*state),
            };
        });
        debug_assert_eq!(
            cls.len(),
            inside.len() + outside.len(),
            "No states of the original class must get lost."
        );
        Some((inside, outside))
    } else {
        None
    }
}

/// Computes the observability projection of a TS.
pub fn observability_projection<S, A, F>(ts: &Ts<S, A>, is_observable: F) -> Ts<SparseStateSet, A>
where
    S: Debug + Clone + Eq + Hash,
    A: Debug + Clone + Eq + Hash,
    F: Fn(&A) -> bool,
{
    // 1️⃣ For each state, determine all outgoing transitions and indiscernible states.
    let mut outgoing = ts.create_default_state_labeling::<im::HashSet<_>>();
    let mut indiscernible = ts.create_state_labeling(|id| im::HashSet::<StateId>::from_iter([id]));

    let mut iteration = 0;
    let mut changed = true;
    while changed {
        info!("Starting iteration {iteration}.");
        changed = false;
        let mut visited = 0;
        for state in Dfs::new(ts).iter_post_order() {
            for transition in ts.outgoing(&state) {
                if is_observable(transition.action()) {
                    changed |= outgoing[state]
                        .insert(TransitionParts::from_transition(transition))
                        .is_none();
                } else {
                    let target_outgoing = outgoing[transition.target()].clone();
                    for transition in target_outgoing {
                        changed |= outgoing[state].insert(transition).is_none();
                    }
                    let target_indiscernible = indiscernible[transition.target()].clone();
                    for x in target_indiscernible {
                        changed |= indiscernible[state].insert(x).is_none();
                    }
                }
            }
            visited += 1;
            if visited % 100_000 == 0 {
                info!(
                    "Processed {} out of {} states ({} %).",
                    visited,
                    ts.num_states(),
                    (visited * 100) / ts.num_states(),
                )
            }
        }
        iteration += 1;
    }

    info!("Building VTS.");

    // 2️⃣ Construct the resulting TS.
    let mut builder = TsBuilder::new();

    for state_id in ts.states() {
        let mut reachable_states = ts.make_sparse_state_set();
        reachable_states.insert(state_id);
        for other_id in &indiscernible[state_id] {
            reachable_states.insert(*other_id);
        }
        let new_state_id = builder.insert_state(reachable_states);
        if ts.is_initial(&state_id) {
            builder.mark_initial(new_state_id);
        }
    }

    for state_id in ts.states() {
        let mut source_states = ts.make_sparse_state_set();
        for weak_reach in &indiscernible[state_id] {
            source_states.insert(*weak_reach);
        }
        let source_id = builder.lookup_state(&source_states).unwrap();

        for transition in &outgoing[state_id] {
            let mut target_states = ts.make_sparse_state_set();
            for weak_reach in &indiscernible[transition.target] {
                target_states.insert(*weak_reach);
            }
            let target_id = builder.lookup_state(&target_states).unwrap();

            builder.insert_transition(source_id.clone(), transition.action.clone(), target_id)
        }
    }

    builder.build()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TransitionParts<A> {
    action: A,
    target: StateId,
}

impl<A> TransitionParts<A> {
    pub fn new(act: A, target: StateId) -> Self {
        Self {
            action: act,
            target,
        }
    }

    pub fn from_transition(transition: &Transition<A>) -> Self
    where
        A: Clone,
    {
        Self::new(transition.action().clone(), transition.target())
    }
}
