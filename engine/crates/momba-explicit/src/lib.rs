#![doc = include_str!("../README.md")]

use core::num;
use std::{
    collections::VecDeque,
    error::Error,
    marker::PhantomData,
    ops::Range,
    ptr::NonNull,
    time::{Duration, Instant},
};

use bumpalo::Bump;
use compiler::compiled::{
    ActionIdx, CompiledLinkPattern, CompiledModel, DestinationIdx, EdgeIdx, InstanceIdx,
};
use datatypes::idxvec::new_idx_type;
use hashbrown::HashSet;
use momba_model::models::Model;
use params::Params;
use parking_lot::Mutex;

pub mod exhaustive;

use crate::{
    compiler::{compile_model, compiled::CompiledDestination, Options, StateLayout},
    datatypes::idxvec::IdxVec,
    values::{
        memory::{bits::BitSlice, Load},
        types::ValueTyKind,
        FromWord,
    },
    vm::evaluate::Env,
};

pub mod compiler;
pub(crate) mod datatypes;
//pub mod engine;
pub mod params;
//pub mod simulator;
pub mod state_allocator;
pub mod values;
pub mod vm;

pub fn print_state(model: &CompiledModel, state: &BitSlice<StateLayout>) {
    for (field_idx, field) in model.variables.state_layout.fields.indexed_iter() {
        let addr = model.variables.state_offsets[field_idx];
        match field.ty().kind() {
            ValueTyKind::Bool => {
                println!(
                    "{:?}: {}",
                    field.name(),
                    bool::from_word(state.load_bool(addr))
                )
            }
            ValueTyKind::SignedInt(ty) => println!(
                "{:?}: {}",
                field.name(),
                i64::from_word(state.load_signed_int(addr, ty))
            ),
            ValueTyKind::UnsignedInt(ty) => println!(
                "{:?}: {}",
                field.name(),
                i64::from_word(state.load_unsigned_int(addr, ty))
            ),
            ValueTyKind::Float32 => todo!(),
            ValueTyKind::Float64 => todo!(),
            _ => todo!(),
        }
    }
}

new_idx_type! {
    pub TransitionItemIdx(u16)
}

#[derive(Debug, Clone)]
pub struct Transition {
    items: Range<TransitionItemIdx>,
}

#[derive(Debug, Clone)]
pub struct TransitionItem {
    instance_idx: InstanceIdx,
    edge_idx: EdgeIdx,
}

#[derive(Debug, Clone)]
pub struct DestinationItem {
    instance_idx: InstanceIdx,
    edge_idx: EdgeIdx,
    destination_idx: DestinationIdx,
}

struct StackTransitionItem<'stack> {
    parent: Option<&'stack StackTransitionItem<'stack>>,
    item: &'stack TransitionItem,
}

impl<'stack> StackTransitionItem<'stack> {
    pub fn build_transition(
        &self,
        out_items: &mut IdxVec<TransitionItemIdx, TransitionItem>,
    ) -> Transition {
        let start = out_items.next_idx();
        let mut current = self;
        loop {
            out_items.push(self.item.clone());
            match self.parent {
                Some(item) => current = item,
                None => break,
            }
        }
        let end = out_items.next_idx();
        Transition { items: start..end }
    }
}

// fn synchronize<'stack>(
//     //items: Option<&'stack StackTransitionItem<'stack>>,
//     out_items: &mut IdxVec<TransitionItemIdx, TransitionItem>,
//     out_transitions: &mut Vec<Transition>,
//     enabled_sync_edges: Vec<(EdgeIdx, ActionIdx)>,
//     instance_sync_edges: IdxVec<InstanceIdx, Range<usize>>,
//     patterns: &[CompiledLinkPattern],
// ) {
//     // match remaining {
//     //     [pattern, rest @ ..] => {}
//     //     [] => match items {
//     //         Some(items) => out_transitions.push(items.build_transition(out_items)),
//     //         None => {
//     //             // Do nothing!
//     //         }
//     //     },
//     // }
// }

fn filter_edges(
    mut edges: &[(TransitionItem, ActionIdx)],
    action: ActionIdx,
) -> &[(TransitionItem, ActionIdx)] {
    while let [edge, rest @ ..] = edges {
        if edge.1 != action {
            edges = rest;
        } else {
            break;
        }
    }
    edges
}

pub fn count_states_concurrent(
    model: &Model,
    params: &Params,
    num_workers: usize,
) -> Result<(), Box<dyn Error>> {
    assert!(num_workers > 0);

    let start = Instant::now();

    let compiled = compile_model(&model, &params, &Options::new())?;
    let state_size = compiled.variables.initial_state.len();

    let state_storage = (0..num_workers)
        .map(|_| Mutex::new(Bump::new()))
        .collect::<Vec<_>>();

    let initial_state = unsafe {
        &*{
            let bump = state_storage[0].lock();
            let initial_state = bump.alloc_slice_fill_copy(state_size, 0u8);
            initial_state.copy_from_slice(&compiled.variables.initial_state);
            initial_state as *const [u8]
        }
    };

    let visited = dashmap::DashSet::new();
    visited.insert(initial_state);

    exhaustive::workers::spawn_and_run_workers(
        num_workers,
        |_| {
            |ctx| {
                let bump = state_storage[ctx.worker_id()].lock();
                let mut next_state_bump =
                    bump.alloc_slice_fill_copy(state_size, 0u8) as *mut [u8];

                //let s = core::hash::BuildHasherDefault::<fxhash::FxHasher>::default();

                let mut transition_items = IdxVec::new();
                let mut transitions = Vec::new();

                let mut enabled_sync_edges = Vec::new();
                let mut instance_sync_edges = IdxVec::<InstanceIdx, _>::new();

                let mut sync_stack = Vec::new();
                let mut destinations_product = CartesianProductReusable::new();
                while let Some(state) = ctx.next_task() {
                    // if visited.len() % (1 << 18) == 0 {
                    //     println!("Visited: {}", visited.len());
                    // }

                    let mut env = Env::new(BitSlice::<StateLayout>::from_slice(&state));

                    transition_items.clear();
                    transitions.clear();
                    enabled_sync_edges.clear();
                    instance_sync_edges.clear();

                    #[cfg(feature = "statistics")]
                    let computing_enabled_edges_start = Instant::now();

                    for (instance_idx, instance) in compiled.instances.indexed_iter() {
                        let enabled_sync_edges_start = enabled_sync_edges.len();
                        instance.foreach_enabled_edge(&mut env, |edge_idx| {
                            let item = TransitionItem {
                                instance_idx,
                                edge_idx,
                            };
                            let edge = &instance.edges[edge_idx];
                            if let Some(action) = edge.action {
                                // This edge requires synchronizing.
                                enabled_sync_edges.push((item, action));
                            } else {
                                // This edge is internal and does not require synchronizing.
                                let item_idx = transition_items.next_idx();
                                transition_items.push(item);
                                transitions.push(Transition {
                                    items: (item_idx..transition_items.next_idx()),
                                })
                            }
                        });
                        instance_sync_edges
                            .push(enabled_sync_edges_start..enabled_sync_edges.len());
                    }

                    for link in &compiled.links.links {
                        sync_stack.clear();
                        let patterns = &compiled.links.patterns[link.patterns.clone()];
                        let [first_pattern, remaining_patterns @ ..] = patterns else {
                // Link does not contain any patters => No transitions!
                continue;
            };
                        let enabled_edges = filter_edges(
                            &enabled_sync_edges
                                [instance_sync_edges[first_pattern.instance].clone()],
                            first_pattern.action,
                        );
                        match enabled_edges {
                            [selected_edge, remaining_edges @ ..] => {
                                debug_assert_eq!(selected_edge.1, first_pattern.action);
                                sync_stack.push((
                                    remaining_edges as *const [(TransitionItem, ActionIdx)],
                                    selected_edge as *const (TransitionItem, ActionIdx),
                                    remaining_patterns as *const [CompiledLinkPattern],
                                ))
                            }
                            [] => {
                                // No edges with a matching action => No transitions!
                                continue;
                            }
                        }

                        // println!("Sync stack:");
                        // for (remaining_edges, selected_edge, remaining_patterns) in &sync_stack {
                        //     println!(
                        //         "  {:?}, {:?}, {:?}",
                        //         remaining_edges, selected_edge, remaining_patterns
                        //     );
                        // }

                        while let Some((_, _, remaining_patterns)) = sync_stack.last() {
                            let remaining_patterns = unsafe { &**remaining_patterns };
                            // println!("Sync stack:");
                            // for (remaining_edges, selected_edge, remaining_patterns) in &sync_stack {
                            //     println!(
                            //         "  {:?}, {:?}, {:?}",
                            //         remaining_edges, selected_edge, remaining_patterns
                            //     );
                            // }
                            match remaining_patterns {
                                [next_pattern, remaining_patterns @ ..] => {
                                    let enabled_edges = filter_edges(
                                        &enabled_sync_edges
                                            [instance_sync_edges[next_pattern.instance].clone()],
                                        next_pattern.action,
                                    );
                                    match enabled_edges {
                                        [selected_edge, remaining_edges @ ..] => {
                                            debug_assert_eq!(selected_edge.1, next_pattern.action);
                                            sync_stack.push((
                                                remaining_edges,
                                                selected_edge,
                                                remaining_patterns,
                                            ))
                                        }
                                        [] => {
                                            // No edges with a matching action => No transitions!
                                            break;
                                        }
                                    }
                                }
                                [] => {
                                    // We found a transition.
                                    let items_start = transition_items.next_idx();
                                    for (_, selected_edge, _) in &sync_stack {
                                        let (item, _) = unsafe { &**selected_edge };
                                        transition_items.push(item.clone());
                                    }
                                    let items_end = transition_items.next_idx();
                                    //println!("Add sync transition!");
                                    transitions.push(Transition {
                                        items: items_start..items_end,
                                    });

                                    // After adding the transition, we need to update the stack.
                                    while let Some((
                                        remaining_edges,
                                        previous_edge,
                                        remaining_patterns,
                                    )) = sync_stack.pop()
                                    {
                                        let remaining_edges = unsafe { &*remaining_edges };
                                        let previous_edge = unsafe { &*previous_edge };
                                        let remaining_patterns = unsafe { &*remaining_patterns };
                                        let enabled_edges =
                                            filter_edges(remaining_edges, previous_edge.1);
                                        match enabled_edges {
                                            [selected_edge, remaining_edges @ ..] => {
                                                debug_assert_eq!(selected_edge.1, previous_edge.1);
                                                sync_stack.push((
                                                    remaining_edges,
                                                    selected_edge,
                                                    remaining_patterns,
                                                ));
                                                break;
                                            }
                                            [] => {
                                                // No more edges. Try next.
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // for pattern in patterns {
                        //     sync_stack.push(&enabled_sync_edges[instance_sync_edges[pattern.instance].clone()])
                        // }
                    }

                    // println!("Transition Items: {:?}", transition_items);

                    for transition in &transitions {
                        let items = &transition_items[transition.items.clone()];
                        // let product = CartesianProduct::new(
                        //     items,
                        //     |item| {
                        //         let instance = &compiled.instances[item.instance_idx];
                        //         let edge = &instance.edges[item.edge_idx];
                        //         instance.destinations(edge)
                        //     },
                        //     |_, destination| Some(destination),
                        // );
                        let mut destinations_counter = 0;
                        destinations_product.produce(
                            items,
                            |item| {
                                let instance = &compiled.instances[item.instance_idx];
                                let edge = &instance.edges[item.edge_idx];
                                instance.destinations(edge)
                            },
                            |_, destination| Some(destination as *const _),
                            |product| {
                                //let mut probability = 1.0;
                                unsafe { (&mut *next_state_bump).copy_from_slice(state) }
                                let dst_state_mut =
                                    BitSlice::<StateLayout>::from_slice_mut(unsafe {
                                        &mut *next_state_bump
                                    });
                                // println!("Destination:");

                                // println!("  Source: {}", compiled.fmt_state(&env.state));
                                // println!(
                                //     "  Locations: {}",
                                //     compiled
                                //         .instances
                                //         .indexed_iter()
                                //         .map(|(idx, instance)| {
                                //             let loc = instance.load_location(&env.state);

                                //             format!(
                                //                 "{} = {:?} ({})",
                                //                 idx.as_usize(),
                                //                 instance.locations[loc].name,
                                //                 loc.as_usize()
                                //             )
                                //         })
                                //         .collect::<Vec<_>>()
                                //         .join("; ")
                                // );
                                for (item, destination) in product.items() {
                                    let instance = &compiled.instances[item.instance_idx];
                                    //let edge = &instance.edges[item.edge_idx];
                                    let destination: &CompiledDestination =
                                        unsafe { &**destination }; //&instance.destinations[destination.idx];
                                                                   //  probability *= destination.probability.evaluate(&mut env);
                                    for assignment in
                                        &instance.assignments[destination.assignments.clone()]
                                    {
                                        assignment.execute(dst_state_mut, &mut env);
                                    }
                                    // println!(
                                    //     "  Destination: {}:{}({}):{}",
                                    //     item.instance_idx.as_usize(),
                                    //     item.edge_idx.as_usize(),
                                    //     edge.original_idx,
                                    //     destination.idx.as_usize()
                                    // );
                                    instance.store_location(dst_state_mut, destination.target);
                                }

                                // println!("  Target: {}", compiled.fmt_state(dst_state_mut));
                                // println!(
                                //     "  Locations: {}",
                                //     compiled
                                //         .instances
                                //         .indexed_iter()
                                //         .map(|(idx, instance)| {
                                //             let loc = instance.load_location(dst_state_mut);

                                //             format!(
                                //                 "{} = {:?} ({})",
                                //                 idx.as_usize(),
                                //                 instance.locations[loc].name,
                                //                 loc.as_usize()
                                //             )
                                //         })
                                //         .collect::<Vec<_>>()
                                //         .join("; ")
                                // );

                                drop(dst_state_mut);

                                if visited.insert(unsafe { &*next_state_bump }) {
                                    ctx.push_task(unsafe { &*next_state_bump });
                                    next_state_bump = bump.alloc_slice_fill_copy(state_size, 0u8);
                                    // println!("Pushed!");
                                }
                                destinations_counter += 1;
                            },
                        );
                        //println!("Destinations: {}", destinations_counter);
                    }
                }

                // println!("{:?}", enabled_sync_edges);
                // println!("{:?}", instance_sync_edges);

                // println!("Queue Pressure: {}", queue_pressure);
                // println!("States: {}", visited.len());
            }
        },
        exhaustive::workers::WorkerQueueStrategy::Fifo,
        [initial_state],
    );

    let end = Instant::now();

    println!("Total Time: {:.02}", (end - start).as_secs_f64());
    println!("States: {}", visited.len());

    std::process::exit(0);
}

pub fn count_states(model: &Model, params: &Params) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    println!("Compiling...");
    let compiled = compile_model(&model, &params, &Options::new())?;

    println!("Counting states...");

    println!("State Layout: {}", compiled.variables.state_layout);
    println!(
        "State Size: {} bytes",
        usize::from(
            compiled
                .variables
                .state_layout
                .size::<StateLayout>()
                .to_bytes()
        )
    );

    let bump = Bump::with_capacity(10 * 1024 * 1024 * 1024); // 10 GiB

    let state_size = compiled.variables.initial_state.len();

    let initial_state_bump = bump.alloc_slice_fill_copy(state_size, 0u8);
    initial_state_bump.copy_from_slice(&compiled.variables.initial_state);

    let mut state_stack = VecDeque::from(vec![&*initial_state_bump]);

    let mut next_state_bump = bump.alloc_slice_fill_copy(state_size, 0u8) as *mut [u8];

    //let s = core::hash::BuildHasherDefault::<fxhash::FxHasher>::default();

    let mut visited = HashSet::<&[u8]>::with_capacity(100_000_000);
    for state in &state_stack {
        visited.insert(*state);
    }

    let mut transition_items = IdxVec::new();
    let mut transitions = Vec::new();

    let mut enabled_sync_edges = Vec::new();
    let mut instance_sync_edges = IdxVec::<InstanceIdx, _>::new();

    #[cfg(feature = "statistics")]
    let mut computing_enabled_edges = Duration::default();
    #[cfg(feature = "statistics")]
    let mut computing_transitions = Duration::default();
    #[cfg(feature = "statistics")]
    let mut computing_successors = Duration::default();

    let mut queue_pressure = 0;

    let mut sync_stack = Vec::new();
    let mut destinations_product = CartesianProductReusable::new();
    while let Some(state) = state_stack.pop_front() {
        // if visited.len() % (1 << 18) == 0 {
        //     println!("Visited: {}", visited.len());
        // }

        queue_pressure = queue_pressure.max(state_stack.len());

        // println!("State: {:?}", state);

        let mut env = Env::new(BitSlice::<StateLayout>::from_slice(&state));

        //compiled.print_state(&env.state);

        transition_items.clear();
        transitions.clear();
        enabled_sync_edges.clear();
        instance_sync_edges.clear();

        #[cfg(feature = "statistics")]
        let computing_enabled_edges_start = Instant::now();

        for (instance_idx, instance) in compiled.instances.indexed_iter() {
            let enabled_sync_edges_start = enabled_sync_edges.len();
            instance.foreach_enabled_edge(&mut env, |edge_idx| {
                let item = TransitionItem {
                    instance_idx,
                    edge_idx,
                };
                let edge = &instance.edges[edge_idx];
                if let Some(action) = edge.action {
                    // This edge requires synchronizing.
                    enabled_sync_edges.push((item, action));
                } else {
                    // This edge is internal and does not require synchronizing.
                    let item_idx = transition_items.next_idx();
                    transition_items.push(item);
                    transitions.push(Transition {
                        items: (item_idx..transition_items.next_idx()),
                    })
                }
            });
            instance_sync_edges.push(enabled_sync_edges_start..enabled_sync_edges.len());
        }

        #[cfg(feature = "statistics")]
        {
            computing_enabled_edges += Instant::now() - computing_enabled_edges_start;
        }

        // println!("Transition Items: {:?}", transition_items);
        // println!("Transitions: {:?}", transitions);

        // println!("Enabled Sync Transitions:",);
        // for (instance_idx, enabled_range) in instance_sync_edges.indexed_iter() {
        //     let instance = &compiled.instances[instance_idx];
        //     println!(
        //         "  {:?}: {}",
        //         instance.automaton,
        //         enabled_range.end - enabled_range.start
        //     );
        // }

        #[cfg(feature = "statistics")]
        let computing_transitions_start = Instant::now();

        for link in &compiled.links.links {
            sync_stack.clear();
            let patterns = &compiled.links.patterns[link.patterns.clone()];
            let [first_pattern, remaining_patterns @ ..] = patterns else {
                // Link does not contain any patters => No transitions!
                continue;
            };
            let enabled_edges = filter_edges(
                &enabled_sync_edges[instance_sync_edges[first_pattern.instance].clone()],
                first_pattern.action,
            );
            match enabled_edges {
                [selected_edge, remaining_edges @ ..] => {
                    debug_assert_eq!(selected_edge.1, first_pattern.action);
                    sync_stack.push((
                        remaining_edges as *const [(TransitionItem, ActionIdx)],
                        selected_edge as *const (TransitionItem, ActionIdx),
                        remaining_patterns as *const [CompiledLinkPattern],
                    ))
                }
                [] => {
                    // No edges with a matching action => No transitions!
                    continue;
                }
            }

            // println!("Sync stack:");
            // for (remaining_edges, selected_edge, remaining_patterns) in &sync_stack {
            //     println!(
            //         "  {:?}, {:?}, {:?}",
            //         remaining_edges, selected_edge, remaining_patterns
            //     );
            // }

            while let Some((_, _, remaining_patterns)) = sync_stack.last() {
                let remaining_patterns = unsafe { &**remaining_patterns };
                // println!("Sync stack:");
                // for (remaining_edges, selected_edge, remaining_patterns) in &sync_stack {
                //     println!(
                //         "  {:?}, {:?}, {:?}",
                //         remaining_edges, selected_edge, remaining_patterns
                //     );
                // }
                match remaining_patterns {
                    [next_pattern, remaining_patterns @ ..] => {
                        let enabled_edges = filter_edges(
                            &enabled_sync_edges[instance_sync_edges[next_pattern.instance].clone()],
                            next_pattern.action,
                        );
                        match enabled_edges {
                            [selected_edge, remaining_edges @ ..] => {
                                debug_assert_eq!(selected_edge.1, next_pattern.action);
                                sync_stack.push((
                                    remaining_edges,
                                    selected_edge,
                                    remaining_patterns,
                                ))
                            }
                            [] => {
                                // No edges with a matching action => No transitions!
                                break;
                            }
                        }
                    }
                    [] => {
                        // We found a transition.
                        let items_start = transition_items.next_idx();
                        for (_, selected_edge, _) in &sync_stack {
                            let (item, _) = unsafe { &**selected_edge };
                            transition_items.push(item.clone());
                        }
                        let items_end = transition_items.next_idx();
                        //println!("Add sync transition!");
                        transitions.push(Transition {
                            items: items_start..items_end,
                        });

                        // After adding the transition, we need to update the stack.
                        while let Some((remaining_edges, previous_edge, remaining_patterns)) =
                            sync_stack.pop()
                        {
                            let remaining_edges = unsafe { &*remaining_edges };
                            let previous_edge = unsafe { &*previous_edge };
                            let remaining_patterns = unsafe { &*remaining_patterns };
                            let enabled_edges = filter_edges(remaining_edges, previous_edge.1);
                            match enabled_edges {
                                [selected_edge, remaining_edges @ ..] => {
                                    debug_assert_eq!(selected_edge.1, previous_edge.1);
                                    sync_stack.push((
                                        remaining_edges,
                                        selected_edge,
                                        remaining_patterns,
                                    ));
                                    break;
                                }
                                [] => {
                                    // No more edges. Try next.
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // for pattern in patterns {
            //     sync_stack.push(&enabled_sync_edges[instance_sync_edges[pattern.instance].clone()])
            // }
        }

        #[cfg(feature = "statistics")]
        {
            computing_transitions += Instant::now() - computing_transitions_start;
        }

        // println!("Transition Items: {:?}", transition_items);
        //println!("Transitions: {}", transitions.len());

        #[cfg(feature = "statistics")]
        let computing_successors_start = Instant::now();

        for transition in &transitions {
            let items = &transition_items[transition.items.clone()];
            // let product = CartesianProduct::new(
            //     items,
            //     |item| {
            //         let instance = &compiled.instances[item.instance_idx];
            //         let edge = &instance.edges[item.edge_idx];
            //         instance.destinations(edge)
            //     },
            //     |_, destination| Some(destination),
            // );
            let mut destinations_counter = 0;
            destinations_product.produce(
                items,
                |item| {
                    let instance = &compiled.instances[item.instance_idx];
                    let edge = &instance.edges[item.edge_idx];
                    instance.destinations(edge)
                },
                |_, destination| Some(destination as *const _),
                |product| {
                    //let mut probability = 1.0;
                    unsafe { (&mut *next_state_bump).copy_from_slice(state) }
                    let dst_state_mut =
                        BitSlice::<StateLayout>::from_slice_mut(unsafe { &mut *next_state_bump });
                    // println!("Destination:");

                    // println!("  Source: {}", compiled.fmt_state(&env.state));
                    // println!(
                    //     "  Locations: {}",
                    //     compiled
                    //         .instances
                    //         .indexed_iter()
                    //         .map(|(idx, instance)| {
                    //             let loc = instance.load_location(&env.state);

                    //             format!(
                    //                 "{} = {:?} ({})",
                    //                 idx.as_usize(),
                    //                 instance.locations[loc].name,
                    //                 loc.as_usize()
                    //             )
                    //         })
                    //         .collect::<Vec<_>>()
                    //         .join("; ")
                    // );
                    for (item, destination) in product.items() {
                        let instance = &compiled.instances[item.instance_idx];
                        //let edge = &instance.edges[item.edge_idx];
                        let destination: &CompiledDestination = unsafe { &**destination }; //&instance.destinations[destination.idx];
                                                                                           //  probability *= destination.probability.evaluate(&mut env);
                        for assignment in &instance.assignments[destination.assignments.clone()] {
                            assignment.execute(dst_state_mut, &mut env);
                        }
                        // println!(
                        //     "  Destination: {}:{}({}):{}",
                        //     item.instance_idx.as_usize(),
                        //     item.edge_idx.as_usize(),
                        //     edge.original_idx,
                        //     destination.idx.as_usize()
                        // );
                        instance.store_location(dst_state_mut, destination.target);
                    }

                    // println!("  Target: {}", compiled.fmt_state(dst_state_mut));
                    // println!(
                    //     "  Locations: {}",
                    //     compiled
                    //         .instances
                    //         .indexed_iter()
                    //         .map(|(idx, instance)| {
                    //             let loc = instance.load_location(dst_state_mut);

                    //             format!(
                    //                 "{} = {:?} ({})",
                    //                 idx.as_usize(),
                    //                 instance.locations[loc].name,
                    //                 loc.as_usize()
                    //             )
                    //         })
                    //         .collect::<Vec<_>>()
                    //         .join("; ")
                    // );

                    drop(dst_state_mut);

                    if visited.insert(unsafe { &*next_state_bump }) {
                        state_stack.push_back(unsafe { &*next_state_bump });
                        next_state_bump = bump.alloc_slice_fill_copy(state_size, 0u8);
                        // println!("Pushed!");
                    }
                    destinations_counter += 1;
                },
            );
            //println!("Destinations: {}", destinations_counter);
        }

        #[cfg(feature = "statistics")]
        {
            computing_successors += Instant::now() - computing_successors_start;
        }
    }

    // println!("{:?}", enabled_sync_edges);
    // println!("{:?}", instance_sync_edges);

    println!("Queue Pressure: {}", queue_pressure);
    println!("States: {}", visited.len());

    let end = Instant::now();

    println!("Total Time: {:.02}", (end - start).as_secs_f64());
    #[cfg(feature = "statistics")]
    {
        println!(
            "Enabled Edges: {:.02}",
            computing_enabled_edges.as_secs_f64()
        );
        println!("Transitions: {:.02}", computing_transitions.as_secs_f64());
        println!("Successors: {:.02}", computing_successors.as_secs_f64());
    }

    std::process::exit(0);
}

/// Computes the Cartesian product using a single stack.
pub struct CartesianProduct<
    'p,
    K,
    I,
    T: 'p,
    R: Fn(&'p K) -> &'p [T],
    S: Fn(&'p K, &'p T) -> Option<I>,
> {
    resolve: R,
    select: S,
    stack: Vec<(&'p [T], &'p K, I, &'p [K])>,
}

impl<'p, K, I, T: 'p, R: Fn(&'p K) -> &'p [T], S: Fn(&'p K, &'p T) -> Option<I>>
    CartesianProduct<'p, K, I, T, R, S>
{
    pub fn new(keys: &'p [K], resolve: R, select: S) -> Self {
        let [first_key, remaining_keys @ ..]  = keys else {
            return Self { resolve, select, stack: Vec::new()};
        };
        let mut handles = resolve(first_key);
        while let [first_handle, remaining_handles @ ..] = handles {
            if let Some(item) = select(first_key, first_handle) {
                return Self {
                    resolve,
                    select,
                    stack: vec![(remaining_handles, first_key, item, remaining_keys)],
                };
            }
            handles = remaining_handles;
        }
        Self {
            resolve,
            select,
            stack: Vec::new(),
        }
    }

    pub fn items(&self) -> impl Iterator<Item = (&'p K, &I)> {
        self.stack.iter().map(|(_, key, item, _)| (*key, item))
    }

    pub fn produce(mut self, mut emit: impl FnMut(&Self)) {
        'produce: while let Some((_, _, _, remaining_keys)) = self.stack.last() {
            match remaining_keys {
                [next_key, remaining_keys @ ..] => {
                    let mut handles = (self.resolve)(next_key);
                    while let [handle, remaining_handles @ ..] = handles {
                        if let Some(item) = (self.select)(next_key, handle) {
                            self.stack
                                .push((remaining_handles, next_key, item, remaining_keys));
                            continue 'produce;
                        }
                        handles = remaining_handles;
                    }
                    return;
                }
                [] => {
                    emit(&self);
                    'outer: while let Some((mut handles, key, _, remaining_keys)) = self.stack.pop()
                    {
                        while let [handle, remaining_handles @ ..] = handles {
                            if let Some(item) = (self.select)(key, handle) {
                                self.stack
                                    .push((remaining_handles, key, item, remaining_keys));
                                break 'outer;
                            }
                            handles = remaining_handles;
                        }
                    }
                }
            }
        }
    }
}

pub struct CartesianProductReusable<K, I, T> {
    stack: Vec<(*const [T], *const K, I, *const [K])>,
}

pub struct Items<'prod, 'p, K, I, T> {
    product: &'prod CartesianProductReusable<K, I, T>,
    _phantom_data: PhantomData<&'p K>,
}

impl<'prod, 'p, K, I, T> Items<'prod, 'p, K, I, T> {
    pub fn items(&self) -> impl Iterator<Item = (&'p K, &I)> {
        self.product
            .stack
            .iter()
            .map(|(_, key, item, _)| (unsafe { &**key }, item))
    }
}

impl<K, I, T> CartesianProductReusable<K, I, T> {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn produce<'p, R: Fn(&'p K) -> &'p [T], S: Fn(&'p K, &'p T) -> Option<I>>(
        &mut self,
        keys: &'p [K],
        resolve: R,
        select: S,
        mut emit: impl FnMut(Items<'_, 'p, K, I, T>),
    ) where
        T: 'p,
    {
        self.stack.clear();
        let [first_key, remaining_keys @ ..]  = keys else {
            return;
        };
        let mut handles = resolve(first_key);
        while let [first_handle, remaining_handles @ ..] = handles {
            if let Some(item) = select(first_key, first_handle) {
                self.stack
                    .push((remaining_handles, first_key, item, remaining_keys));
                break;
            }
            handles = remaining_handles;
        }
        'produce: while let Some((_, _, _, remaining_keys)) = self.stack.last() {
            let remaining_keys = unsafe { &**remaining_keys };
            match remaining_keys {
                [next_key, remaining_keys @ ..] => {
                    let mut handles = (resolve)(next_key);
                    while let [handle, remaining_handles @ ..] = handles {
                        if let Some(item) = (select)(next_key, handle) {
                            self.stack
                                .push((remaining_handles, next_key, item, remaining_keys));
                            continue 'produce;
                        }
                        handles = remaining_handles;
                    }
                    return;
                }
                [] => {
                    emit(Items {
                        product: self,
                        _phantom_data: PhantomData,
                    });
                    'outer: while let Some((handles, key, _, remaining_keys)) = self.stack.pop() {
                        let mut handles = unsafe { &*handles };
                        let key = unsafe { &*key };
                        while let [handle, remaining_handles @ ..] = handles {
                            if let Some(item) = (select)(key, handle) {
                                self.stack
                                    .push((remaining_handles, key, item, remaining_keys));
                                break 'outer;
                            }
                            handles = remaining_handles;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_product() {
        let keys = vec![0, 1, 2, 3];
        let handles = vec![vec![2, 4, 6, 1, 3], vec![5, 12], vec![16], vec![2, 4]];

        let product = CartesianProduct::new(
            &keys,
            |key| &handles[*key],
            |_, value| {
                if *value % 2 == 0 {
                    Some((*value, "even"))
                } else {
                    None
                }
            },
        );

        let mut counter = 0;
        product.produce(|product| {
            println!("{:?}", product.items().collect::<Vec<_>>());
            counter += 1;
        });
        assert_eq!(counter, 6);
    }

    #[test]
    pub fn test_product_reusable() {
        let keys = vec![0, 1, 2, 3];
        let handles = vec![vec![2, 4, 6, 1, 3], vec![5, 12], vec![16], vec![2, 4]];

        let mut counter = 0;
        let mut product = CartesianProductReusable::new();
        product.produce(
            &keys,
            |key| &handles[*key],
            |_, value| {
                if *value % 2 == 0 {
                    Some((*value, "even"))
                } else {
                    None
                }
            },
            |product| {
                println!("{:?}", product.items().collect::<Vec<_>>());
                counter += 1;
            },
        );
        assert_eq!(counter, 6);
    }
}
