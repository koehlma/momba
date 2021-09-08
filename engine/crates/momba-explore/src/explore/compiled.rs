use std::convert::TryInto;

use indexmap::IndexSet;

use crate::model;
use crate::time;

use super::evaluate;
use super::evaluate::Scope;
use super::*;

pub type LocationIndex = usize;

pub type GlobalEnvironment<'s> = evaluate::Environment<'s, 2>;
pub type EdgeEnvironment<'s> = evaluate::Environment<'s, 3>;

pub struct CompiledAssignment {
    pub target: evaluate::CompiledTargetExpression<3>,
    pub value: evaluate::CompiledExpression<3>,
}

pub struct CompiledAutomaton<Z: time::Time> {
    pub reference: model::AutomatonReference,
    pub locations: Vec<CompiledLocation<Z>>,
}

pub struct CompiledLocation<Z: time::Time> {
    pub reference: model::LocationReference,
    pub invariant: Vec<CompiledClockConstraint<Z>>,
    pub internal_edges: Vec<CompiledEdge<Z>>,
    pub visible_edges: Vec<Vec<CompiledVisibleEdge<Z>>>,
}

impl<Z: time::Time> CompiledLocation<Z> {
    fn new(
        network: &model::Network,
        global_scope: &evaluate::Scope<2>,
        zone_compiler: &Z,
        automaton: &model::Automaton,
        automaton_index: usize,
        location: &model::Location,
        reference: model::LocationReference,
        assignment_groups: &IndexSet<usize>,
    ) -> Self {
        let mut internal_edges = Vec::new();
        let mut visible_edges: Vec<Vec<CompiledVisibleEdge<Z>>> =
            (0..network.declarations.action_labels.len())
                .map(|_| Vec::new())
                .collect();
        for (edge_index, edge) in location.edges.iter().enumerate() {
            let edge_reference = model::EdgeReference {
                location: reference.clone(),
                index: edge_index,
            };
            let compiled_edge = CompiledEdge::new(
                network,
                zone_compiler,
                automaton,
                automaton_index,
                edge,
                edge_reference,
                assignment_groups,
            );
            match &edge.pattern {
                model::ActionPattern::Silent => internal_edges.push(compiled_edge),
                model::ActionPattern::Labeled(model::LabeledPattern { label, arguments }) => {
                    let action_type_index = network
                        .declarations
                        .action_labels
                        .get_index_of(label)
                        .unwrap();
                    visible_edges[action_type_index].push(CompiledVisibleEdge {
                        base: compiled_edge,
                        write_arguments: arguments
                            .iter()
                            .enumerate()
                            .filter_map(|(argument_index, argument)| match argument {
                                model::PatternArgument::Write { value } => {
                                    Some((argument_index, global_scope.compile(value)))
                                }
                                _ => None,
                            })
                            .collect(),
                    })
                }
            }
        }
        CompiledLocation {
            reference,
            invariant: location
                .invariant
                .iter()
                .map(|constraint| {
                    CompiledClockConstraint::compile(
                        zone_compiler,
                        constraint,
                        &network.global_scope(),
                    )
                })
                .collect(),
            internal_edges: internal_edges,
            visible_edges: visible_edges,
        }
    }
}

pub struct CompiledEdge<Z: time::Time> {
    pub reference: model::EdgeReference,
    pub guard: CompiledGuard<Z>,
    pub destinations: Vec<CompiledDestination<Z>>,
    pub observations: Vec<CompiledObservation>,
    pub numeric_reference: (usize, usize),
}

pub struct CompiledObservation {
    pub label: LabelIndex,
    pub arguments: Vec<evaluate::CompiledExpression<3>>,
    pub probability: evaluate::CompiledExpression<3>,
}

pub struct CompiledVisibleEdge<Z: time::Time> {
    pub base: CompiledEdge<Z>,
    pub write_arguments: Box<[(usize, evaluate::CompiledExpression<2>)]>,
}

impl<Z: time::Time> CompiledEdge<Z> {
    pub fn new(
        network: &model::Network,
        time_type: &Z,
        automaton: &model::Automaton,
        automaton_index: usize,
        edge: &model::Edge,
        reference: model::EdgeReference,
        assignment_groups: &IndexSet<usize>,
    ) -> Self {
        let global_scope = network.global_scope();
        let edge_scope = edge.edge_scope(network, edge);
        let guard = CompiledGuard {
            boolean_condition: global_scope.compile(&edge.guard.boolean_condition),
            clock_constraints: edge
                .guard
                .clock_constraints
                .iter()
                .map(|constraint| {
                    CompiledClockConstraint::compile(time_type, constraint, &global_scope)
                })
                .collect(),
        };
        let destinations = edge
            .destinations
            .iter()
            .enumerate()
            .map(|(destination_index, destination)| CompiledDestination {
                automaton_index,
                reference: model::DestinationReference {
                    edge: reference.clone(),
                    index: destination_index,
                },
                location: automaton
                    .locations
                    .get_index_of(&destination.location)
                    .unwrap(),
                probability: edge_scope.compile(&destination.probability),
                reset: time_type.compile_clocks(&destination.reset),
                assignments: assignment_groups
                    .iter()
                    .map(|index| {
                        destination
                            .assignments
                            .iter()
                            .filter(|assignment| assignment.index == *index)
                            .map(|assignment| CompiledAssignment {
                                target: edge_scope.compile_target(&assignment.target),
                                value: edge_scope.compile(&assignment.value),
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect();
        let observations = edge
            .observations
            .iter()
            .map(|observation| CompiledObservation {
                label: network
                    .declarations
                    .action_labels
                    .get_index_of(&observation.label)
                    .unwrap(),
                arguments: observation
                    .arguments
                    .iter()
                    .map(|argument| edge_scope.compile(argument))
                    .collect(),
                probability: edge_scope.compile(&observation.probability),
            })
            .collect();
        CompiledEdge {
            reference,
            guard,
            destinations,
            observations,
            numeric_reference: (automaton_index, edge.number),
        }
    }

    pub fn is_enabled(&self, global_env: &evaluate::Environment<2>) -> bool {
        self.guard
            .boolean_condition
            .evaluate(global_env)
            .try_into()
            .unwrap()
    }
}

pub struct CompiledGuard<Z: time::Time> {
    pub boolean_condition: evaluate::CompiledExpression<2>,
    pub clock_constraints: Vec<CompiledClockConstraint<Z>>,
}

pub struct CompiledClockConstraint<T: time::Time> {
    pub difference: T::CompiledDifference,
    pub is_strict: bool,
    pub bound: evaluate::CompiledExpression<2>,
}

impl<T: time::Time> CompiledClockConstraint<T> {
    pub fn compile(
        time_type: &T,
        constraint: &model::ClockConstraint,
        global_scope: &evaluate::Scope<2>,
    ) -> Self {
        CompiledClockConstraint {
            difference: time_type.compile_difference(&constraint.left, &constraint.right),
            is_strict: constraint.is_strict,
            bound: global_scope.compile(&constraint.bound),
        }
    }

    pub fn evaluate(&self, global_env: &GlobalEnvironment) -> time::Constraint<T> {
        time::Constraint {
            difference: self.difference.clone(),
            is_strict: self.is_strict,
            bound: self.bound.evaluate(global_env),
        }
    }
}

pub struct CompiledDestination<Z: time::Time> {
    pub automaton_index: usize,
    pub reference: model::DestinationReference,
    pub location: LocationIndex,
    pub probability: evaluate::CompiledExpression<3>,
    pub reset: Z::CompiledClocks,
    pub assignments: Box<[Box<[CompiledAssignment]>]>,
}

pub struct SyncVectorItem {
    pub automaton_index: usize,
    pub action_type: model::LabelIndex,
    pub slot_mapping: Box<[usize]>,
}

impl SyncVectorItem {
    pub fn argument_to_slot_index(&self, argument_index: usize) -> usize {
        self.slot_mapping[argument_index]
    }

    pub fn compute_link_edges<'c, Z: time::Time>(
        &'c self,
        global_env: &evaluate::Environment<2>,
        enabled_edges: &Box<[Box<[Box<[&'c CompiledVisibleEdge<Z>]>]>]>,
    ) -> Vec<LinkEdge<Z>> {
        enabled_edges[self.automaton_index][self.action_type]
            .iter()
            .map(|edge| LinkEdge {
                compiled: edge,
                write_slots: edge
                    .write_arguments
                    .iter()
                    .map(|(argument_index, expression)| {
                        let slot_index = self.argument_to_slot_index(*argument_index);
                        let value = expression.evaluate(global_env);
                        (slot_index, value)
                    })
                    .collect(),
            })
            .collect()
    }
}

pub struct LinkEdge<'c, Z: time::Time> {
    pub compiled: &'c CompiledVisibleEdge<Z>,
    pub write_slots: Box<[(usize, model::Value)]>,
}

pub struct CompiledLink {
    pub slots_template: Vec<Option<model::Value>>,
    pub sync_vector: Box<[SyncVectorItem]>,
    pub result: CompiledLinkResult,
}

pub enum CompiledLinkResult {
    Silent,
    Labeled {
        action_label: model::LabelIndex,
        slot_mapping: Box<[usize]>,
    },
}

pub struct CompiledNetwork<Z: time::Time> {
    pub zone_compiler: Z,
    pub global_scope: Scope<2>,
    pub automata: Box<[CompiledAutomaton<Z>]>,
    pub links: Box<[CompiledLink]>,
    pub transient_values: Box<[evaluate::CompiledExpression<1>]>,
    pub assignment_groups: IndexSet<usize>,
}

impl<Z: time::Time> CompiledNetwork<Z> {
    pub fn new(network: &model::Network) -> Self {
        let zone_compiler = Z::new(network).unwrap();
        let global_scope = network.global_scope();
        let mut assignment_groups: IndexSet<usize> = network
            .automata
            .values()
            .map(|automaton| automaton.locations.values())
            .flatten()
            .map(|location| location.edges.iter())
            .flatten()
            .map(|edge| edge.destinations.iter())
            .flatten()
            .map(|destination| destination.assignments.iter())
            .flatten()
            .map(|assignment| assignment.index)
            .collect();
        assignment_groups.sort();
        let automata = network
            .automata
            .iter()
            .enumerate()
            .map(|(automaton_index, (name, automaton))| CompiledAutomaton {
                reference: model::AutomatonReference { name: name.clone() },
                locations: automaton
                    .locations
                    .iter()
                    .map(|(location_name, location)| {
                        CompiledLocation::new(
                            network,
                            &global_scope,
                            &zone_compiler,
                            automaton,
                            automaton_index,
                            location,
                            model::LocationReference {
                                automaton: model::AutomatonReference { name: name.clone() },
                                name: location_name.clone(),
                            },
                            &assignment_groups,
                        )
                    })
                    .collect(),
            })
            .collect();
        let links = network
            .links
            .iter()
            .map(|link| CompiledLink {
                slots_template: vec![None; link.slots.len()],
                sync_vector: link
                    .vector
                    .iter()
                    .map(|(automaton_name, link_pattern)| SyncVectorItem {
                        automaton_index: network.automata.get_index_of(automaton_name).unwrap(),
                        action_type: network
                            .declarations
                            .action_labels
                            .get_index_of(&link_pattern.action_type)
                            .unwrap(),
                        slot_mapping: link_pattern
                            .arguments
                            .iter()
                            .map(|slot_name| link.slots.get_index_of(slot_name).unwrap())
                            .collect(),
                    })
                    .collect(),
                result: match &link.result {
                    model::LinkResult::Silent => CompiledLinkResult::Silent,
                    model::LinkResult::Labeled(model::LinkPattern {
                        action_type,
                        arguments,
                    }) => CompiledLinkResult::Labeled {
                        action_label: network
                            .declarations
                            .action_labels
                            .get_index_of(action_type)
                            .unwrap(),
                        slot_mapping: arguments
                            .iter()
                            .map(|slot_name| link.slots.get_index_of(slot_name).unwrap())
                            .collect(),
                    },
                },
            })
            .collect();
        let transient_scope = network.transient_scope();
        let transient_values = network
            .declarations
            .transient_variables
            .values()
            .map(|expr| transient_scope.compile(expr))
            .collect();
        CompiledNetwork {
            zone_compiler,
            global_scope,
            automata,
            links,
            transient_values,
            assignment_groups,
        }
    }

    pub fn compile_global_expression(
        &self,
        expr: &model::Expression,
    ) -> evaluate::CompiledExpression<2> {
        self.global_scope.compile(expr)
    }

    pub fn compute_transition<'c>(
        &self,
        state: &State<Z>,
        global_env: &GlobalEnvironment,
        link: &'c CompiledLink,
        link_edges: &[&LinkEdge<'c, Z>],
    ) -> Option<Transition<'c, Z>> {
        let zone = &state.valuations;
        debug_assert_eq!(link.sync_vector.len(), link_edges.len());
        let mut slots = link.slots_template.clone();
        let mut valuations = zone.clone();
        for (link_edge, _) in link_edges.iter().zip(link.sync_vector.iter()) {
            for (slot_index, value) in link_edge.write_slots.iter() {
                match &slots[*slot_index] {
                    None => slots[*slot_index] = Some(value.clone()),
                    Some(other_value) => {
                        if value != other_value {
                            return None;
                        }
                    }
                }
            }
            // We may want to improve the efficiency of this function in the future.
            //
            // Instead of applying each constraint individually applying them in bulk
            // makes canonicalization more efficient for clock zones.
            valuations = link_edge.compiled.base.guard.clock_constraints.iter().fold(
                valuations,
                |valuations, constraint| {
                    self.zone_compiler.constrain(
                        valuations,
                        &constraint.difference,
                        constraint.is_strict,
                        constraint.bound.evaluate(&global_env),
                    )
                },
            );
            if self.zone_compiler.is_empty(&valuations) {
                return None;
            }
        }
        slots
            .into_iter()
            .collect::<Option<Box<[_]>>>()
            .map(|slots| {
                let actions: Box<[Action]> = link
                    .sync_vector
                    .iter()
                    .map(|vector_item| {
                        Action::new(
                            vector_item.action_type,
                            vector_item
                                .slot_mapping
                                .iter()
                                .map(|slot_index| slots[*slot_index].clone())
                                .collect(),
                        )
                    })
                    .collect();
                let observations = link_edges
                    .iter()
                    .zip(actions.iter())
                    .map(|(link_edge, action)| {
                        let edge_env = state.edge_env(action.arguments());
                        link_edge
                            .compiled
                            .base
                            .observations
                            .iter()
                            .map(|observation| Observation {
                                label: observation.label,
                                arguments: observation
                                    .arguments
                                    .iter()
                                    .map(|argument| argument.evaluate(&edge_env))
                                    .collect(),
                                probability: observation.probability.evaluate(&edge_env),
                            })
                            .collect::<Box<_>>()
                    })
                    .collect();
                Transition {
                    edges: link_edges
                        .iter()
                        .map(|link_edge| &link_edge.compiled.base)
                        .collect(),
                    bare: BareTransition {
                        valuations,
                        actions,
                        action: match &link.result {
                            CompiledLinkResult::Silent => Action::Silent,
                            CompiledLinkResult::Labeled {
                                action_label,
                                slot_mapping,
                            } => Action::new(
                                *action_label,
                                slot_mapping
                                    .iter()
                                    .map(|slot_index| slots[*slot_index].clone())
                                    .collect(),
                            ),
                        },
                        observations,
                    },
                }
            })
    }

    pub fn compute_transient_values(&self, global_store: &[model::Value]) -> Box<[model::Value]> {
        let env = evaluate::Environment::new([global_store]);
        self.transient_values
            .iter()
            .map(|expr| expr.evaluate(&env))
            .collect()
    }

    // pub fn get_compiled_location(&self, reference: &LocationReference) -> &CompiledLocation<Z> {
    //     &self.automata[reference.automaton].locations[reference.index]
    // }

    // pub fn get_compiled_edge(&self, reference: &EdgeReference) -> &CompiledEdge<Z> {
    //     let location = self.get_compiled_location(&reference.location);
    //     for edge in location.internal_edges.iter() {
    //         if edge.reference.index == reference.index {
    //             return edge;
    //         }
    //     }
    //     for edges in location.visible_edges.iter() {
    //         for edge in edges.iter() {
    //             if edge.base.reference.index == reference.index {
    //                 return &edge.base;
    //             }
    //         }
    //     }
    //     panic!()
    // }
}
