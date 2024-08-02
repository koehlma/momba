//! Synthesis of configuration monitors.

use std::{collections::HashMap, hash::Hash, ops::Deref, time::Instant};

use hashbrown::HashSet;
use indexmap::IndexSet;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    algorithms,
    cudd::{self, BddManager, BddNode},
    frontends,
    lattice::PartiallyOrdered,
    logic::propositional::Formula,
    ts::{
        self,
        output::dot::TransitionAttributes,
        traits::{
            BaseTs, InitialStates, MakeDenseStateSet, MakeSparseStateSet, Predecessors, StateSet,
            States, Successors, Transitions, TsRef,
        },
        types::{VatsLabel, Vts, VtsState},
        StateId, Ts,
    },
    Arguments,
};

pub fn bdd_to_formula<T: Clone>(atoms: &[T], node: &BddNode) -> Formula<T> {
    let mut result = Formula::False;
    for cube in node.cubes() {
        let mut formula = Formula::True;
        for (idx, trinary) in cube.iter().enumerate() {
            match trinary {
                cudd::Trinary::False => {
                    formula = formula.and(Formula::Atom(atoms[idx].clone()).not())
                }
                cudd::Trinary::True => formula = formula.and(Formula::Atom(atoms[idx].clone())),
                cudd::Trinary::Any => {
                    // Value is irrelevant. Do nothing.
                }
            }
        }
        result = result.or(formula);
    }
    result
}

fn translate_guard<'m>(
    manager: &'m BddManager,
    features: &mut IndexSet<String>,
    formula: &Formula<String>,
) -> BddNode<'m> {
    match formula {
        Formula::Atom(feature) => {
            if let Some(index) = features.get_index_of(feature) {
                manager.var(index as i32)
            } else {
                features.insert(feature.clone());
                manager.var(features.get_index_of(feature).unwrap() as i32)
            }
        }
        Formula::And(operators) => operators
            .iter()
            .map(|operand| translate_guard(manager, features, operand))
            .fold(manager.one(), |left, right| left.and(&right)),
        Formula::Or(operators) => operators
            .iter()
            .map(|operand| translate_guard(manager, features, operand))
            .fold(manager.zero(), |left, right| left.or(&right)),
        Formula::Not(operand) => translate_guard(manager, features, operand).complement(),
        Formula::True => manager.one(),
        Formula::False => manager.zero(),
        Formula::Xor(operators) => operators
            .iter()
            .map(|operand| translate_guard(manager, features, operand))
            .fold(manager.zero(), |left, right| left.xor(&right)),
    }
}

fn print_ts_info<
    TS: TsRef + States + InitialStates + Transitions + Successors + Predecessors + MakeDenseStateSet,
>(
    ts: TS,
) {
    println!(
        "States: {} (initial: {})",
        ts.num_states(),
        ts.num_initial_states()
    );
    println!("Transitions: {}", ts.num_transitions());
    println!("SCCs: {}", ts::algorithms::scc(ts).count());
}

pub fn is_state_deterministic<S, L: std::fmt::Debug + Eq + Hash>(
    vts: &Ts<S, L>,
    state: StateId,
) -> bool {
    let mut labels = HashSet::new();
    for outgoing in vts.outgoing(&state) {
        if !labels.insert(outgoing.action()) {
            println!("non-deterministic action: {:?}", outgoing.action());
            return false;
        }
    }
    true
}

pub fn is_deterministic<S, L: std::fmt::Debug + Eq + Hash>(vts: &Ts<S, L>) -> bool {
    vts.states().all(|state| is_state_deterministic(vts, state))
}

pub fn is_monotonic<Q, V, L>(vts: &Vts<Q, V, L>) -> bool
where
    V: PartiallyOrdered,
{
    vts.transitions().all(|transition| {
        let source = vts.get_label(&transition.source());
        let target = vts.get_label(&transition.target());
        target.verdict.is_at_most(&source.verdict)
    })
}

fn print_vts_info<Q, B: PartiallyOrdered, L: std::fmt::Debug + Eq + Hash>(vts: &Vts<Q, B, L>) {
    print_ts_info(vts);
    println!("Deterministic: {:?}", is_deterministic(vts));
    println!("Monotonic: {:?}", is_monotonic(vts));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepInfo {
    duration_ms: u64,
    input_ts: TsInfo,
    output_ts: TsInfo,
    data: HashMap<String, serde_json::Value>,
}

impl StepInfo {
    pub fn new(duration_ms: u64, input_ts: TsInfo, output_ts: TsInfo) -> Self {
        Self {
            duration_ms,
            input_ts,
            output_ts,
            data: HashMap::new(),
        }
    }

    pub fn set<V: Into<serde_json::Value>>(&mut self, key: &str, value: V) {
        self.data.insert(key.to_owned(), value.into());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TsInfo {
    num_states: u64,
    num_transitions: u64,
}

pub fn ts_info<T: TsRef + States + Transitions>(ts: T) -> TsInfo {
    TsInfo {
        num_states: ts.num_states() as u64,
        num_transitions: ts.num_transitions() as u64,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    input: TsInfo,
    output: TsInfo,
    data: HashMap<String, serde_json::Value>,
    steps: Vec<StepInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityMetrics {
    num_runs: u64,
    run_length: u64,
    non_live_runs: u64,
    score: f64,
    final_state: f64,
    num_runs_end: u64,
    run_length_end: f64,
}

pub fn synthesize(arguments: &Arguments) {
    let xml = std::fs::read_to_string(&arguments.model_path).unwrap();

    if let Some(output_path) = &arguments.output_path {
        std::fs::create_dir_all(output_path).unwrap();
    }

    let mut steps = Vec::new();

    let fts = frontends::vibes::from_str(&xml).unwrap();

    if let Some(output_path) = &arguments.output_path {
        ts::output::dot::write_to_file(
            &fts,
            "Input FTS",
            &output_path.join("0_input_fts.dot"),
            |_, state| state.to_string(),
            |transition| {
                ts::output::dot::TransitionAttributes::new(&format!(
                    "{} [{}]",
                    transition.action().action.as_deref().unwrap_or("τ"),
                    transition.action().guard.to_string()
                ))
            },
        )
        .unwrap();
    }

    // Extract the features of the model.
    let mut features = fts
        .transitions()
        .map(|transition| {
            transition.action().guard.traverse().filter_map(|formula| {
                if let Formula::Atom(feature) = formula {
                    Some(feature)
                } else {
                    None
                }
            })
        })
        .flatten()
        .cloned()
        .collect::<IndexSet<_>>();

    // Extract the observables of the model.
    let observables = fts
        .transitions()
        .filter_map(|transition| transition.action().action.clone())
        .filter(|action| {
            arguments
                .unobservable
                .iter()
                .all(|unobservable| !action.starts_with(unobservable.deref()))
        })
        .collect::<IndexSet<_>>();

    let actions = observables
        .iter()
        .map(|act| Some(act.clone()))
        .collect::<Vec<_>>();

    // Construct an equivalent FTS using BDDs as guards.
    let manager = BddManager::with_vars(features.len() as u32);
    // manager.enable_reordering(cudd::ReorderingType::Sift);

    let feature_model = if let Some(feature_model_path) = &arguments.feature_model {
        let formula = std::fs::read_to_string(feature_model_path)
            .unwrap()
            .parse::<Formula<String>>()
            .unwrap();

        translate_guard(&manager, &mut features, &formula)
    } else {
        manager.one()
    };

    let fts = fts.map_labels(|label| {
        VatsLabel::new(
            label.action.clone(),
            translate_guard(&manager, &mut features, &label.guard),
        )
    });

    let atoms = features.iter().cloned().collect::<Vec<_>>();

    println!("Features: {features:?}");
    println!("\n\nConfigs: {}", feature_model.count_solutions());
    println!("\n\nObservables: {} {observables:?}\n\n", observables.len());

    println!("");
    print_ts_info(&fts);

    let manager = &manager;

    // 1️⃣ Construct the initial VTS.
    let start = Instant::now();
    let vts = algorithms::annotation_tracking(&manager, &fts, &feature_model, true);
    info!(
        "Annotation tracking completed in {:.2}s.",
        start.elapsed().as_secs_f32()
    );

    steps.push(StepInfo::new(
        start.elapsed().as_millis().try_into().unwrap(),
        ts_info(&fts),
        ts_info(&vts),
    ));

    println!("\n== Initial VTS ==");
    print_vts_info(&vts);

    if let Some(output_path) = &arguments.output_path {
        if arguments.output_state_spaces {
            ts::output::dot::write_to_file(
                &vts,
                "Initial VTS",
                &output_path.join("1_initial_vts.dot"),
                |_, state| {
                    format!(
                        "({}, {})",
                        fts.get_label(&state.control),
                        bdd_to_formula(&atoms, &state.verdict)
                    )
                },
                |transition| {
                    let mut attrs =
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"));
                    let target = vts.get_label(&transition.target());
                    let source = vts.get_label(&transition.source());
                    if !target.verdict.is_at_most(&source.verdict) {
                        attrs = attrs.with_color("red".to_owned());
                    }
                    attrs
                },
            )
            .unwrap();
        }
    }

    // 2️⃣ Refine the beliefs by lookahead propagation.
    let vts = if !arguments.with_lookahead_refinement {
        vts
    } else {
        let new_vts = algorithms::lookahead_refinement(&vts);
        drop(vts);
        let vts = new_vts;
        println!("\n== Refined VTS ==");
        print_vts_info(&vts);

        if let Some(output_path) = &arguments.output_path {
            if arguments.output_state_spaces {
                ts::output::dot::write_to_file(
                    &vts,
                    "Refined VTS",
                    &output_path.join("2_refined_vts.dot"),
                    |_, state| {
                        format!(
                            "{}",
                            // "(q = {}, B = {})",
                            //fts.get_state(&state.control),
                            bdd_to_formula(&atoms, &state.verdict)
                        )
                    },
                    |transition| {
                        let mut attrs = TransitionAttributes::new(
                            transition.action().as_deref().unwrap_or("τ"),
                        );
                        let target = vts.get_label(&transition.target());
                        let source = vts.get_label(&transition.source());
                        if !target.verdict.is_at_most(&source.verdict) {
                            attrs = attrs.with_color("red".to_owned());
                        }
                        attrs
                    },
                )
                .unwrap();
            }
        }
        vts
    };

    let has_unobservable_transitions = vts.transitions().any(|transition| {
        transition
            .action()
            .as_ref()
            .map(|action| !observables.contains(action))
            .unwrap_or(true)
    });

    let vts = if has_unobservable_transitions {
        // 3️⃣ Remove unobservables.
        println!("\n== Observability Projection ==");
        let projection = algorithms::observability_projection(&vts, |action| {
            action
                .as_ref()
                .map(|action| observables.contains(action))
                .unwrap_or(false)
        });
        print_ts_info(&projection);

        if let Some(output_path) = &arguments.output_path {
            if arguments.output_state_spaces {
                ts::output::dot::write_to_file(
                    &projection,
                    "Observability Projection",
                    &output_path.join("3_observability_projection.dot"),
                    |_, state| {
                        state
                            .iter()
                            .map(|id| {
                                let state = vts.get_label(&id);
                                bdd_to_formula(&atoms, &state.verdict).to_string()
                                // state
                                //     .verdict
                                //     .iter()
                                //     .map(|bdd| {
                                //         format!(
                                //             "{}",
                                //             // fts.get_label(&state.control),
                                //             bdd_to_formula(&atoms, bdd)
                                //         )
                                //     })
                                //     .collect::<Vec<_>>()
                                //     .join(", ")
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
            }
        }

        // 5️⃣ Flatten beliefs.
        println!("\n== Flatten Verdicts ==");
        let new_vts = algorithms::join_verdicts(&vts, &projection, &manager.one()); // &power_set.top());
        drop(vts);
        let vts = new_vts;
        print_vts_info(&vts);

        if let Some(output_path) = &arguments.output_path {
            if arguments.output_state_spaces {
                ts::output::dot::write_to_file(
                    &vts,
                    "Flattened Verdicts",
                    &output_path.join("3_2_flattened_beliefs.dot"),
                    |_, state| {
                        bdd_to_formula(&atoms, &state.verdict).to_string()
                        // state
                        //     .verdict
                        //     .iter()
                        //     .map(|bdd| {
                        //         format!(
                        //             "{}",
                        //             // fts.get_label(&state.control),
                        //             bdd_to_formula(&atoms, bdd)
                        //         )
                        //     })
                        //     .collect::<Vec<_>>()
                        //     .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
                ts::output::dot::write_to_file(
                    &vts,
                    "Flattened Verdicts",
                    &output_path.join("3_2_flattened_beliefs_count.dot"),
                    |_, state| {
                        bdd_to_formula(&atoms, &state.verdict).to_string()
                        // state
                        //     .verdict
                        //     .iter()
                        //     .map(|bdd| bdd.count_solutions().to_string())
                        //     .collect::<Vec<_>>()
                        //     .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
            }
        }
        vts
    } else {
        println!("VTS does not contain unobservable transitions!");
        vts.map_states(|id, state| {
            let mut control = vts.make_sparse_state_set();
            control.insert(id);
            VtsState {
                control,
                verdict: state.verdict.clone(),
            }
        })
    };
    //  else {
    //     let power_set = PowerSet::new();

    //     vts.map_states(|x| VtsState::new(x.control, power_set.singleton(x.verdict.clone())))
    // };

    let is_deterministic = is_deterministic(&vts);

    let vts = if !is_deterministic {
        // 4️⃣ Determinize.
        println!("\n== Determinize ==");
        let determinized = algorithms::determinize(&vts);
        print_ts_info(&determinized);

        if let Some(output_path) = &arguments.output_path {
            if arguments.output_state_spaces {
                ts::output::dot::write_to_file(
                    &determinized,
                    "Determinized",
                    &output_path.join("4_determinized.dot"),
                    |_, state| {
                        state
                            .iter()
                            .map(|id| {
                                let state = vts.get_label(&id);
                                bdd_to_formula(&atoms, &state.verdict).to_string()
                                // state
                                //     .verdict
                                //     .iter()
                                //     .map(|bdd| {
                                //         format!(
                                //             "{}",
                                //             // fts.get_label(&state.control),
                                //             bdd_to_formula(&atoms, bdd)
                                //         )
                                //     })
                                //     .collect::<Vec<_>>()
                                //     .join(", ")
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
            }
        }

        // 5️⃣ Flatten beliefs.
        println!("\n== Flatten Verdicts ==");
        let new_vts = algorithms::join_verdicts(&vts, &determinized, &manager.one()); //&power_set.top());
        drop(vts);
        let vts = new_vts;
        print_vts_info(&vts);

        if let Some(output_path) = &arguments.output_path {
            if arguments.output_state_spaces {
                ts::output::dot::write_to_file(
                    &vts,
                    "Flattened Verdicts",
                    &output_path.join("5_flattened_beliefs.dot"),
                    |_, state| {
                        bdd_to_formula(&atoms, &state.verdict).to_string()
                        // state
                        //     .verdict
                        //     .iter()
                        //     .map(|bdd| {
                        //         format!(
                        //             "{}",
                        //             // fts.get_label(&state.control),
                        //             bdd_to_formula(&atoms, bdd)
                        //         )
                        //     })
                        //     .collect::<Vec<_>>()
                        //     .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
                ts::output::dot::write_to_file(
                    &vts,
                    "Flattened Verdicts",
                    &output_path.join("5_flattened_beliefs_count.dot"),
                    |_, state| {
                        bdd_to_formula(&atoms, &state.verdict).to_string()
                        // state
                        //     .verdict
                        //     .iter()
                        //     .map(|bdd| bdd.count_solutions().to_string())
                        //     .collect::<Vec<_>>()
                        //     .join(", ")
                    },
                    |transition| {
                        TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                    },
                )
                .unwrap();
            }
        }
        vts
    } else {
        println!("VTS is already deterministic!");
        vts.map_states(|id, state| {
            let mut control = vts.make_sparse_state_set();
            control.insert(id);
            VtsState {
                control,
                verdict: state.verdict.clone(),
            }
        })
    };

    // 5️⃣ Minimize.
    println!("\n== Minimize ==");
    let start = Instant::now();
    let old_vts = vts;
    print_vts_info(&old_vts);
    type Minimize<'cx, Q, V, A> = algorithms::minimize::MinimizeFast<'cx, Q, V, A>;
    // type Minimize<'cx, Q, V, A> = algorithms::Minimize<'cx, Q, V, A>;
    let vts = Minimize::new(&old_vts)
        .with_language_insensitive(arguments.minimization.relax_language)
        .run();
    print_vts_info(&vts);

    steps.push(StepInfo::new(
        start.elapsed().as_millis().try_into().unwrap(),
        ts_info(&old_vts),
        ts_info(&vts),
    ));

    if let Some(output_path) = &arguments.output_path {
        if arguments.output_state_spaces {
            ts::output::dot::write_to_file(
                &vts,
                "Minimized",
                &output_path.join("6_minimized.dot"),
                |_, state| {
                    bdd_to_formula(&atoms, &state.verdict).to_string()
                    // state
                    //     .verdict
                    //     .iter()
                    //     .map(|bdd| {
                    //         format!(
                    //             "{}",
                    //             // fts.get_label(&state.control),
                    //             bdd_to_formula(&atoms, bdd)
                    //         )
                    //     })
                    //     .collect::<Vec<_>>()
                    //     .join(", ")
                },
                |transition| {
                    TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                },
            )
            .unwrap();
            ts::output::dot::write_to_file(
                &vts,
                "Minimized",
                &output_path.join("6_minimized_count.dot"),
                |state_id, state| {
                    if !is_state_deterministic(&vts, state_id) {
                        "NON DET!!!".to_owned()
                    } else {
                        state.verdict.count_solutions().to_string()
                    }
                    // state
                    //     .verdict
                    //     .iter()
                    //     .map(|bdd| bdd.count_solutions().to_string())
                    //     .collect::<Vec<_>>()
                    //     .join(", ")
                },
                |transition| {
                    TransitionAttributes::new(transition.action().as_deref().unwrap_or("τ"))
                },
            )
            .unwrap();
        }

        let mut data = HashMap::new();

        data.insert(
            "numConfigs".to_owned(),
            (feature_model.count_solutions() as u64).into(),
        );

        data.insert("numActions".to_owned(), actions.len().into());

        data.insert(
            "minConfigs".to_owned(),
            vts.states()
                .map(|state| vts.get_label(&state).verdict.count_solutions() as u64)
                .min()
                .into(),
        );

        let report = Report {
            steps,
            data,
            output: ts_info(&vts),
            input: ts_info(&fts),
        };

        std::fs::write(
            output_path.join("report.json"),
            serde_json::to_string_pretty(&report).unwrap(),
        )
        .unwrap();
    }

    let monitor = vts;

    if !arguments.simulate {
        return;
    }

    println!("\n== Simulation ==");

    let num_runs = 160_000;
    let run_length = arguments.steps.unwrap_or(1_000);

    let mut monitor_last_step_sum = 0;
    let mut config_sum = 0.0;
    let mut num_runs_end = 0;
    let mut run_length_end_sum = 0;

    let mut non_live_runs = 0;

    for run in 0..num_runs {
        if run % 5_000 == 0 {
            println!("Run {}.", run);
        }
        let config = feature_model.sample_solution().0;
        // let mut enabled = HashSet::new();
        // for feature in &arguments.feature {
        //     config = config.and(&manager.var(features.get_index_of(feature).unwrap() as i32));
        //     enabled.insert(feature);
        // }
        // for feature in &features {
        //     if !enabled.contains(feature) {
        //         config = config.and(
        //             &manager
        //                 .var(features.get_index_of(feature).unwrap() as i32)
        //                 .complement(),
        //         );
        //     }
        // }

        // println!("Enabled Features: {:?}", enabled);
        // println!("Config: {}", bdd_to_formula(&atoms, &config));

        let mut monitor_state = monitor.initial_states().next().unwrap();
        let mut sys_state = fts.initial_states().next().unwrap();

        let mut monitor_last_step = 0;
        let mut reached_final = false;

        let mut non_live = false;

        'outer: for step in 0..run_length {
            // println!("# Step: {step}");
            // println!(
            //     "Verdict: {}",
            //     bdd_to_formula(&atoms, &monitor.get_label(&monitor_state).verdict)
            // );
            // println!(
            //     "Configurations: {}",
            //     monitor.get_label(&monitor_state).verdict.count_solutions()
            // );

            let enabled = fts
                .outgoing(&sys_state)
                .filter(|transition| !transition.action().guard.and(&config).is_zero())
                .collect::<Vec<_>>();

            let Some(transition) = enabled.choose(&mut rand::thread_rng()) else {
                // println!("No outgoing transition! FTS is not live.");
                non_live = true;
                break;
            };

            if let Some(action) = &transition.action().action {
                let is_observable = observables.contains(action);
                // println!(
                //     "Action: {action} ({})",
                //     if is_observable {
                //         "observable"
                //     } else {
                //         "unobservable"
                //     }
                // );

                if is_observable {
                    // let mut found = false;
                    for monitor_transition in monitor.outgoing(&monitor_state) {
                        let monitor_label = monitor_transition.action();
                        if monitor_label.as_ref() == Some(action) {
                            let target = monitor_transition.target();
                            if target != monitor_state {
                                monitor_state = target;
                                monitor_last_step = step;
                                if monitor.outgoing(&monitor_state).next().is_none() {
                                    // Monitor is in final state. Nothing will happen anymore.
                                    reached_final = true;
                                    break 'outer;
                                }
                            }
                            // found = true;
                            break;
                        }
                    }
                    // if !found {
                    //     println!("Monitor transition not found!");
                    // }
                }
            }

            sys_state = transition.target();
        }

        config_sum += monitor.get_label(&monitor_state).verdict.count_solutions();
        monitor_last_step_sum += monitor_last_step;
        if reached_final {
            num_runs_end += 1;
            run_length_end_sum += monitor_last_step;
        }

        if non_live {
            non_live_runs += 1;
        }
    }

    // let config_count = feature_model.count_solutions();

    println!("Quality: {}", config_sum / (num_runs as f64));

    if let Some(output_path) = &arguments.output_path {
        let quality_metrics = QualityMetrics {
            num_runs,
            non_live_runs: non_live_runs,
            run_length: run_length as u64,
            score: 1.0
                - (((config_sum) / (num_runs as f64) - 1.0)
                    / (feature_model.count_solutions() - 1.0))
                    .max(0.0)
                    .min(1.0),
            final_state: (monitor_last_step_sum as f64) / (num_runs as f64),
            num_runs_end,
            run_length_end: if num_runs_end > 0 {
                (run_length_end_sum as f64) / (num_runs_end as f64)
            } else {
                0.0
            },
        };
        std::fs::write(
            output_path.join("quality-metrics.json"),
            serde_json::to_string_pretty(&quality_metrics).unwrap(),
        )
        .unwrap();
    }
}
