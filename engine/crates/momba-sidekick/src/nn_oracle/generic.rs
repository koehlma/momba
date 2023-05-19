#![allow(dead_code, unused_variables, unused_assignments)]
use hashbrown::{HashMap, HashSet};
use momba_explore::{model::Automaton, *};
use std::sync::Arc;

pub enum Actions {
    EdgeByIndex,
    //The edge is chosen based on its index.
    EdgeByLabel,
    //The edge is chosen based on its label.
}

#[derive(PartialEq, Debug)]
pub enum Observations {
    //Specifies what is observable by the agent
    GlobalOnly,
    //Only global variables are observable
    LocalAndGlobal,
    //Local and global variables are observable
    Omniscient,
    //All (non-transient) variables are observable
}

pub trait ActionResolver<T: time::Time> {
    fn available_v0(&self, state: &State<T>, out: &mut Vec<bool>);
    fn resolve_v0<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> Vec<&'t Transition<'s, T>>;
    fn resolve<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action_map: &HashMap<i64, f64>,
    ) -> Vec<&'t Transition<'s, T>>;
}

#[derive(Clone)]
pub struct EdgeByIndexResolver<T>
where
    T: time::Time,
{
    num_actions: i64,
    explorer: Arc<Explorer<T>>,
    instance: usize,
}

impl<'a, T: time::Time> EdgeByIndexResolver<T> {
    pub fn new(explorer: Arc<Explorer<T>>, name: Option<String>) -> Self {
        let default = String::from("0");
        let instance_name = format!("_{}", name.unwrap_or_default());
        let instance_id = &explorer
            .network
            .automata
            .keys()
            .filter(|name| name.ends_with(&instance_name))
            .next()
            .unwrap_or_else(|| {
                println!("Using default value for index automata: 0");
                &default
            })
            .strip_suffix(&instance_name)
            .unwrap_or("0");

        let mut num_actions: i64 = 0;
        for (_, l) in (&explorer.network.automata.values().next().unwrap().locations).into_iter() {
            num_actions += l.edges.len() as i64;
        }

        EdgeByIndexResolver {
            num_actions,
            instance: instance_id.parse::<usize>().unwrap(),
            explorer,
        }
    }

    fn get_instance(&self) -> usize {
        self.instance
    }
}

impl<'a, T> ActionResolver<T> for EdgeByIndexResolver<T>
where
    T: time::Time,
{
    fn available_v0(&self, state: &State<T>, out: &mut Vec<bool>) {
        out.clear();
        let mut available_actions: HashSet<i64> = HashSet::new();
        let id = self.get_instance();
        for t in self.explorer.transitions(&state).iter() {
            for (ins, value) in t.numeric_reference_vector().iter() {
                if *ins == id {
                    available_actions.insert(*value as i64);
                }
            }
        }
        available_actions.remove(&-1);
        for act in 0..self.num_actions {
            out.push(available_actions.contains(&act))
        }
    }

    //Available should, given a state, return the available actions for the state
    //Resolve, should given an action (the one available with the highest value) return the transitions.

    fn resolve_v0<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> Vec<&'t Transition<'s, T>> {
        let id = self.get_instance();
        let mut remove_trans_idxs = vec![];
        for (i, t) in transitions.into_iter().enumerate() {
            for (ins, value) in t.numeric_reference_vector().iter() {
                if *ins == id && !(*value == action as usize) {
                    //println!("Removing action: {:?}, at index:{:?}", value, i);
                    remove_trans_idxs.push(i);
                };
            }
        }

        let out_transitions: Vec<&Transition<T>> = transitions
            .iter()
            .enumerate()
            .filter(|(i, _)| !remove_trans_idxs.contains(i))
            .map(|(_, t)| t)
            .collect();
        /*
        This will clean all the available transitions that do not match with the
        computed action.
        println!("{:#?}", remove_trans_idxs);
        println!("len after: {:#?}. len Before: {:?}", out_transitions.len(), transitions.len());
        */
        out_transitions
    }

    fn resolve<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action_map: &HashMap<i64, f64>,
    ) -> Vec<&'t Transition<'s, T>> {
        let id = self.get_instance();
        let mut actions = vec![];
        for t in transitions.into_iter() {
            for (ins, value) in t.numeric_reference_vector().into_iter() {
                if ins == id {
                    actions.push(value as i64);
                };
            }
        }

        // Only the available actions remains on the tensor map.
        let mut keep_actions = vec![];
        for (i, _) in action_map.into_iter() {
            if actions.contains(i) {
                keep_actions.push(i);
            }
        }
        let filtered_map: HashMap<i64, f64> = action_map
            .iter()
            .filter(|(k, _v)| keep_actions.contains(k)) //_v.is_nan() ||
            .map(|(k, v)| (*k, *v))
            .collect();
        //println!("Filtered Map: {:#?}", filtered_map);
        // Take the action with the highest score
        let mut max_key: i64 = -1;
        let mut max_val: f64 = f64::NEG_INFINITY;
        for (k, v) in filtered_map.iter() {
            if *v > max_val {
                max_key = *k;
                max_val = *v;
            }
        }
        //Filter the transitions. Remains only the ones with the max key on the
        //numeric ref vector
        //println!("Choosen Action: {:?}, with Qval: {:?}.", max_key, max_val);
        let out_transitions = self.resolve_v0(transitions, max_key);
        out_transitions
    }
}

//This is basically garbage, needs a lot of love.
pub struct EdgeByLabelResolver<'a, T>
where
    T: time::Time,
{
    num_actions: i64,
    _automaton: &'a Automaton,
    action_mapping: HashMap<i64, String>,
    reverse_action_mapping: HashMap<String, i64>,
    explorer: &'a Explorer<T>,
}

impl<'a, T: time::Time> EdgeByLabelResolver<'a, T> {
    pub fn new(explorer: &'a Explorer<T>) -> Self {
        let automaton = explorer.network.automata.values().next().unwrap();
        let mut num_actions: i64 = 0;
        let mut action_types = vec![];
        let mut action_mapping: HashMap<i64, String> = HashMap::new();
        let mut rev_action_mapping: HashMap<String, i64> = HashMap::new();
        for (_, l) in (&automaton.locations).into_iter() {
            num_actions += (&l.edges).len() as i64;
            for e in (&l.edges).into_iter() {
                // Python ActionType ~~~> edges.observations.
                // Soo, i will only take the label. But I need to ask about the parameters.
                if !e.observations.is_empty() {
                    for o in (&e.observations).into_iter() {
                        action_types.push(o.label.clone())
                    }
                }
            }
            for (label, _a) in (&explorer.network.declarations.action_labels).into_iter() {
                if action_types.contains(label) {
                    action_mapping.insert(action_mapping.len() as i64, label.clone());
                    rev_action_mapping.insert(label.clone(), rev_action_mapping.len() as i64);
                }
            }
        }
        EdgeByLabelResolver {
            num_actions,
            _automaton: &automaton,
            action_mapping,
            reverse_action_mapping: rev_action_mapping,
            explorer,
        }
    }
}

impl<'a, T> ActionResolver<T> for EdgeByLabelResolver<'a, T>
where
    T: time::Time,
{
    fn available_v0(&self, state: &State<T>, out: &mut Vec<bool>) {
        out.clear();
        let mut available_actions: HashSet<i64> = HashSet::new();
        for t in self.explorer.transitions(&state).iter() {
            for e in t.edges().into_iter() {
                //index for the label
                available_actions.insert(e.index as i64);
            }
        }
        available_actions.remove(&-1);
        //MM check this part!
        for act in 0..self.num_actions {
            out.push(available_actions.contains(&act))
        }
        todo!()
    }
    // Im really not sure about this way of playing with labels/actions/index.
    // Otherway, may be the labeldActions thing
    fn resolve_v0<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> Vec<&'t Transition<'s, T>> {
        let mut remove_trans_idxs = vec![];
        // No need to compare with the string, because already got the
        // index of the action in the edge reference.
        //let action_type;
        //match self.action_mapping.get(&action) {
        //    None => panic!("something went wrong"),
        //    Some(s) => action_type = s,
        //}
        for (i, t) in transitions.into_iter().enumerate() {
            //The edges() method gives me the one associated with the instance already.
            for e in t.edges().into_iter() {
                let instance_action = e.index;
                if instance_action != action as usize {
                    remove_trans_idxs.push(i)
                }
            }
        }
        let transitions = transitions
            .iter()
            .enumerate()
            .filter(|(i, _)| remove_trans_idxs.contains(i))
            .map(|(_, t)| t)
            .collect();
        transitions
        //todo!()
    }
    fn resolve<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action_map: &HashMap<i64, f64>,
    ) -> Vec<&'t Transition<'s, T>> {
        todo!()
    }
}

struct Rewards {
    //Specifies the rewards for a reachability objective.
    _goal_reached: f64,
    //The reward when a goal state is reached.
    _dead_end: f64,
    //The reward when a dead end or bad state is reached.
    _step_taken: f64,
    //The reward when a valid decision has been taken.
    _invalid_action: f64,
    //The reward when an invalid decision has been taken.
}
impl Rewards {
    pub fn new() -> Self {
        Rewards {
            _goal_reached: 100.0,
            _dead_end: -100.0,
            _step_taken: 0.0,
            _invalid_action: -100.0,
        }
    }
}
