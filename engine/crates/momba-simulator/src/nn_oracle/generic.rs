#[allow(dead_code)]
use hashbrown::HashMap;
use momba_explore::{model::Automaton, *};
use std::sync::Arc;

// pub enum Actions {
//     EdgeByIndex,
//     //The edge is chosen based on its index.
//     EdgeByLabel,
//     //The edge is chosen based on its label.
// }

// #[derive(PartialEq, Debug)]
// pub enum Observations {
//     //Specifies what is observable by the agent
//     GlobalOnly,
//     //Only global variables are observable
//     LocalAndGlobal,
//     //Local and global variables are observable
//     Omniscient,
//     //All (non-transient) variables are observable
// }

pub trait ActionResolver<T: time::Time> {
    /// Available puts on the vector out boolean values indicating which
    /// index of transitions are available from that state.
    fn available(&self, state: &State<T>, out: &mut Vec<bool>);
    /// Resolve takes a set of transitions and an action, and returns a
    /// vector with the transitions that can be done with that action.
    fn resolve<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> Vec<&'t Transition<'s, T>>;
}

//--- By index resolver.
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
    fn available(&self, state: &State<T>, out: &mut Vec<bool>) {
        out.clear();
        let id = self.get_instance();
        let mut actions: Vec<usize> = vec![];
        for t in self.explorer.transitions(&state).into_iter() {
            actions.append(
                &mut t
                    .numeric_reference_vector()
                    .into_iter()
                    .filter(|(ins, _)| *ins == id)
                    .map(|(_, act)| act)
                    .collect(),
            )
        }
        for act in 0..self.num_actions {
            out.push(actions.contains(&(act as usize)))
        }
    }

    fn resolve<'s, 't>(
        &self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> Vec<&'t Transition<'s, T>> {
        let id = self.get_instance();
        let action = action as usize;
        let mut keep_idx: Vec<usize> = vec![];
        for (i, t) in transitions.into_iter().enumerate() {
            let actions_on_transition: Vec<(usize, usize)> = t
                .numeric_reference_vector()
                .into_iter()
                .filter(|(ins, act)| *ins == id && *act == action)
                .collect();
            // We keep the idx of the transitions such that they have an available action
            if !actions_on_transition.is_empty() {
                keep_idx.push(i)
            }
        }

        transitions
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep_idx.contains(i))
            .map(|(_, t)| t)
            .collect()
    }
}
//--- By Label resolver.
pub struct EdgeByLabelResolver<'a, T>
where
    T: time::Time,
{
    _num_actions: i64,
    _automaton: &'a Automaton,
    _action_mapping: HashMap<i64, String>,
    _reverse_action_mapping: HashMap<String, i64>,
    _explorer: &'a Explorer<T>,
}

impl<'a, T: time::Time> EdgeByLabelResolver<'a, T> {
    pub fn _new(explorer: &'a Explorer<T>) -> Self {
        let automaton = explorer.network.automata.values().next().unwrap();
        let mut num_actions: i64 = 0;
        let mut action_types = vec![];
        let mut action_mapping: HashMap<i64, String> = HashMap::new();
        let mut rev_action_mapping: HashMap<String, i64> = HashMap::new();
        for (_, l) in (&automaton.locations).into_iter() {
            num_actions += (&l.edges).len() as i64;
            for e in (&l.edges).into_iter() {
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
            _num_actions: num_actions,
            _automaton: &automaton,
            _action_mapping: action_mapping,
            _reverse_action_mapping: rev_action_mapping,
            _explorer: explorer,
        }
    }
}

impl<'a, T> ActionResolver<T> for EdgeByLabelResolver<'a, T>
where
    T: time::Time,
{
    fn available(&self, _state: &State<T>, _out: &mut Vec<bool>) {
        todo!()
    }

    fn resolve<'s, 't>(
        &self,
        _transitions: &'t [Transition<'s, T>],
        _action: i64,
    ) -> Vec<&'t Transition<'s, T>> {
        todo!()
    }
}
