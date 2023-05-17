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
//May be useful. edge.number contains a number from the jani model.
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

                //println!("{:#?}", v);
                //self.get_instance();
                //match v {
                //    (id, value) => available_actions.insert(*value as i64),
                //    (_, _) => {
                //        continue;
                //    } //panic!("Error, only 1 instance of automaton"),
                //};
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
                    remove_trans_idxs.push(i);
                };
            }
        }
        //for v in t.numeric_reference_vector().into_iter() {
        // match v {
        //     (0, value) => {
        //         //println!(
        //         //    "Action: {:?} vs num ref value value: {:?}",
        //         //    action, value
        //         //);
        //         if action as usize != value {
        //             remove_trans_idxs.push(i);
        //         }
        //     }
        //     (_, _) => panic!("Error, only 1 instance of automaton"),

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
                //match ins {
                //    0 => actions.push(value as i64),
                //    _ => continue,
                //}
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
            .filter(|(k, _v)| keep_actions.contains(k))
            .map(|(k, v)| (*k, *v))
            .collect();

        // Take the action with the highest score
        let mut max_val: f64 = 0.0;
        let mut max_key: i64 = -1;
        for (k, v) in filtered_map.iter() {
            if *v > max_val {
                max_key = *k;
                max_val = *v;
            }
        }
        //Filter the transitions. Remains only the ones with the max key on the
        //numeric ref vector
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

/*
//pub struct Context<'a, T, A>
pub struct Context<'a, T>
where
    T: time::Time,
    //A: ActionResolver<T>,
{
    explorer: &'a Explorer<T>,
    _rewards: Rewards,
    _actions: Actions,
    observations: Observations,
    //action_resolver: &'a A,
    action_resolver: EdgeByIndexResolver<T>,
    global_variables: Vec<String>,
    local_variables: Vec<String>,
    other_variables: HashMap<String, Vec<String>>,
}

impl<'a, T> Context<'a, T>
//impl<'a, T, A> Context<'a, T, A>
where
    T: time::Time,
    //A: ActionResolver<T>,
{
    pub fn new(explorer: &'a Explorer<T>, actions: Actions, observations: Observations) -> Self {
        let action_resolver = match actions {
            Actions::EdgeByIndex => panic!("Not yet implemented the edge by index resolver"), //EdgeByIndexResolver::new(explorer),
            Actions::EdgeByLabel => panic!("Not yet implemented the edge by label resolver"),
        };

        let mut _global_variables = vec![];
        for (s, _) in &explorer.network.declarations.global_variables {
            _global_variables.push(s.into());
        }

        let mut local_variables = vec![];
        if vec![Observations::GlobalOnly, Observations::Omniscient].contains(&observations) {
            for (s, _) in &explorer.network.declarations.transient_variables {
                local_variables.push(s.into());
            }
        }
        let other_variables: HashMap<String, Vec<String>> = HashMap::new();
        if observations == Observations::Omniscient {
            // TODO: Check this, i can know the controlled instance via a new argument on cmd.
            // But the issue is when trying to get the string from the automaton.
            //for (idx, aut) in &explorer.network.automata{
            //    let mut ins_variables: Vec<String> = vec![];
            //    for d in explorer.network.
            //}
        }

        Context {
            explorer,
            _rewards: Rewards::new(),
            _actions: actions,
            observations,
            action_resolver,
            global_variables: _global_variables,
            local_variables,
            other_variables,
        }
    }
}

#[derive(Debug)]
pub enum StepOutput {
    GoalReached,
    DeadEnd,
    InvalidAction,
    StepTaken,
}
//pub struct GenericExplorer<'a, T, G, A>
pub struct GenericExplorer<'a, T, G>
where
    T: time::Time,
    //A: ActionResolver<T>,
{
    context: Context<'a, T>,
    pub state: State<T>,
    goal: G,
}

//impl<'a, T, G, A> GenericExplorer<'a, T, G, A>
impl<'a, T, G> GenericExplorer<'a, T, G>
where
    T: time::Time,
    G: Fn(&State<T>) -> bool,
    //A: ActionResolver<T>,
{
    pub fn new(context: Context<'a, T>, goal: G) -> Self {
        let init_state = (&context)
            .explorer
            .initial_states()
            .into_iter()
            .next()
            .unwrap();
        GenericExplorer {
            context,
            state: init_state,
            goal,
        }
    }
    pub fn has_terminated(&self) -> bool {
        let transitions = self.context.explorer.transitions(&self.state);
        self.has_reached_goal() || transitions.len() == 0
        //Should add the is dead predicate, that can be another function.
    }
    pub fn has_choice(&self) -> bool {
        let transitions = self.context.explorer.transitions(&self.state);
        let has_choice = transitions.len() > 1;
        has_choice
    }
    pub fn has_reached_goal(&self) -> bool {
        (self.goal)(&self.state)
    }

    fn _explore_until_choice(&mut self) {
        while !self.has_choice() && !self.has_terminated() {
            let mut rng = rand::thread_rng();
            let all_transitions = self.context.explorer.transitions(&self.state);
            if all_transitions.len() > 1 {
                println!("Uncontrolled nondeterminism has been resolved uniformly.")
            }
            let transition = all_transitions.into_iter().choose(&mut rng).unwrap();
            let destination = self
                .context
                .explorer
                .destinations(&self.state, &transition)
                .into_iter()
                .choose(&mut rng)
                .unwrap();
            self.state = self
                .context
                .explorer
                .successor(&self.state, &transition, &destination);
        }
    }

    fn _available_actions(&self) -> Vec<bool> {
        let mut out = vec![];
        self.context
            .action_resolver
            .available_v0(&self.state, &mut out);
        out
    }

    pub fn step(&mut self, action: i64) -> StepOutput {
        let mut rng = rand::thread_rng();
        if self.has_terminated() {
            if self.has_reached_goal() {
                return StepOutput::GoalReached;
            } else {
                return StepOutput::DeadEnd;
            }
        }
        let all_transitions = self.context.explorer.transitions(&self.state);
        let selected_transitions = self
            .context
            .action_resolver
            .resolve_v0(&all_transitions, action);
        if selected_transitions.is_empty() {
            return StepOutput::InvalidAction;
        } else {
            if selected_transitions.len() > 1 {
                println!("Uncontrolled nondeterminism has been resolved uniformly.")
            }
            let transition = selected_transitions.into_iter().choose(&mut rng).unwrap();
            let destination = self
                .context
                .explorer
                .destinations(&self.state, transition)
                .into_iter()
                .choose(&mut rng)
                .unwrap();
            self.state = self
                .context
                .explorer
                .successor(&self.state, &transition, &destination);
            return StepOutput::StepTaken;
        }
    }

    pub fn uniform_step(&mut self) -> StepOutput {
        if self.has_terminated() {
            if self.has_reached_goal() {
                return StepOutput::GoalReached;
            } else {
                return StepOutput::DeadEnd;
            }
        }
        let mut rng = rand::thread_rng();
        let all_transitions = self.context.explorer.transitions(&self.state);
        let transition = all_transitions.into_iter().choose(&mut rng).unwrap();
        let destination = self
            .context
            .explorer
            .destinations(&self.state, &transition)
            .into_iter()
            .choose(&mut rng)
            .unwrap();
        self.state = self
            .context
            .explorer
            .successor(&self.state, &transition, &destination);
        return StepOutput::StepTaken;
    }

    /*
    Transform the state into a vectorial representation using the local and
    global variables of the model. Then, cast the vector into a tch Tensor.
    */
    pub fn state_to_tensor(&self, state: &State<T>, size: usize) -> Tensor {
        let input_size = size;
        let mut tensor = Tensor::empty(&[input_size as i64], DOUBLE_CPU);

        //for id in &self.context.explorer.network.declarations.global_variables {
        for id in &self.context.global_variables {
            let g_value = state.get_global_value(&self.context.explorer, &id).unwrap();
            tensor = match g_value.get_type() {
                model::Type::Bool => panic!("Tensor type not valid"),
                //tensor.f_add_scalar(g_value.unwrap_bool()).unwrap(),
                model::Type::Float64 => tensor
                    .f_add_scalar(g_value.unwrap_float64().into_inner())
                    .unwrap(),
                model::Type::Int64 => tensor.f_add_scalar(g_value.unwrap_int64()).unwrap(),
                model::Type::Vector { element_type: _ } => panic!("Tensor type not valid"),
                //tensor.f_add_(g_value.unwrap_vector()).unwrap(),
                model::Type::Unknown => panic!("Tensor type not valid"),
            };
        }

        //for id in &self.context.explorer.network.declarations.transient_variables {
        for id in &self.context.local_variables {
            let l_value = state
                .get_transient_value(&self.context.explorer.network, id)
                .unwrap_int64();
            tensor = tensor.f_add_scalar(l_value).unwrap();
        }
        if size != tensor.size()[0] as usize {
            panic!("Vector size and NN input size does not match")
        };
        tensor
    }

    /*
    Interpret the vector according to the what it means.
    Currently, uses argmax to find the index of the action with the highest score.
    */
    pub fn tensor_to_action(&self, tensor: Tensor) -> i64 {
        let idx = tensor.argmax(None, true).int64_value(&[0]);
        //println!("Value {:?} at index {:?}", tensor.double_value(&[idx]), idx);
        idx
    }

    pub fn reset(&mut self) {
        self.state = self
            .context
            .explorer
            .initial_states()
            .into_iter()
            .next()
            .unwrap();
        // In python actually creates another GenExplorer. idk why, it looks not efficient.
    }
}
*/