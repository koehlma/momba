pub enum Actions {
    EdgeByIndex,
    //The edge is chosen based on its index.
    EdgeByLabel,
    //The edge is chosen based on its label.
}

pub enum Observations {
    //Specifies what is observable by the agent
    GlobalOnly,
    //Only global variables are observable
    LocalAndGlobal,
    //Local and global variables are observable
    Omniscient,
    //All (non-transient) variables are observable
}

trait ActionResolver<T: time::Time> {
    //num_actions: i64;
    fn available<'t>(self, state: State<T>) -> &'t [bool];
    fn resolve<'s, 't>(
        self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> &'t [Transition<'t, T>];
}

struct EdgeByIndexResolver {
    num_actions: i64,
    //instance: model.Instance??
}

struct EdgeByLabelResolver {
    num_actions: i64,
    action_mapping: HashMap<i64, String>,
    //instance: Explorer<>
    // the action use as integers from the delcarations
    reverse_action_mapping: HashMap<String, i64>,
}

impl EdgeByIndexResolver {
    pub fn new() {
        todo!()
    }
}

impl<T> ActionResolver<T> for EdgeByIndexResolver
where
    T: time::Time,
{
    fn available<'t>(self, state: State<T>) -> &'t [bool] {
        todo!()
    }
    fn resolve<'s, 't>(
        self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> &'t [Transition<'t, T>] {
        todo!()
    }
}

impl EdgeByLabelResolver {
    pub fn new() {
        todo!()
    }
}

impl<T> ActionResolver<T> for EdgeByLabelResolver
where
    T: time::Time,
{
    fn available<'t>(self, state: State<T>) -> &'t [bool] {
        todo!()
    }
    fn resolve<'s, 't>(
        self,
        transitions: &'t [Transition<'s, T>],
        action: i64,
    ) -> &'t [Transition<'t, T>] {
        todo!()
    }
}

fn create_action_resolver() {
    todo!();
}

fn count_features() {
    todo!()
}

fn extend_state_vector() {
    todo!()
}

struct Rewards {
    //Specifies the rewards for a reachability objective.
    goal_reached: f64,
    //The reward when a goal state is reached.
    dead_end: f64,
    //The reward when a dead end or bad state is reached.
    step_taken: f64,
    //The reward when a valid decision has been taken.
    invalid_action: f64,
    //The reward when an invalid decision has been taken.
}
impl Rewards {
    pub fn new() -> Self {
        Rewards {
            goal_reached: 100.0,
            dead_end: -100.0,
            step_taken: 0.0,
            invalid_action: -100.0,
        }
    }
}

struct Context<T>
where
    T: time::Time,
{
    _phantom_bound: std::marker::PhantomData<T>,
    explorer: Explorer<T>,
    rewards: Rewards,
}

//I think that it should set the time here
struct GenericExplorer<T>
where
    T: time::Time,
{
    context: Context<T>,
    state: State<T>,
}

impl<T: time::Time> GenericExplorer<T> {
    pub fn available_transitions<'t>(&self) -> &'t [Transition<'t, T>] {
        todo!()
        /*
        if terminated, return empty
        else enumerate
         */
    }
}