use std::mem;
use std::sync::Arc;

use pyo3::conversion::IntoPy;
use pyo3::prelude::*;

use momba_explore;

enum WrappedExplorer {
    MDPExplorer(Arc<momba_explore::MDPExplorer>),
    Zone64Explorer(Arc<momba_explore::Explorer<momba_explore::time::Float64Zone>>),
}

struct Value(momba_explore::model::Value);

impl IntoPy<PyObject> for Value {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            momba_explore::model::Value::Int64(value) => value.into_py(py),
            momba_explore::model::Value::Float64(value) => value.into_py(py),
            momba_explore::model::Value::Bool(value) => value.into_py(py),
            momba_explore::model::Value::Vector(value) => value
                .into_iter()
                .map(|value| Value(value))
                .collect::<Vec<_>>()
                .into_py(py),
        }
    }
}

#[pyclass]
struct Action {
    explorer: Arc<momba_explore::MDPExplorer>,
    action: momba_explore::Action,
}

#[pymethods]
impl Action {
    fn is_silent(&self) -> bool {
        matches!(self.action, momba_explore::Action::Silent)
    }

    fn is_labeled(&self) -> bool {
        matches!(self.action, momba_explore::Action::Labeled(_))
    }

    fn label(&self) -> Option<&str> {
        match &self.action {
            momba_explore::Action::Labeled(labeled) => {
                Some(labeled.label(&self.explorer.network).unwrap())
            }
            _ => None,
        }
    }

    fn arguments(&self) -> Vec<Value> {
        match &self.action {
            momba_explore::Action::Labeled(labeled) => labeled
                .arguments()
                .iter()
                .map(|value| Value(value.clone()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[pyclass]
struct State {
    explorer: Arc<momba_explore::MDPExplorer>,
    state: momba_explore::State<
        <momba_explore::time::NoClocks as momba_explore::time::TimeType>::Valuations,
    >,
}

#[pymethods]
impl State {
    fn transitions(&self) -> Vec<Transition> {
        self.explorer
            .transitions(&self.state)
            .into_iter()
            .map(|transition| unsafe {
                // This is safe because `transition` borrows from `self.explorer`.
                Transition::new(self.explorer.clone(), transition)
            })
            .collect()
    }

    fn get_global_value(&self, identifier: &str) -> Option<Value> {
        if !self
            .explorer
            .network
            .declarations
            .global_variables
            .contains_key(identifier)
        {
            None
        } else {
            Some(Value(
                self.state
                    .get_global_value(&self.explorer, identifier)
                    .clone(),
            ))
        }
    }

    fn get_location_of(&self, automaton_name: &str) -> Option<&String> {
        Some(self.state.get_location_of(&self.explorer, automaton_name))
    }
}

#[pyclass]
struct Transition {
    /// In lack of a better alternative, we use 'static here. The transition actually
    /// borrows from the `MDPExplorer` owned by the `Arc` stored in `explorer`.
    ///
    /// The order of the fields is important for safety on dropping!
    unsafe_transition: momba_explore::Transition<'static, momba_explore::time::NoClocks>,
    explorer: Arc<momba_explore::MDPExplorer>,
}

impl Transition {
    /// Constructs a new transition.
    ///
    /// # Safety
    /// The transition has to borrow from the MDPExplorer owned by the `Arc`.
    unsafe fn new<'e>(
        explorer: Arc<momba_explore::MDPExplorer>,
        transition: momba_explore::Transition<'e, momba_explore::time::NoClocks>,
    ) -> Self {
        Transition {
            unsafe_transition: mem::transmute(transition),
            explorer: explorer.clone(),
        }
    }
}

impl Transition {
    fn transition<'t>(
        &'t self,
    ) -> &'t momba_explore::Transition<'t, momba_explore::time::NoClocks> {
        &self.unsafe_transition
    }
}

#[pymethods]
impl Transition {
    fn destinations(&self, state: &State) -> Vec<Destination> {
        self.explorer
            .destinations(&state.state, self.transition())
            .into_iter()
            .map(|destination| unsafe { Destination::new(&self.explorer, destination) })
            .collect()
    }

    fn result_action(&self) -> Action {
        Action {
            explorer: self.explorer.clone(),
            action: self.transition().result_action().clone(),
        }
    }

    fn action_vector(&self) -> Vec<Action> {
        self.transition()
            .local_actions()
            .iter()
            .map(|action| Action {
                explorer: self.explorer.clone(),
                action: action.clone(),
            })
            .collect()
    }

    fn edge_vector(&self) -> String {
        serde_json::to_string(&self.transition().edge_references()).unwrap()
    }
}

#[pyclass]
struct Destination {
    unsafe_destination: momba_explore::Destination<'static, momba_explore::time::NoClocks>,
    explorer: Arc<momba_explore::MDPExplorer>,
}

impl Destination {
    unsafe fn new<'e>(
        explorer: &'e Arc<momba_explore::MDPExplorer>,
        destination: momba_explore::Destination<'e, momba_explore::time::NoClocks>,
    ) -> Self {
        Destination {
            explorer: explorer.clone(),
            unsafe_destination: mem::transmute(destination),
        }
    }

    fn destination<'d>(
        &'d self,
    ) -> &'d momba_explore::Destination<'d, momba_explore::time::NoClocks> {
        &self.unsafe_destination
    }
}

#[pymethods]
impl Destination {
    fn probability(&self) -> f64 {
        self.destination().probability()
    }

    fn successor(&self, state: &State, transition: &Transition) -> State {
        State {
            explorer: self.explorer.clone(),
            state: self.explorer.successor(
                &state.state,
                transition.transition(),
                self.destination(),
            ),
        }
    }
}

#[pyclass]
struct Explorer {
    explorer: Arc<momba_explore::MDPExplorer>,
}

#[pymethods]
impl Explorer {
    #[new]
    fn new(json_representation: &str) -> Self {
        Explorer {
            explorer: Arc::new(momba_explore::MDPExplorer::new(
                serde_json::from_str(json_representation).expect("Error while reading model file!"),
            )),
        }
    }

    fn initial_states(&self) -> Vec<State> {
        self.explorer
            .initial_states()
            .into_iter()
            .map(|state| State {
                explorer: self.explorer.clone(),
                state: state,
            })
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn momba_engine(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<Explorer>()?;
    module.add_class::<State>()?;
    module.add_class::<Transition>()?;
    module.add_class::<Destination>()?;

    module.add_class::<Action>()?;

    Ok(())
}
