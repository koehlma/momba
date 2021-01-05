use std::mem;
use std::sync::Arc;

use pyo3::conversion::IntoPy;
use pyo3::prelude::*;

use momba_explore;

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
struct MDPState {
    explorer: Arc<momba_explore::MDPExplorer>,
    state: momba_explore::State<
        <momba_explore::time::NoClocks as momba_explore::time::TimeType>::Valuations,
    >,
}

#[pymethods]
impl MDPState {
    fn transitions(&self) -> Vec<MDPTransition> {
        self.explorer
            .transitions(&self.state)
            .into_iter()
            .map(|transition| unsafe {
                // This is safe because `transition` borrows from `self.explorer`.
                MDPTransition::new(self.explorer.clone(), transition)
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
struct MDPTransition {
    /// In lack of a better alternative, we use 'static here. The transition actually
    /// borrows from the `MDPExplorer` owned by the `Arc` stored in `explorer`.
    ///
    /// The order of the fields is important for safety on dropping!
    unsafe_transition: momba_explore::Transition<'static, momba_explore::time::NoClocks>,
    explorer: Arc<momba_explore::MDPExplorer>,
}

impl MDPTransition {
    /// Constructs a new transition.
    ///
    /// # Safety
    /// The transition has to borrow from the MDPExplorer owned by the `Arc`.
    unsafe fn new<'e>(
        explorer: Arc<momba_explore::MDPExplorer>,
        transition: momba_explore::Transition<'e, momba_explore::time::NoClocks>,
    ) -> Self {
        MDPTransition {
            unsafe_transition: mem::transmute(transition),
            explorer: explorer.clone(),
        }
    }
}

impl MDPTransition {
    fn transition<'t>(
        &'t self,
    ) -> &'t momba_explore::Transition<'t, momba_explore::time::NoClocks> {
        &self.unsafe_transition
    }
}

#[pymethods]
impl MDPTransition {
    fn destinations(&self, state: &MDPState) -> Vec<MDPDestination> {
        self.explorer
            .destinations(&state.state, self.transition())
            .into_iter()
            .map(|destination| unsafe { MDPDestination::new(&self.explorer, destination) })
            .collect()
    }

    fn result_action(&self) -> Action {
        Action {
            explorer: self.explorer.clone(),
            action: self.transition().result_action().clone(),
        }
    }
}

#[pyclass]
struct MDPDestination {
    unsafe_destination: momba_explore::Destination<'static, momba_explore::time::NoClocks>,
    explorer: Arc<momba_explore::MDPExplorer>,
}

impl MDPDestination {
    unsafe fn new<'e>(
        explorer: &'e Arc<momba_explore::MDPExplorer>,
        destination: momba_explore::Destination<'e, momba_explore::time::NoClocks>,
    ) -> Self {
        MDPDestination {
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
impl MDPDestination {
    fn probability(&self) -> f64 {
        self.destination().probability()
    }

    fn successor(&self, state: &MDPState, transition: &MDPTransition) -> MDPState {
        MDPState {
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
struct MDPExplorer {
    explorer: Arc<momba_explore::MDPExplorer>,
}

#[pymethods]
impl MDPExplorer {
    #[new]
    fn new(json_representation: &str) -> Self {
        MDPExplorer {
            explorer: Arc::new(momba_explore::MDPExplorer::new(
                serde_json::from_str(json_representation).expect("Error while reading model file!"),
            )),
        }
    }

    fn initial_states(&self) -> Vec<MDPState> {
        self.explorer
            .initial_states()
            .into_iter()
            .map(|state| MDPState {
                explorer: self.explorer.clone(),
                state: state,
            })
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn momba_engine(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<MDPExplorer>()?;
    module.add_class::<MDPState>()?;
    module.add_class::<MDPTransition>()?;
    module.add_class::<MDPDestination>()?;

    module.add_class::<Action>()?;

    Ok(())
}
