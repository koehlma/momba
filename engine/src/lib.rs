use std::sync::Arc;

use pyo3::prelude::*;
use transitions::DynTransition;

pub mod actions;
pub mod destinations;
pub mod explorer;
pub mod states;
pub mod time;
pub mod transitions;
pub mod values;
pub mod zones;

#[pyclass(name = "Action")]
pub struct PyAction {
    action: Arc<dyn actions::DynAction>,
}

#[pymethods]
impl PyAction {
    fn is_silent(&self) -> bool {
        self.action.is_silent()
    }

    fn is_labeled(&self) -> bool {
        self.action.is_labeled()
    }

    fn label(&self) -> Option<String> {
        self.action.label()
    }

    fn arguments(&self, py: Python) -> Vec<PyObject> {
        self.action.arguments(py)
    }
}

impl<T: time::Time> From<actions::Action<T>> for PyAction {
    fn from(action: actions::Action<T>) -> Self {
        PyAction {
            action: Arc::new(action),
        }
    }
}

#[pyclass(name = "Destination")]
pub struct PyDestination {
    destination: Arc<dyn destinations::DynDestination>,
}

impl<T: time::Time> From<destinations::Destination<T>> for PyDestination
where
    T::Valuations: time::ConvertValuations,
{
    fn from(destination: destinations::Destination<T>) -> Self {
        PyDestination {
            destination: Arc::new(destination),
        }
    }
}

#[pymethods]
impl PyDestination {
    fn probability(&self) -> f64 {
        self.destination.probability()
    }

    fn successor(&self) -> PyState {
        self.destination.successor()
    }
}

#[pyclass(name = "State")]
pub struct PyState {
    state: Arc<dyn states::DynState>,
}

impl<T: time::Time> From<states::State<T>> for PyState
where
    T::Valuations: time::ConvertValuations,
{
    fn from(state: states::State<T>) -> Self {
        PyState {
            state: Arc::new(state),
        }
    }
}

#[pymethods]
impl PyState {
    fn get_global_value(&self, identifier: &str) -> Option<values::Value> {
        self.state.get_global_value(identifier)
    }

    fn get_location_of(&self, automaton_name: &str) -> Option<String> {
        self.state.get_location_of(automaton_name)
    }

    fn valuations(&self, py: Python) -> PyObject {
        self.state.valuations(py)
    }

    fn transitions(&self) -> Vec<PyTransition> {
        self.state.transitions()
    }
}

#[pyclass(name = "Transition")]
pub struct PyTransition {
    transition: Box<dyn transitions::DynTransition>,
}

#[pymethods]
impl PyTransition {
    pub fn valuations(&self, py: Python) -> PyObject {
        self.transition.valuations(py)
    }

    pub fn action(&self) -> PyAction {
        self.transition.action()
    }

    pub fn action_vector(&self) -> Vec<PyAction> {
        self.transition.action_vector()
    }

    pub fn edge_vector(&self) -> String {
        self.transition.edge_vector()
    }

    pub fn numeric_reference_vector(&self) -> Vec<(usize, usize)> {
        self.transition.numeric_reference_vector()
    }

    pub fn replace_valuations(&mut self, valuations: &PyAny) -> PyResult<()> {
        self.transition.replace_valuations(valuations)
    }

    pub fn destinations(&self) -> Vec<PyDestination> {
        self.transition.destinations()
    }
}

#[pyclass(name = "Explorer")]
pub struct PyExplorer {
    explorer: Arc<dyn explorer::DynExplorer>,
}

impl<E: 'static + explorer::DynExplorer> From<E> for PyExplorer {
    fn from(explorer: E) -> Self {
        PyExplorer {
            explorer: Arc::new(explorer),
        }
    }
}

#[pymethods]
impl PyExplorer {
    #[staticmethod]
    fn new_no_clocks(json_representation: &str) -> Self {
        explorer::Explorer::from(momba_explore::mdp::Explorer::new(
            serde_json::from_str(json_representation).expect("Error while reading model file!"),
        ))
        .into()
    }

    #[staticmethod]
    fn new_global_time(json_representation: &str) -> Self {
        explorer::Explorer::from(
            momba_explore::Explorer::<momba_explore::time::Float64Zone>::new(
                serde_json::from_str(json_representation).expect("Error while reading model file!"),
            ),
        )
        .into()
    }

    fn initial_states(&self) -> Vec<PyState> {
        self.explorer.initial_states()
    }

    fn count_states_and_transitions(&self) -> (usize, usize) {
        self.explorer.count_states_and_transitions()
    }

    fn compile_global_expression(&self, json_representation: &str) -> CompiledExpression {
        let expr =
            serde_json::from_str(json_representation).expect("Error while loading expression");
        self.explorer.compile_global_expression(&expr)
    }
}

#[pyclass(name = "CompiledExpression")]
pub struct CompiledExpression {
    pub expr: momba_explore::evaluate::CompiledExpression<2>,
}

#[pymethods]
impl CompiledExpression {
    fn evaluate(&self, state: &PyState) -> Option<values::Value> {
        state.state.evaluate_global_expression(&self.expr)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn momba_engine(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyExplorer>()?;
    module.add_class::<PyAction>()?;
    module.add_class::<PyState>()?;
    module.add_class::<PyTransition>()?;

    module.add_submodule(zones::zones_module(py)?)?;

    Ok(())
}
