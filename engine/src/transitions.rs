use std::sync::{Arc, RwLock};

use downcast_rs::{impl_downcast, DowncastSync};

use pyo3::{PyAny, PyObject, PyResult, Python};

use crate::{
    actions::{self, Action},
    destinations::Destination,
    time::{ConvertValuations, Time},
    PyAction, PyDestination,
};

pub struct Transition<T: Time> {
    pub explorer: Arc<momba_explore::Explorer<T>>,
    pub state: Arc<momba_explore::State<T>>,
    pub transition: Arc<RwLock<momba_explore::Transition<'static, T>>>,
}

pub trait DynTransition: DowncastSync {
    fn action(&self) -> PyAction;

    fn valuations(&self, py: Python) -> PyObject;

    fn edge_vector(&self) -> String;
    fn action_vector(&self) -> Vec<PyAction>;

    fn numeric_reference_vector(&self) -> Vec<(usize, usize)>;

    fn destinations(&self) -> Vec<PyDestination>;

    fn replace_valuations(&mut self, valuations: &PyAny) -> PyResult<()>;
}

impl_downcast!(sync DynTransition);

impl<T: Time> DynTransition for Transition<T>
where
    T::Valuations: ConvertValuations,
{
    fn action(&self) -> PyAction {
        actions::Action {
            explorer: self.explorer.clone(),
            action: self.transition.read().unwrap().result_action().clone(),
        }
        .into()
    }

    fn edge_vector(&self) -> String {
        serde_json::to_string(&self.transition.read().unwrap().edges()).unwrap()
    }

    fn numeric_reference_vector(&self) -> Vec<(usize, usize)> {
        self.transition.read().unwrap().numeric_reference_vector()
    }

    fn action_vector(&self) -> Vec<PyAction> {
        self.transition
            .read()
            .unwrap()
            .local_actions()
            .iter()
            .map(|action| {
                Action {
                    explorer: self.explorer.clone(),
                    action: action.clone(),
                }
                .into()
            })
            .collect()
    }

    fn valuations(&self, py: Python) -> PyObject {
        ConvertValuations::to_python(py, self.transition.read().unwrap().valuations().clone())
    }

    fn destinations(&self) -> Vec<PyDestination> {
        self.explorer
            .destinations(&self.state, &self.transition.read().unwrap())
            .into_iter()
            .map(|destination| {
                Destination {
                    explorer: self.explorer.clone(),
                    state: self.state.clone(),
                    transition: self.transition.clone(),
                    destination: Arc::new(unsafe { std::mem::transmute(destination) }),
                }
                .into()
            })
            .collect()
    }

    fn replace_valuations(&mut self, valuations: &PyAny) -> PyResult<()> {
        let valuations = ConvertValuations::from_python(valuations)?;
        self.transition.write().unwrap().set_valuations(valuations);
        Ok(())
    }
}
