use std::sync::Arc;

use downcast_rs::{impl_downcast, DowncastSync};

use pyo3::{PyObject, Python};

use crate::{
    actions,
    time::{ConvertValuations, Time},
    PyAction,
};

#[derive(Clone)]
pub struct Transition<T: Time> {
    pub explorer: Arc<momba_explore::Explorer<T>>,
    pub state: Arc<momba_explore::State<T>>,
    pub transition: Arc<momba_explore::DetachedTransition<T>>,
}

pub trait DynTransition: DowncastSync {
    fn action(&self) -> PyAction;

    fn valuations(&self, py: Python) -> PyObject;
}

impl_downcast!(sync DynTransition);

impl<T: Time> DynTransition for Transition<T>
where
    T::Valuations: ConvertValuations,
{
    fn action(&self) -> PyAction {
        actions::Action {
            explorer: self.explorer.clone(),
            action: self.transition.result_action().clone(),
        }
        .into()
    }

    fn valuations(&self, py: Python) -> PyObject {
        ConvertValuations::to_python(py, self.transition.valuations().clone())
    }
}
