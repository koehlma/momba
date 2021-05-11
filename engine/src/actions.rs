use std::sync::Arc;

use pyo3::{IntoPy, PyObject, Python};

use crate::{time::Time, values};

#[derive(Clone)]
pub struct Action<T: Time> {
    pub explorer: Arc<momba_explore::Explorer<T>>,
    pub action: momba_explore::Action,
}

pub trait DynAction: Send + Sync {
    fn is_silent(&self) -> bool;
    fn is_labeled(&self) -> bool;

    fn label(&self) -> Option<String>;

    fn arguments(&self, py: Python) -> Vec<PyObject>;
}

impl<T: Time> DynAction for Action<T> {
    fn is_silent(&self) -> bool {
        matches!(self.action, momba_explore::Action::Silent)
    }

    fn is_labeled(&self) -> bool {
        matches!(self.action, momba_explore::Action::Labeled(_))
    }

    fn label(&self) -> Option<String> {
        match &self.action {
            momba_explore::Action::Silent => None,
            momba_explore::Action::Labeled(labeled) => {
                labeled.label(&self.explorer.network).cloned()
            }
        }
    }

    fn arguments(&self, py: Python) -> Vec<PyObject> {
        self.action
            .arguments()
            .iter()
            .map(|value| values::Value::from(value.clone()).into_py(py))
            .collect()
    }
}
