use std::sync::{Arc, RwLock};

use pyo3::{PyObject, Python};

use crate::{time, transitions, values};

#[derive(Clone)]
pub struct State<T: time::Time> {
    pub explorer: Arc<momba_explore::Explorer<T>>,
    pub state: Arc<momba_explore::State<T>>,
}

pub trait DynState: Send + Sync {
    fn get_global_value(&self, identifier: &str) -> Option<values::Value>;
    fn get_location_of(&self, automaton_name: &str) -> Option<String>;

    fn valuations(&self, py: Python) -> PyObject;

    fn evaluate_global_expression(
        &self,
        expr: &momba_explore::evaluate::CompiledExpression<2>,
    ) -> Option<values::Value>;

    fn transitions(&self) -> Vec<crate::PyTransition>;
}

impl<T: time::Time> DynState for State<T>
where
    T::Valuations: time::ConvertValuations,
{
    fn get_global_value(&self, identifier: &str) -> Option<values::Value> {
        self.state
            .get_global_value(&self.explorer, identifier)
            .cloned()
            .map(values::Value)
    }

    fn get_location_of(&self, automaton_name: &str) -> Option<String> {
        self.state
            .get_location_of(&self.explorer, automaton_name)
            .cloned()
    }

    fn valuations(&self, py: Python) -> PyObject {
        time::ConvertValuations::to_python(py, self.state.valuations().clone())
    }

    fn transitions(&self) -> Vec<crate::PyTransition> {
        self.explorer
            .transitions(&self.state)
            .into_iter()
            .map(|transition| crate::PyTransition {
                transition: Box::new(transitions::Transition {
                    explorer: self.explorer.clone(),
                    state: self.state.clone(),
                    transition: unsafe { std::mem::transmute(Arc::new(RwLock::new(transition))) },
                }),
            })
            .collect()
    }

    fn evaluate_global_expression(
        &self,
        expr: &momba_explore::evaluate::CompiledExpression<2>,
    ) -> Option<values::Value> {
        Some(self.state.evaluate(expr).into())
    }
}
