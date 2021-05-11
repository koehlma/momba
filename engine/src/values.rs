use pyo3::{IntoPy, PyObject, Python};

pub struct Value(pub(crate) momba_explore::model::Value);

impl From<momba_explore::model::Value> for Value {
    fn from(value: momba_explore::model::Value) -> Self {
        Self(value)
    }
}

impl IntoPy<PyObject> for Value {
    fn into_py(self, py: Python) -> PyObject {
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
