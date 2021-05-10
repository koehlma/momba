use pyo3::{
    exceptions::PyValueError, PyAny, PyCell, PyObject, PyRef, PyResult, Python, ToPyObject,
};

use crate::zones;

pub trait Time: 'static + momba_explore::time::Time + Sync + Send {}

impl Time for momba_explore::time::NoClocks {}
impl Time for momba_explore::time::Float64Zone {}

pub trait ConvertValuations: Sized {
    fn to_python(py: Python, valuations: Self) -> PyObject;
    fn from_python(obj: &PyAny) -> PyResult<Self>;
}

impl ConvertValuations for () {
    fn to_python(py: Python, _: Self) -> PyObject {
        py.None().to_object(py)
    }

    fn from_python(obj: &PyAny) -> PyResult<Self> {
        if obj.is_none() {
            Ok(())
        } else {
            Err(PyValueError::new_err("valuations have to be `None`"))
        }
    }
}

impl ConvertValuations for clock_zones::ZoneF64 {
    fn to_python(py: Python, valuations: Self) -> PyObject {
        PyCell::new(py, zones::PyZone::from(valuations))
            .unwrap()
            .to_object(py)
    }

    fn from_python(obj: &PyAny) -> PyResult<Self> {
        let value: PyRef<zones::PyZone> = obj.extract()?;
        value
            .zone
            .downcast_ref::<clock_zones::ZoneF64>()
            .cloned()
            .ok_or_else(|| PyValueError::new_err("valuations have to be a ZoneF64"))
    }
}
