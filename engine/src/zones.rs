//! Python wrapper around the [clock_zones] crate.

use std::convert::TryInto;

use downcast_rs::{impl_downcast, DowncastSync};
use ordered_float::NotNan;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyFloat};
use pyo3::{types::PyModule, PyResult, Python};

use clock_zones::{Bound, Clock, Constraint, Variable, Zone};

/// Trait to convert [Constant][clock_zones::Constant] objects from and to Python.
pub trait ConvertConstant: Sized {
    fn to_python(py: Python, constant: Self) -> PyObject;
    fn from_python(obj: &PyAny) -> PyResult<Self>;
}

impl ConvertConstant for ordered_float::NotNan<f64> {
    fn to_python(py: Python, constant: Self) -> PyObject {
        constant.into_inner().into_py(py)
    }

    fn from_python(obj: &PyAny) -> PyResult<Self> {
        PyTryFrom::try_from(obj)
            .map_err(|_| PyValueError::new_err("constant has to be a float"))
            .and_then(|val: &PyFloat| {
                NotNan::new(val.value())
                    .map_err(|_| PyValueError::new_err("float constant must not be NaN"))
            })
    }
}

impl ConvertConstant for i64 {
    fn to_python(py: Python, constant: Self) -> PyObject {
        constant.into_py(py)
    }

    fn from_python(obj: &PyAny) -> PyResult<Self> {
        <i64 as FromPyObject>::extract(obj)
    }
}

/// Python object representing a [Bound].
#[pyclass(name = "Bound")]
#[derive(Clone, Debug)]
pub struct PyBound {
    is_strict: bool,
    constant: Option<PyObject>,
}

#[pymethods]
impl PyBound {
    #[new]
    pub fn new(is_strict: bool, constant: Option<PyObject>) -> Self {
        Self {
            is_strict,
            constant,
        }
    }

    #[getter]
    pub fn is_strict(&self) -> bool {
        self.is_strict
    }

    #[getter]
    pub fn constant(&self) -> Option<&PyObject> {
        self.constant.as_ref()
    }
}

/// Python object representing a [Constraint].
#[pyclass(name = "Constraint")]
#[derive(Clone, Debug)]
pub struct PyConstraint {
    left: usize,
    right: usize,
    bound: Py<PyBound>,
}

#[pymethods]
impl PyConstraint {
    #[new]
    pub fn new(left: usize, right: usize, bound: Py<PyBound>) -> Self {
        Self { left, right, bound }
    }

    #[getter]
    pub fn left(&self) -> usize {
        self.left
    }

    #[getter]
    pub fn right(&self) -> usize {
        self.right
    }

    #[getter]
    pub fn bound(&self) -> &Py<PyBound> {
        &self.bound
    }
}

/// Trait to dynamically abstract over [Zone].
pub trait DynZone: DowncastSync + std::fmt::Debug {
    fn num_variables(&self) -> usize;
    fn num_clocks(&self) -> usize;

    fn get_bound(&self, py: Python, left: Clock, right: Clock) -> PyResult<PyBound>;

    fn is_empty(&self) -> bool;

    fn add_constraint(&mut self, constraint: &PyConstraint) -> PyResult<()>;

    fn intersect(&mut self, other: &Box<dyn DynZone>) -> PyResult<()>;

    fn future(&mut self);
    fn past(&mut self);

    fn reset(&mut self, clock: Variable, value: &PyAny) -> PyResult<()>;

    fn is_unbounded(&self, clock: Clock) -> PyResult<bool>;

    fn get_upper_bound(&self, py: Python, clock: Clock) -> PyResult<Option<PyObject>>;
    fn get_lower_bound(&self, py: Python, clock: Clock) -> PyResult<Option<PyObject>>;

    fn is_satisfied(&self, constraint: &PyConstraint) -> PyResult<bool>;

    fn includes(&self, other: &Box<dyn DynZone>) -> PyResult<bool>;

    fn resize(&self, num_variables: usize) -> Box<dyn DynZone>;

    fn check_clock(&self, clock: Clock) -> PyResult<()>;
}

impl_downcast!(sync DynZone);

macro_rules! convert_constraint {
    ($dyn_zone:expr, $constraint:expr) => {{
        let left = to_clock($constraint.left);
        let right = to_clock($constraint.right);
        DynZone::check_clock($dyn_zone, left)?;
        DynZone::check_clock($dyn_zone, right)?;
        let guard = Python::acquire_gil();
        let bound: &PyBound = &$constraint.bound.as_ref(guard.python()).borrow();
        let constant = ConvertConstant::from_python(
            bound
                .constant
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("bound constant must not be None"))?
                .extract(guard.python())?,
        )?;
        Constraint::new(left, right, Bound::new(bound.is_strict, constant))
    }};
}

fn to_clock(index: usize) -> Clock {
    if index == 0 {
        Clock::ZERO
    } else {
        Clock::variable(index - 1).into()
    }
}

impl<Z: Zone + 'static + Send + Sync + std::fmt::Debug> DynZone for Z
where
    <Z::Bound as Bound>::Constant: ConvertConstant,
{
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn num_clocks(&self) -> usize {
        self.num_clocks()
    }

    fn get_bound(&self, py: Python, left: Clock, right: Clock) -> PyResult<PyBound> {
        DynZone::check_clock(self, left)?;
        DynZone::check_clock(self, right)?;
        let bound = self.get_bound(left, right);
        Ok(PyBound {
            is_strict: bound.is_strict(),
            constant: bound
                .constant()
                .map(|constant| ConvertConstant::to_python(py, constant)),
        })
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn add_constraint(&mut self, constraint: &PyConstraint) -> PyResult<()> {
        let left = to_clock(constraint.left);
        let right = to_clock(constraint.right);
        DynZone::check_clock(self, left)?;
        DynZone::check_clock(self, right)?;
        let guard = Python::acquire_gil();
        let bound: &PyBound = &constraint.bound.as_ref(guard.python()).borrow();
        let constant = ConvertConstant::from_python(
            bound
                .constant
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("bound constant must not be None"))?
                .extract(guard.python())?,
        )?;
        self.add_constraint(Constraint::new(
            left,
            right,
            Bound::new(bound.is_strict, constant),
        ));
        Ok(())
    }

    fn intersect(&mut self, other: &Box<dyn DynZone>) -> PyResult<()> {
        if self.num_variables() != other.num_variables() {
            return Err(PyValueError::new_err(
                "zones have a different number of variables",
            ));
        }
        match other.downcast_ref::<Self>() {
            Some(other) => Ok(self.intersect(other)),
            None => Err(PyValueError::new_err("zones have different types")),
        }
    }

    fn future(&mut self) {
        self.future();
    }

    fn past(&mut self) {
        self.past();
    }

    fn reset(&mut self, clock: Variable, value: &PyAny) -> PyResult<()> {
        DynZone::check_clock(self, clock.into())?;
        self.reset(clock, ConvertConstant::from_python(value)?);
        Ok(())
    }

    fn is_unbounded(&self, clock: Clock) -> PyResult<bool> {
        DynZone::check_clock(self, clock)?;
        Ok(self.is_unbounded(clock))
    }

    fn get_upper_bound(&self, py: Python, clock: Clock) -> PyResult<Option<PyObject>> {
        DynZone::check_clock(self, clock)?;
        match self.get_upper_bound(clock) {
            Some(constant) => Ok(Some(ConvertConstant::to_python(py, constant))),
            None => Ok(None),
        }
    }

    fn get_lower_bound(&self, py: Python, clock: Clock) -> PyResult<Option<PyObject>> {
        DynZone::check_clock(self, clock)?;
        match self.get_lower_bound(clock) {
            Some(constant) => Ok(Some(ConvertConstant::to_python(py, constant))),
            None => Ok(None),
        }
    }

    fn is_satisfied(&self, constraint: &PyConstraint) -> PyResult<bool> {
        let constraint = convert_constraint!(self, constraint);
        Ok(self.is_satisfied(&constraint))
    }

    fn includes(&self, other: &Box<dyn DynZone>) -> PyResult<bool> {
        if self.num_variables() != other.num_variables() {
            return Err(PyValueError::new_err(
                "zones have a different number of variables",
            ));
        }
        match other.downcast_ref::<Self>() {
            Some(other) => Ok(self.includes(other)),
            None => Err(PyValueError::new_err("zones have different types")),
        }
    }

    fn resize(&self, num_variables: usize) -> Box<dyn DynZone> {
        Box::new(self.resize(num_variables))
    }

    fn check_clock(&self, clock: Clock) -> PyResult<()> {
        if self.check_clock(clock) {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "the provided clock does not exist on the zone",
            ))
        }
    }
}

/// Python object representing a [Zone].
#[pyclass(name = "Zone")]
pub struct PyZone {
    pub(crate) zone: Box<dyn DynZone>,
}

impl<Z: Zone + 'static + Send + Sync + std::fmt::Debug> From<Z> for PyZone
where
    <Z::Bound as Bound>::Constant: ConvertConstant,
{
    fn from(zone: Z) -> Self {
        Self {
            zone: Box::new(zone),
        }
    }
}

#[pymethods]
impl PyZone {
    #[staticmethod]
    pub fn new_i64_unconstrained(num_variables: usize) -> Self {
        clock_zones::ZoneI64::new_unconstrained(num_variables).into()
    }

    #[staticmethod]
    pub fn new_i64_zero(num_variables: usize) -> Self {
        clock_zones::ZoneI64::new_zero(num_variables).into()
    }

    #[staticmethod]
    pub fn new_f64_unconstrained(num_variables: usize) -> Self {
        clock_zones::ZoneF64::new_unconstrained(num_variables).into()
    }

    #[staticmethod]
    pub fn new_f64_zero(num_variables: usize) -> Self {
        clock_zones::ZoneF64::new_zero(num_variables).into()
    }

    #[getter]
    pub fn num_variables(&self) -> usize {
        self.zone.num_variables()
    }

    #[getter]
    pub fn num_clocks(&self) -> usize {
        self.zone.num_clocks()
    }

    fn get_constraint(&self, py: Python, left: usize, right: usize) -> PyResult<PyConstraint> {
        let bound = self.get_bound(py, left, right)?;
        Ok(PyConstraint {
            left,
            right,
            bound: Py::new(py, bound)?,
        })
    }

    fn get_bound(&self, py: Python, left: usize, right: usize) -> PyResult<PyBound> {
        self.zone.get_bound(py, to_clock(left), to_clock(right))
    }

    #[getter]
    fn is_empty(&self) -> bool {
        self.zone.is_empty()
    }

    fn add_constraint(&mut self, constraint: &PyConstraint) -> PyResult<()> {
        self.zone.add_constraint(constraint)
    }

    fn intersect(&mut self, other: &PyZone) -> PyResult<()> {
        self.zone.intersect(&other.zone)
    }

    fn future(&mut self) {
        self.zone.future()
    }

    fn past(&mut self) {
        self.zone.past()
    }

    fn reset(&mut self, clock: usize, value: &PyAny) -> PyResult<()> {
        let clock = to_clock(clock);
        let variable: Variable = clock
            .try_into()
            .map_err(|_| PyValueError::new_err("the provided clock is not a clock variable"))?;
        self.zone.reset(variable, value)
    }

    fn is_unbounded(&self, clock: usize) -> PyResult<bool> {
        self.zone.is_unbounded(to_clock(clock))
    }

    fn get_upper_bound(&self, py: Python, clock: usize) -> PyResult<Option<PyObject>> {
        self.zone.get_upper_bound(py, to_clock(clock))
    }

    fn get_lower_bound(&self, py: Python, clock: usize) -> PyResult<Option<PyObject>> {
        self.zone.get_lower_bound(py, to_clock(clock))
    }

    fn is_satisfied(&self, constraint: &PyConstraint) -> PyResult<bool> {
        self.zone.is_satisfied(constraint)
    }

    fn includes(&self, other: &PyZone) -> PyResult<bool> {
        self.zone.includes(&other.zone)
    }

    fn resize(&self, num_variables: usize) -> Self {
        Self {
            zone: self.zone.resize(num_variables),
        }
    }
}

/// Creates the `zones` Python module.
pub fn zones_module(py: Python) -> PyResult<&PyModule> {
    let module = PyModule::new(py, "zones")?;
    module.add_class::<PyZone>()?;
    module.add_class::<PyBound>()?;
    module.add_class::<PyConstraint>()?;
    Ok(module)
}
