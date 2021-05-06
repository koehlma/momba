use pyo3::prelude::*;

pub trait Trait: 'static + Send + Sync {
    fn say_hello(&self, message: &str) -> String;
}

pub struct A;
pub struct B;

impl Trait for A {
    fn say_hello(&self, message: &str) -> String {
        format!("this is A: {}", message)
    }
}

impl Trait for B {
    fn say_hello(&self, message: &str) -> String {
        format!("this is B: {}", message)
    }
}

#[pyo3_generic_wrap::pyclass]
pub struct MyClass<T: Trait> {
    inner: T,
}

#[pyo3_generic_wrap::pymethods]
impl<T: Trait> MyClass<T>
where
    T: Send,
{
    fn say_hello(&self, message: &str) -> String {
        self.inner.say_hello(message)
    }
}

#[pymethods]
impl PyMyClass {
    #[staticmethod]
    fn new_a() -> Self {
        Self {
            wrapped: Box::new(MyClass { inner: A }),
        }
    }

    #[staticmethod]
    fn new_b() -> Self {
        Self {
            wrapped: Box::new(MyClass { inner: B }),
        }
    }
}

#[pymodule]
fn pyo3_generic_wrap_example(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyMyClass>()?;

    Ok(())
}
