use hashbrown::HashMap;
use momba_model::{expressions::Identifier, values::Value};

#[derive(Debug, Default)]
pub struct Params(HashMap<Identifier, Value>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: impl Into<String>, value: Value) {
        self.0.insert(Identifier::from(name.into()), value);
    }

    pub fn get(&self, name: &Identifier) -> Option<&Value> {
        self.0.get(name)
    }
}
