//! Lexical scopes for looking up identifiers and function names.

use std::sync::Arc;

use hashbrown::HashMap;
use momba_model::{expressions::Identifier, functions::FunctionName, values::Value};
use parking_lot::Mutex;

use crate::values::{layout::FieldIdx, types::ValueTy, units::NumWords};

use super::{functions::FunctionIdx, ConstantIdx};

struct StackLayoutInner {
    parent: Option<StackLayout>,
    variables: Vec<StackVariable>,
    depth: NumWords,
}

impl Default for StackLayoutInner {
    fn default() -> Self {
        Self {
            parent: None,
            variables: Vec::new(),
            depth: 0.into(),
        }
    }
}

pub struct StackVariable {
    name: Identifier,
    ty: ValueTy,
    depth: NumWords,
}

struct StackLayoutBuilder {
    inner: StackLayoutInner,
}

impl StackLayoutBuilder {
    pub fn new(parent: Option<StackLayout>) -> Self {
        let depth = if let Some(parent) = &parent {
            parent.inner.depth
        } else {
            0.into()
        };
        Self {
            inner: StackLayoutInner {
                parent,
                variables: Vec::new(),
                depth,
            },
        }
    }

    pub fn add_stack_variable(&mut self, name: Identifier, ty: ValueTy) {
        let depth = self.inner.depth;
        // self.inner.depth += ty.size_words();
        self.inner.variables.push(StackVariable { name, ty, depth })
    }

    pub fn build(self) -> StackLayout {
        StackLayout {
            inner: Arc::new(self.inner),
        }
    }
}

#[derive(Default, Clone)]
pub struct StackLayout {
    inner: Arc<StackLayoutInner>,
}

impl StackLayout {
    pub fn lookup(&self, identifier: &Identifier) -> Option<&StackVariable> {
        for variable in self.inner.variables.iter().rev() {
            if &variable.name == identifier {
                return Some(variable);
            }
        }
        self.inner
            .parent
            .as_ref()
            .and_then(|parent| parent.lookup(identifier))
    }

    /// Computes the offset of the stack variable.
    pub fn offset(&self, variable: &StackVariable) -> NumWords {
        debug_assert!(self.inner.depth >= variable.depth);
        self.inner.depth - variable.depth
    }
}

pub struct ScopeBuilder {
    parent: Option<Scope>,
    variables: HashMap<Identifier, ScopeItem>,
    functions: HashMap<FunctionName, FunctionIdx>,
}

#[derive(Default)]
struct ScopeInner {
    parent: Option<Scope>,
    table: Mutex<HashMap<Identifier, ScopeItem>>,
    stack_depth: Mutex<NumWords>,
    stack_layout: StackLayout,
}

#[derive(Clone, Default)]
pub struct Scope(Arc<ScopeInner>);

impl Scope {
    pub fn new() -> Self {
        Self(Arc::new(ScopeInner {
            parent: None,
            table: Mutex::default(),
            stack_depth: Mutex::new(0.into()),
            stack_layout: StackLayoutBuilder::new(None).build(),
        }))
    }

    pub fn create_child(&self) -> Self {
        Self(Arc::new(ScopeInner {
            parent: Some(self.clone()),
            table: Mutex::default(),
            stack_depth: Mutex::new(*self.0.stack_depth.lock()),
            stack_layout: StackLayoutBuilder::new(Some(self.0.stack_layout.clone())).build(),
        }))
    }

    pub fn lookup(&self, identifier: &Identifier) -> Option<ScopeItem> {
        if let Some(stack_variable) = self.0.stack_layout.lookup(identifier) {
            let offset = self.0.stack_layout.offset(stack_variable);
            return Some(ScopeItem::StackVariable {
                offset,
                ty: stack_variable.ty.clone(),
            });
        }
        if let Some(item) = self.0.table.lock().get(identifier) {
            return Some(item.clone());
        }
        self.0
            .parent
            .as_ref()
            .and_then(|parent| parent.lookup(identifier))
    }

    pub fn insert(&mut self, identifier: Identifier, item: ScopeItem) {
        self.0.table.lock().insert(identifier, item);
    }

    // pub fn add_stack_variable(&mut self, identifier: Identifier, ty: ValueTy) -> ScopeItem {
    //     let mut stack_depth = self.0.stack_depth.lock();
    //     let word_size = ty.size_words();
    //     // let item = ScopeItem::StackVariable {
    //     //     depth: *stack_depth,
    //     //     ty,
    //     // };
    //     *stack_depth += word_size;
    //     //item
    // }
}

#[derive(Debug, Clone)]
pub enum ScopeItem {
    Constant(ConstantIdx),
    StateVariable(FieldIdx),
    StackVariable { offset: NumWords, ty: ValueTy },
}

impl ScopeItem {
    pub fn constant(idx: ConstantIdx) -> Self {
        Self::Constant(idx)
    }
}
