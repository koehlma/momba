//! Building blocks for building _evaluators_.

use std::marker::PhantomData;

use crate::{compiler::StateLayout, values::memory::bits::BitSlice};

pub struct Env<'env> {
    pub(crate) state: &'env BitSlice<StateLayout>,
}

impl<'env> Env<'env> {
    pub fn new(state: &'env BitSlice<StateLayout>) -> Self {
        Self { state }
    }
}

pub trait Evaluate<T> {
    fn evaluate(&self, env: &mut Env) -> T;

    #[inline(always)]
    fn evaluate_const(&self) -> Option<T> {
        None
    }
}

pub type Evaluator<T> = Box<dyn 'static + Evaluate<T> + Send + Sync>;

pub trait EvaluateExt<T>: Evaluate<T> {
    fn evaluator(self) -> Evaluator<T>
    where
        Self: 'static + Sized + Send + Sync,
    {
        Box::new(self)
    }

    fn map<U, F: 'static + Send + Sync + Fn(T) -> U>(self, fun: F) -> UnaryFn<Self, F, T, U>
    where
        Self: Sized,
    {
        UnaryFn::new(fun, self)
    }
}

impl<T, E: Evaluate<T>> EvaluateExt<T> for E {}

impl<T> Evaluate<T> for Evaluator<T> {
    #[inline(always)]
    fn evaluate(&self, env: &mut Env) -> T {
        (**self).evaluate(env)
    }

    #[inline(always)]
    fn evaluate_const(&self) -> Option<T> {
        (**self).evaluate_const()
    }
}

pub struct Constant<T: Clone>(T);

impl<T: Clone> Constant<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: Clone> Evaluate<T> for Constant<T> {
    #[inline(always)]
    fn evaluate(&self, _: &mut Env) -> T {
        self.0.clone()
    }

    #[inline(always)]
    fn evaluate_const(&self) -> Option<T> {
        Some(self.0.clone())
    }
}

pub struct Closure<F, T> {
    closure: F,
    _phantom_fun: PhantomData<fn(&mut Env) -> T>,
}

impl<T, F: Fn(&mut Env) -> T> Closure<F, T> {
    pub fn new(closure: F) -> Self {
        Self {
            closure,
            _phantom_fun: PhantomData,
        }
    }
}

impl<T, F: Fn(&mut Env) -> T> Evaluate<T> for Closure<F, T> {
    #[inline(always)]
    fn evaluate(&self, env: &mut Env) -> T {
        (self.closure)(env)
    }
}

pub struct Conditional<C, Ec, Ea> {
    condition: C,
    consequence: Ec,
    alternative: Ea,
}

impl<C, Ec, Ea> Conditional<C, Ec, Ea> {
    pub fn new(condition: C, consequence: Ec, alternative: Ea) -> Self {
        Self {
            condition,
            consequence,
            alternative,
        }
    }
}

impl<T, C: Evaluate<bool>, Ec: Evaluate<T>, Ea: Evaluate<T>> Evaluate<T>
    for Conditional<C, Ec, Ea>
{
    #[inline(always)]
    fn evaluate(&self, env: &mut Env) -> T {
        if self.condition.evaluate(env) {
            self.consequence.evaluate(env)
        } else {
            self.alternative.evaluate(env)
        }
    }

    #[inline(always)]
    fn evaluate_const(&self) -> Option<T> {
        self.condition.evaluate_const().and_then(|condition| {
            if condition {
                self.consequence.evaluate_const()
            } else {
                self.alternative.evaluate_const()
            }
        })
    }
}

pub struct UnaryFn<E, F, T, U> {
    fun: F,
    arg: E,
    _phantom_fun: PhantomData<fn(T) -> U>,
}

impl<T, U, E, F: Fn(T) -> U> UnaryFn<E, F, T, U> {
    fn new(fun: F, arg: E) -> Self {
        Self {
            fun,
            arg,
            _phantom_fun: PhantomData,
        }
    }
}

impl<T, U, E: Evaluate<T>, F: Fn(T) -> U> Evaluate<U> for UnaryFn<E, F, T, U> {
    #[inline(always)]
    fn evaluate(&self, env: &mut Env) -> U {
        (self.fun)(self.arg.evaluate(env))
    }

    #[inline(always)]
    fn evaluate_const(&self) -> Option<U> {
        self.arg.evaluate_const().map(&self.fun)
    }
}

pub struct BinaryFn<F, El, Er, Tl, Tr, U> {
    left: El,
    right: Er,
    fun: F,
    _phantom_fun: PhantomData<fn(Tl, Tr) -> U>,
}

impl<F, El, Er, Tl, Tr, U> BinaryFn<F, El, Er, Tl, Tr, U> {
    pub fn new(fun: F, left: El, right: Er) -> Self {
        Self {
            fun,
            left,
            right,
            _phantom_fun: PhantomData,
        }
    }
}

impl<F, El, Er, Tl, Tr, U> Evaluate<U> for BinaryFn<F, El, Er, Tl, Tr, U>
where
    F: 'static + Send + Sync + Fn(Tl, Tr) -> U,
    El: Evaluate<Tl>,
    Er: Evaluate<Tr>,
{
    fn evaluate(&self, env: &mut Env) -> U {
        let left = self.left.evaluate(env);
        let right = self.right.evaluate(env);
        (self.fun)(left, right)
    }

    fn evaluate_const(&self) -> Option<U> {
        let left = self.left.evaluate_const()?;
        let right = self.right.evaluate_const()?;
        Some((self.fun)(left, right))
    }
}
