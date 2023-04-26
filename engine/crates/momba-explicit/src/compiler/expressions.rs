// Compiler for expressions.

use std::fmt;

use momba_model::{expressions::*, values::Value};

use crate::{
    compiler::return_error,
    values::{
        memory::Load,
        types::{HasType, TypeCtx, ValueTyKind, WordTy, WordTyKind},
        FromWord, IntoWord, Word,
    },
    vm::evaluate::{self, BinaryFn, Env, Evaluate, EvaluateExt, Evaluator},
};

use super::{
    scope::{Scope, ScopeItem},
    CompileResult, Ctx,
};

/// A typed compile time constant.
#[derive(Clone, PartialEq, Eq)]
pub struct Constant {
    /// The value of the constant.
    value: Word,
    /// The type of the constant.
    ty: WordTy,
}

impl Constant {
    /// Constructs a [`Constant`] from the provided parts.
    pub fn from_parts<V: IntoWord>(value: V, ty: WordTy) -> Self {
        Self {
            value: value.into_word(),
            ty,
        }
    }

    /// The value of the constant.
    pub fn value(&self) -> Word {
        self.value
    }

    /// The type of the constant.
    pub fn ty(&self) -> &WordTy {
        &self.ty
    }
}

// impl From<Value> for Constant {
//     fn from(value: Value) -> Self {
//         match value {
//             Value::Int(value) => Constant::from_parts(value, WordTy::int()),
//             Value::Float(value) => Constant::from_parts(value, WordTy::float()),
//             Value::Bool(value) => Constant::from_parts(value, WordTy::bool()),
//         }
//     }
// }

// impl From<i64> for Constant {
//     fn from(value: i64) -> Self {
//         Constant::from_parts(value, WordTy::int())
//     }
// }

// impl From<f64> for Constant {
//     fn from(value: f64) -> Self {
//         Constant::from_parts(value, WordTy::float())
//     }
// }

// impl From<bool> for Constant {
//     fn from(value: bool) -> Self {
//         Constant::from_parts(value, WordTy::bool())
//     }
// }

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty.kind() {
            WordTyKind::Int => i64::from_word(self.value).fmt(f),
            WordTyKind::Float => f64::from_word(self.value).fmt(f),
            WordTyKind::Bool => bool::from_word(self.value).fmt(f),
            _ => panic!(""),
        }
    }
}

/// A compiled expression combines an [`Evaluator`] with an [`ExprTy`].
pub struct CompiledExpression<T = Word> {
    /// The evaluator for the expression.
    evaluator: Evaluator<T>,
    /// The inferred type of the expression.
    ty: WordTy,
}

impl<T> CompiledExpression<T> {
    pub fn evaluate(&self, env: &mut Env) -> T {
        self.evaluator.evaluate(env)
    }
}

impl CompiledExpression {
    fn from_parts(evaluator: Evaluator<Word>, ty: WordTy) -> Self {
        Self { evaluator, ty }
    }

    /// Constructs a [`CompiledExpression`] from the provided parts.
    fn from_eval<E: 'static + Evaluate<Word> + Send + Sync>(eval: E, ty: WordTy) -> Self {
        Self {
            evaluator: eval.evaluator(),
            ty,
        }
    }

    pub fn cast<U: 'static + HasType + FromWord>(
        self,
        tcx: &TypeCtx,
    ) -> CompileResult<CompiledExpression<U>> {
        if U::word_ty(tcx) == self.ty {
            Ok(CompiledExpression {
                evaluator: self.evaluator.map(U::from_word).evaluator(),
                ty: self.ty,
            })
        } else {
            return_error!("Unable to cast expression.");
        }
    }

    /// Creates a compiled expression applying a closure to the value.
    pub fn map<
        T: HasType + FromWord,
        U: HasType + IntoWord,
        F: 'static + Send + Sync + Fn(T) -> U,
    >(
        self,
        tcx: &TypeCtx,
        fun: F,
    ) -> Self {
        assert_eq!(T::word_ty(tcx), self.ty);
        Self::from_eval(
            self.evaluator
                .map(move |value| fun(T::from_word(value)).into_word())
                .evaluator(),
            U::word_ty(tcx),
        )
    }

    /// Coerces the compiled expression to the provided type.
    pub fn coerce_to(self, tcx: &TypeCtx, ty: &WordTy) -> CompileResult<CompiledExpression> {
        if self.ty == *ty {
            // No coercion necessary, types are identical.
            return Ok(self);
        }
        Ok(match (self.ty.kind(), ty.kind()) {
            // Actually, there is just one possible coercion from `Int` to `Float`.
            (WordTyKind::Int, WordTyKind::Float) => self.map(tcx, |value: i64| value as f64),
            _ => return_error!(
                "Expression of type `{}` cannot be coerced to type `{}`.",
                self.ty,
                ty
            ),
        })
    }

    /// If possible, turns the expression into a constant one.
    pub fn make_constant(self) -> CompiledExpression {
        if let Some(value) = self.evaluator.evaluate_const() {
            Self::from_eval(evaluate::Constant::new(value), self.ty)
        } else {
            self
        }
    }

    /// Tries to evaluate the expression to a constant.
    pub fn evaluate_const(&self) -> Option<Constant> {
        self.evaluator
            .evaluate_const()
            .map(|value| Constant::from_parts(value, self.ty.clone()))
    }
}

fn binary_fn<
    Tl: HasType + FromWord,
    Tr: HasType + FromWord,
    U: HasType + IntoWord,
    F: 'static + Send + Sync + Fn(Tl, Tr) -> U,
>(
    tcx: &TypeCtx,
    left: CompiledExpression,
    right: CompiledExpression,
    fun: F,
) -> CompiledExpression {
    assert_eq!(Tl::word_ty(tcx), left.ty);
    assert_eq!(Tr::word_ty(tcx), right.ty);
    CompiledExpression::from_eval(
        BinaryFn::new(
            move |left: Word, right: Word| {
                fun(Tl::from_word(left), Tr::from_word(right)).into_word()
            },
            left.evaluator,
            right.evaluator,
        ),
        U::word_ty(tcx),
    )
}

pub(crate) fn compile_expression(
    ctx: &Ctx,
    scope: &Scope,
    expr: &Expression,
) -> CompileResult<CompiledExpression> {
    ExprCtx::new(ctx, scope).compile(expr)
}

/// Compilation context for expressions.
struct ExprCtx<'cx> {
    /// The compilation context.
    ctx: &'cx Ctx<'cx>,
    /// The scope in which the expression is compiled.
    scope: &'cx Scope,
}

impl<'cx> ExprCtx<'cx> {
    pub(crate) fn new(ctx: &'cx Ctx<'cx>, scope: &'cx Scope) -> Self {
        Self { ctx, scope }
    }

    pub fn compile(&self, expr: &Expression) -> CompileResult<CompiledExpression> {
        let compiled = match expr {
            Expression::Constant(expr) => self.compile_constant(expr),
            Expression::Identifier(expr) => self.compile_identifier(expr),
            Expression::Conditional(expr) => self.compile_conditional(expr),
            Expression::Unary(expr) => self.compile_unary(expr),
            Expression::Binary(expr) => self.compile_binary(expr),
            Expression::Index(expr) => self.compile_index(expr),
            Expression::Array(expr) => self.compile_array(expr),
            Expression::Comprehension(expr) => self.compile_comprehension(expr),
            Expression::Call(expr) => self.compile_call(expr),
        }?;
        // By applying `make_constant` here, we essentially constant-fold all expressions.
        Ok(compiled.make_constant())
    }

    fn compile_constant(&self, expr: &ConstantExpression) -> CompileResult<CompiledExpression> {
        Ok(CompiledExpression::from_eval(
            evaluate::Constant::new(expr.value.clone().into_word()),
            self.ctx.tcx.type_of_value(&expr.value),
        ))
    }

    fn compile_identifier(&self, expr: &IdentifierExpression) -> CompileResult<CompiledExpression> {
        let Some(item) = self.scope.lookup(&expr.identifier) else {
            return_error!("Unable to resolve identifier `{}`.", expr.identifier);
        };
        Ok(match item {
            ScopeItem::Constant(idx) => {
                let constant = self.ctx.query_constant_value(idx)?;
                CompiledExpression::from_eval(
                    evaluate::Constant::new(constant.value()),
                    constant.ty().clone(),
                )
            }
            ScopeItem::StackVariable { .. } => todo!(),
            ScopeItem::StateVariable(field_idx) => {
                let variables = self.ctx.query_variables()?;
                let field = &variables.state_layout[field_idx];
                let addr = variables.state_offsets[field_idx];
                let expr_ty = self.ctx.tcx.loaded_value_ty(field.ty())?;
                CompiledExpression::from_parts(
                    match field.ty().kind() {
                        ValueTyKind::Bool => {
                            evaluate::Closure::new(move |env| env.state.load_bool(addr)).evaluator()
                        }
                        ValueTyKind::SignedInt(ty) => {
                            let ty = ty.clone();
                            evaluate::Closure::new(move |env| env.state.load_signed_int(addr, &ty))
                                .evaluator()
                        }
                        ValueTyKind::UnsignedInt(ty) => {
                            let ty = ty.clone();
                            evaluate::Closure::new(move |env| {
                                env.state.load_unsigned_int(addr, &ty)
                            })
                            .evaluator()
                        }
                        ValueTyKind::Float32 => {
                            evaluate::Closure::new(move |env| env.state.load_float32(addr))
                                .evaluator()
                        }
                        ValueTyKind::Float64 => {
                            evaluate::Closure::new(move |env| env.state.load_float64(addr))
                                .evaluator()
                        }
                        _ => {
                            return_error!("Type cannot be loaded.");
                        }
                    },
                    expr_ty,
                )
            }
        })
    }

    /// Compiles an conditional expression.
    fn compile_conditional(
        &self,
        expr: &ConditionalExpression,
    ) -> CompileResult<CompiledExpression> {
        let condition = self.compile(&expr.condition)?;
        if !condition.ty.is_bool() {
            return_error!("Type of condition must be `Bool`.");
        }
        // Short circuit in case the condition is constant.
        if let Some(condition) = condition.evaluator.evaluate_const() {
            if bool::from_word(condition) {
                return self.compile(&expr.consequence);
            } else {
                return self.compile(&expr.alternative);
            }
        } else {
            let consequence = self.compile(&expr.consequence)?;
            let alternative = self.compile(&expr.alternative)?;
            let ty = self
                .ctx
                .tcx
                .common_coercion(&consequence.ty, &alternative.ty)?;
            let consequence = consequence.coerce_to(&self.ctx.tcx, &ty)?;
            let alternative = alternative.coerce_to(&self.ctx.tcx, &ty)?;
            debug_assert_eq!(
                consequence.ty, alternative.ty,
                "Type of `consequence` and `alternative` should match after coercion."
            );
            Ok(CompiledExpression::from_eval(
                evaluate::Conditional::new(
                    condition.evaluator.map(bool::from_word),
                    consequence.evaluator,
                    alternative.evaluator,
                ),
                ty,
            ))
        }
    }

    /// Compiles a unary expression.
    fn compile_unary(&self, expr: &UnaryExpression) -> CompileResult<CompiledExpression> {
        let operand = self.compile(&expr.operand)?;

        macro_rules! invalid_operand_type {
            () => {
                return_error!(
                    "Unsupported operand type `{}` of `{:?}` operator.",
                    operand.ty,
                    expr.operator,
                )
            };
        }

        macro_rules! compile_numeric {
            (|$operand:ident| $value:expr) => {
                compile_numeric!(|$operand| $value, |$operand| $value)
            };
            (|$int_operand:ident| $int_value:expr, |$float_operand:ident| $float_value:expr) => {
                match operand.ty.kind() {
                    WordTyKind::Int => operand.map(&self.ctx.tcx, |$int_operand: i64| $int_value),
                    WordTyKind::Float => {
                        operand.map(&self.ctx.tcx, |$float_operand: f64| $float_value)
                    }
                    _ => invalid_operand_type!(),
                }
            };
        }

        macro_rules! compile_trigonometric {
            ($fun:ident) => {
                compile_numeric! {
                    |operand| (operand as f64).$fun(),
                    |operand| operand.$fun()
                }
            };
        }

        Ok(match expr.operator {
            UnaryOperator::Not => match operand.ty.kind() {
                WordTyKind::Bool => operand.map(&self.ctx.tcx, |value: bool| !value),
                _ => invalid_operand_type!(),
            },
            UnaryOperator::Minus => compile_numeric! {
                |operand| -operand
            },
            UnaryOperator::Floor => compile_numeric! {
                |operand| operand,
                |operand| operand.floor() as i64
            },
            UnaryOperator::Ceil => compile_numeric! {
                |operand| operand,
                |operand| operand.ceil() as i64
            },
            UnaryOperator::Abs => compile_numeric! {
                |operand| operand.abs()
            },
            UnaryOperator::Sgn => compile_numeric! {
                |operand| operand.signum(),
                |operand| operand.signum() as i64
            },
            UnaryOperator::Trc => compile_numeric! {
                |operand| operand,
                |operand| operand.trunc() as i64
            },
            UnaryOperator::Sin => compile_trigonometric!(sin),
            UnaryOperator::Cos => compile_trigonometric!(cos),
            UnaryOperator::Tan => compile_trigonometric!(tan),
            UnaryOperator::ArcSin => compile_trigonometric!(asin),
            UnaryOperator::ArcCos => compile_trigonometric!(acos),
            UnaryOperator::ArcTan => compile_trigonometric!(atan),
            _ => return_error!(
                "Support for unary operator `{:?}` has not been implemented.",
                expr.operator
            ),
        })
    }

    /// Compiles a binary expression.
    fn compile_binary(&self, expr: &BinaryExpression) -> CompileResult<CompiledExpression> {
        let left = self.compile(&expr.left)?;
        let right = self.compile(&expr.right)?;

        macro_rules! invalid_operand_types {
            () => {
                return_error!(
                    "Unsupported operand types `{}` and `{}` of `{:?}` operator.",
                    left.ty,
                    right.ty,
                    expr.operator,
                )
            };
        }

        macro_rules! compile_numeric {
            (|$left:ident, $right:ident| $value:expr) => {
                compile_numeric!(|$left, $right| $value, |$left, $right| $value)
            };
            (
                |$int_left:ident, $int_right:ident| $int_value:expr,
                |$float_left:ident, $float_right:ident| $float_value:expr
            ) => {
                match (left.ty.kind(), right.ty.kind()) {
                    (WordTyKind::Int, WordTyKind::Int) => binary_fn(
                        &self.ctx.tcx,
                        left,
                        right,
                        |$int_left: i64, $int_right: i64| $int_value,
                    ),
                    (WordTyKind::Float, WordTyKind::Int) => binary_fn(
                        &self.ctx.tcx,
                        left,
                        right,
                        |$float_left: f64, right: i64| {
                            let $float_right = right as f64;
                            $float_value
                        },
                    ),
                    (WordTyKind::Int, WordTyKind::Float) => binary_fn(
                        &self.ctx.tcx,
                        left,
                        right,
                        |left: i64, $float_right: f64| {
                            let $float_left = left as f64;
                            $float_value
                        },
                    ),
                    (WordTyKind::Float, WordTyKind::Float) => binary_fn(
                        &self.ctx.tcx,
                        left,
                        right,
                        |$float_left: f64, $float_right: f64| $float_value,
                    ),
                    _ => invalid_operand_types!(),
                }
            };
        }

        macro_rules! compile_boolean {
            (|$left:ident, $right:ident| $value:expr) => {
                match (left.ty.kind(), right.ty.kind()) {
                    (WordTyKind::Bool, WordTyKind::Bool) => {
                        binary_fn(&self.ctx.tcx, left, right, |$left: bool, $right: bool| {
                            $value
                        })
                    }
                    _ => invalid_operand_types!(),
                }
            };
        }

        macro_rules! compile_eq {
            (|$left:ident, $right:ident| $value:expr) => {{
                let ty = self.ctx.tcx.common_coercion(&left.ty, &right.ty)?;
                let left = left.coerce_to(&self.ctx.tcx, &ty)?;
                let right = right.coerce_to(&self.ctx.tcx, &ty)?;
                CompiledExpression::from_eval(
                    evaluate::BinaryFn::new(
                        |$left: Word, $right: Word| $value.into_word(),
                        left.evaluator,
                        right.evaluator,
                    ),
                    self.ctx.tcx.word_bool(),
                )
            }};
        }

        Ok(match expr.operator {
            BinaryOperator::Add => compile_numeric! {
                |left, right| left + right
            },
            BinaryOperator::Sub => compile_numeric! {
                |left, right| left - right
            },
            BinaryOperator::Mul => compile_numeric! {
                |left, right| left * right
            },
            BinaryOperator::FloorDiv => compile_numeric! {
                |left, right| left / right,
                |left, right| (left / right).floor() as i64
            },
            BinaryOperator::RealDiv => compile_numeric! {
                |left, right| (left as f64) / (right as f64),
                |left, right| left / right
            },
            BinaryOperator::Mod => compile_numeric! {
                |left, right| left.rem_euclid(right)
            },
            BinaryOperator::Pow => compile_numeric! {
                |left, right| (left as f64).powf(right as f64),
                |left, right| left.powf(right)
            },
            BinaryOperator::Log => compile_numeric!(
                |left, right| (left as f64).log(right as f64),
                |left, right| left.log(right)
            ),
            BinaryOperator::Min => compile_numeric! {
                |left, right| left.min(right)
            },
            BinaryOperator::Max => compile_numeric! {
                |left, right| left.max(right)
            },
            BinaryOperator::And => compile_boolean! {
                |left, right| left && right
            },
            BinaryOperator::Or => compile_boolean! {
                |left, right| left || right
            },
            BinaryOperator::Xor => compile_boolean! {
                |left, right| (left && !right) || (!left && right)
            },
            BinaryOperator::Equiv => compile_boolean! {
                |left, right| left == right
            },
            BinaryOperator::Imply => compile_boolean! {
                |left, right| !left || right
            },
            BinaryOperator::Le => compile_numeric! {
                |left, right| left <= right
            },
            BinaryOperator::Lt => compile_numeric! {
                |left, right| left < right
            },
            BinaryOperator::Ge => compile_numeric! {
                |left, right| left >= right
            },
            BinaryOperator::Gt => compile_numeric! {
                |left, right| left > right
            },
            BinaryOperator::Eq => compile_eq! {
                |left, right| left == right
            },
            BinaryOperator::Ne => compile_eq! {
                |left, right| left != right
            },
        })
    }

    /// Compiles an index expression.
    fn compile_index(&self, expr: &IndexExpression) -> CompileResult<CompiledExpression> {
        // let index = self.compile(&expr.index)?;
        // if !index.ty.is_int() {
        //     return_error!("Invalid type `{}` of index in index expression.", &index.ty);
        // }
        // let slice = self.compile(&expr.slice)?;
        // let WordTyKind::Slice { element_ty, area, length } = slice.ty.kind() else {
        //     return_error!("Invalid type `{}` of slice in index expression.", &slice.ty)
        // };
        // drop(element_ty);
        // drop(area);
        // drop(length);
        todo!()
    }

    /// Compiles an array expression.
    fn compile_array(&self, expr: &ArrayExpression) -> CompileResult<CompiledExpression> {
        // let elements = expr
        //     .elements
        //     .iter()
        //     .map(|element| self.compile(element))
        //     .collect::<Result<Vec<_>, _>>()?;
        // let Some(_element_ty) = elements.iter().try_fold(None, |ty, element| {
        //     if let Some(other) = ty {
        //         self.ctx.tcx.common_coercion(&other, &element.ty).map(Some)
        //     } else {
        //         Ok(Some(element.ty.clone()))
        //     }
        // })? else {
        //     // Create an empty dangling slice. The type does not matter as it is empty.
        //     return Ok(CompiledExpression::from_parts(
        //         evaluate::Constant::new(Word::from(Pointer::invalid())),
        //         WordTy::slice_of(ValueTy::zst(), None, Some(0)),
        //     ));
        // };
        todo!()
    }

    fn compile_comprehension(
        &self,
        expr: &ComprehensionExpression,
    ) -> CompileResult<CompiledExpression> {
        let length = self.compile(&expr.length)?;
        if !length.ty.is_int() {
            return_error!("Invalid type `{}` of comprehension length.", length.ty);
        }
        todo!()
    }

    fn compile_call(&self, expr: &CallExpression) -> CompileResult<CompiledExpression> {
        drop(expr);
        todo!()
    }
}
