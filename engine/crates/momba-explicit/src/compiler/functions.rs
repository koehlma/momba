use momba_model::functions::FunctionDeclaration;

use crate::compiler::expressions::compile_expression;

use super::{expressions::CompiledExpression, scope::Scope, CompileResult, Ctx};

/// Identifies a function in the compiled model and compilation context.
pub type FunctionIdx = usize;

/// A compiled function.
pub struct CompiledFunction {
    /// The compiled body of the function.
    pub(crate) body: CompiledExpression,
    /// The local scope of the function.
    pub(crate) scope: Scope,
}

pub(crate) fn compile_function(
    ctx: &Ctx,
    parent: &Scope,
    decl: &FunctionDeclaration,
) -> CompileResult<CompiledFunction> {
    let scope = parent.create_child();
    for param in &decl.parameters {
        let ty = ctx.compute_word_ty(&param.typ)?;
        // TODO: Declare parameter.
    }
    let body = compile_expression(ctx, &scope, &decl.body)?;
    Ok(CompiledFunction { body, scope })
}
