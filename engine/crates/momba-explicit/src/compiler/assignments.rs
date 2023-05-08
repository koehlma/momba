use std::panic::{self, Location};

use momba_model::{expressions::Expression, variables::Assignment};

use crate::{
    compiler::return_error,
    datatypes::idxvec::{new_idx_type, Idx},
    values::{
        memory::{bits::BitSlice, Store},
        types::ValueTyKind,
    },
    vm::evaluate::Env,
};

use super::{
    compiled::{DestinationIdx, EdgeIdx, InstanceIdx},
    expressions::compile_expression,
    scope::{Scope, ScopeItem},
    CompileResult, Ctx, StateLayout,
};

new_idx_type! {
    /// Uniquely identifies an assignment of a destination of compiled instance.
    pub AssignmentIdx(u32)
}

new_idx_type! {
    /// Uniquely identifies an assignment group of the compiled model.
    pub AssignmentGroupIdx(u16)
}

pub struct CompiledAssignment {
    assign: Box<dyn Assign>,
}

impl CompiledAssignment {
    //#[track_caller]
    pub fn execute(&self, target: &mut BitSlice<StateLayout>, env: &mut Env) {
        //println!("Execute Assignment at {:?}.", panic::Location::caller());
        self.assign.assign(target, env);
    }
}

pub trait Assign: 'static + Send + Sync {
    fn assign(&self, target: &mut BitSlice<StateLayout>, env: &mut Env);
}

struct ClosureAssign<F>(F);

impl<F: 'static + Send + Sync + Fn(&mut BitSlice<StateLayout>, &mut Env)> ClosureAssign<F> {
    pub fn new(closure: F) -> Self {
        Self(closure)
    }
}

impl<F: 'static + Send + Sync + Fn(&mut BitSlice<StateLayout>, &mut Env)> Assign
    for ClosureAssign<F>
{
    fn assign(&self, target: &mut BitSlice<StateLayout>, env: &mut Env) {
        (self.0)(target, env)
    }
}

pub(crate) fn compile_assignment(
    ctx: &Ctx,
    scope: &Scope,
    assignment: &Assignment,
    instance: InstanceIdx,
    edge: EdgeIdx,
    dest: DestinationIdx,
) -> CompileResult<CompiledAssignment> {
    let variables = ctx.query_variables()?;
    if assignment.index.unwrap_or(0) != 0 {
        return_error!("Assignment indices are not supported yet!");
    }
    let Expression::Identifier(target) = &assignment.target else {
        return_error!("Can only assign to identifiers.");
    };
    // println!(
    //     "Compile Assignment {}:{}:{} ({:?} = {:?})",
    //     instance.as_usize(),
    //     edge.as_usize(),
    //     dest.as_usize(),
    //     target,
    //     assignment.value
    // );
    let Some(item) = scope.lookup(&target.identifier) else {
        return_error!("Unable to resolve identifier {:?}.", target.identifier);
    };
    let ScopeItem::StateVariable(field_idx) = item else {
        return_error!("Cannot assign to {:?} (not a state variable).", target.identifier);
    };
    let field = &variables.state_layout[field_idx];
    let addr = variables.state_offsets[field_idx];
    let value = compile_expression(ctx, scope, &assignment.value)?
        .coerce_to(&ctx.tcx, &ctx.tcx.loaded_value_ty(field.ty())?)?;
    let field_name = field.name().unwrap().to_owned();
    let assign = match field.ty().kind() {
        ValueTyKind::Bool => Box::new(ClosureAssign::new(move |state, env| {
            // println!(
            //     "  >> Assign To: {} ({}:{}:{})",
            //     field_name,
            //     instance.as_usize(),
            //     edge.as_usize(),
            //     dest.as_usize()
            // );
            state.store_bool(addr, value.evaluate(env));
        })) as Box<dyn Assign>,
        ValueTyKind::SignedInt(ty) => {
            let ty = ty.clone();
            Box::new(ClosureAssign::new(move |state, env| {
                // println!(
                //     "  >> Assign To: {} ({}:{}:{})",
                //     field_name,
                //     instance.as_usize(),
                //     edge.as_usize(),
                //     dest.as_usize()
                // );
                state.store_signed_int(addr, &ty, value.evaluate(env));
            }))
        }
        ValueTyKind::UnsignedInt(ty) => {
            let ty = ty.clone();
            Box::new(ClosureAssign::new(move |state, env| {
                // println!(
                //     "  >> Assign To: {} ({}:{}:{})",
                //     field_name,
                //     instance.as_usize(),
                //     edge.as_usize(),
                //     dest.as_usize()
                // );
                state.store_unsigned_int(addr, &ty, value.evaluate(env));
            }))
        }
        ValueTyKind::Float32 => todo!(),
        ValueTyKind::Float64 => todo!(),
        _ => todo!(),
    };
    Ok(CompiledAssignment { assign })
}
