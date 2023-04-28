# -*- coding:utf-8 -*-


import dataclasses as d
import typing as t


from .. import engine, model
@d.dataclass(frozen=True)
class Objective:
    r"""A reach-avoid objective.

    Represents a reach-avoid objective of the form
    :math:`\lnot\phi\mathbin{\mathbf{U}}\psi`, i.e., :math:`\lnot\phi` has to
    be true until the goal :math:`\psi` is reached. Used in
    conjunction with :class:`Rewards` to provide rewards.
    """

    goal_predicate: model.Expression
    r"""A boolean expression for :math:`\psi`."""

    dead_predicate: model.Expression
    r"""A boolean expression for :math:`\phi`."""


def extract_objective(prop: model.Expression) -> Objective:
    if isinstance(prop, model.properties.Aggregate):
        assert prop.function in {
            model.operators.AggregationFunction.MIN,
            model.operators.AggregationFunction.MAX,
            model.operators.AggregationFunction.VALUES,
            model.operators.AggregationFunction.EXISTS,
            model.operators.AggregationFunction.FORALL,
        }, f"Unsupported aggregation function {prop.function}"
        assert isinstance(
            prop.predicate, model.properties.StateSelector
        ), f"Unsupported state predicate {prop.predicate} in aggregation"
        assert (
            prop.predicate.predicate is model.properties.StatePredicate.INITIAL
        ), "Unsupported state predicate for aggregation."
        prop = prop.values

    if isinstance(prop, model.properties.Probability):
        prop = prop.formula
    
    if isinstance(prop, model.properties.UnaryPathFormula):
        assert (
            prop.operator is model.operators.UnaryPathOperator.EVENTUALLY
        ), "Unsupported unary path formula."
        return Objective(
            goal_predicate=prop.formula, dead_predicate=model.ensure_expr(False)
        )
    elif isinstance(prop, model.properties.BinaryPathFormula):
        assert (
            prop.operator is model.operators.BinaryPathOperator.UNTIL
        ), "Unsupported binary path formula."
        left = prop.left
        right = prop.right
        return Objective(
            goal_predicate=right, dead_predicate=model.expressions.logic_not(left)
        )
    elif isinstance(prop, model.properties.ExpectedReward):
        #print(f'TESTING: {prop.reachability.left} {prop.reachability.right}')
        return Objective(goal_predicate=prop.reachability,
                        dead_predicate=model.ensure_expr(False))
    else:
        raise Exception("Unsupported property!")