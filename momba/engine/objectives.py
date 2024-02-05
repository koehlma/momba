# -*- coding:utf-8 -*-


import dataclasses as d


from .. import model


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

    # add here the min?max
    op: model.operators.MinMax
    r"""The type of Objective (min or max)."""


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
        # match prop.operator:
        #     case MinMax.MIN:
        #         op = prop.operator.MIN.name
        #     case MinMax.MAX:
        #         op = prop.operator.MAX.name
        op = prop.operator
        prop = prop.formula

    if isinstance(prop, model.properties.UnaryPathFormula):
        assert (
            prop.operator is model.operators.UnaryPathOperator.EVENTUALLY
        ), "Unsupported unary path formula."
        return Objective(
            goal_predicate=prop.formula, dead_predicate=model.ensure_expr(False), op=op
        )
    elif isinstance(prop, model.properties.BinaryPathFormula):
        assert (
            prop.operator is model.operators.BinaryPathOperator.UNTIL
        ), "Unsupported binary path formula."
        lft = prop.left
        right = prop.right
        return Objective(
            goal_predicate=right, dead_predicate=model.expressions.logic_not(lft), op=op
        )
    # elif isinstance(prop, model.properties.ExpectedReward):
    #     match prop.operator:
    #         case MinMax.MIN:
    #             op = prop.operator.MIN.name
    #         case MinMax.MAX:
    #             op = prop.operator.MAX.name
    #     # TODO: change this, we need to support it but this its not the way.
    #     return Objective(
    #         goal_predicate=prop.reachability,
    #         dead_predicate=model.ensure_expr(False),
    #         op=op,
    #     )
    # elif isinstance(prop, model.expressions.Comparison):
    #     # TODO: change this, we need to support it but this its not the way.
    #     """
    #     For example, the first 3 properties of the firewire model are a composition
    #     of expression with probabilities.
    #     """
    #     subprop_l = prop.left

    #     if isinstance(subprop_l.formula, model.properties.BinaryPathFormula):
    #         assert (
    #             subprop_l.formula.operator is model.operators.BinaryPathOperator.UNTIL
    #         ), "Unsupported unary path formula."
    #         lft = prop.left
    #         right = prop.right
    #         obj = Objective(
    #             goal_predicate=lft,
    #             dead_predicate=model.expressions.logic_not(lft),
    #             op=op,
    #         )

    #         return obj
    else:
        raise Exception("Unsupported property!")
