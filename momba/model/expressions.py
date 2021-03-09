# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import decimal
import enum
import fractions
import math
import warnings

from . import errors, operators, types

if t.TYPE_CHECKING:
    from . import context, distributions


class Expression(abc.ABC):
    """
    Abstract base class for expressions.
    """

    @abc.abstractmethod
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()

    def infer_target_type(self, scope: context.Scope) -> types.Type:
        raise errors.InvalidTypeError(
            f"expression {self} is not a valid assignment target"
        )

    @property
    @abc.abstractmethod
    def children(self) -> t.Sequence[Expression]:
        """
        The direct children of the expression.
        """
        raise NotImplementedError()

    def traverse(self) -> t.Iterator[Expression]:
        """
        Returns an iterator over the subexpressions.
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    @property
    def subexpressions(self) -> t.AbstractSet[Expression]:
        """
        A set of subexpressions.
        """
        return frozenset(self.traverse())

    @property
    def is_sampling_free(self) -> bool:
        """
        Returns whether the expression is *sampling-free*.
        """
        return all(not isinstance(e, Sample) for e in self.traverse())

    @property
    def used_names(self) -> t.AbstractSet[Name]:
        """
        Returns a set of *name expressions* occuring in the expression.
        """
        return frozenset(child for child in self.traverse() if isinstance(child, Name))


class _Leaf(Expression):
    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


class Constant(_Leaf, abc.ABC):
    pass


@d.dataclass(frozen=True)
class BooleanConstant(Constant):
    """
    A boolean constant.

    Attributes
    ----------
    boolean:
        The value of the boolean constant.
    """

    boolean: bool

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.BOOL


class NumericConstant(Constant, abc.ABC):
    """
    Abstract base class for numeric constants.
    """

    @property
    @abc.abstractmethod
    def as_float(self) -> float:
        """
        The numeric constant as a floating-point number.

        Note that this may result in a loss of precision.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def as_fraction(self) -> fractions.Fraction:
        """
        The numeric constant as a fraction.

        Note that this may result in a loss of precision
        """
        raise NotImplementedError()


TRUE: Expression = BooleanConstant(True)
FALSE: Expression = BooleanConstant(False)


@d.dataclass(frozen=True)
class IntegerConstant(NumericConstant):
    """
    An integer constant.

    Attributes
    ----------
    integer:
        The value of the integer constant.
    """

    integer: int

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.INT

    @property
    def as_float(self) -> float:
        warnings.warn(
            "converting an integer constant to a float may result in a loss of precision"
        )
        return float(self.integer)

    @property
    def as_fraction(self) -> fractions.Fraction:
        return fractions.Fraction(self.integer)


_NAMED_REAL_MAP: t.Dict[str, NamedReal] = {}


class NamedReal(enum.Enum):
    """
    An enum of named reals.

    Attributes
    ----------
    symbol:
        The mathematical symbol of the real.
    float_value:
        A floating-point approximation of the real.
    """

    PI = "π", math.pi
    """ The number π. """

    E = "e", math.e
    """ The number e. """

    symbol: str
    float_value: float

    def __init__(self, symbol: str, float_value: float) -> None:
        self.symbol = symbol
        self.float_value = float_value
        _NAMED_REAL_MAP[symbol] = self


Real = t.Union[NamedReal, fractions.Fraction]


@d.dataclass(frozen=True)
class RealConstant(NumericConstant):
    """
    A real constant.

    Attributes
    ----------
    real:
        The real (either :class:`NamedReal` or a fraction).
    """

    real: Real

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    @property
    def as_float(self) -> float:
        warnings.warn(
            "converting a real constant to a float may result in a loss of precision"
        )
        if isinstance(self.real, NamedReal):
            return self.real.float_value
        return float(self.real)

    @property
    def as_fraction(self) -> fractions.Fraction:
        if isinstance(self.real, fractions.Fraction):
            return self.real
        else:
            warnings.warn(
                "converting a named real constant do a fraction does result in a loss of precision"
            )
            return fractions.Fraction(self.real.float_value)


@d.dataclass(frozen=True)
class Name(_Leaf):
    """
    A name expression.

    Attributes
    ----------
    identifier:
        The identifier.
    """

    identifier: str

    def infer_type(self, scope: context.Scope) -> types.Type:
        return scope.lookup(self.identifier).typ

    def infer_target_type(self, scope: context.Scope) -> types.Type:
        return self.infer_type(scope)


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class BinaryExpression(Expression):
    """
    Abstract base class for binary expressions.

    Attributes
    ----------
    operator:
        The binary operator (:class:`~momba.model.operators.BinaryOperator`).
    left:
        The left operand.
    right:
        The right operand.
    """

    operator: operators.BinaryOperator
    left: Expression
    right: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.left, self.right

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


class Boolean(BinaryExpression):
    """
    A boolean binary expression.

    Attributes
    ----------
    operator:
        The boolean operator (:class:`~momba.model.operators.BooleanOperator`).
    """

    operator: operators.BooleanOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if left_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {left_type}")
        right_type = scope.get_type(self.right)
        if right_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {right_type}")
        return types.BOOL


_REAL_RESULT_OPERATORS = {
    operators.ArithmeticBinaryOperator.REAL_DIV,
    operators.ArithmeticBinaryOperator.LOG,
    operators.ArithmeticBinaryOperator.POW,
}


class ArithmeticBinary(BinaryExpression):
    """
    An arithmetic binary expression.

    Attributes
    ----------
    operator:
        The arithmetic operator (:class:`~momba.model.operators.ArithmeticBinaryOperator`).
    """

    operator: operators.ArithmeticBinaryOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        if not left_type.is_numeric or not right_type.is_numeric:
            raise errors.InvalidTypeError(
                "operands of arithmetic expressions must have a numeric type"
            )
        is_int = (
            types.INT.is_assignable_from(left_type)
            and types.INT.is_assignable_from(right_type)
            and self.operator not in _REAL_RESULT_OPERATORS
        )
        if is_int:
            return types.INT
        else:
            return types.REAL


class Equality(BinaryExpression):
    """
    An equality binary expression.

    Attributes
    ----------
    operator:
        The equality operator (:class:`~momba.model.operators.EqualityOperator`).
    """

    operator: operators.EqualityOperator

    def get_common_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        if left_type.is_assignable_from(right_type):
            return left_type
        elif right_type.is_assignable_from(left_type):
            return right_type
        raise AssertionError(
            "type-inference should ensure that some of the above is true"
        )

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        left_assignable_right = left_type.is_assignable_from(right_type)
        right_assignable_left = right_type.is_assignable_from(left_type)
        if left_assignable_right or right_assignable_left:
            return types.BOOL
        # XXX: JANI specifies that “left and right must be assignable to some common type”
        # not sure whether this implementation reflects this specification
        raise errors.InvalidTypeError(
            "invalid combination of type for equality comparison"
        )


class Comparison(BinaryExpression):
    """
    A comparison expression.

    Attributes
    ----------
    operator:
        The comparison operator (:class:`~momba.model.operators.ComparisonOperator`).
    """

    operator: operators.ComparisonOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if not left_type.is_numeric:
            raise errors.InvalidTypeError(f"expected numeric type but got {left_type}")
        right_type = scope.get_type(self.right)
        if not right_type.is_numeric:
            raise errors.InvalidTypeError(f"expected numeric type but got {right_type}")
        return types.BOOL


@d.dataclass(frozen=True)
class Conditional(Expression):
    """
    A ternary conditional expression.

    Attributes
    ----------
    condition:
        The condition.
    consequence:
        The consequence to be evaluated if the condition is true.
    alternative:
        The alternative to be evaluated if the condition is false.
    """

    condition: Expression
    consequence: Expression
    alternative: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.condition, self.consequence, self.alternative

    def infer_type(self, scope: context.Scope) -> types.Type:
        condition_type = scope.get_type(self.condition)
        if condition_type != types.BOOL:
            raise errors.InvalidTypeError(
                f"expected `types.BOOL` but got `{condition_type}`"
            )
        consequence_type = scope.get_type(self.consequence)
        alternative_type = scope.get_type(self.alternative)
        if consequence_type.is_assignable_from(alternative_type):
            return consequence_type
        elif alternative_type.is_assignable_from(consequence_type):
            return alternative_type
        else:
            raise errors.InvalidTypeError(
                "invalid combination of consequence and alternative types"
            )


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class UnaryExpression(Expression, abc.ABC):
    """
    Base class of all unary expressions.

    Attributes
    ----------
    operator:
        The unary operator (:class:`~momba.model.operators.UnaryOperator`).
    operand:
        The operand.
    """

    operator: operators.UnaryOperator
    operand: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return (self.operand,)

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


class ArithmeticUnary(UnaryExpression):
    """
    An arithmetic unary expression.

    Attributes
    ----------
    operator:
        The arithmetic operator (:class:`~momba.model.operators.ArithmeticUnaryOperator`).
    """

    operator: operators.ArithmeticUnaryOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if not operand_type.is_numeric:
            raise errors.InvalidTypeError(
                f"expected a numeric type but got {operand_type}"
            )
        return self.operator.infer_result_type(operand_type)


class Not(UnaryExpression):
    """
    Logical negation.

    Attributes
    ----------
    operator:
        The logical negation operator (:class:`~momba.model.operators.NotOperator`).
    """

    operator: operators.NotOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if operand_type != types.BOOL:
            raise errors.InvalidTypeError(
                f"expected `types.BOOL` but got {operand_type}"
            )
        return types.BOOL


@d.dataclass(frozen=True)
class Sample(Expression):
    """
    A sample expression.

    Attributes
    ----------
    distribution:
        The type of the distribution to sample from
        (:class:`~momba.model.distributions.DistributionType`).
    arguments:
        The arguments to the distribution.
    """

    distribution: distributions.DistributionType
    arguments: t.Sequence[Expression]

    def __post_init__(self) -> None:
        if len(self.arguments) != self.distribution.arity:
            raise errors.InvalidTypeError(
                f"distribution {self.distribution} requires {self.distribution.arity} "
                f"arguments but {len(self.arguments)} were given"
            )

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.arguments

    def infer_type(self, scope: context.Scope) -> types.Type:
        # we already know that the arity of the parameters and arguments match
        for argument, parameter_type in zip(
            self.arguments, self.distribution.parameter_types
        ):
            argument_type = scope.get_type(argument)
            if not parameter_type.is_assignable_from(argument_type):
                raise errors.InvalidTypeError(
                    f"parameter type `{parameter_type}` is not assignable "
                    f"from argument type `{argument_type}`"
                )
        return self.distribution.result_type


# requires JANI extension `nondet-selection`
@d.dataclass(frozen=True)
class Selection(Expression):
    """
    A non-deterministic selection expression.

    Attributes
    ----------
    variable:
        The identifier to select over.
    condition:
        The condition that should be satisfied.
    """

    variable: str
    condition: Expression

    def infer_type(self, scope: context.Scope) -> types.Type:
        condition_scope = scope.create_child_scope()
        condition_scope.declare_variable(self.variable, typ=types.REAL)
        condition_type = condition_scope.get_type(self.condition)
        if condition_type != types.BOOL:
            raise errors.InvalidTypeError("condition must have type `types.BOOL`")
        return types.REAL

    @property
    def children(self) -> t.Sequence[Expression]:
        return (self.condition,)


@d.dataclass(frozen=True)
class Derivative(Expression):
    """
    Derivative of a continuous variable.

    Attributes
    ==========
    identifier:
        The continuous variable.
    """

    identifier: str

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


@d.dataclass(frozen=True)
class ArrayAccess(Expression):
    """
    An array access expression.

    Attributes
    ----------
    array:
        The array to access.
    index:
        The index where to access the array.
    """

    array: Expression
    index: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.array, self.index

    def infer_type(self, scope: context.Scope) -> types.Type:
        array_type = scope.get_type(self.array)
        # TODO: check the type of the index
        if isinstance(array_type, types.ArrayType):
            return array_type.element
        else:
            raise errors.InvalidTypeError("array of array access must have array type")

    def infer_target_type(self, scope: context.Scope) -> types.Type:
        return self.infer_type(scope)


@d.dataclass(frozen=True)
class ArrayValue(Expression):
    """
    An array value expression.

    Attributes
    ----------
    elements:
        The elements of the array to construct.
    """

    elements: t.Tuple[Expression, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            raise errors.ModelingError("array value expression needs to have elements")

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.elements

    def infer_type(self, scope: context.Scope) -> types.Type:
        common_type: t.Optional[types.Type] = None
        for element in self.elements:
            element_type = scope.get_type(element)
            if common_type is None:
                common_type = element_type
            elif element_type.is_assignable_from(common_type):
                common_type = element_type
            elif not common_type.is_assignable_from(element_type):
                raise errors.InvalidTypeError(
                    "element types are not assignable to a common type"
                )
        assert common_type is not None
        return types.array_of(common_type)


@d.dataclass(frozen=True)
class ArrayConstructor(Expression):
    """
    An array constructor expression.

    Attributes
    ----------
    variable:
        The identifier to range over.
    length:
        The length of the array.
    expression:
        The expression to compute the elements of the array.
    """

    variable: str
    length: Expression
    expression: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.length, self.expression

    def _create_scope(self, scope: context.Scope) -> context.Scope:
        child_scope = scope.create_child_scope()
        child_scope.declare_constant(self.variable, types.INT)
        return child_scope

    def infer_type(self, scope: context.Scope) -> types.Type:
        if not types.INT.is_assignable_from(scope.get_type(self.length)):
            raise errors.InvalidTypeError(
                "length of array constructor has to be an integer"
            )
        return types.array_of(self._create_scope(scope).get_type(self.expression))


class Trigonometric(UnaryExpression):
    """
    A trigonometric expression.

    Attributes
    ----------
    operator:
        The trigonometric function to apply
        (:class:`~momba.model.operators.TrigonometricFunction`).
    """

    operator: operators.TrigonometricFunction

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if not operand_type.is_numeric:
            raise errors.InvalidTypeError(
                "expected numeric type for operand of trigonometric function"
            )
        return types.REAL


RealValue = t.Union[t.Literal["π", "e"], float, fractions.Fraction, decimal.Decimal]
NumericValue = t.Union[int, RealValue]
Value = t.Union[bool, NumericValue]

ValueOrExpression = t.Union[Expression, Value]


class ConversionError(ValueError):
    """
    Unable to convert the provided value.
    """


def ensure_expr(value_or_expression: ValueOrExpression) -> Expression:
    """
    Takes a Python value or expression and returns an expression.

    Implicitly converts floats, integers, and booleans to constant expressions.

    Raises :class:`~momba.model.expressions.ConversionError` if the conversion fails.
    """
    if isinstance(value_or_expression, Expression):
        return value_or_expression
    elif isinstance(value_or_expression, bool):
        return BooleanConstant(value_or_expression)
    elif isinstance(value_or_expression, int):
        return IntegerConstant(value_or_expression)
    elif isinstance(value_or_expression, (float, fractions.Fraction, decimal.Decimal)):
        return RealConstant(fractions.Fraction(value_or_expression))
    elif isinstance(value_or_expression, str):
        try:
            return RealConstant(_NAMED_REAL_MAP[value_or_expression])
        except KeyError:
            pass
    raise ConversionError(
        f"unable to convert Python value {value_or_expression!r} to expression"
    )


def ite(
    condition: ValueOrExpression,
    consequence: ValueOrExpression,
    alternative: ValueOrExpression,
) -> Expression:
    """
    Constructs a conditional expression implicitly converting the arguments.
    """
    return Conditional(
        ensure_expr(condition), ensure_expr(consequence), ensure_expr(alternative)
    )


BinaryConstructor = t.Callable[[ValueOrExpression, ValueOrExpression], Expression]


def _boolean_binary_expression(
    operator: operators.BooleanOperator, expressions: t.Sequence[ValueOrExpression]
) -> Expression:
    if len(expressions) == 1:
        return ensure_expr(expressions[0])
    result = Boolean(
        operator,
        ensure_expr(expressions[0]),
        ensure_expr(expressions[1]),
    )
    for operand in expressions[2:]:
        result = Boolean(operator, result, ensure_expr(operand))
    return result


def logic_not(operand: ValueOrExpression) -> Expression:
    """
    Constructs a logical negation expression.
    """
    return Not(operators.NotOperator.NOT, ensure_expr(operand))


def logic_any(*expressions: ValueOrExpression) -> Expression:
    """
    Constructs a disjunction over the provided expressions.

    Returns :attr:`~momba.model.expressions.FALSE` if there are no expressions.
    """
    if len(expressions) == 0:
        return BooleanConstant(False)
    return logic_or(*expressions)


def logic_or(*expressions: ValueOrExpression) -> Expression:
    """
    Constructs a disjunction over the provided expressions.
    """
    return _boolean_binary_expression(operators.BooleanOperator.OR, expressions)


def logic_all(*expressions: ValueOrExpression) -> Expression:
    """
    Constructs a conjunction over the provided expressions.

    Returns :attr:`~momba.model.expressions.TRUE` if there are no expressions.
    """
    if len(expressions) == 0:
        return BooleanConstant(True)
    return logic_and(*expressions)


def logic_and(*expressions: ValueOrExpression) -> Expression:
    """
    Constructs a conjunction over the provided expressions.
    """
    return _boolean_binary_expression(operators.BooleanOperator.AND, expressions)


def logic_xor(*expressions: ValueOrExpression) -> Expression:
    """
    Constructs an exclusive disjunction over the provided expressions.
    """
    return _boolean_binary_expression(operators.BooleanOperator.XOR, expressions)


def logic_implies(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a logical implication.
    """
    return Boolean(
        operators.BooleanOperator.IMPLY, ensure_expr(left), ensure_expr(right)
    )


def logic_equiv(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a logical equivalence.
    """
    return Boolean(
        operators.BooleanOperator.EQUIV, ensure_expr(left), ensure_expr(right)
    )


def equals(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs an *equality* expression.
    """
    return Equality(
        operators.EqualityOperator.EQ, ensure_expr(left), ensure_expr(right)
    )


def not_equals(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs an *inequality* expression.
    """
    return Equality(
        operators.EqualityOperator.NEQ, ensure_expr(left), ensure_expr(right)
    )


def less(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a *less than* expression.
    """
    return Comparison(
        operators.ComparisonOperator.LT, ensure_expr(left), ensure_expr(right)
    )


def less_or_equal(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a *less than or equal to* expression.
    """
    return Comparison(
        operators.ComparisonOperator.LE, ensure_expr(left), ensure_expr(right)
    )


def greater(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a *greater than* expression.
    """
    return Comparison(
        operators.ComparisonOperator.GT, ensure_expr(left), ensure_expr(right)
    )


def greater_or_equal(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs a *greater than or equal to* expression.
    """
    return Comparison(
        operators.ComparisonOperator.GE, ensure_expr(left), ensure_expr(right)
    )


def add(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    """
    Constructs an arithmetic addition expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.ADD, ensure_expr(left), ensure_expr(right)
    )


def sub(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs an arithmetic substraction expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.SUB, ensure_expr(left), ensure_expr(right)
    )


def mul(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs an arithmetic multiplication expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MUL, ensure_expr(left), ensure_expr(right)
    )


def mod(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs an euclidean remainder expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MOD, ensure_expr(left), ensure_expr(right)
    )


def real_div(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs a real-division expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.REAL_DIV,
        ensure_expr(left),
        ensure_expr(right),
    )


def log(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs a logarithm expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.LOG, ensure_expr(left), ensure_expr(right)
    )


def power(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs a power expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.POW, ensure_expr(left), ensure_expr(right)
    )


def minimum(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs a minimum expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MIN, ensure_expr(left), ensure_expr(right)
    )


def maximum(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs a maximum expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MAX, ensure_expr(left), ensure_expr(right)
    )


def floor_div(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    """
    Constructs an euclidean division expression.
    """
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.FLOOR_DIV,
        ensure_expr(left),
        ensure_expr(right),
    )


UnaryConstructor = t.Callable[[ValueOrExpression], Expression]


def floor(operand: ValueOrExpression) -> Expression:
    """
    Constructs a floor expression.
    """
    return ArithmeticUnary(
        operators.ArithmeticUnaryOperator.FLOOR, ensure_expr(operand)
    )


def ceil(operand: ValueOrExpression) -> Expression:
    """
    Constructs a ceil expression.
    """
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.CEIL, ensure_expr(operand))


def absolute(operand: ValueOrExpression) -> Expression:
    """
    Constructs an absolute value expression.
    """
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.ABS, ensure_expr(operand))


def sgn(operand: ValueOrExpression) -> Expression:
    """
    Constructs a sign expression.
    """
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.SGN, ensure_expr(operand))


def trunc(operand: ValueOrExpression) -> Expression:
    """
    Constructs a truncate expression.
    """
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.TRC, ensure_expr(operand))


def name(identifier: str) -> Name:
    """
    Constructs a name expression.
    """
    return Name(identifier)
