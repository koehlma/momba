# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import enum
import inspect
import warnings

from .clstools import get_subclasses


class NoImplementationError(Exception):
    pass


class NoImplementationWarning(UserWarning):
    pass


def check_singledispatch(
    function: t.Callable[..., t.Any],
    superclass: type,
    *,
    error: bool = False,
    recursive: bool = True,
    ignore: t.AbstractSet[type] = frozenset(),
) -> None:
    if not hasattr(function, "registry") or not hasattr(
        function.registry, "keys"  # type:ignore
    ):
        raise ValueError(f"{function} is not a singledispatch function")
    subclasses = set(get_subclasses(superclass, recursive=recursive))
    subclasses -= ignore
    for cls in function.registry.keys():  # type: ignore
        if not issubclass(cls, superclass):
            continue
        subclasses.discard(cls)
        if recursive:
            subclasses -= get_subclasses(cls, recursive=True)
    for subclass in subclasses:
        if inspect.isabstract(subclass):
            continue
        msg = (
            f"implementation of {function.__name__} for subclass {subclass} is missing"
        )
        if error:
            raise NoImplementationError(msg)
        else:
            warnings.warn(NoImplementationWarning(msg), stacklevel=2)


class NoEntryError(Exception):
    pass


class NoEntryWarning(UserWarning):
    pass


EnumType = t.TypeVar("EnumType", bound=enum.Enum)


def check_enum_map(
    enum_typ: t.Type[EnumType],
    mapping: t.Mapping[EnumType, t.Any],
    *,
    error: bool = False,
) -> None:
    for entry in enum_typ:
        if entry in mapping:
            continue
        msg = f"entry for {entry} is missing"
        if error:
            raise NoEntryError(msg)
        else:
            warnings.warn(NoEntryWarning(msg))
