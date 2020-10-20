# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t


def get_subclasses(cls: type, recursive: bool = True) -> t.AbstractSet[type]:
    subclasses: t.Set[type] = set()
    if recursive:
        queue = [cls]
        while queue:
            element = queue.pop()
            subclasses.update(element.__subclasses__())
            queue.extend(element.__subclasses__())
    else:
        subclasses.update(cls.__subclasses__())
    return subclasses
