# -*- coding:utf-8 -*-
#
# Copyright (C) 2020-2021, Saarland University
# Copyright (C) 2020-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import typing as t

import collections
import gc
import pathlib
import sys
import time

from mxu.maps import FrozenMap


Classes = t.Union[type, t.Tuple[type, ...]]
Objects = t.Mapping[type, t.Sequence[t.Any]]

ALL_CLASSES: t.Tuple[type, ...] = (object,)

MAX_DEPTH = 5

_CONTAINERS = (list, tuple, dict, FrozenMap)


def fmt_size(
    size: t.Union[float, int],
) -> str:
    result: str
    for unit in ("B", "KB", "MB", "GB"):
        result = f"{size} {unit}"
        if size > 1000:
            size /= 1000
        else:
            break
    return result


def get_objects(classes: Classes = ALL_CLASSES, prefix: str = "momba") -> Objects:
    objects: t.Dict[type, t.List[t.Any]] = collections.defaultdict(list)
    for obj in gc.get_objects():
        cls = obj.__class__
        if cls.__module__.startswith(prefix):
            if issubclass(cls, classes):
                objects[obj.__class__].append(obj)
    return objects


def compute_usage(objects: Objects) -> t.Mapping[type, int]:
    return {
        cls: sum(sys.getsizeof(obj) for obj in cls_objects)
        for cls, cls_objects in objects.items()
    }


def compute_counts(objects: Objects) -> t.Mapping[type, int]:
    return {cls: len(cls_objects) for cls, cls_objects in objects.items()}


def _collect_referrers(obj: t.Any, counts: t.Dict[type, int], depth: int = 0) -> None:
    for referrer in gc.get_referrers(obj):
        if depth < MAX_DEPTH and isinstance(referrer, _CONTAINERS):
            _collect_referrers(referrer, counts, depth + 1)
        else:
            counts[referrer.__class__] += 1


def compute_referrers(objects: Objects) -> t.Mapping[type, t.Mapping[type, int]]:
    referrers: t.Dict[type, t.Dict[type, int]] = {}
    for cls, cls_objects in objects.items():
        counts = referrers[cls] = collections.defaultdict(int)
        for obj in cls_objects:
            _collect_referrers(obj, counts)
    return referrers


def dump_statistics(path: pathlib.Path, prefix: str = "momba") -> None:
    with path.open("at", encoding="utf-8") as file:
        file.write(f"dump_statistics @ {time.monotonic_ns()}\n")
        objects = get_objects(prefix=prefix)
        usage = compute_usage(objects)
        counts = compute_counts(objects)
        referrers = compute_referrers(objects)
        for cls in sorted(objects.keys(), key=lambda cls: usage[cls], reverse=True):
            cls_name = f"{cls.__module__}.{cls.__name__}"
            file.write(
                f"  {cls_name} | count: {counts[cls]} | size: {fmt_size(usage[cls])}\n"
                f"  referrrers: {sum(referrers[cls].values())}\n"
            )
            for ref_cls, ref_count in sorted(
                referrers[cls].items(), key=lambda item: item[1], reverse=True
            ):
                ref_cls_name = f"{ref_cls.__module__}.{ref_cls.__name__}"
                file.write(f"    {ref_cls_name}: {ref_count}\n")
