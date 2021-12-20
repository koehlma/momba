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


Classes = t.Union[type, t.Tuple[type, ...]]
Objects = t.Mapping[type, t.Sequence[t.Any]]

ALL_CLASSES: t.Tuple[type, ...] = (object,)


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


def get_objects(classes: Classes = ALL_CLASSES) -> Objects:
    objects: t.Dict[type, t.List[t.Any]] = collections.defaultdict(list)
    for obj in gc.get_objects():
        cls = obj.__class__
        if cls.__module__.startswith("momba"):
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


def compute_referrers(objects: Objects) -> t.Mapping[type, t.Mapping[type, int]]:
    referrers: t.Dict[type, t.Dict[type, int]] = {}
    for cls, cls_objects in objects.items():
        cls_referrers = referrers[cls] = collections.defaultdict(int)
        for obj in cls_objects:
            for referrer in gc.get_referrers(obj):
                cls_referrers[referrer.__class__] += 1
    return referrers


def dump_statistics(path: pathlib.Path) -> None:
    with path.open("at", encoding="utf-8") as file:
        file.write(f"dump_statistics @ {time.monotonic_ns()}\n")
        objects = get_objects()
        usage = compute_usage(objects)
        counts = compute_counts(objects)
        referrers = compute_referrers(objects)
        for cls in sorted(objects.keys(), key=lambda cls: usage[cls], reverse=True):
            file.write(
                f"  {cls.__name__} | count: {counts[cls]} | size: {fmt_size(usage[cls])}\n"
                f"  referrrers: {sum(referrers[cls].values())}\n"
            )
            for ref_cls, ref_count in sorted(
                referrers[cls].items(), key=lambda item: item[1], reverse=True
            ):
                file.write(f"    {ref_cls.__name__}: {ref_count}\n")
