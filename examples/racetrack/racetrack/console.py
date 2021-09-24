# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>
# Copyright (C) 2020, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import colorama

from . import model


colorama.init()


_BACKGROUND_COLORS = {
    model.CellType.BLANK: colorama.Back.BLACK,
    model.CellType.BLOCKED: colorama.Back.RED,
    model.CellType.GOAL: colorama.Back.GREEN,
    model.CellType.START: colorama.Back.BLUE,
}


def format_cell(
    track: model.Track, cell: model.Coordinate, car: t.Optional[model.Coordinate] = None
) -> str:
    typ = track.get_cell_type(cell)
    background = _BACKGROUND_COLORS[typ]
    is_car_cell = cell == car
    if is_car_cell:
        foreground = (
            colorama.Fore.RED if typ is model.CellType.GOAL else colorama.Fore.YELLOW
        )
    else:
        foreground = (
            colorama.Fore.WHITE if typ is model.CellType.BLANK else colorama.Fore.BLACK
        )
    symbol = "*" if is_car_cell else "."
    return f"{background}{foreground}{symbol}{colorama.Style.RESET_ALL}"


def format_track(track: model.Track, car: t.Optional[model.Coordinate] = None) -> str:
    lines: t.List[str] = []
    for y in range(track.height):
        lines.append(
            "".join(
                format_cell(track, model.Coordinate(x, y), car=car)
                for x in range(track.width)
            )
        )
    return "\n".join(lines)
