# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from . import model


_CELL_COLORS = {
    model.CellType.BLANK: "black",
    model.CellType.BLOCKED: "red",
    model.CellType.GOAL: "green",
    model.CellType.START: "blue",
}

_CAR_COLOR = "yellow"


@d.dataclass(frozen=True)
class SVG:
    source: str


def format_track(track: model.Track, car: t.Optional[model.Coordinate] = None) -> SVG:
    height_px = track.height * 10
    width_px = track.width * 10
    svg = ['<?xml version="1.0" encoding="UTF-8"?>']
    svg.append(
        f"""<svg
            xmlns="http://www.w3.org/2000/svg"
            version="1.1" baseProfile="full"
            width="{width_px}" height="{height_px}"
            viewBox="0 0 {width_px} {height_px}">
        """
    )
    for x in range(track.width):
        for y in range(track.height):
            cell = model.Coordinate(x, y)
            if cell == car:
                color = _CAR_COLOR
            else:
                color = _CELL_COLORS[track.get_cell_type(cell)]
            svg.append(
                f"""<rect
                    x="{x * 10}"
                    y="{y * 10}"
                    width="10"
                    height="10"
                    fill="{color}"
                    stroke="black"
                    stroke-width="2pt"
                    />
                """
            )
    svg.append("</svg>")
    return SVG("\n".join(svg))
