# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from . import model

_CELL_COLORS = {
    model.CellType.BLANK: "black",
    model.CellType.BLOCKED: "red",
    model.CellType.GOAL: "green",
    model.CellType.START: "blue",
}


def _track_repr_svg_(track: model.Track) -> str:
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
    for cell in track.cells:
        coordinate = track.cell_to_coordinate(cell)
        svg.append(
            f"""<rect
                x="{coordinate.x * 10}"
                y="{coordinate.y * 10}"
                width="10"
                height="10"
                fill="{_CELL_COLORS[track.get_cell_type(cell)]}"
                stroke="black"
                stroke-width="2pt"
                />
            """
        )
    svg.append("</svg>")
    return "\n".join(svg)


model.Track._repr_svg_ = _track_repr_svg_
