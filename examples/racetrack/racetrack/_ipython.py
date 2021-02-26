# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from . import model

from . import svg


def _track_repr_svg_(track: model.Track) -> str:
    return svg.format_track(track).source


def _svg_repr_svg_(svg: svg.SVG) -> str:
    return svg.source


model.Track._repr_svg_ = _track_repr_svg_
svg.SVG._repr_svg_ = _svg_repr_svg_
