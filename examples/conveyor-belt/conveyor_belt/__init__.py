# -*- coding: utf-8 -*-
#
# Copyright (C) 2022, Saarland University
# Copyright (C) 2022, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
A family of industrial automation conveyor belt models.
"""

from .builder import Scenario, Sensor, build_model


__all__ = ["Scenario", "Sensor", "build_model"]
