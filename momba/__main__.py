# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """
    The Momba toolset.
    """
