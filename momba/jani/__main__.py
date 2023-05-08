# -*- coding:utf-8 -*-
#
# Copyright (C) 2023, Saarland University
# Copyright (C) 2023, Maximilian Köhl <koehl@cs.uni-saarland.de>

import pathlib

import click

from .. import jani


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("jani_in", type=pathlib.Path)
@click.argument("jani_out", type=pathlib.Path)
def pass_through(jani_in: pathlib.Path, jani_out: pathlib.Path) -> None:
    model = jani.load_model(jani_in.read_text("utf-8"))
    jani_out.write_text(jani.dump_model(model))


if __name__ == "__main__":
    main()
