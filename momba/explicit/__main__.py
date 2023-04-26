# -*- coding:utf-8 -*-
#
# Copyright (C) 2023, Saarland University
# Copyright (C) 2023, Maximilian Köhl <koehl@cs.uni-saarland.de>

import json
import pathlib

import click


from .. import jani

from .translator import translate_model


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("jani_model", type=pathlib.Path)
@click.argument("momba_model", type=pathlib.Path)
def convert(jani_model: pathlib.Path, momba_model: pathlib.Path) -> None:
    network = jani.load_model(jani_model.read_text("utf-8"))
    momba_model.write_text(json.dumps(translate_model(network)))


@main.command()
@click.argument("path", type=pathlib.Path)
def convert_all(path: pathlib.Path) -> None:
    for path in path.glob("**/*.jani"):
        try:
            network = jani.load_model(path.read_text("utf-8-sig"))
            path.with_suffix(".momba.json").write_text(
                json.dumps(translate_model(network))
            )
        except Exception as error:
            print(f"Error converting model `{path}`.")
            print(error)


if __name__ == "__main__":
    main()
