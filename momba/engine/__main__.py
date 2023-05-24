# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib

import click

from ..engine import translator
from .. import jani
from . import objectives


import json


@click.group()
def main() -> None:
    """
    Toolset for working with MOML models.
    """


def parse_constants(cmd_input: str) -> dict:
    """
    Input expected:
    Cons_1:Val_1,...,Const_k:Val_k.
    And for all i Val_i in (Int, Bool)
    """
    data = {}
    for l in cmd_input.split(","):
        if l.split("=")[1].isdecimal():
            data[l.split("=")[0]] = int(l.split("=")[1])
        elif (l.split("=")[1]).lower() in ("false", "true"):
            match l.split("=")[1]:
                case "False":
                    data[l.split("=")[0]] = False
                case "false":
                    data[l.split("=")[0]] = False
                case "True":
                    data[l.split("=")[0]] = True
                case "true":
                    data[l.split("=")[0]] = True
    return data


@main.command()
@click.argument(
    "model_path",
    metavar="MODEL",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument(
    "output_path",
    metavar="OUTPUT",
    type=click.Path(exists=False, dir_okay=True, writable=True),
)
@click.option("-c", "--consts")
def translate(model_path: str, output_path: str, consts=None) -> None:
    """
    Translates a MOML model to MombaIR.
    """

    p = pathlib.Path(f"{output_path}/")
    p.mkdir(parents=True, exist_ok=True)

    parameters = parse_constants(consts) if consts is not None else None

    network = jani.load_model(pathlib.Path(model_path).read_text("utf-8"))
    translation = translator.translate_network(network, parameters=parameters)
    properties = network.ctx.properties

    for i, (name, definition) in enumerate(properties.items()):
        txt = name.lower().replace(" ", "_").strip()
        print(f"Saving property: {txt}")
        obj = objectives.extract_objective(definition.expression)
        goal = translation.translate_global_expression(obj.goal_predicate)
        # pathlib.Path(f"{output_path}/prop_{i}.json").write_text(goal, "utf-8")
        pathlib.Path(f"{output_path}/prop_{txt}.json").write_text(goal, "utf-8")

    pathlib.Path(f"{output_path}/model.json").write_text(
        translation.json_network, "utf-8"
    )


if __name__ == "__main__":
    main()
