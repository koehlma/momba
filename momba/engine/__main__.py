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


@click.group()
def main() -> None:
    """
    Toolset for working with MOML models.
    """


def is_float(string: str) -> bool:
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False


def parse_constants(cmd_input: str) -> t.Any:
    """
    Input expected:
    Cons_1:Val_1,...,Const_k:Val_k.
    And for all i Val_i in (Int, Bool)
    """
    data: t.Dict[str, t.Union[bool, int, float]] = {}
    for l in cmd_input.split(","):  # noqa: E741
        idx = l.split("=")[0].strip()
        if l.split("=")[1].isnumeric():
            data[idx] = int(l.split("=")[1])
        elif is_float(l.split("=")[1]):
            data[idx] = float(l.split("=")[1])
        elif (l.split("=")[1]).lower() in ("false", "true"):
            match l.split("=")[1]:
                case "False":
                    data[idx] = False
                case "false":
                    data[idx] = False
                case "True":
                    data[idx] = True
                case "true":
                    data[idx] = True
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
def translate(model_path: str, output_path: str, consts: t.Any = None) -> None:
    """
    Translates a MOML model to MombaIR.
    """

    p = pathlib.Path(f"{output_path}/")
    p.mkdir(parents=True, exist_ok=True)

    parameters = parse_constants(consts) if consts is not None else None

    network = jani.load_model(pathlib.Path(model_path).read_text("utf-8"))
    translation = translator.translate_network(network, parameters=parameters)
    properties = network.ctx.properties

    for name, definition in properties.items():
        txt = name.lower().replace(" ", "_").strip()
        print(f"Saving property: {txt}")

        obj = objectives.extract_objective(definition.expression)
        goal = translation.translate_global_expression(obj.goal_predicate)
        dead = translation.translate_global_expression(obj.dead_predicate)

        prop = (
            '{"operator":"' + str(obj.op) + '","goal":' + goal + ',"dead":' + dead + "}"
        )

        # prop = json.dumps({"operator": str(obj.op), "goal": goal, "dead": dead})

        pathlib.Path(f"{output_path}/prop_{txt}.json").write_text(prop, "utf-8")

    pathlib.Path(f"{output_path}/model.json").write_text(
        translation.json_network, "utf-8"
    )


if __name__ == "__main__":
    main()
