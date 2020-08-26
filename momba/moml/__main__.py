# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib
import sys

import click

from .parser import TokenStream, parse_moml, MomlSyntaxError

from .. import model

from ..explore import engine
from ..pure import pta

from ..ext.jani import dump_model


@click.group()
def main() -> None:
    """
    The *Momba Modeling Language* (MOML) tool.
    """


@main.command()
@click.argument("moml_file", type=pathlib.Path)
@click.argument("output_directory", type=pathlib.Path)
@click.option(
    "--indent", type=int, default=None, help="Indentation to use for the JANI files.",
)
@click.option(
    "--network",
    "networks",
    type=str,
    multiple=True,
    help="Name of the network(s) to export.",
)
def export(
    moml_file: pathlib.Path,
    output_directory: pathlib.Path,
    indent: t.Optional[int] = None,
    networks: t.Optional[t.Sequence[str]] = None,
) -> None:
    """
    Exports a MOML model to a set of JANI models.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    ctx = parse_moml(TokenStream(moml_file.read_text(encoding="utf-8")))

    for network in ctx.networks:
        if networks and network.name not in networks:
            continue
        print(f"Exporting network `{network.name}` to JANI...")
        (output_directory / f"model_{network.name}.jani").write_bytes(
            dump_model(network, indent=indent,)
        )


@main.command()
@click.argument("moml_file", type=pathlib.Path)
def check(moml_file: pathlib.Path) -> None:
    """
    Checks the provided MOML file for validity.
    """
    try:
        parse_moml(TokenStream(moml_file.read_text(encoding="utf-8")))
    except MomlSyntaxError as error:
        print(error.args[0])
        sys.exit(1)


@main.command()
@click.argument("moml_file", type=pathlib.Path)
@click.option(
    "--network", "network_name", type=str, help="Name of the network to simulate.",
)
@click.option("--steps", type=int, default=1000, help="Number of steps to simulate.")
@click.option(
    "--weight",
    "weights",
    type=(str, int),
    multiple=True,
    help="Weights for the oracle.",
)
@click.option(
    "--watch-global",
    "watch_globals",
    type=str,
    multiple=True,
    help="Global variables to monitor the value of.",
)
@click.option(
    "--default-weight",
    type=int,
    default=10000,
    help="The default weight for non-deterministic options.",
)
def simulate(
    moml_file: pathlib.Path,
    network_name: t.Optional[str],
    steps: int,
    weights: t.List[t.Tuple[str, int]],
    default_weight: int,
    watch_globals: t.List[str],
) -> None:
    """
    Checks the provided MOML file for validity.
    """
    try:
        ctx = parse_moml(TokenStream(moml_file.read_text(encoding="utf-8")))
        if ctx.model_type not in engine.MombaPTA.SUPPORTED_TYPES:
            print(f"Unsupported model type {ctx.model_type.name}!")
            sys.exit(1)
        if not ctx.networks:
            print("No networks have been specified in the supplied model.")
            sys.exit(1)
        if len(ctx.networks) != 1 and network_name is None:
            print("Multiple networks found — please specify a network name.")
            sys.exit(1)
        network: t.Optional[model.Network] = None
        for network in ctx.networks:
            if network.name == network_name or network_name is None:
                break
        else:
            print(f"No network with name {network_name!r} found.")
            sys.exit(1)
        assert network is not None

        simulator = pta.PTASimulator(
            engine.MombaPTA(network),
            oracle=engine.ActionTypeOracle(
                {
                    network.ctx.get_action_type_by_name(name): weight
                    for name, weight in weights
                },
                default_weight,
            ),
        )

        counter = 0
        for step in simulator.run():
            action = "τ"
            if step.decision.edge.action:
                arguments = ", ".join(map(str, step.decision.edge.action.arguments))
                action = f"{step.decision.edge.action.action_type.name}({arguments})"
            print(f"t = {step.time}, {action}")
            for name in watch_globals:
                print(
                    f"  {name}:", step.destination.location.state.binding[name],
                )
            counter += 1
            if counter >= steps:
                break
    except MomlSyntaxError as error:
        print(error.args[0])
        sys.exit(1)


if __name__ == "__main__":
    main()
