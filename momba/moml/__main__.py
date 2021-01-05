# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib

import click

from ..engine import translator

from . import parse


@click.group()
def main() -> None:
    """
    Toolset for working with MOML models.
    """


@main.command()
@click.argument(
    "model_path",
    metavar="MODEL",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument(
    "output_path",
    metavar="OUTPUT",
    type=click.Path(exists=False, dir_okay=False, writable=True),
)
@click.option(
    "--network", "network_name", help="The name of the network to translate.", type=str
)
def translate(model_path: str, output_path: str, network_name: t.Optional[str]) -> None:
    """
    Translates a MOML model to MombaIR.
    """

    ctx = parse(pathlib.Path(model_path).read_text("utf-8"))

    if network_name is None:
        if len(ctx.networks) != 1:
            network_names = {network.name for network in ctx.networks}
            print(
                f"Please specify a network name. Valid network names are {network_names}."
            )
            return
        (network,) = ctx.networks
    else:
        network = ctx.get_network_by_name(network_name)

    translation = translator.translate_network(network)

    pathlib.Path(output_path).write_text(translation.json_network, "utf-8")


if __name__ == "__main__":
    main()
