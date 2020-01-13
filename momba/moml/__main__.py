# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib
import sys

import click

from .parser import TokenStream, parse_moml, MomlSyntaxError

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


if __name__ == "__main__":
    main()
