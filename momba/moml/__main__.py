# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import argparse
import pathlib

from .parser import TokenStream, parse_moml

from ..ext.jani import dump_model


parser = argparse.ArgumentParser(description="Converts models between MOML and JANI.")
parser.add_argument("moml_file", type=pathlib.Path)
parser.add_argument("output_directory", type=pathlib.Path)
parser.add_argument("--indent", type=int, default=None)


def main(args: t.Optional[t.Sequence[str]] = None) -> None:
    namespace = parser.parse_args(args)

    assert namespace.moml_file.exists()

    namespace.output_directory.mkdir(parents=True, exist_ok=True)

    ctx = parse_moml(TokenStream(namespace.moml_file.read_text(encoding="utf-8")))

    for network in ctx.networks:
        print(f"Exporting network `{network.name}` to JANI...")
        (namespace.output_directory / f"model_{network.name}.jani").write_bytes(
            dump_model(network, indent=namespace.indent,)
        )


if __name__ == "__main__":
    main()
