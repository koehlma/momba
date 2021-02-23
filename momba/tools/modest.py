# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import fractions
import json
import pathlib
import subprocess
import tempfile

from .. import model
from ..analysis import checkers
from ..jani import dump_model

from .errors import ToolError, ToolTimeoutError


Timeout = t.Optional[t.Union[float, int]]
Command = t.Sequence[t.Union[str, pathlib.Path]]


@d.dataclass(eq=False)
class Toolset:
    executable: t.Union[str, pathlib.Path]

    environment: t.Optional[t.Mapping[str, str]] = None

    def check(
        self,
        arguments: t.Iterable[t.Union[str, int, float, pathlib.Path]],
        *,
        timeout: Timeout = None,
        capture_output: bool = True,
    ) -> t.Mapping[str, t.Any]:
        """ Runs `modest check` with the provided arguments. """
        with tempfile.TemporaryDirectory(prefix="modest") as directory_name:
            output_file = pathlib.Path(directory_name) / "output.json"
            command: Command = (
                self.executable,
                "check",
                "-O",
                output_file,
                "json",
                *map(str, arguments),
            )
            try:
                process = subprocess.run(
                    command,
                    env=self.environment,
                    timeout=timeout,
                    capture_output=capture_output,
                )
            except subprocess.TimeoutExpired as timeout_error:
                raise ToolTimeoutError(
                    "timeout expired during execution of `modest check`",
                    command=command,
                    stdout=timeout_error.stdout,
                    stderr=timeout_error.stderr,
                )
            if process.returncode != 0:
                raise ToolError(
                    f"`modest check` terminated with non-zero returncode {process.returncode}",
                    command=command,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    returncode=process.returncode,
                )
            try:
                return json.loads(output_file.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                raise ToolError(
                    "`modest check` did not generate an output file",
                    command=command,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    returncode=0,
                )


@d.dataclass(frozen=True, eq=False)
class ModestChecker(checkers.Checker):
    toolset: Toolset

    def check(
        self,
        network: model.Network,
        *,
        properties: t.Optional[checkers.Properties] = None,
        property_names: t.Optional[t.Iterable[str]] = None,
    ) -> checkers.Result:
        with tempfile.TemporaryDirectory(prefix="modest") as directory_name:
            input_file = pathlib.Path(directory_name) / "input.jani"
            named_properties: t.Dict[str, model.Expression] = {}
            if properties is None and property_names is None:
                named_properties.update(
                    {
                        definition.name: definition.expression
                        for definition in network.ctx.properties.values()
                    }
                )
            if property_names is not None:
                for name in property_names:
                    named_properties[
                        name
                    ] = network.ctx.get_property_definition_by_name(name).expression
            input_file.write_text(
                dump_model(network, properties=named_properties), encoding="utf-8"
            )
            result = self.toolset.check((input_file,))
            return {
                dataset["property"]: fractions.Fraction(dataset["value"])
                for dataset in result["data"]
                if "property" in dataset
            }


toolset = Toolset("modest")
checker = ModestChecker(toolset)
