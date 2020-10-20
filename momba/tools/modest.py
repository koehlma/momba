# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import json
import pathlib
import subprocess
import tempfile

from .. import model
from ..analysis import checkers
from ..jani import dump_model


Timeout = t.Optional[t.Union[float, int]]
Command = t.Sequence[t.Union[str, pathlib.Path]]


class ModestError(Exception):
    command: Command

    stdout: t.Optional[bytes]
    stderr: t.Optional[bytes]

    returncode: t.Optional[int]

    def __init__(
        self,
        message: str,
        command: Command = (),
        *,
        stdout: t.Optional[bytes] = None,
        stderr: t.Optional[bytes] = None,
        returncode: t.Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class ModestTimeoutError(ModestError):
    pass


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
                raise ModestTimeoutError(
                    "timeout expired during execution of `modest check`",
                    command=command,
                    stdout=timeout_error.stdout,
                    stderr=timeout_error.stderr,
                )
            if process.returncode != 0:
                raise ModestError(
                    f"`modest check` terminated with non-zero returncode {process.returncode}",
                    command=command,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    returncode=process.returncode,
                )
            try:
                return json.loads(output_file.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                raise ModestError(
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
            named_properties: t.Dict[str, model.Property] = {}
            if properties is None and property_names is None:
                named_properties.update(
                    {
                        definition.name: definition.prop
                        for definition in network.ctx.named_properties.values()
                    }
                )
            if property_names is not None:
                for name in property_names:
                    named_properties[name] = network.ctx.get_property_by_name(name)
            input_file.write_bytes(dump_model(network, properties=named_properties))
            result = self.toolset.check((input_file,))
            return {
                dataset["property"]: dataset["value"]
                for dataset in result["data"]
                if "property" in dataset
            }


toolset = Toolset("modest")
checker = ModestChecker(toolset)
