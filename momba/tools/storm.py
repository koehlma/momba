# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import fractions
import pathlib
import subprocess
import tempfile
import re

from .. import model
from ..analysis import checkers
from ..jani import dump_model

from .errors import ToolError, ToolTimeoutError


Timeout = t.Optional[t.Union[float, int]]
Command = t.Sequence[t.Union[str, pathlib.Path]]


# XXX: is there a better way to do this?
_result_regex = re.compile(
    r"Model checking property \"(?P<prop_name>[^\"]+)\""
    r".*?"
    r"Result \(for initial states\): (?P<prop_value>\d+(\.\d+)?)",
    flags=re.DOTALL,
)


@d.dataclass(frozen=True)
class Output:
    stdout: str
    stderr: str


@d.dataclass(eq=False)
class Toolset:
    """Interface to Storm."""

    executable: t.Union[str, pathlib.Path]
    """Path to the executable."""

    environment: t.Optional[t.Mapping[str, str]] = None
    """Environment variables for execution."""

    def invoke(
        self,
        arguments: t.Iterable[t.Union[str, int, float, pathlib.Path]],
        *,
        timeout: Timeout = None,
    ) -> Output:
        """Runs storm with the provided arguments."""
        command: Command = (
            self.executable,
            *map(str, arguments),
        )
        try:
            process = subprocess.run(
                command,
                env=self.environment,
                timeout=timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as timeout_error:
            raise ToolTimeoutError(
                "timeout expired during invocation of `storm`",
                command=command,
                stdout=timeout_error.stdout,
                stderr=timeout_error.stderr,
            )
        if process.returncode != 0:
            raise ToolError(
                f"`storm` terminated with non-zero returncode {process.returncode}",
                command=command,
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        return Output(process.stdout.decode("utf-8"), process.stderr.decode("utf-8"))


@d.dataclass(frozen=True, eq=False)
class StormChecker(checkers.Checker):
    toolset: Toolset

    engine: str = "dd"

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
            output = self.toolset.invoke(
                ("--jani", input_file, "--janiproperty", "--engine", self.engine)
            )
        return {
            match.group("prop_name"): fractions.Fraction(match.group("prop_value"))
            for match in _result_regex.finditer(output.stdout)
        }


toolset = Toolset("storm")

checker_sparse = StormChecker(toolset, engine="sparse")
checker_dd = StormChecker(toolset, engine="dd")

checker = checker_sparse


def get_checker(*, accept_license: bool) -> checkers.Checker:
    """Returns an instance of :class:`~momba.analysis.Checker`."""
    try:
        from . import storm_docker

        return storm_docker.checker
    except ImportError:
        return checker
