# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import fractions
import json
import pathlib
import subprocess
import sys
import tempfile


from .. import model
from ..analysis import checkers
from ..jani import dump_model

from .errors import ToolError, ToolTimeoutError


Timeout = t.Optional[t.Union[float, int]]
Command = t.Sequence[t.Union[str, pathlib.Path]]


_MODEST_URL = "https://www.modestchecker.net/Downloads/"

_USER_HOME = pathlib.Path("~").expanduser()

_is_windows = sys.platform == "win32"
_is_macos = sys.platform == "darwin"


if _is_windows:
    _DATA_PATH = _USER_HOME / "Application Data" / "koehlma" / "Momba"
    _MODEST_PLATFORM = "win"
elif _is_macos:
    _DATA_PATH = _USER_HOME / "Library" / "Application Support" / "Momba"
    _MODEST_PLATFORM = "osx"
else:
    _DATA_PATH = _USER_HOME / ".local" / "share" / "momba"
    _MODEST_PLATFORM = "linux"


_MODEST_PATH = _DATA_PATH / "modest"

if _is_windows:
    _MODEST_EXE = _MODEST_PATH / "Modest" / "modest.exe"
else:
    _MODEST_EXE = _MODEST_PATH / "Modest" / "modest"


_FILE_REGEX = (
    r"""<a href="(?P<file>Modest-Toolset-(?P<version>[v.0-9\-a-z]+?)"""
    r"""-(?P<platform>[a-z]+)-x64\.zip)"""
)


def _install_locally() -> None:
    import os
    import re
    import stat
    import zipfile

    from urllib import request

    response = request.urlopen(_MODEST_URL)
    if response.status != 200:
        raise Exception(f"unable to download Modest (error {response.status})")
    page = response.read().decode("utf-8")
    for match in re.finditer(_FILE_REGEX, page):
        if match["platform"] != _MODEST_PLATFORM:
            continue
        response = request.urlopen(f"{_MODEST_URL}{match['file']}")
        if response.status != 200:
            raise Exception(f"unable to download Modest (error {response.status})")
        with tempfile.TemporaryDirectory(prefix="momba") as temp_directory:
            zip_path = pathlib.Path(temp_directory) / "modest.zip"
            zip_path.write_bytes(response.read())
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(_MODEST_PATH)
        os.chmod(_MODEST_EXE, os.stat(_MODEST_EXE).st_mode | stat.S_IEXEC)
        break
    else:
        raise Exception(f"unable to download Modest for platform {_MODEST_PLATFORM}")


def _setup_locally() -> None:
    if not _MODEST_EXE.exists():
        _install_locally()


@d.dataclass(eq=False)
class Toolset:
    """Interface to the Modest Toolset."""

    executable: t.Union[str, pathlib.Path]
    """Path to the executable of the Modest Toolset."""

    environment: t.Optional[t.Mapping[str, str]] = None
    """Environment variables for the execution."""

    def check(
        self,
        arguments: t.Iterable[t.Union[str, int, float, pathlib.Path]],
        *,
        timeout: Timeout = None,
        capture_output: bool = True,
    ) -> t.Mapping[str, t.Any]:
        """Runs `modest check` with the provided arguments."""
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

    @property
    def description(self) -> str:
        return "Modest (mcsta)"

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
            if properties is not None:
                named_properties.update(properties)
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


def get_checker(*, accept_license: bool) -> checkers.Checker:
    """Returns an instance of :class:`~momba.analysis.Checker`."""
    assert accept_license, "you need to accept the license of the Modest Toolset"
    _setup_locally()
    toolset = Toolset(_MODEST_EXE)
    return ModestChecker(toolset)
