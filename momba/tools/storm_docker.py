# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import fractions
import pathlib
import re
import tempfile

from .. import jani, model
from ..analysis import checkers


try:
    import docker
    import docker.types
except ImportError:
    raise ImportError(
        "Missing optional dependency `docker`.\n"
        "Using Storm in a Docker container requires installing `docker`."
    )


DOCKER_IMAGE = "movesrwth/storm"


# XXX: is there a better way to do this?
_result_regex = re.compile(
    r"Model checking property \"(?P<prop_name>[^\"]+)\""
    r".*?"
    r"Result \(for initial states\): (?P<prop_value>\d+(\.\d+)?)",
    flags=re.DOTALL,
)


class Toolset:
    """Interface to Storm running in a Docker container."""

    client: t.Any
    tag: str

    def __init__(self, client: t.Optional[t.Any] = None, tag: str = "travis") -> None:
        self.client = client or docker.from_env()
        self.tag = tag

    def pull(self) -> None:
        self.client.images.pull(DOCKER_IMAGE, tag=self.tag)

    def run(self, arguments: t.Sequence[str], mounts: t.Sequence[t.Any] = ()) -> str:
        """Runs storm with the provided arguments and mounts."""
        command = ["sh", "-c", "'", "./storm"]
        command.extend(arguments)
        command.extend((";", "exit 0", "'"))
        return self.client.containers.run(
            f"{DOCKER_IMAGE}:{self.tag}",
            command=" ".join(command),
            working_dir="/opt/storm/build/bin/",
            auto_remove=True,
            mounts=list(mounts),
        ).decode("utf-8")


@d.dataclass(frozen=True, eq=False)
class StormChecker(checkers.Checker):
    """Checker implementation for Storm running in Docker."""

    toolset: Toolset
    """The toolset to use."""

    engine: str = "dd"
    """The engine to use."""

    @property
    def description(self) -> str:
        return f"Storm in Docker (engine = {self.engine})"

    def check(
        self,
        network: model.Network,
        *,
        properties: t.Optional[checkers.Properties] = None,
        property_names: t.Optional[t.Iterable[str]] = None,
    ) -> checkers.Result:
        with tempfile.TemporaryDirectory(prefix="momba") as directory_name:
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
                jani.dump_model(network, properties=named_properties), encoding="utf-8"
            )
            output = self.toolset.run(
                (
                    "--jani",
                    "/tmp/momba-jani/input.jani",
                    "--janiproperty",
                    "--engine",
                    self.engine,
                ),
                mounts=[
                    docker.types.Mount("/tmp/momba-jani", directory_name, type="bind")
                ],
            )
        return {
            match.group("prop_name"): fractions.Fraction(match.group("prop_value"))
            for match in _result_regex.finditer(output)
        }


toolset = Toolset()

checker_sparse = StormChecker(toolset, engine="sparse")
checker_dd = StormChecker(toolset, engine="dd")

checker = checker_sparse
