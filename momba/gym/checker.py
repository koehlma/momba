# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import asyncio
import json
import fractions
import pathlib
import struct
import subprocess
import tempfile

from .. import jani, model

from ..tools import modest

from . import generic
from .abstract import Oracle
from .dump_nn import dump_nn


if t.TYPE_CHECKING:
    import torch


HEADER = struct.Struct("!II")  # (num_features, num_actions)
DECISION = struct.Struct("!I")  # (action,)


@d.dataclass(frozen=True)
class ModesOptions:
    max_run_length_as_end: bool = True

    def apply(self, command: t.List[str]) -> None:
        if self.max_run_length_as_end:
            command.append("--max-run-length-as-end")


_DEFAULT_OPTIONS = ModesOptions()


def _apply_actions_observations(
    command: t.List[str], actions: generic.Actions, observations: generic.Observations
) -> None:
    if actions is generic.Actions.EDGE_BY_LABEL:
        command.append("--select-by-label")
    else:
        assert actions is generic.Actions.EDGE_BY_INDEX
    if observations is generic.Observations.LOCAL_AND_GLOBAL:
        command.append("--observations-local-global")
    elif observations is generic.Observations.OMNISCIENT:
        command.append("--observations-omniscient")
    else:
        assert observations is generic.Observations.GLOBAL_ONLY


def _create_vector_decoder(num_features: int) -> struct.Struct:
    return struct.Struct(f"!{num_features}f")


def _create_available_decoder(num_actions: int) -> struct.Struct:
    return struct.Struct(f"!{num_actions}?")


@d.dataclass()
class OracleServer:
    oracle: Oracle

    host: str = "0.0.0.0"
    port: t.Optional[int] = None

    server: t.Optional[asyncio.AbstractServer] = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        address = writer.get_extra_info("peername")
        print(f"New connection from {address}.")
        num_features, num_actions = HEADER.unpack(await reader.readexactly(HEADER.size))
        print(f"Features: {num_features}, Actions: {num_actions}")
        state_decoder = _create_vector_decoder(num_features)
        available_decoder = _create_available_decoder(num_actions)
        try:
            while True:
                state = state_decoder.unpack(
                    await reader.readexactly(state_decoder.size)
                )
                available = available_decoder.unpack(
                    await reader.readexactly(available_decoder.size)
                )
                decision = self.oracle(state, available)
                writer.write(DECISION.pack(decision))
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        assert self.server.sockets is not None
        self.port = self.server.sockets[0].getsockname()[1]
        await self.server.start_serving()

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


_NO_PARAMETERS: t.Dict[str, t.Any] = {}


def _load_result(output_file: pathlib.Path) -> t.Mapping[str, fractions.Fraction]:
    result = json.loads(output_file.read_text(encoding="utf-8-sig"))
    return {
        dataset["property"]: fractions.Fraction(dataset["value"])
        for dataset in result["data"]
        if "data" in dataset
        for dataset in dataset["data"]
        if "property" in dataset
    }


async def _check_oracle(
    model_path: pathlib.Path,
    automaton_name: str,
    oracle: Oracle,
    output_file: pathlib.Path,
    *,
    parameters: t.Mapping[str, t.Any],
    toolset: modest.Toolset,
    options: ModesOptions,
    actions: generic.Actions,
    observations: generic.Observations,
) -> None:
    server = OracleServer(oracle)
    await server.start()
    address = f"localhost:{server.port}"
    print(f"Address: {address}")
    arguments: t.List[str] = [
        "modes",
        str(model_path),
        "-O",
        str(output_file),
        "json",
        "-R",
        "oracle",
        "--socket",
        address,
        "-A",
        automaton_name,
        "--threads",
        "1",
    ]
    options.apply(arguments)
    for param_name, param_value in parameters.items():
        arguments.extend(["-E", f"{param_name}={param_value}"])
    _apply_actions_observations(arguments, actions, observations)
    process = await asyncio.subprocess.create_subprocess_exec(
        toolset.executable, *arguments
    )
    await process.wait()


async def check_oracle_async(
    network: model.Network,
    instance: model.Instance,
    oracle: Oracle,
    *,
    parameters: t.Mapping[str, t.Any] = _NO_PARAMETERS,
    toolset: modest.Toolset = modest.toolset,
    options: ModesOptions = _DEFAULT_OPTIONS,
    actions: generic.Actions = generic.Actions.EDGE_BY_INDEX,
    observations: generic.Observations = generic.Observations.GLOBAL_ONLY,
) -> t.Mapping[str, fractions.Fraction]:
    """Checks an arbitrary Python function implementing a decsion agent."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        model_path = temp_path / "model.jani"
        model_path.write_text(jani.dump_model(network), encoding="utf-8")
        output_file = temp_path / "output.json"
        assert instance.automaton.name is not None
        await _check_oracle(
            model_path,
            instance.automaton.name,
            oracle,
            output_file,
            parameters=parameters,
            toolset=toolset,
            options=options,
            actions=actions,
            observations=observations,
        )
        return _load_result(output_file)


def check_oracle(
    network: model.Network,
    instance: model.Instance,
    oracle: Oracle,
    *,
    parameters: t.Mapping[str, t.Any] = _NO_PARAMETERS,
    toolset: modest.Toolset = modest.toolset,
    options: ModesOptions = _DEFAULT_OPTIONS,
    actions: generic.Actions = generic.Actions.EDGE_BY_INDEX,
    observations: generic.Observations = generic.Observations.GLOBAL_ONLY,
) -> t.Mapping[str, fractions.Fraction]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        check_oracle_async(
            network,
            instance,
            oracle,
            parameters=parameters,
            toolset=toolset,
            options=options,
            actions=actions,
            observations=observations,
        )
    )


def check_nn(
    network: model.Network,
    instance: model.Instance,
    nn: torch.nn.Module,
    *,
    parameters: t.Mapping[str, t.Any] = _NO_PARAMETERS,
    toolset: modest.Toolset = modest.toolset,
    options: ModesOptions = _DEFAULT_OPTIONS,
    actions: generic.Actions = generic.Actions.EDGE_BY_INDEX,
    observations: generic.Observations = generic.Observations.GLOBAL_ONLY,
) -> t.Mapping[str, fractions.Fraction]:
    """Checks a PyTorch neural network."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        model_path = temp_path / "model.jani"
        model_path.write_text(jani.dump_model(network), encoding="utf-8")
        nn_path = temp_path / "nn.json"
        nn_path.write_text(dump_nn(nn))
        output_file = temp_path / "output.json"
        command: t.List[str] = [
            str(toolset.executable),
            "modes",
            str(model_path),
            "-O",
            str(output_file),
            "json",
            "-R",
            "NN",
            "-NN",
            str(nn_path),
            "-A",
            str(instance.automaton.name),
            "--threads",
            "1",
        ]
        for param_name, param_value in parameters.items():
            command.extend(["-E", f"{param_name}={param_value}"])
        options.apply(command)
        _apply_actions_observations(command, actions, observations)
        subprocess.check_call(command)
        return _load_result(output_file)
