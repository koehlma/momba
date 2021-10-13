# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import pathlib
import typing as t

import asyncio
import struct
import tempfile

from .. import jani, model

from ..tools import modest

from .abstract import Oracle


HEADER = struct.Struct("!II")  # (num_features, num_actions)
DECISION = struct.Struct("!I")  # (action,)


MODEST = modest.toolset.executable


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
                # print("Received state:", state)
                # TODO: receive available actions vector
                decision = self.oracle(state, available)
                # print("Send decision:", decision)
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


async def _check(
    model_path: pathlib.Path,
    automaton_name: str,
    oracle: Oracle,
    *,
    parameters: t.Mapping[str, t.Any] = _NO_PARAMETERS,
) -> None:
    server = OracleServer(oracle)
    await server.start()
    address = f"localhost:{server.port}"
    print(f"Address: {address}")
    arguments: t.List[str] = [
        "modes",
        str(model_path),
        "-R",
        "oracle",
        "--socket",
        address,
        "-A",
        automaton_name,
        "--threads",
        "1",
    ]
    for param_name, param_value in parameters.items():
        arguments.extend(["-E", f"{param_name}={param_value}"])
    process = await asyncio.subprocess.create_subprocess_exec(MODEST, *arguments)
    await process.wait()


def check(
    network: model.Network,
    instance: model.Instance,
    oracle: Oracle,
    *,
    parameters: t.Mapping[str, t.Any] = _NO_PARAMETERS,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        model_path = temp_path / "model.jani"
        model_path.write_text(jani.dump_model(network), encoding="utf-8")
        loop = asyncio.get_event_loop()
        assert instance.automaton.name is not None
        loop.run_until_complete(
            _check(model_path, instance.automaton.name, oracle, parameters=parameters)
        )
