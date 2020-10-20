# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib


Command = t.Sequence[t.Union[str, pathlib.Path]]


class ToolError(Exception):
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


class ToolTimeoutError(ToolError):
    pass
