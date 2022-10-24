# -*- coding: utf-8 -*-
#
# Copyright (C) 2022, Saarland University
# Copyright (C) 2022, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import pathlib
import subprocess
import resource
import multiprocessing

import click


binary = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / ".."
    / "target"
    / "release"
    / "robust-diagnosis"
).resolve()

assert binary.exists()


def generate(
    model: pathlib.Path,
    diagnose_parameters: pathlib.Path,
    generate_parameters: pathlib.Path,
    observations_output: pathlib.Path,
    events_output: pathlib.Path,
    simulation_time: float,
) -> None:
    subprocess.check_output(
        [
            binary,
            "generate",
            model,
            diagnose_parameters,
            generate_parameters,
            observations_output,
            events_output,
            str(simulation_time),
        ]
    )


@d.dataclass(frozen=True)
class GenerateJob:
    model: pathlib.Path
    diagnose_parameters: pathlib.Path
    generate_parameters: pathlib.Path
    observations_output: pathlib.Path
    events_output: pathlib.Path
    simulation_time: float


def run_generate_job(job: GenerateJob) -> None:
    generate(
        job.model,
        job.diagnose_parameters,
        job.generate_parameters,
        job.observations_output,
        job.events_output,
        job.simulation_time,
    )


def diagnose(
    model: pathlib.Path,
    parameters: pathlib.Path,
    observations: pathlib.Path,
    output: pathlib.Path,
    history_bound: int = -1,
    memory_limit: t.Optional[int] = None,
) -> None:
    command = [binary, "diagnose", model, parameters, observations, output]
    if history_bound >= 0:
        command.extend(["--history-bound", history_bound])

    preexec_fn = None
    if memory_limit is not None:

        def _set_memory_limit():
            resource.setrlimit(resource.RLIMIT_DATA, memory_limit)

        preexec_fn = _set_memory_limit

    subprocess.check_output(tuple(map(str, command)), preexec_fn=preexec_fn)


@d.dataclass(frozen=True)
class DiagnoseJob:
    model: pathlib.Path
    parameters: pathlib.Path
    observations: pathlib.Path
    output: pathlib.Path
    history_bound: int
    memory_limit: t.Optional[int] = 4 * 1024 * 1024 * 1024  # Default is 4GB.


def run_diagnose_job(job: DiagnoseJob) -> None:
    diagnose(job.model, job.parameters, job.observations, job.output, job.history_bound)


PROCESSES = multiprocessing.cpu_count() - 1


class ProcessPool:
    _pool: t.Any

    def __init__(self, processes: t.Optional[int] = None) -> None:
        self._pool = multiprocessing.Pool(processes or PROCESSES)

    def run_generate_jobs(self, jobs: t.Sequence[GenerateJob]) -> None:
        with click.progressbar(
            self._pool.imap_unordered(run_generate_job, jobs),
            length=len(jobs),
            show_pos=True,
        ) as bar:
            for _ in bar:
                pass

    def run_diagnose_jobs(self, jobs: t.Sequence[DiagnoseJob]) -> None:
        with click.progressbar(
            self._pool.imap_unordered(run_diagnose_job, jobs),
            length=len(jobs),
            show_pos=True,
        ) as bar:
            for _ in bar:
                pass
