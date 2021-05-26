# -*- coding: utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d

import pathlib
import subprocess


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
) -> None:
    command = [binary, "diagnose", model, parameters, observations, output]
    if history_bound >= 0:
        command.extend(["--history-bound", history_bound])
    subprocess.check_output(tuple(map(str, command)))


@d.dataclass(frozen=True)
class DiagnoseJob:
    model: pathlib.Path
    parameters: pathlib.Path
    observations: pathlib.Path
    output: pathlib.Path
    history_bound: int


def run_diagnose_job(job: DiagnoseJob) -> None:
    diagnose(job.model, job.parameters, job.observations, job.output, job.history_bound)
