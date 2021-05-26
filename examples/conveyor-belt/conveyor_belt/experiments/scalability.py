# -*- coding: utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import itertools
import json
import multiprocessing
import pathlib
import re
import statistics


import click

from momba import engine, model
from momba.engine import translator

from ..builder import Scenario, Sensor, build_model

from . import runner


DIAGNOSIS_PARAMETERS = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / "parameters"
    / "diagnosis"
    / "realistic_latency_and_jitter.toml"
).resolve()

assert DIAGNOSIS_PARAMETERS.exists()


GENERATE_PARAMETERS = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / "parameters"
    / "generate"
    / "inject_friction_with_rate_0_000005.toml"
).resolve()

assert GENERATE_PARAMETERS.exists()


def _generate_scenarios(*, fault_sporadic: bool = False) -> t.Iterator[Scenario]:
    for length in (4, 6, 10):
        yield Scenario(
            length=length,
            sensors=tuple(
                Sensor(
                    position=position,
                    tick_lower_bound=100,
                    tick_upper_bound=100,
                    fault_sporadic=fault_sporadic,
                )
                for position in (length // 2, length // 2 + 1)
            ),
            running_tick_lower_bound=500,
            running_tick_upper_bound=550,
            fault_tick_lower_bound=600,
            fault_tick_upper_bound=750,
            fault_friction=True,
        )


@d.dataclass(frozen=True)
class ModelFile:
    identifier: str
    scenario: Scenario
    fault_sporadic: bool
    network: model.Network
    zones: int
    transitions: int
    path: pathlib.Path
    generated_path: pathlib.Path


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("output_dir", type=pathlib.Path, metavar="output")
@click.option("--processes", type=int, default=max(multiprocessing.cpu_count() - 1, 1))
def run(output_dir: pathlib.Path, processes: int) -> None:
    pool = multiprocessing.Pool(processes)

    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Building models...")
    models = []
    for fault_sporadic in (True, False):
        for scenario in _generate_scenarios(fault_sporadic=fault_sporadic):
            identifier = f"length_{scenario.length}_sporadic_{fault_sporadic}"
            network = build_model(scenario)
            explorer = engine.Explorer(network, engine.time.ZoneF64)
            translation = translator.translate_network(network)
            path = models_dir / f"{identifier}.json"
            path.write_text(translation.json_network, encoding="utf-8")
            generated_path = generated_dir / identifier
            generated_path.mkdir()
            models.append(
                ModelFile(
                    identifier,
                    scenario,
                    fault_sporadic,
                    network,
                    explorer.count_states(),
                    explorer.count_transitions(),
                    path,
                    generated_path,
                )
            )

    print("Generating observations...")

    generate_jobs = tuple(
        runner.GenerateJob(
            model.path,
            DIAGNOSIS_PARAMETERS,
            GENERATE_PARAMETERS,
            model.generated_path / f"observations_{index}.json",
            model.generated_path / f"events_{index}.json",
            60_000,  # 60s
        )
        for model, index in itertools.product(models, range(100))
    )

    with click.progressbar(
        pool.imap_unordered(runner.run_generate_job, generate_jobs),
        length=len(generate_jobs),
        show_pos=True,
    ) as bar:
        for _ in bar:
            pass

    print("Analyzing fault ratio...")
    for model_file in models:
        total = 0
        with_fault = 0
        for events_path in model_file.generated_path.glob("events_*.json"):
            total += 1
            for event in json.loads(events_path.read_text(encoding="utf-8")):
                if event["label"] == "fault_friction":
                    with_fault += 1
        print(f"{model_file.identifier}: {total} {with_fault} {with_fault / total}")

    print("Diagnosing faults...")
    diagnose_jobs = tuple(
        runner.DiagnoseJob(
            model.path,
            DIAGNOSIS_PARAMETERS,
            model.generated_path / f"observations_{index}.json",
            model.generated_path / f"diagnosis_{index}.json",
            history_bound=2,
        )
        for model, index in itertools.product(models, range(100))
    )

    with click.progressbar(
        pool.imap_unordered(runner.run_diagnose_job, diagnose_jobs),
        length=len(diagnose_jobs),
        show_pos=True,
    ) as bar:
        for _ in bar:
            pass


@main.command()
@click.argument("output_dir", type=pathlib.Path, metavar="output")
def analyze(output_dir: pathlib.Path) -> None:
    generated_dir = output_dir / "generated"

    for model_dir in generated_dir.iterdir():
        match = re.match(r"length_(\d+)_sporadic_(True|False)", model_dir.name)
        assert match is not None
        length = int(match[1])
        fault_sporadic = match[2] == "True"
        obs_per_sec = []
        for result_path in model_dir.glob("diagnosis_*.json"):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            obs_per_sec.append(len(data) / data[-1]["duration"])
        print(
            length,
            fault_sporadic,
            statistics.mean(obs_per_sec),
        )


if __name__ == "__main__":
    main()
