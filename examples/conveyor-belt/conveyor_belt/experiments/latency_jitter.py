# -*- coding: utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import typing as t

import json
import pathlib
import re

import click

from . import boxplot, runner


MODEL_PATH = (
    pathlib.Path(__file__).parent / ".." / ".." / "models" / "short_two_sensors.json"
).resolve()

assert MODEL_PATH.exists()


DIAGNOSIS_PARAMETERS = list(
    (pathlib.Path(__file__).parent / ".." / ".." / "parameters" / "diagnosis")
    .resolve()
    .iterdir()
)

assert all(parameters_path.exists() for parameters_path in DIAGNOSIS_PARAMETERS)


GENERATE_PARAMETERS = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / "parameters"
    / "generate"
    / "inject_friction_with_rate_0_000005.toml"
).resolve()

assert GENERATE_PARAMETERS.exists()


@click.group()
def main() -> None:
    """
    An experiment to study the effect of the history bound.
    """


@main.command()
@click.argument("data_dir", type=pathlib.Path)
@click.option("--time", type=float, default=120_000.0)
@click.option("--history-bound", type=int, default=2)
@click.option("--traces", type=int, default=1000)
@click.option("--processes", type=int, default=None)
def run(
    data_dir: pathlib.Path,
    time: float,
    history_bound: int,
    traces: int,
    processes: t.Optional[int],
) -> None:
    """
    Runs the experiment with the provided parameters.
    """
    pool = runner.ProcessPool(processes)

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True)

    traces_dir = data_dir / "traces"
    traces_dir.mkdir()

    results_dir = data_dir / "results"
    results_dir.mkdir()

    for parameters_path in DIAGNOSIS_PARAMETERS:
        (traces_dir / parameters_path.stem).mkdir()
        (results_dir / parameters_path.stem).mkdir()

    print("Generating observations...")
    pool.run_generate_jobs(
        [
            runner.GenerateJob(
                MODEL_PATH,
                parameters_path,
                GENERATE_PARAMETERS,
                traces_dir / parameters_path.stem / f"{trace}_observations.json",
                traces_dir / parameters_path.stem / f"{trace}_events.json",
                time,
            )
            for parameters_path in DIAGNOSIS_PARAMETERS
            for trace in range(traces)
        ]
    )

    print("Running diagnosis...")
    diagnose_jobs: t.List[runner.DiagnoseJob] = [
        runner.DiagnoseJob(
            MODEL_PATH,
            parameters_path,
            traces_dir / parameters_path.stem / f"{trace}_observations.json",
            results_dir / parameters_path.stem / f"{trace}_result.json",
            history_bound,
        )
        for parameters_path in DIAGNOSIS_PARAMETERS
        for trace in range(traces)
    ]

    pool.run_diagnose_jobs(diagnose_jobs)


@main.command()
@click.argument("data_dir", type=pathlib.Path)
@click.option("--exclude-fault-runs", type=bool, default=False, is_flag=True)
def analyze(data_dir: pathlib.Path, exclude_fault_runs: bool) -> None:
    """
    Analyzes the gathered data.
    """
    data_dir = data_dir.resolve()

    boxplots: t.List[boxplot.Box] = []

    for out_dir in sorted((data_dir / "results").iterdir(), key=lambda path: path.name):
        values: t.List[float] = []
        for result_file in out_dir.glob("*_result.json"):
            match = re.search(r"(\d+)_result", result_file.name)
            assert match is not None
            trace = int(match[1])
            events = json.loads(
                (data_dir / "traces" / out_dir.name / f"{trace}_events.json").read_text(
                    encoding="utf-8"
                )
            )
            contains_fault = any(event["label"] == "fault_friction" for event in events)
            if contains_fault and exclude_fault_runs:
                continue
            records = json.loads(result_file.read_text(encoding="utf-8"))
            values.append(records[-1]["duration"] / (len(records) - 1))
        boxplots.append(
            boxplot.Box.from_data(
                out_dir.name.replace("_", " "),
                values=values[: 4 * (len(values) // 4)],
                whisker_type=boxplot.WhiskerType.EXTREM_VALUES,
            )
        )

    pathlib.Path("latency_jitter.tex").write_text(
        boxplot.Plot(boxes=boxplots).latex_source
    )


if __name__ == "__main__":
    main()
