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


def bound_label(bound: int) -> str:
    if bound < 0:
        return "$\\infty$"
    else:
        return f"${bound}$"


@click.group()
def main() -> None:
    """
    An experiment to study the effect of the history bound.
    """


@main.command()
@click.argument("data_dir", type=pathlib.Path)
@click.option("--time", type=float, default=120_000.0)
@click.option(
    "--bounds",
    type=int,
    multiple=True,
    default=[0, 2, 3, 4, 5, 10, -1],
)
@click.option("--traces", type=int, default=1000)
@click.option("--processes", type=int, default=None)
def run(
    data_dir: pathlib.Path,
    time: float,
    bounds: t.List[int],
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

    print("Generating observations...")
    pool.run_generate_jobs(
        [
            runner.GenerateJob(
                MODEL_PATH,
                DIAGNOSIS_PARAMETERS,
                GENERATE_PARAMETERS,
                traces_dir / f"{trace}_observations.json",
                traces_dir / f"{trace}_events.json",
                time,
            )
            for trace in range(traces)
        ]
    )

    results_dir = data_dir / "results"
    results_dir.mkdir()

    print("Running diagnosis...")
    diagnose_jobs: t.List[runner.DiagnoseJob] = []
    for bound in bounds:
        out_dir = results_dir / f"{bound}"
        out_dir.mkdir()
        diagnose_jobs.extend(
            runner.DiagnoseJob(
                MODEL_PATH,
                DIAGNOSIS_PARAMETERS,
                traces_dir / f"{trace}_observations.json",
                out_dir / f"{trace}_result.json",
                bound,
            )
            for trace in range(traces)
        )

    pool.run_diagnose_jobs(diagnose_jobs)


@main.command()
@click.argument("data_dir", type=pathlib.Path)
@click.option("--exclude-fault-runs", type=bool, default=False, is_flag=True)
def analyze(data_dir: pathlib.Path, exclude_fault_runs: bool) -> None:
    """
    Analyzes the gathered data.
    """
    data_dir = data_dir.resolve()

    results_dir = data_dir / "results"

    bounds = [int(bound_dir.name) for bound_dir in results_dir.iterdir()]
    bounds.sort()

    second_quarter: t.Dict[int, t.List[float]] = {bound: [] for bound in bounds}
    last_quarter: t.Dict[int, t.List[float]] = {bound: [] for bound in bounds}

    for bound in bounds:
        out_dir = results_dir / f"{bound}"
        for result_file in out_dir.glob("*_result.json"):
            match = re.search(r"(\d+)_result", result_file.name)
            assert match is not None
            trace = int(match[1])
            events = json.loads(
                (data_dir / "traces" / f"{trace}_events.json").read_text(
                    encoding="utf-8"
                )
            )
            contains_fault = any(event["label"] == "fault_friction" for event in events)
            if contains_fault and exclude_fault_runs:
                continue
            records = json.loads(result_file.read_text(encoding="utf-8"))
            mid_offset = len(records) // 2
            first_offset = mid_offset - len(records) // 4
            second_offset = mid_offset + len(records) // 4
            second_quarter[bound].append(
                (mid_offset - first_offset)
                / (records[mid_offset]["duration"] - records[first_offset]["duration"])
            )
            last_quarter[bound].append(
                (len(records) - second_offset)
                / (records[-1]["duration"] - records[second_offset]["duration"])
            )

    for name, values in (
        ("second_quarter", second_quarter),
        ("last_quarter", last_quarter),
    ):
        boxplots = [
            boxplot.Box.from_data(
                bound_label(bound),
                values=values[bound][: 4 * (len(values[bound]) // 4)],
                whisker_type=boxplot.WhiskerType.EXTREM_VALUES,
            )
            for bound in bounds
        ]

        tikz_picture = f"""
        \\begin{{tikzpicture}}[font=\\small]
            \\begin{{axis}}
                [   
                    xtick={{{", ".join(map(str, range(1, len(bounds) + 1)))}}},
                    xticklabels={{{", ".join(map(bound_label, bounds))}}},
                    boxplot/draw direction=y,
                    width=5cm,
                    height=4cm,
                    title={{{name.replace("_", " ")}}},
                    ylabel={{Delay [ms]}},
                    xlabel={{History Bound}},
                    every axis plot post/.append style={{
                        koehlma-blue,
                        solid,
                        mark=x,
                    }},
                ]
                {"".join(boxplot.latex_source for boxplot in boxplots)}
            \\end{{axis}}
        \\end{{tikzpicture}}
        """

        pathlib.Path(f"history_bounds_{name}.tex").write_text(tikz_picture)


if __name__ == "__main__":
    main()
