#!/usr/bin/env bash

poetry run python -m conveyor_belt.experiments.scalability run output-scalability
poetry run python -m conveyor_belt.experiments.history_bound run output-history-bound
poetry run python -m conveyor_belt.experiments.latency_jitter run output-latency-jitter