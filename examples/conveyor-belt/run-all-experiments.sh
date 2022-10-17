#!/usr/bin/env bash

python -m conveyor_belt.experiments.scalability run output-scalability
python -m conveyor_belt.experiments.history_bound run output-history-bound
python -m conveyor_belt.experiments.latency_jitter run output-latency-jitter