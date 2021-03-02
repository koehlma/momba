#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import shutil
import subprocess
import sys

import tomlkit


PACKAGE_DIR = pathlib.Path(__file__).parent


pyproject_file = PACKAGE_DIR / "pyproject.toml"

project_metadata = tomlkit.parse(pyproject_file.read_text())
engine_metadata = tomlkit.parse((PACKAGE_DIR / "engine" / "Cargo.toml").read_text())

momba_version = project_metadata["tool"]["poetry"]["version"]
engine_version = engine_metadata["package"]["version"]


assert (
    momba_version == engine_version
), f"Version of `momba` ({momba_version}) and `momba_engine` ({engine_version}) do not match."


# save original `pyproject.toml` for later
shutil.copy(pyproject_file, PACKAGE_DIR / "pyproject.toml.orig")

# update project metadata to depend on precise version of `momba_engine`
project_metadata["tool"]["poetry"]["dependencies"]["momba_engine"][
    "version"
] = engine_version
pyproject_file.write_text(tomlkit.dumps(project_metadata))

# call poetry to build the package
subprocess.check_call([sys.executable, "-m", "poetry", "build"], cwd=PACKAGE_DIR)

# restore original `pyproject.toml`
shutil.move(str(PACKAGE_DIR / "pyproject.toml.orig"), pyproject_file)
