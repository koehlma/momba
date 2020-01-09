# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

from distutils.core import setup


README: pathlib.Path = pathlib.Path(__file__).parent / "README.rst"


setup(
    name="momba",
    version="0.1.3dev",
    description="A Python library for quantitative models.",
    long_description=README.read_text(encoding="utf-8"),
    author="Maximilian Köhl",
    author_email="mkoehl@cs.uni-saarland.de",
    url="https://depend.cs.uni-saarland.de/~koehl/momba/",
    tests_require=["pytest", "pytest-cov"],
    packages=["momba"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
