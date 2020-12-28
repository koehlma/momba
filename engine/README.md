# Momba Engine

[![PyPi Package](https://img.shields.io/pypi/v/momba_engine.svg?label=latest%20version)](https://pypi.python.org/pypi/momba_engine)

*Momba Engine* is a Python package partially written in Rust using [PyO3](https://pyo3.rs/).
While Python is a great language for many purposes, it certainly lacks the speed necessary for computing with large and complex models.
Hence, Momba Engine implements carefully chosen functionality, such as state space exploration, in Rust combining the convenience of Python with the raw power of Rust.
Note that we do not provide any stability guarantees for the API of Momba Engine.
The functionality is exposed as part of [Momba](https://https://github.com/koehlma/momba/)'s public API.
