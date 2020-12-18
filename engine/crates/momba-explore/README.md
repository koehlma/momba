# `momba-vm`

[![crate](https://img.shields.io/crates/v/momba-vm.svg)](https://crates.io/crates/momba-vm)
[![documentation](https://docs.rs/momba-vm/badge.svg)](https://docs.rs/momba-vm)

A VM for *Momba's compiled model representation* (MombaCR).
[Momba](https://github.com/koehlma/momba) is a Python framework for dealing with quantitative models centered around the [JANI-model](https://jani-spec.org/) interchange format.
For efficient state space exploration, Momba compiles JANI models into MombaCR which is then executed within a MombaCR VM.
This library implements such a MombaCR VM.
