# `clock-zones`

[![crate](https://img.shields.io/crates/v/clock-zones.svg)](https://crates.io/crates/clock-zones)
[![documentation](https://docs.rs/clock-zones/badge.svg)](https://docs.rs/clock-zones)

A library for handling *clock zones* as they appear in the context of [*Timed Automata*](https://link.springer.com/chapter/10.1007/BFb0031987).
Let $X$ be a finite set of *clock variables*.
A *zone* over $X$ is a convex $(|X| + 1)$-dimensional polytope described by equations of the form $x - y \prec c$ where $x, y \in X \cup \{\mathbf{0}\}$, $c \in \mathbb{Q}$, and ${\prec} \in \{<, \leq\}$.
This library implements [*Difference Bound Matrices*](https://link.springer.com/chapter/10.1007/978-3-540-27755-2_3) for efficient handling of zones.
