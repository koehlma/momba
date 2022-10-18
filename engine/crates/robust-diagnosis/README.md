# Robust Real-Time Diagnosis Tool

This directory contains the reference implementation of the robust real-time diagnosis algorithm (under submission).

The file `src/robust/observer.rs` contains the data structures for representing observations and maintaining the set of arrived observations $Ω_t$ as a transitivity reduced DAG. The core procedure is `insort` which adds an observation to the DAG. It runs an `ObservationIndex` which is used to uniquely identify the respective observation in the system. The `Imprecisions` struct is used to represent the temporal imprecisions and has a method for computing the difference bound function.


The file `src/robust/mod.rs` contains the actual diagnosis algorithm and the exploration procedure.
