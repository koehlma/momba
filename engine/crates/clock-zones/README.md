# clock-zones

[![crate](https://img.shields.io/crates/v/clock-zones.svg)](https://crates.io/crates/clock-zones)
[![documentation](https://docs.rs/clock-zones/badge.svg)](https://docs.rs/clock-zones)

A library for handling *[clock zones]* as they appear in the context of
*[timed automata]*.
[Timed automata] have been pioneered by [Rajeev Alur] and [David Dill] in
1994 to model real-time systems [[1]].
Timed automata extend finite automata with real-valued *clocks*.
This crate provides an implementation of the *[difference bound matrix]*
(DBM) data structure to efficiently represent [clock zones].
The implementation is mostly based on [[2]].


[clock zones]: https://en.wikipedia.org/wiki/Difference_bound_matrix#Zone
[timed automata]: https://en.wikipedia.org/wiki/Timed_automaton
[difference bound matrix]: https://en.wikipedia.org/wiki/Difference_bound_matrix

[Rajeev Alur]: https://www.cis.upenn.edu/~alur/
[David Dill]: https://profiles.stanford.edu/david-dill

[1]: https://www.cis.upenn.edu/~alur/TCS94.pdf
[2]: https://doi.org/10.1007/978-3-540-27755-2_3


