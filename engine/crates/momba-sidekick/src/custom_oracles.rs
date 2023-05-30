use std::{cell::RefCell, sync::Arc};

use hashbrown::HashMap;
use momba_explore::*;
use rand::{rngs::StdRng, seq::IteratorRandom};
use serde::{Deserialize, Serialize};
use tch::{
    kind::*,
    nn,
    nn::{Linear, Module, Sequential},
    Device, Tensor,
};

use crate::simulate::Oracle;

/*
structs to have some oracle abstraction for custom functions
that takes the oracle function uses some lexicografic order of the labels
or something

*/
