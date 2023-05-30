#[allow(dead_code)]
use momba_explore::*;
use rand::rngs::StdRng;
use std::{cell::RefCell, sync::Arc};

use crate::simulate::Oracle;
/*
structs to have some oracle abstraction for custom functions
that takes the oracle function uses some lexicografic order of the labels
or something
*/

struct CustomOracle<T, G>
where
    T: time::Time,
{
    /// Explorer reference to the oracle.
    _explorer: Arc<Explorer<T>>,
    /// RNG for the oracle.
    _rng: RefCell<StdRng>,
    /// Function to use as the Oracle.
    /// The type of the function will be defined when the structure is created.
    _oracle_fn: G,
    //oracle_fn: Arc<dyn Fn(???) -> i64>,
    //I can have something like a buffer, so if wanted can have some memory.
}

impl<'a, T, G> CustomOracle<T, G>
where
    T: time::Time,
    G: Fn(&State<T>) -> i64,
{
    pub fn _new(
        //func: impl Fn(???) -> i64 + 'static,
        explorer: Arc<Explorer<T>>,
        rng: StdRng,
        func: G,
    ) -> Self {
        CustomOracle {
            //    oracle_fn: Arc::new(func),
            _oracle_fn: func,
            _explorer: explorer,
            _rng: RefCell::new(rng),
        }
    }
}

impl<T, G> Oracle<T> for CustomOracle<T, G>
where
    T: time::Time,
    // But this is too restrictive, it can be nice to have like dinamic parameters,
    // so if needed the function can take transitions, destinations, history, etc.
    G: Fn(&State<T>) -> i64,
{
    fn choose<'s, 't>(
        &self,
        _state: &State<T>,
        _transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        todo!();
        // let mut rng = self.rng.borrow_mut();
        // let action = (self.oracle_fn)(state);
    }
}
