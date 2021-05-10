use std::sync::{Arc, RwLock};

use downcast_rs::{impl_downcast, DowncastSync};

use crate::{
    states::State,
    time::{ConvertValuations, Time},
    PyState,
};

pub struct Destination<T: Time> {
    pub explorer: Arc<momba_explore::Explorer<T>>,
    pub state: Arc<momba_explore::State<T>>,
    pub transition: Arc<RwLock<momba_explore::Transition<'static, T>>>,
    pub destination: Arc<momba_explore::Destination<'static, T>>,
}

pub trait DynDestination: DowncastSync {
    fn probability(&self) -> f64;

    fn successor(&self) -> PyState;
}

impl_downcast!(sync DynDestination);

impl<T: Time> DynDestination for Destination<T>
where
    T::Valuations: ConvertValuations,
{
    fn probability(&self) -> f64 {
        self.destination.probability()
    }

    fn successor(&self) -> PyState {
        State {
            explorer: self.explorer.clone(),
            state: Arc::new(self.explorer.successor(
                &self.state,
                &self.transition.read().unwrap(),
                &self.destination,
            )),
        }
        .into()
    }
}
