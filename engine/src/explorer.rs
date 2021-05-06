use std::sync::Arc;

use crate::{states, time};

#[derive(Clone)]
pub struct Explorer<T: time::Time> {
    explorer: Arc<momba_explore::Explorer<T>>,
}

impl<T: time::Time> From<momba_explore::Explorer<T>> for Explorer<T> {
    fn from(explorer: momba_explore::Explorer<T>) -> Self {
        Self {
            explorer: Arc::new(explorer),
        }
    }
}

/// Trait to dynamically abstract over [Explorer][momba_explore::Explorer].
pub trait DynExplorer: Send + Sync {
    fn initial_states(&self) -> Vec<crate::PyState>;
}

impl<T: time::Time> DynExplorer for Explorer<T>
where
    T::Valuations: time::ConvertValuations,
{
    fn initial_states(&self) -> Vec<crate::PyState> {
        self.explorer
            .initial_states()
            .into_iter()
            .map(|state| {
                states::State {
                    explorer: self.explorer.clone(),
                    state: Arc::new(state),
                }
                .into()
            })
            .collect()
    }
}
