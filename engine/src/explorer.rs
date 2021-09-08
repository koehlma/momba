use std::{sync::Arc, time::Instant};

use hashbrown::HashSet;

use momba_explore::State;

use crate::{states, time, CompiledExpression};

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

    fn compile_global_expression(
        &self,
        expr: &momba_explore::model::Expression,
    ) -> CompiledExpression;

    fn count_states_and_transitions(&self) -> (usize, usize);
}

impl<T: time::Time + Eq> DynExplorer for Explorer<T>
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

    fn compile_global_expression(
        &self,
        expr: &momba_explore::model::Expression,
    ) -> CompiledExpression {
        CompiledExpression {
            expr: self
                .explorer
                .compiled_network
                .compile_global_expression(expr),
        }
    }

    fn count_states_and_transitions(&self) -> (usize, usize) {
        let mut visited: HashSet<State<_>> = HashSet::new();
        let mut pending: Vec<_> = self.explorer.initial_states();

        let mut count_transitions = 0;

        let start = Instant::now();

        let mut processed = 0;

        while let Some(state) = pending.pop() {
            processed += 1;

            // if processed % 20000 == 0 {
            //     let duration = start.elapsed();
            //     println!(
            //         "States: {} ({:.2} [states/s], [waiting {}])",
            //         visited.len(),
            //         (processed as f64) / duration.as_secs_f64(),
            //         pending.len(),
            //     )
            // }

            if !visited.contains(&state) {
                let transitions = self.explorer.transitions(&state);

                for transition in transitions {
                    count_transitions += 1;
                    let destinations = self.explorer.destinations(&state, &transition);
                    for destination in destinations {
                        let successor = self.explorer.successor(&state, &transition, &destination);
                        pending.push(successor);
                    }
                }

                visited.insert(state);
            }
        }

        (visited.len(), count_transitions)
    }
}
