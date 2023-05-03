//! Worker infrastructure for exhaustive concurrent state space exploration.

/// A worker context holding all relevant queues.
pub struct WorkerCtx<'cx, T> {
    /// The unique id of the worker.
    worker_id: usize,
    /// The global injector queue.
    global_queue: &'cx crossbeam_deque::Injector<T>,
    /// The local queue.
    local_queue: crossbeam_deque::Worker<T>,
    /// The work stealers.
    stealers: Vec<crossbeam_deque::Stealer<T>>,
}

impl<'cx, T> WorkerCtx<'cx, T> {
    /// Returns the unique worker id.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Pushes a task into the context.
    pub fn push_task(&self, task: T) {
        self.local_queue.push(task);
    }

    /// Pops a task from the context.
    pub fn next_task(&self) -> Option<T> {
        self.local_queue.pop().or_else(|| self.pop_task_slow())
    }

    /// Pops a task from the global queue or steals it from another worker.
    #[cold]
    fn pop_task_slow(&self) -> Option<T> {
        debug_assert_eq!(
            self.local_queue.len(),
            0,
            "Local queue should be empty at this point."
        );
        std::iter::repeat_with(|| {
            self.global_queue
                .steal_batch_and_pop(&self.local_queue)
                .or_else(|| {
                    self.stealers
                        .iter()
                        .map(|stealer| stealer.steal())
                        .collect()
                })
        })
        .find(|stolen| !stolen.is_retry())
        .and_then(|stolen| stolen.success())
    }
}

/// Queuing strategy for work local queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerQueueStrategy {
    /// First-in-first-out (FIFO) strategy.
    Fifo,
    /// Last-in-first-out (LIFO) strategy.
    Lifo,
}

/// Spawns the given number of workers with the given tasks and waits until all tasks are completed.
pub fn spawn_and_run_workers<F, W, T, I>(
    num_workers: usize,
    mut factory: F,
    strategy: WorkerQueueStrategy,
    tasks: I,
) where
    T: Send,
    F: FnMut(&WorkerCtx<'_, T>) -> W,
    W: Send + FnOnce(WorkerCtx<'_, T>),
    I: IntoIterator<Item = T>,
{
    let global_queue = crossbeam_deque::Injector::new();
    let local_queues = (0..num_workers)
        .map(|_| match strategy {
            WorkerQueueStrategy::Fifo => crossbeam_deque::Worker::new_fifo(),
            WorkerQueueStrategy::Lifo => crossbeam_deque::Worker::new_lifo(),
        })
        .collect::<Vec<_>>();
    let stealers = local_queues
        .iter()
        .map(|local_queue| local_queue.stealer())
        .collect::<Vec<_>>();
    let ctxs = local_queues
        .into_iter()
        .enumerate()
        .map(|(worker_id, local_queue)| WorkerCtx {
            worker_id,
            global_queue: &global_queue,
            local_queue,
            stealers: stealers
                .iter()
                .enumerate()
                .filter_map(|(stealer_worker_id, stealer)| {
                    if stealer_worker_id == worker_id {
                        // Workers should not steal tasks from themselves.
                        None
                    } else {
                        Some(stealer.clone())
                    }
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    for task in tasks {
        global_queue.push(task);
    }
    crossbeam::scope(|scope| {
        for ctx in ctxs.into_iter() {
            let worker = factory(&ctx);
            scope.spawn(move |_| worker(ctx));
        }
    })
    .expect("Panic while executing workers.")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic;

    use super::*;

    #[test]
    pub fn test_workers() {
        let sum = atomic::AtomicI32::new(0);
        spawn_and_run_workers(
            2,
            |_| {
                |ctx| {
                    while let Some(mut task) = ctx.next_task() {
                        sum.fetch_add(task, atomic::Ordering::Relaxed);
                        task -= 1;
                        if task > 0 {
                            ctx.push_task(task);
                        }
                    }
                }
            },
            WorkerQueueStrategy::Lifo,
            [3, 6, 4],
        );
        assert_eq!(
            sum.load(atomic::Ordering::Acquire),
            3 * (1 + 2 + 3) + 2 * 4 + 5 + 6
        );
    }
}
