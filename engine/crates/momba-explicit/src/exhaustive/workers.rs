//! Worker infrastructure for exhaustive concurrent state space exploration.

use std::{sync::atomic, time::Duration};

/// A context in which work is done.
struct Ctx {
    /// The number of workers.
    num_workers: usize,
    /// The number of workers which are waiting for tasks.
    waiting: atomic::AtomicUsize,
}

impl Ctx {
    /// Creates a new context with the given number of workers.
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            waiting: atomic::AtomicUsize::new(0),
        }
    }

    /// Checks whether all work is done.
    pub fn is_all_done(&self) -> bool {
        self.waiting.load(atomic::Ordering::Acquire) == self.num_workers
    }

    /// Waits in case there are no more tasks.
    pub fn wait_with<F, T>(&self, closure: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.waiting.fetch_add(1, atomic::Ordering::AcqRel);
        let output = closure();
        if !self.is_all_done() {
            self.waiting.fetch_sub(1, atomic::Ordering::AcqRel);
        }
        output
    }
}

/// A worker context holding all relevant data.
pub struct WorkerCtx<'cx, T> {
    /// The unique id of the worker.
    worker_id: usize,
    /// The work context.
    ctx: &'cx Ctx,
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

    /// Returns the next task to execute.
    pub fn next_task(&self) -> Option<T> {
        self.local_queue.pop().or_else(|| self.steal_next_task())
    }

    /// Tries steal a task from other workers.
    #[cold]
    fn steal_next_task(&self) -> Option<T> {
        debug_assert!(
            self.local_queue.is_empty(),
            "Local queue should be empty at this point."
        );
        self.ctx.wait_with(|| {
            let mut spin_counter = 0;
            let mut back_off_duration = Duration::from_nanos(100);
            loop {
                for stealer in &self.stealers {
                    let stolen_task = stealer.steal().success();
                    if stolen_task.is_some() {
                        return stolen_task;
                    }
                }
                if self.ctx.is_all_done() {
                    return None;
                }
                spin_counter += 1;
                if spin_counter >= 16 {
                    spin_counter = 0;
                    std::thread::sleep(back_off_duration);
                    // The back-off duration increases exponentially up to `10ms`.
                    back_off_duration = Duration::from_nanos(
                        (back_off_duration.as_nanos() * back_off_duration.as_nanos())
                            .min(10_000_000) as u64,
                    );
                } else {
                    std::thread::yield_now();
                }
            }
        })
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
    let ctx = Ctx::new(num_workers);
    let local_queues = (0..num_workers)
        .map(|_| match strategy {
            WorkerQueueStrategy::Fifo => crossbeam_deque::Worker::new_fifo(),
            WorkerQueueStrategy::Lifo => crossbeam_deque::Worker::new_lifo(),
        })
        .collect::<Vec<_>>();
    for (task_idx, task) in tasks.into_iter().enumerate() {
        local_queues[task_idx % num_workers].push(task);
    }
    let stealers = local_queues
        .iter()
        .map(|local_queue| local_queue.stealer())
        .collect::<Vec<_>>();
    let ctxs = local_queues
        .into_iter()
        .enumerate()
        .map(|(worker_id, local_queue)| WorkerCtx {
            worker_id,
            ctx: &ctx,
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
    std::thread::scope(|scope| {
        for ctx in ctxs.into_iter() {
            let worker = factory(&ctx);
            scope.spawn(move || worker(ctx));
        }
    });
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
