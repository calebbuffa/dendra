use log::debug;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, Receiver, RecvError, SendError, SyncSender, TryRecvError, TrySendError},
};
use std::thread::{self, JoinHandle};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TaskSystemError {
    #[error("Task queue is full")]
    QueueFull,
    #[error("Task system is disconnected")]
    Disconnected,
}

/// Generic bounded background task runner.
/// - `TTask`: task payload sent by producers
/// - `TResult`: result payload produced by workers
pub(crate) struct TaskSystem<TTask, TResult>
where
    TTask: Send + 'static,
    TResult: Send + 'static,
{
    task_tx: Option<SyncSender<TTask>>,
    result_rx: Receiver<TResult>,
    workers: Vec<JoinHandle<()>>,
}

impl<TTask, TResult> TaskSystem<TTask, TResult>
where
    TTask: Send + 'static,
    TResult: Send + 'static,
{
    /// Create a bounded task system with `worker_count` worker threads.
    /// `handler` is executed for each task and its return value is sent to the result channel.
    pub(crate) fn new<F>(queue_capacity: usize, worker_count: usize, handler: F) -> Self
    where
        F: Fn(TTask) -> TResult + Send + Sync + 'static,
    {
        let (task_tx, task_rx) = mpsc::sync_channel::<TTask>(queue_capacity);
        let (result_tx, result_rx) = mpsc::channel::<TResult>();

        let shared_rx = Arc::new(Mutex::new(task_rx));
        let handler = Arc::new(handler);

        let mut workers = Vec::with_capacity(worker_count.max(1));
        for _ in 0..worker_count.max(1) {
            let rx = Arc::clone(&shared_rx);
            let tx = result_tx.clone();
            let handler = Arc::clone(&handler);

            let handle = thread::spawn(move || {
                loop {
                    let task = match rx.lock() {
                        Ok(guard) => guard.recv(),
                        Err(_) => break,
                    };

                    let task = match task {
                        Ok(task) => task,
                        Err(_) => break,
                    };

                    let result = handler(task);
                    if tx.send(result).is_err() {
                        break;
                    }
                }
            });

            workers.push(handle);
        }

        drop(result_tx);

        Self {
            task_tx: Some(task_tx),
            result_rx,
            workers,
        }
    }

    /// Blocking submit. Applies backpressure when queue is full.
    pub(crate) fn submit(&self, task: TTask) -> Result<(), TaskSystemError> {
        debug!("Task submitted to queue");
        self.task_tx
            .as_ref()
            .ok_or(TaskSystemError::Disconnected)?
            .send(task)
            .map_err(|_: SendError<TTask>| TaskSystemError::Disconnected)
    }

    /// Non-blocking submit.
    pub(crate) fn try_submit(&self, task: TTask) -> Result<(), TaskSystemError> {
        self.task_tx
            .as_ref()
            .ok_or(TaskSystemError::Disconnected)?
            .try_send(task)
            .map_err(|e| match e {
                TrySendError::Full(_) => TaskSystemError::QueueFull,
                TrySendError::Disconnected(_) => TaskSystemError::Disconnected,
            })
    }

    /// Blocking receive for next completed result.
    pub(crate) fn recv_result(&self) -> Result<TResult, TaskSystemError> {
        self.result_rx
            .recv()
            .map_err(|_: RecvError| TaskSystemError::Disconnected)
    }

    /// Non-blocking receive for next completed result.
    pub(crate) fn try_recv_result(&self) -> Result<Option<TResult>, TaskSystemError> {
        self.result_rx.try_recv().map_or_else(
            |err| match err {
                TryRecvError::Empty => Ok(None),
                TryRecvError::Disconnected => Err(TaskSystemError::Disconnected),
            },
            |result| {
                debug!("Task result received from worker");
                Ok(Some(result))
            },
        )
    }

    /// Cloneable sender for submitting tasks from other owners/threads.
    pub(crate) fn task_sender(&self) -> SyncSender<TTask> {
        self.task_tx
            .as_ref()
            .expect("task sender requested after shutdown")
            .clone()
    }

    /// Number of worker threads.
    pub(crate) fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl<TTask, TResult> Drop for TaskSystem<TTask, TResult>
where
    TTask: Send + 'static,
    TResult: Send + 'static,
{
    fn drop(&mut self) {
        // Close task channel first so workers unblock from recv() and can exit.
        self.task_tx.take();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
