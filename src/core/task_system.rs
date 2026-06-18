use log::debug;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, Receiver, RecvError, SendError, SyncSender, TryRecvError},
};
use std::thread::{self, JoinHandle};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TaskSystemError {
    #[error("Task system is disconnected")]
    Disconnected,
}

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

            workers.push(thread::spawn(move || {
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
            }));
        }

        drop(result_tx);

        Self {
            task_tx: Some(task_tx),
            result_rx,
            workers,
        }
    }

    pub(crate) fn submit(&self, task: TTask) -> Result<(), TaskSystemError> {
        debug!("Task submitted to queue");
        self.task_tx
            .as_ref()
            .ok_or(TaskSystemError::Disconnected)?
            .send(task)
            .map_err(|_: SendError<TTask>| TaskSystemError::Disconnected)
    }

    pub(crate) fn try_recv_result(&self) -> Result<Option<TResult>, TaskSystemError> {
        self.result_rx.try_recv().map_or_else(
            |err| match err {
                TryRecvError::Empty => Ok(None),
                TryRecvError::Disconnected => Err(TaskSystemError::Disconnected),
            },
            |result| Ok(Some(result)),
        )
    }

    pub(crate) fn recv_result(&self) -> Result<TResult, TaskSystemError> {
        self.result_rx
            .recv()
            .map_err(|_: RecvError| TaskSystemError::Disconnected)
    }
}

impl<TTask, TResult> Drop for TaskSystem<TTask, TResult>
where
    TTask: Send + 'static,
    TResult: Send + 'static,
{
    fn drop(&mut self) {
        self.task_tx.take();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
