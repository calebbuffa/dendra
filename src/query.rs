use crate::distance::MetricFn;

pub struct Query {
    pub vector: Vec<f32>,
    pub k: usize,
    pub metric: MetricFn,
    pub threshold: Option<f32>,
}

impl Query {
    pub fn new(vector: Vec<f32>, k: usize, metric: MetricFn, threshold: Option<f32>) -> Self {
        Self {
            vector,
            k,
            metric,
            threshold,
        }
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = Some(threshold);
    }

    pub fn clear_threshold(&mut self) {
        self.threshold = None;
    }
}
