use crate::math::MetricFn;

pub struct Query {
    vector: Vec<f32>,
    k: usize,
    metric: Option<MetricFn>,
    threshold: Option<f32>,
    delta: f32,
}

impl Query {
    pub fn new(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector,
            k,
            metric: None,
            threshold: None,
            delta: 0.05,
        }
    }

    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn metric(&self) -> Option<MetricFn> {
        self.metric
    }

    pub fn threshold(&self) -> Option<f32> {
        self.threshold
    }

    pub fn delta(&self) -> f32 {
        self.delta
    }

    pub fn with_metric(mut self, metric: MetricFn) -> Self {
        self.metric = Some(metric);
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn without_threshold(mut self) -> Self {
        self.threshold = None;
        self
    }

    pub fn with_delta(mut self, delta: f32) -> Self {
        self.delta = delta;
        self
    }
}
