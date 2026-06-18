use serde::{Deserialize, Serialize};

/// A Normal-Inverse-Gamma (NIG) prior for Bayesian inference of a Gaussian distribution with unknown mean and variance.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct NigPrior {
    pub(crate) mu0: f64,
    pub(crate) kappa0: f64,
    pub(crate) alpha0: f64,
    pub(crate) beta0: f64,
}

impl Default for NigPrior {
    fn default() -> Self {
        Self {
            mu0: 0.0,
            kappa0: 0.01,
            alpha0: 0.01,
            beta0: 0.01,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct NigStats {
    pub(crate) n: f64,
    pub(crate) mean: f64,
    pub(crate) m2: f64,
}

impl NigStats {
    pub(crate) fn new() -> Self {
        Self {
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub(crate) fn update(&mut self, x: f32) {
        let value = x as f64;
        self.n += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.n;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub(crate) fn predictive_log_likelihood(&self, x: f32, prior: NigPrior) -> f64 {
        let kappa_n = prior.kappa0 + self.n;
        let mu_n = if kappa_n > 0.0 {
            (prior.kappa0 * prior.mu0 + self.n * self.mean) / kappa_n
        } else {
            prior.mu0
        };
        let alpha_n = prior.alpha0 + 0.5 * self.n;
        let beta_n = prior.beta0
            + 0.5 * self.m2
            + if kappa_n > 0.0 {
                (prior.kappa0 * self.n * (self.mean - prior.mu0).powi(2)) / (2.0 * kappa_n)
            } else {
                0.0
            };

        let nu = (2.0 * alpha_n).max(1e-8);
        let sigma2 = (beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n.max(1e-8))).max(1e-12);
        student_t_log_pdf(x as f64, nu, mu_n, sigma2)
    }

    pub(crate) fn combined(a: &Self, b: &Self) -> Self {
        if a.n <= 0.0 {
            return *b;
        }
        if b.n <= 0.0 {
            return *a;
        }

        let n = a.n + b.n;
        let delta = b.mean - a.mean;
        let mean = (a.mean * a.n + b.mean * b.n) / n;
        let m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n);

        Self { n, mean, m2 }
    }
}

impl Default for NigStats {
    fn default() -> Self {
        Self::new()
    }
}

fn student_t_log_pdf(x: f64, nu: f64, mu: f64, sigma2: f64) -> f64 {
    let z = (x - mu) * (x - mu) / (nu * sigma2);
    let lgamma_half_nu_plus_half = statrs::function::gamma::ln_gamma((nu + 1.0) * 0.5);
    let lgamma_half_nu = statrs::function::gamma::ln_gamma(nu * 0.5);
    lgamma_half_nu_plus_half
        - lgamma_half_nu
        - 0.5 * (nu.ln() + std::f64::consts::PI.ln() + sigma2.ln())
        - ((nu + 1.0) * 0.5) * (1.0 + z).ln()
}
