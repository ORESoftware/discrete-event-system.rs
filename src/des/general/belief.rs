//! Port of `src/des/general/belief.ts` — module `des::general::belief`.
//!
//! `DiscreteBelief`: a probability distribution over a finite set of hidden
//! states with Bayesian update. This is the workhorse of every POMDP in the
//! framework:
//!
//!   * The agent maintains `b(s)` — a vector of weights, one per possible
//!     hidden state.
//!   * After observing `o`, `b'(s) ∝ P(o | s) · b(s)`; we renormalise.
//!   * After taking action `a` (with stochastic transitions `T(s' | s, a)`),
//!     `b'(s') = Σ_s T(s' | s, a) · b(s)`.
//!
//! Conversion notes from the TS source:
//!   * `update`/`propagate`/`expectation` take closures (`likelihood` /
//!     `transition` / `f`); mapped to generic `F: Fn(&S, usize) -> _` params.
//!   * `sample(rng: () => number)` -> takes an injected `FnMut() -> f64` RNG
//!     (no ambient impurity; `rand` is not an available dependency).
//!   * `mean()` / `variance()` use `Number(state)` and are only valid when `S`
//!     is numeric; bounded here with `where S: Copy + Into<f64>`.
//!   * `throw new Error(...)` are invariant violations -> `panic!`.

/// A probability distribution over a finite set of hidden states `S`, with
/// Bayesian update. `states` and `weights` are parallel; `weights` always sums
/// to 1 (modulo floating-point error) after construction and every update.
#[derive(Debug)]
pub struct DiscreteBelief<S = f64> {
    pub states: Vec<S>,
    pub weights: Vec<f64>,
}

impl<S> DiscreteBelief<S> {
    /// Construct from a state set and an optional (unnormalised) prior. With no
    /// prior, the belief is uniform. Panics on an empty state set, a prior of
    /// the wrong length, or a degenerate prior (non-positive / non-finite sum).
    pub fn new(states: Vec<S>, prior: Option<&[f64]>) -> Self {
        if states.is_empty() {
            panic!("DiscreteBelief: empty state set");
        }
        let weights = match prior {
            Some(prior) => {
                if prior.len() != states.len() {
                    panic!(
                        "prior length {} ≠ states length {}",
                        prior.len(),
                        states.len()
                    );
                }
                let total: f64 = prior.iter().sum();
                if total <= 0.0 || !total.is_finite() {
                    panic!("prior is degenerate (sum={})", total);
                }
                prior.iter().map(|w| w / total).collect()
            }
            None => {
                let u = 1.0 / states.len() as f64;
                vec![u; states.len()]
            }
        };
        DiscreteBelief { states, weights }
    }

    /// Bayesian update `b'(s) ∝ likelihood(s) · b(s)`. Returns the normalising
    /// constant `Σ_s likelihood(s) · b(s)` (the marginal likelihood of the
    /// observation). On belief collapse (non-finite or non-positive total) the
    /// belief falls back to uniform and the (degenerate) total is still
    /// returned so the caller can decide whether that is acceptable.
    pub fn update<F>(&mut self, likelihood: F) -> f64
    where
        F: Fn(&S, usize) -> f64,
    {
        let mut total = 0.0;
        let mut next = vec![0.0; self.weights.len()];
        for i in 0..self.weights.len() {
            let l = likelihood(&self.states[i], i);
            if l < 0.0 {
                panic!("likelihood({}) returned negative value {}", i, l);
            }
            next[i] = self.weights[i] * l;
            total += next[i];
        }
        if !total.is_finite() || total <= 0.0 {
            let u = 1.0 / self.weights.len() as f64;
            for w in self.weights.iter_mut() {
                *w = u;
            }
            return total;
        }
        for i in 0..self.weights.len() {
            self.weights[i] = next[i] / total;
        }
        total
    }

    /// Predictive update for a hidden state evolving via `T(s' | s)`.
    /// `transition(prev_state, prev_index)` returns a weight vector parallel to
    /// `self.states`. The new belief is `b'(s') = Σ_s T(s' | s) · b(s)`. Panics
    /// on a wrong-length transition row or a degenerate result distribution.
    pub fn propagate<F>(&mut self, transition: F)
    where
        F: Fn(&S, usize) -> Vec<f64>,
    {
        let k = self.weights.len();
        let mut next = vec![0.0; k];
        for i in 0..k {
            let t_row = transition(&self.states[i], i);
            if t_row.len() != k {
                panic!("transition row length {} ≠ K = {}", t_row.len(), k);
            }
            let w = self.weights[i];
            for j in 0..t_row.len() {
                next[j] += w * t_row[j];
            }
        }
        let total: f64 = next.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            panic!("belief.propagate: degenerate distribution (sum={})", total);
        }
        for i in 0..k {
            self.weights[i] = next[i] / total;
        }
    }

    /// `E[f(s)]` under the current belief.
    pub fn expectation<F>(&self, f: F) -> f64
    where
        F: Fn(&S, usize) -> f64,
    {
        let mut m = 0.0;
        for i in 0..self.weights.len() {
            m += f(&self.states[i], i) * self.weights[i];
        }
        m
    }

    /// Shannon entropy in nats. `H = -Σ b log b`.
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &w in &self.weights {
            if w > 0.0 {
                h -= w * w.ln();
            }
        }
        h
    }

    /// Argmax (mode) index of the belief.
    pub fn mode_index(&self) -> usize {
        let mut bi = 0;
        for i in 1..self.weights.len() {
            if self.weights[i] > self.weights[bi] {
                bi = i;
            }
        }
        bi
    }

    /// The most probable hidden state.
    pub fn mode(&self) -> S
    where
        S: Clone,
    {
        self.states[self.mode_index()].clone()
    }

    /// Sample one hidden state from the belief. `rng` yields a uniform draw in
    /// `[0, 1)`; it is consulted exactly once.
    pub fn sample<F>(&self, mut rng: F) -> S
    where
        F: FnMut() -> f64,
        S: Clone,
    {
        let u = rng();
        let mut acc = 0.0;
        for i in 0..self.weights.len() {
            acc += self.weights[i];
            if u <= acc {
                return self.states[i].clone();
            }
        }
        self.states[self.weights.len() - 1].clone()
    }

    /// A defensive copy of the weight vector.
    pub fn as_array(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Clone the belief. Mirrors the TS `clone()`, which re-runs the constructor
    /// with the current (already normalised) weights as the prior.
    pub fn clone(&self) -> DiscreteBelief<S>
    where
        S: Clone,
    {
        DiscreteBelief::new(self.states.clone(), Some(&self.weights))
    }
}

impl<S> DiscreteBelief<S>
where
    S: Copy + Into<f64>,
{
    /// `E[s]` when `S` is numeric.
    pub fn mean(&self) -> f64 {
        let mut m = 0.0;
        for i in 0..self.weights.len() {
            m += self.states[i].into() * self.weights[i];
        }
        m
    }

    /// `Var[s]` when `S` is numeric.
    pub fn variance(&self) -> f64 {
        let mu = self.mean();
        let mut v = 0.0;
        for i in 0..self.weights.len() {
            let x: f64 = self.states[i].into();
            v += (x - mu) * (x - mu) * self.weights[i];
        }
        v
    }
}

// -----------------------------------------------------------------------------
// Standalone helpers for cross-checking calibration in validators.
// -----------------------------------------------------------------------------

/// A binary outcome `y ∈ {0, 1}` (the TS `0 | 1` literal union -> enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOutcome {
    Zero,
    One,
}

impl BinaryOutcome {
    fn as_f64(self) -> f64 {
        match self {
            BinaryOutcome::Zero => 0.0,
            BinaryOutcome::One => 1.0,
        }
    }
}

/// Brier score for a probabilistic prediction `p ∈ [0, 1]` against a binary
/// outcome `y ∈ {0, 1}`. Smaller is better. Always in `[0, 1]`.
pub fn brier_score(p: f64, y: BinaryOutcome) -> f64 {
    let y = y.as_f64();
    (p - y) * (p - y)
}

/// KL divergence `KL(p || q)` for two discrete distributions of the same
/// length. Returns `f64::INFINITY` when `q` assigns zero mass where `p` does
/// not. Panics on a length mismatch (invariant violation).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        panic!("klDivergence: length mismatch");
    }
    let mut kl = 0.0;
    for i in 0..p.len() {
        if p[i] == 0.0 {
            continue;
        }
        if q[i] <= 0.0 {
            return f64::INFINITY;
        }
        kl += p[i] * (p[i] / q[i]).ln();
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_prior_and_normalised_prior() {
        let b = DiscreteBelief::new(vec![0.0, 1.0, 2.0], None);
        assert_eq!(b.weights, vec![1.0 / 3.0; 3]);

        let b2 = DiscreteBelief::new(vec![0.0, 1.0], Some(&[3.0, 1.0]));
        assert!((b2.weights[0] - 0.75).abs() < 1e-12);
        assert!((b2.weights[1] - 0.25).abs() < 1e-12);
        // mean over numeric states.
        assert!((b2.mean() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn bayesian_update_returns_marginal_likelihood() {
        let mut b = DiscreteBelief::new(vec![0u32, 1u32], Some(&[0.5, 0.5]));
        // likelihood favouring state index 1.
        let total = b.update(|_s, i| if i == 1 { 0.8 } else { 0.2 });
        assert!((total - 0.5).abs() < 1e-12);
        assert!((b.weights[0] - 0.2).abs() < 1e-12);
        assert!((b.weights[1] - 0.8).abs() < 1e-12);
        assert_eq!(b.mode_index(), 1);
    }

    #[test]
    fn kl_and_brier() {
        assert_eq!(brier_score(1.0, BinaryOutcome::One), 0.0);
        assert!((brier_score(0.25, BinaryOutcome::Zero) - 0.0625).abs() < 1e-12);

        assert_eq!(kl_divergence(&[0.5, 0.5], &[0.5, 0.5]), 0.0);
        assert!(kl_divergence(&[0.5, 0.5], &[1.0, 0.0]).is_infinite());
    }
}
