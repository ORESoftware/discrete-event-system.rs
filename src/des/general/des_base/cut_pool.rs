//! Port of `src/des/general/des-base/cut-pool.ts` — reusable affine cut pools
//! (upper / lower envelopes) for decomposition algorithms: Benders / L-shaped,
//! SDDP, Kelley cutting planes, and outer approximation.
//!
//! A cut is an affine function `f(x) = alpha + beta·x`; a pool is either an
//! UPPER envelope (min of cuts) or a LOWER envelope (max of cuts).
//!
//! ## Rust shape (faithful translation)
//!
//!   * `type CutEnvelopeSense` → [`CutEnvelopeSense`] enum.
//!   * `interface AffineCut` → struct with `Option<String>` source.
//!   * `class AffineCutPool` (a plain numeric class, NOT a `DESStation`) →
//!     struct with private `cuts: Vec<AffineCut>`.
//!   * `beta.slice()` defensive copies → ownership transfer on `add`, `.clone()`
//!     on `all` / `active_cut`.
//!   * Empty-pool `evaluate` returns ±Infinity → `f64::INFINITY` /
//!     `f64::NEG_INFINITY`.
//!   * `activeCut(): AffineCut | null` → `Option<AffineCut>`.
//!   * `Preconditions.*` (which `throw` in TS) → `Result<_, PreconditionError>`
//!     via the ported [`Preconditions`] guards, propagated with `?`.

use super::preconditions::{Check, PreconditionError, Preconditions};

/// Whether the pool is an upper (min-of-cuts) or lower (max-of-cuts) envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutEnvelopeSense {
    Upper,
    Lower,
}

/// An affine cut `f(x) = alpha + beta·x`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineCut {
    pub alpha: f64,
    pub beta: Vec<f64>,
    /// Optional provenance string: `"terminal"`, `"iter=12 stage=2"`, ...
    pub source: Option<String>,
}

/// A pool of affine cuts forming an upper or lower envelope.
#[derive(Clone, Debug)]
pub struct AffineCutPool {
    pub dimension: usize,
    pub sense: CutEnvelopeSense,
    cuts: Vec<AffineCut>,
}

impl AffineCutPool {
    pub fn new(
        dimension: usize,
        sense: CutEnvelopeSense,
        initial_cuts: &[AffineCut],
    ) -> Result<Self, PreconditionError> {
        Preconditions::integer_in_range("AffineCutPool", "dimension", dimension as f64, 1.0, 1e6)?;
        let mut pool = AffineCutPool { dimension, sense, cuts: Vec::new() };
        for c in initial_cuts {
            pool.add(c.clone())?;
        }
        Ok(pool)
    }

    /// Validate and append a cut (takes ownership — the defensive-copy analogue
    /// of the TS `beta.slice()`).
    pub fn add(&mut self, cut: AffineCut) -> Check {
        self.assert_cut(&cut)?;
        self.cuts.push(cut);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.cuts.len()
    }

    /// Defensive copy of all cuts.
    pub fn all(&self) -> Vec<AffineCut> {
        self.cuts.clone()
    }

    pub fn evaluate_cut(&self, cut: &AffineCut, x: &[f64]) -> Result<f64, PreconditionError> {
        self.assert_point(x)?;
        let mut v = cut.alpha;
        for i in 0..self.dimension {
            v += cut.beta[i] * x[i];
        }
        Ok(v)
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<f64, PreconditionError> {
        self.assert_point(x)?;
        if self.cuts.is_empty() {
            return Ok(match self.sense {
                CutEnvelopeSense::Upper => f64::INFINITY,
                CutEnvelopeSense::Lower => f64::NEG_INFINITY,
            });
        }
        let mut best = match self.sense {
            CutEnvelopeSense::Upper => f64::INFINITY,
            CutEnvelopeSense::Lower => f64::NEG_INFINITY,
        };
        for cut in &self.cuts {
            let v = self.evaluate_cut(cut, x)?;
            match self.sense {
                CutEnvelopeSense::Upper => {
                    if v < best {
                        best = v;
                    }
                }
                CutEnvelopeSense::Lower => {
                    if v > best {
                        best = v;
                    }
                }
            }
        }
        Ok(best)
    }

    pub fn active_cut(&self, x: &[f64]) -> Result<Option<AffineCut>, PreconditionError> {
        self.assert_point(x)?;
        if self.cuts.is_empty() {
            return Ok(None);
        }
        let mut best_idx = 0;
        let mut best = self.evaluate_cut(&self.cuts[0], x)?;
        for i in 1..self.cuts.len() {
            let v = self.evaluate_cut(&self.cuts[i], x)?;
            if (self.sense == CutEnvelopeSense::Upper && v < best)
                || (self.sense == CutEnvelopeSense::Lower && v > best)
            {
                best = v;
                best_idx = i;
            }
        }
        Ok(Some(self.cuts[best_idx].clone()))
    }

    fn assert_cut(&self, cut: &AffineCut) -> Check {
        Preconditions::finite("AffineCutPool", "cut.alpha", cut.alpha)?;
        Preconditions::length_eq("AffineCutPool", "cut.beta", &cut.beta, self.dimension)?;
        Preconditions::all_finite("AffineCutPool", "cut.beta", &cut.beta)?;
        Ok(())
    }

    fn assert_point(&self, x: &[f64]) -> Check {
        Preconditions::length_eq("AffineCutPool", "x", x, self.dimension)?;
        Preconditions::all_finite("AffineCutPool", "x", x)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(alpha: f64, beta: Vec<f64>) -> AffineCut {
        AffineCut { alpha, beta, source: None }
    }

    #[test]
    fn upper_envelope_takes_min_and_empty_is_infinity() {
        let mut pool = AffineCutPool::new(1, CutEnvelopeSense::Upper, &[]).unwrap();
        assert_eq!(pool.evaluate(&[0.0]).unwrap(), f64::INFINITY);
        pool.add(cut(10.0, vec![0.0])).unwrap(); // constant 10
        pool.add(cut(0.0, vec![1.0])).unwrap(); // x
        // min(10, x) at x = 3 -> 3
        assert_eq!(pool.evaluate(&[3.0]).unwrap(), 3.0);
        // at x = 20 -> min(10, 20) = 10
        assert_eq!(pool.evaluate(&[20.0]).unwrap(), 10.0);
    }

    #[test]
    fn lower_envelope_active_cut() {
        let mut pool = AffineCutPool::new(1, CutEnvelopeSense::Lower, &[]).unwrap();
        pool.add(cut(0.0, vec![1.0])).unwrap(); // x
        pool.add(cut(0.0, vec![-1.0])).unwrap(); // -x
        // max(x, -x) at x = -5 -> 5 from the -x cut (beta = -1)
        assert_eq!(pool.evaluate(&[-5.0]).unwrap(), 5.0);
        let active = pool.active_cut(&[-5.0]).unwrap().unwrap();
        assert_eq!(active.beta, vec![-1.0]);
    }

    #[test]
    fn precondition_failures_are_errors() {
        assert!(AffineCutPool::new(0, CutEnvelopeSense::Upper, &[]).is_err());
        let mut pool = AffineCutPool::new(2, CutEnvelopeSense::Upper, &[]).unwrap();
        // wrong beta length
        assert!(pool.add(cut(1.0, vec![1.0])).is_err());
        // non-finite alpha
        assert!(pool.add(cut(f64::NAN, vec![1.0, 2.0])).is_err());
        // wrong point length
        pool.add(cut(1.0, vec![1.0, 2.0])).unwrap();
        assert!(pool.evaluate(&[1.0]).is_err());
    }
}
