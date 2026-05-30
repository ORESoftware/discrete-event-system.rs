//! Port of `src/des/general/des-base/preconditions.ts` — fail-fast parameter
//! guards.
//!
//! `class PreconditionError extends Error` → a struct implementing
//! `std::error::Error`; the `Preconditions` namespace → associated functions on
//! a zero-sized type. Guards `throw` in TS; here they return
//! `Result<(), PreconditionError>` (recoverable construction-time failures, per
//! the migration rules). Callers `?`-propagate or `.expect()` at the edge.

use std::fmt;

use crate::des::shared::linalg::Matrix;

/// Returned by a failed `Preconditions` guard.
#[derive(Clone, Debug, PartialEq)]
pub struct PreconditionError {
    pub model: String,
    pub param: String,
    pub condition: String,
    pub observed: Option<String>,
}

impl PreconditionError {
    pub fn new(model: &str, param: &str, condition: &str, observed: Option<String>) -> Self {
        PreconditionError {
            model: model.to_string(),
            param: param.to_string(),
            condition: condition.to_string(),
            observed,
        }
    }
}

impl fmt::Display for PreconditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let obs = match &self.observed {
            Some(s) => format!("; got {s}"),
            None => String::new(),
        };
        write!(
            f,
            "{}: {} must {}{}",
            self.model, self.param, self.condition, obs
        )
    }
}

impl std::error::Error for PreconditionError {}

pub type Check = Result<(), PreconditionError>;

fn err(model: &str, param: &str, condition: &str, observed: Option<String>) -> Check {
    Err(PreconditionError::new(model, param, condition, observed))
}

/// Namespace of fail-fast guards (TS `namespace Preconditions`).
pub struct Preconditions;

impl Preconditions {
    /// Reject NaN / ±∞.
    pub fn finite(model: &str, param: &str, x: f64) -> Check {
        if !x.is_finite() {
            return err(model, param, "be a finite number", Some(x.to_string()));
        }
        Ok(())
    }

    /// Require `x > 0`.
    pub fn positive(model: &str, param: &str, x: f64) -> Check {
        Self::finite(model, param, x)?;
        if x <= 0.0 {
            return err(
                model,
                param,
                "be > 0 (positive, not zero)",
                Some(x.to_string()),
            );
        }
        Ok(())
    }

    /// Require `x >= 0`.
    pub fn non_negative(model: &str, param: &str, x: f64) -> Check {
        Self::finite(model, param, x)?;
        if x < 0.0 {
            return err(model, param, "be >= 0", Some(x.to_string()));
        }
        Ok(())
    }

    /// Require `lo <= x <= hi`.
    pub fn in_range(model: &str, param: &str, x: f64, lo: f64, hi: f64) -> Check {
        Self::finite(model, param, x)?;
        if x < lo || x > hi {
            return err(
                model,
                param,
                &format!("be in [{lo}, {hi}]"),
                Some(x.to_string()),
            );
        }
        Ok(())
    }

    /// Require `x` integral.
    pub fn integer(model: &str, param: &str, x: f64) -> Check {
        Self::finite(model, param, x)?;
        if x.fract() != 0.0 {
            return err(model, param, "be an integer", Some(x.to_string()));
        }
        Ok(())
    }

    /// Require `x` an integer in `[lo, hi]`.
    pub fn integer_in_range(model: &str, param: &str, x: f64, lo: f64, hi: f64) -> Check {
        Self::integer(model, param, x)?;
        if x < lo || x > hi {
            return err(
                model,
                param,
                &format!("be an integer in [{lo}, {hi}]"),
                Some(x.to_string()),
            );
        }
        Ok(())
    }

    /// Every element finite.
    pub fn all_finite(model: &str, param: &str, arr: &[f64]) -> Check {
        for (i, &v) in arr.iter().enumerate() {
            if !v.is_finite() {
                return err(
                    model,
                    &format!("{param}[{i}]"),
                    "be a finite number",
                    Some(v.to_string()),
                );
            }
        }
        Ok(())
    }

    /// Non-empty slice.
    pub fn non_empty<T>(model: &str, param: &str, arr: &[T]) -> Check {
        if arr.is_empty() {
            return err(model, param, "be non-empty", Some("0".to_string()));
        }
        Ok(())
    }

    /// `arr.len() == expected`.
    pub fn length_eq<T>(model: &str, param: &str, arr: &[T], expected: usize) -> Check {
        if arr.len() != expected {
            return err(
                model,
                &format!("{param}.length"),
                &format!("equal {expected}"),
                Some(arr.len().to_string()),
            );
        }
        Ok(())
    }

    /// Every element `>= 0`.
    pub fn arr_non_negative(model: &str, param: &str, arr: &[f64]) -> Check {
        for (i, &v) in arr.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return err(
                    model,
                    &format!("{param}[{i}]"),
                    "be >= 0",
                    Some(v.to_string()),
                );
            }
        }
        Ok(())
    }

    /// Probability mass function: each entry in `[0,1]`, total within `tol` of 1.
    pub fn probability_vector(model: &str, param: &str, arr: &[f64], tol: f64) -> Check {
        Self::non_empty(model, param, arr)?;
        let mut s = 0.0;
        for (i, &p) in arr.iter().enumerate() {
            if !p.is_finite() || p < 0.0 || p > 1.0 + tol {
                return err(
                    model,
                    &format!("{param}[{i}]"),
                    "be in [0, 1]",
                    Some(p.to_string()),
                );
            }
            s += p;
        }
        if (s - 1.0).abs() > tol {
            return err(
                model,
                param,
                &format!("sum to 1 (within {tol})"),
                Some(s.to_string()),
            );
        }
        Ok(())
    }

    /// Rectangular (uniform row length) and all finite.
    pub fn rectangular_matrix(model: &str, param: &str, m: &Matrix) -> Check {
        if m.is_empty() {
            return err(model, param, "be a non-empty matrix", None);
        }
        let cols = m[0].len();
        for (i, row) in m.iter().enumerate() {
            if row.len() != cols {
                return err(
                    model,
                    &format!("{param}[{i}].length"),
                    &format!("equal {cols}"),
                    Some(row.len().to_string()),
                );
            }
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return err(
                        model,
                        &format!("{param}[{i}][{j}]"),
                        "be finite",
                        Some(v.to_string()),
                    );
                }
            }
        }
        Ok(())
    }

    /// Square matrix.
    pub fn square_matrix(model: &str, param: &str, m: &Matrix) -> Check {
        Self::rectangular_matrix(model, param, m)?;
        if m.len() != m[0].len() {
            return err(
                model,
                param,
                "be a square matrix",
                Some(format!("[{}, {}]", m.len(), m[0].len())),
            );
        }
        Ok(())
    }

    /// Symmetric to within `tol`.
    pub fn symmetric_matrix(model: &str, param: &str, m: &Matrix, tol: f64) -> Check {
        Self::square_matrix(model, param, m)?;
        let n = m.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if (m[i][j] - m[j][i]).abs() > tol {
                    return err(
                        model,
                        param,
                        &format!("be symmetric (M[{i}][{j}] vs M[{j}][{i}])"),
                        None,
                    );
                }
            }
        }
        Ok(())
    }

    /// Square, symmetric, non-negative diagonal (necessary PSD condition).
    pub fn positive_semidefinite_diag(model: &str, param: &str, m: &Matrix, tol: f64) -> Check {
        Self::symmetric_matrix(model, param, m, tol)?;
        for i in 0..m.len() {
            if m[i][i] < -tol {
                return err(
                    model,
                    &format!("{param}[{i}][{i}]"),
                    "be >= 0 (PSD diagonal)",
                    Some(m[i][i].to_string()),
                );
            }
        }
        Ok(())
    }

    /// Positive-definite via Cholesky.
    pub fn positive_definite_cholesky(model: &str, param: &str, m: &Matrix) -> Check {
        Self::square_matrix(model, param, m)?;
        let n = m.len();
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = m[i][j];
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 1e-12 {
                        return err(
                            model,
                            param,
                            "be positive-definite (Cholesky failed)",
                            Some(s.to_string()),
                        );
                    }
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        Ok(())
    }

    /// Reject a near-zero denominator.
    pub fn not_div_by_zero(model: &str, param: &str, denom: f64, tol: f64) -> Check {
        Self::finite(model, param, denom)?;
        if denom.abs() < tol {
            return err(
                model,
                param,
                &format!("be non-zero (>{tol} in magnitude) — would divide by zero"),
                Some(denom.to_string()),
            );
        }
        Ok(())
    }

    /// Generic predicate guard.
    pub fn check(
        model: &str,
        param: &str,
        condition: &str,
        ok: bool,
        observed: Option<String>,
    ) -> Check {
        if !ok {
            return err(model, param, condition, observed);
        }
        Ok(())
    }

    /// `|x| <= bound`.
    pub fn magnitude_leq(model: &str, param: &str, x: f64, bound: f64) -> Check {
        Self::finite(model, param, x)?;
        if x.abs() > bound {
            return err(
                model,
                param,
                &format!("have magnitude <= {bound}"),
                Some(x.to_string()),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_guard() {
        assert!(Preconditions::positive("M", "dt", 0.5).is_ok());
        assert!(Preconditions::positive("M", "dt", 0.0).is_err());
        assert!(Preconditions::positive("M", "dt", f64::NAN).is_err());
    }

    #[test]
    fn probability_vector_guard() {
        assert!(Preconditions::probability_vector("M", "p", &[0.5, 0.5], 1e-6).is_ok());
        assert!(Preconditions::probability_vector("M", "p", &[0.5, 0.4], 1e-6).is_err());
    }

    #[test]
    fn pd_cholesky() {
        let pd = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        assert!(Preconditions::positive_definite_cholesky("M", "Q", &pd).is_ok());
        let bad = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        assert!(Preconditions::positive_definite_cholesky("M", "Q", &bad).is_err());
    }
}
