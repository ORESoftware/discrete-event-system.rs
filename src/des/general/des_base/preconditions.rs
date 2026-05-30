//! Rust port of `src/des/general/des-base/preconditions.ts`.

use crate::migration::MigrationFile;
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt::{Display, Formatter};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/preconditions.ts",
    "src/des/general/des_base/preconditions.rs",
    &[
        "PreconditionError is a typed std::error::Error value instead of a thrown Error.",
        "The TypeScript Preconditions namespace maps to associated functions on Preconditions.",
        "Matrix and vector guards accept Rust slices.",
        "Validation callers should propagate Result<(), PreconditionError> before a DES run starts.",
    ],
    &[
        "PreconditionError",
        "Preconditions",
        "allFinite",
        "arrNonNegative",
        "check",
        "equal",
        "finite",
        "inRange",
        "integer",
        "integerInRange",
        "lengthEq",
        "magnitudeLeq",
        "nonEmpty",
        "nonNegative",
        "notDivByZero",
        "positive",
        "positiveDefiniteCholesky",
        "positiveSemidefiniteDiag",
        "probabilityVector",
        "rectangularMatrix",
        "squareMatrix",
        "symmetricMatrix",
    ],
);

#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionError {
    pub model: String,
    pub param: String,
    pub condition: String,
    pub observed: Option<Value>,
}

impl PreconditionError {
    pub fn new(
        model: impl Into<String>,
        param: impl Into<String>,
        condition: impl Into<String>,
        observed: Option<Value>,
    ) -> Self {
        Self {
            model: model.into(),
            param: param.into(),
            condition: condition.into(),
            observed,
        }
    }
}

impl Display for PreconditionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} must {}", self.model, self.param, self.condition)?;
        if let Some(observed) = &self.observed {
            write!(f, "; got {observed}")?;
        }
        Ok(())
    }
}

impl std::error::Error for PreconditionError {}

pub type PreconditionResult = Result<(), PreconditionError>;

pub struct Preconditions;

impl Preconditions {
    pub fn finite(model: &str, param: &str, x: f64) -> PreconditionResult {
        if !x.is_finite() {
            return Err(error(model, param, "be a finite number", json!(x)));
        }
        Ok(())
    }

    pub fn positive(model: &str, param: &str, x: f64) -> PreconditionResult {
        Self::finite(model, param, x)?;
        if x <= 0.0 {
            return Err(error(model, param, "be > 0 (positive, not zero)", json!(x)));
        }
        Ok(())
    }

    pub fn non_negative(model: &str, param: &str, x: f64) -> PreconditionResult {
        Self::finite(model, param, x)?;
        if x < 0.0 {
            return Err(error(model, param, "be >= 0", json!(x)));
        }
        Ok(())
    }

    pub fn in_range(model: &str, param: &str, x: f64, lo: f64, hi: f64) -> PreconditionResult {
        Self::finite(model, param, x)?;
        if x < lo || x > hi {
            return Err(error(
                model,
                param,
                &format!("be in [{lo}, {hi}]"),
                json!(x),
            ));
        }
        Ok(())
    }

    pub fn integer(model: &str, param: &str, x: f64) -> PreconditionResult {
        Self::finite(model, param, x)?;
        if x.fract() != 0.0 {
            return Err(error(model, param, "be an integer", json!(x)));
        }
        Ok(())
    }

    pub fn integer_in_range(
        model: &str,
        param: &str,
        x: f64,
        lo: i64,
        hi: i64,
    ) -> PreconditionResult {
        Self::integer(model, param, x)?;
        if x < lo as f64 || x > hi as f64 {
            return Err(error(
                model,
                param,
                &format!("be an integer in [{lo}, {hi}]"),
                json!(x),
            ));
        }
        Ok(())
    }

    pub fn all_finite(model: &str, param: &str, arr: &[f64]) -> PreconditionResult {
        for (i, value) in arr.iter().enumerate() {
            if !value.is_finite() {
                return Err(error(
                    model,
                    &format!("{param}[{i}]"),
                    "be a finite number",
                    json!(value),
                ));
            }
        }
        Ok(())
    }

    pub fn non_empty<T>(model: &str, param: &str, arr: &[T]) -> PreconditionResult {
        if arr.is_empty() {
            return Err(error(model, param, "be non-empty", json!(arr.len())));
        }
        Ok(())
    }

    pub fn length_eq<T>(
        model: &str,
        param: &str,
        arr: &[T],
        expected: usize,
    ) -> PreconditionResult {
        if arr.len() != expected {
            return Err(error(
                model,
                &format!("{param}.length"),
                &format!("equal {expected}"),
                json!(arr.len()),
            ));
        }
        Ok(())
    }

    pub fn arr_non_negative(model: &str, param: &str, arr: &[f64]) -> PreconditionResult {
        for (i, value) in arr.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 {
                return Err(error(
                    model,
                    &format!("{param}[{i}]"),
                    "be >= 0",
                    json!(value),
                ));
            }
        }
        Ok(())
    }

    pub fn probability_vector(
        model: &str,
        param: &str,
        arr: &[f64],
        tol: f64,
    ) -> PreconditionResult {
        Self::non_empty(model, param, arr)?;
        let mut total = 0.0;
        for (i, value) in arr.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 || *value > 1.0 + tol {
                return Err(error(
                    model,
                    &format!("{param}[{i}]"),
                    "be in [0, 1]",
                    json!(value),
                ));
            }
            total += value;
        }
        if (total - 1.0).abs() > tol {
            return Err(error(
                model,
                param,
                &format!("sum to 1 (within {tol})"),
                json!(total),
            ));
        }
        Ok(())
    }

    pub fn rectangular_matrix(model: &str, param: &str, matrix: &[Vec<f64>]) -> PreconditionResult {
        if matrix.is_empty() {
            return Err(error(model, param, "be a non-empty matrix", json!(matrix)));
        }
        let cols = matrix[0].len();
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != cols {
                return Err(error(
                    model,
                    &format!("{param}[{i}].length"),
                    &format!("equal {cols}"),
                    json!(row.len()),
                ));
            }
            for (j, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(error(
                        model,
                        &format!("{param}[{i}][{j}]"),
                        "be finite",
                        json!(value),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn square_matrix(model: &str, param: &str, matrix: &[Vec<f64>]) -> PreconditionResult {
        Self::rectangular_matrix(model, param, matrix)?;
        if matrix.len() != matrix[0].len() {
            return Err(error(
                model,
                param,
                "be a square matrix",
                json!([matrix.len(), matrix[0].len()]),
            ));
        }
        Ok(())
    }

    pub fn symmetric_matrix(
        model: &str,
        param: &str,
        matrix: &[Vec<f64>],
        tol: f64,
    ) -> PreconditionResult {
        Self::square_matrix(model, param, matrix)?;
        for (i, row) in matrix.iter().enumerate() {
            for (j, other_row) in matrix.iter().enumerate().skip(i + 1) {
                if (row[j] - other_row[i]).abs() > tol {
                    return Err(error(
                        model,
                        param,
                        &format!("be symmetric (M[{i}][{j}] vs M[{j}][{i}])"),
                        json!([row[j], other_row[i]]),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn positive_semidefinite_diag(
        model: &str,
        param: &str,
        matrix: &[Vec<f64>],
        tol: f64,
    ) -> PreconditionResult {
        Self::symmetric_matrix(model, param, matrix, tol)?;
        for (i, row) in matrix.iter().enumerate() {
            if row[i] < -tol {
                return Err(error(
                    model,
                    &format!("{param}[{i}][{i}]"),
                    "be >= 0 (PSD diagonal)",
                    json!(row[i]),
                ));
            }
        }
        Ok(())
    }

    pub fn positive_definite_cholesky(
        model: &str,
        param: &str,
        matrix: &[Vec<f64>],
    ) -> PreconditionResult {
        Self::square_matrix(model, param, matrix)?;
        let n = matrix.len();
        let mut lower = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = matrix[i][j];
                for (left, right) in lower[i].iter().zip(lower[j].iter()).take(j) {
                    sum -= left * right;
                }
                if i == j {
                    if sum <= 1e-12 {
                        return Err(error(
                            model,
                            param,
                            "be positive-definite (Cholesky failed)",
                            json!(sum),
                        ));
                    }
                    lower[i][j] = sum.sqrt();
                } else {
                    lower[i][j] = sum / lower[j][j];
                }
            }
        }
        Ok(())
    }

    pub fn not_div_by_zero(model: &str, param: &str, denom: f64, tol: f64) -> PreconditionResult {
        Self::finite(model, param, denom)?;
        if denom.abs() < tol {
            return Err(error(
                model,
                param,
                &format!("be non-zero (>{tol} in magnitude) - would divide by zero"),
                json!(denom),
            ));
        }
        Ok(())
    }

    pub fn check(
        model: &str,
        param: &str,
        condition: &str,
        ok: bool,
        observed: Option<Value>,
    ) -> PreconditionResult {
        if !ok {
            return Err(PreconditionError::new(model, param, condition, observed));
        }
        Ok(())
    }

    pub fn equal<T>(model: &str, param: &str, x: &T, expected: &T) -> PreconditionResult
    where
        T: PartialEq + Serialize,
    {
        if x != expected {
            return Err(error(
                model,
                param,
                &format!("equal {}", to_json_string(expected)),
                to_json_value(x),
            ));
        }
        Ok(())
    }

    pub fn magnitude_leq(model: &str, param: &str, x: f64, bound: f64) -> PreconditionResult {
        Self::finite(model, param, x)?;
        if x.abs() > bound {
            return Err(error(
                model,
                param,
                &format!("have magnitude <= {bound}"),
                json!(x),
            ));
        }
        Ok(())
    }
}

fn error(model: &str, param: &str, condition: &str, observed: Value) -> PreconditionError {
    PreconditionError::new(model, param, condition, Some(observed))
}

fn to_json_value<T>(value: &T) -> Value
where
    T: Serialize,
{
    serde_json::to_value(value).unwrap_or_else(|_| json!("<unserializable>"))
}

fn to_json_string<T>(value: &T) -> String
where
    T: Serialize,
{
    serde_json::to_string(value).unwrap_or_else(|_| "\"<unserializable>\"".to_owned())
}
