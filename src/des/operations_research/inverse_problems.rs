//! Small dense inverse-problem routines: Tikhonov regularization, truncated SVD
//! filtering, residual norms, and finite-difference gradient checks.

use crate::des::shared::linalg::{LinAlg, LinearSystem, Matrix, SymmetricEigen, VecOps, Vector};

#[derive(Clone, Debug, PartialEq)]
pub enum TikhonovRegularization {
    /// `alpha * ||x - prior||_2^2`.
    Identity { alpha: f64, prior: Option<Vector> },
    /// `alpha * ||L(x - prior)||_2^2`.
    Matrix {
        alpha: f64,
        operator: Matrix,
        prior: Option<Vector>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearInverseResult {
    pub x: Vector,
    pub residual_norm: f64,
    pub regularization_norm: f64,
    pub objective: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TruncatedSvdResult {
    pub x: Vector,
    pub singular_values: Vector,
    pub kept: usize,
    pub residual_norm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientCheck {
    pub finite_difference: f64,
    pub directional_derivative: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
}

pub fn tikhonov_solve(
    a: &Matrix,
    b: &[f64],
    reg: TikhonovRegularization,
) -> Result<LinearInverseResult, String> {
    validate_matrix_vector(a, b)?;
    let n = LinAlg::cols(a);
    let (alpha, operator, prior) = match reg {
        TikhonovRegularization::Identity { alpha, prior } => (
            alpha,
            LinAlg::identity(n),
            prior.unwrap_or_else(|| vec![0.0; n]),
        ),
        TikhonovRegularization::Matrix {
            alpha,
            operator,
            prior,
        } => {
            validate_matrix(&operator, "regularization operator")?;
            if LinAlg::cols(&operator) != n {
                return Err(format!(
                    "regularization operator has {} columns, expected {n}",
                    LinAlg::cols(&operator)
                ));
            }
            let prior = prior.unwrap_or_else(|| vec![0.0; n]);
            (alpha, operator, prior)
        }
    };
    if alpha < 0.0 || !alpha.is_finite() {
        return Err("alpha must be non-negative and finite".to_string());
    }
    if prior.len() != n {
        return Err(format!(
            "prior length {} != parameter dimension {n}",
            prior.len()
        ));
    }
    if prior.iter().any(|v| !v.is_finite()) {
        return Err("prior must contain only finite values".to_string());
    }

    let at = LinAlg::transpose(a);
    let ata = LinAlg::mat_mul(&at, a);
    let atb = LinAlg::mat_vec(&at, b);
    let lt = LinAlg::transpose(&operator);
    let ltl = LinAlg::mat_mul(&lt, &operator);
    let prior_penalty = LinAlg::mat_vec(&ltl, &prior);

    let normal = LinAlg::add(&ata, &LinAlg::scale(&ltl, alpha));
    let rhs: Vector = atb
        .iter()
        .zip(prior_penalty)
        .map(|(lhs, rhs)| lhs + alpha * rhs)
        .collect();
    let x = LinearSystem::new(&normal, &rhs, 1e-12)
        .try_solve()
        .ok_or_else(|| "regularized normal equations are singular".to_string())?;
    Ok(linear_inverse_result(a, b, &operator, &prior, alpha, x))
}

/// Truncated-SVD style spectral filtering using the eigendecomposition of
/// `A^T A`. Suitable for small dense teaching/model-checking problems.
pub fn truncated_svd_solve(
    a: &Matrix,
    b: &[f64],
    singular_cutoff: f64,
) -> Result<TruncatedSvdResult, String> {
    validate_matrix_vector(a, b)?;
    if singular_cutoff < 0.0 || !singular_cutoff.is_finite() {
        return Err("singular_cutoff must be non-negative and finite".to_string());
    }
    let n = LinAlg::cols(a);
    let at = LinAlg::transpose(a);
    let ata = LinAlg::mat_mul(&at, a);
    let mut eigen = SymmetricEigen::new(&ata, 80);
    let values = eigen.values();
    let vectors = eigen.vectors();
    let mut pairs: Vec<(f64, Vector)> = (0..n)
        .map(|j| {
            let sigma = values[j].max(0.0).sqrt();
            let v = (0..n).map(|i| vectors[i][j]).collect::<Vector>();
            (sigma, v)
        })
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut x = vec![0.0; n];
    let mut kept = 0usize;
    let mut singular_values = Vec::with_capacity(n);
    for (sigma, v) in pairs {
        singular_values.push(sigma);
        if sigma <= singular_cutoff {
            continue;
        }
        kept += 1;
        let av = LinAlg::mat_vec(a, &v);
        let coeff = VecOps::dot(&av, b) / (sigma * sigma);
        for i in 0..n {
            x[i] += coeff * v[i];
        }
    }

    let residual = try_residual_vector(a, &x, b)?;
    Ok(TruncatedSvdResult {
        x,
        singular_values,
        kept,
        residual_norm: VecOps::norm2(&residual),
    })
}

pub fn residual_vector(a: &Matrix, x: &[f64], b: &[f64]) -> Vector {
    try_residual_vector(a, x, b).expect("residual_vector: invalid matrix/vector dimensions")
}

pub fn try_residual_vector(a: &Matrix, x: &[f64], b: &[f64]) -> Result<Vector, String> {
    validate_matrix_vector(a, b)?;
    if x.len() != LinAlg::cols(a) {
        return Err(format!(
            "x length {} != matrix column count {}",
            x.len(),
            LinAlg::cols(a)
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("x contains a non-finite value".to_string());
    }
    Ok(LinAlg::mat_vec(a, x)
        .iter()
        .zip(b)
        .map(|(ax, bi)| ax - bi)
        .collect())
}

pub fn finite_difference_gradient_check<F, G>(
    objective: F,
    gradient: G,
    x: &[f64],
    direction: &[f64],
    epsilon: f64,
) -> Result<GradientCheck, String>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vector,
{
    if x.len() != direction.len() {
        return Err(format!(
            "direction length {} != x length {}",
            direction.len(),
            x.len()
        ));
    }
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err("epsilon must be positive and finite".to_string());
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("x must contain only finite values".to_string());
    }
    if direction.iter().any(|v| !v.is_finite()) {
        return Err("direction must contain only finite values".to_string());
    }
    if VecOps::norm2(direction) <= 0.0 {
        return Err("direction must be non-zero".to_string());
    }
    let x_plus: Vector = x
        .iter()
        .zip(direction)
        .map(|(xi, di)| xi + epsilon * di)
        .collect();
    let x_minus: Vector = x
        .iter()
        .zip(direction)
        .map(|(xi, di)| xi - epsilon * di)
        .collect();
    let f_plus = objective(&x_plus);
    let f_minus = objective(&x_minus);
    if !f_plus.is_finite() || !f_minus.is_finite() {
        return Err("objective returned a non-finite value during gradient check".to_string());
    }
    let finite_difference = (f_plus - f_minus) / (2.0 * epsilon);
    let grad = gradient(x);
    if grad.len() != x.len() {
        return Err(format!(
            "gradient length {} != x length {}",
            grad.len(),
            x.len()
        ));
    }
    if grad.iter().any(|v| !v.is_finite()) {
        return Err("gradient returned a non-finite value".to_string());
    }
    let directional_derivative = VecOps::dot(&grad, direction);
    let absolute_error = (finite_difference - directional_derivative).abs();
    let scale = finite_difference
        .abs()
        .max(directional_derivative.abs())
        .max(1.0);
    Ok(GradientCheck {
        finite_difference,
        directional_derivative,
        absolute_error,
        relative_error: absolute_error / scale,
    })
}

fn linear_inverse_result(
    a: &Matrix,
    b: &[f64],
    operator: &Matrix,
    prior: &[f64],
    alpha: f64,
    x: Vector,
) -> LinearInverseResult {
    let residual = try_residual_vector(a, &x, b).expect("validated inverse result dimensions");
    let shifted: Vector = x.iter().zip(prior).map(|(xi, pi)| xi - pi).collect();
    let regularized = LinAlg::mat_vec(operator, &shifted);
    let residual_norm = VecOps::norm2(&residual);
    let regularization_norm = VecOps::norm2(&regularized);
    LinearInverseResult {
        x,
        residual_norm,
        regularization_norm,
        objective: 0.5 * residual_norm * residual_norm
            + 0.5 * alpha * regularization_norm * regularization_norm,
    }
}

fn validate_matrix_vector(a: &Matrix, b: &[f64]) -> Result<(), String> {
    validate_matrix(a, "matrix")?;
    if b.len() != a.len() {
        return Err(format!(
            "b length {} != matrix row count {}",
            b.len(),
            a.len()
        ));
    }
    if b.iter().any(|x| !x.is_finite()) {
        return Err("b contains a non-finite value".to_string());
    }
    Ok(())
}

fn validate_matrix(a: &Matrix, label: &str) -> Result<(), String> {
    if a.is_empty() {
        return Err(format!("{label} must have at least one row"));
    }
    let cols = LinAlg::cols(a);
    if cols == 0 {
        return Err(format!("{label} must have at least one column"));
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != cols {
            return Err(format!(
                "{label} row {i} has length {}, expected {cols}",
                row.len()
            ));
        }
        if row.iter().any(|x| !x.is_finite()) {
            return Err(format!("{label} row {i} contains a non-finite value"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tikhonov_solves_small_regularized_problem() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        let result = tikhonov_solve(
            &a,
            &b,
            TikhonovRegularization::Identity {
                alpha: 1e-6,
                prior: None,
            },
        )
        .unwrap();
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn truncated_svd_filters_small_singular_direction() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-8]];
        let b = vec![2.0, 1.0];
        let result = truncated_svd_solve(&a, &b, 1e-6).unwrap();
        assert_eq!(result.kept, 1);
        assert!((result.x[0] - 2.0).abs() < 1e-8);
        assert!(result.x[1].abs() < 1e-6);
    }

    #[test]
    fn finite_difference_check_matches_gradient() {
        let check = finite_difference_gradient_check(
            |x| x[0] * x[0] + 3.0 * x[1],
            |x| vec![2.0 * x[0], 3.0],
            &[2.0, 4.0],
            &[1.0, -2.0],
            1e-6,
        )
        .unwrap();
        assert!(check.relative_error < 1e-8, "{check:?}");
    }

    #[test]
    fn tikhonov_rejects_ragged_regularization_operator() {
        let err = tikhonov_solve(
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            &[1.0, 1.0],
            TikhonovRegularization::Matrix {
                alpha: 1.0,
                operator: vec![vec![1.0, 0.0], vec![0.0]],
                prior: None,
            },
        )
        .unwrap_err();
        assert!(err.contains("regularization operator row 1"));
    }

    #[test]
    fn checked_residual_reports_bad_x_shape() {
        let err = try_residual_vector(&vec![vec![1.0, 0.0]], &[1.0], &[1.0]).unwrap_err();
        assert!(err.contains("x length"));
    }

    #[test]
    fn gradient_check_rejects_zero_direction() {
        let err = finite_difference_gradient_check(|x| x[0], |_| vec![1.0], &[1.0], &[0.0], 1e-6)
            .unwrap_err();
        assert!(err.contains("non-zero"));
    }
}
