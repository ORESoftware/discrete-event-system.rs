//! Nonlinear optimization surface: reexports of the crate's differentiable
//! solvers plus small constrained helpers used by OR models.

pub use crate::des::general::optim::{
    AutoGradient, Bfgs, FirstOrderProblem, GradientDescent, HistoryEntry, NewtonOptim,
    OptimOptions, OptimResult, SecondOrderProblem,
};

use crate::des::shared::linalg::{VecOps, Vector};

#[derive(Clone, Debug, PartialEq)]
pub struct BoxConstraints {
    pub lower: Vector,
    pub upper: Vector,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedGradientOptions {
    pub initial_step: f64,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for ProjectedGradientOptions {
    fn default() -> Self {
        ProjectedGradientOptions {
            initial_step: 1.0,
            tol: 1e-8,
            max_iter: 1000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedGradientResult {
    pub x: Vector,
    pub fx: f64,
    pub projected_gradient_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn project_box(x: &[f64], constraints: &BoxConstraints) -> Result<Vector, String> {
    validate_box(x.len(), constraints)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err("x must contain only finite values".to_string());
    }
    Ok(x.iter()
        .enumerate()
        .map(|(i, xi)| constraints.upper[i].min(constraints.lower[i].max(*xi)))
        .collect())
}

pub fn projected_gradient_norm<G>(
    gradient: G,
    x: &[f64],
    constraints: &BoxConstraints,
) -> Result<f64, String>
where
    G: Fn(&[f64]) -> Vector,
{
    validate_box(x.len(), constraints)?;
    let g = gradient(x);
    if g.len() != x.len() {
        return Err(format!(
            "gradient length {} != x length {}",
            g.len(),
            x.len()
        ));
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err("gradient returned a non-finite value".to_string());
    }
    let projected = project_box(
        &x.iter().zip(&g).map(|(xi, gi)| xi - gi).collect::<Vector>(),
        constraints,
    )?;
    let residual: Vector = x.iter().zip(projected).map(|(xi, pi)| xi - pi).collect();
    Ok(VecOps::norm2(&residual))
}

/// Projected-gradient descent for smooth objectives over box constraints.
pub fn projected_gradient_descent<F, G>(
    f: F,
    gradient: G,
    x0: &[f64],
    constraints: &BoxConstraints,
    options: ProjectedGradientOptions,
) -> Result<ProjectedGradientResult, String>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vector,
{
    validate_projected_gradient_options(&options)?;
    let mut x = project_box(x0, constraints)?;
    let mut fx = f(&x);
    if !fx.is_finite() {
        return Err("objective returned a non-finite value at x0".to_string());
    }
    let mut pg_norm = f64::INFINITY;
    let mut iterations = 0usize;
    for iter in 0..options.max_iter {
        let g = gradient(&x);
        if g.len() != x.len() {
            return Err(format!(
                "gradient length {} != x length {}",
                g.len(),
                x.len()
            ));
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err("gradient returned a non-finite value".to_string());
        }
        pg_norm = projected_gradient_norm(|_| g.clone(), &x, constraints)?;
        if pg_norm <= options.tol {
            iterations = iter;
            break;
        }
        let mut step = options.initial_step;
        let mut accepted = false;
        while step >= 1e-14 {
            let candidate = project_box(
                &x.iter()
                    .zip(&g)
                    .map(|(xi, gi)| xi - step * gi)
                    .collect::<Vector>(),
                constraints,
            )?;
            let candidate_fx = f(&candidate);
            if candidate_fx.is_finite()
                && (candidate_fx <= fx - 1e-4 * step * pg_norm * pg_norm || candidate_fx < fx)
            {
                x = candidate;
                fx = candidate_fx;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        iterations = iter + 1;
        if !accepted {
            break;
        }
    }
    if !pg_norm.is_finite() {
        pg_norm = projected_gradient_norm(gradient, &x, constraints)?;
    }
    Ok(ProjectedGradientResult {
        x,
        fx,
        projected_gradient_norm: pg_norm,
        iterations,
        converged: pg_norm <= options.tol,
    })
}

fn validate_box(n: usize, constraints: &BoxConstraints) -> Result<(), String> {
    if constraints.lower.len() != n || constraints.upper.len() != n {
        return Err(format!(
            "box dimensions lower={}, upper={}, expected {n}",
            constraints.lower.len(),
            constraints.upper.len()
        ));
    }
    for i in 0..n {
        if !constraints.lower[i].is_finite() || !constraints.upper[i].is_finite() {
            return Err(format!("box bound {i} is non-finite"));
        }
        if constraints.lower[i] > constraints.upper[i] {
            return Err(format!(
                "lower[{i}]={} > upper[{i}]={}",
                constraints.lower[i], constraints.upper[i]
            ));
        }
    }
    Ok(())
}

fn validate_projected_gradient_options(options: &ProjectedGradientOptions) -> Result<(), String> {
    if options.initial_step <= 0.0 || !options.initial_step.is_finite() {
        return Err(format!(
            "initial_step must be positive and finite; got {}",
            options.initial_step
        ));
    }
    if options.tol < 0.0 || !options.tol.is_finite() {
        return Err(format!(
            "tol must be non-negative and finite; got {}",
            options.tol
        ));
    }
    if options.max_iter == 0 {
        return Err("max_iter must be positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_gradient_respects_box() {
        let constraints = BoxConstraints {
            lower: vec![0.0],
            upper: vec![1.0],
        };
        let result = projected_gradient_descent(
            |x| (x[0] - 3.0).powi(2),
            |x| vec![2.0 * (x[0] - 3.0)],
            &[0.2],
            &constraints,
            ProjectedGradientOptions::default(),
        )
        .unwrap();
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!(result.projected_gradient_norm < 1e-6);
    }

    #[test]
    fn projected_gradient_rejects_bad_options_and_gradient() {
        let constraints = BoxConstraints {
            lower: vec![0.0],
            upper: vec![1.0],
        };
        let err = projected_gradient_descent(
            |x| x[0] * x[0],
            |_| vec![0.0],
            &[0.5],
            &constraints,
            ProjectedGradientOptions {
                initial_step: 0.0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("initial_step"));

        let err = projected_gradient_norm(|_| vec![f64::NAN], &[0.5], &constraints).unwrap_err();
        assert!(err.contains("non-finite"));
    }
}
