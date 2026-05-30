//! Port of `src/des/general/optim.ts` — multivariable optimization.
//!
//! Each solver is a config struct implementing `Transform<Problem, OptimResult>`
//! (PureTransform in TS). The problem (objective + gradient + start, plus
//! Hessian for Newton) is the transform input; the `@deprecated` free-fn shims
//! are dropped — call the struct directly. Closures `(x: number[]) => number`
//! become `impl Fn(&[f64]) -> f64`.
//!
//! All solvers minimize `f`. For maximization, negate.

use crate::des::general::expr::NumericalGradient;
use crate::des::shared::linalg::{LinAlg, LinearSystem, Matrix, VecOps, Vector};
use crate::des::shared::transform::Transform;

#[derive(Clone, Debug)]
pub struct OptimOptions {
    /// Gradient-norm tolerance for stopping.
    pub tol: f64,
    /// Hard iteration cap.
    pub max_iter: usize,
    /// Initial step size.
    pub initial_step: f64,
    /// Armijo constant c1.
    pub c1: f64,
    /// Backtracking ratio.
    pub rho: f64,
}

impl Default for OptimOptions {
    fn default() -> Self {
        OptimOptions {
            tol: 1e-8,
            max_iter: 500,
            initial_step: 1.0,
            c1: 1e-4,
            rho: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub iter: usize,
    pub fx: f64,
    pub grad_norm: f64,
}

#[derive(Clone, Debug)]
pub struct OptimResult {
    pub x: Vector,
    pub fx: f64,
    pub iterations: usize,
    pub converged: bool,
    pub final_grad_norm: f64,
    pub history: Vec<HistoryEntry>,
}

/// Objective + gradient + starting point (first-order solvers).
pub struct FirstOrderProblem<F, G> {
    pub f: F,
    pub grad: G,
    pub x0: Vector,
}

/// First-order problem plus an analytical Hessian (Newton).
pub struct SecondOrderProblem<F, G, H> {
    pub f: F,
    pub grad: G,
    pub hess: H,
    pub x0: Vector,
}

/// Gradient descent with Armijo backtracking line search.
///   x_{k+1} = x_k − α_k · ∇f(x_k)
pub struct GradientDescent {
    pub opts: OptimOptions,
}

impl GradientDescent {
    pub fn new(opts: OptimOptions) -> Self {
        GradientDescent { opts }
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        GradientDescent {
            opts: OptimOptions::default(),
        }
    }
}

impl<F, G> Transform<FirstOrderProblem<F, G>, OptimResult> for GradientDescent
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vector,
{
    fn transform(&self, problem: FirstOrderProblem<F, G>) -> OptimResult {
        let FirstOrderProblem { f, grad, x0 } = problem;
        let o = &self.opts;
        let mut x = x0.clone();
        let mut fx = f(&x);
        let mut history = Vec::new();
        for iter in 0..o.max_iter {
            let g = grad(&x);
            let gn = VecOps::norm2(&g);
            history.push(HistoryEntry {
                iter,
                fx,
                grad_norm: gn,
            });
            if gn < o.tol {
                return OptimResult {
                    x,
                    fx,
                    iterations: iter,
                    converged: true,
                    final_grad_norm: gn,
                    history,
                };
            }
            let mut alpha = o.initial_step;
            let mut x_new: Vector = x.iter().zip(&g).map(|(v, gi)| v - alpha * gi).collect();
            let mut f_new = f(&x_new);
            let mut bt = 0;
            while f_new > fx - o.c1 * alpha * gn * gn && bt < 50 {
                alpha *= o.rho;
                bt += 1;
                x_new = x.iter().zip(&g).map(|(v, gi)| v - alpha * gi).collect();
                f_new = f(&x_new);
            }
            x = x_new;
            fx = f_new;
        }
        let g = grad(&x);
        let gn = VecOps::norm2(&g);
        if gn >= o.tol {
            eprintln!("[optim.GradientDescent] hit max_iter={} without converging; final |grad|={gn} (tol={}), fx={fx}.", o.max_iter, o.tol);
        }
        OptimResult {
            x,
            fx,
            iterations: o.max_iter,
            converged: gn < o.tol,
            final_grad_norm: gn,
            history,
        }
    }
}

/// Damped Newton's method. Solves H·p = ∇f via Gaussian elimination, with
/// Armijo backtracking; falls back to the gradient direction if H is singular.
pub struct NewtonOptim {
    pub opts: OptimOptions,
}

impl NewtonOptim {
    pub fn new(opts: OptimOptions) -> Self {
        NewtonOptim { opts }
    }
}

impl Default for NewtonOptim {
    fn default() -> Self {
        NewtonOptim {
            opts: OptimOptions {
                tol: 1e-10,
                max_iter: 100,
                ..OptimOptions::default()
            },
        }
    }
}

impl<F, G, H> Transform<SecondOrderProblem<F, G, H>, OptimResult> for NewtonOptim
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vector,
    H: Fn(&[f64]) -> Matrix,
{
    fn transform(&self, problem: SecondOrderProblem<F, G, H>) -> OptimResult {
        let SecondOrderProblem { f, grad, hess, x0 } = problem;
        let o = &self.opts;
        let mut x = x0.clone();
        let mut fx = f(&x);
        let mut history = Vec::new();
        for iter in 0..o.max_iter {
            let g = grad(&x);
            let gn = VecOps::norm2(&g);
            history.push(HistoryEntry {
                iter,
                fx,
                grad_norm: gn,
            });
            if gn < o.tol {
                return OptimResult {
                    x,
                    fx,
                    iterations: iter,
                    converged: true,
                    final_grad_norm: gn,
                    history,
                };
            }
            let h = hess(&x);
            let p = match LinearSystem::new(&h, &g, 1e-12).try_solve() {
                Some(p) => p,
                None => {
                    eprintln!("[optim.NewtonOptim] Hessian singular at iter {iter}; falling back to gradient direction.");
                    g.clone()
                }
            };
            let mut alpha = 1.0;
            let mut x_new: Vector = x.iter().zip(&p).map(|(v, pi)| v - alpha * pi).collect();
            let mut f_new = f(&x_new);
            let directional = VecOps::dot(&p, &g);
            let mut bt = 0;
            while f_new > fx - o.c1 * alpha * directional && bt < 50 && directional > 0.0 {
                alpha *= o.rho;
                bt += 1;
                x_new = x.iter().zip(&p).map(|(v, pi)| v - alpha * pi).collect();
                f_new = f(&x_new);
            }
            x = x_new;
            fx = f_new;
        }
        let g = grad(&x);
        let gn = VecOps::norm2(&g);
        if gn >= o.tol {
            eprintln!("[optim.NewtonOptim] hit max_iter={} without converging; final |grad|={gn} (tol={}), fx={fx}.", o.max_iter, o.tol);
        }
        OptimResult {
            x,
            fx,
            iterations: o.max_iter,
            converged: gn < o.tol,
            final_grad_norm: gn,
            history,
        }
    }
}

/// BFGS quasi-Newton with a maintained inverse-Hessian approximation and Armijo
/// line search.
pub struct Bfgs {
    pub opts: OptimOptions,
}

impl Bfgs {
    pub fn new(opts: OptimOptions) -> Self {
        Bfgs { opts }
    }
}

impl Default for Bfgs {
    fn default() -> Self {
        Bfgs {
            opts: OptimOptions {
                tol: 1e-8,
                max_iter: 200,
                ..OptimOptions::default()
            },
        }
    }
}

impl<F, G> Transform<FirstOrderProblem<F, G>, OptimResult> for Bfgs
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vector,
{
    fn transform(&self, problem: FirstOrderProblem<F, G>) -> OptimResult {
        let FirstOrderProblem { f, grad, x0 } = problem;
        let o = &self.opts;
        let n = x0.len();
        let mut x = x0.clone();
        let mut fx = f(&x);
        let mut g = grad(&x);
        let mut h_inv = LinAlg::identity(n);
        let mut history = Vec::new();
        for iter in 0..o.max_iter {
            let gn = VecOps::norm2(&g);
            history.push(HistoryEntry {
                iter,
                fx,
                grad_norm: gn,
            });
            if gn < o.tol {
                return OptimResult {
                    x,
                    fx,
                    iterations: iter,
                    converged: true,
                    final_grad_norm: gn,
                    history,
                };
            }
            let p: Vector = LinAlg::mat_vec(&h_inv, &g).iter().map(|v| -v).collect();
            let directional = VecOps::dot(&g, &p);
            let mut alpha = 1.0;
            let mut x_new: Vector = x.iter().zip(&p).map(|(v, pi)| v + alpha * pi).collect();
            let mut f_new = f(&x_new);
            let mut bt = 0;
            while f_new > fx + o.c1 * alpha * directional && bt < 50 {
                alpha *= o.rho;
                bt += 1;
                x_new = x.iter().zip(&p).map(|(v, pi)| v + alpha * pi).collect();
                f_new = f(&x_new);
            }
            let s: Vector = x_new.iter().zip(&x).map(|(nv, ov)| nv - ov).collect();
            let g_new = grad(&x_new);
            let y: Vector = g_new.iter().zip(&g).map(|(nv, ov)| nv - ov).collect();
            let sy = VecOps::dot(&s, &y);
            if sy > 1e-12 {
                let hy = LinAlg::mat_vec(&h_inv, &y);
                let yhy = VecOps::dot(&y, &hy);
                let rho2 = 1.0 / sy;
                let mut h_new = LinAlg::identity(n);
                for i in 0..n {
                    for j in 0..n {
                        h_new[i][j] = h_inv[i][j] - rho2 * (s[i] * hy[j] + hy[i] * s[j])
                            + rho2 * rho2 * yhy * s[i] * s[j]
                            + rho2 * s[i] * s[j];
                    }
                }
                h_inv = h_new;
            }
            x = x_new;
            fx = f_new;
            g = g_new;
        }
        let gn = VecOps::norm2(&g);
        if gn >= o.tol {
            eprintln!("[optim.Bfgs] hit max_iter={} without converging; final |grad|={gn} (tol={}), fx={fx}.", o.max_iter, o.tol);
        }
        OptimResult {
            x,
            fx,
            iterations: o.max_iter,
            converged: gn < o.tol,
            final_grad_norm: gn,
            history,
        }
    }
}

/// Wrap an objective in a numerical gradient when none is supplied analytically.
/// Returns a closure `|x| -> grad`, mirroring the TS `AutoGradient`.
pub struct AutoGradient {
    pub h: f64,
}

impl Default for AutoGradient {
    fn default() -> Self {
        AutoGradient { h: 1e-6 }
    }
}

impl AutoGradient {
    pub fn new(h: f64) -> Self {
        AutoGradient { h }
    }

    /// Build a gradient evaluator for `f`.
    pub fn of<F>(&self, f: F) -> impl Fn(&[f64]) -> Vector
    where
        F: Fn(&[f64]) -> f64,
    {
        let h = self.h;
        move |x: &[f64]| NumericalGradient { h }.eval(&f, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // f = (x-3)^2 + (y+1)^2, min at (3,-1).
    fn quad_f(x: &[f64]) -> f64 {
        (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2)
    }
    fn quad_grad(x: &[f64]) -> Vector {
        vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)]
    }
    fn quad_hess(_x: &[f64]) -> Matrix {
        vec![vec![2.0, 0.0], vec![0.0, 2.0]]
    }

    #[test]
    fn gradient_descent_quadratic() {
        let r = GradientDescent::default().transform(FirstOrderProblem {
            f: quad_f,
            grad: quad_grad,
            x0: vec![0.0, 0.0],
        });
        assert!(r.converged);
        assert!((r.x[0] - 3.0).abs() < 1e-4);
        assert!((r.x[1] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn newton_quadratic_one_step() {
        let r = NewtonOptim::default().transform(SecondOrderProblem {
            f: quad_f,
            grad: quad_grad,
            hess: quad_hess,
            x0: vec![0.0, 0.0],
        });
        assert!(r.converged);
        assert!((r.x[0] - 3.0).abs() < 1e-6);
        assert!((r.x[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn bfgs_quadratic() {
        let r = Bfgs::default().transform(FirstOrderProblem {
            f: quad_f,
            grad: quad_grad,
            x0: vec![0.0, 0.0],
        });
        assert!(r.converged);
        assert!((r.x[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn auto_gradient_matches_analytic() {
        let ag = AutoGradient::default().of(quad_f);
        let g = ag(&[0.0, 0.0]);
        assert!((g[0] - (-6.0)).abs() < 1e-4);
        assert!((g[1] - 2.0).abs() < 1e-4);
    }
}
