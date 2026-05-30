//! Port of src/des/test/nonlinear-optimization-test.ts
//!
//! Tests Newton / quasi-Newton (BFGS) and nonlinear least-squares (Gauss-Newton,
//! Levenberg-Marquardt) DES models.
//!
//! PORT NOTE: the "registry smoke" group (get_model / run_from_spec) depends on
//! `des-registry`, which is not yet ported; it is deferred.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::nonlinear_optimization_models::{
        run_bfgs_rosenbrock, run_gauss_newton_curve_fit, run_levenberg_marquardt_curve_fit,
        run_newton_rosenbrock, NonlinearLeastSquaresParams, UnconstrainedOptParams,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn unconstrained_default() -> UnconstrainedOptParams {
        UnconstrainedOptParams { x0: None, max_iter: None, tol: None }
    }

    fn nls_default() -> NonlinearLeastSquaresParams {
        NonlinearLeastSquaresParams { points: None, initial: None, max_iter: None, tol: None, lambda: None }
    }

    #[test]
    fn newton_rosenbrock() {
        let r = run_newton_rosenbrock(unconstrained_default());
        assert!(close(r.x[0], 1.0, 1e-4), "x0={}", r.x[0]);
        assert!(close(r.x[1], 1.0, 1e-4), "x1={}", r.x[1]);
        assert!(r.objective < 1e-12, "f={}", r.objective);
        assert!(r.topology.movables.iter().any(|m| m == "OptStateToken"));

        let threw = std::panic::catch_unwind(|| {
            run_newton_rosenbrock(UnconstrainedOptParams { x0: Some(vec![f64::NAN, 1.0]), max_iter: None, tol: None })
        })
        .is_err();
        assert!(threw);
    }

    #[test]
    fn bfgs_rosenbrock() {
        let r = run_bfgs_rosenbrock(unconstrained_default());
        assert!(close(r.x[0], 1.0, 1e-4), "x0={}", r.x[0]);
        assert!(close(r.x[1], 1.0, 1e-4), "x1={}", r.x[1]);
        assert!(r.objective < 1e-10, "f={}", r.objective);
        assert!(r.topology.movables.iter().any(|m| m == "OptStateToken"));
    }

    #[test]
    fn nonlinear_least_squares() {
        let gn = run_gauss_newton_curve_fit(nls_default());
        let lm = run_levenberg_marquardt_curve_fit(nls_default());
        assert!(close(gn.params[0], 2.0, 2e-3), "a={}", gn.params[0]);
        assert!(close(gn.params[1], -0.5, 6e-3), "b={}", gn.params[1]);
        assert!(close(lm.params[0], 2.0, 2e-3), "a={}", lm.params[0]);
        assert!(close(lm.params[1], -0.5, 6e-3), "b={}", lm.params[1]);
        assert!(gn.topology.movables.iter().any(|m| m == "NLStateToken"));
        assert!(lm.topology.movables.iter().any(|m| m == "NLStateToken"));
    }
}
