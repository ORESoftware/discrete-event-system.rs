//! Port of `src/des/general/root.ts` — scalar root finding for `f: R -> R`.
//!
//! Methods:
//!   - [`bisection`] on `[a, b]` with a sign change (linear convergence, robust).
//!   - [`newton`] using an analytical derivative (quadratic convergence, Armijo damping).
//!   - [`secant`] derivative-free Newton via finite-difference slope (superlinear).
//!
//! All return a [`RootResult`]. Fully deterministic (no RNG/clock). The TS
//! `(x: number) => number` closures map to generic `F: Fn(f64) -> f64` params,
//! `console.warn` maps to `eprintln!`, and the bisection bracket invariant
//! violation (`throw`) maps to `panic!`.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootResult {
    pub root: f64,
    pub iterations: usize,
    pub converged: bool,
    pub final_residual: f64,
}

/// Bisection on `[a, b]` with sign change. Linear convergence; one bit per
/// iteration. Robust: always converges if `f` is continuous and `f(a)·f(b) < 0`.
pub fn bisection<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> RootResult {
    let mut fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        eprintln!(
            "[root.bisection] no sign change on [{}, {}]: f(a)={}, f(b)={}; bracket does not contain a root.",
            a, b, fa, fb
        );
        panic!(
            "bisection: no sign change on [{}, {}]: f(a)={}, f(b)={}",
            a, b, fa, fb
        );
    }
    let mut iter = 0;
    while iter < max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if fm.abs() < tol || (b - a) / 2.0 < tol {
            return RootResult {
                root: m,
                iterations: iter + 1,
                converged: true,
                final_residual: fm.abs(),
            };
        }
        if fa * fm < 0.0 {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
        iter += 1;
    }
    let m = 0.5 * (a + b);
    eprintln!(
        "[root.bisection] hit maxIter={} without reaching tol={}; residual={}, bracket width={}.",
        max_iter,
        tol,
        f(m).abs(),
        (b - a).abs()
    );
    RootResult {
        root: m,
        iterations: iter,
        converged: false,
        final_residual: f(m).abs(),
    }
}

/// Newton's method using analytical derivative `df`. Quadratic convergence
/// near a simple root; can diverge if `df(x_k) ≈ 0` or `x_0` is far. Uses
/// Armijo-style damping if a step would increase `|f|`.
pub fn newton<F: Fn(f64) -> f64, DF: Fn(f64) -> f64>(
    f: F,
    df: DF,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> RootResult {
    let mut x = x0;
    let mut fx = f(x);
    for i in 0..max_iter {
        if fx.abs() < tol {
            return RootResult {
                root: x,
                iterations: i,
                converged: true,
                final_residual: fx.abs(),
            };
        }
        let dfx = df(x);
        if dfx == 0.0 || !dfx.is_finite() {
            eprintln!(
                "[root.newton] derivative degenerate at iter {} (x={}, f'={}); aborting Newton iteration early.",
                i, x, dfx
            );
            break;
        }
        let step = fx / dfx;
        let mut alpha = 1.0;
        let mut x_next = x - alpha * step;
        let mut f_next = f(x_next);
        let mut damp = 0;
        while f_next.abs() > fx.abs() && damp < 20 {
            alpha *= 0.5;
            x_next = x - alpha * step;
            f_next = f(x_next);
            damp += 1;
        }
        x = x_next;
        fx = f_next;
    }
    let newton_converged = fx.abs() < tol;
    if !newton_converged {
        eprintln!(
            "[root.newton] did not converge within {} iters; final residual={} (tol={}).",
            max_iter,
            fx.abs(),
            tol
        );
    }
    RootResult {
        root: x,
        iterations: max_iter,
        converged: newton_converged,
        final_residual: fx.abs(),
    }
}

/// Secant method: derivative-free Newton, using finite-difference slope
/// between the last two iterates. Superlinear (golden ratio) convergence.
pub fn secant<F: Fn(f64) -> f64>(f: F, x0: f64, x1: f64, tol: f64, max_iter: usize) -> RootResult {
    let mut x0 = x0;
    let mut x1 = x1;
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    for i in 0..max_iter {
        if f1.abs() < tol {
            return RootResult {
                root: x1,
                iterations: i,
                converged: true,
                final_residual: f1.abs(),
            };
        }
        if f0 == f1 {
            eprintln!(
                "[root.secant] equal function values f(x0)=f(x1)={} at iter {}; secant slope is zero, aborting.",
                f1, i
            );
            break;
        }
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
    }
    let secant_converged = f1.abs() < tol;
    if !secant_converged {
        eprintln!(
            "[root.secant] did not converge within {} iters; final residual={} (tol={}).",
            max_iter,
            f1.abs(),
            tol
        );
    }
    RootResult {
        root: x1,
        iterations: max_iter,
        converged: secant_converged,
        final_residual: f1.abs(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_finds_sqrt2() {
        // f(x) = x^2 - 2 on [0, 2] -> root at sqrt(2).
        let res = bisection(|x| x * x - 2.0, 0.0, 2.0, 1e-12, 200);
        assert!(res.converged);
        assert!((res.root - 2.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn newton_and_secant_agree() {
        // f(x) = x^2 - 2, f'(x) = 2x.
        let n = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-12, 100);
        let s = secant(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100);
        assert!(n.converged);
        assert!(s.converged);
        assert!((n.root - s.root).abs() < 1e-9);
    }
}
