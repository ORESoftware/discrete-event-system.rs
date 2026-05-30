//! Port of `src/des/runners/validate-calculus.ts`.
//!
//! Validates the station-network calculus solvers in six studies: symbolic vs
//! numerical derivative, quadrature agreement, ODE station-network ≡ pure-math
//! RK4 ≡ scipy DOP853, 1-D heat (FTCS/BTCS), 1-D wave (leapfrog), and 2-D
//! Poisson (Jacobi/Gauss-Seidel/SOR). The TS top-level driver becomes [`run`].
//!
//! ## PORT NOTE
//!   * `execFileSync(python, [script], {env})` (scipy ground truth) →
//!     [`std::process::Command`]; a failed/absent interpreter yields `None`
//!     (TS `catch` → `null`), printing the same `SKIP` lines.
//!   * `richardsonDerivative` is not present in the Rust `expr` module, so a
//!     faithful local copy ([`richardson_derivative`]) is provided.
//!   * `toFunction` lives in [`crate::des::general::equation_to_stations`]; the
//!     quadrature/ODE/PDE solvers are `Transform`s (rule structs) rather than
//!     free functions, so each TS call becomes `Rule::new(..).transform(input)`.
//!   * `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

use crate::des::general::equation_to_stations::{
    build_field1d, build_ode_system, solve_poisson2d, to_function, Bc, Field1DFamily,
    Field1DScheme, Field1DSpec, Field2DScheme, OdeScheme, OdeSystemSpec, Poisson2DSpec,
};
use crate::des::general::expr::{diff, parse};
use crate::des::general::ode::{RK4Integrator, IVP};
use crate::des::general::quadrature::{
    AdaptiveSimpsonRule, GaussLegendreRule, Integrand1D, SimpsonRule, TrapezoidRule,
};
use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// Local numerics + scipy bridge.
// -----------------------------------------------------------------------------

/// PORT NOTE: faithful copy of `expr.RichardsonDerivative` (five-point
/// Richardson extrapolation of the central difference), absent from the Rust
/// `expr` port.
fn richardson_derivative<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-3;
    let d1 = (f(x + h) - f(x - h)) / (2.0 * h);
    let d2 = (f(x + h / 2.0) - f(x - h / 2.0)) / h;
    (4.0 * d2 - d1) / 3.0
}

fn root() -> PathBuf {
    match std::env::var("REPO_ROOT") {
        Ok(r) => PathBuf::from(r),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

/// `runPython(env)` — runs the scipy reference and parses the last stdout line.
/// Returns `None` on any failure (mirrors the TS `try/catch → null`).
fn run_python(extra_env: &[(&str, String)]) -> Option<JsonValue> {
    let python = std::env::var("CALCULUS_PY").unwrap_or_else(|_| "python3".to_string());
    let script = root().join("external-references").join("calculus").join("calculus.py");
    let mut cmd = Command::new(&python);
    cmd.arg(&script);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let last = stdout.trim().lines().last()?.to_string();
    parse_json(&last).ok()
}

// -----------------------------------------------------------------------------
// Formatting + check accounting.
// -----------------------------------------------------------------------------

fn fixed(n: f64, digits: usize) -> String {
    format!("{n:.digits$}")
}

/// `${n}` for a JS number — integers print without a decimal point.
fn js_num(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// `n.toExponential(digits)`.
fn to_exp(n: f64, digits: usize) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    if n == 0.0 {
        return format!("0.{}e+0", "0".repeat(digits));
    }
    let neg = n < 0.0;
    let a = n.abs();
    let mut exp = a.log10().floor() as i32;
    let mant = a / 10f64.powi(exp);
    let factor = 10f64.powi(digits as i32);
    let mut r = (mant * factor).round() / factor;
    if r >= 10.0 {
        r /= 10.0;
        exp += 1;
    }
    let sign = if exp >= 0 { "+" } else { "-" };
    let s = format!("{:.*}e{}{}", digits, r, sign, exp.abs());
    if neg {
        format!("-{s}")
    } else {
        s
    }
}

#[derive(Default)]
struct Counter {
    pass: usize,
    fail: usize,
}

impl Counter {
    fn check(&mut self, label: &str, ok: bool, detail: Option<String>) {
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
        let d = detail.map(|x| format!("  ({x})")).unwrap_or_default();
        println!("  {}    {label}{d}", if ok { "PASS" } else { "FAIL" });
    }
}

fn jget(v: &JsonValue, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

const PI: f64 = std::f64::consts::PI;

/// `main` — returns the exit code (0 = no failures).
pub fn run() -> i32 {
    let mut c = Counter::default();
    let xargs = ["x".to_string()];

    // -------------------------------------------------------------------------
    // STUDY 1: Symbolic derivative ≡ numerical
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 1: Symbolic vs numerical derivative ===");
    let deriv_cases: [(&str, f64); 6] = [
        ("x^2", 1.7),
        ("sin(x) * cos(x)", 0.4),
        ("exp(-x^2)", 0.6),
        ("x^3 + 2*x^2 - 5*x + 1", 2.0),
        ("log(x) * sin(x)", 1.2),
        ("1 / (1 + x^2)", 0.5),
    ];
    for (expr_str, x) in deriv_cases {
        let ast = parse(expr_str);
        let f = to_function(&ast, &xargs);
        let df_sym = to_function(&diff(&ast, "x"), &xargs);
        let num_val = richardson_derivative(|xx| f(&[xx]), x);
        let sym_val = df_sym(&[x]);
        let err = (sym_val - num_val).abs();
        c.check(
            &format!(
                "d/dx[{expr_str}] @ x={}: sym={}  num={}",
                js_num(x),
                fixed(sym_val, 8),
                fixed(num_val, 8)
            ),
            err < 1e-7,
            Some(format!("|err|={}", to_exp(err, 2))),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 2: Quadrature methods agree
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 2: Quadrature methods on ∫_0^π (x²·sin(x) + e^{{−x}}) dx ===");
    let integrand = |x: f64| x * x * x.sin() + (-x).exp();
    let a = 0.0;
    let b = PI;
    let ref_ts = AdaptiveSimpsonRule::new(1e-15, 40)
        .transform(Integrand1D::new(integrand, a, b))
        .value;
    match run_python(&[("PROBLEM", "quad".to_string())]) {
        Some(py) => {
            let v = jget(&py, "value").unwrap_or(f64::NAN);
            c.check(
                "scipy.integrate.quad agrees with adaptive Simpson",
                (v - ref_ts).abs() < 1e-10,
                Some(format!("|Δ|={}", to_exp((v - ref_ts).abs(), 2))),
            );
        }
        None => println!("  SKIP    scipy reference unavailable (set CALCULUS_PY)"),
    }
    let trap = TrapezoidRule::new(64).transform(Integrand1D::new(integrand, a, b)).value;
    let simp = SimpsonRule::new(64).transform(Integrand1D::new(integrand, a, b)).value;
    let gauss = GaussLegendreRule::new(10).transform(Integrand1D::new(integrand, a, b)).value;
    c.check(
        "Simpson n=64 vs reference",
        (simp - ref_ts).abs() < 1e-7,
        Some(format!("|Δ|={}", to_exp((simp - ref_ts).abs(), 2))),
    );
    c.check(
        "Gauss-Legendre n=10 vs reference",
        (gauss - ref_ts).abs() < 1e-12,
        Some(format!("|Δ|={}", to_exp((gauss - ref_ts).abs(), 2))),
    );
    c.check(
        "trapezoidal n=64 within O(1/n²)",
        (trap - ref_ts).abs() < 5e-3,
        Some(format!("|Δ|={}", to_exp((trap - ref_ts).abs(), 2))),
    );

    // -------------------------------------------------------------------------
    // STUDY 3: ODE station network ≡ pure-math RK4 ≡ scipy DOP853
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 3: ODE station network ≡ pure-math RK4 ≡ scipy DOP853 ===");
    {
        let t1 = 4.0 * PI;
        let dt = 0.001;
        let mut sim = build_ode_system(&OdeSystemSpec {
            names: vec!["y".to_string(), "v".to_string()],
            rhs: vec!["v".to_string(), "-y".to_string()],
            y0: vec![1.0, 0.0],
            scheme: OdeScheme::Rk4,
            rhs_exprs: None,
        });
        let station_out = sim.run(0.0, t1, dt);
        let f_ref = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let pure_out = RK4Integrator::new(dt).transform(IVP {
            f: f_ref,
            y0: vec![1.0, 0.0],
            t0: 0.0,
            t1,
        });
        let station_y = station_out.final_values[0];
        let pure_y = pure_out.y[pure_out.y.len() - 1][0];
        let d_station_vs_pure = (station_y - pure_y).abs();
        c.check(
            "station-network RK4 ≡ pure-math RK4 (bit-level on same grid)",
            d_station_vs_pure < 1e-13,
            Some(format!("|Δ| = {}", to_exp(d_station_vs_pure, 2))),
        );
        c.check(
            "station-network RK4 vs cos(4π) = 1",
            (station_y - 1.0).abs() < 5e-6,
            Some(format!("|Δ| = {}", to_exp((station_y - 1.0).abs(), 2))),
        );
        match run_python(&[("PROBLEM", "ode".to_string()), ("T_END", js_num(t1))]) {
            Some(sci) => {
                let y_at = jget(&sci, "y_at_t1").unwrap_or(f64::NAN);
                c.check(
                    "scipy DOP853 vs cos(4π) = 1",
                    (y_at - 1.0).abs() < 1e-12,
                    Some(format!("|Δ| = {}", to_exp((y_at - 1.0).abs(), 2))),
                );
            }
            None => println!("  SKIP    scipy DOP853 reference unavailable"),
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 4: 1-D heat
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 4: 1-D heat (FTCS / BTCS station network) ===");
    {
        let n = 51usize;
        let alpha = 0.1;
        let t_end = 0.5;
        let dx = 1.0 / (n as f64 - 1.0);
        let dt_ftcs = 0.4 * dx * dx / alpha;
        let dt_btcs = 0.05;
        let init_expr = "sin(3.14159265358979 * x)".to_string();
        let decay = (-alpha * PI * PI * t_end).exp();
        let expected_peak = decay;

        let mut r1 = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: init_expr.clone(),
            family: Field1DFamily::Heat,
            alpha_expr: Some(js_num(alpha)),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Ftcs,
        });
        let o1 = r1.run(0.0, t_end, dt_ftcs);

        let mut r2 = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: init_expr.clone(),
            family: Field1DFamily::Heat,
            alpha_expr: Some(js_num(alpha)),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Btcs,
        });
        let o2 = r2.run(0.0, t_end, dt_btcs);

        let mut err_ftcs = 0.0_f64;
        let mut err_btcs = 0.0_f64;
        for i in 0..n {
            let exact = decay * (PI * r1.xs[i]).sin();
            err_ftcs = err_ftcs.max((o1.final_values[i] - exact).abs());
            err_btcs = err_btcs.max((o2.final_values[i] - exact).abs());
        }
        c.check(
            &format!("FTCS station-net ({} ticks at dt={})", o1.ticks, to_exp(dt_ftcs, 2)),
            err_ftcs < 5e-3,
            Some(format!("max|err vs analytical|={}", to_exp(err_ftcs, 3))),
        );
        c.check(
            &format!(
                "BTCS station-net ({} ticks at dt={}, FTCS would be UNSTABLE)",
                o2.ticks,
                js_num(dt_btcs)
            ),
            err_btcs < 5e-2,
            Some(format!("max|err vs analytical|={}", to_exp(err_btcs, 3))),
        );
        let ftcs_peak = o1.final_values[n / 2];
        let btcs_peak = o2.final_values[n / 2];
        c.check(
            &format!("FTCS peak agrees with exp(-απ²T) = {}", fixed(expected_peak, 6)),
            (ftcs_peak - expected_peak).abs() < 5e-3,
            Some(format!("peak={}", fixed(ftcs_peak, 6))),
        );
        c.check(
            &format!("BTCS peak agrees with exp(-απ²T) = {}", fixed(expected_peak, 6)),
            (btcs_peak - expected_peak).abs() < 5e-2,
            Some(format!("peak={}", fixed(btcs_peak, 6))),
        );
        match run_python(&[
            ("PROBLEM", "pde-heat".to_string()),
            ("N", js_num(n as f64)),
            ("ALPHA", js_num(alpha)),
            ("T_END", js_num(t_end)),
        ]) {
            Some(sci) => {
                let final_values = sci.get("final_values").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let mut err_sci = 0.0_f64;
                for i in 0..n {
                    let sv = final_values.get(i).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
                    err_sci = err_sci.max((o1.final_values[i] - sv).abs());
                }
                c.check(
                    "FTCS station-net ≡ scipy.LSODA on same FD spatial discretisation",
                    err_sci < 5e-3,
                    Some(format!("max|Δ|={}", to_exp(err_sci, 3))),
                );
            }
            None => println!("  SKIP    scipy LSODA reference unavailable"),
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 5: 1-D wave
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 5: 1-D wave (leapfrog) ===");
    {
        let n = 51usize;
        let cc = 1.0;
        let t_end = 0.5;
        let dx = 1.0 / (n as f64 - 1.0);
        let dt = 0.5 * dx / cc;
        let init_expr = "sin(3.14159265358979 * x)".to_string();
        let mut r = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr,
            family: Field1DFamily::Wave,
            alpha_expr: None,
            source_expr: None,
            c_expr: Some(js_num(cc)),
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Leapfrog,
        });
        let o = r.run(0.0, t_end, dt);
        let expected_amplitude = (PI * cc * t_end).cos();
        let mut err = 0.0_f64;
        for i in 0..n {
            let exact = (PI * r.xs[i]).sin() * expected_amplitude;
            err = err.max((o.final_values[i] - exact).abs());
        }
        c.check(
            &format!("leapfrog ({} ticks, CFL=0.5) vs sin(πx)·cos(πct)", o.ticks),
            err < 0.05,
            Some(format!("max|err|={}", to_exp(err, 3))),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 6: 2-D Poisson — Jacobi / Gauss-Seidel / SOR
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 6: 2-D Poisson, Jacobi / Gauss-Seidel / SOR ===");
    {
        let n = 41usize;
        let tol = 1e-8;
        let rho = "2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)".to_string();
        let make = |scheme: Field2DScheme, omega: Option<f64>| Poisson2DSpec {
            nx: n,
            ny: n,
            x_lo: 0.0,
            x_hi: 1.0,
            y_lo: 0.0,
            y_hi: 1.0,
            rho_expr: rho.clone(),
            init_expr: None,
            bc_expr: None,
            scheme,
            omega,
            max_iter: Some(50000),
            tol: Some(tol),
        };
        let r_j = solve_poisson2d(&make(Field2DScheme::Jacobi, None));
        let r_g = solve_poisson2d(&make(Field2DScheme::GaussSeidel, None));
        let r_s = solve_poisson2d(&make(Field2DScheme::Sor, Some(1.85)));
        let mut err_j = 0.0_f64;
        let mut err_g = 0.0_f64;
        let mut err_s = 0.0_f64;
        for j in 0..n {
            for i in 0..n {
                let exact = (PI * r_j.xs[i]).sin() * (PI * r_j.ys[j]).sin();
                err_j = err_j.max((r_j.u[j * n + i] - exact).abs());
                err_g = err_g.max((r_g.u[j * n + i] - exact).abs());
                err_s = err_s.max((r_s.u[j * n + i] - exact).abs());
            }
        }
        c.check(
            &format!("Jacobi pins to sin·sin within 1e-3 ({} iters)", r_j.iterations),
            err_j < 1e-3,
            Some(format!("maxErr={}", to_exp(err_j, 2))),
        );
        c.check(
            &format!("Gauss-Seidel pins to sin·sin within 1e-3 ({} iters)", r_g.iterations),
            err_g < 1e-3,
            Some(format!("maxErr={}", to_exp(err_g, 2))),
        );
        c.check(
            &format!("SOR(ω=1.85) pins to sin·sin within 1e-3 ({} iters)", r_s.iterations),
            err_s < 1e-3,
            Some(format!("maxErr={}", to_exp(err_s, 2))),
        );
        c.check(
            &format!("Gauss-Seidel ~2× faster than Jacobi ({} vs {})", r_g.iterations, r_j.iterations),
            (r_g.iterations as f64) < r_j.iterations as f64 * 0.7,
            None,
        );
        c.check(
            &format!("SOR(ω=1.85) ~10× faster than Jacobi ({} vs {})", r_s.iterations, r_j.iterations),
            (r_s.iterations as f64) < r_j.iterations as f64 * 0.1,
            None,
        );
        match run_python(&[
            ("PROBLEM", "poisson".to_string()),
            ("N", js_num(n as f64)),
            ("TOL", js_num(tol)),
        ]) {
            Some(sci) => {
                let sci_iters = jget(&sci, "iterations").unwrap_or(f64::NAN);
                c.check(
                    "station Jacobi iteration count == scipy Jacobi iteration count",
                    r_j.iterations as f64 == sci_iters,
                    Some(format!("{} vs {}", r_j.iterations, js_num(sci_iters))),
                );
                let sci_err = jget(&sci, "max_err_vs_analytical").unwrap_or(f64::NAN);
                c.check(
                    "station Jacobi maxErr ≡ scipy Jacobi maxErr (bit-comparable Jacobi)",
                    (err_j - sci_err).abs() < 1e-12,
                    Some(format!("|Δ|={}", to_exp((err_j - sci_err).abs(), 2))),
                );
            }
            None => println!("  SKIP    scipy Jacobi reference unavailable"),
        }
    }

    println!("\n=== Summary: {} passed, {} failed ===", c.pass, c.fail);
    if c.fail > 0 {
        1
    } else {
        0
    }
}
