//! Port of `src/des/main-calculus.ts`.
//!
//! CLI dispatching `expr` / `ode` / `pde` / `poisson` problems onto the station
//! network and comparing against reference solvers.
//!
//! Delegates to `crate::des::general::{expr, quadrature, ode, equation_to_stations}`.
//! `process.env` → `std::env::var`; dispatch → `match` on `PROBLEM`.
//!
//! PORT NOTE: `toFunction` is reused from `crate::des::general::equation_to_stations`
//! (it lives there, not in `expr`). `richardsonDerivative` maps to
//! `expr::RichardsonDerivative`.
//!
//! PORT NOTE: `ANIMATE=1` rendering (FrameRecorder + calculus scene) is omitted.
//! The `out/*.json` artifacts are written via hand-built JSON strings (no
//! `serde` dependency assumed), to paths relative to the current directory.

#![allow(dead_code)]

use std::f64::consts::PI;

use crate::des::general::equation_to_stations::{
    build_field1d, build_ode_system, solve_poisson2d, Bc, Field1DFamily, Field1DScheme,
    Field1DSpec, Field2DScheme, OdeScheme, OdeSystemSpec, Poisson2DSpec,
};
use crate::des::general::expr::{diff, parse, stringify, RichardsonDerivative};
use crate::des::general::equation_to_stations::to_function;
use crate::des::general::ode::{RK45Integrator, RK45Options, IVP};
use crate::des::general::prng::mulberry32;
use crate::des::general::quadrature::{
    AdaptiveSimpsonRule, GaussLegendreRule, Integrand1D, MonteCarloIntegrator, QuadResult,
    SimpsonRule, TrapezoidRule,
};
use crate::des::shared::transform::{StatefulTransform, Transform};

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}
fn jarr(xs: &[f64]) -> String {
    format!("[{}]", xs.iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(","))
}
fn jarr2(rows: &[Vec<f64>]) -> String {
    format!("[{}]", rows.iter().map(|r| jarr(r)).collect::<Vec<_>>().join(","))
}

fn report_method(name: &str, r: QuadResult, reference: f64) {
    let err = (r.value - reference).abs();
    let mut line = format!(
        "#   {:<28} = {:.8}  err={:.2e}  evals={:>6}",
        name, r.value, err, r.evaluations
    );
    if let Some(se) = r.stderr {
        line.push_str(&format!("  stderr={se:.2e}"));
    }
    println!("{line}");
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let problem = env_str("PROBLEM", "expr");

    match problem.as_str() {
        "expr" => run_expr(),
        "ode" => run_ode(),
        "pde" => run_pde(),
        "poisson" => run_poisson(),
        other => {
            eprintln!("Unknown PROBLEM='{other}'. Try expr | ode | pde | poisson.");
        }
    }
}

fn run_expr() {
    let expr_str = env_str("EXPR", "x^2 * sin(x) + exp(-x)");
    let x_val = env_f64("X", 1.0);
    let a = env_f64("A", 0.0);
    let b = env_f64("B", PI);
    println!("# Expression: f(x) = {expr_str}");
    let f = parse(&expr_str);
    let f_fn = to_function(&f, &["x".to_string()]);
    let dfdx = diff(&f, "x");
    let df_fn = to_function(&dfdx, &["x".to_string()]);
    println!("# f({x_val}) = {:.10}", f_fn(&[x_val]));
    println!("# Symbolic derivative: f'(x) = {}", stringify(&dfdx));
    println!("# f'({x_val}) symbolic   = {:.10}", df_fn(&[x_val]));
    println!(
        "# f'({x_val}) Richardson = {:.10}",
        RichardsonDerivative::default().eval(|x| f_fn(&[x]), x_val)
    );
    println!("\n# Quadrature ∫_{a}^{b} f(x) dx, comparing 5 methods:");
    let ref_true = AdaptiveSimpsonRule::new(1e-15, 50).transform(Integrand1D::new(|x| f_fn(&[x]), a, b));
    let reference = ref_true.value;
    println!(
        "#   reference (adaptive Simpson at tol 1e-15) = {:.12}  ({} evals)",
        reference, ref_true.evaluations
    );
    report_method(
        "trapezoidal n=64",
        TrapezoidRule::new(64).transform(Integrand1D::new(|x| f_fn(&[x]), a, b)),
        reference,
    );
    report_method(
        "Simpson    n=64",
        SimpsonRule::new(64).transform(Integrand1D::new(|x| f_fn(&[x]), a, b)),
        reference,
    );
    report_method(
        "adaptive Simpson tol=1e-9",
        AdaptiveSimpsonRule::new(1e-9, 50).transform(Integrand1D::new(|x| f_fn(&[x]), a, b)),
        reference,
    );
    report_method(
        "Gauss-Legendre n=10",
        GaussLegendreRule::new(10).transform(Integrand1D::new(|x| f_fn(&[x]), a, b)),
        reference,
    );
    let mut mc = MonteCarloIntegrator::new(100_000, mulberry32(1));
    report_method(
        "Monte Carlo n=100k",
        mc.transform(Integrand1D::new(|x| f_fn(&[x]), a, b)),
        reference,
    );
}

fn run_ode() {
    let omega = env_f64("OMEGA", 1.0);
    let names: Vec<String> = env_str("NAMES", "y,v").split(',').map(|s| s.to_string()).collect();
    let rhs: Vec<String> = env_str("RHS", &format!("v;-{omega}*{omega}*y"))
        .split(';')
        .map(|s| s.to_string())
        .collect();
    let y0: Vec<f64> = env_str("Y0", "1,0").split(',').map(|s| s.parse().unwrap_or(0.0)).collect();
    let t1 = env_f64("T_END", 2.0 * PI);
    let dt = env_f64("DT", 0.01);
    println!("# ODE system:  d/dt [{}] = [{}]", names.join(", "), rhs.join(", "));
    println!(
        "#   y(0) = [{}],  t ∈ [0, {:.4}],  dt = {}",
        y0.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        t1,
        dt
    );

    // Pure-math reference: RK45 (adaptive).
    let arg_names: Vec<String> = {
        let mut v = vec!["t".to_string()];
        v.extend(names.iter().cloned());
        v
    };
    let fns: Vec<_> = rhs.iter().map(|s| to_function(&parse(s), &arg_names)).collect();
    let n = names.len();
    let f_ref = move |t: f64, y: &[f64]| -> Vec<f64> {
        let mut arg = Vec::with_capacity(y.len() + 1);
        arg.push(t);
        arg.extend_from_slice(y);
        fns.iter().map(|fi| fi(&arg)).collect()
    };
    let reference = RK45Integrator::new(RK45Options {
        rtol: Some(1e-12),
        atol: Some(1e-14),
        ..Default::default()
    })
    .transform(IVP { f: f_ref, y0: y0.clone(), t0: 0.0, t1 });

    println!("\n# Reference (RK45 adaptive, rtol=1e-12): {} accepted steps", reference.t.len());
    let ref_final = reference.y.last().expect("non-empty trace").clone();
    for (i, name) in names.iter().enumerate() {
        println!("#   {}({:.4}) = {:.10}", name, t1, ref_final[i]);
    }

    println!("\n# Station-network solvers (one station per state variable, dt = {dt}):");
    for (label, scheme) in [("euler", OdeScheme::Euler), ("rk2", OdeScheme::Rk2), ("rk4", OdeScheme::Rk4)] {
        let mut sim = build_ode_system(&OdeSystemSpec {
            names: names.clone(),
            rhs: rhs.clone(),
            y0: y0.clone(),
            scheme,
            rhs_exprs: None,
        });
        let sim_out = sim.run(0.0, t1, dt);
        let mut max_err = 0.0_f64;
        for i in 0..n {
            max_err = max_err.max((sim_out.final_values[i] - ref_final[i]).abs());
        }
        let finals: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, nm)| format!("{}={:.8}", nm, sim_out.final_values[i]))
            .collect();
        println!("#   {:<6} {}  max|Δ vs RK45 ref| = {:.3e}", label, finals.join("  "), max_err);
    }

    // Artifact write (cwd-relative; see PORT NOTE).
    let _ = std::fs::create_dir_all("out");
    let json = format!(
        "{{\"names\":[{}],\"rhs\":[{}],\"y0\":{},\"t1\":{},\"dt\":{},\"reference\":{{\"t\":{},\"y\":{}}}}}",
        names.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(","),
        rhs.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(","),
        jarr(&y0),
        jnum(t1),
        jnum(dt),
        jarr(&reference.t),
        jarr2(&reference.y),
    );
    let _ = std::fs::write("out/calculus-ode.json", json);
}

fn run_pde() {
    let family = env_str("FAMILY", "heat");
    let n = env_usize("N", 51);
    let t = env_f64("T_END", 0.5);
    if family == "heat" {
        let alpha = env_f64("ALPHA", 0.1);
        let init_expr = env_str("INIT", "sin(3.14159265358979 * x)");
        let dx = 1.0 / (n as f64 - 1.0);
        let dt_safe = 0.4 * dx * dx / alpha;
        let dt_big = 0.05;
        println!("# PDE: heat 1D  u_t = {alpha} · u_xx,  init={init_expr},  N={n},  T={t}");
        println!(
            "#   FTCS stability bound:  dt ≤ {:.6};  using dt={:.6}",
            dx * dx / (2.0 * alpha),
            dt_safe
        );
        println!("#   BTCS unconditionally stable;             using dt={dt_big}");
        let mut r_ftcs = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: init_expr.clone(),
            family: Field1DFamily::Heat,
            alpha_expr: Some(alpha.to_string()),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Ftcs,
        });
        let mut r_btcs = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: init_expr.clone(),
            family: Field1DFamily::Heat,
            alpha_expr: Some(alpha.to_string()),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Btcs,
        });
        let xs = r_ftcs.xs.clone();
        let out_ftcs = r_ftcs.run(0.0, t, dt_safe);
        let out_btcs = r_btcs.run(0.0, t, dt_big);

        if std::env::var("ANIMATE").as_deref() == Ok("1") {
            println!("#   (animation omitted in Rust port — see PORT NOTE)");
        }

        let decay = (-alpha * PI * PI * t).exp();
        let mut err_ftcs = 0.0_f64;
        let mut err_btcs = 0.0_f64;
        for i in 0..n {
            let exact = decay * (PI * xs[i]).sin();
            err_ftcs = err_ftcs.max((out_ftcs.final_values[i] - exact).abs());
            err_btcs = err_btcs.max((out_btcs.final_values[i] - exact).abs());
        }
        println!("\n# Final t = {t},  analytical peak = exp(-απ²T) = {decay:.6}");
        println!("#   FTCS ({} ticks):  max|err| = {:.3e}", out_ftcs.ticks, err_ftcs);
        println!("#   BTCS ({} ticks):  max|err| = {:.3e}", out_btcs.ticks, err_btcs);

        let _ = std::fs::create_dir_all("out");
        let analytical: Vec<f64> = xs.iter().map(|&x| decay * (PI * x).sin()).collect();
        let json = format!(
            "{{\"N\":{n},\"T\":{},\"alpha\":{},\"dtSafe\":{},\"dtBig\":{},\"xs\":{},\"finalFtcs\":{},\"finalBtcs\":{},\"analytical\":{}}}",
            jnum(t),
            jnum(alpha),
            jnum(dt_safe),
            jnum(dt_big),
            jarr(&xs),
            jarr(&out_ftcs.final_values),
            jarr(&out_btcs.final_values),
            jarr(&analytical),
        );
        let _ = std::fs::write("out/calculus-heat1d.json", json);
    } else if family == "wave" {
        let c = env_f64("C", 1.0);
        let init_expr = env_str("INIT", "sin(3.14159265358979 * x)");
        let dx = 1.0 / (n as f64 - 1.0);
        let dt = 0.5 * dx / c;
        println!("# PDE: wave 1D  u_tt = {c}² · u_xx,  init={init_expr},  v(x,0)=0,  N={n},  T={t}");
        println!("#   CFL bound:  c·dt/dx ≤ 1;  using dt={dt:.6} (c·dt/dx = 0.5)");
        let mut r = build_field1d(&Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr,
            family: Field1DFamily::Wave,
            alpha_expr: None,
            source_expr: None,
            c_expr: Some(c.to_string()),
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Leapfrog,
        });
        let xs = r.xs.clone();
        let out = r.run(0.0, t, dt);
        let mut err = 0.0_f64;
        for i in 0..n {
            let exact = (PI * xs[i]).sin() * (PI * c * t).cos();
            err = err.max((out.final_values[i] - exact).abs());
        }
        println!("#   leapfrog ({} ticks):  max|err vs cos(πct)·sin(πx)| = {:.3e}", out.ticks, err);
    }
}

fn run_poisson() {
    let n = env_usize("N", 41);
    let rho_expr = env_str(
        "RHO",
        "2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)",
    );
    let tol = env_f64("TOL", 1e-8);
    println!("# 2-D Poisson: ∇²u = −ρ(x, y),  ρ = {rho_expr}");
    println!("#   grid {n}×{n}, [0,1]², u=0 on boundary, tol={tol}");
    let omega = env_f64("OMEGA", 1.85);
    for (label, scheme) in [
        ("jacobi", Field2DScheme::Jacobi),
        ("gauss-seidel", Field2DScheme::GaussSeidel),
        ("sor", Field2DScheme::Sor),
    ] {
        let r = solve_poisson2d(&Poisson2DSpec {
            nx: n,
            ny: n,
            x_lo: 0.0,
            x_hi: 1.0,
            y_lo: 0.0,
            y_hi: 1.0,
            rho_expr: rho_expr.clone(),
            init_expr: None,
            bc_expr: None,
            scheme,
            omega: Some(omega),
            max_iter: Some(50000),
            tol: Some(tol),
        });
        let mut max_err = 0.0_f64;
        for j in 0..r.ny {
            for i in 0..r.nx {
                let exact = (PI * r.xs[i]).sin() * (PI * r.ys[j]).sin();
                let got = r.u[j * r.nx + i];
                max_err = max_err.max((exact - got).abs());
            }
        }
        println!(
            "#   {:<13}  iters={:>6}  finalΔ={:.2e}  maxErr vs sin·sin = {:.2e}",
            label, r.iterations, r.final_delta, max_err
        );
    }
    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        println!("#   (animation omitted in Rust port — see PORT NOTE)");
    }
}
