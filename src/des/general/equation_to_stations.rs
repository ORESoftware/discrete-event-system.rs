//! Port of `src/des/general/equation-to-stations.ts` — module
//! `des::general::equation_to_stations`.
//!
//! Compiles ODE / PDE expression specs (expression strings) into wired
//! [`FieldSimulation`] station networks. Each builder returns a simulation that
//! is already wired and ready to run; the caller selects an integration scheme:
//!
//!   ODE:        euler | rk2 | rk4
//!   1-D PDE:    ftcs | btcs (heat); leapfrog (wave); upwind (advection)
//!   2-D PDE:    jacobi | gauss-seidel | sor (Poisson, iterative relaxation)
//!
//! The integration recipe is encoded in each station's updater closure; the
//! `FieldSimulation` engine just walks tick by tick.
//!
//! Conversion notes from the TS source:
//!   * `toFunction(expr, args)` has no Rust analogue in `expr.rs`, so a local
//!     [`to_function`] helper builds a `Box<dyn Fn(&[f64]) -> f64>` that maps
//!     positional arguments onto a named [`Env`] and evaluates.
//!   * `FieldUpdater` updaters are `Box<dyn Fn(...)>` closures capturing the
//!     compiled RHS functions via `Rc` (so each closure is `'static`).
//!   * RUNTIME METHOD REASSIGNMENT (`sim.run = function(...)` for the BTCS path)
//!     has no Rust analogue. It is modelled as an optional [`Btcs`] strategy
//!     carried on [`Field1DBuild`]; [`Field1DBuild::run`] dispatches to the
//!     tridiagonal solver when present, else delegates to `FieldSimulation::run`.
//!   * `null as any` placeholder for a station's census back-ref -> a shared
//!     placeholder census that `FieldSimulation::new` rewires.
//!   * `Float64Array` -> `Vec<f64>`; string scheme unions -> enums + `match`;
//!     `throw new Error` (length-mismatch / unsupported scheme) -> `panic!`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::rc::Rc;

use std::cell::RefCell;

use crate::des::general::expr::{evaluate, parse, Env, Expr};
use crate::des::general::field_station::{
    FieldSimulationOptions, FieldSimulationResult, FieldTrace, FieldUpdater, Position,
};
use crate::des::general::time_stepped_station::TimeSteppedStation;
// Re-export the simulation types so callers can inspect / extend (TS re-export
// of `FieldSimulation` / `FieldStation` / `Census`). The `pub use` also brings
// them into this module's scope for the builders below.
pub use crate::des::general::field_station::{Census, FieldSimulation, FieldStation};

/// A compiled scalar function over positional arguments.
type CompiledFn = Box<dyn Fn(&[f64]) -> f64>;

/// Compile `expr` into a callable over the positional argument list `args`
/// (TS `toFunction(expr, args)`). The returned closure maps each positional
/// value onto the corresponding named variable and evaluates the expression.
pub fn to_function(expr: &Expr, args: &[String]) -> CompiledFn {
    let expr = expr.clone();
    let args: Vec<String> = args.to_vec();
    Box::new(move |vals: &[f64]| {
        let mut env: Env = HashMap::with_capacity(args.len());
        for (i, name) in args.iter().enumerate() {
            env.insert(name.clone(), vals.get(i).copied().unwrap_or(0.0));
        }
        evaluate(&expr, &env)
    })
}

/// Invoke a compiled function. Accepts anything that derefs to a
/// `CompiledFn` (e.g. `&CompiledFn` or `&Rc<CompiledFn>`), since `Rc<F>` does
/// not itself implement `Fn`.
#[inline]
fn eval_fn(f: &CompiledFn, args: &[f64]) -> f64 {
    f(args)
}

/// Build the placeholder census that `FieldStation::new` requires before
/// `FieldSimulation::new` rewires the real one (the TS `null as any`).
fn placeholder_census() -> Rc<RefCell<Census>> {
    Rc::new(RefCell::new(Census::new("placeholder", Vec::new())))
}

// -----------------------------------------------------------------------------
// ODE system as a station network.
// -----------------------------------------------------------------------------

/// Integration scheme for [`build_ode_system`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeScheme {
    Euler,
    Rk2,
    Rk4,
}

/// Specification of an ODE system. Variables in scope inside each RHS
/// expression are `t` and every name.
pub struct OdeSystemSpec {
    pub names: Vec<String>,
    /// Expression strings, one per name; vars are `[t, names...]`.
    pub rhs: Vec<String>,
    pub y0: Vec<f64>,
    pub scheme: OdeScheme,
    /// Optional precomputed RHS expressions (skips parse).
    pub rhs_exprs: Option<Vec<Expr>>,
}

/// Evaluate the RHS vector `f(t, y)` (TS local `fAt`).
fn f_at(fns: &[CompiledFn], n: usize, t: f64, y: &[f64]) -> Vec<f64> {
    let mut arg: Vec<f64> = Vec::with_capacity(y.len() + 1);
    arg.push(t);
    arg.extend_from_slice(y);
    let mut vals = vec![0.0; n];
    for (i, val) in vals.iter_mut().enumerate() {
        *val = fns[i](&arg);
    }
    vals
}

/// Build a `FieldSimulation` that integrates the ODE system (TS
/// `buildODESystem`).
pub fn build_ode_system(spec: &OdeSystemSpec) -> FieldSimulation {
    let n = spec.names.len();
    if spec.rhs.len() != n || spec.y0.len() != n {
        panic!(
            "build_ode_system: names/rhs/y0 lengths must match (got {}, {}, {})",
            n,
            spec.rhs.len(),
            spec.y0.len()
        );
    }
    let exprs: Vec<Expr> = match &spec.rhs_exprs {
        Some(es) => es.clone(),
        None => spec.rhs.iter().map(|s| parse(s)).collect(),
    };
    // Compile each RHS to a function over (t, y_1, …, y_n).
    let mut args: Vec<String> = Vec::with_capacity(n + 1);
    args.push("t".to_string());
    args.extend(spec.names.iter().cloned());
    let fns: Rc<Vec<CompiledFn>> = Rc::new(exprs.iter().map(|e| to_function(e, &args)).collect());

    let mut stations: Vec<Rc<RefCell<FieldStation>>> = Vec::with_capacity(n);
    for i in 0..n {
        let fns_i = fns.clone();
        let idx = i;
        let updater: FieldUpdater = match spec.scheme {
            OdeScheme::Euler => Box::new(move |_prev, cur, _self, dt, t| {
                cur[idx] + dt * f_at(&fns_i, n, t, cur)[idx]
            }),
            OdeScheme::Rk2 => Box::new(move |_prev, cur, _self, dt, t| {
                let k1 = f_at(&fns_i, n, t, cur);
                let mut y_mid = vec![0.0; n];
                for j in 0..n {
                    y_mid[j] = cur[j] + dt * k1[j];
                }
                let k2 = f_at(&fns_i, n, t + dt, &y_mid);
                cur[idx] + dt / 2.0 * (k1[idx] + k2[idx])
            }),
            OdeScheme::Rk4 => Box::new(move |_prev, cur, _self, dt, t| {
                let k1 = f_at(&fns_i, n, t, cur);
                let mut yk2 = vec![0.0; n];
                for j in 0..n {
                    yk2[j] = cur[j] + dt / 2.0 * k1[j];
                }
                let k2 = f_at(&fns_i, n, t + dt / 2.0, &yk2);
                let mut yk3 = vec![0.0; n];
                for j in 0..n {
                    yk3[j] = cur[j] + dt / 2.0 * k2[j];
                }
                let k3 = f_at(&fns_i, n, t + dt / 2.0, &yk3);
                let mut yk4 = vec![0.0; n];
                for j in 0..n {
                    yk4[j] = cur[j] + dt * k3[j];
                }
                let k4 = f_at(&fns_i, n, t + dt, &yk4);
                cur[idx] + dt / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx])
            }),
        };
        let fs = FieldStation::new(
            spec.names[i].clone(),
            spec.y0[i],
            updater,
            placeholder_census(),
        );
        stations.push(Rc::new(RefCell::new(fs)));
    }
    FieldSimulation::new(stations, FieldSimulationOptions::default())
}

// -----------------------------------------------------------------------------
// 1-D PDE on a uniform grid.
// -----------------------------------------------------------------------------

/// Discretisation scheme for [`build_field1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field1DScheme {
    Ftcs,
    Btcs,
    Leapfrog,
    Upwind,
}

/// Equation family for [`build_field1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field1DFamily {
    Heat,
    Wave,
    Advection,
}

/// Boundary condition (TS `type BC = number | 'neumann'`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bc {
    /// Dirichlet: pin the boundary to this value.
    Value(f64),
    /// Neumann zero-flux (ghost-cell mirror).
    Neumann,
}

impl Bc {
    fn is_value(&self) -> bool {
        matches!(self, Bc::Value(_))
    }
}

/// Specification of a 1-D PDE field (TS `interface Field1DSpec`).
pub struct Field1DSpec {
    pub n: usize,
    pub x_lo: f64,
    pub x_hi: f64,
    /// Initial condition `u(x, t=0)` as expression string in `x`.
    pub init_expr: String,
    pub family: Field1DFamily,
    /// `heat`: diffusion coefficient α(x) expression in `x`.
    pub alpha_expr: Option<String>,
    /// Optional source term `s(t, x)` for heat (defaults to 0).
    pub source_expr: Option<String>,
    /// `wave`: wave speed c(x) expression in `x`.
    pub c_expr: Option<String>,
    /// `advection`: velocity a(x) expression in `x`.
    pub a_expr: Option<String>,
    pub bc_left: Bc,
    pub bc_right: Bc,
    pub scheme: Field1DScheme,
}

/// BTCS implicit-solve strategy, carried out-of-line because it couples all
/// stations through a tridiagonal system (TS overrode `sim.run`).
#[derive(Clone)]
struct Btcs {
    a: f64,
    dx: f64,
    xs: Vec<f64>,
    source: Option<Rc<CompiledFn>>,
    bc_left: Bc,
    bc_right: Bc,
}

/// Result of [`build_field1d`] (TS `interface Field1DBuild`), carrying the wired
/// simulation plus the grid metadata.
pub struct Field1DBuild {
    pub sim: FieldSimulation,
    pub xs: Vec<f64>,
    pub dx: f64,
    btcs: Option<Btcs>,
}

impl Field1DBuild {
    /// Advance the field from `t0` to `t1` in steps of `dt`. Dispatches to the
    /// BTCS tridiagonal solver when the spec selected it, else delegates to the
    /// generic per-station tick loop.
    pub fn run(&mut self, t0: f64, t1: f64, dt: f64) -> FieldSimulationResult {
        let bt = self.btcs.clone();
        match bt {
            Some(bt) => run_btcs(&mut self.sim, &bt, t0, t1, dt),
            None => self.sim.run(t0, t1, dt),
        }
    }
}

/// BTCS implicit-Euler heat solve (TS `sim.run` override). Always records the
/// trace, matching the TS override.
fn run_btcs(
    sim: &mut FieldSimulation,
    bt: &Btcs,
    t0: f64,
    t1: f64,
    dt: f64,
) -> FieldSimulationResult {
    let n = sim.fields.len();
    let r = bt.a * dt / (bt.dx * bt.dx);
    let mut t: Vec<f64> = vec![t0];
    let mut values: Vec<Vec<f64>> = vec![sim.census.borrow().snap.clone()];
    let mut tn = t0;
    let mut tick = 0usize;
    let a_sub = -r;
    let a_diag = 1.0 + 2.0 * r;
    let a_sup = -r;
    while tn + 0.5 * dt < t1 {
        sim.census.borrow_mut().run_time_step(dt, tn);
        let snap = sim.census.borrow().snap.clone();
        // RHS vector with source contribution.
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            let s = bt
                .source
                .as_ref()
                .map(|f| eval_fn(f, &[tn + dt, bt.xs[i]]))
                .unwrap_or(0.0);
            rhs[i] = snap[i] + dt * s;
        }
        if let Bc::Value(v) = bt.bc_left {
            rhs[0] = v;
        }
        if let Bc::Value(v) = bt.bc_right {
            rhs[n - 1] = v;
        }
        // Per-row tridiagonal coefficients with Dirichlet overrides.
        let mut sub = vec![a_sub; n];
        let mut dg = vec![a_diag; n];
        let mut sup = vec![a_sup; n];
        if bt.bc_left.is_value() {
            sub[0] = 0.0;
            dg[0] = 1.0;
            sup[0] = 0.0;
        }
        if bt.bc_right.is_value() {
            sub[n - 1] = 0.0;
            dg[n - 1] = 1.0;
            sup[n - 1] = 0.0;
        }
        let u = thomas(&sub, &dg, &sup, &rhs);
        for i in 0..n {
            sim.fields[i].borrow_mut().value = u[i];
        }
        tn += dt;
        tick += 1;
        t.push(tn);
        values.push(u.clone());
    }
    let final_values: Vec<f64> = sim.fields.iter().map(|f| f.borrow().value).collect();
    let out = FieldSimulationResult {
        trace: FieldTrace { t, values },
        final_values,
        ticks: tick,
    };
    out
}

/// Build a 1-D PDE field simulation (TS `buildField1D`).
pub fn build_field1d(spec: &Field1DSpec) -> Field1DBuild {
    let n = spec.n;
    let dx = (spec.x_hi - spec.x_lo) / (n as f64 - 1.0);
    let xs: Vec<f64> = (0..n).map(|i| spec.x_lo + i as f64 * dx).collect();
    let init_fn = to_function(&parse(&spec.init_expr), &["x".to_string()]);
    let u0: Vec<f64> = xs.iter().map(|&x| init_fn(&[x])).collect();

    let alpha_fn = spec
        .alpha_expr
        .as_ref()
        .map(|e| to_function(&parse(e), &["x".to_string()]));
    let source_fn = spec
        .source_expr
        .as_ref()
        .map(|e| Rc::new(to_function(&parse(e), &["t".to_string(), "x".to_string()])));
    let c_fn = spec
        .c_expr
        .as_ref()
        .map(|e| to_function(&parse(e), &["x".to_string()]));
    let a_fn = spec
        .a_expr
        .as_ref()
        .map(|e| to_function(&parse(e), &["x".to_string()]));

    let bc_left = spec.bc_left;
    let bc_right = spec.bc_right;
    let mut stations: Vec<Rc<RefCell<FieldStation>>> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = xs[i];
        let is_left = i == 0;
        let is_right = i == n - 1;

        let updater: FieldUpdater = match (spec.family, spec.scheme) {
            (Field1DFamily::Heat, Field1DScheme::Ftcs) => {
                let a = alpha_fn.as_ref().map(|f| f(&[xi])).unwrap_or(0.0);
                let src = source_fn.clone();
                Box::new(move |_prev, cur, slf, dt, t| {
                    if (is_left && bc_left.is_value()) || (is_right && bc_right.is_value()) {
                        return boundary_value(is_left, bc_left, bc_right);
                    }
                    let lap = (read_right(cur, i, n, bc_right) - 2.0 * cur[slf]
                        + read_left(cur, i, bc_left))
                        / (dx * dx);
                    let s = src.as_ref().map(|f| eval_fn(f, &[t, xi])).unwrap_or(0.0);
                    cur[slf] + dt * (a * lap + s)
                })
            }
            (Field1DFamily::Wave, Field1DScheme::Leapfrog) => {
                let c = c_fn.as_ref().map(|f| f(&[xi])).unwrap_or(1.0);
                Box::new(move |prev, cur, slf, dt, _t| {
                    if (is_left && bc_left.is_value()) || (is_right && bc_right.is_value()) {
                        return boundary_value(is_left, bc_left, bc_right);
                    }
                    let lap = (read_right(cur, i, n, bc_right) - 2.0 * cur[slf]
                        + read_left(cur, i, bc_left))
                        / (dx * dx);
                    2.0 * cur[slf] - prev[slf] + (c * c) * (dt * dt) * lap
                })
            }
            (Field1DFamily::Advection, Field1DScheme::Upwind) => {
                let a = a_fn.as_ref().map(|f| f(&[xi])).unwrap_or(1.0);
                Box::new(move |_prev, cur, slf, dt, _t| {
                    if is_left && bc_left.is_value() {
                        return boundary_value(true, bc_left, bc_right);
                    }
                    if is_right && bc_right.is_value() {
                        return boundary_value(false, bc_left, bc_right);
                    }
                    if a >= 0.0 {
                        cur[slf] - a * dt / dx * (cur[slf] - read_left(cur, i, bc_left))
                    } else {
                        cur[slf] - a * dt / dx * (read_right(cur, i, n, bc_right) - cur[slf])
                    }
                })
            }
            (Field1DFamily::Heat, Field1DScheme::Btcs) => {
                // BTCS handled out-of-line; mark each station with a no-op.
                Box::new(move |_prev, cur, slf, _dt, _t| cur[slf])
            }
            _ => panic!(
                "Field1D: scheme {:?} not supported for family {:?}",
                spec.scheme, spec.family
            ),
        };
        let mut fs = FieldStation::new(format!("x_{i}"), u0[i], updater, placeholder_census());
        fs.position = Some(Position::Scalar(xi));
        stations.push(Rc::new(RefCell::new(fs)));
    }
    let sim = FieldSimulation::new(stations, FieldSimulationOptions::default());

    let btcs = if spec.family == Field1DFamily::Heat && spec.scheme == Field1DScheme::Btcs {
        // Homogeneous α only for the BTCS demo (TS used α(0)).
        let a = alpha_fn.as_ref().map(|f| f(&[0.0])).unwrap_or(1.0);
        Some(Btcs {
            a,
            dx,
            xs: xs.clone(),
            source: source_fn.clone(),
            bc_left,
            bc_right,
        })
    } else {
        None
    };

    Field1DBuild { sim, xs, dx, btcs }
}

/// Read the left neighbour with boundary handling (TS local `readLeft`).
fn read_left(cur: &[f64], i: usize, bc_left: Bc) -> f64 {
    if i == 0 {
        match bc_left {
            Bc::Neumann => cur[0],
            Bc::Value(v) => v,
        }
    } else {
        cur[i - 1]
    }
}

/// Read the right neighbour with boundary handling (TS local `readRight`).
fn read_right(cur: &[f64], i: usize, n: usize, bc_right: Bc) -> f64 {
    if i == n - 1 {
        match bc_right {
            Bc::Neumann => cur[n - 1],
            Bc::Value(v) => v,
        }
    } else {
        cur[i + 1]
    }
}

/// Pinned Dirichlet boundary value for the relevant side.
fn boundary_value(is_left: bool, bc_left: Bc, bc_right: Bc) -> f64 {
    let bc = if is_left { bc_left } else { bc_right };
    match bc {
        Bc::Value(v) => v,
        Bc::Neumann => 0.0,
    }
}

/// Thomas algorithm for tridiagonal systems `A·x = d` with subdiagonal `a`,
/// diagonal `b`, superdiagonal `c` (TS `thomas`). `a[0]` and `c[n-1]` are
/// unused. O(n).
pub fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / m;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

// -----------------------------------------------------------------------------
// 2-D Poisson / Laplace on an Nx × Ny grid:  ∇²u = −ρ(x, y).
// -----------------------------------------------------------------------------

/// Iterative relaxation scheme for [`solve_poisson2d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field2DScheme {
    Jacobi,
    GaussSeidel,
    Sor,
}

/// Specification of a 2-D Poisson problem (TS `interface Poisson2DSpec`).
pub struct Poisson2DSpec {
    pub nx: usize,
    pub ny: usize,
    pub x_lo: f64,
    pub x_hi: f64,
    pub y_lo: f64,
    pub y_hi: f64,
    /// ρ(x, y) expression in `x` and `y`.
    pub rho_expr: String,
    /// `u(x, y, t=0)` initial guess (defaults to zero).
    pub init_expr: Option<String>,
    /// Dirichlet boundary `u_b(x, y)` on all four edges; default 0.
    pub bc_expr: Option<String>,
    pub scheme: Field2DScheme,
    pub omega: Option<f64>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

/// Result of [`solve_poisson2d`] (TS `interface Poisson2DResult`).
#[derive(Clone, Debug)]
pub struct Poisson2DResult {
    /// Length `nx*ny`, row-major `j*nx + i`.
    pub u: Vec<f64>,
    pub iterations: usize,
    pub final_delta: f64,
    pub residual_history: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

/// Solve the 2-D Poisson equation via iterative relaxation (TS
/// `solvePoisson2D`).
pub fn solve_poisson2d(spec: &Poisson2DSpec) -> Poisson2DResult {
    let nx = spec.nx;
    let ny = spec.ny;
    let dx = (spec.x_hi - spec.x_lo) / (nx as f64 - 1.0);
    let dy = (spec.y_hi - spec.y_lo) / (ny as f64 - 1.0);
    let xs: Vec<f64> = (0..nx).map(|i| spec.x_lo + i as f64 * dx).collect();
    let ys: Vec<f64> = (0..ny).map(|j| spec.y_lo + j as f64 * dy).collect();
    let rho_fn = to_function(&parse(&spec.rho_expr), &["x".to_string(), "y".to_string()]);
    let bc_src = spec.bc_expr.clone().unwrap_or_else(|| "0".to_string());
    let bc_fn = to_function(&parse(&bc_src), &["x".to_string(), "y".to_string()]);
    let init_src = spec.init_expr.clone().unwrap_or_else(|| "0".to_string());
    let init_fn = to_function(&parse(&init_src), &["x".to_string(), "y".to_string()]);

    let idx = |i: usize, j: usize| j * nx + i;
    let mut u = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let on_boundary = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
            u[idx(i, j)] = if on_boundary {
                bc_fn(&[xs[i], ys[j]])
            } else {
                init_fn(&[xs[i], ys[j]])
            };
        }
    }

    let omega = spec.omega.unwrap_or(1.5);
    let max_iter = spec.max_iter.unwrap_or(5000);
    let tol = spec.tol.unwrap_or(1e-8);
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (dx2 + dy2);
    let mut residual_history: Vec<f64> = Vec::new();
    let mut iter = 0usize;
    let mut final_delta = f64::INFINITY;
    let mut u_old = vec![0.0; nx * ny];
    while iter < max_iter {
        u_old.copy_from_slice(&u);
        let mut max_delta = 0.0f64;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = idx(i, j);
                let rho = rho_fn(&[xs[i], ys[j]]);
                let jacobi = spec.scheme == Field2DScheme::Jacobi;
                let u_e = if jacobi {
                    u_old[idx(i + 1, j)]
                } else {
                    u[idx(i + 1, j)]
                };
                let u_w = if jacobi {
                    u_old[idx(i - 1, j)]
                } else {
                    u[idx(i - 1, j)]
                };
                let u_n = if jacobi {
                    u_old[idx(i, j + 1)]
                } else {
                    u[idx(i, j + 1)]
                };
                let u_s = if jacobi {
                    u_old[idx(i, j - 1)]
                } else {
                    u[idx(i, j - 1)]
                };
                let gs = (dy2 * (u_e + u_w) + dx2 * (u_n + u_s) + dx2 * dy2 * rho) / denom;
                let next = if spec.scheme == Field2DScheme::Sor {
                    (1.0 - omega) * u[k] + omega * gs
                } else {
                    gs
                };
                let delta = (next - u[k]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                u[k] = next;
            }
        }
        iter += 1;
        final_delta = max_delta;
        residual_history.push(max_delta);
        if max_delta < tol {
            break;
        }
    }
    Poisson2DResult {
        u,
        iterations: iter,
        final_delta,
        residual_history,
        nx,
        ny,
        dx,
        dy,
        xs,
        ys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ode_euler_exponential_decay() {
        // y' = -y, explicit Euler.
        let spec = OdeSystemSpec {
            names: vec!["y".to_string()],
            rhs: vec!["-y".to_string()],
            y0: vec![1.0],
            scheme: OdeScheme::Euler,
            rhs_exprs: None,
        };
        let mut sim = build_ode_system(&spec);
        let res = sim.run(0.0, 1.0, 0.1);
        let expected = 0.9f64.powi(10);
        assert!((res.final_values[0] - expected).abs() < 1e-9);
        assert_eq!(res.ticks, 10);
    }

    #[test]
    fn field1d_heat_ftcs_runs() {
        let spec = Field1DSpec {
            n: 5,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: "sin(3.14159265*x)".to_string(),
            family: Field1DFamily::Heat,
            alpha_expr: Some("0.1".to_string()),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Ftcs,
        };
        let mut build = build_field1d(&spec);
        let res = build.run(0.0, 0.05, 0.01);
        assert_eq!(build.xs.len(), 5);
        assert!(res.ticks >= 1);
        // Dirichlet boundaries stay pinned.
        assert_eq!(res.final_values[0], 0.0);
        assert_eq!(res.final_values[4], 0.0);
    }

    #[test]
    fn field1d_btcs_dispatches_to_implicit_solver() {
        let spec = Field1DSpec {
            n: 6,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: "1".to_string(),
            family: Field1DFamily::Heat,
            alpha_expr: Some("0.2".to_string()),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme: Field1DScheme::Btcs,
        };
        let mut build = build_field1d(&spec);
        let res = build.run(0.0, 0.1, 0.02);
        assert!(res.ticks >= 1);
        assert_eq!(res.final_values[0], 0.0);
        assert_eq!(res.final_values[5], 0.0);
    }

    #[test]
    fn poisson2d_jacobi_converges_on_laplace() {
        // ρ = 0, zero boundary -> u ≡ 0 (already satisfied), few iterations.
        let spec = Poisson2DSpec {
            nx: 5,
            ny: 5,
            x_lo: 0.0,
            x_hi: 1.0,
            y_lo: 0.0,
            y_hi: 1.0,
            rho_expr: "0".to_string(),
            init_expr: None,
            bc_expr: None,
            scheme: Field2DScheme::Jacobi,
            omega: None,
            max_iter: Some(100),
            tol: Some(1e-10),
        };
        let res = solve_poisson2d(&spec);
        assert!(res.final_delta < 1e-9);
        assert_eq!(res.u.len(), 25);
    }
}
