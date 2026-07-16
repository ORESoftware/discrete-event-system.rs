//! Small dense convex quadratic programming.
//!
//! Solves
//!
//! ```text
//! min  1/2 x^T Q x + c^T x
//! s.t. A_ub x <= b_ub
//!      A_eq x  = b_eq
//!      lb <= x <= ub
//! ```
//!
//! The implementation is intentionally compact: for small dense convex models it
//! enumerates active inequality/bound sets, solves the corresponding KKT system,
//! and keeps the feasible stationary point with the best objective. This is not
//! a large-scale production QP engine, but it gives the crate a real constrained
//! QP primitive and a strong internal baseline for external-solver validation.

use crate::des::shared::linalg::{LinearSystem, Matrix, VecOps, Vector};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuadraticProgram {
    pub q: Matrix,
    pub c: Vector,
    pub a_ub: Option<Matrix>,
    pub b_ub: Option<Vector>,
    pub a_eq: Option<Matrix>,
    pub b_eq: Option<Vector>,
    pub lb: Option<Vec<Option<f64>>>,
    pub ub: Option<Vec<Option<f64>>>,
    pub var_names: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixedIntegerQuadraticProgram {
    pub qp: QuadraticProgram,
    pub integer_vars: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPStatus {
    Optimal,
    Infeasible,
    NumericalError,
}

impl QPStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            QPStatus::Optimal => "optimal",
            QPStatus::Infeasible => "infeasible",
            QPStatus::NumericalError => "numerical-error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QPSolution {
    pub status: QPStatus,
    pub x: Vector,
    pub objective: f64,
    pub dual_ub: Vector,
    pub dual_eq: Vector,
    pub dual_lower_bounds: Vector,
    pub dual_upper_bounds: Vector,
    pub reduced_gradient: Vector,
    pub active_ub_rows: Vec<usize>,
    pub active_lower_bounds: Vec<usize>,
    pub active_upper_bounds: Vec<usize>,
    pub iterations: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MIQPSolution {
    pub status: QPStatus,
    pub x: Vector,
    pub objective: f64,
    pub enumerated: usize,
    pub qp_subproblems: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderCone {
    pub a: Matrix,
    pub b: Vector,
    pub c: Vector,
    pub d: f64,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SecondOrderConeProgram {
    pub c: Vector,
    pub a_ub: Option<Matrix>,
    pub b_ub: Option<Vector>,
    pub a_eq: Option<Matrix>,
    pub b_eq: Option<Vector>,
    pub lb: Option<Vec<Option<f64>>>,
    pub ub: Option<Vec<Option<f64>>>,
    pub cones: Vec<SecondOrderCone>,
    pub var_names: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixedIntegerSecondOrderConeProgram {
    pub socp: SecondOrderConeProgram,
    pub integer_vars: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocpStatus {
    Optimal,
    Infeasible,
    NumericalError,
}

impl SocpStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SocpStatus::Optimal => "optimal",
            SocpStatus::Infeasible => "infeasible",
            SocpStatus::NumericalError => "numerical-error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SocpSolution {
    pub status: SocpStatus,
    pub x: Vector,
    pub objective: f64,
    pub iterations: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticConstraint {
    pub q: Matrix,
    pub c: Vector,
    pub rhs: f64,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuadraticallyConstrainedProgram {
    pub q: Matrix,
    pub c: Vector,
    pub a_ub: Option<Matrix>,
    pub b_ub: Option<Vector>,
    pub a_eq: Option<Matrix>,
    pub b_eq: Option<Vector>,
    pub lb: Option<Vec<Option<f64>>>,
    pub ub: Option<Vec<Option<f64>>>,
    pub quadratic_constraints: Vec<QuadraticConstraint>,
    pub var_names: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixedIntegerQuadraticallyConstrainedProgram {
    pub qcp: QuadraticallyConstrainedProgram,
    pub integer_vars: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QcpStatus {
    Optimal,
    Infeasible,
    NumericalError,
}

impl QcpStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            QcpStatus::Optimal => "optimal",
            QcpStatus::Infeasible => "infeasible",
            QcpStatus::NumericalError => "numerical-error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QcpSolution {
    pub status: QcpStatus,
    pub x: Vector,
    pub objective: f64,
    pub iterations: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct QPOptions {
    pub tol: f64,
    pub max_active_sets: usize,
}

impl Default for QPOptions {
    fn default() -> Self {
        QPOptions {
            tol: 1e-8,
            max_active_sets: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MIQPOptions {
    pub qp_options: QPOptions,
    pub max_enumerations: usize,
}

impl Default for MIQPOptions {
    fn default() -> Self {
        MIQPOptions {
            qp_options: QPOptions::default(),
            max_enumerations: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QcpOptions {
    pub tol: f64,
    pub max_iters: usize,
    pub shrink: f64,
}

impl Default for QcpOptions {
    fn default() -> Self {
        QcpOptions {
            tol: 1e-7,
            max_iters: 20_000,
            shrink: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SocpOptions {
    pub tol: f64,
    pub max_iters: usize,
    pub shrink: f64,
}

impl Default for SocpOptions {
    fn default() -> Self {
        SocpOptions {
            tol: 1e-7,
            max_iters: 20_000,
            shrink: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveKind {
    Inequality(usize),
    Lower(usize),
    Upper(usize),
}

#[derive(Clone, Debug)]
struct KktSolution {
    x: Vector,
    multipliers: Vector,
}

fn validate_qp(p: &QuadraticProgram) {
    let n = p.c.len();
    if p.q.len() != n {
        panic!("qp: Q row count {} != c length {n}", p.q.len());
    }
    for (i, row) in p.q.iter().enumerate() {
        if row.len() != n {
            panic!("qp: Q row {i} length {} != {n}", row.len());
        }
        for j in 0..n {
            if (p.q[i][j] - p.q[j][i]).abs() > 1e-7 {
                panic!("qp: Q must be symmetric");
            }
        }
    }
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("qp: A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("qp: A_eq / b_eq length mismatch");
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            panic!("qp: A_ub row {i} length {} != {n}", row.len());
        }
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            panic!("qp: A_eq row {i} length {} != {n}", row.len());
        }
    }
    if let Some(lb) = &p.lb {
        if lb.len() != n {
            panic!("qp: lb length {} != {n}", lb.len());
        }
    }
    if let Some(ub) = &p.ub {
        if ub.len() != n {
            panic!("qp: ub length {} != {n}", ub.len());
        }
    }
}

fn objective(p: &QuadraticProgram, x: &[f64]) -> f64 {
    let qx = mat_vec(&p.q, x);
    0.5 * VecOps::dot(x, &qx) + VecOps::dot(&p.c, x)
}

fn qp_gradient(p: &QuadraticProgram, x: &[f64]) -> Vector {
    let qx = mat_vec(&p.q, x);
    qx.iter().zip(&p.c).map(|(qi, ci)| qi + ci).collect()
}

fn mat_vec(a: &Matrix, x: &[f64]) -> Vector {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(ai, xi)| ai * xi).sum())
        .collect()
}

fn dot(row: &[f64], x: &[f64]) -> f64 {
    row.iter().zip(x).map(|(a, xi)| a * xi).sum()
}

/// Effective per-variable bounds. **Default convention:** when `p.lb` is
/// `None`, every variable takes a lower bound of `0` (the standard LP/QP
/// nonnegativity default); when `p.ub` is `None`, variables are unbounded
/// above. A model needing a *free* (possibly-negative, unbounded) variable must
/// therefore pass an explicit `lb`/`ub` of `Some(vec![None; n])` — relying on
/// the default silently restricts the feasible set to `x >= 0`, which is a
/// common source of surprising "wrong optimum" results in QP-style fits (e.g.
/// an MPC acceleration that can be negative).
fn bounds(p: &QuadraticProgram) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = p.c.len();
    (
        p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]),
        p.ub.clone().unwrap_or_else(|| vec![None; n]),
    )
}

fn candidate_constraints(p: &QuadraticProgram) -> Vec<ActiveKind> {
    let (lb, ub) = bounds(p);
    let mut out = Vec::new();
    for i in 0..p.a_ub.as_ref().map(|a| a.len()).unwrap_or(0) {
        out.push(ActiveKind::Inequality(i));
    }
    for i in 0..p.c.len() {
        if lb[i].is_some() {
            out.push(ActiveKind::Lower(i));
        }
        if ub[i].is_some() {
            out.push(ActiveKind::Upper(i));
        }
    }
    out
}

fn active_row_rhs(p: &QuadraticProgram, kind: ActiveKind) -> (Vector, f64) {
    let n = p.c.len();
    let (lb, ub) = bounds(p);
    match kind {
        ActiveKind::Inequality(i) => (
            p.a_ub.as_ref().expect("A_ub exists")[i].clone(),
            p.b_ub.as_ref().expect("b_ub exists")[i],
        ),
        ActiveKind::Lower(i) => {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            (row, lb[i].expect("lower bound exists"))
        }
        ActiveKind::Upper(i) => {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            (row, ub[i].expect("upper bound exists"))
        }
    }
}

fn solve_kkt(p: &QuadraticProgram, active: &[ActiveKind], tol: f64) -> Option<KktSolution> {
    let n = p.c.len();
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    let m = a_eq.len() + active.len();
    let dim = n + m;
    let mut kkt = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for i in 0..n {
        for j in 0..n {
            kkt[i][j] = p.q[i][j];
        }
        rhs[i] = -p.c[i];
    }

    let mut row_index = 0usize;
    for (row, &rhs_i) in a_eq.iter().zip(b_eq) {
        for j in 0..n {
            kkt[j][n + row_index] = row[j];
            kkt[n + row_index][j] = row[j];
        }
        rhs[n + row_index] = rhs_i;
        row_index += 1;
    }
    for &kind in active {
        let (row, rhs_i) = active_row_rhs(p, kind);
        for j in 0..n {
            kkt[j][n + row_index] = row[j];
            kkt[n + row_index][j] = row[j];
        }
        rhs[n + row_index] = rhs_i;
        row_index += 1;
    }
    LinearSystem::new(&kkt, &rhs, tol)
        .try_solve()
        .map(|v| KktSolution {
            x: v[..n].to_vec(),
            multipliers: v[n..].to_vec(),
        })
}

fn feasible(p: &QuadraticProgram, x: &[f64], tol: f64) -> bool {
    if x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let (lb, ub) = bounds(p);
    for i in 0..x.len() {
        if let Some(l) = lb[i] {
            if x[i] < l - tol {
                return false;
            }
        }
        if let Some(u) = ub[i] {
            if x[i] > u + tol {
                return false;
            }
        }
    }
    if let Some(a_ub) = &p.a_ub {
        let b_ub = p.b_ub.as_ref().expect("b_ub");
        for (row, &rhs) in a_ub.iter().zip(b_ub) {
            if dot(row, x) > rhs + tol {
                return false;
            }
        }
    }
    if let Some(a_eq) = &p.a_eq {
        let b_eq = p.b_eq.as_ref().expect("b_eq");
        for (row, &rhs) in a_eq.iter().zip(b_eq) {
            if (dot(row, x) - rhs).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn decode_active(active: &[ActiveKind]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut active_ub_rows = Vec::new();
    let mut active_lower_bounds = Vec::new();
    let mut active_upper_bounds = Vec::new();
    for &kind in active {
        match kind {
            ActiveKind::Inequality(i) => active_ub_rows.push(i),
            ActiveKind::Lower(i) => active_lower_bounds.push(i),
            ActiveKind::Upper(i) => active_upper_bounds.push(i),
        }
    }
    (active_ub_rows, active_lower_bounds, active_upper_bounds)
}

type QPCertificate = (Vector, Vector, Vector, Vector, Vector);

fn qp_certificate(
    p: &QuadraticProgram,
    x: &[f64],
    active: &[ActiveKind],
    multipliers: &[f64],
    tol: f64,
) -> Option<QPCertificate> {
    let n = p.c.len();
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    if multipliers.len() != a_eq.len() + active.len() {
        return None;
    }

    let dual_eq = multipliers[..a_eq.len()].to_vec();
    let mut dual_ub = vec![0.0; a_ub.len()];
    let mut dual_lower_bounds = vec![0.0; n];
    let mut dual_upper_bounds = vec![0.0; n];
    for (offset, &kind) in active.iter().enumerate() {
        let lambda = multipliers[a_eq.len() + offset];
        match kind {
            ActiveKind::Inequality(row) => {
                if lambda < -tol {
                    return None;
                }
                dual_ub[row] = lambda.max(0.0);
            }
            ActiveKind::Lower(var) => {
                let dual = -lambda;
                if dual < -tol {
                    return None;
                }
                dual_lower_bounds[var] = dual.max(0.0);
            }
            ActiveKind::Upper(var) => {
                if lambda < -tol {
                    return None;
                }
                dual_upper_bounds[var] = lambda.max(0.0);
            }
        }
    }

    let mut reduced_gradient = qp_gradient(p, x);
    for (row, &dual) in a_ub.iter().zip(&dual_ub) {
        if dual == 0.0 {
            continue;
        }
        for j in 0..n {
            reduced_gradient[j] += dual * row[j];
        }
    }
    for (row, &dual) in a_eq.iter().zip(&dual_eq) {
        for j in 0..n {
            reduced_gradient[j] += dual * row[j];
        }
    }
    for j in 0..n {
        let stationarity = reduced_gradient[j] - dual_lower_bounds[j] + dual_upper_bounds[j];
        if stationarity.abs() > 10.0 * tol {
            return None;
        }
    }

    Some((
        dual_ub,
        dual_eq,
        dual_lower_bounds,
        dual_upper_bounds,
        reduced_gradient,
    ))
}

/// A `NumericalError` result with an explanatory message and no primal point —
/// the honest outcome when this engine cannot solve the model as posed
/// (non-finite data, or more constraints than the enumeration can represent).
fn qp_numerical_error(message: &str) -> QPSolution {
    QPSolution {
        status: QPStatus::NumericalError,
        x: Vec::new(),
        objective: f64::NAN,
        dual_ub: Vec::new(),
        dual_eq: Vec::new(),
        dual_lower_bounds: Vec::new(),
        dual_upper_bounds: Vec::new(),
        reduced_gradient: Vec::new(),
        active_ub_rows: Vec::new(),
        active_lower_bounds: Vec::new(),
        active_upper_bounds: Vec::new(),
        iterations: 0,
        solver: "internal-active-set-enumeration".to_string(),
        message: Some(message.to_string()),
    }
}

/// True iff every numeric entry of the model (Q, c, the A/b rows, and any finite
/// bound) is finite. A NaN/∞ leaking in from upstream can never yield a valid
/// optimum; detecting it lets the solver report `NumericalError` instead of
/// silently returning "infeasible" (which a caller could misread as "the model
/// has no solution" rather than "the model is malformed").
fn qp_data_all_finite(p: &QuadraticProgram) -> bool {
    let all_finite = |xs: &[f64]| xs.iter().all(|v| v.is_finite());
    if !all_finite(&p.c) || !p.q.iter().all(|row| all_finite(row)) {
        return false;
    }
    for m in [p.a_ub.as_ref(), p.a_eq.as_ref()].into_iter().flatten() {
        if !m.iter().all(|row| all_finite(row)) {
            return false;
        }
    }
    for v in [p.b_ub.as_ref(), p.b_eq.as_ref()].into_iter().flatten() {
        if !all_finite(v) {
            return false;
        }
    }
    for bnds in [p.lb.as_ref(), p.ub.as_ref()].into_iter().flatten() {
        if bnds.iter().flatten().any(|v| !v.is_finite()) {
            return false;
        }
    }
    true
}

/// Solve a small dense convex QP by active-set enumeration.
pub fn solve_qp_active_set(p: &QuadraticProgram, opts: QPOptions) -> QPSolution {
    validate_qp(p);
    if !opts.tol.is_finite() || opts.tol <= 0.0 {
        return qp_numerical_error("QP tolerance must be finite and greater than zero");
    }
    if opts.max_active_sets == 0 {
        return qp_numerical_error("QP max_active_sets must be greater than zero");
    }
    // Hardening: non-finite problem data can never produce a valid optimum and
    // would otherwise be silently misreported as "infeasible" — fail honestly.
    if !qp_data_all_finite(p) {
        return qp_numerical_error("non-finite problem data (NaN/inf in Q, c, A, b, or a bound)");
    }
    let candidates = candidate_constraints(p);
    let n = p.c.len();
    let Some(total_active_sets) = u32::try_from(candidates.len())
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
    else {
        return qp_numerical_error(
            "too many inequality/bound constraints for active-set enumeration",
        );
    };
    if total_active_sets > opts.max_active_sets {
        return qp_numerical_error(&format!(
            "active-set search requires {total_active_sets} enumerations, exceeding max_active_sets={}",
            opts.max_active_sets
        ));
    }
    let mut best_x = Vec::new();
    let mut best_obj = f64::INFINITY;
    let mut best_active = Vec::new();
    let mut best_certificate: Option<QPCertificate> = None;
    let mut iterations = 0usize;

    for mask in 0usize..total_active_sets {
        iterations += 1;
        let mut active = Vec::new();
        for (i, &kind) in candidates.iter().enumerate() {
            if (mask & (1usize << i)) != 0 {
                active.push(kind);
            }
        }
        if active.len() > n {
            continue;
        }
        let Some(kkt) = solve_kkt(p, &active, opts.tol.max(1e-12)) else {
            continue;
        };
        if !feasible(p, &kkt.x, opts.tol.max(1e-8)) {
            continue;
        }
        let Some(certificate) =
            qp_certificate(p, &kkt.x, &active, &kkt.multipliers, opts.tol.max(1e-8))
        else {
            continue;
        };
        let obj = objective(p, &kkt.x);
        if obj < best_obj - opts.tol {
            best_obj = obj;
            best_x = kkt.x;
            best_active = active;
            best_certificate = Some(certificate);
        }
    }

    if best_x.is_empty() {
        return QPSolution {
            status: QPStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            dual_ub: Vec::new(),
            dual_eq: Vec::new(),
            dual_lower_bounds: Vec::new(),
            dual_upper_bounds: Vec::new(),
            reduced_gradient: Vec::new(),
            active_ub_rows: Vec::new(),
            active_lower_bounds: Vec::new(),
            active_upper_bounds: Vec::new(),
            iterations,
            solver: "internal-active-set-enumeration".to_string(),
            message: Some("no feasible KKT candidate found".to_string()),
        };
    }
    let (active_ub_rows, active_lower_bounds, active_upper_bounds) = decode_active(&best_active);
    let (dual_ub, dual_eq, dual_lower_bounds, dual_upper_bounds, reduced_gradient) =
        best_certificate.expect("optimal QP candidate has KKT certificate");
    QPSolution {
        status: QPStatus::Optimal,
        x: best_x,
        objective: best_obj,
        dual_ub,
        dual_eq,
        dual_lower_bounds,
        dual_upper_bounds,
        reduced_gradient,
        active_ub_rows,
        active_lower_bounds,
        active_upper_bounds,
        iterations,
        solver: "internal-active-set-enumeration".to_string(),
        message: Some("convex QP active-set enumeration".to_string()),
    }
}

fn validate_miqp(p: &MixedIntegerQuadraticProgram) {
    validate_qp(&p.qp);
    let n = p.qp.c.len();
    if p.integer_vars.len() != n {
        panic!(
            "miqp: integer_vars length {} != variable count {n}",
            p.integer_vars.len()
        );
    }
    let (lb, ub) = bounds(&p.qp);
    for i in 0..n {
        if !p.integer_vars[i] {
            continue;
        }
        let Some(lower) = lb[i] else {
            panic!("miqp: integer variable {i} needs a finite lower bound");
        };
        let Some(upper) = ub[i] else {
            panic!("miqp: integer variable {i} needs a finite upper bound");
        };
        if lower.ceil() > upper.floor() {
            panic!("miqp: integer variable {i} has no integer value in its bounds");
        }
    }
}

fn fixed_integer_subproblem(
    p: &MixedIntegerQuadraticProgram,
    assignment: &[(usize, f64)],
) -> QuadraticProgram {
    let n = p.qp.c.len();
    let mut sub = p.qp.clone();
    let mut a_eq = sub.a_eq.clone().unwrap_or_default();
    let mut b_eq = sub.b_eq.clone().unwrap_or_default();
    for &(var, value) in assignment {
        let mut row = vec![0.0; n];
        row[var] = 1.0;
        a_eq.push(row);
        b_eq.push(value);
    }
    sub.a_eq = Some(a_eq);
    sub.b_eq = Some(b_eq);
    sub
}

/// Solve a bounded small-model mixed-integer convex QP by enumerating integer
/// assignments and solving each remaining continuous QP with the active-set
/// engine. This gives the crate a native MIQP modelling surface comparable to
/// commercial solvers for modest models; scale remains limited by enumeration.
pub fn solve_miqp_enumeration(p: &MixedIntegerQuadraticProgram, opts: MIQPOptions) -> MIQPSolution {
    validate_miqp(p);
    let integer_indices: Vec<usize> = p
        .integer_vars
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_integer)| is_integer.then_some(idx))
        .collect();
    if integer_indices.is_empty() {
        let sol = solve_qp_active_set(&p.qp, opts.qp_options);
        return MIQPSolution {
            status: sol.status,
            x: sol.x,
            objective: sol.objective,
            enumerated: 1,
            qp_subproblems: 1,
            solver: "internal-miqp-enumeration".to_string(),
            message: Some("no integer variables; delegated to continuous QP".to_string()),
        };
    }

    let (lb, ub) = bounds(&p.qp);
    let mut domains = Vec::with_capacity(integer_indices.len());
    for &idx in &integer_indices {
        let lower = lb[idx].expect("validated finite lower bound").ceil() as i64;
        let upper = ub[idx].expect("validated finite upper bound").floor() as i64;
        domains.push((lower, upper));
    }

    let mut current = Vec::with_capacity(integer_indices.len());
    let mut best_x = Vec::new();
    let mut best_obj = f64::INFINITY;
    let mut enumerated = 0usize;
    let mut qp_subproblems = 0usize;
    let mut hit_limit = false;

    fn dfs(
        depth: usize,
        integer_indices: &[usize],
        domains: &[(i64, i64)],
        current: &mut Vec<(usize, f64)>,
        p: &MixedIntegerQuadraticProgram,
        opts: MIQPOptions,
        enumerated: &mut usize,
        qp_subproblems: &mut usize,
        best_x: &mut Vector,
        best_obj: &mut f64,
        hit_limit: &mut bool,
    ) {
        if *hit_limit {
            return;
        }
        if depth == integer_indices.len() {
            *enumerated += 1;
            if *enumerated > opts.max_enumerations {
                *hit_limit = true;
                return;
            }
            let sub = fixed_integer_subproblem(p, current);
            *qp_subproblems += 1;
            let sol = solve_qp_active_set(&sub, opts.qp_options);
            if sol.status == QPStatus::Optimal && sol.objective < *best_obj - opts.qp_options.tol {
                *best_obj = sol.objective;
                *best_x = sol.x;
            }
            return;
        }

        let var = integer_indices[depth];
        let (lower, upper) = domains[depth];
        for value in lower..=upper {
            current.push((var, value as f64));
            dfs(
                depth + 1,
                integer_indices,
                domains,
                current,
                p,
                opts,
                enumerated,
                qp_subproblems,
                best_x,
                best_obj,
                hit_limit,
            );
            current.pop();
            if *hit_limit {
                return;
            }
        }
    }

    dfs(
        0,
        &integer_indices,
        &domains,
        &mut current,
        p,
        opts,
        &mut enumerated,
        &mut qp_subproblems,
        &mut best_x,
        &mut best_obj,
        &mut hit_limit,
    );

    if hit_limit {
        return MIQPSolution {
            status: QPStatus::NumericalError,
            x: best_x,
            objective: best_obj,
            enumerated,
            qp_subproblems,
            solver: "internal-miqp-enumeration".to_string(),
            message: Some("MIQP enumeration limit reached".to_string()),
        };
    }
    if best_x.is_empty() {
        return MIQPSolution {
            status: QPStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            enumerated,
            qp_subproblems,
            solver: "internal-miqp-enumeration".to_string(),
            message: Some("no feasible integer assignment found".to_string()),
        };
    }
    MIQPSolution {
        status: QPStatus::Optimal,
        x: best_x,
        objective: best_obj,
        enumerated,
        qp_subproblems,
        solver: "internal-miqp-enumeration".to_string(),
        message: Some("bounded MIQP enumeration over integer variables".to_string()),
    }
}

fn validate_square_symmetric(name: &str, q: &Matrix, n: usize) {
    if q.len() != n {
        panic!("{name}: Q row count {} != variable count {n}", q.len());
    }
    for (i, row) in q.iter().enumerate() {
        if row.len() != n {
            panic!("{name}: Q row {i} length {} != {n}", row.len());
        }
        for j in 0..n {
            if (q[i][j] - q[j][i]).abs() > 1e-7 {
                panic!("{name}: Q must be symmetric");
            }
        }
    }
}

fn validate_qcp(p: &QuadraticallyConstrainedProgram) {
    let n = p.c.len();
    if n == 0 {
        panic!("qcp: objective vector must be non-empty");
    }
    validate_square_symmetric("qcp objective", &p.q, n);
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("qcp: A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("qcp: A_eq / b_eq length mismatch");
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            panic!("qcp: A_ub row {i} length {} != {n}", row.len());
        }
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            panic!("qcp: A_eq row {i} length {} != {n}", row.len());
        }
    }
    if let Some(lb) = &p.lb {
        if lb.len() != n {
            panic!("qcp: lb length {} != {n}", lb.len());
        }
    }
    if let Some(ub) = &p.ub {
        if ub.len() != n {
            panic!("qcp: ub length {} != {n}", ub.len());
        }
    }
    if p.quadratic_constraints.is_empty() {
        panic!("qcp: at least one quadratic constraint is required");
    }
    for (k, qc) in p.quadratic_constraints.iter().enumerate() {
        validate_square_symmetric(&format!("qcp constraint {k}"), &qc.q, n);
        if qc.c.len() != n {
            panic!("qcp: constraint {k} c length {} != {n}", qc.c.len());
        }
        if !qc.rhs.is_finite() {
            panic!("qcp: constraint {k} rhs must be finite");
        }
    }
}

fn qcp_bounds(p: &QuadraticallyConstrainedProgram) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = p.c.len();
    (
        p.lb.clone().unwrap_or_else(|| vec![None; n]),
        p.ub.clone().unwrap_or_else(|| vec![None; n]),
    )
}

fn qcp_objective(p: &QuadraticallyConstrainedProgram, x: &[f64]) -> f64 {
    let qx = mat_vec(&p.q, x);
    0.5 * VecOps::dot(x, &qx) + VecOps::dot(&p.c, x)
}

fn qcp_constraint_value(qc: &QuadraticConstraint, x: &[f64]) -> f64 {
    let qx = mat_vec(&qc.q, x);
    VecOps::dot(x, &qx) + VecOps::dot(&qc.c, x)
}

fn qcp_gradient(p: &QuadraticallyConstrainedProgram, x: &[f64]) -> Vector {
    let qx = mat_vec(&p.q, x);
    qx.iter().zip(&p.c).map(|(qi, ci)| qi + ci).collect()
}

fn qcp_feasible(p: &QuadraticallyConstrainedProgram, x: &[f64], tol: f64) -> bool {
    if x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let (lb, ub) = qcp_bounds(p);
    for i in 0..x.len() {
        if let Some(l) = lb[i] {
            if x[i] < l - tol {
                return false;
            }
        }
        if let Some(u) = ub[i] {
            if x[i] > u + tol {
                return false;
            }
        }
    }
    if let Some(a_ub) = &p.a_ub {
        let b_ub = p.b_ub.as_ref().expect("b_ub");
        for (row, &rhs) in a_ub.iter().zip(b_ub) {
            if dot(row, x) > rhs + tol {
                return false;
            }
        }
    }
    if let Some(a_eq) = &p.a_eq {
        let b_eq = p.b_eq.as_ref().expect("b_eq");
        for (row, &rhs) in a_eq.iter().zip(b_eq) {
            if (dot(row, x) - rhs).abs() > tol {
                return false;
            }
        }
    }
    for qc in &p.quadratic_constraints {
        if qcp_constraint_value(qc, x) > qc.rhs + tol {
            return false;
        }
    }
    true
}

fn clamp_qcp_bounds(p: &QuadraticallyConstrainedProgram, x: &mut [f64]) {
    let (lb, ub) = qcp_bounds(p);
    for i in 0..x.len() {
        if let Some(l) = lb[i] {
            x[i] = x[i].max(l);
        }
        if let Some(u) = ub[i] {
            x[i] = x[i].min(u);
        }
    }
}

fn push_bounded_candidate(
    values: &mut Vec<f64>,
    value: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    tol: f64,
) {
    if !value.is_finite() {
        return;
    }
    let mut value = value;
    if let Some(l) = lower {
        if value < l - tol {
            return;
        }
        value = value.max(l);
    }
    if let Some(u) = upper {
        if value > u + tol {
            return;
        }
        value = value.min(u);
    }
    values.push(value);
}

fn add_single_linear_row_candidates(
    values: &mut [Vec<f64>],
    rows: &Option<Matrix>,
    rhs: &Option<Vector>,
    lb: &[Option<f64>],
    ub: &[Option<f64>],
    tol: f64,
) {
    let (Some(rows), Some(rhs)) = (rows, rhs) else {
        return;
    };
    for (row, &rhs_i) in rows.iter().zip(rhs) {
        let mut single = None;
        let mut nonzero_count = 0usize;
        for (j, &coef) in row.iter().enumerate() {
            if coef.abs() <= tol {
                continue;
            }
            single = Some((j, coef));
            nonzero_count += 1;
            if nonzero_count > 1 {
                break;
            }
        }
        if nonzero_count == 1 {
            let (j, coef) = single.expect("single nonzero coefficient");
            push_bounded_candidate(&mut values[j], rhs_i / coef, lb[j], ub[j], tol);
        }
    }
}

fn normalize_candidate_values(values: &mut [Vec<f64>], tol: f64) {
    for vals in values {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| (*a - *b).abs() <= tol);
    }
}

fn initial_qcp_point(p: &QuadraticallyConstrainedProgram, tol: f64) -> Option<Vector> {
    let n = p.c.len();
    let (lb, ub) = qcp_bounds(p);
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = match (lb[i], ub[i]) {
            (Some(l), Some(u)) => 0.5 * (l + u),
            (Some(l), None) if l > 0.0 => l,
            (None, Some(u)) if u < 0.0 => u,
            _ => 0.0,
        };
    }
    if qcp_feasible(p, &x, tol) {
        return Some(x);
    }
    let mut values: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut vals = Vec::new();
        push_bounded_candidate(&mut vals, 0.0, lb[i], ub[i], tol);
        if let Some(l) = lb[i] {
            push_bounded_candidate(&mut vals, l, lb[i], ub[i], tol);
        }
        if let Some(u) = ub[i] {
            push_bounded_candidate(&mut vals, u, lb[i], ub[i], tol);
        }
        if let (Some(l), Some(u)) = (lb[i], ub[i]) {
            push_bounded_candidate(&mut vals, 0.5 * (l + u), lb[i], ub[i], tol);
        }
        values.push(vals);
    }
    add_single_linear_row_candidates(&mut values, &p.a_eq, &p.b_eq, &lb, &ub, tol);
    normalize_candidate_values(&mut values, tol);
    fn search(
        p: &QuadraticallyConstrainedProgram,
        values: &[Vec<f64>],
        cur: &mut [f64],
        idx: usize,
        tol: f64,
    ) -> Option<Vector> {
        if idx == cur.len() {
            return qcp_feasible(p, cur, tol).then(|| cur.to_vec());
        }
        for &value in &values[idx] {
            cur[idx] = value;
            if let Some(x) = search(p, values, cur, idx + 1, tol) {
                return Some(x);
            }
        }
        None
    }
    search(p, &values, &mut x, 0, tol)
}

fn initial_qcp_step(p: &QuadraticallyConstrainedProgram) -> f64 {
    let (lb, ub) = qcp_bounds(p);
    let span = lb
        .iter()
        .zip(&ub)
        .filter_map(|(l, u)| match (l, u) {
            (Some(l), Some(u)) if u > l => Some(u - l),
            _ => None,
        })
        .fold(0.0_f64, f64::max);
    if span > 0.0 {
        (0.5 * span).max(1.0)
    } else {
        1.0
    }
}

fn qcp_directions(p: &QuadraticallyConstrainedProgram, x: &[f64]) -> Vec<Vector> {
    let n = p.c.len();
    let mut dirs = Vec::new();
    for i in 0..n {
        let mut plus = vec![0.0; n];
        plus[i] = 1.0;
        dirs.push(plus);
        let mut minus = vec![0.0; n];
        minus[i] = -1.0;
        dirs.push(minus);
    }
    let grad = qcp_gradient(p, x);
    let norm = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-12 {
        dirs.push(grad.iter().map(|v| -v / norm).collect());
    }
    dirs
}

/// Solve a small convex QCP by feasible pattern search.
///
/// This covers validation-scale quadratically constrained programs with convex
/// quadratic rows. It deliberately favours a transparent source-only baseline
/// over large-scale barrier performance.
pub fn solve_qcp_pattern_search(
    p: &QuadraticallyConstrainedProgram,
    opts: QcpOptions,
) -> QcpSolution {
    validate_qcp(p);
    let tol = opts.tol.max(1e-10);
    let Some(mut best_x) = initial_qcp_point(p, tol) else {
        return QcpSolution {
            status: QcpStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            iterations: 0,
            solver: "internal-qcp-pattern-search".to_string(),
            message: Some("no feasible starting point found".to_string()),
        };
    };
    let mut best_obj = qcp_objective(p, &best_x);
    let mut step = initial_qcp_step(p);
    let mut iterations = 0usize;
    let shrink = opts.shrink.clamp(0.1, 0.9);

    while iterations < opts.max_iters && step > tol {
        iterations += 1;
        let mut improved = false;
        let mut trial_best = best_x.clone();
        let mut trial_obj = best_obj;
        for dir in qcp_directions(p, &best_x) {
            let mut x: Vector = best_x
                .iter()
                .zip(&dir)
                .map(|(xi, di)| xi + step * di)
                .collect();
            clamp_qcp_bounds(p, &mut x);
            if !qcp_feasible(p, &x, tol) {
                continue;
            }
            let obj = qcp_objective(p, &x);
            if obj < trial_obj - tol {
                trial_obj = obj;
                trial_best = x;
                improved = true;
            }
        }
        if improved {
            best_x = trial_best;
            best_obj = trial_obj;
        } else {
            step *= shrink;
        }
    }

    if iterations >= opts.max_iters && step > tol {
        QcpSolution {
            status: QcpStatus::NumericalError,
            x: best_x,
            objective: best_obj,
            iterations,
            solver: "internal-qcp-pattern-search".to_string(),
            message: Some("pattern search iteration limit reached".to_string()),
        }
    } else {
        QcpSolution {
            status: QcpStatus::Optimal,
            x: best_x,
            objective: best_obj,
            iterations,
            solver: "internal-qcp-pattern-search".to_string(),
            message: Some("small convex QCP feasible pattern search".to_string()),
        }
    }
}

fn validate_socp(p: &SecondOrderConeProgram) {
    let n = p.c.len();
    if n == 0 {
        panic!("socp: objective vector must be non-empty");
    }
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("socp: A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("socp: A_eq / b_eq length mismatch");
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            panic!("socp: A_ub row {i} length {} != {n}", row.len());
        }
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            panic!("socp: A_eq row {i} length {} != {n}", row.len());
        }
    }
    if let Some(lb) = &p.lb {
        if lb.len() != n {
            panic!("socp: lb length {} != {n}", lb.len());
        }
    }
    if let Some(ub) = &p.ub {
        if ub.len() != n {
            panic!("socp: ub length {} != {n}", ub.len());
        }
    }
    if p.cones.is_empty() {
        panic!("socp: at least one cone is required");
    }
    for (k, cone) in p.cones.iter().enumerate() {
        if cone.a.len() != cone.b.len() {
            panic!("socp: cone {k} A / b length mismatch");
        }
        if cone.c.len() != n {
            panic!("socp: cone {k} c length {} != {n}", cone.c.len());
        }
        if !cone.d.is_finite() {
            panic!("socp: cone {k} d must be finite");
        }
        for (i, row) in cone.a.iter().enumerate() {
            if row.len() != n {
                panic!("socp: cone {k} A row {i} length {} != {n}", row.len());
            }
        }
    }
}

fn socp_bounds(p: &SecondOrderConeProgram) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = p.c.len();
    (
        p.lb.clone().unwrap_or_else(|| vec![None; n]),
        p.ub.clone().unwrap_or_else(|| vec![None; n]),
    )
}

fn socp_objective(p: &SecondOrderConeProgram, x: &[f64]) -> f64 {
    VecOps::dot(&p.c, x)
}

fn socp_feasible(p: &SecondOrderConeProgram, x: &[f64], tol: f64) -> bool {
    if x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let (lb, ub) = socp_bounds(p);
    for i in 0..x.len() {
        if let Some(l) = lb[i] {
            if x[i] < l - tol {
                return false;
            }
        }
        if let Some(u) = ub[i] {
            if x[i] > u + tol {
                return false;
            }
        }
    }
    if let Some(a_ub) = &p.a_ub {
        let b_ub = p.b_ub.as_ref().expect("b_ub");
        for (row, &rhs) in a_ub.iter().zip(b_ub) {
            if dot(row, x) > rhs + tol {
                return false;
            }
        }
    }
    if let Some(a_eq) = &p.a_eq {
        let b_eq = p.b_eq.as_ref().expect("b_eq");
        for (row, &rhs) in a_eq.iter().zip(b_eq) {
            if (dot(row, x) - rhs).abs() > tol {
                return false;
            }
        }
    }
    for cone in &p.cones {
        let ax = mat_vec(&cone.a, x);
        let lhs = ax
            .iter()
            .zip(&cone.b)
            .map(|(ai, bi)| {
                let v = ai + bi;
                v * v
            })
            .sum::<f64>()
            .sqrt();
        let rhs = dot(&cone.c, x) + cone.d;
        if rhs < -tol || lhs > rhs + tol {
            return false;
        }
    }
    true
}

fn clamp_socp_bounds(p: &SecondOrderConeProgram, x: &mut [f64]) {
    let (lb, ub) = socp_bounds(p);
    for i in 0..x.len() {
        if let Some(l) = lb[i] {
            x[i] = x[i].max(l);
        }
        if let Some(u) = ub[i] {
            x[i] = x[i].min(u);
        }
    }
}

fn initial_socp_point(p: &SecondOrderConeProgram, tol: f64) -> Option<Vector> {
    let n = p.c.len();
    let (lb, ub) = socp_bounds(p);
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = match (lb[i], ub[i]) {
            (Some(l), Some(u)) => 0.5 * (l + u),
            (Some(l), None) if l > 0.0 => l,
            (None, Some(u)) if u < 0.0 => u,
            _ => 0.0,
        };
    }
    if socp_feasible(p, &x, tol) {
        return Some(x);
    }
    let mut values: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut vals = Vec::new();
        push_bounded_candidate(&mut vals, 0.0, lb[i], ub[i], tol);
        if let Some(l) = lb[i] {
            push_bounded_candidate(&mut vals, l, lb[i], ub[i], tol);
        }
        if let Some(u) = ub[i] {
            push_bounded_candidate(&mut vals, u, lb[i], ub[i], tol);
        }
        if let (Some(l), Some(u)) = (lb[i], ub[i]) {
            push_bounded_candidate(&mut vals, 0.5 * (l + u), lb[i], ub[i], tol);
        }
        values.push(vals);
    }
    add_single_linear_row_candidates(&mut values, &p.a_eq, &p.b_eq, &lb, &ub, tol);
    normalize_candidate_values(&mut values, tol);
    fn search(
        p: &SecondOrderConeProgram,
        values: &[Vec<f64>],
        cur: &mut [f64],
        idx: usize,
        tol: f64,
    ) -> Option<Vector> {
        if idx == cur.len() {
            return socp_feasible(p, cur, tol).then(|| cur.to_vec());
        }
        for &value in &values[idx] {
            cur[idx] = value;
            if let Some(x) = search(p, values, cur, idx + 1, tol) {
                return Some(x);
            }
        }
        None
    }
    search(p, &values, &mut x, 0, tol)
}

fn initial_socp_step(p: &SecondOrderConeProgram) -> f64 {
    let (lb, ub) = socp_bounds(p);
    let span = lb
        .iter()
        .zip(&ub)
        .filter_map(|(l, u)| match (l, u) {
            (Some(l), Some(u)) if u > l => Some(u - l),
            _ => None,
        })
        .fold(0.0_f64, f64::max);
    if span > 0.0 {
        (0.5 * span).max(1.0)
    } else {
        1.0
    }
}

fn socp_directions(c: &[f64]) -> Vec<Vector> {
    let n = c.len();
    let mut dirs = Vec::new();
    for i in 0..n {
        let mut plus = vec![0.0; n];
        plus[i] = 1.0;
        dirs.push(plus);
        let mut minus = vec![0.0; n];
        minus[i] = -1.0;
        dirs.push(minus);
    }
    let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-12 {
        dirs.push(c.iter().map(|v| -v / norm).collect());
    }
    dirs
}

/// Solve a small SOCP by feasible pattern search.
///
/// This is a compact validation-scale conic solver. It is designed to exercise
/// the modelling surface and cross-check small convex instances, not to replace
/// a production barrier method for large cone programs.
pub fn solve_socp_pattern_search(p: &SecondOrderConeProgram, opts: SocpOptions) -> SocpSolution {
    validate_socp(p);
    let tol = opts.tol.max(1e-10);
    let Some(mut best_x) = initial_socp_point(p, tol) else {
        return SocpSolution {
            status: SocpStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            iterations: 0,
            solver: "internal-socp-pattern-search".to_string(),
            message: Some("no feasible starting point found".to_string()),
        };
    };
    let directions = socp_directions(&p.c);
    let mut best_obj = socp_objective(p, &best_x);
    let mut step = initial_socp_step(p);
    let mut iterations = 0usize;
    let shrink = opts.shrink.clamp(0.1, 0.9);

    while iterations < opts.max_iters && step > tol {
        iterations += 1;
        let mut improved = false;
        let mut trial_best = best_x.clone();
        let mut trial_obj = best_obj;
        for dir in &directions {
            let mut x: Vector = best_x
                .iter()
                .zip(dir)
                .map(|(xi, di)| xi + step * di)
                .collect();
            clamp_socp_bounds(p, &mut x);
            if !socp_feasible(p, &x, tol) {
                continue;
            }
            let obj = socp_objective(p, &x);
            if obj < trial_obj - tol {
                trial_obj = obj;
                trial_best = x;
                improved = true;
            }
        }
        if improved {
            best_x = trial_best;
            best_obj = trial_obj;
        } else {
            step *= shrink;
        }
    }

    if iterations >= opts.max_iters && step > tol {
        SocpSolution {
            status: SocpStatus::NumericalError,
            x: best_x,
            objective: best_obj,
            iterations,
            solver: "internal-socp-pattern-search".to_string(),
            message: Some("pattern search iteration limit reached".to_string()),
        }
    } else {
        SocpSolution {
            status: SocpStatus::Optimal,
            x: best_x,
            objective: best_obj,
            iterations,
            solver: "internal-socp-pattern-search".to_string(),
            message: Some("small SOCP feasible pattern search".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_equality_constrained_qp() {
        let qp = QuadraticProgram {
            q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            c: vec![-2.0, -5.0],
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![1.0]),
            lb: Some(vec![Some(0.0), Some(0.0)]),
            ..Default::default()
        };
        let sol = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(sol.status, QPStatus::Optimal);
        assert!((sol.x[0] - 0.0).abs() < 1e-8, "{sol:?}");
        assert!((sol.x[1] - 1.0).abs() < 1e-8, "{sol:?}");
        assert!((sol.objective + 4.0).abs() < 1e-8, "{sol:?}");
    }

    #[test]
    fn non_finite_data_reports_numerical_error() {
        // A NaN in `c` is malformed data, not an infeasible model: the solver
        // must say so (NumericalError), never silently "optimal"/"infeasible".
        let qp = QuadraticProgram {
            q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            c: vec![f64::NAN, -1.0],
            lb: Some(vec![Some(-1.0), Some(-1.0)]),
            ub: Some(vec![Some(1.0), Some(1.0)]),
            ..Default::default()
        };
        let sol = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(sol.status, QPStatus::NumericalError, "{sol:?}");
        assert!(sol.x.is_empty());
    }

    #[test]
    fn oversize_constraint_count_reports_numerical_error_without_panic() {
        // 70 inequality rows => 70 active-set candidates. `1usize << 70` would
        // overflow the enumeration shift (debug panic / release wrap); the guard
        // must return NumericalError cleanly instead of crashing.
        let rows = 70usize;
        let qp = QuadraticProgram {
            q: vec![vec![2.0]],
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0]; rows]),
            b_ub: Some(vec![1.0; rows]),
            lb: Some(vec![None]),
            ub: Some(vec![None]),
            ..Default::default()
        };
        let sol = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(sol.status, QPStatus::NumericalError, "{sol:?}");
    }

    #[test]
    fn invalid_solver_options_report_numerical_error_before_enumeration() {
        let qp = QuadraticProgram {
            q: vec![vec![2.0]],
            c: vec![-2.0],
            ..Default::default()
        };
        for opts in [
            QPOptions {
                tol: f64::NAN,
                ..Default::default()
            },
            QPOptions {
                tol: 0.0,
                ..Default::default()
            },
            QPOptions {
                max_active_sets: 0,
                ..Default::default()
            },
        ] {
            let sol = solve_qp_active_set(&qp, opts);
            assert_eq!(sol.status, QPStatus::NumericalError, "{sol:?}");
            assert_eq!(sol.iterations, 0, "{sol:?}");
            assert!(sol.x.is_empty(), "{sol:?}");
        }
    }

    #[test]
    fn enumeration_limit_is_rejected_before_partial_search() {
        let rows = 8usize;
        let qp = QuadraticProgram {
            q: vec![vec![2.0]],
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0]; rows]),
            b_ub: Some(vec![1.0; rows]),
            lb: Some(vec![None]),
            ub: Some(vec![None]),
            ..Default::default()
        };
        let sol = solve_qp_active_set(
            &qp,
            QPOptions {
                max_active_sets: 100,
                ..Default::default()
            },
        );
        assert_eq!(sol.status, QPStatus::NumericalError, "{sol:?}");
        assert_eq!(sol.iterations, 0, "{sol:?}");
        assert!(
            sol.message
                .as_deref()
                .is_some_and(|message| message.contains("requires 256 enumerations")),
            "{sol:?}"
        );
    }

    #[test]
    fn omitted_lower_bound_defaults_to_zero() {
        // Pins the documented nonnegativity default: min of x^2 + 6x is x=-3, but
        // with no `lb` the default lb=0 restricts the solver to x >= 0 => x=0.
        // Callers wanting a free variable must pass an explicit lb of None.
        let qp = QuadraticProgram {
            q: vec![vec![2.0]],
            c: vec![6.0],
            ..Default::default()
        };
        let sol = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(sol.status, QPStatus::Optimal, "{sol:?}");
        assert!(
            (sol.x[0] - 0.0).abs() < 1e-8,
            "default lb=0 should pin x at 0: {sol:?}"
        );
    }

    #[test]
    fn solves_bounded_mixed_integer_qp() {
        // min (x - 1.4)^2 + (y - 0.6)^2, x integer, x + y >= 1.5.
        // Continuous optimum has fractional x; bounded MIQP chooses x=1,y=0.6.
        let miqp = MixedIntegerQuadraticProgram {
            qp: QuadraticProgram {
                q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
                c: vec![-2.8, -1.2],
                a_ub: Some(vec![vec![-1.0, -1.0]]),
                b_ub: Some(vec![-1.5]),
                lb: Some(vec![Some(0.0), Some(0.0)]),
                ub: Some(vec![Some(3.0), Some(3.0)]),
                var_names: Some(vec!["x".to_string(), "y".to_string()]),
                ..Default::default()
            },
            integer_vars: vec![true, false],
        };
        let sol = solve_miqp_enumeration(&miqp, MIQPOptions::default());
        assert_eq!(sol.status, QPStatus::Optimal, "{sol:?}");
        assert!((sol.x[0] - 1.0).abs() < 1e-8, "{sol:?}");
        assert!((sol.x[1] - 0.6).abs() < 1e-8, "{sol:?}");
        assert!((sol.objective + 2.16).abs() < 1e-8, "{sol:?}");
        assert_eq!(sol.enumerated, 4);
    }

    #[test]
    fn solves_unit_ball_socp() {
        let socp = SecondOrderConeProgram {
            c: vec![1.0, 0.0],
            lb: Some(vec![Some(-2.0), Some(-2.0)]),
            ub: Some(vec![Some(2.0), Some(2.0)]),
            cones: vec![SecondOrderCone {
                a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                b: vec![0.0, 0.0],
                c: vec![0.0, 0.0],
                d: 1.0,
                name: Some("unit_ball".to_string()),
            }],
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..Default::default()
        };
        let sol = solve_socp_pattern_search(&socp, SocpOptions::default());
        assert_eq!(sol.status, SocpStatus::Optimal, "{sol:?}");
        assert!((sol.objective + 1.0).abs() < 1e-6, "{sol:?}");
        assert!((sol.x[0] + 1.0).abs() < 1e-6, "{sol:?}");
        assert!(sol.x[1].abs() < 1e-6, "{sol:?}");
    }

    #[test]
    fn solves_unit_ball_qcp() {
        let qcp = QuadraticallyConstrainedProgram {
            q: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            c: vec![1.0, 0.0],
            lb: Some(vec![Some(-2.0), Some(-2.0)]),
            ub: Some(vec![Some(2.0), Some(2.0)]),
            quadratic_constraints: vec![QuadraticConstraint {
                q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                c: vec![0.0, 0.0],
                rhs: 1.0,
                name: Some("unit_disk".to_string()),
            }],
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..Default::default()
        };
        let sol = solve_qcp_pattern_search(&qcp, QcpOptions::default());
        assert_eq!(sol.status, QcpStatus::Optimal, "{sol:?}");
        assert!((sol.objective + 1.0).abs() < 1e-6, "{sol:?}");
        assert!((sol.x[0] + 1.0).abs() < 1e-6, "{sol:?}");
        assert!(sol.x[1].abs() < 1e-6, "{sol:?}");
    }

    #[test]
    fn solves_rotated_socp_with_fixed_affine_variables() {
        let socp = SecondOrderConeProgram {
            c: vec![0.0, 1.0, 0.0],
            a_eq: Some(vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]]),
            b_eq: Some(vec![2.0, 4.0]),
            lb: Some(vec![Some(0.0), Some(0.0), Some(0.0)]),
            ub: Some(vec![Some(5.0), Some(10.0), Some(4.0)]),
            cones: vec![SecondOrderCone {
                a: vec![vec![0.0, 0.0, 2.0_f64.sqrt()], vec![1.0, -1.0, 0.0]],
                b: vec![0.0, 0.0],
                c: vec![1.0, 1.0, 0.0],
                d: 0.0,
                name: Some("rotated-as-standard-soc".to_string()),
            }],
            var_names: Some(vec!["u".to_string(), "v".to_string(), "z".to_string()]),
            ..Default::default()
        };
        let sol = solve_socp_pattern_search(&socp, SocpOptions::default());
        assert_eq!(sol.status, SocpStatus::Optimal, "{sol:?}");
        assert!((sol.x[0] - 2.0).abs() < 1e-7, "{sol:?}");
        assert!((sol.x[1] - 4.0).abs() < 1e-6, "{sol:?}");
        assert!((sol.x[2] - 4.0).abs() < 1e-7, "{sol:?}");
        assert!((sol.objective - 4.0).abs() < 1e-6, "{sol:?}");
    }

    #[test]
    fn solves_qcp_epigraph_with_fixed_affine_variable() {
        let qcp = QuadraticallyConstrainedProgram {
            q: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            c: vec![0.0, 1.0],
            a_eq: Some(vec![vec![1.0, 0.0]]),
            b_eq: Some(vec![3.0]),
            lb: Some(vec![Some(0.0), Some(0.0)]),
            ub: Some(vec![Some(5.0), Some(20.0)]),
            quadratic_constraints: vec![QuadraticConstraint {
                q: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
                c: vec![0.0, -1.0],
                rhs: 0.0,
                name: Some("square-epigraph".to_string()),
            }],
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..Default::default()
        };
        let sol = solve_qcp_pattern_search(&qcp, QcpOptions::default());
        assert_eq!(sol.status, QcpStatus::Optimal, "{sol:?}");
        assert!((sol.x[0] - 3.0).abs() < 1e-7, "{sol:?}");
        assert!((sol.x[1] - 9.0).abs() < 1e-6, "{sol:?}");
        assert!((sol.objective - 9.0).abs() < 1e-6, "{sol:?}");
    }
}
