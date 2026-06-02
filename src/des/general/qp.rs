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
    pub active_ub_rows: Vec<usize>,
    pub active_lower_bounds: Vec<usize>,
    pub active_upper_bounds: Vec<usize>,
    pub iterations: usize,
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

fn mat_vec(a: &Matrix, x: &[f64]) -> Vector {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(ai, xi)| ai * xi).sum())
        .collect()
}

fn dot(row: &[f64], x: &[f64]) -> f64 {
    row.iter().zip(x).map(|(a, xi)| a * xi).sum()
}

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

fn solve_kkt(p: &QuadraticProgram, active: &[ActiveKind], tol: f64) -> Option<Vector> {
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
        .map(|v| v[..n].to_vec())
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

/// Solve a small dense convex QP by active-set enumeration.
pub fn solve_qp_active_set(p: &QuadraticProgram, opts: QPOptions) -> QPSolution {
    validate_qp(p);
    let candidates = candidate_constraints(p);
    let n = p.c.len();
    let mut best_x = Vec::new();
    let mut best_obj = f64::INFINITY;
    let mut best_active = Vec::new();
    let mut iterations = 0usize;

    for mask in 0usize..(1usize << candidates.len()) {
        iterations += 1;
        if iterations > opts.max_active_sets {
            return QPSolution {
                status: QPStatus::NumericalError,
                x: best_x,
                objective: best_obj,
                active_ub_rows: Vec::new(),
                active_lower_bounds: Vec::new(),
                active_upper_bounds: Vec::new(),
                iterations,
                solver: "internal-active-set-enumeration".to_string(),
                message: Some("active-set enumeration limit reached".to_string()),
            };
        }
        let mut active = Vec::new();
        for (i, &kind) in candidates.iter().enumerate() {
            if (mask & (1usize << i)) != 0 {
                active.push(kind);
            }
        }
        if active.len() > n {
            continue;
        }
        let Some(x) = solve_kkt(p, &active, opts.tol.max(1e-12)) else {
            continue;
        };
        if !feasible(p, &x, opts.tol.max(1e-8)) {
            continue;
        }
        let obj = objective(p, &x);
        if obj < best_obj - opts.tol {
            best_obj = obj;
            best_x = x;
            best_active = active;
        }
    }

    if best_x.is_empty() {
        return QPSolution {
            status: QPStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            active_ub_rows: Vec::new(),
            active_lower_bounds: Vec::new(),
            active_upper_bounds: Vec::new(),
            iterations,
            solver: "internal-active-set-enumeration".to_string(),
            message: Some("no feasible KKT candidate found".to_string()),
        };
    }
    let (active_ub_rows, active_lower_bounds, active_upper_bounds) = decode_active(&best_active);
    QPSolution {
        status: QPStatus::Optimal,
        x: best_x,
        objective: best_obj,
        active_ub_rows,
        active_lower_bounds,
        active_upper_bounds,
        iterations,
        solver: "internal-active-set-enumeration".to_string(),
        message: Some("convex QP active-set enumeration".to_string()),
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
        let mut vals = vec![0.0];
        if let Some(l) = lb[i] {
            vals.push(l);
        }
        if let Some(u) = ub[i] {
            vals.push(u);
        }
        if let (Some(l), Some(u)) = (lb[i], ub[i]) {
            vals.push(0.5 * (l + u));
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup_by(|a, b| (*a - *b).abs() <= tol);
        values.push(vals);
    }
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
        let mut vals = vec![0.0];
        if let Some(l) = lb[i] {
            vals.push(l);
        }
        if let Some(u) = ub[i] {
            vals.push(u);
        }
        if let (Some(l), Some(u)) = (lb[i], ub[i]) {
            vals.push(0.5 * (l + u));
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup_by(|a, b| (*a - *b).abs() <= tol);
        values.push(vals);
    }
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
}
