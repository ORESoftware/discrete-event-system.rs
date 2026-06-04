//! Rust-facing bridge for external/reference TSP solvers.
//!
//! The native Rust reference computes an exact Held-Karp check without Python
//! startup. The checked-in Python bridge (`scripts/tsp_reference.py`) remains
//! available for OR-Tools Routing's one-vehicle TSP result when OR-Tools is
//! available locally.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalTspReferenceSolver {
    Auto,
    RustHeldKarp,
    OrTools,
    Fallback,
}

impl ExternalTspReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalTspReferenceSolver::Auto => "auto",
            ExternalTspReferenceSolver::RustHeldKarp => "rust-held-karp",
            ExternalTspReferenceSolver::OrTools => "ortools",
            ExternalTspReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspReferenceOptions {
    pub solver: ExternalTspReferenceSolver,
}

impl Default for ExternalTspReferenceOptions {
    fn default() -> Self {
        ExternalTspReferenceOptions {
            solver: ExternalTspReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalTspReferenceStatus {
    Optimal,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalTspReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalTspReferenceStatus::Optimal => "optimal",
            ExternalTspReferenceStatus::Infeasible => "infeasible",
            ExternalTspReferenceStatus::Unsupported => "unsupported",
            ExternalTspReferenceStatus::NumericalError => "numerical-error",
            ExternalTspReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspPoint {
    pub id: Option<String>,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTspReferenceSolution {
    pub status: ExternalTspReferenceStatus,
    pub solver: String,
    pub tour: Vec<usize>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_tour: Vec<usize>,
    pub ortools_objective: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct TspReferencePayload {
    status: String,
    solver: Option<String>,
    tour: Option<Vec<usize>>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsTour")]
    ortools_tour: Option<Vec<usize>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalTspReferenceStatus {
    match status {
        "optimal" => ExternalTspReferenceStatus::Optimal,
        "infeasible" => ExternalTspReferenceStatus::Infeasible,
        "unsupported" => ExternalTspReferenceStatus::Unsupported,
        "unavailable" => ExternalTspReferenceStatus::Unavailable,
        _ => ExternalTspReferenceStatus::NumericalError,
    }
}

const RUST_TSP_MAX_HELD_KARP_N: usize = 16;
const RUST_TSP_EPS: f64 = 1e-12;

fn rust_tsp_empty_solution(
    status: ExternalTspReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalTspReferenceSolution {
    ExternalTspReferenceSolution {
        status,
        solver: solver.into(),
        tour: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn validate_rust_tsp_distance_matrix(distance_matrix: &[Vec<f64>]) -> Result<usize, String> {
    let n = distance_matrix.len();
    if n < 2 {
        return Err("TSP requires at least two cities".to_string());
    }
    for (row_index, row) in distance_matrix.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "distance row {row_index} length {} != {n}",
                row.len()
            ));
        }
        for (column_index, &value) in row.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "distance[{row_index}][{column_index}] must be finite and non-negative"
                ));
            }
        }
        if row[row_index].abs() > RUST_TSP_EPS {
            return Err(format!("distance[{row_index}][{row_index}] must be zero"));
        }
    }
    Ok(n)
}

fn rust_tsp_tour_length(distance_matrix: &[Vec<f64>], tour: &[usize]) -> f64 {
    if tour.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for window in tour.windows(2) {
        total += distance_matrix[window[0]][window[1]];
    }
    total + distance_matrix[*tour.last().expect("tour is non-empty")][tour[0]]
}

fn rust_tsp_reconstruct(parent: &[i64], mut mask: usize, mut end: usize, n: usize) -> Vec<usize> {
    let mut tour = Vec::new();
    loop {
        tour.push(end);
        let previous = parent[mask * n + end];
        mask ^= 1usize << end;
        if previous < 0 {
            break;
        }
        end = previous as usize;
    }
    tour.reverse();
    tour
}

fn solve_tsp_with_rust_reference(distance_matrix: &[Vec<f64>]) -> ExternalTspReferenceSolution {
    let started = Instant::now();
    let n = match validate_rust_tsp_distance_matrix(distance_matrix) {
        Ok(n) => n,
        Err(message) => {
            return rust_tsp_empty_solution(
                ExternalTspReferenceStatus::NumericalError,
                "rust:held-karp-tsp",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if n > RUST_TSP_MAX_HELD_KARP_N {
        return rust_tsp_empty_solution(
            ExternalTspReferenceStatus::Unsupported,
            "rust:held-karp-tsp",
            format!("Held-Karp TSP only practical for n <= {RUST_TSP_MAX_HELD_KARP_N}, got {n}"),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let state_count = 1usize << n;
    let mut dp = vec![f64::INFINITY; state_count * n];
    let mut parent = vec![-1_i64; state_count * n];
    dp[n] = 0.0;
    for mask in 1..state_count {
        if mask & 1 == 0 {
            continue;
        }
        for end in 0..n {
            if mask & (1usize << end) == 0 {
                continue;
            }
            let current = dp[mask * n + end];
            if !current.is_finite() {
                continue;
            }
            for next in 0..n {
                if mask & (1usize << next) != 0 {
                    continue;
                }
                let next_mask = mask | (1usize << next);
                let candidate = current + distance_matrix[end][next];
                let index = next_mask * n + next;
                if candidate < dp[index] - RUST_TSP_EPS {
                    dp[index] = candidate;
                    parent[index] = end as i64;
                }
            }
        }
    }

    let full_mask = state_count - 1;
    let mut best_tour = Vec::new();
    let mut best_objective = f64::INFINITY;
    for end in 1..n {
        let candidate = dp[full_mask * n + end] + distance_matrix[end][0];
        if !candidate.is_finite() {
            continue;
        }
        let candidate_tour = rust_tsp_reconstruct(&parent, full_mask, end, n);
        if candidate < best_objective - RUST_TSP_EPS
            || ((candidate - best_objective).abs() <= RUST_TSP_EPS
                && (best_tour.is_empty() || candidate_tour < best_tour))
        {
            best_objective = candidate;
            best_tour = candidate_tour;
        }
    }

    if best_tour.is_empty() {
        return rust_tsp_empty_solution(
            ExternalTspReferenceStatus::Infeasible,
            "rust:held-karp-tsp",
            "no Hamiltonian cycle",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    ExternalTspReferenceSolution {
        status: ExternalTspReferenceStatus::Optimal,
        solver: "rust:held-karp-tsp".to_string(),
        objective: Some(rust_tsp_tour_length(distance_matrix, &best_tour)),
        tour: best_tour,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: "exact Held-Karp dynamic program".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn rust_tsp_points_to_distance_matrix(points: &[ExternalTspPoint]) -> Vec<Vec<f64>> {
    points
        .iter()
        .map(|from| {
            points
                .iter()
                .map(|to| {
                    let dx = from.x - to.x;
                    let dy = from.y - to.y;
                    (dx * dx + dy * dy).sqrt()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalTspReferenceSolution {
    ExternalTspReferenceSolution {
        status: ExternalTspReferenceStatus::Unavailable,
        solver: "external-tsp-reference".to_string(),
        tour: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(message: impl Into<String>, elapsed_ms: f64) -> ExternalTspReferenceSolution {
    ExternalTspReferenceSolution {
        status: ExternalTspReferenceStatus::NumericalError,
        solver: "external-tsp-reference".to_string(),
        tour: Vec::new(),
        objective: None,
        ortools_status: None,
        ortools_tour: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("tsp_reference.py")
}

fn run_tsp_reference_json(
    payload: Value,
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg());
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start tsp_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write tsp_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for tsp_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<TspReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalTspReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-tsp-reference".to_string()),
            tour: parsed.tour.unwrap_or_default(),
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_tour: parsed.ortools_tour.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse tsp_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_tsp_with_external_reference(
    distance_matrix: &[Vec<f64>],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    if matches!(
        opts.solver,
        ExternalTspReferenceSolver::Auto
            | ExternalTspReferenceSolver::RustHeldKarp
            | ExternalTspReferenceSolver::Fallback
    ) {
        return solve_tsp_with_rust_reference(distance_matrix);
    }

    run_tsp_reference_json(
        json!({
            "distanceMatrix": distance_matrix,
        }),
        opts,
    )
}

pub fn solve_euclidean_tsp_with_external_reference(
    points: &[ExternalTspPoint],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    if matches!(
        opts.solver,
        ExternalTspReferenceSolver::Auto
            | ExternalTspReferenceSolver::RustHeldKarp
            | ExternalTspReferenceSolver::Fallback
    ) {
        let distance_matrix = rust_tsp_points_to_distance_matrix(points);
        return solve_tsp_with_rust_reference(&distance_matrix);
    }

    let points_json: Vec<Value> = points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            json!({
                "id": point.id.clone().unwrap_or_else(|| idx.to_string()),
                "x": point.x,
                "y": point.y,
            })
        })
        .collect();
    run_tsp_reference_json(
        json!({
            "points": points_json,
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 2.0_f64.sqrt(), 1.0],
            vec![1.0, 0.0, 1.0, 2.0_f64.sqrt()],
            vec![2.0_f64.sqrt(), 1.0, 0.0, 1.0],
            vec![1.0, 2.0_f64.sqrt(), 1.0, 0.0],
        ]
    }

    #[test]
    fn rust_reference_solves_unit_square_tsp() {
        let solution = solve_tsp_with_external_reference(
            &unit_square_matrix(),
            &ExternalTspReferenceOptions {
                solver: ExternalTspReferenceSolver::RustHeldKarp,
            },
        );

        assert_eq!(solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:held-karp-tsp");
        assert_eq!(solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(solution.objective, Some(4.0));
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_euclidean_points() {
        let points = vec![
            ExternalTspPoint {
                id: Some("A".to_string()),
                x: 0.0,
                y: 0.0,
            },
            ExternalTspPoint {
                id: Some("B".to_string()),
                x: 1.0,
                y: 0.0,
            },
            ExternalTspPoint {
                id: Some("C".to_string()),
                x: 1.0,
                y: 1.0,
            },
            ExternalTspPoint {
                id: Some("D".to_string()),
                x: 0.0,
                y: 1.0,
            },
        ];

        let solution = solve_euclidean_tsp_with_external_reference(
            &points,
            &ExternalTspReferenceOptions {
                solver: ExternalTspReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:held-karp-tsp");
        assert_eq!(solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(solution.objective, Some(4.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let solution = solve_tsp_with_external_reference(
            &unit_square_matrix(),
            &ExternalTspReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:held-karp-tsp");
        assert_eq!(solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(solution.objective, Some(4.0));
    }
}
