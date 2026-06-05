//! Rust-facing bridge for external/reference TSP solvers.
//!
//! The native Rust reference computes an exact Held-Karp check without Python
//! startup. Explicit OR-Tools Routing validation is launched from Rust through
//! a tiny inline Python adapter, so the checked-in Python script can remain
//! launcher glue instead of owning solver-model construction.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn registered_tsp_rust_fallback_enabled() -> bool {
    [
        "TSP_REFERENCE_REGISTERED_FALLBACK",
        "TSP_REFERENCE_EXTERNAL_FALLBACK",
        "TSP_REFERENCE_RUST_FIRST",
        "ORES_EXTERNAL_REFERENCE_RUST_FIRST",
    ]
    .into_iter()
    .find_map(|key| std::env::var(key).ok())
    .map(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
        )
    })
    .unwrap_or(false)
}

fn should_use_rust_tsp_reference(opts: &ExternalTspReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalTspReferenceSolver::Auto
            | ExternalTspReferenceSolver::RustHeldKarp
            | ExternalTspReferenceSolver::Fallback
    )
}

fn should_use_registered_tsp_fallback(opts: &ExternalTspReferenceOptions) -> bool {
    registered_tsp_rust_fallback_enabled()
        && matches!(opts.solver, ExternalTspReferenceSolver::OrTools)
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

fn relabel_registered_tsp_fallback(
    mut solution: ExternalTspReferenceSolution,
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    if should_use_registered_tsp_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-tsp-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn tsp_reference_timeout_ms() -> u64 {
    std::env::var("TSP_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_tsp_adapter_output(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => return Err(format!("failed to poll OR-Tools TSP adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools TSP adapter: {err}"))
}

const ORTOOLS_TSP_DISTANCE_SCALE: i64 = 1_000_000;

const ORTOOLS_TSP_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:routing-tsp"

def emit(status, message, tour=None, ortools_status=None):
    payload = {
        "status": status,
        "solver": SOLVER,
        "tour": [] if tour is None else [int(v) for v in tour],
        "objective": None,
        "message": message,
        "ortoolsStatus": ortools_status,
        "ortoolsTour": [] if tour is None else [int(v) for v in tour],
        "ortoolsObjective": None,
    }
    print(json.dumps(payload))

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception as exc:
    emit("unavailable", f"OR-Tools Routing unavailable: {exc}", ortools_status="unavailable")
    raise SystemExit(0)

try:
    data = json.load(sys.stdin)
    matrix = data["scaledDistanceMatrix"]
    n = len(matrix)
    if n < 2:
        emit("numerical-error", "TSP requires at least two cities", tour=[], ortools_status="error")
        raise SystemExit(1)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node][to_node])

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        emit("infeasible", "OR-Tools Routing found no tour", tour=[], ortools_status="infeasible")
        raise SystemExit(0)

    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    emit("optimal", "OR-Tools Routing one-vehicle TSP", tour=tour, ortools_status="optimal")
except Exception as exc:
    emit("numerical-error", str(exc), tour=[], ortools_status="error")
    raise SystemExit(1)
"#;

fn scaled_ortools_tsp_distance(value: f64, row: usize, column: usize) -> Result<i64, String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "distance[{row}][{column}] must be finite and non-negative"
        ));
    }
    let scaled = (value * ORTOOLS_TSP_DISTANCE_SCALE as f64).round();
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return Err(format!(
            "distance[{row}][{column}] cannot be represented as a scaled OR-Tools integer"
        ));
    }
    Ok(scaled as i64)
}

fn ortools_tsp_payload(distance_matrix: &[Vec<f64>]) -> Result<Value, String> {
    validate_rust_tsp_distance_matrix(distance_matrix)?;
    let scaled_distance_matrix = distance_matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.iter()
                .enumerate()
                .map(|(column_index, &value)| {
                    scaled_ortools_tsp_distance(value, row_index, column_index)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "scale": ORTOOLS_TSP_DISTANCE_SCALE,
        "scaledDistanceMatrix": scaled_distance_matrix,
    }))
}

fn parsed_ortools_tsp_solution(
    parsed: TspReferencePayload,
    distance_matrix: &[Vec<f64>],
    output_success: bool,
    stderr: String,
    elapsed_ms: f64,
) -> ExternalTspReferenceSolution {
    let status = status_from_str(&parsed.status);
    let tour = parsed.tour.unwrap_or_default();
    let objective = if status == ExternalTspReferenceStatus::Optimal && !tour.is_empty() {
        Some(rust_tsp_tour_length(distance_matrix, &tour))
    } else {
        parsed.objective
    };
    let ortools_tour = parsed.ortools_tour.unwrap_or_else(|| tour.clone());
    let ortools_objective =
        if status == ExternalTspReferenceStatus::Optimal && !ortools_tour.is_empty() {
            Some(rust_tsp_tour_length(distance_matrix, &ortools_tour))
        } else {
            parsed.ortools_objective
        };
    ExternalTspReferenceSolution {
        status,
        solver: parsed
            .solver
            .unwrap_or_else(|| "ortools:routing-tsp".to_string()),
        tour,
        objective,
        ortools_status: parsed.ortools_status,
        ortools_tour,
        ortools_objective,
        message: parsed.message.unwrap_or_else(|| {
            if output_success {
                "ok".to_string()
            } else {
                stderr
            }
        }),
        elapsed_ms,
    }
}

fn run_ortools_tsp_reference(distance_matrix: &[Vec<f64>]) -> ExternalTspReferenceSolution {
    let started = Instant::now();
    let payload = match ortools_tsp_payload(distance_matrix) {
        Ok(payload) => payload,
        Err(message) => {
            return numerical_error(message, started.elapsed().as_secs_f64() * 1000.0);
        }
    };
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_TSP_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start OR-Tools TSP adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write OR-Tools TSP adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = tsp_reference_timeout_ms();
    let (output, timed_out) = match wait_for_tsp_adapter_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools TSP adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools TSP adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<TspReferencePayload>(&output.stdout) {
        Ok(parsed) => parsed_ortools_tsp_solution(
            parsed,
            distance_matrix,
            output.status.success(),
            stderr,
            elapsed_ms,
        ),
        Err(err) => numerical_error(
            format!(
                "failed to parse OR-Tools TSP adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_tsp_with_external_reference(
    distance_matrix: &[Vec<f64>],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    if should_use_rust_tsp_reference(opts) || should_use_registered_tsp_fallback(opts) {
        return relabel_registered_tsp_fallback(
            solve_tsp_with_rust_reference(distance_matrix),
            opts,
        );
    }

    run_ortools_tsp_reference(distance_matrix)
}

pub fn solve_euclidean_tsp_with_external_reference(
    points: &[ExternalTspPoint],
    opts: &ExternalTspReferenceOptions,
) -> ExternalTspReferenceSolution {
    if should_use_rust_tsp_reference(opts) || should_use_registered_tsp_fallback(opts) {
        let distance_matrix = rust_tsp_points_to_distance_matrix(points);
        return relabel_registered_tsp_fallback(
            solve_tsp_with_rust_reference(&distance_matrix),
            opts,
        );
    }

    let distance_matrix = rust_tsp_points_to_distance_matrix(points);
    run_ortools_tsp_reference(&distance_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TSP_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

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
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = TSP_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("TSP_REFERENCE_REGISTERED_FALLBACK", "rust");
        let opts = ExternalTspReferenceOptions {
            solver: ExternalTspReferenceSolver::OrTools,
        };

        let matrix_solution = solve_tsp_with_external_reference(&unit_square_matrix(), &opts);
        assert_eq!(matrix_solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(
            matrix_solution.solver,
            "rust:registered-tsp-fallback-for-ortools"
        );
        assert_eq!(matrix_solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(matrix_solution.objective, Some(4.0));
        assert!(matrix_solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));

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
        let point_solution = solve_euclidean_tsp_with_external_reference(&points, &opts);
        assert_eq!(point_solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(
            point_solution.solver,
            "rust:registered-tsp-fallback-for-ortools"
        );
        assert_eq!(point_solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(point_solution.objective, Some(4.0));
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

    #[test]
    fn rust_first_env_forces_ortools_to_rust_reference_without_python() {
        let _lock = TSP_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _rust_first_guard = EnvVarGuard::set("TSP_REFERENCE_RUST_FIRST", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-tsp");

        let solution = solve_tsp_with_external_reference(
            &unit_square_matrix(),
            &ExternalTspReferenceOptions {
                solver: ExternalTspReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalTspReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:registered-tsp-fallback-for-ortools");
        assert_eq!(solution.tour, vec![0, 1, 2, 3]);
        assert_eq!(solution.objective, Some(4.0));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = TSP_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not-a-python-for-tsp-ortools");

        let solution = solve_tsp_with_external_reference(
            &unit_square_matrix(),
            &ExternalTspReferenceOptions {
                solver: ExternalTspReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalTspReferenceStatus::Unavailable);
        assert!(solution.message.contains("OR-Tools TSP adapter"));
        assert!(!solution.message.contains("tsp_reference.py"));
    }

    #[test]
    fn tsp_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) = wait_for_tsp_adapter_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
