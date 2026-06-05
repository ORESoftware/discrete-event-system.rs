//! Rust-facing bridge for external/reference bin-packing solvers.
//!
//! The native Rust reference computes a deterministic exact small-instance
//! check without Python startup. The checked-in Python bridge
//! (`scripts/bin_packing_reference.py`) remains available for OR-Tools CP-SAT
//! on the same item/capacity input.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::bin_packing::BinPackingProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalBinPackingReferenceSolver {
    Auto,
    RustExact,
    OrTools,
    Fallback,
}

impl ExternalBinPackingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalBinPackingReferenceSolver::Auto => "auto",
            ExternalBinPackingReferenceSolver::RustExact => "rust-exact",
            ExternalBinPackingReferenceSolver::OrTools => "ortools",
            ExternalBinPackingReferenceSolver::Fallback => "fallback",
        }
    }
}

fn registered_bin_packing_rust_fallback_enabled() -> bool {
    std::env::var("BIN_PACKING_REFERENCE_REGISTERED_FALLBACK")
        .or_else(|_| std::env::var("BIN_PACKING_REFERENCE_EXTERNAL_FALLBACK"))
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
            )
        })
        .unwrap_or(false)
}

fn should_use_rust_bin_packing_reference(opts: &ExternalBinPackingReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalBinPackingReferenceSolver::Auto
            | ExternalBinPackingReferenceSolver::RustExact
            | ExternalBinPackingReferenceSolver::Fallback
    )
}

fn should_use_registered_bin_packing_fallback(opts: &ExternalBinPackingReferenceOptions) -> bool {
    registered_bin_packing_rust_fallback_enabled()
        && matches!(opts.solver, ExternalBinPackingReferenceSolver::OrTools)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceOptions {
    pub solver: ExternalBinPackingReferenceSolver,
}

impl Default for ExternalBinPackingReferenceOptions {
    fn default() -> Self {
        ExternalBinPackingReferenceOptions {
            solver: ExternalBinPackingReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalBinPackingReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalBinPackingReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalBinPackingReferenceStatus::Optimal => "optimal",
            ExternalBinPackingReferenceStatus::Feasible => "feasible",
            ExternalBinPackingReferenceStatus::Infeasible => "infeasible",
            ExternalBinPackingReferenceStatus::Unsupported => "unsupported",
            ExternalBinPackingReferenceStatus::NumericalError => "numerical-error",
            ExternalBinPackingReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceBin {
    pub item_ids: Vec<String>,
    pub load: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceSolution {
    pub status: ExternalBinPackingReferenceStatus,
    pub solver: String,
    pub bins: Vec<ExternalBinPackingReferenceBin>,
    pub objective: Option<usize>,
    pub total_weight: Option<f64>,
    pub lower_bound_bins: Option<usize>,
    pub ortools_status: Option<String>,
    pub ortools_bins: Vec<ExternalBinPackingReferenceBin>,
    pub ortools_objective: Option<usize>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct BinPackingReferencePayload {
    status: String,
    solver: Option<String>,
    bins: Option<Vec<BinPackingReferenceBinPayload>>,
    objective: Option<usize>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "lowerBoundBins")]
    lower_bound_bins: Option<usize>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsBins")]
    ortools_bins: Option<Vec<BinPackingReferenceBinPayload>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<usize>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BinPackingReferenceBinPayload {
    items: Vec<String>,
    load: f64,
}

impl From<BinPackingReferenceBinPayload> for ExternalBinPackingReferenceBin {
    fn from(value: BinPackingReferenceBinPayload) -> Self {
        ExternalBinPackingReferenceBin {
            item_ids: value.items,
            load: value.load,
        }
    }
}

#[derive(Clone, Debug)]
struct RustBinPackingSearchItem {
    index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct RustBinPackingSearchBin {
    item_indices: Vec<usize>,
    load: f64,
}

const RUST_BIN_PACKING_EPS: f64 = 1e-9;
const RUST_BIN_PACKING_MAX_EXACT_ITEMS: usize = 24;

fn status_from_str(status: &str) -> ExternalBinPackingReferenceStatus {
    match status {
        "optimal" => ExternalBinPackingReferenceStatus::Optimal,
        "feasible" => ExternalBinPackingReferenceStatus::Feasible,
        "infeasible" => ExternalBinPackingReferenceStatus::Infeasible,
        "unsupported" => ExternalBinPackingReferenceStatus::Unsupported,
        "unavailable" => ExternalBinPackingReferenceStatus::Unavailable,
        _ => ExternalBinPackingReferenceStatus::NumericalError,
    }
}

fn validate_rust_bin_packing_problem(problem: &BinPackingProblem) -> Result<(), String> {
    if !problem.capacity.is_finite() || problem.capacity <= 0.0 {
        return Err("capacity must be finite and > 0".to_string());
    }
    if problem.items.is_empty() {
        return Err("items must be non-empty".to_string());
    }
    let mut seen = HashSet::new();
    for (index, item) in problem.items.iter().enumerate() {
        if item.id.trim().is_empty() {
            return Err(format!("items[{index}].id must be non-empty"));
        }
        if !seen.insert(item.id.clone()) {
            return Err(format!("duplicate item id {:?}", item.id));
        }
        if !item.weight.is_finite() || item.weight <= 0.0 {
            return Err(format!("items[{index}].weight must be finite and > 0"));
        }
        if item.weight > problem.capacity + RUST_BIN_PACKING_EPS {
            return Err(format!("items[{index}].weight exceeds capacity"));
        }
    }
    Ok(())
}

fn rust_bin_packing_total_weight(problem: &BinPackingProblem) -> f64 {
    problem.items.iter().map(|item| item.weight).sum()
}

fn rust_bin_packing_lower_bound_bins(problem: &BinPackingProblem) -> usize {
    (rust_bin_packing_total_weight(problem) / problem.capacity).ceil() as usize
}

fn rust_bin_packing_empty_solution(
    status: ExternalBinPackingReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status,
        solver: solver.into(),
        bins: Vec::new(),
        objective: None,
        total_weight: None,
        lower_bound_bins: None,
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn relabel_registered_bin_packing_fallback(
    mut solution: ExternalBinPackingReferenceSolution,
    opts: &ExternalBinPackingReferenceOptions,
) -> ExternalBinPackingReferenceSolution {
    if should_use_registered_bin_packing_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-bin-packing-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn rust_bin_packing_unsupported_solution(
    problem: &BinPackingProblem,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status: ExternalBinPackingReferenceStatus::Unsupported,
        solver: "rust:exact-bin-packing".to_string(),
        bins: Vec::new(),
        objective: None,
        total_weight: Some(rust_bin_packing_total_weight(problem)),
        lower_bound_bins: Some(rust_bin_packing_lower_bound_bins(problem)),
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_bin_packing_sorted_items(problem: &BinPackingProblem) -> Vec<RustBinPackingSearchItem> {
    let mut items = problem
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| RustBinPackingSearchItem {
            index,
            weight: item.weight,
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.index.cmp(&right.index))
    });
    items
}

fn rust_bin_packing_solution(
    problem: &BinPackingProblem,
    status: ExternalBinPackingReferenceStatus,
    bins: Vec<RustBinPackingSearchBin>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalBinPackingReferenceSolution {
    let bins = bins
        .into_iter()
        .map(|bin| {
            let mut item_indices = bin.item_indices;
            item_indices.sort_unstable();
            let load = item_indices
                .iter()
                .map(|&index| problem.items[index].weight)
                .sum::<f64>();
            let item_ids = item_indices
                .iter()
                .map(|&index| problem.items[index].id.clone())
                .collect::<Vec<_>>();
            ExternalBinPackingReferenceBin { item_ids, load }
        })
        .collect::<Vec<_>>();
    ExternalBinPackingReferenceSolution {
        status,
        solver: "rust:exact-bin-packing".to_string(),
        objective: Some(bins.len()),
        bins,
        total_weight: Some(rust_bin_packing_total_weight(problem)),
        lower_bound_bins: Some(rust_bin_packing_lower_bound_bins(problem)),
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_bin_packing_first_fit_decreasing(
    problem: &BinPackingProblem,
) -> Vec<RustBinPackingSearchBin> {
    let mut bins = Vec::<RustBinPackingSearchBin>::new();
    for item in rust_bin_packing_sorted_items(problem) {
        if let Some(bin) = bins
            .iter_mut()
            .find(|bin| bin.load + item.weight <= problem.capacity + RUST_BIN_PACKING_EPS)
        {
            bin.load += item.weight;
            bin.item_indices.push(item.index);
        } else {
            bins.push(RustBinPackingSearchBin {
                item_indices: vec![item.index],
                load: item.weight,
            });
        }
    }
    bins
}

fn rust_bin_packing_exact_search(
    capacity: f64,
    order: &[RustBinPackingSearchItem],
    suffix_weight: &[f64],
    pos: usize,
    current: &mut Vec<RustBinPackingSearchBin>,
    best_bins: &mut Vec<RustBinPackingSearchBin>,
    best_count: &mut usize,
) {
    if current.len() >= *best_count {
        return;
    }
    if pos == order.len() {
        *best_count = current.len();
        *best_bins = current.clone();
        return;
    }

    let free_capacity = current
        .iter()
        .map(|bin| (capacity - bin.load).max(0.0))
        .sum::<f64>();
    let additional_weight = (suffix_weight[pos] - free_capacity).max(0.0);
    let additional_bins = (additional_weight / capacity).ceil() as usize;
    if current.len() + additional_bins >= *best_count {
        return;
    }

    let item = &order[pos];
    let mut tried_loads = Vec::<f64>::new();
    for bin_index in 0..current.len() {
        let load = current[bin_index].load;
        if load + item.weight > capacity + RUST_BIN_PACKING_EPS
            || tried_loads
                .iter()
                .any(|&previous| (previous - load).abs() <= RUST_BIN_PACKING_EPS)
        {
            continue;
        }
        tried_loads.push(load);
        current[bin_index].load += item.weight;
        current[bin_index].item_indices.push(item.index);
        rust_bin_packing_exact_search(
            capacity,
            order,
            suffix_weight,
            pos + 1,
            current,
            best_bins,
            best_count,
        );
        current[bin_index].item_indices.pop();
        current[bin_index].load = load;
    }

    if current.len() + 1 < *best_count {
        current.push(RustBinPackingSearchBin {
            item_indices: vec![item.index],
            load: item.weight,
        });
        rust_bin_packing_exact_search(
            capacity,
            order,
            suffix_weight,
            pos + 1,
            current,
            best_bins,
            best_count,
        );
        current.pop();
    }
}

fn solve_bin_packing_with_rust_reference(
    problem: &BinPackingProblem,
) -> ExternalBinPackingReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_bin_packing_problem(problem) {
        return rust_bin_packing_empty_solution(
            ExternalBinPackingReferenceStatus::NumericalError,
            "rust:exact-bin-packing",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if problem.items.len() > RUST_BIN_PACKING_MAX_EXACT_ITEMS {
        return rust_bin_packing_unsupported_solution(
            problem,
            format!(
                "exact bin-packing only practical for <= {RUST_BIN_PACKING_MAX_EXACT_ITEMS} items, got {}",
                problem.items.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut best_bins = rust_bin_packing_first_fit_decreasing(problem);
    let mut best_count = best_bins.len();
    let lower_bound = rust_bin_packing_lower_bound_bins(problem);
    if best_count == lower_bound {
        return rust_bin_packing_solution(
            problem,
            ExternalBinPackingReferenceStatus::Optimal,
            best_bins,
            "exact branch-and-bound certified by volume lower bound",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let order = rust_bin_packing_sorted_items(problem);
    let mut suffix_weight = vec![0.0; order.len() + 1];
    for index in (0..order.len()).rev() {
        suffix_weight[index] = suffix_weight[index + 1] + order[index].weight;
    }
    let mut current = Vec::new();
    rust_bin_packing_exact_search(
        problem.capacity,
        &order,
        &suffix_weight,
        0,
        &mut current,
        &mut best_bins,
        &mut best_count,
    );

    rust_bin_packing_solution(
        problem,
        ExternalBinPackingReferenceStatus::Optimal,
        best_bins,
        "exact branch-and-bound",
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status: ExternalBinPackingReferenceStatus::Unavailable,
        solver: "external-bin-packing-reference".to_string(),
        bins: Vec::new(),
        objective: None,
        total_weight: None,
        lower_bound_bins: None,
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status: ExternalBinPackingReferenceStatus::NumericalError,
        solver: "external-bin-packing-reference".to_string(),
        bins: Vec::new(),
        objective: None,
        total_weight: None,
        lower_bound_bins: None,
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("bin_packing_reference.py")
}

fn bin_packing_reference_timeout_ms() -> u64 {
    std::env::var("BIN_PACKING_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_bin_packing_reference_output(
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
            Err(err) => return Err(format!("failed to poll bin_packing_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for bin_packing_reference.py: {err}"))
}

fn run_bin_packing_reference_json(
    payload: Value,
    opts: &ExternalBinPackingReferenceOptions,
) -> ExternalBinPackingReferenceSolution {
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
                format!("failed to start bin_packing_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write bin_packing_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    drop(child.stdin.take());
    let timeout_ms = bin_packing_reference_timeout_ms();
    let (mut output, timed_out) = match wait_for_bin_packing_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0);
        }
    };
    if timed_out {
        let timeout_message = format!("bin_packing_reference.py timed out after {}ms", timeout_ms);
        if output.stderr.is_empty() {
            output.stderr = timeout_message.into_bytes();
        } else {
            let mut stderr = timeout_message.into_bytes();
            stderr.push(b'\n');
            stderr.extend(output.stderr);
            output.stderr = stderr;
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<BinPackingReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalBinPackingReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-bin-packing-reference".to_string()),
            bins: parsed
                .bins
                .unwrap_or_default()
                .into_iter()
                .map(ExternalBinPackingReferenceBin::from)
                .collect(),
            objective: parsed.objective,
            total_weight: parsed.total_weight,
            lower_bound_bins: parsed.lower_bound_bins,
            ortools_status: parsed.ortools_status,
            ortools_bins: parsed
                .ortools_bins
                .unwrap_or_default()
                .into_iter()
                .map(ExternalBinPackingReferenceBin::from)
                .collect(),
            ortools_objective: parsed.ortools_objective,
            ortools_objective_bound: parsed.ortools_objective_bound,
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
                "failed to parse bin_packing_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_bin_packing_with_external_reference(
    problem: &BinPackingProblem,
    opts: &ExternalBinPackingReferenceOptions,
) -> ExternalBinPackingReferenceSolution {
    if should_use_rust_bin_packing_reference(opts)
        || should_use_registered_bin_packing_fallback(opts)
    {
        return relabel_registered_bin_packing_fallback(
            solve_bin_packing_with_rust_reference(problem),
            opts,
        );
    }

    run_bin_packing_reference_json(
        json!({
            "capacity": problem.capacity,
            "items": problem.items.iter().map(|item| json!({
                "id": &item.id,
                "weight": item.weight,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::bin_packing::{
        bin_packing_problem_from_weights, build_sample_bin_packing_problem,
    };
    use std::sync::Mutex;

    static BIN_PACKING_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn packed_load_sum(bins: &[ExternalBinPackingReferenceBin]) -> f64 {
        bins.iter().map(|bin| bin.load).sum()
    }

    #[test]
    fn rust_reference_solves_sample_bin_packing() {
        let problem = build_sample_bin_packing_problem();
        let solution = solve_bin_packing_with_external_reference(
            &problem,
            &ExternalBinPackingReferenceOptions {
                solver: ExternalBinPackingReferenceSolver::RustExact,
            },
        );

        assert_eq!(solution.status, ExternalBinPackingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-bin-packing");
        assert_eq!(solution.objective, Some(3));
        assert_eq!(solution.lower_bound_bins, Some(3));
        assert_eq!(solution.total_weight, Some(30.0));
        assert!((packed_load_sum(&solution.bins) - 30.0).abs() <= 1e-9);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_when_volume_bound_certifies() {
        let problem = bin_packing_problem_from_weights(10.0, vec![6.0, 4.0, 5.0, 5.0]);
        let solution = solve_bin_packing_with_external_reference(
            &problem,
            &ExternalBinPackingReferenceOptions {
                solver: ExternalBinPackingReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalBinPackingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-bin-packing");
        assert_eq!(solution.objective, Some(2));
        assert_eq!(solution.lower_bound_bins, Some(2));
        assert_eq!(solution.total_weight, Some(20.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_bin_packing_problem();

        let solution = solve_bin_packing_with_external_reference(
            &problem,
            &ExternalBinPackingReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalBinPackingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-bin-packing");
        assert_eq!(solution.objective, Some(3));
        assert_eq!(solution.total_weight, Some(30.0));
    }

    #[test]
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = BIN_PACKING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _guard = EnvVarGuard::set("BIN_PACKING_REFERENCE_REGISTERED_FALLBACK", "rust");
        let problem = build_sample_bin_packing_problem();

        let solution = solve_bin_packing_with_external_reference(
            &problem,
            &ExternalBinPackingReferenceOptions {
                solver: ExternalBinPackingReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalBinPackingReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-bin-packing-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(3));
        assert_eq!(solution.total_weight, Some(30.0));
        assert!((packed_load_sum(&solution.bins) - 30.0).abs() <= 1e-9);
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn bin_packing_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_bin_packing_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn bin_packing_python_bridge_wait_observes_closed_stdin() {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("cat >/dev/null; printf done")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn stdin reader");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(b"{\"capacity\":1,\"items\":[{\"id\":\"A\",\"weight\":1}]}")
            .expect("write stdin");
        drop(child.stdin.take());

        let (output, timed_out) =
            wait_for_bin_packing_reference_output(child, 1_000).expect("closed stdin output");

        assert!(!timed_out);
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "done");
    }
}
