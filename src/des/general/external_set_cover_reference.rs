//! Rust-facing bridge for external/reference set-cover solvers.
//!
//! The native Rust reference computes an exact small-instance check without
//! Python startup. Explicit OR-Tools CP-SAT validation is launched from Rust
//! with a tiny Python adapter over an integer-scaled copy of the same model.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::set_cover::SetCoverProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSetCoverReferenceSolver {
    Auto,
    RustExact,
    OrTools,
    Fallback,
}

impl ExternalSetCoverReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalSetCoverReferenceSolver::Auto => "auto",
            ExternalSetCoverReferenceSolver::RustExact => "rust-exact",
            ExternalSetCoverReferenceSolver::OrTools => "ortools",
            ExternalSetCoverReferenceSolver::Fallback => "fallback",
        }
    }
}

fn registered_set_cover_rust_fallback_enabled() -> bool {
    [
        "SET_COVER_REFERENCE_REGISTERED_FALLBACK",
        "SET_COVER_REFERENCE_EXTERNAL_FALLBACK",
        "SET_COVER_REFERENCE_RUST_FIRST",
        "ORES_EXTERNAL_REFERENCE_RUST_FIRST",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
                )
            })
            .unwrap_or(false)
    })
}

fn should_use_rust_set_cover_reference(opts: &ExternalSetCoverReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalSetCoverReferenceSolver::Auto
            | ExternalSetCoverReferenceSolver::RustExact
            | ExternalSetCoverReferenceSolver::Fallback
    )
}

fn should_use_registered_set_cover_fallback(opts: &ExternalSetCoverReferenceOptions) -> bool {
    registered_set_cover_rust_fallback_enabled()
        && matches!(opts.solver, ExternalSetCoverReferenceSolver::OrTools)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSetCoverReferenceOptions {
    pub solver: ExternalSetCoverReferenceSolver,
}

impl Default for ExternalSetCoverReferenceOptions {
    fn default() -> Self {
        ExternalSetCoverReferenceOptions {
            solver: ExternalSetCoverReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSetCoverReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalSetCoverReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalSetCoverReferenceStatus::Optimal => "optimal",
            ExternalSetCoverReferenceStatus::Feasible => "feasible",
            ExternalSetCoverReferenceStatus::Infeasible => "infeasible",
            ExternalSetCoverReferenceStatus::Unsupported => "unsupported",
            ExternalSetCoverReferenceStatus::NumericalError => "numerical-error",
            ExternalSetCoverReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSetCoverReferenceSolution {
    pub status: ExternalSetCoverReferenceStatus,
    pub solver: String,
    pub selected_set_indices: Vec<usize>,
    pub selected_set_ids: Vec<String>,
    pub objective: Option<f64>,
    pub covered_elements: Vec<String>,
    pub ortools_status: Option<String>,
    pub ortools_selected_set_indices: Vec<usize>,
    pub ortools_selected_set_ids: Vec<String>,
    pub ortools_objective: Option<f64>,
    pub ortools_covered_elements: Vec<String>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct SetCoverReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedSetIndices")]
    selected_set_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedSets")]
    selected_sets: Option<Vec<String>>,
    objective: Option<f64>,
    #[serde(rename = "coveredElements")]
    covered_elements: Option<Vec<String>>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedSetIndices")]
    ortools_selected_set_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedSets")]
    ortools_selected_sets: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsCoveredElements")]
    ortools_covered_elements: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    #[serde(rename = "objectiveBound")]
    objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalSetCoverReferenceStatus {
    match status {
        "optimal" => ExternalSetCoverReferenceStatus::Optimal,
        "feasible" => ExternalSetCoverReferenceStatus::Feasible,
        "infeasible" => ExternalSetCoverReferenceStatus::Infeasible,
        "unsupported" => ExternalSetCoverReferenceStatus::Unsupported,
        "unavailable" => ExternalSetCoverReferenceStatus::Unavailable,
        _ => ExternalSetCoverReferenceStatus::NumericalError,
    }
}

const RUST_SET_COVER_EPS: f64 = 1e-9;
const RUST_SET_COVER_MAX_EXACT_SETS: usize = 32;
const RUST_SET_COVER_MAX_EXACT_ELEMENTS: usize = 128;
const ORTOOLS_COST_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_SET_COVER_SOLVER: &str = "ortools:cp-sat-set-cover";

const ORTOOLS_SET_COVER_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:cp-sat-set-cover"


def output(status, problem, selected_indices=None, objective_bound=None, message=""):
    if selected_indices is None:
        selected = []
        selected_ids = []
        objective = None
        covered_elements = []
    else:
        selected = sorted(set(int(index) for index in selected_indices))
        selected_ids = [problem["sets"][index]["id"] for index in selected]
        objective = float(sum(float(problem["sets"][index]["cost"]) for index in selected))
        covered = {
            element
            for index in selected
            for element in problem["sets"][index]["elements"]
        }
        covered_elements = [element for element in problem["universe"] if element in covered]
    result = {
        "status": status,
        "solver": SOLVER,
        "selectedSetIndices": selected,
        "selectedSets": selected_ids,
        "objective": objective,
        "coveredElements": covered_elements,
        "message": message,
    }
    if objective_bound is not None:
        result["objectiveBound"] = objective_bound
    return result


try:
    from ortools.sat.python import cp_model
except Exception as exc:
    print(json.dumps(output("unavailable", {"sets": [], "universe": []}, None, None, str(exc))))
    sys.exit(0)


try:
    problem = json.load(sys.stdin)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_s{index}") for index in range(len(problem["sets"]))]
    for element in problem["universe"]:
        covering = [
            x[index]
            for index, set_ in enumerate(problem["sets"])
            if element in set_["elements"]
        ]
        if not covering:
            print(json.dumps(output(
                "infeasible",
                problem,
                None,
                None,
                f"element {element!r} is uncovered by all sets",
            )))
            sys.exit(0)
        model.Add(sum(covering) >= 1)

    model.Minimize(sum(
        int(set_["scaledCost"]) * x[index]
        for index, set_ in enumerate(problem["sets"])
    ))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(json.dumps(output(
            "infeasible" if status_name == "infeasible" else status_name,
            problem,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )))
        sys.exit(0)
    selected = [index for index, var in enumerate(x) if solver.BooleanValue(var)]
    print(json.dumps(output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        problem,
        selected,
        solver.BestObjectiveBound() / float(problem["costScale"]),
        f"OR-Tools CP-SAT status {status_name}",
    )))
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "solver": SOLVER,
        "selectedSetIndices": [],
        "selectedSets": [],
        "objective": None,
        "coveredElements": [],
        "message": str(exc),
    }))
    sys.exit(1)
"#;

fn validate_rust_set_cover_problem(
    problem: &SetCoverProblem,
) -> Result<HashMap<String, usize>, String> {
    if problem.universe.is_empty() {
        return Err("universe must be non-empty".to_string());
    }
    if problem.sets.is_empty() {
        return Err("sets must be non-empty".to_string());
    }

    let mut element_index = HashMap::with_capacity(problem.universe.len());
    for (index, element) in problem.universe.iter().enumerate() {
        if element.trim().is_empty() {
            return Err("universe elements must be non-empty".to_string());
        }
        if element_index.insert(element.clone(), index).is_some() {
            return Err("universe elements must be unique".to_string());
        }
    }

    let mut seen_set_ids = HashSet::new();
    for (set_index, set) in problem.sets.iter().enumerate() {
        if set.id.trim().is_empty() {
            return Err(format!("sets[{set_index}].id must be non-empty"));
        }
        if !seen_set_ids.insert(set.id.clone()) {
            return Err(format!("duplicate set id {:?}", set.id));
        }
        if !set.cost.is_finite() || set.cost < 0.0 {
            return Err(format!("sets[{set_index}].cost must be finite and >= 0"));
        }
        if set.elements.is_empty() {
            return Err(format!("sets[{set_index}].elements must be non-empty"));
        }
        let mut seen_elements = HashSet::new();
        for element in &set.elements {
            if !seen_elements.insert(element.clone()) {
                return Err(format!("sets[{set_index}].elements must be unique"));
            }
            if !element_index.contains_key(element) {
                return Err(format!(
                    "sets[{set_index}].elements not in universe: {element:?}"
                ));
            }
        }
    }

    Ok(element_index)
}

fn rust_set_cover_empty_solution(
    status: ExternalSetCoverReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalSetCoverReferenceSolution {
    ExternalSetCoverReferenceSolution {
        status,
        solver: solver.into(),
        selected_set_indices: Vec::new(),
        selected_set_ids: Vec::new(),
        objective: None,
        covered_elements: Vec::new(),
        ortools_status: None,
        ortools_selected_set_indices: Vec::new(),
        ortools_selected_set_ids: Vec::new(),
        ortools_objective: None,
        ortools_covered_elements: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn ortools_set_cover_empty_solution(
    status: ExternalSetCoverReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalSetCoverReferenceSolution {
    rust_set_cover_empty_solution(status, ORTOOLS_SET_COVER_SOLVER, message, elapsed_ms)
}

fn scaled_ortools_cost(value: f64, scale: i64) -> Option<i64> {
    if !value.is_finite() || scale <= 0 {
        return None;
    }
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return None;
    }
    let rounded = scaled.round();
    if (scaled - rounded).abs() > 1e-6 {
        return None;
    }
    Some(rounded as i64)
}

fn choose_ortools_cost_scale(problem: &SetCoverProblem) -> Option<i64> {
    ORTOOLS_COST_SCALES.iter().copied().find(|&scale| {
        problem
            .sets
            .iter()
            .all(|set| scaled_ortools_cost(set.cost, scale).is_some())
    })
}

fn ortools_set_cover_payload(problem: &SetCoverProblem, cost_scale: i64) -> Value {
    json!({
        "universe": &problem.universe,
        "costScale": cost_scale,
        "sets": problem.sets.iter().enumerate().map(|(index, set)| json!({
            "index": index,
            "id": &set.id,
            "cost": set.cost,
            "scaledCost": scaled_ortools_cost(set.cost, cost_scale)
                .expect("selected OR-Tools scale must scale every set cost"),
            "elements": &set.elements,
        })).collect::<Vec<_>>(),
    })
}

fn relabel_registered_set_cover_fallback(
    mut solution: ExternalSetCoverReferenceSolution,
    opts: &ExternalSetCoverReferenceOptions,
) -> ExternalSetCoverReferenceSolution {
    if should_use_registered_set_cover_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-set-cover-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn rust_set_cover_masks(
    problem: &SetCoverProblem,
    element_index: &HashMap<String, usize>,
) -> (Vec<u128>, u128) {
    let set_masks = problem
        .sets
        .iter()
        .map(|set| {
            let mut mask = 0_u128;
            for element in &set.elements {
                mask |= 1_u128 << element_index[element];
            }
            mask
        })
        .collect::<Vec<_>>();
    let full_mask = if problem.universe.len() == 128 {
        u128::MAX
    } else {
        (1_u128 << problem.universe.len()) - 1
    };
    (set_masks, full_mask)
}

fn rust_set_cover_solution(
    problem: &SetCoverProblem,
    status: ExternalSetCoverReferenceStatus,
    mut selected_set_indices: Vec<usize>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalSetCoverReferenceSolution {
    selected_set_indices.sort_unstable();
    selected_set_indices.dedup();
    let selected_set_ids = selected_set_indices
        .iter()
        .map(|&index| problem.sets[index].id.clone())
        .collect::<Vec<_>>();
    let objective = selected_set_indices
        .iter()
        .map(|&index| problem.sets[index].cost)
        .sum::<f64>();
    let covered = selected_set_indices
        .iter()
        .flat_map(|&index| problem.sets[index].elements.iter().cloned())
        .collect::<HashSet<_>>();
    let covered_elements = problem
        .universe
        .iter()
        .filter(|element| covered.contains(*element))
        .cloned()
        .collect::<Vec<_>>();

    ExternalSetCoverReferenceSolution {
        status,
        solver: "rust:exact-set-cover".to_string(),
        selected_set_indices,
        selected_set_ids,
        objective: Some(objective),
        covered_elements,
        ortools_status: None,
        ortools_selected_set_indices: Vec::new(),
        ortools_selected_set_ids: Vec::new(),
        ortools_objective: None,
        ortools_covered_elements: Vec::new(),
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_set_cover_greedy(
    problem: &SetCoverProblem,
    set_masks: &[u128],
    full_mask: u128,
) -> Option<Vec<usize>> {
    let mut covered = 0_u128;
    let mut selected = Vec::new();
    while covered != full_mask {
        let mut best: Option<usize> = None;
        let mut best_ratio = f64::INFINITY;
        let mut best_new_bits = 0_u32;
        for (index, set) in problem.sets.iter().enumerate() {
            if selected.contains(&index) {
                continue;
            }
            let new_bits = (set_masks[index] & !covered).count_ones();
            if new_bits == 0 {
                continue;
            }
            let ratio = set.cost / f64::from(new_bits);
            if ratio < best_ratio - RUST_SET_COVER_EPS
                || ((ratio - best_ratio).abs() <= RUST_SET_COVER_EPS
                    && (new_bits > best_new_bits
                        || (new_bits == best_new_bits && best.is_none_or(|old| index < old))))
            {
                best = Some(index);
                best_ratio = ratio;
                best_new_bits = new_bits;
            }
        }
        let best = best?;
        selected.push(best);
        covered |= set_masks[best];
    }
    Some(selected)
}

#[allow(clippy::too_many_arguments)]
fn rust_set_cover_exact_search(
    full_mask: u128,
    covered: u128,
    current_cost: f64,
    current: &mut Vec<usize>,
    covering_sets: &[Vec<usize>],
    set_masks: &[u128],
    costs: &[f64],
    best_indices: &mut Vec<usize>,
    best_cost: &mut f64,
) {
    if current_cost >= *best_cost - RUST_SET_COVER_EPS {
        return;
    }
    if covered == full_mask {
        let mut candidate = current.clone();
        candidate.sort_unstable();
        let mut incumbent = best_indices.clone();
        incumbent.sort_unstable();
        if current_cost < *best_cost - RUST_SET_COVER_EPS
            || ((current_cost - *best_cost).abs() <= RUST_SET_COVER_EPS && candidate < incumbent)
        {
            *best_indices = candidate;
            *best_cost = current_cost;
        }
        return;
    }

    let uncovered = full_mask & !covered;
    let mut chosen_candidates: Option<Vec<usize>> = None;
    for (element_index, candidates) in covering_sets.iter().enumerate() {
        if uncovered & (1_u128 << element_index) == 0 {
            continue;
        }
        let available = candidates
            .iter()
            .copied()
            .filter(|set_index| {
                !current.contains(set_index) && (set_masks[*set_index] & !covered) != 0
            })
            .collect::<Vec<_>>();
        if chosen_candidates
            .as_ref()
            .is_none_or(|chosen| available.len() < chosen.len())
        {
            chosen_candidates = Some(available);
        }
    }

    let Some(mut chosen_candidates) = chosen_candidates else {
        return;
    };
    if chosen_candidates.is_empty() {
        return;
    }
    chosen_candidates.sort_by(|left, right| {
        costs[*left]
            .total_cmp(&costs[*right])
            .then_with(|| left.cmp(right))
    });
    for set_index in chosen_candidates {
        current.push(set_index);
        rust_set_cover_exact_search(
            full_mask,
            covered | set_masks[set_index],
            current_cost + costs[set_index],
            current,
            covering_sets,
            set_masks,
            costs,
            best_indices,
            best_cost,
        );
        current.pop();
    }
}

fn solve_set_cover_with_rust_reference(
    problem: &SetCoverProblem,
) -> ExternalSetCoverReferenceSolution {
    let started = Instant::now();
    let element_index = match validate_rust_set_cover_problem(problem) {
        Ok(element_index) => element_index,
        Err(message) => {
            return rust_set_cover_empty_solution(
                ExternalSetCoverReferenceStatus::NumericalError,
                "rust:exact-set-cover",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    if problem.sets.len() > RUST_SET_COVER_MAX_EXACT_SETS
        || problem.universe.len() > RUST_SET_COVER_MAX_EXACT_ELEMENTS
    {
        return rust_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::Unsupported,
            "rust:exact-set-cover",
            format!(
                "exact set-cover only practical for <= {RUST_SET_COVER_MAX_EXACT_SETS} sets and <= {RUST_SET_COVER_MAX_EXACT_ELEMENTS} elements, got {} sets and {} elements",
                problem.sets.len(),
                problem.universe.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let (set_masks, full_mask) = rust_set_cover_masks(problem, &element_index);
    let Some(mut best_indices) = rust_set_cover_greedy(problem, &set_masks, full_mask) else {
        return rust_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::Infeasible,
            "rust:exact-set-cover",
            "greedy could not cover remaining elements",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let mut best_cost = best_indices
        .iter()
        .map(|&index| problem.sets[index].cost)
        .sum::<f64>();

    let mut covering_sets = vec![Vec::<usize>::new(); problem.universe.len()];
    for (set_index, mask) in set_masks.iter().enumerate() {
        for element_index in 0..problem.universe.len() {
            if mask & (1_u128 << element_index) != 0 {
                covering_sets[element_index].push(set_index);
            }
        }
    }
    if covering_sets.iter().any(Vec::is_empty) {
        return rust_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::Infeasible,
            "rust:exact-set-cover",
            "at least one universe element is uncovered by all sets",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let costs = problem.sets.iter().map(|set| set.cost).collect::<Vec<_>>();
    let mut current = Vec::new();
    rust_set_cover_exact_search(
        full_mask,
        0,
        0.0,
        &mut current,
        &covering_sets,
        &set_masks,
        &costs,
        &mut best_indices,
        &mut best_cost,
    );

    rust_set_cover_solution(
        problem,
        ExternalSetCoverReferenceStatus::Optimal,
        best_indices,
        "exact branch-and-bound",
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn set_cover_reference_timeout_ms() -> u64 {
    std::env::var("SET_COVER_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_set_cover_reference_output(
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
            Err(err) => return Err(format!("failed to poll OR-Tools set-cover adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools set-cover adapter: {err}"))
}

fn run_ortools_set_cover_reference(problem: &SetCoverProblem) -> ExternalSetCoverReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_set_cover_problem(problem) {
        return ortools_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::NumericalError,
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let Some(cost_scale) = choose_ortools_cost_scale(problem) else {
        return ortools_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::Unsupported,
            "OR-Tools CP-SAT set-cover bridge requires integer-scalable costs",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let payload = ortools_set_cover_payload(problem, cost_scale);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_SET_COVER_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ortools_set_cover_empty_solution(
                ExternalSetCoverReferenceStatus::Unavailable,
                format!("failed to start OR-Tools set-cover adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return ortools_set_cover_empty_solution(
                ExternalSetCoverReferenceStatus::NumericalError,
                format!("failed to write OR-Tools set-cover adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = set_cover_reference_timeout_ms();
    let (output, timed_out) = match wait_for_set_cover_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return ortools_set_cover_empty_solution(
                ExternalSetCoverReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools set-cover adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools set-cover adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<SetCoverReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalSetCoverReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-set-cover-reference".to_string()),
            selected_set_indices: parsed.selected_set_indices.unwrap_or_default(),
            selected_set_ids: parsed.selected_sets.unwrap_or_default(),
            objective: parsed.objective,
            covered_elements: parsed.covered_elements.unwrap_or_default(),
            ortools_status: parsed.ortools_status,
            ortools_selected_set_indices: parsed.ortools_selected_set_indices.unwrap_or_default(),
            ortools_selected_set_ids: parsed.ortools_selected_sets.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            ortools_covered_elements: parsed.ortools_covered_elements.unwrap_or_default(),
            ortools_objective_bound: parsed.ortools_objective_bound.or(parsed.objective_bound),
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => ortools_set_cover_empty_solution(
            ExternalSetCoverReferenceStatus::NumericalError,
            format!("failed to parse OR-Tools set-cover adapter output: {err}; stderr={stderr}"),
            elapsed_ms,
        ),
    }
}

pub fn solve_set_cover_with_external_reference(
    problem: &SetCoverProblem,
    opts: &ExternalSetCoverReferenceOptions,
) -> ExternalSetCoverReferenceSolution {
    if should_use_rust_set_cover_reference(opts) || should_use_registered_set_cover_fallback(opts) {
        return relabel_registered_set_cover_fallback(
            solve_set_cover_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_set_cover_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::set_cover::{
        build_sample_set_cover_problem, SetCoverProblem, SetCoverSet,
    };
    use std::sync::Mutex;

    static SET_COVER_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn rust_reference_solves_sample_set_cover() {
        let problem = build_sample_set_cover_problem();
        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::RustExact,
            },
        );

        assert_eq!(solution.status, ExternalSetCoverReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-set-cover");
        assert_eq!(solution.objective, Some(7.0));
        assert_eq!(solution.selected_set_ids, vec!["A", "B", "D"]);
        assert_eq!(solution.covered_elements, problem.universe);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_infeasible_cover() {
        let problem = SetCoverProblem {
            universe: vec!["A".to_string(), "B".to_string()],
            sets: vec![SetCoverSet {
                id: "only-a".to_string(),
                cost: 1.0,
                elements: vec!["A".to_string()],
            }],
        };

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalSetCoverReferenceStatus::Infeasible);
        assert_eq!(solution.solver, "rust:exact-set-cover");
        assert!(solution.selected_set_indices.is_empty());
        assert!(solution.objective.is_none());
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_set_cover_problem();

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalSetCoverReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-set-cover");
        assert_eq!(solution.objective, Some(7.0));
        assert_eq!(solution.selected_set_ids, vec!["A", "B", "D"]);
    }

    #[test]
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = SET_COVER_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("SET_COVER_REFERENCE_REGISTERED_FALLBACK", "rust");
        let problem = build_sample_set_cover_problem();

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalSetCoverReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-set-cover-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(7.0));
        assert_eq!(solution.selected_set_ids, vec!["A", "B", "D"]);
        assert_eq!(solution.covered_elements, problem.universe);
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn rust_first_env_forces_ortools_to_rust_reference_without_python() {
        let _lock = SET_COVER_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _rust_first_guard = EnvVarGuard::set("SET_COVER_REFERENCE_RUST_FIRST", "true");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-set-cover");
        let problem = build_sample_set_cover_problem();

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalSetCoverReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-set-cover-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(7.0));
        assert_eq!(solution.selected_set_ids, vec!["A", "B", "D"]);
        assert_eq!(solution.covered_elements, problem.universe);
    }

    #[test]
    fn ortools_adapter_rejects_unscaled_costs_without_python() {
        let _lock = SET_COVER_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _fallback_guard = EnvVarGuard::set("SET_COVER_REFERENCE_REGISTERED_FALLBACK", "0");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = SetCoverProblem {
            universe: vec!["A".to_string()],
            sets: vec![SetCoverSet {
                id: "S".to_string(),
                cost: 1.0 / 3.0,
                elements: vec!["A".to_string()],
            }],
        };

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalSetCoverReferenceStatus::Unsupported
        );
        assert_eq!(solution.solver, ORTOOLS_SET_COVER_SOLVER);
        assert!(solution.message.contains("integer-scalable costs"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = SET_COVER_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _fallback_guard = EnvVarGuard::set("SET_COVER_REFERENCE_REGISTERED_FALLBACK", "0");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = build_sample_set_cover_problem();

        let solution = solve_set_cover_with_external_reference(
            &problem,
            &ExternalSetCoverReferenceOptions {
                solver: ExternalSetCoverReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalSetCoverReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, ORTOOLS_SET_COVER_SOLVER);
        assert!(solution.message.contains("OR-Tools set-cover adapter"));
        assert!(!solution.message.contains("set_cover_reference.py"));
    }

    #[test]
    fn set_cover_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_set_cover_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn set_cover_adapter_wait_observes_closed_stdin() {
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
            .write_all(
                b"{\"universe\":[\"A\"],\"sets\":[{\"id\":\"S\",\"cost\":1,\"elements\":[\"A\"]}]}",
            )
            .expect("write stdin");
        drop(child.stdin.take());

        let (output, timed_out) =
            wait_for_set_cover_reference_output(child, 1_000).expect("closed stdin output");

        assert!(!timed_out);
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "done");
    }
}
