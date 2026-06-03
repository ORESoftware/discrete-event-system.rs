//! Scale-envelope checks for native optimization solvers against external
//! open-source engines.
//!
//! This is not a performance shootout. It is a deterministic guardrail that
//! answers the practical parity question: as generated LP/MIP instances grow
//! beyond the tiny feature tests, do the native Rust solvers still agree with
//! installed reference engines, and where do timings start to separate?

#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use crate::des::general::cp_sat::{
    solve_cp_model, CpConstraint, CpModel, CpObjective, CpSolution, CpSolveOptions, CpStatus,
    CpVariable, LinearSense, LinearTerm, ObjectiveSense,
};
use crate::des::general::external_cp_sat_reference::{
    cp_sat_model_to_reference_json, solve_cp_sat_json_with_external_reference,
    ExternalCpSatReferenceOptions, ExternalCpSatReferenceSolver,
};
use crate::des::general::external_linear_cli::{
    solve_ipmip_with_external_cli, solve_lp_with_external_cli, ExternalLinearCliModelFormat,
    ExternalLinearCliOptions, ExternalLinearCliSolver, ExternalLinearCliStatus,
};
use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, solve_ipmip_with_des, ConcreteLpRelaxationAlgorithm, IPMIPProblem,
    IPMIPSolveOptions, IPMIPStatus, LpRelaxationAlgorithm,
};
use crate::des::general::lp::{
    solve_lp_external, solve_lp_internal, ExternalSolverOptions, InternalSimplexOptions, LPProblem,
    LPStatus, Sense,
};

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Clone, Debug, Serialize)]
struct ScaleRow {
    family: String,
    size: usize,
    constraints: usize,
    native_solver: String,
    external_solver: String,
    native_status: String,
    external_status: String,
    native_objective: f64,
    external_objective: f64,
    objective_abs_diff: f64,
    native_ms: f64,
    external_ms: f64,
    native_nodes: Option<usize>,
    native_lp_solves: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct ScaleReport {
    generated_at_unix_ms: u128,
    lp_sizes: Vec<usize>,
    lp_methods: Vec<String>,
    lp_cli_solvers: Vec<String>,
    mip_sizes: Vec<usize>,
    mip_solvers: Vec<String>,
    mip_cli_solvers: Vec<String>,
    cli_formats: Vec<String>,
    cp_sizes: Vec<usize>,
    rows: Vec<ScaleRow>,
}

#[derive(Debug)]
struct CpSatReference {
    status: String,
    assignment: Vec<i64>,
    objective: Option<i64>,
    nodes: Option<usize>,
    solver: String,
    message: Option<String>,
}

struct Driver {
    root: PathBuf,
    checks: Vec<CheckRow>,
    rows: Vec<ScaleRow>,
}

impl Driver {
    fn new() -> Self {
        let root = std::env::var("REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        Driver {
            root,
            checks: Vec::new(),
            rows: Vec::new(),
        }
    }

    fn check(&mut self, name: impl Into<String>, passed: bool, detail: impl Into<String>) {
        let name = name.into();
        let detail = detail.into();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            if detail.is_empty() {
                String::new()
            } else {
                format!(" - {detail}")
            }
        );
        self.checks.push(CheckRow {
            name,
            passed,
            detail,
        });
    }

    fn close(&mut self, name: impl Into<String>, a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        self.check(
            name,
            diff <= tol,
            format!("native={a:.10} external={b:.10} diff={diff:.3e} tol={tol:.1e}"),
        );
    }

    fn run_lp_case(&mut self, n: usize, method: &str) {
        let m = (n / 2).max(2);
        let problem = build_resource_lp(n, m);
        let native_t0 = Instant::now();
        let native = solve_lp_internal(
            &problem,
            &InternalSimplexOptions {
                max_iter: Some(20_000),
                tol: Some(1e-8),
                basis_start: None,
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external = solve_lp_external(
            &problem,
            &ExternalSolverOptions {
                method: Some(method.to_string()),
                ..Default::default()
            },
        );
        let case = format!("LP n={n} method={method}");
        self.check(
            format!("{case} statuses optimal"),
            native.status == LPStatus::Optimal && external.status == LPStatus::Optimal,
            format!(
                "native={} external={} solver={}",
                native.status.as_str(),
                external.status.as_str(),
                external.solver
            ),
        );
        self.close(
            format!("{case} objective"),
            native.objective,
            external.objective,
            lp_method_objective_tolerance(method, native.objective, external.objective),
        );
        self.rows.push(ScaleRow {
            family: "lp-resource".to_string(),
            size: n,
            constraints: m,
            native_solver: native.solver,
            external_solver: external.solver,
            native_status: native.status.as_str().to_string(),
            external_status: external.status.as_str().to_string(),
            native_objective: native.objective,
            external_objective: external.objective,
            objective_abs_diff: (native.objective - external.objective).abs(),
            native_ms,
            external_ms: external.elapsed_ms,
            native_nodes: None,
            native_lp_solves: None,
        });
    }

    fn run_lp_cli_case(
        &mut self,
        n: usize,
        solver: ExternalLinearCliSolver,
        model_format: ExternalLinearCliModelFormat,
    ) {
        let m = (n / 2).max(2);
        let problem = build_resource_lp(n, m);
        let native_t0 = Instant::now();
        let native = solve_lp_internal(
            &problem,
            &InternalSimplexOptions {
                max_iter: Some(20_000),
                tol: Some(1e-8),
                basis_start: None,
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external = solve_lp_with_external_cli(
            &problem,
            &ExternalLinearCliOptions {
                solver,
                time_limit_secs: Some(10.0),
                model_format,
                ..Default::default()
            },
        );
        if external.status == ExternalLinearCliStatus::Unavailable {
            println!(
                "  SKIP  LP resource n={n} rust-cli {} {}: {}",
                solver.as_str(),
                model_format.as_str(),
                external.message
            );
            return;
        }
        let external_objective = external.objective.unwrap_or(f64::NAN);
        let case = format!(
            "LP n={n} rust-cli {} {}",
            solver.as_str(),
            model_format.as_str()
        );
        self.check(
            format!("{case} statuses optimal"),
            native.status == LPStatus::Optimal
                && external.status == ExternalLinearCliStatus::Optimal,
            format!(
                "native={} external={} solver={} message={}",
                native.status.as_str(),
                external.status.as_str(),
                external.solver,
                external.message
            ),
        );
        let objective_tol = match solver {
            ExternalLinearCliSolver::Cbc | ExternalLinearCliSolver::Clp => 1e-6,
            _ => 1e-7,
        };
        self.close(
            format!("{case} objective"),
            native.objective,
            external_objective,
            objective_tol,
        );
        self.rows.push(ScaleRow {
            family: "lp-resource-rust-cli".to_string(),
            size: n,
            constraints: m,
            native_solver: native.solver,
            external_solver: format!("{}:{}", external.solver, model_format.as_str()),
            native_status: native.status.as_str().to_string(),
            external_status: external.status.as_str().to_string(),
            native_objective: native.objective,
            external_objective,
            objective_abs_diff: (native.objective - external_objective).abs(),
            native_ms,
            external_ms: external.elapsed_ms,
            native_nodes: None,
            native_lp_solves: None,
        });
    }

    fn run_mip_case(&mut self, n: usize, solver: &str) {
        let problem = build_scale_knapsack(n);
        let native_t0 = Instant::now();
        let native = solve_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                max_nodes: Some(25_000),
                max_ticks: Some(250_000),
                ..Default::default()
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let Some(external_solver) = external_solver_from_name(solver) else {
            println!("  SKIP  MIP knapsack n={n} solver={solver}: unknown external solver");
            return;
        };
        let external = solve_ipmip_with_external_cli(
            &problem,
            &ExternalLinearCliOptions {
                solver: external_solver,
                time_limit_secs: Some(10.0),
                node_limit: Some(100_000),
                relative_gap: Some(0.0),
                threads: Some(1),
                random_seed: Some(7),
                ..Default::default()
            },
        );
        if external.status == ExternalLinearCliStatus::Unavailable {
            println!(
                "  SKIP  MIP knapsack n={n} solver={solver}: {}",
                external.message
            );
            return;
        }
        let external_objective = external.objective.unwrap_or(f64::NAN);
        let case = format!("MIP knapsack n={n} solver={solver}");
        self.check(
            format!("{case} statuses optimal"),
            native.status == IPMIPStatus::Optimal
                && external.status == ExternalLinearCliStatus::Optimal,
            format!(
                "native={} external={} solver={} message={}",
                native.status.as_str(),
                external.status.as_str(),
                external.solver,
                external.message
            ),
        );
        self.close(
            format!("{case} objective"),
            native.z,
            external_objective,
            1e-7,
        );
        self.rows.push(ScaleRow {
            family: "mip-binary-knapsack".to_string(),
            size: n,
            constraints: problem.a.len(),
            native_solver: native.solver_kind.to_string(),
            external_solver: external.solver,
            native_status: native.status.as_str().to_string(),
            external_status: external.status.as_str().to_string(),
            native_objective: native.z,
            external_objective,
            objective_abs_diff: (native.z - external_objective).abs(),
            native_ms,
            external_ms: external.elapsed_ms,
            native_nodes: Some(native.nodes_explored),
            native_lp_solves: Some(native.lp_solves),
        });
    }

    fn run_mip_cli_case(
        &mut self,
        n: usize,
        solver: ExternalLinearCliSolver,
        model_format: ExternalLinearCliModelFormat,
    ) {
        let problem = build_scale_knapsack(n);
        let native_t0 = Instant::now();
        let native = solve_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                max_nodes: Some(25_000),
                max_ticks: Some(250_000),
                ..Default::default()
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external = solve_ipmip_with_external_cli(
            &problem,
            &ExternalLinearCliOptions {
                solver,
                time_limit_secs: Some(10.0),
                node_limit: Some(100_000),
                relative_gap: Some(0.0),
                threads: Some(1),
                random_seed: Some(7),
                model_format,
                ..Default::default()
            },
        );
        if external.status == ExternalLinearCliStatus::Unavailable {
            println!(
                "  SKIP  MIP knapsack n={n} rust-cli {} {}: {}",
                solver.as_str(),
                model_format.as_str(),
                external.message
            );
            return;
        }
        let external_objective = external.objective.unwrap_or(f64::NAN);
        let case = format!(
            "MIP knapsack n={n} rust-cli {} {}",
            solver.as_str(),
            model_format.as_str()
        );
        self.check(
            format!("{case} statuses optimal"),
            native.status == IPMIPStatus::Optimal
                && external.status == ExternalLinearCliStatus::Optimal,
            format!(
                "native={} external={} solver={} message={}",
                native.status.as_str(),
                external.status.as_str(),
                external.solver,
                external.message
            ),
        );
        self.close(
            format!("{case} objective"),
            native.z,
            external_objective,
            1e-7,
        );
        self.rows.push(ScaleRow {
            family: "mip-binary-knapsack-rust-cli".to_string(),
            size: n,
            constraints: problem.a.len(),
            native_solver: native.solver_kind.to_string(),
            external_solver: format!("{}:{}", external.solver, model_format.as_str()),
            native_status: native.status.as_str().to_string(),
            external_status: external.status.as_str().to_string(),
            native_objective: native.z,
            external_objective,
            objective_abs_diff: (native.z - external_objective).abs(),
            native_ms,
            external_ms: external.elapsed_ms,
            native_nodes: Some(native.nodes_explored),
            native_lp_solves: Some(native.lp_solves),
        });
    }

    fn run_cp_case(&mut self, n: usize) {
        let model = build_cp_permutation_model(n);
        let native_t0 = Instant::now();
        let native = solve_cp_model(&model, CpSolveOptions::default());
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external_t0 = Instant::now();
        let external = self.run_cp_sat_reference(&model);
        let external_ms = external_t0.elapsed().as_secs_f64() * 1000.0;
        if external.status == "unavailable" {
            println!(
                "  SKIP  CP-SAT permutation n={n} solver={}: {}",
                external.solver,
                external
                    .message
                    .as_deref()
                    .unwrap_or("CP-SAT reference unavailable")
            );
            return;
        }
        let case = format!("CP-SAT permutation n={n}");
        self.check(
            format!("{case} statuses optimal"),
            native.status == CpStatus::Optimal && external.status == "optimal",
            format!(
                "native={} external={} solver={} native_nodes={} external_nodes={:?}",
                native.status.as_str(),
                external.status,
                external.solver,
                native.nodes,
                external.nodes
            ),
        );
        self.check(
            format!("{case} objective"),
            native.objective == external.objective,
            format!(
                "native={:?} external={:?}",
                native.objective, external.objective
            ),
        );
        self.check(
            format!("{case} assignment"),
            native.assignment == external.assignment,
            format!(
                "native={} external={}",
                compact_assignment(&native),
                compact_vec(&external.assignment)
            ),
        );
        self.rows.push(ScaleRow {
            family: "cp-sat-permutation".to_string(),
            size: n,
            constraints: model.constraints.len(),
            native_solver: native.solver,
            external_solver: external.solver,
            native_status: native.status.as_str().to_string(),
            external_status: external.status,
            native_objective: native.objective.unwrap_or_default() as f64,
            external_objective: external.objective.unwrap_or_default() as f64,
            objective_abs_diff: (native.objective.unwrap_or_default()
                - external.objective.unwrap_or_default())
            .unsigned_abs() as f64,
            native_ms,
            external_ms,
            native_nodes: Some(native.nodes),
            native_lp_solves: None,
        });
    }

    fn run_cp_sat_reference(&self, model: &CpModel) -> CpSatReference {
        let run = solve_cp_sat_json_with_external_reference(
            &cp_sat_model_to_reference_json(model),
            &ExternalCpSatReferenceOptions {
                solver: scale_cp_reference_solver(),
                ..Default::default()
            },
        );
        CpSatReference {
            status: run.status.as_str().to_string(),
            assignment: run.assignment,
            objective: run.objective.map(|objective| objective.round() as i64),
            nodes: run.nodes.and_then(|nodes| usize::try_from(nodes).ok()),
            solver: run.backend,
            message: if run.message.is_empty() {
                None
            } else {
                Some(run.message)
            },
        }
    }

    fn write_report(
        &self,
        lp_sizes: &[usize],
        lp_methods: &[String],
        lp_cli_solvers: &[ExternalLinearCliSolver],
        mip_sizes: &[usize],
        mip_solvers: &[String],
        mip_cli_solvers: &[ExternalLinearCliSolver],
        cli_formats: &[ExternalLinearCliModelFormat],
        cp_sizes: &[usize],
    ) {
        let out_dir = self
            .root
            .join("out")
            .join("external")
            .join("optimization-scale");
        std::fs::create_dir_all(&out_dir).expect("create optimization-scale output dir");
        let report = ScaleReport {
            generated_at_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before epoch")
                .as_millis(),
            lp_sizes: lp_sizes.to_vec(),
            lp_methods: lp_methods.to_vec(),
            lp_cli_solvers: external_solver_names(lp_cli_solvers),
            mip_sizes: mip_sizes.to_vec(),
            mip_solvers: mip_solvers.to_vec(),
            mip_cli_solvers: external_solver_names(mip_cli_solvers),
            cli_formats: cli_format_names(cli_formats),
            cp_sizes: cp_sizes.to_vec(),
            rows: self.rows.clone(),
        };
        let path = out_dir.join("scale-report.json");
        let json = serde_json::to_string_pretty(&report).expect("serialize scale report");
        std::fs::write(&path, format!("{json}\n")).expect("write scale report");
        println!("\nWrote scale report: {}", path.display());
    }

    fn finish(self) {
        let failed: Vec<_> = self.checks.iter().filter(|c| !c.passed).collect();
        println!(
            "\nvalidate-optimization-scale: {}/{} checks passed.",
            self.checks.len() - failed.len(),
            self.checks.len()
        );
        if !failed.is_empty() {
            eprintln!("FAILED:");
            for row in failed {
                eprintln!("  - {}: {}", row.name, row.detail);
            }
            std::process::exit(1);
        }
    }
}

fn external_solver_from_name(name: &str) -> Option<ExternalLinearCliSolver> {
    match name.trim().to_ascii_lowercase().as_str() {
        "highs" => Some(ExternalLinearCliSolver::Highs),
        "glpk" | "glpsol" => Some(ExternalLinearCliSolver::Glpk),
        "scip" => Some(ExternalLinearCliSolver::Scip),
        "cbc" => Some(ExternalLinearCliSolver::Cbc),
        "clp" => Some(ExternalLinearCliSolver::Clp),
        "gurobi" | "gurobi_cl" => Some(ExternalLinearCliSolver::Gurobi),
        "cplex" => Some(ExternalLinearCliSolver::Cplex),
        "xpress" | "optimizer" => Some(ExternalLinearCliSolver::Xpress),
        "lindo" | "runlindo" | "lindoapi" => Some(ExternalLinearCliSolver::Lindo),
        _ => None,
    }
}

fn parse_external_solver_list(
    env_name: &str,
    defaults: &[ExternalLinearCliSolver],
) -> Vec<ExternalLinearCliSolver> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.to_vec();
    };
    let values: Vec<ExternalLinearCliSolver> = raw
        .split(',')
        .filter_map(external_solver_from_name)
        .collect();
    if values.is_empty() {
        defaults.to_vec()
    } else {
        values
    }
}

fn cp_reference_solver_from_name(name: &str) -> Option<ExternalCpSatReferenceSolver> {
    let key = name.trim().to_ascii_lowercase();
    let solver = match key.as_str() {
        "rust" | "rust-enumeration" | "rust_enumeration" => {
            ExternalCpSatReferenceSolver::RustEnumeration
        }
        "python" | "fallback" | "python-enumeration" | "python_enumeration" => {
            ExternalCpSatReferenceSolver::PythonEnumeration
        }
        "ortools" | "ortools-cp-sat" | "ortools_cp_sat" | "cp-sat" | "cp_sat" => {
            ExternalCpSatReferenceSolver::OrToolsCpSat
        }
        "auto" => ExternalCpSatReferenceSolver::Auto,
        _ => ExternalCpSatReferenceSolver::all()
            .iter()
            .copied()
            .find(|solver| solver.as_arg() == key)?,
    };
    if solver.supports_cp_sat_json() {
        Some(solver)
    } else {
        None
    }
}

fn scale_cp_reference_solver() -> ExternalCpSatReferenceSolver {
    std::env::var("SCALE_CP_SOLVER")
        .ok()
        .as_deref()
        .and_then(cp_reference_solver_from_name)
        .unwrap_or(ExternalCpSatReferenceSolver::RustEnumeration)
}

fn model_format_from_name(name: &str) -> Option<ExternalLinearCliModelFormat> {
    match name.trim().to_ascii_lowercase().as_str() {
        "lp" | "cplex-lp" | "cplex_lp" => Some(ExternalLinearCliModelFormat::CplexLp),
        "mps" => Some(ExternalLinearCliModelFormat::Mps),
        _ => None,
    }
}

fn parse_model_format_list(
    env_name: &str,
    defaults: &[ExternalLinearCliModelFormat],
) -> Vec<ExternalLinearCliModelFormat> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.to_vec();
    };
    let values: Vec<ExternalLinearCliModelFormat> =
        raw.split(',').filter_map(model_format_from_name).collect();
    if values.is_empty() {
        defaults.to_vec()
    } else {
        values
    }
}

fn external_solver_names(solvers: &[ExternalLinearCliSolver]) -> Vec<String> {
    solvers
        .iter()
        .map(|solver| solver.as_str().to_string())
        .collect()
}

fn cli_format_names(formats: &[ExternalLinearCliModelFormat]) -> Vec<String> {
    formats
        .iter()
        .map(|format| format.as_str().to_string())
        .collect()
}

fn build_resource_lp(n: usize, m: usize) -> LPProblem {
    let ub_value = 6.0;
    let mut a_ub = Vec::with_capacity(m);
    let mut b_ub = Vec::with_capacity(m);
    for i in 0..m {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let coef = if (i + 2 * j) % 5 == 0 {
                0.0
            } else {
                1.0 + ((7 * i + 11 * j + 3) % 13) as f64 / 3.0
            };
            row.push(coef);
        }
        let full_use: f64 = row.iter().map(|a| a * ub_value).sum();
        b_ub.push(full_use * (0.28 + 0.02 * (i % 4) as f64));
        a_ub.push(row);
    }
    LPProblem {
        sense: Sense::Max,
        c: (0..n)
            .map(|j| 1.0 + ((17 * j + 5) % 19) as f64 / 4.0)
            .collect(),
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: None,
        b_eq: None,
        lb: Some(vec![Some(0.0); n]),
        ub: Some(vec![Some(ub_value); n]),
        var_names: Some((0..n).map(|j| format!("x{j}")).collect()),
        con_names: Some((0..m).map(|i| format!("resource_{i}")).collect()),
    }
}

fn build_scale_knapsack(n: usize) -> IPMIPProblem {
    let values: Vec<f64> = (0..n)
        .map(|j| 8.0 + ((37 * j + 13) % 41) as f64 + (j % 3) as f64 * 0.25)
        .collect();
    let weights: Vec<f64> = (0..n).map(|j| 3.0 + ((19 * j + 7) % 23) as f64).collect();
    let capacity = weights.iter().sum::<f64>() * 0.42;
    build_binary_knapsack_ip(values, weights, capacity)
}

fn build_cp_permutation_model(n: usize) -> CpModel {
    let n = n.max(2);
    let vars: Vec<usize> = (0..n).collect();
    let sum_rhs = (n * (n - 1) / 2) as i64;
    CpModel {
        variables: (0..n)
            .map(|j| CpVariable {
                name: format!("p{j}"),
                domain: (0..n as i64).collect(),
            })
            .collect(),
        constraints: vec![
            CpConstraint::AllDifferent(vars.clone()),
            CpConstraint::Linear {
                terms: vars
                    .iter()
                    .map(|&var| LinearTerm { var, coeff: 1 })
                    .collect(),
                sense: LinearSense::Eq,
                rhs: sum_rhs,
            },
        ],
        objective: Some(CpObjective {
            sense: ObjectiveSense::Max,
            terms: vars
                .iter()
                .map(|&var| LinearTerm {
                    var,
                    coeff: var as i64 + 1,
                })
                .collect(),
        }),
    }
}

fn compact_assignment(solution: &CpSolution) -> String {
    compact_vec(&solution.assignment)
}

fn compact_vec(values: &[i64]) -> String {
    if values.len() <= 12 {
        return format!("{values:?}");
    }
    let mut preview: Vec<String> = values.iter().take(8).map(ToString::to_string).collect();
    preview.push("...".to_string());
    preview.extend(values.iter().rev().take(3).rev().map(ToString::to_string));
    format!("[{}] len={}", preview.join(", "), values.len())
}

fn parse_size_list(env_name: &str, defaults: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.to_vec();
    };
    let values: Vec<usize> = raw
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect();
    if values.is_empty() {
        defaults.to_vec()
    } else {
        values
    }
}

fn parse_solver_list(env_name: &str, defaults: &[&str]) -> Vec<String> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.iter().map(|s| (*s).to_string()).collect();
    };
    let values: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if values.is_empty() {
        defaults.iter().map(|s| (*s).to_string()).collect()
    } else {
        values
    }
}

fn lp_method_objective_tolerance(
    method: &str,
    native_objective: f64,
    external_objective: f64,
) -> f64 {
    if method.trim().to_ascii_lowercase().contains("pdlp") {
        1e-6 * native_objective
            .abs()
            .max(external_objective.abs())
            .max(1.0)
    } else {
        1e-7
    }
}

pub fn run() {
    println!("Optimization scale envelope: native solvers vs external engines");
    println!("===============================================================");

    let lp_sizes = parse_size_list("SCALE_LP_SIZES", &[8, 16, 24]);
    let mip_sizes = parse_size_list("SCALE_MIP_SIZES", &[8, 12, 16]);
    let cp_sizes = parse_size_list("SCALE_CP_SIZES", &[5, 6, 7]);
    let lp_methods = parse_solver_list("SCALE_LP_METHODS", &["highs", "glop", "pdlp"]);
    let mip_solvers = parse_solver_list("SCALE_MIP_SOLVERS", &["highs", "glpk", "scip", "cbc"]);
    let lp_cli_solvers = parse_external_solver_list(
        "SCALE_LP_CLI_SOLVERS",
        ExternalLinearCliSolver::open_source_lp(),
    );
    let mip_cli_solvers = parse_external_solver_list(
        "SCALE_MIP_CLI_SOLVERS",
        ExternalLinearCliSolver::open_source_mip(),
    );
    let cli_formats =
        parse_model_format_list("SCALE_CLI_FORMATS", &[ExternalLinearCliModelFormat::Mps]);

    let mut driver = Driver::new();

    println!("\n-- LP resource family --");
    for &n in &lp_sizes {
        for method in &lp_methods {
            driver.run_lp_case(n, method);
        }
        for &solver in &lp_cli_solvers {
            for &model_format in &cli_formats {
                driver.run_lp_cli_case(n, solver, model_format);
            }
        }
    }

    println!("\n-- MIP binary-knapsack family --");
    for &n in &mip_sizes {
        for solver in &mip_solvers {
            driver.run_mip_case(n, solver);
        }
        for &solver in &mip_cli_solvers {
            for &model_format in &cli_formats {
                driver.run_mip_cli_case(n, solver, model_format);
            }
        }
    }

    println!("\n-- CP-SAT permutation family --");
    for &n in &cp_sizes {
        driver.run_cp_case(n);
    }

    driver.write_report(
        &lp_sizes,
        &lp_methods,
        &lp_cli_solvers,
        &mip_sizes,
        &mip_solvers,
        &mip_cli_solvers,
        &cli_formats,
        &cp_sizes,
    );
    driver.finish();
}
