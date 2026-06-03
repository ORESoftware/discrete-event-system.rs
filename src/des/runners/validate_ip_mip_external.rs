//! Cross-check the in-house DES IP/MIP station graph against installed external
//! solver CLIs.
//!
//! The default path uses the Rust `external_linear_cli` adapter to serialize the
//! same source model and call installed open-source/commercial CLIs. The older
//! `scripts/ip_mip_reference.py` bridge remains as an explicit fallback for
//! environments without a local solver command.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::des::general::external_linear_cli::{
    external_linear_cli_command, general_linear_ipmip_problem_to_cli_json,
    indicator_ipmip_problem_to_cli_json, ipmip_problem_to_cli_json,
    lower_bounded_ipmip_problem_to_cli_json, multi_objective_ipmip_problem_to_cli_json,
    pwl_ipmip_problem_to_cli_json, semi_ipmip_problem_to_cli_json, solve_linear_cli_json,
    sos_ipmip_problem_to_cli_json, ExternalLinearCliKind, ExternalLinearCliOptions,
    ExternalLinearCliSolution, ExternalLinearCliSolver, ExternalLinearCliStatus,
};
use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, build_fixed_charge_indicator_ip, build_general_linear_rows_ip,
    build_lexicographic_choice_ip, build_lower_bounded_production_ip,
    build_piecewise_linear_reward_ip, build_semi_continuous_gate_ip, build_semi_integer_lot_ip,
    build_sos1_choice_ip, build_sos2_adjacency_ip, linearize_general_linear_problem,
    linearize_indicator_problem, linearize_pwl_problem, linearize_semi_problem,
    linearize_sos_problem, solve_general_linear_ipmip_with_des, solve_indicator_ipmip_with_des,
    solve_ipmip_with_des, solve_lower_bounded_ipmip_with_des, solve_multi_objective_ipmip_with_des,
    solve_pwl_ipmip_with_des, solve_semi_ipmip_with_des, solve_sos_ipmip_with_des,
    ConcreteLpRelaxationAlgorithm, GeneralLinearIPMIPProblem, IPMIPProblem, IPMIPSolveOptions,
    IPMIPStatus, IndicatorIPMIPProblem, LowerBoundedIPMIPProblem, LpRelaxationAlgorithm,
    MultiObjectiveIPMIPProblem, PwlIPMIPProblem, SemiIPMIPProblem, SosIPMIPProblem,
};
use crate::des::general::lp::Sense;

#[derive(Clone, Debug, Default, Deserialize)]
struct ExternalResultInner {
    status: String,
    solver: String,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    objective_values: Option<Vec<f64>>,
    message: Option<String>,
    enumerated: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ExternalPayload {
    result: ExternalResultInner,
}

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
    root: PathBuf,
    out_dir: PathBuf,
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        self.checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    fn close(&mut self, name: &str, actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        self.check(
            name,
            diff <= tol,
            Some(format!(
                "actual={actual} expected={expected} diff={diff:.3e} tol={tol}"
            )),
        );
    }

    fn write_problem(&self, name: &str, problem: &IPMIPProblem) -> PathBuf {
        let value = ipmip_problem_to_cli_json(problem);
        self.write_problem_value(name, &value)
    }

    fn write_problem_value(&self, name: &str, value: &serde_json::Value) -> PathBuf {
        std::fs::create_dir_all(&self.out_dir).expect("create external validation output dir");
        let p = self.out_dir.join(format!("{name}-problem.json"));
        let json =
            serde_json::to_string_pretty(value).expect("serialize IP/MIP validation problem");
        std::fs::write(&p, format!("{json}\n")).expect("write IP/MIP validation problem");
        p
    }

    fn run_external(
        &mut self,
        name: &str,
        problem: &IPMIPProblem,
        solver: &str,
    ) -> ExternalPayload {
        let problem_value = ipmip_problem_to_cli_json(problem);
        self.run_external_value(name, &problem_value, solver)
    }

    fn run_external_value(
        &mut self,
        name: &str,
        problem: &serde_json::Value,
        solver: &str,
    ) -> ExternalPayload {
        let problem_path = self.write_problem_value(name, problem);
        if !force_python_reference() {
            match select_cli_solver(solver) {
                Some((cli_solver, command)) => {
                    println!(
                        "  external rust-cli: solver={} command={}",
                        cli_solver.as_str(),
                        command.display()
                    );
                    let solution = solve_linear_cli_json(
                        ExternalLinearCliKind::Mip,
                        problem.clone(),
                        &ExternalLinearCliOptions {
                            solver: cli_solver,
                            command_path: Some(command),
                            time_limit_secs: Some(10.0),
                            random_seed: Some(7),
                            threads: Some(1),
                            ..Default::default()
                        },
                    );
                    if solution.status != ExternalLinearCliStatus::Unavailable {
                        return payload_from_cli_solution(solution);
                    }
                }
                None if solver != "auto" => {
                    return ExternalPayload {
                        result: ExternalResultInner {
                            status: "unavailable".to_string(),
                            solver: solver.to_string(),
                            message: Some(format!(
                                "no installed command found for requested solver `{solver}`"
                            )),
                            ..Default::default()
                        },
                    };
                }
                None => {}
            }
        }
        self.run_python_external_path(name, problem_path, solver)
    }

    fn run_python_external_path(
        &mut self,
        name: &str,
        problem_path: PathBuf,
        solver: &str,
    ) -> ExternalPayload {
        let script = self.root.join("scripts").join("ip_mip_reference.py");
        if !script.exists() {
            return ExternalPayload {
                result: ExternalResultInner {
                    status: "unavailable".to_string(),
                    solver: solver.to_string(),
                    message: Some(format!(
                        "Python fallback script missing: {}",
                        script.display()
                    )),
                    ..Default::default()
                },
            };
        }
        let out = self.out_dir.join(format!("{name}-reference.json"));
        let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let max_enumerations =
            std::env::var("IP_MIP_MAX_ENUMERATIONS").unwrap_or_else(|_| "1000000".to_string());
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&problem_path)
            .arg("--out")
            .arg(&out)
            .arg("--solver")
            .arg(solver)
            .arg("--max-enumerations")
            .arg(max_enumerations)
            .output()
            .unwrap_or_else(|e| panic!("failed to start external IP/MIP reference: {e}"));

        println!(
            "  external command: {} {:?} --problem {:?} --out {:?} --solver {}",
            python, script, problem_path, out, solver
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.trim().is_empty() {
            println!("  external stdout: {}", stdout.trim());
        }
        if !stderr.trim().is_empty() {
            eprintln!("{}", stderr.trim());
        }
        if !output.status.success() {
            self.check(
                &format!("{name}: external reference process"),
                false,
                Some(format!("exit status {:?}", output.status.code())),
            );
            return ExternalPayload {
                result: ExternalResultInner {
                    status: "unavailable".to_string(),
                    solver: solver.to_string(),
                    message: Some(stderr.trim().to_string()),
                    ..Default::default()
                },
            };
        }
        let bytes = std::fs::read(&out).expect("read external IP/MIP reference output");
        serde_json::from_slice(&bytes).expect("parse external IP/MIP reference output")
    }

    fn compare_scenario(&mut self, name: &str, problem: IPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external = self.run_external(name, &problem, &solver);
        self.compare(name, &problem, &internal, &external);
    }

    fn compare_indicator_scenario(&mut self, name: &str, problem: IndicatorIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_indicator_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let linearized = linearize_indicator_problem(&problem);
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = indicator_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare(name, &linearized, &internal, &external);
    }

    fn compare_sos_scenario(&mut self, name: &str, problem: SosIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_sos_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let linearized = linearize_sos_problem(&problem);
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = sos_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare(name, &linearized, &internal, &external);
    }

    fn compare_semi_scenario(&mut self, name: &str, problem: SemiIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_semi_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let linearized = linearize_semi_problem(&problem);
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = semi_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare(name, &linearized, &internal, &external);
    }

    fn compare_lower_bounded_scenario(&mut self, name: &str, problem: LowerBoundedIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_lower_bounded_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = lower_bounded_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare_with_lb(name, &problem, &internal, &external);
    }

    fn compare_general_linear_scenario(&mut self, name: &str, problem: GeneralLinearIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_general_linear_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let linearized = linearize_general_linear_problem(&problem);
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = general_linear_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare(name, &linearized, &internal, &external);
    }

    fn compare_pwl_scenario(&mut self, name: &str, problem: PwlIPMIPProblem) {
        println!();
        println!("-- {name} --");
        let internal = solve_pwl_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let linearized = linearize_pwl_problem(&problem);
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = pwl_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.compare(name, &linearized, &internal, &external);
    }

    fn compare_multi_objective_scenario(
        &mut self,
        name: &str,
        problem: MultiObjectiveIPMIPProblem,
    ) {
        println!();
        println!("-- {name} --");
        let internal = solve_multi_objective_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let solver = std::env::var("IP_MIP_EXTERNAL_SOLVER").unwrap_or_else(|_| "auto".to_string());
        let external_problem = multi_objective_ipmip_problem_to_cli_json(&problem);
        let external = self.run_external_value(name, &external_problem, &solver);
        self.check(
            &format!("{name}: external reference available"),
            external.result.status != "unavailable",
            external.result.message.clone(),
        );
        self.check(
            &format!("{name}: statuses agree optimal"),
            internal.status == IPMIPStatus::Optimal && external.result.status == "optimal",
            Some(format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                external.result.status,
                external.result.solver
            )),
        );
        if external.result.status != "optimal" {
            return;
        }
        let external_values = external.result.objective_values.clone().unwrap_or_default();
        self.check(
            &format!("{name}: objective vector length"),
            internal.objective_values.len() == external_values.len(),
            Some(format!(
                "internal={:?} external={:?}",
                internal.objective_values, external_values
            )),
        );
        for i in 0..internal.objective_values.len().min(external_values.len()) {
            self.close(
                &format!("{name}: objective[{i}]"),
                internal.objective_values[i],
                external_values[i],
                1e-8,
            );
        }
        self.check(
            &format!("{name}: internal incumbent feasible"),
            feasible(&problem.base, &internal.x, 1e-8),
            Some(format!("x={}", fmt_vec(&internal.x))),
        );
        let ext_x = external.result.x.clone().unwrap_or_default();
        self.check(
            &format!("{name}: external incumbent feasible"),
            feasible(&problem.base, &ext_x, 1e-8),
            Some(format!("x={}", fmt_vec(&ext_x))),
        );
    }

    fn compare_with_lb(
        &mut self,
        name: &str,
        problem: &LowerBoundedIPMIPProblem,
        internal: &crate::des::general::ip_mip_des::IPMIPSolution,
        external: &ExternalPayload,
    ) {
        self.check(
            &format!("{name}: external reference available"),
            external.result.status != "unavailable",
            external.result.message.clone(),
        );
        self.check(
            &format!("{name}: statuses agree optimal"),
            internal.status == IPMIPStatus::Optimal && external.result.status == "optimal",
            Some(format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                external.result.status,
                external.result.solver
            )),
        );
        if external.result.status != "optimal" || external.result.objective.is_none() {
            return;
        }
        let obj = external.result.objective.unwrap();
        self.close(&format!("{name}: objective"), internal.z, obj, 1e-8);
        self.check(
            &format!("{name}: internal incumbent feasible"),
            feasible_with_lb(&problem.base, &problem.lb, &internal.x, 1e-8),
            Some(format!("x={}", fmt_vec(&internal.x))),
        );
        let ext_x = external.result.x.clone().unwrap_or_default();
        self.check(
            &format!("{name}: external incumbent feasible"),
            feasible_with_lb(&problem.base, &problem.lb, &ext_x, 1e-8),
            Some(format!("x={}", fmt_vec(&ext_x))),
        );
    }

    fn compare(
        &mut self,
        name: &str,
        problem: &IPMIPProblem,
        internal: &crate::des::general::ip_mip_des::IPMIPSolution,
        external: &ExternalPayload,
    ) {
        self.check(
            &format!("{name}: external reference available"),
            external.result.status != "unavailable",
            external.result.message.clone(),
        );
        self.check(
            &format!("{name}: statuses agree optimal"),
            internal.status == IPMIPStatus::Optimal && external.result.status == "optimal",
            Some(format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                external.result.status,
                external.result.solver
            )),
        );
        if external.result.status != "optimal" || external.result.objective.is_none() {
            return;
        }
        let obj = external.result.objective.unwrap();
        self.close(&format!("{name}: objective"), internal.z, obj, 1e-8);
        self.check(
            &format!("{name}: internal incumbent feasible"),
            feasible(problem, &internal.x, 1e-8),
            Some(format!("x={}", fmt_vec(&internal.x))),
        );
        let ext_x = external.result.x.clone().unwrap_or_default();
        self.check(
            &format!("{name}: external incumbent feasible"),
            feasible(problem, &ext_x, 1e-8),
            Some(format!(
                "x={} enumerated={}",
                fmt_vec(&ext_x),
                external
                    .result
                    .enumerated
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            )),
        );
    }
}

fn fmt_vec(x: &[f64]) -> String {
    format!(
        "[{}]",
        x.iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn feasible(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    if x.len() != p.c.len() {
        return false;
    }
    for j in 0..x.len() {
        if x[j] < -tol {
            return false;
        }
        if let Some(ub) = &p.ub {
            let u = ub[j];
            if u.is_finite() && x[j] > u + tol {
                return false;
            }
        }
        if p.integer_vars[j] && (x[j] - x[j].round()).abs() > tol {
            return false;
        }
    }
    for i in 0..p.a.len() {
        let lhs: f64 = (0..x.len()).map(|j| p.a[i][j] * x[j]).sum();
        if lhs > p.b[i] + tol {
            return false;
        }
    }
    true
}

fn feasible_with_lb(p: &IPMIPProblem, lb: &[f64], x: &[f64], tol: f64) -> bool {
    if x.len() != p.c.len() || lb.len() != p.c.len() {
        return false;
    }
    for j in 0..x.len() {
        if x[j] < lb[j] - tol {
            return false;
        }
        if let Some(ub) = &p.ub {
            let u = ub[j];
            if u.is_finite() && x[j] > u + tol {
                return false;
            }
        }
        if p.integer_vars[j] && (x[j] - x[j].round()).abs() > tol {
            return false;
        }
    }
    for i in 0..p.a.len() {
        let lhs: f64 = (0..x.len()).map(|j| p.a[i][j] * x[j]).sum();
        if lhs > p.b[i] + tol {
            return false;
        }
    }
    true
}

fn force_python_reference() -> bool {
    std::env::var("IP_MIP_EXTERNAL_BRIDGE")
        .map(|value| value.eq_ignore_ascii_case("python"))
        .unwrap_or(false)
}

fn select_cli_solver(requested: &str) -> Option<(ExternalLinearCliSolver, PathBuf)> {
    if requested == "auto" {
        return ExternalLinearCliSolver::open_source_mip()
            .iter()
            .copied()
            .find_map(|solver| {
                external_linear_cli_command(solver).map(|command| (solver, command))
            });
    }
    let solver = parse_cli_solver(requested)?;
    external_linear_cli_command(solver).map(|command| (solver, command))
}

fn parse_cli_solver(value: &str) -> Option<ExternalLinearCliSolver> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "highs" | "highs-cli" => Some(ExternalLinearCliSolver::Highs),
        "glpk" | "glpsol" | "glpsol-cli" => Some(ExternalLinearCliSolver::Glpk),
        "scip" | "scip-cli" => Some(ExternalLinearCliSolver::Scip),
        "cbc" | "coin-cbc" | "coin-or-cbc" | "cbc-cli" => Some(ExternalLinearCliSolver::Cbc),
        "lp-solve" | "lpsolve" | "lp-solve-cli" => Some(ExternalLinearCliSolver::LpSolve),
        "gurobi" | "gurobi-cl" | "gurobi-cli" => Some(ExternalLinearCliSolver::Gurobi),
        "cplex" | "cplex-cli" => Some(ExternalLinearCliSolver::Cplex),
        "xpress" | "optimizer" | "xpress-cli" => Some(ExternalLinearCliSolver::Xpress),
        "lindo" | "runlindo" | "lindoapi" | "lindo-cli" => Some(ExternalLinearCliSolver::Lindo),
        _ => None,
    }
}

fn payload_from_cli_solution(solution: ExternalLinearCliSolution) -> ExternalPayload {
    let status = solution.status.as_str().to_string();
    ExternalPayload {
        result: ExternalResultInner {
            status,
            solver: solution.solver,
            x: (!solution.x.is_empty()).then_some(solution.x),
            objective: solution.objective,
            objective_values: solution.objective_values,
            message: Some(solution.message),
            enumerated: None,
        },
    }
}

fn root_from_env() -> PathBuf {
    std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn assert_script_exists(root: &Path) {
    let script = root.join("scripts").join("ip_mip_reference.py");
    assert!(
        script.exists(),
        "external IP/MIP reference script missing: {}",
        script.display()
    );
}

/// Binary driver.
pub fn run() {
    let root = root_from_env();
    let mut d = Driver {
        checks: Vec::new(),
        out_dir: root.join("out").join("external").join("ip-mip"),
        root,
    };

    println!("IP/MIP DES: framework vs external open-source/reference bridge");
    println!("==============================================================");

    d.compare_scenario(
        "knapsack-4item",
        build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0),
    );
    d.compare_scenario(
        "cover-cut-lab",
        build_binary_knapsack_ip(vec![10.0, 10.0, 10.0], vec![2.0, 2.0, 2.0], 3.0),
    );
    d.compare_scenario(
        "integer-bounded",
        IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![2.0, 3.0]],
            b: vec![12.0],
            integer_vars: vec![true, true],
            ub: Some(vec![6.0, 6.0]),
            var_names: Some(vec!["a".to_string(), "b".to_string()]),
            con_names: Some(vec!["resource".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
    );
    d.compare_lower_bounded_scenario(
        "lower-bounded-production",
        build_lower_bounded_production_ip(),
    );
    d.compare_general_linear_scenario("general-linear-rows", build_general_linear_rows_ip());
    d.compare_indicator_scenario("fixed-charge-indicator", build_fixed_charge_indicator_ip());
    d.compare_sos_scenario("sos1-choice", build_sos1_choice_ip());
    d.compare_sos_scenario("sos2-adjacency", build_sos2_adjacency_ip());
    d.compare_semi_scenario("semi-continuous-gate", build_semi_continuous_gate_ip());
    d.compare_semi_scenario("semi-integer-lot", build_semi_integer_lot_ip());
    d.compare_pwl_scenario(
        "piecewise-linear-reward",
        build_piecewise_linear_reward_ip(),
    );
    d.compare_multi_objective_scenario("lexicographic-choice", build_lexicographic_choice_ip());

    println!();
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-ip-mip-external: {}/{} checks passed.",
        passed,
        d.checks.len()
    );
    if passed < d.checks.len() {
        println!("FAILED:");
        for c in &d.checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|x| format!(": {}", x))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}
