//! Unified optimisation cross-check harness.
//!
//! Runs representative same-input comparisons across the native solvers and the
//! source-only external/reference bridges:
//! LP, IP/MIP, min-cost flow, convex QP, and CP-SAT-style finite-domain models.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::Deserialize;

use crate::des::general::cp_sat::{
    solve_cp_model, BoolLiteral, CpConstraint, CpDemandInterval, CpElement, CpInterval, CpModel,
    CpObjective, CpSolveOptions, CpStatus, CpVariable, LinearSense, LinearTerm, ObjectiveSense,
};
use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, build_fixed_charge_indicator_ip, build_general_linear_rows_ip,
    build_lexicographic_choice_ip, build_lower_bounded_production_ip,
    build_piecewise_linear_reward_ip, build_semi_continuous_gate_ip, build_semi_integer_lot_ip,
    build_sos1_choice_ip, build_sos2_adjacency_ip, linearize_indicator_problem,
    linearize_pwl_problem, linearize_semi_problem, linearize_sos_problem,
    solve_general_linear_ipmip_with_des, solve_indicator_ipmip_with_des, solve_ipmip_with_des,
    solve_lower_bounded_ipmip_with_des, solve_multi_objective_ipmip_with_des,
    solve_pwl_ipmip_with_des, solve_semi_ipmip_with_des, solve_sos_ipmip_with_des,
    ConcreteLpRelaxationAlgorithm, IPMIPSolveOptions, IPMIPStatus, LpRelaxationAlgorithm,
};
use crate::des::general::lp::{
    solve_lp_external, solve_lp_internal, ExternalSolverOptions, InternalSimplexOptions, LPProblem,
    LPStatus, Sense,
};
use crate::des::general::min_cost_flow::{
    min_cost_flow_to_lp, solve_min_cost_flow, MinCostFlowArc, MinCostFlowProblem, MinCostFlowStatus,
};
use crate::des::general::qp::{
    solve_qcp_pattern_search, solve_qp_active_set, solve_socp_pattern_search, QPOptions, QPStatus,
    QcpOptions, QcpStatus, QuadraticConstraint, QuadraticProgram, QuadraticallyConstrainedProgram,
    SecondOrderCone, SecondOrderConeProgram, SocpOptions, SocpStatus,
};

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Default)]
struct Driver {
    checks: Vec<CheckRow>,
    root: PathBuf,
}

#[derive(Debug, Deserialize)]
struct MipReferenceInner {
    status: String,
    solver: String,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    objective_values: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct MipReference {
    result: MipReferenceInner,
}

#[derive(Debug, Deserialize)]
struct QPReference {
    status: String,
    solver: String,
    x: Vec<f64>,
    objective: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CpReference {
    status: String,
    solver: String,
    assignment: Vec<i64>,
    objective: Option<i64>,
}

impl Driver {
    fn new() -> Self {
        let root = std::env::var("REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        Driver {
            checks: Vec::new(),
            root,
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

    fn close(&mut self, name: &str, a: f64, b: f64, tol: f64) {
        self.check(
            name,
            (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs()),
            format!("a={a:.10} b={b:.10} diff={:.3e}", (a - b).abs()),
        );
    }

    fn max_abs_close(&mut self, name: &str, a: &[f64], b: &[f64], tol: f64) {
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        self.check(
            name,
            a.len() == b.len() && max_abs <= tol,
            format!("max_abs={max_abs:.3e}"),
        );
    }

    fn run_python_json(&self, script: &str, args: &[&str], stdin_json: &str) -> serde_json::Value {
        let path = self.root.join("scripts").join(script);
        let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let mut child = Command::new(&python)
            .arg(&path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to start {script}: {e}"));
        {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .expect("python stdin")
                .write_all(stdin_json.as_bytes())
                .expect("write python stdin");
        }
        let out = child
            .wait_with_output()
            .unwrap_or_else(|e| panic!("failed to wait for {script}: {e}"));
        if !out.status.success() {
            panic!(
                "{script} failed with {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("parse {script} stdout as JSON: {e}"))
    }

    fn validate_lp(&mut self) {
        println!("\n-- LP: internal simplex vs external LP bridge --");
        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
            b_ub: Some(vec![4.0, 6.0]),
            ..Default::default()
        };
        let internal = solve_lp_internal(&lp, &InternalSimplexOptions::default());
        let external = solve_lp_external(
            &lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..Default::default()
            },
        );
        self.check(
            "LP statuses optimal",
            internal.status == LPStatus::Optimal && external.status == LPStatus::Optimal,
            format!(
                "internal={:?} external={:?}",
                internal.status, external.status
            ),
        );
        self.close("LP objective", internal.objective, external.objective, 1e-9);
        self.max_abs_close("LP x", &internal.x, &external.x, 1e-8);
    }

    fn validate_ip_mip(&mut self) {
        println!("\n-- IP/MIP: DES branch-and-cut vs external MIP bridge --");
        let p =
            build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        let internal = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let out_dir = self
            .root
            .join("out")
            .join("external")
            .join("optimization-suite");
        std::fs::create_dir_all(&out_dir).expect("create optimization-suite out dir");
        let problem_path = out_dir.join("knapsack-problem.json");
        let reference_path = out_dir.join("knapsack-reference.json");
        let problem_json = serde_json::json!({
            "sense": p.sense.as_str(),
            "c": p.c,
            "a": p.a,
            "b": p.b,
            "integer_vars": p.integer_vars,
            "ub": p.ub,
            "var_names": p.var_names,
            "con_names": p.con_names,
        });
        std::fs::write(
            &problem_path,
            serde_json::to_string_pretty(&problem_json).expect("serialize MIP problem"),
        )
        .expect("write MIP problem");
        let script = self.root.join("scripts").join("ip_mip_reference.py");
        let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&problem_path)
            .arg("--out")
            .arg(&reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run MIP reference");
        if !output.status.success() {
            panic!(
                "ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let reference: MipReference = serde_json::from_slice(
            &std::fs::read(&reference_path).expect("read MIP reference JSON"),
        )
        .expect("parse MIP reference JSON");
        self.check(
            "IP/MIP statuses optimal",
            internal.status == IPMIPStatus::Optimal && reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                reference.result.status,
                reference.result.solver
            ),
        );
        self.close(
            "IP/MIP objective",
            internal.z,
            reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = reference.result.x.as_deref() {
            self.check("IP/MIP external x length", x.len() == internal.x.len(), "");
        }

        let lower_problem = build_lower_bounded_production_ip();
        let lower_internal = solve_lower_bounded_ipmip_with_des(
            lower_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let lower_problem_path = out_dir.join("lower-bounded-production-problem.json");
        let lower_reference_path = out_dir.join("lower-bounded-production-reference.json");
        let base = &lower_problem.base;
        let lower_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &lower_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
        });
        std::fs::write(
            &lower_problem_path,
            serde_json::to_string_pretty(&lower_json).expect("serialize lower-bounded MIP problem"),
        )
        .expect("write lower-bounded MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&lower_problem_path)
            .arg("--out")
            .arg(&lower_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run lower-bounded MIP reference");
        if !output.status.success() {
            panic!(
                "lower-bounded ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let lower_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&lower_reference_path).expect("read lower-bounded MIP reference JSON"),
        )
        .expect("parse lower-bounded MIP reference JSON");
        self.check(
            "IP/MIP lower-bounded statuses optimal",
            lower_internal.status == IPMIPStatus::Optimal
                && lower_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                lower_internal.status.as_str(),
                lower_reference.result.status,
                lower_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP lower-bounded objective",
            lower_internal.z,
            lower_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = lower_reference.result.x.as_deref() {
            self.max_abs_close("IP/MIP lower-bounded x", &lower_internal.x, x, 1e-8);
        }

        let general_problem = build_general_linear_rows_ip();
        let general_internal = solve_general_linear_ipmip_with_des(
            general_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let general_problem_path = out_dir.join("general-linear-rows-problem.json");
        let general_reference_path = out_dir.join("general-linear-rows-reference.json");
        let base = &general_problem.base;
        let general_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "linear_constraints": general_problem.linear_constraints.iter().map(|constraint| serde_json::json!({
                "coefs": &constraint.coefs,
                "lower": constraint.lower,
                "upper": constraint.upper,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &general_problem_path,
            serde_json::to_string_pretty(&general_json)
                .expect("serialize general-linear MIP problem"),
        )
        .expect("write general-linear MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&general_problem_path)
            .arg("--out")
            .arg(&general_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run general-linear MIP reference");
        if !output.status.success() {
            panic!(
                "general-linear ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let general_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&general_reference_path)
                .expect("read general-linear MIP reference JSON"),
        )
        .expect("parse general-linear MIP reference JSON");
        self.check(
            "IP/MIP general-linear statuses optimal",
            general_internal.status == IPMIPStatus::Optimal
                && general_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                general_internal.status.as_str(),
                general_reference.result.status,
                general_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP general-linear objective",
            general_internal.z,
            general_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = general_reference.result.x.as_deref() {
            self.max_abs_close("IP/MIP general-linear x", &general_internal.x, x, 1e-8);
        }

        let indicator = build_fixed_charge_indicator_ip();
        let linearized_indicator = linearize_indicator_problem(&indicator);
        let indicator_internal = solve_indicator_ipmip_with_des(
            indicator.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let indicator_problem_path = out_dir.join("fixed-charge-indicator-problem.json");
        let indicator_reference_path = out_dir.join("fixed-charge-indicator-reference.json");
        let base = &indicator.base;
        let indicator_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "indicators": indicator.indicators.iter().map(|ind| serde_json::json!({
                "binary_var": ind.binary_var,
                "active_value": ind.active_value,
                "coefs": &ind.coefs,
                "sense": ind.sense.as_str(),
                "rhs": ind.rhs,
                "name": &ind.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &indicator_problem_path,
            serde_json::to_string_pretty(&indicator_json).expect("serialize indicator MIP problem"),
        )
        .expect("write indicator MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&indicator_problem_path)
            .arg("--out")
            .arg(&indicator_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run indicator MIP reference");
        if !output.status.success() {
            panic!(
                "indicator ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let indicator_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&indicator_reference_path).expect("read indicator MIP reference JSON"),
        )
        .expect("parse indicator MIP reference JSON");
        self.check(
            "IP/MIP indicator statuses optimal",
            indicator_internal.status == IPMIPStatus::Optimal
                && indicator_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                indicator_internal.status.as_str(),
                indicator_reference.result.status,
                indicator_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP indicator objective",
            indicator_internal.z,
            indicator_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = indicator_reference.result.x.as_deref() {
            self.check(
                "IP/MIP indicator external x length",
                x.len() == linearized_indicator.c.len(),
                "",
            );
        }

        for (case_name, sos_problem) in [
            ("sos1-choice", build_sos1_choice_ip()),
            ("sos2-adjacency", build_sos2_adjacency_ip()),
        ] {
            let linearized_sos = linearize_sos_problem(&sos_problem);
            let sos_internal = solve_sos_ipmip_with_des(
                sos_problem.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                    )),
                    max_cut_rounds: Some(1),
                    ..Default::default()
                },
            );
            let sos_problem_path = out_dir.join(format!("{case_name}-problem.json"));
            let sos_reference_path = out_dir.join(format!("{case_name}-reference.json"));
            let base = &sos_problem.base;
            let sos_json = serde_json::json!({
                "sense": base.sense.as_str(),
                "c": &base.c,
                "a": &base.a,
                "b": &base.b,
                "integer_vars": &base.integer_vars,
                "ub": &base.ub,
                "var_names": &base.var_names,
                "con_names": &base.con_names,
                "sos": sos_problem.sos.iter().map(|set| serde_json::json!({
                    "kind": set.kind.as_str(),
                    "vars": &set.vars,
                    "weights": &set.weights,
                    "name": &set.name,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(
                &sos_problem_path,
                serde_json::to_string_pretty(&sos_json).expect("serialize SOS MIP problem"),
            )
            .expect("write SOS MIP problem");
            let output = Command::new(&python)
                .arg(&script)
                .arg("--problem")
                .arg(&sos_problem_path)
                .arg("--out")
                .arg(&sos_reference_path)
                .arg("--solver")
                .arg("auto")
                .output()
                .expect("run SOS MIP reference");
            if !output.status.success() {
                panic!(
                    "SOS ip_mip_reference failed for {case_name}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let sos_reference: MipReference = serde_json::from_slice(
                &std::fs::read(&sos_reference_path).expect("read SOS MIP reference JSON"),
            )
            .expect("parse SOS MIP reference JSON");
            self.check(
                &format!("IP/MIP {case_name} statuses optimal"),
                sos_internal.status == IPMIPStatus::Optimal
                    && sos_reference.result.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    sos_internal.status.as_str(),
                    sos_reference.result.status,
                    sos_reference.result.solver
                ),
            );
            self.close(
                &format!("IP/MIP {case_name} objective"),
                sos_internal.z,
                sos_reference.result.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            if let Some(x) = sos_reference.result.x.as_deref() {
                self.check(
                    &format!("IP/MIP {case_name} external x length"),
                    x.len() == linearized_sos.c.len(),
                    "",
                );
            }
        }

        for (case_name, semi_problem) in [
            ("semi-continuous-gate", build_semi_continuous_gate_ip()),
            ("semi-integer-lot", build_semi_integer_lot_ip()),
        ] {
            let linearized_semi = linearize_semi_problem(&semi_problem);
            let semi_internal = solve_semi_ipmip_with_des(
                semi_problem.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                    )),
                    max_cut_rounds: Some(1),
                    ..Default::default()
                },
            );
            let semi_problem_path = out_dir.join(format!("{case_name}-problem.json"));
            let semi_reference_path = out_dir.join(format!("{case_name}-reference.json"));
            let base = &semi_problem.base;
            let semi_json = serde_json::json!({
                "sense": base.sense.as_str(),
                "c": &base.c,
                "a": &base.a,
                "b": &base.b,
                "integer_vars": &base.integer_vars,
                "ub": &base.ub,
                "var_names": &base.var_names,
                "con_names": &base.con_names,
                "semi_variables": semi_problem.semi_variables.iter().map(|semi| serde_json::json!({
                    "kind": semi.kind.as_str(),
                    "var": semi.var,
                    "lower": semi.lower,
                    "name": &semi.name,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(
                &semi_problem_path,
                serde_json::to_string_pretty(&semi_json)
                    .expect("serialize semi-variable MIP problem"),
            )
            .expect("write semi-variable MIP problem");
            let output = Command::new(&python)
                .arg(&script)
                .arg("--problem")
                .arg(&semi_problem_path)
                .arg("--out")
                .arg(&semi_reference_path)
                .arg("--solver")
                .arg("auto")
                .output()
                .expect("run semi-variable MIP reference");
            if !output.status.success() {
                panic!(
                    "semi-variable ip_mip_reference failed for {case_name}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let semi_reference: MipReference = serde_json::from_slice(
                &std::fs::read(&semi_reference_path)
                    .expect("read semi-variable MIP reference JSON"),
            )
            .expect("parse semi-variable MIP reference JSON");
            self.check(
                &format!("IP/MIP {case_name} statuses optimal"),
                semi_internal.status == IPMIPStatus::Optimal
                    && semi_reference.result.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    semi_internal.status.as_str(),
                    semi_reference.result.status,
                    semi_reference.result.solver
                ),
            );
            self.close(
                &format!("IP/MIP {case_name} objective"),
                semi_internal.z,
                semi_reference.result.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            if let Some(x) = semi_reference.result.x.as_deref() {
                self.check(
                    &format!("IP/MIP {case_name} external x length"),
                    x.len() == linearized_semi.c.len(),
                    "",
                );
            }
        }

        let pwl_problem = build_piecewise_linear_reward_ip();
        let linearized_pwl = linearize_pwl_problem(&pwl_problem);
        let pwl_internal = solve_pwl_ipmip_with_des(
            pwl_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let pwl_problem_path = out_dir.join("piecewise-linear-reward-problem.json");
        let pwl_reference_path = out_dir.join("piecewise-linear-reward-reference.json");
        let base = &pwl_problem.base;
        let pwl_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "pwl": pwl_problem.pwl.iter().map(|pwl| serde_json::json!({
                "x_var": pwl.x_var,
                "y_var": pwl.y_var,
                "points": pwl.points.iter().map(|point| serde_json::json!({
                    "x": point.x,
                    "y": point.y,
                })).collect::<Vec<_>>(),
                "name": &pwl.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &pwl_problem_path,
            serde_json::to_string_pretty(&pwl_json).expect("serialize PWL MIP problem"),
        )
        .expect("write PWL MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&pwl_problem_path)
            .arg("--out")
            .arg(&pwl_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run PWL MIP reference");
        if !output.status.success() {
            panic!(
                "PWL ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let pwl_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&pwl_reference_path).expect("read PWL MIP reference JSON"),
        )
        .expect("parse PWL MIP reference JSON");
        self.check(
            "IP/MIP piecewise-linear-reward statuses optimal",
            pwl_internal.status == IPMIPStatus::Optimal && pwl_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                pwl_internal.status.as_str(),
                pwl_reference.result.status,
                pwl_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP piecewise-linear-reward objective",
            pwl_internal.z,
            pwl_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = pwl_reference.result.x.as_deref() {
            self.check(
                "IP/MIP piecewise-linear-reward external x length",
                x.len() == linearized_pwl.c.len(),
                "",
            );
        }

        let multi_problem = build_lexicographic_choice_ip();
        let multi_internal = solve_multi_objective_ipmip_with_des(
            multi_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let multi_problem_path = out_dir.join("lexicographic-choice-problem.json");
        let multi_reference_path = out_dir.join("lexicographic-choice-reference.json");
        let base = &multi_problem.base;
        let multi_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "multi_objectives": multi_problem.objectives.iter().map(|objective| serde_json::json!({
                "sense": objective.sense.as_str(),
                "c": &objective.c,
                "name": &objective.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &multi_problem_path,
            serde_json::to_string_pretty(&multi_json)
                .expect("serialize multi-objective MIP problem"),
        )
        .expect("write multi-objective MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&multi_problem_path)
            .arg("--out")
            .arg(&multi_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run multi-objective MIP reference");
        if !output.status.success() {
            panic!(
                "multi-objective ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let multi_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&multi_reference_path).expect("read multi-objective MIP reference JSON"),
        )
        .expect("parse multi-objective MIP reference JSON");
        self.check(
            "IP/MIP lexicographic-choice statuses optimal",
            multi_internal.status == IPMIPStatus::Optimal
                && multi_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                multi_internal.status.as_str(),
                multi_reference.result.status,
                multi_reference.result.solver
            ),
        );
        let external_values = multi_reference
            .result
            .objective_values
            .clone()
            .unwrap_or_default();
        self.check(
            "IP/MIP lexicographic-choice objective vector length",
            multi_internal.objective_values.len() == external_values.len(),
            format!(
                "internal={:?} external={:?}",
                multi_internal.objective_values, external_values
            ),
        );
        for i in 0..multi_internal
            .objective_values
            .len()
            .min(external_values.len())
        {
            self.close(
                &format!("IP/MIP lexicographic-choice objective[{i}]"),
                multi_internal.objective_values[i],
                external_values[i],
                1e-9,
            );
        }
    }

    fn validate_min_cost_flow(&mut self) {
        println!("\n-- Min-cost flow: native network solver vs external LP bridge --");
        let p = MinCostFlowProblem {
            num_nodes: 4,
            supplies: vec![5.0, 7.0, -6.0, -6.0],
            arcs: vec![
                MinCostFlowArc {
                    from: 0,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 2.0,
                    name: Some("s0_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 0,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 4.0,
                    name: Some("s0_d1".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 6.0,
                    cost: 5.0,
                    name: Some("s1_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 8.0,
                    cost: 1.0,
                    name: Some("s1_d1".to_string()),
                },
            ],
        };
        let flow = solve_min_cost_flow(p.clone());
        let lp = min_cost_flow_to_lp(&p);
        let external = solve_lp_external(
            &lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..Default::default()
            },
        );
        self.check(
            "Min-cost-flow statuses optimal",
            flow.status == MinCostFlowStatus::Optimal && external.status == LPStatus::Optimal,
            format!("flow={:?} external={:?}", flow.status, external.status),
        );
        self.close(
            "Min-cost-flow objective",
            flow.total_cost,
            external.objective,
            1e-8,
        );
    }

    fn sample_qp(&self) -> QuadraticProgram {
        QuadraticProgram {
            q: vec![vec![2.0, 0.5], vec![0.5, 2.0]],
            c: vec![-5.0, -6.0],
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![3.0]),
            lb: Some(vec![Some(0.0), Some(0.0)]),
            ub: Some(vec![Some(4.0), Some(4.0)]),
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..Default::default()
        }
    }

    fn sample_socp(&self) -> SecondOrderConeProgram {
        SecondOrderConeProgram {
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
        }
    }

    fn sample_qcp(&self) -> QuadraticallyConstrainedProgram {
        QuadraticallyConstrainedProgram {
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
        }
    }

    fn validate_qp(&mut self) {
        println!("\n-- QP: active-set solver vs QP reference bridge --");
        let qp = self.sample_qp();
        let internal = solve_qp_active_set(&qp, QPOptions::default());
        let qp_json = serde_json::json!({
            "Q": qp.q,
            "c": qp.c,
            "A_ub": qp.a_ub,
            "b_ub": qp.b_ub,
            "A_eq": qp.a_eq,
            "b_eq": qp.b_eq,
            "lb": qp.lb,
            "ub": qp.ub,
        })
        .to_string();
        let value = self.run_python_json("qp_reference.py", &["--solver", "auto"], &qp_json);
        let reference: QPReference = serde_json::from_value(value).expect("parse QP reference");
        self.check(
            "QP statuses optimal",
            internal.status == QPStatus::Optimal && reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                reference.status,
                reference.solver
            ),
        );
        self.close(
            "QP objective",
            internal.objective,
            reference.objective.unwrap_or(f64::NAN),
            1e-8,
        );
        self.max_abs_close("QP x", &internal.x, &reference.x, 1e-8);

        let socp = self.sample_socp();
        let socp_internal = solve_socp_pattern_search(&socp, SocpOptions::default());
        let socp_json = serde_json::json!({
            "c": &socp.c,
            "A_ub": &socp.a_ub,
            "b_ub": &socp.b_ub,
            "A_eq": &socp.a_eq,
            "b_eq": &socp.b_eq,
            "lb": &socp.lb,
            "ub": &socp.ub,
            "cones": socp.cones.iter().map(|cone| serde_json::json!({
                "A": &cone.a,
                "b": &cone.b,
                "c": &cone.c,
                "d": cone.d,
                "name": &cone.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();
        let value = self.run_python_json("qp_reference.py", &["--solver", "auto"], &socp_json);
        let socp_reference: QPReference =
            serde_json::from_value(value).expect("parse SOCP reference");
        self.check(
            "SOCP statuses optimal",
            socp_internal.status == SocpStatus::Optimal && socp_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                socp_internal.status.as_str(),
                socp_reference.status,
                socp_reference.solver
            ),
        );
        self.close(
            "SOCP objective",
            socp_internal.objective,
            socp_reference.objective.unwrap_or(f64::NAN),
            1e-6,
        );
        self.max_abs_close("SOCP x", &socp_internal.x, &socp_reference.x, 1e-6);

        let qcp = self.sample_qcp();
        let qcp_internal = solve_qcp_pattern_search(&qcp, QcpOptions::default());
        let qcp_json = serde_json::json!({
            "Q": &qcp.q,
            "c": &qcp.c,
            "A_ub": &qcp.a_ub,
            "b_ub": &qcp.b_ub,
            "A_eq": &qcp.a_eq,
            "b_eq": &qcp.b_eq,
            "lb": &qcp.lb,
            "ub": &qcp.ub,
            "quadratic_constraints": qcp.quadratic_constraints.iter().map(|constraint| serde_json::json!({
                "Q": &constraint.q,
                "c": &constraint.c,
                "rhs": constraint.rhs,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();
        let value = self.run_python_json("qp_reference.py", &["--solver", "auto"], &qcp_json);
        let qcp_reference: QPReference =
            serde_json::from_value(value).expect("parse QCP reference");
        self.check(
            "QCP statuses optimal",
            qcp_internal.status == QcpStatus::Optimal && qcp_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                qcp_internal.status.as_str(),
                qcp_reference.status,
                qcp_reference.solver
            ),
        );
        self.close(
            "QCP objective",
            qcp_internal.objective,
            qcp_reference.objective.unwrap_or(f64::NAN),
            1e-6,
        );
        self.max_abs_close("QCP x", &qcp_internal.x, &qcp_reference.x, 1e-6);
    }

    fn sample_cp_model(&self) -> CpModel {
        CpModel {
            variables: vec![
                CpVariable {
                    name: "slot_a".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "slot_b".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "slot_c".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "use_bonus".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "machine_a_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "machine_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "machine_c_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "route_index".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_cost".to_string(),
                    domain: vec![3, 8],
                },
                CpVariable {
                    name: "handoff_mode".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "handler".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "expedite".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "service_level".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
            ],
            constraints: vec![
                CpConstraint::AllDifferent(vec![0, 1, 2]),
                CpConstraint::Linear {
                    terms: vec![
                        LinearTerm { var: 0, coeff: 1 },
                        LinearTerm { var: 1, coeff: 1 },
                    ],
                    sense: LinearSense::Ge,
                    rhs: 1,
                },
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 3,
                    positive: true,
                }]),
                CpConstraint::NoOverlap(vec![
                    CpInterval {
                        start: 4,
                        duration: 3,
                        name: Some("task_a".to_string()),
                    },
                    CpInterval {
                        start: 5,
                        duration: 2,
                        name: Some("task_b".to_string()),
                    },
                ]),
                CpConstraint::Cumulative {
                    intervals: vec![
                        CpDemandInterval {
                            start: 6,
                            duration: 3,
                            demand: 2,
                            name: Some("machine_a".to_string()),
                        },
                        CpDemandInterval {
                            start: 7,
                            duration: 2,
                            demand: 2,
                            name: Some("machine_b".to_string()),
                        },
                        CpDemandInterval {
                            start: 8,
                            duration: 2,
                            demand: 1,
                            name: Some("machine_c".to_string()),
                        },
                    ],
                    capacity: 3,
                },
                CpConstraint::Element(CpElement {
                    index: 9,
                    values: vec![3, 8],
                    target: 10,
                }),
                CpConstraint::AllowedAssignments {
                    vars: vec![11, 12],
                    tuples: vec![vec![0, 1], vec![1, 0]],
                },
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 13,
                    positive: true,
                }]),
                CpConstraint::EnforcedLinear {
                    enforcement: vec![BoolLiteral {
                        var: 13,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 14, coeff: 1 }],
                    sense: LinearSense::Ge,
                    rhs: 2,
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 8 },
                    LinearTerm { var: 1, coeff: 2 },
                    LinearTerm { var: 2, coeff: 5 },
                    LinearTerm { var: 3, coeff: -1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 6, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                    LinearTerm { var: 8, coeff: 1 },
                    LinearTerm { var: 10, coeff: 1 },
                    LinearTerm { var: 11, coeff: 2 },
                    LinearTerm { var: 12, coeff: 1 },
                    LinearTerm { var: 14, coeff: 1 },
                ],
            }),
        }
    }

    fn validate_cp_sat(&mut self) {
        println!("\n-- CP-SAT: finite-domain solver vs CP reference bridge --");
        let model = self.sample_cp_model();
        let internal = solve_cp_model(&model, CpSolveOptions::default());
        let variables: Vec<_> = model
            .variables
            .iter()
            .map(|v| serde_json::json!({"name": v.name, "domain": v.domain}))
            .collect();
        let constraints: Vec<_> = model
            .constraints
            .iter()
            .map(|c| match c {
                CpConstraint::Linear { terms, sense, rhs } => serde_json::json!({
                    "kind": "linear",
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "sense": sense.as_str(),
                    "rhs": rhs,
                }),
                CpConstraint::EnforcedLinear {
                    enforcement,
                    terms,
                    sense,
                    rhs,
                } => serde_json::json!({
                    "kind": "enforced_linear",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "sense": sense.as_str(),
                    "rhs": rhs,
                }),
                CpConstraint::AllDifferent(vars) => {
                    serde_json::json!({"kind": "all_different", "vars": vars})
                }
                CpConstraint::BoolOr(lits) => serde_json::json!({
                    "kind": "bool_or",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::AllowedAssignments { vars, tuples } => serde_json::json!({
                    "kind": "allowed_assignments",
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::Element(element) => serde_json::json!({
                    "kind": "element",
                    "index": element.index,
                    "values": &element.values,
                    "target": element.target,
                }),
                CpConstraint::NoOverlap(intervals) => serde_json::json!({
                    "kind": "no_overlap",
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::Cumulative {
                    intervals,
                    capacity,
                } => serde_json::json!({
                    "kind": "cumulative",
                    "capacity": capacity,
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "demand": interval.demand,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
            })
            .collect();
        let objective = model.objective.as_ref().map(|obj| {
            serde_json::json!({
                "sense": obj.sense.as_str(),
                "terms": obj.terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>()
            })
        });
        let model_json = serde_json::json!({
            "variables": variables,
            "constraints": constraints,
            "objective": objective,
        })
        .to_string();
        let value = self.run_python_json("cp_sat_reference.py", &["--solver", "auto"], &model_json);
        let reference: CpReference = serde_json::from_value(value).expect("parse CP reference");
        self.check(
            "CP-SAT statuses optimal",
            internal.status == CpStatus::Optimal && reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                internal.status.as_str(),
                reference.status,
                reference.solver
            ),
        );
        self.check(
            "CP-SAT objective",
            internal.objective == reference.objective,
            format!(
                "internal={:?} external={:?}",
                internal.objective, reference.objective
            ),
        );
        self.check(
            "CP-SAT assignment",
            internal.assignment == reference.assignment,
            format!(
                "internal={:?} external={:?}",
                internal.assignment, reference.assignment
            ),
        );
    }

    fn run_all(&mut self) {
        self.validate_lp();
        self.validate_ip_mip();
        self.validate_min_cost_flow();
        self.validate_qp();
        self.validate_cp_sat();
    }
}

pub fn run() {
    println!("Optimization suite: native solvers vs external/reference bridges");
    println!("===============================================================");
    let mut d = Driver::new();
    d.run_all();
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "\nvalidate-optimization-suite: {passed}/{} checks passed.",
        d.checks.len()
    );
    if passed != d.checks.len() {
        println!("FAILED:");
        for row in &d.checks {
            if !row.passed {
                println!("  - {}: {}", row.name, row.detail);
            }
        }
        std::process::exit(1);
    }
}
