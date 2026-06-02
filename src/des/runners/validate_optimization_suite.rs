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
    solve_cp_model, BoolLiteral, CpAutomaton, CpCircuitArc, CpConstraint, CpDemandInterval,
    CpDomainInterval, CpElement, CpInterval, CpModel, CpObjective, CpRectangle, CpReservoirEvent,
    CpSolveOptions, CpStatus, CpTransition, CpVariable, LinearSense, LinearTerm, ObjectiveSense,
};
use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, build_fixed_charge_indicator_ip, build_general_linear_rows_ip,
    build_lexicographic_choice_ip, build_lower_bounded_production_ip,
    build_piecewise_linear_reward_ip, build_semi_continuous_gate_ip, build_semi_integer_lot_ip,
    build_sos1_choice_ip, build_sos2_adjacency_ip, build_source_feature_mix_ip,
    linearize_indicator_problem, linearize_pwl_problem, linearize_semi_problem,
    linearize_sos_problem, linearize_source_ipmip_problem, solve_general_linear_ipmip_with_des,
    solve_indicator_ipmip_with_des, solve_ipmip_with_des, solve_lower_bounded_ipmip_with_des,
    solve_multi_objective_ipmip_with_des, solve_pwl_ipmip_with_des, solve_semi_ipmip_with_des,
    solve_sos_ipmip_with_des, solve_source_ipmip_with_des, ConcreteLpRelaxationAlgorithm,
    IPMIPSolveOptions, IPMIPStatus, LpRelaxationAlgorithm,
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
    #[serde(rename = "dualUB")]
    dual_ub: Option<Vec<f64>>,
    #[serde(rename = "dualEQ")]
    dual_eq: Option<Vec<f64>>,
    #[serde(rename = "dualLowerBounds")]
    dual_lower_bounds: Option<Vec<f64>>,
    #[serde(rename = "dualUpperBounds")]
    dual_upper_bounds: Option<Vec<f64>>,
    #[serde(rename = "reducedGradient")]
    reduced_gradient: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct CpReference {
    status: String,
    solver: String,
    assignment: Vec<i64>,
    objective: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct LinearCliReference {
    status: String,
    solver: String,
    x: Vec<f64>,
    objective: Option<f64>,
    message: String,
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

    fn max_abs_close_optional(
        &mut self,
        name: &str,
        a: Option<&[f64]>,
        b: Option<&[f64]>,
        tol: f64,
    ) {
        match (a, b) {
            (Some(a), Some(b)) => self.max_abs_close(name, a, b, tol),
            _ => self.check(name, false, "missing certificate vector"),
        }
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

        let certificate_lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![2.0, 1.0]]),
            b_ub: Some(vec![4.0, 5.0]),
            ..Default::default()
        };
        let certificate_internal =
            solve_lp_internal(&certificate_lp, &InternalSimplexOptions::default());
        let certificate_external = solve_lp_external(
            &certificate_lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..Default::default()
            },
        );
        self.check(
            "LP certificate statuses optimal",
            certificate_internal.status == LPStatus::Optimal
                && certificate_external.status == LPStatus::Optimal,
            format!(
                "internal={:?} external={:?}",
                certificate_internal.status, certificate_external.status
            ),
        );
        self.close(
            "LP certificate objective",
            certificate_internal.objective,
            certificate_external.objective,
            1e-9,
        );
        self.max_abs_close(
            "LP certificate x",
            &certificate_internal.x,
            &certificate_external.x,
            1e-8,
        );
        self.max_abs_close_optional(
            "LP certificate dual_ub internal expected",
            certificate_internal.dual_ub.as_deref(),
            Some(&[1.0, 1.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP certificate dual_ub external expected",
            certificate_external.dual_ub.as_deref(),
            Some(&[1.0, 1.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP certificate dual_ub internal/external",
            certificate_internal.dual_ub.as_deref(),
            certificate_external.dual_ub.as_deref(),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP certificate reduced costs",
            certificate_internal.reduced_costs.as_deref(),
            certificate_external.reduced_costs.as_deref(),
            1e-8,
        );

        let bound_certificate_lp = LPProblem {
            sense: Sense::Max,
            c: vec![-5.0, 2.0, 4.0],
            a_ub: Some(vec![vec![0.0, 1.0, 0.0]]),
            b_ub: Some(vec![2.0]),
            ub: Some(vec![None, None, Some(1.0)]),
            ..Default::default()
        };
        let bound_certificate_internal =
            solve_lp_internal(&bound_certificate_lp, &InternalSimplexOptions::default());
        let bound_certificate_external = solve_lp_external(
            &bound_certificate_lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..Default::default()
            },
        );
        self.check(
            "LP bound certificate statuses optimal",
            bound_certificate_internal.status == LPStatus::Optimal
                && bound_certificate_external.status == LPStatus::Optimal,
            format!(
                "internal={:?} external={:?}",
                bound_certificate_internal.status, bound_certificate_external.status
            ),
        );
        self.close(
            "LP bound certificate objective",
            bound_certificate_internal.objective,
            bound_certificate_external.objective,
            1e-9,
        );
        self.max_abs_close(
            "LP bound certificate x",
            &bound_certificate_internal.x,
            &bound_certificate_external.x,
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate dual_ub internal expected",
            bound_certificate_internal.dual_ub.as_deref(),
            Some(&[2.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate dual_ub external expected",
            bound_certificate_external.dual_ub.as_deref(),
            Some(&[2.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate dual_ub internal/external",
            bound_certificate_internal.dual_ub.as_deref(),
            bound_certificate_external.dual_ub.as_deref(),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate reduced internal expected",
            bound_certificate_internal.reduced_costs.as_deref(),
            Some(&[-5.0, 0.0, 4.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate reduced external expected",
            bound_certificate_external.reduced_costs.as_deref(),
            Some(&[-5.0, 0.0, 4.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "LP bound certificate reduced internal/external",
            bound_certificate_internal.reduced_costs.as_deref(),
            bound_certificate_external.reduced_costs.as_deref(),
            1e-8,
        );

        let glop = solve_lp_external(
            &lp,
            &ExternalSolverOptions {
                method: Some("glop".to_string()),
                ..Default::default()
            },
        );
        self.check(
            "LP OR-Tools GLOP status optimal",
            internal.status == LPStatus::Optimal
                && glop.status == LPStatus::Optimal
                && glop.solver == "ortools:glop",
            format!(
                "internal={:?} external={:?} solver={}",
                internal.status, glop.status, glop.solver
            ),
        );
        self.close(
            "LP OR-Tools GLOP objective",
            internal.objective,
            glop.objective,
            1e-9,
        );
        self.max_abs_close("LP OR-Tools GLOP x", &internal.x, &glop.x, 1e-8);
    }

    fn run_linear_cli_reference(
        &self,
        kind: &str,
        solver: &str,
        stdin_json: &str,
    ) -> LinearCliReference {
        let value = self.run_python_json(
            "linear_cli_reference.py",
            &["--kind", kind, "--solver", solver],
            stdin_json,
        );
        serde_json::from_value(value).expect("parse linear CLI reference")
    }

    fn validate_external_solver_clis(&mut self) {
        println!(
            "\n-- External solver CLIs: GLPK/HiGHS/SCIP/CBC/CLP + optional commercial checks --"
        );
        let lp_solvers = ["highs", "glpk", "scip", "cbc", "clp"];
        let mip_solvers = ["highs", "glpk", "scip", "cbc"];
        let commercial_mip_solvers = ["gurobi", "cplex", "xpress", "lindo"];

        let lp = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
            b_ub: Some(vec![4.0, 6.0]),
            ..Default::default()
        };
        let lp_internal = solve_lp_internal(&lp, &InternalSimplexOptions::default());
        let lp_json = serde_json::json!({
            "lp": {
                "sense": lp.sense.as_str(),
                "c": &lp.c,
                "a_ub": &lp.a_ub,
                "b_ub": &lp.b_ub,
                "a_eq": &lp.a_eq,
                "b_eq": &lp.b_eq,
                "lb": &lp.lb,
                "ub": &lp.ub,
            }
        })
        .to_string();

        for solver in lp_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("lp", solver, &lp_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  LP {solver}: executable not found");
                continue;
            }
            self.check(
                format!("LP {solver}:cli status optimal"),
                lp_internal.status == LPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={:?} external={} solver={}",
                    lp_internal.status, reference.status, reference.solver
                ),
            );
            self.close(
                &format!("LP {solver}:cli objective"),
                lp_internal.objective,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("LP {solver}:cli x"),
                &lp_internal.x,
                &reference.x,
                1e-8,
            );
        }

        let mip =
            build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        let mip_internal = solve_ipmip_with_des(
            mip.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let mip_json = serde_json::json!({
            "sense": mip.sense.as_str(),
            "c": mip.c,
            "a": mip.a,
            "b": mip.b,
            "integer_vars": mip.integer_vars,
            "ub": mip.ub,
            "var_names": mip.var_names,
            "con_names": mip.con_names,
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &mip_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP {solver}:cli status optimal"),
                mip_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    mip_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP {solver}:cli objective"),
                mip_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP {solver}:cli x"),
                &mip_internal.x,
                &reference.x,
                1e-8,
            );
        }

        for solver in commercial_mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &mip_json);
            if reference.status == "unavailable" {
                println!("  SKIP  IP/MIP commercial {solver}: {}", reference.message);
                continue;
            }
            self.check(
                format!("IP/MIP commercial {solver}:cli status optimal"),
                mip_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    mip_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP commercial {solver}:cli objective"),
                mip_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP commercial {solver}:cli x"),
                &mip_internal.x,
                &reference.x,
                1e-8,
            );
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
        let lower_base = &lower_problem.base;
        let lower_json = serde_json::json!({
            "sense": lower_base.sense.as_str(),
            "c": &lower_base.c,
            "a": &lower_base.a,
            "b": &lower_base.b,
            "integer_vars": &lower_base.integer_vars,
            "lb": &lower_problem.lb,
            "ub": &lower_base.ub,
            "var_names": &lower_base.var_names,
            "con_names": &lower_base.con_names,
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &lower_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP lower-bounded {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP lower-bounded {solver}:cli status optimal"),
                lower_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    lower_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP lower-bounded {solver}:cli objective"),
                lower_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP lower-bounded {solver}:cli x"),
                &lower_internal.x,
                &reference.x,
                1e-8,
            );
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
        let general_base = &general_problem.base;
        let general_json = serde_json::json!({
            "sense": general_base.sense.as_str(),
            "c": &general_base.c,
            "a": &general_base.a,
            "b": &general_base.b,
            "integer_vars": &general_base.integer_vars,
            "ub": &general_base.ub,
            "var_names": &general_base.var_names,
            "con_names": &general_base.con_names,
            "linear_constraints": general_problem.linear_constraints.iter().map(|constraint| serde_json::json!({
                "coefs": &constraint.coefs,
                "lower": constraint.lower,
                "upper": constraint.upper,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &general_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP general-linear {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP general-linear {solver}:cli status optimal"),
                general_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    general_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP general-linear {solver}:cli objective"),
                general_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP general-linear {solver}:cli x"),
                &general_internal.x,
                &reference.x,
                1e-8,
            );
        }

        let indicator_problem = build_fixed_charge_indicator_ip();
        let indicator_internal = solve_indicator_ipmip_with_des(
            indicator_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                ..Default::default()
            },
        );
        let indicator_base = &indicator_problem.base;
        let indicator_json = serde_json::json!({
            "sense": indicator_base.sense.as_str(),
            "c": &indicator_base.c,
            "a": &indicator_base.a,
            "b": &indicator_base.b,
            "integer_vars": &indicator_base.integer_vars,
            "ub": &indicator_base.ub,
            "var_names": &indicator_base.var_names,
            "con_names": &indicator_base.con_names,
            "indicators": indicator_problem.indicators.iter().map(|indicator| serde_json::json!({
                "binary_var": indicator.binary_var,
                "active_value": indicator.active_value,
                "coefs": &indicator.coefs,
                "sense": indicator.sense.as_str(),
                "rhs": indicator.rhs,
                "name": &indicator.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &indicator_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP indicator {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP indicator {solver}:cli status optimal"),
                indicator_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    indicator_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP indicator {solver}:cli objective"),
                indicator_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP indicator {solver}:cli x"),
                &indicator_internal.x,
                &reference.x,
                1e-8,
            );
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
        let pwl_base = &pwl_problem.base;
        let pwl_json = serde_json::json!({
            "sense": pwl_base.sense.as_str(),
            "c": &pwl_base.c,
            "a": &pwl_base.a,
            "b": &pwl_base.b,
            "integer_vars": &pwl_base.integer_vars,
            "ub": &pwl_base.ub,
            "var_names": &pwl_base.var_names,
            "con_names": &pwl_base.con_names,
            "pwl": pwl_problem.pwl.iter().map(|pwl| serde_json::json!({
                "x_var": pwl.x_var,
                "y_var": pwl.y_var,
                "points": pwl.points.iter().map(|point| serde_json::json!({
                    "x": point.x,
                    "y": point.y,
                })).collect::<Vec<_>>(),
                "name": &pwl.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &pwl_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP piecewise-linear {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP piecewise-linear {solver}:cli status optimal"),
                pwl_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    pwl_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP piecewise-linear {solver}:cli objective"),
                pwl_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP piecewise-linear {solver}:cli expanded x length"),
                reference.x.len() == linearized_pwl.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_pwl.c.len(),
                    reference.x.len()
                ),
            );
        }

        let source_problem = build_source_feature_mix_ip();
        let (linearized_source, _, source_original_vars) =
            linearize_source_ipmip_problem(&source_problem);
        let source_internal = solve_source_ipmip_with_des(
            source_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let source_base = &source_problem.base;
        let source_json = serde_json::json!({
            "sense": source_base.sense.as_str(),
            "c": &source_base.c,
            "a": &source_base.a,
            "b": &source_base.b,
            "integer_vars": &source_base.integer_vars,
            "lb": &source_problem.lb,
            "ub": &source_base.ub,
            "var_names": &source_base.var_names,
            "con_names": &source_base.con_names,
            "linear_constraints": source_problem.linear_constraints.iter().map(|constraint| serde_json::json!({
                "coefs": &constraint.coefs,
                "lower": constraint.lower,
                "upper": constraint.upper,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "indicators": source_problem.indicators.iter().map(|indicator| serde_json::json!({
                "binary_var": indicator.binary_var,
                "active_value": indicator.active_value,
                "coefs": &indicator.coefs,
                "sense": indicator.sense.as_str(),
                "rhs": indicator.rhs,
                "name": &indicator.name,
            })).collect::<Vec<_>>(),
            "pwl": source_problem.pwl.iter().map(|pwl| serde_json::json!({
                "x_var": pwl.x_var,
                "y_var": pwl.y_var,
                "points": pwl.points.iter().map(|point| serde_json::json!({
                    "x": point.x,
                    "y": point.y,
                })).collect::<Vec<_>>(),
                "name": &pwl.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &source_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP source-feature-mix {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP source-feature-mix {solver}:cli status optimal"),
                source_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    source_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP source-feature-mix {solver}:cli objective"),
                source_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP source-feature-mix {solver}:cli expanded x length"),
                reference.x.len() == linearized_source.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_source.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= source_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP source-feature-mix {solver}:cli original x"),
                    &source_internal.x[..source_original_vars],
                    &reference.x[..source_original_vars],
                    1e-8,
                );
            }
        }
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

        let source_problem = build_source_feature_mix_ip();
        let (linearized_source, _, source_original_vars) =
            linearize_source_ipmip_problem(&source_problem);
        let source_internal = solve_source_ipmip_with_des(
            source_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let source_problem_path = out_dir.join("source-feature-mix-problem.json");
        let source_reference_path = out_dir.join("source-feature-mix-reference.json");
        let base = &source_problem.base;
        let source_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &source_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "linear_constraints": source_problem.linear_constraints.iter().map(|constraint| serde_json::json!({
                "coefs": &constraint.coefs,
                "lower": constraint.lower,
                "upper": constraint.upper,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "indicators": source_problem.indicators.iter().map(|indicator| serde_json::json!({
                "binary_var": indicator.binary_var,
                "active_value": indicator.active_value,
                "coefs": &indicator.coefs,
                "sense": indicator.sense.as_str(),
                "rhs": indicator.rhs,
                "name": &indicator.name,
            })).collect::<Vec<_>>(),
            "pwl": source_problem.pwl.iter().map(|pwl| serde_json::json!({
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
            &source_problem_path,
            serde_json::to_string_pretty(&source_json)
                .expect("serialize source-feature MIP problem"),
        )
        .expect("write source-feature MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&source_problem_path)
            .arg("--out")
            .arg(&source_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run source-feature MIP reference");
        if !output.status.success() {
            panic!(
                "source-feature ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let source_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&source_reference_path).expect("read source-feature MIP reference JSON"),
        )
        .expect("parse source-feature MIP reference JSON");
        self.check(
            "IP/MIP source-feature-mix statuses optimal",
            source_internal.status == IPMIPStatus::Optimal
                && source_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                source_internal.status.as_str(),
                source_reference.result.status,
                source_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP source-feature-mix objective",
            source_internal.z,
            source_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = source_reference.result.x.as_deref() {
            self.check(
                "IP/MIP source-feature-mix external x length",
                x.len() == linearized_source.c.len(),
                "",
            );
            if x.len() >= source_original_vars {
                self.max_abs_close(
                    "IP/MIP source-feature-mix original x",
                    &source_internal.x[..source_original_vars],
                    &x[..source_original_vars],
                    1e-8,
                );
            }
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
        self.max_abs_close_optional(
            "QP dual_ub internal expected",
            Some(internal.dual_ub.as_slice()),
            Some(&[1.75]),
            1e-8,
        );
        self.max_abs_close_optional(
            "QP dual_ub external expected",
            reference.dual_ub.as_deref(),
            Some(&[1.75]),
            1e-7,
        );
        self.max_abs_close_optional(
            "QP dual_ub internal/external",
            Some(internal.dual_ub.as_slice()),
            reference.dual_ub.as_deref(),
            1e-7,
        );
        self.max_abs_close_optional(
            "QP reduced-gradient internal/external",
            Some(internal.reduced_gradient.as_slice()),
            reference.reduced_gradient.as_deref(),
            1e-7,
        );

        let bound_qp = QuadraticProgram {
            q: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            c: vec![3.0, -2.0, -4.0],
            a_ub: Some(vec![vec![0.0, 1.0, 0.0]]),
            b_ub: Some(vec![1.0]),
            lb: Some(vec![Some(0.0), Some(0.0), Some(0.0)]),
            ub: Some(vec![None, None, Some(1.0)]),
            var_names: Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            ..Default::default()
        };
        let bound_internal = solve_qp_active_set(&bound_qp, QPOptions::default());
        let bound_qp_json = serde_json::json!({
            "Q": &bound_qp.q,
            "c": &bound_qp.c,
            "A_ub": &bound_qp.a_ub,
            "b_ub": &bound_qp.b_ub,
            "A_eq": &bound_qp.a_eq,
            "b_eq": &bound_qp.b_eq,
            "lb": &bound_qp.lb,
            "ub": &bound_qp.ub,
        })
        .to_string();
        let value = self.run_python_json("qp_reference.py", &["--solver", "auto"], &bound_qp_json);
        let bound_reference: QPReference =
            serde_json::from_value(value).expect("parse bound QP reference");
        self.check(
            "QP bound certificate statuses optimal",
            bound_internal.status == QPStatus::Optimal && bound_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                bound_internal.status.as_str(),
                bound_reference.status,
                bound_reference.solver
            ),
        );
        self.close(
            "QP bound certificate objective",
            bound_internal.objective,
            bound_reference.objective.unwrap_or(f64::NAN),
            1e-8,
        );
        self.max_abs_close(
            "QP bound certificate x",
            &bound_internal.x,
            &bound_reference.x,
            1e-7,
        );
        self.max_abs_close_optional(
            "QP bound certificate dual_ub internal expected",
            Some(bound_internal.dual_ub.as_slice()),
            Some(&[1.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "QP bound certificate dual_ub external expected",
            bound_reference.dual_ub.as_deref(),
            Some(&[1.0]),
            1e-6,
        );
        self.max_abs_close_optional(
            "QP bound certificate lower dual internal expected",
            Some(bound_internal.dual_lower_bounds.as_slice()),
            Some(&[3.0, 0.0, 0.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "QP bound certificate lower dual external expected",
            bound_reference.dual_lower_bounds.as_deref(),
            Some(&[3.0, 0.0, 0.0]),
            1e-6,
        );
        self.max_abs_close_optional(
            "QP bound certificate upper dual internal expected",
            Some(bound_internal.dual_upper_bounds.as_slice()),
            Some(&[0.0, 0.0, 3.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "QP bound certificate upper dual external expected",
            bound_reference.dual_upper_bounds.as_deref(),
            Some(&[0.0, 0.0, 3.0]),
            1e-6,
        );
        self.max_abs_close_optional(
            "QP bound certificate reduced-gradient internal expected",
            Some(bound_internal.reduced_gradient.as_slice()),
            Some(&[3.0, 0.0, -3.0]),
            1e-8,
        );
        self.max_abs_close_optional(
            "QP bound certificate reduced-gradient external expected",
            bound_reference.reduced_gradient.as_deref(),
            Some(&[3.0, 0.0, -3.0]),
            1e-6,
        );
        self.max_abs_close_optional(
            "QP bound certificate reduced-gradient internal/external",
            Some(bound_internal.reduced_gradient.as_slice()),
            bound_reference.reduced_gradient.as_deref(),
            1e-6,
        );

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
                CpVariable {
                    name: "choice_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "choice_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "choice_c".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "approved".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "direct_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "direct_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inverse_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inverse_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "score_a".to_string(),
                    domain: vec![2, 4],
                },
                CpVariable {
                    name: "score_b".to_string(),
                    domain: vec![3, 5],
                },
                CpVariable {
                    name: "max_score".to_string(),
                    domain: vec![3, 4, 5],
                },
                CpVariable {
                    name: "min_score".to_string(),
                    domain: vec![2, 3, 4],
                },
                CpVariable {
                    name: "deviation".to_string(),
                    domain: vec![-3, -1, 2],
                },
                CpVariable {
                    name: "absolute_deviation".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "mandatory_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mandatory_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "xor_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "xor_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "score_product".to_string(),
                    domain: vec![6, 10, 12, 20],
                },
                CpVariable {
                    name: "pack_a_x".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "pack_a_y".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "pack_b_x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "pack_b_y".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "automaton_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "automaton_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "automaton_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_0_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_1_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_2_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "arith_value".to_string(),
                    domain: vec![5, 6, 7],
                },
                CpVariable {
                    name: "arith_divisor".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "arith_quotient".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "arith_remainder".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "reservoir_fill_time".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "reservoir_drain_time".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "reservoir_overfill_active".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "linear_domain_x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "linear_domain_y".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "mapped_mode".to_string(),
                    domain: vec![5],
                },
                CpVariable {
                    name: "mapped_is_five".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mapped_is_six".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mapped_is_seven".to_string(),
                    domain: vec![0, 1],
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
                CpConstraint::ForbiddenAssignments {
                    vars: vec![15, 18],
                    tuples: vec![vec![1, 1]],
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
                CpConstraint::ExactlyOne(vec![
                    BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 16,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 17,
                        positive: true,
                    },
                ]),
                CpConstraint::AtMostOne(vec![
                    BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 17,
                        positive: true,
                    },
                ]),
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 18,
                        positive: true,
                    },
                },
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 18,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 19,
                        positive: true,
                    },
                },
                CpConstraint::Inverse {
                    direct: vec![20, 21],
                    inverse: vec![22, 23],
                },
                CpConstraint::MaxEquality {
                    target: 26,
                    vars: vec![24, 25],
                },
                CpConstraint::MinEquality {
                    target: 27,
                    vars: vec![24, 25],
                },
                CpConstraint::AbsEquality {
                    target: 29,
                    var: 28,
                },
                CpConstraint::BoolAnd(vec![
                    BoolLiteral {
                        var: 30,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 31,
                        positive: true,
                    },
                ]),
                CpConstraint::BoolXor(vec![
                    BoolLiteral {
                        var: 32,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 33,
                        positive: true,
                    },
                ]),
                CpConstraint::MultiplicationEquality {
                    target: 34,
                    vars: vec![24, 25],
                },
                CpConstraint::NoOverlap2D(vec![
                    CpRectangle {
                        x_start: 35,
                        y_start: 36,
                        width: 2,
                        height: 2,
                        name: Some("pack_a".to_string()),
                    },
                    CpRectangle {
                        x_start: 37,
                        y_start: 38,
                        width: 2,
                        height: 2,
                        name: Some("pack_b".to_string()),
                    },
                ]),
                CpConstraint::Automaton(CpAutomaton {
                    vars: vec![39, 40, 41],
                    starting_state: 0,
                    final_states: vec![1],
                    transitions: vec![
                        CpTransition {
                            tail: 0,
                            label: 0,
                            head: 0,
                        },
                        CpTransition {
                            tail: 0,
                            label: 1,
                            head: 1,
                        },
                        CpTransition {
                            tail: 1,
                            label: 0,
                            head: 1,
                        },
                        CpTransition {
                            tail: 1,
                            label: 1,
                            head: 2,
                        },
                        CpTransition {
                            tail: 2,
                            label: 0,
                            head: 2,
                        },
                        CpTransition {
                            tail: 2,
                            label: 1,
                            head: 2,
                        },
                    ],
                }),
                CpConstraint::Circuit(vec![
                    CpCircuitArc {
                        tail: 0,
                        head: 1,
                        literal: BoolLiteral {
                            var: 42,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 1,
                        head: 2,
                        literal: BoolLiteral {
                            var: 43,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 2,
                        head: 0,
                        literal: BoolLiteral {
                            var: 44,
                            positive: true,
                        },
                    },
                ]),
                CpConstraint::DivisionEquality {
                    target: 47,
                    numerator: 45,
                    denominator: 46,
                },
                CpConstraint::ModuloEquality {
                    target: 48,
                    var: 45,
                    modulus: 46,
                },
                CpConstraint::Reservoir {
                    events: vec![
                        CpReservoirEvent {
                            time: 49,
                            level_change: 4,
                            active: None,
                        },
                        CpReservoirEvent {
                            time: 50,
                            level_change: -3,
                            active: None,
                        },
                        CpReservoirEvent {
                            time: 50,
                            level_change: 10,
                            active: Some(BoolLiteral {
                                var: 51,
                                positive: true,
                            }),
                        },
                    ],
                    min_level: 0,
                    max_level: 4,
                },
                CpConstraint::LinearDomain {
                    terms: vec![
                        LinearTerm { var: 52, coeff: 1 },
                        LinearTerm { var: 53, coeff: 2 },
                    ],
                    intervals: vec![
                        CpDomainInterval { lb: 1, ub: 1 },
                        CpDomainInterval { lb: 4, ub: 4 },
                    ],
                },
                CpConstraint::MapDomain {
                    var: 54,
                    bools: vec![55, 56, 57],
                    offset: 5,
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
                    LinearTerm { var: 15, coeff: 1 },
                    LinearTerm { var: 16, coeff: 5 },
                    LinearTerm { var: 17, coeff: 4 },
                    LinearTerm { var: 18, coeff: 1 },
                    LinearTerm { var: 19, coeff: 1 },
                    LinearTerm { var: 20, coeff: 1 },
                    LinearTerm { var: 21, coeff: 2 },
                    LinearTerm { var: 26, coeff: 1 },
                    LinearTerm { var: 27, coeff: 1 },
                    LinearTerm { var: 29, coeff: 1 },
                    LinearTerm { var: 32, coeff: 1 },
                    LinearTerm { var: 33, coeff: 2 },
                    LinearTerm { var: 34, coeff: 1 },
                    LinearTerm { var: 39, coeff: 4 },
                    LinearTerm { var: 40, coeff: 2 },
                    LinearTerm { var: 41, coeff: 1 },
                    LinearTerm { var: 45, coeff: 1 },
                    LinearTerm { var: 47, coeff: 10 },
                    LinearTerm { var: 48, coeff: 1 },
                    LinearTerm { var: 49, coeff: 1 },
                    LinearTerm { var: 51, coeff: -1 },
                    LinearTerm { var: 52, coeff: 1 },
                    LinearTerm { var: 53, coeff: 1 },
                    LinearTerm { var: 54, coeff: 1 },
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
                CpConstraint::LinearDomain { terms, intervals } => serde_json::json!({
                    "kind": "linear_domain",
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "lb": interval.lb,
                        "ub": interval.ub,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::MapDomain {
                    var,
                    bools,
                    offset,
                } => serde_json::json!({
                    "kind": "map_domain",
                    "var": var,
                    "bools": bools,
                    "offset": offset,
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
                CpConstraint::BoolAnd(lits) => serde_json::json!({
                    "kind": "bool_and",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::BoolXor(lits) => serde_json::json!({
                    "kind": "bool_xor",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::AtMostOne(lits) => serde_json::json!({
                    "kind": "at_most_one",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::ExactlyOne(lits) => serde_json::json!({
                    "kind": "exactly_one",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::Implication {
                    antecedent,
                    consequent,
                } => serde_json::json!({
                    "kind": "implication",
                    "antecedent": {"var": antecedent.var, "positive": antecedent.positive},
                    "consequent": {"var": consequent.var, "positive": consequent.positive},
                }),
                CpConstraint::Circuit(arcs) => serde_json::json!({
                    "kind": "circuit",
                    "arcs": arcs.iter().map(|arc| serde_json::json!({
                        "tail": arc.tail,
                        "head": arc.head,
                        "literal": {"var": arc.literal.var, "positive": arc.literal.positive},
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::AllowedAssignments { vars, tuples } => serde_json::json!({
                    "kind": "allowed_assignments",
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::ForbiddenAssignments { vars, tuples } => serde_json::json!({
                    "kind": "forbidden_assignments",
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::Inverse { direct, inverse } => serde_json::json!({
                    "kind": "inverse",
                    "direct": direct,
                    "inverse": inverse,
                }),
                CpConstraint::MaxEquality { target, vars } => serde_json::json!({
                    "kind": "max_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::MinEquality { target, vars } => serde_json::json!({
                    "kind": "min_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::AbsEquality { target, var } => serde_json::json!({
                    "kind": "abs_equality",
                    "target": target,
                    "var": var,
                }),
                CpConstraint::MultiplicationEquality { target, vars } => serde_json::json!({
                    "kind": "multiplication_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::DivisionEquality {
                    target,
                    numerator,
                    denominator,
                } => serde_json::json!({
                    "kind": "division_equality",
                    "target": target,
                    "numerator": numerator,
                    "denominator": denominator,
                }),
                CpConstraint::ModuloEquality {
                    target,
                    var,
                    modulus,
                } => serde_json::json!({
                    "kind": "modulo_equality",
                    "target": target,
                    "var": var,
                    "modulus": modulus,
                }),
                CpConstraint::Automaton(automaton) => serde_json::json!({
                    "kind": "automaton",
                    "vars": automaton.vars,
                    "starting_state": automaton.starting_state,
                    "final_states": automaton.final_states,
                    "transitions": automaton.transitions.iter().map(|transition| serde_json::json!({
                        "tail": transition.tail,
                        "label": transition.label,
                        "head": transition.head,
                    })).collect::<Vec<_>>(),
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
                CpConstraint::NoOverlap2D(rectangles) => serde_json::json!({
                    "kind": "no_overlap_2d",
                    "rectangles": rectangles.iter().map(|rectangle| serde_json::json!({
                        "x_start": rectangle.x_start,
                        "y_start": rectangle.y_start,
                        "width": rectangle.width,
                        "height": rectangle.height,
                        "name": rectangle.name,
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
                CpConstraint::Reservoir {
                    events,
                    min_level,
                    max_level,
                } => serde_json::json!({
                    "kind": "reservoir",
                    "min_level": min_level,
                    "max_level": max_level,
                    "events": events.iter().map(|event| serde_json::json!({
                        "time": event.time,
                        "level_change": event.level_change,
                        "active": event.active.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
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
        self.validate_external_solver_clis();
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
