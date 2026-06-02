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
    enumerate_cp_solutions, find_cp_assumption_unsat_core, solve_cp_model, BoolLiteral,
    CpAlternative, CpAssumptionCoreOptions, CpAutomaton, CpCircuitArc, CpConstraint,
    CpDecisionStrategy, CpDemandInterval, CpDomainInterval, CpDomainValueStrategy, CpElement,
    CpEnumerateOptions, CpInterval, CpModel, CpObjective, CpRectangle, CpReservoirEvent,
    CpSolutionHint, CpSolveOptions, CpStatus, CpTransition, CpVariable, CpVariableDemandInterval,
    CpVariableInterval, CpVariableRectangle, CpVariableSelectionStrategy, LinearSense, LinearTerm,
    ObjectiveSense,
};
use crate::des::general::external_linear_cli::{
    probe_external_linear_cli_solver, solve_ipmip_with_external_cli, solve_lp_with_external_cli,
    ExternalLinearCliKind, ExternalLinearCliOptions, ExternalLinearCliProbeStatus,
    ExternalLinearCliSolver, ExternalLinearCliStatus,
};
use crate::des::general::external_optimization_ecosystem::{
    probe_external_optimization_tool, ExternalOptimizationProbeStatus, ExternalOptimizationTool,
};
use crate::des::general::ip_mip_des::{
    build_absolute_value_penalty_ip, build_binary_knapsack_ip, build_binary_product_gate_ip,
    build_fixed_charge_indicator_ip, build_general_linear_rows_ip,
    build_ipmip_feasibility_relaxation_problem, build_l1_norm_deviation_ip,
    build_lexicographic_choice_ip, build_linf_norm_deviation_ip, build_logical_gate_ip,
    build_lower_bounded_production_ip, build_maximum_peak_ip, build_minimum_floor_ip,
    build_piecewise_linear_reward_ip, build_product_activation_ip,
    build_quadratic_objective_mix_ip, build_semi_continuous_gate_ip, build_semi_integer_lot_ip,
    build_sos1_choice_ip, build_sos2_adjacency_ip, build_source_feature_mix_ip,
    find_ipmip_infeasibility_conflict, ipmip_feasibility_problem_from_conflict_members,
    linearize_indicator_problem, linearize_pwl_problem, linearize_quadratic_objective_problem,
    linearize_semi_problem, linearize_sos_problem, linearize_source_ipmip_problem,
    solve_general_linear_ipmip_with_des, solve_indicator_ipmip_with_des,
    solve_ipmip_feasibility_relaxation_with_des, solve_ipmip_solution_pool_with_des,
    solve_ipmip_with_des, solve_lower_bounded_ipmip_with_des, solve_multi_objective_ipmip_with_des,
    solve_pwl_ipmip_with_des, solve_quadratic_objective_ipmip_with_des, solve_semi_ipmip_with_des,
    solve_sos_ipmip_with_des, solve_source_ipmip_with_des, BranchRule,
    ConcreteLpRelaxationAlgorithm, IPMIPConflictMember, IPMIPConflictOptions, IPMIPFeasRelaxMember,
    IPMIPFeasRelaxOptions, IPMIPProblem, IPMIPSolutionPoolOptions, IPMIPSolveOptions, IPMIPStatus,
    LpRelaxationAlgorithm, TraceAction,
};
use crate::des::general::lp::{
    build_lp_feasibility_relaxation_problem, find_lp_infeasibility_conflict,
    lp_feasibility_problem_from_conflict_members, solve_general_linear_lp_internal,
    solve_lp_external, solve_lp_feasibility_relaxation_internal, solve_lp_internal,
    solve_objective_offset_lp_internal, ExternalSolverOptions, GeneralLinearLPProblem,
    InternalSimplexOptions, LPConflictMember, LPConflictOptions, LPFeasRelaxMember,
    LPFeasRelaxOptions, LPProblem, LPRowConstraint, LPStatus, ObjectiveOffsetLPProblem, Sense,
};
use crate::des::general::math_program::{
    cross_check_math_program_conflict_with_external,
    cross_check_math_program_feas_relaxation_with_external,
    cross_check_math_program_solution_pool_with_external, cross_check_math_program_with_external,
    export_math_program_cplex_lp, ExternalMathProgramOptions, MathProgram,
    MathProgramConflictOptions, MathProgramFeasRelaxOptions, MathProgramSolutionPoolOptions,
    MathProgramSolveOptions, MathProgramStatus, ObjectiveSense as MathObjectiveSense, RowSense,
};
use crate::des::general::min_cost_flow::{
    min_cost_flow_to_lp, solve_min_cost_flow, MinCostFlowArc, MinCostFlowProblem, MinCostFlowStatus,
};
use crate::des::general::qp::{
    solve_miqp_enumeration, solve_qcp_pattern_search, solve_qp_active_set,
    solve_socp_pattern_search, MIQPOptions, MixedIntegerQuadraticProgram, QPOptions, QPStatus,
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
    solutions: Option<Vec<MipPoolReferenceSolution>>,
    exhausted: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct MipReference {
    result: MipReferenceInner,
}

#[derive(Debug, Deserialize)]
struct MipPoolReferenceSolution {
    x: Vec<f64>,
    objective: f64,
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
struct LPReference {
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

#[derive(Debug, Deserialize)]
struct CpPoolReference {
    status: String,
    solver: String,
    solutions: Vec<CpPoolReferenceSolution>,
    exhausted: bool,
}

#[derive(Debug, Deserialize)]
struct CpPoolReferenceSolution {
    assignment: Vec<i64>,
    objective: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct CpAssumptionCoreReference {
    status: String,
    solver: String,
    assumptions: Vec<CpLiteralReference>,
    minimal: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct CpLiteralReference {
    var: usize,
    positive: bool,
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

        let range_lp = GeneralLinearLPProblem {
            base: LPProblem {
                sense: Sense::Max,
                c: vec![3.0, 2.0],
                a_ub: Some(vec![vec![1.0, 0.0]]),
                b_ub: Some(vec![10.0]),
                lb: Some(vec![Some(0.0), Some(0.0)]),
                ub: Some(vec![Some(10.0), Some(10.0)]),
                var_names: Some(vec!["x".to_string(), "y".to_string()]),
                con_names: Some(vec!["x_cap".to_string()]),
                ..Default::default()
            },
            linear_constraints: vec![
                LPRowConstraint {
                    coefs: vec![1.0, 2.0],
                    lower: Some(8.0),
                    upper: Some(8.0),
                    name: Some("balance_eq".to_string()),
                },
                LPRowConstraint {
                    coefs: vec![1.0, -1.0],
                    lower: Some(1.0),
                    upper: None,
                    name: Some("dominance_ge".to_string()),
                },
                LPRowConstraint {
                    coefs: vec![1.0, 1.0],
                    lower: Some(5.0),
                    upper: Some(7.0),
                    name: Some("throughput_range".to_string()),
                },
            ],
        };
        let range_internal =
            solve_general_linear_lp_internal(&range_lp, &InternalSimplexOptions::default());
        let range_json = serde_json::json!({
            "lp": {
                "sense": range_lp.base.sense.as_str(),
                "c": &range_lp.base.c,
                "A_ub": &range_lp.base.a_ub,
                "b_ub": &range_lp.base.b_ub,
                "A_eq": &range_lp.base.a_eq,
                "b_eq": &range_lp.base.b_eq,
                "lb": &range_lp.base.lb,
                "ub": &range_lp.base.ub,
                "linear_constraints": range_lp.linear_constraints.iter().map(|row| serde_json::json!({
                    "coefs": &row.coefs,
                    "lower": row.lower,
                    "upper": row.upper,
                    "name": &row.name,
                })).collect::<Vec<_>>(),
            },
            "method": "highs",
        })
        .to_string();
        let value = self.run_python_json("lp_solve.py", &["--method", "highs"], &range_json);
        let range_reference: LPReference =
            serde_json::from_value(value).expect("parse range-row LP reference");
        self.check(
            "LP range-row statuses optimal",
            range_internal.status == LPStatus::Optimal && range_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                range_internal.status.as_str(),
                range_reference.status,
                range_reference.solver
            ),
        );
        self.close(
            "LP range-row objective",
            range_internal.objective,
            range_reference.objective.unwrap_or(f64::NAN),
            1e-8,
        );
        self.max_abs_close(
            "LP range-row x",
            &range_internal.x,
            &range_reference.x,
            1e-8,
        );

        let offset_lp = ObjectiveOffsetLPProblem {
            base: LPProblem {
                sense: Sense::Max,
                c: vec![1.0, 1.0],
                a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
                b_ub: Some(vec![4.0, 3.0]),
                ..Default::default()
            },
            objective_offset: 5.5,
        };
        let offset_internal =
            solve_objective_offset_lp_internal(&offset_lp, &InternalSimplexOptions::default());
        let offset_json = serde_json::json!({
            "lp": {
                "sense": offset_lp.base.sense.as_str(),
                "c": &offset_lp.base.c,
                "A_ub": &offset_lp.base.a_ub,
                "b_ub": &offset_lp.base.b_ub,
                "A_eq": &offset_lp.base.a_eq,
                "b_eq": &offset_lp.base.b_eq,
                "lb": &offset_lp.base.lb,
                "ub": &offset_lp.base.ub,
                "objective_offset": offset_lp.objective_offset,
            },
            "method": "highs",
        })
        .to_string();
        let value = self.run_python_json("lp_solve.py", &["--method", "highs"], &offset_json);
        let offset_reference: LPReference =
            serde_json::from_value(value).expect("parse objective-offset LP reference");
        self.check(
            "LP objective-offset statuses optimal",
            offset_internal.status == LPStatus::Optimal && offset_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                offset_internal.status.as_str(),
                offset_reference.status,
                offset_reference.solver
            ),
        );
        self.close(
            "LP objective-offset objective",
            offset_internal.objective,
            offset_reference.objective.unwrap_or(f64::NAN),
            1e-8,
        );
        self.max_abs_close(
            "LP objective-offset x",
            &offset_internal.x,
            &offset_reference.x,
            1e-8,
        );

        let conflict_lp = LPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0], vec![-1.0], vec![1.0]]),
            b_ub: Some(vec![0.0, -1.0, 5.0]),
            lb: Some(vec![Some(0.0)]),
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_at_most_zero".to_string(),
                "x_at_least_one".to_string(),
                "redundant_cap".to_string(),
            ]),
            ..Default::default()
        };
        let conflict = find_lp_infeasibility_conflict(&conflict_lp, &LPConflictOptions::default());
        self.check(
            "LP infeasibility conflict minimal",
            conflict.infeasible && conflict.minimal,
            format!(
                "members={:?} checks={} message={:?}",
                conflict.members, conflict.checks, conflict.message
            ),
        );
        self.check(
            "LP infeasibility conflict expected rows",
            conflict.members == vec![LPConflictMember::UpperRow(0), LPConflictMember::UpperRow(1)],
            format!("members={:?}", conflict.members),
        );
        let conflict_subproblem =
            lp_feasibility_problem_from_conflict_members(&conflict_lp, &conflict.members);
        let conflict_json = serde_json::json!({
            "lp": {
                "sense": conflict_subproblem.sense.as_str(),
                "c": &conflict_subproblem.c,
                "A_ub": &conflict_subproblem.a_ub,
                "b_ub": &conflict_subproblem.b_ub,
                "A_eq": &conflict_subproblem.a_eq,
                "b_eq": &conflict_subproblem.b_eq,
                "lb": &conflict_subproblem.lb,
                "ub": &conflict_subproblem.ub,
            },
            "method": "highs",
        })
        .to_string();
        let value = self.run_python_json("lp_solve.py", &["--method", "highs"], &conflict_json);
        let conflict_reference: LPReference =
            serde_json::from_value(value).expect("parse LP conflict reference");
        self.check(
            "LP conflict subsystem external infeasible",
            conflict_reference.status == "infeasible",
            format!(
                "external={} solver={}",
                conflict_reference.status, conflict_reference.solver
            ),
        );
        let mut deletion_statuses = Vec::new();
        let mut all_single_deletions_feasible = true;
        for idx in 0..conflict.members.len() {
            let mut trial = conflict.members.clone();
            trial.remove(idx);
            let trial_subproblem =
                lp_feasibility_problem_from_conflict_members(&conflict_lp, &trial);
            let trial_json = serde_json::json!({
                "lp": {
                    "sense": trial_subproblem.sense.as_str(),
                    "c": &trial_subproblem.c,
                    "A_ub": &trial_subproblem.a_ub,
                    "b_ub": &trial_subproblem.b_ub,
                    "A_eq": &trial_subproblem.a_eq,
                    "b_eq": &trial_subproblem.b_eq,
                    "lb": &trial_subproblem.lb,
                    "ub": &trial_subproblem.ub,
                },
                "method": "highs",
            })
            .to_string();
            let value = self.run_python_json("lp_solve.py", &["--method", "highs"], &trial_json);
            let reference: LPReference =
                serde_json::from_value(value).expect("parse LP conflict deletion reference");
            all_single_deletions_feasible &= reference.status == "optimal";
            deletion_statuses.push(reference.status);
        }
        self.check(
            "LP conflict single-deletion external feasibility",
            all_single_deletions_feasible,
            format!("statuses={deletion_statuses:?}"),
        );

        let feas_relax_lp = LPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0], vec![-1.0]]),
            b_ub: Some(vec![0.0, -1.0]),
            a_eq: None,
            b_eq: None,
            lb: Some(vec![None]),
            ub: Some(vec![None]),
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_at_most_zero".to_string(),
                "x_at_least_one".to_string(),
            ]),
            ..Default::default()
        };
        let feas_relax_options = LPFeasRelaxOptions {
            upper_row_penalties: Some(vec![3.0, 1.0]),
            ..Default::default()
        };
        let feas_relax_internal =
            solve_lp_feasibility_relaxation_internal(&feas_relax_lp, &feas_relax_options);
        self.check(
            "LP feasibility-relaxation internal status",
            feas_relax_internal.status == LPStatus::Optimal,
            format!(
                "status={} cost={} violations={:?}",
                feas_relax_internal.status.as_str(),
                feas_relax_internal.relaxation_cost,
                feas_relax_internal.violations
            ),
        );
        self.check(
            "LP feasibility-relaxation weighted violation",
            (feas_relax_internal.relaxation_cost - 1.0).abs() <= 1e-9
                && feas_relax_internal.violations.len() == 1
                && feas_relax_internal.violations[0].member == LPFeasRelaxMember::UpperRow(1)
                && (feas_relax_internal.violations[0].amount - 1.0).abs() <= 1e-9,
            format!(
                "cost={} violations={:?}",
                feas_relax_internal.relaxation_cost, feas_relax_internal.violations
            ),
        );
        let feas_relax_model =
            build_lp_feasibility_relaxation_problem(&feas_relax_lp, &feas_relax_options);
        let feas_relax_reference = solve_lp_with_external_cli(
            &feas_relax_model.problem,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        self.check(
            "LP feasibility-relaxation HiGHS status optimal",
            feas_relax_reference.status == ExternalLinearCliStatus::Optimal,
            format!(
                "external={} solver={}",
                feas_relax_reference.status.as_str(),
                feas_relax_reference.solver
            ),
        );
        self.close(
            "LP feasibility-relaxation cost vs HiGHS",
            feas_relax_internal.relaxation_cost,
            feas_relax_reference.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        let external_original_x =
            if feas_relax_reference.x.len() >= feas_relax_model.original_var_count {
                feas_relax_reference.x[..feas_relax_model.original_var_count].to_vec()
            } else {
                Vec::new()
            };
        self.max_abs_close(
            "LP feasibility-relaxation x vs HiGHS",
            &feas_relax_internal.x,
            &external_original_x,
            1e-9,
        );

        let lp_status_cases = vec![
            (
                "infeasible",
                LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a_ub: Some(vec![vec![1.0]]),
                    b_ub: Some(vec![0.0]),
                    lb: Some(vec![Some(1.0)]),
                    ..Default::default()
                },
                LPStatus::Infeasible,
            ),
            (
                "unbounded",
                LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    ..Default::default()
                },
                LPStatus::Unbounded,
            ),
        ];
        for (case_name, problem, expected_status) in lp_status_cases {
            let internal = solve_lp_internal(&problem, &InternalSimplexOptions::default());
            let external = solve_lp_external(
                &problem,
                &ExternalSolverOptions {
                    method: Some("highs".to_string()),
                    ..Default::default()
                },
            );
            self.check(
                format!("LP {case_name} status internal/HiGHS"),
                internal.status == expected_status && external.status == expected_status,
                format!(
                    "internal={} external={} expected={}",
                    internal.status.as_str(),
                    external.status.as_str(),
                    expected_status.as_str()
                ),
            );
        }
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

        for solver in ExternalLinearCliSolver::open_source_lp().iter().copied() {
            let probe = probe_external_linear_cli_solver(
                ExternalLinearCliKind::Lp,
                &ExternalLinearCliOptions {
                    solver,
                    time_limit_secs: Some(2.0),
                    ..Default::default()
                },
            );
            if probe.status == ExternalLinearCliProbeStatus::NotInstalled {
                println!("  SKIP  LP {}: {}", solver.as_str(), probe.message);
                continue;
            }
            self.check(
                format!("LP {}:rust-cli probe ready", solver.as_str()),
                probe.status == ExternalLinearCliProbeStatus::Ready,
                format!(
                    "status={} command={:?} smoke={:?} message={}",
                    probe.status.as_str(),
                    probe.command,
                    probe.smoke_status.map(|status| status.as_str()),
                    probe.message
                ),
            );
        }

        for solver in ExternalLinearCliSolver::open_source_mip().iter().copied() {
            let probe = probe_external_linear_cli_solver(
                ExternalLinearCliKind::Mip,
                &ExternalLinearCliOptions {
                    solver,
                    time_limit_secs: Some(2.0),
                    ..Default::default()
                },
            );
            if probe.status == ExternalLinearCliProbeStatus::NotInstalled {
                println!("  SKIP  IP/MIP {}: {}", solver.as_str(), probe.message);
                continue;
            }
            self.check(
                format!("IP/MIP {}:rust-cli probe ready", solver.as_str()),
                probe.status == ExternalLinearCliProbeStatus::Ready,
                format!(
                    "status={} command={:?} smoke={:?} message={}",
                    probe.status.as_str(),
                    probe.command,
                    probe.smoke_status.map(|status| status.as_str()),
                    probe.message
                ),
            );
        }

        for solver in ExternalLinearCliSolver::optional_commercial_mip()
            .iter()
            .copied()
        {
            let probe = probe_external_linear_cli_solver(
                ExternalLinearCliKind::Mip,
                &ExternalLinearCliOptions {
                    solver,
                    time_limit_secs: Some(2.0),
                    ..Default::default()
                },
            );
            if matches!(
                probe.status,
                ExternalLinearCliProbeStatus::NotInstalled
                    | ExternalLinearCliProbeStatus::BridgeUnsupported
            ) {
                println!(
                    "  SKIP  IP/MIP commercial {} probe: {}",
                    solver.as_str(),
                    probe.message
                );
                continue;
            }
            self.check(
                format!("IP/MIP commercial {}:rust-cli probe ready", solver.as_str()),
                probe.status == ExternalLinearCliProbeStatus::Ready,
                format!(
                    "status={} command={:?} smoke={:?} message={}",
                    probe.status.as_str(),
                    probe.command,
                    probe.smoke_status.map(|status| status.as_str()),
                    probe.message
                ),
            );
        }

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

        for solver in ExternalLinearCliSolver::open_source_lp().iter().copied() {
            let solver_name = solver.as_str();
            let reference = solve_lp_with_external_cli(
                &lp,
                &ExternalLinearCliOptions {
                    solver,
                    ..Default::default()
                },
            );
            if reference.status == ExternalLinearCliStatus::Unavailable
                && reference.message.contains("not found")
            {
                println!("  SKIP  LP {solver_name}: rust CLI interface executable not found");
                continue;
            }
            self.check(
                format!("LP {solver_name}:rust-cli status optimal"),
                lp_internal.status == LPStatus::Optimal
                    && reference.status == ExternalLinearCliStatus::Optimal,
                format!(
                    "internal={:?} external={} solver={}",
                    lp_internal.status,
                    reference.status.as_str(),
                    reference.solver
                ),
            );
            self.close(
                &format!("LP {solver_name}:rust-cli objective"),
                lp_internal.objective,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("LP {solver_name}:rust-cli x"),
                &lp_internal.x,
                &reference.x,
                1e-8,
            );
        }

        let lp_status_cases = vec![
            (
                "infeasible",
                LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a_ub: Some(vec![vec![1.0]]),
                    b_ub: Some(vec![0.0]),
                    lb: Some(vec![Some(1.0)]),
                    ..Default::default()
                },
                LPStatus::Infeasible,
            ),
            (
                "unbounded",
                LPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    ..Default::default()
                },
                LPStatus::Unbounded,
            ),
        ];
        for (case_name, status_lp, expected_status) in lp_status_cases {
            let status_internal = solve_lp_internal(&status_lp, &InternalSimplexOptions::default());
            let status_json = serde_json::json!({
                "lp": {
                    "sense": status_lp.sense.as_str(),
                    "c": &status_lp.c,
                    "a_ub": &status_lp.a_ub,
                    "b_ub": &status_lp.b_ub,
                    "a_eq": &status_lp.a_eq,
                    "b_eq": &status_lp.b_eq,
                    "lb": &status_lp.lb,
                    "ub": &status_lp.ub,
                }
            })
            .to_string();
            self.check(
                format!("LP {case_name}:internal status"),
                status_internal.status == expected_status,
                format!(
                    "internal={} expected={}",
                    status_internal.status.as_str(),
                    expected_status.as_str()
                ),
            );
            for solver in lp_solvers.iter().copied() {
                let reference = self.run_linear_cli_reference("lp", solver, &status_json);
                if reference.status == "unavailable" && reference.message.contains("not found") {
                    println!("  SKIP  LP {case_name} {solver}: executable not found");
                    continue;
                }
                self.check(
                    format!("LP {case_name} {solver}:cli status"),
                    reference.status == expected_status.as_str(),
                    format!(
                        "external={} expected={} solver={}",
                        reference.status,
                        expected_status.as_str(),
                        reference.solver
                    ),
                );
            }
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

        for solver in ExternalLinearCliSolver::open_source_mip().iter().copied() {
            let solver_name = solver.as_str();
            let reference = solve_ipmip_with_external_cli(
                &mip,
                &ExternalLinearCliOptions {
                    solver,
                    ..Default::default()
                },
            );
            if reference.status == ExternalLinearCliStatus::Unavailable
                && reference.message.contains("not found")
            {
                println!("  SKIP  IP/MIP {solver_name}: rust CLI interface executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP {solver_name}:rust-cli status optimal"),
                mip_internal.status == IPMIPStatus::Optimal
                    && reference.status == ExternalLinearCliStatus::Optimal,
                format!(
                    "internal={} external={} solver={}",
                    mip_internal.status.as_str(),
                    reference.status.as_str(),
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP {solver_name}:rust-cli objective"),
                mip_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP {solver_name}:rust-cli x"),
                &mip_internal.x,
                &reference.x,
                1e-8,
            );
        }

        let mip_status_cases = vec![
            (
                "infeasible",
                IPMIPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a: vec![vec![1.0], vec![-1.0]],
                    b: vec![0.0, -1.0],
                    integer_vars: vec![true],
                    ub: Some(vec![1.0]),
                    var_names: Some(vec!["x".to_string()]),
                    con_names: Some(vec!["x_le_0".to_string(), "x_ge_1".to_string()]),
                    lazy_constraints: None,
                    variable_nodes: None,
                    constraint_nodes: None,
                },
                IPMIPStatus::Infeasible,
            ),
            (
                "unbounded",
                IPMIPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a: vec![vec![0.0]],
                    b: vec![0.0],
                    integer_vars: vec![false],
                    ub: None,
                    var_names: Some(vec!["x".to_string()]),
                    con_names: Some(vec!["dummy".to_string()]),
                    lazy_constraints: None,
                    variable_nodes: None,
                    constraint_nodes: None,
                },
                IPMIPStatus::Unbounded,
            ),
        ];
        for (case_name, status_mip, expected_status) in mip_status_cases {
            let status_internal = solve_ipmip_with_des(
                status_mip.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
            );
            let status_json = serde_json::json!({
                "sense": status_mip.sense.as_str(),
                "c": &status_mip.c,
                "a": &status_mip.a,
                "b": &status_mip.b,
                "integer_vars": &status_mip.integer_vars,
                "ub": &status_mip.ub,
                "var_names": &status_mip.var_names,
                "con_names": &status_mip.con_names,
            })
            .to_string();
            self.check(
                format!("IP/MIP {case_name}:internal status"),
                status_internal.status == expected_status,
                format!(
                    "internal={} expected={}",
                    status_internal.status.as_str(),
                    expected_status.as_str()
                ),
            );
            for solver in mip_solvers.iter().copied() {
                let reference = self.run_linear_cli_reference("mip", solver, &status_json);
                if reference.status == "unavailable" && reference.message.contains("not found") {
                    println!("  SKIP  IP/MIP {case_name} {solver}: executable not found");
                    continue;
                }
                self.check(
                    format!("IP/MIP {case_name} {solver}:cli status"),
                    reference.status == expected_status.as_str(),
                    format!(
                        "external={} expected={} solver={}",
                        reference.status,
                        expected_status.as_str(),
                        reference.solver
                    ),
                );
            }
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

        let abs_problem = build_absolute_value_penalty_ip();
        let (linearized_abs, _, abs_original_vars) = linearize_source_ipmip_problem(&abs_problem);
        let abs_internal = solve_source_ipmip_with_des(
            abs_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let abs_base = &abs_problem.base;
        let abs_json = serde_json::json!({
            "sense": abs_base.sense.as_str(),
            "c": &abs_base.c,
            "a": &abs_base.a,
            "b": &abs_base.b,
            "integer_vars": &abs_base.integer_vars,
            "lb": &abs_problem.lb,
            "ub": &abs_base.ub,
            "var_names": &abs_base.var_names,
            "con_names": &abs_base.con_names,
            "abs": abs_problem.abs.iter().map(|constraint| serde_json::json!({
                "arg_var": constraint.arg_var,
                "target_var": constraint.target_var,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &abs_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP abs-value {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP abs-value {solver}:cli status optimal"),
                abs_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    abs_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP abs-value {solver}:cli objective"),
                abs_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP abs-value {solver}:cli expanded x length"),
                reference.x.len() == linearized_abs.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_abs.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= abs_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP abs-value {solver}:cli original x"),
                    &abs_internal.x[..abs_original_vars],
                    &reference.x[..abs_original_vars],
                    1e-8,
                );
            }
        }

        let maximum_problem = build_maximum_peak_ip();
        let (linearized_maximum, _, maximum_original_vars) =
            linearize_source_ipmip_problem(&maximum_problem);
        let maximum_internal = solve_source_ipmip_with_des(
            maximum_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let maximum_base = &maximum_problem.base;
        let maximum_json = serde_json::json!({
            "sense": maximum_base.sense.as_str(),
            "c": &maximum_base.c,
            "a": &maximum_base.a,
            "b": &maximum_base.b,
            "integer_vars": &maximum_base.integer_vars,
            "lb": &maximum_problem.lb,
            "ub": &maximum_base.ub,
            "var_names": &maximum_base.var_names,
            "con_names": &maximum_base.con_names,
            "maximums": maximum_problem.maximums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &maximum_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP maximum {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP maximum {solver}:cli status optimal"),
                maximum_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    maximum_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP maximum {solver}:cli objective"),
                maximum_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP maximum {solver}:cli expanded x length"),
                reference.x.len() == linearized_maximum.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_maximum.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= maximum_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP maximum {solver}:cli original x"),
                    &maximum_internal.x[..maximum_original_vars],
                    &reference.x[..maximum_original_vars],
                    1e-8,
                );
            }
        }

        let minimum_problem = build_minimum_floor_ip();
        let (linearized_minimum, _, minimum_original_vars) =
            linearize_source_ipmip_problem(&minimum_problem);
        let minimum_internal = solve_source_ipmip_with_des(
            minimum_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let minimum_base = &minimum_problem.base;
        let minimum_json = serde_json::json!({
            "sense": minimum_base.sense.as_str(),
            "c": &minimum_base.c,
            "a": &minimum_base.a,
            "b": &minimum_base.b,
            "integer_vars": &minimum_base.integer_vars,
            "lb": &minimum_problem.lb,
            "ub": &minimum_base.ub,
            "var_names": &minimum_base.var_names,
            "con_names": &minimum_base.con_names,
            "minimums": minimum_problem.minimums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &minimum_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP minimum {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP minimum {solver}:cli status optimal"),
                minimum_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    minimum_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP minimum {solver}:cli objective"),
                minimum_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP minimum {solver}:cli expanded x length"),
                reference.x.len() == linearized_minimum.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_minimum.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= minimum_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP minimum {solver}:cli original x"),
                    &minimum_internal.x[..minimum_original_vars],
                    &reference.x[..minimum_original_vars],
                    1e-8,
                );
            }
        }

        let logical_problem = build_logical_gate_ip();
        let (linearized_logical, _, logical_original_vars) =
            linearize_source_ipmip_problem(&logical_problem);
        let logical_internal = solve_source_ipmip_with_des(
            logical_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let logical_base = &logical_problem.base;
        let logical_json = serde_json::json!({
            "sense": logical_base.sense.as_str(),
            "c": &logical_base.c,
            "a": &logical_base.a,
            "b": &logical_base.b,
            "integer_vars": &logical_base.integer_vars,
            "lb": &logical_problem.lb,
            "ub": &logical_base.ub,
            "var_names": &logical_base.var_names,
            "con_names": &logical_base.con_names,
            "logical": logical_problem.logical.iter().map(|constraint| serde_json::json!({
                "kind": constraint.kind.as_str(),
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &logical_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP logical {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP logical {solver}:cli status optimal"),
                logical_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    logical_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP logical {solver}:cli objective"),
                logical_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP logical {solver}:cli expanded x length"),
                reference.x.len() == linearized_logical.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_logical.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= logical_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP logical {solver}:cli original x"),
                    &logical_internal.x[..logical_original_vars],
                    &reference.x[..logical_original_vars],
                    1e-8,
                );
            }
        }

        let l1_problem = build_l1_norm_deviation_ip();
        let (linearized_l1, _, l1_original_vars) = linearize_source_ipmip_problem(&l1_problem);
        let l1_internal = solve_source_ipmip_with_des(
            l1_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let l1_base = &l1_problem.base;
        let l1_json = serde_json::json!({
            "sense": l1_base.sense.as_str(),
            "c": &l1_base.c,
            "a": &l1_base.a,
            "b": &l1_base.b,
            "integer_vars": &l1_base.integer_vars,
            "lb": &l1_problem.lb,
            "ub": &l1_base.ub,
            "var_names": &l1_base.var_names,
            "con_names": &l1_base.con_names,
            "l1_norms": l1_problem.l1_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &l1_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP L1 norm {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP L1 norm {solver}:cli status optimal"),
                l1_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    l1_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP L1 norm {solver}:cli objective"),
                l1_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP L1 norm {solver}:cli expanded x length"),
                reference.x.len() == linearized_l1.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_l1.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= l1_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP L1 norm {solver}:cli original x"),
                    &l1_internal.x[..l1_original_vars],
                    &reference.x[..l1_original_vars],
                    1e-8,
                );
            }
        }

        let linf_problem = build_linf_norm_deviation_ip();
        let (linearized_linf, _, linf_original_vars) =
            linearize_source_ipmip_problem(&linf_problem);
        let linf_internal = solve_source_ipmip_with_des(
            linf_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let linf_base = &linf_problem.base;
        let linf_json = serde_json::json!({
            "sense": linf_base.sense.as_str(),
            "c": &linf_base.c,
            "a": &linf_base.a,
            "b": &linf_base.b,
            "integer_vars": &linf_base.integer_vars,
            "lb": &linf_problem.lb,
            "ub": &linf_base.ub,
            "var_names": &linf_base.var_names,
            "con_names": &linf_base.con_names,
            "linf_norms": linf_problem.linf_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &linf_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP Linf norm {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP Linf norm {solver}:cli status optimal"),
                linf_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    linf_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP Linf norm {solver}:cli objective"),
                linf_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP Linf norm {solver}:cli expanded x length"),
                reference.x.len() == linearized_linf.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_linf.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= linf_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP Linf norm {solver}:cli original x"),
                    &linf_internal.x[..linf_original_vars],
                    &reference.x[..linf_original_vars],
                    1e-8,
                );
            }
        }

        for (product_name, product_problem) in vec![
            ("activation", build_product_activation_ip()),
            ("binary-gate", build_binary_product_gate_ip()),
        ] {
            let (linearized_product, _, product_original_vars) =
                linearize_source_ipmip_problem(&product_problem);
            let product_internal = solve_source_ipmip_with_des(
                product_problem.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
            );
            let product_base = &product_problem.base;
            let product_json = serde_json::json!({
                "sense": product_base.sense.as_str(),
                "c": &product_base.c,
                "a": &product_base.a,
                "b": &product_base.b,
                "integer_vars": &product_base.integer_vars,
                "lb": &product_problem.lb,
                "ub": &product_base.ub,
                "var_names": &product_base.var_names,
                "con_names": &product_base.con_names,
                "products": product_problem.products.iter().map(|constraint| serde_json::json!({
                    "target_var": constraint.target_var,
                    "x_var": constraint.x_var,
                    "y_var": constraint.y_var,
                    "name": &constraint.name,
                })).collect::<Vec<_>>(),
            })
            .to_string();

            for solver in mip_solvers.iter().copied() {
                let reference = self.run_linear_cli_reference("mip", solver, &product_json);
                if reference.status == "unavailable" && reference.message.contains("not found") {
                    println!(
                        "  SKIP  IP/MIP product {product_name} {solver}: executable not found"
                    );
                    continue;
                }
                self.check(
                    format!("IP/MIP product {product_name} {solver}:cli status optimal"),
                    product_internal.status == IPMIPStatus::Optimal
                        && reference.status == "optimal",
                    format!(
                        "internal={} external={} solver={}",
                        product_internal.status.as_str(),
                        reference.status,
                        reference.solver
                    ),
                );
                self.close(
                    &format!("IP/MIP product {product_name} {solver}:cli objective"),
                    product_internal.z,
                    reference.objective.unwrap_or(f64::NAN),
                    1e-9,
                );
                self.check(
                    format!("IP/MIP product {product_name} {solver}:cli expanded x length"),
                    reference.x.len() == linearized_product.c.len(),
                    format!(
                        "expected={} actual={}",
                        linearized_product.c.len(),
                        reference.x.len()
                    ),
                );
                if reference.x.len() >= product_original_vars {
                    self.max_abs_close(
                        &format!("IP/MIP product {product_name} {solver}:cli original x"),
                        &product_internal.x[..product_original_vars],
                        &reference.x[..product_original_vars],
                        1e-8,
                    );
                }
            }
        }

        let quadratic_problem = build_quadratic_objective_mix_ip();
        let (linearized_quadratic, _, quadratic_original_vars) =
            linearize_quadratic_objective_problem(&quadratic_problem);
        let quadratic_internal = solve_quadratic_objective_ipmip_with_des(
            quadratic_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let quadratic_base = &quadratic_problem.base;
        let quadratic_json = serde_json::json!({
            "sense": quadratic_base.sense.as_str(),
            "c": &quadratic_base.c,
            "a": &quadratic_base.a,
            "b": &quadratic_base.b,
            "integer_vars": &quadratic_base.integer_vars,
            "lb": &quadratic_problem.lb,
            "ub": &quadratic_base.ub,
            "var_names": &quadratic_base.var_names,
            "con_names": &quadratic_base.con_names,
            "quadratic_objective": quadratic_problem.quadratic_objective.iter().map(|term| serde_json::json!({
                "x_var": term.x_var,
                "y_var": term.y_var,
                "coeff": term.coeff,
                "name": &term.name,
            })).collect::<Vec<_>>(),
        })
        .to_string();

        for solver in mip_solvers.iter().copied() {
            let reference = self.run_linear_cli_reference("mip", solver, &quadratic_json);
            if reference.status == "unavailable" && reference.message.contains("not found") {
                println!("  SKIP  IP/MIP quadratic objective {solver}: executable not found");
                continue;
            }
            self.check(
                format!("IP/MIP quadratic objective {solver}:cli status optimal"),
                quadratic_internal.status == IPMIPStatus::Optimal && reference.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    quadratic_internal.status.as_str(),
                    reference.status,
                    reference.solver
                ),
            );
            self.close(
                &format!("IP/MIP quadratic objective {solver}:cli objective"),
                quadratic_internal.z,
                reference.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            self.check(
                format!("IP/MIP quadratic objective {solver}:cli expanded x length"),
                reference.x.len() == linearized_quadratic.c.len(),
                format!(
                    "expected={} actual={}",
                    linearized_quadratic.c.len(),
                    reference.x.len()
                ),
            );
            if reference.x.len() >= quadratic_original_vars {
                self.max_abs_close(
                    &format!("IP/MIP quadratic objective {solver}:cli original x"),
                    &quadratic_internal.x[..quadratic_original_vars],
                    &reference.x[..quadratic_original_vars],
                    1e-8,
                );
            }
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
            "abs": source_problem.abs.iter().map(|constraint| serde_json::json!({
                "arg_var": constraint.arg_var,
                "target_var": constraint.target_var,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "maximums": source_problem.maximums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "minimums": source_problem.minimums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "logical": source_problem.logical.iter().map(|constraint| serde_json::json!({
                "kind": constraint.kind.as_str(),
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "l1_norms": source_problem.l1_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "linf_norms": source_problem.linf_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "products": source_problem.products.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "x_var": constraint.x_var,
                "y_var": constraint.y_var,
                "name": &constraint.name,
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

    fn validate_external_optimization_ecosystems(&mut self) {
        println!("\n-- Optional Java/Rust optimization ecosystems: classpath and Cargo probes --");
        for tool in ExternalOptimizationTool::all().iter().copied() {
            let probe = probe_external_optimization_tool(tool);
            if matches!(
                probe.status,
                ExternalOptimizationProbeStatus::NotConfigured
                    | ExternalOptimizationProbeStatus::RuntimeMissing
            ) {
                println!(
                    "  SKIP  {} {}: {}",
                    probe.ecosystem.as_str(),
                    tool.as_str(),
                    probe.message
                );
                continue;
            }
            self.check(
                format!(
                    "{} {} ecosystem probe ready",
                    probe.ecosystem.as_str(),
                    tool.as_str()
                ),
                probe.status == ExternalOptimizationProbeStatus::Ready,
                format!(
                    "status={} command={:?} env={} artifact={:?} message={}",
                    probe.status.as_str(),
                    probe.command,
                    probe.env_var,
                    probe.artifact,
                    probe.message
                ),
            );
        }
    }

    fn check_math_program_cli_cross_check(
        &mut self,
        label: &str,
        program: &MathProgram,
        solve_opts: &MathProgramSolveOptions,
        kind: ExternalLinearCliKind,
        solver: ExternalLinearCliSolver,
        method: &str,
    ) {
        let probe = probe_external_linear_cli_solver(
            kind,
            &ExternalLinearCliOptions {
                solver,
                time_limit_secs: Some(5.0),
                ..Default::default()
            },
        );
        if probe.status == ExternalLinearCliProbeStatus::NotInstalled {
            println!(
                "  SKIP  MathProgram {label} facade {}:cli cross-check: {}",
                solver.as_str(),
                probe.message
            );
            return;
        }
        if probe.status != ExternalLinearCliProbeStatus::Ready {
            self.check(
                format!(
                    "MathProgram {label} facade {}:cli probe ready",
                    solver.as_str()
                ),
                false,
                format!(
                    "status={} command={:?} smoke={:?} message={}",
                    probe.status.as_str(),
                    probe.command,
                    probe.smoke_status.map(|status| status.as_str()),
                    probe.message
                ),
            );
            return;
        }

        let external_opts = ExternalMathProgramOptions {
            method: Some(method.to_string()),
            time_limit_ms: Some(5_000.0),
            ..Default::default()
        };
        match cross_check_math_program_with_external(program, solve_opts, &external_opts, 1e-7) {
            Ok(report) => self.check(
                format!(
                    "MathProgram {label} facade {}:cli same-input cross-check",
                    solver.as_str()
                ),
                report.within_tolerance
                    && report.internal.status == MathProgramStatus::Optimal
                    && report.external.status == MathProgramStatus::Optimal
                    && report.max_x_abs_diff.is_some_and(|diff| diff <= 1e-7),
                format!(
                    "method={} internal={:?} external={:?} obj_diff={:?} x_diff={:?} violations=({:?},{:?})",
                    method,
                    report.internal.status,
                    report.external.status,
                    report.objective_abs_diff,
                    report.max_x_abs_diff,
                    report.internal_max_violation,
                    report.external_max_violation
                ),
            ),
            Err(err) => self.check(
                format!(
                    "MathProgram {label} facade {}:cli same-input cross-check",
                    solver.as_str()
                ),
                false,
                format!("{err:?}"),
            ),
        }
    }

    fn validate_math_program_facade(&mut self) {
        println!("\n-- Math-program facade: native lowering vs external solver oracles --");
        let solve_opts = MathProgramSolveOptions::default();
        let external_opts = ExternalMathProgramOptions {
            method: Some("highs".to_string()),
            time_limit_ms: Some(5_000.0),
            ..Default::default()
        };

        let mut lp = MathProgram::new(MathObjectiveSense::Max);
        let lp_x = lp
            .add_continuous_var("x", 5.0, Some(0.0), Some(4.0))
            .expect("LP x");
        let lp_y = lp
            .add_continuous_var("y", 3.0, Some(0.0), Some(4.0))
            .expect("LP y");
        let lp_z = lp
            .add_continuous_var("z", 0.5, Some(0.0), Some(5.0))
            .expect("LP z");
        lp.add_constraint(
            "balance",
            vec![(lp_x, 1.0), (lp_y, 1.0), (lp_z, 1.0)],
            RowSense::Eq,
            5.0,
        )
        .expect("LP balance");
        lp.add_constraint(
            "preference",
            vec![(lp_x, 1.0), (lp_y, -1.0)],
            RowSense::Ge,
            1.0,
        )
        .expect("LP preference");

        match cross_check_math_program_with_external(&lp, &solve_opts, &external_opts, 1e-7) {
            Ok(report) => self.check(
                "MathProgram LP facade same-input HiGHS cross-check",
                report.within_tolerance
                    && report.internal.status == MathProgramStatus::Optimal
                    && report.external.status == MathProgramStatus::Optimal
                    && report.max_x_abs_diff.is_some_and(|diff| diff <= 1e-7),
                format!(
                    "internal={:?} external={:?} obj_diff={:?} x_diff={:?} violations=({:?},{:?})",
                    report.internal.status,
                    report.external.status,
                    report.objective_abs_diff,
                    report.max_x_abs_diff,
                    report.internal_max_violation,
                    report.external_max_violation
                ),
            ),
            Err(err) => self.check(
                "MathProgram LP facade same-input HiGHS cross-check",
                false,
                format!("{err:?}"),
            ),
        }

        match export_math_program_cplex_lp(&lp) {
            Ok(export) => self.check(
                "MathProgram LP facade CPLEX-LP export",
                !export.is_mip
                    && export.original_variable_count == 3
                    && export.variable_names.len() == 3
                    && export.text.contains("Maximize\n")
                    && export.text.contains("Subject To\n")
                    && export.text.contains("Bounds\n")
                    && export.text.ends_with("End\n"),
                format!(
                    "vars={} rows={} bytes={}",
                    export.variable_names.len(),
                    export.constraint_names.len(),
                    export.text.len()
                ),
            ),
            Err(err) => self.check(
                "MathProgram LP facade CPLEX-LP export",
                false,
                format!("{err:?}"),
            ),
        }

        for (solver, method) in [
            (ExternalLinearCliSolver::Highs, "highs:cli"),
            (ExternalLinearCliSolver::Glpk, "glpsol:cli"),
            (ExternalLinearCliSolver::Scip, "scip:cli"),
            (ExternalLinearCliSolver::Cbc, "cbc:cli"),
        ] {
            self.check_math_program_cli_cross_check(
                "LP",
                &lp,
                &solve_opts,
                ExternalLinearCliKind::Lp,
                solver,
                method,
            );
        }

        let mut mip = MathProgram::new(MathObjectiveSense::Max);
        let open_a = mip.add_binary_var("open-a", 4.0).expect("open a");
        let open_b = mip.add_binary_var("open-b", 3.0).expect("open b");
        let load = mip
            .add_integer_var("load", 2.0, Some(0.0), Some(4.0))
            .expect("load");
        let reserve = mip
            .add_integer_var("reserve", 0.0, Some(0.0), Some(2.0))
            .expect("reserve");
        let peak = mip
            .add_continuous_var("peak", -1.0, Some(0.0), Some(4.0))
            .expect("peak");
        mip.add_exactly_one("choose-one-site", vec![open_a, open_b])
            .expect("exactly one");
        mip.add_constraint(
            "crew-capacity",
            vec![(load, 1.0), (reserve, 1.0)],
            RowSense::Le,
            4.0,
        )
        .expect("crew capacity");
        mip.add_indicator(
            "open-a-min-load",
            open_a,
            true,
            vec![(load, 1.0)],
            RowSense::Ge,
            3.0,
        )
        .expect("open a indicator");
        mip.add_indicator(
            "open-b-max-load",
            open_b,
            true,
            vec![(load, 1.0)],
            RowSense::Le,
            1.0,
        )
        .expect("open b indicator");
        mip.add_max("peak-load", peak, vec![load, reserve])
            .expect("peak max");

        match cross_check_math_program_with_external(&mip, &solve_opts, &external_opts, 1e-7) {
            Ok(report) => self.check(
                "MathProgram MIP facade indicator/general-constraint cross-check",
                report.within_tolerance
                    && report.internal.status == MathProgramStatus::Optimal
                    && report.external.status == MathProgramStatus::Optimal
                    && report.max_x_abs_diff.is_some_and(|diff| diff <= 1e-7),
                format!(
                    "internal={:?} external={:?} obj_diff={:?} x_diff={:?} violations=({:?},{:?})",
                    report.internal.status,
                    report.external.status,
                    report.objective_abs_diff,
                    report.max_x_abs_diff,
                    report.internal_max_violation,
                    report.external_max_violation
                ),
            ),
            Err(err) => self.check(
                "MathProgram MIP facade indicator/general-constraint cross-check",
                false,
                format!("{err:?}"),
            ),
        }

        match export_math_program_cplex_lp(&mip) {
            Ok(export) => self.check(
                "MathProgram MIP facade compiled CPLEX-LP export",
                export.is_mip
                    && export.original_variable_count == 5
                    && export.variable_names.len() > export.original_variable_count
                    && export.text.contains("Maximize\n")
                    && export.text.contains("Subject To\n")
                    && export.text.contains("Bounds\n")
                    && export.text.contains("Binaries\n")
                    && export.text.ends_with("End\n"),
                format!(
                    "original_vars={} exported_vars={} rows={} bytes={}",
                    export.original_variable_count,
                    export.variable_names.len(),
                    export.constraint_names.len(),
                    export.text.len()
                ),
            ),
            Err(err) => self.check(
                "MathProgram MIP facade compiled CPLEX-LP export",
                false,
                format!("{err:?}"),
            ),
        }

        for (solver, method) in [
            (ExternalLinearCliSolver::Highs, "highs:cli"),
            (ExternalLinearCliSolver::Glpk, "glpsol:cli"),
            (ExternalLinearCliSolver::Scip, "scip:cli"),
            (ExternalLinearCliSolver::Cbc, "cbc:cli"),
        ] {
            self.check_math_program_cli_cross_check(
                "MIP",
                &mip,
                &solve_opts,
                ExternalLinearCliKind::Mip,
                solver,
                method,
            );
        }

        let mut conflict_model = MathProgram::new(MathObjectiveSense::Min);
        let conflict_x = conflict_model
            .add_continuous_var("x", 0.0, None, None)
            .expect("conflict x");
        let conflict_y = conflict_model
            .add_continuous_var("y", 0.0, Some(0.0), None)
            .expect("conflict y");
        conflict_model
            .add_constraint("x-at-least-two", vec![(conflict_x, 1.0)], RowSense::Ge, 2.0)
            .expect("conflict lower row");
        conflict_model
            .add_constraint("x-at-most-one", vec![(conflict_x, 1.0)], RowSense::Le, 1.0)
            .expect("conflict upper row");
        conflict_model
            .add_constraint("redundant-y", vec![(conflict_y, 1.0)], RowSense::Ge, 0.0)
            .expect("conflict redundant row");

        match cross_check_math_program_conflict_with_external(
            &conflict_model,
            &solve_opts,
            &external_opts,
            &MathProgramConflictOptions::default(),
        ) {
            Ok(report) => self.check(
                "MathProgram conflict refiner subsystem external cross-check",
                report.within_tolerance
                    && report.internal.minimal
                    && report.internal.items.len() == 2
                    && report.external.status == MathProgramStatus::Infeasible,
                format!(
                    "internal={:?} external={:?} items={} minimal={} status_agree={}",
                    report.internal.status,
                    report.external.status,
                    report.internal.items.len(),
                    report.internal.minimal,
                    report.status_agree
                ),
            ),
            Err(err) => self.check(
                "MathProgram conflict refiner subsystem external cross-check",
                false,
                format!("{err:?}"),
            ),
        }

        let mut relax_model = MathProgram::new(MathObjectiveSense::Min);
        let relax_x = relax_model
            .add_continuous_var("x", 0.0, Some(2.0), None)
            .expect("relax x");
        relax_model
            .add_constraint("cap", vec![(relax_x, 1.0)], RowSense::Le, 1.0)
            .expect("relax cap");

        match cross_check_math_program_feas_relaxation_with_external(
            &relax_model,
            &solve_opts,
            &external_opts,
            &MathProgramFeasRelaxOptions {
                linear_penalty: 10.0,
                bound_penalty: 1.0,
                ..Default::default()
            },
            1e-7,
        ) {
            Ok(report) => self.check(
                "MathProgram feasibility-relaxation external cross-check",
                report.within_tolerance
                    && report.internal.status == MathProgramStatus::Optimal
                    && report.external.status == MathProgramStatus::Optimal
                    && (report.internal.violation_objective - 1.0).abs() <= 1e-7,
                format!(
                    "internal={:?} external={:?} violation_obj={} obj_diff={:?} violations={}",
                    report.internal.status,
                    report.external.status,
                    report.internal.violation_objective,
                    report.objective_abs_diff,
                    report.internal.violations.len()
                ),
            ),
            Err(err) => self.check(
                "MathProgram feasibility-relaxation external cross-check",
                false,
                format!("{err:?}"),
            ),
        }

        let mut pool_model = MathProgram::new(MathObjectiveSense::Max);
        let pool_a = pool_model.add_binary_var("a", 4.0).expect("pool a");
        let pool_b = pool_model.add_binary_var("b", 2.0).expect("pool b");
        let pool_c = pool_model.add_binary_var("c", 1.0).expect("pool c");
        pool_model
            .add_constraint(
                "choose-at-most-two",
                vec![(pool_a, 1.0), (pool_b, 1.0), (pool_c, 1.0)],
                RowSense::Le,
                2.0,
            )
            .expect("pool capacity");

        match cross_check_math_program_solution_pool_with_external(
            &pool_model,
            &solve_opts,
            &external_opts,
            &MathProgramSolutionPoolOptions {
                max_solutions: 3,
                ..Default::default()
            },
            1e-7,
        ) {
            Ok(report) => self.check(
                "MathProgram solution-pool external cross-check",
                report.within_tolerance
                    && report.len_agree
                    && report.internal.solutions.len() == 3
                    && !report.internal.exhausted,
                format!(
                    "internal_len={} external_len={} exhausted=({},{}) obj_diffs={:?} x_diffs={:?}",
                    report.internal.solutions.len(),
                    report.external.solutions.len(),
                    report.internal.exhausted,
                    report.external.exhausted,
                    report.objective_abs_diffs,
                    report.max_x_abs_diffs
                ),
            ),
            Err(err) => self.check(
                "MathProgram solution-pool external cross-check",
                false,
                format!("{err:?}"),
            ),
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
            "c": &p.c,
            "a": &p.a,
            "b": &p.b,
            "integer_vars": &p.integer_vars,
            "ub": &p.ub,
            "var_names": &p.var_names,
            "con_names": &p.con_names,
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

        let pool_problem = IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec!["choose_a".to_string(), "choose_b".to_string()]),
            con_names: Some(vec!["choose_at_most_one".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let pool_internal = solve_ipmip_solution_pool_with_des(
            pool_problem.clone(),
            IPMIPSolutionPoolOptions {
                max_solutions: Some(4),
                solve_options: IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let pool_problem_path = out_dir.join("solution-pool-problem.json");
        let pool_reference_path = out_dir.join("solution-pool-reference.json");
        let pool_problem_json = serde_json::json!({
            "sense": pool_problem.sense.as_str(),
            "c": &pool_problem.c,
            "a": &pool_problem.a,
            "b": &pool_problem.b,
            "integer_vars": &pool_problem.integer_vars,
            "ub": &pool_problem.ub,
            "var_names": &pool_problem.var_names,
            "con_names": &pool_problem.con_names,
        });
        std::fs::write(
            &pool_problem_path,
            serde_json::to_string_pretty(&pool_problem_json)
                .expect("serialize solution-pool MIP problem"),
        )
        .expect("write solution-pool MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&pool_problem_path)
            .arg("--out")
            .arg(&pool_reference_path)
            .arg("--solver")
            .arg("auto")
            .arg("--pool-size")
            .arg("4")
            .output()
            .expect("run solution-pool MIP reference");
        if !output.status.success() {
            panic!(
                "ip_mip_reference solution pool failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let pool_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&pool_reference_path).expect("read solution-pool MIP reference JSON"),
        )
        .expect("parse solution-pool MIP reference JSON");
        let pool_reference_solutions = pool_reference.result.solutions.as_deref().unwrap_or(&[]);
        self.check(
            "IP/MIP solution-pool statuses optimal",
            pool_internal.status == IPMIPStatus::Optimal
                && pool_reference.result.status == "optimal"
                && pool_internal.exhausted
                && pool_reference.result.exhausted == Some(true),
            format!(
                "internal={} external={} exhausted={}/{} solver={}",
                pool_internal.status.as_str(),
                pool_reference.result.status,
                pool_internal.exhausted,
                pool_reference.result.exhausted.unwrap_or(false),
                pool_reference.result.solver
            ),
        );
        self.check(
            "IP/MIP solution-pool length",
            pool_internal.solutions.len() == pool_reference_solutions.len()
                && pool_internal.solutions.len() == 3,
            format!(
                "internal={} external={}",
                pool_internal.solutions.len(),
                pool_reference_solutions.len()
            ),
        );
        for (idx, (internal_solution, reference_solution)) in pool_internal
            .solutions
            .iter()
            .zip(pool_reference_solutions.iter())
            .enumerate()
        {
            self.close(
                &format!("IP/MIP solution-pool objective[{idx}]"),
                internal_solution.z,
                reference_solution.objective,
                1e-9,
            );
            self.max_abs_close(
                &format!("IP/MIP solution-pool x[{idx}]"),
                &internal_solution.x,
                &reference_solution.x,
                1e-8,
            );
        }

        let conflict_problem = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a: vec![vec![1.0], vec![-1.0], vec![1.0]],
            b: vec![0.5, -0.5, 10.0],
            integer_vars: vec![true],
            ub: None,
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_le_half".to_string(),
                "x_ge_half".to_string(),
                "redundant_cap".to_string(),
            ]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let conflict =
            find_ipmip_infeasibility_conflict(&conflict_problem, &IPMIPConflictOptions::default());
        self.check(
            "IP/MIP infeasibility conflict minimal",
            conflict.infeasible && conflict.minimal,
            format!(
                "members={:?} checks={} message={:?}",
                conflict.members, conflict.checks, conflict.message
            ),
        );
        let expected_conflict = vec![
            IPMIPConflictMember::LinearRow(0),
            IPMIPConflictMember::LinearRow(1),
            IPMIPConflictMember::Integrality(0),
        ];
        self.check(
            "IP/MIP infeasibility conflict expected members",
            conflict.members == expected_conflict,
            format!("members={:?}", conflict.members),
        );
        let conflict_subproblem =
            ipmip_feasibility_problem_from_conflict_members(&conflict_problem, &conflict.members);
        let conflict_ub = conflict_subproblem.ub.as_ref().map(|ub| {
            ub.iter()
                .map(|&upper| upper.is_finite().then_some(upper))
                .collect::<Vec<_>>()
        });
        let conflict_json = serde_json::json!({
            "sense": conflict_subproblem.sense.as_str(),
            "c": &conflict_subproblem.c,
            "a": &conflict_subproblem.a,
            "b": &conflict_subproblem.b,
            "integer_vars": &conflict_subproblem.integer_vars,
            "ub": conflict_ub,
            "var_names": &conflict_subproblem.var_names,
            "con_names": &conflict_subproblem.con_names,
        })
        .to_string();
        let conflict_reference = self.run_linear_cli_reference("mip", "highs", &conflict_json);
        self.check(
            "IP/MIP conflict subsystem HiGHS infeasible",
            conflict_reference.status == "infeasible",
            format!(
                "external={} solver={}",
                conflict_reference.status, conflict_reference.solver
            ),
        );
        let mut deletion_statuses = Vec::new();
        let mut all_single_deletions_feasible = true;
        for idx in 0..conflict.members.len() {
            let mut trial = conflict.members.clone();
            trial.remove(idx);
            let trial_subproblem =
                ipmip_feasibility_problem_from_conflict_members(&conflict_problem, &trial);
            let trial_ub = trial_subproblem.ub.as_ref().map(|ub| {
                ub.iter()
                    .map(|&upper| upper.is_finite().then_some(upper))
                    .collect::<Vec<_>>()
            });
            let trial_json = serde_json::json!({
                "sense": trial_subproblem.sense.as_str(),
                "c": &trial_subproblem.c,
                "a": &trial_subproblem.a,
                "b": &trial_subproblem.b,
                "integer_vars": &trial_subproblem.integer_vars,
                "ub": trial_ub,
                "var_names": &trial_subproblem.var_names,
                "con_names": &trial_subproblem.con_names,
            })
            .to_string();
            let reference = self.run_linear_cli_reference("mip", "highs", &trial_json);
            all_single_deletions_feasible &= reference.status == "optimal";
            deletion_statuses.push(reference.status);
        }
        self.check(
            "IP/MIP conflict single-deletion HiGHS feasibility",
            all_single_deletions_feasible,
            format!("statuses={deletion_statuses:?}"),
        );

        let mip_feas_relax_options = IPMIPFeasRelaxOptions {
            row_penalties: Some(vec![3.0, 1.0, 1.0]),
            solve_options: IPMIPSolveOptions {
                max_cut_rounds: Some(0),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                ..Default::default()
            },
            ..Default::default()
        };
        let mip_feas_relax_internal =
            solve_ipmip_feasibility_relaxation_with_des(&conflict_problem, &mip_feas_relax_options);
        self.check(
            "IP/MIP feasibility-relaxation internal status",
            mip_feas_relax_internal.status == IPMIPStatus::Optimal,
            format!(
                "status={} cost={} violations={:?}",
                mip_feas_relax_internal.status.as_str(),
                mip_feas_relax_internal.relaxation_cost,
                mip_feas_relax_internal.violations
            ),
        );
        self.check(
            "IP/MIP feasibility-relaxation weighted violation",
            (mip_feas_relax_internal.relaxation_cost - 0.5).abs() <= 1e-9
                && mip_feas_relax_internal.violations.len() == 1
                && mip_feas_relax_internal.violations[0].member
                    == IPMIPFeasRelaxMember::LinearRow(1)
                && (mip_feas_relax_internal.violations[0].amount - 0.5).abs() <= 1e-9,
            format!(
                "cost={} violations={:?}",
                mip_feas_relax_internal.relaxation_cost, mip_feas_relax_internal.violations
            ),
        );
        let mip_feas_relax_model =
            build_ipmip_feasibility_relaxation_problem(&conflict_problem, &mip_feas_relax_options);
        let mip_feas_relax_reference = solve_ipmip_with_external_cli(
            &mip_feas_relax_model.problem,
            &ExternalLinearCliOptions {
                solver: ExternalLinearCliSolver::Highs,
                time_limit_secs: Some(2.0),
                ..Default::default()
            },
        );
        self.check(
            "IP/MIP feasibility-relaxation HiGHS status optimal",
            mip_feas_relax_reference.status == ExternalLinearCliStatus::Optimal,
            format!(
                "external={} solver={}",
                mip_feas_relax_reference.status.as_str(),
                mip_feas_relax_reference.solver
            ),
        );
        self.close(
            "IP/MIP feasibility-relaxation cost vs HiGHS",
            mip_feas_relax_internal.relaxation_cost,
            mip_feas_relax_reference.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        let mip_feas_relax_external_x =
            if mip_feas_relax_reference.x.len() >= mip_feas_relax_model.original_var_count {
                mip_feas_relax_reference.x[..mip_feas_relax_model.original_var_count].to_vec()
            } else {
                Vec::new()
            };
        self.max_abs_close(
            "IP/MIP feasibility-relaxation x vs HiGHS",
            &mip_feas_relax_internal.x,
            &mip_feas_relax_external_x,
            1e-9,
        );

        let start = vec![1.0, 0.0, 0.0, 0.0];
        let start_limited = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                max_nodes: Some(0),
                mip_start: Some(start.clone()),
                ..Default::default()
            },
        );
        self.check(
            "IP/MIP mip-start zero-node incumbent",
            start_limited.status == IPMIPStatus::MaxNodes
                && start_limited.incumbent_source.as_deref() == Some("user-mip-start"),
            format!(
                "status={} source={:?}",
                start_limited.status.as_str(),
                start_limited.incumbent_source
            ),
        );
        self.close(
            "IP/MIP mip-start zero-node objective",
            start_limited.z,
            10.0,
            1e-9,
        );
        self.max_abs_close(
            "IP/MIP mip-start zero-node x",
            &start_limited.x,
            &start,
            1e-9,
        );

        let branch_priority_problem = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            b: vec![0.5, 0.5],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec![
                "low_priority".to_string(),
                "high_priority".to_string(),
            ]),
            con_names: Some(vec!["cap_low".to_string(), "cap_high".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let branch_priority_internal = solve_ipmip_with_des(
            branch_priority_problem.clone(),
            IPMIPSolveOptions {
                branch_rule: Some(BranchRule::FirstFractional),
                branch_priorities: Some(vec![0, 10]),
                max_cut_rounds: Some(0),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                ..Default::default()
            },
        );
        let first_branch_var = branch_priority_internal
            .trace
            .iter()
            .find(|event| event.action == TraceAction::Branch)
            .and_then(|event| event.branch_var);
        self.check(
            "IP/MIP branch-priority first branch variable",
            first_branch_var == Some(1),
            format!("first_branch_var={first_branch_var:?}"),
        );
        let branch_priority_json = serde_json::json!({
            "sense": branch_priority_problem.sense.as_str(),
            "c": &branch_priority_problem.c,
            "a": &branch_priority_problem.a,
            "b": &branch_priority_problem.b,
            "integer_vars": &branch_priority_problem.integer_vars,
            "ub": &branch_priority_problem.ub,
            "var_names": &branch_priority_problem.var_names,
            "con_names": &branch_priority_problem.con_names,
        })
        .to_string();
        let branch_priority_reference =
            self.run_linear_cli_reference("mip", "highs", &branch_priority_json);
        self.check(
            "IP/MIP branch-priority statuses optimal",
            branch_priority_internal.status == IPMIPStatus::Optimal
                && branch_priority_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                branch_priority_internal.status.as_str(),
                branch_priority_reference.status,
                branch_priority_reference.solver
            ),
        );
        self.close(
            "IP/MIP branch-priority objective",
            branch_priority_internal.z,
            branch_priority_reference.objective.unwrap_or(f64::NAN),
            1e-9,
        );

        let mip_gap_problem = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 10.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            b: vec![0.5, 1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec!["fractional_bonus".to_string(), "accepted".to_string()]),
            con_names: Some(vec!["bonus_cap".to_string(), "accepted_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let mip_gap_internal = solve_ipmip_with_des(
            mip_gap_problem.clone(),
            IPMIPSolveOptions {
                mip_start: Some(vec![0.0, 1.0]),
                mip_gap_rel: Some(0.051),
                max_cut_rounds: Some(0),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                ..Default::default()
            },
        );
        self.check(
            "IP/MIP gap-limit status",
            mip_gap_internal.status == IPMIPStatus::GapLimit,
            format!(
                "status={} z={} bound={} gap={}",
                mip_gap_internal.status.as_str(),
                mip_gap_internal.z,
                mip_gap_internal.best_bound,
                mip_gap_internal.gap
            ),
        );
        self.check(
            "IP/MIP gap-limit relative gap",
            mip_gap_internal.gap <= 0.051 + 1e-9
                && (mip_gap_internal.best_bound - 10.5).abs() <= 1e-9,
            format!(
                "bound={} gap={}",
                mip_gap_internal.best_bound, mip_gap_internal.gap
            ),
        );
        let mip_gap_json = serde_json::json!({
            "sense": mip_gap_problem.sense.as_str(),
            "c": &mip_gap_problem.c,
            "a": &mip_gap_problem.a,
            "b": &mip_gap_problem.b,
            "integer_vars": &mip_gap_problem.integer_vars,
            "ub": &mip_gap_problem.ub,
            "var_names": &mip_gap_problem.var_names,
            "con_names": &mip_gap_problem.con_names,
        })
        .to_string();
        let mip_gap_reference = self.run_linear_cli_reference("mip", "highs", &mip_gap_json);
        self.check(
            "IP/MIP gap-limit HiGHS status optimal",
            mip_gap_reference.status == "optimal",
            format!(
                "external={} solver={}",
                mip_gap_reference.status, mip_gap_reference.solver
            ),
        );
        self.close(
            "IP/MIP gap-limit incumbent objective vs HiGHS",
            mip_gap_internal.z,
            mip_gap_reference.objective.unwrap_or(f64::NAN),
            1e-9,
        );

        let start_internal = solve_ipmip_with_des(
            p.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                )),
                max_cut_rounds: Some(1),
                mip_start: Some(start),
                ..Default::default()
            },
        );
        self.check(
            "IP/MIP mip-start statuses optimal",
            start_internal.status == IPMIPStatus::Optimal && reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                start_internal.status.as_str(),
                reference.result.status,
                reference.result.solver
            ),
        );
        self.close(
            "IP/MIP mip-start objective",
            start_internal.z,
            reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );

        let mip_status_cases = vec![
            (
                "infeasible",
                IPMIPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a: vec![vec![1.0], vec![-1.0]],
                    b: vec![0.0, -1.0],
                    integer_vars: vec![true],
                    ub: Some(vec![1.0]),
                    var_names: Some(vec!["x".to_string()]),
                    con_names: Some(vec!["x_le_0".to_string(), "x_ge_1".to_string()]),
                    lazy_constraints: None,
                    variable_nodes: None,
                    constraint_nodes: None,
                },
                IPMIPStatus::Infeasible,
            ),
            (
                "unbounded",
                IPMIPProblem {
                    sense: Sense::Max,
                    c: vec![1.0],
                    a: vec![vec![0.0]],
                    b: vec![0.0],
                    integer_vars: vec![false],
                    ub: None,
                    var_names: Some(vec!["x".to_string()]),
                    con_names: Some(vec!["dummy".to_string()]),
                    lazy_constraints: None,
                    variable_nodes: None,
                    constraint_nodes: None,
                },
                IPMIPStatus::Unbounded,
            ),
        ];
        for (case_name, status_mip, expected_status) in mip_status_cases {
            let status_internal = solve_ipmip_with_des(
                status_mip.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
            );
            let status_problem_path = out_dir.join(format!("mip-{case_name}-problem.json"));
            let status_reference_path = out_dir.join(format!("mip-{case_name}-reference.json"));
            let status_json = serde_json::json!({
                "sense": status_mip.sense.as_str(),
                "c": &status_mip.c,
                "a": &status_mip.a,
                "b": &status_mip.b,
                "integer_vars": &status_mip.integer_vars,
                "ub": &status_mip.ub,
                "var_names": &status_mip.var_names,
                "con_names": &status_mip.con_names,
            });
            std::fs::write(
                &status_problem_path,
                serde_json::to_string_pretty(&status_json).expect("serialize status MIP problem"),
            )
            .expect("write status MIP problem");
            let output = Command::new(&python)
                .arg(&script)
                .arg("--problem")
                .arg(&status_problem_path)
                .arg("--out")
                .arg(&status_reference_path)
                .arg("--solver")
                .arg("auto")
                .output()
                .expect("run status MIP reference");
            if !output.status.success() {
                panic!(
                    "status ip_mip_reference failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let status_reference: MipReference = serde_json::from_slice(
                &std::fs::read(&status_reference_path).expect("read status MIP reference JSON"),
            )
            .expect("parse status MIP reference JSON");
            self.check(
                format!("IP/MIP {case_name} status internal/reference"),
                status_internal.status == expected_status
                    && status_reference.result.status == expected_status.as_str(),
                format!(
                    "internal={} external={} expected={} solver={}",
                    status_internal.status.as_str(),
                    status_reference.result.status,
                    expected_status.as_str(),
                    status_reference.result.solver
                ),
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

        let abs_problem = build_absolute_value_penalty_ip();
        let (linearized_abs, _, abs_original_vars) = linearize_source_ipmip_problem(&abs_problem);
        let abs_internal = solve_source_ipmip_with_des(
            abs_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let abs_problem_path = out_dir.join("absolute-value-penalty-problem.json");
        let abs_reference_path = out_dir.join("absolute-value-penalty-reference.json");
        let base = &abs_problem.base;
        let abs_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &abs_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "abs": abs_problem.abs.iter().map(|constraint| serde_json::json!({
                "arg_var": constraint.arg_var,
                "target_var": constraint.target_var,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &abs_problem_path,
            serde_json::to_string_pretty(&abs_json).expect("serialize abs-value MIP problem"),
        )
        .expect("write abs-value MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&abs_problem_path)
            .arg("--out")
            .arg(&abs_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run abs-value MIP reference");
        if !output.status.success() {
            panic!(
                "abs-value ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let abs_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&abs_reference_path).expect("read abs-value MIP reference JSON"),
        )
        .expect("parse abs-value MIP reference JSON");
        self.check(
            "IP/MIP abs-value statuses optimal",
            abs_internal.status == IPMIPStatus::Optimal && abs_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                abs_internal.status.as_str(),
                abs_reference.result.status,
                abs_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP abs-value objective",
            abs_internal.z,
            abs_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = abs_reference.result.x.as_deref() {
            self.check(
                "IP/MIP abs-value external x length",
                x.len() == linearized_abs.c.len(),
                "",
            );
            if x.len() >= abs_original_vars {
                self.max_abs_close(
                    "IP/MIP abs-value original x",
                    &abs_internal.x[..abs_original_vars],
                    &x[..abs_original_vars],
                    1e-8,
                );
            }
        }

        let maximum_problem = build_maximum_peak_ip();
        let (linearized_maximum, _, maximum_original_vars) =
            linearize_source_ipmip_problem(&maximum_problem);
        let maximum_internal = solve_source_ipmip_with_des(
            maximum_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let maximum_problem_path = out_dir.join("maximum-peak-problem.json");
        let maximum_reference_path = out_dir.join("maximum-peak-reference.json");
        let base = &maximum_problem.base;
        let maximum_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &maximum_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "maximums": maximum_problem.maximums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &maximum_problem_path,
            serde_json::to_string_pretty(&maximum_json).expect("serialize maximum MIP problem"),
        )
        .expect("write maximum MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&maximum_problem_path)
            .arg("--out")
            .arg(&maximum_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run maximum MIP reference");
        if !output.status.success() {
            panic!(
                "maximum ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let maximum_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&maximum_reference_path).expect("read maximum MIP reference JSON"),
        )
        .expect("parse maximum MIP reference JSON");
        self.check(
            "IP/MIP maximum statuses optimal",
            maximum_internal.status == IPMIPStatus::Optimal
                && maximum_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                maximum_internal.status.as_str(),
                maximum_reference.result.status,
                maximum_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP maximum objective",
            maximum_internal.z,
            maximum_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = maximum_reference.result.x.as_deref() {
            self.check(
                "IP/MIP maximum external x length",
                x.len() == linearized_maximum.c.len(),
                "",
            );
            if x.len() >= maximum_original_vars {
                self.max_abs_close(
                    "IP/MIP maximum original x",
                    &maximum_internal.x[..maximum_original_vars],
                    &x[..maximum_original_vars],
                    1e-8,
                );
            }
        }

        let minimum_problem = build_minimum_floor_ip();
        let (linearized_minimum, _, minimum_original_vars) =
            linearize_source_ipmip_problem(&minimum_problem);
        let minimum_internal = solve_source_ipmip_with_des(
            minimum_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let minimum_problem_path = out_dir.join("minimum-floor-problem.json");
        let minimum_reference_path = out_dir.join("minimum-floor-reference.json");
        let base = &minimum_problem.base;
        let minimum_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &minimum_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "minimums": minimum_problem.minimums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &minimum_problem_path,
            serde_json::to_string_pretty(&minimum_json).expect("serialize minimum MIP problem"),
        )
        .expect("write minimum MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&minimum_problem_path)
            .arg("--out")
            .arg(&minimum_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run minimum MIP reference");
        if !output.status.success() {
            panic!(
                "minimum ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let minimum_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&minimum_reference_path).expect("read minimum MIP reference JSON"),
        )
        .expect("parse minimum MIP reference JSON");
        self.check(
            "IP/MIP minimum statuses optimal",
            minimum_internal.status == IPMIPStatus::Optimal
                && minimum_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                minimum_internal.status.as_str(),
                minimum_reference.result.status,
                minimum_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP minimum objective",
            minimum_internal.z,
            minimum_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = minimum_reference.result.x.as_deref() {
            self.check(
                "IP/MIP minimum external x length",
                x.len() == linearized_minimum.c.len(),
                "",
            );
            if x.len() >= minimum_original_vars {
                self.max_abs_close(
                    "IP/MIP minimum original x",
                    &minimum_internal.x[..minimum_original_vars],
                    &x[..minimum_original_vars],
                    1e-8,
                );
            }
        }

        let logical_problem = build_logical_gate_ip();
        let (linearized_logical, _, logical_original_vars) =
            linearize_source_ipmip_problem(&logical_problem);
        let logical_internal = solve_source_ipmip_with_des(
            logical_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let logical_problem_path = out_dir.join("logical-gate-problem.json");
        let logical_reference_path = out_dir.join("logical-gate-reference.json");
        let base = &logical_problem.base;
        let logical_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &logical_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "logical": logical_problem.logical.iter().map(|constraint| serde_json::json!({
                "kind": constraint.kind.as_str(),
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &logical_problem_path,
            serde_json::to_string_pretty(&logical_json).expect("serialize logical MIP problem"),
        )
        .expect("write logical MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&logical_problem_path)
            .arg("--out")
            .arg(&logical_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run logical MIP reference");
        if !output.status.success() {
            panic!(
                "logical ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let logical_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&logical_reference_path).expect("read logical MIP reference JSON"),
        )
        .expect("parse logical MIP reference JSON");
        self.check(
            "IP/MIP logical statuses optimal",
            logical_internal.status == IPMIPStatus::Optimal
                && logical_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                logical_internal.status.as_str(),
                logical_reference.result.status,
                logical_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP logical objective",
            logical_internal.z,
            logical_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = logical_reference.result.x.as_deref() {
            self.check(
                "IP/MIP logical external x length",
                x.len() == linearized_logical.c.len(),
                "",
            );
            if x.len() >= logical_original_vars {
                self.max_abs_close(
                    "IP/MIP logical original x",
                    &logical_internal.x[..logical_original_vars],
                    &x[..logical_original_vars],
                    1e-8,
                );
            }
        }

        let l1_problem = build_l1_norm_deviation_ip();
        let (linearized_l1, _, l1_original_vars) = linearize_source_ipmip_problem(&l1_problem);
        let l1_internal = solve_source_ipmip_with_des(
            l1_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let l1_problem_path = out_dir.join("l1-norm-deviation-problem.json");
        let l1_reference_path = out_dir.join("l1-norm-deviation-reference.json");
        let base = &l1_problem.base;
        let l1_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &l1_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "l1_norms": l1_problem.l1_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &l1_problem_path,
            serde_json::to_string_pretty(&l1_json).expect("serialize L1 norm MIP problem"),
        )
        .expect("write L1 norm MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&l1_problem_path)
            .arg("--out")
            .arg(&l1_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run L1 norm MIP reference");
        if !output.status.success() {
            panic!(
                "L1 norm ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let l1_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&l1_reference_path).expect("read L1 norm MIP reference JSON"),
        )
        .expect("parse L1 norm MIP reference JSON");
        self.check(
            "IP/MIP L1 norm statuses optimal",
            l1_internal.status == IPMIPStatus::Optimal && l1_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                l1_internal.status.as_str(),
                l1_reference.result.status,
                l1_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP L1 norm objective",
            l1_internal.z,
            l1_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = l1_reference.result.x.as_deref() {
            self.check(
                "IP/MIP L1 norm external x length",
                x.len() == linearized_l1.c.len(),
                "",
            );
            if x.len() >= l1_original_vars {
                self.max_abs_close(
                    "IP/MIP L1 norm original x",
                    &l1_internal.x[..l1_original_vars],
                    &x[..l1_original_vars],
                    1e-8,
                );
            }
        }

        let linf_problem = build_linf_norm_deviation_ip();
        let (linearized_linf, _, linf_original_vars) =
            linearize_source_ipmip_problem(&linf_problem);
        let linf_internal = solve_source_ipmip_with_des(
            linf_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let linf_problem_path = out_dir.join("linf-norm-deviation-problem.json");
        let linf_reference_path = out_dir.join("linf-norm-deviation-reference.json");
        let base = &linf_problem.base;
        let linf_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &linf_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "linf_norms": linf_problem.linf_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &linf_problem_path,
            serde_json::to_string_pretty(&linf_json).expect("serialize Linf norm MIP problem"),
        )
        .expect("write Linf norm MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&linf_problem_path)
            .arg("--out")
            .arg(&linf_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run Linf norm MIP reference");
        if !output.status.success() {
            panic!(
                "Linf norm ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let linf_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&linf_reference_path).expect("read Linf norm MIP reference JSON"),
        )
        .expect("parse Linf norm MIP reference JSON");
        self.check(
            "IP/MIP Linf norm statuses optimal",
            linf_internal.status == IPMIPStatus::Optimal
                && linf_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                linf_internal.status.as_str(),
                linf_reference.result.status,
                linf_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP Linf norm objective",
            linf_internal.z,
            linf_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = linf_reference.result.x.as_deref() {
            self.check(
                "IP/MIP Linf norm external x length",
                x.len() == linearized_linf.c.len(),
                "",
            );
            if x.len() >= linf_original_vars {
                self.max_abs_close(
                    "IP/MIP Linf norm original x",
                    &linf_internal.x[..linf_original_vars],
                    &x[..linf_original_vars],
                    1e-8,
                );
            }
        }

        for (product_name, product_problem) in vec![
            ("activation", build_product_activation_ip()),
            ("binary-gate", build_binary_product_gate_ip()),
        ] {
            let (linearized_product, _, product_original_vars) =
                linearize_source_ipmip_problem(&product_problem);
            let product_internal = solve_source_ipmip_with_des(
                product_problem.clone(),
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
            );
            let product_problem_path = out_dir.join(format!("product-{product_name}-problem.json"));
            let product_reference_path =
                out_dir.join(format!("product-{product_name}-reference.json"));
            let base = &product_problem.base;
            let product_json = serde_json::json!({
                "sense": base.sense.as_str(),
                "c": &base.c,
                "a": &base.a,
                "b": &base.b,
                "integer_vars": &base.integer_vars,
                "lb": &product_problem.lb,
                "ub": &base.ub,
                "var_names": &base.var_names,
                "con_names": &base.con_names,
                "products": product_problem.products.iter().map(|constraint| serde_json::json!({
                    "target_var": constraint.target_var,
                    "x_var": constraint.x_var,
                    "y_var": constraint.y_var,
                    "name": &constraint.name,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(
                &product_problem_path,
                serde_json::to_string_pretty(&product_json).expect("serialize product MIP problem"),
            )
            .expect("write product MIP problem");
            let output = Command::new(&python)
                .arg(&script)
                .arg("--problem")
                .arg(&product_problem_path)
                .arg("--out")
                .arg(&product_reference_path)
                .arg("--solver")
                .arg("auto")
                .output()
                .expect("run product MIP reference");
            if !output.status.success() {
                panic!(
                    "product {product_name} ip_mip_reference failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let product_reference: MipReference = serde_json::from_slice(
                &std::fs::read(&product_reference_path).expect("read product MIP reference JSON"),
            )
            .expect("parse product MIP reference JSON");
            self.check(
                format!("IP/MIP product {product_name} statuses optimal"),
                product_internal.status == IPMIPStatus::Optimal
                    && product_reference.result.status == "optimal",
                format!(
                    "internal={} external={} solver={}",
                    product_internal.status.as_str(),
                    product_reference.result.status,
                    product_reference.result.solver
                ),
            );
            self.close(
                &format!("IP/MIP product {product_name} objective"),
                product_internal.z,
                product_reference.result.objective.unwrap_or(f64::NAN),
                1e-9,
            );
            if let Some(x) = product_reference.result.x.as_deref() {
                self.check(
                    format!("IP/MIP product {product_name} external x length"),
                    x.len() == linearized_product.c.len(),
                    "",
                );
                if x.len() >= product_original_vars {
                    self.max_abs_close(
                        &format!("IP/MIP product {product_name} original x"),
                        &product_internal.x[..product_original_vars],
                        &x[..product_original_vars],
                        1e-8,
                    );
                }
            }
        }

        let quadratic_problem = build_quadratic_objective_mix_ip();
        let (linearized_quadratic, _, quadratic_original_vars) =
            linearize_quadratic_objective_problem(&quadratic_problem);
        let quadratic_internal = solve_quadratic_objective_ipmip_with_des(
            quadratic_problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        let quadratic_problem_path = out_dir.join("quadratic-objective-mix-problem.json");
        let quadratic_reference_path = out_dir.join("quadratic-objective-mix-reference.json");
        let base = &quadratic_problem.base;
        let quadratic_json = serde_json::json!({
            "sense": base.sense.as_str(),
            "c": &base.c,
            "a": &base.a,
            "b": &base.b,
            "integer_vars": &base.integer_vars,
            "lb": &quadratic_problem.lb,
            "ub": &base.ub,
            "var_names": &base.var_names,
            "con_names": &base.con_names,
            "quadratic_objective": quadratic_problem.quadratic_objective.iter().map(|term| serde_json::json!({
                "x_var": term.x_var,
                "y_var": term.y_var,
                "coeff": term.coeff,
                "name": &term.name,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &quadratic_problem_path,
            serde_json::to_string_pretty(&quadratic_json)
                .expect("serialize quadratic-objective MIP problem"),
        )
        .expect("write quadratic-objective MIP problem");
        let output = Command::new(&python)
            .arg(&script)
            .arg("--problem")
            .arg(&quadratic_problem_path)
            .arg("--out")
            .arg(&quadratic_reference_path)
            .arg("--solver")
            .arg("auto")
            .output()
            .expect("run quadratic-objective MIP reference");
        if !output.status.success() {
            panic!(
                "quadratic-objective ip_mip_reference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let quadratic_reference: MipReference = serde_json::from_slice(
            &std::fs::read(&quadratic_reference_path)
                .expect("read quadratic-objective MIP reference JSON"),
        )
        .expect("parse quadratic-objective MIP reference JSON");
        self.check(
            "IP/MIP quadratic objective statuses optimal",
            quadratic_internal.status == IPMIPStatus::Optimal
                && quadratic_reference.result.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                quadratic_internal.status.as_str(),
                quadratic_reference.result.status,
                quadratic_reference.result.solver
            ),
        );
        self.close(
            "IP/MIP quadratic objective",
            quadratic_internal.z,
            quadratic_reference.result.objective.unwrap_or(f64::NAN),
            1e-9,
        );
        if let Some(x) = quadratic_reference.result.x.as_deref() {
            self.check(
                "IP/MIP quadratic objective external x length",
                x.len() == linearized_quadratic.c.len(),
                "",
            );
            if x.len() >= quadratic_original_vars {
                self.max_abs_close(
                    "IP/MIP quadratic objective original x",
                    &quadratic_internal.x[..quadratic_original_vars],
                    &x[..quadratic_original_vars],
                    1e-8,
                );
            }
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
            "abs": source_problem.abs.iter().map(|constraint| serde_json::json!({
                "arg_var": constraint.arg_var,
                "target_var": constraint.target_var,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "maximums": source_problem.maximums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "minimums": source_problem.minimums.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "constant": constraint.constant,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "logical": source_problem.logical.iter().map(|constraint| serde_json::json!({
                "kind": constraint.kind.as_str(),
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "l1_norms": source_problem.l1_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "linf_norms": source_problem.linf_norms.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "arg_vars": &constraint.arg_vars,
                "name": &constraint.name,
            })).collect::<Vec<_>>(),
            "products": source_problem.products.iter().map(|constraint| serde_json::json!({
                "target_var": constraint.target_var,
                "x_var": constraint.x_var,
                "y_var": constraint.y_var,
                "name": &constraint.name,
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

    fn sample_miqp(&self) -> MixedIntegerQuadraticProgram {
        MixedIntegerQuadraticProgram {
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
        self.max_abs_close("QP x", &internal.x, &reference.x, 1e-7);
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

        let miqp = self.sample_miqp();
        let miqp_internal = solve_miqp_enumeration(&miqp, MIQPOptions::default());
        let miqp_json = serde_json::json!({
            "Q": &miqp.qp.q,
            "c": &miqp.qp.c,
            "A_ub": &miqp.qp.a_ub,
            "b_ub": &miqp.qp.b_ub,
            "A_eq": &miqp.qp.a_eq,
            "b_eq": &miqp.qp.b_eq,
            "lb": &miqp.qp.lb,
            "ub": &miqp.qp.ub,
            "integer_vars": &miqp.integer_vars,
        })
        .to_string();
        let value = self.run_python_json("qp_reference.py", &["--solver", "auto"], &miqp_json);
        let miqp_reference: QPReference =
            serde_json::from_value(value).expect("parse MIQP reference");
        self.check(
            "MIQP statuses optimal",
            miqp_internal.status == QPStatus::Optimal && miqp_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                miqp_internal.status.as_str(),
                miqp_reference.status,
                miqp_reference.solver
            ),
        );
        self.close(
            "MIQP objective",
            miqp_internal.objective,
            miqp_reference.objective.unwrap_or(f64::NAN),
            1e-8,
        );
        self.max_abs_close("MIQP x", &miqp_internal.x, &miqp_reference.x, 1e-7);

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
                CpVariable {
                    name: "route_0_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_1_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_0_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_2_0".to_string(),
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
                        presence: None,
                        name: Some("task_a".to_string()),
                    },
                    CpInterval {
                        start: 5,
                        duration: 2,
                        presence: None,
                        name: Some("task_b".to_string()),
                    },
                ]),
                CpConstraint::Cumulative {
                    intervals: vec![
                        CpDemandInterval {
                            start: 6,
                            duration: 3,
                            demand: 2,
                            presence: None,
                            name: Some("machine_a".to_string()),
                        },
                        CpDemandInterval {
                            start: 7,
                            duration: 2,
                            demand: 2,
                            presence: None,
                            name: Some("machine_b".to_string()),
                        },
                        CpDemandInterval {
                            start: 8,
                            duration: 2,
                            demand: 1,
                            presence: None,
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
                        presence: None,
                        name: Some("pack_a".to_string()),
                    },
                    CpRectangle {
                        x_start: 37,
                        y_start: 38,
                        width: 2,
                        height: 2,
                        presence: None,
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
                CpConstraint::MultipleCircuit(vec![
                    CpCircuitArc {
                        tail: 0,
                        head: 1,
                        literal: BoolLiteral {
                            var: 58,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 1,
                        head: 0,
                        literal: BoolLiteral {
                            var: 59,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 0,
                        head: 2,
                        literal: BoolLiteral {
                            var: 60,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 2,
                        head: 0,
                        literal: BoolLiteral {
                            var: 61,
                            positive: true,
                        },
                    },
                ]),
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
                    LinearTerm { var: 58, coeff: 1 },
                    LinearTerm { var: 59, coeff: 1 },
                    LinearTerm { var: 60, coeff: 1 },
                    LinearTerm { var: 61, coeff: 1 },
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
                CpConstraint::EnforcedLinearDomain {
                    enforcement,
                    terms,
                    intervals,
                } => serde_json::json!({
                    "kind": "enforced_linear_domain",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
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
                CpConstraint::EnforcedBoolOr {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_bool_or",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedBoolAnd {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_bool_and",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedBoolXor {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_bool_xor",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedAtMostOne {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_at_most_one",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedAtLeastOne {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_at_least_one",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedExactlyOne {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_exactly_one",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
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
                CpConstraint::AtLeastOne(lits) => serde_json::json!({
                    "kind": "at_least_one",
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
                CpConstraint::MultipleCircuit(arcs) => serde_json::json!({
                    "kind": "multiple_circuit",
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
                CpConstraint::EnforcedAllowedAssignments {
                    enforcement,
                    vars,
                    tuples,
                } => serde_json::json!({
                    "kind": "enforced_allowed_assignments",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement,
                    vars,
                    tuples,
                } => serde_json::json!({
                    "kind": "enforced_forbidden_assignments",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
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
                CpConstraint::VariableElement(element) => serde_json::json!({
                    "kind": "variable_element",
                    "index": element.index,
                    "vars": &element.vars,
                    "target": element.target,
                }),
                CpConstraint::Alternative(alternative) => serde_json::json!({
                    "kind": "alternative",
                    "start": alternative.start,
                    "duration": alternative.duration,
                    "end": alternative.end,
                    "presence": alternative.presence.as_ref().map(|lit| serde_json::json!({
                        "var": lit.var,
                        "positive": lit.positive,
                    })),
                    "alternatives": alternative.alternatives.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                    "name": alternative.name,
                }),
                CpConstraint::NoOverlap(intervals) => serde_json::json!({
                    "kind": "no_overlap",
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::NoOverlapVariable(intervals) => serde_json::json!({
                    "kind": "no_overlap_variable",
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
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
                        "presence": rectangle.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                        "name": rectangle.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::NoOverlap2DVariable(rectangles) => serde_json::json!({
                    "kind": "no_overlap_2d_variable",
                    "rectangles": rectangles.iter().map(|rectangle| serde_json::json!({
                        "x_start": rectangle.x_start,
                        "x_size": rectangle.x_size,
                        "x_end": rectangle.x_end,
                        "y_start": rectangle.y_start,
                        "y_size": rectangle.y_size,
                        "y_end": rectangle.y_end,
                        "presence": rectangle.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
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
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::CumulativeVariable {
                    intervals,
                    capacity,
                } => serde_json::json!({
                    "kind": "cumulative_variable",
                    "capacity": capacity,
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "demand": interval.demand,
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
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

        let variable_element_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "choice".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "expensive".to_string(),
                    domain: vec![4],
                },
                CpVariable {
                    name: "cheap".to_string(),
                    domain: vec![1, 3],
                },
                CpVariable {
                    name: "selected".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::VariableElement(
                crate::des::general::cp_sat::CpVariableElement {
                    index: 0,
                    vars: vec![1, 2],
                    target: 3,
                },
            )],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 3, coeff: 1 }],
            }),
        };
        let variable_element_internal =
            solve_cp_model(&variable_element_model, CpSolveOptions::default());
        let variable_element_json = serde_json::json!({
            "variables": [
                {"name": "choice", "domain": [0, 1]},
                {"name": "expensive", "domain": [4]},
                {"name": "cheap", "domain": [1, 3]},
                {"name": "selected", "domain": [1, 2, 3, 4]},
            ],
            "constraints": [
                {
                    "kind": "variable_element",
                    "index": 0,
                    "vars": [1, 2],
                    "target": 3,
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [{"var": 3, "coeff": 1}],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &variable_element_json,
        );
        let variable_element_reference: CpReference =
            serde_json::from_value(value).expect("parse variable-element CP reference");
        self.check(
            "CP-SAT variable element status internal/reference",
            variable_element_internal.status == CpStatus::Optimal
                && variable_element_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                variable_element_internal.status.as_str(),
                variable_element_reference.status,
                variable_element_reference.solver
            ),
        );
        self.check(
            "CP-SAT variable element objective",
            variable_element_internal.objective == variable_element_reference.objective
                && variable_element_internal.objective == Some(1),
            format!(
                "internal={:?} external={:?}",
                variable_element_internal.objective, variable_element_reference.objective
            ),
        );
        self.check(
            "CP-SAT variable element assignment",
            variable_element_internal.assignment == variable_element_reference.assignment
                && variable_element_internal.assignment == vec![1, 4, 1, 1],
            format!(
                "internal={:?} external={:?}",
                variable_element_internal.assignment, variable_element_reference.assignment
            ),
        );

        let enumeration_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Le,
                rhs: 1,
            }],
            objective: None,
        };
        let enumeration_internal = enumerate_cp_solutions(
            &enumeration_model,
            CpEnumerateOptions {
                max_nodes: 100,
                max_solutions: 4,
            },
        );
        let enumeration_json = serde_json::json!({
            "variables": [
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 1},
                    ],
                    "sense": "le",
                    "rhs": 1,
                },
            ],
            "objective": serde_json::Value::Null,
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto", "--enumerate-solutions", "4"],
            &enumeration_json,
        );
        let enumeration_reference: CpPoolReference =
            serde_json::from_value(value).expect("parse CP solution enumeration reference");
        let internal_assignments: Vec<Vec<i64>> = enumeration_internal
            .solutions
            .iter()
            .map(|solution| solution.assignment.clone())
            .collect();
        let reference_assignments: Vec<Vec<i64>> = enumeration_reference
            .solutions
            .iter()
            .map(|solution| solution.assignment.clone())
            .collect();
        let expected_assignments = vec![vec![0, 0], vec![0, 1], vec![1, 0]];
        self.check(
            "CP-SAT solution enumeration status",
            enumeration_internal.status == CpStatus::Feasible
                && enumeration_reference.status == "feasible"
                && enumeration_internal.exhausted
                && enumeration_reference.exhausted,
            format!(
                "internal={} exhausted={} external={} exhausted={} solver={}",
                enumeration_internal.status.as_str(),
                enumeration_internal.exhausted,
                enumeration_reference.status,
                enumeration_reference.exhausted,
                enumeration_reference.solver
            ),
        );
        self.check(
            "CP-SAT solution enumeration count",
            enumeration_internal.solutions.len() == 3 && enumeration_reference.solutions.len() == 3,
            format!(
                "internal={} external={}",
                enumeration_internal.solutions.len(),
                enumeration_reference.solutions.len()
            ),
        );
        self.check(
            "CP-SAT solution enumeration assignments",
            internal_assignments == expected_assignments
                && reference_assignments == internal_assignments,
            format!(
                "internal={:?} external={:?}",
                internal_assignments, reference_assignments
            ),
        );

        let assumption_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "irrelevant".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::BoolOr(vec![
                BoolLiteral {
                    var: 0,
                    positive: false,
                },
                BoolLiteral {
                    var: 1,
                    positive: false,
                },
            ])],
            objective: None,
        };
        let assumptions = vec![
            BoolLiteral {
                var: 0,
                positive: true,
            },
            BoolLiteral {
                var: 1,
                positive: true,
            },
            BoolLiteral {
                var: 2,
                positive: true,
            },
        ];
        let assumption_internal = find_cp_assumption_unsat_core(
            &assumption_model,
            &assumptions,
            CpAssumptionCoreOptions::default(),
        );
        let assumption_json = serde_json::json!({
            "variables": [
                {"name": "a", "domain": [0, 1]},
                {"name": "b", "domain": [0, 1]},
                {"name": "irrelevant", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [
                        {"var": 0, "positive": false},
                        {"var": 1, "positive": false},
                    ],
                },
            ],
            "objective": serde_json::Value::Null,
            "assumptions": assumptions.iter().map(|lit| serde_json::json!({
                "var": lit.var,
                "positive": lit.positive,
            })).collect::<Vec<_>>(),
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto", "--assumption-core"],
            &assumption_json,
        );
        let assumption_reference: CpAssumptionCoreReference =
            serde_json::from_value(value).expect("parse CP assumption-core reference");
        let internal_core: Vec<CpLiteralReference> = assumption_internal
            .assumptions
            .iter()
            .map(|lit| CpLiteralReference {
                var: lit.var,
                positive: lit.positive,
            })
            .collect();
        let expected_core = vec![
            CpLiteralReference {
                var: 0,
                positive: true,
            },
            CpLiteralReference {
                var: 1,
                positive: true,
            },
        ];
        self.check(
            "CP-SAT assumption core status",
            assumption_internal.status == CpStatus::Infeasible
                && assumption_internal.minimal
                && assumption_reference.status == "infeasible"
                && assumption_reference.minimal,
            format!(
                "internal={} minimal={} external={} minimal={} solver={}",
                assumption_internal.status.as_str(),
                assumption_internal.minimal,
                assumption_reference.status,
                assumption_reference.minimal,
                assumption_reference.solver
            ),
        );
        self.check(
            "CP-SAT assumption core literals",
            internal_core == expected_core && assumption_reference.assumptions == internal_core,
            format!(
                "internal={:?} external={:?}",
                internal_core, assumption_reference.assumptions
            ),
        );

        let optional_interval_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "required_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "optional_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "use_optional".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::NoOverlap(vec![
                    CpInterval {
                        start: 0,
                        duration: 3,
                        presence: None,
                        name: Some("required".to_string()),
                    },
                    CpInterval {
                        start: 1,
                        duration: 2,
                        presence: Some(BoolLiteral {
                            var: 2,
                            positive: true,
                        }),
                        name: Some("optional".to_string()),
                    },
                ]),
                CpConstraint::Cumulative {
                    intervals: vec![
                        CpDemandInterval {
                            start: 0,
                            duration: 3,
                            demand: 2,
                            presence: None,
                            name: Some("required_resource".to_string()),
                        },
                        CpDemandInterval {
                            start: 1,
                            duration: 2,
                            demand: 1,
                            presence: Some(BoolLiteral {
                                var: 2,
                                positive: true,
                            }),
                            name: Some("optional_resource".to_string()),
                        },
                    ],
                    capacity: 2,
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 2, coeff: 1 }],
            }),
        };
        let optional_interval_internal =
            solve_cp_model(&optional_interval_model, CpSolveOptions::default());
        let optional_interval_json = serde_json::json!({
            "variables": [
                {"name": "required_start", "domain": [0]},
                {"name": "optional_start", "domain": [0]},
                {"name": "use_optional", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "no_overlap",
                    "intervals": [
                        {"start": 0, "duration": 3, "name": "required"},
                        {
                            "start": 1,
                            "duration": 2,
                            "presence": {"var": 2, "positive": true},
                            "name": "optional",
                        },
                    ],
                },
                {
                    "kind": "cumulative",
                    "capacity": 2,
                    "intervals": [
                        {"start": 0, "duration": 3, "demand": 2, "name": "required_resource"},
                        {
                            "start": 1,
                            "duration": 2,
                            "demand": 1,
                            "presence": {"var": 2, "positive": true},
                            "name": "optional_resource",
                        },
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [{"var": 2, "coeff": 1}],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &optional_interval_json,
        );
        let optional_interval_reference: CpReference =
            serde_json::from_value(value).expect("parse optional interval CP reference");
        self.check(
            "CP-SAT optional interval status internal/reference",
            optional_interval_internal.status == CpStatus::Optimal
                && optional_interval_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                optional_interval_internal.status.as_str(),
                optional_interval_reference.status,
                optional_interval_reference.solver
            ),
        );
        self.check(
            "CP-SAT optional interval objective",
            optional_interval_internal.objective == optional_interval_reference.objective,
            format!(
                "internal={:?} external={:?}",
                optional_interval_internal.objective, optional_interval_reference.objective
            ),
        );
        self.check(
            "CP-SAT optional interval assignment",
            optional_interval_internal.assignment == optional_interval_reference.assignment
                && optional_interval_internal.assignment == vec![0, 0, 0],
            format!(
                "internal={:?} external={:?}",
                optional_interval_internal.assignment, optional_interval_reference.assignment
            ),
        );

        let variable_interval_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "task_a_duration".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_a_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "task_b_duration".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "task_b_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::NoOverlapVariable(vec![
                CpVariableInterval {
                    start: 0,
                    duration: 1,
                    end: 2,
                    presence: None,
                    name: Some("task_a".to_string()),
                },
                CpVariableInterval {
                    start: 3,
                    duration: 4,
                    end: 5,
                    presence: None,
                    name: Some("task_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                ],
            }),
        };
        let variable_interval_internal =
            solve_cp_model(&variable_interval_model, CpSolveOptions::default());
        let variable_interval_json = serde_json::json!({
            "variables": [
                {"name": "task_a_start", "domain": [0]},
                {"name": "task_a_duration", "domain": [2]},
                {"name": "task_a_end", "domain": [2]},
                {"name": "task_b_start", "domain": [0, 1, 2]},
                {"name": "task_b_duration", "domain": [1, 2]},
                {"name": "task_b_end", "domain": [1, 2, 3, 4]},
            ],
            "constraints": [
                {
                    "kind": "no_overlap_variable",
                    "intervals": [
                        {
                            "start": 0,
                            "duration": 1,
                            "end": 2,
                            "name": "task_a",
                        },
                        {
                            "start": 3,
                            "duration": 4,
                            "end": 5,
                            "name": "task_b",
                        },
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 3, "coeff": 1},
                    {"var": 4, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &variable_interval_json,
        );
        let variable_interval_reference: CpReference =
            serde_json::from_value(value).expect("parse variable interval CP reference");
        self.check(
            "CP-SAT variable interval status internal/reference",
            variable_interval_internal.status == CpStatus::Optimal
                && variable_interval_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                variable_interval_internal.status.as_str(),
                variable_interval_reference.status,
                variable_interval_reference.solver
            ),
        );
        self.check(
            "CP-SAT variable interval objective",
            variable_interval_internal.objective == variable_interval_reference.objective
                && variable_interval_internal.objective == Some(3),
            format!(
                "internal={:?} external={:?}",
                variable_interval_internal.objective, variable_interval_reference.objective
            ),
        );
        self.check(
            "CP-SAT variable interval assignment",
            variable_interval_internal.assignment == variable_interval_reference.assignment
                && variable_interval_internal.assignment == vec![0, 2, 2, 2, 1, 3],
            format!(
                "internal={:?} external={:?}",
                variable_interval_internal.assignment, variable_interval_reference.assignment
            ),
        );

        let alternative_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "job_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "job_duration".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "job_end".to_string(),
                    domain: vec![2, 3, 4],
                },
                CpVariable {
                    name: "fast_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "fast_duration".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "fast_end".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "use_fast".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "slow_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "slow_duration".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "slow_end".to_string(),
                    domain: vec![3, 4],
                },
                CpVariable {
                    name: "use_slow".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Alternative(CpAlternative {
                start: 0,
                duration: 1,
                end: 2,
                presence: None,
                alternatives: vec![
                    CpVariableInterval {
                        start: 3,
                        duration: 4,
                        end: 5,
                        presence: Some(BoolLiteral {
                            var: 6,
                            positive: true,
                        }),
                        name: Some("fast".to_string()),
                    },
                    CpVariableInterval {
                        start: 7,
                        duration: 8,
                        end: 9,
                        presence: Some(BoolLiteral {
                            var: 10,
                            positive: true,
                        }),
                        name: Some("slow".to_string()),
                    },
                ],
                name: Some("job_modes".to_string()),
            })],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 1, coeff: 10 },
                    LinearTerm { var: 0, coeff: 1 },
                ],
            }),
        };
        let alternative_internal = solve_cp_model(&alternative_model, CpSolveOptions::default());
        let alternative_json = serde_json::json!({
            "variables": [
                {"name": "job_start", "domain": [0, 1]},
                {"name": "job_duration", "domain": [2, 3]},
                {"name": "job_end", "domain": [2, 3, 4]},
                {"name": "fast_start", "domain": [0, 1]},
                {"name": "fast_duration", "domain": [2]},
                {"name": "fast_end", "domain": [2, 3]},
                {"name": "use_fast", "domain": [0, 1]},
                {"name": "slow_start", "domain": [0, 1]},
                {"name": "slow_duration", "domain": [3]},
                {"name": "slow_end", "domain": [3, 4]},
                {"name": "use_slow", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "alternative",
                    "start": 0,
                    "duration": 1,
                    "end": 2,
                    "alternatives": [
                        {
                            "start": 3,
                            "duration": 4,
                            "end": 5,
                            "presence": {"var": 6, "positive": true},
                            "name": "fast",
                        },
                        {
                            "start": 7,
                            "duration": 8,
                            "end": 9,
                            "presence": {"var": 10, "positive": true},
                            "name": "slow",
                        },
                    ],
                    "name": "job_modes",
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 1, "coeff": 10},
                    {"var": 0, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &alternative_json,
        );
        let alternative_reference: CpReference =
            serde_json::from_value(value).expect("parse alternative CP reference");
        self.check(
            "CP-SAT alternative interval status internal/reference",
            alternative_internal.status == CpStatus::Optimal
                && alternative_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                alternative_internal.status.as_str(),
                alternative_reference.status,
                alternative_reference.solver
            ),
        );
        self.check(
            "CP-SAT alternative interval objective",
            alternative_internal.objective == alternative_reference.objective
                && alternative_internal.objective == Some(20),
            format!(
                "internal={:?} external={:?}",
                alternative_internal.objective, alternative_reference.objective
            ),
        );
        self.check(
            "CP-SAT alternative interval assignment",
            alternative_internal.assignment == alternative_reference.assignment
                && alternative_internal.assignment == vec![0, 2, 2, 0, 2, 2, 1, 0, 3, 3, 0],
            format!(
                "internal={:?} external={:?}",
                alternative_internal.assignment, alternative_reference.assignment
            ),
        );

        let variable_no_overlap_2d_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "box_a_x_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_a_x_size".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_x_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_y_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_a_y_size".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_y_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_b_x_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "box_b_x_size".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "box_b_x_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
                CpVariable {
                    name: "box_b_y_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "box_b_y_size".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "box_b_y_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::NoOverlap2DVariable(vec![
                CpVariableRectangle {
                    x_start: 0,
                    x_size: 1,
                    x_end: 2,
                    y_start: 3,
                    y_size: 4,
                    y_end: 5,
                    presence: None,
                    name: Some("box_a".to_string()),
                },
                CpVariableRectangle {
                    x_start: 6,
                    x_size: 7,
                    x_end: 8,
                    y_start: 9,
                    y_size: 10,
                    y_end: 11,
                    presence: None,
                    name: Some("box_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 6, coeff: 10 },
                    LinearTerm { var: 9, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                    LinearTerm { var: 10, coeff: 1 },
                ],
            }),
        };
        let variable_no_overlap_2d_internal =
            solve_cp_model(&variable_no_overlap_2d_model, CpSolveOptions::default());
        let variable_no_overlap_2d_json = serde_json::json!({
            "variables": [
                {"name": "box_a_x_start", "domain": [0]},
                {"name": "box_a_x_size", "domain": [2]},
                {"name": "box_a_x_end", "domain": [2]},
                {"name": "box_a_y_start", "domain": [0]},
                {"name": "box_a_y_size", "domain": [2]},
                {"name": "box_a_y_end", "domain": [2]},
                {"name": "box_b_x_start", "domain": [0, 1, 2]},
                {"name": "box_b_x_size", "domain": [1, 2]},
                {"name": "box_b_x_end", "domain": [1, 2, 3, 4]},
                {"name": "box_b_y_start", "domain": [0, 1, 2]},
                {"name": "box_b_y_size", "domain": [1, 2]},
                {"name": "box_b_y_end", "domain": [1, 2, 3, 4]},
            ],
            "constraints": [
                {
                    "kind": "no_overlap_2d_variable",
                    "rectangles": [
                        {
                            "x_start": 0,
                            "x_size": 1,
                            "x_end": 2,
                            "y_start": 3,
                            "y_size": 4,
                            "y_end": 5,
                            "name": "box_a",
                        },
                        {
                            "x_start": 6,
                            "x_size": 7,
                            "x_end": 8,
                            "y_start": 9,
                            "y_size": 10,
                            "y_end": 11,
                            "name": "box_b",
                        },
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 6, "coeff": 10},
                    {"var": 9, "coeff": 1},
                    {"var": 7, "coeff": 1},
                    {"var": 10, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &variable_no_overlap_2d_json,
        );
        let variable_no_overlap_2d_reference: CpReference =
            serde_json::from_value(value).expect("parse variable no-overlap-2d CP reference");
        self.check(
            "CP-SAT variable no-overlap-2d status internal/reference",
            variable_no_overlap_2d_internal.status == CpStatus::Optimal
                && variable_no_overlap_2d_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                variable_no_overlap_2d_internal.status.as_str(),
                variable_no_overlap_2d_reference.status,
                variable_no_overlap_2d_reference.solver
            ),
        );
        self.check(
            "CP-SAT variable no-overlap-2d objective",
            variable_no_overlap_2d_internal.objective == variable_no_overlap_2d_reference.objective
                && variable_no_overlap_2d_internal.objective == Some(4),
            format!(
                "internal={:?} external={:?}",
                variable_no_overlap_2d_internal.objective,
                variable_no_overlap_2d_reference.objective
            ),
        );
        self.check(
            "CP-SAT variable no-overlap-2d assignment",
            variable_no_overlap_2d_internal.assignment
                == variable_no_overlap_2d_reference.assignment
                && variable_no_overlap_2d_internal.assignment
                    == vec![0, 2, 2, 0, 2, 2, 0, 1, 1, 2, 1, 3],
            format!(
                "internal={:?} external={:?}",
                variable_no_overlap_2d_internal.assignment,
                variable_no_overlap_2d_reference.assignment
            ),
        );

        let variable_cumulative_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "task_a_duration".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "task_a_end".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "task_a_demand".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "task_b_duration".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "task_b_end".to_string(),
                    domain: vec![1, 2, 3, 4, 5],
                },
                CpVariable {
                    name: "task_b_demand".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "capacity".to_string(),
                    domain: vec![3, 4],
                },
            ],
            constraints: vec![CpConstraint::CumulativeVariable {
                intervals: vec![
                    CpVariableDemandInterval {
                        start: 0,
                        duration: 1,
                        end: 2,
                        demand: 3,
                        presence: None,
                        name: Some("task_a".to_string()),
                    },
                    CpVariableDemandInterval {
                        start: 4,
                        duration: 5,
                        end: 6,
                        demand: 7,
                        presence: None,
                        name: Some("task_b".to_string()),
                    },
                ],
                capacity: 8,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 8, coeff: 100 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                ],
            }),
        };
        let variable_cumulative_internal =
            solve_cp_model(&variable_cumulative_model, CpSolveOptions::default());
        let variable_cumulative_json = serde_json::json!({
            "variables": [
                {"name": "task_a_start", "domain": [0]},
                {"name": "task_a_duration", "domain": [3]},
                {"name": "task_a_end", "domain": [3]},
                {"name": "task_a_demand", "domain": [2]},
                {"name": "task_b_start", "domain": [0, 1, 2, 3]},
                {"name": "task_b_duration", "domain": [1, 2]},
                {"name": "task_b_end", "domain": [1, 2, 3, 4, 5]},
                {"name": "task_b_demand", "domain": [1, 2]},
                {"name": "capacity", "domain": [3, 4]},
            ],
            "constraints": [
                {
                    "kind": "cumulative_variable",
                    "capacity": 8,
                    "intervals": [
                        {
                            "start": 0,
                            "duration": 1,
                            "end": 2,
                            "demand": 3,
                            "name": "task_a",
                        },
                        {
                            "start": 4,
                            "duration": 5,
                            "end": 6,
                            "demand": 7,
                            "name": "task_b",
                        },
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 8, "coeff": 100},
                    {"var": 4, "coeff": 1},
                    {"var": 5, "coeff": 1},
                    {"var": 7, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &variable_cumulative_json,
        );
        let variable_cumulative_reference: CpReference =
            serde_json::from_value(value).expect("parse variable cumulative CP reference");
        self.check(
            "CP-SAT variable cumulative status internal/reference",
            variable_cumulative_internal.status == CpStatus::Optimal
                && variable_cumulative_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                variable_cumulative_internal.status.as_str(),
                variable_cumulative_reference.status,
                variable_cumulative_reference.solver
            ),
        );
        self.check(
            "CP-SAT variable cumulative objective",
            variable_cumulative_internal.objective == variable_cumulative_reference.objective
                && variable_cumulative_internal.objective == Some(302),
            format!(
                "internal={:?} external={:?}",
                variable_cumulative_internal.objective, variable_cumulative_reference.objective
            ),
        );
        self.check(
            "CP-SAT variable cumulative assignment",
            variable_cumulative_internal.assignment == variable_cumulative_reference.assignment
                && variable_cumulative_internal.assignment == vec![0, 3, 3, 2, 0, 1, 1, 1, 3],
            format!(
                "internal={:?} external={:?}",
                variable_cumulative_internal.assignment, variable_cumulative_reference.assignment
            ),
        );

        let hinted_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "hint_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "hint_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Eq,
                rhs: 2,
            }],
            objective: None,
        };
        let hinted_internal = solve_cp_model(
            &hinted_model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: vec![
                    CpSolutionHint { var: 0, value: 1 },
                    CpSolutionHint { var: 1, value: 1 },
                ],
                decision_strategies: Vec::new(),
            },
        );
        let unhinted_internal = solve_cp_model(
            &hinted_model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: Vec::new(),
            },
        );
        let hinted_json = serde_json::json!({
            "variables": [
                {"name": "hint_x", "domain": [0, 1]},
                {"name": "hint_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 1},
                    ],
                    "sense": "eq",
                    "rhs": 2,
                },
            ],
            "objective": serde_json::Value::Null,
            "solution_hint": [
                {"var": 0, "value": 1},
                {"var": 1, "value": 1},
            ],
        })
        .to_string();
        let value =
            self.run_python_json("cp_sat_reference.py", &["--solver", "auto"], &hinted_json);
        let hinted_reference: CpReference =
            serde_json::from_value(value).expect("parse hinted CP reference");
        self.check(
            "CP-SAT solution hint status internal/reference",
            hinted_internal.status == CpStatus::Feasible && hinted_reference.status == "feasible",
            format!(
                "internal={} external={} solver={}",
                hinted_internal.status.as_str(),
                hinted_reference.status,
                hinted_reference.solver
            ),
        );
        self.check(
            "CP-SAT solution hint assignment",
            hinted_internal.assignment == hinted_reference.assignment
                && hinted_internal.assignment == vec![1, 1],
            format!(
                "internal={:?} external={:?}",
                hinted_internal.assignment, hinted_reference.assignment
            ),
        );
        self.check(
            "CP-SAT solution hint native node-cap contrast",
            hinted_internal.status == CpStatus::Feasible
                && unhinted_internal.status == CpStatus::Infeasible,
            format!(
                "hinted={} nodes={} unhinted={} nodes={}",
                hinted_internal.status.as_str(),
                hinted_internal.nodes,
                unhinted_internal.status.as_str(),
                unhinted_internal.nodes
            ),
        );

        let strategy_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "strategy_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "strategy_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Eq,
                rhs: 2,
            }],
            objective: None,
        };
        let strategic_internal = solve_cp_model(
            &strategy_model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: vec![CpDecisionStrategy {
                    vars: vec![1, 0],
                    variable_strategy: CpVariableSelectionStrategy::First,
                    domain_strategy: CpDomainValueStrategy::MaxValue,
                }],
            },
        );
        let unstrategic_internal = solve_cp_model(
            &strategy_model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: Vec::new(),
            },
        );
        let strategy_json = serde_json::json!({
            "variables": [
                {"name": "strategy_x", "domain": [0, 1]},
                {"name": "strategy_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 1},
                    ],
                    "sense": "eq",
                    "rhs": 2,
                },
            ],
            "objective": serde_json::Value::Null,
            "decision_strategies": [
                {
                    "vars": [1, 0],
                    "variable_strategy": "first",
                    "domain_strategy": "max_value",
                },
            ],
        })
        .to_string();
        let value =
            self.run_python_json("cp_sat_reference.py", &["--solver", "auto"], &strategy_json);
        let strategy_reference: CpReference =
            serde_json::from_value(value).expect("parse decision-strategy CP reference");
        self.check(
            "CP-SAT decision strategy status internal/reference",
            strategic_internal.status == CpStatus::Feasible
                && strategy_reference.status == "feasible",
            format!(
                "internal={} external={} solver={}",
                strategic_internal.status.as_str(),
                strategy_reference.status,
                strategy_reference.solver
            ),
        );
        self.check(
            "CP-SAT decision strategy assignment",
            strategic_internal.assignment == strategy_reference.assignment
                && strategic_internal.assignment == vec![1, 1],
            format!(
                "internal={:?} external={:?}",
                strategic_internal.assignment, strategy_reference.assignment
            ),
        );
        self.check(
            "CP-SAT decision strategy native node-cap contrast",
            strategic_internal.status == CpStatus::Feasible
                && unstrategic_internal.status == CpStatus::Infeasible,
            format!(
                "strategic={} nodes={} unstrategic={} nodes={}",
                strategic_internal.status.as_str(),
                strategic_internal.nodes,
                unstrategic_internal.status.as_str(),
                unstrategic_internal.nodes
            ),
        );

        for (label, domain_strategy, domain_strategy_json, expected_assignment) in [
            (
                "lower-half",
                CpDomainValueStrategy::LowerHalf,
                "lower_half",
                vec![0, 4],
            ),
            (
                "upper-half",
                CpDomainValueStrategy::UpperHalf,
                "upper_half",
                vec![4, 0],
            ),
            (
                "median-value",
                CpDomainValueStrategy::MedianValue,
                "median_value",
                vec![2, 2],
            ),
        ] {
            let domain_strategy_model = CpModel {
                variables: vec![
                    CpVariable {
                        name: "domain_strategy_x".to_string(),
                        domain: vec![0, 1, 2, 3, 4],
                    },
                    CpVariable {
                        name: "domain_strategy_y".to_string(),
                        domain: vec![0, 1, 2, 3, 4],
                    },
                ],
                constraints: vec![CpConstraint::Linear {
                    terms: vec![
                        LinearTerm { var: 0, coeff: 1 },
                        LinearTerm { var: 1, coeff: 1 },
                    ],
                    sense: LinearSense::Eq,
                    rhs: 4,
                }],
                objective: None,
            };
            let domain_strategy_internal = solve_cp_model(
                &domain_strategy_model,
                CpSolveOptions {
                    max_nodes: 100,
                    solution_hint: Vec::new(),
                    decision_strategies: vec![CpDecisionStrategy {
                        vars: vec![0, 1],
                        variable_strategy: CpVariableSelectionStrategy::First,
                        domain_strategy,
                    }],
                },
            );
            let domain_strategy_json = serde_json::json!({
                "variables": [
                    {"name": "domain_strategy_x", "domain": [0, 1, 2, 3, 4]},
                    {"name": "domain_strategy_y", "domain": [0, 1, 2, 3, 4]},
                ],
                "constraints": [
                    {
                        "kind": "linear",
                        "terms": [
                            {"var": 0, "coeff": 1},
                            {"var": 1, "coeff": 1},
                        ],
                        "sense": "eq",
                        "rhs": 4,
                    },
                ],
                "objective": serde_json::Value::Null,
                "decision_strategies": [
                    {
                        "vars": [0, 1],
                        "variable_strategy": "first",
                        "domain_strategy": domain_strategy_json,
                    },
                ],
            })
            .to_string();
            let value = self.run_python_json(
                "cp_sat_reference.py",
                &["--solver", "auto"],
                &domain_strategy_json,
            );
            let domain_strategy_reference: CpReference =
                serde_json::from_value(value).expect("parse domain-reduction CP reference");
            self.check(
                format!("CP-SAT decision domain strategy {label} status internal/reference"),
                domain_strategy_internal.status == CpStatus::Feasible
                    && domain_strategy_reference.status == "feasible",
                format!(
                    "internal={} external={} solver={}",
                    domain_strategy_internal.status.as_str(),
                    domain_strategy_reference.status,
                    domain_strategy_reference.solver
                ),
            );
            self.check(
                format!("CP-SAT decision domain strategy {label} assignment"),
                domain_strategy_internal.assignment == domain_strategy_reference.assignment
                    && domain_strategy_internal.assignment == expected_assignment,
                format!(
                    "internal={:?} external={:?}",
                    domain_strategy_internal.assignment, domain_strategy_reference.assignment
                ),
            );
        }

        let enforced_bool_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::EnforcedBoolOr {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 3,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedBoolAnd {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![BoolLiteral {
                        var: 3,
                        positive: true,
                    }],
                },
                CpConstraint::EnforcedBoolAnd {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 4,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 5,
                            positive: true,
                        },
                    ],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                ],
            }),
        };
        let enforced_bool_internal =
            solve_cp_model(&enforced_bool_model, CpSolveOptions::default());
        let enforced_bool_json = serde_json::json!({
            "variables": [
                {"name": "active_gate", "domain": [0, 1]},
                {"name": "inactive_gate", "domain": [0, 1]},
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
                {"name": "inactive_x", "domain": [0, 1]},
                {"name": "inactive_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [{"var": 0, "positive": true}],
                },
                {
                    "kind": "bool_or",
                    "literals": [{"var": 1, "positive": false}],
                },
                {
                    "kind": "enforced_bool_or",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [
                        {"var": 2, "positive": true},
                        {"var": 3, "positive": true},
                    ],
                },
                {
                    "kind": "enforced_bool_and",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [{"var": 3, "positive": true}],
                },
                {
                    "kind": "enforced_bool_and",
                    "enforcement": [{"var": 1, "positive": true}],
                    "literals": [
                        {"var": 4, "positive": true},
                        {"var": 5, "positive": true},
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 2, "coeff": 1},
                    {"var": 3, "coeff": 1},
                    {"var": 4, "coeff": 1},
                    {"var": 5, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &enforced_bool_json,
        );
        let enforced_bool_reference: CpReference =
            serde_json::from_value(value).expect("parse enforced-bool CP reference");
        self.check(
            "CP-SAT enforced bool status internal/reference",
            enforced_bool_internal.status == CpStatus::Optimal
                && enforced_bool_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                enforced_bool_internal.status.as_str(),
                enforced_bool_reference.status,
                enforced_bool_reference.solver
            ),
        );
        self.check(
            "CP-SAT enforced bool objective",
            enforced_bool_internal.objective == enforced_bool_reference.objective
                && enforced_bool_internal.objective == Some(1),
            format!(
                "internal={:?} external={:?}",
                enforced_bool_internal.objective, enforced_bool_reference.objective
            ),
        );
        self.check(
            "CP-SAT enforced bool assignment",
            enforced_bool_internal.assignment == enforced_bool_reference.assignment
                && enforced_bool_internal.assignment == vec![1, 0, 0, 1, 0, 0],
            format!(
                "internal={:?} external={:?}",
                enforced_bool_internal.assignment, enforced_bool_reference.assignment
            ),
        );

        let at_least_one_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "z".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "required".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::AtLeastOne(vec![BoolLiteral {
                    var: 5,
                    positive: true,
                }]),
                CpConstraint::EnforcedAtLeastOne {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 3,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 4,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedAtLeastOne {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 6,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 7,
                            positive: true,
                        },
                    ],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 3 },
                    LinearTerm { var: 3, coeff: 2 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 6, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                ],
            }),
        };
        let at_least_one_internal = solve_cp_model(&at_least_one_model, CpSolveOptions::default());
        let at_least_one_json = serde_json::json!({
            "variables": [
                {"name": "active_gate", "domain": [0, 1]},
                {"name": "inactive_gate", "domain": [0, 1]},
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
                {"name": "z", "domain": [0, 1]},
                {"name": "required", "domain": [0, 1]},
                {"name": "inactive_x", "domain": [0, 1]},
                {"name": "inactive_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [{"var": 0, "positive": true}],
                },
                {
                    "kind": "bool_or",
                    "literals": [{"var": 1, "positive": false}],
                },
                {
                    "kind": "at_least_one",
                    "literals": [{"var": 5, "positive": true}],
                },
                {
                    "kind": "enforced_at_least_one",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [
                        {"var": 2, "positive": true},
                        {"var": 3, "positive": true},
                        {"var": 4, "positive": true},
                    ],
                },
                {
                    "kind": "enforced_at_least_one",
                    "enforcement": [{"var": 1, "positive": true}],
                    "literals": [
                        {"var": 6, "positive": true},
                        {"var": 7, "positive": true},
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 2, "coeff": 3},
                    {"var": 3, "coeff": 2},
                    {"var": 4, "coeff": 1},
                    {"var": 5, "coeff": 1},
                    {"var": 6, "coeff": 1},
                    {"var": 7, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &at_least_one_json,
        );
        let at_least_one_reference: CpReference =
            serde_json::from_value(value).expect("parse at-least-one CP reference");
        self.check(
            "CP-SAT at-least-one status internal/reference",
            at_least_one_internal.status == CpStatus::Optimal
                && at_least_one_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                at_least_one_internal.status.as_str(),
                at_least_one_reference.status,
                at_least_one_reference.solver
            ),
        );
        self.check(
            "CP-SAT at-least-one objective",
            at_least_one_internal.objective == at_least_one_reference.objective
                && at_least_one_internal.objective == Some(2),
            format!(
                "internal={:?} external={:?}",
                at_least_one_internal.objective, at_least_one_reference.objective
            ),
        );
        self.check(
            "CP-SAT at-least-one assignment",
            at_least_one_internal.assignment == at_least_one_reference.assignment
                && at_least_one_internal.assignment == vec![1, 0, 0, 0, 1, 1, 0, 0],
            format!(
                "internal={:?} external={:?}",
                at_least_one_internal.assignment, at_least_one_reference.assignment
            ),
        );

        let enforced_linear_domain_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1, 2, 3, 4],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1, 2, 3, 4],
                },
                CpVariable {
                    name: "free".to_string(),
                    domain: vec![0, 1, 2],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::EnforcedLinearDomain {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    terms: vec![
                        LinearTerm { var: 2, coeff: 1 },
                        LinearTerm { var: 3, coeff: 1 },
                    ],
                    intervals: vec![
                        CpDomainInterval { lb: 3, ub: 3 },
                        CpDomainInterval { lb: 7, ub: 7 },
                    ],
                },
                CpConstraint::EnforcedLinearDomain {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 4, coeff: 1 }],
                    intervals: vec![CpDomainInterval { lb: 2, ub: 2 }],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 10 },
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                ],
            }),
        };
        let enforced_linear_domain_internal =
            solve_cp_model(&enforced_linear_domain_model, CpSolveOptions::default());
        let enforced_linear_domain_json = serde_json::json!({
            "variables": [
                {"name": "active_gate", "domain": [0, 1]},
                {"name": "inactive_gate", "domain": [0, 1]},
                {"name": "x", "domain": [0, 1, 2, 3, 4]},
                {"name": "y", "domain": [0, 1, 2, 3, 4]},
                {"name": "free", "domain": [0, 1, 2]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [{"var": 0, "positive": true}],
                },
                {
                    "kind": "bool_or",
                    "literals": [{"var": 1, "positive": false}],
                },
                {
                    "kind": "enforced_linear_domain",
                    "enforcement": [{"var": 0, "positive": true}],
                    "terms": [
                        {"var": 2, "coeff": 1},
                        {"var": 3, "coeff": 1},
                    ],
                    "intervals": [
                        {"lb": 3, "ub": 3},
                        {"lb": 7, "ub": 7},
                    ],
                },
                {
                    "kind": "enforced_linear_domain",
                    "enforcement": [{"var": 1, "positive": true}],
                    "terms": [{"var": 4, "coeff": 1}],
                    "intervals": [{"lb": 2, "ub": 2}],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 2, "coeff": 10},
                    {"var": 3, "coeff": 1},
                    {"var": 4, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &enforced_linear_domain_json,
        );
        let enforced_linear_domain_reference: CpReference =
            serde_json::from_value(value).expect("parse enforced-linear-domain CP reference");
        self.check(
            "CP-SAT enforced linear-domain status internal/reference",
            enforced_linear_domain_internal.status == CpStatus::Optimal
                && enforced_linear_domain_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                enforced_linear_domain_internal.status.as_str(),
                enforced_linear_domain_reference.status,
                enforced_linear_domain_reference.solver
            ),
        );
        self.check(
            "CP-SAT enforced linear-domain objective",
            enforced_linear_domain_internal.objective == enforced_linear_domain_reference.objective
                && enforced_linear_domain_internal.objective == Some(3),
            format!(
                "internal={:?} external={:?}",
                enforced_linear_domain_internal.objective,
                enforced_linear_domain_reference.objective
            ),
        );
        self.check(
            "CP-SAT enforced linear-domain assignment",
            enforced_linear_domain_internal.assignment
                == enforced_linear_domain_reference.assignment
                && enforced_linear_domain_internal.assignment == vec![1, 0, 0, 3, 0],
            format!(
                "internal={:?} external={:?}",
                enforced_linear_domain_internal.assignment,
                enforced_linear_domain_reference.assignment
            ),
        );

        let enforced_cardinality_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "z".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::EnforcedExactlyOne {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 3,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedBoolXor {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 3,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 4,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedAtMostOne {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 4,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedExactlyOne {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 5,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 6,
                            positive: true,
                        },
                    ],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 6, coeff: 1 },
                ],
            }),
        };
        let enforced_cardinality_internal =
            solve_cp_model(&enforced_cardinality_model, CpSolveOptions::default());
        let enforced_cardinality_json = serde_json::json!({
            "variables": [
                {"name": "active_gate", "domain": [0, 1]},
                {"name": "inactive_gate", "domain": [0, 1]},
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
                {"name": "z", "domain": [0, 1]},
                {"name": "inactive_x", "domain": [0, 1]},
                {"name": "inactive_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [{"var": 0, "positive": true}],
                },
                {
                    "kind": "bool_or",
                    "literals": [{"var": 1, "positive": false}],
                },
                {
                    "kind": "enforced_exactly_one",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [
                        {"var": 2, "positive": true},
                        {"var": 3, "positive": true},
                    ],
                },
                {
                    "kind": "enforced_bool_xor",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [
                        {"var": 3, "positive": true},
                        {"var": 4, "positive": true},
                    ],
                },
                {
                    "kind": "enforced_at_most_one",
                    "enforcement": [{"var": 0, "positive": true}],
                    "literals": [
                        {"var": 2, "positive": true},
                        {"var": 4, "positive": true},
                    ],
                },
                {
                    "kind": "enforced_exactly_one",
                    "enforcement": [{"var": 1, "positive": true}],
                    "literals": [
                        {"var": 5, "positive": true},
                        {"var": 6, "positive": true},
                    ],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 2, "coeff": 1},
                    {"var": 3, "coeff": 1},
                    {"var": 4, "coeff": 1},
                    {"var": 5, "coeff": 1},
                    {"var": 6, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &enforced_cardinality_json,
        );
        let enforced_cardinality_reference: CpReference =
            serde_json::from_value(value).expect("parse enforced-cardinality CP reference");
        self.check(
            "CP-SAT enforced cardinality status internal/reference",
            enforced_cardinality_internal.status == CpStatus::Optimal
                && enforced_cardinality_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                enforced_cardinality_internal.status.as_str(),
                enforced_cardinality_reference.status,
                enforced_cardinality_reference.solver
            ),
        );
        self.check(
            "CP-SAT enforced cardinality objective",
            enforced_cardinality_internal.objective == enforced_cardinality_reference.objective
                && enforced_cardinality_internal.objective == Some(1),
            format!(
                "internal={:?} external={:?}",
                enforced_cardinality_internal.objective, enforced_cardinality_reference.objective
            ),
        );
        self.check(
            "CP-SAT enforced cardinality assignment",
            enforced_cardinality_internal.assignment == enforced_cardinality_reference.assignment
                && enforced_cardinality_internal.assignment == vec![1, 0, 0, 1, 0, 0, 0],
            format!(
                "internal={:?} external={:?}",
                enforced_cardinality_internal.assignment, enforced_cardinality_reference.assignment
            ),
        );

        let enforced_table_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::EnforcedAllowedAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    vars: vec![2, 3],
                    tuples: vec![vec![0, 1], vec![1, 0]],
                },
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    vars: vec![2, 3],
                    tuples: vec![vec![0, 1]],
                },
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    vars: vec![4, 5],
                    tuples: vec![vec![0, 0]],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                ],
            }),
        };
        let enforced_table_internal =
            solve_cp_model(&enforced_table_model, CpSolveOptions::default());
        let enforced_table_json = serde_json::json!({
            "variables": [
                {"name": "active_gate", "domain": [0, 1]},
                {"name": "inactive_gate", "domain": [0, 1]},
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
                {"name": "inactive_x", "domain": [0, 1]},
                {"name": "inactive_y", "domain": [0, 1]},
            ],
            "constraints": [
                {
                    "kind": "bool_or",
                    "literals": [{"var": 0, "positive": true}],
                },
                {
                    "kind": "bool_or",
                    "literals": [{"var": 1, "positive": false}],
                },
                {
                    "kind": "enforced_allowed_assignments",
                    "enforcement": [{"var": 0, "positive": true}],
                    "vars": [2, 3],
                    "tuples": [[0, 1], [1, 0]],
                },
                {
                    "kind": "enforced_forbidden_assignments",
                    "enforcement": [{"var": 0, "positive": true}],
                    "vars": [2, 3],
                    "tuples": [[0, 1]],
                },
                {
                    "kind": "enforced_forbidden_assignments",
                    "enforcement": [{"var": 1, "positive": true}],
                    "vars": [4, 5],
                    "tuples": [[0, 0]],
                },
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 2, "coeff": 1},
                    {"var": 3, "coeff": 1},
                    {"var": 4, "coeff": 1},
                    {"var": 5, "coeff": 1},
                ],
            },
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &enforced_table_json,
        );
        let enforced_table_reference: CpReference =
            serde_json::from_value(value).expect("parse enforced-table CP reference");
        self.check(
            "CP-SAT enforced table status internal/reference",
            enforced_table_internal.status == CpStatus::Optimal
                && enforced_table_reference.status == "optimal",
            format!(
                "internal={} external={} solver={}",
                enforced_table_internal.status.as_str(),
                enforced_table_reference.status,
                enforced_table_reference.solver
            ),
        );
        self.check(
            "CP-SAT enforced table objective",
            enforced_table_internal.objective == enforced_table_reference.objective
                && enforced_table_internal.objective == Some(1),
            format!(
                "internal={:?} external={:?}",
                enforced_table_internal.objective, enforced_table_reference.objective
            ),
        );
        self.check(
            "CP-SAT enforced table assignment",
            enforced_table_internal.assignment == enforced_table_reference.assignment
                && enforced_table_internal.assignment == vec![1, 0, 1, 0, 0, 0],
            format!(
                "internal={:?} external={:?}",
                enforced_table_internal.assignment, enforced_table_reference.assignment
            ),
        );

        let feasible_model = CpModel {
            variables: vec![CpVariable {
                name: "x".to_string(),
                domain: vec![0, 1],
            }],
            constraints: vec![CpConstraint::Linear {
                terms: vec![LinearTerm { var: 0, coeff: 1 }],
                sense: LinearSense::Ge,
                rhs: 1,
            }],
            objective: None,
        };
        let feasible_internal = solve_cp_model(&feasible_model, CpSolveOptions::default());
        let feasible_json = serde_json::json!({
            "variables": [{"name": "x", "domain": [0, 1]}],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [{"var": 0, "coeff": 1}],
                    "sense": "ge",
                    "rhs": 1,
                },
            ],
            "objective": serde_json::Value::Null,
        })
        .to_string();
        let value =
            self.run_python_json("cp_sat_reference.py", &["--solver", "auto"], &feasible_json);
        let feasible_reference: CpReference =
            serde_json::from_value(value).expect("parse feasible CP reference");
        self.check(
            "CP-SAT feasible status internal/reference",
            feasible_internal.status == CpStatus::Feasible
                && feasible_reference.status == "feasible",
            format!(
                "internal={} external={} solver={}",
                feasible_internal.status.as_str(),
                feasible_reference.status,
                feasible_reference.solver
            ),
        );
        self.check(
            "CP-SAT feasible assignment",
            feasible_internal.assignment == feasible_reference.assignment,
            format!(
                "internal={:?} external={:?}",
                feasible_internal.assignment, feasible_reference.assignment
            ),
        );

        let infeasible_model = CpModel {
            variables: vec![CpVariable {
                name: "x".to_string(),
                domain: vec![0, 1],
            }],
            constraints: vec![
                CpConstraint::Linear {
                    terms: vec![LinearTerm { var: 0, coeff: 1 }],
                    sense: LinearSense::Eq,
                    rhs: 0,
                },
                CpConstraint::Linear {
                    terms: vec![LinearTerm { var: 0, coeff: 1 }],
                    sense: LinearSense::Eq,
                    rhs: 1,
                },
            ],
            objective: None,
        };
        let infeasible_internal = solve_cp_model(&infeasible_model, CpSolveOptions::default());
        let infeasible_json = serde_json::json!({
            "variables": [{"name": "x", "domain": [0, 1]}],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [{"var": 0, "coeff": 1}],
                    "sense": "eq",
                    "rhs": 0,
                },
                {
                    "kind": "linear",
                    "terms": [{"var": 0, "coeff": 1}],
                    "sense": "eq",
                    "rhs": 1,
                },
            ],
            "objective": serde_json::Value::Null,
        })
        .to_string();
        let value = self.run_python_json(
            "cp_sat_reference.py",
            &["--solver", "auto"],
            &infeasible_json,
        );
        let infeasible_reference: CpReference =
            serde_json::from_value(value).expect("parse infeasible CP reference");
        self.check(
            "CP-SAT infeasible status internal/reference",
            infeasible_internal.status == CpStatus::Infeasible
                && infeasible_reference.status == "infeasible",
            format!(
                "internal={} external={} solver={}",
                infeasible_internal.status.as_str(),
                infeasible_reference.status,
                infeasible_reference.solver
            ),
        );
    }

    fn run_all(&mut self) {
        self.validate_lp();
        self.validate_ip_mip();
        self.validate_external_solver_clis();
        self.validate_external_optimization_ecosystems();
        self.validate_math_program_facade();
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
