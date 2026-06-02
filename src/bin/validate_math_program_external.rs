//! Validate `des::general::math_program` against optional external solver oracles.
//!
//! This binary is intentionally non-fatal when an external solver is not
//! installed: the internal solve still runs, and that comparison is reported as
//! SKIP.

use des_engine::des::general::math_program::{
    cross_check_math_program_conflict_with_external,
    cross_check_math_program_feas_relaxation_with_external,
    cross_check_math_program_solution_pool_with_external, cross_check_math_program_with_external,
    ExternalMathProgramOptions, MathProgram, MathProgramConflictCrossCheck,
    MathProgramConflictOptions, MathProgramCrossCheck, MathProgramFeasRelaxCrossCheck,
    MathProgramFeasRelaxOptions, MathProgramFeasRelaxViolation, MathProgramLpBackend,
    MathProgramSolutionPoolCrossCheck, MathProgramSolutionPoolOptions, MathProgramSolveOptions,
    MathProgramStatus, ObjectiveSense, RowSense,
};

fn main() {
    let cases = vec![
        ("lp-row-senses", build_lp_case()),
        (
            "lp-equality-certificates",
            build_lp_equality_certificate_case(),
        ),
        ("multi-objective", build_multi_objective_case()),
        ("continuous-qp", build_continuous_qp_case()),
        ("continuous-qcp", build_continuous_qcp_case()),
        ("continuous-soc", build_continuous_soc_case()),
        (
            "continuous-rotated-soc",
            build_continuous_rotated_soc_case(),
        ),
        ("mixed-integer-qp", build_mixed_integer_qp_case()),
        ("mixed-integer-qcp", build_mixed_integer_qcp_case()),
        ("mixed-integer-soc", build_mixed_integer_soc_case()),
        ("binary-mip", build_binary_mip_case()),
        ("lazy-constraint", build_lazy_constraint_case()),
        ("binary-quadratic", build_binary_quadratic_case()),
        ("semi-continuous", build_semi_continuous_case()),
        ("semi-integer", build_semi_integer_case()),
        ("indicator-row", build_indicator_case()),
        ("sos1", build_sos1_case()),
        ("sos2", build_sos2_case()),
        ("binary-general", build_binary_general_case()),
        ("absolute-value", build_abs_case()),
        ("maximum", build_max_case()),
        ("minimum", build_min_case()),
        ("piecewise-linear", build_piecewise_linear_case()),
        ("all-different", build_all_different_case()),
        ("allowed-assignments", build_allowed_assignments_case()),
        ("no-overlap", build_no_overlap_case()),
        ("no-overlap-2d", build_no_overlap_2d_case()),
        ("cumulative", build_cumulative_case()),
    ];

    let mut failed = 0usize;
    for (name, program) in cases {
        match run_case(name, &program) {
            Ok(true) => {}
            Ok(false) => failed += 1,
            Err(err) => {
                failed += 1;
                println!("FAIL  {name}: {err:?}");
            }
        }
    }
    match run_mip_start_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  mip-start: {err:?}");
        }
    }
    match run_solution_pool_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  solution-pool: {err:?}");
        }
    }
    match run_conflict_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  linear-conflict: {err:?}");
        }
    }
    match run_feas_relax_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  feasibility-relaxation: {err:?}");
        }
    }

    if failed > 0 {
        std::process::exit(1);
    }
}

fn run_case(name: &str, program: &MathProgram) -> Result<bool, String> {
    let continuous_nonlinear = !program.has_discrete_features()
        && (program.has_quadratic_objective()
            || program.has_quadratic_constraints()
            || program.has_conic_constraints());
    let mixed_integer_nonlinear = program.has_discrete_features()
        && (program.has_quadratic_constraints() || program.has_conic_constraints());
    let direct_mixed_integer_qp = name == "mixed-integer-qp";
    let ortools_method = if program.has_discrete_features() {
        "ortools:SCIP"
    } else {
        "ortools:GLOP"
    };
    let mut external_methods = if continuous_nonlinear {
        vec![
            ("scipy-slsqp", Some("SLSQP".to_string())),
            ("gurobi", Some("gurobi:default".to_string())),
            ("cplex", Some("cplex:default".to_string())),
            ("xpress", Some("xpress:default".to_string())),
        ]
    } else if mixed_integer_nonlinear || direct_mixed_integer_qp {
        vec![
            ("gurobi", Some("gurobi:default".to_string())),
            ("cplex", Some("cplex:default".to_string())),
            ("xpress", Some("xpress:default".to_string())),
        ]
    } else {
        vec![
            ("scipy-highs", None),
            ("highs-cli", Some("highs-cli:default".to_string())),
            ("cbc-cli", Some("cbc-cli:default".to_string())),
            ("ortools", Some(ortools_method.to_string())),
            ("glpk", Some("glpk:default".to_string())),
            ("glpk-cli", Some("glpk-cli:default".to_string())),
            ("scip-cli", Some("scip-cli:default".to_string())),
            ("gurobi", Some("gurobi:default".to_string())),
            ("cplex", Some("cplex:default".to_string())),
            ("xpress", Some("xpress:default".to_string())),
        ]
    };
    if program.has_discrete_features() && !mixed_integer_nonlinear && !direct_mixed_integer_qp {
        external_methods.push(("ortools-cp-sat", Some("ortools:CP-SAT".to_string())));
    }

    let mut ok = true;
    for (label, method) in external_methods {
        let report = cross_check_math_program_with_external(
            program,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method: method.clone(),
                ..Default::default()
            },
            1e-6,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_report(name, label, &report);
        if report.external.status != MathProgramStatus::NumericalError {
            ok &= report.within_tolerance;
            if let Some(expectation) = lp_certificate_expectation(name) {
                ok &= lp_certificate_case_ok(name, label, &report, expectation);
            }
        }
        if report.external.status != MathProgramStatus::NumericalError {
            if let Some(expectation) = lp_certificate_expectation(name) {
                let des_label = format!("{label}/des-simplex");
                let des_report = cross_check_math_program_with_external(
                    program,
                    &MathProgramSolveOptions {
                        lp_backend: MathProgramLpBackend::DESSimplex,
                        ..Default::default()
                    },
                    &ExternalMathProgramOptions {
                        method,
                        ..Default::default()
                    },
                    1e-6,
                )
                .map_err(|err| format!("{err:?}"))?;

                print_report(name, &des_label, &des_report);
                if des_report.external.status != MathProgramStatus::NumericalError {
                    ok &= des_report.within_tolerance;
                    ok &= lp_certificate_case_ok(name, &des_label, &des_report, expectation);
                }
            }
        }
    }
    Ok(ok)
}

#[derive(Clone, Copy)]
struct LPCertificateExpectation {
    dual_ub: Option<&'static [f64]>,
    dual_eq: Option<&'static [f64]>,
    reduced_costs: &'static [f64],
}

const LP_ROW_SENSES_DUAL_UB: [f64; 3] = [7.0 / 3.0, 0.0, 2.0 / 3.0];
const LP_ROW_SENSES_REDUCED_COSTS: [f64; 2] = [0.0, 0.0];
const LP_EQUALITY_DUAL_EQ: [f64; 1] = [1.0];
const LP_EQUALITY_REDUCED_COSTS: [f64; 1] = [0.0];

fn lp_certificate_expectation(name: &str) -> Option<LPCertificateExpectation> {
    match name {
        "lp-row-senses" => Some(LPCertificateExpectation {
            dual_ub: Some(&LP_ROW_SENSES_DUAL_UB),
            dual_eq: None,
            reduced_costs: &LP_ROW_SENSES_REDUCED_COSTS,
        }),
        "lp-equality-certificates" => Some(LPCertificateExpectation {
            dual_ub: None,
            dual_eq: Some(&LP_EQUALITY_DUAL_EQ),
            reduced_costs: &LP_EQUALITY_REDUCED_COSTS,
        }),
        _ => None,
    }
}

fn lp_certificate_case_ok(
    name: &str,
    label: &str,
    report: &MathProgramCrossCheck,
    expectation: LPCertificateExpectation,
) -> bool {
    let internal_ok = certificate_field_ok(report.internal.dual_ub.as_deref(), expectation.dual_ub)
        && certificate_field_ok(report.internal.dual_eq.as_deref(), expectation.dual_eq)
        && certificate_vectors_close(
            report.internal.reduced_costs.as_deref(),
            expectation.reduced_costs,
            1e-6,
        );
    let external_ok = certificate_field_ok(report.external.dual_ub.as_deref(), expectation.dual_ub)
        && certificate_field_ok(report.external.dual_eq.as_deref(), expectation.dual_eq)
        && certificate_vectors_close(
            report.external.reduced_costs.as_deref(),
            expectation.reduced_costs,
            1e-6,
        );
    if internal_ok && external_ok {
        println!("PASS  {name} [{label}] certificates: duals/reduced-costs");
        true
    } else if !external_ok && !external_certificates_required(label) {
        println!("SKIP  {name} [{label}] certificates: external certificate fields unavailable");
        internal_ok
    } else {
        println!(
            "FAIL  {name} [{label}] certificates: internal_dual_ub={:?} external_dual_ub={:?} internal_dual_eq={:?} external_dual_eq={:?} internal_reduced={:?} external_reduced={:?}",
            report.internal.dual_ub,
            report.external.dual_ub,
            report.internal.dual_eq,
            report.external.dual_eq,
            report.internal.reduced_costs,
            report.external.reduced_costs
        );
        false
    }
}

fn external_certificates_required(label: &str) -> bool {
    let base = label.strip_suffix("/des-simplex").unwrap_or(label);
    matches!(
        base,
        "scipy-highs"
            | "highs-cli"
            | "ortools"
            | "glpk"
            | "glpk-cli"
            | "gurobi"
            | "cplex"
            | "xpress"
    )
}

fn certificate_field_ok(actual: Option<&[f64]>, expected: Option<&[f64]>) -> bool {
    match expected {
        Some(expected) => certificate_vectors_close(actual, expected, 1e-6),
        None => true,
    }
}

fn certificate_vectors_close(actual: Option<&[f64]>, expected: &[f64], tol: f64) -> bool {
    actual.is_some_and(|actual| {
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, e)| (a - e).abs() <= tol)
    })
}

fn run_mip_start_case() -> Result<bool, String> {
    let name = "mip-start";
    let program = build_mip_start_case();
    let start = vec![0.0, 1.0, 1.0];
    let methods = vec![
        ("scipy-highs", None),
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
    ];

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_with_external(
            &program,
            &MathProgramSolveOptions {
                mip_start: Some(start.clone()),
                ..Default::default()
            },
            &ExternalMathProgramOptions {
                method,
                mip_start: Some(start.clone()),
                ..Default::default()
            },
            1e-6,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_report(name, label, &report);
        if report.external.status == MathProgramStatus::NumericalError {
            continue;
        }
        ok &= report.within_tolerance
            && report
                .internal
                .message
                .as_deref()
                .is_some_and(|message| message.contains("incumbent_source=mip-start"));
    }
    Ok(ok)
}

fn run_conflict_case() -> Result<bool, String> {
    let name = "linear-conflict";
    let program = build_conflict_case();
    let methods = vec![
        ("scipy-highs", None),
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
    ];
    let conflict_opts = MathProgramConflictOptions::default();

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_conflict_with_external(
            &program,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method,
                ..Default::default()
            },
            &conflict_opts,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_conflict_report(name, label, &report);
        if report.external.status == MathProgramStatus::NumericalError {
            continue;
        }
        ok &=
            report.within_tolerance && report.internal.minimal && report.internal.items.len() == 2;
    }
    Ok(ok)
}

fn run_feas_relax_case() -> Result<bool, String> {
    let name = "feasibility-relaxation";
    let program = build_feas_relax_case();
    let methods = vec![
        ("scipy-highs", None),
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:GLOP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
    ];
    let relax_opts = MathProgramFeasRelaxOptions {
        linear_penalty: 10.0,
        bound_penalty: 1.0,
        ..Default::default()
    };

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_feas_relaxation_with_external(
            &program,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method,
                ..Default::default()
            },
            &relax_opts,
            1e-6,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_feas_relax_report(name, label, &report);
        if report.external.status == MathProgramStatus::NumericalError {
            continue;
        }
        ok &= report.within_tolerance
            && (report.internal.violation_objective - 1.0).abs() <= 1e-6
            && report.internal.violations.len() == 1
            && matches!(
                report.internal.violations.first(),
                Some(MathProgramFeasRelaxViolation::VariableLowerBound {
                    name,
                    violation,
                    ..
                }) if name == "x" && (*violation - 1.0).abs() <= 1e-6
            );
    }
    Ok(ok)
}

fn run_solution_pool_case() -> Result<bool, String> {
    let name = "solution-pool";
    let program = build_solution_pool_case();
    let methods = vec![
        ("scipy-highs", None),
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
    ];
    let pool_opts = MathProgramSolutionPoolOptions {
        max_solutions: 3,
        ..Default::default()
    };

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_solution_pool_with_external(
            &program,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method,
                ..Default::default()
            },
            &pool_opts,
            1e-6,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_pool_report(name, label, &report);
        if report.external.solutions.is_empty() {
            continue;
        }
        ok &= report.within_tolerance;
    }
    Ok(ok)
}

fn print_conflict_report(name: &str, label: &str, report: &MathProgramConflictCrossCheck) {
    println!(
        "{} [{}]  internal_conflict={:?} minimal={} items={:?}",
        name, label, report.internal.status, report.internal.minimal, report.internal.items
    );
    println!(
        "{} [{}]  external={} {:?}",
        name, label, report.external.solver, report.external.status
    );
    if report.external.status == MathProgramStatus::NumericalError {
        println!(
            "SKIP  {name} [{label}]: external solver unavailable ({})",
            report
                .external
                .message
                .as_deref()
                .unwrap_or("no external diagnostic")
        );
    } else if report.within_tolerance && report.internal.minimal && report.internal.items.len() == 2
    {
        println!(
            "PASS  {name} [{label}]: status_agree={} conflict_items={}",
            report.status_agree,
            report.internal.items.len()
        );
    } else {
        println!(
            "FAIL  {name} [{label}]: status_agree={} conflict_items={} minimal={}",
            report.status_agree,
            report.internal.items.len(),
            report.internal.minimal
        );
    }
}

fn print_feas_relax_report(name: &str, label: &str, report: &MathProgramFeasRelaxCrossCheck) {
    println!(
        "{} [{}]  internal_feas_relax={:?} obj={:.8} x={:?} violations={:?}",
        name,
        label,
        report.internal.status,
        report.internal.violation_objective,
        report.internal.x,
        report.internal.violations
    );
    println!(
        "{} [{}]  external={} {:?} obj={:.8} x={:?}",
        name,
        label,
        report.external.solver,
        report.external.status,
        report.external.objective,
        report.external.x
    );
    if report.external.status == MathProgramStatus::NumericalError {
        println!(
            "SKIP  {name} [{label}]: external solver unavailable ({})",
            report
                .external
                .message
                .as_deref()
                .unwrap_or("no external diagnostic")
        );
    } else if report.within_tolerance && report.internal.violations.len() == 1 {
        println!(
            "PASS  {name} [{label}]: status_agree={} objective_diff={:?} violations={}",
            report.status_agree,
            report.objective_abs_diff,
            report.internal.violations.len()
        );
    } else {
        println!(
            "FAIL  {name} [{label}]: status_agree={} objective_diff={:?} violations={:?}",
            report.status_agree, report.objective_abs_diff, report.internal.violations
        );
    }
}

fn print_report(name: &str, label: &str, report: &MathProgramCrossCheck) {
    println!(
        "{} [{}]  internal={:?} obj={:.8} x={:?}",
        name, label, report.internal.status, report.internal.objective, report.internal.x
    );
    println!(
        "{} [{}]  external={} {:?} obj={:.8} x={:?}",
        name,
        label,
        report.external.solver,
        report.external.status,
        report.external.objective,
        report.external.x
    );
    if report.external.status == MathProgramStatus::NumericalError {
        println!(
            "SKIP  {name} [{label}]: external solver unavailable ({})",
            report
                .external
                .message
                .as_deref()
                .unwrap_or("no external diagnostic")
        );
    } else if report.within_tolerance {
        println!(
            "PASS  {name} [{label}]: objective_diff={:?} max_x_diff={:?} internal_violation={:?} external_violation={:?}",
            report.objective_abs_diff,
            report.max_x_abs_diff,
            report.internal_max_violation,
            report.external_max_violation
        );
    } else {
        println!(
            "FAIL  {name} [{label}]: status_agree={} objective_diff={:?} max_x_diff={:?} internal_violation={:?} external_violation={:?}",
            report.status_agree,
            report.objective_abs_diff,
            report.max_x_abs_diff,
            report.internal_max_violation,
            report.external_max_violation
        );
    }
}

fn print_pool_report(name: &str, label: &str, report: &MathProgramSolutionPoolCrossCheck) {
    let internal = report
        .internal
        .solutions
        .iter()
        .map(|sol| (sol.objective, sol.x.clone()))
        .collect::<Vec<_>>();
    let external = report
        .external
        .solutions
        .iter()
        .map(|sol| (sol.objective, sol.x.clone()))
        .collect::<Vec<_>>();
    println!("{name} [{label}]  internal_pool={internal:?}");
    println!(
        "{name} [{label}]  external_pool={} {external:?}",
        report.external.solver
    );
    if report.external.solutions.is_empty() {
        println!(
            "SKIP  {name} [{label}]: external solver unavailable ({})",
            report
                .external
                .message
                .as_deref()
                .unwrap_or("no external diagnostic")
        );
    } else if report.within_tolerance {
        println!(
            "PASS  {name} [{label}]: objective_diffs={:?} max_x_diffs={:?}",
            report.objective_abs_diffs, report.max_x_abs_diffs
        );
    } else {
        println!(
            "FAIL  {name} [{label}]: len_agree={} objective_diffs={:?} max_x_diffs={:?} internal_violations={:?} external_violations={:?}",
            report.len_agree,
            report.objective_abs_diffs,
            report.max_x_abs_diffs,
            report.internal_max_violations,
            report.external_max_violations
        );
    }
}

fn build_lp_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_continuous_var("x", 3.0, Some(0.0), None).unwrap();
    let y = p.add_continuous_var("y", 4.0, Some(0.0), None).unwrap();
    p.add_constraint("c0", vec![(x, 1.0), (y, 2.0)], RowSense::Le, 14.0)
        .unwrap();
    p.add_constraint("c1", vec![(x, 3.0), (y, -1.0)], RowSense::Ge, 0.0)
        .unwrap();
    p.add_constraint("c2", vec![(x, 1.0), (y, -1.0)], RowSense::Le, 2.0)
        .unwrap();
    p
}

fn build_lp_equality_certificate_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_continuous_var("x", 1.0, Some(0.0), None).unwrap();
    p.add_constraint("fixed", vec![(x, 1.0)], RowSense::Eq, 2.0)
        .unwrap();
    p
}

fn build_multi_objective_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p
        .add_continuous_var("x", 1.0, Some(0.0), Some(4.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 1.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("budget", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 4.0)
        .unwrap();
    p.add_secondary_objective("prefer-y", ObjectiveSense::Min, 10, 1.0, vec![(x, 1.0)])
        .unwrap();
    p
}

fn build_mip_start_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 60.0).unwrap();
    let b = p.add_binary_var("b", 100.0).unwrap();
    let c = p.add_binary_var("c", 120.0).unwrap();
    p.add_constraint(
        "capacity",
        vec![(a, 10.0), (b, 20.0), (c, 30.0)],
        RowSense::Le,
        50.0,
    )
    .unwrap();
    p
}

fn build_conflict_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_binary_var("x", 0.0).unwrap();
    p.add_constraint("x-on", vec![(x, 1.0)], RowSense::Ge, 1.0)
        .unwrap();
    p.add_constraint("x-off", vec![(x, 1.0)], RowSense::Le, 0.0)
        .unwrap();
    p.add_constraint("redundant-nonnegative", vec![(x, 1.0)], RowSense::Ge, 0.0)
        .unwrap();
    p
}

fn build_feas_relax_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_continuous_var("x", 0.0, Some(2.0), None).unwrap();
    p.add_constraint("cap", vec![(x, 1.0)], RowSense::Le, 1.0)
        .unwrap();
    p
}

fn build_solution_pool_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 4.0).unwrap();
    let b = p.add_binary_var("b", 2.0).unwrap();
    let c = p.add_binary_var("c", 1.0).unwrap();
    p.add_constraint(
        "choose-at-most-two",
        vec![(a, 1.0), (b, 1.0), (c, 1.0)],
        RowSense::Le,
        2.0,
    )
    .unwrap();
    p
}

fn build_continuous_qp_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", -4.0, Some(0.0), Some(5.0))
        .unwrap();
    p.add_quadratic_objective_term(x, x, 1.0).unwrap();
    p
}

fn build_continuous_qcp_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 1.0, Some(0.0), Some(20.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_quadratic_constraint(
        "epigraph-square",
        vec![(x, x, 1.0)],
        vec![(y, -1.0)],
        RowSense::Le,
        0.0,
    )
    .unwrap();
    p
}

fn build_continuous_soc_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(0.0), Some(3.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let t = p
        .add_continuous_var("t", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_second_order_cone(
        "norm-bound",
        vec![
            MathProgram::affine_term(vec![(x, 1.0)], 0.0),
            MathProgram::affine_term(vec![(y, 1.0)], 0.0),
        ],
        vec![(t, 1.0)],
        0.0,
    )
    .unwrap();
    p
}

fn build_continuous_rotated_soc_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let u = p
        .add_continuous_var("u", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let v = p
        .add_continuous_var("v", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    let z = p
        .add_continuous_var("z", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-u", vec![(u, 1.0)], RowSense::Eq, 2.0)
        .unwrap();
    p.add_constraint("fix-z", vec![(z, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_rotated_second_order_cone(
        "rotated-energy",
        MathProgram::affine_term(vec![(u, 1.0)], 0.0),
        MathProgram::affine_term(vec![(v, 1.0)], 0.0),
        vec![MathProgram::affine_term(vec![(z, 1.0)], 0.0)],
    )
    .unwrap();
    p
}

fn build_mixed_integer_qp_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", -4.0, Some(0.0), Some(5.0)).unwrap();
    p.add_quadratic_objective_term(x, x, 1.0).unwrap();
    p
}

fn build_mixed_integer_qcp_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(0.0), Some(5.0)).unwrap();
    let y = p
        .add_continuous_var("y", 1.0, Some(0.0), Some(20.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_quadratic_constraint(
        "integer-square",
        vec![(x, x, 1.0)],
        vec![(y, -1.0)],
        RowSense::Le,
        0.0,
    )
    .unwrap();
    p
}

fn build_mixed_integer_soc_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(0.0), Some(3.0)).unwrap();
    let y = p.add_integer_var("y", 0.0, Some(0.0), Some(4.0)).unwrap();
    let t = p
        .add_continuous_var("t", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_second_order_cone(
        "integer-norm",
        vec![
            MathProgram::affine_term(vec![(x, 1.0)], 0.0),
            MathProgram::affine_term(vec![(y, 1.0)], 0.0),
        ],
        vec![(t, 1.0)],
        0.0,
    )
    .unwrap();
    p
}

fn build_binary_mip_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 10.0).unwrap();
    let b = p.add_binary_var("b", 6.0).unwrap();
    p.add_constraint("packing", vec![(a, 2.0), (b, 1.0)], RowSense::Le, 2.0)
        .unwrap();
    p
}

fn build_lazy_constraint_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_binary_var("x", 1.0).unwrap();
    let y = p.add_binary_var("y", 1.0).unwrap();
    p.add_lazy_constraint(
        "lazy-at-most-one",
        vec![(x, 1.0), (y, 1.0)],
        RowSense::Le,
        1.0,
    )
    .unwrap();
    p
}

fn build_binary_quadratic_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 4.0).unwrap();
    let b = p.add_binary_var("b", 3.0).unwrap();
    p.add_quadratic_objective_term(a, b, -4.0).unwrap();
    p
}

fn build_semi_continuous_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_semi_continuous_var("x", 1.0, 5.0, 10.0).unwrap();
    p.add_constraint("must-produce", vec![(x, 1.0)], RowSense::Ge, 1.0)
        .unwrap();
    p
}

fn build_semi_integer_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_semi_integer_var("x", 1.0, 3.0, 7.0).unwrap();
    p.add_constraint("must-produce", vec![(x, 1.0)], RowSense::Ge, 1.0)
        .unwrap();
    p
}

fn build_indicator_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let active = p.add_binary_var("active", 0.0).unwrap();
    let x = p.add_integer_var("x", 1.0, Some(0.0), Some(5.0)).unwrap();
    p.add_constraint("force-active", vec![(active, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_indicator(
        "cap-when-active",
        active,
        true,
        vec![(x, 1.0)],
        RowSense::Le,
        2.0,
    )
    .unwrap();
    p
}

fn build_sos1_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p
        .add_continuous_var("x0", 5.0, Some(0.0), Some(1.0))
        .unwrap();
    let x1 = p
        .add_continuous_var("x1", 7.0, Some(0.0), Some(1.0))
        .unwrap();
    let x2 = p
        .add_continuous_var("x2", 3.0, Some(0.0), Some(1.0))
        .unwrap();
    p.add_sos1("choose-one", vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)])
        .unwrap();
    p
}

fn build_sos2_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p
        .add_continuous_var("x0", 7.0, Some(0.0), Some(1.0))
        .unwrap();
    let x1 = p
        .add_continuous_var("x1", 1.0, Some(0.0), Some(1.0))
        .unwrap();
    let x2 = p
        .add_continuous_var("x2", 6.0, Some(0.0), Some(1.0))
        .unwrap();
    p.add_constraint(
        "pick-two",
        vec![(x0, 1.0), (x1, 1.0), (x2, 1.0)],
        RowSense::Eq,
        2.0,
    )
    .unwrap();
    p.add_sos2("adjacent-pair", vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)])
        .unwrap();
    p
}

fn build_binary_general_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 0.0).unwrap();
    let b = p.add_binary_var("b", 0.0).unwrap();
    let both = p.add_binary_var("both", 2.0).unwrap();
    let either = p.add_binary_var("either", 1.0).unwrap();
    p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_constraint("force-b-off", vec![(b, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_binary_and("both-active", both, vec![a, b]).unwrap();
    p.add_binary_or("either-active", either, vec![a, b])
        .unwrap();
    p
}

fn build_abs_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(-5.0), Some(4.0))
        .unwrap();
    let r = p
        .add_continuous_var("abs_x", 1.0, Some(0.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -3.0)
        .unwrap();
    p.add_abs("absolute-value", r, x).unwrap();
    p
}

fn build_max_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a = p
        .add_continuous_var("a", 0.0, Some(-2.0), Some(5.0))
        .unwrap();
    let b = p
        .add_continuous_var("b", 0.0, Some(-2.0), Some(5.0))
        .unwrap();
    let r = p
        .add_continuous_var("max_ab", 1.0, Some(-2.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-a", vec![(a, 1.0)], RowSense::Eq, 2.0)
        .unwrap();
    p.add_constraint("fix-b", vec![(b, 1.0)], RowSense::Eq, -1.0)
        .unwrap();
    p.add_max("maximum", r, vec![a, b]).unwrap();
    p
}

fn build_min_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p
        .add_continuous_var("a", 0.0, Some(-2.0), Some(5.0))
        .unwrap();
    let b = p
        .add_continuous_var("b", 0.0, Some(-2.0), Some(5.0))
        .unwrap();
    let r = p
        .add_continuous_var("min_ab", 1.0, Some(-2.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-a", vec![(a, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_constraint("fix-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_min("minimum", r, vec![a, b]).unwrap();
    p
}

fn build_piecewise_linear_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 1.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 1.5)
        .unwrap();
    p.add_piecewise_linear("square-ish", x, y, vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)])
        .unwrap();
    p
}

fn build_all_different_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p
        .add_integer_var("x0", 100.0, Some(0.0), Some(2.0))
        .unwrap();
    let x1 = p.add_integer_var("x1", 10.0, Some(0.0), Some(2.0)).unwrap();
    let x2 = p.add_integer_var("x2", 1.0, Some(0.0), Some(2.0)).unwrap();
    p.add_all_different("permute", vec![x0, x1, x2]).unwrap();
    p
}

fn build_allowed_assignments_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_integer_var("x", 10.0, Some(0.0), Some(2.0)).unwrap();
    let y = p.add_integer_var("y", 1.0, Some(0.0), Some(2.0)).unwrap();
    p.add_allowed_assignments("allowed-pairs", vec![x, y], vec![vec![0, 2], vec![1, 1]])
        .unwrap();
    p
}

fn build_no_overlap_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 1.0, Some(0.0), Some(5.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_no_overlap(
        "single-machine",
        vec![
            MathProgram::interval(a_start, 3.0, a_end),
            MathProgram::interval(b_start, 2.0, b_end),
        ],
    )
    .unwrap();
    p
}

fn build_no_overlap_2d_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_x_start = p
        .add_integer_var("a_x_start", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let a_x_end = p
        .add_integer_var("a_x_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let a_y_start = p
        .add_integer_var("a_y_start", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let a_y_end = p
        .add_integer_var("a_y_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let b_x_start = p
        .add_integer_var("b_x_start", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_x_end = p
        .add_integer_var("b_x_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let b_y_start = p
        .add_integer_var("b_y_start", 1.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_y_end = p
        .add_integer_var("b_y_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();

    p.add_constraint("fix-a-x-start", vec![(a_x_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_constraint("fix-a-y-start", vec![(a_y_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_constraint("fix-b-x-start", vec![(b_x_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_no_overlap_2d(
        "packing",
        vec![
            MathProgram::interval(a_x_start, 2.0, a_x_end),
            MathProgram::interval(b_x_start, 2.0, b_x_end),
        ],
        vec![
            MathProgram::interval(a_y_start, 2.0, a_y_end),
            MathProgram::interval(b_y_start, 2.0, b_y_end),
        ],
    )
    .unwrap();
    p
}

fn build_cumulative_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 1.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_cumulative(
        "shared-resource",
        vec![
            MathProgram::interval(a_start, 2.0, a_end),
            MathProgram::interval(b_start, 2.0, b_end),
        ],
        vec![2.0, 2.0],
        3.0,
    )
    .unwrap();
    p
}
