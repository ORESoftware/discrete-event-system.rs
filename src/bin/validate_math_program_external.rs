//! Validate `des::general::math_program` against optional external solver oracles.
//!
//! This binary is intentionally non-fatal when an external solver is not
//! installed: the internal solve still runs, and that comparison is reported as
//! SKIP.

use des_engine::des::general::math_program::{
    cross_check_math_program_assumption_core_with_external,
    cross_check_math_program_conflict_with_external,
    cross_check_math_program_feas_relaxation_with_external,
    cross_check_math_program_solution_pool_with_external, cross_check_math_program_with_external,
    AffineTerm, ExternalMathProgramOptions, MathProgram, MathProgramAssumptionCoreCrossCheck,
    MathProgramAssumptionCoreOptions, MathProgramConflictCrossCheck, MathProgramConflictOptions,
    MathProgramCrossCheck, MathProgramFeasRelaxCrossCheck, MathProgramFeasRelaxOptions,
    MathProgramFeasRelaxViolation, MathProgramLpBackend, MathProgramSolution,
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
        ("multi-literal-enforced-row", build_enforced_row_case()),
        ("sos1", build_sos1_case()),
        ("integer-sos1", build_integer_sos1_case()),
        ("sos2", build_sos2_case()),
        ("integer-sos2", build_integer_sos2_case()),
        ("binary-general", build_binary_general_case()),
        ("binary-xor", build_binary_xor_case()),
        ("binary-cardinality", build_binary_cardinality_case()),
        ("boolean-clause", build_boolean_clause_case()),
        ("integer-product", build_integer_product_case()),
        (
            "integer-division-modulo",
            build_integer_division_modulo_case(),
        ),
        ("absolute-value", build_abs_case()),
        ("integer-absolute-value", build_integer_abs_case()),
        ("maximum", build_max_case()),
        ("integer-maximum", build_integer_max_case()),
        ("minimum", build_min_case()),
        ("integer-minimum", build_integer_min_case()),
        ("l1-norm", build_l1_norm_case()),
        ("integer-l1-norm", build_integer_l1_norm_case()),
        ("l-infinity-norm", build_l_infinity_norm_case()),
        (
            "integer-l-infinity-norm",
            build_integer_l_infinity_norm_case(),
        ),
        ("l2-norm", build_l2_norm_case()),
        ("integer-l2-norm", build_integer_l2_norm_case()),
        ("piecewise-linear", build_piecewise_linear_case()),
        ("all-different", build_all_different_case()),
        ("allowed-assignments", build_allowed_assignments_case()),
        ("forbidden-assignments", build_forbidden_assignments_case()),
        ("bin-packing", build_bin_packing_case()),
        ("element", build_element_case()),
        ("variable-element", build_variable_element_case()),
        ("inverse", build_inverse_case()),
        ("circuit", build_circuit_case()),
        ("multiple-circuit", build_multiple_circuit_case()),
        ("automaton", build_automaton_case()),
        ("alternative-interval", build_alternative_interval_case()),
        ("no-overlap", build_no_overlap_case()),
        ("variable-no-overlap", build_variable_no_overlap_case()),
        ("optional-no-overlap", build_optional_no_overlap_case()),
        ("no-overlap-2d", build_no_overlap_2d_case()),
        (
            "variable-no-overlap-2d",
            build_variable_no_overlap_2d_case(),
        ),
        (
            "optional-no-overlap-2d",
            build_optional_no_overlap_2d_case(),
        ),
        ("cumulative", build_cumulative_case()),
        ("variable-cumulative", build_variable_cumulative_case()),
        ("affine-cumulative", build_affine_cumulative_case()),
        ("optional-cumulative", build_optional_cumulative_case()),
        ("reservoir", build_reservoir_case()),
        ("optional-reservoir", build_optional_reservoir_case()),
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
    match run_external_mip_options_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  external-mip-options: {err:?}");
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
    match run_assumption_core_case() {
        Ok(true) => {}
        Ok(false) => failed += 1,
        Err(err) => {
            failed += 1;
            println!("FAIL  assumption-core: {err:?}");
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
    let external_methods = external_methods_for_case(name, program);

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
            if program.has_discrete_features() {
                ok &= mip_quality_case_ok(name, label, &report);
            }
            if let Some(expectation) = lp_certificate_expectation(name) {
                ok &= lp_certificate_case_ok(name, label, &report, expectation);
                ok &= lp_basis_case_ok(name, label, &report);
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
                    ok &= lp_basis_case_ok(name, &des_label, &des_report);
                }
            }
        }
    }
    Ok(ok)
}

fn external_methods_for_case(
    name: &str,
    program: &MathProgram,
) -> Vec<(&'static str, Option<String>)> {
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
    } else if program.has_discrete_features() {
        mixed_integer_linear_methods(ortools_method)
    } else {
        continuous_linear_methods(ortools_method)
    };
    if program.has_discrete_features() && !mixed_integer_nonlinear && !direct_mixed_integer_qp {
        external_methods.push(("ortools-cp-sat", Some("ortools:CP-SAT".to_string())));
        external_methods.push(("scipy-highs", None));
    }
    external_methods
}

fn mixed_integer_linear_methods(ortools_method: &str) -> Vec<(&'static str, Option<String>)> {
    vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some(ortools_method.to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
    ]
}

fn continuous_linear_methods(ortools_method: &str) -> Vec<(&'static str, Option<String>)> {
    vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("clp-cli", Some("clp-cli".to_string())),
        ("soplex-cli", Some("soplex-cli".to_string())),
        ("qsopt-ex-cli", Some("qsopt-ex-cli".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("ortools", Some(ortools_method.to_string())),
        ("ortools-pdlp", Some("ortools:PDLP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("scipy-highs", None),
    ]
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
    } else if !external_ok && !external_certificates_required(name, label) {
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

fn lp_basis_case_ok(name: &str, label: &str, report: &MathProgramCrossCheck) -> bool {
    if name != "lp-row-senses" {
        return true;
    }
    let expected_var = ["basic", "basic"];
    let expected_row = ["at_upper", "basic", "at_upper"];
    let internal_ok = basis_vectors_equal(report.internal.var_basis.as_deref(), &expected_var)
        && basis_vectors_equal(report.internal.row_basis.as_deref(), &expected_row);
    let external_ok = basis_vectors_equal(report.external.var_basis.as_deref(), &expected_var)
        && basis_vectors_equal(report.external.row_basis.as_deref(), &expected_row);
    if internal_ok && external_ok {
        println!("PASS  {name} [{label}] basis: var/row statuses");
        true
    } else if !external_ok && !external_basis_required(label) {
        println!("SKIP  {name} [{label}] basis: external basis fields unavailable");
        internal_ok
    } else {
        println!(
            "FAIL  {name} [{label}] basis: internal_var={:?} external_var={:?} internal_row={:?} external_row={:?}",
            report.internal.var_basis,
            report.external.var_basis,
            report.internal.row_basis,
            report.external.row_basis
        );
        false
    }
}

fn external_certificates_required(_name: &str, label: &str) -> bool {
    let base = label.strip_suffix("/des-simplex").unwrap_or(label);
    matches!(
        base,
        "scipy-highs"
            | "highs-cli"
            | "cbc-cli"
            | "ortools"
            | "glpk"
            | "glpk-cli"
            | "gurobi"
            | "cplex"
            | "xpress"
    )
}

fn external_basis_required(label: &str) -> bool {
    let base = label.strip_suffix("/des-simplex").unwrap_or(label);
    matches!(
        base,
        "highs-cli" | "cbc-cli" | "ortools" | "glpk" | "glpk-cli" | "gurobi" | "cplex" | "xpress"
    )
}

fn mip_quality_case_ok(name: &str, label: &str, report: &MathProgramCrossCheck) -> bool {
    let internal_ok = report.internal.best_bound.is_some()
        && report
            .internal
            .mip_gap
            .is_some_and(|gap| gap.is_finite() && gap >= -1e-9)
        && report.internal.nodes_explored.is_some();
    let external_has_quality = report.external.best_bound.is_some()
        || report.external.mip_gap.is_some()
        || report.external.nodes_explored.is_some();
    let external_ok = external_has_quality && quality_metadata_consistent(&report.external);
    if internal_ok && external_ok {
        println!(
            "PASS  {name} [{label}] quality: internal_bound={:?} external_bound={:?} external_gap={:?} external_nodes={:?}",
            report.internal.best_bound,
            report.external.best_bound,
            report.external.mip_gap,
            report.external.nodes_explored
        );
        true
    } else if internal_ok && !external_has_quality && !external_quality_required(label) {
        println!("SKIP  {name} [{label}] quality: external MIP quality fields unavailable");
        true
    } else {
        println!(
            "FAIL  {name} [{label}] quality: internal_bound={:?} internal_gap={:?} internal_nodes={:?} external_bound={:?} external_gap={:?} external_nodes={:?}",
            report.internal.best_bound,
            report.internal.mip_gap,
            report.internal.nodes_explored,
            report.external.best_bound,
            report.external.mip_gap,
            report.external.nodes_explored
        );
        false
    }
}

fn external_quality_required(label: &str) -> bool {
    let base = label.strip_suffix("/des-simplex").unwrap_or(label);
    matches!(base, "scipy-highs" | "ortools-cp-sat" | "gurobi" | "cplex")
}

fn quality_metadata_consistent(solution: &MathProgramSolution) -> bool {
    if let Some(gap) = solution.mip_gap {
        if !gap.is_finite() || gap < -1e-9 {
            return false;
        }
    }
    if let (Some(best_bound), Some(gap)) = (solution.best_bound, solution.mip_gap) {
        if solution.objective.is_finite() {
            let implied_gap =
                (best_bound - solution.objective).abs() / 1.0_f64.max(solution.objective.abs());
            if implied_gap > gap.max(1e-6) + 1e-6 {
                return false;
            }
        }
    }
    true
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

fn basis_vectors_equal(actual: Option<&[String]>, expected: &[&str]) -> bool {
    actual.is_some_and(|actual| {
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual == expected)
    })
}

fn run_mip_start_case() -> Result<bool, String> {
    let name = "mip-start";
    let program = build_mip_start_case();
    let start = vec![0.0, 1.0, 1.0];
    let methods = vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
        ("scipy-highs", None),
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
                .is_some_and(|message| message.contains("incumbent_source=user-mip-start"));
    }
    Ok(ok)
}

fn run_external_mip_options_case() -> Result<bool, String> {
    let name = "external-mip-options";
    let program = build_mip_start_case();
    let node_limit = 3usize;
    let methods = vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
        ("scipy-highs", None),
    ];

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_with_external(
            &program,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method,
                time_limit_ms: Some(60_000.0),
                node_limit: Some(node_limit),
                relative_gap: Some(0.25),
                ..Default::default()
            },
            1e-6,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_report(name, label, &report);
        if report.external.status == MathProgramStatus::NumericalError {
            continue;
        }
        let node_limit_ok = match report.external.nodes_explored {
            Some(nodes) => nodes <= node_limit,
            None => true,
        };
        if node_limit_ok {
            println!(
                "PASS  {name} [{label}] options: external_nodes={:?}",
                report.external.nodes_explored
            );
        } else {
            println!(
                "FAIL  {name} [{label}] options: external_nodes={:?} node_limit={node_limit}",
                report.external.nodes_explored
            );
        }
        ok &= report.within_tolerance && node_limit_ok;
    }
    Ok(ok)
}

fn run_conflict_case() -> Result<bool, String> {
    let name = "linear-conflict";
    let program = build_conflict_case();
    let methods = vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
        ("scipy-highs", None),
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

fn run_assumption_core_case() -> Result<bool, String> {
    let name = "assumption-core";
    let mut program = MathProgram::new(ObjectiveSense::Min);
    let assume_a = program.add_binary_var("assume-a", 0.0).unwrap();
    let assume_b = program.add_binary_var("assume-b", 0.0).unwrap();
    let assume_noise = program.add_binary_var("assume-noise", 0.0).unwrap();
    program
        .add_constraint(
            "assumption-at-most-one",
            vec![(assume_a, 1.0), (assume_b, 1.0)],
            RowSense::Le,
            1.0,
        )
        .unwrap();
    let assumptions = vec![
        MathProgram::bool_lit(assume_a),
        MathProgram::bool_lit(assume_b),
        MathProgram::not_lit(assume_noise),
    ];
    let methods = vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
        ("scipy-highs", None),
    ];
    let core_opts = MathProgramAssumptionCoreOptions::default();

    let mut ok = true;
    for (label, method) in methods {
        let report = cross_check_math_program_assumption_core_with_external(
            &program,
            &assumptions,
            &MathProgramSolveOptions::default(),
            &ExternalMathProgramOptions {
                method,
                ..Default::default()
            },
            &core_opts,
        )
        .map_err(|err| format!("{err:?}"))?;

        print_assumption_core_report(name, label, &report);
        if report.external.status == MathProgramStatus::NumericalError {
            continue;
        }
        ok &= report.within_tolerance
            && report.internal.minimal
            && report.internal.assumptions.len() == 2;
    }
    Ok(ok)
}

fn run_feas_relax_case() -> Result<bool, String> {
    let name = "feasibility-relaxation";
    let program = build_feas_relax_case();
    let methods = vec![
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("clp-cli", Some("clp-cli".to_string())),
        ("soplex-cli", Some("soplex-cli".to_string())),
        ("qsopt-ex-cli", Some("qsopt-ex-cli".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("ortools", Some("ortools:GLOP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("scipy-highs", None),
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
        ("highs-cli", Some("highs-cli:default".to_string())),
        ("cbc-cli", Some("cbc-cli:default".to_string())),
        ("ortools", Some("ortools:SCIP".to_string())),
        ("glpk", Some("glpk:default".to_string())),
        ("glpk-cli", Some("glpk-cli:default".to_string())),
        ("scip-cli", Some("scip-cli:default".to_string())),
        ("lp-solve-cli", Some("lp-solve-cli".to_string())),
        ("gurobi", Some("gurobi:default".to_string())),
        ("cplex", Some("cplex:default".to_string())),
        ("xpress", Some("xpress:default".to_string())),
        ("lindo-cli", Some("lindo-cli".to_string())),
        ("ortools-cp-sat", Some("ortools:CP-SAT".to_string())),
        ("scipy-highs", None),
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

fn print_assumption_core_report(
    name: &str,
    label: &str,
    report: &MathProgramAssumptionCoreCrossCheck,
) {
    println!(
        "{} [{}]  internal_core={:?} minimal={} assumptions={:?}",
        name, label, report.internal.status, report.internal.minimal, report.internal.assumptions
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
    } else if report.within_tolerance
        && report.internal.minimal
        && report.internal.assumptions.len() == 2
    {
        println!(
            "PASS  {name} [{label}]: status_agree={} core_assumptions={}",
            report.status_agree,
            report.internal.assumptions.len()
        );
    } else {
        println!(
            "FAIL  {name} [{label}]: status_agree={} core_assumptions={} minimal={}",
            report.status_agree,
            report.internal.assumptions.len(),
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

fn build_enforced_row_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_integer_var("x", 1.0, Some(0.0), Some(10.0)).unwrap();
    let a = p.add_binary_var("a", 0.0).unwrap();
    let b = p.add_binary_var("b", 0.0).unwrap();
    p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_constraint("force-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_enforced_constraint(
        "missed-literal-does-not-cap",
        vec![MathProgram::bool_lit(a), MathProgram::not_lit(b)],
        vec![(x, 1.0)],
        RowSense::Le,
        2.0,
    )
    .unwrap();
    p.add_enforced_constraint(
        "all-literals-cap",
        vec![MathProgram::bool_lit(a), MathProgram::bool_lit(b)],
        vec![(x, 1.0)],
        RowSense::Le,
        7.0,
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

fn build_integer_sos1_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p.add_integer_var("x0", 5.0, Some(0.0), Some(1.0)).unwrap();
    let x1 = p.add_integer_var("x1", 7.0, Some(0.0), Some(1.0)).unwrap();
    let x2 = p.add_integer_var("x2", 3.0, Some(0.0), Some(1.0)).unwrap();
    p.add_sos1("choose-one-integer", vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)])
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

fn build_integer_sos2_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p.add_integer_var("x0", 7.0, Some(0.0), Some(1.0)).unwrap();
    let x1 = p.add_integer_var("x1", 1.0, Some(0.0), Some(1.0)).unwrap();
    let x2 = p.add_integer_var("x2", 6.0, Some(0.0), Some(1.0)).unwrap();
    p.add_constraint(
        "pick-two-integers",
        vec![(x0, 1.0), (x1, 1.0), (x2, 1.0)],
        RowSense::Eq,
        2.0,
    )
    .unwrap();
    p.add_sos2(
        "adjacent-integer-pair",
        vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)],
    )
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

fn build_binary_xor_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 0.0).unwrap();
    let b = p.add_binary_var("b", 0.0).unwrap();
    let c = p.add_binary_var("c", 0.0).unwrap();
    let d = p.add_binary_var("d", 0.0).unwrap();
    let even = p.add_binary_var("even", 1.0).unwrap();
    let odd = p.add_binary_var("odd", 2.0).unwrap();
    p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_constraint("force-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_constraint("force-c-off", vec![(c, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_constraint("force-d-off", vec![(d, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_binary_xor("even-parity", even, vec![a, b, c])
        .unwrap();
    p.add_binary_xor("odd-parity", odd, vec![a, c, d]).unwrap();
    p
}

fn build_binary_cardinality_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 5.0).unwrap();
    let b = p.add_binary_var("b", 4.0).unwrap();
    let c = p.add_binary_var("c", 3.0).unwrap();
    let d = p.add_binary_var("d", 2.0).unwrap();
    p.add_at_most_one("at-most-one-ab", vec![a, b]).unwrap();
    p.add_at_least_one("at-least-one-cd", vec![c, d]).unwrap();
    p.add_exactly_k("exactly-two-total", vec![a, b, c, d], 2)
        .unwrap();
    p
}

fn build_boolean_clause_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_binary_var("a", 4.0).unwrap();
    let b = p.add_binary_var("b", 3.0).unwrap();
    let c = p.add_binary_var("c", 2.0).unwrap();
    p.add_binary_implication("a-implies-b", a, b).unwrap();
    p.add_binary_implication("b-implies-c", b, c).unwrap();
    p.add_boolean_clause(
        "choose-something",
        vec![
            MathProgram::bool_lit(a),
            MathProgram::bool_lit(b),
            MathProgram::bool_lit(c),
        ],
    )
    .unwrap();
    p
}

fn build_integer_product_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(0.0), Some(3.0)).unwrap();
    let y = p.add_integer_var("y", 0.0, Some(0.0), Some(3.0)).unwrap();
    let product = p
        .add_integer_var("product", 1.0, Some(0.0), Some(9.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 2.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_multiplication_equality("x-times-y", product, vec![x, y])
        .unwrap();
    p
}

fn build_integer_division_modulo_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let numerator = p
        .add_integer_var("numerator", 0.0, Some(-8.0), Some(8.0))
        .unwrap();
    let denominator = p
        .add_integer_var("denominator", 0.0, Some(1.0), Some(4.0))
        .unwrap();
    let quotient = p
        .add_integer_var("quotient", 1.0, Some(-8.0), Some(8.0))
        .unwrap();
    let remainder = p
        .add_integer_var("remainder", 1.0, Some(-4.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-numerator", vec![(numerator, 1.0)], RowSense::Eq, -7.0)
        .unwrap();
    p.add_constraint(
        "fix-denominator",
        vec![(denominator, 1.0)],
        RowSense::Eq,
        3.0,
    )
    .unwrap();
    p.add_division_equality("division", quotient, numerator, denominator)
        .unwrap();
    p.add_modulo_equality("modulo", remainder, numerator, denominator)
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

fn build_integer_abs_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(-5.0), Some(4.0)).unwrap();
    let r = p
        .add_integer_var("abs_x", 1.0, Some(0.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-x-integer", vec![(x, 1.0)], RowSense::Eq, -3.0)
        .unwrap();
    p.add_abs("integer-absolute-value", r, x).unwrap();
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

fn build_integer_max_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a = p.add_integer_var("a", 0.0, Some(-2.0), Some(5.0)).unwrap();
    let b = p.add_integer_var("b", 0.0, Some(-2.0), Some(5.0)).unwrap();
    let r = p
        .add_integer_var("max_ab", 1.0, Some(-2.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-a-integer", vec![(a, 1.0)], RowSense::Eq, 2.0)
        .unwrap();
    p.add_constraint("fix-b-integer", vec![(b, 1.0)], RowSense::Eq, -1.0)
        .unwrap();
    p.add_max("integer-maximum", r, vec![a, b]).unwrap();
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

fn build_integer_min_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a = p.add_integer_var("a", 0.0, Some(-2.0), Some(5.0)).unwrap();
    let b = p.add_integer_var("b", 0.0, Some(-2.0), Some(5.0)).unwrap();
    let r = p
        .add_integer_var("min_ab", 1.0, Some(-2.0), Some(5.0))
        .unwrap();
    p.add_constraint("fix-a-integer", vec![(a, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_constraint("fix-b-integer", vec![(b, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_min("integer-minimum", r, vec![a, b]).unwrap();
    p
}

fn build_l1_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(-4.0), Some(4.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 0.0, Some(-4.0), Some(4.0))
        .unwrap();
    let norm = p
        .add_continuous_var("norm", 1.0, Some(0.0), Some(8.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -2.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_l1_norm("l1", norm, vec![x, y]).unwrap();
    p
}

fn build_integer_l1_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(-4.0), Some(4.0)).unwrap();
    let y = p.add_integer_var("y", 0.0, Some(-4.0), Some(4.0)).unwrap();
    let norm = p
        .add_integer_var("norm", 1.0, Some(0.0), Some(8.0))
        .unwrap();
    p.add_constraint("fix-x-integer", vec![(x, 1.0)], RowSense::Eq, -2.0)
        .unwrap();
    p.add_constraint("fix-y-integer", vec![(y, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_l1_norm("integer-l1", norm, vec![x, y]).unwrap();
    p
}

fn build_l_infinity_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(-4.0), Some(4.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 0.0, Some(-4.0), Some(4.0))
        .unwrap();
    let norm = p
        .add_continuous_var("norm", 1.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -2.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_l_infinity_norm("linf", norm, vec![x, y]).unwrap();
    p
}

fn build_integer_l_infinity_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(-4.0), Some(4.0)).unwrap();
    let y = p.add_integer_var("y", 0.0, Some(-4.0), Some(4.0)).unwrap();
    let norm = p
        .add_integer_var("norm", 1.0, Some(0.0), Some(4.0))
        .unwrap();
    p.add_constraint("fix-x-integer", vec![(x, 1.0)], RowSense::Eq, -2.0)
        .unwrap();
    p.add_constraint("fix-y-integer", vec![(y, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_l_infinity_norm("integer-linf", norm, vec![x, y])
        .unwrap();
    p
}

fn build_l2_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p
        .add_continuous_var("x", 0.0, Some(-5.0), Some(5.0))
        .unwrap();
    let y = p
        .add_continuous_var("y", 0.0, Some(-5.0), Some(5.0))
        .unwrap();
    let norm = p
        .add_continuous_var("norm", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_l2_norm("l2", norm, vec![x, y]).unwrap();
    p
}

fn build_integer_l2_norm_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let x = p.add_integer_var("x", 0.0, Some(-5.0), Some(5.0)).unwrap();
    let y = p.add_integer_var("y", 0.0, Some(-5.0), Some(5.0)).unwrap();
    let norm = p
        .add_integer_var("norm", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    p.add_constraint("fix-x-integer", vec![(x, 1.0)], RowSense::Eq, 3.0)
        .unwrap();
    p.add_constraint("fix-y-integer", vec![(y, 1.0)], RowSense::Eq, 4.0)
        .unwrap();
    p.add_l2_norm("integer-l2", norm, vec![x, y]).unwrap();
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

fn build_forbidden_assignments_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x = p.add_integer_var("x", 10.0, Some(0.0), Some(2.0)).unwrap();
    let y = p.add_integer_var("y", 1.0, Some(0.0), Some(2.0)).unwrap();
    p.add_forbidden_assignments("forbidden-pairs", vec![x, y], vec![vec![2, 2]])
        .unwrap();
    p
}

fn build_bin_packing_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let item0 = p
        .add_integer_var("item0_bin", 0.0, Some(0.0), Some(1.0))
        .unwrap();
    let item1 = p
        .add_integer_var("item1_bin", 0.0, Some(0.0), Some(1.0))
        .unwrap();
    let item2 = p
        .add_integer_var("item2_bin", 0.0, Some(0.0), Some(1.0))
        .unwrap();
    let load0 = p
        .add_integer_var("load0", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let load1 = p
        .add_integer_var("load1", 1.0, Some(0.0), Some(9.0))
        .unwrap();
    p.add_bin_packing(
        "packing",
        vec![item0, item1, item2],
        vec![load0, load1],
        vec![2.0, 3.0, 4.0],
    )
    .unwrap();
    p
}

fn build_element_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let index = p
        .add_integer_var("index", 0.0, Some(0.0), Some(3.0))
        .unwrap();
    let picked = p
        .add_integer_var("picked", 1.0, Some(0.0), Some(9.0))
        .unwrap();
    p.add_element("lookup", index, picked, vec![1.0, 7.0, 4.0, 9.0])
        .unwrap();
    p
}

fn build_variable_element_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let index = p
        .add_integer_var("index", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let a = p.add_integer_var("a", 0.0, Some(2.0), Some(2.0)).unwrap();
    let b = p.add_integer_var("b", 0.0, Some(8.0), Some(8.0)).unwrap();
    let c = p.add_integer_var("c", 0.0, Some(5.0), Some(5.0)).unwrap();
    let picked = p
        .add_integer_var("picked", 1.0, Some(0.0), Some(10.0))
        .unwrap();
    p.add_variable_element("variable-lookup", index, picked, vec![a, b, c])
        .unwrap();
    p
}

fn build_inverse_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p.add_integer_var("x0", 0.0, Some(0.0), Some(2.0)).unwrap();
    let x1 = p.add_integer_var("x1", 1.0, Some(0.0), Some(2.0)).unwrap();
    let x2 = p.add_integer_var("x2", 0.0, Some(0.0), Some(2.0)).unwrap();
    let y0 = p.add_integer_var("y0", 0.0, Some(0.0), Some(2.0)).unwrap();
    let y1 = p.add_integer_var("y1", 0.0, Some(0.0), Some(2.0)).unwrap();
    let y2 = p.add_integer_var("y2", 0.0, Some(0.0), Some(2.0)).unwrap();
    p.add_constraint("force-x0", vec![(x0, 1.0)], RowSense::Eq, 1.0)
        .unwrap();
    p.add_inverse("inverse-permutation", vec![x0, x1, x2], vec![y0, y1, y2])
        .unwrap();
    p
}

fn build_circuit_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let mut arcs = Vec::new();
    for tail in 0..4 {
        for head in 0..4 {
            if tail == head {
                continue;
            }
            let obj = match (tail, head) {
                (0, 1) | (1, 2) | (2, 3) | (3, 0) => 10.0,
                _ => 0.0,
            };
            let var = p.add_binary_var(format!("x_{tail}_{head}"), obj).unwrap();
            arcs.push((tail, head, var));
        }
    }
    p.add_circuit("tour", 4, arcs).unwrap();
    p
}

fn build_multiple_circuit_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let depot_to_first = p.add_binary_var("x_0_1", 1.0).unwrap();
    let first_to_second = p.add_binary_var("x_1_2", 10.0).unwrap();
    let second_to_depot = p.add_binary_var("x_2_0", 1.0).unwrap();
    let second_to_first = p.add_binary_var("x_2_1", 10.0).unwrap();
    let skipped = p.add_binary_var("x_3_3", 0.0).unwrap();
    p.add_multiple_circuit(
        "routes",
        4,
        vec![
            (0, 1, depot_to_first),
            (1, 2, first_to_second),
            (2, 0, second_to_depot),
            (2, 1, second_to_first),
            (3, 3, skipped),
        ],
    )
    .unwrap();
    p
}

fn build_automaton_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let x0 = p.add_binary_var("x0", 4.0).unwrap();
    let x1 = p.add_binary_var("x1", 3.0).unwrap();
    let x2 = p.add_binary_var("x2", 2.0).unwrap();
    p.add_automaton(
        "no-consecutive-ones",
        vec![x0, x1, x2],
        0,
        vec![0, 1],
        vec![(0, 0, 0), (0, 1, 1), (1, 0, 0)],
    )
    .unwrap();
    p
}

fn build_alternative_interval_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let start = p
        .add_integer_var("task_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let size = p
        .add_integer_var("task_size", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let end = p
        .add_integer_var("task_end", 1.0, Some(0.0), Some(5.0))
        .unwrap();
    let slow_start = p
        .add_integer_var("slow_start", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let slow_end = p
        .add_integer_var("slow_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let fast_start = p
        .add_integer_var("fast_start", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let fast_end = p
        .add_integer_var("fast_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let slow_present = p.add_binary_var("slow_present", 0.0).unwrap();
    let fast_present = p.add_binary_var("fast_present", 0.0).unwrap();
    p.add_alternative(
        "choose-mode",
        MathProgram::variable_interval(start, size, end),
        vec![
            MathProgram::optional_interval(slow_start, 4.0, slow_end, slow_present),
            MathProgram::optional_interval(fast_start, 2.0, fast_end, fast_present),
        ],
    )
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

fn build_variable_no_overlap_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let a_size = p
        .add_integer_var("a_size", 0.0, Some(2.0), Some(4.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(0.0), Some(8.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 1.0, Some(0.0), Some(8.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(8.0))
        .unwrap();
    p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
        .unwrap();
    p.add_constraint(
        "force-a-size-through-end",
        vec![(a_end, 1.0)],
        RowSense::Ge,
        4.0,
    )
    .unwrap();
    p.add_no_overlap(
        "variable-single-machine",
        vec![
            MathProgram::variable_interval(a_start, a_size, a_end),
            MathProgram::interval(b_start, 2.0, b_end),
        ],
    )
    .unwrap();
    p
}

fn build_optional_no_overlap_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(3.0), Some(3.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 0.0, Some(1.0), Some(1.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_present = p.add_binary_var("b_present", 1.0).unwrap();
    p.add_no_overlap(
        "optional-single-machine",
        vec![
            MathProgram::interval(a_start, 3.0, a_end),
            MathProgram::optional_interval(b_start, 2.0, b_end, b_present),
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

fn build_variable_no_overlap_2d_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_x_start = p
        .add_integer_var("a_x_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_x_size = p
        .add_integer_var("a_x_size", 0.0, Some(2.0), Some(2.0))
        .unwrap();
    let a_x_end = p
        .add_integer_var("a_x_end", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let a_y_start = p
        .add_integer_var("a_y_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_y_size = p
        .add_integer_var("a_y_size", 0.0, Some(2.0), Some(3.0))
        .unwrap();
    let a_y_end = p
        .add_integer_var("a_y_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    let b_x_start = p
        .add_integer_var("b_x_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_x_end = p
        .add_integer_var("b_x_end", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_y_start = p
        .add_integer_var("b_y_start", 1.0, Some(0.0), Some(5.0))
        .unwrap();
    let b_y_end = p
        .add_integer_var("b_y_end", 0.0, Some(0.0), Some(7.0))
        .unwrap();

    p.add_constraint(
        "force-a-y-size-through-end",
        vec![(a_y_end, 1.0)],
        RowSense::Ge,
        3.0,
    )
    .unwrap();
    p.add_no_overlap_2d(
        "variable-packing",
        vec![
            MathProgram::variable_interval(a_x_start, a_x_size, a_x_end),
            MathProgram::interval(b_x_start, 2.0, b_x_end),
        ],
        vec![
            MathProgram::variable_interval(a_y_start, a_y_size, a_y_end),
            MathProgram::interval(b_y_start, 2.0, b_y_end),
        ],
    )
    .unwrap();
    p
}

fn build_optional_no_overlap_2d_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a_x_start = p
        .add_integer_var("a_x_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_x_end = p
        .add_integer_var("a_x_end", 0.0, Some(2.0), Some(2.0))
        .unwrap();
    let a_y_start = p
        .add_integer_var("a_y_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_y_end = p
        .add_integer_var("a_y_end", 0.0, Some(2.0), Some(2.0))
        .unwrap();
    let b_x_start = p
        .add_integer_var("b_x_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_x_end = p
        .add_integer_var("b_x_end", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_y_start = p
        .add_integer_var("b_y_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_y_end = p
        .add_integer_var("b_y_end", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_present = p.add_binary_var("b_present", 1.0).unwrap();

    p.add_no_overlap_2d(
        "optional-packing",
        vec![
            MathProgram::interval(a_x_start, 2.0, a_x_end),
            MathProgram::optional_interval(b_x_start, 2.0, b_x_end, b_present),
        ],
        vec![
            MathProgram::interval(a_y_start, 2.0, a_y_end),
            MathProgram::optional_interval(b_y_start, 2.0, b_y_end, b_present),
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

fn build_variable_cumulative_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_size = p
        .add_integer_var("a_size", 0.0, Some(1.0), Some(3.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(0.0), Some(3.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 1.0, Some(0.0), Some(3.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(5.0))
        .unwrap();
    p.add_constraint("force-a-size", vec![(a_end, 1.0)], RowSense::Ge, 3.0)
        .unwrap();
    p.add_cumulative(
        "shared-resource",
        vec![
            MathProgram::variable_interval(a_start, a_size, a_end),
            MathProgram::interval(b_start, 2.0, b_end),
        ],
        vec![2.0, 2.0],
        3.0,
    )
    .unwrap();
    p
}

fn build_affine_cumulative_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Min);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 1.0, Some(0.0), Some(2.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(4.0))
        .unwrap();
    let a_demand = p
        .add_integer_var("a_demand", 0.0, Some(1.0), Some(2.0))
        .unwrap();
    let capacity = p
        .add_integer_var("capacity", 0.0, Some(3.0), Some(4.0))
        .unwrap();
    p.add_constraint("force-a-demand", vec![(a_demand, 1.0)], RowSense::Ge, 2.0)
        .unwrap();
    p.add_constraint("force-capacity", vec![(capacity, 1.0)], RowSense::Le, 3.0)
        .unwrap();
    p.add_cumulative_affine(
        "shared-resource",
        vec![
            MathProgram::interval(a_start, 2.0, a_end),
            MathProgram::interval(b_start, 2.0, b_end),
        ],
        vec![
            AffineTerm {
                coeffs: vec![(a_demand, 1.0)],
                constant: 0.0,
            },
            AffineTerm {
                coeffs: Vec::new(),
                constant: 2.0,
            },
        ],
        AffineTerm {
            coeffs: vec![(capacity, 1.0)],
            constant: 0.0,
        },
    )
    .unwrap();
    p
}

fn build_optional_cumulative_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let a_start = p
        .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let a_end = p
        .add_integer_var("a_end", 0.0, Some(2.0), Some(2.0))
        .unwrap();
    let b_start = p
        .add_integer_var("b_start", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_end = p
        .add_integer_var("b_end", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let b_present = p.add_binary_var("b_present", 1.0).unwrap();
    p.add_cumulative(
        "optional-cumulative",
        vec![
            MathProgram::interval(a_start, 2.0, a_end),
            MathProgram::optional_interval(b_start, 2.0, b_end, b_present),
        ],
        vec![3.0, 2.0],
        3.0,
    )
    .unwrap();
    p
}

fn build_reservoir_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let supply_time = p
        .add_integer_var("supply_time", 1.0, Some(0.0), Some(2.0))
        .unwrap();
    let drain_time = p
        .add_integer_var("drain_time", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    p.add_reservoir(
        "tank",
        vec![
            MathProgram::reservoir_event(supply_time, 2.0),
            MathProgram::reservoir_event(drain_time, -2.0),
        ],
        0.0,
        2.0,
    )
    .unwrap();
    p
}

fn build_optional_reservoir_case() -> MathProgram {
    let mut p = MathProgram::new(ObjectiveSense::Max);
    let surge_time = p
        .add_integer_var("surge_time", 0.0, Some(0.0), Some(0.0))
        .unwrap();
    let surge_active = p.add_binary_var("surge_active", 1.0).unwrap();
    p.add_reservoir(
        "optional-reservoir",
        vec![MathProgram::optional_reservoir_event(
            surge_time,
            3.0,
            surge_active,
        )],
        0.0,
        2.0,
    )
    .unwrap();
    p
}

#[cfg(test)]
mod tests {
    #[test]
    fn assumption_core_external_case_runs() {
        assert!(super::run_assumption_core_case().unwrap());
    }

    #[test]
    fn continuous_lp_matrix_includes_ortools_pdlp() {
        let lp = super::build_lp_case();
        let methods = super::external_methods_for_case("lp-row-senses", &lp);
        assert_eq!(methods.first().map(|(label, _)| *label), Some("highs-cli"));
        assert!(methods
            .iter()
            .any(|(label, method)| *label == "ortools-pdlp"
                && method.as_deref() == Some("ortools:PDLP")));
        assert_eq!(methods.last().map(|(label, _)| *label), Some("scipy-highs"));
    }

    #[test]
    fn linear_external_matrices_keep_scipy_as_compatibility_oracle() {
        let mip = super::build_binary_mip_case();
        let mip_methods = super::external_methods_for_case("binary-mip", &mip);
        assert_eq!(
            mip_methods.first().map(|(label, _)| *label),
            Some("highs-cli")
        );
        assert!(mip_methods
            .iter()
            .any(|(label, _)| *label == "ortools-cp-sat"));
        let scipy_position = mip_methods
            .iter()
            .position(|(label, _)| *label == "scipy-highs")
            .expect("scipy compatibility oracle");
        let cp_sat_position = mip_methods
            .iter()
            .position(|(label, _)| *label == "ortools-cp-sat")
            .expect("cp-sat oracle");
        assert!(scipy_position > cp_sat_position);

        let lp_methods = super::continuous_linear_methods("ortools:GLOP");
        assert_eq!(
            lp_methods.first().map(|(label, _)| *label),
            Some("highs-cli")
        );
        assert_eq!(
            lp_methods.last().map(|(label, _)| *label),
            Some("scipy-highs")
        );
    }

    #[test]
    fn non_lp_matrices_do_not_use_ortools_pdlp() {
        let mip = super::build_binary_mip_case();
        let mip_methods = super::external_methods_for_case("binary-mip", &mip);
        assert!(!mip_methods
            .iter()
            .any(|(label, _)| *label == "ortools-pdlp"));

        let qp = super::build_continuous_qp_case();
        let qp_methods = super::external_methods_for_case("continuous-qp", &qp);
        assert!(!qp_methods.iter().any(|(label, _)| *label == "ortools-pdlp"));
    }
}
