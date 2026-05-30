//! Port of src/des/test/feasibility-pipeline-test.ts
//!
//! Tests for the structured-optimization feasibility checker pipeline
//! (`general/feasibility-pipeline`): direct candidate evaluation, the
//! check-and-improve DES pipeline on binary and continuous problems, and the
//! wall-clock time limit.
//!
//! PORT NOTE: TS group [4] ("Registry, JSON input, logs, and animation") relies
//! on `general/des-registry` (`getModel`, `runFromSpec`, `runFromJsonFile`),
//! JSON file I/O, and the animation frame writer, none of which are ported to
//! Rust yet, so group [4] is deferred. Groups [1]–[3] call `evaluate_candidate`
//! / `run_feasibility_pipeline` directly and are ported faithfully.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::feasibility_pipeline::{
        evaluate_candidate, run_feasibility_pipeline, CandidateSolutionInput, ConstraintSense,
        FeasibilityImprovementOptions, FeasibilityPipelineParams, FeasibilityStatus,
        LinearConstraint, LinearObjective, ObjectiveSense, OptimizationVariable,
        StructuredOptimizationProblem, VariableKind,
    };
    use std::collections::HashMap;

    const PENALTY: f64 = 1_000_000.0;

    fn vals(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn candidate(id: Option<&str>, pairs: &[(&str, f64)]) -> CandidateSolutionInput {
        CandidateSolutionInput {
            id: id.map(|s| s.to_string()),
            values: Some(vals(pairs)),
            vector: None,
        }
    }

    fn coeffs(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn binary(name: &str) -> OptimizationVariable {
        OptimizationVariable {
            name: name.to_string(),
            kind: Some(VariableKind::Binary),
            lb: None,
            ub: None,
            step: None,
        }
    }

    fn knapsack() -> StructuredOptimizationProblem {
        StructuredOptimizationProblem {
            sense: ObjectiveSense::Max,
            variables: vec![binary("x0"), binary("x1"), binary("x2")],
            objective: LinearObjective {
                constant: None,
                coefficients: coeffs(&[("x0", 60.0), ("x1", 100.0), ("x2", 120.0)]),
            },
            constraints: Some(vec![LinearConstraint {
                name: Some("capacity".to_string()),
                coefficients: coeffs(&[("x0", 10.0), ("x1", 20.0), ("x2", 30.0)]),
                sense: ConstraintSense::Le,
                rhs: 50.0,
                tolerance: None,
            }]),
            tolerance: Some(1e-8),
        }
    }

    fn production() -> StructuredOptimizationProblem {
        StructuredOptimizationProblem {
            sense: ObjectiveSense::Min,
            variables: vec![
                OptimizationVariable {
                    name: "regular".to_string(),
                    kind: Some(VariableKind::Continuous),
                    lb: Some(0.0),
                    ub: Some(100.0),
                    step: Some(5.0),
                },
                OptimizationVariable {
                    name: "overtime".to_string(),
                    kind: Some(VariableKind::Continuous),
                    lb: Some(0.0),
                    ub: Some(50.0),
                    step: Some(5.0),
                },
            ],
            objective: LinearObjective {
                constant: None,
                coefficients: coeffs(&[("regular", 4.0), ("overtime", 7.0)]),
            },
            constraints: Some(vec![
                LinearConstraint {
                    name: Some("demand".to_string()),
                    coefficients: coeffs(&[("regular", 1.0), ("overtime", 1.0)]),
                    sense: ConstraintSense::Ge,
                    rhs: 80.0,
                    tolerance: None,
                },
                LinearConstraint {
                    name: Some("regular-cap".to_string()),
                    coefficients: coeffs(&[("regular", 1.0)]),
                    sense: ConstraintSense::Le,
                    rhs: 60.0,
                    tolerance: None,
                },
            ]),
            tolerance: None,
        }
    }

    // [1] Direct feasibility evaluation.
    #[test]
    fn direct_feasibility_evaluation() {
        let ok = evaluate_candidate(
            &knapsack(),
            &candidate(None, &[("x0", 1.0), ("x1", 1.0), ("x2", 0.0)]),
            PENALTY,
        );
        assert!(
            ok.feasible && ok.objective_value == 160.0,
            "objective={}",
            ok.objective_value
        );

        let bad = evaluate_candidate(
            &knapsack(),
            &candidate(None, &[("x0", 1.0), ("x1", 1.0), ("x2", 1.25)]),
            PENALTY,
        );
        assert!(
            !bad.feasible
                && bad.domain_violations.len() >= 1
                && bad.constraint_violations.len() == 1,
            "domain={} constraints={}",
            bad.domain_violations.len(),
            bad.constraint_violations.len()
        );
    }

    #[test]
    #[should_panic(expected = "reference a declared variable")]
    fn unknown_objective_coefficient_rejected() {
        let mut p = knapsack();
        p.objective = LinearObjective {
            constant: None,
            coefficients: coeffs(&[("x0", 1.0), ("missing", 2.0)]),
        };
        let _ = evaluate_candidate(
            &p,
            &candidate(None, &[("x0", 1.0), ("x1", 0.0), ("x2", 0.0)]),
            PENALTY,
        );
    }

    // [2] Pipeline checks and improves candidate solutions.
    #[test]
    fn pipeline_improves_binary_incumbent() {
        let r = run_feasibility_pipeline(FeasibilityPipelineParams {
            problem: knapsack(),
            candidate: candidate(Some("user-best"), &[("x0", 1.0), ("x1", 1.0), ("x2", 0.0)]),
            improvement: Some(FeasibilityImprovementOptions {
                enabled: Some(true),
                max_iterations: Some(80),
                seed: Some(4),
                integer_step: Some(1.0),
                ..Default::default()
            }),
            time_limit_ms: None,
            max_ticks: None,
            check_every_ticks: None,
        });
        assert_eq!(r.status, FeasibilityStatus::Improved);
        assert_eq!(r.best.objective_value, 220.0);
        assert!(r.best.feasible && r.best.total_violation == 0.0);
        assert!(r.network.stationary_entities.len() == 7 && r.network.edges.len() == 7);
        assert!(
            r.validation.iter().all(|c| c.passed),
            "{}",
            r.validation
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn checker_only_reports_infeasible() {
        let check_only = run_feasibility_pipeline(FeasibilityPipelineParams {
            problem: knapsack(),
            candidate: candidate(None, &[("x0", 1.0), ("x1", 1.0), ("x2", 1.0)]),
            improvement: Some(FeasibilityImprovementOptions {
                enabled: Some(false),
                ..Default::default()
            }),
            time_limit_ms: None,
            max_ticks: None,
            check_every_ticks: None,
        });
        assert_eq!(check_only.status, FeasibilityStatus::Infeasible);
        assert!(!check_only.initial.feasible);
    }

    // [3] Continuous problem improvement and time limits.
    #[test]
    fn continuous_incumbent_improves() {
        let prod = run_feasibility_pipeline(FeasibilityPipelineParams {
            problem: production(),
            candidate: candidate(None, &[("regular", 50.0), ("overtime", 40.0)]),
            improvement: Some(FeasibilityImprovementOptions {
                enabled: Some(true),
                max_iterations: Some(60),
                seed: Some(9),
                continuous_step: Some(5.0),
                ..Default::default()
            }),
            time_limit_ms: None,
            max_ticks: None,
            check_every_ticks: None,
        });
        assert_eq!(prod.status, FeasibilityStatus::Improved);
        assert!(
            prod.best.objective_value < prod.initial.objective_value,
            "initial={} best={}",
            prod.initial.objective_value,
            prod.best.objective_value
        );
        assert!(prod.best.feasible);
    }

    #[test]
    fn zero_wall_clock_budget_stops() {
        let timed = run_feasibility_pipeline(FeasibilityPipelineParams {
            problem: production(),
            candidate: candidate(None, &[("regular", 50.0), ("overtime", 40.0)]),
            improvement: Some(FeasibilityImprovementOptions {
                enabled: Some(true),
                max_iterations: Some(1000),
                seed: Some(1),
                ..Default::default()
            }),
            time_limit_ms: Some(0.0),
            max_ticks: None,
            check_every_ticks: None,
        });
        assert_eq!(timed.status, FeasibilityStatus::TimeLimit);
        assert!(timed.stop_signals.len() >= 1);
    }
}
