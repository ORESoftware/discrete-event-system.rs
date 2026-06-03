//! Tests for the small dense convex QP solver and reference bridge.

#[cfg(test)]
mod tests {
    use crate::des::general::external_quadratic_reference::{
        solve_qcp_with_external_reference, solve_qp_with_external_reference,
        solve_socp_with_external_reference, ExternalQuadraticReferenceOptions,
        ExternalQuadraticReferenceSolution, ExternalQuadraticReferenceSolver,
        ExternalQuadraticReferenceStatus,
    };
    use crate::des::general::qp::{
        solve_qcp_pattern_search, solve_qp_active_set, solve_socp_pattern_search, QPOptions,
        QPStatus, QcpOptions, QcpStatus, QuadraticConstraint, QuadraticProgram,
        QuadraticallyConstrainedProgram, SecondOrderCone, SecondOrderConeProgram, SocpOptions,
        SocpStatus,
    };

    fn sample_qp() -> QuadraticProgram {
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

    fn sample_socp() -> SecondOrderConeProgram {
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

    fn sample_qcp() -> QuadraticallyConstrainedProgram {
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

    fn rust_reference_options() -> ExternalQuadraticReferenceOptions {
        ExternalQuadraticReferenceOptions {
            solver: ExternalQuadraticReferenceSolver::RustInternal,
            ..Default::default()
        }
    }

    fn run_reference(qp: &QuadraticProgram) -> ExternalQuadraticReferenceSolution {
        solve_qp_with_external_reference(qp, &rust_reference_options())
    }

    fn run_socp_reference(socp: &SecondOrderConeProgram) -> ExternalQuadraticReferenceSolution {
        solve_socp_with_external_reference(socp, &rust_reference_options())
    }

    fn run_qcp_reference(
        qcp: &QuadraticallyConstrainedProgram,
    ) -> ExternalQuadraticReferenceSolution {
        solve_qcp_with_external_reference(qcp, &rust_reference_options())
    }

    #[test]
    fn convex_qp_matches_reference_bridge() {
        let qp = sample_qp();
        let internal = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(internal.status, QPStatus::Optimal);
        assert!((internal.x[0] - 7.0 / 6.0).abs() < 1e-8, "{internal:?}");
        assert!((internal.x[1] - 11.0 / 6.0).abs() < 1e-8, "{internal:?}");

        let reference = run_reference(&qp);
        assert_eq!(
            reference.status,
            ExternalQuadraticReferenceStatus::Optimal,
            "{reference:?}"
        );
        assert!((reference.objective.unwrap() - internal.objective).abs() < 1e-8);
        let reference_x_tol = 1e-7;
        assert!(
            reference
                .x
                .iter()
                .zip(&internal.x)
                .all(|(a, b)| (a - b).abs() < reference_x_tol),
            "reference={reference:?} internal={internal:?}"
        );
        assert!(!reference.solver.is_empty());
    }

    #[test]
    fn socp_matches_reference_bridge() {
        let socp = sample_socp();
        let internal = solve_socp_pattern_search(&socp, SocpOptions::default());
        assert_eq!(internal.status, SocpStatus::Optimal);
        assert!((internal.objective + 1.0).abs() < 1e-6, "{internal:?}");

        let reference = run_socp_reference(&socp);
        assert_eq!(
            reference.status,
            ExternalQuadraticReferenceStatus::Optimal,
            "{reference:?}"
        );
        assert!((reference.objective.unwrap() - internal.objective).abs() < 1e-6);
        assert!(
            reference
                .x
                .iter()
                .zip(&internal.x)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "reference={reference:?} internal={internal:?}"
        );
        assert!(!reference.solver.is_empty());
    }

    #[test]
    fn qcp_matches_reference_bridge() {
        let qcp = sample_qcp();
        let internal = solve_qcp_pattern_search(&qcp, QcpOptions::default());
        assert_eq!(internal.status, QcpStatus::Optimal);
        assert!((internal.objective + 1.0).abs() < 1e-6, "{internal:?}");

        let reference = run_qcp_reference(&qcp);
        assert_eq!(
            reference.status,
            ExternalQuadraticReferenceStatus::Optimal,
            "{reference:?}"
        );
        assert!((reference.objective.unwrap() - internal.objective).abs() < 1e-6);
        assert!(
            reference
                .x
                .iter()
                .zip(&internal.x)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "reference={reference:?} internal={internal:?}"
        );
        assert!(!reference.solver.is_empty());
    }
}
