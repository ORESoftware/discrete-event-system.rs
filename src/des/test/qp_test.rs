//! Tests for the small dense convex QP solver and reference bridge.

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::{Command, Stdio};

    use serde::Deserialize;

    use crate::des::general::qp::{
        solve_qcp_pattern_search, solve_qp_active_set, solve_socp_pattern_search, QPOptions,
        QPStatus, QcpOptions, QcpStatus, QuadraticConstraint, QuadraticProgram,
        QuadraticallyConstrainedProgram, SecondOrderCone, SecondOrderConeProgram, SocpOptions,
        SocpStatus,
    };

    #[derive(Debug, Deserialize)]
    struct QPReference {
        status: String,
        x: Vec<f64>,
        objective: Option<f64>,
        solver: String,
    }

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

    fn qp_json(qp: &QuadraticProgram) -> String {
        serde_json::json!({
            "Q": qp.q,
            "c": qp.c,
            "A_ub": qp.a_ub,
            "b_ub": qp.b_ub,
            "A_eq": qp.a_eq,
            "b_eq": qp.b_eq,
            "lb": qp.lb,
            "ub": qp.ub,
        })
        .to_string()
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

    fn socp_json(socp: &SecondOrderConeProgram) -> String {
        serde_json::json!({
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
        .to_string()
    }

    fn qcp_json(qcp: &QuadraticallyConstrainedProgram) -> String {
        serde_json::json!({
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
        .to_string()
    }

    fn run_reference_json(input: String) -> QPReference {
        let root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let script = format!("{root}/scripts/qp_reference.py");
        let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let mut child = Command::new(python)
            .arg(script)
            .arg("--solver")
            .arg("auto")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start QP reference bridge");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(input.as_bytes())
            .expect("write optimization JSON");
        let out = child.wait_with_output().expect("wait for QP reference");
        assert!(
            out.status.success(),
            "reference failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).expect("parse QP reference JSON")
    }

    fn run_reference(qp: &QuadraticProgram) -> QPReference {
        run_reference_json(qp_json(qp))
    }

    fn run_socp_reference(socp: &SecondOrderConeProgram) -> QPReference {
        run_reference_json(socp_json(socp))
    }

    fn run_qcp_reference(qcp: &QuadraticallyConstrainedProgram) -> QPReference {
        run_reference_json(qcp_json(qcp))
    }

    #[test]
    fn convex_qp_matches_reference_bridge() {
        let qp = sample_qp();
        let internal = solve_qp_active_set(&qp, QPOptions::default());
        assert_eq!(internal.status, QPStatus::Optimal);
        assert!((internal.x[0] - 7.0 / 6.0).abs() < 1e-8, "{internal:?}");
        assert!((internal.x[1] - 11.0 / 6.0).abs() < 1e-8, "{internal:?}");

        let reference = run_reference(&qp);
        assert_eq!(reference.status, "optimal", "{reference:?}");
        assert!((reference.objective.unwrap() - internal.objective).abs() < 1e-8);
        assert!(
            reference
                .x
                .iter()
                .zip(&internal.x)
                .all(|(a, b)| (a - b).abs() < 1e-8),
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
        assert_eq!(reference.status, "optimal", "{reference:?}");
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
        assert_eq!(reference.status, "optimal", "{reference:?}");
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
