//! Port of src/des/test/classical-optimization-test.ts
//!
//! Tests for the classical optimization station-graph models
//! (`general/classical-optimization-models`): projected-gradient and
//! coordinate-descent QP, Hungarian and auction assignment, Clarke-Wright /
//! nearest-neighbour VRP, and job-shop / flow-shop scheduling.
//!
//! PORT NOTE: the TS "registry smoke" section uses `general/des-registry`
//! (`getModel`, `runFromSpec`), which is not yet ported to Rust; it is deferred.
//! Every `run*` model call is ported faithfully.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::classical_optimization_models::{
        run_auction_assignment, run_flow_shop_neh, run_hungarian_assignment,
        run_job_shop_dispatch, run_qp_coordinate_descent, run_qp_projected_gradient,
        run_vrp_nearest_neighbor, run_vrp_savings, AssignmentParams, AuctionAssignmentParams,
        DispatchRule, FlowShopNEHParams, JobShopDispatchParams, QPProjectedGradientParams,
        ScheduledOperation, VRPSavingsParams,
    };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-5 * f64::max(1.0, f64::max(a.abs(), b.abs()))
    }

    fn has(movables: &[String], name: &str) -> bool {
        movables.iter().any(|m| m == name)
    }

    fn respects_precedence(schedule: &[ScheduledOperation]) -> bool {
        for op in schedule {
            if op.op_index == 0 {
                continue;
            }
            match schedule
                .iter()
                .find(|o| o.job_id == op.job_id && o.op_index == op.op_index - 1)
            {
                Some(prev) if prev.finish <= op.start => {}
                _ => return false,
            }
        }
        true
    }

    fn no_machine_overlap(schedule: &[ScheduledOperation]) -> bool {
        let mut machines: Vec<String> = schedule.iter().map(|o| o.machine.clone()).collect();
        machines.sort();
        machines.dedup();
        for machine in machines {
            let mut ops: Vec<&ScheduledOperation> =
                schedule.iter().filter(|o| o.machine == machine).collect();
            ops.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
            for i in 1..ops.len() {
                if ops[i - 1].finish > ops[i].start {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn qp_projected_gradient() {
        let r = run_qp_projected_gradient(QPProjectedGradientParams::default());
        assert!(close(r.x[0], 10.0 / 7.0), "x0={}", r.x[0]);
        assert!(close(r.x[1], 16.0 / 7.0), "x1={}", r.x[1]);
        assert!(r.gradient_norm < 1e-6, "norm={}", r.gradient_norm);
        assert!(has(&r.topology.movables, "QPStateToken"));
    }

    #[test]
    #[should_panic]
    fn qp_source_rejects_invalid_initial_state() {
        let _ = run_qp_projected_gradient(QPProjectedGradientParams {
            x0: Some(vec![11.0, 0.0]),
            ..Default::default()
        });
    }

    #[test]
    fn qp_coordinate_descent() {
        let r = run_qp_coordinate_descent(QPProjectedGradientParams::default());
        assert!(close(r.x[0], 10.0 / 7.0), "x0={}", r.x[0]);
        assert!(close(r.x[1], 16.0 / 7.0), "x1={}", r.x[1]);
        assert!(r.gradient_norm < 1e-6, "norm={}", r.gradient_norm);
        assert!(has(&r.topology.movables, "QPStateToken"));
    }

    #[test]
    fn hungarian_assignment() {
        let r = run_hungarian_assignment(AssignmentParams::default());
        assert_eq!(r.objective, 9.0);
        assert_eq!(r.assignment.len(), 3);
        let mut uniq = r.assignment.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), r.assignment.len());
        assert!(has(&r.topology.movables, "AssignmentMatrixToken"));
    }

    #[test]
    fn auction_assignment() {
        let r = run_auction_assignment(AuctionAssignmentParams::default());
        assert_eq!(r.objective, 9.0);
        assert_eq!(r.assignment.len(), 3);
        let mut uniq = r.assignment.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), r.assignment.len());
        assert!(has(&r.topology.movables, "AssignmentAuctionStateToken"));
    }

    #[test]
    fn vrp_savings() {
        let r = run_vrp_savings(VRPSavingsParams::default());
        assert_eq!(r.routes.len(), 2);
        assert!(r.routes.iter().all(|route| route.load <= 5.0));
        assert!(r.total_distance > 0.0);
        assert!(has(&r.topology.movables, "VRPSavingsToken"));
    }

    #[test]
    fn vrp_nearest_neighbor() {
        let r = run_vrp_nearest_neighbor(VRPSavingsParams::default());
        assert_eq!(r.routes.len(), 2);
        assert!(r.routes.iter().all(|route| route.load <= 5.0));
        assert!(r.total_distance > 0.0);
        assert!(has(&r.topology.movables, "VRPProblemToken"));
        assert!(has(&r.topology.movables, "VRPResultToken"));
    }

    #[test]
    fn job_shop_dispatch() {
        let r = run_job_shop_dispatch(JobShopDispatchParams {
            jobs: None,
            rule: Some(DispatchRule::Spt),
        });
        assert_eq!(r.schedule.len(), 6);
        assert_eq!(r.makespan, 10.0);
        assert!(respects_precedence(&r.schedule));
        assert!(no_machine_overlap(&r.schedule));
        assert!(has(&r.topology.movables, "JobToken"));
    }

    #[test]
    fn flow_shop_neh() {
        let r = run_flow_shop_neh(FlowShopNEHParams::default());
        assert_eq!(r.schedule.len(), 12);
        assert_eq!(r.makespan, 16.0);
        assert!(respects_precedence(&r.schedule));
        assert!(no_machine_overlap(&r.schedule));
        assert!(has(&r.topology.movables, "FlowSequenceToken"));
    }
}
