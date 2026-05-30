//! Port of src/des/test/optimal-control-test.ts
//!
//! End-to-end tests for the entity-based optimal-control models: Pontryagin
//! bang-bang (PMP), Kalman filter (radar tracking), sliding-mode control, MRAC,
//! iterative learning control, feedback linearization, and constrained MPC.
//! Each test checks the canonical theoretical invariant the algorithm satisfies.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::rc::Rc;

    use crate::des::general::feedback_linearization::{
        run_feedback_linearization, FeedbackLinearizationOpts, Reference,
    };
    use crate::des::general::iterative_learning_control::{
        run_iterative_learning_control, ILCReferenceKind, IterativeLearningControlParams,
    };
    use crate::des::general::kalman_filter::{run_radar_tracking, RadarTrackingOpts};
    use crate::des::general::mpc_double_integrator::{run_mpc_double_integrator, MpcDoubleIntOpts};
    use crate::des::general::mrac::{run_mrac, MRACOpts};
    use crate::des::general::pontryagin_bang_bang::{
        optimal_time_double_integrator, run_pontryagin_bang_bang, PontryaginOpts,
    };
    use crate::des::general::sliding_mode_control::{
        run_sliding_mode, DisturbanceType, SlidingModeOpts,
    };

    // 1. Pontryagin bang-bang (time-optimal double integrator via PMP)
    #[test]
    fn pontryagin_bang_bang() {
        let r = run_pontryagin_bang_bang(PontryaginOpts {
            x0: Some([3.0, 0.0]),
            u_max: Some(1.0),
            dt: Some(0.02),
            num_steps: Some(500),
            deadband: Some(0.1),
        })
        .unwrap();
        assert_eq!(r.switch_count, 1, "switches={}", r.switch_count);
        let t_arrival = r.arrival_tick as f64 * 0.02;
        assert!((t_arrival - r.theoretical_arrival_time).abs() < 0.5);
        assert!(r.controls.iter().all(|u| u[0].abs() <= 1.0001));
        let f = r.trajectory.last().unwrap();
        assert!(f[0].abs() + f[1].abs() < 0.05);
        let t_form = optimal_time_double_integrator(3.0, 0.0, 1.0);
        assert!((t_form - 2.0 * 3.0_f64.sqrt()).abs() < 0.05);
    }

    // 2. Kalman filter (radar tracking)
    #[test]
    fn kalman_radar_tracking() {
        let r = run_radar_tracking(RadarTrackingOpts {
            x0: Some([0.0, 1.0]),
            dt: Some(0.1),
            num_steps: Some(200),
            proc_noise_std: Some(0.1),
            meas_noise_std: Some(1.0),
            p0_scale: Some(10.0),
            seed: Some(7),
        })
        .unwrap();
        assert!(r.rmse_pos < r.rmse_meas_pos);
        assert!(r.rmse_pos < 0.5 * r.rmse_meas_pos);
        assert!(r.final_cov_trace < 20.0);
        assert!(r.true_trajectory.len() == r.num_steps + 1 && r.estimates.len() == r.num_steps);
    }

    // 3. Sliding-mode control (robust under bounded disturbance)
    #[test]
    fn sliding_mode_control() {
        let base = |d: DisturbanceType, seed: u32| SlidingModeOpts {
            x0: Some([3.0, 0.0]),
            dt: Some(0.05),
            num_steps: Some(400),
            lambda: Some(2.0),
            eta: Some(3.0),
            boundary: Some(0.05),
            u_bound: Some(5.0),
            disturbance_amp: Some(1.0),
            disturbance_type: Some(d),
            seed: Some(seed),
        };
        let r = run_sliding_mode(base(DisturbanceType::Sin, 1));
        assert!(r.reaching_tick < 100);
        assert!(r.stayed_near_origin);
        assert!(r.final_distance_from_origin < 0.5);

        let r2 = run_sliding_mode(base(DisturbanceType::Square, 2));
        assert!(r2.stayed_near_origin && r2.final_distance_from_origin < 0.5);
    }

    // 4. MRAC (model-reference adaptive control)
    #[test]
    fn mrac_adaptive_control() {
        let r = run_mrac(MRACOpts {
            a: Some(1.0),
            b: Some(2.0),
            am: Some(-2.0),
            bm: Some(2.0),
            x0: Some(0.0),
            xm0: Some(0.0),
            gamma: Some(5.0),
            dt: Some(0.01),
            num_steps: Some(4000),
            ..MRACOpts::default()
        });
        assert!(r.rms_error_steady_state < 0.05);
        assert!((r.final_theta[0] - r.ideal_theta[0]).abs() < 0.2);
        assert!((r.final_theta[1] - r.ideal_theta[1]).abs() < 0.2);

        let r2 = run_mrac(MRACOpts {
            a: Some(-0.5),
            b: Some(1.5),
            am: Some(-3.0),
            bm: Some(3.0),
            gamma: Some(8.0),
            dt: Some(0.01),
            num_steps: Some(4000),
            ..MRACOpts::default()
        });
        assert!(r2.rms_error_steady_state < 0.05);
    }

    // 5. Iterative learning control (repeated-trial feedforward learning)
    #[test]
    fn iterative_learning_control() {
        let params = IterativeLearningControlParams {
            trials: Some(30),
            horizon: Some(80),
            dt: Some(0.1),
            plant_rate: Some(1.2),
            plant_gain: Some(1.0),
            learning_gain: Some(0.8),
            feedback_gain: Some(0.8),
            control_max: Some(5.0),
            reference_kind: Some(ILCReferenceKind::Sine),
            ..Default::default()
        };
        let r = run_iterative_learning_control(&params);
        assert!(r.final_rms_error < 0.05 * r.initial_rms_error);
        assert!(r.final_rms_error < 0.01);
        assert_eq!(r.topology.stations[0], "ilc-trial-source");
        assert!(r.topology.stations.iter().any(|s| s == "ilc-learning-update-station"));
        assert!(r.topology.stations.iter().any(|s| s == "ilc-result-sink"));
        for t in ["ILCTrialPlanToken", "ILCControlProgramToken", "ILCTrialResultToken"] {
            assert!(r.topology.movables.iter().any(|m| m == t), "missing movable {t}");
        }
    }

    // 6. Feedback linearization (nonlinear pendulum tracking)
    #[test]
    fn feedback_linearization() {
        let r = run_feedback_linearization(FeedbackLinearizationOpts {
            params: None,
            theta0: Some(PI),
            theta_dot0: Some(0.0),
            reference: None,
            kp: Some(25.0),
            kv: Some(10.0),
            u_bound: None,
            dt: Some(0.01),
            num_steps: Some(1000),
        });
        assert!(r.rms_error_steady_state < 1e-3);
        assert!(r.trajectory.iter().all(|x| x[0].abs() < 100.0 && x[1].abs() < 100.0));

        let step_ref: Rc<dyn Fn(f64) -> Reference> =
            Rc::new(|_t| Reference { theta: 0.0, theta_dot: 0.0, theta_ddot: 0.0 });
        let r2 = run_feedback_linearization(FeedbackLinearizationOpts {
            params: None,
            theta0: Some(PI),
            theta_dot0: Some(0.0),
            reference: Some(step_ref),
            kp: Some(25.0),
            kv: Some(10.0),
            u_bound: None,
            dt: Some(0.01),
            num_steps: Some(500),
        });
        let final_angle = r2.trajectory.last().unwrap()[0];
        assert!(final_angle.abs() < 0.05);
    }

    // 7. MPC double integrator (constrained receding-horizon QP)
    #[test]
    fn mpc_double_integrator() {
        let r = run_mpc_double_integrator(MpcDoubleIntOpts {
            x0: Some([3.0, 0.0]),
            u_max: Some(1.0),
            n: Some(15),
            q: Some([10.0, 1.0]),
            qf: Some([50.0, 5.0]),
            r: Some(0.1),
            dt: Some(0.1),
            num_steps: Some(100),
        })
        .unwrap();
        assert!(r.max_abs_u <= 1.0001);
        assert!(r.arrival_tick < 100);
        assert!(r.max_abs_u > 0.95);

        let r_tight = run_mpc_double_integrator(MpcDoubleIntOpts {
            x0: Some([3.0, 0.0]),
            u_max: Some(0.5),
            n: Some(20),
            q: Some([10.0, 1.0]),
            qf: Some([50.0, 5.0]),
            r: Some(0.1),
            dt: Some(0.1),
            num_steps: Some(200),
        })
        .unwrap();
        assert!(r_tight.arrival_tick > r.arrival_tick);
    }
}
