//! Port of src/des/test/preconditions-test.ts
//!
//! Verifies that the pre-run guards (`Preconditions::*` plus each model's
//! parameter validation) actually fire on bad inputs. The TypeScript original
//! throws a `PreconditionError` naming the offending parameter; the Rust models
//! either return `Result<_, PreconditionError>` or panic (via `.unwrap()` /
//! `.expect()` / `panic!`) depending on the module, so this port checks both
//! shapes. The two TS cases that select an invalid string for what is now a
//! Rust enum (`solver`, `disturbanceType`) are unrepresentable and are noted
//! rather than ported.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::des::general::des_base::preconditions::Preconditions;

    use crate::des::general::actor_critic_gridworld::{
        run_actor_critic_gridworld, ActorCriticTrainOpts,
    };
    use crate::des::general::blackjack::{run_blackjack_mc, BlackjackTrainOpts};
    use crate::des::general::double_integrator_lqr::{
        run_double_integrator_lqr, DoubleIntegratorOpts,
    };
    use crate::des::general::feedback_linearization::{
        run_feedback_linearization, FeedbackLinearizationOpts, PartialPendulumParams,
    };
    use crate::des::general::four_rooms::{run_four_rooms_smdp, FourRoomsTrainOpts};
    use crate::des::general::inventory_dp::{solve_inventory_dp, InventoryProblem};
    use crate::des::general::kalman_filter::{run_radar_tracking, RadarTrackingOpts};
    use crate::des::general::mountain_car::{run_mountain_car, MountainCarTrainOpts};
    use crate::des::general::mpc_double_integrator::{
        run_mpc_double_integrator, MpcDoubleIntOpts,
    };
    use crate::des::general::mrac::{run_mrac, MRACOpts};
    use crate::des::general::pontryagin_bang_bang::{
        run_pontryagin_bang_bang, PontryaginOpts,
    };
    use crate::des::general::sliding_mode_control::{run_sliding_mode, SlidingModeOpts};
    use crate::des::general::stag_hunt::{run_stag_hunt, StagHuntOpts};
    use crate::des::general::temp_control::{run_temp_control, ControllerSpec, SimConfig};
    use crate::des::general::tiger_pomdp::{
        build_tiger_spec, simulate_tiger, TigerOpts, TigerSimOpts, TigerSolver,
    };

    /// Run `f` (expected to panic) and return its panic message.
    fn panic_message(f: impl FnOnce()) -> String {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(_) => panic!("expected a panic, but the call succeeded"),
            Err(e) => {
                if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else {
                    String::from("<non-string panic payload>")
                }
            }
        }
    }

    /// Assert a panicking guard fired and its message mentions `fragment`.
    fn assert_panics_with(fragment: &str, f: impl FnOnce()) {
        let msg = panic_message(f);
        assert!(
            msg.to_lowercase().contains(&fragment.to_lowercase()),
            "panic message did not mention {fragment:?}: {msg}"
        );
    }

    // =========================================================================
    // 1. PRECONDITIONS UTILITY (low-level guards)
    // =========================================================================

    #[test]
    fn low_level_guards_reject_bad_inputs() {
        assert!(Preconditions::finite("m", "x", f64::NAN).is_err());
        assert!(Preconditions::finite("m", "x", f64::INFINITY).is_err());
        assert!(Preconditions::positive("m", "x", 0.0).is_err());
        assert!(Preconditions::positive("m", "x", -0.5).is_err());
        assert!(Preconditions::in_range("m", "x", 1.2, 0.0, 1.0).is_err());
        assert!(Preconditions::in_range("m", "x", -0.2, 0.0, 1.0).is_err());
        assert!(Preconditions::integer("m", "k", 3.7).is_err());
        assert!(Preconditions::probability_vector("m", "p", &[0.5, 0.4, 0.09], 1e-9).is_err());
        assert!(Preconditions::probability_vector("m", "p", &[0.5, -0.1, 0.6], 1e-9).is_err());
        assert!(
            Preconditions::symmetric_matrix("m", "M", &vec![vec![1.0, 2.0], vec![3.0, 4.0]], 1e-9)
                .is_err()
        );
        assert!(Preconditions::positive_definite_cholesky(
            "m",
            "M",
            &vec![vec![0.0, 0.0], vec![0.0, 1.0]]
        )
        .is_err());
        assert!(Preconditions::positive_definite_cholesky(
            "m",
            "M",
            &vec![vec![1.0, 2.0], vec![2.0, 1.0]]
        )
        .is_err());
        assert!(Preconditions::not_div_by_zero("m", "d", 0.0, 1e-12).is_err());
        assert!(Preconditions::length_eq("m", "arr", &[1, 2], 3).is_err());
        assert!(Preconditions::integer_in_range("m", "k", 11.0, 0.0, 10.0).is_err());
    }

    #[test]
    fn low_level_guards_accept_valid_inputs() {
        assert!(Preconditions::finite("m", "x", 1.5).is_ok());
        assert!(Preconditions::positive("m", "x", 0.001).is_ok());
        assert!(Preconditions::in_range("m", "x", 0.5, 0.0, 1.0).is_ok());
        assert!(Preconditions::probability_vector("m", "p", &[0.4, 0.3, 0.3], 1e-9).is_ok());
        assert!(
            Preconditions::symmetric_matrix("m", "M", &vec![vec![1.0, 2.0], vec![2.0, 3.0]], 1e-9)
                .is_ok()
        );
        assert!(Preconditions::positive_definite_cholesky(
            "m",
            "M",
            &vec![vec![2.0, 1.0], vec![1.0, 2.0]]
        )
        .is_ok());
        assert!(Preconditions::not_div_by_zero("m", "d", 0.5, 1e-12).is_ok());
    }

    // =========================================================================
    // 2. ENTITY-BASED CONTROL MODELS
    // =========================================================================

    #[test]
    fn pontryagin_bang_bang_preconditions() {
        // These guards surface as `Result::Err(PreconditionError)`.
        assert!(run_pontryagin_bang_bang(PontryaginOpts {
            u_max: Some(0.0),
            ..Default::default()
        })
        .is_err());
        assert!(run_pontryagin_bang_bang(PontryaginOpts {
            dt: Some(0.0),
            ..Default::default()
        })
        .is_err());
        // TS uses numSteps = -1; `num_steps` is `usize`, so 0 is the smallest
        // out-of-range value the guard rejects.
        assert!(run_pontryagin_bang_bang(PontryaginOpts {
            num_steps: Some(0),
            ..Default::default()
        })
        .is_err());
        let e = run_pontryagin_bang_bang(PontryaginOpts {
            x0: Some([f64::NAN, 0.0]),
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.to_string().to_lowercase().contains("x0"));
    }

    #[test]
    fn kalman_filter_preconditions() {
        let e = run_radar_tracking(RadarTrackingOpts {
            meas_noise_std: Some(0.0),
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.to_string().contains("measNoiseStd"));
        assert!(run_radar_tracking(RadarTrackingOpts {
            proc_noise_std: Some(-0.1),
            ..Default::default()
        })
        .is_err());
        assert!(run_radar_tracking(RadarTrackingOpts {
            dt: Some(0.0),
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn sliding_mode_preconditions() {
        // SMC reaching condition: eta must strictly exceed the disturbance bound.
        assert_panics_with("eta", || {
            run_sliding_mode(SlidingModeOpts {
                eta: Some(0.5),
                disturbance_amp: Some(1.0),
                num_steps: Some(10),
                ..Default::default()
            });
        });
        assert_panics_with("lambda", || {
            run_sliding_mode(SlidingModeOpts {
                lambda: Some(0.0),
                ..Default::default()
            });
        });
        assert_panics_with("boundary", || {
            run_sliding_mode(SlidingModeOpts {
                boundary: Some(0.0),
                ..Default::default()
            });
        });
        // PORT NOTE: the TS "rejects unknown disturbanceType" case is
        // unrepresentable — `disturbance_type` is the `DisturbanceType` enum.
    }

    #[test]
    fn mrac_preconditions() {
        assert_panics_with("b", || {
            run_mrac(MRACOpts {
                b: Some(0.0),
                ..Default::default()
            });
        });
        assert_panics_with("am", || {
            run_mrac(MRACOpts {
                am: Some(0.1),
                ..Default::default()
            });
        });
        assert_panics_with("gamma*dt", || {
            run_mrac(MRACOpts {
                gamma: Some(1000.0),
                dt: Some(0.01),
                num_steps: Some(10),
                ..Default::default()
            });
        });
    }

    #[test]
    fn feedback_linearization_preconditions() {
        assert_panics_with("params.m", || {
            run_feedback_linearization(FeedbackLinearizationOpts {
                params: Some(PartialPendulumParams {
                    m: Some(0.0),
                    ..Default::default()
                }),
                ..Default::default()
            });
        });
        assert_panics_with("params.l", || {
            run_feedback_linearization(FeedbackLinearizationOpts {
                params: Some(PartialPendulumParams {
                    l: Some(0.0),
                    ..Default::default()
                }),
                ..Default::default()
            });
        });
        assert_panics_with("kp", || {
            run_feedback_linearization(FeedbackLinearizationOpts {
                kp: Some(-1.0),
                ..Default::default()
            });
        });
    }

    #[test]
    fn mpc_preconditions() {
        let e = run_mpc_double_integrator(MpcDoubleIntOpts {
            r: Some(0.0),
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.to_string().contains('R'));
        let e = run_mpc_double_integrator(MpcDoubleIntOpts {
            n: Some(0),
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.to_string().contains("N (horizon)"));
        assert!(run_mpc_double_integrator(MpcDoubleIntOpts {
            u_max: Some(0.0),
            ..Default::default()
        })
        .is_err());
    }

    // =========================================================================
    // 3. MDP-ADJACENT MODELS
    // =========================================================================

    fn base_inventory_problem() -> InventoryProblem {
        InventoryProblem {
            horizon: 3,
            s_max: 5,
            demand_pmf: vec![0.3, 0.3, 0.4],
            price: 8.0,
            cost: 3.0,
            fixed_cost: 1.0,
            hold_cost: 0.5,
            stockout_cost: 5.0,
            salvage_value: 0.0,
            discount: Some(1.0),
            initial_inventory: 0,
        }
    }

    #[test]
    fn inventory_dp_preconditions() {
        assert_panics_with("horizon", || {
            let mut p = base_inventory_problem();
            p.horizon = 0;
            solve_inventory_dp(&p, None);
        });
        assert_panics_with("demandPmf", || {
            let mut p = base_inventory_problem();
            p.demand_pmf = vec![0.2, 0.3, 0.4];
            solve_inventory_dp(&p, None);
        });
        assert_panics_with("price", || {
            let mut p = base_inventory_problem();
            p.price = -1.0;
            solve_inventory_dp(&p, None);
        });
        assert_panics_with("discount", || {
            let mut p = base_inventory_problem();
            p.discount = Some(1.5);
            solve_inventory_dp(&p, None);
        });
        assert_panics_with("initialInventory", || {
            let mut p = base_inventory_problem();
            p.initial_inventory = 999;
            solve_inventory_dp(&p, None);
        });
        // Valid inputs run successfully.
        let _ = solve_inventory_dp(&base_inventory_problem(), None);
    }

    #[test]
    fn mountain_car_preconditions() {
        assert_panics_with("numEpisodes", || {
            run_mountain_car(MountainCarTrainOpts {
                num_episodes: 0,
                ..Default::default()
            });
        });
        assert_panics_with("alpha", || {
            run_mountain_car(MountainCarTrainOpts {
                num_episodes: 1,
                alpha: Some(0.0),
                ..Default::default()
            });
        });
        assert_panics_with("gamma", || {
            run_mountain_car(MountainCarTrainOpts {
                num_episodes: 1,
                gamma: Some(1.5),
                ..Default::default()
            });
        });
        assert_panics_with("epsilon", || {
            run_mountain_car(MountainCarTrainOpts {
                num_episodes: 1,
                epsilon: Some(2.0),
                ..Default::default()
            });
        });
    }

    #[test]
    fn four_rooms_preconditions() {
        assert_panics_with("numEpisodes", || {
            run_four_rooms_smdp(FourRoomsTrainOpts {
                num_episodes: 0,
                ..Default::default()
            });
        });
        assert_panics_with("gamma", || {
            run_four_rooms_smdp(FourRoomsTrainOpts {
                num_episodes: 1,
                gamma: Some(1.5),
                ..Default::default()
            });
        });
        assert_panics_with("slip", || {
            run_four_rooms_smdp(FourRoomsTrainOpts {
                num_episodes: 1,
                slip: Some(1.5),
                ..Default::default()
            });
        });
    }

    #[test]
    fn actor_critic_preconditions() {
        assert_panics_with("numEpisodes", || {
            run_actor_critic_gridworld(ActorCriticTrainOpts {
                num_episodes: 0,
                ..Default::default()
            });
        });
        assert_panics_with("alphaV", || {
            run_actor_critic_gridworld(ActorCriticTrainOpts {
                num_episodes: 1,
                alpha_v: Some(0.0),
                ..Default::default()
            });
        });
        assert_panics_with("width", || {
            run_actor_critic_gridworld(ActorCriticTrainOpts {
                num_episodes: 1,
                width: Some(0),
                ..Default::default()
            });
        });
    }

    #[test]
    fn blackjack_preconditions() {
        assert_panics_with("numEpisodes", || {
            run_blackjack_mc(BlackjackTrainOpts {
                num_episodes: 0,
                ..Default::default()
            });
        });
        assert_panics_with("gamma", || {
            run_blackjack_mc(BlackjackTrainOpts {
                num_episodes: 1,
                gamma: Some(2.0),
                ..Default::default()
            });
        });
    }

    #[test]
    fn stag_hunt_preconditions() {
        assert_panics_with("numEpisodes", || {
            run_stag_hunt(&StagHuntOpts {
                num_episodes: 0,
                ..StagHuntOpts::new(0)
            });
        });
        assert_panics_with("alpha", || {
            run_stag_hunt(&StagHuntOpts {
                alpha: Some(-0.1),
                ..StagHuntOpts::new(1)
            });
        });
    }

    #[test]
    fn tiger_pomdp_preconditions() {
        let spec = build_tiger_spec(&TigerOpts::default());
        assert_panics_with("numSteps", || {
            simulate_tiger(TigerSimOpts {
                spec: Some(spec.clone()),
                solver: TigerSolver::Qmdp,
                num_steps: 0,
                seed: None,
                initial_state: None,
                initial_belief: None,
            });
        });
        assert_panics_with("initialBelief", || {
            simulate_tiger(TigerSimOpts {
                spec: Some(spec.clone()),
                solver: TigerSolver::Qmdp,
                num_steps: 5,
                seed: None,
                initial_state: None,
                initial_belief: Some(vec![0.3, 0.3]),
            });
        });
        // PORT NOTE: the TS "rejects unknown solver" case is unrepresentable —
        // `solver` is the `TigerSolver` enum.
    }

    #[test]
    fn double_integrator_lqr_preconditions() {
        assert!(run_double_integrator_lqr(DoubleIntegratorOpts {
            r_u: Some(0.0),
            ..Default::default()
        })
        .is_err());
        assert!(run_double_integrator_lqr(DoubleIntegratorOpts {
            q_pos: Some(-1.0),
            ..Default::default()
        })
        .is_err());
        assert!(run_double_integrator_lqr(DoubleIntegratorOpts {
            dt: Some(0.0),
            ..Default::default()
        })
        .is_err());
        assert!(run_double_integrator_lqr(DoubleIntegratorOpts {
            gamma: Some(0.0),
            ..Default::default()
        })
        .is_err());
    }

    // =========================================================================
    // 4. TEMP CONTROL
    // =========================================================================

    fn base_temp_cfg() -> SimConfig {
        SimConfig {
            t_target: 20.0,
            band: None,
            duration_h: 1.0,
            dt_min: 1.0,
            controller: ControllerSpec::Pid {
                kp: 100.0,
                ki: 5.0,
                kd: 10.0,
            },
            house: None,
            outdoor: None,
            cost_per_kwh: 1.0,
            comfort_penalty: 1.0,
            sensor_noise_std: None,
            forecast_noise_std: None,
            forecast_horizon_h: None,
        }
    }

    #[test]
    fn temp_control_preconditions() {
        assert_panics_with("dt_min", || {
            let mut cfg = base_temp_cfg();
            cfg.dt_min = 0.0;
            run_temp_control(cfg);
        });
        assert_panics_with("duration_h", || {
            let mut cfg = base_temp_cfg();
            cfg.duration_h = 0.0;
            run_temp_control(cfg);
        });
        assert_panics_with("cost_per_kWh", || {
            let mut cfg = base_temp_cfg();
            cfg.cost_per_kwh = -1.0;
            run_temp_control(cfg);
        });
        assert_panics_with("sensorNoiseStd", || {
            let mut cfg = base_temp_cfg();
            cfg.sensor_noise_std = Some(-0.1);
            run_temp_control(cfg);
        });
    }
}
