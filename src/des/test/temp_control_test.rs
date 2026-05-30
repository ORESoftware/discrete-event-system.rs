//! Port of src/des/test/temp-control-test.ts
//!
//! Unit tests for general/temp-control physics and controllers. The TS
//! check()/tally harness becomes `#[test]` functions; stochastic outdoor /
//! controller runs are seeded for reproducibility.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::prng::mulberry32;
    use crate::des::general::temp_control::{
        controller_step, fuzzy_delta_controller, house_step, mdp_mpc_controller, run_temp_control,
        true_outdoor_temp, ControllerSpec, ControllerState, HouseParams, OutdoorPattern,
        OutdoorPatternPartial, SimConfig, TempObs, DEFAULT_HOUSE,
    };
    use crate::des::shared::capabilities::RandomSource;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn base_cfg() -> SimConfig {
        SimConfig {
            t_target: 70.0,
            band: Some(2.0),
            duration_h: 4.0,
            dt_min: 1.0,
            controller: ControllerSpec::Fuzzy,
            house: None,
            outdoor: None,
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: Some(0.0),
            forecast_noise_std: Some(0.0),
            forecast_horizon_h: Some(1.0),
            seed: Some(1),
        }
    }

    // [1] House physics — forward-Euler step
    #[test]
    fn house_physics_step() {
        assert!(close(
            house_step(70.0, 30.0, 5.0, 0.0, &DEFAULT_HOUSE),
            70.0,
            1e-9
        ));
        assert!(close(
            house_step(60.0, 60.0, 0.0, 1.0, &DEFAULT_HOUSE),
            60.0,
            1e-9
        ));
        let a = house_step(70.0, 30.0, 0.0, 0.5, &DEFAULT_HOUSE);
        let b = house_step(70.0, 30.0, 3.0, 0.5, &DEFAULT_HOUSE);
        let c = house_step(70.0, 30.0, 5.0, 0.5, &DEFAULT_HOUSE);
        assert!(close(b - a, (3.0 / 5.0) * (c - a), 1e-9));
        let q_ss = (70.0 - 30.0) / (DEFAULT_HOUSE.tau * DEFAULT_HOUSE.g);
        assert!(close(
            house_step(70.0, 30.0, q_ss, 1.0, &DEFAULT_HOUSE),
            70.0,
            1e-9
        ));
        let ins = HouseParams {
            tau: 1e9,
            ..DEFAULT_HOUSE
        };
        assert!(close(
            house_step(70.0, 30.0, 5.0, 1.0, &ins) - 70.0,
            5.0,
            1e-3
        ));
    }

    // [2] Outdoor temperature pattern
    #[test]
    fn outdoor_temperature_pattern() {
        let p = OutdoorPattern {
            mean: 30.0,
            amp: 0.0,
            phase: 0.0,
            noise_std: 0.0,
        };
        for t in [0.0, 6.0, 12.0, 18.0, 24.0] {
            assert!(close(true_outdoor_temp(t, &p, None), 30.0, 1e-9));
        }
        let q = OutdoorPattern {
            mean: 25.0,
            amp: 15.0,
            phase: 9.0,
            noise_std: 0.0,
        };
        assert!(close(true_outdoor_temp(15.0, &q, None), 40.0, 1e-9));
        assert!(close(true_outdoor_temp(3.0, &q, None), 10.0, 1e-9));
        assert!(close(
            true_outdoor_temp(7.0, &q, None),
            true_outdoor_temp(31.0, &q, None),
            1e-9
        ));
    }

    // [3] PRNG reproducibility
    #[test]
    fn prng_reproducibility() {
        let mut r1 = mulberry32(123);
        let mut r2 = mulberry32(123);
        for _ in 0..100 {
            assert_eq!(r1.next_float(), r2.next_float());
        }
    }

    // [4] Fuzzy-PI controller — Δ-Q rule output
    #[test]
    fn fuzzy_delta_controller_rules() {
        let (e, de) = (3.0, 2.0);
        let a = fuzzy_delta_controller(e, de);
        let b = fuzzy_delta_controller(-e, -de);
        assert!(close(a, -b, 1e-9));
        assert!(fuzzy_delta_controller(20.0, 20.0) > 0.99);
        assert!(fuzzy_delta_controller(-20.0, -20.0) < -0.99);
        for e in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
            for de in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
                let v = fuzzy_delta_controller(e, de);
                assert!(v >= -1.0 - 1e-12 && v <= 1.0 + 1e-12);
            }
        }
    }

    // [5] PID controller — anti-windup + steady state
    #[test]
    fn pid_steady_state() {
        let cfg = SimConfig {
            duration_h: 80.0,
            controller: ControllerSpec::Pid {
                kp: 3.0,
                ki: 0.5,
                kd: 0.5,
            },
            outdoor: Some(OutdoorPatternPartial {
                mean: Some(25.0),
                amp: Some(0.0),
                phase: Some(0.0),
                noise_std: Some(0.0),
            }),
            ..base_cfg()
        };
        let r = run_temp_control(cfg);
        let last: Vec<f64> = r.t_in.iter().rev().take(60).copied().collect();
        let mean = last.iter().sum::<f64>() / last.len() as f64;
        assert!((mean - 70.0).abs() < 0.1, "mean = {mean}");
        assert!(r
            .q
            .iter()
            .all(|&q| q >= 0.0 && q <= DEFAULT_HOUSE.q_max + 1e-9));
    }

    // [6] MDP-MPC — basic correctness
    #[test]
    fn mdp_mpc_basic() {
        let fc = vec![20.0; 360];
        let q_cold = mdp_mpc_controller(
            60.0,
            &fc,
            6.0,
            6,
            70.0,
            1.0 / 60.0,
            5.0,
            &DEFAULT_HOUSE,
            0.5,
            0.15,
            1.0,
        );
        assert!(q_cold >= 4.0, "Q = {q_cold}");
        let q_hot = mdp_mpc_controller(
            75.0,
            &fc,
            6.0,
            6,
            70.0,
            1.0 / 60.0,
            5.0,
            &DEFAULT_HOUSE,
            0.5,
            0.15,
            1.0,
        );
        assert_eq!(q_hot, 0.0);
        let q_at = mdp_mpc_controller(
            70.0,
            &fc,
            6.0,
            6,
            70.0,
            1.0 / 60.0,
            5.0,
            &DEFAULT_HOUSE,
            0.5,
            0.15,
            1.0,
        );
        assert!(q_at > 0.0 && q_at <= 5.0, "Q = {q_at}");

        let fc_mild = vec![50.0; 60];
        let q5 = mdp_mpc_controller(
            70.0,
            &fc_mild,
            1.0,
            5,
            70.0,
            1.0 / 60.0,
            5.0,
            &DEFAULT_HOUSE,
            0.5,
            0.15,
            1.0,
        );
        let valid = [0.0, 1.25, 2.5, 3.75, 5.0]
            .iter()
            .any(|v| (v - q5).abs() < 1e-9);
        assert!(valid, "Q = {q5}");
    }

    // [7] Bang-bang controller
    #[test]
    fn bang_bang_controller() {
        let mut ctx = TempObs {
            t_target: 70.0,
            t_in_meas: 0.0,
            forecast: vec![30.0],
            dt_h: 1.0 / 60.0,
            q_max: 5.0,
            house: DEFAULT_HOUSE,
        };
        let mut st = ControllerState::default();
        ctx.t_in_meas = 65.0;
        assert_eq!(
            controller_step(&ControllerSpec::BangBang, &mut st, &ctx),
            5.0
        );
        ctx.t_in_meas = 75.0;
        assert_eq!(
            controller_step(&ControllerSpec::BangBang, &mut st, &ctx),
            0.0
        );
        ctx.t_in_meas = 70.0;
        assert_eq!(
            controller_step(&ControllerSpec::BangBang, &mut st, &ctx),
            0.0
        );
    }

    // [8] Full-run invariants
    #[test]
    fn full_run_invariants() {
        let cfg = SimConfig {
            duration_h: 4.0,
            controller: ControllerSpec::Fuzzy,
            ..base_cfg()
        };
        let r = run_temp_control(cfg.clone());
        for k in 1..r.energy.len() {
            assert!(r.energy[k] >= r.energy[k - 1] - 1e-9);
        }
        let expected = (cfg.duration_h / (cfg.dt_min / 60.0)).round() as usize;
        assert_eq!(r.trace.len(), expected);
        for k in 1..r.trace.len() {
            assert!(r.trace[k].violation_fh >= r.trace[k - 1].violation_fh - 1e-9);
        }
        let cost_check = cfg.cost_per_kwh * r.energy_kwh + cfg.comfort_penalty * r.violation_fh;
        assert!(close(r.cost, cost_check, 1e-9));
        assert!(close(
            r.trace[r.trace.len() - 1].energy_cum_kwh,
            r.energy_kwh,
            1e-9
        ));
    }

    // [9] Different controllers, same scenario, all stay in band
    #[test]
    fn controllers_stay_in_band() {
        let base = SimConfig {
            duration_h: 12.0,
            sensor_noise_std: Some(0.1),
            forecast_noise_std: Some(1.0),
            forecast_horizon_h: Some(4.0),
            seed: Some(42),
            ..base_cfg()
        };
        let specs = [
            ControllerSpec::BangBang,
            ControllerSpec::Pid {
                kp: 3.0,
                ki: 0.5,
                kd: 0.5,
            },
            ControllerSpec::Fuzzy,
            ControllerSpec::MdpMpc {
                horizon_h: 4.0,
                n_levels: 6,
                comfort_penalty: 0.5,
                cost_per_kwh: 0.15,
                track_weight: Some(1.0),
            },
        ];
        for spec in specs {
            let cfg = SimConfig {
                controller: spec,
                ..base.clone()
            };
            let r = run_temp_control(cfg);
            assert!(
                r.comfort_pct >= 0.99,
                "{:?}: {}%",
                spec,
                100.0 * r.comfort_pct
            );
        }
    }
}
