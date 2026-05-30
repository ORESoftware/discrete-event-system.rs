//! Port of src/des/test/wind-mppt-test.ts
//!
//! Unit tests for general/control-systems/wind-mppt and the shared numerical
//! solvers. Groups [1]-[3] (ODE integrators, aerodynamics, wind profile) are
//! ported faithfully.
//!
//! PORT NOTE: groups [4] and [5] (closed-loop MPPT driven by the DES station
//! graph WindTurbinePlantStation → controller → WindMpptSinkStation via
//! `run_iterative_des` + `pipe`) are deferred; they require wiring the
//! station-graph runner, which is exercised elsewhere.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::control_systems::numerical_solvers::{
        FixedStepIntegrator, ForwardEulerIntegrator, OdeSystem, RungeKutta4Integrator,
    };
    use crate::des::general::control_systems::wind_mppt::{
        WindProfile, WindProfileSegment, WindTurbineAeroOpts, WindTurbineAerodynamics,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    /// dx/dt = -k·x (analytic: x(t) = x0·e^{-kt}).
    struct ExponentialDecay {
        k: f64,
    }
    impl OdeSystem for ExponentialDecay {
        fn dimension(&self) -> usize {
            1
        }
        fn derivative(&self, _t: f64, state: &[f64]) -> Vec<f64> {
            vec![-self.k * state[0]]
        }
    }

    fn aero() -> WindTurbineAerodynamics {
        WindTurbineAerodynamics::new(WindTurbineAeroOpts {
            air_density: None,
            blade_radius: 2.5,
            pitch_deg: Some(0.0),
        })
    }

    // [1] Numerical solvers — RK4 vs forward Euler accuracy
    #[test]
    fn numerical_solvers_accuracy() {
        let sys = ExponentialDecay { k: 1.0 };
        let exact = (-1.0_f64).exp();
        let rk4 = RungeKutta4Integrator::new().integrate(&sys, 0.0, &[1.0], 0.1, 10);
        let euler = ForwardEulerIntegrator::new().integrate(&sys, 0.0, &[1.0], 0.1, 10);
        let rk4_err = (rk4.states[rk4.states.len() - 1][0] - exact).abs();
        let euler_err = (euler.states[euler.states.len() - 1][0] - exact).abs();
        assert!(rk4.times.len() == 11 && close(rk4.times[10], 1.0, 1e-12));
        assert!(rk4_err < 1e-6, "err={rk4_err:e}");
        assert!(
            rk4_err < euler_err / 100.0,
            "rk4={rk4_err:e} euler={euler_err:e}"
        );

        let threw = std::panic::catch_unwind(|| {
            RungeKutta4Integrator::new().integrate(
                &ExponentialDecay { k: 1.0 },
                0.0,
                &[1.0],
                0.0,
                1,
            )
        })
        .is_err();
        assert!(threw);
    }

    // [2] Aerodynamics — C_p model & optimal operating point
    #[test]
    fn aerodynamics_cp_model() {
        let aero = aero();
        assert!(close(aero.swept_area(), std::f64::consts::PI * 6.25, 1e-9));
        assert!(close(aero.tip_speed_ratio(16.0, 5.0), 8.0, 1e-9));
        let lambda_star = aero.optimal_tip_speed_ratio();
        assert!(
            lambda_star > 7.5 && lambda_star < 8.5,
            "lambda*={lambda_star}"
        );
        let cp_max = aero.max_power_coefficient();
        assert!(cp_max > 0.45 && cp_max < 0.50, "cp_max={cp_max}");
        for i in 1..=36 {
            let l = i as f64 * 0.5;
            assert!(aero.power_coefficient(l) <= cp_max + 1e-9);
        }
        assert!(aero.optimal_torque_gain() > 0.0);

        let (v, omega) = (9.0, 20.0);
        let lambda = aero.tip_speed_ratio(omega, v);
        let expected = 0.5 * 1.225 * aero.swept_area() * aero.power_coefficient(lambda) * v.powi(3);
        assert!(close(aero.mechanical_power(v, omega), expected, 1e-9));
        assert!(close(aero.aero_torque(v, omega), expected / omega, 1e-9));
    }

    // [3] WindProfile — piecewise-constant schedule
    #[test]
    fn wind_profile_schedule() {
        let wp = WindProfile::new(&[
            WindProfileSegment {
                from_time: 0.0,
                speed: 8.0,
            },
            WindProfileSegment {
                from_time: 20.0,
                speed: 11.0,
            },
            WindProfileSegment {
                from_time: 40.0,
                speed: 9.0,
            },
        ]);
        assert!(close(wp.speed_at(5.0), 8.0, 1e-9));
        assert!(close(wp.speed_at(20.0), 11.0, 1e-9));
        assert!(close(wp.speed_at(100.0), 9.0, 1e-9));
    }
}
