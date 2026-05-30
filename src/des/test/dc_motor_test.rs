//! Port of src/des/test/dc-motor-test.ts
//!
//! Unit tests for general/control-systems/dc-motor (back-EMF ODE system) driven
//! through the des-base iterative runner. The TS station `pipe`/`runIterativeDES`
//! wiring maps onto `Rc<RefCell<…>>` station handles plus `run_iterative_des`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::general::control_systems::dc_motor::{
        DcMotorChannels, DcMotorDynamics, DcMotorParams, DcMotorPlantOpts, DcMotorPlantStation,
        DcMotorSinkStation, LoadProfile, LoadSegment, SpeedPiVoltageController, SpeedPiVoltageOpts,
        SpeedReferenceSegment,
    };
    use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
    use crate::des::general::des_base::station::{DESStation, StationRef};

    fn params() -> DcMotorParams {
        DcMotorParams {
            resistance: 2.0,
            inductance: 0.5,
            back_emf_constant: 0.1,
            torque_constant: 0.1,
            inertia: 0.02,
            friction: 0.002,
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    /// Analytic open-loop steady-state speed for constant armature voltage V.
    fn omega_steady_state(v: f64) -> f64 {
        let p = params();
        v / ((p.resistance * p.friction) / p.torque_constant + p.back_emf_constant)
    }

    // [1] Back-EMF & dynamics algebra.
    #[test]
    fn back_emf_and_dynamics_algebra() {
        let p = params();
        let mut dyn_ = DcMotorDynamics::new(p.clone());
        assert!(close(dyn_.back_emf(50.0), 0.1 * 50.0, 1e-12));
        assert!(close(dyn_.electromagnetic_torque(3.0), 0.1 * 3.0, 1e-12));

        dyn_.set_inputs(12.0, 0.0);
        let d = dyn_.derivative(0.0, &[2.0, 30.0]);
        let di_expected = (12.0 - p.resistance * 2.0 - p.back_emf_constant * 30.0) / p.inductance;
        let dw_expected = (p.torque_constant * 2.0 - p.friction * 30.0 - 0.0) / p.inertia;
        assert!(close(d[0], di_expected, 1e-12));
        assert!(close(d[1], dw_expected, 1e-12));

        let ss = dyn_.state_space();
        assert!(close(ss.a[0][0], -p.resistance / p.inductance, 1e-12));
        assert!(close(ss.a[0][1], -p.back_emf_constant / p.inductance, 1e-12));
        assert!(close(ss.a[1][0], p.torque_constant / p.inertia, 1e-12));
        assert!(close(ss.a[1][1], -p.friction / p.inertia, 1e-12));
        assert!(close(ss.b[0][0], 1.0 / p.inductance, 1e-12));
        assert!(close(ss.b[1][0], 0.0, 1e-12));
        assert_eq!(ss.c[0][0], 0.0);
        assert_eq!(ss.c[0][1], 1.0);
    }

    // [2] LoadProfile schedule.
    #[test]
    fn load_profile_schedule() {
        let lp = LoadProfile::new(&[
            LoadSegment { from_time: 0.0, torque: 0.0 },
            LoadSegment { from_time: 5.0, torque: 0.4 },
        ]);
        assert!(close(lp.torque_at(2.0), 0.0, 1e-9));
        assert!(close(lp.torque_at(5.0), 0.4, 1e-9));
        assert!(close(lp.torque_at(100.0), 0.4, 1e-9));
    }

    // [3] Open-loop step — back-EMF rises, current limited, ω → analytic.
    #[test]
    fn open_loop_step_reaches_analytic_steady_state() {
        let dt = 0.005;
        let steps = 4000;
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts { params: params(), dt, steps, initial_state: None, load: None },
        )));
        plant.borrow_mut().set_open_loop_voltage(12.0);
        let sink = Rc::new(RefCell::new(DcMotorSinkStation::new("sink")));
        plant.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            DcMotorChannels::STATE,
            DcMotorChannels::STATE,
        );
        run_iterative_des(
            vec![plant.clone() as StationRef, sink.clone() as StationRef],
            IterativeRunOptions { shuffle: false, max_ticks: Some(steps + 5), ..Default::default() },
        );

        let sb = sink.borrow();
        let s = &sb.samples;
        assert_eq!(s.len(), steps, "one sample per tick");
        assert!(s[0].omega.abs() < 0.05 && s[0].current.abs() < 0.5);

        let mut mono = true;
        for k in 1..s.len() {
            if s[k].back_emf < s[k - 1].back_emf - 1e-6 {
                mono = false;
                break;
            }
        }
        assert!(mono, "back-EMF rises monotonically");

        let omega_ss = omega_steady_state(12.0);
        assert!(close(sb.final_omega(), omega_ss, 1e-3));
        assert!(close(sb.final_back_emf(), 0.1 * omega_ss, 1e-3));
        assert!(sb.final_back_emf() < 12.0);
        let i_ss = (12.0 - sb.final_back_emf()) / params().resistance;
        assert!(close(sb.final_state().unwrap().current, i_ss, 1e-2));
    }

    // [4] Closed-loop PI — tracks reference, rejects load disturbance.
    #[test]
    fn closed_loop_pi_tracks_and_rejects_load() {
        let dt = 0.005;
        let steps = 6000;
        let load = LoadProfile::new(&[
            LoadSegment { from_time: 0.0, torque: 0.0 },
            LoadSegment { from_time: 18.0, torque: 0.3 },
        ]);
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: params(),
                dt,
                steps,
                initial_state: None,
                load: Some(load),
            },
        )));
        let controller = Rc::new(RefCell::new(SpeedPiVoltageController::new(
            "speed-pi",
            SpeedPiVoltageOpts {
                kp: 1.5,
                ki: 1.0,
                dt,
                max_voltage: Some(48.0),
                reference: vec![
                    SpeedReferenceSegment { from_time: 0.0, speed: 60.0 },
                    SpeedReferenceSegment { from_time: 10.0, speed: 100.0 },
                ],
            },
        )));
        let sink = Rc::new(RefCell::new(DcMotorSinkStation::new("sink")));

        plant.borrow_mut().core_mut().pipe(
            controller.clone() as StationRef,
            DcMotorChannels::STATE,
            DcMotorChannels::STATE,
        );
        plant.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            DcMotorChannels::STATE,
            DcMotorChannels::STATE,
        );
        controller.borrow_mut().core_mut().pipe(
            plant.clone() as StationRef,
            DcMotorChannels::VOLTAGE,
            DcMotorChannels::VOLTAGE,
        );
        run_iterative_des(
            vec![
                plant.clone() as StationRef,
                controller.clone() as StationRef,
                sink.clone() as StationRef,
            ],
            IterativeRunOptions { shuffle: false, max_ticks: Some(steps + 5), ..Default::default() },
        );

        let sb = sink.borrow();
        let s = &sb.samples;
        let at = |t: f64| {
            let idx = ((t / dt).round() as usize).saturating_sub(1).min(s.len() - 1);
            &s[idx]
        };
        assert!((at(9.5).omega - 60.0).abs() < 0.5, "ω@9.5 = {}", at(9.5).omega);
        assert!((at(17.5).omega - 100.0).abs() < 0.5, "ω@17.5 = {}", at(17.5).omega);
        assert!((sb.final_omega() - 100.0).abs() < 0.1, "ω = {}", sb.final_omega());

        let f = sb.final_state().unwrap();
        let p = params();
        assert!(close(
            p.torque_constant * f.current,
            p.friction * f.omega + f.load_torque,
            1e-2
        ));
        assert!(close(f.voltage, p.resistance * f.current + f.back_emf, 1e-2));

        let mut peak = 0.0_f64;
        for x in s.iter() {
            if x.time <= 10.0 {
                peak = peak.max(x.omega);
            }
        }
        assert!(peak < 72.0, "overshoot peak = {peak}");
    }
}
