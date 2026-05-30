//! Port of `src/des/main-dc-motor.ts`.
//!
//! Runnable demo of the back-EMF DC-motor ODE system: open-loop voltage step
//! (back-EMF rise) vs closed-loop PI speed control.
//!
//! Delegates to `crate::des::general::control_systems::dc_motor` and the
//! iterative DES runner. `class DcMotorDemo` → struct + impl; `process.env.MODE`
//! → `std::env` + a `Mode` enum. Stations are wired with `.pipe(...)` exactly
//! like the TS and driven by `run_iterative_des`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::control_systems::dc_motor::{
    DcMotorChannels, DcMotorParams, DcMotorPlantOpts, DcMotorPlantStation, DcMotorSinkStation,
    LoadProfile, LoadSegment, MotorStateToken, SpeedPiVoltageController, SpeedPiVoltageOpts,
    SpeedReferenceSegment,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Open,
    Closed,
}

struct DcMotorDemo {
    params: DcMotorParams,
    dt: f64,
    steps: usize,
}

impl DcMotorDemo {
    fn new() -> Self {
        DcMotorDemo {
            params: DcMotorParams {
                resistance: 2.0,
                inductance: 0.5,
                back_emf_constant: 0.1,
                torque_constant: 0.1,
                inertia: 0.02,
                friction: 0.002,
            },
            dt: 0.005,
            steps: 3000,
        }
    }

    fn run(&self, mode: Mode) {
        match mode {
            Mode::Open => self.run_open_loop(),
            Mode::Closed => self.run_closed_loop(),
        }
    }

    fn run_open_loop(&self) {
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: self.params.clone(),
                dt: self.dt,
                steps: self.steps,
                initial_state: None,
                load: None,
            },
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
            IterativeRunOptions { shuffle: false, max_ticks: Some(self.steps + 5), ..Default::default() },
        );

        let p = &self.params;
        let wss_denominator = (p.resistance * p.friction) / p.torque_constant + p.back_emf_constant;
        let omega_ss = 12.0 / wss_denominator;
        println!("\n============================================================");
        println!(" DC motor — OPEN LOOP (12 V step), back-EMF rise");
        println!("============================================================");
        self.print_params(self.steps);
        println!("  analytic ω_ss           : {:.3} rad/s", omega_ss);
        println!("  analytic back-EMF_ss    : {:.3} V", p.back_emf_constant * omega_ss);
        self.print_table(&sink.borrow().samples);
        let sink_ref = sink.borrow();
        let f = sink_ref.final_state().expect("final state");
        println!("  final ω                 : {:.3} rad/s", f.omega);
        println!("  final back-EMF          : {:.3} V", f.back_emf);
        println!("  final current           : {:.4} A", f.current);
        println!("============================================================\n");
    }

    fn run_closed_loop(&self) {
        let closed_loop_steps = 6000usize;
        let load = LoadProfile::new(&[
            LoadSegment { from_time: 0.0, torque: 0.0 },
            LoadSegment { from_time: 18.0, torque: 0.3 },
        ]);
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: self.params.clone(),
                dt: self.dt,
                steps: closed_loop_steps,
                initial_state: None,
                load: Some(load),
            },
        )));
        let controller = Rc::new(RefCell::new(SpeedPiVoltageController::new(
            "speed-pi",
            SpeedPiVoltageOpts {
                kp: 1.5,
                ki: 1.0,
                dt: self.dt,
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
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(closed_loop_steps + 5),
                ..Default::default()
            },
        );

        println!("\n============================================================");
        println!(" DC motor — CLOSED LOOP PI speed control");
        println!("============================================================");
        self.print_params(closed_loop_steps);
        println!("  reference: 60 rad/s → 100 rad/s @ t=10s;  load step 0.3 N·m @ t=18s");
        self.print_table(&sink.borrow().samples);
        let sink_ref = sink.borrow();
        let f = sink_ref.final_state().expect("final state");
        println!("  final ω (ref 100)       : {:.3} rad/s", f.omega);
        println!("  final tracking error    : {:.4} rad/s", 100.0 - f.omega);
        println!("  final back-EMF          : {:.3} V", f.back_emf);
        println!("  final armature voltage  : {:.3} V", f.voltage);
        println!("============================================================\n");
    }

    fn print_params(&self, steps: usize) {
        let p = &self.params;
        println!(
            "  R={}Ω  L={}H  K_e={}  K_t={}  J={}  B={}",
            p.resistance, p.inductance, p.back_emf_constant, p.torque_constant, p.inertia, p.friction
        );
        println!("  dt={}s  steps={}", self.dt, steps);
    }

    fn print_table(&self, samples: &[Rc<MotorStateToken>]) {
        println!("  ----------------------------------------------------------");
        println!("    t[s]     V[V]    i[A]    ω[rad/s]   E=K_eω[V]   T_L[N·m]");
        let n = samples.len();
        if n == 0 {
            println!("  ----------------------------------------------------------");
            return;
        }
        let idxs = [
            0,
            (n as f64 * 0.1).floor() as usize,
            (n as f64 * 0.25).floor() as usize,
            (n as f64 * 0.5).floor() as usize,
            (n as f64 * 0.75).floor() as usize,
            n - 1,
        ];
        for &i in &idxs {
            let s = &samples[i];
            println!(
                "   {:>6}  {:>6}  {:>7}  {:>8}  {:>8}  {:>8}",
                format!("{:.3}", s.time),
                format!("{:.2}", s.voltage),
                format!("{:.4}", s.current),
                format!("{:.3}", s.omega),
                format!("{:.3}", s.back_emf),
                format!("{:.3}", s.load_torque),
            );
        }
        println!("  ----------------------------------------------------------");
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    let mode = if std::env::var("MODE").unwrap_or_default().to_lowercase() == "open" {
        Mode::Open
    } else {
        Mode::Closed
    };
    DcMotorDemo::new().run(mode);
}
