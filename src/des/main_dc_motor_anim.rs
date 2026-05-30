//! Port of `src/des/main-dc-motor-anim.ts`.
//!
//! Drives the back-EMF DC-motor DES (open-loop step vs closed-loop PI speed
//! control) and would render an HTML animation of the trajectory.
//!
//! Delegates the simulation to `crate::des::general::control_systems::dc_motor`
//! + the iterative DES runner (identical wiring to `main_dc_motor`).
//!
//! PORT NOTE: the HTML animation is produced in TS by `FrameRecorder` +
//! `animation/scenes/dc-motor-scene`. `animation::scenes::dc_motor_scene` is NOT
//! yet ported (the `animation/scenes` directory has no `dc_motor_scene.rs`), so
//! the rendering step is stubbed: we run the simulation, compute the closed-loop
//! reference series, and print a note. Wire `DcMotorScene` + `FrameRecorder`
//! (`crate::des::animation::frame_recorder`) once the scene exists.

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

struct DcMotorAnimator {
    params: DcMotorParams,
    dt: f64,
}

impl DcMotorAnimator {
    fn new() -> Self {
        DcMotorAnimator {
            params: DcMotorParams {
                resistance: 2.0,
                inductance: 0.5,
                back_emf_constant: 0.1,
                torque_constant: 0.1,
                inertia: 0.02,
                friction: 0.002,
            },
            dt: 0.005,
        }
    }

    fn run(&self, mode: Mode) {
        match mode {
            Mode::Open => self.run_open(),
            Mode::Closed => self.run_closed(),
        }
    }

    fn run_open(&self) {
        let steps = 3000usize;
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: self.params.clone(),
                dt: self.dt,
                steps,
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
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(steps + 5),
                ..Default::default()
            },
        );

        let samples = sink.borrow().samples.clone();
        self.record(&samples, None, "open", 8);
    }

    fn run_closed(&self) {
        let steps = 6000usize;
        let load = LoadProfile::new(&[
            LoadSegment {
                from_time: 0.0,
                torque: 0.0,
            },
            LoadSegment {
                from_time: 18.0,
                torque: 0.3,
            },
        ]);
        let plant = Rc::new(RefCell::new(DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: self.params.clone(),
                dt: self.dt,
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
                dt: self.dt,
                max_voltage: Some(48.0),
                reference: vec![
                    SpeedReferenceSegment {
                        from_time: 0.0,
                        speed: 60.0,
                    },
                    SpeedReferenceSegment {
                        from_time: 10.0,
                        speed: 100.0,
                    },
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
                max_ticks: Some(steps + 5),
                ..Default::default()
            },
        );

        let samples = sink.borrow().samples.clone();
        let reference: Vec<f64> = samples
            .iter()
            .map(|s| controller.borrow().reference_at(s.time))
            .collect();
        self.record(&samples, Some(reference), "closed", 15);
    }

    /// PORT NOTE: stands in for `FrameRecorder` + `DcMotorScene`. Reports the
    /// trajectory that would have been rendered.
    fn record(
        &self,
        samples: &[Rc<MotorStateToken>],
        reference: Option<Vec<f64>>,
        tag: &str,
        stride: usize,
    ) {
        let frames = samples.len().div_ceil(stride.max(1));
        let ref_note = match &reference {
            Some(r) => format!(", reference series len={}", r.len()),
            None => String::new(),
        };
        println!(
            "DC-motor animation ({tag}): omitted in Rust port — {} samples, ~{} frames @ stride {}{} (see PORT NOTE)",
            samples.len(),
            frames,
            stride,
            ref_note
        );
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    let mode = if std::env::var("MODE").unwrap_or_default().to_lowercase() == "open" {
        Mode::Open
    } else {
        Mode::Closed
    };
    DcMotorAnimator::new().run(mode);
}
