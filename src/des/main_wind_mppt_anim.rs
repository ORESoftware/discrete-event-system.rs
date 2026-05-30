//! Port of `src/des/main-wind-mppt-anim.ts`.
//!
//! Generates an HTML animation of the wind-MPPT DES. Wires the self-clocking
//! turbine plant to an MPPT controller (optimal-torque or PI speed-loop) and a
//! trajectory sink — identical to `main_wind_mppt` — then would render the
//! captured samples via `FrameRecorder` + `WindMpptScene`.
//!
//! Conversion notes:
//!   - `class WindMpptAnimator` → struct + impl; async `run()` → [`run`].
//!   - `process.env.CONTROLLER` → `std::env`.
//!
//! PORT NOTE: the HTML render uses `animation/scenes/wind-mppt-scene`, which is
//! NOT yet ported (`animation::scenes` has no `wind_mppt_scene.rs`). As with
//! `main_dc_motor_anim`, the rendering step is stubbed: the DES is run faithfully
//! and the trajectory that would have been drawn is reported. Wire
//! `WindMpptScene` + `crate::des::animation::frame_recorder::FrameRecorder` once
//! the scene exists.

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::control_systems::wind_mppt::{
    OptimalTorqueMpptController, SpeedPiMpptController, SpeedPiMpptOpts, WindMpptChannels,
    WindMpptSinkStation, WindProfile, WindProfileSegment, WindTurbineAeroOpts,
    WindTurbineAerodynamics, WindTurbinePlantOpts, WindTurbinePlantStation,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

struct WindMpptAnimator {
    aero: WindTurbineAerodynamics,
    dt: f64,
    steps: usize,
}

impl WindMpptAnimator {
    fn new() -> Self {
        WindMpptAnimator {
            aero: WindTurbineAerodynamics::new(WindTurbineAeroOpts {
                air_density: None,
                blade_radius: 2.5,
                pitch_deg: Some(0.0),
            }),
            dt: 0.05,
            steps: 1200,
        }
    }

    fn run(&self, kind: &str) {
        let wind_profile = WindProfile::new(&[
            WindProfileSegment { from_time: 0.0, speed: 8.0 },
            WindProfileSegment { from_time: 20.0, speed: 11.0 },
            WindProfileSegment { from_time: 40.0, speed: 9.0 },
        ]);

        let plant = Rc::new(RefCell::new(WindTurbinePlantStation::new(
            "turbine",
            WindTurbinePlantOpts {
                aero: self.aero.clone(),
                wind_profile,
                inertia: 6.0,
                friction: 0.02,
                dt: self.dt,
                steps: self.steps,
                initial_omega: 2.0,
            },
        )));

        let controller: StationRef = if kind == "pi" {
            Rc::new(RefCell::new(SpeedPiMpptController::new(
                "mppt-pi",
                &self.aero,
                SpeedPiMpptOpts { kp: 8.0, ki: 4.0, dt: self.dt, max_torque: None },
            )))
        } else {
            Rc::new(RefCell::new(OptimalTorqueMpptController::new("mppt-opt-torque", &self.aero)))
        };

        let sink = Rc::new(RefCell::new(WindMpptSinkStation::new("sink")));

        let plant_ref: StationRef = plant.clone();
        let sink_ref: StationRef = sink.clone();

        plant.borrow_mut().core_mut().pipe(
            controller.clone(),
            WindMpptChannels::STATE,
            WindMpptChannels::STATE,
        );
        plant.borrow_mut().core_mut().pipe(
            sink_ref.clone(),
            WindMpptChannels::STATE,
            WindMpptChannels::STATE,
        );
        controller.borrow_mut().core_mut().pipe(
            plant_ref.clone(),
            WindMpptChannels::TORQUE,
            WindMpptChannels::TORQUE,
        );

        run_iterative_des(
            vec![plant_ref, controller, sink_ref],
            IterativeRunOptions { shuffle: false, max_ticks: Some(self.steps + 5), ..Default::default() },
        );

        let controller_name = if kind == "pi" { "PI speed loop" } else { "optimal torque" };
        self.record(kind, controller_name, &sink.borrow());
    }

    /// PORT NOTE: stands in for `FrameRecorder` + `WindMpptScene`. Reports the
    /// trajectory that would have been rendered.
    fn record(&self, kind: &str, controller_name: &str, sink: &WindMpptSinkStation) {
        let stride = 3usize; // 1200 samples → ~400 frames @ 30 fps
        let sample_count = sink.samples.len();
        let frames = sample_count.div_ceil(stride.max(1));
        let out = std::path::Path::new("out").join("wind-mppt").join(format!("animation-{}.html", kind));
        println!(
            "Wind-MPPT animation ({}): omitted in Rust port — {} samples, ~{} frames @ stride {} (λ*={:.2}, C_p,max={:.3}); would write {} (see PORT NOTE)",
            controller_name,
            sample_count,
            frames,
            stride,
            self.aero.optimal_tip_speed_ratio(),
            self.aero.max_power_coefficient(),
            out.display()
        );
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    let kind = if std::env::var("CONTROLLER").unwrap_or_default().to_lowercase() == "pi" {
        "pi"
    } else {
        "optimal-torque"
    };
    WindMpptAnimator::new().run(kind);
}
