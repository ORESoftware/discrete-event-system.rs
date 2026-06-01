//! Port of `src/des/main-wind-mppt-anim.ts`.
//!
//! Generates an HTML animation of the wind-MPPT DES. Wires the self-clocking
//! turbine plant to an MPPT controller (optimal-torque or PI speed-loop) and a
//! trajectory sink — identical to `main_wind_mppt` — then renders the captured
//! samples via `FrameRecorder` + `WindMpptScene`.
//!
//! Conversion notes:
//!   - `class WindMpptAnimator` → struct + impl; async `run()` → [`run`].
//!   - `process.env.CONTROLLER` → `std::env`.
//!
use std::cell::RefCell;
use std::io;
use std::path::Path;
use std::rc::Rc;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::wind_mppt_scene::{
    self as scene, WindMpptScene, WindSceneOpts, WIND_STAGE_H, WIND_STAGE_W,
};
use crate::des::general::control_systems::wind_mppt::{
    OptimalTorqueMpptController, SpeedPiMpptController, SpeedPiMpptOpts,
    TurbineStateToken as ModelTurbineStateToken, WindMpptChannels, WindMpptSinkStation,
    WindProfile, WindProfileSegment, WindTurbineAeroOpts, WindTurbineAerodynamics,
    WindTurbinePlantOpts, WindTurbinePlantStation,
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
                SpeedPiMpptOpts {
                    kp: 8.0,
                    ki: 4.0,
                    dt: self.dt,
                    max_torque: None,
                },
            )))
        } else {
            Rc::new(RefCell::new(OptimalTorqueMpptController::new(
                "mppt-opt-torque",
                &self.aero,
            )))
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
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(self.steps + 5),
                ..Default::default()
            },
        );

        let controller_name = if kind == "pi" {
            "PI speed loop"
        } else {
            "optimal torque"
        };
        self.record(kind, controller_name, &sink.borrow())
            .expect("write Wind MPPT animation");
    }

    fn to_scene_samples(samples: &[Rc<ModelTurbineStateToken>]) -> Vec<scene::TurbineStateToken> {
        samples
            .iter()
            .map(|s| scene::TurbineStateToken {
                time: s.time,
                omega: s.omega,
                wind_speed: s.wind_speed,
                lambda: s.lambda,
                cp: s.cp,
                mech_power: s.mech_power,
                gen_torque: s.gen_torque,
            })
            .collect()
    }

    fn animation_paths(kind: &str) -> (String, String) {
        let dir = Path::new("out").join("wind-mppt");
        let frames = dir.join(format!("animation-{kind}.frames.jsonl"));
        let html = dir.join(format!("animation-{kind}.html"));
        (
            frames.to_string_lossy().into_owned(),
            html.to_string_lossy().into_owned(),
        )
    }

    fn record(
        &self,
        kind: &str,
        controller_name: &str,
        sink: &WindMpptSinkStation,
    ) -> io::Result<()> {
        if sink.samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Wind MPPT animation has no samples to render",
            ));
        }
        let stride = 3usize; // 1200 samples -> ~400 frames @ 30 fps
        let (frames_path, html_path) = Self::animation_paths(kind);
        let scene = WindMpptScene::new(WindSceneOpts {
            samples: Self::to_scene_samples(&sink.samples),
            dt: self.dt,
            lambda_star: self.aero.optimal_tip_speed_ratio(),
            cp_max: self.aero.max_power_coefficient(),
            k_opt: self.aero.optimal_torque_gain(),
            controller_name: controller_name.to_string(),
        });
        let mut recorder = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.clone(),
            html_path: Some(html_path.clone()),
            width: WIND_STAGE_W,
            height: WIND_STAGE_H,
            fps: Some(30.0),
            title: Some(format!("Wind MPPT — {controller_name}")),
            subtitle: Some(
                "Variable-speed PMSG turbine with wind steps, tip-speed ratio tracking, C_p capture, and generator torque."
                    .to_string(),
            ),
            background: Some("#0b1021".to_string()),
            live_tick_line: Some(false),
            record_every_ticks: Some(stride.max(1) as f64),
            visual_blocks: None,
        })?;
        recorder.set_charts(scene.charts());
        for i in 0..scene.frame_count() {
            recorder.frame(scene.time_at(i), i as f64, || scene.frame_at(i));
        }
        let recorded = recorder.get_frame_count();
        let anim = recorder.finish()?;
        println!(
            "Wind-MPPT animation ({}): {} samples, {} recorded frames -> {}",
            controller_name,
            sink.samples.len(),
            anim.frames.len().max(recorded as usize),
            html_path
        );
        Ok(())
    }
}

pub fn run_controller(kind: &str) {
    let normalized = if kind.eq_ignore_ascii_case("pi") {
        "pi"
    } else {
        "optimal-torque"
    };
    WindMpptAnimator::new().run(normalized);
}

/// Entry point (TS top-level script).
pub fn run() {
    let kind = if std::env::var("CONTROLLER")
        .unwrap_or_default()
        .to_lowercase()
        == "pi"
    {
        "pi"
    } else {
        "optimal-torque"
    };
    run_controller(kind);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animation_paths_match_site_links() {
        let (_, opt_html) = WindMpptAnimator::animation_paths("optimal-torque");
        let (_, pi_html) = WindMpptAnimator::animation_paths("pi");
        assert!(opt_html.ends_with("out/wind-mppt/animation-optimal-torque.html"));
        assert!(pi_html.ends_with("out/wind-mppt/animation-pi.html"));
    }
}
