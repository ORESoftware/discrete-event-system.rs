//! Port of `src/des/main-dc-motor-anim.ts`.
//!
//! Drives the back-EMF DC-motor DES (open-loop step vs closed-loop PI speed
//! control) and renders an HTML animation of the trajectory.
//!
//! Delegates the simulation to `crate::des::general::control_systems::dc_motor`
//! + the iterative DES runner (identical wiring to `main_dc_motor`).
//!
//! The generated `out/dc-motor/shadow-observability-controllability.*`
//! artifacts are the dual/shadow evaluation for this simulation: they assess
//! the exact state-space plant used by the animator for controllability and
//! observability degree, rather than depending on visual inspection.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::Path;
use std::rc::Rc;

use serde_json::{json, Value};

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::dc_motor_scene::{
    self as scene, DcMotorScene, DcMotorSceneOpts, MOTOR_STAGE_H, MOTOR_STAGE_W,
};
use crate::des::general::control_systems::dc_motor::DcMotorDynamics;
use crate::des::general::control_systems::dc_motor::{
    DcMotorChannels, DcMotorParams, DcMotorPlantOpts, DcMotorPlantStation, DcMotorSinkStation,
    LoadProfile, LoadSegment, MotorStateToken, SpeedPiVoltageController, SpeedPiVoltageOpts,
    SpeedReferenceSegment,
};
use crate::des::general::control_systems::empirical_control::{
    ControllabilityGramian, DiscreteLinearSystem, ObservabilityGramian,
};
use crate::des::general::control_systems::observability_controllability::{
    StateSpaceModel, StateSpaceSpec,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};
use crate::des::model::RunArtifact;

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
        self.write_shadow_assessment()
            .expect("write DC motor shadow observability/controllability report");
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
        self.record(&samples, None, "open", 8)
            .expect("write DC motor open-loop animation");
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
        self.record(&samples, Some(reference), "closed", 15)
            .expect("write DC motor closed-loop animation");
    }

    fn scene_params(&self) -> scene::DcMotorParams {
        scene::DcMotorParams {
            resistance: self.params.resistance,
            inductance: self.params.inductance,
        }
    }

    fn to_scene_samples(samples: &[Rc<MotorStateToken>]) -> Vec<scene::MotorStateToken> {
        samples
            .iter()
            .map(|s| scene::MotorStateToken {
                time: s.time,
                omega: s.omega,
                current: s.current,
                voltage: s.voltage,
                back_emf: s.back_emf,
                load_torque: s.load_torque,
            })
            .collect()
    }

    fn animation_paths(tag: &str) -> (String, String) {
        let dir = Path::new("out").join("dc-motor");
        let frames = dir.join(format!("animation-{tag}.frames.jsonl"));
        let html = dir.join(format!("animation-{tag}.html"));
        (
            frames.to_string_lossy().into_owned(),
            html.to_string_lossy().into_owned(),
        )
    }

    fn record(
        &self,
        samples: &[Rc<MotorStateToken>],
        reference: Option<Vec<f64>>,
        tag: &str,
        stride: usize,
    ) -> io::Result<()> {
        if samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "DC motor animation has no samples to render",
            ));
        }
        let (frames_path, html_path) = Self::animation_paths(tag);
        let mode_name = match tag {
            "open" => "open-loop 12 V step".to_string(),
            "closed" => "closed-loop PI speed control".to_string(),
            other => other.to_string(),
        };
        let scene = DcMotorScene::new(DcMotorSceneOpts {
            samples: Self::to_scene_samples(samples),
            dt: self.dt,
            params: self.scene_params(),
            mode_name,
            reference,
        });
        let mut recorder = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.clone(),
            html_path: Some(html_path.clone()),
            width: MOTOR_STAGE_W,
            height: MOTOR_STAGE_H,
            fps: Some(30.0),
            title: Some(format!("DC Motor — {tag}")),
            subtitle: Some(
                "Back-EMF ODE plant with armature current, voltage, speed, and load torque."
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
            "DC-motor animation ({tag}): {} samples, {} recorded frames -> {}",
            samples.len(),
            anim.frames.len().max(recorded as usize),
            html_path
        );
        Ok(())
    }

    fn state_space_model(&self) -> StateSpaceModel {
        let matrices = DcMotorDynamics::new(self.params.clone()).state_space();
        StateSpaceModel::new(StateSpaceSpec {
            a: matrices.a,
            b: matrices.b,
            c: matrices.c,
            d: Some(matrices.d),
        })
    }

    fn shadow_assessment(&self) -> Value {
        let model = self.state_space_model();
        let horizon = (2.0 / self.dt).round() as usize;
        let discrete = DiscreteLinearSystem::from_continuous(&model, self.dt);
        let wc = ControllabilityGramian::new(&discrete, horizon);
        let wo = ObservabilityGramian::new(&discrete, horizon);
        let c_rank = model.controllability_rank();
        let o_rank = model.observability_rank();
        json!({
            "kind": "dc-motor-shadow-observability-controllability",
            "simulation": "dc-motor/back-emf",
            "evaluationMode": "dual shadow LTI evaluator",
            "state": ["armature current i", "rotor speed omega"],
            "input": ["armature voltage V"],
            "output": ["rotor speed omega"],
            "dt": self.dt,
            "horizonSteps": horizon,
            "continuousStateSpace": {
                "a": model.a,
                "b": model.b,
                "c": model.c,
                "d": model.d,
            },
            "ltiRankTest": {
                "stateDim": model.state_dim(),
                "inputDim": model.input_dim(),
                "outputDim": model.output_dim(),
                "controllabilityRank": c_rank,
                "observabilityRank": o_rank,
                "controllable": model.is_controllable(),
                "observable": model.is_observable(),
            },
            "controllabilityDegree": {
                "eigenvalues": Self::finite_array(&wc.eigenvalues()),
                "min": Self::finite_or_string(wc.min()),
                "max": Self::finite_or_string(wc.max()),
                "conditionNumber": Self::finite_or_string(wc.condition_number()),
                "weakestDirection": Self::finite_array(&wc.weakest_direction()),
                "strongestDirection": Self::finite_array(&wc.strongest_direction()),
            },
            "observabilityDegree": {
                "eigenvalues": Self::finite_array(&wo.eigenvalues()),
                "min": Self::finite_or_string(wo.min()),
                "max": Self::finite_or_string(wo.max()),
                "conditionNumber": Self::finite_or_string(wo.condition_number()),
                "weakestDirection": Self::finite_array(&wo.weakest_direction()),
                "strongestDirection": Self::finite_array(&wo.strongest_direction()),
            },
            "mdpPomdpNote": {
                "nestedModelUsed": false,
                "reason": "This plant is a fully-observed two-state LTI system for the default speed sensor, so the shadow evaluator uses Kalman rank tests and Gramians directly.",
                "whenToNest": "Use a nested MDP/POMDP layer when controller modes, faults, partially observed sensors, or learned policies become part of the simulation state."
            }
        })
    }

    fn write_shadow_assessment(&self) -> io::Result<()> {
        let dir = Path::new("out").join("dc-motor");
        fs::create_dir_all(&dir)?;
        let result = self.shadow_assessment();
        let json_path = dir.join("shadow-observability-controllability.json");
        let html_path = dir.join("shadow-observability-controllability.html");
        let text = serde_json::to_string_pretty(&result).map_err(io::Error::other)?;
        fs::write(&json_path, text)?;
        let artifact = RunArtifact::results(
            "dc-motor-shadow-observability-controllability",
            "DC Motor Shadow Observability / Controllability",
            "Dual evaluation of the same back-EMF state-space plant used by the DC motor animation.",
            result,
            vec![],
            "Shadow evaluator confirms the default DC motor plant is controllable and observable, with Gramian degree metrics for weak/strong directions.",
        );
        fs::write(&html_path, artifact.to_player_html())?;
        println!(
            "DC-motor shadow assessment: {} and {}",
            json_path.display(),
            html_path.display()
        );
        Ok(())
    }

    fn finite_array(values: &[f64]) -> Vec<Value> {
        values.iter().copied().map(Self::finite_or_string).collect()
    }

    fn finite_or_string(value: f64) -> Value {
        if value.is_finite() {
            json!(value)
        } else {
            json!("infinity")
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animation_paths_match_site_links() {
        let (_, closed_html) = DcMotorAnimator::animation_paths("closed");
        let (_, open_html) = DcMotorAnimator::animation_paths("open");
        assert!(closed_html.ends_with("out/dc-motor/animation-closed.html"));
        assert!(open_html.ends_with("out/dc-motor/animation-open.html"));
    }

    #[test]
    fn shadow_assessment_confirms_default_motor_is_full_rank() {
        let assessment = DcMotorAnimator::new().shadow_assessment();
        let rank = &assessment["ltiRankTest"];
        assert_eq!(rank["stateDim"].as_u64(), Some(2));
        assert_eq!(rank["controllabilityRank"].as_u64(), Some(2));
        assert_eq!(rank["observabilityRank"].as_u64(), Some(2));
        assert_eq!(rank["controllable"].as_bool(), Some(true));
        assert_eq!(rank["observable"].as_bool(), Some(true));
    }
}
