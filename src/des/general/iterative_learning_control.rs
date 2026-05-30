//! Port of `src/des/general/iterative-learning-control.ts` — Iterative Learning
//! Control (ILC) as an explicit DES station graph.
//!
//! A repeated-trial controller learns a feedforward control sequence for a plant
//! that must track the same reference trajectory on every trial:
//!
//!   u_{j+1}[k] = sat(u_j[k] + L * e_j[k + 1])
//!
//! where j is the trial index and k the time index inside the trial. The model
//! is deliberately expressed as source/station/sink pieces: a trial source feeds
//! a controller-program station, which feeds a plant, whose results fan out to a
//! learning-update station (closing the loop) and a result sink.
//!
//! ## Rust shape
//!
//! Each `class …Station extends DESStation` → a struct embedding [`StationCore`]
//! and implementing the [`DESStation`] trait (overriding `has_work` /
//! `run_time_step`). The payload-carrying token classes become plain structs
//! travelling as `Rc<dyn Any>`. `runIterativeLearningControl` stays a free fn.
//! `throw` on a bad trial count → `panic!`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};

// NOTE (dep flag): the TS file imports `stationGraph`/`channelEdge`/
// `StationGraphSummary` from the `des-base` barrel. In the Rust port those
// topology helpers live in `des_base::learning_optimization`, so they are
// imported from there.

const CH_TRIAL: &str = "trial-plan";
const CH_PROGRAM: &str = "control-program";
const CH_RESULT: &str = "trial-result";

/// `type ILCReferenceKind = 'sine' | 'step' | 'ramp'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ILCReferenceKind {
    Sine,
    Step,
    Ramp,
}

/// Tunable parameters for [`run_iterative_learning_control`]; `None` ⇒ default.
#[derive(Clone, Debug, Default)]
pub struct IterativeLearningControlParams {
    pub trials: Option<usize>,
    pub horizon: Option<usize>,
    pub dt: Option<f64>,
    pub plant_rate: Option<f64>,
    pub plant_gain: Option<f64>,
    pub learning_gain: Option<f64>,
    pub feedback_gain: Option<f64>,
    pub control_max: Option<f64>,
    pub reference_kind: Option<ILCReferenceKind>,
    pub reference_amplitude: Option<f64>,
    pub initial_output: Option<f64>,
}

/// Per-trial scalar diagnostics.
#[derive(Clone, Debug)]
pub struct ILCTrialSummary {
    pub trial: usize,
    pub rms_error: f64,
    pub max_abs_error: f64,
    pub max_abs_control: f64,
    pub final_output: f64,
    pub final_reference: f64,
}

/// Output of a full ILC run.
#[derive(Clone, Debug)]
pub struct IterativeLearningControlResult {
    pub reference_trajectory: Vec<f64>,
    pub trial_summaries: Vec<ILCTrialSummary>,
    pub initial_rms_error: f64,
    pub final_rms_error: f64,
    pub improvement_ratio: f64,
    pub final_output_trajectory: Vec<f64>,
    pub final_control_sequence: Vec<f64>,
    pub final_feedforward_sequence: Vec<f64>,
    pub topology: StationGraphSummary,
}

// ── Tokens ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ILCTrialPlanToken {
    trial: usize,
    reference: Vec<f64>,
    feedforward: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ILCControlProgramToken {
    trial: usize,
    reference: Vec<f64>,
    feedforward: Vec<f64>,
    feedback_gain: f64,
    control_max: f64,
}

#[derive(Clone, Debug)]
struct ILCTrialResultToken {
    trial: usize,
    reference: Vec<f64>,
    feedforward: Vec<f64>,
    controls: Vec<f64>,
    output: Vec<f64>,
    #[allow(dead_code)]
    errors: Vec<f64>,
    rms_error: f64,
    max_abs_error: f64,
    max_abs_control: f64,
}

// ── Stations ───────────────────────────────────────────────────────────────────

struct ILCTrialSourceStation {
    core: StationCore,
    reference: Vec<f64>,
    horizon: usize,
    emitted: bool,
}

impl ILCTrialSourceStation {
    fn new(id: &str, reference: Vec<f64>, horizon: usize) -> Self {
        ILCTrialSourceStation { core: StationCore::new(id), reference, horizon, emitted: false }
    }
}

impl DESStation for ILCTrialSourceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let token = ILCTrialPlanToken {
            trial: 0,
            reference: self.reference.clone(),
            feedforward: vec![0.0; self.horizon],
        };
        self.core.emit(Rc::new(token), CH_TRIAL);
        self.emitted = true;
    }
}

struct ILCControllerProgramStation {
    core: StationCore,
    feedback_gain: f64,
    control_max: f64,
}

impl ILCControllerProgramStation {
    fn new(id: &str, feedback_gain: f64, control_max: f64) -> Self {
        ILCControllerProgramStation { core: StationCore::new(id), feedback_gain, control_max }
    }
}

impl DESStation for ILCControllerProgramStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_TRIAL) > 0
    }
    fn run_time_step(&mut self) {
        let trials = self.core.drain::<ILCTrialPlanToken>(CH_TRIAL);
        for trial in trials {
            let token = ILCControlProgramToken {
                trial: trial.trial,
                reference: trial.reference.clone(),
                feedforward: trial.feedforward.clone(),
                feedback_gain: self.feedback_gain,
                control_max: self.control_max,
            };
            self.core.emit(Rc::new(token), CH_PROGRAM);
        }
    }
}

struct ILCPlantTrialStation {
    core: StationCore,
    plant_rate: f64,
    plant_gain: f64,
    dt: f64,
    initial_output: f64,
}

impl ILCPlantTrialStation {
    fn new(id: &str, plant_rate: f64, plant_gain: f64, dt: f64, initial_output: f64) -> Self {
        ILCPlantTrialStation { core: StationCore::new(id), plant_rate, plant_gain, dt, initial_output }
    }

    fn run_trial(&self, program: &ILCControlProgramToken) -> ILCTrialResultToken {
        let horizon = program.feedforward.len();
        let mut y = self.initial_output;
        let mut output = vec![y];
        let mut controls: Vec<f64> = Vec::new();
        let mut errors: Vec<f64> = Vec::new();

        for k in 0..horizon {
            let error = program.reference[k] - y;
            let u = clamp(
                program.feedforward[k] + program.feedback_gain * error,
                -program.control_max,
                program.control_max,
            );
            controls.push(u);
            errors.push(error);
            y += self.dt * (-self.plant_rate * y + self.plant_gain * u);
            output.push(y);
        }

        let rms_error = rms(&errors);
        let max_abs_error = errors.iter().fold(0.0_f64, |acc, e| acc.max(e.abs()));
        let max_abs_control = controls.iter().fold(0.0_f64, |acc, u| acc.max(u.abs()));
        ILCTrialResultToken {
            trial: program.trial,
            reference: program.reference.clone(),
            feedforward: program.feedforward.clone(),
            controls,
            output,
            errors,
            rms_error,
            max_abs_error,
            max_abs_control,
        }
    }
}

impl DESStation for ILCPlantTrialStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_PROGRAM) > 0
    }
    fn run_time_step(&mut self) {
        let programs = self.core.drain::<ILCControlProgramToken>(CH_PROGRAM);
        for program in programs {
            let result = self.run_trial(&program);
            self.core.emit(Rc::new(result), CH_RESULT);
        }
    }
}

struct ILCLearningUpdateStation {
    core: StationCore,
    max_trials: usize,
    learning_gain: f64,
    control_max: f64,
}

impl ILCLearningUpdateStation {
    fn new(id: &str, max_trials: usize, learning_gain: f64, control_max: f64) -> Self {
        ILCLearningUpdateStation { core: StationCore::new(id), max_trials, learning_gain, control_max }
    }
}

impl DESStation for ILCLearningUpdateStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let results = self.core.drain::<ILCTrialResultToken>(CH_RESULT);
        for result in results {
            let next_trial = result.trial + 1;
            if next_trial >= self.max_trials {
                continue;
            }
            let next_feedforward: Vec<f64> = result
                .feedforward
                .iter()
                .enumerate()
                .map(|(k, &u)| {
                    let next_error = result.reference[k + 1] - result.output[k + 1];
                    clamp(u + self.learning_gain * next_error, -self.control_max, self.control_max)
                })
                .collect();
            let token = ILCTrialPlanToken {
                trial: next_trial,
                reference: result.reference.clone(),
                feedforward: next_feedforward,
            };
            self.core.emit(Rc::new(token), CH_TRIAL);
        }
    }
}

struct ILCResultSinkStation {
    core: StationCore,
    results: Vec<Rc<ILCTrialResultToken>>,
}

impl ILCResultSinkStation {
    fn new(id: &str) -> Self {
        ILCResultSinkStation { core: StationCore::new(id), results: Vec::new() }
    }
}

impl DESStation for ILCResultSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let drained = self.core.drain::<ILCTrialResultToken>(CH_RESULT);
        self.results.extend(drained);
    }
}

/// Run the ILC station graph and reduce to scalar/series diagnostics.
pub fn run_iterative_learning_control(
    params: &IterativeLearningControlParams,
) -> IterativeLearningControlResult {
    let trials = params.trials.unwrap_or(30);
    let horizon = params.horizon.unwrap_or(80);
    let dt = params.dt.unwrap_or(0.1);
    let plant_rate = params.plant_rate.unwrap_or(1.2);
    let plant_gain = params.plant_gain.unwrap_or(1.0);
    let learning_gain = params.learning_gain.unwrap_or(0.8);
    let feedback_gain = params.feedback_gain.unwrap_or(0.8);
    let control_max = params.control_max.unwrap_or(5.0);
    let reference_kind = params.reference_kind.unwrap_or(ILCReferenceKind::Sine);
    let reference_amplitude = params.reference_amplitude.unwrap_or(1.0);
    let initial_output = params.initial_output.unwrap_or(0.0);

    let cls = "runIterativeLearningControl";
    Preconditions::integer_in_range(cls, "trials", trials as f64, 1.0, 1e6).expect("trials");
    Preconditions::integer_in_range(cls, "horizon", horizon as f64, 2.0, 1e6).expect("horizon");
    Preconditions::positive(cls, "dt", dt).expect("dt");
    Preconditions::positive(cls, "plantRate", plant_rate).expect("plantRate");
    Preconditions::positive(cls, "plantGain", plant_gain).expect("plantGain");
    Preconditions::in_range(cls, "learningGain", learning_gain, 0.0, 2.0).expect("learningGain");
    Preconditions::non_negative(cls, "feedbackGain", feedback_gain).expect("feedbackGain");
    Preconditions::positive(cls, "controlMax", control_max).expect("controlMax");
    Preconditions::non_negative(cls, "referenceAmplitude", reference_amplitude)
        .expect("referenceAmplitude");
    Preconditions::finite(cls, "initialOutput", initial_output).expect("initialOutput");

    let reference = build_reference(reference_kind, horizon, reference_amplitude);

    let source = Rc::new(RefCell::new(ILCTrialSourceStation::new(
        "ilc-trial-source",
        reference.clone(),
        horizon,
    )));
    let controller = Rc::new(RefCell::new(ILCControllerProgramStation::new(
        "ilc-controller-program-station",
        feedback_gain,
        control_max,
    )));
    let plant = Rc::new(RefCell::new(ILCPlantTrialStation::new(
        "ilc-plant-trial-station",
        plant_rate,
        plant_gain,
        dt,
        initial_output,
    )));
    let learner = Rc::new(RefCell::new(ILCLearningUpdateStation::new(
        "ilc-learning-update-station",
        trials,
        learning_gain,
        control_max,
    )));
    let sink = Rc::new(RefCell::new(ILCResultSinkStation::new("ilc-result-sink")));

    source
        .borrow_mut()
        .core_mut()
        .pipe(controller.clone() as StationRef, CH_TRIAL, CH_TRIAL);
    controller
        .borrow_mut()
        .core_mut()
        .pipe(plant.clone() as StationRef, CH_PROGRAM, CH_PROGRAM);
    plant
        .borrow_mut()
        .core_mut()
        .pipe(learner.clone() as StationRef, CH_RESULT, CH_RESULT);
    plant
        .borrow_mut()
        .core_mut()
        .pipe(sink.clone() as StationRef, CH_RESULT, CH_RESULT);
    learner
        .borrow_mut()
        .core_mut()
        .pipe(controller.clone() as StationRef, CH_TRIAL, CH_TRIAL);

    let stations: Vec<StationRef> = vec![
        source.clone(),
        controller.clone(),
        plant.clone(),
        learner.clone(),
        sink.clone(),
    ];
    run_iterative_des(
        stations,
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(trials + 5),
            run_validators: false,
            ..Default::default()
        },
    );

    let sink_ref = sink.borrow();
    if sink_ref.results.len() != trials {
        panic!(
            "iterative-learning-control produced {} trials, expected {}",
            sink_ref.results.len(),
            trials
        );
    }

    let first = sink_ref.results[0].clone();
    let last = sink_ref.results[sink_ref.results.len() - 1].clone();

    let topology = station_graph(
        &[
            StationOrId::Id("ilc-trial-source".to_string()),
            StationOrId::Id("ilc-controller-program-station".to_string()),
            StationOrId::Id("ilc-plant-trial-station".to_string()),
            StationOrId::Id("ilc-learning-update-station".to_string()),
            StationOrId::Id("ilc-result-sink".to_string()),
        ],
        &[
            "ILCTrialPlanToken".to_string(),
            "ILCControlProgramToken".to_string(),
            "ILCTrialResultToken".to_string(),
        ],
        &[
            channel_edge(
                &StationOrId::Id("ilc-trial-source".to_string()),
                CH_TRIAL,
                &StationOrId::Id("ilc-controller-program-station".to_string()),
                Some(CH_TRIAL),
            ),
            channel_edge(
                &StationOrId::Id("ilc-controller-program-station".to_string()),
                CH_PROGRAM,
                &StationOrId::Id("ilc-plant-trial-station".to_string()),
                Some(CH_PROGRAM),
            ),
            channel_edge(
                &StationOrId::Id("ilc-plant-trial-station".to_string()),
                CH_RESULT,
                &StationOrId::Id("ilc-learning-update-station".to_string()),
                Some(CH_RESULT),
            ),
            channel_edge(
                &StationOrId::Id("ilc-plant-trial-station".to_string()),
                CH_RESULT,
                &StationOrId::Id("ilc-result-sink".to_string()),
                Some(CH_RESULT),
            ),
            channel_edge(
                &StationOrId::Id("ilc-learning-update-station".to_string()),
                CH_TRIAL,
                &StationOrId::Id("ilc-controller-program-station".to_string()),
                Some(CH_TRIAL),
            ),
        ],
    );

    IterativeLearningControlResult {
        reference_trajectory: reference.clone(),
        trial_summaries: sink_ref.results.iter().map(|r| to_summary(r)).collect(),
        initial_rms_error: first.rms_error,
        final_rms_error: last.rms_error,
        improvement_ratio: last.rms_error / first.rms_error.max(1e-12),
        final_output_trajectory: last.output.clone(),
        final_control_sequence: last.controls.clone(),
        final_feedforward_sequence: last.feedforward.clone(),
        topology,
    }
}

fn to_summary(result: &ILCTrialResultToken) -> ILCTrialSummary {
    ILCTrialSummary {
        trial: result.trial,
        rms_error: result.rms_error,
        max_abs_error: result.max_abs_error,
        max_abs_control: result.max_abs_control,
        final_output: *result.output.last().unwrap(),
        final_reference: *result.reference.last().unwrap(),
    }
}

fn build_reference(kind: ILCReferenceKind, horizon: usize, amplitude: f64) -> Vec<f64> {
    let denom = ((horizon as f64) - 1.0).max(1.0);
    (0..=horizon)
        .map(|k| {
            let kf = k as f64;
            match kind {
                ILCReferenceKind::Step => {
                    if k < (0.15 * horizon as f64).floor() as usize {
                        0.0
                    } else {
                        amplitude
                    }
                }
                ILCReferenceKind::Ramp => amplitude * kf / horizon as f64,
                ILCReferenceKind::Sine => {
                    let phase = 2.0 * std::f64::consts::PI * kf / denom;
                    amplitude * (phase.sin() + 0.4 * (2.0 * phase).sin())
                }
            }
        })
        .collect()
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

fn rms(xs: &[f64]) -> f64 {
    (xs.iter().map(|x| x * x).sum::<f64>() / (xs.len().max(1) as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    //! ILC tests: the learner must drive the trial-to-trial tracking error down,
    //! and the run must produce exactly `trials` recorded trials.

    use super::*;

    #[test]
    fn reduces_tracking_error_across_trials() {
        let res = run_iterative_learning_control(&IterativeLearningControlParams {
            trials: Some(20),
            horizon: Some(40),
            ..Default::default()
        });
        // Learning should shrink the RMS tracking error substantially.
        assert!(
            res.final_rms_error < res.initial_rms_error,
            "no improvement: {} -> {}",
            res.initial_rms_error,
            res.final_rms_error
        );
        assert!(res.improvement_ratio < 1.0);
    }

    #[test]
    fn produces_expected_trial_count() {
        let trials = 12;
        let res = run_iterative_learning_control(&IterativeLearningControlParams {
            trials: Some(trials),
            horizon: Some(20),
            reference_kind: Some(ILCReferenceKind::Step),
            ..Default::default()
        });
        assert_eq!(res.trial_summaries.len(), trials);
        assert_eq!(res.trial_summaries[0].trial, 0);
        assert_eq!(res.trial_summaries[trials - 1].trial, trials - 1);
    }
}
