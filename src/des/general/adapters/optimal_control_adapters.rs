//! Port of `src/des/general/adapters/optimal-control-adapters.ts`
//! (module `des::general::adapters::optimal_control_adapters`).
//!
//! Registers seven optimal-control JSON adapters: Pontryagin bang-bang, Kalman
//! radar tracking, sliding-mode, MRAC, iterative-learning-control,
//! feedback-linearization, and MPC double-integrator.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * The TS `run` bodies coerce decoded arrays via `numberPair(p.x0, ..)` /
//!     `optionalNumberPair(p.Q, ..)`. The Rust engine `Opts` already type the
//!     fields as `Option<[f64; 2]>` and default them internally (e.g.
//!     `run_pontryagin_bang_bang` falls back to `[3, 0]`), so the adapters pass
//!     the typed params straight through — no coercion (per the GotChA note).
//!   * `feedback-linearization` remaps the FLAT [`FlatFeedbackLinParams`] into
//!     the nested [`FeedbackLinearizationOpts`]`{ params: {m,l,g,c}, .. }` inside
//!     `run`.
//!   * `disturbanceType` / `referenceKind` literal unions -> the engine
//!     [`DisturbanceType`] / [`ILCReferenceKind`] enums (with local
//!     `*_str` renderers for the summaries).
//!   * Engine `run`s that return `Result<_, PreconditionError>` are unwrapped
//!     with `panic!` (the TS `throw`).
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; each adapter is
//! exposed via the `adapter_*()` constructors for explicit registration later.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and the ILC `buildILCFrame`/`metricBar`/`drawMiniSeries`
//! helpers) is not ported, so the ILC `animate` is a no-op here.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};

use crate::des::general::feedback_linearization::{
    run_feedback_linearization, FeedbackLinearizationOpts, FeedbackLinearizationResult,
    PartialPendulumParams,
};
use crate::des::general::iterative_learning_control::{
    run_iterative_learning_control, ILCReferenceKind, IterativeLearningControlParams,
    IterativeLearningControlResult,
};
use crate::des::general::kalman_filter::{run_radar_tracking, RadarTrackingOpts, RadarTrackingResult};
use crate::des::general::mpc_double_integrator::{
    run_mpc_double_integrator, MpcDoubleIntOpts, MpcDoubleIntResult,
};
use crate::des::general::mrac::{run_mrac, MRACOpts, MRACResult};
use crate::des::general::pontryagin_bang_bang::{
    run_pontryagin_bang_bang, PontryaginOpts, PontryaginResult,
};
use crate::des::general::sliding_mode_control::{
    run_sliding_mode, DisturbanceType, SlidingModeOpts, SlidingModeResult,
};

// =============================================================================
// Formatting helpers (JS parity)
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `Number.prototype.toExponential(digits)`.
fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

/// `numbers.map(String).join(', ')`.
fn join_nums(values: &[f64]) -> String {
    values.iter().map(|v| js_number(*v)).collect::<Vec<_>>().join(", ")
}

/// `numbers.map(v => v.toFixed(digits)).join(', ')`.
fn fixed_join(values: &[f64], digits: usize) -> String {
    values.iter().map(|v| format!("{:.*}", digits, v)).collect::<Vec<_>>().join(", ")
}

fn disturbance_type_str(d: DisturbanceType) -> &'static str {
    match d {
        DisturbanceType::Sin => "sin",
        DisturbanceType::Square => "square",
        DisturbanceType::Random => "random",
    }
}

fn ilc_reference_kind_str(k: ILCReferenceKind) -> &'static str {
    match k {
        ILCReferenceKind::Sine => "sine",
        ILCReferenceKind::Step => "step",
        ILCReferenceKind::Ramp => "ramp",
    }
}

/// `t,x,v,u` CSV shared by pontryagin / sliding-mode / mpc.
fn write_xvu_csv(trajectory: &[Vec<f64>], controls: &[Vec<f64>], csv_path: &str) {
    let mut lines = vec!["t,x,v,u".to_string()];
    for i in 0..controls.len() {
        lines.push(csv_row([
            i.to_string(),
            format!("{:.6}", trajectory[i][0]),
            format!("{:.6}", trajectory[i][1]),
            format!("{:.6}", controls[i][0]),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn arr_mm(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length, description: None }
}

fn obj_desc(description: &str, fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: Some(description.to_string()),
    }
}

/// A length-2 number array field (`{kind:'array', items:{kind:'number', ...},
/// minLength:2, maxLength:2}`).
fn pair_schema(item_min: Option<f64>) -> ParamSchema {
    arr_mm(num(item_min, None, None, None), Some(2), Some(2))
}

// =============================================================================
// 1. pontryagin-bang-bang
// =============================================================================

fn pontryagin_schema() -> ParamSchema {
    obj_desc(
        "Time-optimal control of a double integrator via Pontryagin's Maximum Principle (bang-bang).",
        vec![
            ("x0", pair_schema(None)),
            ("uMax", num(Some(0.0), None, None, Some(1.0))),
            ("dt", num(Some(1e-6), None, None, Some(0.02))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(500.0))),
            ("deadband", num(Some(0.0), None, None, Some(0.1))),
        ],
        vec![],
    )
}

pub struct PontryaginBangBangAdapter;
pub fn adapter_pontryagin_bang_bang() -> PontryaginBangBangAdapter {
    PontryaginBangBangAdapter
}

impl DESModelRegistration<PontryaginOpts, PontryaginResult> for PontryaginBangBangAdapter {
    fn id(&self) -> &str {
        "pontryagin-bang-bang"
    }
    fn description(&self) -> &str {
        "Pontryagin Maximum Principle: time-optimal bang-bang control on a double integrator."
    }
    fn schema(&self) -> ParamSchema {
        pontryagin_schema()
    }
    fn run(&self, p: PontryaginOpts, _runtime: &DESRuntimeConfig) -> PontryaginResult {
        run_pontryagin_bang_bang(p).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &PontryaginResult, p: &PontryaginOpts) -> String {
        let dt = p.dt.unwrap_or(0.02);
        let x0 = p.x0.unwrap_or([3.0, 0.0]);
        let last = &r.trajectory[r.trajectory.len() - 1];
        [
            "PONTRYAGIN BANG-BANG (time-optimal)".to_string(),
            "────────────────────────────────────".to_string(),
            format!("  Initial state:           [{}]", join_nums(&x0)),
            format!("  |u| bound:               {}", js_number(p.u_max.unwrap_or(1.0))),
            format!("  Bang-bang switches:      {}  (PMP predicts ≤ 1)", r.switch_count),
            format!("  Arrival tick:            {}    (entered deadband)", r.arrival_tick),
            format!("  Arrival time:            {:.3} s", r.arrival_tick as f64 * dt),
            format!("  Theoretical optimum t*:  {:.3} s", r.theoretical_arrival_time),
            format!("  Final state:             [{}]", fixed_join(last, 3)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &PontryaginResult, csv_path: &str) {
        write_xvu_csv(&r.trajectory, &r.controls, csv_path);
    }
}

// =============================================================================
// 2. kalman-filter
// =============================================================================

fn kalman_schema() -> ParamSchema {
    obj_desc(
        "Linear Kalman filter on a noisy 1-D radar tracking problem.",
        vec![
            ("x0", pair_schema(None)),
            ("dt", num(Some(1e-6), None, None, Some(0.1))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(200.0))),
            ("procNoiseStd", num(Some(0.0), None, None, Some(0.1))),
            ("measNoiseStd", num(Some(0.0), None, None, Some(1.0))),
            ("P0Scale", num(Some(0.0), None, None, Some(10.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
        ],
        vec![],
    )
}

pub struct KalmanFilterAdapter;
pub fn adapter_kalman_filter() -> KalmanFilterAdapter {
    KalmanFilterAdapter
}

impl DESModelRegistration<RadarTrackingOpts, RadarTrackingResult> for KalmanFilterAdapter {
    fn id(&self) -> &str {
        "kalman-filter"
    }
    fn description(&self) -> &str {
        "Linear Kalman filter — radar tracking of a constant-velocity target with position-only sensor."
    }
    fn schema(&self) -> ParamSchema {
        kalman_schema()
    }
    fn run(&self, p: RadarTrackingOpts, _runtime: &DESRuntimeConfig) -> RadarTrackingResult {
        run_radar_tracking(p).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &RadarTrackingResult, p: &RadarTrackingOpts) -> String {
        [
            "KALMAN FILTER — RADAR TRACKING".to_string(),
            "────────────────────────────────────".to_string(),
            format!("  Process noise σ_w:       {}", js_number(p.proc_noise_std.unwrap_or(0.1))),
            format!("  Sensor noise σ_v:        {}", js_number(p.meas_noise_std.unwrap_or(1.0))),
            format!("  Steps:                   {}", r.num_steps),
            format!("  RMSE (KF estimate):      {:.3} m", r.rmse_pos),
            format!("  RMSE (raw measurement):  {:.3} m", r.rmse_meas_pos),
            format!("  Final cov trace:         {:.3}", r.final_cov_trace),
            format!(
                "  KF beats raw sensor by:  {:.1} %",
                100.0 * (1.0 - r.rmse_pos / r.rmse_meas_pos)
            ),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &RadarTrackingResult, csv_path: &str) {
        let mut lines = vec!["t,truePos,trueVel,measPos,estPos,estVel".to_string()];
        for i in 0..r.estimates.len() {
            let tp = &r.true_trajectory[i + 1];
            lines.push(csv_row([
                i.to_string(),
                format!("{:.6}", tp[0]),
                format!("{:.6}", tp[1]),
                format!("{:.6}", r.measurements[i][0]),
                format!("{:.6}", r.estimates[i][0]),
                format!("{:.6}", r.estimates[i][1]),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 3. sliding-mode
// =============================================================================

fn sliding_mode_schema() -> ParamSchema {
    obj_desc(
        "Sliding-mode control of an uncertain double integrator with bounded matched disturbance.",
        vec![
            ("x0", pair_schema(None)),
            ("dt", num(Some(1e-6), None, None, Some(0.05))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(400.0))),
            ("lambda", num(Some(0.0), None, None, Some(2.0))),
            ("eta", num(Some(0.0), None, None, Some(3.0))),
            ("boundary", num(Some(0.0), None, None, Some(0.05))),
            ("uBound", num(Some(0.0), None, None, Some(5.0))),
            ("disturbanceAmp", num(Some(0.0), None, None, Some(1.0))),
            ("disturbanceType", str_enum(&["sin", "square", "random"], "sin")),
            ("seed", num(None, None, Some(true), Some(1.0))),
        ],
        vec![],
    )
}

pub struct SlidingModeAdapter;
pub fn adapter_sliding_mode() -> SlidingModeAdapter {
    SlidingModeAdapter
}

impl DESModelRegistration<SlidingModeOpts, SlidingModeResult> for SlidingModeAdapter {
    fn id(&self) -> &str {
        "sliding-mode"
    }
    fn description(&self) -> &str {
        "Sliding-mode (robust) control of an uncertain plant with bounded disturbance."
    }
    fn schema(&self) -> ParamSchema {
        sliding_mode_schema()
    }
    fn run(&self, p: SlidingModeOpts, _runtime: &DESRuntimeConfig) -> SlidingModeResult {
        run_sliding_mode(p)
    }
    fn summarize(&self, r: &SlidingModeResult, p: &SlidingModeOpts) -> String {
        [
            "SLIDING-MODE CONTROL (robust)".to_string(),
            "────────────────────────────────────".to_string(),
            format!(
                "  Disturbance:             type={} amp={}",
                disturbance_type_str(p.disturbance_type.unwrap_or(DisturbanceType::Sin)),
                js_number(p.disturbance_amp.unwrap_or(1.0))
            ),
            format!("  Reaching tick:           {}    (s = 0 hit)", r.reaching_tick),
            format!(
                "  Stayed near origin?      {}",
                if r.stayed_near_origin { "YES" } else { "no" }
            ),
            format!("  Final |x|+|v|:           {:.3}", r.final_distance_from_origin),
            format!("  λ:                       {}", js_number(p.lambda.unwrap_or(2.0))),
            format!("  η:                       {}", js_number(p.eta.unwrap_or(3.0))),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &SlidingModeResult, csv_path: &str) {
        write_xvu_csv(&r.trajectory, &r.controls, csv_path);
    }
}

// =============================================================================
// 4. mrac
// =============================================================================

fn mrac_schema() -> ParamSchema {
    obj_desc(
        "Model-Reference Adaptive Control on a first-order plant with unknown a, b > 0.",
        vec![
            ("a", num(None, None, None, Some(1.0))),
            ("b", num(None, None, None, Some(2.0))),
            ("am", num(None, Some(-1e-9), None, Some(-2.0))),
            ("bm", num(None, None, None, Some(2.0))),
            ("x0", num(None, None, None, Some(0.0))),
            ("xm0", num(None, None, None, Some(0.0))),
            ("gamma", num(Some(0.0), None, None, Some(5.0))),
            ("dt", num(Some(1e-6), None, None, Some(0.01))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(4000.0))),
            ("uBound", num(Some(0.0), None, None, None)),
        ],
        vec![],
    )
}

pub struct MracAdapter;
pub fn adapter_mrac() -> MracAdapter {
    MracAdapter
}

impl DESModelRegistration<MRACOpts, MRACResult> for MracAdapter {
    fn id(&self) -> &str {
        "mrac"
    }
    fn description(&self) -> &str {
        "Model-Reference Adaptive Control with unknown plant gain (Lyapunov-based MIT rule)."
    }
    fn schema(&self) -> ParamSchema {
        mrac_schema()
    }
    fn run(&self, p: MRACOpts, _runtime: &DESRuntimeConfig) -> MRACResult {
        run_mrac(p)
    }
    fn summarize(&self, r: &MRACResult, p: &MRACOpts) -> String {
        [
            "MRAC (Model-Reference Adaptive Control)".to_string(),
            "────────────────────────────────────".to_string(),
            format!(
                "  True plant:              a={} b={}",
                js_number(p.a.unwrap_or(1.0)),
                js_number(p.b.unwrap_or(2.0))
            ),
            format!(
                "  Reference model:         a_m={} b_m={}",
                js_number(p.am.unwrap_or(-2.0)),
                js_number(p.bm.unwrap_or(2.0))
            ),
            format!("  Adaptation gain γ:       {}", js_number(p.gamma.unwrap_or(5.0))),
            format!("  Final θ_x, θ_r:          [{}]", fixed_join(&r.final_theta, 3)),
            format!("  Ideal θ*_x, θ*_r:        [{}]", fixed_join(&r.ideal_theta, 3)),
            format!("  Steady-state RMS error:  {:.4}", r.rms_error_steady_state),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &MRACResult, csv_path: &str) {
        let mut lines = vec!["t,x,xm,r,theta_x,theta_r,error".to_string()];
        for i in 0..r.tracking_error.len() {
            lines.push(csv_row([
                i.to_string(),
                format!("{:.6}", r.trajectory[i + 1][0]),
                format!("{:.6}", r.reference_trajectory[i + 1]),
                format!("{:.6}", r.r_history[i]),
                format!("{:.6}", r.theta_x_history[i]),
                format!("{:.6}", r.theta_r_history[i]),
                format!("{:.6}", r.tracking_error[i]),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 5. iterative-learning-control
// =============================================================================

fn ilc_schema() -> ParamSchema {
    obj_desc(
        "Iterative Learning Control on a repeated first-order tracking task.",
        vec![
            ("trials", num(Some(1.0), None, Some(true), Some(30.0))),
            ("horizon", num(Some(2.0), None, Some(true), Some(80.0))),
            ("dt", num(Some(1e-6), None, None, Some(0.1))),
            ("plantRate", num(Some(1e-12), None, None, Some(1.2))),
            ("plantGain", num(Some(1e-12), None, None, Some(1.0))),
            ("learningGain", num(Some(0.0), Some(2.0), None, Some(0.8))),
            ("feedbackGain", num(Some(0.0), None, None, Some(0.8))),
            ("controlMax", num(Some(1e-12), None, None, Some(5.0))),
            ("referenceKind", str_enum(&["sine", "step", "ramp"], "sine")),
            ("referenceAmplitude", num(Some(0.0), None, None, Some(1.0))),
            ("initialOutput", num(None, None, None, Some(0.0))),
        ],
        vec![],
    )
}

pub struct IterativeLearningControlAdapter;
pub fn adapter_iterative_learning_control() -> IterativeLearningControlAdapter {
    IterativeLearningControlAdapter
}

impl DESModelRegistration<IterativeLearningControlParams, IterativeLearningControlResult>
    for IterativeLearningControlAdapter
{
    fn id(&self) -> &str {
        "iterative-learning-control"
    }
    fn description(&self) -> &str {
        "Iterative Learning Control: repeated-trial feedforward adaptation over source/station/sink movables."
    }
    fn schema(&self) -> ParamSchema {
        ilc_schema()
    }
    fn run(
        &self,
        p: IterativeLearningControlParams,
        _runtime: &DESRuntimeConfig,
    ) -> IterativeLearningControlResult {
        run_iterative_learning_control(&p)
    }
    fn summarize(
        &self,
        r: &IterativeLearningControlResult,
        p: &IterativeLearningControlParams,
    ) -> String {
        [
            "ITERATIVE LEARNING CONTROL (DES)".to_string(),
            "--------------------------------".to_string(),
            format!("  Trials:         {}", r.trial_summaries.len()),
            format!(
                "  Reference:      {}  amplitude={}",
                ilc_reference_kind_str(p.reference_kind.unwrap_or(ILCReferenceKind::Sine)),
                js_number(p.reference_amplitude.unwrap_or(1.0))
            ),
            format!("  Initial RMS:    {:.6}", r.initial_rms_error),
            format!("  Final RMS:      {:.6}", r.final_rms_error),
            format!(
                "  Improvement:    {:.1}% RMS reduction",
                100.0 * (1.0 - r.improvement_ratio)
            ),
            format!("  Stations:       {}", r.topology.stations.join(" -> ")),
            format!("  Movables:       {}", r.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &IterativeLearningControlResult, csv_path: &str) {
        let mut lines =
            vec!["trial,rms_error,max_abs_error,max_abs_control,final_output,final_reference".to_string()];
        for row in &r.trial_summaries {
            lines.push(csv_row([
                row.trial.to_string(),
                js_number(row.rms_error),
                js_number(row.max_abs_error),
                js_number(row.max_abs_control),
                js_number(row.final_output),
                js_number(row.final_reference),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _r: &IterativeLearningControlResult,
        _p: &IterativeLearningControlParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
}

// =============================================================================
// 6. feedback-linearization
// =============================================================================

fn feedback_lin_schema() -> ParamSchema {
    obj_desc(
        "Feedback-linearization (computed-torque) tracking control of a pendulum.",
        vec![
            ("m", num(Some(0.0), None, None, Some(1.0))),
            ("l", num(Some(0.0), None, None, Some(1.0))),
            ("g", num(Some(0.0), None, None, Some(9.81))),
            ("c", num(Some(0.0), None, None, Some(0.1))),
            ("theta0", num(None, None, None, Some(3.141592653589793))),
            ("thetaDot0", num(None, None, None, Some(0.0))),
            ("kp", num(Some(0.0), None, None, Some(25.0))),
            ("kv", num(Some(0.0), None, None, Some(10.0))),
            ("uBound", num(Some(0.0), None, None, Some(100.0))),
            ("dt", num(Some(1e-6), None, None, Some(0.01))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(1000.0))),
        ],
        vec![],
    )
}

/// `interface FlatFeedbackLinParams` — the flat adapter params remapped into the
/// nested [`FeedbackLinearizationOpts`] inside `run`.
#[derive(Clone, Debug, Default)]
pub struct FlatFeedbackLinParams {
    pub m: Option<f64>,
    pub l: Option<f64>,
    pub g: Option<f64>,
    pub c: Option<f64>,
    pub theta0: Option<f64>,
    pub theta_dot0: Option<f64>,
    pub kp: Option<f64>,
    pub kv: Option<f64>,
    pub u_bound: Option<f64>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
}

pub struct FeedbackLinearizationAdapter;
pub fn adapter_feedback_linearization() -> FeedbackLinearizationAdapter {
    FeedbackLinearizationAdapter
}

impl DESModelRegistration<FlatFeedbackLinParams, FeedbackLinearizationResult>
    for FeedbackLinearizationAdapter
{
    fn id(&self) -> &str {
        "feedback-linearization"
    }
    fn description(&self) -> &str {
        "Feedback linearization (nonlinear control): pendulum tracking via computed torque."
    }
    fn schema(&self) -> ParamSchema {
        feedback_lin_schema()
    }
    fn run(&self, p: FlatFeedbackLinParams, _runtime: &DESRuntimeConfig) -> FeedbackLinearizationResult {
        let opts = FeedbackLinearizationOpts {
            params: Some(PartialPendulumParams {
                m: Some(p.m.unwrap_or(1.0)),
                l: Some(p.l.unwrap_or(1.0)),
                g: Some(p.g.unwrap_or(9.81)),
                c: Some(p.c.unwrap_or(0.1)),
            }),
            theta0: p.theta0,
            theta_dot0: p.theta_dot0,
            kp: p.kp,
            kv: p.kv,
            u_bound: p.u_bound,
            dt: p.dt,
            num_steps: p.num_steps,
        };
        run_feedback_linearization(opts)
    }
    fn summarize(&self, r: &FeedbackLinearizationResult, p: &FlatFeedbackLinParams) -> String {
        let last = &r.trajectory[r.trajectory.len() - 1];
        [
            "FEEDBACK LINEARIZATION (pendulum)".to_string(),
            "────────────────────────────────────".to_string(),
            format!(
                "  Mass / length / g / damping: {} / {} / {} / {}",
                js_number(p.m.unwrap_or(1.0)),
                js_number(p.l.unwrap_or(1.0)),
                js_number(p.g.unwrap_or(9.81)),
                js_number(p.c.unwrap_or(0.1))
            ),
            format!(
                "  PD gains kp / kv:        {} / {}",
                js_number(p.kp.unwrap_or(25.0)),
                js_number(p.kv.unwrap_or(10.0))
            ),
            format!("  Steps:                   {}", r.num_steps),
            format!(
                "  Steady-state RMS error:  {} rad",
                to_exponential(r.rms_error_steady_state, 2)
            ),
            format!("  Final angle:             {:.4} rad", last[0]),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &FeedbackLinearizationResult, csv_path: &str) {
        let mut lines = vec!["t,theta,thetaDot,thetaRef,torque".to_string()];
        for i in 0..r.controls.len() {
            lines.push(csv_row([
                i.to_string(),
                format!("{:.6}", r.trajectory[i + 1][0]),
                format!("{:.6}", r.trajectory[i + 1][1]),
                format!("{:.6}", r.theta_d_history[i]),
                format!("{:.6}", r.controls[i][0]),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 7. mpc-double-integrator
// =============================================================================

fn mpc_schema() -> ParamSchema {
    obj_desc(
        "Constrained MPC on a double integrator: receding-horizon QP via projected gradient.",
        vec![
            ("x0", pair_schema(None)),
            ("uMax", num(Some(0.0), None, None, Some(1.0))),
            ("N", num(Some(1.0), None, Some(true), Some(15.0))),
            ("Q", pair_schema(Some(0.0))),
            ("Qf", pair_schema(Some(0.0))),
            ("R", num(Some(1e-9), None, None, Some(0.1))),
            ("dt", num(Some(1e-6), None, None, Some(0.1))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(100.0))),
        ],
        vec![],
    )
}

pub struct MpcDoubleIntegratorAdapter;
pub fn adapter_mpc_double_integrator() -> MpcDoubleIntegratorAdapter {
    MpcDoubleIntegratorAdapter
}

impl DESModelRegistration<MpcDoubleIntOpts, MpcDoubleIntResult> for MpcDoubleIntegratorAdapter {
    fn id(&self) -> &str {
        "mpc-double-integrator"
    }
    fn description(&self) -> &str {
        "MPC: constrained receding-horizon QP on a double integrator."
    }
    fn schema(&self) -> ParamSchema {
        mpc_schema()
    }
    fn run(&self, p: MpcDoubleIntOpts, _runtime: &DESRuntimeConfig) -> MpcDoubleIntResult {
        run_mpc_double_integrator(p).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &MpcDoubleIntResult, p: &MpcDoubleIntOpts) -> String {
        let dt = p.dt.unwrap_or(0.1);
        let x0 = p.x0.unwrap_or([3.0, 0.0]);
        let last = &r.trajectory[r.trajectory.len() - 1];
        [
            "MPC — DOUBLE INTEGRATOR (constrained QP)".to_string(),
            "──────────────────────────────────────────".to_string(),
            format!("  Initial state:           [{}]", join_nums(&x0)),
            format!("  |u| bound:               {}", js_number(p.u_max.unwrap_or(1.0))),
            format!("  Horizon N:               {}", p.n.unwrap_or(15)),
            format!("  Sample period dt:        {}", js_number(dt)),
            format!(
                "  Arrival tick:            {}    (~ {:.2} s)",
                r.arrival_tick,
                r.arrival_tick as f64 * dt
            ),
            format!("  Max realised |u|:        {:.4}", r.max_abs_u),
            format!("  Final state:             [{}]", fixed_join(last, 3)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &MpcDoubleIntResult, csv_path: &str) {
        write_xvu_csv(&r.trajectory, &r.controls, csv_path);
    }
}
