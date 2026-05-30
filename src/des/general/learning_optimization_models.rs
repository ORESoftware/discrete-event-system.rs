//! Port of `src/des/general/learning-optimization-models.ts`
//! (module `des::general::learning_optimization_models`).
//!
//! Station-graph implementations for supervised optimization models:
//!   - linear-regression-ls
//!   - ridge-regression-ls
//!   - logistic-regression-sgd
//!   - backprop-mlp-classifier
//!
//! Every runner builds stationary source/batch/evaluator/update/sink stations
//! and moves typed tokens between them. Numerical routines live inside stations,
//! not as hidden one-shot logic in adapters.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `SupervisedSample.y: number | number[]` → the [`Label`] enum
//!     (`Scalar(f64)` / `Vector(Vec<f64>)`).
//!   * `optimizer: 'sgd' | 'adam'` → the ported [`Optimizer`] enum;
//!     `RidgeRegressionParams extends LinearRegressionParams` is composed by
//!     forwarding into [`run_linear_regression_ls`].
//!   * `mulberry32(seed)` MLP weight init → the injected
//!     [`mulberry32`](crate::des::general::prng::mulberry32) `SeededRandom`.
//!   * `GradientOptimizerStation`'s `protected evaluateBatch` hook → the ported
//!     [`GradientOptimizerHook`] trait; the two concrete learners are hook
//!     structs (`LogisticRegressionHook`, `BackpropMLPHook`) carried by a
//!     [`GradientOptimizerStation<H>`].
//!   * `solveLinearSystem` (Gaussian elimination) is kept file-local; its
//!     throw-on-singular becomes a `panic!` (an invariant for the LS normal
//!     equations — caller is told to add ridge).
//!   * predict closures `(params, input) => number` → `impl Fn(&[f64], &[f64])
//!     -> f64`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, dot, non_empty_array, sigmoid, softmax, station_graph, zeros, GradientEvaluation,
    GradientOptimizerHook, GradientOptimizerOptions, GradientOptimizerStation, GradientTraceSinkStation,
    MiniBatchStation, Optimizer, StationGraphSummary, StationOrId, VectorBatchToken, VectorSampleSourceStation,
    VectorSampleToken,
};
use crate::des::general::des_base::neural_network::{Meta, MetaValue, NumericVector};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

// ── Shared types ──────────────────────────────────────────────────────────────

/// `SupervisedSample.y: number | number[]` → a tagged scalar/vector label.
#[derive(Clone, Debug, PartialEq)]
pub enum Label {
    Scalar(f64),
    Vector(Vec<f64>),
}

/// A labelled supervised training sample.
#[derive(Clone, Debug, PartialEq)]
pub struct SupervisedSample {
    pub x: Vec<f64>,
    pub y: Label,
}

#[derive(Clone, Debug, Default)]
pub struct LinearRegressionParams {
    pub samples: Option<Vec<SupervisedSample>>,
    pub fit_intercept: Option<bool>,
    pub ridge: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct LinearRegressionResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub mse: f64,
    pub predictions: Vec<f64>,
    pub residuals: Vec<f64>,
    pub sample_count: usize,
    pub topology: StationGraphSummary,
}

/// `RidgeRegressionParams extends LinearRegressionParams`.
#[derive(Clone, Debug, Default)]
pub struct RidgeRegressionParams {
    pub samples: Option<Vec<SupervisedSample>>,
    pub fit_intercept: Option<bool>,
    pub ridge: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct LogisticRegressionSGDParams {
    pub samples: Option<Vec<SupervisedSample>>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub optimizer: Option<Optimizer>,
    pub l2: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct BackpropMLPParams {
    pub samples: Option<Vec<SupervisedSample>>,
    pub hidden_units: Option<usize>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub optimizer: Option<Optimizer>,
    pub seed: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct GradientTrainingResult {
    pub parameters: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub loss_history: Vec<f64>,
    pub gradient_norm_history: Vec<f64>,
    pub final_loss: f64,
    pub accuracy: f64,
    pub predictions: Vec<f64>,
    pub topology: StationGraphSummary,
}

// ── RegressionFitToken / NormalEquationStation / sink ─────────────────────────

/// Fit emitted by the normal-equation accumulator.
#[derive(Clone, Debug)]
pub struct RegressionFitToken {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub sample_count: usize,
}

/// Accumulates `XᵀX` / `Xᵀy`, then solves the (optionally ridge-regularised)
/// normal equations.
pub struct NormalEquationStation {
    core: StationCore,
    input_dim: usize,
    fit_intercept: bool,
    ridge: f64,
    xtx: Vec<Vec<f64>>,
    xty: Vec<f64>,
    sample_count: usize,
}

impl NormalEquationStation {
    pub const CH_SAMPLE: &'static str = VectorSampleSourceStation::CH_SAMPLE;
    pub const CH_FIT: &'static str = "fit";

    pub fn new(id: impl Into<String>, input_dim: usize, fit_intercept: bool, ridge: f64) -> Self {
        let d = input_dim + if fit_intercept { 1 } else { 0 };
        NormalEquationStation {
            core: StationCore::new(id),
            input_dim,
            fit_intercept,
            ridge,
            xtx: (0..d).map(|_| zeros(d)).collect(),
            xty: zeros(d),
            sample_count: 0,
        }
    }

    fn design_row(&self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.input_dim {
            panic!("expected input dimension {}, got {}", self.input_dim, x.len());
        }
        if self.fit_intercept {
            let mut row = x.to_vec();
            row.push(1.0);
            row
        } else {
            x.to_vec()
        }
    }
}

impl DESStation for NormalEquationStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_SAMPLE) > 0
    }
    fn run_time_step(&mut self) {
        let samples = self.core.drain::<VectorSampleToken>(Self::CH_SAMPLE);
        for sample in samples {
            let row = self.design_row(&sample.input);
            let y = sample.target[0];
            for i in 0..row.len() {
                self.xty[i] += sample.weight * row[i] * y;
                for j in 0..row.len() {
                    self.xtx[i][j] += sample.weight * row[i] * row[j];
                }
            }
            self.sample_count += 1;
        }
        if self.sample_count > 0 {
            let mut mat = self.xtx.clone();
            for (i, row) in mat.iter_mut().enumerate() {
                row[i] += self.ridge;
            }
            let beta = solve_linear_system(&mat, &self.xty);
            let intercept = if self.fit_intercept { beta[beta.len() - 1] } else { 0.0 };
            let coefficients = if self.fit_intercept { beta[..beta.len() - 1].to_vec() } else { beta.clone() };
            let token = RegressionFitToken { coefficients, intercept, sample_count: self.sample_count };
            self.core.emit(Rc::new(token), Self::CH_FIT);
        }
    }
}

/// Keeps the latest [`RegressionFitToken`].
pub struct RegressionFitSinkStation {
    core: StationCore,
    pub fit: Option<RegressionFitToken>,
}

impl RegressionFitSinkStation {
    pub const CH_FIT: &'static str = NormalEquationStation::CH_FIT;

    pub fn new(id: impl Into<String>) -> Self {
        RegressionFitSinkStation { core: StationCore::new(id), fit: None }
    }
}

impl DESStation for RegressionFitSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_FIT) > 0
    }
    fn run_time_step(&mut self) {
        let fits = self.core.drain::<RegressionFitToken>(Self::CH_FIT);
        if let Some(last) = fits.last() {
            self.fit = Some((**last).clone());
        }
    }
}

pub fn run_linear_regression_ls(params: LinearRegressionParams) -> LinearRegressionResult {
    let raw = non_empty_array(params.samples.as_deref(), &default_regression_samples());
    let samples = to_vector_samples(&raw);
    let input_dim = samples.first().map(|s| s.input.len()).unwrap_or(0);

    let source = Rc::new(RefCell::new(VectorSampleSourceStation::new("sample-source", samples.clone(), 1)));
    let normal = Rc::new(RefCell::new(NormalEquationStation::new(
        "normal-equation-accumulator",
        input_dim,
        params.fit_intercept.unwrap_or(true),
        params.ridge.unwrap_or(0.0),
    )));
    let sink = Rc::new(RefCell::new(RegressionFitSinkStation::new("regression-fit-sink")));

    source.borrow_mut().core_mut().pipe(
        normal.clone() as StationRef,
        VectorSampleSourceStation::CH_SAMPLE,
        NormalEquationStation::CH_SAMPLE,
    );
    normal.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        NormalEquationStation::CH_FIT,
        RegressionFitSinkStation::CH_FIT,
    );

    run_iterative_des(
        vec![source.clone() as StationRef, normal.clone() as StationRef, sink.clone() as StationRef],
        IterativeRunOptions { shuffle: false, ..Default::default() },
    );

    let fit = sink.borrow().fit.clone().unwrap_or_else(|| panic!("linear-regression-ls did not produce a fit"));
    let predictions: Vec<f64> = samples.iter().map(|s| dot(&s.input, &fit.coefficients) + fit.intercept).collect();
    let residuals: Vec<f64> = samples.iter().enumerate().map(|(i, s)| predictions[i] - s.target[0]).collect();
    let mse = residuals.iter().map(|r| r * r).sum::<f64>() / (residuals.len().max(1) as f64);

    let stations = [
        StationOrId::Id("sample-source".to_string()),
        StationOrId::Id("normal-equation-accumulator".to_string()),
        StationOrId::Id("regression-fit-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(&stations[0], VectorSampleSourceStation::CH_SAMPLE, &stations[1], Some(NormalEquationStation::CH_SAMPLE)),
        channel_edge(&stations[1], NormalEquationStation::CH_FIT, &stations[2], Some(RegressionFitSinkStation::CH_FIT)),
    ];
    let topology = station_graph(
        &stations,
        &["VectorSampleToken".to_string(), "RegressionFitToken".to_string()],
        &edges,
    );

    LinearRegressionResult {
        coefficients: fit.coefficients.clone(),
        intercept: fit.intercept,
        mse,
        predictions,
        residuals,
        sample_count: fit.sample_count,
        topology,
    }
}

pub fn run_ridge_regression_ls(params: RidgeRegressionParams) -> LinearRegressionResult {
    run_linear_regression_ls(LinearRegressionParams {
        samples: params.samples,
        fit_intercept: params.fit_intercept,
        ridge: Some(params.ridge.unwrap_or(0.1)),
    })
}

// ── LogisticRegressionHook ────────────────────────────────────────────────────

/// Logistic-regression batch evaluation (cross-entropy + L2).
pub struct LogisticRegressionHook {
    l2: f64,
}

impl GradientOptimizerHook for LogisticRegressionHook {
    fn evaluate_batch(&mut self, batch: &VectorBatchToken, parameters: &[f64]) -> GradientEvaluation {
        let n = parameters.len();
        let mut gradient = zeros(n);
        let mut loss = 0.0;
        for sample in &batch.samples {
            let z = dot(&parameters[..n - 1], &sample.input) + parameters[n - 1];
            let p = sigmoid(z);
            let y = sample.target[0];
            loss += -sample.weight * (y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln());
            let err = sample.weight * (p - y);
            for i in 0..sample.input.len() {
                gradient[i] += err * sample.input[i];
            }
            gradient[n - 1] += err;
        }
        for i in 0..n - 1 {
            loss += 0.5 * self.l2 * parameters[i] * parameters[i];
            gradient[i] += self.l2 * parameters[i];
        }
        let denom = batch.samples.len().max(1) as f64;
        let gradient: Vec<f64> = gradient.iter().map(|g| g / denom).collect();
        GradientEvaluation { loss: loss / denom, gradient, meta: Some(batch_meta(&batch.id)) }
    }
}

pub fn run_logistic_regression_sgd(params: LogisticRegressionSGDParams) -> GradientTrainingResult {
    let raw = non_empty_array(params.samples.as_deref(), &default_logistic_samples());
    let samples = to_vector_samples(&raw);
    let input_dim = samples.first().map(|s| s.input.len()).unwrap_or(0);

    let source = Rc::new(RefCell::new(VectorSampleSourceStation::new(
        "sample-source",
        samples.clone(),
        params.epochs.unwrap_or(120),
    )));
    let batcher = Rc::new(RefCell::new(MiniBatchStation::new("mini-batch", params.batch_size.unwrap_or(4), true)));
    let learner = Rc::new(RefCell::new(GradientOptimizerStation::new(
        "logistic-gradient-update",
        LogisticRegressionHook { l2: params.l2.unwrap_or(0.0) },
        GradientOptimizerOptions {
            initial_parameters: zeros(input_dim + 1),
            learning_rate: params.learning_rate.unwrap_or(0.2),
            optimizer: Some(params.optimizer.unwrap_or(Optimizer::Sgd)),
            beta1: None,
            beta2: None,
            epsilon: None,
        },
    )));
    let trace = Rc::new(RefCell::new(GradientTraceSinkStation::new("gradient-trace-sink")));

    wire_gradient_pipeline::<LogisticRegressionHook>(&source, &batcher, &learner, &trace);
    run_iterative_des(
        vec![
            source.clone() as StationRef,
            batcher.clone() as StationRef,
            learner.clone() as StationRef,
            trace.clone() as StationRef,
        ],
        IterativeRunOptions { shuffle: false, ..Default::default() },
    );

    let topology = gradient_topology("logistic-gradient-update");
    let learner_ref = learner.borrow();
    let trace_ref = trace.borrow();
    gradient_training_result(&samples, &*learner_ref, &*trace_ref, topology, |p, i| logistic_predict(p, i))
}

// ── BackpropMLPHook ───────────────────────────────────────────────────────────

/// Single-hidden-layer sigmoid MLP with cross-entropy loss; weights are laid out
/// as `[W1 (hidden×input) | b1 (hidden) | W2 (hidden) | b2 (1)]`.
pub struct BackpropMLPHook {
    input_dim: usize,
    hidden_units: usize,
}

impl BackpropMLPHook {
    fn forward(&self, parameters: &[f64], input: &[f64]) -> (Vec<f64>, f64) {
        let mut hidden = zeros(self.hidden_units);
        let b1_offset = self.hidden_units * self.input_dim;
        let w2_offset = b1_offset + self.hidden_units;
        for h in 0..self.hidden_units {
            let mut z = parameters[b1_offset + h];
            for i in 0..self.input_dim {
                z += parameters[h * self.input_dim + i] * input[i];
            }
            hidden[h] = sigmoid(z);
        }
        let output = sigmoid(dot(&parameters[w2_offset..w2_offset + self.hidden_units], &hidden) + parameters[w2_offset + self.hidden_units]);
        (hidden, output)
    }
}

impl GradientOptimizerHook for BackpropMLPHook {
    fn evaluate_batch(&mut self, batch: &VectorBatchToken, parameters: &[f64]) -> GradientEvaluation {
        let mut gradient = zeros(parameters.len());
        let mut loss = 0.0;
        let w2_offset = self.hidden_units * self.input_dim + self.hidden_units;
        for sample in &batch.samples {
            let (hidden, output) = self.forward(parameters, &sample.input);
            let y = sample.target[0];
            loss += -sample.weight * (y * output.max(1e-12).ln() + (1.0 - y) * (1.0 - output).max(1e-12).ln());
            let d_out = sample.weight * (output - y);
            for h in 0..self.hidden_units {
                gradient[w2_offset + h] += d_out * hidden[h];
            }
            gradient[w2_offset + self.hidden_units] += d_out;
            for h in 0..self.hidden_units {
                let w2 = parameters[w2_offset + h];
                let d_hidden = d_out * w2 * hidden[h] * (1.0 - hidden[h]);
                for i in 0..self.input_dim {
                    gradient[h * self.input_dim + i] += d_hidden * sample.input[i];
                }
                gradient[self.hidden_units * self.input_dim + h] += d_hidden;
            }
        }
        let denom = batch.samples.len().max(1) as f64;
        let gradient: Vec<f64> = gradient.iter().map(|g| g / denom).collect();
        GradientEvaluation { loss: loss / denom, gradient, meta: Some(batch_meta(&batch.id)) }
    }
}

pub fn run_backprop_mlp_classifier(params: BackpropMLPParams) -> GradientTrainingResult {
    let raw = non_empty_array(params.samples.as_deref(), &default_xor_samples());
    let samples = to_vector_samples(&raw);
    let input_dim = samples.first().map(|s| s.input.len()).unwrap_or(0);
    let hidden_units = params.hidden_units.unwrap_or(4);
    let epochs = params.epochs.unwrap_or(800);

    let mut rng = mulberry32(params.seed.unwrap_or(7));
    let parameter_count = hidden_units * input_dim + hidden_units + hidden_units + 1;
    let initial_parameters: Vec<f64> = (0..parameter_count).map(|_| (rng.next_float() - 0.5) * 0.6).collect();

    let source = Rc::new(RefCell::new(VectorSampleSourceStation::new("sample-source", samples.clone(), epochs)));
    let batcher = Rc::new(RefCell::new(MiniBatchStation::new("mini-batch", params.batch_size.unwrap_or(samples.len()), true)));
    let learner = Rc::new(RefCell::new(GradientOptimizerStation::new(
        "backprop-gradient-update",
        BackpropMLPHook { input_dim, hidden_units },
        GradientOptimizerOptions {
            initial_parameters,
            learning_rate: params.learning_rate.unwrap_or(0.08),
            optimizer: Some(params.optimizer.unwrap_or(Optimizer::Adam)),
            beta1: None,
            beta2: None,
            epsilon: None,
        },
    )));
    let trace = Rc::new(RefCell::new(GradientTraceSinkStation::new("gradient-trace-sink")));

    wire_gradient_pipeline::<BackpropMLPHook>(&source, &batcher, &learner, &trace);
    run_iterative_des(
        vec![
            source.clone() as StationRef,
            batcher.clone() as StationRef,
            learner.clone() as StationRef,
            trace.clone() as StationRef,
        ],
        IterativeRunOptions { shuffle: false, max_ticks: Some(epochs * 5 + 20), ..Default::default() },
    );

    let topology = gradient_topology("backprop-gradient-update");
    let learner_ref = learner.borrow();
    let trace_ref = trace.borrow();
    let hook_ref = learner_ref.hook();
    // `p` is the learner's current parameters (passed in by `gradient_training_result`),
    // matching the TS `learner.predict(input)` which uses `getParameters()`.
    gradient_training_result(&samples, &*learner_ref, &*trace_ref, topology, |p, input| hook_ref.forward(p, input).1)
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn wire_gradient_pipeline<H: GradientOptimizerHook + 'static>(
    source: &Rc<RefCell<VectorSampleSourceStation>>,
    batcher: &Rc<RefCell<MiniBatchStation>>,
    learner: &Rc<RefCell<GradientOptimizerStation<H>>>,
    trace: &Rc<RefCell<GradientTraceSinkStation>>,
) {
    source.borrow_mut().core_mut().pipe(
        batcher.clone() as StationRef,
        VectorSampleSourceStation::CH_SAMPLE,
        MiniBatchStation::CH_SAMPLE,
    );
    batcher.borrow_mut().core_mut().pipe(
        learner.clone() as StationRef,
        MiniBatchStation::CH_BATCH,
        GradientOptimizerStation::<H>::CH_BATCH,
    );
    learner.borrow_mut().core_mut().pipe(
        trace.clone() as StationRef,
        GradientOptimizerStation::<H>::CH_STEP,
        GradientTraceSinkStation::CH_STEP,
    );
}

fn gradient_topology(learner_id: &str) -> StationGraphSummary {
    let stations = [
        StationOrId::Id("sample-source".to_string()),
        StationOrId::Id("mini-batch".to_string()),
        StationOrId::Id(learner_id.to_string()),
        StationOrId::Id("gradient-trace-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(&stations[0], VectorSampleSourceStation::CH_SAMPLE, &stations[1], Some(MiniBatchStation::CH_SAMPLE)),
        channel_edge(&stations[1], MiniBatchStation::CH_BATCH, &stations[2], Some(MiniBatchStation::CH_BATCH)),
        channel_edge(&stations[2], GradientTraceSinkStation::CH_STEP, &stations[3], Some(GradientTraceSinkStation::CH_STEP)),
    ];
    station_graph(
        &stations,
        &["VectorSampleToken".to_string(), "VectorBatchToken".to_string(), "GradientStepToken".to_string()],
        &edges,
    )
}

fn gradient_training_result<H: GradientOptimizerHook + 'static>(
    samples: &[VectorSampleToken],
    learner: &GradientOptimizerStation<H>,
    _trace: &GradientTraceSinkStation,
    topology: StationGraphSummary,
    predict: impl Fn(&[f64], &[f64]) -> f64,
) -> GradientTrainingResult {
    let parameters = learner.get_parameters();
    let weights = parameters[..parameters.len() - 1].to_vec();
    let bias = parameters[parameters.len() - 1];
    let predictions: Vec<f64> = samples.iter().map(|s| predict(&parameters, &s.input)).collect();
    let correct = predictions
        .iter()
        .enumerate()
        .filter(|(i, p)| (if **p >= 0.5 { 1.0 } else { 0.0 }) == samples[*i].target[0])
        .count();
    let accuracy = correct as f64 / (samples.len().max(1) as f64);
    let loss_history = learner.get_loss_history();
    let final_loss = loss_history.last().copied().unwrap_or(f64::NAN);
    GradientTrainingResult {
        parameters,
        weights,
        bias,
        loss_history,
        gradient_norm_history: learner.get_gradient_norm_history(),
        final_loss,
        accuracy,
        predictions,
        topology,
    }
}

fn logistic_predict(parameters: &[f64], input: &[f64]) -> f64 {
    let n = parameters.len();
    sigmoid(dot(&parameters[..n - 1], input) + parameters[n - 1])
}

fn batch_meta(batch_id: &str) -> Meta {
    let mut meta = Meta::new();
    meta.insert("batch".to_string(), MetaValue::Text(batch_id.to_string()));
    meta
}

fn to_vector_samples(samples: &[SupervisedSample]) -> Vec<VectorSampleToken> {
    samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let target: NumericVector = match &s.y {
                Label::Scalar(v) => vec![*v],
                Label::Vector(v) => v.clone(),
            };
            VectorSampleToken::new(format!("sample-{i}"), s.x.clone(), target)
        })
        .collect()
}

/// Gaussian elimination with partial pivoting (TS `solveLinearSystem`).
/// `panic!`s on a singular matrix (TS threw — an invariant for the LS normal
/// equations; the message tells the caller to add ridge).
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < 1e-12 {
            panic!("normal equations are singular; add ridge regularization");
        }
        m.swap(col, pivot);
        let div = m[col][col];
        for c in col..=n {
            m[col][c] /= div;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            for c in col..=n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }
    m.iter().map(|row| row[n]).collect()
}

fn default_regression_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample { x: vec![0.0], y: Label::Scalar(1.0) },
        SupervisedSample { x: vec![1.0], y: Label::Scalar(3.0) },
        SupervisedSample { x: vec![2.0], y: Label::Scalar(5.0) },
        SupervisedSample { x: vec![3.0], y: Label::Scalar(7.0) },
        SupervisedSample { x: vec![4.0], y: Label::Scalar(9.0) },
    ]
}

fn default_logistic_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample { x: vec![-2.0, -1.0], y: Label::Scalar(0.0) },
        SupervisedSample { x: vec![-1.0, -1.0], y: Label::Scalar(0.0) },
        SupervisedSample { x: vec![-1.0, 0.0], y: Label::Scalar(0.0) },
        SupervisedSample { x: vec![0.0, 1.0], y: Label::Scalar(1.0) },
        SupervisedSample { x: vec![1.0, 1.0], y: Label::Scalar(1.0) },
        SupervisedSample { x: vec![2.0, 1.0], y: Label::Scalar(1.0) },
    ]
}

fn default_xor_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample { x: vec![0.0, 0.0], y: Label::Scalar(0.0) },
        SupervisedSample { x: vec![0.0, 1.0], y: Label::Scalar(1.0) },
        SupervisedSample { x: vec![1.0, 0.0], y: Label::Scalar(1.0) },
        SupervisedSample { x: vec![1.0, 1.0], y: Label::Scalar(0.0) },
    ]
}

/// Top-1 accuracy of softmax(logits) vs integer `labels`.
pub fn multiclass_accuracy(logits: &[Vec<f64>], labels: &[f64]) -> f64 {
    let mut ok = 0usize;
    for i in 0..logits.len() {
        let p = softmax(&logits[i]);
        let mut best = 0usize;
        for k in 1..p.len() {
            if p[k] > p[best] {
                best = k;
            }
        }
        if best as f64 == labels[i] {
            ok += 1;
        }
    }
    ok as f64 / (logits.len().max(1) as f64)
}

#[cfg(test)]
mod tests {
    //! Each learner is run on a small fixture and must reach its known optimum:
    //! the regression lines recover y = 2x + 1, logistic SGD separates a
    //! linearly-separable set, and the backprop MLP learns XOR.

    use super::*;

    #[test]
    fn linear_regression_recovers_line() {
        let result = run_linear_regression_ls(LinearRegressionParams::default());
        // y = 2x + 1 exactly.
        assert!((result.coefficients[0] - 2.0).abs() < 1e-6, "slope = {}", result.coefficients[0]);
        assert!((result.intercept - 1.0).abs() < 1e-6, "intercept = {}", result.intercept);
        assert!(result.mse < 1e-10, "mse = {}", result.mse);
        assert_eq!(result.sample_count, 5);
    }

    #[test]
    fn ridge_regression_shrinks_but_fits_well() {
        let result = run_ridge_regression_ls(RidgeRegressionParams { ridge: Some(0.1), ..Default::default() });
        // Ridge biases the slope slightly below 2 but should stay close.
        assert!(result.coefficients[0] > 1.5 && result.coefficients[0] < 2.0, "slope = {}", result.coefficients[0]);
        assert!(result.mse < 0.5, "mse = {}", result.mse);
    }

    #[test]
    fn logistic_sgd_separates_classes() {
        let result = run_logistic_regression_sgd(LogisticRegressionSGDParams::default());
        assert!(result.accuracy >= 0.99, "accuracy = {}", result.accuracy);
        assert!(result.final_loss < 0.3, "final loss = {}", result.final_loss);
    }

    #[test]
    fn backprop_mlp_learns_xor() {
        let result = run_backprop_mlp_classifier(BackpropMLPParams::default());
        assert_eq!(result.accuracy, 1.0, "accuracy = {}", result.accuracy);
        // XOR predictions must straddle the 0.5 decision boundary correctly.
        assert!(result.predictions[0] < 0.5 && result.predictions[3] < 0.5);
        assert!(result.predictions[1] >= 0.5 && result.predictions[2] >= 0.5);
    }

    #[test]
    fn multiclass_accuracy_argmax() {
        let logits = vec![vec![2.0, 1.0, 0.0], vec![0.0, 0.0, 3.0]];
        assert_eq!(multiclass_accuracy(&logits, &[0.0, 2.0]), 1.0);
        assert_eq!(multiclass_accuracy(&logits, &[1.0, 2.0]), 0.5);
    }
}
