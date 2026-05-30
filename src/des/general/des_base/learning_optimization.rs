//! Port of `src/des/general/des-base/learning-optimization.ts`.
//!
//! Shared stationary stations and movable tokens for supervised learning and
//! vector/candidate optimization models. The DES topology stays explicit:
//! samples, batches, candidates, evaluations, gradients, parameters, and
//! incumbents move through named station channels instead of hiding inside a
//! monolithic solver call.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//!   * The `*Token` classes → plain structs (tokens travel as `Rc<dyn Any>` via
//!     the station core; there is **no `Token` trait** in the ported
//!     `station.rs`, so the TS `implements Token` / `T extends Token` bounds
//!     collapse to a plain `'static` bound — FLAGGED).
//!   * `class …Station extends DESStation` → `struct { core: StationCore, … }`
//!     plus `impl DESStation` (the established pattern in `neural_network.rs`).
//!   * The two `abstract …Station` bases (`GradientOptimizerStation`,
//!     `CandidateEvaluatorStation`) are modelled as **hook trait + generic core
//!     struct + provided template `run_time_step`**: the abstract method becomes
//!     the one required hook method, and the `final`-style tick body is the
//!     template step.
//!   * `optimizer?: 'sgd' | 'adam'` string-union → [`Optimizer`] enum; Adam keeps
//!     `m`/`v` moment buffers.
//!   * `meta: Record<string, unknown>` → the [`Meta`] / [`MetaValue`] stand-in
//!     reused from `neural_network.rs` (FLAGGED: `serde_json` is not a crate
//!     dependency, so the header's `serde_json::Value` is replaced by that
//!     minimal tagged-value map).
//!   * `throw new Error` for bad `batchSize` / `learningRate` / gradient length →
//!     `panic!` (invariant violations).
//!   * The generic math helpers (`zeros`/`dot`/`norm2`) DUPLICATE
//!     `shared::linalg::VecOps`; per the migration header the local computation
//!     is dropped — the public fns are kept as thin wrappers delegating to
//!     [`VecOps`] for API parity. `sigmoid`/`softmax` (no `VecOps` equivalent)
//!     are kept as free fns.

use std::marker::PhantomData;
use std::rc::Rc;

use super::neural_network::{Meta, MetaValue, NumericVector};
use super::runner::{run_iterative_des, IterativeRunOptions, IterativeRunSummary};
use super::station::{DESStation, StationCore, StationRef, DEFAULT_CHANNEL};
use crate::des::shared::linalg::VecOps;

/// Shared `gradient-step` channel name (the TS `GradientOptimizerStation.CH_STEP`
/// that `GradientTraceSinkStation` re-uses).
const CH_GRADIENT_STEP: &str = "gradient-step";
/// Shared `evaluated` channel name (`CandidateEvaluatorStation.CH_EVALUATED` /
/// `IncumbentSinkStation.CH_EVALUATED`).
const CH_EVALUATED: &str = "evaluated";

/// Summary of a station graph (ids only): the `StationGraphSummary` interface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StationGraphSummary {
    pub stations: Vec<String>,
    pub movables: Vec<String>,
    pub edges: Vec<String>,
}

// ── Tokens ───────────────────────────────────────────────────────────────────

/// A single supervised sample (`input → target`) with an optional weight.
#[derive(Clone, Debug)]
pub struct VectorSampleToken {
    pub id: String,
    pub input: NumericVector,
    pub target: NumericVector,
    pub weight: f64,
    pub meta: Meta,
}

impl VectorSampleToken {
    /// `weight` defaults to `1`, `meta` to empty (matching the TS defaults).
    pub fn new(id: impl Into<String>, input: NumericVector, target: NumericVector) -> Self {
        VectorSampleToken { id: id.into(), input, target, weight: 1.0, meta: Meta::new() }
    }
}

/// A grouped batch of samples for one optimizer step.
#[derive(Clone, Debug)]
pub struct VectorBatchToken {
    pub id: String,
    pub samples: Vec<VectorSampleToken>,
    pub epoch: usize,
    pub batch_index: usize,
}

/// A single recorded gradient step (loss, ‖∇‖, current parameters).
#[derive(Clone, Debug)]
pub struct GradientStepToken {
    pub step: usize,
    pub loss: f64,
    pub gradient_norm: f64,
    pub parameters: NumericVector,
    pub meta: Meta,
}

/// A candidate solution awaiting evaluation.
#[derive(Clone, Debug)]
pub struct CandidateToken<S> {
    pub id: String,
    pub candidate: S,
    pub meta: Meta,
}

impl<S> CandidateToken<S> {
    pub fn new(id: impl Into<String>, candidate: S) -> Self {
        CandidateToken { id: id.into(), candidate, meta: Meta::new() }
    }
}

/// A candidate together with its objective value and feasibility flag.
#[derive(Clone, Debug)]
pub struct EvaluatedCandidateToken<S> {
    pub id: String,
    pub candidate: S,
    pub objective: f64,
    pub feasible: bool,
    pub meta: Meta,
}

impl<S> EvaluatedCandidateToken<S> {
    /// `feasible` defaults to `true`, `meta` to empty (matching the TS defaults).
    pub fn new(id: impl Into<String>, candidate: S, objective: f64) -> Self {
        EvaluatedCandidateToken { id: id.into(), candidate, objective, feasible: true, meta: Meta::new() }
    }
}

/// The best feasible candidate seen so far, plus how many were evaluated.
#[derive(Clone, Debug)]
pub struct IncumbentToken<S> {
    pub id: String,
    pub candidate: S,
    pub objective: f64,
    pub evaluation_count: usize,
    pub meta: Meta,
}

// ── VectorSampleSourceStation ─────────────────────────────────────────────────

/// Emits the configured sample set once per epoch, tagging `meta.epoch`.
pub struct VectorSampleSourceStation {
    core: StationCore,
    samples: Vec<VectorSampleToken>,
    epochs: usize,
    epoch: usize,
}

impl VectorSampleSourceStation {
    pub const CH_SAMPLE: &'static str = "sample";

    /// `epochs` defaults to `1` in the TS; pass it explicitly here.
    pub fn new(id: impl Into<String>, samples: Vec<VectorSampleToken>, epochs: usize) -> Self {
        VectorSampleSourceStation { core: StationCore::new(id), samples, epochs, epoch: 0 }
    }

    pub fn get_epoch(&self) -> usize {
        self.epoch
    }
}

impl DESStation for VectorSampleSourceStation {
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
        self.epoch < self.epochs
    }
    fn run_time_step(&mut self) {
        if self.epoch >= self.epochs {
            return;
        }
        let epoch = self.epoch;
        // Clone the source samples first so we don't hold a borrow of
        // `self.samples` while mutating `self.core` via `emit`.
        let outgoing: Vec<VectorSampleToken> = self
            .samples
            .iter()
            .map(|sample| {
                let mut meta = sample.meta.clone();
                meta.insert("epoch".to_string(), MetaValue::Number(epoch as f64));
                VectorSampleToken {
                    id: format!("{}:e{epoch}", sample.id),
                    input: sample.input.clone(),
                    target: sample.target.clone(),
                    weight: sample.weight,
                    meta,
                }
            })
            .collect();
        for token in outgoing {
            self.core.emit(Rc::new(token), Self::CH_SAMPLE);
        }
        self.epoch += 1;
    }
}

// ── MiniBatchStation ──────────────────────────────────────────────────────────

/// Buffers samples and emits fixed-size batches (optionally flushing a partial
/// trailing batch when no more input arrives).
pub struct MiniBatchStation {
    core: StationCore,
    batch_size: usize,
    flush_partial: bool,
    buffer: Vec<VectorSampleToken>,
    batch_index: usize,
}

impl MiniBatchStation {
    pub const CH_SAMPLE: &'static str = VectorSampleSourceStation::CH_SAMPLE;
    pub const CH_BATCH: &'static str = "batch";

    /// `flush_partial` defaults to `true` in the TS. Panics on a non-positive
    /// `batch_size` (the TS `throw`; `usize` already guarantees integrality).
    pub fn new(id: impl Into<String>, batch_size: usize, flush_partial: bool) -> Self {
        if batch_size == 0 {
            panic!("batchSize must be a positive integer");
        }
        MiniBatchStation {
            core: StationCore::new(id),
            batch_size,
            flush_partial,
            buffer: Vec::new(),
            batch_index: 0,
        }
    }

    pub fn get_batch_count(&self) -> usize {
        self.batch_index
    }

    fn emit_next_batch(&mut self, size: usize) {
        let samples: Vec<VectorSampleToken> = self.buffer.drain(0..size).collect();
        let epoch = samples
            .first()
            .and_then(|s| s.meta.get("epoch"))
            .and_then(|v| match v {
                MetaValue::Number(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);
        let token = VectorBatchToken {
            id: format!("batch-{}", self.batch_index),
            samples,
            epoch,
            batch_index: self.batch_index,
        };
        self.core.emit(Rc::new(token), Self::CH_BATCH);
        self.batch_index += 1;
    }
}

impl DESStation for MiniBatchStation {
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
        self.core.inbox_size(Self::CH_SAMPLE) > 0 || !self.buffer.is_empty()
    }
    fn run_time_step(&mut self) {
        let incoming = self.core.drain::<VectorSampleToken>(Self::CH_SAMPLE);
        let incoming_len = incoming.len();
        self.buffer.extend(incoming.into_iter().map(|rc| (*rc).clone()));
        while self.buffer.len() >= self.batch_size {
            let size = self.batch_size;
            self.emit_next_batch(size);
        }
        if incoming_len == 0 && self.flush_partial && !self.buffer.is_empty() {
            let size = self.buffer.len();
            self.emit_next_batch(size);
        }
    }
}

// ── GradientOptimizerStation (template-method base) ───────────────────────────

/// The result of evaluating one batch: a scalar loss + a gradient w.r.t. the
/// parameters (`GradientEvaluation`).
#[derive(Clone, Debug)]
pub struct GradientEvaluation {
    pub loss: f64,
    pub gradient: NumericVector,
    pub meta: Option<Meta>,
}

/// First-order optimizer choice (`'sgd' | 'adam'` string-union → enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optimizer {
    Sgd,
    Adam,
}

/// Options for [`GradientOptimizerStation`]. `Option` fields take the TS
/// defaults when `None`.
#[derive(Clone, Debug)]
pub struct GradientOptimizerOptions {
    pub initial_parameters: NumericVector,
    pub learning_rate: f64,
    pub optimizer: Option<Optimizer>,
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    pub epsilon: Option<f64>,
}

/// The abstract template-method hook: subclasses supply `evaluate_batch`.
/// (`protected abstract evaluateBatch(batch, parameters): GradientEvaluation`.)
pub trait GradientOptimizerHook {
    fn evaluate_batch(&mut self, batch: &VectorBatchToken, parameters: &[f64]) -> GradientEvaluation;
}

/// Drains batches, runs the (SGD or Adam) update template, and emits a
/// [`GradientStepToken`] per step. The abstract `GradientOptimizerStation` base
/// becomes this generic core + the [`GradientOptimizerHook`] trait.
pub struct GradientOptimizerStation<H: GradientOptimizerHook + 'static> {
    core: StationCore,
    hook: H,
    parameters: NumericVector,
    step: usize,
    loss_history: Vec<f64>,
    gradient_norm_history: Vec<f64>,
    learning_rate: f64,
    optimizer: Optimizer,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: NumericVector,
    v: NumericVector,
}

impl<H: GradientOptimizerHook + 'static> GradientOptimizerStation<H> {
    pub const CH_BATCH: &'static str = MiniBatchStation::CH_BATCH;
    pub const CH_STEP: &'static str = CH_GRADIENT_STEP;

    /// Panics on a non-positive `learning_rate` (the TS `throw`).
    pub fn new(id: impl Into<String>, hook: H, opts: GradientOptimizerOptions) -> Self {
        if opts.learning_rate <= 0.0 {
            panic!("learningRate must be positive");
        }
        let parameters = opts.initial_parameters.clone();
        let n = parameters.len();
        GradientOptimizerStation {
            core: StationCore::new(id),
            hook,
            parameters,
            step: 0,
            loss_history: Vec::new(),
            gradient_norm_history: Vec::new(),
            learning_rate: opts.learning_rate,
            optimizer: opts.optimizer.unwrap_or(Optimizer::Sgd),
            beta1: opts.beta1.unwrap_or(0.9),
            beta2: opts.beta2.unwrap_or(0.999),
            epsilon: opts.epsilon.unwrap_or(1e-8),
            m: VecOps::zeros(n),
            v: VecOps::zeros(n),
        }
    }

    pub fn get_parameters(&self) -> NumericVector {
        self.parameters.clone()
    }
    pub fn get_step(&self) -> usize {
        self.step
    }
    pub fn get_loss_history(&self) -> NumericVector {
        self.loss_history.clone()
    }
    pub fn get_gradient_norm_history(&self) -> NumericVector {
        self.gradient_norm_history.clone()
    }
    pub fn hook(&self) -> &H {
        &self.hook
    }
    pub fn hook_mut(&mut self) -> &mut H {
        &mut self.hook
    }

    fn apply_gradient(&mut self, gradient: &[f64]) {
        if self.optimizer == Optimizer::Sgd {
            for i in 0..self.parameters.len() {
                self.parameters[i] -= self.learning_rate * gradient[i];
            }
            return;
        }
        for i in 0..self.parameters.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];
            let m_hat = self.m[i] / (1.0 - self.beta1.powf(self.step as f64));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(self.step as f64));
            self.parameters[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

impl<H: GradientOptimizerHook + 'static> DESStation for GradientOptimizerStation<H> {
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
        self.core.inbox_size(Self::CH_BATCH) > 0
    }
    fn run_time_step(&mut self) {
        let batches = self.core.drain::<VectorBatchToken>(Self::CH_BATCH);
        for batch in batches {
            let evaluation = self.hook.evaluate_batch(&batch, &self.parameters);
            if evaluation.gradient.len() != self.parameters.len() {
                panic!(
                    "gradient length {} != parameter length {}",
                    evaluation.gradient.len(),
                    self.parameters.len()
                );
            }
            self.step += 1;
            self.apply_gradient(&evaluation.gradient);
            let grad_norm = VecOps::norm2(&evaluation.gradient);
            self.loss_history.push(evaluation.loss);
            self.gradient_norm_history.push(grad_norm);
            let token = GradientStepToken {
                step: self.step,
                loss: evaluation.loss,
                gradient_norm: grad_norm,
                parameters: self.parameters.clone(),
                meta: evaluation.meta.unwrap_or_default(),
            };
            self.core.emit(Rc::new(token), Self::CH_STEP);
        }
    }
}

// ── GradientTraceSinkStation ──────────────────────────────────────────────────

/// Collects every [`GradientStepToken`] into `trace`.
pub struct GradientTraceSinkStation {
    core: StationCore,
    pub trace: Vec<GradientStepToken>,
}

impl GradientTraceSinkStation {
    pub const CH_STEP: &'static str = CH_GRADIENT_STEP;

    pub fn new(id: impl Into<String>) -> Self {
        GradientTraceSinkStation { core: StationCore::new(id), trace: Vec::new() }
    }
}

impl DESStation for GradientTraceSinkStation {
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
        self.core.inbox_size(Self::CH_STEP) > 0
    }
    fn run_time_step(&mut self) {
        let steps = self.core.drain::<GradientStepToken>(Self::CH_STEP);
        self.trace.extend(steps.into_iter().map(|rc| (*rc).clone()));
    }
}

// ── CandidateSourceStation ────────────────────────────────────────────────────

/// Emits the configured candidate set once.
pub struct CandidateSourceStation<S: Clone + 'static> {
    core: StationCore,
    candidates: Vec<CandidateToken<S>>,
    emitted: bool,
}

impl<S: Clone + 'static> CandidateSourceStation<S> {
    pub const CH_CANDIDATE: &'static str = "candidate";

    pub fn new(id: impl Into<String>, candidates: Vec<CandidateToken<S>>) -> Self {
        CandidateSourceStation { core: StationCore::new(id), candidates, emitted: false }
    }
}

impl<S: Clone + 'static> DESStation for CandidateSourceStation<S> {
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
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let outgoing = self.candidates.clone();
        for candidate in outgoing {
            self.core.emit(Rc::new(candidate), Self::CH_CANDIDATE);
        }
        self.emitted = true;
    }
}

// ── CandidateEvaluatorStation (template-method base) ──────────────────────────

/// The abstract template-method hook: subclasses supply `evaluate_candidate`.
/// (`protected abstract evaluateCandidate(token): EvaluatedCandidateToken<S>`.)
pub trait CandidateEvaluatorHook<S> {
    fn evaluate_candidate(&mut self, token: &CandidateToken<S>) -> EvaluatedCandidateToken<S>;
}

/// Drains candidates, scores each via the [`CandidateEvaluatorHook`], and emits
/// an [`EvaluatedCandidateToken`]. The abstract `CandidateEvaluatorStation` base
/// becomes this generic core + the hook trait.
pub struct CandidateEvaluatorStation<S: 'static, H: CandidateEvaluatorHook<S> + 'static> {
    core: StationCore,
    hook: H,
    _marker: PhantomData<fn() -> S>,
}

impl<S: 'static, H: CandidateEvaluatorHook<S> + 'static> CandidateEvaluatorStation<S, H> {
    pub const CH_CANDIDATE: &'static str = CandidateSourceStation::<()>::CH_CANDIDATE;
    pub const CH_EVALUATED: &'static str = CH_EVALUATED;

    pub fn new(id: impl Into<String>, hook: H) -> Self {
        CandidateEvaluatorStation { core: StationCore::new(id), hook, _marker: PhantomData }
    }

    pub fn hook(&self) -> &H {
        &self.hook
    }
    pub fn hook_mut(&mut self) -> &mut H {
        &mut self.hook
    }
}

impl<S: 'static, H: CandidateEvaluatorHook<S> + 'static> DESStation for CandidateEvaluatorStation<S, H> {
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
        self.core.inbox_size(Self::CH_CANDIDATE) > 0
    }
    fn run_time_step(&mut self) {
        let candidates = self.core.drain::<CandidateToken<S>>(Self::CH_CANDIDATE);
        for candidate in candidates {
            let evaluated = self.hook.evaluate_candidate(&candidate);
            self.core.emit(Rc::new(evaluated), Self::CH_EVALUATED);
        }
    }
}

// ── IncumbentSinkStation ──────────────────────────────────────────────────────

/// Tracks the best feasible (minimum-objective) candidate seen so far and emits
/// an [`IncumbentToken`] on the default channel whenever an incumbent exists.
pub struct IncumbentSinkStation<S: Clone + 'static> {
    core: StationCore,
    pub evaluations: Vec<EvaluatedCandidateToken<S>>,
    incumbent: Option<EvaluatedCandidateToken<S>>,
}

impl<S: Clone + 'static> IncumbentSinkStation<S> {
    pub const CH_EVALUATED: &'static str = CH_EVALUATED;

    pub fn new(id: impl Into<String>) -> Self {
        IncumbentSinkStation { core: StationCore::new(id), evaluations: Vec::new(), incumbent: None }
    }

    pub fn get_incumbent(&self) -> Option<&EvaluatedCandidateToken<S>> {
        self.incumbent.as_ref()
    }
}

impl<S: Clone + 'static> DESStation for IncumbentSinkStation<S> {
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
        self.core.inbox_size(Self::CH_EVALUATED) > 0
    }
    fn run_time_step(&mut self) {
        let evaluated = self.core.drain::<EvaluatedCandidateToken<S>>(Self::CH_EVALUATED);
        for item in evaluated {
            let item = (*item).clone();
            self.evaluations.push(item.clone());
            if !item.feasible {
                continue;
            }
            match &self.incumbent {
                Some(inc) if item.objective >= inc.objective => {}
                _ => self.incumbent = Some(item),
            }
        }
        if let Some(inc) = &self.incumbent {
            let token = IncumbentToken {
                id: inc.id.clone(),
                candidate: inc.candidate.clone(),
                objective: inc.objective,
                evaluation_count: self.evaluations.len(),
                meta: inc.meta.clone(),
            };
            self.core.emit(Rc::new(token), DEFAULT_CHANNEL);
        }
    }
}

// ── SingleTokenSourceStation ──────────────────────────────────────────────────

/// Emits a single lazily-built token (built once, validated, then emitted).
///
/// `T extends Token` → `T: 'static` (there is no `Token` trait in the ported
/// `station.rs`). The `tokenFactory` / `validateToken` closures become boxed
/// `FnMut` / `Fn`.
pub struct SingleTokenSourceStation<T: 'static> {
    core: StationCore,
    output_channel: String,
    token_factory: Box<dyn FnMut() -> T>,
    validate_token: Box<dyn Fn(&T)>,
    emitted: bool,
    token: Option<Rc<T>>,
}

impl<T: 'static> SingleTokenSourceStation<T> {
    /// `validateToken` defaults to a no-op (the TS `() => {}`).
    pub fn new(
        id: impl Into<String>,
        output_channel: impl Into<String>,
        token_factory: impl FnMut() -> T + 'static,
    ) -> Self {
        Self::with_validator(id, output_channel, token_factory, |_| {})
    }

    pub fn with_validator(
        id: impl Into<String>,
        output_channel: impl Into<String>,
        token_factory: impl FnMut() -> T + 'static,
        validate_token: impl Fn(&T) + 'static,
    ) -> Self {
        SingleTokenSourceStation {
            core: StationCore::new(id),
            output_channel: output_channel.into(),
            token_factory: Box::new(token_factory),
            validate_token: Box::new(validate_token),
            emitted: false,
            token: None,
        }
    }

    fn initial_token(&mut self) -> Rc<T> {
        if self.token.is_none() {
            let built = (self.token_factory)();
            (self.validate_token)(&built);
            self.token = Some(Rc::new(built));
        }
        self.token.clone().unwrap()
    }
}

impl<T: 'static> DESStation for SingleTokenSourceStation<T> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn assert_preconditions(&mut self) {
        self.initial_token();
    }
    fn has_work(&self) -> bool {
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let token = self.initial_token();
        self.core.emit(token, &self.output_channel);
        self.emitted = true;
    }
}

// ── LatestTokenSinkStation ────────────────────────────────────────────────────

/// Keeps only the most recently received token on a channel.
pub struct LatestTokenSinkStation<T: 'static> {
    core: StationCore,
    input_channel: String,
    pub latest: Option<Rc<T>>,
}

impl<T: 'static> LatestTokenSinkStation<T> {
    pub fn new(id: impl Into<String>, input_channel: impl Into<String>) -> Self {
        LatestTokenSinkStation { core: StationCore::new(id), input_channel: input_channel.into(), latest: None }
    }
}

impl<T: 'static> DESStation for LatestTokenSinkStation<T> {
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
        self.core.inbox_size(&self.input_channel) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self.core.drain::<T>(&self.input_channel);
        if let Some(last) = tokens.last() {
            self.latest = Some(last.clone());
        }
    }
}

// ── Topology helpers ──────────────────────────────────────────────────────────

/// A station reference or a bare id string (the TS `DESStation | string`
/// union). Resolves to an id via [`StationOrId::id`].
#[derive(Clone)]
pub enum StationOrId {
    Id(String),
    Station(StationRef),
}

impl StationOrId {
    pub fn id(&self) -> String {
        match self {
            StationOrId::Id(s) => s.clone(),
            StationOrId::Station(s) => s.borrow().id().to_string(),
        }
    }
}

impl From<&str> for StationOrId {
    fn from(s: &str) -> Self {
        StationOrId::Id(s.to_string())
    }
}
impl From<String> for StationOrId {
    fn from(s: String) -> Self {
        StationOrId::Id(s)
    }
}
impl From<StationRef> for StationOrId {
    fn from(s: StationRef) -> Self {
        StationOrId::Station(s)
    }
}

/// Build a [`StationGraphSummary`] from stations (resolved to ids), movable
/// ids, and edge labels.
pub fn station_graph(stations: &[StationOrId], movables: &[String], edges: &[String]) -> StationGraphSummary {
    StationGraphSummary {
        stations: stations.iter().map(StationOrId::id).collect(),
        movables: movables.to_vec(),
        edges: edges.to_vec(),
    }
}

/// The empty graph.
pub fn empty_station_graph() -> StationGraphSummary {
    station_graph(&[], &[], &[])
}

/// Format a directed channel edge label: `"src:srcChan -> tgt:tgtChan"`.
/// `target_channel` defaults to `source_channel` when `None`.
pub fn channel_edge(
    source: &StationOrId,
    source_channel: &str,
    target: &StationOrId,
    target_channel: Option<&str>,
) -> String {
    let target_channel = target_channel.unwrap_or(source_channel);
    format!("{}:{source_channel} -> {}:{target_channel}", source.id(), target.id())
}

/// A self-looping state-update topology: `source → update`, `update → update`,
/// `update → sink`.
pub fn state_loop_topology(
    source: &dyn DESStation,
    update: &dyn DESStation,
    sink: &dyn DESStation,
    state_channel: &str,
    result_channel: &str,
    movables: &[String],
) -> StationGraphSummary {
    let s = StationOrId::Id(source.id().to_string());
    let u = StationOrId::Id(update.id().to_string());
    let k = StationOrId::Id(sink.id().to_string());
    let edges = vec![
        channel_edge(&s, state_channel, &u, Some(state_channel)),
        channel_edge(&u, state_channel, &u, Some(state_channel)),
        channel_edge(&u, result_channel, &k, Some(result_channel)),
    ];
    station_graph(&[s.clone(), u.clone(), k.clone()], movables, &edges)
}

/// Wire and run a self-looping state-update pipeline.
///
/// FLAGGED divergence: the TS spread `{shuffle: false, ...opts}` let a caller
/// override `shuffle`; the Rust port always forces `shuffle = false` (the
/// deterministic intent and the TS default for an absent option). All other
/// `opts` fields are honoured.
pub fn run_state_loop_pipeline(
    source: StationRef,
    update: StationRef,
    sink: StationRef,
    state_channel: &str,
    result_channel: &str,
    mut opts: IterativeRunOptions,
) -> IterativeRunSummary {
    source.borrow_mut().core_mut().pipe(update.clone(), state_channel, state_channel);
    update.borrow_mut().core_mut().pipe(update.clone(), state_channel, state_channel);
    update.borrow_mut().core_mut().pipe(sink.clone(), result_channel, result_channel);
    opts.shuffle = false;
    run_iterative_des(vec![source, update, sink], opts)
}

// ── Small array / math helpers ────────────────────────────────────────────────

/// Return `value` (cloned) if non-empty, else `fallback` (cloned).
pub fn non_empty_array<T: Clone>(value: Option<&[T]>, fallback: &[T]) -> Vec<T> {
    match value {
        Some(v) if !v.is_empty() => v.to_vec(),
        _ => fallback.to_vec(),
    }
}

/// Deep-copy a row-major matrix.
pub fn clone_matrix<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    matrix.iter().cloned().collect()
}

/// Zero vector of length `n`. Thin wrapper over [`VecOps::zeros`] (the local
/// copy is dropped per the migration header).
pub fn zeros(n: usize) -> NumericVector {
    VecOps::zeros(n)
}

/// Dot product. Thin wrapper over [`VecOps::dot`].
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    VecOps::dot(a, b)
}

/// Euclidean norm. Thin wrapper over [`VecOps::norm2`].
pub fn norm2(v: &[f64]) -> f64 {
    VecOps::norm2(v)
}

/// Numerically-stable logistic sigmoid (no `VecOps` equivalent; kept local).
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Numerically-stable softmax (no `VecOps` equivalent; kept local).
pub fn softmax(logits: &[f64]) -> NumericVector {
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|v| v / z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Convex quadratic: loss = Σ(p−target)², gradient = 2(p−target). A first-
    /// order optimizer must drive the loss down toward zero.
    struct QuadHook {
        target: Vec<f64>,
    }

    impl GradientOptimizerHook for QuadHook {
        fn evaluate_batch(&mut self, _batch: &VectorBatchToken, parameters: &[f64]) -> GradientEvaluation {
            let mut loss = 0.0;
            let gradient: Vec<f64> = parameters
                .iter()
                .zip(&self.target)
                .map(|(p, t)| {
                    let e = p - t;
                    loss += e * e;
                    2.0 * e
                })
                .collect();
            GradientEvaluation { loss, gradient, meta: None }
        }
    }

    #[test]
    fn gradient_optimizer_reduces_loss() {
        let mut station = GradientOptimizerStation::new(
            "opt",
            QuadHook { target: vec![1.0, -2.0] },
            GradientOptimizerOptions {
                initial_parameters: vec![0.0, 0.0],
                learning_rate: 0.1,
                optimizer: Some(Optimizer::Sgd),
                beta1: None,
                beta2: None,
                epsilon: None,
            },
        );

        for _ in 0..40 {
            station.core_mut().take(
                Rc::new(VectorBatchToken {
                    id: "b".to_string(),
                    samples: vec![],
                    epoch: 0,
                    batch_index: 0,
                }),
                GradientOptimizerStation::<QuadHook>::CH_BATCH,
            );
            station.run_time_step();
        }

        let history = station.get_loss_history();
        assert_eq!(history.len(), 40);
        assert!(history[history.len() - 1] < history[0], "loss should drop over steps");
        assert!(history[history.len() - 1] < 1e-4, "loss should converge near zero");
        let p = station.get_parameters();
        assert!((p[0] - 1.0).abs() < 1e-2 && (p[1] + 2.0).abs() < 1e-2);
    }

    #[test]
    fn mini_batch_station_groups_and_flushes() {
        let mut station = MiniBatchStation::new("mb", 2, true);
        for i in 0..3 {
            station.core_mut().take(
                Rc::new(VectorSampleToken::new(format!("s{i}"), vec![i as f64], vec![0.0])),
                MiniBatchStation::CH_SAMPLE,
            );
        }
        // First tick: one full batch of 2 emitted, 1 sample left buffered.
        station.run_time_step();
        assert_eq!(station.get_batch_count(), 1);
        assert!(station.has_work());
        // Second tick: no input, so the partial trailing batch is flushed.
        station.run_time_step();
        assert_eq!(station.get_batch_count(), 2);
        assert!(!station.has_work());
    }

    struct SquareEval;
    impl CandidateEvaluatorHook<f64> for SquareEval {
        fn evaluate_candidate(&mut self, token: &CandidateToken<f64>) -> EvaluatedCandidateToken<f64> {
            EvaluatedCandidateToken::new(token.id.clone(), token.candidate, token.candidate * token.candidate)
        }
    }

    #[test]
    fn candidate_pipeline_selects_min_incumbent() {
        let source = Rc::new(RefCell::new(CandidateSourceStation::new(
            "src",
            vec![
                CandidateToken::new("a", 3.0_f64),
                CandidateToken::new("b", -1.0_f64),
                CandidateToken::new("c", 2.0_f64),
            ],
        )));
        let evaluator =
            Rc::new(RefCell::new(CandidateEvaluatorStation::<f64, SquareEval>::new("eval", SquareEval)));
        let sink = Rc::new(RefCell::new(IncumbentSinkStation::<f64>::new("sink")));

        source.borrow_mut().core_mut().pipe(
            evaluator.clone() as StationRef,
            CandidateSourceStation::<f64>::CH_CANDIDATE,
            CandidateEvaluatorStation::<f64, SquareEval>::CH_CANDIDATE,
        );
        evaluator.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            CandidateEvaluatorStation::<f64, SquareEval>::CH_EVALUATED,
            IncumbentSinkStation::<f64>::CH_EVALUATED,
        );

        let summary = run_iterative_des(
            vec![source.clone() as StationRef, evaluator.clone() as StationRef, sink.clone() as StationRef],
            IterativeRunOptions { shuffle: false, ..Default::default() },
        );
        assert!(summary.ticks >= 1);

        let sink_ref = sink.borrow();
        let inc = sink_ref.get_incumbent().expect("an incumbent should be chosen");
        assert_eq!(inc.id, "b");
        assert!((inc.objective - 1.0).abs() < 1e-12);
        assert_eq!(sink_ref.evaluations.len(), 3);
    }
}
