//! Port of `src/des/general/math-blocks.ts`.
//!
//! Calculus / control block diagrams (sources, sums, gains, integrators,
//! differentiators, filters, comparators, logic, expressions, Laplacians) as
//! DES `VisualBlock` stations. Numeric [`MathSignal`] tokens move between blocks
//! over named channels.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//! * `abstract class MathBlock extends VisualBlock` → the [`MathBlock`] trait
//!   (shared step/precondition logic) plus a [`MathBlockCore`] state struct that
//!   *composes* a [`VisualBlock`] (Rust has no inheritance). Every concrete block
//!   embeds a `MathBlockCore` field `m` and implements both [`MathBlock`] and
//!   [`DESStation`] (the latter via the [`math_block_station!`] macro, which
//!   delegates `core`/`core_mut`/`run_time_step`/`has_work`/`assert_preconditions`
//!   to the trait + core).
//! * `SubtractBlock extends SumBlock` → composition: [`SubtractBlock`] wraps a
//!   configured [`SumBlock`] and forwards the `MathBlock` methods to it.
//! * `FunctionSourceBlock`'s `(t, tick) => number` closure → a
//!   `Box<dyn Fn(f64, usize) -> f64>` field.
//! * `IntegratorMethod` / `ComparatorOp` / `LogicOp` string unions → enums.
//! * `MathSignal extends Token` (`metadata?: Record<string, unknown>`) → a
//!   `'static` struct stored as `Rc<dyn Any>` and recovered by `drain::<MathSignal>()`.
//!
//! ## PORT NOTEs (cross-module / behavioural)
//!
//! * **Validators.** The TS base constructor registers an `intrinsicCheck`
//!   finite-output validator on every block. Registering it on `StationCore`
//!   would require the validator closure to downcast `&dyn DESStation` back to
//!   each concrete block type (one wiring site per block). The observable output
//!   is identical if the check is computed by the driver, so
//!   [`run_math_block_diagram`] computes the per-block
//!   `math-block-finite-output/<id>` checks directly and writes them into both
//!   the returned `validation` and `summary.validation`.
//! * **`MathSignal.metadata`.** In TS only `ExpressionBlock` ever sets metadata
//!   (`{expression}`), and no consumer reads it; the field is omitted here.
//! * **Wrong-token guard.** `drainMath` in TS `throw`s if a non-`MathSignal`
//!   token arrives. Here `drain::<MathSignal>()` is type-safe and only yields
//!   `MathSignal`s, so that throw path is unreachable; the finite-value/time
//!   guards are preserved.
//! * **Heterogeneous block list.** `runMathBlockDiagram(blocks)` takes mixed
//!   block types. Rust models this as `Vec<MathBlockHandle>`, where each handle
//!   bundles a `StationRef` (for the runner) and a `Rc<RefCell<dyn MathBlockObj>>`
//!   (for reading `outputHistory` / visual specs). Both are built by *sized*
//!   coercion from the same concrete `Rc`, so no trait-upcasting is required.

use std::any::Any;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::expr::{evaluate, parse, Env, Expr};
use super::des_base::preconditions::{Check, Preconditions};
use super::des_base::runner::{run_iterative_des, IterativeRunOptions, IterativeRunSummary};
use super::des_base::station::{AnyToken, DESStation, StationCore, StationRef};
use super::des_base::validation::ValidationCheck;
use super::des_base::visual_block::{
    visual_block_specs, VisualBlock, VisualBlockOptions, VisualBlockPortSpec, VisualBlockSpec,
    VisualBlockStyle, VisualPortInput, VisualPortOptions,
};

/// Default input channel name (`MATH_IN = 'in'`).
pub const MATH_IN: &str = "in";
/// Default output channel name (`MATH_OUT = 'out'`).
pub const MATH_OUT: &str = "out";

/// Convert a recoverable [`Check`] into a `panic!` (the TS guards `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Logging
// =============================================================================

/// `interface BlockModelLogger`.
///
/// PORT NOTE: the TS `log` accepts a variadic `{kind, level?, [k]: unknown}`
/// payload. It is modelled here as a `kind`/`level` plus stringified extra
/// `fields`, which is sufficient for the instrumentation the engine emits.
pub trait BlockModelLogger {
    fn log(&self, event: LogEvent);
}

/// A single structured log event for a [`BlockModelLogger`].
#[derive(Clone, Debug)]
pub struct LogEvent {
    pub kind: String,
    pub level: Option<String>,
    pub fields: Vec<(String, String)>,
}

// =============================================================================
// Signals / samples / options / result
// =============================================================================

/// `interface MathSignal extends Token`. Stored as `Rc<dyn Any>` on a channel.
#[derive(Clone, Debug)]
pub struct MathSignal {
    pub kind: &'static str,
    pub source_id: String,
    pub channel: String,
    pub tick: usize,
    pub time: f64,
    pub value: f64,
}

/// `interface MathSample`.
#[derive(Clone, Debug)]
pub struct MathSample {
    pub block_id: String,
    pub channel: String,
    pub tick: usize,
    pub time: f64,
    pub value: f64,
}

/// `interface MathBlockOptions`.
#[derive(Clone, Copy, Debug)]
pub struct MathBlockOptions {
    pub dt: f64,
    /// Number of `run_time_step` executions. A trace over N integration steps
    /// uses N + 1 ticks.
    pub ticks: usize,
    pub t0: Option<f64>,
}

/// `interface MathBlockRunResult`.
#[derive(Debug)]
pub struct MathBlockRunResult {
    pub summary: IterativeRunSummary,
    pub validation: Vec<ValidationCheck>,
    pub outputs: Vec<MathSample>,
    pub visual_blocks: Vec<VisualBlockSpec>,
}

// =============================================================================
// Free helpers (module-private, mirroring the TS file-scope functions)
// =============================================================================

/// `/^[A-Za-z_][A-Za-z0-9_]*$/`.
fn is_valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn assert_name(model: &str, param: &str, name: &str) {
    require(Preconditions::check(
        model,
        param,
        "match /^[A-Za-z_][A-Za-z0-9_]*$/",
        is_valid_identifier(name),
        Some(name.to_string()),
    ));
}

fn assert_unique(model: &str, param: &str, values: &[String]) {
    let mut seen: HashSet<&str> = HashSet::new();
    for value in values {
        require(Preconditions::check(
            model,
            param,
            "contain unique names",
            !seen.contains(value.as_str()),
            Some(value.clone()),
        ));
        seen.insert(value.as_str());
    }
}

fn last_owned(xs: &[MathSignal]) -> Option<MathSignal> {
    xs.last().cloned()
}

/// `latestUnusedAtOrBefore`.
fn latest_unused_at_or_before(
    signals: &[MathSignal],
    tick: usize,
    consumed_through_tick: i64,
) -> Option<MathSignal> {
    let mut best: Option<MathSignal> = None;
    for s in signals {
        let st = s.tick as i64;
        if st > tick as i64 || st <= consumed_through_tick {
            continue;
        }
        match &best {
            None => best = Some(s.clone()),
            Some(b) if s.tick >= b.tick => best = Some(s.clone()),
            _ => {}
        }
    }
    best
}

fn duration_steps(model: &str, t0: f64, t1: f64, dt: f64) -> usize {
    require(Preconditions::finite(model, "t0", t0));
    require(Preconditions::finite(model, "t1", t1));
    require(Preconditions::positive(model, "dt", dt));
    require(Preconditions::check(
        model,
        "t1",
        "be greater than t0",
        t1 > t0,
        Some(format!("{{t0: {t0}, t1: {t1}}}")),
    ));
    let exact = (t1 - t0) / dt;
    let steps = exact.round();
    require(Preconditions::check(
        model,
        "duration/dt",
        "be an integer number of steps",
        (exact - steps).abs() <= 1e-9 * 1.0_f64.max(exact.abs()),
        Some(exact.to_string()),
    ));
    require(Preconditions::integer_in_range(model, "steps", steps, 1.0, 1_000_000.0));
    steps as usize
}

fn finite_record(model: &str, param: &str, r: Option<&HashMap<String, f64>>) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    let Some(r) = r else {
        return out;
    };
    for (k, v) in r {
        assert_name(model, &format!("{param}.{k}"), k);
        require(Preconditions::finite(model, &format!("{param}.{k}"), *v));
        out.insert(k.clone(), *v);
    }
    out
}

// =============================================================================
// Enums (string unions → enums)
// =============================================================================

/// `type IntegratorMethod = 'euler' | 'trapezoid'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegratorMethod {
    Euler,
    Trapezoid,
}

impl IntegratorMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            IntegratorMethod::Euler => "euler",
            IntegratorMethod::Trapezoid => "trapezoid",
        }
    }
}

/// `type ComparatorOp = 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparatorOp {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
}

/// `type LogicOp = 'and' | 'or' | 'not' | 'xor'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Not,
    Xor,
}

// =============================================================================
// MathBlockCore — the shared state of the TS `abstract class MathBlock`.
// =============================================================================

/// Shared state for every math block (fields of the TS `abstract class`).
pub struct MathBlockCore {
    pub id: String,
    visual: VisualBlock,
    current_tick: usize,
    latest_by_channel: HashMap<String, MathSignal>,
    pub output_history: Vec<MathSample>,
    pub dt: f64,
    pub ticks: usize,
    pub t0: f64,
}

impl MathBlockCore {
    /// `super(id, {kind: 'math-block', ports: …, style: …})` + field init.
    pub fn new(id: &str, opts: MathBlockOptions) -> Self {
        let visual = VisualBlock::new(
            id,
            VisualBlockOptions {
                kind: Some("math-block".to_string()),
                ports: Some(VisualBlockPortSpec {
                    inputs: vec![VisualPortInput::Opts(VisualPortOptions {
                        id: MATH_IN.to_string(),
                        kind: Some("MathSignal".to_string()),
                        ..Default::default()
                    })],
                    outputs: vec![VisualPortInput::Opts(VisualPortOptions {
                        id: MATH_OUT.to_string(),
                        kind: Some("MathSignal".to_string()),
                        ..Default::default()
                    })],
                }),
                style: Some(VisualBlockStyle {
                    fill: Some("#f8fafc".to_string()),
                    stroke: Some("#2563eb".to_string()),
                    text: Some("#0f172a".to_string()),
                }),
                ..Default::default()
            },
        );
        MathBlockCore {
            id: id.to_string(),
            visual,
            current_tick: 0,
            latest_by_channel: HashMap::new(),
            output_history: Vec::new(),
            dt: opts.dt,
            ticks: opts.ticks,
            t0: opts.t0.unwrap_or(0.0),
        }
    }

    fn visual_ref(&self) -> &VisualBlock {
        &self.visual
    }

    fn station_core(&self) -> &StationCore {
        self.visual.core()
    }

    fn station_core_mut(&mut self) -> &mut StationCore {
        self.visual.core_mut()
    }

    /// Base preconditions (the body of `MathBlock.assertPreconditions`).
    fn assert_base_preconditions(&self) {
        require(Preconditions::positive("MathBlock", &format!("{}.dt", self.id), self.dt));
        require(Preconditions::integer_in_range(
            "MathBlock",
            &format!("{}.ticks", self.id),
            self.ticks as f64,
            1.0,
            1_000_000.0,
        ));
        require(Preconditions::finite("MathBlock", &format!("{}.t0", self.id), self.t0));
    }

    /// Connect this block's `src` output channel to `target`'s `tgt` channel.
    fn pipe(&mut self, target: StationRef, src: &str, tgt: &str) {
        self.station_core_mut().pipe(target, src, tgt);
    }

    /// `drainMath(channel)`.
    fn drain_math(&mut self, channel: &str) -> Vec<MathSignal> {
        let id = self.id.clone();
        let tokens = self.station_core_mut().drain::<MathSignal>(channel);
        let mut signals = Vec::with_capacity(tokens.len());
        for token in tokens {
            require(Preconditions::finite(&id, &format!("signal.{channel}.value"), token.value));
            require(Preconditions::finite(&id, &format!("signal.{channel}.time"), token.time));
            self.latest_by_channel.insert(channel.to_string(), (*token).clone());
            signals.push((*token).clone());
        }
        signals
    }

    fn latest_input(&self, channel: &str) -> Option<&MathSignal> {
        self.latest_by_channel.get(channel)
    }

    /// `inputValue(channel, holdLast)`.
    fn input_value(&mut self, channel: &str, hold_last: bool) -> Option<f64> {
        let fresh = self.drain_math(channel);
        let signal = last_owned(&fresh).or_else(|| {
            if hold_last {
                self.latest_input(channel).cloned()
            } else {
                None
            }
        });
        signal.map(|s| s.value)
    }

    /// `inputValues(channels, holdLast)`.
    fn input_values(&mut self, channels: &[String], hold_last: bool) -> Option<Vec<f64>> {
        let mut values = Vec::with_capacity(channels.len());
        for channel in channels {
            let value = self.input_value(channel, hold_last)?;
            values.push(value);
        }
        Some(values)
    }

    /// `emitValue(value, tick, time, channel)`.
    fn emit_value(&mut self, value: f64, tick: usize, time: f64, channel: &str) {
        let id = self.id.clone();
        require(Preconditions::finite(&id, &format!("output.{channel}"), value));
        let signal = MathSignal {
            kind: "math-signal",
            source_id: id.clone(),
            channel: channel.to_string(),
            tick,
            time,
            value,
        };
        self.output_history.push(MathSample {
            block_id: id,
            channel: channel.to_string(),
            tick,
            time,
            value,
        });
        let token: AnyToken = Rc::new(signal);
        self.station_core_mut().emit(token, channel);
    }
}

// =============================================================================
// Traits: MathBlock (step/precondition contract) + MathBlockObj (type-erased
// reads needed by the diagram runner).
// =============================================================================

/// The math-block contract (the abstract template-method hook + shared loop).
pub trait MathBlock {
    fn m(&self) -> &MathBlockCore;
    fn m_mut(&mut self) -> &mut MathBlockCore;

    /// One integration tick (the TS abstract `step`).
    fn step(&mut self, tick: usize, time: f64, dt: f64);

    /// Block-specific preconditions. Default = base checks only; concrete blocks
    /// override and call [`MathBlockCore::assert_base_preconditions`] first.
    fn assert_block_preconditions(&self) {
        self.m().assert_base_preconditions();
    }

    /// `runTimeStep` template method.
    fn run_step(&mut self) {
        if self.m().current_tick >= self.m().ticks {
            return;
        }
        let tick = self.m().current_tick;
        let dt = self.m().dt;
        let time = self.m().t0 + tick as f64 * dt;
        self.step(tick, time, dt);
        self.m_mut().current_tick += 1;
    }
}

/// Type-erased reads the diagram runner needs from a heterogeneous block list.
pub trait MathBlockObj: DESStation {
    fn output_history(&self) -> &[MathSample];
    fn ticks(&self) -> usize;
    fn visual(&self) -> &VisualBlock;
}

/// Implement [`DESStation`] + [`MathBlockObj`] for a concrete math block,
/// delegating to its [`MathBlockCore`] and [`MathBlock`] impl.
macro_rules! math_block_station {
    ($t:ty) => {
        impl MathBlockObj for $t {
            fn output_history(&self) -> &[MathSample] {
                &self.m().output_history
            }
            fn ticks(&self) -> usize {
                self.m().ticks
            }
            fn visual(&self) -> &VisualBlock {
                self.m().visual_ref()
            }
        }

        impl DESStation for $t {
            fn core(&self) -> &StationCore {
                self.m().station_core()
            }
            fn core_mut(&mut self) -> &mut StationCore {
                self.m_mut().station_core_mut()
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn run_time_step(&mut self) {
                MathBlock::run_step(self);
            }
            fn has_work(&self) -> bool {
                self.m().current_tick < self.m().ticks
            }
            fn assert_preconditions(&mut self) {
                MathBlock::assert_block_preconditions(self);
            }
        }
    };
}

// =============================================================================
// Concrete blocks
// =============================================================================

/// `class ConstantSourceBlock extends MathBlock`.
pub struct ConstantSourceBlock {
    m: MathBlockCore,
    pub value: f64,
    pub output_channel: String,
}

impl ConstantSourceBlock {
    pub fn new(id: &str, value: f64, opts: MathBlockOptions, output_channel: &str) -> Self {
        ConstantSourceBlock { m: MathBlockCore::new(id, opts), value, output_channel: output_channel.to_string() }
    }
}

impl MathBlock for ConstantSourceBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::finite("ConstantSourceBlock", &format!("{}.value", self.m.id), self.value));
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let value = self.value;
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(ConstantSourceBlock);

/// `class FunctionSourceBlock extends MathBlock`. Wraps a `(t, tick) → value`.
pub struct FunctionSourceBlock {
    m: MathBlockCore,
    pub f: Box<dyn Fn(f64, usize) -> f64>,
    pub output_channel: String,
}

impl FunctionSourceBlock {
    pub fn new(
        id: &str,
        f: Box<dyn Fn(f64, usize) -> f64>,
        opts: MathBlockOptions,
        output_channel: &str,
    ) -> Self {
        FunctionSourceBlock { m: MathBlockCore::new(id, opts), f, output_channel: output_channel.to_string() }
    }
}

impl MathBlock for FunctionSourceBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let value = (self.f)(time, tick);
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(FunctionSourceBlock);

/// `class ExpressionSourceBlock extends MathBlock`.
pub struct ExpressionSourceBlock {
    m: MathBlockCore,
    pub expression: String,
    ast: Expr,
    constants: HashMap<String, f64>,
    pub output_channel: String,
}

impl ExpressionSourceBlock {
    pub fn new(
        id: &str,
        expression: &str,
        opts: MathBlockOptions,
        constants: Option<&HashMap<String, f64>>,
        output_channel: &str,
    ) -> Self {
        let ast = parse(expression);
        let constants = finite_record("ExpressionSourceBlock", &format!("{id}.constants"), constants);
        ExpressionSourceBlock {
            m: MathBlockCore::new(id, opts),
            expression: expression.to_string(),
            ast,
            constants,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for ExpressionSourceBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let mut env: Env = self.constants.clone();
        env.insert("t".to_string(), time);
        env.insert("tick".to_string(), tick as f64);
        let value = evaluate(&self.ast, &env);
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(ExpressionSourceBlock);

/// `class SinkBlock extends MathBlock`.
pub struct SinkBlock {
    m: MathBlockCore,
    pub input_channels: Vec<String>,
    pub received: Vec<MathSignal>,
}

impl SinkBlock {
    pub fn new(id: &str, opts: MathBlockOptions, input_channels: Vec<String>) -> Self {
        SinkBlock { m: MathBlockCore::new(id, opts), input_channels, received: Vec::new() }
    }

    /// `series(sourceId?)`.
    pub fn series(&self, source_id: Option<&str>) -> Vec<MathSample> {
        self.received
            .iter()
            .filter(|s| source_id.is_none() || source_id == Some(s.source_id.as_str()))
            .map(|s| MathSample {
                block_id: s.source_id.clone(),
                channel: s.channel.clone(),
                tick: s.tick,
                time: s.time,
                value: s.value,
            })
            .collect()
    }
}

impl MathBlock for SinkBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::non_empty(
            "SinkBlock",
            &format!("{}.inputChannels", self.m.id),
            &self.input_channels,
        ));
    }
    fn step(&mut self, _tick: usize, _time: f64, _dt: f64) {
        let channels = self.input_channels.clone();
        for channel in &channels {
            let drained = self.m.drain_math(channel);
            self.received.extend(drained);
        }
    }
}
math_block_station!(SinkBlock);

/// `class SumBlock extends MathBlock`.
pub struct SumBlock {
    m: MathBlockCore,
    pub input_channels: Vec<String>,
    weights: Vec<f64>,
    pub hold_last: bool,
    pub output_channel: String,
}

impl SumBlock {
    pub fn new(
        id: &str,
        input_channels: Vec<String>,
        opts: MathBlockOptions,
        weights: Option<Vec<f64>>,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        let weights = weights.unwrap_or_else(|| input_channels.iter().map(|_| 1.0).collect());
        SumBlock {
            m: MathBlockCore::new(id, opts),
            input_channels,
            weights,
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for SumBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::non_empty("SumBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels));
        require(Preconditions::length_eq("SumBlock", &format!("{}.weights", self.m.id), &self.weights, self.input_channels.len()));
        assert_unique("SumBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels);
        require(Preconditions::all_finite("SumBlock", &format!("{}.weights", self.m.id), &self.weights));
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let channels = self.input_channels.clone();
        let hold = self.hold_last;
        let Some(xs) = self.m.input_values(&channels, hold) else {
            return;
        };
        let mut y = 0.0;
        for (i, x) in xs.iter().enumerate() {
            y += self.weights[i] * x;
        }
        let oc = self.output_channel.clone();
        self.m.emit_value(y, tick, time, &oc);
    }
}
math_block_station!(SumBlock);

/// `class SubtractBlock extends SumBlock` — composition wrapper (PORT NOTE:
/// Rust has no class inheritance; this delegates to a configured [`SumBlock`]).
pub struct SubtractBlock {
    inner: SumBlock,
}

impl SubtractBlock {
    pub fn new(
        id: &str,
        positive_input: &str,
        negative_input: &str,
        opts: MathBlockOptions,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        SubtractBlock {
            inner: SumBlock::new(
                id,
                vec![positive_input.to_string(), negative_input.to_string()],
                opts,
                Some(vec![1.0, -1.0]),
                hold_last,
                output_channel,
            ),
        }
    }
}

impl MathBlock for SubtractBlock {
    fn m(&self) -> &MathBlockCore {
        self.inner.m()
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        self.inner.m_mut()
    }
    fn assert_block_preconditions(&self) {
        self.inner.assert_block_preconditions();
    }
    fn step(&mut self, tick: usize, time: f64, dt: f64) {
        self.inner.step(tick, time, dt);
    }
}
math_block_station!(SubtractBlock);

/// `class ProductBlock extends MathBlock`.
pub struct ProductBlock {
    m: MathBlockCore,
    pub input_channels: Vec<String>,
    pub hold_last: bool,
    pub output_channel: String,
}

impl ProductBlock {
    pub fn new(id: &str, input_channels: Vec<String>, opts: MathBlockOptions, hold_last: bool, output_channel: &str) -> Self {
        ProductBlock {
            m: MathBlockCore::new(id, opts),
            input_channels,
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for ProductBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::non_empty("ProductBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels));
        assert_unique("ProductBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels);
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let channels = self.input_channels.clone();
        let hold = self.hold_last;
        let Some(xs) = self.m.input_values(&channels, hold) else {
            return;
        };
        let y = xs.iter().product::<f64>();
        let oc = self.output_channel.clone();
        self.m.emit_value(y, tick, time, &oc);
    }
}
math_block_station!(ProductBlock);

/// `class GainBlock extends MathBlock`.
pub struct GainBlock {
    m: MathBlockCore,
    pub gain: f64,
    pub input_channel: String,
    pub hold_last: bool,
    pub output_channel: String,
}

impl GainBlock {
    pub fn new(id: &str, gain: f64, opts: MathBlockOptions, input_channel: &str, hold_last: bool, output_channel: &str) -> Self {
        GainBlock {
            m: MathBlockCore::new(id, opts),
            gain,
            input_channel: input_channel.to_string(),
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for GainBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::finite("GainBlock", &format!("{}.gain", self.m.id), self.gain));
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let ch = self.input_channel.clone();
        let hold = self.hold_last;
        let Some(x) = self.m.input_value(&ch, hold) else {
            return;
        };
        let value = self.gain * x;
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(GainBlock);

/// `class SaturationBlock extends MathBlock`.
pub struct SaturationBlock {
    m: MathBlockCore,
    pub min: f64,
    pub max: f64,
    pub input_channel: String,
    pub hold_last: bool,
    pub output_channel: String,
}

impl SaturationBlock {
    pub fn new(
        id: &str,
        min: f64,
        max: f64,
        opts: MathBlockOptions,
        input_channel: &str,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        SaturationBlock {
            m: MathBlockCore::new(id, opts),
            min,
            max,
            input_channel: input_channel.to_string(),
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for SaturationBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::finite("SaturationBlock", &format!("{}.min", self.m.id), self.min));
        require(Preconditions::finite("SaturationBlock", &format!("{}.max", self.m.id), self.max));
        require(Preconditions::check(
            "SaturationBlock",
            &format!("{}.bounds", self.m.id),
            "satisfy min <= max",
            self.min <= self.max,
            Some(format!("{{min: {}, max: {}}}", self.min, self.max)),
        ));
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let ch = self.input_channel.clone();
        let hold = self.hold_last;
        let Some(x) = self.m.input_value(&ch, hold) else {
            return;
        };
        let value = self.min.max(self.max.min(x));
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(SaturationBlock);

/// `class IntegratorBlock extends MathBlock`.
pub struct IntegratorBlock {
    m: MathBlockCore,
    state: f64,
    state_tick: usize,
    last_input: Option<MathSignal>,
    consumed_through_tick: i64,
    pub method: IntegratorMethod,
    pub input_channel: String,
    pub hold_last: bool,
    pub output_channel: String,
}

impl IntegratorBlock {
    pub fn new(
        id: &str,
        initial_state: f64,
        opts: MathBlockOptions,
        method: IntegratorMethod,
        input_channel: &str,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        IntegratorBlock {
            m: MathBlockCore::new(id, opts),
            state: initial_state,
            state_tick: 0,
            last_input: None,
            // TS initialised `consumedThroughTick = -Infinity`.
            consumed_through_tick: i64::MIN,
            method,
            input_channel: input_channel.to_string(),
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }

    pub fn current_state(&self) -> f64 {
        self.state
    }

    fn advance_toward(&mut self, target_tick: usize, incoming: &[MathSignal], dt: f64) {
        while self.state_tick < target_tick {
            let fresh = latest_unused_at_or_before(incoming, self.state_tick, self.consumed_through_tick);
            let sig = fresh.clone().or_else(|| if self.hold_last { self.last_input.clone() } else { None });
            let Some(sig) = sig else {
                return;
            };
            let slope = if self.method == IntegratorMethod::Trapezoid {
                match &self.last_input {
                    Some(li) => 0.5 * (li.value + sig.value),
                    None => sig.value,
                }
            } else {
                sig.value
            };
            self.state += dt * slope;
            require(Preconditions::finite("IntegratorBlock", &format!("{}.state", self.m.id), self.state));
            self.last_input = Some(sig);
            if let Some(f) = &fresh {
                self.consumed_through_tick = f.tick as i64;
            }
            self.state_tick += 1;
        }
    }
}

impl MathBlock for IntegratorBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::finite("IntegratorBlock", &format!("{}.initialState", self.m.id), self.state));
        require(Preconditions::check(
            "IntegratorBlock",
            &format!("{}.method", self.m.id),
            "be euler or trapezoid",
            matches!(self.method, IntegratorMethod::Euler | IntegratorMethod::Trapezoid),
            Some(self.method.as_str().to_string()),
        ));
    }
    fn step(&mut self, tick: usize, time: f64, dt: f64) {
        let ch = self.input_channel.clone();
        let incoming = self.m.drain_math(&ch);
        if self.state_tick < tick {
            self.advance_toward(tick, &incoming, dt);
        }
        let state = self.state;
        let oc = self.output_channel.clone();
        self.m.emit_value(state, tick, time, &oc);
        if self.state_tick == tick && tick < self.m.ticks - 1 {
            self.advance_toward(tick + 1, &incoming, dt);
        }
    }
}
math_block_station!(IntegratorBlock);

/// `class DerivativeBlock extends MathBlock`.
pub struct DerivativeBlock {
    m: MathBlockCore,
    previous: Option<MathSignal>,
    pub input_channel: String,
    pub hold_last: bool,
    pub initial_output: f64,
    pub output_channel: String,
}

impl DerivativeBlock {
    pub fn new(
        id: &str,
        opts: MathBlockOptions,
        input_channel: &str,
        hold_last: bool,
        initial_output: f64,
        output_channel: &str,
    ) -> Self {
        DerivativeBlock {
            m: MathBlockCore::new(id, opts),
            previous: None,
            input_channel: input_channel.to_string(),
            hold_last,
            initial_output,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for DerivativeBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn step(&mut self, tick: usize, time: f64, dt: f64) {
        let ch = self.input_channel.clone();
        let fresh = self.m.drain_math(&ch);
        let sig = last_owned(&fresh).or_else(|| {
            if self.hold_last {
                self.m.latest_input(&ch).cloned()
            } else {
                None
            }
        });
        let Some(sig) = sig else {
            return;
        };
        if self.previous.is_none() {
            self.previous = Some(sig);
            let oc = self.output_channel.clone();
            let io = self.initial_output;
            self.m.emit_value(io, tick, time, &oc);
            return;
        }
        let prev = self.previous.clone().unwrap();
        let denom = if (sig.time - prev.time).abs() > 1e-12 { sig.time - prev.time } else { dt };
        require(Preconditions::not_div_by_zero("DerivativeBlock", &format!("{}.dt", self.m.id), denom, 1e-12));
        let value = (sig.value - prev.value) / denom;
        self.previous = Some(sig);
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(DerivativeBlock);

/// `class FirstOrderFilterBlock extends MathBlock`.
pub struct FirstOrderFilterBlock {
    m: MathBlockCore,
    pub tau: f64,
    y: f64,
    pub input_channel: String,
    pub hold_last: bool,
    pub output_channel: String,
}

impl FirstOrderFilterBlock {
    pub fn new(
        id: &str,
        tau: f64,
        initial: f64,
        opts: MathBlockOptions,
        input_channel: &str,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        FirstOrderFilterBlock {
            m: MathBlockCore::new(id, opts),
            tau,
            y: initial,
            input_channel: input_channel.to_string(),
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for FirstOrderFilterBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::positive("FirstOrderFilterBlock", &format!("{}.tau", self.m.id), self.tau));
        require(Preconditions::finite("FirstOrderFilterBlock", &format!("{}.initial", self.m.id), self.y));
    }
    fn step(&mut self, tick: usize, time: f64, dt: f64) {
        let ch = self.input_channel.clone();
        let hold = self.hold_last;
        if let Some(x) = self.m.input_value(&ch, hold) {
            let alpha = dt / (self.tau + dt);
            self.y += alpha * (x - self.y);
        }
        let y = self.y;
        let oc = self.output_channel.clone();
        self.m.emit_value(y, tick, time, &oc);
    }
}
math_block_station!(FirstOrderFilterBlock);

/// `class ComparatorBlock extends MathBlock`.
pub struct ComparatorBlock {
    m: MathBlockCore,
    pub op: ComparatorOp,
    pub left_channel: String,
    pub right_channel: Option<String>,
    pub threshold: Option<f64>,
    pub tolerance: f64,
    pub hold_last: bool,
    pub output_channel: String,
}

impl ComparatorBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: &str,
        op: ComparatorOp,
        opts: MathBlockOptions,
        left_channel: &str,
        right_channel: Option<&str>,
        threshold: Option<f64>,
        tolerance: f64,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        ComparatorBlock {
            m: MathBlockCore::new(id, opts),
            op,
            left_channel: left_channel.to_string(),
            right_channel: right_channel.map(|s| s.to_string()),
            threshold,
            tolerance,
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }

    fn compare(&self, a: f64, b: f64) -> bool {
        match self.op {
            ComparatorOp::Gt => a > b,
            ComparatorOp::Gte => a >= b,
            ComparatorOp::Lt => a < b,
            ComparatorOp::Lte => a <= b,
            ComparatorOp::Eq => (a - b).abs() <= self.tolerance,
            ComparatorOp::Neq => (a - b).abs() > self.tolerance,
        }
    }
}

impl MathBlock for ComparatorBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::check(
            "ComparatorBlock",
            &format!("{}.op", self.m.id),
            "be a supported comparison",
            true,
            None,
        ));
        require(Preconditions::non_negative("ComparatorBlock", &format!("{}.tolerance", self.m.id), self.tolerance));
        if self.right_channel.is_none() {
            require(Preconditions::finite(
                "ComparatorBlock",
                &format!("{}.threshold", self.m.id),
                self.threshold.unwrap_or(f64::NAN),
            ));
        }
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let left_ch = self.left_channel.clone();
        let hold = self.hold_last;
        let Some(left) = self.m.input_value(&left_ch, hold) else {
            return;
        };
        let right = match self.right_channel.clone() {
            None => self.threshold,
            Some(ch) => self.m.input_value(&ch, hold),
        };
        let Some(right) = right else {
            return;
        };
        let value = if self.compare(left, right) { 1.0 } else { 0.0 };
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(ComparatorBlock);

/// `class LogicBlock extends MathBlock`.
pub struct LogicBlock {
    m: MathBlockCore,
    pub op: LogicOp,
    pub input_channels: Vec<String>,
    pub threshold: f64,
    pub hold_last: bool,
    pub output_channel: String,
}

impl LogicBlock {
    pub fn new(
        id: &str,
        op: LogicOp,
        input_channels: Vec<String>,
        opts: MathBlockOptions,
        threshold: f64,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        LogicBlock {
            m: MathBlockCore::new(id, opts),
            op,
            input_channels,
            threshold,
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for LogicBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::check(
            "LogicBlock",
            &format!("{}.op", self.m.id),
            "be and, or, not, or xor",
            true,
            None,
        ));
        require(Preconditions::non_empty("LogicBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels));
        if self.op == LogicOp::Not {
            require(Preconditions::length_eq("LogicBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels, 1));
        }
        assert_unique("LogicBlock", &format!("{}.inputChannels", self.m.id), &self.input_channels);
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let channels = self.input_channels.clone();
        let hold = self.hold_last;
        let Some(xs) = self.m.input_values(&channels, hold) else {
            return;
        };
        let bits: Vec<bool> = xs.iter().map(|x| *x > self.threshold).collect();
        let result = match self.op {
            LogicOp::And => bits.iter().all(|b| *b),
            LogicOp::Or => bits.iter().any(|b| *b),
            LogicOp::Not => !bits[0],
            LogicOp::Xor => bits.iter().filter(|b| **b).count() % 2 == 1,
        };
        let oc = self.output_channel.clone();
        self.m.emit_value(if result { 1.0 } else { 0.0 }, tick, time, &oc);
    }
}
math_block_station!(LogicBlock);

/// `class ExpressionBlock extends MathBlock`.
pub struct ExpressionBlock {
    m: MathBlockCore,
    pub expression: String,
    /// Ordered `(variable_name, channel)` pairs (TS `Record<string,string>`,
    /// iterated via `Object.entries`; a `Vec` preserves insertion order).
    pub variable_channels: Vec<(String, String)>,
    ast: Expr,
    constants: HashMap<String, f64>,
    pub hold_last: bool,
    pub output_channel: String,
}

impl ExpressionBlock {
    pub fn new(
        id: &str,
        expression: &str,
        variable_channels: Vec<(String, String)>,
        opts: MathBlockOptions,
        constants: Option<&HashMap<String, f64>>,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        let ast = parse(expression);
        let constants = finite_record("ExpressionBlock", &format!("{id}.constants"), constants);
        ExpressionBlock {
            m: MathBlockCore::new(id, opts),
            expression: expression.to_string(),
            variable_channels,
            ast,
            constants,
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for ExpressionBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        let names: Vec<String> = self.variable_channels.iter().map(|(n, _)| n.clone()).collect();
        require(Preconditions::non_empty("ExpressionBlock", &format!("{}.variables", self.m.id), &names));
        for name in &names {
            assert_name("ExpressionBlock", &format!("{}.variable.{name}", self.m.id), name);
        }
        assert_unique("ExpressionBlock", &format!("{}.variables", self.m.id), &names);
        for key in self.constants.keys() {
            require(Preconditions::check(
                "ExpressionBlock",
                &format!("{}.constants", self.m.id),
                "not use reserved names t or tick",
                key != "t" && key != "tick",
                Some(key.clone()),
            ));
        }
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let entries = self.variable_channels.clone();
        let channels: Vec<String> = entries.iter().map(|(_, channel)| channel.clone()).collect();
        let hold = self.hold_last;
        let Some(values) = self.m.input_values(&channels, hold) else {
            return;
        };
        let mut env: Env = self.constants.clone();
        env.insert("t".to_string(), time);
        env.insert("tick".to_string(), tick as f64);
        for (i, (name, _)) in entries.iter().enumerate() {
            env.insert(name.clone(), values[i]);
        }
        let value = evaluate(&self.ast, &env);
        let oc = self.output_channel.clone();
        // PORT NOTE: TS attaches `{expression}` metadata to the emitted signal;
        // no consumer reads it, so it is dropped here.
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(ExpressionBlock);

/// `class Laplacian1DBlock extends MathBlock`.
pub struct Laplacian1DBlock {
    m: MathBlockCore,
    pub coefficient: f64,
    pub left_channel: String,
    pub center_channel: String,
    pub right_channel: String,
    pub hold_last: bool,
    pub output_channel: String,
}

impl Laplacian1DBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: &str,
        coefficient: f64,
        opts: MathBlockOptions,
        left_channel: &str,
        center_channel: &str,
        right_channel: &str,
        hold_last: bool,
        output_channel: &str,
    ) -> Self {
        Laplacian1DBlock {
            m: MathBlockCore::new(id, opts),
            coefficient,
            left_channel: left_channel.to_string(),
            center_channel: center_channel.to_string(),
            right_channel: right_channel.to_string(),
            hold_last,
            output_channel: output_channel.to_string(),
        }
    }
}

impl MathBlock for Laplacian1DBlock {
    fn m(&self) -> &MathBlockCore {
        &self.m
    }
    fn m_mut(&mut self) -> &mut MathBlockCore {
        &mut self.m
    }
    fn assert_block_preconditions(&self) {
        self.m.assert_base_preconditions();
        require(Preconditions::finite("Laplacian1DBlock", &format!("{}.coefficient", self.m.id), self.coefficient));
    }
    fn step(&mut self, tick: usize, time: f64, _dt: f64) {
        let lc = self.left_channel.clone();
        let cc = self.center_channel.clone();
        let rc = self.right_channel.clone();
        let hold = self.hold_last;
        let left = self.m.input_value(&lc, hold);
        let center = self.m.input_value(&cc, hold);
        let right = self.m.input_value(&rc, hold);
        let (Some(left), Some(center), Some(right)) = (left, center, right) else {
            return;
        };
        let value = self.coefficient * (left - 2.0 * center + right);
        let oc = self.output_channel.clone();
        self.m.emit_value(value, tick, time, &oc);
    }
}
math_block_station!(Laplacian1DBlock);

// =============================================================================
// Diagram runner
// =============================================================================

/// Bundles the two shared handles a block needs in [`run_math_block_diagram`]:
/// a [`StationRef`] for the iterative runner and a `Rc<RefCell<dyn MathBlockObj>>`
/// for reading `output_history` and visual specs. Both are built from the same
/// concrete `Rc` via *sized* coercions (no trait upcasting).
pub struct MathBlockHandle {
    pub station: StationRef,
    pub reader: Rc<RefCell<dyn MathBlockObj>>,
}

impl MathBlockHandle {
    pub fn new<B: MathBlockObj + 'static>(b: Rc<RefCell<B>>) -> Self {
        let station: StationRef = b.clone();
        let reader: Rc<RefCell<dyn MathBlockObj>> = b;
        MathBlockHandle { station, reader }
    }
}

/// Options for [`run_math_block_diagram`] (`{maxTicks?, logger?}`).
#[derive(Default)]
pub struct RunDiagramOptions<'a> {
    pub max_ticks: Option<usize>,
    pub logger: Option<&'a dyn BlockModelLogger>,
}

/// `runMathBlockDiagram(blocks, opts)`.
pub fn run_math_block_diagram(blocks: Vec<MathBlockHandle>, opts: RunDiagramOptions) -> MathBlockRunResult {
    require(Preconditions::non_empty("runMathBlockDiagram", "blocks", &blocks));
    let ids: Vec<String> = blocks.iter().map(|h| h.reader.borrow().id().to_string()).collect();
    assert_unique("runMathBlockDiagram", "block ids", &ids);

    let max_block_ticks = blocks.iter().map(|h| h.reader.borrow().ticks()).max().unwrap_or(0);
    let max_ticks = opts.max_ticks.unwrap_or(max_block_ticks + 1);

    if let Some(l) = opts.logger {
        l.log(LogEvent {
            kind: "math-block-run-start".to_string(),
            level: Some("info".to_string()),
            fields: vec![
                ("blocks".to_string(), blocks.len().to_string()),
                ("maxTicks".to_string(), max_ticks.to_string()),
            ],
        });
    }

    let stations: Vec<StationRef> = blocks.iter().map(|h| h.station.clone()).collect();
    let mut summary = run_iterative_des(
        stations,
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), ..Default::default() },
    );

    let mut outputs: Vec<MathSample> = Vec::new();
    for h in &blocks {
        outputs.extend(h.reader.borrow().output_history().iter().cloned());
    }

    if let Some(l) = opts.logger {
        l.log(LogEvent {
            kind: "math-block-run-finish".to_string(),
            level: Some("info".to_string()),
            fields: vec![
                ("ticks".to_string(), summary.ticks.to_string()),
                ("reason".to_string(), summary.reason.map(|r| r.as_str().to_string()).unwrap_or_default()),
                ("outputs".to_string(), outputs.len().to_string()),
            ],
        });
    }

    // PORT NOTE: per-block finite-output validators are computed here (the TS
    // base class registers them on each station; see module docs).
    let mut checks: Vec<ValidationCheck> = Vec::new();
    for h in &blocks {
        let b = h.reader.borrow();
        let passed = b.output_history().iter().all(|x| x.value.is_finite() && x.time.is_finite());
        checks.push(ValidationCheck {
            name: format!("math-block-finite-output/{}", b.id()),
            passed,
            observed: None,
            expected: Some("all emitted math samples are finite".to_string()),
            group: Some("math-blocks".to_string()),
            details: None,
        });
    }
    if !checks.is_empty() {
        let all_ok = checks.iter().all(|c| c.passed);
        summary.validation = Some(checks.clone());
        summary.validation_ok = Some(all_ok);
    }

    // Visual specs: hold the borrows alive in `guards` so the `&VisualBlock`
    // references passed to `visual_block_specs` stay valid.
    let guards: Vec<Ref<dyn MathBlockObj>> = blocks.iter().map(|h| h.reader.borrow()).collect();
    let refs: Vec<&VisualBlock> = guards.iter().map(|g| g.visual()).collect();
    let visual_blocks = visual_block_specs(&refs);

    let validation = summary.validation.clone().unwrap_or_default();
    MathBlockRunResult { summary, validation, outputs, visual_blocks }
}

// =============================================================================
// ODE block system
// =============================================================================

/// `interface ODEStateSpec`.
#[derive(Clone, Debug)]
pub struct ODEStateSpec {
    pub name: String,
    pub initial: f64,
    pub derivative: String,
}

/// `interface ODEBlockSystemParams`.
#[derive(Clone, Debug)]
pub struct ODEBlockSystemParams {
    pub states: Vec<ODEStateSpec>,
    pub t0: Option<f64>,
    pub t1: f64,
    pub dt: f64,
    pub method: Option<IntegratorMethod>,
    pub constants: Option<HashMap<String, f64>>,
}

/// `interface ODETraceRow`.
#[derive(Clone, Debug)]
pub struct ODETraceRow {
    pub tick: usize,
    pub time: f64,
    pub state: HashMap<String, f64>,
    pub derivatives: HashMap<String, f64>,
}

/// `inputs?: Record<string,string> | string[]` on a [`BlockGraphNode`].
#[derive(Clone, Debug)]
pub enum BlockGraphInputs {
    List(Vec<String>),
    Map(Vec<(String, String)>),
}

/// `interface BlockGraphNode`.
#[derive(Clone, Debug)]
pub struct BlockGraphNode {
    pub id: String,
    pub kind: String,
    pub expression: Option<String>,
    pub inputs: Option<BlockGraphInputs>,
    pub output: Option<String>,
}

/// `interface BlockGraphEdge` (`signal` is always `"MathSignal"`).
#[derive(Clone, Debug)]
pub struct BlockGraphEdge {
    pub from: String,
    pub to: String,
    pub from_channel: String,
    pub to_channel: String,
    pub signal: String,
}

/// `interface ODEBlockSystemResult`.
#[derive(Debug)]
pub struct ODEBlockSystemResult {
    pub params: ODEBlockSystemParams,
    pub steps: usize,
    pub trace: Vec<ODETraceRow>,
    pub final_state: HashMap<String, f64>,
    pub block_graph: Vec<BlockGraphNode>,
    pub block_graph_edges: Vec<BlockGraphEdge>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub validation: Vec<ValidationCheck>,
    pub run_summary: IterativeRunSummary,
}

/// `runODEBlockSystem(params, logger?)`.
pub fn run_ode_block_system(
    params: ODEBlockSystemParams,
    logger: Option<&dyn BlockModelLogger>,
) -> ODEBlockSystemResult {
    validate_ode_params(&params);
    let t0 = params.t0.unwrap_or(0.0);
    let steps = duration_steps("ODEBlockSystem", t0, params.t1, params.dt);
    let ticks = steps + 1;
    let opts = MathBlockOptions { dt: params.dt, ticks, t0: Some(t0) };
    let constants = finite_record("ODEBlockSystem", "constants", params.constants.as_ref());
    let method = params.method.unwrap_or(IntegratorMethod::Euler);
    let names: Vec<String> = params.states.iter().map(|s| s.name.clone()).collect();

    let mut block_graph: Vec<BlockGraphNode> = Vec::new();
    for s in &params.states {
        block_graph.push(BlockGraphNode {
            id: format!("integrator:{}", s.name),
            kind: "integrator".to_string(),
            expression: None,
            inputs: Some(BlockGraphInputs::List(vec![MATH_IN.to_string()])),
            output: Some(MATH_OUT.to_string()),
        });
    }
    for s in &params.states {
        block_graph.push(BlockGraphNode {
            id: format!("rhs:{}", s.name),
            kind: "expression".to_string(),
            expression: Some(s.derivative.clone()),
            inputs: Some(BlockGraphInputs::Map(names.iter().map(|n| (n.clone(), n.clone())).collect())),
            output: Some(MATH_OUT.to_string()),
        });
    }
    let mut block_graph_edges: Vec<BlockGraphEdge> = Vec::new();

    let integrators: Vec<Rc<RefCell<IntegratorBlock>>> = params
        .states
        .iter()
        .map(|s| {
            Rc::new(RefCell::new(IntegratorBlock::new(
                &format!("integrator:{}", s.name),
                s.initial,
                opts,
                method,
                MATH_IN,
                false,
                MATH_OUT,
            )))
        })
        .collect();
    let rhs_blocks: Vec<Rc<RefCell<ExpressionBlock>>> = params
        .states
        .iter()
        .map(|s| {
            let variable_channels: Vec<(String, String)> = names.iter().map(|n| (n.clone(), n.clone())).collect();
            Rc::new(RefCell::new(ExpressionBlock::new(
                &format!("rhs:{}", s.name),
                &s.derivative,
                variable_channels,
                opts,
                Some(&constants),
                true,
                MATH_OUT,
            )))
        })
        .collect();

    for (i, integ) in integrators.iter().enumerate() {
        let state_name = names[i].clone();
        for (j, rhs) in rhs_blocks.iter().enumerate() {
            let target: StationRef = Rc::clone(rhs);
            integ.borrow_mut().m_mut().pipe(target, MATH_OUT, &state_name);
            block_graph_edges.push(BlockGraphEdge {
                from: format!("integrator:{state_name}"),
                to: format!("rhs:{}", names[j]),
                from_channel: MATH_OUT.to_string(),
                to_channel: state_name.clone(),
                signal: "MathSignal".to_string(),
            });
        }
    }
    for i in 0..rhs_blocks.len() {
        let target: StationRef = Rc::clone(&integrators[i]);
        rhs_blocks[i].borrow_mut().m_mut().pipe(target, MATH_OUT, MATH_IN);
        block_graph_edges.push(BlockGraphEdge {
            from: format!("rhs:{}", names[i]),
            to: format!("integrator:{}", names[i]),
            from_channel: MATH_OUT.to_string(),
            to_channel: MATH_IN.to_string(),
            signal: "MathSignal".to_string(),
        });
    }

    if let Some(l) = logger {
        l.log(LogEvent {
            kind: "math-ode-start".to_string(),
            level: Some("info".to_string()),
            fields: vec![
                ("states".to_string(), names.join(",")),
                ("steps".to_string(), steps.to_string()),
                ("dt".to_string(), params.dt.to_string()),
            ],
        });
    }

    let mut handles: Vec<MathBlockHandle> = Vec::new();
    for integ in &integrators {
        handles.push(MathBlockHandle::new(Rc::clone(integ)));
    }
    for rhs in &rhs_blocks {
        handles.push(MathBlockHandle::new(Rc::clone(rhs)));
    }
    let run = run_math_block_diagram(handles, RunDiagramOptions { max_ticks: None, logger });

    let mut trace: Vec<ODETraceRow> = Vec::new();
    for tick in 0..ticks {
        let mut state: HashMap<String, f64> = HashMap::new();
        let mut derivatives: HashMap<String, f64> = HashMap::new();
        for i in 0..params.states.len() {
            let name = names[i].clone();
            let sv = integrators[i].borrow().output_history().get(tick).map(|s| s.value).unwrap_or(f64::NAN);
            let dv = rhs_blocks[i].borrow().output_history().get(tick).map(|s| s.value).unwrap_or(f64::NAN);
            state.insert(name.clone(), sv);
            derivatives.insert(name, dv);
        }
        let row = ODETraceRow { tick, time: t0 + tick as f64 * params.dt, state, derivatives };
        if let Some(l) = logger {
            l.log(LogEvent {
                kind: "math-ode-tick".to_string(),
                level: Some("debug".to_string()),
                fields: vec![("tick".to_string(), tick.to_string()), ("time".to_string(), row.time.to_string())],
            });
        }
        trace.push(row);
    }
    let final_state = trace.last().map(|r| r.state.clone()).unwrap_or_default();
    let mut validation = run.validation.clone();
    validation.extend(validate_ode_trace(&trace, ticks, params.t1));

    ODEBlockSystemResult {
        params,
        steps,
        trace,
        final_state,
        block_graph,
        block_graph_edges,
        visual_blocks: run.visual_blocks,
        validation,
        run_summary: run.summary,
    }
}

fn validate_ode_params(params: &ODEBlockSystemParams) {
    require(Preconditions::non_empty("ODEBlockSystem", "states", &params.states));
    require(Preconditions::integer_in_range("ODEBlockSystem", "states.length", params.states.len() as f64, 1.0, 100.0));
    let names: Vec<String> = params.states.iter().map(|s| s.name.clone()).collect();
    for name in &names {
        assert_name("ODEBlockSystem", "state.name", name);
    }
    assert_unique("ODEBlockSystem", "state.name", &names);
    for s in &params.states {
        require(Preconditions::finite("ODEBlockSystem", &format!("{}.initial", s.name), s.initial));
        require(Preconditions::check(
            "ODEBlockSystem",
            &format!("{}.derivative", s.name),
            "be non-empty",
            !s.derivative.trim().is_empty(),
            Some(s.derivative.clone()),
        ));
        parse(&s.derivative);
    }
    finite_record("ODEBlockSystem", "constants", params.constants.as_ref());
    // `method` is a typed enum here; the TS string-union guard is unnecessary.
}

fn validate_ode_trace(trace: &[ODETraceRow], ticks: usize, t1: f64) -> Vec<ValidationCheck> {
    let finite = trace.iter().all(|row| {
        row.time.is_finite()
            && row.state.values().all(|v| v.is_finite())
            && row.derivatives.values().all(|v| v.is_finite())
    });
    let last = trace.last().expect("ODE trace is non-empty");
    vec![
        ValidationCheck {
            name: "ode-trace-length".to_string(),
            passed: trace.len() == ticks,
            observed: Some(trace.len().to_string()),
            expected: Some(ticks.to_string()),
            group: Some("math-ode".to_string()),
            details: None,
        },
        ValidationCheck {
            name: "ode-trace-finite".to_string(),
            passed: finite,
            observed: None,
            expected: Some("all state and derivative values finite".to_string()),
            group: Some("math-ode".to_string()),
            details: None,
        },
        ValidationCheck {
            name: "ode-final-time".to_string(),
            passed: (last.time - t1).abs() <= 1e-9 * 1.0_f64.max(t1.abs()),
            observed: Some(format!("{:.12e}", last.time)),
            expected: Some(format!("{t1:.12e}")),
            group: Some("math-ode".to_string()),
            details: None,
        },
    ]
}

// =============================================================================
// 1-D heat block grid
// =============================================================================

/// `interface Heat1DBlockParams`.
#[derive(Clone, Debug)]
pub struct Heat1DBlockParams {
    pub cells: usize,
    pub length: f64,
    pub alpha: f64,
    pub t0: Option<f64>,
    pub t1: f64,
    pub dt: f64,
    pub initial_expression: Option<String>,
    pub initial_values: Option<Vec<f64>>,
    pub left_boundary: Option<f64>,
    pub right_boundary: Option<f64>,
    pub constants: Option<HashMap<String, f64>>,
}

/// `interface Heat1DTraceRow`.
#[derive(Clone, Debug)]
pub struct Heat1DTraceRow {
    pub tick: usize,
    pub time: f64,
    pub values: Vec<f64>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

/// `interface Heat1DBlockResult`.
#[derive(Debug)]
pub struct Heat1DBlockResult {
    pub params: Heat1DBlockParams,
    pub dx: f64,
    pub cfl: f64,
    pub steps: usize,
    pub x: Vec<f64>,
    pub trace: Vec<Heat1DTraceRow>,
    pub final_values: Vec<f64>,
    pub block_graph: Vec<BlockGraphNode>,
    pub block_graph_edges: Vec<BlockGraphEdge>,
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub validation: Vec<ValidationCheck>,
    pub run_summary: IterativeRunSummary,
}

/// `runHeat1DBlockGrid(params, logger?)`.
pub fn run_heat1d_block_grid(
    params: Heat1DBlockParams,
    logger: Option<&dyn BlockModelLogger>,
) -> Heat1DBlockResult {
    validate_heat_params(&params);
    let t0 = params.t0.unwrap_or(0.0);
    let steps = duration_steps("Heat1DBlockGrid", t0, params.t1, params.dt);
    let ticks = steps + 1;
    let dx = params.length / (params.cells as f64 - 1.0);
    let coefficient = params.alpha / (dx * dx);
    let cfl = coefficient * params.dt;
    require(Preconditions::check(
        "Heat1DBlockGrid",
        "alpha*dt/dx^2",
        "be <= 0.5 for explicit block-grid stability",
        cfl <= 0.5 + 1e-12,
        Some(cfl.to_string()),
    ));
    let x: Vec<f64> = (0..params.cells).map(|i| i as f64 * dx).collect();
    let mut initial = build_heat_initial_values(&params, &x);
    let left_boundary = params.left_boundary.unwrap_or(initial[0]);
    let right_boundary = params.right_boundary.unwrap_or(initial[initial.len() - 1]);
    initial[0] = left_boundary;
    let last_idx = initial.len() - 1;
    initial[last_idx] = right_boundary;

    let opts = MathBlockOptions { dt: params.dt, ticks, t0: Some(t0) };
    let mut block_graph_edges: Vec<BlockGraphEdge> = Vec::new();

    // Cells occupy `handles[0..params.cells]`.
    let mut handles: Vec<MathBlockHandle> = Vec::new();
    for i in 0..params.cells {
        if i == 0 {
            let b = Rc::new(RefCell::new(ConstantSourceBlock::new(&format!("cell:{i}"), left_boundary, opts, MATH_OUT)));
            handles.push(MathBlockHandle::new(b));
        } else if i == params.cells - 1 {
            let b = Rc::new(RefCell::new(ConstantSourceBlock::new(&format!("cell:{i}"), right_boundary, opts, MATH_OUT)));
            handles.push(MathBlockHandle::new(b));
        } else {
            let b = Rc::new(RefCell::new(IntegratorBlock::new(
                &format!("cell:{i}"),
                initial[i],
                opts,
                IntegratorMethod::Euler,
                MATH_IN,
                false,
                MATH_OUT,
            )));
            handles.push(MathBlockHandle::new(b));
        }
    }

    let mut lap_handles: Vec<MathBlockHandle> = Vec::new();
    for i in 1..params.cells - 1 {
        let lap = Rc::new(RefCell::new(Laplacian1DBlock::new(
            &format!("laplacian:{i}"),
            coefficient,
            opts,
            "left",
            "center",
            "right",
            true,
            MATH_OUT,
        )));
        let lap_station: StationRef = Rc::clone(&lap);
        handles[i - 1].reader.borrow_mut().core_mut().pipe(lap_station.clone(), MATH_OUT, "left");
        handles[i].reader.borrow_mut().core_mut().pipe(lap_station.clone(), MATH_OUT, "center");
        handles[i + 1].reader.borrow_mut().core_mut().pipe(lap_station.clone(), MATH_OUT, "right");
        lap.borrow_mut().m_mut().pipe(handles[i].station.clone(), MATH_OUT, MATH_IN);
        block_graph_edges.push(BlockGraphEdge { from: format!("cell:{}", i - 1), to: format!("laplacian:{i}"), from_channel: MATH_OUT.to_string(), to_channel: "left".to_string(), signal: "MathSignal".to_string() });
        block_graph_edges.push(BlockGraphEdge { from: format!("cell:{i}"), to: format!("laplacian:{i}"), from_channel: MATH_OUT.to_string(), to_channel: "center".to_string(), signal: "MathSignal".to_string() });
        block_graph_edges.push(BlockGraphEdge { from: format!("cell:{}", i + 1), to: format!("laplacian:{i}"), from_channel: MATH_OUT.to_string(), to_channel: "right".to_string(), signal: "MathSignal".to_string() });
        block_graph_edges.push(BlockGraphEdge { from: format!("laplacian:{i}"), to: format!("cell:{i}"), from_channel: MATH_OUT.to_string(), to_channel: MATH_IN.to_string(), signal: "MathSignal".to_string() });
        lap_handles.push(MathBlockHandle::new(lap));
    }

    if let Some(l) = logger {
        l.log(LogEvent {
            kind: "math-heat1d-start".to_string(),
            level: Some("info".to_string()),
            fields: vec![
                ("cells".to_string(), params.cells.to_string()),
                ("steps".to_string(), steps.to_string()),
                ("dx".to_string(), dx.to_string()),
                ("cfl".to_string(), cfl.to_string()),
            ],
        });
    }

    let cell_count = params.cells;
    handles.extend(lap_handles);
    let run = run_math_block_diagram(handles, RunDiagramOptions { max_ticks: None, logger });

    // Re-acquire cell readers from `run` is not possible (handles moved); instead
    // read cell output from the run's flattened outputs keyed by block id.
    let mut cell_history: Vec<Vec<f64>> = vec![Vec::new(); cell_count];
    for sample in &run.outputs {
        if let Some(rest) = sample.block_id.strip_prefix("cell:") {
            if let Ok(idx) = rest.parse::<usize>() {
                if idx < cell_count {
                    // outputs are appended in tick order per block.
                    cell_history[idx].push(sample.value);
                }
            }
        }
    }

    let mut trace: Vec<Heat1DTraceRow> = Vec::new();
    for tick in 0..ticks {
        let values: Vec<f64> = (0..cell_count)
            .map(|i| cell_history[i].get(tick).copied().unwrap_or(f64::NAN))
            .collect();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let row = Heat1DTraceRow { tick, time: t0 + tick as f64 * params.dt, values, min, max, mean };
        if let Some(l) = logger {
            l.log(LogEvent {
                kind: "math-heat1d-tick".to_string(),
                level: Some("debug".to_string()),
                fields: vec![
                    ("tick".to_string(), tick.to_string()),
                    ("time".to_string(), row.time.to_string()),
                    ("min".to_string(), min.to_string()),
                    ("max".to_string(), max.to_string()),
                    ("mean".to_string(), mean.to_string()),
                ],
            });
        }
        trace.push(row);
    }

    let mut block_graph: Vec<BlockGraphNode> = Vec::new();
    for i in 0..params.cells {
        block_graph.push(BlockGraphNode {
            id: format!("cell:{i}"),
            kind: if i == 0 || i == params.cells - 1 { "constant-boundary".to_string() } else { "integrator".to_string() },
            expression: None,
            inputs: None,
            output: Some(MATH_OUT.to_string()),
        });
    }
    for i in 1..params.cells - 1 {
        block_graph.push(BlockGraphNode {
            id: format!("laplacian:{i}"),
            kind: "laplacian-1d".to_string(),
            expression: None,
            inputs: Some(BlockGraphInputs::List(vec!["left".to_string(), "center".to_string(), "right".to_string()])),
            output: Some(MATH_OUT.to_string()),
        });
    }

    let final_values = trace.last().map(|r| r.values.clone()).unwrap_or_default();
    let mut validation = run.validation.clone();
    validation.extend(validate_heat_trace(&trace, ticks, &initial, left_boundary, right_boundary));

    Heat1DBlockResult {
        params,
        dx,
        cfl,
        steps,
        x,
        trace,
        final_values,
        block_graph,
        block_graph_edges,
        visual_blocks: run.visual_blocks,
        validation,
        run_summary: run.summary,
    }
}

fn validate_heat_params(params: &Heat1DBlockParams) {
    require(Preconditions::integer_in_range("Heat1DBlockGrid", "cells", params.cells as f64, 3.0, 1000.0));
    require(Preconditions::positive("Heat1DBlockGrid", "length", params.length));
    require(Preconditions::non_negative("Heat1DBlockGrid", "alpha", params.alpha));
    require(Preconditions::positive("Heat1DBlockGrid", "dt", params.dt));
    require(Preconditions::finite("Heat1DBlockGrid", "t1", params.t1));
    if let Some(t0) = params.t0 {
        require(Preconditions::finite("Heat1DBlockGrid", "t0", t0));
    }
    if let Some(iv) = &params.initial_values {
        require(Preconditions::length_eq("Heat1DBlockGrid", "initialValues", iv, params.cells));
        require(Preconditions::all_finite("Heat1DBlockGrid", "initialValues", iv));
    }
    if let Some(ie) = &params.initial_expression {
        parse(ie);
    }
    if let Some(lb) = params.left_boundary {
        require(Preconditions::finite("Heat1DBlockGrid", "leftBoundary", lb));
    }
    if let Some(rb) = params.right_boundary {
        require(Preconditions::finite("Heat1DBlockGrid", "rightBoundary", rb));
    }
    finite_record("Heat1DBlockGrid", "constants", params.constants.as_ref());
}

fn build_heat_initial_values(params: &Heat1DBlockParams, x: &[f64]) -> Vec<f64> {
    if let Some(iv) = &params.initial_values {
        return iv.clone();
    }
    let expression = params.initial_expression.clone().unwrap_or_else(|| "sin(pi*x/length)".to_string());
    let ast = parse(&expression);
    let mut constants: HashMap<String, f64> = HashMap::new();
    constants.insert("pi".to_string(), std::f64::consts::PI);
    constants.insert("e".to_string(), std::f64::consts::E);
    constants.insert("length".to_string(), params.length);
    if let Some(extra) = &params.constants {
        for (k, v) in extra {
            constants.insert(k.clone(), *v);
        }
    }
    x.iter()
        .map(|xi| {
            let mut env = constants.clone();
            env.insert("x".to_string(), *xi);
            let value = evaluate(&ast, &env);
            require(Preconditions::finite("Heat1DBlockGrid", "initialExpression", value));
            value
        })
        .collect()
}

fn validate_heat_trace(
    trace: &[Heat1DTraceRow],
    ticks: usize,
    initial: &[f64],
    left_boundary: f64,
    right_boundary: f64,
) -> Vec<ValidationCheck> {
    let finite = trace.iter().all(|row| row.time.is_finite() && row.values.iter().all(|v| v.is_finite()));
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in initial.iter().chain([left_boundary, right_boundary].iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    lo -= 1e-9;
    hi += 1e-9;
    let max_principle = trace.iter().all(|row| row.values.iter().all(|&v| v >= lo && v <= hi));
    let obs_lo = trace.iter().map(|r| r.min).fold(f64::INFINITY, f64::min);
    let obs_hi = trace.iter().map(|r| r.max).fold(f64::NEG_INFINITY, f64::max);
    vec![
        ValidationCheck {
            name: "heat-trace-length".to_string(),
            passed: trace.len() == ticks,
            observed: Some(trace.len().to_string()),
            expected: Some(ticks.to_string()),
            group: Some("math-heat1d".to_string()),
            details: None,
        },
        ValidationCheck {
            name: "heat-trace-finite".to_string(),
            passed: finite,
            observed: None,
            expected: Some("all grid values finite".to_string()),
            group: Some("math-heat1d".to_string()),
            details: None,
        },
        ValidationCheck {
            name: "heat-maximum-principle".to_string(),
            passed: max_principle,
            observed: Some(format!("[{obs_lo:.8e}, {obs_hi:.8e}]")),
            expected: Some(format!("[{lo:.8e}, {hi:.8e}]")),
            group: Some("math-heat1d".to_string()),
            details: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_source_emits_each_tick() {
        let opts = MathBlockOptions { dt: 0.1, ticks: 3, t0: Some(0.0) };
        let block = Rc::new(RefCell::new(ConstantSourceBlock::new("c", 2.5, opts, MATH_OUT)));
        let run = run_math_block_diagram(vec![MathBlockHandle::new(block)], RunDiagramOptions::default());
        assert_eq!(run.outputs.len(), 3);
        assert!(run.outputs.iter().all(|s| (s.value - 2.5).abs() < 1e-12));
        assert!(run.validation.iter().all(|c| c.passed));
    }

    #[test]
    fn integrator_of_constant_is_a_ramp() {
        // dx/dt = 1, x0 = 0, dt = 1, over 4 steps -> x(4) ≈ 4.
        let params = ODEBlockSystemParams {
            states: vec![ODEStateSpec { name: "x".to_string(), initial: 0.0, derivative: "1".to_string() }],
            t0: Some(0.0),
            t1: 4.0,
            dt: 1.0,
            method: Some(IntegratorMethod::Euler),
            constants: None,
        };
        let result = run_ode_block_system(params, None);
        assert_eq!(result.trace.len(), 5);
        let xf = *result.final_state.get("x").unwrap();
        assert!((xf - 4.0).abs() < 1e-9, "x(4) = {xf}");
        assert!(result.validation.iter().all(|c| c.passed), "validation: {:?}", result.validation);
    }

    #[test]
    fn exponential_decay_ode() {
        // dx/dt = -x, x0 = 1, small dt -> decays toward 0, stays in (0, 1].
        let params = ODEBlockSystemParams {
            states: vec![ODEStateSpec { name: "x".to_string(), initial: 1.0, derivative: "-x".to_string() }],
            t0: Some(0.0),
            t1: 1.0,
            dt: 0.05,
            method: Some(IntegratorMethod::Euler),
            constants: None,
        };
        let result = run_ode_block_system(params, None);
        let xf = *result.final_state.get("x").unwrap();
        assert!(xf > 0.0 && xf < 1.0, "x(1) = {xf}");
        assert!(result.validation.iter().all(|c| c.passed));
    }

    #[test]
    fn heat_grid_obeys_maximum_principle() {
        let params = Heat1DBlockParams {
            cells: 5,
            length: 1.0,
            alpha: 0.1,
            t0: Some(0.0),
            t1: 0.2,
            dt: 0.05,
            initial_expression: None,
            initial_values: None,
            left_boundary: Some(0.0),
            right_boundary: Some(0.0),
            constants: None,
        };
        let result = run_heat1d_block_grid(params, None);
        assert_eq!(result.trace.len(), result.steps + 1);
        assert!(result.cfl <= 0.5 + 1e-12);
        assert!(result.validation.iter().all(|c| c.passed), "validation: {:?}", result.validation);
    }
}
