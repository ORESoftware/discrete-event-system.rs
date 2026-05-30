//! Port of `src/des/general/des-base/transform-entity.ts`.
//!
//! A zero-backlog stationary entity for modeling ordinary program functions as
//! DES graph nodes. `take(token, channel)` immediately applies a transform and
//! emits any returned token(s) — useful for lightweight adapters, parsers,
//! validators, projections, feature extractors, and cost/reward functions that
//! should be visible in the station graph without adding artificial waiting
//! time.
//!
//! ## Rust shape (faithful translation of the TS template-method hierarchy)
//!
//! TypeScript layers three `abstract class`es plus one concrete `class` on top
//! of `DESStation`. Rust has no inheritance, so:
//!
//!   * The shared per-station fields (input channels, output-channel policy,
//!     validators, zero-backlog counters) live in [`TransformEntityCore`],
//!     which embeds a [`StationCore`] and is held as a field by every concrete
//!     entity.
//!   * [`TransformEntity`] is the base trait: accessors plus the shared
//!     precondition guard.
//!   * [`PureTransformEntity`] adds the required `transform` hook and a `take`
//!     that processes immediately (`has_work` is `false`).
//!   * [`MemoryTransformEntity`] adds the required `transform_queued` hook, a
//!     `take` that queues into the inbox, and `run_queued` that drains+processes
//!     on a tick (`has_work` reflects queued input).
//!   * [`FunctionEntity`] is the concrete pure station wrapping a boxed closure.
//!
//! ## Notable deviations (flagged)
//!
//!   * The TS `TransformContext` exposes a back-reference to the owning station
//!     plus an eager `emit` callback. To avoid an aliasing borrow of `&mut self`
//!     while the user transform runs, [`TransformContext`] instead *buffers*
//!     emissions; the core flushes them (in TS order: context emissions first,
//!     then return-value emissions) once the hook returns. The drop-accounting
//!     semantics are preserved exactly.
//!   * The ported [`StationCore::emit`](crate::des::general::des_base::station::StationCore::emit)
//!     routes by calling `core_mut().take()` directly, so graph-routed tokens
//!     land in a station's inbox rather than invoking this module's overriding
//!     `take`. Drive a `PureTransformEntity` by calling its `take` directly
//!     (mirrors `entity.take(...)` in TS); this is a limitation inherited from
//!     the already-ported `StationCore`, not introduced here.
//!   * The unchecked TS `token as I` downcast becomes a checked
//!     `Rc<dyn Any>::downcast` that `panic!`s on a type mismatch (an invariant
//!     violation, per the migration rules).

use std::any::Any;
use std::collections::HashSet;
use std::rc::Rc;

use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{
    AnyToken, ChannelName, DESStation, StationCore, DEFAULT_CHANNEL,
};

/// Result of a single transform invocation.
///
/// Mirrors the TS union `O | readonly O[] | null | undefined`: a single token,
/// many tokens, or nothing (which — if the context emitted nothing either —
/// counts as a drop).
pub enum TransformResult<O> {
    /// `null` / `undefined`: emit nothing from the return value.
    None,
    /// A single output token.
    One(O),
    /// Zero or more output tokens (an empty `Vec` behaves like `None`).
    Many(Vec<O>),
}

/// Destination-channel policy for emitted tokens.
///
/// Mirrors the TS union
/// `ChannelName | ((inputChannel, token) => ChannelName) | undefined`.
pub enum OutputChannel<O> {
    /// `undefined`: emit on the same channel the input arrived on.
    Default,
    /// A fixed destination channel.
    Fixed(ChannelName),
    /// Compute the destination channel from the input channel and token.
    Compute(Box<dyn Fn(&ChannelName, &O) -> ChannelName>),
}

/// Per-invocation context handed to a transform hook.
///
/// Unlike the TS interface, emissions are *buffered* here (see module docs) and
/// flushed by the core after the hook returns.
pub struct TransformContext<O> {
    /// Channel the input token arrived on.
    pub channel: ChannelName,
    /// 0-based index of this invocation within the entity's lifetime.
    pub sequence: usize,
    emitted: Vec<(O, Option<ChannelName>)>,
}

impl<O> TransformContext<O> {
    /// Emit a token, letting the entity's [`OutputChannel`] policy choose the
    /// destination channel (TS `emit(token)`).
    pub fn emit(&mut self, token: O) {
        self.emitted.push((token, None));
    }

    /// Emit a token to an explicit channel (TS `emit(token, channel)`).
    pub fn emit_to(&mut self, token: O, channel: impl Into<ChannelName>) {
        self.emitted.push((token, Some(channel.into())));
    }
}

/// Construction options (TS `TransformEntityOptions`). Closure fields default to
/// no-ops; an empty `input_channels` resolves to `[DEFAULT_CHANNEL]`.
pub struct TransformEntityOptions<I, O> {
    /// Accepted input channel(s). Empty → the default channel.
    pub input_channels: Vec<ChannelName>,
    /// Destination policy for returned token(s). Defaults to the input channel.
    pub output_channel: OutputChannel<O>,
    /// Fail-fast guard for incoming tokens.
    pub validate_input: Box<dyn Fn(&I, &ChannelName)>,
    /// Fail-fast guard for outgoing tokens.
    pub validate_output: Box<dyn Fn(&O, &ChannelName)>,
}

impl<I, O> Default for TransformEntityOptions<I, O> {
    fn default() -> Self {
        TransformEntityOptions {
            input_channels: Vec::new(),
            output_channel: OutputChannel::Default,
            validate_input: Box::new(|_, _| {}),
            validate_output: Box::new(|_, _| {}),
        }
    }
}

/// Shared state for every transform entity (the fields of the TS
/// `abstract class TransformEntity`), embedding a [`StationCore`].
pub struct TransformEntityCore<I, O> {
    /// The underlying DES station state (inboxes, edges, validators).
    pub station: StationCore,
    input_channels: HashSet<ChannelName>,
    output_channel: OutputChannel<O>,
    validate_input: Box<dyn Fn(&I, &ChannelName)>,
    validate_output: Box<dyn Fn(&O, &ChannelName)>,
    /// Number of tokens fed into a transform hook.
    pub processed_count: usize,
    /// Number of tokens emitted downstream.
    pub emitted_count: usize,
    /// Number of inputs that produced no output.
    pub dropped_count: usize,
}

impl<I: Any, O: Any> TransformEntityCore<I, O> {
    /// Build a core from an id and options (TS constructor).
    pub fn new(id: impl Into<String>, opts: TransformEntityOptions<I, O>) -> Self {
        let input_channels: HashSet<ChannelName> = if opts.input_channels.is_empty() {
            std::iter::once(DEFAULT_CHANNEL.to_string()).collect()
        } else {
            opts.input_channels.into_iter().collect()
        };
        TransformEntityCore {
            station: StationCore::new(id),
            input_channels,
            output_channel: opts.output_channel,
            validate_input: opts.validate_input,
            validate_output: opts.validate_output,
            processed_count: 0,
            emitted_count: 0,
            dropped_count: 0,
        }
    }

    /// TS `assertPreconditions`: require at least one input channel.
    pub fn assert_preconditions(&self) -> Check {
        Preconditions::non_empty(
            "TransformEntity",
            &format!("{}.inputChannels", self.station.id),
            &self.input_channel_names(),
        )
    }

    /// TS `inputChannelNames`.
    pub fn input_channel_names(&self) -> Vec<ChannelName> {
        self.input_channels.iter().cloned().collect()
    }

    /// TS `hasQueuedInput`.
    pub fn has_queued_input(&self) -> bool {
        self.input_channel_names()
            .iter()
            .any(|channel| self.station.inbox_size(channel) > 0)
    }

    /// TS `validateTransformInput`: assert the channel is accepted, downcast the
    /// token to `I`, then run the user input guard. `panic!`s on an unexpected
    /// channel or a token of the wrong type (invariant violations).
    pub fn validate_transform_input(&self, token: AnyToken, channel: &str) -> Rc<I> {
        if !self.input_channels.contains(channel) {
            panic!(
                "TransformEntity({}): unexpected input channel \"{}\"",
                self.station.id, channel
            );
        }
        let input = token.downcast::<I>().unwrap_or_else(|_| {
            panic!(
                "TransformEntity({}): token on channel \"{}\" was not the expected input type",
                self.station.id, channel
            )
        });
        (self.validate_input)(&input, &channel.to_string());
        input
    }

    /// Begin a transform invocation: capture the sequence number and bump
    /// `processed_count` (TS prefix of `processTransformResult`).
    fn next_context(&mut self, channel: &str) -> TransformContext<O> {
        let sequence = self.processed_count;
        self.processed_count += 1;
        TransformContext {
            channel: channel.to_string(),
            sequence,
            emitted: Vec::new(),
        }
    }

    /// TS `resolveOutputChannel`.
    fn resolve_output_channel(&self, input_channel: &str, token: &O) -> ChannelName {
        match &self.output_channel {
            OutputChannel::Compute(f) => f(&input_channel.to_string(), token),
            OutputChannel::Fixed(channel) => channel.clone(),
            OutputChannel::Default => input_channel.to_string(),
        }
    }

    /// TS `emitOutput`: validate, route through the station, count.
    fn emit_output(&mut self, token: O, channel: ChannelName) {
        (self.validate_output)(&token, &channel);
        self.station.emit(Rc::new(token) as AnyToken, &channel);
        self.emitted_count += 1;
    }

    /// Flush a completed invocation: emit buffered context tokens (TS
    /// `ctx.emit(...)` happens eagerly) followed by the return value, and apply
    /// the same drop accounting (`dropped_count++` iff nothing was emitted).
    fn flush_result(
        &mut self,
        input_channel: &str,
        ctx: TransformContext<O>,
        result: TransformResult<O>,
    ) {
        let emitted_before = self.emitted_count;
        for (token, channel) in ctx.emitted {
            let channel = channel.unwrap_or_else(|| self.resolve_output_channel(input_channel, &token));
            self.emit_output(token, channel);
        }
        let outputs: Vec<O> = match result {
            TransformResult::None => Vec::new(),
            TransformResult::One(token) => vec![token],
            TransformResult::Many(tokens) => tokens,
        };
        if outputs.is_empty() {
            if self.emitted_count == emitted_before {
                self.dropped_count += 1;
            }
            return;
        }
        for token in outputs {
            let channel = self.resolve_output_channel(input_channel, &token);
            self.emit_output(token, channel);
        }
    }
}

/// Base contract shared by every transform entity (TS `abstract class
/// TransformEntity`). Concrete types expose their embedded [`TransformEntityCore`].
pub trait TransformEntity<I: Any, O: Any>: DESStation {
    /// Borrow the embedded transform core.
    fn tcore(&self) -> &TransformEntityCore<I, O>;
    /// Mutably borrow the embedded transform core.
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<I, O>;

    /// TS `assertPreconditions` — `panic!`s on failure (invariant violation),
    /// to be called from `DESStation::assert_preconditions`.
    fn assert_transform_preconditions(&self) {
        if let Err(e) = self.tcore().assert_preconditions() {
            panic!("{e}");
        }
    }
}

/// Pure-function station (TS `abstract class PureTransformEntity`): subclasses
/// implement only [`transform`](PureTransformEntity::transform); the base owns
/// channels, validation, fan-out, and zero-backlog accounting.
pub trait PureTransformEntity<I: Any, O: Any>: TransformEntity<I, O> {
    /// The pure transform hook (TS `transform`).
    fn transform(&mut self, token: &I, ctx: &mut TransformContext<O>) -> TransformResult<O>;

    /// TS `override take`: validate and process immediately (no queueing).
    fn take(&mut self, token: AnyToken, channel: &str) {
        let input = self.tcore().validate_transform_input(token, channel);
        let mut ctx = self.tcore_mut().next_context(channel);
        let result = self.transform(&input, &mut ctx);
        self.tcore_mut().flush_result(channel, ctx, result);
    }
}

/// Queue-backed transform (TS `abstract class MemoryTransformEntity`): inputs
/// wait in named inboxes until a tick, supporting local memory or per-time-slice
/// processing.
pub trait MemoryTransformEntity<I: Any, O: Any>: TransformEntity<I, O> {
    /// The queued transform hook (TS `transformQueued`).
    fn transform_queued(&mut self, token: &I, ctx: &mut TransformContext<O>) -> TransformResult<O>;

    /// TS inherited `take`: validate, then queue into the inbox.
    fn take(&mut self, token: AnyToken, channel: &str) {
        let input = self.tcore().validate_transform_input(token, channel);
        self.tcore_mut().station.take(input, channel);
    }

    /// TS `runTimeStep`: drain every input channel and process each token.
    fn run_queued(&mut self) {
        let channels = self.tcore().input_channel_names();
        for channel in channels {
            let inputs = self.tcore_mut().station.drain::<I>(&channel);
            for input in inputs {
                let mut ctx = self.tcore_mut().next_context(&channel);
                let result = self.transform_queued(&input, &mut ctx);
                self.tcore_mut().flush_result(&channel, ctx, result);
            }
        }
    }
}

/// Concrete pure-function station wrapping a closure (TS `class FunctionEntity`).
pub struct FunctionEntity<I: Any, O: Any> {
    tcore: TransformEntityCore<I, O>,
    f: Box<dyn Fn(&I, &mut TransformContext<O>) -> TransformResult<O>>,
}

impl<I: Any, O: Any> FunctionEntity<I, O> {
    /// Build a function entity from an id, a transform closure, and options.
    pub fn new(
        id: impl Into<String>,
        f: impl Fn(&I, &mut TransformContext<O>) -> TransformResult<O> + 'static,
        opts: TransformEntityOptions<I, O>,
    ) -> Self {
        FunctionEntity {
            tcore: TransformEntityCore::new(id, opts),
            f: Box::new(f),
        }
    }
}

impl<I: Any, O: Any> TransformEntity<I, O> for FunctionEntity<I, O> {
    fn tcore(&self) -> &TransformEntityCore<I, O> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<I, O> {
        &mut self.tcore
    }
}

impl<I: Any, O: Any> PureTransformEntity<I, O> for FunctionEntity<I, O> {
    fn transform(&mut self, token: &I, ctx: &mut TransformContext<O>) -> TransformResult<O> {
        (self.f)(token, ctx)
    }
}

impl<I: Any, O: Any> DESStation for FunctionEntity<I, O> {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::des::general::des_base::station::StationRef;

    /// Test sink that collects the `f64` tokens routed to it.
    struct CollectSink {
        core: StationCore,
        got: Vec<f64>,
    }

    impl CollectSink {
        fn new(id: &str) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(CollectSink {
                core: StationCore::new(id),
                got: Vec::new(),
            }))
        }
    }

    impl DESStation for CollectSink {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {
            for token in self.core.drain::<f64>(DEFAULT_CHANNEL) {
                self.got.push(*token);
            }
        }
    }

    /// Hand-written pure entity that doubles inbound numeric tokens.
    struct Doubler {
        tcore: TransformEntityCore<f64, f64>,
    }

    impl Doubler {
        fn new(id: &str) -> Self {
            Doubler {
                tcore: TransformEntityCore::new(id, TransformEntityOptions::default()),
            }
        }
    }

    impl TransformEntity<f64, f64> for Doubler {
        fn tcore(&self) -> &TransformEntityCore<f64, f64> {
            &self.tcore
        }
        fn tcore_mut(&mut self) -> &mut TransformEntityCore<f64, f64> {
            &mut self.tcore
        }
    }

    impl PureTransformEntity<f64, f64> for Doubler {
        fn transform(&mut self, token: &f64, _ctx: &mut TransformContext<f64>) -> TransformResult<f64> {
            TransformResult::One(token * 2.0)
        }
    }

    impl DESStation for Doubler {
        fn core(&self) -> &StationCore {
            &self.tcore.station
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.tcore.station
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {}
        fn has_work(&self) -> bool {
            false
        }
        fn assert_preconditions(&mut self) {
            self.assert_transform_preconditions();
        }
    }

    /// Queue-backed entity that doubles tokens on a tick and remembers the last.
    struct QueuedDoubler {
        tcore: TransformEntityCore<f64, f64>,
        previous: f64,
    }

    impl TransformEntity<f64, f64> for QueuedDoubler {
        fn tcore(&self) -> &TransformEntityCore<f64, f64> {
            &self.tcore
        }
        fn tcore_mut(&mut self) -> &mut TransformEntityCore<f64, f64> {
            &mut self.tcore
        }
    }

    impl MemoryTransformEntity<f64, f64> for QueuedDoubler {
        fn transform_queued(
            &mut self,
            token: &f64,
            _ctx: &mut TransformContext<f64>,
        ) -> TransformResult<f64> {
            self.previous = *token;
            TransformResult::One(token * 2.0)
        }
    }

    impl DESStation for QueuedDoubler {
        fn core(&self) -> &StationCore {
            &self.tcore.station
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.tcore.station
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {
            self.run_queued();
        }
        fn has_work(&self) -> bool {
            self.tcore().has_queued_input()
        }
        fn assert_preconditions(&mut self) {
            self.assert_transform_preconditions();
        }
    }

    #[test]
    fn pure_entity_doubles_and_emits_immediately() {
        let sink = CollectSink::new("sink");
        let mut d = Doubler::new("d");
        d.tcore_mut()
            .station
            .pipe(sink.clone() as StationRef, DEFAULT_CHANNEL, DEFAULT_CHANNEL);

        d.take(Rc::new(3.0f64), DEFAULT_CHANNEL);
        d.take(Rc::new(10.0f64), DEFAULT_CHANNEL);

        assert!(!d.has_work()); // zero-backlog: nothing queued
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().got, vec![6.0, 20.0]);
        assert_eq!(d.tcore().processed_count, 2);
        assert_eq!(d.tcore().emitted_count, 2);
        assert_eq!(d.tcore().dropped_count, 0);
    }

    #[test]
    fn function_entity_doubles_and_drops_negatives() {
        let sink = CollectSink::new("sink");
        let mut f = FunctionEntity::new(
            "f",
            |x: &f64, _ctx: &mut TransformContext<f64>| {
                if *x < 0.0 {
                    TransformResult::None
                } else {
                    TransformResult::One(x * 2.0)
                }
            },
            TransformEntityOptions::default(),
        );
        f.tcore_mut()
            .station
            .pipe(sink.clone() as StationRef, DEFAULT_CHANNEL, DEFAULT_CHANNEL);

        f.take(Rc::new(5.0f64), DEFAULT_CHANNEL);
        f.take(Rc::new(-1.0f64), DEFAULT_CHANNEL);

        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().got, vec![10.0]);
        assert_eq!(f.tcore().emitted_count, 1);
        assert_eq!(f.tcore().dropped_count, 1);
    }

    #[test]
    fn memory_entity_queues_then_doubles_on_tick() {
        let sink = CollectSink::new("sink");
        let mut q = QueuedDoubler {
            tcore: TransformEntityCore::new("q", TransformEntityOptions::default()),
            previous: 0.0,
        };
        q.tcore_mut()
            .station
            .pipe(sink.clone() as StationRef, DEFAULT_CHANNEL, DEFAULT_CHANNEL);

        q.take(Rc::new(4.0f64), DEFAULT_CHANNEL);
        assert!(q.has_work()); // queued, not yet processed
        sink.borrow_mut().run_time_step();
        assert!(sink.borrow().got.is_empty());

        q.run_time_step();
        assert!(!q.has_work());
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().got, vec![8.0]);
        assert_eq!(q.previous, 4.0);
        assert_eq!(q.tcore().emitted_count, 1);
    }
}
