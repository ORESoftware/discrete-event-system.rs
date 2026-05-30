//! Port of `src/des/general/des-base/station.ts`.
//!
//! `DESStation` — the foundation for the "iterative algorithm as DES"
//! hierarchy: typed inboxes, an explicit edge graph, and a `run_time_step`
//! tick hook.
//!
//! ## Rust shape (faithful translation of TS inheritance)
//!
//! TypeScript's `abstract class DESStation` carries shared fields + provided
//! methods + one abstract hook. Rust has no inheritance, so:
//!
//!   * Shared fields live in [`StationCore`], which every concrete station
//!     embeds as a field and exposes via `core()` / `core_mut()`.
//!   * The contract is the [`DESStation`] trait: the required `run_time_step`
//!     (the abstract template-method hook), `core`/`core_mut`/`as_any`
//!     accessors, and provided default methods (`assert_preconditions`,
//!     `has_work`, `on_finalize`, `add_validator`, …).
//!   * Tokens are `Rc<dyn Any>` so a single `emit` can fan a payload out to
//!     several targets (TS shared the same object reference); `drain::<T>()`
//!     downcasts.
//!   * Graph back-edges are `Rc<RefCell<dyn DESStation>>` (the header's
//!     suggested handle), so `emit` can route into another station's inbox.
//!   * Validators are `this`-polymorphic in TS (`Validator<this>`); here they
//!     are stored as `Validator<dyn DESStation>` and the runner validates via
//!     the free fn [`run_station_validation`] (avoids a `&Self`→`&dyn`
//!     coercion inside a default method). Concrete validators downcast through
//!     `station.as_any()`.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::validation::{format_validation_report, run_validators, ValidationCheck, Validator};

/// Channel names are simple strings. The default channel is `"default"`.
pub type ChannelName = String;
pub const DEFAULT_CHANNEL: &str = "default";

/// A payload that flows on an edge. Any `'static` type qualifies; stored as
/// `Rc<dyn Any>` and recovered with `drain::<T>()`.
pub type AnyToken = Rc<dyn Any>;

/// Shared handle to a station in the graph.
pub type StationRef = Rc<RefCell<dyn DESStation>>;

/// An outbound edge: a target station and the channel name on that target.
#[derive(Clone)]
pub struct OutEdge {
    pub target: StationRef,
    pub target_channel: ChannelName,
}

/// Shared state for every station (the fields of the TS `abstract class`).
#[derive(Default)]
pub struct StationCore {
    pub id: String,
    inboxes: HashMap<ChannelName, Vec<AnyToken>>,
    outs: HashMap<ChannelName, Vec<OutEdge>>,
    validators: Vec<Box<dyn Validator<dyn DESStation>>>,
}

impl StationCore {
    pub fn new(id: impl Into<String>) -> Self {
        StationCore {
            id: id.into(),
            inboxes: HashMap::new(),
            outs: HashMap::new(),
            validators: Vec::new(),
        }
    }

    /// Connect this station's `src_channel` output to `target`'s `tgt_channel`
    /// input.
    pub fn pipe(&mut self, target: StationRef, src_channel: &str, tgt_channel: &str) {
        self.outs
            .entry(src_channel.to_string())
            .or_default()
            .push(OutEdge { target, target_channel: tgt_channel.to_string() });
    }

    /// Place a token on the inbox of the given channel.
    pub fn take(&mut self, t: AnyToken, channel: &str) {
        self.inboxes.entry(channel.to_string()).or_default().push(t);
    }

    /// Drain (and clear) the inbox for a channel, downcasting each token to `T`.
    /// Tokens that are not a `T` are dropped (mirrors the unchecked TS `as T`,
    /// but type-safe).
    pub fn drain<T: Any>(&mut self, channel: &str) -> Vec<Rc<T>> {
        let Some(arr) = self.inboxes.get_mut(channel) else {
            return Vec::new();
        };
        let taken = std::mem::take(arr);
        taken.into_iter().filter_map(|t| t.downcast::<T>().ok()).collect()
    }

    /// Drain the raw `Rc<dyn Any>` tokens without downcasting.
    pub fn drain_any(&mut self, channel: &str) -> Vec<AnyToken> {
        self.inboxes.get_mut(channel).map(std::mem::take).unwrap_or_default()
    }

    /// Peek (clone the handles) without consuming.
    pub fn peek(&self, channel: &str) -> Vec<AnyToken> {
        self.inboxes.get(channel).cloned().unwrap_or_default()
    }

    /// Number of pending tokens on a channel.
    pub fn inbox_size(&self, channel: &str) -> usize {
        self.inboxes.get(channel).map_or(0, |v| v.len())
    }

    /// Emit a token on a channel to every connected target.
    ///
    /// SELF-EDGE handling: a station may wire an output channel back to its own
    /// input (the iterative state-loop pipeline pattern: `source → update
    /// (self-loop) → sink`). When that happens, the target `RefCell` is the very
    /// station currently executing `run_time_step`, so it is already borrowed
    /// and `borrow_mut()` would panic. We detect this via `try_borrow_mut` and
    /// route directly into our OWN inbox (`self` *is* the target's core on a
    /// self-edge). This matches the TypeScript engine, where `emit` to self just
    /// appended to the shared inbox (no `RefCell`).
    pub fn emit(&mut self, t: AnyToken, channel: &str) {
        let Some(edges) = self.outs.get(channel) else {
            return;
        };
        // Clone handles first so we don't hold a borrow of `self.outs` while
        // mutating targets.
        let edges: Vec<OutEdge> = edges.clone();
        for edge in edges {
            match edge.target.try_borrow_mut() {
                Ok(mut target) => target.core_mut().take(t.clone(), &edge.target_channel),
                Err(_) => self.take(t.clone(), &edge.target_channel),
            }
        }
    }

    /// Snapshot of inbox sizes by channel.
    pub fn inbox_sizes(&self) -> HashMap<ChannelName, usize> {
        self.inboxes.iter().map(|(k, v)| (k.clone(), v.len())).collect()
    }

    /// Whether any inbox holds work.
    pub fn has_work(&self) -> bool {
        self.inboxes.values().any(|v| !v.is_empty())
    }
}

/// The foundation contract for the iterative-algorithm hierarchy.
pub trait DESStation: Any {
    /// Borrow the shared station state.
    fn core(&self) -> &StationCore;
    /// Mutably borrow the shared station state.
    fn core_mut(&mut self) -> &mut StationCore;
    /// Upcast for validator downcasting / runner introspection.
    fn as_any(&self) -> &dyn Any;

    /// Single iteration of this station's behaviour (the abstract template hook
    /// — family bases implement it as a template method calling finer hooks).
    fn run_time_step(&mut self);

    /// This station's id.
    fn id(&self) -> &str {
        &self.core().id
    }

    /// Pre-run guard, called once before any tick. Default no-op; algorithms
    /// override to fail-fast on invalid parameters.
    fn assert_preconditions(&mut self) {}

    /// Default: any non-empty inbox counts as work.
    fn has_work(&self) -> bool {
        self.core().has_work()
    }

    /// Called once after the loop terminates and before validators run.
    fn on_finalize(&mut self) {}

    /// Register a validator on this station.
    fn add_validator(&mut self, v: Box<dyn Validator<dyn DESStation>>) {
        self.core_mut().validators.push(v);
    }

    /// Number of validators currently registered.
    fn num_validators(&self) -> usize {
        self.core().validators.len()
    }
}

/// Run all validators registered on `station` (free fn so we can pass the
/// `&dyn DESStation` to `Validator<dyn DESStation>` without a `&Self`→`&dyn`
/// coercion inside a default trait method).
pub fn run_station_validation(station: &dyn DESStation) -> Vec<ValidationCheck> {
    run_validators(station, &station.core().validators)
}

/// Pretty-printed validation report for `station`.
pub fn station_validation_report(station: &dyn DESStation) -> String {
    format_validation_report(&run_station_validation(station))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        core: StationCore,
        ticks: u32,
        received: u32,
    }

    impl Counter {
        fn new(id: &str) -> Self {
            Counter { core: StationCore::new(id), ticks: 0, received: 0 }
        }
    }

    impl DESStation for Counter {
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
            let msgs = self.core.drain::<u32>(DEFAULT_CHANNEL);
            self.received += msgs.len() as u32;
            self.ticks += 1;
        }
    }

    #[test]
    fn take_drain_roundtrip() {
        let mut c = Counter::new("c");
        c.core_mut().take(Rc::new(7u32), DEFAULT_CHANNEL);
        c.core_mut().take(Rc::new(8u32), DEFAULT_CHANNEL);
        assert!(c.has_work());
        assert_eq!(c.core().inbox_size(DEFAULT_CHANNEL), 2);
        let drained = c.core_mut().drain::<u32>(DEFAULT_CHANNEL);
        assert_eq!(drained.iter().map(|r| **r).collect::<Vec<_>>(), vec![7, 8]);
        assert!(!c.has_work());
    }

    #[test]
    fn emit_routes_through_graph() {
        let sink = Rc::new(RefCell::new(Counter::new("sink")));
        let mut src = Counter::new("src");
        src.core_mut().pipe(sink.clone() as StationRef, DEFAULT_CHANNEL, DEFAULT_CHANNEL);
        src.core_mut().emit(Rc::new(42u32), DEFAULT_CHANNEL);
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().received, 1);
    }

    #[test]
    fn validators_via_free_fn() {
        use super::super::validation::FnValidator;
        let mut c = Counter::new("c");
        c.ticks = 5;
        let v: Box<dyn Validator<dyn DESStation>> = FnValidator::new("ticked", |s: &dyn DESStation| {
            let counter = s.as_any().downcast_ref::<Counter>().unwrap();
            vec![ValidationCheck {
                name: "ticked".to_string(),
                passed: counter.ticks > 0,
                ..Default::default()
            }]
        })
        .boxed();
        c.add_validator(v);
        assert_eq!(c.num_validators(), 1);
        let checks = run_station_validation(&c);
        assert!(checks[0].passed);
    }
}
