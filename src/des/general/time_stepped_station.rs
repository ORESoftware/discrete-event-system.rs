//! Port of `src/des/general/time-stepped-station.ts` — lightweight base classes
//! for fixed-step (tick-driven) DES stations.
//!
//! These lighter bases cover the fixed-step simulations that still need the
//! same discipline as the full queueing framework: explicit stationary
//! entities, movable payloads/tokens, stable IDs for visualisation, and
//! order-independent tick loops.
//!
//! Declarations -> Rust:
//!   * `abstract class TimeSteppedStation`              -> [`TimeSteppedStation`] trait
//!   * `abstract class BufferedTimeSteppedStation<T>`   -> [`BufferedTimeSteppedStation`]
//!     trait + composable base [`BufferedInbox`]
//!   * `abstract class RoutedTimeSteppedStation<T>`     -> [`RoutedTimeSteppedStation`]
//!     trait + composable base [`RoutedOut`]
//!   * `abstract class BidirectionalTimeSteppedStation` -> [`BidirectionalTimeSteppedStation`]
//!     trait + composable base [`BidirectionalChannels`]
//!   * `interface SynchronousDataflowConnection<V>`     -> [`SynchronousDataflowConnection`]
//!   * `abstract class SynchronousDataflowStation<V>`   -> [`SynchronousDataflowStation`]
//!     trait + composable base [`SynchronousDataflowState`]
//!
//! Conversion notes:
//!   * Rust has no class inheritance. The `Routed -> Buffered -> TimeSteppedStation`
//!     chain becomes a `trait TimeSteppedStation` with concrete structs that
//!     COMPOSE a shared base (the inbox / out-connection fields) and `impl` the
//!     traits, rather than mirroring `extends`.
//!   * `inbox: T[]` FIFO -> `VecDeque<T>`; `id: string` -> `String`.
//!   * Downstream-station collections (`out`, `forwardOut`, …) hold shared,
//!     mutable references. TS aliasing maps to `Rc<RefCell<dyn …>>` so that
//!     `emit` can push into each target's inbox.
//!   * `Map<string, V>` (SDF inbox/pending) -> `HashMap<String, V>`.
//!   * Pure scheduling primitives: no RNG/clock here.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

/// Common contract for stationary entities driven by a fixed-size tick.
///
/// `id` is the TS public field exposed as an accessor; `run_time_step` is the
/// abstract method (no default — every concrete station must implement it).
pub trait TimeSteppedStation {
    fn id(&self) -> &str;
    fn run_time_step(&mut self, step_size: f64, t: f64);
}

// -----------------------------------------------------------------------------
// BUFFERED — a station with a FIFO inbox of movable items/tokens.
// -----------------------------------------------------------------------------

/// Composable base holding the FIFO inbox. Concrete buffered stations embed one
/// of these and delegate to it (replaces the `protected inbox` field of the TS
/// `BufferedTimeSteppedStation`).
pub struct BufferedInbox<T> {
    inbox: VecDeque<T>,
}

impl<T> Default for BufferedInbox<T> {
    fn default() -> Self {
        BufferedInbox {
            inbox: VecDeque::new(),
        }
    }
}

impl<T> BufferedInbox<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take_item(&mut self, item: T) {
        self.inbox.push_back(item);
    }

    pub fn inbox_size(&self) -> usize {
        self.inbox.len()
    }

    /// Drain the inbox, returning every queued item and leaving it empty
    /// (`protected drainInbox` in TS).
    pub fn drain_inbox(&mut self) -> Vec<T> {
        self.inbox.drain(..).collect()
    }
}

/// Trait surface needed to route into a buffered station. Extends
/// [`TimeSteppedStation`]; `take_item` / `inbox_size` are the public TS methods.
pub trait BufferedTimeSteppedStation<T>: TimeSteppedStation {
    fn take_item(&mut self, item: T);
    fn inbox_size(&self) -> usize;
}

/// Shared handle to a downstream buffered station (`Rc<RefCell<dyn …>>` so a
/// station can be referenced from multiple upstream `out` lists).
pub type BufferedRef<T> = Rc<RefCell<dyn BufferedTimeSteppedStation<T>>>;

// -----------------------------------------------------------------------------
// ROUTED — a buffered station that emits each item to all downstream stations.
// -----------------------------------------------------------------------------

/// Composable base holding the list of downstream stations (replaces the
/// `protected out` field of the TS `RoutedTimeSteppedStation`).
pub struct RoutedOut<T> {
    out: Vec<BufferedRef<T>>,
}

impl<T> Default for RoutedOut<T> {
    fn default() -> Self {
        RoutedOut { out: Vec::new() }
    }
}

impl<T: Clone> RoutedOut<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_out_connection(&mut self, station: BufferedRef<T>) {
        self.out.push(station);
    }

    /// Emit `item` to every declared downstream station (`protected emit`).
    /// `T: Clone` because the same item is delivered to each target.
    pub fn emit(&self, item: T) {
        for target in &self.out {
            target.borrow_mut().take_item(item.clone());
        }
    }
}

/// Trait surface for a routed station: it is itself buffered and exposes the
/// fluent `add_out_connection`.
pub trait RoutedTimeSteppedStation<T>: BufferedTimeSteppedStation<T> {
    fn add_out_connection(&mut self, station: BufferedRef<T>);
}

// -----------------------------------------------------------------------------
// BIDIRECTIONAL — forward & backward passes over the same graph (e.g. backprop).
// -----------------------------------------------------------------------------

/// Composable base holding the forward/backward inboxes and out-lists (replaces
/// the public fields of the TS `BidirectionalTimeSteppedStation`).
pub struct BidirectionalChannels<F, B> {
    pub forward_inbox: Vec<F>,
    pub backward_inbox: Vec<B>,
    pub forward_out: Vec<BidirectionalRef<F, B>>,
    pub backward_out: Vec<BidirectionalRef<F, B>>,
}

impl<F, B> Default for BidirectionalChannels<F, B> {
    fn default() -> Self {
        BidirectionalChannels {
            forward_inbox: Vec::new(),
            backward_inbox: Vec::new(),
            forward_out: Vec::new(),
            backward_out: Vec::new(),
        }
    }
}

impl<F: Clone, B: Clone> BidirectionalChannels<F, B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take_forward(&mut self, token: F) {
        self.forward_inbox.push(token);
    }

    pub fn take_backward(&mut self, token: B) {
        self.backward_inbox.push(token);
    }

    pub fn add_forward_out(&mut self, station: BidirectionalRef<F, B>) {
        self.forward_out.push(station);
    }

    pub fn add_backward_out(&mut self, station: BidirectionalRef<F, B>) {
        self.backward_out.push(station);
    }

    /// Push `token` to every forward-downstream station (`protected emitForward`).
    pub fn emit_forward(&self, token: F) {
        for target in &self.forward_out {
            target.borrow_mut().take_forward(token.clone());
        }
    }

    /// Push `token` to every backward-downstream station (`protected emitBackward`).
    pub fn emit_backward(&self, token: B) {
        for target in &self.backward_out {
            target.borrow_mut().take_backward(token.clone());
        }
    }
}

/// Trait surface for a bidirectional station.
pub trait BidirectionalTimeSteppedStation<F, B>: TimeSteppedStation {
    fn take_forward(&mut self, token: F);
    fn take_backward(&mut self, token: B);
}

/// Shared handle to a bidirectional station.
pub type BidirectionalRef<F, B> = Rc<RefCell<dyn BidirectionalTimeSteppedStation<F, B>>>;

// -----------------------------------------------------------------------------
// SYNCHRONOUS DATAFLOW — emissions staged in `pending`, visible after commit().
// -----------------------------------------------------------------------------

/// A typed connection to a downstream SDF station (TS
/// `interface SynchronousDataflowConnection<V>`).
pub struct SynchronousDataflowConnection<V> {
    pub kind: String,
    pub target: SynchronousDataflowRef<V>,
}

/// Shared handle to an SDF station.
pub type SynchronousDataflowRef<V> = Rc<RefCell<dyn SynchronousDataflowStation<V>>>;

/// Composable base holding the inbox / pending maps and out-connections
/// (replaces the public fields of the TS `SynchronousDataflowStation`).
///
/// Emissions are staged in each target's `pending` map and become visible only
/// after [`commit`](SynchronousDataflowState::commit), giving every station a
/// frozen view of inputs for the current tick.
pub struct SynchronousDataflowState<V> {
    pub inbox: HashMap<String, V>,
    pub pending: HashMap<String, V>,
    pub out_connections: Vec<SynchronousDataflowConnection<V>>,
}

impl<V> Default for SynchronousDataflowState<V> {
    fn default() -> Self {
        SynchronousDataflowState {
            inbox: HashMap::new(),
            pending: HashMap::new(),
            out_connections: Vec::new(),
        }
    }
}

impl<V> SynchronousDataflowState<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_out(&mut self, kind: impl Into<String>, target: SynchronousDataflowRef<V>) {
        self.out_connections.push(SynchronousDataflowConnection {
            kind: kind.into(),
            target,
        });
    }

    /// Move every pending emission into the next-tick inbox.
    pub fn commit(&mut self) {
        for (key, value) in self.pending.drain() {
            self.inbox.insert(key, value);
        }
    }
}

impl<V: Clone> SynchronousDataflowState<V> {
    /// Stage `value` in the `pending` map of every connection matching `kind`
    /// (`protected emit`).
    pub fn emit(&self, kind: &str, value: V) {
        for c in &self.out_connections {
            if c.kind == kind {
                c.target
                    .borrow_mut()
                    .pending_mut()
                    .insert(kind.to_string(), value.clone());
            }
        }
    }
}

/// Trait surface for an SDF station. `pending_mut` exposes the staging map so
/// upstream stations can stage emissions into it.
pub trait SynchronousDataflowStation<V>: TimeSteppedStation {
    fn pending_mut(&mut self) -> &mut HashMap<String, V>;
    fn commit(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal concrete `TimeSteppedStation`: counts ticks and accumulates time.
    struct CounterStation {
        id: String,
        ticks: usize,
        elapsed: f64,
    }

    impl TimeSteppedStation for CounterStation {
        fn id(&self) -> &str {
            &self.id
        }
        fn run_time_step(&mut self, step_size: f64, _t: f64) {
            self.ticks += 1;
            self.elapsed += step_size;
        }
    }

    #[test]
    fn stepped_station_tick_advances_state() {
        let mut s = CounterStation {
            id: "counter".to_string(),
            ticks: 0,
            elapsed: 0.0,
        };
        assert_eq!(s.id(), "counter");
        s.run_time_step(0.5, 0.0);
        s.run_time_step(0.5, 0.5);
        s.run_time_step(0.5, 1.0);
        assert_eq!(s.ticks, 3);
        assert_eq!(s.elapsed, 1.5);
    }

    /// Buffered station composing a `BufferedInbox`.
    struct Accumulator {
        id: String,
        buf: BufferedInbox<i64>,
        total: i64,
    }

    impl TimeSteppedStation for Accumulator {
        fn id(&self) -> &str {
            &self.id
        }
        fn run_time_step(&mut self, _step_size: f64, _t: f64) {
            for item in self.buf.drain_inbox() {
                self.total += item;
            }
        }
    }

    impl BufferedTimeSteppedStation<i64> for Accumulator {
        fn take_item(&mut self, item: i64) {
            self.buf.take_item(item);
        }
        fn inbox_size(&self) -> usize {
            self.buf.inbox_size()
        }
    }

    #[test]
    fn buffered_station_drains_inbox_on_tick() {
        let mut acc = Accumulator {
            id: "acc".to_string(),
            buf: BufferedInbox::new(),
            total: 0,
        };
        acc.take_item(3);
        acc.take_item(4);
        acc.take_item(5);
        assert_eq!(acc.inbox_size(), 3);
        acc.run_time_step(1.0, 0.0);
        assert_eq!(acc.total, 12);
        assert_eq!(acc.inbox_size(), 0);
        // FIFO ordering preserved on a fresh drain.
        acc.take_item(10);
        assert_eq!(acc.inbox_size(), 1);
    }

    #[test]
    fn routed_station_emits_to_downstream() {
        let downstream: Rc<RefCell<Accumulator>> = Rc::new(RefCell::new(Accumulator {
            id: "down".to_string(),
            buf: BufferedInbox::new(),
            total: 0,
        }));
        let mut router: RoutedOut<i64> = RoutedOut::new();
        router.add_out_connection(downstream.clone() as BufferedRef<i64>);
        router.emit(7);
        router.emit(8);
        assert_eq!(downstream.borrow().inbox_size(), 2);
        downstream.borrow_mut().run_time_step(1.0, 0.0);
        assert_eq!(downstream.borrow().total, 15);
    }
}
