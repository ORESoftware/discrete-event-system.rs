//! Port of `src/des/general/des-base/composite-station.ts`.
//!
//! A [`DESStation`] that owns an internal station sub-graph and exposes it to
//! the outer topology through explicit input/output ports. Outer tokens are
//! routed in through input ports; each tick the children run once in declared
//! order, then egress bridges are drained back out.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//! * `children: DESStation[]` → `Vec<StationRef>` (`Rc<RefCell<dyn DESStation>>`).
//!   `add_substation` returns the typed shared handle for further wiring.
//! * The private `class CompositePortBridgeStation` → a `pub` struct + `impl
//!   DESStation` whose `run_time_step` is a no-op egress buffer. Output ports
//!   keep a concrete `Rc<RefCell<CompositePortBridgeStation>>` handle (to drain
//!   the bridge) AND the bridge is registered as a child (to be ticked /
//!   validated), exactly like the single shared object in TS.
//! * `override hasWork/onFinalize/numValidators/assertPreconditions` → trait
//!   default-method overrides that recurse into children (no `super`; the
//!   "own" part is `self.core().*`).
//! * `runValidation()` (virtual in TS) → an inherent method
//!   [`CompositeDESStation::run_validation`]. **Dep flag:** the ported
//!   `station.rs` exposes validation via the free fn `run_station_validation`
//!   (which reads only `core().validators`), not a virtual trait method. So the
//!   runner's aggregation does NOT auto-recurse into a composite's children, and
//!   a *nested* composite child cannot re-dispatch to its own
//!   `run_validation`. For the common case (leaf children) this method
//!   reproduces the TS behaviour; for runner-driven aggregation of composite
//!   children, call `run_validation()` explicitly.
//! * `Record<string, number>` / nested → `HashMap<String, usize>` /
//!   `HashMap<String, HashMap<String, usize>>`.
//! * Borrow-checker: ticking children needs scoped `RefCell` borrows; ingress/
//!   egress clone the (cheap, `Rc`-backed) port descriptors so `self.core_mut()`
//!   can be borrowed without aliasing the port vectors.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::station::{
    run_station_validation, AnyToken, ChannelName, DESStation, StationCore, StationRef,
};
use super::validation::{run_validators, ValidationCheck, Validator};

/// Internal egress buffer station: holds tokens emitted by an internal source
/// on the bridge's channel until the parent's egress pass re-emits them out.
pub struct CompositePortBridgeStation {
    core: StationCore,
}

impl CompositePortBridgeStation {
    fn new(id: impl Into<String>) -> Self {
        CompositePortBridgeStation {
            core: StationCore::new(id),
        }
    }

    /// Drain (and clear) the bridge's buffer for `channel`, returning the raw
    /// tokens for re-emission. (`drainPort<T>` in TS; egress re-emits opaque
    /// tokens, so this returns `AnyToken` rather than a downcast type.)
    pub fn drain_port(&mut self, channel: &str) -> Vec<AnyToken> {
        self.core.drain_any(channel)
    }

    /// Whether the bridge currently buffers any token.
    pub fn has_port_work(&self) -> bool {
        self.core.has_work()
    }
}

impl DESStation for CompositePortBridgeStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
}

/// An input port: outer-channel tokens routed to a child's target channel.
#[derive(Clone)]
struct CompositeInputPort {
    outer_channel: ChannelName,
    target: StationRef,
    target_channel: ChannelName,
}

/// An output port: an internal egress bridge surfaced on an outer channel.
#[derive(Clone)]
struct CompositeOutputPort {
    outer_channel: ChannelName,
    bridge: Rc<RefCell<CompositePortBridgeStation>>,
}

/// Snapshot of a composite station's inbox occupancy (own + per child).
#[derive(Clone, Debug, Default)]
pub struct CompositeStationSnapshot {
    pub id: String,
    pub child_ids: Vec<String>,
    pub inboxes: HashMap<String, usize>,
    pub child_inboxes: HashMap<String, HashMap<String, usize>>,
}

/// A DES station that owns an internal station graph.
///
/// Useful when a model has meaningful internal queueing/protocol structure but
/// callers should still see a single station in the outer topology. Outer
/// tokens are routed through explicit input/output ports; internal substations
/// run one tick per parent tick in declared order.
pub struct CompositeDESStation {
    core: StationCore,
    children: Vec<StationRef>,
    input_ports: Vec<CompositeInputPort>,
    output_ports: Vec<CompositeOutputPort>,
    tick: usize,
    /// The composite's OWN validators. Stored here rather than in `StationCore`
    /// because that struct exposes no cross-module count/iterate accessor (and
    /// `station.rs` must not be modified); this lets the `num_validators` /
    /// `run_validation` overrides faithfully reproduce TS `super.*` behaviour.
    own_validators: Vec<Box<dyn Validator<dyn DESStation>>>,
}

impl CompositeDESStation {
    pub fn new(id: impl Into<String>) -> Self {
        CompositeDESStation {
            core: StationCore::new(id),
            children: Vec::new(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            tick: 0,
            own_validators: Vec::new(),
        }
    }

    /// Register a substation and return its shared handle for further wiring.
    pub fn add_substation<S: DESStation + 'static>(
        &mut self,
        station: Rc<RefCell<S>>,
    ) -> Rc<RefCell<S>> {
        self.children.push(station.clone());
        station
    }

    /// Expose an outer input channel, routing its tokens into `target`'s
    /// `target_channel`. (TS defaulted `targetChannel = outerChannel`; pass it
    /// explicitly here.)
    pub fn expose_input(
        &mut self,
        outer_channel: &str,
        target: StationRef,
        target_channel: &str,
    ) -> &mut Self {
        self.input_ports.push(CompositeInputPort {
            outer_channel: outer_channel.to_string(),
            target,
            target_channel: target_channel.to_string(),
        });
        self
    }

    /// Expose an internal `source`'s `source_channel` output as the composite's
    /// `outer_channel` output (via a generated egress bridge). (TS defaulted
    /// `source_channel = DEFAULT_CHANNEL`, `outer_channel = source_channel`;
    /// pass them explicitly here.)
    pub fn expose_output(
        &mut self,
        source: StationRef,
        source_channel: &str,
        outer_channel: &str,
    ) -> &mut Self {
        let bridge_id = format!(
            "{}:out:{}:{}",
            self.core.id,
            outer_channel,
            self.output_ports.len()
        );
        let bridge = Rc::new(RefCell::new(CompositePortBridgeStation::new(bridge_id)));
        self.children.push(bridge.clone() as StationRef);
        source.borrow_mut().core_mut().pipe(
            bridge.clone() as StationRef,
            source_channel,
            outer_channel,
        );
        self.output_ports.push(CompositeOutputPort {
            outer_channel: outer_channel.to_string(),
            bridge,
        });
        self
    }

    /// The internal substations (children), in declared order.
    pub fn child_stations(&self) -> &[StationRef] {
        &self.children
    }

    /// Run own + recursive child validators (faithful port of the virtual
    /// `runValidation` override). See the module-level dep flag: the runner's
    /// own aggregation does not invoke this automatically.
    pub fn run_validation(&self) -> Vec<ValidationCheck> {
        let mut out = run_validators(self as &dyn DESStation, &self.own_validators);
        for child in &self.children {
            out.extend(run_station_validation(&*child.borrow()));
        }
        out
    }

    /// Snapshot of own + per-child inbox sizes.
    pub fn snapshot_composite(&self) -> CompositeStationSnapshot {
        let mut child_ids = Vec::with_capacity(self.children.len());
        let mut child_inboxes: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for child in &self.children {
            let cb = child.borrow();
            let id = cb.id().to_string();
            child_ids.push(id.clone());
            child_inboxes.insert(id, cb.core().inbox_sizes());
        }
        CompositeStationSnapshot {
            id: self.core.id.clone(),
            child_ids,
            inboxes: self.core.inbox_sizes(),
            child_inboxes,
        }
    }

    /// The composite's own tick counter (TS `protected compositeTick()`).
    pub fn composite_tick(&self) -> usize {
        self.tick
    }

    fn route_ingress(&mut self) {
        // Clone the (Rc-backed) port descriptors so we can mutate `self.core`
        // without aliasing `self.input_ports`.
        let ports = self.input_ports.clone();
        for port in &ports {
            let tokens = self.core.drain_any(&port.outer_channel);
            for token in tokens {
                port.target
                    .borrow_mut()
                    .core_mut()
                    .take(token, &port.target_channel);
            }
        }
    }

    fn route_egress(&mut self) {
        let ports = self.output_ports.clone();
        for port in &ports {
            let tokens = port.bridge.borrow_mut().drain_port(&port.outer_channel);
            for token in tokens {
                self.core.emit(token, &port.outer_channel);
            }
        }
    }
}

impl DESStation for CompositeDESStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn assert_preconditions(&mut self) {
        for child in &self.children {
            child.borrow_mut().assert_preconditions();
        }
    }

    fn has_work(&self) -> bool {
        if self.core.has_work() {
            return true;
        }
        for child in &self.children {
            if child.borrow().has_work() {
                return true;
            }
        }
        for port in &self.output_ports {
            if port.bridge.borrow().has_port_work() {
                return true;
            }
        }
        false
    }

    fn run_time_step(&mut self) {
        self.route_ingress();
        for child in &self.children {
            child.borrow_mut().run_time_step();
        }
        self.route_egress();
        self.tick += 1;
    }

    fn on_finalize(&mut self) {
        for child in &self.children {
            child.borrow_mut().on_finalize();
        }
    }

    fn add_validator(&mut self, v: Box<dyn Validator<dyn DESStation>>) {
        self.own_validators.push(v);
    }

    fn num_validators(&self) -> usize {
        let own = self.own_validators.len();
        let children: usize = self
            .children
            .iter()
            .map(|c| c.borrow().num_validators())
            .sum();
        own + children
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::validation::{FnValidator, Validator};

    /// A child station that forwards tokens from its `"in"` inbox to its
    /// `"out"` channel each tick.
    struct Passthrough {
        core: StationCore,
    }

    impl Passthrough {
        fn new(id: &str) -> Self {
            Passthrough {
                core: StationCore::new(id),
            }
        }
    }

    impl DESStation for Passthrough {
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
            let tokens = self.core.drain_any("in");
            for t in tokens {
                self.core.emit(t, "out");
            }
        }
    }

    /// A terminal station that counts tokens arriving on its `"in"` channel.
    struct Collector {
        core: StationCore,
        received: usize,
    }

    impl Collector {
        fn new(id: &str) -> Self {
            Collector {
                core: StationCore::new(id),
                received: 0,
            }
        }
    }

    impl DESStation for Collector {
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
            self.received += self.core.drain_any("in").len();
        }
    }

    fn build() -> (
        Rc<RefCell<CompositeDESStation>>,
        Rc<RefCell<Passthrough>>,
        Rc<RefCell<Collector>>,
    ) {
        let comp = Rc::new(RefCell::new(CompositeDESStation::new("comp")));
        let child = Rc::new(RefCell::new(Passthrough::new("pass")));
        let sink = Rc::new(RefCell::new(Collector::new("sink")));
        {
            let mut c = comp.borrow_mut();
            c.add_substation(child.clone());
            c.expose_input("in", child.clone() as StationRef, "in");
            c.expose_output(child.clone() as StationRef, "out", "out");
            // Route the composite's outer "out" to the external sink's "in".
            c.core_mut().pipe(sink.clone() as StationRef, "out", "in");
        }
        (comp, child, sink)
    }

    #[test]
    fn routes_token_end_to_end_in_one_tick() {
        let (comp, _child, sink) = build();
        comp.borrow_mut().core_mut().take(Rc::new(99u32), "in");
        assert!(comp.borrow().has_work());

        comp.borrow_mut().run_time_step();
        // Egress emitted into the sink's "in" inbox; have the sink consume it.
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().received, 1);
        assert_eq!(comp.borrow().composite_tick(), 1);
        assert!(!comp.borrow().has_work());
    }

    #[test]
    fn num_validators_aggregates_children() {
        let (comp, child, _sink) = build();
        // One validator on the composite itself, one on the child.
        {
            let v: Box<dyn Validator<dyn DESStation>> =
                FnValidator::new("comp-ok", |_: &dyn DESStation| vec![]).boxed();
            comp.borrow_mut().add_validator(v);
            let cv: Box<dyn Validator<dyn DESStation>> =
                FnValidator::new("child-ok", |_: &dyn DESStation| vec![]).boxed();
            child.borrow_mut().add_validator(cv);
        }
        // own(1) + child(1) + bridge(0).
        assert_eq!(comp.borrow().num_validators(), 2);
    }

    #[test]
    fn snapshot_reports_children() {
        let (comp, _child, _sink) = build();
        comp.borrow_mut().core_mut().take(Rc::new(1u8), "in");
        let snap = comp.borrow().snapshot_composite();
        assert_eq!(snap.id, "comp");
        // child + one egress bridge.
        assert_eq!(snap.child_ids.len(), 2);
        assert!(snap.child_ids.iter().any(|c| c == "pass"));
        assert_eq!(snap.inboxes.get("in").copied(), Some(1));
    }
}
