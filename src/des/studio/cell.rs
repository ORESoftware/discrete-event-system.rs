//! Layer 2 — the **runtime cell**: what actually executes inside a visual block.
//!
//! A [`RuntimeCell`] is an ordered pipeline of one or more [`RuntimeOp`]s — the
//! runtime-layer elements (`Transform` / `StatefulTransform` primitives wrapped
//! as graph-steppable ops). This is the layer that "executes algorithms and does
//! calculations"; the visual block in [`super::graph`] is only the topology +
//! presentation skin around it.
//!
//! A single visual block may therefore hold **several** Layer-2 elements (e.g. a
//! controller block = `[Sum, Gain, Saturation]`). The cell threads scalar port
//! signals through its stages and validates that the stages chain (each stage's
//! output-port count equals the next stage's input-port count).

use std::collections::VecDeque;

use crate::des::shared::transform::{StatefulTransform, Transform};

use super::graph::StudioError;

/// One scalar per port (a width-1 signal). Buses are modeled as multiple ports.
pub type Scalar = f64;

/// A Layer-2 runtime element: steppable, with a fixed input/output port count.
/// Object-safe so a cell can hold a heterogeneous pipeline behind `dyn`.
pub trait RuntimeOp {
    fn name(&self) -> &str;
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    /// Advance one step at time `t`. `inputs.len() == n_in()`; returns `n_out()`.
    fn step(&mut self, t: f64, inputs: &[Scalar]) -> Vec<Scalar>;
    /// Reset any internal state (for re-runs).
    fn reset(&mut self) {}
    /// Whether this op carries state across steps (a `StatefulTransform`). The
    /// executive seam reads this to decide a graph needs a stepped (discrete)
    /// executive rather than a pure stateless evaluation.
    fn is_stateful(&self) -> bool {
        false
    }
    /// Nesting depth of this op: a leaf op is `0`; a [`Composite`] reports how
    /// deeply runtime elements are nested inside it. Visual blocks never nest,
    /// but runtime ops may, so this is measured here rather than on the graph.
    fn nested_depth(&self) -> usize {
        0
    }
}

// ── Source ───────────────────────────────────────────────────────────────────

/// A time-driven signal source (no inputs, one output).
#[derive(Clone, Debug)]
pub enum SourceKind {
    Const(f64),
    /// `before` until `t0`, then `after`.
    Step {
        t0: f64,
        before: f64,
        after: f64,
    },
    /// `slope * t + intercept`.
    Ramp {
        slope: f64,
        intercept: f64,
    },
    /// `amp * sin(2π·freq·t) + bias`.
    Sine {
        amp: f64,
        freq: f64,
        bias: f64,
    },
}

pub struct Source {
    name: String,
    kind: SourceKind,
}

impl Source {
    pub fn new(name: &str, kind: SourceKind) -> Self {
        Source {
            name: name.to_string(),
            kind,
        }
    }
    pub fn value(&self, t: f64) -> f64 {
        match &self.kind {
            SourceKind::Const(c) => *c,
            SourceKind::Step { t0, before, after } => {
                if t >= *t0 {
                    *after
                } else {
                    *before
                }
            }
            SourceKind::Ramp { slope, intercept } => slope * t + intercept,
            SourceKind::Sine { amp, freq, bias } => {
                amp * (2.0 * std::f64::consts::PI * freq * t).sin() + bias
            }
        }
    }
}

impl RuntimeOp for Source {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        0
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, t: f64, _inputs: &[Scalar]) -> Vec<Scalar> {
        vec![self.value(t)]
    }
}

// ── Pure 1→1 ops (genuine `Transform<f64, f64>`) ──────────────────────────────

/// Scalar gain `y = k·x` — a pure [`Transform`].
pub struct Gain {
    name: String,
    pub k: f64,
}
impl Gain {
    pub fn new(name: &str, k: f64) -> Self {
        Gain {
            name: name.to_string(),
            k,
        }
    }
}
impl Transform<f64, f64> for Gain {
    fn transform(&self, x: f64) -> f64 {
        self.k * x
    }
}

/// Clamp `y = min(max(x, lo), hi)` — a pure [`Transform`].
pub struct Saturation {
    name: String,
    pub lo: f64,
    pub hi: f64,
}
impl Saturation {
    pub fn new(name: &str, lo: f64, hi: f64) -> Self {
        Saturation {
            name: name.to_string(),
            lo,
            hi,
        }
    }
}
impl Transform<f64, f64> for Saturation {
    fn transform(&self, x: f64) -> f64 {
        x.max(self.lo).min(self.hi)
    }
}

/// Affine map `y = m·x + b` — a pure [`Transform`].
pub struct Affine {
    name: String,
    pub m: f64,
    pub b: f64,
}
impl Affine {
    pub fn new(name: &str, m: f64, b: f64) -> Self {
        Affine {
            name: name.to_string(),
            m,
            b,
        }
    }
}
impl Transform<f64, f64> for Affine {
    fn transform(&self, x: f64) -> f64 {
        self.m * x + self.b
    }
}

macro_rules! impl_pure_1to1 {
    ($ty:ty) => {
        impl RuntimeOp for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn n_in(&self) -> usize {
                1
            }
            fn n_out(&self) -> usize {
                1
            }
            fn step(&mut self, _t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
                vec![self.transform(inputs.first().copied().unwrap_or(0.0))]
            }
        }
    };
}
impl_pure_1to1!(Gain);
impl_pure_1to1!(Saturation);
impl_pure_1to1!(Affine);

// ── Weighted sum (k→1) ────────────────────────────────────────────────────────

/// Weighted sum `y = Σ wᵢ·xᵢ` over `weights.len()` input ports.
pub struct Sum {
    name: String,
    weights: Vec<f64>,
}
impl Sum {
    pub fn new(name: &str, weights: Vec<f64>) -> Self {
        Sum {
            name: name.to_string(),
            weights,
        }
    }
}
impl RuntimeOp for Sum {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        self.weights.len()
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, _t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        let y = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum();
        vec![y]
    }
}

// ── Integrator (stateful 1→1, a `StatefulTransform`) ──────────────────────────

/// Forward-Euler integrator `s += (t − t_prev)·x` — a [`StatefulTransform`].
pub struct Integrator {
    name: String,
    state: f64,
    last_t: f64,
}
impl Integrator {
    pub fn new(name: &str, x0: f64) -> Self {
        Integrator {
            name: name.to_string(),
            state: x0,
            last_t: 0.0,
        }
    }
}
impl StatefulTransform<(f64, f64), f64> for Integrator {
    /// `(t, x) -> integral`.
    fn transform(&mut self, input: (f64, f64)) -> f64 {
        let (t, x) = input;
        let dt = (t - self.last_t).max(0.0);
        self.state += dt * x;
        self.last_t = t;
        self.state
    }
}
impl RuntimeOp for Integrator {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        1
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        vec![self.transform((t, inputs.first().copied().unwrap_or(0.0)))]
    }
    fn reset(&mut self) {
        self.state = 0.0;
        self.last_t = 0.0;
    }
    fn is_stateful(&self) -> bool {
        true
    }
}

// ── Runtime-layer station / movable ops ───────────────────────────────────────
//
// These bring the DES runtime primitives — a queueing `StationEntity` and a
// `Movable` in transit — into the dataflow studio as Layer-2 elements, expressed
// at the per-tick signal level (a `StatefulTransform`). The full token run-loop
// `DESStation` remains reachable via the DES-run-loop executive (see
// `crate::des::exec`); these ops model the same semantics inside a signal graph.

/// Single-server queue (a `StationEntity` at the signal level): backlog absorbs
/// arrivals, the server drains up to `service_rate` per tick, and departures are
/// emitted. `arrivals -> departures`, with backlog as internal state.
pub struct Queue {
    name: String,
    service_rate: f64,
    backlog: f64,
}
impl Queue {
    pub fn new(name: &str, service_rate: f64) -> Self {
        Queue {
            name: name.to_string(),
            service_rate: service_rate.max(0.0),
            backlog: 0.0,
        }
    }
    pub fn backlog(&self) -> f64 {
        self.backlog
    }
}
impl StatefulTransform<f64, f64> for Queue {
    /// `arrivals -> departures`.
    fn transform(&mut self, arrivals: f64) -> f64 {
        self.backlog += arrivals.max(0.0);
        let served = self.backlog.min(self.service_rate);
        self.backlog -= served;
        served
    }
}
impl RuntimeOp for Queue {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        1
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, _t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        vec![self.transform(inputs.first().copied().unwrap_or(0.0))]
    }
    fn reset(&mut self) {
        self.backlog = 0.0;
    }
    fn is_stateful(&self) -> bool {
        true
    }
}

/// Transport delay (a `Movable` in transit): a fixed-length conveyor that emits
/// the value it received `delay` ticks ago. `x(k) -> x(k − delay)`.
pub struct TransportDelay {
    name: String,
    delay: usize,
    buf: VecDeque<f64>,
}
impl TransportDelay {
    pub fn new(name: &str, delay: usize) -> Self {
        let delay = delay.max(1);
        TransportDelay {
            name: name.to_string(),
            delay,
            buf: VecDeque::from(vec![0.0; delay]),
        }
    }
}
impl StatefulTransform<f64, f64> for TransportDelay {
    fn transform(&mut self, x: f64) -> f64 {
        self.buf.push_back(x);
        self.buf.pop_front().unwrap_or(0.0)
    }
}
impl RuntimeOp for TransportDelay {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        1
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, _t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        vec![self.transform(inputs.first().copied().unwrap_or(0.0))]
    }
    fn reset(&mut self) {
        self.buf = VecDeque::from(vec![0.0; self.delay]);
    }
    fn is_stateful(&self) -> bool {
        true
    }
}

// ── Closure map (1→1) ─────────────────────────────────────────────────────────

/// A 1→1 op backed by a closure (the `FnTransform` escape hatch).
pub struct Map {
    name: String,
    f: Box<dyn Fn(f64) -> f64>,
}
impl Map {
    pub fn new(name: &str, f: impl Fn(f64) -> f64 + 'static) -> Self {
        Map {
            name: name.to_string(),
            f: Box::new(f),
        }
    }
}
impl RuntimeOp for Map {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        1
    }
    fn n_out(&self) -> usize {
        1
    }
    fn step(&mut self, _t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        vec![(self.f)(inputs.first().copied().unwrap_or(0.0))]
    }
}

// ── Composite (nested runtime elements) ──────────────────────────────────────
//
// The architecture has two distinct nesting rules:
//   * **Visual blocks cannot nest** — a `VisualNode` owns a `RuntimeCell`, never
//     another node/graph (enforced structurally in `super::graph`).
//   * **Runtime elements CAN nest** — a Layer-2 op may itself contain a whole
//     sub-cell of further Layer-2 ops. `Composite` is that recursion: it wraps a
//     `RuntimeCell` and presents it as a single op, so a block's cell can hold a
//     `Composite` that holds a cell that holds ops, to any depth, all inside one
//     visual block.

/// A Layer-2 op that is itself a [`RuntimeCell`] of nested ops. Lets runtime
/// elements compose recursively while the visual block stays flat.
pub struct Composite {
    name: String,
    inner: RuntimeCell,
}
impl Composite {
    pub fn new(name: &str, inner: RuntimeCell) -> Self {
        Composite {
            name: name.to_string(),
            inner,
        }
    }
    /// How many Layer-2 elements are nested directly inside this composite.
    pub fn inner_len(&self) -> usize {
        self.inner.len()
    }
    /// Maximum nesting depth: a flat composite is depth 1; one wrapping another
    /// composite is depth 2; and so on.
    pub fn depth(&self) -> usize {
        1 + self.inner.max_child_depth()
    }
}
impl RuntimeOp for Composite {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_in(&self) -> usize {
        self.inner.n_in()
    }
    fn n_out(&self) -> usize {
        self.inner.n_out()
    }
    fn step(&mut self, t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        self.inner.step(t, inputs)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn is_stateful(&self) -> bool {
        self.inner.has_state()
    }
    fn nested_depth(&self) -> usize {
        self.depth()
    }
}

// ── The cell: a pipeline of ≥1 Layer-2 ops ────────────────────────────────────

/// A runtime cell — the executable contents of one visual block. Holds an
/// ordered pipeline of one or more [`RuntimeOp`]s; the block's input ports feed
/// the first stage, each stage feeds the next, and the last stage produces the
/// block's output ports.
///
/// This is **Layer 2 only**: a cell can never contain a graph or another visual
/// block, which is exactly why visual blocks cannot nest.
pub struct RuntimeCell {
    stages: Vec<Box<dyn RuntimeOp>>,
}

impl RuntimeCell {
    /// Build a cell from a non-empty op pipeline, validating that consecutive
    /// stages chain (`stage[i].n_out() == stage[i+1].n_in()`).
    pub fn new(stages: Vec<Box<dyn RuntimeOp>>) -> Result<Self, StudioError> {
        if stages.is_empty() {
            return Err(StudioError::EmptyCell);
        }
        for w in stages.windows(2) {
            if w[0].n_out() != w[1].n_in() {
                return Err(StudioError::CellPipelineMismatch {
                    upstream: w[0].name().to_string(),
                    out_ports: w[0].n_out(),
                    downstream: w[1].name().to_string(),
                    in_ports: w[1].n_in(),
                });
            }
        }
        Ok(RuntimeCell { stages })
    }

    /// A single-op cell.
    pub fn single(op: Box<dyn RuntimeOp>) -> Self {
        RuntimeCell { stages: vec![op] }
    }

    pub fn n_in(&self) -> usize {
        self.stages[0].n_in()
    }
    pub fn n_out(&self) -> usize {
        self.stages[self.stages.len() - 1].n_out()
    }
    /// Number of Layer-2 elements in this block.
    pub fn len(&self) -> usize {
        self.stages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
    /// The names of the Layer-2 elements, in pipeline order.
    pub fn element_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name().to_string()).collect()
    }

    /// Whether any Layer-2 element in this cell carries state across steps.
    pub fn has_state(&self) -> bool {
        self.stages.iter().any(|s| s.is_stateful())
    }

    /// The deepest runtime-element nesting under this cell (0 if every stage is
    /// a leaf op). Each nested [`Composite`] adds a level. Visual blocks are
    /// always flat; this measures only the runtime-element tree inside a block.
    pub fn max_child_depth(&self) -> usize {
        self.stages
            .iter()
            .map(|s| s.nested_depth())
            .max()
            .unwrap_or(0)
    }

    /// Thread the block's `inputs` through every stage and return its outputs.
    pub fn step(&mut self, t: f64, inputs: &[Scalar]) -> Vec<Scalar> {
        let mut signal = inputs.to_vec();
        for stage in self.stages.iter_mut() {
            signal = stage.step(t, &signal);
        }
        signal
    }

    pub fn reset(&mut self) {
        for s in self.stages.iter_mut() {
            s.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_threads_multiple_layer2_ops() {
        // One visual block containing TWO Layer-2 elements: gain then saturation.
        let mut cell = RuntimeCell::new(vec![
            Box::new(Gain::new("gain", 0.5)),
            Box::new(Saturation::new("sat", -1.0, 2.0)),
        ])
        .unwrap();
        assert_eq!(cell.len(), 2);
        assert_eq!(cell.n_in(), 1);
        assert_eq!(cell.n_out(), 1);
        // 10 * 0.5 = 5, clamped to 2.
        assert_eq!(cell.step(0.0, &[10.0]), vec![2.0]);
        // 2 * 0.5 = 1, within band.
        assert_eq!(cell.step(0.0, &[2.0]), vec![1.0]);
    }

    #[test]
    fn empty_cell_is_rejected() {
        match RuntimeCell::new(vec![]) {
            Err(StudioError::EmptyCell) => {}
            Err(other) => panic!("expected EmptyCell, got {other:?}"),
            Ok(_) => panic!("expected EmptyCell, got Ok"),
        }
    }

    #[test]
    fn cell_rejects_mismatched_pipeline() {
        // Sum has 1 output port; Sum-then-Sum(2-in) cannot chain.
        let result = RuntimeCell::new(vec![
            Box::new(Sum::new("a", vec![1.0, 1.0])),
            Box::new(Sum::new("b", vec![1.0, 1.0])),
        ]);
        match result {
            Err(StudioError::CellPipelineMismatch { .. }) => {}
            Err(other) => panic!("expected CellPipelineMismatch, got {other:?}"),
            Ok(_) => panic!("expected CellPipelineMismatch, got Ok"),
        }
    }

    #[test]
    fn integrator_accumulates_over_time() {
        let mut cell = RuntimeCell::single(Box::new(Integrator::new("int", 0.0)));
        assert_eq!(cell.step(1.0, &[2.0]), vec![2.0]); // dt=1, +2
        assert_eq!(cell.step(2.0, &[3.0]), vec![5.0]); // dt=1, +3
    }

    #[test]
    fn queue_caps_departures_at_service_rate_and_tracks_state() {
        let mut q = Queue::new("server", 5.0);
        assert_eq!(q.transform(8.0), 5.0); // 8 arrive, serve 5
        assert_eq!(q.backlog(), 3.0); // 3 wait
        assert_eq!(q.transform(8.0), 5.0); // 3+8=11, serve 5
        assert_eq!(q.backlog(), 6.0);
        let cell = RuntimeCell::single(Box::new(Queue::new("server", 5.0)));
        assert!(cell.has_state(), "a queue is stateful");
    }

    #[test]
    fn runtime_elements_nest_inside_one_block_cell() {
        // A composite wraps a sub-cell (gain ▸ saturation); another composite
        // wraps THAT — runtime elements nesting two levels deep, all of which is
        // still the contents of a *single* (flat) visual block's cell.
        let inner = RuntimeCell::new(vec![
            Box::new(Gain::new("gain", 2.0)),
            Box::new(Saturation::new("sat", -10.0, 10.0)),
        ])
        .unwrap();
        let lvl1 = Composite::new("shaper", inner);
        assert_eq!(lvl1.depth(), 1);
        assert_eq!(lvl1.inner_len(), 2);

        let lvl2 = Composite::new("outer", RuntimeCell::single(Box::new(lvl1)));
        assert_eq!(lvl2.depth(), 2, "two levels of nested runtime elements");

        // The whole nested tree is one stage of a block cell and runs normally.
        let mut cell = RuntimeCell::new(vec![
            Box::new(Affine::new("bias", 1.0, 1.0)),
            Box::new(lvl2),
        ])
        .unwrap();
        assert_eq!(cell.max_child_depth(), 2);
        // x=3 → affine 3+1=4 → composite(gain 2 =8, sat=8) = 8.
        assert_eq!(cell.step(0.0, &[3.0]), vec![8.0]);
    }

    #[test]
    fn composite_reports_state_from_nested_ops() {
        // A composite wrapping a stateful op is itself stateful.
        let inner = RuntimeCell::single(Box::new(Integrator::new("int", 0.0)));
        let comp = Composite::new("wrapped-int", inner);
        let cell = RuntimeCell::single(Box::new(comp));
        assert!(
            cell.has_state(),
            "nested integrator makes the cell stateful"
        );
    }

    #[test]
    fn transport_delay_shifts_signal_by_n() {
        let mut cell = RuntimeCell::single(Box::new(TransportDelay::new("belt", 3)));
        assert_eq!(cell.step(0.0, &[1.0]), vec![0.0]); // buffer warming up
        assert_eq!(cell.step(0.0, &[2.0]), vec![0.0]);
        assert_eq!(cell.step(0.0, &[3.0]), vec![0.0]);
        assert_eq!(cell.step(0.0, &[4.0]), vec![1.0]); // 1.0 arrives 3 ticks later
        assert_eq!(cell.step(0.0, &[5.0]), vec![2.0]);
    }
}
