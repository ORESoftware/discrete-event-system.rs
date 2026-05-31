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
}

// ── Source ───────────────────────────────────────────────────────────────────

/// A time-driven signal source (no inputs, one output).
#[derive(Clone, Debug)]
pub enum SourceKind {
    Const(f64),
    /// `before` until `t0`, then `after`.
    Step { t0: f64, before: f64, after: f64 },
    /// `slope * t + intercept`.
    Ramp { slope: f64, intercept: f64 },
    /// `amp * sin(2π·freq·t) + bias`.
    Sine { amp: f64, freq: f64, bias: f64 },
}

pub struct Source {
    name: String,
    kind: SourceKind,
}

impl Source {
    pub fn new(name: &str, kind: SourceKind) -> Self {
        Source { name: name.to_string(), kind }
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
        Gain { name: name.to_string(), k }
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
        Saturation { name: name.to_string(), lo, hi }
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
        Affine { name: name.to_string(), m, b }
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
        Sum { name: name.to_string(), weights }
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
        Integrator { name: name.to_string(), state: x0, last_t: 0.0 }
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
}

// ── Closure map (1→1) ─────────────────────────────────────────────────────────

/// A 1→1 op backed by a closure (the `FnTransform` escape hatch).
pub struct Map {
    name: String,
    f: Box<dyn Fn(f64) -> f64>,
}
impl Map {
    pub fn new(name: &str, f: impl Fn(f64) -> f64 + 'static) -> Self {
        Map { name: name.to_string(), f: Box::new(f) }
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
}
