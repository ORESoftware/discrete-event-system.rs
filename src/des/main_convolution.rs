//! Port of `src/des/main-convolution.ts`.
//!
//! 1-D convolution as a streaming DES pipeline:
//!
//! ```text
//!     SignalSource  ── x[n] ─▶  ConvolutionStation  ── y[n] ─▶  CollectorSink
//! ```
//!
//! `y[n] = (x * h)[n] = Σ_k h[k] · x[n−k]` (DSP-textbook convolution, mode
//! `full`).
//!
//! ## Rust shape
//!   * The three TS classes (`Sample`, `SignalSource`, `ConvolutionStation`,
//!     `CollectorSink`) extend `RoutedTimeSteppedStation<Sample>`. Here they are
//!     plain structs and the dataflow is wired explicitly in [`run_convolution`]
//!     (each station's `run_time_step` takes its downstream station by
//!     `&mut`), faithfully reproducing the per-tick emit semantics without the
//!     `Rc<RefCell<dyn …>>` graph. See
//!     `crate::des::general::time_stepped_station` for the base traits.
//!   * PORT NOTE: the TS `fisherYatesShuffle(order)` per tick is purely
//!     cosmetic — the output `y` is order-independent by construction (the
//!     convolver emits in `outIdx` order regardless of tick scheduling), and
//!     only the reported `ticks` count depends on order. We run the stations in
//!     declared order (`src → conv → sink`) and drop the shuffle + its
//!     `withSeed` wrapper.
//!   * `mulberry32` for the test-signal generator is reused from
//!     `crate::des::general::prng`.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

/// One movable sample flowing through the pipeline.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub n: usize,
    pub value: f64,
}

/// Emits one input value per tick.
struct SignalSource {
    id: String,
    signal: Vec<f64>,
    idx: usize,
}

impl SignalSource {
    fn new(id: &str, signal: Vec<f64>) -> Self {
        SignalSource { id: id.to_string(), signal, idx: 0 }
    }
    fn run_time_step(&mut self, conv: &mut ConvolutionStation) {
        if self.idx < self.signal.len() {
            conv.take(Sample { n: self.idx, value: self.signal[self.idx] });
            self.idx += 1;
        }
    }
    fn is_done(&self) -> bool {
        self.idx >= self.signal.len()
    }
}

/// Convolves the incoming stream with a fixed kernel via a circular ring buffer.
struct ConvolutionStation {
    id: String,
    kernel: Vec<f64>,
    inbox: VecDeque<Sample>,
    buffer: Vec<f64>,
    head: usize,
    warmup: usize,
    out_idx: usize,
    flushed_after: usize,
    want_full_mode: bool,
}

impl ConvolutionStation {
    fn new(id: &str, kernel: Vec<f64>, full_mode: bool) -> Self {
        let k = kernel.len();
        ConvolutionStation {
            id: id.to_string(),
            kernel,
            inbox: VecDeque::new(),
            buffer: vec![0.0; k],
            head: 0,
            warmup: 0,
            out_idx: 0,
            flushed_after: 0,
            want_full_mode: full_mode,
        }
    }

    fn take(&mut self, s: Sample) {
        self.inbox.push_back(s);
    }

    /// `y[n] = Σ_{k=0}^{K-1} h[k] · x[n − k]`, reading the ring buffer.
    fn dot(&self) -> f64 {
        let k_len = self.kernel.len();
        let mut y = 0.0;
        for k in 0..k_len {
            let idx = (self.head + k_len * 2 - 1 - k) % k_len;
            y += self.kernel[k] * self.buffer[idx];
        }
        y
    }

    /// Flush a single trailing zero (mode `full`), emitting one output to `sink`.
    fn flush_once(&mut self, sink: &mut CollectorSink) {
        self.buffer[self.head] = 0.0;
        self.head = (self.head + 1) % self.kernel.len();
        self.flushed_after += 1;
        let y = self.dot();
        sink.take(Sample { n: self.out_idx, value: y });
        self.out_idx += 1;
    }

    fn needs_flush(&self) -> bool {
        self.want_full_mode && self.flushed_after < self.kernel.len() - 1
    }

    fn run_time_step(&mut self, sink: &mut CollectorSink) {
        while let Some(s) = self.inbox.pop_front() {
            self.buffer[self.head] = s.value;
            self.head = (self.head + 1) % self.kernel.len();
            self.warmup += 1;
            let y = self.dot();
            sink.take(Sample { n: self.out_idx, value: y });
            self.out_idx += 1;
        }
    }
}

/// Accumulates the convolution outputs.
struct CollectorSink {
    id: String,
    inbox: VecDeque<Sample>,
    results: Vec<Sample>,
}

impl CollectorSink {
    fn new(id: &str) -> Self {
        CollectorSink { id: id.to_string(), inbox: VecDeque::new(), results: Vec::new() }
    }
    fn take(&mut self, s: Sample) {
        self.inbox.push_back(s);
    }
    fn run_time_step(&mut self) {
        while let Some(s) = self.inbox.pop_front() {
            self.results.push(s);
        }
    }
}

/// Mode metadata reported alongside the result.
#[derive(Clone, Debug)]
pub struct ConvolutionMeta {
    pub signal_len: usize,
    pub kernel_len: usize,
    pub mode: &'static str,
}

/// Result of a convolution run.
#[derive(Clone, Debug)]
pub struct ConvolutionResult {
    pub y: Vec<f64>,
    pub ticks: usize,
    pub meta: ConvolutionMeta,
}

/// Run the streaming convolution to completion (mode `full`).
pub fn run_convolution(signal: &[f64], kernel: &[f64]) -> ConvolutionResult {
    let mut src = SignalSource::new("src", signal.to_vec());
    let mut conv = ConvolutionStation::new("conv", kernel.to_vec(), true);
    let mut sink = CollectorSink::new("sink");

    let mut ticks = 0usize;
    loop {
        // Declared-order tick (see PORT NOTE on the omitted shuffle).
        src.run_time_step(&mut conv);
        conv.run_time_step(&mut sink);
        sink.run_time_step();
        ticks += 1;
        if src.is_done() && conv.needs_flush() {
            conv.flush_once(&mut sink);
        } else if src.is_done() && !conv.needs_flush() {
            sink.run_time_step();
            break;
        }
    }

    ConvolutionResult {
        y: sink.results.iter().map(|s| s.value).collect(),
        ticks,
        meta: ConvolutionMeta {
            signal_len: signal.len(),
            kernel_len: kernel.len(),
            mode: "full",
        },
    }
}

/// Symmetric triangular FIR (exact peak in the middle), normalized to sum = 1.
fn make_triangle_kernel(k: usize) -> Vec<f64> {
    let mut h = vec![0.0; k];
    let peak = (k as f64 - 1.0) / 2.0;
    let mut s = 0.0;
    for (i, hi) in h.iter_mut().enumerate() {
        *hi = 1.0 - (i as f64 - peak).abs() / (peak + 1.0);
        s += *hi;
    }
    h.iter().map(|v| v / s).collect()
}

/// Multiscale test signal: 0.1 Hz sine + 0.4 Hz cosine + small white noise.
fn make_test_signal(n: usize, seed: u32) -> Vec<f64> {
    let mut rng = mulberry32(seed);
    let mut out = vec![0.0; n];
    for (i, oi) in out.iter_mut().enumerate() {
        let i_f = i as f64;
        *oi = (2.0 * PI * 0.1 * i_f).sin()
            + 0.5 * (2.0 * PI * 0.4 * i_f).cos()
            + 0.1 * (rng.next_float() - 0.5);
    }
    out
}

/// Entry point (TS top-level `main`). Env vars: `N`, `K`, `SEED`.
pub fn run() {
    let n = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(64usize);
    let k = std::env::var("K").ok().and_then(|v| v.parse().ok()).unwrap_or(7usize);
    let seed = std::env::var("SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(42u32);

    let signal = make_test_signal(n, seed);
    let kernel = make_triangle_kernel(k);

    println!("# Convolution simulation");
    println!("#   signal length = {}", signal.len());
    println!("#   kernel length = {}  (triangular, normalized)", kernel.len());
    println!("#   seed          = {seed}");

    let result = run_convolution(&signal, &kernel);

    println!("# output length    = {}", result.y.len());
    println!("# wall-clock ticks = {}", result.ticks);
    println!("# first 12 outputs:");
    for i in 0..result.y.len().min(12) {
        println!("  y[{i}] = {:.6}", result.y[i]);
    }
    println!("# ...");
    println!("# last 4 outputs:");
    for i in result.y.len().saturating_sub(4)..result.y.len() {
        println!("  y[{i}] = {:.6}", result.y[i]);
    }

    // PORT NOTE: the TS writes out/convolution-framework.json via fs; the JSON
    // serialization is omitted here (no serde dependency assumed). Wire to
    // serde_json when the output-artifact layer is ported.
    println!("# (JSON artifact write omitted in port — see PORT NOTE)");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convolution with a unit-impulse kernel returns the signal (plus zeros).
    #[test]
    fn impulse_kernel_is_identity() {
        let signal = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let r = run_convolution(&signal, &kernel);
        assert_eq!(r.y.len(), 3);
        assert!((r.y[0] - 1.0).abs() < 1e-12);
        assert!((r.y[1] - 2.0).abs() < 1e-12);
        assert!((r.y[2] - 3.0).abs() < 1e-12);
    }

    /// Full-mode length is signalLen + kernelLen − 1.
    #[test]
    fn full_mode_length() {
        let signal = [1.0, 1.0, 1.0, 1.0];
        let kernel = [0.5, 0.5];
        let r = run_convolution(&signal, &kernel);
        assert_eq!(r.y.len(), signal.len() + kernel.len() - 1);
    }
}
