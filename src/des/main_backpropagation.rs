//! Port of `src/des/main-backpropagation.ts`.
//!
//! Backpropagation through a 2-3-1 fully-connected sigmoid network as a
//! discrete-event system. Each tick is one mini-batch step; layers are
//! stations; activations and gradients flow forward and backward over the same
//! edge graph:
//!
//! ```text
//!   XorSource ──▶ Layer1 (2→3) ──▶ Layer2 (3→1) ──▶ LossStation
//!        ▲             ▲                ▲                │
//!        └─ backward ──┴── backward ────┴─── backward ───┘
//! ```
//!
//! Sequential SGD: the source waits for a backward done-signal before emitting
//! the next sample, so per-sample weight updates apply in the same order
//! regardless of station-execution order.
//!
//! ## Rust shape
//!   * The TS classes extend `BidirectionalTimeSteppedStation<ForwardToken,
//!     BackwardToken>` (`crate::des::general::time_stepped_station`). Here they
//!     are plain structs; each `run` returns its forward/backward emissions and
//!     [`run_backprop`] routes them along the fixed topology — reproducing the
//!     dataflow without the `Rc<RefCell<dyn …>>` graph.
//!   * PORT NOTE: the TS `fisherYatesShuffle(order)` per tick is cosmetic. With
//!     sequential SGD exactly one sample is in flight, so the sequence of weight
//!     updates (and thus the final weights) is order-independent; only the tick
//!     count depends on order. We run in declared order (`src → L1 → L2 → loss`)
//!     and drop the shuffle.
//!   * `mulberry32` is reused from `crate::des::general::prng`.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
/// `d/dz σ(z) = σ(z)(1−σ(z))`, expressed via the activation `a = σ(z)`.
fn sigmoid_prime_from_a(a: f64) -> f64 {
    a * (1.0 - a)
}

#[derive(Clone)]
struct SampleCache {
    input: Vec<f64>,
    activation: Vec<f64>,
}

#[derive(Clone)]
struct ForwardToken {
    sample_id: usize,
    payload: Vec<f64>,
    target: Vec<f64>,
}

#[derive(Clone)]
struct BackwardToken {
    sample_id: usize,
    grad: Vec<f64>,
}

/// A fully-connected sigmoid layer.
struct LayerStation {
    id: String,
    w: Vec<Vec<f64>>, // [outDim][inDim]
    b: Vec<f64>,      // [outDim]
    lr: f64,
    cache: HashMap<usize, SampleCache>,
    forward_inbox: Vec<ForwardToken>,
    backward_inbox: Vec<BackwardToken>,
}

impl LayerStation {
    fn new(id: &str, w: Vec<Vec<f64>>, b: Vec<f64>, lr: f64) -> Self {
        LayerStation {
            id: id.to_string(),
            w,
            b,
            lr,
            cache: HashMap::new(),
            forward_inbox: Vec::new(),
            backward_inbox: Vec::new(),
        }
    }

    /// Returns `(forward_emissions, backward_emissions)`.
    fn run_time_step(&mut self) -> (Vec<ForwardToken>, Vec<BackwardToken>) {
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();
        // Forward pass: naive nested loop (matches the Python reference's
        // float-summation order exactly).
        let fwd_tokens = std::mem::take(&mut self.forward_inbox);
        for t in fwd_tokens {
            let input = t.payload;
            let mut a = vec![0.0; self.w.len()];
            for i in 0..self.w.len() {
                let mut zi = self.b[i];
                for j in 0..self.w[i].len() {
                    zi += self.w[i][j] * input[j];
                }
                a[i] = sigmoid(zi);
            }
            self.cache.insert(
                t.sample_id,
                SampleCache {
                    input,
                    activation: a.clone(),
                },
            );
            fwd.push(ForwardToken {
                sample_id: t.sample_id,
                payload: a,
                target: t.target,
            });
        }
        // Backward pass.
        let bwd_tokens = std::mem::take(&mut self.backward_inbox);
        for t in bwd_tokens {
            let c = self
                .cache
                .remove(&t.sample_id)
                .expect("missing cache entry");
            let a = &c.activation;
            let in_dim = c.input.len();
            let out_dim = a.len();
            // dL/dz = grad_a ∘ σ'(z),  σ'(z) = a(1−a)
            let mut dz = vec![0.0; out_dim];
            for i in 0..out_dim {
                dz[i] = t.grad[i] * sigmoid_prime_from_a(a[i]);
            }
            // grad_input[j] = Σ_i W[i][j]·dz[i]  (BEFORE mutating W).
            let mut grad_input = vec![0.0; in_dim];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    grad_input[j] += self.w[i][j] * dz[i];
                }
            }
            for i in 0..out_dim {
                for j in 0..in_dim {
                    self.w[i][j] -= self.lr * dz[i] * c.input[j];
                }
                self.b[i] -= self.lr * dz[i];
            }
            bwd.push(BackwardToken {
                sample_id: t.sample_id,
                grad: grad_input,
            });
        }
        (fwd, bwd)
    }
}

/// MSE loss: `L = ½ Σ (a − y)²`; the initial backward gradient is `(a − y)`.
struct LossStation {
    id: String,
    losses: Vec<f64>,
    forward_inbox: Vec<ForwardToken>,
}

impl LossStation {
    fn new() -> Self {
        LossStation {
            id: "loss".to_string(),
            losses: Vec::new(),
            forward_inbox: Vec::new(),
        }
    }
    fn run_time_step(&mut self) -> Vec<BackwardToken> {
        let mut bwd = Vec::new();
        let fwd = std::mem::take(&mut self.forward_inbox);
        for t in fwd {
            let a = &t.payload;
            let y = &t.target;
            let mut loss = 0.0;
            let mut grad = vec![0.0; a.len()];
            for i in 0..a.len() {
                let e = a[i] - y[i];
                loss += 0.5 * e * e;
                grad[i] = e;
            }
            self.losses.push(loss);
            bwd.push(BackwardToken {
                sample_id: t.sample_id,
                grad,
            });
        }
        bwd
    }
}

/// Cycles through the 4 XOR samples; sequential SGD (one sample in flight).
struct XorSource {
    id: String,
    total: usize,
    idx: usize,
    in_flight: i64,
    backward_inbox: Vec<BackwardToken>,
}

const XOR_SAMPLES: [([f64; 2], [f64; 1]); 4] = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

impl XorSource {
    fn new(id: &str, total: usize) -> Self {
        XorSource {
            id: id.to_string(),
            total,
            idx: 0,
            in_flight: 0,
            backward_inbox: Vec::new(),
        }
    }
    fn run_time_step(&mut self) -> Vec<ForwardToken> {
        // Drain done-signal backward tokens.
        let drained = std::mem::take(&mut self.backward_inbox);
        self.in_flight -= drained.len() as i64;
        let mut fwd = Vec::new();
        if self.in_flight == 0 && self.idx < self.total {
            let s = &XOR_SAMPLES[self.idx % 4];
            fwd.push(ForwardToken {
                sample_id: self.idx,
                payload: s.0.to_vec(),
                target: s.1.to_vec(),
            });
            self.idx += 1;
            self.in_flight = 1;
        }
        fwd
    }
    fn is_done(&self) -> bool {
        self.idx >= self.total && self.in_flight == 0
    }
}

/// Initial weights for the 2-`hidden`-1 net.
#[derive(Clone, Debug)]
pub struct InitialWeights {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}

/// Backprop run outcome.
#[derive(Clone, Debug)]
pub struct BackpropResult {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
    pub loss_history: Vec<f64>,
    pub ticks: usize,
    pub predictions: Vec<f64>,
}

/// Initial weights from `U(-1, 1)` using `mulberry32(seed)`.
pub fn init_weights(seed: u32, hidden: usize) -> InitialWeights {
    let mut rng = mulberry32(seed);
    let mut draw = || 2.0 * rng.next_float() - 1.0;
    let w1: Vec<Vec<f64>> = (0..hidden)
        .map(|_| (0..2).map(|_| draw()).collect())
        .collect();
    let b1: Vec<f64> = (0..hidden).map(|_| draw()).collect();
    let w2: Vec<Vec<f64>> = vec![(0..hidden).map(|_| draw()).collect()];
    let b2: Vec<f64> = vec![draw()];
    InitialWeights { w1, b1, w2, b2 }
}

/// Train the network by running the DES to completion.
pub fn run_backprop(init: &InitialWeights, total_samples: usize, lr: f64) -> BackpropResult {
    let mut src = XorSource::new("src", total_samples);
    let mut l1 = LayerStation::new("L1", init.w1.clone(), init.b1.clone(), lr);
    let mut l2 = LayerStation::new("L2", init.w2.clone(), init.b2.clone(), lr);
    let mut loss = LossStation::new();

    let mut ticks = 0usize;
    while !src.is_done()
        || !l1.forward_inbox.is_empty()
        || !l1.backward_inbox.is_empty()
        || !l2.forward_inbox.is_empty()
        || !l2.backward_inbox.is_empty()
        || !loss.forward_inbox.is_empty()
    {
        // src → L1 (forward).
        let src_fwd = src.run_time_step();
        l1.forward_inbox.extend(src_fwd);
        // L1 → L2 (forward), L1 → src (backward).
        let (l1_fwd, l1_bwd) = l1.run_time_step();
        l2.forward_inbox.extend(l1_fwd);
        src.backward_inbox.extend(l1_bwd);
        // L2 → loss (forward), L2 → L1 (backward).
        let (l2_fwd, l2_bwd) = l2.run_time_step();
        loss.forward_inbox.extend(l2_fwd);
        l1.backward_inbox.extend(l2_bwd);
        // loss → L2 (backward).
        let loss_bwd = loss.run_time_step();
        l2.backward_inbox.extend(loss_bwd);

        ticks += 1;
        if ticks > total_samples * 100 {
            panic!("runaway: training did not converge ticks");
        }
    }

    // Final predictions on the 4 XOR cases (forward-only).
    let mut predictions = Vec::new();
    for s in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
        let mut a1 = vec![0.0; l1.w.len()];
        for i in 0..l1.w.len() {
            let mut z = l1.b[i];
            for j in 0..l1.w[i].len() {
                z += l1.w[i][j] * s[j];
            }
            a1[i] = sigmoid(z);
        }
        let mut a2 = vec![0.0; l2.w.len()];
        for i in 0..l2.w.len() {
            let mut z = l2.b[i];
            for j in 0..l2.w[i].len() {
                z += l2.w[i][j] * a1[j];
            }
            a2[i] = sigmoid(z);
        }
        predictions.push(a2[0]);
    }

    BackpropResult {
        w1: l1.w,
        b1: l1.b,
        w2: l2.w,
        b2: l2.b,
        loss_history: loss.losses,
        ticks,
        predictions,
    }
}

/// Entry point (TS top-level `main`). Env vars: `SEED`, `N`, `LR`.
pub fn run() {
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7u32);
    let n = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10000usize);
    let lr = std::env::var("LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5_f64);

    let init = init_weights(seed, 3);

    println!("# Backpropagation simulation (2-3-1, sigmoid + MSE)");
    println!("#   seed = {seed}   samples = {n}   lr = {lr}");

    let result = run_backprop(&init, n, lr);

    println!("# ticks = {}", result.ticks);
    let last100: Vec<f64> = result
        .loss_history
        .iter()
        .rev()
        .take(100)
        .copied()
        .collect();
    let avg_loss = last100.iter().sum::<f64>() / last100.len() as f64;
    println!("# avg loss over last 100 samples = {avg_loss:.3e}");
    println!("# XOR predictions:");
    println!("    0 XOR 0  →  {:.4}    (target 0)", result.predictions[0]);
    println!("    0 XOR 1  →  {:.4}    (target 1)", result.predictions[1]);
    println!("    1 XOR 0  →  {:.4}    (target 1)", result.predictions[2]);
    println!("    1 XOR 1  →  {:.4}    (target 0)", result.predictions[3]);

    // PORT NOTE: the TS writes out/backprop-framework.json via fs; omitted here
    // (no serde dependency assumed).
    println!("# (JSON artifact write omitted in port — see PORT NOTE)");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XOR is learnable: after enough samples the predictions separate the
    /// two classes.
    #[test]
    fn learns_xor() {
        let init = init_weights(7, 3);
        let r = run_backprop(&init, 20000, 0.5);
        assert!(r.predictions[0] < 0.5, "0,0 -> {}", r.predictions[0]);
        assert!(r.predictions[1] > 0.5, "0,1 -> {}", r.predictions[1]);
        assert!(r.predictions[2] > 0.5, "1,0 -> {}", r.predictions[2]);
        assert!(r.predictions[3] < 0.5, "1,1 -> {}", r.predictions[3]);
    }
}
