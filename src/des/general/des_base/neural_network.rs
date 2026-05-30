//! Port of `src/des/general/des-base/neural-network.ts`.
//!
//! DES base "classes" for neural networks. Neural nets are the hybrid case:
//! their forward/backward passes are numerical computations, but their training
//! data, inference requests, and snapshots move through the station graph as
//! queued tokens. A neural model is just a [`NeuralNetworkLike`]; stations
//! provide queueing semantics and typed channels.
//!
//! ## Rust shape (faithful translation of TS inheritance)
//!
//!   * `interface NeuralNetworkLike` → trait [`NeuralNetworkLike`] (`predict`
//!     required; `parameterCount`/`clone` optional → `parameter_count` has a
//!     provided `None` default, `clone` is dropped — we avoid `Clone` of trait
//!     objects and hold the model by a generic `N` instead).
//!   * `interface TrainableNeuralNetwork` → trait [`TrainableNeuralNetwork`]
//!     (`trainSample`'s inline `{loss, prediction}` → named
//!     [`TrainSampleResult`]).
//!   * The five `Neural*Token` classes → plain structs (tokens are carried as
//!     `Rc<dyn Any>` by the station core).
//!   * `class NeuralNetworkStation<N> extends DESStation` → struct holding a
//!     [`StationCore`] + the model `N`, plus `impl DESStation`.
//!   * `class SupervisedNeuralNetworkStation extends NeuralNetworkStation` →
//!     composition: it embeds a [`NeuralNetworkStation`] as `base` and calls its
//!     methods (no `extends`); the shared station core lives in `base`.
//!
//! `meta: Record<string, unknown>` becomes a [`Meta`] map keyed to a minimal
//! [`MetaValue`] enum (FLAGGED stand-in for `serde_json::Value`, which is not a
//! crate dependency).

use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::station::{DESStation, StationCore};

/// Numeric vector payload (`type NumericVector = number[]`).
pub type NumericVector = Vec<f64>;

/// Opaque token metadata. FLAGGED: minimal local stand-in for the TS
/// `Record<string, unknown>` / `serde_json::Value`; `serde`/`serde_json` are
/// not crate dependencies, so we carry a small tagged value enum that is enough
/// to round-trip metadata through the station graph unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
}

/// Token metadata map (`Record<string, unknown>`).
pub type Meta = HashMap<String, MetaValue>;

// ── Model contracts ─────────────────────────────────────────────────────────

/// A neural model: a deterministic forward pass plus its dimensions. `clone()`
/// from the TS interface is intentionally dropped (we hold the model by a
/// generic `N` rather than as a trait object, so trait-object cloning is not
/// needed).
pub trait NeuralNetworkLike {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn predict(&self, input: &[f64]) -> NumericVector;
    /// Optional in TS → provided default returning `None`.
    fn parameter_count(&self) -> Option<usize> {
        None
    }
}

/// Result of one supervised training step (TS inline `{loss, prediction}`).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSampleResult {
    pub loss: f64,
    pub prediction: NumericVector,
}

/// A model that can be trained one (input, target) sample at a time.
pub trait TrainableNeuralNetwork: NeuralNetworkLike {
    fn train_sample(
        &mut self,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
    ) -> TrainSampleResult;
}

// ── Tokens ───────────────────────────────────────────────────────────────────

/// Inference request flowing on the `infer` channel.
#[derive(Clone, Debug)]
pub struct NeuralInferenceToken {
    pub id: String,
    pub input: NumericVector,
    pub meta: Meta,
}

impl NeuralInferenceToken {
    pub fn new(id: impl Into<String>, input: NumericVector) -> Self {
        NeuralInferenceToken { id: id.into(), input, meta: Meta::new() }
    }
}

/// Supervised training sample flowing on the `train` channel.
#[derive(Clone, Debug)]
pub struct SupervisedSampleToken {
    pub id: String,
    pub input: NumericVector,
    pub target: NumericVector,
    pub meta: Meta,
}

impl SupervisedSampleToken {
    pub fn new(id: impl Into<String>, input: NumericVector, target: NumericVector) -> Self {
        SupervisedSampleToken { id: id.into(), input, target, meta: Meta::new() }
    }
}

/// Prediction emitted on the `prediction` channel.
#[derive(Clone, Debug)]
pub struct NeuralPredictionToken {
    pub id: String,
    pub input: NumericVector,
    pub output: NumericVector,
    pub meta: Meta,
}

/// Per-sample training result emitted on the `training-result` channel.
#[derive(Clone, Debug)]
pub struct NeuralTrainingResultToken {
    pub sample_id: String,
    pub loss: f64,
    pub prediction: NumericVector,
    pub target: NumericVector,
    pub step: usize,
    pub meta: Meta,
}

/// Periodic snapshot emitted on the `snapshot` channel. `loss` /
/// `parameter_count` are `Option` (TS `number | null`).
#[derive(Clone, Debug)]
pub struct NeuralSnapshotToken {
    pub training_step: usize,
    pub loss: Option<f64>,
    pub parameter_count: Option<usize>,
}

// ── NeuralNetworkStation ──────────────────────────────────────────────────────

/// Station that answers inference requests from a [`NeuralNetworkLike`] model.
pub struct NeuralNetworkStation<N: NeuralNetworkLike + 'static> {
    core: StationCore,
    network: N,
}

impl<N: NeuralNetworkLike + 'static> NeuralNetworkStation<N> {
    pub const CH_INFER: &'static str = "infer";
    pub const CH_PREDICTION: &'static str = "prediction";
    pub const CH_SNAPSHOT: &'static str = "snapshot";

    pub fn new(id: impl Into<String>, network: N) -> Self {
        NeuralNetworkStation { core: StationCore::new(id), network }
    }

    pub fn get_network(&self) -> &N {
        &self.network
    }

    pub fn get_network_mut(&mut self) -> &mut N {
        &mut self.network
    }

    /// Drain pending inference requests, predict, and emit one
    /// [`NeuralPredictionToken`] per request on `CH_PREDICTION`.
    pub fn process_inference_queue(&mut self) {
        let requests = self.core.drain::<NeuralInferenceToken>(Self::CH_INFER);
        for req in requests {
            let output = self.network.predict(&req.input);
            let token = NeuralPredictionToken {
                id: req.id.clone(),
                input: req.input.clone(),
                output,
                meta: req.meta.clone(),
            };
            self.core.emit(Rc::new(token), Self::CH_PREDICTION);
        }
    }
}

impl<N: NeuralNetworkLike + 'static> DESStation for NeuralNetworkStation<N> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn run_time_step(&mut self) {
        self.process_inference_queue();
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_INFER) > 0
    }
}

// ── SupervisedNeuralNetworkStation ────────────────────────────────────────────

/// Options for [`SupervisedNeuralNetworkStation`].
#[derive(Clone, Copy, Debug)]
pub struct SupervisedNeuralNetworkStationOptions {
    pub learning_rate: f64,
    /// Emit a snapshot after every N samples. `0` disables snapshots.
    pub snapshot_every: usize,
}

impl SupervisedNeuralNetworkStationOptions {
    pub fn new(learning_rate: f64) -> Self {
        SupervisedNeuralNetworkStationOptions { learning_rate, snapshot_every: 0 }
    }
}

/// Station that trains a [`TrainableNeuralNetwork`] on samples and still answers
/// inference requests. Composes a [`NeuralNetworkStation`] (the shared station
/// core lives in `base`) instead of inheriting it.
pub struct SupervisedNeuralNetworkStation<N: TrainableNeuralNetwork + 'static> {
    base: NeuralNetworkStation<N>,
    pub loss_history: Vec<f64>,
    training_step: usize,
    learning_rate: f64,
    snapshot_every: usize,
}

impl<N: TrainableNeuralNetwork + 'static> SupervisedNeuralNetworkStation<N> {
    pub const CH_TRAIN: &'static str = "train";
    pub const CH_TRAINING_RESULT: &'static str = "training-result";

    pub fn new(
        id: impl Into<String>,
        network: N,
        opts: SupervisedNeuralNetworkStationOptions,
    ) -> Self {
        SupervisedNeuralNetworkStation {
            base: NeuralNetworkStation::new(id, network),
            loss_history: Vec::new(),
            training_step: 0,
            learning_rate: opts.learning_rate,
            snapshot_every: opts.snapshot_every,
        }
    }

    pub fn get_network(&self) -> &N {
        self.base.get_network()
    }

    pub fn get_network_mut(&mut self) -> &mut N {
        self.base.get_network_mut()
    }

    pub fn get_training_step(&self) -> usize {
        self.training_step
    }
}

impl<N: TrainableNeuralNetwork + 'static> DESStation for SupervisedNeuralNetworkStation<N> {
    fn core(&self) -> &StationCore {
        self.base.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.base.core_mut()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn has_work(&self) -> bool {
        self.base.has_work() || self.base.core().inbox_size(Self::CH_TRAIN) > 0
    }

    fn run_time_step(&mut self) {
        let samples = self.base.core_mut().drain::<SupervisedSampleToken>(Self::CH_TRAIN);
        for sample in samples {
            let r = self.base.network.train_sample(
                &sample.input,
                &sample.target,
                self.learning_rate,
            );
            self.training_step += 1;
            self.loss_history.push(r.loss);
            let result = NeuralTrainingResultToken {
                sample_id: sample.id.clone(),
                loss: r.loss,
                prediction: r.prediction,
                target: sample.target.clone(),
                step: self.training_step,
                meta: sample.meta.clone(),
            };
            self.base.core_mut().emit(Rc::new(result), Self::CH_TRAINING_RESULT);
            if self.snapshot_every > 0 && self.training_step.is_multiple_of(self.snapshot_every) {
                let snapshot = NeuralSnapshotToken {
                    training_step: self.training_step,
                    loss: Some(r.loss),
                    parameter_count: self.base.network.parameter_count(),
                };
                self.base
                    .core_mut()
                    .emit(Rc::new(snapshot), NeuralNetworkStation::<N>::CH_SNAPSHOT);
            }
        }
        self.base.process_inference_queue();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny dense linear model `y = W·x + b` trained by MSE gradient descent —
    /// enough to exercise both forward and backward passes.
    struct LinearNet {
        w: Vec<Vec<f64>>, // out_dim × in_dim
        b: Vec<f64>,      // out_dim
        in_dim: usize,
        out_dim: usize,
    }

    impl LinearNet {
        fn new(w: Vec<Vec<f64>>, b: Vec<f64>) -> Self {
            let out_dim = w.len();
            let in_dim = if out_dim == 0 { 0 } else { w[0].len() };
            LinearNet { w, b, in_dim, out_dim }
        }
    }

    impl NeuralNetworkLike for LinearNet {
        fn input_dim(&self) -> usize {
            self.in_dim
        }
        fn output_dim(&self) -> usize {
            self.out_dim
        }
        fn predict(&self, input: &[f64]) -> NumericVector {
            (0..self.out_dim)
                .map(|o| {
                    let mut acc = self.b[o];
                    for i in 0..self.in_dim {
                        acc += self.w[o][i] * input[i];
                    }
                    acc
                })
                .collect()
        }
        fn parameter_count(&self) -> Option<usize> {
            Some(self.out_dim * self.in_dim + self.out_dim)
        }
    }

    impl TrainableNeuralNetwork for LinearNet {
        fn train_sample(
            &mut self,
            input: &[f64],
            target: &[f64],
            learning_rate: f64,
        ) -> TrainSampleResult {
            let prediction = self.predict(input);
            let n = self.out_dim as f64;
            let mut loss = 0.0;
            let errors: Vec<f64> = (0..self.out_dim)
                .map(|o| {
                    let e = prediction[o] - target[o];
                    loss += e * e;
                    e
                })
                .collect();
            loss /= n;
            for o in 0..self.out_dim {
                let g = 2.0 / n * errors[o];
                for i in 0..self.in_dim {
                    self.w[o][i] -= learning_rate * g * input[i];
                }
                self.b[o] -= learning_rate * g;
            }
            TrainSampleResult { loss, prediction }
        }
    }

    #[test]
    fn forward_predict_matches_hand_calc() {
        let net = LinearNet::new(vec![vec![1.0, 2.0], vec![0.0, -1.0]], vec![0.5, 1.0]);
        let y = net.predict(&[3.0, 4.0]);
        // row0: 1*3 + 2*4 + 0.5 = 11.5 ; row1: 0*3 + -1*4 + 1 = -3
        assert_eq!(y, vec![11.5, -3.0]);
        assert_eq!(net.parameter_count(), Some(6));
    }

    #[test]
    fn training_reduces_loss() {
        let mut net = LinearNet::new(vec![vec![0.0, 0.0]], vec![0.0]);
        let input = [1.0, 2.0];
        let target = [5.0];
        let first = net.train_sample(&input, &target, 0.05).loss;
        let mut last = first;
        for _ in 0..200 {
            last = net.train_sample(&input, &target, 0.05).loss;
        }
        assert!(last < first, "loss should decrease: {last} !< {first}");
        assert!(last < 1e-3, "loss should converge near zero: {last}");
    }

    #[test]
    fn supervised_station_trains_and_records() {
        let net = LinearNet::new(vec![vec![0.0]], vec![0.0]);
        let mut station = SupervisedNeuralNetworkStation::new(
            "nn",
            net,
            SupervisedNeuralNetworkStationOptions::new(0.1),
        );
        assert!(!station.has_work());
        station.core_mut().take(
            Rc::new(SupervisedSampleToken::new("s0", vec![1.0], vec![2.0])),
            SupervisedNeuralNetworkStation::<LinearNet>::CH_TRAIN,
        );
        assert!(station.has_work());
        station.run_time_step();
        assert_eq!(station.get_training_step(), 1);
        assert_eq!(station.loss_history.len(), 1);
        assert!(!station.has_work());
    }
}
