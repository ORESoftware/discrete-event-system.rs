//! Port of `src/des/general/neural-network.ts` — feed-forward neural networks
//! as DES components:
//!   1. a small trainable MLP ([`FeedForwardNetwork`]) for forward/backward
//!      passes;
//!   2. supervised-learning stations that receive samples as queued tokens;
//!   3. a neural Q-learning agent for MDP/RL environments;
//!   4. a neural-ODE solver station treating a network as `dy/dt = f(t, y)`.
//!
//! ## TS → Rust mapping
//!
//!   * The string unions become enums: `ActivationName` → [`ActivationName`],
//!     `NeuralODESolverName` → [`NeuralODESolverName`].
//!   * `StateEncoder<S> = (s) => NumericVector` → boxed closure
//!     [`StateEncoder`] (`Box<dyn Fn(&S) -> NumericVector>`).
//!   * `class FeedForwardNetwork implements TrainableNeuralNetwork` → a struct
//!     `impl NeuralNetworkLike + TrainableNeuralNetwork` (both from
//!     `des_base::neural_network`).
//!   * `class SupervisedDatasetSource / NeuralODESolverStation /
//!     NeuralPredictionSink extends DESStation` → structs `impl DESStation`.
//!   * `class NeuralQLearningAgent<S> extends RLAgentStation<S, number>` → a
//!     struct embedding [`StationCore`] + [`RLAgentCore`] (`A = usize`).
//!   * The TS ODE imports (`euler/rk2Heun/rk4/rk45`) are FREE FUNCTIONS; the
//!     Rust `ode` module exposes them as `*Integrator` structs implementing
//!     [`Transform`], so the solver dispatch constructs the integrator and calls
//!     `.transform(IVP { .. })`.
//!   * RNG injection: weight init / ε-greedy take `&mut dyn RandomSource` (a
//!     seeded mulberry32), never an ambient `Math.random`.
//!   * `throw` (shape/dim/lr violations) → `panic!`.
//!
//! ### FLAGGED deviations
//!   * `runNeuralQLearningDES` returns `network` as
//!     `Rc<RefCell<Box<dyn TrainableNeuralNetwork>>>` (shared with the agent)
//!     rather than an owned model, because the trained network lives inside the
//!     agent which is itself shared through the DES graph (no `Rc::try_unwrap`).
//!   * `runSupervisedNeuralNetDES`'s optional `desOptions` override is dropped
//!     (the `IterativeRunOptions` callbacks are not clonable into a params
//!     struct); the deterministic defaults are kept.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::environment::{
    EnvironmentStation, EnvironmentStationOptions, PureEnvironment, CH_ACTION, CH_STATE,
    CH_TRANSITION,
};
use crate::des::general::des_base::neural_network::{
    Meta, MetaValue, NeuralNetworkLike, NumericVector, SupervisedNeuralNetworkStation,
    SupervisedNeuralNetworkStationOptions, SupervisedSampleToken, TrainSampleResult,
    TrainableNeuralNetwork,
};
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation, RngRef};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions, RunReason};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::ode::{
    EulerIntegrator, HeunIntegrator, ODETrace, RK45Integrator, RK45Options, RK4Integrator, IVP,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::Transform;

// =============================================================================
// RNG adapter — one shared mulberry32 stream (matches the TS single closure).
// =============================================================================

#[derive(Clone)]
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl SharedRng {
    fn new(seed: u32) -> Self {
        SharedRng(Rc::new(RefCell::new(mulberry32(seed))))
    }
}

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

// =============================================================================
// FEED-FORWARD NETWORK
// =============================================================================

/// Layer activation. (TS `'linear' | 'sigmoid' | 'tanh' | 'relu'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationName {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
}

/// One dense layer. (TS `interface DenseLayerConfig`.)
#[derive(Clone, Debug)]
pub struct DenseLayerConfig {
    /// `weights[out][in]`.
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationName,
}

/// Forward-pass cache (TS `interface ForwardTrace`).
struct ForwardTrace {
    z: Vec<Vec<f64>>,
    /// `activations[0]` is the input.
    activations: Vec<Vec<f64>>,
}

/// Spec for [`FeedForwardNetwork::random`]. (TS inline options object.)
#[derive(Clone, Debug)]
pub struct RandomNetworkSpec {
    pub input_dim: usize,
    pub hidden_layers: Vec<usize>,
    pub output_dim: usize,
    pub hidden_activation: ActivationName,
    pub output_activation: ActivationName,
    /// Override the default Xavier limit `sqrt(6 / (fan_in + fan_out))`.
    pub weight_scale: Option<f64>,
}

/// A trainable multilayer perceptron. (TS `class FeedForwardNetwork`.)
#[derive(Clone, Debug)]
pub struct FeedForwardNetwork {
    pub layers: Vec<DenseLayerConfig>,
    pub input_dim: usize,
    pub output_dim: usize,
}

fn validate_shape(layers: &[DenseLayerConfig]) {
    let mut prev_out: Option<usize> = None;
    for (k, layer) in layers.iter().enumerate() {
        if layer.biases.is_empty() {
            panic!("layer {k}: biases cannot be empty");
        }
        if layer.weights.len() != layer.biases.len() {
            panic!("layer {k}: weights rows must equal biases length");
        }
        let width = layer.weights.first().map(|r| r.len()).unwrap_or(0);
        if width < 1 {
            panic!("layer {k}: weights rows cannot be empty");
        }
        for row in &layer.weights {
            if row.len() != width {
                panic!("layer {k}: ragged weight matrix");
            }
        }
        if let Some(po) = prev_out {
            if width != po {
                panic!("layer {k}: input dim {width} does not match previous output dim {po}");
            }
        }
        prev_out = Some(layer.biases.len());
    }
}

impl FeedForwardNetwork {
    pub fn new(layers: Vec<DenseLayerConfig>) -> Self {
        if layers.is_empty() {
            panic!("FeedForwardNetwork requires at least one layer");
        }
        validate_shape(&layers);
        let input_dim = layers[0].weights[0].len();
        let output_dim = layers[layers.len() - 1].biases.len();
        FeedForwardNetwork {
            layers,
            input_dim,
            output_dim,
        }
    }

    /// Build a randomly-initialised network. (TS `FeedForwardNetwork.random`,
    /// with the RNG injected rather than defaulting to `Math.random`.)
    pub fn random(spec: &RandomNetworkSpec, rng: &mut dyn RandomSource) -> FeedForwardNetwork {
        let mut dims = vec![spec.input_dim];
        dims.extend_from_slice(&spec.hidden_layers);
        dims.push(spec.output_dim);
        let mut layers: Vec<DenseLayerConfig> = Vec::new();
        for k in 0..dims.len() - 1 {
            let fan_in = dims[k];
            let fan_out = dims[k + 1];
            let limit = spec
                .weight_scale
                .unwrap_or_else(|| (6.0 / (fan_in + fan_out) as f64).sqrt());
            let weights: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| (2.0 * rng.next_float() - 1.0) * limit)
                        .collect()
                })
                .collect();
            let activation = if k == dims.len() - 2 {
                spec.output_activation
            } else {
                spec.hidden_activation
            };
            layers.push(DenseLayerConfig {
                weights,
                biases: vec![0.0; fan_out],
                activation,
            });
        }
        FeedForwardNetwork::new(layers)
    }

    /// Deep copy of this network's layer configs. (TS `toLayerConfigs`.)
    pub fn to_layer_configs(&self) -> Vec<DenseLayerConfig> {
        self.layers.clone()
    }

    /// Mean loss over a batch. (TS `trainBatch`.)
    pub fn train_batch(
        &mut self,
        samples: &[(NumericVector, NumericVector)],
        learning_rate: f64,
    ) -> f64 {
        self.train_batch_slices(
            samples
                .iter()
                .map(|(input, target)| (input.as_slice(), target.as_slice())),
            learning_rate,
        )
    }

    /// Mean loss over a borrowed batch without cloning input/target vectors.
    pub fn train_batch_slices<'a, I>(&mut self, samples: I, learning_rate: f64) -> f64
    where
        I: IntoIterator<Item = (&'a [f64], &'a [f64])>,
    {
        let mut total = 0.0;
        let mut count = 0usize;
        for (input, target) in samples {
            total += self.train_sample(input, target, learning_rate).loss;
            count += 1;
        }
        total / (count.max(1) as f64)
    }

    /// Total number of weights + biases. (TS `parameterCount`.)
    pub fn num_parameters(&self) -> usize {
        let mut n = 0;
        for layer in &self.layers {
            n += layer.biases.len();
            for row in &layer.weights {
                n += row.len();
            }
        }
        n
    }

    /// Euclidean norm of all parameters. (TS `l2Norm`.)
    pub fn l2_norm(&self) -> f64 {
        let mut ss = 0.0;
        for layer in &self.layers {
            for &b in &layer.biases {
                ss += b * b;
            }
            for row in &layer.weights {
                for &w in row {
                    ss += w * w;
                }
            }
        }
        ss.sqrt()
    }

    /// Latent embedding of `input`: the activations of the **last hidden layer**
    /// — the representation the output layer actually reads. For a value/critic
    /// network this is a learned, lower-dimensional summary of the input, useful
    /// as a similarity key for retrieval (RAG over states/moments). A
    /// single-layer (no hidden) network has no learned latent, so it returns the
    /// input itself. The result is always finite (the forward pass asserts it).
    pub fn embed(&self, input: &[f64]) -> NumericVector {
        let trace = self.forward(input);
        // activations[0] = input; activations[layers.len()] = output. The last
        // hidden layer is activations[layers.len() - 1] (== input when len == 1).
        let hidden_index = self.layers.len().saturating_sub(1);
        trace.activations[hidden_index].clone()
    }

    /// Dimension of the [`Self::embed`] latent (width of the last hidden layer,
    /// or `input_dim` for a single-layer network).
    pub fn embedding_dim(&self) -> usize {
        if self.layers.len() <= 1 {
            self.input_dim
        } else {
            self.layers[self.layers.len() - 2].biases.len()
        }
    }

    fn forward(&self, input: &[f64]) -> ForwardTrace {
        self.assert_vector(input, self.input_dim, "input");
        let mut activations: Vec<Vec<f64>> = vec![input.to_vec()];
        let mut z: Vec<Vec<f64>> = Vec::new();
        let mut a = input.to_vec();
        for layer in &self.layers {
            let mut zk = vec![0.0; layer.biases.len()];
            let mut ak = vec![0.0; layer.biases.len()];
            for i in 0..layer.biases.len() {
                let mut zi = layer.biases[i];
                for j in 0..layer.weights[i].len() {
                    zi += layer.weights[i][j] * a[j];
                }
                zk[i] = zi;
                ak[i] = activate(layer.activation, zi);
            }
            z.push(zk);
            activations.push(ak.clone());
            a = ak;
        }
        ForwardTrace { z, activations }
    }

    fn assert_vector(&self, v: &[f64], dim: usize, name: &str) {
        if v.len() != dim {
            panic!("{name} dim {} does not match expected {dim}", v.len());
        }
        for &x in v {
            if !x.is_finite() {
                panic!("{name} contains non-finite value {x}");
            }
        }
    }
}

impl NeuralNetworkLike for FeedForwardNetwork {
    fn input_dim(&self) -> usize {
        self.input_dim
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
    fn predict(&self, input: &[f64]) -> NumericVector {
        self.forward(input).activations[self.layers.len()].clone()
    }
    fn parameter_count(&self) -> Option<usize> {
        Some(self.num_parameters())
    }
}

impl TrainableNeuralNetwork for FeedForwardNetwork {
    fn train_sample(
        &mut self,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
    ) -> TrainSampleResult {
        if learning_rate < 0.0 {
            panic!("learningRate must be non-negative, got {learning_rate}");
        }
        // A non-finite input/target is a runtime data hazard, not a structural bug:
        // drop the step (like a divergent gradient) instead of panicking in `forward`,
        // so a single NaN feature in a (correctly sized) sample can't crash a detached
        // batch-training worker or silently halt learning. A *dimension* mismatch is
        // still a programmer error and panics below via `assert_vector`. Mirrors
        // `train_sample_clipped`.
        if (!vector_is_finite(input, self.input_dim) || !vector_is_finite(target, self.output_dim))
            && input.len() == self.input_dim
            && target.len() == self.output_dim
        {
            return TrainSampleResult {
                loss: f64::NAN,
                prediction: vec![f64::NAN; self.output_dim],
            };
        }
        self.assert_vector(input, self.input_dim, "input");
        self.assert_vector(target, self.output_dim, "target");

        let trace = self.forward(input);
        let prediction = trace.activations[self.layers.len()].clone();
        let mut loss = 0.0;
        let mut d_a = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let e = prediction[i] - target[i];
            loss += 0.5 * e * e;
            d_a[i] = e;
        }

        // Finite-guarded backprop, clipping DISABLED (`max_grad_norm = 0`): a finite
        // gradient yields a weight update byte-identical to the previous plain SGD
        // step, but a non-finite gradient (e.g. an exploding update from an extreme
        // sample) now drops the step and leaves the weights untouched instead of
        // poisoning them — which would otherwise make the next `forward` assert-panic.
        self.apply_output_error_gradient(&trace, d_a, loss, learning_rate, 0.0);

        TrainSampleResult { loss, prediction }
    }
}

/// Outcome of a single [`FeedForwardNetwork::train_sample_clipped`] step.
#[derive(Clone, Copy, Debug)]
pub struct ClippedTrainResult {
    /// Pre-update squared-error loss for the sample.
    pub loss: f64,
    /// Whether the weight update was applied (false ⇒ gradient was non-finite
    /// and the step was dropped to avoid poisoning the network).
    pub applied: bool,
    /// Whether the gradient was rescaled to satisfy `max_grad_norm`.
    pub clipped: bool,
}

impl FeedForwardNetwork {
    /// Gradient-clipped, divergence-guarded SGD step.
    ///
    /// Unlike [`TrainableNeuralNetwork::train_sample`] (a faithful port that
    /// mutates weights inline as it walks the layers), this accumulates the full
    /// parameter gradient *before* applying it, which lets it:
    ///   1. **drop the update entirely** when any gradient component is
    ///      non-finite — so a single exploding sample can never permanently
    ///      poison the weights (the network is a *value head* whose output is
    ///      blended back into live decisions, so a NaN would otherwise propagate
    ///      into play); and
    ///   2. **rescale the gradient** by `max_grad_norm / ‖g‖₂` when its global
    ///      L2 norm exceeds `max_grad_norm` (≤ 0 disables clipping).
    ///
    /// Returns the pre-update loss plus whether the step was applied/clipped.
    pub fn train_sample_clipped(
        &mut self,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
        max_grad_norm: f64,
    ) -> ClippedTrainResult {
        if learning_rate < 0.0 {
            panic!("learningRate must be non-negative, got {learning_rate}");
        }
        // A non-finite input or target is a runtime data hazard, not a structural
        // bug: drop the step (like a divergent gradient) instead of panicking in
        // `forward`, so a single NaN feature can't crash a detached training
        // worker and silently halt learning. A *dimension* mismatch is still a
        // programmer error and panics below via `assert_vector`.
        if !vector_is_finite(input, self.input_dim) || !vector_is_finite(target, self.output_dim) {
            if input.len() == self.input_dim && target.len() == self.output_dim {
                return ClippedTrainResult {
                    loss: f64::NAN,
                    applied: false,
                    clipped: false,
                };
            }
        }
        self.assert_vector(input, self.input_dim, "input");
        self.assert_vector(target, self.output_dim, "target");

        let trace = self.forward(input);
        let prediction = &trace.activations[self.layers.len()];
        let mut loss = 0.0;
        let mut d_a = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let e = prediction[i] - target[i];
            loss += 0.5 * e * e;
            d_a[i] = e;
        }
        self.apply_output_error_gradient(&trace, d_a, loss, learning_rate, max_grad_norm)
    }

    /// Backprop a precomputed output gradient `d_a` (∂loss/∂output_activation)
    /// through the network and apply it with a finite guard + global grad-norm
    /// clip. Shared by [`train_sample_clipped`] (MSE error) and
    /// [`train_policy_gradient_sample`] (advantage-weighted softmax). `loss` is
    /// only reported. Returns `applied=false` (weights untouched) if any gradient
    /// component is non-finite.
    fn apply_output_error_gradient(
        &mut self,
        trace: &ForwardTrace,
        mut d_a: Vec<f64>,
        loss: f64,
        learning_rate: f64,
        max_grad_norm: f64,
    ) -> ClippedTrainResult {
        // Accumulate per-layer gradients without mutating the network yet.
        let mut weight_grads: Vec<Vec<Vec<f64>>> = self
            .layers
            .iter()
            .map(|l| l.weights.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut bias_grads: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.biases.len()])
            .collect();

        for k in (0..self.layers.len()).rev() {
            let activation = self.layers[k].activation;
            let prev_a = &trace.activations[k];
            let cur_a = &trace.activations[k + 1];
            let cur_z = &trace.z[k];
            let delta: Vec<f64> = (0..cur_a.len())
                .map(|i| d_a[i] * activation_prime_from_output(activation, cur_a[i], cur_z[i]))
                .collect();

            let weights = &self.layers[k].weights;
            let mut d_prev = vec![0.0; prev_a.len()];
            for i in 0..weights.len() {
                for j in 0..weights[i].len() {
                    d_prev[j] += weights[i][j] * delta[i];
                    weight_grads[k][i][j] = delta[i] * prev_a[j];
                }
                bias_grads[k][i] = delta[i];
            }
            d_a = d_prev;
        }

        // Global gradient L2 norm with a finite guard over every component.
        let mut sum_sq = 0.0;
        let mut finite = loss.is_finite();
        if finite {
            'scan: for k in 0..self.layers.len() {
                for row in &weight_grads[k] {
                    for &g in row {
                        if !g.is_finite() {
                            finite = false;
                            break 'scan;
                        }
                        sum_sq += g * g;
                    }
                }
                for &g in &bias_grads[k] {
                    if !g.is_finite() {
                        finite = false;
                        break 'scan;
                    }
                    sum_sq += g * g;
                }
            }
        }
        if !finite {
            // Divergent gradient — drop the step, leave the weights untouched.
            return ClippedTrainResult {
                loss,
                applied: false,
                clipped: false,
            };
        }

        let grad_norm = sum_sq.sqrt();
        let scale = if max_grad_norm > 0.0 && grad_norm > max_grad_norm {
            max_grad_norm / grad_norm
        } else {
            1.0
        };
        let step = learning_rate * scale;
        for k in 0..self.layers.len() {
            let layer = &mut self.layers[k];
            for i in 0..layer.weights.len() {
                for j in 0..layer.weights[i].len() {
                    layer.weights[i][j] -= step * weight_grads[k][i][j];
                }
                layer.biases[i] -= step * bias_grads[k][i];
            }
        }
        ClippedTrainResult {
            loss,
            applied: true,
            clipped: scale < 1.0,
        }
    }

    /// One advantage-weighted policy-gradient (REINFORCE-with-baseline) step for a
    /// softmax policy. The network's outputs are treated as **logits** (the output
    /// layer must be `Linear`); softmax is applied here. Minimises the surrogate
    ///   `loss = -advantage·log π(action) − entropy_coeff·H(π)`,
    /// whose logit gradient is
    ///   `advantage·(softmaxᵢ − onehotᵢ) + entropy_coeff·pᵢ·(ln pᵢ + H)`.
    /// A positive advantage pushes probability toward `action`; the entropy term
    /// keeps the policy from collapsing too early. Same finite-guard + grad-norm
    /// clip as [`train_sample_clipped`], so a bad step never poisons the actor.
    pub fn train_policy_gradient_sample(
        &mut self,
        input: &[f64],
        action: usize,
        advantage: f64,
        entropy_coeff: f64,
        learning_rate: f64,
        max_grad_norm: f64,
    ) -> ClippedTrainResult {
        if learning_rate < 0.0 {
            panic!("learningRate must be non-negative, got {learning_rate}");
        }
        assert!(
            action < self.output_dim,
            "policy action index {action} out of range for {} outputs",
            self.output_dim
        );
        // The outputs are treated as logits — softmax is applied here, so the
        // output layer must be Linear or the gradient is wrong. Enforce the
        // contract (debug builds / tests) rather than silently miscomputing.
        debug_assert_eq!(
            self.layers.last().map(|layer| layer.activation),
            Some(ActivationName::Linear),
            "train_policy_gradient_sample requires a Linear output layer"
        );
        // A non-finite advantage or input is nothing to learn from — skip the
        // step cleanly rather than panicking in `forward`, so a degenerate
        // sample can't crash the actor's (possibly detached) trainer. A
        // dimension mismatch is a structural bug and still panics below.
        if !advantage.is_finite() || !vector_is_finite(input, self.input_dim) {
            if input.len() == self.input_dim {
                return ClippedTrainResult {
                    loss: 0.0,
                    applied: false,
                    clipped: false,
                };
            }
        }
        self.assert_vector(input, self.input_dim, "input");

        let trace = self.forward(input);
        let logits = &trace.activations[self.layers.len()];
        let probs = softmax(logits);
        let entropy: f64 = -probs
            .iter()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f64>();
        let log_pi_action = probs[action].max(1e-12).ln();
        let loss = -advantage * log_pi_action - entropy_coeff * entropy;

        let mut d_a = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let onehot = if i == action { 1.0 } else { 0.0 };
            let policy_grad = advantage * (probs[i] - onehot);
            let entropy_grad = entropy_coeff * probs[i] * (probs[i].max(1e-12).ln() + entropy);
            d_a[i] = policy_grad + entropy_grad;
        }
        self.apply_output_error_gradient(&trace, d_a, loss, learning_rate, max_grad_norm)
    }

    /// Softmax distribution over the network's current outputs (logits). Returns a
    /// uniform distribution if the logits are degenerate (non-finite sum).
    pub fn action_probabilities(&self, input: &[f64]) -> NumericVector {
        softmax(&self.predict(input))
    }

    /// Mean loss over a borrowed batch trained with [`train_sample_clipped`].
    /// Samples whose gradient is non-finite are skipped (no update) and excluded
    /// from the reported mean, so a poisoned sample neither corrupts the network
    /// nor silently inflates the loss average.
    pub fn train_batch_slices_clipped<'a, I>(
        &mut self,
        samples: I,
        learning_rate: f64,
        max_grad_norm: f64,
    ) -> f64
    where
        I: IntoIterator<Item = (&'a [f64], &'a [f64])>,
    {
        let mut total = 0.0;
        let mut count = 0usize;
        for (input, target) in samples {
            let result = self.train_sample_clipped(input, target, learning_rate, max_grad_norm);
            if result.applied && result.loss.is_finite() {
                total += result.loss;
                count += 1;
            }
        }
        total / (count.max(1) as f64)
    }

    /// Compat shim for the soccer-engine `main` momentum-optimizer API.
    ///
    /// WARNING: momentum is NOT implemented here. `_momentum` and `_state` are
    /// accepted and ignored, so this performs plain (momentum-free) SGD by
    /// delegating to `train_batch_slices_clipped`. The soccer engine's default
    /// `optimizer_momentum` is 0.0, so default runs are unaffected — but any run
    /// configured with a nonzero momentum silently gets plain SGD, not the
    /// heavy-ball update it asks for. To honor it, store per-parameter velocity
    /// buffers in `FeedForwardMomentumState` and apply them here.
    pub fn train_batch_slices_clipped_with_momentum<'a, I>(
        &mut self,
        samples: I,
        learning_rate: f64,
        max_grad_norm: f64,
        _momentum: f64,
        _state: &mut FeedForwardMomentumState,
    ) -> f64
    where
        I: IntoIterator<Item = (&'a [f64], &'a [f64])>,
    {
        self.train_batch_slices_clipped(samples, learning_rate, max_grad_norm)
    }
}

/// Compat placeholder for the soccer-engine `main` per-network momentum state.
///
/// Currently EMPTY: momentum is not implemented, so no velocity is stored and
/// threading `&mut FeedForwardMomentumState` through the trainer is a no-op. It
/// exists only so the engine compiles/links against this des. To implement
/// heavy-ball momentum, hold per-parameter velocity buffers here and consume
/// them in [`FeedForwardNetwork::train_batch_slices_clipped_with_momentum`].
#[derive(Clone, Debug, Default)]
pub struct FeedForwardMomentumState;

/// Whether `v` has the expected length and every component is finite. Used by
/// the divergence-guarded trainers to drop a step on non-finite **data** (a
/// runtime hazard) rather than panic inside [`FeedForwardNetwork::forward`].
fn vector_is_finite(v: &[f64], dim: usize) -> bool {
    v.len() == dim && v.iter().all(|x| x.is_finite())
}

fn activate(name: ActivationName, z: f64) -> f64 {
    match name {
        ActivationName::Linear => z,
        ActivationName::Sigmoid => 1.0 / (1.0 + (-z).exp()),
        ActivationName::Tanh => z.tanh(),
        ActivationName::Relu => {
            if z > 0.0 {
                z
            } else {
                0.0
            }
        }
    }
}

/// Numerically-stable softmax. Falls back to a uniform distribution if the
/// logits are degenerate (empty or non-finite sum).
fn softmax(logits: &[f64]) -> NumericVector {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    let exps: NumericVector = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f64; logits.len()]
    }
}

fn activation_prime_from_output(name: ActivationName, a: f64, z: f64) -> f64 {
    match name {
        ActivationName::Linear => 1.0,
        ActivationName::Sigmoid => a * (1.0 - a),
        ActivationName::Tanh => 1.0 - a * a,
        ActivationName::Relu => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

// =============================================================================
// SUPERVISED LEARNING OVER DES QUEUES
// =============================================================================

/// One (input, target) training pair. (TS `interface SupervisedSample`.)
#[derive(Clone, Debug)]
pub struct SupervisedSample {
    pub input: NumericVector,
    pub target: NumericVector,
}

/// Source station that streams a dataset as [`SupervisedSampleToken`]s over a
/// number of epochs. (TS `class SupervisedDatasetSource`.)
pub struct SupervisedDatasetSource {
    core: StationCore,
    dataset: Vec<SupervisedSample>,
    epochs: usize,
    samples_per_tick: usize,
    shuffle_each_epoch: bool,
    rng: Box<dyn RandomSource>,
    epoch: usize,
    cursor: usize,
    emitted: usize,
    order: Vec<usize>,
}

impl SupervisedDatasetSource {
    pub const CH_TRAIN: &'static str = "train";

    pub fn new(
        id: impl Into<String>,
        dataset: Vec<SupervisedSample>,
        epochs: usize,
        samples_per_tick: usize,
        shuffle_each_epoch: bool,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        if dataset.is_empty() {
            panic!("SupervisedDatasetSource requires at least one sample");
        }
        let order: Vec<usize> = (0..dataset.len()).collect();
        let mut src = SupervisedDatasetSource {
            core: StationCore::new(id),
            dataset,
            epochs,
            samples_per_tick: samples_per_tick.max(1),
            shuffle_each_epoch,
            rng,
            epoch: 0,
            cursor: 0,
            emitted: 0,
            order,
        };
        if src.shuffle_each_epoch {
            src.shuffle_order();
        }
        src
    }

    pub fn get_emitted_count(&self) -> usize {
        self.emitted
    }

    /// Returns `(index, epoch)` of the next sample to emit, or `None` when the
    /// epoch budget is exhausted.
    fn next_sample(&mut self) -> Option<(usize, usize)> {
        if self.epoch >= self.epochs {
            return None;
        }
        if self.cursor >= self.dataset.len() {
            self.epoch += 1;
            self.cursor = 0;
            if self.epoch >= self.epochs {
                return None;
            }
            if self.shuffle_each_epoch {
                self.shuffle_order();
            }
        }
        let index = self.order[self.cursor];
        self.cursor += 1;
        Some((index, self.epoch))
    }

    fn shuffle_order(&mut self) {
        for i in (1..self.order.len()).rev() {
            let j = (self.rng.next_float() * (i as f64 + 1.0)).floor() as usize;
            self.order.swap(i, j);
        }
    }
}

impl DESStation for SupervisedDatasetSource {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.epoch < self.epochs
    }
    fn run_time_step(&mut self) {
        let n = self.samples_per_tick;
        for _ in 0..n {
            let Some((index, epoch)) = self.next_sample() else {
                return;
            };
            let sample = &self.dataset[index];
            let mut meta = Meta::new();
            meta.insert("epoch".to_string(), MetaValue::Number(epoch as f64));
            meta.insert("index".to_string(), MetaValue::Number(index as f64));
            let token = SupervisedSampleToken {
                id: format!("sample-{}", self.emitted),
                input: sample.input.clone(),
                target: sample.target.clone(),
                meta,
            };
            self.core.emit(Rc::new(token), Self::CH_TRAIN);
            self.emitted += 1;
        }
    }
}

/// Result of a supervised DES run. (TS `interface SupervisedNeuralNetDESResult`.)
#[derive(Clone, Debug)]
pub struct SupervisedNeuralNetDESResult<N> {
    pub network: N,
    pub loss_history: Vec<f64>,
    pub predictions: Vec<NumericVector>,
    pub ticks: usize,
    pub reason: Option<RunReason>,
}

/// Parameters for [`run_supervised_neural_net_des`]. (TS inline options object;
/// the `desOptions` override is dropped — see module docs.)
pub struct SupervisedRunParams<N> {
    pub network: N,
    pub dataset: Vec<SupervisedSample>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: Option<u32>,
    pub samples_per_tick: Option<usize>,
    pub shuffle_each_epoch: Option<bool>,
    pub snapshot_every: Option<usize>,
}

/// Train a network on a dataset via a DES source→trainer pipeline. (TS
/// `runSupervisedNeuralNetDES`.)
pub fn run_supervised_neural_net_des<N: TrainableNeuralNetwork + Clone + 'static>(
    params: SupervisedRunParams<N>,
) -> SupervisedNeuralNetDESResult<N> {
    let seed = params.seed.unwrap_or(1);
    let samples_per_tick = params.samples_per_tick.unwrap_or(1).max(1);
    let dataset = params.dataset;

    let source = Rc::new(RefCell::new(SupervisedDatasetSource::new(
        "dataset",
        dataset.clone(),
        params.epochs,
        samples_per_tick,
        params.shuffle_each_epoch.unwrap_or(false),
        Box::new(mulberry32(seed)),
    )));
    let trainer = Rc::new(RefCell::new(SupervisedNeuralNetworkStation::new(
        "nn",
        params.network,
        SupervisedNeuralNetworkStationOptions {
            learning_rate: params.learning_rate,
            snapshot_every: params.snapshot_every.unwrap_or(0),
        },
    )));

    source.borrow_mut().core_mut().pipe(
        trainer.clone() as StationRef,
        SupervisedDatasetSource::CH_TRAIN,
        SupervisedNeuralNetworkStation::<N>::CH_TRAIN,
    );

    let ticks_per_epoch = dataset.len().div_ceil(samples_per_tick);
    let max_ticks = params.epochs * ticks_per_epoch + 1000;
    let summary = run_iterative_des(
        vec![source as StationRef, trainer.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            ..Default::default()
        },
    );

    let trainer_ref = trainer.borrow();
    let network = trainer_ref.get_network().clone();
    let loss_history = trainer_ref.loss_history.clone();
    let predictions: Vec<NumericVector> =
        dataset.iter().map(|s| network.predict(&s.input)).collect();

    SupervisedNeuralNetDESResult {
        network,
        loss_history,
        predictions,
        ticks: summary.ticks,
        reason: summary.reason,
    }
}

/// The canonical XOR truth table. (TS `XOR_DATASET`.)
pub fn xor_dataset() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample {
            input: vec![0.0, 0.0],
            target: vec![0.0],
        },
        SupervisedSample {
            input: vec![0.0, 1.0],
            target: vec![1.0],
        },
        SupervisedSample {
            input: vec![1.0, 0.0],
            target: vec![1.0],
        },
        SupervisedSample {
            input: vec![1.0, 1.0],
            target: vec![0.0],
        },
    ]
}

/// Options for [`run_xor_neural_net_des`]. (TS `interface XorNeuralNetOptions`.)
#[derive(Clone, Debug, Default)]
pub struct XorNeuralNetOptions {
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub seed: Option<u32>,
    pub hidden_layers: Option<Vec<usize>>,
    pub samples_per_tick: Option<usize>,
    pub shuffle_each_epoch: Option<bool>,
}

/// Train a small MLP to learn XOR. (TS `runXorNeuralNetDES`.)
pub fn run_xor_neural_net_des(
    opts: XorNeuralNetOptions,
) -> SupervisedNeuralNetDESResult<FeedForwardNetwork> {
    let seed = opts.seed.unwrap_or(7);
    let mut rng = mulberry32(seed);
    let network = FeedForwardNetwork::random(
        &RandomNetworkSpec {
            input_dim: 2,
            hidden_layers: opts.hidden_layers.clone().unwrap_or_else(|| vec![4]),
            output_dim: 1,
            hidden_activation: ActivationName::Tanh,
            output_activation: ActivationName::Sigmoid,
            weight_scale: None,
        },
        &mut rng,
    );
    run_supervised_neural_net_des(SupervisedRunParams {
        network,
        dataset: xor_dataset(),
        epochs: opts.epochs.unwrap_or(8000),
        learning_rate: opts.learning_rate.unwrap_or(0.3),
        seed: Some(seed),
        samples_per_tick: Some(opts.samples_per_tick.unwrap_or(1)),
        shuffle_each_epoch: Some(opts.shuffle_each_epoch.unwrap_or(false)),
        snapshot_every: None,
    })
}

// =============================================================================
// NEURAL Q-LEARNING
// =============================================================================

/// Maps a typed state into the network's input vector. (TS `StateEncoder<S>`.)
pub type StateEncoder<S> = Box<dyn Fn(&S) -> NumericVector>;

/// Q-learning hyperparameters (sans RNG, which is injected separately into the
/// agent core). (TS `interface NeuralQLearningOptions`.)
pub struct NeuralQLearningParams<S> {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub num_actions: usize,
    pub state_encoder: StateEncoder<S>,
}

/// An ε-greedy Q-learning agent backed by a neural function approximator. (TS
/// `class NeuralQLearningAgent<S>`.)
pub struct NeuralQLearningAgent<S: Clone + 'static = f64> {
    core: StationCore,
    agent: RLAgentCore,
    network: Rc<RefCell<Box<dyn TrainableNeuralNetwork>>>,
    pub loss_history: Vec<f64>,
    pub td_error_history: Vec<f64>,
    current_epsilon: f64,
    opts: NeuralQLearningParams<S>,
}

impl<S: Clone + 'static> NeuralQLearningAgent<S> {
    pub fn new(
        id: impl Into<String>,
        network: Rc<RefCell<Box<dyn TrainableNeuralNetwork>>>,
        opts: NeuralQLearningParams<S>,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        let current_epsilon = opts.epsilon;
        NeuralQLearningAgent {
            core: StationCore::new(id),
            agent: RLAgentCore::new(rng),
            network,
            loss_history: Vec::new(),
            td_error_history: Vec::new(),
            current_epsilon,
            opts,
        }
    }

    pub fn predict_q(&self, state: &S) -> NumericVector {
        self.network
            .borrow()
            .predict(&(self.opts.state_encoder)(state))
    }

    pub fn get_epsilon(&self) -> f64 {
        self.current_epsilon
    }

    pub fn get_network(&self) -> Rc<RefCell<Box<dyn TrainableNeuralNetwork>>> {
        self.network.clone()
    }
}

impl<S: Clone + 'static> DESStation for NeuralQLearningAgent<S> {
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
        self.rl_agent_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.rl_agent_has_work()
    }
    fn assert_preconditions(&mut self) {
        let out = self.network.borrow().output_dim();
        if out != self.opts.num_actions {
            panic!(
                "network outputDim {out} must equal numActions {}",
                self.opts.num_actions
            );
        }
    }
}

impl<S: Clone + 'static> RLAgentStation<S, usize> for NeuralQLearningAgent<S> {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }

    fn pick_action(&self, state: &S, rng: &mut dyn RandomSource) -> usize {
        if rng.next_float() < self.current_epsilon {
            return (rng.next_float() * self.opts.num_actions as f64).floor() as usize;
        }
        let q = self
            .network
            .borrow()
            .predict(&(self.opts.state_encoder)(state));
        arg_max_with_tie_break(&q, &mut RngRef(rng), ARGMAX_EPS_DEFAULT).unwrap_or(0)
    }

    fn update(&mut self, state: &S, action: &usize, reward: f64, next_state: &S, done: bool) {
        let x = (self.opts.state_encoder)(state);
        let q = self.network.borrow().predict(&x);
        let old_q = q[*action];
        let mut target = q;
        let add = if done {
            0.0
        } else {
            let q_next = self
                .network
                .borrow()
                .predict(&(self.opts.state_encoder)(next_state));
            self.opts.gamma * q_next.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        target[*action] = reward + add;
        let r = self
            .network
            .borrow_mut()
            .train_sample(&x, &target, self.opts.alpha);
        self.loss_history.push(r.loss);
        self.td_error_history.push(target[*action] - old_q);
    }

    fn end_of_episode(&mut self, _episode_id: f64) {
        if let Some(decay) = self.opts.epsilon_decay {
            self.current_epsilon = self
                .opts
                .epsilon_min
                .unwrap_or(0.0)
                .max(self.current_epsilon * decay);
        }
    }
}

/// One-hot encoder over `num_states` integer states. (TS `oneHotEncoder`.)
pub fn one_hot_encoder(num_states: usize) -> StateEncoder<f64> {
    Box::new(move |state: &f64| {
        let s = *state;
        if s < 0.0 || s >= num_states as f64 || s.fract() != 0.0 {
            panic!("state {s} is outside [0, {num_states})");
        }
        let mut x = vec![0.0; num_states];
        x[s as usize] = 1.0;
        x
    })
}

/// Result of [`run_neural_q_learning_des`]. (TS `interface NeuralQLearningResult`;
/// `network` is shared via `Rc` — see module docs.)
#[derive(Clone)]
pub struct NeuralQLearningResult {
    pub network: Rc<RefCell<Box<dyn TrainableNeuralNetwork>>>,
    pub policy: Vec<usize>,
    pub q_values: Vec<Vec<f64>>,
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub loss_history: Vec<f64>,
    pub td_error_history: Vec<f64>,
    pub total_episodes: usize,
    pub total_steps: u64,
    pub total_ticks: usize,
}

/// Parameters for [`run_neural_q_learning_des`]. (TS inline options object; the
/// `desOptions` override is dropped.)
pub struct NeuralQLearningRunParams {
    pub num_episodes: usize,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub max_steps_per_episode: Option<usize>,
    pub seed: Option<u32>,
    pub network: Option<Box<dyn TrainableNeuralNetwork>>,
    pub hidden_layers: Option<Vec<usize>>,
    pub hidden_activation: Option<ActivationName>,
    pub state_encoder: Option<StateEncoder<f64>>,
}

/// Train a neural Q-learning agent against an environment via a DES loop. (TS
/// `runNeuralQLearningDES`.)
pub fn run_neural_q_learning_des(
    env: Box<dyn PureEnvironment<f64, usize>>,
    params: NeuralQLearningRunParams,
) -> NeuralQLearningResult {
    let seed = params.seed.unwrap_or(1);
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let rng = SharedRng::new(seed);

    let encoder: StateEncoder<f64> = params
        .state_encoder
        .unwrap_or_else(|| one_hot_encoder(num_states));

    let net_box: Box<dyn TrainableNeuralNetwork> = match params.network {
        Some(n) => n,
        None => {
            let mut init_rng = rng.clone();
            Box::new(FeedForwardNetwork::random(
                &RandomNetworkSpec {
                    input_dim: num_states,
                    hidden_layers: params.hidden_layers.clone().unwrap_or_default(),
                    output_dim: num_actions,
                    hidden_activation: params.hidden_activation.unwrap_or(ActivationName::Tanh),
                    output_activation: ActivationName::Linear,
                    weight_scale: Some(0.01),
                },
                &mut init_rng,
            ))
        }
    };
    let network = Rc::new(RefCell::new(net_box));

    let agent = Rc::new(RefCell::new(NeuralQLearningAgent::<f64>::new(
        "neural-q-agent",
        network.clone(),
        NeuralQLearningParams {
            alpha: params.alpha,
            gamma: params.gamma,
            epsilon: params.epsilon,
            epsilon_min: params.epsilon_min,
            epsilon_decay: params.epsilon_decay,
            num_actions,
            state_encoder: encoder,
        },
        Box::new(rng.clone()),
    )));
    let env_st = Rc::new(RefCell::new(EnvironmentStation::<f64, usize>::new(
        "env",
        env,
        EnvironmentStationOptions {
            num_episodes: Some(params.num_episodes as f64),
            max_steps_per_episode: params.max_steps_per_episode,
        },
    )));

    env_st
        .borrow_mut()
        .core_mut()
        .pipe(agent.clone() as StationRef, CH_STATE, CH_STATE);
    env_st
        .borrow_mut()
        .core_mut()
        .pipe(agent.clone() as StationRef, CH_TRANSITION, CH_TRANSITION);
    agent
        .borrow_mut()
        .core_mut()
        .pipe(env_st.clone() as StationRef, CH_ACTION, CH_ACTION);

    let summary = run_iterative_des(
        vec![env_st as StationRef, agent.clone() as StationRef],
        IterativeRunOptions::default(),
    );

    let agent_ref = agent.borrow();
    let q_values: Vec<Vec<f64>> = (0..num_states)
        .map(|s| agent_ref.predict_q(&(s as f64)))
        .collect();
    let policy: Vec<usize> = q_values
        .iter()
        .map(|row| arg_max_with_tie_break(row, &mut rng.clone(), ARGMAX_EPS_DEFAULT).unwrap_or(0))
        .collect();

    NeuralQLearningResult {
        network: network.clone(),
        policy,
        q_values,
        reward_history: agent_ref.reward_history().to_vec(),
        length_history: agent_ref.length_history().to_vec(),
        loss_history: agent_ref.loss_history.clone(),
        td_error_history: agent_ref.td_error_history.clone(),
        total_episodes: agent_ref.reward_history().len(),
        total_steps: agent_ref.total_steps(),
        total_ticks: summary.ticks,
    }
}

// =============================================================================
// NEURAL ODE
// =============================================================================

/// Fixed/adaptive solver selector. (TS `'euler' | 'heun' | 'rk4' | 'rk45'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NeuralODESolverName {
    Euler,
    Heun,
    Rk4,
    Rk45,
}

/// Adaptive-solver tuning. (TS inline `rk45` sub-options.)
#[derive(Clone, Debug, Default)]
pub struct RK45Tuning {
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub h_init: Option<f64>,
    pub h_min: Option<f64>,
    pub h_max: Option<f64>,
    pub max_steps: Option<usize>,
}

/// Options for [`solve_neural_ode`]. (TS `interface NeuralODEOptions`.)
#[derive(Clone, Debug)]
pub struct NeuralODEOptions {
    pub y0: NumericVector,
    pub t0: f64,
    pub t1: f64,
    pub dt: f64,
    pub solver: Option<NeuralODESolverName>,
    pub include_time: Option<bool>,
    pub rk45: Option<RK45Tuning>,
}

/// Integrate `dy/dt = network(t, y)` (or `network(y)` when `include_time` is
/// false). (TS `solveNeuralODE`.)
pub fn solve_neural_ode(network: &dyn NeuralNetworkLike, opts: &NeuralODEOptions) -> ODETrace {
    let include_time = opts.include_time.unwrap_or(false);
    let input_dim = if include_time {
        opts.y0.len() + 1
    } else {
        opts.y0.len()
    };
    if network.input_dim() != input_dim {
        panic!(
            "neural ODE network inputDim {} must equal {input_dim}",
            network.input_dim()
        );
    }
    if network.output_dim() != opts.y0.len() {
        panic!(
            "neural ODE network outputDim {} must equal state dim {}",
            network.output_dim(),
            opts.y0.len()
        );
    }
    let rhs = move |t: f64, y: &[f64]| -> Vec<f64> {
        if include_time {
            let mut input = Vec::with_capacity(y.len() + 1);
            input.push(t);
            input.extend_from_slice(y);
            network.predict(&input)
        } else {
            network.predict(y)
        }
    };
    let y0 = opts.y0.clone();
    let (t0, t1, dt) = (opts.t0, opts.t1, opts.dt);
    match opts.solver.unwrap_or(NeuralODESolverName::Rk4) {
        NeuralODESolverName::Euler => {
            EulerIntegrator::new(dt).transform(IVP { f: rhs, y0, t0, t1 })
        }
        NeuralODESolverName::Heun => HeunIntegrator::new(dt).transform(IVP { f: rhs, y0, t0, t1 }),
        NeuralODESolverName::Rk4 => RK4Integrator::new(dt).transform(IVP { f: rhs, y0, t0, t1 }),
        NeuralODESolverName::Rk45 => {
            let tuning = opts.rk45.clone().unwrap_or_default();
            let rk_opts = RK45Options {
                rtol: tuning.rtol,
                atol: tuning.atol,
                h_init: tuning.h_init.or(Some(dt)),
                h_min: tuning.h_min,
                h_max: tuning.h_max.or(Some(dt)),
                max_steps: tuning.max_steps,
            };
            RK45Integrator::new(rk_opts).transform(IVP { f: rhs, y0, t0, t1 })
        }
    }
}

/// Token requesting an ODE solve. (TS `class NeuralODESolveToken`.)
#[derive(Clone, Debug)]
pub struct NeuralODESolveToken {
    pub id: String,
    pub options: NeuralODEOptions,
}

/// Token carrying an ODE solution trace. (TS `class NeuralODESolutionToken`.)
#[derive(Clone, Debug)]
pub struct NeuralODESolutionToken {
    pub id: String,
    pub trace: ODETrace,
}

/// Station that solves queued neural-ODE requests. (TS `class
/// NeuralODESolverStation`.)
pub struct NeuralODESolverStation<N: NeuralNetworkLike + 'static> {
    core: StationCore,
    network: N,
}

impl<N: NeuralNetworkLike + 'static> NeuralODESolverStation<N> {
    pub const CH_SOLVE: &'static str = "solve";
    pub const CH_SOLUTION: &'static str = "solution";

    pub fn new(id: impl Into<String>, network: N) -> Self {
        NeuralODESolverStation {
            core: StationCore::new(id),
            network,
        }
    }
}

impl<N: NeuralNetworkLike + 'static> DESStation for NeuralODESolverStation<N> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_SOLVE) > 0
    }
    fn run_time_step(&mut self) {
        let requests = self.core.drain::<NeuralODESolveToken>(Self::CH_SOLVE);
        for req in requests {
            let trace = solve_neural_ode(&self.network as &dyn NeuralNetworkLike, &req.options);
            let token = NeuralODESolutionToken {
                id: req.id.clone(),
                trace,
            };
            self.core.emit(Rc::new(token), Self::CH_SOLUTION);
        }
    }
}

/// Sink station collecting neural prediction tokens. (TS `class
/// NeuralPredictionSink`.)
pub struct NeuralPredictionSink {
    core: StationCore,
    pub predictions: Vec<crate::des::general::des_base::neural_network::NeuralPredictionToken>,
}

impl NeuralPredictionSink {
    /// Matches `NeuralNetworkStation::CH_PREDICTION`.
    pub const CH_PREDICTION: &'static str = "prediction";

    pub fn new(id: impl Into<String>) -> Self {
        NeuralPredictionSink {
            core: StationCore::new(id),
            predictions: Vec::new(),
        }
    }
}

impl DESStation for NeuralPredictionSink {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_PREDICTION) > 0
    }
    fn run_time_step(&mut self) {
        let toks = self
            .core
            .drain::<crate::des::general::des_base::neural_network::NeuralPredictionToken>(
                Self::CH_PREDICTION,
            );
        for t in toks {
            self.predictions.push((*t).clone());
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Neural-network smoke tests with fixed seeds. A hand-checked forward pass
    //! verifies the dense linear arithmetic; a tiny 1-D linear regression
    //! (`y = 2x + 1`) is driven near zero loss through the supervised DES
    //! pipeline; and the classic XOR truth table is learned to correct
    //! classification by a 2-4-1 tanh/sigmoid MLP.

    use super::*;

    #[test]
    fn policy_gradient_raises_probability_of_a_positive_advantage_action() {
        // 3-action softmax policy over a 2-d state. Repeatedly reinforce action 2
        // with a positive advantage from a fixed state; its probability must rise.
        let mut policy = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
            biases: vec![0.0, 0.0, 0.0],
            activation: ActivationName::Linear, // outputs are logits
        }]);
        let state = [0.5, -0.25];
        let before = policy.action_probabilities(&state)[2];
        for _ in 0..200 {
            let r = policy.train_policy_gradient_sample(&state, 2, 1.0, 0.0, 0.1, 5.0);
            assert!(r.applied, "finite advantage step should apply");
        }
        let after = policy.action_probabilities(&state)[2];
        assert!(
            after > before + 0.2,
            "reinforced action prob should rise markedly: before={before}, after={after}"
        );
        // Distribution stays normalised and finite.
        let probs = policy.action_probabilities(&state);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9 && probs.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn policy_gradient_negative_advantage_lowers_probability() {
        // A negative advantage on an action should push probability away from it.
        let mut policy = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.0], vec![0.0]],
            biases: vec![0.0, 0.0],
            activation: ActivationName::Linear,
        }]);
        let state = [1.0];
        let before = policy.action_probabilities(&state)[0];
        for _ in 0..150 {
            policy.train_policy_gradient_sample(&state, 0, -1.0, 0.0, 0.1, 5.0);
        }
        let after = policy.action_probabilities(&state)[0];
        assert!(
            after < before - 0.15,
            "penalised action prob should fall: before={before}, after={after}"
        );
    }

    #[test]
    fn clipped_step_bounds_parameter_update_under_a_huge_target() {
        // A single linear unit y = w·x + b. With an enormous target the raw error
        // gradient would yank the weights a long way in one step; the global
        // gradient-norm clip must bound the parameter delta to lr · max_grad_norm.
        let layer = DenseLayerConfig {
            weights: vec![vec![0.2, -0.1]],
            biases: vec![0.05],
            activation: ActivationName::Linear,
        };
        let mut net = FeedForwardNetwork::new(vec![layer.clone()]);
        let before: Vec<f64> = net.layers[0].weights[0]
            .iter()
            .copied()
            .chain(net.layers[0].biases.iter().copied())
            .collect();

        let lr = 0.1;
        let max_grad_norm = 1.0;
        let result = net.train_sample_clipped(&[1.0, 1.0], &[1.0e9], lr, max_grad_norm);
        assert!(result.applied, "a finite-gradient step must be applied");
        assert!(result.clipped, "a huge target must trip the clip");

        let after: Vec<f64> = net.layers[0].weights[0]
            .iter()
            .copied()
            .chain(net.layers[0].biases.iter().copied())
            .collect();
        let step_norm = before
            .iter()
            .zip(after.iter())
            .map(|(b, a)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        // ‖Δθ‖ = lr · ‖clipped grad‖ ≤ lr · max_grad_norm (+ fp slack).
        assert!(
            step_norm <= lr * max_grad_norm + 1e-9,
            "clipped step norm {step_norm} exceeded lr·max_grad_norm {}",
            lr * max_grad_norm
        );
        assert!(after.iter().all(|v| v.is_finite()), "weights stay finite");
    }

    #[test]
    fn clipped_batch_skips_diverged_steps_without_poisoning_weights() {
        // Drive a non-finite gradient by letting weights blow up first, then feed a
        // batch through the clipped trainer: the guard must keep every weight
        // finite (poisoned steps are dropped, not applied).
        let mut net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![f64::MAX, f64::MAX]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        // f64::MAX inputs would overflow to non-finite z; the guard drops the step.
        let samples: Vec<(&[f64], &[f64])> = vec![(&[f64::MAX, f64::MAX][..], &[0.0][..])];
        let _ = net.train_batch_slices_clipped(samples.into_iter(), 0.1, 4.0);
        assert!(
            net.layers[0]
                .weights
                .iter()
                .flatten()
                .chain(net.layers[0].biases.iter())
                .all(|v| v.is_finite()),
            "non-finite gradient must not poison the weights"
        );
    }

    #[test]
    fn clipped_step_drops_non_finite_input_or_target_without_panicking() {
        // A NaN/Inf in the data (not the weights) must drop the step rather than
        // panic in `forward` — otherwise a single bad feature crashes a detached
        // training worker and silently halts learning.
        let mut net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.3, -0.2]],
            biases: vec![0.1],
            activation: ActivationName::Linear,
        }]);
        let before = net.layers[0].weights[0].clone();

        let nan_input = net.train_sample_clipped(&[f64::NAN, 1.0], &[0.5], 0.1, 4.0);
        assert!(!nan_input.applied, "NaN input must drop the step");
        let inf_target = net.train_sample_clipped(&[1.0, 1.0], &[f64::INFINITY], 0.1, 4.0);
        assert!(!inf_target.applied, "Inf target must drop the step");

        assert_eq!(
            net.layers[0].weights[0], before,
            "dropped steps must leave weights untouched"
        );
    }

    #[test]
    fn unclipped_batch_train_is_finite_guarded_too() {
        // `train_batch_slices` (the UN-clipped batch path the soccer learner uses via
        // its overnight self-play training) must be finite-guarded as well: neither a
        // diverging gradient nor a non-finite sample may write NaN/Inf into the weights
        // (which would assert-panic the next `forward`).
        // (a) exploding gradient (huge weights × huge input) → step dropped:
        let mut net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![f64::MAX, f64::MAX]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        let blow: Vec<(&[f64], &[f64])> = vec![(&[f64::MAX, f64::MAX][..], &[0.0][..])];
        let _ = net.train_batch_slices(blow.into_iter(), 0.1);
        assert!(
            net.layers[0]
                .weights
                .iter()
                .flatten()
                .chain(net.layers[0].biases.iter())
                .all(|v| v.is_finite()),
            "diverged gradient must not poison weights via train_batch_slices"
        );

        // (b) non-finite sample on a healthy net → step dropped, weights untouched:
        let mut net2 = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.3, -0.2]],
            biases: vec![0.1],
            activation: ActivationName::Linear,
        }]);
        let before = net2.layers[0].weights[0].clone();
        let bad: Vec<(&[f64], &[f64])> = vec![(&[f64::NAN, 1.0][..], &[0.5][..])];
        let _ = net2.train_batch_slices(bad.into_iter(), 0.1);
        assert_eq!(
            net2.layers[0].weights[0], before,
            "NaN sample must leave weights untouched (no poisoning, no panic)"
        );
    }

    #[test]
    fn policy_gradient_drops_non_finite_input_without_panicking() {
        let mut policy = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: 2,
                hidden_layers: vec![3],
                output_dim: 3,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Linear,
                weight_scale: None,
            },
            &mut mulberry32(9),
        );
        let r = policy.train_policy_gradient_sample(&[f64::NAN, 0.2], 1, 1.0, 0.0, 0.1, 5.0);
        assert!(!r.applied, "NaN input must drop the policy step");
    }

    #[test]
    #[should_panic(expected = "does not match expected")]
    fn clipped_step_still_panics_on_dimension_mismatch() {
        // A wrong-length input is a structural/programmer error and must stay
        // loud — only non-finite *data* is downgraded to a silent dropped step.
        let mut net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.3, -0.2]],
            biases: vec![0.1],
            activation: ActivationName::Linear,
        }]);
        let _ = net.train_sample_clipped(&[1.0, 2.0, 3.0], &[0.5], 0.1, 4.0);
    }

    #[test]
    fn forward_pass_matches_hand_calc() {
        // One linear layer: row0 = 1*3 + 2*4 + 0.5 = 11.5 ; row1 = -1*4 + 1 = -3.
        let net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![1.0, 2.0], vec![0.0, -1.0]],
            biases: vec![0.5, 1.0],
            activation: ActivationName::Linear,
        }]);
        assert_eq!(net.input_dim, 2);
        assert_eq!(net.output_dim, 2);
        assert_eq!(net.predict(&[3.0, 4.0]), vec![11.5, -3.0]);
        assert_eq!(net.num_parameters(), 6);
    }

    #[test]
    fn embed_returns_last_hidden_layer_latent() {
        // 2 -> 3 -> 1: the latent is the 3-wide hidden layer the output reads.
        let mut rng = mulberry32(5);
        let net = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: 2,
                hidden_layers: vec![3],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Linear,
                weight_scale: None,
            },
            &mut rng,
        );
        assert_eq!(net.embedding_dim(), 3);
        let z = net.embed(&[0.4, -0.2]);
        assert_eq!(z.len(), 3);
        assert!(z.iter().all(|v| v.is_finite()));
        // The hidden latent must differ from both the input and the scalar output.
        assert_ne!(z, vec![0.4, -0.2]);
        assert_ne!(z.len(), net.predict(&[0.4, -0.2]).len());

        // A single-layer (no hidden) network has no learned latent: embed == input.
        let linear = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![1.0, 2.0]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        assert_eq!(linear.embedding_dim(), 2);
        assert_eq!(linear.embed(&[3.0, 4.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn borrowed_batch_training_matches_owned_batch_training() {
        let mut owned_network = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.2, -0.1]],
            biases: vec![0.05],
            activation: ActivationName::Linear,
        }]);
        let mut borrowed_network = owned_network.clone();
        let samples = vec![
            (vec![1.0, 2.0], vec![0.4]),
            (vec![-1.0, 0.5], vec![-0.3]),
            (vec![0.25, -0.75], vec![0.15]),
        ];

        let owned_loss = owned_network.train_batch(&samples, 0.02);
        let borrowed_loss = borrowed_network.train_batch_slices(
            samples
                .iter()
                .map(|(input, target)| (input.as_slice(), target.as_slice())),
            0.02,
        );

        assert!((owned_loss - borrowed_loss).abs() <= f64::EPSILON);
        assert_eq!(
            owned_network.predict(&[0.5, -0.25]),
            borrowed_network.predict(&[0.5, -0.25])
        );
    }

    #[test]
    fn learns_tiny_linear_regression() {
        // Fit y = 2x + 1 with a single linear neuron via the supervised DES.
        let network = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![0.0]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        let dataset = vec![
            SupervisedSample {
                input: vec![0.0],
                target: vec![1.0],
            },
            SupervisedSample {
                input: vec![1.0],
                target: vec![3.0],
            },
            SupervisedSample {
                input: vec![2.0],
                target: vec![5.0],
            },
            SupervisedSample {
                input: vec![3.0],
                target: vec![7.0],
            },
        ];
        let result = run_supervised_neural_net_des(SupervisedRunParams {
            network,
            dataset: dataset.clone(),
            epochs: 4000,
            learning_rate: 0.03,
            seed: Some(1),
            samples_per_tick: Some(1),
            shuffle_each_epoch: Some(false),
            snapshot_every: None,
        });
        for (sample, pred) in dataset.iter().zip(result.predictions.iter()) {
            assert!(
                (pred[0] - sample.target[0]).abs() < 0.2,
                "input {:?}: predicted {} vs target {}",
                sample.input,
                pred[0],
                sample.target[0]
            );
        }
        // Loss should have driven down well below the initial error.
        let last = *result.loss_history.last().unwrap();
        assert!(last < 0.05, "final loss {last} did not converge");
    }

    #[test]
    fn learns_xor() {
        let result = run_xor_neural_net_des(XorNeuralNetOptions {
            epochs: Some(8000),
            learning_rate: Some(0.3),
            seed: Some(7),
            ..Default::default()
        });
        let dataset = xor_dataset();
        for (sample, pred) in dataset.iter().zip(result.predictions.iter()) {
            // Each output must land on the correct side of 0.5.
            assert!(
                (pred[0] - sample.target[0]).abs() < 0.5,
                "XOR {:?}: predicted {} vs target {}",
                sample.input,
                pred[0],
                sample.target[0]
            );
        }
    }
}
