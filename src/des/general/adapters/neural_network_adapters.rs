//! Port of `src/des/general/adapters/neural-network-adapters.ts`
//! (module `des::general::adapters::neural_network_adapters`).
//!
//! Registers three neural-network JSON adapters: `neural-xor`,
//! `neural-qlearning-corridor`, and `neural-ode-decay`.
//!
//! ## Conversion notes
//!
//!   * `type NeuralQCorridorResult = ReturnType<typeof runNeuralQLearningDES> &
//!     {eval}` -> the named [`NeuralQCorridorResult`] struct (base run result in
//!     `base`, plus an `eval` field).
//!   * `s => r.policy[s]` greedy-policy closure -> a closure capturing a clone
//!     of the learned `policy` vector.
//!   * `solver: 'euler'|'heun'|'rk4'|'rk45'` -> the engine
//!     [`NeuralODESolverName`] enum.
//!   * `hiddenLayers && len>0 ? .. : [4]` -> `Option::filter(non-empty)` then
//!     `unwrap_or`.
//!   * The policy-evaluation RNG is injected ([`SeededRandom`]) seeded by the
//!     model `seed`, replacing the ambient `Math.random` used by the TS
//!     `evalPolicy` default.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/scenes/neural-network-scene`) is not ported; the three `animate`
//! hooks are left as the trait's no-op default.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the three
//! adapters are exposed via the `*_adapter()` constructors.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::write_csv_lines;
use crate::des::general::des_base::runner::RunReason;
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};
use crate::des::general::neural_network::{
    run_neural_q_learning_des, run_xor_neural_net_des, solve_neural_ode, ActivationName,
    DenseLayerConfig, FeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
    NeuralQLearningResult, NeuralQLearningRunParams, SupervisedNeuralNetDESResult,
    XorNeuralNetOptions,
};
use crate::des::general::ode::ODETrace;
use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::rl_environments::{
    eval_policy, Corridor, Environment, EvalPolicyOptions, EvalPolicyResult,
};
use crate::des::shared::capabilities::SeededRandom;

/// Bridges the pure [`Environment`] trait to the [`PureEnvironment<f64, usize>`]
/// that the neural Q-learning runner requires (state indices carried as `f64`).
/// TS passes the `Corridor` directly; Rust's neural runner is generic over a
/// float state encoding, so we adapt the index-based environment here.
struct PureEnvAdapter<E: Environment> {
    env: E,
}

impl<E: Environment> PureEnvironment<f64, usize> for PureEnvAdapter<E> {
    fn num_states(&self) -> usize {
        self.env.num_states()
    }
    fn num_actions(&self) -> usize {
        self.env.num_actions()
    }
    fn reset(&mut self) -> f64 {
        self.env.reset() as f64
    }
    fn step(&mut self, state: f64, action: usize) -> StepResult<f64> {
        let o = self.env.step(state as usize, action);
        StepResult { next_state: o.next_state as f64, reward: o.reward, done: o.done }
    }
}

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `Number.prototype.toExponential(digits)` (signed exponent, no leading zeros).
fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    let (mant, exp) = raw.split_once('e').unwrap_or((raw.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{mant}e{sign}{}", exp_num.abs())
}

/// `function meanLast(xs, n)`.
fn mean_last(xs: &[f64], n: usize) -> f64 {
    let take = n.min(xs.len());
    let tail = &xs[xs.len() - take..];
    tail.iter().sum::<f64>() / (tail.len().max(1) as f64)
}

fn reason_str(reason: &Option<RunReason>) -> &'static str {
    reason.as_ref().map(|r| r.as_str()).unwrap_or("done")
}

fn solver_str(s: NeuralODESolverName) -> &'static str {
    match s {
        NeuralODESolverName::Euler => "euler",
        NeuralODESolverName::Heun => "heun",
        NeuralODESolverName::Rk4 => "rk4",
        NeuralODESolverName::Rk45 => "rk45",
    }
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn boolean(default: Option<bool>) -> ParamSchema {
    ParamSchema::Boolean { default, description: None }
}

/// `const hiddenLayersSchema: ParamSchema`.
fn hidden_layers_schema() -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(num(Some(1.0), None, Some(true), None)),
        min_length: None,
        max_length: None,
        description: Some("Hidden layer widths. Missing/empty uses [4].".to_string()),
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, description: &str) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(Vec::new()),
        description: Some(description.to_string()),
    }
}

// =============================================================================
// 1. neural-xor
// =============================================================================

/// `interface NeuralXorParams`.
#[derive(Clone, Debug, Default)]
pub struct NeuralXorParams {
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub seed: Option<u32>,
    pub hidden_layers: Option<Vec<usize>>,
    pub samples_per_tick: Option<usize>,
    pub shuffle_each_epoch: Option<bool>,
}

impl NeuralXorParams {
    fn resolved_hidden_layers(&self) -> Vec<usize> {
        self.hidden_layers
            .clone()
            .filter(|h| !h.is_empty())
            .unwrap_or_else(|| vec![4])
    }
}

pub struct NeuralXorAdapter;

pub fn neural_xor_adapter() -> NeuralXorAdapter {
    NeuralXorAdapter
}

impl DESModelRegistration<NeuralXorParams, SupervisedNeuralNetDESResult<FeedForwardNetwork>>
    for NeuralXorAdapter
{
    fn id(&self) -> &str {
        "neural-xor"
    }
    fn description(&self) -> &str {
        "Feed-forward neural net trained on XOR with queued DES sample tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("epochs", num(Some(1.0), None, Some(true), Some(8000.0))),
                ("learningRate", num(Some(0.0), None, None, Some(0.3))),
                ("seed", num(None, None, Some(true), Some(7.0))),
                ("hiddenLayers", hidden_layers_schema()),
                ("samplesPerTick", num(Some(1.0), None, Some(true), Some(1.0))),
                ("shuffleEachEpoch", boolean(Some(false))),
            ],
            "XOR learned by a feed-forward neural network running as DES training stations.",
        )
    }
    fn run(
        &self,
        p: NeuralXorParams,
        _runtime: &DESRuntimeConfig,
    ) -> SupervisedNeuralNetDESResult<FeedForwardNetwork> {
        run_xor_neural_net_des(XorNeuralNetOptions {
            epochs: p.epochs,
            learning_rate: p.learning_rate,
            seed: p.seed,
            hidden_layers: Some(p.resolved_hidden_layers()),
            samples_per_tick: p.samples_per_tick,
            shuffle_each_epoch: p.shuffle_each_epoch,
        })
    }
    fn summarize(
        &self,
        r: &SupervisedNeuralNetDESResult<FeedForwardNetwork>,
        p: &NeuralXorParams,
    ) -> String {
        let avg = mean_last(&r.loss_history, 100);
        let preds = r
            .predictions
            .iter()
            .map(|v| format!("{:.4}", v[0]))
            .collect::<Vec<_>>()
            .join(", ");
        let hidden = p
            .resolved_hidden_layers()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        [
            "NEURAL XOR (supervised DES training)".to_string(),
            "─".repeat(36),
            format!("  Epochs:                 {}", p.epochs.unwrap_or(8000)),
            format!("  Hidden layers:          {hidden}"),
            format!("  Samples trained:        {}", r.loss_history.len()),
            format!("  Ticks:                  {} ({})", r.ticks, reason_str(&r.reason)),
            format!("  Avg loss (last 100):    {}", to_exponential(avg, 3)),
            format!("  XOR predictions:        [{preds}]"),
            format!("  Parameter count:        {}", r.network.num_parameters()),
        ]
        .join("\n")
    }
}

// =============================================================================
// 2. neural-qlearning-corridor
// =============================================================================

/// `interface NeuralQCorridorParams`.
#[derive(Clone, Debug, Default)]
pub struct NeuralQCorridorParams {
    pub length: Option<usize>,
    pub num_episodes: Option<usize>,
    pub max_steps_per_episode: Option<usize>,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub seed: Option<u32>,
    pub hidden_layers: Option<Vec<usize>>,
}

/// `type NeuralQCorridorResult = ReturnType<...> & {eval}`.
#[derive(Clone)]
pub struct NeuralQCorridorResult {
    pub base: NeuralQLearningResult,
    pub eval: EvalPolicyResult,
}

pub struct NeuralQlearningCorridorAdapter;

pub fn neural_qlearning_corridor_adapter() -> NeuralQlearningCorridorAdapter {
    NeuralQlearningCorridorAdapter
}

impl DESModelRegistration<NeuralQCorridorParams, NeuralQCorridorResult>
    for NeuralQlearningCorridorAdapter
{
    fn id(&self) -> &str {
        "neural-qlearning-corridor"
    }
    fn description(&self) -> &str {
        "Neural Q-learning agent learning a corridor MDP through DES environment tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("length", num(Some(2.0), None, Some(true), Some(6.0))),
                ("numEpisodes", num(Some(1.0), None, Some(true), Some(600.0))),
                ("maxStepsPerEpisode", num(Some(1.0), None, Some(true), Some(40.0))),
                ("alpha", num(Some(0.0), None, None, Some(0.25))),
                ("gamma", num(Some(0.0), Some(1.0), None, Some(0.95))),
                ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.8))),
                ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(0.99))),
                ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.02))),
                ("seed", num(None, None, Some(true), Some(1.0))),
                ("hiddenLayers", hidden_layers_schema()),
            ],
            "Neural Q-learning on a small corridor MDP.",
        )
    }
    fn run(&self, p: NeuralQCorridorParams, _runtime: &DESRuntimeConfig) -> NeuralQCorridorResult {
        let length = p.length.unwrap_or(6);
        let max_steps = p.max_steps_per_episode.unwrap_or(40);
        let gamma = p.gamma.unwrap_or(0.95);
        let seed = p.seed.unwrap_or(1);
        let hidden = p.hidden_layers.clone().filter(|h| !h.is_empty()).unwrap_or_default();

        let base = run_neural_q_learning_des(
            Box::new(PureEnvAdapter { env: Corridor::new(length, 0) }),
            NeuralQLearningRunParams {
                num_episodes: p.num_episodes.unwrap_or(600),
                alpha: p.alpha.unwrap_or(0.25),
                gamma,
                epsilon: p.epsilon.unwrap_or(0.8),
                epsilon_decay: Some(p.epsilon_decay.unwrap_or(0.99)),
                epsilon_min: Some(p.epsilon_min.unwrap_or(0.02)),
                max_steps_per_episode: Some(max_steps),
                seed: Some(seed),
                network: None,
                hidden_layers: Some(hidden),
                hidden_activation: None,
                state_encoder: None,
            },
        );

        let policy = base.policy.clone();
        let env_eval = Corridor::new(length, 0);
        let mut eval_rng = SeededRandom::new(seed);
        let eval = eval_policy(
            &env_eval,
            |s, _rng| policy[s],
            &mut eval_rng,
            EvalPolicyOptions { num_episodes: 50, max_steps_per_episode: max_steps, gamma },
        );

        NeuralQCorridorResult { base, eval }
    }
    fn summarize(&self, r: &NeuralQCorridorResult, p: &NeuralQCorridorParams) -> String {
        [
            "NEURAL Q-LEARNING (Corridor MDP)".to_string(),
            "─".repeat(32),
            format!("  Corridor length:         {}", p.length.unwrap_or(6)),
            format!("  Episodes:                {}", r.base.total_episodes),
            format!("  Steps:                   {}", r.base.total_steps),
            format!("  Ticks:                   {}", r.base.total_ticks),
            format!(
                "  Greedy policy:           [{}]",
                r.base.policy.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
            ),
            format!("  Eval success rate:       {:.1}%", 100.0 * r.eval.success_rate),
            format!("  Eval mean length:        {:.1}", r.eval.mean_length),
            format!(
                "  Avg TD loss (last 100):  {}",
                to_exponential(mean_last(&r.base.loss_history, 100), 3)
            ),
        ]
        .join("\n")
    }
}

// =============================================================================
// 3. neural-ode-decay
// =============================================================================

/// `interface NeuralODEDecayParams`.
#[derive(Clone, Debug, Default)]
pub struct NeuralODEDecayParams {
    pub rate: Option<f64>,
    pub y0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub solver: Option<NeuralODESolverName>,
}

/// `interface NeuralODEDecayResult`.
#[derive(Clone, Debug)]
pub struct NeuralODEDecayResult {
    pub trace: ODETrace,
    pub exact_final: f64,
    pub error: f64,
}

pub struct NeuralOdeDecayAdapter;

pub fn neural_ode_decay_adapter() -> NeuralOdeDecayAdapter {
    NeuralOdeDecayAdapter
}

impl DESModelRegistration<NeuralODEDecayParams, NeuralODEDecayResult> for NeuralOdeDecayAdapter {
    fn id(&self) -> &str {
        "neural-ode-decay"
    }
    fn description(&self) -> &str {
        "Neural ODE demo: a network supplies dy/dt and the existing ODE solver integrates it."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("rate", num(Some(0.0), None, None, Some(0.5))),
                ("y0", num(None, None, None, Some(1.0))),
                ("t1", num(Some(0.0), None, None, Some(2.0))),
                ("dt", num(Some(1e-9), None, None, Some(0.05))),
                (
                    "solver",
                    ParamSchema::String {
                        allowed: Some(
                            ["euler", "heun", "rk4", "rk45"].iter().map(|s| s.to_string()).collect(),
                        ),
                        default: Some("rk4".to_string()),
                        description: None,
                    },
                ),
            ],
            "Solve y' = -rate*y where the vector field is represented by a one-layer neural net.",
        )
    }
    fn run(&self, p: NeuralODEDecayParams, _runtime: &DESRuntimeConfig) -> NeuralODEDecayResult {
        let rate = p.rate.unwrap_or(0.5);
        let y0 = p.y0.unwrap_or(1.0);
        let t1 = p.t1.unwrap_or(2.0);
        let network = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![-rate]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        let trace = solve_neural_ode(
            &network,
            &NeuralODEOptions {
                y0: vec![y0],
                t0: 0.0,
                t1,
                dt: p.dt.unwrap_or(0.05),
                solver: Some(p.solver.unwrap_or(NeuralODESolverName::Rk4)),
                include_time: None,
                rk45: None,
            },
        );
        let final_y = trace.y[trace.y.len() - 1][0];
        let exact_final = y0 * (-rate * t1).exp();
        NeuralODEDecayResult { error: (final_y - exact_final).abs(), exact_final, trace }
    }
    fn summarize(&self, r: &NeuralODEDecayResult, p: &NeuralODEDecayParams) -> String {
        let final_y = r.trace.y[r.trace.y.len() - 1][0];
        [
            "NEURAL ODE DECAY".to_string(),
            "─".repeat(32),
            format!("  Equation:                y' = -{} y", js_number(p.rate.unwrap_or(0.5))),
            format!("  Solver:                  {}", solver_str(p.solver.unwrap_or(NeuralODESolverName::Rk4))),
            format!("  Steps recorded:          {}", r.trace.t.len()),
            format!("  Final y:                 {:.6}", final_y),
            format!("  Exact y:                 {:.6}", r.exact_final),
            format!("  Abs error:               {}", to_exponential(r.error, 3)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &NeuralODEDecayResult, csv_path: &str) {
        let mut lines = vec!["t,y".to_string()];
        for i in 0..r.trace.t.len() {
            lines.push(format!("{:.8},{:.12}", r.trace.t[i], r.trace.y[i][0]));
        }
        write_csv_lines(csv_path, &lines);
    }
}
