//! Port of `src/des/main-neural-net.ts`.
//!
//! Thin runner: supervised XOR net, neural Q-learning on a Corridor MDP, and a
//! neural-ODE demo.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`]; `process.env` (`SEED`, `XOR_EPOCHS`,
//!     `XOR_LR`, `Q_EPISODES`, `Q_ALPHA`) → `std::env::var`.
//!   - the policy evaluation RNG is injected (`SeededRandom`) rather than
//!     ambient `Math.random`.
//!   - delegates to `general::neural_network` and `general::rl_environments`.

use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::neural_network::{
    run_neural_q_learning_des, run_xor_neural_net_des, solve_neural_ode, ActivationName,
    DenseLayerConfig, FeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
    NeuralQLearningRunParams, XorNeuralNetOptions,
};
use crate::des::general::rl_environments::{eval_policy, Corridor, Environment, EvalPolicyOptions};
use crate::des::shared::capabilities::SeededRandom;

// PORT NOTE: the `rl_environments` envs implement the pure `Environment` trait
// but not the `des_base` `PureEnvironment<f64, usize>` that the neural
// Q-learning runner requires. This thin local adapter bridges the two (state
// indices are carried as `f64`, matching the TS env-as-number usage). Replace
// with a canonical crate adapter once one is exposed.
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
        StepResult {
            next_state: o.next_state as f64,
            reward: o.reward,
            done: o.done,
        }
    }
}

/// `Number.prototype.toExponential(digits)` (signed exponent, no leading zeros).
fn to_exponential(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    let s = format!("{:.*e}", digits, x);
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{}e{}{}", mant, sign, exp_num.abs())
}

/// `Number.prototype` default string for an integer-valued f64 (no `.0`).
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

fn mean_last(xs: &[f64], n: usize) -> f64 {
    let take = n.min(xs.len());
    let tail = &xs[xs.len() - take..];
    tail.iter().sum::<f64>() / (tail.len().max(1) as f64)
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let seed: u32 = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    println!("# Neural-net DES demo");
    println!("# seed = {}", seed);

    let xor = run_xor_neural_net_des(XorNeuralNetOptions {
        seed: Some(seed),
        epochs: Some(
            std::env::var("XOR_EPOCHS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8000),
        ),
        learning_rate: Some(
            std::env::var("XOR_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.3),
        ),
        hidden_layers: Some(vec![4]),
        samples_per_tick: None,
        shuffle_each_epoch: None,
    });
    println!();
    println!("## Supervised XOR");
    println!("samples trained = {}", xor.loss_history.len());
    println!(
        "ticks = {} ({})",
        xor.ticks,
        xor.reason.map(|r| r.as_str()).unwrap_or("done")
    );
    println!(
        "avg loss last 100 = {}",
        to_exponential(mean_last(&xor.loss_history, 100), 3)
    );
    println!(
        "predictions [00, 01, 10, 11] = [{}]",
        xor.predictions
            .iter()
            .map(|v| format!("{:.4}", v[0]))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let q = run_neural_q_learning_des(
        Box::new(PureEnvAdapter {
            env: Corridor::new(6, 0),
        }),
        NeuralQLearningRunParams {
            num_episodes: std::env::var("Q_EPISODES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            max_steps_per_episode: Some(40),
            alpha: std::env::var("Q_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25),
            gamma: 0.95,
            epsilon: 0.8,
            epsilon_decay: Some(0.99),
            epsilon_min: Some(0.02),
            seed: Some(seed),
            network: None,
            hidden_layers: None,
            hidden_activation: None,
            state_encoder: None,
        },
    );
    let env_eval = Corridor::new(6, 0);
    let mut eval_rng = SeededRandom::new(seed);
    let eval_q = eval_policy(
        &env_eval,
        |s, _rng| q.policy[s],
        &mut eval_rng,
        EvalPolicyOptions {
            num_episodes: 50,
            max_steps_per_episode: 40,
            gamma: 1.0,
        },
    );
    println!();
    println!("## Neural Q-learning on Corridor MDP");
    println!(
        "episodes = {}, steps = {}, ticks = {}",
        q.total_episodes, q.total_steps, q.total_ticks
    );
    println!(
        "greedy policy = [{}]",
        q.policy
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "eval success = {:.1}%, mean length = {:.1}",
        100.0 * eval_q.success_rate,
        eval_q.mean_length
    );
    println!(
        "avg TD loss last 100 = {}",
        to_exponential(mean_last(&q.loss_history, 100), 3)
    );

    let rate = 0.5;
    let ode_net = FeedForwardNetwork::new(vec![DenseLayerConfig {
        weights: vec![vec![-rate]],
        biases: vec![0.0],
        activation: ActivationName::Linear,
    }]);
    let trace = solve_neural_ode(
        &ode_net,
        &NeuralODEOptions {
            y0: vec![1.0],
            t0: 0.0,
            t1: 2.0,
            dt: 0.05,
            solver: Some(NeuralODESolverName::Rk4),
            include_time: None,
            rk45: None,
        },
    );
    let final_y = trace.y[trace.y.len() - 1][0];
    let exact = (-rate * 2.0).exp();
    println!();
    println!("## Neural ODE");
    println!(
        "dy/dt = -{} y, final = {:.6}, exact = {:.6}, abs error = {}",
        num_str(rate),
        final_y,
        exact,
        to_exponential((final_y - exact).abs(), 3)
    );
}
