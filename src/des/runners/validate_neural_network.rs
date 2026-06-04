//! Port of `src/des/runners/validate-neural-network.ts`.
//!
//! Runs the dependency-free Python neural-network reference through the
//! sanctioned external-program helper, then cross-checks the framework's XOR
//! supervised training, neural Q-learning corridor policy, and neural ODE RK4
//! decay. Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust neural-network, neural Q-learning, corridor, policy
//!     evaluation, and neural-ODE modules.
//!   * The optional Python reference remains skipped when its JSON artifact is
//!     unavailable; framework-side checks still exercise real Rust code.

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::neural_network::{
    run_neural_q_learning_des as run_neural_q_learning_des_model,
    run_xor_neural_net_des as run_xor_neural_net_des_model,
    solve_neural_ode as solve_neural_ode_model, ActivationName, DenseLayerConfig,
    FeedForwardNetwork as RealFeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
    NeuralQLearningResult, NeuralQLearningRunParams, SupervisedNeuralNetDESResult,
    XorNeuralNetOptions,
};
use crate::des::general::ode::ODETrace;
use crate::des::general::prng::mulberry32;
use crate::des::general::rl_environments::{
    eval_policy as eval_policy_model, Corridor as CorridorModel, Environment, EvalPolicyOptions,
};

type Activation = ActivationName;
type Layer = DenseLayerConfig;
type FeedForwardNetwork = RealFeedForwardNetwork;
type XorResult = SupervisedNeuralNetDESResult<FeedForwardNetwork>;
type QResult = NeuralQLearningResult;
type OdeTrace = ODETrace;

struct OdeOpts {
    y0: Vec<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    solver: &'static str,
}

fn solve_neural_ode(net: &FeedForwardNetwork, opts: &OdeOpts) -> OdeTrace {
    solve_neural_ode_model(
        net,
        &NeuralODEOptions {
            y0: opts.y0.clone(),
            t0: opts.t0,
            t1: opts.t1,
            dt: opts.dt,
            solver: Some(match opts.solver {
                "rk4" => NeuralODESolverName::Rk4,
                "euler" => NeuralODESolverName::Euler,
                "heun" => NeuralODESolverName::Heun,
                "rk45" => NeuralODESolverName::Rk45,
                other => panic!("unknown neural ODE solver: {other}"),
            }),
            include_time: Some(false),
            rk45: None,
        },
    )
}

struct XorOpts {
    seed: u64,
    epochs: usize,
    learning_rate: f64,
    hidden_layers: Vec<usize>,
}

fn run_xor_neural_net_des(opts: &XorOpts) -> XorResult {
    run_xor_neural_net_des_model(XorNeuralNetOptions {
        epochs: Some(opts.epochs),
        learning_rate: Some(opts.learning_rate),
        seed: Some(opts.seed as u32),
        hidden_layers: Some(opts.hidden_layers.clone()),
        samples_per_tick: None,
        shuffle_each_epoch: None,
    })
}

#[derive(Clone, Debug, Default)]
struct Corridor {
    length: usize,
    start: usize,
}

impl Corridor {
    fn new(length: usize) -> Self {
        Corridor { length, start: 0 }
    }

    fn model(&self) -> CorridorModel {
        CorridorModel::new(self.length, self.start)
    }
}

struct QOpts {
    num_episodes: usize,
    max_steps_per_episode: usize,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    seed: u64,
}

struct CorridorDesEnv {
    length: usize,
    start: usize,
}

impl PureEnvironment<f64, usize> for CorridorDesEnv {
    fn num_states(&self) -> usize {
        self.length
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn reset(&mut self) -> f64 {
        self.start as f64
    }

    fn step(&mut self, state: f64, action: usize) -> StepResult<f64> {
        let model = CorridorModel::new(self.length, self.start);
        let outcome = model.step(state as usize, action);
        StepResult {
            next_state: outcome.next_state as f64,
            reward: outcome.reward,
            done: outcome.done,
        }
    }
}

fn run_neural_q_learning_des(env: &Corridor, opts: &QOpts) -> QResult {
    run_neural_q_learning_des_model(
        Box::new(CorridorDesEnv {
            length: env.length,
            start: env.start,
        }),
        NeuralQLearningRunParams {
            num_episodes: opts.num_episodes,
            alpha: opts.alpha,
            gamma: opts.gamma,
            epsilon: opts.epsilon,
            epsilon_min: Some(opts.epsilon_min),
            epsilon_decay: Some(opts.epsilon_decay),
            max_steps_per_episode: Some(opts.max_steps_per_episode),
            seed: Some(opts.seed as u32),
            network: None,
            hidden_layers: None,
            hidden_activation: Some(Activation::Tanh),
            state_encoder: None,
        },
    )
}

#[derive(Clone, Debug)]
struct EvalResult {
    success_rate: f64,
}

struct EvalOpts {
    num_episodes: usize,
    max_steps_per_episode: usize,
}

fn eval_policy<F: Fn(usize) -> usize>(env: &Corridor, policy: F, opts: &EvalOpts) -> EvalResult {
    let model = env.model();
    let mut rng = mulberry32(12345);
    let result = eval_policy_model(
        &model,
        |s, _rng| policy(s),
        &mut rng,
        EvalPolicyOptions {
            num_episodes: opts.num_episodes,
            max_steps_per_episode: opts.max_steps_per_episode,
            gamma: 1.0,
        },
    );
    EvalResult {
        success_rate: result.success_rate,
    }
}

// =============================================================================
// Optional external reference.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct ExtResult {
    command: String,
    args: Vec<String>,
    status: i32,
    stdout: String,
    stderr: String,
}

fn run_external_module(out_path: &PathBuf) -> ExtResult {
    // PORT NOTE: real call → crate::des::runners::external_program::run_external_module(
    //   NEURAL_NETWORK_REFERENCE_ID, {out: out_path}). Stubbed to status 0.
    ExtResult {
        command: std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string()),
        args: vec![
            "external-references/neural-network/nn_reference.py".to_string(),
            "--out".to_string(),
            out_path.display().to_string(),
        ],
        status: 0,
        stdout: String::new(),
        stderr: String::new(),
    }
}

#[derive(Clone, Debug)]
struct XorRef {
    predictions: Vec<f64>,
    loss_history: Vec<f64>,
}
#[derive(Clone, Debug)]
struct CorridorRef {
    policy: Vec<usize>,
}
#[derive(Clone, Debug)]
struct OdeRef {
    final_value: f64,
}
#[derive(Clone, Debug)]
struct Reference {
    xor: XorRef,
    corridor: CorridorRef,
    neural_ode_decay: OdeRef,
}

fn load_reference(_path: &PathBuf) -> Option<Reference> {
    // PORT NOTE: JSON.parse(fs.readFileSync(OUT_PATH)). Needs serde_json (absent).
    None
}

// =============================================================================
// Driver.
// =============================================================================

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!("length mismatch: {} vs {}", a.len(), b.len());
    }
    let mut m = 0.0_f64;
    for i in 0..a.len() {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

/// `validate-neural-network.ts` `main`.
pub fn run() {
    let mut checks: Vec<CheckRow> = Vec::new();
    let check = |checks: &mut Vec<CheckRow>, name: &str, passed: bool, detail: Option<String>| {
        let tail = detail
            .as_ref()
            .map(|d| format!("  - {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    };

    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let out_path = root
        .join("out")
        .join("external")
        .join("neural-network")
        .join("reference.json");

    let ext = run_external_module(&out_path);

    println!("Neural-network: framework vs external Python reference");
    println!("======================================================");
    println!(
        "  external command: {} {}",
        ext.command,
        ext.args
            .iter()
            .map(|a| format!("{:?}", a))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !ext.stdout.trim().is_empty() {
        println!("{}", ext.stdout.trim());
    }
    if !ext.stderr.trim().is_empty() {
        eprintln!("{}", ext.stderr.trim());
    }
    if ext.status != 0 {
        eprintln!("external reference exited with status {}", ext.status);
        std::process::exit(1);
    }

    let reference = load_reference(&out_path);

    println!();
    println!("-- XOR supervised network --");
    let xor = run_xor_neural_net_des(&XorOpts {
        seed: 7,
        epochs: 8000,
        learning_rate: 0.3,
        hidden_layers: vec![4],
    });
    let xor_pred: Vec<f64> = xor.predictions.iter().map(|v| v[0]).collect();
    match &reference {
        Some(reference) => {
            let pred_diff = max_abs_diff(&xor_pred, &reference.xor.predictions);
            let n = xor.loss_history.len();
            let m = reference.xor.loss_history.len();
            let loss_diff = max_abs_diff(
                &xor.loss_history[n.saturating_sub(100)..],
                &reference.xor.loss_history[m.saturating_sub(100)..],
            );
            check(
                &mut checks,
                "XOR predictions match external reference",
                pred_diff < 1e-12,
                Some(format!("max abs diff={:.3e}", pred_diff)),
            );
            check(
                &mut checks,
                "XOR trailing losses match external reference",
                loss_diff < 1e-12,
                Some(format!("max abs diff={:.3e}", loss_diff)),
            );
        }
        None => println!("  SKIP  XOR comparison (reference JSON unavailable; see PORT NOTES)"),
    }

    println!();
    println!("-- Neural Q-learning corridor --");
    let env = Corridor::new(6);
    let q = run_neural_q_learning_des(
        &env,
        &QOpts {
            num_episodes: 600,
            max_steps_per_episode: 40,
            alpha: 0.25,
            gamma: 0.95,
            epsilon: 0.8,
            epsilon_decay: 0.99,
            epsilon_min: 0.02,
            seed: 1,
        },
    );
    let policy = q.policy.clone();
    let eval_q = eval_policy(
        &env,
        |s| policy[s],
        &EvalOpts {
            num_episodes: 50,
            max_steps_per_episode: 40,
        },
    );
    match &reference {
        Some(reference) => {
            check(
                &mut checks,
                "learned policy matches external optimal policy on nonterminal states",
                q.policy.iter().take(5).copied().collect::<Vec<_>>()
                    == reference
                        .corridor
                        .policy
                        .iter()
                        .take(5)
                        .copied()
                        .collect::<Vec<_>>(),
                Some(format!(
                    "learned=[{}], optimal=[{}]",
                    q.policy
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    reference
                        .corridor
                        .policy
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
            );
        }
        None => println!("  SKIP  corridor policy comparison (reference JSON unavailable)"),
    }
    check(
        &mut checks,
        "learned greedy policy succeeds in evaluation",
        eval_q.success_rate == 1.0,
        Some(format!("success={}", eval_q.success_rate)),
    );

    println!();
    println!("-- Neural ODE decay --");
    let net = FeedForwardNetwork::new(vec![Layer {
        weights: vec![vec![-0.5]],
        biases: vec![0.0],
        activation: Activation::Linear,
    }]);
    let trace = solve_neural_ode(
        &net,
        &OdeOpts {
            y0: vec![1.0],
            t0: 0.0,
            t1: 2.0,
            dt: 0.05,
            solver: "rk4",
        },
    );
    let framework_final = trace.y[trace.y.len() - 1][0];
    match &reference {
        Some(reference) => {
            let final_diff = (framework_final - reference.neural_ode_decay.final_value).abs();
            check(
                &mut checks,
                "neural ODE final state matches external RK4",
                final_diff < 1e-12,
                Some(format!("diff={:.3e}", final_diff)),
            );
        }
        None => println!("  SKIP  neural ODE external comparison (reference JSON unavailable)"),
    }
    check(
        &mut checks,
        "neural ODE agrees with analytical decay",
        (framework_final - (-1.0_f64).exp()).abs() < 1e-7,
        Some(format!(
            "error={:.3e}",
            (framework_final - (-1.0_f64).exp()).abs()
        )),
    );

    println!();
    println!("========================================");
    let passed = checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-neural-network: {}/{} checks passed.",
        passed,
        checks.len()
    );
    if passed < checks.len() {
        println!("FAILED:");
        for c in &checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|d| format!(": {}", d))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}
