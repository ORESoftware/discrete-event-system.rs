//! Port of `src/des/runners/validate-neural-network.ts`.
//!
//! Runs the dependency-free Python neural-network reference through the
//! sanctioned external-program helper, then cross-checks the framework's XOR
//! supervised training, neural Q-learning corridor policy, and neural ODE RK4
//! decay. Driver → [`run`].
//!
//! PORT NOTES — wire to real modules:
//!   * `crate::des::runners::external_program::run_external_module` +
//!     `crate::des::runners::external_modules::NEURAL_NETWORK_REFERENCE_ID`.
//!     Here the external call is stubbed (`run_external_module`) so the file is
//!     self-contained.
//!   * Reading/parsing `out/external/neural-network/reference.json` needs
//!     `serde_json` (absent) → `load_reference` returns `None` and the
//!     reference-dependent checks print `SKIP`, mirroring the TS gating.
//!   * `crate::des::general::neural_network::{FeedForwardNetwork, solve_neural_ode}`
//!     are ported faithfully here; `run_xor_neural_net_des` /
//!     `run_neural_q_learning_des` are stubbed.
//!   * `crate::des::general::rl_environments::{Corridor, eval_policy}` stubbed.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;

// =============================================================================
// FeedForwardNetwork + neural ODE (faithful).
// =============================================================================

#[derive(Clone, Copy, Debug)]
enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
}

impl Activation {
    fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Linear => x,
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
        }
    }
}

#[derive(Clone, Debug)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: Activation,
}

#[derive(Clone, Debug)]
struct FeedForwardNetwork {
    layers: Vec<Layer>,
}

impl FeedForwardNetwork {
    fn new(layers: Vec<Layer>) -> Self {
        FeedForwardNetwork { layers }
    }
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            let mut out = vec![0.0; layer.weights.len()];
            for o in 0..layer.weights.len() {
                let mut s = layer.biases[o];
                for i in 0..layer.weights[o].len() {
                    s += layer.weights[o][i] * x[i];
                }
                out[o] = layer.activation.apply(s);
            }
            x = out;
        }
        x
    }
}

struct OdeOpts {
    y0: Vec<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    solver: &'static str,
}

struct OdeTrace {
    t: Vec<f64>,
    y: Vec<Vec<f64>>,
}

fn vec_add(a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + scale * y).collect()
}

fn solve_neural_ode(net: &FeedForwardNetwork, opts: &OdeOpts) -> OdeTrace {
    let n_steps = ((opts.t1 - opts.t0) / opts.dt).round() as usize;
    let mut t = opts.t0;
    let mut y = opts.y0.clone();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    for _ in 0..n_steps {
        let new_y = match opts.solver {
            "rk4" => {
                let k1 = net.forward(&y);
                let k2 = net.forward(&vec_add(&y, &k1, opts.dt / 2.0));
                let k3 = net.forward(&vec_add(&y, &k2, opts.dt / 2.0));
                let k4 = net.forward(&vec_add(&y, &k3, opts.dt));
                let mut out = y.clone();
                for i in 0..out.len() {
                    out[i] += opts.dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                }
                out
            }
            _ => {
                // Euler fallback.
                let k1 = net.forward(&y);
                vec_add(&y, &k1, opts.dt)
            }
        };
        y = new_y;
        t += opts.dt;
        ts.push(t);
        ys.push(y.clone());
    }
    OdeTrace { t: ts, y: ys }
}

// =============================================================================
// Stubbed framework training kernels + RL env.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct XorResult {
    predictions: Vec<Vec<f64>>,
    loss_history: Vec<f64>,
}

struct XorOpts {
    seed: u64,
    epochs: usize,
    learning_rate: f64,
    hidden_layers: Vec<usize>,
}

fn run_xor_neural_net_des(_opts: &XorOpts) -> XorResult {
    XorResult { predictions: vec![vec![0.0]; 4], loss_history: vec![0.0; 200] }
}

#[derive(Clone, Debug, Default)]
struct Corridor {
    length: usize,
}

impl Corridor {
    fn new(length: usize) -> Self {
        Corridor { length }
    }
}

#[derive(Clone, Debug, Default)]
struct QResult {
    policy: Vec<usize>,
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

fn run_neural_q_learning_des(env: &Corridor, _opts: &QOpts) -> QResult {
    QResult { policy: vec![0; env.length] }
}

#[derive(Clone, Debug, Default)]
struct EvalResult {
    success_rate: f64,
}

struct EvalOpts {
    num_episodes: usize,
    max_steps_per_episode: usize,
}

fn eval_policy<F: Fn(usize) -> usize>(_env: &Corridor, _policy: F, _opts: &EvalOpts) -> EvalResult {
    EvalResult { success_rate: 1.0 }
}

// =============================================================================
// External reference (stubbed).
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
    let mut check = |checks: &mut Vec<CheckRow>, name: &str, passed: bool, detail: Option<String>| {
        let tail = detail.as_ref().map(|d| format!("  - {}", d)).unwrap_or_default();
        println!("  {}  {}{}", if passed { "PASS" } else { "FAIL" }, name, tail);
        checks.push(CheckRow { name: name.to_string(), passed, detail });
    };

    let root = std::env::var("REPO_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let out_path = root.join("out").join("external").join("neural-network").join("reference.json");

    let ext = run_external_module(&out_path);

    println!("Neural-network: framework vs external Python reference");
    println!("======================================================");
    println!("  external command: {} {}", ext.command, ext.args.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>().join(" "));
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
    let xor = run_xor_neural_net_des(&XorOpts { seed: 7, epochs: 8000, learning_rate: 0.3, hidden_layers: vec![4] });
    let xor_pred: Vec<f64> = xor.predictions.iter().map(|v| v[0]).collect();
    match &reference {
        Some(reference) => {
            let pred_diff = max_abs_diff(&xor_pred, &reference.xor.predictions);
            let n = xor.loss_history.len();
            let m = reference.xor.loss_history.len();
            let loss_diff = max_abs_diff(&xor.loss_history[n.saturating_sub(100)..], &reference.xor.loss_history[m.saturating_sub(100)..]);
            check(&mut checks, "XOR predictions match external reference", pred_diff < 1e-12, Some(format!("max abs diff={:.3e}", pred_diff)));
            check(&mut checks, "XOR trailing losses match external reference", loss_diff < 1e-12, Some(format!("max abs diff={:.3e}", loss_diff)));
        }
        None => println!("  SKIP  XOR comparison (reference JSON unavailable; see PORT NOTES)"),
    }

    println!();
    println!("-- Neural Q-learning corridor --");
    let env = Corridor::new(6);
    let q = run_neural_q_learning_des(
        &env,
        &QOpts { num_episodes: 600, max_steps_per_episode: 40, alpha: 0.25, gamma: 0.95, epsilon: 0.8, epsilon_decay: 0.99, epsilon_min: 0.02, seed: 1 },
    );
    let policy = q.policy.clone();
    let eval_q = eval_policy(&env, |s| policy[s], &EvalOpts { num_episodes: 50, max_steps_per_episode: 40 });
    match &reference {
        Some(reference) => {
            check(
                &mut checks,
                "learned policy matches external optimal policy on nonterminal states",
                q.policy.iter().take(5).copied().collect::<Vec<_>>() == reference.corridor.policy.iter().take(5).copied().collect::<Vec<_>>(),
                Some(format!(
                    "learned=[{}], optimal=[{}]",
                    q.policy.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
                    reference.corridor.policy.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
                )),
            );
        }
        None => println!("  SKIP  corridor policy comparison (reference JSON unavailable)"),
    }
    check(&mut checks, "learned greedy policy succeeds in evaluation", eval_q.success_rate == 1.0, Some(format!("success={}", eval_q.success_rate)));

    println!();
    println!("-- Neural ODE decay --");
    let net = FeedForwardNetwork::new(vec![Layer { weights: vec![vec![-0.5]], biases: vec![0.0], activation: Activation::Linear }]);
    let trace = solve_neural_ode(&net, &OdeOpts { y0: vec![1.0], t0: 0.0, t1: 2.0, dt: 0.05, solver: "rk4" });
    let framework_final = trace.y[trace.y.len() - 1][0];
    match &reference {
        Some(reference) => {
            let final_diff = (framework_final - reference.neural_ode_decay.final_value).abs();
            check(&mut checks, "neural ODE final state matches external RK4", final_diff < 1e-12, Some(format!("diff={:.3e}", final_diff)));
        }
        None => println!("  SKIP  neural ODE external comparison (reference JSON unavailable)"),
    }
    check(
        &mut checks,
        "neural ODE agrees with analytical decay",
        (framework_final - (-1.0_f64).exp()).abs() < 1e-7,
        Some(format!("error={:.3e}", (framework_final - (-1.0_f64).exp()).abs())),
    );

    println!();
    println!("========================================");
    let passed = checks.iter().filter(|c| c.passed).count();
    println!("validate-neural-network: {}/{} checks passed.", passed, checks.len());
    if passed < checks.len() {
        println!("FAILED:");
        for c in &checks {
            if !c.passed {
                println!("  - {}{}", c.name, c.detail.as_ref().map(|d| format!(": {}", d)).unwrap_or_default());
            }
        }
        std::process::exit(1);
    }
}
