//! Port of `src/des/runners/validate-neural-network.ts`.
//!
//! Runs the optional Python neural-network reference when its script is present,
//! then cross-checks the framework's XOR supervised training, neural
//! Q-learning corridor policy, and neural ODE RK4 decay. Driver -> [`run`].
//!
//! The early Rust runner kept local network/RL stand-ins and reported a fake
//! external command success. The production neural-network and RL environment
//! modules are now available, so the in-repo checks exercise those real paths;
//! the external JSON comparison is skipped only when the reference script or
//! output is genuinely unavailable.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::general::des_base::environment::{
    PureEnvironment as DesPureEnvironment, StepResult as DesStepResult,
};
use crate::des::general::neural_network::{
    run_neural_q_learning_des, run_xor_neural_net_des, solve_neural_ode, ActivationName,
    DenseLayerConfig, FeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
    NeuralQLearningRunParams, XorNeuralNetOptions,
};
use crate::des::general::rl_environments::{
    eval_policy, Corridor as RealCorridor, Environment as RealEnvironment, EvalPolicyOptions,
};
use crate::des::runners::external_program::{run_python_reference, ExternalProgramResult};
use crate::des::shared::capabilities::SeededRandom;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct XorRef {
    predictions: Vec<f64>,
    #[serde(alias = "loss_history")]
    loss_history: Vec<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct CorridorRef {
    policy: Vec<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct OdeRef {
    #[serde(alias = "final_value")]
    final_value: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Reference {
    xor: XorRef,
    corridor: CorridorRef,
    #[serde(alias = "neural_ode_decay")]
    neural_ode_decay: OdeRef,
}

struct CorridorDesEnv {
    inner: RealCorridor,
}

impl CorridorDesEnv {
    fn new(length: usize) -> Self {
        CorridorDesEnv {
            inner: RealCorridor::new(length, 0),
        }
    }
}

impl DesPureEnvironment<f64, usize> for CorridorDesEnv {
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    fn reset(&mut self) -> f64 {
        self.inner.reset() as f64
    }

    fn step(&mut self, state: f64, action: usize) -> DesStepResult<f64> {
        let s = checked_state(state, self.inner.num_states());
        let result = self.inner.step(s, action);
        DesStepResult {
            next_state: result.next_state as f64,
            reward: result.reward,
            done: result.done,
        }
    }
}

fn checked_state(state: f64, n: usize) -> usize {
    if !state.is_finite() || state.fract() != 0.0 || state < 0.0 || state >= n as f64 {
        panic!("corridor state {state} outside [0, {n})");
    }
    state as usize
}

fn reference_script(root: &Path) -> PathBuf {
    root.join("external-references")
        .join("neural-network")
        .join("nn_reference.py")
}

fn run_optional_external_reference(root: &Path, out_path: &Path) -> Option<ExternalProgramResult> {
    let script = reference_script(root);
    if !script.exists() {
        println!(
            "  SKIP  external Python reference script unavailable: {}",
            script.display()
        );
        return None;
    }
    if let Some(parent) = out_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "failed to create external output dir {}: {e}",
                parent.display()
            );
            std::process::exit(1);
        }
    }
    let args = vec![
        "--out".to_string(),
        out_path.display().to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--xor-epochs".to_string(),
        "8000".to_string(),
        "--xor-lr".to_string(),
        "0.3".to_string(),
        "--corridor-length".to_string(),
        "6".to_string(),
        "--corridor-gamma".to_string(),
        "0.95".to_string(),
        "--ode-rate".to_string(),
        "0.5".to_string(),
        "--ode-y0".to_string(),
        "1".to_string(),
        "--ode-t1".to_string(),
        "2".to_string(),
        "--ode-dt".to_string(),
        "0.05".to_string(),
    ];
    match run_python_reference("external-references/neural-network/nn_reference.py", &args) {
        Ok(result) => Some(result),
        Err(e) => {
            eprintln!("failed to run neural-network external reference: {e}");
            std::process::exit(1);
        }
    }
}

fn load_reference(path: &Path) -> Option<Reference> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

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

fn check(checks: &mut Vec<CheckRow>, name: &str, passed: bool, detail: Option<String>) {
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
}

/// `validate-neural-network.ts` `main`.
pub fn run() {
    let mut checks: Vec<CheckRow> = Vec::new();

    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let out_path = root
        .join("out")
        .join("external")
        .join("neural-network")
        .join("reference.json");

    println!("Neural-network: framework vs optional external Python reference");
    println!("================================================================");
    let ext = run_optional_external_reference(&root, &out_path);
    if let Some(ext) = &ext {
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
        if ext.status != Some(0) {
            eprintln!("external reference exited with status {:?}", ext.status);
            std::process::exit(1);
        }
    }

    let reference = load_reference(&out_path);

    println!();
    println!("-- XOR supervised network --");
    let xor = run_xor_neural_net_des(XorNeuralNetOptions {
        seed: Some(7),
        epochs: Some(8000),
        learning_rate: Some(0.3),
        hidden_layers: Some(vec![4]),
        samples_per_tick: None,
        shuffle_each_epoch: None,
    });
    let xor_pred: Vec<f64> = xor.predictions.iter().map(|v| v[0]).collect();
    let xor_truth = [0.0, 1.0, 1.0, 0.0];
    let xor_err = max_abs_diff(&xor_pred, &xor_truth);
    check(
        &mut checks,
        "XOR predictions solve truth table",
        xor_err < 0.08,
        Some(format!(
            "max abs error={:.3e} predictions=[{}]",
            xor_err,
            xor_pred
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    );
    match &reference {
        Some(reference) if !reference.xor.predictions.is_empty() => {
            let pred_diff = max_abs_diff(&xor_pred, &reference.xor.predictions);
            check(
                &mut checks,
                "XOR predictions match external reference",
                pred_diff < 1e-12,
                Some(format!("max abs diff={:.3e}", pred_diff)),
            );
            if !reference.xor.loss_history.is_empty() {
                let n = xor.loss_history.len();
                let m = reference.xor.loss_history.len();
                let loss_diff = max_abs_diff(
                    &xor.loss_history[n.saturating_sub(100)..],
                    &reference.xor.loss_history[m.saturating_sub(100)..],
                );
                check(
                    &mut checks,
                    "XOR trailing losses match external reference",
                    loss_diff < 1e-12,
                    Some(format!("max abs diff={:.3e}", loss_diff)),
                );
            }
        }
        _ => println!("  SKIP  XOR external comparison (reference JSON unavailable)"),
    }

    println!();
    println!("-- Neural Q-learning corridor --");
    let length = 6usize;
    let q = run_neural_q_learning_des(
        Box::new(CorridorDesEnv::new(length)),
        NeuralQLearningRunParams {
            num_episodes: 600,
            max_steps_per_episode: Some(40),
            alpha: 0.25,
            gamma: 0.95,
            epsilon: 0.8,
            epsilon_decay: Some(0.99),
            epsilon_min: Some(0.02),
            seed: Some(1),
            network: None,
            hidden_layers: Some(Vec::new()),
            hidden_activation: None,
            state_encoder: None,
        },
    );
    let policy = q.policy.clone();
    let env_eval = RealCorridor::new(length, 0);
    let mut rng = SeededRandom::new(123);
    let eval_q = eval_policy(
        &env_eval,
        |s, _rng| policy[s],
        &mut rng,
        EvalPolicyOptions {
            num_episodes: 50,
            max_steps_per_episode: 40,
            gamma: 0.95,
        },
    );
    match &reference {
        Some(reference) if !reference.corridor.policy.is_empty() => {
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
        _ => println!("  SKIP  corridor policy external comparison (reference JSON unavailable)"),
    }
    check(
        &mut checks,
        "learned greedy policy succeeds in evaluation",
        eval_q.success_rate == 1.0,
        Some(format!(
            "success={} policy=[{}]",
            eval_q.success_rate,
            q.policy
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    );

    println!();
    println!("-- Neural ODE decay --");
    let net = FeedForwardNetwork::new(vec![DenseLayerConfig {
        weights: vec![vec![-0.5]],
        biases: vec![0.0],
        activation: ActivationName::Linear,
    }]);
    let trace = solve_neural_ode(
        &net,
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
    let framework_final = trace.y[trace.y.len() - 1][0];
    match &reference {
        Some(reference) if reference.neural_ode_decay.final_value.is_finite() => {
            let final_diff = (framework_final - reference.neural_ode_decay.final_value).abs();
            check(
                &mut checks,
                "neural ODE final state matches external RK4",
                final_diff < 1e-12,
                Some(format!("diff={:.3e}", final_diff)),
            );
        }
        _ => println!("  SKIP  neural ODE external comparison (reference JSON unavailable)"),
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
