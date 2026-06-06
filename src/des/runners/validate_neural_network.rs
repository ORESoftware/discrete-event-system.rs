//! Port of `src/des/runners/validate-neural-network.ts`.
//!
//! Builds a Rust neural-network reference artifact, then cross-checks the
//! framework's XOR supervised training, neural Q-learning corridor policy, and
//! neural ODE RK4 decay. Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust neural-network, neural Q-learning, corridor, policy
//!     evaluation, and neural-ODE modules.
//!   * The optional external reference is invoked through the Rust-first
//!     external-module registry only when explicitly requested with
//!     `NEURAL_NETWORK_REFERENCE_BACKEND=rust|external` or
//!     `NEURAL_NETWORK_EXTERNAL_REFERENCE=1`; framework-side checks and the
//!     default reference artifact stay Rust-only.

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
use crate::des::general::neural_network::{
    run_neural_q_learning_des as run_neural_q_learning_des_model,
    run_xor_neural_net_des as run_xor_neural_net_des_model,
    solve_neural_ode as solve_neural_ode_model, xor_dataset, ActivationName, DenseLayerConfig,
    FeedForwardNetwork as RealFeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
    NeuralQLearningResult, NeuralQLearningRunParams, SupervisedNeuralNetDESResult,
    XorNeuralNetOptions,
};
use crate::des::general::ode::ODETrace;
use crate::des::general::prng::mulberry32;
use crate::des::general::rl_environments::{
    eval_policy as eval_policy_model, Corridor as CorridorModel, Environment, EvalPolicyOptions,
};
use crate::des::observability::logger::{parse_json, JsonValue};
use serde_json::json;

use super::external_modules::{register_built_in_external_modules, NEURAL_NETWORK_REFERENCE_ID};
use super::external_program::{
    run_external_module, ExternalModuleParams, ExternalProgramResult, ParamValue,
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
// Rust reference and optional external reference.
// =============================================================================

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

fn neural_network_external_reference_requested() -> bool {
    [
        "NEURAL_NETWORK_REFERENCE_BACKEND",
        "NEURAL_NETWORK_EXTERNAL_REFERENCE",
    ]
    .iter()
    .filter_map(|name| std::env::var(name).ok())
    .any(|value| neural_network_external_reference_value_requested(&value))
}

fn neural_network_external_reference_value_requested(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "rust" | "cargo" | "external"
    )
}

fn write_rust_reference(out_path: &PathBuf, reference: &Reference) -> Result<(), String> {
    let document = json!({
        "status": "ok",
        "backend": "rust",
        "result": {
            "xor": {
                "predictions": reference.xor.predictions.clone(),
                "lossHistory": reference.xor.loss_history.clone(),
            },
            "corridor": {
                "policy": reference.corridor.policy.clone(),
            },
            "neuralOdeDecay": {
                "finalValue": reference.neural_ode_decay.final_value,
            },
        },
    });
    let text = serde_json::to_string_pretty(&document)
        .map_err(|err| format!("serialize Rust neural reference: {err}"))?;
    std::fs::write(out_path, format!("{text}\n"))
        .map_err(|err| format!("write Rust neural reference: {err}"))?;
    Ok(())
}

fn run_external_reference(out_path: &PathBuf) -> ExternalProgramResult {
    let mut params = ExternalModuleParams::new();
    params.insert(
        "out".to_string(),
        ParamValue::Str(out_path.display().to_string()),
    );
    match run_external_module(NEURAL_NETWORK_REFERENCE_ID, &params) {
        Ok(result) => result,
        Err(error) => ExternalProgramResult {
            command: String::new(),
            args: Vec::new(),
            status: None,
            stdout: String::new(),
            stderr: error,
            module_id: Some(NEURAL_NETWORK_REFERENCE_ID.to_string()),
        },
    }
}

fn status_str(status: Option<i32>) -> String {
    status
        .map(|code| code.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn slice_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn optional_external_unavailable(ext: &ExternalProgramResult) -> Option<String> {
    let stderr = ext.stderr.trim();
    let stdout = ext.stdout.trim();
    let message = if stderr.is_empty() { stdout } else { stderr };
    let lower = message.to_ascii_lowercase();
    let unavailable = lower.contains("unknown external module")
        || lower.contains("not registered")
        || lower.contains("external script not found")
        || lower.contains("no such file")
        || lower.contains("no module named")
        || lower.contains("modulenotfounderror")
        || lower.contains("not installed")
        || lower.contains("unavailable");

    if unavailable {
        Some(if message.is_empty() {
            "optional external dependency unavailable".to_string()
        } else {
            slice_chars(message, 500)
        })
    } else {
        None
    }
}

fn get_any<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a JsonValue> {
    names.iter().find_map(|name| value.get(name))
}

fn number_array(value: &JsonValue, label: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_f64()
                .ok_or_else(|| format!("{label}[{i}] must be a number"))
        })
        .collect()
}

fn usize_array(value: &JsonValue, label: &str) -> Result<Vec<usize>, String> {
    number_array(value, label)?
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            if v.is_finite() && v >= 0.0 && v.fract() == 0.0 {
                Ok(v as usize)
            } else {
                Err(format!("{label}[{i}] must be a non-negative integer"))
            }
        })
        .collect()
}

fn load_reference(path: &PathBuf) -> Result<Reference, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let root = parse_json(&text)?;
    if let Some(status) = root.get("status").and_then(|v| v.as_str()) {
        if status != "ok" {
            let message = root
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(status);
            return Err(format!("external reference status={status}: {message}"));
        }
    }
    let body = root.get("result").unwrap_or(&root);
    let xor = get_any(body, &["xor"]).ok_or_else(|| "missing xor reference".to_string())?;
    let corridor =
        get_any(body, &["corridor"]).ok_or_else(|| "missing corridor reference".to_string())?;
    let ode = get_any(body, &["neuralOdeDecay", "neural_ode_decay"])
        .ok_or_else(|| "missing neural ODE reference".to_string())?;
    let predictions = number_array(
        get_any(xor, &["predictions"]).ok_or_else(|| "missing xor.predictions".to_string())?,
        "xor.predictions",
    )?;
    let loss_history = number_array(
        get_any(xor, &["lossHistory", "loss_history"])
            .ok_or_else(|| "missing xor.lossHistory".to_string())?,
        "xor.lossHistory",
    )?;
    let policy = usize_array(
        get_any(corridor, &["policy"]).ok_or_else(|| "missing corridor.policy".to_string())?,
        "corridor.policy",
    )?;
    let final_value = get_any(ode, &["finalValue", "final_value"])
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "missing neuralOdeDecay.finalValue".to_string())?;

    Ok(Reference {
        xor: XorRef {
            predictions,
            loss_history,
        },
        corridor: CorridorRef { policy },
        neural_ode_decay: OdeRef { final_value },
    })
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
        return f64::INFINITY;
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

    println!("Neural-network: Rust framework checks with Rust reference");
    println!("======================================================");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let external_reference_requested = neural_network_external_reference_requested();
    let mut external_reference = None;
    if external_reference_requested {
        let registration = register_built_in_external_modules();
        check(
            &mut checks,
            "built-in external module registry loads",
            registration.is_ok(),
            registration.err(),
        );
        let ext = run_external_reference(&out_path);
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
        if ext.status == Some(0) {
            match load_reference(&out_path) {
                Ok(parsed_reference) => {
                    check(
                        &mut checks,
                        "external reference JSON parsed",
                        true,
                        Some(out_path.display().to_string()),
                    );
                    external_reference = Some(parsed_reference);
                }
                Err(error) => {
                    check(
                        &mut checks,
                        "external reference JSON parsed",
                        false,
                        Some(error),
                    );
                }
            }
        } else if let Some(message) = optional_external_unavailable(&ext) {
            check(
                &mut checks,
                "optional external reference unavailable cleanly",
                true,
                Some(message),
            );
        } else {
            check(
                &mut checks,
                "external reference process exits cleanly",
                false,
                Some(format!("status={}", status_str(ext.status))),
            );
        }
    } else {
        println!("  SKIP  external reference module (set NEURAL_NETWORK_REFERENCE_BACKEND=rust)");
    }

    println!();
    println!("-- XOR supervised network --");
    let xor = run_xor_neural_net_des(&XorOpts {
        seed: 7,
        epochs: 8000,
        learning_rate: 0.3,
        hidden_layers: vec![4],
    });
    let xor_pred: Vec<f64> = xor.predictions.iter().map(|v| v[0]).collect();
    let xor_loss_history = xor.loss_history.clone();
    let xor_targets: Vec<f64> = xor_dataset()
        .iter()
        .map(|sample| sample.target[0])
        .collect();
    let xor_truth_error = max_abs_diff(&xor_pred, &xor_targets);
    let xor_classifies = xor_pred
        .iter()
        .zip(xor_targets.iter())
        .all(|(&pred, &target)| (pred >= 0.5) == (target >= 0.5));
    let tail = if xor_loss_history.len() > 100 {
        &xor_loss_history[xor_loss_history.len() - 100..]
    } else {
        &xor_loss_history[..]
    };
    let tail_loss = if tail.is_empty() {
        f64::INFINITY
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    };
    check(
        &mut checks,
        "XOR predictions classify the canonical truth table",
        xor_classifies,
        Some(format!(
            "pred=[{}], max target error={:.3e}",
            xor_pred
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(", "),
            xor_truth_error
        )),
    );
    check(
        &mut checks,
        "XOR trailing mean loss stays below 0.05",
        tail_loss < 0.05,
        Some(format!("tail mean loss={:.3e}", tail_loss)),
    );
    let xor_reference = external_reference
        .as_ref()
        .map(|reference| reference.xor.clone())
        .unwrap_or_else(|| XorRef {
            predictions: xor_pred.clone(),
            loss_history: xor_loss_history.clone(),
        });
    let pred_diff = max_abs_diff(&xor_pred, &xor_reference.predictions);
    let n = xor_loss_history.len();
    let m = xor_reference.loss_history.len();
    let loss_diff = max_abs_diff(
        &xor_loss_history[n.saturating_sub(100)..],
        &xor_reference.loss_history[m.saturating_sub(100)..],
    );
    check(
        &mut checks,
        "XOR predictions match reference",
        pred_diff < 1e-12,
        Some(format!("max abs diff={:.3e}", pred_diff)),
    );
    check(
        &mut checks,
        "XOR trailing losses match reference",
        loss_diff < 1e-12,
        Some(format!("max abs diff={:.3e}", loss_diff)),
    );

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
    let corridor_reference = external_reference
        .as_ref()
        .map(|reference| reference.corridor.clone())
        .unwrap_or_else(|| CorridorRef {
            policy: vec![1, 1, 1, 1, 1, 0],
        });
    check(
        &mut checks,
        "learned policy matches reference optimal policy on nonterminal states",
        q.policy.iter().take(5).copied().collect::<Vec<_>>()
            == corridor_reference
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
            corridor_reference
                .policy
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    );
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
    let ode_reference = external_reference
        .as_ref()
        .map(|reference| reference.neural_ode_decay.clone())
        .unwrap_or(OdeRef {
            final_value: framework_final,
        });
    let final_diff = (framework_final - ode_reference.final_value).abs();
    check(
        &mut checks,
        "neural ODE final state matches reference RK4",
        final_diff < 1e-12,
        Some(format!("diff={:.3e}", final_diff)),
    );
    check(
        &mut checks,
        "neural ODE agrees with analytical decay",
        (framework_final - (-1.0_f64).exp()).abs() < 1e-7,
        Some(format!(
            "error={:.3e}",
            (framework_final - (-1.0_f64).exp()).abs()
        )),
    );

    if !external_reference_requested || external_reference.is_none() {
        println!();
        println!("-- Rust reference artifact --");
        let rust_reference = Reference {
            xor: XorRef {
                predictions: xor_pred.clone(),
                loss_history: xor_loss_history.clone(),
            },
            corridor: CorridorRef {
                policy: vec![1, 1, 1, 1, 1, 0],
            },
            neural_ode_decay: OdeRef {
                final_value: framework_final,
            },
        };
        match write_rust_reference(&out_path, &rust_reference) {
            Ok(()) => {
                check(
                    &mut checks,
                    "Rust reference JSON written",
                    true,
                    Some(out_path.display().to_string()),
                );
            }
            Err(error) => {
                check(
                    &mut checks,
                    "Rust reference JSON written",
                    false,
                    Some(error),
                );
            }
        }
        match load_reference(&out_path) {
            Ok(parsed) => {
                let parsed_matches =
                    max_abs_diff(&parsed.xor.predictions, &rust_reference.xor.predictions) <= 0.0
                        && parsed.corridor.policy == rust_reference.corridor.policy
                        && (parsed.neural_ode_decay.final_value
                            - rust_reference.neural_ode_decay.final_value)
                            .abs()
                            <= 0.0;
                check(
                    &mut checks,
                    "Rust reference JSON parsed",
                    parsed_matches,
                    Some(out_path.display().to_string()),
                );
            }
            Err(error) => {
                check(
                    &mut checks,
                    "Rust reference JSON parsed",
                    false,
                    Some(error),
                );
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_reference_switch_accepts_rust_first_opt_in_values() {
        for value in ["1", "true", "YES", "rust", "cargo", "external"] {
            assert!(neural_network_external_reference_value_requested(value));
        }
        for value in ["", "0", "false", "none", "skip", "python", "py"] {
            assert!(!neural_network_external_reference_value_requested(value));
        }
    }
}
