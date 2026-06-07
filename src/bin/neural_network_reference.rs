use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use des_engine::des::general::neural_network::{
    run_xor_neural_net_des, solve_neural_ode, ActivationName, DenseLayerConfig, FeedForwardNetwork,
    NeuralODEOptions, NeuralODESolverName, XorNeuralNetOptions,
};
use des_engine::des::general::rl_environments::Corridor;
use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Clone, Debug)]
struct Args {
    out: Option<PathBuf>,
    seed: u32,
    xor_epochs: usize,
    xor_lr: f64,
    corridor_length: usize,
    corridor_gamma: f64,
    ode_rate: f64,
    ode_y0: f64,
    ode_t1: f64,
    ode_dt: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            out: None,
            seed: 7,
            xor_epochs: 8000,
            xor_lr: 0.3,
            corridor_length: 6,
            corridor_gamma: 0.95,
            ode_rate: 0.5,
            ode_y0: 1.0,
            ode_t1: 2.0,
            ode_dt: 0.05,
        }
    }
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--out PATH] [--seed N] [--xor-epochs N] [--xor-lr F] [--corridor-length N] [--corridor-gamma F] [--ode-rate F] [--ode-y0 F] [--ode-t1 F] [--ode-dt F]"
    )
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{option} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_usize(program: &str, option: &str, value: String) -> Result<usize, CliError> {
    value.parse::<usize>().map_err(|err| {
        CliError(format!(
            "{option} must be an unsigned integer, got {value:?}: {err}\n{}",
            usage(program)
        ))
    })
}

fn parse_u32(program: &str, option: &str, value: String) -> Result<u32, CliError> {
    value.parse::<u32>().map_err(|err| {
        CliError(format!(
            "{option} must be a u32, got {value:?}: {err}\n{}",
            usage(program)
        ))
    })
}

fn parse_f64(program: &str, option: &str, value: String) -> Result<f64, CliError> {
    let parsed = value.parse::<f64>().map_err(|err| {
        CliError(format!(
            "{option} must be a finite number, got {value:?}: {err}\n{}",
            usage(program)
        ))
    })?;
    if !parsed.is_finite() {
        return Err(CliError(format!(
            "{option} must be finite, got {value:?}\n{}",
            usage(program)
        )));
    }
    Ok(parsed)
}

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<Args, CliError> {
    let mut parsed = Args::default();
    let mut values = args.into_iter();
    while let Some(raw) = values.next() {
        if raw == "-h" || raw == "--help" {
            return Err(CliError(usage(program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        match key.as_str() {
            "--out" => {
                parsed.out = Some(PathBuf::from(next_option_value(
                    program,
                    "--out",
                    inline_value,
                    &mut values,
                )?));
            }
            "--seed" => {
                let value = next_option_value(program, "--seed", inline_value, &mut values)?;
                parsed.seed = parse_u32(program, "--seed", value)?;
            }
            "--xor-epochs" => {
                let value = next_option_value(program, "--xor-epochs", inline_value, &mut values)?;
                parsed.xor_epochs = parse_usize(program, "--xor-epochs", value)?;
            }
            "--xor-lr" => {
                let value = next_option_value(program, "--xor-lr", inline_value, &mut values)?;
                parsed.xor_lr = parse_f64(program, "--xor-lr", value)?;
            }
            "--corridor-length" => {
                let value =
                    next_option_value(program, "--corridor-length", inline_value, &mut values)?;
                parsed.corridor_length = parse_usize(program, "--corridor-length", value)?.max(2);
            }
            "--corridor-gamma" => {
                let value =
                    next_option_value(program, "--corridor-gamma", inline_value, &mut values)?;
                parsed.corridor_gamma = parse_f64(program, "--corridor-gamma", value)?;
            }
            "--ode-rate" => {
                let value = next_option_value(program, "--ode-rate", inline_value, &mut values)?;
                parsed.ode_rate = parse_f64(program, "--ode-rate", value)?;
            }
            "--ode-y0" => {
                let value = next_option_value(program, "--ode-y0", inline_value, &mut values)?;
                parsed.ode_y0 = parse_f64(program, "--ode-y0", value)?;
            }
            "--ode-t1" => {
                let value = next_option_value(program, "--ode-t1", inline_value, &mut values)?;
                parsed.ode_t1 = parse_f64(program, "--ode-t1", value)?;
            }
            "--ode-dt" => {
                let value = next_option_value(program, "--ode-dt", inline_value, &mut values)?;
                parsed.ode_dt = parse_f64(program, "--ode-dt", value)?;
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )));
            }
        }
    }
    if parsed.ode_dt <= 0.0 {
        return Err(CliError("--ode-dt must be positive".to_string()));
    }
    if !(0.0..=1.0).contains(&parsed.corridor_gamma) {
        return Err(CliError(
            "--corridor-gamma must be in the interval [0, 1]".to_string(),
        ));
    }
    Ok(parsed)
}

fn run(args: &Args) -> Value {
    let xor = run_xor_neural_net_des(XorNeuralNetOptions {
        epochs: Some(args.xor_epochs),
        learning_rate: Some(args.xor_lr),
        seed: Some(args.seed),
        hidden_layers: Some(vec![4]),
        samples_per_tick: None,
        shuffle_each_epoch: None,
    });
    let xor_predictions = xor
        .predictions
        .iter()
        .filter_map(|row| row.first().copied())
        .collect::<Vec<_>>();

    let corridor = Corridor::new(args.corridor_length, 0);
    let policy = corridor
        .optimal_v(args.corridor_gamma, 1e-9, 5000)
        .pi
        .into_iter()
        .collect::<Vec<_>>();

    let net = FeedForwardNetwork::new(vec![DenseLayerConfig {
        weights: vec![vec![-args.ode_rate]],
        biases: vec![0.0],
        activation: ActivationName::Linear,
    }]);
    let trace = solve_neural_ode(
        &net,
        &NeuralODEOptions {
            y0: vec![args.ode_y0],
            t0: 0.0,
            t1: args.ode_t1,
            dt: args.ode_dt,
            solver: Some(NeuralODESolverName::Rk4),
            include_time: Some(false),
            rk45: None,
        },
    );
    let final_value = trace
        .y
        .last()
        .and_then(|row| row.first())
        .copied()
        .unwrap_or(args.ode_y0);

    json!({
        "status": "ok",
        "backend": "rust",
        "result": {
            "xor": {
                "predictions": xor_predictions,
                "lossHistory": xor.loss_history,
            },
            "corridor": {
                "policy": policy,
            },
            "neuralOdeDecay": {
                "finalValue": final_value,
            },
        },
    })
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "failed",
        "backend": "rust",
        "message": message.into(),
        "result": {},
    })
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "neural_network_reference".to_string());
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage(&program));
        return;
    }
    let output = match parse_args(&program, raw_args.into_iter().skip(1)).map(|args| {
        let output = run(&args);
        if let Some(path) = &args.out {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)
                    .map_err(|err| CliError(format!("create {}: {err}", parent.display())))?;
            }
            fs::write(
                path,
                format!(
                    "{}\n",
                    serde_json::to_string_pretty(&output).expect("serialize pretty output")
                ),
            )
            .map_err(|err| CliError(format!("write {}: {err}", path.display())))?;
            return Ok::<Value, CliError>(json!({
                "status": output.get("status").cloned().unwrap_or_else(|| json!("ok")),
                "backend": output.get("backend").cloned().unwrap_or_else(|| json!("rust")),
                "out": path.display().to_string(),
            }));
        }
        Ok::<Value, CliError>(output)
    }) {
        Ok(Ok(output)) => output,
        Ok(Err(err)) | Err(err) => {
            let output = error_json(err.to_string());
            println!(
                "{}",
                serde_json::to_string(&output).expect("serialize error")
            );
            std::process::exit(1);
        }
    };
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "{}",
        serde_json::to_string(&output).expect("serialize output")
    )
    .expect("write stdout");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static NEURAL_NETWORK_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }

        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn neural_network_python_off_guards() -> Vec<EnvVarGuard> {
        vec![
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-neural-network"),
            EnvVarGuard::set("PYTHON", "/definitely/not-python-for-neural-network"),
            EnvVarGuard::set(
                "PYTORCH_PYTHON",
                "/definitely/not-python-for-neural-network",
            ),
            EnvVarGuard::set(
                "TENSORFLOW_PYTHON",
                "/definitely/not-python-for-neural-network",
            ),
            EnvVarGuard::clear("NEURAL_NETWORK_REFERENCE_FORCE_PYTHON"),
            EnvVarGuard::clear("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON"),
        ]
    }

    #[test]
    fn accepts_external_module_args() {
        let args = parse_args(
            "neural_network_reference",
            vec![
                "--out=/tmp/neural-reference.json".to_string(),
                "--seed".to_string(),
                "9".to_string(),
                "--xor-epochs=12".to_string(),
                "--xor-lr=0.2".to_string(),
                "--corridor-length=4".to_string(),
                "--corridor-gamma=0.9".to_string(),
                "--ode-rate=0.25".to_string(),
                "--ode-y0=2.0".to_string(),
                "--ode-t1=1.5".to_string(),
                "--ode-dt=0.1".to_string(),
            ],
        )
        .expect("parse args");

        assert_eq!(args.seed, 9);
        assert_eq!(args.xor_epochs, 12);
        assert_eq!(args.corridor_length, 4);
        assert_eq!(args.out, Some(PathBuf::from("/tmp/neural-reference.json")));
    }

    #[test]
    fn emits_reference_json_shape() {
        let output = run(&Args {
            xor_epochs: 8,
            corridor_length: 4,
            ode_t1: 0.2,
            ode_dt: 0.1,
            ..Args::default()
        });

        assert_eq!(output["status"], "ok");
        assert_eq!(output["backend"], "rust");
        assert_eq!(
            output["result"]["xor"]["predictions"]
                .as_array()
                .expect("xor predictions")
                .len(),
            4
        );
        assert_eq!(
            output["result"]["corridor"]["policy"]
                .as_array()
                .expect("corridor policy")
                .len(),
            4
        );
        assert!(output["result"]["neuralOdeDecay"]["finalValue"]
            .as_f64()
            .expect("ode final")
            .is_finite());
    }

    #[test]
    fn reference_generation_ignores_python_env_and_runs_in_rust() {
        let _env_lock = NEURAL_NETWORK_REFERENCE_ENV_LOCK
            .lock()
            .expect("neural-network env lock");
        let _guards = neural_network_python_off_guards();

        let output = run(&Args {
            seed: 11,
            xor_epochs: 6,
            corridor_length: 5,
            ode_y0: 2.0,
            ode_t1: 0.3,
            ode_dt: 0.1,
            ..Args::default()
        });

        assert_eq!(output["status"], "ok");
        assert_eq!(output["backend"], "rust");
        assert_eq!(
            output["result"]["xor"]["predictions"]
                .as_array()
                .expect("xor predictions")
                .len(),
            4
        );
        assert_eq!(
            output["result"]["corridor"]["policy"]
                .as_array()
                .expect("corridor policy")
                .len(),
            5
        );
        let final_value = output["result"]["neuralOdeDecay"]["finalValue"]
            .as_f64()
            .expect("ode final");
        assert!(final_value.is_finite());
        assert!((0.0..2.0).contains(&final_value));
    }

    #[test]
    fn force_python_env_still_uses_rust_native_reference() {
        let _env_lock = NEURAL_NETWORK_REFERENCE_ENV_LOCK
            .lock()
            .expect("neural-network env lock");
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-neural-network");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-neural-network");
        let _torch_guard = EnvVarGuard::set(
            "PYTORCH_PYTHON",
            "/definitely/not-python-for-neural-network",
        );
        let _tensorflow_guard = EnvVarGuard::set(
            "TENSORFLOW_PYTHON",
            "/definitely/not-python-for-neural-network",
        );
        let _force_guard = EnvVarGuard::set("NEURAL_NETWORK_REFERENCE_FORCE_PYTHON", "1");
        let _global_force_guard = EnvVarGuard::set("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON", "1");

        let output = run(&Args {
            seed: 17,
            xor_epochs: 4,
            corridor_length: 4,
            ode_y0: 1.5,
            ode_t1: 0.2,
            ode_dt: 0.1,
            ..Args::default()
        });

        assert_eq!(output["status"], "ok");
        assert_eq!(output["backend"], "rust");
        assert_eq!(
            output["result"]["xor"]["predictions"]
                .as_array()
                .expect("xor predictions")
                .len(),
            4
        );
        assert_eq!(
            output["result"]["corridor"]["policy"]
                .as_array()
                .expect("corridor policy")
                .len(),
            4
        );
        assert!(output["result"]["neuralOdeDecay"]["finalValue"]
            .as_f64()
            .expect("ode final")
            .is_finite());
        assert!(!serde_json::to_string(&output)
            .expect("reference output json")
            .contains("/definitely/not-python-for-neural-network"));
    }
}
