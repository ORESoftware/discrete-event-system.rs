//! Run soccer self-play through the des-rs HTTP training endpoint.
//!
//! This is the Rust equivalent of the JSON-building/parsing glue in
//! `scripts/soccer_self_play_server.sh`; the script keeps environment defaults
//! and run metadata, while this binary owns payload construction, curl dispatch,
//! and response artifact validation.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

#[derive(Debug)]
struct Args {
    endpoint: String,
    payload_path: PathBuf,
    response_path: PathBuf,
    artifact_path: PathBuf,
    learned_params_path: PathBuf,
    episode_log_path: PathBuf,
    server_artifact_path: String,
    server_learned_params_path: String,
    auth_header_name: String,
    auth_value: Option<String>,
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn next_arg<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let value = args
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))?;
    value
        .into_string()
        .map_err(|_| invalid_input(format!("{flag} must be valid UTF-8")).into())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut endpoint = None::<String>;
    let mut payload_path = None::<PathBuf>;
    let mut response_path = None::<PathBuf>;
    let mut artifact_path = None::<PathBuf>;
    let mut learned_params_path = None::<PathBuf>;
    let mut episode_log_path = None::<PathBuf>;
    let mut server_artifact_path = None::<String>;
    let mut server_learned_params_path = None::<String>;
    let mut auth_header_name = "Auth".to_string();
    let mut auth_value = None::<String>;

    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let flag = arg
            .into_string()
            .map_err(|_| invalid_input("argument flag must be valid UTF-8"))?;
        match flag.as_str() {
            "--endpoint" => endpoint = Some(next_arg(&mut args, "--endpoint")?),
            "--payload" => payload_path = Some(PathBuf::from(next_arg(&mut args, "--payload")?)),
            "--response" => response_path = Some(PathBuf::from(next_arg(&mut args, "--response")?)),
            "--artifact" => artifact_path = Some(PathBuf::from(next_arg(&mut args, "--artifact")?)),
            "--learned-params" => {
                learned_params_path = Some(PathBuf::from(next_arg(&mut args, "--learned-params")?))
            }
            "--episode-log" => {
                episode_log_path = Some(PathBuf::from(next_arg(&mut args, "--episode-log")?))
            }
            "--server-artifact-path" => {
                server_artifact_path = Some(next_arg(&mut args, "--server-artifact-path")?)
            }
            "--server-learned-params-path" => {
                server_learned_params_path =
                    Some(next_arg(&mut args, "--server-learned-params-path")?)
            }
            "--auth-header-name" => auth_header_name = next_arg(&mut args, "--auth-header-name")?,
            "--auth-value" => {
                let value = next_arg(&mut args, "--auth-value")?;
                if !value.trim().is_empty() {
                    auth_value = Some(value);
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: main_soccer_learning_server --endpoint URL --payload PATH --response PATH --artifact PATH --learned-params PATH --episode-log PATH --server-artifact-path PATH --server-learned-params-path PATH [--auth-header-name NAME] [--auth-value VALUE]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {flag}")).into()),
        }
    }

    Ok(Args {
        endpoint: endpoint.ok_or_else(|| invalid_input("--endpoint is required"))?,
        payload_path: payload_path.ok_or_else(|| invalid_input("--payload is required"))?,
        response_path: response_path.ok_or_else(|| invalid_input("--response is required"))?,
        artifact_path: artifact_path.ok_or_else(|| invalid_input("--artifact is required"))?,
        learned_params_path: learned_params_path
            .ok_or_else(|| invalid_input("--learned-params is required"))?,
        episode_log_path: episode_log_path
            .ok_or_else(|| invalid_input("--episode-log is required"))?,
        server_artifact_path: server_artifact_path
            .ok_or_else(|| invalid_input("--server-artifact-path is required"))?,
        server_learned_params_path: server_learned_params_path
            .ok_or_else(|| invalid_input("--server-learned-params-path is required"))?,
        auth_header_name,
        auth_value,
    })
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<f64>()
            .map_err(|err| invalid_input(format!("{name}={raw:?} is not a float: {err}")).into()),
        Err(_) => Ok(default),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(raw) => raw.parse::<usize>().map_err(|err| {
            invalid_input(format!(
                "{name}={raw:?} is not a non-negative integer: {err}"
            ))
            .into()
        }),
        Err(_) => Ok(default),
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<u32>()
            .map_err(|err| invalid_input(format!("{name}={raw:?} is not a u32: {err}")).into()),
        Err(_) => Ok(default),
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => default,
    }
}

fn build_payload(args: &Args) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "episodes": env_usize("SOCCER_GAMES", 100)?,
        "minutes": env_f64("SOCCER_MINUTES", 90.0)?,
        "periodCount": env_usize("SOCCER_HALVES", 2)?,
        "periodBreakRecoverySeconds": env_f64("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS", 900.0)?,
        "dtSeconds": env_f64("SOCCER_DT_SECONDS", 1.0)?,
        "learningIntervalTicks": env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?,
        "seed": env_u32("SOCCER_SEED", 2026)?,
        "options": {
            "alpha": env_f64("SOCCER_ALPHA", 0.20)?,
            "gamma": env_f64("SOCCER_GAMMA", 0.96)?,
        },
        "tacticalLearning": {
            "attackSpacingDeltaWeight": env_f64("SOCCER_ATTACK_SPACING_DELTA_WEIGHT", 0.22)?,
            "attackSpacingScoreWeight": env_f64("SOCCER_ATTACK_SPACING_SCORE_WEIGHT", 0.06)?,
            "attackWidthDeltaWeight": env_f64("SOCCER_ATTACK_WIDTH_DELTA_WEIGHT", 0.52)?,
            "attackWidthScoreWeight": env_f64("SOCCER_ATTACK_WIDTH_SCORE_WEIGHT", 0.14)?,
            "attackFlankLaneWeight": env_f64("SOCCER_ATTACK_FLANK_LANE_WEIGHT", 0.28)?,
            "defenseSpacingDeltaWeight": env_f64("SOCCER_DEFENSE_SPACING_DELTA_WEIGHT", 0.08)?,
            "defenseSpacingScoreWeight": env_f64("SOCCER_DEFENSE_SPACING_SCORE_WEIGHT", 0.04)?,
            "defenseContractDeltaWeight": env_f64("SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT", 0.42)?,
            "defenseCompactnessScoreWeight": env_f64("SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT", 0.14)?,
        },
        "artifactPath": args.server_artifact_path,
        "learnedParamsPath": args.server_learned_params_path,
        "importIntoSession": env_bool("SOCCER_IMPORT_INTO_SESSION", true),
    }))
}

fn ensure_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    Ok(())
}

fn write_payload(path: &Path, payload: &Value) -> Result<(), Box<dyn Error>> {
    write_json_pretty(path, payload)
}

fn run_curl(args: &Args) -> Result<(), Box<dyn Error>> {
    ensure_parent(&args.response_path)?;
    let curl_bin = env_string("CURL_BIN", "curl");
    let payload_arg = format!("@{}", args.payload_path.display());
    let mut command = Command::new(curl_bin);
    command
        .arg("-fsS")
        .arg("-X")
        .arg("POST")
        .arg(&args.endpoint)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("--data-binary")
        .arg(payload_arg)
        .arg("-o")
        .arg(&args.response_path);
    if let Some(auth_value) = args.auth_value.as_deref() {
        command
            .arg("-H")
            .arg(format!("{}: {}", args.auth_header_name, auth_value));
    }
    let status = command.status()?;
    if !status.success() {
        return Err(invalid_data(format!("curl exited with status {status}")).into());
    }
    Ok(())
}

fn write_episode_log(path: &Path, episodes: &[Value]) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    for episode in episodes {
        serde_json::to_writer(&mut writer, episode)?;
        writeln!(writer)?;
    }
    Ok(())
}

fn extract_response(args: &Args) -> Result<(usize, usize, usize), Box<dyn Error>> {
    let raw = fs::read_to_string(&args.response_path)?;
    let response: Value = serde_json::from_str(&raw)?;
    if response.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(invalid_data(format!(
            "server returned error response: {}",
            response.get("error").unwrap_or(&response)
        ))
        .into());
    }
    let artifact = response
        .get("artifact")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid_data("server response did not include an artifact object"))?;
    let learned_params = response
        .get("learnedParams")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid_data("server response did not include learnedParams"))?;

    write_json_pretty(&args.artifact_path, artifact)?;
    write_json_pretty(&args.learned_params_path, learned_params)?;

    let episodes = artifact
        .get("episodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    write_episode_log(&args.episode_log_path, episodes)?;

    let home_entries = artifact
        .get("homeEntries")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let away_entries = artifact
        .get("awayEntries")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    Ok((episodes.len(), home_entries, away_entries))
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let payload = build_payload(&args)?;
    write_payload(&args.payload_path, &payload)?;
    run_curl(&args)?;
    let (episodes, home_entries, away_entries) = extract_response(&args)?;

    println!("server_response={}", args.response_path.display());
    println!("artifact={}", args.artifact_path.display());
    println!("learned_params={}", args.learned_params_path.display());
    println!("episode_log={}", args.episode_log_path.display());
    println!("episodes={episodes}");
    println!("home_entries={home_entries}");
    println!("away_entries={away_entries}");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("main_soccer_learning_server: {err}");
        std::process::exit(1);
    }
}
