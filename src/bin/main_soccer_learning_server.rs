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
use std::process::{Command, Stdio};

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
    auth_env_name: Option<String>,
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
    let mut auth_env_name = Some("DES_RS_AUTH".to_string());
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
            "--auth-env-name" => {
                let value = next_arg(&mut args, "--auth-env-name")?;
                auth_env_name = if value.trim().is_empty() {
                    None
                } else {
                    Some(value)
                };
            }
            "--auth-value" => {
                let value = next_arg(&mut args, "--auth-value")?;
                if !value.trim().is_empty() {
                    auth_value = Some(value);
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: main_soccer_learning_server --endpoint URL --payload PATH --response PATH --artifact PATH --learned-params PATH --episode-log PATH --server-artifact-path PATH --server-learned-params-path PATH [--auth-header-name NAME] [--auth-env-name NAME]"
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
        auth_env_name,
        auth_value,
    })
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => {
            let parsed = raw.parse::<f64>().map_err(|err| {
                invalid_input(format!("{name}={raw:?} is not a finite number: {err}"))
            })?;
            if !parsed.is_finite() {
                return Err(invalid_input(format!("{name}={raw:?} is not finite")).into());
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => raw.parse::<usize>().map_err(|err| {
            invalid_input(format!(
                "{name}={raw:?} is not a non-negative integer: {err}"
            ))
            .into()
        }),
        None => Ok(default),
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => raw
            .parse::<u32>()
            .map_err(|err| invalid_input(format!("{name}={raw:?} is not a u32: {err}")).into()),
        None => Ok(default),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => Ok(true),
            "0" | "false" | "no" | "n" | "off" => Ok(false),
            _ => Err(invalid_input(format!("{name}={raw:?} is not a boolean")).into()),
        },
        None => Ok(default),
    }
}

fn validate_payload_settings(
    episodes: usize,
    minutes: f64,
    period_count: usize,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    alpha: f64,
    gamma: f64,
    tactical_weights: &[(&str, f64)],
) -> Result<(), Box<dyn Error>> {
    if episodes == 0 {
        return Err(invalid_input("SOCCER_GAMES must be at least 1").into());
    }
    if !minutes.is_finite() || minutes <= 0.0 || minutes > 24.0 * 60.0 {
        return Err(invalid_input("SOCCER_MINUTES must be finite and in (0, 1440]").into());
    }
    if !(1..=8).contains(&period_count) {
        return Err(invalid_input("SOCCER_HALVES must be between 1 and 8").into());
    }
    if !period_break_recovery_seconds.is_finite()
        || !(0.0..=60.0 * 60.0).contains(&period_break_recovery_seconds)
    {
        return Err(invalid_input(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS must be finite and in [0, 3600]",
        )
        .into());
    }
    if !dt_seconds.is_finite() || !(0.01..=5.0).contains(&dt_seconds) {
        return Err(invalid_input("SOCCER_DT_SECONDS must be finite and in [0.01, 5.0]").into());
    }
    if learning_interval_ticks == 0 {
        return Err(invalid_input("SOCCER_LEARNING_INTERVAL_TICKS must be at least 1").into());
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(invalid_input("SOCCER_ALPHA must be finite and in [0, 1]").into());
    }
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(invalid_input("SOCCER_GAMMA must be finite and in [0, 1]").into());
    }
    for (name, value) in tactical_weights {
        if !value.is_finite() {
            return Err(invalid_input(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn build_payload(args: &Args) -> Result<Value, Box<dyn Error>> {
    let episodes = env_usize("SOCCER_GAMES", 100)?;
    let minutes = env_f64("SOCCER_MINUTES", 90.0)?;
    let period_count = env_usize("SOCCER_HALVES", 2)?;
    let period_break_recovery_seconds = env_f64("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS", 900.0)?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
    let seed = env_u32("SOCCER_SEED", 2026)?;
    let alpha = env_f64("SOCCER_ALPHA", 0.20)?;
    let gamma = env_f64("SOCCER_GAMMA", 0.96)?;
    let attack_spacing_delta_weight = env_f64("SOCCER_ATTACK_SPACING_DELTA_WEIGHT", 0.22)?;
    let attack_spacing_score_weight = env_f64("SOCCER_ATTACK_SPACING_SCORE_WEIGHT", 0.06)?;
    let attack_width_delta_weight = env_f64("SOCCER_ATTACK_WIDTH_DELTA_WEIGHT", 0.52)?;
    let attack_width_score_weight = env_f64("SOCCER_ATTACK_WIDTH_SCORE_WEIGHT", 0.14)?;
    let attack_flank_lane_weight = env_f64("SOCCER_ATTACK_FLANK_LANE_WEIGHT", 0.28)?;
    let defense_spacing_delta_weight = env_f64("SOCCER_DEFENSE_SPACING_DELTA_WEIGHT", 0.08)?;
    let defense_spacing_score_weight = env_f64("SOCCER_DEFENSE_SPACING_SCORE_WEIGHT", 0.04)?;
    let defense_contract_delta_weight = env_f64("SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT", 0.42)?;
    let defense_compactness_score_weight =
        env_f64("SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT", 0.14)?;
    let defense_ball_depth_score_weight = env_f64("SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT", 0.22)?;
    let defense_endline_soft_penalty_weight =
        env_f64("SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT", 0.18)?;
    let defense_endline_hard_penalty_weight =
        env_f64("SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT", 0.90)?;
    let defender_midfielder_press_weight =
        env_f64("SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT", 0.18)?;
    let midfielder_press_weight = env_f64("SOCCER_MIDFIELDER_PRESS_WEIGHT", 0.20)?;
    let tactical_weights = [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            attack_width_score_weight,
        ),
        ("SOCCER_ATTACK_FLANK_LANE_WEIGHT", attack_flank_lane_weight),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            defense_compactness_score_weight,
        ),
        (
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            defense_ball_depth_score_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            defense_endline_soft_penalty_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            defense_endline_hard_penalty_weight,
        ),
        (
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            defender_midfielder_press_weight,
        ),
        ("SOCCER_MIDFIELDER_PRESS_WEIGHT", midfielder_press_weight),
    ];
    validate_payload_settings(
        episodes,
        minutes,
        period_count,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        alpha,
        gamma,
        &tactical_weights,
    )?;
    let import_into_session = env_bool("SOCCER_IMPORT_INTO_SESSION", true)?;
    Ok(json!({
        "episodes": episodes,
        "minutes": minutes,
        "periodCount": period_count,
        "periodBreakRecoverySeconds": period_break_recovery_seconds,
        "dtSeconds": dt_seconds,
        "learningIntervalTicks": learning_interval_ticks,
        "seed": seed,
        "options": {
            "alpha": alpha,
            "gamma": gamma,
        },
        "tacticalLearning": {
            "attackSpacingDeltaWeight": attack_spacing_delta_weight,
            "attackSpacingScoreWeight": attack_spacing_score_weight,
            "attackWidthDeltaWeight": attack_width_delta_weight,
            "attackWidthScoreWeight": attack_width_score_weight,
            "attackFlankLaneWeight": attack_flank_lane_weight,
            "defenseSpacingDeltaWeight": defense_spacing_delta_weight,
            "defenseSpacingScoreWeight": defense_spacing_score_weight,
            "defenseContractDeltaWeight": defense_contract_delta_weight,
            "defenseCompactnessScoreWeight": defense_compactness_score_weight,
            "defenseBallDepthScoreWeight": defense_ball_depth_score_weight,
            "defenseEndlineSoftPenaltyWeight": defense_endline_soft_penalty_weight,
            "defenseEndlineHardPenaltyWeight": defense_endline_hard_penalty_weight,
            "defenderMidfielderPressWeight": defender_midfielder_press_weight,
            "midfielderPressWeight": midfielder_press_weight,
        },
        "artifactPath": &args.server_artifact_path,
        "learnedParamsPath": &args.server_learned_params_path,
        "importIntoSession": import_into_session,
    }))
}

fn ensure_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn sync_parent_best_effort(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|dir| dir.sync_all());
    }
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writeln!(writer)?;
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
    sync_parent_best_effort(path);
    Ok(())
}

fn write_payload(path: &Path, payload: &Value) -> Result<(), Box<dyn Error>> {
    write_json_pretty(path, payload)
}

fn run_curl(args: &Args) -> Result<(), Box<dyn Error>> {
    ensure_parent(&args.response_path)?;
    let curl_bin = env_string("CURL_BIN", "curl");
    let auth_value = resolved_auth_value(args);
    if args.endpoint.starts_with("https://54.91.17.58") && auth_value.is_none() {
        return Err(invalid_input(
            "DES_RS_AUTH is required for the protected https://54.91.17.58 des-rs endpoint",
        )
        .into());
    }
    let curl_config = curl_config(args, auth_value.as_deref())?;
    let mut command = Command::new(curl_bin);
    command.arg("-fsS").arg("-K").arg("-").stdin(Stdio::piped());
    let mut child = command.spawn()?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| invalid_data("failed to open curl stdin"))?;
        stdin.write_all(curl_config.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(invalid_data(format!("curl exited with status {status}")).into());
    }
    Ok(())
}

fn resolved_auth_value(args: &Args) -> Option<String> {
    args.auth_value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| args.auth_env_name.as_deref().and_then(env_value))
}

fn curl_config_quote(value: &str) -> Result<String, Box<dyn Error>> {
    if value.contains('\n') || value.contains('\r') {
        return Err(invalid_input("curl config values must not contain newlines").into());
    }
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    Ok(format!("\"{escaped}\""))
}

fn curl_config(args: &Args, auth_value: Option<&str>) -> Result<String, Box<dyn Error>> {
    let payload_arg = format!("@{}", args.payload_path.display());
    let mut lines = vec![
        format!("url = {}", curl_config_quote(&args.endpoint)?),
        "request = \"POST\"".to_string(),
        "header = \"Content-Type: application/json\"".to_string(),
        format!("data-binary = {}", curl_config_quote(&payload_arg)?),
        format!(
            "output = {}",
            curl_config_quote(&args.response_path.display().to_string())?
        ),
    ];
    if let Some(auth_value) = auth_value {
        lines.push(format!(
            "header = {}",
            curl_config_quote(&format!("{}: {}", args.auth_header_name, auth_value))?
        ));
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn write_episode_log(path: &Path, episodes: &[Value]) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        for episode in episodes {
            serde_json::to_writer(&mut writer, episode)?;
            writeln!(writer)?;
        }
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
    sync_parent_best_effort(path);
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
