//! Blocking live server for the 2D soccer simulation.

use crate::des::general::soccer::{run_live_soccer_server, SoccerLiveServerConfig};

fn env_positive_f64(primary: &str, fallback: &str) -> Option<f64> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_positive_u64(primary: &str, fallback: &str) -> Option<u64> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn env_bool(primary: &str, fallback: &str) -> Option<bool> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn env_nonnegative_f64(primary: &str, fallback: &str) -> Option<f64> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn env_positive_usize(primary: &str, fallback: &str) -> Option<usize> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub fn run() {
    let mut cfg = SoccerLiveServerConfig::default();
    if let Ok(host) = std::env::var("SOCCER_LIVE_HOST") {
        cfg.host = host;
    }
    if let Ok(port) = std::env::var("SOCCER_LIVE_PORT") {
        if let Ok(port) = port.parse::<u16>() {
            cfg.port = port;
        }
    }
    if let Some(duration) =
        env_positive_f64("SOCCER_MATCH_DURATION_SECONDS", "SOCCER_DURATION_SECONDS")
    {
        cfg.match_config.duration_seconds = duration;
    }
    if let Some(dt) = env_positive_f64("SOCCER_MATCH_DT_SECONDS", "SOCCER_DT_SECONDS") {
        cfg.match_config.dt_seconds = dt;
    }
    if let Some(period_count) = env_positive_usize("SOCCER_MATCH_HALVES", "SOCCER_HALVES") {
        cfg.match_config.period_count = period_count;
    }
    if let Some(period_break_recovery_seconds) = env_nonnegative_f64(
        "SOCCER_MATCH_PERIOD_BREAK_RECOVERY_SECONDS",
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
    ) {
        cfg.match_config.period_break_recovery_seconds = period_break_recovery_seconds;
    }
    if let Ok(path) =
        std::env::var("SOCCER_LIVE_POLICY_PATH").or_else(|_| std::env::var("SOCCER_POLICY_PATH"))
    {
        if !path.trim().is_empty() {
            cfg.policy_disk_path = path;
        }
    }
    if let Ok(path) =
        std::env::var("SOCCER_LIVE_MOMENT_PATH").or_else(|_| std::env::var("SOCCER_MOMENT_PATH"))
    {
        if !path.trim().is_empty() {
            cfg.moment_disk_path = path;
        }
    }
    if let Some(autoload) = env_bool("SOCCER_LIVE_AUTOLOAD_POLICY", "SOCCER_AUTOLOAD_POLICY") {
        cfg.autoload_team_policy = autoload;
    }
    if let Some(autosave) = env_bool("SOCCER_LIVE_AUTOSAVE_POLICY", "SOCCER_AUTOSAVE_POLICY") {
        cfg.autosave_team_policy = autosave;
    }
    if let Some(interval) = env_positive_u64(
        "SOCCER_LIVE_POLICY_AUTOSAVE_INTERVAL_TICKS",
        "SOCCER_POLICY_AUTOSAVE_INTERVAL_TICKS",
    ) {
        cfg.policy_autosave_interval_ticks = interval;
    }
    run_live_soccer_server(cfg).expect("run live soccer server");
}
