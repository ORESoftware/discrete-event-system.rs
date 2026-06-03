//! Blocking live server for the 2D soccer simulation.

use crate::des::general::soccer::{run_live_soccer_server, SoccerLiveServerConfig};

fn env_positive_f64(primary: &str, fallback: &str) -> Option<f64> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
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
    run_live_soccer_server(cfg).expect("run live soccer server");
}
