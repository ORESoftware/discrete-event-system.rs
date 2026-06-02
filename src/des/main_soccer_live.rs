//! Blocking live server for the 2D soccer simulation.

use crate::des::general::soccer::{run_live_soccer_server, SoccerLiveServerConfig};

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
    run_live_soccer_server(cfg).expect("run live soccer server");
}
