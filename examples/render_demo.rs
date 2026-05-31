//! Driver demonstrating the whole plugin flow end-to-end: it uses the
//! `des_engine` plugin API to *spawn the two example plugin binaries*, capture
//! their JSON / JSONL, and render self-contained HTML players to `out/plugin/`.
//!
//! ```bash
//! cargo build --example plugin_queue --example plugin_lp
//! cargo run   --example render_demo
//! # open out/plugin/queue.html and out/plugin/lp.html
//! ```

use des_engine::des::plugin::{
    run_and_render, OutputKind, PlayerKind, PluginManifest, RunSpec, UiControl,
};

fn main() {
    std::fs::create_dir_all("out/plugin").expect("create out/plugin");

    // --- a streaming sim plugin (JSONL frames with shapes) -> sim player ---
    let queue = PluginManifest {
        id: "queue".to_string(),
        name: "M/M/1 Queue (external Rust plugin)".to_string(),
        version: "1.0.0".to_string(),
        description: "An external Rust program streams JSONL frames; the core renders a sim player.".to_string(),
        run: RunSpec::new("target/debug/examples/plugin_queue", &[]),
        output: OutputKind::Jsonl,
        player: PlayerKind::Sim,
        controls: vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
            UiControl::toggle("show_n", "Show n(t)", true, Some("n")),
            UiControl::toggle("show_busy", "Show server busy", true, Some("serverBusy")),
        ],
        title: None,
    };
    let html = run_and_render(&queue).expect("run + render queue plugin");
    std::fs::write("out/plugin/queue.html", &html).expect("write queue.html");

    // --- a single-result plugin (one JSON doc) -> results player ---
    let lp = PluginManifest {
        id: "lp".to_string(),
        name: "LP Solver (external Rust plugin)".to_string(),
        version: "1.0.0".to_string(),
        description: "An external Rust program emits one JSON result; the core renders a results player.".to_string(),
        run: RunSpec::new("target/debug/examples/plugin_lp", &[]),
        output: OutputKind::Json,
        player: PlayerKind::Results,
        controls: vec![
            UiControl::toggle("show_vars", "Variables", true, Some("variables")),
            UiControl::toggle("show_cons", "Constraints", true, Some("constraints")),
            UiControl::toggle("raw", "Show raw JSON", false, Some("rawJson")),
        ],
        title: None,
    };
    let html = run_and_render(&lp).expect("run + render lp plugin");
    std::fs::write("out/plugin/lp.html", &html).expect("write lp.html");

    println!("wrote out/plugin/queue.html and out/plugin/lp.html");
}
