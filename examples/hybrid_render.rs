//! Render hybrid block-diagram runs through the existing plugin sim-player.
//!
//! This shows the spine end-to-end: build a `des::hybrid` diagram, simulate it,
//! and feed the resulting `Trace` (as JSONL frames) into the plugin player —
//! the same player external plugins use. No process is spawned; we construct a
//! `PluginRun` from in-memory frames directly.
//!
//! Run with: `cargo run --example hybrid_render`
//! Outputs:  out/hybrid/closed-loop.html, out/hybrid/bouncing-ball.html

use serde_json::{json, Value};

use des_engine::des::hybrid::{demos, executive::simulate, Trace};
use des_engine::des::plugin::{
    render_player_html, OutputKind, PlayerKind, PluginManifest, PluginOutput, PluginRun, RunSpec,
    UiControl,
};

fn manifest(id: &str, name: &str, desc: &str, controls: Vec<UiControl>) -> PluginManifest {
    PluginManifest {
        id: id.to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        description: desc.to_string(),
        // Placeholder: we render from in-memory frames, so this is never spawned.
        run: RunSpec::new("hybrid-internal", &[]),
        output: OutputKind::Jsonl,
        player: PlayerKind::Sim,
        controls,
        title: Some(name.to_string()),
    }
}

fn render(manifest: &PluginManifest, frames: Vec<Value>) -> String {
    let run = PluginRun {
        plugin_id: manifest.id.clone(),
        output: PluginOutput::Jsonl(frames),
        exit_code: Some(0),
        stderr: String::new(),
    };
    render_player_html(manifest, &run)
}

/// Bouncing-ball frames: chart height/velocity AND draw an animated ball so the
/// zero-crossing events are visible in the stage.
fn ball_frames(trace: &Trace) -> Vec<Value> {
    let (ts, hs) = trace.series("ball.p0[0]").expect("height channel");
    let (_, vs) = trace.series("ball.p0[1]").expect("velocity channel");
    ts.iter()
        .enumerate()
        .map(|(k, &t)| {
            let h = hs[k];
            let cy = 222.0 - h.max(0.0) * 180.0; // floor at y=222, h=1 near the top
            json!({
                "t": t,
                "height": h,
                "velocity": vs[k],
                "shapes": [
                    { "kind": "line", "x1": 20.0, "y1": 224.0, "x2": 180.0, "y2": 224.0,
                      "stroke": "#475569", "strokeWidth": 2.0 },
                    { "kind": "circle", "x": 100.0, "y": cy, "r": 12.0, "fill": "#2563eb" },
                    { "kind": "text", "x": 100.0, "y": 18.0, "text": format!("h = {h:.3}"),
                      "anchor": "middle", "fontSize": 12.0, "fill": "#0f172a" }
                ]
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("out/hybrid")?;

    // 1) Multirate closed loop -> scope view (continuous plant + discrete PI).
    let (compiled, opts) = demos::closed_loop()?;
    let trace = simulate(&compiled, &opts);
    let m = manifest(
        "hybrid-closed-loop",
        "Hybrid: multirate closed loop",
        "Continuous first-order plant regulated to a setpoint by a discrete-time PI controller (10 Hz). The plant integrates continuously; the command (pi.p0) is a zero-order-hold staircase.",
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 20.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &["all", "plant.p0", "pi.p0", "error.p0", "reference.p0"],
                "all",
                Some("metric"),
            ),
        ],
    );
    let html = render(&m, trace.to_jsonl_frames());
    std::fs::write("out/hybrid/closed-loop.html", &html)?;
    println!(
        "closed-loop: {} frames, {} events -> out/hybrid/closed-loop.html",
        trace.times.len(),
        trace.events
    );

    // 2) Bouncing ball -> animated stage + height/velocity timeline.
    let (compiled, opts) = demos::bouncing_ball()?;
    let trace = simulate(&compiled, &opts);
    let m = manifest(
        "hybrid-bouncing-ball",
        "Hybrid: bouncing ball",
        "A purely continuous plant with a zero-crossing at the floor and an energy-losing reflection event (restitution 0.8). Each bounce is located by bisection.",
        vec![UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 30.0)],
    );
    let html = render(&m, ball_frames(&trace));
    std::fs::write("out/hybrid/bouncing-ball.html", &html)?;
    println!(
        "bouncing-ball: {} frames, {} events -> out/hybrid/bouncing-ball.html",
        trace.times.len(),
        trace.events
    );

    Ok(())
}
