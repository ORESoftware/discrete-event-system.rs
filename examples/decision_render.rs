//! Render the first-class decision-process demos (and a hybrid peer) through the
//! shared model contract + plugin player.
//!
//! This exercises the whole platform seam end-to-end: a canonical serde spec is
//! serialized to JSON, run through the `CitizenRegistry` purely from that JSON
//! (the same path an English-prompt → spec pipeline would take), and the uniform
//! `RunArtifact` is rendered to a self-contained HTML player. MDP, POMDP and the
//! hybrid block diagram all flow through one registry as peers.
//!
//! Run with: `cargo run --example decision_render`
//! Outputs:  out/decision/{mdp,pomdp,hybrid}.html (+ .jsonl frame streams)

use serde_json::{json, Value};

use des_engine::des::decision::{machine_maintenance_mdp, tiger_pomdp};
use des_engine::des::model::with_builtins;

/// Merge extra run options (start/steps/method/…) into a spec value.
fn with_opts(mut spec: Value, opts: Value) -> Value {
    if let (Value::Object(map), Value::Object(extra)) = (&mut spec, opts) {
        for (k, v) in extra {
            map.insert(k, v);
        }
    }
    spec
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("out/decision")?;
    let registry = with_builtins();

    // 1) Control MDP — machine maintenance. Animated state graph + value/return.
    let mdp_spec = with_opts(
        serde_json::to_value(machine_maintenance_mdp())?,
        json!({ "start": 0, "steps": 24, "seed": 7 }),
    );
    let mdp = registry.run("mdp", &mdp_spec)?;
    std::fs::write("out/decision/mdp.html", mdp.to_player_html())?;
    std::fs::write("out/decision/mdp.frames.jsonl", mdp.to_jsonl())?;
    println!("mdp:    {}", mdp.summary);

    // 2) Belief POMDP — the tiger. Animated belief bars + entropy/return.
    let pomdp_spec = with_opts(
        serde_json::to_value(tiger_pomdp())?,
        json!({ "method": "lookahead", "horizon": 3, "steps": 18, "seed": 5 }),
    );
    let pomdp = registry.run("pomdp", &pomdp_spec)?;
    std::fs::write("out/decision/pomdp.html", pomdp.to_player_html())?;
    std::fs::write("out/decision/pomdp.frames.jsonl", pomdp.to_jsonl())?;
    println!("pomdp:  {}", pomdp.summary);

    // 3) Hybrid block diagram — the same contract, a different paradigm.
    let hybrid = registry.run("hybrid", &json!({ "demo": "bouncing-ball" }))?;
    std::fs::write("out/decision/hybrid.html", hybrid.to_player_html())?;
    println!("hybrid: {}", hybrid.summary);

    // 4) Visual-block studio — the two-layer core (flat VisualBlocks over runtime
    //    cells of one-or-more Layer-2 elements), rendered as a live wiring diagram.
    for demo in ["signal-chain", "mixer"] {
        let studio = registry.run("studio", &json!({ "demo": demo }))?;
        std::fs::write(format!("out/decision/studio-{demo}.html"), studio.to_player_html())?;
        println!("studio: {} — {}", demo, studio.summary);
    }

    // Discovery: the registry advertises every first-class kind as JSON.
    let descriptors: Vec<Value> = registry
        .descriptors()
        .iter()
        .map(|d| serde_json::to_value(d).unwrap())
        .collect();
    std::fs::write(
        "out/decision/citizens.json",
        serde_json::to_string_pretty(&json!({ "citizens": descriptors }))?,
    )?;
    println!("\nregistered first-class kinds: {:?}", registry.kinds());

    Ok(())
}
