//! Port of `src/des/main-factory-floor-track3t.ts`.
//!
//! Warehouse/factory-floor sim comparing a conventional floor vs a
//! Track3t-enabled floor; the smart forklift is a QMDP POMDP controller.
//!
//! Delegates to `crate::des::general::factory_floor_track3t`. `process.env.*` →
//! `std::env::var`.
//!
//! PORT NOTE: the HTML animation uses `FrameRecorder` +
//! `animation/scenes/warehouse-track3t-scene`, which is NOT yet ported
//! (`animation::scenes` has no `warehouse_track3t_scene`). The rendering step is
//! stubbed. The full `JSON.stringify(result, null, 2)` artifact write IS ported
//! here via `serde_json::to_string_pretty` over the `WarehouseComparisonResult`
//! tree (which derives `Serialize` with `rename_all = "camelCase"`).

#![allow(dead_code)]

use crate::des::general::factory_floor_track3t::{
    run_warehouse_comparison, summarize_warehouse_comparison, track3t_archive_grounding,
    WarehouseSimulationOptions,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let jobs = env_usize("JOBS", 120);
    let seed = env_usize("SEED", 7) as u32;
    let animate = std::env::var("ANIMATE").as_deref() != Ok("0");
    let result = run_warehouse_comparison(WarehouseSimulationOptions {
        jobs: Some(jobs),
        seed: Some(seed),
        record_trace: Some(true),
        max_steps_per_job: None,
        layout: None,
        destination_plan: None,
    });

    println!("# Factory-floor / warehouse Track3t comparison");
    println!("# jobs={jobs}, seed={seed}");
    println!("# model: source -> movable pallets -> smart-movable forklift -> stationary sinks");
    println!("# controller: POMDP belief updates + QMDP over the underlying MDP");
    println!();
    println!("{}", summarize_warehouse_comparison(&result));
    println!();
    println!("# Improvement deltas");
    println!(
        "mean cycle time reduction = {:.1}%",
        result.deltas.mean_cycle_time_reduction_pct
    );
    println!(
        "throughput lift           = {:.1}%",
        result.deltas.throughput_lift_pct
    );
    println!(
        "search miss reduction     = {:.1}%",
        result.deltas.search_miss_reduction_pct
    );
    println!(
        "shipping error reduction  = {:.1}%",
        result.deltas.error_reduction_pct
    );
    println!(
        "belief entropy reduction  = {:.1}%",
        result.deltas.entropy_reduction_pct
    );
    println!();
    println!("# Archived Track3t grounding");
    for source in track3t_archive_grounding() {
        println!("- {}: {}", source.label, source.url);
    }

    let _ = std::fs::create_dir_all("out");
    let json_path = "out/factory-floor-track3t.json";
    std::fs::write(
        json_path,
        serde_json::to_string_pretty(&result).expect("serialize warehouse comparison"),
    )
    .expect("write factory-floor-track3t.json");
    println!("# wrote {json_path}");

    // PORT NOTE: animation frames/html omitted (warehouse-track3t scene not ported).
    if animate {
        println!("# (animation frames/html not rendered — warehouse-track3t scene not ported)");
    }
}
