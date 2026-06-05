//! Port of `src/des/main-factory-floor-track3t.ts`.
//!
//! Warehouse/factory-floor sim comparing a conventional floor vs a
//! Track3t-enabled floor; the smart forklift is a QMDP POMDP controller.
//!
//! Delegates to `crate::des::general::factory_floor_track3t`. `process.env.*` →
//! `std::env::var`.
//!
//! The full `JSON.stringify(result, null, 2)` artifact write is ported here via
//! `serde_json::to_string_pretty`, and the warehouse Track3t scene is rendered
//! through the shared first-class model helper so the catalogue simulation writes
//! `out/factory-floor-track3t.html` as well as the JSON result.

#![allow(dead_code)]

use std::path::Path;

use crate::des::general::factory_floor_track3t::{
    run_warehouse_comparison, summarize_warehouse_comparison, track3t_archive_grounding,
    WarehouseSimulationOptions,
};
use crate::des::model::track3t_warehouse::{
    write_track3t_outputs, Track3tRenderOptions, DEFAULT_FPS, DEFAULT_FRAMES_PER_TRACE_STEP,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i64_opt(key: &str) -> Option<i64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_f64(key: &str, default: f64) -> f64 {
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

    if animate {
        let frames_path = Path::new("out")
            .join("factory-floor-track3t.frames.jsonl")
            .to_string_lossy()
            .into_owned();
        let html_path = Path::new("out")
            .join("factory-floor-track3t.html")
            .to_string_lossy()
            .into_owned();
        let frames_per_trace_step =
            env_i64("MOTION_FRAMES_PER_STEP", DEFAULT_FRAMES_PER_TRACE_STEP);
        let animation_frames = env_i64_opt("ANIM_FRAMES");
        let fps = env_f64("FPS", DEFAULT_FPS);
        let recorded = write_track3t_outputs(
            &result,
            Track3tRenderOptions {
                frames_path: frames_path.clone(),
                html_path: html_path.clone(),
                frames_per_trace_step,
                animation_frames,
                fps,
            },
        )
        .expect("write factory-floor-track3t animation");
        println!("# wrote {frames_path} ({recorded} frames)");
        println!("# wrote {html_path}");
    }
}
