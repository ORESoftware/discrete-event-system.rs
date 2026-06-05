//! First-class Track3t warehouse/factory-floor comparison model.
//!
//! This wraps the existing `factory_floor_track3t` simulation in the model
//! citizen contract: JSON spec in, comparison + animation artifact out. The
//! same renderer helper is used by the `main_factory_floor_track3t` catalogue
//! simulation so `/out/factory-floor-track3t.html` and `/models/track3t-warehouse/run`
//! stay aligned.

use std::io;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::warehouse_track3t_scene as scene;
use crate::des::animation::types::{ChartSpec, Frame};
use crate::des::general::factory_floor_track3t as warehouse;
use crate::des::observability::logger::JsonValue;
use crate::des::plugin::UiControl;

use super::artifact::RunArtifact;
use super::registry::{CitizenError, ModelCitizen, ModelDescriptor};

pub const TRACK3T_WAREHOUSE_SCHEMA: &str = "des/track3t-warehouse/v1";
pub const DEFAULT_JOBS: usize = 120;
pub const DEFAULT_SEED: u32 = 7;
pub const DEFAULT_FRAMES_PER_TRACE_STEP: i64 = 6;
pub const DEFAULT_FPS: f64 = 10.0;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Track3tWarehouseSpec {
    #[serde(rename = "$schema")]
    pub schema: Option<String>,
    pub jobs: Option<usize>,
    pub seed: Option<u32>,
    pub max_steps_per_job: Option<usize>,
    pub frames_per_trace_step: Option<i64>,
    pub animation_frames: Option<i64>,
    pub fps: Option<f64>,
}

impl Track3tWarehouseSpec {
    pub fn example_spec() -> Value {
        json!({
            "$schema": TRACK3T_WAREHOUSE_SCHEMA,
            "jobs": 16,
            "seed": DEFAULT_SEED,
            "framesPerTraceStep": 3,
            "animationFrames": 180,
            "fps": DEFAULT_FPS
        })
    }

    pub fn full_default() -> Self {
        Track3tWarehouseSpec {
            schema: Some(TRACK3T_WAREHOUSE_SCHEMA.to_string()),
            jobs: Some(DEFAULT_JOBS),
            seed: Some(DEFAULT_SEED),
            max_steps_per_job: None,
            frames_per_trace_step: Some(DEFAULT_FRAMES_PER_TRACE_STEP),
            animation_frames: None,
            fps: Some(DEFAULT_FPS),
        }
    }

    pub fn job_count(&self) -> usize {
        self.jobs.unwrap_or(DEFAULT_JOBS)
    }

    pub fn seed(&self) -> u32 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }

    pub fn frames_per_trace_step(&self) -> i64 {
        self.frames_per_trace_step
            .unwrap_or(DEFAULT_FRAMES_PER_TRACE_STEP)
            .max(1)
    }

    pub fn fps(&self) -> f64 {
        self.fps.unwrap_or(DEFAULT_FPS)
    }
}

#[derive(Clone, Debug)]
pub struct Track3tAnimation {
    pub frames: Vec<Frame>,
    pub charts: Vec<ChartSpec>,
    pub frames_per_trace_step: i64,
}

#[derive(Clone, Debug)]
pub struct Track3tRenderOptions {
    pub frames_path: String,
    pub html_path: String,
    pub frames_per_trace_step: i64,
    pub animation_frames: Option<i64>,
    pub fps: f64,
}

pub struct Track3tWarehouseCitizen;

impl ModelCitizen for Track3tWarehouseCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "track3t-warehouse".to_string(),
            title: "Track3t Warehouse Floor".to_string(),
            description: "Warehouse/factory-floor comparison: conventional WMS lookup vs a \
                          Track3t-enabled floor, both driven by a POMDP belief update and \
                          QMDP forklift-routing controller."
                .to_string(),
            spec_schema: TRACK3T_WAREHOUSE_SCHEMA.to_string(),
            methods: vec![
                "simulate".to_string(),
                "qmdp-pomdp".to_string(),
                "compare".to_string(),
                "animate".to_string(),
            ],
            example_spec: Track3tWarehouseSpec::example_spec(),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let config = parse_track3t_spec(spec)?;
        let result = run_comparison_from_spec(&config)?;
        let animation = build_track3t_animation(
            &result,
            config.frames_per_trace_step(),
            config.animation_frames,
        );
        let frames = frame_values(&animation);
        let charts = chart_values(&animation);
        let summary_table = warehouse::summarize_warehouse_comparison(&result);
        let summary = summary_sentence(&result);
        let results = json!({
            "summaryText": summary_table,
            "comparison": &result,
            "charts": charts,
            "framesPerTraceStep": animation.frames_per_trace_step,
            "sourceNotes": &result.source_notes,
        });
        Ok(RunArtifact::sim(
            "track3t-warehouse",
            "Track3t Warehouse Floor",
            "2D smart-movable forklift and pallet motion, comparing baseline lookup against \
             Track3t high-frequency location sensing.",
            frames,
            results,
            vec![UiControl::range(
                "speed",
                "Speed (fps)",
                1.0,
                30.0,
                1.0,
                10.0,
            )],
            &summary,
        ))
    }
}

pub fn parse_track3t_spec(spec: &Value) -> Result<Track3tWarehouseSpec, CitizenError> {
    let config: Track3tWarehouseSpec = serde_json::from_value(spec.clone())
        .map_err(|e| CitizenError::InvalidSpec(format!("invalid Track3t warehouse spec: {e}")))?;
    validate_track3t_spec(&config)?;
    Ok(config)
}

pub fn run_comparison_from_spec(
    config: &Track3tWarehouseSpec,
) -> Result<warehouse::WarehouseComparisonResult, CitizenError> {
    validate_track3t_spec(config)?;
    Ok(warehouse::run_warehouse_comparison(
        warehouse::WarehouseSimulationOptions {
            jobs: Some(config.job_count()),
            seed: Some(config.seed()),
            max_steps_per_job: config.max_steps_per_job,
            layout: None,
            record_trace: Some(true),
            destination_plan: None,
        },
    ))
}

pub fn build_track3t_animation(
    result: &warehouse::WarehouseComparisonResult,
    frames_per_trace_step: i64,
    animation_frames: Option<i64>,
) -> Track3tAnimation {
    let frames_per_trace_step = frames_per_trace_step.max(1);
    let scene_result = to_scene_comparison_result(result);
    let total =
        scene::warehouse_comparison_frame_count(&scene_result, frames_per_trace_step).max(0);
    let frame_count = select_frame_count(total, animation_frames);
    let frames = (0..frame_count)
        .map(|i| {
            scene::build_warehouse_comparison_frame(&scene_result, i, frames_per_trace_step)
                .into_frame(
                    scene::warehouse_comparison_frame_time(&scene_result, i, frames_per_trace_step),
                    i as f64,
                )
        })
        .collect();
    let charts = scene::build_warehouse_comparison_charts(&scene_result);
    Track3tAnimation {
        frames,
        charts,
        frames_per_trace_step,
    }
}

pub fn write_track3t_outputs(
    result: &warehouse::WarehouseComparisonResult,
    options: Track3tRenderOptions,
) -> io::Result<usize> {
    let frames_per_trace_step = options.frames_per_trace_step.max(1);
    let scene_result = to_scene_comparison_result(result);
    let total =
        scene::warehouse_comparison_frame_count(&scene_result, frames_per_trace_step).max(0);
    let frame_count = select_frame_count(total, options.animation_frames);
    let mut recorder = FrameRecorder::new(FrameRecorderOpts {
        frames_path: options.frames_path.clone(),
        html_path: Some(options.html_path.clone()),
        width: scene::WAREHOUSE_TRACK3T_STAGE_W,
        height: scene::WAREHOUSE_TRACK3T_STAGE_H,
        fps: Some(options.fps),
        title: Some("Warehouse floor: Track3t comparison".to_string()),
        subtitle: Some(
            "2D smart-movable forklift and pallet motion; default visual dt = 0.1 sec at 1x"
                .to_string(),
        ),
        background: Some("#f8fafc".to_string()),
        live_tick_line: Some(false),
        record_every_ticks: None,
        visual_blocks: None,
    })?;
    for i in 0..frame_count {
        recorder.frame(
            scene::warehouse_comparison_frame_time(&scene_result, i, frames_per_trace_step),
            i as f64,
            || scene::build_warehouse_comparison_frame(&scene_result, i, frames_per_trace_step),
        );
    }
    recorder.set_charts(scene::build_warehouse_comparison_charts(&scene_result));
    let recorded = recorder.get_frame_count();
    let anim = recorder.finish()?;
    Ok(anim.frames.len().max(recorded as usize))
}

pub fn frame_values(animation: &Track3tAnimation) -> Vec<Value> {
    animation
        .frames
        .iter()
        .map(|frame| json_value_to_serde(frame.to_json()))
        .collect()
}

pub fn chart_values(animation: &Track3tAnimation) -> Vec<Value> {
    animation
        .charts
        .iter()
        .map(|chart| json_value_to_serde(chart.to_json()))
        .collect()
}

fn validate_track3t_spec(config: &Track3tWarehouseSpec) -> Result<(), CitizenError> {
    if let Some(schema) = &config.schema {
        if schema != TRACK3T_WAREHOUSE_SCHEMA {
            return Err(CitizenError::InvalidSpec(format!(
                "$schema must be `{TRACK3T_WAREHOUSE_SCHEMA}`, got `{schema}`"
            )));
        }
    }
    let jobs = config.job_count();
    if !(1..=10_000).contains(&jobs) {
        return Err(CitizenError::InvalidSpec(format!(
            "jobs must be between 1 and 10000, got {jobs}"
        )));
    }
    if let Some(max_steps) = config.max_steps_per_job {
        if !(1..=200).contains(&max_steps) {
            return Err(CitizenError::InvalidSpec(format!(
                "maxStepsPerJob must be between 1 and 200, got {max_steps}"
            )));
        }
    }
    let frames_per_trace_step = config.frames_per_trace_step();
    if !(1..=60).contains(&frames_per_trace_step) {
        return Err(CitizenError::InvalidSpec(format!(
            "framesPerTraceStep must be between 1 and 60, got {frames_per_trace_step}"
        )));
    }
    if let Some(animation_frames) = config.animation_frames {
        if !(1..=50_000).contains(&animation_frames) {
            return Err(CitizenError::InvalidSpec(format!(
                "animationFrames must be between 1 and 50000, got {animation_frames}"
            )));
        }
    }
    let fps = config.fps();
    if !fps.is_finite() || !(1.0..=60.0).contains(&fps) {
        return Err(CitizenError::InvalidSpec(format!(
            "fps must be a finite value between 1 and 60, got {fps}"
        )));
    }
    Ok(())
}

fn select_frame_count(total: i64, requested: Option<i64>) -> i64 {
    match requested {
        Some(n) => n.max(1).min(total),
        None => total,
    }
}

fn summary_sentence(result: &warehouse::WarehouseComparisonResult) -> String {
    format!(
        "Track3t warehouse comparison: {:.1}% mean cycle-time reduction, {:.1}% throughput lift, {:.1}% search-miss reduction.",
        result.deltas.mean_cycle_time_reduction_pct,
        result.deltas.throughput_lift_pct,
        result.deltas.search_miss_reduction_pct
    )
}

fn json_value_to_serde(value: JsonValue) -> Value {
    serde_json::from_str(&value.to_string()).unwrap_or(Value::Null)
}

fn to_scene_comparison_result(
    result: &warehouse::WarehouseComparisonResult,
) -> scene::WarehouseComparisonResult {
    scene::WarehouseComparisonResult {
        layout: to_scene_layout(&result.layout),
        baseline: to_scene_scenario_result(&result.baseline),
        track3t: to_scene_scenario_result(&result.track3t),
        deltas: scene::WarehouseDeltas {
            mean_cycle_time_reduction_pct: result.deltas.mean_cycle_time_reduction_pct,
            throughput_lift_pct: result.deltas.throughput_lift_pct,
            search_miss_reduction_pct: result.deltas.search_miss_reduction_pct,
            error_reduction_pct: result.deltas.error_reduction_pct,
        },
    }
}

fn to_scene_scenario_result(
    result: &warehouse::WarehouseScenarioResult,
) -> scene::WarehouseScenarioResult {
    scene::WarehouseScenarioResult {
        scenario: scene::WarehouseScenario {
            label: result.scenario.label.clone(),
        },
        layout: to_scene_layout(&result.layout),
        trace: result.trace.iter().map(to_scene_trace_row).collect(),
        metrics: scene::WarehouseMetrics {
            completed_jobs: result.metrics.completed_jobs as f64,
            jobs_created: result.metrics.jobs_created as f64,
            mean_cycle_time: result.metrics.mean_cycle_time,
            throughput_per_hour: result.metrics.throughput_per_hour,
            shipping_error_rate: result.metrics.shipping_error_rate,
        },
    }
}

fn to_scene_layout(layout: &warehouse::WarehouseLayout) -> scene::WarehouseLayout {
    scene::WarehouseLayout {
        stations: layout
            .stations
            .iter()
            .map(|station| scene::StationDefinition {
                id: station.id.clone(),
                label: station.label.clone(),
                kind: station.kind.as_str().to_string(),
                x: station.x,
                y: station.y,
            })
            .collect(),
        route_edges: layout.route_edges.clone(),
    }
}

fn to_scene_trace_row(row: &warehouse::WarehouseStepTrace) -> scene::WarehouseStepTrace {
    scene::WarehouseStepTrace {
        belief_by_station: row.belief_by_station.clone(),
        destination: Some(row.destination.clone()),
        forklift_before: row.forklift_before.clone(),
        forklift_after: row.forklift_after.clone(),
        carrying_before: row.carrying_before,
        carrying_after: row.carrying_after,
        event: row.event.as_str().to_string(),
        pallet_before: row.pallet_before.clone(),
        pallet_after: row.pallet_after.clone(),
        job_id: row.job_id.clone(),
        action_target: row.action_target.clone(),
        observation: row.observation.clone(),
        cycle_time_so_far: row.cycle_time_so_far,
        belief_entropy: row.belief_entropy,
        cumulative_errors: row.cumulative_errors as f64,
        cumulative_delivered: row.cumulative_delivered as f64,
        cumulative_search_misses: row.cumulative_search_misses as f64,
        time_start: row.time_start,
        time_end: row.time_end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_spec_runs_to_nonempty_animation() {
        let spec = parse_track3t_spec(&Track3tWarehouseSpec::example_spec()).unwrap();
        let result = run_comparison_from_spec(&spec).unwrap();
        let animation =
            build_track3t_animation(&result, spec.frames_per_trace_step(), spec.animation_frames);
        assert!(!animation.frames.is_empty());
        assert!(!animation.frames[0].shapes.is_empty());
        assert_eq!(animation.charts.len(), 2);
    }
}
