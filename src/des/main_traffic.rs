//! Port of `src/des/main-traffic.ts`.
//!
//! Thin runner: small traffic-flow simulation over DES cell stations (cars
//! carry position / velocity / acceleration / jerk). The runner also writes
//! self-contained HTML players for the canonical traffic demos.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`]; the fixed seed is passed through to the
//!     seeded sim inside `general::network_flow`.
//!   - delegates to `general::network_flow`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{json, Value};

use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, run_traffic_flow, TrafficCarSnapshot, TrafficLane,
    TrafficNetwork, TrafficNodeKind, TrafficParams, TrafficResult, TrafficTraceRow,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficCarSnapshot, SmartTrafficParams, SmartTrafficResult,
    SmartTrafficTraceRow,
};
use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

const MAX_TRAFFIC_FRAMES: usize = 220;
const MAX_DRAWN_CARS: usize = 80;

/// `fmt(x, digits=2)` — finite numbers to `digits` decimals, else `"n/a"`.
fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{:.*}", digits, x)
    } else {
        "n/a".to_string()
    }
}

fn mean_abs_jerk(result: &TrafficResult) -> f64 {
    let jerks: Vec<f64> = result
        .trace
        .iter()
        .flat_map(|row| row.cars.iter().map(|car| car.jerk_mps3.abs()))
        .collect();
    if jerks.is_empty() {
        0.0
    } else {
        jerks.iter().sum::<f64>() / jerks.len() as f64
    }
}

fn min_leader_gap(result: &TrafficResult) -> f64 {
    let gaps: Vec<f64> = result
        .trace
        .iter()
        .flat_map(|row| row.cars.iter().filter_map(|car| car.leader_gap_m))
        .collect();
    if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

fn traffic_demo_params() -> TrafficParams {
    TrafficParams {
        builtin: Some("five-intersection".to_string()),
        network: Some(build_five_intersection_traffic_network()),
        duration_sec: 180.0,
        dt_sec: 0.25,
        seed: 19.0,
        max_cars: 250,
        car_length_m: None,
        car_width_m: None,
        lane_width_m: None,
        min_gap_m: None,
        max_accel_mps2: None,
        max_decel_mps2: None,
        max_jerk_mps3: Some(2.5),
        reaction_time_sec: Some(1.0),
        time_headway_sec: Some(1.2),
        grid_cell_size_m: Some(0.3048),
        grid_look_ahead_m: None,
        spawn_rate_multiplier: Some(1.0),
        scheduled_trips: None,
    }
}

fn smart_traffic_demo_params() -> SmartTrafficParams {
    SmartTrafficParams {
        base: TrafficParams {
            builtin: Some("five-intersection".to_string()),
            network: None,
            duration_sec: 180.0,
            dt_sec: 0.2,
            seed: 19.0,
            max_cars: 250,
            car_length_m: None,
            car_width_m: None,
            lane_width_m: None,
            min_gap_m: None,
            max_accel_mps2: None,
            max_decel_mps2: None,
            max_jerk_mps3: Some(4.0),
            reaction_time_sec: Some(0.8),
            time_headway_sec: Some(1.1),
            grid_cell_size_m: Some(0.3048),
            grid_look_ahead_m: None,
            spawn_rate_multiplier: Some(3.0),
            scheduled_trips: None,
        },
        smart_car_pool_size: Some(400),
        actor_shuffle_seed: Some(2026.0),
        accident_risk_scale: Some(16.0),
        accident_probability: None,
        accident_accel_boost_mps2: Some(12.0),
        accident_fault_duration_sec: Some(1.0),
        distance_preference_spread: Some(0.54),
        start_preference_spread: Some(0.65),
        accident_flash_seconds: Some(2.5),
    }
}

fn sample_indices(len: usize, max_frames: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    if len <= max_frames {
        return (0..len).collect();
    }
    let step = len.div_ceil(max_frames).max(1);
    let mut indices = (0..len).step_by(step).collect::<Vec<_>>();
    let last = len - 1;
    if indices.last().copied() != Some(last) {
        indices.push(last);
    }
    indices
}

fn node_positions(network: &TrafficNetwork) -> HashMap<String, (f64, f64)> {
    network
        .nodes
        .iter()
        .map(|n| (n.id.clone(), (70.0 + n.x * 120.0, 80.0 + n.y * 118.0)))
        .collect()
}

fn green_lanes(
    network: &TrafficNetwork,
    signal_phases: &HashMap<String, String>,
) -> HashSet<String> {
    let mut lanes = HashSet::new();
    if let Some(signals) = &network.signals {
        for signal in signals {
            let Some(phase_name) = signal_phases.get(&signal.node_id) else {
                continue;
            };
            if let Some(phase) = signal.phases.iter().find(|p| &p.name == phase_name) {
                for lane in &phase.green_lanes {
                    lanes.insert(lane.clone());
                }
            }
        }
    }
    lanes
}

fn lane_point(
    network: &TrafficNetwork,
    positions: &HashMap<String, (f64, f64)>,
    lane_id: &str,
    position_m: f64,
) -> Option<(f64, f64)> {
    let lane = network.lanes.iter().find(|l| l.id == lane_id)?;
    let (x1, y1) = *positions.get(&lane.from)?;
    let (x2, y2) = *positions.get(&lane.to)?;
    let u = if lane.length_m <= 0.0 {
        0.0
    } else {
        (position_m / lane.length_m).clamp(0.0, 1.0)
    };
    Some((x1 + (x2 - x1) * u, y1 + (y2 - y1) * u))
}

fn push_left_text(
    shapes: &mut Vec<Value>,
    x: f64,
    y: f64,
    text: impl Into<String>,
    font_size: f64,
    fill: &str,
    font_weight: Option<&str>,
) {
    let mut shape = json!({
        "kind": "text",
        "x": x,
        "y": y,
        "text": text.into(),
        "fontSize": font_size,
        "fill": fill,
    });
    if let Value::Object(map) = &mut shape {
        if let Some(font_weight) = font_weight {
            map.insert("fontWeight".to_string(), json!(font_weight));
        }
    }
    shapes.push(shape);
}

fn push_network(
    shapes: &mut Vec<Value>,
    network: &TrafficNetwork,
    signal_phases: &HashMap<String, String>,
) {
    let positions = node_positions(network);
    let green = green_lanes(network, signal_phases);
    for lane in &network.lanes {
        push_lane(shapes, lane, &positions, green.contains(&lane.id));
    }
    for node in &network.nodes {
        let Some((x, y)) = positions.get(&node.id).copied() else {
            continue;
        };
        let (fill, stroke) = match node.kind {
            TrafficNodeKind::Source => ("#dcfce7", "#16a34a"),
            TrafficNodeKind::Intersection => ("#fef3c7", "#d97706"),
            TrafficNodeKind::Sink => ("#fee2e2", "#dc2626"),
        };
        shapes.push(json!({
            "kind": "circle", "x": x, "y": y, "r": 19.0,
            "fill": fill, "stroke": stroke, "strokeWidth": 2.0
        }));
        shapes.push(json!({
            "kind": "text", "x": x, "y": y + 4.0, "text": node.id,
            "anchor": "middle", "fontSize": 11.0, "fill": "#0f172a", "fontWeight": "bold"
        }));
        if let Some(phase) = signal_phases.get(&node.id) {
            shapes.push(json!({
                "kind": "text", "x": x, "y": y + 34.0, "text": phase,
                "anchor": "middle", "fontSize": 9.0, "fill": "#166534"
            }));
        }
    }
}

fn push_lane(
    shapes: &mut Vec<Value>,
    lane: &TrafficLane,
    positions: &HashMap<String, (f64, f64)>,
    is_green: bool,
) {
    let (Some((x1, y1)), Some((x2, y2))) = (positions.get(&lane.from), positions.get(&lane.to))
    else {
        return;
    };
    shapes.push(json!({
        "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "stroke": if is_green { "#16a34a" } else { "#94a3b8" },
        "strokeWidth": if is_green { 5.0 } else { 2.0 },
        "opacity": if is_green { 0.9 } else { 0.65 }
    }));
    let mx = (x1 + x2) / 2.0;
    let my = (y1 + y2) / 2.0;
    shapes.push(json!({
        "kind": "text", "x": mx, "y": my - 8.0, "text": lane.id,
        "anchor": "middle", "fontSize": 8.0,
        "fill": if is_green { "#166534" } else { "#64748b" }
    }));
}

fn push_panel(shapes: &mut Vec<Value>, title: &str, rows: &[String], accent: &str) {
    let x = 820.0;
    let y = 42.0;
    shapes.push(json!({
        "kind": "rect", "x": x, "y": y, "w": 300.0, "h": 298.0,
        "rx": 8.0, "fill": "#f8fafc", "stroke": "#cbd5e1", "strokeWidth": 1.0
    }));
    push_left_text(
        shapes,
        x + 16.0,
        y + 30.0,
        title,
        15.0,
        "#0f172a",
        Some("bold"),
    );
    shapes.push(json!({
        "kind": "rect", "x": x + 16.0, "y": y + 43.0, "w": 92.0, "h": 4.0,
        "rx": 2.0, "fill": accent
    }));
    for (i, row) in rows.iter().enumerate() {
        push_left_text(
            shapes,
            x + 16.0,
            y + 76.0 + i as f64 * 23.0,
            row,
            11.0,
            "#334155",
            None,
        );
    }
}

fn push_basic_cars(shapes: &mut Vec<Value>, network: &TrafficNetwork, cars: &[TrafficCarSnapshot]) {
    let positions = node_positions(network);
    for car in cars.iter().take(MAX_DRAWN_CARS) {
        let Some((x, y)) = lane_point(network, &positions, &car.lane_id, car.position_m) else {
            continue;
        };
        let hot = car.leader_gap_m.is_some_and(|g| g < 3.0);
        shapes.push(json!({
            "kind": "circle", "x": x, "y": y, "r": if hot { 5.5 } else { 4.6 },
            "fill": if hot { "#f97316" } else { "#2563eb" },
            "stroke": "#ffffff", "strokeWidth": 1.2,
            "opacity": 0.94
        }));
    }
}

fn push_smart_cars(
    shapes: &mut Vec<Value>,
    network: &TrafficNetwork,
    cars: &[SmartTrafficCarSnapshot],
) {
    let positions = node_positions(network);
    for car in cars.iter().take(MAX_DRAWN_CARS) {
        let Some((x, y)) = lane_point(network, &positions, &car.lane_id, car.position_m) else {
            continue;
        };
        let faulted = car.fault_mode.is_some() || car.accident_count > 0;
        shapes.push(json!({
            "kind": "circle", "x": x, "y": y, "r": if faulted { 6.2 } else { 4.7 },
            "fill": if faulted { "#dc2626" } else { "#7c3aed" },
            "stroke": if faulted { "#fecaca" } else { "#ffffff" },
            "strokeWidth": if faulted { 2.0 } else { 1.1 },
            "opacity": 0.95
        }));
    }
}

fn traffic_frame(result: &TrafficResult, row: &TrafficTraceRow) -> Value {
    let mut shapes = Vec::new();
    push_network(&mut shapes, &result.network, &row.signal_phases);
    push_basic_cars(&mut shapes, &result.network, &row.cars);
    push_panel(
        &mut shapes,
        "Traffic flow",
        &[
            format!("t = {:.1}s  tick = {}", row.time_sec, row.tick),
            format!(
                "active = {} / max {}",
                row.active_cars, result.max_active_cars
            ),
            format!("entered = {}  exited = {}", row.entered, row.exited),
            format!(
                "queue = {}  active cells = {}",
                row.queue_length, row.active_grid_cells
            ),
            format!("mean speed = {:.2} m/s", row.mean_speed_mps),
            format!("mean travel = {:.1}s", row.mean_travel_time_sec),
            format!(
                "drawn cars = {} / {}",
                row.cars.len().min(MAX_DRAWN_CARS),
                row.cars.len()
            ),
            "green links are open signal phases".to_string(),
        ],
        "#2563eb",
    );
    json!({
        "t": row.time_sec,
        "activeCars": row.active_cars as f64,
        "entered": row.entered as f64,
        "exited": row.exited as f64,
        "queueLength": row.queue_length as f64,
        "meanSpeedMps": row.mean_speed_mps,
        "meanTravelSec": row.mean_travel_time_sec,
        "activeGridCells": row.active_grid_cells as f64,
        "shapes": shapes,
        "caption": format!("traffic-flow five-intersection · t={:.1}s · active={} · exited={}", row.time_sec, row.active_cars, row.exited),
    })
}

fn smart_traffic_frame(result: &SmartTrafficResult, row: &SmartTrafficTraceRow) -> Value {
    let mut shapes = Vec::new();
    push_network(&mut shapes, &result.network, &row.signal_phases);
    push_smart_cars(&mut shapes, &result.network, &row.cars);
    for accident in &row.accidents {
        let positions = node_positions(&result.network);
        if let Some((x, y)) = lane_point(
            &result.network,
            &positions,
            &accident.lane_id,
            accident.position_m,
        ) {
            shapes.push(json!({
                "kind": "circle", "x": x, "y": y, "r": 12.0,
                "fill": "#fee2e2", "stroke": "#dc2626", "strokeWidth": 3.0,
                "opacity": 0.9
            }));
            shapes.push(json!({
                "kind": "text", "x": x, "y": y + 4.0, "text": "!",
                "anchor": "middle", "fontSize": 14.0, "fill": "#991b1b",
                "fontWeight": "bold"
            }));
        }
    }
    push_panel(
        &mut shapes,
        "Smart traffic",
        &[
            format!("t = {:.1}s  tick = {}", row.time_sec, row.tick),
            format!(
                "active = {} / max {}",
                row.active_cars, result.max_active_cars
            ),
            format!("smart runs = {}", row.smart_movable_runs),
            format!("scheduled smart cars = {}", row.scheduled_smart_cars),
            format!("entered = {}  exited = {}", row.entered, row.exited),
            format!(
                "crashed = {}  total accidents = {}",
                row.crashed,
                result.accidents.len()
            ),
            format!(
                "queue = {}  active cells = {}",
                row.queue_length, row.active_grid_cells
            ),
            format!("mean speed = {:.2} m/s", row.mean_speed_mps),
            format!(
                "drawn cars = {} / {}",
                row.cars.len().min(MAX_DRAWN_CARS),
                row.cars.len()
            ),
        ],
        "#7c3aed",
    );
    json!({
        "t": row.time_sec,
        "activeCars": row.active_cars as f64,
        "entered": row.entered as f64,
        "exited": row.exited as f64,
        "crashed": row.crashed as f64,
        "accidentsThisTick": row.accidents.len() as f64,
        "smartMovableRuns": row.smart_movable_runs as f64,
        "queueLength": row.queue_length as f64,
        "meanSpeedMps": row.mean_speed_mps,
        "meanTravelSec": row.mean_travel_time_sec,
        "activeGridCells": row.active_grid_cells as f64,
        "shapes": shapes,
        "caption": format!("smart-traffic-flow · t={:.1}s · active={} · exited={} · crashed={}", row.time_sec, row.active_cars, row.exited, row.crashed),
    })
}

fn metric_controls(metrics: &[&str]) -> Vec<UiControl> {
    let mut opts = vec!["all"];
    opts.extend_from_slice(metrics);
    vec![
        UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
        UiControl::select("metric", "Feature signal", &opts, "all", Some("metric")),
    ]
}

fn traffic_artifact(result: &TrafficResult) -> RunArtifact {
    let frames = sample_indices(result.trace.len(), MAX_TRAFFIC_FRAMES)
        .into_iter()
        .map(|i| traffic_frame(result, &result.trace[i]))
        .collect::<Vec<_>>();
    let summary = format!(
        "Five-intersection traffic-flow: entered {}, exited {}, mean travel {:.1}s, max active {}.",
        result.entered, result.exited, result.mean_travel_time_sec, result.max_active_cars
    );
    RunArtifact::sim(
        "traffic-flow",
        "Traffic Flow — Five Intersection",
        "Continuous-position traffic flow with signal phases, lane occupancy, and moving cars.",
        frames,
        json!({
            "kind": "traffic-flow",
            "entered": result.entered,
            "exited": result.exited,
            "dropped": result.dropped,
            "meanTravelSec": result.mean_travel_time_sec,
            "meanSpeedMps": result.mean_speed_mps,
            "maxActiveCars": result.max_active_cars,
            "network": {
                "nodes": result.network.nodes.len(),
                "lanes": result.network.lanes.len(),
                "sources": result.network.sources.len(),
                "sinks": result.network.sinks.len(),
            },
            "cellStats": {
                "cellSizeM": result.cell_stats.cell_size_m,
                "activeCells": result.cell_stats.active_cells,
                "createdCellStations": result.cell_stats.created_cell_stations,
                "maxCellOccupancy": result.cell_stats.max_cell_occupancy,
            },
        }),
        metric_controls(&[
            "activeCars",
            "entered",
            "exited",
            "queueLength",
            "meanSpeedMps",
            "meanTravelSec",
            "activeGridCells",
        ]),
        &summary,
    )
}

fn smart_traffic_artifact(result: &SmartTrafficResult) -> RunArtifact {
    let frames = sample_indices(result.trace.len(), MAX_TRAFFIC_FRAMES)
        .into_iter()
        .map(|i| smart_traffic_frame(result, &result.trace[i]))
        .collect::<Vec<_>>();
    let summary = format!(
        "Smart traffic-flow: entered {}, exited {}, crashed {}, accidents {}, smart runs {}.",
        result.entered,
        result.exited,
        result.crashed,
        result.accidents.len(),
        result.execution.total_smart_movable_runs
    );
    RunArtifact::sim(
        "smart-traffic-flow",
        "Smart Traffic Flow — Five Intersection",
        "Smart movable cars on the five-intersection network with shuffled actor updates and accident events.",
        frames,
        json!({
            "kind": "smart-traffic-flow",
            "entered": result.entered,
            "exited": result.exited,
            "crashed": result.crashed,
            "dropped": result.dropped,
            "accidents": result.accidents.len(),
            "meanTravelSec": result.mean_travel_time_sec,
            "meanSpeedMps": result.mean_speed_mps,
            "maxActiveCars": result.max_active_cars,
            "execution": {
                "participants": result.execution.participant_count,
                "smartMovables": result.execution.smart_movable_count,
                "totalSmartMovableRuns": result.execution.total_smart_movable_runs,
                "maxSmartMovableRunsPerTick": result.execution.max_smart_movable_runs_per_tick,
                "shuffledByRunner": result.execution.shuffled_by_runner,
            },
            "cellStats": {
                "cellSizeM": result.cell_stats.cell_size_m,
                "activeCells": result.cell_stats.active_cells,
                "createdCellStations": result.cell_stats.created_cell_stations,
                "accidentCellStations": result.cell_stats.accident_cell_stations,
                "accidentCellHits": result.cell_stats.accident_cell_hits,
                "maxCellOccupancy": result.cell_stats.max_cell_occupancy,
            },
        }),
        metric_controls(&[
            "activeCars",
            "entered",
            "exited",
            "crashed",
            "accidentsThisTick",
            "smartMovableRuns",
            "queueLength",
            "meanSpeedMps",
            "meanTravelSec",
            "activeGridCells",
        ]),
        &summary,
    )
}

pub fn write_traffic_html_pages() -> std::io::Result<(String, String)> {
    let out_dir = Path::new("out");
    std::fs::create_dir_all(out_dir)?;

    let traffic = run_traffic_flow(traffic_demo_params(), None);
    let traffic_artifact = traffic_artifact(&traffic);
    let traffic_html = out_dir.join("traffic-flow-five-intersection.html");
    let traffic_frames = out_dir.join("traffic-flow-five-intersection.frames.jsonl");
    std::fs::write(&traffic_html, traffic_artifact.to_player_html())?;
    std::fs::write(&traffic_frames, traffic_artifact.to_jsonl())?;

    let smart = run_smart_traffic_flow(smart_traffic_demo_params(), None);
    let smart_artifact = smart_traffic_artifact(&smart);
    let smart_html = out_dir.join("smart-traffic-flow.html");
    let smart_frames = out_dir.join("smart-traffic-flow.frames.jsonl");
    std::fs::write(&smart_html, smart_artifact.to_player_html())?;
    std::fs::write(&smart_frames, smart_artifact.to_jsonl())?;

    Ok((
        traffic_html.display().to_string(),
        smart_html.display().to_string(),
    ))
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let result = run_traffic_flow(traffic_demo_params(), None);
    let mean_abs_jerk_mps3 = mean_abs_jerk(&result);
    let min_leader_gap_m = min_leader_gap(&result);

    println!("# Traffic-flow DES");
    println!("# TrafficCellStation grid + moving car snapshots; kinematics stepped at dt");
    println!(
        "# nodes={}, lanes={}, sources={}, cells={}",
        result.network.nodes.len(),
        result.network.lanes.len(),
        result.network.sources.len(),
        result.cell_stats.created_cell_stations
    );
    println!(
        "# dt={}s, cell={}m, configured cap={}, max active={}",
        result.params.dt_sec,
        fmt(result.cell_stats.cell_size_m, 4),
        result.params.max_cars,
        result.max_active_cars
    );
    println!();

    println!("## Demand and throughput");
    println!("  entered cars:         {}", result.entered);
    println!("  exited cars:          {}", result.exited);
    println!("  active at stop:       {}", result.final_cars.len());
    println!("  dropped attempts:     {}", result.dropped);
    println!();

    println!("## Kinematics");
    println!(
        "  mean travel:       {} sec",
        fmt(result.mean_travel_time_sec, 1)
    );
    println!("  mean speed:        {} m/s", fmt(result.mean_speed_mps, 2));
    println!("  mean |jerk|:       {} m/s^3", fmt(mean_abs_jerk_mps3, 2));
    println!("  min leader gap:    {} m", fmt(min_leader_gap_m, 3));
    println!(
        "  max cell occup.:   {}",
        result.cell_stats.max_cell_occupancy
    );
    println!();

    println!("## Final sample");
    for car in result.final_cars.iter().take(10) {
        let cells: Vec<String> = car.grid_cell_ids.iter().take(3).cloned().collect();
        println!(
            "  car={:>3} lane={:<6} x={}m v={}m/s a={}m/s^2 cells={}",
            car.id,
            car.lane_id,
            fmt(car.position_m, 2),
            fmt(car.speed_mps, 2),
            fmt(car.acceleration_mps2, 2),
            cells.join("|")
        );
    }

    match write_traffic_html_pages() {
        Ok((traffic_html, smart_html)) => {
            println!();
            println!("## HTML outputs");
            println!("  traffic: {traffic_html}");
            println!("  smart:   {smart_html}");
        }
        Err(e) => {
            eprintln!("traffic HTML generation failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traffic_artifact_contains_frames_and_network_shapes() {
        let mut params = traffic_demo_params();
        params.duration_sec = 6.0;
        params.dt_sec = 1.0;
        params.max_cars = 30;
        let result = run_traffic_flow(params, None);
        let artifact = traffic_artifact(&result);
        assert_eq!(artifact.kind, "traffic-flow");
        assert!(!artifact.frames.is_empty());
        assert!(artifact.to_player_html().contains("Traffic Flow"));
        assert!(artifact
            .frames
            .iter()
            .any(|f| f["shapes"].as_array().is_some_and(|s| !s.is_empty())));
    }

    #[test]
    fn smart_traffic_artifact_contains_smart_metrics() {
        let mut params = smart_traffic_demo_params();
        params.base.duration_sec = 4.0;
        params.base.dt_sec = 0.5;
        params.base.max_cars = 30;
        params.smart_car_pool_size = Some(40);
        params.accident_risk_scale = Some(0.0);
        let result = run_smart_traffic_flow(params, None);
        let artifact = smart_traffic_artifact(&result);
        assert_eq!(artifact.kind, "smart-traffic-flow");
        assert!(!artifact.frames.is_empty());
        assert!(artifact.to_player_html().contains("Smart Traffic Flow"));
        assert!(
            artifact.results["execution"]["smartMovables"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
    }
}
