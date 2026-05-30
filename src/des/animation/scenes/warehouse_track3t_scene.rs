//! Port of `src/des/animation/scenes/warehouse-track3t-scene.ts`.
//!
//! Builds frames + charts for the warehouse Track-3T forklift-routing comparison
//! animation (two side-by-side panels: baseline vs. track3t). Forklifts and
//! pallets move in 2-D space and POMDP belief rings show pallet-location belief.
//!
//! ## Conversion notes
//!
//! * `stationById` → `Option<&StationDefinition>`; `reserveRowKey` →
//!   `Option<String>`.
//! * `Map`/`Set` become `std::collections::HashMap` / `Vec` membership.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::factory_floor_track3t::*` the scene reads is mirrored
//!   locally below.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::animation::types::{
    to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts, LineShape,
    PathShape, RectShape, Shape, TextShape,
};

pub const WAREHOUSE_TRACK3T_STAGE_W: f64 = 1200.0;
pub const WAREHOUSE_TRACK3T_STAGE_H: f64 = 720.0;

// PORT NOTE: local mirror of the factory-floor track3t result types (subset).
#[derive(Clone, Debug, Default)]
pub struct StationDefinition {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseLayout {
    pub stations: Vec<StationDefinition>,
    pub route_edges: Option<Vec<(String, String)>>,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseStepTrace {
    pub belief_by_station: Vec<f64>,
    pub destination: Option<String>,
    pub forklift_before: String,
    pub forklift_after: String,
    pub carrying_before: bool,
    pub carrying_after: bool,
    pub event: String,
    pub pallet_before: String,
    pub pallet_after: String,
    pub job_id: String,
    pub action_target: String,
    pub observation: String,
    pub cycle_time_so_far: f64,
    pub belief_entropy: f64,
    pub cumulative_errors: f64,
    pub cumulative_delivered: f64,
    pub cumulative_search_misses: f64,
    pub time_start: f64,
    pub time_end: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseMetrics {
    pub completed_jobs: f64,
    pub jobs_created: f64,
    pub mean_cycle_time: f64,
    pub throughput_per_hour: f64,
    pub shipping_error_rate: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseScenario {
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseScenarioResult {
    pub scenario: WarehouseScenario,
    pub layout: WarehouseLayout,
    pub trace: Vec<WarehouseStepTrace>,
    pub metrics: WarehouseMetrics,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseDeltas {
    pub mean_cycle_time_reduction_pct: f64,
    pub throughput_lift_pct: f64,
    pub search_miss_reduction_pct: f64,
    pub error_reduction_pct: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseComparisonResult {
    pub layout: WarehouseLayout,
    pub baseline: WarehouseScenarioResult,
    pub track3t: WarehouseScenarioResult,
    pub deltas: WarehouseDeltas,
}

#[derive(Clone, Copy, Debug)]
struct PanelGeom {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    plot_x: f64,
    plot_y: f64,
    plot_w: f64,
    plot_h: f64,
    max_x: f64,
    max_y: f64,
}

struct MotionFrame<'a> {
    row: Option<&'a WarehouseStepTrace>,
    row_index: i64,
    phase: f64,
}

pub fn build_warehouse_comparison_frame(
    result: &WarehouseComparisonResult,
    frame_index: i64,
    frames_per_trace_step: i64,
) -> FrameParts {
    let mut shapes: Vec<Shape> = vec![
        Shape::Rect(RectShape { x: 0.0, y: 0.0, w: WAREHOUSE_TRACK3T_STAGE_W, h: WAREHOUSE_TRACK3T_STAGE_H, fill: "#f8fafc".to_string(), ..Default::default() }),
        Shape::Text(TextShape { x: 34.0, y: 34.0, text: "Warehouse floor: smart-movable forklifts and pallet flow".to_string(), font_size: Some(21.0), fill: Some("#111827".to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }),
        Shape::Text(TextShape { x: 34.0, y: 58.0, text: "Forklifts and pallets move in 2D space; rings show the POMDP belief over pallet location.".to_string(), font_size: Some(13.0), fill: Some("#4b5563".to_string()), ..Default::default() }),
    ];

    let base_panel = make_panel(34.0, 86.0, &result.layout.stations);
    let track_panel = make_panel(626.0, 86.0, &result.layout.stations);
    let base_motion = motion_at(&result.baseline.trace, frame_index, frames_per_trace_step);
    let track_motion = motion_at(&result.track3t.trace, frame_index, frames_per_trace_step);
    draw_panel(
        &mut shapes,
        &result.baseline,
        &base_motion,
        &base_panel,
        "#b91c1c",
    );
    draw_panel(
        &mut shapes,
        &result.track3t,
        &track_motion,
        &track_panel,
        "#047857",
    );
    draw_delta_summary(&mut shapes, result);

    let caption = [
        format!("frame {}", frame_index + 1),
        format!(
            "baseline t={} min",
            to_fixed(interpolated_time(&base_motion), 1)
        ),
        format!(
            "track3t t={} min",
            to_fixed(interpolated_time(&track_motion), 1)
        ),
    ]
    .join(" | ");
    FrameParts::with_caption(shapes, caption)
}

pub fn warehouse_comparison_frame_count(
    result: &WarehouseComparisonResult,
    frames_per_trace_step: i64,
) -> i64 {
    (result.baseline.trace.len().max(result.track3t.trace.len())) as i64
        * 1.max(frames_per_trace_step)
}

pub fn warehouse_comparison_frame_time(
    result: &WarehouseComparisonResult,
    frame_index: i64,
    frames_per_trace_step: i64,
) -> f64 {
    let base = interpolated_time(&motion_at(
        &result.baseline.trace,
        frame_index,
        frames_per_trace_step,
    ));
    let track = interpolated_time(&motion_at(
        &result.track3t.trace,
        frame_index,
        frames_per_trace_step,
    ));
    base.max(track)
}

pub fn build_warehouse_comparison_charts(result: &WarehouseComparisonResult) -> Vec<ChartSpec> {
    let misses_max = {
        let mut m = 1.0_f64;
        for r in result
            .baseline
            .trace
            .iter()
            .chain(result.track3t.trace.iter())
        {
            m = m.max(r.cumulative_search_misses + r.cumulative_errors);
        }
        m
    };
    vec![
        ChartSpec {
            x: 34.0,
            y: 552.0,
            w: 544.0,
            h: 130.0,
            title: Some("Cumulative delivered jobs".to_string()),
            y_label: Some("jobs".to_string()),
            y_min: Some(0.0),
            y_max: Some(
                result
                    .baseline
                    .metrics
                    .completed_jobs
                    .max(result.track3t.metrics.completed_jobs),
            ),
            series: vec![
                cumulative_series(&result.baseline, "baseline", "#b91c1c", |r| {
                    r.cumulative_delivered
                }),
                cumulative_series(&result.track3t, "track3t", "#047857", |r| {
                    r.cumulative_delivered
                }),
            ],
            ..Default::default()
        },
        ChartSpec {
            x: 626.0,
            y: 552.0,
            w: 544.0,
            h: 130.0,
            title: Some("Cumulative search misses and delivery errors".to_string()),
            y_label: Some("count".to_string()),
            y_min: Some(0.0),
            y_max: Some(misses_max),
            series: vec![
                cumulative_series(&result.baseline, "baseline", "#b91c1c", |r| {
                    r.cumulative_search_misses + r.cumulative_errors
                }),
                cumulative_series(&result.track3t, "track3t", "#047857", |r| {
                    r.cumulative_search_misses + r.cumulative_errors
                }),
            ],
            ..Default::default()
        },
    ]
}

fn draw_panel(
    shapes: &mut Vec<Shape>,
    scenario_result: &WarehouseScenarioResult,
    motion: &MotionFrame,
    g: &PanelGeom,
    accent: &str,
) {
    let row = motion.row;
    shapes.push(Shape::Rect(RectShape {
        x: g.x,
        y: g.y,
        w: g.w,
        h: g.h,
        fill: "#ffffff".to_string(),
        stroke: Some("#d1d5db".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: g.x + 18.0,
        y: g.y + 28.0,
        text: scenario_result.scenario.label.clone(),
        font_size: Some(17.0),
        fill: Some("#111827".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: g.x + 18.0,
        y: g.y + 50.0,
        text: metric_line(scenario_result),
        font_size: Some(12.0),
        fill: Some("#4b5563".to_string()),
        ..Default::default()
    }));

    draw_routes(shapes, &scenario_result.layout, g);
    let compact = scenario_result.layout.stations.len() > 12;
    for (i, station) in scenario_result.layout.stations.iter().enumerate() {
        let p = station_point(station, g);
        let belief = row.map_or(0.0, |r| r.belief_by_station.get(i).copied().unwrap_or(0.0));
        if belief > 0.01 {
            shapes.push(Shape::Circle(CircleShape {
                x: p.0,
                y: p.1,
                r: (if compact { 9.0 } else { 14.0 })
                    + (if compact { 22.0 } else { 30.0 }) * belief,
                fill: accent.to_string(),
                opacity: Some(0.13 + 0.28 * belief.min(1.0)),
                title: Some(format!("{}: belief {}", station.label, to_fixed(belief, 2))),
                ..Default::default()
            }));
        }
    }

    let destination = row.and_then(|r| r.destination.clone());
    for station in &scenario_result.layout.stations {
        let p = station_point(station, g);
        let is_destination = destination.as_deref() == Some(station.id.as_str());
        let box_w = if compact {
            if station.kind == "storage" {
                34.0
            } else {
                52.0
            }
        } else {
            68.0
        };
        let box_h = if compact {
            if station.kind == "storage" {
                24.0
            } else {
                30.0
            }
        } else {
            34.0
        };
        shapes.push(Shape::Rect(RectShape {
            x: p.0 - box_w / 2.0,
            y: p.1 - box_h / 2.0,
            w: box_w,
            h: box_h,
            rx: Some(5.0),
            fill: station_fill(station),
            stroke: Some(if is_destination {
                accent.to_string()
            } else {
                "#374151".to_string()
            }),
            stroke_width: Some(if is_destination { 3.0 } else { 1.0 }),
            title: Some(format!("{} ({})", station.label, station.kind)),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: p.0,
            y: p.1 + 4.0,
            text: station.label.clone(),
            font_size: Some(if compact { 9.0 } else { 10.0 }),
            fill: Some("#111827".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    if let Some(row) = row {
        let route = route_path_points(
            &scenario_result.layout,
            &row.forklift_before,
            &row.forklift_after,
            g,
        );
        draw_motion_path(shapes, &route, accent);
        let forklift = point_on_polyline(&route, motion.phase);
        let pallet = pallet_point(&scenario_result.layout, row, motion.phase, forklift, g);
        draw_pallet(
            shapes,
            pallet.0,
            pallet.1,
            row.carrying_before
                || row.carrying_after
                || row.event == "pickup"
                || row.event == "delivered",
        );
        draw_forklift(
            shapes,
            forklift.0,
            forklift.1,
            accent,
            row.carrying_before || row.carrying_after,
            direction_angle(&route, motion.phase),
        );
        shapes.push(Shape::Text(TextShape {
            x: g.x + 18.0,
            y: g.y + g.h - 58.0,
            text: format!("{}: {} via {}", row.job_id, row.event, row.action_target),
            font_size: Some(12.0),
            fill: Some("#111827".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: g.x + 18.0,
            y: g.y + g.h - 38.0,
            text: format!("obs: {}", row.observation),
            font_size: Some(11.0),
            fill: Some("#4b5563".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: g.x + 18.0,
            y: g.y + g.h - 18.0,
            text: format!(
                "cycle {} min | entropy {} | errors {}",
                to_fixed(row.cycle_time_so_far, 1),
                to_fixed(row.belief_entropy, 2),
                crate::des::animation::types::js_num(row.cumulative_errors)
            ),
            font_size: Some(11.0),
            fill: Some("#4b5563".to_string()),
            ..Default::default()
        }));
    }
}

fn draw_motion_path(shapes: &mut Vec<Shape>, points: &[(f64, f64)], accent: &str) {
    if points.len() < 2 {
        return;
    }
    let d = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            format!(
                "{} {} {}",
                if i == 0 { "M" } else { "L" },
                to_fixed(p.0, 1),
                to_fixed(p.1, 1)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    shapes.push(Shape::Path(PathShape {
        d,
        stroke: Some(accent.to_string()),
        stroke_width: Some(2.0),
        fill: Some("none".to_string()),
        opacity: Some(0.3),
        ..Default::default()
    }));
}

fn draw_forklift(shapes: &mut Vec<Shape>, x: f64, y: f64, accent: &str, loaded: bool, angle: f64) {
    let body_w = 28.0;
    let body_h = 16.0;
    shapes.push(Shape::Rect(RectShape {
        x: x - body_w / 2.0,
        y: y - body_h / 2.0 - 20.0,
        w: body_w,
        h: body_h,
        rx: Some(4.0),
        fill: accent.to_string(),
        stroke: Some("#111827".to_string()),
        stroke_width: Some(1.3),
        title: Some("smart-movable forklift".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: x - 4.0,
        y: y - body_h / 2.0 - 29.0,
        w: 13.0,
        h: 10.0,
        rx: Some(3.0),
        fill: "#e0f2fe".to_string(),
        stroke: Some("#0f172a".to_string()),
        stroke_width: Some(1.0),
        ..Default::default()
    }));
    shapes.push(Shape::Circle(CircleShape {
        x: x - 9.0,
        y: y - 10.0,
        r: 3.0,
        fill: "#111827".to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Circle(CircleShape {
        x: x + 9.0,
        y: y - 10.0,
        r: 3.0,
        fill: "#111827".to_string(),
        ..Default::default()
    }));
    let fork_start_x = x + angle.cos() * 12.0;
    let fork_start_y = y - 20.0 + angle.sin() * 12.0;
    shapes.push(Shape::Line(LineShape {
        x1: fork_start_x,
        y1: fork_start_y,
        x2: fork_start_x + angle.cos() * 18.0,
        y2: fork_start_y + angle.sin() * 18.0,
        stroke: "#111827".to_string(),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    shapes.push(Shape::Line(LineShape {
        x1: fork_start_x,
        y1: fork_start_y + 5.0,
        x2: fork_start_x + angle.cos() * 18.0,
        y2: fork_start_y + 5.0 + angle.sin() * 18.0,
        stroke: "#111827".to_string(),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x,
        y: y - 17.0,
        text: "F".to_string(),
        font_size: Some(10.0),
        fill: Some("#ffffff".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    if loaded {
        shapes.push(Shape::Text(TextShape {
            x: x + 22.0,
            y: y - 24.0,
            text: "loaded".to_string(),
            font_size: Some(10.0),
            fill: Some(accent.to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
}

fn draw_pallet(shapes: &mut Vec<Shape>, x: f64, y: f64, active: bool) {
    shapes.push(Shape::Rect(RectShape {
        x: x - 10.0,
        y: y - 7.0,
        w: 20.0,
        h: 14.0,
        rx: Some(2.0),
        fill: if active {
            "#f59e0b".to_string()
        } else {
            "#fbbf24".to_string()
        },
        stroke: Some("#92400e".to_string()),
        stroke_width: Some(1.0),
        label: Some("P".to_string()),
        title: Some("movable pallet".to_string()),
        ..Default::default()
    }));
}

fn pallet_point(
    layout: &WarehouseLayout,
    row: &WarehouseStepTrace,
    phase: f64,
    forklift: (f64, f64),
    g: &PanelGeom,
) -> (f64, f64) {
    let before = station_by_id(layout, &row.pallet_before);
    let after = station_by_id(layout, &row.pallet_after);
    let before_p = before.map_or(forklift, |s| station_point(s, g));
    let after_p = after.map_or(forklift, |s| station_point(s, g));
    if row.carrying_before && row.carrying_after {
        return (forklift.0 + 18.0, forklift.1 - 22.0);
    }
    if !row.carrying_before && row.carrying_after {
        return if phase < 0.78 {
            (before_p.0, before_p.1 + 24.0)
        } else {
            (forklift.0 + 18.0, forklift.1 - 22.0)
        };
    }
    if row.carrying_before && !row.carrying_after {
        return if phase < 0.86 {
            (forklift.0 + 18.0, forklift.1 - 22.0)
        } else {
            (after_p.0, after_p.1 + 24.0)
        };
    }
    (
        before_p.0 + (after_p.0 - before_p.0) * phase,
        before_p.1 + 24.0 + (after_p.1 - before_p.1) * phase,
    )
}

fn draw_routes(shapes: &mut Vec<Shape>, layout: &WarehouseLayout, g: &PanelGeom) {
    let by_id: HashMap<&str, &StationDefinition> =
        layout.stations.iter().map(|s| (s.id.as_str(), s)).collect();
    let empty = Vec::new();
    let route_pairs = layout.route_edges.as_ref().unwrap_or(&empty);
    for (a_id, b_id) in route_pairs {
        let (a, b) = match (by_id.get(a_id.as_str()), by_id.get(b_id.as_str())) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue,
        };
        let pa = station_point(a, g);
        let pb = station_point(b, g);
        shapes.push(Shape::Line(LineShape {
            x1: pa.0,
            y1: pa.1,
            x2: pb.0,
            y2: pb.1,
            stroke: "#cbd5e1".to_string(),
            stroke_width: Some(4.0),
            opacity: Some(0.8),
            ..Default::default()
        }));
    }
}

fn draw_delta_summary(shapes: &mut Vec<Shape>, result: &WarehouseComparisonResult) {
    let x = 34.0;
    let y = 504.0;
    shapes.push(Shape::Rect(RectShape {
        x,
        y,
        w: 1136.0,
        h: 32.0,
        fill: "#111827".to_string(),
        rx: Some(5.0),
        ..Default::default()
    }));
    let d = &result.deltas;
    shapes.push(Shape::Text(TextShape {
        x: x + 16.0,
        y: y + 21.0,
        text: format!(
            "Track3t lift: cycle {}% faster, throughput {}% higher, search misses {}% lower, errors {}% lower",
            to_fixed(d.mean_cycle_time_reduction_pct, 1),
            to_fixed(d.throughput_lift_pct, 1),
            to_fixed(d.search_miss_reduction_pct, 1),
            to_fixed(d.error_reduction_pct, 1)
        ),
        font_size: Some(13.0),
        fill: Some("#f9fafb".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
}

fn make_panel(x: f64, y: f64, stations: &[StationDefinition]) -> PanelGeom {
    let max_x = stations
        .iter()
        .map(|s| s.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = stations
        .iter()
        .map(|s| s.y)
        .fold(f64::NEG_INFINITY, f64::max);
    PanelGeom {
        x,
        y,
        w: 540.0,
        h: 398.0,
        plot_x: x + 58.0,
        plot_y: y + 82.0,
        plot_w: 420.0,
        plot_h: 230.0,
        max_x,
        max_y,
    }
}

fn station_point(station: &StationDefinition, g: &PanelGeom) -> (f64, f64) {
    let x = g.plot_x + station.x / 1.0_f64.max(g.max_x) * g.plot_w;
    let y = g.plot_y + station.y / 1.0_f64.max(g.max_y) * g.plot_h;
    (x, y)
}

fn station_fill(station: &StationDefinition) -> String {
    match station.kind.as_str() {
        "source" => "#dbeafe".to_string(),
        "storage" => "#fef3c7".to_string(),
        "sink" => "#dcfce7".to_string(),
        _ => "#e5e7eb".to_string(),
    }
}

fn motion_at(
    trace: &[WarehouseStepTrace],
    frame_index: i64,
    frames_per_trace_step: i64,
) -> MotionFrame<'_> {
    if trace.is_empty() {
        return MotionFrame {
            row: None,
            row_index: -1,
            phase: 1.0,
        };
    }
    let subframes = 1.max(frames_per_trace_step);
    let row_index = (trace.len() as i64 - 1).min(frame_index / subframes);
    let sub_index = (subframes - 1).min(frame_index % subframes);
    let phase = if subframes <= 1 {
        1.0
    } else {
        sub_index as f64 / (subframes as f64 - 1.0)
    };
    MotionFrame {
        row: Some(&trace[row_index as usize]),
        row_index,
        phase,
    }
}

fn interpolated_time(motion: &MotionFrame) -> f64 {
    match motion.row {
        None => 0.0,
        Some(row) => row.time_start + (row.time_end - row.time_start) * motion.phase,
    }
}

fn station_by_id<'a>(layout: &'a WarehouseLayout, id: &str) -> Option<&'a StationDefinition> {
    layout.stations.iter().find(|s| s.id == id)
}

fn route_path_points(
    layout: &WarehouseLayout,
    from_id: &str,
    to_id: &str,
    g: &PanelGeom,
) -> Vec<(f64, f64)> {
    let from = match station_by_id(layout, from_id) {
        Some(s) => s,
        None => return vec![],
    };
    let to = match station_by_id(layout, to_id) {
        Some(s) => s,
        None => return vec![],
    };
    if from_id == to_id {
        let p = station_point(from, g);
        return vec![p, p];
    }
    if let Some(corridor) = row_corridor_path(layout, from, to, g) {
        return corridor;
    }
    let by_id: HashMap<&str, &StationDefinition> =
        layout.stations.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    let add = |adj: &mut HashMap<String, Vec<String>>, a: &str, b: &str| {
        adj.entry(a.to_string()).or_default().push(b.to_string());
    };
    if let Some(edges) = &layout.route_edges {
        for (a, b) in edges {
            add(&mut adjacency, a, b);
            add(&mut adjacency, b, a);
        }
    }
    let mut prev: HashMap<String, Option<String>> = HashMap::new();
    let mut q: Vec<String> = vec![from_id.to_string()];
    prev.insert(from_id.to_string(), None);
    let mut i = 0;
    while i < q.len() {
        let cur = q[i].clone();
        if cur == to_id {
            break;
        }
        if let Some(neighbors) = adjacency.get(&cur).cloned() {
            for next in neighbors {
                if prev.contains_key(&next) {
                    continue;
                }
                prev.insert(next.clone(), Some(cur.clone()));
                q.push(next);
            }
        }
        i += 1;
    }
    if !prev.contains_key(to_id) {
        return vec![station_point(from, g), station_point(to, g)];
    }
    let mut ids: Vec<String> = Vec::new();
    let mut cur: Option<String> = Some(to_id.to_string());
    while let Some(c) = cur {
        ids.push(c.clone());
        cur = prev.get(&c).cloned().flatten();
    }
    ids.reverse();
    ids.iter()
        .filter_map(|id| by_id.get(id.as_str()).map(|s| station_point(s, g)))
        .collect()
}

#[derive(Clone, Debug)]
struct ReserveRow {
    key: String,
    y: f64,
    stations: Vec<StationDefinition>,
}

fn push_opt(points: &mut Vec<StationDefinition>, s: Option<&StationDefinition>) {
    if let Some(s) = s {
        if points.last().is_none_or(|last| last.id != s.id) {
            points.push(s.clone());
        }
    }
}

fn push_point(points: &mut Vec<StationDefinition>, station: &StationDefinition) {
    if points.last().is_none_or(|last| last.id != station.id) {
        points.push(station.clone());
    }
}

fn row_corridor_path(
    layout: &WarehouseLayout,
    from: &StationDefinition,
    to: &StationDefinition,
    g: &PanelGeom,
) -> Option<Vec<(f64, f64)>> {
    let rows = reserve_rows(layout);
    let main = station_by_id(layout, "aisle-main");
    let staging = station_by_id(layout, "staging");
    let receiving = station_by_id(layout, "receiving");
    let (main, staging) = match (main, staging) {
        (Some(m), Some(s)) if !rows.is_empty() => (m, s),
        _ => return None,
    };

    let from_reserve = reserve_row_key(&from.id);
    let to_reserve = reserve_row_key(&to.id);
    let uses_right_side =
        from.kind == "sink" || to.kind == "sink" || from.id == main.id || to.id == main.id;
    let uses_reserve = from_reserve.is_some() || to_reserve.is_some();
    if !uses_right_side && !uses_reserve {
        return None;
    }

    let row = choose_corridor_row(&rows, from, to);
    let mut points: Vec<StationDefinition> = Vec::new();

    if from.kind == "sink" {
        push_opt(&mut points, Some(from));
        push_opt(&mut points, Some(main));
        if to.kind == "sink" {
            push_opt(&mut points, Some(to));
        } else if to.id == main.id {
            push_opt(&mut points, Some(to));
        } else {
            append_row_from_main(&mut points, &row.stations, to);
            if to.id == staging.id {
                push_opt(&mut points, Some(staging));
            } else if receiving.is_some_and(|r| to.id == r.id) {
                push_opt(&mut points, Some(staging));
                push_opt(&mut points, receiving);
            } else if to_reserve.is_none()
                && to.id != staging.id
                && receiving.is_none_or(|r| to.id != r.id)
            {
                push_opt(&mut points, Some(to));
            }
        }
        return Some(points.iter().map(|s| station_point(s, g)).collect());
    }

    push_opt(&mut points, Some(from));
    if receiving.is_some_and(|r| from.id == r.id) {
        push_opt(&mut points, Some(staging));
    }
    if to.kind == "sink" || to.id == main.id {
        append_row_to_main(&mut points, &row.stations, from);
        push_opt(&mut points, Some(main));
        if to.kind == "sink" {
            push_opt(&mut points, Some(to));
        }
        return Some(points.iter().map(|s| station_point(s, g)).collect());
    }

    if from.id == main.id {
        append_row_from_main(&mut points, &row.stations, to);
        if receiving.is_some_and(|r| to.id == r.id) {
            push_opt(&mut points, Some(staging));
            push_opt(&mut points, receiving);
        } else if to.id == staging.id {
            push_opt(&mut points, Some(staging));
        }
        return Some(points.iter().map(|s| station_point(s, g)).collect());
    }

    if to_reserve.is_some() {
        let same_row = from_reserve == to_reserve;
        if same_row {
            push_opt(&mut points, Some(to));
        } else {
            append_row_to_main(&mut points, &row.stations, from);
            push_opt(&mut points, Some(main));
            let to_row = rows
                .iter()
                .find(|r| Some(&r.key) == to_reserve.as_ref())
                .unwrap_or(&row);
            append_row_from_main(&mut points, &to_row.stations, to);
        }
        return Some(points.iter().map(|s| station_point(s, g)).collect());
    }

    None
}

fn reserve_rows(layout: &WarehouseLayout) -> Vec<ReserveRow> {
    let mut keys: Vec<String> = Vec::new();
    let mut buckets: HashMap<String, Vec<StationDefinition>> = HashMap::new();
    for station in &layout.stations {
        let key = match reserve_row_key(&station.id) {
            Some(k) => k,
            None => continue,
        };
        if !buckets.contains_key(&key) {
            keys.push(key.clone());
        }
        buckets.entry(key).or_default().push(station.clone());
    }
    let mut rows: Vec<ReserveRow> = keys
        .into_iter()
        .map(|key| {
            let mut stations = buckets.remove(&key).unwrap_or_default();
            stations.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
            let y = stations.iter().map(|s| s.y).sum::<f64>() / stations.len() as f64;
            ReserveRow { key, stations, y }
        })
        .collect();
    rows.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

fn reserve_row_key(id: &str) -> Option<String> {
    let rest = id.strip_prefix("reserve-")?;
    let mut chars = rest.chars();
    let c = chars.next()?;
    if !c.is_ascii_lowercase() {
        return None;
    }
    let digits: String = chars.collect();
    if digits.is_empty() || !digits.chars().all(|d| d.is_ascii_digit()) {
        return None;
    }
    Some(c.to_string())
}

fn choose_corridor_row(
    rows: &[ReserveRow],
    from: &StationDefinition,
    to: &StationDefinition,
) -> ReserveRow {
    let from_key = reserve_row_key(&from.id);
    let to_key = reserve_row_key(&to.id);
    let exact = if let Some(fk) = &from_key {
        rows.iter().find(|r| &r.key == fk)
    } else if let Some(tk) = &to_key {
        rows.iter().find(|r| &r.key == tk)
    } else {
        None
    };
    if let Some(exact) = exact {
        return exact.clone();
    }
    let y = if from.kind == "sink" {
        from.y
    } else if to.kind == "sink" {
        to.y
    } else {
        (from.y + to.y) / 2.0
    };
    let mut best = &rows[0];
    for row in rows {
        if (row.y - y).abs() < (best.y - y).abs() {
            best = row;
        }
    }
    best.clone()
}

fn append_row_to_main(
    points: &mut Vec<StationDefinition>,
    row: &[StationDefinition],
    from: &StationDefinition,
) {
    let from_idx = row.iter().position(|s| s.id == from.id);
    let start = from_idx.map_or(0, |i| i + 1);
    for s in row.iter().skip(start) {
        push_point(points, s);
    }
}

fn append_row_from_main(
    points: &mut Vec<StationDefinition>,
    row: &[StationDefinition],
    to: &StationDefinition,
) {
    let to_idx = row.iter().position(|s| s.id == to.id);
    let end = to_idx.unwrap_or(0);
    let mut i = row.len() as i64 - 1;
    while i >= end as i64 {
        push_point(points, &row[i as usize]);
        i -= 1;
    }
}

fn point_on_polyline(points: &[(f64, f64)], phase: f64) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }
    if points.len() == 1 {
        return points[0];
    }
    let mut lengths: Vec<f64> = Vec::new();
    let mut total = 0.0;
    for i in 1..points.len() {
        let len = (points[i].0 - points[i - 1].0).hypot(points[i].1 - points[i - 1].1);
        lengths.push(len);
        total += len;
    }
    if total <= 0.0 {
        return points[points.len() - 1];
    }
    let mut target = phase.clamp(0.0, 1.0) * total;
    for i in 1..points.len() {
        let len = lengths[i - 1];
        if target <= len || i == points.len() - 1 {
            let local = if len <= 0.0 { 1.0 } else { target / len };
            return (
                points[i - 1].0 + (points[i].0 - points[i - 1].0) * local,
                points[i - 1].1 + (points[i].1 - points[i - 1].1) * local,
            );
        }
        target -= len;
    }
    points[points.len() - 1]
}

fn direction_angle(points: &[(f64, f64)], phase: f64) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let p = point_on_polyline(points, (phase - 0.03).max(0.0));
    let q = point_on_polyline(points, (phase + 0.03).min(1.0));
    (q.1 - p.1).atan2(q.0 - p.0)
}

fn metric_line(r: &WarehouseScenarioResult) -> String {
    let m = &r.metrics;
    [
        format!(
            "{}/{} done",
            crate::des::animation::types::js_num(m.completed_jobs),
            crate::des::animation::types::js_num(m.jobs_created)
        ),
        format!("{} min/job", to_fixed(m.mean_cycle_time, 1)),
        format!("{} jobs/hr", to_fixed(m.throughput_per_hour, 1)),
        format!("{}% err", to_fixed(m.shipping_error_rate * 100.0, 1)),
    ]
    .join(" | ")
}

fn cumulative_series(
    result: &WarehouseScenarioResult,
    label: &str,
    color: &str,
    value: impl Fn(&WarehouseStepTrace) -> f64,
) -> ChartSeries {
    let mut t: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for row in &result.trace {
        t.push(row.time_end);
        y.push(value(row));
    }
    ChartSeries {
        label: label.to_string(),
        color: color.to_string(),
        t,
        y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_row_key_matches_pattern() {
        assert_eq!(reserve_row_key("reserve-a12"), Some("a".to_string()));
        assert_eq!(reserve_row_key("reserve-12"), None);
        assert_eq!(reserve_row_key("aisle-main"), None);
        assert_eq!(reserve_row_key("reserve-ab1"), None);
    }

    #[test]
    fn point_on_polyline_midpoint() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0)];
        let p = point_on_polyline(&pts, 0.5);
        assert!((p.0 - 5.0).abs() < 1e-9);
        assert!((p.1 - 0.0).abs() < 1e-9);
    }
}
