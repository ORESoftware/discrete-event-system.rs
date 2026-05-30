//! Port of `src/des/animation/scenes/shortest-path-scene.ts`.
//!
//! Builds frames + charts for the shortest-path-DES relaxation animation: graph
//! nodes coloured by their current distance estimate, directed edges (thicker /
//! brighter where a relaxation wave fired this tick), and a sidebar of metrics.
//!
//! ## Conversion notes
//!
//! * The `project` closure and the `Infinity` / `isFinite` distance handling
//!   become a Rust closure plus [`f64::is_finite`] / `f64::INFINITY`.
//! * `distanceColor` builds an `rgb(..)` string via `format!`.
//! * PORT NOTE: the canonical `crate::des::general::shortest_path_des::{Graph,
//!   …}` types are not relied on here; the minimal [`Graph`]/[`Edge`] mirror
//!   below captures exactly the fields the scene reads. Replace with the
//!   canonical types once their field layout is confirmed.

#![allow(dead_code)]

use std::collections::HashSet;

use crate::des::animation::types::{
    to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts, LineShape,
    RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1000.0;
pub const STAGE_H: f64 = 640.0;
const VIEW_X: f64 = 60.0;
const VIEW_Y: f64 = 40.0;
const VIEW_W: f64 = 600.0;
const VIEW_H: f64 = 560.0;
const META_X: f64 = 700.0;
const META_Y: f64 = 40.0;
const META_W: f64 = 260.0;
const META_H: f64 = 560.0;

// PORT NOTE: local mirror of the shortest-path graph (subset used by the scene).
#[derive(Clone, Debug)]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub num_nodes: usize,
    pub edges: Vec<Vec<Edge>>,
    pub coordinates: Option<Vec<[f64; 2]>>,
    pub node_names: Option<Vec<String>>,
}

impl Graph {
    fn coord(&self, i: usize) -> [f64; 2] {
        // `coords = graph.coordinates ?? []; coords[i] ?? [50, 50]`.
        self.coordinates
            .as_ref()
            .and_then(|c| c.get(i))
            .copied()
            .unwrap_or([50.0, 50.0])
    }

    fn node_name(&self, v: usize) -> String {
        // `graph.nodeNames?.[v] ?? v`.
        self.node_names
            .as_ref()
            .and_then(|n| n.get(v))
            .cloned()
            .unwrap_or_else(|| v.to_string())
    }
}

/// One relaxation event observed during a tick.
#[derive(Clone, Debug)]
pub struct WaveEvent {
    pub from: usize,
    pub to: usize,
    pub new_distance: f64,
    pub improved: bool,
}

/// `'bellman-ford-des' | 'dijkstra-des'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpAlgorithm {
    BellmanFordDes,
    DijkstraDes,
}

impl SpAlgorithm {
    pub fn label(self) -> &'static str {
        match self {
            SpAlgorithm::BellmanFordDes => "bellman-ford-des",
            SpAlgorithm::DijkstraDes => "dijkstra-des",
        }
    }
}

fn distance_color(d: f64, max_finite: f64) -> String {
    if !d.is_finite() {
        return "#475569".to_string();
    }
    let t = if max_finite > 0.0 { d / max_finite } else { 0.0 };
    // Cool to warm gradient (low distance = bright yellow, high = blue).
    let r = (250.0 - 230.0 * t).round() as i64;
    let g = (220.0 - 80.0 * t).round() as i64;
    let b = (60.0 + 180.0 * t).round() as i64;
    format!("rgb({r}, {g}, {b})")
}

fn meta_line(shapes: &mut Vec<Shape>, y0: &mut f64, text: String, color: &str) {
    shapes.push(Shape::Text(TextShape {
        x: META_X + 20.0,
        y: *y0,
        text,
        font_size: Some(12.0),
        fill: Some(color.to_string()),
        ..Default::default()
    }));
    *y0 += 22.0;
}

#[allow(clippy::too_many_arguments)]
pub fn build_shortest_path_frame(
    _t: f64,
    _tick: f64,
    graph: &Graph,
    distance_now: &[f64],
    wave_events: &[WaveEvent],
    source: usize,
    iteration: f64,
    algorithm: SpAlgorithm,
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();

    // Figure out the projection.
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in 0..graph.num_nodes {
        let c = graph.coord(i);
        if c[0] < x_min {
            x_min = c[0];
        }
        if c[0] > x_max {
            x_max = c[0];
        }
        if c[1] < y_min {
            y_min = c[1];
        }
        if c[1] > y_max {
            y_max = c[1];
        }
    }
    let pad = 30.0;
    let sx = (VIEW_W - 2.0 * pad) / (x_max - x_min).max(1e-9);
    let sy = (VIEW_H - 2.0 * pad) / (y_max - y_min).max(1e-9);
    let project = |i: usize| -> (f64, f64) {
        let c = graph.coord(i);
        (VIEW_X + pad + (c[0] - x_min) * sx, VIEW_Y + pad + (c[1] - y_min) * sy)
    };

    // Find max finite distance for the colour scale.
    let mut max_finite = 0.0_f64;
    for &d in distance_now {
        if d.is_finite() && d > max_finite {
            max_finite = d;
        }
    }

    // Frame.
    shapes.push(Shape::Rect(RectShape {
        x: VIEW_X,
        y: VIEW_Y,
        w: VIEW_W,
        h: VIEW_H,
        fill: "#0b1220".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));

    // Edges: thin grey by default, thick yellow if a wave fired this tick.
    let fired_set: HashSet<String> =
        wave_events.iter().map(|e| format!("{}->{}", e.from, e.to)).collect();
    let improved_set: HashSet<String> = wave_events
        .iter()
        .filter(|e| e.improved)
        .map(|e| format!("{}->{}", e.from, e.to))
        .collect();
    for u in 0..graph.num_nodes {
        for e in &graph.edges[u] {
            let (x1, y1) = project(u);
            let (x2, y2) = project(e.to);
            let key = format!("{}->{}", u, e.to);
            let is_improved = improved_set.contains(&key);
            let is_fired = fired_set.contains(&key);
            shapes.push(Shape::Line(LineShape {
                x1,
                y1,
                x2,
                y2,
                stroke: if is_improved {
                    "#fde68a".to_string()
                } else if is_fired {
                    "#facc15".to_string()
                } else {
                    "#475569".to_string()
                },
                stroke_width: Some(if is_improved {
                    3.0
                } else if is_fired {
                    2.0
                } else {
                    1.0
                }),
                opacity: Some(if is_fired { 1.0 } else { 0.5 }),
                ..Default::default()
            }));
            // Edge weight label at midpoint.
            let mx = (x1 + x2) / 2.0;
            let my = (y1 + y2) / 2.0;
            shapes.push(Shape::Text(TextShape {
                x: mx,
                y: my - 4.0,
                text: to_fixed(e.weight, 1),
                font_size: Some(9.0),
                fill: Some("#94a3b8".to_string()),
                anchor: Some(Anchor::Middle),
                ..Default::default()
            }));
        }
    }

    // Nodes.
    for v in 0..graph.num_nodes {
        let (x, y) = project(v);
        let color = if v == source {
            "#22c55e".to_string()
        } else {
            distance_color(distance_now[v], max_finite)
        };
        let dist = distance_now[v];
        let title_dist = if dist.is_finite() { to_fixed(dist, 2) } else { "\u{221e}".to_string() };
        shapes.push(Shape::Circle(CircleShape {
            x,
            y,
            r: 14.0,
            fill: color,
            stroke: Some("#0b1220".to_string()),
            stroke_width: Some(2.0),
            title: Some(format!("{}: distance = {}", graph.node_name(v), title_dist)),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x,
            y: y + 4.0,
            text: graph.node_name(v),
            font_size: Some(11.0),
            fill: Some("#0b1220".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        // Distance label below.
        let d_str = if dist.is_finite() { to_fixed(dist, 1) } else { "\u{221e}".to_string() };
        shapes.push(Shape::Text(TextShape {
            x,
            y: y + 26.0,
            text: format!("d={d_str}"),
            font_size: Some(10.0),
            fill: Some("#94a3b8".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    // Sidebar.
    shapes.push(Shape::Rect(RectShape {
        x: META_X,
        y: META_Y,
        w: META_W,
        h: META_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 32.0,
        text: "Shortest Path DES".to_string(),
        font_size: Some(22.0),
        fill: Some("#f1f5f9".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 56.0,
        text: algorithm.label().to_uppercase(),
        font_size: Some(12.0),
        fill: Some("#94a3b8".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    let mut y0 = META_Y + 100.0;
    let improved_count = wave_events.iter().filter(|e| e.improved).count();
    meta_line(&mut shapes, &mut y0, format!("Iteration {}", crate::des::animation::types::js_num(iteration)), "#cbd5e1");
    meta_line(&mut shapes, &mut y0, format!("Source = {}", graph.node_name(source)), "#22c55e");
    meta_line(&mut shapes, &mut y0, format!("Waves this tick = {}", wave_events.len()), "#facc15");
    meta_line(&mut shapes, &mut y0, format!("Improved this tick = {improved_count}"), "#fde68a");
    y0 += 8.0;
    meta_line(&mut shapes, &mut y0, "Settled / known finite distances:".to_string(), "#f1f5f9");
    for v in 0..graph.num_nodes {
        let dist = distance_now[v];
        let d_str = if dist.is_finite() { to_fixed(dist, 2) } else { "\u{221e}".to_string() };
        let color = distance_color(dist, max_finite);
        meta_line(&mut shapes, &mut y0, format!("  {}: {}", graph.node_name(v), d_str), &color);
    }

    let caption = format!(
        "iter {}  waves={}  improved={}",
        crate::des::animation::types::js_num(iteration),
        wave_events.len(),
        improved_count
    );
    FrameParts::with_caption(shapes, caption)
}

pub fn build_shortest_path_charts(
    ticks: &[f64],
    min_finite_distance: &[f64],
    max_finite_distance: &[f64],
) -> Vec<ChartSpec> {
    vec![ChartSpec {
        x: META_X,
        y: META_Y + META_H + 10.0,
        w: META_W,
        h: 100.0,
        title: Some("min / max finite distance per tick".to_string()),
        series: vec![
            ChartSeries {
                label: "min".to_string(),
                color: "#22d3ee".to_string(),
                t: ticks.to_vec(),
                y: min_finite_distance.to_vec(),
            },
            ChartSeries {
                label: "max".to_string(),
                color: "#fde68a".to_string(),
                t: ticks.to_vec(),
                y: max_finite_distance.to_vec(),
            },
        ],
        ..Default::default()
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_graph() -> Graph {
        Graph {
            num_nodes: 2,
            edges: vec![vec![Edge { to: 1, weight: 3.0 }], vec![]],
            coordinates: Some(vec![[0.0, 0.0], [10.0, 0.0]]),
            node_names: Some(vec!["A".to_string(), "B".to_string()]),
        }
    }

    #[test]
    fn infinite_distance_uses_slate_and_infinity_glyph() {
        assert_eq!(distance_color(f64::INFINITY, 5.0), "#475569");
        let g = tiny_graph();
        let fp = build_shortest_path_frame(
            0.0,
            0.0,
            &g,
            &[0.0, f64::INFINITY],
            &[WaveEvent { from: 0, to: 1, new_distance: 3.0, improved: true }],
            0,
            1.0,
            SpAlgorithm::BellmanFordDes,
        );
        let has_inf = fp.shapes.iter().any(|s| match s {
            Shape::Text(t) => t.text.contains('\u{221e}'),
            _ => false,
        });
        assert!(has_inf);
        assert_eq!(fp.caption.as_deref(), Some("iter 1  waves=1  improved=1"));
    }
}
