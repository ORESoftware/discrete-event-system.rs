//! Port of `src/des/animation/scenes/genetic-tsp-scene.ts`.
//!
//! Builds frames + charts for the genetic-algorithm TSP animation: the DES
//! station chain (Selection → … → Replacement) with chromosomes in flight on
//! the left, the elite-tour polygon over the cities on the lower left, and a
//! telemetry sidebar on the right.
//!
//! ## Conversion notes
//!
//! * `drawStation` pushes into a shared array → takes `&mut Vec<Shape>`.
//! * `STATION_NAMES` is a `const` slice.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::genetic_tsp::{TSPInstance, Tour}` the scene reads is
//!   mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1200.0;
pub const STAGE_H: f64 = 720.0;

// Architecture pipeline panel.
const ARCH_X: f64 = 20.0;
const ARCH_Y: f64 = 40.0;
const ARCH_W: f64 = 700.0;
const ARCH_H: f64 = 200.0;
// Cities + tour view (right panel, below the architecture).
const VIEW_X: f64 = 20.0;
const VIEW_Y: f64 = 260.0;
const VIEW_W: f64 = 700.0;
const VIEW_H: f64 = 440.0;
// Sidebar.
const META_X: f64 = 740.0;
const META_Y: f64 = 40.0;
const META_W: f64 = 440.0;
const META_H: f64 = 660.0;

const STATION_NAMES: [&str; 6] =
    ["Selection", "Crossover", "Mutation", "Feasibility", "Fitness", "Replacement"];
const STATION_COLOR: &str = "#1e293b";
const STATION_ACTIVE_FILL: &str = "#fef3c7";
const STATION_ACTIVE_STROKE: &str = "#f59e0b";

// PORT NOTE: local mirror of the genetic-TSP model (subset used by the scene).
#[derive(Clone, Debug, Default)]
pub struct TSPInstance {
    pub n: usize,
    pub coordinates: Vec<[f64; 2]>,
    pub precedence: Option<Vec<[usize; 2]>>,
}

/// A tour: a permutation of city indices.
pub type Tour = Vec<usize>;

/// `Partial<ArchitectureFrameArgs>` — every field optional; only `phase` is read.
#[derive(Clone, Debug, Default)]
pub struct ArchitectureFrameArgs {
    pub generation: Option<f64>,
    pub phase: Option<usize>,
    pub cut_this_gen: Option<f64>,
    pub accept_this_gen: Option<f64>,
}

/// Inputs to [`build_genetic_tsp_frame`].
pub struct GeneticTSPFrameArgs<'a> {
    pub instance: &'a TSPInstance,
    pub elite_tour: &'a Tour,
    pub best: f64,
    pub mean: f64,
    pub worst: f64,
    pub generation: f64,
    pub num_feasible_children: f64,
    pub num_infeasible_children: f64,
    pub precedence_count: f64,
    pub arch: Option<ArchitectureFrameArgs>,
}

#[allow(clippy::too_many_arguments)]
fn draw_station(
    shapes: &mut Vec<Shape>,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    title: &str,
    sub: &str,
    active: bool,
) {
    shapes.push(Shape::Rect(RectShape {
        x: cx - w / 2.0,
        y: cy - h / 2.0,
        w,
        h,
        fill: if active { STATION_ACTIVE_FILL.to_string() } else { STATION_COLOR.to_string() },
        stroke: Some(if active { STATION_ACTIVE_STROKE.to_string() } else { "#475569".to_string() }),
        stroke_width: Some(if active { 3.0 } else { 1.5 }),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: cx,
        y: cy - 6.0,
        text: title.to_string(),
        font_size: Some(12.0),
        fill: Some(if active { "#92400e".to_string() } else { "#fde68a".to_string() }),
        font_weight: Some(FontWeight::Bold),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    if !sub.is_empty() {
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: cy + 12.0,
            text: sub.to_string(),
            font_size: Some(10.0),
            fill: Some(if active { "#1f2937".to_string() } else { "#cbd5e1".to_string() }),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
}

fn meta_line(shapes: &mut Vec<Shape>, y: &mut f64, text: String, color: &str) {
    shapes.push(Shape::Text(TextShape {
        x: META_X + 20.0,
        y: *y,
        text,
        font_size: Some(12.0),
        fill: Some(color.to_string()),
        ..Default::default()
    }));
    *y += 22.0;
}

pub fn build_genetic_tsp_frame(_t: f64, _tick: f64, args: &GeneticTSPFrameArgs) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let instance = args.instance;

    // ============ ARCHITECTURE (top panel) ============
    shapes.push(Shape::Rect(RectShape {
        x: ARCH_X,
        y: ARCH_Y,
        w: ARCH_W,
        h: ARCH_H,
        fill: "#0b1220".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.5),
        rx: Some(6.0),
        ..Default::default()
    }));
    let phase = args.arch.as_ref().and_then(|a| a.phase).unwrap_or(0);
    shapes.push(Shape::Text(TextShape {
        x: ARCH_X + ARCH_W / 2.0,
        y: ARCH_Y + 22.0,
        text: format!(
            "GA station chain — generation {} — phase: {}",
            js_num(args.generation),
            STATION_NAMES[phase]
        ),
        font_size: Some(13.0),
        fill: Some("#f1f5f9".to_string()),
        font_weight: Some(FontWeight::Bold),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));

    let n = STATION_NAMES.len();
    let station_w = 95.0;
    let station_h = 60.0;
    let padding = 12.0;
    let total_w = n as f64 * station_w + (n as f64 - 1.0) * padding * 2.0;
    let start_x = ARCH_X + (ARCH_W - total_w) / 2.0 + station_w / 2.0;
    let station_y = ARCH_Y + 100.0;
    let mut station_xs: Vec<f64> = Vec::new();
    for i in 0..n {
        let x = start_x + i as f64 * (station_w + padding * 2.0);
        station_xs.push(x);
        let sub_text = match i {
            0 => "tournament".to_string(),
            1 => "OX".to_string(),
            2 => "inv/swap".to_string(),
            3 => format!("cut {}", js_num(args.num_infeasible_children)),
            4 => "tour len".to_string(),
            _ => "\u{03bc}+\u{03bb}".to_string(),
        };
        draw_station(&mut shapes, x, station_y, station_w, station_h, STATION_NAMES[i], &sub_text, phase == i);
    }
    // Edges with chromosomes in flight.
    for i in 0..n - 1 {
        let x1 = station_xs[i] + station_w / 2.0;
        let x2 = station_xs[i + 1] - station_w / 2.0;
        let y = station_y;
        shapes.push(Shape::Line(LineShape {
            x1,
            y1: y,
            x2,
            y2: y,
            stroke: "#64748b".to_string(),
            stroke_width: Some(1.5),
            opacity: Some(0.7),
            ..Default::default()
        }));
        // Arrow tip.
        shapes.push(Shape::Line(LineShape {
            x1: x2 - 6.0,
            y1: y - 3.0,
            x2,
            y2: y,
            stroke: "#64748b".to_string(),
            stroke_width: Some(1.5),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x2 - 6.0,
            y1: y + 3.0,
            x2,
            y2: y,
            stroke: "#64748b".to_string(),
            stroke_width: Some(1.5),
            ..Default::default()
        }));
        // Movables (chromosome dots) in transit between active phase and the next.
        let in_flight = phase == i;
        if in_flight {
            let num_dots = if i == 3 { 3 } else { 5 };
            let is_cut_edge = i == 3;
            for k in 0..num_dots {
                let tt = 0.2 + (k as f64 / num_dots as f64) * 0.6;
                let dx = x1 + (x2 - x1) * tt;
                let cut_this = is_cut_edge && k == 0 && args.num_infeasible_children > 0.0;
                if cut_this {
                    shapes.push(Shape::Text(TextShape {
                        x: dx,
                        y: y + 4.0,
                        text: "\u{2717}".to_string(),
                        font_size: Some(14.0),
                        fill: Some("#ef4444".to_string()),
                        anchor: Some(Anchor::Middle),
                        font_weight: Some(FontWeight::Bold),
                        ..Default::default()
                    }));
                } else {
                    shapes.push(Shape::Circle(CircleShape {
                        x: dx,
                        y,
                        r: 3.0,
                        fill: "#22d3ee".to_string(),
                        stroke: Some("#0b1220".to_string()),
                        stroke_width: Some(0.5),
                        ..Default::default()
                    }));
                }
            }
        }
    }
    // Side annotation: "cut" branch from feasibility station drops down.
    if args.num_infeasible_children > 0.0 && phase == 3 {
        let f_x = station_xs[3];
        shapes.push(Shape::Line(LineShape {
            x1: f_x,
            y1: station_y + station_h / 2.0,
            x2: f_x,
            y2: station_y + station_h / 2.0 + 28.0,
            stroke: "#ef4444".to_string(),
            stroke_width: Some(1.5),
            dasharray: Some("3 3".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: f_x,
            y: station_y + station_h / 2.0 + 42.0,
            text: format!("{} cut", js_num(args.num_infeasible_children)),
            font_size: Some(10.0),
            fill: Some("#ef4444".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    // ============ TOUR / CITIES (left bottom) ============
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for c in &instance.coordinates {
        let (x, y) = (c[0], c[1]);
        if x < x_min {
            x_min = x;
        }
        if x > x_max {
            x_max = x;
        }
        if y < y_min {
            y_min = y;
        }
        if y > y_max {
            y_max = y;
        }
    }
    let pad = 30.0;
    let sx = (VIEW_W - 2.0 * pad) / (x_max - x_min).max(1e-9);
    let sy = (VIEW_H - 2.0 * pad) / (y_max - y_min).max(1e-9);
    let project = |p: [f64; 2]| -> (f64, f64) {
        (VIEW_X + pad + (p[0] - x_min) * sx, VIEW_Y + pad + (p[1] - y_min) * sy)
    };

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
    shapes.push(Shape::Text(TextShape {
        x: VIEW_X + VIEW_W / 2.0,
        y: VIEW_Y + 18.0,
        text: format!(
            "Elite tour for generation {} (length = {})",
            js_num(args.generation),
            to_fixed(args.best, 2)
        ),
        font_size: Some(13.0),
        fill: Some("#f1f5f9".to_string()),
        font_weight: Some(FontWeight::Bold),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));

    // Tour polygon.
    for i in 0..instance.n {
        let (x1, y1) = project(instance.coordinates[args.elite_tour[i]]);
        let (x2, y2) = project(instance.coordinates[args.elite_tour[(i + 1) % instance.n]]);
        shapes.push(Shape::Line(LineShape {
            x1,
            y1,
            x2,
            y2,
            stroke: "#22d3ee".to_string(),
            stroke_width: Some(2.0),
            opacity: Some(0.9),
            ..Default::default()
        }));
    }
    // Cities.
    for i in 0..instance.n {
        let (x, y) = project(instance.coordinates[i]);
        shapes.push(Shape::Circle(CircleShape {
            x,
            y,
            r: 6.0,
            fill: "#fde68a".to_string(),
            stroke: Some("#f59e0b".to_string()),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + 9.0,
            y: y + 4.0,
            text: i.to_string(),
            font_size: Some(10.0),
            fill: Some("#cbd5e1".to_string()),
            ..Default::default()
        }));
    }
    // Precedence arcs.
    if let Some(precedence) = &instance.precedence {
        for pair in precedence {
            let (x1, y1) = project(instance.coordinates[pair[0]]);
            let (x2, y2) = project(instance.coordinates[pair[1]]);
            shapes.push(Shape::Line(LineShape {
                x1,
                y1,
                x2,
                y2,
                stroke: "#ef4444".to_string(),
                stroke_width: Some(1.0),
                dasharray: Some("3,3".to_string()),
                opacity: Some(0.5),
                ..Default::default()
            }));
        }
    }

    // ============ SIDEBAR ============
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
        text: "Genetic-TSP".to_string(),
        font_size: Some(22.0),
        fill: Some("#f1f5f9".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 60.0,
        text: format!("Generation {}", js_num(args.generation)),
        font_size: Some(14.0),
        fill: Some("#cbd5e1".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 78.0,
        text: format!("Phase: {}", STATION_NAMES[phase]),
        font_size: Some(12.0),
        fill: Some("#fde68a".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    let mut y = META_Y + 110.0;
    meta_line(&mut shapes, &mut y, format!("best  tour length = {}", to_fixed(args.best, 2)), "#22d3ee");
    meta_line(&mut shapes, &mut y, format!("mean  tour length = {}", to_fixed(args.mean, 2)), "#94a3b8");
    meta_line(&mut shapes, &mut y, format!("worst tour length = {}", to_fixed(args.worst, 2)), "#ef4444");
    y += 8.0;
    meta_line(&mut shapes, &mut y, format!("# feasible kids   = {}", js_num(args.num_feasible_children)), "#22c55e");
    meta_line(&mut shapes, &mut y, format!("# cut (infeasible)= {}", js_num(args.num_infeasible_children)), "#facc15");
    y += 8.0;
    meta_line(&mut shapes, &mut y, format!("# cities            = {}", instance.n), "#cbd5e1");
    meta_line(&mut shapes, &mut y, format!("# precedence pairs  = {}", js_num(args.precedence_count)), "#cbd5e1");
    y += 8.0;
    // Architecture legend.
    meta_line(&mut shapes, &mut y, "Architecture legend:".to_string(), "#f1f5f9");
    shapes.push(Shape::Circle(CircleShape {
        x: META_X + 30.0,
        y: y - 4.0,
        r: 4.0,
        fill: "#22d3ee".to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + 42.0,
        y,
        text: "chromosome (movable)".to_string(),
        font_size: Some(11.0),
        fill: Some("#cbd5e1".to_string()),
        ..Default::default()
    }));
    y += 18.0;
    shapes.push(Shape::Text(TextShape {
        x: META_X + 30.0,
        y: y - 1.0,
        text: "\u{2717}".to_string(),
        font_size: Some(12.0),
        fill: Some("#ef4444".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + 42.0,
        y,
        text: "cut by feasibility station".to_string(),
        font_size: Some(11.0),
        fill: Some("#cbd5e1".to_string()),
        ..Default::default()
    }));

    let caption = format!(
        "gen={} phase={} best={}",
        js_num(args.generation),
        STATION_NAMES[phase],
        to_fixed(args.best, 2)
    );
    FrameParts::with_caption(shapes, caption)
}

pub fn build_genetic_tsp_charts(generations: &[f64], best: &[f64], mean: &[f64]) -> Vec<ChartSpec> {
    vec![ChartSpec {
        x: META_X,
        y: META_Y + 380.0,
        w: META_W,
        h: 280.0,
        title: Some("Best & mean tour length per generation".to_string()),
        series: vec![
            ChartSeries { label: "best".to_string(), color: "#22d3ee".to_string(), t: generations.to_vec(), y: best.to_vec() },
            ChartSeries { label: "mean".to_string(), color: "#94a3b8".to_string(), t: generations.to_vec(), y: mean.to_vec() },
        ],
        ..Default::default()
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_caption_reports_phase_and_best() {
        let instance = TSPInstance {
            n: 3,
            coordinates: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            precedence: Some(vec![[0, 2]]),
        };
        let elite: Tour = vec![0, 1, 2];
        let args = GeneticTSPFrameArgs {
            instance: &instance,
            elite_tour: &elite,
            best: 3.5,
            mean: 4.0,
            worst: 5.0,
            generation: 7.0,
            num_feasible_children: 10.0,
            num_infeasible_children: 2.0,
            precedence_count: 1.0,
            arch: Some(ArchitectureFrameArgs { phase: Some(3), ..Default::default() }),
        };
        let fp = build_genetic_tsp_frame(0.0, 0.0, &args);
        assert_eq!(fp.caption.as_deref(), Some("gen=7 phase=Feasibility best=3.50"));
    }
}
