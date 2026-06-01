//! Generic animation scene for the `numerical_solver_models` family.
//!
//! Every model in `crate::des::general::numerical_solver_models` runs as a
//! `source → solver → sink` pipeline of [`VisualBlockSpec`]s and produces an
//! iteration trace. This scene turns that into frames: the static DES pipeline
//! (rendered with [`render_visual_block_spec`]) is drawn on top, and a
//! convergence curve grows one point per frame underneath, with a moving marker
//! and a live caption. The same series is also returned as a [`ChartSpec`] panel
//! for the player's chart strip.

#![allow(dead_code)]

use crate::des::animation::types::{
    to_exponential, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, Frame,
    LineShape, PathShape, RectShape, Shape, TextShape,
};
use crate::des::general::des_base::visual_block::{
    render_visual_block_spec, ResolvedLayout, VisualBlockSpec,
};

/// Stage size shared by every solver animation.
pub const SOLVER_STAGE_W: f64 = 860.0;
pub const SOLVER_STAGE_H: f64 = 500.0;

const BG: &str = "#0b1021";
const FG: &str = "#e5e7eb";
const MUTED: &str = "#94a3b8";
const GRID: &str = "#1e293b";
const ACCENT: &str = "#38bdf8";
const MARKER: &str = "#f97316";

// Pipeline block layout.
const BW: f64 = 180.0;
const BH: f64 = 64.0;
const BY: f64 = 34.0;
const BX0: f64 = 90.0;
const BSTEP: f64 = 250.0;

// Convergence-plot rectangle.
const PLOT_X: f64 = 86.0;
const PLOT_Y: f64 = 168.0;
const PLOT_W: f64 = 700.0;
const PLOT_H: f64 = 286.0;

/// One `(x, y)` trace series to plot.
#[derive(Clone, Debug)]
pub struct SolverSeries {
    /// Y-axis / metric label (e.g. "objective f(x)").
    pub label: String,
    /// X-axis label (e.g. "iteration").
    pub x_label: String,
    /// Stroke color of the curve.
    pub color: String,
    /// `(x, y)` points in iteration order.
    pub points: Vec<(f64, f64)>,
    /// Decimal places for the formatted metric readout.
    pub decimals: usize,
}

/// Everything needed to animate one solver run.
#[derive(Clone, Debug)]
pub struct SolverAnimationInput {
    pub visual_blocks: Vec<VisualBlockSpec>,
    pub series: SolverSeries,
}

/// Build one frame per series point (curve grown up to that point).
pub fn build_solver_frames(input: &SolverAnimationInput) -> Vec<Frame> {
    let pipeline = pipeline_shapes(&input.visual_blocks);
    let n = input.series.points.len().max(1);
    let (y_min, y_max) = y_range(&input.series.points);
    let (x_min, x_max) = x_range(&input.series.points);

    let mut frames = Vec::with_capacity(n);
    for i in 0..input.series.points.len() {
        let mut shapes = pipeline.clone();
        shapes.extend(plot_shapes(input, i, x_min, x_max, y_min, y_max));
        let (px, py) = input.series.points[i];
        let caption = format!(
            "{} {} — {} = {}",
            input.series.x_label,
            js_int(px),
            input.series.label,
            fmt_val(py, input.series.decimals)
        );
        frames.push(Frame {
            t: i as f64,
            tick: i as f64,
            shapes,
            caption: Some(caption),
        });
    }
    if frames.is_empty() {
        frames.push(Frame {
            t: 0.0,
            tick: 0.0,
            shapes: pipeline,
            caption: Some("no iterations".to_string()),
        });
    }
    frames
}

/// A single chart panel mirroring the series (shown in the player's chart strip).
pub fn build_solver_charts(input: &SolverAnimationInput) -> Vec<ChartSpec> {
    let t: Vec<f64> = input.series.points.iter().map(|p| p.0).collect();
    let y: Vec<f64> = input.series.points.iter().map(|p| p.1).collect();
    vec![ChartSpec {
        x: 24.0,
        y: SOLVER_STAGE_H + 16.0,
        w: SOLVER_STAGE_W - 48.0,
        h: 150.0,
        title: Some(input.series.label.clone()),
        y_min: None,
        y_max: None,
        y_label: Some(input.series.label.clone()),
        series: vec![ChartSeries {
            label: input.series.label.clone(),
            color: input.series.color.clone(),
            t,
            y,
        }],
        cursor: Some(true),
    }]
}

// ── Pipeline rendering ────────────────────────────────────────────────────────

/// Render the `source → solver → sink` blocks left-to-right with arrows between
/// them. Block specs are re-laid-out so they flow horizontally.
fn pipeline_shapes(blocks: &[VisualBlockSpec]) -> Vec<Shape> {
    let mut shapes: Vec<Shape> = Vec::new();
    shapes.push(Shape::Text(TextShape {
        x: SOLVER_STAGE_W / 2.0,
        y: 20.0,
        text: "discrete-event solver pipeline".to_string(),
        font_size: Some(13.0),
        fill: Some(MUTED.to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        font_family: None,
        visual_block_id: None,
    }));
    for (i, spec) in blocks.iter().enumerate() {
        let bx = BX0 + i as f64 * BSTEP;
        let mut placed = spec.clone();
        placed.layout = ResolvedLayout {
            x: bx,
            y: BY,
            w: BW,
            h: BH,
        };
        shapes.extend(render_visual_block_spec(&placed));
        if i + 1 < blocks.len() {
            let x1 = bx + BW;
            let x2 = bx + BSTEP;
            let y = BY + BH / 2.0;
            shapes.push(Shape::Line(LineShape {
                x1,
                y1: y,
                x2: x2 - 8.0,
                y2: y,
                stroke: ACCENT.to_string(),
                stroke_width: Some(2.0),
                opacity: None,
                dasharray: None,
                visual_block_id: None,
            }));
            shapes.push(Shape::Path(PathShape {
                d: format!(
                    "M {} {} L {} {} L {} {} Z",
                    x2 - 8.0,
                    y - 5.0,
                    x2,
                    y,
                    x2 - 8.0,
                    y + 5.0
                ),
                stroke: None,
                stroke_width: None,
                fill: Some(ACCENT.to_string()),
                opacity: None,
                visual_block_id: None,
            }));
        }
    }
    shapes
}

// ── Convergence plot ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn plot_shapes(
    input: &SolverAnimationInput,
    upto: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<Shape> {
    let pts = &input.series.points;
    let mut shapes: Vec<Shape> = Vec::new();

    // Plot panel background + frame.
    shapes.push(Shape::Rect(RectShape {
        x: PLOT_X,
        y: PLOT_Y,
        w: PLOT_W,
        h: PLOT_H,
        fill: "#0f172a".to_string(),
        stroke: Some(GRID.to_string()),
        stroke_width: Some(1.0),
        opacity: None,
        label: None,
        rx: Some(6.0),
        title: None,
        visual_block_id: None,
    }));

    let map_x = |x: f64| -> f64 {
        if (x_max - x_min).abs() < 1e-12 {
            PLOT_X + PLOT_W / 2.0
        } else {
            PLOT_X + (x - x_min) / (x_max - x_min) * PLOT_W
        }
    };
    let map_y = |y: f64| -> f64 {
        if (y_max - y_min).abs() < 1e-12 {
            PLOT_Y + PLOT_H / 2.0
        } else {
            PLOT_Y + PLOT_H - (y - y_min) / (y_max - y_min) * PLOT_H
        }
    };

    // Horizontal gridlines + y labels (top = max, bottom = min).
    for k in 0..=2 {
        let frac = k as f64 / 2.0;
        let yval = y_max - frac * (y_max - y_min);
        let py = PLOT_Y + frac * PLOT_H;
        shapes.push(Shape::Line(LineShape {
            x1: PLOT_X,
            y1: py,
            x2: PLOT_X + PLOT_W,
            y2: py,
            stroke: GRID.to_string(),
            stroke_width: Some(1.0),
            opacity: Some(0.7),
            dasharray: Some("3,4".to_string()),
            visual_block_id: None,
        }));
        shapes.push(Shape::Text(TextShape {
            x: PLOT_X - 8.0,
            y: py + 4.0,
            text: fmt_val(yval, input.series.decimals),
            font_size: Some(10.0),
            fill: Some(MUTED.to_string()),
            anchor: Some(Anchor::End),
            font_weight: None,
            font_family: None,
            visual_block_id: None,
        }));
    }

    // Axis labels.
    shapes.push(Shape::Text(TextShape {
        x: PLOT_X,
        y: PLOT_Y - 8.0,
        text: input.series.label.clone(),
        font_size: Some(12.0),
        fill: Some(FG.to_string()),
        anchor: Some(Anchor::Start),
        font_weight: Some(FontWeight::Bold),
        font_family: None,
        visual_block_id: None,
    }));
    shapes.push(Shape::Text(TextShape {
        x: PLOT_X + PLOT_W,
        y: PLOT_Y + PLOT_H + 18.0,
        text: input.series.x_label.clone(),
        font_size: Some(11.0),
        fill: Some(MUTED.to_string()),
        anchor: Some(Anchor::End),
        font_weight: None,
        font_family: None,
        visual_block_id: None,
    }));

    // Growing polyline up to `upto`.
    if upto >= 1 {
        let mut d = String::new();
        for (idx, &(x, y)) in pts.iter().take(upto + 1).enumerate() {
            let px = map_x(x);
            let py = map_y(y);
            if idx == 0 {
                d.push_str(&format!("M {:.2} {:.2}", px, py));
            } else {
                d.push_str(&format!(" L {:.2} {:.2}", px, py));
            }
        }
        shapes.push(Shape::Path(PathShape {
            d,
            stroke: Some(input.series.color.clone()),
            stroke_width: Some(2.0),
            fill: Some("none".to_string()),
            opacity: None,
            visual_block_id: None,
        }));
    }

    // Current-point marker + readout.
    if let Some(&(x, y)) = pts.get(upto) {
        let px = map_x(x);
        let py = map_y(y);
        shapes.push(Shape::Circle(CircleShape {
            x: px,
            y: py,
            r: 4.5,
            fill: MARKER.to_string(),
            stroke: Some("#ffffff".to_string()),
            stroke_width: Some(1.5),
            opacity: None,
            label: None,
            title: None,
            visual_block_id: None,
        }));
        shapes.push(Shape::Text(TextShape {
            x: (px + 10.0).min(PLOT_X + PLOT_W - 4.0),
            y: (py - 8.0).max(PLOT_Y + 12.0),
            text: fmt_val(y, input.series.decimals),
            font_size: Some(11.0),
            fill: Some(MARKER.to_string()),
            anchor: Some(Anchor::Start),
            font_weight: Some(FontWeight::Bold),
            font_family: None,
            visual_block_id: None,
        }));
    }

    shapes
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn x_range(points: &[(f64, f64)]) -> (f64, f64) {
    let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (0.0, 1.0)
    }
}

fn y_range(points: &[(f64, f64)]) -> (f64, f64) {
    let ys: Vec<f64> = points
        .iter()
        .map(|p| p.1)
        .filter(|v| v.is_finite())
        .collect();
    if ys.is_empty() {
        return (0.0, 1.0);
    }
    let mut min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        min -= 1.0;
        max += 1.0;
    } else {
        let pad = (max - min) * 0.06;
        min -= pad;
        max += pad;
    }
    (min, max)
}

fn js_int(x: f64) -> String {
    format!("{}", x.round() as i64)
}

fn fmt_val(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "n/a".to_string();
    }
    let abs = v.abs();
    if abs != 0.0 && (abs >= 1.0e4 || abs < 1.0e-3) {
        to_exponential(v, 2)
    } else {
        to_fixed(v, decimals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::numerical_solver_models::{run_lbfgs, LbfgsParams};

    #[test]
    fn builds_a_frame_per_trace_point() {
        let result = run_lbfgs(LbfgsParams::default());
        let input = SolverAnimationInput {
            visual_blocks: result.visual_blocks.clone(),
            series: SolverSeries {
                label: "objective f(x)".to_string(),
                x_label: "iteration".to_string(),
                color: ACCENT.to_string(),
                points: result
                    .trace
                    .iter()
                    .map(|r| (r.iteration as f64, r.value))
                    .collect(),
                decimals: 4,
            },
        };
        let frames = build_solver_frames(&input);
        assert_eq!(frames.len(), result.trace.len());
        // Each frame carries the static 3-block pipeline plus plot shapes.
        assert!(frames[0].shapes.len() > 3);
        assert!(frames.last().unwrap().caption.is_some());
        let charts = build_solver_charts(&input);
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].series[0].t.len(), result.trace.len());
    }
}
