//! Port of `src/des/animation/scenes/incremental-lp-scene.ts`.
//!
//! Builds frames + charts for the incremental-LP animation: a 2-D polytope with
//! the objective gradient and the simplex trajectory on the left, and a simplex
//! tableau readout on the right. With 3+ structural variables the polytope view
//! degrades to a notice.
//!
//! ## Conversion notes
//!
//! * `projectFn(...)` returns a closure → [`project_fn`] returns `impl Fn`.
//! * `computePolytopeVertices(A, b)` works over `&[Vec<f64>]` / `&[f64]`.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::incremental_lp::LPSnapshot` the scene reads is
//!   mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1180.0;
pub const STAGE_H: f64 = 700.0;
const POLY_X: f64 = 30.0;
const POLY_Y: f64 = 40.0;
const POLY_W: f64 = 600.0;
const POLY_H: f64 = 600.0;
const TAB_X: f64 = 660.0;
const TAB_Y: f64 = 40.0;
const TAB_W: f64 = 490.0;
const TAB_H: f64 = 600.0;

const VIEW_PAD: f64 = 30.0;

/// `'max' | 'min'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Max,
    Min,
}

impl Sense {
    fn label(self) -> &'static str {
        match self {
            Sense::Max => "max",
            Sense::Min => "min",
        }
    }
}

// PORT NOTE: local mirror of the incremental-LP snapshot (subset used here).
#[derive(Clone, Debug, Default)]
pub struct LPSnapshot {
    pub num_struct: usize,
    pub num_constraints: usize,
    pub x: Vec<f64>,
    pub z: f64,
    pub tick: f64,
    pub mode: String,
    pub is_optimal: bool,
    pub primal_feasible: bool,
    pub dual_feasible: bool,
    pub basis: Vec<usize>,
    pub var_names: Vec<String>,
    pub con_names: Vec<String>,
    pub reduced_costs: Vec<f64>,
    pub rhs: Vec<f64>,
}

/// Inputs to [`build_incremental_lp_frame`].
pub struct IncrementalLPFrameArgs<'a> {
    pub snap: &'a LPSnapshot,
    /// Current full-form constraint matrix (m × n_struct).
    pub a: &'a [Vec<f64>],
    /// Current full-form rhs (length m).
    pub b: &'a [f64],
    /// Current objective (length n_struct).
    pub c: &'a [f64],
    pub sense: Sense,
    /// List of past x* visited (for the simplex trail).
    pub history: &'a [Vec<f64>],
    pub event_label: Option<String>,
    pub event_flash: Option<f64>,
    pub pivot_label: Option<String>,
}

fn project_fn(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> impl Fn(f64, f64) -> (f64, f64) {
    let sx = (POLY_W - 2.0 * VIEW_PAD) / (x_max - x_min).max(1e-9);
    let sy = (POLY_H - 2.0 * VIEW_PAD) / (y_max - y_min).max(1e-9);
    move |x: f64, y: f64| {
        (
            POLY_X + VIEW_PAD + (x - x_min) * sx,
            POLY_Y + POLY_H - VIEW_PAD - (y - y_min) * sy,
        )
    }
}

/// Compute polytope vertices as the feasible pairwise constraint intersections
/// (plus `x ≥ 0`), sorted by polar angle around the centroid.
fn compute_polytope_vertices(a: &[Vec<f64>], b: &[f64]) -> Vec<[f64; 2]> {
    let mut all: Vec<[f64; 2]> = Vec::new();
    let mut ax: Vec<[f64; 2]> = a.iter().map(|r| [r[0], r[1]]).collect();
    ax.push([-1.0, 0.0]);
    ax.push([0.0, -1.0]);
    let mut bx: Vec<f64> = b.to_vec();
    bx.push(0.0);
    bx.push(0.0);
    for i in 0..ax.len() {
        for j in i + 1..ax.len() {
            let (a1, c1) = (ax[i][0], ax[i][1]);
            let (a2, c2) = (ax[j][0], ax[j][1]);
            let det = a1 * c2 - a2 * c1;
            if det.abs() < 1e-9 {
                continue;
            }
            let x = (bx[i] * c2 - bx[j] * c1) / det;
            let y = (a1 * bx[j] - a2 * bx[i]) / det;
            let mut ok = true;
            for k in 0..ax.len() {
                if ax[k][0] * x + ax[k][1] * y > bx[k] + 1e-7 {
                    ok = false;
                    break;
                }
            }
            if ok {
                all.push([x, y]);
            }
        }
    }
    if all.is_empty() {
        return vec![];
    }
    let cx = all.iter().map(|p| p[0]).sum::<f64>() / all.len() as f64;
    let cy = all.iter().map(|p| p[1]).sum::<f64>() / all.len() as f64;
    all.sort_by(|u, v| {
        (u[1] - cy)
            .atan2(u[0] - cx)
            .partial_cmp(&(v[1] - cy).atan2(v[0] - cx))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<[f64; 2]> = Vec::new();
    for p in &all {
        let keep = match out.last() {
            None => true,
            Some(last) => (p[0] - last[0]).hypot(p[1] - last[1]) > 1e-6,
        };
        if keep {
            out.push(*p);
        }
    }
    if out.len() > 1 {
        let f = out[0];
        let l = out[out.len() - 1];
        if (f[0] - l[0]).hypot(f[1] - l[1]) < 1e-6 {
            out.pop();
        }
    }
    out
}

fn var_name_or(snap: &LPSnapshot, idx: usize, fallback: String) -> String {
    snap.var_names.get(idx).cloned().unwrap_or(fallback)
}

fn con_name_or(snap: &LPSnapshot, idx: usize, fallback: String) -> String {
    snap.con_names.get(idx).cloned().unwrap_or(fallback)
}

/// `basis[i]`'s display name (`s${...}` lacks `??`, so this matches JS's
/// `undefined` only on the bare `varNames[b]` path).
fn basis_name(snap: &LPSnapshot, col: usize) -> String {
    if col < snap.num_struct {
        // No `??` fallback in the TS for this path.
        snap.var_names
            .get(col)
            .cloned()
            .unwrap_or_else(|| "undefined".to_string())
    } else {
        let idx = col - snap.num_struct;
        format!("{}_s", con_name_or(snap, idx, format!("s{}", idx + 1)))
    }
}

pub fn build_incremental_lp_frame(
    _t: f64,
    _tick: f64,
    args: &IncrementalLPFrameArgs,
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let snap = args.snap;
    let (a, b, c, sense, history) = (args.a, args.b, args.c, args.sense, args.history);

    // ---------------- Left panel: polytope ----------------
    shapes.push(Shape::Rect(RectShape {
        x: POLY_X,
        y: POLY_Y,
        w: POLY_W,
        h: POLY_H,
        fill: "#0b1220".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: POLY_X + POLY_W / 2.0,
        y: POLY_Y + 26.0,
        text: "Polytope (feasible region) + simplex trajectory".to_string(),
        font_size: Some(14.0),
        fill: Some("#f1f5f9".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    if snap.num_struct == 2 {
        let verts = compute_polytope_vertices(a, b);
        let mut x_min = 0.0;
        let mut x_max = 1.0;
        let mut y_min = 0.0;
        let mut y_max = 1.0;
        for v in &verts {
            if v[0] < x_min {
                x_min = v[0];
            }
            if v[0] > x_max {
                x_max = v[0];
            }
            if v[1] < y_min {
                y_min = v[1];
            }
            if v[1] > y_max {
                y_max = v[1];
            }
        }
        for x in history {
            if x[0] < x_min {
                x_min = x[0];
            }
            if x[0] > x_max {
                x_max = x[0];
            }
            if x[1] < y_min {
                y_min = x[1];
            }
            if x[1] > y_max {
                y_max = x[1];
            }
        }
        if snap.x[0] > x_max {
            x_max = snap.x[0];
        }
        if snap.x[1] > y_max {
            y_max = snap.x[1];
        }
        x_max = x_max.max(1.0);
        y_max = y_max.max(1.0);
        let pad_factor = 0.08;
        x_max += (x_max - x_min) * pad_factor;
        y_max += (y_max - y_min) * pad_factor;
        let project = project_fn(x_min, x_max, y_min, y_max);
        // Axes.
        let (ox, oy) = project(0.0, 0.0);
        let ax_end = project(x_max, 0.0).0;
        let y_axis_end = project(0.0, y_max).1;
        shapes.push(Shape::Line(LineShape {
            x1: ox,
            y1: oy,
            x2: ax_end,
            y2: oy,
            stroke: "#334155".to_string(),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: ox,
            y1: oy,
            x2: ox,
            y2: y_axis_end,
            stroke: "#334155".to_string(),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: ax_end - 8.0,
            y: oy + 16.0,
            text: var_name_or(snap, 0, "x1".to_string()),
            font_size: Some(11.0),
            fill: Some("#94a3b8".to_string()),
            anchor: Some(Anchor::End),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: ox - 8.0,
            y: y_axis_end + 12.0,
            text: var_name_or(snap, 1, "x2".to_string()),
            font_size: Some(11.0),
            fill: Some("#94a3b8".to_string()),
            anchor: Some(Anchor::End),
            ..Default::default()
        }));
        // Polytope (filled polygon path).
        if verts.len() >= 2 {
            let mut d = String::new();
            for (k, v) in verts.iter().enumerate() {
                let (px, py) = project(v[0], v[1]);
                d.push_str(if k == 0 { "M" } else { "L" });
                d.push_str(&js_num(px));
                d.push(',');
                d.push_str(&js_num(py));
                d.push(' ');
            }
            d.push('Z');
            shapes.push(Shape::Path(PathShape {
                d,
                fill: Some("#1e293b".to_string()),
                stroke: Some("#38bdf8".to_string()),
                stroke_width: Some(2.0),
                opacity: Some(0.8),
                ..Default::default()
            }));
        }
        // Each constraint line.
        for i in 0..a.len() {
            let a1 = a[i][0];
            let a2 = a[i][1];
            let rhs = b[i];
            let mut pts: Vec<[f64; 2]> = Vec::new();
            if a2.abs() > 1e-9 {
                for x_c in [x_min, x_max] {
                    let y_c = (rhs - a1 * x_c) / a2;
                    if y_c >= y_min - 1e-3 && y_c <= y_max + 1e-3 {
                        pts.push([x_c, y_c]);
                    }
                }
            }
            if a1.abs() > 1e-9 {
                for y_c in [y_min, y_max] {
                    let x_c = (rhs - a2 * y_c) / a1;
                    if x_c >= x_min - 1e-3 && x_c <= x_max + 1e-3 {
                        pts.push([x_c, y_c]);
                    }
                }
            }
            if pts.len() >= 2 {
                let (pa, pb) = (pts[0], pts[1]);
                let (x1, y1) = project(pa[0], pa[1]);
                let (x2, y2) = project(pb[0], pb[1]);
                shapes.push(Shape::Line(LineShape {
                    x1,
                    y1,
                    x2,
                    y2,
                    stroke: "#38bdf8".to_string(),
                    stroke_width: Some(1.5),
                    dasharray: Some("4,3".to_string()),
                    opacity: Some(0.7),
                    ..Default::default()
                }));
                let mx = (x1 + x2) / 2.0;
                let my = (y1 + y2) / 2.0;
                shapes.push(Shape::Text(TextShape {
                    x: mx,
                    y: my - 4.0,
                    text: con_name_or(snap, i, format!("c{}", i + 1)),
                    font_size: Some(10.0),
                    fill: Some("#7dd3fc".to_string()),
                    anchor: Some(Anchor::Middle),
                    ..Default::default()
                }));
            }
        }
        // Objective gradient arrow at the centroid of the polytope.
        if !verts.is_empty() {
            let cx = verts.iter().map(|p| p[0]).sum::<f64>() / verts.len() as f64;
            let cy = verts.iter().map(|p| p[1]).sum::<f64>() / verts.len() as f64;
            let norm = {
                let h = c[0].hypot(c[1]);
                if h == 0.0 {
                    1.0
                } else {
                    h
                }
            };
            let dir_x = c[0] / norm * (x_max - x_min) * 0.18;
            let dir_y = c[1] / norm * (y_max - y_min) * 0.18;
            let (ax_, ay) = project(cx, cy);
            let (bx, by) = project(cx + dir_x, cy + dir_y);
            shapes.push(Shape::Line(LineShape {
                x1: ax_,
                y1: ay,
                x2: bx,
                y2: by,
                stroke: "#f59e0b".to_string(),
                stroke_width: Some(3.0),
                ..Default::default()
            }));
            let dx = bx - ax_;
            let dy = by - ay;
            let a_len = {
                let h = dx.hypot(dy);
                if h == 0.0 {
                    1.0
                } else {
                    h
                }
            };
            let ux = dx / a_len;
            let uy = dy / a_len;
            let px = -uy;
            let py = ux;
            let head_size = 8.0;
            shapes.push(Shape::Line(LineShape {
                x1: bx,
                y1: by,
                x2: bx - ux * head_size + px * head_size / 2.0,
                y2: by - uy * head_size + py * head_size / 2.0,
                stroke: "#f59e0b".to_string(),
                stroke_width: Some(3.0),
                ..Default::default()
            }));
            shapes.push(Shape::Line(LineShape {
                x1: bx,
                y1: by,
                x2: bx - ux * head_size - px * head_size / 2.0,
                y2: by - uy * head_size - py * head_size / 2.0,
                stroke: "#f59e0b".to_string(),
                stroke_width: Some(3.0),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: bx + 8.0,
                y: by - 4.0,
                text: format!(
                    "{} \u{2207}c = ({}, {})",
                    sense.label(),
                    js_num(c[0]),
                    js_num(c[1])
                ),
                font_size: Some(11.0),
                fill: Some("#fbbf24".to_string()),
                ..Default::default()
            }));
        }
        // Trail of past x*'s.
        for k in 1..history.len() {
            let (x1, y1) = project(history[k - 1][0], history[k - 1][1]);
            let (x2, y2) = project(history[k][0], history[k][1]);
            shapes.push(Shape::Line(LineShape {
                x1,
                y1,
                x2,
                y2,
                stroke: "#f97316".to_string(),
                stroke_width: Some(2.0),
                opacity: Some(0.6),
                ..Default::default()
            }));
        }
        // Visited vertices.
        for x in history {
            let (px, py) = project(x[0], x[1]);
            shapes.push(Shape::Circle(CircleShape {
                x: px,
                y: py,
                r: 4.0,
                fill: "#fb923c".to_string(),
                opacity: Some(0.7),
                ..Default::default()
            }));
        }
        // Current x*.
        let (cx, cy) = project(snap.x[0], snap.x[1]);
        shapes.push(Shape::Circle(CircleShape {
            x: cx,
            y: cy,
            r: 9.0,
            fill: if snap.is_optimal {
                "#22c55e".to_string()
            } else {
                "#fbbf24".to_string()
            },
            stroke: Some("#0b1220".to_string()),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: cx + 14.0,
            y: cy + 4.0,
            text: format!(
                "x* = ({}, {})  z = {}",
                to_fixed(snap.x[0], 2),
                to_fixed(snap.x[1], 2),
                to_fixed(snap.z, 2)
            ),
            font_size: Some(11.0),
            fill: Some("#f1f5f9".to_string()),
            ..Default::default()
        }));
    } else {
        // 3+ structural variables: just print a notice.
        shapes.push(Shape::Text(TextShape {
            x: POLY_X + POLY_W / 2.0,
            y: POLY_Y + POLY_H / 2.0,
            text: format!(
                "{} structural variables — polytope view limited to 2D",
                snap.num_struct
            ),
            font_size: Some(16.0),
            fill: Some("#94a3b8".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: POLY_X + POLY_W / 2.0,
            y: POLY_Y + POLY_H / 2.0 + 24.0,
            text: "See tableau panel for the full state".to_string(),
            font_size: Some(12.0),
            fill: Some("#64748b".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    // Event flash.
    let event_truthy = matches!(&args.event_label, Some(s) if !s.is_empty());
    if event_truthy && args.event_flash.unwrap_or(0.0) > 0.0 {
        let alpha = args.event_flash.unwrap_or(0.0).clamp(0.0, 1.0);
        shapes.push(Shape::Rect(RectShape {
            x: POLY_X + 30.0,
            y: POLY_Y + 50.0,
            w: POLY_W - 60.0,
            h: 36.0,
            fill: "#dc2626".to_string(),
            opacity: Some(0.55 * alpha),
            stroke: Some("#fca5a5".to_string()),
            stroke_width: Some(1.0),
            rx: Some(4.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: POLY_X + POLY_W / 2.0,
            y: POLY_Y + 73.0,
            text: format!("EVENT: {}", args.event_label.clone().unwrap_or_default()),
            font_size: Some(14.0),
            fill: Some("#fff".to_string()),
            font_weight: Some(FontWeight::Bold),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    // ---------------- Right panel: tableau + status ----------------
    shapes.push(Shape::Rect(RectShape {
        x: TAB_X,
        y: TAB_Y,
        w: TAB_W,
        h: TAB_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + TAB_W / 2.0,
        y: TAB_Y + 26.0,
        text: "Simplex tableau".to_string(),
        font_size: Some(14.0),
        fill: Some("#f1f5f9".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    // Status line.
    let mode_color = match snap.mode.as_str() {
        "optimal" => "#22c55e",
        "primal" => "#fb923c",
        "dual" => "#a78bfa",
        "unbounded" => "#ef4444",
        "infeasible" => "#ef4444",
        _ => "#94a3b8",
    };
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: TAB_Y + 56.0,
        text: format!("mode: {}", snap.mode.to_uppercase()),
        font_size: Some(12.0),
        fill: Some(mode_color.to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: TAB_Y + 76.0,
        text: format!(
            "tick {}  \u{2022}  z = {}",
            js_num(snap.tick),
            to_fixed(snap.z, 3)
        ),
        font_size: Some(12.0),
        fill: Some("#cbd5e1".to_string()),
        ..Default::default()
    }));
    let basis_str: Vec<String> = snap
        .basis
        .iter()
        .map(|&bcol| basis_name(snap, bcol))
        .collect();
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: TAB_Y + 94.0,
        text: format!(
            "n={}  m={}  basis=[{}]",
            snap.num_struct,
            snap.num_constraints,
            basis_str.join(", ")
        ),
        font_size: Some(11.0),
        fill: Some("#94a3b8".to_string()),
        ..Default::default()
    }));

    if matches!(&args.pivot_label, Some(s) if !s.is_empty()) {
        shapes.push(Shape::Text(TextShape {
            x: TAB_X + 14.0,
            y: TAB_Y + 112.0,
            text: format!("pivot: {}", args.pivot_label.clone().unwrap_or_default()),
            font_size: Some(11.0),
            fill: Some("#7dd3fc".to_string()),
            ..Default::default()
        }));
    }

    // Header row.
    let mut y_row = TAB_Y + 142.0;
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: y_row,
        text: "basic".to_string(),
        font_size: Some(10.0),
        fill: Some("#64748b".to_string()),
        ..Default::default()
    }));
    let mut col_names: Vec<String> = Vec::new();
    for j in 0..snap.num_struct {
        col_names.push(var_name_or(snap, j, format!("x{}", j + 1)));
    }
    for j in 0..snap.num_constraints {
        col_names.push(format!("{}_s", con_name_or(snap, j, format!("c{}", j + 1))));
    }
    col_names.push("rhs".to_string());
    let col_w = (TAB_W - 80.0) / col_names.len() as f64;
    for (j, name) in col_names.iter().enumerate() {
        shapes.push(Shape::Text(TextShape {
            x: TAB_X + 80.0 + j as f64 * col_w + col_w / 2.0,
            y: y_row,
            text: name.clone(),
            font_size: Some(9.0),
            fill: Some("#94a3b8".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
    y_row += 14.0;
    // Z-row.
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: y_row,
        text: "z".to_string(),
        font_size: Some(10.0),
        fill: Some("#fbbf24".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for j in 0..snap.num_struct + snap.num_constraints {
        let val = snap.reduced_costs.get(j).copied().unwrap_or(0.0);
        let color = if val < -1e-7 {
            "#fb923c"
        } else if val.abs() < 1e-9 {
            "#475569"
        } else {
            "#cbd5e1"
        };
        shapes.push(Shape::Text(TextShape {
            x: TAB_X + 80.0 + j as f64 * col_w + col_w / 2.0,
            y: y_row,
            text: to_fixed(val, 2),
            font_size: Some(9.0),
            fill: Some(color.to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 80.0 + (snap.num_struct + snap.num_constraints) as f64 * col_w + col_w / 2.0,
        y: y_row,
        text: to_fixed(snap.z, 2),
        font_size: Some(9.0),
        fill: Some("#fbbf24".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    y_row += 14.0;
    // Constraint rows.
    for i in 0..snap.num_constraints {
        let base_col = snap.basis[i];
        let base_name = if base_col < snap.num_struct {
            var_name_or(snap, base_col, format!("x{}", base_col + 1))
        } else {
            let idx = base_col - snap.num_struct;
            format!("{}_s", con_name_or(snap, idx, format!("c{}", idx + 1)))
        };
        shapes.push(Shape::Text(TextShape {
            x: TAB_X + 14.0,
            y: y_row,
            text: base_name,
            font_size: Some(10.0),
            fill: Some("#22d3ee".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        let rhs_col = snap.num_struct + snap.num_constraints;
        shapes.push(Shape::Text(TextShape {
            x: TAB_X + 80.0 + rhs_col as f64 * col_w + col_w / 2.0,
            y: y_row,
            text: to_fixed(snap.rhs[i], 2),
            font_size: Some(9.0),
            fill: Some(if snap.rhs[i] < -1e-7 {
                "#ef4444".to_string()
            } else {
                "#cbd5e1".to_string()
            }),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
        y_row += 13.0;
        if y_row > TAB_Y + TAB_H - 80.0 {
            break;
        }
    }

    // Footer: feasibility flags.
    let foot_y = TAB_Y + TAB_H - 60.0;
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: foot_y,
        text: format!(
            "primal-feasible:  {}",
            if snap.primal_feasible { "YES" } else { "NO" }
        ),
        font_size: Some(11.0),
        fill: Some(if snap.primal_feasible {
            "#22c55e".to_string()
        } else {
            "#ef4444".to_string()
        }),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: foot_y + 18.0,
        text: format!(
            "dual-feasible:    {}",
            if snap.dual_feasible { "YES" } else { "NO" }
        ),
        font_size: Some(11.0),
        fill: Some(if snap.dual_feasible {
            "#22c55e".to_string()
        } else {
            "#ef4444".to_string()
        }),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: TAB_X + 14.0,
        y: foot_y + 36.0,
        text: format!(
            "optimal:          {}",
            if snap.is_optimal { "YES" } else { "NO" }
        ),
        font_size: Some(11.0),
        fill: Some(if snap.is_optimal {
            "#22c55e".to_string()
        } else {
            "#94a3b8".to_string()
        }),
        ..Default::default()
    }));

    // Caption.
    let mut caption = format!(
        "tick {}  \u{2022}  z = {}  \u{2022}  mode = {}",
        js_num(snap.tick),
        to_fixed(snap.z, 3),
        snap.mode
    );
    if event_truthy {
        caption += &format!(
            "  \u{2022}  event: {}",
            args.event_label.clone().unwrap_or_default()
        );
    }
    if matches!(&args.pivot_label, Some(s) if !s.is_empty()) {
        caption += &format!(
            "  \u{2022}  pivot: {}",
            args.pivot_label.clone().unwrap_or_default()
        );
    }

    FrameParts::with_caption(shapes, caption)
}

/// Telemetry charts to plot underneath the main panels.
pub fn build_incremental_lp_charts(
    ticks: &[f64],
    z_values: &[f64],
    x_series: &[Vec<f64>],
) -> Vec<ChartSpec> {
    let series = vec![ChartSeries {
        label: "z".to_string(),
        color: "#fbbf24".to_string(),
        t: ticks.to_vec(),
        y: z_values.to_vec(),
    }];
    let x_colors = [
        "#22d3ee", "#a78bfa", "#fb923c", "#34d399", "#f472b6", "#facc15",
    ];
    let mut x_charts: Vec<ChartSpec> = Vec::new();
    if !x_series.is_empty() {
        // `xs = xSeries[0].map((_, j) => xSeries.map(x => x[j] ?? 0))` — transpose.
        let cols = x_series[0].len();
        let x_series_out: Vec<ChartSeries> = (0..cols)
            .map(|j| ChartSeries {
                label: format!("x{}", j + 1),
                color: x_colors[j % x_colors.len()].to_string(),
                t: ticks.to_vec(),
                y: x_series
                    .iter()
                    .map(|x| x.get(j).copied().unwrap_or(0.0))
                    .collect(),
            })
            .collect();
        x_charts.push(ChartSpec {
            x: 30.0,
            y: 660.0,
            w: 600.0,
            h: 30.0,
            title: Some("x* (per structural variable)".to_string()),
            series: x_series_out,
            ..Default::default()
        });
    }
    let mut out = vec![ChartSpec {
        x: 660.0,
        y: 660.0,
        w: 490.0,
        h: 30.0,
        title: Some("objective z over time".to_string()),
        series,
        ..Default::default()
    }];
    out.extend(x_charts);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polytope_vertices_for_a_triangle() {
        // x1 + x2 <= 1, with x >= 0 -> triangle (0,0),(1,0),(0,1).
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        let verts = compute_polytope_vertices(&a, &b);
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn frame_caption_includes_mode_and_event() {
        let snap = LPSnapshot {
            num_struct: 2,
            num_constraints: 1,
            x: vec![0.5, 0.5],
            z: 1.0,
            tick: 4.0,
            mode: "primal".to_string(),
            is_optimal: false,
            primal_feasible: true,
            dual_feasible: false,
            basis: vec![0],
            var_names: vec!["x1".to_string(), "x2".to_string()],
            con_names: vec!["c1".to_string()],
            reduced_costs: vec![-1.0, 0.0, 0.0],
            rhs: vec![1.0],
        };
        let args = IncrementalLPFrameArgs {
            snap: &snap,
            a: &[vec![1.0, 1.0]],
            b: &[1.0],
            c: &[1.0, 1.0],
            sense: Sense::Max,
            history: &[vec![0.0, 0.0], vec![0.5, 0.5]],
            event_label: Some("add constraint".to_string()),
            event_flash: Some(0.8),
            pivot_label: None,
        };
        let fp = build_incremental_lp_frame(0.0, 0.0, &args);
        let cap = fp.caption.unwrap();
        assert!(cap.starts_with("tick 4  \u{2022}  z = 1.000  \u{2022}  mode = primal"));
        assert!(cap.contains("event: add constraint"));
    }
}
