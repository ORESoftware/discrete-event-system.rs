//! Animation scene builders for analytic calculus-of-variations models.
//!
//! Each frame progressively reveals the stationary curve, keeps the full
//! solution ghosted underneath, and shows the corresponding functional,
//! Euler-Lagrange model, first integral, topology, and diagnostics.

#![allow(dead_code)]

use crate::des::animation::types::{
    to_exponential, to_fixed, Anchor, Animation, ChartSeries, ChartSpec, CircleShape, FontWeight,
    Frame, FrameParts, LineShape, PathShape, RectShape, Shape, TextShape,
};
use crate::des::general::calculus_of_variations::{
    SolutionSample, VariationalProblemKind, VariationalSolutionModel,
};

pub const COV_STAGE_W: f64 = 1040.0;
pub const COV_STAGE_H: f64 = 740.0;

const PLOT_X: f64 = 64.0;
const PLOT_Y: f64 = 82.0;
const PLOT_W: f64 = 650.0;
const PLOT_H: f64 = 420.0;
const PANEL_X: f64 = 744.0;
const PANEL_Y: f64 = 82.0;
const PANEL_W: f64 = 232.0;
const PANEL_H: f64 = 420.0;
const CHART_Y: f64 = 552.0;
const CHART_H: f64 = 140.0;
const MAX_FRAMES: usize = 96;
const FIT_ITERS: usize = 72;
const FIT_GRID: usize = 96;
const FIT_BASIS: usize = 6;

#[derive(Clone, Copy)]
struct Bounds {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

impl Bounds {
    fn from_samples(samples: &[SolutionSample]) -> Self {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for s in samples {
            x_min = x_min.min(s.x);
            x_max = x_max.max(s.x);
            y_min = y_min.min(s.y);
            y_max = y_max.max(s.y);
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            x_min = 0.0;
            x_max = 1.0;
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = 0.0;
            y_max = 1.0;
        }
        let x_pad = ((x_max - x_min).abs() * 0.08).max(1e-6);
        let y_pad = ((y_max - y_min).abs() * 0.12).max(1e-6);
        Bounds {
            x_min: x_min - x_pad,
            x_max: x_max + x_pad,
            y_min: y_min - y_pad,
            y_max: y_max + y_pad,
        }
    }

    fn sx(&self, x: f64) -> f64 {
        PLOT_X + (x - self.x_min) / (self.x_max - self.x_min) * PLOT_W
    }

    fn sy(&self, y: f64) -> f64 {
        PLOT_Y + PLOT_H - (y - self.y_min) / (self.y_max - self.y_min) * PLOT_H
    }
}

/// Build a complete animation document for one variational model.
pub fn build_variational_animation(model: &VariationalSolutionModel) -> Animation {
    if model.problem.kind == VariationalProblemKind::Brachistochrone {
        return Animation {
            width: COV_STAGE_W,
            height: COV_STAGE_H,
            fps: 10.0,
            title: Some("Brachistochrone - numerical curve fit".to_string()),
            subtitle: Some(
                "Gradient descent fits a boundary-respecting sine-basis curve to the cycloid solution."
                    .to_string(),
            ),
            frames: build_variational_frames(model),
            charts: Some(build_variational_charts(model)),
            background: Some("#f8fafc".to_string()),
        };
    }

    Animation {
        width: COV_STAGE_W,
        height: COV_STAGE_H,
        fps: 10.0,
        title: Some(format!("Calculus of variations - {}", model.problem.name)),
        subtitle: Some(model.problem.description.clone()),
        frames: build_variational_frames(model),
        charts: Some(build_variational_charts(model)),
        background: Some("#f8fafc".to_string()),
    }
}

/// Build all progressive-reveal frames for one model.
pub fn build_variational_frames(model: &VariationalSolutionModel) -> Vec<Frame> {
    if model.problem.kind == VariationalProblemKind::Brachistochrone {
        return build_brachistochrone_fit_frames(model);
    }

    let n = model.samples.len();
    if n == 0 {
        return vec![empty_frame(model)];
    }
    let frame_count = n.min(MAX_FRAMES).max(1);
    (0..frame_count)
        .map(|i| {
            let idx = if frame_count <= 1 {
                0
            } else {
                i * (n - 1) / (frame_count - 1)
            };
            build_variational_frame(model, idx + 1).into_frame(i as f64, i as f64)
        })
        .collect()
}

/// Build the chart strip for the HTML player.
pub fn build_variational_charts(model: &VariationalSolutionModel) -> Vec<ChartSpec> {
    if model.problem.kind == VariationalProblemKind::Brachistochrone {
        return build_brachistochrone_fit_charts(model);
    }

    let xs = model.samples.iter().map(|s| s.x).collect::<Vec<_>>();
    let ys = model.samples.iter().map(|s| s.y).collect::<Vec<_>>();
    let cumulative = cumulative_functional(&model.samples);
    vec![
        ChartSpec {
            x: 64.0,
            y: CHART_Y,
            w: 420.0,
            h: CHART_H,
            title: Some("stationary curve y(x)".to_string()),
            y_label: Some("y".to_string()),
            series: vec![ChartSeries {
                label: "y(x)".to_string(),
                color: accent(model).to_string(),
                t: xs.clone(),
                y: ys,
            }],
            cursor: Some(false),
            ..Default::default()
        },
        ChartSpec {
            x: 532.0,
            y: CHART_Y,
            w: 444.0,
            h: CHART_H,
            title: Some("cumulative functional value".to_string()),
            y_label: Some("J".to_string()),
            series: vec![ChartSeries {
                label: "J up to x".to_string(),
                color: "#0f766e".to_string(),
                t: xs,
                y: cumulative,
            }],
            cursor: Some(false),
            ..Default::default()
        },
    ]
}

#[derive(Clone, Debug)]
struct CurveFitIteration {
    iteration: usize,
    samples: Vec<SolutionSample>,
    rmse: f64,
    max_error: f64,
    time_value: f64,
}

fn build_brachistochrone_fit_frames(model: &VariationalSolutionModel) -> Vec<Frame> {
    let trace = brachistochrone_fit_trace(model);
    trace
        .iter()
        .enumerate()
        .map(|(i, step)| {
            build_brachistochrone_fit_frame(model, &trace, step).into_frame(i as f64, i as f64)
        })
        .collect()
}

fn build_brachistochrone_fit_charts(model: &VariationalSolutionModel) -> Vec<ChartSpec> {
    let trace = brachistochrone_fit_trace(model);
    let iter = trace
        .iter()
        .map(|row| row.iteration as f64)
        .collect::<Vec<_>>();
    let rmse = trace.iter().map(|row| row.rmse).collect::<Vec<_>>();
    let time = trace.iter().map(|row| row.time_value).collect::<Vec<_>>();
    let target_time = vec![model.diagnostics.functional_value; trace.len()];

    vec![
        ChartSpec {
            x: 64.0,
            y: CHART_Y,
            w: 420.0,
            h: CHART_H,
            title: Some("curve-fit residual".to_string()),
            y_label: Some("RMSE".to_string()),
            series: vec![ChartSeries {
                label: "candidate vs cycloid".to_string(),
                color: "#f97316".to_string(),
                t: iter.clone(),
                y: rmse,
            }],
            ..Default::default()
        },
        ChartSpec {
            x: 532.0,
            y: CHART_Y,
            w: 444.0,
            h: CHART_H,
            title: Some("descent-time objective estimate".to_string()),
            y_label: Some("time".to_string()),
            series: vec![
                ChartSeries {
                    label: "candidate".to_string(),
                    color: "#0f766e".to_string(),
                    t: iter.clone(),
                    y: time,
                },
                ChartSeries {
                    label: "cycloid reference".to_string(),
                    color: "#94a3b8".to_string(),
                    t: iter,
                    y: target_time,
                },
            ],
            ..Default::default()
        },
    ]
}

fn brachistochrone_fit_trace(model: &VariationalSolutionModel) -> Vec<CurveFitIteration> {
    let start = model.problem.functional.boundary.0;
    let end = model.problem.functional.boundary.1;
    let span = end.x - start.x;
    if model.samples.len() < 2 || span.abs() <= 1e-12 {
        return Vec::new();
    }

    let grid = (0..FIT_GRID)
        .map(|i| {
            let u = i as f64 / (FIT_GRID - 1) as f64;
            let x = start.x + span * u;
            (u, x, target_y_at(model, x))
        })
        .collect::<Vec<_>>();

    let mut coeffs = vec![0.0; FIT_BASIS];
    let mut trace = Vec::with_capacity(FIT_ITERS + 1);
    let learning_rate = 0.68;
    for iteration in 0..=FIT_ITERS {
        let samples = candidate_samples(&grid, start.y, end.y, span, &coeffs);
        let (rmse, max_error) = fit_error(&samples, &grid);
        let time_value = cumulative_functional(&samples)
            .last()
            .copied()
            .unwrap_or(0.0);
        trace.push(CurveFitIteration {
            iteration,
            samples,
            rmse,
            max_error,
            time_value,
        });

        let mut gradient = vec![0.0; coeffs.len()];
        for (u, _x, target_y) in &grid {
            let predicted = candidate_y(*u, start.y, end.y, &coeffs);
            let residual = predicted - target_y;
            for (j, g) in gradient.iter_mut().enumerate() {
                *g += 2.0 * residual * basis(*u, j) / grid.len() as f64;
            }
        }
        for (c, g) in coeffs.iter_mut().zip(gradient.iter()) {
            *c -= learning_rate * g;
        }
    }
    trace
}

fn build_brachistochrone_fit_frame(
    model: &VariationalSolutionModel,
    trace: &[CurveFitIteration],
    step: &CurveFitIteration,
) -> FrameParts {
    let bounds = Bounds::from_samples(&model.samples);
    let mut shapes = Vec::new();

    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: COV_STAGE_W,
        h: COV_STAGE_H,
        fill: "#f8fafc".to_string(),
        ..Default::default()
    }));
    text(
        &mut shapes,
        40.0,
        38.0,
        "Brachistochrone curve fitting",
        22.0,
        "#0f172a",
        true,
        Anchor::Start,
    );
    text(
        &mut shapes,
        40.0,
        60.0,
        "Each frame is a numerical solver iteration fitting a boundary-locked sine curve toward the cycloid.",
        12.0,
        "#475569",
        false,
        Anchor::Start,
    );

    draw_fit_plot(&mut shapes, model, trace, step, &bounds);
    draw_fit_panel(&mut shapes, model, step);

    FrameParts::with_caption(
        shapes,
        format!(
            "fit iteration {}: RMSE {}, max error {}, time estimate {}",
            step.iteration,
            fmt(step.rmse),
            fmt(step.max_error),
            fmt(step.time_value)
        ),
    )
}

fn draw_fit_plot(
    shapes: &mut Vec<Shape>,
    model: &VariationalSolutionModel,
    trace: &[CurveFitIteration],
    step: &CurveFitIteration,
    bounds: &Bounds,
) {
    shapes.push(Shape::Rect(RectShape {
        x: PLOT_X,
        y: PLOT_Y,
        w: PLOT_W,
        h: PLOT_H,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    text(
        shapes,
        PLOT_X,
        PLOT_Y - 16.0,
        "Candidate curve converging to the cycloid",
        13.0,
        "#334155",
        true,
        Anchor::Start,
    );
    draw_axes(shapes, bounds);

    let target_path = path_for(&model.samples, model.samples.len(), bounds);
    shapes.push(Shape::Path(PathShape {
        d: target_path,
        stroke: Some("#94a3b8".to_string()),
        stroke_width: Some(3.0),
        fill: Some("none".to_string()),
        opacity: Some(0.9),
        ..Default::default()
    }));

    for old in trace
        .iter()
        .take(step.iteration)
        .step_by(12)
        .filter(|old| old.iteration + 1 < step.iteration)
    {
        shapes.push(Shape::Path(PathShape {
            d: path_for(&old.samples, old.samples.len(), bounds),
            stroke: Some("#fed7aa".to_string()),
            stroke_width: Some(1.2),
            fill: Some("none".to_string()),
            opacity: Some(0.45),
            ..Default::default()
        }));
    }

    for i in (4..step.samples.len().saturating_sub(4)).step_by(12) {
        let candidate = &step.samples[i];
        let target_y = target_y_at(model, candidate.x);
        line(
            shapes,
            bounds.sx(candidate.x),
            bounds.sy(candidate.y),
            bounds.sx(candidate.x),
            bounds.sy(target_y),
            "#ef4444",
            0.9,
        );
    }

    shapes.push(Shape::Path(PathShape {
        d: path_for(&step.samples, step.samples.len(), bounds),
        stroke: Some("#f97316".to_string()),
        stroke_width: Some(4.0),
        fill: Some("none".to_string()),
        ..Default::default()
    }));

    if let Some(start) = model.samples.first() {
        point(
            shapes,
            bounds.sx(start.x),
            bounds.sy(start.y),
            6.0,
            "#ffffff",
            "#f97316",
            "fixed start",
        );
    }
    if let Some(end) = model.samples.last() {
        point(
            shapes,
            bounds.sx(end.x),
            bounds.sy(end.y),
            6.0,
            "#ffffff",
            "#0f172a",
            "fixed endpoint",
        );
    }

    text(
        shapes,
        PLOT_X + 16.0,
        PLOT_Y + 22.0,
        "gray: cycloid target",
        10.5,
        "#64748b",
        false,
        Anchor::Start,
    );
    text(
        shapes,
        PLOT_X + 16.0,
        PLOT_Y + 40.0,
        "orange: current fitted curve",
        10.5,
        "#9a3412",
        true,
        Anchor::Start,
    );
    text(
        shapes,
        PLOT_X + 16.0,
        PLOT_Y + 58.0,
        "red ticks: residual to target",
        10.5,
        "#b91c1c",
        false,
        Anchor::Start,
    );
}

fn draw_fit_panel(
    shapes: &mut Vec<Shape>,
    model: &VariationalSolutionModel,
    step: &CurveFitIteration,
) {
    shapes.push(Shape::Rect(RectShape {
        x: PANEL_X,
        y: PANEL_Y,
        w: PANEL_W,
        h: PANEL_H,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    text(
        shapes,
        PANEL_X + 14.0,
        PANEL_Y + 26.0,
        "Numerical solver",
        14.0,
        "#0f172a",
        true,
        Anchor::Start,
    );

    let mut y = PANEL_Y + 56.0;
    for line in wrap("curve: line + sum a_k sin(k*pi*x/L)", 32) {
        text(
            shapes,
            PANEL_X + 14.0,
            y,
            &line,
            10.5,
            "#475569",
            false,
            Anchor::Start,
        );
        y += 15.0;
    }
    y += 6.0;
    for line in wrap("step: finite-grid least-squares gradient descent", 32) {
        text(
            shapes,
            PANEL_X + 14.0,
            y,
            &line,
            10.5,
            "#475569",
            false,
            Anchor::Start,
        );
        y += 15.0;
    }

    y = PANEL_Y + 162.0;
    metric(
        shapes,
        y,
        "iteration",
        &format!("{}/{}", step.iteration, FIT_ITERS),
    );
    metric(shapes, y + 24.0, "RMSE", &fmt(step.rmse));
    metric(shapes, y + 48.0, "max |error|", &fmt(step.max_error));
    metric(shapes, y + 72.0, "time estimate", &fmt(step.time_value));
    metric(
        shapes,
        y + 96.0,
        "cycloid time",
        &fmt(model.diagnostics.functional_value),
    );

    let bar_x = PANEL_X + 14.0;
    let bar_y = PANEL_Y + 292.0;
    let bar_w = PANEL_W - 28.0;
    shapes.push(Shape::Rect(RectShape {
        x: bar_x,
        y: bar_y,
        w: bar_w,
        h: 12.0,
        fill: "#ffedd5".to_string(),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: bar_x,
        y: bar_y,
        w: bar_w * step.iteration as f64 / FIT_ITERS as f64,
        h: 12.0,
        fill: "#f97316".to_string(),
        rx: Some(4.0),
        ..Default::default()
    }));

    draw_fit_topology(shapes, PANEL_Y + PANEL_H - 76.0);
}

fn draw_axes(shapes: &mut Vec<Shape>, bounds: &Bounds) {
    let zero_y = if bounds.y_min <= 0.0 && 0.0 <= bounds.y_max {
        bounds.sy(0.0)
    } else {
        PLOT_Y + PLOT_H
    };
    let zero_x = if bounds.x_min <= 0.0 && 0.0 <= bounds.x_max {
        bounds.sx(0.0)
    } else {
        PLOT_X
    };
    line(
        shapes,
        PLOT_X,
        zero_y,
        PLOT_X + PLOT_W,
        zero_y,
        "#e2e8f0",
        1.0,
    );
    line(
        shapes,
        zero_x,
        PLOT_Y,
        zero_x,
        PLOT_Y + PLOT_H,
        "#e2e8f0",
        1.0,
    );
    for k in 0..=4 {
        let x = PLOT_X + PLOT_W * k as f64 / 4.0;
        line(shapes, x, PLOT_Y, x, PLOT_Y + PLOT_H, "#f1f5f9", 0.8);
        let y = PLOT_Y + PLOT_H * k as f64 / 4.0;
        line(shapes, PLOT_X, y, PLOT_X + PLOT_W, y, "#f1f5f9", 0.8);
    }
}

fn draw_fit_topology(shapes: &mut Vec<Shape>, y: f64) {
    let labels = ["curve", "residual", "gradient", "update"];
    let w = 48.0;
    let gap = 7.0;
    for (i, label) in labels.iter().enumerate() {
        let x = PANEL_X + 14.0 + i as f64 * (w + gap);
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w,
            h: 24.0,
            fill: "#f8fafc".to_string(),
            stroke: Some("#cbd5e1".to_string()),
            stroke_width: Some(1.0),
            rx: Some(4.0),
            ..Default::default()
        }));
        text(
            shapes,
            x + w / 2.0,
            y + 16.0,
            label,
            8.0,
            "#334155",
            false,
            Anchor::Middle,
        );
        if i < labels.len() - 1 {
            line(
                shapes,
                x + w + 1.0,
                y + 12.0,
                x + w + gap - 1.0,
                y + 12.0,
                "#94a3b8",
                1.0,
            );
        }
    }
}

fn target_y_at(model: &VariationalSolutionModel, x: f64) -> f64 {
    let samples = &model.samples;
    if samples.is_empty() {
        return 0.0;
    }
    if x <= samples[0].x {
        return samples[0].y;
    }
    for pair in samples.windows(2) {
        if x <= pair[1].x {
            let dx = pair[1].x - pair[0].x;
            let u = if dx.abs() <= 1e-12 {
                0.0
            } else {
                (x - pair[0].x) / dx
            };
            return pair[0].y + u * (pair[1].y - pair[0].y);
        }
    }
    samples.last().map(|s| s.y).unwrap_or(0.0)
}

fn candidate_samples(
    grid: &[(f64, f64, f64)],
    y0: f64,
    y1: f64,
    span: f64,
    coeffs: &[f64],
) -> Vec<SolutionSample> {
    grid.iter()
        .map(|(u, x, _target_y)| {
            let y = candidate_y(*u, y0, y1, coeffs);
            let slope = candidate_slope(*u, y0, y1, span, coeffs);
            let depth = -y;
            let integrand = if depth > 1e-9 {
                Some(((1.0 + slope * slope) / (2.0 * 9.81 * depth)).sqrt())
            } else {
                None
            };
            SolutionSample {
                parameter: Some(*u),
                x: *x,
                y,
                dy_dx: Some(slope),
                integrand,
            }
        })
        .collect()
}

fn candidate_y(u: f64, y0: f64, y1: f64, coeffs: &[f64]) -> f64 {
    let mut y = y0 + u * (y1 - y0);
    for (j, c) in coeffs.iter().enumerate() {
        y += c * basis(u, j);
    }
    y
}

fn candidate_slope(u: f64, y0: f64, y1: f64, span: f64, coeffs: &[f64]) -> f64 {
    let mut dy_du = y1 - y0;
    for (j, c) in coeffs.iter().enumerate() {
        let k = (j + 1) as f64;
        dy_du += c * k * std::f64::consts::PI * (k * std::f64::consts::PI * u).cos();
    }
    dy_du / span
}

fn basis(u: f64, j: usize) -> f64 {
    let k = (j + 1) as f64;
    (k * std::f64::consts::PI * u).sin()
}

fn fit_error(samples: &[SolutionSample], grid: &[(f64, f64, f64)]) -> (f64, f64) {
    let mut sum_sq = 0.0;
    let mut max_error = 0.0_f64;
    for (sample, (_u, _x, target_y)) in samples.iter().zip(grid.iter()) {
        let err = sample.y - target_y;
        sum_sq += err * err;
        max_error = max_error.max(err.abs());
    }
    ((sum_sq / samples.len().max(1) as f64).sqrt(), max_error)
}

fn cumulative_functional(samples: &[SolutionSample]) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(0.0);
    let mut total = 0.0;
    for pair in samples.windows(2) {
        let dx = (pair[1].x - pair[0].x).abs();
        let left = pair[0].integrand.filter(|v| v.is_finite());
        let right = pair[1].integrand.filter(|v| v.is_finite());
        let area = match (left, right) {
            (Some(a), Some(b)) => 0.5 * (a + b) * dx,
            (Some(a), None) | (None, Some(a)) => 0.5 * a * dx,
            (None, None) => 0.0,
        };
        total += area;
        out.push(total);
    }
    out
}

fn empty_frame(model: &VariationalSolutionModel) -> Frame {
    FrameParts::with_caption(
        vec![Shape::Text(TextShape {
            x: 40.0,
            y: 60.0,
            text: format!("{} has no solution samples", model.problem.name),
            font_size: Some(18.0),
            fill: Some("#334155".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        })],
        "No solution samples were supplied.",
    )
    .into_frame(0.0, 0.0)
}

fn build_variational_frame(model: &VariationalSolutionModel, reveal: usize) -> FrameParts {
    let bounds = Bounds::from_samples(&model.samples);
    let accent = accent(model);
    let reveal = reveal.clamp(1, model.samples.len());
    let current = &model.samples[reveal - 1];
    let mut shapes = Vec::new();

    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: COV_STAGE_W,
        h: COV_STAGE_H,
        fill: "#f8fafc".to_string(),
        ..Default::default()
    }));
    text(
        &mut shapes,
        40.0,
        38.0,
        &model.problem.name,
        22.0,
        "#0f172a",
        true,
        Anchor::Start,
    );
    text(
        &mut shapes,
        40.0,
        60.0,
        &shorten(&model.problem.description, 92),
        12.0,
        "#475569",
        false,
        Anchor::Start,
    );

    draw_plot(&mut shapes, model, reveal, current, &bounds, accent);
    draw_panel(&mut shapes, model, reveal, current);

    FrameParts::with_caption(
        shapes,
        format!(
            "{}: revealed {}/{} samples; functional value {}",
            model.problem.id,
            reveal,
            model.samples.len(),
            fmt(model.diagnostics.functional_value)
        ),
    )
}

fn draw_plot(
    shapes: &mut Vec<Shape>,
    model: &VariationalSolutionModel,
    reveal: usize,
    current: &SolutionSample,
    bounds: &Bounds,
    accent: &str,
) {
    shapes.push(Shape::Rect(RectShape {
        x: PLOT_X,
        y: PLOT_Y,
        w: PLOT_W,
        h: PLOT_H,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    text(
        shapes,
        PLOT_X,
        PLOT_Y - 16.0,
        "Stationary solution path",
        13.0,
        "#334155",
        true,
        Anchor::Start,
    );

    let zero_y = if bounds.y_min <= 0.0 && 0.0 <= bounds.y_max {
        bounds.sy(0.0)
    } else {
        PLOT_Y + PLOT_H
    };
    let zero_x = if bounds.x_min <= 0.0 && 0.0 <= bounds.x_max {
        bounds.sx(0.0)
    } else {
        PLOT_X
    };
    line(
        shapes,
        PLOT_X,
        zero_y,
        PLOT_X + PLOT_W,
        zero_y,
        "#e2e8f0",
        1.0,
    );
    line(
        shapes,
        zero_x,
        PLOT_Y,
        zero_x,
        PLOT_Y + PLOT_H,
        "#e2e8f0",
        1.0,
    );

    for k in 0..=4 {
        let x = PLOT_X + PLOT_W * k as f64 / 4.0;
        line(shapes, x, PLOT_Y, x, PLOT_Y + PLOT_H, "#f1f5f9", 0.8);
        let y = PLOT_Y + PLOT_H * k as f64 / 4.0;
        line(shapes, PLOT_X, y, PLOT_X + PLOT_W, y, "#f1f5f9", 0.8);
    }

    let full = path_for(&model.samples, model.samples.len(), bounds);
    shapes.push(Shape::Path(PathShape {
        d: full,
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(2.5),
        fill: Some("none".to_string()),
        opacity: Some(0.95),
        ..Default::default()
    }));
    let shown = path_for(&model.samples, reveal, bounds);
    shapes.push(Shape::Path(PathShape {
        d: shown,
        stroke: Some(accent.to_string()),
        stroke_width: Some(4.0),
        fill: Some("none".to_string()),
        ..Default::default()
    }));

    if let Some(start) = model.samples.first() {
        point(
            shapes,
            bounds.sx(start.x),
            bounds.sy(start.y),
            6.0,
            "#ffffff",
            accent,
            "start",
        );
    }
    if let Some(end) = model.samples.last() {
        point(
            shapes,
            bounds.sx(end.x),
            bounds.sy(end.y),
            6.0,
            "#ffffff",
            "#0f172a",
            "end",
        );
    }
    point(
        shapes,
        bounds.sx(current.x),
        bounds.sy(current.y),
        7.0,
        accent,
        "#ffffff",
        "current sample",
    );

    text(
        shapes,
        PLOT_X,
        PLOT_Y + PLOT_H + 22.0,
        &format!("x in [{} , {}]", fmt(bounds.x_min), fmt(bounds.x_max)),
        11.0,
        "#64748b",
        false,
        Anchor::Start,
    );
    text(
        shapes,
        PLOT_X + PLOT_W,
        PLOT_Y + PLOT_H + 22.0,
        &format!("current: x={}, y={}", fmt(current.x), fmt(current.y)),
        11.0,
        "#334155",
        true,
        Anchor::End,
    );
}

fn draw_panel(
    shapes: &mut Vec<Shape>,
    model: &VariationalSolutionModel,
    reveal: usize,
    current: &SolutionSample,
) {
    shapes.push(Shape::Rect(RectShape {
        x: PANEL_X,
        y: PANEL_Y,
        w: PANEL_W,
        h: PANEL_H,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));

    text(
        shapes,
        PANEL_X + 14.0,
        PANEL_Y + 26.0,
        "Variational model",
        14.0,
        "#0f172a",
        true,
        Anchor::Start,
    );
    let mut y = PANEL_Y + 54.0;
    for line in wrap(
        &format!("J[y] = integral {}", model.problem.functional.integrand),
        33,
    ) {
        text(
            shapes,
            PANEL_X + 14.0,
            y,
            &line,
            10.5,
            "#475569",
            false,
            Anchor::Start,
        );
        y += 15.0;
    }
    y += 4.0;
    for line in wrap(
        &format!("EL: {}", model.problem.euler_lagrange.equation),
        33,
    ) {
        text(
            shapes,
            PANEL_X + 14.0,
            y,
            &line,
            10.5,
            "#475569",
            false,
            Anchor::Start,
        );
        y += 15.0;
    }
    if let Some(first) = &model.problem.euler_lagrange.first_integral {
        y += 4.0;
        for line in wrap(&format!("First integral: {first}"), 33) {
            text(
                shapes,
                PANEL_X + 14.0,
                y,
                &line,
                10.5,
                "#475569",
                false,
                Anchor::Start,
            );
            y += 15.0;
        }
    }

    y = PANEL_Y + 260.0;
    metric(
        shapes,
        y,
        "J value",
        &fmt(model.diagnostics.functional_value),
    );
    metric(
        shapes,
        y + 24.0,
        "boundary err",
        &fmt(model.diagnostics.boundary_error),
    );
    metric(
        shapes,
        y + 48.0,
        "first-integral L2",
        &fmt(model.diagnostics.first_integral_residual_l2),
    );
    metric(
        shapes,
        y + 72.0,
        "dy/dx",
        &current
            .dy_dx
            .map(fmt)
            .unwrap_or_else(|| "singular".to_string()),
    );

    let top = PANEL_Y + PANEL_H - 54.0;
    draw_topology(shapes, top);
    text(
        shapes,
        PANEL_X + PANEL_W - 14.0,
        PANEL_Y + PANEL_H - 14.0,
        &format!("sample {reveal}/{}", model.samples.len()),
        10.5,
        "#64748b",
        false,
        Anchor::End,
    );
}

fn draw_topology(shapes: &mut Vec<Shape>, y: f64) {
    let labels = ["problem", "EL solver", "sampler"];
    let w = 62.0;
    let gap = 12.0;
    for (i, label) in labels.iter().enumerate() {
        let x = PANEL_X + 14.0 + i as f64 * (w + gap);
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w,
            h: 24.0,
            fill: "#f8fafc".to_string(),
            stroke: Some("#cbd5e1".to_string()),
            stroke_width: Some(1.0),
            rx: Some(4.0),
            ..Default::default()
        }));
        text(
            shapes,
            x + w / 2.0,
            y + 16.0,
            label,
            9.0,
            "#334155",
            false,
            Anchor::Middle,
        );
        if i < labels.len() - 1 {
            line(
                shapes,
                x + w + 2.0,
                y + 12.0,
                x + w + gap - 2.0,
                y + 12.0,
                "#94a3b8",
                1.2,
            );
        }
    }
}

fn path_for(samples: &[SolutionSample], count: usize, bounds: &Bounds) -> String {
    let mut d = String::new();
    for (i, s) in samples.iter().take(count).enumerate() {
        if i == 0 {
            d.push_str(&format!("M {:.3} {:.3}", bounds.sx(s.x), bounds.sy(s.y)));
        } else {
            d.push_str(&format!(" L {:.3} {:.3}", bounds.sx(s.x), bounds.sy(s.y)));
        }
    }
    d
}

fn accent(model: &VariationalSolutionModel) -> &'static str {
    match model.problem.kind {
        VariationalProblemKind::ShortestCurve => "#2563eb",
        VariationalProblemKind::Brachistochrone => "#f97316",
        VariationalProblemKind::MinimalSurfaceOfRevolution => "#059669",
    }
}

fn line(shapes: &mut Vec<Shape>, x1: f64, y1: f64, x2: f64, y2: f64, color: &str, width: f64) {
    shapes.push(Shape::Line(LineShape {
        x1,
        y1,
        x2,
        y2,
        stroke: color.to_string(),
        stroke_width: Some(width),
        ..Default::default()
    }));
}

fn point(shapes: &mut Vec<Shape>, x: f64, y: f64, r: f64, fill: &str, stroke: &str, title: &str) {
    shapes.push(Shape::Circle(CircleShape {
        x,
        y,
        r,
        fill: fill.to_string(),
        stroke: Some(stroke.to_string()),
        stroke_width: Some(2.0),
        title: Some(title.to_string()),
        ..Default::default()
    }));
}

fn text(
    shapes: &mut Vec<Shape>,
    x: f64,
    y: f64,
    value: &str,
    size: f64,
    fill: &str,
    bold: bool,
    anchor: Anchor,
) {
    shapes.push(Shape::Text(TextShape {
        x,
        y,
        text: value.to_string(),
        font_size: Some(size),
        fill: Some(fill.to_string()),
        anchor: Some(anchor),
        font_weight: Some(if bold {
            FontWeight::Bold
        } else {
            FontWeight::Normal
        }),
        ..Default::default()
    }));
}

fn metric(shapes: &mut Vec<Shape>, y: f64, label: &str, value: &str) {
    text(
        shapes,
        PANEL_X + 14.0,
        y,
        label,
        10.5,
        "#64748b",
        false,
        Anchor::Start,
    );
    text(
        shapes,
        PANEL_X + PANEL_W - 14.0,
        y,
        value,
        10.5,
        "#0f172a",
        true,
        Anchor::End,
    );
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > width {
            lines.push(cur);
            cur = String::new();
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

fn fmt(v: f64) -> String {
    if !v.is_finite() {
        return "n/a".to_string();
    }
    let a = v.abs();
    if a != 0.0 && (a >= 1.0e4 || a < 1.0e-3) {
        to_exponential(v, 2)
    } else {
        to_fixed(v, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::calculus_of_variations::built_in_variational_models;

    #[test]
    fn every_variational_model_builds_a_playable_animation() {
        let models = built_in_variational_models();
        assert_eq!(models.len(), 3);
        for model in &models {
            let anim = build_variational_animation(model);
            assert_eq!(anim.width, COV_STAGE_W);
            assert_eq!(anim.height, COV_STAGE_H);
            assert!(
                !anim.frames.is_empty(),
                "{} has no frames",
                model.problem.id
            );
            assert_eq!(anim.charts.as_ref().map(Vec::len), Some(2));
            assert!(
                anim.frames[0].shapes.len() > 12,
                "{} first frame is too sparse",
                model.problem.id
            );
        }
    }

    #[test]
    fn brachistochrone_frames_show_solver_iterations() {
        let model = built_in_variational_models().remove(1);
        let trace = brachistochrone_fit_trace(&model);
        assert!(trace.last().unwrap().rmse < trace.first().unwrap().rmse * 0.2);

        let frames = build_variational_frames(&model);
        assert!(frames.len() <= MAX_FRAMES);
        let last = frames.last().unwrap();
        assert_eq!(last.tick, (frames.len() - 1) as f64);
        assert!(last.caption.as_ref().unwrap().contains("fit iteration"));

        let charts = build_variational_charts(&model);
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].title.as_deref(), Some("curve-fit residual"));
        assert_eq!(
            charts[1].title.as_deref(),
            Some("descent-time objective estimate")
        );
    }
}
