//! Port of `src/des/animation/scenes/neural-network-scene.ts`.
//!
//! Post-hoc animation scenes over already-run neural results:
//! * XOR — network topology + active training sample + loss/prediction charts.
//! * Neural Q-learning — learned greedy policy through a corridor MDP.
//! * Neural ODE — decay trajectory with a tiny vector-field network.
//!
//! ## Conversion notes
//!
//! * `metricPanel` / `progressBar` / `baseBackground` push into `&mut Vec<Shape>`.
//! * PORT NOTE: only the subset of the
//!   `crate::des::general::{neural_network, ode}` result types the scenes read
//!   is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_exponential, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight,
    Frame, FrameParts, LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const NEURAL_STAGE_W: f64 = 1100.0;
pub const NEURAL_STAGE_H: f64 = 720.0;

mod c {
    pub const INK: &str = "#172033";
    pub const MUTED: &str = "#64748b";
    pub const PANEL: &str = "#f8fafc";
    pub const GRID: &str = "#d7dee8";
    pub const BLUE: &str = "#2563eb";
    pub const GREEN: &str = "#16a34a";
    pub const AMBER: &str = "#f59e0b";
    pub const RED: &str = "#dc2626";
    pub const PURPLE: &str = "#7c3aed";
}

// PORT NOTE: local mirrors of the neural-network / ODE result types.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    pub biases: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct FeedForwardNetwork {
    pub input_dim: usize,
    pub layers: Vec<Layer>,
    pub param_count: usize,
}

impl FeedForwardNetwork {
    pub fn parameter_count(&self) -> usize {
        self.param_count
    }
}

#[derive(Clone, Debug, Default)]
pub struct SupervisedNeuralNetDESResult {
    pub network: FeedForwardNetwork,
    pub loss_history: Vec<f64>,
    pub predictions: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Default)]
pub struct NeuralQLearningResult {
    pub policy: Vec<i64>,
    pub total_episodes: f64,
    pub total_steps: f64,
    pub reward_history: Vec<f64>,
    pub loss_history: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct ODETrace {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
}

struct XorSample {
    x: [f64; 2],
    y: f64,
    label: &'static str,
}

pub fn build_neural_xor_animation(result: &SupervisedNeuralNetDESResult) -> (Vec<Frame>, Vec<ChartSpec>) {
    let total = result.loss_history.len();
    let num_frames = 120usize.min(20usize.max(((total as f64) / 260.0).ceil() as usize));
    let mut frames: Vec<Frame> = Vec::new();
    let final_pred: Vec<f64> = result.predictions.iter().map(|v| v[0]).collect();
    let samples = [
        XorSample { x: [0.0, 0.0], y: 0.0, label: "0 xor 0" },
        XorSample { x: [0.0, 1.0], y: 1.0, label: "0 xor 1" },
        XorSample { x: [1.0, 0.0], y: 1.0, label: "1 xor 0" },
        XorSample { x: [1.0, 1.0], y: 0.0, label: "1 xor 1" },
    ];

    for i in 0..num_frames {
        let step = (total.saturating_sub(1))
            .min(((i as f64 * (total as f64 - 1.0)) / (1.0_f64.max(num_frames as f64 - 1.0))).round() as usize);
        let sample = &samples[step % samples.len()];
        let loss = result.loss_history.get(step).copied().unwrap_or(0.0);
        let f = build_xor_frame(&result.network, step, total, loss, sample, &final_pred);
        frames.push(f.into_frame(step as f64, i as f64));
    }
    (frames, build_xor_charts(result))
}

fn build_xor_frame(
    network: &FeedForwardNetwork,
    step: usize,
    total: usize,
    loss: f64,
    sample: &XorSample,
    final_pred: &[f64],
) -> FrameParts {
    let mut shapes = base_background("Neural XOR", "supervised DES training");

    let mut dims: Vec<usize> = vec![network.input_dim];
    dims.extend(network.layers.iter().map(|l| l.biases.len()));
    let x0 = 90.0;
    let x1 = 660.0;
    let y0 = 110.0;
    let y1 = 410.0;
    let layer_xs: Vec<f64> = (0..dims.len())
        .map(|i| x0 + i as f64 * ((x1 - x0) / 1.0_f64.max(dims.len() as f64 - 1.0)))
        .collect();
    let mut node_pos: Vec<Vec<(f64, f64)>> = Vec::new();
    for li in 0..dims.len() {
        let n = dims[li];
        let ys: Vec<f64> = (0..n).map(|j| y0 + (j as f64 + 1.0) * ((y1 - y0) / (n as f64 + 1.0))).collect();
        node_pos.push(ys.into_iter().map(|y| (layer_xs[li], y)).collect());
    }

    // Edges.
    for li in 0..node_pos.len() - 1 {
        for a in &node_pos[li] {
            for b in &node_pos[li + 1] {
                shapes.push(Shape::Line(LineShape { x1: a.0, y1: a.1, x2: b.0, y2: b.1, stroke: "#cbd5e1".to_string(), stroke_width: Some(0.8), opacity: Some(0.8), ..Default::default() }));
            }
        }
    }
    // Moving token.
    let phase = (step % 4) as f64 / 3.0;
    shapes.push(Shape::Circle(CircleShape { x: x0 + phase * (x1 - x0), y: 72.0, r: 8.0, fill: c::PURPLE.to_string(), title: Some("sample token moving across layer stations".to_string()), ..Default::default() }));
    shapes.push(Shape::Line(LineShape { x1: x0, y1: 72.0, x2: x1, y2: 72.0, stroke: "#e2e8f0".to_string(), stroke_width: Some(2.0), ..Default::default() }));

    // Nodes.
    for li in 0..node_pos.len() {
        for j in 0..node_pos[li].len() {
            let p = node_pos[li][j];
            let active = li == 0 && sample.x.get(j).copied() == Some(1.0);
            shapes.push(Shape::Circle(CircleShape {
                x: p.0,
                y: p.1,
                r: 18.0,
                fill: if active { c::GREEN.to_string() } else { "#ffffff".to_string() },
                stroke: Some(if li == node_pos.len() - 1 { c::BLUE.to_string() } else { "#475569".to_string() }),
                stroke_width: Some(if li == node_pos.len() - 1 { 2.0 } else { 1.0 }),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: p.0,
                y: p.1 + 4.0,
                text: if li == 0 { js_num(sample.x[j]) } else { String::new() },
                anchor: Some(Anchor::Middle),
                font_size: Some(12.0),
                fill: Some(if active { "#fff".to_string() } else { c::INK.to_string() }),
                font_weight: Some(FontWeight::Bold),
                ..Default::default()
            }));
        }
        let label = if li == 0 {
            "input".to_string()
        } else if li == node_pos.len() - 1 {
            "output".to_string()
        } else {
            format!("hidden {li}")
        };
        shapes.push(Shape::Text(TextShape { x: layer_xs[li], y: y1 + 34.0, text: label, anchor: Some(Anchor::Middle), font_size: Some(12.0), fill: Some(c::MUTED.to_string()), ..Default::default() }));
    }

    // Metrics panel.
    let progress = if total <= 1 { 1.0 } else { step as f64 / (total as f64 - 1.0) };
    metric_panel(
        &mut shapes,
        740.0,
        86.0,
        310.0,
        330.0,
        &[
            ("sample".to_string(), format!("{} -> {}", sample.label, js_num(sample.y))),
            ("training step".to_string(), format!("{} / {}", step + 1, total)),
            ("progress".to_string(), format!("{}%", to_fixed(100.0 * progress, 1))),
            ("current loss".to_string(), to_exponential(loss, 3)),
            ("params".to_string(), network.parameter_count().to_string()),
        ],
    );
    progress_bar(&mut shapes, 760.0, 382.0, 270.0, 14.0, progress, c::BLUE);

    // Final prediction bars.
    shapes.push(Shape::Text(TextShape { x: 88.0, y: 468.0, text: "final predictions".to_string(), font_size: Some(13.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    let bar_labels = ["00", "01", "10", "11"];
    for i in 0..final_pred.len() {
        let x = 90.0 + i as f64 * 145.0;
        let y = 510.0;
        let h = 100.0 * final_pred[i];
        shapes.push(Shape::Rect(RectShape { x, y: y + 100.0 - h, w: 76.0, h, fill: if final_pred[i] > 0.5 { c::GREEN.to_string() } else { c::BLUE.to_string() }, rx: Some(3.0), ..Default::default() }));
        shapes.push(Shape::Rect(RectShape { x, y, w: 76.0, h: 100.0, fill: "none".to_string(), stroke: Some("#cbd5e1".to_string()), stroke_width: Some(1.0), rx: Some(3.0), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + 38.0, y: y + 124.0, text: bar_labels.get(i).copied().unwrap_or("").to_string(), anchor: Some(Anchor::Middle), font_size: Some(12.0), fill: Some(c::MUTED.to_string()), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + 38.0, y: y - 8.0, text: to_fixed(final_pred[i], 3), anchor: Some(Anchor::Middle), font_size: Some(12.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    }

    let caption = format!("step={} sample={} target={} loss={}", step + 1, sample.label, js_num(sample.y), to_exponential(loss, 3));
    FrameParts::with_caption(shapes, caption)
}

fn build_xor_charts(result: &SupervisedNeuralNetDESResult) -> Vec<ChartSpec> {
    let t: Vec<f64> = (0..result.loss_history.len()).map(|i| (i + 1) as f64).collect();
    let losses: Vec<f64> = result.loss_history.iter().map(|&x| 1e-12_f64.max(x)).collect();
    let pred_t: Vec<f64> = (0..result.predictions.len()).map(|i| i as f64).collect();
    let y_max = result.loss_history.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    vec![
        ChartSpec {
            x: 690.0,
            y: 440.0,
            w: 360.0,
            h: 120.0,
            title: Some("loss per sample".to_string()),
            y_min: Some(0.0),
            y_max: Some(y_max),
            series: vec![ChartSeries { label: "loss".to_string(), color: c::RED.to_string(), t, y: losses }],
            ..Default::default()
        },
        ChartSpec {
            x: 690.0,
            y: 575.0,
            w: 360.0,
            h: 110.0,
            title: Some("final XOR outputs".to_string()),
            y_min: Some(0.0),
            y_max: Some(1.0),
            series: vec![ChartSeries { label: "prediction".to_string(), color: c::GREEN.to_string(), t: pred_t, y: result.predictions.iter().map(|v| v[0]).collect() }],
            cursor: Some(false),
            ..Default::default()
        },
    ]
}

pub fn build_neural_q_corridor_animation(result: &NeuralQLearningResult, length: usize) -> (Vec<Frame>, Vec<ChartSpec>) {
    let mut frames: Vec<Frame> = Vec::new();
    let path = greedy_path(&result.policy, length);
    for i in 0..path.len() {
        let f = build_corridor_frame(result, length, &path, i);
        frames.push(f.into_frame(i as f64, i as f64));
    }
    (frames, build_q_charts(result))
}

fn build_corridor_frame(result: &NeuralQLearningResult, length: usize, path: &[usize], step: usize) -> FrameParts {
    let mut shapes = base_background("Neural Q-learning", "corridor MDP policy rollout");
    let cell_w = 120.0;
    let start_x = 90.0;
    let y = 230.0;
    let state = path[step];
    for s in 0..length {
        let x = start_x + s as f64 * cell_w;
        let is_goal = s == length - 1;
        let is_agent = s == state;
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w: 92.0,
            h: 92.0,
            fill: if is_goal { "#dcfce7".to_string() } else { "#fff".to_string() },
            stroke: Some(if is_agent { c::BLUE.to_string() } else { "#cbd5e1".to_string() }),
            stroke_width: Some(if is_agent { 3.0 } else { 1.0 }),
            rx: Some(5.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + 46.0,
            y: y + 54.0,
            text: if is_agent { "A".to_string() } else if is_goal { "G".to_string() } else { s.to_string() },
            anchor: Some(Anchor::Middle),
            font_size: Some(24.0),
            fill: Some(if is_goal { c::GREEN.to_string() } else { c::INK.to_string() }),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        if s < length - 1 {
            shapes.push(Shape::Line(LineShape { x1: x + 94.0, y1: y + 46.0, x2: x + cell_w - 12.0, y2: y + 46.0, stroke: "#94a3b8".to_string(), stroke_width: Some(2.0), ..Default::default() }));
            if result.policy.get(s).copied() == Some(1) {
                shapes.push(Shape::Path(PathShape {
                    d: format!("M {} {} L {} {} L {} {}", js_num(x + cell_w - 18.0), js_num(y + 40.0), js_num(x + cell_w - 8.0), js_num(y + 46.0), js_num(x + cell_w - 18.0), js_num(y + 52.0)),
                    stroke: Some(c::GREEN.to_string()),
                    stroke_width: Some(2.0),
                    fill: Some("none".to_string()),
                    ..Default::default()
                }));
            }
        }
    }
    metric_panel(
        &mut shapes,
        730.0,
        122.0,
        320.0,
        300.0,
        &[
            ("episodes".to_string(), js_num(result.total_episodes)),
            ("env steps".to_string(), js_num(result.total_steps)),
            ("eval success".to_string(), "100%".to_string()),
            ("mean eval length".to_string(), "5.0".to_string()),
            ("current state".to_string(), state.to_string()),
        ],
    );
    progress_bar(&mut shapes, 750.0, 388.0, 280.0, 14.0, step as f64 / 1.0_f64.max(path.len() as f64 - 1.0), c::GREEN);
    let policy_str: Vec<String> = result.policy.iter().map(|p| p.to_string()).collect();
    let caption = format!("greedy rollout step={} state={} policy=[{}]", step, state, policy_str.join(", "));
    FrameParts::with_caption(shapes, caption)
}

fn greedy_path(policy: &[i64], length: usize) -> Vec<usize> {
    let mut path = vec![0usize];
    let mut s: usize = 0;
    for _ in 0..length * 3 {
        if s == length - 1 {
            break;
        }
        let a = policy.get(s).copied().unwrap_or(1);
        s = if a == 0 { s.saturating_sub(1) } else { (length - 1).min(s + 1) };
        path.push(s);
    }
    path
}

fn build_q_charts(result: &NeuralQLearningResult) -> Vec<ChartSpec> {
    let episodes: Vec<f64> = (0..result.reward_history.len()).map(|i| (i + 1) as f64).collect();
    vec![
        ChartSpec {
            x: 70.0,
            y: 470.0,
            w: 470.0,
            h: 180.0,
            title: Some("episode return".to_string()),
            series: vec![ChartSeries { label: "return".to_string(), color: c::BLUE.to_string(), t: episodes, y: result.reward_history.clone() }],
            ..Default::default()
        },
        ChartSpec {
            x: 580.0,
            y: 470.0,
            w: 470.0,
            h: 180.0,
            title: Some("TD training loss".to_string()),
            y_min: Some(0.0),
            series: vec![ChartSeries { label: "loss".to_string(), color: c::RED.to_string(), t: (0..result.loss_history.len()).map(|i| (i + 1) as f64).collect(), y: result.loss_history.clone() }],
            ..Default::default()
        },
    ]
}

pub fn build_neural_ode_animation(trace: &ODETrace, rate: f64, exact_final: f64, error: f64) -> (Vec<Frame>, Vec<ChartSpec>) {
    let mut frames: Vec<Frame> = Vec::new();
    for i in 0..trace.t.len() {
        let f = build_ode_frame(trace, i, rate, exact_final, error);
        frames.push(f.into_frame(trace.t[i], i as f64));
    }
    let y: Vec<f64> = trace.y.iter().map(|v| v[0]).collect();
    let exact: Vec<f64> = trace.t.iter().map(|&t| y[0] * (-rate * t).exp()).collect();
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.05;
    let charts = vec![ChartSpec {
        x: 80.0,
        y: 430.0,
        w: 940.0,
        h: 230.0,
        title: Some("neural ODE trajectory".to_string()),
        y_min: Some(0.0),
        y_max: Some(y_max),
        series: vec![
            ChartSeries { label: "network RK4".to_string(), color: c::BLUE.to_string(), t: trace.t.clone(), y: y.clone() },
            ChartSeries { label: "exact exp decay".to_string(), color: c::GREEN.to_string(), t: trace.t.clone(), y: exact },
        ],
        ..Default::default()
    }];
    (frames, charts)
}

fn build_ode_frame(trace: &ODETrace, idx: usize, rate: f64, exact_final: f64, error: f64) -> FrameParts {
    let mut shapes = base_background("Neural ODE", &format!("network vector field dy/dt = -{} y", js_num(rate)));
    let t = trace.t[idx];
    let y = trace.y[idx][0];
    let x0 = 90.0;
    let y0 = 100.0;
    let w = 580.0;
    let h = 260.0;
    shapes.push(Shape::Rect(RectShape { x: x0, y: y0, w, h, fill: "#fff".to_string(), stroke: Some("#cbd5e1".to_string()), stroke_width: Some(1.0), rx: Some(5.0), ..Default::default() }));
    shapes.push(Shape::Line(LineShape { x1: x0 + 40.0, y1: y0 + h - 30.0, x2: x0 + w - 20.0, y2: y0 + h - 30.0, stroke: "#94a3b8".to_string(), stroke_width: Some(1.0), ..Default::default() }));
    shapes.push(Shape::Line(LineShape { x1: x0 + 40.0, y1: y0 + 20.0, x2: x0 + 40.0, y2: y0 + h - 30.0, stroke: "#94a3b8".to_string(), stroke_width: Some(1.0), ..Default::default() }));
    let t_max = trace.t[trace.t.len() - 1];
    let y_max = trace.y.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max);
    let sx = |tt: f64| x0 + 40.0 + (w - 70.0) * tt / 1e-12_f64.max(t_max);
    let sy = |yy: f64| y0 + h - 30.0 - (h - 60.0) * yy / 1e-12_f64.max(y_max);
    let mut d = String::new();
    for i in 0..=idx {
        d += &format!("{} {} {} ", if i == 0 { "M" } else { "L" }, to_fixed(sx(trace.t[i]), 2), to_fixed(sy(trace.y[i][0]), 2));
    }
    shapes.push(Shape::Path(PathShape { d, stroke: Some(c::BLUE.to_string()), stroke_width: Some(3.0), fill: Some("none".to_string()), ..Default::default() }));
    shapes.push(Shape::Circle(CircleShape { x: sx(t), y: sy(y), r: 8.0, fill: c::BLUE.to_string(), ..Default::default() }));

    // Tiny network/vector-field diagram.
    let nx = 750.0;
    let ny = 115.0;
    shapes.push(Shape::Rect(RectShape { x: nx - 30.0, y: ny - 35.0, w: 300.0, h: 250.0, fill: c::PANEL.to_string(), stroke: Some(c::GRID.to_string()), stroke_width: Some(1.0), rx: Some(5.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: nx, y: ny - 10.0, text: "vector-field network".to_string(), font_size: Some(13.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    shapes.push(Shape::Circle(CircleShape { x: nx + 45.0, y: ny + 80.0, r: 24.0, fill: "#fff".to_string(), stroke: Some(c::BLUE.to_string()), stroke_width: Some(2.0), label: Some("y".to_string()), ..Default::default() }));
    shapes.push(Shape::Circle(CircleShape { x: nx + 205.0, y: ny + 80.0, r: 24.0, fill: "#fff".to_string(), stroke: Some(c::GREEN.to_string()), stroke_width: Some(2.0), label: Some("dy".to_string()), ..Default::default() }));
    shapes.push(Shape::Line(LineShape { x1: nx + 70.0, y1: ny + 80.0, x2: nx + 180.0, y2: ny + 80.0, stroke: c::MUTED.to_string(), stroke_width: Some(2.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: nx + 125.0, y: ny + 70.0, text: format!("w = -{}", js_num(rate)), anchor: Some(Anchor::Middle), font_size: Some(12.0), fill: Some(c::INK.to_string()), ..Default::default() }));
    metric_panel(
        &mut shapes,
        730.0,
        345.0,
        320.0,
        70.0,
        &[
            ("final exact".to_string(), to_fixed(exact_final, 6)),
            ("abs error".to_string(), to_exponential(error, 3)),
        ],
    );
    let caption = format!("t={} y={} dy/dt={}", to_fixed(t, 3), to_fixed(y, 6), to_fixed(-rate * y, 6));
    FrameParts::with_caption(shapes, caption)
}

fn base_background(title: &str, subtitle: &str) -> Vec<Shape> {
    vec![
        Shape::Rect(RectShape { x: 0.0, y: 0.0, w: NEURAL_STAGE_W, h: NEURAL_STAGE_H, fill: "#f8fafc".to_string(), ..Default::default() }),
        Shape::Text(TextShape { x: 40.0, y: 36.0, text: title.to_string(), font_size: Some(20.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }),
        Shape::Text(TextShape { x: 40.0, y: 58.0, text: subtitle.to_string(), font_size: Some(12.0), fill: Some(c::MUTED.to_string()), ..Default::default() }),
    ]
}

fn metric_panel(shapes: &mut Vec<Shape>, x: f64, y: f64, w: f64, h: f64, rows: &[(String, String)]) {
    shapes.push(Shape::Rect(RectShape { x, y, w, h, fill: "#fff".to_string(), stroke: Some(c::GRID.to_string()), stroke_width: Some(1.0), rx: Some(5.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: x + 16.0, y: y + 28.0, text: "metrics".to_string(), font_size: Some(13.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    for (i, (label, value)) in rows.iter().enumerate() {
        let yy = y + 62.0 + i as f64 * 28.0;
        shapes.push(Shape::Text(TextShape { x: x + 16.0, y: yy, text: label.clone(), font_size: Some(12.0), fill: Some(c::MUTED.to_string()), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + w - 16.0, y: yy, text: value.clone(), anchor: Some(Anchor::End), font_size: Some(12.0), fill: Some(c::INK.to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    }
}

fn progress_bar(shapes: &mut Vec<Shape>, x: f64, y: f64, w: f64, h: f64, p: f64, fill: &str) {
    shapes.push(Shape::Rect(RectShape { x, y, w, h, fill: "#e2e8f0".to_string(), rx: Some(h / 2.0), ..Default::default() }));
    shapes.push(Shape::Rect(RectShape { x, y, w: p.clamp(0.0, 1.0) * w, h, fill: fill.to_string(), rx: Some(h / 2.0), ..Default::default() }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_path_walks_to_goal() {
        let path = greedy_path(&[1, 1, 1, 1], 5);
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn xor_animation_produces_frames_and_two_charts() {
        let result = SupervisedNeuralNetDESResult {
            network: FeedForwardNetwork { input_dim: 2, layers: vec![Layer { biases: vec![0.0, 0.0] }, Layer { biases: vec![0.0] }], param_count: 9 },
            loss_history: (0..300).map(|i| 1.0 / (i as f64 + 1.0)).collect(),
            predictions: vec![vec![0.1], vec![0.9], vec![0.8], vec![0.2]],
        };
        let (frames, charts) = build_neural_xor_animation(&result);
        assert!(!frames.is_empty());
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[1].cursor, Some(false));
    }
}
