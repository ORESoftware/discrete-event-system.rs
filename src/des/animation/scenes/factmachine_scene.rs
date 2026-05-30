//! Port of `src/des/animation/scenes/factmachine-scene.ts`.
//!
//! Builds frames for the fact-machine entity-architecture animation: a DES
//! station graph (NoiseTrader / Market / Bettor / Census / Resolution) with
//! movables in flight on the left, and belief-histogram / price / order-flow /
//! entropy analytics panels on the right.
//!
//! ## Conversion notes
//!
//! * `viridis(t)` builds an `#rrggbb` string via `format!`.
//! * `drawStation` / `drawEdge` push into `&mut Vec<Shape>`.
//! * PORT NOTE: only the subset of
//!   `crate::des::main_factmachine::{FactMachineParams, FactMachineResult}` the
//!   scene reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSpec, CircleShape, FontWeight, FrameParts, LineShape, PathShape,
    RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1080.0;
pub const STAGE_H: f64 = 720.0;

// Architecture panel.
const ARCH_X: f64 = 20.0;
const ARCH_Y: f64 = 40.0;
const ARCH_W: f64 = 540.0;
const ARCH_H: f64 = 660.0;
// Analytics panel (right side).
const HIST_X: f64 = 580.0;
const HIST_Y: f64 = 60.0;
const HIST_W: f64 = 480.0;
const HIST_H: f64 = 220.0;
const PRICE_X: f64 = 580.0;
const PRICE_Y: f64 = 300.0;
const PRICE_W: f64 = 480.0;
const PRICE_H: f64 = 200.0;
const FLOW_X: f64 = 580.0;
const FLOW_Y: f64 = 520.0;
const FLOW_W: f64 = 232.0;
const FLOW_H: f64 = 180.0;
const ENT_X: f64 = 828.0;
const ENT_Y: f64 = 520.0;
const ENT_W: f64 = 232.0;
const ENT_H: f64 = 180.0;

// Station boxes inside ARCH panel.
const NOISE_X: f64 = ARCH_X + 30.0;
const NOISE_Y: f64 = ARCH_Y + 60.0;
const STATION_W: f64 = 150.0;
const STATION_H: f64 = 90.0;
const MARKET_X: f64 = ARCH_X + 280.0;
const MARKET_Y: f64 = ARCH_Y + 240.0;
const BETTOR_X: f64 = ARCH_X + 30.0;
const BETTOR_Y: f64 = ARCH_Y + 240.0;
const CENSUS_X: f64 = ARCH_X + 280.0;
const CENSUS_Y: f64 = ARCH_Y + 60.0;
const RESOL_X: f64 = ARCH_X + 280.0;
const RESOL_Y: f64 = ARCH_Y + 440.0;

// PORT NOTE: local mirror of the fact-machine params/result (subset used here).
#[derive(Clone, Debug, Default)]
pub struct FactMachineParams {
    pub k_noise: f64,
    pub informedness: f64,
    pub liquidity: f64,
    pub market_type: String,
    pub theta_bins: f64,
    pub policy: String,
    pub t: f64,
    pub n_voters: f64,
    pub true_theta: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FactMachineResult {
    pub price_history: Vec<Vec<f64>>,
    pub belief_mean: Vec<f64>,
    pub yes_orders_history: Vec<f64>,
    pub total_orders_history: Vec<f64>,
    pub belief_entropy: Vec<f64>,
}

/// `Partial<ArchitectureFrameArgs>` — every field optional.
#[derive(Clone, Debug, Default)]
pub struct ArchitectureFrameArgs {
    pub tick: Option<f64>,
    pub phase: Option<usize>,
    pub noise_order_count: Option<f64>,
    pub noise_yes: Option<f64>,
    pub noise_total: Option<f64>,
    pub bettor_action: Option<f64>,
    pub voter_count: Option<f64>,
    pub resolution_outcome: Option<f64>,
    pub vote_fraction: Option<f64>,
    pub belief_weights: Option<Vec<f64>>,
    pub prices: Option<Vec<f64>>,
}

fn hex2(n: f64) -> String {
    let v = n.round().clamp(0.0, 255.0) as u32;
    format!("{v:02x}")
}

fn viridis(t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    let r = 68.0 + (253.0 - 68.0) * t;
    let g = 1.0 + (231.0 - 1.0) * t.sqrt();
    let b = 84.0 + (37.0 - 84.0) * t * t;
    format!("#{}{}{}", hex2(r), hex2(g), hex2(b))
}

/// Draw a labelled station box. Optionally highlighted (active this phase).
fn draw_station(
    shapes: &mut Vec<Shape>,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    title: &str,
    lines: &[String],
    active: bool,
) {
    shapes.push(Shape::Rect(RectShape {
        x,
        y,
        w,
        h,
        fill: if active { "#fef3c7".to_string() } else { "#1e293b".to_string() },
        stroke: Some(if active { "#f59e0b".to_string() } else { "#475569".to_string() }),
        stroke_width: Some(if active { 3.0 } else { 1.5 }),
        rx: Some(8.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: x + w / 2.0,
        y: y + 22.0,
        text: title.to_string(),
        font_size: Some(13.0),
        fill: Some(if active { "#92400e".to_string() } else { "#fde68a".to_string() }),
        font_weight: Some(FontWeight::Bold),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    for (i, line) in lines.iter().enumerate() {
        shapes.push(Shape::Text(TextShape {
            x: x + w / 2.0,
            y: y + 44.0 + i as f64 * 16.0,
            text: line.clone(),
            font_size: Some(11.0),
            fill: Some(if active { "#1f2937".to_string() } else { "#cbd5e1".to_string() }),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
}

/// Draw an edge with optional in-flight movables; `progress` is the dot centre.
#[allow(clippy::too_many_arguments)]
fn draw_edge(
    shapes: &mut Vec<Shape>,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    label: &str,
    dot_count: f64,
    dot_color: &str,
    progress: f64,
) {
    shapes.push(Shape::Line(LineShape { x1, y1, x2, y2, stroke: "#64748b".to_string(), stroke_width: Some(1.5), opacity: Some(0.7), ..Default::default() }));
    // Arrow tip.
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = dx.hypot(dy);
    if len > 0.0 {
        let ux = dx / len;
        let uy = dy / len;
        let ax = x2 - 8.0 * ux;
        let ay = y2 - 8.0 * uy;
        let px = -uy;
        let py = ux;
        shapes.push(Shape::Line(LineShape { x1: ax, y1: ay, x2: ax + 4.0 * px, y2: ay + 4.0 * py, stroke: "#64748b".to_string(), stroke_width: Some(1.5), ..Default::default() }));
        shapes.push(Shape::Line(LineShape { x1: ax, y1: ay, x2: ax - 4.0 * px, y2: ay - 4.0 * py, stroke: "#64748b".to_string(), stroke_width: Some(1.5), ..Default::default() }));
    }
    // Label at midpoint.
    let mx = (x1 + x2) / 2.0;
    let my = (y1 + y2) / 2.0;
    shapes.push(Shape::Rect(RectShape { x: mx - 50.0, y: my - 8.0, w: 100.0, h: 16.0, fill: "#0f172a".to_string(), stroke: Some("#334155".to_string()), stroke_width: Some(1.0), rx: Some(3.0), opacity: Some(0.9), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: mx, y: my + 4.0, text: label.to_string(), font_size: Some(10.0), fill: Some("#fde68a".to_string()), anchor: Some(Anchor::Middle), ..Default::default() }));
    // Movable dots: spread along the edge, centred at `progress`.
    let n = dot_count.min(8.0);
    let count = n.max(0.0) as usize;
    for i in 0..count {
        let t = (progress + (i as f64 - (n - 1.0) / 2.0) * 0.04).clamp(0.02, 0.98);
        let cx = x1 + (x2 - x1) * t;
        let cy = y1 + (y2 - y1) * t;
        shapes.push(Shape::Circle(CircleShape { x: cx, y: cy, r: 3.5, fill: dot_color.to_string(), stroke: Some("#0b1220".to_string()), stroke_width: Some(0.5), ..Default::default() }));
    }
}

pub fn build_fact_machine_frame(
    tick: f64,
    belief_weights: &[f64],
    result: &FactMachineResult,
    params: &FactMachineParams,
    arch: Option<&ArchitectureFrameArgs>,
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();

    // ============ ARCHITECTURE (left panel) ============
    let phase = arch.and_then(|a| a.phase).unwrap_or(1);
    let is_last = tick >= params.t;
    let noise_active = phase == 0;
    let market_active = phase == 1 || phase == 3 || phase == 4;
    let census_active = phase == 1;
    let bettor_active = phase == 2;
    let resolution_active = phase == 4 && is_last;

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
    let phase_labels = [
        "phase 0: noise \u{2192} market",
        "phase 1: noise settles \u{00b7} census reads",
        "phase 2: bettor reads \u{00b7} sends order",
        "phase 3: bettor settles",
        "phase 4: RESOLUTION (votes flow)",
    ];
    let phase_label = phase_labels[phase];
    shapes.push(Shape::Text(TextShape {
        x: ARCH_X + ARCH_W / 2.0,
        y: ARCH_Y + 24.0,
        text: format!("DES architecture — t = {}/{} — {}", js_num(tick), js_num(params.t), phase_label),
        font_size: Some(14.0),
        fill: Some("#f1f5f9".to_string()),
        font_weight: Some(FontWeight::Bold),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));

    // Stations.
    let noise_order_count = arch.and_then(|a| a.noise_order_count).unwrap_or(0.0);
    let bettor_action = arch.and_then(|a| a.bettor_action).unwrap_or(-1.0);
    let yes = arch.and_then(|a| a.noise_yes).unwrap_or(0.0);
    let tot = arch.and_then(|a| a.noise_total).unwrap_or(0.0);
    draw_station(
        &mut shapes,
        NOISE_X,
        NOISE_Y,
        STATION_W,
        STATION_H,
        "NoiseTrader",
        &[
            format!("K = {}", js_num(params.k_noise)),
            format!("informedness {}", to_fixed(params.informedness, 2)),
            if tot > 0.0 { format!("last batch {}/{}", js_num(yes), js_num(tot)) } else { "idle".to_string() },
        ],
        noise_active,
    );
    let price0 = arch.and_then(|a| a.prices.as_ref().and_then(|p| p.first().copied())).unwrap_or(0.5);
    let b_divisor = (if params.market_type == "binary" { 2.0 } else { params.theta_bins }).ln();
    draw_station(
        &mut shapes,
        MARKET_X,
        MARKET_Y,
        STATION_W,
        STATION_H,
        "Market (LMSR)",
        &[
            format!("P(YES) = {}", to_fixed(price0, 3)),
            format!("liq L = {}", js_num(params.liquidity)),
            format!("b = {}", to_fixed(params.liquidity / b_divisor, 2)),
        ],
        market_active,
    );
    let mut b_mean = 0.0;
    for i in 0..belief_weights.len() {
        b_mean += belief_weights[i] * (i as f64 / (belief_weights.len() as f64 - 1.0));
    }
    draw_station(
        &mut shapes,
        BETTOR_X,
        BETTOR_Y,
        STATION_W,
        STATION_H,
        "Bettor",
        &[
            format!("policy = {}", params.policy),
            format!("E[\u{03b8}] = {}", to_fixed(b_mean, 3)),
            if bettor_action < 0.0 { "action = hold".to_string() } else { format!("action = buy {}", js_num(bettor_action)) },
        ],
        bettor_active,
    );
    draw_station(
        &mut shapes,
        CENSUS_X,
        CENSUS_Y,
        STATION_W,
        STATION_H,
        "Census",
        &["snapshots".to_string(), "prices + b(\u{03b8})".to_string(), "for trace".to_string()],
        census_active,
    );
    let resolution_lines: Vec<String> = if is_last && phase == 4 {
        vec![
            "fires NOW".to_string(),
            format!("{} votes", js_num(params.n_voters)),
            match arch.and_then(|a| a.resolution_outcome) {
                Some(outcome) => format!("outcome = {}", if outcome == 0.0 { "YES" } else { "NO" }),
                None => String::new(),
            },
        ]
    } else {
        vec![
            "pending".to_string(),
            format!("fires at t={}", js_num(params.t)),
            format!("{} voters", js_num(params.n_voters)),
        ]
    };
    draw_station(&mut shapes, RESOL_X, RESOL_Y, STATION_W, STATION_H, "Resolution", &resolution_lines, resolution_active);

    // Edges with movables.
    let cxf = |x: f64, w: f64| x + w / 2.0;
    // 1. NoiseTrader → Market.
    if phase == 0 {
        draw_edge(&mut shapes, cxf(NOISE_X, STATION_W), NOISE_Y + STATION_H, cxf(MARKET_X, STATION_W), MARKET_Y, &format!("{} noise orders", js_num(noise_order_count)), noise_order_count, "#22d3ee", 0.5);
    } else {
        draw_edge(&mut shapes, cxf(NOISE_X, STATION_W), NOISE_Y + STATION_H, cxf(MARKET_X, STATION_W), MARKET_Y, &(if tot > 0.0 { format!("{} settled", js_num(tot)) } else { "idle".to_string() }), 0.0, "#22d3ee", 0.0);
    }
    // 2. Market → Census (read).
    if phase == 1 {
        draw_edge(&mut shapes, MARKET_X + STATION_W / 2.0, MARKET_Y, CENSUS_X + STATION_W / 2.0, CENSUS_Y + STATION_H, "snapshot", 1.0, "#a78bfa", 0.4);
    } else {
        draw_edge(&mut shapes, MARKET_X + STATION_W / 2.0, MARKET_Y, CENSUS_X + STATION_W / 2.0, CENSUS_Y + STATION_H, "reads", 0.0, "#a78bfa", 0.0);
    }
    // 3. Market → Bettor (read prices).
    if phase == 2 {
        draw_edge(&mut shapes, MARKET_X, MARKET_Y + STATION_H / 2.0, BETTOR_X + STATION_W, BETTOR_Y + STATION_H / 2.0, "reads prices", 1.0, "#fb7185", 0.5);
    }
    // 4. Bettor → Market (send order).
    if phase == 2 && bettor_action >= 0.0 {
        let order_label = if bettor_action == 0.0 {
            "YES".to_string()
        } else if bettor_action == 1.0 {
            "NO".to_string()
        } else {
            format!("#{}", js_num(bettor_action))
        };
        draw_edge(&mut shapes, BETTOR_X + STATION_W, BETTOR_Y + 20.0, MARKET_X, MARKET_Y + 20.0, &format!("1 order (buy {order_label})"), 1.0, "#f472b6", 0.5);
    }
    // 5. Resolution receives votes (only final phase 4 of last tick).
    if resolution_active {
        draw_edge(&mut shapes, MARKET_X + STATION_W / 2.0, MARKET_Y + STATION_H, RESOL_X + STATION_W / 2.0, RESOL_Y, &format!("{} votes", js_num(params.n_voters)), params.n_voters, "#fda4af", 0.6);
    }

    // ============ ANALYTICS (right panel) ============
    let k = belief_weights.len();

    // Belief histogram (compact).
    shapes.push(Shape::Rect(RectShape { x: HIST_X - 6.0, y: HIST_Y - 6.0, w: HIST_W + 12.0, h: HIST_H + 12.0, fill: "#0f172a".to_string(), stroke: Some("#334155".to_string()), stroke_width: Some(1.0), rx: Some(4.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: HIST_X, y: HIST_Y - 12.0, text: format!("Bettor's belief b(\u{03b8}) — t = {}", js_num(tick)), font_size: Some(12.0), fill: Some("#e2e8f0".to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    let mut max_w = 0.0_f64;
    for &w in belief_weights {
        if w > max_w {
            max_w = w;
        }
    }
    max_w = max_w.max(1.2 / k as f64);
    let bar_w = HIST_W / k as f64;
    for i in 0..k {
        let theta = i as f64 / (k as f64 - 1.0);
        let h = (belief_weights[i] / max_w) * (HIST_H - 30.0);
        shapes.push(Shape::Rect(RectShape {
            x: HIST_X + i as f64 * bar_w + bar_w * 0.05,
            y: HIST_Y + (HIST_H - 30.0) - h + 4.0,
            w: bar_w * 0.9,
            h: h.max(0.5),
            fill: viridis(theta),
            ..Default::default()
        }));
    }
    // True θ marker.
    let truth_x = HIST_X + params.true_theta * HIST_W;
    shapes.push(Shape::Line(LineShape { x1: truth_x, y1: HIST_Y + 4.0, x2: truth_x, y2: HIST_Y + HIST_H - 26.0, stroke: "#dc2626".to_string(), stroke_width: Some(2.0), dasharray: Some("4 3".to_string()), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: truth_x + 4.0, y: HIST_Y + 16.0, text: format!("true \u{03b8} = {}", to_fixed(params.true_theta, 2)), font_size: Some(10.0), fill: Some("#fca5a5".to_string()), ..Default::default() }));
    // E_b[θ] marker.
    let mean_x = HIST_X + b_mean * HIST_W;
    shapes.push(Shape::Line(LineShape { x1: mean_x, y1: HIST_Y + 4.0, x2: mean_x, y2: HIST_Y + HIST_H - 26.0, stroke: "#60a5fa".to_string(), stroke_width: Some(2.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: mean_x + 4.0, y: HIST_Y + 32.0, text: format!("E[\u{03b8}] = {}", to_fixed(b_mean, 3)), font_size: Some(10.0), fill: Some("#93c5fd".to_string()), ..Default::default() }));

    // Price + E[θ] line chart (compact).
    shapes.push(Shape::Rect(RectShape { x: PRICE_X - 6.0, y: PRICE_Y - 6.0, w: PRICE_W + 12.0, h: PRICE_H + 12.0, fill: "#0f172a".to_string(), stroke: Some("#334155".to_string()), stroke_width: Some(1.0), rx: Some(4.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: PRICE_X, y: PRICE_Y - 12.0, text: format!("P(YES) red, E[\u{03b8}] blue — through t = {}", js_num(tick)), font_size: Some(12.0), fill: Some("#e2e8f0".to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    let y_true = PRICE_Y + (1.0 - params.true_theta) * (PRICE_H - 18.0) + 4.0;
    shapes.push(Shape::Line(LineShape { x1: PRICE_X, y1: y_true, x2: PRICE_X + PRICE_W, y2: y_true, stroke: "#dc2626".to_string(), stroke_width: Some(1.0), dasharray: Some("3 3".to_string()), ..Default::default() }));
    let t_idx = result.price_history.len() as f64 - 1.0;
    let x_at = |i: f64| PRICE_X + (i / t_idx.max(1.0)) * PRICE_W;
    let y_at = |v: f64| PRICE_Y + (1.0 - v) * (PRICE_H - 18.0) + 4.0;
    let up_to = (tick.min(t_idx)).max(0.0) as usize;
    let price_scalar = |t: usize| -> f64 {
        let ph = &result.price_history[t];
        if ph.len() == 2 {
            return ph[0];
        }
        let mut best_j = 0;
        for j in 1..ph.len() {
            if ph[j] > ph[best_j] {
                best_j = j;
            }
        }
        (best_j as f64 + 0.5) / ph.len() as f64
    };
    let mut price_d = String::new();
    let mut mean_d = String::new();
    for i in 0..=up_to {
        price_d += &format!("{}{} {}", if i == 0 { "M" } else { " L" }, to_fixed(x_at(i as f64), 1), to_fixed(y_at(price_scalar(i)), 1));
        mean_d += &format!("{}{} {}", if i == 0 { "M" } else { " L" }, to_fixed(x_at(i as f64), 1), to_fixed(y_at(result.belief_mean[i]), 1));
    }
    shapes.push(Shape::Path(PathShape { d: price_d, stroke: Some("#dc2626".to_string()), stroke_width: Some(2.0), fill: Some("none".to_string()), ..Default::default() }));
    shapes.push(Shape::Path(PathShape { d: mean_d, stroke: Some("#60a5fa".to_string()), stroke_width: Some(2.0), fill: Some("none".to_string()), ..Default::default() }));

    // Order flow ratio panel.
    shapes.push(Shape::Rect(RectShape { x: FLOW_X - 6.0, y: FLOW_Y - 6.0, w: FLOW_W + 12.0, h: FLOW_H + 12.0, fill: "#0f172a".to_string(), stroke: Some("#334155".to_string()), stroke_width: Some(1.0), rx: Some(4.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: FLOW_X, y: FLOW_Y - 12.0, text: "Order flow YES/total".to_string(), font_size: Some(11.0), fill: Some("#e2e8f0".to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    let flow_true = FLOW_Y + (1.0 - params.true_theta) * (FLOW_H - 18.0) + 4.0;
    shapes.push(Shape::Line(LineShape { x1: FLOW_X, y1: flow_true, x2: FLOW_X + FLOW_W, y2: flow_true, stroke: "#dc2626".to_string(), stroke_width: Some(1.0), dasharray: Some("3 3".to_string()), ..Default::default() }));
    let flow_upper = (tick - 1.0).min(result.yes_orders_history.len() as f64 - 1.0);
    if flow_upper >= 0.0 {
        for i in 0..=(flow_upper as usize) {
            let ratio = if result.total_orders_history[i] > 0.0 {
                result.yes_orders_history[i] / result.total_orders_history[i]
            } else {
                0.5
            };
            let x = FLOW_X + ((i as f64 + 0.5) / t_idx.max(1.0)) * FLOW_W;
            let y = FLOW_Y + (1.0 - ratio) * (FLOW_H - 18.0) + 4.0;
            shapes.push(Shape::Circle(CircleShape { x, y, r: 2.5, fill: "#fbbf24".to_string(), ..Default::default() }));
        }
    }

    // Entropy panel.
    shapes.push(Shape::Rect(RectShape { x: ENT_X - 6.0, y: ENT_Y - 6.0, w: ENT_W + 12.0, h: ENT_H + 12.0, fill: "#0f172a".to_string(), stroke: Some("#334155".to_string()), stroke_width: Some(1.0), rx: Some(4.0), ..Default::default() }));
    shapes.push(Shape::Text(TextShape { x: ENT_X, y: ENT_Y - 12.0, text: "Belief entropy H(b)".to_string(), font_size: Some(11.0), fill: Some("#e2e8f0".to_string()), font_weight: Some(FontWeight::Bold), ..Default::default() }));
    let max_h = (k as f64).ln();
    let mut ent_d = String::new();
    for i in 0..=up_to {
        let x = ENT_X + (i as f64 / t_idx.max(1.0)) * ENT_W;
        let y = ENT_Y + (1.0 - result.belief_entropy[i] / max_h) * (ENT_H - 18.0) + 4.0;
        ent_d += &format!("{}{} {}", if i == 0 { "M" } else { " L" }, to_fixed(x, 1), to_fixed(y, 1));
    }
    shapes.push(Shape::Path(PathShape { d: ent_d, stroke: Some("#a78bfa".to_string()), stroke_width: Some(2.0), fill: Some("none".to_string()), ..Default::default() }));

    let caption = format!("t={} {}", js_num(tick), phase_label);
    FrameParts::with_caption(shapes, caption)
}

pub fn build_fact_machine_charts(_r: &FactMachineResult) -> Vec<ChartSpec> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints() {
        assert_eq!(viridis(0.0), "#440154");
        assert_eq!(viridis(1.0), "#fde725");
    }

    #[test]
    fn frame_caption_has_phase_label() {
        let params = FactMachineParams {
            k_noise: 10.0,
            informedness: 0.3,
            liquidity: 50.0,
            market_type: "binary".to_string(),
            theta_bins: 11.0,
            policy: "greedy".to_string(),
            t: 20.0,
            n_voters: 100.0,
            true_theta: 0.6,
        };
        let result = FactMachineResult {
            price_history: vec![vec![0.5, 0.5], vec![0.6, 0.4]],
            belief_mean: vec![0.5, 0.55],
            yes_orders_history: vec![3.0, 4.0],
            total_orders_history: vec![6.0, 8.0],
            belief_entropy: vec![1.0, 0.9],
        };
        let belief = vec![0.2, 0.3, 0.5];
        let fp = build_fact_machine_frame(1.0, &belief, &result, &params, None);
        assert_eq!(fp.caption.as_deref(), Some("t=1 phase 1: noise settles \u{00b7} census reads"));
    }
}
