//! Port of `src/des/animation/scenes/computer-network-scene.ts`.
//!
//! Builds frames + charts for the computer-network DES animation: packet
//! motion, per-link queue buildup, bottleneck highlighting, a metrics panel and
//! a fan-out-policy explainer.
//!
//! ## Conversion notes
//!
//! * `draw*` helpers that push into a shared array take `&mut Vec<Shape>`.
//! * `layoutNodes` returns a `HashMap<String, Point>`.
//! * `PROTOCOL_COLOR` keyed by the protocol union becomes [`protocol_color`].
//! * Interpolated `rgb(..)` / path `d` strings use `format!` + `js_num`.
//! * PORT NOTE: only the subset of the
//!   `crate::des::general::computer_network` problem/result types the scene
//!   reads is mirrored locally below.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, Frame, FrameParts,
    LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const COMPUTER_NETWORK_STAGE_W: f64 = 1200.0;
pub const COMPUTER_NETWORK_STAGE_H: f64 = 760.0;

const NET_X: f64 = 40.0;
const NET_Y: f64 = 50.0;
const NET_W: f64 = 820.0;
const NET_H: f64 = 440.0;
const PANEL_X: f64 = 890.0;
const PANEL_Y: f64 = 50.0;
const PANEL_W: f64 = 270.0;
const PANEL_H: f64 = 440.0;

/// `'raw' | 'tcp' | 'udp' | 'http'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkProtocol {
    Raw,
    Tcp,
    Udp,
    Http,
}

fn protocol_color(p: NetworkProtocol) -> &'static str {
    match p {
        NetworkProtocol::Raw => "#64748b",
        NetworkProtocol::Tcp => "#2563eb",
        NetworkProtocol::Udp => "#16a34a",
        NetworkProtocol::Http => "#dc2626",
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// =============================================================================
// PORT NOTE: local mirror of the computer-network problem/result types
// (only the fields the scene reads). Replace with the canonical types once
// `computer_network.rs`'s public shape is confirmed.
// =============================================================================

#[derive(Clone, Debug)]
pub struct NetworkNode {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug)]
pub struct NetworkLinkSpec {
    pub id: String,
    pub from: String,
    pub to: String,
    pub bidirectional: bool,
}

#[derive(Clone, Debug)]
pub struct ComputerNetworkProblem {
    pub nodes: Vec<NetworkNode>,
    pub links: Vec<NetworkLinkSpec>,
}

#[derive(Clone, Debug)]
pub struct NetworkPacketSnapshot {
    pub packet_id: String,
    pub flow_id: String,
    pub protocol: NetworkProtocol,
    pub hops: Vec<String>,
    pub created_at_ms: f64,
    pub delivered_at_ms: Option<f64>,
    pub dropped_at_ms: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct NetworkTimeSample {
    pub t_ms: f64,
    pub generated_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub active_packets: f64,
    pub node_queues: HashMap<String, f64>,
    pub link_in_flight: HashMap<String, f64>,
    pub link_utilization: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct LinkStat {
    pub id: String,
    pub utilization: f64,
    pub dropped_packets: f64,
    pub delivered_packets: f64,
    pub final_in_flight: f64,
    pub queue_limit_packets: f64,
}

#[derive(Clone, Debug)]
pub struct NodeStat {
    pub id: String,
    pub final_queue: f64,
    pub dropped_packets: f64,
    pub queue_limit_packets: f64,
}

#[derive(Clone, Debug)]
pub struct Bottleneck {
    pub id: String,
    pub kind: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct FlowStat {
    pub id: String,
    pub goodput_mbps: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ComputerNetworkResult {
    pub delivered_packets_trace: Vec<NetworkPacketSnapshot>,
    pub dropped_packets_trace: Vec<NetworkPacketSnapshot>,
    pub time_series: Vec<NetworkTimeSample>,
    pub generated_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub active_packets: f64,
    pub max_active_packets: f64,
    pub link_stats: Vec<LinkStat>,
    pub node_stats: Vec<NodeStat>,
    pub bottlenecks: Vec<Bottleneck>,
    pub offered_load_mbps: f64,
    pub throughput_mbps: f64,
    pub goodput_mbps: f64,
    pub delivery_ratio: f64,
    pub flow_stats: Vec<FlowStat>,
    pub total_simulated_ms: f64,
}

impl Default for ComputerNetworkProblem {
    fn default() -> Self {
        ComputerNetworkProblem { nodes: Vec::new(), links: Vec::new() }
    }
}

pub fn build_computer_network_animation(
    problem: &ComputerNetworkProblem,
    result: &ComputerNetworkResult,
) -> (Vec<Frame>, Vec<ChartSpec>) {
    let normalized = normalize_links(&problem.links);
    let coords = layout_nodes(problem);
    let mut packet_traces = result.delivered_packets_trace.clone();
    packet_traces.extend(result.dropped_packets_trace.clone());

    let mut frames: Vec<Frame> = Vec::new();
    let samples: Vec<NetworkTimeSample> = if !result.time_series.is_empty() {
        result.time_series.clone()
    } else {
        vec![NetworkTimeSample {
            t_ms: 0.0,
            generated_packets: result.generated_packets,
            delivered_packets: result.delivered_packets,
            dropped_packets: result.dropped_packets,
            active_packets: result.active_packets,
            ..Default::default()
        }]
    };

    for (i, sample) in samples.iter().enumerate() {
        let frame = build_computer_network_frame(sample, i, problem, &normalized, &coords, result, &packet_traces);
        frames.push(frame.into_frame(sample.t_ms, i as f64));
    }

    (frames, build_computer_network_charts(result))
}

#[allow(clippy::too_many_arguments)]
fn build_computer_network_frame(
    sample: &NetworkTimeSample,
    tick: usize,
    problem: &ComputerNetworkProblem,
    links: &[NetworkLinkSpec],
    coords: &HashMap<String, Point>,
    result: &ComputerNetworkResult,
    packet_traces: &[NetworkPacketSnapshot],
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let link_stats: HashMap<String, LinkStat> =
        result.link_stats.iter().map(|l| (l.id.clone(), l.clone())).collect();
    let node_stats: HashMap<String, NodeStat> =
        result.node_stats.iter().map(|n| (n.id.clone(), n.clone())).collect();
    let top_bottleneck_id = result.bottlenecks.first().map(|b| b.id.clone());

    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: COMPUTER_NETWORK_STAGE_W,
        h: COMPUTER_NETWORK_STAGE_H,
        fill: "#f8fafc".to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 40.0,
        y: 28.0,
        text: "Network topology".to_string(),
        font_size: Some(18.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 220.0,
        y: 28.0,
        text: format!(
            "t={}ms  active={}  delivered={}  dropped={}",
            to_fixed(sample.t_ms, 0),
            js_num(sample.active_packets),
            js_num(sample.delivered_packets),
            js_num(sample.dropped_packets)
        ),
        font_size: Some(12.0),
        fill: Some("#475569".to_string()),
        ..Default::default()
    }));

    shapes.push(Shape::Rect(RectShape {
        x: NET_X,
        y: NET_Y,
        w: NET_W,
        h: NET_H,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));

    for link in links {
        draw_link(&mut shapes, link, coords, &link_stats, sample, &top_bottleneck_id);
    }
    for pkt in packet_traces {
        draw_packet_if_active(&mut shapes, pkt, sample.t_ms, coords);
    }
    for node in &problem.nodes {
        draw_node(&mut shapes, &node.id, &node.kind, coords, sample, &node_stats, &top_bottleneck_id);
    }

    draw_legend(&mut shapes);
    draw_metrics_panel(&mut shapes, result, sample);
    draw_fanout_policy_panel(&mut shapes, tick);

    let caption = match result.bottlenecks.first() {
        Some(top) => format!(
            "top bottleneck {}:{} ({}); active={}, dropped={}",
            top.kind,
            top.id,
            top.reason,
            js_num(sample.active_packets),
            js_num(sample.dropped_packets)
        ),
        None => format!(
            "active={}, dropped={}",
            js_num(sample.active_packets),
            js_num(sample.dropped_packets)
        ),
    };
    FrameParts::with_caption(shapes, caption)
}

fn draw_link(
    shapes: &mut Vec<Shape>,
    link: &NetworkLinkSpec,
    coords: &HashMap<String, Point>,
    link_stats: &HashMap<String, LinkStat>,
    sample: &NetworkTimeSample,
    top_bottleneck_id: &Option<String>,
) {
    let (a, b) = match (coords.get(&link.from), coords.get(&link.to)) {
        (Some(a), Some(b)) => (*a, *b),
        _ => return,
    };
    let is_reverse = link.id.contains(":rev");
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let offset = if is_reverse { -7.0 } else { 7.0 };
    let n = Point { x: -dy / len * offset, y: dx / len * offset };
    let aa = Point { x: a.x + n.x, y: a.y + n.y };
    let bb = Point { x: b.x + n.x, y: b.y + n.y };
    let st = link_stats.get(&link.id);
    let util = sample
        .link_utilization
        .get(&link.id)
        .copied()
        .or_else(|| st.map(|s| s.utilization))
        .unwrap_or(0.0);
    let drops = st.map(|s| s.dropped_packets).unwrap_or(0.0);
    let is_top = Some(&link.id) == top_bottleneck_id.as_ref();
    let stroke = if drops > 0.0 {
        "#dc2626"
    } else if util > 0.9 {
        "#f97316"
    } else if util > 0.5 {
        "#eab308"
    } else {
        "#64748b"
    };
    let width = if is_top { 5.0 } else { 1.4 + 4.0 * 1.0_f64.min(util) };
    shapes.push(Shape::Line(LineShape {
        x1: aa.x,
        y1: aa.y,
        x2: bb.x,
        y2: bb.y,
        stroke: stroke.to_string(),
        stroke_width: Some(width),
        opacity: Some(if is_reverse { 0.45 } else { 0.85 }),
        dasharray: if is_reverse { Some("4,4".to_string()) } else { None },
        ..Default::default()
    }));
    draw_arrow(shapes, aa, bb, stroke);
    let mid = Point { x: (aa.x + bb.x) / 2.0, y: (aa.y + bb.y) / 2.0 };
    let inflight = sample
        .link_in_flight
        .get(&link.id)
        .copied()
        .or_else(|| st.map(|s| s.final_in_flight))
        .unwrap_or(0.0);
    let q_w = 74.0_f64.min(8.0 + inflight * 0.7);
    shapes.push(Shape::Rect(RectShape {
        x: mid.x - q_w / 2.0,
        y: mid.y - 27.0,
        w: q_w,
        h: 12.0,
        fill: queue_color(inflight, st.map(|s| s.queue_limit_packets).unwrap_or(1.0)),
        stroke: Some("#ffffff".to_string()),
        stroke_width: Some(1.0),
        rx: Some(3.0),
        title: Some(format!(
            "{}: in flight {}, util {}%",
            link.id,
            js_num(inflight),
            to_fixed(util * 100.0, 1)
        )),
        ..Default::default()
    }));
    if !is_reverse {
        shapes.push(Shape::Text(TextShape {
            x: mid.x,
            y: mid.y - 33.0,
            text: link.id.clone(),
            font_size: Some(9.0),
            fill: Some("#334155".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
}

fn draw_node(
    shapes: &mut Vec<Shape>,
    id: &str,
    kind: &str,
    coords: &HashMap<String, Point>,
    sample: &NetworkTimeSample,
    node_stats: &HashMap<String, NodeStat>,
    top_bottleneck_id: &Option<String>,
) {
    let p = match coords.get(id) {
        Some(p) => *p,
        None => return,
    };
    let st = node_stats.get(id);
    let q = sample
        .node_queues
        .get(id)
        .copied()
        .or_else(|| st.map(|s| s.final_queue))
        .unwrap_or(0.0);
    let is_top = Some(id.to_string()) == *top_bottleneck_id;
    let fill = if kind == "host" {
        "#0ea5e9"
    } else if kind == "switch" {
        "#7c3aed"
    } else {
        "#0f766e"
    };
    shapes.push(Shape::Circle(CircleShape {
        x: p.x,
        y: p.y,
        r: if is_top { 28.0 } else { 24.0 },
        fill: fill.to_string(),
        stroke: Some(if is_top { "#dc2626".to_string() } else { "#0f172a".to_string() }),
        stroke_width: Some(if is_top { 4.0 } else { 2.0 }),
        title: Some(format!(
            "{} ({}) queue={}, dropped={}",
            id,
            kind,
            js_num(q),
            js_num(st.map(|s| s.dropped_packets).unwrap_or(0.0))
        )),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: p.x,
        y: p.y + 4.0,
        text: short_node_label(id),
        font_size: Some(10.0),
        fill: Some("#ffffff".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: p.x,
        y: p.y + 42.0,
        text: id.to_string(),
        font_size: Some(10.0),
        fill: Some("#0f172a".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    let q_h = 54.0_f64.min(q * 2.0);
    shapes.push(Shape::Rect(RectShape {
        x: p.x + 30.0,
        y: p.y + 24.0 - q_h,
        w: 10.0,
        h: q_h,
        fill: queue_color(q, st.map(|s| s.queue_limit_packets).unwrap_or(1.0)),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(2.0),
        ..Default::default()
    }));
}

fn draw_packet_if_active(
    shapes: &mut Vec<Shape>,
    pkt: &NetworkPacketSnapshot,
    t_ms: f64,
    coords: &HashMap<String, Point>,
) {
    let end = match pkt.delivered_at_ms.or(pkt.dropped_at_ms) {
        Some(e) => e,
        None => return,
    };
    if t_ms < pkt.created_at_ms || t_ms > end {
        return;
    }
    if pkt.hops.len() < 2 {
        return;
    }
    let progress = (((t_ms - pkt.created_at_ms) / (end - pkt.created_at_ms).max(1.0))).clamp(0.0, 0.999);
    let seg_float = progress * (pkt.hops.len() as f64 - 1.0);
    let seg = ((pkt.hops.len() - 2) as f64).min(seg_float.floor()) as usize;
    let local = seg_float - seg as f64;
    let (a, b) = match (coords.get(&pkt.hops[seg]), coords.get(&pkt.hops[seg + 1])) {
        (Some(a), Some(b)) => (*a, *b),
        _ => return,
    };
    let x = a.x + (b.x - a.x) * local;
    let y = a.y + (b.y - a.y) * local;
    shapes.push(Shape::Circle(CircleShape {
        x,
        y,
        r: if pkt.protocol == NetworkProtocol::Http { 4.5 } else { 3.7 },
        fill: protocol_color(pkt.protocol).to_string(),
        stroke: Some("#ffffff".to_string()),
        stroke_width: Some(1.0),
        opacity: Some(if pkt.dropped_at_ms.is_some() { 0.55 } else { 0.9 }),
        title: Some(format!("packet {} {} {}", pkt.packet_id, protocol_label(pkt.protocol), pkt.flow_id)),
        ..Default::default()
    }));
}

fn protocol_label(p: NetworkProtocol) -> &'static str {
    match p {
        NetworkProtocol::Raw => "raw",
        NetworkProtocol::Tcp => "tcp",
        NetworkProtocol::Udp => "udp",
        NetworkProtocol::Http => "http",
    }
}

fn draw_metrics_panel(shapes: &mut Vec<Shape>, result: &ComputerNetworkResult, sample: &NetworkTimeSample) {
    shapes.push(Shape::Rect(RectShape {
        x: PANEL_X,
        y: PANEL_Y,
        w: PANEL_W,
        h: 145.0,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: PANEL_X + 14.0,
        y: PANEL_Y + 24.0,
        text: "Metrics".to_string(),
        font_size: Some(16.0),
        fill: Some("#f8fafc".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    let rows = [
        format!("offered {} Mbps", to_fixed(result.offered_load_mbps, 2)),
        format!("wire {} Mbps", to_fixed(result.throughput_mbps, 2)),
        format!("goodput {} Mbps", to_fixed(result.goodput_mbps, 2)),
        format!("delivery {}%", to_fixed(result.delivery_ratio * 100.0, 1)),
        format!("active now {} / max {}", js_num(sample.active_packets), js_num(result.max_active_packets)),
    ];
    for (i, row) in rows.iter().enumerate() {
        shapes.push(Shape::Text(TextShape {
            x: PANEL_X + 16.0,
            y: PANEL_Y + 52.0 + i as f64 * 18.0,
            text: row.clone(),
            font_size: Some(12.0),
            fill: Some("#cbd5e1".to_string()),
            ..Default::default()
        }));
    }

    shapes.push(Shape::Rect(RectShape {
        x: PANEL_X,
        y: PANEL_Y + 160.0,
        w: PANEL_W,
        h: 100.0,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: PANEL_X + 14.0,
        y: PANEL_Y + 183.0,
        text: "Top bottlenecks".to_string(),
        font_size: Some(14.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for i in 0..result.bottlenecks.len().min(3) {
        let b = &result.bottlenecks[i];
        shapes.push(Shape::Text(TextShape {
            x: PANEL_X + 16.0,
            y: PANEL_Y + 207.0 + i as f64 * 18.0,
            text: format!("{}. {}:{} {}", i + 1, b.kind, b.id, b.reason),
            font_size: Some(11.0),
            fill: Some(if i == 0 { "#dc2626".to_string() } else { "#334155".to_string() }),
            ..Default::default()
        }));
    }
}

fn draw_fanout_policy_panel(shapes: &mut Vec<Shape>, tick: usize) {
    let x = PANEL_X;
    let y = PANEL_Y + 280.0;
    shapes.push(Shape::Rect(RectShape {
        x,
        y,
        w: PANEL_W,
        h: 210.0,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: x + 14.0,
        y: y + 24.0,
        text: "Fan-out order / bias".to_string(),
        font_size: Some(14.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: x + 14.0,
        y: y + 43.0,
        text: "Competitive out-connections; queues stay FIFO.".to_string(),
        font_size: Some(10.0),
        fill: Some("#64748b".to_string()),
        ..Default::default()
    }));

    let random_pick = ["B", "C", "A", "B", "A", "C"][tick % 6];
    let rr_pick = ["A", "B", "C"][tick % 3];
    let rows: [(&str, &str, &str, &str); 3] = [
        ("random", "shuffle each entity", random_pick, "#2563eb"),
        ("round-robin", "rotate declared order", rr_pick, "#16a34a"),
        ("ordered", "priority / bias", "A", "#dc2626"),
    ];
    for (r, (policy, desc, pick, color)) in rows.iter().enumerate() {
        let yy = y + 72.0 + r as f64 * 44.0;
        shapes.push(Shape::Text(TextShape {
            x: x + 14.0,
            y: yy,
            text: (*policy).to_string(),
            font_size: Some(12.0),
            fill: Some("#0f172a".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + 14.0,
            y: yy + 15.0,
            text: (*desc).to_string(),
            font_size: Some(9.0),
            fill: Some("#64748b".to_string()),
            ..Default::default()
        }));
        let labels = ["A", "B", "C"];
        for (i, label) in labels.iter().enumerate() {
            let lx = x + 140.0 + i as f64 * 36.0;
            let active = pick == label;
            shapes.push(Shape::Circle(CircleShape {
                x: lx,
                y: yy - 4.0,
                r: if active { 13.0 } else { 10.0 },
                fill: if active { (*color).to_string() } else { "#e2e8f0".to_string() },
                stroke: Some(if active { "#0f172a".to_string() } else { "#94a3b8".to_string() }),
                stroke_width: Some(1.2),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: lx,
                y: yy,
                text: (*label).to_string(),
                font_size: Some(10.0),
                fill: Some(if active { "#ffffff".to_string() } else { "#334155".to_string() }),
                anchor: Some(Anchor::Middle),
                font_weight: Some(FontWeight::Bold),
                ..Default::default()
            }));
        }
    }
}

fn draw_legend(shapes: &mut Vec<Shape>) {
    let x = NET_X + 16.0;
    let y = NET_Y + NET_H - 68.0;
    shapes.push(Shape::Rect(RectShape {
        x,
        y,
        w: 250.0,
        h: 50.0,
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        rx: Some(5.0),
        ..Default::default()
    }));
    let entries: [(NetworkProtocol, &str); 4] = [
        (NetworkProtocol::Http, "HTTP"),
        (NetworkProtocol::Tcp, "TCP"),
        (NetworkProtocol::Udp, "UDP"),
        (NetworkProtocol::Raw, "raw"),
    ];
    for (i, (p, label)) in entries.iter().enumerate() {
        let xx = x + 18.0 + i as f64 * 58.0;
        shapes.push(Shape::Circle(CircleShape {
            x: xx,
            y: y + 22.0,
            r: 5.0,
            fill: protocol_color(*p).to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: xx + 10.0,
            y: y + 26.0,
            text: (*label).to_string(),
            font_size: Some(10.0),
            fill: Some("#334155".to_string()),
            ..Default::default()
        }));
    }
}

fn build_computer_network_charts(result: &ComputerNetworkResult) -> Vec<ChartSpec> {
    let t: Vec<f64> = result.time_series.iter().map(|s| s.t_ms).collect();
    let flow_colors = [
        protocol_color(NetworkProtocol::Http).to_string(),
        protocol_color(NetworkProtocol::Tcp).to_string(),
        protocol_color(NetworkProtocol::Udp).to_string(),
        "#7c3aed".to_string(),
    ];
    vec![
        ChartSpec {
            x: 40.0,
            y: 520.0,
            w: 360.0,
            h: 200.0,
            title: Some("Traffic buildup".to_string()),
            y_min: Some(0.0),
            series: vec![
                ChartSeries { label: "active".to_string(), color: "#2563eb".to_string(), t: t.clone(), y: result.time_series.iter().map(|s| s.active_packets).collect() },
                ChartSeries { label: "dropped".to_string(), color: "#dc2626".to_string(), t: t.clone(), y: result.time_series.iter().map(|s| s.dropped_packets).collect() },
                ChartSeries { label: "delivered".to_string(), color: "#16a34a".to_string(), t: t.clone(), y: result.time_series.iter().map(|s| s.delivered_packets).collect() },
            ],
            ..Default::default()
        },
        ChartSpec {
            x: 430.0,
            y: 520.0,
            w: 360.0,
            h: 200.0,
            title: Some("Top link utilization".to_string()),
            y_min: Some(0.0),
            y_max: Some(1.0),
            series: top_utilization_series(result),
            ..Default::default()
        },
        ChartSpec {
            x: 820.0,
            y: 520.0,
            w: 330.0,
            h: 200.0,
            title: Some("Flow goodput (Mbps)".to_string()),
            y_min: Some(0.0),
            series: result
                .flow_stats
                .iter()
                .enumerate()
                .map(|(i, f)| ChartSeries {
                    label: f.id.clone(),
                    color: flow_colors[i % 4].clone(),
                    t: vec![0.0, result.total_simulated_ms],
                    y: vec![f.goodput_mbps, f.goodput_mbps],
                })
                .collect(),
            ..Default::default()
        },
    ]
}

fn top_utilization_series(result: &ComputerNetworkResult) -> Vec<ChartSeries> {
    let mut top_links: Vec<&LinkStat> = result
        .link_stats
        .iter()
        .filter(|l| l.delivered_packets > 0.0 || l.dropped_packets > 0.0)
        .collect();
    top_links.sort_by(|a, b| b.utilization.partial_cmp(&a.utilization).unwrap_or(std::cmp::Ordering::Equal));
    top_links.truncate(3);
    let colors = ["#dc2626", "#f97316", "#2563eb"];
    top_links
        .iter()
        .enumerate()
        .map(|(i, l)| ChartSeries {
            label: l.id.clone(),
            color: colors[i].to_string(),
            t: result.time_series.iter().map(|s| s.t_ms).collect(),
            y: result
                .time_series
                .iter()
                .map(|s| s.link_utilization.get(&l.id).copied().unwrap_or(l.utilization))
                .collect(),
        })
        .collect()
}

fn layout_nodes(problem: &ComputerNetworkProblem) -> HashMap<String, Point> {
    let ids: Vec<String> = problem.nodes.iter().map(|n| n.id.clone()).collect();
    let preset: HashMap<&str, Point> = [
        ("web-client", Point { x: 145.0, y: 165.0 }),
        ("telemetry-client", Point { x: 145.0, y: 355.0 }),
        ("edge", Point { x: 370.0, y: 260.0 }),
        ("wan-router", Point { x: 590.0, y: 260.0 }),
        ("api-server", Point { x: 780.0, y: 260.0 }),
        ("client-a", Point { x: 145.0, y: 165.0 }),
        ("client-b", Point { x: 145.0, y: 355.0 }),
        ("edge-1", Point { x: 370.0, y: 260.0 }),
        ("core-1", Point { x: 590.0, y: 260.0 }),
        ("server", Point { x: 780.0, y: 260.0 }),
    ]
    .into_iter()
    .collect();

    let mut out: HashMap<String, Point> = HashMap::new();
    if ids.iter().all(|id| preset.contains_key(id.as_str())) {
        for id in &ids {
            out.insert(id.clone(), preset[id.as_str()]);
        }
        return out;
    }

    let cx = NET_X + NET_W / 2.0;
    let cy = NET_Y + NET_H / 2.0;
    let r = NET_W.min(NET_H) * 0.36;
    for (i, id) in ids.iter().enumerate() {
        let a = -std::f64::consts::PI / 2.0
            + 2.0 * std::f64::consts::PI * i as f64 / (ids.len().max(1)) as f64;
        out.insert(id.clone(), Point { x: cx + r * a.cos(), y: cy + r * a.sin() });
    }
    out
}

fn normalize_links(links: &[NetworkLinkSpec]) -> Vec<NetworkLinkSpec> {
    let mut out: Vec<NetworkLinkSpec> = Vec::new();
    let mut ids: HashSet<String> = HashSet::new();
    for link in links {
        out.push(NetworkLinkSpec { bidirectional: false, ..link.clone() });
        ids.insert(link.id.clone());
        if !link.bidirectional {
            continue;
        }
        let mut reverse_id = format!("{}:rev", link.id);
        let mut i = 2;
        while ids.contains(&reverse_id) {
            reverse_id = format!("{}:rev{}", link.id, i);
            i += 1;
        }
        ids.insert(reverse_id.clone());
        out.push(NetworkLinkSpec {
            id: reverse_id,
            from: link.to.clone(),
            to: link.from.clone(),
            bidirectional: false,
        });
    }
    out
}

fn draw_arrow(shapes: &mut Vec<Shape>, a: Point, b: Point, color: &str) {
    let ang = (b.y - a.y).atan2(b.x - a.x);
    let tip = Point { x: b.x - 28.0 * ang.cos(), y: b.y - 28.0 * ang.sin() };
    let left = Point {
        x: tip.x - 9.0 * (ang - std::f64::consts::PI / 6.0).cos(),
        y: tip.y - 9.0 * (ang - std::f64::consts::PI / 6.0).sin(),
    };
    let right = Point {
        x: tip.x - 9.0 * (ang + std::f64::consts::PI / 6.0).cos(),
        y: tip.y - 9.0 * (ang + std::f64::consts::PI / 6.0).sin(),
    };
    shapes.push(Shape::Path(PathShape {
        d: format!(
            "M {} {} L {} {} L {} {} Z",
            js_num(tip.x),
            js_num(tip.y),
            js_num(left.x),
            js_num(left.y),
            js_num(right.x),
            js_num(right.y)
        ),
        fill: Some(color.to_string()),
        stroke: Some(color.to_string()),
        opacity: Some(0.9),
        ..Default::default()
    }));
}

fn queue_color(q: f64, cap: f64) -> String {
    let p = q / cap.max(1.0);
    if p > 0.85 {
        "#dc2626".to_string()
    } else if p > 0.45 {
        "#f97316".to_string()
    } else if q > 0.0 {
        "#eab308".to_string()
    } else {
        "#e2e8f0".to_string()
    }
}

fn short_node_label(id: &str) -> String {
    let initials: String = id
        .split(['-', '_'])
        .map(|s| s.chars().next().map(|c| c.to_ascii_uppercase().to_string()).unwrap_or_default())
        .collect();
    let sliced: String = initials.chars().take(3).collect();
    if !sliced.is_empty() {
        sliced
    } else {
        id.chars().take(2).collect::<String>().to_uppercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_label_takes_initials() {
        assert_eq!(short_node_label("web-client"), "WC");
        assert_eq!(short_node_label("a-b-c-d"), "ABC");
        assert_eq!(short_node_label("server"), "S");
    }

    #[test]
    fn normalize_links_adds_reverse_for_bidirectional() {
        let links = vec![NetworkLinkSpec {
            id: "l1".to_string(),
            from: "a".to_string(),
            to: "b".to_string(),
            bidirectional: true,
        }];
        let out = normalize_links(&links);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].id, "l1:rev");
        assert_eq!(out[1].from, "b");
        assert_eq!(out[1].to, "a");
        assert!(!out[0].bidirectional);
    }

    #[test]
    fn queue_color_thresholds() {
        assert_eq!(queue_color(9.0, 10.0), "#dc2626");
        assert_eq!(queue_color(5.0, 10.0), "#f97316");
        assert_eq!(queue_color(1.0, 10.0), "#eab308");
        assert_eq!(queue_color(0.0, 10.0), "#e2e8f0");
    }
}
