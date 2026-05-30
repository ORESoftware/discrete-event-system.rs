//! Port of `src/des/animation/scenes/newsvendor-scene.ts`.
//!
//! Builds frames + the two chart panels for the newsvendor / inventory
//! animation: a left panel of stacked inventory-flow bars with a demand line, a
//! right metrics panel, and two time-series charts beneath.
//!
//! Pure data builder; the `COLORS` object literal becomes a module of `&str`
//! consts.

#![allow(dead_code)]

use crate::des::animation::types::{
    to_fixed, Anchor, ChartSeries, ChartSpec, FontWeight, FrameParts, LineShape, RectShape, Shape,
    TextShape,
};

pub const STAGE_W: f64 = 1100.0;
pub const STAGE_H: f64 = 680.0;

mod colors {
    pub const INVENTORY: &str = "#3b82f6";
    pub const ORDERED: &str = "#10b981";
    pub const SOLD: &str = "#22c55e";
    pub const LEFTOVER: &str = "#f59e0b";
    pub const LOST: &str = "#ef4444";
    pub const DEMAND: &str = "#a855f7";
    pub const PROFIT_POS: &str = "#16a34a";
    pub const PROFIT_NEG: &str = "#dc2626";
}

#[derive(Clone, Debug, Default)]
pub struct NewsvendorFrameData {
    pub day: f64,
    /// Starting inventory before order.
    pub start_inv: f64,
    /// Quantity ordered this period.
    pub ordered: f64,
    /// Realised demand.
    pub demand: f64,
    /// Units sold (= min(start+order, demand)).
    pub sold: f64,
    /// Leftover at end of period.
    pub leftover: f64,
    /// Unmet demand.
    pub lost: f64,
    /// Profit / reward this period.
    pub profit: f64,
    /// Cumulative profit.
    pub cum_profit: f64,
    /// Maximum quantity for axis scaling.
    pub q_scale: f64,
    /// Optional policy label (e.g., "(s,S) = (14, 47)").
    pub policy: Option<String>,
}

pub fn build_newsvendor_frame(d: &NewsvendorFrameData) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let left = 40.0;
    let top = 40.0;
    let panel_w = 700.0;
    let panel_h = 360.0;
    let margin = 40.0;
    let inner_w = panel_w - margin * 2.0;

    // Left panel: stacked bars for the day's inventory flow.
    shapes.push(Shape::Rect(RectShape {
        x: left,
        y: top,
        w: panel_w,
        h: panel_h,
        fill: "#fafafa".to_string(),
        stroke: Some("#ccc".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: left + 16.0,
        y: top + 24.0,
        text: format!("Day {}", crate::des::animation::types::js_num(d.day)),
        font_size: Some(16.0),
        fill: Some("#222".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    if let Some(policy) = &d.policy {
        shapes.push(Shape::Text(TextShape {
            x: left + panel_w - 16.0,
            y: top + 24.0,
            text: policy.clone(),
            font_size: Some(12.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::End),
            ..Default::default()
        }));
    }

    // Bar geometry: four stacked bars at heights proportional to qScale.
    let bar_top = top + 60.0;
    let bar_h = panel_h - 120.0;
    let bar_w = (inner_w - 60.0) / 4.0;
    let x_centers: Vec<f64> = (0..4)
        .map(|i| left + margin + bar_w / 2.0 + i as f64 * (bar_w + 20.0))
        .collect();
    let labels = ["start inv", "ordered", "sold", "leftover"];
    let values = [d.start_inv, d.ordered, d.sold, d.leftover];
    let bar_colors = [colors::INVENTORY, colors::ORDERED, colors::SOLD, colors::LEFTOVER];

    for i in 0..4 {
        let v = values[i];
        let h = (v / d.q_scale.max(1.0)) * bar_h;
        shapes.push(Shape::Rect(RectShape {
            x: x_centers[i] - bar_w / 2.0,
            y: bar_top + bar_h - h,
            w: bar_w,
            h,
            fill: bar_colors[i].to_string(),
            stroke: Some("#333".to_string()),
            stroke_width: Some(0.6),
            rx: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x_centers[i],
            y: bar_top + bar_h - h - 6.0,
            text: crate::des::animation::types::js_num(v),
            font_size: Some(12.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x_centers[i],
            y: bar_top + bar_h + 18.0,
            text: labels[i].to_string(),
            font_size: Some(11.0),
            fill: Some("#555".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
    shapes.push(Shape::Line(LineShape {
        x1: left + margin - 5.0,
        y1: bar_top + bar_h,
        x2: left + panel_w - margin + 5.0,
        y2: bar_top + bar_h,
        stroke: "#999".to_string(),
        stroke_width: Some(1.0),
        ..Default::default()
    }));

    // Demand line over the whole bar area.
    let dem_h = (d.demand / d.q_scale.max(1.0)) * bar_h;
    shapes.push(Shape::Line(LineShape {
        x1: left + margin - 5.0,
        y1: bar_top + bar_h - dem_h,
        x2: left + panel_w - margin + 5.0,
        y2: bar_top + bar_h - dem_h,
        stroke: colors::DEMAND.to_string(),
        stroke_width: Some(1.5),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: left + panel_w - margin + 8.0,
        y: bar_top + bar_h - dem_h + 4.0,
        text: format!("D={}", crate::des::animation::types::js_num(d.demand)),
        font_size: Some(11.0),
        fill: Some(colors::DEMAND.to_string()),
        ..Default::default()
    }));

    // Right panel: metrics.
    let rx = left + panel_w + 24.0;
    let ry = top;
    let rw = STAGE_W - rx - 40.0;
    let rh = panel_h;
    shapes.push(Shape::Rect(RectShape {
        x: rx,
        y: ry,
        w: rw,
        h: rh,
        fill: "#fff".to_string(),
        stroke: Some("#ccc".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: rx + 14.0,
        y: ry + 24.0,
        text: "Metrics".to_string(),
        font_size: Some(14.0),
        fill: Some("#222".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    let rows: Vec<Row> = vec![
        Row::metric("demand", crate::des::animation::types::js_num(d.demand), colors::DEMAND),
        Row::metric("sold", crate::des::animation::types::js_num(d.sold), colors::SOLD),
        Row::metric("leftover", crate::des::animation::types::js_num(d.leftover), colors::LEFTOVER),
        Row::metric("lost", crate::des::animation::types::js_num(d.lost), colors::LOST),
        Row::Separator,
        Row::metric(
            "profit",
            to_fixed(d.profit, 2),
            if d.profit >= 0.0 { colors::PROFIT_POS } else { colors::PROFIT_NEG },
        ),
        Row::metric(
            "cumulative",
            to_fixed(d.cum_profit, 2),
            if d.cum_profit >= 0.0 { colors::PROFIT_POS } else { colors::PROFIT_NEG },
        ),
    ];
    let mut yy = ry + 56.0;
    for row in &rows {
        match row {
            Row::Separator => {
                yy += 12.0;
                continue;
            }
            Row::Metric { label, val, color } => {
                shapes.push(Shape::Text(TextShape {
                    x: rx + 14.0,
                    y: yy,
                    text: label.clone(),
                    font_size: Some(12.0),
                    fill: Some("#555".to_string()),
                    ..Default::default()
                }));
                shapes.push(Shape::Text(TextShape {
                    x: rx + rw - 14.0,
                    y: yy,
                    text: val.clone(),
                    font_size: Some(13.0),
                    fill: Some(color.clone()),
                    anchor: Some(Anchor::End),
                    font_weight: Some(FontWeight::Bold),
                    ..Default::default()
                }));
                yy += 24.0;
            }
        }
    }

    let caption = format!(
        "day={}  start={}  order={}  D={}  sold={}  leftover={}  lost={}  profit={}  cum={}",
        crate::des::animation::types::js_num(d.day),
        crate::des::animation::types::js_num(d.start_inv),
        crate::des::animation::types::js_num(d.ordered),
        crate::des::animation::types::js_num(d.demand),
        crate::des::animation::types::js_num(d.sold),
        crate::des::animation::types::js_num(d.leftover),
        crate::des::animation::types::js_num(d.lost),
        to_fixed(d.profit, 2),
        to_fixed(d.cum_profit, 2)
    );
    FrameParts::with_caption(shapes, caption)
}

/// A metrics-panel row: a labelled value (with colour) or a vertical spacer.
enum Row {
    Metric { label: String, val: String, color: String },
    Separator,
}

impl Row {
    fn metric(label: &str, val: String, color: &str) -> Row {
        Row::Metric { label: label.to_string(), val, color: color.to_string() }
    }
}

/// A trace of inventory / profit over time for [`build_newsvendor_chart`].
#[derive(Clone, Debug, Default)]
pub struct NewsvendorTrace {
    pub t: Vec<f64>,
    pub inv: Vec<f64>,
    pub profit: Vec<f64>,
    pub cum_profit: Vec<f64>,
}

fn max_or(values: &[f64], floor: f64) -> f64 {
    // JS `Math.max(floor, Math.max(...values))` (empty -> Math.max() === -Inf).
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    floor.max(m)
}

fn min_or(values: &[f64], ceil: f64) -> f64 {
    // JS `Math.min(ceil, Math.min(...values))` (empty -> Math.min() === +Inf).
    let m = values.iter().copied().fold(f64::INFINITY, f64::min);
    ceil.min(m)
}

pub fn build_newsvendor_chart(trace: &NewsvendorTrace) -> Vec<ChartSpec> {
    let y_max_a = max_or(&trace.inv, 1.0);
    let y_max_b = max_or(&trace.cum_profit, 1.0);
    let y_min_b = min_or(&trace.cum_profit, 0.0);
    vec![
        ChartSpec {
            x: 40.0,
            y: 420.0,
            w: 510.0,
            h: 230.0,
            title: Some("inventory & per-period profit".to_string()),
            y_min: Some(-y_max_a),
            y_max: Some(y_max_a),
            series: vec![
                ChartSeries {
                    label: "inv (start)".to_string(),
                    color: colors::INVENTORY.to_string(),
                    t: trace.t.clone(),
                    y: trace.inv.clone(),
                },
                ChartSeries {
                    label: "profit".to_string(),
                    color: colors::PROFIT_POS.to_string(),
                    t: trace.t.clone(),
                    y: trace.profit.clone(),
                },
            ],
            ..Default::default()
        },
        ChartSpec {
            x: 570.0,
            y: 420.0,
            w: 510.0,
            h: 230.0,
            title: Some("cumulative profit".to_string()),
            y_min: Some(y_min_b),
            y_max: Some(y_max_b),
            series: vec![ChartSeries {
                label: "cumulative".to_string(),
                color: "#2563eb".to_string(),
                t: trace.t.clone(),
                y: trace.cum_profit.clone(),
            }],
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_caption_summarises_the_day() {
        let d = NewsvendorFrameData {
            day: 3.0,
            start_inv: 10.0,
            ordered: 20.0,
            demand: 18.0,
            sold: 18.0,
            leftover: 12.0,
            lost: 0.0,
            profit: 9.5,
            cum_profit: -2.25,
            q_scale: 50.0,
            policy: Some("(s,S)=(14,47)".to_string()),
        };
        let fp = build_newsvendor_frame(&d);
        let cap = fp.caption.unwrap();
        assert!(cap.contains("day=3  start=10  order=20"));
        assert!(cap.contains("profit=9.50  cum=-2.25"));
    }

    #[test]
    fn chart_axis_bounds() {
        let trace = NewsvendorTrace {
            t: vec![0.0, 1.0, 2.0],
            inv: vec![10.0, 30.0, 5.0],
            profit: vec![1.0, -2.0, 4.0],
            cum_profit: vec![1.0, -1.0, 3.0],
        };
        let charts = build_newsvendor_chart(&trace);
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].y_max, Some(30.0));
        assert_eq!(charts[0].y_min, Some(-30.0));
        assert_eq!(charts[1].y_min, Some(-1.0));
        assert_eq!(charts[1].y_max, Some(3.0));
    }
}
