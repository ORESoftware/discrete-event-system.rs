//! Port of `src/des/animation/scenes/calculus-scene.ts`.
//!
//! Builds frames + charts for the calculus (1-D field / 2-D Poisson)
//! animation: a 1-D PDE field `u(x, t)` drawn as a coloured strip of vertical
//! bars plus a peak-amplitude time series, and a static false-colour image of a
//! converged 2-D Poisson solution.
//!
//! Pure data builders — every coordinate/value is `f64`; `valueToColor`
//! composes an `#rrggbb` string via `format!`.

#![allow(dead_code)]

use crate::des::animation::types::{
    to_fixed, Anchor, ChartSeries, ChartSpec, FontWeight, FrameParts, LineShape, RectShape, Shape,
    TextShape,
};

pub const STAGE_W: f64 = 1000.0;
pub const STAGE_H: f64 = 720.0;
const STRIP_X: f64 = 30.0;
const STRIP_Y: f64 = 60.0;
const STRIP_W: f64 = 720.0;
const STRIP_H: f64 = 360.0;
const METRIC_X: f64 = 770.0;
const METRIC_Y: f64 = 60.0;
const METRIC_W: f64 = 200.0;
const METRIC_H: f64 = 360.0;
const CHART_X: f64 = 30.0;
const CHART_Y: f64 = 460.0;
const CHART_W: f64 = 940.0;
const CHART_H: f64 = 240.0;

fn hex2(n: f64) -> String {
    // JS `Math.round(x).toString(16).padStart(2, '0')` for x in [0, 255].
    let v = n.round().clamp(0.0, 255.0) as u32;
    format!("{v:02x}")
}

/// Map a value `v ∈ [-vMax, vMax]` to a hex colour (blue → white → red).
fn value_to_color(v: f64, v_max: f64) -> String {
    let t = (v / v_max.max(1e-12)).clamp(-1.0, 1.0);
    if t >= 0.0 {
        let r = 255.0;
        let g = 255.0 * (1.0 - t);
        let b = 255.0 * (1.0 - t);
        format!("#{}{}{}", hex2(r), hex2(g), hex2(b))
    } else {
        let r = 255.0 * (1.0 + t);
        let g = 255.0 * (1.0 + t);
        let b = 255.0;
        format!("#{}{}{}", hex2(r), hex2(g), hex2(b))
    }
}

/// A time-ordered trace of field snapshots for [`build_field1d_chart`].
#[derive(Clone, Debug, Default)]
pub struct Field1DTrace {
    pub t: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

pub fn build_field1d_frame(
    t: f64,
    tick: f64,
    values: &[f64],
    xs: &[f64],
    v_max: f64,
    scheme: &str,
    family: &str,
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let n = values.len();

    // Border + title.
    shapes.push(Shape::Rect(RectShape {
        x: STRIP_X - 6.0,
        y: STRIP_Y - 6.0,
        w: STRIP_W + 12.0,
        h: STRIP_H + 12.0,
        fill: "#fff".to_string(),
        stroke: Some("#bbb".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: STRIP_X,
        y: STRIP_Y - 14.0,
        text: format!("Field u(x, t),  family={family},  scheme={scheme},  N={n} stations"),
        font_size: Some(13.0),
        fill: Some("#333".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    // Zero line.
    let y_mid = STRIP_Y + STRIP_H / 2.0;
    shapes.push(Shape::Line(LineShape {
        x1: STRIP_X,
        y1: y_mid,
        x2: STRIP_X + STRIP_W,
        y2: y_mid,
        stroke: "#bbb".to_string(),
        stroke_width: Some(0.6),
        ..Default::default()
    }));

    // Bars: each station is a vertical bar; colour by signed magnitude.
    let cell_w = STRIP_W / n as f64;
    let mut peak = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let v = values[i];
        if v.abs() > peak.abs() {
            peak = v;
        }
        sum_sq += v * v;
        let h = (v.abs() / v_max) * (STRIP_H / 2.0 - 4.0);
        let x = STRIP_X + i as f64 * cell_w + cell_w * 0.05;
        let w = cell_w * 0.9;
        let y_top = if v >= 0.0 { y_mid - h } else { y_mid };
        shapes.push(Shape::Rect(RectShape {
            x,
            y: y_top,
            w,
            h: h.max(0.1),
            fill: value_to_color(v, v_max),
            ..Default::default()
        }));
    }

    // x-axis ticks (5).
    for k in 0..=4 {
        let xx = STRIP_X + (STRIP_W * k as f64) / 4.0;
        let idx = ((n as f64 - 1.0) * k as f64 / 4.0).round() as usize;
        let idx = idx.min(n - 1);
        let xv = xs[idx];
        shapes.push(Shape::Text(TextShape {
            x: xx,
            y: STRIP_Y + STRIP_H + 14.0,
            text: to_fixed(xv, 2),
            font_size: Some(10.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    // Metrics panel.
    shapes.push(Shape::Rect(RectShape {
        x: METRIC_X,
        y: METRIC_Y,
        w: METRIC_W,
        h: METRIC_H,
        fill: "#fafafa".to_string(),
        stroke: Some("#ddd".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 12.0,
        y: METRIC_Y + 22.0,
        text: format!("t = {}", to_fixed(t, 4)),
        font_size: Some(14.0),
        fill: Some("#222".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 12.0,
        y: METRIC_Y + 40.0,
        text: format!("tick {}", crate::des::animation::types::js_num(tick)),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        ..Default::default()
    }));
    let stats: [(String, String); 4] = [
        ("peak |u|".to_string(), to_fixed(peak.abs(), 4)),
        (
            "L2 norm".to_string(),
            to_fixed((sum_sq / n as f64).sqrt(), 4),
        ),
        ("scheme".to_string(), scheme.to_string()),
        ("family".to_string(), family.to_string()),
    ];
    for (i, (label, value)) in stats.iter().enumerate() {
        let y = METRIC_Y + 80.0 + i as f64 * 22.0;
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 12.0,
            y,
            text: label.clone(),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + METRIC_W - 12.0,
            y,
            text: value.clone(),
            font_size: Some(11.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
    FrameParts::new(shapes)
}

pub fn build_field1d_chart(trace: &Field1DTrace) -> ChartSpec {
    // Track peak amplitude over time as the headline metric.
    let mut series: Vec<f64> = Vec::new();
    for v in &trace.values {
        let mut m = 0.0_f64;
        for &x in v {
            if x.abs() > m {
                m = x.abs();
            }
        }
        series.push(m);
    }
    ChartSpec {
        x: CHART_X,
        y: CHART_Y,
        w: CHART_W,
        h: CHART_H,
        title: Some("peak |u(x, t)| over time".to_string()),
        y_label: Some("peak |u|".to_string()),
        series: vec![ChartSeries {
            label: "peak".to_string(),
            color: "#ef4444".to_string(),
            t: trace.t.clone(),
            y: series,
        }],
        ..Default::default()
    }
}

// -----------------------------------------------------------------------------
// 2-D Poisson scene: a single static frame showing the converged u(x, y).
// -----------------------------------------------------------------------------
pub const POISSON_W: f64 = 720.0;
pub const POISSON_H: f64 = 720.0;

pub fn build_poisson_frame(u: &[f64], nx: usize, ny: usize, v_max: f64) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let pad = 40.0;
    let w = POISSON_W - 2.0 * pad;
    let h = POISSON_H - 2.0 * pad;
    let cell_w = w / nx as f64;
    let cell_h = h / ny as f64;
    shapes.push(Shape::Rect(RectShape {
        x: pad - 4.0,
        y: pad - 4.0,
        w: w + 8.0,
        h: h + 8.0,
        fill: "#fff".to_string(),
        stroke: Some("#bbb".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: pad,
        y: pad - 12.0,
        text: format!("2-D Poisson  \u{2207}\u{00b2}u = -\u{03c1}   grid {nx}\u{00d7}{ny}"),
        font_size: Some(13.0),
        fill: Some("#333".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for j in 0..ny {
        for i in 0..nx {
            let v = u[j * nx + i];
            shapes.push(Shape::Rect(RectShape {
                x: pad + i as f64 * cell_w,
                y: pad + (ny - 1 - j) as f64 * cell_h,
                w: cell_w + 0.5,
                h: cell_h + 0.5,
                fill: value_to_color(v, v_max),
                ..Default::default()
            }));
        }
    }
    FrameParts::new(shapes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_to_color_endpoints() {
        assert_eq!(value_to_color(0.0, 1.0), "#ffffff");
        assert_eq!(value_to_color(1.0, 1.0), "#ff0000");
        assert_eq!(value_to_color(-1.0, 1.0), "#0000ff");
    }

    #[test]
    fn field_frame_has_border_title_and_bars() {
        let vals = [0.0, 0.5, -0.5, 1.0];
        let xs = [0.0, 1.0, 2.0, 3.0];
        let fp = build_field1d_frame(0.25, 5.0, &vals, &xs, 1.0, "ftcs", "heat");
        // border + title + zero line + 4 bars + 5 ticks + metric panel + 2
        // headers + 4*2 stat rows = 1+1+1+4+5+1+2+8 = 23.
        assert_eq!(fp.shapes.len(), 23);
    }

    #[test]
    fn poisson_frame_cell_count() {
        let u = vec![0.0; 9];
        let fp = build_poisson_frame(&u, 3, 3, 1.0);
        // border + title + 9 cells.
        assert_eq!(fp.shapes.len(), 11);
    }
}
