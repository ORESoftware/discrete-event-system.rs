//! Port of `src/des/animation/scenes/contact-seir-scene.ts`.
//!
//! Builds frames + the compartment chart for the contact-network SEIR
//! animation: each person is a dot on a grid coloured by SEIR state (radius ∝
//! √contact-rate), with a metrics panel, a legend, and a time-series chart.
//!
//! Pure data builder; `layoutGrid` returns parallel arrays as a [`GridLayout`]
//! struct and `COLORS` becomes [`color`].

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, FontWeight, FrameParts, LineShape, RectShape,
    Shape, TextShape,
};

pub const STAGE_W: f64 = 1000.0;
pub const STAGE_H: f64 = 720.0;
const GRID_X: f64 = 30.0;
const GRID_Y: f64 = 40.0;
const GRID_W: f64 = 720.0;
const GRID_H: f64 = 420.0;
const METRIC_X: f64 = 770.0;
const METRIC_Y: f64 = 40.0;
const METRIC_W: f64 = 200.0;
const METRIC_H: f64 = 420.0;
const CHART_X: f64 = 30.0;
const CHART_Y: f64 = 480.0;
const CHART_W: f64 = 940.0;
const CHART_H: f64 = 220.0;

/// `'S' | 'E' | 'I' | 'R'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeirState {
    S,
    E,
    I,
    R,
}

/// `COLORS[state]`.
pub fn color(state: SeirState) -> &'static str {
    match state {
        SeirState::S => "#3b82f6", // blue
        SeirState::E => "#f59e0b", // amber
        SeirState::I => "#ef4444", // red
        SeirState::R => "#9ca3af", // gray
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PersonView {
    pub id: f64,
    pub state: SeirState,
    /// Contact rate (controls dot radius).
    pub c: f64,
}

/// Parallel `(x, y)` arrays from [`layout_grid`].
#[derive(Clone, Debug, Default)]
pub struct GridLayout {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Compute fixed `(x, y)` positions for `N` people on a regular grid. Returns
/// parallel arrays; idempotent — call once and reuse for every frame.
pub fn layout_grid(n: usize) -> GridLayout {
    let cols = (((n as f64) * GRID_W / GRID_H).sqrt()).ceil() as usize;
    let cols = cols.max(1);
    let _rows = ((n as f64) / cols as f64).ceil();
    let cell_w = GRID_W / cols as f64;
    let cell_h = GRID_H / _rows;
    let mut x = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let c = i % cols;
        let r = (i / cols) as f64;
        x[i] = GRID_X + (c as f64 + 0.5) * cell_w;
        y[i] = GRID_Y + (r + 0.5) * cell_h;
    }
    GridLayout { x, y }
}

#[allow(clippy::too_many_arguments)]
pub fn build_contact_frame(
    t: f64,
    tick: f64,
    people: &[PersonView],
    pos: &GridLayout,
    mean_c: f64,
    total_contacts: f64,
    total_transmissions: f64,
    kernel: &str,
) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let n = people.len();

    // Grid border + title.
    shapes.push(Shape::Rect(RectShape {
        x: GRID_X - 6.0,
        y: GRID_Y - 6.0,
        w: GRID_W + 12.0,
        h: GRID_H + 12.0,
        fill: "#fff".to_string(),
        stroke: Some("#bbb".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: GRID_X,
        y: GRID_Y - 14.0,
        text: format!("Population ({n} people, kernel={kernel})"),
        font_size: Some(13.0),
        fill: Some("#333".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    // Dots: radius proportional to per-person contact rate, capped.
    let base_r = GRID_W.min(GRID_H) / (n as f64).sqrt() / 4.0;
    let mut n_s = 0.0_f64;
    let mut n_e = 0.0_f64;
    let mut n_i = 0.0_f64;
    let mut n_r = 0.0_f64;
    for (i, p) in people.iter().enumerate() {
        match p.state {
            SeirState::S => n_s += 1.0,
            SeirState::E => n_e += 1.0,
            SeirState::I => n_i += 1.0,
            SeirState::R => n_r += 1.0,
        }
        let r = base_r * 2.5_f64.min(0.5_f64.max((p.c / mean_c.max(1e-9)).sqrt()));
        // No per-dot title — it would balloon the frames file by ~10×.
        shapes.push(Shape::Circle(crate::des::animation::types::CircleShape {
            x: pos.x[i],
            y: pos.y[i],
            r,
            fill: color(p.state).to_string(),
            stroke: Some("#fff".to_string()),
            stroke_width: Some(0.3),
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
        text: format!("t = {}", to_fixed(t, 2)),
        font_size: Some(14.0),
        fill: Some("#222".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 12.0,
        y: METRIC_Y + 40.0,
        text: format!("tick {}", js_num(tick)),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        ..Default::default()
    }));

    let lines: [(&str, String, &str); 4] = [
        ("S", js_num(n_s), color(SeirState::S)),
        ("E", js_num(n_e), color(SeirState::E)),
        ("I", js_num(n_i), color(SeirState::I)),
        ("R", js_num(n_r), color(SeirState::R)),
    ];
    for (i, (label, count, col)) in lines.iter().enumerate() {
        let y = METRIC_Y + 70.0 + i as f64 * 26.0;
        shapes.push(Shape::Rect(RectShape {
            x: METRIC_X + 12.0,
            y: y - 11.0,
            w: 14.0,
            h: 14.0,
            fill: (*col).to_string(),
            rx: Some(3.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 32.0,
            y,
            text: (*label).to_string(),
            font_size: Some(12.0),
            fill: Some("#333".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + METRIC_W - 12.0,
            y,
            text: count.clone(),
            font_size: Some(12.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
    let y_end = METRIC_Y + 70.0 + lines.len() as f64 * 26.0 + 6.0;
    shapes.push(Shape::Line(LineShape {
        x1: METRIC_X + 12.0,
        y1: y_end,
        x2: METRIC_X + METRIC_W - 12.0,
        y2: y_end,
        stroke: "#ddd".to_string(),
        stroke_width: Some(1.0),
        ..Default::default()
    }));

    let stats: [(String, String); 4] = [
        ("attack rate".to_string(), format!("{}%", to_fixed((1.0 - n_s / n as f64) * 100.0, 1))),
        ("contacts".to_string(), js_num(total_contacts)),
        ("transmissions".to_string(), js_num(total_transmissions)),
        ("kernel".to_string(), kernel.to_string()),
    ];
    for (i, (label, value)) in stats.iter().enumerate() {
        let y = y_end + 18.0 + i as f64 * 22.0;
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
            font_size: Some(12.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    // Legend.
    let leg_y = GRID_Y + GRID_H + 28.0;
    let mut lx = GRID_X;
    let legend: [(&str, &str); 4] = [
        ("S = susceptible", color(SeirState::S)),
        ("E = exposed", color(SeirState::E)),
        ("I = infectious", color(SeirState::I)),
        ("R = recovered", color(SeirState::R)),
    ];
    for (state, col) in legend {
        shapes.push(Shape::Rect(RectShape {
            x: lx,
            y: leg_y - 10.0,
            w: 14.0,
            h: 14.0,
            fill: col.to_string(),
            rx: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: lx + 20.0,
            y: leg_y,
            text: state.to_string(),
            font_size: Some(11.0),
            fill: Some("#555".to_string()),
            ..Default::default()
        }));
        lx += 150.0;
    }
    shapes.push(Shape::Text(TextShape {
        x: GRID_X + 700.0,
        y: leg_y,
        text: "dot radius \u{221d} \u{221a}(contact rate)".to_string(),
        font_size: Some(11.0),
        fill: Some("#888".to_string()),
        anchor: Some(Anchor::End),
        ..Default::default()
    }));

    let caption = format!(
        "tick={}  t={}  S={} E={} I={} R={}   attack={}%  transmissions={}",
        js_num(tick),
        to_fixed(t, 2),
        js_num(n_s),
        js_num(n_e),
        js_num(n_i),
        js_num(n_r),
        to_fixed((1.0 - n_s / n as f64) * 100.0, 1),
        js_num(total_transmissions)
    );
    FrameParts::with_caption(shapes, caption)
}

/// A trace of SEIR compartment populations over time.
#[derive(Clone, Debug, Default)]
pub struct ContactTrace {
    pub t: Vec<f64>,
    pub s: Vec<f64>,
    pub e: Vec<f64>,
    pub i: Vec<f64>,
    pub r: Vec<f64>,
}

pub fn build_contact_chart(trace: &ContactTrace, n: f64) -> ChartSpec {
    ChartSpec {
        x: CHART_X,
        y: CHART_Y,
        w: CHART_W,
        h: CHART_H,
        title: Some("Compartment populations over time".to_string()),
        y_min: Some(0.0),
        y_max: Some(n),
        series: vec![
            ChartSeries { label: "S".to_string(), color: color(SeirState::S).to_string(), t: trace.t.clone(), y: trace.s.clone() },
            ChartSeries { label: "E".to_string(), color: color(SeirState::E).to_string(), t: trace.t.clone(), y: trace.e.clone() },
            ChartSeries { label: "I".to_string(), color: color(SeirState::I).to_string(), t: trace.t.clone(), y: trace.i.clone() },
            ChartSeries { label: "R".to_string(), color: color(SeirState::R).to_string(), t: trace.t.clone(), y: trace.r.clone() },
        ],
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_grid_fills_n_positions() {
        let g = layout_grid(10);
        assert_eq!(g.x.len(), 10);
        assert_eq!(g.y.len(), 10);
        // All inside the grid rectangle.
        assert!(g.x.iter().all(|&x| x >= GRID_X && x <= GRID_X + GRID_W));
    }

    #[test]
    fn frame_caption_reports_compartments_and_attack_rate() {
        let people = vec![
            PersonView { id: 0.0, state: SeirState::S, c: 1.0 },
            PersonView { id: 1.0, state: SeirState::I, c: 2.0 },
        ];
        let pos = layout_grid(2);
        let fp = build_contact_frame(1.0, 4.0, &people, &pos, 1.5, 7.0, 2.0, "exponential");
        let cap = fp.caption.unwrap();
        assert!(cap.contains("S=1 E=0 I=1 R=0"));
        assert!(cap.contains("attack=50.0%"));
        assert!(cap.contains("transmissions=2"));
    }
}
