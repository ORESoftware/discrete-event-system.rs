//! Port of `src/des/animation/scenes/two-disease-scene.ts`.
//!
//! Builds frames + the global compartment-over-time chart for the two-disease
//! (co-infection) animation: six vertical population bars (top half) over a
//! line chart (bottom half).
//!
//! ## Rust shape
//!
//! The `keyof CompartmentCounts` keys `S/A/B/AB/R/D` are a closed set →
//! [`Compartment`] enum. `ORDER` is a `[Compartment; 6]`; `COLORS` becomes
//! [`color`]; `counts[k]` becomes [`CompartmentCounts::get`].

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, FontWeight, FrameParts, RectShape, Shape,
    TextShape,
};

/// The six compartments, in display order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compartment {
    S,
    A,
    B,
    AB,
    R,
    D,
}

impl Compartment {
    /// The single-letter label used in bars / captions.
    pub fn label(self) -> &'static str {
        match self {
            Compartment::S => "S",
            Compartment::A => "A",
            Compartment::B => "B",
            Compartment::AB => "AB",
            Compartment::R => "R",
            Compartment::D => "D",
        }
    }
}

/// `const ORDER: Array<keyof CompartmentCounts>`.
pub const ORDER: [Compartment; 6] = [
    Compartment::S,
    Compartment::A,
    Compartment::B,
    Compartment::AB,
    Compartment::R,
    Compartment::D,
];

/// `COLORS[k]` — per-compartment fill colour.
pub fn color(c: Compartment) -> &'static str {
    match c {
        Compartment::S => "#3b82f6",  // blue   = susceptible
        Compartment::A => "#f59e0b",  // amber  = disease A
        Compartment::B => "#10b981",  // emerald = disease B
        Compartment::AB => "#8b5cf6", // violet = co-infected
        Compartment::R => "#6b7280",  // gray   = recovered
        Compartment::D => "#ef4444",  // red    = dead
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompartmentCounts {
    pub s: f64,
    pub a: f64,
    pub b: f64,
    pub ab: f64,
    pub r: f64,
    pub d: f64,
}

impl CompartmentCounts {
    /// `counts[k]`.
    pub fn get(&self, c: Compartment) -> f64 {
        match c {
            Compartment::S => self.s,
            Compartment::A => self.a,
            Compartment::B => self.b,
            Compartment::AB => self.ab,
            Compartment::R => self.r,
            Compartment::D => self.d,
        }
    }
}

pub const STAGE_W: f64 = 900.0;
pub const STAGE_H: f64 = 640.0;
const BAR_AREA_X: f64 = 60.0;
const BAR_AREA_Y: f64 = 60.0;
const BAR_AREA_W: f64 = STAGE_W - 120.0;
const BAR_AREA_H: f64 = 220.0;
const CHART_X: f64 = 60.0;
const CHART_Y: f64 = 320.0;
const CHART_W: f64 = STAGE_W - 120.0;
const CHART_H: f64 = 280.0;

pub fn build_bars(counts: &CompartmentCounts, n: f64) -> Vec<Shape> {
    let mut shapes: Vec<Shape> = Vec::new();
    shapes.push(Shape::Text(TextShape {
        x: BAR_AREA_X,
        y: BAR_AREA_Y - 14.0,
        text: "Population by compartment".to_string(),
        font_size: Some(13.0),
        fill: Some("#333".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    let bw = BAR_AREA_W / 6.0 * 0.7;
    let gap = (BAR_AREA_W - bw * 6.0) / 7.0;

    for (i, &k) in ORDER.iter().enumerate() {
        let v = counts.get(k);
        let h = (v / n) * BAR_AREA_H;
        let x = BAR_AREA_X + gap + i as f64 * (bw + gap);
        let y = BAR_AREA_Y + BAR_AREA_H - h;
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w: bw,
            h,
            fill: color(k).to_string(),
            rx: Some(3.0),
            title: Some(format!("{} = {}", k.label(), js_num(v))),
            ..Default::default()
        }));
        shapes.push(Shape::Rect(RectShape {
            x,
            y: BAR_AREA_Y,
            w: bw,
            h: BAR_AREA_H,
            fill: "none".to_string(),
            stroke: Some("#ddd".to_string()),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + bw / 2.0,
            y: BAR_AREA_Y + BAR_AREA_H + 16.0,
            text: k.label().to_string(),
            font_size: Some(14.0),
            fill: Some("#333".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + bw / 2.0,
            y: y - 6.0,
            text: js_num(v),
            font_size: Some(11.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }
    shapes
}

/// Build the entire frame (bars + caption). The line chart is global to the
/// animation and is drawn separately by the player.
pub fn build_frame(t: f64, _tick: f64, counts: &CompartmentCounts, n: f64) -> FrameParts {
    let shapes = build_bars(counts, n);
    let live = counts.s + counts.a + counts.b + counts.ab + counts.r;
    let dead = counts.d;
    let per_comp: Vec<String> = ORDER
        .iter()
        .map(|&k| format!("{}={}", k.label(), js_num(counts.get(k))))
        .collect();
    let caption = format!(
        "t={}   alive={}   dead={}   {}",
        to_fixed(t, 2),
        js_num(live),
        js_num(dead),
        per_comp.join("  ")
    );
    FrameParts::with_caption(shapes, caption)
}

/// A trace of compartment populations over time for [`build_compartment_chart`].
#[derive(Clone, Debug, Default)]
pub struct TwoDiseaseTrace {
    pub t: Vec<f64>,
    pub s: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub ab: Vec<f64>,
    pub r: Vec<f64>,
    pub d: Vec<f64>,
}

impl TwoDiseaseTrace {
    fn series_for(&self, c: Compartment) -> &Vec<f64> {
        match c {
            Compartment::S => &self.s,
            Compartment::A => &self.a,
            Compartment::B => &self.b,
            Compartment::AB => &self.ab,
            Compartment::R => &self.r,
            Compartment::D => &self.d,
        }
    }
}

/// Build the global compartment-over-time chart (the panel beneath the bars).
pub fn build_compartment_chart(trace: &TwoDiseaseTrace, n: f64) -> ChartSpec {
    let series: Vec<ChartSeries> = ORDER
        .iter()
        .map(|&k| ChartSeries {
            label: k.label().to_string(),
            color: color(k).to_string(),
            t: trace.t.clone(),
            y: trace.series_for(k).clone(),
        })
        .collect();
    ChartSpec {
        x: CHART_X,
        y: CHART_Y,
        w: CHART_W,
        h: CHART_H,
        title: Some("Compartment populations over time".to_string()),
        y_min: Some(0.0),
        y_max: Some(n),
        series,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_produce_four_shapes_per_compartment_plus_header() {
        let counts = CompartmentCounts { s: 90.0, a: 5.0, b: 3.0, ab: 1.0, r: 1.0, d: 0.0 };
        let shapes = build_bars(&counts, 100.0);
        // 1 header + 6 * 4 shapes.
        assert_eq!(shapes.len(), 25);
    }

    #[test]
    fn frame_caption_lists_compartments() {
        let counts = CompartmentCounts { s: 90.0, a: 5.0, b: 3.0, ab: 1.0, r: 1.0, d: 0.0 };
        let fp = build_frame(1.5, 3.0, &counts, 100.0);
        let cap = fp.caption.unwrap();
        assert!(cap.starts_with("t=1.50   alive=100   dead=0   "));
        assert!(cap.contains("S=90"));
        assert!(cap.contains("AB=1"));
    }

    #[test]
    fn chart_has_six_series_bounded_by_n() {
        let trace = TwoDiseaseTrace {
            t: vec![0.0, 1.0],
            s: vec![100.0, 90.0],
            a: vec![0.0, 5.0],
            b: vec![0.0, 3.0],
            ab: vec![0.0, 1.0],
            r: vec![0.0, 1.0],
            d: vec![0.0, 0.0],
        };
        let chart = build_compartment_chart(&trace, 100.0);
        assert_eq!(chart.series.len(), 6);
        assert_eq!(chart.y_max, Some(100.0));
    }
}
