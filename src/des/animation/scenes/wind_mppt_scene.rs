//! Port of `src/des/animation/scenes/wind-mppt-scene.ts`.
//!
//! Class-based builder for the wind-turbine MPPT animation. Renders a spinning
//! variable-speed wind turbine, live wind arrows, λ / C_p gauges against their
//! optimal targets, and time-series charts of λ, C_p, ω and captured power,
//! replayed from a recorded `TurbineStateToken` trajectory.
//!
//! ## Conversion notes
//!
//! * `class WindMpptScene` → struct + `impl`; private methods push into
//!   `&mut Vec<Shape>`.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::control_systems::wind_mppt::TurbineStateToken` the
//!   scene reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    to_exponential, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const WIND_STAGE_W: f64 = 1000.0;
pub const WIND_STAGE_H: f64 = 760.0;

// PORT NOTE: local mirror of the turbine state token (subset used here).
#[derive(Clone, Debug, Default)]
pub struct TurbineStateToken {
    pub time: f64,
    pub omega: f64,
    pub wind_speed: f64,
    pub lambda: f64,
    pub cp: f64,
    pub mech_power: f64,
    pub gen_torque: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WindSceneOpts {
    pub samples: Vec<TurbineStateToken>,
    pub dt: f64,
    pub lambda_star: f64,
    pub cp_max: f64,
    pub k_opt: f64,
    pub controller_name: String,
}

const COL_BG: &str = "#0b1021";
const COL_PANEL: &str = "#161d33";
const COL_BLADE: &str = "#e2e8f0";
const COL_HUB: &str = "#94a3b8";
const COL_WIND: &str = "#38bdf8";
const COL_LAMBDA: &str = "#f59e0b";
const COL_CP: &str = "#34d399";
const COL_OMEGA: &str = "#a78bfa";
const COL_POWER: &str = "#fb7185";
const COL_TARGET: &str = "#22c55e";

pub struct WindMpptScene {
    opts: WindSceneOpts,
    spin_angle: Vec<f64>,
    times: Vec<f64>,
    max_power: f64,
    max_omega: f64,
}

impl WindMpptScene {
    pub fn new(opts: WindSceneOpts) -> Self {
        let mut spin_angle = Vec::new();
        let mut angle = 0.0;
        for s in &opts.samples {
            angle += s.omega * opts.dt * 0.25;
            spin_angle.push(angle);
        }
        let times: Vec<f64> = opts.samples.iter().map(|s| s.time).collect();
        let max_power = opts.samples.iter().map(|s| s.mech_power).fold(1.0_f64, f64::max);
        let max_omega = opts.samples.iter().map(|s| s.omega).fold(1.0_f64, f64::max);
        WindMpptScene { opts, spin_angle, times, max_power, max_omega }
    }

    pub fn frame_count(&self) -> usize {
        self.opts.samples.len()
    }

    /// Simulation time at sample index `i`.
    pub fn time_at(&self, i: usize) -> f64 {
        self.times[i]
    }

    /// Build the scene at sample index `i`.
    pub fn frame_at(&self, i: usize) -> FrameParts {
        let s = &self.opts.samples[i];
        let mut shapes: Vec<Shape> = Vec::new();
        shapes.push(Shape::Rect(RectShape { x: 0.0, y: 0.0, w: WIND_STAGE_W, h: WIND_STAGE_H, fill: COL_BG.to_string(), ..Default::default() }));
        shapes.push(Shape::Text(TextShape {
            x: WIND_STAGE_W / 2.0,
            y: 34.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(22.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some("#f8fafc".to_string()),
            text: format!("Wind MPPT — PMSG WECS  \u{00b7}  {}", self.opts.controller_name),
            ..Default::default()
        }));

        self.draw_wind(&mut shapes, s.wind_speed);
        self.draw_turbine(&mut shapes, 360.0, 300.0, self.spin_angle[i]);
        self.draw_gauges(&mut shapes, s);
        self.draw_readouts(&mut shapes, s);

        let capture_pct = (s.cp / self.opts.cp_max) * 100.0;
        let caption = format!(
            "t={}s   V={} m/s   \u{03c9}={} rad/s   \u{03bb}={} (\u{03bb}*={})   C_p={} ({}% of max)   P={} kW",
            to_fixed(s.time, 2),
            to_fixed(s.wind_speed, 2),
            to_fixed(s.omega, 2),
            to_fixed(s.lambda, 2),
            to_fixed(self.opts.lambda_star, 2),
            to_fixed(s.cp, 3),
            to_fixed(capture_pct, 1),
            to_fixed(s.mech_power / 1000.0, 2)
        );
        FrameParts::with_caption(shapes, caption)
    }

    pub fn charts(&self) -> Vec<ChartSpec> {
        let t = &self.times;
        let samples = &self.opts.samples;
        let end = t[t.len() - 1];
        let lambda_max = samples.iter().map(|s| s.lambda).fold(self.opts.lambda_star * 1.4, f64::max) + 1.0;
        vec![
            ChartSpec {
                x: 40.0,
                y: 540.0,
                w: 460.0,
                h: 200.0,
                title: Some("Tip-speed ratio \u{03bb}".to_string()),
                y_label: Some("\u{03bb}".to_string()),
                y_min: Some(0.0),
                y_max: Some(lambda_max),
                series: vec![
                    ChartSeries { label: "\u{03bb}".to_string(), color: COL_LAMBDA.to_string(), t: t.clone(), y: samples.iter().map(|s| s.lambda).collect() },
                    ChartSeries { label: "\u{03bb}*".to_string(), color: COL_TARGET.to_string(), t: vec![t[0], end], y: vec![self.opts.lambda_star, self.opts.lambda_star] },
                ],
                ..Default::default()
            },
            ChartSpec {
                x: 520.0,
                y: 540.0,
                w: 440.0,
                h: 200.0,
                title: Some("Power coefficient C_p".to_string()),
                y_label: Some("C_p".to_string()),
                y_min: Some(0.0),
                y_max: Some(self.opts.cp_max * 1.25),
                series: vec![
                    ChartSeries { label: "C_p".to_string(), color: COL_CP.to_string(), t: t.clone(), y: samples.iter().map(|s| s.cp).collect() },
                    ChartSeries { label: "C_p,max".to_string(), color: COL_TARGET.to_string(), t: vec![t[0], end], y: vec![self.opts.cp_max, self.opts.cp_max] },
                ],
                ..Default::default()
            },
            ChartSpec {
                x: 40.0,
                y: 300.0,
                w: 240.0,
                h: 150.0,
                title: Some("Rotor speed \u{03c9} (rad/s)".to_string()),
                y_label: Some("\u{03c9}".to_string()),
                y_min: Some(0.0),
                y_max: Some(self.max_omega * 1.15),
                series: vec![ChartSeries { label: "\u{03c9}".to_string(), color: COL_OMEGA.to_string(), t: t.clone(), y: samples.iter().map(|s| s.omega).collect() }],
                ..Default::default()
            },
            ChartSpec {
                x: 40.0,
                y: 460.0,
                w: 240.0,
                h: 70.0,
                title: Some("Captured power (kW)".to_string()),
                y_label: Some("kW".to_string()),
                y_min: Some(0.0),
                y_max: Some((self.max_power / 1000.0) * 1.15),
                series: vec![ChartSeries { label: "P".to_string(), color: COL_POWER.to_string(), t: t.clone(), y: samples.iter().map(|s| s.mech_power / 1000.0).collect() }],
                ..Default::default()
            },
        ]
    }

    fn draw_wind(&self, shapes: &mut Vec<Shape>, wind_speed: f64) {
        let arrow_count = 2.max((wind_speed / 2.0).round() as i64);
        let len = 40.0 + wind_speed * 6.0;
        for k in 0..arrow_count {
            let y = 140.0 + k as f64 * 36.0;
            shapes.push(Shape::Line(LineShape { x1: 60.0, y1: y, x2: 60.0 + len, y2: y, stroke: COL_WIND.to_string(), stroke_width: Some(2.0), opacity: Some(0.8), ..Default::default() }));
            shapes.push(Shape::Path(PathShape {
                d: format!("M {},{} L {},{} L {},{}", to_fixed_raw(60.0 + len - 10.0), to_fixed_raw(y - 5.0), to_fixed_raw(60.0 + len), to_fixed_raw(y), to_fixed_raw(60.0 + len - 10.0), to_fixed_raw(y + 5.0)),
                stroke: Some(COL_WIND.to_string()),
                fill: Some(COL_WIND.to_string()),
                ..Default::default()
            }));
        }
        shapes.push(Shape::Text(TextShape { x: 60.0, y: 120.0, anchor: Some(Anchor::Start), font_size: Some(14.0), fill: Some(COL_WIND.to_string()), font_weight: Some(FontWeight::Bold), text: format!("wind {} m/s \u{2192}", to_fixed(wind_speed, 1)), ..Default::default() }));
    }

    fn draw_turbine(&self, shapes: &mut Vec<Shape>, cx: f64, cy: f64, angle: f64) {
        // Tower.
        shapes.push(Shape::Path(PathShape {
            d: format!("M {},{} L {},{} L {},{} L {},{} Z", to_fixed_raw(cx - 14.0), to_fixed_raw(cy + 260.0), to_fixed_raw(cx - 5.0), to_fixed_raw(cy), to_fixed_raw(cx + 5.0), to_fixed_raw(cy), to_fixed_raw(cx + 14.0), to_fixed_raw(cy + 260.0)),
            fill: Some("#475569".to_string()),
            stroke: Some("#1e293b".to_string()),
            ..Default::default()
        }));
        // Nacelle.
        shapes.push(Shape::Rect(RectShape { x: cx - 18.0, y: cy - 14.0, w: 50.0, h: 28.0, rx: Some(6.0), fill: "#64748b".to_string(), stroke: Some("#1e293b".to_string()), ..Default::default() }));
        // Three blades.
        for b in 0..3 {
            let a = angle + (b as f64 * 2.0 * std::f64::consts::PI) / 3.0;
            let tip_x = cx + 170.0 * a.cos();
            let tip_y = cy + 170.0 * a.sin();
            let perp_x = 12.0 * (a + std::f64::consts::FRAC_PI_2).cos();
            let perp_y = 12.0 * (a + std::f64::consts::FRAC_PI_2).sin();
            shapes.push(Shape::Path(PathShape {
                d: format!("M {},{} L {},{} L {},{} Z", to_fixed_raw(cx + perp_x), to_fixed_raw(cy + perp_y), to_fixed_raw(tip_x), to_fixed_raw(tip_y), to_fixed_raw(cx - perp_x), to_fixed_raw(cy - perp_y)),
                fill: Some(COL_BLADE.to_string()),
                stroke: Some("#94a3b8".to_string()),
                opacity: Some(0.95),
                ..Default::default()
            }));
        }
        // Hub.
        shapes.push(Shape::Circle(CircleShape { x: cx, y: cy, r: 14.0, fill: COL_HUB.to_string(), stroke: Some("#1e293b".to_string()), stroke_width: Some(2.0), ..Default::default() }));
    }

    fn draw_gauges(&self, shapes: &mut Vec<Shape>, s: &TurbineStateToken) {
        self.draw_bar(shapes, 720.0, 110.0, "\u{03bb} / \u{03bb}*", s.lambda, self.opts.lambda_star, self.opts.lambda_star * 1.4, COL_LAMBDA);
        self.draw_bar(shapes, 850.0, 110.0, "C_p / C_p,max", s.cp, self.opts.cp_max, self.opts.cp_max * 1.25, COL_CP);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_bar(&self, shapes: &mut Vec<Shape>, x: f64, y: f64, label: &str, value: f64, target: f64, max: f64, color: &str) {
        let h = 300.0;
        let w = 46.0;
        shapes.push(Shape::Rect(RectShape { x, y, w, h, rx: Some(6.0), fill: COL_PANEL.to_string(), stroke: Some("#334155".to_string()), ..Default::default() }));
        let frac = (value / max).clamp(0.0, 1.0);
        let fill_h = frac * (h - 4.0);
        shapes.push(Shape::Rect(RectShape { x: x + 2.0, y: y + h - 2.0 - fill_h, w: w - 4.0, h: fill_h, rx: Some(4.0), fill: color.to_string(), opacity: Some(0.9), ..Default::default() }));
        let tgt_y = y + h - 2.0 - (target / max).min(1.0) * (h - 4.0);
        shapes.push(Shape::Line(LineShape { x1: x - 6.0, y1: tgt_y, x2: x + w + 6.0, y2: tgt_y, stroke: COL_TARGET.to_string(), stroke_width: Some(2.0), dasharray: Some("5,3".to_string()), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + w / 2.0, y: y - 10.0, anchor: Some(Anchor::Middle), font_size: Some(12.0), fill: Some("#cbd5e1".to_string()), font_weight: Some(FontWeight::Bold), text: label.to_string(), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + w / 2.0, y: y + h + 18.0, anchor: Some(Anchor::Middle), font_size: Some(13.0), fill: Some(color.to_string()), font_weight: Some(FontWeight::Bold), text: to_fixed(value, 2), ..Default::default() }));
    }

    fn draw_readouts(&self, shapes: &mut Vec<Shape>, s: &TurbineStateToken) {
        let x = 700.0;
        let y = 440.0;
        let w = 260.0;
        let h = 80.0;
        shapes.push(Shape::Rect(RectShape { x, y, w, h, rx: Some(8.0), fill: COL_PANEL.to_string(), stroke: Some("#334155".to_string()), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + 14.0, y: y + 24.0, anchor: Some(Anchor::Start), font_size: Some(13.0), fill: Some("#94a3b8".to_string()), text: "Captured power".to_string(), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + 14.0, y: y + 56.0, anchor: Some(Anchor::Start), font_size: Some(26.0), fill: Some(COL_POWER.to_string()), font_weight: Some(FontWeight::Bold), text: format!("{} kW", to_fixed(s.mech_power / 1000.0, 2)), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + w - 14.0, y: y + 24.0, anchor: Some(Anchor::End), font_size: Some(12.0), fill: Some("#94a3b8".to_string()), text: format!("T_gen = {} N\u{00b7}m", to_fixed(s.gen_torque, 2)), ..Default::default() }));
        shapes.push(Shape::Text(TextShape { x: x + w - 14.0, y: y + 56.0, anchor: Some(Anchor::End), font_size: Some(12.0), fill: Some("#94a3b8".to_string()), text: format!("K_opt = {}", to_exponential(self.opts.k_opt, 2)), ..Default::default() }));
    }
}

/// `n` formatted as `String(n)` would be in the TS path-string interpolation
/// (no fixed decimals) — used for SVG path coordinates.
fn to_fixed_raw(n: f64) -> String {
    crate::des::animation::types::js_num(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time: f64) -> TurbineStateToken {
        TurbineStateToken { time, omega: 2.0, wind_speed: 8.0, lambda: 7.0, cp: 0.45, mech_power: 1500.0, gen_torque: 5.0 }
    }

    #[test]
    fn frame_count_and_caption() {
        let scene = WindMpptScene::new(WindSceneOpts {
            samples: vec![sample(0.0), sample(0.5)],
            dt: 0.5,
            lambda_star: 7.0,
            cp_max: 0.48,
            k_opt: 0.0012,
            controller_name: "TSR".to_string(),
        });
        assert_eq!(scene.frame_count(), 2);
        let fp = scene.frame_at(1);
        assert!(fp.caption.unwrap().starts_with("t=0.50s"));
        assert_eq!(scene.charts().len(), 4);
    }
}
