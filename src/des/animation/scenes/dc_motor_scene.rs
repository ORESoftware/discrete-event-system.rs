//! Port of `src/des/animation/scenes/dc-motor-scene.ts`.
//!
//! Class-based builder for the DC-motor (armature circuit) animation. Renders
//! the armature circuit (supply V, R, L, back-EMF source E = K_eω), a spinning
//! rotor, live gauges, and time-series charts of speed (with reference),
//! back-EMF, current and applied voltage.
//!
//! ## Conversion notes
//!
//! * `class DcMotorScene` → struct + `impl`; private methods become `&self`
//!   methods that push into `&mut Vec<Shape>`.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::control_systems::dc_motor::{DcMotorParams,
//!   MotorStateToken}` the scene reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, RectShape, Shape, TextShape,
};

pub const MOTOR_STAGE_W: f64 = 1000.0;
pub const MOTOR_STAGE_H: f64 = 760.0;

// PORT NOTE: local mirror of the DC-motor params + state token (subset used).
#[derive(Clone, Debug, Default)]
pub struct DcMotorParams {
    pub resistance: f64,
    pub inductance: f64,
}

#[derive(Clone, Debug, Default)]
pub struct MotorStateToken {
    pub time: f64,
    pub omega: f64,
    pub current: f64,
    pub voltage: f64,
    pub back_emf: f64,
    pub load_torque: f64,
}

#[derive(Clone, Debug, Default)]
pub struct DcMotorSceneOpts {
    pub samples: Vec<MotorStateToken>,
    pub dt: f64,
    pub params: DcMotorParams,
    pub mode_name: String,
    /// Reference speed per sample (closed loop), or `None` for open loop.
    pub reference: Option<Vec<f64>>,
}

const COL_BG: &str = "#0b1021";
const COL_PANEL: &str = "#161d33";
const COL_WIRE: &str = "#64748b";
const COL_V: &str = "#fbbf24";
const COL_EMF: &str = "#f472b6";
const COL_I: &str = "#38bdf8";
const COL_OMEGA: &str = "#a78bfa";
const COL_REF: &str = "#22c55e";

pub struct DcMotorScene {
    opts: DcMotorSceneOpts,
    spin_angle: Vec<f64>,
    times: Vec<f64>,
    max_abs_i: f64,
    max_v: f64,
    max_omega: f64,
}

impl DcMotorScene {
    pub fn new(opts: DcMotorSceneOpts) -> Self {
        let mut spin_angle = Vec::new();
        let mut angle = 0.0;
        for s in &opts.samples {
            angle += s.omega * opts.dt * 0.15;
            spin_angle.push(angle);
        }
        let times: Vec<f64> = opts.samples.iter().map(|s| s.time).collect();
        let max_abs_i = opts
            .samples
            .iter()
            .map(|s| s.current.abs())
            .fold(1.0_f64, f64::max);
        let max_v = opts
            .samples
            .iter()
            .map(|s| s.voltage.abs())
            .fold(1.0_f64, f64::max);
        let mut max_omega = opts.samples.iter().map(|s| s.omega).fold(1.0_f64, f64::max);
        if let Some(reference) = &opts.reference {
            max_omega = reference.iter().copied().fold(max_omega, f64::max);
        }
        DcMotorScene {
            opts,
            spin_angle,
            times,
            max_abs_i,
            max_v,
            max_omega,
        }
    }

    pub fn frame_count(&self) -> usize {
        self.opts.samples.len()
    }

    pub fn time_at(&self, i: usize) -> f64 {
        self.times[i]
    }

    pub fn frame_at(&self, i: usize) -> FrameParts {
        let s = &self.opts.samples[i];
        let ref_val: Option<f64> = self.opts.reference.as_ref().map(|r| r[i]);
        let mut shapes: Vec<Shape> = Vec::new();
        shapes.push(Shape::Rect(RectShape {
            x: 0.0,
            y: 0.0,
            w: MOTOR_STAGE_W,
            h: MOTOR_STAGE_H,
            fill: COL_BG.to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: MOTOR_STAGE_W / 2.0,
            y: 34.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(22.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some("#f8fafc".to_string()),
            text: format!("DC Motor (back-EMF ODE) — {}", self.opts.mode_name),
            ..Default::default()
        }));

        self.draw_circuit(&mut shapes, s);
        self.draw_rotor(&mut shapes, 700.0, 250.0, self.spin_angle[i], s.omega);
        self.draw_gauges(&mut shapes, s, ref_val);

        let ref_str = match ref_val {
            None => String::new(),
            Some(r) => format!("   \u{03c9}*={}", to_fixed(r, 1)),
        };
        let caption = format!(
            "t={}s   V={} V   i={} A   \u{03c9}={} rad/s{}   E=K_e\u{03c9}={} V   T_L={} N\u{00b7}m",
            to_fixed(s.time, 3),
            to_fixed(s.voltage, 2),
            to_fixed(s.current, 3),
            to_fixed(s.omega, 2),
            ref_str,
            to_fixed(s.back_emf, 2),
            to_fixed(s.load_torque, 2)
        );
        FrameParts::with_caption(shapes, caption)
    }

    pub fn charts(&self) -> Vec<ChartSpec> {
        let t = &self.times;
        let s = &self.opts.samples;
        let mut series = vec![ChartSeries {
            label: "\u{03c9}".to_string(),
            color: COL_OMEGA.to_string(),
            t: t.clone(),
            y: s.iter().map(|x| x.omega).collect(),
        }];
        if let Some(reference) = &self.opts.reference {
            series.push(ChartSeries {
                label: "\u{03c9}*".to_string(),
                color: COL_REF.to_string(),
                t: t.clone(),
                y: reference.clone(),
            });
        }
        let back_emf_max = s.iter().map(|x| x.back_emf).fold(1.0_f64, f64::max) * 1.2;
        let voltage_min = s.iter().map(|x| x.voltage).fold(0.0_f64, f64::min) * 1.1 - 0.1;
        vec![
            ChartSpec {
                x: 40.0,
                y: 540.0,
                w: 460.0,
                h: 200.0,
                title: Some("Rotor speed \u{03c9} (rad/s)".to_string()),
                y_label: Some("rad/s".to_string()),
                y_min: Some(0.0),
                y_max: Some(self.max_omega * 1.15),
                series,
                ..Default::default()
            },
            ChartSpec {
                x: 520.0,
                y: 540.0,
                w: 440.0,
                h: 200.0,
                title: Some("Back-EMF  E = K_e\u{00b7}\u{03c9} (V)".to_string()),
                y_label: Some("V".to_string()),
                y_min: Some(0.0),
                y_max: Some(back_emf_max),
                series: vec![ChartSeries {
                    label: "E".to_string(),
                    color: COL_EMF.to_string(),
                    t: t.clone(),
                    y: s.iter().map(|x| x.back_emf).collect(),
                }],
                ..Default::default()
            },
            ChartSpec {
                x: 40.0,
                y: 470.0,
                w: 460.0,
                h: 60.0,
                title: Some("Armature current i (A)".to_string()),
                y_label: Some("A".to_string()),
                y_min: Some(-self.max_abs_i * 1.1),
                y_max: Some(self.max_abs_i * 1.1),
                series: vec![ChartSeries {
                    label: "i".to_string(),
                    color: COL_I.to_string(),
                    t: t.clone(),
                    y: s.iter().map(|x| x.current).collect(),
                }],
                ..Default::default()
            },
            ChartSpec {
                x: 520.0,
                y: 470.0,
                w: 440.0,
                h: 60.0,
                title: Some("Applied voltage V (V)".to_string()),
                y_label: Some("V".to_string()),
                y_min: Some(voltage_min),
                y_max: Some(self.max_v * 1.1),
                series: vec![ChartSeries {
                    label: "V".to_string(),
                    color: COL_V.to_string(),
                    t: t.clone(),
                    y: s.iter().map(|x| x.voltage).collect(),
                }],
                ..Default::default()
            },
        ]
    }

    fn draw_circuit(&self, shapes: &mut Vec<Shape>, s: &MotorStateToken) {
        let l = 90.0;
        let r = 560.0;
        let t = 110.0;
        let b = 360.0;
        shapes.push(Shape::Rect(RectShape {
            x: l - 30.0,
            y: t - 30.0,
            w: r - l + 120.0,
            h: b - t + 70.0,
            rx: Some(10.0),
            fill: COL_PANEL.to_string(),
            stroke: Some("#334155".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: l,
            y1: t,
            x2: r,
            y2: t,
            stroke: COL_WIRE.to_string(),
            stroke_width: Some(3.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: l,
            y1: b,
            x2: r,
            y2: b,
            stroke: COL_WIRE.to_string(),
            stroke_width: Some(3.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: l,
            y1: t,
            x2: l,
            y2: b,
            stroke: COL_WIRE.to_string(),
            stroke_width: Some(3.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: r,
            y1: t,
            x2: r,
            y2: b,
            stroke: COL_WIRE.to_string(),
            stroke_width: Some(3.0),
            ..Default::default()
        }));

        // Supply source (left).
        let mid = (t + b) / 2.0;
        shapes.push(Shape::Circle(CircleShape {
            x: l,
            y: mid,
            r: 26.0,
            fill: "#1e293b".to_string(),
            stroke: Some(COL_V.to_string()),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: l,
            y: mid - 2.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(13.0),
            fill: Some(COL_V.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: "V".to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: l,
            y: mid + 16.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(11.0),
            fill: Some(COL_V.to_string()),
            text: to_fixed(s.voltage, 1),
            ..Default::default()
        }));

        // Resistor R (top wire, box).
        shapes.push(Shape::Rect(RectShape {
            x: 220.0,
            y: t - 12.0,
            w: 70.0,
            h: 24.0,
            rx: Some(3.0),
            fill: "#1e293b".to_string(),
            stroke: Some(COL_WIRE.to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 255.0,
            y: t + 5.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(12.0),
            fill: Some("#cbd5e1".to_string()),
            text: format!("R={}\u{03a9}", js_num(self.opts.params.resistance)),
            ..Default::default()
        }));
        // Inductor L (top wire, coil hint).
        shapes.push(Shape::Rect(RectShape {
            x: 340.0,
            y: t - 12.0,
            w: 70.0,
            h: 24.0,
            rx: Some(12.0),
            fill: "#1e293b".to_string(),
            stroke: Some(COL_WIRE.to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 375.0,
            y: t + 5.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(12.0),
            fill: Some("#cbd5e1".to_string()),
            text: format!("L={}H", js_num(self.opts.params.inductance)),
            ..Default::default()
        }));

        // Back-EMF source (right vertical).
        let emf_x = r;
        let emf_y = (t + b) / 2.0;
        shapes.push(Shape::Circle(CircleShape {
            x: emf_x,
            y: emf_y,
            r: 26.0,
            fill: "#1e293b".to_string(),
            stroke: Some(COL_EMF.to_string()),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: emf_x,
            y: emf_y - 2.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(12.0),
            fill: Some(COL_EMF.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: "E".to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: emf_x,
            y: emf_y + 16.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(11.0),
            fill: Some(COL_EMF.to_string()),
            text: to_fixed(s.back_emf, 1),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: emf_x,
            y: emf_y + 44.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(10.0),
            fill: Some("#94a3b8".to_string()),
            text: "K_e\u{00b7}\u{03c9}".to_string(),
            ..Default::default()
        }));

        // Current-flow markers around the loop.
        self.draw_current_markers(shapes, l, r, t, b, s.current);
        shapes.push(Shape::Text(TextShape {
            x: 255.0,
            y: b + 26.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(13.0),
            fill: Some(COL_I.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: format!("i = {} A", to_fixed(s.current, 3)),
            ..Default::default()
        }));
    }

    fn draw_current_markers(
        &self,
        shapes: &mut Vec<Shape>,
        l: f64,
        r: f64,
        t: f64,
        b: f64,
        current: f64,
    ) {
        let perim = 2.0 * (r - l) + 2.0 * (b - t);
        let count = 8;
        let sign = if current >= 0.0 { 1.0 } else { -1.0 };
        let phase = ((sign * (current.abs() * 7.0) % perim) + perim) % perim;
        for k in 0..count {
            let d = (phase + (k as f64 / count as f64) * perim) % perim;
            let p = self.perimeter_point(d, l, r, t, b);
            let opacity = 0.35 + 0.5 * (current.abs() / 1.0_f64.max(self.max_abs_i)).min(1.0);
            shapes.push(Shape::Circle(CircleShape {
                x: p.0,
                y: p.1,
                r: 4.0,
                fill: COL_I.to_string(),
                opacity: Some(opacity),
                ..Default::default()
            }));
        }
    }

    fn perimeter_point(&self, mut d: f64, l: f64, r: f64, t: f64, b: f64) -> (f64, f64) {
        let top = r - l;
        let right = b - t;
        let bottom = r - l;
        if d < top {
            return (l + d, t);
        }
        d -= top;
        if d < right {
            return (r, t + d);
        }
        d -= right;
        if d < bottom {
            return (r - d, b);
        }
        d -= bottom;
        (l, b - d)
    }

    fn draw_rotor(&self, shapes: &mut Vec<Shape>, cx: f64, cy: f64, angle: f64, omega: f64) {
        shapes.push(Shape::Circle(CircleShape {
            x: cx,
            y: cy,
            r: 64.0,
            fill: "#1e293b".to_string(),
            stroke: Some(COL_WIRE.to_string()),
            stroke_width: Some(3.0),
            ..Default::default()
        }));
        shapes.push(Shape::Circle(CircleShape {
            x: cx,
            y: cy,
            r: 58.0,
            fill: "transparent".to_string(),
            stroke: Some("#334155".to_string()),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        for k in 0..4 {
            let a = angle + (k as f64 * std::f64::consts::PI) / 2.0;
            shapes.push(Shape::Line(LineShape {
                x1: cx,
                y1: cy,
                x2: cx + 54.0 * a.cos(),
                y2: cy + 54.0 * a.sin(),
                stroke: COL_OMEGA.to_string(),
                stroke_width: Some(3.0),
                ..Default::default()
            }));
        }
        shapes.push(Shape::Circle(CircleShape {
            x: cx,
            y: cy,
            r: 8.0,
            fill: COL_OMEGA.to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: cy + 92.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(13.0),
            fill: Some(COL_OMEGA.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: format!("\u{03c9} = {} rad/s", to_fixed(omega, 1)),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: cy - 84.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(12.0),
            fill: Some("#94a3b8".to_string()),
            text: "ROTOR".to_string(),
            ..Default::default()
        }));
    }

    fn draw_gauges(&self, shapes: &mut Vec<Shape>, s: &MotorStateToken, ref_val: Option<f64>) {
        let x = 820.0;
        let y = 110.0;
        let w = 150.0;
        let row_h = 64.0;
        let mut rows: Vec<(String, String, &str)> = vec![
            (
                "speed \u{03c9}".to_string(),
                format!("{} rad/s", to_fixed(s.omega, 1)),
                COL_OMEGA,
            ),
            (
                "back-EMF E".to_string(),
                format!("{} V", to_fixed(s.back_emf, 2)),
                COL_EMF,
            ),
            (
                "current i".to_string(),
                format!("{} A", to_fixed(s.current, 3)),
                COL_I,
            ),
            (
                "voltage V".to_string(),
                format!("{} V", to_fixed(s.voltage, 2)),
                COL_V,
            ),
        ];
        if let Some(r) = ref_val {
            rows.insert(
                0,
                (
                    "reference \u{03c9}*".to_string(),
                    format!("{} rad/s", to_fixed(r, 1)),
                    COL_REF,
                ),
            );
        }
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w,
            h: rows.len() as f64 * row_h + 12.0,
            rx: Some(8.0),
            fill: COL_PANEL.to_string(),
            stroke: Some("#334155".to_string()),
            ..Default::default()
        }));
        for (k, row) in rows.iter().enumerate() {
            let ry = y + 12.0 + k as f64 * row_h;
            shapes.push(Shape::Text(TextShape {
                x: x + 12.0,
                y: ry + 18.0,
                anchor: Some(Anchor::Start),
                font_size: Some(11.0),
                fill: Some("#94a3b8".to_string()),
                text: row.0.clone(),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: x + 12.0,
                y: ry + 42.0,
                anchor: Some(Anchor::Start),
                font_size: Some(18.0),
                fill: Some(row.2.to_string()),
                font_weight: Some(FontWeight::Bold),
                text: row.1.clone(),
                ..Default::default()
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time: f64, omega: f64) -> MotorStateToken {
        MotorStateToken {
            time,
            omega,
            current: 0.5,
            voltage: 12.0,
            back_emf: 4.0,
            load_torque: 0.2,
        }
    }

    #[test]
    fn frame_count_and_open_loop_caption() {
        let scene = DcMotorScene::new(DcMotorSceneOpts {
            samples: vec![sample(0.0, 0.0), sample(0.1, 10.0)],
            dt: 0.1,
            params: DcMotorParams {
                resistance: 1.0,
                inductance: 0.5,
            },
            mode_name: "open loop".to_string(),
            reference: None,
        });
        assert_eq!(scene.frame_count(), 2);
        let fp = scene.frame_at(1);
        let cap = fp.caption.unwrap();
        assert!(cap.starts_with("t=0.100s"));
        assert!(!cap.contains("\u{03c9}*="));
        // Open loop -> two charts series-less reference; first chart has one series.
        assert_eq!(scene.charts()[0].series.len(), 1);
    }

    #[test]
    fn closed_loop_adds_reference() {
        let scene = DcMotorScene::new(DcMotorSceneOpts {
            samples: vec![sample(0.0, 0.0), sample(0.1, 10.0)],
            dt: 0.1,
            params: DcMotorParams {
                resistance: 1.0,
                inductance: 0.5,
            },
            mode_name: "PI".to_string(),
            reference: Some(vec![5.0, 9.0]),
        });
        let fp = scene.frame_at(1);
        assert!(fp.caption.unwrap().contains("\u{03c9}*=9.0"));
        assert_eq!(scene.charts()[0].series.len(), 2);
    }
}
