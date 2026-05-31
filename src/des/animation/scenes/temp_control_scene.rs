//! Port of `src/des/animation/scenes/temp-control-scene.ts`.
//!
//! Builds frames + charts for the temperature-control DES animation: a top-row
//! station diagram with a travelling pulse, a thermometer with target band, a
//! heater dial, outdoor/comfort readouts, and time-series charts.
//!
//! ## Conversion notes
//!
//! * Multi-line station labels via `s.label.split('\n')` → [`str::split`].
//! * SVG `path` `d` strings are rebuilt with `format!` + `js_num` so each
//!   interpolated number matches `String(number)`.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::temp_control::{RunResult, TickRecord}` the scene
//!   reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, Frame, FrameParts,
    LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1000.0;
pub const STAGE_H: f64 = 700.0;

const COL_TARGET: &str = "#16a34a";
const COL_BAND: &str = "#bbf7d0";
const COL_T_IN: &str = "#dc2626";
const COL_T_OUT: &str = "#1d4ed8";
const COL_HEAT: &str = "#f97316";
const COL_COOL: &str = "#0284c7";
const COL_BG: &str = "#f9fafb";

// PORT NOTE: local mirror of the temp-control model (subset used by the scene).
#[derive(Clone, Debug, Default)]
pub struct TickRecord {
    pub t_h: f64,
    pub tick: f64,
    pub t_in_true: f64,
    pub t_out_true: f64,
    pub q: f64,
    pub in_band: bool,
    pub energy_cum_k_wh: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunConfig {
    pub t_target: f64,
    pub band: Option<f64>,
    pub q_min: f64,
    pub q_max: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunResult {
    pub cfg: RunConfig,
    pub trace: Vec<TickRecord>,
    pub t_in: Vec<f64>,
    pub t_out: Vec<f64>,
    pub q: Vec<f64>,
}

/// Per-frame scene inputs (the TS `SceneData`).
#[derive(Clone, Debug)]
pub struct SceneData<'a> {
    pub tick: &'a TickRecord,
    pub t_target: f64,
    pub band: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub controller_name: String,
    pub energy: f64,
    pub comfort_pct: f64,
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

/// Build one frame's static layout (thermometer, heater, controller diagram).
pub fn build_temp_control_frame(_t: f64, tick: f64, d: &SceneData) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();

    // Background.
    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: STAGE_W,
        h: STAGE_H,
        fill: COL_BG.to_string(),
        ..Default::default()
    }));

    // Top row: station-flow diagram.
    let actuator_label = if d.q_min < 0.0 {
        "Heat/Cool".to_string()
    } else {
        "Heater".to_string()
    };
    let stations: Vec<(f64, String)> = vec![
        (60.0, "Outdoor\nSource".to_string()),
        (220.0, "Forecast\nStation".to_string()),
        (380.0, "Sensor +\nComparator".to_string()),
        (540.0, d.controller_name.clone()),
        (700.0, actuator_label),
        (860.0, "House\n(Physics)".to_string()),
    ];
    let st_row_y = 70.0;
    for (sx, label) in &stations {
        shapes.push(Shape::Rect(RectShape {
            x: sx - 60.0,
            y: st_row_y,
            w: 120.0,
            h: 60.0,
            fill: "#fff".to_string(),
            stroke: Some("#888".to_string()),
            stroke_width: Some(1.2),
            rx: Some(6.0),
            ..Default::default()
        }));
        let label_lines: Vec<&str> = label.split('\n').collect();
        for (i, line) in label_lines.iter().enumerate() {
            shapes.push(Shape::Text(TextShape {
                x: *sx,
                y: st_row_y + 25.0 + i as f64 * 16.0,
                text: (*line).to_string(),
                font_size: Some(12.0),
                fill: Some("#222".to_string()),
                anchor: Some(Anchor::Middle),
                font_weight: Some(if i == 0 {
                    FontWeight::Bold
                } else {
                    FontWeight::Normal
                }),
                ..Default::default()
            }));
        }
    }
    // Arrows between stations.
    for i in 0..stations.len() - 1 {
        let x1 = stations[i].0 + 60.0;
        let x2 = stations[i + 1].0 - 60.0;
        shapes.push(Shape::Line(LineShape {
            x1,
            y1: st_row_y + 30.0,
            x2: x2 - 6.0,
            y2: st_row_y + 30.0,
            stroke: "#888".to_string(),
            stroke_width: Some(1.5),
            ..Default::default()
        }));
        shapes.push(Shape::Path(PathShape {
            d: format!(
                "M {},{} L {},{} L {},{}",
                js_num(x2 - 8.0),
                js_num(st_row_y + 26.0),
                js_num(x2),
                js_num(st_row_y + 30.0),
                js_num(x2 - 8.0),
                js_num(st_row_y + 34.0)
            ),
            fill: Some("#888".to_string()),
            stroke: Some("#888".to_string()),
            ..Default::default()
        }));
    }
    // Feedback arrow from House → Sensor.
    shapes.push(Shape::Path(PathShape {
        d: format!(
            "M {},{} V {} H {} V {}",
            js_num(stations[5].0),
            js_num(st_row_y + 60.0),
            js_num(st_row_y + 100.0),
            js_num(stations[2].0),
            js_num(st_row_y + 60.0)
        ),
        stroke: Some("#888".to_string()),
        stroke_width: Some(1.5),
        fill: Some("transparent".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 620.0,
        y: st_row_y + 115.0,
        text: "feedback".to_string(),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    // Pulse circle on the line representing the current movable.
    let phase = (tick % 60.0) / 60.0;
    let idx = (phase * (stations.len() as f64 - 1.0)).floor() as usize;
    let seg = phase * (stations.len() as f64 - 1.0) - idx as f64;
    let cx = stations[idx].0 + 60.0 + seg * (stations[idx + 1].0 - 60.0 - (stations[idx].0 + 60.0));
    shapes.push(Shape::Circle(CircleShape {
        x: cx,
        y: st_row_y + 30.0,
        r: 4.0,
        fill: "#ec4899".to_string(),
        stroke: Some("#831843".to_string()),
        ..Default::default()
    }));

    // Middle row: thermometer + heater dial + numeric readouts.
    let mid_y = 200.0;
    let therm_x = 120.0;
    let therm_w = 40.0;
    let therm_y_top = mid_y;
    let therm_y_bot = mid_y + 280.0;
    let t_min = 50.0;
    let t_max = 90.0;
    let t_to_y =
        |tt: f64| therm_y_bot - ((tt - t_min) / (t_max - t_min)) * (therm_y_bot - therm_y_top);
    // Frame.
    shapes.push(Shape::Rect(RectShape {
        x: therm_x,
        y: therm_y_top,
        w: therm_w,
        h: therm_y_bot - therm_y_top,
        fill: "#fff".to_string(),
        stroke: Some("#444".to_string()),
        stroke_width: Some(1.5),
        rx: Some(4.0),
        ..Default::default()
    }));
    // Target band shading.
    let y_band_hi = t_to_y(d.t_target + d.band);
    let y_band_lo = t_to_y(d.t_target - d.band);
    shapes.push(Shape::Rect(RectShape {
        x: therm_x + 1.0,
        y: y_band_hi,
        w: therm_w - 2.0,
        h: y_band_lo - y_band_hi,
        fill: COL_BAND.to_string(),
        opacity: Some(0.7),
        ..Default::default()
    }));
    // Indoor temperature column (red).
    let y_in = t_to_y(d.tick.t_in_true);
    shapes.push(Shape::Rect(RectShape {
        x: therm_x + 12.0,
        y: y_in,
        w: therm_w - 24.0,
        h: therm_y_bot - y_in,
        fill: COL_T_IN.to_string(),
        opacity: Some(0.85),
        ..Default::default()
    }));
    // Tick marks every 5°F.
    let mut temp = t_min;
    while temp <= t_max {
        let y = t_to_y(temp);
        shapes.push(Shape::Line(LineShape {
            x1: therm_x + therm_w,
            y1: y,
            x2: therm_x + therm_w + 6.0,
            y2: y,
            stroke: "#444".to_string(),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: therm_x + therm_w + 10.0,
            y: y + 4.0,
            text: js_num(temp),
            font_size: Some(10.0),
            fill: Some("#444".to_string()),
            anchor: Some(Anchor::Start),
            ..Default::default()
        }));
        temp += 5.0;
    }
    // Target line.
    shapes.push(Shape::Line(LineShape {
        x1: therm_x - 5.0,
        y1: t_to_y(d.t_target),
        x2: therm_x + therm_w + 5.0,
        y2: t_to_y(d.t_target),
        stroke: COL_TARGET.to_string(),
        stroke_width: Some(2.0),
        dasharray: Some("4,3".to_string()),
        ..Default::default()
    }));
    // Bulb at the bottom.
    shapes.push(Shape::Circle(CircleShape {
        x: therm_x + therm_w / 2.0,
        y: therm_y_bot + 18.0,
        r: 22.0,
        fill: COL_T_IN.to_string(),
        stroke: Some("#444".to_string()),
        stroke_width: Some(1.5),
        ..Default::default()
    }));
    // T_in numeric readout.
    shapes.push(Shape::Text(TextShape {
        x: therm_x + therm_w / 2.0,
        y: therm_y_top - 12.0,
        text: format!("T_in = {}\u{00b0}F", to_fixed(d.tick.t_in_true, 2)),
        font_size: Some(14.0),
        fill: Some("#222".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    // HVAC command: heating-only keeps the original circular dial; heat/cool
    // runs use a centered bidirectional bar so negative cooling power is honest.
    let dial_x = 320.0;
    let dial_y = mid_y + 130.0;
    if d.q_min < 0.0 {
        let box_x = dial_x - 105.0;
        let box_y = dial_y - 85.0;
        let box_w = 210.0;
        let box_h = 170.0;
        let bar_x = box_x + 24.0;
        let bar_y = box_y + 90.0;
        let bar_w = box_w - 48.0;
        let bar_h = 20.0;
        let zero_x = bar_x + bar_w * (-d.q_min) / (d.q_max - d.q_min);
        shapes.push(Shape::Rect(RectShape {
            x: box_x,
            y: box_y,
            w: box_w,
            h: box_h,
            fill: "#fff".to_string(),
            stroke: Some("#444".to_string()),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: dial_x,
            y: box_y + 24.0,
            text: "HVAC COMMAND".to_string(),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: dial_x,
            y: box_y + 56.0,
            text: format!("Q = {} kW", to_fixed(d.tick.q, 2)),
            font_size: Some(17.0),
            fill: Some(if d.tick.q < -1e-9 {
                COL_COOL.to_string()
            } else if d.tick.q > 1e-9 {
                COL_HEAT.to_string()
            } else {
                "#222".to_string()
            }),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Rect(RectShape {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            fill: "#eef2f7".to_string(),
            stroke: Some("#94a3b8".to_string()),
            rx: Some(3.0),
            ..Default::default()
        }));
        if d.tick.q < 0.0 {
            let w = (zero_x - bar_x) * (d.tick.q / d.q_min).clamp(0.0, 1.0);
            shapes.push(Shape::Rect(RectShape {
                x: zero_x - w,
                y: bar_y + 1.0,
                w,
                h: bar_h - 2.0,
                fill: COL_COOL.to_string(),
                opacity: Some(0.9),
                ..Default::default()
            }));
        } else if d.tick.q > 0.0 {
            let w = (bar_x + bar_w - zero_x) * (d.tick.q / d.q_max).clamp(0.0, 1.0);
            shapes.push(Shape::Rect(RectShape {
                x: zero_x,
                y: bar_y + 1.0,
                w,
                h: bar_h - 2.0,
                fill: COL_HEAT.to_string(),
                opacity: Some(0.9),
                ..Default::default()
            }));
        }
        shapes.push(Shape::Line(LineShape {
            x1: zero_x,
            y1: bar_y - 8.0,
            x2: zero_x,
            y2: bar_y + bar_h + 8.0,
            stroke: "#111827".to_string(),
            stroke_width: Some(1.3),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: bar_x,
            y: bar_y + bar_h + 20.0,
            text: "cool".to_string(),
            font_size: Some(11.0),
            fill: Some(COL_COOL.to_string()),
            anchor: Some(Anchor::Start),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: bar_x + bar_w,
            y: bar_y + bar_h + 20.0,
            text: "heat".to_string(),
            font_size: Some(11.0),
            fill: Some(COL_HEAT.to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: dial_x,
            y: bar_y + bar_h + 44.0,
            text: format!(
                "range {}..{} kW",
                to_fixed(d.q_min, 0),
                to_fixed(d.q_max, 0)
            ),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    } else {
        let dial_r = 90.0;
        shapes.push(Shape::Circle(CircleShape {
            x: dial_x,
            y: dial_y,
            r: dial_r + 5.0,
            fill: "#fff".to_string(),
            stroke: Some("#444".to_string()),
            stroke_width: Some(1.5),
            ..Default::default()
        }));
        let q_frac = 0.0_f64.max(1.0_f64.min(d.tick.q / d.q_max));
        let start_angle = std::f64::consts::PI * (5.0 / 6.0);
        let end_angle =
            std::f64::consts::PI * (5.0 / 6.0) + q_frac * (std::f64::consts::PI * 4.0 / 3.0);
        let arc_end_x = dial_x + dial_r * end_angle.cos();
        let arc_end_y = dial_y + dial_r * end_angle.sin();
        let arc_start_x = dial_x + dial_r * start_angle.cos();
        let arc_start_y = dial_y + dial_r * start_angle.sin();
        let large_arc = if (end_angle - start_angle) > std::f64::consts::PI {
            1
        } else {
            0
        };
        shapes.push(Shape::Path(PathShape {
            d: format!(
                "M {} {} L {} {} A {} {} 0 {} 1 {} {} Z",
                js_num(dial_x),
                js_num(dial_y),
                js_num(arc_start_x),
                js_num(arc_start_y),
                js_num(dial_r),
                js_num(dial_r),
                large_arc,
                js_num(arc_end_x),
                js_num(arc_end_y)
            ),
            fill: Some(COL_HEAT.to_string()),
            opacity: Some(0.85),
            stroke: Some("#7c2d12".to_string()),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: dial_x,
            y: dial_y - 5.0,
            text: format!("Q = {} kW", to_fixed(d.tick.q, 2)),
            font_size: Some(14.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: dial_x,
            y: dial_y + 18.0,
            text: format!("{}% of max", to_fixed(q_frac * 100.0, 0)),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    // Outdoor temperature mini-display.
    let out_x = 480.0;
    let out_y = mid_y;
    shapes.push(Shape::Rect(RectShape {
        x: out_x,
        y: out_y,
        w: 200.0,
        h: 80.0,
        fill: "#fff".to_string(),
        stroke: Some("#444".to_string()),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: out_x + 100.0,
        y: out_y + 18.0,
        text: "OUTSIDE".to_string(),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: out_x + 100.0,
        y: out_y + 50.0,
        text: format!("{}\u{00b0}F", to_fixed(d.tick.t_out_true, 1)),
        font_size: Some(28.0),
        fill: Some(COL_T_OUT.to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: out_x + 100.0,
        y: out_y + 70.0,
        text: format!("t = {} h", to_fixed(d.tick.t_h, 2)),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));

    // Comfort + energy box.
    let cmf_x = 480.0;
    let cmf_y = mid_y + 100.0;
    shapes.push(Shape::Rect(RectShape {
        x: cmf_x,
        y: cmf_y,
        w: 200.0,
        h: 70.0,
        fill: "#fff".to_string(),
        stroke: Some("#444".to_string()),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: cmf_x + 8.0,
        y: cmf_y + 18.0,
        text: "COMFORT".to_string(),
        font_size: Some(11.0),
        fill: Some("#666".to_string()),
        anchor: Some(Anchor::Start),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: cmf_x + 8.0,
        y: cmf_y + 38.0,
        text: format!("{}% in band", to_fixed(d.comfort_pct * 100.0, 1)),
        font_size: Some(14.0),
        fill: Some("#222".to_string()),
        anchor: Some(Anchor::Start),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: cmf_x + 8.0,
        y: cmf_y + 56.0,
        text: format!("Energy: {} kWh", to_fixed(d.energy, 2)),
        font_size: Some(12.0),
        fill: Some("#444".to_string()),
        anchor: Some(Anchor::Start),
        ..Default::default()
    }));
    // Out-of-band indicator.
    if !d.tick.in_band {
        shapes.push(Shape::Rect(RectShape {
            x: cmf_x + 168.0,
            y: cmf_y + 8.0,
            w: 24.0,
            h: 24.0,
            fill: "#dc2626".to_string(),
            stroke: Some("#7f1d1d".to_string()),
            rx: Some(4.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: cmf_x + 180.0,
            y: cmf_y + 25.0,
            text: "!".to_string(),
            font_size: Some(18.0),
            fill: Some("#fff".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    // Title.
    shapes.push(Shape::Text(TextShape {
        x: STAGE_W / 2.0,
        y: 30.0,
        text: format!("Temperature Control \u{2014} {}", d.controller_name),
        font_size: Some(18.0),
        fill: Some("#111".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    let caption = format!(
        "t = {}h   T_in = {}\u{00b0}F   T_out = {}\u{00b0}F   Q = {} kW   in_band={}",
        to_fixed(d.tick.t_h, 2),
        to_fixed(d.tick.t_in_true, 2),
        to_fixed(d.tick.t_out_true, 1),
        to_fixed(d.tick.q, 2),
        bool_str(d.tick.in_band)
    );
    FrameParts::with_caption(shapes, caption)
}

fn spread_min(a: &[f64], b: &[f64]) -> f64 {
    a.iter().chain(b).copied().fold(f64::INFINITY, f64::min)
}

fn spread_max(a: &[f64], b: &[f64]) -> f64 {
    a.iter().chain(b).copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Convenience: build all frames + charts for a given [`RunResult`].
pub fn build_temp_control_animation(
    run: &RunResult,
    controller_name: &str,
    record_every: usize,
) -> (Vec<Frame>, Vec<ChartSpec>) {
    let t_target = run.cfg.t_target;
    let band = run.cfg.band.unwrap_or(2.0);
    let q_min = run.cfg.q_min;
    let q_max = run.cfg.q_max;
    // Build per-tick scene data.
    let mut frames: Vec<Frame> = Vec::new();
    let mut in_band_count = 0usize;
    let n = run.trace.len();
    for k in 0..n {
        if run.trace[k].in_band {
            in_band_count += 1;
        }
        if k % record_every != 0 && k != n - 1 {
            continue;
        }
        let tk = &run.trace[k];
        let data = SceneData {
            tick: tk,
            t_target,
            band,
            q_min,
            q_max,
            controller_name: controller_name.to_string(),
            energy: tk.energy_cum_k_wh,
            comfort_pct: in_band_count as f64 / (k as f64 + 1.0),
        };
        let f = build_temp_control_frame(tk.t_h, tk.tick, &data);
        frames.push(f.into_frame(tk.t_h, tk.tick));
    }

    // Charts.
    let t_h_arr: Vec<f64> = run.trace.iter().map(|r| r.t_h).collect();
    let first = t_h_arr.first().copied().unwrap_or(0.0);
    let last = t_h_arr.last().copied().unwrap_or(0.0);
    let charts = vec![
        ChartSpec {
            x: 40.0,
            y: 510.0,
            w: 600.0,
            h: 170.0,
            title: Some("Temperatures (\u{00b0}F)".to_string()),
            y_min: Some(spread_min(&run.t_out, &run.t_in) - 2.0),
            y_max: Some(spread_max(&run.t_out, &run.t_in) + 2.0),
            y_label: Some("\u{00b0}F".to_string()),
            series: vec![
                ChartSeries {
                    label: "T_in".to_string(),
                    color: COL_T_IN.to_string(),
                    t: t_h_arr.clone(),
                    y: run.t_in.clone(),
                },
                ChartSeries {
                    label: "T_out".to_string(),
                    color: COL_T_OUT.to_string(),
                    t: t_h_arr.clone(),
                    y: run.t_out.clone(),
                },
                ChartSeries {
                    label: "target".to_string(),
                    color: COL_TARGET.to_string(),
                    t: vec![first, last],
                    y: vec![t_target, t_target],
                },
            ],
            ..Default::default()
        },
        ChartSpec {
            x: 660.0,
            y: 510.0,
            w: 320.0,
            h: 170.0,
            title: Some(if q_min < 0.0 {
                "HVAC Q (kW): cooling < 0, heating > 0".to_string()
            } else {
                "Heater Q (kW)".to_string()
            }),
            y_min: Some(q_min - 0.2),
            y_max: Some(q_max + 0.2),
            y_label: Some("kW".to_string()),
            series: if q_min < 0.0 {
                vec![
                    ChartSeries {
                        label: "Q".to_string(),
                        color: COL_HEAT.to_string(),
                        t: t_h_arr.clone(),
                        y: run.q.clone(),
                    },
                    ChartSeries {
                        label: "zero".to_string(),
                        color: "#64748b".to_string(),
                        t: vec![first, last],
                        y: vec![0.0, 0.0],
                    },
                ]
            } else {
                vec![ChartSeries {
                    label: "Q".to_string(),
                    color: COL_HEAT.to_string(),
                    t: t_h_arr.clone(),
                    y: run.q.clone(),
                }]
            },
            ..Default::default()
        },
    ];
    (frames, charts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_caption_includes_in_band_flag() {
        let tk = TickRecord {
            t_h: 1.0,
            tick: 30.0,
            t_in_true: 68.5,
            t_out_true: 45.0,
            q: 2.5,
            in_band: false,
            energy_cum_k_wh: 1.2,
        };
        let d = SceneData {
            tick: &tk,
            t_target: 70.0,
            band: 2.0,
            q_min: 0.0,
            q_max: 5.0,
            controller_name: "PID".to_string(),
            energy: 1.2,
            comfort_pct: 0.5,
        };
        let fp = build_temp_control_frame(1.0, 30.0, &d);
        let cap = fp.caption.unwrap();
        assert!(cap.contains("in_band=false"));
        assert!(cap.contains("Q = 2.50 kW"));
    }

    #[test]
    fn animation_records_every_nth_and_last() {
        let trace: Vec<TickRecord> = (0..10)
            .map(|k| TickRecord {
                t_h: k as f64 * 0.1,
                tick: k as f64,
                in_band: true,
                ..Default::default()
            })
            .collect();
        let run = RunResult {
            cfg: RunConfig {
                t_target: 70.0,
                band: Some(2.0),
                q_min: 0.0,
                q_max: 5.0,
            },
            trace,
            t_in: vec![68.0; 10],
            t_out: vec![40.0; 10],
            q: vec![1.0; 10],
        };
        let (frames, charts) = build_temp_control_animation(&run, "PID", 5);
        // k = 0, 5, and last (9).
        assert_eq!(frames.len(), 3);
        assert_eq!(charts.len(), 2);
    }
}
