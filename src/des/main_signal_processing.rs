//! Port of `src/des/main-signal-processing.ts`.
//!
//! Small CLI demo and HTML player for the signal-transform models. For
//! JSON-driven runs use the from-json entry point.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`].
//!   - complex outputs use `general::signal_transforms::ComplexValue`.
//!   - `Number.prototype.toPrecision(6)` reproduced by [`to_precision`].

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::des::general::control_systems::lagrange::{
    generalized_acceleration, lagrange_to_state_space, LagrangeSecondOrderSystem,
    LagrangeStateSpace,
};
use crate::des::general::signal_transforms::{
    format_complex, run_dft_transform, run_fft_transform, run_fourier_transform,
    run_laplace_transform, run_mellin_transform, run_radon_transform, run_wavelet_transform,
    run_z_transform, ComplexPointInput, ComplexValue, DiscreteFourierTransformParams,
    FastFourierTransformParams, FourierTransformParams, LaplaceTransformParams,
    MellinTransformParams, QuadratureRule, RadonProjectionInput, RadonRunResult,
    RadonTransformParams, TransformContributionRecord, TransformRunResult, WaveletMother,
    WaveletPointInput, WaveletTransformParams, ZTransformParams,
};
use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

pub const SIGNAL_PROCESSING_PLAYER_REL_PATH: &str = "signal-processing/player.html";
pub const SIGNAL_PROCESSING_FRAMES_REL_PATH: &str = "signal-processing/player.frames.jsonl";
const MAX_FRAMES_PER_TRANSFORM: usize = 32;

#[derive(Clone, Debug)]
struct LagrangeRunResult {
    system: LagrangeSecondOrderSystem,
    state_space: LagrangeStateSpace,
    frames: Vec<Value>,
}

/// JS `Number.prototype.toPrecision(p)` — `p` significant digits.
fn to_precision(x: f64, p: usize) -> String {
    if x == 0.0 {
        return format!("{:.*}", p.saturating_sub(1), 0.0);
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -6 || exp >= p as i32 {
        format!("{:.*e}", p.saturating_sub(1), x)
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        format!("{:.*}", decimals, x)
    }
}

fn print_result(result: &TransformRunResult) {
    println!("\n{} TRANSFORM", result.kind.as_str().to_uppercase());
    println!("  {}", result.convention);
    println!(
        "  samples={} points={}",
        result.samples.len(),
        result.outputs.len()
    );
    println!(
        "  source={} stations={} sink={}",
        result.entity_framework.sources.join(", "),
        result.entity_framework.stations.join(" -> "),
        result.entity_framework.sinks.join(", ")
    );
    for output in &result.outputs {
        println!(
            "  {:<12} {}  |.|={}",
            output.label,
            format_complex(output.value, 6),
            to_precision(output.magnitude, 6)
        );
    }
}

fn sample_indices(len: usize, max: usize) -> Vec<usize> {
    if len <= max {
        return (0..len).collect();
    }
    let last = len - 1;
    let mut out = Vec::new();
    for i in 0..max {
        out.push((i * last + (max - 1) / 2) / (max - 1));
    }
    out.dedup();
    out
}

fn csub(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

fn cmag(z: ComplexValue) -> f64 {
    z.re.hypot(z.im)
}

fn shape_rect(x: f64, y: f64, w: f64, h: f64, fill: &str, stroke: &str) -> Value {
    json!({"kind":"rect","x":x,"y":y,"w":w,"h":h,"rx":4.0,"fill":fill,"stroke":stroke,"strokeWidth":1.0})
}

fn shape_line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) -> Value {
    json!({"kind":"line","x1":x1,"y1":y1,"x2":x2,"y2":y2,"stroke":stroke,"strokeWidth":width})
}

fn shape_circle(x: f64, y: f64, r: f64, fill: &str, stroke: &str) -> Value {
    json!({"kind":"circle","x":x,"y":y,"r":r,"fill":fill,"stroke":stroke,"strokeWidth":1.0})
}

fn shape_text(x: f64, y: f64, text: impl Into<String>, size: f64, fill: &str) -> Value {
    json!({"kind":"text","x":x,"y":y,"text":text.into(),"fontSize":size,"fill":fill,"anchor":"start"})
}

fn complex_xy(z: ComplexValue, scale: f64) -> (f64, f64) {
    let cx = 360.0;
    let cy = 240.0;
    (cx + 220.0 * z.re / scale, cy - 150.0 * z.im / scale)
}

fn transform_plot_scale(result: &TransformRunResult) -> f64 {
    let mut max_mag = 1.0_f64;
    for output in &result.outputs {
        max_mag = max_mag.max(output.magnitude);
    }
    for tr in &result.trace {
        max_mag = max_mag.max(cmag(tr.cumulative)).max(cmag(tr.contribution));
    }
    max_mag
}

fn base_transform_shapes(result: &TransformRunResult, scale: f64) -> Vec<Value> {
    let mut shapes = vec![
        shape_rect(0.0, 0.0, 720.0, 420.0, "#f8fafc", "#e2e8f0"),
        shape_line(80.0, 240.0, 640.0, 240.0, "#94a3b8", 1.0),
        shape_line(360.0, 60.0, 360.0, 370.0, "#94a3b8", 1.0),
        shape_text(
            28.0,
            34.0,
            format!("{} transform", result.kind.as_str()),
            18.0,
            "#0f172a",
        ),
        shape_text(
            28.0,
            58.0,
            format!(
                "samples={} points={}",
                result.samples.len(),
                result.outputs.len()
            ),
            12.0,
            "#475569",
        ),
        shape_text(622.0, 236.0, "Re", 11.0, "#64748b"),
        shape_text(368.0, 72.0, "Im", 11.0, "#64748b"),
    ];
    for output in &result.outputs {
        let (x, y) = complex_xy(output.value, scale);
        shapes.push(shape_circle(x, y, 4.0, "#bfdbfe", "#2563eb"));
    }
    shapes
}

fn transform_trace_frame(
    result: &TransformRunResult,
    tr: &TransformContributionRecord,
    tick: usize,
) -> Value {
    let scale = transform_plot_scale(result);
    let prev = csub(tr.cumulative, tr.contribution);
    let (px, py) = complex_xy(prev, scale);
    let (cx, cy) = complex_xy(tr.cumulative, scale);
    let mut shapes = base_transform_shapes(result, scale);
    shapes.push(shape_line(px, py, cx, cy, "#f97316", 2.0));
    shapes.push(shape_line(360.0, 240.0, cx, cy, "#2563eb", 2.4));
    shapes.push(shape_circle(cx, cy, 7.0, "#f97316", "#9a3412"));
    shapes.push(shape_text(
        28.0,
        390.0,
        format!(
            "{} sample {} -> {}",
            result.kind.as_str(),
            tr.sample_index,
            tr.point_label
        ),
        13.0,
        "#0f172a",
    ));
    json!({
        "tick": tick,
        "t": tick as f64,
        "transform": result.kind.as_str(),
        "sampleIndex": tr.sample_index as f64,
        "pointIndex": tr.point_index as f64,
        "real": tr.cumulative.re,
        "imag": tr.cumulative.im,
        "magnitude": cmag(tr.cumulative),
        "shapes": shapes,
        "caption": format!("{}: sample {} contribution accumulated at {}", result.kind.as_str(), tr.sample_index, tr.point_label),
    })
}

fn transform_output_frame(result: &TransformRunResult, output_index: usize, tick: usize) -> Value {
    let scale = transform_plot_scale(result);
    let output = &result.outputs[output_index];
    let (x, y) = complex_xy(output.value, scale);
    let mut shapes = base_transform_shapes(result, scale);
    shapes.push(shape_line(360.0, 240.0, x, y, "#16a34a", 2.8));
    shapes.push(shape_circle(x, y, 8.0, "#22c55e", "#166534"));
    shapes.push(shape_text(
        28.0,
        390.0,
        format!("{} final {}", result.kind.as_str(), output.label),
        13.0,
        "#0f172a",
    ));
    json!({
        "tick": tick,
        "t": tick as f64,
        "transform": result.kind.as_str(),
        "sampleIndex": output.samples_used as f64,
        "pointIndex": output.point_index as f64,
        "real": output.value.re,
        "imag": output.value.im,
        "magnitude": output.magnitude,
        "shapes": shapes,
        "caption": format!("{}: final {} = {}", result.kind.as_str(), output.label, format_complex(output.value, 5)),
    })
}

fn transform_frames(result: &TransformRunResult, tick_start: usize) -> Vec<Value> {
    if result.trace.is_empty() {
        return (0..result.outputs.len())
            .map(|i| transform_output_frame(result, i, tick_start + i))
            .collect();
    }
    sample_indices(result.trace.len(), MAX_FRAMES_PER_TRANSFORM)
        .into_iter()
        .enumerate()
        .map(|(frame_i, trace_i)| {
            transform_trace_frame(result, &result.trace[trace_i], tick_start + frame_i)
        })
        .collect()
}

fn transform_result_json(result: &TransformRunResult) -> Value {
    json!({
        "kind": result.kind.as_str(),
        "convention": &result.convention,
        "samples": result.samples.len(),
        "points": result.outputs.len(),
        "validationPassed": result.validation.iter().filter(|c| c.passed).count(),
        "validationTotal": result.validation.len(),
        "outputs": result.outputs.iter().map(|o| json!({
            "label": &o.label,
            "point": {"re": o.point.re, "im": o.point.im},
            "value": {"re": o.value.re, "im": o.value.im},
            "magnitude": o.magnitude,
            "phase": o.phase,
            "absoluteError": o.absolute_error,
        })).collect::<Vec<_>>(),
    })
}

fn radon_frames(result: &RadonRunResult, tick_start: usize) -> Vec<Value> {
    let max_value = result
        .outputs
        .iter()
        .map(|o| o.value.abs())
        .fold(1.0_f64, f64::max);
    result
        .outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            let mut shapes = vec![
                shape_rect(0.0, 0.0, 720.0, 420.0, "#f8fafc", "#e2e8f0"),
                shape_text(28.0, 34.0, "radon transform", 18.0, "#0f172a"),
                shape_text(
                    28.0,
                    58.0,
                    format!("grid={}x{}", result.width, result.height),
                    12.0,
                    "#475569",
                ),
                shape_rect(120.0, 90.0, 260.0, 260.0, "#ffffff", "#94a3b8"),
            ];
            let theta = output.theta;
            let rho = output.rho.clamp(-2.0, 2.0) * 50.0;
            let cx = 250.0 + rho * theta.cos();
            let cy = 220.0 - rho * theta.sin();
            let dx = 150.0 * (-theta.sin());
            let dy = 150.0 * (-theta.cos());
            shapes.push(shape_line(cx - dx, cy - dy, cx + dx, cy + dy, "#dc2626", 3.0));
            let bar_h = 220.0 * output.value.abs() / max_value;
            shapes.push(shape_rect(475.0, 330.0 - bar_h, 70.0, bar_h, "#60a5fa", "#1d4ed8"));
            shapes.push(shape_text(455.0, 360.0, output.label.clone(), 12.0, "#0f172a"));
            shapes.push(shape_text(
                455.0,
                382.0,
                format!("value={}", to_precision(output.value, 5)),
                12.0,
                "#475569",
            ));
            json!({
                "tick": tick_start + i,
                "t": (tick_start + i) as f64,
                "transform": "radon",
                "sampleIndex": output.cells_used as f64,
                "pointIndex": output.point_index as f64,
                "real": output.value,
                "imag": 0.0,
                "magnitude": output.value.abs(),
                "shapes": shapes,
                "caption": format!("radon: projection {} integrates {} grid cells", output.label, output.cells_used),
            })
        })
        .collect()
}

fn radon_result_json(result: &RadonRunResult) -> Value {
    json!({
        "kind": result.kind.as_str(),
        "convention": &result.convention,
        "width": result.width,
        "height": result.height,
        "validationPassed": result.validation.iter().filter(|c| c.passed).count(),
        "validationTotal": result.validation.len(),
        "outputs": result.outputs.iter().map(|o| json!({
            "label": &o.label,
            "theta": o.theta,
            "rho": o.rho,
            "value": o.value,
            "cellsUsed": o.cells_used,
            "absoluteError": o.absolute_error,
        })).collect::<Vec<_>>(),
    })
}

fn lagrange_shapes(q: f64, qd: f64, qdd: f64, u: f64, energy: f64, t: f64) -> Vec<Value> {
    let wall_x = 92.0;
    let track_y = 220.0;
    let base_x = 315.0;
    let mass_x = base_x + 92.0 * q.clamp(-1.2, 1.2);
    let mass_y = track_y - 35.0;
    let mass_w = 92.0;
    let mass_h = 70.0;
    let spring_start = wall_x + 20.0;
    let spring_end = mass_x;
    let segments = 9;
    let mut spring_points = Vec::new();
    for i in 0..=segments {
        let x = spring_start + (spring_end - spring_start) * i as f64 / segments as f64;
        let y = track_y
            + if i == 0 || i == segments {
                0.0
            } else if i % 2 == 0 {
                -18.0
            } else {
                18.0
            };
        spring_points.push((x, y));
    }

    let mut shapes = vec![
        shape_rect(0.0, 0.0, 720.0, 420.0, "#f8fafc", "#e2e8f0"),
        shape_text(28.0, 34.0, "lagrange -> state space", 18.0, "#0f172a"),
        shape_text(
            28.0,
            58.0,
            "M qdd + C qd + K q = B u, x = [q, qd]",
            12.0,
            "#475569",
        ),
        shape_line(70.0, track_y + 46.0, 610.0, track_y + 46.0, "#94a3b8", 2.0),
        shape_rect(wall_x, track_y - 72.0, 20.0, 144.0, "#e2e8f0", "#64748b"),
    ];
    for pair in spring_points.windows(2) {
        let (x1, y1) = pair[0];
        let (x2, y2) = pair[1];
        shapes.push(shape_line(x1, y1, x2, y2, "#64748b", 2.0));
    }
    shapes.push(shape_rect(
        mass_x, mass_y, mass_w, mass_h, "#bfdbfe", "#1d4ed8",
    ));
    shapes.push(shape_text(
        mass_x + 24.0,
        mass_y + 41.0,
        "mass",
        13.0,
        "#0f172a",
    ));
    let force_len = (u / 22.0).clamp(-1.0, 1.0) * 85.0;
    shapes.push(shape_line(
        mass_x + mass_w / 2.0,
        mass_y - 18.0,
        mass_x + mass_w / 2.0 + force_len,
        mass_y - 18.0,
        "#dc2626",
        3.0,
    ));
    shapes.push(shape_circle(
        mass_x + mass_w / 2.0 + force_len,
        mass_y - 18.0,
        5.0,
        "#dc2626",
        "#991b1b",
    ));
    shapes.push(shape_text(430.0, 98.0, "state-space rows", 12.0, "#475569"));
    shapes.push(shape_text(430.0, 122.0, "qdot = qd", 13.0, "#0f172a"));
    shapes.push(shape_text(
        430.0,
        146.0,
        "qdd = -4q - 0.5qd + 0.5u",
        13.0,
        "#0f172a",
    ));

    let bars = [
        ("q", q, "#2563eb"),
        ("qd", qd, "#16a34a"),
        ("qdd", qdd, "#ea580c"),
        ("u", u / 10.0, "#dc2626"),
        ("E", energy / 6.0, "#9333ea"),
    ];
    for (i, (label, value, color)) in bars.iter().enumerate() {
        let y = 305.0 + i as f64 * 18.0;
        let w = 92.0 * value.abs().min(1.0);
        let x = if *value >= 0.0 { 168.0 } else { 168.0 - w };
        shapes.push(shape_text(28.0, y + 4.0, *label, 11.0, "#475569"));
        shapes.push(shape_line(168.0, y, 168.0, y + 9.0, "#cbd5e1", 1.0));
        shapes.push(shape_rect(x, y - 5.0, w, 10.0, color, color));
    }
    shapes.push(shape_text(
        430.0,
        346.0,
        format!(
            "t={} q={} qd={}",
            to_precision(t, 4),
            to_precision(q, 4),
            to_precision(qd, 4)
        ),
        13.0,
        "#0f172a",
    ));
    shapes
}

fn lagrange_example() -> LagrangeRunResult {
    let system = LagrangeSecondOrderSystem {
        mass: vec![vec![2.0]],
        damping: vec![vec![1.0]],
        stiffness: vec![vec![8.0]],
        input: vec![vec![1.0]],
        force_bias: Some(vec![0.0]),
    };
    let state_space = lagrange_to_state_space(&system);
    let dt = 0.05;
    let q_ref = 1.0;
    let kp = 18.0;
    let kd = 4.0;
    let mut q = -0.75;
    let mut qd = 0.0;
    let mut frames = Vec::new();
    for tick in 0..72 {
        let t = tick as f64 * dt;
        let u = kp * (q_ref - q) - kd * qd;
        let qdd = generalized_acceleration(&system, &[q], &[qd], &[u])[0];
        let energy = 0.5 * system.mass[0][0] * qd * qd + 0.5 * system.stiffness[0][0] * q * q;
        let shapes = lagrange_shapes(q, qd, qdd, u, energy, t);
        frames.push(json!({
            "tick": tick,
            "t": t,
            "transform": "lagrange",
            "method": "lagrange",
            "position": q,
            "velocity": qd,
            "acceleration": qdd,
            "control": u,
            "energy": energy,
            "stateNorm": (q * q + qd * qd).sqrt(),
            "shapes": shapes,
            "caption": format!("lagrange: q={} qd={} u={}", to_precision(q, 4), to_precision(qd, 4), to_precision(u, 4)),
        }));
        qd += qdd * dt;
        q += qd * dt;
    }
    LagrangeRunResult {
        system,
        state_space,
        frames,
    }
}

fn lagrange_result_json(result: &LagrangeRunResult) -> Value {
    json!({
        "kind": "lagrange",
        "equation": "M qdd + C qd + K q = B u + bias",
        "mass": result.system.mass,
        "damping": result.system.damping,
        "stiffness": result.system.stiffness,
        "input": result.system.input,
        "stateSpace": {
            "a": result.state_space.a,
            "b": result.state_space.b,
            "bias": result.state_space.bias,
        },
        "frames": result.frames.len(),
    })
}

fn example_runs() -> (Vec<TransformRunResult>, RadonRunResult) {
    let z = run_z_transform(ZTransformParams {
        sequence: Some(vec![1.0, 0.5, 0.25, 0.125, 0.0625]),
        z_values: Some(vec![
            ComplexPointInput {
                label: Some("z=2".to_string()),
                re: 2.0,
                im: None,
            },
            ComplexPointInput {
                label: Some("z=1".to_string()),
                re: 1.0,
                im: None,
            },
        ]),
        ..Default::default()
    });

    let mut laplace_constants = HashMap::new();
    laplace_constants.insert("a".to_string(), 2.0);
    let laplace = run_laplace_transform(LaplaceTransformParams {
        expression: Some("exp(-a*t)".to_string()),
        constants: Some(laplace_constants),
        t0: Some(0.0),
        t1: Some(8.0),
        dt: Some(0.02),
        s_values: Some(vec![
            ComplexPointInput {
                label: Some("s=1".to_string()),
                re: 1.0,
                im: None,
            },
            ComplexPointInput {
                label: Some("s=0.5+i".to_string()),
                re: 0.5,
                im: Some(1.0),
            },
        ]),
        ..Default::default()
    });

    let mut fourier_constants = HashMap::new();
    fourier_constants.insert("omega0".to_string(), 2.0);
    let fourier = run_fourier_transform(FourierTransformParams {
        expression: Some("sin(omega0*t)".to_string()),
        constants: Some(fourier_constants),
        t0: Some(0.0),
        t1: Some(2.0 * std::f64::consts::PI),
        dt: Some(2.0 * std::f64::consts::PI / 160.0),
        omega_values: Some(vec![0.0, 2.0, -2.0]),
        ..Default::default()
    });

    let dft = run_dft_transform(DiscreteFourierTransformParams {
        sequence: Some(vec![1.0, 0.0, -1.0, 0.0]),
        k_values: Some(vec![0, 1, 2, 3]),
        ..Default::default()
    });

    let fft = run_fft_transform(FastFourierTransformParams {
        sequence: Some(vec![1.0, 0.0, -1.0, 0.0]),
        ..Default::default()
    });

    let wavelet = run_wavelet_transform(WaveletTransformParams {
        expression: Some("t".to_string()),
        t0: Some(0.0),
        t1: Some(1.0),
        dt: Some(0.05),
        quadrature: Some(QuadratureRule::Rectangular),
        mother: Some(WaveletMother::Haar),
        scale_shift_values: Some(vec![
            WaveletPointInput {
                label: Some("a=1,b=0".to_string()),
                scale: 1.0,
                shift: 0.0,
            },
            WaveletPointInput {
                label: Some("a=0.5,b=0.25".to_string()),
                scale: 0.5,
                shift: 0.25,
            },
        ]),
        ..Default::default()
    });

    let mellin = run_mellin_transform(MellinTransformParams {
        expression: Some("1".to_string()),
        t0: Some(1.0),
        t1: Some(3.0),
        dt: Some(0.02),
        quadrature: Some(QuadratureRule::Trapezoid),
        s_values: Some(vec![ComplexPointInput {
            label: Some("s=1".to_string()),
            re: 1.0,
            im: None,
        }]),
        ..Default::default()
    });

    let radon = run_radon_transform(RadonTransformParams {
        grid: vec![
            vec![0.0, 0.1, 0.0, 0.0, 0.0],
            vec![0.0, 0.3, 0.7, 0.2, 0.0],
            vec![0.1, 0.6, 1.0, 0.6, 0.1],
            vec![0.0, 0.2, 0.7, 0.3, 0.0],
            vec![0.0, 0.0, 0.0, 0.1, 0.0],
        ],
        projections: Some(vec![
            RadonProjectionInput {
                label: Some("vertical".to_string()),
                theta: 0.0,
                rho: 0.0,
            },
            RadonProjectionInput {
                label: Some("horizontal".to_string()),
                theta: std::f64::consts::FRAC_PI_2,
                rho: 0.0,
            },
            RadonProjectionInput {
                label: Some("diagonal".to_string()),
                theta: std::f64::consts::FRAC_PI_4,
                rho: 0.0,
            },
        ]),
        line_width: Some(1.1),
        ..Default::default()
    });

    (vec![z, laplace, fourier, dft, fft, wavelet, mellin], radon)
}

pub fn signal_processing_artifact() -> RunArtifact {
    let (runs, radon) = example_runs();
    let lagrange = lagrange_example();
    let mut frames = Vec::new();
    for result in &runs {
        let mut next = transform_frames(result, frames.len());
        frames.append(&mut next);
    }
    let mut radon_next = radon_frames(&radon, frames.len());
    frames.append(&mut radon_next);
    let lagrange_tick_start = frames.len();
    for (i, frame) in lagrange.frames.iter().enumerate() {
        let mut frame = frame.clone();
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("tick".to_string(), json!(lagrange_tick_start + i));
            obj.insert("t".to_string(), json!((lagrange_tick_start + i) as f64));
        }
        frames.push(frame);
    }
    let results = json!({
        "kind": "signal-control-analysis",
        "transforms": runs.iter().map(transform_result_json).collect::<Vec<_>>(),
        "radon": radon_result_json(&radon),
        "lagrange": lagrange_result_json(&lagrange),
    });
    let summary = format!(
        "Animated {} transform families plus Lagrange state-space dynamics over {} frames.",
        runs.len() + 1,
        frames.len()
    );
    RunArtifact::sim(
        "signal-processing-transforms",
        "Signal & Control Analysis Methods",
        "Animated Z, Laplace, Fourier, DFT/FFT, wavelet, Mellin, Radon, and Lagrange state-space analyses.",
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &[
                    "all",
                    "magnitude",
                    "real",
                    "imag",
                    "position",
                    "velocity",
                    "acceleration",
                    "control",
                    "energy",
                    "stateNorm",
                ],
                "all",
                Some("metric"),
            ),
            UiControl::toggle("show-magnitude", "Magnitude", true, Some("magnitude")),
            UiControl::toggle("show-real", "Real", true, Some("real")),
            UiControl::toggle("show-imag", "Imaginary", true, Some("imag")),
            UiControl::toggle("show-position", "Position", true, Some("position")),
            UiControl::toggle("show-velocity", "Velocity", true, Some("velocity")),
            UiControl::toggle("show-control", "Control", true, Some("control")),
        ],
        &summary,
    )
}

pub fn write_signal_processing_player_html(out_root: impl AsRef<Path>) -> io::Result<PathBuf> {
    let artifact = signal_processing_artifact();
    let root = out_root.as_ref();
    let html_path = root.join(SIGNAL_PROCESSING_PLAYER_REL_PATH);
    let frames_path = root.join(SIGNAL_PROCESSING_FRAMES_REL_PATH);
    if let Some(dir) = html_path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&html_path, artifact.to_player_html())?;
    fs::write(&frames_path, artifact.to_jsonl())?;
    Ok(html_path)
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let (runs, radon) = example_runs();
    for result in &runs {
        print_result(result);
    }
    println!("\nRADON TRANSFORM");
    println!("  {}", radon.convention);
    println!(
        "  grid={}x{} points={}",
        radon.width,
        radon.height,
        radon.outputs.len()
    );
    for output in &radon.outputs {
        println!(
            "  {:<12} value={} cells={}",
            output.label,
            to_precision(output.value, 6),
            output.cells_used
        );
    }
    match write_signal_processing_player_html("out") {
        Ok(path) => println!("Wrote player: {}", path.display()),
        Err(err) => eprintln!("Could not write signal-processing player: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_processing_artifact_has_transform_player() {
        let artifact = signal_processing_artifact();
        assert!(artifact.frames.len() >= 8);
        assert_eq!(artifact.kind, "signal-processing-transforms");
        let html = artifact.to_player_html();
        assert!(html.contains("Signal &amp; Control Analysis Methods"));
        assert!(html.contains("radon"));
        assert!(html.contains("lagrange"));
        assert!(artifact.to_jsonl().contains("\"magnitude\""));
        assert!(artifact.to_jsonl().contains("\"control\""));
    }
}
