//! Port of `src/des/main-temp-control-anim.ts`.
//!
//! Generates an HTML animation of the temperature-control DES for a chosen
//! controller (`--controller bang-bang|pid|fuzzy|mdp-mpc`, `--out path`).
//!
//! Conversion notes:
//!   - `process.argv` flag parsing → `std::env::args`; `process.exit` →
//!     `std::process::exit`.
//!   - the controller union → [`ControllerSpec`]; delegates the simulation to
//!     `general::temp_control` and the rendering to
//!     `animation::scenes::temp_control_scene` (both already ported).

use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::temp_control_scene::{
    build_temp_control_animation, STAGE_H, STAGE_W,
};
use crate::des::animation::types::{Frame, FrameParts};
use crate::des::general::temp_control::{run_temp_control, ControllerSpec, SimConfig};

struct Args {
    controller: String,
    out: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut controller = "pid".to_string();
    let mut out = std::path::Path::new("out")
        .join("temp-control")
        .join("animation.html")
        .to_string_lossy()
        .into_owned();
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--controller" && i + 1 < argv.len() {
            i += 1;
            let v = argv[i].as_str();
            if ["bang-bang", "pid", "fuzzy", "mdp-mpc"].contains(&v) {
                controller = v.to_string();
            } else {
                eprintln!("unknown controller \"{}\"", v);
                std::process::exit(1);
            }
        } else if argv[i] == "--out" && i + 1 < argv.len() {
            i += 1;
            out = argv[i].clone();
        } else if argv[i] == "-h" || argv[i] == "--help" {
            println!("Usage: main-temp-control-anim [--controller bang-bang|pid|fuzzy|mdp-mpc] [--out path]");
            std::process::exit(0);
        }
        i += 1;
    }
    Args { controller, out }
}

fn controller_spec(kind: &str) -> ControllerSpec {
    match kind {
        "bang-bang" => ControllerSpec::BangBang,
        "fuzzy" => ControllerSpec::Fuzzy,
        "mdp-mpc" => ControllerSpec::MdpMpc {
            horizon_h: 6.0,
            n_levels: 6,
            comfort_penalty: 0.5,
            cost_per_kwh: 0.15,
            track_weight: Some(1.0),
        },
        // "pid" and any default.
        _ => ControllerSpec::Pid { kp: 3.0, ki: 0.5, kd: 0.5 },
    }
}

fn controller_label(kind: &str) -> &'static str {
    match kind {
        "bang-bang" => "Bang-bang",
        "fuzzy" => "Fuzzy-PI (Mamdani)",
        "mdp-mpc" => "MDP-MPC (H=6h)",
        _ => "PID (filtered-D)",
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let args = parse_args();
    let cfg = SimConfig {
        t_target: 70.0,
        band: Some(2.0),
        duration_h: 24.0,
        dt_min: 1.0,
        controller: controller_spec(&args.controller),
        house: None,
        outdoor: None,
        cost_per_kwh: 0.15,
        comfort_penalty: 0.5,
        sensor_noise_std: Some(0.2),
        forecast_noise_std: Some(1.5),
        forecast_horizon_h: Some(6.0),
        seed: Some(42),
    };
    let t_target = cfg.t_target;
    let band = cfg.band.unwrap_or(2.0);
    let duration_h = cfg.duration_h;

    let t0 = Instant::now();
    let r = run_temp_control(cfg);
    let elapsed = t0.elapsed().as_millis();
    println!("Simulated {}h with {} in {}ms", duration_h, args.controller, elapsed);
    println!("  energy = {:.2} kWh", r.energy_kwh);
    println!("  comfort = {:.1}%", 100.0 * r.comfort_pct);
    println!("  cost = ${:.2}", r.cost);

    // Build the animation.
    let ctl_name = controller_label(&args.controller);
    let record_every = 5usize; // 5-min frames → 24h × 12fps
    let (frames, charts) = build_temp_control_animation(&r, ctl_name, record_every);

    let frames_path = if let Some(stripped) = args.out.strip_suffix(".html") {
        format!("{}.frames.jsonl", stripped)
    } else {
        format!("{}.frames.jsonl", args.out)
    };
    let mut recorder = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.clone(),
        html_path: Some(args.out.clone()),
        width: STAGE_W,
        height: STAGE_H,
        fps: Some(12.0),
        title: Some(format!("Temperature Control — {}", ctl_name)),
        subtitle: Some(format!(
            "24-hour winter scenario, target = {}°F ± {}°F   |   energy = {:.2} kWh, comfort = {:.1}%, cost = ${:.2}",
            jn(t_target),
            jn(band),
            r.energy_kwh,
            100.0 * r.comfort_pct,
            r.cost
        )),
        background: Some("#f9fafb".to_string()),
        ..Default::default()
    })
    .expect("create frame recorder");

    for f in frames {
        let Frame { t, tick, shapes, caption } = f;
        recorder.frame(t, tick, move || FrameParts { shapes, caption });
    }
    recorder.set_charts(charts);
    recorder.finish().expect("finish recorder");

    println!();
    println!("Frames: {}", frames_path);
    println!("HTML:   {}", args.out);
    let abs = std::fs::canonicalize(&args.out)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| args.out.clone());
    println!("Open in browser: file://{}", abs);
}

/// Integer-valued floats without a trailing `.0` (mirrors JS `${number}`).
fn jn(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}
