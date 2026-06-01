//! Port of `src/des/main-temp-control-anim.ts`.
//!
//! Generates an HTML animation of the temperature-control DES for a chosen
//! controller (`--controller bang-bang|pid|fuzzy|mdp-mpc`,
//! `--scenario winter|heat-cool`, `--out path`).
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
    build_temp_control_animation, RunConfig as SceneRunConfig, RunResult as SceneRunResult,
    TickRecord as SceneTick, STAGE_H, STAGE_W,
};
use crate::des::animation::types::{Frame, FrameParts};
use crate::des::general::temp_control::{
    run_temp_control, ControllerSpec, HouseParamsPartial, OutdoorPatternPartial, SimConfig,
    DEFAULT_HOUSE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    Winter,
    HeatCool,
}

impl Scenario {
    fn as_str(self) -> &'static str {
        match self {
            Scenario::Winter => "winter",
            Scenario::HeatCool => "heat-cool",
        }
    }

    fn title_suffix(self) -> &'static str {
        match self {
            Scenario::Winter => "winter heat",
            Scenario::HeatCool => "heat/cool",
        }
    }

    fn subtitle(self) -> &'static str {
        match self {
            Scenario::Winter => "24-hour winter scenario",
            Scenario::HeatCool => {
                "24-hour shoulder-season scenario with cold night heating and hot afternoon cooling"
            }
        }
    }
}

struct Args {
    controller: String,
    out: String,
    scenario: Scenario,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut controller = "pid".to_string();
    let mut scenario = Scenario::Winter;
    let mut out = std::path::Path::new("out")
        .join("temp-control")
        .join("animation.html")
        .to_string_lossy()
        .into_owned();
    let mut explicit_out = false;
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--controller" && i + 1 < argv.len() {
            i += 1;
            let v = argv[i].as_str();
            if ["bang-bang", "pid", "fuzzy", "mdp-mpc"].contains(&v) {
                controller = v.to_string();
            } else {
                eprintln!("unknown controller \"{}\"; using default", v);
                return Args {
                    controller,
                    out,
                    scenario,
                };
            }
        } else if argv[i] == "--scenario" && i + 1 < argv.len() {
            i += 1;
            match argv[i].as_str() {
                "winter" => scenario = Scenario::Winter,
                "heat-cool" | "heatcool" | "mixed" => scenario = Scenario::HeatCool,
                v => {
                    eprintln!("unknown scenario \"{}\"; using winter", v);
                    scenario = Scenario::Winter;
                }
            }
        } else if argv[i] == "--heat-cool" {
            scenario = Scenario::HeatCool;
        } else if argv[i] == "--out" && i + 1 < argv.len() {
            i += 1;
            out = argv[i].clone();
            explicit_out = true;
        } else if argv[i] == "-h" || argv[i] == "--help" {
            println!("Usage: main-temp-control-anim [--controller bang-bang|pid|fuzzy|mdp-mpc] [--scenario winter|heat-cool] [--out path]");
            return Args {
                controller,
                out,
                scenario,
            };
        }
        i += 1;
    }
    if !explicit_out && scenario == Scenario::HeatCool {
        out = std::path::Path::new("out")
            .join("temp-control")
            .join("animation-heat-cool.html")
            .to_string_lossy()
            .into_owned();
    }
    Args {
        controller,
        out,
        scenario,
    }
}

fn controller_spec(kind: &str, scenario: Scenario) -> ControllerSpec {
    match kind {
        "bang-bang" => ControllerSpec::BangBang,
        "fuzzy" => ControllerSpec::Fuzzy,
        "mdp-mpc" => ControllerSpec::MdpMpc {
            horizon_h: 6.0,
            n_levels: if scenario == Scenario::HeatCool {
                11
            } else {
                6
            },
            comfort_penalty: 0.5,
            cost_per_kwh: 0.15,
            track_weight: Some(1.0),
        },
        // "pid" and any default.
        _ => {
            if scenario == Scenario::HeatCool {
                ControllerSpec::Pid {
                    kp: 2.4,
                    ki: 0.35,
                    kd: 0.4,
                }
            } else {
                ControllerSpec::Pid {
                    kp: 3.0,
                    ki: 0.5,
                    kd: 0.5,
                }
            }
        }
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

fn build_config(scenario: Scenario, controller: ControllerSpec) -> SimConfig {
    match scenario {
        Scenario::Winter => SimConfig {
            t_target: 70.0,
            band: Some(2.0),
            duration_h: 24.0,
            dt_min: 1.0,
            controller,
            house: None,
            outdoor: None,
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: Some(0.2),
            forecast_noise_std: Some(1.5),
            forecast_horizon_h: Some(6.0),
            seed: Some(42),
        },
        Scenario::HeatCool => SimConfig {
            t_target: 70.0,
            band: Some(2.0),
            duration_h: 24.0,
            dt_min: 1.0,
            controller,
            house: Some(HouseParamsPartial {
                q_min: Some(-5.0),
                q_max: Some(5.0),
                t_init: Some(70.0),
                ..Default::default()
            }),
            outdoor: Some(OutdoorPatternPartial {
                mean: Some(70.0),
                amp: Some(22.0),
                phase: Some(9.0),
                noise_std: Some(1.0),
            }),
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: Some(0.2),
            forecast_noise_std: Some(1.0),
            forecast_horizon_h: Some(6.0),
            seed: Some(84),
        },
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    run_with(parse_args());
}

/// Run a preset scenario to a fixed output path (used by the site builder).
pub fn run_preset(scenario: Scenario, out: &str) {
    run_with(Args {
        controller: "pid".to_string(),
        out: out.to_string(),
        scenario,
    });
}

fn run_with(args: Args) {
    let cfg = build_config(
        args.scenario,
        controller_spec(&args.controller, args.scenario),
    );
    let t_target = cfg.t_target;
    let band = cfg.band.unwrap_or(2.0);
    let duration_h = cfg.duration_h;
    let q_min = cfg
        .house
        .as_ref()
        .and_then(|h| h.q_min)
        .unwrap_or(DEFAULT_HOUSE.q_min);
    let q_max = cfg
        .house
        .as_ref()
        .and_then(|h| h.q_max)
        .unwrap_or(DEFAULT_HOUSE.q_max);

    let t0 = Instant::now();
    let r = run_temp_control(cfg);
    let elapsed = t0.elapsed().as_millis();
    println!(
        "Simulated {}h {} scenario with {} in {}ms",
        duration_h,
        args.scenario.as_str(),
        args.controller,
        elapsed
    );
    println!("  HVAC energy = {:.2} kWh", r.energy_kwh);
    println!("  comfort = {:.1}%", 100.0 * r.comfort_pct);
    println!("  cost = ${:.2}", r.cost);

    // Build the animation. The scene module carries a trace-focused mirror of
    // the run result (TS used one shared type); adapt the model result into it.
    let ctl_name = controller_label(&args.controller);
    let record_every = 5usize; // 5-min frames → 24h × 12fps
    let scene_r = SceneRunResult {
        cfg: SceneRunConfig {
            t_target,
            band: Some(band),
            q_min,
            q_max,
        },
        trace: r
            .trace
            .iter()
            .map(|t| SceneTick {
                t_h: t.t_h,
                tick: t.tick as f64,
                t_in_true: t.t_in_true,
                t_out_true: t.t_out_true,
                q: t.q,
                in_band: t.in_band,
                energy_cum_k_wh: t.energy_cum_kwh,
            })
            .collect(),
        t_in: r.t_in.clone(),
        t_out: r.t_out.clone(),
        q: r.q.clone(),
    };
    let (frames, charts) = build_temp_control_animation(&scene_r, ctl_name, record_every);

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
        title: Some(format!(
            "Temperature Control — {} ({})",
            ctl_name,
            args.scenario.title_suffix()
        )),
        subtitle: Some(format!(
            "{}, target = {}°F ± {}°F   |   energy = {:.2} kWh, comfort = {:.1}%, cost = ${:.2}",
            args.scenario.subtitle(),
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
        let Frame {
            t,
            tick,
            shapes,
            caption,
        } = f;
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
