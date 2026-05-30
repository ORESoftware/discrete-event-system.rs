//! Port of `src/des/main-temp-control.ts`.
//!
//! CLI driver comparing bang-bang / PID / fuzzy-PI / MDP-MPC controllers on one
//! scenario, an MDP-MPC sensitivity sweep, a stress test, and a CSV dump.
//!
//! Conversion notes:
//!   - controller specs → `general::temp_control::ControllerSpec`.
//!   - noisy outdoor sampling is seeded inside `general::temp_control`.
//!   - `fs.writeFileSync` → `std::fs`; top-level `main()` → [`run`].

use std::time::Instant;

use crate::des::general::temp_control::{
    run_temp_control, ControllerSpec, OutdoorPatternPartial, RunResult, SimConfig,
};

fn header(s: &str) {
    println!();
    println!("{}", "═".repeat(120));
    println!("  {}", s);
    println!("{}", "═".repeat(120));
}

fn fmt(r: &RunResult, name: &str) -> String {
    let min_t = r.t_in.iter().copied().fold(f64::INFINITY, f64::min);
    let max_t = r.t_in.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    [
        format!("{:<22}", name),
        format!("energy={:>7} kWh", format!("{:.2}", r.energy_kwh)),
        format!("comfort={:>5}%", format!("{:.1}", 100.0 * r.comfort_pct)),
        format!("violation={:>6} °F·h", format!("{:.2}", r.violation_fh)),
        format!("cost=${:>6}", format!("{:.2}", r.cost)),
        format!("T_in=[{:.2}, {:.2}]", min_t, max_t),
    ]
    .join("   ")
}

/// `name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')`.
fn safe_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let mut out = String::new();
    let mut last_dash = false;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    const SEED: u32 = 42;
    let cfg = SimConfig {
        t_target: 70.0,
        band: Some(2.0),
        duration_h: 24.0,
        dt_min: 1.0,
        cost_per_kwh: 0.15,
        comfort_penalty: 0.5,
        sensor_noise_std: Some(0.2),
        forecast_noise_std: Some(1.5),
        forecast_horizon_h: Some(6.0),
        seed: Some(SEED),
        controller: ControllerSpec::BangBang,
        house: None,
        outdoor: None,
    };

    header("TEMPERATURE CONTROL — 24-hour winter day, T_target = 70°F ± 2°F");
    println!("  Outdoor: cold winter day. Mean 25°F, ±15°F diurnal swing, 1.5°F noise.");
    println!("           Coldest at 3 AM (≈ 10°F), warmest at 3 PM (≈ 40°F).");
    println!("  House:   τ = 12 h thermal time constant, heater max 5 kW.");
    println!("  Sensors: indoor sensor noise σ = 0.2°F, forecast noise σ = 1.5°F.");
    println!("  Cost:    $0.15/kWh energy + $0.50/(°F)²/h comfort violation outside band.");

    header("CONTROLLER COMPARISON (same scenario, same seed)");
    let mut runs: Vec<(String, RunResult)> = Vec::new();
    let comparison: Vec<(&str, ControllerSpec)> = vec![
        ("bang-bang", ControllerSpec::BangBang),
        ("PID (filtered-D)", ControllerSpec::Pid { kp: 3.0, ki: 0.5, kd: 0.5 }),
        ("Fuzzy-PI (Mamdani)", ControllerSpec::Fuzzy),
        ("MDP-MPC (H=1h)", ControllerSpec::MdpMpc { horizon_h: 1.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: Some(1.0) }),
        ("MDP-MPC (H=6h)", ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: Some(1.0) }),
        ("MDP-MPC (loose,w=0.05)", ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: Some(0.05) }),
    ];
    for (name, spec) in comparison {
        let mut c = cfg.clone();
        c.controller = spec;
        let t0 = Instant::now();
        let r = run_temp_control(c);
        let dt = t0.elapsed().as_millis();
        println!("  {}   wall={}ms", fmt(&r, name), dt);
        runs.push((name.to_string(), r));
    }

    header("MDP-MPC SENSITIVITY: forecast horizon × tracking weight");
    println!("  Demonstrates the energy/comfort frontier the MDP discovers when given more lookahead");
    println!("  or different relative penalties. With weak tracking (w=0.05) the controller saves");
    println!("  energy by riding closer to the band edges; with strong tracking (w=1.0) it stays");
    println!("  near the centre at slightly higher cost.");
    println!();
    println!(
        "  {:<11}{:<13}  {:>11}  {:>9}  {:>11}  {:>8}",
        "horizon_h", "trackWeight", "energy_kWh", "comfort%", "violation", "cost_$"
    );
    for h in [1.0_f64, 2.0, 4.0, 6.0] {
        for w in [0.05_f64, 0.5, 1.0, 2.0] {
            let mut c = cfg.clone();
            c.controller = ControllerSpec::MdpMpc {
                horizon_h: h,
                n_levels: 6,
                comfort_penalty: 0.5,
                cost_per_kwh: 0.15,
                track_weight: Some(w),
            };
            let r = run_temp_control(c);
            println!(
                "  {:<11}{:<13}  {:>11}  {:>9}  {:>11}  {:>8}",
                format!("{}", h as i64),
                format!("{:.2}", w),
                format!("{:.2}", r.energy_kwh),
                format!("{:.1}", 100.0 * r.comfort_pct),
                format!("{:.3}", r.violation_fh),
                format!("{:.2}", r.cost)
            );
        }
    }

    header("STRESS TEST: tight ±1°F band, harder weather");
    let mut stress = cfg.clone();
    stress.band = Some(1.0);
    stress.comfort_penalty = 2.0;
    stress.outdoor = Some(OutdoorPatternPartial {
        mean: Some(15.0),
        amp: Some(20.0),
        phase: Some(9.0),
        noise_std: Some(2.5),
    });
    let stress_controllers: Vec<(&str, ControllerSpec)> = vec![
        ("bang-bang", ControllerSpec::BangBang),
        ("PID", ControllerSpec::Pid { kp: 5.0, ki: 1.0, kd: 1.0 }),
        ("Fuzzy-PI", ControllerSpec::Fuzzy),
        ("MDP-MPC (H=6h, w=1)", ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 2.0, cost_per_kwh: 0.15, track_weight: Some(1.0) }),
    ];
    for (name, spec) in stress_controllers {
        let mut c = stress.clone();
        c.controller = spec;
        let r = run_temp_control(c);
        println!("  {}", fmt(&r, name));
    }

    header("SAVE TIME-SERIES TRACES TO out/temp-control/");
    let out_dir = std::path::Path::new("out").join("temp-control");
    let _ = std::fs::create_dir_all(&out_dir);
    for (name, result) in &runs {
        let safe = safe_name(name);
        let mut lines: Vec<String> =
            vec!["tick,t_h,T_out,T_in,Q,energy_cum_kWh,error,in_band,violation_Fh".to_string()];
        for r in &result.trace {
            lines.push(format!(
                "{},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.4}",
                r.tick,
                r.t_h,
                r.t_out_true,
                r.t_in_true,
                r.q,
                r.energy_cum_kwh,
                r.error,
                if r.in_band { 1 } else { 0 },
                r.violation_fh
            ));
        }
        let f = out_dir.join(format!("{}.csv", safe));
        let _ = std::fs::write(&f, lines.join("\n"));
        println!("  {} → {}", format!("{:<22}", name), f.display());
    }
    println!();
}
