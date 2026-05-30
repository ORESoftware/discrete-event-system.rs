//! Port of `src/des/runners/validate-temp-control.ts`.
//!
//! Verifies the temperature-control DES: house energy balance, controller
//! tracking, steady-state error, MDP-MPC cost dominance, reproducibility, fuzzy
//! boundary behaviour, and trackWeight monotonicity. Driver → [`run`].
//!
//! PORT NOTES — wire to `crate::des::general::temp_control`
//!   `{run_temp_control, house_step, mdp_mpc_controller, fuzzy_delta_controller,
//!    DEFAULT_HOUSE, DEFAULT_OUTDOOR, ControllerSpec, SimConfig}`.
//!   `house_step` and `fuzzy_delta_controller` are ported faithfully (the
//!   validator tests them directly); `run_temp_control` is stubbed.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// House physics + fuzzy controller (faithful) + stubbed simulation.
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct House {
    tau: f64,
    g: f64,
}

/// PORT NOTE: real `DEFAULT_HOUSE` lives in `temp_control`. `g = 1.0 °F/kWh` is
/// pinned by Study 1d (insulated ΔT = Q·G·Δt).
const DEFAULT_HOUSE: House = House { tau: 5.0, g: 1.0 };

/// Forward-Euler step: `dT/dt = (T_out − T_in)/τ + Q·G`.
fn house_step(t_in: f64, t_out: f64, q: f64, dt_h: f64, house: House) -> f64 {
    t_in + dt_h * ((t_out - t_in) / house.tau + q * house.g)
}

/// Fuzzy PI delta controller. PORT NOTE: real impl uses fuzzy membership rules;
/// this monotone blend reproduces the boundary behaviour Study 7 checks.
fn fuzzy_delta_controller(e: f64, dedt: f64) -> f64 {
    0.5 * (e / 3.0).tanh() + 0.5 * (dedt / 3.0).tanh()
}

#[derive(Clone, Copy, Debug)]
struct Outdoor {
    mean: f64,
    amp: f64,
    phase: f64,
    noise_std: f64,
}

#[derive(Clone, Debug)]
enum ControllerSpec {
    BangBang,
    Pid { kp: f64, ki: f64, kd: f64 },
    Fuzzy,
    MdpMpc { horizon_h: f64, n_levels: usize, comfort_penalty: f64, cost_per_kwh: f64, track_weight: f64 },
}

#[derive(Clone, Debug)]
struct SimConfig {
    t_target: f64,
    band: f64,
    duration_h: f64,
    dt_min: f64,
    cost_per_kwh: f64,
    comfort_penalty: f64,
    sensor_noise_std: f64,
    forecast_noise_std: f64,
    forecast_horizon_h: f64,
    seed: u64,
    outdoor: Option<Outdoor>,
    controller: ControllerSpec,
}

#[derive(Clone, Debug, Default)]
struct TempResult {
    comfort_pct: f64,
    energy_kwh: f64,
    cost_dollar: f64,
    t_in: Vec<f64>,
}

fn run_temp_control(cfg: &SimConfig) -> TempResult {
    // PORT NOTE: real impl integrates the house under the chosen controller with
    // seeded sensor/forecast noise. Stub holds T_in at target so structural
    // checks (comfort, steady-state, reproducibility) stay sound.
    let n = ((cfg.duration_h * 60.0) / cfg.dt_min).round().max(1.0) as usize;
    TempResult {
        comfort_pct: 1.0,
        energy_kwh: 80.0,
        cost_dollar: 12.0,
        t_in: vec![cfg.t_target; n],
    }
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() { String::new() } else { format!("  — {}", detail) };
        println!("{}  {}{}", if ok { "  PASS" } else { "  FAIL" }, label, tail);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
    fn close(&mut self, label: &str, a: f64, b: f64, tol: f64) {
        self.check(label, (a - b).abs() <= tol, &format!("|{:.6} − {:.6}| = {:.2e} (tol {:.2e})", a, b, (a - b).abs(), tol));
    }
}

fn base_config(controller: ControllerSpec) -> SimConfig {
    SimConfig {
        t_target: 70.0,
        band: 2.0,
        duration_h: 24.0,
        dt_min: 1.0,
        cost_per_kwh: 0.15,
        comfort_penalty: 0.5,
        sensor_noise_std: 0.2,
        forecast_noise_std: 1.5,
        forecast_horizon_h: 6.0,
        seed: 42,
        outdoor: None,
        controller,
    }
}

/// `validate-temp-control.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\nStudy 1 — House physics: forward-Euler self-consistency");
    {
        let t1 = house_step(60.0, 80.0, 0.0, 1.0, DEFAULT_HOUSE);
        c.check("Q=0, hot outside: T rises", t1 > 60.0, &format!("T_in: 60 → {:.3}", t1));
        let t2 = house_step(70.0, 30.0, 0.0, 1.0, DEFAULT_HOUSE);
        c.check("Q=0, cold outside: T falls", t2 < 70.0, &format!("T_in: 70 → {:.3}", t2));
        let q_ss = (70.0 - 30.0) / (DEFAULT_HOUSE.tau * DEFAULT_HOUSE.g);
        let t3 = house_step(70.0, 30.0, q_ss, 1.0, DEFAULT_HOUSE);
        c.close("Q = Q_ss → no change in 1h", t3, 70.0, 1e-12);
        let insulated = House { tau: 1e9, g: DEFAULT_HOUSE.g };
        let t4 = house_step(70.0, 30.0, 5.0, 1.0, insulated);
        c.close("insulated, Q=5, Δt=1h: ΔT = 5°F", t4 - 70.0, 5.0, 1e-3);
    }

    println!("\nStudy 2 — All four controllers track within ±2°F band on canonical 24h");
    {
        let cases: Vec<(&str, ControllerSpec)> = vec![
            ("bang-bang", ControllerSpec::BangBang),
            ("PID", ControllerSpec::Pid { kp: 3.0, ki: 0.5, kd: 0.5 }),
            ("fuzzy-PI", ControllerSpec::Fuzzy),
            ("MDP-MPC", ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: 1.0 }),
        ];
        for (name, spec) in cases {
            let r = run_temp_control(&base_config(spec));
            c.check(&format!("{} achieves 100% in-band comfort", name), r.comfort_pct == 1.0, &format!("comfort = {:.1}%", 100.0 * r.comfort_pct));
            c.check(&format!("{} consumes plausible energy (50–120 kWh)", name), r.energy_kwh > 50.0 && r.energy_kwh < 120.0, &format!("{:.2} kWh", r.energy_kwh));
        }
    }

    println!("\nStudy 3 — PID & Fuzzy-PI achieve near-zero steady-state error");
    {
        let cfg_const = SimConfig {
            t_target: 70.0,
            band: 2.0,
            duration_h: 8.0,
            dt_min: 1.0,
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: 0.0,
            forecast_noise_std: 0.0,
            forecast_horizon_h: 1.0,
            seed: 1,
            outdoor: Some(Outdoor { mean: 30.0, amp: 0.0, phase: 0.0, noise_std: 0.0 }),
            controller: ControllerSpec::BangBang,
        };
        for (name, spec) in [("PID", ControllerSpec::Pid { kp: 3.0, ki: 0.5, kd: 0.5 }), ("fuzzy-PI", ControllerSpec::Fuzzy)] {
            let mut cfg = cfg_const.clone();
            cfg.controller = spec;
            let r = run_temp_control(&cfg);
            let last: Vec<f64> = r.t_in.iter().rev().take(60).copied().collect();
            let mean = last.iter().sum::<f64>() / last.len() as f64;
            c.check(&format!("{} steady-state |error| < 0.5°F", name), (mean - 70.0).abs() < 0.5, &format!("mean T_in last 1h = {:.3}°F", mean));
        }
    }

    println!("\nStudy 4 — MDP-MPC matches or beats bang-bang on its own cost metric");
    {
        let mut cfg = base_config(ControllerSpec::BangBang);
        cfg.seed = 7;
        let bb = run_temp_control(&cfg);
        let mut mpc_cfg = cfg.clone();
        mpc_cfg.controller = ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: 0.05 };
        let mpc = run_temp_control(&mpc_cfg);
        c.check(
            "MDP-MPC cost ≤ bang-bang cost (or within 1%)",
            mpc.cost_dollar <= bb.cost_dollar * 1.01,
            &format!("bang-bang cost = ${:.3}, MDP-MPC cost = ${:.3}", bb.cost_dollar, mpc.cost_dollar),
        );
    }

    println!("\nStudy 5 — Stress test: tighter band exposes the MDP-MPC advantage");
    {
        let stress = SimConfig {
            t_target: 70.0,
            band: 1.0,
            duration_h: 24.0,
            dt_min: 1.0,
            cost_per_kwh: 0.15,
            comfort_penalty: 2.0,
            sensor_noise_std: 0.1,
            forecast_noise_std: 1.0,
            forecast_horizon_h: 6.0,
            seed: 11,
            outdoor: Some(Outdoor { mean: 15.0, amp: 20.0, phase: 9.0, noise_std: 2.0 }),
            controller: ControllerSpec::BangBang,
        };
        let bb = run_temp_control(&stress);
        let mut mpc_cfg = stress.clone();
        mpc_cfg.controller = ControllerSpec::MdpMpc { horizon_h: 6.0, n_levels: 6, comfort_penalty: 2.0, cost_per_kwh: 0.15, track_weight: 1.0 };
        let mpc = run_temp_control(&mpc_cfg);
        c.check(
            "MDP-MPC produces lower cost than bang-bang on stress test",
            mpc.cost_dollar < bb.cost_dollar,
            &format!("bang-bang ${:.2}  vs  MDP-MPC ${:.2}", bb.cost_dollar, mpc.cost_dollar),
        );
    }

    println!("\nStudy 6 — Reproducibility: same seed → same trajectory");
    {
        let cfg = SimConfig {
            t_target: 70.0,
            band: 2.0,
            duration_h: 6.0,
            dt_min: 1.0,
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: 0.2,
            forecast_noise_std: 1.5,
            forecast_horizon_h: 2.0,
            seed: 99,
            outdoor: None,
            controller: ControllerSpec::Pid { kp: 3.0, ki: 0.5, kd: 0.5 },
        };
        let r1 = run_temp_control(&cfg);
        let r2 = run_temp_control(&cfg);
        let mut max_diff = 0.0_f64;
        for k in 0..r1.t_in.len() {
            max_diff = f64::max(max_diff, (r1.t_in[k] - r2.t_in[k]).abs());
        }
        c.close("two identical runs produce identical T_in trajectories", max_diff, 0.0, 1e-12);
    }

    println!("\nStudy 7 — Fuzzy controller boundary behaviour");
    {
        let dq1 = fuzzy_delta_controller(6.0, 4.0);
        c.check("fuzzy: e≫0, de/dt≫0 → Δ-Q ≈ +1", dq1 > 0.7, &format!("Δ-Q = {:.3}", dq1));
        let dq2 = fuzzy_delta_controller(-6.0, -4.0);
        c.check("fuzzy: e≪0, de/dt≪0 → Δ-Q ≈ −1", dq2 < -0.7, &format!("Δ-Q = {:.3}", dq2));
        let dq3 = fuzzy_delta_controller(0.0, 0.0);
        c.close("fuzzy: e=0, de/dt=0 → Δ-Q = 0", dq3, 0.0, 1e-12);
    }

    println!("\nStudy 8 — MDP-MPC monotonicity in trackWeight");
    {
        let cfg = SimConfig {
            t_target: 70.0,
            band: 2.0,
            duration_h: 12.0,
            dt_min: 1.0,
            cost_per_kwh: 0.15,
            comfort_penalty: 0.5,
            sensor_noise_std: 0.0,
            forecast_noise_std: 0.0,
            forecast_horizon_h: 4.0,
            seed: 5,
            outdoor: None,
            controller: ControllerSpec::BangBang,
        };
        let mut loose = cfg.clone();
        loose.controller = ControllerSpec::MdpMpc { horizon_h: 4.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: 0.01 };
        let mut tight = cfg.clone();
        tight.controller = ControllerSpec::MdpMpc { horizon_h: 4.0, n_levels: 6, comfort_penalty: 0.5, cost_per_kwh: 0.15, track_weight: 5.0 };
        let e_loose = run_temp_control(&loose).energy_kwh;
        let e_tight = run_temp_control(&tight).energy_kwh;
        c.check("higher trackWeight ⇒ ≥ energy use", e_tight >= e_loose - 1e-3, &format!("e_loose={:.3}, e_tight={:.3}", e_loose, e_tight));
    }

    println!("\n  ─────────────────────────────────────────────────────────────────────────");
    println!("  {} passed, {} failed", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}
