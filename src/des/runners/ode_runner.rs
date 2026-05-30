//! Port of `src/des/runners/ode-runner.ts`.
//!
//! Deterministic mean-field SEIR ODE solver (RK4) as a fourth independent
//! reference: no randomness, no entities, no events — the continuous-flow limit
//! of the same compartmental model. The optional `JsonlLogger` is reused from
//! `crate::des::observability::logger`.
//!
//! The TS `interface State` (with hyphenated keys + a `C` cumulative-emissions
//! field) becomes a struct with a [`State::compartment`] accessor and an `lin`
//! helper.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use crate::des::observability::logger::{JsonValue, JsonlLogger, LogLevel};

use super::shared::{
    analytical_transition_tables, average_record, mean_residence, update_peaks,
    zero_compartment_record,
};
use super::types::{
    Kernel, Probabilities, RunOpts, RunResult, SimConfig, Totals, COMPARTMENT_ORDER,
};

// --- tiny JSON construction helpers (mirror the reference port) -------------
fn js(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}
fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}
fn jb(v: bool) -> JsonValue {
    JsonValue::Bool(v)
}
fn jobj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

#[derive(Clone, Copy, Debug, Default)]
struct State {
    s: f64,
    e: f64,
    i_p: f64,
    i_a: f64,
    i_s: f64,
    i_h: f64,
    r: f64,
    d: f64,
    c: f64,
}

impl State {
    fn zeros() -> Self {
        State::default()
    }

    fn compartment(&self, c: &str) -> f64 {
        match c {
            "S" => self.s,
            "E" => self.e,
            "I-P" => self.i_p,
            "I-A" => self.i_a,
            "I-S" => self.i_s,
            "I-H" => self.i_h,
            "R" => self.r,
            "D" => self.d,
            "C" => self.c,
            _ => 0.0,
        }
    }

    fn compartment_record(&self) -> HashMap<String, f64> {
        COMPARTMENT_ORDER.iter().map(|c| (c.to_string(), self.compartment(c))).collect()
    }
}

/// `lin(a, b, k)` — `a + k * b` componentwise.
fn lin(a: &State, b: &State, k: f64) -> State {
    State {
        s: a.s + k * b.s,
        e: a.e + k * b.e,
        i_p: a.i_p + k * b.i_p,
        i_a: a.i_a + k * b.i_a,
        i_s: a.i_s + k * b.i_s,
        i_h: a.i_h + k * b.i_h,
        r: a.r + k * b.r,
        d: a.d + k * b.d,
        c: a.c + k * b.c,
    }
}

struct Mu {
    arrival: f64,
    s: f64,
    e: f64,
    i_p: f64,
    i_a: f64,
    i_s: f64,
    i_h: f64,
    r: f64,
}

fn lambda_src(config: &SimConfig, mu: &Mu, t: f64, big_c: f64) -> f64 {
    if big_c < config.source_cap && t < config.phase1_days {
        1.0 / mu.arrival
    } else {
        0.0
    }
}

fn deriv(config: &SimConfig, mu: &Mu, p: &Probabilities, t: f64, x: &State) -> State {
    let src = lambda_src(config, mu, t, x.c);
    State {
        s: x.r / mu.r + src - x.s / mu.s,
        e: x.s / mu.s - x.e / mu.e,
        i_p: x.e / mu.e - x.i_p / mu.i_p,
        i_a: x.i_p * p.asymptomatic_share / mu.i_p - x.i_a / mu.i_a,
        i_s: x.i_p * (1.0 - p.asymptomatic_share) / mu.i_p - x.i_s / mu.i_s,
        i_h: x.i_s * p.hospitalization_given_symptom / mu.i_s - x.i_h / mu.i_h,
        r: x.i_a / mu.i_a
            + x.i_s * (1.0 - p.hospitalization_given_symptom) / mu.i_s
            + x.i_h * (1.0 - p.case_fatality_given_hospital) / mu.i_h
            - x.r / mu.r,
        d: x.i_h * p.case_fatality_given_hospital / mu.i_h,
        c: src,
    }
}

/// `runOdeOnce` — RK4-integrate the mean-field ODE to the horizon.
pub fn run_ode_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let sample_every = opts.sample_every_days.unwrap_or(1.0);
    let mut logger = if opts.log_events {
        opts.log_path
            .as_ref()
            .map(|path| JsonlLogger::new(path, LogLevel::Info))
    } else {
        None
    };

    let mu = Mu {
        arrival: (config.arrivals_interarrival.0 + config.arrivals_interarrival.1) / 2.0,
        s: mean_residence(config, "S"),
        e: mean_residence(config, "E"),
        i_p: mean_residence(config, "I-P"),
        i_a: mean_residence(config, "I-A"),
        i_s: mean_residence(config, "I-S"),
        i_h: mean_residence(config, "I-H"),
        r: mean_residence(config, "R"),
    };
    let p = config.probabilities;

    let dt = 0.05; // ODE integration timestep, fine enough for RK4
    let steps_per_sample = (sample_every / dt).round().max(1.0) as i64;
    let total_steps = (config.horizon_days / dt).round() as i64;

    let mut x = State::zeros();
    let mut t = 0.0_f64;
    let mut pop_sums = zero_compartment_record();
    let mut peak = zero_compartment_record();
    let mut samples = 0.0_f64;

    if let Some(logger) = logger.as_mut() {
        let seed_val = match opts.seed {
            Some(seed) => jn(seed as f64),
            None => js("deterministic"),
        };
        logger.log(jobj(vec![
            ("kind", js("sim_start")),
            (
                "config",
                jobj(vec![
                    ("kernel", js("ode-rk4")),
                    ("seed", seed_val),
                    ("dt", jn(dt)),
                    ("tPhase1", jn(config.phase1_days)),
                    ("tMax", jn(config.horizon_days)),
                    ("sourceCap", jn(config.source_cap)),
                ]),
            ),
        ]));
    }

    let started_at = Instant::now();

    for i in 0..total_steps {
        // Left-Riemann integration for time-averaged populations.
        for c in COMPARTMENT_ORDER {
            *pop_sums.get_mut(c).unwrap() += x.compartment(c) * dt;
        }

        // RK4 step.
        let k1 = deriv(config, &mu, &p, t, &x);
        let k2 = deriv(config, &mu, &p, t + dt / 2.0, &lin(&x, &k1, dt / 2.0));
        let k3 = deriv(config, &mu, &p, t + dt / 2.0, &lin(&x, &k2, dt / 2.0));
        let k4 = deriv(config, &mu, &p, t + dt, &lin(&x, &k3, dt));
        x = State {
            s: x.s + dt * (k1.s + 2.0 * k2.s + 2.0 * k3.s + k4.s) / 6.0,
            e: x.e + dt * (k1.e + 2.0 * k2.e + 2.0 * k3.e + k4.e) / 6.0,
            i_p: x.i_p + dt * (k1.i_p + 2.0 * k2.i_p + 2.0 * k3.i_p + k4.i_p) / 6.0,
            i_a: x.i_a + dt * (k1.i_a + 2.0 * k2.i_a + 2.0 * k3.i_a + k4.i_a) / 6.0,
            i_s: x.i_s + dt * (k1.i_s + 2.0 * k2.i_s + 2.0 * k3.i_s + k4.i_s) / 6.0,
            i_h: x.i_h + dt * (k1.i_h + 2.0 * k2.i_h + 2.0 * k3.i_h + k4.i_h) / 6.0,
            r: x.r + dt * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r) / 6.0,
            d: x.d + dt * (k1.d + 2.0 * k2.d + 2.0 * k3.d + k4.d) / 6.0,
            c: x.c + dt * (k1.c + 2.0 * k2.c + 2.0 * k3.c + k4.c) / 6.0,
        };
        t += dt;

        update_peaks(&mut peak, &x.compartment_record());

        if (i + 1) % steps_per_sample == 0 {
            samples += 1.0;
            if let Some(logger) = logger.as_mut() {
                let populations: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
                    .iter()
                    .map(|c| (c.to_string(), jn(x.compartment(c))))
                    .collect();
                let alive: f64 = COMPARTMENT_ORDER.iter().map(|c| x.compartment(c)).sum();
                logger.log(jobj(vec![
                    ("kind", js("tick")),
                    ("t", jn(t)),
                    ("populations", JsonValue::Object(populations)),
                    ("cumD", jn(x.d)),
                    ("alive", jn(alive)),
                    (
                        "sourcesActive",
                        jb(t < config.phase1_days && x.c < config.source_cap),
                    ),
                ]));
            }
        }
    }

    let _ = samples;
    let elapsed = started_at.elapsed().as_millis();
    let mut final_populations: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        final_populations.insert(c.to_string(), x.compartment(c));
    }

    // ODE splits are exact; emit the analytical splits.
    let tables = analytical_transition_tables(&p);
    let time_avg = average_record(&pop_sums, config.horizon_days);

    if let Some(logger) = logger.as_mut() {
        let final_pop_json: Vec<(String, JsonValue)> = COMPARTMENT_ORDER
            .iter()
            .map(|c| (c.to_string(), jn(x.compartment(c))))
            .collect();
        logger.log(jobj(vec![
            ("kind", js("sim_end")),
            ("t", jn(config.horizon_days)),
            ("elapsedMs", jn(elapsed as f64)),
            (
                "totals",
                JsonValue::Object(vec![
                    ("created".to_string(), jn(x.c)),
                    ("absorbed".to_string(), jn(x.d)),
                    ("finalPopulations".to_string(), JsonValue::Object(final_pop_json)),
                ]),
            ),
        ]));
        logger.close();
    }

    RunResult {
        kernel: Kernel::Ode,
        config: config.clone(),
        seed: opts.seed.unwrap_or(0),
        totals: Totals { created: x.c, absorbed: x.d },
        final_populations,
        transition_counts: tables.counts,
        split_probs: tables.splits,
        time_avg_populations: time_avg,
        peak_populations: peak,
        elapsed_ms: elapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::default_config;

    #[test]
    fn ode_kernel_runs_and_is_finite() {
        let cfg = SimConfig { horizon_days: 50.0, ..default_config() };
        let r = run_ode_once(&cfg, &RunOpts::default());
        assert_eq!(r.kernel, Kernel::Ode);
        for c in COMPARTMENT_ORDER {
            assert!(r.final_populations[c].is_finite());
        }
    }
}
