//! Port of `src/des/runners/difference-runner.ts`.
//!
//! Discrete-time difference-equation kernel for the SEIR-with-hospitalization
//! model, plus the closed-form steady-state solution and the largest
//! provably-stable forward-Euler step. Fully deterministic: no RNG, no clock
//! beyond timing.
//!
//! The TS `interface State` (hyphenated keys like `'I-P'`) becomes a struct
//! with snake-case fields plus a [`State::compartment`] accessor so the
//! `COMPARTMENT_ORDER` loops can index by name. `Record<string, number>`
//! population maps become `HashMap<String, f64>`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use super::shared::{
    analytical_transition_tables, average_record, mean_residence, update_peaks,
    zero_compartment_record,
};
use super::types::{Kernel, RunOpts, RunResult, SimConfig, Totals, COMPARTMENT_ORDER};

/// Continuous-state vector. Hyphenated TS keys → snake fields.
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

    /// Index a compartment by its `COMPARTMENT_ORDER` name (the TS `x[c]`).
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

    /// The 7 alive-compartment values as a `HashMap` (for `update_peaks`).
    fn compartment_record(&self) -> HashMap<String, f64> {
        COMPARTMENT_ORDER
            .iter()
            .map(|c| (c.to_string(), self.compartment(c)))
            .collect()
    }
}

/// Mean residence times used throughout the derivation.
#[derive(Clone, Copy, Debug)]
pub struct MeanResidences {
    pub arrival: f64,
    pub s: f64,
    pub e: f64,
    pub i_p: f64,
    pub i_a: f64,
    pub i_s: f64,
    pub i_h: f64,
    pub r: f64,
    pub d: f64,
}

fn mus(config: &SimConfig) -> MeanResidences {
    MeanResidences {
        arrival: (config.arrivals_interarrival.0 + config.arrivals_interarrival.1) / 2.0,
        s: mean_residence(config, "S"),
        e: mean_residence(config, "E"),
        i_p: mean_residence(config, "I-P"),
        i_a: mean_residence(config, "I-A"),
        i_s: mean_residence(config, "I-S"),
        i_h: mean_residence(config, "I-H"),
        r: mean_residence(config, "R"),
        d: mean_residence(config, "D"),
    }
}

/// `runDifferenceOnce` — forward-Euler integrate the SEIR ODE to the horizon.
pub fn run_difference_once(config: &SimConfig, opts: &RunOpts) -> RunResult {
    let dt = config.step_size;
    let mu = mus(config);
    let p = config.probabilities;

    let lambda_src = |t: f64, big_c: f64| -> f64 {
        if big_c < config.source_cap && t < config.phase1_days {
            1.0 / mu.arrival
        } else {
            0.0
        }
    };

    let mut x = State::zeros();
    let mut t = 0.0_f64;
    let total_steps = (config.horizon_days / dt).round().max(1.0) as i64;
    let mut pop_sums = zero_compartment_record();
    let mut peak = zero_compartment_record();
    let started_at = Instant::now();

    for _ in 0..total_steps {
        for c in COMPARTMENT_ORDER {
            *pop_sums.get_mut(c).unwrap() += x.compartment(c) * dt;
        }

        let src = lambda_src(t, x.c);
        let d_s = src + x.r / mu.r - x.s / mu.s;
        let d_e = x.s / mu.s - x.e / mu.e;
        let d_ip = x.e / mu.e - x.i_p / mu.i_p;
        let d_ia = x.i_p * p.asymptomatic_share / mu.i_p - x.i_a / mu.i_a;
        let d_is = x.i_p * (1.0 - p.asymptomatic_share) / mu.i_p - x.i_s / mu.i_s;
        let d_ih = x.i_s * p.hospitalization_given_symptom / mu.i_s - x.i_h / mu.i_h;
        let d_r = x.i_a / mu.i_a
            + x.i_s * (1.0 - p.hospitalization_given_symptom) / mu.i_s
            + x.i_h * (1.0 - p.case_fatality_given_hospital) / mu.i_h
            - x.r / mu.r;
        let d_d = x.i_h * p.case_fatality_given_hospital / mu.i_h;

        x = State {
            s: x.s + dt * d_s,
            e: x.e + dt * d_e,
            i_p: x.i_p + dt * d_ip,
            i_a: x.i_a + dt * d_ia,
            i_s: x.i_s + dt * d_is,
            i_h: x.i_h + dt * d_ih,
            r: x.r + dt * d_r,
            d: x.d + dt * d_d,
            c: x.c + dt * src,
        };
        t += dt;

        update_peaks(&mut peak, &x.compartment_record());
    }

    let elapsed = started_at.elapsed().as_millis();
    let mut final_populations: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        final_populations.insert(c.to_string(), x.compartment(c));
    }

    let tables = analytical_transition_tables(&p);
    let time_avg = average_record(&pop_sums, config.horizon_days);

    RunResult {
        kernel: Kernel::Difference,
        config: config.clone(),
        seed: opts.seed.unwrap_or(0),
        totals: Totals {
            created: x.c,
            absorbed: x.d,
        },
        final_populations,
        transition_counts: tables.counts,
        split_probs: tables.splits,
        time_avg_populations: time_avg,
        peak_populations: peak,
        elapsed_ms: elapsed,
    }
}

/// Closed-form analytical steady state for the open system (`lambda` const).
#[derive(Clone, Debug)]
pub struct SteadyState {
    /// Mean residence times (days) used in the derivation.
    pub mu: MeanResidences,
    /// Source emission rate (entities/day).
    pub lambda: f64,
    /// Per-S-pass death fraction `(1 - p_a) * p_h * p_d`.
    pub q: f64,
    /// Throughput at S (entities/day).
    pub f_s: f64,
    /// Throughput rates `f_c` into each compartment (entities/day).
    pub flows: HashMap<String, f64>,
    /// Fixed-point populations `N*_c = mu_c * f_c`.
    pub populations: HashMap<String, f64>,
    /// Sum `N*_S + … + N*_R + N*_D`.
    pub total_alive: f64,
}

/// `analyticalSteadyState`.
pub fn analytical_steady_state(config: &SimConfig) -> SteadyState {
    let mu = mus(config);
    let p = config.probabilities;

    let lambda = 1.0 / mu.arrival;
    let q = (1.0 - p.asymptomatic_share)
        * p.hospitalization_given_symptom
        * p.case_fatality_given_hospital;
    let f_s = lambda / q;

    let mut flows: HashMap<String, f64> = HashMap::new();
    flows.insert("S".to_string(), f_s);
    flows.insert("E".to_string(), f_s);
    flows.insert("I-P".to_string(), f_s);
    flows.insert("I-A".to_string(), p.asymptomatic_share * f_s);
    flows.insert("I-S".to_string(), (1.0 - p.asymptomatic_share) * f_s);
    flows.insert(
        "I-H".to_string(),
        p.hospitalization_given_symptom * (1.0 - p.asymptomatic_share) * f_s,
    );
    flows.insert("D".to_string(), q * f_s); // == lambda by construction
    flows.insert("R".to_string(), (1.0 - q) * f_s);

    let mut populations: HashMap<String, f64> = HashMap::new();
    populations.insert("S".to_string(), mu.s * flows["S"]);
    populations.insert("E".to_string(), mu.e * flows["E"]);
    populations.insert("I-P".to_string(), mu.i_p * flows["I-P"]);
    populations.insert("I-A".to_string(), mu.i_a * flows["I-A"]);
    populations.insert("I-S".to_string(), mu.i_s * flows["I-S"]);
    populations.insert("I-H".to_string(), mu.i_h * flows["I-H"]);
    populations.insert("R".to_string(), mu.r * flows["R"]);
    populations.insert("D".to_string(), mu.d * flows["D"]);

    let total_alive: f64 = COMPARTMENT_ORDER
        .iter()
        .map(|c| populations.get(*c).copied().unwrap_or(0.0))
        .sum::<f64>()
        + populations["D"];

    SteadyState {
        mu,
        lambda,
        q,
        f_s,
        flows,
        populations,
        total_alive,
    }
}

/// `maxStableStep` — largest provably-stable forward-Euler step, `2·min(mu_c)`.
pub fn max_stable_step(config: &SimConfig) -> f64 {
    let mu = mus(config);
    let mins = [mu.s, mu.e, mu.i_p, mu.i_a, mu.i_s, mu.i_h, mu.r, mu.d];
    2.0 * mins.iter().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::runners::types::default_config;

    #[test]
    fn steady_state_death_flow_equals_lambda() {
        let cfg = default_config();
        let ss = analytical_steady_state(&cfg);
        // f_D = q * f_S = lambda by construction.
        assert!((ss.flows["D"] - ss.lambda).abs() < 1e-9);
    }

    #[test]
    fn difference_kernel_runs() {
        // The default step (1.0) intentionally exceeds max_stable_step (0.40),
        // so explicit Euler is unstable there. Exercise the kernel at a STABLE
        // step where populations stay finite and (essentially) non-negative.
        let mut cfg = default_config();
        cfg.step_size = 0.1;
        assert!(cfg.step_size < max_stable_step(&cfg));
        let r = run_difference_once(&cfg, &RunOpts::default());
        assert_eq!(r.kernel, Kernel::Difference);
        for c in COMPARTMENT_ORDER {
            let v = r.final_populations[c];
            assert!(v.is_finite() && v >= -1e-6, "{c} = {v}");
        }
    }

    #[test]
    fn difference_kernel_unstable_above_max_step() {
        // Documents the instability the engine guards against: at the default
        // step (> max_stable_step) the kernel does not stay non-negative.
        let cfg = default_config();
        assert!(cfg.step_size > max_stable_step(&cfg));
        let r = run_difference_once(&cfg, &RunOpts::default());
        let all_nonneg = COMPARTMENT_ORDER
            .iter()
            .all(|c| r.final_populations[*c] >= 0.0);
        assert!(
            !all_nonneg,
            "expected explicit Euler to be unstable at dt > max_stable_step"
        );
    }

    #[test]
    fn max_stable_step_for_defaults() {
        let cfg = default_config();
        // min mu_c = mu_D = 0.20 → 2*0.20 = 0.40.
        assert!((max_stable_step(&cfg) - 0.40).abs() < 1e-9);
    }
}
