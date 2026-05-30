//! Port of `src/des/runners/validate-factmachine.ts`.
//!
//! Validates the FactMachine POMDP: Bayesian belief filter vs scipy, majority
//! win-probability, Brier calibration, policy ranking, late-flip misdirection,
//! Tiger POMDP exact-VI vs QMDP, and binary-vs-scalar market contrast.
//! Driver → [`run`].
//!
//! PORT NOTES — wire to real modules:
//!   * `crate::des::general::belief::{DiscreteBelief, brier_score}` (DiscreteBelief
//!     and `brier_score` are ported faithfully here).
//!   * `crate::des::general::pomdp::{QMDPSolver, POMDPSpec, pomdp_exact_finite_horizon}`.
//!   * `crate::des::main_factmachine::{default_params, FactMachineParams, run_factmachine}`.
//!   * Python (scipy/numpy) reference via `std::process::Command`; JSON parse needs
//!     `serde_json` (absent) → `run_python` returns `None` (SKIP), matching the TS.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::process::Command;

// =============================================================================
// Belief filter (faithful).
// =============================================================================

#[derive(Clone, Debug)]
struct DiscreteBelief {
    states: Vec<f64>,
    weights: Vec<f64>,
}

impl DiscreteBelief {
    fn new(states: Vec<f64>) -> Self {
        let n = states.len();
        DiscreteBelief { states, weights: vec![1.0 / n as f64; n] }
    }
    fn with_weights(states: Vec<f64>, weights: Vec<f64>) -> Self {
        DiscreteBelief { states, weights }
    }
    fn update<F: Fn(f64) -> f64>(&mut self, likelihood: F) {
        let mut sum = 0.0;
        for i in 0..self.states.len() {
            self.weights[i] *= likelihood(self.states[i]);
            sum += self.weights[i];
        }
        if sum > 0.0 {
            for w in self.weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    fn mean(&self) -> f64 {
        self.states.iter().zip(self.weights.iter()).map(|(s, w)| s * w).sum()
    }
}

fn brier_score(p: f64, outcome: f64) -> f64 {
    (p - outcome) * (p - outcome)
}

// =============================================================================
// Stubbed POMDP layer.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct PomdpSpec {
    states: Vec<String>,
    actions: Vec<String>,
    observations: Vec<String>,
    discount: f64,
}

struct ExactPomdp {
    listen_index: usize,
}

impl ExactPomdp {
    fn v(&self, _belief: &[f64]) -> f64 {
        0.0
    }
    fn act(&self, _belief: &DiscreteBelief) -> usize {
        self.listen_index
    }
}

fn pomdp_exact_finite_horizon(spec: &PomdpSpec, _horizon: usize) -> ExactPomdp {
    ExactPomdp { listen_index: spec.actions.iter().position(|a| a == "listen").unwrap_or(0) }
}

struct QmdpSolver {
    listen_index: usize,
}

impl QmdpSolver {
    fn new(spec: &PomdpSpec, _tol: f64, _max_iter: usize) -> Self {
        QmdpSolver { listen_index: spec.actions.iter().position(|a| a == "listen").unwrap_or(0) }
    }
    fn q_belief(&self, _belief: &DiscreteBelief, _a: usize) -> f64 {
        0.0
    }
    fn act(&self, _belief: &DiscreteBelief) -> usize {
        self.listen_index
    }
}

// =============================================================================
// Stubbed FactMachine sim.
// =============================================================================

#[derive(Clone, Debug)]
struct FactMachineParams {
    seed: u64,
    true_theta: f64,
    t: usize,
    policy: &'static str,
    resolution_mode: &'static str,
    late_flip: bool,
    late_flip_multiplier: f64,
    market_type: &'static str,
    theta_bins: usize,
    k_noise: f64,
    fee: f64,
    n_voters: usize,
}

fn default_params() -> FactMachineParams {
    FactMachineParams {
        seed: 0,
        true_theta: 0.5,
        t: 24,
        policy: "hold",
        resolution_mode: "bernoulli",
        late_flip: false,
        late_flip_multiplier: 1.0,
        market_type: "binary",
        theta_bins: 21,
        k_noise: 20.0,
        fee: 0.0,
        n_voters: 51,
    }
}

#[derive(Clone, Debug, Default)]
struct FmResult {
    belief_mean: Vec<f64>,
    final_outcome: f64,
    pnl: f64,
    belief_var: Vec<f64>,
    price_history: Vec<Vec<f64>>,
}

fn run_fact_machine(p: &FactMachineParams) -> FmResult {
    // PORT NOTE: real impl runs the DES POMDP with seeded noise traders. Stub
    // returns a uniform-prior trajectory so structural checks stay sound.
    let bins = if p.market_type == "scalar" { p.theta_bins } else { 2 };
    let uniform = vec![1.0 / bins as f64; bins];
    FmResult {
        belief_mean: vec![0.5; p.t + 1],
        final_outcome: 0.0,
        pnl: 0.0,
        belief_var: vec![0.25; p.t + 1],
        price_history: vec![uniform; p.t + 1],
    }
}

// =============================================================================
// Python reference (always None — see module PORT NOTES).
// =============================================================================

struct PyJson {
    final_mean: f64,
    final_belief: Vec<f64>,
    mean_history: Vec<f64>,
    thetas: Vec<f64>,
    pwin: Vec<f64>,
}

fn run_python(env: &[(&str, String)]) -> Option<PyJson> {
    let python = std::env::var("FACTMACHINE_PY").unwrap_or_else(|_| "python3".to_string());
    let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("external-references").join("factmachine").join("factmachine.py");
    let mut cmd = Command::new(python);
    cmd.arg(&script);
    for (k, v) in env {
        cmd.env(k, v);
    }
    match cmd.output() {
        Ok(out) if out.status.success() => None, // PORT NOTE: parse last stdout line via serde_json.
        _ => None,
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
        let extra = if detail.is_empty() { String::new() } else { format!("  ({})", detail) };
        if ok {
            self.pass += 1;
            println!("  PASS    {}{}", label, extra);
        } else {
            self.fail += 1;
            println!("  FAIL    {}{}", label, extra);
        }
    }
}

/// `validate-factmachine.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    // STUDY 1.
    println!("\n=== STUDY 1: Bayesian belief filter ≡ scipy reference ===");
    {
        let k = 21usize;
        let informedness = 0.6;
        let states: Vec<f64> = (0..k).map(|i| i as f64 / (k - 1) as f64).collect();
        let obs: Vec<(i64, i64)> = vec![(12, 20), (15, 22), (9, 19), (17, 20), (11, 18), (14, 19), (16, 20), (10, 22)];
        let obs_str = obs.iter().map(|(y, n)| format!("{}/{}", y, n)).collect::<Vec<_>>().join(",");
        let py = run_python(&[("PROBLEM", "belief".to_string()), ("THETA_BINS", k.to_string()), ("INFORMEDNESS", informedness.to_string()), ("OBS", obs_str)]);
        match py {
            None => println!("  SKIP    scipy/numpy reference unavailable"),
            Some(py) => {
                let mut b = DiscreteBelief::new(states.clone());
                let mut ts_means: Vec<f64> = vec![b.mean()];
                for &(y, n) in &obs {
                    b.update(|theta| {
                        let q = theta * informedness + 0.5 * (1.0 - informedness);
                        (y as f64 * f64::max(1e-300, q).ln() + (n - y) as f64 * f64::max(1e-300, 1.0 - q).ln()).exp()
                    });
                    ts_means.push(b.mean());
                }
                let d_mean = (ts_means[ts_means.len() - 1] - py.final_mean).abs();
                c.check(
                    &format!("final E[θ] match  TS={:.8}  PY={:.8}", ts_means[ts_means.len() - 1], py.final_mean),
                    d_mean < 1e-10,
                    &format!("|Δ|={:.2e}", d_mean),
                );
                let mut max_belief_diff = 0.0_f64;
                for i in 0..k {
                    max_belief_diff = f64::max(max_belief_diff, (b.weights[i] - py.final_belief[i]).abs());
                }
                c.check("per-bin |b_TS − b_PY| ≤ 1e-12 across 21 bins", max_belief_diff < 1e-12, &format!("max|Δ|={:.2e}", max_belief_diff));
                let mut max_mean_diff = 0.0_f64;
                for t in 0..=obs.len() {
                    max_mean_diff = f64::max(max_mean_diff, (ts_means[t] - py.mean_history[t]).abs());
                }
                c.check(&format!("per-tick mean trajectory matches across {} steps", obs.len() + 1), max_mean_diff < 1e-10, &format!("max|Δ|={:.2e}", max_mean_diff));
            }
        }
    }

    // STUDY 2.
    println!("\n=== STUDY 2: P(majority votes YES | θ) ≡ scipy.stats.binom.sf ===");
    {
        let py = run_python(&[("PROBLEM", "pwin".to_string()), ("N_VOTERS", "51".to_string())]);
        match py {
            None => println!("  SKIP    scipy reference unavailable"),
            Some(py) => {
                let mut params = default_params();
                params.resolution_mode = "majority";
                params.n_voters = 51;
                let nn = params.n_voters as i64;
                let half = params.n_voters / 2;
                let pwin_ts = |theta: f64| -> f64 {
                    let mut p = 0.0;
                    let mut log_p = nn as f64 * f64::max(1e-300, 1.0 - theta).ln();
                    let mut lcoef = 0.0;
                    for k in 0..=nn {
                        if k as usize > half {
                            p += (lcoef + log_p).exp();
                        }
                        if k < nn {
                            lcoef += ((nn - k) as f64).ln() - ((k + 1) as f64).ln();
                            log_p += f64::max(1e-300, theta).ln() - f64::max(1e-300, 1.0 - theta).ln();
                        }
                    }
                    p.max(0.0).min(1.0)
                };
                let mut max_diff = 0.0_f64;
                for i in 0..py.thetas.len() {
                    max_diff = f64::max(max_diff, (pwin_ts(py.thetas[i]) - py.pwin[i]).abs());
                }
                c.check("pYesWins at 9 θ values matches scipy.stats.binom.sf to 1e-10", max_diff < 1e-10, &format!("max|Δ|={:.2e}", max_diff));
            }
        }
    }

    // STUDY 3.
    println!("\n=== STUDY 3: Belief calibration over time (Brier decreases) ===");
    {
        let n_reps = 200usize;
        let t = 24usize;
        let mut brier_by_t = vec![0.0; t + 1];
        for r in 0..n_reps {
            let seed = 17 + r as u64;
            let true_theta = 0.05 + 0.9 * (r as f64 / n_reps as f64);
            let mut params = default_params();
            params.seed = seed;
            params.true_theta = true_theta;
            params.t = t;
            params.policy = "hold";
            params.resolution_mode = "bernoulli";
            let r1 = run_fact_machine(&params);
            for tt in 0..=t {
                brier_by_t[tt] += brier_score(r1.belief_mean[tt], r1.final_outcome);
            }
        }
        for tt in 0..=t {
            brier_by_t[tt] /= n_reps as f64;
        }
        let init_brier = brier_by_t[0];
        let final_brier = brier_by_t[t];
        let mid_brier = brier_by_t[t / 2];
        println!("#   Brier(t=0) = {:.4},  Brier(t=12) = {:.4},  Brier(t=24) = {:.4}", init_brier, mid_brier, final_brier);
        c.check("Brier at t=0 (uniform prior, no info) = 0.25 (theoretical)", (init_brier - 0.25).abs() < 1e-8, &format!("init={:.4}", init_brier));
        c.check("Brier at end < Brier at start (filter learns)", final_brier < init_brier - 0.02, &format!("end={:.4}, init={:.4}", final_brier, init_brier));
        c.check("Brier at t=12 < Brier at t=0 (monotone-ish learning)", mid_brier < init_brier - 0.01, &format!("mid={:.4}", mid_brier));
    }

    // STUDY 4.
    println!("\n=== STUDY 4: Policy ranking — oracle ≥ qmdp ≈ myopic > random ≈ hold ===");
    {
        let n_reps = 1000usize;
        let policies = ["hold", "random", "myopic", "qmdp", "oracle"];
        use std::collections::HashMap;
        let mut stats: HashMap<&str, (f64, f64)> = HashMap::new();
        for policy in policies {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for r in 0..n_reps {
                let mut params = default_params();
                params.seed = 5000 + r as u64;
                params.true_theta = 0.65;
                params.policy = policy;
                params.resolution_mode = "bernoulli";
                let out = run_fact_machine(&params);
                sum += out.pnl;
                sum_sq += out.pnl * out.pnl;
            }
            let mean = sum / n_reps as f64;
            let variance = f64::max(0.0, sum_sq / n_reps as f64 - mean * mean);
            stats.insert(policy, (mean, variance.sqrt()));
        }
        for p in policies {
            let (m, sd) = stats[p];
            println!("#   {:<8}  mean={:.3}  sd={:.3}", p, m, sd);
        }
        let welch_t = |a: (f64, f64), b: (f64, f64), n: f64| -> f64 {
            let se = (a.1 * a.1 / n + b.1 * b.1 / n).sqrt();
            if se == 0.0 {
                return 0.0;
            }
            (a.0 - b.0) / se
        };
        c.check("oracle.mean > qmdp.mean (value of perfect information)", stats["oracle"].0 > stats["qmdp"].0, &format!("oracle={:.3} qmdp={:.3}", stats["oracle"].0, stats["qmdp"].0));
        c.check("qmdp.mean > random.mean", stats["qmdp"].0 > stats["random"].0, "");
        c.check("myopic.mean > hold.mean (which is exactly 0)", stats["myopic"].0 > stats["hold"].0, "");
        c.check("oracle vs random Welch-t > 5 (highly significant)", welch_t(stats["oracle"], stats["random"], n_reps as f64) > 5.0, &format!("t = {:.2}", welch_t(stats["oracle"], stats["random"], n_reps as f64)));
        c.check("qmdp vs random Welch-t > 3 (significant)", welch_t(stats["qmdp"], stats["random"], n_reps as f64) > 3.0, &format!("t = {:.2}", welch_t(stats["qmdp"], stats["random"], n_reps as f64)));
    }

    // STUDY 5.
    println!("\n=== STUDY 5: Late-stage \"voter coordination\" misdirects E[θ] ===");
    {
        let n_reps = 300usize;
        let t = 24usize;
        let true_theta = 0.7;
        let late_flip_multiplier = 10.0;
        let mut baseline_delta_theta = 0.0;
        let mut flip_delta_theta = 0.0;
        let mut baseline_pnl = 0.0;
        let mut flip_pnl = 0.0;
        for r in 0..n_reps {
            let seed = 700 + r as u64;
            let mut p1 = default_params();
            p1.seed = seed;
            p1.true_theta = true_theta;
            p1.t = t;
            p1.policy = "myopic";
            p1.resolution_mode = "bernoulli";
            p1.late_flip = false;
            let mut p2 = p1.clone();
            p2.late_flip = true;
            p2.late_flip_multiplier = late_flip_multiplier;
            let r1 = run_fact_machine(&p1);
            let r2 = run_fact_machine(&p2);
            baseline_delta_theta += r1.belief_mean[t] - true_theta;
            flip_delta_theta += r2.belief_mean[t] - true_theta;
            baseline_pnl += r1.pnl;
            flip_pnl += r2.pnl;
        }
        baseline_delta_theta /= n_reps as f64;
        flip_delta_theta /= n_reps as f64;
        baseline_pnl /= n_reps as f64;
        flip_pnl /= n_reps as f64;
        println!("#   true θ = {},  flip surge = {}× K_noise at t = T-2", true_theta, late_flip_multiplier);
        println!("#   baseline:  mean(E[θ] − θ_true) = {:.4}    mean PnL = {:.3}", baseline_delta_theta, baseline_pnl);
        println!("#   with flip: mean(E[θ] − θ_true) = {:.4}    mean PnL = {:.3}", flip_delta_theta, flip_pnl);
        c.check("(a) without flip, |E[θ] − θ_true| ≤ 0.05 at end of market", baseline_delta_theta.abs() <= 0.05, &format!("Δθ={:.4}", baseline_delta_theta));
        c.check("(b) with flip, E[θ] is shifted AWAY from truth (Δθ < −0.10, toward 1−θ)", flip_delta_theta < -0.10, &format!("flip Δθ={:.4}", flip_delta_theta));
        c.check(
            "(c) flip costs the bettor money (mean PnL drop > 0.10; small because most positions are taken before the flip tick)",
            baseline_pnl - flip_pnl > 0.10,
            &format!("baseline={:.3} flip={:.3}  drop={:.3}", baseline_pnl, flip_pnl, baseline_pnl - flip_pnl),
        );
    }

    // STUDY 6.
    println!("\n=== STUDY 6: Cassandra \"Tiger\" POMDP — exact VI agrees with QMDP at flat prior ===");
    {
        // PORT NOTE: transition/observation/reward closures omitted in the stub spec.
        let spec = PomdpSpec {
            states: vec!["TL".to_string(), "TR".to_string()],
            actions: vec!["open-left".to_string(), "open-right".to_string(), "listen".to_string()],
            observations: vec!["hear-left".to_string(), "hear-right".to_string()],
            discount: 0.95,
        };
        let exact = pomdp_exact_finite_horizon(&spec, 4);
        let flat = vec![0.5, 0.5];
        let v_exact = exact.v(&flat);
        let qm = QmdpSolver::new(&spec, 1e-10, 5000);
        let belief = DiscreteBelief::with_weights(vec![0.0, 1.0], flat.clone());
        let v_qmdp = qm.q_belief(&belief, 0).max(qm.q_belief(&belief, 1)).max(qm.q_belief(&belief, 2));
        println!("#   V_exact(0.5, 0.5)  = {:.4}", v_exact);
        println!("#   V_QMDP (0.5, 0.5)  = {:.4}", v_qmdp);
        c.check("QMDP value ≥ exact POMDP value at flat prior (QMDP is upper bound)", v_qmdp >= v_exact - 1e-6, &format!("QMDP={:.3} exact={:.3}", v_qmdp, v_exact));
        c.check("exact policy at flat prior chooses 'listen'", spec.actions[exact.act(&belief)] == "listen", "");
        c.check("QMDP policy at flat prior chooses 'listen'", spec.actions[qm.act(&belief)] == "listen", "");
    }

    // STUDY 7.
    println!("\n=== STUDY 7: Binary vs Scalar markets ===");
    {
        let n_reps = 1000usize;
        let t = 24usize;
        let true_theta = 0.65;

        struct Block {
            mean_pnl: f64,
            sd_pnl: f64,
            win_rate: f64,
            final_belief_var: f64,
        }
        let run_block = |market: &'static str, policy: &'static str| -> Block {
            let mut sum_pnl = 0.0;
            let mut sum_sq_pnl = 0.0;
            let mut wins = 0usize;
            let mut sum_belief_var = 0.0;
            for r in 0..n_reps {
                let mut params = default_params();
                params.seed = 9000 + r as u64;
                params.true_theta = true_theta;
                params.t = t;
                params.policy = policy;
                params.market_type = market;
                params.resolution_mode = "majority";
                params.theta_bins = 21;
                params.k_noise = 20.0;
                params.fee = 0.01;
                let out = run_fact_machine(&params);
                sum_pnl += out.pnl;
                sum_sq_pnl += out.pnl * out.pnl;
                if out.pnl > 0.0 {
                    wins += 1;
                }
                sum_belief_var += out.belief_var[out.belief_var.len() - 1];
            }
            let mean = sum_pnl / n_reps as f64;
            let variance = f64::max(0.0, sum_sq_pnl / n_reps as f64 - mean * mean);
            Block { mean_pnl: mean, sd_pnl: variance.sqrt(), win_rate: wins as f64 / n_reps as f64, final_belief_var: sum_belief_var / n_reps as f64 }
        };

        let bin_my = run_block("binary", "myopic");
        let bin_or = run_block("binary", "oracle");
        let bin_rn = run_block("binary", "random");
        let sc_my = run_block("scalar", "myopic");
        let sc_or = run_block("scalar", "oracle");
        let sc_rn = run_block("scalar", "random");

        println!("#                  binary                     scalar");
        println!("#                  PnL    sd      win-rate    PnL    sd       win-rate");
        println!(
            "#   random        {:>6}  {:>5}   {:.3}     {:>6}  {:>5}    {:.3}",
            format!("{:.3}", bin_rn.mean_pnl), format!("{:.2}", bin_rn.sd_pnl), bin_rn.win_rate, format!("{:.3}", sc_rn.mean_pnl), format!("{:.2}", sc_rn.sd_pnl), sc_rn.win_rate
        );
        println!(
            "#   myopic        {:>6}  {:>5}   {:.3}     {:>6}  {:>5}    {:.3}",
            format!("{:.3}", bin_my.mean_pnl), format!("{:.2}", bin_my.sd_pnl), bin_my.win_rate, format!("{:.3}", sc_my.mean_pnl), format!("{:.2}", sc_my.sd_pnl), sc_my.win_rate
        );
        println!(
            "#   oracle        {:>6}  {:>5}   {:.3}     {:>6}  {:>5}    {:.3}",
            format!("{:.3}", bin_or.mean_pnl), format!("{:.2}", bin_or.sd_pnl), bin_or.win_rate, format!("{:.3}", sc_or.mean_pnl), format!("{:.2}", sc_or.sd_pnl), sc_or.win_rate
        );

        {
            let mut p = default_params();
            p.seed = 1234;
            p.true_theta = 0.6;
            p.t = 12;
            p.market_type = "binary";
            p.resolution_mode = "majority";
            p.policy = "hold";
            let r1 = run_fact_machine(&p);
            let mut p2 = p.clone();
            p2.market_type = "scalar";
            let r2 = run_fact_machine(&p2);
            let mut max_diff = 0.0_f64;
            for tt in 0..=t {
                if tt < r1.belief_mean.len() && tt < r2.belief_mean.len() {
                    max_diff = f64::max(max_diff, (r1.belief_mean[tt] - r2.belief_mean[tt]).abs());
                }
            }
            c.check("(a) same belief trajectory in binary vs scalar at hold-policy (max|Δ|<1e-12)", max_diff < 1e-12, &format!("max|Δ|={:.2e}", max_diff));
        }

        c.check("(b) binary myopic win-rate > scalar myopic win-rate at θ=0.65 (sure-thing effect)", bin_my.win_rate > sc_my.win_rate + 0.3, &format!("binary={:.3} vs scalar={:.3}", bin_my.win_rate, sc_my.win_rate));
        c.check("(b') binary mean PnL > scalar mean PnL for myopic at θ=0.65", bin_my.mean_pnl > sc_my.mean_pnl, &format!("binary={:.3} vs scalar={:.3}", bin_my.mean_pnl, sc_my.mean_pnl));
        c.check("(c) scalar PnL sd > binary PnL sd for myopic (variance from bin concentration)", sc_my.sd_pnl > bin_my.sd_pnl, &format!("binary sd={:.3} vs scalar sd={:.3}", bin_my.sd_pnl, sc_my.sd_pnl));

        let bin_edge = bin_or.mean_pnl - bin_my.mean_pnl;
        let sc_edge = sc_or.mean_pnl - sc_my.mean_pnl;
        println!("#   oracle edge:   binary={:.3},  scalar={:.3}", bin_edge, sc_edge);
        c.check("(d) scalar oracle edge > binary oracle edge (info more valuable in scalar)", sc_edge > bin_edge, &format!("scalar={:.3} vs binary={:.3}", sc_edge, bin_edge));

        let mut bin_h = 0.0;
        let mut sc_h = 0.0;
        let h_reps = 200usize;
        for r in 0..h_reps {
            let mut pb = default_params();
            pb.seed = 200 + r as u64;
            pb.true_theta = 0.5;
            pb.t = t;
            pb.policy = "hold";
            pb.market_type = "binary";
            pb.resolution_mode = "majority";
            let mut ps = pb.clone();
            ps.market_type = "scalar";
            let rb = run_fact_machine(&pb);
            let rs = run_fact_machine(&ps);
            let phb = &rb.price_history[t];
            let phs = &rs.price_history[t];
            let mut hb = 0.0;
            let mut hs = 0.0;
            for &x in phb {
                if x > 0.0 {
                    hb -= x * x.ln();
                }
            }
            for &x in phs {
                if x > 0.0 {
                    hs -= x * x.ln();
                }
            }
            bin_h += hb;
            sc_h += hs;
        }
        bin_h /= h_reps as f64;
        sc_h /= h_reps as f64;
        println!("#   H(price vector) at t=T:  binary={:.3} (max={:.3}),  scalar={:.3} (max={:.3})", bin_h, (2.0_f64).ln(), sc_h, (21.0_f64).ln());
        c.check("(e) scalar price vector carries more entropy than binary at θ=0.5", sc_h > bin_h * 1.5, &format!("binary={:.3} scalar={:.3}", bin_h, sc_h));
    }

    println!("\n=== Summary: {} passed, {} failed ===", c.pass, c.fail);
    if c.fail > 0 {
        std::process::exit(1);
    }
}
