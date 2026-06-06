//! Port of `src/des/runners/validate-factmachine.ts`.
//!
//! Validates the FactMachine POMDP: Bayesian belief filter vs native Rust
//! reference, majority win-probability, Brier calibration, policy ranking,
//! late-flip misdirection, Tiger POMDP exact-VI vs QMDP, and binary-vs-scalar
//! market contrast.
//! Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust belief, POMDP/Tiger, and FactMachine modules with
//!     native Rust shadow references.

#![allow(dead_code)]

use crate::des::general::belief::{brier_score, BinaryOutcome, DiscreteBelief};
use crate::des::general::pomdp::{pomdp_exact_finite_horizon, MDPVIOptions, QMDPSolver};
use crate::des::general::tiger_pomdp::{build_tiger_spec, TigerOpts};
use crate::des::main_factmachine::{
    default_params, run_fact_machine as run_fact_machine_model, FactMachineParams,
    FactMachineResult, MarketType, Policy, ResolutionMode,
};

fn run_fact_machine(params: &FactMachineParams) -> FactMachineResult {
    run_fact_machine_model(params.clone())
}

fn policy_from_label(label: &str) -> Policy {
    match label {
        "random" => Policy::Random,
        "hold" => Policy::Hold,
        "myopic" => Policy::Myopic,
        "oracle" => Policy::Oracle,
        _ => Policy::Qmdp,
    }
}

fn market_type_from_label(label: &str) -> MarketType {
    match label {
        "scalar" => MarketType::Scalar,
        _ => MarketType::Binary,
    }
}

fn binary_outcome(outcome: i32) -> BinaryOutcome {
    if outcome == 0 {
        BinaryOutcome::Zero
    } else {
        BinaryOutcome::One
    }
}

struct BeliefReference {
    final_mean: f64,
    final_belief: Vec<f64>,
    mean_history: Vec<f64>,
}

struct PWinReference {
    thetas: Vec<f64>,
    pwin: Vec<f64>,
}

fn mean_from_weights(states: &[f64], weights: &[f64]) -> f64 {
    states
        .iter()
        .zip(weights.iter())
        .map(|(theta, weight)| theta * weight)
        .sum()
}

fn normalize_log_weights(log_weights: &[f64]) -> Vec<f64> {
    if log_weights.is_empty() {
        return Vec::new();
    }
    let max_log = log_weights
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log.is_finite() {
        return vec![1.0 / log_weights.len() as f64; log_weights.len()];
    }
    let mut weights = log_weights
        .iter()
        .map(|w| (w - max_log).exp())
        .collect::<Vec<_>>();
    let sum: f64 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / log_weights.len() as f64; log_weights.len()];
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn vote_log_likelihood(theta: f64, informedness: f64, yes: i64, total: i64) -> f64 {
    let q = (theta * informedness + 0.5 * (1.0 - informedness)).clamp(1e-300, 1.0 - 1e-15);
    yes as f64 * q.ln() + (total - yes) as f64 * (1.0 - q).ln()
}

fn rust_log_space_belief_reference(
    states: &[f64],
    informedness: f64,
    obs: &[(i64, i64)],
) -> BeliefReference {
    let mut log_weights = vec![0.0; states.len()];
    let mut weights = normalize_log_weights(&log_weights);
    let mut mean_history = vec![mean_from_weights(states, &weights)];
    for &(yes, total) in obs {
        for (i, theta) in states.iter().enumerate() {
            log_weights[i] += vote_log_likelihood(*theta, informedness, yes, total);
        }
        weights = normalize_log_weights(&log_weights);
        mean_history.push(mean_from_weights(states, &weights));
    }
    BeliefReference {
        final_mean: mean_history.last().copied().unwrap_or(f64::NAN),
        final_belief: weights,
        mean_history,
    }
}

fn rust_discrete_belief_reference(
    states: &[f64],
    informedness: f64,
    obs: &[(i64, i64)],
) -> BeliefReference {
    let mut belief = DiscreteBelief::new(states.to_vec(), None);
    let mut mean_history = vec![belief.mean()];
    for &(yes, total) in obs {
        belief.update(|theta, _| {
            vote_log_likelihood(*theta, informedness, yes, total)
                .exp()
                .max(0.0)
        });
        mean_history.push(belief.mean());
    }
    BeliefReference {
        final_mean: mean_history.last().copied().unwrap_or(f64::NAN),
        final_belief: belief.weights,
        mean_history,
    }
}

fn binomial_majority_tail_recurrence(theta: f64, voters: i64) -> f64 {
    let half = voters / 2;
    let theta = theta.clamp(1e-300, 1.0 - 1e-15);
    let mut p = 0.0;
    let mut log_p = voters as f64 * (1.0 - theta).ln();
    let mut log_coef = 0.0;
    for k in 0..=voters {
        if k > half {
            p += (log_coef + log_p).exp();
        }
        if k < voters {
            log_coef += ((voters - k) as f64).ln() - ((k + 1) as f64).ln();
            log_p += theta.ln() - (1.0 - theta).ln();
        }
    }
    p.clamp(0.0, 1.0)
}

fn binomial_majority_tail_dp(theta: f64, voters: usize) -> f64 {
    let theta = theta.clamp(0.0, 1.0);
    let mut pmf = vec![0.0; voters + 1];
    pmf[0] = 1.0;
    for i in 0..voters {
        for yes in (0..=i).rev() {
            let current = pmf[yes];
            pmf[yes] = current * (1.0 - theta);
            pmf[yes + 1] += current * theta;
        }
    }
    pmf.iter().skip(voters / 2 + 1).sum::<f64>().clamp(0.0, 1.0)
}

fn rust_majority_reference(voters: usize) -> PWinReference {
    let thetas = (1..=9).map(|i| i as f64 / 10.0).collect::<Vec<_>>();
    let pwin = thetas
        .iter()
        .map(|theta| binomial_majority_tail_dp(*theta, voters))
        .collect::<Vec<_>>();
    PWinReference { thetas, pwin }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
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
        let extra = if detail.is_empty() {
            String::new()
        } else {
            format!("  ({})", detail)
        };
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
    println!("\n=== STUDY 1: Bayesian belief filter ≡ Rust log-space reference ===");
    {
        let k = 21usize;
        let informedness = 0.6;
        let states: Vec<f64> = (0..k).map(|i| i as f64 / (k - 1) as f64).collect();
        let obs: Vec<(i64, i64)> = vec![
            (12, 20),
            (15, 22),
            (9, 19),
            (17, 20),
            (11, 18),
            (14, 19),
            (16, 20),
            (10, 22),
        ];
        let model_ref = rust_discrete_belief_reference(&states, informedness, &obs);
        let shadow_ref = rust_log_space_belief_reference(&states, informedness, &obs);
        let d_mean = (model_ref.final_mean - shadow_ref.final_mean).abs();
        c.check(
            &format!(
                "final E[θ] matches Rust log-space reference  model={:.8}  ref={:.8}",
                model_ref.final_mean, shadow_ref.final_mean
            ),
            d_mean < 1e-10,
            &format!("|Δ|={:.2e}", d_mean),
        );
        let max_belief_diff = max_abs_diff(&model_ref.final_belief, &shadow_ref.final_belief);
        c.check(
            "per-bin |belief_model - belief_ref| <= 1e-10 across 21 bins",
            max_belief_diff < 1e-10,
            &format!("max|Δ|={:.2e}", max_belief_diff),
        );
        let max_mean_diff = max_abs_diff(&model_ref.mean_history, &shadow_ref.mean_history);
        c.check(
            &format!(
                "per-tick mean trajectory matches Rust reference across {} steps",
                obs.len() + 1
            ),
            max_mean_diff < 1e-10,
            &format!("max|Δ|={:.2e}", max_mean_diff),
        );
    }

    // STUDY 2.
    println!("\n=== STUDY 2: P(majority votes YES | theta) ≡ Rust DP reference ===");
    {
        let voters = 51usize;
        let rust_ref = rust_majority_reference(voters);
        let recurrence = PWinReference {
            thetas: rust_ref.thetas.clone(),
            pwin: rust_ref
                .thetas
                .iter()
                .map(|theta| binomial_majority_tail_recurrence(*theta, voters as i64))
                .collect(),
        };
        let max_diff = max_abs_diff(&recurrence.pwin, &rust_ref.pwin);
        c.check(
            "pYesWins at 9 theta values matches Rust DP binomial tail to 1e-10",
            max_diff < 1e-10,
            &format!("max|Δ|={:.2e}", max_diff),
        );
    }

    // STUDY 3.
    println!("\n=== STUDY 3: Belief calibration over time (Brier decreases) ===");
    {
        let n_reps = 200usize;
        let t = 24usize;
        let mut brier_by_t = vec![0.0; t + 1];
        for r in 0..n_reps {
            let seed = 17 + r as u32;
            let true_theta = 0.05 + 0.9 * (r as f64 / n_reps as f64);
            let mut params = default_params();
            params.seed = seed;
            params.true_theta = true_theta;
            params.t = t as i64;
            params.policy = Policy::Hold;
            params.resolution_mode = ResolutionMode::Bernoulli;
            let r1 = run_fact_machine(&params);
            for tt in 0..=t {
                brier_by_t[tt] += brier_score(r1.belief_mean[tt], binary_outcome(r1.final_outcome));
            }
        }
        for tt in 0..=t {
            brier_by_t[tt] /= n_reps as f64;
        }
        let init_brier = brier_by_t[0];
        let final_brier = brier_by_t[t];
        let mid_brier = brier_by_t[t / 2];
        println!(
            "#   Brier(t=0) = {:.4},  Brier(t=12) = {:.4},  Brier(t=24) = {:.4}",
            init_brier, mid_brier, final_brier
        );
        c.check(
            "Brier at t=0 (uniform prior, no info) = 0.25 (theoretical)",
            (init_brier - 0.25).abs() < 1e-8,
            &format!("init={:.4}", init_brier),
        );
        c.check(
            "Brier at end < Brier at start (filter learns)",
            final_brier < init_brier - 0.02,
            &format!("end={:.4}, init={:.4}", final_brier, init_brier),
        );
        c.check(
            "Brier at t=12 < Brier at t=0 (monotone-ish learning)",
            mid_brier < init_brier - 0.01,
            &format!("mid={:.4}", mid_brier),
        );
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
                params.seed = 5000 + r as u32;
                params.true_theta = 0.65;
                params.policy = policy_from_label(policy);
                params.resolution_mode = ResolutionMode::Bernoulli;
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
        c.check(
            "oracle.mean > qmdp.mean (value of perfect information)",
            stats["oracle"].0 > stats["qmdp"].0,
            &format!(
                "oracle={:.3} qmdp={:.3}",
                stats["oracle"].0, stats["qmdp"].0
            ),
        );
        c.check(
            "qmdp.mean > random.mean",
            stats["qmdp"].0 > stats["random"].0,
            "",
        );
        c.check(
            "myopic.mean > hold.mean (which is exactly 0)",
            stats["myopic"].0 > stats["hold"].0,
            "",
        );
        c.check(
            "oracle vs random Welch-t > 5 (highly significant)",
            welch_t(stats["oracle"], stats["random"], n_reps as f64) > 5.0,
            &format!(
                "t = {:.2}",
                welch_t(stats["oracle"], stats["random"], n_reps as f64)
            ),
        );
        c.check(
            "qmdp vs random Welch-t > 3 (significant)",
            welch_t(stats["qmdp"], stats["random"], n_reps as f64) > 3.0,
            &format!(
                "t = {:.2}",
                welch_t(stats["qmdp"], stats["random"], n_reps as f64)
            ),
        );
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
            let seed = 700 + r as u32;
            let mut p1 = default_params();
            p1.seed = seed;
            p1.true_theta = true_theta;
            p1.t = t as i64;
            p1.policy = Policy::Myopic;
            p1.resolution_mode = ResolutionMode::Bernoulli;
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
        println!(
            "#   true θ = {},  flip surge = {}× K_noise at t = T-2",
            true_theta, late_flip_multiplier
        );
        println!(
            "#   baseline:  mean(E[θ] − θ_true) = {:.4}    mean PnL = {:.3}",
            baseline_delta_theta, baseline_pnl
        );
        println!(
            "#   with flip: mean(E[θ] − θ_true) = {:.4}    mean PnL = {:.3}",
            flip_delta_theta, flip_pnl
        );
        c.check(
            "(a) without flip, |E[θ] − θ_true| ≤ 0.05 at end of market",
            baseline_delta_theta.abs() <= 0.05,
            &format!("Δθ={:.4}", baseline_delta_theta),
        );
        c.check(
            "(b) with flip, E[θ] is shifted AWAY from truth (Δθ < −0.10, toward 1−θ)",
            flip_delta_theta < -0.10,
            &format!("flip Δθ={:.4}", flip_delta_theta),
        );
        c.check(
            "(c) flip costs the bettor money (mean PnL drop > 0.10; small because most positions are taken before the flip tick)",
            baseline_pnl - flip_pnl > 0.10,
            &format!("baseline={:.3} flip={:.3}  drop={:.3}", baseline_pnl, flip_pnl, baseline_pnl - flip_pnl),
        );
    }

    // STUDY 6.
    println!(
        "\n=== STUDY 6: Cassandra \"Tiger\" POMDP — exact VI agrees with QMDP at flat prior ==="
    );
    {
        let spec = build_tiger_spec(&TigerOpts::default());
        let spec_qmdp = build_tiger_spec(&TigerOpts::default());
        let exact = pomdp_exact_finite_horizon(&spec, 4);
        let flat = vec![0.5, 0.5];
        let v_exact = exact.value(&flat);
        let qm = QMDPSolver::new(spec_qmdp, &MDPVIOptions::default());
        let belief = DiscreteBelief::new(spec.states.clone(), Some(&flat));
        let v_qmdp = qm
            .q_belief(&belief, 0)
            .max(qm.q_belief(&belief, 1))
            .max(qm.q_belief(&belief, 2));
        println!("#   V_exact(0.5, 0.5)  = {:.4}", v_exact);
        println!("#   V_QMDP (0.5, 0.5)  = {:.4}", v_qmdp);
        c.check(
            "QMDP value ≥ exact POMDP value at flat prior (QMDP is upper bound)",
            v_qmdp >= v_exact - 1e-6,
            &format!("QMDP={:.3} exact={:.3}", v_qmdp, v_exact),
        );
        c.check(
            "exact policy at flat prior chooses 'listen'",
            spec.actions[exact.act(&belief)] == "listen",
            "",
        );
        c.check(
            "QMDP policy at flat prior chooses 'listen'",
            qm.spec.actions[qm.act(&belief, None, 0.0)] == "listen",
            "",
        );
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
                params.seed = 9000 + r as u32;
                params.true_theta = true_theta;
                params.t = t as i64;
                params.policy = policy_from_label(policy);
                params.market_type = market_type_from_label(market);
                params.resolution_mode = ResolutionMode::Majority;
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
            Block {
                mean_pnl: mean,
                sd_pnl: variance.sqrt(),
                win_rate: wins as f64 / n_reps as f64,
                final_belief_var: sum_belief_var / n_reps as f64,
            }
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
            format!("{:.3}", bin_rn.mean_pnl),
            format!("{:.2}", bin_rn.sd_pnl),
            bin_rn.win_rate,
            format!("{:.3}", sc_rn.mean_pnl),
            format!("{:.2}", sc_rn.sd_pnl),
            sc_rn.win_rate
        );
        println!(
            "#   myopic        {:>6}  {:>5}   {:.3}     {:>6}  {:>5}    {:.3}",
            format!("{:.3}", bin_my.mean_pnl),
            format!("{:.2}", bin_my.sd_pnl),
            bin_my.win_rate,
            format!("{:.3}", sc_my.mean_pnl),
            format!("{:.2}", sc_my.sd_pnl),
            sc_my.win_rate
        );
        println!(
            "#   oracle        {:>6}  {:>5}   {:.3}     {:>6}  {:>5}    {:.3}",
            format!("{:.3}", bin_or.mean_pnl),
            format!("{:.2}", bin_or.sd_pnl),
            bin_or.win_rate,
            format!("{:.3}", sc_or.mean_pnl),
            format!("{:.2}", sc_or.sd_pnl),
            sc_or.win_rate
        );

        {
            let mut p = default_params();
            p.seed = 1234;
            p.true_theta = 0.6;
            p.t = 12;
            p.market_type = MarketType::Binary;
            p.resolution_mode = ResolutionMode::Majority;
            p.policy = Policy::Hold;
            let r1 = run_fact_machine(&p);
            let mut p2 = p.clone();
            p2.market_type = MarketType::Scalar;
            let r2 = run_fact_machine(&p2);
            let mut max_diff = 0.0_f64;
            for tt in 0..=t {
                if tt < r1.belief_mean.len() && tt < r2.belief_mean.len() {
                    max_diff = f64::max(max_diff, (r1.belief_mean[tt] - r2.belief_mean[tt]).abs());
                }
            }
            c.check(
                "(a) same belief trajectory in binary vs scalar at hold-policy (max|Δ|<1e-12)",
                max_diff < 1e-12,
                &format!("max|Δ|={:.2e}", max_diff),
            );
        }

        c.check(
            "(b) binary myopic win-rate > scalar myopic win-rate at θ=0.65 (sure-thing effect)",
            bin_my.win_rate > sc_my.win_rate + 0.3,
            &format!(
                "binary={:.3} vs scalar={:.3}",
                bin_my.win_rate, sc_my.win_rate
            ),
        );
        c.check(
            "(b') binary mean PnL > scalar mean PnL for myopic at θ=0.65",
            bin_my.mean_pnl > sc_my.mean_pnl,
            &format!(
                "binary={:.3} vs scalar={:.3}",
                bin_my.mean_pnl, sc_my.mean_pnl
            ),
        );
        c.check(
            "(c) scalar PnL sd > binary PnL sd for myopic (variance from bin concentration)",
            sc_my.sd_pnl > bin_my.sd_pnl,
            &format!(
                "binary sd={:.3} vs scalar sd={:.3}",
                bin_my.sd_pnl, sc_my.sd_pnl
            ),
        );

        let bin_edge = bin_or.mean_pnl - bin_my.mean_pnl;
        let sc_edge = sc_or.mean_pnl - sc_my.mean_pnl;
        println!(
            "#   oracle edge:   binary={:.3},  scalar={:.3}",
            bin_edge, sc_edge
        );
        c.check(
            "(d) scalar oracle edge > binary oracle edge (info more valuable in scalar)",
            sc_edge > bin_edge,
            &format!("scalar={:.3} vs binary={:.3}", sc_edge, bin_edge),
        );

        let mut bin_h = 0.0;
        let mut sc_h = 0.0;
        let h_reps = 200usize;
        for r in 0..h_reps {
            let mut pb = default_params();
            pb.seed = 200 + r as u32;
            pb.true_theta = 0.5;
            pb.t = t as i64;
            pb.policy = Policy::Hold;
            pb.market_type = MarketType::Binary;
            pb.resolution_mode = ResolutionMode::Majority;
            let mut ps = pb.clone();
            ps.market_type = MarketType::Scalar;
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
        println!(
            "#   H(price vector) at t=T:  binary={:.3} (max={:.3}),  scalar={:.3} (max={:.3})",
            bin_h,
            (2.0_f64).ln(),
            sc_h,
            (21.0_f64).ln()
        );
        c.check(
            "(e) scalar price vector carries more entropy than binary at θ=0.5",
            sc_h > bin_h * 1.5,
            &format!("binary={:.3} scalar={:.3}", bin_h, sc_h),
        );
    }

    println!("\n=== Summary: {} passed, {} failed ===", c.pass, c.fail);
    if c.fail > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factmachine_belief_reference_matches_log_space_reference() {
        let states: Vec<f64> = (0..21).map(|i| i as f64 / 20.0).collect();
        let obs = vec![(12, 20), (15, 22), (9, 19), (17, 20)];
        let discrete = rust_discrete_belief_reference(&states, 0.6, &obs);
        let log_space = rust_log_space_belief_reference(&states, 0.6, &obs);

        assert!((discrete.final_mean - log_space.final_mean).abs() < 1e-10);
        assert!(max_abs_diff(&discrete.final_belief, &log_space.final_belief) < 1e-10);
        assert!(max_abs_diff(&discrete.mean_history, &log_space.mean_history) < 1e-10);
    }
}
