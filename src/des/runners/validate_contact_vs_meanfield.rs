//! Port of `src/des/runners/validate-contact-vs-meanfield.ts`.
//!
//! Validates the contact-SEIR kernels against each other and the mass-action
//! mean-field in three studies (N→∞ convergence, heterogeneity/super-spreader
//! Gini, triplet threshold). The TS top-level driver becomes [`run`].
//!
//! ## PORT NOTE — production contact-SEIR kernel
//!
//! The early Rust port reproduced the Contact-SEIR kernel locally before
//! `main_contact_seir` existed. This validator now calls the production
//! [`crate::des::main_contact_seir::run_contact_seir`] implementation directly
//! and keeps only the study/statistics code local.

#![allow(dead_code)]

use crate::des::main_contact_seir::{
    run_contact_seir as run_production_contact_seir, ContactSEIRParams as ContactSeirParams,
    ContactSEIRResult as ContactSeirResult, Kernel,
};

fn run_contact_seir(params: ContactSeirParams) -> ContactSeirResult {
    run_production_contact_seir(&params, None)
}

// =============================================================================
// Validator statistics (local copies, matching the TS file).
// =============================================================================

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let denom = (xs.len() as f64 - 1.0).max(1.0);
    xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / denom
}

struct Welch {
    t: f64,
    p: f64,
}

fn welch(xs: &[f64], ys: &[f64]) -> Welch {
    let mx = mean(xs);
    let my = mean(ys);
    let vx = variance(xs);
    let vy = variance(ys);
    let se = (vx / xs.len() as f64 + vy / ys.len() as f64).sqrt();
    let t = if se == 0.0 { 0.0 } else { (mx - my) / se };
    let z = t.abs();
    let phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    Welch {
        t,
        p: 2.0 * (1.0 - phi),
    }
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let tt = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * (-x * x).exp();
    sign * y
}

fn gini(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let s: f64 = sorted.iter().sum();
    if s == 0.0 {
        return 0.0;
    }
    let mut cum = 0.0;
    for (i, v) in sorted.iter().enumerate() {
        cum += (i as f64 + 1.0) * v;
    }
    (2.0 * cum) / (n as f64 * s) - (n as f64 + 1.0) / n as f64
}

fn share_top_k(xs: &[f64], k: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = (xs.len() as f64 * k).floor() as usize;
    let total: f64 = sorted.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let top: f64 = sorted.iter().take(cutoff).sum();
    top / total
}

// -----------------------------------------------------------------------------
// Formatting helpers.
// -----------------------------------------------------------------------------

fn fixed(n: f64, digits: usize) -> String {
    format!("{n:.digits$}")
}

fn pad_start(s: &str, len: usize) -> String {
    let count = s.chars().count();
    if count >= len {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(len - count))
    }
}

/// `${n}` for a JS number — integers print without a decimal point.
fn js_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

struct Counter {
    pass: usize,
    fail: usize,
}

impl Counter {
    fn check(&mut self, label: &str, ok: bool, detail: Option<String>) {
        if ok {
            self.pass += 1;
            println!("  PASS    {label}");
        } else {
            self.fail += 1;
            let d = detail.map(|x| format!("  ({x})")).unwrap_or_default();
            println!("  FAIL    {label}{d}");
        }
    }
}

fn baseline(n: usize, kernel: Kernel, seed: u32) -> ContactSeirParams {
    ContactSeirParams {
        n,
        initial_i: 10,
        contact_rate: 6.0,
        contact_rate_cv: 0.0,
        p_transmit: 0.05,
        sigma: 1.0 / 5.2,
        gamma: 1.0 / 7.0,
        sim_t: 120.0,
        step_size: 0.1,
        seed,
        kernel,
    }
}

/// `main` — returns the exit code (0 = all checks pass).
pub fn run() -> i32 {
    let mut c = Counter { pass: 0, fail: 0 };

    // -------------------------------------------------------------------------
    // STUDY 1: convergence mass-action ≡ pairwise as N → ∞.
    // -------------------------------------------------------------------------
    println!("\nStudy 1  Convergence: mass-action ≡ pairwise as N → ∞");
    println!("==========================================================================");
    {
        const REPS: u32 = 12;
        for n in [500usize, 2000, 5000] {
            let mut mass_attack = Vec::new();
            let mut pair_attack = Vec::new();
            let mut mass_r0 = Vec::new();
            let mut pair_r0 = Vec::new();
            for r in 0..REPS {
                let seed = 1 + r;
                let mass = run_contact_seir(baseline(n, Kernel::MassAction, seed));
                let pair = run_contact_seir(baseline(n, Kernel::Pairwise, seed));
                mass_attack.push(mass.final_attack_rate);
                pair_attack.push(pair.final_attack_rate);
                mass_r0.push(mass.r0_index_only);
                pair_r0.push(pair.r0_index_only);
            }
            let t_attack = welch(&mass_attack, &pair_attack);
            let t_r0 = welch(&mass_r0, &pair_r0);
            let a_mass = mean(&mass_attack);
            let a_pair = mean(&pair_attack);
            let r0_mass = mean(&mass_r0);
            let r0_pair = mean(&pair_r0);
            println!(
                "  N={}  attack: mass={}% pair={}%  Welch p={}    R₀(idx): mass={} pair={}  Welch p={}",
                pad_start(&n.to_string(), 5),
                fixed(a_mass * 100.0, 1),
                fixed(a_pair * 100.0, 1),
                fixed(t_attack.p, 3),
                fixed(r0_mass, 2),
                fixed(r0_pair, 2),
                fixed(t_r0.p, 3),
            );
            let _ = t_attack.t;
            let _ = t_r0.t;
            let tol_p = if n >= 2000 { 0.05 } else { 0.01 };
            c.check(
                &format!("N={n}: attack-rate Welch p > {}", js_num(tol_p)),
                t_attack.p > tol_p,
                Some(format!("p={}", fixed(t_attack.p, 3))),
            );
            c.check(
                &format!("N={n}: R₀(index) Welch p > {}", js_num(tol_p)),
                t_r0.p > tol_p,
                Some(format!("p={}", fixed(t_r0.p, 3))),
            );
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 2: heterogeneity / super-spreader Gini.
    // -------------------------------------------------------------------------
    println!("\nStudy 2  Heterogeneity: super-spreader effect (Gini coefficient of offspring)");
    println!("==========================================================================");
    println!("  Theoretical: with heterogeneous contact rates, a small fraction of cases");
    println!("  produces a large fraction of secondary infections — the 20/80 rule.");
    println!("  Mean-field (mass-action) cannot reproduce this because infectors are");
    println!("  selected uniformly at random; the offspring distribution stays Poisson");
    println!("  regardless of CV. Symmetric pairwise reproduces it because high-c");
    println!("  individuals BOTH initiate more contacts AND are partner more often,");
    println!("  multiplicatively increasing their per-individual offspring count.");
    println!("  We measure: Gini coefficient of offspring distribution, and \"share of");
    println!("  secondaries from top 20% of cases\".");
    {
        let n = 5000usize;
        const REPS: u32 = 20;
        println!("  CV    pairwise Gini   pairwise top-20% share    mass-action Gini   mass-action top-20% share");
        println!("  ────  ─────────────   ──────────────────────    ────────────────   ─────────────────────────");
        let mut pairwise_high_cv_gini = 0.0;
        let mut mass_action_high_cv_gini = 0.0;
        for cv in [0.0, 0.5, 1.0, 2.0] {
            let mut ginis_p = Vec::new();
            let mut shares_p = Vec::new();
            let mut ginis_m = Vec::new();
            let mut shares_m = Vec::new();
            for r in 0..REPS {
                let mut pp = baseline(n, Kernel::Pairwise, 1 + r);
                pp.initial_i = 5;
                pp.contact_rate_cv = cv;
                let mut mp = baseline(n, Kernel::MassAction, 1 + r);
                mp.initial_i = 5;
                mp.contact_rate_cv = cv;
                let pair = run_contact_seir(pp);
                let mass = run_contact_seir(mp);
                let o_p: Vec<f64> = pair
                    .per_person
                    .iter()
                    .filter(|p| p.ever)
                    .map(|p| p.offspring)
                    .collect();
                let o_m: Vec<f64> = mass
                    .per_person
                    .iter()
                    .filter(|p| p.ever)
                    .map(|p| p.offspring)
                    .collect();
                ginis_p.push(gini(&o_p));
                shares_p.push(share_top_k(&o_p, 0.2));
                ginis_m.push(gini(&o_m));
                shares_m.push(share_top_k(&o_m, 0.2));
            }
            let g_p = mean(&ginis_p);
            let g_m = mean(&ginis_m);
            let s_p = mean(&shares_p);
            let s_m = mean(&shares_m);
            if cv == 2.0 {
                pairwise_high_cv_gini = g_p;
                mass_action_high_cv_gini = g_m;
            }
            println!(
                "  {}   {}           {}              {}              {}",
                pad_start(&fixed(cv, 1), 3),
                fixed(g_p, 3),
                pad_start(&format!("{}%", fixed(s_p * 100.0, 1)), 14),
                fixed(g_m, 3),
                pad_start(&format!("{}%", fixed(s_m * 100.0, 1)), 14),
            );
            if cv == 0.0 {
                c.check(
                    "CV=0: pairwise Gini ≈ mass-action Gini",
                    (g_p - g_m).abs() < 0.10,
                    Some(format!("pair={} mass={}", fixed(g_p, 3), fixed(g_m, 3))),
                );
            } else {
                c.check(
                    &format!("CV={}: pairwise Gini > mass-action Gini", js_num(cv)),
                    g_p > g_m,
                    Some(format!("pair={} mass={}", fixed(g_p, 3), fixed(g_m, 3))),
                );
            }
        }
        c.check(
            "CV=2: pairwise Gini > 0.6 (heavy super-spreader regime)",
            pairwise_high_cv_gini > 0.6,
            Some(format!("Gini={}", fixed(pairwise_high_cv_gini, 3))),
        );
        c.check(
            "CV=2: mass-action Gini < pairwise Gini by > 0.05",
            pairwise_high_cv_gini - mass_action_high_cv_gini > 0.05,
            Some(format!(
                "pair={} mass={}",
                fixed(pairwise_high_cv_gini, 3),
                fixed(mass_action_high_cv_gini, 3)
            )),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 3: triplet sharp threshold; pairwise does not.
    // -------------------------------------------------------------------------
    println!("\nStudy 3  Triplet has a sharp threshold; pairwise does not");
    println!("==========================================================================");
    println!("  Sweep I₀ from 5 → 1000 (in N=5000) and measure final attack rate.");
    println!("  Pairwise: epidemic ignites at any I₀ ≥ 1 (linear-in-I₀ early growth).");
    println!("  Triplet: epidemic needs I₀ above a critical density (quadratic-in-I₀).");
    {
        let n = 5000usize;
        const REPS: u32 = 6;
        let i0s = [5usize, 50, 200, 500, 1000];
        println!("   I₀     I₀/N     pairwise-attack    triplet-attack");
        println!("  ─────  ───────   ──────────────     ──────────────");
        let mut pairwise_always_high = true;
        let mut triplet_starts_low = false;
        let mut triplet_ends_high = false;
        for &initial_i in &i0s {
            let mut pa = Vec::new();
            let mut ta = Vec::new();
            for r in 0..REPS {
                let mut pp = baseline(n, Kernel::Pairwise, 1 + r);
                pp.initial_i = initial_i;
                pa.push(run_contact_seir(pp).final_attack_rate);

                // tripletParams = {...baseline, contactRate: 30, pTransmit: 0.05}.
                let mut tp = baseline(n, Kernel::Triplet, 1 + r);
                tp.initial_i = initial_i;
                tp.contact_rate = 30.0;
                tp.p_transmit = 0.05;
                ta.push(run_contact_seir(tp).final_attack_rate);
            }
            let p_avg = mean(&pa) * 100.0;
            let t_avg = mean(&ta) * 100.0;
            println!(
                "  {}  {}    {}      {}",
                pad_start(&initial_i.to_string(), 5),
                fixed(initial_i as f64 / n as f64, 4),
                pad_start(&format!("{}%", fixed(p_avg, 1)), 14),
                pad_start(&format!("{}%", fixed(t_avg, 1)), 14),
            );
            if p_avg < 30.0 {
                pairwise_always_high = false;
            }
            if initial_i == i0s[0] && t_avg < 5.0 {
                triplet_starts_low = true;
            }
            if initial_i == i0s[i0s.len() - 1] && t_avg > 50.0 {
                triplet_ends_high = true;
            }
        }
        c.check(
            "pairwise attack rate > 30% for all I₀",
            pairwise_always_high,
            None,
        );
        c.check(
            "triplet attack rate < 5% at smallest I₀",
            triplet_starts_low,
            None,
        );
        c.check(
            "triplet attack rate > 50% at largest I₀",
            triplet_ends_high,
            None,
        );
    }

    println!("\nsummary: {} pass, {} fail", c.pass, c.fail);
    if c.fail == 0 {
        0
    } else {
        1
    }
}
