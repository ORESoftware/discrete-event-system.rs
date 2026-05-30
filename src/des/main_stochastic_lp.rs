//! Port of `src/des/main-stochastic-lp.ts`.
//!
//! CLI: two-stage stochastic LP (2-product capacity planning under demand
//! uncertainty) solved three ways — closed-form newsvendor, SAA monolithic LP,
//! and Benders / L-shaped decomposition expressed as a DES.
//!
//! Conversion notes:
//!   - `process.env` params (`N`, `SEED`, `BUDGET`, `VERBOSE`) → `std::env::var`.
//!   - scenario sampling is seeded inside `general::stochastic_lp`.
//!   - `async main` → [`run`].
//!   - delegates to `general::stochastic_lp`.

use crate::des::general::stochastic_lp::{
    build_production_scenarios, build_production_slp, solve_production_closed_form,
    solve_slp_benders, solve_slp_monolithic, BendersOpts, BendersStopReason, SLPStatus,
    UniformDemandSpec,
};

fn pad(s: &str, n: usize) -> String {
    let len = s.chars().count();
    if len >= n {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(n - len))
    }
}

#[allow(dead_code)]
fn fmt(v: f64, w: usize) -> String {
    format!("{:>w$}", format!("{:.2}", v), w = w)
}

fn status_str(s: SLPStatus) -> &'static str {
    match s {
        SLPStatus::Optimal => "optimal",
        SLPStatus::Unbounded => "unbounded",
        SLPStatus::Infeasible => "infeasible",
        SLPStatus::IterLimit => "iter-limit",
    }
}

fn stop_reason_str(s: BendersStopReason) -> &'static str {
    match s {
        BendersStopReason::Converged => "converged",
        BendersStopReason::IterLimit => "iter-limit",
        BendersStopReason::SubproblemError => "subproblem-error",
    }
}

/// `[a, b].map(v => v.toFixed(d)).join(', ')` wrapped in brackets.
fn vec_fixed(xs: &[f64], d: usize) -> String {
    let parts: Vec<String> = xs.iter().map(|v| format!("{:.*}", d, v)).collect();
    parts.join(", ")
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let seed: u32 = std::env::var("SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(42);
    let budget: Option<f64> = std::env::var("BUDGET").ok().and_then(|v| v.parse().ok());
    let verbose: bool = std::env::var("VERBOSE").map(|v| v == "1").unwrap_or(false);

    let c: Vec<f64> = vec![10.0, 12.0];
    let p: Vec<f64> = vec![25.0, 28.0];
    let ranges: Vec<(f64, f64)> = vec![(50.0, 100.0), (40.0, 80.0)];

    println!("# Two-stage stochastic LP — capacity planning under demand uncertainty");
    println!("#   first-stage cost c    = [{}]", vec_fixed_int(&c));
    println!("#   second-stage revenue p = [{}]", vec_fixed_int(&p));
    println!(
        "#   demand D_i ~ Uniform{}",
        ranges
            .iter()
            .map(|r| format!("[{}, {}]", num_str(r.0), num_str(r.1)))
            .collect::<Vec<_>>()
            .join(" × ")
    );
    println!(
        "#   budget = {},   N = {},   seed = {}",
        budget.map(num_str).unwrap_or_else(|| "∞".to_string()),
        n,
        seed
    );
    println!();

    // ── 1. Closed-form (only valid when there's no budget) ───────────────
    let cf = if budget.is_none() {
        let cf = solve_production_closed_form(c.clone(), p.clone(), ranges.clone());
        println!("## Method 1 — analytical closed form (newsvendor critical fractile)");
        println!("     x*       = [{}]", vec_fixed(&cf.x, 4));
        println!("     z*_true  = {:.4}", cf.objective);
        println!("     elapsed  = {} ms", num_str(cf.elapsed_ms));
        println!();
        Some(cf)
    } else {
        None
    };

    // ── 2. Monolithic SAA (single big LP) ────────────────────────────────
    println!("## Method 2 — monolithic SAA (one giant LP via solveLPInternal)");
    let slp_mono = build_production_slp(c.clone(), p.clone(), budget);
    let scenarios_mono =
        build_production_scenarios(UniformDemandSpec { ranges: ranges.clone(), seed }, n);
    let mono = solve_slp_monolithic(slp_mono, scenarios_mono);
    println!("     status    = {}", status_str(mono.status));
    println!("     x*        = [{}]", vec_fixed(&mono.x, 4));
    println!("     z*        = {:.4}", mono.objective);
    println!("     simplex iters = {},  elapsed = {} ms", mono.iterations, num_str(mono.elapsed_ms));
    if let Some(cf) = &cf {
        println!(
            "     vs closed-form Δ = {:.4}  (Monte Carlo error)",
            mono.objective - cf.objective
        );
    }
    println!();

    // ── 3. Benders L-shaped as a DES ─────────────────────────────────────
    println!("## Method 3 — Benders decomposition AS A DES (master = IncrementalLP, cuts arrive as movables)");
    let slp_bend = build_production_slp(c.clone(), p.clone(), budget);
    let scenarios_bend =
        build_production_scenarios(UniformDemandSpec { ranges: ranges.clone(), seed }, n);
    let bend = solve_slp_benders(
        slp_bend,
        scenarios_bend,
        BendersOpts {
            max_iter: Some(200),
            tol: Some(1e-7),
            verbose: Some(verbose),
            reference_path: None,
            reference_tol: None,
            silent_if_missing: None,
        },
    );
    let trace = bend.benders_trace.clone().unwrap_or_default();
    println!("     status    = {}", status_str(bend.status));
    println!("     x*        = [{}]", vec_fixed(&bend.x, 4));
    println!("     z*        = {:.4}", bend.objective);
    println!("     iters     = {}  (one tick per master+subproblem round)", bend.iterations);
    println!("     cuts      = {}", trace.iter().filter(|t| t.cut_added.is_some()).count());
    println!("     elapsed   = {} ms", num_str(bend.elapsed_ms));
    println!(
        "     vs monolithic: |Δz| = {:.2e},  speedup ≈ {:.1}×",
        (bend.objective - mono.objective).abs(),
        mono.elapsed_ms / bend.elapsed_ms.max(1.0)
    );
    if let Some(cf) = &cf {
        println!("     vs closed-form Δ = {:.4}", bend.objective - cf.objective);
    }
    println!();

    // ── Benders convergence table ────────────────────────────────────────
    println!("## Benders convergence trace (UB = master objective, LB = feasible value at this x*, gap = UB − LB)");
    println!(
        "     {}{}{}{}{}{}{}",
        pad("iter", 5),
        pad("x_master", 28),
        pad("θ_master", 12),
        pad("E[Q]", 12),
        pad("UB", 12),
        pad("LB", 12),
        pad("gap", 12)
    );
    for it in &trace {
        let x_master = format!("[{}]", vec_fixed(&it.x_master, 2));
        let stop = it
            .stop_reason
            .map(|r| format!("  {}", stop_reason_str(r)))
            .unwrap_or_default();
        println!(
            "     {}{}{}{}{}{}{}{}",
            pad(&it.iter.to_string(), 5),
            pad(&x_master, 28),
            pad(&format!("{:.3}", it.theta_master), 12),
            pad(&format!("{:.3}", it.expected_q), 12),
            pad(&format!("{:.3}", it.upper_bound), 12),
            pad(&format!("{:.3}", it.lower_bound), 12),
            pad(&format!("{:.2e}", it.gap), 12),
            stop
        );
        if let Some(cut) = &it.cut_added {
            if verbose {
                let n_coefs = cut.coefs.len();
                let terms: Vec<String> = cut
                    .coefs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let var = if i < n_coefs - 1 {
                            format!("x{}", i + 1)
                        } else {
                            "θ".to_string()
                        };
                        format!("{:.3}·{}", v, var)
                    })
                    .collect();
                println!("         cut  {} ≤ {:.3}", terms.join(" + "), cut.rhs);
            }
        }
    }
    println!();

    // ── 4. Out-of-sample evaluation ──────────────────────────────────────
    let oo_n = 50000usize;
    let oo_scenarios =
        build_production_scenarios(UniformDemandSpec { ranges: ranges.clone(), seed: 99999 }, oo_n);
    let eval_out_of_sample = |x: &[f64]| -> f64 {
        let mut z = 0.0;
        for i in 0..c.len() {
            z += -c[i] * x[i];
        }
        let mut q_sum = 0.0;
        for sc in &oo_scenarios {
            let d = sc.meta.as_ref().expect("scenario meta").d.clone();
            for i in 0..c.len() {
                q_sum += p[i] * x[i].min(d[i]);
            }
        }
        z + q_sum / oo_n as f64
    };
    println!("## Out-of-sample policy evaluation (N_oos = 50000 fresh scenarios)");
    println!("     monolithic x*: z_oos = {:.4}", eval_out_of_sample(&mono.x));
    println!("     Benders   x*: z_oos = {:.4}", eval_out_of_sample(&bend.x));
    if let Some(cf) = &cf {
        println!("     closed-form x*: z_oos = {:.4}  (≈ true z*)", eval_out_of_sample(&cf.x));
    }
    println!();
    println!("# Done.");
}

/// `Number.prototype` default string for an integer-valued f64 (no trailing
/// `.0`), used to mirror `${c.join(', ')}` and friends.
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

/// `[10, 12].join(', ')` for integer-valued cost/revenue vectors.
fn vec_fixed_int(xs: &[f64]) -> String {
    xs.iter().map(|v| num_str(*v)).collect::<Vec<_>>().join(", ")
}
