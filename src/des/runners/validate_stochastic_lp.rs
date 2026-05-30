//! Port of `src/des/runners/validate-stochastic-lp.ts`.
//!
//! Three-way audit of the stochastic LP solver: monolithic SAA vs Benders-as-DES
//! equivalence (Part A), 1/√N convergence to the closed form (Part B),
//! Benders-over-monolithic speedup (Part C), and a budget-constrained scenario
//! (Part D). Top-level driver → [`run`].
//!
//! PORT NOTES (stubbed cross-module deps):
//!   * `crate::des::general::stochastic_lp::{build_production_slp,
//!     build_production_scenarios, mulberry32, solve_slp_monolithic,
//!     solve_slp_benders, solve_production_closed_form}`.
//!   * Scenario sampling RNG would route through `mulberry32` / `SeededRandom`;
//!     `Date.now()` timing → `std::time::Instant`.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::time::Instant;

// =============================================================================
// Stubbed stochastic-LP layer.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct Slp {
    c: Vec<f64>,
    p: Vec<f64>,
    budget: Option<f64>,
}

/// One sampled scenario; `meta.D` is the realised demand vector.
#[derive(Clone, Debug, Default)]
struct Scenario {
    d: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct SlpResult {
    status: String,
    x: Vec<f64>,
    objective: f64,
    iterations: usize,
}

#[derive(Clone, Debug, Default)]
struct ClosedForm {
    x: Vec<f64>,
    objective: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ScenarioOpts {
    seed: u64,
}

/// PORT NOTE: `stochastic_lp::mulberry32` (re-exported for parity).
fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x6D2B_79F5);
        let mut t = (s ^ (s >> 15)).wrapping_mul(1 | s);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

fn build_production_slp(c: &[f64], p: &[f64], budget: Option<f64>) -> Slp {
    Slp {
        c: c.to_vec(),
        p: p.to_vec(),
        budget,
    }
}

fn build_production_scenarios(ranges: &[(f64, f64)], _opts: ScenarioOpts, n: usize) -> Vec<Scenario> {
    // PORT NOTE: real impl draws U(a,b) demands via a seeded RNG. The stub
    // returns `n` deterministic mid-range scenarios so downstream indexing is sound.
    let d: Vec<f64> = ranges.iter().map(|(a, b)| 0.5 * (a + b)).collect();
    vec![Scenario { d: d.clone() }; n]
}

fn solve_slp_monolithic(slp: &Slp, _scenarios: &[Scenario]) -> SlpResult {
    SlpResult {
        status: "optimal".to_string(),
        x: vec![0.0; slp.c.len()],
        objective: 0.0,
        iterations: 0,
    }
}

fn solve_slp_benders(slp: &Slp, _scenarios: &[Scenario], _tol: f64) -> SlpResult {
    SlpResult {
        status: "optimal".to_string(),
        x: vec![0.0; slp.c.len()],
        objective: 0.0,
        iterations: 0,
    }
}

fn solve_production_closed_form(c: &[f64], _p: &[f64], _ranges: &[(f64, f64)]) -> ClosedForm {
    ClosedForm {
        x: vec![0.0; c.len()],
        objective: 0.0,
    }
}

// =============================================================================
// Driver helpers.
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
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!("{}  {}{}", if ok { "  PASS" } else { "  FAIL" }, label, tail);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
    fn close(&mut self, label: &str, a: f64, b: f64, tol: f64) {
        self.check(label, (a - b).abs() <= tol, &format!("|{:.6} − {:.6}| = {:.2e}", a, b, (a - b).abs()));
    }
    fn arr_close(&mut self, label: &str, a: &[f64], b: &[f64], tol: f64) {
        if a.len() != b.len() {
            self.check(label, false, &format!("lengths {} vs {}", a.len(), b.len()));
            return;
        }
        let mut max_d = 0.0_f64;
        for i in 0..a.len() {
            max_d = f64::max(max_d, (a[i] - b[i]).abs());
        }
        self.check(label, max_d <= tol, &format!("max|Δ|={:.2e}", max_d));
    }
}

fn eval_saa_objective(x: &[f64], scenarios: &[Scenario], c: &[f64], p: &[f64]) -> f64 {
    let mut z = 0.0;
    for i in 0..c.len() {
        z += -c[i] * x[i];
    }
    let mut q = 0.0;
    for sc in scenarios {
        for i in 0..c.len() {
            q += p[i] * f64::min(x[i], sc.d[i]);
        }
    }
    z += q / scenarios.len() as f64;
    z
}

/// `validate-stochastic-lp.ts` top-level driver.
pub fn run() {
    let mut c_chk = Checker::new();
    let c = [10.0, 12.0];
    let p = [25.0, 28.0];
    let ranges: [(f64, f64); 2] = [(50.0, 100.0), (40.0, 80.0)];

    // Part A — Monolithic SAA ≡ Benders-as-DES on the same scenario set.
    println!("\nPart A — Monolithic SAA ≡ Benders-as-DES on the same scenario set");
    {
        for n in [10usize, 50, 200] {
            for seed in [1u64, 2, 3] {
                let slp = build_production_slp(&c, &p, None);
                let scenarios = build_production_scenarios(&ranges, ScenarioOpts { seed }, n);
                let mono = solve_slp_monolithic(&slp, &scenarios);
                let bend = solve_slp_benders(&slp, &scenarios, 1e-9);
                c_chk.check(
                    &format!("A.{}.{} both optimal", n, seed),
                    mono.status == "optimal" && bend.status == "optimal",
                    "",
                );
                c_chk.close(&format!("A.{}.{} z (mono ≡ Benders)", n, seed), mono.objective, bend.objective, 1e-6);
                let z_mono_eval = eval_saa_objective(&mono.x, &scenarios, &c, &p);
                let z_bend_eval = eval_saa_objective(&bend.x, &scenarios, &c, &p);
                c_chk.close(&format!("A.{}.{} mono.x evaluates to mono.z", n, seed), z_mono_eval, mono.objective, 1e-6);
                c_chk.close(&format!("A.{}.{} Benders.x evaluates to Benders.z", n, seed), z_bend_eval, bend.objective, 1e-6);
                c_chk.close(&format!("A.{}.{} both x's are equally optimal under same scenarios", n, seed), z_mono_eval, z_bend_eval, 1e-6);
            }
        }
    }

    // Part B — Statistical convergence of SAA to closed-form true optimum.
    println!("\nPart B — Statistical convergence of SAA to closed-form true optimum");
    {
        let slp_unc = build_production_slp(&c, &p, None);
        let cf = solve_production_closed_form(&c, &p, &ranges);
        let z_true = cf.objective;
        println!(
            "  closed-form z* = {:.4}   x* = [{}]",
            z_true,
            cf.x.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", ")
        );

        let ns = [10usize, 100, 1000, 10000];
        let r = 20usize;
        struct Stat {
            n: usize,
            mean_z: f64,
            stderr_z: f64,
            bias_z: f64,
            mean_gap: f64,
            stderr_gap: f64,
        }
        let mut stats: Vec<Stat> = Vec::new();
        for &n in &ns {
            let mut zs: Vec<f64> = Vec::new();
            for seed in 1..=r {
                let sc = build_production_scenarios(&ranges, ScenarioOpts { seed: (seed * 1000 + n) as u64 }, n);
                let sol = solve_slp_benders(&slp_unc, &sc, 1e-7);
                zs.push(sol.objective);
            }
            let mean_z = zs.iter().sum::<f64>() / r as f64;
            let var_z = zs.iter().map(|z| (z - mean_z).powi(2)).sum::<f64>() / (r as f64 - 1.0);
            let stderr_z = (var_z / r as f64).sqrt();
            let bias_z = mean_z - z_true;

            let mut gap_sum = 0.0;
            let mut gap_vals: Vec<f64> = Vec::new();
            for seed in 1..=r {
                let sc = build_production_scenarios(&ranges, ScenarioOpts { seed: (seed * 1000 + n) as u64 }, n);
                let sol = solve_slp_benders(&slp_unc, &sc, 1e-7);
                let oo_seed = 999000 + seed * 7;
                let oo_scenarios = build_production_scenarios(&ranges, ScenarioOpts { seed: oo_seed as u64 }, 5000);
                let mut z_eval = 0.0;
                for i in 0..c.len() {
                    z_eval += -c[i] * sol.x[i];
                }
                let mut q_sum = 0.0;
                for oo_sc in &oo_scenarios {
                    let mut q = 0.0;
                    for i in 0..c.len() {
                        q += p[i] * f64::min(sol.x[i], oo_sc.d[i]);
                    }
                    q_sum += q;
                }
                z_eval += q_sum / oo_scenarios.len() as f64;
                let g = z_true - z_eval;
                gap_vals.push(g);
                gap_sum += g;
            }
            let mean_gap = gap_sum / r as f64;
            let var_gap = gap_vals.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / (r as f64 - 1.0);
            let stderr_gap = (var_gap / r as f64).sqrt();
            stats.push(Stat { n, mean_z, stderr_z, bias_z, mean_gap, stderr_gap });
            println!(
                "  N={:>5}   mean SAA z* = {:.3} ± {:.3}   bias = {:.3}   out-of-sample gap = {:.3} ± {:.3}",
                n, mean_z, stderr_z, bias_z, mean_gap, stderr_gap
            );
        }
        let ratio_100_10000 = stats[1].stderr_z / stats[3].stderr_z;
        c_chk.check(
            "stderr decays with √N (factor 100→10000 ≈ 10)",
            ratio_100_10000 > 5.0 && ratio_100_10000 < 20.0,
            &format!("ratio={:.2}", ratio_100_10000),
        );
        c_chk.check(
            "SAA z* approaches true z* at N = 10000",
            stats[3].bias_z.abs() <= 0.02 * z_true.abs(),
            &format!("bias={:.3} vs 2% of zTrue={:.3}", stats[3].bias_z, 0.02 * z_true.abs()),
        );
        let gap_shrinks = stats[3].mean_gap < stats[0].mean_gap;
        c_chk.check(
            "out-of-sample optimality gap shrinks with N",
            gap_shrinks,
            &format!("{:.3} → {:.3}", stats[0].mean_gap, stats[3].mean_gap),
        );
    }

    // Part C — Benders is much faster than monolithic for large N.
    println!("\nPart C — Benders is much faster than monolithic for large N");
    {
        let slp_unc = build_production_slp(&c, &p, None);
        let ns = [50usize, 200, 500];
        for &n in &ns {
            let sc = build_production_scenarios(&ranges, ScenarioOpts { seed: 99 }, n);
            let t_mono = Instant::now();
            let mono = solve_slp_monolithic(&slp_unc, &sc);
            let mono_ms = t_mono.elapsed().as_millis();
            let t_bend = Instant::now();
            let bend = solve_slp_benders(&slp_unc, &sc, 1e-7);
            let bend_ms = t_bend.elapsed().as_millis();
            let speedup = mono_ms as f64 / (1u128.max(bend_ms)) as f64;
            println!(
                "  N={:>5}   mono = {:>5} ms ({:>4} iters)   Benders = {:>4} ms ({:>2} iters)   speedup ≈ {:.1}×",
                n, mono_ms, mono.iterations, bend_ms, bend.iterations, speedup
            );
            c_chk.check(
                &format!("C.{} mono and Benders agree", n),
                (mono.objective - bend.objective).abs() <= 1e-5,
                &format!("Δz = {:.2e}", (mono.objective - bend.objective).abs()),
            );
        }
    }

    // Part D — Budget-constrained scenario (no closed form).
    println!("\nPart D — Budget-constrained scenario (no closed form)");
    {
        for budget in [80.0_f64, 120.0, 200.0] {
            let slp = build_production_slp(&c, &p, Some(budget));
            let sc = build_production_scenarios(&ranges, ScenarioOpts { seed: 7 }, 500);
            let mono = solve_slp_monolithic(&slp, &sc);
            let bend = solve_slp_benders(&slp, &sc, 1e-9);
            c_chk.check(
                &format!("D.{} mono ≡ Benders z", budget),
                (mono.objective - bend.objective).abs() <= 1e-5,
                &format!("mono z = {:.4}, Benders z = {:.4}", mono.objective, bend.objective),
            );
            c_chk.arr_close(&format!("D.{} mono ≡ Benders x", budget), &mono.x, &bend.x, 1e-4);
            let total_x = mono.x[0] + mono.x[1];
            c_chk.check(
                &format!("D.{} budget feasibility (Σx ≤ {})", budget, budget),
                total_x <= budget + 1e-7,
                &format!("Σx = {:.4}", total_x),
            );
        }
    }

    println!("\n{} checks: {} passed, {} failed", c_chk.pass + c_chk.fail, c_chk.pass, c_chk.fail);
    if c_chk.fail > 0 {
        std::process::exit(1);
    }
}
