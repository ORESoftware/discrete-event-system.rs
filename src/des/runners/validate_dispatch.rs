//! Port of `src/des/runners/validate-dispatch.ts`.
//!
//! Validates the dispatch combo across problem instances and quantifies how each
//! architectural layer (greedy / fluid-LP / MDP-VI / MCTS) performs, via five
//! Welch-t-tested studies. Top-level `main()` → [`run`].
//!
//! PORT NOTES (stubbed cross-module deps):
//!   * `crate::des::general::dispatch::{DispatchProblem, evaluate_policy, welch_t,
//!     policy_random, policy_round_robin, policy_shortest_queue, policy_sect,
//!     policy_fluid_lp, policy_mdp_vi, policy_mcts, build_dispatch_fluid_lp}`.
//!   * `crate::des::general::lp::{solve_lp_internal, solve_lp_external}` and
//!     `crate::des::general::lp_des::solve_lp_via_des`.
//!   * The MDP/MCTS RNG would route through `crate::des::shared::capabilities`.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// Stubbed dispatch + LP layer.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct DispatchProblem {
    m: usize,
    k: usize,
    arrival_rate: f64,
    class_prob: Vec<f64>,
    service_rate: Vec<Vec<f64>>,
}

/// Opaque dispatch policy handle (PORT NOTE: `dispatch::Policy`).
#[derive(Clone, Debug, Default)]
struct Policy;

#[derive(Clone, Debug, Default)]
struct EvalResult {
    mean_wait: f64,
    raw_waits: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct FluidLpResult {
    policy: Policy,
}

#[derive(Clone, Debug, Default)]
struct DispatchViResult {
    v: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct MdpViOpts {
    q_max: usize,
    gamma: f64,
    rollouts_per_sa: usize,
    seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MctsOpts {
    iterations: usize,
    rollout_depth: usize,
}

fn policy_random(_seed: u64) -> Policy {
    Policy
}
fn policy_round_robin() -> Policy {
    Policy
}
fn policy_shortest_queue() -> Policy {
    Policy
}
fn policy_sect(_p: &DispatchProblem) -> Policy {
    Policy
}
fn policy_fluid_lp(_p: &DispatchProblem) -> FluidLpResult {
    FluidLpResult::default()
}
fn policy_mdp_vi(_p: &DispatchProblem, _o: MdpViOpts) -> DispatchViResult {
    DispatchViResult { v: vec![0.0; 4] }
}
fn policy_mcts(_p: &DispatchProblem, _o: MctsOpts) -> Policy {
    Policy
}
fn build_dispatch_fluid_lp(_p: &DispatchProblem) -> LpProblem {
    LpProblem::default()
}

#[allow(clippy::too_many_arguments)]
fn evaluate_policy<F: Fn() -> Policy>(
    _problem: &DispatchProblem,
    _factory: F,
    _name: &str,
    num_reps: usize,
    _num_arrivals: usize,
    _seed_base: u64,
    _warmup: usize,
) -> EvalResult {
    // PORT NOTE: real impl simulates `num_reps` replications and returns the
    // per-rep mean waits. Stub returns zeros.
    EvalResult {
        mean_wait: 0.0,
        raw_waits: vec![0.0; num_reps],
    }
}

/// Welch t-statistic for two samples (matches `dispatch::welch_t`).
fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let var = |xs: &[f64]| {
        let m = mean(xs);
        xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
    };
    let se_sq = var(a) / a.len() as f64 + var(b) / b.len() as f64;
    if se_sq <= 0.0 {
        return 0.0;
    }
    (mean(a) - mean(b)) / se_sq.sqrt()
}

#[derive(Clone, Debug, Default)]
struct LpProblem {
    c: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct LpResult {
    status: String,
    objective: f64,
}

fn stub_lp() -> LpResult {
    LpResult {
        status: "optimal".to_string(),
        objective: 0.0,
    }
}
fn solve_lp_internal(_lp: &LpProblem) -> LpResult {
    stub_lp()
}
fn solve_lp_external(_lp: &LpProblem, _method: &str) -> LpResult {
    stub_lp()
}
fn solve_lp_via_des(_lp: &LpProblem) -> LpResult {
    stub_lp()
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
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!(
            "{}  {}{}",
            if ok { "  PASS" } else { "  FAIL" },
            label,
            tail
        );
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
}

fn study1(c: &mut Checker) {
    println!("\nStudy 1 — well-specialised dispatch (M=2, K=2)");
    let problem = DispatchProblem {
        m: 2,
        k: 2,
        arrival_rate: 1.6,
        class_prob: vec![0.6, 0.4],
        service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
    };
    let num_reps = 20;
    let num_arrivals = 2500;
    let warmup = 250;
    let seed_base = 5000;
    let random = evaluate_policy(
        &problem,
        || policy_random(11),
        "random",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let rr = evaluate_policy(
        &problem,
        policy_round_robin,
        "round-robin",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let sq = evaluate_policy(
        &problem,
        policy_shortest_queue,
        "shortest-queue",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let sect = evaluate_policy(
        &problem,
        || policy_sect(&problem),
        "SECT",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let fluid = evaluate_policy(
        &problem,
        || policy_fluid_lp(&problem).policy,
        "fluid-LP",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    println!("    random         mean = {:.3}", random.mean_wait);
    println!("    round-robin    mean = {:.3}", rr.mean_wait);
    println!("    shortest-queue mean = {:.3}", sq.mean_wait);
    println!("    SECT           mean = {:.3}", sect.mean_wait);
    println!("    fluid-LP       mean = {:.3}", fluid.mean_wait);
    c.check(
        "SECT < random  (Welch-t > 6)",
        welch_t(&random.raw_waits, &sect.raw_waits) > 6.0,
        &format!("t = {:.2}", welch_t(&random.raw_waits, &sect.raw_waits)),
    );
    c.check(
        "SECT < shortest-queue  (Welch-t > 5)",
        welch_t(&sq.raw_waits, &sect.raw_waits) > 5.0,
        &format!("t = {:.2}", welch_t(&sq.raw_waits, &sect.raw_waits)),
    );
    c.check(
        "shortest-queue < random  (Welch-t > 4)",
        welch_t(&random.raw_waits, &sq.raw_waits) > 4.0,
        &format!("t = {:.2}", welch_t(&random.raw_waits, &sq.raw_waits)),
    );
    c.check(
        "SECT and fluid-LP within 25% of each other",
        (sect.mean_wait - fluid.mean_wait).abs() / sect.mean_wait < 0.25,
        &format!("Δ = {:.3}", fluid.mean_wait - sect.mean_wait),
    );
}

fn study2(c: &mut Checker) {
    println!("\nStudy 2 — heavily-loaded weak-specialisation (M=3, K=3, ρ̄ ≈ 0.85)");
    let problem = DispatchProblem {
        m: 3,
        k: 3,
        arrival_rate: 2.55,
        class_prob: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        service_rate: vec![
            vec![1.6, 0.9, 0.7],
            vec![0.7, 1.6, 0.9],
            vec![0.9, 0.7, 1.6],
        ],
    };
    let num_reps = 20;
    let num_arrivals = 3000;
    let warmup = 300;
    let seed_base = 9000;
    let random = evaluate_policy(
        &problem,
        || policy_random(31),
        "random",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let rr = evaluate_policy(
        &problem,
        policy_round_robin,
        "round-robin",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let sq = evaluate_policy(
        &problem,
        policy_shortest_queue,
        "shortest-queue",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let sect = evaluate_policy(
        &problem,
        || policy_sect(&problem),
        "SECT",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let fluid = evaluate_policy(
        &problem,
        || policy_fluid_lp(&problem).policy,
        "fluid-LP",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    println!("    random         mean = {:.3}", random.mean_wait);
    println!("    round-robin    mean = {:.3}", rr.mean_wait);
    println!("    shortest-queue mean = {:.3}", sq.mean_wait);
    println!("    SECT           mean = {:.3}", sect.mean_wait);
    println!("    fluid-LP       mean = {:.3}", fluid.mean_wait);
    c.check(
        "fluid-LP < random  (Welch-t > 4)",
        welch_t(&random.raw_waits, &fluid.raw_waits) > 4.0,
        &format!("t = {:.2}", welch_t(&random.raw_waits, &fluid.raw_waits)),
    );
    c.check(
        "shortest-queue < random  (Welch-t > 3)",
        welch_t(&random.raw_waits, &sq.raw_waits) > 3.0,
        &format!("t = {:.2}", welch_t(&random.raw_waits, &sq.raw_waits)),
    );
    c.check(
        "SECT < random  (Welch-t > 3)",
        welch_t(&random.raw_waits, &sect.raw_waits) > 3.0,
        &format!("t = {:.2}", welch_t(&random.raw_waits, &sect.raw_waits)),
    );
}

fn study3(c: &mut Checker) {
    println!("\nStudy 3 — fluid LP solved by 4 different solvers must agree");
    let problem = DispatchProblem {
        m: 3,
        k: 3,
        arrival_rate: 2.55,
        class_prob: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        service_rate: vec![
            vec![1.6, 0.9, 0.7],
            vec![0.7, 1.6, 0.9],
            vec![0.9, 0.7, 1.6],
        ],
    };
    let lp = build_dispatch_fluid_lp(&problem);
    let s_internal = solve_lp_internal(&lp);
    let s_des = solve_lp_via_des(&lp);
    let s_scipy_ds = solve_lp_external(&lp, "highs-ds");
    let s_scipy_ipm = solve_lp_external(&lp, "highs-ipm");
    println!(
        "    internal simplex     status={} obj={:.8}",
        s_internal.status, s_internal.objective
    );
    println!(
        "    DES-engine simplex   status={}      obj={:.8}",
        s_des.status, s_des.objective
    );
    println!(
        "    scipy:highs-ds       status={}  obj={:.8}",
        s_scipy_ds.status, s_scipy_ds.objective
    );
    println!(
        "    scipy:highs-ipm      status={} obj={:.8}",
        s_scipy_ipm.status, s_scipy_ipm.objective
    );
    let objs: Vec<f64> = [&s_internal, &s_des, &s_scipy_ds, &s_scipy_ipm]
        .iter()
        .filter(|s| s.status == "optimal")
        .map(|s| s.objective)
        .collect();
    if objs.len() < 2 {
        c.check(
            "LP solvers available",
            false,
            "fewer than 2 solvers returned optimal",
        );
        return;
    }
    let ref_obj = objs[0];
    let max_diff = objs
        .iter()
        .map(|o| (o - ref_obj).abs())
        .fold(0.0_f64, f64::max);
    c.check(
        "all available solvers agree on the LP objective to 1e-6",
        max_diff < 1e-6,
        &format!("max |Δobj| = {:.2e}", max_diff),
    );
}

fn study4(c: &mut Checker) {
    println!("\nStudy 4 — MDP-VI: V*(empty system) stable as qMax grows");
    let problem = DispatchProblem {
        m: 2,
        k: 2,
        arrival_rate: 1.6,
        class_prob: vec![0.6, 0.4],
        service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
    };
    let r4 = policy_mdp_vi(
        &problem,
        MdpViOpts {
            q_max: 4,
            gamma: 0.95,
            rollouts_per_sa: 200,
            seed: 42,
        },
    );
    let r6 = policy_mdp_vi(
        &problem,
        MdpViOpts {
            q_max: 6,
            gamma: 0.95,
            rollouts_per_sa: 200,
            seed: 42,
        },
    );
    let empty = |vi: &DispatchViResult| [vi.v[0], vi.v[1]];
    let v4 = empty(&r4);
    let v6 = empty(&r6);
    println!(
        "    V_qMax=4(0,0,c)  = [{}]",
        v4.iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "    V_qMax=6(0,0,c)  = [{}]",
        v6.iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let max_diff = v4
        .iter()
        .zip(v6.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    c.check(
        "|V_qMax=4 − V_qMax=6| at empty system < 0.5 (sampling noise)",
        max_diff < 0.5,
        &format!("max diff = {:.4}", max_diff),
    );
}

fn study5(c: &mut Checker) {
    println!("\nStudy 5 — MCTS converges toward its rollout policy (SECT) as iters grow");
    let problem = DispatchProblem {
        m: 2,
        k: 2,
        arrival_rate: 1.6,
        class_prob: vec![0.6, 0.4],
        service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
    };
    let num_reps = 8;
    let num_arrivals = 1200;
    let warmup = 120;
    let seed_base = 3300;
    let sect = evaluate_policy(
        &problem,
        || policy_sect(&problem),
        "sect",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let mcts_low = evaluate_policy(
        &problem,
        || {
            policy_mcts(
                &problem,
                MctsOpts {
                    iterations: 20,
                    rollout_depth: 20,
                },
            )
        },
        "mcts-20",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let mcts_mid = evaluate_policy(
        &problem,
        || {
            policy_mcts(
                &problem,
                MctsOpts {
                    iterations: 100,
                    rollout_depth: 25,
                },
            )
        },
        "mcts-100",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    let mcts_high = evaluate_policy(
        &problem,
        || {
            policy_mcts(
                &problem,
                MctsOpts {
                    iterations: 300,
                    rollout_depth: 35,
                },
            )
        },
        "mcts-300",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    println!("    SECT          mean = {:.3}", sect.mean_wait);
    println!("    MCTS  20 iter mean = {:.3}", mcts_low.mean_wait);
    println!("    MCTS 100 iter mean = {:.3}", mcts_mid.mean_wait);
    println!("    MCTS 300 iter mean = {:.3}", mcts_high.mean_wait);
    let random = evaluate_policy(
        &problem,
        || policy_random(11),
        "random",
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    println!(
        "    random        mean = {:.3} (sanity check)",
        random.mean_wait
    );
    c.check(
        "MCTS-300 < random (Welch-t > 3)",
        welch_t(&random.raw_waits, &mcts_high.raw_waits) > 3.0,
        &format!(
            "t = {:.2}",
            welch_t(&random.raw_waits, &mcts_high.raw_waits)
        ),
    );
    c.check(
        "MCTS-300 within 2.5× of SECT (bounded by rollout policy)",
        mcts_high.mean_wait < 2.5 * sect.mean_wait,
        &format!(
            "MCTS = {:.3}, 2.5×SECT = {:.3}",
            mcts_high.mean_wait,
            2.5 * sect.mean_wait
        ),
    );
}

/// `validate-dispatch.ts` `main()`.
pub fn run() {
    let mut c = Checker::new();
    println!("# DES + MDP + LP + MCTS dispatch validation");
    println!("# (each study uses Welch t-tests on independent replications,");
    println!("#  so individual reps may noise-up but the conclusions hold)");
    study1(&mut c);
    study2(&mut c);
    study3(&mut c);
    study4(&mut c);
    study5(&mut c);
    println!(
        "\n{} checks: {} passed, {} failed",
        c.pass + c.fail,
        c.pass,
        c.fail
    );
    if c.fail > 0 {
        std::process::exit(1);
    }
}
