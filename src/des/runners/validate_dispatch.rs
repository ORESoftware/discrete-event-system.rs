//! Port of `src/des/runners/validate-dispatch.ts`.
//!
//! Validates the dispatch combo across problem instances and quantifies how each
//! architectural layer (greedy / fluid-LP / MDP-VI / MCTS) performs, via five
//! Welch-t-tested studies. Top-level `main()` → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust dispatch simulator, policies, fluid LP builder, LP
//!     solvers, MDP-VI, and MCTS modules.

#![allow(dead_code)]

use crate::des::general::dispatch::{
    build_dispatch_fluid_lp as build_dispatch_fluid_lp_model,
    policy_fluid_lp as policy_fluid_lp_model, policy_mcts as policy_mcts_model,
    policy_mdp_vi as policy_mdp_vi_model, policy_random as policy_random_model,
    policy_round_robin as policy_round_robin_model, policy_sect as policy_sect_model,
    policy_shortest_queue as policy_shortest_queue_model, simulate_dispatch,
    welch_t as welch_t_model, DispatchPolicy, DispatchProblem, EvaluationResult,
    FluidLpPolicyResult as RealFluidLpPolicyResult, MctsPolicyOptions, MdpViPolicyOptions,
    MdpViPolicyResult,
};
use crate::des::general::lp::{
    solve_lp_internal as solve_lp_internal_model, ExternalSolver, ExternalSolverOptions,
    InternalSimplexOptions, LPProblem,
};
use crate::des::general::lp_des::{solve_lp_via_des as solve_lp_via_des_model, DESSimplexOptions};
use crate::des::shared::transform::Transform;

type Policy = Box<dyn DispatchPolicy>;
type EvalResult = EvaluationResult;
type DispatchViResult = MdpViPolicyResult;
type LpProblem = LPProblem;

struct FluidLpResult {
    policy: Policy,
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

#[derive(Clone, Copy, Debug)]
struct DispatchValidationProfile {
    full: bool,
    mcts_reps: usize,
    mcts_arrivals: usize,
    mcts_warmup: usize,
    mcts_low_iter: usize,
    mcts_low_depth: usize,
    mcts_mid_iter: usize,
    mcts_mid_depth: usize,
    mcts_high_iter: usize,
    mcts_high_depth: usize,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "full" | "FULL"
            )
        })
        .unwrap_or(false)
}

fn dispatch_validation_profile() -> DispatchValidationProfile {
    if env_flag("ORES_DISPATCH_FULL") {
        DispatchValidationProfile {
            full: true,
            mcts_reps: 8,
            mcts_arrivals: 1200,
            mcts_warmup: 120,
            mcts_low_iter: 20,
            mcts_low_depth: 20,
            mcts_mid_iter: 100,
            mcts_mid_depth: 25,
            mcts_high_iter: 300,
            mcts_high_depth: 35,
        }
    } else {
        DispatchValidationProfile {
            full: false,
            mcts_reps: 4,
            mcts_arrivals: 300,
            mcts_warmup: 40,
            mcts_low_iter: 8,
            mcts_low_depth: 8,
            mcts_mid_iter: 24,
            mcts_mid_depth: 12,
            mcts_high_iter: 60,
            mcts_high_depth: 16,
        }
    }
}

fn policy_random(_seed: u64) -> Policy {
    Box::new(policy_random_model(_seed as u32))
}
fn policy_round_robin() -> Policy {
    Box::new(policy_round_robin_model())
}
fn policy_shortest_queue() -> Policy {
    Box::new(policy_shortest_queue_model())
}
fn policy_sect(p: &DispatchProblem) -> Policy {
    Box::new(policy_sect_model(p))
}
fn policy_fluid_lp(p: &DispatchProblem) -> FluidLpResult {
    let RealFluidLpPolicyResult { policy, .. } = policy_fluid_lp_model(p, 12345);
    FluidLpResult {
        policy: Box::new(policy),
    }
}
fn policy_mdp_vi(p: &DispatchProblem, o: MdpViOpts) -> DispatchViResult {
    policy_mdp_vi_model(
        p,
        MdpViPolicyOptions {
            q_max: Some(o.q_max),
            gamma: Some(o.gamma),
            rollouts_per_sa: Some(o.rollouts_per_sa),
            seed: Some(o.seed as u32),
            ..Default::default()
        },
    )
}
fn policy_mcts(p: &DispatchProblem, o: MctsOpts) -> Policy {
    Box::new(policy_mcts_model(
        p,
        MctsPolicyOptions {
            iterations: Some(o.iterations),
            rollout_depth: Some(o.rollout_depth),
            ..Default::default()
        },
    ))
}
fn build_dispatch_fluid_lp(p: &DispatchProblem) -> LpProblem {
    build_dispatch_fluid_lp_model(p)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_policy<F: Fn() -> Policy>(
    problem: &DispatchProblem,
    factory: F,
    name: &str,
    num_reps: usize,
    num_arrivals: usize,
    seed_base: u64,
    warmup: usize,
) -> EvalResult {
    let mut waits = Vec::with_capacity(num_reps);
    let mut utils: Vec<Vec<f64>> = Vec::with_capacity(num_reps);
    for r in 0..num_reps {
        let mut policy = factory();
        let result = simulate_dispatch(
            problem,
            policy.as_mut(),
            num_arrivals,
            seed_base as u32 + r as u32,
            warmup,
        );
        waits.push(result.mean_sojourn);
        utils.push(result.per_machine_utilisation);
    }
    let mean_wait = waits.iter().sum::<f64>() / waits.len() as f64;
    let denom = ((waits.len() as i64 - 1).max(1)) as f64;
    let sd_wait = (waits.iter().map(|w| (w - mean_wait).powi(2)).sum::<f64>() / denom).sqrt();
    let mut utilisation = vec![0.0; problem.m];
    for u in &utils {
        for mm in 0..problem.m {
            utilisation[mm] += u[mm] / utils.len() as f64;
        }
    }
    EvalResult {
        policy_name: name.to_string(),
        mean_wait,
        sd_wait,
        raw_waits: waits,
        utilisation,
    }
}

fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    welch_t_model(a.to_vec(), b.to_vec())
}

#[derive(Clone, Debug, Default)]
struct LpResult {
    status: String,
    objective: f64,
}

fn lp_result(status: crate::des::general::lp::LPStatus, objective: f64) -> LpResult {
    LpResult {
        status: status.as_str().to_string(),
        objective,
    }
}
fn solve_lp_internal(lp: &LpProblem) -> LpResult {
    let sol = solve_lp_internal_model(lp, &InternalSimplexOptions::default());
    lp_result(sol.status, sol.objective)
}
fn solve_lp_external(lp: &LpProblem, method: &str) -> LpResult {
    let sol = ExternalSolver::new(ExternalSolverOptions {
        method: Some(method.to_string()),
        ..Default::default()
    })
    .transform(lp.clone());
    lp_result(sol.status, sol.objective)
}
fn solve_lp_via_des(lp: &LpProblem) -> LpResult {
    let sol = solve_lp_via_des_model(lp, &DESSimplexOptions::default());
    lp_result(sol.status, sol.objective)
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
    let s_external_ds = solve_lp_external(&lp, "highs-ds");
    let s_external_ipm = solve_lp_external(&lp, "highs-ipm");
    println!(
        "    internal simplex     status={} obj={:.8}",
        s_internal.status, s_internal.objective
    );
    println!(
        "    DES-engine simplex   status={}      obj={:.8}",
        s_des.status, s_des.objective
    );
    println!(
        "    external:highs-ds    status={}  obj={:.8}",
        s_external_ds.status, s_external_ds.objective
    );
    println!(
        "    external:highs-ipm   status={} obj={:.8}",
        s_external_ipm.status, s_external_ipm.objective
    );
    let objs: Vec<f64> = [&s_internal, &s_des, &s_external_ds, &s_external_ipm]
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

fn study5(c: &mut Checker, profile: &DispatchValidationProfile) {
    println!("\nStudy 5 — MCTS converges toward its rollout policy (SECT) as iters grow");
    let problem = DispatchProblem {
        m: 2,
        k: 2,
        arrival_rate: 1.6,
        class_prob: vec![0.6, 0.4],
        service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
    };
    let num_reps = profile.mcts_reps;
    let num_arrivals = profile.mcts_arrivals;
    let warmup = profile.mcts_warmup;
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
                    iterations: profile.mcts_low_iter,
                    rollout_depth: profile.mcts_low_depth,
                },
            )
        },
        &format!("mcts-{}", profile.mcts_low_iter),
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
                    iterations: profile.mcts_mid_iter,
                    rollout_depth: profile.mcts_mid_depth,
                },
            )
        },
        &format!("mcts-{}", profile.mcts_mid_iter),
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
                    iterations: profile.mcts_high_iter,
                    rollout_depth: profile.mcts_high_depth,
                },
            )
        },
        &format!("mcts-{}", profile.mcts_high_iter),
        num_reps,
        num_arrivals,
        seed_base,
        warmup,
    );
    println!("    SECT          mean = {:.3}", sect.mean_wait);
    println!(
        "    MCTS {:>3} iter mean = {:.3}",
        profile.mcts_low_iter, mcts_low.mean_wait
    );
    println!(
        "    MCTS {:>3} iter mean = {:.3}",
        profile.mcts_mid_iter, mcts_mid.mean_wait
    );
    println!(
        "    MCTS {:>3} iter mean = {:.3}",
        profile.mcts_high_iter, mcts_high.mean_wait
    );
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
    let random_vs_high_t = welch_t(&random.raw_waits, &mcts_high.raw_waits);
    let beats_random = if profile.full {
        random_vs_high_t > 3.0
    } else {
        mcts_high.mean_wait < random.mean_wait
    };
    let random_label = if profile.full {
        format!("MCTS-{} < random (Welch-t > 3)", profile.mcts_high_iter)
    } else {
        format!(
            "MCTS-{} < random mean (fast profile)",
            profile.mcts_high_iter
        )
    };
    c.check(
        &random_label,
        beats_random,
        &format!(
            "t = {:.2}, random = {:.3}, MCTS = {:.3}",
            random_vs_high_t, random.mean_wait, mcts_high.mean_wait
        ),
    );
    let sect_ratio_limit = if profile.full { 2.5 } else { 3.0 };
    let sect_label = format!(
        "MCTS-{} within {:.1}× of SECT (bounded by rollout policy)",
        profile.mcts_high_iter, sect_ratio_limit
    );
    c.check(
        &sect_label,
        mcts_high.mean_wait < sect_ratio_limit * sect.mean_wait,
        &format!(
            "MCTS = {:.3}, {:.1}×SECT = {:.3}",
            mcts_high.mean_wait,
            sect_ratio_limit,
            sect_ratio_limit * sect.mean_wait
        ),
    );
}

/// `validate-dispatch.ts` `main()`.
pub fn run() {
    let mut c = Checker::new();
    let profile = dispatch_validation_profile();
    println!("# DES + MDP + LP + MCTS dispatch validation");
    println!("# (each study uses Welch t-tests on independent replications,");
    println!("#  so individual reps may noise-up but the conclusions hold)");
    if profile.full {
        println!("# profile: full");
    } else {
        println!("# profile: fast (set ORES_DISPATCH_FULL=1 for full MCTS study)");
    }
    study1(&mut c);
    study2(&mut c);
    study3(&mut c);
    study4(&mut c);
    study5(&mut c, &profile);
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
