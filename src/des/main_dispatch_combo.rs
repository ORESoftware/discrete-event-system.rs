//! Port of `src/des/main-dispatch-combo.ts`.
//!
//! One dispatch problem (multi-class parallel-server) solved by DES + MDP + LP
//! + MCTS plus heuristics, all evaluated by the SAME DES with the SAME seeds.
//!
//! Delegates to `crate::des::general::{dispatch, lp}`. `process.env.*` →
//! `std::env::var`; the MCTS / MDP randomness is reproducible via the option
//! seeds inside `dispatch`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use crate::des::general::dispatch::{
    build_dispatch_fluid_lp, evaluate_policy, policy_fluid_lp, policy_mcts, policy_mdp_vi,
    policy_random, policy_round_robin, policy_sect, policy_shortest_queue, welch_t, DispatchPolicy,
    DispatchProblem, DispatchState, MctsPolicyOptions, MdpViPolicy, MdpViPolicyOptions,
};
use crate::des::general::lp::lp_to_string;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// A shared dispatch policy: the factory hands out clones of the same `Rc`, so
/// the (stateless argmax) MDP-VI policy is reused across replications — exactly
/// like the TS `() => mdpResult.policy` factory returning one object.
struct SharedPolicy(Rc<RefCell<dyn DispatchPolicy>>);
impl DispatchPolicy for SharedPolicy {
    fn pick(&mut self, state: &DispatchState, c: usize) -> usize {
        self.0.borrow_mut().pick(state, c)
    }
    fn reset(&mut self) {
        self.0.borrow_mut().reset();
    }
}

type Factory = Box<dyn Fn() -> Box<dyn DispatchPolicy>>;

/// Entry point (TS top-level `main`).
pub fn run() {
    let problem = DispatchProblem {
        m: 2,
        k: 2,
        arrival_rate: 1.6,
        class_prob: vec![0.6, 0.4],
        service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
    };
    let num_reps = env_usize("N_REPS", 30);
    let num_arrivals = env_usize("N_ARRIVALS", 3000);
    let warmup = (num_arrivals as f64 * 0.1).floor() as usize;
    let seed_base = env_u32("SEED_BASE", 1000);
    let skip_mdp = std::env::var("SKIP_MDP").as_deref() == Ok("1");

    println!("# Multi-class dispatch — DES + MDP + LP + MCTS combo");
    println!("# M={} machines, K={} classes", problem.m, problem.k);
    println!("# arrival rate λ = {}", problem.arrival_rate);
    println!(
        "# class probs    = [{}]",
        problem.class_prob.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
    );
    println!("# service rates μ_{{c,m}}:");
    for c in 0..problem.k {
        println!(
            "#   class {}: [{}]",
            c + 1,
            problem.service_rate[c].iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", ")
        );
    }
    let mut total_capacity = 0.0;
    for m in 0..problem.m {
        let mut cap = 0.0;
        for c in 0..problem.k {
            cap += problem.class_prob[c] * problem.service_rate[c][m];
        }
        total_capacity += cap;
    }
    println!("# total capacity Σ_m Σ_c p_c μ_{{c,m}} = {total_capacity:.3}");
    println!(
        "# theoretical ρ_avg if balanced = λ / capacity = {:.3}",
        problem.arrival_rate / total_capacity
    );
    println!("# (should be < 1 for stability)");
    println!();
    println!("# Replications per policy = {num_reps}");
    println!("# Arrivals per replication = {num_arrivals}  (first {warmup} discarded as warmup)");
    println!();

    println!("# Layer-3 LP fluid relaxation (solved via simplex / interior-point):");
    let lp = build_dispatch_fluid_lp(&problem);
    for line in lp_to_string(&lp).split('\n') {
        println!("#   {line}");
    }
    println!();

    let fluid = policy_fluid_lp(&problem, 12345);
    println!("# Fluid LP solved via {} in {} iterations", fluid.solver, fluid.iters);
    println!("#   bottleneck load t* = max_m ρ_m = {:.4}", fluid.bottleneck_load);
    for c in 0..problem.k {
        println!(
            "#   class {} → x* = [{}]",
            c + 1,
            fluid.x[c].iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(", ")
        );
    }
    println!();

    let mut mdp_shared: Option<Rc<RefCell<MdpViPolicy>>> = None;
    if !skip_mdp {
        print!("# Building MDP via DES rollouts and running value iteration ... ");
        let t0 = Instant::now();
        let res = policy_mdp_vi(
            &problem,
            MdpViPolicyOptions {
                q_max: Some(5),
                gamma: Some(0.95),
                rollouts_per_sa: Some(50),
                ..Default::default()
            },
        );
        println!(
            "done in {}ms (|S|={}, qMax={})",
            t0.elapsed().as_millis(),
            res.num_states,
            res.q_max
        );
        println!();
        mdp_shared = Some(Rc::new(RefCell::new(res.policy)));
    }

    // Policy registry: (name, note, factory).
    let mut entries: Vec<(&str, &str, Factory)> = Vec::new();
    entries.push(("random", "Layer 3: trivial baseline", Box::new(|| Box::new(policy_random(13)))));
    entries.push((
        "round-robin",
        "Layer 3: state-blind heuristic",
        Box::new(|| Box::new(policy_round_robin())),
    ));
    entries.push((
        "shortest-queue",
        "Layer 3: queue-aware heuristic",
        Box::new(|| Box::new(policy_shortest_queue())),
    ));
    {
        let p = problem.clone();
        entries.push((
            "SECT",
            "Layer 3: class-aware heuristic",
            Box::new(move || Box::new(policy_sect(&p))),
        ));
    }
    {
        let p = problem.clone();
        entries.push((
            "fluid-LP",
            "Layer 3: simplex / interior-point on the fluid relaxation",
            Box::new(move || Box::new(policy_fluid_lp(&p, 12345).policy)),
        ));
    }
    if let Some(shared) = &mdp_shared {
        let rc = shared.clone();
        entries.push((
            "MDP-VI",
            "Layer 3 ∘ Layer 2: value iteration on the empirical MDP whose transitions came from DES rollouts",
            Box::new(move || Box::new(SharedPolicy(rc.clone()))),
        ));
    }
    {
        let p = problem.clone();
        entries.push((
            "MCTS",
            "Layer 3 ∘ Layer 1: tree search using DES as the rollout oracle",
            Box::new(move || {
                Box::new(policy_mcts(
                    &p,
                    MctsPolicyOptions { iterations: Some(200), rollout_depth: Some(35), ..Default::default() },
                ))
            }),
        ));
    }

    // Keep (name, note) for the architectural recap before consuming factories.
    let meta: Vec<(&str, &str)> = entries.iter().map(|(n, note, _)| (*n, *note)).collect();

    // Evaluation (consumes each factory by value).
    let mut results = Vec::new();
    for (name, _note, factory) in entries.into_iter() {
        print!("# Evaluating '{name}' ({num_reps} reps × {num_arrivals} arrivals) ... ");
        let t0 = Instant::now();
        let r = evaluate_policy(&problem, factory, name, num_reps, num_arrivals, seed_base, warmup);
        println!("done in {}ms,  mean sojourn = {:.4}", t0.elapsed().as_millis(), r.mean_wait);
        results.push(r);
    }
    println!();

    let bar = "─".repeat(78);
    println!("# {bar}");
    println!("# Mean sojourn time per policy (lower is better):");
    println!("# {bar}");
    println!(
        "#   {:<18}{:>15}{:>11}{:>20}",
        "policy", "mean sojourn", "sd", "utilisation"
    );
    for r in &results {
        let util = format!(
            "[{}]",
            r.utilisation.iter().map(|u| format!("{:.1}%", u * 100.0)).collect::<Vec<_>>().join(", ")
        );
        println!(
            "#   {:<18}{:>15}{:>11}{:>20}",
            r.policy_name,
            format!("{:.4}", r.mean_wait),
            format!("{:.4}", r.sd_wait),
            util
        );
    }
    println!();

    println!("# {bar}");
    println!("# Welch t-statistic vs random (large positive ⇒ policy is significantly better):");
    println!("# {bar}");
    let random = results.iter().find(|r| r.policy_name == "random").expect("random present");
    for r in &results {
        if r.policy_name == "random" {
            continue;
        }
        let t = welch_t(random.raw_waits.clone(), r.raw_waits.clone());
        println!(
            "#   random vs {:<18} t = {:.2}    Δmean = {:.4}",
            r.policy_name,
            t,
            random.mean_wait - r.mean_wait
        );
    }
    println!();

    println!("# {bar}");
    println!("# Architectural recap:");
    println!("# {bar}");
    for (name, note) in &meta {
        println!("#   {:<18} → {}", name, note);
    }
    println!();
    println!("#   The same DES (`simulateDispatch`) evaluates EVERY policy.");
    println!("#   The MDP-VI policy uses the DES as its transition oracle.");
    println!("#   The MCTS policy uses the DES as its rollout simulator.");
    println!("#   The fluid-LP policy is a randomised assignment from the LP relaxation,");
    println!("#     which the simplex / interior-point of choice solves in milliseconds.");
}
