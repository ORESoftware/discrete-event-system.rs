//! Port of `src/des/main-inventory-mdp.ts`.
//!
//! Multi-period inventory MDP (leftover stock carries over): discover the
//! optimal ordering policy via value iteration, classify its structure
//! (base-stock vs (s, S)), then simulate under the discovered policy.
//!
//! ## Rust shape
//!   * `MDPSpec`/`Outcome`/`value_iteration` are reused from
//!     `crate::des::general::value_iteration`; `mulberry32`/`with_seed` from
//!     `crate::des::general::prng`.
//!   * Newsvendor demand helpers are shared with
//!     `crate::des::main_newsvendor` so the single-period and multi-period
//!     inventory models use one PMF/sampling implementation.
//!   * PORT NOTE: the optional HTML animation (FrameRecorder / newsvendor scene)
//!     and the `out/*.json` artifact writes are omitted (no animation engine /
//!     serde dependency assumed).

#![allow(dead_code)]

use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
pub use crate::des::main_newsvendor::{demand_poisson_pmf, sample_demand, DemandDist};

// -----------------------------------------------------------------------------
// Inventory MDP.
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InventoryParams {
    pub x_max: usize,
    pub a_max: usize,
    pub demand: DemandDist,
    pub unit_cost: f64,
    pub fixed_cost: f64,
    pub unit_price: f64,
    pub hold_cost: f64,
    pub lost_cost: f64,
    pub gamma: f64,
}

/// Build the lost-sales inventory MDP spec, pre-caching outcomes per `(x, a)`.
pub fn inventory_mdp_spec(p: &InventoryParams) -> MDPSpec {
    let num_states = p.x_max + 1;
    let x_max = p.x_max;
    let a_max = p.a_max;

    let mut cache: Vec<Vec<Vec<Outcome>>> = vec![Vec::new(); num_states];
    for (x, slot) in cache.iter_mut().enumerate() {
        let a_count = x_max.saturating_sub(x).min(a_max) + 1;
        let mut per_action: Vec<Vec<Outcome>> = vec![Vec::new(); a_count];
        for (a, ol) in per_action.iter_mut().enumerate() {
            let after = x + a; // inventory available before demand
            let order_cost = p.unit_cost * a as f64 + if a > 0 { p.fixed_cost } else { 0.0 };
            for d in 0..p.demand.pmf.len() {
                let pr = p.demand.pmf[d];
                if pr == 0.0 {
                    continue;
                }
                let sold = after.min(d) as f64;
                let leftover = after.saturating_sub(d);
                let lost = d.saturating_sub(after) as f64;
                let reward = p.unit_price * sold
                    - order_cost
                    - p.hold_cost * leftover as f64
                    - p.lost_cost * lost;
                let next_x = leftover;
                // Coalesce outcomes by (next_state, reward).
                if let Some(o) = ol
                    .iter_mut()
                    .find(|o| o.next_state == next_x && (o.reward - reward).abs() < 1e-12)
                {
                    o.prob += pr;
                } else {
                    ol.push(Outcome {
                        prob: pr,
                        reward,
                        next_state: next_x,
                    });
                }
            }
        }
        *slot = per_action;
    }

    MDPSpec {
        num_states,
        num_actions: Box::new(move |s: usize| x_max.saturating_sub(s).min(a_max) + 1),
        outcomes: Box::new(move |s: usize, a: usize| {
            cache
                .get(s)
                .and_then(|per| per.get(a))
                .cloned()
                .unwrap_or_default()
        }),
        is_terminal: Some(Box::new(|_s| false)),
        terminal_reward: None,
        state_label: Some(Box::new(|x| format!("x={x}"))),
        action_label: Some(Box::new(|a| format!("order={a}"))),
    }
}

/// One simulated day record.
#[derive(Clone, Copy, Debug)]
pub struct DayRecord {
    pub day: usize,
    pub x: usize,
    pub a: usize,
    pub d: usize,
    pub sold: usize,
    pub reward: f64,
    pub next_x: usize,
}

/// Aggregate simulation outcome.
#[derive(Clone, Debug)]
pub struct InventorySim {
    pub mean_reward: f64,
    pub mean_inventory: f64,
    pub mean_lost: f64,
    pub mean_leftover: f64,
    pub history: Vec<DayRecord>,
}

/// Simulate the MDP under a fixed (state → order) policy.
pub fn simulate_inventory_mdp(
    p: &InventoryParams,
    policy: &[usize],
    days: usize,
    seed: u32,
    initial_inventory: usize,
) -> InventorySim {
    with_seed(seed, |_| {
        let mut rng = mulberry32(seed);
        let mut x = initial_inventory;
        let (mut total_reward, mut total_inv, mut total_lost, mut total_leftover) =
            (0.0, 0.0, 0.0, 0.0);
        let mut history = Vec::with_capacity(days);
        for day in 0..days {
            let a = p.a_max.min(p.x_max.saturating_sub(x)).min(policy[x]);
            let d = sample_demand(&p.demand, &mut rng);
            let after = x + a;
            let sold = after.min(d);
            let leftover = after.saturating_sub(d);
            let lost = d.saturating_sub(after);
            let order_cost = p.unit_cost * a as f64 + if a > 0 { p.fixed_cost } else { 0.0 };
            let reward = p.unit_price * sold as f64
                - order_cost
                - p.hold_cost * leftover as f64
                - p.lost_cost * lost as f64;
            total_reward += reward;
            total_inv += x as f64;
            total_lost += lost as f64;
            total_leftover += leftover as f64;
            history.push(DayRecord {
                day,
                x,
                a,
                d,
                sold,
                reward,
                next_x: leftover,
            });
            x = leftover;
        }
        InventorySim {
            mean_reward: total_reward / days as f64,
            mean_inventory: total_inv / days as f64,
            mean_lost: total_lost / days as f64,
            mean_leftover: total_leftover / days as f64,
            history,
        }
    })
}

/// Detected policy structure.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicyKind {
    BaseStock,
    SS,
    Irregular,
}

#[derive(Clone, Debug)]
pub struct PolicyStructure {
    pub kind: PolicyKind,
    pub s_level: i64,
    pub reorder_point: i64,
    pub per_state: Vec<i64>,
}

/// Classify a discovered policy as base-stock / (s, S) / irregular.
pub fn detect_policy_structure(policy: &[i64]) -> PolicyStructure {
    let x_max = policy.len() as i64 - 1;
    let per_state: Vec<i64> = policy.to_vec();
    let t: Vec<i64> = per_state
        .iter()
        .enumerate()
        .map(|(x, &a)| x as i64 + a)
        .collect();
    let mut s: i64 = -1;
    let mut s_targets: Vec<i64> = Vec::new();
    for x in 0..=x_max {
        if per_state[x as usize] > 0 {
            s = x;
            let tx = t[x as usize];
            if !s_targets.contains(&tx) {
                s_targets.push(tx);
            }
        }
    }
    if s == -1 {
        return PolicyStructure {
            kind: PolicyKind::Irregular,
            s_level: 0,
            reorder_point: -1,
            per_state,
        };
    }
    if s_targets.len() > 1 {
        let max_s = *s_targets.iter().max().unwrap();
        return PolicyStructure {
            kind: PolicyKind::Irregular,
            s_level: max_s,
            reorder_point: s,
            per_state,
        };
    }
    let s_level = s_targets[0];
    for x in (s + 1)..=x_max {
        if per_state[x as usize] > 0 {
            return PolicyStructure {
                kind: PolicyKind::Irregular,
                s_level,
                reorder_point: s,
                per_state,
            };
        }
    }
    if s == s_level - 1 {
        PolicyStructure {
            kind: PolicyKind::BaseStock,
            s_level,
            reorder_point: s_level - 1,
            per_state,
        }
    } else {
        PolicyStructure {
            kind: PolicyKind::SS,
            s_level,
            reorder_point: s,
            per_state,
        }
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let lambda = env_f64("LAMBDA", 20.0);
    let d_max = env_usize("D_MAX", (lambda * 2.5).ceil() as usize);
    let x_max = env_usize("X_MAX", (lambda * 2.5).ceil() as usize);
    let a_max = env_usize("A_MAX", (lambda * 2.5).ceil() as usize);
    let params = InventoryParams {
        x_max,
        a_max,
        demand: demand_poisson_pmf(lambda, d_max),
        unit_cost: env_f64("UNIT_COST", 1.0),
        fixed_cost: env_f64("FIXED_COST", 0.0),
        unit_price: env_f64("UNIT_PRICE", 2.0),
        hold_cost: env_f64("HOLD_COST", 0.1),
        lost_cost: env_f64("LOST_COST", 0.5),
        gamma: env_f64("GAMMA", 0.95),
    };
    let days = env_usize("DAYS", 5000);
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1u32);

    println!("# Multi-period inventory MDP");
    println!("#   demand = Poisson(λ={lambda})  truncated at {d_max}");
    println!(
        "#   xMax={}, aMax={}, γ={}",
        params.x_max, params.a_max, params.gamma
    );
    println!(
        "#   unitCost={}, fixedCost={}, unitPrice={}",
        params.unit_cost, params.fixed_cost, params.unit_price
    );
    println!(
        "#   holdCost={}, lostCost={}",
        params.hold_cost, params.lost_cost
    );

    let spec = inventory_mdp_spec(&params);
    let result = value_iteration(
        spec,
        VIOptions {
            gamma: params.gamma,
            tol: 1e-8,
            max_iter: 5000,
            ..Default::default()
        },
    );

    println!(
        "\n# Value iteration converged in {} sweeps  (final ΔV = {:.2e})",
        result.iterations, result.final_delta
    );

    let policy_usize: Vec<usize> = result.policy.iter().map(|&v| v.max(0) as usize).collect();
    let policy_i64: Vec<i64> = result.policy.iter().map(|&v| v.max(0) as i64).collect();
    let st = detect_policy_structure(&policy_i64);
    let kind_label = match st.kind {
        PolicyKind::BaseStock => "base-stock",
        PolicyKind::SS => "s-S",
        PolicyKind::Irregular => "irregular",
    };
    println!("\n# Discovered policy structure: {kind_label}");
    println!("#   S (order-up-to) = {}", st.s_level);
    println!("#   reorder point s = {}", st.reorder_point);
    if st.kind == PolicyKind::SS {
        println!("#   (s, S) = ({}, {})", st.reorder_point, st.s_level);
    }

    let preview_n = 20.min(params.x_max + 1);
    println!("\n# π(x) and V(x) for x ∈ [0, {}]", preview_n - 1);
    println!("     x   action a   x+a (target)    V(x)");
    for x in 0..preview_n {
        println!(
            "  {:>4}    {:>4}        {:>4}        {:>10.3}",
            x,
            policy_usize[x],
            x + policy_usize[x],
            result.v[x]
        );
    }

    let sim = simulate_inventory_mdp(&params, &policy_usize, days, seed, 0);
    println!("\n# {days}-day simulation under discovered policy (initial x=0)");
    println!("    mean reward/day      = {:.3}", sim.mean_reward);
    println!("    mean inventory       = {:.2}", sim.mean_inventory);
    println!("    mean lost demand     = {:.2}", sim.mean_lost);
    println!("    mean leftover        = {:.2}", sim.mean_leftover);
    println!(
        "    long-run avg reward  ≈ V(0) · (1−γ) = {:.3}",
        result.v[0] * (1.0 - params.gamma)
    );

    // PORT NOTE: optional animation + JSON artifact writes omitted (see header).
    println!("# (animation + JSON artifact writes omitted in port — see PORT NOTE)");
}
