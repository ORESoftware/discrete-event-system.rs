//! Port of `src/des/main-stochastic-flow-mdp.ts`.
//!
//! Thin runner: MDP interpretation of stochastic max-flow; prints the
//! policy path plus a simulated trajectory.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`]; `process.env.SEED` → `std::env` + seed.
//!   - delegates to `general::stochastic_flow_mdp`.

use crate::des::general::stochastic_flow_mdp::{
    build_default_stochastic_flow_mdp_problem, solve_stochastic_flow_mdp, FlowMDPActionKind,
    SolveStochasticFlowMDPOptions,
};

/// `fmt(x, digits=3)` — finite numbers to 3 decimals, else `"n/a"`.
fn fmt(x: f64) -> String {
    if x.is_finite() {
        format!("{:.3}", x)
    } else {
        "n/a".to_string()
    }
}

/// JS `String(x)` for a number: integer-valued floats print bare.
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let seed: u32 = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let result = solve_stochastic_flow_mdp(
        build_default_stochastic_flow_mdp_problem(),
        SolveStochasticFlowMDPOptions {
            seed: Some(seed),
            max_policy_rows: Some(16),
        },
    );

    println!("# Stochastic flow control MDP");
    println!("# state=(current node, remaining capacities), action=edge attempt or wait");
    println!("# horizon={}, states={}", result.horizon, result.num_states);
    println!(
        "# deterministic max-flow upper bound={}",
        num_str(result.deterministic_max_flow)
    );
    println!("# optimal expected reward={}", fmt(result.expected_reward));
    println!(
        "# simulated delivered units (seed={})={}",
        seed,
        num_str(result.simulation.delivered)
    );
    println!(
        "# simulated total reward={}",
        fmt(result.simulation.total_reward)
    );
    println!();

    println!("## Initial-state policy path (success branch)");
    for row in &result.initial_policy {
        let caps: Vec<String> = row.state.capacities.iter().map(|c| num_str(*c)).collect();
        println!(
            "  t={}: node={}, caps=[{}] -> {}  V={}",
            row.stage,
            row.state.node,
            caps.join(","),
            row.action.label,
            fmt(row.value)
        );
    }
    println!();

    println!("## Simulated trajectory");
    for step in &result.simulation.steps {
        let ok = if step.action.kind == FlowMDPActionKind::Wait {
            "wait"
        } else if step.success {
            "success"
        } else {
            "fail"
        };
        println!(
            "  t={}: {} --{}/{}--> {}  r={}  delivered={}",
            step.stage,
            step.node_before,
            step.action.label,
            ok,
            step.node_after,
            fmt(step.reward),
            num_str(step.delivered_so_far)
        );
    }
}
