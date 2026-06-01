//! Port of `src/des/main-ip-mip-des.ts`.
//!
//! Thin runner: solve a binary-knapsack integer program via the IP/MIP
//! solver-as-DES and print the solution + station graph + trace.
//!
//! Delegates to `crate::des::general::ip_mip_des`. TS string casts
//! (`as LPRelaxationAlgorithm`, `as 'dfs' | 'best-bound'`) are parsed into the
//! corresponding Rust enums. `process.env.*` → `std::env::var`.

#![allow(dead_code)]

use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, solve_ipmip_with_des, ConcreteLpRelaxationAlgorithm,
    IPMIPSolveOptions, LpRelaxationAlgorithm, NodeSelection, TraceAction,
};

fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{x:.digits$}")
    } else {
        format!("{x}")
    }
}

/// Parse the `LP_ALGO` env value into an [`LpRelaxationAlgorithm`].
fn parse_lp_algo(s: &str) -> LpRelaxationAlgorithm {
    use ConcreteLpRelaxationAlgorithm::*;
    match s {
        "auto" => LpRelaxationAlgorithm::Auto,
        "incremental-primal-dual" => LpRelaxationAlgorithm::Concrete(IncrementalPrimalDual),
        "des-simplex-dantzig" => LpRelaxationAlgorithm::Concrete(DesSimplexDantzig),
        "des-simplex-bland" => LpRelaxationAlgorithm::Concrete(DesSimplexBland),
        "internal-simplex" => LpRelaxationAlgorithm::Concrete(InternalSimplex),
        "internal-ipm" | "internal-interior-point" => {
            LpRelaxationAlgorithm::Concrete(InternalInteriorPoint)
        }
        "external-highs" => LpRelaxationAlgorithm::Concrete(ExternalHighs),
        "external-highs-ds" => LpRelaxationAlgorithm::Concrete(ExternalHighsDs),
        "external-highs-ipm" => LpRelaxationAlgorithm::Concrete(ExternalHighsIpm),
        _ => LpRelaxationAlgorithm::Auto,
    }
}

fn trace_action_str(a: TraceAction) -> &'static str {
    match a {
        TraceAction::Branch => "branch",
        TraceAction::Cut => "cut",
        TraceAction::Prune => "prune",
        TraceAction::Incumbent => "incumbent",
        TraceAction::Unbounded => "unbounded",
    }
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let lp_algorithm = parse_lp_algo(&std::env::var("LP_ALGO").unwrap_or_else(|_| "auto".into()));
    let node_selection = match std::env::var("NODE_SELECTION")
        .unwrap_or_else(|_| "dfs".into())
        .as_str()
    {
        "best-bound" => NodeSelection::BestBound,
        _ => NodeSelection::Dfs,
    };
    let problem =
        build_binary_knapsack_ip(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
    let result = solve_ipmip_with_des(
        problem,
        IPMIPSolveOptions {
            lp_algorithm: Some(lp_algorithm),
            allow_external_solvers: Some(
                std::env::var("ALLOW_EXTERNAL_SOLVERS").as_deref() == Ok("1"),
            ),
            max_nodes: Some(
                std::env::var("MAX_NODES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(200),
            ),
            max_cut_rounds: Some(
                std::env::var("MAX_CUT_ROUNDS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1),
            ),
            node_selection: Some(node_selection),
            ..Default::default()
        },
    );

    println!("# IP/MIP solver graph as DES");
    println!("# LP backend:       {}", result.lp_algorithm.as_str());
    let usage = result
        .lp_algorithm_usage
        .iter()
        .map(|(k, v)| format!("{}={}", k.as_str(), v))
        .collect::<Vec<_>>()
        .join(", ");
    println!("# LP usage:         {usage}");
    println!(
        "# root plan:        {}",
        result.technique_plan.root_lp_algorithm.as_str()
    );
    println!("# execution mode:   {}", result.execution_mode);
    println!("# in-house only:    {}", result.in_house_only);
    println!("# status:           {}", result.status.as_str());
    println!("# z*:               {}", fmt(result.z, 4));
    println!("# best bound:       {}", fmt(result.best_bound, 4));
    println!("# gap:              {:.2e}", result.gap);
    println!(
        "# x*:               [{}]",
        result
            .x
            .iter()
            .map(|v| fmt(*v, 3))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("# nodes explored:   {}", result.nodes_explored);
    println!(
        "# elapsed:          {} ms ({:.2} nodes/s)",
        result.performance.elapsed_ms, result.performance.nodes_per_second
    );
    println!("# LP solves:        {}", result.lp_solves);
    println!(
        "# LP solver time:   {} ms",
        result.performance.total_lp_solver_ms
    );
    println!("# LP iterations:    {}", result.total_lp_iterations);
    println!("# cuts added:       {}", result.cuts_added);
    println!("# candidates tried: {}", result.candidates_tried);
    println!(
        "# solver tokens:    {} created ({} stateful, {} stateless)",
        result.token_stats.created, result.token_stats.stateful, result.token_stats.stateless
    );
    println!(
        "# incumbent source: {}",
        result.incumbent_source.as_deref().unwrap_or("none")
    );
    println!();

    println!("## Station graph");
    for n in &result.topology {
        println!("  {:<22} {}", n.id, n.role);
    }
    println!();

    println!("## First trace events");
    for ev in result.trace.iter().take(12) {
        let z = ev.lp_z.map(|v| fmt(v, 3)).unwrap_or_else(|| "n/a".into());
        let frac = if ev.fractional.is_empty() {
            String::new()
        } else {
            format!(
                " frac={{{}}}",
                ev.fractional
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let kids = match &ev.children {
            Some(c) => format!(
                " children=[{}]",
                c.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            None => String::new(),
        };
        let reason = match &ev.reason {
            Some(r) => format!(" -- {r}"),
            None => String::new(),
        };
        println!(
            "  node={} d={} z={} {}{}{}{}",
            ev.node_id,
            ev.depth,
            z,
            trace_action_str(ev.action),
            frac,
            kids,
            reason
        );
    }
}
