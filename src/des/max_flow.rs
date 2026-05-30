//! Port of `src/des/max-flow.ts`.
//!
//! Thin runner: solves a textbook maximum-flow problem as a DES and prints the
//! augmenting-path trace, edge flows, and min cut.
//!
//! NOTE: this is the TOP-LEVEL `des::max_flow` module — distinct from
//! `des::general::max_flow`, which holds the solver this runner delegates to.
//!
//! Conversion notes:
//!   - `if (require.main === module) main()` → [`run`] (library crate, not `fn main`).
//!   - delegates to `general::max_flow::{build_textbook_max_flow_problem,
//!     solve_max_flow}`.
//!   - `number` is `f64`; the `fmt` helper is a free fn.

use crate::des::general::max_flow::{build_textbook_max_flow_problem, solve_max_flow};

/// `fmt(x)` — integers print bare, non-integers to 4 decimals.
fn fmt(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{:.4}", x)
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let result = solve_max_flow(build_textbook_max_flow_problem());

    println!("# Maximum-flow optimiser as DES");
    println!("# one augmenting path = one DES fixed-point tick");
    println!(
        "# source={}, sink={}, nodes={}",
        result.source, result.sink, result.num_nodes
    );
    println!("# max flow = {}", fmt(result.max_flow));
    println!(
        "# augmentations = {}, iterations = {}",
        result.trace.len(),
        result.iterations
    );
    println!();

    println!("## Augmenting-path trace");
    for t in &result.trace {
        let path: Vec<String> = t.path.iter().map(|p| p.to_string()).collect();
        println!(
            "  iter {:>2}: path {}  bottleneck={}  flow={}",
            t.iter,
            path.join(" -> "),
            fmt(t.bottleneck),
            fmt(t.flow_after)
        );
    }
    println!();

    println!("## Edge flows");
    for e in &result.edge_flows {
        let name = match &e.name {
            Some(n) => format!("{} ", n),
            None => String::new(),
        };
        println!(
            "  {}{} -> {}: {} / {}",
            name,
            e.from,
            e.to,
            fmt(e.flow),
            fmt(e.capacity)
        );
    }
    println!();

    println!("## Min cut");
    let source_side: Vec<String> = result
        .min_cut
        .source_side
        .iter()
        .map(|n| n.to_string())
        .collect();
    let sink_side: Vec<String> = result
        .min_cut
        .sink_side
        .iter()
        .map(|n| n.to_string())
        .collect();
    println!("  S = {{{}}}", source_side.join(", "));
    println!("  T = {{{}}}", sink_side.join(", "));
    println!("  capacity = {}", fmt(result.min_cut.capacity));
}
