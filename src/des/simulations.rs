//! Serial simulation driver.
//!
//! Each TypeScript `main-*.ts` script was a standalone executable you would run
//! as its own Node process. In Rust they are plain library modules exposing a
//! `pub fn run()`, so the natural way to exercise the whole catalogue is to call
//! them one after another in a single process — strictly in series, not in
//! parallel. (Parallelism would interleave their `println!` output and race on
//! the process-global simulation clock / RNG seeding, so serial is the correct
//! model here.)
//!
//! The cluster/server scaffolding scripts (`child`, `parent`, `program`) are
//! intentionally excluded: they model long-lived master/worker processes and
//! socket servers, not finite simulations.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

/// Outcome of running one simulation entry point.
#[derive(Clone, Debug)]
pub struct SimOutcome {
    pub name: &'static str,
    pub ok: bool,
    pub millis: u128,
}

/// The full catalogue of simulation entry points, in a stable order. Each entry
/// is a `(name, fn())` pair where the function is the module's `run`.
pub fn simulation_catalogue() -> Vec<(&'static str, fn())> {
    vec![
        ("main", crate::des::main::run),
        (
            "main_backpropagation",
            crate::des::main_backpropagation::run,
        ),
        ("main_build_site", crate::des::main_build_site::run),
        ("main_calculus", crate::des::main_calculus::run),
        (
            "main_computer_network",
            crate::des::main_computer_network::run,
        ),
        ("main_contact_seir", crate::des::main_contact_seir::run),
        ("main_convolution", crate::des::main_convolution::run),
        ("main_court_mdp", crate::des::main_court_mdp::run),
        ("main_dc_motor", crate::des::main_dc_motor::run),
        ("main_dc_motor_anim", crate::des::main_dc_motor_anim::run),
        (
            "main_delivery_planner",
            crate::des::main_delivery_planner::run,
        ),
        ("main_dispatch_combo", crate::des::main_dispatch_combo::run),
        (
            "main_electric_circuit",
            crate::des::main_electric_circuit::run,
        ),
        ("main_elevator", crate::des::main_elevator::run),
        (
            "main_elevator_highrise",
            crate::des::main_elevator_highrise::run,
        ),
        (
            "main_empirical_control",
            crate::des::main_empirical_control::run,
        ),
        (
            "main_empirical_control_report",
            crate::des::main_empirical_control_report::run,
        ),
        ("main_epidemic", crate::des::main_epidemic::run),
        (
            "main_epidemic_improved",
            crate::des::main_epidemic_improved::run,
        ),
        ("main_factmachine", crate::des::main_factmachine::run),
        (
            "main_factmachine_markets",
            crate::des::main_factmachine_markets::run,
        ),
        (
            "main_factory_floor_track3t",
            crate::des::main_factory_floor_track3t::run,
        ),
        (
            "main_fibonacci_recursion",
            crate::des::main_fibonacci_recursion::run,
        ),
        (
            "main_fibonacci_scheduled",
            crate::des::fibonacci_scheduled::run,
        ),
        (
            "main_checkpoint_precedence",
            crate::des::checkpoint_precedence::run,
        ),
        (
            "main_task_build_order",
            crate::des::checkpoint_precedence::task_dag::run,
        ),
        ("main_from_json", crate::des::main_from_json::run),
        ("main_evolution_lab", crate::des::main_evolution_lab::run),
        ("main_genetic_tsp", crate::des::main_genetic_tsp::run),
        (
            "main_hazard_function_survival_analysis",
            crate::des::main_hazard_function_survival_analysis::run,
        ),
        ("main_incremental_lp", crate::des::main_incremental_lp::run),
        ("main_inventory_mdp", crate::des::main_inventory_mdp::run),
        ("main_ip_mip_des", crate::des::main_ip_mip_des::run),
        (
            "main_knapsack_problem",
            crate::des::main_knapsack_problem::run,
        ),
        ("main_lp_des", crate::des::main_lp_des::run),
        ("main_lp_factory", crate::des::main_lp_factory::run),
        ("main_markov", crate::des::main_markov::run),
        ("main_mdp_lp", crate::des::main_mdp_lp::run),
        ("main_milp_bnb", crate::des::main_milp_bnb::run),
        (
            "main_monte_carlo_sim",
            crate::des::main_monte_carlo_sim::run,
        ),
        ("main_network_mutex", crate::des::main_network_mutex::run),
        ("main_neural_net", crate::des::main_neural_net::run),
        ("main_newsvendor", crate::des::main_newsvendor::run),
        (
            "main_numerical_solver_anim",
            crate::des::main_numerical_solver_anim::run,
        ),
        (
            "main_observability_controllability",
            crate::des::main_observability_controllability::run,
        ),
        (
            "main_observability_controllability_anim",
            crate::des::main_observability_controllability_anim::run,
        ),
        (
            "main_optimization_as_des",
            crate::des::main_optimization_as_des::run,
        ),
        ("main_plumbing_flow", crate::des::main_plumbing_flow::run),
        ("main_shadow_eval", crate::des::main_shadow_eval::run),
        ("main_shortest_path", crate::des::main_shortest_path::run),
        (
            "main_shortest_path_algo",
            crate::des::main_shortest_path_algo::run,
        ),
        (
            "main_signal_processing",
            crate::des::main_signal_processing::run,
        ),
        (
            "main_simulated_annealing",
            crate::des::main_simulated_annealing::run,
        ),
        ("main_snowball", crate::des::main_snowball::run),
        (
            "main_soccer_rotation",
            crate::des::main_soccer_rotation::run,
        ),
        (
            "main_soccer_rotation_anim",
            crate::des::main_soccer_rotation::run_anim,
        ),
        ("main_soccer", crate::des::main_soccer::run),
        ("main_soccer_planner", crate::des::main_soccer_planner::run),
        (
            "main_stochastic_flow_mdp",
            crate::des::main_stochastic_flow_mdp::run,
        ),
        ("main_stochastic_lp", crate::des::main_stochastic_lp::run),
        ("main_stochastic_sde", crate::des::main_stochastic_sde::run),
        (
            "main_stochastic_sde_report",
            crate::des::main_stochastic_sde_report::run,
        ),
        ("main_temp_control", crate::des::main_temp_control::run),
        (
            "main_temp_control_anim",
            crate::des::main_temp_control_anim::run,
        ),
        ("main_traffic", crate::des::main_traffic::run),
        ("main_two_disease", crate::des::main_two_disease::run),
        ("main_wind_mppt", crate::des::main_wind_mppt::run),
        ("main_wind_mppt_anim", crate::des::main_wind_mppt_anim::run),
        ("max_flow", crate::des::max_flow::run),
    ]
}

/// Run only the simulations whose name contains `needle`, in series. Useful for
/// targeted cross-validation against the reference implementation.
pub fn run_simulations_matching(needle: &str) -> Vec<SimOutcome> {
    let selected: Vec<(&'static str, fn())> = simulation_catalogue()
        .into_iter()
        .filter(|(n, _)| n.contains(needle))
        .collect();
    if selected.is_empty() {
        eprintln!("no simulation matches `{needle}`");
        return Vec::new();
    }
    run_list(selected)
}

/// Run every simulation in [`simulation_catalogue`] one at a time, in series.
///
/// Each simulation is isolated with [`catch_unwind`] so that one failure (a
/// missing CLI default, a `panic!` invariant, etc.) is recorded and the driver
/// keeps going through the rest of the catalogue. Returns one [`SimOutcome`] per
/// simulation in run order.
pub fn run_all_simulations() -> Vec<SimOutcome> {
    run_list(simulation_catalogue())
}

/// Run a specific list of `(name, fn)` simulations in series with panic
/// isolation, printing per-simulation headers and a final summary.
fn run_list(catalogue: Vec<(&'static str, fn())>) -> Vec<SimOutcome> {
    let total = catalogue.len();
    let mut outcomes = Vec::with_capacity(total);

    for (idx, (name, sim)) in catalogue.into_iter().enumerate() {
        println!("\n========== [{}/{}] {name} ==========", idx + 1, total);
        let start = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(sim));
        let millis = start.elapsed().as_millis();
        let ok = result.is_ok();
        if !ok {
            eprintln!("!! simulation `{name}` panicked (see stderr above)");
        }
        outcomes.push(SimOutcome { name, ok, millis });
    }

    let passed = outcomes.iter().filter(|o| o.ok).count();
    println!("\n========== SUMMARY ==========");
    for o in &outcomes {
        println!(
            "  {:<42} {:>6} ms   {}",
            o.name,
            o.millis,
            if o.ok { "OK" } else { "FAILED" }
        );
    }
    println!("\n{passed}/{total} simulations completed without panicking.");

    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke check that the observability/controllability pipeline runs without
    /// panicking (it previously panicked because pure evaluator stations never
    /// drained their DES inbox). `#[ignore]`d because it prints a full report;
    /// run with `cargo test obs_ctrl_simulation_runs -- --ignored`.
    #[test]
    #[ignore]
    fn obs_ctrl_simulation_runs() {
        crate::des::main_observability_controllability::run();
    }

    /// The catalogue is non-empty and free of duplicate names.
    #[test]
    fn catalogue_is_well_formed() {
        let cat = simulation_catalogue();
        assert!(cat.len() >= 56, "expected the full simulation catalogue");
        let mut names: Vec<&str> = cat.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(
            before,
            names.len(),
            "duplicate simulation names in catalogue"
        );
    }
}
