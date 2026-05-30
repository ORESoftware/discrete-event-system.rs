//! Port of src/des/test/newsvendor-test.ts
//!
//! PORT NOTE: depends on `main-newsvendor` (analytical_optimal_q,
//! brute_search_optimal_q, demand_poisson_pmf, demand_uniform_pmf, profit,
//! expected_profit, mdp_optimal_q, …) and `main-inventory-mdp`
//! (inventory_mdp_spec, detect_policy_structure), neither of which is ported
//! yet. Only general::value_iteration exists. Test body deferred until the
//! newsvendor / inventory-MDP modules are migrated.
#![allow(dead_code)]

#[cfg(test)]
mod tests {}
