//! Port of `src/des/general/inventory-dp.ts` — multi-period STOCHASTIC INVENTORY
//! MANAGEMENT solved by FINITE-HORIZON DYNAMIC PROGRAMMING (backward induction).
//!
//! T-period horizon. At the start of each period the manager observes on-hand
//! inventory s and chooses an order quantity a. Demand is random with a known
//! PMF; the per-period cash flow trades selling price against purchase, fixed,
//! holding, and stockout costs. Terminal salvage is salvageValue per leftover
//! unit. The Bellman recursion V_t(s) = max_a E[ r + gamma V_{t+1}(s') ] is the
//! workhorse textbook example, and DP gives the exact optimum for any PMF.
//!
//! ## Rust shape
//!
//! `class InventoryDPStation extends FiniteHorizonDPStation` → a struct holding
//! a `StationCore` + a `DpState` (the base's protected state) + the problem, and
//! implementing the [`FiniteHorizonDPStation`] hook trait. The free functions
//! `solveInventoryDP` / `simulateInventory` / `sampleFromPmf` stay free; the
//! `mulberry32` closure RNG is injected as a [`SeededRandom`].

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::finite_horizon_dp::{
    DPOutcome, DpState, FiniteHorizonDPStation,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Specification of a multi-period stochastic inventory problem.
#[derive(Clone, Debug)]
pub struct InventoryProblem {
    /// Number of decision periods T.
    pub horizon: usize,
    /// Maximum on-hand inventory. State space is {0, …, S_max}.
    pub s_max: usize,
    /// Per-period demand distribution. `demand_pmf[d] = P(D = d)`.
    pub demand_pmf: Vec<f64>,
    /// Selling price per unit.
    pub price: f64,
    /// Variable order cost per unit.
    pub cost: f64,
    /// Fixed ordering cost (per period when a > 0).
    pub fixed_cost: f64,
    /// Holding cost per unit per period.
    pub hold_cost: f64,
    /// Stockout penalty per unit shortage.
    pub stockout_cost: f64,
    /// Salvage value per leftover unit at terminal stage.
    pub salvage_value: f64,
    /// Discount factor gamma ∈ (0, 1]. `None` ⇒ 1 (undiscounted finite horizon).
    pub discount: Option<f64>,
    /// Initial inventory at t=0.
    pub initial_inventory: usize,
}

impl InventoryProblem {
    fn discount_value(&self) -> f64 {
        self.discount.unwrap_or(1.0)
    }
}

/// `class InventoryDPStation extends FiniteHorizonDPStation`.
pub struct InventoryDPStation {
    core: StationCore,
    dp: DpState,
    p: InventoryProblem,
    /// Pre-computed E[demand] for diagnostics.
    pub mean_demand: f64,
}

impl InventoryDPStation {
    pub fn new(p: InventoryProblem) -> Self {
        if p.horizon < 1 {
            panic!("horizon must be ≥ 1");
        }
        // `s_max < 0` is unrepresentable for `usize`; the demand PMF guards remain.
        if p.demand_pmf.is_empty() {
            panic!("demandPmf must be non-empty");
        }
        let sum_p: f64 = p.demand_pmf.iter().sum();
        if (sum_p - 1.0).abs() > 1e-9 {
            panic!("demandPmf must sum to 1, got {sum_p}");
        }
        let mean_demand: f64 = p
            .demand_pmf
            .iter()
            .enumerate()
            .map(|(d, &pd)| pd * d as f64)
            .sum();

        let mut st = InventoryDPStation {
            core: StationCore::new("inventory-dp"),
            dp: DpState::default(),
            p,
            mean_demand,
        };
        st.bootstrap();
        st.add_inventory_validators();
        st
    }

    fn add_inventory_validators(&mut self) {
        // a ∈ [0, S_max − s] for every (t, s).
        let v1 = intrinsic_check::<dyn DESStation>(
            "inventory-dp.policy-feasible",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<InventoryDPStation>().unwrap();
                for t in 0..st.p.horizon {
                    let pol = &st.dp.policy[t];
                    for (state, slot) in pol.iter().enumerate() {
                        if let Some(a) = *slot {
                            if state + a > st.p.s_max {
                                return false;
                            }
                        }
                    }
                }
                true
            },
            Some("a ∈ [0, S_max − s] for every (t, s)".to_string()),
            Some(Box::new(|s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<InventoryDPStation>().unwrap();
                format!("T={}  S_max={}", st.p.horizon, st.p.s_max)
            })),
            Some("inventory-dp-intrinsic".to_string()),
            None,
        );
        // V_T(s) = salvageValue · s.
        let v2 = intrinsic_check::<dyn DESStation>(
            "inventory-dp.terminal-V-equals-salvage",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<InventoryDPStation>().unwrap();
                let t = st.p.horizon;
                let sv = st.p.salvage_value;
                let vt = &st.dp.v[t];
                for (state, &val) in vt.iter().enumerate() {
                    if (val - sv * state as f64).abs() > 1e-9 {
                        return false;
                    }
                }
                true
            },
            Some("V_T(s) = salvageValue · s".to_string()),
            Some(Box::new(|s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<InventoryDPStation>().unwrap();
                format!(
                    "salvage={}, V_T(0)={}",
                    st.p.salvage_value, st.dp.v[st.p.horizon][0]
                )
            })),
            Some("inventory-dp-intrinsic".to_string()),
            None,
        );
        // every V[t][s] finite.
        let v3 = intrinsic_check::<dyn DESStation>(
            "inventory-dp.value-finite-everywhere",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<InventoryDPStation>().unwrap();
                for t in 0..=st.p.horizon {
                    for &v in &st.dp.v[t] {
                        if !v.is_finite() {
                            return false;
                        }
                    }
                }
                true
            },
            Some("every V[t][s] finite".to_string()),
            Some(Box::new(|_s: &dyn DESStation| "no NaN/Inf".to_string())),
            Some("inventory-dp-intrinsic".to_string()),
            None,
        );
        self.add_validator(v1.boxed());
        self.add_validator(v2.boxed());
        self.add_validator(v3.boxed());
    }

    /// `V[t][s]` for `t = 0 … T` (full value table).
    pub fn value_table(&self) -> &[Vec<f64>] {
        &self.dp.v
    }

    /// `π[t][s]` for `t = 0 … T-1` (sentinel `None` slots → never occurs here).
    pub fn policy_table(&self) -> &[Vec<Option<usize>>] {
        &self.dp.policy
    }
}

impl DESStation for InventoryDPStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.run_dp_step();
    }
    fn has_work(&self) -> bool {
        self.dp_has_work()
    }
    fn assert_preconditions(&mut self) {
        // The TS base threw on a bad instance; fail fast here too.
        self.assert_preconditions_dp().expect("inventory-dp preconditions");
    }
}

impl FiniteHorizonDPStation for InventoryDPStation {
    fn dp_state(&self) -> &DpState {
        &self.dp
    }
    fn dp_state_mut(&mut self) -> &mut DpState {
        &mut self.dp
    }

    fn horizon(&self) -> usize {
        self.p.horizon
    }
    fn num_states(&self) -> usize {
        self.p.s_max + 1
    }
    fn num_actions(&self, state: usize, _stage: usize) -> usize {
        self.p.s_max - state + 1 // a = 0 … S_max - s
    }
    fn stage_discount(&self, _stage: usize) -> f64 {
        self.p.discount_value()
    }
    fn terminal_reward(&self, state: usize) -> f64 {
        self.p.salvage_value * state as f64
    }

    fn transitions(&self, state: usize, action: usize, _stage: usize) -> Vec<DPOutcome> {
        let p = &self.p;
        let inv = state + action;
        let order_cost = p.cost * action as f64 + if action > 0 { p.fixed_cost } else { 0.0 };
        let mut out = Vec::new();
        for (d, &prob) in p.demand_pmf.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let sales = d.min(inv) as f64;
            let leftover = inv.saturating_sub(d);
            let shortage = d.saturating_sub(inv) as f64;
            let reward = p.price * sales
                - order_cost
                - p.hold_cost * leftover as f64
                - p.stockout_cost * shortage;
            out.push(DPOutcome { prob, reward, next_state: leftover });
        }
        out
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// A simulated trajectory under a policy (the TS inline `simulation` object).
#[derive(Clone, Debug)]
pub struct InventorySimulation {
    pub inventory: Vec<usize>,
    pub orders: Vec<usize>,
    pub demands: Vec<usize>,
    pub rewards: Vec<f64>,
    pub total_reward: f64,
}

/// Result of [`solve_inventory_dp`].
#[derive(Clone, Debug)]
pub struct InventoryDPResult {
    /// `V[t][s]` for `t = 0 … T`.
    pub v: Vec<Vec<f64>>,
    /// `π[t][s]` for `t = 0 … T-1`.
    pub policy: Vec<Vec<usize>>,
    /// Expected total reward starting from `(t=0, s=initialInventory)`.
    pub expected_reward: f64,
    /// A simulated trajectory under the optimal policy (for sanity / display).
    pub simulation: InventorySimulation,
    /// Mean demand (for diagnostics).
    pub mean_demand: f64,
    pub ticks: usize,
}

/// Solve the inventory DP, then simulate one path under the optimal policy.
pub fn solve_inventory_dp(p: &InventoryProblem, seed: Option<u32>) -> InventoryDPResult {
    // Pre-construction guards on the problem instance.
    let cls = "solveInventoryDP";
    Preconditions::integer_in_range(cls, "horizon", p.horizon as f64, 1.0, 1e6).expect("horizon");
    Preconditions::integer_in_range(cls, "S_max", p.s_max as f64, 0.0, 1e6).expect("S_max");
    Preconditions::integer_in_range(
        cls,
        "initialInventory",
        p.initial_inventory as f64,
        0.0,
        p.s_max as f64,
    )
    .expect("initialInventory");
    Preconditions::probability_vector(cls, "demandPmf", &p.demand_pmf, 1e-9).expect("demandPmf");
    Preconditions::non_negative(cls, "price", p.price).expect("price");
    Preconditions::non_negative(cls, "cost", p.cost).expect("cost");
    Preconditions::non_negative(cls, "fixedCost", p.fixed_cost).expect("fixedCost");
    Preconditions::non_negative(cls, "holdCost", p.hold_cost).expect("holdCost");
    Preconditions::non_negative(cls, "stockoutCost", p.stockout_cost).expect("stockoutCost");
    Preconditions::finite(cls, "salvageValue", p.salvage_value).expect("salvageValue");
    Preconditions::in_range(cls, "discount", p.discount_value(), 0.0, 1.0).expect("discount");

    let station = Rc::new(RefCell::new(InventoryDPStation::new(p.clone())));
    let summary = run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions::default(),
    );

    let st = station.borrow();
    let v: Vec<Vec<f64>> = st.dp.v.iter().map(|row| row.clone()).collect();
    let policy: Vec<Vec<usize>> = st
        .dp
        .policy
        .iter()
        .map(|row| row.iter().map(|o| o.unwrap_or(0)).collect())
        .collect();
    let expected_reward = st.dp.v[0][p.initial_inventory];
    let mean_demand = st.mean_demand;
    drop(st);

    let sim = simulate_inventory(p, &policy, seed.unwrap_or(1));
    InventoryDPResult {
        v,
        policy,
        expected_reward,
        simulation: sim,
        mean_demand,
        ticks: summary.ticks,
    }
}

/// Forward simulation under a given policy.
pub fn simulate_inventory(
    p: &InventoryProblem,
    policy: &[Vec<usize>],
    seed: u32,
) -> InventorySimulation {
    let mut rng = SeededRandom::new(seed);
    let mut inventory: Vec<usize> = vec![p.initial_inventory];
    let mut orders: Vec<usize> = Vec::new();
    let mut demands: Vec<usize> = Vec::new();
    let mut rewards: Vec<f64> = Vec::new();
    let mut total = 0.0;
    let mut s = p.initial_inventory;
    for t in 0..p.horizon {
        let a = policy[t][s];
        let inv = s + a;
        let d = sample_from_pmf(&p.demand_pmf, &mut rng);
        let sales = d.min(inv) as f64;
        let leftover = inv.saturating_sub(d);
        let shortage = d.saturating_sub(inv) as f64;
        let order_cost = p.cost * a as f64 + if a > 0 { p.fixed_cost } else { 0.0 };
        let r = p.price * sales
            - order_cost
            - p.hold_cost * leftover as f64
            - p.stockout_cost * shortage;
        orders.push(a);
        demands.push(d);
        rewards.push(r);
        total += r;
        s = leftover;
        inventory.push(s);
    }
    // Add terminal salvage as a final reward entry for transparency.
    total += p.salvage_value * s as f64;
    InventorySimulation {
        inventory,
        orders,
        demands,
        rewards,
        total_reward: total,
    }
}

fn sample_from_pmf(pmf: &[f64], rng: &mut dyn RandomSource) -> usize {
    let u = rng.next_float();
    let mut acc = 0.0;
    for (i, &p) in pmf.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    pmf.len() - 1
}

#[cfg(test)]
mod tests {
    //! Inventory-DP tests: a tiny instance with a closed-form optimum, plus the
    //! intrinsic-validator sanity checks.

    use super::*;

    /// Single-period newsvendor: horizon 1, S_max 2, demand is 1 w.p. 1.
    /// price 5, unit holding cost so ordering exactly one unit is strictly
    /// optimal (no tie with over-ordering). Optimal is to order 1 unit and
    /// sell it: V_0(0) = 5.
    fn newsvendor() -> InventoryProblem {
        InventoryProblem {
            horizon: 1,
            s_max: 2,
            demand_pmf: vec![0.0, 1.0, 0.0],
            price: 5.0,
            cost: 0.0,
            fixed_cost: 0.0,
            hold_cost: 1.0,
            stockout_cost: 0.0,
            salvage_value: 0.0,
            discount: None,
            initial_inventory: 0,
        }
    }

    #[test]
    fn optimal_cost_on_tiny_instance() {
        let res = solve_inventory_dp(&newsvendor(), Some(1));
        // Order 1, sell 1 at price 5, everything else free ⇒ expected reward 5.
        assert!((res.expected_reward - 5.0).abs() < 1e-9, "got {}", res.expected_reward);
        // Stage-0 policy at state 0 must order exactly one unit.
        assert_eq!(res.policy[0][0], 1);
    }

    #[test]
    fn ordering_costs_make_it_unprofitable() {
        // Same demand but the variable cost (6) exceeds the price (5): never
        // worth ordering, so the optimal value is 0 and the order is 0.
        let p = InventoryProblem { cost: 6.0, ..newsvendor() };
        let res = solve_inventory_dp(&p, Some(7));
        assert!(res.expected_reward.abs() < 1e-9, "got {}", res.expected_reward);
        assert_eq!(res.policy[0][0], 0);
    }

    #[test]
    fn intrinsic_validators_pass() {
        let station = InventoryDPStation::new(newsvendor());
        // Drive the DP to completion so V/policy are filled.
        let s = Rc::new(RefCell::new(station));
        run_iterative_des(vec![s.clone() as StationRef], IterativeRunOptions::default());
        let checks = crate::des::general::des_base::station::run_station_validation(&*s.borrow());
        assert!(checks.iter().all(|c| c.passed), "validators failed: {checks:?}");
    }
}
