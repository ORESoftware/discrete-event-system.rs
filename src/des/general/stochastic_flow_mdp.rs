//! Port of `src/des/general/stochastic-flow-mdp.ts`
//! (module `des::general::stochastic_flow_mdp`).
//!
//! Max-flow recast as a finite-horizon stochastic-control MDP solved by backward
//! Bellman induction. Deterministic max-flow asks for a static feasible
//! circulation; if edge availability evolves stochastically, the same network
//! becomes an MDP whose state is the current packet node plus the remaining edge
//! capacities, whose action chooses an outgoing edge to try (or to wait), whose
//! noise is whether the chosen edge succeeds this tick, and whose reward is
//! delivered flow value minus routing, failure, and waiting costs. Each
//! `FiniteHorizonDPStation` tick performs one backward Bellman stage.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `class StochasticFlowMDPStation extends FiniteHorizonDPStation` → a struct
//!     holding a `StationCore` + a `DpState` (the base's protected state) + the
//!     problem and enumerated states, implementing the
//!     [`FiniteHorizonDPStation`] hook trait (the established pattern in
//!     `inventory_dp.rs`).
//!   * `stateKey(s): string` keyed the DP maps in TS. The header suggested
//!     `Hash + Eq` on the state, but capacities are `f64` (not `Hash`/`Eq`), so
//!     the faithful port keeps the TS string key in a `HashMap<String, usize>`
//!     (FLAGGED divergence from the header). Capacities are always integral, so
//!     the `f64::to_string` join reproduces the JS `Array.join(',')` key.
//!   * `interface IndexedAction extends FlowMDPAction` adds nothing the base
//!     interface lacks (`edgeIndex?` already exists), so it collapses into
//!     [`FlowMDPAction`]; the `'wait' | 'edge'` union → [`FlowMDPActionKind`].
//!   * `actionCache` is consulted from `transitions` (`&self`), so it lives
//!     behind a `RefCell` for interior mutability; `legalActions` returns a fresh
//!     `Vec` (cloned from the cache) rather than a borrowed array.
//!   * `simulateStochasticFlowPolicy` already used `mulberry32(seed)` (not
//!     ambient `Math.random`), so the deterministic seed is kept and the RNG is a
//!     [`mulberry32`] `SeededRandom` read via `next_float()`.
//!   * `validate*` `throw` → [`Preconditions`] guards turned into `panic!`;
//!     `assertNoValidationFailures` → the ported [`assert_no_validation_failures`]
//!     (`.expect` at the edge, matching the TS throw).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::finite_horizon_dp::{
    DPOutcome, DpState, FiniteHorizonDPStation, StageStat,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{
    assert_no_validation_failures, run_iterative_des, IterativeRunOptions,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::general::max_flow::{solve_max_flow, MaxFlowEdge, MaxFlowProblem};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

const MODEL: &str = "stochastic-flow-mdp";

/// Panic with the precondition message on a failed guard (TS `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// ── Problem / result shapes ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct StochasticFlowEdge {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    /// Probability that one unit successfully traverses the edge when tried.
    pub success_prob: f64,
    /// Optional per-attempt cost. Defaults to 0.
    pub cost: Option<f64>,
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct StochasticFlowMDPProblem {
    pub num_nodes: usize,
    pub source: usize,
    pub sink: usize,
    pub edges: Vec<StochasticFlowEdge>,
    /// Number of sequential control ticks.
    pub horizon: usize,
    /// Reward for delivering one unit to the sink. Defaults to 1.
    pub delivered_reward: Option<f64>,
    /// Penalty for choosing to wait. Defaults to 0.
    pub wait_penalty: Option<f64>,
    /// Penalty when a chosen edge is unavailable. Defaults to 0.
    pub failure_penalty: Option<f64>,
    /// Discount factor. Defaults to 1 for finite-horizon total reward.
    pub discount: Option<f64>,
    /// Guardrail for exact state enumeration. Defaults to 20000.
    pub max_states: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct FlowMDPState {
    pub node: usize,
    pub capacities: Vec<f64>,
}

/// `'wait' | 'edge'` action kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowMDPActionKind {
    Wait,
    Edge,
}

#[derive(Clone, Debug)]
pub struct FlowMDPAction {
    pub kind: FlowMDPActionKind,
    pub edge_index: Option<usize>,
    pub label: String,
}

impl FlowMDPAction {
    fn wait() -> Self {
        FlowMDPAction {
            kind: FlowMDPActionKind::Wait,
            edge_index: None,
            label: "wait".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowMDPDecision {
    pub stage: usize,
    pub state_index: usize,
    pub state: FlowMDPState,
    pub action: FlowMDPAction,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct FlowMDPSimStep {
    pub stage: usize,
    pub node_before: usize,
    pub action: FlowMDPAction,
    pub success: bool,
    pub node_after: usize,
    pub reward: f64,
    pub delivered_so_far: f64,
    pub capacities_after: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct StochasticFlowSimulation {
    pub seed: u32,
    pub delivered: f64,
    pub total_reward: f64,
    pub final_state: FlowMDPState,
    pub steps: Vec<FlowMDPSimStep>,
}

#[derive(Clone, Debug)]
pub struct StochasticFlowMDPResult {
    pub status: String,
    pub horizon: usize,
    pub num_states: usize,
    pub initial_state_index: usize,
    pub expected_reward: f64,
    pub deterministic_max_flow: f64,
    pub policy: Vec<FlowMDPDecision>,
    pub initial_policy: Vec<FlowMDPDecision>,
    pub stage_history: Vec<StageStat>,
    pub simulation: StochasticFlowSimulation,
}

/// Options for [`solve_stochastic_flow_mdp`] (TS `{seed?, maxPolicyRows?}`).
#[derive(Clone, Debug, Default)]
pub struct SolveStochasticFlowMDPOptions {
    pub seed: Option<u32>,
    pub max_policy_rows: Option<usize>,
}

// ── Validation ────────────────────────────────────────────────────────────────

pub fn validate_stochastic_flow_mdp_problem(p: &StochasticFlowMDPProblem) {
    require(Preconditions::integer_in_range(
        MODEL,
        "numNodes",
        p.num_nodes as f64,
        2.0,
        1000.0,
    ));
    require(Preconditions::integer_in_range(
        MODEL,
        "source",
        p.source as f64,
        0.0,
        (p.num_nodes - 1) as f64,
    ));
    require(Preconditions::integer_in_range(
        MODEL,
        "sink",
        p.sink as f64,
        0.0,
        (p.num_nodes - 1) as f64,
    ));
    require(Preconditions::check(
        MODEL,
        "source != sink",
        "hold",
        p.source != p.sink,
        Some(format!("[{}, {}]", p.source, p.sink)),
    ));
    require(Preconditions::non_empty(MODEL, "edges", &p.edges));
    require(Preconditions::integer_in_range(
        MODEL,
        "horizon",
        p.horizon as f64,
        1.0,
        1000.0,
    ));
    require(Preconditions::positive(
        MODEL,
        "deliveredReward",
        p.delivered_reward.unwrap_or(1.0),
    ));
    require(Preconditions::non_negative(
        MODEL,
        "waitPenalty",
        p.wait_penalty.unwrap_or(0.0),
    ));
    require(Preconditions::non_negative(
        MODEL,
        "failurePenalty",
        p.failure_penalty.unwrap_or(0.0),
    ));
    require(Preconditions::in_range(
        MODEL,
        "discount",
        p.discount.unwrap_or(1.0),
        0.0,
        1.0,
    ));
    require(Preconditions::integer_in_range(
        MODEL,
        "maxStates",
        p.max_states.unwrap_or(20000) as f64,
        1.0,
        1e7,
    ));
    for (i, e) in p.edges.iter().enumerate() {
        require(Preconditions::integer_in_range(
            MODEL,
            &format!("edges[{i}].from"),
            e.from as f64,
            0.0,
            (p.num_nodes - 1) as f64,
        ));
        require(Preconditions::integer_in_range(
            MODEL,
            &format!("edges[{i}].to"),
            e.to as f64,
            0.0,
            (p.num_nodes - 1) as f64,
        ));
        require(Preconditions::integer_in_range(
            MODEL,
            &format!("edges[{i}].capacity"),
            e.capacity,
            0.0,
            100.0,
        ));
        require(Preconditions::in_range(
            MODEL,
            &format!("edges[{i}].successProb"),
            e.success_prob,
            0.0,
            1.0,
        ));
        if let Some(cost) = e.cost {
            require(Preconditions::non_negative(
                MODEL,
                &format!("edges[{i}].cost"),
                cost,
            ));
        }
    }
}

// ── Station ───────────────────────────────────────────────────────────────────

pub struct StochasticFlowMDPStation {
    core: StationCore,
    dp: DpState,
    p: StochasticFlowMDPProblem,
    pub states: Vec<FlowMDPState>,
    pub initial_state_index: usize,
    key_to_index: HashMap<String, usize>,
    outgoing: HashMap<usize, Vec<usize>>,
    action_cache: RefCell<HashMap<usize, Vec<FlowMDPAction>>>,
}

impl StochasticFlowMDPStation {
    pub fn new(p: StochasticFlowMDPProblem) -> Self {
        validate_stochastic_flow_mdp_problem(&p);
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, e) in p.edges.iter().enumerate() {
            outgoing.entry(e.from).or_default().push(i);
        }
        let mut st = StochasticFlowMDPStation {
            core: StationCore::new(MODEL),
            dp: DpState::default(),
            p,
            states: Vec::new(),
            initial_state_index: 0,
            key_to_index: HashMap::new(),
            outgoing,
            action_cache: RefCell::new(HashMap::new()),
        };
        st.enumerate_states();
        let init_caps: Vec<f64> = st.p.edges.iter().map(|e| e.capacity).collect();
        st.initial_state_index = st.state_index(&FlowMDPState {
            node: st.p.source,
            capacities: init_caps,
        });
        st.bootstrap();

        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "stochastic-flow-mdp.policy-actions-legal",
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<StochasticFlowMDPStation>()
                        .unwrap();
                    st.policy_actions_legal()
                },
                Some("every policy action is legal for its state".to_string()),
                Some(Box::new(|s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<StochasticFlowMDPStation>()
                        .unwrap();
                    format!("states={}, horizon={}", st.states.len(), st.p.horizon)
                })),
                Some("stochastic-flow-mdp-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "stochastic-flow-mdp.values-finite",
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<StochasticFlowMDPStation>()
                        .unwrap();
                    st.dp.v.iter().all(|row| row.iter().all(|v| v.is_finite()))
                },
                Some("all value-function entries finite".to_string()),
                Some(Box::new(|s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<StochasticFlowMDPStation>()
                        .unwrap();
                    let v0 = st.dp.v.first().map(|r| r[st.initial_state_index]);
                    format!(
                        "V0(initial)={}",
                        v0.map(|x| x.to_string())
                            .unwrap_or_else(|| "undefined".to_string())
                    )
                })),
                Some("stochastic-flow-mdp-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    pub fn legal_actions(&self, state_index: usize) -> Vec<FlowMDPAction> {
        if let Some(cached) = self.action_cache.borrow().get(&state_index) {
            return cached.clone();
        }
        let state = &self.states[state_index];
        let mut out = vec![FlowMDPAction::wait()];
        if state.node != self.p.sink {
            if let Some(edges) = self.outgoing.get(&state.node) {
                for &edge_index in edges {
                    if state.capacities[edge_index] <= 0.0 {
                        continue;
                    }
                    let e = &self.p.edges[edge_index];
                    out.push(FlowMDPAction {
                        kind: FlowMDPActionKind::Edge,
                        edge_index: Some(edge_index),
                        label: e
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("{}->{}", e.from, e.to)),
                    });
                }
            }
        }
        self.action_cache
            .borrow_mut()
            .insert(state_index, out.clone());
        out
    }

    pub fn get_action_detail(&self, stage: usize, state_index: usize) -> FlowMDPAction {
        let action_index = self.get_action(stage, state_index);
        let actions = self.legal_actions(state_index);
        match action_index {
            Some(a) => actions.get(a).cloned().unwrap_or_else(FlowMDPAction::wait),
            None => FlowMDPAction::wait(),
        }
    }

    pub fn build_result(&self, seed: u32, max_policy_rows: usize) -> StochasticFlowMDPResult {
        let mut initial_policy: Vec<FlowMDPDecision> = Vec::new();
        let mut s = self.initial_state_index;
        for t in 0..self.p.horizon {
            let a = self.get_action_detail(t, s);
            initial_policy.push(FlowMDPDecision {
                stage: t,
                state_index: s,
                state: clone_state(&self.states[s]),
                action: a,
                value: self.get_v(t)[s],
            });
            let action_idx = self.get_action(t, s).unwrap_or(0);
            let outs = self.transitions(s, action_idx, t);
            let delivered_success = outs
                .iter()
                .find(|o| o.next_state != s)
                .copied()
                .unwrap_or(outs[0]);
            s = delivered_success.next_state;
        }
        StochasticFlowMDPResult {
            status: "optimal".to_string(),
            horizon: self.p.horizon,
            num_states: self.states.len(),
            initial_state_index: self.initial_state_index,
            expected_reward: self.get_v(0)[self.initial_state_index],
            deterministic_max_flow: solve_max_flow(self.as_deterministic_max_flow()).max_flow,
            policy: self.compact_policy(max_policy_rows),
            initial_policy,
            stage_history: self.dp.stage_history.clone(),
            simulation: simulate_stochastic_flow_policy(&self.p, self, seed),
        }
    }

    pub fn as_deterministic_max_flow(&self) -> MaxFlowProblem {
        MaxFlowProblem {
            num_nodes: self.p.num_nodes,
            source: self.p.source,
            sink: self.p.sink,
            edges: self
                .p
                .edges
                .iter()
                .map(|e| MaxFlowEdge {
                    from: e.from,
                    to: e.to,
                    capacity: e.capacity,
                    name: e.name.clone(),
                })
                .collect(),
        }
    }

    pub fn index_of_state(&self, s: &FlowMDPState) -> usize {
        self.state_index(s)
    }

    fn compact_policy(&self, max_rows: usize) -> Vec<FlowMDPDecision> {
        let mut rows: Vec<FlowMDPDecision> = Vec::new();
        for t in 0..self.p.horizon {
            if rows.len() >= max_rows {
                break;
            }
            for s in 0..self.states.len() {
                if rows.len() >= max_rows {
                    break;
                }
                let a = self.get_action_detail(t, s);
                if a.kind == FlowMDPActionKind::Wait && self.get_v(t)[s] <= 1e-12 {
                    continue;
                }
                rows.push(FlowMDPDecision {
                    stage: t,
                    state_index: s,
                    state: clone_state(&self.states[s]),
                    action: a,
                    value: self.get_v(t)[s],
                });
            }
        }
        rows
    }

    fn enumerate_states(&mut self) {
        let caps: Vec<f64> = self.p.edges.iter().map(|e| e.capacity).collect();
        let mut current = vec![0.0_f64; caps.len()];
        self.visit_caps(0, &caps, &mut current);
    }

    fn visit_caps(&mut self, idx: usize, caps: &[f64], current: &mut Vec<f64>) {
        if idx == caps.len() {
            let max_states = self.p.max_states.unwrap_or(20000);
            for node in 0..self.p.num_nodes {
                let st = FlowMDPState {
                    node,
                    capacities: current.clone(),
                };
                let key = state_key(&st);
                self.key_to_index.insert(key, self.states.len());
                self.states.push(st);
                if self.states.len() > max_states {
                    panic!("{MODEL}: exact state space exceeds maxStates={max_states}");
                }
            }
            return;
        }
        let mut c = 0.0;
        while c <= caps[idx] {
            current[idx] = c;
            self.visit_caps(idx + 1, caps, current);
            c += 1.0;
        }
    }

    fn state_index(&self, s: &FlowMDPState) -> usize {
        let key = state_key(s);
        *self
            .key_to_index
            .get(&key)
            .unwrap_or_else(|| panic!("{MODEL}: missing enumerated state {key}"))
    }

    fn policy_actions_legal(&self) -> bool {
        for t in 0..self.p.horizon {
            for s in 0..self.states.len() {
                match self.dp.policy.get(t).and_then(|row| row.get(s)).copied() {
                    Some(Some(idx)) => {
                        if idx >= self.legal_actions(s).len() {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }
}

impl DESStation for StochasticFlowMDPStation {
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
        self.assert_preconditions_dp()
            .expect("stochastic-flow-mdp preconditions");
    }
}

impl FiniteHorizonDPStation for StochasticFlowMDPStation {
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
        self.states.len()
    }
    fn stage_discount(&self, _stage: usize) -> f64 {
        self.p.discount.unwrap_or(1.0)
    }
    fn num_actions(&self, state: usize, _stage: usize) -> usize {
        self.legal_actions(state).len()
    }

    fn transitions(
        &self,
        state_index: usize,
        action_index: usize,
        _stage: usize,
    ) -> Vec<DPOutcome> {
        let state = &self.states[state_index];
        let actions = self.legal_actions(state_index);
        let action = actions.get(action_index);
        let is_wait = match action {
            None => true,
            Some(a) => a.kind == FlowMDPActionKind::Wait,
        };
        if is_wait {
            return vec![DPOutcome {
                prob: 1.0,
                reward: -(self.p.wait_penalty.unwrap_or(0.0)),
                next_state: state_index,
            }];
        }
        let action = action.unwrap();
        let edge_index = action.edge_index.unwrap();
        let edge = &self.p.edges[edge_index];
        let attempt_cost = edge.cost.unwrap_or(0.0);
        let fail_penalty = self.p.failure_penalty.unwrap_or(0.0);
        let p_succ = edge.success_prob;
        let mut next_caps = state.capacities.clone();
        next_caps[edge_index] -= 1.0;
        let delivered = edge.to == self.p.sink;
        let next_node = if delivered { self.p.source } else { edge.to };
        let success_state = self.state_index(&FlowMDPState {
            node: next_node,
            capacities: next_caps,
        });
        let success_reward = (if delivered {
            self.p.delivered_reward.unwrap_or(1.0)
        } else {
            0.0
        }) - attempt_cost;
        let failure_reward = -attempt_cost - fail_penalty;
        if p_succ <= 0.0 {
            return vec![DPOutcome {
                prob: 1.0,
                reward: failure_reward,
                next_state: state_index,
            }];
        }
        if p_succ >= 1.0 {
            return vec![DPOutcome {
                prob: 1.0,
                reward: success_reward,
                next_state: success_state,
            }];
        }
        vec![
            DPOutcome {
                prob: p_succ,
                reward: success_reward,
                next_state: success_state,
            },
            DPOutcome {
                prob: 1.0 - p_succ,
                reward: failure_reward,
                next_state: state_index,
            },
        ]
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn solve_stochastic_flow_mdp(
    p: StochasticFlowMDPProblem,
    opts: SolveStochasticFlowMDPOptions,
) -> StochasticFlowMDPResult {
    let station = Rc::new(RefCell::new(StochasticFlowMDPStation::new(p)));
    let summary = run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );
    assert_no_validation_failures(&summary, MODEL)
        .expect("stochastic-flow-mdp: post-run validation failed");
    let result = station
        .borrow()
        .build_result(opts.seed.unwrap_or(1), opts.max_policy_rows.unwrap_or(24));
    result
}

pub fn simulate_stochastic_flow_policy(
    p: &StochasticFlowMDPProblem,
    station: &StochasticFlowMDPStation,
    seed: u32,
) -> StochasticFlowSimulation {
    let mut rng = mulberry32(seed);
    let mut state_index = station.initial_state_index;
    let mut delivered = 0.0_f64;
    let mut total_reward = 0.0_f64;
    let mut steps: Vec<FlowMDPSimStep> = Vec::new();
    for t in 0..p.horizon {
        let state = station.states[state_index].clone();
        let action = station.get_action_detail(t, state_index);
        let before = state.node;
        let mut success = false;
        let mut reward = -(p.wait_penalty.unwrap_or(0.0));
        let mut next_state_index = state_index;
        if action.kind == FlowMDPActionKind::Edge {
            let edge = &p.edges[action.edge_index.unwrap()];
            success = rng.next_float() < edge.success_prob;
            reward = -(edge.cost.unwrap_or(0.0));
            if success {
                let mut next_caps = state.capacities.clone();
                next_caps[action.edge_index.unwrap()] -= 1.0;
                let reached_sink = edge.to == p.sink;
                if reached_sink {
                    delivered += 1.0;
                    reward += p.delivered_reward.unwrap_or(1.0);
                }
                next_state_index = station.index_of_state(&FlowMDPState {
                    node: if reached_sink { p.source } else { edge.to },
                    capacities: next_caps,
                });
            } else {
                reward -= p.failure_penalty.unwrap_or(0.0);
            }
        }
        total_reward += reward;
        let next_state = station.states[next_state_index].clone();
        steps.push(FlowMDPSimStep {
            stage: t,
            node_before: before,
            action: action.clone(),
            success,
            node_after: next_state.node,
            reward,
            delivered_so_far: delivered,
            capacities_after: next_state.capacities.clone(),
        });
        state_index = next_state_index;
    }
    StochasticFlowSimulation {
        seed,
        delivered,
        total_reward,
        final_state: clone_state(&station.states[state_index]),
        steps,
    }
}

pub fn build_default_stochastic_flow_mdp_problem() -> StochasticFlowMDPProblem {
    StochasticFlowMDPProblem {
        num_nodes: 5,
        source: 0,
        sink: 4,
        horizon: 8,
        delivered_reward: Some(1.0),
        wait_penalty: Some(0.01),
        failure_penalty: Some(0.03),
        discount: Some(1.0),
        edges: vec![
            StochasticFlowEdge {
                from: 0,
                to: 1,
                capacity: 2.0,
                success_prob: 0.90,
                cost: Some(0.01),
                name: Some("s-a".to_string()),
            },
            StochasticFlowEdge {
                from: 1,
                to: 4,
                capacity: 2.0,
                success_prob: 0.80,
                cost: Some(0.01),
                name: Some("a-t".to_string()),
            },
            StochasticFlowEdge {
                from: 0,
                to: 2,
                capacity: 2.0,
                success_prob: 0.65,
                cost: Some(0.01),
                name: Some("s-b".to_string()),
            },
            StochasticFlowEdge {
                from: 2,
                to: 4,
                capacity: 2.0,
                success_prob: 0.95,
                cost: Some(0.01),
                name: Some("b-t".to_string()),
            },
            StochasticFlowEdge {
                from: 1,
                to: 2,
                capacity: 1.0,
                success_prob: 0.75,
                cost: Some(0.01),
                name: Some("a-b".to_string()),
            },
            StochasticFlowEdge {
                from: 2,
                to: 1,
                capacity: 1.0,
                success_prob: 0.70,
                cost: Some(0.01),
                name: Some("b-a".to_string()),
            },
        ],
        max_states: Some(10000),
    }
}

fn state_key(s: &FlowMDPState) -> String {
    let caps = s
        .capacities
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{}|{}", s.node, caps)
}

fn clone_state(s: &FlowMDPState) -> FlowMDPState {
    FlowMDPState {
        node: s.node,
        capacities: s.capacities.clone(),
    }
}

#[cfg(test)]
mod tests {
    //! No tests accompany the TypeScript source; this smoke test drives the
    //! default instance to confirm the backward-induction station reaches an
    //! optimal policy whose value is finite and whose expected reward does not
    //! exceed the deterministic max-flow upper bound by more than the horizon.

    use super::*;

    #[test]
    fn solves_default_instance() {
        let result = solve_stochastic_flow_mdp(
            build_default_stochastic_flow_mdp_problem(),
            SolveStochasticFlowMDPOptions::default(),
        );
        assert_eq!(result.status, "optimal");
        assert!(result.expected_reward.is_finite());
        assert!(result.num_states > 0);
        assert_eq!(result.initial_policy.len(), 8);
        assert!(result.deterministic_max_flow >= 0.0);
        // The optimal expected reward cannot exceed delivering one unit per tick.
        assert!(
            result.expected_reward <= 8.0 + 1e-9,
            "expected_reward = {}",
            result.expected_reward
        );
    }
}
