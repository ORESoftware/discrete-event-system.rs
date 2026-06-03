//! Port of `src/des/general/multistage-stochastic.ts` — module
//! `des::general::multistage_stochastic`.
//!
//! Multi-stage stochastic programming via an SDDP-style discrete-event system,
//! over a compact one-dimensional inventory / storage problem. The state is the
//! inventory at the start of a stage; each stage observes a random demand and
//! decides an order, sell, stockout, and ending inventory, maximising expected
//! profit with a terminal salvage value.
//!
//! SDDP representation: each stage owns an upper affine cut pool for the concave
//! value function `V_t(s)`; one DES tick performs a forward sampled trajectory
//! and a backward cut pass; each backward stage LP uses the next stage's cut
//! pool through a theta variable. A tiny exact extensive-form scenario-tree LP
//! is included for validation.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * The `interface`s become structs; `SDDPStation` (a
//!     `FixedPointIterationStation<SDDPState>`) becomes a struct + the trait
//!     impl. The `PureTransform` solvers become structs implementing
//!     [`Transform`].
//!   * INJECTED RNG: forward trajectories sample demand via the shared
//!     [`SeededRandom`] (the mulberry32 analogue) routed through
//!     [`RandomSource`], not an ambient global.
//!   * Stage LPs reuse [`LPProblem`] / [`solve_lp_internal`]; [`AffineCut`] /
//!     [`AffineCutPool`] come from `des_base::cut_pool`; the per-stage cut pools
//!     are `Vec<AffineCutPool>` fields.
//!   * `validate*` throws become `panic!` on a failed [`Preconditions`] guard
//!     (construction-time invariants). Numerics are `f64`; stage / scenario
//!     indices are `usize`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::cut_pool::{AffineCut, AffineCutPool, CutEnvelopeSense};
use crate::des::general::des_base::fixed_point::{
    ConvergenceReason, FixedPointCore, FixedPointIterationStation, FixedPointOptions,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::general::lp::{
    self, solve_lp_internal, InternalSimplexOptions, LPProblem, LPStatus,
};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::Transform;

const MODEL: &str = "multi-stage-sddp";

/// Panic with the precondition message on a failed guard (TS `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

/// One demand realisation and its probability mass.
#[derive(Clone, Copy, Debug)]
pub struct DemandOutcome {
    pub demand: f64,
    pub prob: f64,
}

/// The multi-stage inventory problem definition.
#[derive(Clone, Debug)]
pub struct MultiStageInventoryProblem {
    pub horizon: usize,
    pub initial_inventory: f64,
    pub capacity: f64,
    pub max_order: Vec<f64>,
    pub price: Vec<f64>,
    pub order_cost: Vec<f64>,
    pub hold_cost: Vec<f64>,
    pub stockout_cost: Vec<f64>,
    pub salvage_value: f64,
    pub demands: Vec<Vec<DemandOutcome>>,
}

/// The result of solving one stage LP.
#[derive(Clone, Debug)]
pub struct StageDecision {
    pub status: LPStatus,
    pub value: f64,
    pub immediate_reward: f64,
    pub order: f64,
    pub sell: f64,
    pub stockout: f64,
    pub next_inventory: f64,
    pub theta: f64,
}

/// A cut recorded in the iteration trace.
#[derive(Clone, Copy, Debug)]
pub struct CutAddedTrace {
    pub stage: usize,
    pub alpha: f64,
    pub beta: f64,
    pub state: f64,
}

/// One realised step of the last forward trajectory.
#[derive(Clone, Copy, Debug)]
pub struct SamplePathEntry {
    pub stage: usize,
    pub demand: f64,
    pub state: f64,
    pub order: f64,
    pub sell: f64,
    pub stockout: f64,
    pub next_inventory: f64,
}

/// One SDDP iteration's trace entry.
#[derive(Clone, Debug)]
pub struct SDDPIterationTrace {
    pub iter: usize,
    pub sampled_demands: Vec<f64>,
    pub states: Vec<f64>,
    pub terminal_inventory: f64,
    pub cuts_added: Vec<CutAddedTrace>,
    pub upper_bound: f64,
    pub policy_value: Option<f64>,
    pub gap_to_exact: Option<f64>,
}

/// Summary of the exact extensive-form scenario-tree solve.
#[derive(Clone, Debug)]
pub struct ExactTreeNodeResult {
    pub objective: f64,
    pub node_count: usize,
    pub lp_vars: usize,
    pub lp_rows: usize,
    pub status: String,
}

/// SDDP solve status (TS `'optimal' | 'iter-limit'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SDDPStatus {
    Optimal,
    IterLimit,
}

/// Options for [`solve_multi_stage_sddp`]. `None` fields take the TS defaults.
#[derive(Clone, Debug, Default)]
pub struct SDDPOptions {
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub seed: Option<u32>,
    pub exact_objective: Option<f64>,
    pub evaluate_policy_every: Option<usize>,
    pub finite_diff_step: Option<f64>,
    pub cut_grid_size: Option<usize>,
}

/// Resolved options.
#[derive(Clone, Debug)]
struct SDDPFilledOptions {
    max_iter: usize,
    tol: f64,
    seed: u32,
    exact_objective: Option<f64>,
    evaluate_policy_every: usize,
    finite_diff_step: f64,
    cut_grid_size: usize,
}

/// The full SDDP result.
#[derive(Clone, Debug)]
pub struct SDDPResult {
    pub status: SDDPStatus,
    pub iterations: usize,
    pub upper_bound: f64,
    pub policy_value: f64,
    pub exact_objective: Option<f64>,
    pub gap_to_exact: Option<f64>,
    pub cuts_per_stage: Vec<usize>,
    pub cuts: Vec<Vec<AffineCut>>,
    pub trace: Vec<SDDPIterationTrace>,
    pub sample_path: Vec<SamplePathEntry>,
}

/// Both halves of the demo: exact extensive form + SDDP approximation.
#[derive(Clone, Debug)]
pub struct MultiStageRunResult {
    pub exact: ExactTreeNodeResult,
    pub sddp: SDDPResult,
}

// -----------------------------------------------------------------------------
// Public problem builders
// -----------------------------------------------------------------------------

/// The default 4-stage inventory problem from the TS source.
pub fn build_default_multi_stage_inventory_problem() -> MultiStageInventoryProblem {
    MultiStageInventoryProblem {
        horizon: 4,
        initial_inventory: 4.0,
        capacity: 10.0,
        max_order: vec![5.0, 5.0, 5.0, 5.0],
        price: vec![9.0, 9.0, 10.0, 10.0],
        order_cost: vec![3.0, 3.0, 4.0, 4.0],
        hold_cost: vec![0.25, 0.25, 0.35, 0.35],
        stockout_cost: vec![7.0, 7.0, 8.0, 8.0],
        salvage_value: 1.5,
        demands: vec![
            vec![
                DemandOutcome {
                    demand: 2.0,
                    prob: 0.45,
                },
                DemandOutcome {
                    demand: 6.0,
                    prob: 0.55,
                },
            ],
            vec![
                DemandOutcome {
                    demand: 1.0,
                    prob: 0.35,
                },
                DemandOutcome {
                    demand: 5.0,
                    prob: 0.65,
                },
            ],
            vec![
                DemandOutcome {
                    demand: 3.0,
                    prob: 0.50,
                },
                DemandOutcome {
                    demand: 7.0,
                    prob: 0.50,
                },
            ],
            vec![
                DemandOutcome {
                    demand: 2.0,
                    prob: 0.60,
                },
                DemandOutcome {
                    demand: 6.0,
                    prob: 0.40,
                },
            ],
        ],
    }
}

/// Validate the problem definition. TS `throw`s become `panic!` here.
pub fn validate_multi_stage_problem(p: &MultiStageInventoryProblem) {
    require(Preconditions::integer_in_range(
        MODEL,
        "horizon",
        p.horizon as f64,
        1.0,
        200.0,
    ));
    require(Preconditions::positive(MODEL, "capacity", p.capacity));
    require(Preconditions::in_range(
        MODEL,
        "initialInventory",
        p.initial_inventory,
        0.0,
        p.capacity,
    ));
    require(Preconditions::length_eq(
        MODEL,
        "maxOrder",
        &p.max_order,
        p.horizon,
    ));
    require(Preconditions::length_eq(
        MODEL, "price", &p.price, p.horizon,
    ));
    require(Preconditions::length_eq(
        MODEL,
        "orderCost",
        &p.order_cost,
        p.horizon,
    ));
    require(Preconditions::length_eq(
        MODEL,
        "holdCost",
        &p.hold_cost,
        p.horizon,
    ));
    require(Preconditions::length_eq(
        MODEL,
        "stockoutCost",
        &p.stockout_cost,
        p.horizon,
    ));
    require(Preconditions::length_eq(
        MODEL, "demands", &p.demands, p.horizon,
    ));
    require(Preconditions::arr_non_negative(
        MODEL,
        "maxOrder",
        &p.max_order,
    ));
    require(Preconditions::arr_non_negative(MODEL, "price", &p.price));
    require(Preconditions::arr_non_negative(
        MODEL,
        "orderCost",
        &p.order_cost,
    ));
    require(Preconditions::arr_non_negative(
        MODEL,
        "holdCost",
        &p.hold_cost,
    ));
    require(Preconditions::arr_non_negative(
        MODEL,
        "stockoutCost",
        &p.stockout_cost,
    ));
    require(Preconditions::non_negative(
        MODEL,
        "salvageValue",
        p.salvage_value,
    ));
    for t in 0..p.horizon {
        require(Preconditions::non_empty(
            MODEL,
            &format!("demands[{t}]"),
            &p.demands[t],
        ));
        let probs: Vec<f64> = p.demands[t].iter().map(|d| d.prob).collect();
        require(Preconditions::probability_vector(
            MODEL,
            &format!("demands[{t}].prob"),
            &probs,
            1e-9,
        ));
        for i in 0..p.demands[t].len() {
            require(Preconditions::non_negative(
                MODEL,
                &format!("demands[{t}][{i}].demand"),
                p.demands[t][i].demand,
            ));
        }
    }
}

// -----------------------------------------------------------------------------
// Stage LP
// -----------------------------------------------------------------------------

/// Query for one stage LP solve.
#[derive(Clone, Copy, Debug)]
pub struct StageDecisionInput<'a> {
    pub stage: usize,
    pub state: f64,
    pub demand: f64,
    pub next_cuts: &'a AffineCutPool,
}

/// Solve a single stage LP. The problem is configuration; the per-call query is
/// the `transform` input.
pub struct StageDecisionSolver<'p> {
    p: &'p MultiStageInventoryProblem,
}

impl<'p> StageDecisionSolver<'p> {
    pub fn new(p: &'p MultiStageInventoryProblem) -> Self {
        StageDecisionSolver { p }
    }
}

impl<'p, 'a> Transform<StageDecisionInput<'a>, StageDecision> for StageDecisionSolver<'p> {
    fn transform(&self, input: StageDecisionInput<'a>) -> StageDecision {
        let p = self.p;
        let StageDecisionInput {
            stage,
            state,
            demand,
            next_cuts,
        } = input;
        validate_stage_inputs(p, stage, state, demand, next_cuts);
        let c = vec![
            -p.order_cost[stage],
            p.price[stage],
            -p.stockout_cost[stage],
            -p.hold_cost[stage],
            1.0,
        ];
        let mut a_ub: Vec<Vec<f64>> =
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0, 0.0]];
        let mut b_ub: Vec<f64> = vec![p.max_order[stage], p.capacity];
        for cut in next_cuts.all() {
            a_ub.push(vec![0.0, 0.0, 0.0, -cut.beta[0], 1.0]);
            b_ub.push(cut.alpha);
        }
        let lp = LPProblem {
            sense: lp::Sense::Max,
            c,
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: Some(vec![
                vec![-1.0, 1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 1.0, 0.0, 0.0],
            ]),
            b_eq: Some(vec![state, demand]),
            lb: Some(vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0), None]),
            var_names: Some(vec![
                "order".to_string(),
                "sell".to_string(),
                "stockout".to_string(),
                "nextInventory".to_string(),
                "theta".to_string(),
            ]),
            ..Default::default()
        };
        let sol = solve_lp_internal(
            &lp,
            &InternalSimplexOptions {
                max_iter: Some(10000),
                tol: None,
                basis_start: None,
            },
        );
        if sol.status != LPStatus::Optimal {
            return StageDecision {
                status: sol.status,
                value: f64::NAN,
                immediate_reward: f64::NAN,
                order: f64::NAN,
                sell: f64::NAN,
                stockout: f64::NAN,
                next_inventory: f64::NAN,
                theta: f64::NAN,
            };
        }
        let (order, sell, stockout, next_inventory, theta) =
            (sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4]);
        let immediate_reward = p.price[stage] * sell
            - p.order_cost[stage] * order
            - p.hold_cost[stage] * next_inventory
            - p.stockout_cost[stage] * stockout;
        StageDecision {
            status: LPStatus::Optimal,
            value: sol.objective,
            immediate_reward,
            order,
            sell,
            stockout,
            next_inventory,
            theta,
        }
    }
}

/// Deprecated shim: prefer `StageDecisionSolver::new(p).transform({stage, state, demand, next_cuts})`.
pub fn solve_stage_decision(
    p: &MultiStageInventoryProblem,
    stage: usize,
    state: f64,
    demand: f64,
    next_cuts: &AffineCutPool,
) -> StageDecision {
    StageDecisionSolver::new(p).transform(StageDecisionInput {
        stage,
        state,
        demand,
        next_cuts,
    })
}

fn validate_stage_inputs(
    p: &MultiStageInventoryProblem,
    stage: usize,
    state: f64,
    demand: f64,
    next_cuts: &AffineCutPool,
) {
    require(Preconditions::integer_in_range(
        MODEL,
        "stage",
        stage as f64,
        0.0,
        p.horizon as f64 - 1.0,
    ));
    require(Preconditions::in_range(
        MODEL, "state", state, 0.0, p.capacity,
    ));
    require(Preconditions::non_negative(MODEL, "demand", demand));
    require(Preconditions::check(
        MODEL,
        "nextCuts.dimension",
        "equal 1",
        next_cuts.dimension == 1,
        Some(next_cuts.dimension.to_string()),
    ));
    require(Preconditions::check(
        MODEL,
        "nextCuts.size()",
        "be >= 1",
        next_cuts.size() >= 1,
        Some(next_cuts.size().to_string()),
    ));
}

/// Query for the expected stage value.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedStageValueInput<'a> {
    pub stage: usize,
    pub state: f64,
    pub next_cuts: &'a AffineCutPool,
}

/// Expectation of the stage LP value over the stage's demand outcomes.
pub struct ExpectedStageValue<'p> {
    p: &'p MultiStageInventoryProblem,
}

impl<'p> ExpectedStageValue<'p> {
    pub fn new(p: &'p MultiStageInventoryProblem) -> Self {
        ExpectedStageValue { p }
    }
}

impl<'p, 'a> Transform<ExpectedStageValueInput<'a>, f64> for ExpectedStageValue<'p> {
    fn transform(&self, input: ExpectedStageValueInput<'a>) -> f64 {
        let p = self.p;
        let ExpectedStageValueInput {
            stage,
            state,
            next_cuts,
        } = input;
        let mut z = 0.0;
        for d in &p.demands[stage] {
            let dec = solve_stage_decision(p, stage, state, d.demand, next_cuts);
            if dec.status != LPStatus::Optimal {
                panic!(
                    "{MODEL}: stage LP failed with status {}",
                    dec.status.as_str()
                );
            }
            z += d.prob * dec.value;
        }
        z
    }
}

/// Deprecated shim: prefer `ExpectedStageValue::new(p).transform({stage, state, next_cuts})`.
pub fn expected_stage_value(
    p: &MultiStageInventoryProblem,
    stage: usize,
    state: f64,
    next_cuts: &AffineCutPool,
) -> f64 {
    ExpectedStageValue::new(p).transform(ExpectedStageValueInput {
        stage,
        state,
        next_cuts,
    })
}

/// The finite-difference / cut-grid options consumed by [`generate_value_cut`]
/// (TS `Required<Pick<SDDPOptions, 'finiteDiffStep' | 'cutGridSize'>>`).
#[derive(Clone, Copy, Debug)]
struct CutOpts {
    finite_diff_step: f64,
    cut_grid_size: usize,
}

fn generate_value_cut(
    p: &MultiStageInventoryProblem,
    stage: usize,
    state: f64,
    next_cuts: &AffineCutPool,
    opts: CutOpts,
    source: &str,
) -> AffineCut {
    let value = expected_stage_value(p, stage, state, next_cuts);
    let h = opts.finite_diff_step.max(1e-7).min(p.capacity);
    let beta: f64 = if state <= h {
        let up = expected_stage_value(p, stage, p.capacity.min(state + h), next_cuts);
        (up - value) / (p.capacity.min(state + h) - state).max(1e-12)
    } else if p.capacity - state <= h {
        let lo = expected_stage_value(p, stage, 0.0_f64.max(state - h), next_cuts);
        (value - lo) / (state - 0.0_f64.max(state - h)).max(1e-12)
    } else {
        let up = expected_stage_value(p, stage, state + h, next_cuts);
        let lo = expected_stage_value(p, stage, state - h, next_cuts);
        (up - lo) / (2.0 * h)
    };
    let mut alpha = value - beta * state;

    // Lift the finite-difference cut over a small state grid so it remains a
    // valid upper cut on the domain.
    let grid_n = 2.max(opts.cut_grid_size);
    let mut max_violation = 0.0;
    for i in 0..grid_n {
        let x = p.capacity * i as f64 / (grid_n - 1) as f64;
        let vx = expected_stage_value(p, stage, x, next_cuts);
        let cutx = alpha + beta * x;
        if vx > cutx + max_violation {
            max_violation = vx - cutx;
        }
    }
    alpha += max_violation + 1e-8;
    AffineCut {
        alpha,
        beta: vec![beta],
        source: Some(source.to_string()),
    }
}

// -----------------------------------------------------------------------------
// SDDP DES station
// -----------------------------------------------------------------------------

/// The fixed-point iteration state carried by [`SDDPStation`].
#[derive(Clone, Debug)]
struct SDDPState {
    iter: usize,
    upper_bound: f64,
    policy_value: Option<f64>,
    gap_to_exact: Option<f64>,
}

/// SDDP run-loop station: one forward/backward SDDP pass per tick.
pub struct SDDPStation {
    core: StationCore,
    fp: FixedPointCore<SDDPState>,
    pub cut_pools: Vec<AffineCutPool>,
    pub trace: Vec<SDDPIterationTrace>,
    pub last_sample_path: Vec<SamplePathEntry>,
    p: MultiStageInventoryProblem,
    rng: SeededRandom,
    exact_objective: Option<f64>,
    evaluate_policy_every: usize,
    finite_diff_step: f64,
    cut_grid_size: usize,
    final_status: SDDPStatus,
}

fn downcast_sddp(s: &dyn DESStation) -> &SDDPStation {
    s.as_any()
        .downcast_ref::<SDDPStation>()
        .expect("validator received a non-SDDPStation station")
}

impl SDDPStation {
    fn new(p: MultiStageInventoryProblem, opts: &SDDPFilledOptions) -> Self {
        validate_multi_stage_problem(&p);
        let mut station = SDDPStation {
            core: StationCore::new(MODEL),
            fp: FixedPointCore::new(FixedPointOptions {
                max_iter: Some(opts.max_iter),
                tol: Some(opts.tol),
                max_history_len: None,
            }),
            cut_pools: Vec::new(),
            trace: Vec::new(),
            last_sample_path: Vec::new(),
            p,
            rng: SeededRandom::new(opts.seed),
            exact_objective: opts.exact_objective,
            evaluate_policy_every: opts.evaluate_policy_every,
            finite_diff_step: opts.finite_diff_step,
            cut_grid_size: opts.cut_grid_size,
            final_status: SDDPStatus::IterLimit,
        };
        station.initialise_cut_pools();
        station.bootstrap();

        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sddp.cut-pools-nonempty",
                |s: &dyn DESStation| {
                    downcast_sddp(s)
                        .cut_pools
                        .iter()
                        .all(|pool| pool.size() >= 1)
                },
                Some("every stage has at least one affine cut".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    downcast_sddp(s)
                        .cut_pools
                        .iter()
                        .map(|pool| pool.size().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })),
                Some("sddp-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "sddp.upper-bound-above-exact",
                |s: &dyn DESStation| {
                    let st = downcast_sddp(s);
                    match st.exact_objective {
                        None => true,
                        Some(exact) => st.current().upper_bound + 1e-5 >= exact,
                    }
                },
                Some("SDDP upper approximation >= exact objective".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_sddp(s);
                    format!(
                        "upper={}, exact={:?}",
                        st.current().upper_bound,
                        st.exact_objective
                    )
                })),
                Some("sddp-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );

        station
    }

    pub fn get_status(&self) -> SDDPStatus {
        self.final_status
    }

    fn initialise_cut_pools(&mut self) {
        let horizon = self.p.horizon;
        let mut remaining_revenue_upper = vec![0.0; horizon + 1];
        remaining_revenue_upper[horizon] = self.p.salvage_value * self.p.capacity;
        for t in (0..horizon).rev() {
            let max_demand = self.p.demands[t]
                .iter()
                .map(|d| d.demand)
                .fold(f64::NEG_INFINITY, f64::max);
            remaining_revenue_upper[t] =
                remaining_revenue_upper[t + 1] + self.p.price[t] * max_demand;
        }
        for t in 0..=horizon {
            let mut pool = AffineCutPool::new(1, CutEnvelopeSense::Upper, &[]).expect("cut pool");
            if t == horizon {
                pool.add(AffineCut {
                    alpha: 0.0,
                    beta: vec![self.p.salvage_value],
                    source: Some("terminal-salvage".to_string()),
                })
                .expect("terminal cut");
            } else {
                pool.add(AffineCut {
                    alpha: remaining_revenue_upper[t],
                    beta: vec![0.0],
                    source: Some("initial-constant-upper".to_string()),
                })
                .expect("initial cut");
            }
            self.cut_pools.push(pool);
        }
    }
}

impl DESStation for SDDPStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.fixed_point_has_work()
    }
    fn run_time_step(&mut self) {
        self.fixed_point_run_time_step();
    }
}

impl FixedPointIterationStation<SDDPState> for SDDPStation {
    fn fp_core(&self) -> &FixedPointCore<SDDPState> {
        &self.fp
    }
    fn fp_core_mut(&mut self) -> &mut FixedPointCore<SDDPState> {
        &mut self.fp
    }

    fn initial_state(&self) -> SDDPState {
        let ub = self.cut_pools[0]
            .evaluate(&[self.p.initial_inventory])
            .expect("evaluate");
        SDDPState {
            iter: 0,
            upper_bound: ub,
            policy_value: None,
            gap_to_exact: None,
        }
    }

    fn apply_operator(&mut self, prev: &SDDPState) -> SDDPState {
        let iter = prev.iter + 1;
        let horizon = self.p.horizon;
        let mut states = vec![0.0; horizon + 1];
        let mut sampled_demands: Vec<f64> = Vec::new();
        let mut path: Vec<SamplePathEntry> = Vec::new();
        states[0] = self.p.initial_inventory;

        for t in 0..horizon {
            let demand = sample_demand(&self.p.demands[t], &mut self.rng);
            sampled_demands.push(demand);
            let dec = solve_stage_decision(&self.p, t, states[t], demand, &self.cut_pools[t + 1]);
            if dec.status != LPStatus::Optimal {
                panic!(
                    "{MODEL}: forward LP failed at stage {t}: {}",
                    dec.status.as_str()
                );
            }
            path.push(SamplePathEntry {
                stage: t,
                demand,
                state: states[t],
                order: dec.order,
                sell: dec.sell,
                stockout: dec.stockout,
                next_inventory: dec.next_inventory,
            });
            states[t + 1] = clamp(dec.next_inventory, 0.0, self.p.capacity);
        }
        self.last_sample_path = path;

        let mut cuts_added: Vec<CutAddedTrace> = Vec::new();
        for t in (0..horizon).rev() {
            let cut = generate_value_cut(
                &self.p,
                t,
                states[t],
                &self.cut_pools[t + 1],
                CutOpts {
                    finite_diff_step: self.finite_diff_step,
                    cut_grid_size: self.cut_grid_size,
                },
                &format!("iter={iter} stage={t}"),
            );
            let (alpha, beta0) = (cut.alpha, cut.beta[0]);
            self.cut_pools[t].add(cut).expect("add cut");
            cuts_added.push(CutAddedTrace {
                stage: t,
                alpha,
                beta: beta0,
                state: states[t],
            });
        }

        let upper_bound = self.cut_pools[0]
            .evaluate(&[self.p.initial_inventory])
            .expect("evaluate");
        let mut policy_value: Option<f64> = None;
        let mut gap_to_exact: Option<f64> = None;
        if iter.is_multiple_of(self.evaluate_policy_every)
            || iter >= self.fp.max_iter
            || self.exact_objective.is_some()
        {
            let pv = evaluate_policy_exact(&self.p, &self.cut_pools);
            policy_value = Some(pv);
            gap_to_exact = self.exact_objective.map(|exact| exact - pv);
        }
        self.trace.push(SDDPIterationTrace {
            iter,
            sampled_demands,
            states: states.clone(),
            terminal_inventory: states[horizon],
            cuts_added,
            upper_bound,
            policy_value,
            gap_to_exact,
        });
        if let Some(gap) = gap_to_exact {
            if gap.abs() <= self.fp.tol {
                self.final_status = SDDPStatus::Optimal;
            }
        }
        SDDPState {
            iter,
            upper_bound,
            policy_value,
            gap_to_exact,
        }
    }

    fn delta(&self, prev: &SDDPState, next: &SDDPState) -> f64 {
        match self.exact_objective {
            Some(exact) => (next.upper_bound - exact).abs(),
            None => (prev.upper_bound - next.upper_bound).abs(),
        }
    }

    fn should_stop(&mut self, iter: usize, _last_delta: f64) -> bool {
        if self.final_status == SDDPStatus::Optimal && iter > 0 {
            self.fp_core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        if iter >= self.fp_core().max_iter {
            // finalStatus stays 'optimal' if already reached, else 'iter-limit'.
            self.fp_core_mut().convergence_reason = if self.final_status == SDDPStatus::Optimal {
                ConvergenceReason::Converged
            } else {
                ConvergenceReason::MaxIter
            };
            return true;
        }
        false
    }
}

/// Solve the multi-stage inventory model via SDDP.
pub fn solve_multi_stage_sddp(p: MultiStageInventoryProblem, opts: SDDPOptions) -> SDDPResult {
    validate_multi_stage_problem(&p);
    let filled = SDDPFilledOptions {
        max_iter: opts.max_iter.unwrap_or(80),
        tol: opts.tol.unwrap_or(1e-4),
        seed: opts.seed.unwrap_or(1),
        exact_objective: opts.exact_objective,
        evaluate_policy_every: opts.evaluate_policy_every.unwrap_or(usize::MAX),
        finite_diff_step: opts
            .finite_diff_step
            .unwrap_or_else(|| 1e-4_f64.max(p.capacity * 1e-5)),
        cut_grid_size: opts.cut_grid_size.unwrap_or(21),
    };
    let station = Rc::new(RefCell::new(SDDPStation::new(p.clone(), &filled)));
    run_iterative_des(
        vec![station.clone() as Rc<RefCell<dyn DESStation>>],
        IterativeRunOptions::default(),
    );

    let st = station.borrow();
    let policy_value = evaluate_policy_exact(&p, &st.cut_pools);
    let current = st.current();
    let exact = filled.exact_objective;
    SDDPResult {
        status: st.get_status(),
        iterations: st.iteration(),
        upper_bound: current.upper_bound,
        policy_value,
        exact_objective: exact,
        gap_to_exact: exact.map(|e| e - policy_value),
        cuts_per_stage: st.cut_pools.iter().map(|pool| pool.size()).collect(),
        cuts: st.cut_pools.iter().map(|pool| pool.all()).collect(),
        trace: st.trace.clone(),
        sample_path: st.last_sample_path.clone(),
    }
}

/// Run the exact extensive-form solve and the SDDP approximation together.
pub fn run_multi_stage_inventory_demo(
    p: MultiStageInventoryProblem,
    opts: SDDPOptions,
) -> MultiStageRunResult {
    let exact = solve_exact_scenario_tree(&p);
    let mut sddp_opts = opts;
    sddp_opts.exact_objective = Some(exact.objective);
    let sddp = solve_multi_stage_sddp(p, sddp_opts);
    MultiStageRunResult { exact, sddp }
}

// -----------------------------------------------------------------------------
// Exact extensive-form scenario tree LP
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct TreeNode {
    id: usize,
    stage: usize,
    demand: f64,
    prob: f64,
    parent_id: Option<usize>,
}

/// Solve the exact extensive-form scenario-tree LP. Single input: the problem.
pub struct ExactScenarioTreeSolver;

impl Transform<&MultiStageInventoryProblem, ExactTreeNodeResult> for ExactScenarioTreeSolver {
    fn transform(&self, p: &MultiStageInventoryProblem) -> ExactTreeNodeResult {
        solve_exact_scenario_tree_impl(p)
    }
}

/// Deprecated shim: prefer `ExactScenarioTreeSolver.transform(p)`.
pub fn solve_exact_scenario_tree(p: &MultiStageInventoryProblem) -> ExactTreeNodeResult {
    ExactScenarioTreeSolver.transform(p)
}

fn solve_exact_scenario_tree_impl(p: &MultiStageInventoryProblem) -> ExactTreeNodeResult {
    validate_multi_stage_problem(p);
    let nodes = build_scenario_tree(p);
    let var_count = nodes.len() * 4; // order, sell, stockout, nextInventory per node
    let idx = |node_id: usize, local: usize| -> usize { node_id * 4 + local };
    let mut c = vec![0.0; var_count];
    let mut a_ub: Vec<Vec<f64>> = Vec::new();
    let mut b_ub: Vec<f64> = Vec::new();
    let mut a_eq: Vec<Vec<f64>> = Vec::new();
    let mut b_eq: Vec<f64> = Vec::new();

    for node in &nodes {
        let t = node.stage;
        c[idx(node.id, 0)] += node.prob * -p.order_cost[t];
        c[idx(node.id, 1)] += node.prob * p.price[t];
        c[idx(node.id, 2)] += node.prob * -p.stockout_cost[t];
        c[idx(node.id, 3)] += node.prob * -p.hold_cost[t];
        if t == p.horizon - 1 {
            c[idx(node.id, 3)] += node.prob * p.salvage_value;
        }

        let mut bal = vec![0.0; var_count];
        bal[idx(node.id, 0)] = -1.0;
        bal[idx(node.id, 1)] = 1.0;
        bal[idx(node.id, 3)] = 1.0;
        match node.parent_id {
            None => {
                a_eq.push(bal);
                b_eq.push(p.initial_inventory);
            }
            Some(pid) => {
                bal[idx(pid, 3)] = -1.0;
                a_eq.push(bal);
                b_eq.push(0.0);
            }
        }

        let mut demand_row = vec![0.0; var_count];
        demand_row[idx(node.id, 1)] = 1.0;
        demand_row[idx(node.id, 2)] = 1.0;
        a_eq.push(demand_row);
        b_eq.push(node.demand);

        let mut order_bound = vec![0.0; var_count];
        order_bound[idx(node.id, 0)] = 1.0;
        a_ub.push(order_bound);
        b_ub.push(p.max_order[t]);

        let mut inv_bound = vec![0.0; var_count];
        inv_bound[idx(node.id, 3)] = 1.0;
        a_ub.push(inv_bound);
        b_ub.push(p.capacity);
    }

    let lp_rows = a_ub.len() + a_eq.len();
    let lp = LPProblem {
        sense: lp::Sense::Max,
        c,
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: Some(a_eq),
        b_eq: Some(b_eq),
        lb: Some(vec![Some(0.0); var_count]),
        ..Default::default()
    };
    let sol = solve_lp_internal(
        &lp,
        &InternalSimplexOptions {
            max_iter: Some(100000),
            tol: None,
            basis_start: None,
        },
    );
    ExactTreeNodeResult {
        objective: sol.objective,
        node_count: nodes.len(),
        lp_vars: var_count,
        lp_rows,
        status: sol.status.as_str().to_string(),
    }
}

fn build_scenario_tree(p: &MultiStageInventoryProblem) -> Vec<TreeNode> {
    struct FrontierEntry {
        parent_id: Option<usize>,
        prob: f64,
    }
    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut frontier: Vec<FrontierEntry> = vec![FrontierEntry {
        parent_id: None,
        prob: 1.0,
    }];
    for t in 0..p.horizon {
        let mut next: Vec<FrontierEntry> = Vec::new();
        for parent in &frontier {
            for d in &p.demands[t] {
                let id = nodes.len();
                nodes.push(TreeNode {
                    id,
                    stage: t,
                    demand: d.demand,
                    prob: parent.prob * d.prob,
                    parent_id: parent.parent_id,
                });
                next.push(FrontierEntry {
                    parent_id: Some(id),
                    prob: parent.prob * d.prob,
                });
            }
        }
        frontier = next;
    }
    nodes
}

// -----------------------------------------------------------------------------
// Policy evaluation
// -----------------------------------------------------------------------------

/// Evaluate a cut-pool policy exactly by recursion over the scenario tree. The
/// problem is configuration; the per-stage cut pools are the `transform` input.
pub struct ExactPolicyEvaluator<'p> {
    p: &'p MultiStageInventoryProblem,
}

impl<'p> ExactPolicyEvaluator<'p> {
    pub fn new(p: &'p MultiStageInventoryProblem) -> Self {
        ExactPolicyEvaluator { p }
    }
}

impl<'p, 'a> Transform<&'a [AffineCutPool], f64> for ExactPolicyEvaluator<'p> {
    fn transform(&self, cut_pools: &'a [AffineCutPool]) -> f64 {
        let p = self.p;
        validate_multi_stage_problem(p);
        require(Preconditions::length_eq(
            MODEL,
            "cutPools",
            cut_pools,
            p.horizon + 1,
        ));
        policy_rec(p, cut_pools, 0, p.initial_inventory)
    }
}

fn policy_rec(
    p: &MultiStageInventoryProblem,
    cut_pools: &[AffineCutPool],
    stage: usize,
    state: f64,
) -> f64 {
    if stage >= p.horizon {
        return p.salvage_value * state;
    }
    let mut z = 0.0;
    for d in &p.demands[stage] {
        let dec = solve_stage_decision(p, stage, state, d.demand, &cut_pools[stage + 1]);
        if dec.status != LPStatus::Optimal {
            panic!(
                "{MODEL}: policy eval LP failed at stage {stage}: {}",
                dec.status.as_str()
            );
        }
        z += d.prob
            * (dec.immediate_reward
                + policy_rec(
                    p,
                    cut_pools,
                    stage + 1,
                    clamp(dec.next_inventory, 0.0, p.capacity),
                ));
    }
    z
}

/// Deprecated shim: prefer `ExactPolicyEvaluator::new(p).transform(cut_pools)`.
pub fn evaluate_policy_exact(p: &MultiStageInventoryProblem, cut_pools: &[AffineCutPool]) -> f64 {
    ExactPolicyEvaluator::new(p).transform(cut_pools)
}

fn sample_demand(outcomes: &[DemandOutcome], rng: &mut dyn RandomSource) -> f64 {
    let u = rng.next_float();
    let mut acc = 0.0;
    for o in outcomes {
        acc += o.prob;
        if u <= acc {
            return o.demand;
        }
    }
    outcomes[outcomes.len() - 1].demand
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

#[cfg(test)]
mod tests {
    //! The SDDP upper bound stays above the exact extensive-form optimum and the
    //! SDDP policy value converges to it on the default 4-stage problem.

    use super::*;

    #[test]
    fn exact_scenario_tree_is_feasible() {
        let p = build_default_multi_stage_inventory_problem();
        let exact = solve_exact_scenario_tree(&p);
        assert_eq!(exact.status, "optimal");
        assert!(exact.node_count > 0);
        assert!(exact.objective.is_finite());
    }

    #[test]
    fn sddp_upper_bound_dominates_and_policy_converges() {
        let p = build_default_multi_stage_inventory_problem();
        let exact = solve_exact_scenario_tree(&p);
        let res = solve_multi_stage_sddp(
            p,
            SDDPOptions {
                max_iter: Some(40),
                seed: Some(7),
                exact_objective: Some(exact.objective),
                ..Default::default()
            },
        );
        // SDDP's outer (upper) approximation is an upper bound on the optimum.
        assert!(
            res.upper_bound + 1e-4 >= exact.objective,
            "ub={} exact={}",
            res.upper_bound,
            exact.objective
        );
        // The recovered policy value is a lower bound; both bracket the optimum.
        assert!(
            res.policy_value <= exact.objective + 1e-4,
            "policy={} exact={}",
            res.policy_value,
            exact.objective
        );
        // After enough iterations the gap should be small.
        assert!(
            (exact.objective - res.policy_value).abs() < 1.0,
            "gap={}",
            exact.objective - res.policy_value
        );
        for &size in &res.cuts_per_stage {
            assert!(size >= 1);
        }
    }

    #[test]
    fn validate_rejects_bad_horizon() {
        let mut p = build_default_multi_stage_inventory_problem();
        p.price.pop(); // length mismatch vs horizon
        let caught = std::panic::catch_unwind(|| validate_multi_stage_problem(&p)).is_err();
        assert!(caught, "expected a precondition panic");
    }
}
