//! Port of `src/des/general/adapters/mdp-adjacent-adapters.ts`
//! (module `des::general::adapters::mdp_adjacent_adapters`).
//!
//! JSON adapters for nine MDP-adjacent models: `inventory-dp`,
//! `mountain-car-vfa`, `tiger-pomdp`, `grid-localization-pomdp`,
//! `four-rooms-smdp`, `actor-critic-grid`, `blackjack-mc`, `stag-hunt`,
//! `double-integrator-lqr`. Each follows the [`DESModelRegistration`] contract.
//!
//! ## Conversion notes
//!
//!   * `solver: 'qmdp'|'one-step-lookahead'` literal union → the engine's
//!     [`TigerSolver`] enum.
//!   * Intersection-typed `P` shapes (`InventoryProblem & {seed?}`,
//!     `TigerOpts & {solver; numSteps; seed}`) have no Rust analogue and are
//!     flattened into the [`InventoryDpParams`] / [`TigerPomdpParams`] structs.
//!   * `x0` / `hiddenTarget` fixed-length arrays are already `[f64;2]` /
//!     `(usize,usize)` on the engine types, so the TS `numberPair` coercion is a
//!     no-op and is dropped (see the LQR adapter).
//!   * `run_double_integrator_lqr` returns `Result` (the engine's
//!     `Preconditions` are recoverable); the TS `run` threw, so the adapter
//!     `.expect()`s — an invariant violation → `panic!`.
//!
//! PORT NOTE: `registerModel` / the `des_registry` is not wired here (the
//! registry stores type-erased adapters); each model is exposed via an
//! `adapter_*()` constructor returning the registration struct, matching the
//! sibling `signal_transforms_adapter` port.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESOutputs, DESRuntimeConfig, ParamSchema,
    RegistrationExample, DES_MODEL_SPEC_SCHEMA,
};

use crate::des::general::actor_critic_gridworld::{
    run_actor_critic_gridworld, ActorCriticResult, ActorCriticTrainOpts,
};
use crate::des::general::blackjack::{run_blackjack_mc, BlackjackResult, BlackjackTrainOpts};
use crate::des::general::double_integrator_lqr::{
    run_double_integrator_lqr, DoubleIntegratorOpts, DoubleIntegratorResult,
};
use crate::des::general::four_rooms::{run_four_rooms_smdp, FourRoomsResult, FourRoomsTrainOpts};
use crate::des::general::grid_localization_pomdp::{
    run_grid_localization_pomdp, GridLocalizationObservation, GridLocalizationParams,
    GridLocalizationResult,
};
use crate::des::general::inventory_dp::{solve_inventory_dp, InventoryDPResult, InventoryProblem};
use crate::des::general::mountain_car::{
    run_mountain_car, MountainCarResult, MountainCarTrainOpts,
};
use crate::des::general::stag_hunt::{run_stag_hunt, StagHuntOpts, StagHuntResult};
use crate::des::general::tiger_pomdp::{
    build_tiger_spec, simulate_tiger, TigerOpts, TigerSimOpts, TigerSimResult, TigerSolver,
};

// =============================================================================
// Schema builders (JS-parity, copied from the sibling adapter style).
// =============================================================================

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: None,
    }
}

fn boolean(default: Option<bool>) -> ParamSchema {
    ParamSchema::Boolean {
        default,
        description: None,
    }
}

fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length,
        max_length,
        description: None,
    }
}

fn obj(description: &str, fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: Some(description.to_string()),
    }
}

// =============================================================================
// Shared helpers.
// =============================================================================

/// `meanLast(xs, n)` — mean of the last `min(n, len)` elements.
fn mean_last(xs: &[f64], n: usize) -> f64 {
    let take = n.min(xs.len());
    let slice = &xs[xs.len() - take..];
    slice.iter().sum::<f64>() / slice.len().max(1) as f64
}

/// `countActions(actions, names)`.
fn count_actions(actions: &[usize], names: &[&str]) -> String {
    let mut counts = vec![0usize; names.len()];
    for &a in actions {
        if a < counts.len() {
            counts[a] += 1;
        }
    }
    names
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n}={}", counts[i]))
        .collect::<Vec<_>>()
        .join(", ")
}

fn obs_str(o: GridLocalizationObservation) -> &'static str {
    match o {
        GridLocalizationObservation::No => "no",
        GridLocalizationObservation::Yes => "yes",
    }
}

fn solver_str(s: TigerSolver) -> &'static str {
    match s {
        TigerSolver::Qmdp => "qmdp",
        TigerSolver::OneStepLookahead => "one-step-lookahead",
    }
}

// =============================================================================
// 1. inventory-dp
// =============================================================================

/// `InventoryProblem & {seed?: number}`.
#[derive(Clone, Debug)]
pub struct InventoryDpParams {
    pub problem: InventoryProblem,
    pub seed: Option<u32>,
}

fn inventory_schema() -> ParamSchema {
    obj(
        "Multi-period stochastic inventory by finite-horizon DP.",
        vec![
            ("horizon", num(Some(1.0), None, Some(true), None)),
            ("S_max", num(Some(0.0), None, Some(true), None)),
            (
                "demandPmf",
                arr(num(Some(0.0), Some(1.0), None, None), None, None),
            ),
            ("price", num(Some(0.0), None, None, None)),
            ("cost", num(Some(0.0), None, None, None)),
            ("fixedCost", num(Some(0.0), None, None, Some(0.0))),
            ("holdCost", num(Some(0.0), None, None, Some(0.5))),
            ("stockoutCost", num(Some(0.0), None, None, Some(5.0))),
            ("salvageValue", num(None, None, None, Some(0.0))),
            ("discount", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("initialInventory", num(Some(0.0), None, Some(true), None)),
        ],
        vec![
            "horizon",
            "S_max",
            "demandPmf",
            "price",
            "cost",
            "initialInventory",
        ],
    )
}

pub struct InventoryDpAdapter;
pub fn adapter_inventory_dp() -> InventoryDpAdapter {
    InventoryDpAdapter
}

impl DESModelRegistration<InventoryDpParams, InventoryDPResult> for InventoryDpAdapter {
    fn id(&self) -> &str {
        "inventory-dp"
    }
    fn description(&self) -> &str {
        "Multi-period stochastic inventory solved by finite-horizon DP (backward induction)."
    }
    fn schema(&self) -> ParamSchema {
        inventory_schema()
    }
    fn run(&self, params: InventoryDpParams, _runtime: &DESRuntimeConfig) -> InventoryDPResult {
        solve_inventory_dp(&params.problem, Some(params.seed.unwrap_or(1)))
    }
    fn summarize(&self, r: &InventoryDPResult, p: &InventoryDpParams) -> String {
        let seed = p.seed.unwrap_or(1);
        [
            "INVENTORY DP".to_string(),
            "────────────────────────────".to_string(),
            format!("  Horizon:      {}", p.problem.horizon),
            format!("  S_max:        {}", p.problem.s_max),
            format!("  E[demand]:    {:.2}", r.mean_demand),
            format!(
                "  V*(t=0, s={}): {:.3}",
                p.problem.initial_inventory, r.expected_reward
            ),
            format!(
                "  Sim total reward (seed={seed}): {:.3}",
                r.simulation.total_reward
            ),
            format!("  Orders:       {}", join_usize(&r.simulation.orders)),
            format!("  Demands:      {}", join_usize(&r.simulation.demands)),
            format!("  Inventory:    {}", join_usize(&r.simulation.inventory)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &InventoryDPResult, csv_path: &str) {
        let mut lines = vec!["t,inventory,order,demand,reward".to_string()];
        for t in 0..r.simulation.orders.len() {
            lines.push(csv_row([
                t.to_string(),
                r.simulation.inventory[t].to_string(),
                r.simulation.orders[t].to_string(),
                r.simulation.demands[t].to_string(),
                format!("{:.4}", r.simulation.rewards[t]),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

fn join_usize(xs: &[usize]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// =============================================================================
// 2. mountain-car-vfa
// =============================================================================

fn mountain_car_schema() -> ParamSchema {
    obj(
        "Mountain Car solved by linear VFA with tile coding (Sutton-Albus).",
        vec![
            ("numEpisodes", num(Some(1.0), None, Some(true), Some(200.0))),
            (
                "maxStepsPerEpisode",
                num(Some(1.0), None, Some(true), Some(1000.0)),
            ),
            ("alpha", num(Some(0.0), None, None, Some(0.5))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.0))),
            ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("numTilings", num(Some(1.0), None, Some(true), Some(8.0))),
            (
                "numTilesPerDim",
                num(Some(2.0), None, Some(true), Some(8.0)),
            ),
        ],
        vec!["numEpisodes"],
    )
}

pub struct MountainCarAdapter;
pub fn adapter_mountain_car() -> MountainCarAdapter {
    MountainCarAdapter
}

impl DESModelRegistration<MountainCarTrainOpts, MountainCarResult> for MountainCarAdapter {
    fn id(&self) -> &str {
        "mountain-car-vfa"
    }
    fn description(&self) -> &str {
        "Mountain Car (continuous control) by linear VFA with Sutton-Albus tile coding."
    }
    fn schema(&self) -> ParamSchema {
        mountain_car_schema()
    }
    fn run(&self, params: MountainCarTrainOpts, _runtime: &DESRuntimeConfig) -> MountainCarResult {
        run_mountain_car(params)
    }
    fn summarize(&self, r: &MountainCarResult, _p: &MountainCarTrainOpts) -> String {
        let greedy = if r.greedy_solves {
            format!("solves in {} steps", r.greedy_episode_length)
        } else {
            format!("does NOT solve in {} steps", r.greedy_episode_length)
        };
        [
            "MOUNTAIN CAR (linear VFA + tile coding)".to_string(),
            "─────────────────────────────────────────".to_string(),
            format!("  Episodes:               {}", r.reward_history.len()),
            format!(
                "  Mean return (last 20):  {:.2}",
                mean_last(&r.reward_history, 20)
            ),
            format!(
                "  Mean length (last 20):  {:.1}",
                mean_last(&r.length_history, 20)
            ),
            format!(
                "  Mean |TD-err| (last 20):{:.4}",
                mean_last(&r.td_error_history, 20)
            ),
            format!("  Greedy from x=-0.5:     {greedy}"),
            format!("  ‖θ‖₂:                   {:.2}", r.theta_norm),
        ]
        .join("\n")
    }
}

// =============================================================================
// 3. tiger-pomdp
// =============================================================================

/// `TigerOpts & {solver; numSteps; seed}`.
#[derive(Clone, Debug)]
pub struct TigerPomdpParams {
    pub opts: TigerOpts,
    pub solver: TigerSolver,
    pub num_steps: usize,
    pub seed: Option<u32>,
}

fn tiger_schema() -> ParamSchema {
    obj(
        "Cassandra-Kaelbling-Littman 1994 Tiger problem with QMDP / 1-step look-ahead.",
        vec![
            (
                "listenAccuracy",
                num(Some(0.5), Some(1.0), None, Some(0.85)),
            ),
            ("openGood", num(None, None, None, Some(10.0))),
            ("openBad", num(None, None, None, Some(-100.0))),
            ("listenCost", num(None, None, None, Some(-1.0))),
            ("discount", num(Some(0.0), Some(1.0), None, Some(0.95))),
            (
                "solver",
                str_enum(&["qmdp", "one-step-lookahead"], "one-step-lookahead"),
            ),
            ("numSteps", num(Some(1.0), None, Some(true), Some(50.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
        ],
        vec![],
    )
}

pub struct TigerPomdpAdapter;
pub fn adapter_tiger_pomdp() -> TigerPomdpAdapter {
    TigerPomdpAdapter
}

impl DESModelRegistration<TigerPomdpParams, TigerSimResult> for TigerPomdpAdapter {
    fn id(&self) -> &str {
        "tiger-pomdp"
    }
    fn description(&self) -> &str {
        "Tiger POMDP — belief-state planning under partial observability."
    }
    fn schema(&self) -> ParamSchema {
        tiger_schema()
    }
    fn run(&self, params: TigerPomdpParams, _runtime: &DESRuntimeConfig) -> TigerSimResult {
        simulate_tiger(TigerSimOpts {
            spec: Some(build_tiger_spec(&params.opts)),
            solver: params.solver,
            num_steps: params.num_steps,
            seed: params.seed,
            initial_state: None,
            initial_belief: None,
        })
    }
    fn summarize(&self, r: &TigerSimResult, p: &TigerPomdpParams) -> String {
        let action_names = ["LISTEN", "OPEN-LEFT", "OPEN-RIGHT"];
        [
            "TIGER POMDP".to_string(),
            "───────────────────────────────".to_string(),
            format!("  Solver:       {}", solver_str(p.solver)),
            format!("  Steps:        {}", r.steps),
            format!("  Total return: {:.2}", r.total_return),
            format!("  # opens:      {}", r.num_opens),
            format!(
                "  # bad opens:  {}  (catastrophic open of tiger door)",
                r.num_bad_opens
            ),
            format!(
                "  Action mix:   {}",
                count_actions(&r.actions, &action_names)
            ),
        ]
        .join("\n")
    }
}

// =============================================================================
// 4. grid-localization-pomdp
// =============================================================================

fn grid_localization_schema() -> ParamSchema {
    obj(
        "2D hidden-target localization POMDP with row/column scans and inspect actions.",
        vec![
            ("width", num(Some(2.0), Some(8.0), Some(true), Some(3.0))),
            ("height", num(Some(2.0), Some(8.0), Some(true), Some(3.0))),
            ("horizon", num(Some(0.0), Some(6.0), Some(true), Some(3.0))),
            (
                "numSteps",
                num(Some(1.0), Some(100.0), Some(true), Some(8.0)),
            ),
            ("seed", num(None, None, Some(true), Some(1.0))),
            (
                "hiddenTarget",
                arr(num(None, None, Some(true), None), Some(2), Some(2)),
            ),
            (
                "initialBelief",
                arr(num(Some(0.0), Some(1.0), None, None), None, None),
            ),
            ("scanAccuracy", num(Some(0.5), Some(1.0), None, Some(0.9))),
            (
                "inspectAccuracy",
                num(Some(0.5), Some(1.0), None, Some(0.99)),
            ),
            ("scanCost", num(None, None, None, Some(-0.2))),
            ("inspectCorrectReward", num(None, None, None, Some(20.0))),
            ("inspectWrongPenalty", num(None, None, None, Some(-12.0))),
            ("discount", num(Some(0.0), Some(1.0), None, Some(0.95))),
        ],
        vec![],
    )
}

/// `normalizeGridLocalizationParams(p)` — drop empty/malformed optional inputs.
fn normalize_grid_localization_params(p: GridLocalizationParams) -> GridLocalizationParams {
    let hidden_target = p.hidden_target; // already (usize, usize) — length is fixed.
    let initial_belief = match &p.initial_belief {
        Some(b) if !b.is_empty() => Some(b.clone()),
        _ => None,
    };
    GridLocalizationParams {
        hidden_target,
        initial_belief,
        ..p
    }
}

pub struct GridLocalizationAdapter;
pub fn adapter_grid_localization() -> GridLocalizationAdapter {
    GridLocalizationAdapter
}

impl DESModelRegistration<GridLocalizationParams, GridLocalizationResult>
    for GridLocalizationAdapter
{
    fn id(&self) -> &str {
        "grid-localization-pomdp"
    }
    fn description(&self) -> &str {
        "Multi-dimensional POMDP: localize a hidden target on a 2D grid by belief lookahead."
    }
    fn schema(&self) -> ParamSchema {
        grid_localization_schema()
    }
    fn run(
        &self,
        params: GridLocalizationParams,
        _runtime: &DESRuntimeConfig,
    ) -> GridLocalizationResult {
        run_grid_localization_pomdp(&normalize_grid_localization_params(params))
    }
    fn summarize(&self, r: &GridLocalizationResult, _p: &GridLocalizationParams) -> String {
        let first = r.trace.first();
        let last = r.trace.last();
        let first_action = match first {
            Some(row) => format!("{} -> {}", row.action.label, obs_str(row.observation)),
            None => "n/a".to_string(),
        };
        let found = if r.found {
            format!(
                "YES at step {}",
                r.found_at_step.map(|s| s.to_string()).unwrap_or_default()
            )
        } else {
            "no".to_string()
        };
        let entropy = format!(
            "{} -> {}",
            first
                .map(|row| format!("{:.3}", row.entropy))
                .unwrap_or_else(|| "n/a".to_string()),
            last.map(|row| format!("{:.3}", row.entropy))
                .unwrap_or_else(|| "n/a".to_string())
        );
        let p_hidden = last
            .map(|row| format!("{:.3}", row.hidden_probability))
            .unwrap_or_else(|| "n/a".to_string());
        [
            "GRID LOCALIZATION POMDP".to_string(),
            "───────────────────────────────".to_string(),
            format!(
                "  State space:    {} x {} = {} hidden cells",
                r.params.width, r.params.height, r.state_space.num_states
            ),
            format!(
                "  Planner:        belief lookahead horizon={}",
                r.params.horizon
            ),
            format!(
                "  Hidden target:  ({}, {})",
                r.params.hidden_target.0, r.params.hidden_target.1
            ),
            format!("  First action:   {first_action}"),
            format!("  Found target:   {found}"),
            format!("  Entropy:        {entropy}"),
            format!("  P(hidden):      {p_hidden}"),
            format!("  Total return:   {:.3}", r.total_return),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &GridLocalizationResult, csv_path: &str) {
        let mut lines = vec![
            "step,action,observation,mode_x,mode_y,mode_probability,hidden_probability,entropy,found"
                .to_string(),
        ];
        for row in &r.trace {
            lines.push(csv_row([
                row.step.to_string(),
                row.action.label.clone(),
                obs_str(row.observation).to_string(),
                row.mode.0.to_string(),
                row.mode.1.to_string(),
                row.mode_probability.to_string(),
                row.hidden_probability.to_string(),
                row.entropy.to_string(),
                row.found.to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn examples(&self) -> Vec<RegistrationExample<GridLocalizationParams>> {
        vec![RegistrationExample {
            name: "3x3 hidden target search".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "grid-localization-pomdp".to_string(),
                description: None,
                parameters: GridLocalizationParams {
                    width: 3,
                    height: 3,
                    horizon: Some(3),
                    num_steps: Some(8),
                    seed: Some(7),
                    hidden_target: Some((2, 1)),
                    initial_belief: None,
                    scan_accuracy: Some(1.0),
                    inspect_accuracy: Some(1.0),
                    scan_cost: None,
                    inspect_correct_reward: None,
                    inspect_wrong_penalty: None,
                    discount: None,
                },
                runtime: Some(DESRuntimeConfig {
                    outputs: Some(DESOutputs {
                        csv: Some("out/grid-localization-pomdp.csv".to_string()),
                        summary: Some("out/grid-localization-pomdp.summary.json".to_string()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                metadata: None,
            },
        }]
    }
}

// =============================================================================
// 5. four-rooms-smdp
// =============================================================================

fn four_rooms_schema() -> ParamSchema {
    obj(
        "Four-Rooms gridworld with hallway options (Sutton, Precup, Singh 1999).",
        vec![
            ("numEpisodes", num(Some(1.0), None, Some(true), Some(200.0))),
            (
                "maxStepsPerEpisode",
                num(Some(1.0), None, Some(true), Some(5000.0)),
            ),
            ("alpha", num(Some(0.0), None, None, Some(0.25))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(0.99))),
            ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.1))),
            ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.01))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("slip", num(Some(0.0), Some(1.0), None, Some(0.0))),
            ("includePrimitive", boolean(Some(true))),
            ("initQ", num(None, None, None, Some(1.0))),
        ],
        vec![],
    )
}

pub struct FourRoomsAdapter;
pub fn adapter_four_rooms() -> FourRoomsAdapter {
    FourRoomsAdapter
}

impl DESModelRegistration<FourRoomsTrainOpts, FourRoomsResult> for FourRoomsAdapter {
    fn id(&self) -> &str {
        "four-rooms-smdp"
    }
    fn description(&self) -> &str {
        "Four-Rooms with hallway OPTIONS — SMDP Q-learning over temporally extended actions."
    }
    fn schema(&self) -> ParamSchema {
        four_rooms_schema()
    }
    fn run(&self, params: FourRoomsTrainOpts, _runtime: &DESRuntimeConfig) -> FourRoomsResult {
        run_four_rooms_smdp(params)
    }
    fn summarize(&self, r: &FourRoomsResult, _p: &FourRoomsTrainOpts) -> String {
        let mean_l = mean_last(&r.length_history, 20);
        let greedy = if r.greedy_reached_goal {
            format!("YES in {} steps", r.greedy_episode_length)
        } else {
            "no".to_string()
        };
        [
            "FOUR ROOMS (Semi-MDP, options)".to_string(),
            "────────────────────────────────".to_string(),
            format!("  Episodes trained:        {}", r.reward_history.len()),
            format!("  Mean episode len (last20): {:.1}", mean_l),
            format!("  Greedy reaches goal:     {greedy}"),
            "  Optimal-path lower bound: 20 steps (no walls / one path)".to_string(),
        ]
        .join("\n")
    }
}

// =============================================================================
// 6. actor-critic-grid
// =============================================================================

fn actor_critic_schema() -> ParamSchema {
    obj(
        "One-step tabular actor-critic on GridWorld.",
        vec![
            (
                "numEpisodes",
                num(Some(1.0), None, Some(true), Some(1000.0)),
            ),
            (
                "maxStepsPerEpisode",
                num(Some(1.0), None, Some(true), Some(100.0)),
            ),
            ("alphaV", num(Some(0.0), None, None, Some(0.1))),
            ("alphaP", num(Some(0.0), None, None, Some(0.05))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(0.95))),
            ("entropyCoef", num(None, None, None, Some(0.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("width", num(Some(2.0), None, Some(true), Some(4.0))),
            ("height", num(Some(2.0), None, Some(true), Some(4.0))),
        ],
        vec![],
    )
}

pub struct ActorCriticAdapter;
pub fn adapter_actor_critic() -> ActorCriticAdapter {
    ActorCriticAdapter
}

impl DESModelRegistration<ActorCriticTrainOpts, ActorCriticResult> for ActorCriticAdapter {
    fn id(&self) -> &str {
        "actor-critic-grid"
    }
    fn description(&self) -> &str {
        "One-step Actor-Critic with tabular softmax policy + tabular V on GridWorld."
    }
    fn schema(&self) -> ParamSchema {
        actor_critic_schema()
    }
    fn run(&self, params: ActorCriticTrainOpts, _runtime: &DESRuntimeConfig) -> ActorCriticResult {
        run_actor_critic_gridworld(params)
    }
    fn summarize(&self, r: &ActorCriticResult, _p: &ActorCriticTrainOpts) -> String {
        let mean_r = mean_last(&r.reward_history, 20);
        let greedy = if r.greedy_reached {
            format!("reaches goal in {} steps", r.greedy_len)
        } else {
            format!("fails (len={})", r.greedy_len)
        };
        [
            "ACTOR-CRITIC (tabular) on GridWorld".to_string(),
            "─────────────────────────────────────".to_string(),
            format!("  Episodes:                {}", r.reward_history.len()),
            format!("  Mean return (last 20):   {:.2}", mean_r),
            format!("  V(start) (critic):       {:.3}", r.v_start),
            format!("  Greedy from start:       {greedy}"),
        ]
        .join("\n")
    }
}

// =============================================================================
// 7. blackjack-mc
// =============================================================================

fn blackjack_schema() -> ParamSchema {
    obj(
        "Sutton & Barto §5.1 Blackjack with first-visit Monte Carlo control.",
        vec![
            (
                "numEpisodes",
                num(Some(1.0), None, Some(true), Some(50_000.0)),
            ),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.1))),
            ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.05))),
            ("firstVisit", boolean(Some(true))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(1.0))),
            (
                "evalEpisodes",
                num(Some(1.0), None, Some(true), Some(5000.0)),
            ),
        ],
        vec![],
    )
}

pub struct BlackjackAdapter;
pub fn adapter_blackjack() -> BlackjackAdapter {
    BlackjackAdapter
}

impl DESModelRegistration<BlackjackTrainOpts, BlackjackResult> for BlackjackAdapter {
    fn id(&self) -> &str {
        "blackjack-mc"
    }
    fn description(&self) -> &str {
        "Blackjack solved by on-policy first-visit Monte Carlo control (Sutton & Barto §5.1)."
    }
    fn schema(&self) -> ParamSchema {
        blackjack_schema()
    }
    fn run(&self, params: BlackjackTrainOpts, _runtime: &DESRuntimeConfig) -> BlackjackResult {
        run_blackjack_mc(params)
    }
    fn summarize(&self, r: &BlackjackResult, _p: &BlackjackTrainOpts) -> String {
        [
            "BLACKJACK MC".to_string(),
            "──────────────────────".to_string(),
            format!("  Cells visited:          {} / 400", r.visited_cells),
            format!(
                "  Greedy mean return:     {:.3}  (theoretical optimum ≈ -0.04)",
                r.greedy_mean_return
            ),
            format!(
                "  Baseline (stick≥20):    {:.3}  (≈ -0.27)",
                r.baseline_mean_return
            ),
            format!(
                "  Improvement over base:  {:.3}",
                r.greedy_mean_return - r.baseline_mean_return
            ),
        ]
        .join("\n")
    }
}

// =============================================================================
// 8. stag-hunt
// =============================================================================

fn stag_hunt_schema() -> ParamSchema {
    obj(
        "Stag Hunt coordination game with two independent Q-learners (Tan 1993 IQL).",
        vec![
            (
                "numEpisodes",
                num(Some(1.0), None, Some(true), Some(5000.0)),
            ),
            ("alpha", num(Some(0.0), None, None, Some(0.05))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(0.0))),
            ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.2))),
            ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(0.999))),
            ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.01))),
            ("seed", num(None, None, Some(true), Some(1.0))),
        ],
        vec![],
    )
}

pub struct StagHuntAdapter;
pub fn adapter_stag_hunt() -> StagHuntAdapter {
    StagHuntAdapter
}

impl DESModelRegistration<StagHuntOpts, StagHuntResult> for StagHuntAdapter {
    fn id(&self) -> &str {
        "stag-hunt"
    }
    fn description(&self) -> &str {
        "Stag Hunt — two independent Q-learners coordinate on a Nash equilibrium."
    }
    fn schema(&self) -> ParamSchema {
        stag_hunt_schema()
    }
    fn run(&self, params: StagHuntOpts, _runtime: &DESRuntimeConfig) -> StagHuntResult {
        run_stag_hunt(&params)
    }
    fn summarize(&self, r: &StagHuntResult, _p: &StagHuntOpts) -> String {
        let acts = ["STAG", "HARE"];
        [
            "STAG HUNT (Independent Q-Learning, 2 agents)".to_string(),
            "──────────────────────────────────────────────".to_string(),
            format!("  Episodes:               {}", r.reward_history.len()),
            format!(
                "  Recent mean return:     [{:.2}, {:.2}]",
                r.recent_mean_return[0], r.recent_mean_return[1]
            ),
            format!(
                "  Final greedy actions:   [{}, {}]",
                acts[r.final_joint_action[0]], acts[r.final_joint_action[1]]
            ),
            format!(
                "  Coordinated on Stag?    {}",
                if r.coordinated_on_stag {
                    "YES (payoff-dominant)"
                } else {
                    "no"
                }
            ),
            format!(
                "  Coordinated on Hare?    {}",
                if r.coordinated_on_hare {
                    "YES (risk-dominant)"
                } else {
                    "no"
                }
            ),
        ]
        .join("\n")
    }
}

// =============================================================================
// 9. double-integrator-lqr
// =============================================================================

fn lqr_schema() -> ParamSchema {
    obj(
        "Discrete-time LQR on a double integrator, computed by Riccati iteration.",
        vec![
            ("dt", num(Some(1e-6), None, None, Some(0.1))),
            ("qPos", num(Some(0.0), None, None, Some(1.0))),
            ("qVel", num(Some(0.0), None, None, Some(0.1))),
            ("rU", num(Some(1e-6), None, None, Some(0.01))),
            ("noiseStd", num(Some(0.0), None, None, Some(0.05))),
            ("x0", arr(num(None, None, None, None), Some(2), Some(2))),
            ("numSteps", num(Some(1.0), None, Some(true), Some(100.0))),
            ("uSat", num(Some(0.0), None, None, Some(f64::INFINITY))),
            ("gamma", num(Some(0.0), Some(1.0), None, Some(1.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
        ],
        vec![],
    )
}

pub struct DoubleIntegratorLqrAdapter;
pub fn adapter_double_integrator_lqr() -> DoubleIntegratorLqrAdapter {
    DoubleIntegratorLqrAdapter
}

impl DESModelRegistration<DoubleIntegratorOpts, DoubleIntegratorResult>
    for DoubleIntegratorLqrAdapter
{
    fn id(&self) -> &str {
        "double-integrator-lqr"
    }
    fn description(&self) -> &str {
        "Discrete-time LQR feedback control of a double integrator (Riccati DARE)."
    }
    fn schema(&self) -> ParamSchema {
        lqr_schema()
    }
    fn run(
        &self,
        params: DoubleIntegratorOpts,
        _runtime: &DESRuntimeConfig,
    ) -> DoubleIntegratorResult {
        // PORT NOTE: TS `numberPair(p.x0, [3,0], 'x0')` is a no-op here (x0 is
        // `Option<[f64;2]>` and the runner defaults it to `[3,0]`). The runner
        // returns `Result`; the TS `run` threw on failure → `.expect()` (panic).
        run_double_integrator_lqr(params).expect("double-integrator-lqr precondition failed")
    }
    fn summarize(&self, r: &DoubleIntegratorResult, _p: &DoubleIntegratorOpts) -> String {
        let final_state = r.trajectory.last().copied().unwrap_or([0.0, 0.0]);
        let k_row =
            r.k.first()
                .map(|row| {
                    row.iter()
                        .map(|x| format!("{:.3}", x))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
        [
            "DOUBLE-INTEGRATOR LQR".to_string(),
            "────────────────────────────────────".to_string(),
            format!(
                "  Riccati iters:           {}  (residual {:.2e})",
                r.riccati_iters, r.riccati_residual
            ),
            format!("  Optimal feedback K:      [{k_row}]"),
            format!("  Cost-to-go (DARE):       {:.3}", r.riccati_cost_from_x0),
            format!("  Realised cumulative cost {:.3}", r.total_cost),
            format!(
                "  Final state:             [{:.3}, {:.3}]",
                final_state[0], final_state[1]
            ),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &DoubleIntegratorResult, csv_path: &str) {
        let mut lines = vec!["t,pos,vel,u,stage_cost".to_string()];
        for t in 0..r.controls.len() {
            lines.push(csv_row([
                t.to_string(),
                format!("{:.6}", r.trajectory[t][0]),
                format!("{:.6}", r.trajectory[t][1]),
                format!("{:.6}", r.controls[t]),
                format!("{:.6}", r.stage_costs[t]),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_expose_stable_ids() {
        assert_eq!(adapter_inventory_dp().id(), "inventory-dp");
        assert_eq!(adapter_mountain_car().id(), "mountain-car-vfa");
        assert_eq!(adapter_tiger_pomdp().id(), "tiger-pomdp");
        assert_eq!(adapter_grid_localization().id(), "grid-localization-pomdp");
        assert_eq!(adapter_four_rooms().id(), "four-rooms-smdp");
        assert_eq!(adapter_actor_critic().id(), "actor-critic-grid");
        assert_eq!(adapter_blackjack().id(), "blackjack-mc");
        assert_eq!(adapter_stag_hunt().id(), "stag-hunt");
        assert_eq!(
            adapter_double_integrator_lqr().id(),
            "double-integrator-lqr"
        );
    }

    #[test]
    fn count_actions_tallies_by_index() {
        assert_eq!(
            count_actions(&[0, 0, 1, 2, 2, 2], &["A", "B", "C"]),
            "A=2, B=1, C=3"
        );
    }

    #[test]
    fn grid_example_present() {
        let ex = adapter_grid_localization().examples();
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].spec.model, "grid-localization-pomdp");
    }
}
