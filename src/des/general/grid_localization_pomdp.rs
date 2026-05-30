//! Port of `src/des/general/grid-localization-pomdp.ts` — module
//! `des::general::grid_localization_pomdp`.
//!
//! A multi-dimensional POMDP over a hidden target location `(x, y)`. The agent
//! can scan a row, scan a column, or inspect a specific cell. Scans are noisy
//! binary observations; inspecting the right cell ends the episode with a
//! reward. It is small enough for exact belief-tree lookahead while exercising
//! the framework's multi-dimensional state-space utilities.
//!
//! Mapping notes (from the TS "RUST MIGRATION" header):
//!   * `type GridLocalizationActionKind` -> [`GridLocalizationActionKind`] enum;
//!     `type GridLocalizationObservation` -> [`GridLocalizationObservation`]
//!     enum, matched with `match`.
//!   * `interface GridLocalizationParams` / `Action` / `TraceRow` / `Result` /
//!     `Model` -> structs. `Required<Omit<...>> & {hiddenTarget}` ->
//!     [`GridLocalizationResolvedParams`].
//!   * `buildGridLocalizationPOMDP` / `runGridLocalizationPOMDP` -> free fns;
//!     `buildActions` / `validateParams` / `normaliseParams` / `sampleIndex` ->
//!     private fns.
//!   * `sampleIndex(probabilities, rng)` closure RNG -> an injected
//!     [`RandomSource`].
//!   * `validateParams` `throw`s on bad input -> `panic!` via the
//!     `Preconditions` guards (construction-time invariant violations).
//!
//! Builds on [`DiscreteBelief`] + [`CartesianStateSpace`] + the `pomdp`
//! [`BeliefLookaheadSolver`].

use crate::des::general::belief::DiscreteBelief;
use crate::des::general::cartesian_state_space::{CartesianDimension, CartesianStateSpace};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::pomdp::{
    belief_update, BeliefLookaheadLeaf, BeliefLookaheadOptions, BeliefLookaheadSolver, POMDPSpec,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

/// What an action does: scan a row, scan a column, or inspect a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridLocalizationActionKind {
    ScanRow,
    ScanColumn,
    Inspect,
}

/// A binary scan/inspect observation (`'no' | 'yes'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridLocalizationObservation {
    No,
    Yes,
}

/// Observation set, parallel to the spec's `[P(no), P(yes)]` rows.
const OBSERVATIONS: [GridLocalizationObservation; 2] =
    [GridLocalizationObservation::No, GridLocalizationObservation::Yes];

/// User-facing parameters. Absent optionals fall back to the TS defaults.
#[derive(Clone, Debug)]
pub struct GridLocalizationParams {
    pub width: usize,
    pub height: usize,
    pub horizon: Option<usize>,
    pub num_steps: Option<usize>,
    pub seed: Option<u32>,
    pub hidden_target: Option<(usize, usize)>,
    pub initial_belief: Option<Vec<f64>>,
    pub scan_accuracy: Option<f64>,
    pub inspect_accuracy: Option<f64>,
    pub scan_cost: Option<f64>,
    pub inspect_correct_reward: Option<f64>,
    pub inspect_wrong_penalty: Option<f64>,
    pub discount: Option<f64>,
}

impl GridLocalizationParams {
    /// Minimal constructor with only the grid dimensions set.
    pub fn new(width: usize, height: usize) -> Self {
        GridLocalizationParams {
            width,
            height,
            horizon: None,
            num_steps: None,
            seed: None,
            hidden_target: None,
            initial_belief: None,
            scan_accuracy: None,
            inspect_accuracy: None,
            scan_cost: None,
            inspect_correct_reward: None,
            inspect_wrong_penalty: None,
            discount: None,
        }
    }
}

/// One concrete action (`scan row 2`, `inspect (1,0)`, …). `x` / `y` are set
/// only for the dimensions the action constrains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridLocalizationAction {
    pub kind: GridLocalizationActionKind,
    pub index: usize,
    pub x: Option<usize>,
    pub y: Option<usize>,
    pub label: String,
}

/// A per-step trace row.
#[derive(Clone, Debug)]
pub struct GridLocalizationTraceRow {
    pub step: usize,
    pub action: GridLocalizationAction,
    pub observation: GridLocalizationObservation,
    pub hidden_target: (usize, usize),
    pub entropy: f64,
    pub mode: (usize, usize),
    pub mode_probability: f64,
    pub hidden_probability: f64,
    pub found: bool,
}

/// One state-space dimension descriptor in the result.
#[derive(Clone, Debug)]
pub struct DimensionInfo {
    pub name: String,
    pub size: usize,
}

/// State-space summary in the result.
#[derive(Clone, Debug)]
pub struct StateSpaceInfo {
    pub dimensions: Vec<DimensionInfo>,
    pub num_states: usize,
}

/// Fully-resolved parameters (the TS `Required<Omit<...>> & {hiddenTarget}`).
#[derive(Clone, Debug)]
pub struct GridLocalizationResolvedParams {
    pub width: usize,
    pub height: usize,
    pub horizon: usize,
    pub num_steps: usize,
    pub seed: u32,
    pub hidden_target: (usize, usize),
    pub scan_accuracy: f64,
    pub inspect_accuracy: f64,
    pub scan_cost: f64,
    pub inspect_correct_reward: f64,
    pub inspect_wrong_penalty: f64,
    pub discount: f64,
}

/// Run summary.
#[derive(Clone, Debug)]
pub struct GridLocalizationResult {
    pub params: GridLocalizationResolvedParams,
    pub state_space: StateSpaceInfo,
    pub actions: Vec<GridLocalizationAction>,
    pub observations: Vec<GridLocalizationObservation>,
    pub trace: Vec<GridLocalizationTraceRow>,
    pub final_belief: Vec<f64>,
    pub final_entropy: f64,
    pub found: bool,
    pub found_at_step: Option<usize>,
    pub total_return: f64,
}

/// The built model: state space, action set, and the POMDP spec.
pub struct GridLocalizationModel {
    pub space: CartesianStateSpace,
    pub actions: Vec<GridLocalizationAction>,
    pub spec: POMDPSpec<usize, GridLocalizationAction, GridLocalizationObservation>,
}

/// Assemble the POMDP spec from a state space, action set, and (already
/// resolved) parameters. Closures own clones of `space` / `actions` so each is
/// independently `'static` — this lets a fresh spec be built per consumer (the
/// model and the planner each own one, since the planner takes its spec by
/// value).
fn build_grid_spec(
    space: &CartesianStateSpace,
    actions: &[GridLocalizationAction],
    scan_accuracy: f64,
    inspect_accuracy: f64,
    scan_cost: f64,
    inspect_correct_reward: f64,
    inspect_wrong_penalty: f64,
    discount: f64,
    initial_belief: Vec<f64>,
) -> POMDPSpec<usize, GridLocalizationAction, GridLocalizationObservation> {
    let num_states = space.num_states;
    let states: Vec<usize> = (0..num_states).collect();
    let actions_vec = actions.to_vec();

    let trans_n = num_states;
    let transition: Box<dyn Fn(usize, usize) -> Vec<f64>> = Box::new(move |s_idx, _a| {
        let mut row = vec![0.0_f64; trans_n];
        row[s_idx] = 1.0;
        row
    });

    let space_obs = space.clone();
    let actions_obs = actions_vec.clone();
    let observation: Box<dyn Fn(usize, usize) -> Vec<f64>> = Box::new(move |s_idx, a_idx| {
        let action = &actions_obs[a_idx];
        let coords = space_obs.decode(s_idx);
        let (x, y) = (coords[0], coords[1]);
        let true_yes = match action.kind {
            GridLocalizationActionKind::ScanRow => y == action.y.unwrap(),
            GridLocalizationActionKind::ScanColumn => x == action.x.unwrap(),
            GridLocalizationActionKind::Inspect => {
                x == action.x.unwrap() && y == action.y.unwrap()
            }
        };
        let acc = if action.kind == GridLocalizationActionKind::Inspect {
            inspect_accuracy
        } else {
            scan_accuracy
        };
        let p_yes = if true_yes { acc } else { 1.0 - acc };
        vec![1.0 - p_yes, p_yes]
    });

    let space_rew = space.clone();
    let actions_rew = actions_vec.clone();
    let reward: Box<dyn Fn(usize, usize) -> f64> = Box::new(move |s_idx, a_idx| {
        let action = &actions_rew[a_idx];
        if action.kind != GridLocalizationActionKind::Inspect {
            return scan_cost;
        }
        let coords = space_rew.decode(s_idx);
        let (x, y) = (coords[0], coords[1]);
        if x == action.x.unwrap() && y == action.y.unwrap() {
            inspect_correct_reward
        } else {
            inspect_wrong_penalty
        }
    });

    POMDPSpec {
        states,
        actions: actions_vec,
        observations: vec![
            GridLocalizationObservation::No,
            GridLocalizationObservation::Yes,
        ],
        transition,
        observation,
        reward,
        discount,
        initial_belief: Some(initial_belief),
        is_terminal: None,
    }
}

/// Build the grid-localization model (state space, actions, spec).
pub fn build_grid_localization_pomdp(params: &GridLocalizationParams) -> GridLocalizationModel {
    validate_params(params);
    let width = params.width;
    let height = params.height;
    let space = CartesianStateSpace::new(vec![
        CartesianDimension {
            name: "x".to_string(),
            size: width,
            labels: None,
        },
        CartesianDimension {
            name: "y".to_string(),
            size: height,
            labels: None,
        },
    ]);
    let actions = build_actions(width, height);
    let scan_accuracy = params.scan_accuracy.unwrap_or(0.9);
    let inspect_accuracy = params.inspect_accuracy.unwrap_or(0.99);
    let scan_cost = params.scan_cost.unwrap_or(-0.2);
    let inspect_correct_reward = params.inspect_correct_reward.unwrap_or(20.0);
    let inspect_wrong_penalty = params.inspect_wrong_penalty.unwrap_or(-12.0);
    let discount = params.discount.unwrap_or(0.95);
    let n = space.num_states;
    let initial_belief = params
        .initial_belief
        .clone()
        .unwrap_or_else(|| vec![1.0 / n as f64; n]);
    let spec = build_grid_spec(
        &space,
        &actions,
        scan_accuracy,
        inspect_accuracy,
        scan_cost,
        inspect_correct_reward,
        inspect_wrong_penalty,
        discount,
        initial_belief,
    );
    GridLocalizationModel {
        space,
        actions,
        spec,
    }
}

/// Run the grid-localization POMDP under a finite-horizon belief lookahead
/// planner.
pub fn run_grid_localization_pomdp(params: &GridLocalizationParams) -> GridLocalizationResult {
    validate_params(params);
    let model = build_grid_localization_pomdp(params);
    let p = normalise_params(params, &model.space);
    let mut rng = mulberry32(p.seed);
    let hidden_target = p.hidden_target;
    let hidden_index = model.space.encode(&[hidden_target.0, hidden_target.1]);
    let mut belief = DiscreteBelief::new(model.spec.states.clone(), model.spec.initial_belief.as_deref());

    // The planner takes its spec by value, so it gets a private second copy
    // (identical, since the spec is a pure function of the resolved params).
    let planner_spec = build_grid_spec(
        &model.space,
        &model.actions,
        p.scan_accuracy,
        p.inspect_accuracy,
        p.scan_cost,
        p.inspect_correct_reward,
        p.inspect_wrong_penalty,
        p.discount,
        model
            .spec
            .initial_belief
            .clone()
            .unwrap_or_else(|| vec![1.0 / model.space.num_states as f64; model.space.num_states]),
    );
    let mut planner = BeliefLookaheadSolver::new(
        planner_spec,
        BeliefLookaheadOptions {
            horizon: p.horizon,
            leaf: BeliefLookaheadLeaf::Qmdp,
            max_nodes: 500_000,
            ..Default::default()
        },
    );

    let mut trace: Vec<GridLocalizationTraceRow> = Vec::new();
    let mut discount = 1.0;
    let mut total_return = 0.0;
    let mut found = false;
    let mut found_at_step: Option<usize> = None;

    let mut step = 0;
    while step < p.num_steps && !found {
        let action_idx = planner.act(&belief, None, 0.0);
        let action = model.actions[action_idx].clone();
        let obs_dist = (model.spec.observation)(hidden_index, action_idx);
        let obs_idx = sample_index(&obs_dist, &mut rng);
        let observation = OBSERVATIONS[obs_idx];
        total_return += discount * (model.spec.reward)(hidden_index, action_idx);
        discount *= p.discount;
        belief = belief_update(&model.spec, &belief, action_idx, obs_idx);
        found = action.kind == GridLocalizationActionKind::Inspect
            && observation == GridLocalizationObservation::Yes;
        if found {
            found_at_step = Some(step);
        }
        let mode_index = belief.mode_index();
        let mode_coords = model.space.decode(mode_index);
        let mode = (mode_coords[0], mode_coords[1]);
        trace.push(GridLocalizationTraceRow {
            step,
            action,
            observation,
            hidden_target,
            entropy: belief.entropy(),
            mode,
            mode_probability: belief.weights[mode_index],
            hidden_probability: belief.weights[hidden_index],
            found,
        });
        step += 1;
    }

    GridLocalizationResult {
        params: p,
        state_space: StateSpaceInfo {
            dimensions: model
                .space
                .dimensions
                .iter()
                .map(|d| DimensionInfo {
                    name: d.name.clone(),
                    size: d.size,
                })
                .collect(),
            num_states: model.space.num_states,
        },
        actions: model.actions.clone(),
        observations: OBSERVATIONS.to_vec(),
        trace,
        final_belief: belief.as_array(),
        final_entropy: belief.entropy(),
        found,
        found_at_step,
        total_return,
    }
}

fn build_actions(width: usize, height: usize) -> Vec<GridLocalizationAction> {
    let mut actions: Vec<GridLocalizationAction> = Vec::new();
    for y in 0..height {
        let index = actions.len();
        actions.push(GridLocalizationAction {
            kind: GridLocalizationActionKind::ScanRow,
            index,
            x: None,
            y: Some(y),
            label: format!("scan row {y}"),
        });
    }
    for x in 0..width {
        let index = actions.len();
        actions.push(GridLocalizationAction {
            kind: GridLocalizationActionKind::ScanColumn,
            index,
            x: Some(x),
            y: None,
            label: format!("scan column {x}"),
        });
    }
    for y in 0..height {
        for x in 0..width {
            let index = actions.len();
            actions.push(GridLocalizationAction {
                kind: GridLocalizationActionKind::Inspect,
                index,
                x: Some(x),
                y: Some(y),
                label: format!("inspect ({x},{y})"),
            });
        }
    }
    actions
}

fn validate_params(params: &GridLocalizationParams) {
    let cls = "GridLocalizationPOMDP";
    Preconditions::integer_in_range(cls, "width", params.width as f64, 2.0, 8.0).unwrap();
    Preconditions::integer_in_range(cls, "height", params.height as f64, 2.0, 8.0).unwrap();
    if let Some(horizon) = params.horizon {
        Preconditions::integer_in_range(cls, "horizon", horizon as f64, 0.0, 6.0).unwrap();
    }
    if let Some(num_steps) = params.num_steps {
        Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 100.0).unwrap();
    }
    if let Some(seed) = params.seed {
        Preconditions::integer(cls, "seed", seed as f64).unwrap();
    }
    if let Some(acc) = params.scan_accuracy {
        Preconditions::in_range(cls, "scanAccuracy", acc, 0.5, 1.0).unwrap();
    }
    if let Some(acc) = params.inspect_accuracy {
        Preconditions::in_range(cls, "inspectAccuracy", acc, 0.5, 1.0).unwrap();
    }
    if let Some(d) = params.discount {
        Preconditions::in_range(cls, "discount", d, 0.0, 1.0).unwrap();
    }
    if let Some(ib) = &params.initial_belief {
        Preconditions::length_eq(cls, "initialBelief", ib, params.width * params.height).unwrap();
        Preconditions::probability_vector(cls, "initialBelief", ib, 1e-9).unwrap();
    }
    if let Some(ht) = &params.hidden_target {
        // The TS `lengthEq(hiddenTarget, 2)` is structural for a `(usize, usize)`.
        Preconditions::integer_in_range(cls, "hiddenTarget.x", ht.0 as f64, 0.0, (params.width - 1) as f64)
            .unwrap();
        Preconditions::integer_in_range(cls, "hiddenTarget.y", ht.1 as f64, 0.0, (params.height - 1) as f64)
            .unwrap();
    }
}

fn normalise_params(
    params: &GridLocalizationParams,
    space: &CartesianStateSpace,
) -> GridLocalizationResolvedParams {
    let seed = params.seed.unwrap_or(1);
    let mut rng = mulberry32(seed.wrapping_add(10007));
    let hidden_target = match params.hidden_target {
        Some(ht) => ht,
        None => {
            let idx = (rng.next_float() * space.num_states as f64).floor() as usize;
            let coords = space.decode(idx);
            (coords[0], coords[1])
        }
    };
    GridLocalizationResolvedParams {
        width: params.width,
        height: params.height,
        horizon: params.horizon.unwrap_or(3),
        num_steps: params.num_steps.unwrap_or(8),
        seed,
        hidden_target,
        scan_accuracy: params.scan_accuracy.unwrap_or(0.9),
        inspect_accuracy: params.inspect_accuracy.unwrap_or(0.99),
        scan_cost: params.scan_cost.unwrap_or(-0.2),
        inspect_correct_reward: params.inspect_correct_reward.unwrap_or(20.0),
        inspect_wrong_penalty: params.inspect_wrong_penalty.unwrap_or(-12.0),
        discount: params.discount.unwrap_or(0.95),
    }
}

fn sample_index(probabilities: &[f64], rng: &mut dyn RandomSource) -> usize {
    let u = rng.next_float();
    let mut acc = 0.0;
    for i in 0..probabilities.len() {
        acc += probabilities[i];
        if u <= acc {
            return i;
        }
    }
    probabilities.len() - 1
}

#[cfg(test)]
mod tests {
    //! Grid-localization checks. Starting from a uniform prior the planner's
    //! noisy scans and inspects concentrate the belief: the final entropy drops
    //! below the uniform entropy and the posterior mass on the true cell rises
    //! above its uniform share.

    use super::*;

    #[test]
    fn belief_concentrates_on_a_small_grid() {
        let mut params = GridLocalizationParams::new(3, 3);
        params.hidden_target = Some((1, 1));
        params.seed = Some(1);
        params.num_steps = Some(8);
        let result = run_grid_localization_pomdp(&params);

        let n = result.state_space.num_states as f64;
        let uniform_entropy = n.ln();
        let uniform_share = 1.0 / n;

        assert!(!result.trace.is_empty());
        assert!(
            result.final_entropy < uniform_entropy,
            "entropy did not drop: {} >= {}",
            result.final_entropy,
            uniform_entropy
        );
        // Posterior mass on the true hidden cell exceeds the uniform share.
        let hidden_index = 4usize; // encode((col=1, row=1)) on a 3x3 grid = 1 + 1*3.
        assert!(
            result.final_belief[hidden_index] > uniform_share,
            "no information gained on the true cell: {}",
            result.final_belief[hidden_index]
        );
    }

    #[test]
    fn localises_and_gains_information() {
        let mut params = GridLocalizationParams::new(3, 3);
        params.hidden_target = Some((2, 0));
        params.seed = Some(5);
        params.num_steps = Some(12);
        params.horizon = Some(3);
        let result = run_grid_localization_pomdp(&params);

        let n = result.state_space.num_states as f64;
        let uniform_share = 1.0 / n;
        // encode((col=2, row=0)) on a 3x3 grid is 2 + 0*width = 2.
        let hidden_index = 2usize;

        // Localisation concentrates posterior mass on the true cell well above
        // its uniform share, and the final-step trace agrees with the belief.
        assert!(result.final_belief[hidden_index] > 2.0 * uniform_share);
        let final_row = result.trace.last().expect("at least one step");
        assert!(final_row.hidden_probability > uniform_share);
        assert!(result.final_entropy < n.ln());
    }
}
