//! Port of `src/des/general/advanced-control-models.ts` — H-infinity-style
//! robust control and a pursuit/evasion differential game, both built as
//! closed-loop adversarial station graphs (plant vs controller vs disturbance).
//!
//! TS mapping notes:
//!   * `class ScalarRobustPlant / LinearRobustController / WorstCaseScalarDisturbance`
//!     and `class PursuitEvasionPlant / PurePursuitController / PureEvasionController`
//!     subclass the bases in `des-base/adversarial-control.ts`
//!     (`ClosedLoopPlantStation`, `FeedbackPolicyStation`, `DisturbancePolicyStation`)
//!     driven by `runIterativeDES`. That module is NOT yet ported and is NOT among
//!     the allowed dependencies, so the deterministic (`shuffle: false`) game loop
//!     it produces is reproduced privately here as `run_closed_loop_game`, faithful
//!     to the plant→controller→adversary tick ordering.
//!   * the free helpers `clamp` / `norm2` become private functions (vanilla numeric
//!     algorithms, not stations).
//!   * `interface *Params / *Result` become structs; `number[]` → `Vec<f64>`;
//!     `.slice()` defensive copies → `.clone()`.
//!   * `Preconditions.*` throws become `Result<_, PreconditionError>` propagation;
//!     dimension invariants inside the loop become `assert!`/`panic!`.

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

// -----------------------------------------------------------------------------
// CHANNELS + TOPOLOGY METADATA
// -----------------------------------------------------------------------------

/// `CH_OBSERVATION` from `des-base/adversarial-control.ts` (reproduced locally).
pub const CH_OBSERVATION: &str = "observation";
/// `CH_CONTROL` from `des-base/adversarial-control.ts` (reproduced locally).
pub const CH_CONTROL: &str = "control";
/// `CH_DISTURBANCE` from `des-base/adversarial-control.ts` (reproduced locally).
pub const CH_DISTURBANCE: &str = "disturbance";

/// Channel-id bundle (port of the `advancedControlChannels` const object).
pub struct AdvancedControlChannels {
    pub observation: &'static str,
    pub control: &'static str,
    pub disturbance: &'static str,
}

/// Port of `advancedControlChannels`.
pub const ADVANCED_CONTROL_CHANNELS: AdvancedControlChannels = AdvancedControlChannels {
    observation: CH_OBSERVATION,
    control: CH_CONTROL,
    disturbance: CH_DISTURBANCE,
};

/// Shared station-graph topology metadata (port of `StationGraphTopology` from
/// `des-base/model-topology.ts`, reproduced locally as the module is unported).
#[derive(Clone, Debug)]
pub struct StationGraphTopology {
    pub stations: Vec<String>,
    pub movables: Vec<String>,
}

/// Port of `stationGraphTopology` (accepts string slices for ergonomics).
pub fn station_graph_topology(stations: &[&str], movables: &[&str]) -> StationGraphTopology {
    StationGraphTopology {
        stations: stations.iter().map(|s| s.to_string()).collect(),
        movables: movables.iter().map(|s| s.to_string()).collect(),
    }
}

/// One row of the closed-loop game trace (port of `ClosedLoopGameTraceRow`).
#[derive(Clone, Debug)]
pub struct ClosedLoopGameTraceRow {
    pub tick: usize,
    pub time: f64,
    pub state: Vec<f64>,
    pub control: Vec<f64>,
    pub disturbance: Vec<f64>,
    pub cost: f64,
}

// -----------------------------------------------------------------------------
// FREE NUMERIC HELPERS
// -----------------------------------------------------------------------------

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |acc, v| acc + v * v).sqrt()
}

// -----------------------------------------------------------------------------
// GAME PLANT TRAIT + DETERMINISTIC DRIVER
// -----------------------------------------------------------------------------

/// Behavioural contract of a `ClosedLoopPlantStation`: the required `dynamics`
/// hook plus `stage_cost` / `terminal` (defaults provided, as in the TS base).
trait GamePlant {
    fn dynamics(&self, state: &[f64], control: &[f64], disturbance: &[f64], dt: f64) -> Vec<f64>;
    fn stage_cost(&self, state: &[f64], control: &[f64], disturbance: &[f64], next: &[f64]) -> f64;
    fn terminal(&mut self, _state: &[f64], _tick: usize) -> bool {
        false
    }
}

struct GameOutput {
    state_history: Vec<Vec<f64>>,
    trace: Vec<ClosedLoopGameTraceRow>,
}

/// Faithful reproduction of `runClosedLoopGame` with `shuffle: false`.
///
/// Tick ordering is plant → controller → adversary, so the control/disturbance
/// applied while advancing from `x_{k-1}` to `x_k` is `policy(x_{k-1})` — i.e.
/// the policy of the state observed just before the advance. The loop advances
/// while `tick < num_steps` and the plant is not `terminal`, exactly mirroring
/// the station drain/guard/advance/emit sequence.
fn run_closed_loop_game<P, C, D>(
    plant: &mut P,
    x0: &[f64],
    dt: f64,
    num_steps: usize,
    control_dim: usize,
    disturbance_dim: usize,
    controller: C,
    adversary: D,
) -> GameOutput
where
    P: GamePlant,
    C: Fn(&[f64]) -> Vec<f64>,
    D: Fn(&[f64]) -> Vec<f64>,
{
    let mut state = x0.to_vec();
    let mut state_history = vec![state.clone()];
    let mut trace: Vec<ClosedLoopGameTraceRow> = Vec::new();
    let mut tick = 0usize;
    loop {
        // Controller and adversary react to the currently-observed state.
        let control = controller(&state);
        let disturbance = adversary(&state);
        assert_eq!(control.len(), control_dim, "control dimension mismatch");
        assert!(control.iter().all(|v| v.is_finite()), "control not finite");
        assert_eq!(disturbance.len(), disturbance_dim, "disturbance dimension mismatch");
        assert!(disturbance.iter().all(|v| v.is_finite()), "disturbance not finite");

        if tick >= num_steps || plant.terminal(&state, tick) {
            break;
        }

        let next = plant.dynamics(&state, &control, &disturbance, dt);
        assert_eq!(next.len(), state.len(), "next state dimension mismatch");
        assert!(next.iter().all(|v| v.is_finite()), "next state not finite");
        let cost = plant.stage_cost(&state, &control, &disturbance, &next);
        assert!(cost.is_finite(), "stage cost not finite");

        state = next;
        tick += 1;
        state_history.push(state.clone());
        trace.push(ClosedLoopGameTraceRow {
            tick,
            time: tick as f64 * dt,
            state: state.clone(),
            control,
            disturbance,
            cost,
        });
    }
    GameOutput { state_history, trace }
}

// -----------------------------------------------------------------------------
// H-infinity-style bounded-disturbance robust control
// -----------------------------------------------------------------------------

/// Options for [`run_h_infinity_robust_control`]; `None` fields take TS defaults.
#[derive(Clone, Debug, Default)]
pub struct HInfinityRobustControlParams {
    pub x0: Option<f64>,
    pub a: Option<f64>,
    pub b: Option<f64>,
    pub gain: Option<f64>,
    pub disturbance_max: Option<f64>,
    pub control_max: Option<f64>,
    pub gamma: Option<f64>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
}

/// Result of an H-infinity robust-control run.
#[derive(Clone, Debug)]
pub struct HInfinityRobustControlResult {
    pub trace: Vec<ClosedLoopGameTraceRow>,
    pub final_state: f64,
    pub peak_abs_state: f64,
    pub l2_gain_estimate: f64,
    pub gamma: f64,
    pub bounded_by_gamma: bool,
    pub topology: StationGraphTopology,
}

struct ScalarRobustPlant {
    a: f64,
    b: f64,
}

impl GamePlant for ScalarRobustPlant {
    fn dynamics(&self, state: &[f64], control: &[f64], disturbance: &[f64], dt: f64) -> Vec<f64> {
        let xdot = self.a * state[0] + self.b * control[0] + disturbance[0];
        vec![state[0] + dt * xdot]
    }

    fn stage_cost(&self, _state: &[f64], control: &[f64], disturbance: &[f64], next: &[f64]) -> f64 {
        next[0] * next[0] + 0.02 * control[0] * control[0] - 0.02 * disturbance[0] * disturbance[0]
    }
}

/// Run the bounded-disturbance robust-control game and estimate the L2 gain.
pub fn run_h_infinity_robust_control(
    params: HInfinityRobustControlParams,
) -> Result<HInfinityRobustControlResult, PreconditionError> {
    let x0 = params.x0.unwrap_or(2.0);
    let a = params.a.unwrap_or(0.25);
    let b = params.b.unwrap_or(1.0);
    let gain = params.gain.unwrap_or(3.2);
    let disturbance_max = params.disturbance_max.unwrap_or(0.45);
    let control_max = params.control_max.unwrap_or(5.0);
    let gamma = params.gamma.unwrap_or(2.5);
    let dt = params.dt.unwrap_or(0.03);
    let num_steps = params.num_steps.unwrap_or(260);

    let cls = "runHInfinityRobustControl";
    Preconditions::finite(cls, "x0", x0)?;
    Preconditions::finite(cls, "a", a)?;
    Preconditions::finite(cls, "b", b)?;
    Preconditions::positive(cls, "gain", gain)?;
    Preconditions::non_negative(cls, "disturbanceMax", disturbance_max)?;
    Preconditions::positive(cls, "controlMax", control_max)?;
    Preconditions::positive(cls, "gamma", gamma)?;
    // Plant-construction guards from ClosedLoopPlantStation.
    Preconditions::positive("hinfinity-plant", "dt", dt)?;
    Preconditions::integer_in_range("hinfinity-plant", "numSteps", num_steps as f64, 1.0, 1e9)?;

    let mut plant = ScalarRobustPlant { a, b };
    let controller = move |state: &[f64]| vec![clamp(-gain * state[0], -control_max, control_max)];
    let adversary = move |state: &[f64]| {
        vec![if state[0] >= 0.0 { disturbance_max } else { -disturbance_max }]
    };
    let out = run_closed_loop_game(&mut plant, &[x0], dt, num_steps, 1, 1, controller, adversary);

    let state_energy: f64 = out.trace.iter().map(|r| r.state[0] * r.state[0]).sum();
    let disturbance_energy: f64 = out.trace.iter().map(|r| r.disturbance[0] * r.disturbance[0]).sum();
    let l2_gain_estimate = (state_energy / disturbance_energy.max(1e-12)).sqrt();
    let peak_abs_state = out.state_history.iter().fold(0.0_f64, |acc, s| acc.max(s[0].abs()));
    let final_state = out.state_history.last().map(|s| s[0]).unwrap_or(x0);

    Ok(HInfinityRobustControlResult {
        trace: out.trace,
        final_state,
        peak_abs_state,
        l2_gain_estimate,
        gamma,
        bounded_by_gamma: l2_gain_estimate <= gamma,
        topology: station_graph_topology(
            &["hinfinity-plant", "hinfinity-state-feedback-controller", "worst-case-disturbance-station"],
            &["StateObservationToken", "ControlMoveToken", "DisturbanceMoveToken"],
        ),
    })
}

// -----------------------------------------------------------------------------
// Differential game: pursuit/evasion with two competing controllers
// -----------------------------------------------------------------------------

/// Options for [`run_pursuit_evasion_game`]; `None` fields take TS defaults.
#[derive(Clone, Debug, Default)]
pub struct PursuitEvasionGameParams {
    pub pursuer: Option<[f64; 2]>,
    pub evader: Option<[f64; 2]>,
    pub pursuer_speed: Option<f64>,
    pub evader_speed: Option<f64>,
    pub capture_radius: Option<f64>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
}

/// Result of a pursuit/evasion game.
#[derive(Clone, Debug)]
pub struct PursuitEvasionGameResult {
    pub trace: Vec<ClosedLoopGameTraceRow>,
    pub distance_history: Vec<f64>,
    /// Tick at first capture, or `None` if the evader was never captured.
    pub capture_tick: Option<usize>,
    pub final_distance: f64,
    pub topology: StationGraphTopology,
}

struct PursuitEvasionPlant {
    capture_radius: f64,
    capture_tick: Option<usize>,
}

impl PursuitEvasionPlant {
    fn distance(state: &[f64]) -> f64 {
        f64::hypot(state[2] - state[0], state[3] - state[1])
    }
}

impl GamePlant for PursuitEvasionPlant {
    fn dynamics(&self, state: &[f64], control: &[f64], disturbance: &[f64], dt: f64) -> Vec<f64> {
        vec![
            state[0] + dt * control[0],
            state[1] + dt * control[1],
            state[2] + dt * disturbance[0],
            state[3] + dt * disturbance[1],
        ]
    }

    fn stage_cost(&self, _state: &[f64], _control: &[f64], _disturbance: &[f64], next: &[f64]) -> f64 {
        Self::distance(next)
    }

    fn terminal(&mut self, state: &[f64], tick: usize) -> bool {
        if Self::distance(state) <= self.capture_radius {
            if self.capture_tick.is_none() {
                self.capture_tick = Some(tick);
            }
            return true;
        }
        false
    }
}

/// Run a pure-pursuit vs pure-evasion differential game.
pub fn run_pursuit_evasion_game(
    params: PursuitEvasionGameParams,
) -> Result<PursuitEvasionGameResult, PreconditionError> {
    let pursuer = params.pursuer.unwrap_or([0.0, 0.0]);
    let evader = params.evader.unwrap_or([6.0, 2.0]);
    let pursuer_speed = params.pursuer_speed.unwrap_or(1.25);
    let evader_speed = params.evader_speed.unwrap_or(0.6);
    let capture_radius = params.capture_radius.unwrap_or(0.25);
    let dt = params.dt.unwrap_or(0.1);
    let num_steps = params.num_steps.unwrap_or(120);

    let cls = "runPursuitEvasionGame";
    Preconditions::length_eq(cls, "pursuer", &pursuer, 2)?;
    Preconditions::length_eq(cls, "evader", &evader, 2)?;
    Preconditions::all_finite(cls, "pursuer", &pursuer)?;
    Preconditions::all_finite(cls, "evader", &evader)?;
    Preconditions::positive(cls, "pursuerSpeed", pursuer_speed)?;
    Preconditions::non_negative(cls, "evaderSpeed", evader_speed)?;
    Preconditions::positive(cls, "captureRadius", capture_radius)?;
    Preconditions::positive("pursuit-evasion-plant", "dt", dt)?;
    Preconditions::integer_in_range("pursuit-evasion-plant", "numSteps", num_steps as f64, 1.0, 1e9)?;

    let mut plant = PursuitEvasionPlant {
        capture_radius,
        capture_tick: None,
    };
    // Both controllers steer along the pursuer→evader unit vector.
    let pursuit = move |state: &[f64]| {
        let dx = state[2] - state[0];
        let dy = state[3] - state[1];
        let n = f64::hypot(dx, dy).max(1e-12);
        vec![pursuer_speed * dx / n, pursuer_speed * dy / n]
    };
    let evasion = move |state: &[f64]| {
        let dx = state[2] - state[0];
        let dy = state[3] - state[1];
        let n = f64::hypot(dx, dy).max(1e-12);
        vec![evader_speed * dx / n, evader_speed * dy / n]
    };
    let x0 = [pursuer[0], pursuer[1], evader[0], evader[1]];
    let out = run_closed_loop_game(&mut plant, &x0, dt, num_steps, 2, 2, pursuit, evasion);

    let distance_history: Vec<f64> = out
        .state_history
        .iter()
        .map(|s| norm2(&[s[2] - s[0], s[3] - s[1]]))
        .collect();
    let final_distance = *distance_history.last().unwrap();

    Ok(PursuitEvasionGameResult {
        trace: out.trace,
        distance_history,
        capture_tick: plant.capture_tick,
        final_distance,
        topology: station_graph_topology(
            &["pursuit-evasion-plant", "pure-pursuit-controller", "pure-evasion-controller"],
            &["StateObservationToken", "ControlMoveToken", "DisturbanceMoveToken"],
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_infinity_stays_within_gamma_and_stabilises() {
        let r = run_h_infinity_robust_control(HInfinityRobustControlParams::default()).unwrap();
        // 260 advances → 260 trace rows.
        assert_eq!(r.trace.len(), 260);
        // The worst-case L2 gain estimate stays under the design level γ.
        assert!(r.l2_gain_estimate <= r.gamma, "l2 = {}", r.l2_gain_estimate);
        assert!(r.bounded_by_gamma);
        // The state-feedback law drives the plant well below its initial |x0| = 2.
        assert!(r.final_state.abs() < 0.5, "final_state = {}", r.final_state);
    }

    #[test]
    fn pursuit_captures_a_slower_evader() {
        let r = run_pursuit_evasion_game(PursuitEvasionGameParams::default()).unwrap();
        // The faster pursuer closes the gap and captures before the run ends.
        let capture = r.capture_tick.expect("expected a capture");
        assert!(capture > 0 && capture < 120, "capture_tick = {capture}");
        assert!(r.final_distance <= 0.25 + 1e-9, "final_distance = {}", r.final_distance);
        // Distance is monotonically non-increasing while closing in a straight line.
        assert!(r.distance_history[0] > r.final_distance);
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(run_h_infinity_robust_control(HInfinityRobustControlParams {
            gain: Some(-1.0),
            ..Default::default()
        })
        .is_err());
        assert!(run_pursuit_evasion_game(PursuitEvasionGameParams {
            capture_radius: Some(0.0),
            ..Default::default()
        })
        .is_err());
    }
}
