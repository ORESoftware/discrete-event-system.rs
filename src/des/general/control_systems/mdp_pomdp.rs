//! Canonical MDP/POMDP decision-process views for control-system simulations.
//!
//! The concrete controllers in this directory are continuous or hybrid plants;
//! these helpers add a small, JSON-first decision layer over their operating
//! regimes. The returned payloads use the first-class `des/mdp/v1` and
//! `des/pomdp/v1` contracts, so callers can feed them directly to
//! `des::model`'s MDP/POMDP citizens for solving, rollout, and visualization.

use serde_json::{json, Value};

/// DC-motor speed-regime MDP.
///
/// States are coarse speed regimes; actions are voltage-profile choices. The
/// reward model favours reaching and holding the target-speed band while
/// penalising wasted drive/brake commands.
pub fn dc_motor_speed_mdp_spec() -> Value {
    json!({
        "$schema": "des/mdp/v1",
        "numStates": 4,
        "discount": 0.92,
        "stateLabels": ["stopped", "underspeed", "target-speed", "overspeed"],
        "actionLabels": ["brake", "hold", "drive"],
        "transitions": [
            [
                [{"prob": 1.0, "reward": -2.0, "next": 0}],
                [{"prob": 1.0, "reward": -1.0, "next": 0}],
                [{"prob": 1.0, "reward": -0.2, "next": 1}]
            ],
            [
                [{"prob": 1.0, "reward": -3.0, "next": 0}],
                [{"prob": 1.0, "reward": -1.0, "next": 1}],
                [{"prob": 0.75, "reward": 2.0, "next": 2}, {"prob": 0.25, "reward": 0.5, "next": 1}]
            ],
            [
                [{"prob": 1.0, "reward": -0.5, "next": 1}],
                [{"prob": 1.0, "reward": 5.0, "next": 2}],
                [{"prob": 1.0, "reward": -1.0, "next": 3}]
            ],
            [
                [{"prob": 0.80, "reward": 2.0, "next": 2}, {"prob": 0.20, "reward": -0.5, "next": 3}],
                [{"prob": 1.0, "reward": -2.0, "next": 3}],
                [{"prob": 1.0, "reward": -4.0, "next": 3}]
            ]
        ],
        "start": 1,
        "steps": 18,
        "seed": 7
    })
}

/// DC-motor speed-regime POMDP with a noisy tachometer.
///
/// Hidden states match [`dc_motor_speed_mdp_spec`]. Observations are low/ok/high
/// tachometer readings; the target band is usually but not perfectly observed.
pub fn dc_motor_speed_pomdp_spec() -> Value {
    json!({
        "$schema": "des/pomdp/v1",
        "numStates": 4,
        "numActions": 3,
        "numObservations": 3,
        "discount": 0.92,
        "stateLabels": ["stopped", "underspeed", "target-speed", "overspeed"],
        "actionLabels": ["brake", "hold", "drive"],
        "observationLabels": ["tacho-low", "tacho-ok", "tacho-high"],
        "transition": [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.25, 0.75, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.80, 0.20], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ],
        "observation": [
            [[0.90, 0.10, 0.00], [0.90, 0.10, 0.00], [0.90, 0.10, 0.00]],
            [[0.75, 0.20, 0.05], [0.75, 0.20, 0.05], [0.75, 0.20, 0.05]],
            [[0.10, 0.80, 0.10], [0.10, 0.80, 0.10], [0.10, 0.80, 0.10]],
            [[0.05, 0.25, 0.70], [0.05, 0.25, 0.70], [0.05, 0.25, 0.70]]
        ],
        "reward": [
            [-2.0, -1.0, -0.2],
            [-3.0, -1.0, 2.0],
            [-0.5, 5.0, -1.0],
            [2.0, -2.0, -4.0]
        ],
        "initialBelief": [0.10, 0.50, 0.30, 0.10],
        "method": "lookahead",
        "horizon": 3,
        "steps": 18,
        "seed": 7
    })
}

/// Wind-turbine MPPT regime MDP.
///
/// States are tip-speed-ratio/power regimes. Actions adjust generator torque:
/// lowering torque lets the rotor accelerate, raising torque loads/brakes it.
pub fn wind_mppt_regime_mdp_spec() -> Value {
    json!({
        "$schema": "des/mdp/v1",
        "numStates": 4,
        "discount": 0.90,
        "stateLabels": ["below-tsr", "near-mpp", "above-tsr", "gust-limited"],
        "actionLabels": ["lower-torque", "hold", "raise-torque"],
        "transitions": [
            [
                [{"prob": 0.70, "reward": 3.0, "next": 1}, {"prob": 0.30, "reward": 0.5, "next": 0}],
                [{"prob": 1.0, "reward": -1.0, "next": 0}],
                [{"prob": 1.0, "reward": -3.0, "next": 0}]
            ],
            [
                [{"prob": 0.70, "reward": 2.0, "next": 1}, {"prob": 0.30, "reward": 0.0, "next": 2}],
                [{"prob": 1.0, "reward": 6.0, "next": 1}],
                [{"prob": 0.70, "reward": 2.0, "next": 1}, {"prob": 0.30, "reward": 0.0, "next": 0}]
            ],
            [
                [{"prob": 1.0, "reward": -3.0, "next": 2}],
                [{"prob": 1.0, "reward": -1.0, "next": 2}],
                [{"prob": 0.75, "reward": 3.0, "next": 1}, {"prob": 0.25, "reward": 0.5, "next": 2}]
            ],
            [
                [{"prob": 0.60, "reward": -2.0, "next": 2}, {"prob": 0.40, "reward": -4.0, "next": 3}],
                [{"prob": 0.70, "reward": -1.5, "next": 3}, {"prob": 0.30, "reward": 1.0, "next": 1}],
                [{"prob": 0.65, "reward": 2.0, "next": 1}, {"prob": 0.35, "reward": -0.5, "next": 3}]
            ]
        ],
        "start": 0,
        "steps": 20,
        "seed": 11
    })
}

/// Wind MPPT POMDP with noisy power-slope observations.
///
/// Observations do not reveal the exact tip-speed-ratio regime; they only say
/// whether measured power is rising, flat near the peak, or falling.
pub fn wind_mppt_sensor_pomdp_spec() -> Value {
    json!({
        "$schema": "des/pomdp/v1",
        "numStates": 4,
        "numActions": 3,
        "numObservations": 3,
        "discount": 0.90,
        "stateLabels": ["below-tsr", "near-mpp", "above-tsr", "gust-limited"],
        "actionLabels": ["lower-torque", "hold", "raise-torque"],
        "observationLabels": ["power-rising", "power-flat", "power-falling"],
        "transition": [
            [[0.30, 0.70, 0.00, 0.00], [1.00, 0.00, 0.00, 0.00], [1.00, 0.00, 0.00, 0.00]],
            [[0.00, 0.70, 0.30, 0.00], [0.00, 1.00, 0.00, 0.00], [0.30, 0.70, 0.00, 0.00]],
            [[0.00, 0.00, 1.00, 0.00], [0.00, 0.00, 1.00, 0.00], [0.00, 0.75, 0.25, 0.00]],
            [[0.00, 0.00, 0.60, 0.40], [0.00, 0.30, 0.00, 0.70], [0.00, 0.65, 0.00, 0.35]]
        ],
        "observation": [
            [[0.78, 0.17, 0.05], [0.78, 0.17, 0.05], [0.78, 0.17, 0.05]],
            [[0.15, 0.75, 0.10], [0.15, 0.75, 0.10], [0.15, 0.75, 0.10]],
            [[0.08, 0.22, 0.70], [0.08, 0.22, 0.70], [0.08, 0.22, 0.70]],
            [[0.25, 0.25, 0.50], [0.25, 0.25, 0.50], [0.25, 0.25, 0.50]]
        ],
        "reward": [
            [3.0, -1.0, -3.0],
            [2.0, 6.0, 2.0],
            [-3.0, -1.0, 3.0],
            [-2.0, -1.5, 2.0]
        ],
        "initialBelief": [0.35, 0.30, 0.25, 0.10],
        "method": "lookahead",
        "horizon": 3,
        "steps": 20,
        "seed": 11
    })
}

/// Named specs for discovery UIs or batch validators.
pub fn control_system_decision_specs() -> Vec<(&'static str, Value)> {
    vec![
        ("dc-motor-speed-mdp", dc_motor_speed_mdp_spec()),
        ("dc-motor-speed-pomdp", dc_motor_speed_pomdp_spec()),
        ("wind-mppt-regime-mdp", wind_mppt_regime_mdp_spec()),
        ("wind-mppt-sensor-pomdp", wind_mppt_sensor_pomdp_spec()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::solve::{solve_mdp, solve_pomdp_underlying, MdpMethod};
    use crate::des::decision::spec::{MdpSpec, PomdpSpec};

    #[test]
    fn control_system_decision_specs_validate() {
        for (name, spec) in control_system_decision_specs() {
            match spec.get("$schema").and_then(Value::as_str) {
                Some("des/mdp/v1") => {
                    let mdp: MdpSpec = serde_json::from_value(spec).expect(name);
                    mdp.validate().expect(name);
                    let solution = solve_mdp(&mdp, MdpMethod::ValueIteration).expect(name);
                    assert_eq!(solution.policy.len(), mdp.num_states, "{name}");
                }
                Some("des/pomdp/v1") => {
                    let pomdp: PomdpSpec = serde_json::from_value(spec).expect(name);
                    pomdp.validate().expect(name);
                    let solution = solve_pomdp_underlying(&pomdp).expect(name);
                    assert_eq!(solution.underlying_policy.len(), pomdp.num_states, "{name}");
                }
                other => panic!("{name}: unexpected schema {other:?}"),
            }
        }
    }
}
