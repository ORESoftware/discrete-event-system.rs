//! MDP and POMDP as first-class [`ModelCitizen`]s: validate a JSON spec, solve,
//! roll out, and return a uniform [`RunArtifact`]. These are the canonical
//! example of the contract — the same pattern any other paradigm follows.

use serde_json::{json, Value};

use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};

use super::solve::{solve_mdp, solve_pomdp, solve_pomdp_underlying, MdpMethod, PomdpMethod};
use super::spec::{MdpSpec, PomdpSpec, MDP_SCHEMA, POMDP_SCHEMA};
use super::viz::{mdp_artifact, pomdp_artifact};
use super::{rollout_mdp, rollout_pomdp};

fn usize_field(spec: &Value, key: &str, default: usize) -> usize {
    spec.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(default)
}
fn u64_field(spec: &Value, key: &str, default: u64) -> u64 {
    spec.get(key).and_then(Value::as_u64).unwrap_or(default)
}

/// MDP first-class citizen.
pub struct MdpCitizen;

impl ModelCitizen for MdpCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "mdp".to_string(),
            title: "Markov Decision Process".to_string(),
            description: "Fully-observed sequential decision process. Solved by value \
                          iteration into an optimal value function and policy, then rolled \
                          out into an animated state-transition trajectory."
                .to_string(),
            spec_schema: MDP_SCHEMA.to_string(),
            methods: vec!["value-iteration".to_string()],
            example_spec: json!({
                "$schema": MDP_SCHEMA,
                "numStates": 3,
                "discount": 0.9,
                "stateLabels": ["start", "middle", "goal"],
                "actionLabels": ["advance", "wait"],
                "transitions": [
                    [ [{"prob": 1.0, "reward": -1.0, "next": 1}], [{"prob": 1.0, "reward": 0.0, "next": 0}] ],
                    [ [{"prob": 1.0, "reward": -1.0, "next": 2}], [{"prob": 1.0, "reward": 0.0, "next": 1}] ],
                    []
                ],
                "terminal": [{"state": 2, "reward": 10.0}],
                "start": 0, "steps": 12, "seed": 1
            }),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let mdp: MdpSpec = serde_json::from_value(spec.clone())
            .map_err(|e| CitizenError::InvalidSpec(format!("could not parse MdpSpec: {e}")))?;
        mdp.validate().map_err(CitizenError::InvalidSpec)?;

        let sol = solve_mdp(&mdp, MdpMethod::ValueIteration).map_err(CitizenError::Run)?;
        let start = usize_field(spec, "start", 0).min(mdp.num_states.saturating_sub(1));
        let steps = usize_field(spec, "steps", 16);
        let seed = u64_field(spec, "seed", 1);
        let trace = rollout_mdp(&mdp, &sol.policy, start, steps, seed);

        Ok(mdp_artifact(
            &mdp,
            &sol,
            &trace,
            "Markov Decision Process",
            "Value-iteration policy and an animated rollout over the state graph.",
        ))
    }
}

/// POMDP first-class citizen.
pub struct PomdpCitizen;

impl ModelCitizen for PomdpCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "pomdp".to_string(),
            title: "Partially Observable MDP".to_string(),
            description: "Sequential decision process under state uncertainty. A Bayesian \
                          belief is tracked over hidden states; a chosen solver (qmdp, \
                          lookahead, exact-finite-horizon, most-likely-state) picks actions, \
                          and the rollout animates the evolving belief."
                .to_string(),
            spec_schema: POMDP_SCHEMA.to_string(),
            methods: vec![
                "qmdp".to_string(),
                "lookahead".to_string(),
                "exact-finite-horizon".to_string(),
                "most-likely-state".to_string(),
            ],
            example_spec: json!({
                "$schema": POMDP_SCHEMA,
                "numStates": 2, "numActions": 3, "numObservations": 2,
                "discount": 0.95,
                "stateLabels": ["tiger-left", "tiger-right"],
                "actionLabels": ["listen", "open-left", "open-right"],
                "observationLabels": ["hear-left", "hear-right"],
                "transition": [
                    [ [1.0, 0.0], [0.5, 0.5], [0.5, 0.5] ],
                    [ [0.0, 1.0], [0.5, 0.5], [0.5, 0.5] ]
                ],
                "observation": [
                    [ [0.85, 0.15], [0.5, 0.5], [0.5, 0.5] ],
                    [ [0.15, 0.85], [0.5, 0.5], [0.5, 0.5] ]
                ],
                "reward": [ [-1.0, -100.0, 10.0], [-1.0, 10.0, -100.0] ],
                "method": "lookahead", "horizon": 3, "steps": 16, "seed": 1
            }),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let pomdp: PomdpSpec = serde_json::from_value(spec.clone())
            .map_err(|e| CitizenError::InvalidSpec(format!("could not parse PomdpSpec: {e}")))?;
        pomdp.validate().map_err(CitizenError::InvalidSpec)?;

        let method: PomdpMethod = spec
            .get("method")
            .cloned()
            .map(|m| serde_json::from_value(m).unwrap_or_default())
            .unwrap_or_default();
        let horizon = usize_field(spec, "horizon", 3);
        let steps = usize_field(spec, "steps", 16);
        let seed = u64_field(spec, "seed", 1);
        let start = spec
            .get("startState")
            .and_then(Value::as_u64)
            .map(|n| n as usize);

        let mut plan = solve_pomdp(&pomdp, method, horizon).map_err(CitizenError::Run)?;
        let sol = solve_pomdp_underlying(&pomdp).map_err(CitizenError::Run)?;
        let trace = rollout_pomdp(&pomdp, &mut plan, start, steps, seed);

        let method_label = format!("{method:?}");
        Ok(pomdp_artifact(
            &pomdp,
            &sol,
            &trace,
            &method_label,
            "Partially Observable MDP",
            "Belief-tracking policy and an animated belief rollout over hidden states.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdp_citizen_runs_its_example() {
        let c = MdpCitizen;
        let art = c.run_json(&c.descriptor().example_spec).unwrap();
        assert_eq!(art.kind, "mdp");
        assert!(!art.frames.is_empty());
        assert!(art.results["value"].is_array());
    }

    #[test]
    fn pomdp_citizen_runs_its_example() {
        let c = PomdpCitizen;
        let art = c.run_json(&c.descriptor().example_spec).unwrap();
        assert_eq!(art.kind, "pomdp");
        assert!(!art.frames.is_empty());
        assert!(art.results["underlyingValue"].is_array());
    }

    #[test]
    fn invalid_spec_is_reported_not_panicked() {
        let c = MdpCitizen;
        let bad = json!({ "numStates": 1, "transitions": [] });
        match c.run_json(&bad) {
            Err(CitizenError::InvalidSpec(_)) => {}
            other => panic!("expected InvalidSpec, got {other:?}"),
        }
    }
}
