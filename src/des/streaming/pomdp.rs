//! Streaming partially-observable MDP — tabular, build-then-solve.
//!
//! Mirrors [`super::mdp::StreamingMdp`] for the partially-observable case. A
//! command stream declares the ⟨S, A, Ω⟩ sizes, sets transition/observation
//! rows and rewards, optionally sets a belief, and a `solve` command builds a
//! canonical [`PomdpSpec`](crate::des::decision::PomdpSpec), solves it with the
//! requested method, and streams back the underlying value/policy plus the
//! action recommended under the current belief. The model can be edited and
//! re-solved as the stream continues.
//!
//! It reuses the canonical `des::decision` spec + solvers verbatim and runs each
//! solve under `catch_unwind`, so a bad command never tears down the stream.

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Value};

use crate::des::decision::{solve_pomdp, solve_pomdp_underlying, PomdpMethod, PomdpSpec};
use crate::des::general::belief::DiscreteBelief;

use super::{
    error_frame, f64_at, op_of, usize_at, vec_f64, SolverKind, StreamContract, StreamEvent,
    StreamOp, StreamingModel,
};

/// A tabular POMDP assembled by a JSONL command stream and solved on demand.
pub struct StreamingPomdp {
    ns: usize,
    na: usize,
    no: usize,
    transition: Vec<Vec<Vec<f64>>>,  // [s][a] -> dist over next states
    observation: Vec<Vec<Vec<f64>>>, // [s'][a] -> dist over observations
    reward: Vec<Vec<f64>>,           // [s][a]
    belief: Vec<f64>,
    gamma: f64,
    initialized: bool,
}

impl Default for StreamingPomdp {
    fn default() -> Self {
        StreamingPomdp {
            ns: 0,
            na: 0,
            no: 0,
            transition: Vec::new(),
            observation: Vec::new(),
            reward: Vec::new(),
            belief: Vec::new(),
            gamma: 0.95,
            initialized: false,
        }
    }
}

impl StreamingPomdp {
    pub fn new() -> Self {
        Self::default()
    }

    fn init(&mut self, command: &Value) -> Vec<Value> {
        let ns = match usize_at(command, "numStates") {
            Some(n) if n > 0 => n,
            _ => return vec![error_frame("`numStates` (positive integer) is required")],
        };
        let na = match usize_at(command, "numActions") {
            Some(n) if n > 0 => n,
            _ => return vec![error_frame("`numActions` (positive integer) is required")],
        };
        let no = match usize_at(command, "numObservations") {
            Some(n) if n > 0 => n,
            _ => return vec![error_frame("`numObservations` (positive integer) is required")],
        };
        self.ns = ns;
        self.na = na;
        self.no = no;
        self.gamma = f64_at(command, "gamma", 0.95);
        // Defaults: stay-in-place transitions, uniform observations, zero reward.
        self.transition = (0..ns)
            .map(|s| {
                (0..na)
                    .map(|_| {
                        let mut row = vec![0.0; ns];
                        row[s] = 1.0;
                        row
                    })
                    .collect()
            })
            .collect();
        self.observation = vec![vec![vec![1.0 / no as f64; no]; na]; ns];
        self.reward = vec![vec![0.0; na]; ns];
        self.belief = vec![1.0 / ns as f64; ns];
        self.initialized = true;
        vec![json!({
            "event": "initialized",
            "numStates": ns, "numActions": na, "numObservations": no, "gamma": self.gamma,
        })]
    }

    fn set_transition(&mut self, command: &Value) -> Vec<Value> {
        let s = match usize_at(command, "state") {
            Some(s) if s < self.ns => s,
            _ => return vec![error_frame("`state` out of range")],
        };
        let a = match usize_at(command, "action") {
            Some(a) if a < self.na => a,
            _ => return vec![error_frame("`action` out of range")],
        };
        let probs = match vec_f64(command, "probs") {
            Some(p) if p.len() == self.ns => p,
            _ => return vec![error_frame("`probs` must be an array of length numStates")],
        };
        self.transition[s][a] = probs;
        vec![json!({"event":"applied","op":"set_transition","state":s,"action":a})]
    }

    fn set_observation(&mut self, command: &Value) -> Vec<Value> {
        let sp = match usize_at(command, "nextState") {
            Some(s) if s < self.ns => s,
            _ => return vec![error_frame("`nextState` out of range")],
        };
        let a = match usize_at(command, "action") {
            Some(a) if a < self.na => a,
            _ => return vec![error_frame("`action` out of range")],
        };
        let probs = match vec_f64(command, "probs") {
            Some(p) if p.len() == self.no => p,
            _ => return vec![error_frame("`probs` must be an array of length numObservations")],
        };
        self.observation[sp][a] = probs;
        vec![json!({"event":"applied","op":"set_observation","nextState":sp,"action":a})]
    }

    fn set_reward(&mut self, command: &Value) -> Vec<Value> {
        let s = match usize_at(command, "state") {
            Some(s) if s < self.ns => s,
            _ => return vec![error_frame("`state` out of range")],
        };
        let a = match usize_at(command, "action") {
            Some(a) if a < self.na => a,
            _ => return vec![error_frame("`action` out of range")],
        };
        self.reward[s][a] = f64_at(command, "reward", 0.0);
        vec![json!({"event":"applied","op":"set_reward","state":s,"action":a})]
    }

    fn set_belief(&mut self, command: &Value) -> Vec<Value> {
        let b = match vec_f64(command, "belief") {
            Some(p) if p.len() == self.ns => p,
            _ => return vec![error_frame("`belief` must be an array of length numStates")],
        };
        let sum: f64 = b.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return vec![error_frame("`belief` must have a positive, finite sum")];
        }
        self.belief = b.iter().map(|x| x / sum).collect();
        vec![json!({"event":"applied","op":"set_belief","belief":self.belief})]
    }

    fn build_spec(&self) -> PomdpSpec {
        PomdpSpec {
            schema: crate::des::decision::POMDP_SCHEMA.to_string(),
            num_states: self.ns,
            num_actions: self.na,
            num_observations: self.no,
            transition: self.transition.clone(),
            observation: self.observation.clone(),
            reward: self.reward.clone(),
            discount: self.gamma,
            initial_belief: Some(self.belief.clone()),
            state_labels: vec![],
            action_labels: vec![],
            observation_labels: vec![],
        }
    }

    fn solve(&mut self, command: &Value) -> Vec<Value> {
        if !self.initialized {
            return vec![error_frame("no POMDP initialized; send {\"op\":\"init\", ...} first")];
        }
        let method: PomdpMethod = command
            .get("method")
            .cloned()
            .map(|m| serde_json::from_value(m).unwrap_or_default())
            .unwrap_or_default();
        let horizon = usize_at(command, "horizon").unwrap_or(3);
        let spec = self.build_spec();
        let belief = self.belief.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            spec.validate()?;
            let underlying = solve_pomdp_underlying(&spec)?;
            let mut plan = solve_pomdp(&spec, method, horizon)?;
            let b = DiscreteBelief::new((0..spec.num_states).collect(), Some(&belief));
            let action = plan.act(&b);
            Ok::<_, String>((underlying, action))
        }));

        match result {
            Ok(Ok((underlying, action))) => vec![json!({
                "event": "solution",
                "method": method,
                "gamma": self.gamma,
                "recommendedAction": action,
                "belief": self.belief,
                "underlyingValue": underlying.underlying_value,
                "underlyingPolicy": underlying.underlying_policy,
                "q": underlying.q,
            })],
            Ok(Err(msg)) => vec![error_frame(msg)],
            Err(_) => vec![error_frame("POMDP solve failed")],
        }
    }
}

impl StreamingModel for StreamingPomdp {
    fn kind(&self) -> SolverKind {
        SolverKind::IterativeSolver
    }

    fn contract(&self) -> StreamContract {
        StreamContract::new(
            "streaming-pomdp",
            SolverKind::IterativeSolver,
            "Tabular partially-observable MDP assembled by a command stream and solved on \
             demand. The stream declares ⟨S, A, Ω⟩, sets transition/observation rows and \
             rewards, optionally sets a belief, then `solve` streams back the underlying \
             value/policy and the action recommended under the current belief.",
            vec![
                StreamOp::new(
                    "init",
                    "Declare the POMDP sizes. Optional: gamma (default 0.95). Defaults to \
                     stay-in-place transitions, uniform observations, zero reward.",
                    json!({"op":"init","numStates":2,"numActions":3,"numObservations":2,"gamma":0.95}),
                ),
                StreamOp::new(
                    "set_transition",
                    "Set P(s'|s,a) as `probs` (length numStates).",
                    json!({"op":"set_transition","state":0,"action":0,"probs":[1.0,0.0]}),
                ),
                StreamOp::new(
                    "set_observation",
                    "Set P(o|s',a) as `probs` (length numObservations).",
                    json!({"op":"set_observation","nextState":0,"action":0,"probs":[0.85,0.15]}),
                ),
                StreamOp::new(
                    "set_reward",
                    "Set R(s,a).",
                    json!({"op":"set_reward","state":0,"action":2,"reward":10.0}),
                ),
                StreamOp::new(
                    "set_belief",
                    "Set the current belief (length numStates; renormalized).",
                    json!({"op":"set_belief","belief":[0.5,0.5]}),
                ),
                StreamOp::new(
                    "solve",
                    "Solve. Optional: method (qmdp|lookahead|exact-finite-horizon|\
                     most-likely-state), horizon.",
                    json!({"op":"solve","method":"lookahead","horizon":3}),
                ),
            ],
            vec![
                StreamEvent::new("initialized", "POMDP created; reports sizes."),
                StreamEvent::new("applied", "A transition/observation/reward/belief was set."),
                StreamEvent::new(
                    "solution",
                    "Underlying value/policy and the `recommendedAction` under the belief.",
                ),
                StreamEvent::new("error", "A command could not be applied."),
            ],
        )
    }

    fn apply(&mut self, command: &Value) -> Vec<Value> {
        let op = op_of(command);
        match op {
            "init" => return self.init(command),
            "solve" => return self.solve(command),
            _ => {}
        }
        if !self.initialized {
            return vec![error_frame("no POMDP initialized; send {\"op\":\"init\", ...} first")];
        }
        match op {
            "set_transition" => self.set_transition(command),
            "set_observation" => self.set_observation(command),
            "set_reward" => self.set_reward(command),
            "set_belief" => self.set_belief(command),
            other => vec![error_frame(format!("unknown op `{other}` for streaming-pomdp"))],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::streaming::drive;

    #[test]
    fn builds_and_solves_tiger() {
        let mut m = StreamingPomdp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","numStates":2,"numActions":3,"numObservations":2,"gamma":0.95}),
                // transitions: listen stays; open-* reset to 50/50.
                json!({"op":"set_transition","state":0,"action":0,"probs":[1.0,0.0]}),
                json!({"op":"set_transition","state":1,"action":0,"probs":[0.0,1.0]}),
                json!({"op":"set_transition","state":0,"action":1,"probs":[0.5,0.5]}),
                json!({"op":"set_transition","state":1,"action":1,"probs":[0.5,0.5]}),
                json!({"op":"set_transition","state":0,"action":2,"probs":[0.5,0.5]}),
                json!({"op":"set_transition","state":1,"action":2,"probs":[0.5,0.5]}),
                // observations under listen.
                json!({"op":"set_observation","nextState":0,"action":0,"probs":[0.85,0.15]}),
                json!({"op":"set_observation","nextState":1,"action":0,"probs":[0.15,0.85]}),
                // rewards.
                json!({"op":"set_reward","state":0,"action":0,"reward":-1.0}),
                json!({"op":"set_reward","state":1,"action":0,"reward":-1.0}),
                json!({"op":"set_reward","state":0,"action":1,"reward":-100.0}),
                json!({"op":"set_reward","state":0,"action":2,"reward":10.0}),
                json!({"op":"set_reward","state":1,"action":1,"reward":10.0}),
                json!({"op":"set_reward","state":1,"action":2,"reward":-100.0}),
                json!({"op":"set_belief","belief":[0.5,0.5]}),
                json!({"op":"solve","method":"qmdp"}),
            ],
        );
        let sol = frames.last().unwrap();
        assert_eq!(sol["event"], json!("solution"));
        // Uniform belief on tiger: listen (action 0).
        assert_eq!(sol["recommendedAction"].as_u64().unwrap(), 0);
    }

    #[test]
    fn solve_before_init_errors() {
        let mut m = StreamingPomdp::new();
        let frames = drive(&mut m, &[json!({"op":"solve"})]);
        assert_eq!(frames[0]["event"], json!("error"));
    }
}
