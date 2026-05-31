//! Streaming Markov decision process — tabular, build-then-solve.
//!
//! Wraps [`value_iteration`](crate::des::general::value_iteration::value_iteration).
//! A command stream builds a tabular MDP — declare the state count, set the
//! probabilistic outcomes of each (state, action), mark terminal states — and a
//! `solve` command runs value iteration and streams back the optimal value
//! function and greedy policy. The model can be edited (new transitions, new
//! terminal rewards) and re-solved as the stream continues.
//!
//! `MDPSpec` is closure-based; this streamer captures the tabular data in `Rc`
//! and builds the closures on each solve, so the existing solver is reused
//! verbatim. Value iteration validates probabilities and may panic, so each
//! solve runs under `catch_unwind` and reports an `error` frame instead.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use serde_json::{json, Value};

use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};

use super::{
    error_frame, f64_at, op_of, usize_at, SolverKind, StreamContract, StreamEvent, StreamOp,
    StreamingModel,
};

/// One outcome row stored tabularly: `(prob, reward, next_state)`.
type Row = (f64, f64, usize);

/// A tabular MDP assembled by a JSONL command stream and solved on demand.
pub struct StreamingMdp {
    num_states: usize,
    /// `transitions[state][action]` -> outcomes.
    transitions: Vec<Vec<Vec<Row>>>,
    /// `terminal[state]` = Some(reward) marks an absorbing terminal state.
    terminal: Vec<Option<f64>>,
    gamma: f64,
    initialized: bool,
}

impl Default for StreamingMdp {
    fn default() -> Self {
        StreamingMdp {
            num_states: 0,
            transitions: Vec::new(),
            terminal: Vec::new(),
            gamma: 0.95,
            initialized: false,
        }
    }
}

impl StreamingMdp {
    pub fn new() -> Self {
        Self::default()
    }

    fn parse_outcomes(command: &Value) -> Option<Vec<Row>> {
        command.get("outcomes")?.as_array().map(|items| {
            items
                .iter()
                .map(|o| {
                    let prob = o.get("prob").and_then(Value::as_f64).unwrap_or(0.0);
                    let reward = o.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
                    let next = o
                        .get("next")
                        .and_then(Value::as_u64)
                        .map(|n| n as usize)
                        .unwrap_or(0);
                    (prob, reward, next)
                })
                .collect::<Vec<Row>>()
        })
    }

    fn init(&mut self, command: &Value) -> Vec<Value> {
        let num_states = match usize_at(command, "numStates") {
            Some(n) if n > 0 => n,
            _ => return vec![error_frame("`numStates` (positive integer) is required")],
        };
        self.num_states = num_states;
        self.transitions = vec![Vec::new(); num_states];
        self.terminal = vec![None; num_states];
        self.gamma = f64_at(command, "gamma", 0.95);
        self.initialized = true;
        vec![json!({
            "event": "initialized",
            "numStates": num_states,
            "gamma": self.gamma,
        })]
    }

    fn set_transition(&mut self, command: &Value) -> Vec<Value> {
        let state = match usize_at(command, "state") {
            Some(s) if s < self.num_states => s,
            _ => return vec![error_frame("`state` out of range")],
        };
        let action = match usize_at(command, "action") {
            Some(a) => a,
            None => return vec![error_frame("`action` (usize) is required")],
        };
        let outcomes = match Self::parse_outcomes(command) {
            Some(o) => o,
            None => return vec![error_frame("`outcomes` array is required")],
        };
        if outcomes.iter().any(|&(_, _, next)| next >= self.num_states) {
            return vec![error_frame("an outcome `next` is out of range")];
        }
        if action >= self.transitions[state].len() {
            self.transitions[state].resize(action + 1, Vec::new());
        }
        self.transitions[state][action] = outcomes;
        vec![json!({
            "event": "applied",
            "op": "set_transition",
            "state": state,
            "action": action,
            "numActions": self.transitions[state].len(),
        })]
    }

    fn set_terminal(&mut self, command: &Value) -> Vec<Value> {
        let state = match usize_at(command, "state") {
            Some(s) if s < self.num_states => s,
            _ => return vec![error_frame("`state` out of range")],
        };
        let reward = f64_at(command, "reward", 0.0);
        self.terminal[state] = Some(reward);
        vec![json!({"event":"applied","op":"set_terminal","state":state,"reward":reward})]
    }

    fn solve(&mut self, command: &Value) -> Vec<Value> {
        if !self.initialized {
            return vec![error_frame("no MDP initialized; send {\"op\":\"init\", ...} first")];
        }
        let gamma = f64_at(command, "gamma", self.gamma);
        let num_states = self.num_states;
        let transitions = Rc::new(self.transitions.clone());
        let terminal = Rc::new(self.terminal.clone());

        let result = catch_unwind(AssertUnwindSafe(|| {
            let spec = MDPSpec {
                num_states,
                num_actions: {
                    let t = transitions.clone();
                    Box::new(move |s: usize| t.get(s).map(|av| av.len()).unwrap_or(0))
                },
                outcomes: {
                    let t = transitions.clone();
                    Box::new(move |s: usize, a: usize| {
                        t.get(s)
                            .and_then(|av| av.get(a))
                            .map(|outs| {
                                outs.iter()
                                    .map(|&(prob, reward, next_state)| Outcome {
                                        prob,
                                        reward,
                                        next_state,
                                    })
                                    .collect::<Vec<Outcome>>()
                            })
                            .unwrap_or_default()
                    })
                },
                is_terminal: {
                    let term = terminal.clone();
                    Some(Box::new(move |s: usize| {
                        term.get(s).map(|o| o.is_some()).unwrap_or(false)
                    }))
                },
                terminal_reward: {
                    let term = terminal.clone();
                    Some(Box::new(move |s: usize| {
                        term.get(s).and_then(|o| *o).unwrap_or(0.0)
                    }))
                },
                state_label: None,
                action_label: None,
            };
            let opts = VIOptions {
                gamma,
                ..VIOptions::default()
            };
            value_iteration(spec, opts)
        }));

        match result {
            Ok(res) => vec![json!({
                "event": "solution",
                "gamma": res.gamma,
                "iterations": res.iterations,
                "finalDelta": res.final_delta,
                "v": res.v,
                "policy": res.policy,
            })],
            Err(_) => vec![error_frame(
                "value iteration failed (check that outcome probabilities sum to 1)",
            )],
        }
    }
}

impl StreamingModel for StreamingMdp {
    fn kind(&self) -> SolverKind {
        SolverKind::IterativeSolver
    }

    fn contract(&self) -> StreamContract {
        StreamContract::new(
            "streaming-mdp",
            SolverKind::IterativeSolver,
            "Tabular Markov decision process assembled by a command stream and \
             solved on demand by value iteration. The stream declares states, \
             sets per-(state,action) probabilistic outcomes and terminal rewards, \
             then `solve` streams back the optimal value function and policy.",
            vec![
                StreamOp::new(
                    "init",
                    "Declare the MDP. Optional: gamma (default 0.95).",
                    json!({"op":"init","numStates":2,"gamma":0.9}),
                ),
                StreamOp::new(
                    "set_transition",
                    "Set outcomes for taking `action` in `state`; outcomes is an \
                     array of {prob, reward, next}. Probabilities should sum to 1.",
                    json!({"op":"set_transition","state":0,"action":0,
                           "outcomes":[{"prob":1.0,"reward":0.0,"next":1}]}),
                ),
                StreamOp::new(
                    "set_terminal",
                    "Mark `state` absorbing/terminal with a pinned `reward`.",
                    json!({"op":"set_terminal","state":1,"reward":1.0}),
                ),
                StreamOp::new(
                    "solve",
                    "Run value iteration. Optional: gamma override.",
                    json!({"op":"solve","gamma":0.9}),
                ),
            ],
            vec![
                StreamEvent::new("initialized", "MDP created; reports state count."),
                StreamEvent::new("applied", "A transition/terminal was set."),
                StreamEvent::new(
                    "solution",
                    "Optimal value `v` and greedy `policy` (per state).",
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
            return vec![error_frame("no MDP initialized; send {\"op\":\"init\", ...} first")];
        }
        match op {
            "set_transition" => self.set_transition(command),
            "set_terminal" => self.set_terminal(command),
            other => vec![error_frame(format!("unknown op `{other}` for streaming-mdp"))],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::streaming::drive;

    #[test]
    fn solves_two_state_chain() {
        // s0 --(a0, r=0)--> s1 (terminal, reward 1). gamma 0.9.
        // V[s1] = 1, V[s0] = 0.9 * 1 = 0.9, policy[s1] = -1.
        let mut m = StreamingMdp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","numStates":2,"gamma":0.9}),
                json!({"op":"set_transition","state":0,"action":0,
                       "outcomes":[{"prob":1.0,"reward":0.0,"next":1}]}),
                json!({"op":"set_terminal","state":1,"reward":1.0}),
                json!({"op":"solve"}),
            ],
        );
        let sol = frames.last().unwrap();
        assert_eq!(sol["event"], json!("solution"));
        let v = sol["v"].as_array().unwrap();
        assert!((v[1].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((v[0].as_f64().unwrap() - 0.9).abs() < 1e-6);
        let policy = sol["policy"].as_array().unwrap();
        assert_eq!(policy[1].as_i64().unwrap(), -1);
        assert_eq!(policy[0].as_i64().unwrap(), 0);
    }

    #[test]
    fn solve_before_init_errors() {
        let mut m = StreamingMdp::new();
        let frames = drive(&mut m, &[json!({"op":"solve"})]);
        assert_eq!(frames[0]["event"], json!("error"));
    }
}
