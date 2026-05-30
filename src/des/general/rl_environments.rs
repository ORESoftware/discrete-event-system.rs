//! Port of `src/des/general/rl-environments.ts` — small reinforcement-learning
//! environments (`GridWorld`, `Corridor`) shared by the RL agents
//! (`qlearning-des.ts`, `ppo-des.ts`).
//!
//! Both environments implement the same minimal interface:
//!
//!   reset() -> starting state
//!   step(s, a) -> {next_state, reward, done}
//!   num_states / num_actions
//!
//! Environments are PURE (no DES); the DES wrapping happens in the
//! per-algorithm files via an `EnvironmentStation` that routes Action tokens
//! through `step` and emits Transition tokens.
//!
//! Declarations -> Rust:
//!   * `interface Environment`                       -> `trait Environment`
//!   * `class GridWorld` / `class Corridor`          -> structs + `impl Environment`
//!   * the inline `{nextState, reward, done}` record -> [`StepOutcome`]
//!   * the inline `{V, pi}` record                   -> [`OptimalValue`]
//!   * `function evalPolicy`                         -> free fn [`eval_policy`]
//!
//! Conversion notes:
//!   * states/actions are `usize`; rewards are `f64`.
//!   * `render?` (optional method) -> `fn render(&self, s: usize) -> Option<String>`
//!     with a default of `None` on the trait.
//!   * `evalPolicy`'s ambient `Math.random` default RNG is replaced by an
//!     injected [`RandomSource`] per the migration "inject capabilities" rule;
//!     callers pass [`SeededRandom`](crate::des::shared::capabilities::SeededRandom)
//!     for reproducibility.

use std::collections::HashSet;

use crate::des::shared::capabilities::RandomSource;

/// Result of a single environment transition. Replaces the inline TS record
/// `{nextState: number; reward: number; done: boolean}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub next_state: usize,
    pub reward: f64,
    pub done: bool,
}

/// Optimal value function / greedy policy from value iteration. Replaces the
/// inline TS record `{V: number[]; pi: number[]}`.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimalValue {
    pub v: Vec<f64>,
    pub pi: Vec<usize>,
}

/// Minimal environment contract shared by the RL agents.
///
/// In TS `numStates` / `numActions` are fields; here they are accessor methods.
pub trait Environment {
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn reset(&self) -> usize;
    fn step(&self, state: usize, action: usize) -> StepOutcome;

    /// Optional: render an ASCII view of a state (used for debug output).
    /// Defaults to `None` (mirrors the optional `render?` TS method).
    fn render(&self, _state: usize) -> Option<String> {
        None
    }
}

// -----------------------------------------------------------------------------
// 4x4 GRIDWORLD
// -----------------------------------------------------------------------------
//
//   . . . .         start at top-left, goal at bottom-right.
//   . X . .         X = pit (terminal, large negative reward).
//   . . X .
//   . . . G         actions: 0=up, 1=right, 2=down, 3=left.
//
// Reward: -1 per step (encourages short paths); -10 in a pit; +10 at goal.
// -----------------------------------------------------------------------------

/// Options for constructing a [`GridWorld`] (mirrors the TS options object).
/// `None` fields fall back to the TS defaults.
#[derive(Clone, Debug, Default)]
pub struct GridWorldOptions {
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub start: Option<usize>,
    pub goal: Option<usize>,
    pub pits: Option<Vec<usize>>,
}

pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub num_states: usize,
    pub num_actions: usize,
    pub start: usize,
    pub goal: usize,
    pub pits: HashSet<usize>,
}

impl GridWorld {
    /// Action displacements: 0=up, 1=right, 2=down, 3=left.
    pub const DR: [i64; 4] = [-1, 0, 1, 0];
    pub const DC: [i64; 4] = [0, 1, 0, -1];

    pub fn new(opts: GridWorldOptions) -> Self {
        let width = opts.width.unwrap_or(4);
        let height = opts.height.unwrap_or(4);
        let num_states = width * height;
        let start = opts.start.unwrap_or(0);
        let goal = opts.goal.unwrap_or(num_states - 1);
        let pits: HashSet<usize> = opts.pits.unwrap_or_else(|| vec![5, 10]).into_iter().collect();
        GridWorld {
            width,
            height,
            num_states,
            num_actions: 4,
            start,
            goal,
            pits,
        }
    }

    /// Iterate the gridworld's true Bellman optimal V* via value iteration.
    /// Used for validation. (TS defaults: gamma=0.95, tol=1e-9, max_iters=5000.)
    pub fn optimal_v(&self, gamma: f64, tol: f64, max_iters: usize) -> OptimalValue {
        let mut v = vec![0.0_f64; self.num_states];
        let mut pi = vec![0_usize; self.num_states];
        for _ in 0..max_iters {
            let mut max_delta = 0.0_f64;
            for s in 0..self.num_states {
                if s == self.goal || self.pits.contains(&s) {
                    continue;
                }
                let mut best_q = f64::NEG_INFINITY;
                let mut best_a = 0_usize;
                for a in 0..self.num_actions {
                    let o = self.step(s, a);
                    let q = o.reward + if o.done { 0.0 } else { gamma * v[o.next_state] };
                    if q > best_q {
                        best_q = q;
                        best_a = a;
                    }
                }
                let delta = (best_q - v[s]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                v[s] = best_q;
                pi[s] = best_a;
            }
            if max_delta < tol {
                break;
            }
        }
        OptimalValue { v, pi }
    }
}

impl Environment for GridWorld {
    fn num_states(&self) -> usize {
        self.num_states
    }

    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn reset(&self) -> usize {
        self.start
    }

    fn step(&self, state: usize, action: usize) -> StepOutcome {
        if state == self.goal || self.pits.contains(&state) {
            // Terminal absorbing — bug guard: reset on next call.
            return StepOutcome {
                next_state: state,
                reward: 0.0,
                done: true,
            };
        }
        let r = (state / self.width) as i64;
        let c = (state % self.width) as i64;
        let dr = GridWorld::DR[action];
        let dc = GridWorld::DC[action];
        let mut nr = r + dr;
        let mut nc = c + dc;
        // Clamp at walls (no-op move).
        if nr < 0 || nr >= self.height as i64 || nc < 0 || nc >= self.width as i64 {
            nr = r;
            nc = c;
        }
        let ns = (nr * self.width as i64 + nc) as usize;
        if ns == self.goal {
            return StepOutcome {
                next_state: ns,
                reward: 10.0,
                done: true,
            };
        }
        if self.pits.contains(&ns) {
            return StepOutcome {
                next_state: ns,
                reward: -10.0,
                done: true,
            };
        }
        StepOutcome {
            next_state: ns,
            reward: -1.0,
            done: false,
        }
    }

    fn render(&self, state: usize) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        for r in 0..self.height {
            let mut row: Vec<String> = Vec::new();
            for c in 0..self.width {
                let idx = r * self.width + c;
                if idx == state {
                    row.push("A".to_string());
                } else if idx == self.goal {
                    row.push("G".to_string());
                } else if self.pits.contains(&idx) {
                    row.push("X".to_string());
                } else {
                    row.push(".".to_string());
                }
            }
            lines.push(row.join(" "));
        }
        Some(lines.join("\n"))
    }
}

// -----------------------------------------------------------------------------
// 1-D CORRIDOR (length N)
// -----------------------------------------------------------------------------
//
//   o-o-o-o-o-o-G        start at 0, goal at N-1, only two actions: left/right.
//
// Reward: -1 per step, +10 at goal. Optimal value V*(s) = 10*gamma^(N-1-s) - sum.
// Useful for PPO since the action-value gap is monotone — easy to learn.
// -----------------------------------------------------------------------------

pub struct Corridor {
    pub num_states: usize,
    pub num_actions: usize,
    pub start: usize,
    pub goal: usize,
}

impl Corridor {
    /// TS defaults: `length = 8`, `start = 0`.
    pub fn new(length: usize, start: usize) -> Self {
        Corridor {
            num_states: length,
            num_actions: 2, // 0 = left, 1 = right
            start,
            goal: length - 1,
        }
    }

    /// TS defaults: gamma=0.95, tol=1e-9, max_iters=5000.
    pub fn optimal_v(&self, gamma: f64, tol: f64, max_iters: usize) -> OptimalValue {
        let mut v = vec![0.0_f64; self.num_states];
        let mut pi = vec![0_usize; self.num_states];
        for _ in 0..max_iters {
            let mut max_delta = 0.0_f64;
            for s in 0..self.num_states {
                if s == self.goal {
                    continue;
                }
                let mut best_q = f64::NEG_INFINITY;
                let mut best_a = 0_usize;
                for a in 0..self.num_actions {
                    let o = self.step(s, a);
                    let q = o.reward + if o.done { 0.0 } else { gamma * v[o.next_state] };
                    if q > best_q {
                        best_q = q;
                        best_a = a;
                    }
                }
                let delta = (best_q - v[s]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                v[s] = best_q;
                pi[s] = best_a;
            }
            if max_delta < tol {
                break;
            }
        }
        OptimalValue { v, pi }
    }
}

impl Environment for Corridor {
    fn num_states(&self) -> usize {
        self.num_states
    }

    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn reset(&self) -> usize {
        self.start
    }

    fn step(&self, state: usize, action: usize) -> StepOutcome {
        if state == self.goal {
            return StepOutcome {
                next_state: state,
                reward: 0.0,
                done: true,
            };
        }
        // `state - 1` can underflow at state 0; compute with the wall clamp.
        let mut ns: i64 = if action == 0 {
            state as i64 - 1
        } else {
            state as i64 + 1
        };
        if ns < 0 {
            ns = 0;
        }
        if ns >= self.num_states as i64 {
            ns = self.num_states as i64 - 1;
        }
        let ns = ns as usize;
        if ns == self.goal {
            return StepOutcome {
                next_state: ns,
                reward: 10.0,
                done: true,
            };
        }
        StepOutcome {
            next_state: ns,
            reward: -1.0,
            done: false,
        }
    }

    fn render(&self, state: usize) -> Option<String> {
        let mut cells: Vec<String> = Vec::new();
        for i in 0..self.num_states {
            cells.push(if i == state {
                "A".to_string()
            } else if i == self.goal {
                "G".to_string()
            } else {
                "o".to_string()
            });
        }
        Some(cells.join("\u{2500}"))
    }
}

// -----------------------------------------------------------------------------
// EVALUATE A POLICY by Monte-Carlo rollouts.
// -----------------------------------------------------------------------------

/// Options for [`eval_policy`] (mirrors the TS options object). `rng` is no
/// longer part of this struct — it is injected as a [`RandomSource`] argument.
#[derive(Clone, Copy, Debug)]
pub struct EvalPolicyOptions {
    pub num_episodes: usize,
    pub max_steps_per_episode: usize,
    pub gamma: f64,
}

impl Default for EvalPolicyOptions {
    fn default() -> Self {
        EvalPolicyOptions {
            num_episodes: 100,
            max_steps_per_episode: 200,
            gamma: 1.0,
        }
    }
}

/// Aggregate statistics from policy evaluation. Replaces the inline TS record
/// `{meanReturn, meanLength, successRate}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalPolicyResult {
    pub mean_return: f64,
    pub mean_length: f64,
    pub success_rate: f64,
}

/// Evaluate `pick_action` against `env` by Monte-Carlo rollouts.
///
/// `pick_action(state, rng)` chooses an action; the injected `rng` replaces the
/// TS `() => number` (defaulted to `Math.random`).
pub fn eval_policy(
    env: &dyn Environment,
    mut pick_action: impl FnMut(usize, &mut dyn RandomSource) -> usize,
    rng: &mut dyn RandomSource,
    opts: EvalPolicyOptions,
) -> EvalPolicyResult {
    let n = opts.num_episodes;
    let max_steps = opts.max_steps_per_episode;
    let gamma = opts.gamma;
    let mut total_return = 0.0_f64;
    let mut total_len = 0.0_f64;
    let mut successes = 0_usize;
    for _ in 0..n {
        let mut s = env.reset();
        let mut g_return = 0.0_f64;
        let mut len = 0_usize;
        let mut g = 1.0_f64;
        let mut done = false;
        while !done && len < max_steps {
            let a = pick_action(s, rng);
            let r = env.step(s, a);
            g_return += g * r.reward;
            g *= gamma;
            s = r.next_state;
            done = r.done;
            len += 1;
        }
        total_return += g_return;
        total_len += len as f64;
        if done && len < max_steps {
            successes += 1;
        }
    }
    EvalPolicyResult {
        mean_return: total_return / n as f64,
        mean_length: total_len / n as f64,
        success_rate: successes as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    /// Tiny concrete `Environment`: a two-state chain where action 1 advances to
    /// the (terminal) goal with reward +1, action 0 stays put with reward 0.
    struct TwoStateEnv;

    impl Environment for TwoStateEnv {
        fn num_states(&self) -> usize {
            2
        }
        fn num_actions(&self) -> usize {
            2
        }
        fn reset(&self) -> usize {
            0
        }
        fn step(&self, state: usize, action: usize) -> StepOutcome {
            if state == 1 {
                return StepOutcome {
                    next_state: 1,
                    reward: 0.0,
                    done: true,
                };
            }
            if action == 1 {
                StepOutcome {
                    next_state: 1,
                    reward: 1.0,
                    done: true,
                }
            } else {
                StepOutcome {
                    next_state: 0,
                    reward: 0.0,
                    done: false,
                }
            }
        }
    }

    #[test]
    fn tiny_environment_step_and_eval() {
        let env = TwoStateEnv;
        // Greedy policy always advances; every episode succeeds in one step.
        let mut rng = SeededRandom::new(1);
        let res = eval_policy(
            &env,
            |_s, _rng| 1,
            &mut rng,
            EvalPolicyOptions {
                num_episodes: 10,
                max_steps_per_episode: 5,
                gamma: 1.0,
            },
        );
        assert_eq!(res.mean_return, 1.0);
        assert_eq!(res.mean_length, 1.0);
        assert_eq!(res.success_rate, 1.0);
    }

    #[test]
    fn gridworld_step_and_value_iteration() {
        let gw = GridWorld::new(GridWorldOptions::default());
        assert_eq!(gw.num_states(), 16);
        assert_eq!(gw.num_actions(), 4);
        assert_eq!(gw.reset(), 0);

        // Stepping into the goal (state 15) from state 14 by moving right.
        let o = gw.step(14, 1);
        assert_eq!(o.next_state, 15);
        assert_eq!(o.reward, 10.0);
        assert!(o.done);

        // Moving up from the top row is a no-op (wall clamp), reward -1.
        let o2 = gw.step(0, 0);
        assert_eq!(o2.next_state, 0);
        assert_eq!(o2.reward, -1.0);
        assert!(!o2.done);

        // Value iteration: start's optimal value should be positive (goal is
        // reachable) and the greedy action should avoid the pit at 5.
        let opt = gw.optimal_v(0.95, 1e-9, 5000);
        assert_eq!(opt.v.len(), 16);
        assert!(opt.v[gw.start] > 0.0);
        assert!(gw.render(0).is_some());
    }

    #[test]
    fn corridor_reaches_goal() {
        let c = Corridor::new(5, 0);
        assert_eq!(c.num_states(), 5);
        assert_eq!(c.num_actions(), 2);

        // Moving right four times reaches the goal at index 4.
        let mut s = c.reset();
        let mut last = StepOutcome {
            next_state: s,
            reward: 0.0,
            done: false,
        };
        for _ in 0..4 {
            last = c.step(s, 1);
            s = last.next_state;
        }
        assert_eq!(last.next_state, 4);
        assert_eq!(last.reward, 10.0);
        assert!(last.done);

        // Moving left at state 0 clamps (stays at 0).
        let o = c.step(0, 0);
        assert_eq!(o.next_state, 0);
        assert!(!o.done);
    }
}
