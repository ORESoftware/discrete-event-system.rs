//! Port of `src/des/general/des-base/monte-carlo-rl.ts`.
//!
//! On-policy MONTE CARLO CONTROL (Sutton & Barto §5.4): every-visit and
//! first-visit Monte Carlo estimation of `Q*`, with ε-greedy policy
//! improvement. The agent rolls out an entire episode under its ε-soft policy,
//! collects `{(s_t, a_t, r_{t+1})}`, then computes the discounted return
//! `G_t = sum_{k>=0} gamma^k r_{t+k+1}`
//! and updates `Q[s][a]` toward the empirical return either every visit or
//! only on the first visit per `(s, a)` within the episode, using INCREMENTAL
//! averaging: `N(s,a) += 1; Q(s,a) += (G - Q(s,a)) / N(s,a)`.
//!
//! ## Template-method mapping (TS `class … extends RLAgentStation` → Rust)
//!
//! TypeScript modelled this as `class MonteCarloAgent extends
//! RLAgentStation<number, number>`. Rust has no inheritance, so the concrete
//! agent EMBEDS a [`StationCore`] and an [`RLAgentCore`] and implements both
//! [`DESStation`] (delegating `run_time_step` → `rl_agent_run_time_step`,
//! `has_work` → `rl_agent_has_work`) and the
//! [`RLAgentStation`](crate::des::general::des_base::rl_agent::RLAgentStation)
//! hook trait.
//!
//! Hooks implemented:
//!   * `pick_action` — ε-greedy first-max over `Q` (faithful: no tie-break).
//!   * `update`      — APPENDS the transition to the in-progress trajectory and
//!                     applies the episode on `done`. No online Q update.
//!   * `end_of_episode` — decays ε.
//!
//! `Q: Float64Array` / `visitCount: Int32Array` (flat `N×A`) become `Vec<f64>`
//! / `Vec<i32>` indexed `s*A + a`. The first-visit dedup `Set<number>` becomes
//! a `HashSet<usize>`. The parallel trajectory arrays become `Vec<usize>` /
//! `Vec<usize>` / `Vec<f64>`. The injected `rng: () => number` is the boxed
//! [`RandomSource`] held by [`RLAgentCore`].

use std::any::Any;
use std::collections::HashSet;

use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::shared::capabilities::RandomSource;

/// Configuration for [`MonteCarloAgent`]. Mirrors the TS `MonteCarloOptions`
/// interface; the injected `rng` is barred from the struct (passed separately
/// to [`MonteCarloAgent::new`]) so the rest can derive [`Default`]. Optional
/// fields are `None` ⇒ the TS `??` default is applied in the constructor.
#[derive(Default)]
pub struct MonteCarloOptions {
    pub num_states: usize,
    pub num_actions: usize,
    /// First-visit (`true`) vs every-visit (`false`). Default `true`.
    pub first_visit: Option<bool>,
    /// Discount γ. Default `1.0` (canonical for episodic MC).
    pub gamma: Option<f64>,
    /// Exploration ε. Default `0.1`.
    pub epsilon: Option<f64>,
    /// ε-decay multiplier per episode. Default `1`.
    pub epsilon_decay: Option<f64>,
    /// ε-floor. Default `0.01`.
    pub epsilon_min: Option<f64>,
    /// Initial Q value (broadcast). Default `0`.
    pub init_q: Option<f64>,
}

/// On-policy Monte Carlo control agent over a discrete `N×A` table.
pub struct MonteCarloAgent {
    core: StationCore,
    agent: RLAgentCore,
    /// Number of states (configuration; tables are sized from it at construction).
    #[allow(dead_code)]
    n: usize,
    /// Number of actions.
    a: usize,
    /// Action-value table, flat `N × A` indexed `s*A + a`.
    q: Vec<f64>,
    /// Visit counts, flat `N × A` indexed `s*A + a`.
    visit_count: Vec<i32>,
    first_visit: bool,
    gamma: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    /// Per-episode trajectory: parallel arrays of states, actions, rewards.
    traj_s: Vec<usize>,
    traj_a: Vec<usize>,
    traj_r: Vec<f64>,
}

impl MonteCarloAgent {
    /// Mirrors `new MonteCarloAgent(id, opts)`.
    pub fn new(id: &str, rng: Box<dyn RandomSource>, opts: MonteCarloOptions) -> Self {
        let n = opts.num_states;
        let a = opts.num_actions;
        let init_q = opts.init_q.unwrap_or(0.0);
        let mut q = vec![0.0; n * a];
        // TS: `if (opts.initQ) this.Q.fill(opts.initQ)` — truthy ⇒ non-zero.
        if init_q != 0.0 {
            for x in q.iter_mut() {
                *x = init_q;
            }
        }
        MonteCarloAgent {
            core: StationCore::new(id),
            agent: RLAgentCore::new(rng),
            n,
            a,
            q,
            visit_count: vec![0; n * a],
            first_visit: opts.first_visit.unwrap_or(true),
            gamma: opts.gamma.unwrap_or(1.0),
            epsilon: opts.epsilon.unwrap_or(0.1),
            epsilon_decay: opts.epsilon_decay.unwrap_or(1.0),
            epsilon_min: opts.epsilon_min.unwrap_or(0.01),
            traj_s: Vec::new(),
            traj_a: Vec::new(),
            traj_r: Vec::new(),
        }
    }

    /// Apply Monte Carlo first-visit / every-visit updates over the trajectory
    /// just collected, then reset the buffer.
    fn apply_episode(&mut self) {
        let t = self.traj_s.len();
        let mut seen: HashSet<usize> = HashSet::new();
        // Compute returns from the back.
        let mut g = 0.0;
        for idx in (0..t).rev() {
            g = self.gamma * g + self.traj_r[idx];
            let s = self.traj_s[idx];
            let a = self.traj_a[idx];
            let key = s * self.a + a;
            if self.first_visit && seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            self.visit_count[key] += 1;
            self.q[key] += (g - self.q[key]) / self.visit_count[key] as f64;
        }
        self.traj_s.clear();
        self.traj_a.clear();
        self.traj_r.clear();
    }

    /// Argmax over Q (greedy, no exploration).
    pub fn greedy_action(&self, state: usize) -> usize {
        let off = state * self.a;
        let mut best_a = 0usize;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..self.a {
            let q = self.q[off + a];
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        best_a
    }

    // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

    pub fn get_q(&self) -> &[f64] {
        &self.q
    }
    pub fn get_visit_counts(&self) -> &[i32] {
        &self.visit_count
    }
    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl DESStation for MonteCarloAgent {
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
        self.rl_agent_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.rl_agent_has_work()
    }
}

impl RLAgentStation<usize, usize> for MonteCarloAgent {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }

    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        if rng.next_float() < self.epsilon {
            return (rng.next_float() * self.a as f64).floor() as usize;
        }
        let off = state * self.a;
        let mut best_a = 0usize;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..self.a {
            let q = self.q[off + a];
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        best_a
    }

    /// Each transition just APPENDS to the in-progress trajectory; no Q update
    /// happens until `done`.
    fn update(
        &mut self,
        state: &usize,
        action: &usize,
        reward: f64,
        _next_state: &usize,
        done: bool,
    ) {
        self.traj_s.push(*state);
        self.traj_a.push(*action);
        self.traj_r.push(reward);
        if done {
            self.apply_episode();
        }
    }

    fn end_of_episode(&mut self, _episode_id: f64) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::rl_tokens::TransitionToken;
    use crate::des::shared::capabilities::SeededRandom;
    use std::rc::Rc;

    fn agent(seed: u32, n: usize, a: usize, first_visit: bool) -> MonteCarloAgent {
        MonteCarloAgent::new(
            "mc",
            Box::new(SeededRandom::new(seed)),
            MonteCarloOptions {
                num_states: n,
                num_actions: a,
                first_visit: Some(first_visit),
                gamma: Some(1.0),
                epsilon: Some(0.0),
                ..Default::default()
            },
        )
    }

    /// Single-step episodes (a contextual bandit): action 1 yields reward +1,
    /// action 0 yields 0. Each `done` transition is a one-step episode whose
    /// return G = r is averaged into Q. After many episodes the empirical
    /// average makes `Q[1] > Q[0]`, so greedy picks action 1.
    #[test]
    fn mc_control_learns_better_action() {
        let mut a = agent(1, 1, 2, true);
        for ep in 0..200 {
            for action in 0..2usize {
                let reward = if action == 1 { 1.0 } else { 0.0 };
                let t = TransitionToken::new(0usize, action, reward, 0usize, true, ep as f64);
                a.core_mut()
                    .take(Rc::new(t), MonteCarloAgent::CH_TRANSITION);
                a.run_time_step();
            }
        }
        let q = a.get_q();
        assert!(q[1] > q[0], "Q={:?}", q);
        assert_eq!(a.greedy_action(0), 1);
        // Returns averaged to the true means: Q[0]≈0, Q[1]≈1.
        assert!((q[1] - 1.0).abs() < 1e-9, "Q[1]={}", q[1]);
        // 200 episodes × 2 actions terminal steps.
        assert_eq!(a.total_steps(), 400);
    }

    /// A single 2-step episode `(s0,a0,r=0) → (s1,a1,r=1) → done`. With γ=1 the
    /// return is G=1 at BOTH steps, so first visit of each `(s,a)` lands Q=1.
    #[test]
    fn mc_computes_multi_step_return() {
        let mut a = agent(2, 2, 2, true);
        // Step 1: not done — buffered, then template acts on next_state.
        let t1 = TransitionToken::new(0usize, 0usize, 0.0, 1usize, false, 0.0);
        a.core_mut()
            .take(Rc::new(t1), MonteCarloAgent::CH_TRANSITION);
        a.run_time_step();
        // Step 2: done — applies the episode return over the buffered trajectory.
        let t2 = TransitionToken::new(1usize, 1usize, 1.0, 1usize, true, 0.0);
        a.core_mut()
            .take(Rc::new(t2), MonteCarloAgent::CH_TRANSITION);
        a.run_time_step();

        let q = a.get_q();
        // key(s=0,a=0)=0 ; key(s=1,a=1)=1*2+1=3.
        assert!((q[0] - 1.0).abs() < 1e-12, "Q[0,0]={}", q[0]);
        assert!((q[3] - 1.0).abs() < 1e-12, "Q[1,1]={}", q[3]);
        assert_eq!(a.get_visit_counts()[0], 1);
        assert_eq!(a.get_visit_counts()[3], 1);
    }
}
