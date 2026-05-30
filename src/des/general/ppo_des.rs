//! Port of `src/des/general/ppo-des.ts` — Proximal Policy Optimization as a DES.
//!
//! Two concrete leaves on the policy-gradient bases:
//!
//!   * [`TabularPPOAgent`] (on [`PolicyGradientAgent`]): hook
//!     `sample_policy_and_value` softmax-samples over `θ[s][·]` and reports
//!     `V[s]`.
//!   * [`PPOClipUpdateStation`] (on [`PolicyUpdateStation`]): hook `run_update`
//!     computes GAE advantages + returns-to-go, then runs `K` epochs of the
//!     clipped surrogate (+ optional entropy bonus) and value-MSE SGD on the
//!     agent's tabular `θ` and `V`.
//!
//! Topology: Environment ⇄ TabularPPOAgent → (when buffer full) PPOClipUpdate →
//! Resume → agent.
//!
//! ## TS → Rust mapping
//!
//!   * `class TabularPPOAgent extends PolicyGradientAgent<number, number>` → a
//!     struct embedding [`StationCore`] + [`PolicyGradientCore`] over
//!     `S = usize, A = usize` (the tabular state is a row index), delegating the
//!     template method and implementing `sample_policy_and_value`.
//!   * `class PPOClipUpdateStation extends PolicyUpdateStation` → a struct
//!     embedding [`StationCore`] + [`PolicyUpdateCore`], holding an
//!     `Rc<RefCell<TabularPPOAgent>>` to the shared agent (the TS handle) plus
//!     the [`PPOUpdateOptions`].
//!   * the injected `rng` (one `mulberry32` closure shared by agent, updater
//!     shuffle, and runner) → a single [`SeededRandom`] behind `Rc<RefCell<…>>`,
//!     bridged via [`SharedRng`].
//!   * `fn runPPODES` → the free fn [`run_ppo_des`].

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::environment::{
    self, EnvironmentStation, EnvironmentStationOptions, PureEnvironment,
};
use crate::des::general::des_base::policy_gradient_agent::{
    self, PolicyGradientAgent, PolicyGradientCore, PolicyOutput, PolicyUpdateCore, PolicyUpdateStation,
    RolloutEntry,
};
use crate::des::general::des_base::rl_agent::RngRef;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Bridges one shared [`SeededRandom`] into a boxed [`RandomSource`] (the TS
/// shared `() => number` closure). FLAGGED local equivalent — see the sibling
/// `qlearning_des.rs`.
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

// -----------------------------------------------------------------------------
// TABULAR PPO AGENT
// -----------------------------------------------------------------------------

/// PPO actor/critic with tabular policy logits `θ[s][a]` and value table `V[s]`.
pub struct TabularPPOAgent {
    core: StationCore,
    pg: PolicyGradientCore<usize, usize>,
    /// Policy logits `θ[s][a]`.
    pub theta: Vec<Vec<f64>>,
    /// Value table `V[s]`.
    pub v: Vec<f64>,
    pub num_states: usize,
    pub num_actions: usize,
}

impl TabularPPOAgent {
    pub fn new(
        id: impl Into<String>,
        num_states: usize,
        num_actions: usize,
        rollout_len: usize,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        TabularPPOAgent {
            core: StationCore::new(id),
            pg: PolicyGradientCore::new(rollout_len, rng),
            theta: vec![vec![0.0; num_actions]; num_states],
            v: vec![0.0; num_states],
            num_states,
            num_actions,
        }
    }

    /// Greedy action per state (random tie-break, drawing the agent's RNG).
    pub fn greedy_policy(&mut self) -> Vec<usize> {
        let mut rng = self.pg.rng.take().expect("rng already in use");
        let policy = self
            .theta
            .iter()
            .map(|row| arg_max_with_tie_break(row, &mut RngRef(&mut *rng), ARGMAX_EPS_DEFAULT).unwrap_or(0))
            .collect();
        self.pg.rng = Some(rng);
        policy
    }
}

impl DESStation for TabularPPOAgent {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn run_time_step(&mut self) {
        self.policy_gradient_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.policy_gradient_has_work()
    }
}

impl PolicyGradientAgent<usize, usize> for TabularPPOAgent {
    fn pg_core(&self) -> &PolicyGradientCore<usize, usize> {
        &self.pg
    }
    fn pg_core_mut(&mut self) -> &mut PolicyGradientCore<usize, usize> {
        &mut self.pg
    }

    fn sample_policy_and_value(&self, state: &usize, rng: &mut dyn RandomSource) -> PolicyOutput<usize> {
        let logits = &self.theta[*state];
        let mut m = f64::NEG_INFINITY;
        for &x in logits {
            if x > m {
                m = x;
            }
        }
        let mut z = 0.0;
        for &x in logits {
            z += (x - m).exp();
        }
        let log_z = m + z.ln();
        let u = rng.next_float();
        let mut cum = 0.0;
        let mut a = logits.len() - 1;
        for (i, &l) in logits.iter().enumerate() {
            cum += (l - log_z).exp();
            if u <= cum {
                a = i;
                break;
            }
        }
        PolicyOutput { action: a, log_prob: logits[a] - log_z, value: self.v[*state] }
    }
}

// -----------------------------------------------------------------------------
// PPO CLIP UPDATE STATION
// -----------------------------------------------------------------------------

/// `interface PPOUpdateOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PPOUpdateOptions {
    pub gamma: f64,
    pub lambda: f64,
    pub clip_eps: f64,
    pub policy_lr: f64,
    pub value_lr: f64,
    pub num_epochs: usize,
    pub mini_batch_size: usize,
    pub entropy_coef: Option<f64>,
    /// Normalise advantages within the batch before update. `None` → `true`.
    pub normalise_advantage: Option<bool>,
}

/// GAE advantages + returns-to-go over the rollout buffer. Reads `agent.v` for
/// the bootstrap value of non-terminal next states.
fn compute_advantages(
    agent: &TabularPPOAgent,
    buf: &[RolloutEntry<usize, usize>],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = buf.len();
    let mut adv = vec![0.0; n];
    let mut ret = vec![0.0; n];
    let mut gae = 0.0;
    for t in (0..n).rev() {
        let e = &buf[t];
        // Defensive: a tail entry may still be missing its reward if the rollout
        // ended exactly at buffer-full; treat as r=0, done=true, vNext=0.
        let r = e.r.unwrap_or(0.0);
        let done = e.done.unwrap_or(true);
        let v_next = match e.s_next {
            Some(sn) if !done => agent.v[sn],
            _ => 0.0,
        };
        let delta = r + gamma * v_next - e.v;
        gae = delta + gamma * lambda * if done { 0.0 } else { gae };
        adv[t] = gae;
        ret[t] = adv[t] + e.v;
    }
    (adv, ret)
}

/// Fisher–Yates shuffle in place using the injected RNG.
fn shuffle(arr: &mut [usize], rng: &mut dyn RandomSource) {
    if arr.len() < 2 {
        return;
    }
    let mut i = arr.len() - 1;
    while i > 0 {
        let mut j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
        if j > i {
            j = i;
        }
        arr.swap(i, j);
        i -= 1;
    }
}

/// One mini-batch sample's contribution. Tabular gradient:
///   ∂log π(a|s) / ∂θ[s,a'] = δ(a'=a) − π(a'|s)
/// Clipped surrogate gradient: zero outside the clip region.
fn apply_one_sample_update(
    agent: &mut TabularPPOAgent,
    e: &RolloutEntry<usize, usize>,
    a_adv: f64,
    g: f64,
    opts: &PPOUpdateOptions,
) {
    let s = e.s;
    let a = e.a;
    let logits = &agent.theta[s];
    let mut m = f64::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    let mut z = 0.0;
    for &l in logits {
        z += (l - m).exp();
    }
    let log_z = m + z.ln();
    let log_prob_new = logits[a] - log_z;
    let ratio = (log_prob_new - e.log_prob_old).exp();
    let in_clip = (a_adv >= 0.0 && ratio < 1.0 + opts.clip_eps) || (a_adv < 0.0 && ratio > 1.0 - opts.clip_eps);
    if in_clip {
        let n = agent.theta[s].len();
        for a_prime in 0..n {
            let pi_a_prime = (agent.theta[s][a_prime] - log_z).exp();
            let grad = ratio * a_adv * ((if a_prime == a { 1.0 } else { 0.0 }) - pi_a_prime);
            agent.theta[s][a_prime] += opts.policy_lr * grad;
        }
    }
    let entropy_coef = opts.entropy_coef.unwrap_or(0.0);
    if entropy_coef > 0.0 {
        // Recompute π & H from the (pre-clip-update snapshot) logits captured in
        // `log_z`; the tiny inaccuracy from in-place mutation is acceptable.
        let logits_snapshot: Vec<f64> = agent.theta[s].clone();
        let pi: Vec<f64> = logits_snapshot.iter().map(|&l| (l - log_z).exp()).collect();
        let mut h = 0.0;
        for k in 0..pi.len() {
            h -= pi[k] * (logits_snapshot[k] - log_z);
        }
        for a_prime in 0..pi.len() {
            let grad = pi[a_prime] * (h + (logits_snapshot[a_prime] - log_z));
            agent.theta[s][a_prime] += opts.policy_lr * entropy_coef * grad;
        }
    }
    // Value SGD: V[s] ← V[s] + lr_v · (G − V[s]).
    agent.v[s] += opts.value_lr * (g - agent.v[s]);
}

/// PPO clipped-surrogate update station operating on a shared [`TabularPPOAgent`].
pub struct PPOClipUpdateStation {
    core: StationCore,
    pu: PolicyUpdateCore,
    agent: Rc<RefCell<TabularPPOAgent>>,
    opts: PPOUpdateOptions,
    shuffle_rng: Box<dyn RandomSource>,
}

impl PPOClipUpdateStation {
    pub fn new(
        id: impl Into<String>,
        agent: Rc<RefCell<TabularPPOAgent>>,
        opts: PPOUpdateOptions,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        PPOClipUpdateStation { core: StationCore::new(id), pu: PolicyUpdateCore::new(), agent, opts, shuffle_rng: rng }
    }
}

impl DESStation for PPOClipUpdateStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn run_time_step(&mut self) {
        self.policy_update_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.policy_update_has_work()
    }
}

impl PolicyUpdateStation for PPOClipUpdateStation {
    fn pu_core(&self) -> &PolicyUpdateCore {
        &self.pu
    }
    fn pu_core_mut(&mut self) -> &mut PolicyUpdateCore {
        &mut self.pu
    }

    /// Reads from `agent.buffer`, mutates θ and V, clears the buffer at the end.
    fn run_update(&mut self) {
        let opts = self.opts;
        let mut agent = self.agent.borrow_mut();
        let buf: Vec<RolloutEntry<usize, usize>> = agent.get_buffer().to_vec();
        if buf.is_empty() {
            return;
        }
        let n = buf.len();
        // 1. GAE advantages and returns-to-go.
        let (mut adv, ret) = compute_advantages(&agent, &buf, opts.gamma, opts.lambda);
        if opts.normalise_advantage.unwrap_or(true) {
            let mean = adv.iter().sum::<f64>() / n as f64;
            let var = adv.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt() + 1e-8;
            for x in adv.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
        // 2. K epochs of clipped surrogate + value-loss SGD.
        let mut idx: Vec<usize> = (0..n).collect();
        for _epoch in 0..opts.num_epochs {
            shuffle(&mut idx, &mut *self.shuffle_rng);
            let mut mb = 0;
            while mb < n {
                let end = n.min(mb + opts.mini_batch_size);
                for &i in &idx[mb..end] {
                    apply_one_sample_update(&mut agent, &buf[i], adv[i], ret[i], &opts);
                }
                mb += opts.mini_batch_size;
            }
        }
        // 3. Clear the buffer for next rollout.
        agent.clear_buffer();
    }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

/// `interface PPODESResult`.
#[derive(Clone, Debug, Default)]
pub struct PPODESResult {
    pub theta: Vec<Vec<f64>>,
    pub v: Vec<f64>,
    pub policy: Vec<usize>,
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub total_episodes: usize,
    pub total_steps: u64,
    pub total_updates: u64,
    pub total_ticks: usize,
}

/// Options bag for [`run_ppo_des`] (the TS `opts` object).
#[derive(Default)]
pub struct RunPPOOptions {
    pub total_steps: u64,
    pub rollout_len: usize,
    pub num_epochs: usize,
    pub mini_batch_size: usize,
    pub policy_lr: f64,
    pub value_lr: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub clip_eps: f64,
    pub entropy_coef: Option<f64>,
    pub normalise_advantage: Option<bool>,
    pub max_steps_per_episode: Option<usize>,
    pub seed: Option<u32>,
    pub des_options: Option<IterativeRunOptions>,
}

/// Wire the PPO station graph (env ⇄ agent → updater → agent) and run until the
/// global step budget is reached.
pub fn run_ppo_des(env: Box<dyn PureEnvironment<usize, usize>>, opts: RunPPOOptions) -> PPODESResult {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let shared = Rc::new(RefCell::new(mulberry32(opts.seed.unwrap_or(1))));

    let agent = Rc::new(RefCell::new(TabularPPOAgent::new(
        "actor",
        num_states,
        num_actions,
        opts.rollout_len,
        Box::new(SharedRng(shared.clone())),
    )));
    let updater = Rc::new(RefCell::new(PPOClipUpdateStation::new(
        "updater",
        agent.clone(),
        PPOUpdateOptions {
            gamma: opts.gamma,
            lambda: opts.lambda,
            clip_eps: opts.clip_eps,
            policy_lr: opts.policy_lr,
            value_lr: opts.value_lr,
            num_epochs: opts.num_epochs,
            mini_batch_size: opts.mini_batch_size,
            entropy_coef: Some(opts.entropy_coef.unwrap_or(0.0)),
            normalise_advantage: opts.normalise_advantage,
        },
        Box::new(SharedRng(shared.clone())),
    )));
    let env_st = Rc::new(RefCell::new(EnvironmentStation::new(
        "env",
        env,
        EnvironmentStationOptions { num_episodes: None, max_steps_per_episode: opts.max_steps_per_episode },
    )));

    // Channel wiring.
    env_st.borrow_mut().core_mut().pipe(agent.clone() as StationRef, environment::CH_STATE, policy_gradient_agent::CH_STATE);
    env_st.borrow_mut().core_mut().pipe(agent.clone() as StationRef, environment::CH_TRANSITION, policy_gradient_agent::CH_TRANSITION);
    agent.borrow_mut().core_mut().pipe(env_st.clone() as StationRef, policy_gradient_agent::CH_ACTION, environment::CH_ACTION);
    agent.borrow_mut().core_mut().pipe(updater.clone() as StationRef, policy_gradient_agent::CH_TRAIN, policy_gradient_agent::CH_TRAIN);
    updater.borrow_mut().core_mut().pipe(agent.clone() as StationRef, policy_gradient_agent::CH_RESUME, policy_gradient_agent::CH_RESUME);

    let mut des_options = opts.des_options.unwrap_or_default();
    if des_options.rng.is_none() {
        let r = shared.clone();
        des_options.rng = Some(Box::new(move || r.borrow_mut().next_float()));
    }
    let total_steps_budget = opts.total_steps;
    if des_options.stop_when.is_none() {
        let env_for_stop = env_st.clone();
        des_options.stop_when = Some(Box::new(move |_tick, _| env_for_stop.borrow().total_steps() >= total_steps_budget));
    }
    let summary = run_iterative_des(
        vec![env_st.clone() as StationRef, agent.clone() as StationRef, updater.clone() as StationRef],
        des_options,
    );
    env_st.borrow_mut().done = true;

    let theta = agent.borrow().theta.clone();
    let v = agent.borrow().v.clone();
    let policy = agent.borrow_mut().greedy_policy();
    let reward_history = env_st.borrow().reward_history().to_vec();
    let length_history = env_st.borrow().length_history().to_vec();
    let total_steps = env_st.borrow().total_steps();
    let total_updates = updater.borrow().pu_core().num_updates;
    PPODESResult {
        total_episodes: reward_history.len(),
        theta,
        v,
        policy,
        reward_history,
        length_history,
        total_steps,
        total_updates,
        total_ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! PPO over a single-state bandit DES learns to favour the paying action.
    //!
    //! State 0 has two actions; action 1 pays +1 and action 0 pays 0, each step
    //! terminal. After training the greedy policy must pick action 1 and the
    //! mean reward should rise; at least one clipped update must have fired.
    use super::*;
    use crate::des::general::des_base::environment::StepResult;

    /// Single-state bandit: every step terminal, action 1 pays +1.
    struct Bandit;

    impl PureEnvironment<usize, usize> for Bandit {
        fn num_states(&self) -> usize {
            1
        }
        fn num_actions(&self) -> usize {
            2
        }
        fn reset(&mut self) -> usize {
            0
        }
        fn step(&mut self, _state: usize, action: usize) -> StepResult<usize> {
            StepResult { next_state: 0, reward: if action == 1 { 1.0 } else { 0.0 }, done: true }
        }
    }

    fn run() -> PPODESResult {
        run_ppo_des(
            Box::new(Bandit),
            RunPPOOptions {
                total_steps: 4000,
                rollout_len: 16,
                num_epochs: 4,
                mini_batch_size: 8,
                policy_lr: 0.1,
                value_lr: 0.1,
                gamma: 0.99,
                lambda: 0.95,
                clip_eps: 0.2,
                entropy_coef: Some(0.0),
                normalise_advantage: Some(true),
                max_steps_per_episode: Some(1),
                seed: Some(11),
                des_options: None,
            },
        )
    }

    #[test]
    fn learns_to_prefer_paying_action() {
        let res = run();
        assert!(res.theta[0][1] > res.theta[0][0], "theta = {:?}", res.theta[0]);
        assert_eq!(res.policy[0], 1);
    }

    #[test]
    fn runs_updates_and_records_steps() {
        let res = run();
        assert!(res.total_updates > 0, "expected at least one PPO update");
        assert!(res.total_steps >= 4000);
        assert_eq!(res.total_episodes, res.reward_history.len());
    }

    #[test]
    fn mean_reward_rises_over_training() {
        let res = run();
        let h = &res.reward_history;
        let window = 200.min(h.len() / 4);
        let first: f64 = h[..window].iter().sum::<f64>() / window as f64;
        let last: f64 = h[h.len() - window..].iter().sum::<f64>() / window as f64;
        assert!(last >= first, "mean reward should not fall: first {first}, last {last}");
    }
}
