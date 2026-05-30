//! Port of `src/des/general/des-base/semi-mdp.ts`.
//!
//! SEMI-MARKOV DECISION PROCESSES under the OPTIONS FRAMEWORK (Sutton, Precup,
//! Singh 1999). A standard MDP fixes a one-step decision granularity; a
//! Semi-MDP lets decisions span MULTIPLE primitive steps: the agent picks an
//! OPTION ω (a sub-policy), executes it until the option's TERMINATION
//! condition fires, and only then chooses the next option. An option is a
//! triple of an initiation set `I_w(s)`, an internal policy `pi_w(a|s)`, and a
//! termination probability `beta_w(s) in [0,1]`.
//!
//! This module provides INTRA-OPTION SMDP Q-LEARNING, updating
//! `Q(s,w) += alpha [ r_bar + gamma^tau max_w' Q(s',w') - Q(s,w) ]`.
//!
//! ## Mapping (TS `abstract class … extends RLAgentStation` → Rust)
//!
//!   * The TS `Option<S, A>` interface is renamed [`Opt`] (the prelude already
//!     owns `Option`).
//!   * The abstract `SemiMDPAgentStation` becomes the hook trait
//!     [`SemiMDPAgentStation`] (a sub-trait of
//!     [`RLAgentStation`](crate::des::general::des_base::rl_agent::RLAgentStation))
//!     plus a [`SemiMDPCore`] struct the concrete agent EMBEDS for the fields
//!     the abstract class owned. The abstract hooks `options()` / `stateKey()`
//!     are required trait methods; `pickOption`/`backup` and the
//!     `pickAction`/`update`/`endOfEpisode` bodies are provided methods.
//!   * `Q: number[][]` (sparse, keyed by `stateKey`) → `Vec<Vec<f64>>` grown on
//!     demand; an empty row models the TS `undefined` row, `q_get` returns the
//!     TS `?? 0`.
//!   * `optionStartState!` (definite assignment) → `Option<S>`.
//!   * INTERIOR MUTABILITY: the TS `pickAction` mutates `currentOption`,
//!     `optionStartState`, `optionReward`, `optionTau` and calls `backup`
//!     (which writes `Q`). But [`RLAgentStation::pick_action`] takes `&self`, so
//!     these fields live behind `Cell`/`RefCell`.
//!   * non-ASCII `ω`/`γ` → `omega`/`gamma`; `Math.pow(γ, tau)` → `gamma.powi(tau)`.
//!   * `throw new Error` (no legal option) → `panic!`.
//!
//! NOTE: the TS file's `import {StateToken, ActionToken, TransitionToken}` is
//! unused in the body (the tokens are handled by the `RLAgentStation` template)
//! and is therefore dropped here.

use std::cell::{Cell, RefCell};

use crate::des::general::des_base::argmax::ARGMAX_EPS_DEFAULT;
use crate::des::general::des_base::rl_agent::RLAgentStation;
use crate::des::shared::capabilities::RandomSource;

/// An option `ω = ⟨I_ω, π_ω, β_ω⟩`. Renamed from the TS `Option<S, A>`
/// interface to avoid clashing with the prelude `Option`.
pub trait Opt<S = f64, A = usize> {
    /// Human-readable label (used in debug output).
    fn name(&self) -> &str;
    /// Initiation set: true iff this option can start in state `s`.
    fn init(&self, s: &S) -> bool;
    /// Internal policy of the option.
    fn policy(&self, s: &S, rng: &mut dyn RandomSource) -> A;
    /// Termination probability `β_ω(s)`. 1 = terminate; 0 = continue.
    fn terminate(&self, s: &S) -> f64;
}

/// Configuration for a [`SemiMDPAgentStation`]. The injected `rng` is barred
/// (it lives in the embedded `RLAgentCore`) so the rest can derive [`Default`].
#[derive(Default)]
pub struct SemiMDPOptions {
    /// Step size α. Default `0.1`.
    pub alpha: Option<f64>,
    /// Discount γ. Default `0.95`.
    pub gamma: Option<f64>,
    /// ε-greedy probability over the OPTION level. Default `0.1`.
    pub epsilon: Option<f64>,
    /// ε-decay multiplier per primitive episode. Default `1`.
    pub epsilon_decay: Option<f64>,
    /// ε-floor. Default `0.01`.
    pub epsilon_min: Option<f64>,
}

/// `Required<SemiMDPOptions>` after defaults are resolved.
pub struct ResolvedSemiMdpOptions {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
}

/// The fields the TS `abstract class SemiMDPAgentStation` owned, factored into
/// a struct the concrete agent embeds. The per-option bookkeeping is held
/// behind `Cell`/`RefCell` because `pick_action` (a `&self` hook) mutates it.
pub struct SemiMDPCore<S> {
    pub opts: ResolvedSemiMdpOptions,
    /// `Q[s][ω]`, grown on demand; an empty row models the TS `undefined` row.
    pub q: RefCell<Vec<Vec<f64>>>,
    /// Currently executing option (`-1` = none).
    pub current_option: Cell<i64>,
    /// State at which the current option began.
    pub option_start_state: RefCell<Option<S>>,
    /// Cumulative discounted reward inside the current option.
    pub option_reward: Cell<f64>,
    /// Number of primitive steps inside the current option.
    pub option_tau: Cell<i32>,
}

impl<S> SemiMDPCore<S> {
    /// Resolve `SemiMDPOptions` defaults (mirrors the TS `Required<…>` spread).
    pub fn new(semi_opts: SemiMDPOptions) -> Self {
        SemiMDPCore {
            opts: ResolvedSemiMdpOptions {
                alpha: semi_opts.alpha.unwrap_or(0.1),
                gamma: semi_opts.gamma.unwrap_or(0.95),
                epsilon: semi_opts.epsilon.unwrap_or(0.1),
                epsilon_decay: semi_opts.epsilon_decay.unwrap_or(1.0),
                epsilon_min: semi_opts.epsilon_min.unwrap_or(0.01),
            },
            q: RefCell::new(Vec::new()),
            current_option: Cell::new(-1),
            option_start_state: RefCell::new(None),
            option_reward: Cell::new(0.0),
            option_tau: Cell::new(0),
        }
    }
}

/// TS `this.Q[sk]?.[i] ?? 0`: 0 for a missing row, empty row, or out-of-range
/// column.
fn q_get(q: &[Vec<f64>], sk: usize, i: usize) -> f64 {
    q.get(sk).and_then(|row| row.get(i)).copied().unwrap_or(0.0)
}

/// SMDP Q-learning at the option level on top of a discrete-state MDP. A
/// sub-trait of [`RLAgentStation`]; the concrete agent delegates the
/// `RLAgentStation` hooks here (`pick_action` → [`Self::semi_pick_action`],
/// `update` → [`Self::semi_update`], `end_of_episode` →
/// [`Self::semi_end_of_episode`]).
///
/// `S`/`A` must be `'static` (so the agent's `Box<dyn Opt<S, A>>` library and
/// the tokens qualify) and `S: Clone` (the start state is cloned into the
/// option-start cell).
pub trait SemiMDPAgentStation<S: Clone + 'static = f64, A: Clone + 'static = usize>:
    RLAgentStation<S, A>
{
    /// Borrow the embedded semi-MDP bookkeeping.
    fn semi_core(&self) -> &SemiMDPCore<S>;
    /// Mutably borrow the embedded semi-MDP bookkeeping.
    fn semi_core_mut(&mut self) -> &mut SemiMDPCore<S>;

    // ── HOOKS (abstract) ───────────────────────────────────────────────────

    /// Library of available options.
    fn options(&self) -> &[Box<dyn Opt<S, A>>];
    /// State-key for indexing `Q[s]`.
    fn state_key(&self, s: &S) -> usize;

    // ── HOOKS (optional override) ──────────────────────────────────────────

    /// ε-greedy over options at `s`, with UNIFORM RANDOM TIE-BREAKING on the
    /// greedy argmax (necessary because Q starts all-zero). Panics if no option
    /// is legal in `s`.
    fn pick_option(&self, s: &S, rng: &mut dyn RandomSource) -> usize {
        let opts = self.options();
        let mut legal: Vec<usize> = Vec::new();
        for i in 0..opts.len() {
            if opts[i].init(s) {
                legal.push(i);
            }
        }
        if legal.is_empty() {
            panic!("no option available in state");
        }
        let sc = self.semi_core();
        if rng.next_float() < sc.opts.epsilon {
            let idx = (rng.next_float() * legal.len() as f64).floor() as usize;
            return legal[idx.min(legal.len() - 1)];
        }
        let eps = ARGMAX_EPS_DEFAULT;
        let mut best_q = f64::NEG_INFINITY;
        let mut best: i64 = -1;
        let mut tie_count = 0.0;
        let sk = self.state_key(s);
        let q = sc.q.borrow();
        for &i in &legal {
            let qv = q_get(&q, sk, i);
            if best < 0 || qv > best_q + eps {
                best_q = qv;
                best = i as i64;
                tie_count = 1.0;
            } else if qv >= best_q - eps {
                tie_count += 1.0;
                if rng.next_float() * tie_count < 1.0 {
                    best = i as i64;
                }
            }
        }
        best as usize
    }

    // ── ACTION / UPDATE WIRED THROUGH RLAgentStation ────────────────────────

    /// Inside-option primitive action selection. If the option terminated (or
    /// none is active) it backs up `Q(s_start, ω)` and chooses a new option,
    /// then returns the (new) option's primitive action.
    fn semi_pick_action(&self, state: &S, rng: &mut dyn RandomSource) -> A {
        let co = self.semi_core().current_option.get();
        let need_new = if co < 0 {
            true
        } else {
            // Order matters (TS: `terminate(state) >= rng()`).
            let term = self.options()[co as usize].terminate(state);
            term >= rng.next_float()
        };
        if need_new {
            if co >= 0 {
                self.backup(state, false);
            }
            let new_option = self.pick_option(state, rng);
            let sc = self.semi_core();
            sc.current_option.set(new_option as i64);
            *sc.option_start_state.borrow_mut() = Some(state.clone());
            sc.option_reward.set(0.0);
            sc.option_tau.set(0);
        }
        let co2 = self.semi_core().current_option.get() as usize;
        self.options()[co2].policy(state, rng)
    }

    /// Accumulate the discounted intra-option reward; back up on episode end.
    fn semi_update(&mut self, reward: f64, next_state: &S, done: bool) {
        {
            let sc = self.semi_core();
            if sc.current_option.get() < 0 {
                return;
            }
            let gamma = sc.opts.gamma;
            let tau = sc.option_tau.get();
            sc.option_reward
                .set(sc.option_reward.get() + gamma.powi(tau) * reward);
            sc.option_tau.set(tau + 1);
        }
        if done {
            self.backup(next_state, true);
        }
    }

    /// Apply the SMDP Q-learning update for the option that just ended.
    fn backup(&self, s_next: &S, terminal_episode: bool) {
        let sc = self.semi_core();
        let omega = sc.current_option.get();
        let gamma = sc.opts.gamma;
        let tau = sc.option_tau.get();
        let option_reward = sc.option_reward.get();
        let sk = {
            let start = sc.option_start_state.borrow();
            let s_start = start.as_ref().expect("option start state recorded");
            self.state_key(s_start)
        };
        let mut bootstrap = 0.0;
        if !terminal_episode {
            let opts = self.options();
            let skn = self.state_key(s_next);
            let q = sc.q.borrow();
            let mut best = f64::NEG_INFINITY;
            for j in 0..opts.len() {
                if !opts[j].init(s_next) {
                    continue;
                }
                let qv = q_get(&q, skn, j);
                if qv > best {
                    best = qv;
                }
            }
            if best.is_finite() {
                bootstrap = gamma.powi(tau) * best;
            }
        }
        let num_options = self.options().len();
        {
            let mut q = sc.q.borrow_mut();
            // Grow + lazily allocate the row (TS `if (!this.Q[sk]) …`).
            if sk >= q.len() {
                q.resize(sk + 1, Vec::new());
            }
            if q[sk].is_empty() {
                q[sk] = vec![0.0; num_options];
            }
            let target = option_reward + bootstrap;
            let cur = q[sk][omega as usize];
            q[sk][omega as usize] = cur + sc.opts.alpha * (target - cur);
        }
        sc.current_option.set(-1);
    }

    /// Decay ε and reset the running option (TS `endOfEpisode`).
    fn semi_end_of_episode(&mut self) {
        let sc = self.semi_core_mut();
        sc.opts.epsilon = (sc.opts.epsilon * sc.opts.epsilon_decay).max(sc.opts.epsilon_min);
        sc.current_option.set(-1);
    }

    // ── PUBLIC ACCESSORS ────────────────────────────────────────────────────

    fn get_q(&self) -> Vec<Vec<f64>> {
        self.semi_core().q.borrow().clone()
    }
    fn get_epsilon(&self) -> f64 {
        self.semi_core().opts.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::rl_agent::RLAgentCore;
    use crate::des::general::des_base::rl_tokens::{StateToken, TransitionToken};
    use crate::des::general::des_base::station::{DESStation, StationCore};
    use crate::des::shared::capabilities::SeededRandom;
    use std::any::Any;
    use std::rc::Rc;

    /// An option with a constant primitive action and a constant termination
    /// probability `beta` (β=0 ⇒ runs until episode `done`; β=1 ⇒ one-step).
    struct FixedOption {
        name: String,
        action: usize,
        beta: f64,
    }

    impl Opt<usize, usize> for FixedOption {
        fn name(&self) -> &str {
            &self.name
        }
        fn init(&self, _s: &usize) -> bool {
            true
        }
        fn policy(&self, _s: &usize, _rng: &mut dyn RandomSource) -> usize {
            self.action
        }
        fn terminate(&self, _s: &usize) -> f64 {
            self.beta
        }
    }

    /// Concrete options agent over integer states (`state_key` is identity).
    struct OptionsAgent {
        core: StationCore,
        agent: RLAgentCore,
        semi: SemiMDPCore<usize>,
        opt_lib: Vec<Box<dyn Opt<usize, usize>>>,
    }

    impl OptionsAgent {
        fn new(seed: u32, opts: SemiMDPOptions, opt_lib: Vec<Box<dyn Opt<usize, usize>>>) -> Self {
            OptionsAgent {
                core: StationCore::new("smdp"),
                agent: RLAgentCore::new(Box::new(SeededRandom::new(seed))),
                semi: SemiMDPCore::new(opts),
                opt_lib,
            }
        }
    }

    impl DESStation for OptionsAgent {
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

    impl RLAgentStation<usize, usize> for OptionsAgent {
        fn agent_core(&self) -> &RLAgentCore {
            &self.agent
        }
        fn agent_core_mut(&mut self) -> &mut RLAgentCore {
            &mut self.agent
        }
        fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
            self.semi_pick_action(state, rng)
        }
        fn update(
            &mut self,
            _state: &usize,
            _action: &usize,
            reward: f64,
            next_state: &usize,
            done: bool,
        ) {
            self.semi_update(reward, next_state, done);
        }
        fn end_of_episode(&mut self, _episode_id: f64) {
            self.semi_end_of_episode();
        }
    }

    impl SemiMDPAgentStation<usize, usize> for OptionsAgent {
        fn semi_core(&self) -> &SemiMDPCore<usize> {
            &self.semi
        }
        fn semi_core_mut(&mut self) -> &mut SemiMDPCore<usize> {
            &mut self.semi
        }
        fn options(&self) -> &[Box<dyn Opt<usize, usize>>] {
            &self.opt_lib
        }
        fn state_key(&self, s: &usize) -> usize {
            *s
        }
    }

    /// A single option that runs for TWO primitive steps before the episode
    /// ends. The terminal backup must equal the discounted multi-step return:
    ///   r̄ = r0 + γ·r1 = 1 + 0.95·2 = 2.9, target = r̄ (terminal), and with
    ///   α=0.5 from Q=0: Q[0][0] = 0.5·2.9 = 1.45.
    #[test]
    fn semi_mdp_two_step_option_discounts_return() {
        let lib: Vec<Box<dyn Opt<usize, usize>>> = vec![Box::new(FixedOption {
            name: "go".into(),
            action: 0,
            beta: 0.0,
        })];
        let mut agent = OptionsAgent::new(
            1,
            SemiMDPOptions {
                alpha: Some(0.5),
                gamma: Some(0.95),
                epsilon: Some(0.0),
                ..Default::default()
            },
            lib,
        );

        agent.core_mut().take(
            Rc::new(StateToken::new(0usize, 0.0)),
            OptionsAgent::CH_STATE,
        );
        agent.run_time_step();
        agent.core_mut().take(
            Rc::new(TransitionToken::new(
                0usize, 0usize, 1.0, 1usize, false, 0.0,
            )),
            OptionsAgent::CH_TRANSITION,
        );
        agent.run_time_step();
        agent.core_mut().take(
            Rc::new(TransitionToken::new(1usize, 0usize, 2.0, 2usize, true, 0.0)),
            OptionsAgent::CH_TRANSITION,
        );
        agent.run_time_step();

        let q = agent.get_q();
        assert!((q[0][0] - 1.45).abs() < 1e-9, "Q={:?}", q);
    }

    /// Non-terminal option backup must bootstrap with `γ^τ · max_{ω'} Q(s',ω')`.
    /// Pre-seed Q[0]=[1,0] (so the greedy first option picked at s=0 is option
    /// 0) and Q[2]=[10,3]. A one-step option (β=1) takes s=0 →(r=4)→ s=2 and
    /// terminates non-terminally. With α=0.5, γ=0.5, τ=1:
    ///   target = 4 + 0.5·10 = 9 ; Q[0][0] = 1 + 0.5·(9−1) = 5.
    #[test]
    fn semi_mdp_backup_bootstraps_from_next_state() {
        let lib: Vec<Box<dyn Opt<usize, usize>>> = vec![
            Box::new(FixedOption {
                name: "a".into(),
                action: 0,
                beta: 1.0,
            }),
            Box::new(FixedOption {
                name: "b".into(),
                action: 1,
                beta: 1.0,
            }),
        ];
        let mut agent = OptionsAgent::new(
            7,
            SemiMDPOptions {
                alpha: Some(0.5),
                gamma: Some(0.5),
                epsilon: Some(0.0),
                ..Default::default()
            },
            lib,
        );
        *agent.semi.q.borrow_mut() = vec![vec![1.0, 0.0], vec![], vec![10.0, 3.0]];

        agent.core_mut().take(
            Rc::new(StateToken::new(0usize, 0.0)),
            OptionsAgent::CH_STATE,
        );
        agent.run_time_step();
        // r=4, s0 → s2, not done: the option (β=1) terminates on the next pick.
        agent.core_mut().take(
            Rc::new(TransitionToken::new(
                0usize, 0usize, 4.0, 2usize, false, 0.0,
            )),
            OptionsAgent::CH_TRANSITION,
        );
        agent.run_time_step();

        let q = agent.get_q();
        assert!((q[0][0] - 5.0).abs() < 1e-9, "Q={:?}", q);
    }
}
