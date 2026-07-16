//! Port of `src/des/general/value-iteration.ts` — generic value iteration for
//! finite-state, finite-action MDPs, driven as a fixed-point DES.
//!
//! Bellman optimality equation:
//!   V*(s) = max_a Σ_s' T(s'|s,a) · [ r(s, a, s') + γ · V*(s') ]
//!   π*(s) = argmax_a Σ_s' T(s'|s,a) · [ r(s, a, s') + γ · V*(s') ]
//!
//! Caller supplies an [`MDPSpec`]: number of states, number of actions per
//! state (variable), and an `outcomes(s, a)` closure returning the list of
//! `{prob, reward, next_state}` triples for taking action `a` in state `s`.
//!
//! Convergence: `max_s |V_next(s) − V(s)| < tol`. Default tol = 1e-9.
//!
//! STUBBED / INLINED (their real home, `general/des-base/`, is not ported yet):
//!   * `FixedPointIterationStation<Float64Array>` (template-method base) — the
//!     fixed-point loop is inlined directly into [`ValueIterationStation::run`].
//!   * `runIterativeDES([...])` — replaced by the inline loop above.
//!   * `scanArgMaxTieBreak` — reimplemented locally as [`scan_argmax_tie_break`]
//!     (reservoir-sampled random tie-break).
//!   * `intrinsicCheck` / `externalReferenceValidator` / `ValidationCheck`
//!     (diagnostic validators) — DROPPED; they observe station internals that
//!     belong to the unported `des-base` framework.
//!   * Ambient `Math.random` default RNG — replaced by an injected
//!     `Box<dyn Fn() -> f64>`, defaulting to a tiny deterministic xorshift
//!     ([`default_rng`]) per the migration "inject capabilities" rule.

use crate::des::shared::linalg::Vector;

/// One probabilistic transition outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Probability of this outcome, in [0, 1].
    pub prob: f64,
    /// Immediate reward received on this transition.
    pub reward: f64,
    /// Next state index after this outcome.
    pub next_state: usize,
}

/// MDP description consumed by [`value_iteration`].
///
/// `num_actions`/`outcomes`/`is_terminal`/`terminal_reward`/`state_label`/
/// `action_label` are the TS callbacks expressed as boxed closures.
pub struct MDPSpec {
    pub num_states: usize,
    /// Number of legal actions in state `s`. If action `a` is illegal in `s`,
    /// return `[]` from `outcomes(s, a)` and it will be skipped.
    pub num_actions: Box<dyn Fn(usize) -> usize>,
    /// Outcomes for taking action `a` in state `s`. Probabilities must sum to 1
    /// (within `prob_tol`).
    pub outcomes: Box<dyn Fn(usize, usize) -> Vec<Outcome>>,
    /// Predicate marking absorbing terminal states. Default: never terminal.
    pub is_terminal: Option<Box<dyn Fn(usize) -> bool>>,
    /// Reward for being in a terminal state (V is pinned here). Default: 0.
    pub terminal_reward: Option<Box<dyn Fn(usize) -> f64>>,
    /// Optional human-readable label for state `s`.
    pub state_label: Option<Box<dyn Fn(usize) -> String>>,
    /// Optional human-readable label for action `a`.
    pub action_label: Option<Box<dyn Fn(usize) -> String>>,
}

/// Options controlling value iteration. Mirrors the TS `VIOptions` defaults.
pub struct VIOptions {
    /// Discount factor in [0, 1]. Default 0.95.
    pub gamma: f64,
    /// Convergence tolerance on max |ΔV|. Default 1e-9.
    pub tol: f64,
    /// Hard cap on iterations. Default 5000.
    pub max_iter: usize,
    /// Validate that outcome probabilities sum to 1 ± `prob_tol`. Default true.
    pub validate_probs: bool,
    pub prob_tol: f64,
    /// Random tie-breaking in `greedy_policy`. Default true.
    pub random_tie_break: bool,
    pub tie_break_eps: f64,
    /// Injected RNG for tie-breaking. Default: deterministic xorshift.
    pub rng: Box<dyn Fn() -> f64>,
}

impl Default for VIOptions {
    fn default() -> Self {
        VIOptions {
            gamma: 0.95,
            tol: 1e-9,
            max_iter: 5000,
            validate_probs: true,
            prob_tol: 1e-9,
            random_tie_break: true,
            tie_break_eps: 1e-12,
            rng: default_rng(),
        }
    }
}

/// Result of value iteration: optimal value function and greedy policy.
#[derive(Clone, Debug)]
pub struct VIResult {
    pub v: Vector,
    /// Greedy action per state; `-1` for terminal states.
    pub policy: Vec<i32>,
    pub iterations: usize,
    pub final_delta: f64,
    pub gamma: f64,
}

/// Tiny deterministic xorshift64* RNG yielding `f64` in `[0, 1)`. Replaces the
/// ambient `Math.random` default in the TS source (migration rule: inject RNG).
fn default_rng() -> Box<dyn Fn() -> f64> {
    let state = std::cell::Cell::new(0x2545_F491_4F6C_DD1Du64);
    Box::new(move || {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        // Top 53 bits → [0, 1).
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    })
}

/// Argmax over `0..n_actions` with reservoir-sampled random tie-breaking.
///
/// Inlined replacement for `scanArgMaxTieBreak` from the unported `des-base`.
/// Among actions whose Q-value is within `eps` of the running best, one is
/// chosen uniformly at random via the injected `rng`.
fn scan_argmax_tie_break(
    n_actions: usize,
    q: impl Fn(usize) -> f64,
    rng: &dyn Fn() -> f64,
    eps: f64,
) -> i32 {
    let mut best_q = f64::NEG_INFINITY;
    let mut best_a: i32 = -1;
    let mut tie_count = 0.0_f64;
    for a in 0..n_actions {
        let qa = q(a);
        if qa > best_q + eps {
            best_q = qa;
            best_a = a as i32;
            tie_count = 1.0;
        } else if (qa - best_q).abs() <= eps {
            tie_count += 1.0;
            if rng() < 1.0 / tie_count {
                best_a = a as i32;
            }
            if qa > best_q {
                best_q = qa;
            }
        }
    }
    best_a
}

/// Value-iteration station: holds the precomputed transition table and runs the
/// inlined fixed-point Bellman backup. (The `FixedPointIterationStation` base
/// is not ported, so the template method `run` is inlined here.)
pub struct ValueIterationStation {
    spec: MDPSpec,
    gamma: f64,
    transitions: Vec<Vec<Vec<Outcome>>>,
    a_count: Vec<usize>,
    random_tie_break: bool,
    tie_break_eps: f64,
    rng: Box<dyn Fn() -> f64>,
    tol: f64,
    max_iter: usize,
    current: Vector,
    last_delta: f64,
    iteration: usize,
    delta_history: Vec<f64>,
}

impl ValueIterationStation {
    /// Build the station, precomputing the transition table once (a ~50×
    /// speedup over recomputing on every Bellman sweep). Panics if
    /// `validate_probs` is set and an action's outcome probabilities do not sum
    /// to 1 (invariant violation → `throw` in TS).
    pub fn new(spec: MDPSpec, opts: VIOptions) -> Self {
        assert!(
            opts.gamma.is_finite() && (0.0..=1.0).contains(&opts.gamma),
            "value iteration: discount gamma must be finite and in [0, 1], got {}",
            opts.gamma
        );
        assert!(
            opts.tol.is_finite() && opts.tol > 0.0,
            "value iteration: convergence tolerance must be finite and > 0, got {}",
            opts.tol
        );
        assert!(
            opts.max_iter > 0,
            "value iteration: max_iter must be greater than zero"
        );
        assert!(
            opts.prob_tol.is_finite() && opts.prob_tol >= 0.0,
            "value iteration: probability tolerance must be finite and >= 0, got {}",
            opts.prob_tol
        );
        assert!(
            opts.tie_break_eps.is_finite() && opts.tie_break_eps >= 0.0,
            "value iteration: tie-break epsilon must be finite and >= 0, got {}",
            opts.tie_break_eps
        );
        let num_states = spec.num_states;
        let mut transitions: Vec<Vec<Vec<Outcome>>> = Vec::with_capacity(num_states);
        let mut a_count: Vec<usize> = Vec::with_capacity(num_states);

        for s in 0..num_states {
            if Self::is_terminal_with(&spec, s) {
                transitions.push(Vec::new());
                a_count.push(0);
                continue;
            }
            let n = (spec.num_actions)(s);
            a_count.push(n);
            let mut per_action: Vec<Vec<Outcome>> = Vec::with_capacity(n);
            for a in 0..n {
                let ol = (spec.outcomes)(s, a);
                for (outcome_index, outcome) in ol.iter().enumerate() {
                    assert!(
                        outcome.prob.is_finite() && (0.0..=1.0).contains(&outcome.prob),
                        "outcomes({s}, {a})[{outcome_index}] probability must be finite and in [0, 1], got {}",
                        outcome.prob
                    );
                    assert!(
                        outcome.reward.is_finite(),
                        "outcomes({s}, {a})[{outcome_index}] reward must be finite, got {}",
                        outcome.reward
                    );
                    assert!(
                        outcome.next_state < num_states,
                        "outcomes({s}, {a})[{outcome_index}] next_state {} is outside 0..{num_states}",
                        outcome.next_state
                    );
                }
                if opts.validate_probs && !ol.is_empty() {
                    let total: f64 = ol.iter().map(|o| o.prob).sum();
                    if (total - 1.0).abs() > opts.prob_tol {
                        eprintln!(
                            "[value-iteration] outcomes({s}, {a}) probabilities sum to {total} \
                             (expected 1 ± {}) — transition model is not a valid distribution.",
                            opts.prob_tol
                        );
                        panic!(
                            "outcomes({s}, {a}) probabilities sum to {total}, expected 1 (tol={})",
                            opts.prob_tol
                        );
                    }
                }
                per_action.push(ol);
            }
            transitions.push(per_action);
        }

        let mut station = ValueIterationStation {
            gamma: opts.gamma,
            transitions,
            a_count,
            random_tie_break: opts.random_tie_break,
            tie_break_eps: opts.tie_break_eps,
            rng: opts.rng,
            tol: opts.tol,
            max_iter: opts.max_iter,
            current: Vec::new(),
            last_delta: f64::INFINITY,
            iteration: 0,
            delta_history: Vec::new(),
            spec,
        };
        station.current = station.initial_state();
        station
    }

    fn is_terminal_with(spec: &MDPSpec, s: usize) -> bool {
        match &spec.is_terminal {
            Some(f) => f(s),
            None => false,
        }
    }

    fn is_terminal(&self, s: usize) -> bool {
        Self::is_terminal_with(&self.spec, s)
    }

    fn terminal_reward(&self, s: usize) -> f64 {
        match &self.spec.terminal_reward {
            Some(f) => f(s),
            None => 0.0,
        }
    }

    // ── HOOKS ───────────────────────────────────────────────────────────────

    fn initial_state(&self) -> Vector {
        let mut v = vec![0.0_f64; self.spec.num_states];
        for s in 0..self.spec.num_states {
            if self.is_terminal(s) {
                let reward = self.terminal_reward(s);
                assert!(
                    reward.is_finite(),
                    "value iteration: terminal reward for state {s} must be finite, got {reward}"
                );
                v[s] = reward;
            }
        }
        v
    }

    fn apply_operator(&self, v: &[f64]) -> Vector {
        let mut vn = vec![0.0_f64; self.spec.num_states];
        for s in 0..self.spec.num_states {
            if self.is_terminal(s) {
                vn[s] = v[s];
                continue;
            }
            let n = self.a_count[s];
            let mut best = f64::NEG_INFINITY;
            let mut any = false;
            for a in 0..n {
                let ol = &self.transitions[s][a];
                if ol.is_empty() {
                    continue;
                }
                let mut q = 0.0;
                for o in ol {
                    q += o.prob * (o.reward + self.gamma * v[o.next_state]);
                }
                if q > best {
                    best = q;
                    any = true;
                }
            }
            vn[s] = if any { best } else { 0.0 };
        }
        vn
    }

    fn delta(prev: &[f64], next: &[f64]) -> f64 {
        let mut d = 0.0;
        for s in 0..prev.len() {
            let di = (next[s] - prev[s]).abs();
            if di > d {
                d = di;
            }
        }
        d
    }

    /// Inlined fixed-point loop (was `FixedPointIterationStation::runTimeStep`
    /// driven by `runIterativeDES`). Iterates the Bellman backup until the
    /// max-norm change drops below `tol` or `max_iter` is reached.
    pub fn run(&mut self) {
        loop {
            if self.iteration >= self.max_iter {
                break;
            }
            let next = self.apply_operator(&self.current);
            let d = Self::delta(&self.current, &next);
            self.current = next;
            self.last_delta = d;
            self.iteration += 1;
            self.delta_history.push(d);
            if d <= self.tol {
                break;
            }
        }
    }

    pub fn get_current(&self) -> &Vector {
        &self.current
    }

    pub fn get_last_delta(&self) -> f64 {
        self.last_delta
    }

    pub fn get_iteration(&self) -> usize {
        self.iteration
    }

    // ── PUBLIC ──────────────────────────────────────────────────────────────

    /// Extract the greedy policy from the current value function.
    ///
    /// When multiple actions tie for the max Q-value (within `tie_break_eps`),
    /// one is chosen uniformly at random (controlled by `random_tie_break`).
    pub fn greedy_policy(&self) -> Vec<i32> {
        let v = &self.current;
        let mut policy = vec![0_i32; self.spec.num_states];
        for s in 0..self.spec.num_states {
            if self.is_terminal(s) {
                policy[s] = -1;
                continue;
            }
            let n = self.a_count[s];
            if self.random_tie_break {
                policy[s] = scan_argmax_tie_break(
                    n,
                    |a| {
                        let ol = &self.transitions[s][a];
                        if ol.is_empty() {
                            return f64::NEG_INFINITY;
                        }
                        let mut q = 0.0;
                        for o in ol {
                            q += o.prob * (o.reward + self.gamma * v[o.next_state]);
                        }
                        q
                    },
                    self.rng.as_ref(),
                    self.tie_break_eps,
                );
            } else {
                let mut best_a: i32 = -1;
                let mut best_q = f64::NEG_INFINITY;
                for a in 0..n {
                    let ol = &self.transitions[s][a];
                    if ol.is_empty() {
                        continue;
                    }
                    let mut q = 0.0;
                    for o in ol {
                        q += o.prob * (o.reward + self.gamma * v[o.next_state]);
                    }
                    if q > best_q {
                        best_q = q;
                        best_a = a as i32;
                    }
                }
                policy[s] = best_a;
            }
        }
        policy
    }
}

/// Run value iteration on an [`MDPSpec`]. Returns optimal value function `V`
/// and greedy policy π extracted from `V`.
///
/// The transition table is built once at the start and reused on every Bellman
/// sweep (typically a ~50× speedup over recomputing on the fly).
pub fn value_iteration(spec: MDPSpec, opts: VIOptions) -> VIResult {
    let gamma = opts.gamma;
    let tol = opts.tol;
    let mut station = ValueIterationStation::new(spec, opts);
    station.run();
    let final_delta = station.get_last_delta();
    if final_delta > tol {
        eprintln!(
            "[value-iteration] stopped after {} iterations with max|ΔV|={final_delta} > tol={tol}; \
             value function may not be optimal.",
            station.get_iteration()
        );
    }
    let v = station.get_current().clone();
    let policy = station.greedy_policy();
    VIResult {
        v,
        policy,
        iterations: station.get_iteration(),
        final_delta: station.get_last_delta(),
        gamma,
    }
}

/// Q-value of `(s, a)` under value function `V`. Useful for inspecting how
/// close the second-best action is to the optimum (policy fragility).
pub fn q_value(spec: &MDPSpec, v: &[f64], s: usize, a: usize, gamma: f64) -> f64 {
    let ol = (spec.outcomes)(s, a);
    let mut q = 0.0;
    for o in &ol {
        q += o.prob * (o.reward + gamma * v[o.next_state]);
    }
    q
}

/// For each state, return all action Q-values sorted descending. Useful for
/// diagnosing how distinguishable the optimal action is.
pub fn q_values_all(spec: &MDPSpec, v: &[f64], s: usize, gamma: f64) -> Vec<(usize, f64)> {
    let n = (spec.num_actions)(s);
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(n);
    for a in 0..n {
        out.push((a, q_value(spec, v, s, a, gamma)));
    }
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_state_spec(outcomes: Vec<Outcome>) -> MDPSpec {
        MDPSpec {
            num_states: 1,
            num_actions: Box::new(|_s| 1),
            outcomes: Box::new(move |_s, _a| outcomes.clone()),
            is_terminal: None,
            terminal_reward: None,
            state_label: None,
            action_label: None,
        }
    }

    /// Two independent self-loop states: V(s) = r(s) / (1 - γ).
    /// With r = [1, 2], γ = 0.9  →  V = [10, 20].
    #[test]
    fn self_loop_geometric_values() {
        let rewards = [1.0_f64, 2.0_f64];
        let spec = MDPSpec {
            num_states: 2,
            num_actions: Box::new(|_s| 1),
            outcomes: Box::new(move |s, _a| {
                vec![Outcome {
                    prob: 1.0,
                    reward: rewards[s],
                    next_state: s,
                }]
            }),
            is_terminal: None,
            terminal_reward: None,
            state_label: None,
            action_label: None,
        };
        let opts = VIOptions {
            gamma: 0.9,
            ..Default::default()
        };
        let res = value_iteration(spec, opts);
        assert!((res.v[0] - 10.0).abs() < 1e-6, "V[0]={}", res.v[0]);
        assert!((res.v[1] - 20.0).abs() < 1e-6, "V[1]={}", res.v[1]);
    }

    /// A discount outside [0, 1] (here γ = 1.5) would silently diverge the
    /// Bellman backup toward ∞ on a recurrent MDP; the construction guard must
    /// reject it up front rather than return an ∞-poisoned value function.
    #[test]
    #[should_panic(expected = "discount gamma must be finite and in [0, 1]")]
    fn rejects_out_of_range_discount() {
        let spec = MDPSpec {
            num_states: 1,
            num_actions: Box::new(|_s| 1),
            outcomes: Box::new(|s, _a| {
                vec![Outcome {
                    prob: 1.0,
                    reward: 1.0,
                    next_state: s,
                }]
            }),
            is_terminal: None,
            terminal_reward: None,
            state_label: None,
            action_label: None,
        };
        let _ = value_iteration(
            spec,
            VIOptions {
                gamma: 1.5,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "convergence tolerance must be finite and > 0")]
    fn rejects_zero_convergence_tolerance() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: 0.0,
                next_state: 0,
            }]),
            VIOptions {
                tol: 0.0,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "max_iter must be greater than zero")]
    fn rejects_zero_iteration_budget() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: 0.0,
                next_state: 0,
            }]),
            VIOptions {
                max_iter: 0,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "probability tolerance must be finite and >= 0")]
    fn rejects_invalid_probability_tolerance() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: 0.0,
                next_state: 0,
            }]),
            VIOptions {
                prob_tol: f64::NAN,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "tie-break epsilon must be finite and >= 0")]
    fn rejects_negative_tie_break_epsilon() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: 0.0,
                next_state: 0,
            }]),
            VIOptions {
                tie_break_eps: -1.0,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "probability must be finite and in [0, 1]")]
    fn rejects_negative_transition_probability_even_when_sum_is_one() {
        let _ = value_iteration(
            one_state_spec(vec![
                Outcome {
                    prob: -0.25,
                    reward: 0.0,
                    next_state: 0,
                },
                Outcome {
                    prob: 1.25,
                    reward: 0.0,
                    next_state: 0,
                },
            ]),
            VIOptions::default(),
        );
    }

    #[test]
    #[should_panic(expected = "reward must be finite")]
    fn rejects_non_finite_transition_reward() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: f64::NAN,
                next_state: 0,
            }]),
            VIOptions::default(),
        );
    }

    #[test]
    #[should_panic(expected = "next_state 1 is outside 0..1")]
    fn rejects_out_of_range_transition_state_during_table_build() {
        let _ = value_iteration(
            one_state_spec(vec![Outcome {
                prob: 1.0,
                reward: 0.0,
                next_state: 1,
            }]),
            VIOptions::default(),
        );
    }

    #[test]
    #[should_panic(expected = "terminal reward for state 0 must be finite")]
    fn rejects_non_finite_terminal_reward() {
        let spec = MDPSpec {
            num_states: 1,
            num_actions: Box::new(|_s| 0),
            outcomes: Box::new(|_s, _a| Vec::new()),
            is_terminal: Some(Box::new(|_s| true)),
            terminal_reward: Some(Box::new(|_s| f64::INFINITY)),
            state_label: None,
            action_label: None,
        };
        let _ = value_iteration(spec, VIOptions::default());
    }

    /// State 0 chooses between a zero-reward self-loop (action 0) and moving to
    /// terminal state 1 for reward 1 (action 1). With γ < 1 the move wins:
    /// V = [1, 0], policy = [1, -1].
    #[test]
    fn two_state_choice_picks_terminal_move() {
        let spec = MDPSpec {
            num_states: 2,
            num_actions: Box::new(|s| if s == 0 { 2 } else { 0 }),
            outcomes: Box::new(|s, a| {
                if s == 0 && a == 0 {
                    vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: 0,
                    }]
                } else if s == 0 && a == 1 {
                    vec![Outcome {
                        prob: 1.0,
                        reward: 1.0,
                        next_state: 1,
                    }]
                } else {
                    Vec::new()
                }
            }),
            is_terminal: Some(Box::new(|s| s == 1)),
            terminal_reward: Some(Box::new(|_s| 0.0)),
            state_label: None,
            action_label: None,
        };
        let opts = VIOptions {
            gamma: 0.9,
            ..Default::default()
        };
        let res = value_iteration(spec, opts);
        assert!((res.v[0] - 1.0).abs() < 1e-6, "V[0]={}", res.v[0]);
        assert!((res.v[1] - 0.0).abs() < 1e-9, "V[1]={}", res.v[1]);
        assert_eq!(res.policy[0], 1);
        assert_eq!(res.policy[1], -1);
    }

    /// `q_value` should match the Bellman backup of the optimal value function.
    #[test]
    fn q_value_matches_optimum() {
        let rewards = [1.0_f64, 2.0_f64];
        let make_spec = || MDPSpec {
            num_states: 2,
            num_actions: Box::new(|_s| 1),
            outcomes: Box::new(move |s, _a| {
                vec![Outcome {
                    prob: 1.0,
                    reward: rewards[s],
                    next_state: s,
                }]
            }),
            is_terminal: None,
            terminal_reward: None,
            state_label: None,
            action_label: None,
        };
        let res = value_iteration(
            make_spec(),
            VIOptions {
                gamma: 0.9,
                ..Default::default()
            },
        );
        let spec = make_spec();
        let q0 = q_value(&spec, &res.v, 0, 0, 0.9);
        assert!((q0 - res.v[0]).abs() < 1e-6, "q0={q0} V0={}", res.v[0]);
        let all = q_values_all(&spec, &res.v, 1, 0.9);
        assert_eq!(all.len(), 1);
        assert!((all[0].1 - res.v[1]).abs() < 1e-6);
    }
}
