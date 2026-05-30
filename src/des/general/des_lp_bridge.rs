//! Port of `src/des/general/des-lp-bridge.ts` — DES ↔ LP integration patterns.
//!
//! In its SIMULATION framing the DES engine is not itself an exact LP solver,
//! but DES + LP is one of the most commonly deployed combinations in
//! operations research: the LP gives the "nominal optimum", DES gives
//! "operational realism" under stochastic delays, finite buffers, downtime,
//! random arrivals and queueing. In its ALGORITHMIC half the same substrate
//! *does* solve LPs natively (see `general/incremental-lp.rs` /
//! `general/stochastic-lp.rs`); the patterns below remain useful for the
//! simulation-optimisation paradigm.
//!
//! Three patterns are exposed here:
//!
//!   (A) [`solve_lp_then_simulate`] — solve the LP once for the deterministic
//!       plan, hand the plan to a DES simulator, collect realised metrics.
//!       Surfaces the gap between nominal LP optimum and stochastic realisation.
//!
//!   (B) [`build_mdp_lp`] / [`solve_mdp_as_lp`] — convert a finite-state MDP
//!       (`MDPSpec` from `value_iteration.rs`) into its primal-LP formulation
//!       (minimize the state-distribution-weighted value subject to the Bellman
//!       inequalities V(s) >= sum_s' T(s'|s,a)[r + gamma V(s')] for all s, a),
//!       and solve via the external simplex / interior-point. The optimal V*
//!       matches value iteration to ~1e-9.
//!
//!   (C) [`lp_rolling_horizon`] — MPC-style loop: solve the LP, simulate
//!       `replan_every` ticks, observe the realised state, build a fresh LP
//!       from that state, repeat. Used for production planning under
//!       uncertainty.
//!
//! Mapping notes vs. the TypeScript source:
//!   * `fn solveLPThenSimulate<M>` / `fn lpRollingHorizon<S,M>` -> generic fns
//!     with `FnMut` closure bounds (callbacks may be stateful), not boxed.
//!   * `throw` on a non-optimal LP status -> `Result<_, SolveError>`
//!     (recoverable failure, per the migration conventions).
//!   * `throw` on bad `γ` / `stateDist` -> `panic!` (invariant violation).
//!   * `(number | null)[]` bounds -> `Vec<Option<f64>>`; `number` -> `f64`;
//!     indices -> `usize`.
//!   * `interface MDPLPSolution` / `RollingHorizonStep<S,M>` -> structs.
//!   * The combined options bag (`ExternalSolverOptions & InternalSimplexOptions`)
//!     maps onto `LpSolverOptions` from `lp.rs`; the `stateDist` extension on
//!     `solveMDPAsLP`'s options is carried in [`MdpAsLpOptions`].
//!
//! NOTE on deps: every module this file needs is already ported —
//! `crate::des::general::lp` (`LPProblem`, `LPSolution`, `LPStatus`, `Sense`,
//! `solve_lp`, `LpSolverOptions`) and `crate::des::general::value_iteration`
//! (`MDPSpec`). The only *new* type introduced here is [`SolveError`], the
//! recoverable-error carrier mandated by the `throw` → `Result` conversion.

use crate::des::general::lp::{solve_lp, LPProblem, LPSolution, LPStatus, LpSolverOptions, Sense};
use crate::des::general::value_iteration::MDPSpec;

/// Recoverable failure when an LP solve returns a non-optimal status.
///
/// Replaces the TS `throw new Error(...)` on `status !== 'optimal'` (the only
/// *recoverable* failure in this file; bad `γ`/`stateDist` still `panic!`).
#[derive(Clone, Debug, PartialEq)]
pub struct SolveError {
    /// The non-optimal status reported by the solver.
    pub status: LPStatus,
    /// Human-readable diagnostic (mirrors the TS error message text).
    pub message: String,
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SolveError {}

// -----------------------------------------------------------------------------
// (A) Plan-then-simulate.
// -----------------------------------------------------------------------------

/// The `{plan, realised}` pair returned by [`solve_lp_then_simulate`].
#[derive(Clone, Debug)]
pub struct PlanRealised<M> {
    /// The optimal LP plan that was handed to the simulator.
    pub plan: LPSolution,
    /// Whatever metrics the simulator produced under realistic dynamics.
    pub realised: M,
}

/// Run the LP, then pass the optimal `x` to a DES simulator. The simulator
/// returns realised metrics (e.g. throughput, makespan), surfacing the gap
/// between nominal LP optimum and stochastic realisation.
///
/// `simulate` is the DES callback; it receives the optimal plan and returns a
/// metrics value `M`. Returns [`SolveError`] if the LP is not solved to
/// optimality (the TS `throw` becomes a recoverable `Result`).
pub fn solve_lp_then_simulate<M, F>(
    lp: &LPProblem,
    mut simulate: F,
    opts: &LpSolverOptions,
) -> Result<PlanRealised<M>, SolveError>
where
    F: FnMut(&LPSolution) -> M,
{
    let plan = solve_lp(lp, opts);
    if plan.status != LPStatus::Optimal {
        return Err(SolveError {
            status: plan.status,
            message: format!(
                "LP failed with status={}: {}",
                plan.status.as_str(),
                plan.message.clone().unwrap_or_default()
            ),
        });
    }
    let realised = simulate(&plan);
    Ok(PlanRealised { plan, realised })
}

// -----------------------------------------------------------------------------
// (B) MDP-as-LP.
//
// The Bellman optimality equation
//
//   V*(s) = max_a Σ_{s'} T(s'|s,a) [r(s,a,s') + γ V*(s')]
//
// is equivalent to the LP
//
//   min Σ_s μ_s V(s)
//   s.t.  V(s) − γ Σ_{s'} T(s'|s,a) V(s') ≥ Σ_{s'} T(s'|s,a) r(s,a,s')   ∀ s, a
//
// where μ ≻ 0 is any strictly-positive state distribution (we use uniform).
// The optimal V* is unique and matches value iteration; the optimal policy
// π*(s) is the action whose constraint is binding.
// -----------------------------------------------------------------------------

/// Result of solving a finite MDP via its LP formulation.
#[derive(Clone, Debug)]
pub struct MDPLPSolution {
    /// Value function, length `num_states`.
    pub v: Vec<f64>,
    /// Greedy policy, length `num_states`; `-1` for terminal/undefined states.
    pub policy: Vec<i32>,
    /// Raw LP result (for diagnostics).
    pub lp: LPSolution,
}

/// Options for [`solve_mdp_as_lp`]: the underlying LP solver options plus the
/// optional strictly-positive state distribution `μ`. Mirrors the TS
/// `ExternalSolverOptions & InternalSimplexOptions & {stateDist?}`.
#[derive(Clone, Debug, Default)]
pub struct MdpAsLpOptions {
    /// Options forwarded to [`solve_lp`].
    pub solver: LpSolverOptions,
    /// Strictly-positive state distribution `μ`; defaults to uniform if `None`.
    pub state_dist: Option<Vec<f64>>,
}

/// Build the primal LP for a finite MDP.
///
/// Panics if `γ ∉ [0, 1)` or if a supplied `state_dist` has the wrong length
/// (invariant violations → TS `throw`).
pub fn build_mdp_lp(mdp: &MDPSpec, gamma: f64, state_dist: Option<&[f64]>) -> LPProblem {
    let n = mdp.num_states;
    if gamma < 0.0 || gamma >= 1.0 {
        panic!("MDP-as-LP requires 0 ≤ γ < 1");
    }
    let mu: Vec<f64> = match state_dist {
        Some(d) => d.to_vec(),
        None => vec![1.0 / n as f64; n],
    };
    if mu.len() != n {
        panic!("stateDist length mismatch");
    }
    // V(s) is unbounded below for MDPs with negative rewards; allow free V_s.
    let lb: Vec<Option<f64>> = vec![None; n];
    let ub: Vec<Option<f64>> = vec![None; n];

    // For each (s, a) build the inequality:
    //   V(s) − γ Σ_{s'} T(s'|s,a) V(s')  ≥  Σ_{s'} T(s'|s,a) r(s,a,s')
    // encoded in canonical-≤ form as:
    //   −V(s) + γ Σ_{s'} T(s'|s,a) V(s')  ≤  − Σ_{s'} T(s'|s,a) r(s,a,s')
    let mut a_ub: Vec<Vec<f64>> = Vec::new();
    let mut b_ub: Vec<f64> = Vec::new();
    for s in 0..n {
        let terminal = mdp.is_terminal.as_ref().map(|f| f(s)).unwrap_or(false);
        if terminal {
            // Pin V(s) = terminal_reward(s) via two ≤ inequalities.
            let tr = mdp.terminal_reward.as_ref().map(|f| f(s)).unwrap_or(0.0);
            let mut row1 = vec![0.0; n];
            row1[s] = 1.0;
            a_ub.push(row1);
            b_ub.push(tr);
            let mut row2 = vec![0.0; n];
            row2[s] = -1.0;
            a_ub.push(row2);
            b_ub.push(-tr);
            continue;
        }
        let a_n = (mdp.num_actions)(s);
        for a in 0..a_n {
            let outcomes = (mdp.outcomes)(s, a);
            if outcomes.is_empty() {
                continue;
            }
            let mut row = vec![0.0; n];
            let mut rhs = 0.0;
            row[s] = -1.0;
            for o in &outcomes {
                row[o.next_state] += gamma * o.prob;
                rhs -= o.prob * o.reward;
            }
            a_ub.push(row);
            b_ub.push(rhs);
        }
    }

    let var_names: Vec<String> = (0..n)
        .map(|s| match &mdp.state_label {
            Some(f) => format!("V({})", f(s)),
            None => format!("V(s{s})"),
        })
        .collect();

    LPProblem {
        sense: Sense::Min,
        c: mu,
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: None,
        b_eq: None,
        lb: Some(lb),
        ub: Some(ub),
        var_names: Some(var_names),
        con_names: None,
    }
}

/// Solve a finite MDP via its LP formulation. Returns [`SolveError`] if the LP
/// is not solved to optimality.
pub fn solve_mdp_as_lp(
    mdp: &MDPSpec,
    gamma: f64,
    opts: &MdpAsLpOptions,
) -> Result<MDPLPSolution, SolveError> {
    let lp = build_mdp_lp(mdp, gamma, opts.state_dist.as_deref());
    let sol = solve_lp(&lp, &opts.solver);
    if sol.status != LPStatus::Optimal {
        return Err(SolveError {
            status: sol.status,
            message: format!(
                "MDP-LP failed with status={}: {}",
                sol.status.as_str(),
                sol.message.clone().unwrap_or_default()
            ),
        });
    }
    let v = sol.x.clone();
    // Greedy policy from V: π*(s) = argmax_a Σ T(s'|s,a)[r + γV(s')].
    let n = mdp.num_states;
    let mut policy = vec![-1_i32; n];
    for s in 0..n {
        if mdp.is_terminal.as_ref().map(|f| f(s)).unwrap_or(false) {
            continue;
        }
        let a_n = (mdp.num_actions)(s);
        let mut best_q = f64::NEG_INFINITY;
        let mut best_a: i32 = -1;
        for a in 0..a_n {
            let outcomes = (mdp.outcomes)(s, a);
            if outcomes.is_empty() {
                continue;
            }
            let mut q = 0.0;
            for o in &outcomes {
                q += o.prob * (o.reward + gamma * v[o.next_state]);
            }
            if q > best_q + 1e-12 {
                best_q = q;
                best_a = a as i32;
            }
        }
        policy[s] = best_a;
    }
    Ok(MDPLPSolution { v, policy, lp: sol })
}

// -----------------------------------------------------------------------------
// (C) LP-assisted rolling-horizon (MPC-style).
// -----------------------------------------------------------------------------

/// The value returned by the rolling-horizon `step` callback.
#[derive(Clone, Debug)]
pub struct StepResult<S, M> {
    /// State observed after running the simulator for `ticks_to_run` ticks.
    pub next_state: S,
    /// Realised metrics for this chunk.
    pub metrics: M,
}

/// One logged entry of [`lp_rolling_horizon`].
#[derive(Clone, Debug)]
pub struct RollingHorizonStep<S, M> {
    /// Global tick at which this chunk started.
    pub tick_start: usize,
    /// State at the start of the chunk.
    pub state: S,
    /// LP plan solved for this chunk.
    pub plan: LPSolution,
    /// Metrics realised over the chunk.
    pub metrics: M,
}

/// MPC-style rolling-horizon loop. Each `replan_every` ticks the simulator
/// hands us its current state; we build a fresh LP from that state, solve it,
/// and the simulator uses the new plan for the next chunk of ticks.
///
/// `build_lp(state, ticks_left)` constructs the LP for the current state;
/// `step(state, plan, ticks_to_run)` advances the simulator. Returns
/// [`SolveError`] if any LP solve fails (the TS `throw` becomes a `Result`).
pub fn lp_rolling_horizon<S, M, BuildLp, Step>(
    init_state: S,
    mut build_lp: BuildLp,
    mut step: Step,
    total_ticks: usize,
    replan_every: usize,
    opts: &LpSolverOptions,
) -> Result<Vec<RollingHorizonStep<S, M>>, SolveError>
where
    BuildLp: FnMut(&S, usize) -> LPProblem,
    Step: FnMut(&S, &LPSolution, usize) -> StepResult<S, M>,
{
    let mut log: Vec<RollingHorizonStep<S, M>> = Vec::new();
    let mut state = init_state;
    let mut t = 0usize;
    while t < total_ticks {
        let ticks_left = total_ticks - t;
        let lp = build_lp(&state, ticks_left);
        let plan = solve_lp(&lp, opts);
        if plan.status != LPStatus::Optimal {
            return Err(SolveError {
                status: plan.status,
                message: format!(
                    "rolling LP failed at t={t}: {} {}",
                    plan.status.as_str(),
                    plan.message.clone().unwrap_or_default()
                ),
            });
        }
        let ticks_to_run = replan_every.min(ticks_left);
        let result = step(&state, &plan, ticks_to_run);
        log.push(RollingHorizonStep {
            tick_start: t,
            state,
            plan,
            metrics: result.metrics,
        });
        state = result.next_state;
        t += ticks_to_run;
    }
    Ok(log)
}

// -----------------------------------------------------------------------------
// Tests.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::value_iteration::{MDPSpec, Outcome};

    /// Two independent self-loop states: V(s) = r(s) / (1 − γ).
    fn self_loop_spec(rewards: [f64; 2]) -> MDPSpec {
        MDPSpec {
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
        }
    }

    #[test]
    fn build_mdp_lp_has_expected_shape() {
        let lp = build_mdp_lp(&self_loop_spec([1.0, 2.0]), 0.9, None);
        assert_eq!(lp.sense, Sense::Min);
        assert_eq!(lp.c.len(), 2);
        // Uniform μ.
        assert!((lp.c[0] - 0.5).abs() < 1e-12 && (lp.c[1] - 0.5).abs() < 1e-12);
        // V free: every lower bound is None.
        assert!(lp.lb.as_ref().unwrap().iter().all(|b| b.is_none()));
        // One constraint per (s, a): 2 states × 1 action.
        assert_eq!(lp.a_ub.as_ref().unwrap().len(), 2);
        let names = lp.var_names.unwrap();
        assert_eq!(names[0], "V(s0)");
        assert_eq!(names[1], "V(s1)");
    }

    #[test]
    fn solve_mdp_as_lp_matches_closed_form() {
        // Force the in-process simplex so the test needs no python/scipy.
        std::env::set_var("LP_SOLVER", "internal");
        let sol = solve_mdp_as_lp(&self_loop_spec([1.0, 2.0]), 0.9, &MdpAsLpOptions::default())
            .expect("LP should solve to optimality");
        // V = r / (1 − γ) = [1, 2] / 0.1 = [10, 20].
        assert!((sol.v[0] - 10.0).abs() < 1e-6, "V0={}", sol.v[0]);
        assert!((sol.v[1] - 20.0).abs() < 1e-6, "V1={}", sol.v[1]);
        // Single legal action per state.
        assert_eq!(sol.policy, vec![0, 0]);
    }

    #[test]
    fn rolling_horizon_runs_expected_chunks() {
        std::env::set_var("LP_SOLVER", "internal");
        let opts = LpSolverOptions::default();
        // State = remaining budget; LP each chunk is `max x s.t. x ≤ state`.
        let log = lp_rolling_horizon(
            5.0_f64,
            |state: &f64, _ticks_left: usize| LPProblem {
                sense: Sense::Max,
                c: vec![1.0],
                a_ub: Some(vec![vec![1.0]]),
                b_ub: Some(vec![*state]),
                ..Default::default()
            },
            |state: &f64, plan: &LPSolution, _ticks_to_run: usize| StepResult {
                next_state: state - 1.0,
                metrics: plan.objective,
            },
            4, // total_ticks
            2, // replan_every
            &opts,
        )
        .expect("all rolling LPs should solve");

        // total=4, replan_every=2 -> two chunks at t=0 and t=2.
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].tick_start, 0);
        assert_eq!(log[1].tick_start, 2);
        // max x s.t. x ≤ state -> objective == state.
        assert!((log[0].metrics - 5.0).abs() < 1e-6, "m0={}", log[0].metrics);
        assert!((log[1].metrics - 4.0).abs() < 1e-6, "m1={}", log[1].metrics);
    }
}
