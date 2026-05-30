//! Port of `src/des/general/dispatch.ts` — module `des::general::dispatch`.
//!
//! Multi-class parallel-server dispatch problem plus six policies and an
//! evaluation harness. The problem: M heterogeneous machines, K job classes,
//! Poisson(λ) arrivals with class mix `class_prob`, class-c jobs taking
//! Exp(μ_{c,m}) on machine m. The decision at each arrival is which machine to
//! dispatch to; the cost is long-run mean sojourn time.
//!
//! The six policies (random, round-robin, shortest-queue, SECT, fluid-LP,
//! MDP-via-VI, MCTS) are all evaluated by the SAME DES with the same seeds so
//! head-to-head comparisons are fair.
//!
//! Conversion notes from the TS source:
//!   * Policies are objects implementing `DispatchPolicy` -> one struct per
//!     policy implementing the [`DispatchPolicy`] trait. `pick` takes
//!     `&mut self` because several policies carry mutable RNG / counter state.
//!   * `mulberry32(seed)` + `expSample` / `categorical` -> injected
//!     [`SeededRandom`]; the same seed gives a fair comparison.
//!   * `PureTransform` subclasses -> structs implementing
//!     [`Transform`](crate::des::shared::transform::Transform).
//!   * Builds on `lp` / `value_iteration` / `mcts` (see their headers).
//!   * `throw new Error` (illegal machine / LP failure) -> `panic!`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::des::general::lp::{solve_lp, LPProblem, LPStatus, LpSolverOptions, Sense};
use crate::des::general::mcts::{mcts, ApplyResult, MCTSEnv, MCTSOptions, Selection};
use crate::des::general::prng::mulberry32;
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::Transform;

/// Problem parameters (TS `interface DispatchProblem`).
#[derive(Clone, Debug)]
pub struct DispatchProblem {
    /// Number of machines.
    pub m: usize,
    /// Number of job classes.
    pub k: usize,
    /// Arrival rate λ.
    pub arrival_rate: f64,
    /// Length-K class probabilities, summing to 1.
    pub class_prob: Vec<f64>,
    /// K × M matrix of service rates μ_{c,m}.
    pub service_rate: Vec<Vec<f64>>,
}

/// Exponential sample with the given rate (TS `expSample`).
fn exp_sample(rate: f64, rng: &mut SeededRandom) -> f64 {
    -(f64::max(1e-12, rng.next_float())).ln() / rate
}

/// Discrete sample from a probability vector (TS `categorical`).
fn categorical(p: &[f64], rng: &mut SeededRandom) -> usize {
    let u = rng.next_float();
    let mut cum = 0.0;
    for (i, pi) in p.iter().enumerate() {
        cum += *pi;
        if u < cum {
            return i;
        }
    }
    p.len() - 1
}

// =============================================================================
// DES simulator state + policy protocol.
// =============================================================================

/// State exposed to a policy at a dispatch decision (TS `interface
/// DispatchState`).
#[derive(Clone, Debug)]
pub struct DispatchState {
    pub m: usize,
    pub k: usize,
    /// Per-machine queue length (jobs waiting + 1 if currently serving).
    pub q: Vec<i64>,
    /// Per-machine idle-until time; machine m is idle once `now >= idle_until[m]`.
    pub idle_until: Vec<f64>,
    /// Per-machine class currently in service (-1 if idle).
    pub in_service: Vec<i64>,
    /// Current simulation clock.
    pub now: f64,
}

/// A pending (queued) job (TS `interface PendingJob`).
#[derive(Clone, Copy, Debug)]
struct PendingJob {
    arrival_time: f64,
    class_of: usize,
}

/// A dispatch policy (TS `interface DispatchPolicy`). `pick` may mutate the
/// policy's own state (RNG / round-robin cursor).
pub trait DispatchPolicy {
    /// Choose a machine in `[0, M)` for an arriving class-`c` job.
    fn pick(&mut self, state: &DispatchState, c: usize) -> usize;
    /// Optional reset hook called before each replication.
    fn reset(&mut self) {}
}

/// Outcome of one DES replication (TS `interface DispatchResult`).
#[derive(Clone, Debug)]
pub struct DispatchResult {
    pub mean_sojourn: f64,
    pub completed_jobs: usize,
    pub per_machine_jobs: Vec<i64>,
    pub per_machine_utilisation: Vec<f64>,
}

/// Inputs to one DES replication: the (problem, policy) pair being simulated.
pub struct SimulateDispatchInput<'a> {
    pub problem: &'a DispatchProblem,
    pub policy: &'a mut dyn DispatchPolicy,
}

/// Single DES replication. Configuration (run length, seed, warmup) lives on
/// the struct; the (problem, policy) pair is the transform input (TS
/// `class SimulateDispatch extends PureTransform`).
pub struct SimulateDispatch {
    pub num_arrivals: usize,
    pub seed: u32,
    pub warmup: usize,
}

impl SimulateDispatch {
    pub fn new(num_arrivals: usize, seed: u32, warmup: usize) -> Self {
        SimulateDispatch {
            num_arrivals,
            seed,
            warmup,
        }
    }
}

impl<'a> Transform<SimulateDispatchInput<'a>, DispatchResult> for SimulateDispatch {
    fn transform(&self, input: SimulateDispatchInput<'a>) -> DispatchResult {
        let problem = input.problem;
        let policy = input.policy;
        let m = problem.m;
        let k = problem.k;
        let arrival_rate = problem.arrival_rate;
        let class_prob = &problem.class_prob;
        let service_rate = &problem.service_rate;

        let mut rng = mulberry32(self.seed);
        policy.reset();

        let mut queue: Vec<Vec<PendingJob>> = vec![Vec::new(); m];
        let mut idle_until = vec![0.0f64; m];
        let mut in_service = vec![-1i64; m];

        let mut next_arrival = exp_sample(arrival_rate, &mut rng);
        let mut now = 0.0f64;
        let mut arrivals_seen = 0usize;
        let mut total_sojourn = 0.0f64;
        let mut completed_jobs = 0usize;
        let mut per_machine_jobs = vec![0i64; m];
        let mut per_machine_busy = vec![0.0f64; m];

        while arrivals_seen < self.num_arrivals {
            // Find the next event: arrival, or earliest service completion.
            let mut is_departure = false;
            let mut next_time = next_arrival;
            let mut next_machine = 0usize;
            for mm in 0..m {
                if !queue[mm].is_empty() && idle_until[mm] < next_time {
                    is_departure = true;
                    next_time = idle_until[mm];
                    next_machine = mm;
                }
            }
            // Busy-time accounting up to next_time.
            let dt = next_time - now;
            for mm in 0..m {
                if !queue[mm].is_empty() {
                    per_machine_busy[mm] += dt.min(idle_until[mm] - now);
                }
            }
            now = next_time;

            if !is_departure {
                let c = categorical(class_prob, &mut rng);
                let state = DispatchState {
                    m,
                    k,
                    q: queue.iter().map(|qm| qm.len() as i64).collect(),
                    idle_until: idle_until.clone(),
                    in_service: in_service.clone(),
                    now,
                };
                let chosen = policy.pick(&state, c);
                if chosen >= m {
                    panic!("policy returned illegal machine {chosen}");
                }
                queue[chosen].push(PendingJob {
                    arrival_time: now,
                    class_of: c,
                });
                // Start service if the machine was idle.
                if queue[chosen].len() == 1 && idle_until[chosen] <= now {
                    let mu = service_rate[c][chosen];
                    idle_until[chosen] = now + exp_sample(mu, &mut rng);
                    in_service[chosen] = c as i64;
                }
                arrivals_seen += 1;
                per_machine_jobs[chosen] += 1;
                next_arrival = now + exp_sample(arrival_rate, &mut rng);
            } else {
                // Service completion on `next_machine`.
                let job = queue[next_machine].remove(0);
                let sojourn = now - job.arrival_time;
                if arrivals_seen > self.warmup {
                    total_sojourn += sojourn;
                    completed_jobs += 1;
                }
                if !queue[next_machine].is_empty() {
                    let head_class = queue[next_machine][0].class_of;
                    let mu = service_rate[head_class][next_machine];
                    idle_until[next_machine] = now + exp_sample(mu, &mut rng);
                    in_service[next_machine] = head_class as i64;
                } else {
                    idle_until[next_machine] = now;
                    in_service[next_machine] = -1;
                }
            }
        }

        let mean_sojourn = if completed_jobs > 0 {
            total_sojourn / completed_jobs as f64
        } else {
            f64::NAN
        };
        DispatchResult {
            mean_sojourn,
            completed_jobs,
            per_machine_jobs,
            per_machine_utilisation: per_machine_busy.iter().map(|b| b / now).collect(),
        }
    }
}

/// Run a single DES replication (TS `simulateDispatch`).
pub fn simulate_dispatch(
    problem: &DispatchProblem,
    policy: &mut dyn DispatchPolicy,
    num_arrivals: usize,
    seed: u32,
    warmup: usize,
) -> DispatchResult {
    SimulateDispatch::new(num_arrivals, seed, warmup)
        .transform(SimulateDispatchInput { problem, policy })
}

// =============================================================================
// Policies.
// =============================================================================

/// Uniform-random machine (TS `policyRandom`).
pub struct PolicyRandom {
    seed: u32,
    rng: SeededRandom,
}

impl DispatchPolicy for PolicyRandom {
    fn pick(&mut self, state: &DispatchState, _c: usize) -> usize {
        (self.rng.next_float() * state.m as f64).floor() as usize
    }
    fn reset(&mut self) {
        self.rng = mulberry32(self.seed);
    }
}

pub fn policy_random(seed: u32) -> PolicyRandom {
    PolicyRandom {
        seed,
        rng: mulberry32(seed),
    }
}

/// Round-robin (TS `policyRoundRobin`).
pub struct PolicyRoundRobin {
    i: usize,
}

impl DispatchPolicy for PolicyRoundRobin {
    fn pick(&mut self, state: &DispatchState, _c: usize) -> usize {
        let m = self.i % state.m;
        self.i += 1;
        m
    }
    fn reset(&mut self) {
        self.i = 0;
    }
}

pub fn policy_round_robin() -> PolicyRoundRobin {
    PolicyRoundRobin { i: 0 }
}

/// Shortest-queue (TS `policyShortestQueue`).
pub struct PolicyShortestQueue;

impl DispatchPolicy for PolicyShortestQueue {
    fn pick(&mut self, state: &DispatchState, _c: usize) -> usize {
        let mut best_m = 0;
        for mm in 1..state.m {
            if state.q[mm] < state.q[best_m] {
                best_m = mm;
            }
        }
        best_m
    }
}

pub fn policy_shortest_queue() -> PolicyShortestQueue {
    PolicyShortestQueue
}

/// Shortest-expected-completion-time: argmin_m (q_m + 1) / μ_{c,m} (TS
/// `policySECT`).
pub struct PolicySect {
    service_rate: Vec<Vec<f64>>,
}

impl DispatchPolicy for PolicySect {
    fn pick(&mut self, state: &DispatchState, c: usize) -> usize {
        let mut best_m = 0;
        let mut best_t = f64::INFINITY;
        for mm in 0..state.m {
            let t = (state.q[mm] as f64 + 1.0) / f64::max(1e-12, self.service_rate[c][mm]);
            if t < best_t {
                best_t = t;
                best_m = mm;
            }
        }
        best_m
    }
}

pub fn policy_sect(problem: &DispatchProblem) -> PolicySect {
    PolicySect {
        service_rate: problem.service_rate.clone(),
    }
}

// -----------------------------------------------------------------------------
// Fluid LP relaxation.
// -----------------------------------------------------------------------------

/// Build the fluid-relaxation LP (TS `BuildDispatchFluidLP`).
pub struct BuildDispatchFluidLP;

impl Transform<DispatchProblem, LPProblem> for BuildDispatchFluidLP {
    fn transform(&self, problem: DispatchProblem) -> LPProblem {
        let m = problem.m;
        let k = problem.k;
        let arrival_rate = problem.arrival_rate;
        let class_prob = &problem.class_prob;
        let service_rate = &problem.service_rate;

        let n = k * m + 1; // x_{c,m} for c×m, plus t
        let t_idx = k * m;
        let mut c_obj = vec![0.0; n];
        c_obj[t_idx] = 1.0;

        // Σ_m x_{c,m} = 1 for each c.
        let mut a_eq: Vec<Vec<f64>> = Vec::new();
        let mut b_eq: Vec<f64> = Vec::new();
        for c in 0..k {
            let mut row = vec![0.0; n];
            for mm in 0..m {
                row[c * m + mm] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(1.0);
        }
        // λ Σ_c p_c x_{c,m} / μ_{c,m} − t ≤ 0 for each m.
        let mut a_ub: Vec<Vec<f64>> = Vec::new();
        let mut b_ub: Vec<f64> = Vec::new();
        for mm in 0..m {
            let mut row = vec![0.0; n];
            for c in 0..k {
                row[c * m + mm] =
                    arrival_rate * class_prob[c] / f64::max(1e-12, service_rate[c][mm]);
            }
            row[t_idx] = -1.0;
            a_ub.push(row);
            b_ub.push(0.0);
        }

        let mut var_names: Vec<String> = (0..k * m)
            .map(|i| format!("x_{}_{}", i / m + 1, i % m + 1))
            .collect();
        var_names.push("t".to_string());
        let mut con_names: Vec<String> = (0..k)
            .map(|c| format!("class-{} fully served", c + 1))
            .collect();
        con_names.extend((0..m).map(|mm| format!("machine-{} <= t", mm + 1)));

        LPProblem {
            sense: Sense::Min,
            c: c_obj,
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: Some(a_eq),
            b_eq: Some(b_eq),
            lb: None,
            ub: None,
            var_names: Some(var_names),
            con_names: Some(con_names),
        }
    }
}

pub fn build_dispatch_fluid_lp(problem: &DispatchProblem) -> LPProblem {
    BuildDispatchFluidLP.transform(problem.clone())
}

/// Randomized fluid-LP policy: dispatch class c to machine m with probability
/// x*_{c,m} (TS `policyFluidLP` policy object).
pub struct FluidLpPolicy {
    x: Vec<Vec<f64>>,
    seed: u32,
    rng: SeededRandom,
}

impl DispatchPolicy for FluidLpPolicy {
    fn pick(&mut self, _state: &DispatchState, c: usize) -> usize {
        categorical(&self.x[c], &mut self.rng)
    }
    fn reset(&mut self) {
        self.rng = mulberry32(self.seed);
    }
}

/// Solved fluid-LP policy plus diagnostics (TS `interface FluidLPPolicyResult`).
pub struct FluidLpPolicyResult {
    pub policy: FluidLpPolicy,
    /// K × M assignment fractions.
    pub x: Vec<Vec<f64>>,
    /// t* = max_m ρ_m at the LP optimum.
    pub bottleneck_load: f64,
    pub solver: String,
    pub iters: usize,
}

/// Solve the fluid LP and build a randomized policy (TS `PolicyFluidLP`).
pub struct PolicyFluidLP {
    pub seed: u32,
}

impl Transform<DispatchProblem, FluidLpPolicyResult> for PolicyFluidLP {
    fn transform(&self, problem: DispatchProblem) -> FluidLpPolicyResult {
        let seed = self.seed;
        let lp = build_dispatch_fluid_lp(&problem);
        let sol = solve_lp(&lp, &LpSolverOptions::default());
        if sol.status != LPStatus::Optimal {
            panic!(
                "fluid LP failed: status={}: {}",
                sol.status.as_str(),
                sol.message.clone().unwrap_or_default()
            );
        }
        let m = problem.m;
        let k = problem.k;
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(k);
        for c in 0..k {
            let mut row: Vec<f64> = Vec::with_capacity(m);
            let mut s = 0.0;
            for mm in 0..m {
                let v = f64::max(0.0, sol.x[c * m + mm]);
                row.push(v);
                s += v;
            }
            if s > 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            x.push(row);
        }
        let policy = FluidLpPolicy {
            x: x.clone(),
            seed,
            rng: mulberry32(seed),
        };

        FluidLpPolicyResult {
            policy,
            x,
            bottleneck_load: sol.x[k * m],
            solver: sol.solver.clone(),
            iters: sol.iters.unwrap_or(0),
        }
    }
}

pub fn policy_fluid_lp(problem: &DispatchProblem, seed: u32) -> FluidLpPolicyResult {
    PolicyFluidLP { seed }.transform(problem.clone())
}

// -----------------------------------------------------------------------------
// MDP via value iteration on an empirical kernel.
// -----------------------------------------------------------------------------

/// Options for [`PolicyMDPVI`] (TS `interface MDPVIPolicyOptions`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MdpViPolicyOptions {
    pub q_max: Option<usize>,
    pub gamma: Option<f64>,
    pub rollouts_per_sa: Option<usize>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub seed: Option<u32>,
}

/// State index encode (TS `encode`).
fn encode_state(q: &[i64], c: usize, q_max: usize, m: usize, k: usize) -> usize {
    let q1 = q_max + 1;
    let mut idx = 0usize;
    for &qm in q.iter().take(m) {
        let clamped = qm.clamp(0, q_max as i64) as usize;
        idx = idx * q1 + clamped;
    }
    idx * k + c
}

/// State index decode (TS `decode`).
fn decode_state(s: usize, q_max: usize, m: usize, k: usize) -> (Vec<i64>, usize) {
    let q1 = q_max + 1;
    let c = s % k;
    let mut q_idx = s / k;
    let mut q = vec![0i64; m];
    for mm in (0..m).rev() {
        q[mm] = (q_idx % q1) as i64;
        q_idx /= q1;
    }
    (q, c)
}

/// MDP-VI policy: argmax over the precomputed Q-table (TS `policyMDPVI` policy
/// object).
pub struct MdpViPolicy {
    q_table: Vec<Vec<f64>>,
    q_max: usize,
    m: usize,
    k: usize,
}

impl DispatchPolicy for MdpViPolicy {
    fn pick(&mut self, state: &DispatchState, c: usize) -> usize {
        let clamped: Vec<i64> = state
            .q
            .iter()
            .map(|&qq| qq.min(self.q_max as i64))
            .collect();
        let s_idx = encode_state(&clamped, c, self.q_max, self.m, self.k);
        let row = &self.q_table[s_idx];
        let mut best_a = 0;
        for a in 1..self.m {
            if row[a] > row[best_a] {
                best_a = a;
            }
        }
        best_a
    }
}

/// Built & solved MDP-VI policy plus diagnostics (TS `interface
/// MDPVIPolicyResult`).
pub struct MdpViPolicyResult {
    pub policy: MdpViPolicy,
    pub v: Vec<f64>,
    pub q: Vec<Vec<f64>>,
    pub q_max: usize,
    pub num_states: usize,
}

/// Build & solve a tabular MDP whose transitions/rewards are estimated by an
/// empirical fluid kernel, then run value iteration (TS `PolicyMDPVI`).
pub struct PolicyMDPVI {
    pub opts: MdpViPolicyOptions,
}

impl Transform<DispatchProblem, MdpViPolicyResult> for PolicyMDPVI {
    fn transform(&self, problem: DispatchProblem) -> MdpViPolicyResult {
        let opts = self.opts;
        let q_max = opts.q_max.unwrap_or(5);
        let gamma = opts.gamma.unwrap_or(0.95);
        let r_rollouts = opts.rollouts_per_sa.unwrap_or(60);
        let seed = opts.seed.unwrap_or(99);
        let m = problem.m;
        let k = problem.k;
        let arrival_rate = problem.arrival_rate;
        let class_prob = &problem.class_prob;
        let service_rate = &problem.service_rate;

        let q1 = q_max + 1;
        let num_q_states = q1.pow(m as u32);
        let num_states = num_q_states * k;

        // Average per-machine service rate (fluid approximation).
        let mu_bar: Vec<f64> = (0..m)
            .map(|mm| (0..k).map(|c| class_prob[c] * service_rate[c][mm]).sum())
            .collect();

        let mut rng = mulberry32(seed);
        // PORT NOTE: a `BTreeMap` (ordered by next-state index) replaces the TS
        // `Map` so the emitted outcome lists are deterministic across runs;
        // value iteration is order-independent so this is behaviour-neutral.
        let mut outcomes_by_a: Vec<Vec<Vec<Outcome>>> = Vec::with_capacity(num_states);
        for s in 0..num_states {
            let (q, c) = decode_state(s, q_max, m, k);
            let mut per_action: Vec<Vec<Outcome>> = Vec::with_capacity(m);
            for a in 0..m {
                let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
                let mut total_reward = 0.0;
                for _ in 0..r_rollouts {
                    let mut q_new = q.clone();
                    q_new[a] = (q_new[a] + 1).min(q_max as i64);
                    let dt = exp_sample(arrival_rate, &mut rng);
                    let mut q_trans = q_new.clone();
                    for mm in 0..m {
                        if q_trans[mm] == 0 {
                            continue;
                        }
                        let lambda_m = mu_bar[mm] * dt;
                        let mut k_comp = 0i64;
                        let mut p = (-lambda_m).exp();
                        let u = rng.next_float();
                        let mut cum = p;
                        while u > cum && k_comp < q_trans[mm] {
                            k_comp += 1;
                            p *= lambda_m / k_comp as f64;
                            cum += p;
                        }
                        q_trans[mm] = (q_trans[mm] - k_comp).max(0);
                    }
                    let c_new = categorical(class_prob, &mut rng);
                    let s_next = encode_state(&q_trans, c_new, q_max, m, k);
                    *counts.entry(s_next).or_insert(0) += 1;
                    total_reward += -(q_new[a] as f64) / f64::max(1e-12, service_rate[c][a]);
                }
                let mean_reward = total_reward / r_rollouts as f64;
                let out: Vec<Outcome> = counts
                    .iter()
                    .map(|(s2, cnt)| Outcome {
                        prob: *cnt as f64 / r_rollouts as f64,
                        reward: mean_reward,
                        next_state: *s2,
                    })
                    .collect();
                per_action.push(out);
            }
            outcomes_by_a.push(per_action);
        }

        let oba = Rc::new(outcomes_by_a);
        let oba_for_closure = oba.clone();
        let mdp = MDPSpec {
            num_states,
            num_actions: Box::new(move |_s| m),
            outcomes: Box::new(move |s, a| oba_for_closure[s][a].clone()),
            is_terminal: None,
            terminal_reward: None,
            state_label: None,
            action_label: None,
        };
        let vi = value_iteration(
            mdp,
            VIOptions {
                gamma,
                tol: opts.tol.unwrap_or(1e-8),
                max_iter: opts.max_iter.unwrap_or(5000),
                validate_probs: false,
                ..Default::default()
            },
        );

        let mut q_table: Vec<Vec<f64>> = Vec::with_capacity(num_states);
        for s in 0..num_states {
            let mut row: Vec<f64> = Vec::with_capacity(m);
            for a in 0..m {
                let mut qv = 0.0;
                for o in &oba[s][a] {
                    qv += o.prob * (o.reward + gamma * vi.v[o.next_state]);
                }
                row.push(qv);
            }
            q_table.push(row);
        }

        let q_for_result = q_table.clone();
        let policy = MdpViPolicy {
            q_table,
            q_max,
            m,
            k,
        };

        MdpViPolicyResult {
            policy,
            v: vi.v.clone(),
            q: q_for_result,
            q_max,
            num_states,
        }
    }
}

pub fn policy_mdp_vi(problem: &DispatchProblem, opts: MdpViPolicyOptions) -> MdpViPolicyResult {
    PolicyMDPVI { opts }.transform(problem.clone())
}

// -----------------------------------------------------------------------------
// MCTS — DES-driven search over the next K decision epochs.
// -----------------------------------------------------------------------------

/// Options for [`policy_mcts`] (TS `interface MCTSPolicyOptions`).
///
/// PORT NOTE: the TS `rolloutPolicy?` field is dropped — the TS code computes
/// `rolloutPol = opts.rolloutPolicy ?? policySECT(problem)` but never references
/// it; the search environment hardcodes a SECT rollout regardless.
#[derive(Clone, Copy, Debug, Default)]
pub struct MctsPolicyOptions {
    pub iterations: Option<usize>,
    pub rollout_depth: Option<usize>,
    pub c: Option<f64>,
    pub gamma: Option<f64>,
    pub seed: Option<u32>,
}

/// MCTS look-ahead state (TS `interface MCTSDispatchState`).
#[derive(Clone)]
struct MctsDispatchState {
    q: Vec<i64>,
    /// Per-machine head class (-1 idle, ≥0 known class, -2 unknown/approximated).
    head_class: Vec<i64>,
    /// Per-machine FIFO of queued classes behind the head.
    class_queue: Vec<Vec<i64>>,
    idle_until: Vec<f64>,
    /// Upcoming arrivals already sampled (look-ahead).
    arrival_q: Vec<PendingJob>,
    /// Index into `arrival_q` of the next decision.
    cursor: usize,
    now: f64,
    rng_seed: u32,
}

/// Search environment closing over the problem (TS inline `env` object).
struct DispatchMctsEnv {
    m: usize,
    k: usize,
    arrival_rate: f64,
    class_prob: Vec<f64>,
    service_rate: Vec<Vec<f64>>,
    rollout_depth: usize,
    gamma: f64,
}

/// Clone a state, advancing its LCG seed (TS `cloneState`).
fn clone_state(s: &MctsDispatchState) -> MctsDispatchState {
    let mut out = s.clone();
    out.rng_seed = s.rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
    out
}

/// Advance one decision epoch and apply `action` (TS `advance`).
fn advance(
    env: &DispatchMctsEnv,
    s: &MctsDispatchState,
    action: usize,
) -> ApplyResult<MctsDispatchState> {
    let mut out = clone_state(s);
    let mut local_rng = mulberry32(out.rng_seed);
    if out.cursor >= out.arrival_q.len() {
        return ApplyResult {
            next: out,
            reward: 0.0,
            done: true,
        };
    }
    let head = out.arrival_q[out.cursor];
    // Advance machines forward to the head arrival time, in event-time order.
    while out.now < head.arrival_time {
        let mut first_m: i64 = -1;
        let mut first_t = head.arrival_time;
        for mm in 0..env.m {
            if out.q[mm] > 0 && out.idle_until[mm] < first_t {
                first_m = mm as i64;
                first_t = out.idle_until[mm];
            }
        }
        if first_m == -1 {
            out.now = head.arrival_time;
            break;
        }
        let fm = first_m as usize;
        out.now = first_t;
        out.q[fm] -= 1;
        if out.q[fm] > 0 {
            let new_head = if out.class_queue[fm].is_empty() {
                None
            } else {
                Some(out.class_queue[fm].remove(0))
            };
            out.head_class[fm] = new_head.unwrap_or(-2);
            let mu_new = match new_head {
                Some(nh) if nh >= 0 => env.service_rate[nh as usize][fm],
                _ => (0..env.k)
                    .map(|c| env.class_prob[c] * env.service_rate[c][fm])
                    .sum(),
            };
            out.idle_until[fm] = out.now + exp_sample(mu_new, &mut local_rng);
        } else {
            out.head_class[fm] = -1;
            out.idle_until[fm] = out.now;
        }
    }
    out.now = head.arrival_time;
    // Dispatch the head to `action`.
    if out.q[action] == 0 {
        out.head_class[action] = head.class_of as i64;
        out.idle_until[action] =
            out.now + exp_sample(env.service_rate[head.class_of][action], &mut local_rng);
    } else {
        out.class_queue[action].push(head.class_of as i64);
    }
    out.q[action] += 1;
    let sojourn_est =
        out.q[action] as f64 / f64::max(1e-12, env.service_rate[head.class_of][action]);
    out.cursor += 1;
    if out.cursor >= out.arrival_q.len() {
        out.arrival_q.push(PendingJob {
            arrival_time: out.now + exp_sample(env.arrival_rate, &mut local_rng),
            class_of: categorical(&env.class_prob, &mut local_rng),
        });
    }
    let done = out.cursor >= env.rollout_depth + s.cursor;
    ApplyResult {
        next: out,
        reward: -sojourn_est,
        done,
    }
}

impl MCTSEnv<MctsDispatchState> for DispatchMctsEnv {
    fn num_actions(&self, _s: &MctsDispatchState) -> usize {
        self.m
    }
    fn apply_action(&self, s: &MctsDispatchState, a: usize) -> ApplyResult<MctsDispatchState> {
        advance(self, s, a)
    }
    fn is_terminal(&self, s: &MctsDispatchState) -> bool {
        s.cursor >= s.arrival_q.len()
    }
    fn rollout_policy(&self, s: &MctsDispatchState, _rng: &mut dyn RandomSource) -> usize {
        // SECT-style choice on the look-ahead head.
        if s.cursor >= s.arrival_q.len() {
            return 0;
        }
        let head = s.arrival_q[s.cursor];
        let mut best_m = 0;
        let mut best_t = f64::INFINITY;
        for mm in 0..self.m {
            let t = (s.q[mm] as f64 + 1.0) / f64::max(1e-12, self.service_rate[head.class_of][mm]);
            if t < best_t {
                best_t = t;
                best_m = mm;
            }
        }
        best_m
    }
    fn rollout_depth(&self) -> usize {
        self.rollout_depth
    }
    fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// MCTS policy: per `pick`, build a look-ahead horizon and run UCT (TS
/// `policyMCTS` policy object).
pub struct MctsPolicy {
    m: usize,
    k: usize,
    arrival_rate: f64,
    class_prob: Vec<f64>,
    service_rate: Vec<Vec<f64>>,
    iterations: usize,
    rollout_depth: usize,
    c: f64,
    gamma: f64,
    next_seed: u32,
}

impl DispatchPolicy for MctsPolicy {
    fn pick(&mut self, state: &DispatchState, class_of: usize) -> usize {
        let mut local_rng = mulberry32(self.next_seed);
        self.next_seed = self.next_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let mut arrival_q = vec![PendingJob {
            arrival_time: state.now,
            class_of,
        }];
        let mut t = state.now;
        for _ in 0..self.rollout_depth + 4 {
            t += exp_sample(self.arrival_rate, &mut local_rng);
            arrival_q.push(PendingJob {
                arrival_time: t,
                class_of: categorical(&self.class_prob, &mut local_rng),
            });
        }
        let root = MctsDispatchState {
            q: state.q.clone(),
            head_class: state
                .in_service
                .iter()
                .map(|&c| if c < 0 { -1 } else { c })
                .collect(),
            class_queue: state
                .q
                .iter()
                .map(|&qm| vec![-2i64; (qm - 1).max(0) as usize])
                .collect(),
            idle_until: state.idle_until.clone(),
            arrival_q,
            cursor: 0,
            now: state.now,
            rng_seed: if self.next_seed == 0 {
                1
            } else {
                self.next_seed
            },
        };
        let env = DispatchMctsEnv {
            m: self.m,
            k: self.k,
            arrival_rate: self.arrival_rate,
            class_prob: self.class_prob.clone(),
            service_rate: self.service_rate.clone(),
            rollout_depth: self.rollout_depth,
            gamma: self.gamma,
        };
        let opts = MCTSOptions {
            iterations: self.iterations,
            c: self.c,
            selection: Selection::Visits,
        };
        let result = mcts(Box::new(env), root, opts, local_rng);
        result.action
    }
}

pub fn policy_mcts(problem: &DispatchProblem, opts: MctsPolicyOptions) -> MctsPolicy {
    MctsPolicy {
        m: problem.m,
        k: problem.k,
        arrival_rate: problem.arrival_rate,
        class_prob: problem.class_prob.clone(),
        service_rate: problem.service_rate.clone(),
        iterations: opts.iterations.unwrap_or(80),
        rollout_depth: opts.rollout_depth.unwrap_or(30),
        c: opts.c.unwrap_or(std::f64::consts::SQRT_2),
        gamma: opts.gamma.unwrap_or(0.97),
        next_seed: opts.seed.unwrap_or(7),
    }
}

// =============================================================================
// Evaluation harness.
// =============================================================================

/// Per-policy evaluation outcome (TS `interface EvaluationResult`).
#[derive(Clone, Debug)]
pub struct EvaluationResult {
    pub policy_name: String,
    pub mean_wait: f64,
    pub sd_wait: f64,
    pub raw_waits: Vec<f64>,
    pub utilisation: Vec<f64>,
}

/// Inputs to [`EvaluatePolicy`] (TS `interface EvaluatePolicyInput`).
pub struct EvaluatePolicyInput {
    pub problem: DispatchProblem,
    pub factory: Box<dyn Fn() -> Box<dyn DispatchPolicy>>,
}

/// Configuration for [`EvaluatePolicy`] (TS `interface EvaluatePolicyConfig`).
#[derive(Clone, Debug)]
pub struct EvaluatePolicyConfig {
    pub policy_name: String,
    pub num_replications: usize,
    pub num_arrivals_per_rep: usize,
    pub seed_base: u32,
    pub warmup: Option<usize>,
}

/// Replicate the DES across seeds and pool the statistics (TS `EvaluatePolicy`).
pub struct EvaluatePolicy {
    pub config: EvaluatePolicyConfig,
}

impl Transform<EvaluatePolicyInput, EvaluationResult> for EvaluatePolicy {
    fn transform(&self, input: EvaluatePolicyInput) -> EvaluationResult {
        let warmup = self.config.warmup.unwrap_or(0);
        let mut waits: Vec<f64> = Vec::new();
        let mut utils: Vec<Vec<f64>> = Vec::new();
        for r in 0..self.config.num_replications {
            let mut policy = (input.factory)();
            let result = simulate_dispatch(
                &input.problem,
                policy.as_mut(),
                self.config.num_arrivals_per_rep,
                self.config.seed_base + r as u32,
                warmup,
            );
            waits.push(result.mean_sojourn);
            utils.push(result.per_machine_utilisation);
        }
        let mean = waits.iter().sum::<f64>() / waits.len() as f64;
        let denom = ((waits.len() as i64 - 1).max(1)) as f64;
        let sd = (waits.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / denom).sqrt();
        let m = utils[0].len();
        let mut util_mean = vec![0.0; m];
        for u in &utils {
            for j in 0..m {
                util_mean[j] += u[j] / utils.len() as f64;
            }
        }
        EvaluationResult {
            policy_name: self.config.policy_name.clone(),
            mean_wait: mean,
            sd_wait: sd,
            raw_waits: waits,
            utilisation: util_mean,
        }
    }
}

pub fn evaluate_policy(
    problem: &DispatchProblem,
    factory: Box<dyn Fn() -> Box<dyn DispatchPolicy>>,
    policy_name: &str,
    num_replications: usize,
    num_arrivals_per_rep: usize,
    seed_base: u32,
    warmup: usize,
) -> EvaluationResult {
    EvaluatePolicy {
        config: EvaluatePolicyConfig {
            policy_name: policy_name.to_string(),
            num_replications,
            num_arrivals_per_rep,
            seed_base,
            warmup: Some(warmup),
        },
    }
    .transform(EvaluatePolicyInput {
        problem: problem.clone(),
        factory,
    })
}

/// Welch's t-test for difference of means (TS `WelchT`).
pub struct WelchTInput {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

pub struct WelchT;

impl Transform<WelchTInput, f64> for WelchT {
    fn transform(&self, input: WelchTInput) -> f64 {
        let WelchTInput { a, b } = input;
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        let va =
            a.iter().map(|v| (v - ma).powi(2)).sum::<f64>() / ((a.len() as i64 - 1).max(1) as f64);
        let vb =
            b.iter().map(|v| (v - mb).powi(2)).sum::<f64>() / ((b.len() as i64 - 1).max(1) as f64);
        (ma - mb) / (va / a.len() as f64 + vb / b.len() as f64 + 1e-30).sqrt()
    }
}

pub fn welch_t(a: Vec<f64>, b: Vec<f64>) -> f64 {
    WelchT.transform(WelchTInput { a, b })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_problem() -> DispatchProblem {
        // 2 machines, 2 classes. Machine 0 fast for class 0, machine 1 for class 1.
        DispatchProblem {
            m: 2,
            k: 2,
            arrival_rate: 1.2,
            class_prob: vec![0.5, 0.5],
            service_rate: vec![vec![2.0, 0.8], vec![0.8, 2.0]],
        }
    }

    #[test]
    fn simulate_runs_and_reports_sojourn() {
        let problem = small_problem();
        let mut policy = policy_shortest_queue();
        let res = simulate_dispatch(&problem, &mut policy, 500, 42, 50);
        assert!(res.completed_jobs > 0);
        assert!(res.mean_sojourn.is_finite());
        assert_eq!(res.per_machine_utilisation.len(), 2);
    }

    #[test]
    fn sect_beats_random_on_mean_wait() {
        let problem = small_problem();
        let eval_random = EvaluatePolicy {
            config: EvaluatePolicyConfig {
                policy_name: "random".to_string(),
                num_replications: 6,
                num_arrivals_per_rep: 800,
                seed_base: 1000,
                warmup: Some(100),
            },
        }
        .transform(EvaluatePolicyInput {
            problem: problem.clone(),
            factory: Box::new(|| Box::new(policy_random(7))),
        });
        let prob2 = problem.clone();
        let eval_sect = EvaluatePolicy {
            config: EvaluatePolicyConfig {
                policy_name: "sect".to_string(),
                num_replications: 6,
                num_arrivals_per_rep: 800,
                seed_base: 1000,
                warmup: Some(100),
            },
        }
        .transform(EvaluatePolicyInput {
            problem: prob2.clone(),
            factory: Box::new(move || Box::new(policy_sect(&prob2))),
        });
        assert!(eval_sect.mean_wait <= eval_random.mean_wait);
    }

    #[test]
    fn fluid_lp_solves_and_normalises() {
        let problem = small_problem();
        let res = policy_fluid_lp(&problem, 123);
        // Each class's assignment row should sum to ~1.
        for row in &res.x {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert!(res.bottleneck_load >= 0.0);
    }

    #[test]
    fn welch_t_sign_matches_mean_difference() {
        let t = welch_t(vec![1.0, 1.1, 0.9], vec![2.0, 2.1, 1.9]);
        assert!(t < 0.0);
    }

    #[test]
    fn mdp_vi_builds_policy() {
        // Keep the state space tiny: q_max=1 => (1+1)^2 * 2 = 8 states.
        let problem = small_problem();
        let res = policy_mdp_vi(
            &problem,
            MdpViPolicyOptions {
                q_max: Some(1),
                rollouts_per_sa: Some(8),
                max_iter: Some(200),
                seed: Some(3),
                ..Default::default()
            },
        );
        assert_eq!(res.num_states, 8);
        assert_eq!(res.q.len(), 8);
    }
}
