//! Port of `src/des/general/pomdp.ts` — POMDP solvers on top of the framework.
//!
//! We adopt the standard tuple ⟨S, A, Ω, T, O, R, γ⟩:
//!   * `S` = finite hidden states, `A` = actions, `Ω` = observations.
//!   * `T(s, a, s') = P(s' | s, a)`, `O(s', a, o) = P(o | s', a)`,
//!     `R(s, a)` = expected immediate reward, `γ` = discount.
//!
//! Three solvers (in order of cost):
//!   1. [`MostLikelyStateSolver`] — pretend the modal hidden state is the truth.
//!   2. [`QMDPSolver`] (Littman & Cassandra 1995) — solve the underlying MDP,
//!      then act greedily under the belief.
//!   3. [`pomdp_exact_finite_horizon`] — exact α-vector finite-horizon value
//!      iteration; tractable only for very small problems.
//!
//! Mapping notes (from the TS "RUST MIGRATION" header):
//!   * `POMDPSpec`'s `T`/`O`/`R` callbacks → boxed-closure fields returning
//!     `Vec<f64>` / `f64`.
//!   * Generic `<S, A, O>` carries over; states/actions/observations default to
//!     `usize` indices.
//!   * α-vectors and belief vectors are `Vec<f64>` / `Vec<Vec<f64>>`.
//!   * The combinatorial-blowup guard (`> 200000`) becomes a `panic!`.
//!   * Deterministic (planning only); no RNG/clock here except the optional
//!     ε-greedy `rng` callbacks that mirror the TS signatures.
//!
//! ASSUMED `DiscreteBelief` API (see `general/belief.rs`, ported in parallel):
//!   * `DiscreteBelief::new(states: Vec<S>, prior: Option<Vec<f64>>) -> Self`
//!     (uniform when `prior` is `None`; this mirrors the TS
//!     `new DiscreteBelief(states, prior?)` constructor).
//!   * public field `weights: Vec<f64>` (the per-state probabilities).
//!   * method `mode_index(&self) -> usize` (argmax of `weights`).
//!   If the real `DiscreteBelief` exposes different names, only the call sites
//!   in [`belief_update`], the `act` methods, and [`MostLikelyStateSolver`]
//!   need reconciling.

use std::collections::HashMap;

use crate::des::general::belief::DiscreteBelief;

/// POMDP specification ⟨S, A, Ω, T, O, R, γ⟩. `S`/`A`/`O` default to `usize`.
pub struct POMDPSpec<S = usize, A = usize, O = usize> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub observations: Vec<O>,
    /// `P(s' | s, a)` — a vector of length |S| parallel to `states`.
    pub transition: Box<dyn Fn(usize, usize) -> Vec<f64>>,
    /// `P(o | s', a)` — a vector of length |Ω| parallel to `observations`.
    pub observation: Box<dyn Fn(usize, usize) -> Vec<f64>>,
    /// Expected immediate reward `R(s, a)`.
    pub reward: Box<dyn Fn(usize, usize) -> f64>,
    pub discount: f64,
    /// Optional initial belief `b₀`; defaults to uniform.
    pub initial_belief: Option<Vec<f64>>,
    /// Optional terminal flag — `true` if the state is absorbing.
    pub is_terminal: Option<Box<dyn Fn(usize) -> bool>>,
}

// -----------------------------------------------------------------------------
// Belief update (Bayesian filter): b'(s') ∝ O(s', a, o) · Σ_s T(s, a, s') · b(s)
// -----------------------------------------------------------------------------
/// Bayesian belief update for taking action `a_idx` and observing `o_idx`.
/// Falls back to a uniform belief on an impossible observation.
pub fn belief_update<S: Clone, A, O>(
    spec: &POMDPSpec<S, A, O>,
    b: &DiscreteBelief<S>,
    a_idx: usize,
    o_idx: usize,
) -> DiscreteBelief<S> {
    let k = spec.states.len();
    // Predict: bp(s') = Σ_s T(s, a, s') · b(s).
    let mut bp = vec![0.0_f64; k];
    for i in 0..k {
        let t_row = (spec.transition)(i, a_idx);
        let w = b.weights[i];
        for j in 0..k {
            bp[j] += w * t_row[j];
        }
    }
    // Correct: weight by P(o | s', a).
    let mut total = 0.0;
    for j in 0..k {
        let o_row = (spec.observation)(j, a_idx);
        bp[j] *= o_row[o_idx];
        total += bp[j];
    }
    if !total.is_finite() || total <= 0.0 {
        // Impossible observation under the model. Fall back to uniform.
        return DiscreteBelief::new(spec.states.clone(), None);
    }
    for j in 0..k {
        bp[j] /= total;
    }
    DiscreteBelief::new(spec.states.clone(), Some(&bp))
}

// -----------------------------------------------------------------------------
// MDP value iteration (used by QMDP).
// -----------------------------------------------------------------------------
/// Options for [`mdp_value_iteration`]. Defaults: `tol = 1e-8`, `max_iter = 5000`.
#[derive(Clone, Debug)]
pub struct MDPVIOptions {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for MDPVIOptions {
    fn default() -> Self {
        MDPVIOptions {
            tol: 1e-8,
            max_iter: 5000,
        }
    }
}

/// Result of [`mdp_value_iteration`].
#[derive(Clone, Debug)]
pub struct MDPVIResult {
    pub v: Vec<f64>,
    /// `Q[s][a]`.
    pub q: Vec<Vec<f64>>,
    pub iterations: usize,
    pub final_delta: f64,
    /// Greedy action at each state.
    pub policy: Vec<usize>,
}

/// Value iteration on the underlying MDP (treating `S` as fully observable).
pub fn mdp_value_iteration<S, A, O>(spec: &POMDPSpec<S, A, O>, opts: &MDPVIOptions) -> MDPVIResult {
    let tol = opts.tol;
    let max_iter = opts.max_iter;
    let k = spec.states.len();
    let num_a = spec.actions.len();
    let gamma = spec.discount;
    let mut v = vec![0.0_f64; k];
    let mut q: Vec<Vec<f64>> = vec![vec![0.0_f64; num_a]; k];
    let mut iter = 0;
    let mut delta = f64::INFINITY;
    while iter < max_iter && delta > tol {
        delta = 0.0;
        let mut v_new = vec![0.0_f64; k];
        for s in 0..k {
            if let Some(is_term) = &spec.is_terminal {
                if is_term(s) {
                    v_new[s] = 0.0;
                    for a in 0..num_a {
                        q[s][a] = 0.0;
                    }
                    continue;
                }
            }
            let mut best = f64::NEG_INFINITY;
            for a in 0..num_a {
                let mut qv = (spec.reward)(s, a);
                let t_row = (spec.transition)(s, a);
                for sp in 0..k {
                    qv += gamma * t_row[sp] * v[sp];
                }
                q[s][a] = qv;
                if qv > best {
                    best = qv;
                }
            }
            v_new[s] = best;
            let d = (v_new[s] - v[s]).abs();
            if d > delta {
                delta = d;
            }
        }
        v = v_new;
        iter += 1;
    }
    let mut policy = vec![0_usize; k];
    for s in 0..k {
        let mut bi = 0;
        let mut bv = f64::NEG_INFINITY;
        for a in 0..num_a {
            if q[s][a] > bv {
                bv = q[s][a];
                bi = a;
            }
        }
        policy[s] = bi;
    }
    MDPVIResult {
        v,
        q,
        iterations: iter,
        final_delta: delta,
        policy,
    }
}

// -----------------------------------------------------------------------------
// QMDP heuristic (Littman & Cassandra 1995).
// -----------------------------------------------------------------------------
/// QMDP solver: solves the underlying MDP, then acts greedily under the belief.
pub struct QMDPSolver<S, A, O> {
    pub spec: POMDPSpec<S, A, O>,
    pub q: Vec<Vec<f64>>,
}

impl<S, A, O> QMDPSolver<S, A, O> {
    pub fn new(spec: POMDPSpec<S, A, O>, opts: &MDPVIOptions) -> Self {
        let r = mdp_value_iteration(&spec, opts);
        QMDPSolver { spec, q: r.q }
    }

    /// `a* = argmax_a Σ_s b(s) Q(s, a)` with optional ε-greedy exploration.
    pub fn act(&self, b: &DiscreteBelief<S>, rng: Option<&dyn Fn() -> f64>, epsilon: f64) -> usize {
        if let Some(rng) = rng {
            if epsilon > 0.0 && rng() < epsilon {
                return (rng() * self.spec.actions.len() as f64).floor() as usize;
            }
        }
        let num_a = self.spec.actions.len();
        let mut bi = 0;
        let mut bv = f64::NEG_INFINITY;
        for a in 0..num_a {
            let mut q = 0.0;
            for s in 0..self.q.len() {
                q += b.weights[s] * self.q[s][a];
            }
            if q > bv {
                bv = q;
                bi = a;
            }
        }
        bi
    }

    /// Expected QMDP value `E_b[Q(s, a)]` — useful for ranking actions.
    pub fn q_belief(&self, b: &DiscreteBelief<S>, a_idx: usize) -> f64 {
        let mut q = 0.0;
        for s in 0..self.q.len() {
            q += b.weights[s] * self.q[s][a_idx];
        }
        q
    }
}

// -----------------------------------------------------------------------------
// Generic finite-horizon belief lookahead.
//
//   Q_d(b, a) = R(b, a) + γ · Σ_o P(o | b, a) V_{d-1}(τ(b, a, o))
//
// The leaf value can be zero or the QMDP value function.
// -----------------------------------------------------------------------------

/// Leaf-evaluation strategy for [`BeliefLookaheadSolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeliefLookaheadLeaf {
    Zero,
    Qmdp,
}

/// Options for [`BeliefLookaheadSolver`]. Defaults mirror the TS source.
#[derive(Clone, Debug)]
pub struct BeliefLookaheadOptions {
    pub horizon: usize,
    pub leaf: BeliefLookaheadLeaf,
    pub memoize: bool,
    pub belief_precision: f64,
    pub max_nodes: usize,
}

impl Default for BeliefLookaheadOptions {
    fn default() -> Self {
        BeliefLookaheadOptions {
            horizon: 2,
            leaf: BeliefLookaheadLeaf::Qmdp,
            memoize: true,
            belief_precision: 1e-6,
            max_nodes: 250_000,
        }
    }
}

/// A `(action, q-value)` pair returned by [`BeliefLookaheadSolver::action_values`].
#[derive(Clone, Debug)]
pub struct BeliefActionValue {
    pub action: usize,
    pub q: f64,
}

/// `R(b, a) = Σ_s b(s) · R(s, a)`.
pub fn expected_belief_reward<S, A, O>(
    spec: &POMDPSpec<S, A, O>,
    b: &DiscreteBelief<S>,
    a_idx: usize,
) -> f64 {
    let mut r = 0.0;
    for s in 0..spec.states.len() {
        r += b.weights[s] * (spec.reward)(s, a_idx);
    }
    r
}

/// `P(o | b, a) = Σ_{s, s'} b(s) · T(s, a, s') · O(s', a, o)`.
pub fn observation_distribution<S, A, O>(
    spec: &POMDPSpec<S, A, O>,
    b: &DiscreteBelief<S>,
    a_idx: usize,
) -> Vec<f64> {
    let num_o = spec.observations.len();
    let mut out = vec![0.0_f64; num_o];
    for s in 0..spec.states.len() {
        let t_row = (spec.transition)(s, a_idx);
        for sp in 0..spec.states.len() {
            let p_next = b.weights[s] * t_row[sp];
            if p_next == 0.0 {
                continue;
            }
            let o_row = (spec.observation)(sp, a_idx);
            for o in 0..num_o {
                out[o] += p_next * o_row[o];
            }
        }
    }
    out
}

/// Finite-horizon belief-tree lookahead, bottoming out at a configurable leaf
/// value (zero or QMDP).
pub struct BeliefLookaheadSolver<S, A, O> {
    horizon: usize,
    leaf: BeliefLookaheadLeaf,
    memoize: bool,
    precision: f64,
    max_nodes: usize,
    /// Owns the [`POMDPSpec`] (accessed via `self.qmdp.spec`) and the QMDP
    /// Q-table used for leaf evaluation.
    qmdp: QMDPSolver<S, A, O>,
    cache: HashMap<String, f64>,
    nodes_visited: usize,
}

impl<S: Clone, A, O> BeliefLookaheadSolver<S, A, O> {
    pub fn new(spec: POMDPSpec<S, A, O>, opts: BeliefLookaheadOptions) -> Self {
        // The TS `Number.isInteger(horizon) && horizon >= 0` check is vacuous
        // for `usize`; only the precision guard can fire.
        if opts.belief_precision <= 0.0 || !opts.belief_precision.is_finite() {
            panic!(
                "BeliefLookaheadSolver: beliefPrecision must be positive; got {}",
                opts.belief_precision
            );
        }
        let qmdp = QMDPSolver::new(spec, &MDPVIOptions::default());
        BeliefLookaheadSolver {
            horizon: opts.horizon,
            leaf: opts.leaf,
            memoize: opts.memoize,
            precision: opts.belief_precision,
            max_nodes: opts.max_nodes,
            qmdp,
            cache: HashMap::new(),
            nodes_visited: 0,
        }
    }

    /// Greedy belief action (with optional ε-greedy exploration), at the
    /// solver's configured `horizon`.
    pub fn act(&mut self, b: &DiscreteBelief<S>, rng: Option<&dyn Fn() -> f64>, epsilon: f64) -> usize {
        if let Some(rng) = rng {
            if epsilon > 0.0 && rng() < epsilon {
                return (rng() * self.qmdp.spec.actions.len() as f64).floor() as usize;
            }
        }
        let horizon = self.horizon;
        let values = self.action_values(b, horizon);
        let mut best = values[0].clone();
        for v in &values[1..] {
            if v.q > best.q {
                best = v.clone();
            }
        }
        best.action
    }

    /// Per-action Q-values at `depth`, sorted descending by `q` then ascending
    /// by action index. Pass `self.horizon()` for the default depth.
    pub fn action_values(&mut self, b: &DiscreteBelief<S>, depth: usize) -> Vec<BeliefActionValue> {
        self.nodes_visited = 0;
        self.action_values_inner(b, depth)
    }

    /// Belief value at `depth` (max over actions).
    pub fn value(&mut self, b: &DiscreteBelief<S>, depth: usize) -> f64 {
        self.nodes_visited = 0;
        self.value_inner(b, depth)
    }

    /// The configured planning horizon (default depth for [`Self::act`]).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    fn action_values_inner(&mut self, b: &DiscreteBelief<S>, depth: usize) -> Vec<BeliefActionValue> {
        let num_a = self.qmdp.spec.actions.len();
        let discount = self.qmdp.spec.discount;
        let mut out: Vec<BeliefActionValue> = Vec::with_capacity(num_a);
        for a in 0..num_a {
            let mut q = expected_belief_reward(&self.qmdp.spec, b, a);
            if depth > 0 {
                let obs = observation_distribution(&self.qmdp.spec, b, a);
                let mut future = 0.0;
                for o in 0..obs.len() {
                    if obs[o] <= 0.0 {
                        continue;
                    }
                    let bp = belief_update(&self.qmdp.spec, b, a, o);
                    future += obs[o] * self.value_inner(&bp, depth - 1);
                }
                q += discount * future;
            }
            out.push(BeliefActionValue { action: a, q });
        }
        out.sort_by(|x, y| {
            y.q
                .partial_cmp(&x.q)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.action.cmp(&y.action))
        });
        out
    }

    fn value_inner(&mut self, b: &DiscreteBelief<S>, depth: usize) -> f64 {
        self.nodes_visited += 1;
        if self.nodes_visited > self.max_nodes {
            panic!(
                "BeliefLookaheadSolver: exceeded maxNodes={}; reduce horizon or increase maxNodes",
                self.max_nodes
            );
        }
        if depth == 0 {
            return self.leaf_value(b);
        }
        let key = if self.memoize {
            self.cache_key(b, depth)
        } else {
            String::new()
        };
        if !key.is_empty() {
            if let Some(cached) = self.cache.get(&key) {
                return *cached;
            }
        }
        let best = self
            .action_values_inner(b, depth)
            .first()
            .map(|x| x.q)
            .unwrap_or(0.0);
        if !key.is_empty() {
            self.cache.insert(key, best);
        }
        best
    }

    fn leaf_value(&self, b: &DiscreteBelief<S>) -> f64 {
        if self.leaf == BeliefLookaheadLeaf::Zero {
            return 0.0;
        }
        let mut best = f64::NEG_INFINITY;
        for a in 0..self.qmdp.spec.actions.len() {
            best = best.max(self.qmdp.q_belief(b, a));
        }
        best
    }

    fn cache_key(&self, b: &DiscreteBelief<S>, depth: usize) -> String {
        let quantized: Vec<String> = b
            .weights
            .iter()
            .map(|w| ((w / self.precision).round() as i64).to_string())
            .collect();
        format!("{}|{}", depth, quantized.join(","))
    }
}

// -----------------------------------------------------------------------------
// Most-likely-state heuristic: act as if the modal hidden state is the truth.
// -----------------------------------------------------------------------------
/// Most-likely-state solver: greedy on the underlying MDP policy at the modal
/// hidden state.
pub struct MostLikelyStateSolver<S, A, O> {
    pub spec: POMDPSpec<S, A, O>,
    pub mdp_result: MDPVIResult,
}

impl<S, A, O> MostLikelyStateSolver<S, A, O> {
    /// Build, solving the underlying MDP with default options (mirrors the TS
    /// `mdpResult = mdpValueIteration(spec)` default argument).
    pub fn new(spec: POMDPSpec<S, A, O>) -> Self {
        let mdp_result = mdp_value_iteration(&spec, &MDPVIOptions::default());
        MostLikelyStateSolver { spec, mdp_result }
    }

    /// Build with a precomputed MDP result.
    pub fn with_result(spec: POMDPSpec<S, A, O>, mdp_result: MDPVIResult) -> Self {
        MostLikelyStateSolver { spec, mdp_result }
    }

    pub fn act(&self, b: &DiscreteBelief<S>) -> usize {
        self.mdp_result.policy[b.mode_index()]
    }
}

// -----------------------------------------------------------------------------
// Finite-horizon point-based value iteration, EXACT for small POMDPs.
//
// Represent V_t as a set of α-vectors {α_i}, V_t(b) = max_i ⟨α_i, b⟩.
// -----------------------------------------------------------------------------

/// One α-vector tagged with the action that generated it.
#[derive(Clone, Debug)]
pub struct AlphaVector {
    pub vec: Vec<f64>,
    pub action: usize,
}

/// Result of [`pomdp_exact_finite_horizon`]: the α-vector set plus value/act
/// queries.
pub struct POMDPExactResult {
    pub alpha_vectors: Vec<AlphaVector>,
    k: usize,
}

impl POMDPExactResult {
    /// `V(b) = max_i ⟨α_i, b⟩`.
    pub fn value(&self, b: &[f64]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for av in &self.alpha_vectors {
            let mut v = 0.0;
            for i in 0..self.k {
                v += av.vec[i] * b[i];
            }
            if v > best {
                best = v;
            }
        }
        best
    }

    /// `argmax` action of the dominating α-vector at belief `b`.
    pub fn act<S>(&self, b: &DiscreteBelief<S>) -> usize {
        let mut best = f64::NEG_INFINITY;
        let mut best_a = 0;
        for av in &self.alpha_vectors {
            let mut v = 0.0;
            for i in 0..self.k {
                v += av.vec[i] * b.weights[i];
            }
            if v > best {
                best = v;
                best_a = av.action;
            }
        }
        best_a
    }
}

/// Exact finite-horizon α-vector value iteration (Sondik backup). Panics on
/// combinatorial blowup (`|alphas|^|Ω| > 200000`).
pub fn pomdp_exact_finite_horizon<S, A, O>(
    spec: &POMDPSpec<S, A, O>,
    horizon: usize,
) -> POMDPExactResult {
    let k = spec.states.len();
    let na = spec.actions.len();
    let no = spec.observations.len();
    // Initialise V_0 to immediate reward (each action gives one α-vector).
    let mut alphas: Vec<AlphaVector> = Vec::with_capacity(na);
    for a in 0..na {
        let mut v = vec![0.0_f64; k];
        for s in 0..k {
            v[s] = (spec.reward)(s, a);
        }
        alphas.push(AlphaVector { vec: v, action: a });
    }
    alphas = prune_alphas(alphas, k);

    for _t in 1..horizon {
        let mut next: Vec<AlphaVector> = Vec::new();
        for a in 0..na {
            // For each observation, the future α-vectors backprop through O × T.
            // Index over all combinations of |alphas|^|Ω|.
            let total_f = (alphas.len() as f64).powi(no as i32);
            if total_f > 200_000.0 {
                panic!(
                    "pomdpExactFiniteHorizon: combinatorial blowup (|alphas|={}, |Ω|={}, total={}). \
                     Reduce horizon or use QMDP.",
                    alphas.len(),
                    no,
                    total_f
                );
            }
            let total = total_f as u64;
            let mut idxs = vec![0_usize; no];
            for _combo in 0..total {
                let mut v = vec![0.0_f64; k];
                for s in 0..k {
                    let mut val = (spec.reward)(s, a);
                    let t_row = (spec.transition)(s, a);
                    for sp in 0..k {
                        let o_row = (spec.observation)(sp, a);
                        let mut inner = 0.0;
                        for o in 0..no {
                            inner += o_row[o] * alphas[idxs[o]].vec[sp];
                        }
                        val += spec.discount * t_row[sp] * inner;
                    }
                    v[s] = val;
                }
                next.push(AlphaVector { vec: v, action: a });
                // Increment idxs in base alphas.len().
                for kdig in (0..no).rev() {
                    idxs[kdig] += 1;
                    if idxs[kdig] < alphas.len() {
                        break;
                    }
                    idxs[kdig] = 0;
                }
            }
        }
        alphas = prune_alphas(next, k);
    }

    POMDPExactResult {
        alpha_vectors: alphas,
        k,
    }
}

/// Cheap α-vector pruning: drop pointwise-dominated and near-duplicate vectors.
fn prune_alphas(alphas: Vec<AlphaVector>, k: usize) -> Vec<AlphaVector> {
    let mut keep: Vec<AlphaVector> = Vec::new();
    for i in 0..alphas.len() {
        let mut dominated = false;
        for j in 0..alphas.len() {
            if i == j {
                continue;
            }
            let mut all_leq = true;
            let mut strict = false;
            for kk in 0..k {
                if alphas[j].vec[kk] < alphas[i].vec[kk] - 1e-12 {
                    all_leq = false;
                    break;
                }
                if alphas[j].vec[kk] > alphas[i].vec[kk] + 1e-12 {
                    strict = true;
                }
            }
            if all_leq && strict {
                dominated = true;
                break;
            }
        }
        if !dominated {
            // Also dedupe near-identical vectors.
            let mut dup = false;
            for kept in &keep {
                let mut same = true;
                for q in 0..k {
                    if (kept.vec[q] - alphas[i].vec[q]).abs() > 1e-9 {
                        same = false;
                        break;
                    }
                }
                if same {
                    dup = true;
                    break;
                }
            }
            if !dup {
                keep.push(alphas[i].clone());
            }
        }
    }
    if !keep.is_empty() {
        keep
    } else {
        vec![alphas[0].clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two self-loop states, one action, rewards [1, 2], γ = 0.9.
    /// V(s) = r(s) / (1 − γ) → [10, 20].
    #[test]
    fn mdp_value_iteration_self_loops() {
        let spec: POMDPSpec = POMDPSpec {
            states: vec![0, 1],
            actions: vec![0],
            observations: vec![0],
            transition: Box::new(|s, _a| if s == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }),
            observation: Box::new(|_s, _a| vec![1.0]),
            reward: Box::new(|s, _a| if s == 0 { 1.0 } else { 2.0 }),
            discount: 0.9,
            initial_belief: None,
            is_terminal: None,
        };
        let res = mdp_value_iteration(&spec, &MDPVIOptions::default());
        assert!((res.v[0] - 10.0).abs() < 1e-4, "V[0]={}", res.v[0]);
        assert!((res.v[1] - 20.0).abs() < 1e-4, "V[1]={}", res.v[1]);
    }

    /// 2 states, 2 actions, R(s, a) = 1 iff a == s, γ = 0 → Q = identity.
    /// QMDP under a belief concentrated on state 1 should pick action 1.
    #[test]
    fn qmdp_acts_under_belief() {
        let spec: POMDPSpec = POMDPSpec {
            states: vec![0, 1],
            actions: vec![0, 1],
            observations: vec![0],
            transition: Box::new(|s, _a| if s == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }),
            observation: Box::new(|_s, _a| vec![1.0]),
            reward: Box::new(|s, a| if a == s { 1.0 } else { 0.0 }),
            discount: 0.0,
            initial_belief: None,
            is_terminal: None,
        };
        let solver = QMDPSolver::new(spec, &MDPVIOptions::default());
        let b = DiscreteBelief::new(vec![0_usize, 1], Some(&[0.0, 1.0]));
        assert_eq!(solver.act(&b, None, 0.0), 1);
        assert!((solver.q_belief(&b, 1) - 1.0).abs() < 1e-9);
    }

    /// Exact horizon-1 α-vectors equal the immediate reward vectors. With
    /// R(s, a) = 1 iff a == s the value of a corner belief is 1.
    #[test]
    fn exact_finite_horizon_corner_value() {
        let spec: POMDPSpec = POMDPSpec {
            states: vec![0, 1],
            actions: vec![0, 1],
            observations: vec![0, 1],
            transition: Box::new(|s, _a| if s == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }),
            observation: Box::new(|sp, _a| if sp == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }),
            reward: Box::new(|s, a| if a == s { 1.0 } else { 0.0 }),
            discount: 0.9,
            initial_belief: None,
            is_terminal: None,
        };
        let res = pomdp_exact_finite_horizon(&spec, 1);
        assert!((res.value(&[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((res.value(&[0.5, 0.5]) - 0.5).abs() < 1e-9);
        let b = DiscreteBelief::new(vec![0_usize, 1], Some(&[0.0, 1.0]));
        assert_eq!(res.act(&b), 1);
    }
}
