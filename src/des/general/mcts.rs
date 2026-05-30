//! Port of `src/des/general/mcts.ts` — generic Monte Carlo Tree Search (UCT).
//!
//! A pluggable-rollout UCT planner that plugs directly into the DES framework:
//! any state that can be cloned and advanced one decision-epoch via
//! `apply_action(state, a)` is a valid [`MCTSEnv`]. Orchestrated as a
//! [`TreeSearchStation`] leaf — each `run_time_step` performs one UCT iteration
//! (select -> expand -> simulate -> backup). The frontier is implicit (the
//! in-memory tree), so the base's `push_children` / `expand` hooks are no-ops;
//! the new child is created inline during selection.
//!
//! Rust shape (faithful to the TS class):
//!   * `interface MCTSEnv<S>`  -> trait [`MCTSEnv`] (callbacks become methods).
//!   * `interface MCTSOptions` -> struct [`MCTSOptions`] (`Default`-derivable).
//!   * `interface Node<S>`     -> struct [`Node`]; the parent/child graph is an
//!     ARENA (`Vec<Node<S>>` + `usize` indices) rather than `Rc<RefCell>`.
//!   * `class MCTSStation<S> extends TreeSearchStation<Node<S>>` -> struct
//!     [`MCTSStation`] embedding [`StationCore`] + [`TreeSearchCore`]`<usize>`
//!     and implementing the tree-search trait over arena indices.
//!   * `fn mcts<S>` -> the free fn [`mcts`].
//!
//! Conversion notes:
//!   * INJECTED RNG: the TS `rng?: () => number` defaulting to `Math.random`
//!     becomes an injected `RandomSource` passed to the station / [`mcts`]; the
//!     UCT tie-break and rollout draws use it (never a global).
//!   * UCT child selection and the final root-action choice both reduce to
//!     [`arg_max_with_tie_break`] over a score vector (reservoir tie-break with
//!     `ARGMAX_EPS_DEFAULT`), which reproduces the TS inline tie-breaking
//!     exactly (same RNG-draw schedule).
//!   * `children: Map<number, Node<S>>` -> an insertion-ordered
//!     `Vec<(action, child_index)>` so iteration order (which the tie-break
//!     depends on) matches the TS `Map` insertion order deterministically.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::tree_search::{
    NodeEvaluation, SearchObjective, TreeSearchCore, TreeSearchStation,
};
use crate::des::shared::capabilities::RandomSource;

/// Result of applying an action in a state: the next state, the immediate
/// reward on that transition, and whether the transition is terminal.
/// (TS `{next: S; reward: number; done: boolean}`.)
pub struct ApplyResult<S> {
    pub next: S,
    pub reward: f64,
    pub done: bool,
}

/// The search environment (TS `interface MCTSEnv<S>`). The state object handed
/// to `apply_action` MUST be treated as immutable / freshly cloned so siblings
/// in the tree do not share aliased mutable state.
pub trait MCTSEnv<S> {
    /// Number of legal actions in state `s` (constant action sets are simplest).
    fn num_actions(&self, s: &S) -> usize;

    /// Apply action `a` in state `s`; returns the next state, the immediate
    /// reward, and whether the episode ended on that transition.
    fn apply_action(&self, s: &S, a: usize) -> ApplyResult<S>;

    /// Terminal predicate. Default: never terminal (TS `isTerminal?`).
    fn is_terminal(&self, _s: &S) -> bool {
        false
    }

    /// Default rollout policy. If not overridden this is uniform-random over
    /// the legal actions (TS `rolloutPolicy ?? uniform`). For DES-driven
    /// rollouts a fast heuristic (shortest-queue, …) is the natural override.
    fn rollout_policy(&self, s: &S, rng: &mut dyn RandomSource) -> usize {
        let n = self.num_actions(s);
        (rng.next_float() * n as f64).floor() as usize
    }

    /// Decision epochs in a leaf rollout before cut-off (TS `rolloutDepth ?? 50`).
    fn rollout_depth(&self) -> usize {
        50
    }

    /// Discount factor applied along the rollout (TS `gamma ?? 1.0`).
    fn gamma(&self) -> f64 {
        1.0
    }
}

/// Final action-selection rule (TS `selection: 'visits' | 'value'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Most-visited child (robust; the TS default).
    Visits,
    /// Highest mean-value child (greedy).
    Value,
}

/// Tunables for an [`mcts`] / [`MCTSStation`] run (TS `interface MCTSOptions`,
/// minus the `rng` field which is injected separately as a [`RandomSource`]).
#[derive(Clone, Copy, Debug)]
pub struct MCTSOptions {
    /// UCT iterations per decision call (TS default 200).
    pub iterations: usize,
    /// Exploration constant `Cp` in the UCT formula (TS default sqrt(2)).
    pub c: f64,
    /// Final action-selection rule (TS default `'visits'`).
    pub selection: Selection,
}

impl Default for MCTSOptions {
    fn default() -> Self {
        MCTSOptions {
            iterations: 200,
            c: std::f64::consts::SQRT_2,
            selection: Selection::Visits,
        }
    }
}

/// An in-memory tree node (TS `interface Node<S>`). Lives in the station's
/// arena; parents / children are referenced by `usize` arena indices.
pub struct Node<S> {
    pub state: S,
    /// Arena index of the parent, or `None` at the root.
    pub parent: Option<usize>,
    /// Action taken in `parent.state` to reach this node (`-1` at the root).
    pub from_action: i64,
    /// Reward received on the parent -> this transition.
    pub reward_in: f64,
    /// Visit count (a float, mirroring the TS `number`).
    pub visits: f64,
    /// Sum of returns observed below this node (averaged via `/ visits`).
    pub total_return: f64,
    /// `(action, child arena index)` pairs in insertion order (mirrors the TS
    /// `Map<number, Node>` iteration order, which the UCT tie-break relies on).
    pub children: Vec<(usize, usize)>,
    /// Untried action indices; once empty every action has a child.
    pub untried: Vec<usize>,
    pub done: bool,
}

impl<S> Node<S> {
    /// Construct a fresh node (TS `makeNode`).
    pub fn new(
        state: S,
        parent: Option<usize>,
        from_action: i64,
        reward_in: f64,
        num_actions: usize,
        done: bool,
    ) -> Self {
        Node {
            state,
            parent,
            from_action,
            reward_in,
            visits: 0.0,
            total_return: 0.0,
            children: Vec::new(),
            untried: (0..num_actions).collect(),
            done,
        }
    }
}

/// Concrete `TreeSearchStation<usize>` leaf running UCT over an arena.
///
/// `pick_next` walks down from the root via UCT and expands one untried action;
/// `evaluate` runs the rollout from that leaf and backs the discounted return
/// up the stashed path. `expand` / `push_children` are no-ops (the frontier is
/// the in-memory tree itself).
pub struct MCTSStation<S, R: RandomSource> {
    core: StationCore,
    search: TreeSearchCore<usize>,
    /// Node arena; index `0` is always the root.
    arena: Vec<Node<S>>,
    root: usize,
    env: Box<dyn MCTSEnv<S>>,
    max_iters: usize,
    c: f64,
    rng: R,
    rollout_depth: usize,
    gamma: f64,
    /// Last rollout's return (kept because the base separates `evaluate`).
    last_g: f64,
    /// Root -> leaf path for the current iteration (arena indices) for backup.
    last_path: Vec<usize>,
}

impl<S, R: RandomSource> MCTSStation<S, R> {
    /// Build a station rooted at `root_state` (TS `MCTSStation` constructor).
    pub fn new(env: Box<dyn MCTSEnv<S>>, root_state: S, opts: MCTSOptions, rng: R) -> Self {
        let gamma = env.gamma();
        let rollout_depth = env.rollout_depth();
        let root_actions = env.num_actions(&root_state);
        let root_done = env.is_terminal(&root_state);
        let root_node = Node::new(root_state, None, -1, 0.0, root_actions, root_done);
        MCTSStation {
            core: StationCore::new("mcts"),
            search: TreeSearchCore::new(SearchObjective::Maximise, opts.iterations as f64),
            arena: vec![root_node],
            root: 0,
            env,
            max_iters: opts.iterations,
            c: opts.c,
            rng,
            rollout_depth,
            gamma,
            last_g: 0.0,
            last_path: Vec::new(),
        }
    }

    /// The most recent rollout return (exposed for inspection).
    pub fn last_return(&self) -> f64 {
        self.last_g
    }

    /// Visit count of each root child, keyed by action (TS `rootChildVisits`).
    pub fn root_child_visits(&self) -> HashMap<usize, f64> {
        let mut m = HashMap::new();
        for &(a, ci) in &self.arena[self.root].children {
            m.insert(a, self.arena[ci].visits);
        }
        m
    }

    /// Mean value of each root child, keyed by action (TS `rootChildValues`).
    pub fn root_child_values(&self) -> HashMap<usize, f64> {
        let mut m = HashMap::new();
        for &(a, ci) in &self.arena[self.root].children {
            let child = &self.arena[ci];
            let v = if child.visits > 0.0 {
                child.total_return / child.visits
            } else {
                0.0
            };
            m.insert(a, v);
        }
        m
    }
}

impl<S: Clone + 'static, R: RandomSource + 'static> DESStation for MCTSStation<S, R> {
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
        self.run_tree_search_step();
    }
    /// The search has work while it has not finished (the in-memory tree, not an
    /// inbox, drives progress).
    fn has_work(&self) -> bool {
        !self.search.finished
    }
}

impl<S: Clone + 'static, R: RandomSource + 'static> TreeSearchStation<usize> for MCTSStation<S, R> {
    fn search_core(&self) -> &TreeSearchCore<usize> {
        &self.search
    }
    fn search_core_mut(&mut self) -> &mut TreeSearchCore<usize> {
        &mut self.search
    }

    /// Walk the tree via UCT down to a leaf, expand one untried action if
    /// available, stash the root->leaf path, and return the chosen leaf.
    fn pick_next(&mut self) -> Option<usize> {
        if self.search.nodes_processed >= self.max_iters {
            return None;
        }
        let mut node = self.root;
        let mut path = vec![node];

        // ── Selection: descend while fully expanded, non-terminal, with children.
        loop {
            let descend = {
                let n = &self.arena[node];
                n.untried.is_empty() && !n.children.is_empty() && !n.done
            };
            if !descend {
                break;
            }
            let (parent_visits, child_indices): (f64, Vec<usize>) = {
                let n = &self.arena[node];
                (n.visits, n.children.iter().map(|&(_, ci)| ci).collect())
            };
            // UCT score per child; uniform random tie-break is critical at fresh
            // children where the means are equal (deterministic argmax would
            // always descend the lowest action id, biasing the tree's shape).
            let ucts: Vec<f64> = child_indices
                .iter()
                .map(|&ci| {
                    let (cv, ctr) = {
                        let c = &self.arena[ci];
                        (c.visits, c.total_return)
                    };
                    let mean = if cv > 0.0 { ctr / cv } else { 0.0 };
                    mean + self.c * ((parent_visits + 1.0).ln() / (cv + 1e-12)).sqrt()
                })
                .collect();
            let best = arg_max_with_tie_break(&ucts, &mut self.rng, ARGMAX_EPS_DEFAULT)
                .expect("non-empty children");
            node = child_indices[best];
            path.push(node);
        }

        // ── Expansion: instantiate one untried action of the leaf.
        let (untried_empty, leaf_done) = {
            let n = &self.arena[node];
            (n.untried.is_empty(), n.done)
        };
        if !untried_empty && !leaf_done {
            let untried_len = self.arena[node].untried.len();
            let pick = (self.rng.next_float() * untried_len as f64).floor() as usize;
            let action = self.arena[node].untried.remove(pick);
            let res = {
                let s = &self.arena[node].state;
                self.env.apply_action(s, action)
            };
            let child_actions = self.env.num_actions(&res.next);
            let child_done = res.done || self.env.is_terminal(&res.next);
            let child = Node::new(
                res.next,
                Some(node),
                action as i64,
                res.reward,
                child_actions,
                child_done,
            );
            let child_idx = self.arena.len();
            self.arena.push(child);
            self.arena[node].children.push((action, child_idx));
            node = child_idx;
            path.push(node);
        }

        self.last_path = path;
        Some(node)
    }

    /// Roll out from the leaf, accumulate the discounted return into `last_g`,
    /// and back it up the stashed path. Always reports `is_leaf = true` so the
    /// runner never tries to expand (expansion was inlined in `pick_next`).
    fn evaluate(&mut self, node: &usize) -> NodeEvaluation {
        let node = *node;
        let leaf_done = self.arena[node].done;
        let mut g = self.arena[node].reward_in;
        let mut discount = self.gamma;
        if !leaf_done {
            let mut s = self.arena[node].state.clone();
            for _ in 0..self.rollout_depth {
                if self.env.is_terminal(&s) {
                    break;
                }
                let a = self.env.rollout_policy(&s, &mut self.rng);
                let r = self.env.apply_action(&s, a);
                g += discount * r.reward;
                discount *= self.gamma;
                s = r.next;
                if r.done {
                    break;
                }
            }
        }
        self.last_g = g;

        // Backup BEFORE the next pick_next walks the tree again. Each ancestor's
        // stored return is in the leaf's frame, discounted by depth to the leaf.
        let mut g_acc = g;
        for i in (0..self.last_path.len()).rev() {
            let cur = self.last_path[i];
            self.arena[cur].visits += 1.0;
            self.arena[cur].total_return += g_acc;
            g_acc = self.arena[cur].reward_in + self.gamma * g_acc;
        }

        NodeEvaluation { bound: g, is_leaf: true, value: Some(g), is_feasible: false }
    }

    /// Never reached: every `evaluate` returns `is_leaf = true`.
    fn expand(&mut self, _node: &usize, _ev: &NodeEvaluation) -> Vec<usize> {
        Vec::new()
    }

    /// Frontier is the in-memory tree itself — nothing to push.
    fn push_children(&mut self, _children: Vec<usize>) {}
}

/// Returned by [`mcts`]: the recommended root action plus the per-action visit
/// counts and mean values (TS `{action, visits, values}`).
#[derive(Clone, Debug)]
pub struct MctsResult {
    pub action: usize,
    pub visits: HashMap<usize, f64>,
    pub values: HashMap<usize, f64>,
}

/// Run UCT for `opts.iterations` steps from `root_state` and return the action
/// recommended at the root (TS `mcts`).
///
/// The action is the most-visited child (robust; default) or the highest-value
/// child (greedy), with uniform random tie-breaking via the injected RNG — with
/// low iteration budgets several children often tie on visits, and a
/// deterministic argmax would collapse to action 0.
pub fn mcts<S, R>(env: Box<dyn MCTSEnv<S>>, root_state: S, opts: MCTSOptions, rng: R) -> MctsResult
where
    S: Clone + 'static,
    R: RandomSource + 'static,
{
    let sel = opts.selection;
    let station = Rc::new(RefCell::new(MCTSStation::new(env, root_state, opts, rng)));
    run_iterative_des(vec![station.clone() as StationRef], IterativeRunOptions::default());

    let mut st = station.borrow_mut();
    let visits = st.root_child_visits();
    let values = st.root_child_values();

    // Sorted action keys (TS sorts the child keys numerically before scoring).
    let mut child_keys: Vec<usize> = st.arena[st.root].children.iter().map(|&(a, _)| a).collect();
    child_keys.sort_unstable();
    if child_keys.is_empty() {
        return MctsResult { action: 0, visits, values };
    }

    let scores: Vec<f64> = child_keys
        .iter()
        .map(|&a| {
            let ci = st.arena[st.root]
                .children
                .iter()
                .find(|&&(act, _)| act == a)
                .map(|&(_, ci)| ci)
                .expect("action present among root children");
            let child = &st.arena[ci];
            match sel {
                Selection::Visits => child.visits,
                Selection::Value => {
                    if child.visits > 0.0 {
                        child.total_return / child.visits
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            }
        })
        .collect();

    let best = arg_max_with_tie_break(&scores, &mut st.rng, ARGMAX_EPS_DEFAULT)
        .expect("non-empty child keys");
    let action = child_keys[best];

    MctsResult { action, visits, values }
}

#[cfg(test)]
mod tests {
    //! Tests for the UCT planner. A small deterministic chain environment where
    //! action 1 pays a positive reward and action 0 a negative one, so the
    //! planner should strongly prefer action 1. Fixed seeds keep results
    //! reproducible.

    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    /// Position on a line plus the step counter (the horizon clock).
    #[derive(Clone)]
    struct LineState {
        pos: i64,
        step: i64,
    }

    /// A walk on the integer line. Action 1 steps right (`pos + 1`), action 0
    /// steps left (`pos - 1`); the per-step reward equals the new position, so
    /// the higher-position branch yields strictly larger returns. The episode
    /// ends after `horizon` steps. Because the action changes the resulting
    /// STATE (not just the immediate reward), the subtree-return backup makes
    /// the rewarding action's child genuinely more valuable.
    struct LineWalk {
        horizon: i64,
    }

    impl MCTSEnv<LineState> for LineWalk {
        fn num_actions(&self, _s: &LineState) -> usize {
            2
        }
        fn apply_action(&self, s: &LineState, a: usize) -> ApplyResult<LineState> {
            let pos = s.pos + if a == 1 { 1 } else { -1 };
            let step = s.step + 1;
            ApplyResult {
                reward: pos as f64,
                done: step >= self.horizon,
                next: LineState { pos, step },
            }
        }
        fn is_terminal(&self, s: &LineState) -> bool {
            s.step >= self.horizon
        }
        fn rollout_depth(&self) -> usize {
            20
        }
    }

    fn root() -> LineState {
        LineState { pos: 0, step: 0 }
    }

    #[test]
    fn prefers_the_rewarding_action() {
        let res = mcts(
            Box::new(LineWalk { horizon: 6 }),
            root(),
            MCTSOptions { iterations: 300, ..Default::default() },
            SeededRandom::new(42),
        );
        assert_eq!(res.action, 1);
        assert_eq!(res.visits.len(), 2);
        // The right-stepping child reaches higher positions, so its subtree
        // return (the stored child value) is strictly larger.
        assert!(res.values[&1] > res.values[&0]);
    }

    #[test]
    fn root_child_visits_sum_to_iterations() {
        let res = mcts(
            Box::new(LineWalk { horizon: 6 }),
            root(),
            MCTSOptions::default(),
            SeededRandom::new(1),
        );
        // Every iteration's path passes through exactly one root child, so the
        // child visit counts sum to the iteration budget.
        let total: f64 = res.visits.values().sum();
        assert_eq!(total, 200.0);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let opts = MCTSOptions { iterations: 150, ..Default::default() };
        let r1 = mcts(Box::new(LineWalk { horizon: 6 }), root(), opts, SeededRandom::new(7));
        let r2 = mcts(Box::new(LineWalk { horizon: 6 }), root(), opts, SeededRandom::new(7));
        assert_eq!(r1.action, r2.action);
        assert_eq!(r1.visits, r2.visits);
        assert_eq!(r1.values, r2.values);
    }
}
