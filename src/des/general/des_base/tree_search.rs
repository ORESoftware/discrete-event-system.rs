//! Port of `src/des/general/des-base/tree-search.ts`.
//!
//! Template-method base for TREE-STRUCTURED search (MILP branch-and-bound,
//! MCTS/UCT, A*, beam, alpha-beta, generic best/depth/breadth-first) over a
//! generic node `N`. Maintain a frontier of unexplored nodes plus the best
//! feasible solution found so far (the incumbent); each tick selects → evaluates
//! → (updates incumbent?) → (prunes? / expands).
//!
//! ## Rust shape (faithful translation of the TS abstract class)
//!
//!   * `type SearchObjective` → enum [`SearchObjective`].
//!   * `interface NodeEvaluation` → struct [`NodeEvaluation`] (`value?` →
//!     `Option<f64>`, `isFeasible?` → `bool` defaulting to `false`).
//!   * `abstract class TreeSearchStation<N> extends DESStation` → trait
//!     [`TreeSearchStation`]`<N>: DESStation`. Rust traits can't hold fields, so
//!     the shared protected state (objective, counters, incumbent, histories,
//!     `maxNodes`) lives in [`TreeSearchCore`]`<N>`, surfaced via the required
//!     `search_core` / `search_core_mut` accessors.
//!   * TEMPLATE METHOD: `runTimeStep` (final) → the provided
//!     [`TreeSearchStation::run_tree_search_step`]; a concrete station's
//!     `DESStation::run_time_step` just calls it. Required hooks `pickNext` /
//!     `evaluate` / `expand` / `pushChildren` are required trait fns;
//!     `shouldPrune` / `onIncumbentUpdate` / `onPrune` / `onExpand` / `onFinish`
//!     / `currentBestBound` are provided defaults.
//!   * `incumbentValue` seeded to ±Infinity by objective; `maxNodes` defaults to
//!     `f64::INFINITY` (no cap).
//!
//! `N: Clone` is required because the template stores `incumbent = node` while
//! also handing the node to `evaluate`/`expand` (TS passed an object reference
//! freely; in Rust we clone). For arena-style searches `N` is a cheap index.

use crate::des::general::des_base::station::DESStation;

const TIE_EPS: f64 = 1e-12;

/// Direction of optimisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchObjective {
    Minimise,
    Maximise,
}

/// Evaluation of a single node.
#[derive(Clone, Copy, Debug)]
pub struct NodeEvaluation {
    /// Bound on the objective achievable in this subtree (lower bound for
    /// minimise, upper bound for maximise).
    pub bound: f64,
    /// True if no further branching can occur from this node.
    pub is_leaf: bool,
    /// Concrete objective value when the node is feasible; falls back to
    /// `bound` when `None`.
    pub value: Option<f64>,
    /// True iff the node yields a complete feasible solution (incumbent
    /// candidate). TS `isFeasible?` defaults to `false`.
    pub is_feasible: bool,
}

impl NodeEvaluation {
    /// A non-feasible interior/leaf evaluation carrying only a `bound`.
    pub fn new(bound: f64, is_leaf: bool) -> Self {
        NodeEvaluation {
            bound,
            is_leaf,
            value: None,
            is_feasible: false,
        }
    }

    /// A feasible-solution evaluation with a concrete `value`.
    pub fn feasible(bound: f64, is_leaf: bool, value: f64) -> Self {
        NodeEvaluation {
            bound,
            is_leaf,
            value: Some(value),
            is_feasible: true,
        }
    }
}

/// Shared protected state of the TS `abstract class` (counters + incumbent +
/// histories). Each concrete station embeds one and exposes it via the trait's
/// `search_core` / `search_core_mut` accessors.
pub struct TreeSearchCore<N> {
    pub objective: SearchObjective,
    pub nodes_processed: usize,
    pub nodes_expanded: usize,
    pub nodes_pruned: usize,
    pub nodes_fathomed_by_bound: usize,
    pub nodes_incumbent_updates: usize,
    pub finished: bool,
    /// Best feasible objective so far; seeded to ±∞ in the worse-than-anything
    /// direction so any feasible value updates it.
    pub incumbent_value: f64,
    /// Node that produced the current incumbent.
    pub incumbent: Option<N>,
    /// Cap on nodes processed; `f64::INFINITY` = no cap.
    pub max_nodes: f64,
    pub incumbent_history: Vec<f64>,
    pub best_bound_history: Vec<f64>,
}

impl<N> TreeSearchCore<N> {
    pub fn new(objective: SearchObjective, max_nodes: f64) -> Self {
        let incumbent_value = match objective {
            SearchObjective::Minimise => f64::INFINITY,
            SearchObjective::Maximise => f64::NEG_INFINITY,
        };
        TreeSearchCore {
            objective,
            nodes_processed: 0,
            nodes_expanded: 0,
            nodes_pruned: 0,
            nodes_fathomed_by_bound: 0,
            nodes_incumbent_updates: 0,
            finished: false,
            incumbent_value,
            incumbent: None,
            max_nodes,
            incumbent_history: Vec::new(),
            best_bound_history: Vec::new(),
        }
    }

    /// Unbounded search (no node cap).
    pub fn unbounded(objective: SearchObjective) -> Self {
        Self::new(objective, f64::INFINITY)
    }
}

/// Template-method contract for tree-structured search. `DESStation` is a
/// supertrait; a concrete station implements `DESStation::run_time_step` by
/// delegating to [`run_tree_search_step`](TreeSearchStation::run_tree_search_step).
pub trait TreeSearchStation<N: Clone>: DESStation {
    // ── Required accessors ────────────────────────────────────────────────────
    fn search_core(&self) -> &TreeSearchCore<N>;
    fn search_core_mut(&mut self) -> &mut TreeSearchCore<N>;

    // ── Required hooks ─────────────────────────────────────────────────────────

    /// Pick the next node to process, or `None` if the search is exhausted.
    fn pick_next(&mut self) -> Option<N>;
    /// Evaluate one node (bound, leaf-ness, optional feasible value).
    fn evaluate(&mut self, node: &N) -> NodeEvaluation;
    /// Expand a non-leaf node into children.
    fn expand(&mut self, node: &N, ev: &NodeEvaluation) -> Vec<N>;
    /// Push expanded children onto the frontier.
    fn push_children(&mut self, children: Vec<N>);

    // ── Optional hooks (provided defaults) ──────────────────────────────────────

    /// Default prune rule: a node whose bound is dominated by the incumbent can
    /// be discarded.
    fn should_prune(&self, _node: &N, ev: &NodeEvaluation) -> bool {
        self.bound_is_dominated(ev.bound)
    }
    fn on_incumbent_update(&mut self, _node: &N, _value: f64) {}
    fn on_prune(&mut self, _node: &N, _ev: &NodeEvaluation) {}
    fn on_expand(&mut self, _node: &N, _children: &[N]) {}
    fn on_finish(&mut self) {}

    /// Best primal-side bound over open frontier nodes; default ±∞ (override
    /// when the frontier is a bounded priority queue).
    fn current_best_bound(&self) -> f64 {
        match self.search_core().objective {
            SearchObjective::Minimise => f64::NEG_INFINITY,
            SearchObjective::Maximise => f64::INFINITY,
        }
    }

    // ── Internal helpers (provided) ─────────────────────────────────────────────

    /// Is `bound` worse than (or equal to) the incumbent? Equal-to defeats it.
    fn bound_is_dominated(&self, bound: f64) -> bool {
        let core = self.search_core();
        match core.objective {
            SearchObjective::Minimise => bound >= core.incumbent_value - TIE_EPS,
            SearchObjective::Maximise => bound <= core.incumbent_value + TIE_EPS,
        }
    }

    /// Is `value` strictly better than the incumbent?
    fn is_improvement(&self, value: f64) -> bool {
        let core = self.search_core();
        match core.objective {
            SearchObjective::Minimise => value < core.incumbent_value - TIE_EPS,
            SearchObjective::Maximise => value > core.incumbent_value + TIE_EPS,
        }
    }

    // ── Template method (do NOT override) ───────────────────────────────────────

    /// Drives one node of the search. The TS `runTimeStep` was `final`; a
    /// concrete `DESStation::run_time_step` should simply call this.
    fn run_tree_search_step(&mut self) {
        if self.search_core().finished {
            return;
        }
        if self.search_core().nodes_processed as f64 >= self.search_core().max_nodes {
            self.search_core_mut().finished = true;
            self.on_finish();
            return;
        }
        let node = match self.pick_next() {
            None => {
                self.search_core_mut().finished = true;
                self.on_finish();
                return;
            }
            Some(n) => n,
        };
        self.search_core_mut().nodes_processed += 1;
        let ev = self.evaluate(&node);
        if ev.is_feasible {
            if let Some(value) = ev.value {
                if self.is_improvement(value) {
                    {
                        let core = self.search_core_mut();
                        core.incumbent = Some(node.clone());
                        core.incumbent_value = value;
                        core.nodes_incumbent_updates += 1;
                    }
                    self.on_incumbent_update(&node, value);
                }
            }
        }
        if self.should_prune(&node, &ev) {
            self.search_core_mut().nodes_pruned += 1;
            if self.bound_is_dominated(ev.bound) {
                self.search_core_mut().nodes_fathomed_by_bound += 1;
            }
            self.on_prune(&node, &ev);
        } else if !ev.is_leaf {
            let children = self.expand(&node, &ev);
            self.search_core_mut().nodes_expanded += 1;
            self.on_expand(&node, &children);
            if !children.is_empty() {
                self.push_children(children);
            }
        }
        let iv = self.search_core().incumbent_value;
        let bb = self.current_best_bound();
        let core = self.search_core_mut();
        core.incumbent_history.push(iv);
        core.best_bound_history.push(bb);
    }

    // ── Public accessors ────────────────────────────────────────────────────────

    fn get_incumbent(&self) -> Option<&N> {
        self.search_core().incumbent.as_ref()
    }
    fn get_incumbent_value(&self) -> f64 {
        self.search_core().incumbent_value
    }
    fn get_nodes_processed(&self) -> usize {
        self.search_core().nodes_processed
    }
    fn get_nodes_expanded(&self) -> usize {
        self.search_core().nodes_expanded
    }
    fn get_nodes_pruned(&self) -> usize {
        self.search_core().nodes_pruned
    }
    fn get_nodes_fathomed_by_bound(&self) -> usize {
        self.search_core().nodes_fathomed_by_bound
    }
    fn get_nodes_incumbent_updates(&self) -> usize {
        self.search_core().nodes_incumbent_updates
    }
    fn is_finished(&self) -> bool {
        self.search_core().finished
    }
    fn get_objective(&self) -> SearchObjective {
        self.search_core().objective
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
    use crate::des::general::des_base::station::StationCore;
    use crate::des::shared::capabilities::SeededRandom;

    /// A node in the trivial two-arm bandit tree: the root, or a pulled arm with
    /// a known reward.
    #[derive(Clone, Debug)]
    struct BanditNode {
        reward: f64,
        is_root: bool,
    }

    /// Minimal best-first search over a one-level bandit tree. `pick_next`
    /// selects the highest-bound frontier node, breaking ties at random via
    /// `argmax.rs` (the header note that tree search "wants argmax for child
    /// selection").
    struct BanditSearch {
        core: StationCore,
        search: TreeSearchCore<BanditNode>,
        frontier: Vec<BanditNode>,
        rng: SeededRandom,
    }

    impl BanditSearch {
        fn new(seed: u32) -> Self {
            let mut frontier = Vec::new();
            frontier.push(BanditNode {
                reward: 0.0,
                is_root: true,
            });
            BanditSearch {
                core: StationCore::new("bandit"),
                search: TreeSearchCore::unbounded(SearchObjective::Maximise),
                frontier,
                rng: SeededRandom::new(seed),
            }
        }
    }

    impl DESStation for BanditSearch {
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
            self.run_tree_search_step();
        }
        fn has_work(&self) -> bool {
            !self.search.finished
        }
    }

    impl TreeSearchStation<BanditNode> for BanditSearch {
        fn search_core(&self) -> &TreeSearchCore<BanditNode> {
            &self.search
        }
        fn search_core_mut(&mut self) -> &mut TreeSearchCore<BanditNode> {
            &mut self.search
        }

        fn pick_next(&mut self) -> Option<BanditNode> {
            if self.frontier.is_empty() {
                return None;
            }
            let bounds: Vec<f64> = self
                .frontier
                .iter()
                .map(|n| if n.is_root { f64::INFINITY } else { n.reward })
                .collect();
            let idx = arg_max_with_tie_break(&bounds, &mut self.rng, ARGMAX_EPS_DEFAULT)
                .expect("non-empty frontier");
            Some(self.frontier.remove(idx))
        }

        fn evaluate(&mut self, node: &BanditNode) -> NodeEvaluation {
            if node.is_root {
                // Root: interior node, never feasible, never pruned.
                NodeEvaluation::new(f64::INFINITY, false)
            } else {
                NodeEvaluation::feasible(node.reward, true, node.reward)
            }
        }

        fn expand(&mut self, node: &BanditNode, _ev: &NodeEvaluation) -> Vec<BanditNode> {
            if node.is_root {
                vec![
                    BanditNode {
                        reward: 1.0,
                        is_root: false,
                    },
                    BanditNode {
                        reward: 5.0,
                        is_root: false,
                    },
                ]
            } else {
                Vec::new()
            }
        }

        fn push_children(&mut self, children: Vec<BanditNode>) {
            self.frontier.extend(children);
        }
    }

    fn run_to_completion(search: &mut BanditSearch) {
        for _ in 0..100 {
            if search.is_finished() {
                break;
            }
            search.run_time_step();
        }
    }

    #[test]
    fn picks_the_better_arm() {
        let mut search = BanditSearch::new(12345);
        run_to_completion(&mut search);
        assert!(search.is_finished());
        assert_eq!(search.get_incumbent_value(), 5.0);
        let inc = search.get_incumbent().expect("an incumbent");
        assert_eq!(inc.reward, 5.0);
        // root + two arms processed; best-first sets the 5.0 incumbent once
        // (the later 1.0 arm is not an improvement).
        assert_eq!(search.get_nodes_processed(), 3);
        assert_eq!(search.get_nodes_incumbent_updates(), 1);
    }

    #[test]
    fn dominated_arms_are_fathomed() {
        let mut search = BanditSearch::new(999);
        run_to_completion(&mut search);
        // Each feasible leaf's bound equals its value, so once the 5.0
        // incumbent is set both arm leaves are dominated → pruned + fathomed.
        assert_eq!(search.get_nodes_expanded(), 1);
        assert_eq!(search.get_nodes_fathomed_by_bound(), 2);
        assert_eq!(search.get_nodes_pruned(), 2);
    }

    #[test]
    fn objective_seeds_incumbent_to_infinity() {
        let min_core = TreeSearchCore::<u32>::unbounded(SearchObjective::Minimise);
        assert_eq!(min_core.incumbent_value, f64::INFINITY);
        let max_core = TreeSearchCore::<u32>::unbounded(SearchObjective::Maximise);
        assert_eq!(max_core.incumbent_value, f64::NEG_INFINITY);
    }
}
