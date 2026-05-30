//! Port of `src/des/general/milp-bnb.ts` — Mixed-Integer Linear Programming via
//! depth-first Branch-and-Bound, modelled as a discrete-event system that
//! COMPOSES the [`IncrementalLP`] simplex solver and runs on the
//! [`TreeSearchStation`] template-method base.
//!
//! ## TS → Rust mapping
//!
//!   * `interface MILPProblem / MILPSolveOptions / MILPSolution / NodeEvent`
//!     → structs (`number` → `f64`, indices → `usize`, `T | null` → `Option<T>`).
//!     The TS field `A` becomes `a` (snake_case). `sense: 'max'|'min'` reuses
//!     [`Sense`] from `incremental_lp`.
//!   * The string unions become enums: `branchRule` → [`BranchRule`],
//!     `branchType` → [`BranchType`], `lpStatus` → [`LpStatus`], `prunedReason`
//!     → [`PrunedReason`], `MILPSolution.status` → [`MILPStatus`].
//!   * `class MILPBnBStation extends TreeSearchStation<MILPNode>` → a struct
//!     embedding [`StationCore`] + [`TreeSearchCore`] that `impl DESStation`
//!     (delegating `run_time_step` → `run_tree_search_step`, `has_work` →
//!     `!finished`) and `impl TreeSearchStation<MILPNode>` (the hooks). The
//!     single warm-started `IncrementalLP`, the DFS `stack`, the `trace`, and
//!     the branch-tie-break RNG are fields.
//!   * The TS `evaluate` hook MUTATED `node.ev` (caching the LP solution on the
//!     node). The Rust `TreeSearchStation::evaluate` takes `&N`, so instead the
//!     per-node evaluation is stored in a `HashMap<node_id, NodeEvalData>` on
//!     the station; `expand` / `on_prune` / `get_incumbent_x` read it back.
//!   * `pickBranchVar(..., rng = Math.random)` injects a
//!     `&mut dyn RandomSource` (a seeded mulberry32 field) — never an ambient
//!     global RNG.
//!   * `validateProblem` `throw`s → `panic!` (invariant violation).
//!   * `intrinsicCheck` validators are registered as `Validator<dyn DESStation>`
//!     that downcast back to `MILPBnBStation`. `boundValidator` was imported but
//!     unused in the TS source, so it is dropped.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::tree_search::{
    NodeEvaluation, SearchObjective, TreeSearchCore, TreeSearchStation,
};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::general::incremental_lp::{
    IncrementalLP, IncrementalLPInit, PivotMode, SolverStatus,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

pub use crate::des::general::incremental_lp::Sense;

// =============================================================================
// PROBLEM AND SOLUTION TYPES
// =============================================================================

/// Branching rule. (TS `'most-fractional' | 'first-fractional'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BranchRule {
    #[default]
    MostFractional,
    FirstFractional,
}

/// Direction a branch constraint was added in. (TS `'le' | 'ge'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BranchType {
    Le,
    Ge,
}

/// LP relaxation status recorded in a [`NodeEvent`]. (TS `'optimal' |
/// 'infeasible' | 'unbounded'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

/// Why a node was pruned. (TS `'infeasible' | 'unbounded' | 'bound' |
/// 'integer-feasible'`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrunedReason {
    Infeasible,
    Unbounded,
    Bound,
    IntegerFeasible,
}

/// Overall solver outcome. (TS `MILPSolution['status']`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MILPStatus {
    Optimal,
    Infeasible,
    Unbounded,
    MaxNodes,
}

/// A MILP instance: `max/min c·x  s.t.  A·x ≤ b, x ≥ 0, x_j ∈ ℤ for j ∈ I`.
/// (TS `interface MILPProblem`.)
#[derive(Clone, Debug)]
pub struct MILPProblem {
    pub sense: Sense,
    /// Objective coefficients, length n.
    pub c: Vec<f64>,
    /// Constraint matrix, m × n, rows are ≤ inequalities. (TS `A`.)
    pub a: Vec<Vec<f64>>,
    /// Right-hand sides, length m. Must be ≥ 0 (no Phase-1 yet).
    pub b: Vec<f64>,
    /// `integer_vars[j]` = true if x_j must be integer at optimality.
    pub integer_vars: Vec<bool>,
    /// Optional per-variable upper bounds (added as ≤ rows). `None` = unbounded.
    pub ub: Option<Vec<f64>>,
    pub var_names: Option<Vec<String>>,
    pub con_names: Option<Vec<String>>,
}

/// Solver options. (TS `interface MILPSolveOptions`; all optional.)
#[derive(Clone, Debug, Default)]
pub struct MILPSolveOptions {
    /// Max B&B nodes to explore. Default 10_000.
    pub max_nodes: Option<usize>,
    /// Max LP pivots per node. Default 200.
    pub lp_max_iters: Option<usize>,
    /// Tolerance for declaring a value integer. Default 1e-6.
    pub int_tol: Option<f64>,
    /// Branching rule. Default `MostFractional`.
    pub branch_rule: Option<BranchRule>,
    /// Print every node event to stderr. Default false.
    pub verbose: Option<bool>,
    /// Initial incumbent (lower bound for max / upper bound for min).
    pub initial_incumbent_z: Option<f64>,
    /// Seed for the random-tie-break PRNG in [`pick_branch_var`]. Default 1.
    pub branch_seed: Option<u32>,
}

/// A single node-processing event in the B&B trace. (TS `interface NodeEvent`.)
#[derive(Clone, Debug)]
pub struct NodeEvent {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub branch_var: Option<usize>,
    pub branch_type: Option<BranchType>,
    pub branch_value: Option<f64>,
    pub lp_status: LpStatus,
    pub lp_z: Option<f64>,
    /// Fractional integer-variable indices in the LP solution (empty if integer).
    pub fractional: Vec<usize>,
    pub pruned: bool,
    pub pruned_reason: Option<PrunedReason>,
    pub incumbent_updated: bool,
}

/// Final solution. (TS `interface MILPSolution`.)
#[derive(Clone, Debug)]
pub struct MILPSolution {
    pub status: MILPStatus,
    /// Best integer-feasible solution found. Empty if none.
    pub x: Vec<f64>,
    /// Best integer-feasible objective. ±∞ if none found.
    pub z: f64,
    /// Best dual bound at termination.
    pub best_bound: f64,
    /// Optimality gap `(best_bound − z) / max(|z|, 1)`. 0 at proven optimal.
    pub gap: f64,
    pub nodes_explored: usize,
    pub total_pivots: usize,
    pub trace: Vec<NodeEvent>,
}

// =============================================================================
// INTERNAL NODE TYPES
// =============================================================================

/// One branch constraint added to the root LP. (TS `interface MILPBranch`.)
#[derive(Clone, Debug)]
struct MILPBranch {
    coefs: Vec<f64>,
    rhs: f64,
    name: String,
}

/// A B&B node = the trail of branch constraints from root to here. (TS
/// `interface MILPNode`; the `ev?` cache lives in the station's map instead.)
#[derive(Clone, Debug)]
struct MILPNode {
    node_id: usize,
    parent_id: Option<usize>,
    depth: usize,
    branch_var: Option<usize>,
    branch_type: Option<BranchType>,
    branch_value: Option<f64>,
    trail: Vec<MILPBranch>,
}

/// Cached LP evaluation for a node (the TS `MILPNode.ev`).
#[derive(Clone, Debug)]
struct NodeEvalData {
    lp_status: LpStatus,
    lp_z: Option<f64>,
    x: Vec<f64>,
    fractional: Vec<usize>,
}

/// Fully-defaulted options (TS `Required<MILPSolveOptions>`).
#[derive(Clone, Copy, Debug)]
struct FilledOpts {
    max_nodes: usize,
    lp_max_iters: usize,
    int_tol: f64,
    branch_rule: BranchRule,
    verbose: bool,
    initial_incumbent_z: f64,
    branch_seed: u32,
}

// =============================================================================
// MILPBnBStation
// =============================================================================

/// Concrete leaf of [`TreeSearchStation`] performing DFS branch-and-bound.
pub struct MILPBnBStation {
    core: StationCore,
    search: TreeSearchCore<MILPNode>,
    /// Single shared IncrementalLP, kept warm by tree-walk.
    lp: IncrementalLP,
    /// Trail currently realised in `self.lp`.
    current_trail: Vec<MILPBranch>,
    /// DFS frontier.
    stack: Vec<MILPNode>,
    /// Per-node trace for diagnostics.
    pub trace: Vec<NodeEvent>,
    /// LP-relaxation cache, keyed by node id.
    node_evals: HashMap<usize, NodeEvalData>,
    pub total_pivots: usize,
    pub root_bound: Option<f64>,
    verbose: bool,
    lp_max_iters: usize,
    int_tol: f64,
    branch_rule: BranchRule,
    branch_rng: SeededRandom,
    integer_vars: Vec<bool>,
    n: usize,
    node_counter: usize,
    /// Latest LP_z observed at the frontier's deepest unfathomed open subtree.
    latest_open_lpz: f64,
}

fn downcast_milp(s: &dyn DESStation) -> &MILPBnBStation {
    s.as_any()
        .downcast_ref::<MILPBnBStation>()
        .expect("validator received a non-MILPBnBStation station")
}

impl MILPBnBStation {
    fn new(p: &MILPProblem, opts: FilledOpts) -> Self {
        let objective = if p.sense == Sense::Max {
            SearchObjective::Maximise
        } else {
            SearchObjective::Minimise
        };
        let mut search = TreeSearchCore::<MILPNode>::new(objective, opts.max_nodes as f64);
        if opts.initial_incumbent_z.is_finite() {
            search.incumbent_value = opts.initial_incumbent_z;
        }
        let n = p.c.len();
        let latest_open_lpz = if p.sense == Sense::Max {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        // Build root LP, encoding any explicit upper bounds as ≤ rows.
        let mut a: Vec<Vec<f64>> = p.a.iter().cloned().collect();
        let mut b: Vec<f64> = p.b.clone();
        let mut con_names: Vec<String> = match &p.con_names {
            Some(cn) => cn.clone(),
            None => (0..a.len()).map(|i| format!("c{}", i + 1)).collect(),
        };
        if let Some(ub) = &p.ub {
            for (j, &u) in ub.iter().enumerate().take(n) {
                if u.is_finite() {
                    let mut row = vec![0.0; n];
                    row[j] = 1.0;
                    a.push(row);
                    b.push(u);
                    con_names.push(format!("ub_x{j}"));
                }
            }
        }
        let mut lp = IncrementalLP::new(IncrementalLPInit {
            sense: p.sense,
            c: p.c.clone(),
            a,
            b,
            var_names: p.var_names.clone(),
            con_names: Some(con_names),
        });

        // Solve root LP up front, counting genuine (primal/dual) pivots.
        let root_pivots = lp
            .solve_to_optimum(opts.lp_max_iters)
            .iter()
            .filter(|e| matches!(e.mode, PivotMode::Primal | PivotMode::Dual))
            .count();
        let root_bound = if lp.status == SolverStatus::Optimal {
            Some(lp.get_z())
        } else {
            None
        };

        let mut station = MILPBnBStation {
            core: StationCore::new("milp-bnb"),
            search,
            lp,
            current_trail: Vec::new(),
            stack: Vec::new(),
            trace: Vec::new(),
            node_evals: HashMap::new(),
            total_pivots: root_pivots,
            root_bound,
            verbose: opts.verbose,
            lp_max_iters: opts.lp_max_iters,
            int_tol: opts.int_tol,
            branch_rule: opts.branch_rule,
            branch_rng: mulberry32(opts.branch_seed),
            integer_vars: p.integer_vars.clone(),
            n,
            node_counter: 0,
            latest_open_lpz,
        };

        // Push the root node.
        let root_id = station.node_counter;
        station.node_counter += 1;
        station.stack.push(MILPNode {
            node_id: root_id,
            parent_id: None,
            depth: 0,
            branch_var: None,
            branch_type: None,
            branch_value: None,
            trail: Vec::new(),
        });

        // Intrinsic invariants.
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "milp.search-finished",
                |s: &dyn DESStation| {
                    let st = downcast_milp(s);
                    st.is_finished() || st.get_nodes_processed() as f64 >= st.max_nodes_cap()
                },
                Some("finished".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_milp(s);
                    if st.is_finished() {
                        "finished".to_string()
                    } else {
                        format!(
                            "nodesProcessed={}/{}",
                            st.get_nodes_processed(),
                            st.max_nodes_cap()
                        )
                    }
                })),
                Some("milp-bnb-intrinsic".to_string()),
                Some("tree search did not exhaust the frontier nor hit the node cap".to_string()),
            )
            .boxed(),
        );
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "milp.incumbent-bounded-by-relaxation",
                |s: &dyn DESStation| {
                    let st = downcast_milp(s);
                    let inc = st.get_incumbent_value();
                    let root = match st.root_bound {
                        Some(r) => r,
                        None => return true,
                    };
                    if !inc.is_finite() {
                        return true;
                    }
                    match st.get_objective() {
                        SearchObjective::Maximise => inc <= root + 1e-6,
                        SearchObjective::Minimise => inc >= root - 1e-6,
                    }
                },
                Some("inc ⊆ LP relaxation bound".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_milp(s);
                    format!("inc={}  rootBound={:?}", st.get_incumbent_value(), st.root_bound)
                })),
                Some("milp-bnb-intrinsic".to_string()),
                Some(
                    "integer-feasible incumbent is OUTSIDE its LP relaxation — this would indicate a bug in evaluate() / branching".to_string(),
                ),
            )
            .boxed(),
        );

        station
    }

    /// Node cap (TS `maxNodesCap()`), used by an intrinsic validator.
    fn max_nodes_cap(&self) -> f64 {
        self.search.max_nodes
    }

    /// Best integer-feasible solution vector. (TS `getIncumbentX`.)
    pub fn get_incumbent_x(&self) -> Vec<f64> {
        match self.search.incumbent.as_ref() {
            None => Vec::new(),
            Some(node) => self
                .node_evals
                .get(&node.node_id)
                .map(|e| e.x.clone())
                .unwrap_or_default(),
        }
    }

    /// Walk the IncrementalLP from `current_trail` to `target`: pop branch
    /// constraints down to the LCA, then push the new tail. (TS `realiseTrail`.)
    fn realise_trail(&mut self, target: &[MILPBranch]) {
        let mut lca_len = 0;
        while lca_len < self.current_trail.len()
            && lca_len < target.len()
            && self.current_trail[lca_len].name == target[lca_len].name
        {
            lca_len += 1;
        }
        while self.current_trail.len() > lca_len {
            let last_idx = self.lp.tab.len() - 2; // last constraint row index
            self.lp.apply_remove_constraint(last_idx);
            self.current_trail.pop();
        }
        for c in &target[lca_len..] {
            self.lp
                .apply_add_constraint(&c.coefs, c.rhs, Some(c.name.clone()));
        }
        self.current_trail = target.to_vec();
    }
}

impl DESStation for MILPBnBStation {
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
    fn has_work(&self) -> bool {
        !self.search.finished
    }
}

impl TreeSearchStation<MILPNode> for MILPBnBStation {
    fn search_core(&self) -> &TreeSearchCore<MILPNode> {
        &self.search
    }
    fn search_core_mut(&mut self) -> &mut TreeSearchCore<MILPNode> {
        &mut self.search
    }

    fn pick_next(&mut self) -> Option<MILPNode> {
        self.stack.pop()
    }

    fn push_children(&mut self, children: Vec<MILPNode>) {
        // Push in reverse so the FIRST child is popped FIRST (DFS preorder).
        for child in children.into_iter().rev() {
            self.stack.push(child);
        }
    }

    fn evaluate(&mut self, node: &MILPNode) -> NodeEvaluation {
        self.realise_trail(&node.trail);
        let lp_max = self.lp_max_iters;
        let pivots = self
            .lp
            .solve_to_optimum(lp_max)
            .iter()
            .filter(|e| matches!(e.mode, PivotMode::Primal | PivotMode::Dual))
            .count();
        self.total_pivots += pivots;

        let maximise = self.get_objective() == SearchObjective::Maximise;
        let mut ev = NodeEvent {
            node_id: node.node_id,
            parent_id: node.parent_id,
            depth: node.depth,
            branch_var: node.branch_var,
            branch_type: node.branch_type,
            branch_value: node.branch_value,
            lp_status: match self.lp.status {
                SolverStatus::Infeasible => LpStatus::Infeasible,
                SolverStatus::Unbounded => LpStatus::Unbounded,
                _ => LpStatus::Optimal,
            },
            lp_z: None,
            fractional: Vec::new(),
            pruned: false,
            pruned_reason: None,
            incumbent_updated: false,
        };

        if self.lp.status == SolverStatus::Infeasible {
            ev.pruned = true;
            ev.pruned_reason = Some(PrunedReason::Infeasible);
            if self.verbose {
                eprintln!("{}", format_node(&ev));
            }
            self.trace.push(ev);
            self.node_evals.insert(
                node.node_id,
                NodeEvalData {
                    lp_status: LpStatus::Infeasible,
                    lp_z: None,
                    x: Vec::new(),
                    fractional: Vec::new(),
                },
            );
            return NodeEvaluation::new(
                if maximise {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
                true,
            );
        }
        if self.lp.status == SolverStatus::Unbounded {
            ev.pruned = true;
            ev.pruned_reason = Some(PrunedReason::Unbounded);
            if self.verbose {
                eprintln!("{}", format_node(&ev));
            }
            self.trace.push(ev);
            self.node_evals.insert(
                node.node_id,
                NodeEvalData {
                    lp_status: LpStatus::Unbounded,
                    lp_z: None,
                    x: Vec::new(),
                    fractional: Vec::new(),
                },
            );
            return NodeEvaluation::new(
                if maximise {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
                true,
            );
        }

        let lp_z = self.lp.get_z();
        let x = self.lp.get_x();
        let fractionals = list_fractionals(&x, &self.integer_vars, self.int_tol);
        self.node_evals.insert(
            node.node_id,
            NodeEvalData {
                lp_status: LpStatus::Optimal,
                lp_z: Some(lp_z),
                x: x.clone(),
                fractional: fractionals.clone(),
            },
        );
        ev.lp_z = Some(lp_z);
        ev.fractional = fractionals.iter().take(10).copied().collect();
        self.latest_open_lpz = lp_z;

        if fractionals.is_empty() {
            // Integer-feasible leaf — candidate incumbent.
            ev.incumbent_updated = self.is_improvement(lp_z);
            if self.verbose {
                eprintln!("{}", format_node(&ev));
            }
            self.trace.push(ev);
            return NodeEvaluation::feasible(lp_z, true, lp_z);
        }
        // Non-leaf — fathoming-by-bound is handled by the base's should_prune.
        NodeEvaluation::new(lp_z, false)
    }

    fn expand(&mut self, node: &MILPNode, ev: &NodeEvaluation) -> Vec<MILPNode> {
        let (x, fractionals) = {
            let e = self
                .node_evals
                .get(&node.node_id)
                .expect("expand on an evaluated node");
            (e.x.clone(), e.fractional.clone())
        };
        let branch_on = pick_branch_var(&x, &fractionals, self.branch_rule, &mut self.branch_rng);
        let xv = x[branch_on];
        let lo = xv.floor();
        let hi = xv.ceil();

        let frac10: Vec<usize> = fractionals.iter().take(10).copied().collect();
        let last_is_this = self
            .trace
            .last()
            .map(|e| e.node_id == node.node_id)
            .unwrap_or(false);
        if self.verbose {
            if last_is_this {
                eprintln!(
                    "{}  → branch on x{} (= {:.4})",
                    format_node(self.trace.last().unwrap()),
                    branch_on,
                    xv
                );
            } else {
                eprintln!("  branch on x{branch_on} (= {xv:.4})");
            }
        }
        if !last_is_this {
            // evaluate() only pushes an event in pruned/leaf cases; for branched
            // nodes we record one here.
            self.trace.push(NodeEvent {
                node_id: node.node_id,
                parent_id: node.parent_id,
                depth: node.depth,
                branch_var: node.branch_var,
                branch_type: node.branch_type,
                branch_value: node.branch_value,
                lp_status: LpStatus::Optimal,
                lp_z: Some(ev.bound),
                fractional: frac10,
                pruned: false,
                pruned_reason: None,
                incumbent_updated: false,
            });
        }

        let mut coefs_le = vec![0.0; self.n];
        coefs_le[branch_on] = 1.0;
        let mut coefs_ge = vec![0.0; self.n];
        coefs_ge[branch_on] = -1.0;

        let mut left_trail = node.trail.clone();
        left_trail.push(MILPBranch {
            coefs: coefs_le,
            rhs: lo,
            name: format!("x{branch_on}≤{lo}"),
        });
        let left = MILPNode {
            node_id: self.node_counter,
            parent_id: Some(node.node_id),
            depth: node.depth + 1,
            branch_var: Some(branch_on),
            branch_type: Some(BranchType::Le),
            branch_value: Some(lo),
            trail: left_trail,
        };
        self.node_counter += 1;

        let mut right_trail = node.trail.clone();
        right_trail.push(MILPBranch {
            coefs: coefs_ge,
            rhs: -hi,
            name: format!("x{branch_on}≥{hi}"),
        });
        let right = MILPNode {
            node_id: self.node_counter,
            parent_id: Some(node.node_id),
            depth: node.depth + 1,
            branch_var: Some(branch_on),
            branch_type: Some(BranchType::Ge),
            branch_value: Some(hi),
            trail: right_trail,
        };
        self.node_counter += 1;

        vec![left, right] // left first so DFS explores le before ge
    }

    fn on_prune(&mut self, node: &MILPNode, _ev: &NodeEvaluation) {
        if let Some(entry) = self.trace.last_mut() {
            if entry.node_id == node.node_id {
                if !entry.pruned {
                    entry.pruned = true;
                    entry.pruned_reason = Some(PrunedReason::Bound);
                }
                return;
            }
        }
        let (lp_status, lp_z, frac) = match self.node_evals.get(&node.node_id) {
            Some(e) => (
                e.lp_status,
                e.lp_z,
                e.fractional.iter().take(10).copied().collect(),
            ),
            None => (LpStatus::Optimal, None, Vec::new()),
        };
        self.trace.push(NodeEvent {
            node_id: node.node_id,
            parent_id: node.parent_id,
            depth: node.depth,
            branch_var: node.branch_var,
            branch_type: node.branch_type,
            branch_value: node.branch_value,
            lp_status,
            lp_z,
            fractional: frac,
            pruned: true,
            pruned_reason: Some(PrunedReason::Bound),
            incumbent_updated: false,
        });
    }

    fn on_incumbent_update(&mut self, node: &MILPNode, _value: f64) {
        if let Some(entry) = self.trace.last_mut() {
            if entry.node_id == node.node_id {
                entry.pruned = true;
                entry.pruned_reason = Some(PrunedReason::IntegerFeasible);
                entry.incumbent_updated = true;
            }
        }
    }

    fn current_best_bound(&self) -> f64 {
        if self.stack.is_empty()
            && self.root_bound.is_some()
            && self.search.incumbent_value.is_finite()
        {
            // Fully explored — the proven bound is the incumbent itself.
            return self.search.incumbent_value;
        }
        self.root_bound.unwrap_or(self.latest_open_lpz)
    }
}

// =============================================================================
// MAIN SOLVER
// =============================================================================

/// Solve a MILP via depth-first branch-and-bound. (TS `solveMILP`.)
pub fn solve_milp(p: &MILPProblem, opts: MILPSolveOptions) -> MILPSolution {
    let filled = FilledOpts {
        max_nodes: opts.max_nodes.unwrap_or(10_000),
        lp_max_iters: opts.lp_max_iters.unwrap_or(200),
        int_tol: opts.int_tol.unwrap_or(1e-6),
        branch_rule: opts.branch_rule.unwrap_or(BranchRule::MostFractional),
        verbose: opts.verbose.unwrap_or(false),
        initial_incumbent_z: opts
            .initial_incumbent_z
            .unwrap_or(if p.sense == Sense::Max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }),
        branch_seed: opts.branch_seed.unwrap_or(1),
    };
    validate_problem(p);

    let station = Rc::new(RefCell::new(MILPBnBStation::new(p, filled)));
    run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions::default(),
    );

    let st = station.borrow();
    let nodes = st.get_nodes_processed();
    // TS wrote `(nodes >= max && !finished) || nodes >= max`, which is logically
    // just `nodes >= max` — simplified here, preserving the original semantics.
    let stopped_early = nodes >= filled.max_nodes;
    let has_incumbent = st.get_incumbent().is_some();
    let is_optimal = !stopped_early && has_incumbent;
    let status = if stopped_early {
        MILPStatus::MaxNodes
    } else if !has_incumbent {
        MILPStatus::Infeasible
    } else {
        MILPStatus::Optimal
    };

    let z = if !has_incumbent {
        if p.sense == Sense::Max {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    } else {
        st.get_incumbent_value()
    };
    let final_best_bound = if is_optimal {
        z
    } else {
        st.root_bound.unwrap_or(if p.sense == Sense::Max {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        })
    };
    let gap = if !z.is_finite() {
        f64::INFINITY
    } else {
        (final_best_bound - z).abs() / 1.0_f64.max(z.abs())
    };

    MILPSolution {
        status,
        x: st.get_incumbent_x(),
        z,
        best_bound: final_best_bound,
        gap,
        nodes_explored: nodes,
        total_pivots: st.total_pivots,
        trace: st.trace.clone(),
    }
}

// =============================================================================
// HELPERS
// =============================================================================

fn validate_problem(p: &MILPProblem) {
    let n = p.c.len();
    if p.integer_vars.len() != n {
        panic!("integerVars length {} ≠ c length {n}", p.integer_vars.len());
    }
    if p.a.iter().any(|row| row.len() != n) {
        panic!("A has rows of different length than c");
    }
    if p.b.len() != p.a.len() {
        panic!("b length {} ≠ A length {}", p.b.len(), p.a.len());
    }
    if p.b.iter().any(|&v| v < 0.0) {
        panic!("b has negative entries; only b ≥ 0 supported (no Phase-1 yet).");
    }
    if let Some(ub) = &p.ub {
        if ub.len() != n {
            panic!("ub length {} ≠ c length {n}", ub.len());
        }
    }
}

fn list_fractionals(x: &[f64], integer_vars: &[bool], tol: f64) -> Vec<usize> {
    let mut out = Vec::new();
    for (j, &xj) in x.iter().enumerate() {
        if !integer_vars[j] {
            continue;
        }
        let f = xj - xj.floor();
        if f > tol && f < 1.0 - tol {
            out.push(j);
        }
    }
    out
}

fn pick_branch_var(
    x: &[f64],
    fractionals: &[usize],
    rule: BranchRule,
    rng: &mut dyn RandomSource,
) -> usize {
    if rule == BranchRule::FirstFractional {
        return fractionals[0];
    }
    // Most-fractional: maximise f·(1−f) with RANDOM TIE-BREAKING so symmetric
    // MILPs explore varied tree shapes instead of always taking the low index.
    let eps = 1e-12;
    let mut best = fractionals[0];
    let mut best_score = f64::NEG_INFINITY;
    let mut tie_count = 0usize;
    for &j in fractionals {
        let f = x[j] - x[j].floor();
        let score = f * (1.0 - f);
        if tie_count == 0 || score > best_score + eps {
            best_score = score;
            best = j;
            tie_count = 1;
        } else if score >= best_score - eps {
            tie_count += 1;
            if rng.next_float() * (tie_count as f64) < 1.0 {
                best = j;
            }
        }
    }
    best
}

fn format_node(ev: &NodeEvent) -> String {
    let lab = match (ev.branch_var, ev.branch_type, ev.branch_value) {
        (Some(v), Some(bt), Some(val)) => {
            let sym = if bt == BranchType::Le { "≤" } else { "≥" };
            format!("x{v}{sym}{val}")
        }
        _ => "root".to_string(),
    };
    let pruned = if ev.pruned {
        format!("  pruned[{:?}]", ev.pruned_reason)
    } else {
        String::new()
    };
    let inc = if ev.incumbent_updated {
        "  ★ NEW INCUMBENT"
    } else {
        ""
    };
    let z = match ev.lp_z {
        None => "N/A".to_string(),
        Some(v) => format!("{v:.4}"),
    };
    let frac = if ev.fractional.is_empty() {
        String::new()
    } else {
        let joined: Vec<String> = ev.fractional.iter().map(|f| f.to_string()).collect();
        format!("  fractional={{{}}}", joined.join(","))
    };
    format!(
        "  node[{:>4}]  d={:>2}  {:<12}  LP={z}{frac}{pruned}{inc}",
        ev.node_id, ev.depth, lab
    )
}

// =============================================================================
// CONVENIENCE BUILDERS
// =============================================================================

/// Build a 0/1 knapsack as a MILP. (TS `buildKnapsackMILP`.)
pub fn build_knapsack_milp(values: Vec<f64>, weights: Vec<f64>, capacity: f64) -> MILPProblem {
    if values.len() != weights.len() {
        panic!("values and weights must be same length");
    }
    let n = values.len();
    MILPProblem {
        sense: Sense::Max,
        c: values,
        a: vec![weights],
        b: vec![capacity],
        integer_vars: vec![true; n],
        ub: Some(vec![1.0; n]),
        var_names: Some((0..n).map(|i| format!("x{i}")).collect()),
        con_names: Some(vec!["capacity".to_string()]),
    }
}

/// An uncapacitated facility-location instance. (TS `FacilityLocationProblem`.)
#[derive(Clone, Debug)]
pub struct FacilityLocationProblem {
    /// Fixed cost f_i for opening facility i.
    pub fixed_costs: Vec<f64>,
    /// Service cost `service_costs[i][j]` for facility i serving customer j.
    pub service_costs: Vec<Vec<f64>>,
}

/// Build an uncapacitated facility-location MILP (LP-relaxed demand; see the TS
/// source's note — the GE demand part is dropped because IncrementalLP rejects
/// negative initial RHS). (TS `buildFacilityLocationMILP`.)
pub fn build_facility_location_milp(p: &FacilityLocationProblem) -> MILPProblem {
    let f_count = p.fixed_costs.len();
    if p.service_costs.len() != f_count {
        panic!("serviceCosts.length must equal fixedCosts.length");
    }
    if f_count == 0 {
        panic!("at least one facility required");
    }
    let c_count = p.service_costs[0].len();
    if c_count == 0 {
        panic!("at least one customer required");
    }
    let n_y = f_count;
    let n_x = f_count * c_count;
    let n = n_y + n_x;
    let x_idx = |i: usize, j: usize| -> usize { n_y + i * c_count + j };

    let mut c = vec![0.0; n];
    for i in 0..f_count {
        c[i] = p.fixed_costs[i];
    }
    for i in 0..f_count {
        for j in 0..c_count {
            c[x_idx(i, j)] = p.service_costs[i][j];
        }
    }

    let mut a: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut con_names: Vec<String> = Vec::new();
    // demand-le_j:  Σ_i x_{ij} ≤ 1
    for j in 0..c_count {
        let mut row = vec![0.0; n];
        for i in 0..f_count {
            row[x_idx(i, j)] = 1.0;
        }
        a.push(row);
        b.push(1.0);
        con_names.push(format!("demand_le_c{j}"));
    }
    // linking_ij:  x_{ij} − y_i ≤ 0
    for i in 0..f_count {
        for j in 0..c_count {
            let mut row = vec![0.0; n];
            row[x_idx(i, j)] = 1.0;
            row[i] = -1.0;
            a.push(row);
            b.push(0.0);
            con_names.push(format!("link_f{i}_c{j}"));
        }
    }

    let mut integer_vars = vec![false; n];
    for i in 0..f_count {
        integer_vars[i] = true; // y_i is integer
    }
    let ub = vec![1.0; n];

    let mut var_names: Vec<String> = (0..f_count).map(|i| format!("y{i}")).collect();
    for k in 0..(f_count * c_count) {
        let i = k / c_count;
        let j = k % c_count;
        var_names.push(format!("x_{i}_{j}"));
    }

    MILPProblem {
        sense: Sense::Min,
        c,
        a,
        b,
        integer_vars,
        ub: Some(ub),
        var_names: Some(var_names),
        con_names: Some(con_names),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! MILP B&B smoke tests with fixed branch seeds. A tiny 0/1 knapsack is
    //! solved to its known integer optimum (value 220), and a small two-variable
    //! integer program whose LP relaxation is fractional (optimum 21 at x=3,
    //! y=1.5) is driven by branching to its true integer optimum (20 at x=4,
    //! y=0). The branch-rule variants both reach the same optimum.

    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn solves_tiny_knapsack_to_optimum() {
        // items (value, weight): (60,10),(100,20),(120,30); capacity 50.
        // Best 0/1 choice is {1,2}: value 220 at weight 50.
        let p = build_knapsack_milp(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let sol = solve_milp(
            &p,
            MILPSolveOptions {
                branch_seed: Some(1),
                ..Default::default()
            },
        );

        assert_eq!(sol.status, MILPStatus::Optimal);
        assert!(approx(sol.z, 220.0), "z = {}", sol.z);
        assert!(approx(sol.x[0], 0.0), "x0 = {}", sol.x[0]);
        assert!(approx(sol.x[1], 1.0), "x1 = {}", sol.x[1]);
        assert!(approx(sol.x[2], 1.0), "x2 = {}", sol.x[2]);
        assert!(sol.gap.abs() < 1e-6, "gap = {}", sol.gap);
    }

    #[test]
    fn solves_milp_requiring_branching() {
        // max 5x + 4y  s.t.  6x + 4y ≤ 24,  x + 2y ≤ 6,  x,y ≥ 0 integer.
        // LP relaxation optimum is z = 21 at (3, 1.5); integer optimum is
        // z = 20 at (4, 0).
        let p = MILPProblem {
            sense: Sense::Max,
            c: vec![5.0, 4.0],
            a: vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            b: vec![24.0, 6.0],
            integer_vars: vec![true, true],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let sol = solve_milp(
            &p,
            MILPSolveOptions {
                branch_seed: Some(3),
                ..Default::default()
            },
        );

        assert_eq!(sol.status, MILPStatus::Optimal);
        assert!(approx(sol.z, 20.0), "z = {}", sol.z);
        assert!(approx(sol.x[0], 4.0), "x = {:?}", sol.x);
        assert!(approx(sol.x[1], 0.0), "y = {:?}", sol.x);
        // The relaxation bound (21) dominates the integer optimum (20).
        assert!(
            sol.nodes_explored >= 2,
            "expected branching, nodes={}",
            sol.nodes_explored
        );
    }

    #[test]
    fn first_fractional_rule_also_optimal() {
        let p = build_knapsack_milp(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let sol = solve_milp(
            &p,
            MILPSolveOptions {
                branch_rule: Some(BranchRule::FirstFractional),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, MILPStatus::Optimal);
        assert!(approx(sol.z, 220.0), "z = {}", sol.z);
    }
}
