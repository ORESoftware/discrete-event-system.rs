//! Port of `src/des/general/max-flow.ts` — maximum flow / min-cut expressed as a
//! fixed-point DES iteration.
//!
//! Nodes are stationary optimisation state; one Edmonds-Karp augmentation is one
//! DES tick driven by the [`FixedPointIterationStation`] template. This keeps the
//! max-flow model in the same iterative-algorithm family as Benders, SDDP, value
//! iteration, and MILP branch-and-bound.
//!
//! MIGRATION NOTES
//!   * `class MaxFlowStation extends FixedPointIterationStation<MaxFlowState>` →
//!     a struct embedding both [`StationCore`] and [`FixedPointCore`], with
//!     `impl DESStation` + `impl FixedPointIterationStation<MaxFlowState>`.
//!   * `new Set(sourceSide)` for the min-cut side → [`HashSet<usize>`].
//!   * node/edge indices are `usize`; capacities/flow are `f64`.
//!   * `validateMaxFlowProblem` throws → returns `Result<(), PreconditionError>`;
//!     the constructor `.expect()`s it (a construction-time invariant → panic).
//!   * `assertNoValidationFailures` threw → [`assert_no_validation_failures`]
//!     returns `Result`; `solve_max_flow` `.expect()`s it to mirror the throw.
//!   * `MaxFlowState` is made `pub` (the TS interface was private) so it can name
//!     the public `FixedPointIterationStation<MaxFlowState>` impl without leaking
//!     a private type through a public trait. FLAG: visibility widened vs TS.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::des::general::des_base::fixed_point::{
    ConvergenceReason, FixedPointCore, FixedPointIterationStation, FixedPointOptions,
};
use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};
use crate::des::general::des_base::runner::{
    assert_no_validation_failures, run_iterative_des, IterativeRunOptions,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;

// =============================================================================
// Declarations
// =============================================================================

/// A directed capacitated edge (TS `interface MaxFlowEdge`).
#[derive(Clone, Debug, PartialEq)]
pub struct MaxFlowEdge {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub name: Option<String>,
}

/// A max-flow problem instance (TS `interface MaxFlowProblem`).
#[derive(Clone, Debug)]
pub struct MaxFlowProblem {
    pub num_nodes: usize,
    pub source: usize,
    pub sink: usize,
    pub edges: Vec<MaxFlowEdge>,
}

/// One augmentation trace row (TS `interface MaxFlowTraceEntry`).
#[derive(Clone, Debug, PartialEq)]
pub struct MaxFlowTraceEntry {
    pub iter: usize,
    pub path: Vec<usize>,
    pub bottleneck: f64,
    pub flow_after: f64,
}

/// Solve status (TS string-union `'optimal' | 'infeasible'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxFlowStatus {
    Optimal,
    Infeasible,
}

/// An edge annotated with its realised flow (TS `MaxFlowEdge & {flow}`).
#[derive(Clone, Debug, PartialEq)]
pub struct MaxFlowEdgeFlow {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub name: Option<String>,
    pub flow: f64,
}

/// The min-cut induced by the residual graph (TS inline `minCut` object).
#[derive(Clone, Debug)]
pub struct MinCut {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
    pub cut_edges: Vec<MaxFlowEdgeFlow>,
    pub capacity: f64,
}

/// Full max-flow result (TS `interface MaxFlowResult`).
#[derive(Clone, Debug)]
pub struct MaxFlowResult {
    pub status: MaxFlowStatus,
    pub max_flow: f64,
    pub source: usize,
    pub sink: usize,
    pub num_nodes: usize,
    pub edge_flows: Vec<MaxFlowEdgeFlow>,
    pub min_cut: MinCut,
    pub iterations: usize,
    pub trace: Vec<MaxFlowTraceEntry>,
}

/// A residual-graph edge (TS private `interface ResidualEdge`).
#[derive(Clone, Debug)]
pub struct ResidualEdge {
    to: usize,
    rev: usize,
    cap: f64,
    #[allow(dead_code)]
    original_index: usize,
}

/// Back-reference from an original edge to its forward residual edge (TS private
/// `interface ForwardRef`).
#[derive(Clone, Copy, Debug)]
struct ForwardRef {
    from: usize,
    edge_index: usize,
}

/// A discovered augmenting path (TS private `interface AugmentingPath`).
struct AugmentingPath {
    nodes: Vec<usize>,
    bottleneck: f64,
    parent_node: Vec<isize>,
    parent_edge: Vec<isize>,
}

/// The fixed-point iteration state (TS private `interface MaxFlowState`; made
/// `pub` here — see module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxFlowState {
    pub iter: usize,
    pub flow: f64,
    pub done: bool,
}

const MODEL: &str = "max-flow";

/// Validate a problem instance (TS `validateMaxFlowProblem`, which threw → here a
/// `Result`).
pub fn validate_max_flow_problem(p: &MaxFlowProblem) -> Result<(), PreconditionError> {
    Preconditions::integer_in_range(MODEL, "numNodes", p.num_nodes as f64, 2.0, 1e7)?;
    Preconditions::integer_in_range(
        MODEL,
        "source",
        p.source as f64,
        0.0,
        (p.num_nodes - 1) as f64,
    )?;
    Preconditions::integer_in_range(MODEL, "sink", p.sink as f64, 0.0, (p.num_nodes - 1) as f64)?;
    Preconditions::check(
        MODEL,
        "source != sink",
        "hold",
        p.source != p.sink,
        Some(format!("[{}, {}]", p.source, p.sink)),
    )?;
    Preconditions::non_empty(MODEL, "edges", &p.edges)?;
    for i in 0..p.edges.len() {
        let e = &p.edges[i];
        Preconditions::integer_in_range(
            MODEL,
            &format!("edges[{i}].from"),
            e.from as f64,
            0.0,
            (p.num_nodes - 1) as f64,
        )?;
        Preconditions::integer_in_range(
            MODEL,
            &format!("edges[{i}].to"),
            e.to as f64,
            0.0,
            (p.num_nodes - 1) as f64,
        )?;
        Preconditions::non_negative(MODEL, &format!("edges[{i}].capacity"), e.capacity)?;
    }
    Ok(())
}

// =============================================================================
// MaxFlowStation
// =============================================================================

/// Edmonds-Karp max-flow as a fixed-point DES station (TS `class
/// MaxFlowStation`).
pub struct MaxFlowStation {
    core: StationCore,
    fp: FixedPointCore<MaxFlowState>,
    p: MaxFlowProblem,
    residual: Vec<Vec<ResidualEdge>>,
    forward_refs: Vec<ForwardRef>,
    /// Per-iteration augmentation trace.
    pub trace: Vec<MaxFlowTraceEntry>,
    final_flow: f64,
}

impl MaxFlowStation {
    pub fn new(p: MaxFlowProblem) -> Self {
        validate_max_flow_problem(&p).expect("max-flow: invalid problem instance");
        let e = p.edges.len().max(1);
        let max_iter = (p.num_nodes * e * e + 1).max(1);

        let mut residual: Vec<Vec<ResidualEdge>> = (0..p.num_nodes).map(|_| Vec::new()).collect();
        let mut forward_refs: Vec<ForwardRef> = vec![
            ForwardRef {
                from: 0,
                edge_index: 0
            };
            p.edges.len()
        ];
        for i in 0..p.edges.len() {
            Self::add_residual_edge(&mut residual, &mut forward_refs, &p.edges[i], i);
        }

        let mut st = MaxFlowStation {
            core: StationCore::new(MODEL),
            fp: FixedPointCore::new(FixedPointOptions {
                tol: Some(0.0),
                max_iter: Some(max_iter),
                ..Default::default()
            }),
            p,
            residual,
            forward_refs,
            trace: Vec::new(),
            final_flow: 0.0,
        };
        st.bootstrap();

        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "max-flow.conservation",
                |s| {
                    let st = s.as_any().downcast_ref::<MaxFlowStation>().unwrap();
                    st.conservation_error() <= 1e-8
                },
                Some("flow conserved at every non-terminal node".to_string()),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<MaxFlowStation>().unwrap();
                    format!("max imbalance={:.3e}", st.conservation_error())
                })),
                Some("max-flow-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "max-flow.cut-equals-flow",
                |s| {
                    let st = s.as_any().downcast_ref::<MaxFlowStation>().unwrap();
                    (st.build_result().min_cut.capacity - st.final_flow).abs() <= 1e-8
                },
                Some("min-cut capacity equals max flow".to_string()),
                Some(Box::new(|s| {
                    let st = s.as_any().downcast_ref::<MaxFlowStation>().unwrap();
                    format!(
                        "cut={}, flow={}",
                        st.build_result().min_cut.capacity,
                        st.final_flow
                    )
                })),
                Some("max-flow-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    /// Borrow the residual adjacency (TS `getResidual`).
    pub fn get_residual(&self) -> &Vec<Vec<ResidualEdge>> {
        &self.residual
    }

    pub fn build_result(&self) -> MaxFlowResult {
        let edge_flows: Vec<MaxFlowEdgeFlow> = self
            .p
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let r = &self.forward_refs[i];
                let residual_edge = &self.residual[r.from][r.edge_index];
                MaxFlowEdgeFlow {
                    from: e.from,
                    to: e.to,
                    capacity: e.capacity,
                    name: e.name.clone(),
                    flow: e.capacity - residual_edge.cap,
                }
            })
            .collect();
        let source_side = self.reachable_from_source();
        let source_set: HashSet<usize> = source_side.iter().copied().collect();
        let mut sink_side: Vec<usize> = Vec::new();
        for i in 0..self.p.num_nodes {
            if !source_set.contains(&i) {
                sink_side.push(i);
            }
        }
        let cut_edges: Vec<MaxFlowEdgeFlow> = edge_flows
            .iter()
            .filter(|e| source_set.contains(&e.from) && !source_set.contains(&e.to))
            .cloned()
            .collect();
        let cut_capacity: f64 = cut_edges.iter().map(|e| e.capacity).sum();
        MaxFlowResult {
            status: MaxFlowStatus::Optimal,
            max_flow: self.final_flow,
            source: self.p.source,
            sink: self.p.sink,
            num_nodes: self.p.num_nodes,
            edge_flows,
            min_cut: MinCut {
                source_side,
                sink_side,
                cut_edges,
                capacity: cut_capacity,
            },
            iterations: self.iteration(),
            trace: self.trace.clone(),
        }
    }

    fn add_residual_edge(
        residual: &mut [Vec<ResidualEdge>],
        forward_refs: &mut [ForwardRef],
        e: &MaxFlowEdge,
        original_index: usize,
    ) {
        let len_to = residual[e.to].len();
        let len_from = residual[e.from].len();
        let fwd = ResidualEdge {
            to: e.to,
            rev: len_to,
            cap: e.capacity,
            original_index,
        };
        let rev = ResidualEdge {
            to: e.from,
            rev: len_from,
            cap: 0.0,
            original_index,
        };
        residual[e.from].push(fwd);
        residual[e.to].push(rev);
        forward_refs[original_index] = ForwardRef {
            from: e.from,
            edge_index: residual[e.from].len() - 1,
        };
    }

    fn find_augmenting_path(&self) -> Option<AugmentingPath> {
        let n = self.p.num_nodes;
        let mut parent_node = vec![-1isize; n];
        let mut parent_edge = vec![-1isize; n];
        let mut q: Vec<usize> = vec![self.p.source];
        parent_node[self.p.source] = self.p.source as isize;
        let mut qi = 0;
        while qi < q.len() {
            let u = q[qi];
            qi += 1;
            for ei in 0..self.residual[u].len() {
                let e = &self.residual[u][ei];
                if e.cap <= 1e-12 || parent_node[e.to] != -1 {
                    continue;
                }
                parent_node[e.to] = u as isize;
                parent_edge[e.to] = ei as isize;
                if e.to == self.p.sink {
                    let mut nodes: Vec<usize> = Vec::new();
                    let mut bottleneck = f64::INFINITY;
                    let mut v = self.p.sink;
                    while v != self.p.source {
                        nodes.push(v);
                        let pu = parent_node[v] as usize;
                        let pe_idx = parent_edge[v] as usize;
                        bottleneck = bottleneck.min(self.residual[pu][pe_idx].cap);
                        v = parent_node[v] as usize;
                    }
                    nodes.push(self.p.source);
                    nodes.reverse();
                    return Some(AugmentingPath {
                        nodes,
                        bottleneck,
                        parent_node,
                        parent_edge,
                    });
                }
                q.push(e.to);
            }
        }
        None
    }

    fn reachable_from_source(&self) -> Vec<usize> {
        let mut seen = vec![false; self.p.num_nodes];
        let mut q: Vec<usize> = vec![self.p.source];
        seen[self.p.source] = true;
        let mut qi = 0;
        while qi < q.len() {
            let u = q[qi];
            qi += 1;
            for e in &self.residual[u] {
                if e.cap > 1e-12 && !seen[e.to] {
                    seen[e.to] = true;
                    q.push(e.to);
                }
            }
        }
        (0..seen.len()).filter(|&i| seen[i]).collect()
    }

    fn conservation_error(&self) -> f64 {
        let flows = self.build_result().edge_flows;
        let mut balance = vec![0.0_f64; self.p.num_nodes];
        for e in &flows {
            balance[e.from] -= e.flow;
            balance[e.to] += e.flow;
        }
        let mut err: f64 = 0.0;
        for i in 0..balance.len() {
            if i == self.p.source || i == self.p.sink {
                continue;
            }
            err = err.max(balance[i].abs());
        }
        err
    }
}

impl DESStation for MaxFlowStation {
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
        self.fixed_point_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.fixed_point_has_work()
    }
}

impl FixedPointIterationStation<MaxFlowState> for MaxFlowStation {
    fn fp_core(&self) -> &FixedPointCore<MaxFlowState> {
        &self.fp
    }
    fn fp_core_mut(&mut self) -> &mut FixedPointCore<MaxFlowState> {
        &mut self.fp
    }

    fn initial_state(&self) -> MaxFlowState {
        MaxFlowState {
            iter: 0,
            flow: 0.0,
            done: false,
        }
    }

    fn apply_operator(&mut self, prev: &MaxFlowState) -> MaxFlowState {
        match self.find_augmenting_path() {
            None => {
                self.final_flow = prev.flow;
                MaxFlowState {
                    iter: prev.iter + 1,
                    flow: prev.flow,
                    done: true,
                }
            }
            Some(aug) => {
                let mut v = self.p.sink;
                while v != self.p.source {
                    let u = aug.parent_node[v] as usize;
                    let ei = aug.parent_edge[v] as usize;
                    let (to, rev) = {
                        let e = &self.residual[u][ei];
                        (e.to, e.rev)
                    };
                    self.residual[u][ei].cap -= aug.bottleneck;
                    self.residual[to][rev].cap += aug.bottleneck;
                    v = aug.parent_node[v] as usize;
                }
                let flow = prev.flow + aug.bottleneck;
                self.final_flow = flow;
                self.trace.push(MaxFlowTraceEntry {
                    iter: prev.iter + 1,
                    path: aug.nodes.clone(),
                    bottleneck: aug.bottleneck,
                    flow_after: flow,
                });
                MaxFlowState {
                    iter: prev.iter + 1,
                    flow,
                    done: false,
                }
            }
        }
    }

    fn delta(&self, _prev: &MaxFlowState, next: &MaxFlowState) -> f64 {
        if next.done {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let done = self.current().done;
        if iter > 0 && done {
            self.fp_core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        // super.shouldStop:
        if iter >= self.fp_core().max_iter {
            self.fp_core_mut().convergence_reason = ConvergenceReason::MaxIter;
            return true;
        }
        if iter > 0 && last_delta < self.fp_core().tol {
            self.fp_core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        false
    }
}

// =============================================================================
// Driver + textbook instance
// =============================================================================

/// Solve a max-flow problem by running the fixed-point DES to convergence (TS
/// `solveMaxFlow`).
pub fn solve_max_flow(p: MaxFlowProblem) -> MaxFlowResult {
    let st = Rc::new(RefCell::new(MaxFlowStation::new(p)));
    let summary = run_iterative_des(
        vec![st.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );
    assert_no_validation_failures(&summary, "max-flow")
        .expect("max-flow: post-run validation failed");
    let result = st.borrow().build_result();
    result
}

/// The CLRS textbook max-flow instance (max flow = 23) (TS
/// `buildTextbookMaxFlowProblem`).
pub fn build_textbook_max_flow_problem() -> MaxFlowProblem {
    MaxFlowProblem {
        num_nodes: 6,
        source: 0,
        sink: 5,
        edges: vec![
            MaxFlowEdge {
                from: 0,
                to: 1,
                capacity: 16.0,
                name: Some("s-v1".to_string()),
            },
            MaxFlowEdge {
                from: 0,
                to: 2,
                capacity: 13.0,
                name: Some("s-v2".to_string()),
            },
            MaxFlowEdge {
                from: 1,
                to: 2,
                capacity: 10.0,
                name: Some("v1-v2".to_string()),
            },
            MaxFlowEdge {
                from: 2,
                to: 1,
                capacity: 4.0,
                name: Some("v2-v1".to_string()),
            },
            MaxFlowEdge {
                from: 1,
                to: 3,
                capacity: 12.0,
                name: Some("v1-v3".to_string()),
            },
            MaxFlowEdge {
                from: 3,
                to: 2,
                capacity: 9.0,
                name: Some("v3-v2".to_string()),
            },
            MaxFlowEdge {
                from: 2,
                to: 4,
                capacity: 14.0,
                name: Some("v2-v4".to_string()),
            },
            MaxFlowEdge {
                from: 4,
                to: 3,
                capacity: 7.0,
                name: Some("v4-v3".to_string()),
            },
            MaxFlowEdge {
                from: 3,
                to: 5,
                capacity: 20.0,
                name: Some("v3-t".to_string()),
            },
            MaxFlowEdge {
                from: 4,
                to: 5,
                capacity: 4.0,
                name: Some("v4-t".to_string()),
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the max-flow DES station.
    //!
    //! Each case solves a small graph whose analytic max-flow value is known and
    //! checks that the residual min-cut capacity equals the realised flow (the
    //! max-flow / min-cut duality the station also asserts internally).

    use super::*;

    #[test]
    fn textbook_max_flow_is_23() {
        let res = solve_max_flow(build_textbook_max_flow_problem());
        assert_eq!(res.status, MaxFlowStatus::Optimal);
        assert!(
            (res.max_flow - 23.0).abs() < 1e-9,
            "max flow = {}",
            res.max_flow
        );
        // max-flow = min-cut.
        assert!(
            (res.min_cut.capacity - 23.0).abs() < 1e-9,
            "cut = {}",
            res.min_cut.capacity
        );
    }

    #[test]
    fn diamond_graph_bottleneck() {
        // 0→1(3), 0→2(2), 1→3(2), 2→3(3), 1→2(1). Source 0, sink 3.
        // Min-cut at the source = 3 + 2 = 5 ⇒ max flow 5.
        let p = MaxFlowProblem {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 3.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 0,
                    to: 2,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 3,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 2,
                    to: 3,
                    capacity: 3.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 2,
                    capacity: 1.0,
                    name: None,
                },
            ],
        };
        let res = solve_max_flow(p);
        assert!(
            (res.max_flow - 5.0).abs() < 1e-9,
            "max flow = {}",
            res.max_flow
        );
        assert!((res.min_cut.capacity - res.max_flow).abs() < 1e-9);
    }

    #[test]
    fn linear_chain_is_limited_by_narrowest_edge() {
        // 0→1(10), 1→2(4), 2→3(10). Source 0, sink 3 ⇒ max flow 4.
        let p = MaxFlowProblem {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 10.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 2,
                    capacity: 4.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 2,
                    to: 3,
                    capacity: 10.0,
                    name: None,
                },
            ],
        };
        let res = solve_max_flow(p);
        assert!(
            (res.max_flow - 4.0).abs() < 1e-9,
            "max flow = {}",
            res.max_flow
        );
        assert!((res.min_cut.capacity - 4.0).abs() < 1e-9);
        // The narrow edge 1→2 carries the full flow.
        let mid = res
            .edge_flows
            .iter()
            .find(|e| e.from == 1 && e.to == 2)
            .unwrap();
        assert!((mid.flow - 4.0).abs() < 1e-9);
    }
}
