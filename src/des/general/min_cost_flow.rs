//! Minimum-cost flow with supplies/demands and lower/upper arc bounds.
//!
//! This closes a network-optimisation gap next to the existing max-flow solver:
//! many external suites expose min-cost flow / transportation as a first-class
//! primitive, and it is often much cheaper to solve directly than by building a
//! general LP/MIP. The implementation is intentionally dependency-free and
//! small-model friendly: lower bounds are normalised into node balances, then a
//! successive shortest augmenting path algorithm runs on the residual network.

use crate::des::general::lp::{LPProblem, Sense};

const EPS: f64 = 1e-9;

/// Directed arc with lower/upper capacity and linear unit cost.
#[derive(Clone, Debug, PartialEq)]
pub struct MinCostFlowArc {
    pub from: usize,
    pub to: usize,
    /// Required minimum flow on this arc.
    pub lower_bound: f64,
    /// Maximum permitted flow on this arc.
    pub capacity: f64,
    /// Cost per unit of flow.
    pub cost: f64,
    pub name: Option<String>,
}

/// Node supplies use the standard network-flow sign convention:
/// positive = net outflow supplied, negative = net inflow demanded.
#[derive(Clone, Debug, PartialEq)]
pub struct MinCostFlowProblem {
    pub num_nodes: usize,
    pub supplies: Vec<f64>,
    pub arcs: Vec<MinCostFlowArc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinCostFlowStatus {
    Optimal,
    Infeasible,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinCostFlowArcResult {
    pub from: usize,
    pub to: usize,
    pub lower_bound: f64,
    pub capacity: f64,
    pub cost: f64,
    pub flow: f64,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinCostFlowTraceEntry {
    pub iter: usize,
    pub path: Vec<usize>,
    pub bottleneck: f64,
    pub unit_cost: f64,
    pub total_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinCostFlowResult {
    pub status: MinCostFlowStatus,
    pub total_cost: f64,
    pub arc_flows: Vec<MinCostFlowArcResult>,
    pub node_balance: Vec<f64>,
    pub iterations: usize,
    pub trace: Vec<MinCostFlowTraceEntry>,
    pub message: Option<String>,
}

#[derive(Clone, Debug)]
struct ResidualArc {
    to: usize,
    rev: usize,
    cap: f64,
    cost: f64,
    original: Option<usize>,
    direction: f64,
}

fn validate_problem(p: &MinCostFlowProblem) -> Result<(), String> {
    if p.num_nodes == 0 {
        return Err("min-cost-flow: num_nodes must be positive".to_string());
    }
    if p.supplies.len() != p.num_nodes {
        return Err(format!(
            "min-cost-flow: supplies length {} != num_nodes {}",
            p.supplies.len(),
            p.num_nodes
        ));
    }
    if p.supplies.iter().any(|v| !v.is_finite()) {
        return Err("min-cost-flow: supplies must be finite".to_string());
    }
    let total_supply: f64 = p.supplies.iter().sum();
    if total_supply.abs() > 1e-7 {
        return Err(format!(
            "min-cost-flow: supplies must sum to zero, got {total_supply:.3e}"
        ));
    }
    if p.arcs.is_empty() {
        return Err("min-cost-flow: arcs must be non-empty".to_string());
    }
    for (i, arc) in p.arcs.iter().enumerate() {
        if arc.from >= p.num_nodes || arc.to >= p.num_nodes {
            return Err(format!("min-cost-flow: arc {i} endpoint out of range"));
        }
        if arc.from == arc.to {
            return Err(format!("min-cost-flow: arc {i} is a self-loop"));
        }
        if !arc.lower_bound.is_finite() || !arc.capacity.is_finite() || !arc.cost.is_finite() {
            return Err(format!("min-cost-flow: arc {i} fields must be finite"));
        }
        if arc.lower_bound < -EPS {
            return Err(format!("min-cost-flow: arc {i} lower_bound is negative"));
        }
        if arc.capacity + EPS < arc.lower_bound {
            return Err(format!(
                "min-cost-flow: arc {i} capacity {} < lower_bound {}",
                arc.capacity, arc.lower_bound
            ));
        }
    }
    Ok(())
}

fn add_residual_arc(
    residual: &mut [Vec<ResidualArc>],
    from: usize,
    to: usize,
    cap: f64,
    cost: f64,
    original: Option<usize>,
) {
    let fwd_index = residual[from].len();
    let rev_index = residual[to].len();
    residual[from].push(ResidualArc {
        to,
        rev: rev_index,
        cap,
        cost,
        original,
        direction: 1.0,
    });
    residual[to].push(ResidualArc {
        to: from,
        rev: fwd_index,
        cap: 0.0,
        cost: -cost,
        original,
        direction: -1.0,
    });
}

fn shortest_path(
    residual: &[Vec<ResidualArc>],
    source: usize,
    sink: usize,
) -> Option<(Vec<usize>, Vec<usize>, f64)> {
    let n = residual.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev_node = vec![usize::MAX; n];
    let mut prev_edge = vec![usize::MAX; n];
    dist[source] = 0.0;
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for u in 0..n {
            if !dist[u].is_finite() {
                continue;
            }
            for (ei, arc) in residual[u].iter().enumerate() {
                if arc.cap <= EPS {
                    continue;
                }
                let nd = dist[u] + arc.cost;
                if nd < dist[arc.to] - EPS {
                    dist[arc.to] = nd;
                    prev_node[arc.to] = u;
                    prev_edge[arc.to] = ei;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if !dist[sink].is_finite() {
        return None;
    }
    Some((prev_node, prev_edge, dist[sink]))
}

fn path_nodes(prev_node: &[usize], source: usize, sink: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut v = sink;
    out.push(v);
    while v != source {
        v = prev_node[v];
        out.push(v);
    }
    out.reverse();
    out
}

fn compute_balances(p: &MinCostFlowProblem, flows: &[f64]) -> Vec<f64> {
    let mut balance = vec![0.0; p.num_nodes];
    for (arc, &flow) in p.arcs.iter().zip(flows) {
        balance[arc.from] += flow;
        balance[arc.to] -= flow;
    }
    balance
}

fn arc_results(p: &MinCostFlowProblem, flows: &[f64]) -> Vec<MinCostFlowArcResult> {
    p.arcs
        .iter()
        .zip(flows)
        .map(|(arc, &flow)| MinCostFlowArcResult {
            from: arc.from,
            to: arc.to,
            lower_bound: arc.lower_bound,
            capacity: arc.capacity,
            cost: arc.cost,
            flow,
            name: arc.name.clone(),
        })
        .collect()
}

/// Solve a feasible balanced min-cost-flow problem.
pub fn solve_min_cost_flow(p: MinCostFlowProblem) -> MinCostFlowResult {
    if let Err(message) = validate_problem(&p) {
        panic!("{message}");
    }

    let supersource = p.num_nodes;
    let supersink = p.num_nodes + 1;
    let mut residual: Vec<Vec<ResidualArc>> = (0..p.num_nodes + 2).map(|_| Vec::new()).collect();
    let mut adjusted_supply = p.supplies.clone();
    let mut flows: Vec<f64> = p.arcs.iter().map(|arc| arc.lower_bound).collect();
    let mut total_cost = 0.0;

    for (i, arc) in p.arcs.iter().enumerate() {
        adjusted_supply[arc.from] -= arc.lower_bound;
        adjusted_supply[arc.to] += arc.lower_bound;
        total_cost += arc.lower_bound * arc.cost;
        add_residual_arc(
            &mut residual,
            arc.from,
            arc.to,
            arc.capacity - arc.lower_bound,
            arc.cost,
            Some(i),
        );
    }

    let mut required = 0.0;
    for (v, &supply) in adjusted_supply.iter().enumerate() {
        if supply > EPS {
            add_residual_arc(&mut residual, supersource, v, supply, 0.0, None);
            required += supply;
        } else if supply < -EPS {
            add_residual_arc(&mut residual, v, supersink, -supply, 0.0, None);
        }
    }

    let mut sent = 0.0;
    let mut trace = Vec::new();
    while sent < required - EPS {
        let Some((prev_node, prev_edge, unit_cost)) =
            shortest_path(&residual, supersource, supersink)
        else {
            return MinCostFlowResult {
                status: MinCostFlowStatus::Infeasible,
                total_cost: f64::NAN,
                arc_flows: arc_results(&p, &flows),
                node_balance: compute_balances(&p, &flows),
                iterations: trace.len(),
                trace,
                message: Some("not enough residual capacity to satisfy demands".to_string()),
            };
        };

        let mut bottleneck = required - sent;
        let mut v = supersink;
        while v != supersource {
            let u = prev_node[v];
            let ei = prev_edge[v];
            bottleneck = bottleneck.min(residual[u][ei].cap);
            v = u;
        }

        v = supersink;
        while v != supersource {
            let u = prev_node[v];
            let ei = prev_edge[v];
            let to = residual[u][ei].to;
            let rev = residual[u][ei].rev;
            let original = residual[u][ei].original;
            let direction = residual[u][ei].direction;
            residual[u][ei].cap -= bottleneck;
            residual[to][rev].cap += bottleneck;
            if let Some(original) = original {
                flows[original] += direction * bottleneck;
            }
            v = u;
        }

        sent += bottleneck;
        total_cost += bottleneck * unit_cost;
        trace.push(MinCostFlowTraceEntry {
            iter: trace.len(),
            path: path_nodes(&prev_node, supersource, supersink),
            bottleneck,
            unit_cost,
            total_cost,
        });
    }

    MinCostFlowResult {
        status: MinCostFlowStatus::Optimal,
        total_cost,
        arc_flows: arc_results(&p, &flows),
        node_balance: compute_balances(&p, &flows),
        iterations: trace.len(),
        trace,
        message: Some("successive shortest augmenting path".to_string()),
    }
}

/// Build the equivalent LP relaxation:
///
/// ```text
/// min c^T x
/// s.t. outflow(v) - inflow(v) = supply(v)   for all but one node
///      lower <= x <= capacity
/// ```
///
/// One balance row is omitted because the supplied problem is balanced and the
/// full node-arc incidence matrix has rank `num_nodes - 1`.
pub fn min_cost_flow_to_lp(p: &MinCostFlowProblem) -> LPProblem {
    if let Err(message) = validate_problem(p) {
        panic!("{message}");
    }
    let n = p.arcs.len();
    let balance_rows = p.num_nodes.saturating_sub(1);
    let mut a_eq = vec![vec![0.0; n]; balance_rows];
    for (j, arc) in p.arcs.iter().enumerate() {
        if arc.from < balance_rows {
            a_eq[arc.from][j] += 1.0;
        }
        if arc.to < balance_rows {
            a_eq[arc.to][j] -= 1.0;
        }
    }
    LPProblem {
        sense: Sense::Min,
        c: p.arcs.iter().map(|arc| arc.cost).collect(),
        a_eq: Some(a_eq),
        b_eq: Some(p.supplies.iter().take(balance_rows).copied().collect()),
        lb: Some(p.arcs.iter().map(|arc| Some(arc.lower_bound)).collect()),
        ub: Some(p.arcs.iter().map(|arc| Some(arc.capacity)).collect()),
        var_names: Some(
            p.arcs
                .iter()
                .enumerate()
                .map(|(i, arc)| arc.name.clone().unwrap_or_else(|| format!("x{i}")))
                .collect(),
        ),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transportation_problem() -> MinCostFlowProblem {
        MinCostFlowProblem {
            num_nodes: 4,
            supplies: vec![5.0, 7.0, -6.0, -6.0],
            arcs: vec![
                MinCostFlowArc {
                    from: 0,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 2.0,
                    name: Some("s0_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 0,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 4.0,
                    name: Some("s0_d1".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 6.0,
                    cost: 5.0,
                    name: Some("s1_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 8.0,
                    cost: 1.0,
                    name: Some("s1_d1".to_string()),
                },
            ],
        }
    }

    #[test]
    fn solves_transportation_problem() {
        let result = solve_min_cost_flow(transportation_problem());
        assert_eq!(result.status, MinCostFlowStatus::Optimal);
        assert!((result.total_cost - 21.0).abs() < 1e-9, "{result:?}");
        assert!(result
            .node_balance
            .iter()
            .zip([5.0, 7.0, -6.0, -6.0])
            .all(|(a, b)| (a - b).abs() < 1e-9));
    }
}
