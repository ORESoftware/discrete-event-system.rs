//! Minimum spanning tree models.
//!
//! MST is a foundational network-optimization primitive. The native solvers
//! here provide Kruskal and Prim variants over validation-scale undirected
//! graphs; the external bridge cross-checks the same input against a Python
//! Kruskal reference and an OR-Tools CP-SAT connectivity-flow formulation.

use std::collections::{HashMap, HashSet};

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "minimum-spanning-tree";
const EPS: f64 = 1e-8;

#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningTreeEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningTreeProblem {
    pub vertices: Vec<String>,
    pub edges: Vec<MinimumSpanningTreeEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimumSpanningTreeStatus {
    Optimal,
    Infeasible,
}

impl MinimumSpanningTreeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MinimumSpanningTreeStatus::Optimal => "optimal",
            MinimumSpanningTreeStatus::Infeasible => "infeasible",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningTreeSolution {
    pub status: MinimumSpanningTreeStatus,
    pub selected_edge_indices: Vec<usize>,
    pub selected_edge_ids: Vec<String>,
    pub total_weight: Option<f64>,
    pub message: String,
}

pub fn build_sample_minimum_spanning_tree_problem() -> MinimumSpanningTreeProblem {
    MinimumSpanningTreeProblem {
        vertices: ["A", "B", "C", "D", "E"]
            .iter()
            .map(|id| id.to_string())
            .collect(),
        edges: vec![
            edge("AB", "A", "B", 1.0),
            edge("AC", "A", "C", 4.0),
            edge("AE", "A", "E", 7.0),
            edge("BC", "B", "C", 2.0),
            edge("BD", "B", "D", 5.0),
            edge("CD", "C", "D", 1.0),
            edge("CE", "C", "E", 3.0),
            edge("DE", "D", "E", 2.0),
        ],
    }
}

fn edge(id: &str, from: &str, to: &str, weight: f64) -> MinimumSpanningTreeEdge {
    MinimumSpanningTreeEdge {
        id: id.to_string(),
        from: from.to_string(),
        to: to.to_string(),
        weight,
    }
}

pub fn validate_minimum_spanning_tree_problem(
    p: &MinimumSpanningTreeProblem,
) -> Result<(), PreconditionError> {
    Preconditions::non_empty(MODEL, "vertices", &p.vertices)?;
    let mut vertex_seen = HashSet::new();
    let mut vertex_index = HashMap::new();
    for (idx, vertex) in p.vertices.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("vertices[{idx}]"),
            "be non-empty",
            !vertex.trim().is_empty(),
            Some(vertex.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("vertices[{idx}]"),
            "be unique",
            vertex_seen.insert(vertex.clone()),
            Some(vertex.clone()),
        )?;
        vertex_index.insert(vertex.clone(), idx);
    }

    let mut edge_ids = HashSet::new();
    let mut edge_keys = HashSet::new();
    for (edge_idx, edge) in p.edges.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}].id"),
            "be non-empty",
            !edge.id.trim().is_empty(),
            Some(edge.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}].id"),
            "be unique",
            edge_ids.insert(edge.id.clone()),
            Some(edge.id.clone()),
        )?;
        let Some(&from_idx) = vertex_index.get(&edge.from) else {
            return Err(PreconditionError::new(
                MODEL,
                &format!("edges[{edge_idx}].from"),
                "belong to vertices",
                Some(edge.from.clone()),
            ));
        };
        let Some(&to_idx) = vertex_index.get(&edge.to) else {
            return Err(PreconditionError::new(
                MODEL,
                &format!("edges[{edge_idx}].to"),
                "belong to vertices",
                Some(edge.to.clone()),
            ));
        };
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}]"),
            "not be a self-loop",
            from_idx != to_idx,
            Some(format!("{}-{}", edge.from, edge.to)),
        )?;
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}].weight"),
            "be finite",
            edge.weight.is_finite(),
            Some(edge.weight.to_string()),
        )?;
        let key = if from_idx < to_idx {
            (from_idx, to_idx)
        } else {
            (to_idx, from_idx)
        };
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}]"),
            "be unique as an undirected edge",
            edge_keys.insert(key),
            Some(format!("{}-{}", edge.from, edge.to)),
        )?;
    }
    Ok(())
}

pub fn solve_minimum_spanning_tree_kruskal(
    p: &MinimumSpanningTreeProblem,
) -> MinimumSpanningTreeSolution {
    validate_minimum_spanning_tree_problem(p)
        .expect("minimum-spanning-tree: invalid problem instance");
    if p.vertices.len() == 1 {
        return build_solution(p, Vec::new(), "single-vertex MST");
    }
    let vertex_index = vertex_index(p);
    let mut order = (0..p.edges.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        p.edges[a]
            .weight
            .total_cmp(&p.edges[b].weight)
            .then_with(|| p.edges[a].id.cmp(&p.edges[b].id))
    });
    let mut dsu = DisjointSet::new(p.vertices.len());
    let mut selected = Vec::new();
    for edge_idx in order {
        let edge = &p.edges[edge_idx];
        let a = vertex_index[&edge.from];
        let b = vertex_index[&edge.to];
        if dsu.union(a, b) {
            selected.push(edge_idx);
            if selected.len() + 1 == p.vertices.len() {
                break;
            }
        }
    }
    if selected.len() + 1 == p.vertices.len() {
        build_solution(p, selected, "Kruskal minimum spanning tree")
    } else {
        infeasible_solution("graph is disconnected")
    }
}

pub fn solve_minimum_spanning_tree_prim(
    p: &MinimumSpanningTreeProblem,
) -> MinimumSpanningTreeSolution {
    validate_minimum_spanning_tree_problem(p)
        .expect("minimum-spanning-tree: invalid problem instance");
    if p.vertices.len() == 1 {
        return build_solution(p, Vec::new(), "single-vertex MST");
    }
    let vertex_index = vertex_index(p);
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); p.vertices.len()];
    for (edge_idx, edge) in p.edges.iter().enumerate() {
        let a = vertex_index[&edge.from];
        let b = vertex_index[&edge.to];
        adjacency[a].push((b, edge_idx));
        adjacency[b].push((a, edge_idx));
    }
    let mut in_tree = vec![false; p.vertices.len()];
    in_tree[0] = true;
    let mut selected = Vec::new();
    while selected.len() + 1 < p.vertices.len() {
        let mut best: Option<(usize, usize, f64, String)> = None;
        for (from, neighbors) in adjacency.iter().enumerate() {
            if !in_tree[from] {
                continue;
            }
            for &(to, edge_idx) in neighbors {
                if in_tree[to] {
                    continue;
                }
                let edge = &p.edges[edge_idx];
                let candidate = (to, edge_idx, edge.weight, edge.id.clone());
                if best.as_ref().is_none_or(|(_, _, best_weight, best_id)| {
                    edge.weight < *best_weight - EPS
                        || ((edge.weight - *best_weight).abs() <= EPS && edge.id < *best_id)
                }) {
                    best = Some(candidate);
                }
            }
        }
        let Some((to, edge_idx, _, _)) = best else {
            return infeasible_solution("graph is disconnected");
        };
        in_tree[to] = true;
        selected.push(edge_idx);
    }
    build_solution(p, selected, "Prim minimum spanning tree")
}

pub fn minimum_spanning_tree_solution_feasible(
    p: &MinimumSpanningTreeProblem,
    solution: &MinimumSpanningTreeSolution,
) -> bool {
    if validate_minimum_spanning_tree_problem(p).is_err() {
        return false;
    }
    if p.vertices.len() == 1 {
        return solution.selected_edge_indices.is_empty() && solution.total_weight == Some(0.0);
    }
    if solution.selected_edge_indices.len() + 1 != p.vertices.len()
        || solution.selected_edge_indices.len() != solution.selected_edge_ids.len()
    {
        return false;
    }
    let vertex_index = vertex_index(p);
    let mut seen_edges = HashSet::new();
    let mut dsu = DisjointSet::new(p.vertices.len());
    let mut total = 0.0;
    for (&edge_idx, edge_id) in solution
        .selected_edge_indices
        .iter()
        .zip(&solution.selected_edge_ids)
    {
        let Some(edge) = p.edges.get(edge_idx) else {
            return false;
        };
        if edge.id != *edge_id || !seen_edges.insert(edge_idx) {
            return false;
        }
        let a = vertex_index[&edge.from];
        let b = vertex_index[&edge.to];
        if !dsu.union(a, b) {
            return false;
        }
        total += edge.weight;
    }
    let connected = (1..p.vertices.len()).all(|idx| dsu.find(idx) == dsu.find(0));
    connected
        && solution
            .total_weight
            .is_some_and(|reported| (reported - total).abs() <= EPS * 1.0_f64.max(total.abs()))
}

fn vertex_index(p: &MinimumSpanningTreeProblem) -> HashMap<String, usize> {
    p.vertices
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx))
        .collect()
}

fn build_solution(
    p: &MinimumSpanningTreeProblem,
    mut selected_edge_indices: Vec<usize>,
    message: &str,
) -> MinimumSpanningTreeSolution {
    selected_edge_indices.sort_unstable();
    let total_weight = selected_edge_indices
        .iter()
        .map(|&idx| p.edges[idx].weight)
        .sum::<f64>();
    let selected_edge_ids = selected_edge_indices
        .iter()
        .map(|&idx| p.edges[idx].id.clone())
        .collect();
    MinimumSpanningTreeSolution {
        status: MinimumSpanningTreeStatus::Optimal,
        selected_edge_indices,
        selected_edge_ids,
        total_weight: Some(total_weight),
        message: message.to_string(),
    }
}

fn infeasible_solution(message: &str) -> MinimumSpanningTreeSolution {
    MinimumSpanningTreeSolution {
        status: MinimumSpanningTreeStatus::Infeasible,
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        total_weight: None,
        message: message.to_string(),
    }
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        DisjointSet {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kruskal_finds_sample_mst() {
        let p = build_sample_minimum_spanning_tree_problem();
        let solution = solve_minimum_spanning_tree_kruskal(&p);
        assert_eq!(solution.status, MinimumSpanningTreeStatus::Optimal);
        assert_eq!(solution.total_weight, Some(6.0));
        assert!(minimum_spanning_tree_solution_feasible(&p, &solution));
    }

    #[test]
    fn prim_matches_kruskal_on_sample() {
        let p = build_sample_minimum_spanning_tree_problem();
        let kruskal = solve_minimum_spanning_tree_kruskal(&p);
        let prim = solve_minimum_spanning_tree_prim(&p);
        assert_eq!(prim.status, MinimumSpanningTreeStatus::Optimal);
        assert_eq!(prim.total_weight, kruskal.total_weight);
        assert!(minimum_spanning_tree_solution_feasible(&p, &prim));
    }
}
