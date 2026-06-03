//! Weighted maximum independent-set models.
//!
//! A weighted independent set selects mutually non-adjacent vertices with
//! maximum total weight. This is the conflict-graph form of many set-packing,
//! allocation, and scheduling choices, and it is a compact MIP/CP-SAT
//! benchmark. The exact solver here is validation-scale branch-and-bound; the
//! greedy solver is a deterministic descending-weight baseline.

use std::collections::{HashMap, HashSet};

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "weighted-independent-set";
const EPS: f64 = 1e-9;
const MAX_EXACT_VERTICES: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedIndependentSetVertex {
    pub id: String,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedIndependentSetProblem {
    pub vertices: Vec<WeightedIndependentSetVertex>,
    /// Undirected conflict edges as vertex-id pairs.
    pub edges: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightedIndependentSetStatus {
    Optimal,
    Feasible,
    Unsupported,
}

impl WeightedIndependentSetStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightedIndependentSetStatus::Optimal => "optimal",
            WeightedIndependentSetStatus::Feasible => "feasible",
            WeightedIndependentSetStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedIndependentSetSolution {
    pub status: WeightedIndependentSetStatus,
    pub selected_vertex_indices: Vec<usize>,
    pub selected_vertex_ids: Vec<String>,
    pub total_weight: f64,
    pub upper_bound: Option<f64>,
    pub message: String,
}

#[derive(Clone, Debug)]
struct SearchVertex {
    index: usize,
    weight: f64,
}

pub fn build_sample_weighted_independent_set_problem() -> WeightedIndependentSetProblem {
    WeightedIndependentSetProblem {
        vertices: [
            ("A", 8.0),
            ("B", 7.0),
            ("C", 6.0),
            ("D", 6.0),
            ("E", 5.0),
            ("F", 4.0),
            ("G", 3.0),
        ]
        .iter()
        .map(|(id, weight)| WeightedIndependentSetVertex {
            id: id.to_string(),
            weight: *weight,
        })
        .collect(),
        edges: [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "E"),
            ("C", "D"),
            ("C", "F"),
            ("D", "E"),
            ("D", "F"),
            ("E", "F"),
            ("E", "G"),
            ("F", "G"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect(),
    }
}

pub fn validate_weighted_independent_set_problem(
    p: &WeightedIndependentSetProblem,
) -> Result<(), PreconditionError> {
    Preconditions::non_empty(MODEL, "vertices", &p.vertices)?;
    let mut seen = HashSet::new();
    let mut vertex_index = HashMap::new();
    for (idx, vertex) in p.vertices.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("vertices[{idx}].id"),
            "be non-empty",
            !vertex.id.trim().is_empty(),
            Some(vertex.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("vertices[{idx}].id"),
            "be unique",
            seen.insert(vertex.id.clone()),
            Some(vertex.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("vertices[{idx}].weight"),
            "be finite and non-negative",
            vertex.weight.is_finite() && vertex.weight >= 0.0,
            Some(vertex.weight.to_string()),
        )?;
        vertex_index.insert(vertex.id.clone(), idx);
    }

    let mut edge_seen = HashSet::new();
    for (edge_idx, (a, b)) in p.edges.iter().enumerate() {
        let Some(&ai) = vertex_index.get(a) else {
            return Err(PreconditionError::new(
                MODEL,
                &format!("edges[{edge_idx}].0"),
                "belong to vertices",
                Some(a.clone()),
            ));
        };
        let Some(&bi) = vertex_index.get(b) else {
            return Err(PreconditionError::new(
                MODEL,
                &format!("edges[{edge_idx}].1"),
                "belong to vertices",
                Some(b.clone()),
            ));
        };
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}]"),
            "not be a self-loop",
            ai != bi,
            Some(format!("{a}-{b}")),
        )?;
        let key = if ai < bi { (ai, bi) } else { (bi, ai) };
        Preconditions::check(
            MODEL,
            &format!("edges[{edge_idx}]"),
            "be unique as an undirected edge",
            edge_seen.insert(key),
            Some(format!("{a}-{b}")),
        )?;
    }
    Ok(())
}

pub fn solve_weighted_independent_set_greedy(
    p: &WeightedIndependentSetProblem,
) -> WeightedIndependentSetSolution {
    validate_weighted_independent_set_problem(p)
        .expect("weighted-independent-set: invalid problem instance");
    let adjacency = adjacency_matrix(p);
    let mut selected = Vec::new();
    for vertex in sorted_vertices(p) {
        if compatible_with_selected(&adjacency, vertex.index, &selected) {
            selected.push(vertex.index);
        }
    }
    build_solution(
        p,
        WeightedIndependentSetStatus::Feasible,
        selected,
        None,
        "greedy descending-weight independent set",
    )
}

pub fn solve_weighted_independent_set_exact(
    p: &WeightedIndependentSetProblem,
) -> WeightedIndependentSetSolution {
    validate_weighted_independent_set_problem(p)
        .expect("weighted-independent-set: invalid problem instance");
    if p.vertices.len() > MAX_EXACT_VERTICES {
        return WeightedIndependentSetSolution {
            status: WeightedIndependentSetStatus::Unsupported,
            selected_vertex_indices: Vec::new(),
            selected_vertex_ids: Vec::new(),
            total_weight: 0.0,
            upper_bound: None,
            message: format!(
                "exact weighted independent set only practical for <= {MAX_EXACT_VERTICES} vertices, got {}",
                p.vertices.len()
            ),
        };
    }

    let adjacency = adjacency_matrix(p);
    let order = sorted_vertices(p);
    let mut suffix_weight = vec![0.0; order.len() + 1];
    for idx in (0..order.len()).rev() {
        suffix_weight[idx] = suffix_weight[idx + 1] + order[idx].weight;
    }
    let greedy = solve_weighted_independent_set_greedy(p);
    let mut best_indices = greedy.selected_vertex_indices.clone();
    let mut best_weight = greedy.total_weight;
    let mut current = Vec::new();
    exact_search(
        p,
        &adjacency,
        &order,
        &suffix_weight,
        0,
        0.0,
        &mut current,
        &mut best_indices,
        &mut best_weight,
    );
    build_solution(
        p,
        WeightedIndependentSetStatus::Optimal,
        best_indices,
        Some(suffix_weight[0]),
        "exact branch-and-bound weighted independent set",
    )
}

pub fn weighted_independent_set_solution_feasible(
    p: &WeightedIndependentSetProblem,
    solution: &WeightedIndependentSetSolution,
) -> bool {
    if validate_weighted_independent_set_problem(p).is_err()
        || solution.selected_vertex_indices.len() != solution.selected_vertex_ids.len()
    {
        return false;
    }
    let adjacency = adjacency_matrix(p);
    let mut seen = HashSet::new();
    let mut total_weight = 0.0;
    for (&idx, id) in solution
        .selected_vertex_indices
        .iter()
        .zip(&solution.selected_vertex_ids)
    {
        let Some(vertex) = p.vertices.get(idx) else {
            return false;
        };
        if vertex.id != *id || !seen.insert(idx) {
            return false;
        }
        total_weight += vertex.weight;
    }
    for (pos, &a) in solution.selected_vertex_indices.iter().enumerate() {
        for &b in &solution.selected_vertex_indices[pos + 1..] {
            if adjacency[a][b] {
                return false;
            }
        }
    }
    (total_weight - solution.total_weight).abs() <= 1e-8 * 1.0_f64.max(total_weight.abs())
}

fn sorted_vertices(p: &WeightedIndependentSetProblem) -> Vec<SearchVertex> {
    let mut vertices = p
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| SearchVertex {
            index,
            weight: vertex.weight,
        })
        .collect::<Vec<_>>();
    vertices.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| p.vertices[a.index].id.cmp(&p.vertices[b.index].id))
    });
    vertices
}

fn adjacency_matrix(p: &WeightedIndependentSetProblem) -> Vec<Vec<bool>> {
    let vertex_index = p
        .vertices
        .iter()
        .enumerate()
        .map(|(idx, vertex)| (vertex.id.clone(), idx))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![vec![false; p.vertices.len()]; p.vertices.len()];
    for (a, b) in &p.edges {
        let ai = vertex_index[a];
        let bi = vertex_index[b];
        adjacency[ai][bi] = true;
        adjacency[bi][ai] = true;
    }
    adjacency
}

fn compatible_with_selected(adjacency: &[Vec<bool>], vertex: usize, selected: &[usize]) -> bool {
    selected.iter().all(|&other| !adjacency[vertex][other])
}

#[allow(clippy::too_many_arguments)]
fn exact_search(
    p: &WeightedIndependentSetProblem,
    adjacency: &[Vec<bool>],
    order: &[SearchVertex],
    suffix_weight: &[f64],
    pos: usize,
    current_weight: f64,
    current: &mut Vec<usize>,
    best_indices: &mut Vec<usize>,
    best_weight: &mut f64,
) {
    if pos == order.len() {
        if candidate_better(p, current_weight, current, *best_weight, best_indices) {
            *best_indices = current.clone();
            *best_weight = current_weight;
        }
        return;
    }
    if current_weight + suffix_weight[pos] + EPS < *best_weight {
        return;
    }

    let vertex = &order[pos];
    if compatible_with_selected(adjacency, vertex.index, current) {
        current.push(vertex.index);
        exact_search(
            p,
            adjacency,
            order,
            suffix_weight,
            pos + 1,
            current_weight + vertex.weight,
            current,
            best_indices,
            best_weight,
        );
        current.pop();
    }
    exact_search(
        p,
        adjacency,
        order,
        suffix_weight,
        pos + 1,
        current_weight,
        current,
        best_indices,
        best_weight,
    );
}

fn candidate_better(
    p: &WeightedIndependentSetProblem,
    weight: f64,
    indices: &[usize],
    best_weight: f64,
    best_indices: &[usize],
) -> bool {
    if weight > best_weight + EPS {
        return true;
    }
    if (weight - best_weight).abs() <= EPS && indices.len() < best_indices.len() {
        return true;
    }
    if (weight - best_weight).abs() <= EPS && indices.len() == best_indices.len() {
        let mut lhs = indices
            .iter()
            .map(|&idx| p.vertices[idx].id.clone())
            .collect::<Vec<_>>();
        let mut rhs = best_indices
            .iter()
            .map(|&idx| p.vertices[idx].id.clone())
            .collect::<Vec<_>>();
        lhs.sort();
        rhs.sort();
        return lhs < rhs;
    }
    false
}

fn build_solution(
    p: &WeightedIndependentSetProblem,
    status: WeightedIndependentSetStatus,
    mut selected: Vec<usize>,
    upper_bound: Option<f64>,
    message: impl Into<String>,
) -> WeightedIndependentSetSolution {
    selected.sort_unstable();
    let selected_vertex_ids = selected
        .iter()
        .map(|&idx| p.vertices[idx].id.clone())
        .collect::<Vec<_>>();
    let total_weight = selected
        .iter()
        .map(|&idx| p.vertices[idx].weight)
        .sum::<f64>();
    WeightedIndependentSetSolution {
        status,
        selected_vertex_indices: selected,
        selected_vertex_ids,
        total_weight,
        upper_bound,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_finds_sample_weighted_independent_set() {
        let p = build_sample_weighted_independent_set_problem();
        let solution = solve_weighted_independent_set_exact(&p);
        assert_eq!(solution.status, WeightedIndependentSetStatus::Optimal);
        assert_eq!(solution.selected_vertex_ids, vec!["B", "D", "G"]);
        assert!((solution.total_weight - 16.0).abs() <= 1e-9);
        assert!(weighted_independent_set_solution_feasible(&p, &solution));
    }

    #[test]
    fn greedy_returns_feasible_independent_set() {
        let p = build_sample_weighted_independent_set_problem();
        let exact = solve_weighted_independent_set_exact(&p);
        let greedy = solve_weighted_independent_set_greedy(&p);
        assert_eq!(greedy.status, WeightedIndependentSetStatus::Feasible);
        assert!(weighted_independent_set_solution_feasible(&p, &greedy));
        assert!(greedy.total_weight <= exact.total_weight + 1e-9);
    }
}
