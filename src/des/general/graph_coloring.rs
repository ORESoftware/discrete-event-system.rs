//! Small graph-coloring and chromatic-number models.
//!
//! Graph coloring is a compact CP/MIP benchmark: adjacent vertices must take
//! different colors, while the chromatic objective minimizes the number of
//! colors used. The exact solver here is DSATUR-style backtracking for
//! validation-scale graphs; the greedy solver is Welsh-Powell style.

use std::collections::{HashMap, HashSet};

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "graph-coloring";
const UNCOLORED: usize = usize::MAX;
const MAX_EXACT_VERTICES: usize = 40;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphColoringProblem {
    pub vertices: Vec<String>,
    /// Undirected edges as vertex-id pairs.
    pub edges: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphColoringStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
}

impl GraphColoringStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            GraphColoringStatus::Optimal => "optimal",
            GraphColoringStatus::Feasible => "feasible",
            GraphColoringStatus::Infeasible => "infeasible",
            GraphColoringStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphColoringSolution {
    pub status: GraphColoringStatus,
    /// Color index by vertex order.
    pub color_indices: Vec<usize>,
    /// Color label by vertex order.
    pub color_names: Vec<String>,
    pub used_color_count: Option<usize>,
    pub message: String,
}

pub fn build_sample_graph_coloring_problem() -> GraphColoringProblem {
    GraphColoringProblem {
        vertices: ["A", "B", "C", "D", "E", "F"]
            .iter()
            .map(|id| id.to_string())
            .collect(),
        edges: [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "A"),
            ("A", "F"),
            ("C", "F"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect(),
    }
}

pub fn validate_graph_coloring_problem(p: &GraphColoringProblem) -> Result<(), PreconditionError> {
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

pub fn solve_graph_coloring_greedy(p: &GraphColoringProblem) -> GraphColoringSolution {
    validate_graph_coloring_problem(p).expect("graph-coloring: invalid problem instance");
    let adjacency = adjacency_lists(p);
    let mut order = (0..p.vertices.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        adjacency[b]
            .len()
            .cmp(&adjacency[a].len())
            .then_with(|| p.vertices[a].cmp(&p.vertices[b]))
    });
    let mut colors = vec![UNCOLORED; p.vertices.len()];
    for &vertex in &order {
        let mut used_by_neighbors = HashSet::new();
        for &neighbor in &adjacency[vertex] {
            if colors[neighbor] != UNCOLORED {
                used_by_neighbors.insert(colors[neighbor]);
            }
        }
        let mut color = 0;
        while used_by_neighbors.contains(&color) {
            color += 1;
        }
        colors[vertex] = color;
    }
    build_solution(
        GraphColoringStatus::Feasible,
        colors,
        "Welsh-Powell greedy graph coloring",
    )
}

pub fn solve_graph_coloring_exact(p: &GraphColoringProblem) -> GraphColoringSolution {
    validate_graph_coloring_problem(p).expect("graph-coloring: invalid problem instance");
    if p.vertices.len() > MAX_EXACT_VERTICES {
        return GraphColoringSolution {
            status: GraphColoringStatus::Unsupported,
            color_indices: Vec::new(),
            color_names: Vec::new(),
            used_color_count: None,
            message: format!(
                "exact graph-coloring only practical for <= {MAX_EXACT_VERTICES} vertices, got {}",
                p.vertices.len()
            ),
        };
    }
    let adjacency = adjacency_lists(p);
    let greedy = solve_graph_coloring_greedy(p);
    let upper = greedy.used_color_count.unwrap_or(p.vertices.len().max(1));
    let lower = if p.edges.is_empty() { 1 } else { 2 };
    for k in lower..=upper {
        let mut colors = vec![UNCOLORED; p.vertices.len()];
        if dsatur_color(&adjacency, k, &mut colors, 0) {
            return build_solution(
                GraphColoringStatus::Optimal,
                colors,
                "exact DSATUR-style chromatic search",
            );
        }
    }
    GraphColoringSolution {
        status: GraphColoringStatus::Infeasible,
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        message: "no coloring found".to_string(),
    }
}

pub fn graph_coloring_solution_feasible(
    p: &GraphColoringProblem,
    solution: &GraphColoringSolution,
) -> bool {
    if validate_graph_coloring_problem(p).is_err()
        || solution.color_indices.len() != p.vertices.len()
        || solution.used_color_count != Some(solution.color_names.len())
    {
        return false;
    }
    let Some(color_count) = solution.used_color_count else {
        return false;
    };
    if color_count == 0
        || solution
            .color_indices
            .iter()
            .any(|&color| color >= color_count)
    {
        return false;
    }
    let vertex_index = p
        .vertices
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx))
        .collect::<HashMap<_, _>>();
    for (a, b) in &p.edges {
        let ai = vertex_index[a];
        let bi = vertex_index[b];
        if solution.color_indices[ai] == solution.color_indices[bi] {
            return false;
        }
    }
    true
}

fn adjacency_lists(p: &GraphColoringProblem) -> Vec<Vec<usize>> {
    let vertex_index = p
        .vertices
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); p.vertices.len()];
    for (a, b) in &p.edges {
        let ai = vertex_index[a];
        let bi = vertex_index[b];
        adjacency[ai].push(bi);
        adjacency[bi].push(ai);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn dsatur_color(
    adjacency: &[Vec<usize>],
    max_colors: usize,
    colors: &mut [usize],
    used_colors: usize,
) -> bool {
    let Some(vertex) = select_dsatur_vertex(adjacency, colors) else {
        return true;
    };

    let candidate_limit = used_colors.saturating_add(1).min(max_colors);
    for color in 0..candidate_limit {
        if !can_use_color(adjacency, colors, vertex, color) {
            continue;
        }
        colors[vertex] = color;
        let next_used = used_colors.max(color + 1);
        if dsatur_color(adjacency, max_colors, colors, next_used) {
            return true;
        }
        colors[vertex] = UNCOLORED;
    }
    false
}

fn select_dsatur_vertex(adjacency: &[Vec<usize>], colors: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize, usize)> = None;
    for vertex in 0..colors.len() {
        if colors[vertex] != UNCOLORED {
            continue;
        }
        let mut neighbor_colors = HashSet::new();
        for &neighbor in &adjacency[vertex] {
            if colors[neighbor] != UNCOLORED {
                neighbor_colors.insert(colors[neighbor]);
            }
        }
        let sat = neighbor_colors.len();
        let degree = adjacency[vertex].len();
        if best.is_none_or(|(_, best_sat, best_degree)| {
            sat > best_sat || (sat == best_sat && degree > best_degree)
        }) {
            best = Some((vertex, sat, degree));
        }
    }
    best.map(|(vertex, _, _)| vertex)
}

fn can_use_color(adjacency: &[Vec<usize>], colors: &[usize], vertex: usize, color: usize) -> bool {
    adjacency[vertex]
        .iter()
        .all(|&neighbor| colors[neighbor] != color)
}

fn build_solution(
    status: GraphColoringStatus,
    color_indices: Vec<usize>,
    message: &str,
) -> GraphColoringSolution {
    let used_color_count = color_indices
        .iter()
        .copied()
        .filter(|&color| color != UNCOLORED)
        .max()
        .map(|max_color| max_color + 1);
    let color_names = used_color_count
        .map(|count| (0..count).map(|idx| format!("C{}", idx + 1)).collect())
        .unwrap_or_default();
    GraphColoringSolution {
        status,
        color_indices,
        color_names,
        used_color_count,
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_graph_coloring_finds_sample_chromatic_number() {
        let p = build_sample_graph_coloring_problem();
        let exact = solve_graph_coloring_exact(&p);
        assert_eq!(exact.status, GraphColoringStatus::Optimal);
        assert_eq!(exact.used_color_count, Some(3));
        assert!(graph_coloring_solution_feasible(&p, &exact));
    }

    #[test]
    fn greedy_graph_coloring_returns_feasible_coloring() {
        let p = build_sample_graph_coloring_problem();
        let greedy = solve_graph_coloring_greedy(&p);
        assert_eq!(greedy.status, GraphColoringStatus::Feasible);
        assert!(graph_coloring_solution_feasible(&p, &greedy));
    }
}
