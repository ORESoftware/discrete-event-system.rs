//! Small-instance 0/1 knapsack models.
//!
//! The exact solver is a branch-and-bound enumerator with a fractional
//! relaxation upper bound. It is intended for validation-scale same-input
//! oracle comparisons; the density greedy solver provides a fast feasible
//! baseline for larger or streaming-style selection problems.

use std::collections::HashSet;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "knapsack";
const EPS: f64 = 1e-9;
const MAX_EXACT_ITEMS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackItem {
    pub id: String,
    pub weight: f64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackProblem {
    pub capacity: f64,
    pub items: Vec<KnapsackItem>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnapsackStatus {
    Optimal,
    Feasible,
    Unsupported,
}

impl KnapsackStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            KnapsackStatus::Optimal => "optimal",
            KnapsackStatus::Feasible => "feasible",
            KnapsackStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackSolution {
    pub status: KnapsackStatus,
    pub selected_item_indices: Vec<usize>,
    pub selected_item_ids: Vec<String>,
    pub total_weight: f64,
    pub total_value: f64,
    pub upper_bound: Option<f64>,
    pub message: String,
}

#[derive(Clone, Debug)]
struct SearchItem {
    index: usize,
    weight: f64,
    value: f64,
    density: f64,
}

pub fn knapsack_problem_from_weights_values(
    capacity: f64,
    weights: Vec<f64>,
    values: Vec<f64>,
) -> KnapsackProblem {
    assert_eq!(
        weights.len(),
        values.len(),
        "knapsack: weights and values must have the same length"
    );
    KnapsackProblem {
        capacity,
        items: weights
            .into_iter()
            .zip(values)
            .enumerate()
            .map(|(idx, (weight, value))| KnapsackItem {
                id: format!("I{}", idx + 1),
                weight,
                value,
            })
            .collect(),
    }
}

pub fn build_sample_knapsack_problem() -> KnapsackProblem {
    KnapsackProblem {
        capacity: 26.0,
        items: vec![
            KnapsackItem {
                id: "A".to_string(),
                weight: 12.0,
                value: 24.0,
            },
            KnapsackItem {
                id: "B".to_string(),
                weight: 7.0,
                value: 13.0,
            },
            KnapsackItem {
                id: "C".to_string(),
                weight: 11.0,
                value: 23.0,
            },
            KnapsackItem {
                id: "D".to_string(),
                weight: 8.0,
                value: 15.0,
            },
            KnapsackItem {
                id: "E".to_string(),
                weight: 9.0,
                value: 16.0,
            },
            KnapsackItem {
                id: "F".to_string(),
                weight: 4.0,
                value: 9.0,
            },
        ],
    }
}

pub fn validate_knapsack_problem(p: &KnapsackProblem) -> Result<(), PreconditionError> {
    Preconditions::positive(MODEL, "capacity", p.capacity)?;
    Preconditions::non_empty(MODEL, "items", &p.items)?;
    let mut seen = HashSet::new();
    for (idx, item) in p.items.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("items[{idx}].id"),
            "be non-empty",
            !item.id.trim().is_empty(),
            Some(item.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("items[{idx}].id"),
            "be unique",
            seen.insert(item.id.clone()),
            Some(item.id.clone()),
        )?;
        Preconditions::positive(MODEL, &format!("items[{idx}].weight"), item.weight)?;
        Preconditions::check(
            MODEL,
            &format!("items[{idx}].value"),
            "be finite and non-negative",
            item.value.is_finite() && item.value >= 0.0,
            Some(item.value.to_string()),
        )?;
    }
    Ok(())
}

pub fn solve_knapsack_greedy_density(p: &KnapsackProblem) -> KnapsackSolution {
    validate_knapsack_problem(p).expect("knapsack: invalid problem instance");
    let order = sorted_items(p);
    let mut selected = Vec::new();
    let mut total_weight = 0.0;
    for item in order {
        if total_weight + item.weight <= p.capacity + EPS {
            total_weight += item.weight;
            selected.push(item.index);
        }
    }
    build_solution(
        p,
        KnapsackStatus::Feasible,
        selected,
        None,
        "greedy value-density heuristic",
    )
}

pub fn solve_knapsack_exact_branch_and_bound(p: &KnapsackProblem) -> KnapsackSolution {
    validate_knapsack_problem(p).expect("knapsack: invalid problem instance");
    if p.items.len() > MAX_EXACT_ITEMS {
        return KnapsackSolution {
            status: KnapsackStatus::Unsupported,
            selected_item_indices: Vec::new(),
            selected_item_ids: Vec::new(),
            total_weight: 0.0,
            total_value: 0.0,
            upper_bound: None,
            message: format!(
                "exact knapsack branch-and-bound only practical for <= {MAX_EXACT_ITEMS} items, got {}",
                p.items.len()
            ),
        };
    }

    let order = sorted_items(p);
    let root_upper_bound = fractional_upper_bound(p.capacity, &order, 0, 0.0, 0.0);
    let greedy = solve_knapsack_greedy_density(p);
    let mut best_indices = greedy.selected_item_indices.clone();
    let mut best_weight = greedy.total_weight;
    let mut best_value = greedy.total_value;
    let mut current = Vec::new();
    search_branch_and_bound(
        p.capacity,
        &order,
        0,
        0.0,
        0.0,
        &mut current,
        &mut best_indices,
        &mut best_weight,
        &mut best_value,
    );
    build_solution(
        p,
        KnapsackStatus::Optimal,
        best_indices,
        Some(root_upper_bound),
        "exact branch-and-bound with fractional-relaxation bound",
    )
}

pub fn knapsack_solution_feasible(p: &KnapsackProblem, solution: &KnapsackSolution) -> bool {
    if validate_knapsack_problem(p).is_err()
        || solution.selected_item_indices.len() != solution.selected_item_ids.len()
    {
        return false;
    }
    let mut seen = HashSet::new();
    let mut total_weight = 0.0;
    let mut total_value = 0.0;
    for (&idx, id) in solution
        .selected_item_indices
        .iter()
        .zip(&solution.selected_item_ids)
    {
        let Some(item) = p.items.get(idx) else {
            return false;
        };
        if item.id != *id || !seen.insert(idx) {
            return false;
        }
        total_weight += item.weight;
        total_value += item.value;
    }
    total_weight <= p.capacity + 1e-8
        && close(total_weight, solution.total_weight)
        && close(total_value, solution.total_value)
}

fn sorted_items(p: &KnapsackProblem) -> Vec<SearchItem> {
    let mut items = p
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| SearchItem {
            index,
            weight: item.weight,
            value: item.value,
            density: if item.weight > 0.0 {
                item.value / item.weight
            } else {
                f64::INFINITY
            },
        })
        .collect::<Vec<_>>();
    items.sort_by(|a, b| {
        b.density
            .total_cmp(&a.density)
            .then_with(|| b.value.total_cmp(&a.value))
            .then_with(|| a.weight.total_cmp(&b.weight))
            .then_with(|| a.index.cmp(&b.index))
    });
    items
}

fn fractional_upper_bound(
    capacity: f64,
    order: &[SearchItem],
    pos: usize,
    current_weight: f64,
    current_value: f64,
) -> f64 {
    if current_weight > capacity + EPS {
        return f64::NEG_INFINITY;
    }
    let mut bound = current_value;
    let mut remaining = capacity - current_weight;
    for item in &order[pos..] {
        if item.weight <= remaining + EPS {
            bound += item.value;
            remaining -= item.weight;
        } else if remaining > EPS {
            bound += item.value * (remaining / item.weight);
            break;
        } else {
            break;
        }
    }
    bound
}

#[allow(clippy::too_many_arguments)]
fn search_branch_and_bound(
    capacity: f64,
    order: &[SearchItem],
    pos: usize,
    current_weight: f64,
    current_value: f64,
    current: &mut Vec<usize>,
    best_indices: &mut Vec<usize>,
    best_weight: &mut f64,
    best_value: &mut f64,
) {
    if current_weight > capacity + EPS {
        return;
    }
    if pos == order.len() {
        if candidate_better(
            current_value,
            current_weight,
            current,
            *best_value,
            *best_weight,
            best_indices,
        ) {
            *best_indices = current.clone();
            *best_weight = current_weight;
            *best_value = current_value;
        }
        return;
    }
    let bound = fractional_upper_bound(capacity, order, pos, current_weight, current_value);
    if bound + EPS < *best_value {
        return;
    }

    let item = &order[pos];
    current.push(item.index);
    search_branch_and_bound(
        capacity,
        order,
        pos + 1,
        current_weight + item.weight,
        current_value + item.value,
        current,
        best_indices,
        best_weight,
        best_value,
    );
    current.pop();
    search_branch_and_bound(
        capacity,
        order,
        pos + 1,
        current_weight,
        current_value,
        current,
        best_indices,
        best_weight,
        best_value,
    );
}

fn candidate_better(
    value: f64,
    weight: f64,
    indices: &[usize],
    best_value: f64,
    best_weight: f64,
    best_indices: &[usize],
) -> bool {
    if value > best_value + EPS {
        return true;
    }
    if (value - best_value).abs() <= EPS && weight < best_weight - EPS {
        return true;
    }
    if (value - best_value).abs() <= EPS && (weight - best_weight).abs() <= EPS {
        let mut lhs = indices.to_vec();
        let mut rhs = best_indices.to_vec();
        lhs.sort_unstable();
        rhs.sort_unstable();
        return lhs < rhs;
    }
    false
}

fn build_solution(
    p: &KnapsackProblem,
    status: KnapsackStatus,
    mut selected: Vec<usize>,
    upper_bound: Option<f64>,
    message: impl Into<String>,
) -> KnapsackSolution {
    selected.sort_unstable();
    let selected_item_ids = selected
        .iter()
        .map(|&idx| p.items[idx].id.clone())
        .collect::<Vec<_>>();
    let total_weight = selected.iter().map(|&idx| p.items[idx].weight).sum::<f64>();
    let total_value = selected.iter().map(|&idx| p.items[idx].value).sum::<f64>();
    KnapsackSolution {
        status,
        selected_item_indices: selected,
        selected_item_ids,
        total_weight,
        total_value,
        upper_bound,
        message: message.into(),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * 1.0_f64.max(a.abs()).max(b.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_knapsack_finds_sample_optimum() {
        let p = build_sample_knapsack_problem();
        let solution = solve_knapsack_exact_branch_and_bound(&p);
        assert_eq!(solution.status, KnapsackStatus::Optimal);
        assert_eq!(solution.selected_item_ids, vec!["B", "C", "D"]);
        assert!((solution.total_weight - 26.0).abs() <= 1e-9);
        assert!((solution.total_value - 51.0).abs() <= 1e-9);
        assert!(knapsack_solution_feasible(&p, &solution));
    }

    #[test]
    fn greedy_knapsack_is_feasible_on_sample() {
        let p = build_sample_knapsack_problem();
        let exact = solve_knapsack_exact_branch_and_bound(&p);
        let greedy = solve_knapsack_greedy_density(&p);
        assert_eq!(greedy.status, KnapsackStatus::Feasible);
        assert!(knapsack_solution_feasible(&p, &greedy));
        assert!(greedy.total_value <= exact.total_value + 1e-9);
    }
}
