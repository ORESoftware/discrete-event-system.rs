//! Small-instance one-dimensional bin packing.
//!
//! The exact solver is a branch-and-bound enumerator over bin assignments. It
//! is intended for validation-scale models and same-input oracle comparisons;
//! larger production instances should use the first-fit-decreasing heuristic or
//! an external MIP/CP solver through the reference bridges.

use std::collections::HashSet;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "bin-packing";
const EPS: f64 = 1e-9;
const MAX_EXACT_ITEMS: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct BinPackingItem {
    pub id: String,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinPackingProblem {
    pub capacity: f64,
    pub items: Vec<BinPackingItem>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinPackingStatus {
    Optimal,
    Feasible,
    Unsupported,
}

impl BinPackingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            BinPackingStatus::Optimal => "optimal",
            BinPackingStatus::Feasible => "feasible",
            BinPackingStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinPackingBin {
    pub item_indices: Vec<usize>,
    pub item_ids: Vec<String>,
    pub load: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinPackingSolution {
    pub status: BinPackingStatus,
    pub bins: Vec<BinPackingBin>,
    pub objective: Option<usize>,
    pub total_weight: f64,
    pub lower_bound_bins: usize,
    pub message: String,
}

#[derive(Clone, Debug)]
struct SearchItem {
    index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct SearchBin {
    item_indices: Vec<usize>,
    load: f64,
}

pub fn bin_packing_problem_from_weights(capacity: f64, weights: Vec<f64>) -> BinPackingProblem {
    BinPackingProblem {
        capacity,
        items: weights
            .into_iter()
            .enumerate()
            .map(|(idx, weight)| BinPackingItem {
                id: format!("I{}", idx + 1),
                weight,
            })
            .collect(),
    }
}

pub fn build_sample_bin_packing_problem() -> BinPackingProblem {
    bin_packing_problem_from_weights(10.0, vec![4.0, 8.0, 1.0, 4.0, 2.0, 1.0, 7.0, 3.0])
}

pub fn validate_bin_packing_problem(p: &BinPackingProblem) -> Result<(), PreconditionError> {
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
            &format!("items[{idx}].weight"),
            "be <= capacity",
            item.weight <= p.capacity + EPS,
            Some(format!("{} > {}", item.weight, p.capacity)),
        )?;
    }
    Ok(())
}

pub fn solve_bin_packing_first_fit_decreasing(p: &BinPackingProblem) -> BinPackingSolution {
    validate_bin_packing_problem(p).expect("bin-packing: invalid problem instance");
    let order = sorted_items(p);
    let mut bins: Vec<SearchBin> = Vec::new();
    for item in order {
        if let Some(bin) = bins
            .iter_mut()
            .find(|bin| bin.load + item.weight <= p.capacity + EPS)
        {
            bin.load += item.weight;
            bin.item_indices.push(item.index);
        } else {
            bins.push(SearchBin {
                item_indices: vec![item.index],
                load: item.weight,
            });
        }
    }
    build_solution(
        p,
        BinPackingStatus::Feasible,
        bins,
        "first-fit-decreasing heuristic",
    )
}

pub fn solve_bin_packing_exact(p: &BinPackingProblem) -> BinPackingSolution {
    validate_bin_packing_problem(p).expect("bin-packing: invalid problem instance");
    if p.items.len() > MAX_EXACT_ITEMS {
        return BinPackingSolution {
            status: BinPackingStatus::Unsupported,
            bins: Vec::new(),
            objective: None,
            total_weight: total_weight(p),
            lower_bound_bins: lower_bound_bins(p),
            message: format!(
                "exact bin-packing only practical for <= {MAX_EXACT_ITEMS} items, got {}",
                p.items.len()
            ),
        };
    }

    let incumbent = solve_bin_packing_first_fit_decreasing(p);
    let mut best_bins = incumbent
        .bins
        .iter()
        .map(|bin| SearchBin {
            item_indices: bin.item_indices.clone(),
            load: bin.load,
        })
        .collect::<Vec<_>>();
    let mut best_count = best_bins.len();
    let lower_bound = lower_bound_bins(p);
    if best_count == lower_bound {
        return build_solution(
            p,
            BinPackingStatus::Optimal,
            best_bins,
            "exact branch-and-bound certified by volume lower bound",
        );
    }

    let order = sorted_items(p);
    let mut suffix_weight = vec![0.0; order.len() + 1];
    for idx in (0..order.len()).rev() {
        suffix_weight[idx] = suffix_weight[idx + 1] + order[idx].weight;
    }
    let mut current: Vec<SearchBin> = Vec::new();
    exact_search(
        p.capacity,
        &order,
        &suffix_weight,
        0,
        &mut current,
        &mut best_bins,
        &mut best_count,
    );
    build_solution(
        p,
        BinPackingStatus::Optimal,
        best_bins,
        "exact branch-and-bound",
    )
}

pub fn bin_packing_solution_feasible(p: &BinPackingProblem, solution: &BinPackingSolution) -> bool {
    if validate_bin_packing_problem(p).is_err() || solution.objective != Some(solution.bins.len()) {
        return false;
    }
    let mut seen = HashSet::new();
    for bin in &solution.bins {
        if bin.item_indices.len() != bin.item_ids.len() || bin.load > p.capacity + 1e-8 {
            return false;
        }
        let mut load = 0.0;
        for (&idx, id) in bin.item_indices.iter().zip(&bin.item_ids) {
            let Some(item) = p.items.get(idx) else {
                return false;
            };
            if item.id != *id || !seen.insert(idx) {
                return false;
            }
            load += item.weight;
        }
        if (load - bin.load).abs() > 1e-8 * 1.0_f64.max(load.abs()) {
            return false;
        }
    }
    seen.len() == p.items.len()
}

fn sorted_items(p: &BinPackingProblem) -> Vec<SearchItem> {
    let mut items = p
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| SearchItem {
            index,
            weight: item.weight,
        })
        .collect::<Vec<_>>();
    items.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.index.cmp(&b.index))
    });
    items
}

fn total_weight(p: &BinPackingProblem) -> f64 {
    p.items.iter().map(|item| item.weight).sum()
}

fn lower_bound_bins(p: &BinPackingProblem) -> usize {
    (total_weight(p) / p.capacity).ceil() as usize
}

fn build_solution(
    p: &BinPackingProblem,
    status: BinPackingStatus,
    bins: Vec<SearchBin>,
    message: impl Into<String>,
) -> BinPackingSolution {
    let bins = bins
        .into_iter()
        .map(|bin| {
            let mut item_indices = bin.item_indices;
            item_indices.sort_unstable();
            let load = item_indices
                .iter()
                .map(|&idx| p.items[idx].weight)
                .sum::<f64>();
            let item_ids = item_indices
                .iter()
                .map(|&idx| p.items[idx].id.clone())
                .collect::<Vec<_>>();
            BinPackingBin {
                item_indices,
                item_ids,
                load,
            }
        })
        .collect::<Vec<_>>();
    BinPackingSolution {
        status,
        objective: Some(bins.len()),
        bins,
        total_weight: total_weight(p),
        lower_bound_bins: lower_bound_bins(p),
        message: message.into(),
    }
}

fn exact_search(
    capacity: f64,
    order: &[SearchItem],
    suffix_weight: &[f64],
    pos: usize,
    current: &mut Vec<SearchBin>,
    best_bins: &mut Vec<SearchBin>,
    best_count: &mut usize,
) {
    if current.len() >= *best_count {
        return;
    }
    if pos == order.len() {
        *best_count = current.len();
        *best_bins = current.clone();
        return;
    }

    let free_capacity = current
        .iter()
        .map(|bin| (capacity - bin.load).max(0.0))
        .sum::<f64>();
    let additional_weight = (suffix_weight[pos] - free_capacity).max(0.0);
    let additional_bins = (additional_weight / capacity).ceil() as usize;
    if current.len() + additional_bins >= *best_count {
        return;
    }

    let item = &order[pos];
    let mut tried_loads: Vec<f64> = Vec::new();
    for bin_idx in 0..current.len() {
        let load = current[bin_idx].load;
        if load + item.weight > capacity + EPS
            || tried_loads
                .iter()
                .any(|&previous| (previous - load).abs() <= EPS)
        {
            continue;
        }
        tried_loads.push(load);
        current[bin_idx].load += item.weight;
        current[bin_idx].item_indices.push(item.index);
        exact_search(
            capacity,
            order,
            suffix_weight,
            pos + 1,
            current,
            best_bins,
            best_count,
        );
        current[bin_idx].item_indices.pop();
        current[bin_idx].load = load;
    }

    if current.len() + 1 < *best_count {
        current.push(SearchBin {
            item_indices: vec![item.index],
            load: item.weight,
        });
        exact_search(
            capacity,
            order,
            suffix_weight,
            pos + 1,
            current,
            best_bins,
            best_count,
        );
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bin_packing_finds_three_bin_sample() {
        let p = build_sample_bin_packing_problem();
        let exact = solve_bin_packing_exact(&p);
        assert_eq!(exact.status, BinPackingStatus::Optimal);
        assert_eq!(exact.objective, Some(3));
        assert!(bin_packing_solution_feasible(&p, &exact));
    }

    #[test]
    fn first_fit_decreasing_returns_feasible_packing() {
        let p = build_sample_bin_packing_problem();
        let ffd = solve_bin_packing_first_fit_decreasing(&p);
        assert_eq!(ffd.status, BinPackingStatus::Feasible);
        assert!(bin_packing_solution_feasible(&p, &ffd));
    }
}
