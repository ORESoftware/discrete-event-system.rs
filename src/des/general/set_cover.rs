//! Small-instance weighted set cover.
//!
//! This module keeps a named set-cover surface next to the more generic MIP
//! builders. The exact solver is a branch-and-bound enumerator for validation
//! scale instances; the greedy solver is useful as a fast incumbent and a
//! production-style heuristic.

use std::collections::{HashMap, HashSet};

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "set-cover";
const EPS: f64 = 1e-9;
const MAX_EXACT_SETS: usize = 32;
const MAX_EXACT_ELEMENTS: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub struct SetCoverSet {
    pub id: String,
    pub cost: f64,
    pub elements: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SetCoverProblem {
    pub universe: Vec<String>,
    pub sets: Vec<SetCoverSet>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetCoverStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
}

impl SetCoverStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SetCoverStatus::Optimal => "optimal",
            SetCoverStatus::Feasible => "feasible",
            SetCoverStatus::Infeasible => "infeasible",
            SetCoverStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SetCoverSolution {
    pub status: SetCoverStatus,
    pub selected_set_indices: Vec<usize>,
    pub selected_set_ids: Vec<String>,
    pub objective: Option<f64>,
    pub covered_elements: Vec<String>,
    pub message: String,
}

#[derive(Clone, Debug)]
struct IndexedSet {
    index: usize,
    cost: f64,
    mask: u128,
}

pub fn build_sample_set_cover_problem() -> SetCoverProblem {
    SetCoverProblem {
        universe: vec![
            "E1".to_string(),
            "E2".to_string(),
            "E3".to_string(),
            "E4".to_string(),
            "E5".to_string(),
            "E6".to_string(),
        ],
        sets: vec![
            SetCoverSet {
                id: "A".to_string(),
                cost: 3.0,
                elements: vec!["E1".to_string(), "E2".to_string(), "E3".to_string()],
            },
            SetCoverSet {
                id: "B".to_string(),
                cost: 2.0,
                elements: vec!["E2".to_string(), "E4".to_string()],
            },
            SetCoverSet {
                id: "C".to_string(),
                cost: 4.0,
                elements: vec!["E3".to_string(), "E4".to_string(), "E5".to_string()],
            },
            SetCoverSet {
                id: "D".to_string(),
                cost: 2.0,
                elements: vec!["E5".to_string(), "E6".to_string()],
            },
            SetCoverSet {
                id: "E".to_string(),
                cost: 5.0,
                elements: vec!["E1".to_string(), "E4".to_string(), "E6".to_string()],
            },
            SetCoverSet {
                id: "F".to_string(),
                cost: 1.0,
                elements: vec!["E6".to_string()],
            },
        ],
    }
}

pub fn validate_set_cover_problem(p: &SetCoverProblem) -> Result<(), PreconditionError> {
    Preconditions::non_empty(MODEL, "universe", &p.universe)?;
    Preconditions::non_empty(MODEL, "sets", &p.sets)?;
    let mut universe_seen = HashSet::new();
    for (idx, element) in p.universe.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("universe[{idx}]"),
            "be non-empty",
            !element.trim().is_empty(),
            Some(element.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("universe[{idx}]"),
            "be unique",
            universe_seen.insert(element.clone()),
            Some(element.clone()),
        )?;
    }
    let universe = p.universe.iter().cloned().collect::<HashSet<_>>();
    let mut set_ids = HashSet::new();
    for (set_idx, set) in p.sets.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("sets[{set_idx}].id"),
            "be non-empty",
            !set.id.trim().is_empty(),
            Some(set.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("sets[{set_idx}].id"),
            "be unique",
            set_ids.insert(set.id.clone()),
            Some(set.id.clone()),
        )?;
        Preconditions::non_negative(MODEL, &format!("sets[{set_idx}].cost"), set.cost)?;
        Preconditions::non_empty(MODEL, &format!("sets[{set_idx}].elements"), &set.elements)?;
        let mut element_seen = HashSet::new();
        for (element_idx, element) in set.elements.iter().enumerate() {
            Preconditions::check(
                MODEL,
                &format!("sets[{set_idx}].elements[{element_idx}]"),
                "belong to universe",
                universe.contains(element),
                Some(element.clone()),
            )?;
            Preconditions::check(
                MODEL,
                &format!("sets[{set_idx}].elements[{element_idx}]"),
                "be unique within set",
                element_seen.insert(element.clone()),
                Some(element.clone()),
            )?;
        }
    }
    Ok(())
}

pub fn solve_set_cover_greedy(p: &SetCoverProblem) -> SetCoverSolution {
    validate_set_cover_problem(p).expect("set-cover: invalid problem instance");
    let (indexed, full_mask) = indexed_sets(p);
    let mut covered = 0_u128;
    let mut selected = Vec::new();
    while covered != full_mask {
        let mut best: Option<&IndexedSet> = None;
        let mut best_new_bits = 0_u32;
        let mut best_ratio = f64::INFINITY;
        for set in &indexed {
            if selected.contains(&set.index) {
                continue;
            }
            let new_bits = (set.mask & !covered).count_ones();
            if new_bits == 0 {
                continue;
            }
            let ratio = set.cost / f64::from(new_bits);
            if ratio < best_ratio - EPS
                || ((ratio - best_ratio).abs() <= EPS
                    && (new_bits > best_new_bits
                        || (new_bits == best_new_bits
                            && best.is_none_or(|existing| set.index < existing.index))))
            {
                best = Some(set);
                best_new_bits = new_bits;
                best_ratio = ratio;
            }
        }
        let Some(chosen) = best else {
            return infeasible_solution("greedy could not cover remaining elements");
        };
        selected.push(chosen.index);
        covered |= chosen.mask;
    }
    build_solution(
        p,
        SetCoverStatus::Feasible,
        selected,
        "greedy weighted set cover",
    )
}

pub fn solve_set_cover_exact(p: &SetCoverProblem) -> SetCoverSolution {
    validate_set_cover_problem(p).expect("set-cover: invalid problem instance");
    if p.sets.len() > MAX_EXACT_SETS || p.universe.len() > MAX_EXACT_ELEMENTS {
        return SetCoverSolution {
            status: SetCoverStatus::Unsupported,
            selected_set_indices: Vec::new(),
            selected_set_ids: Vec::new(),
            objective: None,
            covered_elements: Vec::new(),
            message: format!(
                "exact set-cover only practical for <= {MAX_EXACT_SETS} sets and <= {MAX_EXACT_ELEMENTS} elements, got {} sets and {} elements",
                p.sets.len(),
                p.universe.len()
            ),
        };
    }
    let greedy = solve_set_cover_greedy(p);
    if greedy.status == SetCoverStatus::Infeasible {
        return greedy;
    }
    let mut best_indices = greedy.selected_set_indices.clone();
    let mut best_cost = greedy.objective.unwrap_or(f64::INFINITY);
    let (indexed, full_mask) = indexed_sets(p);
    let mut covering_sets: Vec<Vec<usize>> = vec![Vec::new(); p.universe.len()];
    for set in &indexed {
        for element_idx in 0..p.universe.len() {
            if set.mask & (1_u128 << element_idx) != 0 {
                covering_sets[element_idx].push(set.index);
            }
        }
    }
    if covering_sets.iter().any(Vec::is_empty) {
        return infeasible_solution("at least one universe element is uncovered by all sets");
    }
    let by_index = indexed
        .iter()
        .map(|set| (set.index, set.clone()))
        .collect::<HashMap<_, _>>();
    let mut current = Vec::new();
    exact_search(
        full_mask,
        0,
        0.0,
        &mut current,
        &covering_sets,
        &by_index,
        &mut best_indices,
        &mut best_cost,
    );
    build_solution(
        p,
        SetCoverStatus::Optimal,
        best_indices,
        "exact branch-and-bound",
    )
}

pub fn set_cover_solution_feasible(p: &SetCoverProblem, solution: &SetCoverSolution) -> bool {
    if validate_set_cover_problem(p).is_err() {
        return false;
    }
    match solution.status {
        SetCoverStatus::Optimal | SetCoverStatus::Feasible => {}
        _ => return false,
    }
    let mut selected_seen = HashSet::new();
    let mut covered = HashSet::new();
    let mut objective = 0.0;
    for (&idx, set_id) in solution
        .selected_set_indices
        .iter()
        .zip(&solution.selected_set_ids)
    {
        let Some(set) = p.sets.get(idx) else {
            return false;
        };
        if set.id != *set_id || !selected_seen.insert(idx) {
            return false;
        }
        objective += set.cost;
        covered.extend(set.elements.iter().cloned());
    }
    let universe = p.universe.iter().cloned().collect::<HashSet<_>>();
    if covered != universe {
        return false;
    }
    let reported = solution
        .covered_elements
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    reported == universe
        && solution
            .objective
            .is_some_and(|value| (value - objective).abs() <= 1e-8 * 1.0_f64.max(objective.abs()))
}

fn indexed_sets(p: &SetCoverProblem) -> (Vec<IndexedSet>, u128) {
    let element_index = p
        .universe
        .iter()
        .enumerate()
        .map(|(idx, element)| (element.clone(), idx))
        .collect::<HashMap<_, _>>();
    let sets = p
        .sets
        .iter()
        .enumerate()
        .map(|(index, set)| {
            let mut mask = 0_u128;
            for element in &set.elements {
                let idx = element_index[element];
                mask |= 1_u128 << idx;
            }
            IndexedSet {
                index,
                cost: set.cost,
                mask,
            }
        })
        .collect::<Vec<_>>();
    let full_mask = if p.universe.len() == 128 {
        u128::MAX
    } else {
        (1_u128 << p.universe.len()) - 1
    };
    (sets, full_mask)
}

#[allow(clippy::too_many_arguments)]
fn exact_search(
    full_mask: u128,
    covered: u128,
    current_cost: f64,
    current: &mut Vec<usize>,
    covering_sets: &[Vec<usize>],
    by_index: &HashMap<usize, IndexedSet>,
    best_indices: &mut Vec<usize>,
    best_cost: &mut f64,
) {
    if current_cost >= *best_cost - EPS {
        return;
    }
    if covered == full_mask {
        let mut candidate = current.clone();
        candidate.sort_unstable();
        let mut incumbent = best_indices.clone();
        incumbent.sort_unstable();
        if current_cost < *best_cost - EPS
            || ((current_cost - *best_cost).abs() <= EPS && candidate < incumbent)
        {
            *best_cost = current_cost;
            *best_indices = candidate;
        }
        return;
    }

    let uncovered = full_mask & !covered;
    let mut chosen_element = None;
    let mut chosen_candidates: Vec<usize> = Vec::new();
    for element_idx in 0..covering_sets.len() {
        if uncovered & (1_u128 << element_idx) == 0 {
            continue;
        }
        let candidates = covering_sets[element_idx]
            .iter()
            .copied()
            .filter(|idx| !current.contains(idx) && by_index[idx].mask & !covered != 0)
            .collect::<Vec<_>>();
        if chosen_element.is_none() || candidates.len() < chosen_candidates.len() {
            chosen_element = Some(element_idx);
            chosen_candidates = candidates;
        }
    }
    if chosen_candidates.is_empty() {
        return;
    }
    chosen_candidates.sort_by(|a, b| {
        by_index[a]
            .cost
            .total_cmp(&by_index[b].cost)
            .then_with(|| a.cmp(b))
    });
    for set_idx in chosen_candidates {
        let set = &by_index[&set_idx];
        current.push(set_idx);
        exact_search(
            full_mask,
            covered | set.mask,
            current_cost + set.cost,
            current,
            covering_sets,
            by_index,
            best_indices,
            best_cost,
        );
        current.pop();
    }
}

fn build_solution(
    p: &SetCoverProblem,
    status: SetCoverStatus,
    mut selected_set_indices: Vec<usize>,
    message: impl Into<String>,
) -> SetCoverSolution {
    selected_set_indices.sort_unstable();
    selected_set_indices.dedup();
    let selected_set_ids = selected_set_indices
        .iter()
        .map(|&idx| p.sets[idx].id.clone())
        .collect::<Vec<_>>();
    let objective = selected_set_indices
        .iter()
        .map(|&idx| p.sets[idx].cost)
        .sum::<f64>();
    let covered = selected_set_indices
        .iter()
        .flat_map(|&idx| p.sets[idx].elements.iter().cloned())
        .collect::<HashSet<_>>();
    let covered_elements = p
        .universe
        .iter()
        .filter(|element| covered.contains(*element))
        .cloned()
        .collect::<Vec<_>>();
    SetCoverSolution {
        status,
        selected_set_indices,
        selected_set_ids,
        objective: Some(objective),
        covered_elements,
        message: message.into(),
    }
}

fn infeasible_solution(message: impl Into<String>) -> SetCoverSolution {
    SetCoverSolution {
        status: SetCoverStatus::Infeasible,
        selected_set_indices: Vec::new(),
        selected_set_ids: Vec::new(),
        objective: None,
        covered_elements: Vec::new(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_set_cover_finds_sample_optimum() {
        let p = build_sample_set_cover_problem();
        let exact = solve_set_cover_exact(&p);
        assert_eq!(exact.status, SetCoverStatus::Optimal);
        assert_eq!(exact.objective, Some(7.0));
        assert!(set_cover_solution_feasible(&p, &exact));
    }

    #[test]
    fn greedy_set_cover_returns_feasible_cover() {
        let p = build_sample_set_cover_problem();
        let greedy = solve_set_cover_greedy(&p);
        assert_eq!(greedy.status, SetCoverStatus::Feasible);
        assert!(set_cover_solution_feasible(&p, &greedy));
    }
}
