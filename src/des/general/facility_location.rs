//! Small uncapacitated facility-location models.
//!
//! This named surface complements the generic MILP builder with direct
//! validation-scale exact search and a fast open/drop local-search heuristic.

use std::collections::HashSet;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "facility-location";
const EPS: f64 = 1e-9;
const MAX_EXACT_FACILITIES: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct FacilityLocationProblem {
    pub facility_ids: Vec<String>,
    pub customer_ids: Vec<String>,
    pub fixed_costs: Vec<f64>,
    /// service_costs[i][j] is the cost of facility i serving customer j.
    pub service_costs: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacilityLocationStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
}

impl FacilityLocationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FacilityLocationStatus::Optimal => "optimal",
            FacilityLocationStatus::Feasible => "feasible",
            FacilityLocationStatus::Infeasible => "infeasible",
            FacilityLocationStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FacilityLocationAssignment {
    pub customer_index: usize,
    pub customer_id: String,
    pub facility_index: usize,
    pub facility_id: String,
    pub cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FacilityLocationSolution {
    pub status: FacilityLocationStatus,
    pub open_facility_indices: Vec<usize>,
    pub open_facility_ids: Vec<String>,
    pub assignments: Vec<FacilityLocationAssignment>,
    pub objective: Option<f64>,
    pub message: String,
}

pub fn build_sample_facility_location_problem() -> FacilityLocationProblem {
    FacilityLocationProblem {
        facility_ids: vec![
            "North".to_string(),
            "Central".to_string(),
            "South".to_string(),
        ],
        customer_ids: vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ],
        fixed_costs: vec![6.0, 10.0, 6.0],
        service_costs: vec![
            vec![2.0, 4.0, 7.0, 9.0, 8.0],
            vec![5.0, 3.0, 4.0, 4.0, 6.0],
            vec![9.0, 7.0, 5.0, 3.0, 2.0],
        ],
    }
}

pub fn validate_facility_location_problem(
    p: &FacilityLocationProblem,
) -> Result<(), PreconditionError> {
    Preconditions::non_empty(MODEL, "facility_ids", &p.facility_ids)?;
    Preconditions::non_empty(MODEL, "customer_ids", &p.customer_ids)?;
    Preconditions::length_eq(MODEL, "fixed_costs", &p.fixed_costs, p.facility_ids.len())?;
    Preconditions::length_eq(
        MODEL,
        "service_costs",
        &p.service_costs,
        p.facility_ids.len(),
    )?;

    let mut seen_facilities = HashSet::new();
    for (idx, id) in p.facility_ids.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("facility_ids[{idx}]"),
            "be non-empty",
            !id.trim().is_empty(),
            Some(id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("facility_ids[{idx}]"),
            "be unique",
            seen_facilities.insert(id.clone()),
            Some(id.clone()),
        )?;
        Preconditions::non_negative(MODEL, &format!("fixed_costs[{idx}]"), p.fixed_costs[idx])?;
    }

    let mut seen_customers = HashSet::new();
    for (idx, id) in p.customer_ids.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("customer_ids[{idx}]"),
            "be non-empty",
            !id.trim().is_empty(),
            Some(id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("customer_ids[{idx}]"),
            "be unique",
            seen_customers.insert(id.clone()),
            Some(id.clone()),
        )?;
    }

    for (facility_idx, row) in p.service_costs.iter().enumerate() {
        Preconditions::length_eq(
            MODEL,
            &format!("service_costs[{facility_idx}]"),
            row,
            p.customer_ids.len(),
        )?;
        for (customer_idx, &cost) in row.iter().enumerate() {
            Preconditions::non_negative(
                MODEL,
                &format!("service_costs[{facility_idx}][{customer_idx}]"),
                cost,
            )?;
        }
    }
    Ok(())
}

pub fn solve_facility_location_greedy(p: &FacilityLocationProblem) -> FacilityLocationSolution {
    validate_facility_location_problem(p).expect("facility-location: invalid problem instance");
    let mut open = (0..p.facility_ids.len()).collect::<Vec<_>>();
    let Some((mut best_cost, mut best_assignments)) = evaluate_open_facilities(p, &open) else {
        return infeasible_solution("no feasible open-facility assignment");
    };

    loop {
        let mut best_move: Option<(Vec<usize>, f64, Vec<FacilityLocationAssignment>)> = None;
        for facility_idx in 0..p.facility_ids.len() {
            let mut candidate_open = open.clone();
            if let Some(pos) = candidate_open.iter().position(|&idx| idx == facility_idx) {
                if candidate_open.len() == 1 {
                    continue;
                }
                candidate_open.remove(pos);
            } else {
                candidate_open.push(facility_idx);
            }
            candidate_open.sort_unstable();
            let Some((candidate_cost, candidate_assignments)) =
                evaluate_open_facilities(p, &candidate_open)
            else {
                continue;
            };
            if candidate_cost < best_cost - EPS
                || ((candidate_cost - best_cost).abs() <= EPS
                    && best_move
                        .as_ref()
                        .is_some_and(|(incumbent, _, _)| candidate_open < *incumbent))
            {
                best_move = Some((candidate_open, candidate_cost, candidate_assignments));
            }
        }
        let Some((candidate_open, candidate_cost, candidate_assignments)) = best_move else {
            break;
        };
        if candidate_cost >= best_cost - EPS {
            break;
        }
        open = candidate_open;
        best_cost = candidate_cost;
        best_assignments = candidate_assignments;
    }

    build_solution(
        p,
        FacilityLocationStatus::Feasible,
        open,
        best_assignments,
        best_cost,
        "greedy open/drop facility-location search",
    )
}

pub fn solve_facility_location_exact(p: &FacilityLocationProblem) -> FacilityLocationSolution {
    validate_facility_location_problem(p).expect("facility-location: invalid problem instance");
    if p.facility_ids.len() > MAX_EXACT_FACILITIES {
        return FacilityLocationSolution {
            status: FacilityLocationStatus::Unsupported,
            open_facility_indices: Vec::new(),
            open_facility_ids: Vec::new(),
            assignments: Vec::new(),
            objective: None,
            message: format!(
                "exact facility-location enumeration only practical for <= {MAX_EXACT_FACILITIES} facilities, got {}",
                p.facility_ids.len()
            ),
        };
    }

    let mut best_open = Vec::new();
    let mut best_assignments = Vec::new();
    let mut best_cost = f64::INFINITY;
    let upper_mask = 1_u128 << p.facility_ids.len();
    for mask in 1_u128..upper_mask {
        let open = (0..p.facility_ids.len())
            .filter(|&idx| mask & (1_u128 << idx) != 0)
            .collect::<Vec<_>>();
        let Some((candidate_cost, candidate_assignments)) = evaluate_open_facilities(p, &open)
        else {
            continue;
        };
        if candidate_cost < best_cost - EPS
            || ((candidate_cost - best_cost).abs() <= EPS && open < best_open)
        {
            best_open = open;
            best_assignments = candidate_assignments;
            best_cost = candidate_cost;
        }
    }

    if best_open.is_empty() {
        return infeasible_solution("no feasible facility subset");
    }
    build_solution(
        p,
        FacilityLocationStatus::Optimal,
        best_open,
        best_assignments,
        best_cost,
        "exact open-facility subset enumeration",
    )
}

pub fn facility_location_solution_feasible(
    p: &FacilityLocationProblem,
    solution: &FacilityLocationSolution,
) -> bool {
    if validate_facility_location_problem(p).is_err()
        || solution.open_facility_indices.len() != solution.open_facility_ids.len()
        || solution.assignments.len() != p.customer_ids.len()
    {
        return false;
    }
    let mut seen_open = HashSet::new();
    let mut open_set = HashSet::new();
    let mut fixed_cost = 0.0;
    for (&facility_idx, facility_id) in solution
        .open_facility_indices
        .iter()
        .zip(&solution.open_facility_ids)
    {
        if facility_idx >= p.facility_ids.len()
            || p.facility_ids[facility_idx] != *facility_id
            || !seen_open.insert(facility_idx)
        {
            return false;
        }
        open_set.insert(facility_idx);
        fixed_cost += p.fixed_costs[facility_idx];
    }
    if open_set.is_empty() {
        return false;
    }

    let mut seen_customers = HashSet::new();
    let mut assignment_cost = 0.0;
    for assignment in &solution.assignments {
        if assignment.customer_index >= p.customer_ids.len()
            || assignment.facility_index >= p.facility_ids.len()
            || p.customer_ids[assignment.customer_index] != assignment.customer_id
            || p.facility_ids[assignment.facility_index] != assignment.facility_id
            || !open_set.contains(&assignment.facility_index)
            || !seen_customers.insert(assignment.customer_index)
        {
            return false;
        }
        let expected_cost = p.service_costs[assignment.facility_index][assignment.customer_index];
        if (assignment.cost - expected_cost).abs() > EPS {
            return false;
        }
        assignment_cost += assignment.cost;
    }
    if seen_customers.len() != p.customer_ids.len() {
        return false;
    }
    let cost = fixed_cost + assignment_cost;
    solution
        .objective
        .is_some_and(|objective| (objective - cost).abs() <= 1e-8 * 1.0_f64.max(cost.abs()))
}

fn evaluate_open_facilities(
    p: &FacilityLocationProblem,
    open: &[usize],
) -> Option<(f64, Vec<FacilityLocationAssignment>)> {
    if open.is_empty() {
        return None;
    }
    let mut total = open.iter().map(|&idx| p.fixed_costs[idx]).sum::<f64>();
    let mut assignments = Vec::with_capacity(p.customer_ids.len());
    for customer_idx in 0..p.customer_ids.len() {
        let mut best: Option<(usize, f64)> = None;
        for &facility_idx in open {
            let cost = p.service_costs[facility_idx][customer_idx];
            if best.is_none_or(|(best_idx, best_cost)| {
                cost < best_cost - EPS
                    || ((cost - best_cost).abs() <= EPS && facility_idx < best_idx)
            }) {
                best = Some((facility_idx, cost));
            }
        }
        let (facility_idx, cost) = best?;
        total += cost;
        assignments.push(FacilityLocationAssignment {
            customer_index: customer_idx,
            customer_id: p.customer_ids[customer_idx].clone(),
            facility_index: facility_idx,
            facility_id: p.facility_ids[facility_idx].clone(),
            cost,
        });
    }
    Some((total, assignments))
}

fn build_solution(
    p: &FacilityLocationProblem,
    status: FacilityLocationStatus,
    mut open_facility_indices: Vec<usize>,
    assignments: Vec<FacilityLocationAssignment>,
    objective: f64,
    message: &str,
) -> FacilityLocationSolution {
    open_facility_indices.sort_unstable();
    let open_facility_ids = open_facility_indices
        .iter()
        .map(|&idx| p.facility_ids[idx].clone())
        .collect();
    FacilityLocationSolution {
        status,
        open_facility_indices,
        open_facility_ids,
        assignments,
        objective: Some(objective),
        message: message.to_string(),
    }
}

fn infeasible_solution(message: &str) -> FacilityLocationSolution {
    FacilityLocationSolution {
        status: FacilityLocationStatus::Infeasible,
        open_facility_indices: Vec::new(),
        open_facility_ids: Vec::new(),
        assignments: Vec::new(),
        objective: None,
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_facility_location_finds_sample_optimum() {
        let p = build_sample_facility_location_problem();
        let exact = solve_facility_location_exact(&p);
        assert_eq!(exact.status, FacilityLocationStatus::Optimal);
        assert_eq!(exact.open_facility_ids, vec!["North", "South"]);
        assert!(
            exact
                .objective
                .is_some_and(|objective| (objective - 28.0).abs() <= 1e-9),
            "objective: {:?}",
            exact.objective
        );
        assert!(facility_location_solution_feasible(&p, &exact));
    }

    #[test]
    fn greedy_facility_location_returns_feasible_solution() {
        let p = build_sample_facility_location_problem();
        let greedy = solve_facility_location_greedy(&p);
        assert_eq!(greedy.status, FacilityLocationStatus::Feasible);
        assert!(facility_location_solution_feasible(&p, &greedy));
    }
}
