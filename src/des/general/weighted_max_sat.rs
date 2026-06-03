//! Weighted partial Max-SAT models.
//!
//! Max-SAT is a compact Boolean optimization benchmark for CP-SAT and local
//! search systems: hard clauses must be satisfied, while soft clauses
//! contribute weighted reward when satisfied. The exact solver here enumerates
//! validation-scale models; the greedy solver uses coordinate-improvement
//! repair and improvement.

use std::collections::HashSet;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

const MODEL: &str = "weighted-max-sat";
const MAX_EXACT_VARS: usize = 26;
const EPS: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMaxSatClause {
    pub id: String,
    /// Literals use DIMACS-style one-based signed variable ids.
    pub literals: Vec<i64>,
    /// Soft-clause reward. Hard clauses are enforced regardless of this value.
    pub weight: f64,
    pub hard: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMaxSatProblem {
    pub num_vars: usize,
    pub clauses: Vec<WeightedMaxSatClause>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightedMaxSatStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
}

impl WeightedMaxSatStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightedMaxSatStatus::Optimal => "optimal",
            WeightedMaxSatStatus::Feasible => "feasible",
            WeightedMaxSatStatus::Infeasible => "infeasible",
            WeightedMaxSatStatus::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMaxSatSolution {
    pub status: WeightedMaxSatStatus,
    /// Boolean assignment by zero-based variable order.
    pub assignment: Vec<bool>,
    pub satisfied_soft_weight: Option<f64>,
    pub unsatisfied_soft_weight: Option<f64>,
    pub satisfied_clause_ids: Vec<String>,
    pub violated_hard_clause_ids: Vec<String>,
    pub message: String,
}

pub fn build_sample_weighted_max_sat_problem() -> WeightedMaxSatProblem {
    WeightedMaxSatProblem {
        num_vars: 3,
        clauses: vec![
            WeightedMaxSatClause {
                id: "H_cover".to_string(),
                literals: vec![1, 2],
                weight: 0.0,
                hard: true,
            },
            WeightedMaxSatClause {
                id: "H_implication".to_string(),
                literals: vec![-2, 3],
                weight: 0.0,
                hard: true,
            },
            WeightedMaxSatClause {
                id: "S_pick_x1".to_string(),
                literals: vec![1],
                weight: 6.0,
                hard: false,
            },
            WeightedMaxSatClause {
                id: "S_pick_x2".to_string(),
                literals: vec![2],
                weight: 6.0,
                hard: false,
            },
            WeightedMaxSatClause {
                id: "S_not_both_x1_x2".to_string(),
                literals: vec![-1, -2],
                weight: 5.0,
                hard: false,
            },
            WeightedMaxSatClause {
                id: "S_pick_x3".to_string(),
                literals: vec![3],
                weight: 4.0,
                hard: false,
            },
            WeightedMaxSatClause {
                id: "S_skip_x3".to_string(),
                literals: vec![-3],
                weight: 3.0,
                hard: false,
            },
        ],
    }
}

pub fn validate_weighted_max_sat_problem(
    p: &WeightedMaxSatProblem,
) -> Result<(), PreconditionError> {
    Preconditions::check(
        MODEL,
        "num_vars",
        "be positive",
        p.num_vars > 0,
        Some(p.num_vars.to_string()),
    )?;
    Preconditions::non_empty(MODEL, "clauses", &p.clauses)?;
    let mut ids = HashSet::new();
    for (clause_idx, clause) in p.clauses.iter().enumerate() {
        Preconditions::check(
            MODEL,
            &format!("clauses[{clause_idx}].id"),
            "be non-empty",
            !clause.id.trim().is_empty(),
            Some(clause.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("clauses[{clause_idx}].id"),
            "be unique",
            ids.insert(clause.id.clone()),
            Some(clause.id.clone()),
        )?;
        Preconditions::non_empty(
            MODEL,
            &format!("clauses[{clause_idx}].literals"),
            &clause.literals,
        )?;
        Preconditions::check(
            MODEL,
            &format!("clauses[{clause_idx}].weight"),
            "be finite and non-negative",
            clause.weight.is_finite() && clause.weight >= 0.0,
            Some(clause.weight.to_string()),
        )?;
        for &literal in &clause.literals {
            let variable = literal.unsigned_abs() as usize;
            Preconditions::check(
                MODEL,
                &format!("clauses[{clause_idx}].literal"),
                "refer to a variable in [1, num_vars]",
                literal != 0 && variable >= 1 && variable <= p.num_vars,
                Some(literal.to_string()),
            )?;
        }
    }
    Ok(())
}

pub fn solve_weighted_max_sat_greedy(p: &WeightedMaxSatProblem) -> WeightedMaxSatSolution {
    validate_weighted_max_sat_problem(p).expect("weighted-max-sat: invalid problem instance");
    let mut assignment = vec![false; p.num_vars];
    let mut current = evaluate_assignment(p, &assignment);
    let mut improved = true;
    let max_sweeps = p.num_vars.saturating_mul(p.clauses.len()).max(1);
    for _ in 0..max_sweeps {
        if !improved {
            break;
        }
        improved = false;
        let mut best_var = None;
        let mut best_eval = current.clone();
        for var in 0..p.num_vars {
            let mut trial = assignment.clone();
            trial[var] = !trial[var];
            let trial_eval = evaluate_assignment(p, &trial);
            if eval_better(&trial_eval, &best_eval) {
                best_var = Some(var);
                best_eval = trial_eval;
            }
        }
        if let Some(var) = best_var {
            assignment[var] = !assignment[var];
            current = best_eval;
            improved = true;
        }
    }
    build_solution(
        if current.violated_hard_clause_ids.is_empty() {
            WeightedMaxSatStatus::Feasible
        } else {
            WeightedMaxSatStatus::Infeasible
        },
        assignment,
        current,
        "greedy coordinate-improvement weighted Max-SAT",
    )
}

pub fn solve_weighted_max_sat_exact(p: &WeightedMaxSatProblem) -> WeightedMaxSatSolution {
    validate_weighted_max_sat_problem(p).expect("weighted-max-sat: invalid problem instance");
    if p.num_vars > MAX_EXACT_VARS {
        return WeightedMaxSatSolution {
            status: WeightedMaxSatStatus::Unsupported,
            assignment: Vec::new(),
            satisfied_soft_weight: None,
            unsatisfied_soft_weight: None,
            satisfied_clause_ids: Vec::new(),
            violated_hard_clause_ids: Vec::new(),
            message: format!(
                "exact weighted Max-SAT only practical for <= {MAX_EXACT_VARS} variables, got {}",
                p.num_vars
            ),
        };
    }

    let mut best_assignment = None;
    let mut best_eval = None;
    let total = 1usize << p.num_vars;
    for mask in 0..total {
        let assignment = (0..p.num_vars)
            .map(|var| ((mask >> var) & 1) == 1)
            .collect::<Vec<_>>();
        let eval = evaluate_assignment(p, &assignment);
        if !eval.violated_hard_clause_ids.is_empty() {
            continue;
        }
        if best_eval
            .as_ref()
            .is_none_or(|current| eval_better(&eval, current))
        {
            best_assignment = Some(assignment);
            best_eval = Some(eval);
        }
    }

    match (best_assignment, best_eval) {
        (Some(assignment), Some(eval)) => build_solution(
            WeightedMaxSatStatus::Optimal,
            assignment,
            eval,
            "exact weighted Max-SAT enumeration",
        ),
        _ => WeightedMaxSatSolution {
            status: WeightedMaxSatStatus::Infeasible,
            assignment: Vec::new(),
            satisfied_soft_weight: None,
            unsatisfied_soft_weight: None,
            satisfied_clause_ids: Vec::new(),
            violated_hard_clause_ids: p
                .clauses
                .iter()
                .filter(|clause| clause.hard)
                .map(|clause| clause.id.clone())
                .collect(),
            message: "no assignment satisfies all hard clauses".to_string(),
        },
    }
}

pub fn weighted_max_sat_solution_feasible(
    p: &WeightedMaxSatProblem,
    solution: &WeightedMaxSatSolution,
) -> bool {
    if validate_weighted_max_sat_problem(p).is_err()
        || solution.assignment.len() != p.num_vars
        || !solution.violated_hard_clause_ids.is_empty()
    {
        return false;
    }
    let eval = evaluate_assignment(p, &solution.assignment);
    eval.violated_hard_clause_ids.is_empty()
        && solution.satisfied_clause_ids == eval.satisfied_clause_ids
        && close_opt(
            solution.satisfied_soft_weight,
            Some(eval.satisfied_soft_weight),
        )
        && close_opt(
            solution.unsatisfied_soft_weight,
            Some(eval.unsatisfied_soft_weight),
        )
}

#[derive(Clone, Debug)]
struct AssignmentEvaluation {
    satisfied_soft_weight: f64,
    unsatisfied_soft_weight: f64,
    satisfied_clause_ids: Vec<String>,
    violated_hard_clause_ids: Vec<String>,
}

fn evaluate_assignment(p: &WeightedMaxSatProblem, assignment: &[bool]) -> AssignmentEvaluation {
    let mut satisfied_soft_weight = 0.0;
    let mut unsatisfied_soft_weight = 0.0;
    let mut satisfied_clause_ids = Vec::new();
    let mut violated_hard_clause_ids = Vec::new();
    for clause in &p.clauses {
        if clause_satisfied(clause, assignment) {
            satisfied_clause_ids.push(clause.id.clone());
            if !clause.hard {
                satisfied_soft_weight += clause.weight;
            }
        } else if clause.hard {
            violated_hard_clause_ids.push(clause.id.clone());
        } else {
            unsatisfied_soft_weight += clause.weight;
        }
    }
    AssignmentEvaluation {
        satisfied_soft_weight,
        unsatisfied_soft_weight,
        satisfied_clause_ids,
        violated_hard_clause_ids,
    }
}

fn eval_better(candidate: &AssignmentEvaluation, incumbent: &AssignmentEvaluation) -> bool {
    let hard_cmp =
        candidate.violated_hard_clause_ids.len() < incumbent.violated_hard_clause_ids.len();
    let hard_same =
        candidate.violated_hard_clause_ids.len() == incumbent.violated_hard_clause_ids.len();
    hard_cmp
        || (hard_same && candidate.satisfied_soft_weight > incumbent.satisfied_soft_weight + EPS)
        || (hard_same
            && (candidate.satisfied_soft_weight - incumbent.satisfied_soft_weight).abs() <= EPS
            && candidate.unsatisfied_soft_weight < incumbent.unsatisfied_soft_weight - EPS)
}

fn build_solution(
    status: WeightedMaxSatStatus,
    assignment: Vec<bool>,
    eval: AssignmentEvaluation,
    message: &str,
) -> WeightedMaxSatSolution {
    WeightedMaxSatSolution {
        status,
        assignment,
        satisfied_soft_weight: Some(eval.satisfied_soft_weight),
        unsatisfied_soft_weight: Some(eval.unsatisfied_soft_weight),
        satisfied_clause_ids: eval.satisfied_clause_ids,
        violated_hard_clause_ids: eval.violated_hard_clause_ids,
        message: message.to_string(),
    }
}

fn clause_satisfied(clause: &WeightedMaxSatClause, assignment: &[bool]) -> bool {
    clause
        .literals
        .iter()
        .any(|&literal| literal_satisfied(literal, assignment))
}

fn literal_satisfied(literal: i64, assignment: &[bool]) -> bool {
    let value = assignment[(literal.unsigned_abs() as usize) - 1];
    if literal > 0 {
        value
    } else {
        !value
    }
}

fn close_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= EPS * 1.0_f64.max(a.abs()).max(b.abs()),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_weighted_max_sat_finds_sample_optimum() {
        let p = build_sample_weighted_max_sat_problem();
        let exact = solve_weighted_max_sat_exact(&p);
        assert_eq!(exact.status, WeightedMaxSatStatus::Optimal);
        assert_eq!(exact.satisfied_soft_weight, Some(16.0));
        assert!(weighted_max_sat_solution_feasible(&p, &exact));
    }

    #[test]
    fn greedy_weighted_max_sat_returns_feasible_assignment() {
        let p = build_sample_weighted_max_sat_problem();
        let greedy = solve_weighted_max_sat_greedy(&p);
        assert_eq!(greedy.status, WeightedMaxSatStatus::Feasible);
        assert!(weighted_max_sat_solution_feasible(&p, &greedy));
    }
}
