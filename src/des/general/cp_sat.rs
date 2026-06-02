//! Small CP-SAT-style finite-domain constraint solver.
//!
//! This is a modelling surface rather than another LP/MIP wrapper: integer
//! domains, `all_different`, boolean clauses, common logical constraints, and
//! linear integer constraints are the kind of constraints users reach for in
//! OR-Tools CP-SAT. The engine below is exact branch-and-bound for small models,
//! with simple partial-consistency pruning and an optional linear objective.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpVariable {
    pub name: String,
    pub domain: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTerm {
    pub var: usize,
    pub coeff: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSense {
    Le,
    Ge,
    Eq,
}

impl LinearSense {
    pub fn as_str(self) -> &'static str {
        match self {
            LinearSense::Le => "le",
            LinearSense::Ge => "ge",
            LinearSense::Eq => "eq",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoolLiteral {
    pub var: usize,
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpInterval {
    pub start: usize,
    pub duration: i64,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpDemandInterval {
    pub start: usize,
    pub duration: i64,
    pub demand: i64,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpElement {
    pub index: usize,
    pub values: Vec<i64>,
    pub target: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpConstraint {
    Linear {
        terms: Vec<LinearTerm>,
        sense: LinearSense,
        rhs: i64,
    },
    EnforcedLinear {
        enforcement: Vec<BoolLiteral>,
        terms: Vec<LinearTerm>,
        sense: LinearSense,
        rhs: i64,
    },
    AllDifferent(Vec<usize>),
    BoolOr(Vec<BoolLiteral>),
    AtMostOne(Vec<BoolLiteral>),
    ExactlyOne(Vec<BoolLiteral>),
    Implication {
        antecedent: BoolLiteral,
        consequent: BoolLiteral,
    },
    AllowedAssignments {
        vars: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    Element(CpElement),
    NoOverlap(Vec<CpInterval>),
    Cumulative {
        intervals: Vec<CpDemandInterval>,
        capacity: i64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Min,
    Max,
}

impl ObjectiveSense {
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectiveSense::Min => "min",
            ObjectiveSense::Max => "max",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpObjective {
    pub sense: ObjectiveSense,
    pub terms: Vec<LinearTerm>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpModel {
    pub variables: Vec<CpVariable>,
    pub constraints: Vec<CpConstraint>,
    pub objective: Option<CpObjective>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpStatus {
    Optimal,
    Feasible,
    Infeasible,
}

impl CpStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CpStatus::Optimal => "optimal",
            CpStatus::Feasible => "feasible",
            CpStatus::Infeasible => "infeasible",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpSolution {
    pub status: CpStatus,
    pub assignment: Vec<i64>,
    pub objective: Option<i64>,
    pub nodes: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct CpSolveOptions {
    pub max_nodes: usize,
}

impl Default for CpSolveOptions {
    fn default() -> Self {
        CpSolveOptions {
            max_nodes: 1_000_000,
        }
    }
}

fn validate_model(model: &CpModel) {
    if model.variables.is_empty() {
        panic!("cp-sat: variables must be non-empty");
    }
    for (i, var) in model.variables.iter().enumerate() {
        if var.domain.is_empty() {
            panic!("cp-sat: variable {i} has an empty domain");
        }
        let mut d = var.domain.clone();
        d.sort_unstable();
        d.dedup();
        if d.len() != var.domain.len() {
            panic!("cp-sat: variable {i} domain contains duplicate values");
        }
    }
    let n = model.variables.len();
    let check_var = |idx: usize| {
        if idx >= n {
            panic!("cp-sat: variable index {idx} out of range {n}");
        }
    };
    let check_bool_literal = |lit: &BoolLiteral| {
        check_var(lit.var);
        if model.variables[lit.var]
            .domain
            .iter()
            .any(|&v| v != 0 && v != 1)
        {
            panic!(
                "cp-sat: boolean literal variable {} is not boolean",
                lit.var
            );
        }
    };
    for constraint in &model.constraints {
        match constraint {
            CpConstraint::Linear { terms, .. } => {
                if terms.is_empty() {
                    panic!("cp-sat: linear constraint has no terms");
                }
                for t in terms {
                    check_var(t.var);
                }
            }
            CpConstraint::EnforcedLinear {
                enforcement, terms, ..
            } => {
                if enforcement.is_empty() {
                    panic!("cp-sat: enforced_linear has no enforcement literals");
                }
                if terms.is_empty() {
                    panic!("cp-sat: enforced_linear constraint has no terms");
                }
                for lit in enforcement {
                    check_bool_literal(lit);
                }
                for t in terms {
                    check_var(t.var);
                }
            }
            CpConstraint::AllDifferent(vars) => {
                if vars.is_empty() {
                    panic!("cp-sat: all_different has no variables");
                }
                for &v in vars {
                    check_var(v);
                }
            }
            CpConstraint::BoolOr(lits) => {
                if lits.is_empty() {
                    panic!("cp-sat: bool_or has no literals");
                }
                for lit in lits {
                    check_bool_literal(lit);
                }
            }
            CpConstraint::AtMostOne(lits) => {
                if lits.is_empty() {
                    panic!("cp-sat: at_most_one has no literals");
                }
                for lit in lits {
                    check_bool_literal(lit);
                }
            }
            CpConstraint::ExactlyOne(lits) => {
                if lits.is_empty() {
                    panic!("cp-sat: exactly_one has no literals");
                }
                for lit in lits {
                    check_bool_literal(lit);
                }
            }
            CpConstraint::Implication {
                antecedent,
                consequent,
            } => {
                check_bool_literal(antecedent);
                check_bool_literal(consequent);
            }
            CpConstraint::AllowedAssignments { vars, tuples } => {
                if vars.is_empty() {
                    panic!("cp-sat: allowed_assignments has no variables");
                }
                if tuples.is_empty() {
                    panic!("cp-sat: allowed_assignments has no tuples");
                }
                for &v in vars {
                    check_var(v);
                }
                for (i, tuple) in tuples.iter().enumerate() {
                    if tuple.len() != vars.len() {
                        panic!(
                            "cp-sat: allowed_assignments tuple {i} length {} != {}",
                            tuple.len(),
                            vars.len()
                        );
                    }
                }
            }
            CpConstraint::Element(element) => {
                check_var(element.index);
                check_var(element.target);
                if element.values.is_empty() {
                    panic!("cp-sat: element has no values");
                }
            }
            CpConstraint::NoOverlap(intervals) => {
                if intervals.is_empty() {
                    panic!("cp-sat: no_overlap has no intervals");
                }
                for interval in intervals {
                    check_var(interval.start);
                    if interval.duration <= 0 {
                        panic!("cp-sat: interval duration must be positive");
                    }
                }
            }
            CpConstraint::Cumulative {
                intervals,
                capacity,
            } => {
                if *capacity < 0 {
                    panic!("cp-sat: cumulative capacity must be non-negative");
                }
                if intervals.is_empty() {
                    panic!("cp-sat: cumulative has no intervals");
                }
                for interval in intervals {
                    check_var(interval.start);
                    if interval.duration <= 0 {
                        panic!("cp-sat: cumulative interval duration must be positive");
                    }
                    if interval.demand <= 0 {
                        panic!("cp-sat: cumulative interval demand must be positive");
                    }
                }
            }
        }
    }
    if let Some(obj) = &model.objective {
        if obj.terms.is_empty() {
            panic!("cp-sat: objective has no terms");
        }
        for t in &obj.terms {
            check_var(t.var);
        }
    }
}

fn term_min_max(model: &CpModel, assignment: &[Option<i64>], term: &LinearTerm) -> (i64, i64) {
    if let Some(value) = assignment[term.var] {
        let v = value * term.coeff;
        return (v, v);
    }
    let domain = &model.variables[term.var].domain;
    let lo = *domain.iter().min().unwrap();
    let hi = *domain.iter().max().unwrap();
    if term.coeff >= 0 {
        (term.coeff * lo, term.coeff * hi)
    } else {
        (term.coeff * hi, term.coeff * lo)
    }
}

fn partial_linear_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    terms: &[LinearTerm],
    sense: LinearSense,
    rhs: i64,
) -> bool {
    let mut min_lhs = 0i64;
    let mut max_lhs = 0i64;
    for term in terms {
        let (lo, hi) = term_min_max(model, assignment, term);
        min_lhs += lo;
        max_lhs += hi;
    }
    match sense {
        LinearSense::Le => min_lhs <= rhs,
        LinearSense::Ge => max_lhs >= rhs,
        LinearSense::Eq => min_lhs <= rhs && rhs <= max_lhs,
    }
}

fn partial_all_different_ok(assignment: &[Option<i64>], vars: &[usize]) -> bool {
    let mut seen = Vec::new();
    for &v in vars {
        if let Some(value) = assignment[v] {
            if seen.contains(&value) {
                return false;
            }
            seen.push(value);
        }
    }
    true
}

fn literal_value(assignment: &[Option<i64>], lit: &BoolLiteral) -> Option<bool> {
    assignment[lit.var].map(|v| if lit.positive { v == 1 } else { v == 0 })
}

fn enforcement_literals_active(
    assignment: &[Option<i64>],
    enforcement: &[BoolLiteral],
) -> Option<bool> {
    let mut has_unknown = false;
    for lit in enforcement {
        match literal_value(assignment, lit) {
            Some(false) => return Some(false),
            Some(true) => {}
            None => has_unknown = true,
        }
    }
    if has_unknown {
        None
    } else {
        Some(true)
    }
}

fn partial_bool_or_ok(assignment: &[Option<i64>], lits: &[BoolLiteral]) -> bool {
    let mut has_unknown = false;
    for lit in lits {
        match literal_value(assignment, lit) {
            Some(true) => return true,
            Some(false) => {}
            None => has_unknown = true,
        }
    }
    has_unknown
}

fn partial_at_most_one_ok(assignment: &[Option<i64>], lits: &[BoolLiteral]) -> bool {
    lits.iter()
        .filter(|lit| literal_value(assignment, lit) == Some(true))
        .count()
        <= 1
}

fn partial_exactly_one_ok(assignment: &[Option<i64>], lits: &[BoolLiteral]) -> bool {
    let true_count = lits
        .iter()
        .filter(|lit| literal_value(assignment, lit) == Some(true))
        .count();
    if true_count > 1 {
        return false;
    }
    true_count == 1
        || lits
            .iter()
            .any(|lit| literal_value(assignment, lit).is_none())
}

fn partial_implication_ok(
    assignment: &[Option<i64>],
    antecedent: &BoolLiteral,
    consequent: &BoolLiteral,
) -> bool {
    !(literal_value(assignment, antecedent) == Some(true)
        && literal_value(assignment, consequent) == Some(false))
}

fn partial_allowed_assignments_ok(
    assignment: &[Option<i64>],
    vars: &[usize],
    tuples: &[Vec<i64>],
) -> bool {
    tuples.iter().any(|tuple| {
        vars.iter().zip(tuple).all(|(&var, &value)| {
            assignment[var]
                .map(|actual| actual == value)
                .unwrap_or(true)
        })
    })
}

fn partial_element_ok(model: &CpModel, assignment: &[Option<i64>], element: &CpElement) -> bool {
    match (assignment[element.index], assignment[element.target]) {
        (Some(index), Some(target)) => {
            let Ok(index) = usize::try_from(index) else {
                return false;
            };
            element
                .values
                .get(index)
                .map(|&value| value == target)
                .unwrap_or(false)
        }
        (Some(index), None) => {
            let Ok(index) = usize::try_from(index) else {
                return false;
            };
            element.values.get(index).is_some()
        }
        (None, Some(target)) => model.variables[element.index]
            .domain
            .iter()
            .filter_map(|&index| usize::try_from(index).ok())
            .filter_map(|index| element.values.get(index))
            .any(|&value| value == target),
        (None, None) => model.variables[element.index]
            .domain
            .iter()
            .filter_map(|&index| usize::try_from(index).ok())
            .any(|index| index < element.values.len()),
    }
}

fn partial_no_overlap_ok(assignment: &[Option<i64>], intervals: &[CpInterval]) -> bool {
    for i in 0..intervals.len() {
        let Some(start_i) = assignment[intervals[i].start] else {
            continue;
        };
        let end_i = start_i + intervals[i].duration;
        for j in (i + 1)..intervals.len() {
            let Some(start_j) = assignment[intervals[j].start] else {
                continue;
            };
            let end_j = start_j + intervals[j].duration;
            if !(end_i <= start_j || end_j <= start_i) {
                return false;
            }
        }
    }
    true
}

fn partial_cumulative_ok(
    assignment: &[Option<i64>],
    intervals: &[CpDemandInterval],
    capacity: i64,
) -> bool {
    let mut assigned = Vec::new();
    for interval in intervals {
        let Some(start) = assignment[interval.start] else {
            continue;
        };
        assigned.push((start, start + interval.duration, interval.demand));
    }
    if assigned.is_empty() {
        return true;
    }
    let mut time_points = Vec::with_capacity(assigned.len() * 2);
    for &(start, end, _) in &assigned {
        time_points.push(start);
        time_points.push(end);
    }
    time_points.sort_unstable();
    time_points.dedup();
    for t in time_points {
        let load: i64 = assigned
            .iter()
            .filter(|&&(start, end, _)| start <= t && t < end)
            .map(|&(_, _, demand)| demand)
            .sum();
        if load > capacity {
            return false;
        }
    }
    true
}

fn partial_constraints_ok(model: &CpModel, assignment: &[Option<i64>]) -> bool {
    for constraint in &model.constraints {
        let ok = match constraint {
            CpConstraint::Linear { terms, sense, rhs } => {
                partial_linear_ok(model, assignment, terms, *sense, *rhs)
            }
            CpConstraint::EnforcedLinear {
                enforcement,
                terms,
                sense,
                rhs,
            } => match enforcement_literals_active(assignment, enforcement) {
                Some(false) => true,
                Some(true) => partial_linear_ok(model, assignment, terms, *sense, *rhs),
                None => true,
            },
            CpConstraint::AllDifferent(vars) => partial_all_different_ok(assignment, vars),
            CpConstraint::BoolOr(lits) => partial_bool_or_ok(assignment, lits),
            CpConstraint::AtMostOne(lits) => partial_at_most_one_ok(assignment, lits),
            CpConstraint::ExactlyOne(lits) => partial_exactly_one_ok(assignment, lits),
            CpConstraint::Implication {
                antecedent,
                consequent,
            } => partial_implication_ok(assignment, antecedent, consequent),
            CpConstraint::AllowedAssignments { vars, tuples } => {
                partial_allowed_assignments_ok(assignment, vars, tuples)
            }
            CpConstraint::Element(element) => partial_element_ok(model, assignment, element),
            CpConstraint::NoOverlap(intervals) => partial_no_overlap_ok(assignment, intervals),
            CpConstraint::Cumulative {
                intervals,
                capacity,
            } => partial_cumulative_ok(assignment, intervals, *capacity),
        };
        if !ok {
            return false;
        }
    }
    true
}

fn objective_value(obj: &CpObjective, full: &[i64]) -> i64 {
    obj.terms.iter().map(|t| t.coeff * full[t.var]).sum()
}

fn objective_bound(model: &CpModel, assignment: &[Option<i64>], obj: &CpObjective) -> i64 {
    obj.terms
        .iter()
        .map(|term| {
            let (lo, hi) = term_min_max(model, assignment, term);
            match obj.sense {
                ObjectiveSense::Min => lo,
                ObjectiveSense::Max => hi,
            }
        })
        .sum()
}

fn better(obj: &CpObjective, candidate: i64, incumbent: i64) -> bool {
    match obj.sense {
        ObjectiveSense::Min => candidate < incumbent,
        ObjectiveSense::Max => candidate > incumbent,
    }
}

fn bound_dominated(obj: &CpObjective, bound: i64, incumbent: i64) -> bool {
    match obj.sense {
        ObjectiveSense::Min => bound >= incumbent,
        ObjectiveSense::Max => bound <= incumbent,
    }
}

fn choose_variable(model: &CpModel, assignment: &[Option<i64>]) -> Option<usize> {
    model
        .variables
        .iter()
        .enumerate()
        .filter(|(i, _)| assignment[*i].is_none())
        .min_by_key(|(_, v)| v.domain.len())
        .map(|(i, _)| i)
}

/// Solve a small finite-domain CP-SAT-style model exactly.
pub fn solve_cp_model(model: &CpModel, opts: CpSolveOptions) -> CpSolution {
    validate_model(model);
    let n = model.variables.len();
    let mut assignment = vec![None; n];
    let mut best_assignment = Vec::new();
    let mut best_objective: Option<i64> = None;
    let mut nodes = 0usize;
    let mut hit_limit = false;

    fn dfs(
        model: &CpModel,
        opts: CpSolveOptions,
        assignment: &mut [Option<i64>],
        best_assignment: &mut Vec<i64>,
        best_objective: &mut Option<i64>,
        nodes: &mut usize,
        hit_limit: &mut bool,
    ) {
        if *hit_limit {
            return;
        }
        *nodes += 1;
        if *nodes > opts.max_nodes {
            *hit_limit = true;
            return;
        }
        if !partial_constraints_ok(model, assignment) {
            return;
        }
        if let (Some(obj), Some(inc)) = (&model.objective, *best_objective) {
            let bound = objective_bound(model, assignment, obj);
            if bound_dominated(obj, bound, inc) {
                return;
            }
        }
        let Some(var) = choose_variable(model, assignment) else {
            let full: Vec<i64> = assignment.iter().map(|v| v.unwrap()).collect();
            let candidate_objective = model
                .objective
                .as_ref()
                .map(|obj| objective_value(obj, &full));
            let should_replace = match (&model.objective, *best_objective, candidate_objective) {
                (None, _, _) => best_assignment.is_empty(),
                (Some(_), None, Some(_)) => true,
                (Some(obj), Some(inc), Some(value)) => better(obj, value, inc),
                _ => false,
            };
            if should_replace {
                *best_assignment = full;
                *best_objective = candidate_objective;
            }
            return;
        };
        for value in model.variables[var].domain.clone() {
            assignment[var] = Some(value);
            dfs(
                model,
                opts,
                assignment,
                best_assignment,
                best_objective,
                nodes,
                hit_limit,
            );
            assignment[var] = None;
        }
    }

    dfs(
        model,
        opts,
        &mut assignment,
        &mut best_assignment,
        &mut best_objective,
        &mut nodes,
        &mut hit_limit,
    );

    if best_assignment.is_empty() {
        CpSolution {
            status: CpStatus::Infeasible,
            assignment: Vec::new(),
            objective: None,
            nodes,
            solver: "internal-cp-enumeration".to_string(),
            message: if hit_limit {
                Some("node limit reached before finding a feasible assignment".to_string())
            } else {
                Some("no feasible assignment".to_string())
            },
        }
    } else {
        CpSolution {
            status: if model.objective.is_some() && !hit_limit {
                CpStatus::Optimal
            } else {
                CpStatus::Feasible
            },
            assignment: best_assignment,
            objective: best_objective,
            nodes,
            solver: "internal-cp-enumeration".to_string(),
            message: Some("exact finite-domain branch-and-bound".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_assignment_with_all_different() {
        let model = CpModel {
            variables: (0..3)
                .map(|i| CpVariable {
                    name: format!("worker_{i}"),
                    domain: vec![0, 1, 2],
                })
                .collect(),
            constraints: vec![CpConstraint::AllDifferent(vec![0, 1, 2])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 8 },
                    LinearTerm { var: 1, coeff: 2 },
                    LinearTerm { var: 2, coeff: 5 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 2, 1]);
        assert_eq!(sol.objective, Some(9));
    }

    #[test]
    fn solves_no_overlap_schedule() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
            ],
            constraints: vec![CpConstraint::NoOverlap(vec![
                CpInterval {
                    start: 0,
                    duration: 3,
                    name: Some("task_a".to_string()),
                },
                CpInterval {
                    start: 1,
                    duration: 2,
                    name: Some("task_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![2, 0]);
        assert_eq!(sol.objective, Some(2));
    }

    #[test]
    fn solves_cumulative_resource_schedule() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "machine_a_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "machine_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "machine_c_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
            ],
            constraints: vec![CpConstraint::Cumulative {
                intervals: vec![
                    CpDemandInterval {
                        start: 0,
                        duration: 3,
                        demand: 2,
                        name: Some("machine_a".to_string()),
                    },
                    CpDemandInterval {
                        start: 1,
                        duration: 2,
                        demand: 2,
                        name: Some("machine_b".to_string()),
                    },
                    CpDemandInterval {
                        start: 2,
                        duration: 2,
                        demand: 1,
                        name: Some("machine_c".to_string()),
                    },
                ],
                capacity: 3,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                    LinearTerm { var: 2, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![2, 0, 0]);
        assert_eq!(sol.objective, Some(2));
    }

    #[test]
    fn solves_element_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "route".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "cost".to_string(),
                    domain: vec![3, 8],
                },
            ],
            constraints: vec![CpConstraint::Element(CpElement {
                index: 0,
                values: vec![3, 8],
                target: 1,
            })],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 1, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 3]);
        assert_eq!(sol.objective, Some(3));
    }

    #[test]
    fn solves_allowed_assignments_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "mode".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "handler".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::AllowedAssignments {
                vars: vec![0, 1],
                tuples: vec![vec![0, 1], vec![1, 0]],
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 2 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 1]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_active_enforced_linear_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "use_rule".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "load".to_string(),
                    domain: vec![0, 1, 2, 3, 4],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::EnforcedLinear {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 1, coeff: 1 }],
                    sense: LinearSense::Ge,
                    rhs: 3,
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 1, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 3]);
        assert_eq!(sol.objective, Some(3));
    }

    #[test]
    fn skips_inactive_enforced_linear_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "use_rule".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "load".to_string(),
                    domain: vec![0, 1, 2, 3, 4],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: false,
                }]),
                CpConstraint::EnforcedLinear {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 1, coeff: 1 }],
                    sense: LinearSense::Ge,
                    rhs: 3,
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 1, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 0]);
        assert_eq!(sol.objective, Some(0));
    }
}
