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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpSolutionHint {
    pub var: usize,
    pub value: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpVariableSelectionStrategy {
    First,
    MinDomainSize,
    MaxDomainSize,
    LowestMin,
    HighestMax,
}

impl CpVariableSelectionStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CpVariableSelectionStrategy::First => "first",
            CpVariableSelectionStrategy::MinDomainSize => "min_domain_size",
            CpVariableSelectionStrategy::MaxDomainSize => "max_domain_size",
            CpVariableSelectionStrategy::LowestMin => "lowest_min",
            CpVariableSelectionStrategy::HighestMax => "highest_max",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpDomainValueStrategy {
    MinValue,
    MaxValue,
}

impl CpDomainValueStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CpDomainValueStrategy::MinValue => "min_value",
            CpDomainValueStrategy::MaxValue => "max_value",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpDecisionStrategy {
    pub vars: Vec<usize>,
    pub variable_strategy: CpVariableSelectionStrategy,
    pub domain_strategy: CpDomainValueStrategy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpDomainInterval {
    pub lb: i64,
    pub ub: i64,
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
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpVariableInterval {
    pub start: usize,
    pub duration: usize,
    pub end: usize,
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpAlternative {
    pub start: usize,
    pub duration: usize,
    pub end: usize,
    pub presence: Option<BoolLiteral>,
    pub alternatives: Vec<CpVariableInterval>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpDemandInterval {
    pub start: usize,
    pub duration: i64,
    pub demand: i64,
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpVariableDemandInterval {
    pub start: usize,
    pub duration: usize,
    pub end: usize,
    pub demand: usize,
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpReservoirEvent {
    pub time: usize,
    pub level_change: i64,
    pub active: Option<BoolLiteral>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpRectangle {
    pub x_start: usize,
    pub y_start: usize,
    pub width: i64,
    pub height: i64,
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpVariableRectangle {
    pub x_start: usize,
    pub x_size: usize,
    pub x_end: usize,
    pub y_start: usize,
    pub y_size: usize,
    pub y_end: usize,
    pub presence: Option<BoolLiteral>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpElement {
    pub index: usize,
    pub values: Vec<i64>,
    pub target: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpVariableElement {
    pub index: usize,
    pub vars: Vec<usize>,
    pub target: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpTransition {
    pub tail: i64,
    pub label: i64,
    pub head: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpAutomaton {
    pub vars: Vec<usize>,
    pub starting_state: i64,
    pub final_states: Vec<i64>,
    pub transitions: Vec<CpTransition>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpCircuitArc {
    pub tail: i64,
    pub head: i64,
    pub literal: BoolLiteral,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpConstraint {
    Linear {
        terms: Vec<LinearTerm>,
        sense: LinearSense,
        rhs: i64,
    },
    LinearDomain {
        terms: Vec<LinearTerm>,
        intervals: Vec<CpDomainInterval>,
    },
    MapDomain {
        var: usize,
        bools: Vec<usize>,
        offset: i64,
    },
    EnforcedLinear {
        enforcement: Vec<BoolLiteral>,
        terms: Vec<LinearTerm>,
        sense: LinearSense,
        rhs: i64,
    },
    EnforcedBoolOr {
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    },
    EnforcedBoolAnd {
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    },
    EnforcedBoolXor {
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    },
    EnforcedAtMostOne {
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    },
    EnforcedExactlyOne {
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    },
    AllDifferent(Vec<usize>),
    BoolOr(Vec<BoolLiteral>),
    BoolAnd(Vec<BoolLiteral>),
    BoolXor(Vec<BoolLiteral>),
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
    ForbiddenAssignments {
        vars: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    EnforcedAllowedAssignments {
        enforcement: Vec<BoolLiteral>,
        vars: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    EnforcedForbiddenAssignments {
        enforcement: Vec<BoolLiteral>,
        vars: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    Inverse {
        direct: Vec<usize>,
        inverse: Vec<usize>,
    },
    MaxEquality {
        target: usize,
        vars: Vec<usize>,
    },
    MinEquality {
        target: usize,
        vars: Vec<usize>,
    },
    AbsEquality {
        target: usize,
        var: usize,
    },
    MultiplicationEquality {
        target: usize,
        vars: Vec<usize>,
    },
    DivisionEquality {
        target: usize,
        numerator: usize,
        denominator: usize,
    },
    ModuloEquality {
        target: usize,
        var: usize,
        modulus: usize,
    },
    Automaton(CpAutomaton),
    Circuit(Vec<CpCircuitArc>),
    MultipleCircuit(Vec<CpCircuitArc>),
    Element(CpElement),
    VariableElement(CpVariableElement),
    Alternative(CpAlternative),
    NoOverlap(Vec<CpInterval>),
    NoOverlapVariable(Vec<CpVariableInterval>),
    NoOverlap2D(Vec<CpRectangle>),
    NoOverlap2DVariable(Vec<CpVariableRectangle>),
    Cumulative {
        intervals: Vec<CpDemandInterval>,
        capacity: i64,
    },
    CumulativeVariable {
        intervals: Vec<CpVariableDemandInterval>,
        capacity: usize,
    },
    Reservoir {
        events: Vec<CpReservoirEvent>,
        min_level: i64,
        max_level: i64,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpSolutionPoolEntry {
    pub assignment: Vec<i64>,
    pub objective: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpSolutionPool {
    pub status: CpStatus,
    pub solutions: Vec<CpSolutionPoolEntry>,
    pub nodes: usize,
    pub exhausted: bool,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpAssumptionCore {
    pub status: CpStatus,
    pub assumptions: Vec<BoolLiteral>,
    pub minimal: bool,
    pub checks: usize,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CpSolveOptions {
    pub max_nodes: usize,
    pub solution_hint: Vec<CpSolutionHint>,
    pub decision_strategies: Vec<CpDecisionStrategy>,
}

#[derive(Clone, Copy, Debug)]
pub struct CpEnumerateOptions {
    pub max_nodes: usize,
    pub max_solutions: usize,
}

#[derive(Clone, Debug)]
pub struct CpAssumptionCoreOptions {
    pub solve_options: CpSolveOptions,
}

impl Default for CpSolveOptions {
    fn default() -> Self {
        CpSolveOptions {
            max_nodes: 1_000_000,
            solution_hint: Vec::new(),
            decision_strategies: Vec::new(),
        }
    }
}

impl Default for CpEnumerateOptions {
    fn default() -> Self {
        CpEnumerateOptions {
            max_nodes: 1_000_000,
            max_solutions: 100,
        }
    }
}

impl Default for CpAssumptionCoreOptions {
    fn default() -> Self {
        CpAssumptionCoreOptions {
            solve_options: CpSolveOptions::default(),
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
            CpConstraint::LinearDomain { terms, intervals } => {
                if terms.is_empty() {
                    panic!("cp-sat: linear_domain constraint has no terms");
                }
                if intervals.is_empty() {
                    panic!("cp-sat: linear_domain has no intervals");
                }
                for t in terms {
                    check_var(t.var);
                }
                for interval in intervals {
                    if interval.ub < interval.lb {
                        panic!("cp-sat: linear_domain interval ub must be >= lb");
                    }
                }
            }
            CpConstraint::MapDomain { var, bools, .. } => {
                check_var(*var);
                if bools.is_empty() {
                    panic!("cp-sat: map_domain has no selector variables");
                }
                for &b in bools {
                    check_var(b);
                    if model.variables[b].domain.iter().any(|&v| v != 0 && v != 1) {
                        panic!("cp-sat: map_domain selector variable {b} is not boolean");
                    }
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
            CpConstraint::EnforcedBoolOr {
                enforcement,
                literals,
            } => {
                if enforcement.is_empty() {
                    panic!("cp-sat: enforced_bool_or has no enforcement literals");
                }
                if literals.is_empty() {
                    panic!("cp-sat: enforced_bool_or has no literals");
                }
                for lit in enforcement {
                    check_bool_literal(lit);
                }
                for lit in literals {
                    check_bool_literal(lit);
                }
            }
            CpConstraint::EnforcedBoolAnd {
                enforcement,
                literals,
            } => {
                if enforcement.is_empty() {
                    panic!("cp-sat: enforced_bool_and has no enforcement literals");
                }
                if literals.is_empty() {
                    panic!("cp-sat: enforced_bool_and has no literals");
                }
                for lit in enforcement {
                    check_bool_literal(lit);
                }
                for lit in literals {
                    check_bool_literal(lit);
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
            CpConstraint::BoolAnd(lits) => {
                if lits.is_empty() {
                    panic!("cp-sat: bool_and has no literals");
                }
                for lit in lits {
                    check_bool_literal(lit);
                }
            }
            CpConstraint::BoolXor(lits) => {
                if lits.is_empty() {
                    panic!("cp-sat: bool_xor has no literals");
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
            CpConstraint::ForbiddenAssignments { vars, tuples } => {
                if vars.is_empty() {
                    panic!("cp-sat: forbidden_assignments has no variables");
                }
                if tuples.is_empty() {
                    panic!("cp-sat: forbidden_assignments has no tuples");
                }
                for &v in vars {
                    check_var(v);
                }
                for (i, tuple) in tuples.iter().enumerate() {
                    if tuple.len() != vars.len() {
                        panic!(
                            "cp-sat: forbidden_assignments tuple {i} length {} != {}",
                            tuple.len(),
                            vars.len()
                        );
                    }
                }
            }
            CpConstraint::EnforcedAllowedAssignments {
                enforcement,
                vars,
                tuples,
            } => {
                if enforcement.is_empty() {
                    panic!("cp-sat: enforced_allowed_assignments has no enforcement literals");
                }
                if vars.is_empty() {
                    panic!("cp-sat: enforced_allowed_assignments has no variables");
                }
                if tuples.is_empty() {
                    panic!("cp-sat: enforced_allowed_assignments has no tuples");
                }
                for lit in enforcement {
                    check_bool_literal(lit);
                }
                for &v in vars {
                    check_var(v);
                }
                for (i, tuple) in tuples.iter().enumerate() {
                    if tuple.len() != vars.len() {
                        panic!(
                            "cp-sat: enforced_allowed_assignments tuple {i} length {} != {}",
                            tuple.len(),
                            vars.len()
                        );
                    }
                }
            }
            CpConstraint::EnforcedForbiddenAssignments {
                enforcement,
                vars,
                tuples,
            } => {
                if enforcement.is_empty() {
                    panic!("cp-sat: enforced_forbidden_assignments has no enforcement literals");
                }
                if vars.is_empty() {
                    panic!("cp-sat: enforced_forbidden_assignments has no variables");
                }
                if tuples.is_empty() {
                    panic!("cp-sat: enforced_forbidden_assignments has no tuples");
                }
                for lit in enforcement {
                    check_bool_literal(lit);
                }
                for &v in vars {
                    check_var(v);
                }
                for (i, tuple) in tuples.iter().enumerate() {
                    if tuple.len() != vars.len() {
                        panic!(
                            "cp-sat: enforced_forbidden_assignments tuple {i} length {} != {}",
                            tuple.len(),
                            vars.len()
                        );
                    }
                }
            }
            CpConstraint::Inverse { direct, inverse } => {
                if direct.is_empty() {
                    panic!("cp-sat: inverse has no variables");
                }
                if direct.len() != inverse.len() {
                    panic!(
                        "cp-sat: inverse direct length {} != inverse length {}",
                        direct.len(),
                        inverse.len()
                    );
                }
                let n_values = direct.len() as i64;
                for &v in direct.iter().chain(inverse.iter()) {
                    check_var(v);
                    if model.variables[v]
                        .domain
                        .iter()
                        .any(|&value| value < 0 || value >= n_values)
                    {
                        panic!(
                            "cp-sat: inverse variable {v} domain must be within 0..{}",
                            n_values - 1
                        );
                    }
                }
            }
            CpConstraint::MaxEquality { target, vars } => {
                check_var(*target);
                if vars.is_empty() {
                    panic!("cp-sat: max_equality has no variables");
                }
                for &v in vars {
                    check_var(v);
                }
            }
            CpConstraint::MinEquality { target, vars } => {
                check_var(*target);
                if vars.is_empty() {
                    panic!("cp-sat: min_equality has no variables");
                }
                for &v in vars {
                    check_var(v);
                }
            }
            CpConstraint::AbsEquality { target, var } => {
                check_var(*target);
                check_var(*var);
                if model.variables[*target]
                    .domain
                    .iter()
                    .any(|&value| value < 0)
                {
                    panic!("cp-sat: abs_equality target domain must be non-negative");
                }
            }
            CpConstraint::MultiplicationEquality { target, vars } => {
                check_var(*target);
                if vars.is_empty() {
                    panic!("cp-sat: multiplication_equality has no variables");
                }
                for &v in vars {
                    check_var(v);
                }
            }
            CpConstraint::DivisionEquality {
                target,
                numerator,
                denominator,
            } => {
                check_var(*target);
                check_var(*numerator);
                check_var(*denominator);
                if model.variables[*denominator]
                    .domain
                    .iter()
                    .any(|&value| value <= 0)
                {
                    panic!("cp-sat: division_equality denominator domain must be positive");
                }
            }
            CpConstraint::ModuloEquality {
                target,
                var,
                modulus,
            } => {
                check_var(*target);
                check_var(*var);
                check_var(*modulus);
                if model.variables[*modulus]
                    .domain
                    .iter()
                    .any(|&value| value <= 0)
                {
                    panic!("cp-sat: modulo_equality modulus domain must be positive");
                }
            }
            CpConstraint::Automaton(automaton) => {
                if automaton.vars.is_empty() {
                    panic!("cp-sat: automaton has no variables");
                }
                if automaton.final_states.is_empty() {
                    panic!("cp-sat: automaton has no final states");
                }
                if automaton.transitions.is_empty() {
                    panic!("cp-sat: automaton has no transitions");
                }
                for &v in &automaton.vars {
                    check_var(v);
                }
            }
            CpConstraint::Circuit(arcs) => {
                if arcs.is_empty() {
                    panic!("cp-sat: circuit has no arcs");
                }
                for arc in arcs {
                    if arc.tail < 0 || arc.head < 0 {
                        panic!("cp-sat: circuit node ids must be non-negative");
                    }
                    check_bool_literal(&arc.literal);
                }
            }
            CpConstraint::MultipleCircuit(arcs) => {
                if arcs.is_empty() {
                    panic!("cp-sat: multiple_circuit has no arcs");
                }
                let mut has_depot = false;
                for arc in arcs {
                    if arc.tail < 0 || arc.head < 0 {
                        panic!("cp-sat: multiple_circuit node ids must be non-negative");
                    }
                    if arc.tail == 0 || arc.head == 0 {
                        has_depot = true;
                    }
                    if arc.tail == 0 && arc.head == 0 {
                        panic!("cp-sat: multiple_circuit depot self-loop is not allowed");
                    }
                    check_bool_literal(&arc.literal);
                }
                if !has_depot {
                    panic!("cp-sat: multiple_circuit requires depot node 0");
                }
            }
            CpConstraint::Element(element) => {
                check_var(element.index);
                check_var(element.target);
                if element.values.is_empty() {
                    panic!("cp-sat: element has no values");
                }
            }
            CpConstraint::VariableElement(element) => {
                check_var(element.index);
                check_var(element.target);
                if element.vars.is_empty() {
                    panic!("cp-sat: variable_element has no variables");
                }
                for &var in &element.vars {
                    check_var(var);
                }
            }
            CpConstraint::Alternative(alternative) => {
                check_var(alternative.start);
                check_var(alternative.duration);
                check_var(alternative.end);
                if let Some(presence) = &alternative.presence {
                    check_bool_literal(presence);
                }
                if model.variables[alternative.duration]
                    .domain
                    .iter()
                    .any(|&value| value <= 0)
                {
                    panic!(
                        "cp-sat: alternative duration variable {} must be positive",
                        alternative.duration
                    );
                }
                if alternative.alternatives.is_empty() {
                    panic!("cp-sat: alternative has no mode intervals");
                }
                for interval in &alternative.alternatives {
                    check_var(interval.start);
                    check_var(interval.duration);
                    check_var(interval.end);
                    let Some(presence) = &interval.presence else {
                        panic!("cp-sat: alternative mode interval must have a presence literal");
                    };
                    check_bool_literal(presence);
                    if model.variables[interval.duration]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: alternative mode duration variable {} must be positive",
                            interval.duration
                        );
                    }
                }
            }
            CpConstraint::NoOverlap(intervals) => {
                if intervals.is_empty() {
                    panic!("cp-sat: no_overlap has no intervals");
                }
                for interval in intervals {
                    check_var(interval.start);
                    if let Some(presence) = &interval.presence {
                        check_bool_literal(presence);
                    }
                    if interval.duration <= 0 {
                        panic!("cp-sat: interval duration must be positive");
                    }
                }
            }
            CpConstraint::NoOverlapVariable(intervals) => {
                if intervals.is_empty() {
                    panic!("cp-sat: no_overlap_variable has no intervals");
                }
                for interval in intervals {
                    check_var(interval.start);
                    check_var(interval.duration);
                    check_var(interval.end);
                    if let Some(presence) = &interval.presence {
                        check_bool_literal(presence);
                    }
                    if model.variables[interval.duration]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: variable interval duration variable {} must be positive",
                            interval.duration
                        );
                    }
                }
            }
            CpConstraint::NoOverlap2D(rectangles) => {
                if rectangles.is_empty() {
                    panic!("cp-sat: no_overlap_2d has no rectangles");
                }
                for rectangle in rectangles {
                    check_var(rectangle.x_start);
                    check_var(rectangle.y_start);
                    if let Some(presence) = &rectangle.presence {
                        check_bool_literal(presence);
                    }
                    if rectangle.width <= 0 || rectangle.height <= 0 {
                        panic!("cp-sat: rectangle dimensions must be positive");
                    }
                }
            }
            CpConstraint::NoOverlap2DVariable(rectangles) => {
                if rectangles.is_empty() {
                    panic!("cp-sat: no_overlap_2d_variable has no rectangles");
                }
                for rectangle in rectangles {
                    check_var(rectangle.x_start);
                    check_var(rectangle.x_size);
                    check_var(rectangle.x_end);
                    check_var(rectangle.y_start);
                    check_var(rectangle.y_size);
                    check_var(rectangle.y_end);
                    if let Some(presence) = &rectangle.presence {
                        check_bool_literal(presence);
                    }
                    if model.variables[rectangle.x_size]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: no_overlap_2d_variable x_size variable {} must be positive",
                            rectangle.x_size
                        );
                    }
                    if model.variables[rectangle.y_size]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: no_overlap_2d_variable y_size variable {} must be positive",
                            rectangle.y_size
                        );
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
                    if let Some(presence) = &interval.presence {
                        check_bool_literal(presence);
                    }
                    if interval.duration <= 0 {
                        panic!("cp-sat: cumulative interval duration must be positive");
                    }
                    if interval.demand <= 0 {
                        panic!("cp-sat: cumulative interval demand must be positive");
                    }
                }
            }
            CpConstraint::CumulativeVariable {
                intervals,
                capacity,
            } => {
                check_var(*capacity);
                if model.variables[*capacity]
                    .domain
                    .iter()
                    .any(|&value| value < 0)
                {
                    panic!("cp-sat: cumulative_variable capacity domain must be non-negative");
                }
                if intervals.is_empty() {
                    panic!("cp-sat: cumulative_variable has no intervals");
                }
                for interval in intervals {
                    check_var(interval.start);
                    check_var(interval.duration);
                    check_var(interval.end);
                    check_var(interval.demand);
                    if let Some(presence) = &interval.presence {
                        check_bool_literal(presence);
                    }
                    if model.variables[interval.duration]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: cumulative_variable duration variable {} must be positive",
                            interval.duration
                        );
                    }
                    if model.variables[interval.demand]
                        .domain
                        .iter()
                        .any(|&value| value <= 0)
                    {
                        panic!(
                            "cp-sat: cumulative_variable demand variable {} must be positive",
                            interval.demand
                        );
                    }
                }
            }
            CpConstraint::Reservoir {
                events,
                min_level,
                max_level,
            } => {
                if *max_level < *min_level {
                    panic!("cp-sat: reservoir max_level must be >= min_level");
                }
                if *min_level > 0 {
                    panic!("cp-sat: reservoir min_level must be <= 0");
                }
                if *max_level < 0 {
                    panic!("cp-sat: reservoir max_level must be >= 0");
                }
                if events.is_empty() {
                    panic!("cp-sat: reservoir has no events");
                }
                for event in events {
                    check_var(event.time);
                    if let Some(active) = &event.active {
                        check_bool_literal(active);
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

fn linear_min_max(model: &CpModel, assignment: &[Option<i64>], terms: &[LinearTerm]) -> (i64, i64) {
    let mut min_lhs = 0i64;
    let mut max_lhs = 0i64;
    for term in terms {
        let (lo, hi) = term_min_max(model, assignment, term);
        min_lhs += lo;
        max_lhs += hi;
    }
    (min_lhs, max_lhs)
}

fn partial_linear_domain_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    terms: &[LinearTerm],
    intervals: &[CpDomainInterval],
) -> bool {
    let (min_lhs, max_lhs) = linear_min_max(model, assignment, terms);
    intervals
        .iter()
        .any(|interval| interval.lb <= max_lhs && min_lhs <= interval.ub)
}

fn partial_map_domain_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    var: usize,
    bools: &[usize],
    offset: i64,
) -> bool {
    let var_value = assignment[var];
    let mut true_target = None;
    for (i, &bool_var) in bools.iter().enumerate() {
        let target = offset + i as i64;
        match assignment[bool_var] {
            Some(1) => {
                if true_target.is_some_and(|existing| existing != target) {
                    return false;
                }
                if var_value.is_some_and(|value| value != target) {
                    return false;
                }
                true_target = Some(target);
            }
            Some(0) => {
                if var_value == Some(target) {
                    return false;
                }
            }
            Some(_) => return false,
            None => {}
        }
    }

    if let Some(target) = true_target {
        return model.variables[var].domain.contains(&target);
    }
    if var_value.is_some() {
        return true;
    }

    model.variables[var].domain.iter().any(|&candidate| {
        bools
            .iter()
            .enumerate()
            .all(|(i, &bool_var)| candidate != offset + i as i64 || assignment[bool_var] != Some(0))
    })
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

fn optional_presence_value(
    assignment: &[Option<i64>],
    presence: &Option<BoolLiteral>,
) -> Option<bool> {
    presence
        .as_ref()
        .map_or(Some(true), |lit| literal_value(assignment, lit))
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

fn partial_bool_and_ok(assignment: &[Option<i64>], lits: &[BoolLiteral]) -> bool {
    lits.iter()
        .all(|lit| literal_value(assignment, lit) != Some(false))
}

fn partial_bool_xor_ok(assignment: &[Option<i64>], lits: &[BoolLiteral]) -> bool {
    let mut has_unknown = false;
    let mut true_count = 0;
    for lit in lits {
        match literal_value(assignment, lit) {
            Some(true) => true_count += 1,
            Some(false) => {}
            None => has_unknown = true,
        }
    }
    has_unknown || true_count % 2 == 1
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

fn partial_forbidden_assignments_ok(
    assignment: &[Option<i64>],
    vars: &[usize],
    tuples: &[Vec<i64>],
) -> bool {
    !tuples.iter().any(|tuple| {
        vars.iter().zip(tuple).all(|(&var, &value)| {
            assignment[var]
                .map(|actual| actual == value)
                .unwrap_or(false)
        })
    })
}

fn partial_inverse_ok(assignment: &[Option<i64>], direct: &[usize], inverse: &[usize]) -> bool {
    let n = direct.len();
    let mut direct_values = Vec::new();
    for (i, &var) in direct.iter().enumerate() {
        let Some(value) = assignment[var] else {
            continue;
        };
        let Ok(j) = usize::try_from(value) else {
            return false;
        };
        if j >= n || direct_values.contains(&j) {
            return false;
        }
        direct_values.push(j);
        if let Some(inverse_value) = assignment[inverse[j]] {
            if inverse_value != i as i64 {
                return false;
            }
        }
    }

    let mut inverse_values = Vec::new();
    for (j, &var) in inverse.iter().enumerate() {
        let Some(value) = assignment[var] else {
            continue;
        };
        let Ok(i) = usize::try_from(value) else {
            return false;
        };
        if i >= n || inverse_values.contains(&i) {
            return false;
        }
        inverse_values.push(i);
        if let Some(direct_value) = assignment[direct[i]] {
            if direct_value != j as i64 {
                return false;
            }
        }
    }
    true
}

fn domain_min_max(domain: &[i64]) -> (i64, i64) {
    (
        *domain.iter().min().expect("validated non-empty domain"),
        *domain.iter().max().expect("validated non-empty domain"),
    )
}

fn variable_min_max(model: &CpModel, assignment: &[Option<i64>], var: usize) -> (i64, i64) {
    if let Some(value) = assignment[var] {
        (value, value)
    } else {
        domain_min_max(&model.variables[var].domain)
    }
}

fn partial_max_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    vars: &[usize],
) -> bool {
    let (target_min, target_max) = variable_min_max(model, assignment, target);
    let mut min_possible_max = i64::MIN;
    let mut max_possible_max = i64::MIN;
    for &var in vars {
        let (lo, hi) = variable_min_max(model, assignment, var);
        min_possible_max = min_possible_max.max(lo);
        max_possible_max = max_possible_max.max(hi);
    }
    if target_max < min_possible_max || target_min > max_possible_max {
        return false;
    }
    if let Some(target_value) = assignment[target] {
        let mut can_attain_target = false;
        for &var in vars {
            let (lo, hi) = variable_min_max(model, assignment, var);
            if lo > target_value {
                return false;
            }
            if lo <= target_value && target_value <= hi {
                can_attain_target = true;
            }
        }
        can_attain_target
    } else {
        true
    }
}

fn partial_min_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    vars: &[usize],
) -> bool {
    let (target_min, target_max) = variable_min_max(model, assignment, target);
    let mut min_possible_min = i64::MAX;
    let mut max_possible_min = i64::MAX;
    for &var in vars {
        let (lo, hi) = variable_min_max(model, assignment, var);
        min_possible_min = min_possible_min.min(lo);
        max_possible_min = max_possible_min.min(hi);
    }
    if target_max < min_possible_min || target_min > max_possible_min {
        return false;
    }
    if let Some(target_value) = assignment[target] {
        let mut can_attain_target = false;
        for &var in vars {
            let (lo, hi) = variable_min_max(model, assignment, var);
            if hi < target_value {
                return false;
            }
            if lo <= target_value && target_value <= hi {
                can_attain_target = true;
            }
        }
        can_attain_target
    } else {
        true
    }
}

fn abs_range(lo: i64, hi: i64) -> (i64, i64) {
    if lo <= 0 && 0 <= hi {
        (0, lo.abs().max(hi.abs()))
    } else {
        (lo.abs().min(hi.abs()), lo.abs().max(hi.abs()))
    }
}

fn partial_abs_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    var: usize,
) -> bool {
    let (var_lo, var_hi) = variable_min_max(model, assignment, var);
    let (abs_lo, abs_hi) = abs_range(var_lo, var_hi);
    let (target_lo, target_hi) = variable_min_max(model, assignment, target);
    if target_hi < abs_lo || target_lo > abs_hi {
        return false;
    }
    match (assignment[target], assignment[var]) {
        (Some(t), Some(v)) => t == v.abs(),
        (Some(t), None) => model.variables[var]
            .domain
            .iter()
            .any(|&value| value.abs() == t),
        (None, Some(v)) => model.variables[target]
            .domain
            .iter()
            .any(|&value| value == v.abs()),
        (None, None) => true,
    }
}

fn product_range(bounds: &[(i64, i64)]) -> (i128, i128) {
    let mut lo = 1_i128;
    let mut hi = 1_i128;
    for &(next_lo, next_hi) in bounds {
        let candidates = [
            lo * i128::from(next_lo),
            lo * i128::from(next_hi),
            hi * i128::from(next_lo),
            hi * i128::from(next_hi),
        ];
        lo = *candidates.iter().min().expect("non-empty candidates");
        hi = *candidates.iter().max().expect("non-empty candidates");
    }
    (lo, hi)
}

fn partial_multiplication_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    vars: &[usize],
) -> bool {
    let mut assigned_product = Some(1_i128);
    let mut bounds = Vec::with_capacity(vars.len());
    for &var in vars {
        if let Some(value) = assignment[var] {
            assigned_product = assigned_product.map(|product| product * i128::from(value));
            bounds.push((value, value));
        } else {
            assigned_product = None;
            bounds.push(variable_min_max(model, assignment, var));
        }
    }

    if let Some(product) = assigned_product {
        return match assignment[target] {
            Some(target_value) => product == i128::from(target_value),
            None => model.variables[target]
                .domain
                .iter()
                .any(|&value| i128::from(value) == product),
        };
    }

    let (product_lo, product_hi) = product_range(&bounds);
    let (target_lo, target_hi) = variable_min_max(model, assignment, target);
    i128::from(target_hi) >= product_lo && i128::from(target_lo) <= product_hi
}

fn possible_values(model: &CpModel, assignment: &[Option<i64>], var: usize) -> Vec<i64> {
    match assignment[var] {
        Some(value) => vec![value],
        None => model.variables[var].domain.clone(),
    }
}

fn partial_division_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    numerator: usize,
    denominator: usize,
) -> bool {
    let target_values = possible_values(model, assignment, target);
    let numerator_values = possible_values(model, assignment, numerator);
    let denominator_values = possible_values(model, assignment, denominator);
    numerator_values.iter().any(|&num| {
        denominator_values
            .iter()
            .filter(|&&den| den != 0)
            .any(|&den| target_values.contains(&(num / den)))
    })
}

fn partial_modulo_equality_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    target: usize,
    var: usize,
    modulus: usize,
) -> bool {
    let target_values = possible_values(model, assignment, target);
    let var_values = possible_values(model, assignment, var);
    let modulus_values = possible_values(model, assignment, modulus);
    var_values.iter().any(|&value| {
        modulus_values
            .iter()
            .filter(|&&modulus| modulus != 0)
            .any(|&modulus| target_values.contains(&(value % modulus)))
    })
}

fn partial_automaton_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    automaton: &CpAutomaton,
) -> bool {
    let mut states = vec![automaton.starting_state];
    for &var in &automaton.vars {
        let labels: Vec<i64> = match assignment[var] {
            Some(value) => vec![value],
            None => model.variables[var].domain.clone(),
        };
        let mut next_states = Vec::new();
        for &state in &states {
            for transition in &automaton.transitions {
                if transition.tail == state
                    && labels.contains(&transition.label)
                    && !next_states.contains(&transition.head)
                {
                    next_states.push(transition.head);
                }
            }
        }
        if next_states.is_empty() {
            return false;
        }
        states = next_states;
    }
    states
        .iter()
        .any(|state| automaton.final_states.contains(state))
}

fn circuit_nodes(arcs: &[CpCircuitArc]) -> Vec<i64> {
    let mut nodes = Vec::new();
    for arc in arcs {
        if !nodes.contains(&arc.tail) {
            nodes.push(arc.tail);
        }
        if !nodes.contains(&arc.head) {
            nodes.push(arc.head);
        }
    }
    nodes
}

fn circuit_complete_ok(selected: &[&CpCircuitArc], nodes: &[i64]) -> bool {
    let mut out = Vec::new();
    let mut incoming = Vec::new();
    for &node in nodes {
        let outgoing: Vec<_> = selected.iter().filter(|arc| arc.tail == node).collect();
        let inbound: Vec<_> = selected.iter().filter(|arc| arc.head == node).collect();
        if outgoing.len() != 1 || inbound.len() != 1 {
            return false;
        }
        out.push((node, outgoing[0].head));
        incoming.push((node, inbound[0].tail));
    }

    let active_nodes: Vec<i64> = out
        .iter()
        .filter_map(|&(tail, head)| if tail != head { Some(tail) } else { None })
        .collect();
    if active_nodes.is_empty() {
        return true;
    }

    let start = active_nodes[0];
    let mut current = start;
    let mut seen = Vec::new();
    loop {
        if seen.contains(&current) {
            return current == start && seen.len() == active_nodes.len();
        }
        if !active_nodes.contains(&current) {
            return false;
        }
        seen.push(current);
        let Some((_, next)) = out.iter().find(|&&(tail, _)| tail == current) else {
            return false;
        };
        current = *next;
    }
}

fn partial_circuit_ok(assignment: &[Option<i64>], arcs: &[CpCircuitArc]) -> bool {
    let nodes = circuit_nodes(arcs);
    for &node in &nodes {
        let true_out = arcs
            .iter()
            .filter(|arc| arc.tail == node && literal_value(assignment, &arc.literal) == Some(true))
            .count();
        let true_in = arcs
            .iter()
            .filter(|arc| arc.head == node && literal_value(assignment, &arc.literal) == Some(true))
            .count();
        if true_out > 1 || true_in > 1 {
            return false;
        }

        let possible_out = arcs
            .iter()
            .filter(|arc| {
                arc.tail == node && literal_value(assignment, &arc.literal) != Some(false)
            })
            .count();
        let possible_in = arcs
            .iter()
            .filter(|arc| {
                arc.head == node && literal_value(assignment, &arc.literal) != Some(false)
            })
            .count();
        if possible_out == 0 || possible_in == 0 {
            return false;
        }
    }

    let mut all_bound = true;
    let mut selected = Vec::new();
    for arc in arcs {
        match literal_value(assignment, &arc.literal) {
            Some(true) => selected.push(arc),
            Some(false) => {}
            None => all_bound = false,
        }
    }
    !all_bound || circuit_complete_ok(&selected, &nodes)
}

fn multiple_circuit_complete_ok(selected: &[&CpCircuitArc], nodes: &[i64]) -> bool {
    if !nodes.contains(&0) {
        return false;
    }
    let depot_out = selected.iter().filter(|arc| arc.tail == 0).count();
    let depot_in = selected.iter().filter(|arc| arc.head == 0).count();
    if depot_out != depot_in {
        return false;
    }

    for &node in nodes {
        if node == 0 {
            continue;
        }
        let outgoing: Vec<_> = selected.iter().filter(|arc| arc.tail == node).collect();
        let inbound: Vec<_> = selected.iter().filter(|arc| arc.head == node).collect();
        if outgoing.len() != 1 || inbound.len() != 1 {
            return false;
        }
    }

    for &start in nodes {
        if start == 0 {
            continue;
        }
        let Some(first) = selected.iter().find(|arc| arc.tail == start) else {
            return false;
        };
        if first.head == start {
            continue;
        }
        let mut current = start;
        let mut seen = Vec::new();
        loop {
            if current == 0 {
                break;
            }
            if seen.contains(&current) {
                return false;
            }
            seen.push(current);
            let Some(next_arc) = selected.iter().find(|arc| arc.tail == current) else {
                return false;
            };
            current = next_arc.head;
        }
    }
    true
}

fn partial_multiple_circuit_ok(assignment: &[Option<i64>], arcs: &[CpCircuitArc]) -> bool {
    let nodes = circuit_nodes(arcs);
    if !nodes.contains(&0) {
        return false;
    }
    for &node in &nodes {
        let true_out = arcs
            .iter()
            .filter(|arc| arc.tail == node && literal_value(assignment, &arc.literal) == Some(true))
            .count();
        let true_in = arcs
            .iter()
            .filter(|arc| arc.head == node && literal_value(assignment, &arc.literal) == Some(true))
            .count();
        let possible_out = arcs
            .iter()
            .filter(|arc| {
                arc.tail == node && literal_value(assignment, &arc.literal) != Some(false)
            })
            .count();
        let possible_in = arcs
            .iter()
            .filter(|arc| {
                arc.head == node && literal_value(assignment, &arc.literal) != Some(false)
            })
            .count();

        if node == 0 {
            if true_out > possible_in || true_in > possible_out {
                return false;
            }
            continue;
        }
        if true_out > 1 || true_in > 1 || possible_out == 0 || possible_in == 0 {
            return false;
        }
    }

    let mut all_bound = true;
    let mut selected = Vec::new();
    for arc in arcs {
        match literal_value(assignment, &arc.literal) {
            Some(true) => selected.push(arc),
            Some(false) => {}
            None => all_bound = false,
        }
    }
    !all_bound || multiple_circuit_complete_ok(&selected, &nodes)
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

fn variable_element_candidate_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    selected_var: usize,
    target: usize,
) -> bool {
    match (assignment[selected_var], assignment[target]) {
        (Some(selected), Some(target_value)) => selected == target_value,
        (Some(selected), None) => model.variables[target].domain.contains(&selected),
        (None, Some(target_value)) => model.variables[selected_var].domain.contains(&target_value),
        (None, None) => model.variables[selected_var]
            .domain
            .iter()
            .any(|value| model.variables[target].domain.contains(value)),
    }
}

fn partial_variable_element_ok(
    model: &CpModel,
    assignment: &[Option<i64>],
    element: &CpVariableElement,
) -> bool {
    let index_values: Vec<i64> = match assignment[element.index] {
        Some(value) => vec![value],
        None => model.variables[element.index].domain.clone(),
    };
    index_values
        .iter()
        .filter_map(|&index| usize::try_from(index).ok())
        .filter_map(|index| element.vars.get(index).copied())
        .any(|selected_var| {
            variable_element_candidate_ok(model, assignment, selected_var, element.target)
        })
}

fn active_interval_consistent(
    assignment: &[Option<i64>],
    start: usize,
    duration: usize,
    end: usize,
    active: Option<bool>,
) -> bool {
    if active != Some(true) {
        return true;
    }
    match (assignment[start], assignment[duration], assignment[end]) {
        (Some(start), Some(duration), Some(end)) => start + duration == end,
        (Some(start), Some(duration), None) => start.checked_add(duration).is_some(),
        _ => true,
    }
}

fn assigned_equal_or_unknown(assignment: &[Option<i64>], lhs: usize, rhs: usize) -> bool {
    match (assignment[lhs], assignment[rhs]) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

fn partial_alternative_ok(assignment: &[Option<i64>], alternative: &CpAlternative) -> bool {
    let parent_active = optional_presence_value(assignment, &alternative.presence);
    let mut true_modes = 0usize;
    let mut unknown_modes = 0usize;

    for mode in &alternative.alternatives {
        match optional_presence_value(assignment, &mode.presence) {
            Some(true) => true_modes += 1,
            Some(false) => {}
            None => unknown_modes += 1,
        }
    }

    if true_modes > 1 {
        return false;
    }
    match parent_active {
        Some(true) if true_modes == 0 && unknown_modes == 0 => return false,
        Some(false) if true_modes > 0 => return false,
        _ => {}
    }

    if !active_interval_consistent(
        assignment,
        alternative.start,
        alternative.duration,
        alternative.end,
        parent_active,
    ) {
        return false;
    }

    for mode in &alternative.alternatives {
        let mode_active = optional_presence_value(assignment, &mode.presence);
        if !active_interval_consistent(assignment, mode.start, mode.duration, mode.end, mode_active)
        {
            return false;
        }
        if mode_active != Some(true) {
            continue;
        }
        if parent_active == Some(false) {
            return false;
        }
        if !assigned_equal_or_unknown(assignment, alternative.start, mode.start)
            || !assigned_equal_or_unknown(assignment, alternative.duration, mode.duration)
            || !assigned_equal_or_unknown(assignment, alternative.end, mode.end)
        {
            return false;
        }
    }

    true
}

fn partial_no_overlap_ok(assignment: &[Option<i64>], intervals: &[CpInterval]) -> bool {
    for i in 0..intervals.len() {
        if optional_presence_value(assignment, &intervals[i].presence) != Some(true) {
            continue;
        }
        let Some(start_i) = assignment[intervals[i].start] else {
            continue;
        };
        let end_i = start_i + intervals[i].duration;
        for j in (i + 1)..intervals.len() {
            if optional_presence_value(assignment, &intervals[j].presence) != Some(true) {
                continue;
            }
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

fn variable_interval_span(
    assignment: &[Option<i64>],
    interval: &CpVariableInterval,
) -> Result<Option<(i64, i64)>, ()> {
    if optional_presence_value(assignment, &interval.presence) != Some(true) {
        return Ok(None);
    }
    let Some(start) = assignment[interval.start] else {
        return Ok(None);
    };
    let Some(duration) = assignment[interval.duration] else {
        return Ok(None);
    };
    let computed_end = start + duration;
    if let Some(end) = assignment[interval.end] {
        if end != computed_end {
            return Err(());
        }
    }
    Ok(Some((start, computed_end)))
}

fn partial_no_overlap_variable_ok(
    assignment: &[Option<i64>],
    intervals: &[CpVariableInterval],
) -> bool {
    let mut spans = Vec::new();
    for interval in intervals {
        match variable_interval_span(assignment, interval) {
            Ok(Some(span)) => spans.push(span),
            Ok(None) => {}
            Err(()) => return false,
        }
    }
    for i in 0..spans.len() {
        let (start_i, end_i) = spans[i];
        for &(start_j, end_j) in spans.iter().skip(i + 1) {
            if !(end_i <= start_j || end_j <= start_i) {
                return false;
            }
        }
    }
    true
}

fn partial_no_overlap_2d_ok(assignment: &[Option<i64>], rectangles: &[CpRectangle]) -> bool {
    for i in 0..rectangles.len() {
        if optional_presence_value(assignment, &rectangles[i].presence) != Some(true) {
            continue;
        }
        let Some(x_i) = assignment[rectangles[i].x_start] else {
            continue;
        };
        let Some(y_i) = assignment[rectangles[i].y_start] else {
            continue;
        };
        let x_end_i = x_i + rectangles[i].width;
        let y_end_i = y_i + rectangles[i].height;
        for j in (i + 1)..rectangles.len() {
            if optional_presence_value(assignment, &rectangles[j].presence) != Some(true) {
                continue;
            }
            let Some(x_j) = assignment[rectangles[j].x_start] else {
                continue;
            };
            let Some(y_j) = assignment[rectangles[j].y_start] else {
                continue;
            };
            let x_end_j = x_j + rectangles[j].width;
            let y_end_j = y_j + rectangles[j].height;
            let x_disjoint = x_end_i <= x_j || x_end_j <= x_i;
            let y_disjoint = y_end_i <= y_j || y_end_j <= y_i;
            if !(x_disjoint || y_disjoint) {
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
        if optional_presence_value(assignment, &interval.presence) != Some(true) {
            continue;
        }
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

fn variable_rectangle_span(
    assignment: &[Option<i64>],
    rectangle: &CpVariableRectangle,
) -> Result<Option<(i64, i64, i64, i64)>, ()> {
    if optional_presence_value(assignment, &rectangle.presence) != Some(true) {
        return Ok(None);
    }
    let Some(x_start) = assignment[rectangle.x_start] else {
        return Ok(None);
    };
    let Some(x_size) = assignment[rectangle.x_size] else {
        return Ok(None);
    };
    let Some(y_start) = assignment[rectangle.y_start] else {
        return Ok(None);
    };
    let Some(y_size) = assignment[rectangle.y_size] else {
        return Ok(None);
    };
    let x_end = x_start + x_size;
    let y_end = y_start + y_size;
    if let Some(bound_x_end) = assignment[rectangle.x_end] {
        if bound_x_end != x_end {
            return Err(());
        }
    }
    if let Some(bound_y_end) = assignment[rectangle.y_end] {
        if bound_y_end != y_end {
            return Err(());
        }
    }
    Ok(Some((x_start, x_end, y_start, y_end)))
}

fn partial_no_overlap_2d_variable_ok(
    assignment: &[Option<i64>],
    rectangles: &[CpVariableRectangle],
) -> bool {
    let mut spans = Vec::new();
    for rectangle in rectangles {
        match variable_rectangle_span(assignment, rectangle) {
            Ok(Some(span)) => spans.push(span),
            Ok(None) => {}
            Err(()) => return false,
        }
    }
    for i in 0..spans.len() {
        let (x_start_i, x_end_i, y_start_i, y_end_i) = spans[i];
        for &(x_start_j, x_end_j, y_start_j, y_end_j) in spans.iter().skip(i + 1) {
            let x_disjoint = x_end_i <= x_start_j || x_end_j <= x_start_i;
            let y_disjoint = y_end_i <= y_start_j || y_end_j <= y_start_i;
            if !(x_disjoint || y_disjoint) {
                return false;
            }
        }
    }
    true
}

fn variable_demand_interval_span(
    assignment: &[Option<i64>],
    interval: &CpVariableDemandInterval,
) -> Result<Option<(i64, i64, i64)>, ()> {
    if optional_presence_value(assignment, &interval.presence) != Some(true) {
        return Ok(None);
    }
    let Some(start) = assignment[interval.start] else {
        return Ok(None);
    };
    let Some(duration) = assignment[interval.duration] else {
        return Ok(None);
    };
    let computed_end = start + duration;
    if let Some(end) = assignment[interval.end] {
        if end != computed_end {
            return Err(());
        }
    }
    let Some(demand) = assignment[interval.demand] else {
        return Ok(None);
    };
    Ok(Some((start, computed_end, demand)))
}

fn partial_cumulative_variable_ok(
    assignment: &[Option<i64>],
    intervals: &[CpVariableDemandInterval],
    capacity: usize,
) -> bool {
    let Some(capacity) = assignment[capacity] else {
        return intervals
            .iter()
            .all(|interval| variable_demand_interval_span(assignment, interval).is_ok());
    };
    let mut assigned = Vec::new();
    for interval in intervals {
        match variable_demand_interval_span(assignment, interval) {
            Ok(Some(span)) => assigned.push(span),
            Ok(None) => {}
            Err(()) => return false,
        }
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

fn reservoir_complete_ok(events: &[(i64, i64)], min_level: i64, max_level: i64) -> bool {
    if !(min_level <= 0 && 0 <= max_level) {
        return false;
    }
    let mut sorted = events.to_vec();
    sorted.sort_unstable_by_key(|&(time, _)| time);
    let mut level = 0i64;
    let mut i = 0usize;
    while i < sorted.len() {
        let time = sorted[i].0;
        while i < sorted.len() && sorted[i].0 == time {
            level += sorted[i].1;
            i += 1;
        }
        if level < min_level || level > max_level {
            return false;
        }
    }
    true
}

fn partial_reservoir_ok(
    assignment: &[Option<i64>],
    events: &[CpReservoirEvent],
    min_level: i64,
    max_level: i64,
) -> bool {
    let mut all_bound = true;
    let mut active_events = Vec::new();
    for event in events {
        if let Some(active) = &event.active {
            match literal_value(assignment, active) {
                Some(false) => continue,
                Some(true) => {}
                None => {
                    all_bound = false;
                    continue;
                }
            }
        }
        let Some(time) = assignment[event.time] else {
            all_bound = false;
            continue;
        };
        active_events.push((time, event.level_change));
    }
    !all_bound || reservoir_complete_ok(&active_events, min_level, max_level)
}

fn partial_constraints_ok(model: &CpModel, assignment: &[Option<i64>]) -> bool {
    for constraint in &model.constraints {
        let ok = match constraint {
            CpConstraint::Linear { terms, sense, rhs } => {
                partial_linear_ok(model, assignment, terms, *sense, *rhs)
            }
            CpConstraint::LinearDomain { terms, intervals } => {
                partial_linear_domain_ok(model, assignment, terms, intervals)
            }
            CpConstraint::MapDomain { var, bools, offset } => {
                partial_map_domain_ok(model, assignment, *var, bools, *offset)
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
            CpConstraint::EnforcedBoolOr {
                enforcement,
                literals,
            } => match enforcement_literals_active(assignment, enforcement) {
                Some(false) => true,
                Some(true) => partial_bool_or_ok(assignment, literals),
                None => true,
            },
            CpConstraint::EnforcedBoolAnd {
                enforcement,
                literals,
            } => match enforcement_literals_active(assignment, enforcement) {
                Some(false) => true,
                Some(true) => partial_bool_and_ok(assignment, literals),
                None => true,
            },
            CpConstraint::AllDifferent(vars) => partial_all_different_ok(assignment, vars),
            CpConstraint::BoolOr(lits) => partial_bool_or_ok(assignment, lits),
            CpConstraint::BoolAnd(lits) => partial_bool_and_ok(assignment, lits),
            CpConstraint::BoolXor(lits) => partial_bool_xor_ok(assignment, lits),
            CpConstraint::AtMostOne(lits) => partial_at_most_one_ok(assignment, lits),
            CpConstraint::ExactlyOne(lits) => partial_exactly_one_ok(assignment, lits),
            CpConstraint::Implication {
                antecedent,
                consequent,
            } => partial_implication_ok(assignment, antecedent, consequent),
            CpConstraint::AllowedAssignments { vars, tuples } => {
                partial_allowed_assignments_ok(assignment, vars, tuples)
            }
            CpConstraint::ForbiddenAssignments { vars, tuples } => {
                partial_forbidden_assignments_ok(assignment, vars, tuples)
            }
            CpConstraint::EnforcedAllowedAssignments {
                enforcement,
                vars,
                tuples,
            } => match enforcement_literals_active(assignment, enforcement) {
                Some(false) => true,
                Some(true) => partial_allowed_assignments_ok(assignment, vars, tuples),
                None => true,
            },
            CpConstraint::EnforcedForbiddenAssignments {
                enforcement,
                vars,
                tuples,
            } => match enforcement_literals_active(assignment, enforcement) {
                Some(false) => true,
                Some(true) => partial_forbidden_assignments_ok(assignment, vars, tuples),
                None => true,
            },
            CpConstraint::Inverse { direct, inverse } => {
                partial_inverse_ok(assignment, direct, inverse)
            }
            CpConstraint::MaxEquality { target, vars } => {
                partial_max_equality_ok(model, assignment, *target, vars)
            }
            CpConstraint::MinEquality { target, vars } => {
                partial_min_equality_ok(model, assignment, *target, vars)
            }
            CpConstraint::AbsEquality { target, var } => {
                partial_abs_equality_ok(model, assignment, *target, *var)
            }
            CpConstraint::MultiplicationEquality { target, vars } => {
                partial_multiplication_equality_ok(model, assignment, *target, vars)
            }
            CpConstraint::DivisionEquality {
                target,
                numerator,
                denominator,
            } => partial_division_equality_ok(model, assignment, *target, *numerator, *denominator),
            CpConstraint::ModuloEquality {
                target,
                var,
                modulus,
            } => partial_modulo_equality_ok(model, assignment, *target, *var, *modulus),
            CpConstraint::Automaton(automaton) => {
                partial_automaton_ok(model, assignment, automaton)
            }
            CpConstraint::Circuit(arcs) => partial_circuit_ok(assignment, arcs),
            CpConstraint::MultipleCircuit(arcs) => partial_multiple_circuit_ok(assignment, arcs),
            CpConstraint::Element(element) => partial_element_ok(model, assignment, element),
            CpConstraint::VariableElement(element) => {
                partial_variable_element_ok(model, assignment, element)
            }
            CpConstraint::Alternative(alternative) => {
                partial_alternative_ok(assignment, alternative)
            }
            CpConstraint::NoOverlap(intervals) => partial_no_overlap_ok(assignment, intervals),
            CpConstraint::NoOverlapVariable(intervals) => {
                partial_no_overlap_variable_ok(assignment, intervals)
            }
            CpConstraint::NoOverlap2D(rectangles) => {
                partial_no_overlap_2d_ok(assignment, rectangles)
            }
            CpConstraint::NoOverlap2DVariable(rectangles) => {
                partial_no_overlap_2d_variable_ok(assignment, rectangles)
            }
            CpConstraint::Cumulative {
                intervals,
                capacity,
            } => partial_cumulative_ok(assignment, intervals, *capacity),
            CpConstraint::CumulativeVariable {
                intervals,
                capacity,
            } => partial_cumulative_variable_ok(assignment, intervals, *capacity),
            CpConstraint::Reservoir {
                events,
                min_level,
                max_level,
            } => partial_reservoir_ok(assignment, events, *min_level, *max_level),
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

fn validate_solution_hints(model: &CpModel, hints: &[CpSolutionHint]) {
    let n = model.variables.len();
    let mut seen = Vec::new();
    for hint in hints {
        if hint.var >= n {
            panic!(
                "cp-sat: solution hint variable index {} out of range {n}",
                hint.var
            );
        }
        if seen.contains(&hint.var) {
            panic!("cp-sat: duplicate solution hint for variable {}", hint.var);
        }
        seen.push(hint.var);
        if !model.variables[hint.var].domain.contains(&hint.value) {
            panic!(
                "cp-sat: solution hint value {} is outside variable {} domain",
                hint.value, hint.var
            );
        }
    }
}

fn hint_values(model: &CpModel, hints: &[CpSolutionHint]) -> Vec<Option<i64>> {
    validate_solution_hints(model, hints);
    let mut values = vec![None; model.variables.len()];
    for hint in hints {
        values[hint.var] = Some(hint.value);
    }
    values
}

fn validate_decision_strategies(model: &CpModel, strategies: &[CpDecisionStrategy]) {
    let n = model.variables.len();
    let mut seen = Vec::new();
    for strategy in strategies {
        if strategy.vars.is_empty() {
            panic!("cp-sat: decision strategy has no variables");
        }
        for &var in &strategy.vars {
            if var >= n {
                panic!("cp-sat: decision strategy variable index {var} out of range {n}");
            }
            if seen.contains(&var) {
                panic!("cp-sat: duplicate decision strategy for variable {var}");
            }
            seen.push(var);
        }
    }
}

fn domain_min(model: &CpModel, var: usize) -> i64 {
    *model.variables[var].domain.iter().min().unwrap()
}

fn domain_max(model: &CpModel, var: usize) -> i64 {
    *model.variables[var].domain.iter().max().unwrap()
}

fn strategy_better_var(
    model: &CpModel,
    strategy: CpVariableSelectionStrategy,
    candidate: usize,
    incumbent: usize,
) -> bool {
    match strategy {
        CpVariableSelectionStrategy::First => false,
        CpVariableSelectionStrategy::MinDomainSize => {
            model.variables[candidate].domain.len() < model.variables[incumbent].domain.len()
        }
        CpVariableSelectionStrategy::MaxDomainSize => {
            model.variables[candidate].domain.len() > model.variables[incumbent].domain.len()
        }
        CpVariableSelectionStrategy::LowestMin => {
            domain_min(model, candidate) < domain_min(model, incumbent)
        }
        CpVariableSelectionStrategy::HighestMax => {
            domain_max(model, candidate) > domain_max(model, incumbent)
        }
    }
}

fn choose_strategy_variable(
    model: &CpModel,
    assignment: &[Option<i64>],
    strategy: &CpDecisionStrategy,
) -> Option<usize> {
    let mut best = None;
    for &var in &strategy.vars {
        if assignment[var].is_some() {
            continue;
        }
        if best.is_some_and(|incumbent| {
            strategy_better_var(model, strategy.variable_strategy, var, incumbent)
        }) {
            best = Some(var);
        } else if best.is_none() {
            best = Some(var);
        }
    }
    best
}

fn decision_strategy_for_var<'a>(
    strategies: &'a [CpDecisionStrategy],
    var: usize,
) -> Option<&'a CpDecisionStrategy> {
    strategies
        .iter()
        .find(|strategy| strategy.vars.contains(&var))
}

fn choose_variable(
    model: &CpModel,
    assignment: &[Option<i64>],
    hints: &[Option<i64>],
    strategies: &[CpDecisionStrategy],
) -> Option<usize> {
    if let Some(hinted) = hints
        .iter()
        .enumerate()
        .find_map(|(idx, value)| value.and_then(|_| assignment[idx].is_none().then_some(idx)))
    {
        return Some(hinted);
    }

    for strategy in strategies {
        if let Some(var) = choose_strategy_variable(model, assignment, strategy) {
            return Some(var);
        }
    }

    model
        .variables
        .iter()
        .enumerate()
        .filter(|(i, _)| assignment[*i].is_none())
        .min_by_key(|(_, v)| v.domain.len())
        .map(|(i, _)| i)
}

fn ordered_domain(
    model: &CpModel,
    var: usize,
    hints: &[Option<i64>],
    strategies: &[CpDecisionStrategy],
) -> Vec<i64> {
    let mut values = model.variables[var].domain.clone();
    if let Some(hint) = hints[var] {
        if let Some(pos) = values.iter().position(|&value| value == hint) {
            values.remove(pos);
            values.insert(0, hint);
        }
    } else if let Some(strategy) = decision_strategy_for_var(strategies, var) {
        match strategy.domain_strategy {
            CpDomainValueStrategy::MinValue => values.sort_unstable(),
            CpDomainValueStrategy::MaxValue => values.sort_unstable_by(|a, b| b.cmp(a)),
        }
    }
    values
}

/// Solve a small finite-domain CP-SAT-style model exactly.
pub fn solve_cp_model(model: &CpModel, opts: CpSolveOptions) -> CpSolution {
    validate_model(model);
    let hints = hint_values(model, &opts.solution_hint);
    validate_decision_strategies(model, &opts.decision_strategies);
    let n = model.variables.len();
    let mut assignment = vec![None; n];
    let mut best_assignment = Vec::new();
    let mut best_objective: Option<i64> = None;
    let mut nodes = 0usize;
    let mut hit_limit = false;

    fn dfs(
        model: &CpModel,
        opts: &CpSolveOptions,
        assignment: &mut [Option<i64>],
        hints: &[Option<i64>],
        strategies: &[CpDecisionStrategy],
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
        let Some(var) = choose_variable(model, assignment, hints, strategies) else {
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
        for value in ordered_domain(model, var, hints, strategies) {
            assignment[var] = Some(value);
            dfs(
                model,
                opts,
                assignment,
                hints,
                strategies,
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
        &opts,
        &mut assignment,
        &hints,
        &opts.decision_strategies,
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

/// Enumerate feasible assignments for a finite-domain CP-SAT-style model.
///
/// For satisfaction models this returns assignments in search order. For
/// objective models it performs exact bounded traversal, sorts by objective, and
/// returns the best `max_solutions` assignments.
pub fn enumerate_cp_solutions(model: &CpModel, opts: CpEnumerateOptions) -> CpSolutionPool {
    validate_model(model);
    if opts.max_solutions == 0 {
        panic!("cp-sat: max_solutions must be positive");
    }
    let n = model.variables.len();
    let mut assignment = vec![None; n];
    let hints = vec![None; n];
    let strategies = Vec::new();
    let mut solutions = Vec::new();
    let mut nodes = 0usize;
    let mut hit_node_limit = false;
    let mut hit_solution_limit = false;

    fn dfs(
        model: &CpModel,
        opts: CpEnumerateOptions,
        assignment: &mut [Option<i64>],
        hints: &[Option<i64>],
        strategies: &[CpDecisionStrategy],
        solutions: &mut Vec<CpSolutionPoolEntry>,
        nodes: &mut usize,
        hit_node_limit: &mut bool,
        hit_solution_limit: &mut bool,
    ) {
        if *hit_node_limit || *hit_solution_limit {
            return;
        }
        *nodes += 1;
        if *nodes > opts.max_nodes {
            *hit_node_limit = true;
            return;
        }
        if !partial_constraints_ok(model, assignment) {
            return;
        }
        let Some(var) = choose_variable(model, assignment, hints, strategies) else {
            let full: Vec<i64> = assignment.iter().map(|v| v.unwrap()).collect();
            let objective = model
                .objective
                .as_ref()
                .map(|obj| objective_value(obj, &full));
            solutions.push(CpSolutionPoolEntry {
                assignment: full,
                objective,
            });
            if model.objective.is_none() && solutions.len() >= opts.max_solutions {
                *hit_solution_limit = true;
            }
            return;
        };

        for value in ordered_domain(model, var, hints, strategies) {
            assignment[var] = Some(value);
            dfs(
                model,
                opts,
                assignment,
                hints,
                strategies,
                solutions,
                nodes,
                hit_node_limit,
                hit_solution_limit,
            );
            assignment[var] = None;
            if *hit_node_limit || *hit_solution_limit {
                break;
            }
        }
    }

    dfs(
        model,
        opts,
        &mut assignment,
        &hints,
        &strategies,
        &mut solutions,
        &mut nodes,
        &mut hit_node_limit,
        &mut hit_solution_limit,
    );

    if let Some(obj) = &model.objective {
        solutions.sort_by(|a, b| match obj.sense {
            ObjectiveSense::Min => a.objective.cmp(&b.objective),
            ObjectiveSense::Max => b.objective.cmp(&a.objective),
        });
        if solutions.len() > opts.max_solutions {
            solutions.truncate(opts.max_solutions);
            hit_solution_limit = true;
        }
    }

    let status = if solutions.is_empty() {
        CpStatus::Infeasible
    } else if model.objective.is_some() && !hit_node_limit {
        CpStatus::Optimal
    } else {
        CpStatus::Feasible
    };
    let exhausted = !hit_node_limit && !hit_solution_limit;
    CpSolutionPool {
        status,
        solutions,
        nodes,
        exhausted,
        solver: "internal-cp-solution-enumeration".to_string(),
        message: Some(if exhausted {
            "exact finite-domain solution enumeration".to_string()
        } else if hit_node_limit {
            "node limit reached during solution enumeration".to_string()
        } else {
            "solution limit reached during solution enumeration".to_string()
        }),
    }
}

fn validate_assumptions(model: &CpModel, assumptions: &[BoolLiteral]) {
    let n = model.variables.len();
    for lit in assumptions {
        if lit.var >= n {
            panic!(
                "cp-sat: assumption variable index {} out of range {n}",
                lit.var
            );
        }
        if model.variables[lit.var]
            .domain
            .iter()
            .any(|&value| value != 0 && value != 1)
        {
            panic!("cp-sat: assumption variable {} is not boolean", lit.var);
        }
    }
}

fn model_with_assumptions(model: &CpModel, assumptions: &[BoolLiteral]) -> CpModel {
    validate_model(model);
    validate_assumptions(model, assumptions);
    let mut out = model.clone();
    out.constraints
        .extend(assumptions.iter().map(|lit| CpConstraint::Linear {
            terms: vec![LinearTerm {
                var: lit.var,
                coeff: 1,
            }],
            sense: LinearSense::Eq,
            rhs: if lit.positive { 1 } else { 0 },
        }));
    out
}

/// Solve a model under CP-SAT-style assumption literals.
pub fn solve_cp_model_with_assumptions(
    model: &CpModel,
    assumptions: &[BoolLiteral],
    opts: CpSolveOptions,
) -> CpSolution {
    let assumed = model_with_assumptions(model, assumptions);
    let mut sol = solve_cp_model(&assumed, opts);
    sol.solver = "internal-cp-assumption-enumeration".to_string();
    sol
}

/// Find a minimal infeasible core over CP-SAT-style assumption literals.
///
/// This mirrors OR-Tools CP-SAT's assumption-core surface for small finite
/// models. The result is minimal by single deletion, not guaranteed
/// minimum-cardinality.
pub fn find_cp_assumption_unsat_core(
    model: &CpModel,
    assumptions: &[BoolLiteral],
    opts: CpAssumptionCoreOptions,
) -> CpAssumptionCore {
    validate_model(model);
    validate_assumptions(model, assumptions);
    let mut core = assumptions.to_vec();
    let mut checks = 1usize;
    let full = solve_cp_model_with_assumptions(model, &core, opts.solve_options.clone());
    if full.status != CpStatus::Infeasible {
        return CpAssumptionCore {
            status: full.status,
            assumptions: Vec::new(),
            minimal: false,
            checks,
            solver: "internal-cp-assumption-core".to_string(),
            message: Some(format!(
                "assumptions are not infeasible; status is {}",
                full.status.as_str()
            )),
        };
    }

    let mut idx = 0usize;
    while idx < core.len() {
        let mut trial = core.clone();
        trial.remove(idx);
        checks += 1;
        if solve_cp_model_with_assumptions(model, &trial, opts.solve_options.clone()).status
            == CpStatus::Infeasible
        {
            core = trial;
        } else {
            idx += 1;
        }
    }

    let mut minimal = true;
    for idx in 0..core.len() {
        let mut trial = core.clone();
        trial.remove(idx);
        checks += 1;
        if solve_cp_model_with_assumptions(model, &trial, opts.solve_options.clone()).status
            == CpStatus::Infeasible
        {
            minimal = false;
            break;
        }
    }

    CpAssumptionCore {
        status: CpStatus::Infeasible,
        assumptions: core,
        minimal,
        checks,
        solver: "internal-cp-assumption-core".to_string(),
        message: Some("minimal infeasible assumption core".to_string()),
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
    fn solution_hint_guides_search_before_node_limit() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Eq,
                rhs: 2,
            }],
            objective: None,
        };
        let hinted = solve_cp_model(
            &model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: vec![
                    CpSolutionHint { var: 0, value: 1 },
                    CpSolutionHint { var: 1, value: 1 },
                ],
                decision_strategies: Vec::new(),
            },
        );
        let unhinted = solve_cp_model(
            &model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: Vec::new(),
            },
        );
        assert_eq!(hinted.status, CpStatus::Feasible);
        assert_eq!(hinted.assignment, vec![1, 1]);
        assert_eq!(unhinted.status, CpStatus::Infeasible);
    }

    #[test]
    fn decision_strategy_guides_search_before_node_limit() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Eq,
                rhs: 2,
            }],
            objective: None,
        };
        let strategic = solve_cp_model(
            &model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: vec![CpDecisionStrategy {
                    vars: vec![1, 0],
                    variable_strategy: CpVariableSelectionStrategy::First,
                    domain_strategy: CpDomainValueStrategy::MaxValue,
                }],
            },
        );
        let unstrategic = solve_cp_model(
            &model,
            CpSolveOptions {
                max_nodes: 3,
                solution_hint: Vec::new(),
                decision_strategies: Vec::new(),
            },
        );
        assert_eq!(strategic.status, CpStatus::Feasible);
        assert_eq!(strategic.assignment, vec![1, 1]);
        assert_eq!(unstrategic.status, CpStatus::Infeasible);
    }

    #[test]
    fn enumerates_satisfaction_solution_pool() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Linear {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
                sense: LinearSense::Le,
                rhs: 1,
            }],
            objective: None,
        };

        let pool = enumerate_cp_solutions(
            &model,
            CpEnumerateOptions {
                max_nodes: 100,
                max_solutions: 4,
            },
        );
        let assignments: Vec<Vec<i64>> = pool
            .solutions
            .iter()
            .map(|solution| solution.assignment.clone())
            .collect();

        assert_eq!(pool.status, CpStatus::Feasible);
        assert!(pool.exhausted);
        assert_eq!(assignments, vec![vec![0, 0], vec![0, 1], vec![1, 0]]);
    }

    #[test]
    fn finds_assumption_unsat_core() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "irrelevant".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::BoolOr(vec![
                BoolLiteral {
                    var: 0,
                    positive: false,
                },
                BoolLiteral {
                    var: 1,
                    positive: false,
                },
            ])],
            objective: None,
        };
        let assumptions = vec![
            BoolLiteral {
                var: 0,
                positive: true,
            },
            BoolLiteral {
                var: 1,
                positive: true,
            },
            BoolLiteral {
                var: 2,
                positive: true,
            },
        ];

        let core =
            find_cp_assumption_unsat_core(&model, &assumptions, CpAssumptionCoreOptions::default());

        assert_eq!(core.status, CpStatus::Infeasible);
        assert!(core.minimal);
        assert_eq!(core.assumptions, assumptions[..2].to_vec());
        assert_eq!(
            solve_cp_model_with_assumptions(&model, &core.assumptions, CpSolveOptions::default())
                .status,
            CpStatus::Infeasible
        );
        for idx in 0..core.assumptions.len() {
            let mut trial = core.assumptions.clone();
            trial.remove(idx);
            assert_eq!(
                solve_cp_model_with_assumptions(&model, &trial, CpSolveOptions::default()).status,
                CpStatus::Feasible
            );
        }
    }

    #[test]
    fn solves_linear_domain_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1, 2],
                },
            ],
            constraints: vec![CpConstraint::LinearDomain {
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 2 },
                ],
                intervals: vec![
                    CpDomainInterval { lb: 1, ub: 1 },
                    CpDomainInterval { lb: 4, ub: 4 },
                ],
            }],
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
        assert_eq!(sol.assignment, vec![1, 0]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_map_domain_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "mode".to_string(),
                    domain: vec![5, 6, 7],
                },
                CpVariable {
                    name: "is_five".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "is_six".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "is_seven".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::MapDomain {
                    var: 0,
                    bools: vec![1, 2, 3],
                    offset: 5,
                },
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 2,
                    positive: true,
                }]),
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 0, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![6, 0, 1, 0]);
        assert_eq!(sol.objective, Some(6));
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
                    presence: None,
                    name: Some("task_a".to_string()),
                },
                CpInterval {
                    start: 1,
                    duration: 2,
                    presence: None,
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
    fn skips_absent_optional_interval_in_no_overlap() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "required_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "optional_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "use_optional".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::NoOverlap(vec![
                CpInterval {
                    start: 0,
                    duration: 3,
                    presence: None,
                    name: Some("required".to_string()),
                },
                CpInterval {
                    start: 1,
                    duration: 2,
                    presence: Some(BoolLiteral {
                        var: 2,
                        positive: true,
                    }),
                    name: Some("optional".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 2, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 0, 0]);
        assert_eq!(sol.objective, Some(0));
    }

    #[test]
    fn variable_size_no_overlap_uses_duration_and_end() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "task_a_duration".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_a_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "task_b_duration".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "task_b_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::NoOverlapVariable(vec![
                CpVariableInterval {
                    start: 0,
                    duration: 1,
                    end: 2,
                    presence: None,
                    name: Some("task_a".to_string()),
                },
                CpVariableInterval {
                    start: 3,
                    duration: 4,
                    end: 5,
                    presence: None,
                    name: Some("task_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 2, 2, 2, 1, 3]);
        assert_eq!(sol.objective, Some(3));
    }

    #[test]
    fn alternative_interval_selects_one_mode_and_syncs_parent() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "job_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "job_duration".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "job_end".to_string(),
                    domain: vec![2, 3, 4],
                },
                CpVariable {
                    name: "fast_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "fast_duration".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "fast_end".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "use_fast".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "slow_start".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "slow_duration".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "slow_end".to_string(),
                    domain: vec![3, 4],
                },
                CpVariable {
                    name: "use_slow".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Alternative(CpAlternative {
                start: 0,
                duration: 1,
                end: 2,
                presence: None,
                alternatives: vec![
                    CpVariableInterval {
                        start: 3,
                        duration: 4,
                        end: 5,
                        presence: Some(BoolLiteral {
                            var: 6,
                            positive: true,
                        }),
                        name: Some("fast".to_string()),
                    },
                    CpVariableInterval {
                        start: 7,
                        duration: 8,
                        end: 9,
                        presence: Some(BoolLiteral {
                            var: 10,
                            positive: true,
                        }),
                        name: Some("slow".to_string()),
                    },
                ],
                name: Some("job_modes".to_string()),
            })],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 1, coeff: 10 },
                    LinearTerm { var: 0, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 2, 2, 0, 2, 2, 1, 0, 3, 3, 0]);
        assert_eq!(sol.objective, Some(20));
    }

    #[test]
    fn solves_no_overlap_2d_packing() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "box_a_x".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_a_y".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_b_x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "box_b_y".to_string(),
                    domain: vec![0],
                },
            ],
            constraints: vec![CpConstraint::NoOverlap2D(vec![
                CpRectangle {
                    x_start: 0,
                    y_start: 1,
                    width: 2,
                    height: 2,
                    presence: None,
                    name: Some("box_a".to_string()),
                },
                CpRectangle {
                    x_start: 2,
                    y_start: 3,
                    width: 2,
                    height: 2,
                    presence: None,
                    name: Some("box_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 2, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 0, 2, 0]);
        assert_eq!(sol.objective, Some(2));
    }

    #[test]
    fn variable_size_no_overlap_2d_uses_axis_sizes_and_ends() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "box_a_x_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_a_x_size".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_x_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_y_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "box_a_y_size".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_a_y_end".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "box_b_x_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "box_b_x_size".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "box_b_x_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
                CpVariable {
                    name: "box_b_y_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "box_b_y_size".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "box_b_y_end".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::NoOverlap2DVariable(vec![
                CpVariableRectangle {
                    x_start: 0,
                    x_size: 1,
                    x_end: 2,
                    y_start: 3,
                    y_size: 4,
                    y_end: 5,
                    presence: None,
                    name: Some("box_a".to_string()),
                },
                CpVariableRectangle {
                    x_start: 6,
                    x_size: 7,
                    x_end: 8,
                    y_start: 9,
                    y_size: 10,
                    y_end: 11,
                    presence: None,
                    name: Some("box_b".to_string()),
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 6, coeff: 10 },
                    LinearTerm { var: 9, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                    LinearTerm { var: 10, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 2, 2, 0, 2, 2, 0, 1, 1, 2, 1, 3]);
        assert_eq!(sol.objective, Some(4));
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
                        presence: None,
                        name: Some("machine_a".to_string()),
                    },
                    CpDemandInterval {
                        start: 1,
                        duration: 2,
                        demand: 2,
                        presence: None,
                        name: Some("machine_b".to_string()),
                    },
                    CpDemandInterval {
                        start: 2,
                        duration: 2,
                        demand: 1,
                        presence: None,
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
    fn variable_cumulative_uses_duration_demand_and_capacity_vars() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "task_a_duration".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "task_a_end".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "task_a_demand".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "task_b_duration".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "task_b_end".to_string(),
                    domain: vec![1, 2, 3, 4, 5],
                },
                CpVariable {
                    name: "task_b_demand".to_string(),
                    domain: vec![1, 2],
                },
                CpVariable {
                    name: "capacity".to_string(),
                    domain: vec![3, 4],
                },
            ],
            constraints: vec![CpConstraint::CumulativeVariable {
                intervals: vec![
                    CpVariableDemandInterval {
                        start: 0,
                        duration: 1,
                        end: 2,
                        demand: 3,
                        presence: None,
                        name: Some("task_a".to_string()),
                    },
                    CpVariableDemandInterval {
                        start: 4,
                        duration: 5,
                        end: 6,
                        demand: 7,
                        presence: None,
                        name: Some("task_b".to_string()),
                    },
                ],
                capacity: 8,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 8, coeff: 100 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 3, 3, 2, 0, 1, 1, 1, 3]);
        assert_eq!(sol.objective, Some(302));
    }

    #[test]
    fn solves_reservoir_schedule() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "fill_time".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "drain_time".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "overfill_active".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Reservoir {
                events: vec![
                    CpReservoirEvent {
                        time: 0,
                        level_change: 4,
                        active: None,
                    },
                    CpReservoirEvent {
                        time: 1,
                        level_change: -3,
                        active: None,
                    },
                    CpReservoirEvent {
                        time: 1,
                        level_change: 10,
                        active: Some(BoolLiteral {
                            var: 2,
                            positive: true,
                        }),
                    },
                ],
                min_level: 0,
                max_level: 4,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 2, coeff: -1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 0, 0]);
        assert_eq!(sol.objective, Some(0));
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
    fn solves_variable_element_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "choice".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "expensive".to_string(),
                    domain: vec![4],
                },
                CpVariable {
                    name: "cheap".to_string(),
                    domain: vec![1, 3],
                },
                CpVariable {
                    name: "selected".to_string(),
                    domain: vec![1, 2, 3, 4],
                },
            ],
            constraints: vec![CpConstraint::VariableElement(CpVariableElement {
                index: 0,
                vars: vec![1, 2],
                target: 3,
            })],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 3, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 4, 1, 1]);
        assert_eq!(sol.objective, Some(1));
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
    fn enforced_table_constraints_respect_gate() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "active_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inactive_y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 1,
                    positive: false,
                }]),
                CpConstraint::EnforcedAllowedAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    vars: vec![2, 3],
                    tuples: vec![vec![0, 1], vec![1, 0]],
                },
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    vars: vec![2, 3],
                    tuples: vec![vec![0, 1]],
                },
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement: vec![BoolLiteral {
                        var: 1,
                        positive: true,
                    }],
                    vars: vec![4, 5],
                    tuples: vec![vec![0, 0]],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 0, 1, 0, 0, 0]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_forbidden_assignments_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "mode".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "handler".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::ForbiddenAssignments {
                vars: vec![0, 1],
                tuples: vec![vec![0, 0], vec![1, 0]],
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 3 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![2, 0]);
        assert_eq!(sol.objective, Some(2));
    }

    #[test]
    fn solves_inverse_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "direct_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "direct_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inverse_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "inverse_1".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![CpConstraint::Inverse {
                direct: vec![0, 1],
                inverse: vec![2, 3],
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 2 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 0, 1, 0]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_min_max_equality_constraints() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "score_a".to_string(),
                    domain: vec![2, 4],
                },
                CpVariable {
                    name: "score_b".to_string(),
                    domain: vec![3, 5],
                },
                CpVariable {
                    name: "max_score".to_string(),
                    domain: vec![3, 4, 5],
                },
                CpVariable {
                    name: "min_score".to_string(),
                    domain: vec![2, 3, 4],
                },
            ],
            constraints: vec![
                CpConstraint::MaxEquality {
                    target: 2,
                    vars: vec![0, 1],
                },
                CpConstraint::MinEquality {
                    target: 3,
                    vars: vec![0, 1],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![2, 3, 3, 2]);
        assert_eq!(sol.objective, Some(5));
    }

    #[test]
    fn solves_abs_equality_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "deviation".to_string(),
                    domain: vec![-3, -1, 2],
                },
                CpVariable {
                    name: "absolute_deviation".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
            ],
            constraints: vec![CpConstraint::AbsEquality { target: 1, var: 0 }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 1, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![-1, 1]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_multiplication_equality_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![-2, -1, 3],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![-3, 2],
                },
                CpVariable {
                    name: "product".to_string(),
                    domain: vec![-9, -4, -3, 2, 6],
                },
            ],
            constraints: vec![CpConstraint::MultiplicationEquality {
                target: 2,
                vars: vec![0, 1],
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 2, coeff: 1 }],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![3, -3, -9]);
        assert_eq!(sol.objective, Some(-9));
    }

    #[test]
    fn solves_division_equality_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "numerator".to_string(),
                    domain: vec![5, 6, 7],
                },
                CpVariable {
                    name: "denominator".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "quotient".to_string(),
                    domain: vec![2, 3],
                },
            ],
            constraints: vec![CpConstraint::DivisionEquality {
                target: 2,
                numerator: 0,
                denominator: 1,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 2, coeff: 10 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![5, 2, 2]);
        assert_eq!(sol.objective, Some(25));
    }

    #[test]
    fn solves_modulo_equality_constraint() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "value".to_string(),
                    domain: vec![5, 6, 7],
                },
                CpVariable {
                    name: "modulus".to_string(),
                    domain: vec![3],
                },
                CpVariable {
                    name: "remainder".to_string(),
                    domain: vec![0, 1, 2],
                },
            ],
            constraints: vec![CpConstraint::ModuloEquality {
                target: 2,
                var: 0,
                modulus: 1,
            }],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 2, coeff: 10 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![6, 3, 0]);
        assert_eq!(sol.objective, Some(6));
    }

    #[test]
    fn solves_automaton_constraint() {
        let model = CpModel {
            variables: (0..3)
                .map(|i| CpVariable {
                    name: format!("bit_{i}"),
                    domain: vec![0, 1],
                })
                .collect(),
            constraints: vec![CpConstraint::Automaton(CpAutomaton {
                vars: vec![0, 1, 2],
                starting_state: 0,
                final_states: vec![1],
                transitions: vec![
                    CpTransition {
                        tail: 0,
                        label: 0,
                        head: 0,
                    },
                    CpTransition {
                        tail: 0,
                        label: 1,
                        head: 1,
                    },
                    CpTransition {
                        tail: 1,
                        label: 0,
                        head: 1,
                    },
                    CpTransition {
                        tail: 1,
                        label: 1,
                        head: 2,
                    },
                    CpTransition {
                        tail: 2,
                        label: 0,
                        head: 2,
                    },
                    CpTransition {
                        tail: 2,
                        label: 1,
                        head: 2,
                    },
                ],
            })],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 4 },
                    LinearTerm { var: 1, coeff: 2 },
                    LinearTerm { var: 2, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![0, 0, 1]);
        assert_eq!(sol.objective, Some(1));
    }

    #[test]
    fn solves_circuit_constraint() {
        let model = CpModel {
            variables: (0..3)
                .map(|i| CpVariable {
                    name: format!("arc_{i}"),
                    domain: vec![0, 1],
                })
                .collect(),
            constraints: vec![CpConstraint::Circuit(vec![
                CpCircuitArc {
                    tail: 0,
                    head: 1,
                    literal: BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                },
                CpCircuitArc {
                    tail: 1,
                    head: 2,
                    literal: BoolLiteral {
                        var: 1,
                        positive: true,
                    },
                },
                CpCircuitArc {
                    tail: 2,
                    head: 0,
                    literal: BoolLiteral {
                        var: 2,
                        positive: true,
                    },
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 2 },
                    LinearTerm { var: 2, coeff: 3 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 1, 1]);
        assert_eq!(sol.objective, Some(6));
    }

    #[test]
    fn solves_multiple_circuit_routes() {
        let model = CpModel {
            variables: (0..4)
                .map(|i| CpVariable {
                    name: format!("route_arc_{i}"),
                    domain: vec![0, 1],
                })
                .collect(),
            constraints: vec![CpConstraint::MultipleCircuit(vec![
                CpCircuitArc {
                    tail: 0,
                    head: 1,
                    literal: BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                },
                CpCircuitArc {
                    tail: 1,
                    head: 0,
                    literal: BoolLiteral {
                        var: 1,
                        positive: true,
                    },
                },
                CpCircuitArc {
                    tail: 0,
                    head: 2,
                    literal: BoolLiteral {
                        var: 2,
                        positive: true,
                    },
                },
                CpCircuitArc {
                    tail: 2,
                    head: 0,
                    literal: BoolLiteral {
                        var: 3,
                        positive: true,
                    },
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 1 },
                    LinearTerm { var: 2, coeff: 1 },
                    LinearTerm { var: 3, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 1, 1, 1]);
        assert_eq!(sol.objective, Some(4));
    }

    #[test]
    fn solves_common_boolean_logic_constraints() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "choice_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "choice_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "choice_c".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "gate".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "approved".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "exclusive_flag".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::ExactlyOne(vec![
                    BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 1,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 2,
                        positive: true,
                    },
                ]),
                CpConstraint::AtMostOne(vec![
                    BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 2,
                        positive: true,
                    },
                ]),
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 3,
                        positive: true,
                    },
                },
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 3,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 4,
                        positive: true,
                    },
                },
                CpConstraint::BoolAnd(vec![
                    BoolLiteral {
                        var: 3,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 4,
                        positive: true,
                    },
                ]),
                CpConstraint::BoolXor(vec![
                    BoolLiteral {
                        var: 0,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 5,
                        positive: true,
                    },
                ]),
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 1 },
                    LinearTerm { var: 1, coeff: 5 },
                    LinearTerm { var: 2, coeff: 4 },
                    LinearTerm { var: 3, coeff: 1 },
                ],
            }),
        };
        let sol = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(sol.status, CpStatus::Optimal);
        assert_eq!(sol.assignment, vec![1, 0, 0, 1, 1, 0]);
        assert_eq!(sol.objective, Some(2));
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

    #[test]
    fn enforced_bool_constraints_respect_gate() {
        let active_model = CpModel {
            variables: vec![
                CpVariable {
                    name: "use_rule".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "x".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "y".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: true,
                }]),
                CpConstraint::EnforcedBoolOr {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 1,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                    ],
                },
                CpConstraint::EnforcedBoolAnd {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![BoolLiteral {
                        var: 2,
                        positive: true,
                    }],
                },
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 1, coeff: 1 },
                    LinearTerm { var: 2, coeff: 1 },
                ],
            }),
        };
        let active = solve_cp_model(&active_model, CpSolveOptions::default());
        assert_eq!(active.status, CpStatus::Optimal);
        assert_eq!(active.assignment, vec![1, 0, 1]);
        assert_eq!(active.objective, Some(1));

        let inactive_model = CpModel {
            variables: active_model.variables,
            constraints: vec![
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 0,
                    positive: false,
                }]),
                CpConstraint::EnforcedBoolAnd {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    literals: vec![
                        BoolLiteral {
                            var: 1,
                            positive: true,
                        },
                        BoolLiteral {
                            var: 2,
                            positive: true,
                        },
                    ],
                },
            ],
            objective: active_model.objective,
        };
        let inactive = solve_cp_model(&inactive_model, CpSolveOptions::default());
        assert_eq!(inactive.status, CpStatus::Optimal);
        assert_eq!(inactive.assignment, vec![0, 0, 0]);
        assert_eq!(inactive.objective, Some(0));
    }
}
