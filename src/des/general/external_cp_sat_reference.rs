//! Rust-facing bridge for CP-SAT and constraint-programming reference checks.
//!
//! The native Rust fallback accepts the crate's compact CP-SAT JSON model and
//! enumerates small finite-domain validation models without a Python dependency.
//! `scripts/cp_sat_reference.py` remains available for OR-Tools CP-SAT and
//! legacy Python fallback checks. Broader CP ecosystems such as Choco, JaCoP,
//! CPMpy, Conjure, clingo, SAT4J, and Open-WBO use the
//! `optimization_ecosystem_reference.py` smoke-model contract instead; this
//! module exposes both paths without pretending they share one model format.

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::cp_sat::{
    enumerate_cp_solutions, find_cp_assumption_unsat_core, solve_cp_model, BoolLiteral,
    CpAlternative, CpAssumptionCoreOptions, CpAutomaton, CpCircuitArc, CpConstraint,
    CpDecisionStrategy, CpDemandInterval, CpDomainInterval, CpDomainValueStrategy, CpElement,
    CpEnumerateOptions, CpInterval, CpModel, CpObjective, CpRectangle, CpReservoirEvent,
    CpSolutionHint, CpSolveOptions, CpStatus, CpTransition, CpVariable, CpVariableDemandInterval,
    CpVariableElement, CpVariableInterval, CpVariableRectangle, CpVariableSelectionStrategy,
    LinearSense, LinearTerm, ObjectiveSense,
};
use crate::des::general::external_optimization_tools::{
    run_external_optimization_ecosystem_reference, ExternalOptimizationAdapterStatus,
    ExternalOptimizationTool,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceSolver {
    Auto,
    OrToolsCpSat,
    RustEnumeration,
    PythonEnumeration,
    ChocoSolver,
    JaCoP,
    IbmCpOptimizer,
    OrToolsJava,
    OrToolsPython,
    Cpmpy,
    PyCsp3,
    Conjure,
    SavileRow,
    Picat,
    Clingo,
    Clingcon,
    Sat4j,
    PySat,
    OpenWbo,
}

impl ExternalCpSatReferenceSolver {
    pub fn all() -> &'static [ExternalCpSatReferenceSolver] {
        &[
            ExternalCpSatReferenceSolver::Auto,
            ExternalCpSatReferenceSolver::OrToolsCpSat,
            ExternalCpSatReferenceSolver::RustEnumeration,
            ExternalCpSatReferenceSolver::PythonEnumeration,
            ExternalCpSatReferenceSolver::ChocoSolver,
            ExternalCpSatReferenceSolver::JaCoP,
            ExternalCpSatReferenceSolver::IbmCpOptimizer,
            ExternalCpSatReferenceSolver::OrToolsJava,
            ExternalCpSatReferenceSolver::OrToolsPython,
            ExternalCpSatReferenceSolver::Cpmpy,
            ExternalCpSatReferenceSolver::PyCsp3,
            ExternalCpSatReferenceSolver::Conjure,
            ExternalCpSatReferenceSolver::SavileRow,
            ExternalCpSatReferenceSolver::Picat,
            ExternalCpSatReferenceSolver::Clingo,
            ExternalCpSatReferenceSolver::Clingcon,
            ExternalCpSatReferenceSolver::Sat4j,
            ExternalCpSatReferenceSolver::PySat,
            ExternalCpSatReferenceSolver::OpenWbo,
        ]
    }

    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalCpSatReferenceSolver::Auto => "auto",
            ExternalCpSatReferenceSolver::OrToolsCpSat => "ortools-cp-sat",
            ExternalCpSatReferenceSolver::RustEnumeration => "rust-enumeration",
            ExternalCpSatReferenceSolver::PythonEnumeration => "python-enumeration",
            ExternalCpSatReferenceSolver::ChocoSolver => "choco-solver",
            ExternalCpSatReferenceSolver::JaCoP => "jacop",
            ExternalCpSatReferenceSolver::IbmCpOptimizer => "ibm-cp-optimizer",
            ExternalCpSatReferenceSolver::OrToolsJava => "ortools-java",
            ExternalCpSatReferenceSolver::OrToolsPython => "ortools-python",
            ExternalCpSatReferenceSolver::Cpmpy => "cpmpy",
            ExternalCpSatReferenceSolver::PyCsp3 => "pycsp3",
            ExternalCpSatReferenceSolver::Conjure => "conjure",
            ExternalCpSatReferenceSolver::SavileRow => "savile-row",
            ExternalCpSatReferenceSolver::Picat => "picat",
            ExternalCpSatReferenceSolver::Clingo => "clingo",
            ExternalCpSatReferenceSolver::Clingcon => "clingcon",
            ExternalCpSatReferenceSolver::Sat4j => "sat4j",
            ExternalCpSatReferenceSolver::PySat => "pysat",
            ExternalCpSatReferenceSolver::OpenWbo => "open-wbo",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalCpSatReferenceSolver::Auto => "Auto",
            ExternalCpSatReferenceSolver::OrToolsCpSat => "Google OR-Tools CP-SAT",
            ExternalCpSatReferenceSolver::RustEnumeration => "Rust exact CP enumeration",
            ExternalCpSatReferenceSolver::PythonEnumeration => {
                "Dependency-free exact CP enumeration"
            }
            ExternalCpSatReferenceSolver::ChocoSolver => "Choco Solver",
            ExternalCpSatReferenceSolver::JaCoP => "JaCoP",
            ExternalCpSatReferenceSolver::IbmCpOptimizer => "IBM ILOG CP Optimizer",
            ExternalCpSatReferenceSolver::OrToolsJava => "OR-Tools Java",
            ExternalCpSatReferenceSolver::OrToolsPython => "OR-Tools Python",
            ExternalCpSatReferenceSolver::Cpmpy => "CPMpy",
            ExternalCpSatReferenceSolver::PyCsp3 => "PyCSP3",
            ExternalCpSatReferenceSolver::Conjure => "Conjure",
            ExternalCpSatReferenceSolver::SavileRow => "Savile Row",
            ExternalCpSatReferenceSolver::Picat => "Picat",
            ExternalCpSatReferenceSolver::Clingo => "clingo",
            ExternalCpSatReferenceSolver::Clingcon => "clingcon",
            ExternalCpSatReferenceSolver::Sat4j => "SAT4J",
            ExternalCpSatReferenceSolver::PySat => "PySAT",
            ExternalCpSatReferenceSolver::OpenWbo => "Open-WBO",
        }
    }

    pub fn family(self) -> ExternalCpSatReferenceFamily {
        match self {
            ExternalCpSatReferenceSolver::Auto => ExternalCpSatReferenceFamily::Auto,
            ExternalCpSatReferenceSolver::OrToolsCpSat => ExternalCpSatReferenceFamily::CpSatScript,
            ExternalCpSatReferenceSolver::RustEnumeration
            | ExternalCpSatReferenceSolver::PythonEnumeration => {
                ExternalCpSatReferenceFamily::Fallback
            }
            ExternalCpSatReferenceSolver::ChocoSolver
            | ExternalCpSatReferenceSolver::JaCoP
            | ExternalCpSatReferenceSolver::IbmCpOptimizer
            | ExternalCpSatReferenceSolver::OrToolsJava
            | ExternalCpSatReferenceSolver::OrToolsPython
            | ExternalCpSatReferenceSolver::Cpmpy
            | ExternalCpSatReferenceSolver::PyCsp3
            | ExternalCpSatReferenceSolver::Conjure
            | ExternalCpSatReferenceSolver::SavileRow
            | ExternalCpSatReferenceSolver::Picat
            | ExternalCpSatReferenceSolver::Clingo
            | ExternalCpSatReferenceSolver::Clingcon
            | ExternalCpSatReferenceSolver::Sat4j
            | ExternalCpSatReferenceSolver::PySat
            | ExternalCpSatReferenceSolver::OpenWbo => {
                ExternalCpSatReferenceFamily::EcosystemReference
            }
        }
    }

    pub fn direct_cp_sat_json_solver_arg(self) -> Option<&'static str> {
        match self {
            ExternalCpSatReferenceSolver::Auto => Some("auto"),
            ExternalCpSatReferenceSolver::OrToolsCpSat => Some("ortools-cp-sat"),
            ExternalCpSatReferenceSolver::RustEnumeration => Some("rust-enumeration"),
            ExternalCpSatReferenceSolver::PythonEnumeration => Some("fallback"),
            _ => None,
        }
    }

    pub fn ecosystem_tool(self) -> Option<ExternalOptimizationTool> {
        match self {
            ExternalCpSatReferenceSolver::ChocoSolver => {
                Some(ExternalOptimizationTool::ChocoSolver)
            }
            ExternalCpSatReferenceSolver::JaCoP => Some(ExternalOptimizationTool::Jacop),
            ExternalCpSatReferenceSolver::IbmCpOptimizer => {
                Some(ExternalOptimizationTool::IbmCpOptimizer)
            }
            ExternalCpSatReferenceSolver::OrToolsJava => {
                Some(ExternalOptimizationTool::OrToolsJava)
            }
            ExternalCpSatReferenceSolver::OrToolsPython => {
                Some(ExternalOptimizationTool::OrToolsPython)
            }
            ExternalCpSatReferenceSolver::Cpmpy => Some(ExternalOptimizationTool::Cpmpy),
            ExternalCpSatReferenceSolver::PyCsp3 => Some(ExternalOptimizationTool::PyCsp3),
            ExternalCpSatReferenceSolver::Conjure => Some(ExternalOptimizationTool::Conjure),
            ExternalCpSatReferenceSolver::SavileRow => Some(ExternalOptimizationTool::SavileRow),
            ExternalCpSatReferenceSolver::Picat => Some(ExternalOptimizationTool::Picat),
            ExternalCpSatReferenceSolver::Clingo => Some(ExternalOptimizationTool::Clingo),
            ExternalCpSatReferenceSolver::Clingcon => Some(ExternalOptimizationTool::Clingcon),
            ExternalCpSatReferenceSolver::Sat4j => Some(ExternalOptimizationTool::Sat4j),
            ExternalCpSatReferenceSolver::PySat => Some(ExternalOptimizationTool::PySat),
            ExternalCpSatReferenceSolver::OpenWbo => Some(ExternalOptimizationTool::OpenWbo),
            _ => None,
        }
    }

    pub fn supports_cp_sat_json(self) -> bool {
        self.direct_cp_sat_json_solver_arg().is_some()
    }

    pub fn supports_ecosystem_cp_assignment(self) -> bool {
        self.ecosystem_tool().is_some()
    }

    pub fn notes(self) -> &'static str {
        match self {
            ExternalCpSatReferenceSolver::Auto => {
                "Use the configured direct CP-SAT bridge; rust-enumeration is the dependency-free same-input fallback for small models."
            }
            ExternalCpSatReferenceSolver::OrToolsCpSat => {
                "Direct same-input OR-Tools CP-SAT bridge through scripts/cp_sat_reference.py."
            }
            ExternalCpSatReferenceSolver::RustEnumeration => {
                "Native Rust exact enumeration for small finite-domain CP-SAT JSON models."
            }
            ExternalCpSatReferenceSolver::PythonEnumeration => {
                "Legacy exact enumeration bridge through scripts/cp_sat_reference.py."
            }
            _ => {
                "Ecosystem smoke bridge through scripts/optimization_ecosystem_reference.py; uses the ecosystem CP-assignment contract rather than the CP-SAT JSON model."
            }
        }
    }

    pub fn spec(self) -> ExternalCpSatReferenceSolverSpec {
        ExternalCpSatReferenceSolverSpec {
            solver: self,
            id: self.as_arg(),
            display_name: self.display_name(),
            family: self.family(),
            supports_cp_sat_json: self.supports_cp_sat_json(),
            supports_ecosystem_cp_assignment: self.supports_ecosystem_cp_assignment(),
            notes: self.notes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceFamily {
    Auto,
    CpSatScript,
    EcosystemReference,
    Fallback,
}

impl ExternalCpSatReferenceFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalCpSatReferenceFamily::Auto => "auto",
            ExternalCpSatReferenceFamily::CpSatScript => "cp-sat-script",
            ExternalCpSatReferenceFamily::EcosystemReference => "ecosystem-reference",
            ExternalCpSatReferenceFamily::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalCpSatReferenceSolverSpec {
    pub solver: ExternalCpSatReferenceSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ExternalCpSatReferenceFamily,
    pub supports_cp_sat_json: bool,
    pub supports_ecosystem_cp_assignment: bool,
    pub notes: &'static str,
}

pub fn external_cp_sat_reference_solver_specs() -> Vec<ExternalCpSatReferenceSolverSpec> {
    ExternalCpSatReferenceSolver::all()
        .iter()
        .copied()
        .map(ExternalCpSatReferenceSolver::spec)
        .collect()
}

pub fn external_cp_sat_reference_solver_manifest() -> Value {
    Value::Array(
        external_cp_sat_reference_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "family": spec.family.as_str(),
                    "supportsCpSatJson": spec.supports_cp_sat_json,
                    "supportsEcosystemCpAssignment": spec.supports_ecosystem_cp_assignment,
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
}

pub fn cp_sat_model_to_reference_json(model: &CpModel) -> Value {
    let variables: Vec<_> = model
        .variables
        .iter()
        .map(cp_variable_to_reference_json)
        .collect();
    let constraints: Vec<_> = model
        .constraints
        .iter()
        .map(cp_constraint_to_reference_json)
        .collect();
    let objective = model.objective.as_ref().map(cp_objective_to_reference_json);

    json!({
        "variables": variables,
        "constraints": constraints,
        "objective": objective,
    })
}

pub fn cp_sat_model_to_reference_json_string(model: &CpModel) -> String {
    cp_sat_model_to_reference_json(model).to_string()
}

fn cp_variable_to_reference_json(variable: &CpVariable) -> Value {
    json!({
        "name": variable.name,
        "domain": variable.domain,
    })
}

fn cp_terms_to_reference_json(terms: &[LinearTerm]) -> Vec<Value> {
    terms
        .iter()
        .map(|term| {
            json!({
                "var": term.var,
                "coeff": term.coeff,
            })
        })
        .collect()
}

fn cp_literal_to_reference_json(literal: &BoolLiteral) -> Value {
    json!({
        "var": literal.var,
        "positive": literal.positive,
    })
}

fn cp_optional_literal_to_reference_json(literal: Option<&BoolLiteral>) -> Value {
    literal
        .map(cp_literal_to_reference_json)
        .unwrap_or(Value::Null)
}

fn cp_literals_to_reference_json(literals: &[BoolLiteral]) -> Vec<Value> {
    literals.iter().map(cp_literal_to_reference_json).collect()
}

fn cp_domain_intervals_to_reference_json(intervals: &[CpDomainInterval]) -> Vec<Value> {
    intervals
        .iter()
        .map(|interval| {
            json!({
                "lb": interval.lb,
                "ub": interval.ub,
            })
        })
        .collect()
}

fn cp_interval_to_reference_json(interval: &CpInterval) -> Value {
    json!({
        "start": interval.start,
        "duration": interval.duration,
        "presence": cp_optional_literal_to_reference_json(interval.presence.as_ref()),
        "name": interval.name,
    })
}

fn cp_variable_interval_to_reference_json(interval: &CpVariableInterval) -> Value {
    json!({
        "start": interval.start,
        "duration": interval.duration,
        "end": interval.end,
        "presence": cp_optional_literal_to_reference_json(interval.presence.as_ref()),
        "name": interval.name,
    })
}

fn cp_demand_interval_to_reference_json(interval: &CpDemandInterval) -> Value {
    json!({
        "start": interval.start,
        "duration": interval.duration,
        "demand": interval.demand,
        "presence": cp_optional_literal_to_reference_json(interval.presence.as_ref()),
        "name": interval.name,
    })
}

fn cp_variable_demand_interval_to_reference_json(interval: &CpVariableDemandInterval) -> Value {
    json!({
        "start": interval.start,
        "duration": interval.duration,
        "end": interval.end,
        "demand": interval.demand,
        "presence": cp_optional_literal_to_reference_json(interval.presence.as_ref()),
        "name": interval.name,
    })
}

fn cp_rectangle_to_reference_json(rectangle: &CpRectangle) -> Value {
    json!({
        "x_start": rectangle.x_start,
        "y_start": rectangle.y_start,
        "width": rectangle.width,
        "height": rectangle.height,
        "presence": cp_optional_literal_to_reference_json(rectangle.presence.as_ref()),
        "name": rectangle.name,
    })
}

fn cp_variable_rectangle_to_reference_json(rectangle: &CpVariableRectangle) -> Value {
    json!({
        "x_start": rectangle.x_start,
        "x_size": rectangle.x_size,
        "x_end": rectangle.x_end,
        "y_start": rectangle.y_start,
        "y_size": rectangle.y_size,
        "y_end": rectangle.y_end,
        "presence": cp_optional_literal_to_reference_json(rectangle.presence.as_ref()),
        "name": rectangle.name,
    })
}

fn cp_objective_to_reference_json(objective: &CpObjective) -> Value {
    json!({
        "sense": objective.sense.as_str(),
        "terms": cp_terms_to_reference_json(&objective.terms),
    })
}

fn cp_constraint_to_reference_json(constraint: &CpConstraint) -> Value {
    match constraint {
        CpConstraint::Linear { terms, sense, rhs } => json!({
            "kind": "linear",
            "terms": cp_terms_to_reference_json(terms),
            "sense": sense.as_str(),
            "rhs": rhs,
        }),
        CpConstraint::LinearDomain { terms, intervals } => json!({
            "kind": "linear_domain",
            "terms": cp_terms_to_reference_json(terms),
            "intervals": cp_domain_intervals_to_reference_json(intervals),
        }),
        CpConstraint::EnforcedLinearDomain {
            enforcement,
            terms,
            intervals,
        } => json!({
            "kind": "enforced_linear_domain",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "terms": cp_terms_to_reference_json(terms),
            "intervals": cp_domain_intervals_to_reference_json(intervals),
        }),
        CpConstraint::MapDomain { var, bools, offset } => json!({
            "kind": "map_domain",
            "var": var,
            "bools": bools,
            "offset": offset,
        }),
        CpConstraint::EnforcedLinear {
            enforcement,
            terms,
            sense,
            rhs,
        } => json!({
            "kind": "enforced_linear",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "terms": cp_terms_to_reference_json(terms),
            "sense": sense.as_str(),
            "rhs": rhs,
        }),
        CpConstraint::EnforcedBoolOr {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_bool_or",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::EnforcedBoolAnd {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_bool_and",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::EnforcedBoolXor {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_bool_xor",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::EnforcedAtMostOne {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_at_most_one",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::EnforcedAtLeastOne {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_at_least_one",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::EnforcedExactlyOne {
            enforcement,
            literals,
        } => json!({
            "kind": "enforced_exactly_one",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::AllDifferent(vars) => json!({
            "kind": "all_different",
            "vars": vars,
        }),
        CpConstraint::BoolOr(literals) => json!({
            "kind": "bool_or",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::BoolAnd(literals) => json!({
            "kind": "bool_and",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::BoolXor(literals) => json!({
            "kind": "bool_xor",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::AtMostOne(literals) => json!({
            "kind": "at_most_one",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::AtLeastOne(literals) => json!({
            "kind": "at_least_one",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::ExactlyOne(literals) => json!({
            "kind": "exactly_one",
            "literals": cp_literals_to_reference_json(literals),
        }),
        CpConstraint::Implication {
            antecedent,
            consequent,
        } => json!({
            "kind": "implication",
            "antecedent": cp_literal_to_reference_json(antecedent),
            "consequent": cp_literal_to_reference_json(consequent),
        }),
        CpConstraint::AllowedAssignments { vars, tuples } => json!({
            "kind": "allowed_assignments",
            "vars": vars,
            "tuples": tuples,
        }),
        CpConstraint::ForbiddenAssignments { vars, tuples } => json!({
            "kind": "forbidden_assignments",
            "vars": vars,
            "tuples": tuples,
        }),
        CpConstraint::EnforcedAllowedAssignments {
            enforcement,
            vars,
            tuples,
        } => json!({
            "kind": "enforced_allowed_assignments",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "vars": vars,
            "tuples": tuples,
        }),
        CpConstraint::EnforcedForbiddenAssignments {
            enforcement,
            vars,
            tuples,
        } => json!({
            "kind": "enforced_forbidden_assignments",
            "enforcement": cp_literals_to_reference_json(enforcement),
            "vars": vars,
            "tuples": tuples,
        }),
        CpConstraint::Inverse { direct, inverse } => json!({
            "kind": "inverse",
            "direct": direct,
            "inverse": inverse,
        }),
        CpConstraint::MaxEquality { target, vars } => json!({
            "kind": "max_equality",
            "target": target,
            "vars": vars,
        }),
        CpConstraint::MinEquality { target, vars } => json!({
            "kind": "min_equality",
            "target": target,
            "vars": vars,
        }),
        CpConstraint::AbsEquality { target, var } => json!({
            "kind": "abs_equality",
            "target": target,
            "var": var,
        }),
        CpConstraint::MultiplicationEquality { target, vars } => json!({
            "kind": "multiplication_equality",
            "target": target,
            "vars": vars,
        }),
        CpConstraint::DivisionEquality {
            target,
            numerator,
            denominator,
        } => json!({
            "kind": "division_equality",
            "target": target,
            "numerator": numerator,
            "denominator": denominator,
        }),
        CpConstraint::ModuloEquality {
            target,
            var,
            modulus,
        } => json!({
            "kind": "modulo_equality",
            "target": target,
            "var": var,
            "modulus": modulus,
        }),
        CpConstraint::Automaton(automaton) => json!({
            "kind": "automaton",
            "vars": automaton.vars,
            "starting_state": automaton.starting_state,
            "final_states": automaton.final_states,
            "transitions": automaton.transitions.iter().map(|transition| json!({
                "tail": transition.tail,
                "label": transition.label,
                "head": transition.head,
            })).collect::<Vec<_>>(),
        }),
        CpConstraint::Circuit(arcs) => json!({
            "kind": "circuit",
            "arcs": arcs.iter().map(|arc| json!({
                "tail": arc.tail,
                "head": arc.head,
                "literal": cp_literal_to_reference_json(&arc.literal),
            })).collect::<Vec<_>>(),
        }),
        CpConstraint::MultipleCircuit(arcs) => json!({
            "kind": "multiple_circuit",
            "arcs": arcs.iter().map(|arc| json!({
                "tail": arc.tail,
                "head": arc.head,
                "literal": cp_literal_to_reference_json(&arc.literal),
            })).collect::<Vec<_>>(),
        }),
        CpConstraint::Element(element) => json!({
            "kind": "element",
            "index": element.index,
            "values": element.values,
            "target": element.target,
        }),
        CpConstraint::VariableElement(element) => json!({
            "kind": "variable_element",
            "index": element.index,
            "vars": element.vars,
            "target": element.target,
        }),
        CpConstraint::Alternative(alternative) => json!({
            "kind": "alternative",
            "start": alternative.start,
            "duration": alternative.duration,
            "end": alternative.end,
            "presence": cp_optional_literal_to_reference_json(alternative.presence.as_ref()),
            "alternatives": alternative.alternatives.iter().map(cp_variable_interval_to_reference_json).collect::<Vec<_>>(),
            "name": alternative.name,
        }),
        CpConstraint::NoOverlap(intervals) => json!({
            "kind": "no_overlap",
            "intervals": intervals.iter().map(cp_interval_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::NoOverlapVariable(intervals) => json!({
            "kind": "no_overlap_variable",
            "intervals": intervals.iter().map(cp_variable_interval_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::NoOverlap2D(rectangles) => json!({
            "kind": "no_overlap_2d",
            "rectangles": rectangles.iter().map(cp_rectangle_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::NoOverlap2DVariable(rectangles) => json!({
            "kind": "no_overlap_2d_variable",
            "rectangles": rectangles.iter().map(cp_variable_rectangle_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::Cumulative {
            intervals,
            capacity,
        } => json!({
            "kind": "cumulative",
            "capacity": capacity,
            "intervals": intervals.iter().map(cp_demand_interval_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::CumulativeVariable {
            intervals,
            capacity,
        } => json!({
            "kind": "cumulative_variable",
            "capacity": capacity,
            "intervals": intervals.iter().map(cp_variable_demand_interval_to_reference_json).collect::<Vec<_>>(),
        }),
        CpConstraint::Reservoir {
            events,
            min_level,
            max_level,
        } => json!({
            "kind": "reservoir",
            "min_level": min_level,
            "max_level": max_level,
            "events": events.iter().map(|event| json!({
                "time": event.time,
                "level_change": event.level_change,
                "active": cp_optional_literal_to_reference_json(event.active.as_ref()),
            })).collect::<Vec<_>>(),
        }),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCpSatReferenceOptions {
    pub solver: ExternalCpSatReferenceSolver,
    pub enumerate_solutions: Option<usize>,
    pub assumption_core: bool,
}

impl Default for ExternalCpSatReferenceOptions {
    fn default() -> Self {
        ExternalCpSatReferenceOptions {
            solver: ExternalCpSatReferenceSolver::Auto,
            enumerate_solutions: None,
            assumption_core: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalCpSatReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Exhausted,
    Unavailable,
    Invalid,
    Unsupported,
    Failed,
    Unknown,
}

impl ExternalCpSatReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalCpSatReferenceStatus::Optimal => "optimal",
            ExternalCpSatReferenceStatus::Feasible => "feasible",
            ExternalCpSatReferenceStatus::Infeasible => "infeasible",
            ExternalCpSatReferenceStatus::Exhausted => "exhausted",
            ExternalCpSatReferenceStatus::Unavailable => "unavailable",
            ExternalCpSatReferenceStatus::Invalid => "invalid",
            ExternalCpSatReferenceStatus::Unsupported => "unsupported",
            ExternalCpSatReferenceStatus::Failed => "failed",
            ExternalCpSatReferenceStatus::Unknown => "unknown",
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label {
            "optimal" => ExternalCpSatReferenceStatus::Optimal,
            "feasible" => ExternalCpSatReferenceStatus::Feasible,
            "infeasible" => ExternalCpSatReferenceStatus::Infeasible,
            "exhausted" => ExternalCpSatReferenceStatus::Exhausted,
            "unavailable" => ExternalCpSatReferenceStatus::Unavailable,
            "invalid" => ExternalCpSatReferenceStatus::Invalid,
            "unsupported" => ExternalCpSatReferenceStatus::Unsupported,
            "failed" => ExternalCpSatReferenceStatus::Failed,
            _ => ExternalCpSatReferenceStatus::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCpSatReferenceRun {
    pub solver: ExternalCpSatReferenceSolver,
    pub backend: String,
    pub status: ExternalCpSatReferenceStatus,
    pub assignment: Vec<i64>,
    pub objective: Option<f64>,
    pub nodes: Option<u64>,
    pub raw: Value,
    pub elapsed_ms: f64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct CpSatScriptOutput {
    status: String,
    #[serde(default)]
    solver: String,
    #[serde(default)]
    assignment: Vec<i64>,
    #[serde(default)]
    objective: Option<f64>,
    #[serde(default)]
    nodes: Option<u64>,
    #[serde(default)]
    message: String,
}

pub fn external_cp_sat_reference_script() -> PathBuf {
    let root = env::var_os("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("cp_sat_reference.py")
}

fn python_command() -> PathBuf {
    env::var_os("PYTHON_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn script_working_dir(script: &Path) -> Option<PathBuf> {
    script
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

fn cp_sat_error_run(
    solver: ExternalCpSatReferenceSolver,
    status: ExternalCpSatReferenceStatus,
    message: impl Into<String>,
    started: Instant,
) -> ExternalCpSatReferenceRun {
    let message = message.into();
    ExternalCpSatReferenceRun {
        solver,
        backend: "rust:cp-enumeration".to_string(),
        status,
        assignment: Vec::new(),
        objective: None,
        nodes: Some(0),
        raw: json!({
            "status": status.as_str(),
            "solver": "rust:cp-enumeration",
            "message": message,
        }),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message,
    }
}

fn native_cp_status(status: CpStatus) -> ExternalCpSatReferenceStatus {
    match status {
        CpStatus::Optimal => ExternalCpSatReferenceStatus::Optimal,
        CpStatus::Feasible => ExternalCpSatReferenceStatus::Feasible,
        CpStatus::Infeasible => ExternalCpSatReferenceStatus::Infeasible,
    }
}

fn native_cp_object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

fn native_cp_array<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array `{key}`"))
}

fn native_cp_optional_array<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<&'a Vec<Value>>, String> {
    match object.get(key) {
        Some(Value::Array(values)) => Ok(Some(values)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(format!("`{key}` must be an array")),
    }
}

fn native_cp_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string `{key}`"))
}

fn native_cp_i64_value(value: &Value, context: &str) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| format!("{context} must be an integer"))
}

fn native_cp_i64(object: &serde_json::Map<String, Value>, key: &str) -> Result<i64, String> {
    object
        .get(key)
        .ok_or_else(|| format!("missing `{key}`"))
        .and_then(|value| native_cp_i64_value(value, key))
}

fn native_cp_usize_value(value: &Value, context: &str) -> Result<usize, String> {
    let raw = native_cp_i64_value(value, context)?;
    usize::try_from(raw).map_err(|_| format!("{context} must be non-negative"))
}

fn native_cp_usize(object: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    object
        .get(key)
        .ok_or_else(|| format!("missing `{key}`"))
        .and_then(|value| native_cp_usize_value(value, key))
}

fn native_cp_bool(object: &serde_json::Map<String, Value>, key: &str, default: bool) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn native_cp_name(object: &serde_json::Map<String, Value>) -> Result<Option<String>, String> {
    match object.get("name") {
        Some(Value::String(name)) => Ok(Some(name.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err("`name` must be a string".to_string()),
    }
}

fn native_cp_indices(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<usize>, String> {
    native_cp_array(object, key)?
        .iter()
        .enumerate()
        .map(|(idx, value)| native_cp_usize_value(value, &format!("{key}[{idx}]")))
        .collect()
}

fn native_cp_i64_values(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<i64>, String> {
    native_cp_array(object, key)?
        .iter()
        .enumerate()
        .map(|(idx, value)| native_cp_i64_value(value, &format!("{key}[{idx}]")))
        .collect()
}

fn native_cp_terms(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<LinearTerm>, String> {
    native_cp_array(object, key)?
        .iter()
        .map(|value| {
            let term = native_cp_object(value, "linear term")?;
            Ok(LinearTerm {
                var: native_cp_usize(term, "var")?,
                coeff: native_cp_i64(term, "coeff")?,
            })
        })
        .collect()
}

fn native_cp_literal(value: &Value) -> Result<BoolLiteral, String> {
    let object = native_cp_object(value, "literal")?;
    Ok(BoolLiteral {
        var: native_cp_usize(object, "var")?,
        positive: native_cp_bool(object, "positive", true),
    })
}

fn native_cp_literals(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<BoolLiteral>, String> {
    native_cp_array(object, key)?
        .iter()
        .map(native_cp_literal)
        .collect()
}

fn native_cp_optional_literal(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<BoolLiteral>, String> {
    match object.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => native_cp_literal(value).map(Some),
    }
}

fn native_cp_intervals(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<CpDomainInterval>, String> {
    native_cp_array(object, key)?
        .iter()
        .map(|value| {
            let interval = native_cp_object(value, "domain interval")?;
            Ok(CpDomainInterval {
                lb: native_cp_i64(interval, "lb")?,
                ub: native_cp_i64(interval, "ub")?,
            })
        })
        .collect()
}

fn native_cp_tuples(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<Vec<i64>>, String> {
    native_cp_array(object, key)?
        .iter()
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| "tuple entry must be an array".to_string())?
                .iter()
                .enumerate()
                .map(|(idx, value)| native_cp_i64_value(value, &format!("tuple[{idx}]")))
                .collect()
        })
        .collect()
}

fn native_cp_linear_sense(object: &serde_json::Map<String, Value>) -> Result<LinearSense, String> {
    match native_cp_string(object, "sense")? {
        "le" => Ok(LinearSense::Le),
        "ge" => Ok(LinearSense::Ge),
        "eq" => Ok(LinearSense::Eq),
        other => Err(format!("unsupported linear sense `{other}`")),
    }
}

fn native_cp_objective_sense(
    object: &serde_json::Map<String, Value>,
) -> Result<ObjectiveSense, String> {
    match object.get("sense").and_then(Value::as_str).unwrap_or("min") {
        "min" => Ok(ObjectiveSense::Min),
        "max" => Ok(ObjectiveSense::Max),
        other => Err(format!("unsupported objective sense `{other}`")),
    }
}

fn native_cp_fixed_interval(value: &Value) -> Result<CpInterval, String> {
    let object = native_cp_object(value, "fixed interval")?;
    Ok(CpInterval {
        start: native_cp_usize(object, "start")?,
        duration: native_cp_i64(object, "duration")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_variable_interval(value: &Value) -> Result<CpVariableInterval, String> {
    let object = native_cp_object(value, "variable interval")?;
    Ok(CpVariableInterval {
        start: native_cp_usize(object, "start")?,
        duration: native_cp_usize(object, "duration")?,
        end: native_cp_usize(object, "end")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_demand_interval(value: &Value) -> Result<CpDemandInterval, String> {
    let object = native_cp_object(value, "demand interval")?;
    Ok(CpDemandInterval {
        start: native_cp_usize(object, "start")?,
        duration: native_cp_i64(object, "duration")?,
        demand: native_cp_i64(object, "demand")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_variable_demand_interval(value: &Value) -> Result<CpVariableDemandInterval, String> {
    let object = native_cp_object(value, "variable demand interval")?;
    Ok(CpVariableDemandInterval {
        start: native_cp_usize(object, "start")?,
        duration: native_cp_usize(object, "duration")?,
        end: native_cp_usize(object, "end")?,
        demand: native_cp_usize(object, "demand")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_rectangle(value: &Value) -> Result<CpRectangle, String> {
    let object = native_cp_object(value, "rectangle")?;
    Ok(CpRectangle {
        x_start: native_cp_usize(object, "x_start")?,
        y_start: native_cp_usize(object, "y_start")?,
        width: native_cp_i64(object, "width")?,
        height: native_cp_i64(object, "height")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_variable_rectangle(value: &Value) -> Result<CpVariableRectangle, String> {
    let object = native_cp_object(value, "variable rectangle")?;
    Ok(CpVariableRectangle {
        x_start: native_cp_usize(object, "x_start")?,
        x_size: native_cp_usize(object, "x_size")?,
        x_end: native_cp_usize(object, "x_end")?,
        y_start: native_cp_usize(object, "y_start")?,
        y_size: native_cp_usize(object, "y_size")?,
        y_end: native_cp_usize(object, "y_end")?,
        presence: native_cp_optional_literal(object, "presence")?,
        name: native_cp_name(object)?,
    })
}

fn native_cp_constraint(value: &Value) -> Result<CpConstraint, String> {
    let object = native_cp_object(value, "constraint")?;
    match native_cp_string(object, "kind")? {
        "linear" => Ok(CpConstraint::Linear {
            terms: native_cp_terms(object, "terms")?,
            sense: native_cp_linear_sense(object)?,
            rhs: native_cp_i64(object, "rhs")?,
        }),
        "linear_domain" => Ok(CpConstraint::LinearDomain {
            terms: native_cp_terms(object, "terms")?,
            intervals: native_cp_intervals(object, "intervals")?,
        }),
        "enforced_linear_domain" => Ok(CpConstraint::EnforcedLinearDomain {
            enforcement: native_cp_literals(object, "enforcement")?,
            terms: native_cp_terms(object, "terms")?,
            intervals: native_cp_intervals(object, "intervals")?,
        }),
        "map_domain" => Ok(CpConstraint::MapDomain {
            var: native_cp_usize(object, "var")?,
            bools: native_cp_indices(object, "bools")?,
            offset: native_cp_i64(object, "offset")?,
        }),
        "enforced_linear" => Ok(CpConstraint::EnforcedLinear {
            enforcement: native_cp_literals(object, "enforcement")?,
            terms: native_cp_terms(object, "terms")?,
            sense: native_cp_linear_sense(object)?,
            rhs: native_cp_i64(object, "rhs")?,
        }),
        "enforced_bool_or" => Ok(CpConstraint::EnforcedBoolOr {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "enforced_bool_and" => Ok(CpConstraint::EnforcedBoolAnd {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "enforced_bool_xor" => Ok(CpConstraint::EnforcedBoolXor {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "enforced_at_most_one" => Ok(CpConstraint::EnforcedAtMostOne {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "enforced_at_least_one" => Ok(CpConstraint::EnforcedAtLeastOne {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "enforced_exactly_one" => Ok(CpConstraint::EnforcedExactlyOne {
            enforcement: native_cp_literals(object, "enforcement")?,
            literals: native_cp_literals(object, "literals")?,
        }),
        "all_different" => Ok(CpConstraint::AllDifferent(native_cp_indices(
            object, "vars",
        )?)),
        "bool_or" => Ok(CpConstraint::BoolOr(native_cp_literals(
            object, "literals",
        )?)),
        "bool_and" => Ok(CpConstraint::BoolAnd(native_cp_literals(
            object, "literals",
        )?)),
        "bool_xor" => Ok(CpConstraint::BoolXor(native_cp_literals(
            object, "literals",
        )?)),
        "at_most_one" => Ok(CpConstraint::AtMostOne(native_cp_literals(
            object, "literals",
        )?)),
        "at_least_one" => Ok(CpConstraint::AtLeastOne(native_cp_literals(
            object, "literals",
        )?)),
        "exactly_one" => Ok(CpConstraint::ExactlyOne(native_cp_literals(
            object, "literals",
        )?)),
        "implication" => Ok(CpConstraint::Implication {
            antecedent: native_cp_literal(
                object
                    .get("antecedent")
                    .ok_or_else(|| "missing `antecedent`".to_string())?,
            )?,
            consequent: native_cp_literal(
                object
                    .get("consequent")
                    .ok_or_else(|| "missing `consequent`".to_string())?,
            )?,
        }),
        "allowed_assignments" => Ok(CpConstraint::AllowedAssignments {
            vars: native_cp_indices(object, "vars")?,
            tuples: native_cp_tuples(object, "tuples")?,
        }),
        "forbidden_assignments" => Ok(CpConstraint::ForbiddenAssignments {
            vars: native_cp_indices(object, "vars")?,
            tuples: native_cp_tuples(object, "tuples")?,
        }),
        "enforced_allowed_assignments" => Ok(CpConstraint::EnforcedAllowedAssignments {
            enforcement: native_cp_literals(object, "enforcement")?,
            vars: native_cp_indices(object, "vars")?,
            tuples: native_cp_tuples(object, "tuples")?,
        }),
        "enforced_forbidden_assignments" => Ok(CpConstraint::EnforcedForbiddenAssignments {
            enforcement: native_cp_literals(object, "enforcement")?,
            vars: native_cp_indices(object, "vars")?,
            tuples: native_cp_tuples(object, "tuples")?,
        }),
        "inverse" => Ok(CpConstraint::Inverse {
            direct: native_cp_indices(object, "direct")?,
            inverse: native_cp_indices(object, "inverse")?,
        }),
        "max_equality" => Ok(CpConstraint::MaxEquality {
            target: native_cp_usize(object, "target")?,
            vars: native_cp_indices(object, "vars")?,
        }),
        "min_equality" => Ok(CpConstraint::MinEquality {
            target: native_cp_usize(object, "target")?,
            vars: native_cp_indices(object, "vars")?,
        }),
        "abs_equality" => Ok(CpConstraint::AbsEquality {
            target: native_cp_usize(object, "target")?,
            var: native_cp_usize(object, "var")?,
        }),
        "multiplication_equality" => Ok(CpConstraint::MultiplicationEquality {
            target: native_cp_usize(object, "target")?,
            vars: native_cp_indices(object, "vars")?,
        }),
        "division_equality" => Ok(CpConstraint::DivisionEquality {
            target: native_cp_usize(object, "target")?,
            numerator: native_cp_usize(object, "numerator")?,
            denominator: native_cp_usize(object, "denominator")?,
        }),
        "modulo_equality" => Ok(CpConstraint::ModuloEquality {
            target: native_cp_usize(object, "target")?,
            var: native_cp_usize(object, "var")?,
            modulus: native_cp_usize(object, "modulus")?,
        }),
        "automaton" => Ok(CpConstraint::Automaton(CpAutomaton {
            vars: native_cp_indices(object, "vars")?,
            starting_state: native_cp_i64(object, "starting_state")?,
            final_states: native_cp_i64_values(object, "final_states")?,
            transitions: native_cp_array(object, "transitions")?
                .iter()
                .map(|value| {
                    let transition = native_cp_object(value, "automaton transition")?;
                    Ok(CpTransition {
                        tail: native_cp_i64(transition, "tail")?,
                        label: native_cp_i64(transition, "label")?,
                        head: native_cp_i64(transition, "head")?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        })),
        "circuit" => Ok(CpConstraint::Circuit(native_cp_circuit_arcs(object)?)),
        "multiple_circuit" => Ok(CpConstraint::MultipleCircuit(native_cp_circuit_arcs(
            object,
        )?)),
        "element" => Ok(CpConstraint::Element(CpElement {
            index: native_cp_usize(object, "index")?,
            values: native_cp_i64_values(object, "values")?,
            target: native_cp_usize(object, "target")?,
        })),
        "variable_element" => Ok(CpConstraint::VariableElement(CpVariableElement {
            index: native_cp_usize(object, "index")?,
            vars: native_cp_indices(object, "vars")?,
            target: native_cp_usize(object, "target")?,
        })),
        "alternative" => Ok(CpConstraint::Alternative(CpAlternative {
            start: native_cp_usize(object, "start")?,
            duration: native_cp_usize(object, "duration")?,
            end: native_cp_usize(object, "end")?,
            presence: native_cp_optional_literal(object, "presence")?,
            alternatives: native_cp_array(object, "alternatives")?
                .iter()
                .map(native_cp_variable_interval)
                .collect::<Result<Vec<_>, String>>()?,
            name: native_cp_name(object)?,
        })),
        "no_overlap" => Ok(CpConstraint::NoOverlap(
            native_cp_array(object, "intervals")?
                .iter()
                .map(native_cp_fixed_interval)
                .collect::<Result<Vec<_>, String>>()?,
        )),
        "no_overlap_variable" => Ok(CpConstraint::NoOverlapVariable(
            native_cp_array(object, "intervals")?
                .iter()
                .map(native_cp_variable_interval)
                .collect::<Result<Vec<_>, String>>()?,
        )),
        "no_overlap_2d" => Ok(CpConstraint::NoOverlap2D(
            native_cp_array(object, "rectangles")?
                .iter()
                .map(native_cp_rectangle)
                .collect::<Result<Vec<_>, String>>()?,
        )),
        "no_overlap_2d_variable" => Ok(CpConstraint::NoOverlap2DVariable(
            native_cp_array(object, "rectangles")?
                .iter()
                .map(native_cp_variable_rectangle)
                .collect::<Result<Vec<_>, String>>()?,
        )),
        "cumulative" => Ok(CpConstraint::Cumulative {
            intervals: native_cp_array(object, "intervals")?
                .iter()
                .map(native_cp_demand_interval)
                .collect::<Result<Vec<_>, String>>()?,
            capacity: native_cp_i64(object, "capacity")?,
        }),
        "cumulative_variable" => Ok(CpConstraint::CumulativeVariable {
            intervals: native_cp_array(object, "intervals")?
                .iter()
                .map(native_cp_variable_demand_interval)
                .collect::<Result<Vec<_>, String>>()?,
            capacity: native_cp_usize(object, "capacity")?,
        }),
        "reservoir" => Ok(CpConstraint::Reservoir {
            events: native_cp_array(object, "events")?
                .iter()
                .map(|value| {
                    let event = native_cp_object(value, "reservoir event")?;
                    Ok(CpReservoirEvent {
                        time: native_cp_usize(event, "time")?,
                        level_change: native_cp_i64(event, "level_change")?,
                        active: native_cp_optional_literal(event, "active")?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            min_level: native_cp_i64(object, "min_level")?,
            max_level: native_cp_i64(object, "max_level")?,
        }),
        other => Err(format!(
            "rust-native parser does not support constraint kind `{other}`"
        )),
    }
}

fn native_cp_circuit_arcs(
    object: &serde_json::Map<String, Value>,
) -> Result<Vec<CpCircuitArc>, String> {
    native_cp_array(object, "arcs")?
        .iter()
        .map(|value| {
            let arc = native_cp_object(value, "circuit arc")?;
            Ok(CpCircuitArc {
                tail: native_cp_i64(arc, "tail")?,
                head: native_cp_i64(arc, "head")?,
                literal: native_cp_literal(
                    arc.get("literal")
                        .ok_or_else(|| "missing `literal`".to_string())?,
                )?,
            })
        })
        .collect()
}

fn native_cp_model(
    value: &Value,
) -> Result<(CpModel, Vec<CpSolutionHint>, Vec<CpDecisionStrategy>), String> {
    let object = native_cp_object(value, "CP-SAT model")?;
    let variables = native_cp_array(object, "variables")?
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let variable = native_cp_object(value, "variable")?;
            let name = variable
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("x{idx}"));
            Ok(CpVariable {
                name,
                domain: native_cp_i64_values(variable, "domain")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let constraints = native_cp_optional_array(object, "constraints")?
        .into_iter()
        .flatten()
        .map(native_cp_constraint)
        .collect::<Result<Vec<_>, String>>()?;
    let objective = match object.get("objective") {
        Some(Value::Null) | None => None,
        Some(value) => {
            let objective = native_cp_object(value, "objective")?;
            Some(CpObjective {
                sense: native_cp_objective_sense(objective)?,
                terms: native_cp_terms(objective, "terms")?,
            })
        }
    };
    let solution_hint = native_cp_optional_array(object, "solution_hint")?
        .into_iter()
        .flatten()
        .map(|value| {
            let hint = native_cp_object(value, "solution hint")?;
            Ok(CpSolutionHint {
                var: native_cp_usize(hint, "var")?,
                value: native_cp_i64(hint, "value")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let decision_strategies = native_cp_optional_array(object, "decision_strategies")?
        .into_iter()
        .flatten()
        .map(native_cp_decision_strategy)
        .collect::<Result<Vec<_>, String>>()?;
    Ok((
        CpModel {
            variables,
            constraints,
            objective,
        },
        solution_hint,
        decision_strategies,
    ))
}

fn native_cp_decision_strategy(value: &Value) -> Result<CpDecisionStrategy, String> {
    let strategy = native_cp_object(value, "decision strategy")?;
    let variable_strategy = match strategy
        .get("variable_strategy")
        .and_then(Value::as_str)
        .unwrap_or("first")
    {
        "first" => CpVariableSelectionStrategy::First,
        "min_domain_size" => CpVariableSelectionStrategy::MinDomainSize,
        "max_domain_size" => CpVariableSelectionStrategy::MaxDomainSize,
        "lowest_min" => CpVariableSelectionStrategy::LowestMin,
        "highest_max" => CpVariableSelectionStrategy::HighestMax,
        other => return Err(format!("unsupported variable strategy `{other}`")),
    };
    let domain_strategy = match strategy
        .get("domain_strategy")
        .and_then(Value::as_str)
        .unwrap_or("min_value")
    {
        "min_value" => CpDomainValueStrategy::MinValue,
        "max_value" => CpDomainValueStrategy::MaxValue,
        "lower_half" => CpDomainValueStrategy::LowerHalf,
        "upper_half" => CpDomainValueStrategy::UpperHalf,
        "median_value" => CpDomainValueStrategy::MedianValue,
        other => return Err(format!("unsupported domain strategy `{other}`")),
    };
    Ok(CpDecisionStrategy {
        vars: native_cp_indices(strategy, "vars")?,
        variable_strategy,
        domain_strategy,
    })
}

fn native_cp_assumptions(value: &Value) -> Result<Vec<BoolLiteral>, String> {
    let object = native_cp_object(value, "CP-SAT model")?;
    native_cp_optional_array(object, "assumptions")?
        .into_iter()
        .flatten()
        .map(native_cp_literal)
        .collect()
}

fn cp_solve_options(
    solution_hint: Vec<CpSolutionHint>,
    decision_strategies: Vec<CpDecisionStrategy>,
) -> CpSolveOptions {
    CpSolveOptions {
        solution_hint,
        decision_strategies,
        ..Default::default()
    }
}

fn solve_cp_sat_json_with_native_rust(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
    started: Instant,
) -> Result<ExternalCpSatReferenceRun, String> {
    let (cp_model, solution_hint, decision_strategies) = native_cp_model(model)?;
    if options.assumption_core {
        let assumptions = native_cp_assumptions(model)?;
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            find_cp_assumption_unsat_core(
                &cp_model,
                &assumptions,
                CpAssumptionCoreOptions {
                    solve_options: cp_solve_options(solution_hint, decision_strategies),
                },
            )
        }))
        .map_err(|_| "native Rust CP assumption-core solve panicked".to_string())?;
        let status = native_cp_status(run.status);
        let assumptions_json = run
            .assumptions
            .iter()
            .map(|literal| json!({"var": literal.var, "positive": literal.positive}))
            .collect::<Vec<_>>();
        let message = run
            .message
            .clone()
            .unwrap_or_else(|| "native Rust assumption core".to_string());
        return Ok(ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: "rust:cp-native-assumption-core".to_string(),
            status,
            assignment: Vec::new(),
            objective: None,
            nodes: Some(run.checks as u64),
            raw: json!({
                "status": status.as_str(),
                "assumptions": assumptions_json,
                "minimal": run.minimal,
                "checks": run.checks,
                "solver": "rust:cp-native-assumption-core",
                "message": message,
            }),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message,
        });
    }

    if let Some(limit) = options.enumerate_solutions {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enumerate_cp_solutions(
                &cp_model,
                CpEnumerateOptions {
                    max_solutions: limit.max(1),
                    ..Default::default()
                },
            )
        }))
        .map_err(|_| "native Rust CP solution enumeration panicked".to_string())?;
        let status = native_cp_status(run.status);
        let first = run.solutions.first();
        let assignment = first
            .map(|solution| solution.assignment.clone())
            .unwrap_or_default();
        let objective = first
            .and_then(|solution| solution.objective)
            .map(|value| value as f64);
        let solutions_json = run
            .solutions
            .iter()
            .map(|solution| {
                json!({
                    "assignment": solution.assignment,
                    "objective": solution.objective,
                })
            })
            .collect::<Vec<_>>();
        let message = run
            .message
            .clone()
            .unwrap_or_else(|| "native Rust solution enumeration".to_string());
        return Ok(ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: "rust:cp-native-solution-enumeration".to_string(),
            status,
            assignment,
            objective,
            nodes: Some(run.nodes as u64),
            raw: json!({
                "status": status.as_str(),
                "assignment": first.map(|solution| solution.assignment.clone()).unwrap_or_default(),
                "objective": first.and_then(|solution| solution.objective),
                "solutions": solutions_json,
                "exhausted": run.exhausted,
                "nodes": run.nodes,
                "solver": "rust:cp-native-solution-enumeration",
                "message": message,
            }),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message,
        });
    }

    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve_cp_model(
            &cp_model,
            cp_solve_options(solution_hint, decision_strategies),
        )
    }))
    .map_err(|_| "native Rust CP solve panicked".to_string())?;
    let status = native_cp_status(run.status);
    let objective = run.objective.map(|value| value as f64);
    let message = run
        .message
        .clone()
        .unwrap_or_else(|| "native Rust exact finite-domain solve".to_string());
    Ok(ExternalCpSatReferenceRun {
        solver: options.solver,
        backend: "rust:cp-native-enumeration".to_string(),
        status,
        assignment: run.assignment.clone(),
        objective,
        nodes: Some(run.nodes as u64),
        raw: json!({
            "status": status.as_str(),
            "assignment": run.assignment,
            "objective": run.objective,
            "nodes": run.nodes,
            "solver": "rust:cp-native-enumeration",
            "message": message,
        }),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message,
    })
}

fn cp_sat_array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array `{key}`"))
}

fn cp_sat_i64(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing or non-integer `{key}`"))
}

fn cp_sat_usize(value: &Value, key: &str, len: usize) -> Result<usize, String> {
    let raw = cp_sat_i64(value, key)?;
    if raw < 0 || raw as usize >= len {
        return Err(format!("`{key}` index {raw} is outside 0..{len}"));
    }
    Ok(raw as usize)
}

fn cp_sat_domains(model: &Value) -> Result<Vec<Vec<i64>>, String> {
    cp_sat_array(model, "variables")?
        .iter()
        .enumerate()
        .map(|(idx, variable)| {
            let domain = cp_sat_array(variable, "domain")?
                .iter()
                .map(|value| {
                    value
                        .as_i64()
                        .ok_or_else(|| format!("variable {idx} has a non-integer domain value"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if domain.is_empty() {
                return Err(format!("variable {idx} has an empty domain"));
            }
            Ok(domain)
        })
        .collect()
}

fn cp_sat_literal_truth(assignment: &[i64], lit: &Value) -> Result<bool, String> {
    let var = cp_sat_usize(lit, "var", assignment.len())?;
    let positive = lit.get("positive").and_then(Value::as_bool).unwrap_or(true);
    Ok(if positive {
        assignment[var] == 1
    } else {
        assignment[var] == 0
    })
}

fn cp_sat_enforcement_active(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let Some(enforcement) = constraint.get("enforcement") else {
        return Ok(true);
    };
    let literals = enforcement
        .as_array()
        .ok_or_else(|| "`enforcement` must be an array".to_string())?;
    for lit in literals {
        if !cp_sat_literal_truth(assignment, lit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn cp_sat_linear_value(assignment: &[i64], terms: &[Value]) -> Result<i64, String> {
    let mut total = 0i64;
    for term in terms {
        let var = cp_sat_usize(term, "var", assignment.len())?;
        let coeff = cp_sat_i64(term, "coeff")?;
        total = total
            .checked_add(
                coeff
                    .checked_mul(assignment[var])
                    .ok_or_else(|| "linear term overflow".to_string())?,
            )
            .ok_or_else(|| "linear expression overflow".to_string())?;
    }
    Ok(total)
}

fn cp_sat_linear_constraint_ok(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let value = cp_sat_linear_value(assignment, cp_sat_array(constraint, "terms")?)?;
    let rhs = cp_sat_i64(constraint, "rhs")?;
    match constraint
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("eq")
    {
        "le" => Ok(value <= rhs),
        "ge" => Ok(value >= rhs),
        "eq" => Ok(value == rhs),
        sense => Err(format!("unsupported linear sense `{sense}`")),
    }
}

fn cp_sat_linear_domain_constraint_ok(
    assignment: &[i64],
    constraint: &Value,
) -> Result<bool, String> {
    let value = cp_sat_linear_value(assignment, cp_sat_array(constraint, "terms")?)?;
    for interval in cp_sat_array(constraint, "intervals")? {
        let lb = cp_sat_i64(interval, "lb")?;
        let ub = cp_sat_i64(interval, "ub")?;
        if lb <= value && value <= ub {
            return Ok(true);
        }
    }
    Ok(false)
}

fn cp_sat_bool_clause_ok(
    assignment: &[i64],
    constraint: &Value,
    mode: &str,
) -> Result<bool, String> {
    let literals = cp_sat_array(constraint, "literals")?;
    let true_count = literals.iter().try_fold(0usize, |count, lit| {
        Ok::<_, String>(count + usize::from(cp_sat_literal_truth(assignment, lit)?))
    })?;
    match mode {
        "or" | "at_least_one" => Ok(true_count >= 1),
        "and" => Ok(true_count == literals.len()),
        "xor" => Ok(true_count % 2 == 1),
        "at_most_one" => Ok(true_count <= 1),
        "exactly_one" => Ok(true_count == 1),
        _ => Err(format!("unsupported Boolean mode `{mode}`")),
    }
}

fn cp_sat_tuple_constraint_ok(
    assignment: &[i64],
    constraint: &Value,
    allowed: bool,
) -> Result<bool, String> {
    let vars = cp_sat_array(constraint, "vars")?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| "tuple variable index must be an integer".to_string())
                .and_then(|idx| {
                    if idx < 0 || idx as usize >= assignment.len() {
                        Err(format!("tuple variable index {idx} out of range"))
                    } else {
                        Ok(idx as usize)
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected = vars.iter().map(|var| assignment[*var]).collect::<Vec<_>>();
    let mut listed = false;
    for tuple in cp_sat_array(constraint, "tuples")? {
        let tuple_values = tuple
            .as_array()
            .ok_or_else(|| "tuple entry must be an array".to_string())?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| "tuple value must be an integer".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tuple_values == selected {
            listed = true;
            break;
        }
    }
    Ok(listed == allowed)
}

fn cp_sat_constraint_ok(assignment: &[i64], constraint: &Value) -> Result<bool, String> {
    let kind = constraint
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "constraint missing string `kind`".to_string())?;
    let active = match kind {
        "enforced_linear"
        | "enforced_linear_domain"
        | "enforced_bool_or"
        | "enforced_at_least_one"
        | "enforced_bool_and"
        | "enforced_bool_xor"
        | "enforced_at_most_one"
        | "enforced_exactly_one"
        | "enforced_allowed_assignments"
        | "enforced_forbidden_assignments" => cp_sat_enforcement_active(assignment, constraint)?,
        _ => true,
    };
    if !active {
        return Ok(true);
    }
    match kind {
        "linear" | "enforced_linear" => cp_sat_linear_constraint_ok(assignment, constraint),
        "linear_domain" | "enforced_linear_domain" => {
            cp_sat_linear_domain_constraint_ok(assignment, constraint)
        }
        "all_different" => {
            let mut seen = std::collections::BTreeSet::new();
            for var in cp_sat_array(constraint, "vars")? {
                let idx = var
                    .as_i64()
                    .ok_or_else(|| "all_different variable must be an integer".to_string())?;
                if idx < 0 || idx as usize >= assignment.len() {
                    return Err(format!("all_different variable index {idx} out of range"));
                }
                if !seen.insert(assignment[idx as usize]) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        "bool_or" | "enforced_bool_or" => cp_sat_bool_clause_ok(assignment, constraint, "or"),
        "bool_and" | "enforced_bool_and" => cp_sat_bool_clause_ok(assignment, constraint, "and"),
        "bool_xor" | "enforced_bool_xor" => cp_sat_bool_clause_ok(assignment, constraint, "xor"),
        "at_most_one" | "enforced_at_most_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "at_most_one")
        }
        "at_least_one" | "enforced_at_least_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "at_least_one")
        }
        "exactly_one" | "enforced_exactly_one" => {
            cp_sat_bool_clause_ok(assignment, constraint, "exactly_one")
        }
        "implication" => {
            let antecedent = cp_sat_literal_truth(
                assignment,
                constraint
                    .get("antecedent")
                    .ok_or_else(|| "implication missing antecedent".to_string())?,
            )?;
            let consequent = cp_sat_literal_truth(
                assignment,
                constraint
                    .get("consequent")
                    .ok_or_else(|| "implication missing consequent".to_string())?,
            )?;
            Ok(!antecedent || consequent)
        }
        "allowed_assignments" | "enforced_allowed_assignments" => {
            cp_sat_tuple_constraint_ok(assignment, constraint, true)
        }
        "forbidden_assignments" | "enforced_forbidden_assignments" => {
            cp_sat_tuple_constraint_ok(assignment, constraint, false)
        }
        other => Err(format!(
            "rust-enumeration does not support constraint kind `{other}`"
        )),
    }
}

fn cp_sat_assignment_feasible(model: &Value, assignment: &[i64]) -> Result<bool, String> {
    for constraint in model
        .get("constraints")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if !cp_sat_constraint_ok(assignment, constraint)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn cp_sat_objective_value(model: &Value, assignment: &[i64]) -> Result<Option<i64>, String> {
    let Some(objective) = model.get("objective") else {
        return Ok(None);
    };
    Ok(Some(cp_sat_linear_value(
        assignment,
        cp_sat_array(objective, "terms")?,
    )?))
}

fn cp_sat_better_objective(model: &Value, candidate: i64, incumbent: i64) -> bool {
    let minimize = model
        .get("objective")
        .and_then(|objective| objective.get("sense"))
        .and_then(Value::as_str)
        .unwrap_or("min")
        != "max";
    if minimize {
        candidate < incumbent
    } else {
        candidate > incumbent
    }
}

fn solve_cp_sat_json_with_rust_enumeration(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
    started: Instant,
) -> ExternalCpSatReferenceRun {
    if let Ok(run) = solve_cp_sat_json_with_native_rust(model, options, started) {
        return run;
    }

    if options.assumption_core {
        return cp_sat_error_run(
            options.solver,
            ExternalCpSatReferenceStatus::Unsupported,
            "rust-enumeration does not compute assumption cores yet",
            started,
        );
    }

    let domains = match cp_sat_domains(model) {
        Ok(domains) => domains,
        Err(message) => {
            return cp_sat_error_run(
                options.solver,
                ExternalCpSatReferenceStatus::Invalid,
                message,
                started,
            )
        }
    };
    let mut assignment = vec![0; domains.len()];
    let mut nodes = 0u64;
    let mut best_assignment = None::<Vec<i64>>;
    let mut best_objective = None::<i64>;
    let mut solutions = Vec::<Value>::new();
    let solution_limit = options.enumerate_solutions.unwrap_or(usize::MAX).max(1);

    fn dfs(
        model: &Value,
        domains: &[Vec<i64>],
        assignment: &mut [i64],
        var_idx: usize,
        nodes: &mut u64,
        best_assignment: &mut Option<Vec<i64>>,
        best_objective: &mut Option<i64>,
        solutions: &mut Vec<Value>,
        solution_limit: usize,
    ) -> Result<(), String> {
        if var_idx == domains.len() {
            *nodes = nodes.saturating_add(1);
            if !cp_sat_assignment_feasible(model, assignment)? {
                return Ok(());
            }
            let objective = cp_sat_objective_value(model, assignment)?;
            let is_better = match (*best_objective, objective) {
                (Some(incumbent), Some(candidate)) => {
                    cp_sat_better_objective(model, candidate, incumbent)
                }
                (None, Some(_)) | (None, None) => best_assignment.is_none(),
                (Some(_), None) => false,
            };
            if is_better {
                *best_assignment = Some(assignment.to_vec());
                *best_objective = objective;
            }
            if solutions.len() < solution_limit {
                solutions.push(json!({
                    "assignment": assignment,
                    "objective": objective.map(|value| value as f64),
                }));
            }
            return Ok(());
        }
        for value in &domains[var_idx] {
            assignment[var_idx] = *value;
            dfs(
                model,
                domains,
                assignment,
                var_idx + 1,
                nodes,
                best_assignment,
                best_objective,
                solutions,
                solution_limit,
            )?;
        }
        Ok(())
    }

    if let Err(message) = dfs(
        model,
        &domains,
        &mut assignment,
        0,
        &mut nodes,
        &mut best_assignment,
        &mut best_objective,
        &mut solutions,
        solution_limit,
    ) {
        let status = if message.contains("does not support") {
            ExternalCpSatReferenceStatus::Unsupported
        } else {
            ExternalCpSatReferenceStatus::Invalid
        };
        return cp_sat_error_run(options.solver, status, message, started);
    }

    let Some(best) = best_assignment else {
        let raw = json!({
            "status": "infeasible",
            "solver": "rust:cp-enumeration",
            "assignment": [],
            "objective": null,
            "nodes": nodes,
        });
        return ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: "rust:cp-enumeration".to_string(),
            status: ExternalCpSatReferenceStatus::Infeasible,
            assignment: Vec::new(),
            objective: None,
            nodes: Some(nodes),
            raw,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            message: String::new(),
        };
    };

    if model.get("objective").is_some() {
        let reverse = model
            .get("objective")
            .and_then(|objective| objective.get("sense"))
            .and_then(Value::as_str)
            .unwrap_or("min")
            == "max";
        solutions.sort_by(|a, b| {
            let lhs = a.get("objective").and_then(Value::as_f64).unwrap_or(0.0);
            let rhs = b.get("objective").and_then(Value::as_f64).unwrap_or(0.0);
            if reverse {
                rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        solutions.truncate(solution_limit);
    }

    let objective = best_objective.map(|value| value as f64);
    let status = if model.get("objective").is_some() {
        ExternalCpSatReferenceStatus::Optimal
    } else {
        ExternalCpSatReferenceStatus::Feasible
    };
    let raw = json!({
        "status": status.as_str(),
        "solver": "rust:cp-enumeration",
        "assignment": best,
        "objective": objective,
        "nodes": nodes,
        "solutions": if options.enumerate_solutions.is_some() { Value::Array(solutions) } else { Value::Null },
        "message": "native Rust exact enumeration fallback",
    });
    ExternalCpSatReferenceRun {
        solver: options.solver,
        backend: "rust:cp-enumeration".to_string(),
        status,
        assignment: raw
            .get("assignment")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default(),
        objective,
        nodes: Some(nodes),
        raw,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        message: "native Rust exact enumeration fallback".to_string(),
    }
}

pub fn solve_cp_sat_json_with_external_reference(
    model: &Value,
    options: &ExternalCpSatReferenceOptions,
) -> ExternalCpSatReferenceRun {
    let started = Instant::now();
    if options.solver == ExternalCpSatReferenceSolver::RustEnumeration {
        return solve_cp_sat_json_with_rust_enumeration(model, options, started);
    }

    let Some(solver_arg) = options.solver.direct_cp_sat_json_solver_arg() else {
        return ExternalCpSatReferenceRun {
            solver: options.solver,
            backend: options.solver.as_arg().to_string(),
            status: ExternalCpSatReferenceStatus::Unsupported,
            assignment: Vec::new(),
            objective: None,
            nodes: None,
            raw: json!({
                "status": "unsupported",
                "solver": options.solver.as_arg(),
                "message": "solver uses the ecosystem CP-assignment contract, not CP-SAT JSON",
            }),
            elapsed_ms: 0.0,
            message: "solver uses the ecosystem CP-assignment contract, not CP-SAT JSON"
                .to_string(),
        };
    };

    let script = external_cp_sat_reference_script();
    let mut command = Command::new(python_command());
    if let Some(working_dir) = script_working_dir(&script) {
        command.current_dir(working_dir);
    }
    command.arg(script).arg("--solver").arg(solver_arg);
    if let Some(limit) = options.enumerate_solutions {
        command
            .arg("--enumerate-solutions")
            .arg(limit.max(1).to_string());
    }
    if options.assumption_core {
        command.arg("--assumption-core");
    }

    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Unavailable,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "unavailable", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(e) = stdin.write_all(model.to_string().as_bytes()) {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "failed", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            };
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({"status": "failed", "message": e.to_string()}),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                message: e.to_string(),
            }
        }
    };

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let raw = match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(raw) => raw,
        Err(e) => {
            return ExternalCpSatReferenceRun {
                solver: options.solver,
                backend: solver_arg.to_string(),
                status: ExternalCpSatReferenceStatus::Failed,
                assignment: Vec::new(),
                objective: None,
                nodes: None,
                raw: json!({
                    "status": "failed",
                    "stdout": stdout.trim(),
                    "stderr": stderr,
                    "message": e.to_string(),
                }),
                elapsed_ms,
                message: e.to_string(),
            }
        }
    };

    let parsed = serde_json::from_value::<CpSatScriptOutput>(raw.clone()).ok();
    let status = parsed
        .as_ref()
        .map(|parsed| ExternalCpSatReferenceStatus::from_label(parsed.status.as_str()))
        .unwrap_or(ExternalCpSatReferenceStatus::Unknown);
    let message = parsed
        .as_ref()
        .map(|parsed| parsed.message.clone())
        .filter(|message| !message.is_empty())
        .unwrap_or(stderr);

    ExternalCpSatReferenceRun {
        solver: options.solver,
        backend: parsed
            .as_ref()
            .map(|parsed| parsed.solver.clone())
            .filter(|solver| !solver.is_empty())
            .unwrap_or_else(|| solver_arg.to_string()),
        status,
        assignment: parsed
            .as_ref()
            .map(|parsed| parsed.assignment.clone())
            .unwrap_or_default(),
        objective: parsed.as_ref().and_then(|parsed| parsed.objective),
        nodes: parsed.as_ref().and_then(|parsed| parsed.nodes),
        raw,
        elapsed_ms,
        message,
    }
}

pub fn solve_cp_assignment_with_external_reference(
    payload: &Value,
    solver: ExternalCpSatReferenceSolver,
) -> ExternalCpSatReferenceRun {
    let Some(tool) = solver.ecosystem_tool() else {
        return ExternalCpSatReferenceRun {
            solver,
            backend: solver.as_arg().to_string(),
            status: ExternalCpSatReferenceStatus::Unsupported,
            assignment: Vec::new(),
            objective: None,
            nodes: None,
            raw: json!({
                "status": "unsupported",
                "solver": solver.as_arg(),
                "message": "solver uses the direct CP-SAT JSON bridge, not ecosystem CP assignment",
            }),
            elapsed_ms: 0.0,
            message: "solver uses the direct CP-SAT JSON bridge, not ecosystem CP assignment"
                .to_string(),
        };
    };

    let run = run_external_optimization_ecosystem_reference(payload, tool);
    let raw = run.output.clone().unwrap_or_else(|| {
        json!({
            "status": run.status.as_str(),
            "tool": tool.as_str(),
            "message": run.message,
        })
    });
    let status_label = raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_else(|| run.status.as_str());
    let status = match run.status {
        ExternalOptimizationAdapterStatus::Ok => {
            ExternalCpSatReferenceStatus::from_label(status_label)
        }
        ExternalOptimizationAdapterStatus::Unavailable => ExternalCpSatReferenceStatus::Unavailable,
        ExternalOptimizationAdapterStatus::Failed => ExternalCpSatReferenceStatus::Failed,
        ExternalOptimizationAdapterStatus::InvalidOutput => ExternalCpSatReferenceStatus::Invalid,
    };
    let assignment = raw
        .get("x")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(|value| value.round() as i64)
                .collect()
        })
        .unwrap_or_default();
    let objective = raw.get("objective").and_then(Value::as_f64);

    ExternalCpSatReferenceRun {
        solver,
        backend: raw
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or(tool.as_str())
            .to_string(),
        status,
        assignment,
        objective,
        nodes: None,
        raw,
        elapsed_ms: run.elapsed_ms,
        message: run.message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cp_sat_model() -> Value {
        json!({
            "variables": [
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]}
            ],
            "constraints": [
                {
                    "kind": "linear",
                    "terms": [
                        {"var": 0, "coeff": 1},
                        {"var": 1, "coeff": 1}
                    ],
                    "sense": "eq",
                    "rhs": 1
                }
            ],
            "objective": {
                "sense": "min",
                "terms": [
                    {"var": 0, "coeff": 1},
                    {"var": 1, "coeff": 2}
                ]
            }
        })
    }

    #[test]
    fn cp_sat_reference_manifest_splits_direct_and_ecosystem_contracts() {
        let specs = external_cp_sat_reference_solver_specs();
        let direct = specs
            .iter()
            .filter(|spec| spec.supports_cp_sat_json)
            .count();
        let ecosystem = specs
            .iter()
            .filter(|spec| spec.supports_ecosystem_cp_assignment)
            .count();

        assert_eq!(specs.len(), 19);
        assert_eq!(direct, 4);
        assert_eq!(ecosystem, 15);
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalCpSatReferenceSolver::RustEnumeration
                && spec.family == ExternalCpSatReferenceFamily::Fallback
                && spec.supports_cp_sat_json
        }));
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalCpSatReferenceSolver::ChocoSolver
                && spec.family == ExternalCpSatReferenceFamily::EcosystemReference
        }));
        assert!(ExternalCpSatReferenceSolver::RustEnumeration.supports_cp_sat_json());
        assert!(ExternalCpSatReferenceSolver::PythonEnumeration.supports_cp_sat_json());
        assert!(!ExternalCpSatReferenceSolver::ChocoSolver.supports_cp_sat_json());
    }

    #[test]
    fn cp_sat_model_serializer_emits_reference_contract() {
        let model = CpModel {
            variables: vec![
                CpVariable {
                    name: "pick".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "load".to_string(),
                    domain: vec![0, 4],
                },
            ],
            constraints: vec![
                CpConstraint::EnforcedLinear {
                    enforcement: vec![BoolLiteral {
                        var: 0,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 1, coeff: 1 }],
                    sense: LinearSense::Ge,
                    rhs: 2,
                },
                CpConstraint::NoOverlap(vec![CpInterval {
                    start: 0,
                    duration: 2,
                    presence: Some(BoolLiteral {
                        var: 0,
                        positive: true,
                    }),
                    name: Some("optional_job".to_string()),
                }]),
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![LinearTerm { var: 1, coeff: 1 }],
            }),
        };

        let payload = cp_sat_model_to_reference_json(&model);

        assert_eq!(payload["variables"][0]["name"], "pick");
        assert_eq!(payload["constraints"][0]["kind"], "enforced_linear");
        assert_eq!(
            payload["constraints"][1]["intervals"][0]["presence"],
            json!({"var": 0, "positive": true})
        );
        assert_eq!(payload["objective"]["sense"], "min");

        let (parsed, hints, strategies) =
            native_cp_model(&payload).expect("parse serialized model");
        assert_eq!(parsed, model);
        assert!(hints.is_empty());
        assert!(strategies.is_empty());
        assert_eq!(
            cp_sat_model_to_reference_json_string(&model),
            payload.to_string()
        );
    }

    #[test]
    fn cp_sat_model_serializer_feeds_rust_reference_solver() {
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
            constraints: vec![CpConstraint::ExactlyOne(vec![
                BoolLiteral {
                    var: 0,
                    positive: true,
                },
                BoolLiteral {
                    var: 1,
                    positive: true,
                },
            ])],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Max,
                terms: vec![
                    LinearTerm { var: 0, coeff: 2 },
                    LinearTerm { var: 1, coeff: 1 },
                ],
            }),
        };
        let payload = cp_sat_model_to_reference_json(&model);
        let run = solve_cp_sat_json_with_external_reference(
            &payload,
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0]);
        assert_eq!(run.objective, Some(2.0));
        assert_eq!(run.backend, "rust:cp-native-enumeration");
    }

    #[test]
    fn cp_sat_rust_enumeration_solves_same_input_json() {
        let run = solve_cp_sat_json_with_external_reference(
            &tiny_cp_sat_model(),
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0]);
        assert_eq!(run.objective, Some(1.0));
        assert_eq!(run.backend, "rust:cp-native-enumeration");
    }

    #[test]
    fn cp_sat_rust_enumeration_handles_bool_and_all_different() {
        let model = json!({
            "variables": [
                {"name": "a", "domain": [0, 1]},
                {"name": "b", "domain": [0, 1]},
                {"name": "c", "domain": [0, 1]}
            ],
            "constraints": [
                {
                    "kind": "exactly_one",
                    "literals": [
                        {"var": 0, "positive": true},
                        {"var": 1, "positive": true}
                    ]
                },
                {"kind": "implication", "antecedent": {"var": 0}, "consequent": {"var": 2}},
                {"kind": "linear_domain", "terms": [{"var": 2, "coeff": 1}], "intervals": [{"lb": 1, "ub": 1}]}
            ],
            "objective": {
                "sense": "max",
                "terms": [
                    {"var": 0, "coeff": 2},
                    {"var": 1, "coeff": 1}
                ]
            }
        });
        let run = solve_cp_sat_json_with_external_reference(
            &model,
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.assignment, vec![1, 0, 1]);
        assert_eq!(run.objective, Some(2.0));
    }

    #[test]
    fn cp_sat_rust_enumeration_uses_native_global_parser() {
        let model = json!({
            "variables": [
                {"name": "choice", "domain": [0, 1]},
                {"name": "expensive", "domain": [4]},
                {"name": "cheap", "domain": [1, 3]},
                {"name": "selected", "domain": [1, 2, 3, 4]}
            ],
            "constraints": [
                {
                    "kind": "variable_element",
                    "index": 0,
                    "vars": [1, 2],
                    "target": 3
                }
            ],
            "objective": {
                "sense": "min",
                "terms": [{"var": 3, "coeff": 1}]
            }
        });
        let run = solve_cp_sat_json_with_external_reference(
            &model,
            &ExternalCpSatReferenceOptions {
                solver: ExternalCpSatReferenceSolver::RustEnumeration,
                ..Default::default()
            },
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.backend, "rust:cp-native-enumeration");
        assert_eq!(run.assignment, vec![1, 4, 1, 1]);
        assert_eq!(run.objective, Some(1.0));
    }

    #[test]
    fn cp_ecosystem_assignment_bridge_runs_choco_reference_contract() {
        let payload = json!({
            "kind": "ecosystem-cp-assignment",
            "costs": [[9, 2, 7], [6, 4, 3], [5, 8, 1]],
            "all_different": true
        });
        let run = solve_cp_assignment_with_external_reference(
            &payload,
            ExternalCpSatReferenceSolver::ChocoSolver,
        );

        assert_eq!(run.status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(run.objective, Some(9.0));
        assert_eq!(run.assignment, vec![1, 0, 2]);
        assert_eq!(run.backend, "builtin:constraint-programming");
    }
}
