//! Tests for the CP-SAT-style finite-domain solver and reference bridge.

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::{Command, Stdio};

    use serde::Deserialize;

    use crate::des::general::cp_sat::{
        solve_cp_model, BoolLiteral, CpAutomaton, CpCircuitArc, CpConstraint, CpDemandInterval,
        CpDomainInterval, CpElement, CpInterval, CpModel, CpObjective, CpRectangle,
        CpReservoirEvent, CpSolveOptions, CpStatus, CpTransition, CpVariable, LinearSense,
        LinearTerm, ObjectiveSense,
    };

    #[derive(Debug, Deserialize)]
    struct CpReference {
        status: String,
        assignment: Vec<i64>,
        objective: Option<i64>,
        solver: String,
    }

    fn sample_model() -> CpModel {
        CpModel {
            variables: vec![
                CpVariable {
                    name: "slot_a".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "slot_b".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "slot_c".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "use_bonus".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "task_a_start".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "task_b_start".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
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
                CpVariable {
                    name: "route_index".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_cost".to_string(),
                    domain: vec![3, 8],
                },
                CpVariable {
                    name: "handoff_mode".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "handler".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "expedite".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "service_level".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
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
                CpVariable {
                    name: "deviation".to_string(),
                    domain: vec![-3, -1, 2],
                },
                CpVariable {
                    name: "absolute_deviation".to_string(),
                    domain: vec![0, 1, 2, 3],
                },
                CpVariable {
                    name: "mandatory_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mandatory_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "xor_a".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "xor_b".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "score_product".to_string(),
                    domain: vec![6, 10, 12, 20],
                },
                CpVariable {
                    name: "pack_a_x".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "pack_a_y".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "pack_b_x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "pack_b_y".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "automaton_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "automaton_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "automaton_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_0_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_1_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "circuit_2_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "arith_value".to_string(),
                    domain: vec![5, 6, 7],
                },
                CpVariable {
                    name: "arith_divisor".to_string(),
                    domain: vec![2],
                },
                CpVariable {
                    name: "arith_quotient".to_string(),
                    domain: vec![2, 3],
                },
                CpVariable {
                    name: "arith_remainder".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "reservoir_fill_time".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "reservoir_drain_time".to_string(),
                    domain: vec![0],
                },
                CpVariable {
                    name: "reservoir_overfill_active".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "linear_domain_x".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "linear_domain_y".to_string(),
                    domain: vec![0, 1, 2],
                },
                CpVariable {
                    name: "mapped_mode".to_string(),
                    domain: vec![5],
                },
                CpVariable {
                    name: "mapped_is_five".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mapped_is_six".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "mapped_is_seven".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_0_1".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_1_0".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_0_2".to_string(),
                    domain: vec![0, 1],
                },
                CpVariable {
                    name: "route_2_0".to_string(),
                    domain: vec![0, 1],
                },
            ],
            constraints: vec![
                CpConstraint::AllDifferent(vec![0, 1, 2]),
                CpConstraint::Linear {
                    terms: vec![
                        LinearTerm { var: 0, coeff: 1 },
                        LinearTerm { var: 1, coeff: 1 },
                    ],
                    sense: LinearSense::Ge,
                    rhs: 1,
                },
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 3,
                    positive: true,
                }]),
                CpConstraint::NoOverlap(vec![
                    CpInterval {
                        start: 4,
                        duration: 3,
                        presence: None,
                        name: Some("task_a".to_string()),
                    },
                    CpInterval {
                        start: 5,
                        duration: 2,
                        presence: None,
                        name: Some("task_b".to_string()),
                    },
                ]),
                CpConstraint::Cumulative {
                    intervals: vec![
                        CpDemandInterval {
                            start: 6,
                            duration: 3,
                            demand: 2,
                            presence: None,
                            name: Some("machine_a".to_string()),
                        },
                        CpDemandInterval {
                            start: 7,
                            duration: 2,
                            demand: 2,
                            presence: None,
                            name: Some("machine_b".to_string()),
                        },
                        CpDemandInterval {
                            start: 8,
                            duration: 2,
                            demand: 1,
                            presence: None,
                            name: Some("machine_c".to_string()),
                        },
                    ],
                    capacity: 3,
                },
                CpConstraint::Element(CpElement {
                    index: 9,
                    values: vec![3, 8],
                    target: 10,
                }),
                CpConstraint::AllowedAssignments {
                    vars: vec![11, 12],
                    tuples: vec![vec![0, 1], vec![1, 0]],
                },
                CpConstraint::ForbiddenAssignments {
                    vars: vec![15, 18],
                    tuples: vec![vec![1, 1]],
                },
                CpConstraint::BoolOr(vec![BoolLiteral {
                    var: 13,
                    positive: true,
                }]),
                CpConstraint::EnforcedLinear {
                    enforcement: vec![BoolLiteral {
                        var: 13,
                        positive: true,
                    }],
                    terms: vec![LinearTerm { var: 14, coeff: 1 }],
                    sense: LinearSense::Ge,
                    rhs: 2,
                },
                CpConstraint::ExactlyOne(vec![
                    BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 16,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 17,
                        positive: true,
                    },
                ]),
                CpConstraint::AtMostOne(vec![
                    BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 17,
                        positive: true,
                    },
                ]),
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 15,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 18,
                        positive: true,
                    },
                },
                CpConstraint::Implication {
                    antecedent: BoolLiteral {
                        var: 18,
                        positive: true,
                    },
                    consequent: BoolLiteral {
                        var: 19,
                        positive: true,
                    },
                },
                CpConstraint::Inverse {
                    direct: vec![20, 21],
                    inverse: vec![22, 23],
                },
                CpConstraint::MaxEquality {
                    target: 26,
                    vars: vec![24, 25],
                },
                CpConstraint::MinEquality {
                    target: 27,
                    vars: vec![24, 25],
                },
                CpConstraint::AbsEquality {
                    target: 29,
                    var: 28,
                },
                CpConstraint::BoolAnd(vec![
                    BoolLiteral {
                        var: 30,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 31,
                        positive: true,
                    },
                ]),
                CpConstraint::BoolXor(vec![
                    BoolLiteral {
                        var: 32,
                        positive: true,
                    },
                    BoolLiteral {
                        var: 33,
                        positive: true,
                    },
                ]),
                CpConstraint::MultiplicationEquality {
                    target: 34,
                    vars: vec![24, 25],
                },
                CpConstraint::NoOverlap2D(vec![
                    CpRectangle {
                        x_start: 35,
                        y_start: 36,
                        width: 2,
                        height: 2,
                        presence: None,
                        name: Some("pack_a".to_string()),
                    },
                    CpRectangle {
                        x_start: 37,
                        y_start: 38,
                        width: 2,
                        height: 2,
                        presence: None,
                        name: Some("pack_b".to_string()),
                    },
                ]),
                CpConstraint::Automaton(CpAutomaton {
                    vars: vec![39, 40, 41],
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
                }),
                CpConstraint::Circuit(vec![
                    CpCircuitArc {
                        tail: 0,
                        head: 1,
                        literal: BoolLiteral {
                            var: 42,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 1,
                        head: 2,
                        literal: BoolLiteral {
                            var: 43,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 2,
                        head: 0,
                        literal: BoolLiteral {
                            var: 44,
                            positive: true,
                        },
                    },
                ]),
                CpConstraint::DivisionEquality {
                    target: 47,
                    numerator: 45,
                    denominator: 46,
                },
                CpConstraint::ModuloEquality {
                    target: 48,
                    var: 45,
                    modulus: 46,
                },
                CpConstraint::Reservoir {
                    events: vec![
                        CpReservoirEvent {
                            time: 49,
                            level_change: 4,
                            active: None,
                        },
                        CpReservoirEvent {
                            time: 50,
                            level_change: -3,
                            active: None,
                        },
                        CpReservoirEvent {
                            time: 50,
                            level_change: 10,
                            active: Some(BoolLiteral {
                                var: 51,
                                positive: true,
                            }),
                        },
                    ],
                    min_level: 0,
                    max_level: 4,
                },
                CpConstraint::LinearDomain {
                    terms: vec![
                        LinearTerm { var: 52, coeff: 1 },
                        LinearTerm { var: 53, coeff: 2 },
                    ],
                    intervals: vec![
                        CpDomainInterval { lb: 1, ub: 1 },
                        CpDomainInterval { lb: 4, ub: 4 },
                    ],
                },
                CpConstraint::MapDomain {
                    var: 54,
                    bools: vec![55, 56, 57],
                    offset: 5,
                },
                CpConstraint::MultipleCircuit(vec![
                    CpCircuitArc {
                        tail: 0,
                        head: 1,
                        literal: BoolLiteral {
                            var: 58,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 1,
                        head: 0,
                        literal: BoolLiteral {
                            var: 59,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 0,
                        head: 2,
                        literal: BoolLiteral {
                            var: 60,
                            positive: true,
                        },
                    },
                    CpCircuitArc {
                        tail: 2,
                        head: 0,
                        literal: BoolLiteral {
                            var: 61,
                            positive: true,
                        },
                    },
                ]),
            ],
            objective: Some(CpObjective {
                sense: ObjectiveSense::Min,
                terms: vec![
                    LinearTerm { var: 0, coeff: 8 },
                    LinearTerm { var: 1, coeff: 2 },
                    LinearTerm { var: 2, coeff: 5 },
                    LinearTerm { var: 3, coeff: -1 },
                    LinearTerm { var: 4, coeff: 1 },
                    LinearTerm { var: 5, coeff: 1 },
                    LinearTerm { var: 6, coeff: 1 },
                    LinearTerm { var: 7, coeff: 1 },
                    LinearTerm { var: 8, coeff: 1 },
                    LinearTerm { var: 10, coeff: 1 },
                    LinearTerm { var: 11, coeff: 2 },
                    LinearTerm { var: 12, coeff: 1 },
                    LinearTerm { var: 14, coeff: 1 },
                    LinearTerm { var: 15, coeff: 1 },
                    LinearTerm { var: 16, coeff: 5 },
                    LinearTerm { var: 17, coeff: 4 },
                    LinearTerm { var: 18, coeff: 1 },
                    LinearTerm { var: 19, coeff: 1 },
                    LinearTerm { var: 20, coeff: 1 },
                    LinearTerm { var: 21, coeff: 2 },
                    LinearTerm { var: 26, coeff: 1 },
                    LinearTerm { var: 27, coeff: 1 },
                    LinearTerm { var: 29, coeff: 1 },
                    LinearTerm { var: 32, coeff: 1 },
                    LinearTerm { var: 33, coeff: 2 },
                    LinearTerm { var: 34, coeff: 1 },
                    LinearTerm { var: 39, coeff: 4 },
                    LinearTerm { var: 40, coeff: 2 },
                    LinearTerm { var: 41, coeff: 1 },
                    LinearTerm { var: 45, coeff: 1 },
                    LinearTerm { var: 47, coeff: 10 },
                    LinearTerm { var: 48, coeff: 1 },
                    LinearTerm { var: 49, coeff: 1 },
                    LinearTerm { var: 51, coeff: -1 },
                    LinearTerm { var: 52, coeff: 1 },
                    LinearTerm { var: 53, coeff: 1 },
                    LinearTerm { var: 54, coeff: 1 },
                    LinearTerm { var: 58, coeff: 1 },
                    LinearTerm { var: 59, coeff: 1 },
                    LinearTerm { var: 60, coeff: 1 },
                    LinearTerm { var: 61, coeff: 1 },
                ],
            }),
        }
    }

    fn model_json(model: &CpModel) -> String {
        let variables: Vec<_> = model
            .variables
            .iter()
            .map(|v| serde_json::json!({"name": v.name, "domain": v.domain}))
            .collect();
        let constraints: Vec<_> = model
            .constraints
            .iter()
            .map(|c| match c {
                CpConstraint::Linear { terms, sense, rhs } => serde_json::json!({
                    "kind": "linear",
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "sense": sense.as_str(),
                    "rhs": rhs,
                }),
                CpConstraint::LinearDomain { terms, intervals } => serde_json::json!({
                    "kind": "linear_domain",
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "lb": interval.lb,
                        "ub": interval.ub,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::MapDomain {
                    var,
                    bools,
                    offset,
                } => serde_json::json!({
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
                } => serde_json::json!({
                    "kind": "enforced_linear",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "terms": terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>(),
                    "sense": sense.as_str(),
                    "rhs": rhs,
                }),
                CpConstraint::EnforcedBoolOr {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_bool_or",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::EnforcedBoolAnd {
                    enforcement,
                    literals,
                } => serde_json::json!({
                    "kind": "enforced_bool_and",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "literals": literals.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::AllDifferent(vars) => {
                    serde_json::json!({"kind": "all_different", "vars": vars})
                }
                CpConstraint::BoolOr(lits) => serde_json::json!({
                    "kind": "bool_or",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::BoolAnd(lits) => serde_json::json!({
                    "kind": "bool_and",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::BoolXor(lits) => serde_json::json!({
                    "kind": "bool_xor",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::AtMostOne(lits) => serde_json::json!({
                    "kind": "at_most_one",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::ExactlyOne(lits) => serde_json::json!({
                    "kind": "exactly_one",
                    "literals": lits.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                }),
                CpConstraint::Implication {
                    antecedent,
                    consequent,
                } => serde_json::json!({
                    "kind": "implication",
                    "antecedent": {"var": antecedent.var, "positive": antecedent.positive},
                    "consequent": {"var": consequent.var, "positive": consequent.positive},
                }),
                CpConstraint::Circuit(arcs) => serde_json::json!({
                    "kind": "circuit",
                    "arcs": arcs.iter().map(|arc| serde_json::json!({
                        "tail": arc.tail,
                        "head": arc.head,
                        "literal": {"var": arc.literal.var, "positive": arc.literal.positive},
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::MultipleCircuit(arcs) => serde_json::json!({
                    "kind": "multiple_circuit",
                    "arcs": arcs.iter().map(|arc| serde_json::json!({
                        "tail": arc.tail,
                        "head": arc.head,
                        "literal": {"var": arc.literal.var, "positive": arc.literal.positive},
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::AllowedAssignments { vars, tuples } => serde_json::json!({
                    "kind": "allowed_assignments",
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::ForbiddenAssignments { vars, tuples } => serde_json::json!({
                    "kind": "forbidden_assignments",
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::EnforcedAllowedAssignments {
                    enforcement,
                    vars,
                    tuples,
                } => serde_json::json!({
                    "kind": "enforced_allowed_assignments",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::EnforcedForbiddenAssignments {
                    enforcement,
                    vars,
                    tuples,
                } => serde_json::json!({
                    "kind": "enforced_forbidden_assignments",
                    "enforcement": enforcement.iter().map(|lit| serde_json::json!({"var": lit.var, "positive": lit.positive})).collect::<Vec<_>>(),
                    "vars": vars,
                    "tuples": tuples,
                }),
                CpConstraint::Inverse { direct, inverse } => serde_json::json!({
                    "kind": "inverse",
                    "direct": direct,
                    "inverse": inverse,
                }),
                CpConstraint::MaxEquality { target, vars } => serde_json::json!({
                    "kind": "max_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::MinEquality { target, vars } => serde_json::json!({
                    "kind": "min_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::AbsEquality { target, var } => serde_json::json!({
                    "kind": "abs_equality",
                    "target": target,
                    "var": var,
                }),
                CpConstraint::MultiplicationEquality { target, vars } => serde_json::json!({
                    "kind": "multiplication_equality",
                    "target": target,
                    "vars": vars,
                }),
                CpConstraint::DivisionEquality {
                    target,
                    numerator,
                    denominator,
                } => serde_json::json!({
                    "kind": "division_equality",
                    "target": target,
                    "numerator": numerator,
                    "denominator": denominator,
                }),
                CpConstraint::ModuloEquality {
                    target,
                    var,
                    modulus,
                } => serde_json::json!({
                    "kind": "modulo_equality",
                    "target": target,
                    "var": var,
                    "modulus": modulus,
                }),
                CpConstraint::Automaton(automaton) => serde_json::json!({
                    "kind": "automaton",
                    "vars": automaton.vars,
                    "starting_state": automaton.starting_state,
                    "final_states": automaton.final_states,
                    "transitions": automaton.transitions.iter().map(|transition| serde_json::json!({
                        "tail": transition.tail,
                        "label": transition.label,
                        "head": transition.head,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::Element(element) => serde_json::json!({
                    "kind": "element",
                    "index": element.index,
                    "values": &element.values,
                    "target": element.target,
                }),
                CpConstraint::VariableElement(element) => serde_json::json!({
                    "kind": "variable_element",
                    "index": element.index,
                    "vars": &element.vars,
                    "target": element.target,
                }),
                CpConstraint::Alternative(alternative) => serde_json::json!({
                    "kind": "alternative",
                    "start": alternative.start,
                    "duration": alternative.duration,
                    "end": alternative.end,
                    "presence": alternative.presence.as_ref().map(|lit| serde_json::json!({
                        "var": lit.var,
                        "positive": lit.positive,
                    })),
                    "alternatives": alternative.alternatives.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "presence": interval.presence.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                    "name": alternative.name,
                }),
                CpConstraint::NoOverlap(intervals) => serde_json::json!({
                    "kind": "no_overlap",
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::NoOverlapVariable(intervals) => serde_json::json!({
                    "kind": "no_overlap_variable",
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::NoOverlap2D(rectangles) => serde_json::json!({
                    "kind": "no_overlap_2d",
                    "rectangles": rectangles.iter().map(|rectangle| serde_json::json!({
                        "x_start": rectangle.x_start,
                        "y_start": rectangle.y_start,
                        "width": rectangle.width,
                        "height": rectangle.height,
                        "name": rectangle.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::NoOverlap2DVariable(rectangles) => serde_json::json!({
                    "kind": "no_overlap_2d_variable",
                    "rectangles": rectangles.iter().map(|rectangle| serde_json::json!({
                        "x_start": rectangle.x_start,
                        "x_size": rectangle.x_size,
                        "x_end": rectangle.x_end,
                        "y_start": rectangle.y_start,
                        "y_size": rectangle.y_size,
                        "y_end": rectangle.y_end,
                        "name": rectangle.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::Cumulative {
                    intervals,
                    capacity,
                } => serde_json::json!({
                    "kind": "cumulative",
                    "capacity": capacity,
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "demand": interval.demand,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::CumulativeVariable {
                    intervals,
                    capacity,
                } => serde_json::json!({
                    "kind": "cumulative_variable",
                    "capacity": capacity,
                    "intervals": intervals.iter().map(|interval| serde_json::json!({
                        "start": interval.start,
                        "duration": interval.duration,
                        "end": interval.end,
                        "demand": interval.demand,
                        "name": interval.name,
                    })).collect::<Vec<_>>(),
                }),
                CpConstraint::Reservoir {
                    events,
                    min_level,
                    max_level,
                } => serde_json::json!({
                    "kind": "reservoir",
                    "min_level": min_level,
                    "max_level": max_level,
                    "events": events.iter().map(|event| serde_json::json!({
                        "time": event.time,
                        "level_change": event.level_change,
                        "active": event.active.as_ref().map(|lit| serde_json::json!({
                            "var": lit.var,
                            "positive": lit.positive,
                        })),
                    })).collect::<Vec<_>>(),
                }),
            })
            .collect();
        let objective = model.objective.as_ref().map(|obj| {
            serde_json::json!({
                "sense": obj.sense.as_str(),
                "terms": obj.terms.iter().map(|t| serde_json::json!({"var": t.var, "coeff": t.coeff})).collect::<Vec<_>>()
            })
        });
        serde_json::json!({
            "variables": variables,
            "constraints": constraints,
            "objective": objective,
        })
        .to_string()
    }

    fn run_reference(model: &CpModel) -> CpReference {
        let root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let script = format!("{root}/scripts/cp_sat_reference.py");
        let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let mut child = Command::new(python)
            .arg(script)
            .arg("--solver")
            .arg("auto")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start CP-SAT reference bridge");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(model_json(model).as_bytes())
            .expect("write CP model JSON");
        let out = child.wait_with_output().expect("wait for CP-SAT reference");
        assert!(
            out.status.success(),
            "reference failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).expect("parse CP-SAT reference JSON")
    }

    #[test]
    fn cp_model_matches_reference_bridge() {
        let model = sample_model();
        let internal = solve_cp_model(&model, CpSolveOptions::default());
        assert_eq!(internal.status, CpStatus::Optimal);

        let reference = run_reference(&model);
        assert_eq!(reference.status, "optimal", "{reference:?}");
        assert_eq!(internal.objective, reference.objective);
        assert_eq!(internal.assignment, reference.assignment);
        assert!(!reference.solver.is_empty());
    }
}
