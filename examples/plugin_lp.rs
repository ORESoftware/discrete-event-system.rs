//! Example **external plugin program** (Rust): a linear-programming solver that
//! emits a *single* JSON document (one result object) to stdout. The host
//! renders it with the **results** player — scalar fields become metric cards,
//! the `variables` / `constraints` arrays become tables, and a control toggles
//! the raw-JSON panel.
//!
//! Build + render via:
//!
//! ```bash
//! cargo build --example plugin_queue --example plugin_lp
//! cargo run   --example render_demo
//! ```

fn main() {
    struct Variable {
        name: &'static str,
        value: f64,
        objective: f64,
        reduced_cost: f64,
        lower: f64,
        upper: Option<f64>,
        basis: &'static str,
    }

    struct Constraint {
        name: &'static str,
        sense: &'static str,
        activity: f64,
        rhs: f64,
        residual: f64,
        dual: f64,
    }

    let variables = [
        Variable {
            name: "pumps",
            value: 12.0,
            objective: 44.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(24.0),
            basis: "basic",
        },
        Variable {
            name: "valves",
            value: 18.0,
            objective: 31.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(30.0),
            basis: "basic",
        },
        Variable {
            name: "motors",
            value: 7.0,
            objective: 86.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(12.0),
            basis: "basic",
        },
        Variable {
            name: "controllers",
            value: 5.0,
            objective: 73.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(10.0),
            basis: "basic",
        },
        Variable {
            name: "frames",
            value: 10.0,
            objective: 22.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(18.0),
            basis: "basic",
        },
        Variable {
            name: "sensors",
            value: 14.0,
            objective: 18.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: Some(25.0),
            basis: "basic",
        },
        Variable {
            name: "premium_kits",
            value: 0.0,
            objective: 105.0,
            reduced_cost: -8.25,
            lower: 0.0,
            upper: Some(8.0),
            basis: "nonbasic",
        },
        Variable {
            name: "rush_service",
            value: 9.0,
            objective: 16.0,
            reduced_cost: 0.0,
            lower: 0.0,
            upper: None,
            basis: "basic",
        },
    ];
    let constraints = [
        Constraint {
            name: "labor_hours",
            sense: "<=",
            activity: 620.0,
            rhs: 620.0,
            residual: 0.0,
            dual: 1.50,
        },
        Constraint {
            name: "cnc_hours",
            sense: "<=",
            activity: 360.0,
            rhs: 360.0,
            residual: 0.0,
            dual: 2.25,
        },
        Constraint {
            name: "assembly_slots",
            sense: "<=",
            activity: 392.0,
            rhs: 410.0,
            residual: 18.0,
            dual: 0.0,
        },
        Constraint {
            name: "steel_kg",
            sense: "<=",
            activity: 900.0,
            rhs: 900.0,
            residual: 0.0,
            dual: 0.42,
        },
        Constraint {
            name: "electronics_units",
            sense: "<=",
            activity: 480.0,
            rhs: 480.0,
            residual: 0.0,
            dual: 1.10,
        },
        Constraint {
            name: "packaging_units",
            sense: "<=",
            activity: 211.0,
            rhs: 260.0,
            residual: 49.0,
            dual: 0.0,
        },
        Constraint {
            name: "min_pumps_contract",
            sense: ">=",
            activity: 12.0,
            rhs: 10.0,
            residual: 2.0,
            dual: 0.0,
        },
        Constraint {
            name: "min_valves_contract",
            sense: ">=",
            activity: 18.0,
            rhs: 15.0,
            residual: 3.0,
            dual: 0.0,
        },
        Constraint {
            name: "shipping_pallets",
            sense: "<=",
            activity: 180.0,
            rhs: 180.0,
            residual: 0.0,
            dual: 0.35,
        },
        Constraint {
            name: "quality_budget",
            sense: "<=",
            activity: 87.0,
            rhs: 95.0,
            residual: 8.0,
            dual: 0.0,
        },
    ];
    let objective: f64 = variables.iter().map(|v| v.value * v.objective).sum();
    let result = serde_json::json!({
        "status": "optimal",
        "objectiveSense": "max",
        "objective": objective,
        "iterations": 18,
        "solveMs": 3.84,
        "algorithm": "revised-simplex",
        "variableCount": variables.len(),
        "constraintCount": constraints.len(),
        "variables": variables.iter().map(|v| serde_json::json!({
            "name": v.name,
            "value": v.value,
            "objective": v.objective,
            "reducedCost": v.reduced_cost,
            "lower": v.lower,
            "upper": v.upper,
            "basis": v.basis,
        })).collect::<Vec<_>>(),
        "constraints": constraints.iter().map(|c| serde_json::json!({
            "name": c.name,
            "sense": c.sense,
            "activity": c.activity,
            "rhs": c.rhs,
            "residual": c.residual,
            "dual": c.dual,
            "binding": c.residual.abs() < 1e-9,
        })).collect::<Vec<_>>(),
    });
    // Emit as one compact line (the host parses whole-stdout for OutputKind::Json).
    println!(
        "{}",
        serde_json::to_string(&result).expect("serialize LP result")
    );
}
