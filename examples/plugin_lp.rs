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
    // A canned but realistic LP solution payload.
    let result = r#"{
        "status": "optimal",
        "objective": 36.0,
        "iterations": 7,
        "solveMs": 0.42,
        "variables": [
            {"name": "x1", "value": 2.0, "reducedCost": 0.0},
            {"name": "x2", "value": 6.0, "reducedCost": 0.0},
            {"name": "x3", "value": 0.0, "reducedCost": -1.5}
        ],
        "constraints": [
            {"name": "labor",    "slack": 0.0, "dual": 1.0},
            {"name": "material", "slack": 3.0, "dual": 0.0},
            {"name": "demand",   "slack": 0.0, "dual": 0.5}
        ]
    }"#;
    // Emit as one compact line (the host parses whole-stdout for OutputKind::Json).
    let compact: String = result.split_whitespace().collect::<Vec<_>>().join(" ");
    println!("{compact}");
}
