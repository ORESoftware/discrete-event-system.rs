use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_optimization_tools::{
    external_optimization_tools, run_external_optimization_ecosystem_reference_with_rust_builtin,
    ExternalOptimizationTool,
};
use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage(program: &str) -> String {
    format!("usage: {program} [--tool TOOL]")
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{option} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<String, CliError> {
    let mut tool =
        env::var("ORES_EXTERNAL_OPTIMIZATION_TOOL").unwrap_or_else(|_| "auto".to_string());
    let mut values = args.into_iter();
    while let Some(raw) = values.next() {
        if raw == "-h" || raw == "--help" {
            return Err(CliError(usage(program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        match key.as_str() {
            "--tool" => {
                tool = next_option_value(program, "--tool", inline_value, &mut values)?;
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(tool)
}

fn normalize_tool(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn default_tool_for_payload(payload: &Value) -> ExternalOptimizationTool {
    match payload
        .get("kind")
        .and_then(Value::as_str)
        .map(normalize_tool)
        .unwrap_or_default()
        .as_str()
    {
        "cp-assignment" | "ecosystem-cp-assignment" | "cp-job-shop" | "ecosystem-cp-job-shop" => {
            ExternalOptimizationTool::ChocoSolver
        }
        "planning-assignment" | "ecosystem-planning-assignment" => {
            ExternalOptimizationTool::OptaPlanner
        }
        "multiobjective-front" | "ecosystem-multiobjective" => ExternalOptimizationTool::JMetal,
        "nonlinear-program" | "ecosystem-nonlinear" => ExternalOptimizationTool::Argmin,
        _ => ExternalOptimizationTool::OjAlgo,
    }
}

fn parse_tool(raw: &str, payload: &Value) -> Result<ExternalOptimizationTool, CliError> {
    let normalized = normalize_tool(raw);
    if normalized == "auto" || normalized.is_empty() {
        return Ok(default_tool_for_payload(payload));
    }
    external_optimization_tools()
        .iter()
        .copied()
        .find(|tool| tool.as_str() == normalized)
        .ok_or_else(|| {
            CliError(format!(
                "unknown optimization ecosystem tool {normalized:?}"
            ))
        })
}

fn error_json(tool: &str, message: impl Into<String>) -> Value {
    json!({
        "kind": "optimization-ecosystem-reference-result",
        "tool": tool,
        "family": "unknown",
        "status": "invalid",
        "objective": null,
        "x": null,
        "message": message.into(),
        "backend": "builtin-rust:optimization-ecosystem-reference",
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "optimization_ecosystem_reference".to_string());
    let raw_tool = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    if !payload.is_object() {
        return Err(CliError("top-level payload must be an object".to_string()));
    }
    let tool = parse_tool(&raw_tool, &payload)?;
    let run = run_external_optimization_ecosystem_reference_with_rust_builtin(&payload, tool);
    Ok(run.output.unwrap_or_else(|| {
        error_json(
            tool.as_str(),
            format!(
                "Rust ecosystem reference returned status {}: {}",
                run.status.as_str(),
                run.message
            ),
        )
    }))
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                raw_args
                    .first()
                    .map(String::as_str)
                    .unwrap_or("optimization_ecosystem_reference")
            )
        );
        return;
    }
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!(
            "{}",
            error_json("auto", format!("failed to read stdin: {err}"))
        );
        std::process::exit(1);
    }
    match run(raw_args, &stdin) {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize optimization ecosystem output")
        ),
        Err(err) => {
            println!("{}", error_json("auto", err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cp_assignment_cli_uses_rust_backend() {
        let output = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool".to_string(),
                "choco-solver".to_string(),
            ],
            r#"{
                "kind": "ecosystem-cp-assignment",
                "costs": [[3, 1], [2, 4]]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["family"], "constraint-programming");
        assert_eq!(output["backend"], "builtin-rust:constraint-programming");
        assert_eq!(output["objective"], 3.0);
    }

    #[test]
    fn multiobjective_cli_uses_rust_backend() {
        let output = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool=jmetal".to_string(),
            ],
            r#"{
                "kind": "ecosystem-multiobjective",
                "senses": ["min", "min"],
                "candidates": [
                    {"x": [0], "objectives": [3, 2]},
                    {"x": [1], "objectives": [2, 2]},
                    {"x": [2], "objectives": [4, 1]}
                ]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["family"], "evolutionary-multiobjective");
        assert_eq!(
            output["backend"],
            "builtin-rust:evolutionary-multiobjective"
        );
        assert_eq!(output["objective"], 4.0);
    }

    #[test]
    fn cuopt_cli_uses_rust_linear_mip_backend() {
        let output = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool=nvidia-cuopt".to_string(),
            ],
            r#"{
                "kind": "ecosystem-linear-binary",
                "sense": "max",
                "objective": [3, 2],
                "constraints": [{"coefs": [1, 1], "sense": "<=", "rhs": 1}],
                "domains": [[0, 1], [0, 1]]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["tool"], "nvidia-cuopt");
        assert_eq!(output["family"], "linear-mip");
        assert_eq!(output["backend"], "builtin-rust:linear-mip");
        assert_eq!(output["objective"], 3.0);
    }

    #[test]
    fn auto_selects_family_from_payload_kind() {
        let output = run(
            vec!["optimization_ecosystem_reference".to_string()],
            r#"{
                "kind": "ecosystem-planning-assignment",
                "task_durations": [2, 3, 4],
                "machines": 2
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["tool"], "optaplanner");
        assert_eq!(output["family"], "planning-metaheuristic");
        assert_eq!(output["backend"], "builtin-rust:planning-metaheuristic");
    }

    #[test]
    fn cli_tool_normalization_is_owned_by_rust_dispatch() {
        let choco = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool".to_string(),
                "ChOcO_SoLvEr".to_string(),
            ],
            r#"{
                "kind": "ecosystem-cp-assignment",
                "costs": [[3, 1], [2, 4]]
            }"#,
        )
        .expect("run");

        assert_eq!(choco["status"], "optimal");
        assert_eq!(choco["tool"], "choco-solver");
        assert_eq!(choco["family"], "constraint-programming");

        let cuopt = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool=NvIdIa_CuOpT".to_string(),
            ],
            r#"{
                "kind": "ecosystem-linear-binary",
                "sense": "max",
                "objective": [3, 2],
                "constraints": [{"coefs": [1, 1], "sense": "<=", "rhs": 1}],
                "domains": [[0, 1], [0, 1]]
            }"#,
        )
        .expect("run");

        assert_eq!(cuopt["status"], "optimal");
        assert_eq!(cuopt["tool"], "nvidia-cuopt");
        assert_eq!(cuopt["family"], "linear-mip");
    }

    #[test]
    fn java_and_rust_ecosystem_aliases_use_rust_builtins() {
        let cp_payload = r#"{
            "kind": "ecosystem-cp-assignment",
            "costs": [[3, 1], [2, 4]]
        }"#;
        for tool in ["jacop", "ortools-java"] {
            let output = run(
                vec![
                    "optimization_ecosystem_reference".to_string(),
                    "--tool".to_string(),
                    tool.to_string(),
                ],
                cp_payload,
            )
            .expect("run CP ecosystem alias");

            assert_eq!(output["status"], "optimal", "{tool}: {output}");
            assert_eq!(output["tool"], tool, "{tool}: {output}");
            assert_eq!(
                output["family"], "constraint-programming",
                "{tool}: {output}"
            );
            assert_eq!(
                output["backend"], "builtin-rust:constraint-programming",
                "{tool}: {output}"
            );
            assert_eq!(output["objective"], 3.0, "{tool}: {output}");
        }

        let planning_payload = r#"{
            "kind": "ecosystem-planning-assignment",
            "task_durations": [2, 3, 4],
            "machines": 2
        }"#;
        let optaplanner = run(
            vec![
                "optimization_ecosystem_reference".to_string(),
                "--tool".to_string(),
                "optaplanner".to_string(),
            ],
            planning_payload,
        )
        .expect("run planning ecosystem alias");
        assert_eq!(optaplanner["status"], "optimal");
        assert_eq!(optaplanner["family"], "planning-metaheuristic");
        assert_eq!(
            optaplanner["backend"],
            "builtin-rust:planning-metaheuristic"
        );

        let multiobjective_payload = r#"{
            "kind": "ecosystem-multiobjective",
            "senses": ["min", "min"],
            "candidates": [
                {"x": [0], "objectives": [3, 2]},
                {"x": [1], "objectives": [2, 2]},
                {"x": [2], "objectives": [4, 1]}
            ]
        }"#;
        for tool in ["moea-framework", "ecj"] {
            let output = run(
                vec![
                    "optimization_ecosystem_reference".to_string(),
                    "--tool".to_string(),
                    tool.to_string(),
                ],
                multiobjective_payload,
            )
            .expect("run multiobjective ecosystem alias");

            assert_eq!(output["status"], "optimal", "{tool}: {output}");
            assert_eq!(output["tool"], tool, "{tool}: {output}");
            assert_eq!(
                output["family"], "evolutionary-multiobjective",
                "{tool}: {output}"
            );
            assert_eq!(
                output["backend"], "builtin-rust:evolutionary-multiobjective",
                "{tool}: {output}"
            );
        }

        let linear_payload = r#"{
            "kind": "ecosystem-linear-binary",
            "sense": "max",
            "objective": [3, 2],
            "constraints": [{"coefs": [1, 1], "sense": "<=", "rhs": 1}],
            "domains": [[0, 1], [0, 1]]
        }"#;
        for tool in ["ojalgo", "good-lp", "lp-modeler"] {
            let output = run(
                vec![
                    "optimization_ecosystem_reference".to_string(),
                    "--tool".to_string(),
                    tool.to_string(),
                ],
                linear_payload,
            )
            .expect("run linear ecosystem alias");

            assert_eq!(output["status"], "optimal", "{tool}: {output}");
            assert_eq!(output["tool"], tool, "{tool}: {output}");
            assert_eq!(output["family"], "linear-mip", "{tool}: {output}");
            assert_eq!(
                output["backend"], "builtin-rust:linear-mip",
                "{tool}: {output}"
            );
            assert_eq!(output["objective"], 3.0, "{tool}: {output}");
        }

        let nonlinear_payload = r#"{
            "kind": "ecosystem-nonlinear",
            "sense": "min",
            "variables": [
                {"name": "x", "lb": 0, "ub": 2, "start": 1},
                {"name": "y", "lb": 0, "ub": 4, "start": 2}
            ],
            "objective": "(x - 1) * (x - 1) + (y - 2) * (y - 2)"
        }"#;
        for tool in ["argmin", "nlopt"] {
            let output = run(
                vec![
                    "optimization_ecosystem_reference".to_string(),
                    "--tool".to_string(),
                    tool.to_string(),
                ],
                nonlinear_payload,
            )
            .expect("run nonlinear ecosystem alias");

            assert_eq!(output["status"], "optimal", "{tool}: {output}");
            assert_eq!(output["tool"], tool, "{tool}: {output}");
            assert_eq!(
                output["family"], "nonlinear-optimization",
                "{tool}: {output}"
            );
            assert_eq!(
                output["backend"], "builtin-rust:nonlinear-optimization",
                "{tool}: {output}"
            );
            assert_eq!(output["objective"], 0.0, "{tool}: {output}");
            assert_eq!(output["x"], json!([1.0, 2.0]), "{tool}: {output}");
        }
    }
}
