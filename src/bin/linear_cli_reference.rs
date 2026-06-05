use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use des_engine::des::general::external_linear_cli::{
    solve_linear_cli_json, ExternalLinearCliBranchRule, ExternalLinearCliKind,
    ExternalLinearCliLpAlgorithm, ExternalLinearCliMipSwitch, ExternalLinearCliModelFormat,
    ExternalLinearCliNodeSelection, ExternalLinearCliOptions, ExternalLinearCliPoolMember,
    ExternalLinearCliPresolve, ExternalLinearCliSolution, ExternalLinearCliSolver,
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
    format!(
        "usage: {program} --kind lp|mip --solver highs|glpk|scip|cbc|clp|soplex|qsopt-ex|lp-solve|gurobi|cplex|xpress|lindo [--problem PATH] [--model-format lp|mps]"
    )
}

fn parse_kind(value: &str) -> Result<ExternalLinearCliKind, String> {
    match value {
        "lp" => Ok(ExternalLinearCliKind::Lp),
        "mip" => Ok(ExternalLinearCliKind::Mip),
        _ => Err(format!("unknown kind {value:?}; expected lp or mip")),
    }
}

fn parse_solver(value: &str) -> Result<ExternalLinearCliSolver, String> {
    match value {
        "highs" => Ok(ExternalLinearCliSolver::Highs),
        "glpk" => Ok(ExternalLinearCliSolver::Glpk),
        "scip" => Ok(ExternalLinearCliSolver::Scip),
        "cbc" => Ok(ExternalLinearCliSolver::Cbc),
        "clp" => Ok(ExternalLinearCliSolver::Clp),
        "soplex" => Ok(ExternalLinearCliSolver::Soplex),
        "qsopt-ex" | "qsopt_ex" | "qsopt" => Ok(ExternalLinearCliSolver::QsoptEx),
        "lp-solve" | "lp_solve" | "lpsolve" => Ok(ExternalLinearCliSolver::LpSolve),
        "gurobi" => Ok(ExternalLinearCliSolver::Gurobi),
        "cplex" => Ok(ExternalLinearCliSolver::Cplex),
        "xpress" => Ok(ExternalLinearCliSolver::Xpress),
        "lindo" => Ok(ExternalLinearCliSolver::Lindo),
        _ => Err(format!("unknown solver {value:?}")),
    }
}

fn parse_model_format(value: &str) -> Result<ExternalLinearCliModelFormat, String> {
    match value {
        "lp" => Ok(ExternalLinearCliModelFormat::CplexLp),
        "mps" => Ok(ExternalLinearCliModelFormat::Mps),
        _ => Err(format!(
            "unknown model format {value:?}; expected lp or mps"
        )),
    }
}

fn parse_lp_algorithm(value: &str) -> Result<ExternalLinearCliLpAlgorithm, String> {
    match value {
        "simplex" => Ok(ExternalLinearCliLpAlgorithm::Simplex),
        "ipm" => Ok(ExternalLinearCliLpAlgorithm::Ipm),
        _ => Err(format!("unknown LP algorithm {value:?}")),
    }
}

fn parse_mip_switch(value: &str, name: &str) -> Result<ExternalLinearCliMipSwitch, String> {
    match value {
        "auto" => Ok(ExternalLinearCliMipSwitch::Auto),
        "on" => Ok(ExternalLinearCliMipSwitch::On),
        "off" => Ok(ExternalLinearCliMipSwitch::Off),
        _ => Err(format!("{name} must be auto, on, or off")),
    }
}

fn parse_presolve(value: &str) -> Result<ExternalLinearCliPresolve, String> {
    match value {
        "auto" => Ok(ExternalLinearCliPresolve::Auto),
        "on" => Ok(ExternalLinearCliPresolve::On),
        "off" => Ok(ExternalLinearCliPresolve::Off),
        _ => Err("presolve must be auto, on, or off".to_string()),
    }
}

fn parse_branch_rule(value: &str) -> Result<ExternalLinearCliBranchRule, String> {
    match value {
        "first-fractional" | "first_fractional" => Ok(ExternalLinearCliBranchRule::FirstFractional),
        "most-fractional" | "most_fractional" => Ok(ExternalLinearCliBranchRule::MostFractional),
        _ => Err("branch rule must be first-fractional or most-fractional".to_string()),
    }
}

fn parse_node_selection(value: &str) -> Result<ExternalLinearCliNodeSelection, String> {
    match value {
        "dfs" => Ok(ExternalLinearCliNodeSelection::Dfs),
        "best-bound" | "best_bound" => Ok(ExternalLinearCliNodeSelection::BestBound),
        _ => Err("node selection must be dfs or best-bound".to_string()),
    }
}

fn parse_json_vec_i32(value: &str, name: &str) -> Result<Vec<i32>, String> {
    serde_json::from_str::<Vec<i32>>(value)
        .map_err(|err| format!("{name} must be JSON ints: {err}"))
}

fn parse_json_vec_f64(value: &str, name: &str) -> Result<Vec<f64>, String> {
    serde_json::from_str::<Vec<f64>>(value)
        .map_err(|err| format!("{name} must be JSON numbers: {err}"))
}

fn next_value(
    program: &str,
    values: &mut impl Iterator<Item = String>,
    key: &str,
    inline_value: Option<String>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{key} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{key} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_args(
    raw_args: impl IntoIterator<Item = String>,
) -> Result<
    (
        ExternalLinearCliKind,
        ExternalLinearCliOptions,
        Option<PathBuf>,
    ),
    CliError,
> {
    let mut args = raw_args.into_iter();
    let program = args
        .next()
        .unwrap_or_else(|| "linear_cli_reference".to_string());
    let mut kind = None::<ExternalLinearCliKind>;
    let mut opts = ExternalLinearCliOptions::default();
    let mut problem_path = None::<PathBuf>;
    while let Some(raw) = args.next() {
        if raw == "-h" || raw == "--help" {
            return Err(CliError(usage(&program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        match key.as_str() {
            "--kind" => {
                let value = next_value(&program, &mut args, "--kind", inline_value)?;
                kind = Some(parse_kind(&value).map_err(CliError)?);
            }
            "--solver" => {
                let value = next_value(&program, &mut args, "--solver", inline_value)?;
                opts.solver = parse_solver(&value).map_err(CliError)?;
            }
            "--problem" => {
                let value = next_value(&program, &mut args, "--problem", inline_value)?;
                problem_path = Some(PathBuf::from(value));
            }
            "--model-format" => {
                let value = next_value(&program, &mut args, "--model-format", inline_value)?;
                opts.model_format = parse_model_format(&value).map_err(CliError)?;
            }
            "--time-limit" => {
                let value = next_value(&program, &mut args, "--time-limit", inline_value)?;
                opts.time_limit_secs = Some(
                    value
                        .parse::<f64>()
                        .map_err(|err| CliError(format!("--time-limit must be numeric: {err}")))?,
                );
            }
            "--node-limit" => {
                let value = next_value(&program, &mut args, "--node-limit", inline_value)?;
                opts.node_limit = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| CliError(format!("--node-limit must be integer: {err}")))?,
                );
            }
            "--solution-limit" => {
                let value = next_value(&program, &mut args, "--solution-limit", inline_value)?;
                opts.solution_limit =
                    Some(value.parse::<u64>().map_err(|err| {
                        CliError(format!("--solution-limit must be integer: {err}"))
                    })?);
            }
            "--solution-pool-size" => {
                let value = next_value(&program, &mut args, "--solution-pool-size", inline_value)?;
                opts.solution_pool_size = Some(value.parse::<u64>().map_err(|err| {
                    CliError(format!("--solution-pool-size must be integer: {err}"))
                })?);
            }
            "--relative-gap" => {
                let value = next_value(&program, &mut args, "--relative-gap", inline_value)?;
                opts.relative_gap =
                    Some(value.parse::<f64>().map_err(|err| {
                        CliError(format!("--relative-gap must be numeric: {err}"))
                    })?);
            }
            "--absolute-gap" => {
                let value = next_value(&program, &mut args, "--absolute-gap", inline_value)?;
                opts.absolute_gap =
                    Some(value.parse::<f64>().map_err(|err| {
                        CliError(format!("--absolute-gap must be numeric: {err}"))
                    })?);
            }
            "--objective-limit" => {
                let value = next_value(&program, &mut args, "--objective-limit", inline_value)?;
                opts.objective_limit = Some(value.parse::<f64>().map_err(|err| {
                    CliError(format!("--objective-limit must be numeric: {err}"))
                })?);
            }
            "--primal-feasibility-tolerance" => {
                let value = next_value(
                    &program,
                    &mut args,
                    "--primal-feasibility-tolerance",
                    inline_value,
                )?;
                opts.primal_feasibility_tolerance = Some(value.parse::<f64>().map_err(|err| {
                    CliError(format!(
                        "--primal-feasibility-tolerance must be numeric: {err}"
                    ))
                })?);
            }
            "--dual-feasibility-tolerance" => {
                let value = next_value(
                    &program,
                    &mut args,
                    "--dual-feasibility-tolerance",
                    inline_value,
                )?;
                opts.dual_feasibility_tolerance = Some(value.parse::<f64>().map_err(|err| {
                    CliError(format!(
                        "--dual-feasibility-tolerance must be numeric: {err}"
                    ))
                })?);
            }
            "--integer-feasibility-tolerance" => {
                let value = next_value(
                    &program,
                    &mut args,
                    "--integer-feasibility-tolerance",
                    inline_value,
                )?;
                opts.integer_feasibility_tolerance = Some(value.parse::<f64>().map_err(|err| {
                    CliError(format!(
                        "--integer-feasibility-tolerance must be numeric: {err}"
                    ))
                })?);
            }
            "--lp-algorithm" => {
                let value = next_value(&program, &mut args, "--lp-algorithm", inline_value)?;
                opts.lp_algorithm = Some(parse_lp_algorithm(&value).map_err(CliError)?);
            }
            "--threads" => {
                let value = next_value(&program, &mut args, "--threads", inline_value)?;
                opts.threads = Some(
                    value
                        .parse::<u32>()
                        .map_err(|err| CliError(format!("--threads must be integer: {err}")))?,
                );
            }
            "--random-seed" => {
                let value = next_value(&program, &mut args, "--random-seed", inline_value)?;
                opts.random_seed =
                    Some(value.parse::<u64>().map_err(|err| {
                        CliError(format!("--random-seed must be integer: {err}"))
                    })?);
            }
            "--presolve" => {
                let value = next_value(&program, &mut args, "--presolve", inline_value)?;
                opts.presolve = Some(parse_presolve(&value).map_err(CliError)?);
            }
            "--cuts" => {
                let value = next_value(&program, &mut args, "--cuts", inline_value)?;
                opts.cuts = Some(parse_mip_switch(&value, "--cuts").map_err(CliError)?);
            }
            "--heuristics" => {
                let value = next_value(&program, &mut args, "--heuristics", inline_value)?;
                opts.heuristics = Some(parse_mip_switch(&value, "--heuristics").map_err(CliError)?);
            }
            "--branch-rule" => {
                let value = next_value(&program, &mut args, "--branch-rule", inline_value)?;
                opts.branch_rule = Some(parse_branch_rule(&value).map_err(CliError)?);
            }
            "--branch-priorities" => {
                let value = next_value(&program, &mut args, "--branch-priorities", inline_value)?;
                opts.branch_priorities =
                    Some(parse_json_vec_i32(&value, "--branch-priorities").map_err(CliError)?);
            }
            "--node-selection" => {
                let value = next_value(&program, &mut args, "--node-selection", inline_value)?;
                opts.node_selection = Some(parse_node_selection(&value).map_err(CliError)?);
            }
            "--mip-start" => {
                let value = next_value(&program, &mut args, "--mip-start", inline_value)?;
                opts.mip_start = Some(parse_json_vec_f64(&value, "--mip-start").map_err(CliError)?);
            }
            "--command-path" => {
                let value = next_value(&program, &mut args, "--command-path", inline_value)?;
                opts.command_path = Some(PathBuf::from(value));
            }
            "--python" => {
                let value = next_value(&program, &mut args, "--python", inline_value)?;
                opts.python = Some(value);
            }
            "--script-path" => {
                let value = next_value(&program, &mut args, "--script-path", inline_value)?;
                opts.script_path = Some(PathBuf::from(value));
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(&program)
                )))
            }
        }
    }
    let kind = kind.ok_or_else(|| CliError(format!("--kind is required\n{}", usage(&program))))?;
    Ok((kind, opts, problem_path))
}

fn pool_member_json(member: &ExternalLinearCliPoolMember) -> Value {
    json!({
        "x": member.x,
        "objective": member.objective,
    })
}

fn optional_solution_fields(output: &mut Value, solution: &ExternalLinearCliSolution) {
    if let Some(value) = &solution.solver_version {
        output["solverVersion"] = json!(value);
    }
    if let Some(value) = &solution.objective_values {
        output["objectiveValues"] = json!(value);
    }
    if let Some(value) = &solution.lp_algorithm {
        output["lpAlgorithm"] = json!(value);
    }
    if let Some(value) = solution.best_bound {
        output["bestBound"] = json!(value);
    }
    if let Some(value) = solution.solution_limit {
        output["solutionLimit"] = json!(value);
    }
    if let Some(value) = solution.solution_pool_size {
        output["solutionPoolSize"] = json!(value);
    }
    if let Some(solutions) = &solution.solutions {
        output["solutions"] = json!(solutions.iter().map(pool_member_json).collect::<Vec<_>>());
    }
    if let Some(value) = solution.exhausted {
        output["exhausted"] = json!(value);
    }
    if let Some(value) = solution.mip_gap {
        output["mipGap"] = json!(value);
    }
    if let Some(value) = solution.absolute_gap {
        output["absoluteGap"] = json!(value);
    }
    if let Some(value) = solution.objective_limit {
        output["objectiveLimit"] = json!(value);
    }
    if let Some(value) = solution.primal_feasibility_tolerance {
        output["primalFeasibilityTolerance"] = json!(value);
    }
    if let Some(value) = solution.dual_feasibility_tolerance {
        output["dualFeasibilityTolerance"] = json!(value);
    }
    if let Some(value) = solution.integer_feasibility_tolerance {
        output["integerFeasibilityTolerance"] = json!(value);
    }
    if let Some(value) = solution.nodes_explored {
        output["nodesExplored"] = json!(value);
    }
    if let Some(value) = solution.threads {
        output["threads"] = json!(value);
    }
    if let Some(value) = solution.random_seed {
        output["randomSeed"] = json!(value);
    }
    if let Some(value) = &solution.presolve {
        output["presolve"] = json!(value);
    }
    if let Some(value) = &solution.cuts {
        output["cuts"] = json!(value);
    }
    if let Some(value) = &solution.heuristics {
        output["heuristics"] = json!(value);
    }
    if let Some(value) = &solution.branch_rule {
        output["branchRule"] = json!(value);
    }
    if let Some(value) = solution.branch_priorities_accepted {
        output["branchPrioritiesAccepted"] = json!(value);
    }
    if let Some(value) = solution.branch_priority_count {
        output["branchPriorityCount"] = json!(value);
    }
    if let Some(value) = &solution.node_selection {
        output["nodeSelection"] = json!(value);
    }
    if let Some(value) = solution.mip_start_accepted {
        output["mipStartAccepted"] = json!(value);
    }
    if let Some(value) = solution.mip_start_objective {
        output["mipStartObjective"] = json!(value);
    }
    if let Some(value) = &solution.dual_ub {
        output["dualUb"] = json!(value);
    }
    if let Some(value) = &solution.dual_eq {
        output["dualEq"] = json!(value);
    }
    if let Some(value) = &solution.reduced_costs {
        output["reducedCosts"] = json!(value);
    }
    if let Some(value) = &solution.var_basis {
        output["varBasis"] = json!(value);
    }
    if let Some(value) = &solution.row_basis {
        output["rowBasis"] = json!(value);
    }
    if let Some(value) = solution.iterations {
        output["iterations"] = json!(value);
    }
}

fn solution_json(solution: &ExternalLinearCliSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "x": solution.x,
        "objective": solution.objective,
        "elapsedMs": solution.elapsed_ms,
        "message": solution.message,
    });
    optional_solution_fields(&mut output, solution);
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:linear-cli-reference",
        "x": [],
        "objective": null,
        "elapsedMs": 0.0,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let (kind, opts, problem_path) = parse_args(raw_args)?;
    let problem_text = if let Some(path) = problem_path {
        fs::read_to_string(&path)
            .map_err(|err| CliError(format!("failed to read {}: {err}", path.display())))?
    } else {
        stdin.to_string()
    };
    let payload = serde_json::from_str::<Value>(&problem_text)
        .map_err(|err| CliError(format!("failed to parse JSON problem: {err}")))?;
    let solution = solve_linear_cli_json(kind, payload, &opts);
    Ok(solution_json(&solution))
}

fn main() {
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!("{}", error_json(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(env::args().collect::<Vec<_>>(), &stdin) {
        Ok(output) => {
            println!(
                "{}",
                serde_json::to_string(&output).expect("serialize linear CLI output")
            );
        }
        Err(error) => {
            println!("{}", error_json(error.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LP: &str = r#"{
        "sense": "min",
        "c": [1.0],
        "A_ub": [[1.0]],
        "b_ub": [2.0],
        "lb": [0.0],
        "ub": [null]
    }"#;

    #[test]
    fn parses_core_python_bridge_options() {
        let (_, opts, problem_path) = parse_args([
            "linear_cli_reference".to_string(),
            "--kind".to_string(),
            "mip".to_string(),
            "--solver".to_string(),
            "cbc".to_string(),
            "--model-format=mps".to_string(),
            "--time-limit".to_string(),
            "3.5".to_string(),
            "--solution-pool-size".to_string(),
            "2".to_string(),
            "--branch-priorities".to_string(),
            "[0,3,1]".to_string(),
            "--mip-start".to_string(),
            "[1.0,0.0,1.0]".to_string(),
            "--problem".to_string(),
            "problem.json".to_string(),
        ])
        .expect("parse args");

        assert_eq!(opts.solver, ExternalLinearCliSolver::Cbc);
        assert_eq!(opts.model_format, ExternalLinearCliModelFormat::Mps);
        assert_eq!(opts.time_limit_secs, Some(3.5));
        assert_eq!(opts.solution_pool_size, Some(2));
        assert_eq!(opts.branch_priorities, Some(vec![0, 3, 1]));
        assert_eq!(opts.mip_start, Some(vec![1.0, 0.0, 1.0]));
        assert_eq!(problem_path, Some(PathBuf::from("problem.json")));
    }

    #[test]
    fn plain_lp_payload_enters_rust_linear_cli_path() {
        let output = run(
            vec![
                "linear_cli_reference".to_string(),
                "--kind".to_string(),
                "lp".to_string(),
                "--solver".to_string(),
                "highs".to_string(),
                "--python".to_string(),
                "/definitely/not/python".to_string(),
            ],
            LP,
        )
        .expect("linear CLI output");

        assert_eq!(output["solver"], json!("highs:cli"));
        assert_ne!(output["status"], json!("error"));
        assert!(!output["message"]
            .as_str()
            .unwrap_or_default()
            .contains("/definitely/not/python"));
    }
}
