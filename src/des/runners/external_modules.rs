//! Port of `src/des/runners/external-modules.ts`.
//!
//! Metadata registry of the built-in external reference modules. This file is
//! metadata only: it registers source scripts under `external-references/` and
//! Rust reference binaries under `src/bin/`, then describes how to invoke them.
//!
//! ## PORT NOTE
//!
//!   * the `let registered = false` idempotency guard → `std::sync::Once`.
//!   * each module's `buildArgs` arrow function → a free `fn` of type
//!     [`BuildArgsFn`](crate::des::runners::external_program::BuildArgsFn).
//!   * `path.join(ctx.moduleOutDir, '…')` → `PathBuf::join`.
//!   * the TS bottom-of-file `registerBuiltInExternalModules()` side-effect (run
//!     on import) becomes an explicit call from
//!     [`register_built_in_external_modules`]; callers (the CLI) invoke it at
//!     startup. Registration validates module metadata and source locations, but
//!     actual missing optional scripts are reported when the module is run.

#![allow(dead_code)]

use std::sync::OnceLock;

use super::external_program::{
    register_external_module, ExternalInterpreterSpec, ExternalModuleContext, ExternalModuleKind,
    ExternalModuleParams, ExternalProgramModule, ParamValue,
};

pub const NEURAL_NETWORK_REFERENCE_ID: &str = "neural-network-reference";
pub const COMPUTER_NETWORK_REFERENCE_ID: &str = "computer-network-reference";
pub const COMPUTER_NETWORK_FEL_REFERENCE_ID: &str = "computer-network-fel-reference";
pub const IP_MIP_REFERENCE_ID: &str = "ip-mip-reference";
pub const CP_SAT_REFERENCE_ID: &str = "cp-sat-reference";
pub const STOCHASTIC_LP_REFERENCE_ID: &str = "stochastic-lp-reference";
pub const QP_REFERENCE_ID: &str = "qp-reference";
pub const NONLINEAR_REFERENCE_ID: &str = "nonlinear-reference";
pub const NONLINEAR_VALIDATION_REFERENCE_ID: &str = "nonlinear-validation-reference";
pub const LINEAR_CLI_REFERENCE_ID: &str = "linear-cli-reference";
pub const LP_SOLVE_REFERENCE_ID: &str = "lp-solve-reference";
pub const OPTIMIZATION_ECOSYSTEM_REFERENCE_ID: &str = "optimization-ecosystem-reference";
pub const ASSIGNMENT_REFERENCE_ID: &str = "assignment-reference";
pub const KNAPSACK_REFERENCE_ID: &str = "knapsack-reference";
pub const BIN_PACKING_REFERENCE_ID: &str = "bin-packing-reference";
pub const FACILITY_LOCATION_REFERENCE_ID: &str = "facility-location-reference";
pub const GRAPH_COLORING_REFERENCE_ID: &str = "graph-coloring-reference";
pub const SET_COVER_REFERENCE_ID: &str = "set-cover-reference";
pub const TSP_REFERENCE_ID: &str = "tsp-reference";
pub const MAX_FLOW_REFERENCE_ID: &str = "max-flow-reference";
pub const MIN_COST_FLOW_REFERENCE_ID: &str = "min-cost-flow-reference";
pub const MINIMUM_SPANNING_TREE_REFERENCE_ID: &str = "minimum-spanning-tree-reference";
pub const ROUTING_REFERENCE_ID: &str = "routing-reference";
pub const SCHEDULING_REFERENCE_ID: &str = "scheduling-reference";
pub const WEIGHTED_INDEPENDENT_SET_REFERENCE_ID: &str = "weighted-independent-set-reference";
pub const WEIGHTED_MAX_SAT_REFERENCE_ID: &str = "weighted-max-sat-reference";
pub const OUTPUT_VALIDATION_REFERENCE_ID: &str = "output-validation-reference";
pub const MODEL_VALIDATION_REFERENCE_ID: &str = "model-validation-reference";
pub const PROOF_VALIDATION_REFERENCE_ID: &str = "proof-validation-reference";
pub const FORMAL_BENCHMARK_VALIDATION_REFERENCE_ID: &str = "formal-benchmark-validation-reference";
pub const SIMULATION_VALIDATION_REFERENCE_ID: &str = "simulation-validation-reference";
pub const TRAFFIC_FEL_REFERENCE_ID: &str = "traffic-fel-reference";
pub const TRAFFIC_SIMPY_REFERENCE_ID: &str = "traffic-simpy-reference";
pub const TRAFFIC_CIW_REFERENCE_ID: &str = "traffic-ciw-reference";
pub const TRAFFIC_SUMO_REFERENCE_ID: &str = "traffic-sumo-reference";

fn python3() -> ExternalInterpreterSpec {
    ExternalInterpreterSpec {
        env_var: "PYTHON_BIN".to_string(),
        default_command: "python3".to_string(),
        label: "Python 3".to_string(),
    }
}

fn rust_cargo() -> ExternalInterpreterSpec {
    ExternalInterpreterSpec {
        env_var: "CARGO".to_string(),
        default_command: "cargo".to_string(),
        label: "Cargo/Rust".to_string(),
    }
}

fn n(v: f64) -> ParamValue {
    ParamValue::Num(v)
}
fn s(v: &str) -> ParamValue {
    ParamValue::Str(v.to_string())
}

fn params(entries: &[(&str, ParamValue)]) -> ExternalModuleParams {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// `stringParam(v, fallback)`.
fn string_param(v: Option<&ParamValue>, fallback: &str) -> String {
    match v {
        None => fallback.to_string(),
        Some(pv) => param_to_string(pv),
    }
}

/// `numberParam(v, fallback)`.
fn number_param(v: Option<&ParamValue>, fallback: f64) -> Result<String, String> {
    match v {
        None => Ok(num_to_js_string(fallback)),
        Some(pv) => {
            let num = match pv {
                ParamValue::Num(x) => *x,
                ParamValue::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ParamValue::Str(text) => text.trim().parse::<f64>().unwrap_or(f64::NAN),
            };
            if !num.is_finite() {
                return Err(format!(
                    "expected finite numeric external param, got {}",
                    param_to_string(pv)
                ));
            }
            Ok(num_to_js_string(num))
        }
    }
}

fn bool_param(v: Option<&ParamValue>, fallback: bool) -> Result<bool, String> {
    match v {
        None => Ok(fallback),
        Some(ParamValue::Bool(value)) => Ok(*value),
        Some(ParamValue::Num(value)) if *value == 0.0 => Ok(false),
        Some(ParamValue::Num(value)) if *value == 1.0 => Ok(true),
        Some(ParamValue::Num(value)) => Err(format!(
            "expected boolean external param, got {}",
            num_to_js_string(*value)
        )),
        Some(ParamValue::Str(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => Err(format!("expected boolean external param, got {text}")),
        },
    }
}

/// `String(value)` for a defined param.
fn param_to_string(v: &ParamValue) -> String {
    match v {
        ParamValue::Str(text) => text.clone(),
        ParamValue::Num(x) => num_to_js_string(*x),
        ParamValue::Bool(b) => b.to_string(),
    }
}

/// JS `String(number)`: integers render without a trailing `.0`.
fn num_to_js_string(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        x.to_string()
    }
}

fn join_out(ctx: &ExternalModuleContext, file: &str) -> String {
    ctx.module_out_dir.join(file).display().to_string()
}

fn build_neural_network(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "reference.json"));
    Ok(vec![
        "--out".into(),
        out,
        "--seed".into(),
        number_param(p.get("seed"), 7.0)?,
        "--xor-epochs".into(),
        number_param(p.get("xorEpochs"), 8000.0)?,
        "--xor-lr".into(),
        number_param(p.get("xorLr"), 0.3)?,
        "--corridor-length".into(),
        number_param(p.get("corridorLength"), 6.0)?,
        "--corridor-gamma".into(),
        number_param(p.get("corridorGamma"), 0.95)?,
        "--ode-rate".into(),
        number_param(p.get("odeRate"), 0.5)?,
        "--ode-y0".into(),
        number_param(p.get("odeY0"), 1.0)?,
        "--ode-t1".into(),
        number_param(p.get("odeT1"), 2.0)?,
        "--ode-dt".into(),
        number_param(p.get("odeDt"), 0.05)?,
    ])
}

fn build_computer_network(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "reference.json"));
    let mut args = vec!["--out".to_string(), out];
    if p.get("problem").is_some() {
        args.push("--problem".into());
        args.push(string_param(p.get("problem"), ""));
    } else {
        args.push("--builtin".into());
        args.push(string_param(p.get("builtin"), "bottleneck-lab"));
    }
    Ok(args)
}

fn build_computer_network_fel(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "fel-reference.json"));
    let mut args = vec!["--out".to_string(), out];
    if p.get("problem").is_some() {
        args.push("--problem".into());
        args.push(string_param(p.get("problem"), ""));
    } else {
        args.push("--builtin".into());
        args.push(string_param(p.get("builtin"), "bottleneck-lab"));
    }
    Ok(args)
}

fn build_ip_mip(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "reference.json"));
    Ok(vec![
        "--problem".into(),
        string_param(p.get("problem"), ""),
        "--out".into(),
        out,
        "--solver".into(),
        string_param(p.get("solver"), "auto"),
        "--max-enumerations".into(),
        number_param(p.get("maxEnumerations"), 1_000_000.0)?,
    ])
}

fn build_assignment(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_stdin_solver(p: &ExternalModuleParams) -> Result<Vec<String>, String> {
    Ok(vec![
        "--solver".into(),
        string_param(p.get("solver"), "auto"),
    ])
}

fn push_optional_number_arg(
    args: &mut Vec<String>,
    p: &ExternalModuleParams,
    key: &str,
    flag: &str,
) -> Result<(), String> {
    if p.get(key).is_some() {
        args.push(flag.into());
        args.push(number_param(p.get(key), 0.0)?);
    }
    Ok(())
}

fn push_optional_string_arg(
    args: &mut Vec<String>,
    p: &ExternalModuleParams,
    key: &str,
    flag: &str,
) {
    if let Some(value) = p.get(key) {
        args.push(flag.into());
        args.push(param_to_string(value));
    }
}

fn build_cp_sat(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let mut args = vec!["--solver".into(), string_param(p.get("solver"), "auto")];
    push_optional_number_arg(&mut args, p, "enumerateSolutions", "--enumerate-solutions")?;
    if bool_param(p.get("assumptionCore"), false)? {
        args.push("--assumption-core".into());
    }
    Ok(args)
}

fn build_qp(p: &ExternalModuleParams, _: &ExternalModuleContext) -> Result<Vec<String>, String> {
    let mut args = vec!["--solver".into(), string_param(p.get("solver"), "auto")];
    push_optional_number_arg(&mut args, p, "maxEnumerations", "--max-enumerations")?;
    Ok(args)
}

fn build_nonlinear(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let mut args = vec!["--solver".into(), string_param(p.get("solver"), "auto")];
    push_optional_number_arg(&mut args, p, "maxIterations", "--max-iterations")?;
    Ok(args)
}

fn build_lp_solve(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    Ok(vec![
        "--method".into(),
        string_param(p.get("method"), "rust"),
    ])
}

fn build_optimization_ecosystem(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    Ok(vec!["--tool".into(), string_param(p.get("tool"), "auto")])
}

fn build_linear_cli(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let mut args = vec![
        "--kind".into(),
        string_param(p.get("kind"), "lp"),
        "--solver".into(),
        string_param(p.get("solver"), "highs"),
    ];
    push_optional_string_arg(&mut args, p, "problem", "--problem");
    push_optional_string_arg(&mut args, p, "modelFormat", "--model-format");
    push_optional_number_arg(&mut args, p, "timeLimit", "--time-limit")?;
    push_optional_number_arg(&mut args, p, "nodeLimit", "--node-limit")?;
    push_optional_number_arg(&mut args, p, "solutionLimit", "--solution-limit")?;
    push_optional_number_arg(&mut args, p, "solutionPoolSize", "--solution-pool-size")?;
    push_optional_number_arg(&mut args, p, "relativeGap", "--relative-gap")?;
    push_optional_number_arg(&mut args, p, "absoluteGap", "--absolute-gap")?;
    push_optional_number_arg(&mut args, p, "objectiveLimit", "--objective-limit")?;
    push_optional_number_arg(
        &mut args,
        p,
        "primalFeasibilityTolerance",
        "--primal-feasibility-tolerance",
    )?;
    push_optional_number_arg(
        &mut args,
        p,
        "dualFeasibilityTolerance",
        "--dual-feasibility-tolerance",
    )?;
    push_optional_number_arg(
        &mut args,
        p,
        "integerFeasibilityTolerance",
        "--integer-feasibility-tolerance",
    )?;
    push_optional_string_arg(&mut args, p, "lpAlgorithm", "--lp-algorithm");
    push_optional_number_arg(&mut args, p, "threads", "--threads")?;
    push_optional_number_arg(&mut args, p, "randomSeed", "--random-seed")?;
    push_optional_string_arg(&mut args, p, "presolve", "--presolve");
    push_optional_string_arg(&mut args, p, "cuts", "--cuts");
    push_optional_string_arg(&mut args, p, "heuristics", "--heuristics");
    push_optional_string_arg(&mut args, p, "branchRule", "--branch-rule");
    push_optional_string_arg(&mut args, p, "branchPriorities", "--branch-priorities");
    push_optional_string_arg(&mut args, p, "nodeSelection", "--node-selection");
    push_optional_string_arg(&mut args, p, "mipStart", "--mip-start");
    push_optional_string_arg(&mut args, p, "commandPath", "--command-path");
    Ok(args)
}

fn build_scheduling(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    Ok(vec![
        "--solver".into(),
        string_param(p.get("solver"), "auto"),
        "--kind".into(),
        string_param(p.get("kind"), "auto"),
    ])
}

fn build_knapsack(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_bin_packing(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_facility_location(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_graph_coloring(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_set_cover(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_tsp(p: &ExternalModuleParams, _: &ExternalModuleContext) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_max_flow(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_min_cost_flow(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_minimum_spanning_tree(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_routing(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_weighted_independent_set(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_weighted_max_sat(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    build_stdin_solver(p)
}

fn build_validation_tool(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    Ok(vec!["--tool".into(), string_param(p.get("tool"), "auto")])
}

fn build_simulation_validation(
    p: &ExternalModuleParams,
    _: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    Ok(vec![
        "--engine".into(),
        string_param(p.get("engine"), "auto"),
    ])
}

fn build_traffic_simpy(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "traffic-simpy-reference.json"));
    Ok(vec![
        "--problem".into(),
        string_param(p.get("problem"), ""),
        "--out".into(),
        out,
    ])
}

fn build_traffic_ciw(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "traffic-ciw-reference.json"));
    Ok(vec![
        "--problem".into(),
        string_param(p.get("problem"), ""),
        "--out".into(),
        out,
    ])
}

fn build_traffic_fel(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "traffic-fel-reference.json"));
    Ok(vec![
        "--problem".into(),
        string_param(p.get("problem"), ""),
        "--out".into(),
        out,
    ])
}

fn build_traffic_sumo(
    p: &ExternalModuleParams,
    ctx: &ExternalModuleContext,
) -> Result<Vec<String>, String> {
    let out = string_param(p.get("out"), &join_out(ctx, "sumo-reference.json"));
    let mut args = vec![
        "--problem".to_string(),
        string_param(p.get("problem"), ""),
        "--out".to_string(),
        out,
    ];
    if p.get("workdir").is_some() {
        args.push("--workdir".into());
        args.push(string_param(p.get("workdir"), ""));
    }
    if p.get("sumoBin").is_some() {
        args.push("--sumo-bin".into());
        args.push(string_param(p.get("sumoBin"), ""));
    }
    if p.get("netconvertBin").is_some() {
        args.push("--netconvert-bin".into());
        args.push(string_param(p.get("netconvertBin"), ""));
    }
    if p.get("collisionAction").is_some() {
        args.push("--collision-action".into());
        args.push(string_param(p.get("collisionAction"), "warn"));
    }
    Ok(args)
}

/// `registerBuiltInExternalModules()` — idempotent via [`OnceLock`].
///
/// Registers the built-in module metadata. The result is memoised so repeated
/// calls are cheap and stable. Optional source script existence is checked when
/// a module is run, which lets callers list/skip unavailable externals cleanly.
pub fn register_built_in_external_modules() -> Result<(), String> {
    static RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    RESULT.get_or_init(do_register).clone()
}

fn do_register() -> Result<(), String> {
    register_external_module(ExternalProgramModule {
        id: NEURAL_NETWORK_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Reference,
        description: "Rust source-only reference for neural XOR, corridor value iteration, and neural ODE decay.".to_string(),
        source_path: "src/bin/neural_network_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[
            ("seed", n(7.0)),
            ("xorEpochs", n(8000.0)),
            ("xorLr", n(0.3)),
            ("corridorLength", n(6.0)),
            ("corridorGamma", n(0.95)),
            ("odeRate", n(0.5)),
            ("odeY0", n(1.0)),
            ("odeT1", n(2.0)),
            ("odeDt", n(0.05)),
        ]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_neural_network,
    })?;

    register_external_module(ExternalProgramModule {
        id: COMPUTER_NETWORK_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Rust source-only reference simulator for computer-network topology, queueing, drops, and bottleneck metrics.".to_string(),
        source_path: "src/bin/computer_network_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("builtin", s("bottleneck-lab"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_computer_network,
    })?;

    register_external_module(ExternalProgramModule {
        id: COMPUTER_NETWORK_FEL_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Rust source-only FEL-style packet-network reference; consumes the same computer-network JSON model spec as the internal registry.".to_string(),
        source_path: "src/bin/computer_network_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("builtin", s("bottleneck-lab"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_computer_network_fel,
    })?;

    register_external_module(ExternalProgramModule {
        id: IP_MIP_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only external IP/MIP reference: bounded enumeration for small integer models, with external-solver aliases mapped to the same CLI contract.".to_string(),
        source_path: "src/bin/ip_mip_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto")), ("maxEnumerations", n(1_000_000.0))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_ip_mip,
    })?;

    register_external_module(ExternalProgramModule {
        id: CP_SAT_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only CP-SAT reference for small finite-domain models, optional solution enumeration, and assumption-core checks.".to_string(),
        source_path: "src/bin/cp_sat_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_cp_sat,
    })?;

    register_external_module(ExternalProgramModule {
        id: STOCHASTIC_LP_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only stochastic LP reference for scenario-expanded linear models and deterministic-equivalent cross-checks.".to_string(),
        source_path: "src/bin/stochastic_lp_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_assignment,
    })?;

    register_external_module(ExternalProgramModule {
        id: QP_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only quadratic-program reference for small convex and bounded-enumeration QP payloads.".to_string(),
        source_path: "src/bin/qp_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_qp,
    })?;

    register_external_module(ExternalProgramModule {
        id: NONLINEAR_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only nonlinear optimization reference for bounded fallback search and curve-fit style models.".to_string(),
        source_path: "src/bin/nonlinear_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_nonlinear,
    })?;

    register_external_module(ExternalProgramModule {
        id: NONLINEAR_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only nonlinear validation reference for comparing nonlinear model solutions across solver adapters.".to_string(),
        source_path: "src/bin/nonlinear_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_assignment,
    })?;

    register_external_module(ExternalProgramModule {
        id: LINEAR_CLI_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only LP/MIP command adapter for open and commercial solver CLIs discovered at runtime.".to_string(),
        source_path: "src/bin/linear_cli_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("kind", s("lp")), ("solver", s("highs"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_linear_cli,
    })?;

    register_external_module(ExternalProgramModule {
        id: LP_SOLVE_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only LP reference with crate-native simplex and interior-point method options.".to_string(),
        source_path: "src/bin/lp_solve_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("method", s("rust"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_lp_solve,
    })?;

    register_external_module(ExternalProgramModule {
        id: OPTIMIZATION_ECOSYSTEM_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Rust source-only optimization ecosystem reference for CP, planning, multiobjective, nonlinear, and solver CLI smoke payloads.".to_string(),
        source_path: "src/bin/optimization_ecosystem_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("tool", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_optimization_ecosystem,
    })?;

    register_external_module(ExternalProgramModule {
        id: ASSIGNMENT_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only assignment reference for JSON stdin cost matrices; exact DP is used by default for small models.".to_string(),
        source_path: "src/bin/assignment_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_assignment,
    })?;

    register_external_module(ExternalProgramModule {
        id: KNAPSACK_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only 0/1 knapsack reference for JSON stdin item weights and values."
                .to_string(),
        source_path: "src/bin/knapsack_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_knapsack,
    })?;

    register_external_module(ExternalProgramModule {
        id: BIN_PACKING_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only bin-packing reference for JSON stdin capacities and item weights."
                .to_string(),
        source_path: "src/bin/bin_packing_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_bin_packing,
    })?;

    register_external_module(ExternalProgramModule {
        id: FACILITY_LOCATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only facility-location reference for JSON stdin fixed costs, assignments, and capacities.".to_string(),
        source_path: "src/bin/facility_location_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_facility_location,
    })?;

    register_external_module(ExternalProgramModule {
        id: GRAPH_COLORING_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only graph-coloring reference for JSON stdin graph and color-count checks."
                .to_string(),
        source_path: "src/bin/graph_coloring_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_graph_coloring,
    })?;

    register_external_module(ExternalProgramModule {
        id: SET_COVER_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only set-cover reference for JSON stdin universe and subset models."
                .to_string(),
        source_path: "src/bin/set_cover_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_set_cover,
    })?;

    register_external_module(ExternalProgramModule {
        id: TSP_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only TSP reference for JSON stdin distance matrices and small exact Held-Karp checks.".to_string(),
        source_path: "src/bin/tsp_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_tsp,
    })?;

    register_external_module(ExternalProgramModule {
        id: MAX_FLOW_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only max-flow reference for JSON stdin directed capacitated networks."
                .to_string(),
        source_path: "src/bin/max_flow_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_max_flow,
    })?;

    register_external_module(ExternalProgramModule {
        id: MIN_COST_FLOW_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only min-cost-flow reference for JSON stdin supply, demand, capacity, and cost networks.".to_string(),
        source_path: "src/bin/min_cost_flow_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_min_cost_flow,
    })?;

    register_external_module(ExternalProgramModule {
        id: MINIMUM_SPANNING_TREE_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only minimum-spanning-tree reference for JSON stdin weighted graph models."
                .to_string(),
        source_path: "src/bin/minimum_spanning_tree_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_minimum_spanning_tree,
    })?;

    register_external_module(ExternalProgramModule {
        id: ROUTING_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only routing reference for JSON stdin distance, depot, and vehicle models."
                .to_string(),
        source_path: "src/bin/routing_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_routing,
    })?;

    register_external_module(ExternalProgramModule {
        id: SCHEDULING_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only scheduling reference for JSON stdin job-shop and flow-shop models."
                .to_string(),
        source_path: "src/bin/scheduling_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto")), ("kind", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_scheduling,
    })?;

    register_external_module(ExternalProgramModule {
        id: WEIGHTED_INDEPENDENT_SET_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Rust source-only weighted-independent-set reference for JSON stdin graph and weight models.".to_string(),
        source_path: "src/bin/weighted_independent_set_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_weighted_independent_set,
    })?;

    register_external_module(ExternalProgramModule {
        id: WEIGHTED_MAX_SAT_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description:
            "Rust source-only weighted Max-SAT reference for JSON stdin weighted clause models."
                .to_string(),
        source_path: "src/bin/weighted_max_sat_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("solver", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_weighted_max_sat,
    })?;

    register_external_module(ExternalProgramModule {
        id: OUTPUT_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description:
            "Rust source-only output validator for JSON schema, table schema, and profile payloads."
                .to_string(),
        source_path: "src/bin/output_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("tool", s("json-schema"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_validation_tool,
    })?;

    register_external_module(ExternalProgramModule {
        id: MODEL_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description:
            "Rust source-only model validator for SAT, Max-SAT, PB, SMT-LIB, and MiniZinc payloads."
                .to_string(),
        source_path: "src/bin/model_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("tool", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_validation_tool,
    })?;

    register_external_module(ExternalProgramModule {
        id: PROOF_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description:
            "Rust source-only proof validator for small DRAT, LRAT, and VeriPB-style artifacts."
                .to_string(),
        source_path: "src/bin/proof_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("tool", s("drat"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_validation_tool,
    })?;

    register_external_module(ExternalProgramModule {
        id: FORMAL_BENCHMARK_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description:
            "Rust source-only formal benchmark validator for TLA, security protocol, and manifest payloads."
                .to_string(),
        source_path: "src/bin/formal_benchmark_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("tool", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_validation_tool,
    })?;

    register_external_module(ExternalProgramModule {
        id: SIMULATION_VALIDATION_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description:
            "Rust source-only simulation validator for event-network, mobility, queueing, and agent-model payloads."
                .to_string(),
        source_path: "src/bin/simulation_validation_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: params(&[("engine", s("auto"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_simulation_validation,
    })?;

    register_external_module(ExternalProgramModule {
        id: TRAFFIC_SIMPY_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Optional SimPy process-oriented traffic FEL reference for shared source/sink scheduled trips.".to_string(),
        source_path: "external-references/traffic/simpy_traffic_reference.py".to_string(),
        interpreter: python3(),
        default_params: ExternalModuleParams::new(),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_traffic_simpy,
    })?;

    register_external_module(ExternalProgramModule {
        id: TRAFFIC_CIW_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Optional Ciw queueing-network traffic FEL reference for shared source/sink scheduled trips.".to_string(),
        source_path: "external-references/traffic/ciw_traffic_reference.py".to_string(),
        interpreter: python3(),
        default_params: ExternalModuleParams::new(),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_traffic_ciw,
    })?;

    register_external_module(ExternalProgramModule {
        id: TRAFFIC_FEL_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Rust source-only traffic FEL reference for shared source/sink scheduled trips using the crate-native smart traffic simulator.".to_string(),
        source_path: "src/bin/traffic_fel_reference.rs".to_string(),
        interpreter: rust_cargo(),
        default_params: ExternalModuleParams::new(),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_traffic_fel,
    })?;

    register_external_module(ExternalProgramModule {
        id: TRAFFIC_SUMO_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Optional SUMO black-box traffic simulator cross-check; calls SUMO/netconvert from PATH or SUMO_BIN without vendoring binaries.".to_string(),
        source_path: "external-references/traffic/sumo_traffic_reference.py".to_string(),
        interpreter: python3(),
        default_params: ExternalModuleParams::new(),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_traffic_sumo,
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::external_program::{get_external_module, list_external_modules};
    use super::*;

    fn is_optional_python_traffic_module(module_id: &str) -> bool {
        matches!(
            module_id,
            TRAFFIC_CIW_REFERENCE_ID | TRAFFIC_SIMPY_REFERENCE_ID | TRAFFIC_SUMO_REFERENCE_ID
        )
    }

    #[test]
    fn built_in_reference_modules_use_rust_sources_where_available() {
        register_built_in_external_modules().expect("register built-in modules");
        for (module_id, source_path) in [
            (
                NEURAL_NETWORK_REFERENCE_ID,
                "src/bin/neural_network_reference.rs",
            ),
            (
                COMPUTER_NETWORK_REFERENCE_ID,
                "src/bin/computer_network_reference.rs",
            ),
            (
                COMPUTER_NETWORK_FEL_REFERENCE_ID,
                "src/bin/computer_network_reference.rs",
            ),
            (IP_MIP_REFERENCE_ID, "src/bin/ip_mip_reference.rs"),
            (CP_SAT_REFERENCE_ID, "src/bin/cp_sat_reference.rs"),
            (
                STOCHASTIC_LP_REFERENCE_ID,
                "src/bin/stochastic_lp_reference.rs",
            ),
            (QP_REFERENCE_ID, "src/bin/qp_reference.rs"),
            (NONLINEAR_REFERENCE_ID, "src/bin/nonlinear_reference.rs"),
            (
                NONLINEAR_VALIDATION_REFERENCE_ID,
                "src/bin/nonlinear_validation_reference.rs",
            ),
            (LINEAR_CLI_REFERENCE_ID, "src/bin/linear_cli_reference.rs"),
            (LP_SOLVE_REFERENCE_ID, "src/bin/lp_solve_reference.rs"),
            (
                OPTIMIZATION_ECOSYSTEM_REFERENCE_ID,
                "src/bin/optimization_ecosystem_reference.rs",
            ),
            (ASSIGNMENT_REFERENCE_ID, "src/bin/assignment_reference.rs"),
            (KNAPSACK_REFERENCE_ID, "src/bin/knapsack_reference.rs"),
            (BIN_PACKING_REFERENCE_ID, "src/bin/bin_packing_reference.rs"),
            (
                FACILITY_LOCATION_REFERENCE_ID,
                "src/bin/facility_location_reference.rs",
            ),
            (
                GRAPH_COLORING_REFERENCE_ID,
                "src/bin/graph_coloring_reference.rs",
            ),
            (SET_COVER_REFERENCE_ID, "src/bin/set_cover_reference.rs"),
            (TSP_REFERENCE_ID, "src/bin/tsp_reference.rs"),
            (MAX_FLOW_REFERENCE_ID, "src/bin/max_flow_reference.rs"),
            (
                MIN_COST_FLOW_REFERENCE_ID,
                "src/bin/min_cost_flow_reference.rs",
            ),
            (
                MINIMUM_SPANNING_TREE_REFERENCE_ID,
                "src/bin/minimum_spanning_tree_reference.rs",
            ),
            (ROUTING_REFERENCE_ID, "src/bin/routing_reference.rs"),
            (SCHEDULING_REFERENCE_ID, "src/bin/scheduling_reference.rs"),
            (
                WEIGHTED_INDEPENDENT_SET_REFERENCE_ID,
                "src/bin/weighted_independent_set_reference.rs",
            ),
            (
                WEIGHTED_MAX_SAT_REFERENCE_ID,
                "src/bin/weighted_max_sat_reference.rs",
            ),
            (
                OUTPUT_VALIDATION_REFERENCE_ID,
                "src/bin/output_validation_reference.rs",
            ),
            (
                MODEL_VALIDATION_REFERENCE_ID,
                "src/bin/model_validation_reference.rs",
            ),
            (
                PROOF_VALIDATION_REFERENCE_ID,
                "src/bin/proof_validation_reference.rs",
            ),
            (
                FORMAL_BENCHMARK_VALIDATION_REFERENCE_ID,
                "src/bin/formal_benchmark_validation_reference.rs",
            ),
            (
                SIMULATION_VALIDATION_REFERENCE_ID,
                "src/bin/simulation_validation_reference.rs",
            ),
            (TRAFFIC_FEL_REFERENCE_ID, "src/bin/traffic_fel_reference.rs"),
        ] {
            let module = get_external_module(module_id).expect("registered module");

            assert_eq!(module.source_path, source_path);
            assert_eq!(module.interpreter.env_var, "CARGO");
            assert_eq!(module.interpreter.default_command, "cargo");
            assert!(
                !module.description.to_ascii_lowercase().contains("python"),
                "{module_id} description should advertise the Rust reference path"
            );
        }
    }

    #[test]
    fn non_optional_built_in_modules_are_rust_cargo_sources() {
        register_built_in_external_modules().expect("register built-in modules");

        for module in list_external_modules() {
            if is_optional_python_traffic_module(&module.id) {
                continue;
            }

            assert_eq!(
                module.interpreter.env_var, "CARGO",
                "{} must use Cargo as the default interpreter",
                module.id
            );
            assert_eq!(
                module.interpreter.default_command, "cargo",
                "{} must default to the Cargo binary",
                module.id
            );
            assert!(
                module.source_path.starts_with("src/bin/") && module.source_path.ends_with(".rs"),
                "{} must use a Rust src/bin source, got {}",
                module.id,
                module.source_path
            );
            assert!(
                module
                    .description
                    .to_ascii_lowercase()
                    .contains("rust source-only"),
                "{} must describe itself as Rust source-only",
                module.id
            );
        }
    }

    #[test]
    fn non_optional_built_in_defaults_do_not_select_python_bridges() {
        register_built_in_external_modules().expect("register built-in modules");
        let pythonish_defaults = ["python", "scipy", "pyomo", "cvxpy"];

        for module in list_external_modules() {
            if is_optional_python_traffic_module(&module.id) {
                continue;
            }
            for (key, value) in module.default_params.iter() {
                let ParamValue::Str(text) = value else {
                    continue;
                };
                let normalized = text.to_ascii_lowercase();
                assert!(
                    !pythonish_defaults
                        .iter()
                        .any(|needle| normalized.contains(needle)),
                    "{} default param {}={:?} must not select a Python bridge",
                    module.id,
                    key,
                    text
                );
            }
        }
    }

    #[test]
    fn only_optional_traffic_modules_use_python_sources() {
        register_built_in_external_modules().expect("register built-in modules");
        let mut python_modules = list_external_modules()
            .into_iter()
            .filter(|module| {
                module.interpreter.env_var == "PYTHON_BIN" || module.source_path.ends_with(".py")
            })
            .map(|module| {
                assert!(
                    module.description.to_ascii_lowercase().contains("optional"),
                    "{} must advertise Python-backed modules as optional",
                    module.id
                );
                assert!(
                    module
                        .source_path
                        .starts_with("external-references/traffic/"),
                    "{} must keep Python-backed sources isolated to optional traffic references",
                    module.id
                );
                module.id
            })
            .collect::<Vec<_>>();
        python_modules.sort();

        let mut expected_python_modules = [
            TRAFFIC_CIW_REFERENCE_ID,
            TRAFFIC_SIMPY_REFERENCE_ID,
            TRAFFIC_SUMO_REFERENCE_ID,
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        expected_python_modules.sort();

        assert_eq!(python_modules, expected_python_modules);
    }

    #[test]
    fn new_solver_reference_builders_pass_runtime_controls() {
        register_built_in_external_modules().expect("register built-in modules");
        let ctx = ExternalModuleContext {
            root: "/repo".into(),
            out_root: "/repo/out".into(),
            module_out_dir: "/repo/out/external/test".into(),
        };

        let cp_sat = get_external_module(CP_SAT_REFERENCE_ID).expect("registered CP-SAT module");
        let cp_sat_args = (cp_sat.build_args)(
            &params(&[
                ("solver", s("rust-enumeration")),
                ("enumerateSolutions", n(2.0)),
                ("assumptionCore", ParamValue::Bool(true)),
            ]),
            &ctx,
        )
        .expect("CP-SAT args");
        assert_eq!(
            cp_sat_args,
            vec![
                "--solver",
                "rust-enumeration",
                "--enumerate-solutions",
                "2",
                "--assumption-core"
            ]
        );

        let linear_cli =
            get_external_module(LINEAR_CLI_REFERENCE_ID).expect("registered linear CLI module");
        let linear_args = (linear_cli.build_args)(
            &params(&[
                ("kind", s("mip")),
                ("solver", s("cbc")),
                ("modelFormat", s("mps")),
                ("timeLimit", n(3.5)),
                ("relativeGap", n(0.01)),
                ("threads", n(4.0)),
                ("presolve", s("on")),
                ("branchPriorities", s("[0,3,1]")),
                ("mipStart", s("[1.0,0.0,1.0]")),
                ("commandPath", s("/opt/homebrew/bin/cbc")),
            ]),
            &ctx,
        )
        .expect("linear CLI args");
        assert_eq!(
            linear_args,
            vec![
                "--kind",
                "mip",
                "--solver",
                "cbc",
                "--model-format",
                "mps",
                "--time-limit",
                "3.5",
                "--relative-gap",
                "0.01",
                "--threads",
                "4",
                "--presolve",
                "on",
                "--branch-priorities",
                "[0,3,1]",
                "--mip-start",
                "[1.0,0.0,1.0]",
                "--command-path",
                "/opt/homebrew/bin/cbc"
            ]
        );

        let ecosystem = get_external_module(OPTIMIZATION_ECOSYSTEM_REFERENCE_ID)
            .expect("registered optimization ecosystem module");
        let ecosystem_args = (ecosystem.build_args)(&params(&[("tool", s("choco-solver"))]), &ctx)
            .expect("optimization ecosystem args");
        assert_eq!(ecosystem_args, vec!["--tool", "choco-solver"]);
    }
}
