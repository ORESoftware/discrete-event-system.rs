//! Port of `src/des/runners/external-modules.ts`.
//!
//! Metadata registry of the built-in external reference modules. This file is
//! metadata only: it registers source scripts that live under
//! `external-references/` and describes how to invoke them.
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
//!     startup. Missing optional source scripts are left unregistered so
//!     validators can skip only the affected external engine.

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
/// Missing optional source scripts under `external-references/` are skipped;
/// malformed module definitions still return an error. The result is memoised
/// so repeated calls are cheap and stable.
pub fn register_built_in_external_modules() -> Result<(), String> {
    static RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    RESULT.get_or_init(do_register).clone()
}

fn register_available_external_module(module: ExternalProgramModule) -> Result<(), String> {
    match register_external_module(module) {
        Ok(()) => Ok(()),
        Err(e) if e.contains("external script not found:") => Ok(()),
        Err(e) if e.contains("already registered") => Ok(()),
        Err(e) => Err(e),
    }
}

fn do_register() -> Result<(), String> {
    register_available_external_module(ExternalProgramModule {
        id: NEURAL_NETWORK_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Reference,
        description: "Dependency-free Python reference for neural XOR, corridor value iteration, and neural ODE decay.".to_string(),
        source_path: "external-references/neural-network/nn_reference.py".to_string(),
        interpreter: python3(),
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

    register_available_external_module(ExternalProgramModule {
        id: COMPUTER_NETWORK_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Dependency-free Python reference simulator for computer-network topology, queueing, drops, and bottleneck metrics.".to_string(),
        source_path: "external-references/computer-network/network_reference.py".to_string(),
        interpreter: python3(),
        default_params: params(&[("builtin", s("bottleneck-lab"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_computer_network,
    })?;

    register_available_external_module(ExternalProgramModule {
        id: COMPUTER_NETWORK_FEL_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Dependency-free Python FEL-style packet-network reference; consumes the same computer-network JSON model spec as the internal registry.".to_string(),
        source_path: "external-references/computer-network/network_fel_reference.py".to_string(),
        interpreter: python3(),
        default_params: params(&[("builtin", s("bottleneck-lab"))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_computer_network_fel,
    })?;

    register_available_external_module(ExternalProgramModule {
        id: IP_MIP_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Solver,
        description: "Source-only external IP/MIP reference: Python brute force for bounded integer models, optional scipy.optimize.milp when installed.".to_string(),
        source_path: "external-references/ip-mip/ip_mip_reference.py".to_string(),
        interpreter: python3(),
        default_params: params(&[("solver", s("auto")), ("maxEnumerations", n(1_000_000.0))]),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_ip_mip,
    })?;

    register_available_external_module(ExternalProgramModule {
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

    register_available_external_module(ExternalProgramModule {
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

    register_available_external_module(ExternalProgramModule {
        id: TRAFFIC_FEL_REFERENCE_ID.to_string(),
        kind: ExternalModuleKind::Validator,
        description: "Dependency-free Python Future Event List traffic reference for model-spec traffic flows and shared source/sink scheduled trips.".to_string(),
        source_path: "external-references/traffic/fel_traffic_reference.py".to_string(),
        interpreter: python3(),
        default_params: ExternalModuleParams::new(),
        timeout_ms: None,
        max_buffer_bytes: None,
        build_args: build_traffic_fel,
    })?;

    register_available_external_module(ExternalProgramModule {
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
