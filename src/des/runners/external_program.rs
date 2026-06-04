//! Port of `src/des/runners/external-program.ts`.
//!
//! Sanctioned external-program invocation helpers and a module registry for
//! validators / reference solvers. Rules (unchanged from TS):
//!
//!   * source scripts must live under `external-references/`
//!   * no shell is used; arguments are passed as an argv array
//!   * the interpreter is explicit (env-var override, stable default)
//!   * stdout/stderr are captured for diagnostics
//!   * binaries/interpreters are NEVER vendored; only source scripts + metadata
//!
//! ## PORT NOTE
//!
//!   * `spawnSync(cmd, args, {shell:false, encoding:'utf8', ...})` →
//!     [`std::process::Command`]`::output()`. The TS `timeout`/`maxBuffer`
//!     options have **no std equivalent**, so they are recorded on the module
//!     but not enforced (would need `wait-timeout`/a reaper thread). The
//!     `EXTERNAL_TIMEOUT_MS` env var is still read for parity.
//!   * `status: number | null` → `Option<i32>` (`ExitStatus::code()`).
//!   * `fs.existsSync` / `path.resolve` → `Path::exists` / `PathBuf`.
//!   * `repoRootFromRunner()` uses `__dirname/../../..`; a compiled Rust binary
//!     has no meaningful source dir, so the repo root is the current working
//!     directory (overridable with `REPO_ROOT`).
//!   * the module-level `EXTERNAL_MODULES` map + `let registered` guard →
//!     a `OnceLock<Mutex<BTreeMap<..>>>` registry + `std::sync::Once`.
//!   * `buildArgs(params, ctx)` per-module closures → a `BuildArgsFn` function
//!     pointer (no captured state, so the module struct stays `Clone`).
//!   * `throw new Error(..)` (user errors) → `Result<_, String>`.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// `ExternalProgramResult`.
#[derive(Clone, Debug)]
pub struct ExternalProgramResult {
    pub command: String,
    pub args: Vec<String>,
    /// Exit code (`null` when killed by signal).
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub module_id: Option<String>,
}

/// `type ExternalModuleKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalModuleKind {
    Reference,
    Solver,
    Validator,
}

impl ExternalModuleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalModuleKind::Reference => "reference",
            ExternalModuleKind::Solver => "solver",
            ExternalModuleKind::Validator => "validator",
        }
    }
}

/// `type ExternalParamValue = string | number | boolean | undefined`.
/// `undefined` is modelled by the key being absent from the params map.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Str(String),
    Num(f64),
    Bool(bool),
}

/// `type ExternalModuleParams = Record<string, ExternalParamValue>`.
pub type ExternalModuleParams = HashMap<String, ParamValue>;

/// `ExternalInterpreterSpec`.
#[derive(Clone, Debug)]
pub struct ExternalInterpreterSpec {
    pub env_var: String,
    pub default_command: String,
    pub label: String,
}

/// `ExternalModuleContext`.
#[derive(Clone, Debug)]
pub struct ExternalModuleContext {
    pub root: PathBuf,
    pub out_root: PathBuf,
    pub module_out_dir: PathBuf,
}

/// `buildArgs(params, ctx) -> string[]` as a function pointer. Returns `Result`
/// because the TS `numberParam` can `throw` on a non-finite value.
pub type BuildArgsFn =
    fn(&ExternalModuleParams, &ExternalModuleContext) -> Result<Vec<String>, String>;

/// `ExternalProgramModule`.
#[derive(Clone)]
pub struct ExternalProgramModule {
    pub id: String,
    pub kind: ExternalModuleKind,
    pub description: String,
    pub source_path: String,
    pub interpreter: ExternalInterpreterSpec,
    /// Empty map means "no defaults" (TS `defaultParams?`).
    pub default_params: ExternalModuleParams,
    pub timeout_ms: Option<u64>,
    pub max_buffer_bytes: Option<usize>,
    pub build_args: BuildArgsFn,
}

fn registry() -> &'static Mutex<BTreeMap<String, ExternalProgramModule>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<String, ExternalProgramModule>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// `repoRootFromRunner()`.
pub fn repo_root_from_runner() -> PathBuf {
    if let Ok(root) = std::env::var("REPO_ROOT") {
        return PathBuf::from(root);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// `resolveExternalScript(root, relativeScript)`.
fn validate_external_script_location(
    root: &Path,
    relative_script: &str,
) -> Result<PathBuf, String> {
    let external_root = root.join("external-references");
    let script = root.join(relative_script);
    let prefix = format!("{}{}", external_root.display(), std::path::MAIN_SEPARATOR);
    if !script.display().to_string().starts_with(&prefix) {
        return Err(format!(
            "external script must live under {}: {}",
            external_root.display(),
            script.display()
        ));
    }
    Ok(script)
}

pub fn resolve_external_script(root: &Path, relative_script: &str) -> Result<PathBuf, String> {
    let script = validate_external_script_location(root, relative_script)?;
    if !script.exists() {
        return Err(format!("external script not found: {}", script.display()));
    }
    Ok(script)
}

fn valid_module_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

/// `registerExternalModule(module)`.
pub fn register_external_module(module: ExternalProgramModule) -> Result<(), String> {
    {
        let reg = registry().lock().unwrap();
        if reg.contains_key(&module.id) {
            return Err(format!(
                "external module \"{}\" already registered",
                module.id
            ));
        }
    }
    if !valid_module_id(&module.id) {
        return Err(format!("invalid external module id \"{}\"", module.id));
    }
    // Validate source paths at registration time, but leave optional
    // installation/existence checks to run time.
    validate_external_script_location(&repo_root_from_runner(), &module.source_path)?;
    registry().lock().unwrap().insert(module.id.clone(), module);
    Ok(())
}

/// `getExternalModule(id)`.
pub fn get_external_module(id: &str) -> Result<ExternalProgramModule, String> {
    let reg = registry().lock().unwrap();
    match reg.get(id) {
        Some(m) => Ok(m.clone()),
        None => {
            let ids: Vec<String> = reg.keys().cloned().collect();
            Err(format!(
                "unknown external module \"{id}\". Registered: [{}]",
                ids.join(", ")
            ))
        }
    }
}

/// `listExternalModules()` — sorted by id (BTreeMap iteration order).
pub fn list_external_modules() -> Vec<ExternalProgramModule> {
    registry().lock().unwrap().values().cloned().collect()
}

/// Options for [`run_external_program`] (the TS inline options object).
#[derive(Clone, Debug, Default)]
pub struct RunExternalOpts {
    pub cwd: Option<PathBuf>,
    pub timeout_ms: Option<u64>,
    pub max_buffer_bytes: Option<usize>,
    pub module_id: Option<String>,
}

/// `runExternalProgram(command, args, opts)`.
pub fn run_external_program(
    command: &str,
    args: &[String],
    opts: &RunExternalOpts,
) -> Result<ExternalProgramResult, String> {
    let cwd = opts.cwd.clone().unwrap_or_else(repo_root_from_runner);
    // PORT NOTE: timeout/maxBuffer are not enforced by std; read for parity.
    let _timeout_ms = opts.timeout_ms.unwrap_or_else(|| {
        std::env::var("EXTERNAL_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120_000)
    });
    let _max_buffer = opts.max_buffer_bytes.unwrap_or(10 * 1024 * 1024);

    let output = Command::new(command)
        .args(args)
        .current_dir(&cwd)
        .output()
        .map_err(|e| format!("failed to run {command}: {e}"))?;

    Ok(ExternalProgramResult {
        command: command.to_string(),
        args: args.to_vec(),
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        module_id: opts.module_id.clone(),
    })
}

/// `runExternalModule(id, params)`.
pub fn run_external_module(
    id: &str,
    params: &ExternalModuleParams,
) -> Result<ExternalProgramResult, String> {
    let module = get_external_module(id)?;
    let root = repo_root_from_runner();
    let script = resolve_external_script(&root, &module.source_path)?;
    let out_root = root.join("out").join("external");
    let trimmed = id.strip_suffix("-reference").unwrap_or(id);
    let module_out_dir = out_root.join(trimmed);

    // {...defaultParams, ...params}
    let mut merged: ExternalModuleParams = module.default_params.clone();
    for (k, v) in params {
        merged.insert(k.clone(), v.clone());
    }

    let command = std::env::var(&module.interpreter.env_var)
        .unwrap_or_else(|_| module.interpreter.default_command.clone());

    let ctx = ExternalModuleContext {
        root: root.clone(),
        out_root,
        module_out_dir,
    };
    let mut args: Vec<String> = vec![script.display().to_string()];
    args.extend((module.build_args)(&merged, &ctx)?);

    run_external_program(
        &command,
        &args,
        &RunExternalOpts {
            cwd: Some(root),
            timeout_ms: module.timeout_ms,
            max_buffer_bytes: module.max_buffer_bytes,
            module_id: Some(id.to_string()),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_id_validation() {
        assert!(valid_module_id("neural-network-reference"));
        assert!(valid_module_id("traffic.fel_v2"));
        assert!(!valid_module_id("-bad"));
        assert!(!valid_module_id("Bad"));
        assert!(!valid_module_id("has space"));
    }
}
