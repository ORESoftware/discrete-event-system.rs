//! Port of `src/des/runners/external-program.ts`.
//!
//! Sanctioned external-program invocation helpers and a module registry for
//! validators / reference solvers. Rules (unchanged from TS):
//!
//!   * source scripts must live under `external-references/`; Rust reference
//!     sources may live under `src/bin/` and are invoked through Cargo
//!   * no shell is used; arguments are passed as an argv array
//!   * the interpreter is explicit (env-var override, stable default)
//!   * stdout/stderr are captured for diagnostics
//!   * binaries/interpreters are NEVER vendored; only source scripts + metadata
//!
//! ## PORT NOTE
//!
//!   * `spawnSync(cmd, args, {shell:false, encoding:'utf8', ...})` →
//!     [`std::process::Command`] with a Rust polling timeout. `maxBuffer` is
//!     still recorded for parity, while `timeout` and `EXTERNAL_TIMEOUT_MS` are
//!     enforced by killing long-running children.
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
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

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

fn relative_source_has_parent_or_root(relative_source: &str) -> bool {
    Path::new(relative_source)
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
}

fn rust_cargo_bin_name(relative_source: &str) -> Option<String> {
    let path = Path::new(relative_source);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(part)) if part == "src") {
        return None;
    }
    if !matches!(components.next(), Some(Component::Normal(part)) if part == "bin") {
        return None;
    }
    if components.clone().count() != 1 {
        return None;
    }
    if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return None;
    }
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(str::to_string)
}

/// `resolveExternalScript(root, relativeScript)`.
fn validate_external_script_location(
    root: &Path,
    relative_script: &str,
) -> Result<PathBuf, String> {
    if relative_source_has_parent_or_root(relative_script) {
        return Err(format!(
            "external source must be a repo-relative path without `..`: {relative_script}"
        ));
    }
    let external_root = root.join("external-references");
    let rust_bin_root = root.join("src").join("bin");
    let script = root.join(relative_script);
    if !script.starts_with(&external_root) && !script.starts_with(&rust_bin_root) {
        return Err(format!(
            "external source must live under {} or {}: {}",
            external_root.display(),
            rust_bin_root.display(),
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

fn external_module_timeout_env_name(id: &str) -> String {
    let mut name = String::from("EXTERNAL_MODULE_");
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() {
            name.push(ch.to_ascii_uppercase());
        } else {
            name.push('_');
        }
    }
    name.push_str("_TIMEOUT_MS");
    name
}

fn parse_external_timeout_ms(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

fn resolve_external_module_timeout_ms_with_lookup<F>(
    module: &ExternalProgramModule,
    mut lookup: F,
) -> Option<u64>
where
    F: FnMut(&str) -> Option<String>,
{
    if module.timeout_ms.is_some() {
        return module.timeout_ms;
    }

    let module_env_name = external_module_timeout_env_name(&module.id);
    if let Some(timeout_ms) =
        lookup(&module_env_name).and_then(|value| parse_external_timeout_ms(&value))
    {
        return Some(timeout_ms);
    }

    lookup("EXTERNAL_MODULE_TIMEOUT_MS").and_then(|value| parse_external_timeout_ms(&value))
}

fn resolve_external_module_timeout_ms(module: &ExternalProgramModule) -> Option<u64> {
    resolve_external_module_timeout_ms_with_lookup(module, |name| std::env::var(name).ok())
}

fn external_module_invocation_args(
    module: &ExternalProgramModule,
    source: &Path,
    module_args: Vec<String>,
) -> Vec<String> {
    let mut args = if let Some(bin_name) = rust_cargo_bin_name(&module.source_path) {
        vec![
            "run".to_string(),
            "--quiet".to_string(),
            "--bin".to_string(),
            bin_name,
            "--".to_string(),
        ]
    } else {
        vec![source.display().to_string()]
    };
    args.extend(module_args);
    args
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
    let timeout_ms = opts.timeout_ms.unwrap_or_else(|| {
        std::env::var("EXTERNAL_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120_000)
    });
    let _max_buffer = opts.max_buffer_bytes.unwrap_or(10 * 1024 * 1024);
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut child = Command::new(command)
        .args(args)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run {command}: {e}"))?;

    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => return Err(format!("failed to poll {command}: {err}")),
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for {command}: {e}"))?;
    let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if timed_out {
        let timeout_message = format!("external program timed out after {timeout_ms}ms");
        if stderr.trim().is_empty() {
            stderr = timeout_message;
        } else {
            stderr = format!("{timeout_message}\n{stderr}");
        }
    }

    Ok(ExternalProgramResult {
        command: command.to_string(),
        args: args.to_vec(),
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr,
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
    let args = external_module_invocation_args(&module, &script, (module.build_args)(&merged, &ctx)?);

    run_external_program(
        &command,
        &args,
        &RunExternalOpts {
            cwd: Some(root),
            timeout_ms: resolve_external_module_timeout_ms(&module),
            max_buffer_bytes: module.max_buffer_bytes,
            module_id: Some(id.to_string()),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_build_args(
        _: &ExternalModuleParams,
        _: &ExternalModuleContext,
    ) -> Result<Vec<String>, String> {
        Ok(Vec::new())
    }

    fn timeout_test_module(id: &str, timeout_ms: Option<u64>) -> ExternalProgramModule {
        ExternalProgramModule {
            id: id.to_string(),
            kind: ExternalModuleKind::Reference,
            description: "timeout test module".to_string(),
            source_path: "external-references/timeout-test.py".to_string(),
            interpreter: ExternalInterpreterSpec {
                env_var: "TIMEOUT_TEST_INTERPRETER".to_string(),
                default_command: "python3".to_string(),
                label: "Python".to_string(),
            },
            default_params: ExternalModuleParams::new(),
            timeout_ms,
            max_buffer_bytes: None,
            build_args: empty_build_args,
        }
    }

    #[test]
    fn external_source_location_allows_external_references_and_rust_bins() {
        let root = PathBuf::from("/repo");

        assert_eq!(
            validate_external_script_location(&root, "external-references/foo/reference.py")
                .expect("external reference path"),
            root.join("external-references/foo/reference.py")
        );
        assert_eq!(
            validate_external_script_location(&root, "src/bin/ip_mip_reference.rs")
                .expect("rust bin reference path"),
            root.join("src/bin/ip_mip_reference.rs")
        );
        assert!(
            validate_external_script_location(&root, "src/lib.rs").is_err(),
            "library sources should not be registered as external modules"
        );
        assert!(
            validate_external_script_location(
                &root,
                "external-references/../src/bin/ip_mip_reference.rs"
            )
            .is_err(),
            "path traversal should not bypass source roots"
        );
    }

    #[test]
    fn rust_bin_external_module_invocation_uses_cargo_run() {
        let mut module = timeout_test_module("ip-mip-reference", None);
        module.source_path = "src/bin/ip_mip_reference.rs".to_string();
        module.interpreter = ExternalInterpreterSpec {
            env_var: "CARGO".to_string(),
            default_command: "cargo".to_string(),
            label: "Cargo/Rust".to_string(),
        };

        let args = external_module_invocation_args(
            &module,
            Path::new("/repo/src/bin/ip_mip_reference.rs"),
            vec!["--solver".to_string(), "auto".to_string()],
        );

        assert_eq!(
            args,
            vec![
                "run",
                "--quiet",
                "--bin",
                "ip_mip_reference",
                "--",
                "--solver",
                "auto"
            ]
        );
    }

    #[test]
    fn script_external_module_invocation_keeps_source_as_first_arg() {
        let module = timeout_test_module("timeout-test", None);

        let args = external_module_invocation_args(
            &module,
            Path::new("/repo/external-references/timeout-test.py"),
            vec!["--flag".to_string()],
        );

        assert_eq!(
            args,
            vec!["/repo/external-references/timeout-test.py", "--flag"]
        );
    }

    #[test]
    fn module_id_validation() {
        assert!(valid_module_id("neural-network-reference"));
        assert!(valid_module_id("traffic.fel_v2"));
        assert!(!valid_module_id("-bad"));
        assert!(!valid_module_id("Bad"));
        assert!(!valid_module_id("has space"));
    }

    #[test]
    fn external_module_timeout_env_name_sanitizes_module_id() {
        assert_eq!(
            external_module_timeout_env_name("traffic-simpy-reference"),
            "EXTERNAL_MODULE_TRAFFIC_SIMPY_REFERENCE_TIMEOUT_MS"
        );
        assert_eq!(
            external_module_timeout_env_name("solver.foo_v2"),
            "EXTERNAL_MODULE_SOLVER_FOO_V2_TIMEOUT_MS"
        );
    }

    #[test]
    fn external_module_timeout_resolution_uses_metadata_then_envs() {
        let module = timeout_test_module("neural-network-reference", Some(55));
        assert_eq!(
            resolve_external_module_timeout_ms_with_lookup(&module, |_| Some("99".to_string())),
            Some(55)
        );

        let module = timeout_test_module("neural-network-reference", None);
        assert_eq!(
            resolve_external_module_timeout_ms_with_lookup(&module, |name| match name {
                "EXTERNAL_MODULE_NEURAL_NETWORK_REFERENCE_TIMEOUT_MS" => Some("77".to_string()),
                "EXTERNAL_MODULE_TIMEOUT_MS" => Some("99".to_string()),
                _ => None,
            }),
            Some(77)
        );

        assert_eq!(
            resolve_external_module_timeout_ms_with_lookup(&module, |name| match name {
                "EXTERNAL_MODULE_TIMEOUT_MS" => Some("99".to_string()),
                _ => None,
            }),
            Some(99)
        );

        assert_eq!(
            resolve_external_module_timeout_ms_with_lookup(&module, |name| match name {
                "EXTERNAL_MODULE_NEURAL_NETWORK_REFERENCE_TIMEOUT_MS" =>
                    Some("not-a-number".to_string()),
                "EXTERNAL_MODULE_TIMEOUT_MS" => Some("99".to_string()),
                _ => None,
            }),
            Some(99)
        );

        assert_eq!(
            resolve_external_module_timeout_ms_with_lookup(&module, |_| None),
            None
        );
    }

    #[test]
    fn run_external_program_enforces_timeout() {
        let result = run_external_program(
            "sleep",
            &["1".to_string()],
            &RunExternalOpts {
                timeout_ms: Some(10),
                ..RunExternalOpts::default()
            },
        )
        .expect("timeout result");

        assert_ne!(result.status, Some(0));
        assert!(result.stderr.contains("timed out after 10ms"));
    }
}
