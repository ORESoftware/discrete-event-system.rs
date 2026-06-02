//! Optional probes for Java and Rust optimization ecosystems.
//!
//! Java CP/planning systems are usually consumed as jars on a classpath, while
//! Rust optimization libraries are usually compile-time crates or FFI bindings.
//! This module gives the crate a typed, non-vendored integration boundary for
//! both styles: callers point environment variables at local classpaths or
//! Cargo manifests, and probes report whether that integration is ready.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Broad ecosystem for an optional optimization integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationEcosystem {
    Java,
    Rust,
}

impl ExternalOptimizationEcosystem {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationEcosystem::Java => "java",
            ExternalOptimizationEcosystem::Rust => "rust",
        }
    }
}

/// Solver/modeling tool families known to the optional ecosystem bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationTool {
    ChocoSolver,
    JaCoP,
    IbmCpOptimizer,
    OptaPlanner,
    JMetal,
    MoeaFramework,
    Ecj,
    OjAlgo,
    OrToolsJava,
    GoodLp,
    LpModeler,
    RustLinprog,
    Argmin,
    NloptRs,
    HighsRust,
    ScipRust,
    CbcRust,
}

impl ExternalOptimizationTool {
    pub fn all() -> &'static [ExternalOptimizationTool] {
        &[
            ExternalOptimizationTool::ChocoSolver,
            ExternalOptimizationTool::JaCoP,
            ExternalOptimizationTool::IbmCpOptimizer,
            ExternalOptimizationTool::OptaPlanner,
            ExternalOptimizationTool::JMetal,
            ExternalOptimizationTool::MoeaFramework,
            ExternalOptimizationTool::Ecj,
            ExternalOptimizationTool::OjAlgo,
            ExternalOptimizationTool::OrToolsJava,
            ExternalOptimizationTool::GoodLp,
            ExternalOptimizationTool::LpModeler,
            ExternalOptimizationTool::RustLinprog,
            ExternalOptimizationTool::Argmin,
            ExternalOptimizationTool::NloptRs,
            ExternalOptimizationTool::HighsRust,
            ExternalOptimizationTool::ScipRust,
            ExternalOptimizationTool::CbcRust,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "choco-solver",
            ExternalOptimizationTool::JaCoP => "jacop",
            ExternalOptimizationTool::IbmCpOptimizer => "ibm-cp-optimizer",
            ExternalOptimizationTool::OptaPlanner => "optaplanner",
            ExternalOptimizationTool::JMetal => "jmetal",
            ExternalOptimizationTool::MoeaFramework => "moea-framework",
            ExternalOptimizationTool::Ecj => "ecj",
            ExternalOptimizationTool::OjAlgo => "ojalgo",
            ExternalOptimizationTool::OrToolsJava => "ortools-java",
            ExternalOptimizationTool::GoodLp => "good-lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
            ExternalOptimizationTool::HighsRust => "highs-rust",
            ExternalOptimizationTool::ScipRust => "scip-rust",
            ExternalOptimizationTool::CbcRust => "cbc-rust",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "Choco Solver",
            ExternalOptimizationTool::JaCoP => "JaCoP",
            ExternalOptimizationTool::IbmCpOptimizer => "IBM ILOG CP Optimizer",
            ExternalOptimizationTool::OptaPlanner => "OptaPlanner",
            ExternalOptimizationTool::JMetal => "jMetal",
            ExternalOptimizationTool::MoeaFramework => "MOEA Framework",
            ExternalOptimizationTool::Ecj => "ECJ",
            ExternalOptimizationTool::OjAlgo => "ojAlgo",
            ExternalOptimizationTool::OrToolsJava => "Google OR-Tools Java",
            ExternalOptimizationTool::GoodLp => "good_lp",
            ExternalOptimizationTool::LpModeler => "lp-modeler",
            ExternalOptimizationTool::RustLinprog => "rust-linprog",
            ExternalOptimizationTool::Argmin => "argmin",
            ExternalOptimizationTool::NloptRs => "nlopt-rs",
            ExternalOptimizationTool::HighsRust => "HiGHS Rust bindings",
            ExternalOptimizationTool::ScipRust => "SCIP Rust bindings",
            ExternalOptimizationTool::CbcRust => "CBC Rust bindings",
        }
    }

    pub fn ecosystem(self) -> ExternalOptimizationEcosystem {
        match self {
            ExternalOptimizationTool::ChocoSolver
            | ExternalOptimizationTool::JaCoP
            | ExternalOptimizationTool::IbmCpOptimizer
            | ExternalOptimizationTool::OptaPlanner
            | ExternalOptimizationTool::JMetal
            | ExternalOptimizationTool::MoeaFramework
            | ExternalOptimizationTool::Ecj
            | ExternalOptimizationTool::OjAlgo
            | ExternalOptimizationTool::OrToolsJava => ExternalOptimizationEcosystem::Java,
            ExternalOptimizationTool::GoodLp
            | ExternalOptimizationTool::LpModeler
            | ExternalOptimizationTool::RustLinprog
            | ExternalOptimizationTool::Argmin
            | ExternalOptimizationTool::NloptRs
            | ExternalOptimizationTool::HighsRust
            | ExternalOptimizationTool::ScipRust
            | ExternalOptimizationTool::CbcRust => ExternalOptimizationEcosystem::Rust,
        }
    }

    pub fn env_var(self) -> &'static str {
        match self {
            ExternalOptimizationTool::ChocoSolver => "CHOCO_SOLVER_CLASSPATH",
            ExternalOptimizationTool::JaCoP => "JACOP_CLASSPATH",
            ExternalOptimizationTool::IbmCpOptimizer => "IBM_CP_OPTIMIZER_CLASSPATH",
            ExternalOptimizationTool::OptaPlanner => "OPTAPLANNER_CLASSPATH",
            ExternalOptimizationTool::JMetal => "JMETAL_CLASSPATH",
            ExternalOptimizationTool::MoeaFramework => "MOEA_FRAMEWORK_CLASSPATH",
            ExternalOptimizationTool::Ecj => "ECJ_CLASSPATH",
            ExternalOptimizationTool::OjAlgo => "OJALGO_CLASSPATH",
            ExternalOptimizationTool::OrToolsJava => "ORTOOLS_JAVA_CLASSPATH",
            ExternalOptimizationTool::GoodLp => "GOOD_LP_CARGO_MANIFEST",
            ExternalOptimizationTool::LpModeler => "LP_MODELER_CARGO_MANIFEST",
            ExternalOptimizationTool::RustLinprog => "RUST_LINPROG_CARGO_MANIFEST",
            ExternalOptimizationTool::Argmin => "ARGMIN_CARGO_MANIFEST",
            ExternalOptimizationTool::NloptRs => "NLOPT_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::HighsRust => "HIGHS_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::ScipRust => "SCIP_RS_CARGO_MANIFEST",
            ExternalOptimizationTool::CbcRust => "CBC_RS_CARGO_MANIFEST",
        }
    }

    pub fn java_probe_classes(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::ChocoSolver => &["org.chocosolver.solver.Model"],
            ExternalOptimizationTool::JaCoP => &["org.jacop.core.Store"],
            ExternalOptimizationTool::IbmCpOptimizer => &["ilog.cp.IloCP"],
            ExternalOptimizationTool::OptaPlanner => {
                &["org.optaplanner.core.api.solver.SolverFactory"]
            }
            ExternalOptimizationTool::JMetal => &["org.uma.jmetal.algorithm.Algorithm"],
            ExternalOptimizationTool::MoeaFramework => &["org.moeaframework.Executor"],
            ExternalOptimizationTool::Ecj => &["ec.Evolve"],
            ExternalOptimizationTool::OjAlgo => &["org.ojalgo.optimisation.ExpressionsBasedModel"],
            ExternalOptimizationTool::OrToolsJava => &[
                "com.google.ortools.Loader",
                "com.google.ortools.sat.CpModel",
            ],
            _ => &[],
        }
    }

    pub fn rust_dependency_names(self) -> &'static [&'static str] {
        match self {
            ExternalOptimizationTool::GoodLp => &["good_lp"],
            ExternalOptimizationTool::LpModeler => &["lp-modeler", "lp_modeler"],
            ExternalOptimizationTool::RustLinprog => &["rust-linprog", "linprog"],
            ExternalOptimizationTool::Argmin => &["argmin"],
            ExternalOptimizationTool::NloptRs => &["nlopt", "nlopt-sys"],
            ExternalOptimizationTool::HighsRust => &["highs", "highs-sys"],
            ExternalOptimizationTool::ScipRust => &["russcip", "scip", "scip-sys"],
            ExternalOptimizationTool::CbcRust => &["coin_cbc", "cbc", "cbc-sys"],
            _ => &[],
        }
    }
}

/// Probe status for an optional ecosystem integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalOptimizationProbeStatus {
    Ready,
    NotConfigured,
    RuntimeMissing,
    ArtifactMissing,
    ProbeFailed,
}

impl ExternalOptimizationProbeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalOptimizationProbeStatus::Ready => "ready",
            ExternalOptimizationProbeStatus::NotConfigured => "not-configured",
            ExternalOptimizationProbeStatus::RuntimeMissing => "runtime-missing",
            ExternalOptimizationProbeStatus::ArtifactMissing => "artifact-missing",
            ExternalOptimizationProbeStatus::ProbeFailed => "probe-failed",
        }
    }
}

/// Probe result for one optional Java/Rust optimization integration.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalOptimizationProbe {
    pub tool: ExternalOptimizationTool,
    pub ecosystem: ExternalOptimizationEcosystem,
    pub status: ExternalOptimizationProbeStatus,
    pub command: Option<PathBuf>,
    pub env_var: &'static str,
    pub artifact: Option<String>,
    pub elapsed_ms: f64,
    pub message: String,
}

/// Probe one Java classpath or Rust Cargo-manifest integration.
pub fn probe_external_optimization_tool(
    tool: ExternalOptimizationTool,
) -> ExternalOptimizationProbe {
    match tool.ecosystem() {
        ExternalOptimizationEcosystem::Java => probe_java_tool(tool),
        ExternalOptimizationEcosystem::Rust => probe_rust_tool(tool),
    }
}

/// Probe all optional Java/Rust optimization integrations known to the bridge.
pub fn probe_external_optimization_tools() -> Vec<ExternalOptimizationProbe> {
    ExternalOptimizationTool::all()
        .iter()
        .copied()
        .map(probe_external_optimization_tool)
        .collect()
}

fn probe_java_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let env_var = tool.env_var();
    let classpath = match env::var_os(env_var) {
        Some(value) if !value.is_empty() => value,
        _ => {
            return probe_result(
                tool,
                ExternalOptimizationProbeStatus::NotConfigured,
                None,
                env_var,
                None,
                elapsed_ms(t0),
                format!(
                    "set {env_var} to a local jar/classpath for {}",
                    tool.display_name()
                ),
            );
        }
    };
    let javap = find_first_command(&["javap"]);
    let Some(javap) = javap else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::RuntimeMissing,
            None,
            env_var,
            Some(classpath.to_string_lossy().to_string()),
            elapsed_ms(t0),
            "javap was not found on PATH; install a local JDK to probe Java solver jars"
                .to_string(),
        );
    };

    let mut last_error = String::new();
    for class_name in tool.java_probe_classes() {
        match Command::new(&javap)
            .arg("-classpath")
            .arg(&classpath)
            .arg(class_name)
            .output()
        {
            Ok(output) if output.status.success() => {
                return probe_result(
                    tool,
                    ExternalOptimizationProbeStatus::Ready,
                    Some(javap),
                    env_var,
                    Some(classpath.to_string_lossy().to_string()),
                    elapsed_ms(t0),
                    format!(
                        "found Java API class {class_name} for {}",
                        tool.display_name()
                    ),
                );
            }
            Ok(output) => {
                last_error = String::from_utf8_lossy(&output.stderr).trim().to_string();
            }
            Err(err) => {
                last_error = err.to_string();
            }
        }
    }

    probe_result(
        tool,
        ExternalOptimizationProbeStatus::ArtifactMissing,
        Some(javap),
        env_var,
        Some(classpath.to_string_lossy().to_string()),
        elapsed_ms(t0),
        format!(
            "{} classpath did not expose any of {:?}: {}",
            tool.display_name(),
            tool.java_probe_classes(),
            last_error
        ),
    )
}

fn probe_rust_tool(tool: ExternalOptimizationTool) -> ExternalOptimizationProbe {
    let t0 = Instant::now();
    let env_var = tool.env_var();
    let manifest = match env::var_os(env_var) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => {
            return probe_result(
                tool,
                ExternalOptimizationProbeStatus::NotConfigured,
                None,
                env_var,
                None,
                elapsed_ms(t0),
                format!(
                    "set {env_var} to a Cargo.toml that uses {}",
                    tool.display_name()
                ),
            );
        }
    };
    let cargo = find_first_command(&["cargo"]);
    let Some(cargo) = cargo else {
        return probe_result(
            tool,
            ExternalOptimizationProbeStatus::RuntimeMissing,
            None,
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            "cargo was not found on PATH; Rust crate integrations build through Cargo".to_string(),
        );
    };

    let raw = match fs::read_to_string(&manifest) {
        Ok(raw) => raw,
        Err(err) => {
            return probe_result(
                tool,
                ExternalOptimizationProbeStatus::ArtifactMissing,
                Some(cargo),
                env_var,
                Some(manifest.display().to_string()),
                elapsed_ms(t0),
                format!(
                    "failed to read Cargo manifest '{}': {err}",
                    manifest.display()
                ),
            );
        }
    };
    let dependency = tool
        .rust_dependency_names()
        .iter()
        .copied()
        .find(|name| cargo_manifest_mentions_dependency(&raw, name));
    match dependency {
        Some(name) => probe_result(
            tool,
            ExternalOptimizationProbeStatus::Ready,
            Some(cargo),
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            format!(
                "Cargo manifest '{}' references dependency '{}'",
                manifest.display(),
                name
            ),
        ),
        None => probe_result(
            tool,
            ExternalOptimizationProbeStatus::ArtifactMissing,
            Some(cargo),
            env_var,
            Some(manifest.display().to_string()),
            elapsed_ms(t0),
            format!(
                "Cargo manifest '{}' did not reference any of {:?}",
                manifest.display(),
                tool.rust_dependency_names()
            ),
        ),
    }
}

fn cargo_manifest_mentions_dependency(raw: &str, dependency: &str) -> bool {
    raw.lines().any(|line| {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            return false;
        }
        trimmed.starts_with(&format!("{dependency} "))
            || trimmed.starts_with(&format!("{dependency}="))
            || trimmed.starts_with(&format!("{dependency} ="))
            || trimmed.starts_with(&format!("\"{dependency}\""))
            || trimmed.starts_with(&format!("'{dependency}'"))
            || trimmed.contains(&format!("package = \"{dependency}\""))
            || trimmed.contains(&format!("package = '{dependency}'"))
    })
}

fn probe_result(
    tool: ExternalOptimizationTool,
    status: ExternalOptimizationProbeStatus,
    command: Option<PathBuf>,
    env_var: &'static str,
    artifact: Option<String>,
    elapsed_ms: f64,
    message: String,
) -> ExternalOptimizationProbe {
    ExternalOptimizationProbe {
        tool,
        ecosystem: tool.ecosystem(),
        status,
        command,
        env_var,
        artifact,
        elapsed_ms,
        message,
    }
}

fn find_first_command(aliases: &[&str]) -> Option<PathBuf> {
    aliases.iter().find_map(|alias| find_command(alias))
}

fn find_command(alias: &str) -> Option<PathBuf> {
    let alias_path = Path::new(alias);
    if alias_path.components().count() > 1 {
        return executable_file(alias_path).then(|| alias_path.to_path_buf());
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(alias))
        .find(|candidate| executable_file(candidate))
}

fn executable_file(path: impl AsRef<Path>) -> bool {
    path.as_ref().is_file()
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecosystem_tool_metadata_covers_java_and_rust() {
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.ecosystem(),
            ExternalOptimizationEcosystem::Java
        );
        assert_eq!(
            ExternalOptimizationTool::GoodLp.ecosystem(),
            ExternalOptimizationEcosystem::Rust
        );
        assert_eq!(
            ExternalOptimizationTool::ChocoSolver.env_var(),
            "CHOCO_SOLVER_CLASSPATH"
        );
        assert!(ExternalOptimizationTool::OjAlgo
            .java_probe_classes()
            .contains(&"org.ojalgo.optimisation.ExpressionsBasedModel"));
        assert!(ExternalOptimizationTool::HighsRust
            .rust_dependency_names()
            .contains(&"highs-sys"));
    }

    #[test]
    fn cargo_manifest_dependency_probe_handles_common_forms() {
        let raw = r#"
            [dependencies]
            good_lp = "1"
            highs-wrapper = { package = "highs", version = "0.1" }
        "#;
        assert!(cargo_manifest_mentions_dependency(raw, "good_lp"));
        assert!(cargo_manifest_mentions_dependency(raw, "highs"));
        assert!(!cargo_manifest_mentions_dependency(raw, "argmin"));
    }
}
