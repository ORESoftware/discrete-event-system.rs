//! Shared JSON authoring metadata for graph-based models.
//!
//! The concrete runnable specs live in `studio::spec` and `hybrid::spec`. This
//! module carries the cross-cutting surface a real modeling tool needs around
//! those graphs: hierarchy, variants, physical connectors/equations, solver
//! policy, FMI intent, requirements/V&V, and Rust code generation settings.

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MODEL_AUTHORING_SCHEMA: &str = "des/model-authoring/v1";

/// JSON Schema for the shared authoring extension block.
pub fn model_authoring_json_schema() -> Value {
    serde_json::to_value(schema_for!(ModelAuthoringSpec))
        .expect("ModelAuthoringSpec schema serializes")
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelAuthoringSpec {
    #[serde(default)]
    pub metadata: ModelMetadataSpec,
    #[serde(default)]
    pub hierarchy: Vec<ModelReferenceSpec>,
    #[serde(default)]
    pub variants: Vec<VariantSpec>,
    #[serde(default)]
    pub libraries: Vec<LibraryReferenceSpec>,
    #[serde(default)]
    pub physical_domains: Vec<PhysicalDomainSpec>,
    #[serde(default)]
    pub physical_connectors: Vec<PhysicalConnectorSpec>,
    #[serde(default)]
    pub equations: Vec<EquationSpec>,
    #[serde(default)]
    pub solver: SolverSelectionSpec,
    #[serde(default)]
    pub statecharts: Vec<StatechartSpec>,
    #[serde(default)]
    pub fmi: FmiInteropSpec,
    #[serde(default)]
    pub verification: VerificationSpec,
    #[serde(default)]
    pub tooling: ToolingSpec,
    #[serde(default)]
    pub codegen: CodegenSpec,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadataSpec {
    pub description: Option<String>,
    pub version: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PortDirection {
    Input,
    Output,
    Inout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum PortDomainSpec {
    Signal,
    Event,
    Bus { bus: String },
    Physical { domain: String },
}

impl Default for PortDomainSpec {
    fn default() -> Self {
        PortDomainSpec::Signal
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PortDescriptorSpec {
    pub name: String,
    pub direction: PortDirection,
    #[serde(default = "one_usize")]
    pub width: usize,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub domain: PortDomainSpec,
}

fn one_usize() -> usize {
    1
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelReferenceSpec {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub interface: Vec<PortDescriptorSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct VariantSpec {
    pub name: String,
    pub condition: String,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub selected_blocks: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LibraryReferenceSpec {
    pub name: String,
    pub version: Option<String>,
    pub path: Option<String>,
    #[serde(default)]
    pub components: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PhysicalVariableRole {
    Across,
    Through,
    Stream,
    Parameter,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalVariableSpec {
    pub name: String,
    pub role: PhysicalVariableRole,
    pub unit: Option<String>,
    #[serde(default)]
    pub nominal: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalDomainSpec {
    pub name: String,
    #[serde(default)]
    pub effort: Vec<PhysicalVariableSpec>,
    #[serde(default)]
    pub flow: Vec<PhysicalVariableSpec>,
    #[serde(default)]
    pub stream: Vec<PhysicalVariableSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalConnectorSpec {
    pub id: String,
    pub domain: String,
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub variables: Vec<PhysicalVariableSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EquationSpec {
    pub id: String,
    /// Human-readable equation text. A future symbolic compiler can lower this
    /// into residual functions and connection equations.
    pub expression: String,
    #[serde(default)]
    pub variables: Vec<String>,
    #[serde(default)]
    pub when: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ContinuousSolverKind {
    Auto,
    Rk4,
    Rk45,
    BackwardEuler,
}

impl Default for ContinuousSolverKind {
    fn default() -> Self {
        ContinuousSolverKind::Auto
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum AlgebraicSolverKind {
    Reject,
    Newton,
    FixedPoint,
}

impl Default for AlgebraicSolverKind {
    fn default() -> Self {
        AlgebraicSolverKind::Reject
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SolverSelectionSpec {
    #[serde(default)]
    pub continuous: ContinuousSolverKind,
    #[serde(default)]
    pub algebraic: AlgebraicSolverKind,
    #[serde(default)]
    pub rel_tol: Option<f64>,
    #[serde(default)]
    pub abs_tol: Option<f64>,
    #[serde(default)]
    pub max_step: Option<f64>,
    #[serde(default)]
    pub index_reduction: Option<String>,
}

impl Default for SolverSelectionSpec {
    fn default() -> Self {
        SolverSelectionSpec {
            continuous: ContinuousSolverKind::Auto,
            algebraic: AlgebraicSolverKind::Reject,
            rel_tol: Some(1e-6),
            abs_tol: Some(1e-9),
            max_step: None,
            index_reduction: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StatechartSpec {
    pub id: String,
    #[serde(default)]
    pub states: Vec<StateSpec>,
    #[serde(default)]
    pub transitions: Vec<StateTransitionSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StateSpec {
    pub id: String,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub initial: bool,
    #[serde(default)]
    pub entry: Vec<String>,
    #[serde(default)]
    pub during: Vec<String>,
    #[serde(default)]
    pub exit: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StateTransitionSpec {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub guard: Option<String>,
    #[serde(default)]
    pub action: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum FmiVersion {
    V2,
    V3,
}

impl Default for FmiVersion {
    fn default() -> Self {
        FmiVersion::V3
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum FmuKind {
    ModelExchange,
    CoSimulation,
    ScheduledExecution,
}

impl Default for FmuKind {
    fn default() -> Self {
        FmuKind::CoSimulation
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FmiInteropSpec {
    #[serde(default)]
    pub imports: Vec<FmuImportSpec>,
    #[serde(default)]
    pub export: Option<FmuExportSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FmuImportSpec {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub version: FmiVersion,
    #[serde(default)]
    pub kind: FmuKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FmuExportSpec {
    pub name: String,
    #[serde(default)]
    pub version: FmiVersion,
    #[serde(default)]
    pub kind: FmuKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct VerificationSpec {
    #[serde(default)]
    pub requirements: Vec<RequirementSpec>,
    #[serde(default)]
    pub tests: Vec<ModelTestSpec>,
    #[serde(default)]
    pub coverage: CoverageSpec,
    #[serde(default)]
    pub formal_checks: Vec<FormalCheckSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RequirementSpec {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub linked_blocks: Vec<String>,
    #[serde(default)]
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelTestSpec {
    pub name: String,
    #[serde(default)]
    pub inputs: Value,
    #[serde(default)]
    pub expected: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CoverageSpec {
    #[serde(default)]
    pub signals: Vec<String>,
    #[serde(default)]
    pub states: Vec<String>,
    #[serde(default)]
    pub transitions: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FormalCheckSpec {
    pub name: String,
    pub property: String,
    #[serde(default)]
    pub horizon: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolingSpec {
    #[serde(default)]
    pub data_dictionaries: Vec<String>,
    #[serde(default)]
    pub signal_inspectors: Vec<SignalInspectorSpec>,
    #[serde(default)]
    pub variant_manager: bool,
    #[serde(default)]
    pub dependency_analyzer: bool,
    #[serde(default)]
    pub parameter_estimation: bool,
    #[serde(default)]
    pub protected_model: bool,
    #[serde(default)]
    pub collaboration: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SignalInspectorSpec {
    pub name: String,
    #[serde(default)]
    pub signals: Vec<String>,
    #[serde(default)]
    pub tolerance: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum CodegenLanguage {
    Rust,
}

impl Default for CodegenLanguage {
    fn default() -> Self {
        CodegenLanguage::Rust
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CodegenSpec {
    #[serde(default)]
    pub language: CodegenLanguage,
    #[serde(default = "default_emit_codegen")]
    pub emit: bool,
    #[serde(default)]
    pub rust: RustCodegenSpec,
}

impl Default for CodegenSpec {
    fn default() -> Self {
        CodegenSpec {
            language: CodegenLanguage::Rust,
            emit: true,
            rust: RustCodegenSpec::default(),
        }
    }
}

fn default_emit_codegen() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RustCodegenSpec {
    #[serde(default = "default_module_name")]
    pub module_name: String,
    #[serde(default = "default_function_name")]
    pub function_name: String,
    #[serde(default = "default_edition")]
    pub edition: String,
}

impl Default for RustCodegenSpec {
    fn default() -> Self {
        RustCodegenSpec {
            module_name: default_module_name(),
            function_name: default_function_name(),
            edition: default_edition(),
        }
    }
}

fn default_module_name() -> String {
    "generated_model".to_string()
}

fn default_function_name() -> String {
    "run_generated_model".to_string()
}

fn default_edition() -> String {
    "2021".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoring_schema_is_generated_from_rust_types() {
        let schema = model_authoring_json_schema();
        assert_eq!(schema["title"], "ModelAuthoringSpec");
        assert!(schema["properties"]["physicalConnectors"].is_object());
        assert!(schema["properties"]["codegen"].is_object());
    }

    #[test]
    fn defaults_select_rust_codegen_and_reject_algebraic_loops() {
        let authoring = ModelAuthoringSpec::default();
        assert_eq!(authoring.codegen.language, CodegenLanguage::Rust);
        assert_eq!(authoring.solver.algebraic, AlgebraicSolverKind::Reject);
    }
}
