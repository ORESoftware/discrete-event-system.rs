//! Port of `src/des/general/universal-model-spec.ts` — module
//! `des::general::universal_model_spec`.
//!
//! The portable JSON document shape for the modeling layer plus converters. A
//! universal document captures (1) the original user input, (2) normalized
//! mathematics, (3) generated DES stationary entities and moving-entity edges,
//! and (4) solver/runtime intent. The existing `des/model-spec/v1` registry
//! envelope remains the execution envelope; a universal document can be
//! converted to one when the solver target is a registered model such as
//! `math-equation`.
//!
//! Conversion notes from the TS source:
//!   * Every shape becomes a plain struct; the TS string-literal unions become
//!     enums. `serde` is NOT a dependency of this crate, so the document is
//!     modelled with the crate's hand-rolled
//!     [`crate::des::general::des_spec::JsonValue`] /
//!     [`crate::des::general::des_spec::JsonObject`] for the open-ended
//!     `Record<string, unknown>` / `unknown` payloads, and typed enums for the
//!     numeric/string/array unions ([`ScalarOrArray`]).
//!   * The `isUniversalDESModelSpec` structural type guard becomes a small
//!     `$schema` probe over a `JsonValue`.
//!   * `validate*` returns `Vec<ValidationCheck>` (recoverable); `assert*`
//!     throws, mapped to a `Result<_, String>` (recoverable construction-time
//!     failure) rather than `panic!`, since the converters call it.
//!   * The math-block types (`ODEBlockSystemParams`, `BlockGraphNode`, ...) and
//!     `MathEquation*` come from [`crate::des::general::math_equation_input`]
//!     (which carries the placeholder math-blocks surface — see its PORT NOTE).
//!
//! PORT NOTE: a handful of TS `Record<string, unknown>` fields that round-trip a
//! `MathEquationInputParams` (notably `UniversalNormalizedMath.parameters` and
//! the `DESModelSpec` parameter payload) are modelled as the typed value
//! directly instead of an erased JSON record, because the crate has no
//! `serde`-based round-trip. HashMap-derived orderings (e.g. ODE constants →
//! parameters) are sorted by key for determinism.

#![allow(dead_code)]

use std::collections::HashSet;

use crate::des::general::des_base::validation::ValidationCheck;
use crate::des::general::des_spec::{
    DESModelMetadata, DESModelSpec, DESRuntimeConfig, JsonObject, JsonValue, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::math_equation_input::{
    BlockGraphEdge, BlockGraphNode, EquationInputFormat, EquationProblemKind, Heat1DBlockParams,
    IntegratorMethod, MathEquationInputParams, MathEquationResult, Normalized, ODEBlockSystemParams,
};

/// The required value of `UniversalDESModelSpec::schema` (TS `$schema` literal).
pub const UNIVERSAL_MODEL_SPEC_SCHEMA: &str = "des/universal-model/v1";

/// Error channel for the converters (TS `throw new Error(...)`).
type UResult<T> = Result<T, String>;

// =============================================================================
// Enums (TS string-literal unions).
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalModelKind {
    Ode,
    Pde,
    Optimization,
    NetworkFlow,
    TrafficFlow,
    Queueing,
    Agent,
    Custom,
}

/// TS `UniversalInputFormat = EquationInputFormat | 'json' | 'xml' | 'text' |
/// 'manual'` (the `'json'`/`'xml'` overlaps collapse).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalInputFormat {
    Json,
    Latex,
    Xml,
    Text,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalVariableRole {
    Independent,
    State,
    Field,
    Input,
    Output,
    Algebraic,
    Parameter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalEquationKind {
    Ode,
    Pde,
    Algebraic,
    Constraint,
    Objective,
    Boundary,
    Initial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalConditionKind {
    Initial,
    Dirichlet,
    Neumann,
    Periodic,
    Source,
    Sink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalEntityRole {
    Source,
    Sink,
    Processor,
    Integrator,
    FieldCell,
    Boundary,
    Logic,
    Optimizer,
    Observer,
}

/// TS `number | string | number[]` value union.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarOrArray {
    Number(f64),
    Text(String),
    Array(Vec<f64>),
}

// =============================================================================
// Document structs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct UniversalDESModelSpec {
    /// TS `$schema` (must equal [`UNIVERSAL_MODEL_SPEC_SCHEMA`]).
    pub schema: String,
    pub id: String,
    pub description: Option<String>,
    pub original_input: UniversalOriginalInput,
    pub math: UniversalMathSpec,
    pub des: UniversalDESNetworkSpec,
    pub solver: UniversalSolverSpec,
    pub runtime: Option<DESRuntimeConfig>,
    /// PORT NOTE: the TS metadata's open `[key]: unknown` extension is dropped;
    /// we reuse [`DESModelMetadata`] (author/createdAt/tags/notes).
    pub metadata: Option<DESModelMetadata>,
}

#[derive(Clone, Debug)]
pub struct UniversalOriginalInput {
    pub format: UniversalInputFormat,
    pub content: Option<String>,
    pub uri: Option<String>,
    pub content_type: Option<String>,
    pub language: Option<String>,
    pub captured_at: Option<String>,
    pub metadata: Option<JsonObject>,
}

#[derive(Clone, Debug)]
pub struct UniversalMathSpec {
    pub kind: UniversalModelKind,
    pub independent_variables: Vec<UniversalMathVariable>,
    pub state_variables: Vec<UniversalMathVariable>,
    pub parameters: Option<Vec<UniversalMathParameter>>,
    pub equations: Vec<UniversalMathEquation>,
    pub initial_conditions: Option<Vec<UniversalMathCondition>>,
    pub boundary_conditions: Option<Vec<UniversalMathCondition>>,
    pub constraints: Option<Vec<UniversalMathEquation>>,
    pub objectives: Option<Vec<UniversalMathEquation>>,
    pub domain: Option<JsonObject>,
    pub numerics: Option<UniversalNumericsSpec>,
    pub normalized: Option<UniversalNormalizedMath>,
}

#[derive(Clone, Debug)]
pub struct UniversalMathVariable {
    pub id: String,
    pub role: UniversalVariableRole,
    pub initial: Option<ScalarOrArray>,
    pub units: Option<String>,
    pub domain: Option<JsonObject>,
    pub description: Option<String>,
}

#[derive(Clone, Debug)]
pub struct UniversalMathParameter {
    pub id: String,
    pub value: ScalarOrArray,
    pub units: Option<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug)]
pub struct UniversalMathEquation {
    pub id: String,
    pub kind: UniversalEquationKind,
    pub lhs: Option<String>,
    pub rhs: Option<String>,
    pub expression: Option<String>,
    pub normalized_expression: Option<String>,
    pub variables: Option<Vec<String>>,
    pub metadata: Option<JsonObject>,
}

#[derive(Clone, Debug)]
pub struct UniversalMathCondition {
    pub id: String,
    pub variable: String,
    pub at: Option<JsonObject>,
    pub value: ScalarOrArray,
    pub kind: UniversalConditionKind,
    pub expression: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalNumericsSpec {
    pub time: Option<UniversalTimeSpec>,
    pub space: Option<UniversalSpaceSpec>,
    pub method: Option<String>,
    pub tolerances: Option<JsonObject>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalTimeSpec {
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub steps: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalSpaceSpec {
    pub dimensions: Option<f64>,
    pub cells: Option<ScalarOrArray>,
    pub length: Option<ScalarOrArray>,
    pub dx: Option<ScalarOrArray>,
}

#[derive(Clone, Debug)]
pub struct UniversalNormalizedMath {
    pub target_model: String,
    pub parameters: UniversalNormalizedParameters,
}

/// PORT NOTE: TS `parameters: Record<string, unknown>` round-trips a
/// `MathEquationInputParams`; modelled here as a typed variant to avoid a serde
/// round-trip, with an `Other` fallback for arbitrary JSON.
#[derive(Clone, Debug)]
pub enum UniversalNormalizedParameters {
    MathEquation(Box<MathEquationInputParams>),
    Other(JsonObject),
}

#[derive(Clone, Debug)]
pub struct UniversalDESNetworkSpec {
    pub time: Option<UniversalTimeSpec>,
    pub stationary_entities: Vec<UniversalStationaryEntity>,
    pub moving_entities: Vec<UniversalMovingEntity>,
    pub graph: UniversalGraph,
    pub sources: Option<Vec<UniversalEndpointSpec>>,
    pub sinks: Option<Vec<UniversalEndpointSpec>>,
    pub observability: Option<UniversalObservability>,
}

#[derive(Clone, Debug)]
pub struct UniversalGraph {
    pub nodes: Vec<UniversalStationaryEntity>,
    pub edges: Vec<UniversalGraphEdge>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalObservability {
    pub record_signals: Option<bool>,
    pub record_state: Option<bool>,
    pub record_graph: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct UniversalStationaryEntity {
    pub id: String,
    pub kind: String,
    pub role: Option<UniversalEntityRole>,
    pub class_name: Option<String>,
    pub parameters: Option<JsonObject>,
    pub ports: Option<UniversalPorts>,
    pub position: Option<UniversalPosition>,
    pub metadata: Option<JsonObject>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalPorts {
    pub inputs: Option<Vec<String>>,
    pub outputs: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default)]
pub struct UniversalPosition {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub index: Option<ScalarOrArray>,
}

#[derive(Clone, Debug)]
pub struct UniversalMovingEntity {
    pub id: String,
    pub kind: String,
    pub token_type: String,
    pub payload_schema: Option<JsonObject>,
    pub semantics: Option<String>,
}

#[derive(Clone, Debug)]
pub struct UniversalGraphEdge {
    pub id: String,
    pub from: UniversalPortRef,
    pub to: UniversalPortRef,
    pub moving_entity: String,
    pub delay_ticks: Option<f64>,
    pub transform: Option<String>,
    pub metadata: Option<JsonObject>,
}

#[derive(Clone, Debug)]
pub struct UniversalPortRef {
    pub entity_id: String,
    pub port: String,
}

#[derive(Clone, Debug)]
pub struct UniversalEndpointSpec {
    pub id: String,
    pub entity_id: String,
    pub port: Option<String>,
    pub role: Option<String>,
    pub variable: Option<String>,
    pub value: Option<JsonValue>,
    pub record: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct UniversalSolverSpec {
    pub target_model: String,
    pub method: Option<String>,
    pub options: Option<JsonObject>,
}

/// Options bag for [`universal_from_math_equation_result`].
#[derive(Clone, Debug, Default)]
pub struct UniversalFromOpts {
    pub id: Option<String>,
    pub description: Option<String>,
    pub runtime: Option<DESRuntimeConfig>,
    pub metadata: Option<DESModelMetadata>,
}

// =============================================================================
// Type guard / validation.
// =============================================================================

/// TS `isUniversalDESModelSpec` — a structural `$schema` probe over JSON.
pub fn is_universal_des_model_spec(value: &JsonValue) -> bool {
    if let JsonValue::Object(o) = value {
        matches!(o.get("$schema"), Some(JsonValue::String(s)) if s == UNIVERSAL_MODEL_SPEC_SCHEMA)
    } else {
        false
    }
}

/// TS `validateUniversalDESModelSpec`.
pub fn validate_universal_des_model_spec(spec: &UniversalDESModelSpec) -> Vec<ValidationCheck> {
    let mut checks: Vec<ValidationCheck> = Vec::new();
    push(
        &mut checks,
        "universal-schema",
        spec.schema == UNIVERSAL_MODEL_SPEC_SCHEMA,
        Some(spec.schema.clone()),
        Some(UNIVERSAL_MODEL_SPEC_SCHEMA.to_string()),
    );
    push(
        &mut checks,
        "universal-id",
        is_non_empty(&spec.id),
        Some(spec.id.clone()),
        Some("non-empty string".to_string()),
    );
    push(
        &mut checks,
        "original-input-present",
        is_non_empty(input_format_str(spec.original_input.format)),
        Some(input_format_str(spec.original_input.format).to_string()),
        Some("format string".to_string()),
    );
    push(
        &mut checks,
        "original-input-content-or-uri",
        is_non_empty_opt(&spec.original_input.content) || is_non_empty_opt(&spec.original_input.uri),
        Some("content/uri".to_string()),
        Some("one must be present".to_string()),
    );
    push(
        &mut checks,
        "math-kind",
        is_non_empty(model_kind_str(spec.math.kind)),
        Some(model_kind_str(spec.math.kind).to_string()),
        Some("math kind".to_string()),
    );
    push(
        &mut checks,
        "math-equations-non-empty",
        !spec.math.equations.is_empty(),
        Some(spec.math.equations.len().to_string()),
        Some("at least one equation".to_string()),
    );
    push(
        &mut checks,
        "stationary-entities-non-empty",
        !spec.des.stationary_entities.is_empty(),
        Some(spec.des.stationary_entities.len().to_string()),
        Some("at least one stationary entity".to_string()),
    );
    push(
        &mut checks,
        "moving-entities-non-empty",
        !spec.des.moving_entities.is_empty(),
        Some(spec.des.moving_entities.len().to_string()),
        Some("at least one moving entity kind".to_string()),
    );
    push(
        &mut checks,
        "solver-target-model",
        is_non_empty(&spec.solver.target_model),
        Some(spec.solver.target_model.clone()),
        Some("registered target model id".to_string()),
    );

    let node_ids: Vec<String> = spec.des.stationary_entities.iter().map(|n| n.id.clone()).collect();
    let moving_ids: Vec<String> = spec.des.moving_entities.iter().map(|m| m.id.clone()).collect();
    push(
        &mut checks,
        "stationary-ids-unique",
        unique(&node_ids),
        Some(duplicate(&node_ids).unwrap_or_else(|| "unique".to_string())),
        Some("unique stationary ids".to_string()),
    );
    push(
        &mut checks,
        "moving-ids-unique",
        unique(&moving_ids),
        Some(duplicate(&moving_ids).unwrap_or_else(|| "unique".to_string())),
        Some("unique moving ids".to_string()),
    );
    let node_set: HashSet<&str> = node_ids.iter().map(|s| s.as_str()).collect();
    let moving_set: HashSet<&str> = moving_ids.iter().map(|s| s.as_str()).collect();
    for edge in &spec.des.graph.edges {
        push(
            &mut checks,
            &format!("edge-from-ref/{}", edge.id),
            node_set.contains(edge.from.entity_id.as_str()),
            Some(edge.from.entity_id.clone()),
            Some("known stationary entity".to_string()),
        );
        push(
            &mut checks,
            &format!("edge-to-ref/{}", edge.id),
            node_set.contains(edge.to.entity_id.as_str()),
            Some(edge.to.entity_id.clone()),
            Some("known stationary entity".to_string()),
        );
        push(
            &mut checks,
            &format!("edge-moving-ref/{}", edge.id),
            moving_set.contains(edge.moving_entity.as_str()),
            Some(edge.moving_entity.clone()),
            Some("known moving entity".to_string()),
        );
    }
    if let Some(sources) = &spec.des.sources {
        for source in sources {
            push(
                &mut checks,
                &format!("source-ref/{}", source.id),
                node_set.contains(source.entity_id.as_str()),
                Some(source.entity_id.clone()),
                Some("known stationary entity".to_string()),
            );
        }
    }
    if let Some(sinks) = &spec.des.sinks {
        for sink in sinks {
            push(
                &mut checks,
                &format!("sink-ref/{}", sink.id),
                node_set.contains(sink.entity_id.as_str()),
                Some(sink.entity_id.clone()),
                Some("known stationary entity".to_string()),
            );
        }
    }
    let dt = spec
        .math
        .numerics
        .as_ref()
        .and_then(|n| n.time.as_ref())
        .and_then(|t| t.dt)
        .or_else(|| spec.des.time.as_ref().and_then(|t| t.dt));
    if let Some(dt) = dt {
        push(
            &mut checks,
            "time-dt-positive",
            dt.is_finite() && dt > 0.0,
            Some(dt.to_string()),
            Some("finite dt > 0".to_string()),
        );
    }
    checks
}

/// TS `assertUniversalDESModelSpec` — `throw` mapped to `Result::Err`.
pub fn assert_universal_des_model_spec(spec: &UniversalDESModelSpec) -> UResult<()> {
    let failed: Vec<ValidationCheck> = validate_universal_des_model_spec(spec)
        .into_iter()
        .filter(|c| !c.passed)
        .collect();
    if failed.is_empty() {
        return Ok(());
    }
    let body = failed
        .iter()
        .map(|c| {
            format!(
                "{}: observed={} expected={}",
                c.name,
                c.observed.clone().unwrap_or_default(),
                c.expected.clone().unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n  ");
    Err(format!("invalid universal DES model spec:\n  {body}"))
}

// =============================================================================
// Converters.
// =============================================================================

/// TS `universalToDESModelSpec`.
pub fn universal_to_des_model_spec(
    spec: &UniversalDESModelSpec,
) -> UResult<DESModelSpec<MathEquationInputParams>> {
    assert_universal_des_model_spec(spec)?;
    if spec.solver.target_model != "math-equation" {
        return Err(format!(
            "universalToDESModelSpec: unsupported targetModel \"{}\"",
            spec.solver.target_model
        ));
    }
    let params = universal_to_math_equation_input(spec)?;
    // PORT NOTE: the TS payload is erased to `Record<string, unknown>`; the typed
    // `MathEquationInputParams` is kept here (no serde round-trip available).
    Ok(DESModelSpec {
        schema: DES_MODEL_SPEC_SCHEMA.to_string(),
        model: "math-equation".to_string(),
        description: spec.description.clone(),
        parameters: params,
        runtime: spec.runtime.clone(),
        metadata: spec.metadata.clone(),
    })
}

/// TS `universalToMathEquationInput`.
pub fn universal_to_math_equation_input(
    spec: &UniversalDESModelSpec,
) -> UResult<MathEquationInputParams> {
    if let Some(normalized) = &spec.math.normalized {
        if normalized.target_model == "math-equation" {
            if let UniversalNormalizedParameters::MathEquation(p) = &normalized.parameters {
                return Ok((**p).clone());
            }
        }
    }
    let format = spec.original_input.format;
    let eq_format = match format {
        UniversalInputFormat::Json => EquationInputFormat::Json,
        UniversalInputFormat::Latex => EquationInputFormat::Latex,
        UniversalInputFormat::Xml => EquationInputFormat::Xml,
        _ => {
            return Err(format!(
                "universalToMathEquationInput: original input format \"{}\" cannot be run by math-equation",
                input_format_str(format)
            ))
        }
    };
    let time = spec.math.numerics.as_ref().and_then(|n| n.time.as_ref());
    Ok(MathEquationInputParams {
        format: eq_format,
        kind: Some(if spec.math.kind == UniversalModelKind::Pde {
            EquationProblemKind::Heat1d
        } else {
            EquationProblemKind::Ode
        }),
        equation: spec.original_input.content.clone(),
        t0: time.and_then(|t| t.t0),
        t1: time.and_then(|t| t.t1),
        dt: time.and_then(|t| t.dt),
        ..Default::default()
    })
}

/// TS `universalFromMathEquationResult`.
pub fn universal_from_math_equation_result(
    input: &MathEquationInputParams,
    result: &MathEquationResult,
    opts: UniversalFromOpts,
) -> UResult<UniversalDESModelSpec> {
    let stationary_entities: Vec<UniversalStationaryEntity> =
        result.network.nodes.iter().map(universal_stationary_from_block).collect();
    let edges: Vec<UniversalGraphEdge> = result
        .network
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| universal_edge_from_block(e, i))
        .collect();
    let (math, time) = match &result.normalized {
        Normalized::Ode(p) => (math_from_ode_result(input, p), time_from_ode(p)),
        Normalized::Heat1d(p) => (math_from_heat_result(input, p), time_from_heat(p)),
    };
    let mut spec = UniversalDESModelSpec {
        schema: UNIVERSAL_MODEL_SPEC_SCHEMA.to_string(),
        id: opts
            .id
            .clone()
            .unwrap_or_else(|| format!("universal-{}", problem_kind_str(result.kind))),
        description: opts.description.clone(),
        original_input: original_from_input(input),
        math,
        des: UniversalDESNetworkSpec {
            time: Some(time),
            stationary_entities: stationary_entities.clone(),
            moving_entities: vec![UniversalMovingEntity {
                id: "MathSignal".to_string(),
                kind: "signal-token".to_string(),
                token_type: "MathSignal".to_string(),
                payload_schema: Some(math_signal_schema()),
                semantics: Some(
                    "Scalar value token moving between stationary DES math blocks.".to_string(),
                ),
            }],
            graph: UniversalGraph { nodes: stationary_entities, edges },
            sources: Some(source_endpoints(&result.network.nodes, &result.normalized)),
            sinks: Some(sink_endpoints(&result.network.nodes)),
            observability: Some(UniversalObservability {
                record_signals: Some(true),
                record_state: Some(true),
                record_graph: Some(true),
            }),
        },
        solver: UniversalSolverSpec {
            target_model: "math-equation".to_string(),
            method: method_from_normalized(&result.normalized),
            options: None,
        },
        runtime: opts.runtime.clone(),
        metadata: opts.metadata.clone(),
    };
    spec.math.normalized = Some(UniversalNormalizedMath {
        target_model: "math-equation".to_string(),
        parameters: UniversalNormalizedParameters::MathEquation(Box::new(input.clone())),
    });
    assert_universal_des_model_spec(&spec)?;
    Ok(spec)
}

// =============================================================================
// math/normalized → universal builders.
// =============================================================================

fn math_from_ode_result(_input: &MathEquationInputParams, params: &ODEBlockSystemParams) -> UniversalMathSpec {
    let state_names: Vec<String> = params.states.iter().map(|s| s.name.clone()).collect();
    UniversalMathSpec {
        kind: UniversalModelKind::Ode,
        independent_variables: vec![ind_var("t")],
        state_variables: params
            .states
            .iter()
            .map(|s| state_var(&s.name, ScalarOrArray::Number(s.initial)))
            .collect(),
        parameters: Some(
            sorted_constants(&params.constants)
                .into_iter()
                .map(|(id, value)| param(&id, ScalarOrArray::Number(value)))
                .collect(),
        ),
        equations: params
            .states
            .iter()
            .map(|s| UniversalMathEquation {
                id: format!("ode:{}", s.name),
                kind: UniversalEquationKind::Ode,
                lhs: Some(format!("d{}/dt", s.name)),
                rhs: Some(s.derivative.clone()),
                expression: None,
                normalized_expression: Some(s.derivative.clone()),
                variables: Some(state_names.clone()),
                metadata: None,
            })
            .collect(),
        initial_conditions: Some(
            params
                .states
                .iter()
                .map(|s| UniversalMathCondition {
                    id: format!("initial:{}", s.name),
                    variable: s.name.clone(),
                    at: Some(jobj("t", JsonValue::Number(params.t0))),
                    value: ScalarOrArray::Number(s.initial),
                    kind: UniversalConditionKind::Initial,
                    expression: None,
                })
                .collect(),
        ),
        boundary_conditions: None,
        constraints: None,
        objectives: None,
        domain: None,
        numerics: Some(UniversalNumericsSpec {
            time: Some(time_from_ode(params)),
            space: None,
            method: Some(integrator_str(params.method).to_string()),
            tolerances: None,
        }),
        normalized: None,
    }
}

fn math_from_heat_result(input: &MathEquationInputParams, params: &Heat1DBlockParams) -> UniversalMathSpec {
    let mut parameters = vec![
        param("alpha", ScalarOrArray::Number(params.alpha)),
        param("length", ScalarOrArray::Number(params.length)),
        param("cells", ScalarOrArray::Number(params.cells)),
    ];
    parameters.extend(
        sorted_constants(&params.constants)
            .into_iter()
            .map(|(id, value)| param(&id, ScalarOrArray::Number(value))),
    );
    let field_initial = heat_initial(params);
    UniversalMathSpec {
        kind: UniversalModelKind::Pde,
        independent_variables: vec![ind_var("t"), {
            let mut x = ind_var("x");
            x.domain = Some(jobj("length", JsonValue::Number(params.length)));
            x
        }],
        state_variables: vec![state_var("u", field_initial.clone())],
        parameters: Some(parameters),
        equations: vec![UniversalMathEquation {
            id: "pde:heat1d".to_string(),
            kind: UniversalEquationKind::Pde,
            lhs: Some("du/dt".to_string()),
            rhs: Some("alpha*d2u/dx2".to_string()),
            expression: None,
            normalized_expression: Some("alpha*(u[i-1] - 2*u[i] + u[i+1]) / dx^2".to_string()),
            variables: Some(vec!["u".to_string(), "x".to_string(), "t".to_string()]),
            metadata: Some(jobj(
                "sourceEquation",
                match &input.equation {
                    Some(e) => JsonValue::String(e.clone()),
                    None => JsonValue::Null,
                },
            )),
        }],
        initial_conditions: Some(vec![UniversalMathCondition {
            id: "initial:u".to_string(),
            variable: "u".to_string(),
            at: Some(jobj("t", JsonValue::Number(params.t0))),
            value: field_initial,
            kind: UniversalConditionKind::Initial,
            expression: None,
        }]),
        boundary_conditions: Some(vec![
            UniversalMathCondition {
                id: "boundary:left".to_string(),
                variable: "u".to_string(),
                at: Some(jobj("x", JsonValue::Number(0.0))),
                value: ScalarOrArray::Number(params.left_boundary.unwrap_or(0.0)),
                kind: UniversalConditionKind::Dirichlet,
                expression: None,
            },
            UniversalMathCondition {
                id: "boundary:right".to_string(),
                variable: "u".to_string(),
                at: Some(jobj("x", JsonValue::Number(params.length))),
                value: ScalarOrArray::Number(params.right_boundary.unwrap_or(0.0)),
                kind: UniversalConditionKind::Dirichlet,
                expression: None,
            },
        ]),
        constraints: None,
        objectives: None,
        domain: Some(jobj(
            "x",
            JsonValue::Array(vec![JsonValue::Number(0.0), JsonValue::Number(params.length)]),
        )),
        numerics: Some(UniversalNumericsSpec {
            time: Some(time_from_heat(params)),
            space: Some(UniversalSpaceSpec {
                dimensions: Some(1.0),
                cells: Some(ScalarOrArray::Number(params.cells)),
                length: Some(ScalarOrArray::Number(params.length)),
                dx: None,
            }),
            method: Some("explicit-euler-laplacian".to_string()),
            tolerances: None,
        }),
        normalized: None,
    }
}

/// TS `params.initialValues ?? params.initialExpression ?? 'sin(pi*x/length)'`.
fn heat_initial(params: &Heat1DBlockParams) -> ScalarOrArray {
    match &params.initial_values {
        Some(v) if !v.is_empty() => ScalarOrArray::Array(v.clone()),
        _ => ScalarOrArray::Text(if params.initial_expression.is_empty() {
            "sin(pi*x/length)".to_string()
        } else {
            params.initial_expression.clone()
        }),
    }
}

fn universal_stationary_from_block(node: &BlockGraphNode) -> UniversalStationaryEntity {
    let inputs = node.inputs.clone().unwrap_or_default();
    let outputs = node.output.clone().map(|o| vec![o]).unwrap_or_default();
    UniversalStationaryEntity {
        id: node.id.clone(),
        kind: node.kind.clone(),
        role: Some(role_from_block_kind(&node.kind)),
        class_name: Some(node.kind.clone()),
        parameters: node
            .expression
            .clone()
            .map(|e| jobj("expression", JsonValue::String(e))),
        ports: Some(UniversalPorts { inputs: Some(inputs), outputs: Some(outputs) }),
        position: None,
        metadata: None,
    }
}

fn universal_edge_from_block(edge: &BlockGraphEdge, index: usize) -> UniversalGraphEdge {
    UniversalGraphEdge {
        id: format!("edge:{}:{}->{}", index, edge.from, edge.to),
        from: UniversalPortRef { entity_id: edge.from.clone(), port: edge.from_channel.clone() },
        to: UniversalPortRef { entity_id: edge.to.clone(), port: edge.to_channel.clone() },
        moving_entity: edge.signal.clone(),
        delay_ticks: None,
        transform: None,
        metadata: None,
    }
}

fn original_from_input(input: &MathEquationInputParams) -> UniversalOriginalInput {
    if let Some(eq) = &input.equation {
        let content_type = match input.format {
            EquationInputFormat::Latex => "application/x-latex",
            EquationInputFormat::Xml => "application/xml",
            _ => "application/json",
        };
        return UniversalOriginalInput {
            format: input_format_from_eq(input.format),
            content: Some(eq.clone()),
            uri: None,
            content_type: Some(content_type.to_string()),
            language: None,
            captured_at: None,
            metadata: None,
        };
    }
    // `JSON.stringify(input.ode ?? input.heat1d ?? input, null, 2)`.
    let content = if let Some(o) = &input.ode {
        json_stringify(&JsonValue::Object(o.clone()))
    } else if let Some(h) = &input.heat1d {
        json_stringify(&JsonValue::Object(h.clone()))
    } else {
        json_stringify(&JsonValue::Object(params_to_json(input)))
    };
    UniversalOriginalInput {
        format: UniversalInputFormat::Json,
        content: Some(content),
        uri: None,
        content_type: Some("application/json".to_string()),
        language: None,
        captured_at: None,
        metadata: None,
    }
}

fn role_from_block_kind(kind: &str) -> UniversalEntityRole {
    if kind.contains("integrator") {
        UniversalEntityRole::Integrator
    } else if kind.contains("boundary") {
        UniversalEntityRole::Boundary
    } else if kind.contains("laplacian") {
        UniversalEntityRole::Processor
    } else if kind.contains("expression") {
        UniversalEntityRole::Processor
    } else if kind.contains("source") {
        UniversalEntityRole::Source
    } else if kind.contains("sink") {
        UniversalEntityRole::Sink
    } else {
        UniversalEntityRole::Processor
    }
}

fn source_endpoints(nodes: &[BlockGraphNode], normalized: &Normalized) -> Vec<UniversalEndpointSpec> {
    match normalized {
        Normalized::Ode(params) => params
            .states
            .iter()
            .map(|s| UniversalEndpointSpec {
                id: format!("source:initial:{}", s.name),
                entity_id: format!("integrator:{}", s.name),
                port: Some("out".to_string()),
                role: Some("initial-condition".to_string()),
                variable: Some(s.name.clone()),
                value: Some(JsonValue::Number(s.initial)),
                record: None,
            })
            .collect(),
        Normalized::Heat1d(_) => nodes
            .iter()
            .filter(|n| n.kind == "constant-boundary")
            .map(|n| UniversalEndpointSpec {
                id: format!("source:{}", n.id),
                entity_id: n.id.clone(),
                port: Some("out".to_string()),
                role: Some("boundary-condition".to_string()),
                variable: None,
                value: None,
                record: None,
            })
            .collect(),
    }
}

fn sink_endpoints(nodes: &[BlockGraphNode]) -> Vec<UniversalEndpointSpec> {
    nodes
        .iter()
        .filter(|n| n.kind.contains("integrator") || n.kind == "constant-boundary")
        .map(|n| UniversalEndpointSpec {
            id: format!("sink:trace:{}", n.id),
            entity_id: n.id.clone(),
            port: Some("out".to_string()),
            role: Some("trace-recorder".to_string()),
            variable: None,
            value: None,
            record: Some(true),
        })
        .collect()
}

fn method_from_normalized(normalized: &Normalized) -> Option<String> {
    match normalized {
        Normalized::Ode(p) => Some(integrator_str(p.method).to_string()),
        Normalized::Heat1d(_) => Some("explicit-euler-laplacian".to_string()),
    }
}

fn time_from_ode(params: &ODEBlockSystemParams) -> UniversalTimeSpec {
    UniversalTimeSpec {
        t0: Some(params.t0),
        t1: Some(params.t1),
        dt: Some(params.dt),
        steps: Some(((params.t1 - params.t0) / params.dt).round()),
    }
}

fn time_from_heat(params: &Heat1DBlockParams) -> UniversalTimeSpec {
    UniversalTimeSpec {
        t0: Some(params.t0),
        t1: Some(params.t1),
        dt: Some(params.dt),
        steps: Some(((params.t1 - params.t0) / params.dt).round()),
    }
}

// =============================================================================
// Small helpers.
// =============================================================================

fn push(
    checks: &mut Vec<ValidationCheck>,
    name: &str,
    passed: bool,
    observed: Option<String>,
    expected: Option<String>,
) {
    checks.push(ValidationCheck {
        name: name.to_string(),
        passed,
        observed,
        expected,
        group: Some("universal-model".to_string()),
        details: None,
    });
}

fn is_non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_non_empty_opt(value: &Option<String>) -> bool {
    value.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false)
}

fn unique(values: &[String]) -> bool {
    let set: HashSet<&String> = values.iter().collect();
    set.len() == values.len()
}

fn duplicate(values: &[String]) -> Option<String> {
    let mut seen: HashSet<&String> = HashSet::new();
    for v in values {
        if seen.contains(v) {
            return Some(v.clone());
        }
        seen.insert(v);
    }
    None
}

/// Constants iterated in key order (TS `Object.entries` is insertion-ordered;
/// HashMap is not, so we sort for determinism — see module PORT NOTE).
fn sorted_constants(constants: &std::collections::HashMap<String, f64>) -> Vec<(String, f64)> {
    let mut entries: Vec<(String, f64)> = constants.iter().map(|(k, v)| (k.clone(), *v)).collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn ind_var(id: &str) -> UniversalMathVariable {
    UniversalMathVariable {
        id: id.to_string(),
        role: UniversalVariableRole::Independent,
        initial: None,
        units: None,
        domain: None,
        description: None,
    }
}

fn state_var(id: &str, initial: ScalarOrArray) -> UniversalMathVariable {
    UniversalMathVariable {
        id: id.to_string(),
        role: UniversalVariableRole::State,
        initial: Some(initial),
        units: None,
        domain: None,
        description: None,
    }
}

fn param(id: &str, value: ScalarOrArray) -> UniversalMathParameter {
    UniversalMathParameter { id: id.to_string(), value, units: None, description: None }
}

fn jobj(key: &str, value: JsonValue) -> JsonObject {
    let mut o = JsonObject::new();
    o.insert(key.to_string(), value);
    o
}

fn math_signal_schema() -> JsonObject {
    let mut o = JsonObject::new();
    o.insert("value".to_string(), JsonValue::String("number".to_string()));
    o.insert("time".to_string(), JsonValue::String("number".to_string()));
    o.insert("tick".to_string(), JsonValue::String("integer".to_string()));
    o.insert("sourceId".to_string(), JsonValue::String("string".to_string()));
    o.insert("channel".to_string(), JsonValue::String("string".to_string()));
    o
}

/// Best-effort serialization of a `MathEquationInputParams` to a JSON object,
/// used only by `originalFromInput`'s `?? input` fallback path.
fn params_to_json(input: &MathEquationInputParams) -> JsonObject {
    let mut o = JsonObject::new();
    o.insert("format".to_string(), JsonValue::String(eq_format_str(input.format).to_string()));
    if let Some(k) = input.kind {
        o.insert("kind".to_string(), JsonValue::String(problem_kind_str(k).to_string()));
    }
    if let Some(eq) = &input.equation {
        o.insert("equation".to_string(), JsonValue::String(eq.clone()));
    }
    if let Some(v) = input.t0 {
        o.insert("t0".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = input.t1 {
        o.insert("t1".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = input.dt {
        o.insert("dt".to_string(), JsonValue::Number(v));
    }
    o
}

/// Compact JSON serialization (stand-in for the private `JsonValue::to_json`).
fn json_stringify(v: &JsonValue) -> String {
    match v {
        JsonValue::Undefined | JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if n.is_finite() {
                n.to_string()
            } else {
                "null".to_string()
            }
        }
        JsonValue::String(s) => json_quote(s),
        JsonValue::Array(items) => {
            let inner: Vec<String> = items.iter().map(json_stringify).collect();
            format!("[{}]", inner.join(","))
        }
        JsonValue::Object(o) => {
            let inner: Vec<String> = o
                .keys()
                .map(|k| format!("{}:{}", json_quote(k), json_stringify(o.get(k).unwrap())))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

// =============================================================================
// Enum ↔ string mappings.
// =============================================================================

fn model_kind_str(k: UniversalModelKind) -> &'static str {
    match k {
        UniversalModelKind::Ode => "ode",
        UniversalModelKind::Pde => "pde",
        UniversalModelKind::Optimization => "optimization",
        UniversalModelKind::NetworkFlow => "network-flow",
        UniversalModelKind::TrafficFlow => "traffic-flow",
        UniversalModelKind::Queueing => "queueing",
        UniversalModelKind::Agent => "agent",
        UniversalModelKind::Custom => "custom",
    }
}

fn input_format_str(f: UniversalInputFormat) -> &'static str {
    match f {
        UniversalInputFormat::Json => "json",
        UniversalInputFormat::Latex => "latex",
        UniversalInputFormat::Xml => "xml",
        UniversalInputFormat::Text => "text",
        UniversalInputFormat::Manual => "manual",
    }
}

fn input_format_from_eq(f: EquationInputFormat) -> UniversalInputFormat {
    match f {
        EquationInputFormat::Json => UniversalInputFormat::Json,
        EquationInputFormat::Latex => UniversalInputFormat::Latex,
        EquationInputFormat::Xml => UniversalInputFormat::Xml,
    }
}

fn eq_format_str(f: EquationInputFormat) -> &'static str {
    match f {
        EquationInputFormat::Json => "json",
        EquationInputFormat::Latex => "latex",
        EquationInputFormat::Xml => "xml",
    }
}

fn problem_kind_str(k: EquationProblemKind) -> &'static str {
    match k {
        EquationProblemKind::Ode => "ode",
        EquationProblemKind::Heat1d => "heat1d",
    }
}

fn integrator_str(m: IntegratorMethod) -> &'static str {
    match m {
        IntegratorMethod::Euler => "euler",
        IntegratorMethod::Trapezoid => "trapezoid",
    }
}

// =============================================================================
// Tests.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::math_equation_input::{MathEquationNetwork, ODEStateSpec};
    use crate::des::general::des_spec::JsonObject as JObj;

    fn ode_params() -> ODEBlockSystemParams {
        ODEBlockSystemParams {
            states: vec![ODEStateSpec { name: "x".to_string(), initial: 1.0, derivative: "-x".to_string() }],
            constants: std::collections::HashMap::new(),
            t0: 0.0,
            t1: 1.0,
            dt: 0.1,
            method: IntegratorMethod::Euler,
        }
    }

    #[test]
    fn schema_guard_recognizes_universal_doc() {
        let mut o = JObj::new();
        o.insert("$schema".to_string(), JsonValue::String(UNIVERSAL_MODEL_SPEC_SCHEMA.to_string()));
        assert!(is_universal_des_model_spec(&JsonValue::Object(o)));
        assert!(!is_universal_des_model_spec(&JsonValue::Null));
    }

    #[test]
    fn round_trip_from_math_equation_result() {
        // PORT NOTE: `math-blocks` is not yet ported, so `runMathEquationProblem`'s
        // stub returns an empty block graph. To exercise the universal builder we
        // hand-assemble a `MathEquationResult` with a non-empty network (one
        // integrator node) so that the final `assert` over the spec passes.
        let states = {
            let mut node = JObj::new();
            node.insert("name".to_string(), JsonValue::String("x".to_string()));
            node.insert("derivative".to_string(), JsonValue::String("-x".to_string()));
            node.insert("initial".to_string(), JsonValue::Number(1.0));
            vec![JsonValue::Object(node)]
        };
        let input = MathEquationInputParams {
            format: EquationInputFormat::Json,
            states: Some(states),
            ..Default::default()
        };
        let result = MathEquationResult {
            input_format: EquationInputFormat::Json,
            kind: EquationProblemKind::Ode,
            equation: None,
            normalized: Normalized::Ode(ode_params()),
            network: MathEquationNetwork {
                nodes: vec![BlockGraphNode {
                    id: "integrator:x".to_string(),
                    kind: "integrator".to_string(),
                    inputs: Some(vec![]),
                    output: Some("x".to_string()),
                    expression: None,
                }],
                edges: vec![],
            },
            ode: None,
            heat1d: None,
            validation: vec![],
        };
        let spec = universal_from_math_equation_result(&input, &result, UniversalFromOpts::default())
            .expect("from result");
        assert_eq!(spec.schema, UNIVERSAL_MODEL_SPEC_SCHEMA);
        assert_eq!(spec.math.kind, UniversalModelKind::Ode);
        assert_eq!(spec.solver.target_model, "math-equation");
        // The normalized payload round-trips back to a MathEquationInputParams.
        let back = universal_to_math_equation_input(&spec).expect("to input");
        assert_eq!(back.format, EquationInputFormat::Json);

        // And conversion to the registry envelope succeeds.
        let envelope = universal_to_des_model_spec(&spec).expect("to des spec");
        assert_eq!(envelope.model, "math-equation");
        assert_eq!(envelope.schema, DES_MODEL_SPEC_SCHEMA);
    }

    #[test]
    fn ode_math_spec_has_equation_per_state() {
        let math = math_from_ode_result(
            &MathEquationInputParams::default(),
            &ode_params(),
        );
        assert_eq!(math.kind, UniversalModelKind::Ode);
        assert_eq!(math.equations.len(), 1);
        assert_eq!(math.equations[0].rhs.as_deref(), Some("-x"));
        assert_eq!(math.state_variables.len(), 1);
    }

    #[test]
    fn validate_flags_missing_pieces() {
        // Hand-build a deliberately invalid spec (empty equations / entities).
        let spec = UniversalDESModelSpec {
            schema: UNIVERSAL_MODEL_SPEC_SCHEMA.to_string(),
            id: "x".to_string(),
            description: None,
            original_input: UniversalOriginalInput {
                format: UniversalInputFormat::Json,
                content: Some("{}".to_string()),
                uri: None,
                content_type: None,
                language: None,
                captured_at: None,
                metadata: None,
            },
            math: UniversalMathSpec {
                kind: UniversalModelKind::Ode,
                independent_variables: vec![],
                state_variables: vec![],
                parameters: None,
                equations: vec![],
                initial_conditions: None,
                boundary_conditions: None,
                constraints: None,
                objectives: None,
                domain: None,
                numerics: None,
                normalized: None,
            },
            des: UniversalDESNetworkSpec {
                time: None,
                stationary_entities: vec![],
                moving_entities: vec![],
                graph: UniversalGraph { nodes: vec![], edges: vec![] },
                sources: None,
                sinks: None,
                observability: None,
            },
            solver: UniversalSolverSpec {
                target_model: "math-equation".to_string(),
                method: None,
                options: None,
            },
            runtime: None,
            metadata: None,
        };
        let checks = validate_universal_des_model_spec(&spec);
        assert!(checks.iter().any(|c| c.name == "math-equations-non-empty" && !c.passed));
        assert!(assert_universal_des_model_spec(&spec).is_err());
    }
}
