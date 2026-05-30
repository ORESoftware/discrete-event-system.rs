//! Port of `src/des/general/math-equation-input.ts` — module
//! `des::general::math_equation_input`.
//!
//! Equation input normalizer for math-block DES models. User-facing input can
//! be structured JSON, constrained LaTeX, or a tiny XML dialect. This module
//! converts those formats into the existing stationary math-block ODE/PDE model
//! parameters, runs the numerical solver, and returns the generated node/edge
//! network.
//!
//! Conversion notes from the TS source:
//!   * `EquationInputFormat` / `EquationProblemKind` string unions become
//!     `Copy` enums; `normalizeMathEquationProblem`'s `{kind, params}` union
//!     becomes the [`Normalized`] enum matched with `match`.
//!   * `unknown` / `Record<string, unknown>` inputs become
//!     [`crate::des::general::des_spec::JsonValue`] /
//!     [`crate::des::general::des_spec::JsonObject`] (the crate's `serde`-free
//!     JSON value type). The `(params as unknown as Record)` fallback is
//!     modelled by overlaying the optional `ode`/`heat1d` sub-object on top of a
//!     synthetic record built from the typed top-level fields.
//!   * `Preconditions` throws become `Result<_, PreconditionError>` propagated
//!     with `?`, per the migration rules; the `tokenizeExpression`
//!     `throw new Error` for an unsupported character is mapped to the same
//!     `PreconditionError` channel so the whole pipeline returns one error type.
//!   * The hand-rolled LaTeX/XML scanners (`textBetween` / `parseAttrs` /
//!     `replaceFractions` / ...) are ported as small char/byte scanners rather
//!     than regex, since `regex` is not a dependency of this crate.
//!
//! PORT NOTE: this file builds on `general/math-blocks`, which is NOT yet ported
//! to the Rust crate (see `src/des/test/math_blocks_test.rs`). The math-block
//! types it imports (`ODEBlockSystemParams`, `Heat1DBlockParams`,
//! `BlockGraphNode`, `runODEBlockSystem`, ...) are therefore defined here as
//! local placeholders, and the two `run_*_block_*` solvers are stubs that
//! return an empty block graph plus a single structural validation check. All
//! of the format-normalization logic above them is a faithful 1:1 port. When
//! `math-blocks` lands, these locals should be replaced with
//! `use crate::des::general::math_blocks::{...}`.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};
use crate::des::general::des_spec::{JsonObject, JsonValue};
use crate::des::general::expr::{evaluate, parse, Env};

/// Result alias for the precondition-bearing pipeline (TS `throw` → `Result`).
type R<T> = Result<T, PreconditionError>;

/// Numeric symbol table (TS `Record<string, number>`).
type NumMap = HashMap<String, f64>;

// =============================================================================
// PLACEHOLDER math-block surface (stand-in for the unported `math-blocks`).
// =============================================================================

/// Integrator selector (TS `IntegratorMethod = 'euler' | 'trapezoid'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IntegratorMethod {
    #[default]
    Euler,
    Trapezoid,
}

/// A single ODE state (TS `ODEStateSpec`).
#[derive(Clone, Debug, PartialEq)]
pub struct ODEStateSpec {
    pub name: String,
    pub initial: f64,
    pub derivative: String,
}

/// ODE system parameters (TS `ODEBlockSystemParams`).
#[derive(Clone, Debug)]
pub struct ODEBlockSystemParams {
    pub states: Vec<ODEStateSpec>,
    pub constants: NumMap,
    pub t0: f64,
    pub t1: f64,
    pub dt: f64,
    pub method: IntegratorMethod,
}

/// 1-D heat equation parameters (TS `Heat1DBlockParams`).
#[derive(Clone, Debug)]
pub struct Heat1DBlockParams {
    pub cells: f64,
    pub length: f64,
    pub alpha: f64,
    pub t0: f64,
    pub t1: f64,
    pub dt: f64,
    pub constants: NumMap,
    pub initial_expression: String,
    pub initial_values: Option<Vec<f64>>,
    pub left_boundary: Option<f64>,
    pub right_boundary: Option<f64>,
}

/// A node in the generated block graph (TS `BlockGraphNode`). Field set mirrors
/// the (unported) math-blocks node so downstream consumers such as
/// `universal_model_spec` can read `inputs`/`output`/`expression`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockGraphNode {
    pub id: String,
    pub kind: String,
    /// PORT NOTE: the TS node's `inputs` is `string[] | Record<...>`; modelled
    /// here as the simple list-of-channel-names case.
    pub inputs: Option<Vec<String>>,
    pub output: Option<String>,
    pub expression: Option<String>,
}

/// An edge in the generated block graph (TS `BlockGraphEdge`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockGraphEdge {
    pub from: String,
    pub to: String,
    pub from_channel: String,
    pub to_channel: String,
    pub signal: String,
}

/// A validation check entry (TS `{name; passed; group?}`).
#[derive(Clone, Debug, PartialEq)]
pub struct EquationValidationCheck {
    pub name: String,
    pub passed: bool,
    pub group: Option<String>,
}

/// ODE solver result (TS `ODEBlockSystemResult`, trimmed to the fields used).
#[derive(Clone, Debug, Default)]
pub struct ODEBlockSystemResult {
    pub block_graph: Vec<BlockGraphNode>,
    pub block_graph_edges: Vec<BlockGraphEdge>,
    pub validation: Vec<EquationValidationCheck>,
}

/// Heat1D solver result (TS `Heat1DBlockResult`, trimmed to the fields used).
#[derive(Clone, Debug, Default)]
pub struct Heat1DBlockResult {
    pub block_graph: Vec<BlockGraphNode>,
    pub block_graph_edges: Vec<BlockGraphEdge>,
    pub validation: Vec<EquationValidationCheck>,
}

/// PORT NOTE: the real numerical integration lives in the unported `math-blocks`
/// module. This placeholder returns an empty block graph and a single passing
/// structural check so the rest of the pipeline can be exercised.
fn run_ode_block_system(
    _params: &ODEBlockSystemParams,
    _logger: Option<&dyn EquationLogger>,
) -> ODEBlockSystemResult {
    ODEBlockSystemResult {
        block_graph: Vec::new(),
        block_graph_edges: Vec::new(),
        validation: vec![EquationValidationCheck {
            name: "ode-block-system-placeholder".to_string(),
            passed: true,
            group: Some("math-blocks".to_string()),
        }],
    }
}

/// PORT NOTE: placeholder for the unported `runHeat1DBlockGrid` (see above).
fn run_heat1d_block_grid(
    _params: &Heat1DBlockParams,
    _logger: Option<&dyn EquationLogger>,
) -> Heat1DBlockResult {
    Heat1DBlockResult {
        block_graph: Vec::new(),
        block_graph_edges: Vec::new(),
        validation: vec![EquationValidationCheck {
            name: "heat1d-block-grid-placeholder".to_string(),
            passed: true,
            group: Some("math-blocks".to_string()),
        }],
    }
}

// =============================================================================
// Public types.
// =============================================================================

/// Input encoding (TS `EquationInputFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EquationInputFormat {
    #[default]
    Json,
    Latex,
    Xml,
}

/// Problem family (TS `EquationProblemKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquationProblemKind {
    Ode,
    Heat1d,
}

/// Optional structured logger (TS `logger?: {log(event): void}`).
pub trait EquationLogger {
    fn log(&self, event: EquationLogEvent);
}

/// A structured log event (TS open-ended `Record<string, unknown>` payload,
/// flattened to stringified `fields`).
#[derive(Clone, Debug, Default)]
pub struct EquationLogEvent {
    pub kind: String,
    pub level: Option<String>,
    pub fields: HashMap<String, String>,
}

/// Normalizer input (TS `MathEquationInputParams`). All optional fields map to
/// `Option`; `Record`/`unknown` fields map to `JsonObject`/`JsonValue`.
#[derive(Clone, Debug, Default)]
pub struct MathEquationInputParams {
    pub format: EquationInputFormat,
    pub kind: Option<EquationProblemKind>,
    pub equation: Option<String>,
    pub ode: Option<JsonObject>,
    pub heat1d: Option<JsonObject>,
    pub states: Option<Vec<JsonValue>>,
    pub constants: Option<JsonObject>,
    pub initial: Option<JsonObject>,
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub method: Option<IntegratorMethod>,
    pub cells: Option<f64>,
    pub length: Option<f64>,
    pub alpha: Option<f64>,
    pub initial_expression: Option<String>,
    pub initial_values: Option<Vec<f64>>,
    pub left_boundary: Option<f64>,
    pub right_boundary: Option<f64>,
}

/// Generated block network (TS `MathEquationNetwork`).
#[derive(Clone, Debug, Default)]
pub struct MathEquationNetwork {
    pub nodes: Vec<BlockGraphNode>,
    pub edges: Vec<BlockGraphEdge>,
}

/// Normalized parameters, either ODE or heat1d (TS
/// `ODEBlockSystemParams | Heat1DBlockParams`, and the
/// `{kind, params}` union returned by `normalizeMathEquationProblem`).
#[derive(Clone, Debug)]
pub enum Normalized {
    Ode(ODEBlockSystemParams),
    Heat1d(Heat1DBlockParams),
}

impl Normalized {
    pub fn kind(&self) -> EquationProblemKind {
        match self {
            Normalized::Ode(_) => EquationProblemKind::Ode,
            Normalized::Heat1d(_) => EquationProblemKind::Heat1d,
        }
    }
}

/// Full run result (TS `MathEquationResult`).
#[derive(Clone, Debug)]
pub struct MathEquationResult {
    pub input_format: EquationInputFormat,
    pub kind: EquationProblemKind,
    pub equation: Option<String>,
    pub normalized: Normalized,
    pub network: MathEquationNetwork,
    pub ode: Option<ODEBlockSystemResult>,
    pub heat1d: Option<Heat1DBlockResult>,
    pub validation: Vec<EquationValidationCheck>,
}

// =============================================================================
// Static tables (TS module-level `FUNCTIONS` set + `GREEK` map).
// =============================================================================

const FUNCTIONS: [&str; 14] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "ln",
    "sqrt", "abs",
];

fn is_function(name: &str) -> bool {
    FUNCTIONS.contains(&name)
}

/// LaTeX greek-command → identifier map (insertion order matters, mirroring the
/// TS object literal).
const GREEK: [(&str, &str); 13] = [
    ("\\alpha", "alpha"),
    ("\\beta", "beta"),
    ("\\gamma", "gamma"),
    ("\\delta", "delta"),
    ("\\epsilon", "epsilon"),
    ("\\varepsilon", "epsilon"),
    ("\\theta", "theta"),
    ("\\lambda", "lambda"),
    ("\\mu", "mu"),
    ("\\sigma", "sigma"),
    ("\\tau", "tau"),
    ("\\omega", "omega"),
    ("\\pi", "pi"),
];

// =============================================================================
// Public entry points.
// =============================================================================

/// TS `runMathEquationProblem`.
pub fn run_math_equation_problem(
    params: &MathEquationInputParams,
    logger: Option<&dyn EquationLogger>,
) -> R<MathEquationResult> {
    // `console.debug(...)` omitted (diagnostic side-effect).
    let normalized = normalize_math_equation_problem(params)?;
    if let Some(l) = logger {
        let mut fields = HashMap::new();
        fields.insert("format".to_string(), format_str(params.format).to_string());
        fields.insert(
            "problemKind".to_string(),
            kind_str(normalized.kind()).to_string(),
        );
        l.log(EquationLogEvent {
            kind: "math-equation-normalized".to_string(),
            level: Some("info".to_string()),
            fields,
        });
    }
    match normalized {
        Normalized::Ode(p) => {
            let ode = run_ode_block_system(&p, logger);
            let network = MathEquationNetwork {
                nodes: ode.block_graph.clone(),
                edges: ode.block_graph_edges.clone(),
            };
            let validation = ode.validation.clone();
            let result = MathEquationResult {
                input_format: params.format,
                kind: EquationProblemKind::Ode,
                equation: params.equation.clone(),
                normalized: Normalized::Ode(p),
                network,
                ode: Some(ode),
                heat1d: None,
                validation,
            };
            Ok(result)
        }
        Normalized::Heat1d(p) => {
            let heat = run_heat1d_block_grid(&p, logger);
            let network = MathEquationNetwork {
                nodes: heat.block_graph.clone(),
                edges: heat.block_graph_edges.clone(),
            };
            let validation = heat.validation.clone();
            let result = MathEquationResult {
                input_format: params.format,
                kind: EquationProblemKind::Heat1d,
                equation: params.equation.clone(),
                normalized: Normalized::Heat1d(p),
                network,
                ode: None,
                heat1d: Some(heat),
                validation,
            };
            Ok(result)
        }
    }
}

/// TS `normalizeMathEquationProblem`.
pub fn normalize_math_equation_problem(params: &MathEquationInputParams) -> R<Normalized> {
    // PORT NOTE: the TS `format` membership check is structurally guaranteed by
    // the `EquationInputFormat` enum, so it is dropped here.
    let kind = infer_problem_kind(params);
    match params.format {
        EquationInputFormat::Json => Ok(match kind {
            EquationProblemKind::Ode => Normalized::Ode(normalize_json_ode(params)?),
            EquationProblemKind::Heat1d => Normalized::Heat1d(normalize_json_heat(params)?),
        }),
        EquationInputFormat::Latex => {
            let ok = params
                .equation
                .as_deref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            Preconditions::check(
                "MathEquationInput",
                "equation",
                "be a non-empty LaTeX string",
                ok,
                params.equation.clone(),
            )?;
            Ok(match kind {
                EquationProblemKind::Ode => Normalized::Ode(normalize_latex_ode(params)?),
                EquationProblemKind::Heat1d => Normalized::Heat1d(normalize_latex_heat(params)?),
            })
        }
        EquationInputFormat::Xml => {
            let ok = params
                .equation
                .as_deref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            Preconditions::check(
                "MathEquationInput",
                "equation",
                "be a non-empty XML string",
                ok,
                params.equation.clone(),
            )?;
            Ok(match kind {
                EquationProblemKind::Ode => Normalized::Ode(normalize_xml_ode(params)?),
                EquationProblemKind::Heat1d => Normalized::Heat1d(normalize_xml_heat(params)?),
            })
        }
    }
}

fn infer_problem_kind(params: &MathEquationInputParams) -> EquationProblemKind {
    if let Some(k) = params.kind {
        // PORT NOTE: the `'be ode or heat1d'` check is total over the enum.
        return k;
    }
    if params.format == EquationInputFormat::Json {
        if params.heat1d.is_some() {
            return EquationProblemKind::Heat1d;
        }
        return EquationProblemKind::Ode;
    }
    let equation = params.equation.as_deref().unwrap_or("");
    if params.format == EquationInputFormat::Xml {
        let root = root_name(equation);
        if matches!(root.as_deref(), Some("heat1d") | Some("pde")) {
            return EquationProblemKind::Heat1d;
        }
        return EquationProblemKind::Ode;
    }
    // TS `/\\partial|partial/` — both alternatives contain the substring
    // "partial".
    if equation.contains("partial") {
        EquationProblemKind::Heat1d
    } else {
        EquationProblemKind::Ode
    }
}

// =============================================================================
// JSON normalization.
// =============================================================================

fn normalize_json_ode(params: &MathEquationInputParams) -> R<ODEBlockSystemParams> {
    let src = src_overlay(params, params.ode.as_ref());
    let constants = merge_constants(&[
        params.constants.as_ref(),
        object_record(src.get("constants")),
    ])?;
    let initial = merge_initial(&[params.initial.as_ref(), object_record(src.get("initial"))])?;
    let states_raw = array_value(src.get("states"), "MathEquationInput.states")?;
    let states = states_raw
        .iter()
        .enumerate()
        .map(|(i, raw)| state_from_json(raw, i, &initial, &constants))
        .collect::<R<Vec<_>>>()?;
    let t0 = number_or_default(
        src.get("t0"),
        params.t0.unwrap_or(0.0),
        "MathEquationInput.t0",
    )?;
    let t1 = number_or_default(
        src.get("t1"),
        params.t1.unwrap_or(1.0),
        "MathEquationInput.t1",
    )?;
    let dt = if is_absent(src.get("dt")) {
        match params.dt {
            Some(d) => d,
            None => default_dt(t0, t1, 100.0)?,
        }
    } else {
        number_or_default(src.get("dt"), 0.0, "MathEquationInput.dt")?
    };
    let method = method_or_default(
        src.get("method"),
        params.method.unwrap_or(IntegratorMethod::Euler),
    )?;
    Ok(ODEBlockSystemParams {
        states,
        constants,
        t0,
        t1,
        dt,
        method,
    })
}

fn normalize_json_heat(params: &MathEquationInputParams) -> R<Heat1DBlockParams> {
    let src = src_overlay(params, params.heat1d.as_ref());
    let constants = merge_constants(&[
        params.constants.as_ref(),
        object_record(src.get("constants")),
    ])?;
    let cells = integer_or_default(
        src.get("cells"),
        params.cells.unwrap_or(31.0),
        "MathEquationInput.cells",
    )?;
    let length = number_or_default(
        src.get("length"),
        params.length.unwrap_or(1.0),
        "MathEquationInput.length",
    )?;
    let alpha_fallback = params
        .alpha
        .or(constants.get("alpha").copied())
        .unwrap_or(0.01);
    let alpha = number_or_default(src.get("alpha"), alpha_fallback, "MathEquationInput.alpha")?;
    let t0 = number_or_default(
        src.get("t0"),
        params.t0.unwrap_or(0.0),
        "MathEquationInput.t0",
    )?;
    let t1 = number_or_default(
        src.get("t1"),
        params.t1.unwrap_or(1.0),
        "MathEquationInput.t1",
    )?;
    let dt = if is_absent(src.get("dt")) {
        match params.dt {
            Some(d) => d,
            None => stable_heat_dt(t0, t1, cells, length, alpha)?,
        }
    } else {
        number_or_default(src.get("dt"), 0.0, "MathEquationInput.dt")?
    };
    let fallback_ie = params
        .initial_expression
        .clone()
        .unwrap_or_else(|| "sin(pi*x/length)".to_string());
    let initial_expression = string_or_default(src.get("initialExpression"), &fallback_ie);
    let initial_values =
        numeric_array_or_undefined(src.get("initialValues"), "MathEquationInput.initialValues")?;
    let left_boundary =
        number_or_undefined(src.get("leftBoundary"), "MathEquationInput.leftBoundary")?;
    let right_boundary =
        number_or_undefined(src.get("rightBoundary"), "MathEquationInput.rightBoundary")?;
    Ok(Heat1DBlockParams {
        cells,
        length,
        alpha,
        t0,
        t1,
        dt,
        constants,
        initial_expression,
        initial_values,
        left_boundary,
        right_boundary,
    })
}

// =============================================================================
// LaTeX normalization.
// =============================================================================

fn normalize_latex_ode(params: &MathEquationInputParams) -> R<ODEBlockSystemParams> {
    let constants = merge_constants(&[params.constants.as_ref()])?;
    let initial = merge_initial(&[params.initial.as_ref()])?;
    let states = parse_latex_ode(
        params.equation.as_deref().unwrap_or(""),
        &initial,
        &constants,
    )?;
    let t0 = params.t0.unwrap_or(0.0);
    let t1 = params.t1.unwrap_or(1.0);
    let dt = match params.dt {
        Some(d) => d,
        None => default_dt(t0, t1, 100.0)?,
    };
    Ok(ODEBlockSystemParams {
        states,
        constants,
        t0,
        t1,
        dt,
        method: params.method.unwrap_or(IntegratorMethod::Euler),
    })
}

fn normalize_latex_heat(params: &MathEquationInputParams) -> R<Heat1DBlockParams> {
    let constants = merge_constants(&[params.constants.as_ref()])?;
    let cells = params.cells.unwrap_or(31.0);
    let length = params.length.unwrap_or(1.0);
    let alpha = params
        .alpha
        .or(constants.get("alpha").copied())
        .unwrap_or(0.01);
    let t0 = params.t0.unwrap_or(0.0);
    let t1 = params.t1.unwrap_or(1.0);
    let dt = match params.dt {
        Some(d) => d,
        None => stable_heat_dt(t0, t1, cells, length, alpha)?,
    };
    Ok(Heat1DBlockParams {
        cells,
        length,
        alpha,
        t0,
        t1,
        dt,
        constants,
        initial_expression: params
            .initial_expression
            .clone()
            .unwrap_or_else(|| "sin(pi*x/length)".to_string()),
        initial_values: params.initial_values.clone().filter(|v| !v.is_empty()),
        left_boundary: Some(params.left_boundary.unwrap_or(0.0)),
        right_boundary: Some(params.right_boundary.unwrap_or(0.0)),
    })
}

fn parse_latex_ode(equation: &str, initial: &NumMap, constants: &NumMap) -> R<Vec<ODEStateSpec>> {
    // Ordered map keyed by state name (later sets overwrite, order preserved).
    let mut states: Vec<(String, ODEStateSpec)> = Vec::new();
    let mut parsed_initials: NumMap = initial.clone();
    for statement in equation_statements(equation) {
        let mut it = statement.split('=');
        let lhs = it.next().unwrap_or("").trim().to_string();
        let rest: Vec<&str> = it.collect();
        if rest.is_empty() {
            continue;
        }
        let rhs = rest.join("=");
        let rhs = rhs.trim().to_string();
        if let Some(init_name) = initial_condition_name(&lhs) {
            let expr_str = expression_text(&rhs)?;
            let mut env: Env = constants.clone();
            for (k, v) in &parsed_initials {
                env.insert(k.clone(), *v);
            }
            let val = evaluate(&parse(&expr_str), &env);
            parsed_initials.insert(init_name, val);
            continue;
        }
        let combined = format!("{lhs}={rhs}");
        if let Some(parsed) = parse_derivative_equation(&combined, constants, &parsed_initials)? {
            let name = parsed.name.clone();
            if let Some(slot) = states.iter_mut().find(|(k, _)| *k == name) {
                slot.1 = parsed;
            } else {
                states.push((name, parsed));
            }
        }
    }
    let values: Vec<ODEStateSpec> = states.iter().map(|(_, v)| v.clone()).collect();
    Preconditions::non_empty("MathEquationInput", "latex.ode.derivatives", &values)?;
    Ok(values
        .into_iter()
        .map(|s| {
            let init = parsed_initials.get(&s.name).copied().unwrap_or(s.initial);
            ODEStateSpec {
                name: s.name,
                initial: init,
                derivative: s.derivative,
            }
        })
        .collect())
}

fn parse_derivative_equation(
    equation: &str,
    _constants: &NumMap,
    initial: &NumMap,
) -> R<Option<ODEStateSpec>> {
    let mut it = equation.split('=');
    let lhs = match it.next() {
        Some(s) => s.trim().to_string(),
        None => return Ok(None),
    };
    let rest: Vec<&str> = it.collect();
    if rest.is_empty() {
        return Ok(None);
    }
    let rhs = rest.join("=");
    let rhs = rhs.trim().to_string();
    let name = match derivative_state_name(&lhs) {
        Some(n) => n,
        None => return Ok(None),
    };
    let derivative = expression_text(&rhs)?;
    // TS calls `parse(derivative)` purely to validate parseability.
    let _ = parse(&derivative);
    Ok(Some(ODEStateSpec {
        initial: initial.get(&name).copied().unwrap_or(0.0),
        name,
        derivative,
    }))
}

fn derivative_state_name(lhs: &str) -> Option<String> {
    let compact = strip_whitespace(lhs);
    // \frac{dNAME}{dt}
    if let Some(rest) = compact.strip_prefix("\\frac{d") {
        if let Some(name) = rest.strip_suffix("}{dt}") {
            if is_ident(name) {
                return Some(name.to_string());
            }
        }
    }
    // \dot{NAME}
    if let Some(rest) = compact.strip_prefix("\\dot{") {
        if let Some(name) = rest.strip_suffix('}') {
            if is_ident(name) {
                return Some(name.to_string());
            }
        }
    }
    // NAME'
    if let Some(name) = compact.strip_suffix('\'') {
        if is_ident(name) {
            return Some(name.to_string());
        }
    }
    // dNAME/dt
    if let Some(rest) = compact.strip_prefix('d') {
        if let Some(name) = rest.strip_suffix("/dt") {
            if is_ident(name) {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn initial_condition_name(lhs: &str) -> Option<String> {
    let compact = strip_whitespace(lhs);
    for suffix in ["(0)", "(t0)"] {
        if let Some(name) = compact.strip_suffix(suffix) {
            if is_ident(name) {
                return Some(name.to_string());
            }
        }
    }
    None
}

// =============================================================================
// XML normalization.
// =============================================================================

fn normalize_xml_ode(params: &MathEquationInputParams) -> R<ODEBlockSystemParams> {
    let xml = safe_xml(params.equation.as_deref().unwrap_or(""))?;
    let attrs = root_attrs(&xml, "ode");
    let xml_consts = constants_from_xml(&xml)?;
    let constants = merge_constants(&[params.constants.as_ref(), Some(&xml_consts)])?;
    let initial = merge_initial(&[params.initial.as_ref()])?;
    let mut states: Vec<ODEStateSpec> = Vec::new();
    for (attrs_raw, body) in find_elements(&xml, "state") {
        let state_attrs = parse_attrs(&attrs_raw);
        let name = required_string(state_attrs.get("name"), "MathEquationInput.xml.state.name")?;
        let rhs_raw: Option<String> = attr_string(&state_attrs, "derivative")
            .or_else(|| attr_string(&state_attrs, "rhs"))
            .or_else(|| text_between(&body, "derivative"))
            .or_else(|| text_between(&body, "rhs"))
            .or_else(|| text_between(&body, "equation"));
        Preconditions::check(
            "MathEquationInput",
            &format!("xml.state.{name}.derivative"),
            "be present",
            rhs_raw.is_some(),
            None,
        )?;
        let init_fallback = initial.get(&name).copied().unwrap_or(0.0);
        let initial_value = number_or_default(
            state_attrs.get("initial"),
            init_fallback,
            &format!("MathEquationInput.xml.state.{name}.initial"),
        )?;
        let derivative = expression_text(&rhs_raw.unwrap_or_default())?;
        states.push(ODEStateSpec {
            name,
            initial: initial_value,
            derivative,
        });
    }
    for (_attrs_raw, body) in find_elements(&xml, "equation") {
        if let Some(parsed) = parse_derivative_equation(&xml_decode(&body), &constants, &initial)? {
            states.push(parsed);
        }
    }
    Preconditions::non_empty("MathEquationInput", "xml.ode.states", &states)?;
    let t0 = number_or_default(
        attrs.get("t0"),
        params.t0.unwrap_or(0.0),
        "MathEquationInput.xml.t0",
    )?;
    let t1 = number_or_default(
        attrs.get("t1"),
        params.t1.unwrap_or(1.0),
        "MathEquationInput.xml.t1",
    )?;
    let dt = if !is_absent(attrs.get("dt")) {
        number_or_default(attrs.get("dt"), 0.0, "MathEquationInput.xml.dt")?
    } else if let Some(d) = params.dt {
        d
    } else {
        default_dt(t0, t1, 100.0)?
    };
    let method = method_or_default(
        attrs.get("method"),
        params.method.unwrap_or(IntegratorMethod::Euler),
    )?;
    Ok(ODEBlockSystemParams {
        states,
        constants,
        t0,
        t1,
        dt,
        method,
    })
}

fn normalize_xml_heat(params: &MathEquationInputParams) -> R<Heat1DBlockParams> {
    let xml = safe_xml(params.equation.as_deref().unwrap_or(""))?;
    let attrs = root_attrs(&xml, "heat1d");
    let xml_consts = constants_from_xml(&xml)?;
    let constants = merge_constants(&[params.constants.as_ref(), Some(&xml_consts)])?;
    let cells = integer_or_default(
        attrs.get("cells"),
        params.cells.unwrap_or(31.0),
        "MathEquationInput.xml.cells",
    )?;
    let length = number_or_default(
        attrs.get("length"),
        params.length.unwrap_or(1.0),
        "MathEquationInput.xml.length",
    )?;
    let alpha_fallback = params
        .alpha
        .or(constants.get("alpha").copied())
        .unwrap_or(0.01);
    let alpha = number_or_default(
        attrs.get("alpha"),
        alpha_fallback,
        "MathEquationInput.xml.alpha",
    )?;
    let t0 = number_or_default(
        attrs.get("t0"),
        params.t0.unwrap_or(0.0),
        "MathEquationInput.xml.t0",
    )?;
    let t1 = number_or_default(
        attrs.get("t1"),
        params.t1.unwrap_or(1.0),
        "MathEquationInput.xml.t1",
    )?;
    let dt = if !is_absent(attrs.get("dt")) {
        number_or_default(attrs.get("dt"), 0.0, "MathEquationInput.xml.dt")?
    } else if let Some(d) = params.dt {
        d
    } else {
        stable_heat_dt(t0, t1, cells, length, alpha)?
    };
    let initial = text_between(&xml, "initial");
    let initial_expression = match &initial {
        Some(s) => expression_text(&xml_decode(s))?,
        None => params
            .initial_expression
            .clone()
            .unwrap_or_else(|| "sin(pi*x/length)".to_string()),
    };
    let initial_values = params.initial_values.clone().filter(|v| !v.is_empty());
    let lb_val = attr_or_param(&attrs, "leftBoundary", params.left_boundary);
    let left_boundary = Some(
        number_or_undefined(lb_val.as_ref(), "MathEquationInput.xml.leftBoundary")?.unwrap_or(0.0),
    );
    let rb_val = attr_or_param(&attrs, "rightBoundary", params.right_boundary);
    let right_boundary = Some(
        number_or_undefined(rb_val.as_ref(), "MathEquationInput.xml.rightBoundary")?.unwrap_or(0.0),
    );
    Ok(Heat1DBlockParams {
        cells,
        length,
        alpha,
        t0,
        t1,
        dt,
        constants,
        initial_expression,
        initial_values,
        left_boundary,
        right_boundary,
    })
}

/// `attrs.<key> ?? params.<field>` as a single `JsonValue` (string attr wins).
fn attr_or_param(attrs: &JsonObject, key: &str, param_value: Option<f64>) -> Option<JsonValue> {
    match attr_string(attrs, key) {
        Some(s) => Some(JsonValue::String(s)),
        None => param_value.map(JsonValue::Number),
    }
}

fn constants_from_xml(xml: &str) -> R<JsonObject> {
    let mut out = JsonObject::new();
    for (attrs_raw, _body) in find_elements(xml, "constant") {
        let attrs = parse_attrs(&attrs_raw);
        let name = required_string(attrs.get("name"), "MathEquationInput.xml.constant.name")?;
        let value = number_or_default(
            attrs.get("value"),
            0.0,
            &format!("MathEquationInput.xml.constant.{name}.value"),
        )?;
        out.insert(name, JsonValue::Number(value));
    }
    Ok(out)
}

fn safe_xml(xml: &str) -> R<String> {
    let lower = xml.to_ascii_lowercase();
    let ok = !(lower.contains("<!doctype") || lower.contains("<!entity"));
    Preconditions::check(
        "MathEquationInput",
        "xml",
        "not contain DOCTYPE or ENTITY declarations",
        ok,
        Some("blocked XML declaration".to_string()),
    )?;
    Ok(xml.trim().to_string())
}

fn root_name(xml: &str) -> Option<String> {
    let c: Vec<char> = xml.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '<' && i + 1 < c.len() && (c[i + 1].is_ascii_alphabetic() || c[i + 1] == '_') {
            let start = i + 1;
            let mut j = start + 1;
            while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '_' || c[j] == '-') {
                j += 1;
            }
            return Some(c[start..j].iter().collect());
        }
        i += 1;
    }
    None
}

fn root_attrs(xml: &str, fallback: &str) -> JsonObject {
    let root = root_name(xml).unwrap_or_else(|| fallback.to_string());
    let open = format!("<{root}");
    if let Some(rel) = xml.find(&open) {
        let after = rel + open.len();
        let boundary = match xml[after..].chars().next() {
            Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            None => true,
        };
        if boundary {
            if let Some(gt_rel) = xml[after..].find('>') {
                let raw = xml[after..after + gt_rel].trim_end_matches('/');
                return parse_attrs(raw);
            }
        }
    }
    JsonObject::new()
}

/// TS `parseAttrs`: scan `key="value"` / `key='value'` pairs. Values are stored
/// as `JsonValue::String` so the numeric/string coercers can be shared with the
/// JSON path.
fn parse_attrs(raw: &str) -> JsonObject {
    let mut attrs = JsonObject::new();
    let c: Vec<char> = raw.chars().collect();
    let n = c.len();
    let mut i = 0;
    while i < n {
        if !(c[i].is_ascii_alphabetic() || c[i] == '_') {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < n && (c[i].is_ascii_alphanumeric() || c[i] == '_' || c[i] == ':' || c[i] == '-') {
            i += 1;
        }
        let key: String = c[start..i].iter().collect();
        while i < n && c[i].is_whitespace() {
            i += 1;
        }
        if i >= n || c[i] != '=' {
            continue;
        }
        i += 1;
        while i < n && c[i].is_whitespace() {
            i += 1;
        }
        if i >= n || (c[i] != '"' && c[i] != '\'') {
            continue;
        }
        let quote = c[i];
        i += 1;
        let vstart = i;
        while i < n && c[i] != quote {
            i += 1;
        }
        let value: String = c[vstart..i].iter().collect();
        if i < n {
            i += 1;
        }
        attrs.insert(key, JsonValue::String(xml_decode(&value)));
    }
    attrs
}

fn text_between(raw: &str, tag: &str) -> Option<String> {
    find_elements(raw, tag)
        .into_iter()
        .next()
        .map(|(_, body)| body.trim().to_string())
}

/// Find all `<tag ...>body</tag>` (and self-closing `<tag .../>`) elements,
/// returning `(raw_attrs, body)` pairs. Stand-in for the TS `String.matchAll`
/// regex loops (`regex` is not a crate dependency).
///
/// PORT NOTE: this scanner approximates the TS regexes: attribute values that
/// contain `>` (or, for `<constant>`, `/`) are handled slightly more leniently
/// than the originals, which is harmless for the supported equation dialect.
fn find_elements(xml: &str, tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut from = 0usize;
    while let Some(rel) = xml[from..].find(&open) {
        let start = from + rel;
        let after = start + open.len();
        let is_boundary = match xml[after..].chars().next() {
            Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            None => true,
        };
        if !is_boundary {
            from = after;
            continue;
        }
        let gt_rel = match xml[after..].find('>') {
            Some(i) => i,
            None => break,
        };
        let gt = after + gt_rel;
        let attrs_raw = &xml[after..gt];
        if attrs_raw.trim_end().ends_with('/') {
            let attrs = attrs_raw.trim_end().trim_end_matches('/').to_string();
            out.push((attrs, String::new()));
            from = gt + 1;
            continue;
        }
        let body_start = gt + 1;
        let close_rel = match xml[body_start..].find(&close) {
            Some(i) => i,
            None => break,
        };
        let body = &xml[body_start..body_start + close_rel];
        out.push((attrs_raw.to_string(), body.to_string()));
        from = body_start + close_rel + close.len();
    }
    out
}

fn xml_decode(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

// =============================================================================
// Expression text → math-block expression syntax.
// =============================================================================

fn expression_text(raw: &str) -> R<String> {
    let decoded = xml_decode(raw);
    let decoded = decoded.trim();
    if looks_latex(decoded) {
        latex_to_expression(decoded)
    } else {
        insert_implicit_multiplication(&rewrite_math_aliases(decoded))
    }
}

/// TS `latexToExpression`.
pub fn latex_to_expression(input: &str) -> R<String> {
    let mut s = input.trim().to_string();
    s = s.replace("$$", "").replace('$', "");
    s = s.replace("\\left", "").replace("\\right", "");
    s = s.replace("\\,", "").replace('&', "");
    s = remove_latex_env(&s);
    s = replace_fractions(&s);
    for (from, to) in GREEK {
        s = s.replace(from, to);
    }
    s = s.replace("\\cdot", "*").replace("\\times", "*");
    s = s.replace("\\ln", "log");
    for fn_name in FUNCTIONS {
        let repl = if fn_name == "ln" { "log" } else { fn_name };
        s = replace_command(&s, fn_name, repl);
    }
    s = caret_braces(&s);
    s = subscript_braces(&s);
    s = s.replace('{', "(").replace('}', ")");
    s = rewrite_math_aliases(&s);
    insert_implicit_multiplication(&s)
}

fn replace_fractions(input: &str) -> String {
    let mut s = input.to_string();
    while let Some(idx) = s.find("\\frac") {
        let after = idx + "\\frac".len();
        let numerator = match read_braced(&s, after) {
            Some(b) => b,
            None => break,
        };
        let denominator = match read_braced(&s, numerator.end) {
            Some(b) => b,
            None => break,
        };
        let replacement = format!(
            "(({})/({}))",
            replace_fractions(&numerator.value),
            replace_fractions(&denominator.value)
        );
        s = format!("{}{}{}", &s[..idx], replacement, &s[denominator.end..]);
    }
    s
}

struct Braced {
    value: String,
    end: usize,
}

fn read_braced(s: &str, start: usize) -> Option<Braced> {
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    let mut depth = 0i32;
    let begin = i + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(Braced {
                        value: s[begin..i].to_string(),
                        end: i + 1,
                    });
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn insert_implicit_multiplication(input: &str) -> R<String> {
    let tokens = tokenize_expression(input)?;
    let mut out = String::new();
    for i in 0..tokens.len() {
        if i > 0 && needs_multiplication(&tokens[i - 1], &tokens[i]) {
            out.push('*');
        }
        out.push_str(&tokens[i].text);
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Num,
    Id,
    Op,
    Lparen,
    Rparen,
}

struct ExprToken {
    kind: TokenKind,
    text: String,
}

fn tokenize_expression(input: &str) -> R<Vec<ExprToken>> {
    let c: Vec<char> = input.chars().collect();
    let n = c.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch.is_ascii_digit() || ch == '.' {
            let mut j = i + 1;
            while j < n && (c[j].is_ascii_digit() || c[j] == '.') {
                j += 1;
            }
            if j < n && (c[j] == 'e' || c[j] == 'E') {
                j += 1;
                if j < n && (c[j] == '+' || c[j] == '-') {
                    j += 1;
                }
                while j < n && c[j].is_ascii_digit() {
                    j += 1;
                }
            }
            tokens.push(ExprToken {
                kind: TokenKind::Num,
                text: c[i..j].iter().collect(),
            });
            i = j;
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut j = i + 1;
            while j < n && (c[j].is_ascii_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            tokens.push(ExprToken {
                kind: TokenKind::Id,
                text: c[i..j].iter().collect(),
            });
            i = j;
            continue;
        }
        if ch == '(' {
            tokens.push(ExprToken {
                kind: TokenKind::Lparen,
                text: ch.to_string(),
            });
            i += 1;
            continue;
        }
        if ch == ')' {
            tokens.push(ExprToken {
                kind: TokenKind::Rparen,
                text: ch.to_string(),
            });
            i += 1;
            continue;
        }
        if matches!(ch, '+' | '-' | '*' | '/' | '^') {
            tokens.push(ExprToken {
                kind: TokenKind::Op,
                text: ch.to_string(),
            });
            i += 1;
            continue;
        }
        // TS `throw new Error(...)` for an unsupported character — surfaced via
        // the shared `PreconditionError` channel.
        return Err(PreconditionError::new(
            "MathEquationInput",
            "expression",
            "contain only supported characters",
            Some(format!(
                "unsupported expression character \"{ch}\" in {input}"
            )),
        ));
    }
    Ok(tokens)
}

fn needs_multiplication(a: &ExprToken, b: &ExprToken) -> bool {
    let a_ends = matches!(a.kind, TokenKind::Num | TokenKind::Id | TokenKind::Rparen);
    let b_starts = matches!(b.kind, TokenKind::Num | TokenKind::Id | TokenKind::Lparen);
    if !a_ends || !b_starts {
        return false;
    }
    if a.kind == TokenKind::Id && b.kind == TokenKind::Lparen && is_function(&a.text) {
        return false;
    }
    true
}

fn rewrite_math_aliases(input: &str) -> String {
    let s = replace_word(input, "ln", "log");
    let s = replace_word(&s, "PI", "pi");
    let s = replace_word(&s, "Pi", "pi");
    replace_word(&s, "euler", "e")
}

fn equation_statements(equation: &str) -> Vec<String> {
    let s = equation
        .replace("\\begin{cases}", "")
        .replace("\\end{cases}", "");
    let s = s.replace("\\\\", "\n").replace(';', "\n");
    s.split('\n')
        .map(|x| x.replace('$', "").replace('&', "").trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn looks_latex(s: &str) -> bool {
    const NEEDLES: [&str; 12] = [
        "\\frac",
        "\\dot",
        "\\partial",
        "\\alpha",
        "\\beta",
        "\\gamma",
        "\\lambda",
        "\\pi",
        "\\sin",
        "\\cos",
        "\\exp",
        "^{",
    ];
    NEEDLES.iter().any(|needle| s.contains(needle))
}

// =============================================================================
// JSON state coercion.
// =============================================================================

fn state_from_json(
    raw: &JsonValue,
    index: usize,
    initial: &NumMap,
    constants: &NumMap,
) -> R<ODEStateSpec> {
    let obj = match object_record(Some(raw)) {
        Some(o) => o,
        None => {
            return Err(PreconditionError::new(
                "MathEquationInput",
                &format!("states[{index}]"),
                "be an object",
                Some(json_to_string(raw)),
            ))
        }
    };
    let name = required_string(
        obj.get("name"),
        &format!("MathEquationInput.states[{index}].name"),
    )?;
    let derivative_raw = match first_defined(obj, &["derivative", "rhs", "equation"]) {
        Some(v) => v,
        None => {
            return Err(PreconditionError::new(
                "MathEquationInput",
                &format!("states[{index}].derivative"),
                "be present",
                Some(json_to_string(raw)),
            ))
        }
    };
    let init_fallback = initial.get(&name).copied().unwrap_or(0.0);
    let initial_value = number_or_default(
        obj.get("initial"),
        init_fallback,
        &format!("MathEquationInput.states[{index}].initial"),
    )?;
    let deriv_str = json_to_string(derivative_raw);
    let mut derivative = expression_text(&deriv_str)?;
    if deriv_str.contains('=') {
        let mut single = NumMap::new();
        single.insert(name.clone(), initial_value);
        derivative = match parse_derivative_equation(&deriv_str, constants, &single)? {
            Some(p) => p.derivative,
            None => {
                let after_eq = match deriv_str.find('=') {
                    Some(i) => &deriv_str[i + 1..],
                    None => "",
                };
                expression_text(after_eq)?
            }
        };
    }
    Ok(ODEStateSpec {
        name,
        initial: initial_value,
        derivative,
    })
}

// =============================================================================
// Symbol-table merging.
// =============================================================================

fn merge_constants(records: &[Option<&JsonObject>]) -> R<NumMap> {
    let mut out: NumMap = NumMap::new();
    out.insert("pi".to_string(), std::f64::consts::PI);
    out.insert("e".to_string(), std::f64::consts::E);
    for rec in records {
        if let Some(obj) = rec {
            for key in obj.keys() {
                let fallback = out.get(key).copied().unwrap_or(0.0);
                let value = number_or_default(
                    obj.get(key),
                    fallback,
                    &format!("MathEquationInput.constants.{key}"),
                )?;
                out.insert(key.clone(), value);
            }
        }
    }
    Ok(out)
}

fn merge_initial(records: &[Option<&JsonObject>]) -> R<NumMap> {
    let mut out: NumMap = NumMap::new();
    for rec in records {
        if let Some(obj) = rec {
            for key in obj.keys() {
                let value = number_or_default(
                    obj.get(key),
                    0.0,
                    &format!("MathEquationInput.initial.{key}"),
                )?;
                out.insert(key.clone(), value);
            }
        }
    }
    Ok(out)
}

// =============================================================================
// Coercion helpers (TS `*OrDefault` / `*OrUndefined`).
// =============================================================================

/// `(params as Record)` synthetic record: the typed top-level fields rendered as
/// JSON, with the optional `ode`/`heat1d` sub-object overlaid on top. This makes
/// every `src.X ?? params.X` read in the TS source a single `src.get("X")`.
fn src_overlay(p: &MathEquationInputParams, sub: Option<&JsonObject>) -> JsonObject {
    let mut base = params_as_record(p);
    if let Some(o) = sub {
        for key in o.keys() {
            if let Some(v) = o.get(key) {
                base.insert(key.clone(), v.clone());
            }
        }
    }
    base
}

fn params_as_record(p: &MathEquationInputParams) -> JsonObject {
    let mut r = JsonObject::new();
    if let Some(v) = &p.states {
        r.insert("states".to_string(), JsonValue::Array(v.clone()));
    }
    if let Some(v) = &p.constants {
        r.insert("constants".to_string(), JsonValue::Object(v.clone()));
    }
    if let Some(v) = &p.initial {
        r.insert("initial".to_string(), JsonValue::Object(v.clone()));
    }
    if let Some(v) = p.t0 {
        r.insert("t0".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = p.t1 {
        r.insert("t1".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = p.dt {
        r.insert("dt".to_string(), JsonValue::Number(v));
    }
    if let Some(m) = p.method {
        r.insert(
            "method".to_string(),
            JsonValue::String(method_str(m).to_string()),
        );
    }
    if let Some(v) = p.cells {
        r.insert("cells".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = p.length {
        r.insert("length".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = p.alpha {
        r.insert("alpha".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = &p.initial_expression {
        r.insert(
            "initialExpression".to_string(),
            JsonValue::String(v.clone()),
        );
    }
    if let Some(v) = &p.initial_values {
        r.insert(
            "initialValues".to_string(),
            JsonValue::Array(v.iter().map(|x| JsonValue::Number(*x)).collect()),
        );
    }
    if let Some(v) = p.left_boundary {
        r.insert("leftBoundary".to_string(), JsonValue::Number(v));
    }
    if let Some(v) = p.right_boundary {
        r.insert("rightBoundary".to_string(), JsonValue::Number(v));
    }
    r
}

fn object_record(value: Option<&JsonValue>) -> Option<&JsonObject> {
    match value {
        Some(JsonValue::Object(o)) => Some(o),
        _ => None,
    }
}

fn first_defined<'a>(obj: &'a JsonObject, keys: &[&str]) -> Option<&'a JsonValue> {
    for k in keys {
        if let Some(v) = obj.get(k) {
            if !matches!(v, JsonValue::Undefined) {
                return Some(v);
            }
        }
    }
    None
}

/// `value === undefined || value === null || value === ''` (exact empty string).
fn is_absent(v: Option<&JsonValue>) -> bool {
    matches!(v, None | Some(JsonValue::Undefined) | Some(JsonValue::Null))
        || matches!(v, Some(JsonValue::String(s)) if s.is_empty())
}

fn array_value(value: Option<&JsonValue>, param: &str) -> R<Vec<JsonValue>> {
    if let Some(JsonValue::Array(a)) = value {
        if !a.is_empty() {
            return Ok(a.clone());
        }
    }
    Err(PreconditionError::new(
        "MathEquationInput",
        param,
        "be a non-empty array",
        value.map(json_to_string),
    ))
}

fn required_string(value: Option<&JsonValue>, param: &str) -> R<String> {
    match value {
        Some(JsonValue::String(s)) if !s.trim().is_empty() => Ok(s.trim().to_string()),
        other => Err(PreconditionError::new(
            "MathEquationInput",
            param,
            "be a non-empty string",
            other.map(json_to_string),
        )),
    }
}

fn string_or_default(value: Option<&JsonValue>, fallback: &str) -> String {
    match value {
        Some(JsonValue::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
        _ => fallback.to_string(),
    }
}

fn number_or_default(value: Option<&JsonValue>, fallback: f64, param: &str) -> R<f64> {
    if is_absent(value) {
        return Ok(fallback);
    }
    let n = match value.unwrap() {
        JsonValue::Number(n) => *n,
        JsonValue::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        JsonValue::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else {
                t.parse::<f64>().unwrap_or(f64::NAN)
            }
        }
        _ => f64::NAN,
    };
    Preconditions::finite("MathEquationInput", param, n)?;
    Ok(n)
}

fn number_or_undefined(value: Option<&JsonValue>, param: &str) -> R<Option<f64>> {
    if is_absent(value) {
        return Ok(None);
    }
    Ok(Some(number_or_default(value, 0.0, param)?))
}

fn integer_or_default(value: Option<&JsonValue>, fallback: f64, param: &str) -> R<f64> {
    let n = number_or_default(value, fallback, param)?;
    Preconditions::integer("MathEquationInput", param, n)?;
    Ok(n)
}

fn method_or_default(value: Option<&JsonValue>, fallback: IntegratorMethod) -> R<IntegratorMethod> {
    if is_absent(value) {
        return Ok(fallback);
    }
    let s = match value.unwrap() {
        JsonValue::String(s) => s.clone(),
        other => json_to_string(other),
    };
    let ok = s == "euler" || s == "trapezoid";
    Preconditions::check(
        "MathEquationInput",
        "method",
        "be euler or trapezoid",
        ok,
        Some(s.clone()),
    )?;
    Ok(if s == "trapezoid" {
        IntegratorMethod::Trapezoid
    } else {
        IntegratorMethod::Euler
    })
}

fn numeric_array_or_undefined(value: Option<&JsonValue>, param: &str) -> R<Option<Vec<f64>>> {
    match value {
        None | Some(JsonValue::Undefined) | Some(JsonValue::Null) => return Ok(None),
        _ => {}
    }
    let arr = match value {
        Some(JsonValue::Array(a)) => a,
        other => {
            return Err(PreconditionError::new(
                "MathEquationInput",
                param,
                "be an array",
                other.map(json_to_string),
            ))
        }
    };
    if arr.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        out.push(number_or_default(Some(v), 0.0, &format!("{param}[{i}]"))?);
    }
    Ok(Some(out))
}

fn default_dt(t0: f64, t1: f64, steps: f64) -> R<f64> {
    Preconditions::check(
        "MathEquationInput",
        "time horizon",
        "satisfy t1 > t0",
        t1 > t0,
        Some(format!("{{t0:{t0},t1:{t1}}}")),
    )?;
    Ok((t1 - t0) / steps)
}

fn stable_heat_dt(t0: f64, t1: f64, cells: f64, length: f64, alpha: f64) -> R<f64> {
    Preconditions::integer_in_range("MathEquationInput", "cells", cells, 3.0, 100000.0)?;
    Preconditions::positive("MathEquationInput", "length", length)?;
    Preconditions::non_negative("MathEquationInput", "alpha", alpha)?;
    if alpha == 0.0 {
        return default_dt(t0, t1, 100.0);
    }
    let dx = length / (cells - 1.0);
    let target = 0.45 * dx * dx / alpha;
    let steps = ((t1 - t0) / target).ceil().max(1.0);
    Ok((t1 - t0) / steps)
}

// =============================================================================
// Small string helpers.
// =============================================================================

fn json_to_string(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Null => "null".to_string(),
        JsonValue::Undefined => "undefined".to_string(),
        other => format!("{other:?}"),
    }
}

fn attr_string(attrs: &JsonObject, key: &str) -> Option<String> {
    match attrs.get(key) {
        Some(JsonValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn strip_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Replace `word` with `repl` only at `\b` word boundaries (both sides).
fn replace_word(s: &str, word: &str, repl: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let w: Vec<char> = word.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if i + w.len() <= c.len() && c[i..i + w.len()] == w[..] {
            let before_ok = i == 0 || !is_word_char(c[i - 1]);
            let after = i + w.len();
            let after_ok = after >= c.len() || !is_word_char(c[after]);
            if before_ok && after_ok {
                out.push_str(repl);
                i += w.len();
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// Replace a LaTeX command `\name` (with a trailing `\b`) with `repl`.
fn replace_command(s: &str, name: &str, repl: &str) -> String {
    let pat: Vec<char> = std::iter::once('\\').chain(name.chars()).collect();
    let c: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if i + pat.len() <= c.len() && c[i..i + pat.len()] == pat[..] {
            let after = i + pat.len();
            let after_ok = after >= c.len() || !is_word_char(c[after]);
            if after_ok {
                out.push_str(repl);
                i += pat.len();
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// TS `s.replace(/\^\{([^{}]+)\}/g, '^($1)')`.
fn caret_braces(s: &str) -> String {
    brace_rewrite(s, '^', "^(", ")", |_| true)
}

/// TS `s.replace(/_\{([A-Za-z0-9]+)\}/g, '_$1')`.
fn subscript_braces(s: &str) -> String {
    brace_rewrite(s, '_', "_", "", |c| c.is_ascii_alphanumeric())
}

/// Rewrite `<prefix>{CONTENT}` into `<open>CONTENT<close>`, where CONTENT
/// contains no braces and every char satisfies `content_ok`.
fn brace_rewrite(
    s: &str,
    prefix: char,
    open: &str,
    close: &str,
    content_ok: impl Fn(char) -> bool,
) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == prefix && i + 1 < c.len() && c[i + 1] == '{' {
            let start = i + 2;
            let mut j = start;
            let mut ok = true;
            while j < c.len() && c[j] != '}' {
                if c[j] == '{' || !content_ok(c[j]) {
                    ok = false;
                    break;
                }
                j += 1;
            }
            if ok && j < c.len() && j > start {
                out.push_str(open);
                out.extend(c[start..j].iter());
                out.push_str(close);
                i = j + 1;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// Remove `\begin{...}` and `\end{...}` environment markers.
fn remove_latex_env(s: &str) -> String {
    let out = remove_command_brace(s, "\\begin");
    remove_command_brace(&out, "\\end")
}

fn remove_command_brace(s: &str, cmd: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let cmdc: Vec<char> = cmd.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if i + cmdc.len() < c.len() && c[i..i + cmdc.len()] == cmdc[..] && c[i + cmdc.len()] == '{'
        {
            let start = i + cmdc.len() + 1;
            let mut j = start;
            while j < c.len() && c[j] != '}' {
                j += 1;
            }
            if j < c.len() && j > start {
                i = j + 1;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn format_str(f: EquationInputFormat) -> &'static str {
    match f {
        EquationInputFormat::Json => "json",
        EquationInputFormat::Latex => "latex",
        EquationInputFormat::Xml => "xml",
    }
}

fn kind_str(k: EquationProblemKind) -> &'static str {
    match k {
        EquationProblemKind::Ode => "ode",
        EquationProblemKind::Heat1d => "heat1d",
    }
}

fn method_str(m: IntegratorMethod) -> &'static str {
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

    fn obj(pairs: &[(&str, JsonValue)]) -> JsonObject {
        let mut o = JsonObject::new();
        for (k, v) in pairs {
            o.insert((*k).to_string(), v.clone());
        }
        o
    }

    #[test]
    fn json_ode_normalizes_states() {
        let states = vec![JsonValue::Object(obj(&[
            ("name", JsonValue::String("x".to_string())),
            ("derivative", JsonValue::String("-x".to_string())),
            ("initial", JsonValue::Number(1.0)),
        ]))];
        let params = MathEquationInputParams {
            format: EquationInputFormat::Json,
            states: Some(states),
            ..Default::default()
        };
        let result = run_math_equation_problem(&params, None).expect("ode run");
        assert_eq!(result.kind, EquationProblemKind::Ode);
        match &result.normalized {
            Normalized::Ode(p) => {
                assert_eq!(p.states.len(), 1);
                assert_eq!(p.states[0].name, "x");
                assert_eq!(p.states[0].derivative, "-x");
                assert_eq!(p.states[0].initial, 1.0);
                assert_eq!(p.method, IntegratorMethod::Euler);
            }
            _ => panic!("expected ODE"),
        }
        assert!(result.ode.is_some());
        assert!(!result.validation.is_empty());
    }

    #[test]
    fn json_heat_inferred_from_heat_block() {
        let params = MathEquationInputParams {
            format: EquationInputFormat::Json,
            heat1d: Some(JsonObject::new()),
            ..Default::default()
        };
        let result = run_math_equation_problem(&params, None).expect("heat run");
        assert_eq!(result.kind, EquationProblemKind::Heat1d);
        match &result.normalized {
            Normalized::Heat1d(p) => {
                assert_eq!(p.cells, 31.0);
                assert!(p.initial_expression.contains("sin"));
            }
            _ => panic!("expected heat1d"),
        }
    }

    #[test]
    fn latex_expression_rewrites_greek_and_implicit_mult() {
        assert_eq!(latex_to_expression("2\\alpha x").unwrap(), "2*alpha*x");
        assert_eq!(latex_to_expression("\\frac{a}{b}").unwrap(), "((a)/(b))");
        assert_eq!(latex_to_expression("\\sin(x)").unwrap(), "sin(x)");
    }

    #[test]
    fn latex_ode_parses_derivative() {
        let params = MathEquationInputParams {
            format: EquationInputFormat::Latex,
            equation: Some("\\frac{dx}{dt} = -k x".to_string()),
            constants: Some(obj(&[("k", JsonValue::Number(0.5))])),
            ..Default::default()
        };
        let result = run_math_equation_problem(&params, None).expect("latex ode run");
        match &result.normalized {
            Normalized::Ode(p) => {
                assert_eq!(p.states.len(), 1);
                assert_eq!(p.states[0].name, "x");
                assert_eq!(p.states[0].derivative, "-k*x");
            }
            _ => panic!("expected ODE"),
        }
    }

    #[test]
    fn xml_ode_reads_state_elements() {
        let xml =
            "<ode t0=\"0\" t1=\"2\"><state name=\"x\" derivative=\"-x\" initial=\"3\"/></ode>";
        let params = MathEquationInputParams {
            format: EquationInputFormat::Xml,
            equation: Some(xml.to_string()),
            ..Default::default()
        };
        let result = run_math_equation_problem(&params, None).expect("xml ode run");
        match &result.normalized {
            Normalized::Ode(p) => {
                assert_eq!(p.states.len(), 1);
                assert_eq!(p.states[0].name, "x");
                assert_eq!(p.states[0].initial, 3.0);
                assert_eq!(p.t1, 2.0);
            }
            _ => panic!("expected ODE"),
        }
    }

    #[test]
    fn unsupported_character_is_rejected() {
        let err = insert_implicit_multiplication("a @ b").unwrap_err();
        assert_eq!(err.param, "expression");
    }
}
