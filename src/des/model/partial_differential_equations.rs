//! First-class partial differential equation catalogue citizen.
//!
//! The numerical heavy lifting for PDEs usually happens after a user chooses a
//! domain, field, boundary data, and discretization. This citizen makes that
//! choice explicit: it exposes the typed catalogue in
//! [`crate::des::general::partial_differential_equations`] as a runnable model
//! artifact with frames and JSON results.

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Map, Value};

use crate::des::general::partial_differential_equations::{
    pde_models, pde_unifying_framework, validate_pde_catalog, PdeDomain, PdeModel, PdePrinciple,
};
use crate::des::plugin::UiControl;

use super::artifact::RunArtifact;
use super::registry::{CitizenError, ModelCitizen, ModelDescriptor};

pub const PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA: &str = "des/partial-differential-equations/v1";

const DOMAIN_OPTIONS: &[&str] = &[
    "all",
    "electromagnetism",
    "quantum-mechanics",
    "heat-transfer-diffusion",
    "solid-mechanics",
    "acoustics-wave-propagation",
    "control-optimal-control",
    "geometry-surfaces",
    "image-processing-vision",
    "finance",
    "population-biology",
    "plasma-astrophysics",
    "materials-science",
];

/// First-class citizen for PDE domain models.
pub struct PartialDifferentialEquationsCitizen;

#[derive(Clone, Debug)]
struct PdeInputs {
    domains: Option<Vec<PdeDomain>>,
    principle: Option<PdePrinciple>,
    include_framework: bool,
}

impl ModelCitizen for PartialDifferentialEquationsCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "partial-differential-equations".to_string(),
            title: "Partial Differential Equations".to_string(),
            description: "Typed catalogue of PDE model families across electromagnetism, quantum mechanics, heat/diffusion, solids, waves, control, geometry, vision, finance, biology, plasma, and materials."
                .to_string(),
            spec_schema: PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA.to_string(),
            methods: vec![
                "catalog".to_string(),
                "domain-filter".to_string(),
                "principle-filter".to_string(),
                "framework-map".to_string(),
            ],
            example_spec: starter_partial_differential_equations_spec(),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        match catch_unwind(AssertUnwindSafe(|| run_pde_catalog_checked(spec))) {
            Ok(result) => result,
            Err(payload) => Err(CitizenError::InvalidSpec(format!(
                "partial-differential-equations model rejected the spec: {}",
                panic_message(payload)
            ))),
        }
    }
}

pub fn starter_partial_differential_equations_spec() -> Value {
    json!({
        "$schema": PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA,
        "domain": "all",
        "includeFramework": true
    })
}

fn run_pde_catalog_checked(spec: &Value) -> Result<RunArtifact, CitizenError> {
    validate_schema(spec)?;
    validate_pde_catalog().map_err(CitizenError::Run)?;

    let inputs = parse_inputs(spec)?;
    let mut models = pde_models();
    if let Some(domains) = &inputs.domains {
        models.retain(|model| domains.iter().any(|domain| model.domain_id == domain.id()));
    }
    if let Some(principle) = inputs.principle {
        models.retain(|model| model.primary_principles.contains(&principle));
    }
    if models.is_empty() {
        return Err(CitizenError::InvalidSpec(
            "PDE catalogue filter selected no models".to_string(),
        ));
    }

    let mut frames = Vec::new();
    if inputs.include_framework {
        frames.push(framework_frame());
    }
    let total = models.len();
    for (i, model) in models.iter().enumerate() {
        frames.push(model_frame(model, i, total));
    }
    stamp_frames(&mut frames);

    let results = results_json(&models, inputs.include_framework);
    let summary = format!(
        "Partial differential equations catalogue run: {} domain model(s), {} frame(s).",
        models.len(),
        frames.len()
    );

    Ok(RunArtifact::sim(
        "partial-differential-equations",
        "Partial Differential Equations",
        "PDE domain catalogue and unifying operator framework.",
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 6.0),
            UiControl::select(
                "domain",
                "Domain",
                DOMAIN_OPTIONS,
                "all",
                Some("domainOrdinal"),
            ),
            UiControl::select(
                "metric",
                "Feature signal",
                &[
                    "modelsCovered",
                    "equationCount",
                    "principleCount",
                    "methodCount",
                    "applicationCount",
                ],
                "modelsCovered",
                Some("metric"),
            ),
        ],
        &summary,
    ))
}

fn validate_schema(spec: &Value) -> Result<(), CitizenError> {
    if let Some(schema) = spec.get("$schema").and_then(Value::as_str) {
        if schema != PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA {
            return Err(CitizenError::InvalidSpec(format!(
                "expected $schema `{PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA}`, got `{schema}`"
            )));
        }
    }
    if !spec.is_object() {
        return Err(CitizenError::InvalidSpec(
            "partial-differential-equations spec must be a JSON object".to_string(),
        ));
    }
    Ok(())
}

fn parse_inputs(spec: &Value) -> Result<PdeInputs, CitizenError> {
    let domains = if let Some(values) = spec.get("domains") {
        let parsed = parse_domain_array(values)?;
        if parsed.is_empty() {
            None
        } else {
            Some(parsed)
        }
    } else {
        match spec
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or("all")
            .trim()
        {
            "all" | "" => None,
            domain => Some(vec![parse_domain(domain)?]),
        }
    };
    let principle = match spec.get("principle").and_then(Value::as_str) {
        Some(value) => Some(parse_principle(value)?),
        None => None,
    };
    let include_framework = spec
        .get("includeFramework")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    Ok(PdeInputs {
        domains,
        principle,
        include_framework,
    })
}

fn parse_domain_array(value: &Value) -> Result<Vec<PdeDomain>, CitizenError> {
    let Value::Array(values) = value else {
        return Err(CitizenError::InvalidSpec(
            "domains must be an array of domain ids".to_string(),
        ));
    };
    if values.is_empty() {
        return Err(CitizenError::InvalidSpec(
            "domains must contain at least one domain id".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let Some(domain) = value.as_str() else {
            return Err(CitizenError::InvalidSpec(format!(
                "domains[{i}] must be a string"
            )));
        };
        if domain == "all" {
            return Ok(Vec::new());
        }
        out.push(parse_domain(domain)?);
    }
    Ok(out)
}

fn parse_domain(value: &str) -> Result<PdeDomain, CitizenError> {
    PdeDomain::from_id(value).ok_or_else(|| {
        CitizenError::InvalidSpec(format!(
            "unknown PDE domain `{value}` (expected one of: {})",
            DOMAIN_OPTIONS.join(", ")
        ))
    })
}

fn parse_principle(value: &str) -> Result<PdePrinciple, CitizenError> {
    PdePrinciple::from_id(value).ok_or_else(|| {
        CitizenError::InvalidSpec(format!(
            "unknown PDE principle `{value}` (expected conservation, diffusion, waves, optimization, geometry, coupled-fields, reaction, or stochastic)"
        ))
    })
}

fn results_json(models: &[PdeModel], include_framework: bool) -> Value {
    json!({
        "kind": "partial-differential-equations",
        "schema": PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA,
        "domainCount": models.len(),
        "domains": models.iter().map(|model| model.domain_id).collect::<Vec<_>>(),
        "models": models
            .iter()
            .map(|model| serde_json::to_value(model).expect("PdeModel serializes"))
            .collect::<Vec<_>>(),
        "framework": include_framework.then(|| serde_json::to_value(pde_unifying_framework()).expect("PdeFramework serializes")),
    })
}

fn framework_frame() -> Value {
    let framework = pde_unifying_framework();
    let mut shapes = canvas_title("Unifying PDE framework");
    shapes.push(rect(42.0, 86.0, 724.0, 92.0, "#f8fafc", "#cbd5e1", 1.0));
    for (line_no, line) in wrap_text(framework.statement, 86, 3)
        .into_iter()
        .enumerate()
    {
        shapes.push(text(
            62.0,
            118.0 + line_no as f64 * 20.0,
            line,
            15.0,
            "#0f172a",
            "start",
            false,
        ));
    }
    let mut y = 218.0;
    for pattern in &framework.patterns {
        shapes.push(rect(58.0, y - 26.0, 690.0, 52.0, "#ffffff", "#e2e8f0", 1.0));
        shapes.push(text(
            76.0,
            y - 7.0,
            pattern.principle.label().to_string(),
            13.0,
            "#0f172a",
            "start",
            true,
        ));
        shapes.push(text(
            310.0,
            y - 7.0,
            clip(pattern.operator_signature, 58),
            12.0,
            "#475569",
            "start",
            false,
        ));
        shapes.push(text(
            76.0,
            y + 15.0,
            clip(pattern.intuition, 94),
            11.0,
            "#64748b",
            "start",
            false,
        ));
        y += 66.0;
    }
    frame(
        "framework",
        "PDE framework: conservation, diffusion, waves, optimization, and geometry.".to_string(),
        &[
            ("modelsCovered", 0.0),
            ("equationCount", 0.0),
            ("principleCount", framework.patterns.len() as f64),
            ("methodCount", framework.common_operators.len() as f64),
            ("applicationCount", 0.0),
        ],
        shapes,
    )
}

fn model_frame(model: &PdeModel, ordinal: usize, total: usize) -> Value {
    let mut shapes = canvas_title("PDE domain model catalogue");
    shapes.extend(domain_rail(model.domain_id));
    shapes.push(rect(300.0, 84.0, 472.0, 368.0, "#ffffff", "#cbd5e1", 1.0));
    shapes.push(text(
        322.0,
        122.0,
        format!("{}. {}", ordinal + 1, model.domain_title),
        22.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes.push(text(
        322.0,
        150.0,
        model.title.to_string(),
        14.0,
        "#475569",
        "start",
        false,
    ));

    let equation = &model.canonical_equations[0];
    shapes.push(text(
        322.0,
        192.0,
        format!("Canonical: {}", equation.name),
        14.0,
        "#0f172a",
        "start",
        true,
    ));
    for (line_no, line) in wrap_text(equation.symbolic_form, 62, 3)
        .into_iter()
        .enumerate()
    {
        shapes.push(text(
            322.0,
            218.0 + line_no as f64 * 19.0,
            line,
            12.0,
            "#334155",
            "start",
            false,
        ));
    }

    shapes.push(text(
        322.0,
        286.0,
        "Principles".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    for (i, principle) in model.primary_principles.iter().enumerate() {
        let x = 322.0 + (i % 2) as f64 * 212.0;
        let y = 304.0 + (i / 2) as f64 * 30.0;
        shapes.push(rect(
            x,
            y,
            194.0,
            22.0,
            principle_fill(*principle),
            "#cbd5e1",
            1.0,
        ));
        shapes.push(text(
            x + 10.0,
            y + 15.0,
            clip(principle.label(), 25),
            10.0,
            "#0f172a",
            "start",
            false,
        ));
    }

    shapes.push(text(
        322.0,
        380.0,
        format!("Numerics: {}", join_limited(&model.numerical_methods, 70)),
        12.0,
        "#334155",
        "start",
        false,
    ));
    shapes.push(text(
        322.0,
        405.0,
        format!("Applications: {}", join_limited(&model.applications, 66)),
        12.0,
        "#334155",
        "start",
        false,
    ));
    shapes.extend(metric_bars(model));

    frame(
        model.domain_id,
        format!(
            "{}: {} equation template(s), {} method family entries.",
            model.domain_title,
            model.canonical_equations.len(),
            model.numerical_methods.len()
        ),
        &[
            ("domainOrdinal", (ordinal + 1) as f64),
            ("modelsCovered", (ordinal + 1) as f64 / total.max(1) as f64),
            ("equationCount", model.canonical_equations.len() as f64),
            ("principleCount", model.primary_principles.len() as f64),
            ("methodCount", model.numerical_methods.len() as f64),
            ("applicationCount", model.applications.len() as f64),
        ],
        shapes,
    )
}

fn domain_rail(active_domain: &str) -> Vec<Value> {
    let mut shapes = vec![rect(42.0, 84.0, 236.0, 368.0, "#f8fafc", "#cbd5e1", 1.0)];
    for (i, domain) in PdeDomain::all().iter().enumerate() {
        let y = 106.0 + i as f64 * 28.0;
        let active = domain.id() == active_domain;
        shapes.push(rect(
            58.0,
            y - 17.0,
            204.0,
            22.0,
            if active { "#dbeafe" } else { "#ffffff" },
            if active { "#2563eb" } else { "#e2e8f0" },
            if active { 2.0 } else { 1.0 },
        ));
        shapes.push(text(
            68.0,
            y - 2.0,
            format!("{:02} {}", i + 1, clip(domain.title(), 25)),
            10.0,
            if active { "#1d4ed8" } else { "#334155" },
            "start",
            active,
        ));
    }
    shapes
}

fn metric_bars(model: &PdeModel) -> Vec<Value> {
    let metrics = [
        ("eqs", model.canonical_equations.len() as f64, 4.0),
        ("principles", model.primary_principles.len() as f64, 4.0),
        ("methods", model.numerical_methods.len() as f64, 5.0),
        ("apps", model.applications.len() as f64, 5.0),
    ];
    let mut shapes = Vec::new();
    let x0 = 322.0;
    for (i, (label, value, max_value)) in metrics.iter().enumerate() {
        let x = x0 + i as f64 * 108.0;
        let h = 48.0 * (*value / *max_value).min(1.0);
        shapes.push(rect(x, 520.0 - h, 58.0, h, "#38bdf8", "#0284c7", 1.0));
        shapes.push(text(
            x + 29.0,
            536.0,
            (*label).to_string(),
            10.0,
            "#334155",
            "middle",
            false,
        ));
        shapes.push(text(
            x + 29.0,
            512.0 - h,
            format!("{value:.0}"),
            10.0,
            "#0f172a",
            "middle",
            true,
        ));
    }
    shapes
}

fn frame(phase: &str, caption: String, metrics: &[(&str, f64)], shapes: Vec<Value>) -> Value {
    let mut obj = Map::new();
    obj.insert("t".to_string(), json!(0.0));
    obj.insert("tick".to_string(), json!(0));
    obj.insert("phase".to_string(), json!(phase));
    obj.insert("caption".to_string(), json!(caption));
    obj.insert("shapes".to_string(), Value::Array(shapes));
    for (key, value) in metrics {
        obj.insert((*key).to_string(), json!(value));
    }
    Value::Object(obj)
}

fn stamp_frames(frames: &mut [Value]) {
    for (tick, frame) in frames.iter_mut().enumerate() {
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("tick".to_string(), json!(tick));
            obj.insert("t".to_string(), json!(tick as f64));
        }
    }
}

fn canvas_title(title: &str) -> Vec<Value> {
    vec![
        rect(22.0, 24.0, 786.0, 548.0, "#f1f5f9", "#cbd5e1", 1.0),
        text(
            42.0,
            58.0,
            title.to_string(),
            22.0,
            "#0f172a",
            "start",
            true,
        ),
    ]
}

fn rect(x: f64, y: f64, w: f64, h: f64, fill: &str, stroke: &str, stroke_width: f64) -> Value {
    json!({
        "kind": "rect",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": 4.0,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width
    })
}

fn text(
    x: f64,
    y: f64,
    text_value: String,
    font_size: f64,
    fill: &str,
    anchor: &str,
    bold: bool,
) -> Value {
    json!({
        "kind": "text",
        "x": x,
        "y": y,
        "text": text_value,
        "fontSize": font_size,
        "fill": fill,
        "anchor": anchor,
        "fontWeight": if bold { "700" } else { "400" },
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    })
}

fn principle_fill(principle: PdePrinciple) -> &'static str {
    match principle {
        PdePrinciple::ConservationLaw => "#dcfce7",
        PdePrinciple::Diffusion => "#e0f2fe",
        PdePrinciple::WavePropagation => "#fef3c7",
        PdePrinciple::OptimizationVariational => "#f3e8ff",
        PdePrinciple::GeometryCurvature => "#fce7f3",
        PdePrinciple::CoupledFields => "#ede9fe",
        PdePrinciple::ReactionKinetics => "#ffedd5",
        PdePrinciple::StochasticDuality => "#ccfbf1",
    }
}

fn join_limited(items: &[&str], max_chars: usize) -> String {
    clip(&items.join(", "), max_chars)
}

fn clip(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, ch) in value.chars().enumerate() {
        if i >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn wrap_text(value: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in value.split_whitespace() {
        if current.len() + word.len() + usize::from(!current.is_empty()) > max_chars {
            if !current.is_empty() {
                lines.push(current);
                current = String::new();
            }
            if lines.len() == max_lines {
                break;
            }
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() && lines.len() < max_lines {
        lines.push(current);
    }
    if lines.len() == max_lines && value.len() > lines.join(" ").len() {
        if let Some(last) = lines.last_mut() {
            *last = clip(last, max_chars.saturating_sub(3));
        }
    }
    lines
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::model::registry::ModelCitizen;

    #[test]
    fn starter_spec_renders_every_requested_pde_domain() {
        let citizen = PartialDifferentialEquationsCitizen;
        let artifact = citizen
            .run_json(&starter_partial_differential_equations_spec())
            .expect("starter PDE catalogue spec should run");
        assert_eq!(artifact.kind, "partial-differential-equations");
        assert_eq!(
            artifact.results["domainCount"].as_u64(),
            Some(PdeDomain::all().len() as u64)
        );
        assert!(artifact.frames.len() >= PdeDomain::all().len());
        assert!(artifact
            .frames
            .iter()
            .any(|frame| frame["shapes"].as_array().is_some_and(|s| !s.is_empty())));
        assert!(artifact
            .to_player_html()
            .contains("Partial Differential Equations"));
    }

    #[test]
    fn domain_filter_runs_one_model() {
        let citizen = PartialDifferentialEquationsCitizen;
        let artifact = citizen
            .run_json(&json!({
                "$schema": PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA,
                "domain": "finance",
                "includeFramework": false
            }))
            .expect("finance PDE catalogue spec should run");
        assert_eq!(artifact.results["domainCount"].as_u64(), Some(1));
        assert_eq!(artifact.results["domains"][0].as_str(), Some("finance"));
        assert_eq!(artifact.frames.len(), 1);
    }

    #[test]
    fn invalid_domain_returns_citizen_error() {
        let citizen = PartialDifferentialEquationsCitizen;
        assert!(matches!(
            citizen.run_json(&json!({
                "$schema": PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA,
                "domain": "not-a-domain"
            })),
            Err(CitizenError::InvalidSpec(_))
        ));
    }
}
