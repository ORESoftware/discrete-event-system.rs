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
const DOMAIN_ANIMATION_STEPS: usize = 4;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

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
        for step in 0..DOMAIN_ANIMATION_STEPS {
            frames.push(model_animation_frame(model, i, total, step));
        }
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
                    "animationStep",
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
        "animationFramesPerDomain": DOMAIN_ANIMATION_STEPS,
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

fn model_animation_frame(model: &PdeModel, ordinal: usize, total: usize, step: usize) -> Value {
    let domain = model.domain().expect("catalogue model has known domain");
    let tau = if DOMAIN_ANIMATION_STEPS <= 1 {
        0.0
    } else {
        step as f64 / (DOMAIN_ANIMATION_STEPS - 1) as f64
    };
    let mut shapes = canvas_title(&format!("PDE animation: {}", model.domain_title));
    shapes.extend(domain_rail(model.domain_id));
    shapes.push(rect(300.0, 84.0, 472.0, 392.0, "#ffffff", "#cbd5e1", 1.0));
    shapes.push(text(
        322.0,
        122.0,
        model.title.to_string(),
        18.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes.push(text(
        322.0,
        148.0,
        format!(
            "{} / frame {} of {}",
            model.domain_title,
            step + 1,
            DOMAIN_ANIMATION_STEPS
        ),
        12.0,
        "#475569",
        "start",
        false,
    ));
    shapes.extend(domain_animation_shapes(domain, tau));
    shapes.push(text(
        322.0,
        444.0,
        clip(model.modeling_notes, 72),
        11.0,
        "#475569",
        "start",
        false,
    ));
    shapes.extend(metric_bars(model));

    let mut out = frame(
        model.domain_id,
        format!(
            "{} animation frame {}/{}.",
            model.domain_title,
            step + 1,
            DOMAIN_ANIMATION_STEPS
        ),
        &[
            ("domainOrdinal", (ordinal + 1) as f64),
            ("modelsCovered", (ordinal + 1) as f64 / total.max(1) as f64),
            ("equationCount", model.canonical_equations.len() as f64),
            ("principleCount", model.primary_principles.len() as f64),
            ("methodCount", model.numerical_methods.len() as f64),
            ("applicationCount", model.applications.len() as f64),
            ("animationStep", (step + 1) as f64),
        ],
        shapes,
    );
    if let Some(obj) = out.as_object_mut() {
        obj.insert("view".to_string(), json!("animation"));
        obj.insert("normalizedTime".to_string(), json!(tau));
    }
    out
}

fn domain_animation_shapes(domain: PdeDomain, tau: f64) -> Vec<Value> {
    match domain {
        PdeDomain::Electromagnetism => electromagnetism_animation(tau),
        PdeDomain::QuantumMechanics => quantum_animation(tau),
        PdeDomain::HeatTransferDiffusion => heat_animation(tau),
        PdeDomain::SolidMechanics => solid_animation(tau),
        PdeDomain::AcousticsWavePropagation => acoustics_animation(tau),
        PdeDomain::ControlOptimalControl => control_animation(tau),
        PdeDomain::GeometrySurfaces => geometry_animation(tau),
        PdeDomain::ImageProcessingVision => vision_animation(tau),
        PdeDomain::Finance => finance_animation(tau),
        PdeDomain::PopulationBiology => biology_animation(tau),
        PdeDomain::PlasmaAstrophysics => plasma_animation(tau),
        PdeDomain::MaterialsScience => materials_animation(tau),
    }
}

fn electromagnetism_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("curl fields and radiation");
    shapes.push(line(354.0, 204.0, 354.0, 354.0, "#334155", 5.0));
    shapes.push(circle(354.0, 204.0, 8.0, "#f97316", "#9a3412"));
    for i in 0..3 {
        let r = 34.0 + i as f64 * 44.0 + tau * 22.0;
        shapes.push(circle(354.0, 204.0, r, "none", "#60a5fa"));
    }
    shapes.push(wave_path(
        420.0,
        230.0,
        214.0,
        28.0,
        tau * TWO_PI,
        "#2563eb",
    ));
    shapes.push(wave_path(
        420.0,
        304.0,
        214.0,
        22.0,
        tau * TWO_PI + std::f64::consts::FRAC_PI_2,
        "#dc2626",
    ));
    shapes.push(text(
        650.0,
        232.0,
        "E".to_string(),
        14.0,
        "#2563eb",
        "start",
        true,
    ));
    shapes.push(text(
        650.0,
        306.0,
        "B".to_string(),
        14.0,
        "#dc2626",
        "start",
        true,
    ));
    shapes
}

fn quantum_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("wavefunction probability and tunneling");
    shapes.push(rect(522.0, 180.0, 42.0, 210.0, "#fee2e2", "#ef4444", 1.0));
    shapes.push(text(
        543.0,
        410.0,
        "V".to_string(),
        12.0,
        "#991b1b",
        "middle",
        true,
    ));
    shapes.push(path(
        density_curve(330.0, 315.0, 330.0, 88.0, 414.0 + 118.0 * tau, 38.0),
        "#7c3aed",
        3.0,
        "none",
    ));
    let leak = 0.18 + 0.5 * tau;
    shapes.push(circle(
        588.0 + 54.0 * tau,
        280.0,
        7.0 + 5.0 * leak,
        "#c4b5fd",
        "#7c3aed",
    ));
    shapes.push(line(330.0, 315.0, 704.0, 315.0, "#94a3b8", 1.0));
    shapes.push(text(
        334.0,
        176.0,
        "psi(x,t)".to_string(),
        12.0,
        "#4c1d95",
        "start",
        true,
    ));
    shapes
}

fn heat_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("heat kernel smoothing from a hot spot");
    draw_field_grid(&mut shapes, 342.0, 178.0, 11, 7, 30.0, |i, j| {
        let x = i as f64 - 5.0;
        let y = j as f64 - 3.0;
        let sigma = 0.75 + 2.6 * tau;
        let heat = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
        heat_color(heat)
    });
    shapes.push(text(
        342.0,
        410.0,
        "d_t u = div(k grad u)".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn solid_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("elastic beam displacement and stress");
    shapes.push(rect(338.0, 300.0, 338.0, 22.0, "#e2e8f0", "#64748b", 1.0));
    let amp = 38.0 * (tau * TWO_PI).sin();
    shapes.push(path(
        sine_beam_path(338.0, 284.0, 338.0, amp),
        "#2563eb",
        4.0,
        "none",
    ));
    for i in 0..7 {
        let x = 352.0 + i as f64 * 52.0;
        let y = 284.0 + amp * (i as f64 / 6.0 * std::f64::consts::PI).sin();
        shapes.push(circle(x, y, 7.0, stress_color(i, tau).as_str(), "#1e293b"));
    }
    shapes.push(line(338.0, 330.0, 360.0, 360.0, "#475569", 2.0));
    shapes.push(line(676.0, 330.0, 654.0, 360.0, "#475569", 2.0));
    shapes.push(text(
        338.0,
        194.0,
        "rho d_tt u = div sigma + b".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn acoustics_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("pressure wavefronts and reflection");
    shapes.push(line(666.0, 164.0, 666.0, 386.0, "#475569", 4.0));
    shapes.push(circle(374.0, 274.0, 12.0, "#f97316", "#9a3412"));
    for i in 0..4 {
        let r = 26.0 + i as f64 * 42.0 + 28.0 * tau;
        shapes.push(circle(374.0, 274.0, r, "none", "#0ea5e9"));
    }
    for i in 0..3 {
        let r = 18.0 + i as f64 * 34.0 + 22.0 * (1.0 - tau);
        shapes.push(circle(666.0, 274.0, r, "none", "#94a3b8"));
    }
    shapes.push(text(
        338.0,
        410.0,
        "d_tt p = c^2 laplacian p".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn control_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("distributed state with boundary feedback");
    let control = 0.35 + 0.55 * (1.0 - tau);
    shapes.push(rect(344.0, 254.0, 314.0, 46.0, "#e0f2fe", "#0284c7", 1.0));
    for i in 0..12 {
        let x = 348.0 + i as f64 * 26.0;
        let temp = ((i as f64 / 11.0 - tau).abs() * -3.0).exp();
        shapes.push(rect(
            x,
            256.0,
            22.0,
            42.0,
            &heat_color(temp),
            "#bae6fd",
            0.5,
        ));
    }
    shapes.push(line(326.0, 277.0, 344.0, 277.0, "#dc2626", 4.0));
    shapes.push(text(
        320.0,
        246.0,
        format!("u={control:.2}"),
        12.0,
        "#991b1b",
        "start",
        true,
    ));
    shapes.push(path(
        value_function_path(348.0, 186.0, 308.0, tau),
        "#7c3aed",
        3.0,
        "none",
    ));
    shapes.push(text(
        348.0,
        176.0,
        "HJB value surface slice".to_string(),
        12.0,
        "#4c1d95",
        "start",
        true,
    ));
    shapes
}

fn geometry_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("curvature flow smooths a surface");
    let amp = 48.0 * (1.0 - 0.72 * tau);
    let surface = sine_beam_path(330.0, 280.0, 364.0, amp);
    shapes.push(path(surface, "#db2777", 4.0, "none"));
    for i in 0..6 {
        let x = 350.0 + i as f64 * 58.0;
        let y = 280.0 + amp * (i as f64 / 5.0 * TWO_PI + tau).sin();
        shapes.push(line(x, y, x, y - 28.0 * (1.0 - tau), "#64748b", 1.5));
    }
    shapes.push(text(
        340.0,
        396.0,
        "d_t X = -H n".to_string(),
        13.0,
        "#831843",
        "start",
        true,
    ));
    shapes
}

fn vision_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("anisotropic diffusion preserves edges");
    draw_field_grid(&mut shapes, 346.0, 178.0, 12, 8, 26.0, |i, j| {
        let edge = if i >= 5 { 0.72 } else { 0.18 };
        let texture = (((i * 17 + j * 11) as f64).sin() * 0.18) * (1.0 - tau);
        gray_color((edge + texture).clamp(0.0, 1.0))
    });
    shapes.push(line(
        346.0 + 5.0 * 26.0,
        178.0,
        346.0 + 5.0 * 26.0,
        386.0,
        "#f97316",
        3.0,
    ));
    shapes.push(text(
        346.0,
        414.0,
        "d_t I = div(g(|grad I|) grad I)".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn finance_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("option payoff diffuses backward in time");
    shapes.push(line(336.0, 360.0, 704.0, 360.0, "#94a3b8", 1.0));
    shapes.push(line(376.0, 166.0, 376.0, 370.0, "#94a3b8", 1.0));
    shapes.push(path(
        payoff_curve(350.0, 348.0, 330.0, 150.0, tau),
        "#16a34a",
        3.0,
        "none",
    ));
    for i in 0..3 {
        shapes.push(path(
            diffusion_fan_curve(350.0, 348.0, 330.0, 112.0, tau + i as f64 * 0.18),
            "#60a5fa",
            1.5,
            "none",
        ));
    }
    shapes.push(text(
        338.0,
        408.0,
        "Black-Scholes: parabolic pricing PDE".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn biology_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("reaction-diffusion pattern formation");
    draw_field_grid(&mut shapes, 344.0, 174.0, 13, 9, 24.0, |i, j| {
        let x = i as f64 * 0.9;
        let y = j as f64 * 0.9;
        let v = 0.5 + 0.5 * ((x + tau * TWO_PI).sin() * (y * 1.4 - tau * TWO_PI).cos());
        bio_color(v)
    });
    shapes.push(text(
        344.0,
        414.0,
        "d_t u_i = div(D_i grad u_i) + f_i(u)".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn plasma_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("MHD couples flow and magnetic fields");
    for i in 0..5 {
        let offset = -62.0 + i as f64 * 31.0;
        shapes.push(path(
            swirl_path(520.0, 278.0 + offset, 135.0, tau + i as f64 * 0.08),
            "#2563eb",
            2.0,
            "none",
        ));
    }
    for i in 0..10 {
        let angle = tau * TWO_PI + i as f64 * 0.63;
        let r = 35.0 + (i % 4) as f64 * 18.0;
        let x = 520.0 + r * angle.cos();
        let y = 278.0 + r * angle.sin();
        shapes.push(circle(x, y, 5.0, "#f97316", "#9a3412"));
    }
    shapes.push(text(
        342.0,
        414.0,
        "d_t B = curl(u x B - eta curl B)".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
}

fn materials_animation(tau: f64) -> Vec<Value> {
    let mut shapes = visual_stage("phase-field domains coarsen by energy descent");
    draw_field_grid(&mut shapes, 344.0, 174.0, 13, 9, 24.0, |i, j| {
        let x = i as f64 - 6.0;
        let y = j as f64 - 4.0;
        let field = ((x * 0.9 + tau * 2.0).sin() + (y * 1.1 - tau * 3.0).cos()) / 2.0;
        phase_color(field.tanh())
    });
    shapes.push(text(
        344.0,
        414.0,
        "Allen-Cahn / Cahn-Hilliard gradient flows".to_string(),
        13.0,
        "#0f172a",
        "start",
        true,
    ));
    shapes
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

fn line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) -> Value {
    json!({
        "kind": "line",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke": stroke,
        "strokeWidth": stroke_width
    })
}

fn circle(x: f64, y: f64, r: f64, fill: &str, stroke: &str) -> Value {
    json!({
        "kind": "circle",
        "x": x,
        "y": y,
        "r": r,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": 1.5
    })
}

fn path(d: String, stroke: &str, stroke_width: f64, fill: &str) -> Value {
    json!({
        "kind": "path",
        "d": d,
        "stroke": stroke,
        "strokeWidth": stroke_width,
        "fill": fill
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

fn visual_stage(label: &str) -> Vec<Value> {
    vec![
        rect(322.0, 164.0, 420.0, 252.0, "#f8fafc", "#e2e8f0", 1.0),
        text(
            336.0,
            190.0,
            label.to_string(),
            13.0,
            "#0f172a",
            "start",
            true,
        ),
    ]
}

fn wave_path(x0: f64, y0: f64, width: f64, amp: f64, phase: f64, stroke: &str) -> Value {
    path(
        sampled_path(48, |i| {
            let u = i as f64 / 47.0;
            let x = x0 + width * u;
            let y = y0 + amp * (u * TWO_PI * 2.0 + phase).sin();
            (x, y)
        }),
        stroke,
        3.0,
        "none",
    )
}

fn sampled_path(n: usize, f: impl Fn(usize) -> (f64, f64)) -> String {
    let mut d = String::new();
    for i in 0..n {
        let (x, y) = f(i);
        if i == 0 {
            d.push_str(&format!("M {x:.1} {y:.1}"));
        } else {
            d.push_str(&format!(" L {x:.1} {y:.1}"));
        }
    }
    d
}

fn density_curve(x0: f64, y0: f64, width: f64, height: f64, center: f64, sigma: f64) -> String {
    sampled_path(64, |i| {
        let u = i as f64 / 63.0;
        let x = x0 + width * u;
        let density = (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp();
        (x, y0 - height * density)
    })
}

fn sine_beam_path(x0: f64, y0: f64, width: f64, amp: f64) -> String {
    sampled_path(48, |i| {
        let u = i as f64 / 47.0;
        let x = x0 + width * u;
        let y = y0 + amp * (u * std::f64::consts::PI).sin();
        (x, y)
    })
}

fn value_function_path(x0: f64, y0: f64, width: f64, tau: f64) -> String {
    sampled_path(56, |i| {
        let u = i as f64 / 55.0;
        let x = x0 + width * u;
        let y = y0 + 68.0 * (u - tau).powi(2) + 10.0 * (u * TWO_PI).sin();
        (x, y)
    })
}

fn payoff_curve(x0: f64, y0: f64, width: f64, height: f64, tau: f64) -> String {
    sampled_path(64, |i| {
        let u = i as f64 / 63.0;
        let smoothed = ((u - 0.42).max(0.0)).powf(0.75 + 0.6 * tau);
        let x = x0 + width * u;
        let y = y0 - height * smoothed;
        (x, y)
    })
}

fn diffusion_fan_curve(x0: f64, y0: f64, width: f64, height: f64, phase: f64) -> String {
    sampled_path(64, |i| {
        let u = i as f64 / 63.0;
        let x = x0 + width * u;
        let bell = (-((u - 0.5 - 0.18 * phase.sin()).powi(2)) / 0.08).exp();
        let y = y0 - 42.0 - height * bell * 0.45;
        (x, y)
    })
}

fn swirl_path(cx: f64, cy: f64, radius: f64, phase: f64) -> String {
    sampled_path(80, |i| {
        let u = i as f64 / 79.0;
        let angle = u * TWO_PI * 1.2 + phase * TWO_PI;
        let r = radius * (0.15 + 0.8 * u);
        let x = cx + r * angle.cos();
        let y = cy + 0.35 * r * angle.sin();
        (x, y)
    })
}

fn draw_field_grid(
    shapes: &mut Vec<Value>,
    x0: f64,
    y0: f64,
    cols: usize,
    rows: usize,
    cell: f64,
    color: impl Fn(usize, usize) -> String,
) {
    for j in 0..rows {
        for i in 0..cols {
            let fill = color(i, j);
            shapes.push(rect(
                x0 + i as f64 * cell,
                y0 + j as f64 * cell,
                cell - 1.0,
                cell - 1.0,
                &fill,
                "#ffffff",
                0.4,
            ));
        }
    }
}

fn heat_color(v: f64) -> String {
    let v = v.clamp(0.0, 1.0);
    let r = (37.0 + 218.0 * v) as u8;
    let g = (99.0 + 90.0 * (1.0 - (v - 0.35).abs()).clamp(0.0, 1.0)) as u8;
    let b = (235.0 * (1.0 - v)) as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn gray_color(v: f64) -> String {
    let c = (255.0 * v.clamp(0.0, 1.0)) as u8;
    format!("#{c:02x}{c:02x}{c:02x}")
}

fn bio_color(v: f64) -> String {
    let v = v.clamp(0.0, 1.0);
    let g = (84.0 + 150.0 * v) as u8;
    let b = (90.0 + 90.0 * (1.0 - v)) as u8;
    format!("#16{g:02x}{b:02x}")
}

fn phase_color(v: f64) -> String {
    if v >= 0.0 {
        let c = (180.0 + 55.0 * v) as u8;
        format!("#{c:02x}e7f3")
    } else {
        let c = (180.0 + 55.0 * -v) as u8;
        format!("#dbeafe").replace("db", &format!("{c:02x}"))
    }
}

fn stress_color(i: usize, tau: f64) -> String {
    let v = (0.5 + 0.5 * (i as f64 * 0.9 + tau * TWO_PI).sin()).clamp(0.0, 1.0);
    heat_color(v)
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
    use crate::des::plugin::PlayerKind;

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
        assert_eq!(
            artifact.frames.len(),
            1 + PdeDomain::all().len() * (1 + DOMAIN_ANIMATION_STEPS)
        );
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
        assert_eq!(artifact.frames.len(), 1 + DOMAIN_ANIMATION_STEPS);
    }

    #[test]
    fn every_domain_has_playable_animation_frames() {
        let citizen = PartialDifferentialEquationsCitizen;
        for domain in PdeDomain::all() {
            let artifact = citizen
                .run_json(&json!({
                    "$schema": PARTIAL_DIFFERENTIAL_EQUATIONS_SCHEMA,
                    "domain": domain.id(),
                    "includeFramework": false
                }))
                .unwrap_or_else(|e| panic!("domain {} failed: {e}", domain.id()));
            assert_eq!(artifact.player, PlayerKind::Sim, "domain {}", domain.id());

            let animation_frames: Vec<_> = artifact
                .frames
                .iter()
                .filter(|frame| frame["view"].as_str() == Some("animation"))
                .collect();
            assert_eq!(
                animation_frames.len(),
                DOMAIN_ANIMATION_STEPS,
                "domain {}",
                domain.id()
            );
            for frame in animation_frames {
                let shapes = frame["shapes"]
                    .as_array()
                    .unwrap_or_else(|| panic!("domain {} frame has no shapes", domain.id()));
                assert!(
                    shapes.len() >= 12,
                    "domain {} animation frame is too sparse: {} shapes",
                    domain.id(),
                    shapes.len()
                );
            }

            let html = artifact.to_player_html();
            assert!(html.contains("plugin-payload"), "domain {}", domain.id());
        }
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
