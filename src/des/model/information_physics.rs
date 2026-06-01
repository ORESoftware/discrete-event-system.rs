//! First-class animated models for the information/thermodynamics lineage.
//!
//! This citizen turns the finite calculations in
//! [`crate::des::general::control_systems::information_theory`] into a
//! runnable `ModelCitizen`: JSON spec in, animated `RunArtifact` out.

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Map, Value};

use crate::des::general::control_systems::information_theory::{
    boltzmann_entropy, channel_capacity_blahut_arimoto_bits, gibbs_canonical_ensemble,
    hartley_information, information_physics_catalog, jarzynski_free_energy_estimate,
    maxwell_demon_budget_from_joint, nonequilibrium_free_energy, shannon_entropy_bits,
    stochastic_thermodynamics_summary, szilard_landauer_budget, BOLTZMANN_CONSTANT_J_PER_K,
};
use crate::des::plugin::UiControl;
use crate::des::shared::linalg::Matrix;

use super::artifact::RunArtifact;
use super::registry::{CitizenError, ModelCitizen, ModelDescriptor};

pub const INFORMATION_PHYSICS_SCHEMA: &str = "des/information-physics/v1";

const DEFAULT_TEMPERATURE_K: f64 = 300.0;
const MAX_SYMBOLS: usize = 256;
const MAX_ENERGY_LEVELS: usize = 24;
const MAX_CHANNEL_DIM: usize = 12;
const MAX_WORK_SAMPLES: usize = 128;

/// First-class citizen for animated Boltzmann/Gibbs/Hartley/Shannon/Szilard
/// models and their modern thermodynamic-information counterparts.
pub struct InformationPhysicsCitizen;

#[derive(Clone, Debug)]
struct InformationPhysicsInputs {
    demo: String,
    microstates: f64,
    symbols: usize,
    energy_levels: Vec<f64>,
    temperature: f64,
    boltzmann_constant: f64,
    information_bits: f64,
    channel: Matrix,
    tol: f64,
    max_iter: usize,
    work_samples_j: Vec<f64>,
    delta_free_energy_j: f64,
}

impl ModelCitizen for InformationPhysicsCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "information-physics".to_string(),
            title: "Information Physics".to_string(),
            description: "Animated finite models connecting Boltzmann and Gibbs entropy, \
                          Hartley/Shannon information, Maxwell/Szilard feedback work, \
                          Landauer erasure, and stochastic thermodynamics."
                .to_string(),
            spec_schema: INFORMATION_PHYSICS_SCHEMA.to_string(),
            methods: vec![
                "lineage".to_string(),
                "boltzmann".to_string(),
                "gibbs".to_string(),
                "channel-capacity".to_string(),
                "maxwell-szilard".to_string(),
                "jarzynski".to_string(),
                "all".to_string(),
            ],
            example_spec: starter_information_physics_spec(),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        match catch_unwind(AssertUnwindSafe(|| run_information_physics_checked(spec))) {
            Ok(result) => result,
            Err(payload) => Err(CitizenError::InvalidSpec(format!(
                "information-physics model rejected the spec: {}",
                panic_message(payload)
            ))),
        }
    }
}

/// Minimal valid spec for the animated information-physics citizen.
pub fn starter_information_physics_spec() -> Value {
    json!({
        "$schema": INFORMATION_PHYSICS_SCHEMA,
        "demo": "all",
        "microstates": 16.0,
        "symbols": 4,
        "energyLevels": [0.0, 1.0, 2.0, 3.0],
        "temperature": DEFAULT_TEMPERATURE_K,
        "boltzmannConstant": BOLTZMANN_CONSTANT_J_PER_K,
        "informationBits": 1.0,
        "channel": [[0.9, 0.1], [0.1, 0.9]],
        "workSamplesJ": [2.0e-21, 2.6e-21, 3.1e-21, 3.4e-21, 4.0e-21],
        "deltaFreeEnergyJ": 2.4e-21,
        "tol": 1.0e-10,
        "maxIter": 200
    })
}

fn run_information_physics_checked(spec: &Value) -> Result<RunArtifact, CitizenError> {
    validate_schema(spec)?;
    let inputs = parse_inputs(spec)?;

    let mut frames = Vec::new();
    match inputs.demo.as_str() {
        "lineage" => append_generated_frames(&mut frames, lineage_frames),
        "boltzmann" => append_generated_frames(&mut frames, |base| boltzmann_frames(&inputs, base)),
        "gibbs" => append_generated_frames(&mut frames, |base| gibbs_frames(&inputs, base)),
        "channel-capacity" | "shannon" => {
            append_generated_frames(&mut frames, |base| channel_capacity_frames(&inputs, base))
        }
        "maxwell-szilard" | "szilard" | "landauer" => {
            append_generated_frames(&mut frames, |base| maxwell_szilard_frames(&inputs, base))
        }
        "jarzynski" | "stochastic-thermodynamics" => {
            append_generated_frames(&mut frames, |base| jarzynski_frames(&inputs, base))
        }
        "all" => {
            append_generated_frames(&mut frames, lineage_frames);
            append_generated_frames(&mut frames, |base| boltzmann_frames(&inputs, base));
            append_generated_frames(&mut frames, |base| gibbs_frames(&inputs, base));
            append_generated_frames(&mut frames, |base| channel_capacity_frames(&inputs, base));
            append_generated_frames(&mut frames, |base| maxwell_szilard_frames(&inputs, base));
            append_generated_frames(&mut frames, |base| jarzynski_frames(&inputs, base));
        }
        other => {
            return Err(CitizenError::InvalidSpec(format!(
                "unknown information-physics demo `{other}`"
            )))
        }
    }

    let results = results_json(&inputs);
    let summary = format!(
        "Information physics `{}` run: {} animated frames across entropy, coding, channel, and thermodynamic-work models.",
        inputs.demo,
        frames.len()
    );

    Ok(RunArtifact::sim(
        "information-physics",
        "Information Physics",
        "Boltzmann/Gibbs physical entropy, Hartley/Shannon abstraction, and Maxwell/Szilard/Landauer thermodynamics rendered as one animated model family.",
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &[
                    "all",
                    "entropyBits",
                    "meanEnergy",
                    "capacityBits",
                    "extractableWorkJ",
                    "jarzynskiDeltaFJ",
                ],
                "all",
                Some("metric"),
            ),
        ],
        &summary,
    ))
}

fn validate_schema(spec: &Value) -> Result<(), CitizenError> {
    if let Some(schema) = spec.get("$schema").and_then(Value::as_str) {
        if schema != INFORMATION_PHYSICS_SCHEMA {
            return Err(CitizenError::InvalidSpec(format!(
                "expected $schema `{INFORMATION_PHYSICS_SCHEMA}`, got `{schema}`"
            )));
        }
    }
    Ok(())
}

fn parse_inputs(spec: &Value) -> Result<InformationPhysicsInputs, CitizenError> {
    let demo = string_field(spec, "demo", "all")?;
    let microstates = finite_field(spec, "microstates", 16.0)?;
    if microstates <= 0.0 {
        return invalid("microstates must be > 0");
    }
    let symbols = usize_field(spec, "symbols", 4, 1, MAX_SYMBOLS)?;
    let energy_levels = f64_array_field(
        spec,
        "energyLevels",
        &[0.0, 1.0, 2.0, 3.0],
        1,
        MAX_ENERGY_LEVELS,
    )?;
    let temperature = finite_field(spec, "temperature", DEFAULT_TEMPERATURE_K)?;
    if temperature <= 0.0 {
        return invalid("temperature must be > 0");
    }
    let boltzmann_constant = finite_field(spec, "boltzmannConstant", BOLTZMANN_CONSTANT_J_PER_K)?;
    if boltzmann_constant <= 0.0 {
        return invalid("boltzmannConstant must be > 0");
    }
    let information_bits = finite_field(spec, "informationBits", 1.0)?;
    if information_bits < 0.0 {
        return invalid("informationBits must be >= 0");
    }
    let channel = matrix_field(
        spec,
        "channel",
        &vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        MAX_CHANNEL_DIM,
        MAX_CHANNEL_DIM,
    )?;
    let tol = finite_field(spec, "tol", 1e-10)?;
    if tol <= 0.0 {
        return invalid("tol must be > 0");
    }
    let max_iter = usize_field(spec, "maxIter", 200, 1, 10_000)?;
    let work_samples_j = f64_array_field(
        spec,
        "workSamplesJ",
        &[2.0e-21, 2.6e-21, 3.1e-21, 3.4e-21, 4.0e-21],
        1,
        MAX_WORK_SAMPLES,
    )?;
    let delta_free_energy_j = finite_field(spec, "deltaFreeEnergyJ", 2.4e-21)?;

    Ok(InformationPhysicsInputs {
        demo,
        microstates,
        symbols,
        energy_levels,
        temperature,
        boltzmann_constant,
        information_bits,
        channel,
        tol,
        max_iter,
        work_samples_j,
        delta_free_energy_j,
    })
}

fn string_field(spec: &Value, key: &str, default: &str) -> Result<String, CitizenError> {
    match spec.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => invalid(&format!("{key} must be a string")),
        None => Ok(default.to_string()),
    }
}

fn finite_field(spec: &Value, key: &str, default: f64) -> Result<f64, CitizenError> {
    match spec.get(key) {
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() => Ok(x),
            _ => invalid(&format!("{key} must be a finite number")),
        },
        None => Ok(default),
    }
}

fn usize_field(
    spec: &Value,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, CitizenError> {
    match spec.get(key) {
        Some(v) => match v.as_u64() {
            Some(x) if x >= min as u64 && x <= max as u64 => Ok(x as usize),
            _ => invalid(&format!("{key} must be an integer in [{min}, {max}]")),
        },
        None => Ok(default),
    }
}

fn f64_array_field(
    spec: &Value,
    key: &str,
    default: &[f64],
    min_len: usize,
    max_len: usize,
) -> Result<Vec<f64>, CitizenError> {
    match spec.get(key) {
        Some(Value::Array(values)) => {
            if values.len() < min_len || values.len() > max_len {
                return invalid(&format!("{key} length must be in [{min_len}, {max_len}]"));
            }
            values
                .iter()
                .enumerate()
                .map(|(i, v)| match v.as_f64() {
                    Some(x) if x.is_finite() => Ok(x),
                    _ => invalid(&format!("{key}[{i}] must be a finite number")),
                })
                .collect()
        }
        Some(_) => invalid(&format!("{key} must be an array of numbers")),
        None => Ok(default.to_vec()),
    }
}

fn matrix_field(
    spec: &Value,
    key: &str,
    default: &Matrix,
    max_rows: usize,
    max_cols: usize,
) -> Result<Matrix, CitizenError> {
    match spec.get(key) {
        Some(Value::Array(rows)) => {
            if rows.is_empty() || rows.len() > max_rows {
                return invalid(&format!("{key} row count must be in [1, {max_rows}]"));
            }
            let mut matrix = Vec::with_capacity(rows.len());
            let mut cols = None;
            for (i, row) in rows.iter().enumerate() {
                let Value::Array(cells) = row else {
                    return invalid(&format!("{key}[{i}] must be an array"));
                };
                if cells.is_empty() || cells.len() > max_cols {
                    return invalid(&format!("{key}[{i}] length must be in [1, {max_cols}]"));
                }
                if let Some(expected) = cols {
                    if cells.len() != expected {
                        return invalid(&format!("{key}[{i}] length must equal {expected}"));
                    }
                } else {
                    cols = Some(cells.len());
                }
                let mut parsed = Vec::with_capacity(cells.len());
                for (j, cell) in cells.iter().enumerate() {
                    match cell.as_f64() {
                        Some(x) if x.is_finite() => parsed.push(x),
                        _ => return invalid(&format!("{key}[{i}][{j}] must be finite")),
                    }
                }
                matrix.push(parsed);
            }
            Ok(matrix)
        }
        Some(_) => invalid(&format!("{key} must be a matrix")),
        None => Ok(default.clone()),
    }
}

fn invalid<T>(message: &str) -> Result<T, CitizenError> {
    Err(CitizenError::InvalidSpec(message.to_string()))
}

fn append_frames(out: &mut Vec<Value>, frames: Vec<Value>) {
    for mut frame in frames {
        let tick = out.len();
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("tick".to_string(), json!(tick));
            obj.insert("t".to_string(), json!(tick as f64));
        }
        out.push(frame);
    }
}

fn append_generated_frames(out: &mut Vec<Value>, build: impl FnOnce(usize) -> Vec<Value>) {
    let base = out.len();
    append_frames(out, build(base));
}

fn frame(phase: &str, caption: String, metrics: &[(&str, f64)], shapes: Vec<Value>) -> Value {
    let mut obj = Map::new();
    obj.insert("phase".to_string(), json!(phase));
    obj.insert("caption".to_string(), json!(caption));
    obj.insert("shapes".to_string(), Value::Array(shapes));
    for (key, value) in metrics {
        obj.insert((*key).to_string(), json!(value));
    }
    Value::Object(obj)
}

fn lineage_frames(base: usize) -> Vec<Value> {
    let catalog = information_physics_catalog();
    let mut frames = Vec::new();
    for active in 0..catalog.len() {
        let mut shapes = canvas_title("Abstraction ladder: physics to information");
        for (i, desc) in catalog.iter().enumerate() {
            let x = 42.0 + i as f64 * 150.0;
            let y = 150.0;
            if i > 0 {
                shapes.push(line(x - 26.0, y + 34.0, x - 8.0, y + 34.0, "#94a3b8", 2.0));
            }
            let fill = if i == active { "#dbeafe" } else { "#f8fafc" };
            let stroke = if i == active { "#2563eb" } else { "#cbd5e1" };
            shapes.push(rect(x, y, 132.0, 76.0, fill, stroke, 2.0));
            shapes.push(text(
                x + 66.0,
                y + 24.0,
                desc.predecessor,
                12.0,
                "#0f172a",
                "middle",
                true,
            ));
            shapes.push(text(
                x + 66.0,
                y + 50.0,
                desc.historical_model,
                10.0,
                "#475569",
                "middle",
                false,
            ));
        }
        let desc = &catalog[active];
        shapes.push(rect(88.0, 282.0, 644.0, 102.0, "#ffffff", "#e2e8f0", 1.0));
        shapes.push(text(
            108.0,
            318.0,
            desc.abstraction,
            18.0,
            "#0f172a",
            "start",
            true,
        ));
        shapes.push(text(
            108.0,
            352.0,
            desc.modern_model,
            13.0,
            "#475569",
            "start",
            false,
        ));
        frames.push(frame(
            "lineage",
            format!(
                "Step {} of {}: {}",
                active + 1,
                catalog.len(),
                desc.predecessor
            ),
            &[
                ("abstractionLevel", (active + 1) as f64),
                ("modelsSeen", (active + 1) as f64),
            ],
            shapes,
        ));
    }
    retime_for_tests(frames, base)
}

fn boltzmann_frames(inputs: &InformationPhysicsInputs, base: usize) -> Vec<Value> {
    let steps = 7;
    let mut out = Vec::new();
    for i in 0..steps {
        let a = i as f64 / (steps - 1) as f64;
        let omega = 1.0 + (inputs.microstates - 1.0) * a;
        let summary = boltzmann_entropy(omega, inputs.boltzmann_constant);
        let hartley = if omega >= 1.0 {
            summary.state_count_bits
        } else {
            0.0
        };
        let mut shapes = canvas_title("Boltzmann: entropy from accessible microstates");
        shapes.push(text(
            64.0,
            92.0,
            "Accessible microstates",
            14.0,
            "#334155",
            "start",
            true,
        ));
        let visible = ((a * 48.0).round() as usize).max(1);
        for n in 0..48 {
            let col = n % 12;
            let row = n / 12;
            let fill = if n < visible { "#2563eb" } else { "#e2e8f0" };
            shapes.push(circle(
                78.0 + col as f64 * 26.0,
                128.0 + row as f64 * 26.0,
                8.0,
                fill,
                "#ffffff",
                1.0,
            ));
        }
        shapes.extend(metric_bar(
            450.0,
            120.0,
            "ln(Omega), nats",
            summary.entropy_nats,
            inputs.microstates.ln().max(1.0),
            "#16a34a",
        ));
        shapes.extend(metric_bar(
            450.0,
            210.0,
            "log2(Omega), bits",
            hartley,
            inputs.microstates.log2().max(1.0),
            "#2563eb",
        ));
        shapes.push(text(
            450.0,
            324.0,
            &format!("S = k_B ln(Omega) = {:.3e} J/K", summary.entropy_j_per_k),
            14.0,
            "#0f172a",
            "start",
            false,
        ));
        out.push(frame(
            "boltzmann",
            format!("Omega={omega:.3}; entropy={hartley:.3} bits"),
            &[
                ("microstates", omega),
                ("entropyBits", summary.state_count_bits),
                ("entropyNats", summary.entropy_nats),
                ("entropyJPerK", summary.entropy_j_per_k),
            ],
            shapes,
        ));
    }
    retime_for_tests(out, base)
}

fn gibbs_frames(inputs: &InformationPhysicsInputs, base: usize) -> Vec<Value> {
    let steps = 8;
    let mut out = Vec::new();
    for i in 0..steps {
        let a = i as f64 / (steps - 1) as f64;
        let temp = inputs.temperature * (4.0 - 3.0 * a).max(1.0);
        let summary =
            gibbs_canonical_ensemble(&inputs.energy_levels, temp, inputs.boltzmann_constant);
        let max_p = summary
            .probabilities
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let mut shapes = canvas_title("Gibbs: probability distribution over energy states");
        shapes.push(text(
            72.0,
            100.0,
            &format!("T = {temp:.3e} K"),
            15.0,
            "#334155",
            "start",
            true,
        ));
        let chart_x = 86.0;
        let chart_y = 312.0;
        let bar_w = (500.0 / summary.probabilities.len() as f64).min(76.0);
        for (idx, (&p, &e)) in summary
            .probabilities
            .iter()
            .zip(summary.energy_levels.iter())
            .enumerate()
        {
            let x = chart_x + idx as f64 * (bar_w + 16.0);
            let h = 170.0 * p / max_p;
            shapes.push(rect(x, chart_y - h, bar_w, h, "#38bdf8", "#0284c7", 1.0));
            shapes.push(text(
                x + bar_w / 2.0,
                chart_y + 24.0,
                &format!("E={e:.2}"),
                11.0,
                "#475569",
                "middle",
                false,
            ));
            shapes.push(text(
                x + bar_w / 2.0,
                chart_y - h - 8.0,
                &format!("{p:.2}"),
                11.0,
                "#0f172a",
                "middle",
                false,
            ));
        }
        shapes.extend(metric_bar(
            640.0,
            120.0,
            "entropy, nats",
            summary.entropy_nats,
            (summary.energy_levels.len() as f64).ln().max(1.0),
            "#16a34a",
        ));
        shapes.push(text(
            640.0,
            252.0,
            &format!("mean E = {:.3e}", summary.mean_energy),
            13.0,
            "#0f172a",
            "start",
            false,
        ));
        shapes.push(text(
            640.0,
            282.0,
            &format!("F = {:.3e}", summary.helmholtz_free_energy),
            13.0,
            "#0f172a",
            "start",
            false,
        ));
        out.push(frame(
            "gibbs",
            format!(
                "Canonical ensemble at T={temp:.3e}; entropy={:.3} nats",
                summary.entropy_nats
            ),
            &[
                ("temperature", temp),
                ("meanEnergy", summary.mean_energy),
                ("entropyNats", summary.entropy_nats),
                ("freeEnergy", summary.helmholtz_free_energy),
            ],
            shapes,
        ));
    }
    retime_for_tests(out, base)
}

fn channel_capacity_frames(inputs: &InformationPhysicsInputs, base: usize) -> Vec<Value> {
    let _summary =
        channel_capacity_blahut_arimoto_bits(&inputs.channel, inputs.tol, inputs.max_iter);
    let frames = inputs.max_iter.min(16);
    let inputs_n = inputs.channel.len();
    let outputs_n = inputs.channel[0].len();
    let mut prior = vec![1.0 / inputs_n as f64; inputs_n];
    let mut out = Vec::new();
    for iter in 0..frames {
        let info = crate::des::general::control_systems::information_theory::channel_information(
            &prior,
            &inputs.channel,
        );
        let mut output = vec![0.0; outputs_n];
        for x in 0..inputs_n {
            for y in 0..outputs_n {
                output[y] += prior[x] * inputs.channel[x][y];
            }
        }
        let mut shapes =
            canvas_title("Shannon: maximize reliable information through a noisy channel");
        shapes.push(text(
            74.0,
            96.0,
            "Input prior",
            14.0,
            "#334155",
            "start",
            true,
        ));
        draw_distribution(&mut shapes, 82.0, 300.0, 44.0, &prior, "#2563eb", "x");
        shapes.push(text(
            330.0, 190.0, "P(y|x)", 15.0, "#0f172a", "middle", true,
        ));
        shapes.push(line(230.0, 205.0, 430.0, 205.0, "#94a3b8", 2.0));
        shapes.push(text(
            520.0,
            96.0,
            "Output distribution",
            14.0,
            "#334155",
            "start",
            true,
        ));
        draw_distribution(&mut shapes, 526.0, 300.0, 44.0, &output, "#16a34a", "y");
        shapes.extend(metric_bar(
            285.0,
            292.0,
            "I(X;Y), bits",
            info.mutual_information_bits,
            1.0_f64.max((inputs_n.min(outputs_n) as f64).log2()),
            "#9333ea",
        ));
        out.push(frame(
            "channel-capacity",
            format!(
                "Blahut-Arimoto iteration {iter}; mutual information={:.4} bits",
                info.mutual_information_bits
            ),
            &[
                ("capacityBits", info.mutual_information_bits),
                ("equivocationBits", info.equivocation_bits),
                ("iteration", iter as f64),
            ],
            shapes,
        ));
        prior = blahut_arimoto_step(&prior, &inputs.channel);
    }
    retime_for_tests(out, base)
}

fn maxwell_szilard_frames(inputs: &InformationPhysicsInputs, base: usize) -> Vec<Value> {
    let steps = 7;
    let mut out = Vec::new();
    for i in 0..steps {
        let a = i as f64 / (steps - 1) as f64;
        let bits = inputs.information_bits * a;
        let budget = szilard_landauer_budget(bits, inputs.temperature, inputs.boltzmann_constant);
        let mut shapes = canvas_title("Maxwell/Szilard/Landauer: information as physical work");
        shapes.push(rect(88.0, 142.0, 210.0, 126.0, "#f8fafc", "#cbd5e1", 2.0));
        shapes.push(text(
            193.0, 178.0, "memory", 16.0, "#0f172a", "middle", true,
        ));
        shapes.push(text(
            193.0,
            214.0,
            &format!("{bits:.3} bits"),
            26.0,
            "#2563eb",
            "middle",
            true,
        ));
        shapes.push(line(304.0, 204.0, 420.0, 204.0, "#64748b", 3.0));
        shapes.push(text(
            362.0, 188.0, "feedback", 12.0, "#475569", "middle", false,
        ));
        shapes.push(circle(522.0, 204.0, 74.0, "#fef3c7", "#d97706", 3.0));
        shapes.push(text(
            522.0, 199.0, "engine", 17.0, "#92400e", "middle", true,
        ));
        shapes.push(text(
            522.0,
            228.0,
            "kBT ln 2 per bit",
            11.0,
            "#92400e",
            "middle",
            false,
        ));
        shapes.extend(metric_bar(
            660.0,
            118.0,
            "extractable work",
            budget.max_extractable_work_j,
            szilard_landauer_budget(
                inputs.information_bits.max(1e-12),
                inputs.temperature,
                inputs.boltzmann_constant,
            )
            .max_extractable_work_j
            .max(1e-30),
            "#16a34a",
        ));
        shapes.extend(metric_bar(
            660.0,
            250.0,
            "erasure cost",
            budget.landauer_erasure_work_j,
            szilard_landauer_budget(
                inputs.information_bits.max(1e-12),
                inputs.temperature,
                inputs.boltzmann_constant,
            )
            .landauer_erasure_work_j
            .max(1e-30),
            "#dc2626",
        ));
        out.push(frame(
            "maxwell-szilard",
            format!("{bits:.3} measured bits bound extractable work and erasure cost equally"),
            &[
                ("informationBits", bits),
                ("extractableWorkJ", budget.max_extractable_work_j),
                ("erasureWorkJ", budget.landauer_erasure_work_j),
                ("netWorkBoundJ", budget.reversible_cycle_net_work_bound_j),
            ],
            shapes,
        ));
    }
    retime_for_tests(out, base)
}

fn jarzynski_frames(inputs: &InformationPhysicsInputs, base: usize) -> Vec<Value> {
    let mut out = Vec::new();
    for n in 1..=inputs.work_samples_j.len() {
        let prefix = &inputs.work_samples_j[..n];
        let estimate =
            jarzynski_free_energy_estimate(prefix, inputs.temperature, inputs.boltzmann_constant);
        let mean_work = prefix.iter().sum::<f64>() / prefix.len() as f64;
        let dissipated = mean_work - inputs.delta_free_energy_j;
        let mut shapes = canvas_title("Modern stochastic thermodynamics: Jarzynski estimator");
        shapes.push(text(
            72.0,
            92.0,
            "Work samples",
            14.0,
            "#334155",
            "start",
            true,
        ));
        let max_work = inputs
            .work_samples_j
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1e-30);
        for (i, w) in inputs.work_samples_j.iter().enumerate() {
            let x = 86.0 + i as f64 * 44.0;
            let h = 170.0 * *w / max_work;
            let fill = if i < n { "#2563eb" } else { "#e2e8f0" };
            shapes.push(rect(x, 308.0 - h, 28.0, h, fill, "#cbd5e1", 1.0));
        }
        shapes.extend(metric_bar(
            470.0,
            120.0,
            "mean work",
            mean_work,
            max_work,
            "#2563eb",
        ));
        shapes.extend(metric_bar(
            470.0,
            220.0,
            "Jarzynski delta F",
            estimate,
            max_work,
            "#9333ea",
        ));
        shapes.push(text(
            470.0,
            352.0,
            &format!("dissipated work = {dissipated:.3e} J"),
            14.0,
            "#0f172a",
            "start",
            false,
        ));
        out.push(frame(
            "jarzynski",
            format!("{n} work samples; Jarzynski delta F={estimate:.3e} J"),
            &[
                ("sampleCount", n as f64),
                ("meanWorkJ", mean_work),
                ("jarzynskiDeltaFJ", estimate),
                ("dissipatedWorkJ", dissipated),
            ],
            shapes,
        ));
    }
    retime_for_tests(out, base)
}

fn retime_for_tests(mut frames: Vec<Value>, base: usize) -> Vec<Value> {
    for (i, frame) in frames.iter_mut().enumerate() {
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("localTick".to_string(), json!(i));
            obj.insert("previewTick".to_string(), json!(base + i));
        }
    }
    frames
}

fn blahut_arimoto_step(prior: &[f64], channel: &Matrix) -> Vec<f64> {
    let inputs = channel.len();
    let outputs = channel[0].len();
    let mut output = vec![0.0; outputs];
    for x in 0..inputs {
        for y in 0..outputs {
            output[y] += prior[x] * channel[x][y];
        }
    }
    let mut d = vec![0.0; inputs];
    for x in 0..inputs {
        for y in 0..outputs {
            let w = channel[x][y];
            if w > 0.0 {
                d[x] += w * (w / output[y]).ln();
            }
        }
    }
    let z: f64 = prior
        .iter()
        .zip(d.iter())
        .map(|(&q, &dx)| q * dx.exp())
        .sum();
    let mut next = vec![0.0; inputs];
    for x in 0..inputs {
        next[x] = prior[x] * d[x].exp() / z;
    }
    next
}

fn results_json(inputs: &InformationPhysicsInputs) -> Value {
    let boltzmann = boltzmann_entropy(inputs.microstates, inputs.boltzmann_constant);
    let hartley = hartley_information(inputs.symbols);
    let gibbs = gibbs_canonical_ensemble(
        &inputs.energy_levels,
        inputs.temperature,
        inputs.boltzmann_constant,
    );
    let noneq = nonequilibrium_free_energy(
        &gibbs.probabilities,
        &inputs.energy_levels,
        inputs.temperature,
        inputs.boltzmann_constant,
    );
    let capacity =
        channel_capacity_blahut_arimoto_bits(&inputs.channel, inputs.tol, inputs.max_iter);
    let szilard = szilard_landauer_budget(
        inputs.information_bits,
        inputs.temperature,
        inputs.boltzmann_constant,
    );
    let perfect_joint = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
    let maxwell = maxwell_demon_budget_from_joint(
        &perfect_joint,
        inputs.temperature,
        inputs.boltzmann_constant,
    );
    let stochastic = stochastic_thermodynamics_summary(
        &inputs.work_samples_j,
        inputs.delta_free_energy_j,
        inputs.temperature,
        inputs.boltzmann_constant,
    );
    json!({
        "kind": "information-physics",
        "schema": INFORMATION_PHYSICS_SCHEMA,
        "demo": inputs.demo,
        "catalog": information_physics_catalog().iter().map(|d| json!({
            "predecessor": d.predecessor,
            "historicalModel": d.historical_model,
            "abstraction": d.abstraction,
            "modernModel": d.modern_model,
        })).collect::<Vec<_>>(),
        "boltzmann": {
            "microstates": boltzmann.microstates,
            "entropyNats": boltzmann.entropy_nats,
            "entropyJPerK": boltzmann.entropy_j_per_k,
            "stateCountBits": boltzmann.state_count_bits,
        },
        "hartley": {
            "symbols": hartley.symbols,
            "informationBits": hartley.information_bits,
            "informationNats": hartley.information_nats,
            "shannonEntropyBits": shannon_entropy_bits(&hartley.uniform_distribution),
        },
        "gibbs": {
            "energyLevels": gibbs.energy_levels,
            "temperature": gibbs.temperature,
            "probabilities": gibbs.probabilities,
            "partitionFunction": gibbs.partition_function,
            "meanEnergy": gibbs.mean_energy,
            "entropyNats": gibbs.entropy_nats,
            "entropyJPerK": gibbs.entropy_j_per_k,
            "helmholtzFreeEnergy": gibbs.helmholtz_free_energy,
        },
        "nonequilibriumFreeEnergy": {
            "freeEnergy": noneq.free_energy,
            "equilibriumFreeEnergy": noneq.equilibrium_free_energy,
            "relativeEntropyToEquilibriumNats": noneq.relative_entropy_to_equilibrium_nats,
            "excessFreeEnergy": noneq.excess_free_energy,
        },
        "channelCapacity": {
            "capacityBits": capacity.capacity_bits,
            "inputDistribution": capacity.input_distribution,
            "outputDistribution": capacity.output_distribution,
            "iterations": capacity.iterations,
            "converged": capacity.converged,
        },
        "szilardLandauer": {
            "informationBits": szilard.information_bits,
            "maxExtractableWorkJ": szilard.max_extractable_work_j,
            "landauerErasureWorkJ": szilard.landauer_erasure_work_j,
            "entropyReductionJPerK": szilard.entropy_reduction_j_per_k,
        },
        "maxwellDemonPerfectMeasurement": {
            "informationBits": maxwell.information_bits,
            "maxExtractableWorkJ": maxwell.max_extractable_work_j,
        },
        "stochasticThermodynamics": {
            "samples": stochastic.samples,
            "meanWorkJ": stochastic.mean_work_j,
            "suppliedDeltaFreeEnergyJ": stochastic.supplied_delta_free_energy_j,
            "jarzynskiDeltaFreeEnergyJ": stochastic.jarzynski_delta_free_energy_j,
            "dissipatedWorkJ": stochastic.dissipated_work_j,
            "jarzynskiGapJ": stochastic.jarzynski_gap_j,
            "secondLawSatisfied": stochastic.second_law_satisfied,
        }
    })
}

fn canvas_title(title: &str) -> Vec<Value> {
    vec![
        rect(0.0, 0.0, 820.0, 430.0, "#ffffff", "#e2e8f0", 1.0),
        text(32.0, 42.0, title, 20.0, "#0f172a", "start", true),
        line(32.0, 62.0, 788.0, 62.0, "#e2e8f0", 1.0),
    ]
}

fn draw_distribution(
    shapes: &mut Vec<Value>,
    x0: f64,
    baseline: f64,
    bar_w: f64,
    probabilities: &[f64],
    fill: &str,
    prefix: &str,
) {
    let max_p = probabilities
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    for (i, &p) in probabilities.iter().enumerate() {
        let x = x0 + i as f64 * (bar_w + 12.0);
        let h = 160.0 * p / max_p;
        shapes.push(rect(x, baseline - h, bar_w, h, fill, "#1e293b", 1.0));
        shapes.push(text(
            x + bar_w / 2.0,
            baseline + 22.0,
            &format!("{prefix}{i}"),
            11.0,
            "#475569",
            "middle",
            false,
        ));
        shapes.push(text(
            x + bar_w / 2.0,
            baseline - h - 8.0,
            &format!("{p:.2}"),
            11.0,
            "#0f172a",
            "middle",
            false,
        ));
    }
}

fn metric_bar(x: f64, y: f64, label: &str, value: f64, max: f64, fill: &str) -> Vec<Value> {
    let w = 132.0;
    let ratio = if max > 0.0 && value.is_finite() {
        (value / max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    vec![
        text(x, y, label, 12.0, "#475569", "start", true),
        rect(x, y + 16.0, w, 18.0, "#f1f5f9", "#cbd5e1", 1.0),
        rect(x, y + 16.0, w * ratio, 18.0, fill, fill, 1.0),
        text(
            x,
            y + 52.0,
            &format!("{value:.3e}"),
            12.0,
            "#0f172a",
            "start",
            false,
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
        "rx": 8.0,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width
    })
}

fn circle(x: f64, y: f64, r: f64, fill: &str, stroke: &str, stroke_width: f64) -> Value {
    json!({
        "kind": "circle",
        "x": x,
        "y": y,
        "r": r,
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

fn text(
    x: f64,
    y: f64,
    value: &str,
    font_size: f64,
    fill: &str,
    anchor: &str,
    bold: bool,
) -> Value {
    json!({
        "kind": "text",
        "x": x,
        "y": y,
        "text": value,
        "fontSize": font_size,
        "fill": fill,
        "anchor": anchor,
        "fontWeight": if bold { "bold" } else { "normal" }
    })
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
    fn starter_spec_renders_animation_player() {
        let citizen = InformationPhysicsCitizen;
        let artifact = citizen
            .run_json(&starter_information_physics_spec())
            .expect("starter information-physics spec should run");
        assert_eq!(artifact.kind, "information-physics");
        assert!(!artifact.frames.is_empty());
        assert!(artifact
            .frames
            .iter()
            .any(|f| f["shapes"].as_array().is_some_and(|s| !s.is_empty())));
        let html = artifact.to_player_html();
        assert!(html.contains("plugin-payload"));
        assert!(html.contains("Information Physics"));
    }

    #[test]
    fn individual_demos_have_frames() {
        let citizen = InformationPhysicsCitizen;
        for demo in [
            "lineage",
            "boltzmann",
            "gibbs",
            "channel-capacity",
            "maxwell-szilard",
            "jarzynski",
        ] {
            let mut spec = starter_information_physics_spec();
            spec["demo"] = json!(demo);
            let artifact = citizen
                .run_json(&spec)
                .unwrap_or_else(|e| panic!("demo {demo} failed: {e}"));
            assert!(!artifact.frames.is_empty(), "demo {demo}");
            assert_eq!(artifact.player, crate::des::plugin::PlayerKind::Sim);
        }
    }

    #[test]
    fn invalid_specs_return_citizen_errors() {
        let citizen = InformationPhysicsCitizen;
        let bad_schema = json!({"$schema": "wrong"});
        assert!(matches!(
            citizen.run_json(&bad_schema),
            Err(CitizenError::InvalidSpec(_))
        ));

        let bad_channel = json!({
            "$schema": INFORMATION_PHYSICS_SCHEMA,
            "demo": "channel-capacity",
            "channel": [[0.7, 0.3], [1.0]]
        });
        assert!(matches!(
            citizen.run_json(&bad_channel),
            Err(CitizenError::InvalidSpec(_))
        ));
    }
}
