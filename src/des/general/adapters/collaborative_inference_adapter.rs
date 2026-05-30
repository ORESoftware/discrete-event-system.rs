//! Port of `src/des/general/adapters/collaborative-inference-adapter.ts`
//! (module `des::general::adapters::collaborative_inference_adapter`).
//!
//! JSON adapter for sparse collaborative preference inference.
//!
//! ## Conversion notes
//!
//!   * `CollaborativeInferenceParams` / `Result` reuse the engine structs.
//!   * `.toLocaleString()` thousands grouping -> [`to_locale_usize`].
//!   * `label.padEnd(24)` -> `format!("{:<24}")`.
//!   * `scenario` string literal union -> [`CollaborativeInferenceScenario`].
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; the adapter is
//! exposed via [`adapter()`].
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and the `collaborativeInferenceScene` builder) is not
//! ported, so `animate` is a no-op here.
//!
//! PORT NOTE: `.toLocaleString()` is reproduced as comma-grouped thousands
//! (the en-US default); other locales would differ.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, validation_line, write_csv_lines};
use crate::des::general::collaborative_inference::{
    run_collaborative_inference, CollaborativeInferenceParams, CollaborativeInferenceResult,
    CollaborativeInferenceScenario,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `(value).toLocaleString()` for a non-negative integer (en-US grouping).
fn to_locale_usize(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn str_enum(allowed: &[&str], default: Option<&str>) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: default.map(|s| s.to_string()),
        description: None,
    }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn boolean(default: Option<bool>) -> ParamSchema {
    ParamSchema::Boolean { default, description: None }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn item_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("label", string_field()),
            ("group", string_field()),
            ("latentUtility", num(Some(0.0), Some(1.0), None, Some(0.5))),
            ("exposure", num(Some(0.0), None, None, Some(1.0))),
            ("priorScore", num(Some(0.0), Some(1.0), None, None)),
        ],
        vec!["id"],
    )
}

fn response_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("itemIds", arr(string_field(), Some(0))),
            ("ratings", obj(vec![], vec![])),
            ("ranking", arr(string_field(), Some(0))),
            ("age", num(Some(0.0), None, None, None)),
            ("experienceYears", obj(vec![], vec![])),
            ("weight", num(Some(0.0), None, None, Some(1.0))),
            ("segment", string_field()),
        ],
        vec![],
    )
}

fn collaborative_schema() -> ParamSchema {
    obj(
        vec![
            (
                "scenario",
                str_enum(
                    &["programming-languages", "model-validation", "learning-resources", "custom"],
                    Some("programming-languages"),
                ),
            ),
            ("items", arr(item_schema(), Some(0))),
            ("responses", arr(response_schema(), Some(0))),
            ("respondentCount", num(Some(1.0), None, Some(true), None)),
            ("respondents", num(Some(1.0), None, Some(true), None)),
            ("minItemsPerRespondent", num(Some(1.0), None, Some(true), None)),
            ("maxItemsPerRespondent", num(Some(1.0), None, Some(true), None)),
            ("respondentsPerTick", num(Some(1.0), None, Some(true), Some(100.0))),
            ("ratingMin", num(None, None, None, None)),
            ("ratingMax", num(None, None, None, None)),
            ("noiseStd", num(Some(0.0), None, None, None)),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("ratingWeight", num(Some(0.0), None, None, Some(0.55))),
            ("rankingWeight", num(Some(0.0), None, None, Some(0.45))),
            ("shrinkage", num(Some(0.0), None, None, Some(12.0))),
            ("topK", num(Some(1.0), None, Some(true), Some(10.0))),
            ("credibilityWeighting", boolean(Some(true))),
            ("credibilityPasses", num(Some(1.0), None, Some(true), Some(2.0))),
            ("minCredibleAge", num(Some(0.0), None, None, Some(15.0))),
            ("referenceAge", num(Some(1.0), None, None, Some(50.0))),
            ("referenceExperienceYears", num(Some(1.0), None, None, Some(15.0))),
            ("ageWeightStrength", num(Some(0.0), None, None, Some(0.35))),
            ("experienceWeightStrength", num(Some(0.0), None, None, Some(0.6))),
            ("highRatedBreadthStrength", num(Some(0.0), None, None, Some(0.4))),
            ("highRatedScoreThreshold", num(Some(0.0), Some(1.0), None, Some(0.72))),
            ("minHighRatedItems", num(Some(1.0), None, Some(true), Some(2.0))),
            ("maxCredibilityMultiplier", num(Some(1.0), None, None, Some(3.0))),
        ],
        vec![],
    )
}

// =============================================================================
// Adapter
// =============================================================================

pub struct CollaborativeInferenceAdapter;

pub fn adapter() -> CollaborativeInferenceAdapter {
    CollaborativeInferenceAdapter
}

fn example(
    name: &str,
    description: &str,
    parameters: CollaborativeInferenceParams,
) -> RegistrationExample<CollaborativeInferenceParams> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: "collaborative-inference".to_string(),
            description: Some(description.to_string()),
            parameters,
            runtime: None,
            metadata: None,
        },
    }
}

impl DESModelRegistration<CollaborativeInferenceParams, CollaborativeInferenceResult>
    for CollaborativeInferenceAdapter
{
    fn id(&self) -> &str {
        "collaborative-inference"
    }
    fn description(&self) -> &str {
        "Sparse subjective ratings/rankings fused into a global item ranking with station-graph evidence aggregation."
    }
    fn schema(&self) -> ParamSchema {
        collaborative_schema()
    }
    fn run(
        &self,
        params: CollaborativeInferenceParams,
        _runtime: &DESRuntimeConfig,
    ) -> CollaborativeInferenceResult {
        run_collaborative_inference(params)
    }
    fn summarize(
        &self,
        result: &CollaborativeInferenceResult,
        _params: &CollaborativeInferenceParams,
    ) -> String {
        let credibility = if result.credibility.enabled {
            format!(
                "{} pass(es), mean weight={:.3}, max={:.3}, capped claims={}, breadth bonuses={}",
                result.credibility.passes,
                result.credibility.mean_respondent_weight,
                result.credibility.max_respondent_weight,
                result.credibility.capped_experience_claims,
                result.credibility.high_rated_bonus_respondents
            )
        } else {
            "disabled".to_string()
        };

        let mut lines: Vec<String> = vec![
            "COLLABORATIVE INFERENCE (sparse preference learning DES)".to_string(),
            "-------------------------------------------------------".to_string(),
            format!("  Scenario:       {}", result.scenario_label),
            format!("  Respondents:    {}", to_locale_usize(result.respondents_processed)),
            format!("  Ratings:        {}", to_locale_usize(result.rating_evidence_count)),
            format!("  Comparisons:    {}", to_locale_usize(result.pairwise_evidence_count)),
            format!(
                "  Coverage:       ratings {}/{}, comparisons {}/{}",
                result.coverage.items_with_ratings,
                result.coverage.items,
                result.coverage.items_with_comparisons,
                result.coverage.items
            ),
            format!("  Credibility:    {credibility}"),
            format!("  Validation:     {}", validation_line(&result.validation)),
            String::new(),
            "  Top inferred items:".to_string(),
        ];
        for row in &result.top {
            lines.push(format!(
                "    {}. {:<24} score={:.3} confidence={:.3} ratings={}",
                row.rank, row.label, row.score, row.confidence, row.rating_count
            ));
        }
        lines.push(String::new());
        lines.push(format!("  Sources:        {}", result.station_roles.sources.join(", ")));
        lines.push(format!("  Stations:       {}", result.station_roles.stations.join(" -> ")));
        lines.push(format!("  Sinks:          {}", result.station_roles.sinks.join(", ")));
        lines.push(format!("  Movables:       {}", result.station_roles.movables.join(", ")));
        lines.join("\n")
    }
    fn write_csv(&self, result: &CollaborativeInferenceResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "rank",
            "item_id",
            "label",
            "group",
            "score",
            "confidence",
            "uncertainty",
            "rating_mean",
            "rating_count",
            "comparison_count",
            "pairwise_win_rate",
        ])];
        for row in &result.rankings {
            lines.push(csv_row([
                row.rank.to_string(),
                row.item_id.clone(),
                row.label.clone(),
                row.group.clone().unwrap_or_default(),
                js_number(row.score),
                js_number(row.confidence),
                js_number(row.uncertainty),
                js_number(row.rating_mean),
                row.rating_count.to_string(),
                js_number(row.comparison_count),
                js_number(row.pairwise_win_rate),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _result: &CollaborativeInferenceResult,
        _params: &CollaborativeInferenceParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<CollaborativeInferenceParams>> {
        vec![
            example(
                "programming-languages",
                "Rank 50 programming languages from 10,000 sparse developer ratings/rankings.",
                CollaborativeInferenceParams {
                    scenario: Some(CollaborativeInferenceScenario::ProgrammingLanguages),
                    respondent_count: Some(10000),
                    min_items_per_respondent: Some(4),
                    max_items_per_respondent: Some(5),
                    seed: Some(7),
                    top_k: Some(12),
                    ..Default::default()
                },
            ),
            example(
                "model-validation",
                "Rank model execution and validation workflows from sparse external reviewer feedback.",
                CollaborativeInferenceParams {
                    scenario: Some(CollaborativeInferenceScenario::ModelValidation),
                    respondent_count: Some(800),
                    min_items_per_respondent: Some(3),
                    max_items_per_respondent: Some(5),
                    seed: Some(11),
                    top_k: Some(8),
                    ..Default::default()
                },
            ),
            example(
                "learning-resources",
                "Rank learning resources from sparse student ratings/rankings.",
                CollaborativeInferenceParams {
                    scenario: Some(CollaborativeInferenceScenario::LearningResources),
                    respondent_count: Some(1200),
                    min_items_per_respondent: Some(3),
                    max_items_per_respondent: Some(4),
                    seed: Some(5),
                    top_k: Some(8),
                    ..Default::default()
                },
            ),
        ]
    }
}
