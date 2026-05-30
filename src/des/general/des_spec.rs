//! Port of `src/des/general/des-spec.ts` — module `des::general::des_spec`.
//!
//! JSON specification format for runnable DES models: a thin envelope
//! (`DESModelSpec`) carrying a registered model id plus model-specific
//! parameters and runtime/output settings, together with a tiny declarative
//! parameter validator ([`validate`]) used by the registry to type-check JSON
//! params before a model runs.
//!
//! Conversion notes from the TS source:
//!   * `ParamSchema` is a discriminated union (`kind`) -> Rust [`ParamSchema`]
//!     enum, matched with `match`.
//!   * TS `unknown` / `Record<string, unknown>` values -> [`JsonValue`] /
//!     [`JsonObject`]. `serde` / `serde_json` are NOT available dependencies in
//!     this crate, so a minimal self-contained JSON value type is defined here
//!     instead of `serde_json::Value`. `JsonObject` preserves insertion order
//!     so validation/iteration order matches the TS `Record` behaviour.
//!   * `DESModelRegistration<P, R>` -> a trait with `run` / `summarize` /
//!     `animate` / `write_csv` / `examples` methods (callbacks become methods,
//!     not field closures). `run`'s `R | Promise<R>` collapses to a sync `R`.
//!   * `zodSchema?: ZodType<P>` is DROPPED — `zod` has no Rust analogue and was
//!     an optional stricter validator layered on top of `ParamSchema`.

use std::collections::BTreeMap;

/// The required value of `DESModelSpec::schema` (the TS `$schema` literal).
pub const DES_MODEL_SPEC_SCHEMA: &str = "des/model-spec/v1";

// =============================================================================
// Minimal JSON value model (stand-in for `serde_json::Value`).
// =============================================================================

/// A JSON value. `Undefined` models the JavaScript `undefined` (an absent
/// value), kept distinct from `Null` because the TS validator treats them
/// alike in some branches but not in others.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(JsonObject),
}

/// An insertion-ordered JSON object (mirrors a JS object / TS `Record`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct JsonObject {
    entries: Vec<(String, JsonValue)>,
}

impl JsonObject {
    pub fn new() -> Self {
        JsonObject {
            entries: Vec::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Set a key, preserving position on overwrite and appending otherwise
    /// (JS assignment semantics).
    pub fn insert(&mut self, key: String, value: JsonValue) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }
}

impl JsonValue {
    /// `JSON.stringify(self)` equivalent. `Undefined`/non-finite numbers
    /// serialise to `null`, matching JS.
    fn to_json(&self) -> String {
        match self {
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
                let inner: Vec<String> = items.iter().map(|v| v.to_json()).collect();
                format!("[{}]", inner.join(","))
            }
            JsonValue::Object(obj) => {
                let inner: Vec<String> = obj
                    .entries
                    .iter()
                    .map(|(k, v)| format!("{}:{}", json_quote(k), v.to_json()))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }
}

/// JSON-escape and quote a string (`JSON.stringify` of a string).
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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `typeOf(v)` from the TS source.
fn type_of(v: &JsonValue) -> &'static str {
    match v {
        JsonValue::Null => "null",
        JsonValue::Array(_) => "array",
        JsonValue::Undefined => "undefined",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Object(_) => "object",
    }
}

// =============================================================================
// Parameter schema — a tiny declarative validator.
// =============================================================================

/// One variant of a `oneOf` schema.
#[derive(Clone, Debug)]
pub struct OneOfVariant {
    pub tag: String,
    /// Field holding the discriminant tag. Defaults to `"kind"`.
    pub tag_field: Option<String>,
    pub schema: ParamSchema,
    pub description: Option<String>,
}

/// The TS `ParamSchema` discriminated union. `fields` of an object schema is an
/// ordered `Vec` to preserve declaration order during validation.
#[derive(Clone, Debug)]
pub enum ParamSchema {
    Number {
        min: Option<f64>,
        max: Option<f64>,
        integer: Option<bool>,
        default: Option<f64>,
        description: Option<String>,
    },
    String {
        /// Allowed values (the TS `enum` field; renamed, `enum` is reserved).
        allowed: Option<Vec<String>>,
        default: Option<String>,
        description: Option<String>,
    },
    Boolean {
        default: Option<bool>,
        description: Option<String>,
    },
    Array {
        items: Box<ParamSchema>,
        min_length: Option<usize>,
        max_length: Option<usize>,
        description: Option<String>,
    },
    Object {
        fields: Vec<(String, ParamSchema)>,
        required: Option<Vec<String>>,
        description: Option<String>,
    },
    OneOf {
        variants: Vec<OneOfVariant>,
        description: Option<String>,
    },
}

/// Result of validating params against a schema.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    /// Validated parameters with defaults filled in. Only present if valid.
    pub value: Option<JsonValue>,
}

/// Validate `value` against `schema`, rooting error paths at `$`.
pub fn validate(value: &JsonValue, schema: &ParamSchema) -> ValidationResult {
    validate_at(value, schema, "$")
}

/// Validate `value` against `schema`, rooting error paths at `path`.
pub fn validate_at(value: &JsonValue, schema: &ParamSchema, path: &str) -> ValidationResult {
    let mut errors: Vec<String> = Vec::new();
    let v = validate_inner(value, schema, path, &mut errors);
    let valid = errors.is_empty();
    ValidationResult {
        valid,
        value: if valid { Some(v) } else { None },
        errors,
    }
}

fn validate_inner(
    value: &JsonValue,
    schema: &ParamSchema,
    path: &str,
    errors: &mut Vec<String>,
) -> JsonValue {
    match schema {
        ParamSchema::Number {
            min,
            max,
            integer,
            default,
            ..
        } => {
            let v = match value {
                JsonValue::Undefined | JsonValue::Null => default
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Undefined),
                other => other.clone(),
            };
            let num = match &v {
                JsonValue::Number(n) if !n.is_nan() => *n,
                _ => {
                    errors.push(format!("{}: expected number, got {}", path, type_of(value)));
                    return v;
                }
            };
            if *integer == Some(true) && num.fract() != 0.0 {
                errors.push(format!("{}: expected integer, got {}", path, num));
            }
            if let Some(mn) = min {
                if num < *mn {
                    errors.push(format!("{}: {} < min {}", path, num, mn));
                }
            }
            if let Some(mx) = max {
                if num > *mx {
                    errors.push(format!("{}: {} > max {}", path, num, mx));
                }
            }
            v
        }
        ParamSchema::String {
            allowed, default, ..
        } => {
            let v = match value {
                JsonValue::Undefined | JsonValue::Null => default
                    .clone()
                    .map(JsonValue::String)
                    .unwrap_or(JsonValue::Undefined),
                other => other.clone(),
            };
            let s = match &v {
                JsonValue::String(s) => s.clone(),
                _ => {
                    errors.push(format!("{}: expected string, got {}", path, type_of(value)));
                    return v;
                }
            };
            if let Some(allowed) = allowed {
                if !allowed.contains(&s) {
                    let list = allowed
                        .iter()
                        .map(|x| json_quote(x))
                        .collect::<Vec<_>>()
                        .join(", ");
                    errors.push(format!("{}: {} not in [{}]", path, json_quote(&s), list));
                }
            }
            v
        }
        ParamSchema::Boolean { default, .. } => {
            let v = match value {
                JsonValue::Undefined | JsonValue::Null => {
                    default.map(JsonValue::Bool).unwrap_or(JsonValue::Undefined)
                }
                other => other.clone(),
            };
            match &v {
                JsonValue::Bool(_) => {}
                _ => {
                    errors.push(format!(
                        "{}: expected boolean, got {}",
                        path,
                        type_of(value)
                    ));
                }
            }
            v
        }
        ParamSchema::Array {
            items,
            min_length,
            max_length,
            ..
        } => match value {
            JsonValue::Undefined | JsonValue::Null => {
                errors.push(format!("{}: required array missing", path));
                JsonValue::Array(Vec::new())
            }
            JsonValue::Array(arr) => {
                if let Some(ml) = min_length {
                    if arr.len() < *ml {
                        errors.push(format!("{}: length {} < {}", path, arr.len(), ml));
                    }
                }
                if let Some(ml) = max_length {
                    if arr.len() > *ml {
                        errors.push(format!("{}: length {} > {}", path, arr.len(), ml));
                    }
                }
                let mapped = arr
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        validate_inner(item, items, &format!("{}[{}]", path, i), errors)
                    })
                    .collect();
                JsonValue::Array(mapped)
            }
            other => {
                errors.push(format!("{}: expected array, got {}", path, type_of(other)));
                other.clone()
            }
        },
        ParamSchema::Object {
            fields, required, ..
        } => {
            let obj = match value {
                JsonValue::Object(o) => o,
                _ => {
                    errors.push(format!("{}: expected object, got {}", path, type_of(value)));
                    return value.clone();
                }
            };
            let mut out = JsonObject::new();
            let required: Vec<String> = required
                .clone()
                .unwrap_or_else(|| fields.iter().map(|(k, _)| k.clone()).collect());
            for (key, sub) in fields {
                let present = obj.contains_key(key);
                if !present && !required.contains(key) {
                    // Missing optional: set default if any (errors discarded).
                    let mut sink: Vec<String> = Vec::new();
                    out.insert(
                        key.clone(),
                        validate_inner(
                            &JsonValue::Undefined,
                            sub,
                            &format!("{}.{}", path, key),
                            &mut sink,
                        ),
                    );
                    continue;
                }
                if !present && required.contains(key) {
                    // Try default; otherwise error.
                    let mut probe: Vec<String> = Vec::new();
                    let probed = validate_inner(
                        &JsonValue::Undefined,
                        sub,
                        &format!("{}.{}", path, key),
                        &mut probe,
                    );
                    if probe.is_empty() {
                        out.insert(key.clone(), probed);
                        continue;
                    }
                    errors.push(format!("{}: missing required field \".{}\"", path, key));
                    continue;
                }
                let item = obj.get(key).unwrap();
                out.insert(
                    key.clone(),
                    validate_inner(item, sub, &format!("{}.{}", path, key), errors),
                );
            }
            // Allow unknown fields (passed through).
            for key in obj.keys() {
                if !fields.iter().any(|(k, _)| k == key) {
                    out.insert(key.clone(), obj.get(key).unwrap().clone());
                }
            }
            JsonValue::Object(out)
        }
        ParamSchema::OneOf { variants, .. } => {
            // TS: `typeof value !== 'object' || value === null` -> error.
            // Arrays are typeof 'object' in JS, so they fall through here.
            match value {
                JsonValue::Object(_) | JsonValue::Array(_) => {}
                _ => {
                    errors.push(format!(
                        "{}: expected one of (object), got {}",
                        path,
                        type_of(value)
                    ));
                    return value.clone();
                }
            }
            let tag_field = variants
                .first()
                .and_then(|v| v.tag_field.clone())
                .unwrap_or_else(|| "kind".to_string());
            let tag: Option<&JsonValue> = match value {
                JsonValue::Object(o) => o.get(&tag_field),
                _ => None,
            };
            let tag_str: Option<&str> = match tag {
                Some(JsonValue::String(s)) => Some(s.as_str()),
                _ => None,
            };
            let variant = variants.iter().find(|v| Some(v.tag.as_str()) == tag_str);
            match variant {
                None => {
                    let list = variants
                        .iter()
                        .map(|v| json_quote(&v.tag))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let tag_repr = match tag {
                        None | Some(JsonValue::Undefined) => "undefined".to_string(),
                        Some(v) => v.to_json(),
                    };
                    errors.push(format!(
                        "{}: {} {} not in [{}]",
                        path, tag_field, tag_repr, list
                    ));
                    value.clone()
                }
                Some(variant) => validate_inner(
                    value,
                    &variant.schema,
                    &format!("{}<{}>", path, variant.tag),
                    errors,
                ),
            }
        }
    }
}

// =============================================================================
// Spec envelope & model registration.
// =============================================================================

/// Top-level envelope. `P` is the model-specific parameter type.
#[derive(Clone, Debug)]
pub struct DESModelSpec<P = JsonValue> {
    /// Spec format version. Must equal [`DES_MODEL_SPEC_SCHEMA`].
    pub schema: String,
    /// Registered model id (looked up in the model registry).
    pub model: String,
    pub description: Option<String>,
    /// Model-specific parameters, validated against the registered schema.
    pub parameters: P,
    pub runtime: Option<DESRuntimeConfig>,
    pub metadata: Option<DESModelMetadata>,
}

#[derive(Clone, Debug, Default)]
pub struct DESRuntimeConfig {
    /// Deterministic random seed for the run.
    pub seed: Option<f64>,
    /// If `Some(false)`, suppress animation even when supported. Defaults to true.
    pub animate: Option<bool>,
    pub outputs: Option<DESOutputs>,
    /// If `Some(false)`, suppress informational console output. Defaults to true.
    pub verbose: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct DESOutputs {
    /// CSV trace path.
    pub csv: Option<String>,
    /// HTML animation path. If set, the model's animator is invoked.
    pub html: Option<String>,
    /// JSONL frames file path (defaults to `html` with `.frames.jsonl` ext).
    pub frames: Option<String>,
    /// JSON summary path.
    pub summary: Option<String>,
    /// JSONL observability log path.
    pub log: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct DESModelMetadata {
    pub author: Option<String>,
    /// ISO 8601 timestamp.
    pub created_at: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

/// One copy-paste example spec attached to a registration.
pub struct RegistrationExample<P> {
    pub name: String,
    pub spec: DESModelSpec<P>,
}

/// What an adapter must provide to be usable from JSON. The TS interface's
/// callback fields (`run` / `summarize` / `animate` / `writeCsv`) become trait
/// methods; `animate` / `write_csv` / `examples` have no-op / empty defaults to
/// mirror their optionality. `zodSchema` is dropped (see module docs).
pub trait DESModelRegistration<P, R> {
    /// Stable id used in JSON's `"model"` field.
    fn id(&self) -> &str;
    /// One-line summary.
    fn description(&self) -> &str;
    /// Schema for validating the parameters object.
    fn schema(&self) -> ParamSchema;
    /// Run the model.
    fn run(&self, params: P, runtime: &DESRuntimeConfig) -> R;
    /// Render a one-page human-readable summary of the result.
    fn summarize(&self, result: &R, params: &P) -> String;
    /// Optional animation hook. Receives the result and writes outputs.
    fn animate(&self, _result: &R, _params: &P, _runtime: &DESRuntimeConfig) {}
    /// Optional CSV writer.
    fn write_csv(&self, _result: &R, _csv_path: &str) {}
    /// Optional examples (each a complete spec the user can copy).
    fn examples(&self) -> Vec<RegistrationExample<P>> {
        Vec::new()
    }
}

// =============================================================================
// Result type returned by `run_from_spec`.
// =============================================================================

/// Kind of output file written during a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Csv,
    Html,
    Frames,
    Summary,
    Log,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputEntry {
    pub kind: OutputKind,
    pub path: String,
}

#[derive(Clone, Debug)]
pub struct DESRunSummary {
    pub model_id: String,
    /// The params that were actually used (after defaults filled in).
    pub params: JsonValue,
    /// Wall-clock run time in ms.
    pub runtime_ms: f64,
    /// Model-specific result (whatever `run` returned, type-erased).
    pub result: JsonValue,
    /// Human-readable summary lines.
    pub summary_text: String,
    /// Files that were written.
    pub outputs: Vec<OutputEntry>,
}

/// Convenience builder for a `JsonObject` from key/value pairs (test helper and
/// general ergonomics; not present in the TS source). Kept generic over
/// iterables of `(String, JsonValue)`.
impl FromIterator<(String, JsonValue)> for JsonObject {
    fn from_iter<I: IntoIterator<Item = (String, JsonValue)>>(iter: I) -> Self {
        let mut obj = JsonObject::new();
        for (k, v) in iter {
            obj.insert(k, v);
        }
        obj
    }
}

/// Build a `JsonObject` from an ordinary `BTreeMap` (convenience only).
impl From<BTreeMap<String, JsonValue>> for JsonObject {
    fn from(map: BTreeMap<String, JsonValue>) -> Self {
        map.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
        JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    #[test]
    fn number_with_default_and_bounds() {
        let schema = ParamSchema::Number {
            min: Some(0.0),
            max: Some(10.0),
            integer: Some(true),
            default: Some(3.0),
            description: None,
        };
        // Missing -> default filled in.
        let r = validate(&JsonValue::Undefined, &schema);
        assert!(r.valid);
        assert_eq!(r.value, Some(JsonValue::Number(3.0)));

        // Out of range + non-integer -> two errors.
        let bad = validate(&JsonValue::Number(11.5), &schema);
        assert!(!bad.valid);
        assert_eq!(bad.errors.len(), 2);
        assert!(bad.errors[0].contains("expected integer"));
        assert!(bad.errors[1].contains("> max 10"));
    }

    #[test]
    fn object_required_and_string_enum() {
        let schema = ParamSchema::Object {
            fields: vec![
                (
                    "mode".to_string(),
                    ParamSchema::String {
                        allowed: Some(vec!["heat".to_string(), "cool".to_string()]),
                        default: None,
                        description: None,
                    },
                ),
                (
                    "gain".to_string(),
                    ParamSchema::Number {
                        min: None,
                        max: None,
                        integer: None,
                        default: Some(1.0),
                        description: None,
                    },
                ),
            ],
            required: Some(vec!["mode".to_string()]),
            description: None,
        };

        // Valid: required present, optional defaulted, unknown passed through.
        let ok = validate(
            &obj(vec![
                ("mode", JsonValue::String("heat".to_string())),
                ("extra", JsonValue::Bool(true)),
            ]),
            &schema,
        );
        assert!(ok.valid, "errors: {:?}", ok.errors);
        if let Some(JsonValue::Object(o)) = ok.value {
            assert_eq!(o.get("gain"), Some(&JsonValue::Number(1.0)));
            assert_eq!(o.get("extra"), Some(&JsonValue::Bool(true)));
        } else {
            panic!("expected object value");
        }

        // Missing required + bad enum value.
        let bad = validate(
            &obj(vec![("mode", JsonValue::String("off".to_string()))]),
            &schema,
        );
        assert!(!bad.valid);
        assert!(bad.errors[0].contains("not in"));
    }

    #[test]
    fn one_of_dispatches_on_tag() {
        let schema = ParamSchema::OneOf {
            variants: vec![OneOfVariant {
                tag: "linear".to_string(),
                tag_field: None,
                schema: ParamSchema::Object {
                    fields: vec![(
                        "slope".to_string(),
                        ParamSchema::Number {
                            min: None,
                            max: None,
                            integer: None,
                            default: Some(0.0),
                            description: None,
                        },
                    )],
                    required: None,
                    description: None,
                },
                description: None,
            }],
            description: None,
        };

        let ok = validate(
            &obj(vec![
                ("kind", JsonValue::String("linear".to_string())),
                ("slope", JsonValue::Number(2.0)),
            ]),
            &schema,
        );
        assert!(ok.valid, "errors: {:?}", ok.errors);

        let bad = validate(
            &obj(vec![("kind", JsonValue::String("quadratic".to_string()))]),
            &schema,
        );
        assert!(!bad.valid);
        assert!(bad.errors[0].contains("kind \"quadratic\" not in"));
    }
}
