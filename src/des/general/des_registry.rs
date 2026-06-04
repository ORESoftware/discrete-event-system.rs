//! Port of `src/des/general/des-registry.ts`
//! (module `des::general::des_registry`).
//!
//! Runtime registry that maps model ids to runnable adapters, plus the
//! `run_from_spec()` / `run_from_json_file()` drivers that JSON files (and the
//! `main-from-json` CLI) call into.
//!
//! ## Conversion notes
//!
//!   * `REGISTRY: Map<string, DESModelRegistration<any,any>>` becomes an owned
//!     [`Registry`] over type-erased [`ModelAdapter`] trait objects (the `<any,
//!     any>` registrations erase `P`/`R` to [`JsonValue`]). The driver feeds and
//!     reads `JsonValue`, matching [`DESRunSummary`]'s erased fields.
//!   * `throw` for an unknown / duplicate model id or a validation failure →
//!     `Result<_, RegistryError>`.
//!   * Node `fs`/`path` → `std::fs`/`std::path`; `JSON.parse` /
//!     `JSON.stringify(_, null, 2)` → the self-contained [`parse_json`] /
//!     [`to_pretty_json`] helpers (no `serde` dependency in this crate).
//!   * `zodSchema` validation is dropped (the trait has no zod hook); only the
//!     lightweight [`ParamSchema`] path remains.
//!
//! PORT NOTE: the TS file's tail auto-registers ~23 built-in adapters via
//! `import './adapters/...'` (import side-effects). Rust has no import
//! side-effects, and the engine's concrete adapters implement the *typed*
//! `DESModelRegistration<P, R>` rather than the erased [`ModelAdapter`] used
//! here, so there is no faithful auto-registration. Callers register adapters
//! explicitly via [`Registry::register_model`] (a typed adapter can be bridged
//! by a thin `ModelAdapter` wrapper once a `JsonValue <-> P` codec exists).

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::Instant;

use crate::des::general::adapters::adapter_utils::default_frames_path;
use crate::des::general::des_spec::{
    validate, DESModelMetadata, DESModelSpec, DESOutputs, DESRunSummary, DESRuntimeConfig,
    JsonObject, JsonValue, OutputEntry, OutputKind, ParamSchema, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::universal_model_spec::is_universal_des_model_spec;

// =============================================================================
// Type-erased adapter contract.
// =============================================================================

/// The erased registration the registry stores (the TS
/// `DESModelRegistration<any, any>`). Params and results are [`JsonValue`].
pub trait ModelAdapter {
    fn id(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> ParamSchema;
    fn run(&self, params: JsonValue, runtime: &DESRuntimeConfig) -> JsonValue;
    fn summarize(&self, result: &JsonValue, params: &JsonValue) -> String;
    fn animate(&self, _result: &JsonValue, _params: &JsonValue, _runtime: &DESRuntimeConfig) {}
    fn write_csv(&self, _result: &JsonValue, _csv_path: &str) {}
    /// Whether [`ModelAdapter::animate`] does real work (TS `reg.animate` truthy).
    fn has_animate(&self) -> bool {
        false
    }
    /// Whether [`ModelAdapter::write_csv`] does real work (TS `reg.writeCsv` truthy).
    fn has_write_csv(&self) -> bool {
        false
    }
}

// =============================================================================
// Errors (TS `throw` sites → typed errors).
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    AlreadyRegistered(String),
    UnknownModel {
        id: String,
        registered: Vec<String>,
    },
    UnknownSchema(String),
    InvalidParameters {
        model_id: String,
        errors: Vec<String>,
    },
    Io(String),
    Parse(String),
    Unsupported(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::AlreadyRegistered(id) => write!(f, "model \"{id}\" already registered"),
            RegistryError::UnknownModel { id, registered } => {
                write!(
                    f,
                    "unknown model \"{id}\". Registered: [{}]",
                    registered.join(", ")
                )
            }
            RegistryError::UnknownSchema(s) => {
                write!(
                    f,
                    "unknown $schema \"{s}\". Expected \"{DES_MODEL_SPEC_SCHEMA}\"."
                )
            }
            RegistryError::InvalidParameters { model_id, errors } => {
                write!(
                    f,
                    "invalid parameters for model \"{model_id}\":\n  {}",
                    errors.join("\n  ")
                )
            }
            RegistryError::Io(s) => write!(f, "{s}"),
            RegistryError::Parse(s) => write!(f, "{s}"),
            RegistryError::Unsupported(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for RegistryError {}

// =============================================================================
// Registry.
// =============================================================================

/// Owned registry of erased adapters (the TS module-level `REGISTRY` map).
/// Insertion order is preserved (the TS `Map` iteration order) for
/// [`Registry::list_models`].
#[derive(Default)]
pub struct Registry {
    map: HashMap<String, Box<dyn ModelAdapter>>,
    order: Vec<String>,
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// TS `registerModel`. Duplicate ids are an error (the TS `throw`).
    pub fn register_model(&mut self, reg: Box<dyn ModelAdapter>) -> Result<(), RegistryError> {
        let id = reg.id().to_string();
        if self.map.contains_key(&id) {
            eprintln!(
                "[des-registry] model \"{id}\" is already registered; duplicate registration usually means an adapter module was imported twice."
            );
            return Err(RegistryError::AlreadyRegistered(id));
        }
        self.order.push(id.clone());
        self.map.insert(id, reg);
        Ok(())
    }

    /// TS `getModel`.
    pub fn get_model(&self, id: &str) -> Result<&dyn ModelAdapter, RegistryError> {
        match self.map.get(id) {
            Some(reg) => Ok(reg.as_ref()),
            None => {
                let registered = self.order.clone();
                eprintln!(
                    "[des-registry] unknown model \"{id}\". Registered models: [{}]",
                    registered.join(", ")
                );
                Err(RegistryError::UnknownModel {
                    id: id.to_string(),
                    registered,
                })
            }
        }
    }

    /// TS `listModels`.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.order
            .iter()
            .filter_map(|id| self.map.get(id))
            .map(|r| ModelInfo {
                id: r.id().to_string(),
                description: r.description().to_string(),
            })
            .collect()
    }

    /// TS `runFromSpec`. Validate, run, summarise, and write any configured
    /// outputs. `P`/`R` are erased to [`JsonValue`].
    pub fn run_from_spec(
        &self,
        spec: &DESModelSpec<JsonValue>,
        opts: &RunFromSpecOptions,
    ) -> Result<DESRunSummary, RegistryError> {
        if spec.schema != DES_MODEL_SPEC_SCHEMA {
            eprintln!(
                "[runFromSpec] unexpected $schema \"{}\" (expected \"{DES_MODEL_SPEC_SCHEMA}\") — the spec file may be the wrong format or version.",
                spec.schema
            );
            return Err(RegistryError::UnknownSchema(spec.schema.clone()));
        }
        let reg = self.get_model(&spec.model)?;
        let runtime = spec.runtime.clone().unwrap_or_default();
        let mut out_cfg = runtime.outputs.clone().unwrap_or_default();
        let animate_enabled = runtime.animate != Some(false);
        if reg.has_animate() && animate_enabled {
            if out_cfg.html.is_none() {
                out_cfg.html = Some(format!("out/{}.html", spec.model));
            }
            if out_cfg.frames.is_none() {
                if let Some(html) = &out_cfg.html {
                    out_cfg.frames = Some(default_frames_path(html));
                }
            }
        }
        let runtime_for_run = DESRuntimeConfig {
            outputs: Some(out_cfg.clone()),
            ..runtime.clone()
        };
        let verbose = opts.verbose.or(runtime.verbose).unwrap_or(true);

        let params = validate_model_parameters(&spec.model, &spec.parameters, reg)?;

        if verbose {
            let desc = spec
                .description
                .clone()
                .unwrap_or_else(|| reg.description().to_string());
            eprintln!(
                "[runFromSpec] model=\"{}\"  description={}",
                spec.model,
                to_pretty_json(&JsonValue::String(desc), 0)
            );
        }

        let t0 = Instant::now();
        let result = reg.run(params.clone(), &runtime_for_run);
        let runtime_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let summary_text = reg.summarize(&result, &params);
        if verbose {
            eprintln!("[runFromSpec] completed in {runtime_ms} ms");
            eprintln!();
            eprintln!("{summary_text}");
        }

        let mut outputs: Vec<OutputEntry> = Vec::new();
        if let Some(csv) = &out_cfg.csv {
            if reg.has_write_csv() {
                mkdir_parent(csv);
                reg.write_csv(&result, csv);
                outputs.push(OutputEntry {
                    kind: OutputKind::Csv,
                    path: csv.clone(),
                });
                if verbose {
                    eprintln!("[runFromSpec] wrote CSV: {csv}");
                }
            }
        }
        if (out_cfg.html.is_some() || out_cfg.frames.is_some())
            && reg.has_animate()
            && animate_enabled
        {
            reg.animate(&result, &params, &runtime_for_run);
            if let Some(html) = &out_cfg.html {
                outputs.push(OutputEntry {
                    kind: OutputKind::Html,
                    path: html.clone(),
                });
            }
            if let Some(frames) = &out_cfg.frames {
                outputs.push(OutputEntry {
                    kind: OutputKind::Frames,
                    path: frames.clone(),
                });
            }
            if verbose {
                if let Some(html) = &out_cfg.html {
                    eprintln!("[runFromSpec] wrote HTML: {html}");
                }
            }
        }
        if let Some(log) = &out_cfg.log {
            if Path::new(log).exists() {
                outputs.push(OutputEntry {
                    kind: OutputKind::Log,
                    path: log.clone(),
                });
                if verbose {
                    eprintln!("[runFromSpec] wrote log: {log}");
                }
            }
        }
        if let Some(summary) = &out_cfg.summary {
            mkdir_parent(summary);
            let mut payload = JsonObject::new();
            payload.insert("modelId".to_string(), JsonValue::String(spec.model.clone()));
            payload.insert("params".to_string(), params.clone());
            payload.insert("runtimeMs".to_string(), JsonValue::Number(runtime_ms));
            payload.insert(
                "summaryText".to_string(),
                JsonValue::String(summary_text.clone()),
            );
            payload.insert("result".to_string(), serialise_result(&result));
            let text = to_pretty_json(&JsonValue::Object(payload), 0);
            std::fs::write(summary, text).map_err(|e| {
                RegistryError::Io(format!("failed to write summary {summary}: {e}"))
            })?;
            outputs.push(OutputEntry {
                kind: OutputKind::Summary,
                path: summary.clone(),
            });
            if verbose {
                eprintln!("[runFromSpec] wrote summary: {summary}");
            }
        }

        Ok(DESRunSummary {
            model_id: spec.model.clone(),
            params,
            runtime_ms,
            result,
            summary_text,
            outputs,
        })
    }

    /// TS `runFromJsonFile`. Load a JSON file, optionally lift a universal spec,
    /// and run it.
    pub fn run_from_json_file(
        &self,
        spec_path: &str,
        opts: &RunFromSpecOptions,
    ) -> Result<DESRunSummary, RegistryError> {
        let text = std::fs::read_to_string(spec_path)
            .map_err(|e| RegistryError::Io(format!("failed to read {spec_path}: {e}")))?;
        let parsed = parse_json(&text).map_err(|e| {
            eprintln!("[runFromJsonFile] failed to parse {spec_path} as JSON: {e}");
            RegistryError::Parse(e)
        })?;
        if is_universal_des_model_spec(&parsed) {
            // PORT NOTE: `universalToDESModelSpec` operates on the *typed*
            // `UniversalDESModelSpec`; deserialising a `JsonValue` into that
            // struct needs a codec that this no-serde port does not provide, so
            // the universal lift is unsupported from a raw JSON file here.
            return Err(RegistryError::Unsupported(
                "universal model specs require a typed deserializer not available in this port"
                    .to_string(),
            ));
        }
        let spec = json_to_model_spec(&parsed)?;
        self.run_from_spec(&spec, opts)
    }
}

/// `{id, description}` row from [`Registry::list_models`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub description: String,
}

/// TS `RunFromSpecOptions`.
#[derive(Clone, Debug, Default)]
pub struct RunFromSpecOptions {
    /// If `Some(true)`, log progress to stderr. Defaults to `runtime.verbose`
    /// then `true`.
    pub verbose: Option<bool>,
}

/// Render a [`ParamSchema`] as the JSON-ish schema object printed by CLI
/// helpers. This mirrors the TS discriminated-union shape closely enough for
/// humans to copy model specs without adding a serde dependency to the
/// registry path.
pub fn param_schema_to_json(schema: &ParamSchema) -> JsonValue {
    let mut out = JsonObject::new();
    match schema {
        ParamSchema::Number {
            min,
            max,
            integer,
            default,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("number".to_string()));
            insert_optional_number(&mut out, "min", *min);
            insert_optional_number(&mut out, "max", *max);
            insert_optional_bool(&mut out, "integer", *integer);
            insert_optional_number(&mut out, "default", *default);
            insert_optional_string(&mut out, "description", description);
        }
        ParamSchema::String {
            allowed,
            default,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("string".to_string()));
            if let Some(allowed) = allowed {
                out.insert(
                    "enum".to_string(),
                    JsonValue::Array(
                        allowed
                            .iter()
                            .cloned()
                            .map(JsonValue::String)
                            .collect::<Vec<_>>(),
                    ),
                );
            }
            insert_optional_string(&mut out, "default", default);
            insert_optional_string(&mut out, "description", description);
        }
        ParamSchema::Boolean {
            default,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("boolean".to_string()));
            insert_optional_bool(&mut out, "default", *default);
            insert_optional_string(&mut out, "description", description);
        }
        ParamSchema::Array {
            items,
            min_length,
            max_length,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("array".to_string()));
            out.insert("items".to_string(), param_schema_to_json(items));
            insert_optional_usize(&mut out, "minLength", *min_length);
            insert_optional_usize(&mut out, "maxLength", *max_length);
            insert_optional_string(&mut out, "description", description);
        }
        ParamSchema::Object {
            fields,
            required,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("object".to_string()));
            let mut field_obj = JsonObject::new();
            for (key, value) in fields {
                field_obj.insert(key.clone(), param_schema_to_json(value));
            }
            out.insert("fields".to_string(), JsonValue::Object(field_obj));
            if let Some(required) = required {
                out.insert(
                    "required".to_string(),
                    JsonValue::Array(
                        required
                            .iter()
                            .cloned()
                            .map(JsonValue::String)
                            .collect::<Vec<_>>(),
                    ),
                );
            }
            insert_optional_string(&mut out, "description", description);
        }
        ParamSchema::OneOf {
            variants,
            description,
        } => {
            out.insert("kind".to_string(), JsonValue::String("oneOf".to_string()));
            out.insert(
                "variants".to_string(),
                JsonValue::Array(
                    variants
                        .iter()
                        .map(|variant| {
                            let mut v = JsonObject::new();
                            v.insert("tag".to_string(), JsonValue::String(variant.tag.clone()));
                            insert_optional_string(&mut v, "tagField", &variant.tag_field);
                            v.insert("schema".to_string(), param_schema_to_json(&variant.schema));
                            insert_optional_string(&mut v, "description", &variant.description);
                            JsonValue::Object(v)
                        })
                        .collect(),
                ),
            );
            insert_optional_string(&mut out, "description", description);
        }
    }
    JsonValue::Object(out)
}

// =============================================================================
// Helpers.
// =============================================================================

fn insert_optional_number(obj: &mut JsonObject, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), JsonValue::Number(value));
    }
}

fn insert_optional_usize(obj: &mut JsonObject, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), JsonValue::Number(value as f64));
    }
}

fn insert_optional_bool(obj: &mut JsonObject, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), JsonValue::Bool(value));
    }
}

fn insert_optional_string(obj: &mut JsonObject, key: &str, value: &Option<String>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), JsonValue::String(value.clone()));
    }
}

fn mkdir_parent(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
}

/// TS `validateModelParameters` (zod path dropped).
fn validate_model_parameters(
    model_id: &str,
    value: &JsonValue,
    reg: &dyn ModelAdapter,
) -> Result<JsonValue, RegistryError> {
    let v = validate(value, &reg.schema());
    if !v.valid {
        eprintln!(
            "[des-registry] parameter validation failed for model \"{model_id}\" ({} error(s)): {}",
            v.errors.len(),
            v.errors.join("; ")
        );
        return Err(RegistryError::InvalidParameters {
            model_id: model_id.to_string(),
            errors: v.errors,
        });
    }
    Ok(v.value.unwrap_or(JsonValue::Undefined))
}

/// TS `serialiseResult` — strip giant numeric arrays for JSON dumps. (There are
/// no functions to skip in a [`JsonValue`].)
fn serialise_result(r: &JsonValue) -> JsonValue {
    match r {
        JsonValue::Array(items) => JsonValue::Array(items.iter().map(serialise_result).collect()),
        JsonValue::Object(obj) => {
            let mut out = JsonObject::new();
            for k in obj.keys().cloned().collect::<Vec<_>>() {
                let v = obj.get(&k).expect("key present");
                if let JsonValue::Array(items) = v {
                    if items.len() > 1000 && matches!(items.first(), Some(JsonValue::Number(_))) {
                        out.insert(
                            k,
                            JsonValue::String(format!(
                                "<array length={} (omitted from summary)>",
                                items.len()
                            )),
                        );
                        continue;
                    }
                }
                out.insert(k, serialise_result(v));
            }
            JsonValue::Object(out)
        }
        other => other.clone(),
    }
}

/// Build a `DESModelSpec<JsonValue>` from a parsed JSON object.
fn json_to_model_spec(value: &JsonValue) -> Result<DESModelSpec<JsonValue>, RegistryError> {
    let obj = match value {
        JsonValue::Object(o) => o,
        _ => {
            return Err(RegistryError::Parse(
                "spec must be a JSON object".to_string(),
            ))
        }
    };
    let schema = string_field(obj, "$schema").unwrap_or_default();
    let model = string_field(obj, "model").unwrap_or_default();
    let description = string_field(obj, "description");
    let parameters = obj
        .get("parameters")
        .cloned()
        .unwrap_or(JsonValue::Undefined);
    let runtime = obj.get("runtime").map(runtime_from_json);
    let metadata = obj.get("metadata").map(metadata_from_json);
    Ok(DESModelSpec {
        schema,
        model,
        description,
        parameters,
        runtime,
        metadata,
    })
}

fn runtime_from_json(value: &JsonValue) -> DESRuntimeConfig {
    let obj = match value {
        JsonValue::Object(o) => o,
        _ => return DESRuntimeConfig::default(),
    };
    DESRuntimeConfig {
        seed: number_field(obj, "seed"),
        animate: bool_field(obj, "animate"),
        verbose: bool_field(obj, "verbose"),
        outputs: obj.get("outputs").map(outputs_from_json),
    }
}

fn outputs_from_json(value: &JsonValue) -> DESOutputs {
    let obj = match value {
        JsonValue::Object(o) => o,
        _ => return DESOutputs::default(),
    };
    DESOutputs {
        csv: string_field(obj, "csv"),
        html: string_field(obj, "html"),
        frames: string_field(obj, "frames"),
        summary: string_field(obj, "summary"),
        log: string_field(obj, "log"),
    }
}

fn metadata_from_json(value: &JsonValue) -> DESModelMetadata {
    let obj = match value {
        JsonValue::Object(o) => o,
        _ => return DESModelMetadata::default(),
    };
    let tags = match obj.get("tags") {
        Some(JsonValue::Array(items)) => Some(
            items
                .iter()
                .filter_map(|v| {
                    if let JsonValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        ),
        _ => None,
    };
    DESModelMetadata {
        author: string_field(obj, "author"),
        created_at: string_field(obj, "createdAt"),
        tags,
        notes: string_field(obj, "notes"),
    }
}

fn string_field(obj: &JsonObject, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(JsonValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}
fn number_field(obj: &JsonObject, key: &str) -> Option<f64> {
    match obj.get(key) {
        Some(JsonValue::Number(n)) => Some(*n),
        _ => None,
    }
}
fn bool_field(obj: &JsonObject, key: &str) -> Option<bool> {
    match obj.get(key) {
        Some(JsonValue::Bool(b)) => Some(*b),
        _ => None,
    }
}

// =============================================================================
// Self-contained JSON parse + pretty-print (no serde).
// =============================================================================

/// Minimal recursive-descent JSON parser into [`JsonValue`] (`JSON.parse`).
pub fn parse_json(text: &str) -> Result<JsonValue, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = JsonParser {
        chars: &chars,
        pos: 0,
    };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format!(
            "unexpected trailing characters at position {}",
            p.pos
        ));
    }
    Ok(value)
}

struct JsonParser<'a> {
    chars: &'a [char],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!(
                "unexpected character '{c}' at position {}",
                self.pos
            )),
            None => Err("unexpected end of input".to_string()),
        }
    }
    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect('{')?;
        let mut obj = JsonObject::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(JsonValue::Object(obj));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            obj.insert(key, value);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some('}') => break,
                other => return Err(format!("expected ',' or '}}' in object, got {other:?}")),
            }
        }
        Ok(JsonValue::Object(obj))
    }
    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some(']') => break,
                other => return Err(format!("expected ',' or ']' in array, got {other:?}")),
            }
        }
        Ok(JsonValue::Array(items))
    }
    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.next() {
                Some('"') => break,
                Some('\\') => match self.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000C}'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let c = self.next().ok_or("unterminated \\u escape")?;
                            let digit = c.to_digit(16).ok_or("invalid \\u hex digit")?;
                            code = code * 16 + digit;
                        }
                        out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    other => return Err(format!("invalid escape \\{other:?}")),
                },
                Some(c) => out.push(c),
                None => return Err("unterminated string".to_string()),
            }
        }
        Ok(out)
    }
    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.starts_with("true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.starts_with("false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(format!("invalid literal at position {}", self.pos))
        }
    }
    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.starts_with("null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(format!("invalid literal at position {}", self.pos))
        }
    }
    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice: String = self.chars[start..self.pos].iter().collect();
        slice
            .parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| format!("invalid number \"{slice}\""))
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.next() {
            Some(got) if got == c => Ok(()),
            other => Err(format!(
                "expected '{c}', got {other:?} at position {}",
                self.pos
            )),
        }
    }
    fn starts_with(&self, lit: &str) -> bool {
        lit.chars()
            .enumerate()
            .all(|(i, c)| self.chars.get(self.pos + i).copied() == Some(c))
    }
}

/// `JSON.stringify(value, null, 2)` — pretty-print with 2-space indentation.
pub fn to_pretty_json(value: &JsonValue, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let pad_inner = "  ".repeat(indent + 1);
    match value {
        JsonValue::Undefined | JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if n.is_finite() {
                n.to_string()
            } else {
                "null".to_string()
            }
        }
        JsonValue::String(s) => quote_json_string(s),
        JsonValue::Array(items) => {
            if items.is_empty() {
                return "[]".to_string();
            }
            let inner: Vec<String> = items
                .iter()
                .map(|v| format!("{pad_inner}{}", to_pretty_json(v, indent + 1)))
                .collect();
            format!("[\n{}\n{pad}]", inner.join(",\n"))
        }
        JsonValue::Object(obj) => {
            let keys: Vec<&String> = obj.keys().collect();
            if keys.is_empty() {
                return "{}".to_string();
            }
            let inner: Vec<String> = keys
                .iter()
                .map(|k| {
                    let v = obj.get(k).expect("key present");
                    format!(
                        "{pad_inner}{}: {}",
                        quote_json_string(k),
                        to_pretty_json(v, indent + 1)
                    )
                })
                .collect();
            format!("{{\n{}\n{pad}}}", inner.join(",\n"))
        }
    }
}

fn quote_json_string(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct DoubleAdapter;
    impl ModelAdapter for DoubleAdapter {
        fn id(&self) -> &str {
            "double"
        }
        fn description(&self) -> &str {
            "doubles its `x` parameter"
        }
        fn schema(&self) -> ParamSchema {
            ParamSchema::Object {
                fields: vec![(
                    "x".to_string(),
                    ParamSchema::Number {
                        min: None,
                        max: None,
                        integer: None,
                        default: Some(1.0),
                        description: None,
                    },
                )],
                required: Some(vec![]),
                description: None,
            }
        }
        fn run(&self, params: JsonValue, _runtime: &DESRuntimeConfig) -> JsonValue {
            let x = match &params {
                JsonValue::Object(o) => match o.get("x") {
                    Some(JsonValue::Number(n)) => *n,
                    _ => 0.0,
                },
                _ => 0.0,
            };
            let mut out = JsonObject::new();
            out.insert("y".to_string(), JsonValue::Number(x * 2.0));
            JsonValue::Object(out)
        }
        fn summarize(&self, result: &JsonValue, _params: &JsonValue) -> String {
            format!("y = {}", to_pretty_json(result, 0))
        }
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.register_model(Box::new(DoubleAdapter)).unwrap();
        r
    }

    #[test]
    fn register_and_list() {
        let r = registry();
        let models = r.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "double");
    }

    #[test]
    fn duplicate_registration_errors() {
        let mut r = registry();
        let err = r.register_model(Box::new(DoubleAdapter)).unwrap_err();
        assert_eq!(err, RegistryError::AlreadyRegistered("double".to_string()));
    }

    #[test]
    fn unknown_model_errors() {
        let r = registry();
        assert!(matches!(
            r.get_model("nope"),
            Err(RegistryError::UnknownModel { .. })
        ));
    }

    #[test]
    fn run_from_spec_validates_and_runs() {
        let r = registry();
        let mut params = JsonObject::new();
        params.insert("x".to_string(), JsonValue::Number(21.0));
        let spec = DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: "double".to_string(),
            description: None,
            parameters: JsonValue::Object(params),
            runtime: Some(DESRuntimeConfig {
                verbose: Some(false),
                ..Default::default()
            }),
            metadata: None,
        };
        let summary = r
            .run_from_spec(
                &spec,
                &RunFromSpecOptions {
                    verbose: Some(false),
                },
            )
            .unwrap();
        assert_eq!(summary.model_id, "double");
        match summary.result {
            JsonValue::Object(o) => assert_eq!(o.get("y"), Some(&JsonValue::Number(42.0))),
            _ => panic!("expected object result"),
        }
    }

    #[test]
    fn run_from_spec_rejects_bad_schema() {
        let r = registry();
        let spec = DESModelSpec {
            schema: "wrong".to_string(),
            model: "double".to_string(),
            description: None,
            parameters: JsonValue::Undefined,
            runtime: None,
            metadata: None,
        };
        assert!(matches!(
            r.run_from_spec(&spec, &RunFromSpecOptions::default()),
            Err(RegistryError::UnknownSchema(_))
        ));
    }

    #[test]
    fn parse_json_round_trips_basic_shapes() {
        let v = parse_json(r#"{"a": 1, "b": [true, null, "x"], "c": {"d": -2.5e1}}"#).unwrap();
        match &v {
            JsonValue::Object(o) => {
                assert_eq!(o.get("a"), Some(&JsonValue::Number(1.0)));
                assert!(matches!(o.get("b"), Some(JsonValue::Array(_))));
                match o.get("c") {
                    Some(JsonValue::Object(c)) => {
                        assert_eq!(c.get("d"), Some(&JsonValue::Number(-25.0)))
                    }
                    _ => panic!("c"),
                }
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn pretty_json_indents() {
        let mut o = JsonObject::new();
        o.insert("k".to_string(), JsonValue::Number(1.0));
        let text = to_pretty_json(&JsonValue::Object(o), 0);
        assert_eq!(text, "{\n  \"k\": 1\n}");
    }

    #[test]
    fn param_schema_to_json_renders_cli_shape() {
        let schema = ParamSchema::Object {
            fields: vec![
                (
                    "x".to_string(),
                    ParamSchema::Number {
                        min: Some(0.0),
                        max: None,
                        integer: Some(false),
                        default: Some(1.5),
                        description: None,
                    },
                ),
                (
                    "mode".to_string(),
                    ParamSchema::String {
                        allowed: Some(vec!["a".to_string(), "b".to_string()]),
                        default: Some("a".to_string()),
                        description: None,
                    },
                ),
            ],
            required: Some(vec!["x".to_string()]),
            description: Some("example".to_string()),
        };
        let rendered = param_schema_to_json(&schema);
        let obj = match rendered {
            JsonValue::Object(obj) => obj,
            _ => panic!("expected object schema"),
        };
        assert_eq!(
            obj.get("kind"),
            Some(&JsonValue::String("object".to_string()))
        );
        assert!(matches!(obj.get("fields"), Some(JsonValue::Object(_))));
        assert_eq!(
            obj.get("required"),
            Some(&JsonValue::Array(vec![JsonValue::String("x".to_string())]))
        );
    }

    #[test]
    fn serialise_result_omits_giant_numeric_arrays() {
        let mut o = JsonObject::new();
        o.insert(
            "big".to_string(),
            JsonValue::Array((0..1001).map(|i| JsonValue::Number(i as f64)).collect()),
        );
        let out = serialise_result(&JsonValue::Object(o));
        match out {
            JsonValue::Object(o) => match o.get("big") {
                Some(JsonValue::String(s)) => assert!(s.contains("omitted from summary")),
                _ => panic!("expected omission marker"),
            },
            _ => panic!("object"),
        }
    }
}
