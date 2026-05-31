//! The JSON-first description of an external plugin program.
//!
//! A plugin is an **external program** (Rust for now, but the contract is just
//! "a process that writes JSON or JSONL to stdout"). A [`PluginManifest`] tells
//! the core how to *run* it, what *shape* its output has, which *player* to
//! render, and what *UI controls* (toggles/selects/sliders) to expose. The
//! manifest is `serde` (de)serializable so a plugin can ship a `plugin.json`
//! and the host can load it without recompiling.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Versioned identity of the plugin stdout protocol (the JSON/JSONL output
/// contract a plugin and the host agree on). Bump the version when the frame
/// envelope or output semantics change incompatibly; a host can compare this to
/// negotiate or reject a plugin before running it.
pub const PLUGIN_PROTOCOL_SCHEMA: &str = "des/plugin/v1";

/// Shape of the program's stdout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputKind {
    /// One JSON document on stdout = a single result object/array.
    Json,
    /// One JSON document per line = a stream of frames (JSON Lines).
    Jsonl,
}

/// Which player the host renders for this plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlayerKind {
    /// Frame-by-frame player: transport (play/pause/step/scrub/speed), a frame
    /// view (SVG when frames carry `shapes`, else a field inspector) and a
    /// timeline of numeric frame fields.
    Sim,
    /// Results viewer: metric cards, tables for arrays-of-objects, raw JSON,
    /// with toggles to switch what is shown.
    Results,
}

/// Kind of an interactive UI control declared by the plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ControlKind {
    /// On/off checkbox.
    Toggle,
    /// Pick one of `options`.
    Select,
    /// Numeric slider (`min`/`max`/`step`).
    Range,
}

/// One UI switch/toggle the player renders. `target` names the data key the
/// control acts on (a frame series / field, or a results section); the player
/// applies a generic interpretation (show/hide, feature, or — for a control
/// whose `id` is `"speed"` — set playback rate).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiControl {
    pub id: String,
    pub label: String,
    pub kind: ControlKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Choices for a [`ControlKind::Select`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    /// Data key this control acts on (e.g. a numeric series id or a section).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

impl UiControl {
    pub fn toggle(id: &str, label: &str, default_on: bool, target: Option<&str>) -> Self {
        UiControl {
            id: id.to_string(),
            label: label.to_string(),
            kind: ControlKind::Toggle,
            default: Some(Value::Bool(default_on)),
            options: Vec::new(),
            min: None,
            max: None,
            step: None,
            target: target.map(str::to_string),
        }
    }

    pub fn select(id: &str, label: &str, options: &[&str], default: &str, target: Option<&str>) -> Self {
        UiControl {
            id: id.to_string(),
            label: label.to_string(),
            kind: ControlKind::Select,
            default: Some(Value::String(default.to_string())),
            options: options.iter().map(|s| s.to_string()).collect(),
            min: None,
            max: None,
            step: None,
            target: target.map(str::to_string),
        }
    }

    pub fn range(id: &str, label: &str, min: f64, max: f64, step: f64, default: f64) -> Self {
        UiControl {
            id: id.to_string(),
            label: label.to_string(),
            kind: ControlKind::Range,
            default: Some(Value::from(default)),
            options: Vec::new(),
            min: Some(min),
            max: Some(max),
            step: Some(step),
            target: None,
        }
    }
}

/// How to launch the external program.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSpec {
    /// Executable to run (an absolute path, a name on `PATH`, or e.g. `cargo`).
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for the child process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Extra environment variables, as `[name, value]` pairs.
    #[serde(default)]
    pub env: Vec<(String, String)>,
    /// Advisory wall-clock budget (not yet enforced by the synchronous runner).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl RunSpec {
    pub fn new(command: &str, args: &[&str]) -> Self {
        RunSpec {
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: None,
            env: Vec::new(),
            timeout_ms: None,
        }
    }
}

/// Everything the host needs to run a plugin and render its player.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginManifest {
    /// Stable unique id (used for routing, dedupe, attribution).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub run: RunSpec,
    pub output: OutputKind,
    pub player: PlayerKind,
    #[serde(default)]
    pub controls: Vec<UiControl>,
    /// Optional page title override (defaults to `name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl PluginManifest {
    /// Canonical pretty JSON for this manifest (what a `plugin.json` looks like).
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse a manifest from JSON (e.g. a plugin's shipped `plugin.json`).
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }

    /// The plugin output protocol this host speaks ([`PLUGIN_PROTOCOL_SCHEMA`]).
    pub fn protocol_schema(&self) -> &'static str {
        PLUGIN_PROTOCOL_SCHEMA
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips_through_json() {
        let m = PluginManifest {
            id: "demo".to_string(),
            name: "Demo".to_string(),
            version: "1.0.0".to_string(),
            description: "a demo plugin".to_string(),
            run: RunSpec::new("./demo", &["--frames", "100"]),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls: vec![
                UiControl::toggle("show_q", "Show queue", true, Some("q")),
                UiControl::select("metric", "Metric", &["q", "n"], "q", Some("metric")),
                UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 6.0),
            ],
            title: None,
        };
        let json = m.to_json_string();
        let back = PluginManifest::from_json_str(&json).unwrap();
        assert_eq!(back.id, "demo");
        assert_eq!(back.output, OutputKind::Jsonl);
        assert_eq!(back.player, PlayerKind::Sim);
        assert_eq!(back.controls.len(), 3);
        assert_eq!(back.run.args, vec!["--frames", "100"]);
    }

    #[test]
    fn enums_serialize_kebab_case() {
        let v = serde_json::to_value(OutputKind::Jsonl).unwrap();
        assert_eq!(v, serde_json::json!("jsonl"));
        let v = serde_json::to_value(PlayerKind::Results).unwrap();
        assert_eq!(v, serde_json::json!("results"));
    }
}
