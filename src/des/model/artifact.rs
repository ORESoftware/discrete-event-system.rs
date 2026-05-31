//! The uniform output of running *any* first-class model — the thing the UI
//! renders and the API returns, regardless of paradigm (MDP, POMDP, hybrid
//! block diagram, DES network, optimization, …).
//!
//! A [`RunArtifact`] carries both a **stream** (`frames`: JSONL documents for an
//! animated sim player) and a **document** (`results`: a single JSON object,
//! e.g. an optimal policy / value function / solution). A model picks the
//! primary [`PlayerKind`]; the artifact always carries both so the platform can
//! show an animated rollout *and* expose the solved artifact programmatically.
//!
//! [`RunArtifact::to_player_html`] renders the artifact through the existing
//! self-contained plugin player ([`crate::des::plugin`]) with no process spawn,
//! so every citizen visualizes the same way the hybrid demos and external
//! plugins do.

use serde_json::Value;

use crate::des::plugin::{
    render_player_html, OutputKind, PlayerKind, PluginManifest, PluginOutput, PluginRun, RunSpec,
    UiControl,
};

/// The uniform result of running a first-class model.
#[derive(Clone, Debug)]
pub struct RunArtifact {
    /// Model kind that produced this (e.g. `"mdp"`, `"pomdp"`, `"hybrid"`).
    pub kind: String,
    pub title: String,
    pub description: String,
    /// Which player the UI should foreground.
    pub player: PlayerKind,
    /// JSONL sim frames (one object per time/iteration step). May be empty for a
    /// pure results model.
    pub frames: Vec<Value>,
    /// A single results document (policy/value/solution/summary). May be `Null`
    /// for a pure-streaming model.
    pub results: Value,
    /// Interactive controls the player should expose.
    pub controls: Vec<UiControl>,
    /// Short human-readable summary.
    pub summary: String,
}

impl RunArtifact {
    /// An animated (frame-by-frame) artifact. `results` carries the solved
    /// document alongside the stream.
    pub fn sim(
        kind: &str,
        title: &str,
        description: &str,
        frames: Vec<Value>,
        results: Value,
        controls: Vec<UiControl>,
        summary: &str,
    ) -> Self {
        RunArtifact {
            kind: kind.to_string(),
            title: title.to_string(),
            description: description.to_string(),
            player: PlayerKind::Sim,
            frames,
            results,
            controls,
            summary: summary.to_string(),
        }
    }

    /// A results-only artifact (no animation).
    pub fn results(
        kind: &str,
        title: &str,
        description: &str,
        results: Value,
        controls: Vec<UiControl>,
        summary: &str,
    ) -> Self {
        RunArtifact {
            kind: kind.to_string(),
            title: title.to_string(),
            description: description.to_string(),
            player: PlayerKind::Results,
            frames: Vec::new(),
            results,
            controls,
            summary: summary.to_string(),
        }
    }

    /// Render this artifact to a self-contained HTML page via the plugin player.
    pub fn to_player_html(&self) -> String {
        let manifest = PluginManifest {
            id: format!("model-{}", self.kind),
            name: self.title.clone(),
            version: "1.0.0".to_string(),
            description: self.description.clone(),
            // Placeholder: rendered from in-memory output, never spawned.
            run: RunSpec::new("model-internal", &[]),
            output: match self.player {
                PlayerKind::Sim => OutputKind::Jsonl,
                PlayerKind::Results => OutputKind::Json,
            },
            player: self.player,
            controls: self.controls.clone(),
            title: Some(self.title.clone()),
        };
        let output = match self.player {
            PlayerKind::Sim => PluginOutput::Jsonl(self.frames.clone()),
            PlayerKind::Results => PluginOutput::Json(self.results.clone()),
        };
        let run = PluginRun {
            plugin_id: manifest.id.clone(),
            output,
            exit_code: Some(0),
            stderr: String::new(),
        };
        render_player_html(&manifest, &run)
    }

    /// The frame stream as JSONL text (one compact JSON object per line).
    pub fn to_jsonl(&self) -> String {
        self.frames
            .iter()
            .map(|f| serde_json::to_string(f).unwrap_or_else(|_| "{}".to_string()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
