//! External-program plugin system: run third-party programs that emit JSON /
//! JSONL and render the result as an interactive HTML player.
//!
//! # The contract
//!
//! A plugin is **any program** that writes to stdout (Rust for now). It is
//! described, JSON-first, by a [`PluginManifest`]:
//!
//! * [`RunSpec`] — how to launch it (command, args, cwd, env).
//! * [`OutputKind`] — `Json` (one document = a result) or `Jsonl` (one document
//!   per line = a frame stream).
//! * [`PlayerKind`] — `Sim` (frame player) or `Results` (results viewer).
//! * `controls: Vec<`[`UiControl`]`>` — switches/toggles/sliders the player
//!   exposes (see [`player`] for their generic semantics).
//!
//! # The flow
//!
//! ```text
//!   PluginManifest ──run_plugin──▶ PluginRun ──render_player_html──▶ HTML
//!        (JSON)         (spawn)      (JSON/JSONL)    (self-contained, vanilla JS)
//! ```
//!
//! [`run_and_render`] does both steps. The host stays thin: it only needs the
//! manifest; running, parsing, and rendering all live here.
//!
//! # Discovery
//!
//! A [`PluginRegistry`] holds installed manifests and, via
//! [`PluginRegistry::as_extension`], plugs into the [`crate::des::service`]
//! [`DesExtension`](crate::des::service::DesExtension) seam — so installed
//! plugins show up in a server's `/api/docs.json` descriptor automatically.
//!
//! # Frame schema
//!
//! The sim player understands the same `shapes` schema as
//! [`crate::des::animation::types`] (`circle`/`rect`/`line`/`text`/`path`), so a
//! plugin can emit animation frames and get SVG rendering for free; any other
//! JSON object is shown as a field inspector with its numeric fields charted on
//! a timeline.
//!
//! # Authoring a Rust plugin
//!
//! A plugin is a normal Rust binary. For a frame stream:
//!
//! ```ignore
//! // src/main.rs of the plugin crate
//! fn main() {
//!     let mut n = 0i64;
//!     for t in 0..200 {
//!         n += if t % 3 == 0 { 1 } else { -1 };
//!         n = n.max(0);
//!         // one compact JSON object per line = one frame
//!         println!("{{\"t\":{t},\"n\":{n}}}");
//!     }
//! }
//! ```
//!
//! Register it and render:
//!
//! ```ignore
//! use des_engine::des::plugin::*;
//! let manifest = PluginManifest {
//!     id: "mm1".into(), name: "M/M/1".into(), version: "1.0.0".into(),
//!     description: "queue length".into(),
//!     run: RunSpec::new("./target/release/mm1-plugin", &[]),
//!     output: OutputKind::Jsonl, player: PlayerKind::Sim,
//!     controls: vec![UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0)],
//!     title: None,
//! };
//! let html = run_and_render(&manifest)?;
//! std::fs::write("out/mm1.html", html)?;
//! # Ok::<(), PluginError>(())
//! ```

pub mod manifest;
pub mod player;
pub mod registry;
pub mod runner;

pub use manifest::{
    ControlKind, OutputKind, PlayerKind, PluginManifest, PluginRuntimeKind, PluginTransportKind,
    RunSpec, UiControl,
};
pub use player::render_player_html;
pub use registry::{DuplicatePlugin, PluginCatalogExtension, PluginRegistry};
pub use runner::{parse_output, run_plugin, PluginError, PluginOutput, PluginRun};

/// Run a plugin program and render its player HTML in one call.
pub fn run_and_render(manifest: &PluginManifest) -> Result<String, PluginError> {
    let run = run_plugin(manifest)?;
    Ok(render_player_html(manifest, &run))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_spawns_program_and_renders_sim_player() {
        // A stand-in for an external Rust binary: emits 3 JSONL frames.
        let manifest = PluginManifest {
            id: "e2e".to_string(),
            name: "End to end".to_string(),
            version: "1.0.0".to_string(),
            description: "spawn + parse + render".to_string(),
            runtime: PluginRuntimeKind::Rust,
            transport: PluginTransportKind::Stdio,
            language: None,
            run: RunSpec::new(
                "sh",
                &[
                    "-c",
                    "printf '{\"t\":0,\"n\":1}\\n{\"t\":1,\"n\":2}\\n{\"t\":2,\"n\":1}\\n'",
                ],
            ),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls: vec![UiControl::range("speed", "Speed", 1.0, 30.0, 1.0, 8.0)],
            title: None,
        };
        let html = run_and_render(&manifest).expect("run + render");
        assert!(html.contains("id=\"plugin-payload\""));
        assert!(html.contains("\"player\":\"sim\""));
        assert!(html.contains("\"n\":2"));
    }
}
