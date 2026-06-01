//! des_engine — a library/SDK for modeling, simulating, solving, and rendering
//! discrete, continuous, and mixed systems (a Rust port of the TypeScript
//! discrete-event-system engine).
//!
//! # Using it as a library
//!
//! The whole engine lives under [`des`], but most consumers — a web server, a
//! desktop app, a CLI — only need the **first-class seams**, which are gathered
//! in [`prelude`]:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use des_engine::prelude::*;
//! # use des_engine::des::plugin::{
//! #     OutputKind, PlayerKind, PluginRuntimeKind, PluginTransportKind, UiControl,
//! # };
//! # let spec_json = with_builtins()
//! #     .get("mdp")
//! #     .expect("built-in MDP citizen")
//! #     .descriptor()
//! #     .example_spec;
//! # let manifest = PluginManifest {
//! #     id: "mm1".into(),
//! #     name: "M/M/1".into(),
//! #     version: "1.0.0".into(),
//! #     description: "queue length".into(),
//! #     runtime: PluginRuntimeKind::Rust,
//! #     transport: PluginTransportKind::Stdio,
//! #     language: None,
//! #     run: RunSpec::new("./target/release/mm1-plugin", &[]),
//! #     output: OutputKind::Jsonl,
//! #     player: PlayerKind::Sim,
//! #     controls: vec![UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0)],
//! #     title: None,
//! # };
//!
//! // 1. Run any first-class model from a JSON spec → a uniform RunArtifact.
//! let registry = with_builtins();
//! let artifact = registry.run("mdp", &spec_json)?;
//! let html = artifact.to_player_html();
//!
//! // 2. Stream an iterative solver (LP/MILP/MDP/POMDP) over JSONL.
//! run_named_jsonl("lp", std::io::stdin().lock(), &mut std::io::stdout())?;
//!
//! // 3. Run an external plugin program and render its player.
//! let html = run_and_render(&manifest)?;
//! # Ok(())
//! # }
//! ```
//!
//! The seams are intentionally JSON-first so the same contracts work across an
//! HTTP boundary ([`des::service`] advertises them for discovery), an IPC
//! boundary ([`des::plugin::PluginTransport`]), or in-process.
//!
//! # The full tree
//!
//! The TypeScript source under `src/des/` maps 1:1 onto the [`des`] module.
//! Foundation modules in `des::shared` are dependency-free and define the
//! engine-wide conventions (the `Transform` trait, `Result`/`Option` helpers,
//! capability ports, and linear algebra). The `main_*` modules are runnable
//! simulation demos rather than part of the SDK surface.

#![allow(
    clippy::approx_constant,
    clippy::assertions_on_constants,
    clippy::clone_on_copy,
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::collapsible_str_replace,
    clippy::derivable_impls,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::erasing_op,
    clippy::excessive_precision,
    clippy::explicit_auto_deref,
    clippy::format_in_format_args,
    clippy::identity_op,
    clippy::if_same_then_else,
    clippy::implicit_saturating_sub,
    clippy::int_plus_one,
    clippy::iter_cloned_collect,
    clippy::large_enum_variant,
    clippy::len_zero,
    clippy::let_and_return,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::manual_flatten,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    clippy::manual_range_contains,
    clippy::manual_repeat_n,
    clippy::map_clone,
    clippy::map_entry,
    clippy::map_identity,
    clippy::module_inception,
    clippy::needless_borrow,
    clippy::needless_option_as_deref,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::overly_complex_bool_expr,
    clippy::print_with_newline,
    clippy::ptr_arg,
    clippy::redundant_closure,
    clippy::redundant_guards,
    clippy::should_implement_trait,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_map_or,
    clippy::unnecessary_sort_by,
    clippy::useless_vec,
    clippy::vec_init_then_push,
    clippy::while_let_loop
)]

pub mod des;
pub mod prelude;

pub use des::{
    authoring, decision, equation, fel, hybrid, model, plugin, service, streaming, studio,
};

/// Stable SDK-facing exports for embedders (servers, desktop apps, CLIs).
pub mod sdk {
    pub use crate::{
        authoring, decision, equation, fel, hybrid, model, plugin, service, streaming, studio,
    };

    /// Modules intended to be treated as the public SDK surface.
    pub const SDK_MODULES: &[&str] = &[
        "service",
        "model",
        "authoring",
        "streaming",
        "equation",
        "studio",
        "hybrid",
        "fel",
        "plugin",
        "decision",
    ];

    /// Lightweight descriptor embedders can expose in their own diagnostics.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SdkSurface {
        pub crate_name: &'static str,
        pub version: &'static str,
        pub modules: &'static [&'static str],
    }

    pub fn surface() -> SdkSurface {
        SdkSurface {
            crate_name: env!("CARGO_PKG_NAME"),
            version: env!("CARGO_PKG_VERSION"),
            modules: SDK_MODULES,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sdk_surface_lists_embedding_modules() {
        let surface = crate::sdk::surface();
        assert!(surface.modules.contains(&"service"));
        assert!(surface.modules.contains(&"authoring"));
        assert!(surface.modules.contains(&"streaming"));
        assert!(surface.modules.contains(&"equation"));
        assert!(surface.modules.contains(&"plugin"));
    }
}
