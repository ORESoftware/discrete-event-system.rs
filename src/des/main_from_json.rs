//! Port of `src/des/main-from-json.ts`.
//!
//! CLI: run any registered DES model from a JSON spec file (with
//! `--list` / `--schema` / `--example` subcommands).
//!
//! `process.argv` → `std::env::args`; `process.exit` → `std::process::exit`;
//! `fs` → `std::fs`.
//!
//! PORT NOTE: the TS imports `./general/des-registry`
//! (`runFromJsonFile`, `listModels`, `getModel`). Rust uses the production
//! `crate::des::general::des_registry::Registry`, but built-in typed adapters
//! are not auto-registered yet because the concrete adapters still need a
//! `JsonValue <-> P` codec bridge. The CLI therefore exercises the real
//! registry/parsing/validation path and reports an empty model set until those
//! wrappers are added.

#![allow(dead_code)]

use crate::des::general::des_registry::{
    param_schema_to_json, to_pretty_json, Registry, RunFromSpecOptions,
};

fn print_help() {
    println!("Usage:");
    println!("  main-from-json <path-to-spec.json>");
    println!("  main-from-json --list");
    println!("  main-from-json --schema   <model-id>");
    println!("  main-from-json --example  <model-id>");
    println!();
    println!("A spec file is a JSON object with at least:");
    println!("  {{ \"$schema\": \"des/model-spec/v1\", \"model\": \"<id>\", \"parameters\": {{ ... }} }}");
    println!("or a universal modeling document:");
    println!("  {{ \"$schema\": \"des/universal-model/v1\", \"originalInput\": ..., \"math\": ..., \"des\": ... }}");
}

fn build_registry() -> Registry {
    Registry::new()
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        print_help();
        // TS called `process.exit`; as a library entry point we return instead
        // so callers (e.g. the serial simulation driver) keep running.
        return;
    }

    let registry = build_registry();

    if argv[0] == "--list" {
        let models = registry.list_models();
        println!("Registered models ({}):", models.len());
        for m in &models {
            println!("  {:<24} - {}", m.id, m.description);
        }
        return;
    }
    if argv[0] == "--schema" && argv.len() > 1 {
        match registry.get_model(&argv[1]) {
            Ok(reg) => {
                println!("Model: {}", reg.id());
                println!("Description: {}", reg.description());
                println!("Schema:");
                println!(
                    "{}",
                    to_pretty_json(&param_schema_to_json(&reg.schema()), 0)
                );
            }
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        }
        return;
    }
    if argv[0] == "--example" && argv.len() > 1 {
        match registry.get_model(&argv[1]) {
            Ok(_) => {
                eprintln!(
                    "Examples for \"{}\" require typed adapter wrappers; none are registered yet.",
                    argv[1]
                );
                return;
            }
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        }
    }

    let spec_path = &argv[0];
    if !std::path::Path::new(spec_path).exists() {
        eprintln!("Spec file not found: {spec_path}");
        return;
    }
    match registry.run_from_json_file(
        spec_path,
        &RunFromSpecOptions {
            verbose: Some(true),
        },
    ) {
        Ok(summary) => {
            if !summary.outputs.is_empty() {
                println!();
                println!("Outputs written:");
                for o in &summary.outputs {
                    println!("  [{}] {}", output_kind_label(o.kind), o.path);
                }
            }
            println!();
            println!("Total wall-clock time: {} ms", summary.runtime_ms);
        }
        Err(e) => {
            eprintln!("Error: {e}");
        }
    }
}

fn output_kind_label(kind: crate::des::general::des_spec::OutputKind) -> &'static str {
    match kind {
        crate::des::general::des_spec::OutputKind::Csv => "csv",
        crate::des::general::des_spec::OutputKind::Html => "html",
        crate::des::general::des_spec::OutputKind::Frames => "frames",
        crate::des::general::des_spec::OutputKind::Summary => "summary",
        crate::des::general::des_spec::OutputKind::Log => "log",
    }
}
