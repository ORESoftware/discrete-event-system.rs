//! Port of `src/des/main-from-json.ts`.
//!
//! CLI: run any registered DES model from a JSON spec file (with
//! `--list` / `--schema` / `--example` subcommands).
//!
//! `process.argv` → `std::env::args`; `process.exit` → `std::process::exit`;
//! `fs` → `std::fs`.
//!
//! PORT NOTE: the TS imports `./general/des-registry`
//! (`runFromJsonFile`, `listModels`, `getModel`). There is no
//! `crate::des::general::des_registry` in the Rust tree yet, so a minimal
//! registry is stubbed locally (empty model set; `run_from_json_file` reads the
//! file but reports that no models are registered). The CLI dispatch — the
//! substance of this script — is ported faithfully. Replace the local
//! `des_registry` stub with `use crate::des::general::des_registry::{...}` once
//! that module is ported. JSON rendering uses pre-built strings (no `serde`).

#![allow(dead_code)]

use des_registry::{get_model, list_models, run_from_json_file, RunFromJsonOptions};

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

/// Entry point (TS top-level `main`).
pub fn run() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        print_help();
        std::process::exit(if argv.is_empty() { 1 } else { 0 });
    }

    if argv[0] == "--list" {
        let models = list_models();
        println!("Registered models ({}):", models.len());
        for m in &models {
            println!("  {:<24} — {}", m.id, m.description);
        }
        return;
    }
    if argv[0] == "--schema" && argv.len() > 1 {
        match get_model(&argv[1]) {
            Some(reg) => {
                println!("Model: {}", reg.id);
                println!("Description: {}", reg.description);
                println!("Schema:");
                println!("{}", reg.schema_json);
            }
            None => {
                eprintln!("Unknown model: {}", argv[1]);
                std::process::exit(1);
            }
        }
        return;
    }
    if argv[0] == "--example" && argv.len() > 1 {
        match get_model(&argv[1]) {
            Some(reg) if !reg.examples.is_empty() => {
                println!("{}", reg.examples[0].spec_json);
            }
            Some(_) => {
                eprintln!("No examples registered for \"{}\".", argv[1]);
                std::process::exit(1);
            }
            None => {
                eprintln!("Unknown model: {}", argv[1]);
                std::process::exit(1);
            }
        }
        return;
    }

    let spec_path = &argv[0];
    if !std::path::Path::new(spec_path).exists() {
        eprintln!("Spec file not found: {spec_path}");
        std::process::exit(1);
    }
    match run_from_json_file(spec_path, RunFromJsonOptions { verbose: true }) {
        Ok(summary) => {
            if !summary.outputs.is_empty() {
                println!();
                println!("Outputs written:");
                for o in &summary.outputs {
                    println!("  [{}] {}", o.kind, o.path);
                }
            }
            println!();
            println!("Total wall-clock time: {} ms", summary.runtime_ms);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

// -----------------------------------------------------------------------------
// PORT NOTE: local stub of `crate::des::general::des_registry` (not yet ported).
// Empty model registry; the runner reads the spec file but reports that no
// models are registered. Replace with the real registry once available.
// -----------------------------------------------------------------------------
mod des_registry {
    #[derive(Clone, Debug)]
    pub struct ModelInfo {
        pub id: String,
        pub description: String,
    }

    #[derive(Clone, Debug)]
    pub struct ModelExample {
        pub spec_json: String,
    }

    #[derive(Clone, Debug)]
    pub struct ModelRegistration {
        pub id: String,
        pub description: String,
        pub schema_json: String,
        pub examples: Vec<ModelExample>,
    }

    #[derive(Clone, Debug)]
    pub struct RunOutput {
        pub kind: String,
        pub path: String,
    }

    #[derive(Clone, Debug)]
    pub struct RunSummary {
        pub outputs: Vec<RunOutput>,
        pub runtime_ms: u128,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct RunFromJsonOptions {
        pub verbose: bool,
    }

    /// Stub: no models registered (see PORT NOTE).
    pub fn list_models() -> Vec<ModelInfo> {
        Vec::new()
    }

    /// Stub: no models registered (see PORT NOTE).
    pub fn get_model(_id: &str) -> Option<ModelRegistration> {
        None
    }

    /// Stub: reads the file to validate it exists, then reports the registry is
    /// empty (the real dispatcher lives in the un-ported `des_registry`).
    pub fn run_from_json_file(path: &str, _opts: RunFromJsonOptions) -> Result<RunSummary, String> {
        let _spec = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        Err("des_registry not ported: no models registered (see PORT NOTE)".to_string())
    }
}
