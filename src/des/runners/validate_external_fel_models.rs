//! Compatibility entrypoint for the legacy `validate-external-fel-models` runner.
//!
//! The original Rust port kept a separate validator that wrote placeholder JSON
//! and synthesized external FEL payloads. The canonical implementation is now
//! [`crate::des::runners::compare_external_fel_models`], which writes real shared
//! JSON inputs, runs the Rust models directly, invokes optional external modules,
//! parses their output when available, and reports missing external scripts as
//! clean skips.

/// Run the canonical external-FEL comparison suite.
pub fn run() {
    let code = crate::des::runners::compare_external_fel_models::run();
    if code != 0 {
        std::process::exit(code);
    }
}
