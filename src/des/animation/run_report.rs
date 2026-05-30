//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/run-report.ts`
//! Rust target: `src/des/animation/run_report.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/run-report.ts",
    "src/des/animation/run_report.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/run_report.rs", "- Convert MetricRow, ReportSection, IndexEntry, IndexGroup, CatalogEntry, and CatalogSection to serde-friendly structs.", "- RunReportPage and SimulationIndexPage become builder structs with inherent add_* methods and Result<String, ReportRenderError> render methods.", "- HTML/string builders should use template/writer helpers; keep escape as a private helper or small trait-free utility.", "- Preserve relative-link behavior and avoid global state so the module ports cleanly to Rust ownership."],
    &["CatalogEntry", "CatalogSection", "IndexEntry", "IndexGroup", "MetricRow", "ReportSection", "RunReportPage", "SimulationIndexPage"],
);
