//! Metadata helpers used by generated file-for-file migration modules.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStatus {
    Scaffolded,
    PortedCore,
    PortedBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrationFile {
    pub ts_source: &'static str,
    pub rust_target: &'static str,
    pub notes: &'static [&'static str],
    pub declarations: &'static [&'static str],
    pub status: MigrationStatus,
}

impl MigrationFile {
    pub const fn scaffolded(
        ts_source: &'static str,
        rust_target: &'static str,
        notes: &'static [&'static str],
        declarations: &'static [&'static str],
    ) -> Self {
        Self {
            ts_source,
            rust_target,
            notes,
            declarations,
            status: MigrationStatus::Scaffolded,
        }
    }

    pub const fn ported_core(
        ts_source: &'static str,
        rust_target: &'static str,
        notes: &'static [&'static str],
        declarations: &'static [&'static str],
    ) -> Self {
        Self {
            ts_source,
            rust_target,
            notes,
            declarations,
            status: MigrationStatus::PortedCore,
        }
    }
}
