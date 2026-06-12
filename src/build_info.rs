//! Release identity (crate version + git commit + timestamps) baked in at
//! compile time by `build.rs`.
//!
//! The values come from compile-time env vars the build script emits
//! (`DES_ENGINE_GIT_COMMIT`, `DES_ENGINE_GIT_COMMIT_DATE`,
//! `DES_ENGINE_BUILD_TIMESTAMP`) plus Cargo's own `CARGO_PKG_*`. A parent binary
//! that embeds several engines (e.g. the `dd-des-rs` web server pulling in both
//! `des_engine` and `soccer_engine`) can surface each layer's identity so an
//! operator can answer "what's actually running?" from the live UI.

use serde::Serialize;

/// A single component's release identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildInfo {
    /// Crate name (`CARGO_PKG_NAME`).
    pub name: &'static str,
    /// Crate semver (`CARGO_PKG_VERSION`).
    pub version: &'static str,
    /// Full git commit hash, or `"unknown"` if it could not be resolved.
    pub git_commit: &'static str,
    /// Abbreviated (12-char) commit, for compact display.
    pub git_commit_short: String,
    /// Committer date of the embedded commit (ISO 8601), the "release" time.
    pub commit_date: &'static str,
    /// UTC instant the binary was compiled (ISO 8601).
    pub build_timestamp: &'static str,
}

impl BuildInfo {
    /// Construct a `BuildInfo`, deriving `git_commit_short` from `git_commit`.
    /// Used by sibling crates (`soccer_engine`, the web server) so every layer
    /// reports the same JSON shape from its own compile-time `env!` values.
    pub fn new(
        name: &'static str,
        version: &'static str,
        git_commit: &'static str,
        commit_date: &'static str,
        build_timestamp: &'static str,
    ) -> Self {
        BuildInfo {
            name,
            version,
            git_commit,
            git_commit_short: short_commit(git_commit),
            commit_date,
            build_timestamp,
        }
    }
}

/// Shorten a commit label for display without splitting a multi-byte char and
/// without touching the `"unknown"` sentinel.
pub fn short_commit(commit: &str) -> String {
    if commit == "unknown" {
        commit.to_string()
    } else {
        commit.chars().take(12).collect()
    }
}

/// This crate's (`des_engine`) release identity.
pub fn build_info() -> BuildInfo {
    BuildInfo::new(
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("DES_ENGINE_GIT_COMMIT"),
        env!("DES_ENGINE_GIT_COMMIT_DATE"),
        env!("DES_ENGINE_BUILD_TIMESTAMP"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_reports_this_crate_identity() {
        let info = build_info();
        assert_eq!(info.name, "des_engine");
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        // The build script always emits a value (sentinel "unknown" at worst).
        assert!(!info.git_commit.is_empty());
        assert!(!info.commit_date.is_empty());
        assert!(!info.build_timestamp.is_empty());
    }

    #[test]
    fn short_commit_is_byte_safe_and_preserves_sentinel() {
        assert_eq!(short_commit("unknown"), "unknown");
        assert_eq!(short_commit("abcdef1234567890"), "abcdef123456");
        let accented = "\u{e9}".repeat(13);
        assert_eq!(short_commit(&accented), "\u{e9}".repeat(12));
    }

    #[test]
    fn build_info_serializes_to_json_object() {
        let json = serde_json::to_value(build_info()).expect("serialize");
        assert!(json.get("name").is_some());
        assert!(json.get("gitCommitShort").is_none()); // fields are snake_case
        assert!(json.get("git_commit_short").is_some());
        assert!(json.get("commit_date").is_some());
        assert!(json.get("build_timestamp").is_some());
    }
}
