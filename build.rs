use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const UNKNOWN: &str = "unknown";

// Prefix for the compile-time env vars this script emits. Each crate that
// surfaces a release identity (git commit + commit date + build timestamp) uses
// its own prefix so a parent binary embedding several engines can tell the
// layers apart: `des_engine` → DES_ENGINE_*, `soccer_engine` → SOCCER_ENGINE_*,
// the web server → DD_DES_RS_*.
const PREFIX: &str = "DES_ENGINE";

fn main() {
    println!("cargo:rerun-if-env-changed={PREFIX}_GIT_COMMIT");
    emit_git_rerun_paths();

    let commit = std::env::var(format!("{PREFIX}_GIT_COMMIT"))
        .ok()
        .and_then(sanitize_commit_label)
        .or_else(|| {
            if repo_git_metadata_exists() {
                git_commit()
            } else {
                None
            }
        })
        .unwrap_or_else(|| UNKNOWN.to_string());

    // Committer date of the embedded commit — the meaningful "release" time,
    // stable across rebuilds. ISO 8601 / RFC 3339 (`git log --format=%cI`).
    let commit_date = if repo_git_metadata_exists() {
        git_commit_date().unwrap_or_else(|| UNKNOWN.to_string())
    } else {
        UNKNOWN.to_string()
    };

    // When this binary was actually compiled (UTC). Distinct from the commit
    // date: a rebuild of the same commit advances this but not the commit date.
    let build_timestamp = build_timestamp_iso_utc();

    println!("cargo:rustc-env={PREFIX}_GIT_COMMIT={commit}");
    println!("cargo:rustc-env={PREFIX}_GIT_COMMIT_DATE={commit_date}");
    println!("cargo:rustc-env={PREFIX}_BUILD_TIMESTAMP={build_timestamp}");
}

fn emit_git_rerun_paths() {
    if !repo_git_metadata_exists() {
        return;
    }

    println!("cargo:rerun-if-changed=.git");
    if let Some(path) = git_path("HEAD") {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(branch_ref) = git_output(["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_path(&branch_ref) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(path) = git_path("packed-refs") {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn repo_git_metadata_exists() -> bool {
    Path::new(".git").exists()
}

fn git_commit() -> Option<String> {
    git_output(["rev-parse", "HEAD"]).and_then(sanitize_commit_label)
}

fn git_commit_date() -> Option<String> {
    git_output(["log", "-1", "--format=%cI"]).and_then(sanitize_iso_timestamp)
}

fn git_path(path: &str) -> Option<String> {
    git_output(["rev-parse", "--git-path", path])
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if output.status.success() {
        non_empty(String::from_utf8(output.stdout).ok()?)
    } else {
        None
    }
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn sanitize_commit_label(value: String) -> Option<String> {
    let trimmed = value.trim();
    let first = trimmed.split_ascii_whitespace().next()?;
    if first.is_empty() || first.len() > 80 {
        return None;
    }
    if first
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
    {
        Some(first.to_string())
    } else {
        None
    }
}

// Keep the committer-date label to a small, safe character set (digits, `-`,
// `:`, `T`, `+`, `Z`, `.`). Anything else (it should never happen for `%cI`)
// degrades to None rather than smuggling junk into a compile-time string.
fn sanitize_iso_timestamp(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 40 {
        return None;
    }
    if trimmed
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '-' | ':' | 'T' | '+' | 'Z' | '.'))
    {
        Some(trimmed.to_string())
    } else {
        None
    }
}

// Format the build instant as an ISO 8601 / RFC 3339 UTC string with no
// external crate (chrono is not a dependency). Honors SOURCE_DATE_EPOCH for
// reproducible builds.
fn build_timestamp_iso_utc() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });
    unix_seconds_to_iso_utc(secs)
}

fn unix_seconds_to_iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// Howard Hinnant's days-from-civil inverse: days since 1970-01-01 → (y, m, d).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}
