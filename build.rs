use std::path::Path;
use std::process::Command;

const UNKNOWN_COMMIT: &str = "unknown";

fn main() {
    println!("cargo:rerun-if-env-changed=DES_ENGINE_GIT_COMMIT");
    emit_git_rerun_paths();

    let commit = std::env::var("DES_ENGINE_GIT_COMMIT")
        .ok()
        .and_then(sanitize_commit_label)
        .or_else(|| {
            if repo_git_metadata_exists() {
                git_commit()
            } else {
                None
            }
        })
        .unwrap_or_else(|| UNKNOWN_COMMIT.to_string());

    println!("cargo:rustc-env=DES_ENGINE_GIT_COMMIT={commit}");
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
