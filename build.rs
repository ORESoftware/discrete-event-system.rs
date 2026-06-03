use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=DES_ENGINE_GIT_COMMIT");
    println!("cargo:rerun-if-changed=.git/HEAD");

    let commit = std::env::var("DES_ENGINE_GIT_COMMIT")
        .ok()
        .and_then(non_empty)
        .or_else(git_commit)
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=DES_ENGINE_GIT_COMMIT={commit}");
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok().and_then(non_empty)
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
