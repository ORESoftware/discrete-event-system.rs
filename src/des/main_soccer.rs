//! Writes `out/soccer-sim.html` for the 2D live-match soccer prototype.

pub fn try_run() -> std::io::Result<()> {
    if std::env::var("SOCCER_ARTIFACTS_PLAYBACK_ONLY").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }) {
        let paths = crate::des::general::soccer::try_write_soccer_playback_artifacts()?;
        crate::des::general::soccer::print_soccer_playback_artifact_paths(&paths);
        return Ok(());
    }

    let paths = crate::des::general::soccer::try_write_soccer_artifacts()?;
    crate::des::general::soccer::print_soccer_artifact_paths(&paths);
    Ok(())
}

pub fn run() {
    if let Err(err) = try_run() {
        eprintln!("main_soccer: {err}");
    }
}
