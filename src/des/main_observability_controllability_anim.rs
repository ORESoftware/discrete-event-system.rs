//! Port of `src/des/main-observability-controllability-anim.ts`.
//!
//! Generates an HTML slideshow walking through the controllability /
//! observability tests for LTI, MDP, and POMDP systems.
//!
//! Conversion notes:
//!   - `class ObsCtrlAnimator` → struct + impl; async `run()` → [`run`].
//!
//! The structural evaluator itself is fully ported in
//! `crate::des::main_observability_controllability`; this renderer is the
//! animated storyboard counterpart.

#![allow(dead_code)]

use std::io;
use std::path::Path;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::obs_ctrl_scene::{ObsCtrlScene, OC_STAGE_H, OC_STAGE_W};

struct ObsCtrlAnimator;

impl ObsCtrlAnimator {
    fn output_paths() -> (String, String) {
        let dir = Path::new("out").join("obs-ctrl");
        let frames = dir.join("animation.frames.jsonl");
        let html = dir.join("animation.html");
        (
            frames.to_string_lossy().into_owned(),
            html.to_string_lossy().into_owned(),
        )
    }

    fn run(&self) -> io::Result<()> {
        let scene = ObsCtrlScene::new();
        let (frames_path, html_path) = Self::output_paths();
        let mut recorder = FrameRecorder::new(FrameRecorderOpts {
            frames_path,
            html_path: Some(html_path.clone()),
            width: OC_STAGE_W,
            height: OC_STAGE_H,
            fps: Some(0.8),
            title: Some("Controllability & Observability".to_string()),
            subtitle: Some(
                "Kalman ranks, MDP reachability/entropy, and POMDP distinguishability/information."
                    .to_string(),
            ),
            background: Some("#0b1021".to_string()),
            live_tick_line: Some(false),
            record_every_ticks: Some(1.0),
            visual_blocks: None,
        })?;
        for (i, step) in scene.steps().iter().cloned().enumerate() {
            recorder.frame(i as f64, i as f64, || step);
        }
        let recorded = recorder.get_frame_count();
        let anim = recorder.finish()?;
        println!(
            "Obs/Ctrl animation: {} storyboard frames -> {}",
            anim.frames.len().max(recorded as usize),
            html_path
        );
        Ok(())
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    ObsCtrlAnimator
        .run()
        .expect("write observability/controllability animation");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_matches_site_link() {
        let (_, html) = ObsCtrlAnimator::output_paths();
        assert!(html.ends_with("out/obs-ctrl/animation.html"));
    }
}
