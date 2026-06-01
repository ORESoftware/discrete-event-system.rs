//! Render HTML players for the built-in calculus-of-variations models.

#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::calculus_of_variations_scene::{
    build_variational_animation, COV_STAGE_H, COV_STAGE_W,
};
use crate::des::animation::types::FrameParts;
use crate::des::general::calculus_of_variations::{
    built_in_variational_models, VariationalSolutionModel,
};

pub const CALCULUS_OF_VARIATIONS_OUT_DIR: &str = "calculus-of-variations";

struct VariationalAnim {
    slug: String,
    model: VariationalSolutionModel,
}

fn animations() -> Vec<VariationalAnim> {
    built_in_variational_models()
        .into_iter()
        .map(|model| VariationalAnim {
            slug: model.problem.id.clone(),
            model,
        })
        .collect()
}

fn output_paths(out_root: impl AsRef<Path>, slug: &str) -> (PathBuf, PathBuf) {
    let dir = out_root.as_ref().join(CALCULUS_OF_VARIATIONS_OUT_DIR);
    (
        dir.join(format!("{slug}.frames.jsonl")),
        dir.join(format!("{slug}.html")),
    )
}

fn render(out_root: impl AsRef<Path>, anim: &VariationalAnim) -> io::Result<(PathBuf, usize)> {
    let animation = build_variational_animation(&anim.model);
    let charts = animation.charts.clone().unwrap_or_default();
    let (frames_path, html_path) = output_paths(out_root, &anim.slug);
    let mut recorder = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.to_string_lossy().into_owned(),
        html_path: Some(html_path.to_string_lossy().into_owned()),
        width: COV_STAGE_W,
        height: COV_STAGE_H,
        fps: Some(animation.fps),
        title: animation.title.clone(),
        subtitle: animation.subtitle.clone(),
        background: animation.background.clone(),
        live_tick_line: Some(false),
        record_every_ticks: Some(1.0),
        visual_blocks: None,
    })?;
    for frame in animation.frames {
        let shapes = frame.shapes;
        let caption = frame.caption;
        recorder.frame(frame.t, frame.tick, move || FrameParts { shapes, caption });
    }
    recorder.set_charts(charts);
    let recorded = recorder.finish()?;
    Ok((html_path, recorded.frames.len()))
}

/// Write one standalone HTML player per built-in variational model.
pub fn write_calculus_of_variations_players(
    out_root: impl AsRef<Path>,
) -> io::Result<Vec<PathBuf>> {
    let dir = out_root.as_ref().join(CALCULUS_OF_VARIATIONS_OUT_DIR);
    std::fs::create_dir_all(&dir)?;
    let mut paths = Vec::new();
    for anim in animations() {
        let (path, _) = render(out_root.as_ref(), &anim)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Entry point for `cargo run --bin main_calculus_of_variations_anim`.
pub fn run() {
    for anim in animations() {
        match render("out", &anim) {
            Ok((path, frames)) => {
                println!(
                    "Calculus-of-variations animation: {} ({frames} frames) -> {}",
                    anim.slug,
                    path.display()
                );
            }
            Err(e) => eprintln!("  ! {} animation failed: {e}", anim.slug),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_animation_per_variational_model() {
        let anims = animations();
        assert_eq!(anims.len(), 3);
        let slugs = anims.iter().map(|a| a.slug.as_str()).collect::<Vec<_>>();
        assert!(slugs.contains(&"shortest-curve"));
        assert!(slugs.contains(&"brachistochrone"));
        assert!(slugs.contains(&"minimal-surface-catenoid"));
    }

    #[test]
    fn output_paths_match_site_links() {
        let (_, html) = output_paths("out", "brachistochrone");
        assert!(html.ends_with("out/calculus-of-variations/brachistochrone.html"));
    }
}
