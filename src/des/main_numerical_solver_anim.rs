//! Renders an animated HTML player for every model in
//! [`crate::des::general::numerical_solver_models`].
//!
//! Each solver already runs as a `source → solver → sink` DES pipeline that
//! records a per-iteration trace. This binary feeds that trace through the
//! generic [`numerical_solver_scene`](crate::des::animation::scenes::numerical_solver_scene)
//! — drawing the live block diagram plus a growing convergence curve — and
//! writes one standalone `out/numerical-solvers/<slug>.html` per model so they
//! show up on the landing index alongside the control-system animations.

#![allow(dead_code)]

use std::io;
use std::path::Path;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::numerical_solver_scene::{
    build_solver_charts, build_solver_frames, SolverAnimationInput, SolverSeries, SOLVER_STAGE_H,
    SOLVER_STAGE_W,
};
use crate::des::animation::types::FrameParts;
use crate::des::general::numerical_solver_models::{
    run_backprop_mlp, run_differential_evolution, run_gaussian_mixture_em, run_lbfgs,
    run_mean_field_vi, run_metropolis_hastings, run_prim_mst, run_sequence_alignment,
    BackpropMlpParams, DifferentialEvolutionParams, GaussianMixtureEMParams, LbfgsParams,
    MeanFieldVIParams, MetropolisParams, PrimMSTParams, SequenceAlignmentParams,
};

/// Cap on frames per animation so the embedded JSON stays small; long traces are
/// strided down to this many evenly-spaced points (first and last kept).
const MAX_FRAMES: usize = 120;

/// One ready-to-render solver animation.
struct SolverAnim {
    slug: &'static str,
    title: &'static str,
    subtitle: String,
    input: SolverAnimationInput,
}

/// Stride `points` down to at most `max` evenly-spaced samples (keeps endpoints).
fn downsample(points: Vec<(f64, f64)>, max: usize) -> Vec<(f64, f64)> {
    let n = points.len();
    if n <= max || max < 2 {
        return points;
    }
    (0..max).map(|i| points[i * (n - 1) / (max - 1)]).collect()
}

fn series(
    label: &str,
    x_label: &str,
    color: &str,
    points: Vec<(f64, f64)>,
    decimals: usize,
) -> SolverSeries {
    SolverSeries {
        label: label.to_string(),
        x_label: x_label.to_string(),
        color: color.to_string(),
        points: downsample(points, MAX_FRAMES),
        decimals,
    }
}

/// Run every model and package it as an animation.
fn animations() -> Vec<SolverAnim> {
    let mut out: Vec<SolverAnim> = Vec::new();

    // 1. Gradient-based optimization — L-BFGS.
    {
        let r = run_lbfgs(LbfgsParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.iteration as f64, t.value))
            .collect();
        out.push(SolverAnim {
            slug: "lbfgs",
            title: "L-BFGS — gradient descent with curvature memory",
            subtitle: format!(
                "Limited-memory BFGS on a smooth objective; reached f = {} in {} iterations.",
                fmt(r.best_value),
                r.iterations
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("objective f(x)", "iteration", "#38bdf8", points, 4),
            },
        });
    }

    // 2. Dynamic programming — Needleman–Wunsch sequence alignment.
    {
        let r = run_sequence_alignment(SequenceAlignmentParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.row as f64, t.running_best))
            .collect();
        out.push(SolverAnim {
            slug: "sequence-alignment",
            title: "Needleman–Wunsch — global sequence alignment",
            subtitle: format!(
                "Dynamic-programming alignment; score {} at {:.0}% identity over {} rows.",
                fmt(r.score),
                r.identity * 100.0,
                r.rows
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("running best score", "DP row", "#a78bfa", points, 1),
            },
        });
    }

    // 3. Monte Carlo / MCMC — random-walk Metropolis.
    {
        let r = run_metropolis_hastings(MetropolisParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.iteration as f64, t.value))
            .collect();
        out.push(SolverAnim {
            slug: "metropolis-hastings",
            title: "Metropolis–Hastings — random-walk MCMC",
            subtitle: format!(
                "Sampling the target chain: mean {} ± {}, acceptance {:.0}%.",
                fmt(r.mean),
                fmt(r.std),
                r.acceptance_rate * 100.0
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("chain state x", "sample", "#22d3ee", points, 3),
            },
        });
    }

    // 4. Evolutionary algorithms — differential evolution.
    {
        let r = run_differential_evolution(DifferentialEvolutionParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.generation as f64, t.best_value))
            .collect();
        out.push(SolverAnim {
            slug: "differential-evolution",
            title: "Differential Evolution — DE/rand/1/bin",
            subtitle: format!(
                "Population search over {} generations; best fitness {}.",
                r.generations,
                fmt(r.best_value)
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("best fitness", "generation", "#34d399", points, 4),
            },
        });
    }

    // 5. Graph optimization — Prim's minimum spanning tree.
    {
        let r = run_prim_mst(PrimMSTParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.step as f64, t.total_weight))
            .collect();
        out.push(SolverAnim {
            slug: "prim-mst",
            title: "Prim's algorithm — minimum spanning tree",
            subtitle: format!(
                "Greedy MST over {} nodes; total weight {} ({} edges).",
                r.node_count,
                fmt(r.total_weight),
                r.mst_edges.len()
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("MST total weight", "edge added", "#fbbf24", points, 2),
            },
        });
    }

    // 6. Deep-learning optimization — backprop MLP.
    {
        let r = run_backprop_mlp(BackpropMlpParams::default());
        let points = r.trace.iter().map(|t| (t.epoch as f64, t.loss)).collect();
        out.push(SolverAnim {
            slug: "backprop-mlp",
            title: "Backpropagation — MLP gradient descent",
            subtitle: format!(
                "Single-hidden-layer MLP trained {} epochs; loss {}, accuracy {:.0}%.",
                r.epochs,
                fmt(r.final_loss),
                r.accuracy * 100.0
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("training loss", "epoch", "#f472b6", points, 4),
            },
        });
    }

    // 7. Probabilistic inference — EM for a Gaussian mixture.
    {
        let r = run_gaussian_mixture_em(GaussianMixtureEMParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.iteration as f64, t.log_likelihood))
            .collect();
        out.push(SolverAnim {
            slug: "gaussian-mixture-em",
            title: "Expectation–Maximization — Gaussian mixture",
            subtitle: format!(
                "EM for a {}-component 1-D mixture; log-likelihood {} after {} iterations.",
                r.weights.len(),
                fmt(r.log_likelihood),
                r.iterations
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("log-likelihood", "EM iteration", "#c084fc", points, 3),
            },
        });
    }

    // 8. Probabilistic inference — mean-field variational inference.
    {
        let r = run_mean_field_vi(MeanFieldVIParams::default());
        let points = r
            .trace
            .iter()
            .map(|t| (t.iteration as f64, t.expected_precision))
            .collect();
        out.push(SolverAnim {
            slug: "mean-field-vi",
            title: "Mean-field VI — coordinate-ascent (CAVI)",
            subtitle: format!(
                "Variational Normal–Gamma fit; E[τ] = {}, posterior mean {} after {} iterations.",
                fmt(r.expected_precision),
                fmt(r.posterior_mean),
                r.iterations
            ),
            input: SolverAnimationInput {
                visual_blocks: r.visual_blocks,
                series: series("E[τ] (precision)", "CAVI iteration", "#2dd4bf", points, 4),
            },
        });
    }

    out
}

fn output_paths(slug: &str) -> (String, String) {
    let dir = Path::new("out").join("numerical-solvers");
    let frames = dir.join(format!("{slug}.frames.jsonl"));
    let html = dir.join(format!("{slug}.html"));
    (
        frames.to_string_lossy().into_owned(),
        html.to_string_lossy().into_owned(),
    )
}

fn render(anim: &SolverAnim) -> io::Result<(String, usize)> {
    let (frames_path, html_path) = output_paths(anim.slug);
    let frames = build_solver_frames(&anim.input);
    let charts = build_solver_charts(&anim.input);
    let mut recorder = FrameRecorder::new(FrameRecorderOpts {
        frames_path,
        html_path: Some(html_path.clone()),
        width: SOLVER_STAGE_W,
        height: SOLVER_STAGE_H,
        fps: Some(10.0),
        title: Some(anim.title.to_string()),
        subtitle: Some(anim.subtitle.clone()),
        background: Some("#0b1021".to_string()),
        live_tick_line: Some(false),
        record_every_ticks: Some(1.0),
        visual_blocks: None,
    })?;
    for f in frames {
        let shapes = f.shapes;
        let caption = f.caption;
        recorder.frame(f.t, f.tick, move || FrameParts { shapes, caption });
    }
    recorder.set_charts(charts);
    let anim_doc = recorder.finish()?;
    Ok((html_path, anim_doc.frames.len()))
}

/// Entry point: render every solver animation into `out/numerical-solvers/`.
pub fn run() {
    let dir = Path::new("out").join("numerical-solvers");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("  ! could not create {}: {e}", dir.display());
        return;
    }
    for anim in animations() {
        match render(&anim) {
            Ok((path, frames)) => {
                println!(
                    "Numerical solver animation: {} ({frames} frames) -> {path}",
                    anim.slug
                )
            }
            Err(e) => eprintln!("  ! {} animation failed: {e}", anim.slug),
        }
    }
}

/// `String(n)` style formatting for subtitle readouts.
fn fmt(v: f64) -> String {
    use crate::des::animation::types::{to_exponential, to_fixed};
    if !v.is_finite() {
        return "n/a".to_string();
    }
    let abs = v.abs();
    if abs != 0.0 && (abs >= 1.0e4 || abs < 1.0e-3) {
        to_exponential(v, 2)
    } else {
        to_fixed(v, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_animation_per_model_with_frames() {
        let anims = animations();
        assert_eq!(anims.len(), 8, "expected one animation per solver model");
        for a in &anims {
            let frames = build_solver_frames(&a.input);
            assert!(!frames.is_empty(), "{} produced no frames", a.slug);
            // Pipeline (3 blocks) + plot shapes on the first frame.
            assert!(frames[0].shapes.len() > 4, "{} frame too sparse", a.slug);
            assert!(
                !a.input.visual_blocks.is_empty(),
                "{} has no blocks",
                a.slug
            );
        }
    }

    #[test]
    fn slugs_are_unique_and_paths_match_site_links() {
        let anims = animations();
        let mut slugs: Vec<&str> = anims.iter().map(|a| a.slug).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), anims.len(), "duplicate slug");
        for a in &anims {
            let (_, html) = output_paths(a.slug);
            assert!(html.ends_with(&format!("out/numerical-solvers/{}.html", a.slug)));
        }
    }

    #[test]
    fn downsample_keeps_endpoints_and_caps_length() {
        let pts: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, i as f64)).collect();
        let ds = downsample(pts.clone(), MAX_FRAMES);
        assert_eq!(ds.len(), MAX_FRAMES);
        assert_eq!(ds.first().unwrap().0, 0.0);
        assert_eq!(ds.last().unwrap().0, 999.0);
        // Short series passes through untouched.
        let short: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
        assert_eq!(downsample(short.clone(), MAX_FRAMES).len(), short.len());
    }
}
