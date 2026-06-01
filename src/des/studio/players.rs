//! First-class Studio player artifacts for the generated HTML catalog.
//!
//! The workbench is an authoring UI; these artifacts make the same modeling
//! surfaces playable through the uniform plugin player used by simulations:
//! model execution, N2 structural analysis, and sweep-driver exploration.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde_json::{json, Value};

use crate::des::{model::RunArtifact, plugin::UiControl};

use super::{
    analysis::{analyze_model_spec, StudioAnalysis},
    demos::blocks_doc,
    run::run,
    spec::{compile_model_spec, StudioModelSpec, StudioObjectiveSense, StudioSpecError},
    sweep::{
        run_design_sweep, run_first_design_sweep, StudioSweepCase, StudioSweepError,
        StudioSweepResult,
    },
};

/// Errors emitted while building/writing Studio player pages.
#[derive(Debug)]
pub enum StudioPlayerError {
    Io(io::Error),
    Spec(StudioSpecError),
    Sweep(StudioSweepError),
}

impl std::fmt::Display for StudioPlayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudioPlayerError::Io(e) => write!(f, "{e}"),
            StudioPlayerError::Spec(e) => write!(f, "{e}"),
            StudioPlayerError::Sweep(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StudioPlayerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StudioPlayerError::Io(e) => Some(e),
            StudioPlayerError::Spec(e) => Some(e),
            StudioPlayerError::Sweep(e) => Some(e),
        }
    }
}

impl From<io::Error> for StudioPlayerError {
    fn from(value: io::Error) -> Self {
        StudioPlayerError::Io(value)
    }
}

impl From<StudioSpecError> for StudioPlayerError {
    fn from(value: StudioSpecError) -> Self {
        StudioPlayerError::Spec(value)
    }
}

impl From<StudioSweepError> for StudioPlayerError {
    fn from(value: StudioSweepError) -> Self {
        StudioPlayerError::Sweep(value)
    }
}

/// Run the model and render its wiring diagram as a uniform sim player.
pub fn studio_run_artifact(spec: &StudioModelSpec) -> Result<RunArtifact, StudioSpecError> {
    let mut compiled = compile_model_spec(spec)?;
    let blocks = blocks_doc(&compiled);
    let run_out = run(&mut compiled, spec.steps, spec.dt);
    Ok(run_out.to_artifact(
        "studio-run",
        &format!("{} - run player", spec.name),
        "A Studio model execution rendered as an animated block wiring diagram.",
        blocks,
    ))
}

/// Build a frame-by-frame N2 matrix reveal for OpenMDAO-style inspection.
pub fn n2_analysis_artifact(spec: &StudioModelSpec) -> RunArtifact {
    let analysis = analyze_model_spec(spec);
    let frames = n2_frames(&analysis);
    let results = json!({
        "kind": "studio-n2",
        "analysis": analysis,
    });
    let summary = format!(
        "Analyzed {} components and {} explicit connections.",
        spec.blocks.len(),
        spec.wires.len()
    );
    RunArtifact::sim(
        "studio-n2",
        &format!("{} - N2 analysis player", spec.name),
        "A playable N2 dependency matrix for Studio components and connections.",
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 5.0),
            UiControl::select(
                "metric",
                "Feature metric",
                &[
                    "all",
                    "revealedConnections",
                    "connections",
                    "components",
                    "valid",
                ],
                "all",
                Some("metric"),
            ),
        ],
        &summary,
    )
}

/// Run one named design-variable sweep and render it as a playable driver plot.
pub fn design_sweep_artifact(
    spec: &StudioModelSpec,
    design_variable_name: &str,
) -> Result<RunArtifact, StudioSweepError> {
    run_design_sweep(spec, design_variable_name).map(|sweep| sweep_artifact(spec, sweep))
}

/// Run the first declared design-variable sweep, if present.
pub fn first_design_sweep_artifact(
    spec: &StudioModelSpec,
) -> Result<Option<RunArtifact>, StudioSweepError> {
    run_first_design_sweep(spec).map(|sweep| sweep.map(|sweep| sweep_artifact(spec, sweep)))
}

/// Write the starter Studio player pages under `<out_root>/studio`.
pub fn write_studio_player_html(
    out_root: impl AsRef<Path>,
    spec: &StudioModelSpec,
) -> Result<Vec<PathBuf>, StudioPlayerError> {
    let studio_dir = out_root.as_ref().join("studio");
    fs::create_dir_all(&studio_dir)?;

    let mut paths = Vec::new();

    let run_path = studio_dir.join("run-player.html");
    fs::write(&run_path, studio_run_artifact(spec)?.to_player_html())?;
    paths.push(run_path);

    let n2_path = studio_dir.join("n2-player.html");
    fs::write(&n2_path, n2_analysis_artifact(spec).to_player_html())?;
    paths.push(n2_path);

    if let Some(artifact) = first_design_sweep_artifact(spec)? {
        let sweep_path = studio_dir.join("sweep-player.html");
        fs::write(&sweep_path, artifact.to_player_html())?;
        paths.push(sweep_path);
    }

    Ok(paths)
}

fn n2_frames(analysis: &StudioAnalysis) -> Vec<Value> {
    let total_cells = analysis.n2.len();
    let total_connections = analysis.connections.len();
    let component_count = analysis.components.len();
    let mut frames = Vec::with_capacity(total_cells.max(1) + 1);
    for reveal in 0..=total_cells {
        let revealed_connections: usize = analysis
            .n2
            .iter()
            .take(reveal)
            .map(|cell| cell.connections.len())
            .sum();
        frames.push(json!({
            "t": reveal as f64,
            "step": reveal as f64,
            "components": component_count as f64,
            "connections": total_connections as f64,
            "revealedConnections": revealed_connections as f64,
            "valid": if analysis.validation.ok { 1.0 } else { 0.0 },
            "shapes": n2_shapes(analysis, reveal),
            "caption": n2_caption(analysis, revealed_connections, total_connections),
        }));
    }
    frames
}

fn n2_caption(analysis: &StudioAnalysis, revealed: usize, total: usize) -> String {
    if analysis.validation.ok {
        format!("N2 matrix: {revealed}/{total} connections revealed")
    } else {
        format!(
            "N2 matrix: validation issue - {}",
            analysis
                .validation
                .message
                .as_deref()
                .unwrap_or("model did not compile")
        )
    }
}

fn n2_shapes(analysis: &StudioAnalysis, reveal: usize) -> Vec<Value> {
    let n = analysis.components.len();
    let cell = if n > 8 { 42.0 } else { 56.0 };
    let left = 170.0;
    let top = 76.0;
    let width = cell * n.max(1) as f64;
    let mut shapes = vec![
        text(
            20.0,
            30.0,
            "N2 dependency matrix",
            20.0,
            "start",
            "#0f172a",
            true,
        ),
        text(
            20.0,
            52.0,
            "Rows consume signals; columns provide signals.",
            12.0,
            "start",
            "#475569",
            false,
        ),
        line(left, top, left + width, top, "#94a3b8", 1.0),
        line(left, top, left, top + width, "#94a3b8", 1.0),
    ];

    if n == 0 {
        shapes.push(text(
            35.0,
            100.0,
            "No components declared.",
            14.0,
            "start",
            "#64748b",
            false,
        ));
        return shapes;
    }

    for (idx, component) in analysis.components.iter().enumerate() {
        let label = short_label(&component.label, 18);
        let cx = left + idx as f64 * cell + cell / 2.0;
        let cy = top + idx as f64 * cell + cell / 2.0;
        shapes.push(text(
            cx,
            top - 16.0,
            &label,
            10.0,
            "middle",
            "#334155",
            false,
        ));
        shapes.push(text(
            left - 12.0,
            cy + 3.0,
            &label,
            10.0,
            "end",
            "#334155",
            false,
        ));
    }

    for row in 0..n {
        for col in 0..n {
            let x = left + col as f64 * cell;
            let y = top + row as f64 * cell;
            let connection_idx = analysis
                .n2
                .iter()
                .position(|n2| n2.row == row && n2.col == col);
            let revealed = connection_idx.is_some_and(|idx| idx < reveal);
            let diagonal = row == col;
            let fill = if revealed {
                "#dbeafe"
            } else if diagonal {
                "#ecfeff"
            } else {
                "#ffffff"
            };
            let stroke = if revealed { "#2563eb" } else { "#cbd5e1" };
            shapes.push(rect(x, y, cell, cell, 0.0, fill, stroke, 1.0));
            if diagonal {
                shapes.push(text(
                    x + cell / 2.0,
                    y + cell / 2.0 + 3.0,
                    &short_label(&analysis.components[row].id, 10),
                    9.5,
                    "middle",
                    "#0f172a",
                    true,
                ));
            }
            if let Some(idx) = connection_idx {
                if revealed {
                    let count = analysis.n2[idx].connections.len();
                    shapes.push(circle(
                        x + cell / 2.0,
                        y + cell / 2.0,
                        10.0,
                        "#2563eb",
                        "#1d4ed8",
                        1.0,
                    ));
                    shapes.push(text(
                        x + cell / 2.0,
                        y + cell / 2.0 + 3.0,
                        &count.to_string(),
                        11.0,
                        "middle",
                        "#ffffff",
                        true,
                    ));
                }
            }
        }
    }

    for (idx, n2) in analysis.n2.iter().enumerate().take(reveal) {
        if let Some(connection) = n2.connections.first() {
            let y = top + width + 36.0 + idx as f64 * 18.0;
            shapes.push(text(
                30.0,
                y,
                &format!(
                    "{}:{} -> {}:{}",
                    connection.from, connection.from_port, connection.to, connection.to_port
                ),
                12.0,
                "start",
                "#1e3a8a",
                false,
            ));
        }
    }

    shapes
}

fn sweep_artifact(spec: &StudioModelSpec, sweep: StudioSweepResult) -> RunArtifact {
    let frames = sweep_frames(&sweep);
    let summary = format!(
        "Swept {} across {} cases.",
        sweep.design_variable.name,
        sweep.cases.len()
    );
    let results = json!({
        "kind": "studio-sweep",
        "model": spec.name,
        "sweep": sweep,
    });
    RunArtifact::sim(
        "studio-sweep",
        &format!("{} - sweep driver player", spec.name),
        "A playable parameter-sweep driver plot with objective and constraint traces.",
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 4.0),
            UiControl::select(
                "metric",
                "Feature metric",
                &[
                    "all",
                    "objective",
                    "constraintViolations",
                    "bestObjective",
                    "designValue",
                ],
                "all",
                Some("metric"),
            ),
        ],
        &summary,
    )
}

fn sweep_frames(sweep: &StudioSweepResult) -> Vec<Value> {
    if sweep.cases.is_empty() {
        return vec![json!({
            "t": 0.0,
            "case": 0.0,
            "designValue": 0.0,
            "objective": 0.0,
            "constraintViolations": 0.0,
            "bestObjective": 0.0,
            "shapes": vec![text(35.0, 80.0, "No sweep cases.", 14.0, "start", "#64748b", false)],
            "caption": "No sweep cases were produced.",
        })];
    }

    (0..sweep.cases.len())
        .map(|idx| {
            let case = &sweep.cases[idx];
            let objective = primary_value(case);
            let best_idx = running_best_case_index(&sweep.cases, idx);
            let best_objective = best_idx
                .and_then(|best| sweep.cases.get(best))
                .map(primary_value)
                .unwrap_or(objective);
            json!({
                "t": idx as f64,
                "case": (idx + 1) as f64,
                "designValue": finite_or_zero(case.value),
                "objective": finite_or_zero(objective),
                "constraintViolations": case.constraints.iter().filter(|c| !c.satisfied).count() as f64,
                "bestObjective": finite_or_zero(best_objective),
                "shapes": sweep_shapes(sweep, idx, best_idx),
                "caption": format!(
                    "case {} / {}: {} = {:.4}, objective = {:.4}",
                    idx + 1,
                    sweep.cases.len(),
                    sweep.design_variable.name,
                    finite_or_zero(case.value),
                    finite_or_zero(objective),
                ),
            })
        })
        .collect()
}

fn sweep_shapes(sweep: &StudioSweepResult, current: usize, best_idx: Option<usize>) -> Vec<Value> {
    let x_values: Vec<f64> = sweep
        .cases
        .iter()
        .map(|case| finite_or_zero(case.value))
        .collect();
    let y_values: Vec<f64> = sweep
        .cases
        .iter()
        .map(primary_value)
        .map(finite_or_zero)
        .collect();
    let (x_min, x_max) = padded_range(&x_values);
    let (y_min, y_max) = padded_range(&y_values);
    let plot_x = 82.0;
    let plot_y = 56.0;
    let plot_w = 680.0;
    let plot_h = 330.0;
    let sx = |x: f64| plot_x + ((x - x_min) / (x_max - x_min)) * plot_w;
    let sy = |y: f64| plot_y + plot_h - ((y - y_min) / (y_max - y_min)) * plot_h;

    let mut points = Vec::with_capacity(sweep.cases.len());
    for case in &sweep.cases {
        points.push((
            sx(finite_or_zero(case.value)),
            sy(finite_or_zero(primary_value(case))),
        ));
    }

    let mut shapes = vec![
        text(24.0, 30.0, "Sweep driver", 20.0, "start", "#0f172a", true),
        text(
            24.0,
            52.0,
            &format!("design variable: {}", sweep.design_variable.name),
            12.0,
            "start",
            "#475569",
            false,
        ),
        rect(
            plot_x, plot_y, plot_w, plot_h, 0.0, "#ffffff", "#cbd5e1", 1.0,
        ),
        line(
            plot_x,
            plot_y + plot_h,
            plot_x + plot_w,
            plot_y + plot_h,
            "#64748b",
            1.4,
        ),
        line(plot_x, plot_y, plot_x, plot_y + plot_h, "#64748b", 1.4),
        text(
            plot_x,
            plot_y + plot_h + 24.0,
            &format!("{x_min:.3}"),
            11.0,
            "middle",
            "#64748b",
            false,
        ),
        text(
            plot_x + plot_w,
            plot_y + plot_h + 24.0,
            &format!("{x_max:.3}"),
            11.0,
            "middle",
            "#64748b",
            false,
        ),
        text(
            plot_x - 10.0,
            plot_y + plot_h,
            &format!("{y_min:.3}"),
            11.0,
            "end",
            "#64748b",
            false,
        ),
        text(
            plot_x - 10.0,
            plot_y + 4.0,
            &format!("{y_max:.3}"),
            11.0,
            "end",
            "#64748b",
            false,
        ),
    ];

    for idx in 1..=current.min(points.len().saturating_sub(1)) {
        let (x1, y1) = points[idx - 1];
        let (x2, y2) = points[idx];
        shapes.push(line(x1, y1, x2, y2, "#2563eb", 2.2));
    }

    for (idx, (x, y)) in points.iter().enumerate().take(current + 1) {
        let is_current = idx == current;
        let is_best = Some(idx) == best_idx;
        let fill = if is_current {
            "#ef4444"
        } else if is_best {
            "#16a34a"
        } else {
            "#60a5fa"
        };
        let radius = if is_current || is_best { 6.5 } else { 4.5 };
        shapes.push(circle(*x, *y, radius, fill, "#1e3a8a", 1.0));
    }

    if let Some(best) = best_idx.and_then(|idx| sweep.cases.get(idx).map(|case| (idx, case))) {
        let (idx, case) = best;
        let (x, y) = points[idx];
        shapes.push(text(
            x + 10.0,
            y - 10.0,
            &format!("best {:.3}", finite_or_zero(primary_value(case))),
            11.0,
            "start",
            "#166534",
            true,
        ));
    }

    shapes
}

fn running_best_case_index(cases: &[StudioSweepCase], upto: usize) -> Option<usize> {
    cases
        .iter()
        .enumerate()
        .take(upto + 1)
        .filter(|(_, case)| {
            case.constraints
                .iter()
                .all(|constraint| constraint.satisfied)
        })
        .filter_map(|(idx, case)| case_score(case).map(|score| (idx, score)))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
}

fn case_score(case: &StudioSweepCase) -> Option<f64> {
    let first = case.objectives.first()?;
    Some(match first.sense {
        StudioObjectiveSense::Minimize => first.value,
        StudioObjectiveSense::Maximize => -first.value,
        StudioObjectiveSense::Track => first.error.unwrap_or(first.value).abs(),
    })
}

fn primary_value(case: &StudioSweepCase) -> f64 {
    case.objectives
        .first()
        .map(|objective| objective.value)
        .or_else(|| case.final_signals.values().next().copied())
        .unwrap_or(0.0)
}

fn padded_range(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if (max - min).abs() < f64::EPSILON {
        let pad = (max.abs() * 0.1).max(1.0);
        (min - pad, max + pad)
    } else {
        let pad = (max - min) * 0.08;
        (min - pad, max + pad)
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn short_label(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(3).max(1);
    let mut out: String = value.chars().take(keep).collect();
    out.push_str("...");
    out
}

fn rect(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    rx: f64,
    fill: &str,
    stroke: &str,
    stroke_width: f64,
) -> Value {
    json!({
        "kind": "rect",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": rx,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width,
    })
}

fn line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) -> Value {
    json!({
        "kind": "line",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke": stroke,
        "strokeWidth": stroke_width,
    })
}

fn circle(x: f64, y: f64, r: f64, fill: &str, stroke: &str, stroke_width: f64) -> Value {
    json!({
        "kind": "circle",
        "x": x,
        "y": y,
        "r": r,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width,
    })
}

fn text(
    x: f64,
    y: f64,
    value: &str,
    font_size: f64,
    anchor: &str,
    fill: &str,
    bold: bool,
) -> Value {
    json!({
        "kind": "text",
        "x": x,
        "y": y,
        "text": value,
        "fontSize": font_size,
        "anchor": anchor,
        "fill": fill,
        "fontWeight": if bold { "bold" } else { "normal" },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::studio::starter_model_spec;

    #[test]
    fn n2_player_has_reveal_frames() {
        let artifact = n2_analysis_artifact(&starter_model_spec());
        assert_eq!(artifact.kind, "studio-n2");
        assert!(artifact.frames.len() >= 2);
        assert!(artifact.to_player_html().contains("N2 analysis player"));
    }

    #[test]
    fn sweep_player_has_case_frames() {
        let artifact = design_sweep_artifact(&starter_model_spec(), "gain.k").unwrap();
        assert_eq!(artifact.kind, "studio-sweep");
        assert_eq!(artifact.frames.len(), 9);
        assert!(artifact.to_jsonl().contains("designValue"));
    }

    #[test]
    fn writes_studio_player_pages() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "des-studio-player-test-{}-{nonce}",
            std::process::id()
        ));
        let paths = write_studio_player_html(&root, &starter_model_spec()).unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().any(|path| path.ends_with("run-player.html")));
        assert!(paths.iter().any(|path| path.ends_with("n2-player.html")));
        assert!(paths.iter().any(|path| path.ends_with("sweep-player.html")));
        let html = std::fs::read_to_string(root.join("studio/sweep-player.html")).unwrap();
        assert!(html.contains("sweep driver player"));
        let _ = std::fs::remove_dir_all(root);
    }
}
