//! Port of `src/des/animation/scenes/soccer-scene.ts`.
//!
//! Builds frames + companion charts for the 7v7 soccer pitch animation: a side
//! panel (scoreboard, period, affinity bar), the pitch with on-field players,
//! and a bench list. Substitutions animate naturally as players teleport
//! between frames.
//!
//! PORT NOTE: only the subset of `crate::des::general::soccer_rotation::
//! SoccerProblem` the scene reads is mirrored locally in [`SoccerProblem`].

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1100.0;
pub const STAGE_H: f64 = 640.0;

const PITCH_X: f64 = 320.0;
const PITCH_Y: f64 = 60.0;
const PITCH_W: f64 = 580.0;
const PITCH_H: f64 = 460.0;

const BENCH_X: f64 = 920.0;
const BENCH_Y: f64 = 60.0;
const BENCH_W: f64 = 160.0;
const BENCH_ROW_H: f64 = 56.0;

const META_X: f64 = 30.0;
const META_Y: f64 = 60.0;
const META_W: f64 = 270.0;
const META_H: f64 = 460.0;

// 7 position layout on the pitch (4-2-1 / 2-3-1 youth diamond-ish).
// Coordinates relative to PITCH origin in [0, 1].
const POSITION_RELATIVE: [(f64, f64); 7] = [
    (0.5, 0.95),  // A: GK (back)
    (0.18, 0.72), // B: LB
    (0.82, 0.72), // C: RB
    (0.5, 0.55),  // D: CB / sweeper
    (0.27, 0.30), // E: LM
    (0.73, 0.30), // F: RM
    (0.5, 0.10),  // G: ST
];

const COLOR_PITCH: &str = "#16a34a"; // grass
const COLOR_LINE: &str = "#ffffff";
const COLOR_PLAYER_FIELD: &str = "#1d4ed8";
const COLOR_PLAYER_BENCH: &str = "#94a3b8";
const COLOR_GOAL_US: &str = "#facc15";
const COLOR_GOAL_THEM: &str = "#dc2626";

// PORT NOTE: local mirror of the soccer model (subset used by the scene).
#[derive(Clone, Debug, Default)]
pub struct SoccerProblem {
    pub num_positions: usize,
    pub num_periods: f64,
    pub player_names: Option<Vec<String>>,
    pub position_names: Option<Vec<String>>,
}

/// `'us' | 'them'` goal flash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalSide {
    Us,
    Them,
}

pub struct SoccerFrameInput<'a> {
    /// Game minute (integer).
    pub t: f64,
    /// Period index (0-based).
    pub period: f64,
    /// `positions[pos] = playerId`.
    pub positions: Vec<usize>,
    /// `bench[i] = playerId`.
    pub bench: Vec<usize>,
    pub goals_for: f64,
    pub goals_against: f64,
    /// Average affinity in [0, 1].
    pub affinity_now: f64,
    /// Optional flash for "this minute had a goal event".
    pub goal_this_tick: Option<GoalSide>,
    pub problem: &'a SoccerProblem,
}

fn player_name(names: &Option<Vec<String>>, player_id: usize) -> String {
    // `playerNames[playerId] ?? `P${playerId}``.
    names
        .as_ref()
        .and_then(|n| n.get(player_id))
        .cloned()
        .unwrap_or_else(|| format!("P{player_id}"))
}

pub fn build_soccer_frame(_t: f64, _tick: f64, input: &SoccerFrameInput) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let player_names = &input.problem.player_names;
    let position_names = &input.problem.position_names;

    // Side panel: scoreboard, period, affinity bar.
    shapes.push(Shape::Rect(RectShape {
        x: META_X,
        y: META_Y,
        w: META_W,
        h: META_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 36.0,
        text: "7v7 Match".to_string(),
        font_size: Some(22.0),
        fill: Some("#f1f5f9".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 70.0,
        text: format!(
            "Period {} / {}",
            js_num(input.period + 1.0),
            js_num(input.problem.num_periods)
        ),
        font_size: Some(16.0),
        fill: Some("#cbd5e1".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 110.0,
        text: format!(
            "Minute {} / {}",
            js_num(input.t),
            js_num(input.problem.num_periods * 20.0)
        ),
        font_size: Some(14.0),
        fill: Some("#94a3b8".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
    // Score.
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W / 2.0,
        y: META_Y + 170.0,
        text: format!(
            "Us  {}  \u{2014}  {}  Them",
            js_num(input.goals_for),
            js_num(input.goals_against)
        ),
        font_size: Some(26.0),
        fill: Some("#fde68a".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    // Goal flash (last-tick event).
    if input.goal_this_tick == Some(GoalSide::Us) {
        shapes.push(Shape::Rect(RectShape {
            x: META_X + 30.0,
            y: META_Y + 200.0,
            w: META_W - 60.0,
            h: 30.0,
            fill: COLOR_GOAL_US.to_string(),
            opacity: Some(0.9),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: META_X + META_W / 2.0,
            y: META_Y + 222.0,
            text: "GOAL!".to_string(),
            font_size: Some(18.0),
            fill: Some("#000".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    } else if input.goal_this_tick == Some(GoalSide::Them) {
        shapes.push(Shape::Rect(RectShape {
            x: META_X + 30.0,
            y: META_Y + 200.0,
            w: META_W - 60.0,
            h: 30.0,
            fill: COLOR_GOAL_THEM.to_string(),
            opacity: Some(0.9),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: META_X + META_W / 2.0,
            y: META_Y + 222.0,
            text: "CONCEDED".to_string(),
            font_size: Some(16.0),
            fill: Some("#fff".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
    // Affinity bar.
    shapes.push(Shape::Text(TextShape {
        x: META_X + 16.0,
        y: META_Y + 280.0,
        text: "On-field avg affinity".to_string(),
        font_size: Some(12.0),
        fill: Some("#94a3b8".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: META_X + 16.0,
        y: META_Y + 290.0,
        w: META_W - 32.0,
        h: 14.0,
        fill: "#334155".to_string(),
        stroke: Some("#475569".to_string()),
        stroke_width: Some(1.0),
        rx: Some(3.0),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: META_X + 16.0,
        y: META_Y + 290.0,
        w: (META_W - 32.0) * 0.0_f64.max(1.0_f64.min(input.affinity_now)),
        h: 14.0,
        fill: "#22d3ee".to_string(),
        rx: Some(3.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: META_X + META_W - 16.0,
        y: META_Y + 320.0,
        text: format!("{}%", to_fixed(input.affinity_now * 100.0, 0)),
        font_size: Some(11.0),
        fill: Some("#94a3b8".to_string()),
        anchor: Some(Anchor::End),
        ..Default::default()
    }));

    // Period boundary watermark.
    if input.t % 20.0 == 0.0 && input.t > 0.0 {
        shapes.push(Shape::Text(TextShape {
            x: META_X + META_W / 2.0,
            y: META_Y + 380.0,
            text: "\u{27f3} SUB WINDOW".to_string(),
            font_size: Some(18.0),
            fill: Some("#fbbf24".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    // Pitch.
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X,
        y: PITCH_Y,
        w: PITCH_W,
        h: PITCH_H,
        fill: COLOR_PITCH.to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Halfway line.
    shapes.push(Shape::Line(LineShape {
        x1: PITCH_X,
        y1: PITCH_Y + PITCH_H / 2.0,
        x2: PITCH_X + PITCH_W,
        y2: PITCH_Y + PITCH_H / 2.0,
        stroke: COLOR_LINE.to_string(),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Centre circle.
    shapes.push(Shape::Circle(CircleShape {
        x: PITCH_X + PITCH_W / 2.0,
        y: PITCH_Y + PITCH_H / 2.0,
        r: 50.0,
        fill: "none".to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Penalty boxes (top + bottom).
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X + PITCH_W / 2.0 - 90.0,
        y: PITCH_Y,
        w: 180.0,
        h: 70.0,
        fill: "none".to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X + PITCH_W / 2.0 - 90.0,
        y: PITCH_Y + PITCH_H - 70.0,
        w: 180.0,
        h: 70.0,
        fill: "none".to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Goals.
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X + PITCH_W / 2.0 - 30.0,
        y: PITCH_Y - 8.0,
        w: 60.0,
        h: 8.0,
        fill: COLOR_LINE.to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X + PITCH_W / 2.0 - 30.0,
        y: PITCH_Y + PITCH_H,
        w: 60.0,
        h: 8.0,
        fill: COLOR_LINE.to_string(),
        ..Default::default()
    }));

    // Players on the pitch.
    for pos in 0..input.problem.num_positions {
        let slot = POSITION_RELATIVE
            .get(pos)
            .copied()
            .unwrap_or(POSITION_RELATIVE[0]);
        let cx = PITCH_X + slot.0 * PITCH_W;
        let cy = PITCH_Y + slot.1 * PITCH_H;
        let player_id = input.positions[pos];
        let name = player_name(player_names, player_id);
        let pos_name = position_names
            .as_ref()
            .and_then(|p| p.get(pos))
            .cloned()
            .unwrap_or_else(|| pos.to_string());
        shapes.push(Shape::Circle(CircleShape {
            x: cx,
            y: cy,
            r: 22.0,
            fill: COLOR_PLAYER_FIELD.to_string(),
            stroke: Some("#fff".to_string()),
            stroke_width: Some(2.0),
            title: Some(format!("{name}  @  position {pos_name}")),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: cy + 5.0,
            text: name,
            font_size: Some(12.0),
            fill: Some("#fff".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        let pos_label = position_names
            .as_ref()
            .and_then(|p| p.get(pos))
            .cloned()
            .unwrap_or_else(|| "?".to_string());
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: cy - 28.0,
            text: pos_label,
            font_size: Some(11.0),
            fill: Some("#fde68a".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    // Bench.
    shapes.push(Shape::Rect(RectShape {
        x: BENCH_X,
        y: BENCH_Y,
        w: BENCH_W,
        h: BENCH_ROW_H * input.bench.len() as f64 + 40.0,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(6.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: BENCH_X + BENCH_W / 2.0,
        y: BENCH_Y + 26.0,
        text: "BENCH".to_string(),
        font_size: Some(14.0),
        fill: Some("#fde68a".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for (i, &player_id) in input.bench.iter().enumerate() {
        let name = player_name(player_names, player_id);
        let y = BENCH_Y + 40.0 + i as f64 * BENCH_ROW_H;
        shapes.push(Shape::Circle(CircleShape {
            x: BENCH_X + BENCH_W / 2.0 - 30.0,
            y: y + BENCH_ROW_H / 2.0,
            r: 18.0,
            fill: COLOR_PLAYER_BENCH.to_string(),
            stroke: Some("#94a3b8".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: BENCH_X + BENCH_W / 2.0 - 30.0,
            y: y + BENCH_ROW_H / 2.0 + 4.0,
            text: name,
            font_size: Some(11.0),
            fill: Some("#0f172a".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }

    let caption = format!(
        "t={}min  P{}  Us {}-{} Them  affinity {}%",
        js_num(input.t),
        js_num(input.period + 1.0),
        js_num(input.goals_for),
        js_num(input.goals_against),
        to_fixed(input.affinity_now * 100.0, 0)
    );
    FrameParts::with_caption(shapes, caption)
}

/// Build companion charts: per-tick affinity and cumulative goal differential.
pub fn build_soccer_charts(
    ts: &[f64],
    affinity: &[f64],
    goals_for: &[f64],
    goals_against: &[f64],
) -> Vec<ChartSpec> {
    vec![
        ChartSpec {
            x: META_X,
            y: META_Y + META_H + 20.0,
            w: META_W,
            h: 100.0,
            title: Some("On-field avg affinity".to_string()),
            y_min: Some(0.0),
            y_max: Some(1.0),
            series: vec![ChartSeries {
                label: "affinity".to_string(),
                color: "#22d3ee".to_string(),
                t: ts.to_vec(),
                y: affinity.to_vec(),
            }],
            ..Default::default()
        },
        ChartSpec {
            x: PITCH_X,
            y: PITCH_Y + PITCH_H + 20.0,
            w: PITCH_W,
            h: 100.0,
            title: Some("Cumulative goals".to_string()),
            series: vec![
                ChartSeries {
                    label: "us".to_string(),
                    color: COLOR_GOAL_US.to_string(),
                    t: ts.to_vec(),
                    y: goals_for.to_vec(),
                },
                ChartSeries {
                    label: "them".to_string(),
                    color: COLOR_GOAL_THEM.to_string(),
                    t: ts.to_vec(),
                    y: goals_against.to_vec(),
                },
            ],
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_caption_shows_score_and_affinity() {
        let problem = SoccerProblem {
            num_positions: 7,
            num_periods: 4.0,
            player_names: Some((0..14).map(|i| format!("Kid{i}")).collect()),
            position_names: Some(vec![
                "GK".into(),
                "LB".into(),
                "RB".into(),
                "CB".into(),
                "LM".into(),
                "RM".into(),
                "ST".into(),
            ]),
        };
        let input = SoccerFrameInput {
            t: 20.0,
            period: 0.0,
            positions: vec![0, 1, 2, 3, 4, 5, 6],
            bench: vec![7, 8, 9],
            goals_for: 2.0,
            goals_against: 1.0,
            affinity_now: 0.75,
            goal_this_tick: Some(GoalSide::Us),
            problem: &problem,
        };
        let fp = build_soccer_frame(0.0, 0.0, &input);
        assert_eq!(
            fp.caption.as_deref(),
            Some("t=20min  P1  Us 2-1 Them  affinity 75%")
        );
        // Goal flash + sub-window watermark present.
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text == "GOAL!")));
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text.contains("SUB WINDOW"))));
    }
}
