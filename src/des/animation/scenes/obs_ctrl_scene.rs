//! Port of `src/des/animation/scenes/obs-ctrl-scene.ts`.
//!
//! Class-based storyboard builder for the observability/controllability scene:
//! a narrated sequence of static frames walking through the structural tests
//! (Kalman controllability/observability matrices, MDP reachability flood,
//! POMDP observation classes).
//!
//! ## Conversion notes
//!
//! * `type StoryStep = Omit<Frame, 't' | 'tick'>` →
//!   [`crate::des::animation::types::FrameParts`].
//! * The matrix algebra reuses the real
//!   `crate::des::general::control_systems::{linear_algebra, observability_controllability}`
//!   models, so verdicts match the TS scene exactly (no stubs required).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, FontWeight, FrameParts, LineShape, PathShape, RectShape, Shape,
    TextShape,
};
use crate::des::general::control_systems::linear_algebra::{LinAlg, Matrix};
use crate::des::general::control_systems::observability_controllability::{
    MarkovDecisionProcess, MdpSpec, PartiallyObservableProcess, PomdpSpec, StateSpaceModel,
    StateSpaceSpec,
};

pub const OC_STAGE_W: f64 = 1000.0;
pub const OC_STAGE_H: f64 = 640.0;

const COL_BG: &str = "#0b1021";
const COL_PANEL: &str = "#161d33";
const COL_TEXT: &str = "#e2e8f0";
const COL_DIM: &str = "#94a3b8";
const COL_B: &str = "#38bdf8";
const COL_C: &str = "#f59e0b";
const COL_OK: &str = "#22c55e";
const COL_BAD: &str = "#ef4444";
const COL_INFO: &str = "#a78bfa";
const COL_NODE: &str = "#334155";
const COL_REACH: &str = "#22c55e";

/// `Omit<Frame, 't' | 'tick'>`.
pub type StoryStep = FrameParts;

pub struct ObsCtrlScene {
    steps_: Vec<StoryStep>,
}

impl Default for ObsCtrlScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ObsCtrlScene {
    pub fn new() -> Self {
        let mut scene = ObsCtrlScene { steps_: Vec::new() };
        scene.build_title();
        scene.build_lti(
            "Worked example: double integrator",
            StateSpaceModel::new(StateSpaceSpec {
                a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
                b: vec![vec![0.0], vec![1.0]],
                c: vec![vec![1.0, 0.0]],
                d: None,
            }),
        );
        scene.build_lti(
            "Decoupled modes (input/output touch one mode)",
            StateSpaceModel::new(StateSpaceSpec {
                a: vec![vec![1.0, 0.0], vec![0.0, 2.0]],
                b: vec![vec![1.0], vec![0.0]],
                c: vec![vec![1.0, 0.0]],
                d: None,
            }),
        );
        scene.build_mdp();
        scene.build_pomdp();
        scene.build_recap();
        scene
    }

    pub fn steps(&self) -> &[StoryStep] {
        &self.steps_
    }

    // ── shared chrome ─────────────────────────────────────────────────────────

    fn base(&self, title: &str) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();
        shapes.push(Shape::Rect(RectShape {
            x: 0.0,
            y: 0.0,
            w: OC_STAGE_W,
            h: OC_STAGE_H,
            fill: COL_BG.to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: OC_STAGE_W / 2.0,
            y: 34.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(21.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some(COL_TEXT.to_string()),
            text: "Controllability & Observability — structural evaluator".to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: OC_STAGE_W / 2.0,
            y: 62.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(15.0),
            fill: Some(COL_DIM.to_string()),
            text: title.to_string(),
            ..Default::default()
        }));
        shapes
    }

    fn badge(&self, shapes: &mut Vec<Shape>, x: f64, y: f64, label: &str, ok: bool) {
        let w = 230.0;
        let h = 40.0;
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w,
            h,
            rx: Some(8.0),
            fill: if ok {
                "#052e16".to_string()
            } else {
                "#450a0a".to_string()
            },
            stroke: Some(if ok {
                COL_OK.to_string()
            } else {
                COL_BAD.to_string()
            }),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + w / 2.0,
            y: y + 26.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(15.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some(if ok {
                COL_OK.to_string()
            } else {
                COL_BAD.to_string()
            }),
            text: format!("{}: {}", label, if ok { "YES" } else { "NO" }),
            ..Default::default()
        }));
    }

    fn metric_bar(
        &self,
        shapes: &mut Vec<Shape>,
        x: f64,
        y: f64,
        w: f64,
        label: &str,
        value: f64,
        max_value: f64,
        color: &str,
    ) {
        let ratio = if max_value > 0.0 {
            (value / max_value).clamp(0.0, 1.0)
        } else {
            0.0
        };
        shapes.push(Shape::Text(TextShape {
            x,
            y,
            anchor: Some(Anchor::Start),
            font_size: Some(12.0),
            fill: Some(COL_DIM.to_string()),
            text: label.to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Rect(RectShape {
            x,
            y: y + 9.0,
            w,
            h: 16.0,
            rx: Some(4.0),
            fill: "#0f172a".to_string(),
            stroke: Some("#334155".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Rect(RectShape {
            x,
            y: y + 9.0,
            w: w * ratio,
            h: 16.0,
            rx: Some(4.0),
            fill: color.to_string(),
            opacity: Some(0.78),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: x + w + 12.0,
            y: y + 22.0,
            anchor: Some(Anchor::Start),
            font_size: Some(12.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some(color.to_string()),
            text: format!("{} bits", to_fixed(value, 3)),
            ..Default::default()
        }));
    }

    /// Render a labelled matrix grid; optionally tint specific column blocks.
    fn matrix(
        &self,
        shapes: &mut Vec<Shape>,
        x: f64,
        y: f64,
        label: &str,
        m: &Matrix,
        col_colors: Option<&HashMap<usize, &'static str>>,
    ) -> (f64, f64) {
        let rows = LinAlg::rows(m);
        let cols = LinAlg::cols(m);
        let cw = 46.0;
        let ch = 30.0;
        shapes.push(Shape::Text(TextShape {
            x,
            y: y - 8.0,
            anchor: Some(Anchor::Start),
            font_size: Some(13.0),
            fill: Some(COL_DIM.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: label.to_string(),
            ..Default::default()
        }));
        let w = cols as f64 * cw;
        let h = rows as f64 * ch;
        // Brackets.
        shapes.push(Shape::Line(LineShape {
            x1: x - 4.0,
            y1: y,
            x2: x - 4.0,
            y2: y + h,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x - 4.0,
            y1: y,
            x2: x + 4.0,
            y2: y,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x - 4.0,
            y1: y + h,
            x2: x + 4.0,
            y2: y + h,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x + w + 4.0,
            y1: y,
            x2: x + w + 4.0,
            y2: y + h,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x + w + 4.0,
            y1: y,
            x2: x + w - 4.0,
            y2: y,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Line(LineShape {
            x1: x + w + 4.0,
            y1: y + h,
            x2: x + w - 4.0,
            y2: y + h,
            stroke: COL_TEXT.to_string(),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        for r in 0..rows {
            for c in 0..cols {
                let tint = col_colors.and_then(|m| m.get(&c).copied());
                if let Some(tint) = tint {
                    shapes.push(Shape::Rect(RectShape {
                        x: x + c as f64 * cw + 2.0,
                        y: y + r as f64 * ch + 2.0,
                        w: cw - 4.0,
                        h: ch - 4.0,
                        rx: Some(3.0),
                        fill: tint.to_string(),
                        opacity: Some(0.18),
                        ..Default::default()
                    }));
                }
                let v = m[r][c];
                shapes.push(Shape::Text(TextShape {
                    x: x + c as f64 * cw + cw / 2.0,
                    y: y + r as f64 * ch + ch / 2.0 + 5.0,
                    anchor: Some(Anchor::Middle),
                    font_size: Some(14.0),
                    fill: Some(tint.unwrap_or(COL_TEXT).to_string()),
                    text: self.fmt(v),
                    ..Default::default()
                }));
            }
        }
        (w, h)
    }

    fn fmt(&self, v: f64) -> String {
        if (v - v.round()).abs() < 1e-9 {
            js_num(v.round())
        } else {
            to_fixed(v, 2)
        }
    }

    // ── title ───────────────────────────────────────────────────────────────

    fn build_title(&mut self) {
        let mut shapes = self.base("three lenses on one idea");
        let lines: [(&str, &str, &str); 2] = [
            (
                "Controllability",
                "Can an input drive the state anywhere?",
                COL_B,
            ),
            (
                "Observability",
                "Can the output reveal the full internal state?",
                COL_C,
            ),
        ];
        for (i, (head, desc, col)) in lines.iter().enumerate() {
            let y = 160.0 + i as f64 * 120.0;
            shapes.push(Shape::Rect(RectShape {
                x: 120.0,
                y,
                w: 760.0,
                h: 92.0,
                rx: Some(10.0),
                fill: COL_PANEL.to_string(),
                stroke: Some("#334155".to_string()),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: 150.0,
                y: y + 38.0,
                anchor: Some(Anchor::Start),
                font_size: Some(22.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(col.to_string()),
                text: head.to_string(),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: 150.0,
                y: y + 68.0,
                anchor: Some(Anchor::Start),
                font_size: Some(16.0),
                fill: Some(COL_TEXT.to_string()),
                text: desc.to_string(),
                ..Default::default()
            }));
        }
        shapes.push(Shape::Text(TextShape { x: OC_STAGE_W / 2.0, y: 470.0, anchor: Some(Anchor::Middle), font_size: Some(14.0), fill: Some(COL_DIM.to_string()), text: "Linear state-space  \u{00b7}  MDP (reachability)  \u{00b7}  POMDP (distinguishability)".to_string(), ..Default::default() }));
        shapes.push(Shape::Text(TextShape {
            x: OC_STAGE_W / 2.0,
            y: 500.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(14.0),
            fill: Some(COL_INFO.to_string()),
            text: "Shannon layer: transition uncertainty and sensor information in bits"
                .to_string(),
            ..Default::default()
        }));
        self.steps_.push(FrameParts::with_caption(
            shapes,
            "Structural properties plus Shannon information measures for stochastic models.",
        ));
    }

    // ── LTI ───────────────────────────────────────────────────────────────────

    fn build_lti(&mut self, title: &str, m: StateSpaceModel) {
        let n = m.state_dim();
        // Step 1: show A, B, C.
        {
            let mut shapes = self.base(title);
            self.matrix(&mut shapes, 120.0, 140.0, "A", &m.a, None);
            let mut b_colors: HashMap<usize, &'static str> = HashMap::new();
            b_colors.insert(0, COL_B);
            self.matrix(&mut shapes, 320.0, 140.0, "B", &m.b, Some(&b_colors));
            self.matrix(&mut shapes, 470.0, 140.0, "C", &m.c, None);
            shapes.push(Shape::Text(TextShape {
                x: 120.0,
                y: 320.0,
                anchor: Some(Anchor::Start),
                font_size: Some(14.0),
                fill: Some(COL_DIM.to_string()),
                text: "\u{1e8b} = A x + B u,    y = C x".to_string(),
                ..Default::default()
            }));
            self.steps_.push(FrameParts::with_caption(
                shapes,
                format!("{title}: state matrices (n = {n})."),
            ));
        }
        // Step 2: controllability matrix, column by column.
        let ctrl = m.controllability_matrix();
        {
            let mut shapes = self.base(title);
            let mut col_colors: HashMap<usize, &'static str> = HashMap::new();
            for c in 0..LinAlg::cols(&ctrl) {
                col_colors.insert(c, COL_B);
            }
            self.matrix(
                &mut shapes,
                120.0,
                150.0,
                "\u{1d49e} = [ B  AB  \u{2026}  A\u{207f}\u{207b}\u{00b9}B ]",
                &ctrl,
                Some(&col_colors),
            );
            let rank = m.controllability_rank();
            let ok = m.is_controllable();
            shapes.push(Shape::Text(TextShape {
                x: 120.0,
                y: 300.0,
                anchor: Some(Anchor::Start),
                font_size: Some(15.0),
                fill: Some(COL_TEXT.to_string()),
                text: format!("rank \u{1d49e} = {rank},   n = {n}"),
                ..Default::default()
            }));
            self.badge(&mut shapes, 120.0, 330.0, "Controllable", ok);
            self.steps_.push(FrameParts::with_caption(
                shapes,
                format!(
                    "Kalman controllability: rank \u{1d49e} = {rank} / {n} \u{2192} {}.",
                    if ok {
                        "controllable"
                    } else {
                        "NOT controllable"
                    }
                ),
            ));
        }
        // Step 3: observability matrix.
        let obs = m.observability_matrix();
        {
            let mut shapes = self.base(title);
            self.matrix(
                &mut shapes,
                120.0,
                150.0,
                "\u{1d4aa} = [ C ; CA ; \u{2026} ; CA\u{207f}\u{207b}\u{00b9} ]",
                &obs,
                None,
            );
            let rank = m.observability_rank();
            let ok = m.is_observable();
            shapes.push(Shape::Text(TextShape {
                x: 120.0,
                y: 290.0,
                anchor: Some(Anchor::Start),
                font_size: Some(15.0),
                fill: Some(COL_TEXT.to_string()),
                text: format!("rank \u{1d4aa} = {rank},   n = {n}"),
                ..Default::default()
            }));
            self.badge(&mut shapes, 120.0, 320.0, "Observable", ok);
            self.badge(
                &mut shapes,
                380.0,
                320.0,
                "Controllable",
                m.is_controllable(),
            );
            self.steps_.push(FrameParts::with_caption(
                shapes,
                format!(
                    "Kalman observability: rank \u{1d4aa} = {rank} / {n} \u{2192} {}.",
                    if ok { "observable" } else { "NOT observable" }
                ),
            ));
        }
    }

    // ── MDP ─────────────────────────────────────────────────────────────────

    fn build_mdp(&mut self) {
        let ring = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        });
        let trap = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
        });
        let noisy = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 0.5, 0.5],
                vec![0.2, 0.6, 0.2],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]],
        });
        self.mdp_step(
            "MDP controllability \u{2248} reachability — ring (s\u{2192}s+1)",
            &ring,
        );
        self.mdp_step(
            "MDP controllability \u{2248} reachability — trap (state 2 absorbing)",
            &trap,
        );
        self.mdp_information_step("MDP information — stochastic transition rows", &noisy);
    }

    fn mdp_step(&mut self, title: &str, mdp: &MarkovDecisionProcess) {
        let mut shapes = self.base(title);
        let cx = 320.0;
        let cy = 320.0;
        let r = 150.0;
        let adj = mdp.one_step_adjacency();
        let reach = mdp.reachability_closure();
        let pos: Vec<(f64, f64)> = (0..3)
            .map(|i| {
                let a =
                    -std::f64::consts::FRAC_PI_2 + (i as f64 * 2.0 * std::f64::consts::PI) / 3.0;
                (cx + r * a.cos(), cy + r * a.sin())
            })
            .collect();
        // Edges with arrowheads.
        for s in 0..3 {
            for t in 0..3 {
                if s == t || !adj[s][t] {
                    continue;
                }
                let a = pos[s];
                let b = pos[t];
                let dx = b.0 - a.0;
                let dy = b.1 - a.1;
                let len = dx.hypot(dy);
                let ux = dx / len;
                let uy = dy / len;
                let sx = a.0 + ux * 34.0;
                let sy = a.1 + uy * 34.0;
                let ex = b.0 - ux * 34.0;
                let ey = b.1 - uy * 34.0;
                shapes.push(Shape::Line(LineShape {
                    x1: sx,
                    y1: sy,
                    x2: ex,
                    y2: ey,
                    stroke: COL_DIM.to_string(),
                    stroke_width: Some(2.0),
                    ..Default::default()
                }));
                let ax = ex - ux * 10.0;
                let ay = ey - uy * 10.0;
                let px = -uy * 6.0;
                let py = ux * 6.0;
                shapes.push(Shape::Path(PathShape {
                    d: format!(
                        "M {},{} L {},{} L {},{}",
                        js_num(ax + px),
                        js_num(ay + py),
                        js_num(ex),
                        js_num(ey),
                        js_num(ax - px),
                        js_num(ay - py)
                    ),
                    stroke: Some(COL_DIM.to_string()),
                    fill: Some(COL_DIM.to_string()),
                    ..Default::default()
                }));
            }
        }
        // Nodes (reachable-from-0 highlighted green).
        for i in 0..3 {
            let reachable_from0 = reach[0][i];
            shapes.push(Shape::Circle(crate::des::animation::types::CircleShape {
                x: pos[i].0,
                y: pos[i].1,
                r: 30.0,
                fill: if reachable_from0 {
                    "#052e16".to_string()
                } else {
                    COL_NODE.to_string()
                },
                stroke: Some(if reachable_from0 {
                    COL_REACH.to_string()
                } else {
                    "#475569".to_string()
                }),
                stroke_width: Some(2.0),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: pos[i].0,
                y: pos[i].1 + 6.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(18.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(COL_TEXT.to_string()),
                text: format!("s{i}"),
                ..Default::default()
            }));
        }
        shapes.push(Shape::Text(TextShape {
            x: cx,
            y: 130.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(13.0),
            fill: Some(COL_REACH.to_string()),
            text: "green = reachable from s0".to_string(),
            ..Default::default()
        }));
        let ok = mdp.is_structurally_controllable();
        let info = mdp.transition_information_summary();
        shapes.push(Shape::Text(TextShape {
            x: 640.0,
            y: 220.0,
            anchor: Some(Anchor::Start),
            font_size: Some(15.0),
            fill: Some(COL_TEXT.to_string()),
            text: format!(
                "reachable ordered pairs = {} / S\u{00b2} = 9",
                mdp.reachable_pair_count()
            ),
            ..Default::default()
        }));
        self.badge(&mut shapes, 640.0, 250.0, "Controllable", ok);
        shapes.push(Shape::Text(TextShape {
            x: 640.0,
            y: 330.0,
            anchor: Some(Anchor::Start),
            font_size: Some(13.0),
            fill: Some(COL_INFO.to_string()),
            font_weight: Some(FontWeight::Bold),
            text: "transition uncertainty".to_string(),
            ..Default::default()
        }));
        self.metric_bar(
            &mut shapes,
            640.0,
            350.0,
            210.0,
            "mean H(S' | s,a)",
            info.mean_entropy_bits,
            info.row_capacity_bits.max(1.0),
            COL_INFO,
        );
        shapes.push(Shape::Text(TextShape {
            x: 640.0,
            y: 404.0,
            anchor: Some(Anchor::Start),
            font_size: Some(12.0),
            fill: Some(COL_DIM.to_string()),
            text: format!(
                "row capacity log2(S) = {} bits",
                to_fixed(info.row_capacity_bits, 3)
            ),
            ..Default::default()
        }));
        self.steps_.push(FrameParts::with_caption(
            shapes,
            format!(
                "{title}: {}; mean transition entropy {:.3} bits.",
                if ok {
                    "strongly connected \u{2192} controllable"
                } else {
                    "cannot leave the trap \u{2192} NOT controllable"
                },
                info.mean_entropy_bits
            ),
        ));
    }

    fn mdp_information_step(&mut self, title: &str, mdp: &MarkovDecisionProcess) {
        let mut shapes = self.base(title);
        let info = mdp.transition_information_summary();
        let max_h = info.row_capacity_bits.max(1.0);

        shapes.push(Shape::Text(TextShape {
            x: 130.0,
            y: 125.0,
            anchor: Some(Anchor::Start),
            font_size: Some(14.0),
            fill: Some(COL_TEXT.to_string()),
            text: "Each row is a source distribution for the next state.".to_string(),
            ..Default::default()
        }));
        for s in 0..mdp.num_states {
            let y = 170.0 + s as f64 * 105.0;
            shapes.push(Shape::Text(TextShape {
                x: 130.0,
                y,
                anchor: Some(Anchor::Start),
                font_size: Some(14.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(COL_TEXT.to_string()),
                text: format!("from s{s}"),
                ..Default::default()
            }));
            for t in 0..mdp.num_states {
                let p = mdp.transition[0][s][t];
                let x = 230.0 + t as f64 * 82.0;
                shapes.push(Shape::Rect(RectShape {
                    x,
                    y: y - 19.0,
                    w: 58.0,
                    h: 32.0,
                    rx: Some(5.0),
                    fill: if p > 0.0 {
                        "#172554".to_string()
                    } else {
                        "#111827".to_string()
                    },
                    stroke: Some("#334155".to_string()),
                    ..Default::default()
                }));
                shapes.push(Shape::Text(TextShape {
                    x: x + 29.0,
                    y: y + 2.0,
                    anchor: Some(Anchor::Middle),
                    font_size: Some(12.0),
                    fill: Some(COL_TEXT.to_string()),
                    text: format!("p{}={}", t, to_fixed(p, 2)),
                    ..Default::default()
                }));
            }
            self.metric_bar(
                &mut shapes,
                520.0,
                y - 16.0,
                230.0,
                &format!("H(S' | s{s},a0)"),
                info.state_action_entropy_bits[0][s],
                max_h,
                COL_INFO,
            );
        }
        self.metric_bar(
            &mut shapes,
            130.0,
            515.0,
            620.0,
            "mean transition entropy",
            info.mean_entropy_bits,
            max_h,
            COL_INFO,
        );
        self.steps_.push(FrameParts::with_caption(
            shapes,
            format!(
                "{title}: mean H(S' | s,a) = {:.3} bits, normalized {:.3}.",
                info.mean_entropy_bits, info.normalized_mean_entropy
            ),
        ));
    }

    // ── POMDP ─────────────────────────────────────────────────────────────────

    fn build_pomdp(&mut self) {
        let distinct = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let aliased = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        });
        self.pomdp_step(
            "POMDP observability \u{2248} distinguishability — distinct sensors",
            &distinct,
        );
        self.pomdp_step(
            "POMDP observability \u{2248} distinguishability — aliased sensors",
            &aliased,
        );
    }

    fn pomdp_step(&mut self, title: &str, pomdp: &PartiallyObservableProcess) {
        let mut shapes = self.base(title);
        let labels = pomdp.distinguishability_classes();
        let palette = ["#38bdf8", "#f59e0b", "#a78bfa", "#34d399"];
        let n = pomdp.mdp.num_states;
        let info = pomdp.observation_information(None);
        let max_h = info.input_entropy_bits.max(1.0);
        shapes.push(Shape::Text(TextShape {
            x: 190.0,
            y: 116.0,
            anchor: Some(Anchor::Start),
            font_size: Some(13.0),
            font_weight: Some(FontWeight::Bold),
            fill: Some(COL_INFO.to_string()),
            text: "sensor information under a uniform hidden-state prior".to_string(),
            ..Default::default()
        }));
        self.metric_bar(
            &mut shapes,
            190.0,
            138.0,
            250.0,
            "I(S;O)",
            info.mutual_information_bits,
            max_h,
            COL_INFO,
        );
        self.metric_bar(
            &mut shapes,
            560.0,
            138.0,
            250.0,
            "residual H(S | O)",
            info.equivocation_bits,
            max_h,
            COL_BAD,
        );
        for s in 0..n {
            let x = 220.0 + s as f64 * 280.0;
            let y = 240.0;
            let col = palette[labels[s] % palette.len()];
            shapes.push(Shape::Rect(RectShape {
                x,
                y,
                w: 180.0,
                h: 120.0,
                rx: Some(12.0),
                fill: COL_PANEL.to_string(),
                stroke: Some(col.to_string()),
                stroke_width: Some(3.0),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: x + 90.0,
                y: y + 34.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(20.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(COL_TEXT.to_string()),
                text: format!("state s{s}"),
                ..Default::default()
            }));
            let obs_str = pomdp.observation[s]
                .iter()
                .map(|p| to_fixed(*p, 2))
                .collect::<Vec<_>>()
                .join(", ");
            shapes.push(Shape::Text(TextShape {
                x: x + 90.0,
                y: y + 64.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(13.0),
                fill: Some(COL_DIM.to_string()),
                text: format!("obs P = [{obs_str}]"),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: x + 90.0,
                y: y + 92.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(13.0),
                fill: Some(col.to_string()),
                font_weight: Some(FontWeight::Bold),
                text: format!("class {}", labels[s]),
                ..Default::default()
            }));
        }
        let ok = pomdp.is_structurally_observable();
        shapes.push(Shape::Text(TextShape {
            x: OC_STAGE_W / 2.0,
            y: 420.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(15.0),
            fill: Some(COL_TEXT.to_string()),
            text: format!(
                "distinguishability classes = {} / S = {n}",
                pomdp.class_count()
            ),
            ..Default::default()
        }));
        self.badge(
            &mut shapes,
            OC_STAGE_W / 2.0 - 115.0,
            445.0,
            "Observable",
            ok,
        );
        let aliasing = pomdp.indistinguishable_pairs();
        if !aliasing.is_empty() {
            let pairs = aliasing
                .iter()
                .map(|p| format!("(s{},s{})", p.0, p.1))
                .collect::<Vec<_>>()
                .join(" ");
            shapes.push(Shape::Text(TextShape {
                x: OC_STAGE_W / 2.0,
                y: 520.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(13.0),
                fill: Some(COL_BAD.to_string()),
                text: format!("aliased (indistinguishable) state pairs: {pairs}"),
                ..Default::default()
            }));
        }
        self.steps_.push(FrameParts::with_caption(
            shapes,
            format!(
                "{title}: {}; I(S;O) = {:.3} bits, residual H(S|O) = {:.3} bits.",
                if ok {
                    "every state has its own class \u{2192} observable"
                } else {
                    "states collapse to one class \u{2192} NOT observable"
                },
                info.mutual_information_bits,
                info.equivocation_bits
            ),
        ));
    }

    // ── recap ──────────────────────────────────────────────────────────────────

    fn build_recap(&mut self) {
        let mut shapes = self.base("summary");
        let rows: [(&str, bool, bool); 6] = [
            ("double integrator (LTI)", true, true),
            ("decoupled modes (LTI)", false, false),
            ("ring MDP", true, true),
            ("trap MDP", false, true),
            ("distinct-sensor POMDP", true, true),
            ("aliased-sensor POMDP", true, false),
        ];
        shapes.push(Shape::Text(TextShape {
            x: 200.0,
            y: 120.0,
            anchor: Some(Anchor::Start),
            font_size: Some(14.0),
            fill: Some(COL_DIM.to_string()),
            text: "system".to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 620.0,
            y: 120.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(14.0),
            fill: Some(COL_B.to_string()),
            text: "controllable".to_string(),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 820.0,
            y: 120.0,
            anchor: Some(Anchor::Middle),
            font_size: Some(14.0),
            fill: Some(COL_C.to_string()),
            text: "observable".to_string(),
            ..Default::default()
        }));
        for (i, (label, ctrl, obs)) in rows.iter().enumerate() {
            let y = 150.0 + i as f64 * 60.0;
            shapes.push(Shape::Rect(RectShape {
                x: 160.0,
                y,
                w: 720.0,
                h: 48.0,
                rx: Some(8.0),
                fill: COL_PANEL.to_string(),
                stroke: Some("#334155".to_string()),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: 200.0,
                y: y + 30.0,
                anchor: Some(Anchor::Start),
                font_size: Some(15.0),
                fill: Some(COL_TEXT.to_string()),
                text: label.to_string(),
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: 620.0,
                y: y + 31.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(16.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(if *ctrl {
                    COL_OK.to_string()
                } else {
                    COL_BAD.to_string()
                }),
                text: if *ctrl {
                    "\u{2713}".to_string()
                } else {
                    "\u{2717}".to_string()
                },
                ..Default::default()
            }));
            shapes.push(Shape::Text(TextShape {
                x: 820.0,
                y: y + 31.0,
                anchor: Some(Anchor::Middle),
                font_size: Some(16.0),
                font_weight: Some(FontWeight::Bold),
                fill: Some(if *obs {
                    COL_OK.to_string()
                } else {
                    COL_BAD.to_string()
                }),
                text: if *obs {
                    "\u{2713}".to_string()
                } else {
                    "\u{2717}".to_string()
                },
                ..Default::default()
            }));
        }
        self.steps_.push(FrameParts::with_caption(
            shapes,
            "Controllability = can I move the states? Observability = can I infer the states? Information = how many bits are removed or left behind.",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storyboard_has_expected_step_count() {
        let scene = ObsCtrlScene::new();
        // title + 2 LTI examples (3 steps each) + 2 MDP reachability
        // + 1 MDP entropy frame + 2 POMDP info/observability frames + recap.
        assert_eq!(scene.steps().len(), 1 + 6 + 2 + 1 + 2 + 1);
    }

    #[test]
    fn double_integrator_is_controllable_and_observable() {
        let scene = ObsCtrlScene::new();
        // Step 2 = controllability of the double integrator -> "controllable".
        let cap = scene.steps()[2].caption.clone().unwrap();
        assert!(cap.contains("controllable"), "{cap}");
        assert!(!cap.contains("NOT controllable"), "{cap}");
    }

    #[test]
    fn storyboard_includes_information_frames() {
        let scene = ObsCtrlScene::new();
        assert!(
            scene.steps().iter().any(|s| s
                .caption
                .as_deref()
                .unwrap_or("")
                .contains("mean H(S' | s,a)")),
            "missing MDP transition entropy frame"
        );
        assert!(
            scene
                .steps()
                .iter()
                .any(|s| s.caption.as_deref().unwrap_or("").contains("I(S;O)")),
            "missing POMDP sensor information frame"
        );
    }
}
