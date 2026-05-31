//! Turn a solved spec + a rollout into a uniform [`RunArtifact`] the platform
//! can render. Both MDP and POMDP produce an animated sim (SVG stage + numeric
//! timeline) plus a results document, so a decision process visualizes with the
//! same quality as the hybrid demos — predictions (trajectory, belief, return)
//! and control (policy, value) in one artifact.

use serde_json::{json, Value};

use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

use super::rollout::EpisodeTrace;
use super::solve::{MdpSolution, PomdpSolution};
use super::spec::{MdpSpec, PomdpSpec};

const MDP_NODE_R: f64 = 23.0;
const MDP_UPDATE_HIGHLIGHT_FRAMES: usize = 3;

fn entropy(b: &[f64]) -> f64 {
    let mut h = 0.0;
    for &w in b {
        if w > 0.0 {
            h -= w * w.ln();
        }
    }
    h
}

/// Node layout on a circle for the MDP state-transition graph.
fn ring_positions(n: usize, cx: f64, cy: f64, r: f64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let theta = -std::f64::consts::FRAC_PI_2
                + 2.0 * std::f64::consts::PI * i as f64 / n.max(1) as f64;
            (cx + r * theta.cos(), cy + r * theta.sin())
        })
        .collect()
}

#[derive(Clone, Debug)]
struct MdpLearningStats {
    counts: Vec<Vec<Vec<usize>>>,
    reward_sums: Vec<Vec<Vec<f64>>>,
    last_update: Vec<Vec<Vec<Option<usize>>>>,
    total_updates: usize,
}

impl MdpLearningStats {
    fn new(spec: &MdpSpec) -> Self {
        let counts = spec
            .transitions
            .iter()
            .map(|actions| {
                actions
                    .iter()
                    .map(|outs| vec![0; outs.len()])
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect::<Vec<Vec<Vec<usize>>>>();
        let reward_sums = spec
            .transitions
            .iter()
            .map(|actions| {
                actions
                    .iter()
                    .map(|outs| vec![0.0; outs.len()])
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();
        let last_update = spec
            .transitions
            .iter()
            .map(|actions| {
                actions
                    .iter()
                    .map(|outs| vec![None; outs.len()])
                    .collect::<Vec<Vec<Option<usize>>>>()
            })
            .collect::<Vec<Vec<Vec<Option<usize>>>>>();
        Self {
            counts,
            reward_sums,
            last_update,
            total_updates: 0,
        }
    }

    fn observe(
        &mut self,
        spec: &MdpSpec,
        step: usize,
        state: usize,
        action: usize,
        next: usize,
        reward: f64,
    ) -> Option<usize> {
        let outcomes = spec.transitions.get(state)?.get(action)?;
        let outcome_idx = outcomes
            .iter()
            .position(|o| o.next == next && (o.reward - reward).abs() < 1e-9)
            .or_else(|| outcomes.iter().position(|o| o.next == next))?;
        self.counts[state][action][outcome_idx] += 1;
        self.reward_sums[state][action][outcome_idx] += reward;
        self.last_update[state][action][outcome_idx] = Some(step);
        self.total_updates += 1;
        Some(outcome_idx)
    }

    fn total_count(&self, state: usize, action: usize) -> usize {
        self.counts
            .get(state)
            .and_then(|actions| actions.get(action))
            .map(|outs| outs.iter().sum())
            .unwrap_or(0)
    }

    fn count(&self, state: usize, action: usize, outcome: usize) -> usize {
        self.counts
            .get(state)
            .and_then(|actions| actions.get(action))
            .and_then(|outs| outs.get(outcome))
            .copied()
            .unwrap_or(0)
    }

    fn learned_prob(&self, state: usize, action: usize, outcome: usize) -> f64 {
        let Some(outcomes) = self
            .counts
            .get(state)
            .and_then(|actions| actions.get(action))
        else {
            return 0.0;
        };
        if outcomes.is_empty() {
            return 0.0;
        }
        let total: usize = outcomes.iter().sum();
        (outcomes[outcome] as f64 + 1.0) / (total as f64 + outcomes.len() as f64)
    }

    fn learned_reward(&self, state: usize, action: usize, outcome: usize) -> f64 {
        let n = self.count(state, action, outcome);
        if n == 0 {
            0.0
        } else {
            self.reward_sums[state][action][outcome] / n as f64
        }
    }

    fn recently_updated(&self, state: usize, action: usize, outcome: usize, frame: usize) -> bool {
        self.last_update
            .get(state)
            .and_then(|actions| actions.get(action))
            .and_then(|outs| outs.get(outcome))
            .and_then(|&last| last)
            .is_some_and(|last| frame >= last && frame - last <= MDP_UPDATE_HIGHLIGHT_FRAMES)
    }
}

#[derive(Clone, Copy, Debug)]
struct MdpObservedUpdate {
    state: usize,
    action: usize,
    outcome: usize,
    next: usize,
    reward: f64,
}

fn policy_action(policy: &[i32], s: usize) -> Option<usize> {
    let a = *policy.get(s)?;
    if a < 0 {
        None
    } else {
        Some(a as usize)
    }
}

fn line_between_nodes(from: (f64, f64), to: (f64, f64)) -> (f64, f64, f64, f64) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 1e-9 {
        return (from.0, from.1, to.0, to.1);
    }
    let ux = dx / len;
    let uy = dy / len;
    (
        from.0 + ux * MDP_NODE_R,
        from.1 + uy * MDP_NODE_R,
        to.0 - ux * MDP_NODE_R,
        to.1 - uy * MDP_NODE_R,
    )
}

fn edge_label_point(from: (f64, f64), to: (f64, f64), outcome_idx: usize) -> (f64, f64) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 1e-9 {
        return (from.0 + 42.0, from.1 - 42.0 - 14.0 * outcome_idx as f64);
    }
    let ux = dx / len;
    let uy = dy / len;
    let px = -uy;
    let py = ux;
    let offset = 14.0 + outcome_idx as f64 * 6.0;
    (
        from.0 + dx * 0.56 + px * offset,
        from.1 + dy * 0.56 + py * offset,
    )
}

fn self_loop_path(x: f64, y: f64) -> String {
    format!(
        "M {:.2} {:.2} C {:.2} {:.2}, {:.2} {:.2}, {:.2} {:.2}",
        x + 18.0,
        y - 18.0,
        x + 72.0,
        y - 74.0,
        x - 72.0,
        y - 74.0,
        x - 18.0,
        y - 18.0
    )
}

fn push_text(shapes: &mut Vec<Value>, x: f64, y: f64, text: String, font_size: f64, fill: &str) {
    shapes.push(json!({
        "kind": "text", "x": x, "y": y, "text": text,
        "anchor": "middle", "fontSize": font_size, "fill": fill
    }));
}

fn push_left_text(
    shapes: &mut Vec<Value>,
    x: f64,
    y: f64,
    text: String,
    font_size: f64,
    fill: &str,
    weight: Option<&str>,
) {
    let mut shape = json!({
        "kind": "text", "x": x, "y": y, "text": text,
        "fontSize": font_size, "fill": fill
    });
    if let Value::Object(map) = &mut shape {
        if let Some(weight) = weight {
            map.insert("fontWeight".to_string(), json!(weight));
        }
    }
    shapes.push(shape);
}

fn push_mdp_learning_edges(
    shapes: &mut Vec<Value>,
    spec: &MdpSpec,
    sol: &MdpSolution,
    stats: &MdpLearningStats,
    pos: &[(f64, f64)],
    frame: usize,
) {
    for s in 0..spec.num_states {
        let Some(a) = policy_action(&sol.policy, s) else {
            continue;
        };
        let Some(outcomes) = spec.transitions.get(s).and_then(|actions| actions.get(a)) else {
            continue;
        };
        for (i, outcome) in outcomes.iter().enumerate() {
            let recent = stats.recently_updated(s, a, i, frame);
            let stroke = if recent { "#dc2626" } else { "#64748b" };
            let text_fill = if recent { "#b91c1c" } else { "#334155" };
            let stroke_width = if recent { 3.0 } else { 1.35 };
            let dash = if stats.count(s, a, i) == 0 {
                Some("4,4")
            } else {
                None
            };
            let from = pos[s];
            let to = pos[outcome.next];
            if s == outcome.next {
                shapes.push(json!({
                    "kind": "path", "d": self_loop_path(from.0, from.1),
                    "stroke": stroke, "strokeWidth": stroke_width, "fill": "none",
                    "opacity": if recent { 0.95 } else { 0.55 },
                    "dasharray": dash
                }));
            } else {
                let (x1, y1, x2, y2) = line_between_nodes(from, to);
                shapes.push(json!({
                    "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "stroke": stroke, "strokeWidth": stroke_width,
                    "opacity": if recent { 0.95 } else { 0.58 },
                    "dasharray": dash
                }));
            }
            let (lx, ly) = edge_label_point(from, to, i);
            shapes.push(json!({
                "kind": "rect", "x": lx - 42.0, "y": ly - 11.0, "w": 84.0, "h": 26.0,
                "rx": 4.0, "fill": if recent { "#fee2e2" } else { "#ffffff" },
                "stroke": if recent { "#ef4444" } else { "#cbd5e1" },
                "strokeWidth": if recent { 1.4 } else { 0.8 },
                "opacity": 0.94
            }));
            push_text(
                shapes,
                lx,
                ly,
                format!(
                    "p={:.2} n={}",
                    stats.learned_prob(s, a, i),
                    stats.count(s, a, i)
                ),
                9.0,
                text_fill,
            );
            push_text(
                shapes,
                lx,
                ly + 11.0,
                format!("r={:.1}", stats.learned_reward(s, a, i)),
                9.0,
                text_fill,
            );
        }
    }
}

fn push_mdp_learning_panel(
    shapes: &mut Vec<Value>,
    spec: &MdpSpec,
    sol: &MdpSolution,
    stats: &MdpLearningStats,
    cur: usize,
    action: Option<usize>,
    update: Option<MdpObservedUpdate>,
    frame: usize,
) {
    let panel_x = 400.0;
    let panel_y = 34.0;
    shapes.push(json!({
        "kind": "rect", "x": panel_x, "y": panel_y, "w": 318.0, "h": 306.0,
        "rx": 8.0, "fill": "#f8fafc", "stroke": "#cbd5e1", "strokeWidth": 1.0
    }));
    push_left_text(
        shapes,
        panel_x + 16.0,
        panel_y + 28.0,
        "Learned transition model".to_string(),
        14.0,
        "#0f172a",
        Some("bold"),
    );
    push_left_text(
        shapes,
        panel_x + 16.0,
        panel_y + 48.0,
        "red = p/reward updated recently".to_string(),
        10.0,
        "#b91c1c",
        None,
    );

    let active_action = action.or_else(|| policy_action(&sol.policy, cur));
    let state_line = match active_action {
        Some(a) => format!(
            "state {} / action {}",
            spec.state_label(cur),
            spec.action_label(a)
        ),
        None => format!("state {} / terminal", spec.state_label(cur)),
    };
    push_left_text(
        shapes,
        panel_x + 16.0,
        panel_y + 78.0,
        state_line,
        12.0,
        "#334155",
        Some("bold"),
    );

    if let Some(u) = update {
        push_left_text(
            shapes,
            panel_x + 16.0,
            panel_y + 101.0,
            format!(
                "sample: {} --{}--> {}  r={:.1}",
                spec.state_label(u.state),
                spec.action_label(u.action),
                spec.state_label(u.next),
                u.reward
            ),
            11.0,
            "#dc2626",
            Some("bold"),
        );
    } else {
        push_left_text(
            shapes,
            panel_x + 16.0,
            panel_y + 101.0,
            format!("updates observed: {}", stats.total_updates),
            11.0,
            "#475569",
            None,
        );
    }

    if let Some(a) = active_action {
        if let Some(outcomes) = spec.transitions.get(cur).and_then(|actions| actions.get(a)) {
            let total = stats.total_count(cur, a);
            push_left_text(
                shapes,
                panel_x + 16.0,
                panel_y + 130.0,
                format!("outcomes sampled from this action: {total}"),
                10.0,
                "#64748b",
                None,
            );
            for (i, outcome) in outcomes.iter().enumerate().take(6) {
                let y = panel_y + 157.0 + i as f64 * 24.0;
                let recent = stats.recently_updated(cur, a, i, frame);
                let fill = if recent { "#dc2626" } else { "#0f172a" };
                let row_bg = if recent { "#fee2e2" } else { "#ffffff" };
                shapes.push(json!({
                    "kind": "rect", "x": panel_x + 12.0, "y": y - 15.0,
                    "w": 294.0, "h": 19.0, "rx": 4.0,
                    "fill": row_bg, "stroke": if recent { "#fecaca" } else { "#e2e8f0" },
                    "strokeWidth": 0.8
                }));
                push_left_text(
                    shapes,
                    panel_x + 20.0,
                    y,
                    format!(
                        "-> {}   p={:.2}   r={:.1}   n={}",
                        spec.state_label(outcome.next),
                        stats.learned_prob(cur, a, i),
                        stats.learned_reward(cur, a, i),
                        stats.count(cur, a, i)
                    ),
                    10.0,
                    fill,
                    if recent { Some("bold") } else { None },
                );
            }
        }
    }

    push_left_text(
        shapes,
        panel_x + 16.0,
        panel_y + 286.0,
        "Dirichlet(1) p prior; reward is sampled mean.".to_string(),
        10.0,
        "#64748b",
        None,
    );
}

/// Build the transition-graph document (nodes + policy edges) for the results
/// payload.
fn mdp_graph_doc(spec: &MdpSpec, sol: &MdpSolution) -> Value {
    let nodes: Vec<Value> = (0..spec.num_states)
        .map(|s| {
            json!({
                "id": s,
                "label": spec.state_label(s),
                "value": sol.value[s],
                "policyAction": sol.policy[s],
                "policyLabel": if sol.policy[s] >= 0 { spec.action_label(sol.policy[s] as usize) } else { "—".to_string() },
                "terminal": spec.is_terminal(s),
            })
        })
        .collect();
    let mut edges = Vec::new();
    for s in 0..spec.num_states {
        let a = sol.policy[s];
        if a < 0 {
            continue;
        }
        if let Some(outs) = spec.transitions.get(s).and_then(|av| av.get(a as usize)) {
            for o in outs {
                edges.push(json!({
                    "from": s, "to": o.next, "action": a, "prob": o.prob, "reward": o.reward,
                }));
            }
        }
    }
    json!({ "nodes": nodes, "edges": edges })
}

/// Build an animated artifact for an MDP rollout: a state-transition graph with
/// the current state highlighted, plus a value/return/reward timeline.
pub fn mdp_artifact(
    spec: &MdpSpec,
    sol: &MdpSolution,
    trace: &EpisodeTrace,
    title: &str,
    description: &str,
) -> RunArtifact {
    let pos = ring_positions(spec.num_states, 210.0, 190.0, 125.0);
    let mut frames = Vec::with_capacity(trace.states.len());
    let mut stats = MdpLearningStats::new(spec);

    for (k, &cur) in trace.states.iter().enumerate() {
        let action = trace.actions.get(k).copied();
        let reward = trace.rewards.get(k).copied().unwrap_or(0.0);
        let next_state = trace.states.get(k + 1).copied();
        let update = match (action, next_state) {
            (Some(a), Some(next)) => {
                stats
                    .observe(spec, k, cur, a, next, reward)
                    .map(|outcome| MdpObservedUpdate {
                        state: cur,
                        action: a,
                        outcome,
                        next,
                        reward,
                    })
            }
            _ => None,
        };
        let mut shapes = Vec::new();
        push_mdp_learning_edges(&mut shapes, spec, sol, &stats, &pos, k);
        // State nodes.
        for s in 0..spec.num_states {
            let (x, y) = pos[s];
            let is_cur = s == cur;
            let is_update_target = update.is_some_and(|u| u.next == s);
            let fill = if is_cur {
                "#2563eb"
            } else if spec.is_terminal(s) {
                "#fca5a5"
            } else {
                "#e2e8f0"
            };
            shapes.push(json!({
                "kind": "circle", "x": x, "y": y, "r": 22.0, "fill": fill,
                "stroke": if is_update_target { "#dc2626" } else if is_cur { "#1d4ed8" } else { "#94a3b8" },
                "strokeWidth": if is_update_target || is_cur { 3.0 } else { 1.0 }
            }));
            shapes.push(json!({
                "kind": "text", "x": x, "y": y + 4.0, "text": spec.state_label(s),
                "anchor": "middle", "fontSize": 11.0,
                "fill": if is_cur { "#ffffff" } else { "#0f172a" }
            }));
        }
        push_mdp_learning_panel(&mut shapes, spec, sol, &stats, cur, action, update, k);
        let ret = if k == 0 {
            0.0
        } else {
            trace.returns.get(k - 1).copied().unwrap_or(0.0)
        };
        let caption = match action {
            Some(a) => format!(
                "t={k} · state {} · action {} · reward {:.2}",
                spec.state_label(cur),
                spec.action_label(a),
                reward
            ),
            None => format!("t={k} · state {} · (done)", spec.state_label(cur)),
        };
        let update_label = update.map(|u| {
            format!(
                "{}|{}|{}",
                spec.state_label(u.state),
                spec.action_label(u.action),
                spec.state_label(u.next)
            )
        });
        frames.push(json!({
            "t": k,
            "state": cur as f64,
            "value": sol.value[cur],
            "reward": reward,
            "return": ret,
            "modelUpdates": stats.total_updates as f64,
            "updatedOutcome": update.map(|u| u.outcome as f64).unwrap_or(-1.0),
            "shapes": shapes,
            "caption": caption,
            "updatedEdge": update_label,
        }));
    }

    let results = json!({
        "kind": "mdp",
        "method": "value-iteration",
        "discount": sol.discount,
        "iterations": sol.iterations,
        "finalDelta": sol.final_delta,
        "value": sol.value,
        "policy": sol.policy,
        "policyLabels": (0..spec.num_states).map(|s| if sol.policy[s] >= 0 { spec.action_label(sol.policy[s] as usize) } else { "—".to_string() }).collect::<Vec<_>>(),
        "q": sol.q,
        "transitionGraph": mdp_graph_doc(spec, sol),
        "learningOverlay": {
            "transitionEstimator": "Dirichlet(1) posterior mean per displayed state-action outcome",
            "rewardEstimator": "sample mean per displayed state-action outcome",
            "highlightFrames": MDP_UPDATE_HIGHLIGHT_FRAMES,
        },
        "rollout": trace,
    });

    let summary = format!(
        "Solved MDP in {} iterations (γ={:.3}); rollout return {:.2} over {} steps.",
        sol.iterations,
        sol.discount,
        trace.discounted_return,
        trace.actions.len()
    );

    RunArtifact::sim(
        "mdp",
        title,
        description,
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 4.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &["all", "value", "return", "reward", "modelUpdates"],
                "all",
                Some("metric"),
            ),
        ],
        &summary,
    )
}

/// Build an animated artifact for a POMDP rollout: belief bars over hidden
/// states (true state outlined), plus an entropy/return timeline.
pub fn pomdp_artifact(
    spec: &PomdpSpec,
    sol: &PomdpSolution,
    trace: &EpisodeTrace,
    method: &str,
    title: &str,
    description: &str,
) -> RunArtifact {
    let ns = spec.num_states;
    let base_y = 210.0;
    let max_h = 150.0;
    let bar_w = 46.0;
    let gap = 24.0;
    let left = 60.0;
    let mut frames = Vec::with_capacity(trace.beliefs.len());

    for (k, belief) in trace.beliefs.iter().enumerate() {
        let mut shapes = Vec::new();
        // Floor line.
        shapes.push(json!({
            "kind": "line", "x1": left - 20.0, "y1": base_y, "x2": left + ns as f64 * (bar_w + gap), "y2": base_y,
            "stroke": "#475569", "strokeWidth": 2.0
        }));
        let true_state = trace.states.get(k).copied();
        for s in 0..ns {
            let x = left + s as f64 * (bar_w + gap);
            let h = (belief[s] * max_h).max(0.5);
            let is_true = Some(s) == true_state;
            shapes.push(json!({
                "kind": "rect", "x": x, "y": base_y - h, "w": bar_w, "h": h, "rx": 4.0,
                "fill": if is_true { "#2563eb" } else { "#93c5fd" },
                "stroke": if is_true { "#dc2626" } else { "#60a5fa" },
                "strokeWidth": if is_true { 3.0 } else { 1.0 }
            }));
            shapes.push(json!({
                "kind": "text", "x": x + bar_w / 2.0, "y": base_y + 16.0, "text": spec.state_label(s),
                "anchor": "middle", "fontSize": 11.0, "fill": "#0f172a"
            }));
            shapes.push(json!({
                "kind": "text", "x": x + bar_w / 2.0, "y": base_y - h - 6.0, "text": format!("{:.2}", belief[s]),
                "anchor": "middle", "fontSize": 10.0, "fill": "#334155"
            }));
        }
        let action = trace.actions.get(k).map(|&a| spec.action_label(a));
        let obs = trace
            .observations
            .get(k)
            .map(|&o| spec.observation_label(o));
        let caption = match (&action, &obs) {
            (Some(a), Some(o)) => {
                format!("t={k} · belief over hidden state · action {a} → observe {o}")
            }
            _ => format!("t={k} · final belief"),
        };
        let ret = if k == 0 {
            0.0
        } else {
            trace.returns.get(k - 1).copied().unwrap_or(0.0)
        };
        let mut frame = json!({
            "t": k,
            "entropy": entropy(belief),
            "return": ret,
            "trueState": true_state.map(|s| s as f64).unwrap_or(-1.0),
            "shapes": shapes,
            "caption": caption,
        });
        // belief of each state as its own numeric series.
        if let Value::Object(map) = &mut frame {
            for s in 0..ns {
                map.insert(format!("belief.{}", spec.state_label(s)), json!(belief[s]));
            }
        }
        frames.push(frame);
    }

    let results = json!({
        "kind": "pomdp",
        "method": method,
        "discount": sol.discount,
        "underlyingValue": sol.underlying_value,
        "underlyingPolicy": sol.underlying_policy,
        "q": sol.q,
        "finalBelief": trace.beliefs.last(),
        "discountedReturn": trace.discounted_return,
        "rollout": trace,
    });

    let summary = format!(
        "POMDP solved by {method} (γ={:.3}); rollout return {:.2} over {} steps, final entropy {:.3}.",
        sol.discount,
        trace.discounted_return,
        trace.actions.len(),
        trace.beliefs.last().map(|b| entropy(b)).unwrap_or(0.0)
    );

    RunArtifact::sim(
        "pomdp",
        title,
        description,
        frames,
        results,
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 3.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &["all", "entropy", "return"],
                "all",
                Some("metric"),
            ),
        ],
        &summary,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::demos::machine_maintenance_mdp;
    use crate::des::decision::rollout::rollout_mdp;
    use crate::des::decision::solve::{solve_mdp, MdpMethod};

    #[test]
    fn mdp_artifact_shows_learning_updates() {
        let spec = machine_maintenance_mdp();
        let sol = solve_mdp(&spec, MdpMethod::ValueIteration).unwrap();
        let trace = rollout_mdp(&spec, &sol.policy, 0, 8, 7);
        let art = mdp_artifact(&spec, &sol, &trace, "MDP", "test");

        assert_eq!(art.kind, "mdp");
        assert_eq!(
            art.results["learningOverlay"]["highlightFrames"].as_u64(),
            Some(MDP_UPDATE_HIGHLIGHT_FRAMES as u64)
        );
        assert!(art
            .frames
            .iter()
            .any(|f| f["modelUpdates"].as_f64().unwrap_or(0.0) > 0.0));
        assert!(art.frames.iter().any(|f| f["updatedEdge"].is_string()));

        let html = art.to_player_html();
        assert!(html.contains("Learned transition model"));
        assert!(html.contains("red = p/reward updated recently"));
        assert!(html.contains("\"modelUpdates\""));
        assert!(html.contains("#dc2626"));
    }
}
