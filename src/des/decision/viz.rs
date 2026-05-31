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
            let theta = -std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * i as f64 / n.max(1) as f64;
            (cx + r * theta.cos(), cy + r * theta.sin())
        })
        .collect()
}

/// The most-likely next state under the greedy policy in state `s` (for drawing
/// the policy's transition edges). `None` if terminal / no action.
fn policy_next(spec: &MdpSpec, policy: &[i32], s: usize) -> Option<usize> {
    let a = *policy.get(s)?;
    if a < 0 {
        return None;
    }
    let outs = spec.transitions.get(s)?.get(a as usize)?;
    outs.iter()
        .max_by(|x, y| x.prob.partial_cmp(&y.prob).unwrap_or(std::cmp::Ordering::Equal))
        .map(|o| o.next)
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
    let pos = ring_positions(spec.num_states, 210.0, 170.0, 120.0);
    let mut frames = Vec::with_capacity(trace.states.len());

    for (k, &cur) in trace.states.iter().enumerate() {
        let mut shapes = Vec::new();
        // Policy edges (faint), drawn under the nodes.
        for s in 0..spec.num_states {
            if let Some(nx) = policy_next(spec, &sol.policy, s) {
                let (x1, y1) = pos[s];
                let (x2, y2) = pos[nx];
                shapes.push(json!({
                    "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "stroke": "#cbd5e1", "strokeWidth": 1.5, "opacity": 0.7
                }));
            }
        }
        // State nodes.
        for s in 0..spec.num_states {
            let (x, y) = pos[s];
            let is_cur = s == cur;
            let fill = if is_cur {
                "#2563eb"
            } else if spec.is_terminal(s) {
                "#fca5a5"
            } else {
                "#e2e8f0"
            };
            shapes.push(json!({
                "kind": "circle", "x": x, "y": y, "r": 22.0, "fill": fill,
                "stroke": if is_cur { "#1d4ed8" } else { "#94a3b8" }, "strokeWidth": if is_cur { 3.0 } else { 1.0 }
            }));
            shapes.push(json!({
                "kind": "text", "x": x, "y": y + 4.0, "text": spec.state_label(s),
                "anchor": "middle", "fontSize": 11.0,
                "fill": if is_cur { "#ffffff" } else { "#0f172a" }
            }));
        }
        let action = trace.actions.get(k).copied();
        let reward = trace.rewards.get(k).copied().unwrap_or(0.0);
        let ret = if k == 0 { 0.0 } else { trace.returns.get(k - 1).copied().unwrap_or(0.0) };
        let caption = match action {
            Some(a) => format!(
                "t={k} · state {} · action {} · reward {:.2}",
                spec.state_label(cur),
                spec.action_label(a),
                reward
            ),
            None => format!("t={k} · state {} · (done)", spec.state_label(cur)),
        };
        frames.push(json!({
            "t": k,
            "state": cur as f64,
            "value": sol.value[cur],
            "reward": reward,
            "return": ret,
            "shapes": shapes,
            "caption": caption,
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
                &["all", "value", "return", "reward"],
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
        let obs = trace.observations.get(k).map(|&o| spec.observation_label(o));
        let caption = match (&action, &obs) {
            (Some(a), Some(o)) => format!(
                "t={k} · belief over hidden state · action {a} → observe {o}"
            ),
            _ => format!("t={k} · final belief"),
        };
        let ret = if k == 0 { 0.0 } else { trace.returns.get(k - 1).copied().unwrap_or(0.0) };
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
