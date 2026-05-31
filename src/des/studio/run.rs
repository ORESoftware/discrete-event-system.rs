//! The **dataflow executive**: run a compiled flat block graph forward in time
//! and render it as an animated wiring diagram (blocks + wires + live values).
//!
//! This is one runtime executive (acyclic signal dataflow over Layer-2 cells),
//! a peer of the DES run-loop and the hybrid signal-flow executive. Each step it
//! evaluates blocks in topological order, threading scalars along the wires, and
//! records a frame.

use serde_json::{json, Value};

use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

use super::graph::CompiledStudio;

/// The result of running a studio graph: per-block signal series + sim frames.
pub struct StudioRun {
    pub steps: usize,
    pub dt: f64,
    pub node_ids: Vec<String>,
    /// `node_series[i]` is block `i`'s primary signal over time (its output port
    /// 0, or — for a sink — the value it received).
    pub node_series: Vec<Vec<f64>>,
    pub frames: Vec<Value>,
}

impl StudioRun {
    pub fn series(&self, id: &str) -> Option<&Vec<f64>> {
        self.node_ids.iter().position(|n| n == id).map(|i| &self.node_series[i])
    }

    /// Final recorded value of a block's primary signal.
    pub fn final_value(&self, id: &str) -> Option<f64> {
        self.series(id).and_then(|s| s.last().copied())
    }
}

fn in_port_xy(x: f64, y: f64, h: f64, p: usize, n: usize) -> (f64, f64) {
    (x, y + h * (p as f64 + 1.0) / (n as f64 + 1.0))
}
fn out_port_xy(x: f64, y: f64, w: f64, h: f64, p: usize, n: usize) -> (f64, f64) {
    (x + w, y + h * (p as f64 + 1.0) / (n as f64 + 1.0))
}

/// Run the graph for `steps` steps of size `dt`.
pub fn run(compiled: &mut CompiledStudio, steps: usize, dt: f64) -> StudioRun {
    let n = compiled.nodes.len();
    for node in compiled.nodes.iter_mut() {
        node.cell.reset();
    }

    let node_ids: Vec<String> = compiled.nodes.iter().map(|nd| nd.id.clone()).collect();
    let mut node_series: Vec<Vec<f64>> = vec![Vec::with_capacity(steps); n];
    let mut frames: Vec<Value> = Vec::with_capacity(steps);

    for k in 0..steps {
        let t = k as f64 * dt;
        let mut outs: Vec<Vec<f64>> = vec![Vec::new(); n];

        // Evaluate blocks in topological order, threading wired inputs.
        let order = compiled.order.clone();
        for idx in order {
            let n_in = compiled.nodes[idx].n_in();
            let mut inputs = vec![0.0; n_in];
            for (p, slot) in inputs.iter_mut().enumerate() {
                if let Some(&(sn, sp)) = compiled.driver.get(&(idx, p)) {
                    *slot = outs[sn].get(sp).copied().unwrap_or(0.0);
                }
            }
            outs[idx] = compiled.nodes[idx].cell.step(t, &inputs);
        }

        // Record each block's primary signal (output 0, or received input for sinks).
        for idx in 0..n {
            let v = if !outs[idx].is_empty() {
                outs[idx][0]
            } else {
                compiled
                    .driver
                    .get(&(idx, 0))
                    .map(|&(sn, sp)| outs[sn].get(sp).copied().unwrap_or(0.0))
                    .unwrap_or(0.0)
            };
            node_series[idx].push(v);
        }

        frames.push(build_frame(compiled, t, k, &outs, &node_series));
    }

    StudioRun { steps, dt, node_ids, node_series, frames }
}

fn build_frame(
    compiled: &CompiledStudio,
    t: f64,
    k: usize,
    outs: &[Vec<f64>],
    node_series: &[Vec<f64>],
) -> Value {
    let mut shapes: Vec<Value> = Vec::new();

    // Wires (under the blocks).
    for w in compiled.wires() {
        let src = &compiled.nodes()[w.from_node];
        let dst = &compiled.nodes()[w.to_node];
        let (x1, y1) = out_port_xy(src.x, src.y, src.w, src.h, w.from_port, src.n_out());
        let (x2, y2) = in_port_xy(dst.x, dst.y, dst.h, w.to_port, dst.n_in());
        let v = outs[w.from_node].get(w.from_port).copied().unwrap_or(0.0);
        let active = v.abs() > 1e-9;
        shapes.push(json!({
            "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "stroke": if active { "#2563eb" } else { "#cbd5e1" },
            "strokeWidth": if active { 2.5 } else { 1.5 }
        }));
    }

    // Blocks.
    for (idx, node) in compiled.nodes().iter().enumerate() {
        let fill = match node.role.as_str() {
            "source" => "#dcfce7",
            "sink" => "#fee2e2",
            _ => "#eef2ff",
        };
        shapes.push(json!({
            "kind": "rect", "x": node.x, "y": node.y, "w": node.w, "h": node.h, "rx": 8.0,
            "fill": fill, "stroke": "#64748b", "strokeWidth": 1.5
        }));
        // Title + the Layer-2 elements stacked inside the block.
        shapes.push(json!({
            "kind": "text", "x": node.x + node.w / 2.0, "y": node.y + 16.0,
            "text": node.label, "anchor": "middle", "fontSize": 12.0, "fill": "#0f172a", "fontWeight": "bold"
        }));
        shapes.push(json!({
            "kind": "text", "x": node.x + node.w / 2.0, "y": node.y + 32.0,
            "text": node.cell.element_names().join(" ▸ "), "anchor": "middle", "fontSize": 9.5, "fill": "#475569"
        }));
        // Live primary value.
        let v = node_series[idx].last().copied().unwrap_or(0.0);
        shapes.push(json!({
            "kind": "text", "x": node.x + node.w / 2.0, "y": node.y + node.h - 8.0,
            "text": format!("{v:.3}"), "anchor": "middle", "fontSize": 11.0, "fill": "#1d4ed8"
        }));
        // Port markers.
        for p in 0..node.n_in() {
            let (px, py) = in_port_xy(node.x, node.y, node.h, p, node.n_in());
            shapes.push(json!({"kind":"circle","x":px,"y":py,"r":3.5,"fill":"#475569"}));
        }
        for p in 0..node.n_out() {
            let (px, py) = out_port_xy(node.x, node.y, node.w, node.h, p, node.n_out());
            shapes.push(json!({"kind":"circle","x":px,"y":py,"r":3.5,"fill":"#475569"}));
        }
    }

    let mut frame = json!({ "t": t, "step": k as f64, "shapes": shapes, "caption": format!("t={t:.2}s") });
    if let Value::Object(map) = &mut frame {
        for (idx, node) in compiled.nodes().iter().enumerate() {
            map.insert(node.id.clone(), json!(node_series[idx].last().copied().unwrap_or(0.0)));
        }
    }
    frame
}

impl StudioRun {
    /// Render this run as a uniform [`RunArtifact`] (animated wiring diagram).
    pub fn to_artifact(&self, kind: &str, title: &str, description: &str, blocks: Value) -> RunArtifact {
        let results = json!({
            "kind": kind,
            "steps": self.steps,
            "dt": self.dt,
            "blocks": blocks,
            "finalSignals": self.node_ids.iter().enumerate()
                .map(|(i, id)| json!({ "block": id, "value": self.node_series[i].last().copied().unwrap_or(0.0) }))
                .collect::<Vec<_>>(),
        });
        let summary = format!(
            "Ran {} blocks for {} steps (dt={}); flat graph, no nested blocks.",
            self.node_ids.len(),
            self.steps,
            self.dt
        );
        RunArtifact::sim(
            kind,
            title,
            description,
            self.frames.clone(),
            results,
            vec![UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 12.0)],
            &summary,
        )
    }
}
