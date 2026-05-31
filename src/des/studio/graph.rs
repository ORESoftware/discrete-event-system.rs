//! Layer 1 — the **visual block graph**: a flat set of blocks wired by typed
//! scalar ports.
//!
//! Two invariants encode the architecture's constraints:
//!
//! * **Blocks do not nest.** A [`VisualNode`] owns a [`RuntimeCell`] (Layer-2
//!   only) — there is no variant that holds another node or graph — so visual
//!   blocks are always visually separate. Composition happens *inside* a block
//!   (a cell may stack several Layer-2 ops), never by nesting blocks.
//! * **Every input port is driven exactly once**, there are no cycles, and a
//!   block's port counts match its cell. Violations are reported as a
//!   [`StudioError`] at [`StudioGraph::build`] time.

use std::collections::HashMap;

use super::cell::RuntimeCell;

/// Construction / validation errors (recoverable; phrased for a user/LLM).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudioError {
    EmptyCell,
    CellPipelineMismatch {
        upstream: String,
        out_ports: usize,
        downstream: String,
        in_ports: usize,
    },
    DuplicateId(String),
    PortOutOfRange {
        node: String,
        port: usize,
        available: usize,
        direction: &'static str,
    },
    InputAlreadyDriven {
        node: String,
        port: usize,
    },
    InputNotDriven {
        node: String,
        port: usize,
    },
    /// A wire feeds a source's (non-existent) input, or a sink drives something.
    Cycle(Vec<String>),
}

impl std::fmt::Display for StudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudioError::EmptyCell => write!(f, "a runtime cell must contain at least one op"),
            StudioError::CellPipelineMismatch { upstream, out_ports, downstream, in_ports } => write!(
                f,
                "cell pipeline mismatch: `{upstream}` outputs {out_ports} port(s) but `{downstream}` expects {in_ports}"
            ),
            StudioError::DuplicateId(id) => write!(f, "duplicate block id `{id}`"),
            StudioError::PortOutOfRange { node, port, available, direction } => write!(
                f,
                "block `{node}` has no {direction} port {port} (has {available})"
            ),
            StudioError::InputAlreadyDriven { node, port } => {
                write!(f, "block `{node}` input port {port} is driven by more than one wire")
            }
            StudioError::InputNotDriven { node, port } => {
                write!(f, "block `{node}` input port {port} is not connected")
            }
            StudioError::Cycle(nodes) => write!(f, "the block graph has a cycle: {}", nodes.join(" → ")),
        }
    }
}

impl std::error::Error for StudioError {}

/// The visual role of a block (presentation + a light structural check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeRole {
    Source,
    Transform,
    Sink,
}

impl NodeRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeRole::Source => "source",
            NodeRole::Transform => "transform",
            NodeRole::Sink => "sink",
        }
    }
}

/// A single visual block: presentation (id/label/layout) + its Layer-2 cell.
pub struct VisualNode {
    pub id: String,
    pub label: String,
    pub role: NodeRole,
    pub cell: RuntimeCell,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl VisualNode {
    pub fn new(id: &str, role: NodeRole, cell: RuntimeCell) -> Self {
        VisualNode {
            id: id.to_string(),
            label: id.to_string(),
            role,
            cell,
            x: 0.0,
            y: 0.0,
            w: 132.0,
            h: 64.0,
        }
    }
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }
    pub fn at(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }
    pub fn n_in(&self) -> usize {
        self.cell.n_in()
    }
    pub fn n_out(&self) -> usize {
        self.cell.n_out()
    }
}

/// A directed connection from one block's output port to another's input port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub from_node: usize,
    pub from_port: usize,
    pub to_node: usize,
    pub to_port: usize,
}

/// A flat builder of visual blocks and wires.
#[derive(Default)]
pub struct StudioGraph {
    nodes: Vec<VisualNode>,
    wires: Vec<Wire>,
    ids: HashMap<String, usize>,
}

impl StudioGraph {
    pub fn new() -> Self {
        StudioGraph::default()
    }

    /// Add a block; returns its handle (index). Rejects duplicate ids.
    pub fn add(&mut self, node: VisualNode) -> Result<usize, StudioError> {
        if self.ids.contains_key(&node.id) {
            return Err(StudioError::DuplicateId(node.id.clone()));
        }
        let idx = self.nodes.len();
        self.ids.insert(node.id.clone(), idx);
        self.nodes.push(node);
        Ok(idx)
    }

    /// Wire `from_node:from_port` → `to_node:to_port`. Range-checks the ports.
    pub fn connect(
        &mut self,
        from_node: usize,
        from_port: usize,
        to_node: usize,
        to_port: usize,
    ) -> Result<(), StudioError> {
        let src = self.nodes.get(from_node).ok_or_else(|| StudioError::PortOutOfRange {
            node: format!("#{from_node}"),
            port: from_port,
            available: 0,
            direction: "output",
        })?;
        if from_port >= src.n_out() {
            return Err(StudioError::PortOutOfRange {
                node: src.id.clone(),
                port: from_port,
                available: src.n_out(),
                direction: "output",
            });
        }
        let dst = self.nodes.get(to_node).ok_or_else(|| StudioError::PortOutOfRange {
            node: format!("#{to_node}"),
            port: to_port,
            available: 0,
            direction: "input",
        })?;
        if to_port >= dst.n_in() {
            return Err(StudioError::PortOutOfRange {
                node: dst.id.clone(),
                port: to_port,
                available: dst.n_in(),
                direction: "input",
            });
        }
        self.wires.push(Wire { from_node, from_port, to_node, to_port });
        Ok(())
    }

    /// Validate and compile: every input port driven exactly once, no cycles.
    pub fn build(self) -> Result<CompiledStudio, StudioError> {
        let n = self.nodes.len();
        // input_driver[(node, in_port)] = (src_node, src_port)
        let mut driver: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for w in &self.wires {
            let key = (w.to_node, w.to_port);
            if driver.contains_key(&key) {
                return Err(StudioError::InputAlreadyDriven {
                    node: self.nodes[w.to_node].id.clone(),
                    port: w.to_port,
                });
            }
            driver.insert(key, (w.from_node, w.from_port));
        }
        // Every input port of every node must be driven.
        for (idx, node) in self.nodes.iter().enumerate() {
            for p in 0..node.n_in() {
                if !driver.contains_key(&(idx, p)) {
                    return Err(StudioError::InputNotDriven {
                        node: node.id.clone(),
                        port: p,
                    });
                }
            }
        }
        // Topological order over node→node dependencies (Kahn). A cycle means a
        // feedback loop with no delay — out of scope for the dataflow executive.
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for w in &self.wires {
            if w.from_node != w.to_node {
                adj[w.from_node].push(w.to_node);
                indeg[w.to_node] += 1;
            } else {
                // self-loop = immediate algebraic feedback.
                return Err(StudioError::Cycle(vec![self.nodes[w.from_node].id.clone()]));
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut qi = 0;
        while qi < queue.len() {
            let u = queue[qi];
            qi += 1;
            order.push(u);
            for &v in &adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    queue.push(v);
                }
            }
        }
        if order.len() != n {
            let stuck: Vec<String> = (0..n)
                .filter(|&i| indeg[i] > 0)
                .map(|i| self.nodes[i].id.clone())
                .collect();
            return Err(StudioError::Cycle(stuck));
        }

        Ok(CompiledStudio {
            nodes: self.nodes,
            wires: self.wires,
            order,
            driver,
        })
    }
}

/// A validated, runnable flat block graph.
pub struct CompiledStudio {
    pub(crate) nodes: Vec<VisualNode>,
    pub(crate) wires: Vec<Wire>,
    pub(crate) order: Vec<usize>,
    pub(crate) driver: HashMap<(usize, usize), (usize, usize)>,
}

impl CompiledStudio {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn nodes(&self) -> &[VisualNode] {
        &self.nodes
    }
    pub fn wires(&self) -> &[Wire] {
        &self.wires
    }
}

#[cfg(test)]
mod tests {
    use super::super::cell::{Gain, Source, SourceKind};
    use super::*;

    fn src() -> VisualNode {
        VisualNode::new(
            "src",
            NodeRole::Source,
            RuntimeCell::single(Box::new(Source::new("ramp", SourceKind::Ramp { slope: 1.0, intercept: 0.0 }))),
        )
    }
    fn gain(id: &str) -> VisualNode {
        VisualNode::new(id, NodeRole::Transform, RuntimeCell::single(Box::new(Gain::new("g", 2.0))))
    }

    #[test]
    fn build_validates_a_simple_chain() {
        let mut g = StudioGraph::new();
        let s = g.add(src()).unwrap();
        let t = g.add(gain("g1")).unwrap();
        g.connect(s, 0, t, 0).unwrap();
        let compiled = g.build().unwrap();
        assert_eq!(compiled.node_count(), 2);
        assert_eq!(compiled.order, vec![0, 1]);
    }

    #[test]
    fn undriven_input_is_rejected() {
        let mut g = StudioGraph::new();
        let _s = g.add(src()).unwrap();
        let _t = g.add(gain("g1")).unwrap();
        // g1's input is never connected.
        match g.build() {
            Err(StudioError::InputNotDriven { .. }) => {}
            Err(other) => panic!("expected InputNotDriven, got {other:?}"),
            Ok(_) => panic!("expected InputNotDriven, got Ok"),
        }
    }

    #[test]
    fn double_driven_input_is_rejected() {
        let mut g = StudioGraph::new();
        let s1 = g.add(src()).unwrap();
        let s2 = g.add(VisualNode::new(
            "src2",
            NodeRole::Source,
            RuntimeCell::single(Box::new(Source::new("c", SourceKind::Const(1.0)))),
        ))
        .unwrap();
        let t = g.add(gain("g1")).unwrap();
        g.connect(s1, 0, t, 0).unwrap();
        g.connect(s2, 0, t, 0).unwrap();
        match g.build() {
            Err(StudioError::InputAlreadyDriven { .. }) => {}
            Err(other) => panic!("expected InputAlreadyDriven, got {other:?}"),
            Ok(_) => panic!("expected InputAlreadyDriven, got Ok"),
        }
    }

    #[test]
    fn out_of_range_port_is_rejected() {
        let mut g = StudioGraph::new();
        let s = g.add(src()).unwrap();
        let t = g.add(gain("g1")).unwrap();
        assert!(matches!(
            g.connect(s, 1, t, 0),
            Err(StudioError::PortOutOfRange { .. })
        ));
    }
}
