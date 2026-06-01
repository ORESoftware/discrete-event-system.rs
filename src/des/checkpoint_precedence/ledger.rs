//! Canonical use path: `crate::des::checkpoint_precedence::ledger::*`
//!
//! The **precedence ledger** — the shared brain that records, by token UUID, the
//! happens-before constraints between movables and which tokens have cleared
//! which checkpoints.
//!
//! Each token is *stamped* with a stable `uuid` (reusing the movable's existing
//! [`moving_uuid`](crate::des::entity_moving::moving::MovingCore)) and a monotonic
//! `seq`. A token may declare, by reference, that it must not pass a checkpoint
//! until certain *other* tokens (named by UUID) have cleared it:
//!
//! ```text
//!   token Y: "I may not pass checkpoint C until token X has cleared C"
//! ```
//!
//! These pairwise references form a happens-before DAG **per checkpoint**.
//! [`validate`](PrecedenceLedger::validate) proves the DAG is acyclic and that
//! every referenced predecessor exists — the token-level analog of the node
//! scheduler's forward-cycle check. The `seq` stamp is the deterministic
//! tie-break when several tokens are simultaneously eligible (see the gate).

#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// One happens-before reference: this token may not pass `checkpoint` until token
/// `predecessor` (by UUID) has cleared `checkpoint`.
#[derive(Clone, Debug)]
pub struct Requirement {
    pub predecessor: String,
    pub checkpoint: String,
}

impl Requirement {
    pub fn new(predecessor: &str, checkpoint: &str) -> Self {
        Requirement {
            predecessor: predecessor.to_string(),
            checkpoint: checkpoint.to_string(),
        }
    }
}

/// The stamp + constraints carried by one token.
#[derive(Clone, Debug)]
pub struct TokenSpec {
    pub uuid: String,
    /// Monotonic, deterministic ordering key (tie-break among eligible tokens).
    pub seq: u64,
    /// A numeric payload, purely for the demo's observable output.
    pub payload: f64,
    pub requirements: Vec<Requirement>,
}

/// Shared store of token stamps/constraints and per-checkpoint clearances. The
/// source registers specs; the gate reads them and records clearances.
#[derive(Default)]
pub struct PrecedenceLedger {
    specs: HashMap<String, TokenSpec>,
    /// Registration order — gives deterministic iteration and error messages.
    order_registered: Vec<String>,
    /// `checkpoint -> set of token UUIDs that have cleared it`.
    cleared: HashMap<String, HashSet<String>>,
}

impl PrecedenceLedger {
    pub fn new() -> Self {
        PrecedenceLedger::default()
    }

    /// Register a token's stamp + constraints. Duplicate UUIDs panic — an
    /// order-sensitive system must never conflate two tokens.
    pub fn register(&mut self, spec: TokenSpec) {
        if self.specs.contains_key(&spec.uuid) {
            panic!("duplicate token uuid registered: {}", spec.uuid);
        }
        self.order_registered.push(spec.uuid.clone());
        self.specs.insert(spec.uuid.clone(), spec);
    }

    pub fn seq_of(&self, uuid: &str) -> Option<u64> {
        self.specs.get(uuid).map(|s| s.seq)
    }

    pub fn payload_of(&self, uuid: &str) -> Option<f64> {
        self.specs.get(uuid).map(|s| s.payload)
    }

    /// Record that `uuid` has cleared `checkpoint`.
    pub fn mark_cleared(&mut self, checkpoint: &str, uuid: &str) {
        self.cleared
            .entry(checkpoint.to_string())
            .or_default()
            .insert(uuid.to_string());
    }

    pub fn is_cleared(&self, checkpoint: &str, uuid: &str) -> bool {
        self.cleared
            .get(checkpoint)
            .map(|s| s.contains(uuid))
            .unwrap_or(false)
    }

    /// Is `uuid` allowed through `checkpoint` *now*? True iff every predecessor it
    /// references **at this checkpoint** has already cleared it. Constraints
    /// targeting other checkpoints are ignored here.
    pub fn requirements_satisfied(&self, uuid: &str, checkpoint: &str) -> bool {
        let spec = match self.specs.get(uuid) {
            Some(s) => s,
            None => return false,
        };
        spec.requirements
            .iter()
            .filter(|r| r.checkpoint == checkpoint)
            .all(|r| self.is_cleared(checkpoint, &r.predecessor))
    }

    /// Validate the declared constraints BEFORE running:
    /// 1. no token references itself;
    /// 2. every referenced predecessor is a registered token;
    /// 3. the happens-before graph at each checkpoint is acyclic.
    ///
    /// Returns a descriptive error on the first violation (fail fast).
    pub fn validate(&self) -> Result<(), String> {
        for uuid in &self.order_registered {
            let spec = &self.specs[uuid];
            for req in &spec.requirements {
                if req.predecessor == *uuid {
                    return Err(format!(
                        "token '{uuid}' lists itself as a predecessor at checkpoint '{}'",
                        req.checkpoint
                    ));
                }
                if !self.specs.contains_key(&req.predecessor) {
                    return Err(format!(
                        "token '{uuid}' requires unknown predecessor '{}' at checkpoint '{}'",
                        req.predecessor, req.checkpoint
                    ));
                }
            }
        }

        // Distinct checkpoints, in first-seen (registration) order.
        let mut checkpoints: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for uuid in &self.order_registered {
            for req in &self.specs[uuid].requirements {
                if seen.insert(req.checkpoint.clone()) {
                    checkpoints.push(req.checkpoint.clone());
                }
            }
        }
        for cp in &checkpoints {
            if let Some(stuck) = self.cycle_at(cp) {
                return Err(format!(
                    "precedence constraints at checkpoint '{cp}' contain a cycle among {stuck:?}; \
                     the release order is unsatisfiable"
                ));
            }
        }
        Ok(())
    }

    /// Detect a cycle in the happens-before graph restricted to `checkpoint`.
    /// Returns the still-blocked nodes (deterministically ordered) on a cycle, or
    /// `None` if acyclic. Uses Kahn's algorithm with a min-index tie-break, mirroring
    /// the node scheduler.
    fn cycle_at(&self, checkpoint: &str) -> Option<Vec<String>> {
        let mut nodes: Vec<String> = Vec::new();
        let mut pos: HashMap<String, usize> = HashMap::new();
        let push = |id: &str, nodes: &mut Vec<String>, pos: &mut HashMap<String, usize>| {
            if !pos.contains_key(id) {
                pos.insert(id.to_string(), nodes.len());
                nodes.push(id.to_string());
            }
        };

        let mut edges: Vec<(usize, usize)> = Vec::new();
        for uuid in &self.order_registered {
            for req in &self.specs[uuid].requirements {
                if req.checkpoint == checkpoint {
                    push(&req.predecessor, &mut nodes, &mut pos);
                    push(uuid, &mut nodes, &mut pos);
                    edges.push((pos[&req.predecessor], pos[uuid]));
                }
            }
        }

        let n = nodes.len();
        let mut indegree = vec![0usize; n];
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (f, t) in edges {
            succ[f].push(t);
            indegree[t] += 1;
        }

        let mut ready: BinaryHeap<Reverse<usize>> =
            (0..n).filter(|&i| indegree[i] == 0).map(Reverse).collect();
        let mut consumed = 0usize;
        while let Some(Reverse(i)) = ready.pop() {
            consumed += 1;
            let mut outs = succ[i].clone();
            outs.sort_unstable();
            for j in outs {
                indegree[j] -= 1;
                if indegree[j] == 0 {
                    ready.push(Reverse(j));
                }
            }
        }

        if consumed == n {
            None
        } else {
            Some(
                (0..n)
                    .filter(|&i| indegree[i] > 0)
                    .map(|i| nodes[i].clone())
                    .collect(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(uuid: &str, seq: u64, reqs: Vec<Requirement>) -> TokenSpec {
        TokenSpec {
            uuid: uuid.to_string(),
            seq,
            payload: seq as f64,
            requirements: reqs,
        }
    }

    #[test]
    fn satisfied_only_after_predecessor_clears() {
        let mut l = PrecedenceLedger::new();
        l.register(spec("X", 1, vec![]));
        l.register(spec("Y", 2, vec![Requirement::new("X", "C")]));
        assert!(l.requirements_satisfied("X", "C"));
        assert!(!l.requirements_satisfied("Y", "C"));
        l.mark_cleared("C", "X");
        assert!(l.requirements_satisfied("Y", "C"));
    }

    #[test]
    fn unknown_predecessor_is_rejected() {
        let mut l = PrecedenceLedger::new();
        l.register(spec("Y", 1, vec![Requirement::new("GHOST", "C")]));
        let err = l.validate().unwrap_err();
        assert!(err.contains("unknown predecessor"), "{err}");
    }

    #[test]
    fn self_reference_is_rejected() {
        let mut l = PrecedenceLedger::new();
        l.register(spec("Y", 1, vec![Requirement::new("Y", "C")]));
        assert!(l.validate().unwrap_err().contains("itself"));
    }

    #[test]
    fn cyclic_precedence_is_rejected() {
        let mut l = PrecedenceLedger::new();
        l.register(spec("A", 1, vec![Requirement::new("B", "C")]));
        l.register(spec("B", 2, vec![Requirement::new("A", "C")]));
        let err = l.validate().unwrap_err();
        assert!(err.contains("cycle"), "{err}");
    }

    #[test]
    fn acyclic_partial_order_validates() {
        let mut l = PrecedenceLedger::new();
        l.register(spec("A", 1, vec![]));
        l.register(spec("B", 2, vec![Requirement::new("A", "C")]));
        l.register(spec("D", 3, vec![Requirement::new("A", "C")]));
        assert!(l.validate().is_ok());
    }
}
