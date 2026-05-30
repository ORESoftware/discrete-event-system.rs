//! Canonical use path: `crate::des::entity_routing::output_routing_policy::*`
//!
//! Port of `src/des/entity-routing/output-routing-policy.ts` — the output
//! connection ordering policies for queueing-style stations that choose ONE
//! accepting downstream target. Deliberately NOT used by broadcast routing.
//!
//! The TS string-union `'random' | 'round-robin' | 'ordered'` becomes the enum
//! [`OutputRoutingPolicy`]. `OutputConnectionRouter<C>` keeps its generic over
//! the connection element so the same router serves any `C: Clone`.
//!
//! DETERMINISM: the `'random'` policy uses `fisher_yates_shuffle`, which consumes
//! randomness. Per the migration header, the router holds an injected
//! `RandomSource` rather than reaching for an ambient one; the no-arg `new`
//! seeds a deterministic [`SeededRandom`] so behaviour is reproducible.

#![allow(dead_code)]

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// `type OutputRoutingPolicy = 'random' | 'round-robin' | 'ordered'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum OutputRoutingPolicy {
    #[default]
    Random,
    RoundRobin,
    Ordered,
}


/// `interface HasOutputRoutingPolicy { outputRouting?: OutputRoutingPolicy }`.
///
/// PORT NOTE: the TS interface was a single optional field; here it is a tiny
/// trait so an entity can advertise its preferred policy. Default is `None`
/// (matching the optional field being absent).
pub trait HasOutputRoutingPolicy {
    fn output_routing(&self) -> Option<OutputRoutingPolicy> {
        None
    }
}

/// `class OutputConnectionRouter<C>`.
///
/// PORT NOTE: the TS class held only `cursor` + `policy`. We add an injected
/// `rng` because the Rust `fisher_yates_shuffle` takes a `RandomSource` (the TS
/// version closed over ambient randomness).
pub struct OutputConnectionRouter {
    policy: OutputRoutingPolicy,
    cursor: usize,
    rng: Box<dyn RandomSource>,
}

impl OutputConnectionRouter {
    /// `new(policy = 'random')` — seeds a deterministic RNG for the random policy.
    pub fn new(policy: OutputRoutingPolicy) -> Self {
        OutputConnectionRouter {
            policy,
            cursor: 0,
            // Deterministic default so simulations reproduce; swap via `with_rng`.
            rng: Box::new(SeededRandom::new(0x0BAD_F00D)),
        }
    }

    /// Construct with an explicit RNG (for reproducible / shared randomness).
    pub fn with_rng(policy: OutputRoutingPolicy, rng: Box<dyn RandomSource>) -> Self {
        OutputConnectionRouter {
            policy,
            cursor: 0,
            rng,
        }
    }

    /// `order(connections)` — return the connections in the policy's preferred
    /// visiting order. Returns owned clones (the TS returned `slice()`/`concat`
    /// copies), so requires `C: Clone`.
    pub fn order<C: Clone>(&mut self, connections: &[C]) -> Vec<C> {
        if connections.len() <= 1 {
            return connections.to_vec();
        }
        match self.policy {
            OutputRoutingPolicy::Ordered => connections.to_vec(),
            OutputRoutingPolicy::RoundRobin => {
                let start = self.cursor % connections.len();
                let mut v: Vec<C> = connections[start..].to_vec();
                v.extend_from_slice(&connections[..start]);
                v
            }
            OutputRoutingPolicy::Random => {
                let mut v = connections.to_vec();
                fisher_yates_shuffle(&mut v, &mut *self.rng);
                v
            }
        }
    }

    /// `markAccepted(connections, accepted)` — advance the round-robin cursor to
    /// just past the accepted connection. Faithful to the TS `indexOf` lookup,
    /// so requires `C: PartialEq`.
    pub fn mark_accepted<C: PartialEq>(&mut self, connections: &[C], accepted: &C) {
        if self.policy != OutputRoutingPolicy::RoundRobin || connections.is_empty() {
            return;
        }
        match connections.iter().position(|c| c == accepted) {
            Some(ix) => self.cursor = (ix + 1) % connections.len(),
            None => {
                eprintln!(
                    "[output-router] round-robin markAccepted: accepted connection not found in the {}-element connection list; cursor left unchanged (rotation may stall).",
                    connections.len()
                );
            }
        }
    }

    /// Identity-free variant of [`OutputConnectionRouter::mark_accepted`] for
    /// element types that cannot derive `PartialEq` (e.g.
    /// `Rc<RefCell<EntityConnection>>`): the caller supplies the index of the
    /// accepted connection in the ORIGINAL (unordered) list.
    pub fn mark_accepted_index(&mut self, connections_len: usize, accepted_index: usize) {
        if self.policy != OutputRoutingPolicy::RoundRobin || connections_len == 0 {
            return;
        }
        if accepted_index >= connections_len {
            eprintln!(
                "[output-router] round-robin markAcceptedIndex: index {accepted_index} out of range for {connections_len} connections; cursor left unchanged."
            );
            return;
        }
        self.cursor = (accepted_index + 1) % connections_len;
    }

    /// `getCursor()`.
    pub fn get_cursor(&self) -> usize {
        self.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_preserves_order() {
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::Ordered);
        let conns = vec![1, 2, 3];
        assert_eq!(r.order(&conns), vec![1, 2, 3]);
    }

    #[test]
    fn single_or_empty_is_returned_verbatim() {
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::Random);
        assert_eq!(r.order::<i32>(&[]), Vec::<i32>::new());
        assert_eq!(r.order(&[7]), vec![7]);
    }

    #[test]
    fn round_robin_rotates_after_accept() {
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::RoundRobin);
        let conns = vec!["a", "b", "c"];
        assert_eq!(r.order(&conns), vec!["a", "b", "c"]);
        r.mark_accepted(&conns, &"a");
        // cursor now 1 -> order starts at "b".
        assert_eq!(r.order(&conns), vec!["b", "c", "a"]);
        assert_eq!(r.get_cursor(), 1);
        r.mark_accepted_index(conns.len(), 2);
        assert_eq!(r.get_cursor(), 0);
    }

    #[test]
    fn random_is_a_permutation() {
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::Random);
        let conns = vec![1, 2, 3, 4, 5];
        let mut ordered = r.order(&conns);
        ordered.sort();
        assert_eq!(ordered, conns);
    }
}
