//! Rust port of `src/des/entity-routing/output-routing-policy.ts`.

use crate::core::{EntityConnection, RandomSource};
use crate::des::general::general::fisher_yates_shuffle;
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-routing/output-routing-policy.ts",
    "src/des/entity_routing/output_routing_policy.rs",
    &[
        "Output routing strategy is a Rust enum plus router state.",
        "Random routing takes an injected RandomSource.",
    ],
    &["OutputConnectionRouter", "OutputRoutingPolicy"],
);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputRoutingPolicy {
    RoundRobin,
    #[default]
    Random,
    Ordered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConnectionRouter {
    pub policy: OutputRoutingPolicy,
    pub next_round_robin_index: usize,
}

impl OutputConnectionRouter {
    pub fn new(policy: OutputRoutingPolicy) -> Self {
        Self {
            policy,
            next_round_robin_index: 0,
        }
    }

    pub fn ordered_connections<R>(
        &mut self,
        connections: &[EntityConnection],
        rng: &mut R,
    ) -> Vec<EntityConnection>
    where
        R: RandomSource,
    {
        match self.policy {
            OutputRoutingPolicy::Ordered => connections.to_vec(),
            OutputRoutingPolicy::Random => {
                let mut shuffled = connections.to_vec();
                fisher_yates_shuffle(&mut shuffled, rng);
                shuffled
            }
            OutputRoutingPolicy::RoundRobin => {
                if connections.is_empty() {
                    return Vec::new();
                }
                let start = self.next_round_robin_index % connections.len();
                connections[start..]
                    .iter()
                    .chain(connections[..start].iter())
                    .cloned()
                    .collect()
            }
        }
    }

    pub fn mark_accepted(&mut self, connections: &[EntityConnection], accepted: &EntityConnection) {
        if self.policy != OutputRoutingPolicy::RoundRobin || connections.is_empty() {
            return;
        }
        if let Some(index) = connections.iter().position(|c| c.id == accepted.id) {
            self.next_round_robin_index = (index + 1) % connections.len();
        }
    }
}
