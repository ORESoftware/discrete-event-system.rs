//! Rust port of `src/des/entity-decision/binary-decision.ts`.

use crate::core::{DesError, DesResult, Entity, EntityConnection, EntityState};
use crate::des::entity_decision::decision::Decision;
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-decision/binary-decision.ts",
    "src/des/entity_decision/binary_decision.rs",
    &[
        "Binary decision is a typed wrapper over Decision with exactly two branches.",
        "The branch selector becomes a predicate closure or trait implementor.",
    ],
    &["BinaryDecision"],
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryBranch {
    False = 0,
    True = 1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryDecision<T> {
    pub decision: Decision<T>,
}

impl<T> BinaryDecision<T> {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            decision: Decision::new(id),
        }
    }

    pub fn connection_for(&self, branch: BinaryBranch) -> Option<&EntityConnection> {
        self.decision.connections_out.get(branch as usize)
    }
}

impl<T> Entity for BinaryDecision<T>
where
    T: Debug,
{
    fn state(&self) -> &EntityState {
        &self.decision.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.decision.state
    }

    fn validate_before_run(&self) -> DesResult<()> {
        if self.decision.connections_out.len() != 2 {
            return Err(DesError::Validation {
                entity_id: self.decision.state.id.clone(),
                message: "binary decision requires exactly two out-connections".to_owned(),
            });
        }
        Ok(())
    }
}
