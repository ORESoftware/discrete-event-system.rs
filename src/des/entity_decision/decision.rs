//! Rust port of `src/des/entity-decision/decision.ts`.

use crate::core::{
    DesResult, Entity, EntityConnection, EntityState, GraphData, QueueState, StationaryEntity,
    TimeStepContext,
};
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-decision/decision.ts",
    "src/des/entity_decision/decision.rs",
    &[
        "Base decision behavior is a queued station with explicit branch connections.",
        "Subclasses/strategies choose branch indices.",
    ],
    &["Decision"],
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision<T> {
    pub state: EntityState,
    pub queue: QueueState<T>,
    pub connections_out: Vec<EntityConnection>,
}

impl<T> Decision<T> {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            state: EntityState::new(id),
            queue: QueueState::new(None),
            connections_out: Vec::new(),
        }
    }

    pub fn add_out_connection(&mut self, connection: EntityConnection) {
        self.connections_out.push(connection);
    }
}

impl<T> Entity for Decision<T>
where
    T: Debug,
{
    fn state(&self) -> &EntityState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.state
    }

    fn run_time_step(&mut self, _ctx: &TimeStepContext) -> DesResult<()> {
        Ok(())
    }

    fn graph_data(&self) -> GraphData {
        GraphData {
            entity_id: self.state.id.clone(),
            kind: "Decision".to_owned(),
            properties: Default::default(),
        }
    }
}

impl<T> StationaryEntity<T> for Decision<T>
where
    T: Debug,
{
    fn take_item(&mut self, item: T) -> DesResult<()> {
        self.queue.push(item)
    }
}
