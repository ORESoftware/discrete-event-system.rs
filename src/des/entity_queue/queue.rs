//! Rust port of `src/des/entity-queue/queue.ts`.

use crate::core::{
    DesResult, Entity, EntityState, GraphData, JsonValue, QueueState, StationaryEntity,
    TimeStepContext,
};
use crate::migration::MigrationFile;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-queue/queue.ts",
    "src/des/entity_queue/queue.rs",
    &[
        "QueueEntity is a concrete station with QueueState<T> storage.",
        "Serialization returns a stable snapshot instead of graph pointers.",
    ],
    &["QueueEntity", "QueueEntityGraphData"],
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEntityGraphData {
    pub entity_id: String,
    pub queue_size: usize,
    pub processed_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEntity<T> {
    pub state: EntityState,
    pub queue: QueueState<T>,
    pub processed_count: u64,
}

impl<T> QueueEntity<T> {
    pub fn new(id: impl Into<String>, max_queue_size: Option<usize>) -> Self {
        Self {
            state: EntityState::new(id),
            queue: QueueState::new(max_queue_size),
            processed_count: 0,
        }
    }

    pub fn serializable_data(&self) -> QueueEntityGraphData {
        QueueEntityGraphData {
            entity_id: self.state.id.clone(),
            queue_size: self.queue.len(),
            processed_count: self.processed_count,
        }
    }
}

impl<T> Entity for QueueEntity<T>
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
        let mut properties: IndexMap<String, JsonValue> = IndexMap::new();
        properties.insert("queue.size".to_owned(), json!(self.queue.len()));
        properties.insert("processedCount".to_owned(), json!(self.processed_count));
        GraphData {
            entity_id: self.state.id.clone(),
            kind: "QueueEntity".to_owned(),
            properties,
        }
    }
}

impl<T> StationaryEntity<T> for QueueEntity<T>
where
    T: Debug,
{
    fn accept_item(&self, _item: &T) -> bool {
        !self.queue.is_full()
    }

    fn take_item(&mut self, item: T) -> DesResult<()> {
        self.queue.push(item)
    }
}
