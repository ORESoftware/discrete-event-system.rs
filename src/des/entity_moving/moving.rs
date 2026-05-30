//! Rust port of `src/des/entity-moving/moving.ts`.

use crate::core::{
    DesDecimal, DesResult, Entity, EntityState, GraphData, MovingEntity, MovingEntityState,
    TimeStepContext,
};
use crate::migration::MigrationFile;
use indexmap::IndexMap;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-moving/moving.ts",
    "src/des/entity_moving/moving.rs",
    &[
        "Shared moving fields live in MovingEntityState.",
        "Concrete moving tokens own EntityState and MovingEntityState.",
        "Time stepping defaults are no-op hooks after elapsed-time accrual.",
    ],
    &[
        "AbstractMovingEntity",
        "BasicMovingEntity",
        "BasicQuantityMovingEntity",
        "ProcessableMovingEntity",
        "ProcessingTimeValue",
    ],
);

static NEXT_MOVING_ID: AtomicU64 = AtomicU64::new(0);

fn next_moving_id() -> u64 {
    NEXT_MOVING_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeValue {
    pub station_id: String,
    pub time_in_input_queue: DesDecimal,
    pub time_in_process_queue: DesDecimal,
    pub time_in_output_queue: DesDecimal,
    pub start_time_in_input_queue: DesDecimal,
    pub start_time_in_process_queue: DesDecimal,
    pub start_time_in_output_queue: DesDecimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicMovingEntity<V>
where
    V: Clone,
{
    pub entity: EntityState,
    pub moving: MovingEntityState,
    pub value: V,
}

impl<V> BasicMovingEntity<V>
where
    V: Clone,
{
    pub fn new(value: V) -> Self {
        let moving_id = next_moving_id();
        Self {
            entity: EntityState::new(format!("moving-{moving_id}")),
            moving: MovingEntityState::new(moving_id),
            value,
        }
    }

    pub fn bump_time_in_system(&mut self, step_size: DesDecimal) {
        self.moving.time_in_system += step_size;
    }
}

impl<V> Entity for BasicMovingEntity<V>
where
    V: Clone + std::fmt::Debug + Serialize,
{
    fn state(&self) -> &EntityState {
        &self.entity
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.entity
    }

    fn do_time_step(&mut self, ctx: &TimeStepContext) -> DesResult<()> {
        self.state_mut().bump_time_step();
        self.bump_time_in_system(ctx.step_size);
        self.run_time_step(ctx)
    }

    fn graph_data(&self) -> GraphData {
        let mut properties = IndexMap::new();
        properties.insert(
            "timeInSystem".to_owned(),
            json!(self.moving.time_in_system.to_string()),
        );
        properties.insert(
            "hasExitedSystem".to_owned(),
            json!(self.moving.has_exited_system),
        );
        GraphData {
            entity_id: self.entity.id.clone(),
            kind: "BasicMovingEntity".to_owned(),
            properties,
        }
    }
}

impl<V> MovingEntity for BasicMovingEntity<V>
where
    V: Clone + std::fmt::Debug + Serialize,
{
    type Value = V;

    fn value(&self) -> Self::Value {
        self.value.clone()
    }

    fn moving_state(&self) -> &MovingEntityState {
        &self.moving
    }

    fn moving_state_mut(&mut self) -> &mut MovingEntityState {
        &mut self.moving
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessableMovingEntity<V>
where
    V: Clone,
{
    pub base: BasicMovingEntity<V>,
    pub processing_time_by_station: IndexMap<String, ProcessingTimeValue>,
}

impl<V> ProcessableMovingEntity<V>
where
    V: Clone,
{
    pub fn new(value: V) -> Self {
        Self {
            base: BasicMovingEntity::new(value),
            processing_time_by_station: IndexMap::new(),
        }
    }

    pub fn start_new_station(&mut self, station_id: impl Into<String>, now: DesDecimal) {
        let station_id = station_id.into();
        self.processing_time_by_station.insert(
            station_id.clone(),
            ProcessingTimeValue {
                station_id,
                time_in_input_queue: -Decimal::ONE,
                time_in_process_queue: -Decimal::ONE,
                time_in_output_queue: -Decimal::ONE,
                start_time_in_input_queue: now,
                start_time_in_process_queue: -Decimal::ONE,
                start_time_in_output_queue: -Decimal::ONE,
            },
        );
    }
}

impl<V> Entity for ProcessableMovingEntity<V>
where
    V: Clone + std::fmt::Debug + Serialize,
{
    fn state(&self) -> &EntityState {
        self.base.state()
    }

    fn state_mut(&mut self) -> &mut EntityState {
        self.base.state_mut()
    }
}

impl<V> MovingEntity for ProcessableMovingEntity<V>
where
    V: Clone + std::fmt::Debug + Serialize,
{
    type Value = V;

    fn value(&self) -> Self::Value {
        self.base.value()
    }

    fn moving_state(&self) -> &MovingEntityState {
        self.base.moving_state()
    }

    fn moving_state_mut(&mut self) -> &mut MovingEntityState {
        self.base.moving_state_mut()
    }
}

pub type BasicQuantityMovingEntity = BasicMovingEntity<i64>;
