//! Core Rust DES abstractions lifted from the TypeScript base classes.

use indexmap::{IndexMap, IndexSet};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fmt::Debug;
use uuid::Uuid;

pub type DesDecimal = Decimal;
pub type JsonValue = Value;
pub type EntityId = String;
pub type ChannelId = String;
pub type Tick = u64;
pub type DesResult<T> = Result<T, DesError>;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DesError {
    #[error("{context}: {message}")]
    InvalidState {
        context: &'static str,
        message: String,
    },
    #[error("validation failed for {entity_id}: {message}")]
    Validation {
        entity_id: EntityId,
        message: String,
    },
    #[error("unsupported migrated behavior in {module}: {detail}")]
    Unsupported {
        module: &'static str,
        detail: &'static str,
    },
}

pub fn short_uuid() -> String {
    Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .rev()
        .take(10)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphData {
    pub entity_id: EntityId,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub properties: IndexMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStepContext {
    pub tick: Tick,
    pub step_size: DesDecimal,
    #[serde(default)]
    pub now: DesDecimal,
}

impl TimeStepContext {
    pub fn new(step_size: DesDecimal) -> Self {
        Self {
            tick: 0,
            step_size,
            now: Decimal::ZERO,
        }
    }

    pub fn advance(&mut self) {
        self.tick += 1;
        self.now += self.step_size;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityState {
    pub id: EntityId,
    pub short_uuid: Option<String>,
    pub time_step_count: Tick,
    pub subscribers_by_event: IndexMap<String, IndexSet<EntityId>>,
}

impl EntityState {
    pub fn new(id: impl Into<EntityId>) -> Self {
        Self {
            id: id.into(),
            short_uuid: None,
            time_step_count: 0,
            subscribers_by_event: IndexMap::new(),
        }
    }

    pub fn bump_time_step(&mut self) {
        self.time_step_count += 1;
    }

    pub fn ensure_short_uuid(&mut self) -> &str {
        if self.short_uuid.is_none() {
            self.short_uuid = Some(short_uuid());
        }
        self.short_uuid.as_deref().expect("short_uuid just set")
    }
}

pub trait Entity: Debug {
    fn state(&self) -> &EntityState;
    fn state_mut(&mut self) -> &mut EntityState;

    fn id(&self) -> &str {
        &self.state().id
    }

    fn validate_before_run(&self) -> DesResult<()> {
        Ok(())
    }

    fn validate(&self) -> DesResult<()> {
        Ok(())
    }

    fn run_time_step(&mut self, _ctx: &TimeStepContext) -> DesResult<()> {
        Ok(())
    }

    fn do_time_step(&mut self, ctx: &TimeStepContext) -> DesResult<()> {
        self.state_mut().bump_time_step();
        self.run_time_step(ctx)
    }

    fn run_finish(&mut self) -> DesResult<()> {
        Ok(())
    }

    fn graph_data(&self) -> GraphData {
        GraphData {
            entity_id: self.id().to_owned(),
            kind: std::any::type_name::<Self>().to_owned(),
            properties: IndexMap::new(),
        }
    }
}

pub trait MovingEntity: Entity {
    type Value: Clone + Debug + Serialize;

    fn value(&self) -> Self::Value;
    fn moving_state(&self) -> &MovingEntityState;
    fn moving_state_mut(&mut self) -> &mut MovingEntityState;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingEntityState {
    pub moving_id: u64,
    pub moving_uuid: String,
    pub stations_visited_count: u64,
    pub total_wait_time: DesDecimal,
    pub total_in_process_time: DesDecimal,
    pub time_in_system: DesDecimal,
    pub has_exited_system: bool,
    pub out_queue_wait_time: DesDecimal,
}

impl MovingEntityState {
    pub fn new(moving_id: u64) -> Self {
        Self {
            moving_id,
            moving_uuid: short_uuid(),
            stations_visited_count: 0,
            total_wait_time: Decimal::ZERO,
            total_in_process_time: Decimal::ZERO,
            time_in_system: Decimal::ZERO,
            has_exited_system: false,
            out_queue_wait_time: Decimal::ZERO,
        }
    }
}

pub trait StationaryEntity<I>: Entity {
    fn accept_item(&self, _item: &I) -> bool {
        true
    }

    fn take_item(&mut self, item: I) -> DesResult<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityConnection {
    pub id: EntityId,
    pub source: EntityId,
    pub target: EntityId,
    pub channel: Option<ChannelId>,
}

impl EntityConnection {
    pub fn new(source: impl Into<EntityId>, target: impl Into<EntityId>) -> Self {
        let source = source.into();
        let target = target.into();
        Self {
            id: format!("{source}->{target}:{}", short_uuid()),
            source,
            target,
            channel: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueState<T> {
    items: VecDeque<T>,
    pub max_len: Option<usize>,
}

impl<T> QueueState<T> {
    pub fn new(max_len: Option<usize>) -> Self {
        Self {
            items: VecDeque::new(),
            max_len,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.max_len
            .map(|max_len| self.items.len() >= max_len)
            .unwrap_or(false)
    }

    pub fn push(&mut self, item: T) -> DesResult<()> {
        if self.is_full() {
            return Err(DesError::InvalidState {
                context: "QueueState::push",
                message: "queue is full".to_owned(),
            });
        }
        self.items.push_back(item);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

pub trait PureTransform<I, O> {
    fn transform(&self, input: I) -> DesResult<O>;
}

pub trait RandomSource {
    fn next_f64(&mut self) -> f64;
}

impl<F> RandomSource for F
where
    F: FnMut() -> f64,
{
    fn next_f64(&mut self) -> f64 {
        self()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub observed: Option<JsonValue>,
    pub expected: Option<JsonValue>,
    pub message: Option<String>,
}
