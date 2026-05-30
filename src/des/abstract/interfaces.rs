//! Rust port of `src/des/abstract/interfaces.ts`.

use crate::core::{
    ChannelId, DesResult, EntityConnection, EntityId, GraphData, QueueState, TimeStepContext,
};
use crate::migration::MigrationFile;
use serde::Serialize;
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/abstract/interfaces.ts",
    "src/des/abstract/interfaces.rs",
    &[
        "Behavioral TypeScript interfaces are Rust traits.",
        "EntityGraphData is represented by GraphData in crate::core.",
        "Input/output connection traits use associated item/endpoint types.",
    ],
    &[
        "EntityGraphData",
        "EventNames",
        "HasEntityValidation",
        "HasId",
        "HasInput",
        "HasInternalQueue",
        "HasManyInputConnections",
        "HasManyOutputConnections",
        "HasOutput",
        "HasSingleInputConnection",
        "HasSingleOutputConnection",
        "IsObservable",
    ],
);

pub type EntityGraphData = GraphData;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventName {
    GraphData,
    Processing,
    Custom(String),
}

impl EventName {
    pub fn as_channel(&self) -> ChannelId {
        match self {
            Self::GraphData => "GRAPH_DATA".to_owned(),
            Self::Processing => "GRAPH_DATA:PROCESSING".to_owned(),
            Self::Custom(v) => v.clone(),
        }
    }
}

pub trait HasId {
    fn id(&self) -> &str;
}

pub trait HasEntityValidation {
    fn validate_before_run(&self) -> DesResult<()>;
    fn validate(&self) -> DesResult<()>;
}

pub trait IsObservable<E> {
    fn subscribe(&mut self, observer_id: EntityId) -> bool;
    fn unsubscribe(&mut self, observer_id: &str) -> bool;
    fn subscribe_to(&mut self, event: EventName, observer_id: EntityId) -> bool;
    fn send_update_to_subscribers(&mut self, event: EventName, event_value: &E) -> DesResult<()>;
}

pub trait HasInput<I> {
    fn accept_item(&self, item: &I) -> bool;
    fn take_item(&mut self, item: I) -> DesResult<()>;
}

pub trait HasOutput<O> {
    fn emit_item(&mut self, item: O) -> DesResult<()>;
}

pub trait HasSingleInputConnection {
    fn input_connection(&self) -> Option<&EntityConnection>;
    fn set_input_connection(&mut self, connection: EntityConnection) -> DesResult<()>;
}

pub trait HasSingleOutputConnection {
    fn output_connection(&self) -> Option<&EntityConnection>;
    fn set_output_connection(&mut self, connection: EntityConnection) -> DesResult<()>;
}

pub trait HasManyInputConnections {
    fn input_connections(&self) -> &[EntityConnection];
    fn add_input_connection(&mut self, connection: EntityConnection);
}

pub trait HasManyOutputConnections {
    fn output_connections(&self) -> &[EntityConnection];
    fn add_output_connection(&mut self, connection: EntityConnection);
}

pub trait HasInternalQueue<T> {
    fn queue(&self) -> &QueueState<T>;
    fn queue_mut(&mut self) -> &mut QueueState<T>;

    fn is_empty(&self) -> bool {
        self.queue().is_empty()
    }

    fn is_full(&self) -> bool {
        self.queue().is_full()
    }
}

pub trait TimeStepped {
    fn run_time_step(&mut self, ctx: &TimeStepContext) -> DesResult<()>;
}

pub trait SerializableGraphData: Debug + Serialize {
    fn graph_data(&self) -> GraphData;
}
