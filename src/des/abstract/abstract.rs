//! Rust port of `src/des/abstract/abstract.ts`.

use crate::core::{
    DesDecimal, DesError, DesResult, Entity as EntityTrait,
    EntityConnection as CoreEntityConnection, EntityState, GraphData, JsonValue,
    StationaryEntity as StationaryEntityTrait, TimeStepContext,
};
use crate::migration::MigrationFile;
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/abstract/abstract.ts",
    "src/des/abstract/abstract.rs",
    &[
        "TypeScript abstract classes are split into state structs plus traits.",
        "EntityConnection stores source/target ids rather than object references.",
        "Throwing validation paths become DesResult.",
    ],
    &[
        "AbstractBidirectionalEntity",
        "Entity",
        "EntityConnection",
        "EntityObserver",
        "HasNumericValue",
        "IsSerializable",
        "Serializable",
        "StationaryEntity",
        "TimeStepOpts",
    ],
);

pub type TimeStepOpts = TimeStepContext;
pub type EntityConnection = CoreEntityConnection;

pub trait EntityObserver<T> {
    fn do_update(&mut self, event_type: &str, message: &T) -> DesResult<()>;
}

pub trait IsSerializable<T> {
    fn serializable_data(&self) -> T;
}

pub trait Serializable: Serialize {
    fn to_json_value(&self) -> DesResult<JsonValue> {
        serde_json::to_value(self).map_err(|err| DesError::InvalidState {
            context: "Serializable::to_json_value",
            message: err.to_string(),
        })
    }
}

impl<T> Serializable for T where T: Serialize {}

pub trait HasNumericValue {
    fn numeric_value(&self) -> DesDecimal;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub state: EntityState,
    #[serde(default)]
    pub subscribers: IndexSet<String>,
    #[serde(default)]
    pub subscribers_by_event: IndexMap<String, IndexSet<String>>,
}

impl Entity {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            state: EntityState::new(id),
            subscribers: IndexSet::new(),
            subscribers_by_event: IndexMap::new(),
        }
    }

    pub fn subscribe(&mut self, observer_id: impl Into<String>) -> bool {
        self.subscribers.insert(observer_id.into())
    }

    pub fn subscribe_to(&mut self, event_name: impl Into<String>, observer_id: impl Into<String>) {
        self.subscribers_by_event
            .entry(event_name.into())
            .or_default()
            .insert(observer_id.into());
    }

    pub fn send_update_to_subscribers<T: Debug>(
        &self,
        _event_type: &str,
        _message: &T,
    ) -> DesResult<()> {
        Ok(())
    }
}

impl EntityTrait for Entity {
    fn state(&self) -> &EntityState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.state
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationaryEntity {
    pub entity: Entity,
    #[serde(default)]
    pub connections_in: Vec<EntityConnection>,
    #[serde(default)]
    pub connections_out: Vec<EntityConnection>,
}

impl StationaryEntity {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            entity: Entity::new(id),
            connections_in: Vec::new(),
            connections_out: Vec::new(),
        }
    }

    pub fn add_in_connection(&mut self, source: impl Into<String>) -> EntityConnection {
        let conn = EntityConnection::new(source, self.entity.state.id.clone());
        self.connections_in.push(conn.clone());
        conn
    }

    pub fn add_out_connection(&mut self, target: impl Into<String>) -> EntityConnection {
        let conn = EntityConnection::new(self.entity.state.id.clone(), target);
        self.connections_out.push(conn.clone());
        conn
    }

    pub fn get_with_computed_properties(&self) -> IndexMap<String, JsonValue> {
        IndexMap::new()
    }
}

impl EntityTrait for StationaryEntity {
    fn state(&self) -> &EntityState {
        &self.entity.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.entity.state
    }

    fn graph_data(&self) -> GraphData {
        GraphData {
            entity_id: self.state().id.clone(),
            kind: "StationaryEntity".to_owned(),
            properties: self.get_with_computed_properties(),
        }
    }
}

impl<I> StationaryEntityTrait<I> for StationaryEntity {
    fn take_item(&mut self, _item: I) -> DesResult<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractBidirectionalEntity {
    pub stationary: StationaryEntity,
}

impl AbstractBidirectionalEntity {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            stationary: StationaryEntity::new(id),
        }
    }
}

impl EntityTrait for AbstractBidirectionalEntity {
    fn state(&self) -> &EntityState {
        self.stationary.state()
    }

    fn state_mut(&mut self) -> &mut EntityState {
        self.stationary.state_mut()
    }
}
