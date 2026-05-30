//! Rust port of `src/des/entity-processing/per-individual-processor.ts`.

use crate::core::{
    DesDecimal, DesResult, Entity, EntityConnection, EntityState, GraphData, JsonValue,
    RandomSource, StationaryEntity, TimeStepContext,
};
use crate::des::entity_routing::output_routing_policy::{
    OutputConnectionRouter, OutputRoutingPolicy,
};
use crate::migration::MigrationFile;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::{Debug, Formatter};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-processing/per-individual-processor.ts",
    "src/des/entity_processing/per_individual_processor.rs",
    &[
        "PerIndividualProcessorOpts is a generic opts struct over the service-time draw closure.",
        "QueuedItem remains local queue storage with per-entity remaining time stored as DesDecimal.",
        "Object-reference out-connections become EntityConnection target ids plus PerIndividualSink trait objects at routing time.",
        "No-downstream/no-accept retry preserves the TypeScript unshift-with-zero-remaining-time behavior.",
    ],
    &["PerIndividualProcessor", "PerIndividualProcessorOpts", "QueuedItem"],
);

pub struct PerIndividualProcessorOpts<D>
where
    D: FnMut() -> DesDecimal,
{
    pub draw_duration: D,
    pub output_routing: OutputRoutingPolicy,
}

impl<D> PerIndividualProcessorOpts<D>
where
    D: FnMut() -> DesDecimal,
{
    pub fn new(draw_duration: D) -> Self {
        Self {
            draw_duration,
            output_routing: OutputRoutingPolicy::default(),
        }
    }

    pub fn with_output_routing(draw_duration: D, output_routing: OutputRoutingPolicy) -> Self {
        Self {
            draw_duration,
            output_routing,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueuedItem<T> {
    pub entity: T,
    pub remaining_time: DesDecimal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerIndividualProcessorAudit {
    pub total_size: usize,
}

pub trait PerIndividualSink<T> {
    fn id(&self) -> &str;

    fn accept_item(&self, _item: &T) -> bool {
        true
    }

    fn take_item(&mut self, item: T) -> DesResult<()>;
}

pub struct PerIndividualProcessor<T> {
    pub state: EntityState,
    pub items: Vec<QueuedItem<T>>,
    pub max_queue_size: isize,
    pub output_router: OutputConnectionRouter,
    pub connections_out: Vec<EntityConnection>,
    draw_duration: Box<dyn FnMut() -> DesDecimal>,
}

impl<T> PerIndividualProcessor<T> {
    pub fn new<D>(id: impl Into<String>, opts: PerIndividualProcessorOpts<D>) -> Self
    where
        D: FnMut() -> DesDecimal + 'static,
    {
        Self {
            state: EntityState::new(id),
            items: Vec::new(),
            max_queue_size: -1,
            output_router: OutputConnectionRouter::new(opts.output_routing),
            connections_out: Vec::new(),
            draw_duration: Box::new(opts.draw_duration),
        }
    }

    pub fn add_out_connection(&mut self, target: impl Into<String>) -> EntityConnection {
        let connection = EntityConnection::new(self.state.id.clone(), target);
        self.connections_out.push(connection.clone());
        connection
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn is_full(&self) -> bool {
        false
    }

    pub fn take_item(&mut self, entity: T) {
        let remaining_time = (self.draw_duration)();
        self.items.push(QueuedItem {
            entity,
            remaining_time,
        });
    }

    pub fn accept_item(&self, _entity: &T) -> bool {
        true
    }

    pub fn do_audit(&self) -> PerIndividualProcessorAudit {
        PerIndividualProcessorAudit {
            total_size: self.items.len(),
        }
    }

    pub fn run_time_step_with_sinks(
        &mut self,
        step_size: DesDecimal,
        sinks: &mut [&mut dyn PerIndividualSink<T>],
    ) -> DesResult<()> {
        let mut rng = || 0.5_f64;
        self.state.bump_time_step();
        self.advance_and_route(step_size, sinks, &mut rng)
    }

    pub fn run_time_step_with_sinks_and_rng<R>(
        &mut self,
        step_size: DesDecimal,
        sinks: &mut [&mut dyn PerIndividualSink<T>],
        rng: &mut R,
    ) -> DesResult<()>
    where
        R: RandomSource,
    {
        self.state.bump_time_step();
        self.advance_and_route(step_size, sinks, rng)
    }

    fn advance_and_route<R>(
        &mut self,
        step_size: DesDecimal,
        sinks: &mut [&mut dyn PerIndividualSink<T>],
        rng: &mut R,
    ) -> DesResult<()>
    where
        R: RandomSource,
    {
        let mut ready = Vec::new();
        let mut still_waiting = Vec::new();
        for mut item in self.items.drain(..) {
            item.remaining_time -= step_size;
            if item.remaining_time <= DesDecimal::ZERO {
                ready.push(item);
            } else {
                still_waiting.push(item);
            }
        }
        self.items = still_waiting;

        for item in ready {
            if let Some(item) = self.route_one(item, sinks, rng)? {
                self.items.insert(0, item);
            }
        }

        Ok(())
    }

    fn route_one<R>(
        &mut self,
        item: QueuedItem<T>,
        sinks: &mut [&mut dyn PerIndividualSink<T>],
        rng: &mut R,
    ) -> DesResult<Option<QueuedItem<T>>>
    where
        R: RandomSource,
    {
        let declared_connections = self.connections_out.clone();
        let ordered_connections = self
            .output_router
            .ordered_connections(&declared_connections, rng);
        let mut entity = Some(item.entity);

        for connection in ordered_connections {
            let Some(sink_index) = sinks
                .iter()
                .position(|sink| sink.id() == connection.target.as_str())
            else {
                continue;
            };
            let accepted = sinks[sink_index].accept_item(
                entity
                    .as_ref()
                    .expect("entity is present until a sink accepts it"),
            );
            if !accepted {
                continue;
            }
            let accepted_entity = entity
                .take()
                .expect("entity is present when an accepting sink consumes it");
            sinks[sink_index].take_item(accepted_entity)?;
            self.output_router
                .mark_accepted(&declared_connections, &connection);
            return Ok(None);
        }

        Ok(Some(QueuedItem {
            entity: entity.expect("unrouted item retains entity"),
            remaining_time: DesDecimal::ZERO,
        }))
    }
}

impl<T> Debug for PerIndividualProcessor<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerIndividualProcessor")
            .field("state", &self.state)
            .field("items", &self.items)
            .field("max_queue_size", &self.max_queue_size)
            .field("output_router", &self.output_router)
            .field("connections_out", &self.connections_out)
            .finish_non_exhaustive()
    }
}

impl<T> Entity for PerIndividualProcessor<T>
where
    T: Debug,
{
    fn state(&self) -> &EntityState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.state
    }

    fn run_time_step(&mut self, ctx: &TimeStepContext) -> DesResult<()> {
        let mut sinks: Vec<&mut dyn PerIndividualSink<T>> = Vec::new();
        let mut rng = || 0.5_f64;
        self.advance_and_route(ctx.step_size, &mut sinks, &mut rng)
    }

    fn graph_data(&self) -> GraphData {
        let mut properties: IndexMap<String, JsonValue> = IndexMap::new();
        properties.insert("processedCount".to_owned(), json!(self.items.len()));
        GraphData {
            entity_id: self.state.id.clone(),
            kind: "PerIndividualProcessor".to_owned(),
            properties,
        }
    }
}

impl<T> StationaryEntity<T> for PerIndividualProcessor<T>
where
    T: Debug,
{
    fn accept_item(&self, item: &T) -> bool {
        PerIndividualProcessor::accept_item(self, item)
    }

    fn take_item(&mut self, item: T) -> DesResult<()> {
        PerIndividualProcessor::take_item(self, item);
        Ok(())
    }
}
