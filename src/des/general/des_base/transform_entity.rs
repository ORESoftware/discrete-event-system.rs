//! Rust port of `src/des/general/des-base/transform-entity.ts`.

use crate::core::{
    DesError, DesResult, Entity, EntityConnection, EntityState, GraphData, PureTransform,
    QueueState, StationaryEntity, TimeStepContext,
};
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/transform-entity.ts",
    "src/des/general/des_base/transform_entity.rs",
    &[
        "Plain TypeScript functions are Rust PureTransform implementors.",
        "Transform entities compose EntityState plus typed queues.",
        "Channel errors are Result values instead of thrown exceptions.",
    ],
    &[
        "FunctionEntity",
        "PureTransform",
        "PureTransformEntity",
        "TransformContext",
        "TransformEntity",
        "TransformResult",
    ],
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformContext {
    pub channel: Option<String>,
    pub tick: u64,
}

impl From<&TimeStepContext> for TransformContext {
    fn from(value: &TimeStepContext) -> Self {
        Self {
            channel: None,
            tick: value.tick,
        }
    }
}

pub type TransformResult<T> = DesResult<T>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformEntity<I, O> {
    pub state: EntityState,
    pub queue: QueueState<I>,
    pub outputs: Vec<EntityConnection>,
    pub last_output: Option<O>,
}

impl<I, O> TransformEntity<I, O> {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            state: EntityState::new(id),
            queue: QueueState::new(None),
            outputs: Vec::new(),
            last_output: None,
        }
    }
}

impl<I, O> Entity for TransformEntity<I, O>
where
    I: Debug,
    O: Debug,
{
    fn state(&self) -> &EntityState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.state
    }

    fn graph_data(&self) -> GraphData {
        GraphData {
            entity_id: self.state.id.clone(),
            kind: "TransformEntity".to_owned(),
            properties: Default::default(),
        }
    }
}

impl<I, O> StationaryEntity<I> for TransformEntity<I, O>
where
    I: Debug,
    O: Debug,
{
    fn take_item(&mut self, item: I) -> DesResult<()> {
        self.queue.push(item)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PureTransformEntity<T, I, O> {
    pub entity: TransformEntity<I, O>,
    pub transform: T,
}

impl<T, I, O> PureTransformEntity<T, I, O> {
    pub fn new(id: impl Into<String>, transform: T) -> Self {
        Self {
            entity: TransformEntity::new(id),
            transform,
        }
    }
}

impl<T, I, O> Entity for PureTransformEntity<T, I, O>
where
    T: PureTransform<I, O> + Debug,
    I: Debug,
    O: Clone + Debug,
{
    fn state(&self) -> &EntityState {
        &self.entity.state
    }

    fn state_mut(&mut self) -> &mut EntityState {
        &mut self.entity.state
    }

    fn run_time_step(&mut self, _ctx: &TimeStepContext) -> DesResult<()> {
        while let Some(input) = self.entity.queue.pop() {
            self.entity.last_output = Some(self.transform.transform(input)?);
        }
        Ok(())
    }
}

impl<T, I, O> StationaryEntity<I> for PureTransformEntity<T, I, O>
where
    T: PureTransform<I, O> + Debug,
    I: Debug,
    O: Clone + Debug,
{
    fn take_item(&mut self, item: I) -> DesResult<()> {
        self.entity.queue.push(item)
    }
}

#[derive(Clone)]
pub struct FunctionEntity<I, O, F>
where
    F: Fn(I) -> DesResult<O>,
{
    pub f: F,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I, O, F> FunctionEntity<I, O, F>
where
    F: Fn(I) -> DesResult<O>,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<I, O, F> Debug for FunctionEntity<I, O, F>
where
    F: Fn(I) -> DesResult<O>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionEntity").finish_non_exhaustive()
    }
}

impl<I, O, F> PureTransform<I, O> for FunctionEntity<I, O, F>
where
    F: Fn(I) -> DesResult<O>,
{
    fn transform(&self, input: I) -> DesResult<O> {
        (self.f)(input).map_err(|err| match err {
            DesError::Unsupported { .. }
            | DesError::InvalidState { .. }
            | DesError::Validation { .. } => err,
        })
    }
}
