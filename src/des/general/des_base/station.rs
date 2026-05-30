//! Rust port of `src/des/general/des-base/station.ts`.

use super::validation::{format_validation_report, run_validators, ValidationCheck, Validator};
use crate::migration::MigrationFile;
use indexmap::IndexMap;
use std::{any::Any, fmt::Debug, marker::PhantomData};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/station.ts",
    "src/des/general/des_base/station.rs",
    &[
        "Token and DESRunLoopEntity are traits.",
        "DESStation is a trait over StationCore<S>, mirroring the TS base class.",
        "Channel inbox maps use IndexMap for deterministic snapshots.",
        "Validators are boxed trait objects typed to the concrete station.",
    ],
    &[
        "ChannelName",
        "DEFAULT_CHANNEL",
        "DESRunLoopEntity",
        "DESStation",
        "HasRunTimeStep",
        "Token",
    ],
);

pub trait Token: Any + Debug {
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T> Token for T
where
    T: Any + Debug,
{
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

pub trait HasRunTimeStep {
    fn run_time_step(&mut self);
}

pub type ChannelName = String;
pub const DEFAULT_CHANNEL: &str = "default";

pub trait DESRunLoopEntity: HasRunTimeStep {
    fn id(&self) -> Option<&str> {
        None
    }

    fn assert_preconditions(&self) -> Result<(), String> {
        Ok(())
    }

    fn has_work(&self) -> bool {
        true
    }

    fn on_finalize(&mut self) {}

    fn num_validators(&self) -> usize {
        0
    }

    fn run_validation(&self) -> Vec<ValidationCheck> {
        Vec::new()
    }
}

pub struct StationCore<S> {
    id: String,
    inboxes: IndexMap<ChannelName, Vec<Box<dyn Token>>>,
    validators: Vec<Box<dyn Validator<S>>>,
    _station: PhantomData<fn(&S)>,
}

impl<S> StationCore<S> {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            inboxes: IndexMap::new(),
            validators: Vec::new(),
            _station: PhantomData,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn take<T>(&mut self, token: T, channel: impl Into<ChannelName>)
    where
        T: Token + 'static,
    {
        self.inboxes
            .entry(channel.into())
            .or_default()
            .push(Box::new(token));
    }

    pub fn take_default<T>(&mut self, token: T)
    where
        T: Token + 'static,
    {
        self.take(token, DEFAULT_CHANNEL);
    }

    pub fn drain<T>(&mut self, channel: impl Into<ChannelName>) -> Vec<T>
    where
        T: Token + 'static,
    {
        let channel = channel.into();
        let tokens = self.inboxes.insert(channel, Vec::new()).unwrap_or_default();
        tokens
            .into_iter()
            .filter_map(|token| token.into_any().downcast::<T>().ok().map(|boxed| *boxed))
            .collect()
    }

    pub fn drain_default<T>(&mut self) -> Vec<T>
    where
        T: Token + 'static,
    {
        self.drain(DEFAULT_CHANNEL)
    }

    pub fn inbox_size(&self, channel: impl AsRef<str>) -> usize {
        self.inboxes
            .get(channel.as_ref())
            .map(|items| items.len())
            .unwrap_or(0)
    }

    pub fn has_inbox_work(&self) -> bool {
        self.inboxes.values().any(|items| !items.is_empty())
    }

    pub fn inbox_sizes(&self) -> IndexMap<ChannelName, usize> {
        self.inboxes
            .iter()
            .map(|(channel, items)| (channel.clone(), items.len()))
            .collect()
    }

    pub fn add_validator(&mut self, validator: Box<dyn Validator<S>>) {
        self.validators.push(validator);
    }

    pub fn num_validators(&self) -> usize {
        self.validators.len()
    }

    pub fn run_validation(&self, station: &S) -> Vec<ValidationCheck> {
        run_validators(station, &self.validators)
    }

    pub fn validation_report(&self, station: &S) -> String {
        format_validation_report(&self.run_validation(station))
    }
}

pub trait DESStation: DESRunLoopEntity + Sized {
    fn core(&self) -> &StationCore<Self>;
    fn core_mut(&mut self) -> &mut StationCore<Self>;

    fn station_id(&self) -> &str {
        self.core().id()
    }

    fn take<T>(&mut self, token: T, channel: impl Into<ChannelName>)
    where
        T: Token + 'static,
    {
        self.core_mut().take(token, channel);
    }

    fn take_default<T>(&mut self, token: T)
    where
        T: Token + 'static,
    {
        self.core_mut().take_default(token);
    }

    fn drain<T>(&mut self, channel: impl Into<ChannelName>) -> Vec<T>
    where
        T: Token + 'static,
    {
        self.core_mut().drain(channel)
    }

    fn drain_default<T>(&mut self) -> Vec<T>
    where
        T: Token + 'static,
    {
        self.core_mut().drain_default()
    }

    fn inbox_size(&self, channel: impl AsRef<str>) -> usize {
        self.core().inbox_size(channel)
    }

    fn inbox_sizes(&self) -> IndexMap<ChannelName, usize> {
        self.core().inbox_sizes()
    }

    fn add_validator(&mut self, validator: Box<dyn Validator<Self>>) -> &mut Self {
        self.core_mut().add_validator(validator);
        self
    }

    fn validation_report(&self) -> String {
        self.core().validation_report(self)
    }
}
