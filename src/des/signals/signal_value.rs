//! Rust port of `src/des/signals/signal-value.ts`.

use crate::core::{short_uuid, DesDecimal, DesResult, Entity, EntityState, TimeStepContext};
use crate::migration::MigrationFile;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/signals/signal-value.ts",
    "src/des/signals/signal_value.rs",
    &[
        "Signal values are concrete structs implementing SignalValueLike.",
        "Unity/zero constructors become typed default constructors.",
    ],
    &[
        "AbstractSignalValue",
        "SignalValue",
        "SignalValueUnity",
        "SignalValueZero",
    ],
);

pub trait SignalValueLike {
    type Value: Clone + Debug + Serialize;

    fn value(&self) -> Self::Value;
    fn set_value(&mut self, value: Self::Value);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalValue<V>
where
    V: Clone,
{
    pub state: EntityState,
    pub value: V,
}

impl<V> SignalValue<V>
where
    V: Clone,
{
    pub fn new(value: V) -> Self {
        Self {
            state: EntityState::new(short_uuid()),
            value,
        }
    }

    pub fn with_id(id: impl Into<String>, value: V) -> Self {
        Self {
            state: EntityState::new(id),
            value,
        }
    }
}

impl<V> Entity for SignalValue<V>
where
    V: Clone + Debug + Serialize,
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
}

impl<V> SignalValueLike for SignalValue<V>
where
    V: Clone + Debug + Serialize,
{
    type Value = V;

    fn value(&self) -> Self::Value {
        self.value.clone()
    }

    fn set_value(&mut self, value: Self::Value) {
        self.value = value;
    }
}

pub type AbstractSignalValue<V> = SignalValue<V>;

pub fn signal_value_unity() -> SignalValue<DesDecimal> {
    SignalValue::new(Decimal::ONE)
}

pub fn signal_value_zero() -> SignalValue<DesDecimal> {
    SignalValue::new(Decimal::ZERO)
}

pub type SignalValueUnity = SignalValue<DesDecimal>;
pub type SignalValueZero = SignalValue<DesDecimal>;
