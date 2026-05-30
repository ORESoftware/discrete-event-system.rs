//! Rust port of `src/des/general/general.ts`.

use crate::core::{short_uuid, DesDecimal, DesError, DesResult, JsonValue, RandomSource};
use crate::migration::MigrationFile;
use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use rust_decimal::Decimal;
use serde::Serialize;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/general.ts",
    "src/des/general/general.rs",
    &[
        "DESSet/DESMap collapse to IndexSet/IndexMap aliases for deterministic iteration.",
        "BigNumber helpers use rust_decimal at the boundary.",
        "Random helpers accept injected RNG sources.",
    ],
    &[
        "DESMap",
        "DESSet",
        "HasComputedProperties",
        "bgn",
        "deJSON",
        "fisherYatesShuffle",
        "getReasonableU",
        "getReasonableUNative",
        "getShortUUID",
        "makeError",
        "sendRaw",
    ],
);

pub type DESSet<T> = IndexSet<T>;
pub type DESMap<K, V> = IndexMap<K, V>;

pub trait HasComputedProperties {
    fn computed_properties(&self) -> JsonValue;
}

pub fn send_raw<T: Serialize>(data: &T) -> DesResult<String> {
    serde_json::to_string(data).map_err(|err| DesError::InvalidState {
        context: "send_raw",
        message: err.to_string(),
    })
}

pub fn de_json(input: &str) -> DesResult<JsonValue> {
    serde_json::from_str(input).map_err(|err| DesError::InvalidState {
        context: "de_json",
        message: err.to_string(),
    })
}

pub fn make_error(message: impl Into<String>) -> DesError {
    DesError::InvalidState {
        context: "des",
        message: message.into(),
    }
}

pub fn get_short_uuid() -> String {
    short_uuid()
}

pub fn bgn<N>(value: N) -> DesDecimal
where
    N: Into<Decimal>,
{
    value.into()
}

pub fn fisher_yates_shuffle<T, R>(items: &mut [T], rng: &mut R)
where
    R: RandomSource,
{
    for i in (1..items.len()).rev() {
        let j = ((rng.next_f64() * ((i + 1) as f64)).floor() as usize).min(i);
        items.swap(i, j);
    }
}

pub fn fisher_yates_shuffle_rand<T, R>(items: &mut [T], rng: &mut R)
where
    R: Rng + ?Sized,
{
    for i in (1..items.len()).rev() {
        let j = rng.gen_range(0..=i);
        items.swap(i, j);
    }
}

pub fn get_reasonable_u<R>(rng: &mut R) -> DesDecimal
where
    R: RandomSource,
{
    Decimal::from_f64_retain(rng.next_f64()).unwrap_or(Decimal::ZERO)
}

pub fn get_reasonable_u_native<R>(rng: &mut R) -> f64
where
    R: RandomSource,
{
    rng.next_f64()
}

pub fn identity<T>(value: T) -> T {
    value
}
