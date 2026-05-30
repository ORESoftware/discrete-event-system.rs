//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/time-stepped-station.ts`
//! Rust target: `src/des/general/time_stepped_station.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/time-stepped-station.ts",
    "src/des/general/time_stepped_station.rs",
    &["RUST MIGRATION: Target module `src/des/general/time_stepped_station.rs`.", "RUST MIGRATION: Replace abstract class inheritance with traits for time-stepped, buffered, routed, bidirectional, and synchronous-dataflow station behavior.", "RUST MIGRATION: Shared queues/buffers should be embedded structs (`VecDeque<T>` where FIFO), while payload types become generic type parameters with trait bounds.", "RUST MIGRATION: Convert `SynchronousDataflowConnection` to a `serde` struct if persisted, otherwise keep it as an internal connection descriptor.", "RUST MIGRATION: Methods that can fail due to missing routes, backpressure, or invalid connections should return `Result` instead of throwing or silently dropping."],
    &["BidirectionalTimeSteppedStation", "BufferedTimeSteppedStation", "RoutedTimeSteppedStation", "SynchronousDataflowConnection", "SynchronousDataflowStation", "TimeSteppedStation"],
);
