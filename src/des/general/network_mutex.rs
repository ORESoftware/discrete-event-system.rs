//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/network-mutex.ts`
//! Rust target: `src/des/general/network_mutex.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/network-mutex.ts",
    "src/des/general/network_mutex.rs",
    &["RUST MIGRATION: Target module `src/des/general/network_mutex.rs`.", "RUST MIGRATION: Convert work/child state string unions to enums and all token/spec/stats/result interfaces to `serde` structs.", "RUST MIGRATION: Port source, lock-service, queue/processor substations, worker, and sink classes as structs implementing `DESStation`/`CompositeDESStation` traits.", "RUST MIGRATION: Model request/grant/release child tokens as typed structs; use `VecDeque` for FIFOs and `HashMap`/`HashSet` for holder and pending indexes.", "RUST MIGRATION: Convert invalid lock or routing cases to `Result` errors; keep `buildNetworkMutexStations`/`runNetworkMutexSimulation` as graph builder/free runner functions."],
    &["LockGrantToken", "LockReleaseToken", "LockRequestToken", "MUTEX_DONE_CHANNEL", "MUTEX_GRANT_CHANNEL", "MUTEX_RELEASE_CHANNEL", "MUTEX_REQUEST_CHANNEL", "MUTEX_WORK_CHANNEL", "MutexChildState", "MutexChildToken", "MutexCompletionSinkStation", "MutexSourceSpec", "MutexWorkItem", "MutexWorkSourceStation", "MutexWorkState", "NetworkMutexLockServiceOpts", "NetworkMutexLockServiceStation", "NetworkMutexLockStats", "NetworkMutexSimulationOpts", "NetworkMutexSimulationResult", "NetworkMutexTraceEvent", "NetworkMutexWorkerOpts", "NetworkMutexWorkerStation", "NetworkMutexWorkerStats", "buildNetworkMutexStations", "runNetworkMutexSimulation"],
);
