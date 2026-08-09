//! Test-only cross-module synchronization helpers.
//!
//! `std::env::{set_var, var, remove_var}` operate on a single process-global
//! table and are NOT thread-safe (this is exactly why `set_var` becomes
//! `unsafe` in the 2024 edition). The `external_*_reference` Python-bridge
//! tests mutate that table, and they share generic variable names — most
//! notably `ORES_EXTERNAL_REFERENCE_FORCE_PYTHON` — across more than twenty
//! modules that all compile into the *same* `des_engine` lib-test binary.
//!
//! Each of those modules used to declare its *own* private `Mutex`, which only
//! serialized that one module's tests. Under a full parallel `cargo test` run,
//! module A (holding lock A) still raced module B (holding lock B) on the
//! shared env table, producing flaky reads; and when a test panicked while
//! holding its private lock, that `Mutex` was poisoned and every *other* test
//! in the module then failed with `PoisonError` instead of running — one
//! genuine failure masquerading as a cascade.
//!
//! [`ENV_LOCK`] is the single lock all of those tests must share. It is also
//! poison-tolerant: the protected critical sections restore their environment
//! through RAII guards regardless of how they exit, so the env table is never
//! actually left corrupt — recovering a poisoned guard is the correct,
//! non-cascading behavior.

#![cfg(test)]

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A process-global, poison-tolerant lock serializing every test that mutates
/// the shared environment. See the module docs for why a per-module `Mutex` is
/// insufficient.
pub struct EnvLock {
    inner: Mutex<()>,
}

impl EnvLock {
    const fn new() -> Self {
        Self {
            inner: Mutex::new(()),
        }
    }

    /// Acquire the lock, recovering from poisoning instead of propagating it.
    ///
    /// Returns a `Result` (always `Ok`) rather than the bare guard so the
    /// existing call sites — `ENV_LOCK.lock().expect("lock env guard")` — keep
    /// compiling unchanged; the poison case is absorbed here, never surfaced to
    /// the caller, so one failing test cannot cascade `PoisonError` into the
    /// rest of the suite.
    pub fn lock(&self) -> Result<MutexGuard<'_, ()>, PoisonError<MutexGuard<'_, ()>>> {
        Ok(self.inner.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// The one shared env lock — see [`EnvLock`]. Every `external_*_reference`
/// test module aliases this under its own former lock name.
pub static ENV_LOCK: EnvLock = EnvLock::new();

/// Restores one process-global environment variable when it leaves scope.
///
/// Callers must hold [`ENV_LOCK`] for the guard's entire lifetime so parallel
/// tests cannot observe the temporary value or overwrite its saved state.
pub struct EnvVarGuard {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    pub fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(previous) => std::env::set_var(self.key, previous),
            None => std::env::remove_var(self.key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_from_poison_instead_of_cascading() {
        // Simulate a test panicking while holding the lock: the Mutex is
        // poisoned, but the next acquisition must still succeed.
        let _ = std::panic::catch_unwind(|| {
            let _guard = ENV_LOCK.lock().expect("first acquire");
            panic!("poison the lock");
        });
        // Would be Err(PoisonError) with a plain Mutex; here it recovers.
        let guard = ENV_LOCK.lock();
        assert!(guard.is_ok(), "poisoned lock should recover, not cascade");
    }

    #[test]
    fn env_guard_restores_the_previous_value() {
        let _lock = ENV_LOCK.lock().expect("lock environment");
        let key = "DES_TEST_SUPPORT_ENV_GUARD";
        let previous = std::env::var_os(key);
        {
            let _guard = EnvVarGuard::set(key, "temporary");
            assert_eq!(
                std::env::var_os(key).as_deref(),
                Some(OsStr::new("temporary"))
            );
        }
        assert_eq!(std::env::var_os(key), previous);
    }
}
