//! Port of `src/des/main-network-mutex.ts`.
//!
//! Thin runner: distributed-mutex DES (source / worker / lock) with invariant
//! checks; prints a summary plus completion order and the first trace events.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`]; `process.env` params (ITEMS, INTERARRIVAL,
//!     PROCESSING_TICKS, GRANT_DELAY_TICKS) → `std::env::var`.
//!   - delegates to `general::network_mutex::run_network_mutex_simulation`.

use crate::des::general::network_mutex::{
    run_network_mutex_simulation, MutexSourceSpec, NetworkMutexLockServiceOpts,
    NetworkMutexSimulationOpts, NetworkMutexWorkerOpts,
};

/// `Number(process.env.KEY ?? default)` for integer env vars.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `fmt(n)` — finite numbers to 3 decimals, else `"n/a"`.
fn fmt(n: f64) -> String {
    if n.is_finite() {
        format!("{:.3}", n)
    } else {
        "n/a".to_string()
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let result = run_network_mutex_simulation(NetworkMutexSimulationOpts {
        source: Some(MutexSourceSpec {
            count: env_usize("ITEMS", 10),
            interarrival_ticks: env_usize("INTERARRIVAL", 1),
            first_arrival_tick: None,
        }),
        worker: Some(NetworkMutexWorkerOpts {
            processing_ticks: env_usize("PROCESSING_TICKS", 4),
        }),
        lock: Some(NetworkMutexLockServiceOpts {
            grant_delay_ticks: Some(env_usize("GRANT_DELAY_TICKS", 2)),
        }),
        max_ticks: None,
    });

    println!("Network mutex DES");
    println!("=================");
    println!("generated:              {}", result.generated);
    println!("completed:              {}", result.completed);
    println!("total ticks:            {}", result.total_ticks);
    println!("worker max queue:       {}", result.worker.max_queue);
    println!(
        "mean queue wait:        {} ticks",
        fmt(result.worker.mean_queue_wait_ticks)
    );
    println!(
        "mean lock wait:         {} ticks",
        fmt(result.worker.mean_lock_wait_ticks)
    );
    println!(
        "mean time in system:    {} ticks",
        fmt(result.worker.mean_time_in_system_ticks)
    );
    println!(
        "child lock requests:    {}",
        result.worker.child_requests_spawned
    );
    println!(
        "child lock releases:    {}",
        result.worker.child_releases_spawned
    );
    println!(
        "lock grants/releases:   {}/{}",
        result.lock.grant_count, result.lock.release_count
    );
    println!("lock max wait queue:    {}", result.lock.max_wait_queue);
    println!("lock utilization:       {}", fmt(result.lock.utilization));
    println!(
        "invariants:             {}",
        if result.invariant_violations.is_empty() {
            "ok".to_string()
        } else {
            result.invariant_violations.join("; ")
        }
    );

    println!();
    println!("Completion order:");
    let order: Vec<String> = result
        .completed_items
        .iter()
        .map(|x| x.item_id.clone())
        .collect();
    println!("{}", order.join(" -> "));

    println!();
    println!("First trace events:");
    for e in result.trace.iter().take(24) {
        let item = e
            .item_id
            .as_ref()
            .map(|i| format!(" {}", i))
            .unwrap_or_default();
        let child = e
            .child_token_id
            .as_ref()
            .map(|c| format!(" child={}", c))
            .unwrap_or_default();
        let detail = e
            .detail
            .as_ref()
            .map(|d| format!(" ({})", d))
            .unwrap_or_default();
        println!(
            "  t={:>3} {:<14} {}{}{}{}",
            e.tick,
            e.station_id,
            e.event.as_str(),
            item,
            child,
            detail
        );
    }
}
