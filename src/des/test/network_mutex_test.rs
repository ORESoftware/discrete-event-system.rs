//! Port of src/des/test/network-mutex-test.ts
//!
//! Drives the network-mutex stations: Station A obtains a lock from Station B
//! before releasing work to Station C. Groups 1, 2 and 5 run the full
//! simulation through `run_network_mutex_simulation`; Group 6 exercises the
//! generic stateful-token lineage helpers; the protocol channel-name constants
//! are pinned too.
//!
//! PORT NOTE: TS Group 3 (composite substation introspection +
//! `childStations()` / per-item `stateHistory` path) and Group 4 (constructing a
//! bare `lock-release` token literal and feeding it to a standalone
//! `NetworkMutexLockServiceStation`) reach into station internals whose Rust
//! handles are not part of the simple public surface used here, so those two
//! sub-groups are deferred.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::stateful_token::{
        make_stateful_token, spawn_stateful_child_token, transition_token, MakeStatefulTokenOpts,
        SpawnStatefulChildTokenOpts, TransitionTokenOpts,
    };
    use crate::des::general::network_mutex::{
        run_network_mutex_simulation, MutexSourceSpec, NetworkMutexLockServiceOpts,
        NetworkMutexSimulationOpts, NetworkMutexWorkerOpts, MUTEX_DONE_CHANNEL,
        MUTEX_GRANT_CHANNEL, MUTEX_RELEASE_CHANNEL, MUTEX_REQUEST_CHANNEL, MUTEX_WORK_CHANNEL,
    };

    fn sim_opts(
        count: usize,
        interarrival: usize,
        processing: usize,
        grant_delay: usize,
    ) -> NetworkMutexSimulationOpts {
        NetworkMutexSimulationOpts {
            source: Some(MutexSourceSpec {
                count,
                interarrival_ticks: interarrival,
                first_arrival_tick: None,
            }),
            worker: Some(NetworkMutexWorkerOpts {
                processing_ticks: processing,
            }),
            lock: Some(NetworkMutexLockServiceOpts {
                grant_delay_ticks: Some(grant_delay),
            }),
            max_ticks: Some(10_000),
        }
    }

    // Group 1 -- Station A obtains lock from Station B before releasing work to C.
    #[test]
    fn group1_lock_acquire_release_flow() {
        let r = run_network_mutex_simulation(sim_opts(5, 1, 3, 2));
        assert!(r.generated == 5 && r.completed == 5);

        let order: Vec<String> = r
            .completed_items
            .iter()
            .map(|x| x.item_id.clone())
            .collect();
        assert_eq!(order.join(","), "item-1,item-2,item-3,item-4,item-5");

        assert!(r.completed_items.iter().all(|x| matches!(
            &x.lock,
            Some(l) if l.granted_tick.is_some() && l.released_tick.is_some()
        )));
        assert_eq!(r.worker.child_requests_spawned, 5);
        assert_eq!(r.worker.child_releases_spawned, 5);
        assert!(r.lock.grant_count == 5 && r.lock.release_count == 5);
        assert!(r.invariant_violations.is_empty());
    }

    // Group 2 -- Contention builds queue and lock wait.
    #[test]
    fn group2_contention_builds_queue() {
        let r = run_network_mutex_simulation(sim_opts(8, 1, 4, 2));
        assert!(r.worker.max_queue > 1, "maxQueue={}", r.worker.max_queue);
        assert!(r.worker.mean_lock_wait_ticks >= 2.0);
        assert!(r.worker.mean_time_in_system_ticks > 4.0);
        assert!(r.lock.utilization > 0.0 && r.lock.utilization <= 1.0);
    }

    // Group 5 -- Faster arrivals have worse queueing than slower arrivals.
    #[test]
    fn group5_faster_arrivals_worse_queueing() {
        let fast = run_network_mutex_simulation(sim_opts(8, 1, 4, 2));
        let slow = run_network_mutex_simulation(sim_opts(8, 8, 4, 2));
        assert!(fast.worker.max_queue > slow.worker.max_queue);
        assert!(fast.worker.mean_queue_wait_ticks > slow.worker.mean_queue_wait_ticks);
        assert_eq!(slow.completed, slow.generated);
    }

    // Group 3 (partial) -- worker/lock channels are named protocol channels.
    #[test]
    fn group3_protocol_channel_names() {
        let joined = [
            MUTEX_WORK_CHANNEL,
            MUTEX_REQUEST_CHANNEL,
            MUTEX_GRANT_CHANNEL,
            MUTEX_RELEASE_CHANNEL,
            MUTEX_DONE_CHANNEL,
        ]
        .join("|");
        assert_eq!(joined, "work|lock-request|lock-grant|lock-release|done");
    }

    // Group 6 -- Generic smart-movable lineage helpers.
    #[test]
    fn group6_stateful_token_lineage() {
        let mut parent = make_stateful_token(MakeStatefulTokenOpts {
            kind: "parent".to_string(),
            token_id: "p1".to_string(),
            initial_state: "new".to_string(),
            tick: 0.0,
            station_id: "src".to_string(),
            event: None,
            detail: None,
        });
        transition_token(
            &mut parent,
            "waiting".to_string(),
            TransitionTokenOpts {
                tick: 1.0,
                station_id: "A".to_string(),
                event: "queued".to_string(),
                detail: None,
            },
        );
        let child = spawn_stateful_child_token(
            &parent.lineage,
            SpawnStatefulChildTokenOpts {
                kind: "child-request".to_string(),
                token_id: "c1".to_string(),
                initial_state: "spawned".to_string(),
                tick: 2.0,
                station_id: "A".to_string(),
                event: None,
                detail: None,
            },
        );

        assert_eq!(
            child.lineage.parent_token_id.as_deref(),
            Some(parent.lineage.token_id.as_str())
        );
        assert_eq!(child.lineage.root_token_id, parent.lineage.root_token_id);
        assert_eq!(child.lineage.generation, 1);

        let history = parent.state_history.as_ref().unwrap();
        assert_eq!(history[1].from.as_deref(), Some("new"));
        assert_eq!(history[1].to, "waiting");
    }
}
