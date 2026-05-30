//! Port of src/des/test/internal-solver-network-test.ts
//!
//! Tests for the internal solver networks (`general/internal-solver-network`):
//! shortest-path as a DES solver network, exact knapsack DP + SA, TSP GA / SA /
//! Held-Karp stations, and the wall-clock checker station.
//!
//! PORT NOTE: TS group [5] ("Registry, JSON input, observability, and
//! animation") exercises `general/des-registry` (`getModel`, `runFromSpec`,
//! `runFromJsonFile`), JSON file I/O, and the animation frame writer. The
//! registry / model-spec runner is not yet ported to Rust, so group [5] is
//! deferred. Groups [1]–[4] call `run_internal_solver_network` directly and are
//! ported faithfully.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::genetic_tsp::{build_pentagon_tsp, held_karp_exact, InitMode};
    use crate::des::general::internal_solver_network::{
        run_internal_solver_network, InternalSolverKind, InternalSolverRunParams,
        InternalSolverStatus, KnapsackParams, ShortestPathAlgorithm, ShortestPathBuiltin,
        ShortestPathSolverParams, SolverBestState, TSPGAOptionsPartial, TSPSAOptionsPartial,
        TSPSolverParams, TspBuiltin,
    };
    use crate::des::general::sa_des::CoolingSchedule;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * f64::max(1.0, f64::max(a.abs(), b.abs()))
    }

    fn base(kind: InternalSolverKind) -> InternalSolverRunParams {
        InternalSolverRunParams {
            kind,
            time_limit_ms: None,
            max_ticks: None,
            check_every_ticks: None,
            shortest_path: None,
            knapsack: None,
            tsp: None,
        }
    }

    // [1] Shortest path as a DES solver network.
    #[test]
    fn shortest_path_solver_network() {
        let r = run_internal_solver_network(InternalSolverRunParams {
            shortest_path: Some(ShortestPathSolverParams {
                algorithm: ShortestPathAlgorithm::Dijkstra,
                source: 0,
                builtin: Some(ShortestPathBuiltin::SmallChain),
                graph: None,
                random_graph: None,
            }),
            ..base(InternalSolverKind::ShortestPath)
        });
        assert_eq!(r.status, InternalSolverStatus::Complete);
        match &r.best.best_state {
            SolverBestState::ShortestPath { distance, .. } => {
                assert!(close(distance[4], 6.0, 1e-8), "d4={}", distance[4]);
            }
            other => panic!("expected shortest-path best state, got {other:?}"),
        }
        assert!(r
            .network
            .stationary_entities
            .iter()
            .any(|n| n.id == "wall-clock-checker"));
        assert!(r
            .network
            .stationary_entities
            .iter()
            .any(|n| n.id == "solution-sink"));
        assert!(r
            .network
            .edges
            .iter()
            .any(|e| e.moving_entity == "SolverSolutionToken"));
        assert!(
            r.validation.iter().all(|c| c.passed),
            "{}",
            r.validation
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // [2] Knapsack exact DP and simulated annealing.
    #[test]
    fn knapsack_dp_and_sa() {
        let dp = run_internal_solver_network(InternalSolverRunParams {
            knapsack: Some(KnapsackParams {
                values: vec![60.0, 100.0, 120.0],
                weights: vec![10.0, 20.0, 30.0],
                capacity: 50.0,
                seed: None,
                max_iterations: None,
                cooling: None,
                stall_limit: None,
                penalty: None,
            }),
            ..base(InternalSolverKind::KnapsackDp)
        });
        match &dp.best.best_state {
            SolverBestState::Knapsack {
                value,
                weight,
                capacity,
                ..
            } => {
                assert_eq!(*value, 220.0);
                assert!(*weight <= *capacity && dp.best.feasible, "weight={weight}");
            }
            other => panic!("expected knapsack best state, got {other:?}"),
        }
        assert!(dp.trace.len() >= 3 && dp.trace.iter().any(|t| t.done));

        let sa = run_internal_solver_network(InternalSolverRunParams {
            knapsack: Some(KnapsackParams {
                values: vec![60.0, 100.0, 120.0],
                weights: vec![10.0, 20.0, 30.0],
                capacity: 50.0,
                seed: Some(4),
                max_iterations: Some(80),
                cooling: Some(CoolingSchedule::Geometric {
                    t0: 30.0,
                    alpha: 0.97,
                    t_min: Some(1e-6),
                }),
                stall_limit: None,
                penalty: None,
            }),
            ..base(InternalSolverKind::KnapsackSa)
        });
        match &sa.best.best_state {
            SolverBestState::Knapsack {
                value,
                weight,
                capacity,
                ..
            } => {
                assert!(*weight <= *capacity && sa.best.feasible, "weight={weight}");
                assert!(*value >= 160.0, "value={value}");
            }
            other => panic!("expected knapsack best state, got {other:?}"),
        }
    }

    // [3] Traveling salesman internal solvers.
    #[test]
    fn tsp_internal_solvers() {
        let ga = run_internal_solver_network(InternalSolverRunParams {
            tsp: Some(TSPSolverParams {
                builtin: Some(TspBuiltin::Pentagon),
                n: Some(7),
                seed: Some(7),
                ga: Some(TSPGAOptionsPartial {
                    pop_size: Some(28),
                    num_generations: Some(25),
                    seed: Some(9),
                    init: Some(InitMode::NearestNeighbor),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..base(InternalSolverKind::TspGa)
        });
        assert_eq!(ga.status, InternalSolverStatus::Complete);
        assert!(ga.best.feasible);
        match &ga.best.best_state {
            SolverBestState::Tour { length, .. } => assert!(length.is_finite()),
            other => panic!("expected tour best state, got {other:?}"),
        }

        let sa = run_internal_solver_network(InternalSolverRunParams {
            tsp: Some(TSPSolverParams {
                builtin: Some(TspBuiltin::Pentagon),
                n: Some(7),
                seed: Some(5),
                sa: Some(TSPSAOptionsPartial {
                    max_iterations: Some(80),
                    seed: Some(5),
                    cooling: Some(CoolingSchedule::Geometric {
                        t0: 100.0,
                        alpha: 0.97,
                        t_min: Some(1e-6),
                    }),
                    init: Some(InitMode::NearestNeighbor),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..base(InternalSolverKind::TspSa)
        });
        assert_eq!(sa.status, InternalSolverStatus::Complete);
        assert!(sa.best.feasible);
        match &sa.best.best_state {
            SolverBestState::Tour { length, .. } => assert!(length.is_finite()),
            other => panic!("expected tour best state, got {other:?}"),
        }

        // Held-Karp station must match the standalone exact solver.
        let inst = build_pentagon_tsp(5, 50.0);
        let exact = held_karp_exact(&inst);
        let hk = run_internal_solver_network(InternalSolverRunParams {
            tsp: Some(TSPSolverParams {
                builtin: Some(TspBuiltin::Pentagon),
                n: Some(5),
                ..Default::default()
            }),
            ..base(InternalSolverKind::TspHeldKarp)
        });
        match &hk.best.best_state {
            SolverBestState::Tour { length, .. } => {
                assert!(
                    close(*length, exact.length, 1e-10),
                    "best={length} exact={}",
                    exact.length
                );
            }
            other => panic!("expected tour best state, got {other:?}"),
        }
    }

    // [4] Wall-clock checker station: a zero budget stops on the first check.
    #[test]
    fn wall_clock_checker_zero_budget() {
        let timed = run_internal_solver_network(InternalSolverRunParams {
            time_limit_ms: Some(0.0),
            check_every_ticks: Some(1),
            tsp: Some(TSPSolverParams {
                builtin: Some(TspBuiltin::Pentagon),
                n: Some(6),
                seed: Some(6),
                sa: Some(TSPSAOptionsPartial {
                    max_iterations: Some(1000),
                    seed: Some(6),
                    cooling: Some(CoolingSchedule::Geometric {
                        t0: 100.0,
                        alpha: 0.999,
                        t_min: Some(1e-9),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..base(InternalSolverKind::TspSa)
        });
        assert_eq!(timed.status, InternalSolverStatus::TimeLimit);
        assert!(timed.stop_signals.len() >= 1 && timed.wall_clock.expired);
    }
}
