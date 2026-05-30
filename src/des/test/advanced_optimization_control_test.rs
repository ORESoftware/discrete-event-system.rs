//! Port of src/des/test/advanced-optimization-control-test.ts
//!
//! Tests for the advanced optimization metaheuristics
//! (`general/advanced-optimization-models`): particle swarm, ant-colony TSP,
//! map-coloring CSP, MAX-SAT local search, SDP max-cut relaxation, and the
//! Pareto-portfolio archive; plus the advanced decision/control games
//! (`general/advanced-control-models`): H-infinity robust control and the
//! pursuit/evasion differential game. The Pareto archive station is exercised
//! directly via the DES runner.
//!
//! PORT NOTE: the TS "registry coverage" section uses `general/des-registry`
//! (`getModel`, `runFromSpec`), which is not yet ported to Rust; it is deferred.
//! All model `run*` calls are ported faithfully. Stochastic metaheuristics are
//! seeded so the asserted properties are reproducible. Where the TS forced an
//! invalid `NaN` dimension (impossible for a Rust `usize`), the equivalent
//! `dimension = 0` precondition violation is used instead.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::general::advanced_control_models::{
        run_h_infinity_robust_control, run_pursuit_evasion_game, HInfinityRobustControlParams,
        PursuitEvasionGameParams,
    };
    use crate::des::general::advanced_optimization_models::{
        pareto_front_is_nondominated, run_ant_colony_tsp, run_map_coloring_csp,
        run_max_sat_local_search, run_pareto_portfolio, run_particle_swarm,
        run_sdp_max_cut_relaxation, AntColonyTSPParams, ContinuousObjectiveName,
        MapColoringCSPParams, MaxSATParams, ParetoPortfolioParams, ParticleSwarmParams, Point2,
        SDPMaxCutParams,
    };
    use crate::des::general::des_base::advanced_optimization::{
        ParetoArchiveStation, ParetoCandidateToken,
    };
    use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
    use crate::des::general::des_base::station::StationRef;

    fn has(v: &[String], s: &str) -> bool {
        v.iter().any(|x| x == s)
    }

    // -- particle-swarm --
    #[test]
    fn particle_swarm() {
        let r = run_particle_swarm(ParticleSwarmParams {
            objective: Some(ContinuousObjectiveName::Sphere),
            dimension: Some(3),
            particles: Some(32),
            iterations: Some(120),
            seed: Some(11),
            ..Default::default()
        });
        assert!(r.best_value < 1e-8, "best={}", r.best_value);
        assert_eq!(r.iterations, 120);
        assert!(has(&r.topology.movables, "NumericSwarmParticle"));
        assert_eq!(r.topology.stations[0].as_str(), "particle-swarm-source");
        assert!(has(&r.topology.stations, "particle-swarm-result-sink"));
    }

    #[test]
    #[should_panic]
    fn particle_swarm_rejects_degenerate_bounds() {
        let _ = run_particle_swarm(ParticleSwarmParams {
            lower: Some(1.0),
            upper: Some(1.0),
            ..Default::default()
        });
    }

    #[test]
    #[should_panic]
    fn particle_swarm_rejects_invalid_dimension() {
        let _ = run_particle_swarm(ParticleSwarmParams {
            dimension: Some(0),
            ..Default::default()
        });
    }

    // -- ant-colony-tsp --
    #[test]
    fn ant_colony_tsp() {
        let r = run_ant_colony_tsp(AntColonyTSPParams {
            iterations: Some(80),
            seed: Some(5),
            ..Default::default()
        });
        assert!(r.best_tour.len() >= 3 && r.best_tour[0] == r.best_tour[r.best_tour.len() - 1]);
        assert!(r.best_length.is_finite() && r.best_length > 0.0);
        assert!(has(&r.topology.movables, "GraphWalkToken"));
        assert_eq!(r.topology.stations[0].as_str(), "ant-colony-tsp-source");
        assert!(has(&r.topology.stations, "ant-colony-tsp-result-sink"));
    }

    #[test]
    #[should_panic]
    fn ant_colony_rejects_duplicate_points() {
        let _ = run_ant_colony_tsp(AntColonyTSPParams {
            points: Some(vec![Point2 { x: 0.0, y: 0.0 }, Point2 { x: 0.0, y: 0.0 }]),
            ..Default::default()
        });
    }

    // -- map-coloring-csp --
    #[test]
    fn map_coloring_csp() {
        let r = run_map_coloring_csp(MapColoringCSPParams::default());
        assert!(r.satisfied, "{:?}", r.assignment);
        assert!(r.nodes_processed > 0);
        assert!(has(&r.topology.movables, "ConstraintAssignmentToken"));
        assert_eq!(r.topology.stations[0].as_str(), "map-coloring-csp-source");
        assert!(has(&r.topology.stations, "map-coloring-csp-result-sink"));
    }

    #[test]
    #[should_panic]
    fn map_coloring_rejects_unknown_variable() {
        let _ = run_map_coloring_csp(MapColoringCSPParams {
            variables: Some(vec!["A".to_string()]),
            colors: Some(vec!["red".to_string()]),
            edges: Some(vec![("A".to_string(), "B".to_string())]),
            ..Default::default()
        });
    }

    // -- max-sat-local-search --
    #[test]
    fn max_sat_local_search() {
        let r = run_max_sat_local_search(MaxSATParams::default());
        assert!(
            r.all_satisfied,
            "{}/{}",
            r.satisfied_clauses, r.total_clauses
        );
        assert!(r
            .topology
            .movables
            .iter()
            .any(|v| v.contains("OptimizationCandidateToken")));
        assert_eq!(
            r.topology.stations[0].as_str(),
            "max-sat-local-search-source"
        );
        assert!(has(
            &r.topology.stations,
            "max-sat-local-search-result-sink"
        ));
    }

    // -- sdp-maxcut-relaxation --
    #[test]
    fn sdp_maxcut_relaxation() {
        let r = run_sdp_max_cut_relaxation(SDPMaxCutParams::default());
        assert!(
            r.sdp_value + 1e-9 >= r.rounded_cut_value,
            "sdp={} cut={}",
            r.sdp_value,
            r.rounded_cut_value
        );
        for (i, row) in r.gram_matrix.iter().enumerate() {
            assert!((row[i] - 1.0).abs() <= 1e-9);
        }
        assert!(has(&r.topology.stations, "sdp-maxcut-relaxation-station"));
        assert_eq!(
            r.topology.stations[0].as_str(),
            "sdp-maxcut-relaxation-source"
        );
        assert!(has(
            &r.topology.stations,
            "sdp-maxcut-relaxation-result-sink"
        ));

        let fallback = run_sdp_max_cut_relaxation(SDPMaxCutParams {
            edges: Some(vec![]),
            ..Default::default()
        });
        assert!(fallback.rounded_cut_value > 0.0);
    }

    // -- pareto-portfolio --
    #[test]
    fn pareto_portfolio() {
        let r = run_pareto_portfolio(ParetoPortfolioParams::default());
        assert!(r.pareto_front.len() >= 2, "front={}", r.pareto_front.len());
        assert!(pareto_front_is_nondominated(&r.pareto_front));
        assert!(r.candidate_count >= 200, "candidates={}", r.candidate_count);
        assert_eq!(r.topology.stations[0].as_str(), "pareto-portfolio-source");
        assert!(has(&r.topology.stations, "pareto-portfolio-archive"));
    }

    #[test]
    fn pareto_archive_reuse_and_dedup() {
        let archive = Rc::new(RefCell::new(ParetoArchiveStation::<i32>::new(
            "pareto-reuse-test",
            vec![],
        )));
        run_iterative_des(
            vec![archive.clone() as StationRef],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(2),
                run_validators: false,
                ..Default::default()
            },
        );
        archive
            .borrow_mut()
            .enqueue(ParetoCandidateToken::new(1, vec![1.0, -1.0]));
        run_iterative_des(
            vec![archive.clone() as StationRef],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(2),
                run_validators: false,
                ..Default::default()
            },
        );
        assert_eq!(archive.borrow().get_processed_count(), 1);

        let dup = Rc::new(RefCell::new(ParetoArchiveStation::<i32>::new(
            "pareto-duplicate-test",
            vec![
                ParetoCandidateToken::new(1, vec![1.0, -1.0]),
                ParetoCandidateToken::new(2, vec![1.0, -1.0]),
            ],
        )));
        run_iterative_des(
            vec![dup.clone() as StationRef],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(4),
                run_validators: false,
                ..Default::default()
            },
        );
        assert_eq!(dup.borrow().get_archive().len(), 1);
    }

    // -- hinfinity-robust-control --
    #[test]
    fn hinfinity_robust_control() {
        let r = run_h_infinity_robust_control(HInfinityRobustControlParams::default()).unwrap();
        assert!(
            r.bounded_by_gamma,
            "gain={} gamma={}",
            r.l2_gain_estimate, r.gamma
        );
        assert!(r.final_state.abs() < 0.5, "final={}", r.final_state);
        assert_eq!(r.topology.stations.len(), 3);
    }

    // -- pursuit-evasion-game --
    #[test]
    fn pursuit_evasion_game() {
        let r = run_pursuit_evasion_game(PursuitEvasionGameParams::default()).unwrap();
        assert!(r.capture_tick.is_some(), "capture={:?}", r.capture_tick);
        assert!(
            r.final_distance < r.distance_history[0],
            "d0={} df={}",
            r.distance_history[0],
            r.final_distance
        );
        assert!(
            has(&r.topology.movables, "ControlMoveToken")
                && has(&r.topology.movables, "DisturbanceMoveToken")
        );
    }
}
