//! Port of src/des/test/simulated-annealing-test.ts
//!
//! Unit tests for general/simulated-annealing. The TS check()/tally harness
//! becomes `#[test]` functions; stochastic runs are seeded for reproducibility.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::genetic_tsp::{
        build_pentagon_tsp, build_random_tsp, is_permutation, tour_length, InitMode, Tour,
    };
    use crate::des::general::prng::mulberry32;
    use crate::des::general::simulated_annealing::{
        build_knapsack_sa_problem, build_tsp_sa_problem, run_simulated_annealing, temperature_at,
        CoolingSchedule, KnapsackInstance, SAMove, SAProblem, SASolverOptions, TSPSAProblemOptions,
    };
    use crate::des::shared::capabilities::RandomSource;
    use std::rc::Rc;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn solver_options(
        max_iterations: usize,
        cooling: CoolingSchedule,
        seed: Option<u32>,
    ) -> SASolverOptions {
        SASolverOptions {
            max_iterations,
            cooling,
            seed,
            stall_limit: None,
            verbose: None,
            record_trace: None,
            trace_stride: None,
        }
    }

    // [1] temperatureAt
    #[test]
    fn temperature_schedules() {
        assert_eq!(
            temperature_at(
                &CoolingSchedule::Geometric {
                    t0: 100.0,
                    alpha: 0.9,
                    t_min: None
                },
                0
            ),
            100.0
        );
        assert!(close(
            temperature_at(
                &CoolingSchedule::Geometric {
                    t0: 100.0,
                    alpha: 0.9,
                    t_min: None
                },
                1
            ),
            90.0,
            1e-6
        ));
        assert!(close(
            temperature_at(
                &CoolingSchedule::Geometric {
                    t0: 100.0,
                    alpha: 0.9,
                    t_min: None
                },
                10
            ),
            100.0 * 0.9_f64.powi(10),
            1e-6
        ));
        assert!(close(
            temperature_at(
                &CoolingSchedule::Logarithmic {
                    t0: 100.0,
                    t_min: None
                },
                0
            ),
            100.0 / 2.0_f64.ln(),
            1e-6
        ));
        assert!(close(
            temperature_at(
                &CoolingSchedule::Linear {
                    t0: 100.0,
                    rate: 1.0,
                    t_min: None
                },
                50
            ),
            50.0,
            1e-6
        ));
        assert_eq!(
            temperature_at(
                &CoolingSchedule::Geometric {
                    t0: 100.0,
                    alpha: 0.5,
                    t_min: Some(5.0)
                },
                1000
            ),
            5.0
        );
        assert!(close(
            temperature_at(
                &CoolingSchedule::ExpRestart {
                    t0: 100.0,
                    alpha: 0.9,
                    period: 10,
                    t_min: None
                },
                5
            ),
            temperature_at(
                &CoolingSchedule::ExpRestart {
                    t0: 100.0,
                    alpha: 0.9,
                    period: 10,
                    t_min: None
                },
                15
            ),
            1e-6
        ));
    }

    // [2] TSP adapter — initial state is a valid permutation
    #[test]
    fn tsp_initial_is_permutation() {
        let p = build_tsp_sa_problem(
            build_random_tsp(15, 42, None),
            TSPSAProblemOptions {
                init: Some(InitMode::Random),
                ..Default::default()
            },
        );
        let mut rng = mulberry32(1);
        let init = p.initial(&mut rng);
        assert!(is_permutation(&init, 15));

        let p2 = build_tsp_sa_problem(
            build_random_tsp(15, 42, None),
            TSPSAProblemOptions {
                init: Some(InitMode::NearestNeighbor),
                ..Default::default()
            },
        );
        let init2 = p2.initial(&mut rng);
        assert!(is_permutation(&init2, 15));

        let inst = build_random_tsp(15, 42, None);
        assert!(tour_length(&inst, &init2).is_finite());
    }

    // [3] TSP adapter — neighbour preserves permutation
    #[test]
    fn tsp_neighbour_preserves_permutation() {
        let p = build_tsp_sa_problem(
            build_random_tsp(15, 42, None),
            TSPSAProblemOptions::default(),
        );
        let mut rng = mulberry32(1);
        let init = p.initial(&mut rng);
        for _ in 0..200 {
            let nb = p.neighbour(&init, &mut rng);
            assert!(is_permutation(&nb, 15));
        }

        let p2 = build_tsp_sa_problem(
            build_random_tsp(15, 42, None),
            TSPSAProblemOptions {
                moves: Some(SAMove::TwoOpt),
                ..Default::default()
            },
        );
        for _ in 0..200 {
            let nb = p2.neighbour(&init, &mut rng);
            assert!(is_permutation(&nb, 15));
        }

        let p3 = build_tsp_sa_problem(
            build_random_tsp(15, 42, None),
            TSPSAProblemOptions {
                moves: Some(SAMove::OrOpt),
                ..Default::default()
            },
        );
        for _ in 0..200 {
            let nb = p3.neighbour(&init, &mut rng);
            assert!(is_permutation(&nb, 15));
        }
    }

    // [4] Pentagon TSP — SA finds exact optimum
    #[test]
    fn pentagon_sa_finds_optimum() {
        let inst = build_pentagon_tsp(5, 50.0);
        let opt = tour_length(&inst, &[0, 1, 2, 3, 4]);
        let problem: Rc<dyn SAProblem<Tour>> =
            Rc::new(build_tsp_sa_problem(inst, TSPSAProblemOptions::default()));
        let r = run_simulated_annealing(
            problem,
            solver_options(
                3000,
                CoolingSchedule::Geometric {
                    t0: 50.0,
                    alpha: 0.999,
                    t_min: None,
                },
                Some(1),
            ),
        );
        assert!(
            close(r.best_cost, opt, 1e-4),
            "SA={}, opt={}",
            r.best_cost,
            opt
        );
        assert!(is_permutation(&r.best_state, 5));
    }

    // [5] Knapsack SA
    #[test]
    fn knapsack_sa() {
        let inst = KnapsackInstance {
            values: vec![60.0, 100.0, 120.0],
            weights: vec![10.0, 20.0, 30.0],
            capacity: 50.0,
        };
        let problem: Rc<dyn SAProblem<Vec<f64>>> =
            Rc::new(build_knapsack_sa_problem(inst.clone(), 1e6));
        let r = run_simulated_annealing(
            problem,
            solver_options(
                3000,
                CoolingSchedule::Geometric {
                    t0: 30.0,
                    alpha: 0.999,
                    t_min: None,
                },
                Some(1),
            ),
        );
        let value = -r.best_cost;
        assert!(close(value, 220.0, 1e-3), "value={value}");
        assert!(r.best_state.iter().all(|&v| v == 0.0 || v == 1.0));
        let weight: f64 = inst
            .weights
            .iter()
            .zip(&r.best_state)
            .map(|(w, x)| w * x)
            .sum();
        assert!(weight <= inst.capacity);
    }

    // [6] Reproducibility
    #[test]
    fn reproducibility() {
        let make = || -> Rc<dyn SAProblem<Tour>> {
            Rc::new(build_tsp_sa_problem(
                build_random_tsp(10, 1, None),
                TSPSAProblemOptions::default(),
            ))
        };
        let r1 = run_simulated_annealing(
            make(),
            solver_options(
                1000,
                CoolingSchedule::Geometric {
                    t0: 50.0,
                    alpha: 0.99,
                    t_min: None,
                },
                Some(42),
            ),
        );
        let r2 = run_simulated_annealing(
            make(),
            solver_options(
                1000,
                CoolingSchedule::Geometric {
                    t0: 50.0,
                    alpha: 0.99,
                    t_min: None,
                },
                Some(42),
            ),
        );
        assert!(close(r1.best_cost, r2.best_cost, 1e-12));
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.accepted_count, r2.accepted_count);
        assert_eq!(r1.final_state, r2.final_state);
    }

    // [7] Best history is monotonic
    #[test]
    fn best_history_monotonic() {
        let problem: Rc<dyn SAProblem<Tour>> = Rc::new(build_tsp_sa_problem(
            build_random_tsp(15, 4, None),
            TSPSAProblemOptions::default(),
        ));
        let r = run_simulated_annealing(
            problem,
            solver_options(
                5000,
                CoolingSchedule::Geometric {
                    t0: 50.0,
                    alpha: 0.999,
                    t_min: None,
                },
                Some(1),
            ),
        );
        for i in 1..r.best_history.len() {
            assert!(r.best_history[i] <= r.best_history[i - 1] + 1e-12);
        }
        assert_eq!(r.best_history.len(), r.iterations);
    }

    // [8] Generic adapter — quadratic minimisation
    #[test]
    fn generic_quadratic_minimisation() {
        struct Quad;
        impl SAProblem<f64> for Quad {
            fn cost(&self, s: &f64) -> f64 {
                (s - 3.0).powi(2)
            }
            fn neighbour(&self, s: &f64, rng: &mut dyn RandomSource) -> f64 {
                let step = if rng.next_float() < 0.5 { -1.0 } else { 1.0 };
                (s + step).clamp(-100.0, 100.0)
            }
            fn initial(&self, _rng: &mut dyn RandomSource) -> f64 {
                50.0
            }
        }
        let problem: Rc<dyn SAProblem<f64>> = Rc::new(Quad);
        let r = run_simulated_annealing(
            problem,
            solver_options(
                5000,
                CoolingSchedule::Geometric {
                    t0: 100.0,
                    alpha: 0.99,
                    t_min: None,
                },
                Some(1),
            ),
        );
        assert!((r.best_state - 3.0).abs() <= 1.0, "x = {}", r.best_state);
        assert!(r.best_cost <= 1.0);
    }

    // [9] Trace recording
    #[test]
    fn trace_recording() {
        let problem: Rc<dyn SAProblem<Tour>> = Rc::new(build_tsp_sa_problem(
            build_random_tsp(8, 2, None),
            TSPSAProblemOptions::default(),
        ));
        let mut opts = solver_options(
            100,
            CoolingSchedule::Geometric {
                t0: 50.0,
                alpha: 0.99,
                t_min: None,
            },
            Some(1),
        );
        opts.record_trace = Some(true);
        let r = run_simulated_annealing(problem, opts);
        let trace = r.trace.as_ref().expect("trace recorded");
        assert_eq!(trace.len(), r.iterations);
        assert_eq!(trace[0].k, 0);
        assert_eq!(trace.iter().filter(|e| e.accept).count(), r.accepted_count);
    }

    // [10] Stall-limit terminates early
    #[test]
    fn stall_limit_early_exit() {
        let problem: Rc<dyn SAProblem<Tour>> = Rc::new(build_tsp_sa_problem(
            build_random_tsp(8, 1, None),
            TSPSAProblemOptions::default(),
        ));
        let mut opts = solver_options(
            100_000,
            CoolingSchedule::Geometric {
                t0: 1e-12,
                alpha: 1.0,
                t_min: None,
            },
            Some(1),
        );
        opts.stall_limit = Some(30);
        let r = run_simulated_annealing(problem, opts);
        assert!(r.iterations < 100_000, "iters={}", r.iterations);
    }
}
