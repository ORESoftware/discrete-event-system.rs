//! Port of src/des/test/genetic-tsp-test.ts
//!
//! Unit tests for the GA-TSP module (operators, exact solver, bounds, and the
//! end-to-end genetic algorithm). `mulberry32` maps onto `SeededRandom`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::des::general::genetic_tsp::{
        build_pentagon_tsp, build_random_tsp, check_precedence, held_karp_exact, inversion_mutate,
        is_permutation, one_tree_lower_bound, order_crossover, repair_precedence, run_genetic_tsp,
        swap_mutate, tour_length, tournament_select, two_opt_improve, Feasibility, GASolverOptions,
        LocalSearch, TSPInstance,
    };
    use crate::des::general::prng::mulberry32;
    use crate::des::shared::capabilities::RandomSource;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn dist_matrix(coords: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let n = coords.len();
        let mut d = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                d[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        d
    }

    // Group 1 — Tour length and feasibility.
    #[test]
    fn tour_length_and_feasibility() {
        let inst = build_pentagon_tsp(4, 50.0);
        let tour = [0, 1, 2, 3];
        let len = tour_length(&inst, &tour);
        assert!(close(len, 4.0 * 50.0 * 2.0_f64.sqrt(), 1e-6));
        assert!(is_permutation(&tour, 4));
        assert!(!is_permutation(&[0, 1, 1, 3], 4));
        assert!(!is_permutation(&[0, 1, 4, 3], 4));
    }

    // Group 2 — Order-Crossover preserves permutations.
    #[test]
    fn order_crossover_preserves_permutations() {
        let mut rng = mulberry32(11);
        for _ in 0..50 {
            let n = 8;
            let p1: Vec<usize> = (0..n).collect();
            let mut p2: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
                p2.swap(i, j);
            }
            let child = order_crossover(&p1, &p2, &mut rng);
            assert!(is_permutation(&child, n), "OX produced {:?}", child);
        }
    }

    // Group 3 — Mutations preserve permutations.
    #[test]
    fn mutations_preserve_permutations() {
        let mut rng = mulberry32(13);
        let tour: Vec<usize> = (0..8).collect();
        for trial in 0..30 {
            let m = if trial % 2 == 0 {
                inversion_mutate(&tour, &mut rng)
            } else {
                swap_mutate(&tour, &mut rng)
            };
            assert!(is_permutation(&m, 8));
        }
    }

    // Group 4 — Tournament selection picks lower-cost chromosome.
    #[test]
    fn tournament_selection_finds_global_best() {
        let lengths = [10.0, 5.0, 100.0, 200.0, 1.0, 50.0];
        let mut rng = mulberry32(1);
        let mut found_one = false;
        for _ in 0..50 {
            let idx = tournament_select(&lengths, 6, &mut rng);
            if idx == 4 {
                found_one = true;
            }
        }
        assert!(found_one, "size-N tournament eventually picks global best");
    }

    // Group 5 — Held–Karp on a 4-city square gives perimeter 40.
    #[test]
    fn held_karp_square() {
        let coords = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let s200 = 200.0_f64.sqrt();
        let inst = TSPInstance {
            n: 4,
            coordinates: coords,
            distance: vec![
                vec![0.0, 10.0, s200, 10.0],
                vec![10.0, 0.0, 10.0, s200],
                vec![s200, 10.0, 0.0, 10.0],
                vec![10.0, s200, 10.0, 0.0],
            ],
            precedence: None,
        };
        let r = held_karp_exact(&inst);
        assert!(close(r.length, 40.0, 1e-9));
        assert!(is_permutation(&r.tour, 4));
    }

    // Group 6 — 1-tree lower bound respects optimum.
    #[test]
    fn one_tree_lower_bound_below_optimum() {
        for seed in [1u32, 5, 11] {
            let inst = build_random_tsp(8, seed, None);
            let lb = one_tree_lower_bound(&inst);
            let exact = held_karp_exact(&inst);
            assert!(
                lb <= exact.length + 1e-9,
                "seed={seed}: lb={lb} opt={}",
                exact.length
            );
        }
    }

    // Group 7 — GA solves a tiny pentagon to optimum.
    #[test]
    fn ga_solves_pentagon() {
        let inst = build_pentagon_tsp(5, 40.0);
        let optimal = 5.0 * 2.0 * 40.0 * (PI / 5.0).sin();
        let r = run_genetic_tsp(
            inst,
            GASolverOptions {
                population_size: Some(50),
                num_generations: Some(60),
                seed: Some(42),
                ..Default::default()
            },
        );
        assert!(
            close(r.best_length, optimal, 1e-6),
            "best={} opt={}",
            r.best_length,
            optimal
        );
    }

    // Group 8 — Reproducibility.
    #[test]
    fn reproducible_for_same_seed() {
        let inst = build_random_tsp(10, 7, None);
        let r1 = run_genetic_tsp(
            inst.clone(),
            GASolverOptions {
                population_size: Some(40),
                num_generations: Some(30),
                seed: Some(99),
                ..Default::default()
            },
        );
        let r2 = run_genetic_tsp(
            inst,
            GASolverOptions {
                population_size: Some(40),
                num_generations: Some(30),
                seed: Some(99),
                ..Default::default()
            },
        );
        assert!(close(r1.best_length, r2.best_length, 1e-12));
        assert_eq!(r1.best_tour, r2.best_tour);
    }

    // Group 9 — Precedence: cut policy yields feasible tours.
    #[test]
    fn precedence_cut_policy_feasible() {
        let mut inst = build_random_tsp(12, 21, None);
        inst.precedence = Some(vec![(0, 11), (1, 10), (2, 9)]);
        let r = run_genetic_tsp(
            inst.clone(),
            GASolverOptions {
                population_size: Some(50),
                num_generations: Some(80),
                seed: Some(21),
                feasibility: Some(Feasibility::Cut),
                retry_limit: Some(12),
                ..Default::default()
            },
        );
        assert!(check_precedence(&inst, &r.best_tour).is_none());
        assert!(is_permutation(&r.best_tour, 12));
    }

    // Group 10 — repairPrecedence does what it says.
    #[test]
    fn repair_precedence_makes_feasible() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let inst = TSPInstance {
            n: 4,
            coordinates: coords,
            distance: vec![
                vec![0.0, 1.0, 2.0, 3.0],
                vec![1.0, 0.0, 1.0, 2.0],
                vec![2.0, 1.0, 0.0, 1.0],
                vec![3.0, 2.0, 1.0, 0.0],
            ],
            precedence: Some(vec![(0, 1), (2, 3)]),
        };
        // Tour [3, 2, 1, 0] violates both pairs.
        let r = repair_precedence(&inst, &[3, 2, 1, 0], 4);
        assert!(r.feasible, "final = {:?}", r.tour);
    }

    // Group 11 — 2-opt local search never worsens a tour; memetic GA.
    #[test]
    fn two_opt_and_memetic_ga() {
        let inst = build_random_tsp(14, 55, None);
        let bad_tour = [0, 5, 2, 9, 3, 10, 4, 11, 1, 12, 6, 13, 7, 8];
        let improved = two_opt_improve(&inst, &bad_tour, 12);
        assert!(is_permutation(&improved, inst.n));
        assert!(
            tour_length(&inst, &improved) <= tour_length(&inst, &bad_tour) + 1e-9,
            "{} -> {}",
            tour_length(&inst, &bad_tour),
            tour_length(&inst, &improved)
        );

        let plain = run_genetic_tsp(
            inst.clone(),
            GASolverOptions {
                population_size: Some(40),
                num_generations: Some(40),
                seed: Some(5),
                ..Default::default()
            },
        );
        let memetic = run_genetic_tsp(
            inst.clone(),
            GASolverOptions {
                population_size: Some(40),
                num_generations: Some(40),
                seed: Some(5),
                local_search: Some(LocalSearch::TwoOpt),
                local_search_passes: Some(2),
                ..Default::default()
            },
        );
        assert!(memetic.local_search_applications > 0);
        assert!(is_permutation(&memetic.best_tour, inst.n));
        assert!(
            memetic.best_length <= plain.best_length + 1e-9,
            "{} -> {}",
            plain.best_length,
            memetic.best_length
        );
        assert!(memetic.performance.elapsed_ms >= 0.0);
        assert!(memetic.performance.estimated_evaluations > 0);
        assert_eq!(memetic.performance.final_best, memetic.best_length);
    }
}
