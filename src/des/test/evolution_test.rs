//! Tests for GA/GP curve fitting and bio-design modules.

#[cfg(test)]
mod evolution_tests {
    use crate::des::general::evolution::curve_fitting::ParametricCurveProblem;
    use crate::des::general::evolution::genetic_programming::tree_size;
    use crate::des::general::evolution::{
        hp_energy, run_curve_fit_ga, run_curve_fit_gp, run_hp_protein_ga, run_piecewise_ga,
        synthetic_noisy_sine, CurveConstraints, FitnessEvaluator, GaFlavor, GaOptions, GpFlavor,
        GpOptions, GpTreeConfig, ParametricFamily,
    };
    #[test]
    fn parametric_ga_beats_naive_line() {
        let data = synthetic_noisy_sine(50, 0.05, 99);
        let r = run_curve_fit_ga(
            data,
            ParametricFamily::Polynomial { degree: 5 },
            CurveConstraints::default(),
            GaOptions::with_defaults(30, 30),
        );
        assert!(r.train_mse < 0.02, "mse={}", r.train_mse);
    }

    #[test]
    fn parametric_problem_batches_population_scores() {
        let data = synthetic_noisy_sine(36, 0.0, 12);
        let problem = ParametricCurveProblem {
            data,
            family: ParametricFamily::Polynomial { degree: 3 },
            constraints: CurveConstraints::default(),
            metric: crate::des::general::evolution::FitMetric::Mse,
            use_hybrid: true,
        };
        let mut rng = crate::des::general::prng::mulberry32(4);
        let pop = crate::des::general::evolution::PopulationInitializer::initial_population(
            &problem, 6, &mut rng,
        );
        let batch = problem.evaluate_population(&pop);
        let scalar: Vec<f64> = pop.iter().map(|c| problem.evaluate(c)).collect();
        assert_eq!(batch.len(), scalar.len());
        for (b, s) in batch.iter().zip(scalar) {
            assert!((b - s).abs() < 1e-12);
        }
    }

    #[test]
    fn gp_finds_sin_like_structure() {
        let data = synthetic_noisy_sine(40, 0.02, 3);
        let gp = run_curve_fit_gp(
            data.clone(),
            CurveConstraints {
                max_terms: Some(20),
                ..Default::default()
            },
            GpOptions {
                ga: GaOptions::with_defaults(40, 35),
                tree: GpTreeConfig::default(),
                flavor: Some(GpFlavor::ParsimonyPressure),
                parsimony_coef: Some(0.005),
            },
        );
        assert!(gp.fitness < 0.05, "fitness={}", gp.fitness);
        assert!(tree_size(&gp.expression) <= 25);
    }

    #[test]
    fn piecewise_segments_reduce_step_error() {
        let data = crate::des::general::evolution::synthetic_piecewise_step(50, 1);
        let r = run_piecewise_ga(data, 3, 2, GaOptions::with_defaults(35, 25));
        assert!(r.train_mse < 0.15, "mse={}", r.train_mse);
    }

    #[test]
    fn hp_protein_self_avoiding_low_energy() {
        let r = run_hp_protein_ga(
            10,
            GaOptions {
                flavor: Some(GaFlavor::Generational),
                ..GaOptions::with_defaults(25, 40)
            },
        );
        assert!(hp_energy(&r.genome) < 0.0);
        assert!(r.energy < 0.0);
    }

    #[test]
    fn island_ga_runs_without_panic() {
        let _ = run_hp_protein_ga(
            8,
            GaOptions {
                flavor: Some(GaFlavor::Island),
                num_islands: Some(2),
                ..GaOptions::with_defaults(20, 15)
            },
        );
    }
}
