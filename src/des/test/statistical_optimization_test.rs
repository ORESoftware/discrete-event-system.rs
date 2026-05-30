//! Port of src/des/test/statistical-optimization-test.ts
//!
//! Tests distribution fitting, risk-aware scenario optimisation, SDDP capacity
//! expansion, and adaptive simulation optimisation. Groups [1]-[4], [6] and the
//! direct-call hardening checks in [7] are ported faithfully.
//!
//! PORT NOTE: group [5] and the `runFromSpec`/`getModel` cases in [7]
//! (7.2, 7.4–7.9) depend on `des-registry`, which is not yet ported; those are
//! deferred. Cases 7.11 (invalid risk kind) and 7.14 (public fitted sampler)
//! have no constructible Rust equivalent (exhaustive enum / private API) and
//! are likewise deferred.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::statistical_optimization::{
        run_adaptive_sim_opt, run_capacity_expansion_sddp, run_distribution_fit, run_risk_capacity,
        AdaptiveAlternative, AdaptiveSimOptParams, DemandRange, DemandSpec, DistributionFamily,
        DistributionFitParams, DistributionFitter, EmpiricalPoint, FitMethod, RiskCapacityParams,
        RiskConfig, RiskKind, SDDPParams,
    };
    use crate::des::shared::transform::Transform;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn risk(kind: RiskKind) -> RiskConfig {
        RiskConfig {
            kind,
            alpha: None,
            lambda: None,
            min_service_level: None,
            shortfall_limit: None,
            radius: None,
        }
    }

    fn ranges(rs: &[(f64, f64)]) -> Vec<DemandRange> {
        rs.iter().map(|&(low, high)| DemandRange { low, high }).collect()
    }

    // [1] Distribution fitting: MLE vs method of moments
    #[test]
    fn distribution_fitting() {
        let samples = vec![8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0];
        let normal_mle = DistributionFitter::new(DistributionFamily::Normal, FitMethod::Mle)
            .transform(samples.as_slice());
        let normal_mom = DistributionFitter::new(DistributionFamily::Normal, FitMethod::Moments)
            .transform(samples.as_slice());
        assert!(close(normal_mle.mean, 10.0, 1e-8));
        assert!(close(normal_mle.variance, 1.5, 1e-8));
        assert!(close(normal_mom.variance, 12.0 / 7.0, 1e-8));

        let fit = run_distribution_fit(DistributionFitParams {
            samples,
            families: Some(vec![
                DistributionFamily::Normal,
                DistributionFamily::Gamma,
                DistributionFamily::Empirical,
            ]),
            methods: Some(vec![FitMethod::Mle, FitMethod::Moments]),
        })
        .unwrap();
        assert!(fit.validation.iter().all(|c| c.passed));
        assert!(fit.fits[0].aic <= fit.fits[fit.fits.len() - 1].aic);
    }

    // [2] Risk-aware scenario optimisation
    #[test]
    fn risk_aware_scenario_optimisation() {
        let chance = run_risk_capacity(RiskCapacityParams {
            cost: vec![10.0, 12.0],
            price: vec![25.0, 28.0],
            demand: DemandSpec::Uniform(ranges(&[(50.0, 100.0), (40.0, 80.0)])),
            num_scenarios: 60,
            seed: 5,
            x_max: 120.0,
            step: 20.0,
            risk: RiskConfig {
                min_service_level: Some(0.8),
                shortfall_limit: Some(10.0),
                ..risk(RiskKind::Chance)
            },
        })
        .unwrap();
        assert!(chance.validation.iter().all(|c| c.passed));
        assert!(chance.best.service_level >= 0.8 - 1e-12);

        let dro = run_risk_capacity(RiskCapacityParams {
            cost: vec![10.0],
            price: vec![25.0],
            demand: DemandSpec::Uniform(ranges(&[(20.0, 80.0)])),
            num_scenarios: 50,
            seed: 9,
            x_max: 100.0,
            step: 10.0,
            risk: RiskConfig { radius: Some(1.0), ..risk(RiskKind::Dro) },
        })
        .unwrap();
        assert!(dro.best.robust_objective <= dro.best.mean_profit + 1e-9);

        let empirical = run_risk_capacity(RiskCapacityParams {
            cost: vec![10.0],
            price: vec![25.0],
            demand: DemandSpec::Empirical(vec![vec![
                EmpiricalPoint { value: 20.0, prob: 0.5 },
                EmpiricalPoint { value: 60.0, prob: 0.5 },
            ]]),
            num_scenarios: 30,
            seed: 4,
            x_max: 80.0,
            step: 20.0,
            risk: risk(RiskKind::Expectation),
        })
        .unwrap();
        assert!(empirical.validation.iter().all(|c| c.passed));
    }

    // [3] Multi-stage SDDP-style capacity expansion
    #[test]
    fn sddp_capacity_expansion() {
        let r = run_capacity_expansion_sddp(
            SDDPParams {
                horizon: 3,
                demand: ranges(&[(20.0, 50.0), (30.0, 70.0), (40.0, 90.0)]),
                price: vec![25.0, 24.0, 23.0],
                expansion_cost: vec![12.0, 10.0, 8.0],
                initial_capacity: 0.0,
                x_max: 100.0,
                step: 10.0,
                samples_per_stage: 30,
                seed: 7,
                max_iter: Some(25),
                tol: Some(0.01),
            },
            None,
        )
        .unwrap();
        assert!(r.validation.iter().all(|c| c.passed));
        assert!(r.final_upper_bound + 1e-6 >= r.exact_objective);
        assert!(r.final_lower_bound <= r.exact_objective + 1e-6);
        assert!(r.gap <= 1e-5, "gap={}", r.gap);
    }

    // [4] Adaptive simulation optimisation
    #[test]
    fn adaptive_sim_opt() {
        let r = run_adaptive_sim_opt(
            AdaptiveSimOptParams {
                cost: vec![10.0, 12.0],
                price: vec![25.0, 28.0],
                demand: DemandSpec::Uniform(ranges(&[(50.0, 100.0), (40.0, 80.0)])),
                alternatives: vec![
                    AdaptiveAlternative { name: "lean".to_string(), x: vec![60.0, 50.0] },
                    AdaptiveAlternative { name: "balanced".to_string(), x: vec![80.0, 65.0] },
                    AdaptiveAlternative { name: "buffered".to_string(), x: vec![100.0, 80.0] },
                ],
                seed: 11,
                initial_samples: 3,
                budget: 45,
                batch_size: 3,
                exploration: 1.5,
            },
            None,
        )
        .unwrap();
        let total: f64 = r.stats.iter().map(|s| s.n).sum();
        assert!(r.validation.iter().all(|c| c.passed));
        assert!((total - 45.0).abs() < 1e-9, "total={total}");
        assert!(r.stats.iter().all(|s| s.n >= 3.0));
        assert!(r.best.stderr.is_finite());
    }

    // [6] Fail-fast preconditions
    #[test]
    fn fail_fast_preconditions() {
        let sddp = run_capacity_expansion_sddp(
            SDDPParams {
                horizon: 1,
                demand: ranges(&[(10.0, 5.0)]),
                price: vec![20.0],
                expansion_cost: vec![10.0],
                initial_capacity: 0.0,
                x_max: 30.0,
                step: 10.0,
                samples_per_stage: 5,
                seed: 1,
                max_iter: None,
                tol: None,
            },
            None,
        );
        assert!(sddp.is_err());

        let adaptive = run_adaptive_sim_opt(
            AdaptiveSimOptParams {
                cost: vec![10.0],
                price: vec![20.0],
                demand: DemandSpec::Uniform(ranges(&[(0.0, 10.0)])),
                alternatives: vec![
                    AdaptiveAlternative { name: "a".to_string(), x: vec![5.0] },
                    AdaptiveAlternative { name: "b".to_string(), x: vec![6.0] },
                ],
                seed: 1,
                initial_samples: 5,
                budget: 2,
                batch_size: 1,
                exploration: 1.0,
            },
            None,
        );
        assert!(adaptive.is_err());
    }

    // [7] Hardening regressions (direct-call subset)
    #[test]
    fn hardening_regressions() {
        // 7.1 empirical probabilities must sum to one
        assert!(run_risk_capacity(RiskCapacityParams {
            cost: vec![10.0],
            price: vec![25.0],
            demand: DemandSpec::Empirical(vec![vec![
                EmpiricalPoint { value: 20.0, prob: 0.4 },
                EmpiricalPoint { value: 60.0, prob: 0.4 },
            ]]),
            num_scenarios: 5,
            seed: 4,
            x_max: 80.0,
            step: 20.0,
            risk: risk(RiskKind::Expectation),
        })
        .is_err());

        // 7.3 oversized risk grid is rejected before enumeration
        assert!(run_risk_capacity(RiskCapacityParams {
            cost: vec![1.0, 1.0, 1.0, 1.0],
            price: vec![2.0, 2.0, 2.0, 2.0],
            demand: DemandSpec::Uniform(ranges(&[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])),
            num_scenarios: 1,
            seed: 1,
            x_max: 100.0,
            step: 1.0,
            risk: risk(RiskKind::Expectation),
        })
        .is_err());

        // 7.10 distribution-fit requires at least two samples
        assert!(run_distribution_fit(DistributionFitParams {
            samples: vec![1.0],
            families: Some(vec![DistributionFamily::Normal]),
            methods: Some(vec![FitMethod::Mle]),
        })
        .is_err());

        // 7.12 SDDP rejects non-finite demand highs
        assert!(run_capacity_expansion_sddp(
            SDDPParams {
                horizon: 1,
                demand: ranges(&[(10.0, f64::INFINITY)]),
                price: vec![20.0],
                expansion_cost: vec![10.0],
                initial_capacity: 0.0,
                x_max: 30.0,
                step: 10.0,
                samples_per_stage: 5,
                seed: 1,
                max_iter: None,
                tol: None,
            },
            None,
        )
        .is_err());

        // 7.13 adaptive simopt rejects duplicate names
        assert!(run_adaptive_sim_opt(
            AdaptiveSimOptParams {
                cost: vec![10.0],
                price: vec![25.0],
                demand: DemandSpec::Uniform(ranges(&[(20.0, 50.0)])),
                alternatives: vec![
                    AdaptiveAlternative { name: "same".to_string(), x: vec![20.0] },
                    AdaptiveAlternative { name: "same".to_string(), x: vec![40.0] },
                ],
                seed: 1,
                initial_samples: 1,
                budget: 2,
                batch_size: 1,
                exploration: 1.0,
            },
            None,
        )
        .is_err());
    }
}
