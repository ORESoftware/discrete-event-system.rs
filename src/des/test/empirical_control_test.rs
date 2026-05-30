//! Port of src/des/test/empirical-control-test.ts
//!
//! Unit tests for `general/control-systems/empirical-control`: the dense
//! linear-algebra helpers (inverse, symmetric eigen), the deterministic RNG,
//! controllability / observability Gramians, the minimum-energy controller, the
//! Monte-Carlo controllability / observability / distinguishability estimators,
//! the MDP controllability degree, and the DES degree-report pipeline.
//!
//! Monte-Carlo estimators are seeded so the sampled degrees are reproducible;
//! the assertions check robust properties rather than exact sample values.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::general::control_systems::empirical_control::{
        ControllabilityGramian, DegreeKind, DegreeReportSinkStation, DiscreteLinearSystem,
        DiscreteSystemSourceStation, DiscreteSystemToken, EmpiricalChannels,
        LtiDegreeEvaluatorStation, MdpControllabilityDegree, MinEnergyController,
        MonteCarloControllability, MonteCarloControllabilityOpts, MonteCarloDistinguishability,
        MonteCarloObservability, MonteCarloObservabilityOpts, Mulberry32, ObservabilityGramian,
        RandomPolicyOpts,
    };
    use crate::des::general::control_systems::linear_algebra::{
        LinAlg, MatrixInverse, SymmetricEigen,
    };
    use crate::des::general::control_systems::observability_controllability::{
        MarkovDecisionProcess, MdpSpec, PartiallyObservableProcess, PomdpSpec, StateSpaceModel,
        StateSpaceSpec,
    };
    use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
    use crate::des::general::des_base::station::{DESStation, StationRef};

    type Mat = Vec<Vec<f64>>;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn mat_close(a: &Mat, b: &Mat, tol: f64) -> bool {
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                if (a[i][j] - b[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn double_integrator() -> DiscreteLinearSystem {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        });
        DiscreteLinearSystem::from_continuous(&model, 0.05)
    }

    fn decoupled() -> DiscreteLinearSystem {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            b: vec![vec![1.0], vec![0.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        });
        DiscreteLinearSystem::from_continuous(&model, 0.05)
    }

    // [1] LinAlg — inverse & symmetric eigen.
    #[test]
    fn linalg_inverse_and_eigen() {
        let a: Mat = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let ai = MatrixInverse::new(&a, None).inverse();
        let prod = LinAlg::mat_mul(&a, &ai);
        assert!(
            close(prod[0][0], 1.0, 1e-9)
                && close(prod[1][1], 1.0, 1e-9)
                && close(prod[0][1], 0.0, 1e-9)
        );

        let mut eig = SymmetricEigen::new(&vec![vec![2.0, 0.0], vec![0.0, 5.0]], 100);
        assert!(close(eig.min_eigenvalue(), 2.0, 1e-9) && close(eig.max_eigenvalue(), 5.0, 1e-9));

        let mut eig2 = SymmetricEigen::new(&vec![vec![2.0, 1.0], vec![1.0, 2.0]], 100);
        assert!(close(eig2.min_eigenvalue(), 1.0, 1e-8) && close(eig2.max_eigenvalue(), 3.0, 1e-8));
        let v = eig2.vectors();
        let dot = v[0][0] * v[0][1] + v[1][0] * v[1][1];
        assert!(close(dot, 0.0, 1e-8));
    }

    #[test]
    #[should_panic]
    fn singular_matrix_inverse_panics() {
        let _ = MatrixInverse::new(&vec![vec![1.0, 2.0], vec![2.0, 4.0]], None).inverse();
    }

    // [2] RNG determinism.
    #[test]
    fn rng_determinism() {
        let mut a = Mulberry32::new(42);
        let mut b = Mulberry32::new(42);
        assert!(a.next() == b.next() && a.next() == b.next());

        let mut c = Mulberry32::new(7);
        for _ in 0..1000 {
            let u = c.uniform(2.0);
            assert!((-2.0..=2.0).contains(&u));
        }

        let mut r = Mulberry32::new(1);
        let mut ones = 0;
        for _ in 0..5000 {
            if r.categorical(&[0.2, 0.8]) == 1 {
                ones += 1;
            }
        }
        assert!(
            close(ones as f64 / 5000.0, 0.8, 0.03),
            "got {}",
            ones as f64 / 5000.0
        );
    }

    // [3] Gramians — controllable vs deficient.
    #[test]
    fn gramians_controllable_vs_deficient() {
        let di = double_integrator();
        let wc = ControllabilityGramian::new(&di, 30);
        let wo = ObservabilityGramian::new(&di, 30);
        assert!(wc.min() > 1e-9);
        assert!(wo.min() > 1e-9);
        assert!(wc.condition_number().is_finite() && wo.condition_number().is_finite());

        let dec = decoupled();
        let wc_d = ControllabilityGramian::new(&dec, 30);
        let wo_d = ObservabilityGramian::new(&dec, 30);
        assert!(wc_d.min() < 1e-9);
        assert!(wo_d.min() < 1e-9);
        assert!(!wc_d.condition_number().is_finite() && !wo_d.condition_number().is_finite());
    }

    // [4] MinEnergyController — reach error detects uncontrollable subspace.
    #[test]
    fn min_energy_controller() {
        let di = double_integrator();
        let ctl = MinEnergyController::new(&di, 30, 1e-9);
        assert!(ctl.reach_error(&[0.5, -0.3]) < 1e-6);

        let dec = decoupled();
        let ctl_d = MinEnergyController::new(&dec, 30, 1e-9);
        assert!(ctl_d.reach_error(&[1.0, 0.0]) < 1e-6);
        assert!(ctl_d.reach_error(&[0.0, 1.0]) > 0.9);
    }

    // [5] Monte-Carlo controllability — recovers Gramian directions.
    #[test]
    fn monte_carlo_controllability() {
        let di = double_integrator();
        let wc = ControllabilityGramian::new(&di, 30);
        let mc = MonteCarloControllability::new(
            &di,
            30,
            MonteCarloControllabilityOpts {
                trials: Some(4000),
                input_bound: Some(1.0),
                seed: Some(1),
                ..Default::default()
            },
        )
        .run();
        let wc_eig = wc.eigenvalues();
        let wc_ratio = wc_eig[1] / wc_eig[0];
        let mc_ratio = mc.spread_eigenvalues[1] / mc.spread_eigenvalues[0];
        assert!(
            close(wc_ratio, mc_ratio, wc_ratio * 0.25),
            "gram={wc_ratio} mc={mc_ratio}"
        );
        assert!(
            mc.target_success_rate > 0.95,
            "rate={}",
            mc.target_success_rate
        );

        let dec = decoupled();
        let mc_d = MonteCarloControllability::new(
            &dec,
            30,
            MonteCarloControllabilityOpts {
                trials: Some(2000),
                seed: Some(3),
                ..Default::default()
            },
        )
        .run();
        assert!(
            mc_d.target_success_rate < 0.3,
            "rate={}",
            mc_d.target_success_rate
        );
    }

    // [6] Monte-Carlo observability — reconstruction error tracks W_o.
    #[test]
    fn monte_carlo_observability() {
        let di = double_integrator();
        let obs = MonteCarloObservability::new(
            &di,
            30,
            MonteCarloObservabilityOpts {
                trials: Some(1500),
                noise_std: Some(0.01),
                seed: Some(2),
                ..Default::default()
            },
        )
        .run();
        assert!(
            obs.mean_reconstruction_error < 0.1,
            "err={}",
            obs.mean_reconstruction_error
        );

        let dec = decoupled();
        let obs_d = MonteCarloObservability::new(
            &dec,
            30,
            MonteCarloObservabilityOpts {
                trials: Some(1500),
                noise_std: Some(0.01),
                seed: Some(2),
                ..Default::default()
            },
        )
        .run();
        assert!(
            obs_d.mean_reconstruction_error > 0.5,
            "err={}",
            obs_d.mean_reconstruction_error
        );
    }

    // [7] MDP controllability degree — value iteration + rollouts.
    #[test]
    fn mdp_controllability_degree() {
        let ring = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        });
        let trap = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
        });
        let ring_p = MdpControllabilityDegree::new(&ring);
        let trap_p = MdpControllabilityDegree::new(&trap);

        let hit_ring = ring_p.expected_hitting_times(0, 1000, 1e-9);
        assert!(close(hit_ring[2], 1.0, 1e-6), "={}", hit_ring[2]);
        assert!(close(hit_ring[1], 2.0, 1e-6), "={}", hit_ring[1]);

        let hit_trap = trap_p.expected_hitting_times(0, 1000, 1e-9);
        assert!(!hit_trap[1].is_finite() && !hit_trap[2].is_finite());

        let deg_ring = ring_p.per_target_degree(&RandomPolicyOpts {
            episodes: Some(400),
            seed: Some(1),
            ..Default::default()
        });
        assert!(deg_ring.iter().all(|&d| d > 0.99));
        let deg_trap = trap_p.per_target_degree(&RandomPolicyOpts {
            episodes: Some(400),
            seed: Some(1),
            ..Default::default()
        });
        assert!(deg_trap.iter().any(|&d| d < 0.99));
    }

    // [8] POMDP observability degree — belief tracking.
    #[test]
    fn pomdp_observability_degree() {
        let distinct = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let aliased = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        });

        let opts = RandomPolicyOpts {
            episodes: Some(600),
            seed: Some(5),
            ..Default::default()
        };
        let r_d = MonteCarloDistinguishability::new(&distinct).run(&opts);
        assert!(r_d.min_degree > 0.95, "min={}", r_d.min_degree);
        assert!(r_d.residual_entropy.iter().all(|&e| e < 0.05));

        let r_a = MonteCarloDistinguishability::new(&aliased).run(&opts);
        assert!(close(r_a.min_degree, 0.5, 0.1), "min={}", r_a.min_degree);
        assert!(r_a.residual_entropy.iter().all(|&e| close(e, 1.0, 0.1)));
    }

    // [9] DES pipeline — Gramian degree reports flow to sink.
    #[test]
    fn des_pipeline_degree_reports() {
        let di = double_integrator();
        let src = Rc::new(RefCell::new(DiscreteSystemSourceStation::new(
            "src",
            vec![DiscreteSystemToken::new("di".to_string(), di, 30)],
        )));
        let evalr = Rc::new(RefCell::new(LtiDegreeEvaluatorStation::new("eval")));
        let sink = Rc::new(RefCell::new(DegreeReportSinkStation::new("sink")));

        src.borrow_mut().core_mut().pipe(
            evalr.clone() as StationRef,
            EmpiricalChannels::SYSTEM,
            EmpiricalChannels::SYSTEM,
        );
        evalr.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            EmpiricalChannels::REPORT,
            EmpiricalChannels::REPORT,
        );
        run_iterative_des(
            vec![
                src.clone() as StationRef,
                evalr.clone() as StationRef,
                sink.clone() as StationRef,
            ],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(10),
                ..Default::default()
            },
        );

        let sink_ref = sink.borrow();
        assert_eq!(sink_ref.reports.len(), 1);
        let r = &sink_ref.reports[0];
        assert!(
            r.kind == DegreeKind::LtiDegree
                && r.min_controllability > 0.0
                && r.min_observability > 0.0
        );
    }

    // [10] Cross-validation invariants — Gramians, eigen, rollouts.
    #[test]
    fn cross_validation_invariants() {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0, 1.0], vec![0.0, -0.5]],
            b: vec![vec![0.0], vec![1.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        });
        let sys = DiscreteLinearSystem::from_continuous(&model, 0.1);
        let h = 12;

        let r = sys.reachability_map(h);
        let rrt = LinAlg::mat_mul(&r, &LinAlg::transpose(&r));
        let cg = ControllabilityGramian::new(&sys, h);
        let wc = cg.matrix();
        let o = sys.observability_map(h);
        let oto = LinAlg::mat_mul(&LinAlg::transpose(&o), &o);
        let og = ObservabilityGramian::new(&sys, h);
        let wo = og.matrix();
        assert!(mat_close(wc, &rrt, 1e-9));
        assert!(mat_close(wo, &oto, 1e-9));
        assert!(close(wc[0][1], wc[1][0], 1e-12) && close(wo[0][1], wo[1][0], 1e-12));

        let x = [0.3, -0.7];
        let winv = MatrixInverse::new(cg.matrix(), None).inverse();
        let w_inv_x = LinAlg::mat_vec(&winv, &x);
        let quad = x[0] * w_inv_x[0] + x[1] * w_inv_x[1];
        assert!(close(
            cg.min_energy_to_reach(&x),
            quad,
            1e-6 * quad.abs() + 1e-9
        ));

        let m: Mat = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 4.0],
        ];
        let mut eig = SymmetricEigen::new(&m, 100);
        let vals = eig.values();
        let vecs = eig.vectors();
        let mut lam = LinAlg::zeros(3, 3);
        for i in 0..3 {
            lam[i][i] = vals[i];
        }
        let recon = LinAlg::mat_mul(&LinAlg::mat_mul(&vecs, &lam), &LinAlg::transpose(&vecs));
        assert!(mat_close(&recon, &m, 1e-7));
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);

        let x0 = [1.0, 0.0];
        let u: Vec<Vec<f64>> = vec![vec![0.5], vec![-0.2], vec![0.1]];
        let mut manual = x0.to_vec();
        for uk in &u {
            manual = sys.step(&manual, uk);
        }
        let rolled = sys.rollout(&x0, &u);
        assert!(close(rolled[0], manual[0], 1e-12) && close(rolled[1], manual[1], 1e-12));
        assert_eq!(sys.outputs(&x0, 7, None).len(), 7);
    }

    // [11] MDP/POMDP degree — extra structural cases.
    #[test]
    fn extra_structural_cases() {
        let full = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]],
        });
        let deg = MdpControllabilityDegree::new(&full).per_target_degree(&RandomPolicyOpts {
            episodes: Some(400),
            seed: Some(2),
            ..Default::default()
        });
        assert!(deg.iter().all(|&d| d > 0.99), "{deg:?}");

        let bidir = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 2,
            transition: vec![
                vec![
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                    vec![0.0, 0.0, 1.0],
                ],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                ],
            ],
        });
        let planner = MdpControllabilityDegree::new(&bidir);
        let hit = planner.expected_hitting_times(0, 1000, 1e-9);
        assert!(hit.iter().all(|h| h.is_finite()));

        let multi_step = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let r = MonteCarloDistinguishability::new(&multi_step).run(&RandomPolicyOpts {
            episodes: Some(600),
            seed: Some(3),
            ..Default::default()
        });
        assert!(r.hit_probability[2] > 0.9, "p(s2)={}", r.hit_probability[2]);
    }
}
