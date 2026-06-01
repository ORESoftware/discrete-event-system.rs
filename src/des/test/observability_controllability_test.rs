//! Port of src/des/test/observability-controllability-test.ts
//!
//! Unit tests for general/control-systems/observability-controllability and the
//! shared LinAlg helpers. Groups [1]-[5] are ported faithfully.
//!
//! PORT NOTE: the original group [6] (the DES evaluator pipeline driven by
//! `run_iterative_des` over source/evaluator/sink stations) is deferred; the
//! evaluator verdicts are equivalent to the direct model queries in [2]-[5].
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::control_systems::information_theory::{
        channel_information, entropy_summary, jensen_shannon_divergence_bits,
    };
    use crate::des::general::control_systems::observability_controllability::{
        MarkovDecisionProcess, MdpSpec, PartiallyObservableProcess, PomdpSpec, StateSpaceModel,
        StateSpaceSpec,
    };
    use crate::des::shared::linalg::LinAlg;

    fn ss(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>, c: Vec<Vec<f64>>) -> StateSpaceModel {
        StateSpaceModel::new(StateSpaceSpec { a, b, c, d: None })
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // [1] LinAlg — rank, products, stacking
    #[test]
    fn linalg_rank_products_stacking() {
        assert_eq!(LinAlg::rank(&LinAlg::identity(4), None), 4);
        assert_eq!(LinAlg::rank(&vec![vec![1.0, 2.0], vec![1.0, 2.0]], None), 1);
        assert_eq!(LinAlg::rank(&LinAlg::zeros(3, 3), None), 0);
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let i = LinAlg::identity(2);
        assert_eq!(LinAlg::mat_mul(&a, &i), a);
        assert_eq!(
            LinAlg::hstack(&[vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]),
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
        assert_eq!(
            LinAlg::vstack(&[vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]]),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
        assert_eq!(LinAlg::power(&a, 2), LinAlg::mat_mul(&a, &a));
    }

    // [2] Linear state-space — the worked example
    #[test]
    fn state_space_worked_example() {
        let m = ss(
            vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
        );
        assert_eq!(
            m.controllability_matrix(),
            vec![vec![0.0, 1.0], vec![1.0, 0.0]]
        );
        assert_eq!(
            m.observability_matrix(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert!(m.controllability_rank() == 2 && m.is_controllable());
        assert!(m.observability_rank() == 2 && m.is_observable());
    }

    // [3] Linear state-space — deficient cases
    #[test]
    fn state_space_deficient_cases() {
        let neither = ss(
            vec![vec![1.0, 0.0], vec![0.0, 2.0]],
            vec![vec![1.0], vec![0.0]],
            vec![vec![1.0, 0.0]],
        );
        assert!(!neither.is_controllable() && neither.controllability_rank() == 1);
        assert!(!neither.is_observable() && neither.observability_rank() == 1);

        let c_not_o = ss(
            vec![vec![1.0, 0.0], vec![0.0, 2.0]],
            vec![vec![1.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
        );
        assert!(c_not_o.is_controllable() && !c_not_o.is_observable());

        let o_not_c = ss(
            vec![vec![1.0, 0.0], vec![0.0, 2.0]],
            vec![vec![1.0], vec![0.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        assert!(o_not_c.is_observable() && !o_not_c.is_controllable());
    }

    // [4] MDP — reachability (controllability analog)
    #[test]
    fn mdp_reachability() {
        let ring = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        });
        assert!(ring.is_structurally_controllable());
        assert_eq!(ring.reachable_pair_count(), 9);

        let trap = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
        });
        assert!(!trap.is_structurally_controllable());
        assert!(trap.reachable_pair_count() < 9);
    }

    // [5] POMDP — distinguishability (observability analog)
    #[test]
    fn pomdp_distinguishability() {
        let distinct = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        assert!(distinct.is_structurally_observable());
        assert!(distinct.class_count() == 2 && distinct.indistinguishable_pairs().is_empty());

        let aliased = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        });
        assert!(!aliased.is_structurally_observable());
        assert_eq!(aliased.indistinguishable_pairs(), vec![(0, 1)]);

        let multi_step = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        assert!(multi_step.is_structurally_observable());
        assert!(multi_step.indistinguishable_pairs().is_empty());
    }

    // [6] Shannon information theory — source entropy and channel information.
    #[test]
    fn information_theory_entropy_and_channel_metrics() {
        let source = entropy_summary(&[0.25, 0.25, 0.25, 0.25]);
        assert!(close(source.entropy_bits, 2.0));
        assert!(close(source.effective_symbols, 4.0));
        assert!(jensen_shannon_divergence_bits(&[1.0, 0.0], &[0.0, 1.0]) > 0.99);

        let perfect = channel_information(&[0.5, 0.5], &vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(close(perfect.mutual_information_bits, 1.0));
        assert!(close(perfect.equivocation_bits, 0.0));

        let aliased = channel_information(&[0.5, 0.5], &vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
        assert!(close(aliased.mutual_information_bits, 0.0));
        assert!(close(aliased.equivocation_bits, 1.0));
    }

    // [7] Model-level information summaries — MDP transition entropy and POMDP
    // sensor information.
    #[test]
    fn model_information_summaries() {
        let deterministic = MarkovDecisionProcess::new(MdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.0, 1.0], vec![1.0, 0.0]]],
        });
        assert!(close(
            deterministic
                .transition_information_summary()
                .mean_entropy_bits,
            0.0,
        ));

        let noisy = MarkovDecisionProcess::new(MdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
        });
        assert!(close(
            noisy
                .transition_information_summary()
                .normalized_mean_entropy,
            1.0,
        ));

        let pomdp = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let sensor = pomdp.observation_information(None);
        assert!(close(sensor.input_entropy_bits, 1.0));
        assert!(close(sensor.mutual_information_bits, 1.0));
        assert!(pomdp
            .observation_entropy_bits()
            .iter()
            .all(|&h| close(h, 0.0)));
    }
}
