//! Port of src/des/test/learning-optimization-test.ts
//!
//! Tests for the learning / optimization station-graph models: ordinary and
//! ridge least-squares regression, logistic-regression SGD, and the backprop
//! MLP classifier (`general/learning-optimization-models`), plus the tabular RL
//! learners — REINFORCE policy gradient and Expected SARSA
//! (`general/rl-learning-models`).
//!
//! PORT NOTE: the TS "registry smoke" section uses `general/des-registry`
//! (`getModel`, `runFromSpec`), which is not yet ported to Rust; it is deferred.
//! Stochastic learners are seeded so the asserted properties are reproducible.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::learning_optimization::Optimizer;
    use crate::des::general::learning_optimization_models::{
        run_backprop_mlp_classifier, run_linear_regression_ls, run_logistic_regression_sgd,
        run_ridge_regression_ls, BackpropMLPParams, LinearRegressionParams,
        LogisticRegressionSGDParams, RidgeRegressionParams,
    };
    use crate::des::general::rl_learning_models::{
        run_expected_sarsa_gridworld, run_policy_gradient_corridor, ExpectedSarsaGridParams,
        PolicyGradientCorridorParams,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * f64::max(1.0, f64::max(a.abs(), b.abs()))
    }

    fn has(v: &[String], s: &str) -> bool {
        v.iter().any(|x| x == s)
    }

    // -- linear-regression-ls --
    #[test]
    fn linear_regression_ls() {
        let r = run_linear_regression_ls(LinearRegressionParams::default());
        assert!(
            close(r.coefficients[0], 2.0, 1e-8),
            "slope={}",
            r.coefficients[0]
        );
        assert!(close(r.intercept, 1.0, 1e-8), "intercept={}", r.intercept);
        assert!(r.mse < 1e-20, "mse={}", r.mse);
        assert!(has(&r.topology.stations, "normal-equation-accumulator"));
        assert!(has(&r.topology.movables, "RegressionFitToken"));
    }

    // -- ridge-regression-ls --
    #[test]
    fn ridge_regression_ls() {
        let r = run_ridge_regression_ls(RidgeRegressionParams {
            ridge: Some(0.01),
            ..Default::default()
        });
        assert!(r.coefficients.iter().all(|c| c.is_finite()) && r.intercept.is_finite());
        assert!(
            close(r.coefficients[0], 2.0, 0.02),
            "slope={}",
            r.coefficients[0]
        );
        assert!(has(&r.topology.movables, "RegressionFitToken"));
    }

    // -- logistic-regression-sgd --
    #[test]
    fn logistic_regression_sgd() {
        let r = run_logistic_regression_sgd(LogisticRegressionSGDParams {
            epochs: Some(160),
            batch_size: Some(3),
            learning_rate: Some(0.2),
            ..Default::default()
        });
        assert!(!r.loss_history.is_empty());
        assert_eq!(r.accuracy, 1.0);
        assert!(
            r.final_loss.is_finite() && r.final_loss < 0.2,
            "loss={}",
            r.final_loss
        );
        assert!(has(&r.topology.movables, "VectorBatchToken"));
        assert!(has(&r.topology.movables, "GradientStepToken"));
    }

    // -- backprop-mlp-classifier --
    #[test]
    fn backprop_mlp_classifier() {
        let r = run_backprop_mlp_classifier(BackpropMLPParams {
            hidden_units: Some(4),
            epochs: Some(800),
            batch_size: Some(4),
            learning_rate: Some(0.08),
            optimizer: Some(Optimizer::Adam),
            seed: Some(7),
            ..Default::default()
        });
        assert!(
            r.loss_history.len() >= 800,
            "steps={}",
            r.loss_history.len()
        );
        assert_eq!(r.accuracy, 1.0);
        assert!(r.final_loss.is_finite());
        assert!(has(&r.topology.stations, "backprop-gradient-update"));
        for t in ["VectorSampleToken", "VectorBatchToken", "GradientStepToken"] {
            assert!(has(&r.topology.movables, t), "missing movable {t}");
        }
    }

    // -- policy-gradient-corridor --
    #[test]
    fn policy_gradient_corridor() {
        let r = run_policy_gradient_corridor(PolicyGradientCorridorParams {
            num_episodes: Some(300),
            rollout_len: Some(12),
            alpha: Some(0.04),
            gamma: Some(0.95),
            seed: Some(1),
            length: Some(7),
            ..Default::default()
        });
        assert_eq!(r.reward_history.len(), 300);
        assert_eq!(r.greedy_success_rate, 1.0);
        assert!(r.updates > 0);
        for t in ["TrainTriggerToken", "ResumeToken"] {
            assert!(has(&r.topology.movables, t), "missing movable {t}");
        }
    }

    // -- expected-sarsa-gridworld --
    #[test]
    fn expected_sarsa_gridworld() {
        let r = run_expected_sarsa_gridworld(ExpectedSarsaGridParams {
            num_episodes: Some(900),
            alpha: Some(0.2),
            gamma: Some(0.95),
            epsilon: Some(0.35),
            epsilon_decay: Some(0.995),
            seed: Some(1),
            ..Default::default()
        });
        assert_eq!(r.reward_history.len(), 900);
        assert!(r.greedy_reached, "len={}", r.greedy_len);
        for t in ["StateToken", "ActionToken", "TransitionToken"] {
            assert!(has(&r.topology.movables, t), "missing movable {t}");
        }
    }
}
