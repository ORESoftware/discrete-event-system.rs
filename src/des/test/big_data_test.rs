//! Tests for `des::general::big_data`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::big_data::{
        fractional_factorial_design, regression_metrics, run_big_data_pipeline,
        screen_pairwise_interactions, target_correlations, BigDataPipelineConfig, CorrelationKind,
        ElasticNetModel, ElasticNetParams, GradientBoostingParams, GradientBoostingRegressor,
        NeuralNetworkParams, NumericDataset, RandomForestParams, RandomForestRegressor,
    };
    use crate::des::general::neural_network::ActivationName;

    fn xor_interaction_dataset() -> NumericDataset {
        let mut records = Vec::new();
        let mut target = Vec::new();
        for rep in 0..24 {
            let nuisance = (rep % 3) as f64 - 1.0;
            for &(a, b) in &[(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                records.push(vec![a, b, nuisance]);
                target.push(a * b);
            }
        }
        NumericDataset::new(
            vec![
                "age_band".to_string(),
                "married".to_string(),
                "nuisance".to_string(),
            ],
            records,
            target,
        )
        .unwrap()
    }

    fn smooth_pipeline_dataset() -> NumericDataset {
        let levels = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut records = Vec::new();
        let mut target = Vec::new();
        for &x0 in &levels {
            for &x1 in &levels {
                for &x2 in &[-1.0, 1.0] {
                    records.push(vec![x0, x1, x2]);
                    target.push(1.0 + 2.0 * x0 - 1.5 * x1 + 3.0 * x0 * x1 + 0.2 * x2);
                }
            }
        }
        NumericDataset::new(
            vec!["age".into(), "married".into(), "segment".into()],
            records,
            target,
        )
        .unwrap()
    }

    #[test]
    fn interaction_screen_finds_signal_hidden_from_main_effects() {
        let ds = xor_interaction_dataset();
        let main = target_correlations(&ds, CorrelationKind::Pearson).unwrap();
        assert!(main[0].score.abs() < 0.05, "main effects={main:?}");

        let interactions = screen_pairwise_interactions(&ds, 3).unwrap();
        let best = &interactions[0];
        assert_eq!((best.left, best.right), (0, 1));
        assert!(best.score > 0.99, "best={best:?}");
        assert!(best.synergy > 0.95, "best={best:?}");

        let model = ElasticNetModel::fit_with_interactions(
            &ds,
            &ElasticNetParams {
                alpha: 0.001,
                l1_ratio: 1.0,
                max_iter: 500,
                tol: 1.0e-9,
            },
            &[(best.left, best.right)],
        )
        .unwrap();
        let selected = model.selected_features(0.05);
        assert!(selected.iter().any(|s| s.name == "age_band*married"));
        assert!(regression_metrics("elastic_net", &model, &ds).r2 > 0.98);
    }

    #[test]
    fn tree_ensembles_learn_non_linear_interactions() {
        let mut ds = xor_interaction_dataset();
        for i in 0..ds.target.len() {
            ds.target[i] += 0.2 * ds.records[i][0];
        }
        let boosting = GradientBoostingRegressor::fit(
            &ds,
            &GradientBoostingParams {
                n_estimators: 25,
                learning_rate: 0.3,
                max_depth: 2,
                min_samples_leaf: 1,
            },
        )
        .unwrap();
        let forest = RandomForestRegressor::fit(
            &ds,
            &RandomForestParams {
                n_trees: 24,
                max_depth: 3,
                min_samples_leaf: 1,
                max_features: Some(2),
                sample_rate: 1.0,
                seed: 11,
            },
        )
        .unwrap();

        assert!(regression_metrics("gb", &boosting, &ds).r2 > 0.95);
        assert!(regression_metrics("rf", &forest, &ds).r2 > 0.85);
        assert!(boosting.feature_importances[0] > 0.0);
        assert!(forest.feature_importances[1] > 0.0);
    }

    #[test]
    fn full_pipeline_and_fractional_design_are_available() {
        let ds = smooth_pipeline_dataset();
        let report = run_big_data_pipeline(
            &ds,
            &BigDataPipelineConfig {
                max_interactions: 5,
                elastic_net_interactions: 3,
                elastic_net: ElasticNetParams {
                    alpha: 0.001,
                    l1_ratio: 0.8,
                    max_iter: 800,
                    tol: 1.0e-9,
                },
                gradient_boosting: GradientBoostingParams {
                    n_estimators: 40,
                    learning_rate: 0.15,
                    max_depth: 2,
                    min_samples_leaf: 2,
                },
                random_forest: RandomForestParams {
                    n_trees: 20,
                    max_depth: 4,
                    min_samples_leaf: 1,
                    max_features: Some(3),
                    sample_rate: 1.0,
                    seed: 3,
                },
                neural_network: NeuralNetworkParams {
                    hidden_layers: vec![8],
                    hidden_activation: ActivationName::Tanh,
                    epochs: 300,
                    learning_rate: 0.03,
                    seed: 5,
                    shuffle_each_epoch: true,
                },
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(report.correlations.values.len(), 3);
        assert_eq!(report.metrics.len(), 4);
        assert!(
            report.interactions[0].score > 0.6,
            "{:?}",
            report.interactions
        );
        assert!(
            report.best_metric().unwrap().r2 > 0.9,
            "{:?}",
            report.metrics
        );
        assert!(
            report.neural_network.loss_history.last().unwrap()
                < report.neural_network.loss_history.first().unwrap()
        );

        let design = fractional_factorial_design(2, &[vec![0, 1]]).unwrap();
        assert_eq!(design.len(), 4);
        assert_eq!(design[0], vec![-1.0, -1.0, 1.0]);
        assert_eq!(design[3], vec![1.0, 1.0, 1.0]);
    }
}
