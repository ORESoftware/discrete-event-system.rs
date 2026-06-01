//! Big-data and AI/ML data-pipeline helpers.
//!
//! This module is intentionally dependency-free and deterministic. It gives the
//! crate a compact analytics surface for the common workflow behind large data
//! pipelines:
//!
//! 1. validate/standardize numeric feature tables,
//! 2. compute pairwise and target correlations,
//! 3. screen pairwise interactions such as `age * marital_status`,
//! 4. train sparse linear and nonlinear predictive models.
//!
//! The algorithms here are practical in-process implementations for simulation
//! studies, demos, and regression tests. For petabyte-scale production jobs the
//! same concepts would usually be executed by Spark/Flink/Arrow-backed systems,
//! with this module acting as the modeling contract.

#![allow(dead_code)]

use std::cmp::Ordering;

use crate::des::general::des_base::neural_network::{NeuralNetworkLike, TrainableNeuralNetwork};
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

/// Dense numeric matrix: `rows[sample][feature]`.
pub type Matrix = Vec<Vec<f64>>;

const EPS: f64 = 1.0e-12;

// =============================================================================
// Dataset + sparse ingress
// =============================================================================

/// A numeric supervised-learning dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericDataset {
    pub feature_names: Vec<String>,
    pub records: Matrix,
    pub target: Vec<f64>,
}

impl NumericDataset {
    pub fn new(
        feature_names: Vec<String>,
        records: Matrix,
        target: Vec<f64>,
    ) -> Result<Self, String> {
        let ds = NumericDataset {
            feature_names,
            records,
            target,
        };
        ds.validate()?;
        Ok(ds)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.records.is_empty() {
            return Err("dataset must contain at least one row".to_string());
        }
        if self.target.len() != self.records.len() {
            return Err(format!(
                "target length {} does not match row count {}",
                self.target.len(),
                self.records.len()
            ));
        }
        let p = self.feature_names.len();
        if p == 0 {
            return Err("dataset must contain at least one feature".to_string());
        }
        for (i, row) in self.records.iter().enumerate() {
            if row.len() != p {
                return Err(format!(
                    "row {i} has {} columns but feature_names has {p}",
                    row.len()
                ));
            }
            for (j, &x) in row.iter().enumerate() {
                if !x.is_finite() {
                    return Err(format!("non-finite feature at row {i}, column {j}: {x}"));
                }
            }
        }
        for (i, &y) in self.target.iter().enumerate() {
            if !y.is_finite() {
                return Err(format!("non-finite target at row {i}: {y}"));
            }
        }
        Ok(())
    }

    pub fn n_rows(&self) -> usize {
        self.records.len()
    }

    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    pub fn column(&self, feature: usize) -> Vec<f64> {
        self.records.iter().map(|row| row[feature]).collect()
    }
}

/// One sparse row, useful for high-dimensional event/indicator data.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseSample {
    pub entries: Vec<(usize, f64)>,
    pub target: f64,
}

/// Sparse supervised-learning dataset. Duplicate entries in a row are summed
/// when converted to dense form.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseDataset {
    pub feature_names: Vec<String>,
    pub samples: Vec<SparseSample>,
}

impl SparseDataset {
    pub fn to_dense(&self) -> Result<NumericDataset, String> {
        if self.samples.is_empty() {
            return Err("sparse dataset must contain at least one sample".to_string());
        }
        let p = self.feature_names.len();
        if p == 0 {
            return Err("sparse dataset must contain at least one feature".to_string());
        }
        let mut records = Vec::with_capacity(self.samples.len());
        let mut target = Vec::with_capacity(self.samples.len());
        for (row_idx, sample) in self.samples.iter().enumerate() {
            let mut row = vec![0.0; p];
            for &(j, value) in &sample.entries {
                if j >= p {
                    return Err(format!(
                        "sparse row {row_idx} references feature {j} >= {p}"
                    ));
                }
                if !value.is_finite() {
                    return Err(format!("non-finite sparse value at row {row_idx}: {value}"));
                }
                row[j] += value;
            }
            if !sample.target.is_finite() {
                return Err(format!(
                    "non-finite sparse target at row {row_idx}: {}",
                    sample.target
                ));
            }
            records.push(row);
            target.push(sample.target);
        }
        NumericDataset::new(self.feature_names.clone(), records, target)
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

fn stddev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

fn standardize_vector(xs: &[f64]) -> (Vec<f64>, f64, f64) {
    let m = mean(xs);
    let s = stddev(xs);
    let scale = if s <= EPS { 1.0 } else { s };
    (xs.iter().map(|x| (x - m) / scale).collect(), m, scale)
}

fn columns(records: &[Vec<f64>]) -> Matrix {
    if records.is_empty() {
        return Vec::new();
    }
    let n = records.len();
    let p = records[0].len();
    let mut cols = vec![vec![0.0; n]; p];
    for (i, row) in records.iter().enumerate() {
        for j in 0..p {
            cols[j][i] = row[j];
        }
    }
    cols
}

fn squared_error(y: &[f64], pred: &[f64]) -> f64 {
    y.iter()
        .zip(pred)
        .map(|(a, b)| {
            let e = a - b;
            e * e
        })
        .sum()
}

// =============================================================================
// Correlations and interaction screening
// =============================================================================

/// Correlation statistic to use for pairwise scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrelationKind {
    Pearson,
    Spearman,
}

/// A square feature-feature correlation matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationMatrix {
    pub feature_names: Vec<String>,
    pub kind: CorrelationKind,
    pub values: Matrix,
}

/// Correlation or model-derived score for a single feature.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureScore {
    pub index: usize,
    pub name: String,
    pub score: f64,
}

/// Score for a pairwise interaction. `score` is the absolute target correlation
/// of the standardized product feature. `synergy` discounts the stronger main
/// effect so pure interactions float to the top.
#[derive(Clone, Debug, PartialEq)]
pub struct InteractionScore {
    pub left: usize,
    pub right: usize,
    pub left_name: String,
    pub right_name: String,
    pub score: f64,
    pub signed_score: f64,
    pub main_effect_max: f64,
    pub synergy: f64,
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let mx = mean(x);
    let my = mean(y);
    let mut num = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    if sx <= EPS || sy <= EPS {
        0.0
    } else {
        (num / (sx.sqrt() * sy.sqrt())).clamp(-1.0, 1.0)
    }
}

pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    pearson_correlation(&rank_average_ties(x), &rank_average_ties(y))
}

fn correlation(kind: CorrelationKind, x: &[f64], y: &[f64]) -> f64 {
    match kind {
        CorrelationKind::Pearson => pearson_correlation(x, y),
        CorrelationKind::Spearman => spearman_correlation(x, y),
    }
}

fn rank_average_ties(xs: &[f64]) -> Vec<f64> {
    let mut order: Vec<(usize, f64)> = xs.iter().copied().enumerate().collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; xs.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && (order[j].1 - order[i].1).abs() <= EPS {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[order[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

pub fn correlation_matrix(
    dataset: &NumericDataset,
    kind: CorrelationKind,
) -> Result<CorrelationMatrix, String> {
    dataset.validate()?;
    let p = dataset.n_features();
    let cols = columns(&dataset.records);
    let mut values = vec![vec![0.0; p]; p];
    for i in 0..p {
        values[i][i] = 1.0;
        for j in i + 1..p {
            let c = correlation(kind, &cols[i], &cols[j]);
            values[i][j] = c;
            values[j][i] = c;
        }
    }
    Ok(CorrelationMatrix {
        feature_names: dataset.feature_names.clone(),
        kind,
        values,
    })
}

pub fn target_correlations(
    dataset: &NumericDataset,
    kind: CorrelationKind,
) -> Result<Vec<FeatureScore>, String> {
    dataset.validate()?;
    let cols = columns(&dataset.records);
    let mut scores: Vec<FeatureScore> = cols
        .iter()
        .enumerate()
        .map(|(j, col)| FeatureScore {
            index: j,
            name: dataset.feature_names[j].clone(),
            score: correlation(kind, col, &dataset.target),
        })
        .collect();
    sort_feature_scores_abs(&mut scores);
    Ok(scores)
}

pub fn screen_pairwise_interactions(
    dataset: &NumericDataset,
    max_results: usize,
) -> Result<Vec<InteractionScore>, String> {
    dataset.validate()?;
    let p = dataset.n_features();
    if p < 2 {
        return Ok(Vec::new());
    }
    let cols = columns(&dataset.records);
    let zcols: Vec<Vec<f64>> = cols.iter().map(|col| standardize_vector(col).0).collect();
    let main: Vec<f64> = zcols
        .iter()
        .map(|col| pearson_correlation(col, &dataset.target).abs())
        .collect();
    let mut scores = Vec::new();
    for i in 0..p {
        for j in i + 1..p {
            let product: Vec<f64> = zcols[i].iter().zip(&zcols[j]).map(|(a, b)| a * b).collect();
            let signed = pearson_correlation(&product, &dataset.target);
            let score = signed.abs();
            let main_effect_max = main[i].max(main[j]);
            scores.push(InteractionScore {
                left: i,
                right: j,
                left_name: dataset.feature_names[i].clone(),
                right_name: dataset.feature_names[j].clone(),
                score,
                signed_score: signed,
                main_effect_max,
                synergy: score - main_effect_max,
            });
        }
    }
    scores.sort_by(|a, b| {
        b.synergy
            .partial_cmp(&a.synergy)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
    });
    scores.truncate(max_results.min(scores.len()));
    Ok(scores)
}

fn sort_feature_scores_abs(scores: &mut [FeatureScore]) {
    scores.sort_by(|a, b| {
        b.score
            .abs()
            .partial_cmp(&a.score.abs())
            .unwrap_or(Ordering::Equal)
    });
}

// =============================================================================
// Shared model metrics
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionMetrics {
    pub model: String,
    pub mse: f64,
    pub mae: f64,
    pub r2: f64,
}

pub trait RegressionPredictor {
    fn predict_one(&self, x: &[f64]) -> f64;

    fn predict_batch(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_one(row)).collect()
    }
}

pub fn regression_metrics<M: RegressionPredictor>(
    model_name: impl Into<String>,
    model: &M,
    dataset: &NumericDataset,
) -> RegressionMetrics {
    let pred = model.predict_batch(&dataset.records);
    let n = dataset.n_rows().max(1) as f64;
    let mse = squared_error(&dataset.target, &pred) / n;
    let mae = dataset
        .target
        .iter()
        .zip(&pred)
        .map(|(y, yhat)| (y - yhat).abs())
        .sum::<f64>()
        / n;
    let y_mean = mean(&dataset.target);
    let ss_tot = dataset
        .target
        .iter()
        .map(|y| (y - y_mean) * (y - y_mean))
        .sum::<f64>();
    let r2 = if ss_tot <= EPS {
        if mse <= EPS {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - squared_error(&dataset.target, &pred) / ss_tot
    };
    RegressionMetrics {
        model: model_name.into(),
        mse,
        mae,
        r2,
    }
}

// =============================================================================
// Elastic Net / LASSO
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNetParams {
    /// Overall regularization strength. Larger values produce sparser models.
    pub alpha: f64,
    /// `1.0` is LASSO, `0.0` is ridge, values in between are Elastic Net.
    pub l1_ratio: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for ElasticNetParams {
    fn default() -> Self {
        ElasticNetParams {
            alpha: 0.01,
            l1_ratio: 0.8,
            max_iter: 1000,
            tol: 1.0e-7,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FeatureTransformKind {
    Base { index: usize },
    Interaction { left: usize, right: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureTransform {
    pub name: String,
    pub kind: FeatureTransformKind,
    pub mean: f64,
    pub scale: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNetModel {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub transforms: Vec<FeatureTransform>,
    pub base_means: Vec<f64>,
    pub base_scales: Vec<f64>,
    pub loss_history: Vec<f64>,
}

impl ElasticNetModel {
    pub fn fit(dataset: &NumericDataset, params: &ElasticNetParams) -> Result<Self, String> {
        Self::fit_with_interactions(dataset, params, &[])
    }

    pub fn fit_with_interactions(
        dataset: &NumericDataset,
        params: &ElasticNetParams,
        interaction_pairs: &[(usize, usize)],
    ) -> Result<Self, String> {
        dataset.validate()?;
        validate_elastic_net_params(params)?;
        for &(a, b) in interaction_pairs {
            if a >= dataset.n_features() || b >= dataset.n_features() || a == b {
                return Err(format!("invalid interaction pair ({a}, {b})"));
            }
        }

        let design = build_design_matrix(dataset, interaction_pairs)?;
        let n = dataset.n_rows();
        let p = design.z[0].len();
        let y_mean = mean(&dataset.target);
        let y_centered: Vec<f64> = dataset.target.iter().map(|y| y - y_mean).collect();
        let mut beta = vec![0.0; p];
        let mut pred = vec![0.0; n];
        let mut loss_history = Vec::new();

        for _iter in 0..params.max_iter {
            let old = beta.clone();
            for j in 0..p {
                let old_beta = beta[j];
                let mut rho = 0.0;
                let mut z_norm = 0.0;
                for i in 0..n {
                    let xij = design.z[i][j];
                    let residual_without_j = y_centered[i] - (pred[i] - xij * old_beta);
                    rho += xij * residual_without_j;
                    z_norm += xij * xij;
                }
                rho /= n as f64;
                z_norm /= n as f64;
                let l1 = params.alpha * params.l1_ratio;
                let l2 = params.alpha * (1.0 - params.l1_ratio);
                let new_beta = soft_threshold(rho, l1) / (z_norm + l2);
                let delta = new_beta - old_beta;
                if delta.abs() > 0.0 {
                    for i in 0..n {
                        pred[i] += design.z[i][j] * delta;
                    }
                }
                beta[j] = new_beta;
            }
            let mse = y_centered
                .iter()
                .zip(&pred)
                .map(|(y, yhat)| {
                    let e = y - yhat;
                    e * e
                })
                .sum::<f64>()
                / n as f64;
            let penalty = params.alpha
                * (params.l1_ratio * beta.iter().map(|b| b.abs()).sum::<f64>()
                    + 0.5 * (1.0 - params.l1_ratio) * beta.iter().map(|b| b * b).sum::<f64>());
            loss_history.push(0.5 * mse + penalty);
            let max_delta = beta
                .iter()
                .zip(&old)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if max_delta <= params.tol {
                break;
            }
        }

        Ok(ElasticNetModel {
            intercept: y_mean,
            coefficients: beta,
            transforms: design.transforms,
            base_means: design.base_means,
            base_scales: design.base_scales,
            loss_history,
        })
    }

    pub fn selected_features(&self, tol: f64) -> Vec<FeatureScore> {
        let mut out: Vec<FeatureScore> = self
            .coefficients
            .iter()
            .enumerate()
            .filter(|(_, b)| b.abs() > tol)
            .map(|(index, &score)| FeatureScore {
                index,
                name: self.transforms[index].name.clone(),
                score,
            })
            .collect();
        sort_feature_scores_abs(&mut out);
        out
    }
}

impl RegressionPredictor for ElasticNetModel {
    fn predict_one(&self, x: &[f64]) -> f64 {
        let mut y = self.intercept;
        for (coef, transform) in self.coefficients.iter().zip(&self.transforms) {
            y += coef
                * standardized_transform_value(x, transform, &self.base_means, &self.base_scales);
        }
        y
    }
}

struct DesignMatrix {
    z: Matrix,
    transforms: Vec<FeatureTransform>,
    base_means: Vec<f64>,
    base_scales: Vec<f64>,
}

fn build_design_matrix(
    dataset: &NumericDataset,
    interaction_pairs: &[(usize, usize)],
) -> Result<DesignMatrix, String> {
    let n = dataset.n_rows();
    let p = dataset.n_features();
    let cols = columns(&dataset.records);
    let mut base_means = Vec::with_capacity(p);
    let mut base_scales = Vec::with_capacity(p);
    let mut base_z = vec![vec![0.0; n]; p];
    let mut raw_features: Vec<Vec<f64>> = Vec::new();
    let mut names = Vec::new();
    let mut kinds = Vec::new();

    for j in 0..p {
        let (z, m, s) = standardize_vector(&cols[j]);
        base_means.push(m);
        base_scales.push(s);
        base_z[j] = z.clone();
        raw_features.push(cols[j].clone());
        names.push(dataset.feature_names[j].clone());
        kinds.push(FeatureTransformKind::Base { index: j });
    }

    for &(a, b) in interaction_pairs {
        let product: Vec<f64> = base_z[a]
            .iter()
            .zip(&base_z[b])
            .map(|(x, y)| x * y)
            .collect();
        raw_features.push(product);
        names.push(format!(
            "{}*{}",
            dataset.feature_names[a], dataset.feature_names[b]
        ));
        kinds.push(FeatureTransformKind::Interaction { left: a, right: b });
    }

    let q = raw_features.len();
    let mut transforms = Vec::with_capacity(q);
    let mut z = vec![vec![0.0; q]; n];
    for j in 0..q {
        let (col_z, col_mean, col_scale) = standardize_vector(&raw_features[j]);
        for i in 0..n {
            z[i][j] = col_z[i];
        }
        transforms.push(FeatureTransform {
            name: names[j].clone(),
            kind: kinds[j].clone(),
            mean: col_mean,
            scale: col_scale,
        });
    }

    Ok(DesignMatrix {
        z,
        transforms,
        base_means,
        base_scales,
    })
}

fn standardized_transform_value(
    x: &[f64],
    transform: &FeatureTransform,
    base_means: &[f64],
    base_scales: &[f64],
) -> f64 {
    let raw = match transform.kind {
        FeatureTransformKind::Base { index } => x[index],
        FeatureTransformKind::Interaction { left, right } => {
            let zl = (x[left] - base_means[left]) / base_scales[left];
            let zr = (x[right] - base_means[right]) / base_scales[right];
            zl * zr
        }
    };
    (raw - transform.mean) / transform.scale
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

fn validate_elastic_net_params(params: &ElasticNetParams) -> Result<(), String> {
    if params.alpha < 0.0 || !params.alpha.is_finite() {
        return Err(format!(
            "alpha must be finite and non-negative, got {}",
            params.alpha
        ));
    }
    if !(0.0..=1.0).contains(&params.l1_ratio) || !params.l1_ratio.is_finite() {
        return Err(format!(
            "l1_ratio must be in [0, 1], got {}",
            params.l1_ratio
        ));
    }
    if params.max_iter == 0 {
        return Err("max_iter must be positive".to_string());
    }
    if params.tol <= 0.0 || !params.tol.is_finite() {
        return Err(format!(
            "tol must be positive and finite, got {}",
            params.tol
        ));
    }
    Ok(())
}

// =============================================================================
// CART regression tree kernel
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTreeParams {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl Default for RegressionTreeParams {
    fn default() -> Self {
        RegressionTreeParams {
            max_depth: 4,
            min_samples_split: 4,
            min_samples_leaf: 2,
            max_features: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        value: f64,
        gain: f64,
        samples: usize,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTree {
    pub root: TreeNode,
    pub feature_importances: Vec<f64>,
}

impl RegressionTree {
    pub fn fit<'data, 'rng>(
        x: &'data [Vec<f64>],
        y: &'data [f64],
        params: &'data RegressionTreeParams,
        rng: Option<&'rng mut dyn RandomSource>,
    ) -> Result<Self, String> {
        let rows: Vec<usize> = (0..x.len()).collect();
        Self::fit_rows(x, y, params, &rows, rng)
    }

    pub fn fit_rows<'data, 'rng>(
        x: &'data [Vec<f64>],
        y: &'data [f64],
        params: &'data RegressionTreeParams,
        rows: &[usize],
        rng: Option<&'rng mut dyn RandomSource>,
    ) -> Result<Self, String> {
        validate_supervised_matrix(x, y)?;
        validate_tree_params(params)?;
        if rows.is_empty() {
            return Err("tree requires at least one training row".to_string());
        }
        let n_features = x[0].len();
        for &row in rows {
            if row >= x.len() {
                return Err(format!("row index {row} out of range {}", x.len()));
            }
        }
        let mut builder = TreeBuilder {
            x,
            y,
            params,
            n_features,
            importances: vec![0.0; n_features],
            rng,
        };
        let root = builder.build(rows.to_vec(), 0);
        Ok(RegressionTree {
            root,
            feature_importances: normalize_importances(&builder.importances),
        })
    }

    pub fn raw_feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    fn predict_node(node: &TreeNode, row: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                if row[*feature] <= *threshold {
                    Self::predict_node(left, row)
                } else {
                    Self::predict_node(right, row)
                }
            }
        }
    }
}

impl RegressionPredictor for RegressionTree {
    fn predict_one(&self, x: &[f64]) -> f64 {
        Self::predict_node(&self.root, x)
    }
}

struct TreeBuilder<'data, 'rng> {
    x: &'data [Vec<f64>],
    y: &'data [f64],
    params: &'data RegressionTreeParams,
    n_features: usize,
    importances: Vec<f64>,
    rng: Option<&'rng mut dyn RandomSource>,
}

#[derive(Clone, Copy, Debug)]
struct SplitCandidate {
    feature: usize,
    threshold: f64,
    gain: f64,
}

impl TreeBuilder<'_, '_> {
    fn build(&mut self, rows: Vec<usize>, depth: usize) -> TreeNode {
        let value = rows.iter().map(|&i| self.y[i]).sum::<f64>() / rows.len() as f64;
        if depth >= self.params.max_depth || rows.len() < self.params.min_samples_split {
            return TreeNode::Leaf {
                value,
                samples: rows.len(),
            };
        }

        let Some(split) = self.best_split(&rows) else {
            return TreeNode::Leaf {
                value,
                samples: rows.len(),
            };
        };
        if split.gain <= EPS {
            return TreeNode::Leaf {
                value,
                samples: rows.len(),
            };
        }

        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for row in rows {
            if self.x[row][split.feature] <= split.threshold {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        if left_rows.len() < self.params.min_samples_leaf
            || right_rows.len() < self.params.min_samples_leaf
        {
            return TreeNode::Leaf {
                value,
                samples: left_rows.len() + right_rows.len(),
            };
        }

        self.importances[split.feature] += split.gain;
        let samples = left_rows.len() + right_rows.len();
        let left = self.build(left_rows, depth + 1);
        let right = self.build(right_rows, depth + 1);
        TreeNode::Split {
            feature: split.feature,
            threshold: split.threshold,
            value,
            gain: split.gain,
            samples,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn best_split(&mut self, rows: &[usize]) -> Option<SplitCandidate> {
        let parent_sse = sse_for_rows(self.y, rows);
        let features = self.candidate_features();
        let mut best: Option<SplitCandidate> = None;
        for feature in features {
            if let Some(candidate) = self.best_feature_split(rows, feature, parent_sse) {
                if best.map_or(true, |b| candidate.gain > b.gain) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    fn candidate_features(&mut self) -> Vec<usize> {
        let mut features: Vec<usize> = (0..self.n_features).collect();
        let Some(k_requested) = self.params.max_features else {
            return features;
        };
        let k = k_requested.clamp(1, self.n_features);
        if k >= self.n_features {
            return features;
        }
        if let Some(rng) = self.rng.as_deref_mut() {
            for i in (1..features.len()).rev() {
                let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
                features.swap(i, j);
            }
        } else {
            features.rotate_left(k % self.n_features);
        }
        features.truncate(k);
        features
    }

    fn best_feature_split(
        &self,
        rows: &[usize],
        feature: usize,
        parent_sse: f64,
    ) -> Option<SplitCandidate> {
        let mut pairs: Vec<(f64, f64)> = rows
            .iter()
            .map(|&i| (self.x[i][feature], self.y[i]))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let n = pairs.len();
        if n < 2 * self.params.min_samples_leaf {
            return None;
        }
        let total_sum = pairs.iter().map(|(_, y)| y).sum::<f64>();
        let total_sq = pairs.iter().map(|(_, y)| y * y).sum::<f64>();
        let mut left_sum = 0.0;
        let mut left_sq = 0.0;
        let mut left_n = 0usize;
        let mut best: Option<SplitCandidate> = None;

        for i in 0..n - 1 {
            let (_, yi) = pairs[i];
            left_n += 1;
            left_sum += yi;
            left_sq += yi * yi;
            if (pairs[i + 1].0 - pairs[i].0).abs() <= EPS {
                continue;
            }
            let right_n = n - left_n;
            if left_n < self.params.min_samples_leaf || right_n < self.params.min_samples_leaf {
                continue;
            }
            let right_sum = total_sum - left_sum;
            let right_sq = total_sq - left_sq;
            let left_sse = left_sq - left_sum * left_sum / left_n as f64;
            let right_sse = right_sq - right_sum * right_sum / right_n as f64;
            let gain = parent_sse - left_sse - right_sse;
            if gain <= EPS {
                continue;
            }
            let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
            let candidate = SplitCandidate {
                feature,
                threshold,
                gain,
            };
            if best.map_or(true, |b| candidate.gain > b.gain) {
                best = Some(candidate);
            }
        }
        best
    }
}

fn validate_supervised_matrix(x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
    if x.is_empty() {
        return Err("matrix must contain at least one row".to_string());
    }
    if x.len() != y.len() {
        return Err(format!(
            "matrix row count {} does not match target length {}",
            x.len(),
            y.len()
        ));
    }
    let p = x[0].len();
    if p == 0 {
        return Err("matrix must contain at least one column".to_string());
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(format!("ragged matrix at row {i}"));
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("non-finite matrix value at ({i}, {j}): {v}"));
            }
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("non-finite target at row {i}: {v}"));
        }
    }
    Ok(())
}

fn validate_tree_params(params: &RegressionTreeParams) -> Result<(), String> {
    if params.min_samples_split == 0 {
        return Err("min_samples_split must be positive".to_string());
    }
    if params.min_samples_leaf == 0 {
        return Err("min_samples_leaf must be positive".to_string());
    }
    Ok(())
}

fn sse_for_rows(y: &[f64], rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let sum = rows.iter().map(|&i| y[i]).sum::<f64>();
    let sq = rows.iter().map(|&i| y[i] * y[i]).sum::<f64>();
    sq - sum * sum / rows.len() as f64
}

fn normalize_importances(xs: &[f64]) -> Vec<f64> {
    let total: f64 = xs.iter().sum();
    if total <= EPS {
        vec![0.0; xs.len()]
    } else {
        xs.iter().map(|x| x / total).collect()
    }
}

// =============================================================================
// Gradient boosting
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct GradientBoostingParams {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

impl Default for GradientBoostingParams {
    fn default() -> Self {
        GradientBoostingParams {
            n_estimators: 60,
            learning_rate: 0.08,
            max_depth: 3,
            min_samples_leaf: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientBoostingRegressor {
    pub initial_prediction: f64,
    pub trees: Vec<RegressionTree>,
    pub learning_rate: f64,
    pub loss_history: Vec<f64>,
    pub feature_importances: Vec<f64>,
}

impl GradientBoostingRegressor {
    pub fn fit(dataset: &NumericDataset, params: &GradientBoostingParams) -> Result<Self, String> {
        dataset.validate()?;
        validate_gradient_boosting_params(params)?;
        let n = dataset.n_rows();
        let p = dataset.n_features();
        let initial_prediction = mean(&dataset.target);
        let mut pred = vec![initial_prediction; n];
        let mut trees = Vec::with_capacity(params.n_estimators);
        let mut loss_history = Vec::with_capacity(params.n_estimators);
        let mut importances = vec![0.0; p];

        for _ in 0..params.n_estimators {
            let residuals: Vec<f64> = dataset
                .target
                .iter()
                .zip(&pred)
                .map(|(y, yhat)| y - yhat)
                .collect();
            let tree_params = RegressionTreeParams {
                max_depth: params.max_depth,
                min_samples_split: (2 * params.min_samples_leaf).max(2),
                min_samples_leaf: params.min_samples_leaf,
                max_features: None,
            };
            let tree = RegressionTree::fit(&dataset.records, &residuals, &tree_params, None)?;
            for (i, row) in dataset.records.iter().enumerate() {
                pred[i] += params.learning_rate * tree.predict_one(row);
            }
            for (j, imp) in tree.feature_importances.iter().enumerate() {
                importances[j] += imp;
            }
            loss_history.push(squared_error(&dataset.target, &pred) / n as f64);
            trees.push(tree);
        }

        Ok(GradientBoostingRegressor {
            initial_prediction,
            trees,
            learning_rate: params.learning_rate,
            loss_history,
            feature_importances: normalize_importances(&importances),
        })
    }
}

impl RegressionPredictor for GradientBoostingRegressor {
    fn predict_one(&self, x: &[f64]) -> f64 {
        let mut y = self.initial_prediction;
        for tree in &self.trees {
            y += self.learning_rate * tree.predict_one(x);
        }
        y
    }
}

fn validate_gradient_boosting_params(params: &GradientBoostingParams) -> Result<(), String> {
    if params.n_estimators == 0 {
        return Err("n_estimators must be positive".to_string());
    }
    if params.learning_rate <= 0.0 || !params.learning_rate.is_finite() {
        return Err(format!(
            "learning_rate must be positive and finite, got {}",
            params.learning_rate
        ));
    }
    if params.min_samples_leaf == 0 {
        return Err("min_samples_leaf must be positive".to_string());
    }
    Ok(())
}

// =============================================================================
// Random forest
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestParams {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub sample_rate: f64,
    pub seed: u32,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        RandomForestParams {
            n_trees: 64,
            max_depth: 6,
            min_samples_leaf: 2,
            max_features: None,
            sample_rate: 1.0,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestRegressor {
    pub trees: Vec<RegressionTree>,
    pub feature_importances: Vec<f64>,
}

impl RandomForestRegressor {
    pub fn fit(dataset: &NumericDataset, params: &RandomForestParams) -> Result<Self, String> {
        dataset.validate()?;
        validate_random_forest_params(params)?;
        let n = dataset.n_rows();
        let p = dataset.n_features();
        let max_features = params
            .max_features
            .unwrap_or_else(|| (p as f64).sqrt().ceil().max(1.0) as usize)
            .clamp(1, p);
        let sample_size = ((n as f64) * params.sample_rate).ceil().max(1.0) as usize;
        let mut rng = mulberry32(params.seed);
        let mut trees = Vec::with_capacity(params.n_trees);
        let mut importances = vec![0.0; p];

        for _ in 0..params.n_trees {
            let rows = bootstrap_rows(n, sample_size, &mut rng);
            let tree_params = RegressionTreeParams {
                max_depth: params.max_depth,
                min_samples_split: (2 * params.min_samples_leaf).max(2),
                min_samples_leaf: params.min_samples_leaf,
                max_features: Some(max_features),
            };
            let tree = RegressionTree::fit_rows(
                &dataset.records,
                &dataset.target,
                &tree_params,
                &rows,
                Some(&mut rng),
            )?;
            for (j, imp) in tree.feature_importances.iter().enumerate() {
                importances[j] += imp;
            }
            trees.push(tree);
        }

        Ok(RandomForestRegressor {
            trees,
            feature_importances: normalize_importances(&importances),
        })
    }
}

impl RegressionPredictor for RandomForestRegressor {
    fn predict_one(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        self.trees
            .iter()
            .map(|tree| tree.predict_one(x))
            .sum::<f64>()
            / self.trees.len() as f64
    }
}

fn bootstrap_rows(n: usize, sample_size: usize, rng: &mut impl RandomSource) -> Vec<usize> {
    (0..sample_size)
        .map(|_| (rng.next_float() * n as f64).floor() as usize)
        .map(|i| i.min(n - 1))
        .collect()
}

fn validate_random_forest_params(params: &RandomForestParams) -> Result<(), String> {
    if params.n_trees == 0 {
        return Err("n_trees must be positive".to_string());
    }
    if params.sample_rate <= 0.0 || !params.sample_rate.is_finite() {
        return Err(format!(
            "sample_rate must be positive and finite, got {}",
            params.sample_rate
        ));
    }
    if params.min_samples_leaf == 0 {
        return Err("min_samples_leaf must be positive".to_string());
    }
    Ok(())
}

// =============================================================================
// Neural-network regressor
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct NeuralNetworkParams {
    pub hidden_layers: Vec<usize>,
    pub hidden_activation: ActivationName,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u32,
    pub shuffle_each_epoch: bool,
}

impl Default for NeuralNetworkParams {
    fn default() -> Self {
        NeuralNetworkParams {
            hidden_layers: vec![8],
            hidden_activation: ActivationName::Tanh,
            epochs: 500,
            learning_rate: 0.03,
            seed: 13,
            shuffle_each_epoch: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuralNetworkRegressor {
    pub network: FeedForwardNetwork,
    pub feature_means: Vec<f64>,
    pub feature_scales: Vec<f64>,
    pub target_mean: f64,
    pub target_scale: f64,
    pub loss_history: Vec<f64>,
}

impl NeuralNetworkRegressor {
    pub fn fit(dataset: &NumericDataset, params: &NeuralNetworkParams) -> Result<Self, String> {
        dataset.validate()?;
        validate_neural_network_params(params)?;
        let p = dataset.n_features();
        let cols = columns(&dataset.records);
        let mut feature_means = Vec::with_capacity(p);
        let mut feature_scales = Vec::with_capacity(p);
        for col in &cols {
            let (_, m, s) = standardize_vector(col);
            feature_means.push(m);
            feature_scales.push(s);
        }
        let (target_z, target_mean, target_scale) = standardize_vector(&dataset.target);
        let x_z: Matrix = dataset
            .records
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, x)| (x - feature_means[j]) / feature_scales[j])
                    .collect()
            })
            .collect();

        let mut rng = mulberry32(params.seed);
        let mut network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: p,
                hidden_layers: params.hidden_layers.clone(),
                output_dim: 1,
                hidden_activation: params.hidden_activation,
                output_activation: ActivationName::Linear,
                weight_scale: None,
            },
            &mut rng,
        );

        let mut order: Vec<usize> = (0..dataset.n_rows()).collect();
        let mut loss_history = Vec::with_capacity(params.epochs);
        for _ in 0..params.epochs {
            if params.shuffle_each_epoch {
                shuffle_in_place(&mut order, &mut rng);
            }
            let mut total_loss = 0.0;
            for &i in &order {
                total_loss += network
                    .train_sample(&x_z[i], &[target_z[i]], params.learning_rate)
                    .loss;
            }
            loss_history.push(total_loss / dataset.n_rows() as f64);
        }

        Ok(NeuralNetworkRegressor {
            network,
            feature_means,
            feature_scales,
            target_mean,
            target_scale,
            loss_history,
        })
    }
}

impl RegressionPredictor for NeuralNetworkRegressor {
    fn predict_one(&self, x: &[f64]) -> f64 {
        let z: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(j, v)| (v - self.feature_means[j]) / self.feature_scales[j])
            .collect();
        let scaled = self.network.predict(&z)[0];
        self.target_mean + self.target_scale * scaled
    }
}

fn validate_neural_network_params(params: &NeuralNetworkParams) -> Result<(), String> {
    if params.epochs == 0 {
        return Err("epochs must be positive".to_string());
    }
    if params.learning_rate <= 0.0 || !params.learning_rate.is_finite() {
        return Err(format!(
            "learning_rate must be positive and finite, got {}",
            params.learning_rate
        ));
    }
    if params.hidden_layers.iter().any(|&n| n == 0) {
        return Err("hidden layer widths must be positive".to_string());
    }
    Ok(())
}

fn shuffle_in_place<T>(xs: &mut [T], rng: &mut impl RandomSource) {
    for i in (1..xs.len()).rev() {
        let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
        xs.swap(i, j);
    }
}

// =============================================================================
// Big-data pipeline orchestration
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct BigDataPipelineConfig {
    pub correlation_kind: CorrelationKind,
    pub max_interactions: usize,
    pub elastic_net_interactions: usize,
    pub elastic_net: ElasticNetParams,
    pub gradient_boosting: GradientBoostingParams,
    pub random_forest: RandomForestParams,
    pub neural_network: NeuralNetworkParams,
}

impl Default for BigDataPipelineConfig {
    fn default() -> Self {
        BigDataPipelineConfig {
            correlation_kind: CorrelationKind::Pearson,
            max_interactions: 25,
            elastic_net_interactions: 10,
            elastic_net: ElasticNetParams::default(),
            gradient_boosting: GradientBoostingParams::default(),
            random_forest: RandomForestParams::default(),
            neural_network: NeuralNetworkParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BigDataPipelineReport {
    pub correlations: CorrelationMatrix,
    pub target_correlations: Vec<FeatureScore>,
    pub interactions: Vec<InteractionScore>,
    pub elastic_net: ElasticNetModel,
    pub gradient_boosting: GradientBoostingRegressor,
    pub random_forest: RandomForestRegressor,
    pub neural_network: NeuralNetworkRegressor,
    pub metrics: Vec<RegressionMetrics>,
}

impl BigDataPipelineReport {
    pub fn best_metric(&self) -> Option<&RegressionMetrics> {
        self.metrics
            .iter()
            .max_by(|a, b| a.r2.partial_cmp(&b.r2).unwrap_or(Ordering::Equal))
    }
}

pub fn run_big_data_pipeline(
    dataset: &NumericDataset,
    config: &BigDataPipelineConfig,
) -> Result<BigDataPipelineReport, String> {
    dataset.validate()?;
    let correlations = correlation_matrix(dataset, config.correlation_kind)?;
    let target_corrs = target_correlations(dataset, config.correlation_kind)?;
    let interactions = screen_pairwise_interactions(dataset, config.max_interactions)?;
    let selected_pairs: Vec<(usize, usize)> = interactions
        .iter()
        .take(config.elastic_net_interactions)
        .map(|s| (s.left, s.right))
        .collect();

    let elastic_net =
        ElasticNetModel::fit_with_interactions(dataset, &config.elastic_net, &selected_pairs)?;
    let gradient_boosting = GradientBoostingRegressor::fit(dataset, &config.gradient_boosting)?;
    let random_forest = RandomForestRegressor::fit(dataset, &config.random_forest)?;
    let neural_network = NeuralNetworkRegressor::fit(dataset, &config.neural_network)?;

    let metrics = vec![
        regression_metrics("elastic_net", &elastic_net, dataset),
        regression_metrics("gradient_boosting", &gradient_boosting, dataset),
        regression_metrics("random_forest", &random_forest, dataset),
        regression_metrics("neural_network", &neural_network, dataset),
    ];

    Ok(BigDataPipelineReport {
        correlations,
        target_correlations: target_corrs,
        interactions,
        elastic_net,
        gradient_boosting,
        random_forest,
        neural_network,
        metrics,
    })
}

/// Generate a two-level fractional-factorial design using a simple generator
/// matrix. Each factor row contains `-1` / `+1` levels. Generator columns may
/// reference earlier factor indices whose signs are multiplied together.
///
/// This is a compact helper for fractional experiments; it does not try to
/// optimize alias structure automatically.
pub fn fractional_factorial_design(
    base_factors: usize,
    generators: &[Vec<usize>],
) -> Result<Matrix, String> {
    if base_factors == 0 {
        return Err("base_factors must be positive".to_string());
    }
    if base_factors >= usize::BITS as usize {
        return Err("too many base factors for in-memory design".to_string());
    }
    let runs = 1usize << base_factors;
    let mut design = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut row = Vec::with_capacity(base_factors + generators.len());
        for j in 0..base_factors {
            let bit = (run >> j) & 1;
            row.push(if bit == 0 { -1.0 } else { 1.0 });
        }
        for generator in generators {
            if generator.is_empty() {
                return Err("generator cannot be empty".to_string());
            }
            let mut value = 1.0;
            for &idx in generator {
                if idx >= row.len() {
                    return Err(format!(
                        "generator references factor {idx} before it exists"
                    ));
                }
                value *= row[idx];
            }
            row.push(value);
        }
        design.push(row);
    }
    Ok(design)
}
