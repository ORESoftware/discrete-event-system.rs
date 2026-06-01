//! Curve fitting via parametric GA, piecewise models, GP symbolic regression,
//! and hybrid ridge refinement with batched residual evaluation.

use std::f64::consts::PI;

use crate::des::general::evolution::ga_core::{
    run_ga, FitnessEvaluator, GaOptions, GaResult, GeneticOperators, PopulationInitializer,
};
use crate::des::general::evolution::genetic_programming::tree_size;
use crate::des::general::evolution::genetic_programming::{
    run_gp, GpCurveProblem, GpFlavor, GpOptions, GpResult,
};
use crate::des::general::evolution::gpu_batch::{
    residuals_for_designs_with_backend, residuals_with_backend,
};
use crate::des::general::expr::{Expr, ExprEvaluator};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::linalg::{LinAlg, LinearSystem, Matrix, Vector};

// =============================================================================
// Data + metrics
// =============================================================================

#[derive(Clone, Debug)]
pub struct CurveDataset {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Per-point weights (defaults to 1.0 if empty).
    pub weights: Vec<f64>,
    /// Independent variable names (default `["x"]`).
    pub var_names: Vec<String>,
}

impl CurveDataset {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        CurveDataset {
            x,
            y,
            weights: Vec::new(),
            var_names: vec!["x".to_string()],
        }
    }

    pub fn with_weights(mut self, w: Vec<f64>) -> Self {
        assert_eq!(w.len(), self.x.len());
        self.weights = w;
        self
    }

    pub fn weight(&self, i: usize) -> f64 {
        if self.weights.is_empty() {
            1.0
        } else {
            self.weights[i]
        }
    }

    pub fn train_holdout_split(&self, holdout_frac: f64) -> (CurveDataset, CurveDataset) {
        let n = self.x.len();
        let cut = ((1.0 - holdout_frac.clamp(0.05, 0.5)) * n as f64).floor() as usize;
        let cut = cut.clamp(1, n.saturating_sub(1));
        let train = CurveDataset {
            x: self.x[..cut].to_vec(),
            y: self.y[..cut].to_vec(),
            weights: if self.weights.is_empty() {
                Vec::new()
            } else {
                self.weights[..cut].to_vec()
            },
            var_names: self.var_names.clone(),
        };
        let test = CurveDataset {
            x: self.x[cut..].to_vec(),
            y: self.y[cut..].to_vec(),
            weights: if self.weights.is_empty() {
                Vec::new()
            } else {
                self.weights[cut..].to_vec()
            },
            var_names: self.var_names.clone(),
        };
        (train, test)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMetric {
    Mse,
    Mae,
    Huber,
}

#[derive(Clone, Debug, Default)]
pub struct CurveConstraints {
    pub max_terms: Option<usize>,
    pub require_monotone: bool,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub max_abs_coeff: Option<f64>,
    pub ridge: Option<f64>,
}

// =============================================================================
// Parametric families
// =============================================================================

#[derive(Clone, Debug)]
pub enum ParametricFamily {
    Polynomial {
        degree: usize,
    },
    Fourier {
        harmonics: usize,
    },
    ExpSum {
        terms: usize,
    },
    Rational {
        degree_num: usize,
        degree_den: usize,
    },
}

/// Unit-box encoded parameters mapped to physical bounds per gene.
#[derive(Clone, Debug)]
pub struct ParametricChromosome {
    pub family: ParametricFamily,
    /// Genes in [0, 1].
    pub genes: Vec<f64>,
    pub bounds: Vec<(f64, f64)>,
}

impl ParametricChromosome {
    pub fn decode(&self) -> Vec<f64> {
        self.genes
            .iter()
            .zip(&self.bounds)
            .map(|(&g, &(lo, hi))| lo + g.clamp(0.0, 1.0) * (hi - lo))
            .collect()
    }
}

pub fn design_matrix(family: &ParametricFamily, data: &CurveDataset, params: &[f64]) -> Matrix {
    let n = data.x.len();
    let cols = parametric_basis_width(family);
    let mut x = vec![vec![0.0; cols]; n];
    for (i, row) in x.iter_mut().enumerate().take(n) {
        let xv = data.x[i];
        fill_basis_row(family, xv, params, row);
    }
    x
}

fn parametric_basis_width(family: &ParametricFamily) -> usize {
    match family {
        ParametricFamily::Polynomial { degree } => degree + 1,
        ParametricFamily::Fourier { harmonics } => 1 + 2 * harmonics,
        ParametricFamily::ExpSum { terms } => 2 * terms,
        ParametricFamily::Rational {
            degree_num,
            degree_den,
        } => degree_num + 1 + degree_den,
    }
}

fn fill_basis_row(family: &ParametricFamily, x: f64, params: &[f64], row: &mut [f64]) {
    match family {
        ParametricFamily::Polynomial { degree } => {
            let mut p = 1.0;
            for c in row.iter_mut().take(*degree + 1) {
                *c = p;
                p *= x;
            }
        }
        ParametricFamily::Fourier { harmonics } => {
            row[0] = 1.0;
            let w = params.first().copied().unwrap_or(1.0);
            for h in 0..*harmonics {
                row[1 + 2 * h] = (w * (h + 1) as f64 * x).sin();
                row[2 + 2 * h] = (w * (h + 1) as f64 * x).cos();
            }
        }
        ParametricFamily::ExpSum { terms } => {
            for t in 0..*terms {
                let a = params.get(2 * t).copied().unwrap_or(0.0);
                let b = params.get(2 * t + 1).copied().unwrap_or(1.0);
                row[2 * t] = (a * x).exp();
                row[2 * t + 1] = (b * x).sin();
            }
        }
        ParametricFamily::Rational {
            degree_num,
            degree_den,
        } => {
            let mut num = vec![0.0; degree_num + 1];
            let mut den = vec![0.0; degree_den + 1];
            let mut p = 1.0;
            for c in num.iter_mut() {
                *c = p;
                p *= x;
            }
            p = 1.0;
            for c in den.iter_mut() {
                *c = p;
                p *= x;
            }
            let nv: f64 = num
                .iter()
                .zip(&params[..=*degree_num])
                .map(|(b, &c)| b * c)
                .sum();
            let dv: f64 = den
                .iter()
                .zip(&params[degree_num + 1..degree_num + 1 + degree_den])
                .map(|(b, &c)| b * c)
                .sum::<f64>()
                + 1e-6;
            row[0] = nv / dv;
            for (j, v) in num.iter().enumerate().skip(1) {
                row[j] = *v / dv;
            }
        }
    }
}

pub fn predict_parametric(
    family: &ParametricFamily,
    coeffs: &[f64],
    shape_params: &[f64],
    x: &[f64],
) -> Vec<f64> {
    x.iter()
        .map(|&xv| {
            let cols = parametric_basis_width(family);
            let mut row = vec![0.0; cols];
            fill_basis_row(family, xv, shape_params, &mut row);
            row.iter().zip(coeffs).map(|(b, c)| b * c).sum()
        })
        .collect()
}

/// Ridge solve for fixed nonlinear shape params (genes) — linear in basis coeffs.
pub fn hybrid_refine(
    family: &ParametricFamily,
    data: &CurveDataset,
    genes: &[f64],
    constraints: &CurveConstraints,
) -> (Vec<f64>, f64) {
    let shape = decode_shape_genes(genes);
    let (xmat, coeffs) = fit_parametric_coefficients(family, data, &shape, constraints);
    let residuals = residuals_with_backend(&xmat, &data.y, std::slice::from_ref(&coeffs));
    let mse = residual_loss(&residuals[0], data, FitMetric::Mse, constraints);
    (coeffs, mse)
}

fn decode_shape_genes(genes: &[f64]) -> Vec<f64> {
    genes.iter().map(|g| g.clamp(0.0, 1.0)).collect()
}

fn fit_parametric_coefficients(
    family: &ParametricFamily,
    data: &CurveDataset,
    shape: &[f64],
    constraints: &CurveConstraints,
) -> (Matrix, Vector) {
    let xmat = design_matrix(family, data, shape);
    let coeffs = ridge_solve(&xmat, &data.y, data, constraints.ridge.unwrap_or(1e-4));
    (xmat, coeffs)
}

fn ridge_solve(x: &Matrix, y: &[f64], data: &CurveDataset, ridge: f64) -> Vector {
    let n = LinAlg::rows(x);
    let p = LinAlg::cols(x);
    let mut a = vec![vec![0.0; p]; p];
    let mut b = vec![0.0; p];
    for i in 0..n {
        let w = data.weight(i);
        for j in 0..p {
            b[j] += w * x[i][j] * y[i];
            for k in 0..p {
                a[j][k] += w * x[i][j] * x[i][k];
            }
        }
    }
    for j in 0..p {
        a[j][j] += ridge;
    }
    LinearSystem::new(&a, &b, 1e-12).solve()
}

fn weighted_mse(actual: &[f64], pred: &[f64], data: &CurveDataset) -> f64 {
    let n = actual.len();
    let mut s = 0.0;
    for i in 0..n {
        let e = actual[i] - pred[i];
        s += data.weight(i) * e * e;
    }
    s / n.max(1) as f64
}

fn residual_loss(
    residuals: &[f64],
    data: &CurveDataset,
    metric: FitMetric,
    constraints: &CurveConstraints,
) -> f64 {
    if residuals.len() != data.y.len() || residuals.iter().any(|e| !e.is_finite()) {
        return f64::INFINITY;
    }
    let n = residuals.len().max(1) as f64;
    let mut loss = match metric {
        FitMetric::Mse => {
            residuals
                .iter()
                .enumerate()
                .map(|(i, e)| data.weight(i) * e * e)
                .sum::<f64>()
                / n
        }
        FitMetric::Mae => {
            residuals
                .iter()
                .enumerate()
                .map(|(i, e)| data.weight(i) * e.abs())
                .sum::<f64>()
                / n
        }
        FitMetric::Huber => {
            let d = 1.0;
            residuals
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let a = e.abs();
                    let h = if a <= d {
                        0.5 * a * a
                    } else {
                        d * (a - 0.5 * d)
                    };
                    data.weight(i) * h
                })
                .sum::<f64>()
                / n
        }
    };

    let pred: Vec<f64> = data.y.iter().zip(residuals).map(|(&y, &e)| y - e).collect();
    if let Some(ymin) = constraints.y_min {
        if pred.iter().any(|&p| p < ymin) {
            loss += 1e6;
        }
    }
    if let Some(ymax) = constraints.y_max {
        if pred.iter().any(|&p| p > ymax) {
            loss += 1e6;
        }
    }
    if constraints.require_monotone {
        for w in pred.windows(2) {
            if w[1] < w[0] - 1e-9 {
                loss += 10.0;
            }
        }
    }
    loss
}

pub fn gp_fitness(
    expr: &Expr,
    data: &CurveDataset,
    constraints: &CurveConstraints,
    metric: FitMetric,
) -> f64 {
    let eval = ExprEvaluator;
    let mut errs = Vec::with_capacity(data.x.len());
    for (i, &xv) in data.x.iter().enumerate() {
        let mut env = crate::des::general::expr::Env::new();
        for name in &data.var_names {
            env.insert(name.clone(), xv);
        }
        let yhat = eval.eval(expr, &env);
        if !yhat.is_finite() {
            return f64::INFINITY;
        }
        if let Some(ymin) = constraints.y_min {
            if yhat < ymin {
                return f64::INFINITY;
            }
        }
        if let Some(ymax) = constraints.y_max {
            if yhat > ymax {
                return f64::INFINITY;
            }
        }
        let e = data.y[i] - yhat;
        errs.push(e);
    }
    let mut loss = match metric {
        FitMetric::Mse => errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64,
        FitMetric::Mae => errs.iter().map(|e| e.abs()).sum::<f64>() / errs.len() as f64,
        FitMetric::Huber => {
            let d = 1.0;
            errs.iter()
                .map(|e| {
                    let a = e.abs();
                    if a <= d {
                        0.5 * a * a
                    } else {
                        d * (a - 0.5 * d)
                    }
                })
                .sum::<f64>()
                / errs.len() as f64
        }
    };
    if constraints.require_monotone {
        let pred = errs
            .iter()
            .enumerate()
            .map(|(i, &e)| data.y[i] - e)
            .collect::<Vec<_>>();
        for w in pred.windows(2) {
            if w[1] < w[0] - 1e-9 {
                loss += 10.0;
            }
        }
    }
    if let Some(max_t) = constraints.max_terms {
        let nodes = tree_size(expr);
        if nodes > max_t {
            loss += (nodes - max_t) as f64;
        }
    }
    loss
}

// =============================================================================
// Parametric GA
// =============================================================================

pub struct ParametricCurveProblem {
    pub data: CurveDataset,
    pub family: ParametricFamily,
    pub constraints: CurveConstraints,
    pub metric: FitMetric,
    pub use_hybrid: bool,
}

impl PopulationInitializer<ParametricChromosome> for ParametricCurveProblem {
    fn initial_population(
        &self,
        size: usize,
        rng: &mut dyn RandomSource,
    ) -> Vec<ParametricChromosome> {
        let dim = match &self.family {
            ParametricFamily::Polynomial { degree } => *degree + 1,
            ParametricFamily::Fourier { harmonics } => 1 + 2 * harmonics,
            ParametricFamily::ExpSum { terms } => 2 * terms,
            ParametricFamily::Rational {
                degree_num,
                degree_den,
            } => degree_num + 1 + degree_den,
        };
        (0..size)
            .map(|_| ParametricChromosome {
                family: self.family.clone(),
                genes: (0..dim).map(|_| rng.next_float()).collect(),
                bounds: vec![(0.0, 1.0); dim],
            })
            .collect()
    }
}

impl FitnessEvaluator<ParametricChromosome> for ParametricCurveProblem {
    fn evaluate(&self, individual: &ParametricChromosome) -> f64 {
        self.evaluate_population(std::slice::from_ref(individual))[0]
    }

    fn evaluate_population(&self, population: &[ParametricChromosome]) -> Vec<f64> {
        if population.is_empty() {
            return Vec::new();
        }

        let mut losses = vec![f64::INFINITY; population.len()];
        let mut active_indices = Vec::with_capacity(population.len());
        let mut designs = Vec::with_capacity(population.len());
        let mut coeffs = Vec::with_capacity(population.len());

        for (i, individual) in population.iter().enumerate() {
            let shape = individual.decode();
            let (xmat, beta) = fit_parametric_coefficients(
                &individual.family,
                &self.data,
                &shape,
                &self.constraints,
            );
            if let Some(max_abs) = self.constraints.max_abs_coeff {
                if beta.iter().any(|c| c.abs() > max_abs) {
                    continue;
                }
            }
            active_indices.push(i);
            designs.push(xmat);
            coeffs.push(beta);
        }

        if designs.is_empty() {
            return losses;
        }

        let residuals = if designs.windows(2).all(|w| w[0] == w[1]) {
            residuals_with_backend(&designs[0], &self.data.y, &coeffs)
        } else {
            residuals_for_designs_with_backend(&designs, &self.data.y, &coeffs)
        };

        for (out_i, residual) in active_indices.into_iter().zip(residuals.iter()) {
            losses[out_i] = residual_loss(residual, &self.data, self.metric, &self.constraints);
        }
        losses
    }
}

impl GeneticOperators<ParametricChromosome> for ParametricCurveProblem {
    fn crossover(
        &self,
        a: &ParametricChromosome,
        b: &ParametricChromosome,
        rng: &mut dyn RandomSource,
    ) -> ParametricChromosome {
        let genes: Vec<f64> = a
            .genes
            .iter()
            .zip(&b.genes)
            .map(|(&x, &y)| if rng.next_float() < 0.5 { x } else { y })
            .collect();
        ParametricChromosome {
            family: a.family.clone(),
            genes,
            bounds: a.bounds.clone(),
        }
    }

    fn mutate(
        &self,
        mut child: ParametricChromosome,
        rng: &mut dyn RandomSource,
    ) -> ParametricChromosome {
        let i = (rng.next_float() * child.genes.len() as f64).floor() as usize % child.genes.len();
        child.genes[i] = (child.genes[i] + rng.next_float() * 0.3 - 0.15).clamp(0.0, 1.0);
        child
    }

    fn local_search(&self, child: ParametricChromosome) -> ParametricChromosome {
        if self.use_hybrid {
            child
        } else {
            child
        }
    }
}

#[derive(Clone, Debug)]
pub struct CurveFitGaResult {
    pub chromosome: ParametricChromosome,
    pub coefficients: Vec<f64>,
    pub train_mse: f64,
    pub ga: GaResult<ParametricChromosome>,
}

pub fn run_curve_fit_ga(
    data: CurveDataset,
    family: ParametricFamily,
    constraints: CurveConstraints,
    ga_opts: GaOptions,
) -> CurveFitGaResult {
    let problem = ParametricCurveProblem {
        data: data.clone(),
        family: family.clone(),
        constraints: constraints.clone(),
        metric: FitMetric::Mse,
        use_hybrid: true,
    };
    let ga = run_ga(problem, ga_opts, None);
    let (coeffs, mse) = hybrid_refine(&family, &data, &ga.best.genes, &constraints);
    CurveFitGaResult {
        chromosome: ga.best.clone(),
        coefficients: coeffs,
        train_mse: mse,
        ga,
    }
}

pub fn run_curve_fit_gp(
    data: CurveDataset,
    constraints: CurveConstraints,
    gp_opts: GpOptions,
) -> GpResult {
    let problem = GpCurveProblem {
        data,
        constraints,
        metric: FitMetric::Mse,
        tree: gp_opts.tree.clone(),
        flavor: gp_opts.flavor.unwrap_or(GpFlavor::ParsimonyPressure),
        parsimony_coef: gp_opts.parsimony_coef.unwrap_or(0.002),
    };
    run_gp(problem, gp_opts)
}

pub fn predict_holdout(
    family: &ParametricFamily,
    coeffs: &[f64],
    shape: &[f64],
    holdout: &CurveDataset,
) -> f64 {
    let pred = predict_parametric(family, coeffs, shape, &holdout.x);
    weighted_mse(&holdout.y, &pred, holdout)
}

// =============================================================================
// Piecewise models
// =============================================================================

#[derive(Clone, Debug)]
pub struct PiecewiseChromosome {
    /// Internal knot locations in (0,1), sorted.
    pub knot_fracs: Vec<f64>,
    /// Polynomial coeffs per segment (degree fixed at construction).
    pub segment_coeffs: Vec<Vec<f64>>,
    pub x_min: f64,
    pub x_max: f64,
}

impl PiecewiseChromosome {
    pub fn knots(&self) -> Vec<f64> {
        let mut k = vec![self.x_min];
        for &f in &self.knot_fracs {
            k.push(self.x_min + f * (self.x_max - self.x_min));
        }
        k.push(self.x_max);
        k
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let knots = self.knots();
        x.iter()
            .map(|&xv| {
                let seg = knots
                    .windows(2)
                    .position(|w| xv >= w[0] && xv <= w[1])
                    .unwrap_or(0)
                    .min(self.segment_coeffs.len().saturating_sub(1));
                let coeffs = &self.segment_coeffs[seg];
                let local = xv - knots[seg];
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(p, &c)| c * local.powi(p as i32))
                    .sum()
            })
            .collect()
    }
}

pub struct PiecewiseProblem {
    pub data: CurveDataset,
    pub num_segments: usize,
    pub poly_degree: usize,
}

impl PopulationInitializer<PiecewiseChromosome> for PiecewiseProblem {
    fn initial_population(
        &self,
        size: usize,
        rng: &mut dyn RandomSource,
    ) -> Vec<PiecewiseChromosome> {
        let x_min = self.data.x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = self
            .data
            .x
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (0..size)
            .map(|_| {
                let mut fracs: Vec<f64> = (0..self.num_segments.saturating_sub(1))
                    .map(|_| rng.next_float())
                    .collect();
                fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                PiecewiseChromosome {
                    knot_fracs: fracs,
                    segment_coeffs: (0..self.num_segments)
                        .map(|_| {
                            (0..=self.poly_degree)
                                .map(|_| rng.next_float() * 2.0 - 1.0)
                                .collect()
                        })
                        .collect(),
                    x_min,
                    x_max,
                }
            })
            .collect()
    }
}

impl FitnessEvaluator<PiecewiseChromosome> for PiecewiseProblem {
    fn evaluate(&self, individual: &PiecewiseChromosome) -> f64 {
        let pred = individual.predict(&self.data.x);
        weighted_mse(&self.data.y, &pred, &self.data)
    }
}

impl GeneticOperators<PiecewiseChromosome> for PiecewiseProblem {
    fn crossover(
        &self,
        a: &PiecewiseChromosome,
        b: &PiecewiseChromosome,
        rng: &mut dyn RandomSource,
    ) -> PiecewiseChromosome {
        PiecewiseChromosome {
            knot_fracs: if rng.next_float() < 0.5 {
                a.knot_fracs.clone()
            } else {
                b.knot_fracs.clone()
            },
            segment_coeffs: a
                .segment_coeffs
                .iter()
                .zip(&b.segment_coeffs)
                .map(|(sa, sb)| {
                    sa.iter()
                        .zip(sb)
                        .map(|(&x, &y)| if rng.next_float() < 0.5 { x } else { y })
                        .collect()
                })
                .collect(),
            x_min: a.x_min,
            x_max: a.x_max,
        }
    }

    fn mutate(
        &self,
        mut child: PiecewiseChromosome,
        rng: &mut dyn RandomSource,
    ) -> PiecewiseChromosome {
        if rng.next_float() < 0.3 && !child.knot_fracs.is_empty() {
            let i = (rng.next_float() * child.knot_fracs.len() as f64).floor() as usize
                % child.knot_fracs.len();
            child.knot_fracs[i] =
                (child.knot_fracs[i] + rng.next_float() * 0.1 - 0.05).clamp(0.01, 0.99);
            child.knot_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            let s = (rng.next_float() * child.segment_coeffs.len() as f64).floor() as usize
                % child.segment_coeffs.len();
            let c = (rng.next_float() * child.segment_coeffs[s].len() as f64).floor() as usize
                % child.segment_coeffs[s].len();
            child.segment_coeffs[s][c] += rng.next_float() * 0.2 - 0.1;
        }
        child
    }
}

#[derive(Clone, Debug)]
pub struct PiecewiseGaResult {
    pub model: PiecewiseChromosome,
    pub train_mse: f64,
    pub ga: GaResult<PiecewiseChromosome>,
}

pub fn run_piecewise_ga(
    data: CurveDataset,
    num_segments: usize,
    poly_degree: usize,
    ga_opts: GaOptions,
) -> PiecewiseGaResult {
    let problem = PiecewiseProblem {
        data: data.clone(),
        num_segments,
        poly_degree,
    };
    let ga = run_ga(problem, ga_opts, None);
    let mse = weighted_mse(&data.y, &ga.best.predict(&data.x), &data);
    PiecewiseGaResult {
        model: ga.best.clone(),
        train_mse: mse,
        ga,
    }
}

// =============================================================================
// Synthetic benchmarks
// =============================================================================

pub fn synthetic_noisy_sine(n: usize, noise: f64, seed: u32) -> CurveDataset {
    let mut rng = mulberry32(seed);
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1).max(1) as f64 * 2.0 * PI)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xv| xv.sin() + (rng.next_float() * 2.0 - 1.0) * noise)
        .collect();
    CurveDataset::new(x, y)
}

pub fn synthetic_piecewise_step(n: usize, seed: u32) -> CurveDataset {
    let mut rng = mulberry32(seed);
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1).max(1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xv| {
            let base = if xv < 0.35 {
                1.0
            } else if xv < 0.7 {
                2.5 - xv
            } else {
                0.5 + xv
            };
            base + (rng.next_float() * 2.0 - 1.0) * 0.05
        })
        .collect();
    CurveDataset::new(x, y)
}
