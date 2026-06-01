//! Batched fitness / residual evaluation shaped for GPU backends.
//!
//! Default path uses dense CPU matrix multiply (`LinAlg::mat_mul`). With the
//! `evolution-gpu` feature, register a custom [`GpuBatchBackend`].

use crate::des::shared::linalg::{LinAlg, Matrix, Vector};

/// Residual vector `y - X β` for a batch of coefficient vectors (columns of `betas`).
pub fn batch_residuals(design: &Matrix, targets: &[f64], betas: &[Vector]) -> Vec<Vector> {
    let evaluator = CpuBatchEvaluator;
    evaluator.residuals_batch(design, targets, betas)
}

/// CPU batched evaluator: one `mat_mul` for all individuals' predictions.
pub struct CpuBatchEvaluator;

impl CpuBatchEvaluator {
    fn betas_as_columns(&self, betas: &[Vector]) -> Matrix {
        if betas.is_empty() {
            return Vec::new();
        }
        let p = betas[0].len();
        let k = betas.len();
        let mut cols = vec![vec![0.0; k]; p];
        for (j, beta) in betas.iter().enumerate() {
            if beta.len() != p {
                panic!(
                    "CpuBatchEvaluator::betas_as_columns: beta[{j}] has length {}, expected {p}",
                    beta.len()
                );
            }
            for i in 0..p {
                cols[i][j] = beta[i];
            }
        }
        cols
    }

    fn columns_as_vectors(&self, predictions: &Matrix) -> Vec<Vector> {
        if predictions.is_empty() {
            return Vec::new();
        }
        let n = LinAlg::rows(predictions);
        let k = LinAlg::cols(predictions);
        let mut out = vec![vec![0.0; n]; k];
        for i in 0..n {
            for j in 0..k {
                out[j][i] = predictions[i][j];
            }
        }
        out
    }

    /// For each β in `betas`, return `y - Xβ`.
    pub fn residuals_batch(
        &self,
        design: &Matrix,
        targets: &[f64],
        betas: &[Vector],
    ) -> Vec<Vector> {
        let predictions = self.batched_predict(design, betas);
        predictions
            .into_iter()
            .map(|pred| targets.iter().zip(pred).map(|(&y, p)| y - p).collect())
            .collect()
    }

    /// Predict all `betas` against one design matrix using a single `X * B`.
    pub fn batched_predict(&self, design: &Matrix, betas: &[Vector]) -> Vec<Vector> {
        if betas.is_empty() {
            return Vec::new();
        }
        let beta_cols = self.betas_as_columns(betas);
        let predictions = LinAlg::mat_mul(design, &beta_cols);
        self.columns_as_vectors(&predictions)
    }

    /// Residuals for a population where each individual owns its own design matrix.
    pub fn residuals_for_designs(
        &self,
        designs: &[Matrix],
        targets: &[f64],
        betas: &[Vector],
    ) -> Vec<Vector> {
        if designs.len() != betas.len() {
            panic!(
                "CpuBatchEvaluator::residuals_for_designs: got {} designs for {} betas",
                designs.len(),
                betas.len()
            );
        }
        designs
            .iter()
            .zip(betas)
            .map(|(design, beta)| {
                let pred = LinAlg::mat_vec(design, beta);
                targets.iter().zip(pred).map(|(&y, p)| y - p).collect()
            })
            .collect()
    }

    /// Weighted sum of squared residuals for each β.
    pub fn batch_mse(
        &self,
        design: &Matrix,
        targets: &[f64],
        weights: &[f64],
        betas: &[Vector],
    ) -> Vec<f64> {
        self.residuals_batch(design, targets, betas)
            .iter()
            .map(|r| {
                r.iter().zip(weights).map(|(e, &w)| w * e * e).sum::<f64>() / r.len().max(1) as f64
            })
            .collect()
    }
}

/// Hook for out-of-tree GPU implementations (CUDA, wgpu compute, etc.).
#[cfg(feature = "evolution-gpu")]
pub trait GpuBatchBackend: Send + Sync {
    fn residuals_batch(&self, design: &Matrix, targets: &[f64], betas: &[Vector]) -> Vec<Vector>;

    fn residuals_for_designs(
        &self,
        designs: &[Matrix],
        targets: &[f64],
        betas: &[Vector],
    ) -> Vec<Vector> {
        CpuBatchEvaluator.residuals_for_designs(designs, targets, betas)
    }
}

#[cfg(feature = "evolution-gpu")]
static GPU_BACKEND: std::sync::OnceLock<Box<dyn GpuBatchBackend>> = std::sync::OnceLock::new();

/// Register a GPU backend (call once at process start). Falls back to CPU if unset.
#[cfg(feature = "evolution-gpu")]
pub fn register_gpu_backend(backend: Box<dyn GpuBatchBackend>) {
    let _ = GPU_BACKEND.set(backend);
}

/// Evaluate residuals using the registered GPU backend, or CPU if none.
pub fn residuals_with_backend(design: &Matrix, targets: &[f64], betas: &[Vector]) -> Vec<Vector> {
    #[cfg(feature = "evolution-gpu")]
    if let Some(gpu) = GPU_BACKEND.get() {
        return gpu.residuals_batch(design, targets, betas);
    }
    CpuBatchEvaluator.residuals_batch(design, targets, betas)
}

/// Evaluate residuals for one-design-per-individual populations.
pub fn residuals_for_designs_with_backend(
    designs: &[Matrix],
    targets: &[f64],
    betas: &[Vector],
) -> Vec<Vector> {
    #[cfg(feature = "evolution-gpu")]
    if let Some(gpu) = GPU_BACKEND.get() {
        return gpu.residuals_for_designs(designs, targets, betas);
    }
    CpuBatchEvaluator.residuals_for_designs(designs, targets, betas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_batch_predicts_columns_with_one_design() {
        let design = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let betas = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let pred = CpuBatchEvaluator.batched_predict(&design, &betas);
        assert_eq!(pred, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    }

    #[test]
    fn cpu_batch_handles_per_individual_designs() {
        let designs = vec![
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
        ];
        let betas = vec![vec![2.0, 4.0], vec![1.5, 2.0]];
        let residuals = CpuBatchEvaluator.residuals_for_designs(&designs, &[2.0, 4.0], &betas);
        assert_eq!(residuals, vec![vec![0.0, 0.0], vec![-1.0, -2.0]]);
    }
}
