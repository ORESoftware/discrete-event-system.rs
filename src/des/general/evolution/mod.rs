//! Genetic algorithms, genetic programming, curve fitting, and bio-design
//! optimizers built on the shared expression engine and dense linear algebra.
//!
//! See [`README.md`](README.md) in this directory for an overview.

pub mod bio_design;
pub mod curve_fitting;
pub mod ga_core;
pub mod genetic_programming;
pub mod gpu_batch;

pub use bio_design::{
    hp_energy, run_hp_protein_ga, run_ligand_design_ga, HpDirection, HpGaResult, HpGenome,
    HpMonomer, LigandGaResult, LigandGenome, LigandPalette,
};
pub use curve_fitting::{
    hybrid_refine, predict_holdout, run_curve_fit_ga, run_curve_fit_gp, run_piecewise_ga,
    synthetic_noisy_sine, synthetic_piecewise_step, CurveConstraints, CurveDataset,
    CurveFitGaResult, FitMetric, ParametricChromosome, ParametricFamily, PiecewiseChromosome,
    PiecewiseGaResult,
};
pub use ga_core::{
    run_ga, run_ga_as_des, EvolutionGaDesResult, EvolutionGaStation, FitnessEvaluator, GaFlavor,
    GaOptions, GaProblem, GaResult, GeneticOperators, PopulationInitializer,
};
pub use genetic_programming::{run_gp, GpFlavor, GpOptions, GpResult, GpTreeConfig};
#[cfg(feature = "evolution-gpu")]
pub use gpu_batch::GpuBatchBackend;
pub use gpu_batch::{
    batch_residuals, residuals_for_designs_with_backend, residuals_with_backend, CpuBatchEvaluator,
};
