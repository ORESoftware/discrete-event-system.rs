//! Course-aligned operations-research technique modules.
//!
//! This namespace gathers reusable Rust kernels for the UT OR/analytics topics
//! that are broader than one simulation demo: Markov decision processes,
//! decision analysis/engineering, inverse problems, nonlinear optimization, and
//! network optimization. Most modules wrap existing crate solvers where those
//! already exist and add compact missing pieces where they do not.

pub mod decision_analysis;
pub mod decision_engineering;
pub mod inverse_problems;
pub mod mdp;
pub mod network_optimization;
pub mod nonlinear_optimization;
