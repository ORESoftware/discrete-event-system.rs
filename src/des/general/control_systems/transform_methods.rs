//! Control-system lenses for transforms and variational methods.
//!
//! The low-level numerical transform station graphs live in
//! `general::signal_transforms`. This module ties them back to control-systems
//! analysis: transforms choose a basis where a dynamics operator becomes simple,
//! while Lagrange/KKT conditions expose constrained optimality as a linear
//! stationarity system.

#![allow(dead_code)]

use crate::des::general::signal_transforms::ComplexValue;

use super::linear_algebra::{LinAlg, LinearSystem, Matrix, VecOps, Vector};
use super::observability_controllability::StateSpaceModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMethodRole {
    ContinuousDynamics,
    SampledDynamics,
    SpectrumComputation,
    MultiscaleLocalization,
    ScaleInvariantAnalysis,
    ProjectionSensing,
    VariationalOptimization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlMethodDescriptor {
    pub id: &'static str,
    pub role: ControlMethodRole,
    pub domain: &'static str,
    pub simplifies: &'static str,
    pub control_system_use: &'static str,
    pub entrypoint: &'static str,
}

pub fn control_transform_catalog() -> Vec<ControlMethodDescriptor> {
    vec![
        ControlMethodDescriptor {
            id: "laplace-transform",
            role: ControlMethodRole::ContinuousDynamics,
            domain: "continuous time",
            simplifies: "differential operators into multiplication by s",
            control_system_use: "transfer functions, poles, stability, continuous compensators",
            entrypoint: "des::general::signal_transforms::run_laplace_transform",
        },
        ControlMethodDescriptor {
            id: "fourier-transform",
            role: ControlMethodRole::ContinuousDynamics,
            domain: "continuous frequency",
            simplifies: "time shifts and steady-state sinusoids into phase factors",
            control_system_use: "frequency response, Bode/Nyquist reasoning, disturbance spectra",
            entrypoint: "des::general::signal_transforms::run_fourier_transform",
        },
        ControlMethodDescriptor {
            id: "z-transform",
            role: ControlMethodRole::SampledDynamics,
            domain: "discrete time",
            simplifies: "difference equations into multiplication by z",
            control_system_use: "sampled-data plants, digital filters, digital controller design",
            entrypoint: "des::general::signal_transforms::run_z_transform",
        },
        ControlMethodDescriptor {
            id: "dft-transform",
            role: ControlMethodRole::SpectrumComputation,
            domain: "finite discrete signals",
            simplifies: "cyclic shifts into frequency-bin multiplication",
            control_system_use: "sampled spectra, finite-horizon diagnostics, controller telemetry",
            entrypoint: "des::general::signal_transforms::run_dft_transform",
        },
        ControlMethodDescriptor {
            id: "fft-transform",
            role: ControlMethodRole::SpectrumComputation,
            domain: "finite discrete signals",
            simplifies: "the same DFT basis, computed through fast bin structure",
            control_system_use: "real-time spectral monitoring and embedded DSP pipelines",
            entrypoint: "des::general::signal_transforms::run_fft_transform",
        },
        ControlMethodDescriptor {
            id: "wavelet-transform",
            role: ControlMethodRole::MultiscaleLocalization,
            domain: "time-scale",
            simplifies: "localized transients into scale/translation coefficients",
            control_system_use:
                "fault detection, switching transients, nonstationary disturbance analysis",
            entrypoint: "des::general::signal_transforms::run_wavelet_transform",
        },
        ControlMethodDescriptor {
            id: "mellin-transform",
            role: ControlMethodRole::ScaleInvariantAnalysis,
            domain: "positive scale",
            simplifies: "dilations into translations in log coordinates",
            control_system_use:
                "gain/scale invariance, multiplicative uncertainty, log-domain signatures",
            entrypoint: "des::general::signal_transforms::run_mellin_transform",
        },
        ControlMethodDescriptor {
            id: "radon-transform",
            role: ControlMethodRole::ProjectionSensing,
            domain: "spatial projections",
            simplifies: "field structure into line-integral measurements",
            control_system_use: "tomography, inverse sensing, distributed field observability",
            entrypoint: "des::general::signal_transforms::run_radon_transform",
        },
        ControlMethodDescriptor {
            id: "lagrange-kkt",
            role: ControlMethodRole::VariationalOptimization,
            domain: "constrained optimization",
            simplifies: "constraints into multipliers and stationarity equations",
            control_system_use:
                "optimal control, equality-constrained MPC subproblems, shadow prices",
            entrypoint: "des::general::control_systems::transform_methods::solve_lagrange_kkt",
        },
    ]
}

pub fn engineering_core_trio() -> Vec<ControlMethodDescriptor> {
    control_transform_catalog()
        .into_iter()
        .filter(|d| {
            d.id == "laplace-transform" || d.id == "fourier-transform" || d.id == "z-transform"
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct LagrangeKktParams {
    pub q: Matrix,
    pub c: Vector,
    pub a: Matrix,
    pub b: Vector,
    pub tol: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LagrangeKktSolution {
    pub primal: Vector,
    pub multipliers: Vector,
    pub stationarity_residual_norm: f64,
    pub feasibility_residual_norm: f64,
}

pub fn solve_lagrange_kkt(params: LagrangeKktParams) -> LagrangeKktSolution {
    let n = params.c.len();
    let m = params.b.len();
    if LinAlg::rows(&params.q) != n || LinAlg::cols(&params.q) != n {
        panic!("solve_lagrange_kkt: Q must be n x n");
    }
    if LinAlg::rows(&params.a) != m || LinAlg::cols(&params.a) != n {
        panic!("solve_lagrange_kkt: A must be m x n");
    }

    let mut kkt = LinAlg::zeros(n + m, n + m);
    for i in 0..n {
        for j in 0..n {
            kkt[i][j] = params.q[i][j];
        }
        for j in 0..m {
            kkt[i][n + j] = params.a[j][i];
        }
    }
    for i in 0..m {
        for j in 0..n {
            kkt[n + i][j] = params.a[i][j];
        }
    }

    let mut rhs = vec![0.0; n + m];
    for i in 0..n {
        rhs[i] = -params.c[i];
    }
    for i in 0..m {
        rhs[n + i] = params.b[i];
    }

    let tol = params.tol.unwrap_or(1e-12);
    let solution = LinearSystem::new(&kkt, &rhs, tol).solve();
    let primal = solution[..n].to_vec();
    let multipliers = solution[n..].to_vec();

    let qx = LinAlg::mat_vec(&params.q, &primal);
    let at = LinAlg::transpose(&params.a);
    let atl = LinAlg::mat_vec(&at, &multipliers);
    let stationarity = VecOps::add(&VecOps::add(&qx, &params.c), &atl);
    let feasibility = VecOps::sub(&LinAlg::mat_vec(&params.a, &primal), &params.b);

    LagrangeKktSolution {
        primal,
        multipliers,
        stationarity_residual_norm: VecOps::norm2(&stationarity),
        feasibility_residual_norm: VecOps::norm2(&feasibility),
    }
}

#[derive(Clone, Debug)]
pub struct StateSpaceFrequencyResponseParams {
    pub model: StateSpaceModel,
    pub points: Vec<ComplexValue>,
    pub labels: Option<Vec<String>>,
    pub tol: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyResponsePoint {
    pub label: String,
    pub s: ComplexValue,
    pub transfer: Vec<Vec<ComplexValue>>,
    pub max_magnitude: f64,
}

pub fn run_state_space_frequency_response(
    params: StateSpaceFrequencyResponseParams,
) -> Vec<FrequencyResponsePoint> {
    if params.points.is_empty() {
        panic!("run_state_space_frequency_response: at least one complex point is required");
    }
    if let Some(labels) = &params.labels {
        if labels.len() != params.points.len() {
            panic!("run_state_space_frequency_response: labels length must match points length");
        }
    }
    let tol = params.tol.unwrap_or(1e-12);
    params
        .points
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let label = params
                .labels
                .as_ref()
                .and_then(|labels| labels.get(i).cloned())
                .unwrap_or_else(|| format!("s={}+{}i", s.re, s.im));
            let transfer = transfer_matrix_at(&params.model, s, tol);
            let max_magnitude = transfer
                .iter()
                .flat_map(|row| row.iter())
                .map(|z| z.re.hypot(z.im))
                .fold(0.0_f64, f64::max);
            FrequencyResponsePoint {
                label,
                s,
                transfer,
                max_magnitude,
            }
        })
        .collect()
}

pub fn run_fourier_frequency_response(
    model: StateSpaceModel,
    omega_values: Vec<f64>,
) -> Vec<FrequencyResponsePoint> {
    let points = omega_values
        .iter()
        .map(|&omega| ComplexValue { re: 0.0, im: omega })
        .collect();
    let labels = omega_values
        .iter()
        .map(|omega| format!("omega={omega}"))
        .collect();
    run_state_space_frequency_response(StateSpaceFrequencyResponseParams {
        model,
        points,
        labels: Some(labels),
        tol: None,
    })
}

fn transfer_matrix_at(
    model: &StateSpaceModel,
    s: ComplexValue,
    tol: f64,
) -> Vec<Vec<ComplexValue>> {
    let n = model.state_dim();
    let inputs = model.input_dim();
    let outputs = model.output_dim();
    let mut transfer = vec![vec![ComplexValue::default(); inputs]; outputs];

    for input in 0..inputs {
        let b_re: Vec<f64> = (0..n).map(|row| model.b[row][input]).collect();
        let b_im = vec![0.0; n];
        let (x_re, x_im) = solve_state_resolvent(model, s, &b_re, &b_im, tol);
        for output in 0..outputs {
            let mut re = model.d[output][input];
            let mut im = 0.0;
            for state in 0..n {
                re += model.c[output][state] * x_re[state];
                im += model.c[output][state] * x_im[state];
            }
            transfer[output][input] = ComplexValue { re, im };
        }
    }

    transfer
}

fn solve_state_resolvent(
    model: &StateSpaceModel,
    s: ComplexValue,
    rhs_re: &[f64],
    rhs_im: &[f64],
    tol: f64,
) -> (Vector, Vector) {
    let n = model.state_dim();
    let mut block = LinAlg::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let m_re = if i == j { s.re } else { 0.0 } - model.a[i][j];
            let m_im = if i == j { s.im } else { 0.0 };
            block[i][j] = m_re;
            block[i][n + j] = -m_im;
            block[n + i][j] = m_im;
            block[n + i][n + j] = m_re;
        }
    }
    let mut rhs = vec![0.0; 2 * n];
    for i in 0..n {
        rhs[i] = rhs_re[i];
        rhs[n + i] = rhs_im[i];
    }
    let solved = LinearSystem::new(&block, &rhs, tol).solve();
    (solved[..n].to_vec(), solved[n..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::control_systems::observability_controllability::StateSpaceSpec;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn catalog_contains_core_and_lagrange_methods() {
        let catalog = control_transform_catalog();
        assert!(catalog.iter().any(|d| d.id == "laplace-transform"));
        assert!(catalog.iter().any(|d| d.id == "z-transform"));
        assert!(catalog.iter().any(|d| d.id == "lagrange-kkt"));
        assert_eq!(engineering_core_trio().len(), 3);
    }

    #[test]
    fn lagrange_kkt_solves_equality_constrained_quadratic() {
        let solution = solve_lagrange_kkt(LagrangeKktParams {
            q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            c: vec![0.0, 0.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            tol: None,
        });
        assert!(close(solution.primal[0], 0.5));
        assert!(close(solution.primal[1], 0.5));
        assert!(close(solution.multipliers[0], -1.0));
        assert!(solution.stationarity_residual_norm < 1e-12);
        assert!(solution.feasibility_residual_norm < 1e-12);
    }

    #[test]
    fn state_space_frequency_response_matches_first_order_transfer() {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![-1.0]],
            b: vec![vec![1.0]],
            c: vec![vec![1.0]],
            d: None,
        });
        let response = run_fourier_frequency_response(model, vec![0.0, 1.0]);
        assert!(close(response[0].transfer[0][0].re, 1.0));
        assert!(close(response[0].transfer[0][0].im, 0.0));
        assert!(close(response[1].transfer[0][0].re, 0.5));
        assert!(close(response[1].transfer[0][0].im, -0.5));
        assert!(close(response[1].max_magnitude, 0.5_f64.sqrt()));
    }
}
