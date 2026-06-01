//! Linear Lagrange mechanics helpers for control-system models.
//!
//! This covers the common small-mechanical-system path:
//!
//! `M qdd + C qd + K q = B u + bias`
//!
//! and converts it into the first-order state-space form used by controllers:
//!
//! `xdot = A x + B u + bias_x`, with `x = [q, qd]`.

use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::shared::linalg::{LinAlg, Matrix, MatrixInverse, Vector};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

#[derive(Clone, Debug)]
pub struct LagrangeSecondOrderSystem {
    pub mass: Matrix,
    pub damping: Matrix,
    pub stiffness: Matrix,
    pub input: Matrix,
    pub force_bias: Option<Vector>,
}

#[derive(Clone, Debug)]
pub struct LagrangeStateSpace {
    pub a: Matrix,
    pub b: Matrix,
    pub bias: Vector,
}

fn validate_square(name: &str, m: &Matrix, n: usize) {
    require(Preconditions::length_eq("lagrange", name, m, n));
    require(Preconditions::rectangular_matrix("lagrange", name, m));
    if n > 0 {
        require(Preconditions::length_eq(
            "lagrange",
            &format!("{name}[0]"),
            &m[0],
            n,
        ));
    }
    for (i, row) in m.iter().enumerate() {
        require(Preconditions::all_finite(
            "lagrange",
            &format!("{name}[{i}]"),
            row,
        ));
    }
}

fn validate_matrix(name: &str, m: &Matrix, rows: usize) {
    require(Preconditions::length_eq("lagrange", name, m, rows));
    require(Preconditions::rectangular_matrix("lagrange", name, m));
    for (i, row) in m.iter().enumerate() {
        require(Preconditions::all_finite(
            "lagrange",
            &format!("{name}[{i}]"),
            row,
        ));
    }
}

fn matrix_neg(m: &Matrix) -> Matrix {
    LinAlg::scale(m, -1.0)
}

pub fn lagrange_to_state_space(system: &LagrangeSecondOrderSystem) -> LagrangeStateSpace {
    let n = system.mass.len();
    require(Preconditions::integer_in_range(
        "lagrange",
        "degrees of freedom",
        n as f64,
        1.0,
        10_000.0,
    ));
    validate_square("mass", &system.mass, n);
    validate_square("damping", &system.damping, n);
    validate_square("stiffness", &system.stiffness, n);
    validate_matrix("input", &system.input, n);
    let m_inputs = LinAlg::cols(&system.input);
    let force_bias = system.force_bias.clone().unwrap_or_else(|| vec![0.0; n]);
    require(Preconditions::length_eq(
        "lagrange",
        "forceBias",
        &force_bias,
        n,
    ));
    require(Preconditions::all_finite(
        "lagrange",
        "forceBias",
        &force_bias,
    ));

    let mut mass_inverse = MatrixInverse::new(&system.mass, None);
    let m_inv = mass_inverse.inverse();
    let lower_q = LinAlg::mat_mul(&matrix_neg(&m_inv), &system.stiffness);
    let lower_qd = LinAlg::mat_mul(&matrix_neg(&m_inv), &system.damping);
    let lower_u = LinAlg::mat_mul(&m_inv, &system.input);
    let lower_bias = LinAlg::mat_vec(&m_inv, &force_bias);

    let mut a = LinAlg::zeros(2 * n, 2 * n);
    for i in 0..n {
        a[i][n + i] = 1.0;
        for j in 0..n {
            a[n + i][j] = lower_q[i][j];
            a[n + i][n + j] = lower_qd[i][j];
        }
    }

    let mut b = LinAlg::zeros(2 * n, m_inputs);
    for i in 0..n {
        for j in 0..m_inputs {
            b[n + i][j] = lower_u[i][j];
        }
    }

    let mut bias = vec![0.0; 2 * n];
    for i in 0..n {
        bias[n + i] = lower_bias[i];
    }

    LagrangeStateSpace { a, b, bias }
}

pub fn generalized_acceleration(
    system: &LagrangeSecondOrderSystem,
    q: &[f64],
    qd: &[f64],
    u: &[f64],
) -> Vector {
    let ss = lagrange_to_state_space(system);
    let n = system.mass.len();
    require(Preconditions::length_eq("lagrange", "q", q, n));
    require(Preconditions::length_eq("lagrange", "qd", qd, n));
    require(Preconditions::all_finite("lagrange", "q", q));
    require(Preconditions::all_finite("lagrange", "qd", qd));
    require(Preconditions::length_eq(
        "lagrange",
        "u",
        u,
        LinAlg::cols(&system.input),
    ));
    require(Preconditions::all_finite("lagrange", "u", u));
    let mut x = q.to_vec();
    x.extend_from_slice(qd);
    let ax = LinAlg::mat_vec(&ss.a, &x);
    let bu = LinAlg::mat_vec(&ss.b, u);
    (0..n)
        .map(|i| ax[n + i] + bu[n + i] + ss.bias[n + i])
        .collect()
}
