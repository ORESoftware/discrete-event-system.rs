//! TypeScript source: `src/des/test/calculus-test.ts`
//! Rust target: `tests/calculus_test.rs`

use discrete_event_system_rs::des::general::ode::{euler, rk2_heun, rk4};
use discrete_event_system_rs::des::general::quadrature::{
    adaptive_simpson, gauss_legendre, simpson, trapezoidal,
};

fn approx(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

#[test]
fn quadrature_primitives_match_closed_form_integral() {
    let f = |x: f64| x * x * x;
    assert!(approx(
        trapezoidal(f, 0.0, 1.0, 1000).unwrap().value,
        0.25,
        1e-3
    ));
    assert!(approx(simpson(f, 0.0, 1.0, 10).unwrap().value, 0.25, 1e-12));
    assert!(approx(
        gauss_legendre(f, 0.0, 1.0, 4).unwrap().value,
        0.25,
        1e-13
    ));
    assert!(approx(
        adaptive_simpson(f, 0.0, 1.0, 1e-12, 40).unwrap().value,
        0.25,
        1e-12
    ));
}

#[test]
fn pure_math_ode_solvers_show_expected_convergence_orders() {
    let rhs = |_t: f64, y: &[f64]| vec![-y[0]];
    let exact = (-1.0_f64).exp();
    let errors = |h: f64| {
        let euler_trace = euler(rhs, &[1.0], 0.0, 1.0, h).unwrap();
        let rk2_trace = rk2_heun(rhs, &[1.0], 0.0, 1.0, h).unwrap();
        let rk4_trace = rk4(rhs, &[1.0], 0.0, 1.0, h).unwrap();
        let index = (1.0 / h).round() as usize;
        (
            (euler_trace.y[index][0] - exact).abs(),
            (rk2_trace.y[index][0] - exact).abs(),
            (rk4_trace.y[index][0] - exact).abs(),
        )
    };

    let e1 = errors(0.1);
    let e2 = errors(0.05);
    assert!((e1.0 / e2.0 - 2.0).abs() < 0.5);
    assert!((e1.1 / e2.1 - 4.0).abs() < 1.0);
    assert!((e1.2 / e2.2 - 16.0).abs() < 4.0);
}

#[test]
fn rk4_matches_simple_harmonic_oscillator_period() {
    let dt = 0.001;
    let period = 2.0 * std::f64::consts::PI;
    let trace = rk4(|_t, y| vec![y[1], -y[0]], &[1.0, 0.0], 0.0, period, dt).unwrap();
    let final_state = trace.y.last().unwrap();
    let energy = final_state[0] * final_state[0] + final_state[1] * final_state[1];
    assert!(approx(energy, 1.0, 1e-10));
    assert!(approx(*trace.t.last().unwrap(), 6.283, 1e-12));
}
