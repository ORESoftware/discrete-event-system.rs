//! Canonical hybrid models built from the standard block library, plus the
//! correctness tests that prove the executive's three defining behaviours:
//! continuous integration, multirate discrete sample times (with ZOH), and
//! zero-crossing state events.

use super::blocks::{BouncingBall, Constant, DiscretePi, Integrator, StateSpace, Sum};
use super::diagram::{Compiled, Diagram, HybridError};
use super::executive::SimOptions;

/// Multirate closed loop: a continuous first-order plant `ẋ = −x + u`, `y = x`
/// regulated to a setpoint of 1 by a discrete-time PI controller running every
/// 0.1 s, while the plant integrates continuously. Returns the compiled diagram
/// and matching options.
pub fn closed_loop() -> Result<(Compiled, SimOptions), HybridError> {
    let mut d = Diagram::new();
    let reference = d.add(Box::new(Constant::scalar("reference", 1.0)));
    let error = d.add(Box::new(Sum::new("error", 1, vec![1.0, -1.0])));
    let controller = d.add(Box::new(DiscretePi::new("pi", 2.0, 1.5, 0.1)));
    let plant = d.add(Box::new(StateSpace::new(
        "plant",
        vec![vec![-1.0]], // A
        vec![vec![1.0]],  // B
        vec![vec![1.0]],  // C
        vec![vec![0.0]],  // D (no feedthrough)
    )));

    d.connect((reference, 0), (error, 0))?; // r -> error(+)
    d.connect((plant, 0), (error, 1))?; // y -> error(-)
    d.connect((error, 0), (controller, 0))?; // e -> controller
    d.connect((controller, 0), (plant, 0))?; // u -> plant

    Ok((d.build()?, SimOptions::new(6.0, 0.01)))
}

/// Bouncing ball: a purely continuous plant with a zero-crossing at the floor
/// and an energy-losing reflection event.
pub fn bouncing_ball() -> Result<(Compiled, SimOptions), HybridError> {
    let mut d = Diagram::new();
    d.add(Box::new(BouncingBall::new("ball", 1.0, 0.0, 0.8)));
    Ok((
        d.build()?,
        SimOptions {
            t_end: 4.0,
            max_step: 0.02,
            zc_tol: 1e-10,
        },
    ))
}

/// Constant -> Integrator: `ẋ = 1`, so `x(t) = t` (smallest continuous test).
pub fn integrator_ramp() -> Result<(Compiled, SimOptions), HybridError> {
    let mut d = Diagram::new();
    let one = d.add(Box::new(Constant::scalar("one", 1.0)));
    let integ = d.add(Box::new(Integrator::new("x", vec![0.0])));
    d.connect((one, 0), (integ, 0))?;
    Ok((d.build()?, SimOptions::new(2.0, 0.01)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::hybrid::blocks::{Counter, Gain};
    use crate::des::hybrid::executive::simulate;

    #[test]
    fn integrator_produces_a_ramp() {
        let (compiled, opts) = integrator_ramp().unwrap();
        let trace = simulate(&compiled, &opts);
        let (_t, x) = trace.series("x.p0").unwrap();
        let last = *x.last().unwrap();
        assert!((last - 2.0).abs() < 1e-6, "x(2) = {last}, expected 2.0");
    }

    #[test]
    fn first_order_plant_reaches_steady_state() {
        // Constant 3 -> plant ẋ=-x+u, y=x ; steady state y -> 3.
        let mut d = Diagram::new();
        let u = d.add(Box::new(Constant::scalar("u", 3.0)));
        let plant = d.add(Box::new(StateSpace::new(
            "plant",
            vec![vec![-1.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        )));
        d.connect((u, 0), (plant, 0)).unwrap();
        let compiled = d.build().unwrap();
        let trace = simulate(&compiled, &SimOptions::new(12.0, 0.01));
        let (_t, y) = trace.series("plant.p0").unwrap();
        let last = *y.last().unwrap();
        assert!(
            (last - 3.0).abs() < 0.02,
            "steady-state y = {last}, expected ~3.0"
        );
    }

    #[test]
    fn closed_loop_regulates_to_setpoint() {
        let (compiled, opts) = closed_loop().unwrap();
        let trace = simulate(&compiled, &opts);
        let (_t, y) = trace.series("plant.p0").unwrap();
        let last = *y.last().unwrap();
        assert!(
            (last - 1.0).abs() < 0.02,
            "closed-loop output {last}, expected ~1.0"
        );
    }

    #[test]
    fn controller_output_is_piecewise_constant_at_the_sample_rate() {
        // The discrete PI runs at 0.1 s; its command must only change at
        // multiples of 0.1 (zero-order hold / multirate behaviour).
        let (compiled, opts) = closed_loop().unwrap();
        let trace = simulate(&compiled, &opts);
        let (t, u) = trace.series("pi.p0").unwrap();
        let mut prev = u[0];
        for k in 1..u.len() {
            if (u[k] - prev).abs() > 1e-12 {
                let ticks = t[k] / 0.1;
                assert!(
                    (ticks - ticks.round()).abs() < 1e-6,
                    "controller changed at t={} (not a 0.1 multiple)",
                    t[k]
                );
                prev = u[k];
            }
        }
    }

    #[test]
    fn bouncing_ball_loses_energy_and_never_penetrates_floor() {
        let (compiled, opts) = bouncing_ball().unwrap();
        let trace = simulate(&compiled, &opts);
        assert!(
            trace.events >= 2,
            "expected multiple bounces, got {}",
            trace.events
        );
        let (t, h) = trace.series("ball.p0[0]").unwrap(); // height channel
        let min_h = h.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_h > -1e-2,
            "ball penetrated the floor: min height {min_h}"
        );
        // Energy loss: the first rebound peak (e=0.8 => 0.64·h0) is well below h0.
        let peak_after_first_bounce = t
            .iter()
            .zip(&h)
            .filter(|(&ti, _)| ti > 0.6)
            .map(|(_, &hi)| hi)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (0.4..0.7).contains(&peak_after_first_bounce),
            "post-bounce peak {peak_after_first_bounce}, expected ~0.64"
        );
    }

    #[test]
    fn pure_discrete_counter_advances_each_hit() {
        let mut d = Diagram::new();
        d.add(Box::new(Counter::new("count", 1.0)));
        let compiled = d.build().unwrap();
        let trace = simulate(&compiled, &SimOptions::new(5.0, 0.5));
        let (_t, c) = trace.series("count.p0").unwrap();
        let last = *c.last().unwrap();
        // Hits at t = 0,1,2,3,4 within [0,5) => count reaches 5.
        assert!(last >= 4.0, "counter only reached {last}");
        // Monotonic non-decreasing.
        for w in c.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn algebraic_loop_is_rejected() {
        // Two feedthrough gains wired into a cycle => algebraic loop.
        let mut d = Diagram::new();
        let g1 = d.add(Box::new(Gain::new("g1", 1, 1.0)));
        let g2 = d.add(Box::new(Gain::new("g2", 1, 1.0)));
        d.connect((g1, 0), (g2, 0)).unwrap();
        d.connect((g2, 0), (g1, 0)).unwrap();
        match d.build() {
            Err(HybridError::AlgebraicLoop(blocks)) => {
                assert!(blocks.contains(&"g1".to_string()) || blocks.contains(&"g2".to_string()));
            }
            Err(other) => panic!("expected AlgebraicLoop, got {other:?}"),
            Ok(_) => panic!("expected AlgebraicLoop, but build() succeeded"),
        }
    }

    #[test]
    fn integrator_in_loop_is_not_an_algebraic_loop() {
        // Same topology but the feedback path goes through an integrator
        // (non-feedthrough), so it must build fine.
        let mut d = Diagram::new();
        let sum = d.add(Box::new(Sum::new("sum", 1, vec![1.0, -1.0])));
        let integ = d.add(Box::new(Integrator::new("x", vec![0.0])));
        let r = d.add(Box::new(Constant::scalar("r", 1.0)));
        d.connect((r, 0), (sum, 0)).unwrap();
        d.connect((integ, 0), (sum, 1)).unwrap();
        d.connect((sum, 0), (integ, 0)).unwrap();
        assert!(d.build().is_ok());
    }

    #[test]
    fn width_mismatch_is_rejected() {
        let mut d = Diagram::new();
        let src = d.add(Box::new(Constant::new("v2", vec![1.0, 2.0]))); // width 2
        let g = d.add(Box::new(Gain::new("g", 1, 1.0))); // width 1
        match d.connect((src, 0), (g, 0)) {
            Err(HybridError::WidthMismatch {
                src_width,
                dst_width,
                ..
            }) => {
                assert_eq!((src_width, dst_width), (2, 1));
            }
            other => panic!("expected WidthMismatch, got {other:?}"),
        }
    }
}
