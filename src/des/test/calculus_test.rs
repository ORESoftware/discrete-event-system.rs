//! Port of src/des/test/calculus-test.ts
//!
//! Unit tests for the calculus pipeline: the expression engine, quadrature
//! primitives, pure-math ODE solvers, the ODE station network, 1-D heat
//! schemes, station-update order independence, the Thomas tridiagonal solver,
//! and the 2-D Poisson relaxation comparison.
//!
//! The TS `PureTransform` solver classes are the `Transform`-implementing
//! config structs (`TrapezoidRule`, `RK4Integrator`, …); `toFunction` lives in
//! `equation_to_stations`.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::des::general::equation_to_stations::{
        build_field1d, build_ode_system, solve_poisson2d, thomas, to_function, Bc, Field1DFamily,
        Field1DScheme, Field1DSpec, Field2DScheme, OdeScheme, OdeSystemSpec, Poisson2DSpec,
    };
    use crate::des::general::expr::{diff, evaluate, parse, stringify, Env};
    use crate::des::general::ode::{EulerIntegrator, HeunIntegrator, RK4Integrator, IVP};
    use crate::des::general::prng::mulberry32;
    use crate::des::general::quadrature::{
        AdaptiveSimpsonRule, GaussLegendreRule, Integrand1D, SimpsonRule, TrapezoidRule,
    };
    use crate::des::shared::transform::Transform;

    fn approx(a: f64, b: f64, t: f64) -> bool {
        (a - b).abs() <= t
    }

    fn env1(name: &str, v: f64) -> Env {
        let mut e: Env = HashMap::new();
        e.insert(name.to_string(), v);
        e
    }

    // T1 — Expression engine.
    #[test]
    fn expression_engine() {
        let e = parse("2*x + 3");
        assert!(approx(evaluate(&e, &env1("x", 4.0)), 11.0, 1e-12));

        let e2 = parse("sin(x)^2 + cos(x)^2");
        for x in [0.0, 0.5, 1.7, -2.3] {
            assert!(approx(evaluate(&e2, &env1("x", x)), 1.0, 1e-15), "x={x}");
        }

        let d_e = diff(&parse("x^3"), "x");
        let d_e_fn = to_function(&d_e, &["x".to_string()]);
        assert!(approx(d_e_fn(&[2.0]), 12.0, 1e-12));

        let d_ch = diff(&parse("sin(x^2)"), "x");
        let d_ch_fn = to_function(&d_ch, &["x".to_string()]);
        for x in [0.4_f64, 1.1, 2.0] {
            let expected = 2.0 * x * (x * x).cos();
            assert!(approx(d_ch_fn(&[x]), expected, 1e-10), "x={x}");
        }

        let e3 = parse("x^2 + 2*x + 1");
        let s = stringify(&e3);
        let e4 = parse(&s);
        for x in [-1.0, 0.0, 0.5, 3.0] {
            assert!(approx(
                evaluate(&e3, &env1("x", x)),
                evaluate(&e4, &env1("x", x)),
                1e-15
            ));
        }
    }

    // T2 — Quadrature primitives on a smooth integrand.
    #[test]
    fn quadrature_primitives() {
        let f = |x: f64| x * x * x; // ∫_0^1 x³ dx = 1/4

        let trap = TrapezoidRule::new(1000).transform(Integrand1D::new(f, 0.0, 1.0));
        assert!(approx(trap.value, 0.25, 1e-3));

        let simp = SimpsonRule::new(10).transform(Integrand1D::new(f, 0.0, 1.0));
        assert!(approx(simp.value, 0.25, 1e-12));

        let gl = GaussLegendreRule::new(4).transform(Integrand1D::new(f, 0.0, 1.0));
        assert!(approx(gl.value, 0.25, 1e-13));

        let adapt = AdaptiveSimpsonRule::new(1e-12, 50).transform(Integrand1D::new(f, 0.0, 1.0));
        assert!(approx(adapt.value, 0.25, 1e-12));
    }

    // T3 — Pure-math ODE solver convergence orders.
    #[test]
    fn ode_convergence_orders() {
        // y' = -y, y(0)=1; analytic y(1) = e^{-1}.
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let exact = (-1.0_f64).exp();
        let errs = |h: f64| -> (f64, f64, f64) {
            let idx = (1.0 / h).round() as usize;
            let eu = EulerIntegrator::new(h)
                .transform(IVP {
                    f,
                    y0: vec![1.0],
                    t0: 0.0,
                    t1: 1.0,
                })
                .y[idx][0];
            let rk = HeunIntegrator::new(h)
                .transform(IVP {
                    f,
                    y0: vec![1.0],
                    t0: 0.0,
                    t1: 1.0,
                })
                .y[idx][0];
            let rk4 = RK4Integrator::new(h)
                .transform(IVP {
                    f,
                    y0: vec![1.0],
                    t0: 0.0,
                    t1: 1.0,
                })
                .y[idx][0];
            ((eu - exact).abs(), (rk - exact).abs(), (rk4 - exact).abs())
        };
        let e1 = errs(0.1);
        let e2 = errs(0.05);
        assert!(
            (e1.0 / e2.0 - 2.0).abs() < 0.5,
            "euler order ~1: {}",
            e1.0 / e2.0
        );
        assert!(
            (e1.1 / e2.1 - 4.0).abs() < 1.0,
            "rk2 order ~2: {}",
            e1.1 / e2.1
        );
        assert!(
            (e1.2 / e2.2 - 16.0).abs() < 4.0,
            "rk4 order ~4: {}",
            e1.2 / e2.2
        );
    }

    // T4 — ODE station network ≡ pure-math RK4 on the same dt grid.
    #[test]
    fn station_network_matches_pure_rk4() {
        let dt = 0.001;
        let t_end = 2.0 * std::f64::consts::PI;
        let mut station = build_ode_system(&OdeSystemSpec {
            names: vec!["y".to_string(), "v".to_string()],
            rhs: vec!["v".to_string(), "-y".to_string()],
            y0: vec![1.0, 0.0],
            scheme: OdeScheme::Rk4,
            rhs_exprs: None,
        });
        let s = station.run(0.0, t_end, dt);
        let pure = RK4Integrator::new(dt).transform(IVP {
            f: |_t: f64, y: &[f64]| vec![y[1], -y[0]],
            y0: vec![1.0, 0.0],
            t0: 0.0,
            t1: t_end,
        });
        let last = pure.y.last().unwrap();
        assert!((s.final_values[0] - last[0]).abs() < 1e-13);
        assert!((s.final_values[1] - last[1]).abs() < 1e-13);
    }

    // T5 — Field1D heat schemes: FTCS stable at safe dt, BTCS stable at 30× dt.
    #[test]
    fn field1d_heat_stability() {
        let n = 41;
        let alpha = 0.1;
        let dx = 1.0 / (n as f64 - 1.0);
        let dt_safe = 0.4 * dx * dx / alpha;
        let t_end = 0.3;

        let mk = |scheme: Field1DScheme| Field1DSpec {
            n,
            x_lo: 0.0,
            x_hi: 1.0,
            init_expr: "sin(3.14159265358979 * x)".to_string(),
            family: Field1DFamily::Heat,
            alpha_expr: Some(format!("{alpha}")),
            source_expr: None,
            c_expr: None,
            a_expr: None,
            bc_left: Bc::Value(0.0),
            bc_right: Bc::Value(0.0),
            scheme,
        };

        let mut r = build_field1d(&mk(Field1DScheme::Ftcs));
        let o = r.run(0.0, t_end, dt_safe);
        assert!(
            o.final_values
                .iter()
                .all(|&v| v.is_finite() && v.abs() <= 2.0),
            "FTCS bounded at safe dt"
        );

        let mut rb = build_field1d(&mk(Field1DScheme::Btcs));
        let ob = rb.run(0.0, t_end, 30.0 * dt_safe);
        assert!(
            ob.final_values
                .iter()
                .all(|&v| v.is_finite() && v.abs() <= 2.0),
            "BTCS bounded at 30× FTCS-bound dt"
        );
    }

    // T6 — Field1D station-update order-independence (shuffle invariant).
    #[test]
    fn field1d_order_independence() {
        let init = "exp(-50 * (x - 0.5)^2)";
        let make = |seed: u32| -> Vec<f64> {
            let mut r = build_field1d(&Field1DSpec {
                n: 31,
                x_lo: 0.0,
                x_hi: 1.0,
                init_expr: init.to_string(),
                family: Field1DFamily::Heat,
                alpha_expr: Some("0.05".to_string()),
                source_expr: None,
                c_expr: None,
                a_expr: None,
                bc_left: Bc::Value(0.0),
                bc_right: Bc::Value(0.0),
                scheme: Field1DScheme::Ftcs,
            });
            // Override the shuffle RNG with a fresh seed (TS `r.sim.rng = …`).
            r.sim.rng = mulberry32(seed);
            r.run(0.0, 0.2, 0.0005).final_values
        };
        let a = make(1);
        let b = make(99);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() <= 1e-15, "field differs at {i}");
        }
    }

    // T7 — Thomas algorithm vs hand solution.
    #[test]
    fn thomas_tridiagonal() {
        // [2 1 0; 1 2 1; 0 1 2] x = [4; 8; 8] -> x = [1, 2, 3].
        let sub = [0.0, 1.0, 1.0];
        let dia = [2.0, 2.0, 2.0];
        let sup = [1.0, 1.0, 0.0];
        let rhs = [4.0, 8.0, 8.0];
        let x = thomas(&sub, &dia, &sup, &rhs);
        assert!(approx(x[0], 1.0, 1e-12));
        assert!(approx(x[1], 2.0, 1e-12));
        assert!(approx(x[2], 3.0, 1e-12));
    }

    // T8 — Poisson 2-D: SOR < Gauss-Seidel < Jacobi iterations.
    #[test]
    fn poisson2d_relaxation_ordering() {
        let n = 31;
        let tol = 1e-7;
        let rho = "2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)"
            .to_string();
        let mk = |scheme: Field2DScheme, omega: Option<f64>| Poisson2DSpec {
            nx: n,
            ny: n,
            x_lo: 0.0,
            x_hi: 1.0,
            y_lo: 0.0,
            y_hi: 1.0,
            rho_expr: rho.clone(),
            init_expr: None,
            bc_expr: None,
            scheme,
            omega,
            max_iter: Some(50000),
            tol: Some(tol),
        };
        let j = solve_poisson2d(&mk(Field2DScheme::Jacobi, None));
        let g = solve_poisson2d(&mk(Field2DScheme::GaussSeidel, None));
        let s = solve_poisson2d(&mk(Field2DScheme::Sor, Some(1.8)));
        assert!(
            g.iterations < j.iterations,
            "GS {} < Jacobi {}",
            g.iterations,
            j.iterations
        );
        assert!(
            s.iterations < g.iterations,
            "SOR {} < GS {}",
            s.iterations,
            g.iterations
        );
        assert!(
            s.iterations * 5 <= j.iterations,
            "SOR {} vs Jacobi {}",
            s.iterations,
            j.iterations
        );
    }
}
