//! Tests for the analytic calculus-of-variations model catalog.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use crate::des::general::calculus_of_variations::{
        built_in_variational_models, solve_brachistochrone, solve_minimal_surface,
        solve_shortest_curve, BoundaryPoint, BrachistochroneParams, MinimalSurfaceParams,
        ShortestCurveParams, VariationalProblemKind,
    };

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn constant(
        model: &crate::des::general::calculus_of_variations::VariationalSolutionModel,
        name: &str,
    ) -> f64 {
        model
            .solution
            .constants
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("missing solution constant {name}"))
    }

    #[test]
    fn built_in_catalog_has_three_independent_models() {
        let models = built_in_variational_models();
        assert_eq!(models.len(), 3);

        let ids = models
            .iter()
            .map(|m| m.problem.id.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains("shortest-curve"));
        assert!(ids.contains("brachistochrone"));
        assert!(ids.contains("minimal-surface-catenoid"));

        for model in models {
            assert!(!model.samples.is_empty());
            assert_eq!(model.topology.stations.len(), 3);
            assert!(model.diagnostics.functional_value.is_finite());
            assert!(model.diagnostics.boundary_error <= 1e-10);
        }
    }

    #[test]
    fn shortest_curve_solution_is_line_with_arc_length() {
        let model = solve_shortest_curve(ShortestCurveParams {
            start: BoundaryPoint::new(-1.0, 2.0),
            end: BoundaryPoint::new(3.0, 5.0),
            samples: 9,
        });

        assert_eq!(model.problem.kind, VariationalProblemKind::ShortestCurve);
        assert!(approx(
            model.diagnostics.functional_value,
            (4.0_f64 * 4.0 + 3.0 * 3.0).sqrt(),
            1e-12
        ));
        assert!(model.diagnostics.first_integral_residual_l2 <= 1e-14);

        for sample in &model.samples {
            let expected = 0.75 * sample.x + 2.75;
            assert!(approx(sample.y, expected, 1e-12));
            assert!(approx(sample.dy_dx.unwrap(), 0.75, 1e-12));
        }
    }

    #[test]
    fn brachistochrone_hits_endpoint_and_preserves_first_integral() {
        let params = BrachistochroneParams {
            horizontal: 1.0,
            drop: 0.75,
            gravity: 9.81,
            samples: 81,
        };
        let model = solve_brachistochrone(params.clone());
        let radius = constant(&model, "radius");
        let theta_end = constant(&model, "theta_end");
        let expected_time = theta_end * (radius / params.gravity).sqrt();
        let last = model.samples.last().unwrap();

        assert_eq!(model.problem.kind, VariationalProblemKind::Brachistochrone);
        assert!(theta_end > 0.0 && theta_end < 2.0 * PI);
        assert!(approx(last.x, params.horizontal, 1e-12));
        assert!(approx(last.y, -params.drop, 1e-12));
        assert!(approx(
            model.diagnostics.functional_value,
            expected_time,
            1e-12
        ));
        assert!(model.samples.first().unwrap().dy_dx.is_none());
        assert!(model.diagnostics.first_integral_residual_l2 <= 1e-12);
    }

    #[test]
    fn minimal_surface_solution_is_catenoid_with_exact_area() {
        let params = MinimalSurfaceParams {
            half_span: 0.6,
            neck_radius: 0.4,
            samples: 101,
        };
        let model = solve_minimal_surface(params.clone());
        let a = params.neck_radius;
        let l = params.half_span;
        let ring_radius = a * (l / a).cosh();
        let expected_area = 2.0 * PI * a * l + PI * a * a * (2.0 * l / a).sinh();
        let first = model.samples.first().unwrap();
        let mid = &model.samples[model.samples.len() / 2];
        let last = model.samples.last().unwrap();

        assert_eq!(
            model.problem.kind,
            VariationalProblemKind::MinimalSurfaceOfRevolution
        );
        assert!(approx(first.y, ring_radius, 1e-12));
        assert!(approx(last.y, ring_radius, 1e-12));
        assert!(approx(mid.y, a, 1e-12));
        assert!(approx(
            model.diagnostics.functional_value,
            expected_area,
            1e-12
        ));
        assert!(model.diagnostics.first_integral_residual_l2 <= 1e-12);
    }
}
