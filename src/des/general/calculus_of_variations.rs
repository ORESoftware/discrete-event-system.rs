//! Analytic calculus-of-variations problem and solution models.
//!
//! This module keeps the model layer explicit: each example exposes the
//! variational functional, the Euler-Lagrange/first-integral model, boundary
//! data, sampled solution geometry, and a compact residual diagnostic.
//!
//! The three built-in examples are intentionally independent:
//!   * fixed-endpoint Euclidean shortest curve,
//!   * brachistochrone time of descent,
//!   * catenoid/minimal surface of revolution.

#![allow(dead_code)]

use std::f64::consts::PI;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};

const EPS: f64 = 1e-12;

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

fn require_samples(model: &str, samples: usize, min: usize) {
    require(Preconditions::integer_in_range(
        model,
        "samples",
        samples as f64,
        min as f64,
        1_000_000.0,
    ));
}

fn finite_pair(model: &str, name: &str, p: BoundaryPoint) {
    require(Preconditions::finite(model, &format!("{name}.x"), p.x));
    require(Preconditions::finite(model, &format!("{name}.y"), p.y));
}

fn graph_for(problem_id: &str) -> StationGraphSummary {
    let source = StationOrId::from(format!("{problem_id}-problem-source"));
    let solver = StationOrId::from(format!("{problem_id}-euler-lagrange-solver"));
    let sampler = StationOrId::from(format!("{problem_id}-solution-sampler"));
    station_graph(
        &[source.clone(), solver.clone(), sampler.clone()],
        &[
            "variational-problem".to_string(),
            "stationary-solution".to_string(),
            "solution-samples".to_string(),
        ],
        &[
            channel_edge(&source, "problem", &solver, Some("problem")),
            channel_edge(&solver, "solution", &sampler, Some("solution")),
        ],
    )
}

fn trapezoid_l2(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean_sq = values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64;
    mean_sq.sqrt()
}

/// Problem family tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariationalProblemKind {
    ShortestCurve,
    Brachistochrone,
    MinimalSurfaceOfRevolution,
}

/// Whether a functional is minimized or maximized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremum {
    Minimize,
    Maximize,
}

/// A fixed point in the independent/dependent variable plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryPoint {
    pub x: f64,
    pub y: f64,
}

impl BoundaryPoint {
    pub fn new(x: f64, y: f64) -> Self {
        BoundaryPoint { x, y }
    }
}

/// The functional and boundary data that define a variational problem.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionalSpec {
    pub objective: Extremum,
    pub integrand: String,
    pub independent_variable: String,
    pub dependent_variable: String,
    pub derivative_variable: String,
    pub domain: (f64, f64),
    pub boundary: (BoundaryPoint, BoundaryPoint),
}

/// The Euler-Lagrange model used to solve or verify the problem.
#[derive(Clone, Debug, PartialEq)]
pub struct EulerLagrangeModel {
    pub equation: String,
    pub first_integral: Option<String>,
    pub natural_boundary_conditions: Vec<String>,
}

/// Human-readable problem model.
#[derive(Clone, Debug, PartialEq)]
pub struct VariationalProblem {
    pub id: String,
    pub name: String,
    pub kind: VariationalProblemKind,
    pub description: String,
    pub functional: FunctionalSpec,
    pub euler_lagrange: EulerLagrangeModel,
}

/// Closed-form solution representation.
#[derive(Clone, Debug, PartialEq)]
pub struct SolutionFormula {
    pub expression: String,
    pub parameterization: Option<Vec<String>>,
    pub constants: Vec<(String, f64)>,
}

/// One sampled point on a solution curve.
#[derive(Clone, Debug, PartialEq)]
pub struct SolutionSample {
    /// Optional curve parameter (`theta` for the brachistochrone, `x` otherwise).
    pub parameter: Option<f64>,
    pub x: f64,
    pub y: f64,
    /// `None` allows vertical tangents or integrable endpoint singularities.
    pub dy_dx: Option<f64>,
    pub integrand: Option<f64>,
}

/// Numerical diagnostics attached to an analytic solution model.
#[derive(Clone, Debug, PartialEq)]
pub struct SolutionDiagnostics {
    pub functional_value: f64,
    pub boundary_error: f64,
    pub first_integral_residual_l2: f64,
}

/// Complete problem + solution + sampling model.
#[derive(Clone, Debug, PartialEq)]
pub struct VariationalSolutionModel {
    pub problem: VariationalProblem,
    pub solution: SolutionFormula,
    pub samples: Vec<SolutionSample>,
    pub diagnostics: SolutionDiagnostics,
    pub topology: StationGraphSummary,
}

/// Parameters for the fixed-endpoint shortest-curve model.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortestCurveParams {
    pub start: BoundaryPoint,
    pub end: BoundaryPoint,
    pub samples: usize,
}

impl Default for ShortestCurveParams {
    fn default() -> Self {
        ShortestCurveParams {
            start: BoundaryPoint::new(0.0, 0.0),
            end: BoundaryPoint::new(1.0, 1.0),
            samples: 32,
        }
    }
}

/// Solve the fixed-endpoint Euclidean shortest-curve problem.
pub fn solve_shortest_curve(params: ShortestCurveParams) -> VariationalSolutionModel {
    let model = "calculus-of-variations.shortest-curve";
    finite_pair(model, "start", params.start);
    finite_pair(model, "end", params.end);
    require(Preconditions::finite(
        model,
        "span",
        params.end.x - params.start.x,
    ));
    if (params.end.x - params.start.x).abs() <= EPS {
        panic!("{model}: end.x - start.x must be non-zero");
    }
    require_samples(model, params.samples, 2);

    let dx = params.end.x - params.start.x;
    let dy = params.end.y - params.start.y;
    let slope = dy / dx;
    let intercept = params.start.y - slope * params.start.x;
    let integrand = (1.0 + slope * slope).sqrt();
    let length = dx.abs() * integrand;
    let samples = (0..params.samples)
        .map(|i| {
            let u = i as f64 / (params.samples - 1) as f64;
            let x = params.start.x + u * dx;
            SolutionSample {
                parameter: Some(x),
                x,
                y: slope * x + intercept,
                dy_dx: Some(slope),
                integrand: Some(integrand),
            }
        })
        .collect::<Vec<_>>();

    let first_integral = slope / integrand;
    let residuals = samples
        .iter()
        .filter_map(|s| s.dy_dx)
        .map(|m| m / (1.0 + m * m).sqrt() - first_integral)
        .collect::<Vec<_>>();

    VariationalSolutionModel {
        problem: VariationalProblem {
            id: "shortest-curve".to_string(),
            name: "Fixed-endpoint shortest curve".to_string(),
            kind: VariationalProblemKind::ShortestCurve,
            description: "Minimize Euclidean arc length between two fixed endpoints.".to_string(),
            functional: FunctionalSpec {
                objective: Extremum::Minimize,
                integrand: "sqrt(1 + y_prime^2)".to_string(),
                independent_variable: "x".to_string(),
                dependent_variable: "y".to_string(),
                derivative_variable: "y_prime".to_string(),
                domain: (params.start.x, params.end.x),
                boundary: (params.start, params.end),
            },
            euler_lagrange: EulerLagrangeModel {
                equation: "d/dx(y_prime / sqrt(1 + y_prime^2)) = 0".to_string(),
                first_integral: Some("y_prime / sqrt(1 + y_prime^2) = constant".to_string()),
                natural_boundary_conditions: Vec::new(),
            },
        },
        solution: SolutionFormula {
            expression: format!("y(x) = {slope} * x + {intercept}"),
            parameterization: None,
            constants: vec![
                ("slope".to_string(), slope),
                ("intercept".to_string(), intercept),
                ("first_integral".to_string(), first_integral),
            ],
        },
        samples,
        diagnostics: SolutionDiagnostics {
            functional_value: length,
            boundary_error: 0.0,
            first_integral_residual_l2: trapezoid_l2(&residuals),
        },
        topology: graph_for("shortest-curve"),
    }
}

/// Parameters for a brachistochrone from `(0, 0)` to `(horizontal, -drop)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BrachistochroneParams {
    pub horizontal: f64,
    pub drop: f64,
    pub gravity: f64,
    pub samples: usize,
}

impl Default for BrachistochroneParams {
    fn default() -> Self {
        BrachistochroneParams {
            horizontal: 1.0,
            drop: 1.0,
            gravity: 9.81,
            samples: 64,
        }
    }
}

fn brachistochrone_theta(horizontal: f64, drop: f64) -> f64 {
    let target = horizontal / drop;
    let mut lo = 1e-9;
    let mut hi = 2.0 * PI - 1e-9;
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        let ratio = (mid - mid.sin()) / (1.0 - mid.cos());
        if ratio < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Solve the classical brachistochrone descent-time problem.
pub fn solve_brachistochrone(params: BrachistochroneParams) -> VariationalSolutionModel {
    let model = "calculus-of-variations.brachistochrone";
    require(Preconditions::positive(
        model,
        "horizontal",
        params.horizontal,
    ));
    require(Preconditions::positive(model, "drop", params.drop));
    require(Preconditions::positive(model, "gravity", params.gravity));
    require_samples(model, params.samples, 3);

    let theta_end = brachistochrone_theta(params.horizontal, params.drop);
    let radius = params.drop / (1.0 - theta_end.cos());
    let descent_time = theta_end * (radius / params.gravity).sqrt();
    let first_integral = 1.0 / (2.0 * (params.gravity * radius).sqrt());

    let mut residuals = Vec::new();
    let samples = (0..params.samples)
        .map(|i| {
            let u = i as f64 / (params.samples - 1) as f64;
            let theta = u * theta_end;
            let x = radius * (theta - theta.sin());
            let y = -radius * (1.0 - theta.cos());
            let (dy_dx, integrand) = if theta <= EPS {
                (None, None)
            } else {
                let denom = 1.0 - theta.cos();
                let slope = -theta.sin() / denom;
                let depth = -y;
                let value = ((1.0 + slope * slope) / (2.0 * params.gravity * depth)).sqrt();
                let observed =
                    1.0 / ((2.0 * params.gravity * depth).sqrt() * (1.0 + slope * slope).sqrt());
                residuals.push(observed - first_integral);
                (Some(slope), Some(value))
            };
            SolutionSample {
                parameter: Some(theta),
                x,
                y,
                dy_dx,
                integrand,
            }
        })
        .collect::<Vec<_>>();

    let last = samples.last().expect("validated non-empty samples");
    let boundary_error = (last.x - params.horizontal).hypot(last.y + params.drop);

    VariationalSolutionModel {
        problem: VariationalProblem {
            id: "brachistochrone".to_string(),
            name: "Brachistochrone descent".to_string(),
            kind: VariationalProblemKind::Brachistochrone,
            description: "Minimize travel time for a bead sliding from rest under gravity."
                .to_string(),
            functional: FunctionalSpec {
                objective: Extremum::Minimize,
                integrand: "sqrt((1 + y_prime^2) / (2 * g * (-y)))".to_string(),
                independent_variable: "x".to_string(),
                dependent_variable: "y".to_string(),
                derivative_variable: "y_prime".to_string(),
                domain: (0.0, params.horizontal),
                boundary: (
                    BoundaryPoint::new(0.0, 0.0),
                    BoundaryPoint::new(params.horizontal, -params.drop),
                ),
            },
            euler_lagrange: EulerLagrangeModel {
                equation: "L - y_prime * partial L/partial y_prime = constant".to_string(),
                first_integral: Some(
                    "1 / (sqrt(2*g*(-y)) * sqrt(1 + y_prime^2)) = constant".to_string(),
                ),
                natural_boundary_conditions: Vec::new(),
            },
        },
        solution: SolutionFormula {
            expression: "cycloid".to_string(),
            parameterization: Some(vec![
                "x(theta) = r * (theta - sin(theta))".to_string(),
                "y(theta) = -r * (1 - cos(theta))".to_string(),
                "theta in [0, theta_end]".to_string(),
            ]),
            constants: vec![
                ("radius".to_string(), radius),
                ("theta_end".to_string(), theta_end),
                ("gravity".to_string(), params.gravity),
                ("first_integral".to_string(), first_integral),
            ],
        },
        samples,
        diagnostics: SolutionDiagnostics {
            functional_value: descent_time,
            boundary_error,
            first_integral_residual_l2: trapezoid_l2(&residuals),
        },
        topology: graph_for("brachistochrone"),
    }
}

/// Parameters for the symmetric catenoid between equal-radius rings.
#[derive(Clone, Debug, PartialEq)]
pub struct MinimalSurfaceParams {
    pub half_span: f64,
    pub neck_radius: f64,
    pub samples: usize,
}

impl Default for MinimalSurfaceParams {
    fn default() -> Self {
        MinimalSurfaceParams {
            half_span: 0.5,
            neck_radius: 0.5,
            samples: 64,
        }
    }
}

/// Solve the minimal surface of revolution whose solution is a catenoid.
pub fn solve_minimal_surface(params: MinimalSurfaceParams) -> VariationalSolutionModel {
    let model = "calculus-of-variations.minimal-surface";
    require(Preconditions::positive(
        model,
        "half_span",
        params.half_span,
    ));
    require(Preconditions::positive(
        model,
        "neck_radius",
        params.neck_radius,
    ));
    require_samples(model, params.samples, 2);

    let a = params.neck_radius;
    let l = params.half_span;
    let ring_radius = a * (l / a).cosh();
    let area = 2.0 * PI * a * l + PI * a * a * (2.0 * l / a).sinh();
    let samples = (0..params.samples)
        .map(|i| {
            let u = i as f64 / (params.samples - 1) as f64;
            let x = -l + 2.0 * l * u;
            let z = x / a;
            let y = a * z.cosh();
            let slope = z.sinh();
            SolutionSample {
                parameter: Some(x),
                x,
                y,
                dy_dx: Some(slope),
                integrand: Some(2.0 * PI * y * (1.0 + slope * slope).sqrt()),
            }
        })
        .collect::<Vec<_>>();

    let residuals = samples
        .iter()
        .filter_map(|s| s.dy_dx.map(|m| (s.y, m)))
        .map(|(y, m)| y / (1.0 + m * m).sqrt() - a)
        .collect::<Vec<_>>();
    let first = samples.first().expect("validated non-empty samples");
    let last = samples.last().expect("validated non-empty samples");
    let boundary_error = (first.y - ring_radius)
        .abs()
        .max((last.y - ring_radius).abs());

    VariationalSolutionModel {
        problem: VariationalProblem {
            id: "minimal-surface-catenoid".to_string(),
            name: "Minimal surface of revolution".to_string(),
            kind: VariationalProblemKind::MinimalSurfaceOfRevolution,
            description: "Minimize surface area between two equal coaxial rings.".to_string(),
            functional: FunctionalSpec {
                objective: Extremum::Minimize,
                integrand: "2*pi*y*sqrt(1 + y_prime^2)".to_string(),
                independent_variable: "x".to_string(),
                dependent_variable: "y".to_string(),
                derivative_variable: "y_prime".to_string(),
                domain: (-l, l),
                boundary: (
                    BoundaryPoint::new(-l, ring_radius),
                    BoundaryPoint::new(l, ring_radius),
                ),
            },
            euler_lagrange: EulerLagrangeModel {
                equation: "d/dx(2*pi*y*y_prime/sqrt(1 + y_prime^2)) - 2*pi*sqrt(1 + y_prime^2) = 0"
                    .to_string(),
                first_integral: Some("y / sqrt(1 + y_prime^2) = constant".to_string()),
                natural_boundary_conditions: Vec::new(),
            },
        },
        solution: SolutionFormula {
            expression: "y(x) = a * cosh(x / a)".to_string(),
            parameterization: None,
            constants: vec![
                ("a".to_string(), a),
                ("half_span".to_string(), l),
                ("ring_radius".to_string(), ring_radius),
            ],
        },
        samples,
        diagnostics: SolutionDiagnostics {
            functional_value: area,
            boundary_error,
            first_integral_residual_l2: trapezoid_l2(&residuals),
        },
        topology: graph_for("minimal-surface-catenoid"),
    }
}

/// Build the three built-in problem/solution models with default parameters.
pub fn built_in_variational_models() -> Vec<VariationalSolutionModel> {
    vec![
        solve_shortest_curve(ShortestCurveParams::default()),
        solve_brachistochrone(BrachistochroneParams::default()),
        solve_minimal_surface(MinimalSurfaceParams::default()),
    ]
}

/// Lightweight catalog for UI/model discovery layers.
pub fn variational_problem_catalog() -> Vec<VariationalProblem> {
    built_in_variational_models()
        .into_iter()
        .map(|m| m.problem)
        .collect()
}
