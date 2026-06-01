//! Partial differential equation domain models.
//!
//! This module is a typed catalogue of canonical PDE model families. It is not a
//! numerical PDE solver; it records the field variables, operators, boundary
//! conditions, numerical methods, and modeling principles that let higher-level
//! tooling choose an equation family before selecting a discretization.

use std::collections::BTreeSet;

use serde::Serialize;

/// Stable ids for the requested PDE application domains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PdeDomain {
    Electromagnetism,
    QuantumMechanics,
    HeatTransferDiffusion,
    SolidMechanics,
    AcousticsWavePropagation,
    ControlOptimalControl,
    GeometrySurfaces,
    ImageProcessingVision,
    Finance,
    PopulationBiology,
    PlasmaAstrophysics,
    MaterialsScience,
}

impl PdeDomain {
    pub fn all() -> &'static [PdeDomain] {
        &[
            PdeDomain::Electromagnetism,
            PdeDomain::QuantumMechanics,
            PdeDomain::HeatTransferDiffusion,
            PdeDomain::SolidMechanics,
            PdeDomain::AcousticsWavePropagation,
            PdeDomain::ControlOptimalControl,
            PdeDomain::GeometrySurfaces,
            PdeDomain::ImageProcessingVision,
            PdeDomain::Finance,
            PdeDomain::PopulationBiology,
            PdeDomain::PlasmaAstrophysics,
            PdeDomain::MaterialsScience,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            PdeDomain::Electromagnetism => "electromagnetism",
            PdeDomain::QuantumMechanics => "quantum-mechanics",
            PdeDomain::HeatTransferDiffusion => "heat-transfer-diffusion",
            PdeDomain::SolidMechanics => "solid-mechanics",
            PdeDomain::AcousticsWavePropagation => "acoustics-wave-propagation",
            PdeDomain::ControlOptimalControl => "control-optimal-control",
            PdeDomain::GeometrySurfaces => "geometry-surfaces",
            PdeDomain::ImageProcessingVision => "image-processing-vision",
            PdeDomain::Finance => "finance",
            PdeDomain::PopulationBiology => "population-biology",
            PdeDomain::PlasmaAstrophysics => "plasma-astrophysics",
            PdeDomain::MaterialsScience => "materials-science",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            PdeDomain::Electromagnetism => "Electromagnetism",
            PdeDomain::QuantumMechanics => "Quantum mechanics",
            PdeDomain::HeatTransferDiffusion => "Heat transfer / diffusion",
            PdeDomain::SolidMechanics => "Solid mechanics",
            PdeDomain::AcousticsWavePropagation => "Acoustics / wave propagation",
            PdeDomain::ControlOptimalControl => "Control theory / optimal control",
            PdeDomain::GeometrySurfaces => "Geometry and surfaces",
            PdeDomain::ImageProcessingVision => "Image processing / computer vision",
            PdeDomain::Finance => "Finance",
            PdeDomain::PopulationBiology => "Population dynamics / biology",
            PdeDomain::PlasmaAstrophysics => "Plasma physics / astrophysics",
            PdeDomain::MaterialsScience => "Materials science",
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "electromagnetism" | "maxwell" | "maxwells-equations" => {
                Some(PdeDomain::Electromagnetism)
            }
            "quantum-mechanics" | "quantum" | "schrodinger" => Some(PdeDomain::QuantumMechanics),
            "heat-transfer-diffusion" | "heat" | "diffusion" | "heat-transfer" => {
                Some(PdeDomain::HeatTransferDiffusion)
            }
            "solid-mechanics" | "elasticity" | "continuum-mechanics" => {
                Some(PdeDomain::SolidMechanics)
            }
            "acoustics-wave-propagation" | "acoustics" | "waves" | "wave-propagation" => {
                Some(PdeDomain::AcousticsWavePropagation)
            }
            "control-optimal-control" | "control" | "optimal-control" | "hjb" => {
                Some(PdeDomain::ControlOptimalControl)
            }
            "geometry-surfaces" | "geometry" | "surfaces" | "curvature-flow" => {
                Some(PdeDomain::GeometrySurfaces)
            }
            "image-processing-vision" | "image-processing" | "computer-vision" | "vision" => {
                Some(PdeDomain::ImageProcessingVision)
            }
            "finance" | "black-scholes" => Some(PdeDomain::Finance),
            "population-biology" | "biology" | "population-dynamics" | "reaction-diffusion" => {
                Some(PdeDomain::PopulationBiology)
            }
            "plasma-astrophysics" | "plasma" | "astrophysics" | "mhd" => {
                Some(PdeDomain::PlasmaAstrophysics)
            }
            "materials-science" | "materials" | "phase-field" => Some(PdeDomain::MaterialsScience),
            _ => None,
        }
    }
}

/// The recurring structural source of a PDE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PdePrinciple {
    ConservationLaw,
    Diffusion,
    WavePropagation,
    OptimizationVariational,
    GeometryCurvature,
    CoupledFields,
    ReactionKinetics,
    StochasticDuality,
}

impl PdePrinciple {
    pub fn id(self) -> &'static str {
        match self {
            PdePrinciple::ConservationLaw => "conservation-law",
            PdePrinciple::Diffusion => "diffusion",
            PdePrinciple::WavePropagation => "wave-propagation",
            PdePrinciple::OptimizationVariational => "optimization-variational",
            PdePrinciple::GeometryCurvature => "geometry-curvature",
            PdePrinciple::CoupledFields => "coupled-fields",
            PdePrinciple::ReactionKinetics => "reaction-kinetics",
            PdePrinciple::StochasticDuality => "stochastic-duality",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PdePrinciple::ConservationLaw => "conservation laws",
            PdePrinciple::Diffusion => "diffusion / smoothing",
            PdePrinciple::WavePropagation => "wave propagation",
            PdePrinciple::OptimizationVariational => "optimization / variational principles",
            PdePrinciple::GeometryCurvature => "geometry / curvature",
            PdePrinciple::CoupledFields => "coupled field systems",
            PdePrinciple::ReactionKinetics => "reaction kinetics",
            PdePrinciple::StochasticDuality => "stochastic duality",
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "conservation-law" | "conservation" | "mass-momentum-energy" => {
                Some(PdePrinciple::ConservationLaw)
            }
            "diffusion" | "smoothing" | "spreading" => Some(PdePrinciple::Diffusion),
            "wave-propagation" | "waves" | "wave" => Some(PdePrinciple::WavePropagation),
            "optimization-variational" | "optimization" | "variational" | "action" => {
                Some(PdePrinciple::OptimizationVariational)
            }
            "geometry-curvature" | "geometry" | "curvature" => {
                Some(PdePrinciple::GeometryCurvature)
            }
            "coupled-fields" | "coupling" | "coupled" => Some(PdePrinciple::CoupledFields),
            "reaction-kinetics" | "reaction" | "kinetics" => Some(PdePrinciple::ReactionKinetics),
            "stochastic-duality" | "stochastic" | "feynman-kac" => {
                Some(PdePrinciple::StochasticDuality)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PdeLinearity {
    Linear,
    Semilinear,
    Quasilinear,
    Nonlinear,
    Mixed,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdeEquationTemplate {
    pub name: &'static str,
    pub symbolic_form: &'static str,
    pub unknowns: Vec<&'static str>,
    pub independent_variables: Vec<&'static str>,
    pub operators: Vec<&'static str>,
    pub order: usize,
    pub linearity: PdeLinearity,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdeModel {
    pub id: &'static str,
    pub domain_id: &'static str,
    pub domain_title: &'static str,
    pub title: &'static str,
    pub field_variables: Vec<&'static str>,
    pub canonical_equations: Vec<PdeEquationTemplate>,
    pub primary_principles: Vec<PdePrinciple>,
    pub boundary_conditions: Vec<&'static str>,
    pub numerical_methods: Vec<&'static str>,
    pub applications: Vec<&'static str>,
    pub couplings: Vec<&'static str>,
    pub modeling_notes: &'static str,
}

impl PdeModel {
    pub fn domain(&self) -> Option<PdeDomain> {
        PdeDomain::from_id(self.domain_id)
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdePatternDescriptor {
    pub principle: PdePrinciple,
    pub operator_signature: &'static str,
    pub intuition: &'static str,
    pub representative_domains: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdeFramework {
    pub name: &'static str,
    pub statement: &'static str,
    pub common_operators: Vec<&'static str>,
    pub patterns: Vec<PdePatternDescriptor>,
}

pub fn pde_models() -> Vec<PdeModel> {
    vec![
        PdeModel {
            id: "electromagnetism-maxwell",
            domain_id: PdeDomain::Electromagnetism.id(),
            domain_title: PdeDomain::Electromagnetism.title(),
            title: "Maxwell field system",
            field_variables: vec!["E(x,t)", "B(x,t)", "D(x,t)", "H(x,t)", "rho(x,t)", "J(x,t)"],
            canonical_equations: vec![
                equation(
                    "Maxwell equations",
                    "curl E = -d_t B; curl H = J + d_t D; div D = rho; div B = 0",
                    &["E", "B", "D", "H"],
                    &["x", "t"],
                    &["curl", "divergence", "time-derivative", "constitutive-law"],
                    1,
                    PdeLinearity::Mixed,
                ),
                equation(
                    "Electromagnetic wave equation",
                    "laplacian E - mu*epsilon*d_tt E = source(E, rho, J)",
                    &["E"],
                    &["x", "t"],
                    &["laplacian", "second-time-derivative", "source"],
                    2,
                    PdeLinearity::Linear,
                ),
            ],
            primary_principles: vec![
                PdePrinciple::ConservationLaw,
                PdePrinciple::WavePropagation,
                PdePrinciple::CoupledFields,
            ],
            boundary_conditions: vec![
                "perfect conductor: tangential E = 0",
                "impedance or absorbing radiation boundary",
                "material interface continuity for tangential fields",
            ],
            numerical_methods: vec![
                "finite-difference time-domain",
                "finite element electromagnetics",
                "boundary element method",
                "discontinuous Galerkin",
            ],
            applications: vec!["radio waves", "antennas", "optics", "microwave engineering"],
            couplings: vec!["charged particles", "circuits", "material polarization"],
            modeling_notes: "Turns radiation and propagation into coupled vector-field boundary-value or initial-boundary-value problems.",
        },
        PdeModel {
            id: "quantum-schrodinger",
            domain_id: PdeDomain::QuantumMechanics.id(),
            domain_title: PdeDomain::QuantumMechanics.title(),
            title: "Schrodinger wave mechanics",
            field_variables: vec!["psi(x,t)", "V(x,t)", "probability density |psi|^2"],
            canonical_equations: vec![equation(
                "Schrodinger equation",
                "i*hbar*d_t psi = -(hbar^2/(2m))*laplacian psi + V*psi",
                &["psi"],
                &["x", "t"],
                &["time-derivative", "laplacian", "potential-operator", "Hamiltonian"],
                2,
                PdeLinearity::Linear,
            )],
            primary_principles: vec![
                PdePrinciple::WavePropagation,
                PdePrinciple::OptimizationVariational,
            ],
            boundary_conditions: vec![
                "normalizable wavefunction",
                "periodic crystal cell",
                "interface matching across potential barriers",
            ],
            numerical_methods: vec![
                "spectral method",
                "finite difference",
                "finite element",
                "split-step Fourier",
            ],
            applications: vec![
                "electron orbitals in atoms",
                "semiconductor physics",
                "quantum tunneling",
                "molecular simulation",
            ],
            couplings: vec!["electromagnetic potentials", "many-body interactions", "lattice potentials"],
            modeling_notes: "A complex-valued wave PDE whose conserved norm encodes probability.",
        },
        PdeModel {
            id: "heat-diffusion",
            domain_id: PdeDomain::HeatTransferDiffusion.id(),
            domain_title: PdeDomain::HeatTransferDiffusion.title(),
            title: "Heat and diffusion equation",
            field_variables: vec!["u(x,t)", "flux q(x,t)", "source s(x,t)"],
            canonical_equations: vec![equation(
                "Heat equation",
                "d_t u = div(k*grad u) + s",
                &["u"],
                &["x", "t"],
                &["time-derivative", "gradient", "divergence", "laplacian"],
                2,
                PdeLinearity::Linear,
            )],
            primary_principles: vec![PdePrinciple::ConservationLaw, PdePrinciple::Diffusion],
            boundary_conditions: vec![
                "Dirichlet temperature or concentration",
                "Neumann insulated/no-flux boundary",
                "Robin convective exchange",
            ],
            numerical_methods: vec!["finite difference", "finite volume", "finite element", "Crank-Nicolson"],
            applications: vec![
                "thermal conduction in solids",
                "chemical diffusion",
                "pollutant spreading in air or water",
                "diffusion approximations",
            ],
            couplings: vec!["reaction terms", "advection fields", "phase change"],
            modeling_notes: "The canonical smoothing PDE: local gradients drive fluxes that spread scalar quantities.",
        },
        PdeModel {
            id: "solid-mechanics-elasticity",
            domain_id: PdeDomain::SolidMechanics.id(),
            domain_title: PdeDomain::SolidMechanics.title(),
            title: "Continuum solid mechanics",
            field_variables: vec!["u(x,t)", "epsilon(u)", "sigma(x,t)", "body force b(x,t)"],
            canonical_equations: vec![equation(
                "Elastodynamics / elasticity",
                "rho*d_tt u = div sigma + b; sigma = C:epsilon(u)",
                &["u", "sigma"],
                &["x", "t"],
                &["divergence", "strain-gradient", "second-time-derivative", "constitutive-law"],
                2,
                PdeLinearity::Mixed,
            )],
            primary_principles: vec![
                PdePrinciple::ConservationLaw,
                PdePrinciple::WavePropagation,
                PdePrinciple::OptimizationVariational,
            ],
            boundary_conditions: vec![
                "Dirichlet displacement constraints",
                "Neumann traction loads",
                "contact, crack, and interface conditions",
            ],
            numerical_methods: vec!["finite element", "boundary element", "isogeometric analysis", "explicit dynamics"],
            applications: vec![
                "elastic stress-strain response",
                "plastic deformation",
                "fracture mechanics",
                "structural vibrations",
            ],
            couplings: vec!["thermal strain", "damage variables", "fluid-structure interaction"],
            modeling_notes: "FEA is a discretization; the underlying model is a balance-law PDE for displacement and stress.",
        },
        PdeModel {
            id: "acoustics-wave",
            domain_id: PdeDomain::AcousticsWavePropagation.id(),
            domain_title: PdeDomain::AcousticsWavePropagation.title(),
            title: "Acoustic and seismic waves",
            field_variables: vec!["p(x,t)", "v(x,t)", "c(x)", "source s(x,t)"],
            canonical_equations: vec![equation(
                "Wave equation",
                "d_tt p = c^2*laplacian p + s",
                &["p"],
                &["x", "t"],
                &["laplacian", "second-time-derivative", "source"],
                2,
                PdeLinearity::Linear,
            )],
            primary_principles: vec![PdePrinciple::ConservationLaw, PdePrinciple::WavePropagation],
            boundary_conditions: vec![
                "reflecting wall",
                "absorbing / perfectly matched layer",
                "free-surface or impedance boundary",
            ],
            numerical_methods: vec!["finite difference time domain", "spectral element", "finite element", "ray / eikonal methods"],
            applications: vec![
                "sound in air",
                "seismic waves",
                "ultrasound imaging",
                "instrument vibration",
                "structural noise analysis",
            ],
            couplings: vec!["elastic media", "fluid cavities", "sensor arrays"],
            modeling_notes: "Disturbances propagate at finite speed; boundaries and material heterogeneity shape reflections and modes.",
        },
        PdeModel {
            id: "control-hjb-distributed",
            domain_id: PdeDomain::ControlOptimalControl.id(),
            domain_title: PdeDomain::ControlOptimalControl.title(),
            title: "Distributed and optimal control PDEs",
            field_variables: vec!["V(x,t)", "state field y(x,t)", "control u(x,t)", "density rho(x,t)"],
            canonical_equations: vec![
                equation(
                    "Hamilton-Jacobi-Bellman equation",
                    "-d_t V = min_u { L(x,u) + grad V dot f(x,u) + 0.5*tr(a*Hessian V) }",
                    &["V"],
                    &["x", "t"],
                    &["time-derivative", "gradient", "Hessian", "minimization"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
                equation(
                    "Controlled transport / heat equation",
                    "d_t y = A(y) + B*u",
                    &["y"],
                    &["x", "t"],
                    &["generator", "control-input", "boundary-actuation"],
                    2,
                    PdeLinearity::Mixed,
                ),
            ],
            primary_principles: vec![
                PdePrinciple::OptimizationVariational,
                PdePrinciple::ConservationLaw,
                PdePrinciple::Diffusion,
            ],
            boundary_conditions: vec![
                "controlled boundary actuation",
                "terminal value condition",
                "state constraints and viability boundaries",
            ],
            numerical_methods: vec!["semi-Lagrangian method", "finite difference", "model predictive control discretization", "level set methods"],
            applications: vec![
                "temperature control across rods or plates",
                "traffic flow models",
                "fluid flow control",
                "optimal control and decision making",
            ],
            couplings: vec!["feedback laws", "state estimators", "optimization solvers"],
            modeling_notes: "PDEs enter when the controlled state is spatially distributed or the value function lives over a continuous state space.",
        },
        PdeModel {
            id: "geometry-curvature-flow",
            domain_id: PdeDomain::GeometrySurfaces.id(),
            domain_title: PdeDomain::GeometrySurfaces.title(),
            title: "Geometric PDEs and curvature flows",
            field_variables: vec!["surface X(s,t)", "height u(x)", "metric g(t)", "curvature H"],
            canonical_equations: vec![
                equation(
                    "Minimal surface equation",
                    "div(grad u / sqrt(1 + |grad u|^2)) = 0",
                    &["u"],
                    &["x"],
                    &["gradient", "divergence", "curvature"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
                equation(
                    "Mean curvature flow",
                    "d_t X = -H*n",
                    &["X"],
                    &["surface coordinate", "t"],
                    &["curvature", "normal-velocity", "geometric-flow"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
                equation(
                    "Ricci flow",
                    "d_t g = -2*Ric(g)",
                    &["g"],
                    &["manifold point", "t"],
                    &["curvature-tensor", "metric-evolution"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
            ],
            primary_principles: vec![
                PdePrinciple::GeometryCurvature,
                PdePrinciple::OptimizationVariational,
                PdePrinciple::Diffusion,
            ],
            boundary_conditions: vec![
                "fixed boundary curve",
                "free boundary contact angle",
                "topological or metric compatibility",
            ],
            numerical_methods: vec!["finite element on surfaces", "level set method", "front tracking", "discrete differential geometry"],
            applications: vec!["minimal surfaces", "soap films", "mean curvature flow", "Ricci flow"],
            couplings: vec!["surface tension", "topology changes", "manifold constraints"],
            modeling_notes: "Geometry becomes dynamics when curvature determines velocity or equilibrium.",
        },
        PdeModel {
            id: "vision-anisotropic-diffusion",
            domain_id: PdeDomain::ImageProcessingVision.id(),
            domain_title: PdeDomain::ImageProcessingVision.title(),
            title: "Image-field PDEs",
            field_variables: vec!["I(x,y,t)", "flow v(x,y)", "edge diffusivity g"],
            canonical_equations: vec![
                equation(
                    "Perona-Malik anisotropic diffusion",
                    "d_t I = div(g(|grad I|)*grad I)",
                    &["I"],
                    &["x", "y", "t"],
                    &["gradient", "divergence", "edge-stopping diffusivity"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
                equation(
                    "Optical flow constraint",
                    "d_t I + v dot grad I = 0",
                    &["I", "v"],
                    &["x", "y", "t"],
                    &["time-derivative", "gradient", "transport"],
                    1,
                    PdeLinearity::Linear,
                ),
            ],
            primary_principles: vec![
                PdePrinciple::Diffusion,
                PdePrinciple::OptimizationVariational,
                PdePrinciple::GeometryCurvature,
            ],
            boundary_conditions: vec![
                "reflecting image border",
                "periodic tiled image",
                "data attachment / fidelity condition",
            ],
            numerical_methods: vec!["finite difference", "variational solver", "multigrid", "level set method"],
            applications: vec!["edge detection", "image denoising", "optical flow", "shape-from-shading"],
            couplings: vec!["regularization energies", "feature detectors", "inverse problems"],
            modeling_notes: "An image is treated as a continuous scalar field whose gradients guide smoothing, transport, and reconstruction.",
        },
        PdeModel {
            id: "finance-black-scholes",
            domain_id: PdeDomain::Finance.id(),
            domain_title: PdeDomain::Finance.title(),
            title: "Diffusion-pricing PDEs",
            field_variables: vec!["V(S,t)", "asset price S", "volatility sigma", "short rate r"],
            canonical_equations: vec![equation(
                "Black-Scholes equation",
                "d_t V + 0.5*sigma^2*S^2*d_SS V + r*S*d_S V - r*V = 0",
                &["V"],
                &["S", "t"],
                &["time-derivative", "first-price-derivative", "second-price-derivative", "discounting"],
                2,
                PdeLinearity::Linear,
            )],
            primary_principles: vec![
                PdePrinciple::Diffusion,
                PdePrinciple::StochasticDuality,
                PdePrinciple::OptimizationVariational,
            ],
            boundary_conditions: vec![
                "terminal payoff condition",
                "asset-price boundary at S = 0",
                "far-field / growth condition for large S",
            ],
            numerical_methods: vec!["finite difference", "tree / lattice discretization", "Monte Carlo via Feynman-Kac", "Fourier methods"],
            applications: vec!["option pricing", "risk diffusion models", "interest rate dynamics"],
            couplings: vec!["stochastic differential equations", "free-boundary exercise policies", "calibration"],
            modeling_notes: "Stochastic processes induce parabolic PDEs; pricing often looks like heat diffusion with financial boundary data.",
        },
        PdeModel {
            id: "biology-reaction-diffusion",
            domain_id: PdeDomain::PopulationBiology.id(),
            domain_title: PdeDomain::PopulationBiology.title(),
            title: "Spatial biology reaction-diffusion",
            field_variables: vec!["u_i(x,t)", "population density", "chemical concentration", "infection compartments"],
            canonical_equations: vec![equation(
                "Reaction-diffusion system",
                "d_t u_i = div(D_i*grad u_i) + f_i(u_1,...,u_n)",
                &["u_i"],
                &["x", "t"],
                &["time-derivative", "diffusion", "reaction-vector-field"],
                2,
                PdeLinearity::Semilinear,
            )],
            primary_principles: vec![
                PdePrinciple::Diffusion,
                PdePrinciple::ReactionKinetics,
                PdePrinciple::ConservationLaw,
            ],
            boundary_conditions: vec![
                "no-flux habitat boundary",
                "absorbing boundary",
                "source/sink patch boundary",
            ],
            numerical_methods: vec!["finite difference", "finite volume", "operator splitting", "agent-to-field coupling"],
            applications: vec![
                "Turing patterns",
                "spatial SIR epidemics",
                "neural field models",
                "tumor growth models",
            ],
            couplings: vec!["chemotaxis", "birth-death reactions", "mobility networks"],
            modeling_notes: "Local reactions create or remove density while diffusion or taxis moves it through space.",
        },
        PdeModel {
            id: "plasma-mhd",
            domain_id: PdeDomain::PlasmaAstrophysics.id(),
            domain_title: PdeDomain::PlasmaAstrophysics.title(),
            title: "Magnetohydrodynamics",
            field_variables: vec!["rho(x,t)", "u(x,t)", "B(x,t)", "p(x,t)", "E(x,t)"],
            canonical_equations: vec![equation(
                "Resistive MHD system",
                "d_t rho + div(rho*u)=0; d_t B = curl(u x B - eta*curl B); momentum/energy balances",
                &["rho", "u", "B", "p"],
                &["x", "t"],
                &["divergence", "curl", "advection", "Lorentz force", "conservation form"],
                2,
                PdeLinearity::Nonlinear,
            )],
            primary_principles: vec![
                PdePrinciple::ConservationLaw,
                PdePrinciple::CoupledFields,
                PdePrinciple::WavePropagation,
            ],
            boundary_conditions: vec![
                "conducting wall",
                "inflow/outflow plasma boundary",
                "divergence-free magnetic constraint",
            ],
            numerical_methods: vec!["finite volume", "constrained transport", "particle-in-cell coupling", "shock-capturing schemes"],
            applications: vec!["magnetohydrodynamics", "star formation", "solar flares", "fusion reactor modeling"],
            couplings: vec!["fluid dynamics", "electromagnetism", "radiation transport", "gravity"],
            modeling_notes: "Combines continuum fluid balance laws with electromagnetic field evolution.",
        },
        PdeModel {
            id: "materials-phase-field",
            domain_id: PdeDomain::MaterialsScience.id(),
            domain_title: PdeDomain::MaterialsScience.title(),
            title: "Phase-field material evolution",
            field_variables: vec!["phi(x,t)", "chemical potential mu(x,t)", "free energy F[phi]"],
            canonical_equations: vec![
                equation(
                    "Allen-Cahn equation",
                    "d_t phi = -M*delta F/delta phi",
                    &["phi"],
                    &["x", "t"],
                    &["variational-derivative", "reaction", "gradient-flow"],
                    2,
                    PdeLinearity::Nonlinear,
                ),
                equation(
                    "Cahn-Hilliard equation",
                    "d_t phi = div(M*grad mu); mu = delta F/delta phi",
                    &["phi", "mu"],
                    &["x", "t"],
                    &["divergence", "gradient", "variational-derivative", "fourth-order-diffusion"],
                    4,
                    PdeLinearity::Nonlinear,
                ),
            ],
            primary_principles: vec![
                PdePrinciple::OptimizationVariational,
                PdePrinciple::Diffusion,
                PdePrinciple::GeometryCurvature,
            ],
            boundary_conditions: vec![
                "periodic microstructure cell",
                "no-flux chemical potential",
                "wetting/contact-angle boundary",
            ],
            numerical_methods: vec!["finite element", "spectral method", "convex splitting", "adaptive mesh refinement"],
            applications: vec!["crystal growth", "phase transitions", "grain boundary evolution"],
            couplings: vec!["elastic strain energy", "thermal fields", "composition constraints"],
            modeling_notes: "Material organization is modeled as energy-driven evolution of order parameters and interfaces.",
        },
    ]
}

pub fn model_for_domain(domain: PdeDomain) -> Option<PdeModel> {
    pde_models()
        .into_iter()
        .find(|model| model.domain_id == domain.id())
}

pub fn models_for_principle(principle: PdePrinciple) -> Vec<PdeModel> {
    pde_models()
        .into_iter()
        .filter(|model| model.primary_principles.contains(&principle))
        .collect()
}

pub fn pde_unifying_framework() -> PdeFramework {
    PdeFramework {
        name: "Variational operators plus balance laws",
        statement: "Most PDE models combine local balance laws, constitutive operators, boundary data, and often an energy/action principle.",
        common_operators: vec![
            "d_t and d_tt",
            "gradient",
            "divergence",
            "curl",
            "laplacian",
            "Hessian",
            "curvature",
            "variational derivative",
            "Hamiltonian / generator",
        ],
        patterns: vec![
            PdePatternDescriptor {
                principle: PdePrinciple::ConservationLaw,
                operator_signature: "d_t conserved_quantity + div(flux) = source",
                intuition: "Mass, charge, momentum, energy, or probability changes only through fluxes and sources.",
                representative_domains: vec![
                    "electromagnetism",
                    "solid-mechanics",
                    "population-biology",
                    "plasma-astrophysics",
                ],
            },
            PdePatternDescriptor {
                principle: PdePrinciple::Diffusion,
                operator_signature: "d_t u = div(k*grad u) + source",
                intuition: "Gradients create fluxes that smooth or spread scalar fields.",
                representative_domains: vec![
                    "heat-transfer-diffusion",
                    "image-processing-vision",
                    "finance",
                    "materials-science",
                ],
            },
            PdePatternDescriptor {
                principle: PdePrinciple::WavePropagation,
                operator_signature: "d_tt u = c^2*laplacian u + source",
                intuition: "Disturbances propagate with finite speed and reflect from boundaries.",
                representative_domains: vec![
                    "electromagnetism",
                    "quantum-mechanics",
                    "solid-mechanics",
                    "acoustics-wave-propagation",
                    "plasma-astrophysics",
                ],
            },
            PdePatternDescriptor {
                principle: PdePrinciple::OptimizationVariational,
                operator_signature: "Euler-Lagrange, gradient flow, or HJB optimality condition",
                intuition: "The PDE expresses stationarity, steepest descent, or dynamic programming over continuous states.",
                representative_domains: vec![
                    "quantum-mechanics",
                    "control-optimal-control",
                    "geometry-surfaces",
                    "materials-science",
                ],
            },
            PdePatternDescriptor {
                principle: PdePrinciple::GeometryCurvature,
                operator_signature: "normal velocity or equilibrium is driven by curvature",
                intuition: "The shape of a surface determines how it moves or equilibrates.",
                representative_domains: vec![
                    "geometry-surfaces",
                    "image-processing-vision",
                    "materials-science",
                ],
            },
        ],
    }
}

pub fn validate_pde_catalog() -> Result<(), String> {
    let models = pde_models();
    if models.len() != PdeDomain::all().len() {
        return Err(format!(
            "expected {} PDE domain models, found {}",
            PdeDomain::all().len(),
            models.len()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut domains = BTreeSet::new();
    for model in &models {
        if !ids.insert(model.id) {
            return Err(format!("duplicate PDE model id `{}`", model.id));
        }
        domains.insert(model.domain_id);
        if model.canonical_equations.is_empty() {
            return Err(format!("PDE model `{}` has no equations", model.id));
        }
        if model.primary_principles.is_empty() {
            return Err(format!("PDE model `{}` has no principles", model.id));
        }
        if model.boundary_conditions.is_empty() {
            return Err(format!(
                "PDE model `{}` has no boundary conditions",
                model.id
            ));
        }
        if model.numerical_methods.is_empty() {
            return Err(format!("PDE model `{}` has no numerical methods", model.id));
        }
        if model.applications.is_empty() {
            return Err(format!("PDE model `{}` has no applications", model.id));
        }
    }

    for domain in PdeDomain::all() {
        if !domains.contains(domain.id()) {
            return Err(format!("missing PDE domain `{}`", domain.id()));
        }
    }

    let framework = pde_unifying_framework();
    for principle in [
        PdePrinciple::ConservationLaw,
        PdePrinciple::Diffusion,
        PdePrinciple::WavePropagation,
        PdePrinciple::OptimizationVariational,
        PdePrinciple::GeometryCurvature,
    ] {
        if !framework
            .patterns
            .iter()
            .any(|pattern| pattern.principle == principle)
        {
            return Err(format!("missing framework principle `{}`", principle.id()));
        }
    }

    Ok(())
}

fn equation(
    name: &'static str,
    symbolic_form: &'static str,
    unknowns: &[&'static str],
    independent_variables: &[&'static str],
    operators: &[&'static str],
    order: usize,
    linearity: PdeLinearity,
) -> PdeEquationTemplate {
    PdeEquationTemplate {
        name,
        symbolic_form,
        unknowns: unknowns.to_vec(),
        independent_variables: independent_variables.to_vec(),
        operators: operators.to_vec(),
        order,
        linearity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_covers_all_requested_domains() {
        validate_pde_catalog().expect("PDE catalogue should be complete");

        let models = pde_models();
        assert_eq!(models.len(), 12);
        for domain in PdeDomain::all() {
            let matches: Vec<_> = models
                .iter()
                .filter(|model| model.domain_id == domain.id())
                .collect();
            assert_eq!(matches.len(), 1, "domain {}", domain.id());
        }
    }

    #[test]
    fn named_domain_models_preserve_user_requested_examples() {
        let maxwell = model_for_domain(PdeDomain::Electromagnetism).unwrap();
        for app in ["radio waves", "antennas", "optics", "microwave engineering"] {
            assert!(maxwell.applications.contains(&app));
        }

        let biology = model_for_domain(PdeDomain::PopulationBiology).unwrap();
        for app in [
            "Turing patterns",
            "spatial SIR epidemics",
            "neural field models",
            "tumor growth models",
        ] {
            assert!(biology.applications.contains(&app));
        }

        let finance = model_for_domain(PdeDomain::Finance).unwrap();
        assert!(finance
            .canonical_equations
            .iter()
            .any(|eq| eq.name == "Black-Scholes equation"));
    }

    #[test]
    fn unifying_framework_contains_big_picture_patterns() {
        let framework = pde_unifying_framework();
        let principles: BTreeSet<_> = framework
            .patterns
            .iter()
            .map(|pattern| pattern.principle)
            .collect();
        for principle in [
            PdePrinciple::ConservationLaw,
            PdePrinciple::Diffusion,
            PdePrinciple::WavePropagation,
            PdePrinciple::OptimizationVariational,
            PdePrinciple::GeometryCurvature,
        ] {
            assert!(principles.contains(&principle), "{}", principle.id());
        }
    }
}
