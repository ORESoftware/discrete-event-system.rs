//! Solver-style math-programming facade over the in-house LP and IP/MIP solvers.
//!
//! The low-level solvers intentionally stay close to their TypeScript ports:
//! `LPProblem` accepts LP rows/bounds, and `IPMIPProblem` accepts non-negative
//! variables with `<=` rows. This module is the compatibility layer users expect
//! from tools such as OR-Tools, Gurobi, CPLEX, FICO Xpress, LINDO, SCIP, GLPK, and HiGHS:
//! named variables, `<=`/`>=`/`=` rows, continuous/integer/binary/semi-continuous
//! domains, and indicator constraints. The compiler lowers those features into
//! the existing native solvers and keeps enough metadata to map solutions back
//! to the user's original variables.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::des::general::ip_mip_des::{
    solve_ipmip_with_des, IPMIPProblem, IPMIPSolveOptions, IPMIPStatus,
};
use crate::des::general::lp::{
    solve_lp_external, solve_lp_internal, solve_lp_internal_ipm, ExternalSolverOptions,
    InternalInteriorPointOptions, InternalSimplexOptions, LPProblem, LPSolution, LPStatus,
    Sense as LpSense,
};
use serde_json::{json, Value};

/// Objective direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Max,
    Min,
}

impl ObjectiveSense {
    fn to_lp(self) -> LpSense {
        match self {
            ObjectiveSense::Max => LpSense::Max,
            ObjectiveSense::Min => LpSense::Min,
        }
    }
}

/// Variable domain supported by the facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableType {
    Continuous,
    Integer,
    Binary,
    /// Either zero, or a continuous value in `[lb, ub]`.
    SemiContinuous,
    /// Either zero, or an integer value in `[lb, ub]`.
    SemiInteger,
}

/// Linear row sense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSense {
    Le,
    Ge,
    Eq,
}

impl RowSense {
    fn as_str(self) -> &'static str {
        match self {
            RowSense::Le => "<=",
            RowSense::Ge => ">=",
            RowSense::Eq => "=",
        }
    }
}

/// A named decision variable.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    pub name: String,
    pub obj: f64,
    pub lb: Option<f64>,
    pub ub: Option<f64>,
    pub var_type: VariableType,
}

/// Quadratic objective term `coeff * x[var_i] * x[var_j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticObjectiveTerm {
    pub var_i: usize,
    pub var_j: usize,
    pub coeff: f64,
}

/// Quadratic constraint term `coeff * x[var_i] * x[var_j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticConstraintTerm {
    pub var_i: usize,
    pub var_j: usize,
    pub coeff: f64,
}

/// A sparse linear row.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraint {
    pub name: String,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

/// Convex quadratic row: `quadratic_terms + linear_terms sense rhs`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticConstraint {
    pub name: String,
    pub quadratic_terms: Vec<QuadraticConstraintTerm>,
    pub linear_terms: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

/// Reified linear row: if `binary_var == active_value`, enforce the row.
#[derive(Clone, Debug, PartialEq)]
pub struct IndicatorConstraint {
    pub name: String,
    pub binary_var: usize,
    pub active_value: bool,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

/// Special ordered set type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SOSType {
    /// At most one member can be non-zero.
    Sos1,
    /// At most two non-zero members, and those members must be adjacent by weight.
    Sos2,
}

/// Special ordered set constraint. Members are sorted by `weight` for SOS2.
#[derive(Clone, Debug, PartialEq)]
pub struct SOSConstraint {
    pub name: String,
    pub sos_type: SOSType,
    pub members: Vec<(usize, f64)>,
}

/// Fixed-size interval used by CP-SAT-style scheduling constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalTerm {
    pub start_var: usize,
    pub duration: f64,
    pub end_var: usize,
    pub presence_var: Option<usize>,
}

/// Affine expression used inside second-order cone constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineTerm {
    pub coeffs: Vec<(usize, f64)>,
    pub constant: f64,
}

/// Convex second-order cone constraint: `||terms(x)||_2 <= rhs_coeffs*x + rhs_constant`.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderConeConstraint {
    pub name: String,
    pub terms: Vec<AffineTerm>,
    pub rhs_coeffs: Vec<(usize, f64)>,
    pub rhs_constant: f64,
}

/// Common commercial-solver "general constraints" that lower to linear MIP.
#[derive(Clone, Debug, PartialEq)]
pub enum GeneralConstraint {
    BinaryAnd {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
    },
    BinaryOr {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
    },
    Abs {
        name: String,
        result_var: usize,
        operand_var: usize,
    },
    Max {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
    },
    Min {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
    },
    PiecewiseLinear {
        name: String,
        x_var: usize,
        y_var: usize,
        points: Vec<(f64, f64)>,
    },
    AllDifferent {
        name: String,
        variables: Vec<usize>,
    },
    AllowedAssignments {
        name: String,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    NoOverlap {
        name: String,
        intervals: Vec<IntervalTerm>,
    },
    NoOverlap2D {
        name: String,
        x_intervals: Vec<IntervalTerm>,
        y_intervals: Vec<IntervalTerm>,
    },
    Cumulative {
        name: String,
        intervals: Vec<IntervalTerm>,
        demands: Vec<f64>,
        capacity: f64,
    },
}

/// Linear secondary objective used for hierarchical/blended multi-objective solves.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearObjective {
    pub name: String,
    pub sense: ObjectiveSense,
    pub priority: i32,
    pub weight: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub coeffs: Vec<(usize, f64)>,
}

/// A solver-style mathematical program.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgram {
    pub sense: ObjectiveSense,
    pub variables: Vec<Variable>,
    pub quadratic_objective: Vec<QuadraticObjectiveTerm>,
    pub secondary_objectives: Vec<LinearObjective>,
    pub quadratic_constraints: Vec<QuadraticConstraint>,
    pub constraints: Vec<LinearConstraint>,
    pub second_order_cones: Vec<SecondOrderConeConstraint>,
    pub indicators: Vec<IndicatorConstraint>,
    pub sos: Vec<SOSConstraint>,
    pub general_constraints: Vec<GeneralConstraint>,
}

impl MathProgram {
    pub fn new(sense: ObjectiveSense) -> Self {
        MathProgram {
            sense,
            variables: Vec::new(),
            quadratic_objective: Vec::new(),
            secondary_objectives: Vec::new(),
            quadratic_constraints: Vec::new(),
            constraints: Vec::new(),
            second_order_cones: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            general_constraints: Vec::new(),
        }
    }

    pub fn add_var(
        &mut self,
        name: impl Into<String>,
        var_type: VariableType,
        obj: f64,
        lb: Option<f64>,
        ub: Option<f64>,
    ) -> Result<usize, MathProgramError> {
        let var = Variable {
            name: name.into(),
            obj,
            lb,
            ub,
            var_type,
        };
        validate_variable(&var)?;
        self.variables.push(var);
        Ok(self.variables.len() - 1)
    }

    pub fn add_continuous_var(
        &mut self,
        name: impl Into<String>,
        obj: f64,
        lb: Option<f64>,
        ub: Option<f64>,
    ) -> Result<usize, MathProgramError> {
        self.add_var(name, VariableType::Continuous, obj, lb, ub)
    }

    pub fn add_integer_var(
        &mut self,
        name: impl Into<String>,
        obj: f64,
        lb: Option<f64>,
        ub: Option<f64>,
    ) -> Result<usize, MathProgramError> {
        self.add_var(name, VariableType::Integer, obj, lb, ub)
    }

    pub fn add_binary_var(
        &mut self,
        name: impl Into<String>,
        obj: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_var(name, VariableType::Binary, obj, Some(0.0), Some(1.0))
    }

    pub fn add_quadratic_objective_term(
        &mut self,
        var_i: usize,
        var_j: usize,
        coeff: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_quadratic_objective_term(var_i, var_j, coeff)?;
        self.quadratic_objective.push(QuadraticObjectiveTerm {
            var_i,
            var_j,
            coeff,
        });
        Ok(self.quadratic_objective.len() - 1)
    }

    pub fn add_quadratic_constraint(
        &mut self,
        name: impl Into<String>,
        quadratic_terms: Vec<(usize, usize, f64)>,
        linear_terms: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        let terms = quadratic_terms
            .into_iter()
            .map(|(var_i, var_j, coeff)| QuadraticConstraintTerm {
                var_i,
                var_j,
                coeff,
            })
            .collect::<Vec<_>>();
        self.validate_quadratic_constraint_args(&terms, &linear_terms, sense, rhs)?;
        self.quadratic_constraints.push(QuadraticConstraint {
            name: name.into(),
            quadratic_terms: terms,
            linear_terms,
            sense,
            rhs,
        });
        Ok(self.quadratic_constraints.len() - 1)
    }

    pub fn add_semi_continuous_var(
        &mut self,
        name: impl Into<String>,
        obj: f64,
        lb: f64,
        ub: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_var(name, VariableType::SemiContinuous, obj, Some(lb), Some(ub))
    }

    pub fn add_semi_integer_var(
        &mut self,
        name: impl Into<String>,
        obj: f64,
        lb: f64,
        ub: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_var(name, VariableType::SemiInteger, obj, Some(lb), Some(ub))
    }

    pub fn add_constraint(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        validate_coeffs(self.variables.len(), &coeffs)?;
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "constraint rhs for `{}`",
                name.into()
            )));
        }
        self.constraints.push(LinearConstraint {
            name: name.into(),
            coeffs,
            sense,
            rhs,
        });
        Ok(self.constraints.len() - 1)
    }

    pub fn affine_term(coeffs: Vec<(usize, f64)>, constant: f64) -> AffineTerm {
        AffineTerm { coeffs, constant }
    }

    pub fn add_second_order_cone(
        &mut self,
        name: impl Into<String>,
        terms: Vec<AffineTerm>,
        rhs_coeffs: Vec<(usize, f64)>,
        rhs_constant: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_second_order_cone_args(&terms, &rhs_coeffs, rhs_constant)?;
        self.second_order_cones.push(SecondOrderConeConstraint {
            name: name.into(),
            terms,
            rhs_coeffs,
            rhs_constant,
        });
        Ok(self.second_order_cones.len() - 1)
    }

    pub fn add_rotated_second_order_cone(
        &mut self,
        name: impl Into<String>,
        lhs_a: AffineTerm,
        lhs_b: AffineTerm,
        terms: Vec<AffineTerm>,
    ) -> Result<usize, MathProgramError> {
        let name = name.into();
        let mut transformed_terms = terms
            .iter()
            .map(|term| scale_affine_term(term, 2.0_f64.sqrt()))
            .collect::<Vec<_>>();
        transformed_terms.push(add_affine_terms(&lhs_a, &lhs_b, -1.0));
        let rhs = add_affine_terms(&lhs_a, &lhs_b, 1.0);
        self.validate_second_order_cone_args(&transformed_terms, &rhs.coeffs, rhs.constant)?;

        self.add_constraint(
            format!("{name}__lhs_a_nonnegative"),
            lhs_a.coeffs,
            RowSense::Ge,
            -lhs_a.constant,
        )?;
        self.add_constraint(
            format!("{name}__lhs_b_nonnegative"),
            lhs_b.coeffs,
            RowSense::Ge,
            -lhs_b.constant,
        )?;
        self.second_order_cones.push(SecondOrderConeConstraint {
            name,
            terms: transformed_terms,
            rhs_coeffs: rhs.coeffs,
            rhs_constant: rhs.constant,
        });
        Ok(self.second_order_cones.len() - 1)
    }

    pub fn add_indicator(
        &mut self,
        name: impl Into<String>,
        binary_var: usize,
        active_value: bool,
        coeffs: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        if binary_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "indicator binary var index {binary_var} out of bounds"
            )));
        }
        if self.variables[binary_var].var_type != VariableType::Binary {
            return Err(MathProgramError::Unsupported(format!(
                "indicator var `{}` must be binary",
                self.variables[binary_var].name
            )));
        }
        validate_coeffs(self.variables.len(), &coeffs)?;
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite("indicator rhs".to_string()));
        }
        self.indicators.push(IndicatorConstraint {
            name: name.into(),
            binary_var,
            active_value,
            coeffs,
            sense,
            rhs,
        });
        Ok(self.indicators.len() - 1)
    }

    pub fn add_sos1(
        &mut self,
        name: impl Into<String>,
        members: Vec<(usize, f64)>,
    ) -> Result<usize, MathProgramError> {
        self.add_sos(name, SOSType::Sos1, members)
    }

    pub fn add_sos2(
        &mut self,
        name: impl Into<String>,
        members: Vec<(usize, f64)>,
    ) -> Result<usize, MathProgramError> {
        self.add_sos(name, SOSType::Sos2, members)
    }

    pub fn add_sos(
        &mut self,
        name: impl Into<String>,
        sos_type: SOSType,
        members: Vec<(usize, f64)>,
    ) -> Result<usize, MathProgramError> {
        validate_sos_members(self.variables.len(), sos_type, &members)?;
        self.sos.push(SOSConstraint {
            name: name.into(),
            sos_type,
            members,
        });
        Ok(self.sos.len() - 1)
    }

    pub fn add_secondary_objective(
        &mut self,
        name: impl Into<String>,
        sense: ObjectiveSense,
        priority: i32,
        weight: f64,
        coeffs: Vec<(usize, f64)>,
    ) -> Result<usize, MathProgramError> {
        self.add_secondary_objective_with_tolerances(
            name, sense, priority, weight, 1e-7, 1e-9, coeffs,
        )
    }

    pub fn add_secondary_objective_with_tolerances(
        &mut self,
        name: impl Into<String>,
        sense: ObjectiveSense,
        priority: i32,
        weight: f64,
        abs_tol: f64,
        rel_tol: f64,
        coeffs: Vec<(usize, f64)>,
    ) -> Result<usize, MathProgramError> {
        validate_linear_objective_args(self.variables.len(), weight, abs_tol, rel_tol, &coeffs)?;
        self.secondary_objectives.push(LinearObjective {
            name: name.into(),
            sense,
            priority,
            weight,
            abs_tol,
            rel_tol,
            coeffs,
        });
        Ok(self.secondary_objectives.len() - 1)
    }

    pub fn add_binary_and(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_binary_general_args(result_var, &operands)?;
        self.general_constraints.push(GeneralConstraint::BinaryAnd {
            name: name.into(),
            result_var,
            operands,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_binary_or(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_binary_general_args(result_var, &operands)?;
        self.general_constraints.push(GeneralConstraint::BinaryOr {
            name: name.into(),
            result_var,
            operands,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_abs(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operand_var: usize,
    ) -> Result<usize, MathProgramError> {
        if result_var >= self.variables.len() || operand_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "abs constraint references result {result_var} and operand {operand_var} with {} variables",
                self.variables.len()
            )));
        }
        if self.variables[result_var].lb.is_some_and(|lb| lb < 0.0) {
            return Err(MathProgramError::InvalidBound(format!(
                "abs result `{}` must have non-negative lower bound",
                self.variables[result_var].name
            )));
        }
        if variable_bounds(&self.variables[operand_var]).is_none() {
            return Err(MathProgramError::UnboundedBigM(format!(
                "abs operand `{}` requires finite bounds",
                self.variables[operand_var].name
            )));
        }
        self.general_constraints.push(GeneralConstraint::Abs {
            name: name.into(),
            result_var,
            operand_var,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_max(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_extreme_general_args("max", result_var, &operands)?;
        self.general_constraints.push(GeneralConstraint::Max {
            name: name.into(),
            result_var,
            operands,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_min(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_extreme_general_args("min", result_var, &operands)?;
        self.general_constraints.push(GeneralConstraint::Min {
            name: name.into(),
            result_var,
            operands,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_piecewise_linear(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        points: Vec<(f64, f64)>,
    ) -> Result<usize, MathProgramError> {
        self.validate_piecewise_linear_args(x_var, y_var, &points)?;
        self.general_constraints
            .push(GeneralConstraint::PiecewiseLinear {
                name: name.into(),
                x_var,
                y_var,
                points,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_all_different(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_all_different_args(&variables)?;
        self.general_constraints
            .push(GeneralConstraint::AllDifferent {
                name: name.into(),
                variables,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_allowed_assignments(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    ) -> Result<usize, MathProgramError> {
        self.validate_allowed_assignments_args(&variables, &tuples)?;
        self.general_constraints
            .push(GeneralConstraint::AllowedAssignments {
                name: name.into(),
                variables,
                tuples,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn interval(start_var: usize, duration: f64, end_var: usize) -> IntervalTerm {
        IntervalTerm {
            start_var,
            duration,
            end_var,
            presence_var: None,
        }
    }

    pub fn optional_interval(
        start_var: usize,
        duration: f64,
        end_var: usize,
        presence_var: usize,
    ) -> IntervalTerm {
        IntervalTerm {
            start_var,
            duration,
            end_var,
            presence_var: Some(presence_var),
        }
    }

    pub fn add_no_overlap(
        &mut self,
        name: impl Into<String>,
        intervals: Vec<IntervalTerm>,
    ) -> Result<usize, MathProgramError> {
        self.validate_interval_args("no-overlap", &intervals)?;
        self.general_constraints.push(GeneralConstraint::NoOverlap {
            name: name.into(),
            intervals,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_no_overlap_2d(
        &mut self,
        name: impl Into<String>,
        x_intervals: Vec<IntervalTerm>,
        y_intervals: Vec<IntervalTerm>,
    ) -> Result<usize, MathProgramError> {
        self.validate_no_overlap_2d_args(&x_intervals, &y_intervals)?;
        self.general_constraints
            .push(GeneralConstraint::NoOverlap2D {
                name: name.into(),
                x_intervals,
                y_intervals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_cumulative(
        &mut self,
        name: impl Into<String>,
        intervals: Vec<IntervalTerm>,
        demands: Vec<f64>,
        capacity: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_cumulative_args(&intervals, &demands, capacity)?;
        self.general_constraints
            .push(GeneralConstraint::Cumulative {
                name: name.into(),
                intervals,
                demands,
                capacity,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn has_discrete_features(&self) -> bool {
        self.variables
            .iter()
            .any(|v| v.var_type != VariableType::Continuous)
            || !self.indicators.is_empty()
            || !self.sos.is_empty()
            || !self.general_constraints.is_empty()
    }

    pub fn has_quadratic_objective(&self) -> bool {
        !self.quadratic_objective.is_empty()
    }

    pub fn has_quadratic_constraints(&self) -> bool {
        !self.quadratic_constraints.is_empty()
    }

    pub fn has_conic_constraints(&self) -> bool {
        !self.second_order_cones.is_empty()
    }

    /// Convert a pure continuous model into the native LP representation.
    pub fn to_lp_problem(&self) -> Result<LPProblem, MathProgramError> {
        self.validate()?;
        if self.has_discrete_features()
            || self.has_quadratic_objective()
            || self.has_quadratic_constraints()
            || self.has_conic_constraints()
        {
            return Err(MathProgramError::Unsupported(
                "to_lp_problem requires a pure continuous linear model".to_string(),
            ));
        }
        self.to_linear_relaxation_lp_problem()
    }

    fn to_linear_relaxation_lp_problem(&self) -> Result<LPProblem, MathProgramError> {
        let n = self.variables.len();
        let mut a_ub = Vec::new();
        let mut b_ub = Vec::new();
        let mut a_eq = Vec::new();
        let mut b_eq = Vec::new();
        let mut con_names = Vec::new();

        for row in &self.constraints {
            let dense = dense_row(n, &row.coeffs);
            match row.sense {
                RowSense::Le => {
                    a_ub.push(dense);
                    b_ub.push(row.rhs);
                    con_names.push(row.name.clone());
                }
                RowSense::Ge => {
                    a_ub.push(scale_row(&dense, -1.0));
                    b_ub.push(-row.rhs);
                    con_names.push(row.name.clone());
                }
                RowSense::Eq => {
                    a_eq.push(dense);
                    b_eq.push(row.rhs);
                    con_names.push(row.name.clone());
                }
            }
        }

        Ok(LPProblem {
            sense: self.sense.to_lp(),
            c: self.variables.iter().map(|v| v.obj).collect(),
            a_ub: (!a_ub.is_empty()).then_some(a_ub),
            b_ub: (!b_ub.is_empty()).then_some(b_ub),
            a_eq: (!a_eq.is_empty()).then_some(a_eq),
            b_eq: (!b_eq.is_empty()).then_some(b_eq),
            lb: Some(self.variables.iter().map(|v| v.lb).collect()),
            ub: Some(self.variables.iter().map(|v| v.ub).collect()),
            var_names: Some(self.variables.iter().map(|v| v.name.clone()).collect()),
            con_names: (!con_names.is_empty()).then_some(con_names),
        })
    }

    pub fn validate(&self) -> Result<(), MathProgramError> {
        if self.variables.is_empty() {
            return Err(MathProgramError::EmptyModel);
        }
        for var in &self.variables {
            validate_variable(var)?;
        }
        for term in &self.quadratic_objective {
            self.validate_quadratic_objective_term(term.var_i, term.var_j, term.coeff)?;
        }
        for objective in &self.secondary_objectives {
            validate_linear_objective_args(
                self.variables.len(),
                objective.weight,
                objective.abs_tol,
                objective.rel_tol,
                &objective.coeffs,
            )?;
        }
        for row in &self.quadratic_constraints {
            self.validate_quadratic_constraint_args(
                &row.quadratic_terms,
                &row.linear_terms,
                row.sense,
                row.rhs,
            )?;
        }
        for row in &self.constraints {
            validate_coeffs(self.variables.len(), &row.coeffs)?;
            if !row.rhs.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "constraint rhs for `{}`",
                    row.name
                )));
            }
        }
        for cone in &self.second_order_cones {
            self.validate_second_order_cone_args(&cone.terms, &cone.rhs_coeffs, cone.rhs_constant)?;
        }
        for indicator in &self.indicators {
            if indicator.binary_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "indicator `{}` binary var index {} out of bounds",
                    indicator.name, indicator.binary_var
                )));
            }
            if self.variables[indicator.binary_var].var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "indicator `{}` variable `{}` is not binary",
                    indicator.name, self.variables[indicator.binary_var].name
                )));
            }
            validate_coeffs(self.variables.len(), &indicator.coeffs)?;
        }
        for sos in &self.sos {
            validate_sos_members(self.variables.len(), sos.sos_type, &sos.members)?;
            for &(idx, _) in &sos.members {
                if variable_bounds(&self.variables[idx]).is_none() {
                    return Err(MathProgramError::UnboundedBigM(format!(
                        "SOS `{}` member `{}` requires finite bounds",
                        sos.name, self.variables[idx].name
                    )));
                }
            }
        }
        for general in &self.general_constraints {
            match general {
                GeneralConstraint::BinaryAnd {
                    result_var,
                    operands,
                    ..
                }
                | GeneralConstraint::BinaryOr {
                    result_var,
                    operands,
                    ..
                } => self.validate_binary_general_args(*result_var, operands)?,
                GeneralConstraint::Abs {
                    result_var,
                    operand_var,
                    name,
                } => {
                    if *result_var >= self.variables.len() || *operand_var >= self.variables.len() {
                        return Err(MathProgramError::BadIndex(format!(
                            "abs `{name}` references out-of-bounds variables"
                        )));
                    }
                    if self.variables[*result_var].lb.is_some_and(|lb| lb < 0.0) {
                        return Err(MathProgramError::InvalidBound(format!(
                            "abs result `{}` must have non-negative lower bound",
                            self.variables[*result_var].name
                        )));
                    }
                    if variable_bounds(&self.variables[*operand_var]).is_none() {
                        return Err(MathProgramError::UnboundedBigM(format!(
                            "abs `{name}` operand `{}` requires finite bounds",
                            self.variables[*operand_var].name
                        )));
                    }
                }
                GeneralConstraint::Max {
                    result_var,
                    operands,
                    ..
                } => self.validate_extreme_general_args("max", *result_var, operands)?,
                GeneralConstraint::Min {
                    result_var,
                    operands,
                    ..
                } => self.validate_extreme_general_args("min", *result_var, operands)?,
                GeneralConstraint::PiecewiseLinear {
                    x_var,
                    y_var,
                    points,
                    ..
                } => self.validate_piecewise_linear_args(*x_var, *y_var, points)?,
                GeneralConstraint::AllDifferent { variables, .. } => {
                    self.validate_all_different_args(variables)?
                }
                GeneralConstraint::AllowedAssignments {
                    variables, tuples, ..
                } => self.validate_allowed_assignments_args(variables, tuples)?,
                GeneralConstraint::NoOverlap { intervals, .. } => {
                    self.validate_interval_args("no-overlap", intervals)?
                }
                GeneralConstraint::NoOverlap2D {
                    x_intervals,
                    y_intervals,
                    ..
                } => self.validate_no_overlap_2d_args(x_intervals, y_intervals)?,
                GeneralConstraint::Cumulative {
                    intervals,
                    demands,
                    capacity,
                    ..
                } => self.validate_cumulative_args(intervals, demands, *capacity)?,
            }
        }
        Ok(())
    }

    fn validate_no_overlap_2d_args(
        &self,
        x_intervals: &[IntervalTerm],
        y_intervals: &[IntervalTerm],
    ) -> Result<(), MathProgramError> {
        self.validate_interval_args("no-overlap-2d x", x_intervals)?;
        self.validate_interval_args("no-overlap-2d y", y_intervals)?;
        if x_intervals.len() != y_intervals.len() {
            return Err(MathProgramError::Unsupported(format!(
                "no-overlap-2d requires one x interval and one y interval per rectangle, got {} x and {} y",
                x_intervals.len(),
                y_intervals.len()
            )));
        }
        Ok(())
    }

    fn validate_all_different_args(&self, variables: &[usize]) -> Result<(), MathProgramError> {
        if variables.len() < 2 {
            return Err(MathProgramError::Unsupported(
                "all-different requires at least two variables".to_string(),
            ));
        }
        let mut literal_count = 0usize;
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "all-different variable index {idx} out of bounds"
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "all-different variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            let (lower, upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "all-different variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
            let domain_size = upper
                .checked_sub(lower)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!(
                        "all-different variable `{}` has an oversized domain",
                        self.variables[idx].name
                    ))
                })?;
            literal_count = literal_count
                .checked_add(domain_size as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported(
                        "all-different assignment literal count overflowed".to_string(),
                    )
                })?;
            if literal_count > 512 {
                return Err(MathProgramError::Unsupported(format!(
                    "all-different exact MIP lowering is limited to 512 value literals, got {literal_count}"
                )));
            }
        }
        Ok(())
    }

    fn validate_allowed_assignments_args(
        &self,
        variables: &[usize],
        tuples: &[Vec<i64>],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "allowed-assignments requires at least one variable".to_string(),
            ));
        }
        if tuples.is_empty() {
            return Err(MathProgramError::Unsupported(
                "allowed-assignments requires at least one tuple".to_string(),
            ));
        }
        if tuples.len() > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "allowed-assignments exact MIP lowering is limited to 512 tuples, got {}",
                tuples.len()
            )));
        }

        let mut bounds = Vec::with_capacity(variables.len());
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "allowed-assignments variable index {idx} out of bounds"
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "allowed-assignments variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            bounds.push(integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "allowed-assignments variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?);
        }

        for (row, tuple) in tuples.iter().enumerate() {
            if tuple.len() != variables.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "allowed-assignments tuple {row} has length {}, expected {}",
                    tuple.len(),
                    variables.len()
                )));
            }
            for (col, &value) in tuple.iter().enumerate() {
                let (lower, upper) = bounds[col];
                if value < lower || value > upper {
                    return Err(MathProgramError::InvalidBound(format!(
                        "allowed-assignments tuple {row} value {value} is outside bounds [{lower}, {upper}] for `{}`",
                        self.variables[variables[col]].name
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_quadratic_constraint_args(
        &self,
        quadratic_terms: &[QuadraticConstraintTerm],
        linear_terms: &[(usize, f64)],
        sense: RowSense,
        rhs: f64,
    ) -> Result<(), MathProgramError> {
        if quadratic_terms.is_empty() {
            return Err(MathProgramError::Unsupported(
                "quadratic constraint requires at least one quadratic term".to_string(),
            ));
        }
        if sense == RowSense::Eq {
            return Err(MathProgramError::Unsupported(
                "quadratic equality constraints are not convex in the native QCP solver"
                    .to_string(),
            ));
        }
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite(
                "quadratic constraint rhs".to_string(),
            ));
        }
        validate_coeffs(self.variables.len(), linear_terms)?;
        for term in quadratic_terms {
            if term.var_i >= self.variables.len() || term.var_j >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "quadratic constraint references {} and {} with {} variables",
                    term.var_i,
                    term.var_j,
                    self.variables.len()
                )));
            }
            if !term.coeff.is_finite() {
                return Err(MathProgramError::NonFinite(
                    "quadratic constraint coefficient".to_string(),
                ));
            }
            if !supports_native_nonlinear_var(self.variables[term.var_i].var_type)
                || !supports_native_nonlinear_var(self.variables[term.var_j].var_type)
            {
                return Err(MathProgramError::Unsupported(format!(
                    "quadratic constraint term `{}` * `{}` requires continuous, integer, or binary variables",
                    self.variables[term.var_i].name, self.variables[term.var_j].name
                )));
            }
        }
        for &(idx, _) in linear_terms {
            if !supports_native_nonlinear_var(self.variables[idx].var_type) {
                return Err(MathProgramError::Unsupported(format!(
                    "quadratic constraint linear variable `{}` must be continuous, integer, or binary",
                    self.variables[idx].name
                )));
            }
        }

        let sign = match sense {
            RowSense::Le => 1.0,
            RowSense::Ge => -1.0,
            RowSense::Eq => unreachable!("quadratic equality rejected above"),
        };
        let min_eig = min_symmetric_eigenvalue(scale_matrix(
            &quadratic_terms_hessian(self.variables.len(), quadratic_terms),
            sign,
        ));
        if min_eig < -1e-8 {
            return Err(MathProgramError::Unsupported(format!(
                "quadratic constraint requires a convex <= row or concave >= row; transformed minimum eigenvalue is {min_eig:.3e}"
            )));
        }

        Ok(())
    }

    fn validate_second_order_cone_args(
        &self,
        terms: &[AffineTerm],
        rhs_coeffs: &[(usize, f64)],
        rhs_constant: f64,
    ) -> Result<(), MathProgramError> {
        if terms.is_empty() {
            return Err(MathProgramError::Unsupported(
                "second-order cone requires at least one norm term".to_string(),
            ));
        }
        if !rhs_constant.is_finite() {
            return Err(MathProgramError::NonFinite(
                "second-order cone rhs constant".to_string(),
            ));
        }
        validate_coeffs(self.variables.len(), rhs_coeffs)?;
        for (i, term) in terms.iter().enumerate() {
            if !term.constant.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "second-order cone term {i} constant"
                )));
            }
            validate_coeffs(self.variables.len(), &term.coeffs)?;
        }
        for &(idx, _) in rhs_coeffs {
            if !supports_native_nonlinear_var(self.variables[idx].var_type) {
                return Err(MathProgramError::Unsupported(format!(
                    "second-order cone rhs variable `{}` must be continuous, integer, or binary",
                    self.variables[idx].name
                )));
            }
        }
        for term in terms {
            for &(idx, _) in &term.coeffs {
                if !supports_native_nonlinear_var(self.variables[idx].var_type) {
                    return Err(MathProgramError::Unsupported(format!(
                        "second-order cone norm variable `{}` must be continuous, integer, or binary",
                        self.variables[idx].name
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_quadratic_objective_term(
        &self,
        var_i: usize,
        var_j: usize,
        coeff: f64,
    ) -> Result<(), MathProgramError> {
        if var_i >= self.variables.len() || var_j >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "quadratic objective references {var_i} and {var_j} with {} variables",
                self.variables.len()
            )));
        }
        if !coeff.is_finite() {
            return Err(MathProgramError::NonFinite(
                "quadratic objective coefficient".to_string(),
            ));
        }
        let left_type = self.variables[var_i].var_type;
        let right_type = self.variables[var_j].var_type;
        if !supports_native_nonlinear_var(left_type) || !supports_native_nonlinear_var(right_type) {
            return Err(MathProgramError::Unsupported(format!(
                "quadratic objective term `{}` * `{}` requires continuous, integer, or binary variables",
                self.variables[var_i].name, self.variables[var_j].name
            )));
        }
        Ok(())
    }

    fn validate_binary_general_args(
        &self,
        result_var: usize,
        operands: &[usize],
    ) -> Result<(), MathProgramError> {
        if result_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "general constraint result index {result_var} out of bounds"
            )));
        }
        if operands.is_empty() {
            return Err(MathProgramError::Unsupported(
                "binary general constraints require at least one operand".to_string(),
            ));
        }
        if self.variables[result_var].var_type != VariableType::Binary {
            return Err(MathProgramError::Unsupported(format!(
                "general constraint result `{}` must be binary",
                self.variables[result_var].name
            )));
        }
        for &operand in operands {
            if operand >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "general constraint operand index {operand} out of bounds"
                )));
            }
            if self.variables[operand].var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "general constraint operand `{}` must be binary",
                    self.variables[operand].name
                )));
            }
        }
        Ok(())
    }

    fn validate_extreme_general_args(
        &self,
        kind: &str,
        result_var: usize,
        operands: &[usize],
    ) -> Result<(), MathProgramError> {
        if result_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{kind} result index {result_var} out of bounds"
            )));
        }
        if operands.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} constraint requires at least one operand"
            )));
        }
        if variable_bounds(&self.variables[result_var]).is_none() {
            return Err(MathProgramError::UnboundedBigM(format!(
                "{kind} result `{}` requires finite bounds",
                self.variables[result_var].name
            )));
        }
        for &operand in operands {
            if operand >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "{kind} operand index {operand} out of bounds"
                )));
            }
            if variable_bounds(&self.variables[operand]).is_none() {
                return Err(MathProgramError::UnboundedBigM(format!(
                    "{kind} operand `{}` requires finite bounds",
                    self.variables[operand].name
                )));
            }
        }
        Ok(())
    }

    fn validate_piecewise_linear_args(
        &self,
        x_var: usize,
        y_var: usize,
        points: &[(f64, f64)],
    ) -> Result<(), MathProgramError> {
        if x_var >= self.variables.len() || y_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "piecewise-linear references x index {x_var} and y index {y_var} with {} variables",
                self.variables.len()
            )));
        }
        if points.len() < 2 {
            return Err(MathProgramError::Unsupported(
                "piecewise-linear constraint requires at least two points".to_string(),
            ));
        }
        for &(x, y) in points {
            if !x.is_finite() || !y.is_finite() {
                return Err(MathProgramError::NonFinite(
                    "piecewise-linear point".to_string(),
                ));
            }
        }
        for pair in points.windows(2) {
            if pair[1].0 <= pair[0].0 {
                return Err(MathProgramError::Unsupported(
                    "piecewise-linear x breakpoints must be strictly increasing".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn validate_interval_args(
        &self,
        kind: &str,
        intervals: &[IntervalTerm],
    ) -> Result<(), MathProgramError> {
        if intervals.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} requires at least one interval"
            )));
        }
        for (i, interval) in intervals.iter().enumerate() {
            if interval.start_var >= self.variables.len()
                || interval.end_var >= self.variables.len()
            {
                return Err(MathProgramError::BadIndex(format!(
                    "{kind} interval {i} references start {} and end {} with {} variables",
                    interval.start_var,
                    interval.end_var,
                    self.variables.len()
                )));
            }
            if !interval.duration.is_finite() || interval.duration < 0.0 {
                return Err(MathProgramError::InvalidBound(format!(
                    "{kind} interval {i} has invalid duration {}",
                    interval.duration
                )));
            }
            if variable_bounds(&self.variables[interval.start_var]).is_none()
                || variable_bounds(&self.variables[interval.end_var]).is_none()
            {
                return Err(MathProgramError::UnboundedBigM(format!(
                    "{kind} interval {i} requires finite start/end bounds"
                )));
            }
            if let Some(presence) = interval.presence_var {
                if presence >= self.variables.len() {
                    return Err(MathProgramError::BadIndex(format!(
                        "{kind} interval {i} presence index {presence} out of bounds"
                    )));
                }
                if self.variables[presence].var_type != VariableType::Binary {
                    return Err(MathProgramError::Unsupported(format!(
                        "{kind} interval {i} presence `{}` must be binary",
                        self.variables[presence].name
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_cumulative_args(
        &self,
        intervals: &[IntervalTerm],
        demands: &[f64],
        capacity: f64,
    ) -> Result<(), MathProgramError> {
        self.validate_interval_args("cumulative", intervals)?;
        if intervals.len() != demands.len() {
            return Err(MathProgramError::Unsupported(format!(
                "cumulative requires one demand per interval, got {} intervals and {} demands",
                intervals.len(),
                demands.len()
            )));
        }
        if !capacity.is_finite() || capacity < 0.0 {
            return Err(MathProgramError::InvalidBound(format!(
                "cumulative capacity must be finite and non-negative, got {capacity}"
            )));
        }
        for (i, (&demand, interval)) in demands.iter().zip(intervals).enumerate() {
            if !demand.is_finite() || demand < 0.0 {
                return Err(MathProgramError::InvalidBound(format!(
                    "cumulative demand {i} must be finite and non-negative, got {demand}"
                )));
            }
            if !is_integer_time_var(&self.variables[interval.start_var])
                || !is_integer_time_var(&self.variables[interval.end_var])
            {
                return Err(MathProgramError::Unsupported(format!(
                    "cumulative interval {i} start/end variables must be integer-time variables"
                )));
            }
            if !is_integer_value(interval.duration) {
                return Err(MathProgramError::Unsupported(format!(
                    "cumulative interval {i} duration {} must be an integer",
                    interval.duration
                )));
            }
        }
        Ok(())
    }
}

/// Which LP backend to use for pure continuous models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProgramLpBackend {
    InternalSimplex,
    InternalInteriorPoint,
    ScipyHighs,
    ScipyHighsDs,
    ScipyHighsIpm,
}

#[derive(Clone, Debug)]
pub struct MathProgramQpOptions {
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for MathProgramQpOptions {
    fn default() -> Self {
        MathProgramQpOptions {
            max_iter: 1_000,
            tolerance: 1e-8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MathProgramConicOptions {
    pub max_cuts: usize,
    pub tolerance: f64,
}

impl Default for MathProgramConicOptions {
    fn default() -> Self {
        MathProgramConicOptions {
            max_cuts: 128,
            tolerance: 1e-7,
        }
    }
}

/// Solve options for the facade.
#[derive(Clone, Debug)]
pub struct MathProgramSolveOptions {
    pub lp_backend: MathProgramLpBackend,
    pub lp_simplex: InternalSimplexOptions,
    pub lp_ipm: InternalInteriorPointOptions,
    pub external_lp: ExternalSolverOptions,
    pub qp: MathProgramQpOptions,
    pub conic: MathProgramConicOptions,
    pub mip: IPMIPSolveOptions,
    /// Optional MIP start in the original math-program variable space.
    pub mip_start: Option<Vec<f64>>,
}

impl Default for MathProgramSolveOptions {
    fn default() -> Self {
        MathProgramSolveOptions {
            lp_backend: MathProgramLpBackend::InternalSimplex,
            lp_simplex: InternalSimplexOptions::default(),
            lp_ipm: InternalInteriorPointOptions::default(),
            external_lp: ExternalSolverOptions::default(),
            qp: MathProgramQpOptions::default(),
            conic: MathProgramConicOptions::default(),
            mip: IPMIPSolveOptions::default(),
            mip_start: None,
        }
    }
}

/// Options for the optional Python external-solver oracle.
#[derive(Clone, Debug, Default)]
pub struct ExternalMathProgramOptions {
    /// Solver method, such as `highs`, `ortools:SCIP`, `glpk:default`,
    /// `gurobi:default`, `cplex:default`, or `xpress:default`.
    pub method: Option<String>,
    /// Python executable. Defaults to `PYTHON` or `python3`.
    pub python: Option<String>,
    /// Script path. Defaults to `external-references/math-program/math_program_solve.py`.
    pub script: Option<String>,
    /// Optional MIP start in the original math-program variable space.
    pub mip_start: Option<Vec<f64>>,
}

/// Facade status normalized across LP and IP/MIP solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProgramStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterLimit,
    NodeLimit,
    TimeLimit,
    NumericalError,
}

/// Solution mapped back to the original variables.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramSolution {
    pub status: MathProgramStatus,
    pub x: Vec<f64>,
    pub objective: f64,
    pub solver: String,
    pub message: Option<String>,
}

/// Constraint or bound retained in a refined infeasible subsystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MathProgramConflictItem {
    VariableLowerBound { var: usize, name: String },
    VariableUpperBound { var: usize, name: String },
    LinearConstraint { index: usize, name: String },
}

/// Options for native deletion-filter conflict refinement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathProgramConflictOptions {
    pub max_candidate_checks: usize,
}

impl Default for MathProgramConflictOptions {
    fn default() -> Self {
        MathProgramConflictOptions {
            max_candidate_checks: usize::MAX,
        }
    }
}

/// IIS-like infeasibility report for linear rows and removable variable bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramConflict {
    pub status: MathProgramStatus,
    pub items: Vec<MathProgramConflictItem>,
    pub subsystem: MathProgram,
    pub minimal: bool,
    pub solver: String,
    pub message: Option<String>,
}

/// Internal-vs-external check for the refined conflict subsystem.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramConflictCrossCheck {
    pub internal: MathProgramConflict,
    pub external: MathProgramSolution,
    pub status_agree: bool,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramFeasRelaxOptions {
    pub relax_linear_constraints: bool,
    pub relax_variable_bounds: bool,
    pub linear_penalty: f64,
    pub bound_penalty: f64,
    pub violation_tolerance: f64,
}

impl Default for MathProgramFeasRelaxOptions {
    fn default() -> Self {
        MathProgramFeasRelaxOptions {
            relax_linear_constraints: true,
            relax_variable_bounds: true,
            linear_penalty: 1.0,
            bound_penalty: 1.0,
            violation_tolerance: 1e-7,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MathProgramFeasRelaxViolation {
    VariableLowerBound {
        var: usize,
        name: String,
        violation: f64,
        penalty: f64,
    },
    VariableUpperBound {
        var: usize,
        name: String,
        violation: f64,
        penalty: f64,
    },
    LinearConstraint {
        index: usize,
        name: String,
        violation: f64,
        penalty: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramFeasRelaxation {
    pub status: MathProgramStatus,
    pub x: Vec<f64>,
    pub violation_objective: f64,
    pub violations: Vec<MathProgramFeasRelaxViolation>,
    pub relaxed_program: MathProgram,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramFeasRelaxCrossCheck {
    pub internal: MathProgramFeasRelaxation,
    pub external: MathProgramSolution,
    pub status_agree: bool,
    pub objective_abs_diff: Option<f64>,
    pub within_tolerance: bool,
}

/// Internal-vs-external comparison for the same model input.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramCrossCheck {
    pub internal: MathProgramSolution,
    pub external: MathProgramSolution,
    pub status_agree: bool,
    pub objective_abs_diff: Option<f64>,
    pub max_x_abs_diff: Option<f64>,
    pub internal_max_violation: Option<f64>,
    pub external_max_violation: Option<f64>,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramSolutionPoolOptions {
    pub max_solutions: usize,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub max_discrete_domain_size: usize,
}

impl Default for MathProgramSolutionPoolOptions {
    fn default() -> Self {
        MathProgramSolutionPoolOptions {
            max_solutions: 10,
            absolute_gap: None,
            relative_gap: None,
            max_discrete_domain_size: 256,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramSolutionPool {
    pub solutions: Vec<MathProgramSolution>,
    pub exhausted: bool,
    pub solver: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramSolutionPoolCrossCheck {
    pub internal: MathProgramSolutionPool,
    pub external: MathProgramSolutionPool,
    pub len_agree: bool,
    pub objective_abs_diffs: Vec<Option<f64>>,
    pub max_x_abs_diffs: Vec<Option<f64>>,
    pub internal_max_violations: Vec<Option<f64>>,
    pub external_max_violations: Vec<Option<f64>>,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MathProgramError {
    EmptyModel,
    BadIndex(String),
    NonFinite(String),
    InvalidBound(String),
    Unsupported(String),
    UnboundedBigM(String),
    External(String),
}

/// Solve a continuous or mixed-integer math program with the native solvers.
pub fn solve_math_program(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty() {
        return solve_hierarchical_math_program(program, opts);
    }
    solve_math_program_single_objective(program, opts)
}

fn solve_math_program_single_objective(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    if !program.has_discrete_features() {
        if program.has_conic_constraints() || program.has_quadratic_constraints() {
            return solve_continuous_conic(program, opts);
        }
        if program.has_quadratic_objective() {
            return solve_continuous_qp(program, opts);
        }
        let lp = program.to_lp_problem()?;
        return Ok(from_lp_solution(solve_lp_with_backend(&lp, opts)));
    }

    if program.has_quadratic_objective() && quadratic_objective_has_native_nonlinear_terms(program)
    {
        return solve_mixed_integer_quadratic_objective(program, opts);
    }
    if program.has_conic_constraints() || program.has_quadratic_constraints() {
        return solve_mixed_integer_conic(program, opts);
    }

    solve_mixed_integer_linear(program, opts)
}

#[derive(Clone, Debug)]
struct ObjectiveStage {
    name: String,
    coeffs: Vec<(usize, f64)>,
    members: Vec<LinearObjective>,
    abs_tol: f64,
    rel_tol: f64,
}

fn solve_hierarchical_math_program(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    solve_hierarchical_math_program_with(program, "des-hierarchical", |stage_program| {
        solve_math_program_single_objective(stage_program, opts)
    })
}

fn solve_hierarchical_math_program_external(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    solve_hierarchical_math_program_with(program, "external-hierarchical", |stage_program| {
        solve_math_program_external_scipy_single_objective(stage_program, opts)
    })
}

fn solve_hierarchical_math_program_with<F>(
    program: &MathProgram,
    solver_prefix: &str,
    mut solve_stage: F,
) -> Result<MathProgramSolution, MathProgramError>
where
    F: FnMut(&MathProgram) -> Result<MathProgramSolution, MathProgramError>,
{
    if program.has_quadratic_objective() {
        return Err(MathProgramError::Unsupported(
            "hierarchical multi-objective solve currently supports linear objectives only"
                .to_string(),
        ));
    }

    let stages = objective_stages(program)?;
    let mut working = program.clone();
    working.secondary_objectives.clear();
    let mut final_solution = None;
    let mut objective_report = Vec::new();

    for (stage_idx, stage) in stages.iter().enumerate() {
        set_stage_objective(&mut working, stage);
        let solution = solve_stage(&working)?;
        if solution.status != MathProgramStatus::Optimal {
            return Ok(solution);
        }

        let stage_score = eval_sparse_affine(&stage.coeffs, 0.0, &solution.x);
        for member in &stage.members {
            let value = eval_sparse_affine(&member.coeffs, 0.0, &solution.x);
            objective_report.push(format!("{}={:.8}", member.name, value));
        }
        add_stage_lock(&mut working, stage_idx, stage, stage_score)?;
        final_solution = Some(solution);
    }

    let mut solution = final_solution.ok_or_else(|| {
        MathProgramError::Unsupported(
            "hierarchical solve requires at least one objective".to_string(),
        )
    })?;
    solution.objective = objective_value(program, &solution.x);
    solution.solver = format!("{solver_prefix}({})", solution.solver);
    solution.message = Some(match solution.message {
        Some(message) => format!(
            "{message}; hierarchical objectives: {}",
            objective_report.join(", ")
        ),
        None => format!("hierarchical objectives: {}", objective_report.join(", ")),
    });
    Ok(solution)
}

fn objective_stages(program: &MathProgram) -> Result<Vec<ObjectiveStage>, MathProgramError> {
    let mut grouped = BTreeMap::<i32, Vec<LinearObjective>>::new();
    grouped.entry(i32::MAX).or_default().push(LinearObjective {
        name: "primary".to_string(),
        sense: program.sense,
        priority: i32::MAX,
        weight: 1.0,
        abs_tol: 1e-7,
        rel_tol: 1e-9,
        coeffs: primary_linear_objective(program),
    });
    for objective in &program.secondary_objectives {
        grouped
            .entry(objective.priority)
            .or_default()
            .push(objective.clone());
    }

    grouped
        .into_iter()
        .rev()
        .map(|(priority, members)| objective_stage(priority, members))
        .collect()
}

fn objective_stage(
    priority: i32,
    members: Vec<LinearObjective>,
) -> Result<ObjectiveStage, MathProgramError> {
    let mut coeffs = Vec::new();
    let mut abs_tol: f64 = 0.0;
    let mut rel_tol: f64 = 0.0;
    for member in &members {
        let direction = match member.sense {
            ObjectiveSense::Max => 1.0,
            ObjectiveSense::Min => -1.0,
        };
        let scale = member.weight * direction;
        coeffs.extend(member.coeffs.iter().map(|&(idx, coef)| (idx, scale * coef)));
        abs_tol = abs_tol.max(member.abs_tol);
        rel_tol = rel_tol.max(member.rel_tol);
    }
    Ok(ObjectiveStage {
        name: format!("priority_{priority}"),
        coeffs: combine_terms(&coeffs),
        members,
        abs_tol,
        rel_tol,
    })
}

fn primary_linear_objective(program: &MathProgram) -> Vec<(usize, f64)> {
    program
        .variables
        .iter()
        .enumerate()
        .filter_map(|(idx, var)| (var.obj.abs() > 1e-12).then_some((idx, var.obj)))
        .collect()
}

fn set_stage_objective(program: &mut MathProgram, stage: &ObjectiveStage) {
    program.sense = ObjectiveSense::Max;
    for var in &mut program.variables {
        var.obj = 0.0;
    }
    for &(idx, coef) in &stage.coeffs {
        program.variables[idx].obj += coef;
    }
}

fn add_stage_lock(
    program: &mut MathProgram,
    stage_idx: usize,
    stage: &ObjectiveStage,
    stage_score: f64,
) -> Result<(), MathProgramError> {
    if stage.coeffs.is_empty() {
        return Ok(());
    }
    let tolerance = stage.abs_tol + stage.rel_tol * stage_score.abs();
    program.add_constraint(
        format!("__multi_objective_stage_{stage_idx}_{}_lock", stage.name),
        stage.coeffs.clone(),
        RowSense::Ge,
        stage_score - tolerance,
    )?;
    Ok(())
}

fn solve_mixed_integer_linear(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    let compiled = compile_mip(program)?;
    let mut mip_opts = opts.mip.clone();
    if let Some(start) = &opts.mip_start {
        mip_opts.mip_start = Some(canonical_mip_start(program, &compiled, start)?);
    }
    let mip = solve_ipmip_with_des(compiled.problem.clone(), mip_opts);
    let x = compiled.original_x(&mip.x);
    let objective = objective_value(program, &x);
    let incumbent_source = mip
        .incumbent_source
        .as_deref()
        .map(|source| format!(", incumbent_source={source}"))
        .unwrap_or_default();
    Ok(MathProgramSolution {
        status: from_ipmip_status(mip.status),
        x,
        objective,
        solver: "des-ipmip".to_string(),
        message: Some(format!(
            "nodes={}, gap={:.3e}, lp_solves={}{}",
            mip.nodes_explored, mip.gap, mip.performance.lp_solves_per_second, incumbent_source
        )),
    })
}

fn solve_mixed_integer_quadratic_objective(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    let original_vars = program.variables.len();
    let transformed = quadratic_objective_epigraph_program(program)?;
    let mut solution = solve_mixed_integer_conic(&transformed, opts)?;
    if solution.x.len() >= original_vars {
        solution.x.truncate(original_vars);
        solution.objective = objective_value(program, &solution.x);
    }
    solution.solver = match program.sense {
        ObjectiveSense::Min => "des-mip-convex-qp-cutting-plane".to_string(),
        ObjectiveSense::Max => "des-mip-concave-qp-cutting-plane".to_string(),
    };
    solution.message = Some(match solution.message {
        Some(message) => format!("quadratic objective epigraph; {message}"),
        None => "quadratic objective epigraph".to_string(),
    });
    Ok(solution)
}

fn solve_mixed_integer_conic(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    let mut relaxation = program.clone();
    relaxation.second_order_cones.clear();
    relaxation.quadratic_constraints.clear();
    for (i, cone) in program.second_order_cones.iter().enumerate() {
        let nonnegative_rhs = cone
            .rhs_coeffs
            .iter()
            .map(|&(idx, coef)| (idx, -coef))
            .collect::<Vec<_>>();
        relaxation.add_constraint(
            format!("{}__rhs_nonnegative_{i}", cone.name),
            nonnegative_rhs,
            RowSense::Le,
            cone.rhs_constant,
        )?;
    }

    let solver_name = if program.has_quadratic_constraints() {
        "des-mip-convex-cutting-plane"
    } else {
        "des-mip-soc-cutting-plane"
    };
    let mut compiled = compile_mip(&relaxation)?;
    let mut best = None;
    for cut in 0..=opts.conic.max_cuts {
        let mip = solve_ipmip_with_des(compiled.problem.clone(), opts.mip.clone());
        let x = compiled.original_x(&mip.x);
        let objective = objective_value(program, &x);
        let mut solution = MathProgramSolution {
            status: from_ipmip_status(mip.status),
            x,
            objective,
            solver: solver_name.to_string(),
            message: Some(format!(
                "cuts={}, nodes={}, gap={:.3e}, lp_solves={}",
                cut, mip.nodes_explored, mip.gap, mip.performance.lp_solves_per_second
            )),
        };
        if solution.status != MathProgramStatus::Optimal {
            return Ok(solution);
        }

        let soc_violation = most_violated_soc(program, &solution.x);
        let qcp_violation = most_violated_quadratic_constraint(program, &solution.x);
        let soc_amount = soc_violation
            .as_ref()
            .map_or(0.0, |(_, amount, _, _)| *amount);
        let qcp_amount = qcp_violation.as_ref().map_or(0.0, |(_, amount)| *amount);
        if soc_amount <= opts.conic.tolerance && qcp_amount <= opts.conic.tolerance {
            solution.message = Some(format!(
                "cuts={}, tolerance={:.3e}",
                cut, opts.conic.tolerance
            ));
            return Ok(solution);
        }

        let cut_x = solution.x.clone();
        best = Some(solution);
        if qcp_amount >= soc_amount {
            let (row_idx, _) = qcp_violation.unwrap();
            let row = &program.quadratic_constraints[row_idx];
            let (coeffs, rhs) = quadratic_constraint_supporting_cut(row, &cut_x);
            add_compiled_mip_cut(
                &mut compiled,
                format!("{}__qcp_cut_{cut}", row.name),
                coeffs,
                RowSense::Le,
                rhs,
            );
        } else {
            let (cone_idx, _, norm_values, rhs_value) = soc_violation.unwrap();
            let cone = &program.second_order_cones[cone_idx];
            let norm = norm2(&norm_values);
            if norm <= 1e-12 {
                return Err(MathProgramError::Unsupported(format!(
                    "second-order cone `{}` is violated with near-zero norm and rhs {rhs_value}",
                    cone.name
                )));
            }
            let unit = norm_values
                .iter()
                .map(|value| value / norm)
                .collect::<Vec<_>>();
            let (coeffs, rhs) = soc_supporting_cut(cone, &unit);
            add_compiled_mip_cut(
                &mut compiled,
                format!("{}__soc_cut_{cut}", cone.name),
                coeffs,
                RowSense::Le,
                rhs,
            );
        }
    }

    let mut solution = best.unwrap_or(MathProgramSolution {
        status: MathProgramStatus::NumericalError,
        x: Vec::new(),
        objective: f64::NAN,
        solver: solver_name.to_string(),
        message: Some("no relaxation solve was attempted".to_string()),
    });
    solution.status = MathProgramStatus::IterLimit;
    solution.solver = solver_name.to_string();
    solution.message = Some(format!(
        "max_cuts={}, tolerance={:.3e}",
        opts.conic.max_cuts, opts.conic.tolerance
    ));
    Ok(solution)
}

/// Solve the same model internally and with an optional Python external oracle.
///
/// If Python or the requested solver are unavailable, `external.status` is `NumericalError` and
/// `within_tolerance` is false. The internal solve is still returned so callers
/// can keep validation runs non-fatal on machines without the external stack.
pub fn cross_check_math_program_with_external(
    program: &MathProgram,
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    tol: f64,
) -> Result<MathProgramCrossCheck, MathProgramError> {
    let internal = solve_math_program(program, internal_opts)?;
    let external = solve_math_program_external_scipy(program, external_opts)?;
    Ok(compare_math_program_solutions(
        program, internal, external, tol,
    ))
}

pub fn refine_math_program_conflict(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    conflict_opts: &MathProgramConflictOptions,
) -> Result<MathProgramConflict, MathProgramError> {
    program.validate()?;
    validate_conflict_refinement_scope(program)?;

    let candidates = conflict_candidates(program);
    let mut active = vec![true; candidates.len()];
    let initial_subsystem = build_conflict_subsystem(program, &candidates, &active);
    let initial = solve_conflict_feasibility(&initial_subsystem, solve_opts)?;
    if initial.status != MathProgramStatus::Infeasible {
        return Ok(MathProgramConflict {
            status: initial.status,
            items: Vec::new(),
            subsystem: initial_subsystem,
            minimal: false,
            solver: "des-conflict-refiner".to_string(),
            message: Some(
                "model is not infeasible under zero-objective feasibility probing".to_string(),
            ),
        });
    }

    let mut subsystem = initial_subsystem;
    let mut checks = 0usize;
    let mut limited = false;
    for i in 0..candidates.len() {
        if checks >= conflict_opts.max_candidate_checks {
            limited = true;
            break;
        }
        active[i] = false;
        let trial = build_conflict_subsystem(program, &candidates, &active);
        let trial_solution = solve_conflict_feasibility(&trial, solve_opts)?;
        checks += 1;
        if trial_solution.status == MathProgramStatus::Infeasible {
            subsystem = trial;
        } else {
            active[i] = true;
        }
    }

    let items = active_conflict_items(program, &candidates, &active);
    Ok(MathProgramConflict {
        status: MathProgramStatus::Infeasible,
        items,
        subsystem,
        minimal: !limited,
        solver: "des-conflict-refiner".to_string(),
        message: Some(format!(
            "candidate_items={}, checks={}, minimal={}",
            candidates.len(),
            checks,
            !limited
        )),
    })
}

pub fn cross_check_math_program_conflict_with_external(
    program: &MathProgram,
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    conflict_opts: &MathProgramConflictOptions,
) -> Result<MathProgramConflictCrossCheck, MathProgramError> {
    let internal = refine_math_program_conflict(program, internal_opts, conflict_opts)?;
    let external = solve_math_program_external_scipy(&internal.subsystem, external_opts)?;
    let status_agree = internal.status == external.status;
    let within_tolerance = status_agree && internal.status == MathProgramStatus::Infeasible;
    Ok(MathProgramConflictCrossCheck {
        internal,
        external,
        status_agree,
        within_tolerance,
    })
}

pub fn solve_math_program_feas_relaxation(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    relax_opts: &MathProgramFeasRelaxOptions,
) -> Result<MathProgramFeasRelaxation, MathProgramError> {
    program.validate()?;
    validate_feas_relaxation_scope(program)?;
    validate_feas_relaxation_options(relax_opts)?;

    let (relaxed_program, slacks) = build_feas_relaxation_program(program, relax_opts)?;
    let mut relaxed_solve_opts = solve_opts.clone();
    if let Some(start) = &solve_opts.mip_start {
        relaxed_solve_opts.mip_start = Some(extend_feas_relax_mip_start(program, &slacks, start)?);
    }
    let solution = solve_math_program(&relaxed_program, &relaxed_solve_opts)?;
    let original_len = program.variables.len();
    let original_x = if solution.x.len() >= original_len {
        solution.x[..original_len].to_vec()
    } else {
        Vec::new()
    };
    let violations = if solution.status == MathProgramStatus::Optimal {
        feas_relax_violations(
            program,
            &slacks,
            &solution.x,
            relax_opts.violation_tolerance,
        )
    } else {
        Vec::new()
    };

    Ok(MathProgramFeasRelaxation {
        status: solution.status,
        x: original_x,
        violation_objective: solution.objective,
        violations,
        relaxed_program,
        solver: format!("des-feas-relax({})", solution.solver),
        message: solution.message,
    })
}

pub fn cross_check_math_program_feas_relaxation_with_external(
    program: &MathProgram,
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    relax_opts: &MathProgramFeasRelaxOptions,
    tol: f64,
) -> Result<MathProgramFeasRelaxCrossCheck, MathProgramError> {
    let internal = solve_math_program_feas_relaxation(program, internal_opts, relax_opts)?;
    let external = solve_math_program_external_scipy(&internal.relaxed_program, external_opts)?;
    let status_agree = internal.status == external.status;
    let objective_abs_diff = (internal.violation_objective.is_finite()
        && external.objective.is_finite())
    .then_some((internal.violation_objective - external.objective).abs());
    let within_tolerance = status_agree
        && internal.status == MathProgramStatus::Optimal
        && objective_abs_diff.is_some_and(|diff| diff <= tol);
    Ok(MathProgramFeasRelaxCrossCheck {
        internal,
        external,
        status_agree,
        objective_abs_diff,
        within_tolerance,
    })
}

pub fn solve_math_program_solution_pool(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    pool_opts: &MathProgramSolutionPoolOptions,
) -> Result<MathProgramSolutionPool, MathProgramError> {
    solve_math_program_solution_pool_with(program, pool_opts, "des-solution-pool", |candidate| {
        solve_math_program(candidate, solve_opts)
    })
}

pub fn solve_math_program_external_solution_pool(
    program: &MathProgram,
    external_opts: &ExternalMathProgramOptions,
    pool_opts: &MathProgramSolutionPoolOptions,
) -> Result<MathProgramSolutionPool, MathProgramError> {
    solve_math_program_solution_pool_with(
        program,
        pool_opts,
        "external-solution-pool",
        |candidate| solve_math_program_external_scipy(candidate, external_opts),
    )
}

pub fn cross_check_math_program_solution_pool_with_external(
    program: &MathProgram,
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    pool_opts: &MathProgramSolutionPoolOptions,
    tol: f64,
) -> Result<MathProgramSolutionPoolCrossCheck, MathProgramError> {
    let internal = solve_math_program_solution_pool(program, internal_opts, pool_opts)?;
    let external = solve_math_program_external_solution_pool(program, external_opts, pool_opts)?;
    Ok(compare_solution_pools(program, internal, external, tol))
}

fn compare_math_program_solutions(
    program: &MathProgram,
    internal: MathProgramSolution,
    external: MathProgramSolution,
    tol: f64,
) -> MathProgramCrossCheck {
    let status_agree = internal.status == external.status;
    let objective_abs_diff = (internal.objective.is_finite() && external.objective.is_finite())
        .then_some((internal.objective - external.objective).abs());
    let max_x_abs_diff = if internal.x.len() == external.x.len() && !internal.x.is_empty() {
        Some(
            internal
                .x
                .iter()
                .zip(&external.x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max),
        )
    } else {
        None
    };
    let internal_max_violation = solution_max_violation(program, &internal.x, tol);
    let external_max_violation = solution_max_violation(program, &external.x, tol);
    let within_tolerance = status_agree
        && internal.status == MathProgramStatus::Optimal
        && objective_abs_diff.is_some_and(|d| d <= tol)
        && internal_max_violation.is_some_and(|d| d <= tol)
        && external_max_violation.is_some_and(|d| d <= tol);

    MathProgramCrossCheck {
        internal,
        external,
        status_agree,
        objective_abs_diff,
        max_x_abs_diff,
        internal_max_violation,
        external_max_violation,
        within_tolerance,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConflictCandidate {
    VariableLowerBound(usize),
    VariableUpperBound(usize),
    LinearConstraint(usize),
}

fn validate_conflict_refinement_scope(program: &MathProgram) -> Result<(), MathProgramError> {
    if program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
        || !program.secondary_objectives.is_empty()
        || !program.indicators.is_empty()
        || !program.sos.is_empty()
        || !program.general_constraints.is_empty()
    {
        return Err(MathProgramError::Unsupported(
            "conflict refinement currently supports linear rows and variable bounds only"
                .to_string(),
        ));
    }
    Ok(())
}

fn conflict_candidates(program: &MathProgram) -> Vec<ConflictCandidate> {
    let mut candidates = Vec::new();
    for (idx, var) in program.variables.iter().enumerate() {
        if has_removable_conflict_bounds(var.var_type) {
            if var.lb.is_some() {
                candidates.push(ConflictCandidate::VariableLowerBound(idx));
            }
            if var.ub.is_some() {
                candidates.push(ConflictCandidate::VariableUpperBound(idx));
            }
        }
    }
    for idx in 0..program.constraints.len() {
        candidates.push(ConflictCandidate::LinearConstraint(idx));
    }
    candidates
}

fn has_removable_conflict_bounds(var_type: VariableType) -> bool {
    matches!(var_type, VariableType::Continuous | VariableType::Integer)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeasRelaxSlackKind {
    VariableLowerBound(usize),
    VariableUpperBound(usize),
    LinearConstraint(usize),
}

#[derive(Clone, Debug, PartialEq)]
struct FeasRelaxSlack {
    kind: FeasRelaxSlackKind,
    slack_vars: Vec<usize>,
    penalty: f64,
}

fn validate_feas_relaxation_scope(program: &MathProgram) -> Result<(), MathProgramError> {
    if program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
        || !program.secondary_objectives.is_empty()
        || !program.indicators.is_empty()
        || !program.sos.is_empty()
        || !program.general_constraints.is_empty()
    {
        return Err(MathProgramError::Unsupported(
            "feasibility relaxation currently supports linear rows and variable bounds only"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_feas_relaxation_options(
    opts: &MathProgramFeasRelaxOptions,
) -> Result<(), MathProgramError> {
    if !opts.relax_linear_constraints && !opts.relax_variable_bounds {
        return Err(MathProgramError::Unsupported(
            "feasibility relaxation requires at least one relaxation target".to_string(),
        ));
    }
    if !opts.linear_penalty.is_finite() || opts.linear_penalty <= 0.0 {
        return Err(MathProgramError::InvalidBound(
            "linear feasibility-relaxation penalty must be finite and positive".to_string(),
        ));
    }
    if !opts.bound_penalty.is_finite() || opts.bound_penalty <= 0.0 {
        return Err(MathProgramError::InvalidBound(
            "bound feasibility-relaxation penalty must be finite and positive".to_string(),
        ));
    }
    if !opts.violation_tolerance.is_finite() || opts.violation_tolerance < 0.0 {
        return Err(MathProgramError::InvalidBound(
            "feasibility-relaxation violation tolerance must be finite and non-negative"
                .to_string(),
        ));
    }
    Ok(())
}

fn build_feas_relaxation_program(
    program: &MathProgram,
    opts: &MathProgramFeasRelaxOptions,
) -> Result<(MathProgram, Vec<FeasRelaxSlack>), MathProgramError> {
    let mut relaxed = MathProgram::new(ObjectiveSense::Min);
    let mut slacks = Vec::new();

    for var in &program.variables {
        let mut relaxed_var = var.clone();
        relaxed_var.obj = 0.0;
        if opts.relax_variable_bounds && has_removable_conflict_bounds(relaxed_var.var_type) {
            relaxed_var.lb = None;
            relaxed_var.ub = None;
        }
        validate_variable(&relaxed_var)?;
        relaxed.variables.push(relaxed_var);
    }

    if opts.relax_variable_bounds {
        add_feas_relax_bound_rows(program, &mut relaxed, opts.bound_penalty, &mut slacks)?;
    }

    for (idx, row) in program.constraints.iter().enumerate() {
        if opts.relax_linear_constraints {
            add_feas_relax_linear_row(idx, row, &mut relaxed, opts.linear_penalty, &mut slacks)?;
        } else {
            relaxed.add_constraint(row.name.clone(), row.coeffs.clone(), row.sense, row.rhs)?;
        }
    }

    Ok((relaxed, slacks))
}

fn add_feas_relax_bound_rows(
    program: &MathProgram,
    relaxed: &mut MathProgram,
    penalty: f64,
    slacks: &mut Vec<FeasRelaxSlack>,
) -> Result<(), MathProgramError> {
    for (idx, var) in program.variables.iter().enumerate() {
        if !has_removable_conflict_bounds(var.var_type) {
            continue;
        }
        if let Some(lb) = var.lb {
            let slack = relaxed.add_continuous_var(
                format!("__feas_relax_lb_{idx}_{}", var.name),
                penalty,
                Some(0.0),
                None,
            )?;
            relaxed.add_constraint(
                format!("__feas_relax_lb_row_{idx}_{}", var.name),
                vec![(idx, 1.0), (slack, 1.0)],
                RowSense::Ge,
                lb,
            )?;
            slacks.push(FeasRelaxSlack {
                kind: FeasRelaxSlackKind::VariableLowerBound(idx),
                slack_vars: vec![slack],
                penalty,
            });
        }
        if let Some(ub) = var.ub {
            let slack = relaxed.add_continuous_var(
                format!("__feas_relax_ub_{idx}_{}", var.name),
                penalty,
                Some(0.0),
                None,
            )?;
            relaxed.add_constraint(
                format!("__feas_relax_ub_row_{idx}_{}", var.name),
                vec![(idx, 1.0), (slack, -1.0)],
                RowSense::Le,
                ub,
            )?;
            slacks.push(FeasRelaxSlack {
                kind: FeasRelaxSlackKind::VariableUpperBound(idx),
                slack_vars: vec![slack],
                penalty,
            });
        }
    }
    Ok(())
}

fn add_feas_relax_linear_row(
    idx: usize,
    row: &LinearConstraint,
    relaxed: &mut MathProgram,
    penalty: f64,
    slacks: &mut Vec<FeasRelaxSlack>,
) -> Result<(), MathProgramError> {
    let mut coeffs = row.coeffs.clone();
    let mut slack_vars = Vec::new();
    match row.sense {
        RowSense::Le => {
            let slack = relaxed.add_continuous_var(
                format!("__feas_relax_row_{idx}_{}", row.name),
                penalty,
                Some(0.0),
                None,
            )?;
            coeffs.push((slack, -1.0));
            slack_vars.push(slack);
        }
        RowSense::Ge => {
            let slack = relaxed.add_continuous_var(
                format!("__feas_relax_row_{idx}_{}", row.name),
                penalty,
                Some(0.0),
                None,
            )?;
            coeffs.push((slack, 1.0));
            slack_vars.push(slack);
        }
        RowSense::Eq => {
            let pos = relaxed.add_continuous_var(
                format!("__feas_relax_row_{idx}_{}_pos", row.name),
                penalty,
                Some(0.0),
                None,
            )?;
            let neg = relaxed.add_continuous_var(
                format!("__feas_relax_row_{idx}_{}_neg", row.name),
                penalty,
                Some(0.0),
                None,
            )?;
            coeffs.push((pos, 1.0));
            coeffs.push((neg, -1.0));
            slack_vars.push(pos);
            slack_vars.push(neg);
        }
    }
    relaxed.add_constraint(row.name.clone(), coeffs, row.sense, row.rhs)?;
    slacks.push(FeasRelaxSlack {
        kind: FeasRelaxSlackKind::LinearConstraint(idx),
        slack_vars,
        penalty,
    });
    Ok(())
}

fn feas_relax_violations(
    program: &MathProgram,
    slacks: &[FeasRelaxSlack],
    x: &[f64],
    tol: f64,
) -> Vec<MathProgramFeasRelaxViolation> {
    let mut violations = Vec::new();
    for slack in slacks {
        let violation = slack
            .slack_vars
            .iter()
            .map(|&idx| x.get(idx).copied().unwrap_or(0.0).max(0.0))
            .sum::<f64>();
        if violation <= tol {
            continue;
        }
        match slack.kind {
            FeasRelaxSlackKind::VariableLowerBound(var) => {
                violations.push(MathProgramFeasRelaxViolation::VariableLowerBound {
                    var,
                    name: program.variables[var].name.clone(),
                    violation,
                    penalty: slack.penalty,
                });
            }
            FeasRelaxSlackKind::VariableUpperBound(var) => {
                violations.push(MathProgramFeasRelaxViolation::VariableUpperBound {
                    var,
                    name: program.variables[var].name.clone(),
                    violation,
                    penalty: slack.penalty,
                });
            }
            FeasRelaxSlackKind::LinearConstraint(index) => {
                violations.push(MathProgramFeasRelaxViolation::LinearConstraint {
                    index,
                    name: program.constraints[index].name.clone(),
                    violation,
                    penalty: slack.penalty,
                });
            }
        }
    }
    violations
}

fn extend_feas_relax_mip_start(
    program: &MathProgram,
    slacks: &[FeasRelaxSlack],
    start: &[f64],
) -> Result<Vec<f64>, MathProgramError> {
    if start.len() != program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "MIP start length {} does not match {} original variables",
            start.len(),
            program.variables.len()
        )));
    }
    if start.iter().any(|v| !v.is_finite()) {
        return Err(MathProgramError::NonFinite(
            "MIP start contains a non-finite value".to_string(),
        ));
    }

    let relaxed_len = slacks
        .iter()
        .flat_map(|slack| slack.slack_vars.iter().copied())
        .max()
        .map_or(program.variables.len(), |idx| idx + 1);
    let mut relaxed_start = start.to_vec();
    relaxed_start.resize(relaxed_len, 0.0);

    for slack in slacks {
        match slack.kind {
            FeasRelaxSlackKind::VariableLowerBound(var) => {
                let violation = program.variables[var]
                    .lb
                    .map_or(0.0, |lb| (lb - start[var]).max(0.0));
                set_feas_relax_single_slack(&mut relaxed_start, slack, violation);
            }
            FeasRelaxSlackKind::VariableUpperBound(var) => {
                let violation = program.variables[var]
                    .ub
                    .map_or(0.0, |ub| (start[var] - ub).max(0.0));
                set_feas_relax_single_slack(&mut relaxed_start, slack, violation);
            }
            FeasRelaxSlackKind::LinearConstraint(index) => {
                let row = &program.constraints[index];
                let lhs = eval_sparse_affine(&row.coeffs, 0.0, start);
                match row.sense {
                    RowSense::Le => {
                        set_feas_relax_single_slack(
                            &mut relaxed_start,
                            slack,
                            (lhs - row.rhs).max(0.0),
                        );
                    }
                    RowSense::Ge => {
                        set_feas_relax_single_slack(
                            &mut relaxed_start,
                            slack,
                            (row.rhs - lhs).max(0.0),
                        );
                    }
                    RowSense::Eq => {
                        if slack.slack_vars.len() == 2 {
                            let diff = lhs - row.rhs;
                            relaxed_start[slack.slack_vars[0]] = (-diff).max(0.0);
                            relaxed_start[slack.slack_vars[1]] = diff.max(0.0);
                        }
                    }
                }
            }
        }
    }

    Ok(relaxed_start)
}

fn set_feas_relax_single_slack(start: &mut [f64], slack: &FeasRelaxSlack, value: f64) {
    if let Some(&idx) = slack.slack_vars.first() {
        start[idx] = value;
    }
}

fn build_conflict_subsystem(
    program: &MathProgram,
    candidates: &[ConflictCandidate],
    active: &[bool],
) -> MathProgram {
    let mut subsystem = MathProgram::new(program.sense);
    for (idx, var) in program.variables.iter().enumerate() {
        let mut reduced = var.clone();
        reduced.obj = 0.0;
        if has_removable_conflict_bounds(var.var_type) {
            if var.lb.is_some()
                && !conflict_candidate_is_active(
                    candidates,
                    active,
                    ConflictCandidate::VariableLowerBound(idx),
                )
            {
                reduced.lb = None;
            }
            if var.ub.is_some()
                && !conflict_candidate_is_active(
                    candidates,
                    active,
                    ConflictCandidate::VariableUpperBound(idx),
                )
            {
                reduced.ub = None;
            }
        }
        subsystem.variables.push(reduced);
    }
    for (idx, row) in program.constraints.iter().enumerate() {
        if conflict_candidate_is_active(
            candidates,
            active,
            ConflictCandidate::LinearConstraint(idx),
        ) {
            subsystem.constraints.push(row.clone());
        }
    }
    subsystem
}

fn conflict_candidate_is_active(
    candidates: &[ConflictCandidate],
    active: &[bool],
    needle: ConflictCandidate,
) -> bool {
    candidates
        .iter()
        .zip(active)
        .any(|(&candidate, &is_active)| candidate == needle && is_active)
}

fn solve_conflict_feasibility(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    let mut feasibility = program.clone();
    feasibility.sense = ObjectiveSense::Min;
    feasibility.quadratic_objective.clear();
    feasibility.secondary_objectives.clear();
    for var in &mut feasibility.variables {
        var.obj = 0.0;
    }
    solve_math_program(&feasibility, opts)
}

fn active_conflict_items(
    program: &MathProgram,
    candidates: &[ConflictCandidate],
    active: &[bool],
) -> Vec<MathProgramConflictItem> {
    candidates
        .iter()
        .zip(active)
        .filter_map(|(&candidate, &is_active)| is_active.then(|| conflict_item(program, candidate)))
        .collect()
}

fn conflict_item(program: &MathProgram, candidate: ConflictCandidate) -> MathProgramConflictItem {
    match candidate {
        ConflictCandidate::VariableLowerBound(var) => MathProgramConflictItem::VariableLowerBound {
            var,
            name: program.variables[var].name.clone(),
        },
        ConflictCandidate::VariableUpperBound(var) => MathProgramConflictItem::VariableUpperBound {
            var,
            name: program.variables[var].name.clone(),
        },
        ConflictCandidate::LinearConstraint(index) => MathProgramConflictItem::LinearConstraint {
            index,
            name: program.constraints[index].name.clone(),
        },
    }
}

fn compare_solution_pools(
    program: &MathProgram,
    internal: MathProgramSolutionPool,
    external: MathProgramSolutionPool,
    tol: f64,
) -> MathProgramSolutionPoolCrossCheck {
    let len = internal.solutions.len().min(external.solutions.len());
    let mut objective_abs_diffs = Vec::with_capacity(len);
    let mut max_x_abs_diffs = Vec::with_capacity(len);
    let mut internal_max_violations = Vec::with_capacity(len);
    let mut external_max_violations = Vec::with_capacity(len);

    for i in 0..len {
        let left = &internal.solutions[i];
        let right = &external.solutions[i];
        objective_abs_diffs.push(
            (left.objective.is_finite() && right.objective.is_finite())
                .then_some((left.objective - right.objective).abs()),
        );
        max_x_abs_diffs.push(if left.x.len() == right.x.len() && !left.x.is_empty() {
            Some(
                left.x
                    .iter()
                    .zip(&right.x)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max),
            )
        } else {
            None
        });
        internal_max_violations.push(solution_max_violation(program, &left.x, tol));
        external_max_violations.push(solution_max_violation(program, &right.x, tol));
    }

    let len_agree = internal.solutions.len() == external.solutions.len();
    let within_tolerance = len_agree
        && !internal.solutions.is_empty()
        && internal.exhausted == external.exhausted
        && objective_abs_diffs
            .iter()
            .all(|diff| diff.is_some_and(|value| value <= tol))
        && max_x_abs_diffs
            .iter()
            .all(|diff| diff.is_some_and(|value| value <= tol))
        && internal_max_violations
            .iter()
            .all(|diff| diff.is_some_and(|value| value <= tol))
        && external_max_violations
            .iter()
            .all(|diff| diff.is_some_and(|value| value <= tol));

    MathProgramSolutionPoolCrossCheck {
        internal,
        external,
        len_agree,
        objective_abs_diffs,
        max_x_abs_diffs,
        internal_max_violations,
        external_max_violations,
        within_tolerance,
    }
}

#[derive(Clone, Debug)]
struct PoolDomainEncoding {
    original_var: usize,
    value_literals: Vec<(i64, usize)>,
}

fn solve_math_program_solution_pool_with<F>(
    program: &MathProgram,
    pool_opts: &MathProgramSolutionPoolOptions,
    solver_name: &str,
    mut solve_candidate: F,
) -> Result<MathProgramSolutionPool, MathProgramError>
where
    F: FnMut(&MathProgram) -> Result<MathProgramSolution, MathProgramError>,
{
    program.validate()?;
    validate_solution_pool_options(pool_opts)?;

    if pool_opts.max_solutions == 0 {
        return Ok(MathProgramSolutionPool {
            solutions: Vec::new(),
            exhausted: false,
            solver: solver_name.to_string(),
            message: Some("max_solutions=0".to_string()),
        });
    }

    let mut working = program.clone();
    let encodings =
        add_solution_pool_domain_encodings(&mut working, pool_opts.max_discrete_domain_size)?;
    let mut solutions = Vec::new();
    let mut exhausted = false;
    let mut best_objective = None;
    let mut message = None;

    while solutions.len() < pool_opts.max_solutions {
        let raw = solve_candidate(&working)?;
        if raw.status == MathProgramStatus::Infeasible {
            exhausted = true;
            message = Some("pool exhausted by no-good cuts".to_string());
            break;
        }
        if raw.status != MathProgramStatus::Optimal {
            message = Some(format!("pool search stopped at status {:?}", raw.status));
            break;
        }

        let mut reported = truncate_pool_solution(program, raw.clone());
        if let Some(best) = best_objective {
            if !solution_pool_objective_within_gap(
                program.sense,
                best,
                reported.objective,
                pool_opts,
            ) {
                message = Some("pool search stopped by objective gap".to_string());
                break;
            }
        } else {
            best_objective = Some(reported.objective);
        }
        reported.solver = format!("{solver_name}({})", reported.solver);

        if encodings.is_empty() {
            solutions.push(reported);
            exhausted = true;
            message =
                Some("no finite discrete variables define additional pool entries".to_string());
            break;
        }

        add_solution_pool_no_good_cut(&mut working, &encodings, &raw.x, solutions.len())?;
        solutions.push(reported);
    }

    if solutions.len() == pool_opts.max_solutions && !exhausted {
        message.get_or_insert_with(|| "pool reached max_solutions".to_string());
    }

    Ok(MathProgramSolutionPool {
        solutions,
        exhausted,
        solver: solver_name.to_string(),
        message,
    })
}

fn validate_solution_pool_options(
    opts: &MathProgramSolutionPoolOptions,
) -> Result<(), MathProgramError> {
    if opts.max_discrete_domain_size == 0 {
        return Err(MathProgramError::Unsupported(
            "solution pool max_discrete_domain_size must be positive".to_string(),
        ));
    }
    if opts
        .absolute_gap
        .is_some_and(|gap| !gap.is_finite() || gap < 0.0)
    {
        return Err(MathProgramError::InvalidBound(
            "solution pool absolute_gap must be finite and non-negative".to_string(),
        ));
    }
    if opts
        .relative_gap
        .is_some_and(|gap| !gap.is_finite() || gap < 0.0)
    {
        return Err(MathProgramError::InvalidBound(
            "solution pool relative_gap must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn truncate_pool_solution(
    program: &MathProgram,
    mut solution: MathProgramSolution,
) -> MathProgramSolution {
    if solution.x.len() >= program.variables.len() {
        solution.x.truncate(program.variables.len());
        solution.objective = objective_value(program, &solution.x);
    }
    solution
}

fn solution_pool_objective_within_gap(
    sense: ObjectiveSense,
    best: f64,
    candidate: f64,
    opts: &MathProgramSolutionPoolOptions,
) -> bool {
    let allowed = match (
        opts.absolute_gap,
        opts.relative_gap.map(|gap| gap * best.abs()),
    ) {
        (None, None) => return true,
        (Some(abs), None) => abs,
        (None, Some(rel)) => rel,
        (Some(abs), Some(rel)) => abs.max(rel),
    };
    match sense {
        ObjectiveSense::Max => candidate >= best - allowed - 1e-8,
        ObjectiveSense::Min => candidate <= best + allowed + 1e-8,
    }
}

fn add_solution_pool_domain_encodings(
    program: &mut MathProgram,
    max_domain_size: usize,
) -> Result<Vec<PoolDomainEncoding>, MathProgramError> {
    let original_len = program.variables.len();
    let mut encodings = Vec::new();
    for var_idx in 0..original_len {
        let values = solution_pool_domain_values(&program.variables[var_idx])?;
        if values.is_empty() {
            continue;
        }
        if values.len() > max_domain_size {
            return Err(MathProgramError::Unsupported(format!(
                "solution pool variable `{}` has domain size {}, above max_discrete_domain_size={max_domain_size}",
                program.variables[var_idx].name,
                values.len()
            )));
        }

        let var_name = program.variables[var_idx].name.clone();
        let mut value_literals = Vec::new();
        for value in values {
            let literal = program.add_binary_var(format!("__pool_{}_is_{value}", var_name), 0.0)?;
            value_literals.push((value, literal));
        }

        let choose_terms = value_literals
            .iter()
            .map(|&(_, lit)| (lit, 1.0))
            .collect::<Vec<_>>();
        program.add_constraint(
            format!("__pool_{}_choose_value", var_name),
            choose_terms,
            RowSense::Eq,
            1.0,
        )?;

        let mut link_terms = vec![(var_idx, 1.0)];
        link_terms.extend(
            value_literals
                .iter()
                .map(|&(value, lit)| (lit, -(value as f64))),
        );
        program.add_constraint(
            format!("__pool_{}_link_value", var_name),
            link_terms,
            RowSense::Eq,
            0.0,
        )?;

        encodings.push(PoolDomainEncoding {
            original_var: var_idx,
            value_literals,
        });
    }
    Ok(encodings)
}

fn solution_pool_domain_values(var: &Variable) -> Result<Vec<i64>, MathProgramError> {
    match var.var_type {
        VariableType::Binary => Ok(vec![0, 1]),
        VariableType::Integer => {
            let (lower, upper) = integer_bounds(var).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "solution pool integer variable `{}` requires finite integer bounds",
                    var.name
                ))
            })?;
            Ok((lower..=upper).collect())
        }
        VariableType::SemiInteger => {
            let lower = var.lb.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "solution pool semi-integer variable `{}` requires finite lb",
                    var.name
                ))
            })?;
            let upper = var.ub.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "solution pool semi-integer variable `{}` requires finite ub",
                    var.name
                ))
            })?;
            let lower = lower.ceil() as i64;
            let upper = upper.floor() as i64;
            let mut values = vec![0];
            values.extend((lower..=upper).filter(|&value| value != 0));
            values.sort_unstable();
            values.dedup();
            Ok(values)
        }
        VariableType::Continuous | VariableType::SemiContinuous => Ok(Vec::new()),
    }
}

fn add_solution_pool_no_good_cut(
    program: &mut MathProgram,
    encodings: &[PoolDomainEncoding],
    full_x: &[f64],
    cut_idx: usize,
) -> Result<(), MathProgramError> {
    let mut terms = Vec::new();
    for encoding in encodings {
        let value = full_x
            .get(encoding.original_var)
            .copied()
            .unwrap_or(f64::NAN)
            .round() as i64;
        let literal = encoding
            .value_literals
            .iter()
            .find_map(|&(candidate, lit)| (candidate == value).then_some(lit))
            .ok_or_else(|| {
                MathProgramError::Unsupported(format!(
                    "solution pool value {value} is outside the encoded domain for variable {}",
                    encoding.original_var
                ))
            })?;
        terms.push((literal, 1.0));
    }
    program.add_constraint(
        format!("__pool_no_good_{cut_idx}"),
        terms,
        RowSense::Le,
        encodings.len() as f64 - 1.0,
    )?;
    Ok(())
}

/// Solve a model through the optional Python external-solver oracle.
pub fn solve_math_program_external_scipy(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty() {
        return solve_hierarchical_math_program_external(program, opts);
    }
    solve_math_program_external_scipy_single_objective(program, opts)
}

fn solve_math_program_external_scipy_single_objective(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    program.validate()?;
    let method = opts.method.clone().unwrap_or_else(|| {
        if !program.has_discrete_features()
            && (program.has_quadratic_objective()
                || program.has_conic_constraints()
                || program.has_quadratic_constraints())
        {
            "SLSQP".to_string()
        } else {
            "highs".to_string()
        }
    });
    let python = opts
        .python
        .clone()
        .or_else(|| std::env::var("PYTHON").ok())
        .unwrap_or_else(|| "python3".to_string());
    let script = opts
        .script
        .clone()
        .unwrap_or_else(|| "external-references/math-program/math_program_solve.py".to_string());

    let (payload, compiled) = if program.has_discrete_features()
        && (program.has_conic_constraints() || program.has_quadratic_constraints())
    {
        if !can_encode_direct_mixed_integer_nonlinear(program) {
            return Ok(MathProgramSolution {
                status: MathProgramStatus::NumericalError,
                x: Vec::new(),
                objective: f64::NAN,
                solver: method,
                message: Some(
                    "external mixed-integer conic/quadratic oracle requires direct continuous/integer/binary variables without lowered discrete constraints"
                        .to_string(),
                ),
            });
        }
        (
            json!({
                "kind": "conic",
                "conic": encode_conic_problem(program)?,
                "method": method,
            }),
            None,
        )
    } else if program.has_discrete_features()
        && program.has_quadratic_objective()
        && quadratic_objective_has_native_nonlinear_terms(program)
    {
        if !can_encode_direct_mixed_integer_nonlinear(program) {
            return Ok(MathProgramSolution {
                status: MathProgramStatus::NumericalError,
                x: Vec::new(),
                objective: f64::NAN,
                solver: method,
                message: Some(
                    "external mixed-integer quadratic objective oracle requires direct continuous/integer/binary variables without lowered discrete constraints"
                        .to_string(),
                ),
            });
        }
        (
            json!({
                "kind": "qp",
                "qp": encode_qp_problem(program)?,
                "method": method,
            }),
            None,
        )
    } else if program.has_discrete_features() {
        let compiled = compile_mip(program)?;
        let mut mip_payload = encode_ipmip_problem(&compiled.problem);
        if let Some(start) = &opts.mip_start {
            if let Some(object) = mip_payload.as_object_mut() {
                object.insert(
                    "mipStart".to_string(),
                    Value::Array(
                        canonical_mip_start(program, &compiled, start)?
                            .into_iter()
                            .map(Value::from)
                            .collect(),
                    ),
                );
            }
        }
        (
            json!({
                "kind": "mip",
                "mip": mip_payload,
                "method": method,
            }),
            Some(compiled),
        )
    } else if program.has_conic_constraints() || program.has_quadratic_constraints() {
        (
            json!({
                "kind": "conic",
                "conic": encode_conic_problem(program)?,
                "method": method,
            }),
            None,
        )
    } else if program.has_quadratic_objective() {
        (
            json!({
                "kind": "qp",
                "qp": encode_qp_problem(program)?,
                "method": method,
            }),
            None,
        )
    } else {
        (
            json!({
                "kind": "lp",
                "lp": encode_lp_problem(&program.to_lp_problem()?),
                "method": method,
            }),
            None,
        )
    };

    let raw = run_external_math_program(&python, &script, &method, &payload)?;
    let status = parse_external_status(&raw);
    let raw_x = raw
        .get("x")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
        .unwrap_or_default();
    let x = match &compiled {
        Some(compiled) if status == MathProgramStatus::Optimal => compiled.original_x(&raw_x),
        _ => raw_x,
    };
    let objective = if status == MathProgramStatus::Optimal && x.len() == program.variables.len() {
        objective_value(program, &x)
    } else {
        raw.get("objective")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN)
    };

    Ok(MathProgramSolution {
        status,
        x,
        objective,
        solver: if let Some(solver) = raw.get("solver").and_then(Value::as_str) {
            solver.to_string()
        } else if method.starts_with("ortools:") {
            method.clone()
        } else if program.has_discrete_features() {
            "scipy:milp".to_string()
        } else {
            format!("scipy:{method}")
        },
        message: raw
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

fn solve_lp_with_backend(lp: &LPProblem, opts: &MathProgramSolveOptions) -> LPSolution {
    match opts.lp_backend {
        MathProgramLpBackend::InternalSimplex => solve_lp_internal(lp, &opts.lp_simplex),
        MathProgramLpBackend::InternalInteriorPoint => solve_lp_internal_ipm(lp, &opts.lp_ipm),
        MathProgramLpBackend::ScipyHighs => solve_lp_external(
            lp,
            &ExternalSolverOptions {
                method: Some("highs".to_string()),
                ..opts.external_lp.clone()
            },
        ),
        MathProgramLpBackend::ScipyHighsDs => solve_lp_external(
            lp,
            &ExternalSolverOptions {
                method: Some("highs-ds".to_string()),
                ..opts.external_lp.clone()
            },
        ),
        MathProgramLpBackend::ScipyHighsIpm => solve_lp_external(
            lp,
            &ExternalSolverOptions {
                method: Some("highs-ipm".to_string()),
                ..opts.external_lp.clone()
            },
        ),
    }
}

fn solve_continuous_qp(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    if program.has_discrete_features() {
        return Err(MathProgramError::Unsupported(
            "continuous QP solver requires continuous variables and linear constraints only"
                .to_string(),
        ));
    }
    if !program
        .variables
        .iter()
        .all(|v| v.var_type == VariableType::Continuous)
    {
        return Err(MathProgramError::Unsupported(
            "continuous QP solver only accepts continuous variables".to_string(),
        ));
    }

    let hessian = quadratic_hessian(program);
    let minimize_sign = match program.sense {
        ObjectiveSense::Min => 1.0,
        ObjectiveSense::Max => -1.0,
    };
    let min_hessian = scale_matrix(&hessian, minimize_sign);
    let min_eig = min_symmetric_eigenvalue(min_hessian.clone());
    if min_eig < -1e-8 {
        return Err(MathProgramError::Unsupported(format!(
            "continuous QP requires convex minimization or concave maximization; transformed minimum eigenvalue is {min_eig:.3e}"
        )));
    }

    let lp = program.to_linear_relaxation_lp_problem()?;
    let mut feasibility_lp = lp.clone();
    feasibility_lp.sense = LpSense::Min;
    feasibility_lp.c = vec![0.0; program.variables.len()];
    let feasibility = solve_lp_with_backend(&feasibility_lp, opts);
    if feasibility.status != LPStatus::Optimal {
        return Ok(MathProgramSolution {
            status: from_lp_status(feasibility.status),
            x: feasibility.x,
            objective: f64::NAN,
            solver: "des-frank-wolfe-qp".to_string(),
            message: feasibility.message,
        });
    }

    let mut x = feasibility.x;
    let mut iterations = 0usize;
    let mut converged = false;
    for iter in 0..opts.qp.max_iter {
        iterations = iter;
        let grad = scale_vec(&quadratic_gradient(program, &x), minimize_sign);
        let mut linear_lp = lp.clone();
        linear_lp.sense = LpSense::Min;
        linear_lp.c = grad.clone();
        let linear = solve_lp_with_backend(&linear_lp, opts);
        if linear.status != LPStatus::Optimal {
            let objective = objective_value(program, &x);
            return Ok(MathProgramSolution {
                status: from_lp_status(linear.status),
                x,
                objective,
                solver: "des-frank-wolfe-qp".to_string(),
                message: linear.message,
            });
        }
        let direction = linear
            .x
            .iter()
            .zip(&x)
            .map(|(s, xi)| s - xi)
            .collect::<Vec<_>>();
        let gap = -dot(&grad, &direction);
        if gap <= opts.qp.tolerance {
            converged = true;
            break;
        }
        let curvature = quadratic_form(&min_hessian, &direction);
        let directional = dot(&grad, &direction);
        let alpha = if curvature > 1e-12 {
            (-directional / curvature).clamp(0.0, 1.0)
        } else if directional < -opts.qp.tolerance {
            1.0
        } else {
            0.0
        };
        if alpha <= 1e-12 {
            converged = true;
            break;
        }
        for (xi, di) in x.iter_mut().zip(direction) {
            *xi += alpha * di;
        }
    }

    let objective = objective_value(program, &x);
    Ok(MathProgramSolution {
        status: if converged {
            MathProgramStatus::Optimal
        } else {
            MathProgramStatus::IterLimit
        },
        x,
        objective,
        solver: "des-frank-wolfe-qp".to_string(),
        message: Some(format!(
            "iterations={}, tolerance={:.3e}",
            iterations, opts.qp.tolerance
        )),
    })
}

fn solve_continuous_conic(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    if program.has_discrete_features() {
        return Err(MathProgramError::Unsupported(
            "continuous conic solver requires continuous variables".to_string(),
        ));
    }

    let mut relaxation = program.clone();
    relaxation.second_order_cones.clear();
    relaxation.quadratic_constraints.clear();
    for (i, cone) in program.second_order_cones.iter().enumerate() {
        let nonnegative_rhs = cone
            .rhs_coeffs
            .iter()
            .map(|&(idx, coef)| (idx, -coef))
            .collect::<Vec<_>>();
        relaxation.add_constraint(
            format!("{}__rhs_nonnegative_{i}", cone.name),
            nonnegative_rhs,
            RowSense::Le,
            cone.rhs_constant,
        )?;
    }

    let solver_name = if program.has_quadratic_constraints() {
        "des-convex-cutting-plane"
    } else {
        "des-soc-cutting-plane"
    };
    let mut best = None;
    for cut in 0..=opts.conic.max_cuts {
        let mut solution = if relaxation.has_quadratic_objective() {
            solve_continuous_qp(&relaxation, opts)?
        } else {
            from_lp_solution(solve_lp_with_backend(
                &relaxation.to_linear_relaxation_lp_problem()?,
                opts,
            ))
        };
        if solution.status != MathProgramStatus::Optimal {
            return Ok(solution);
        }

        let soc_violation = most_violated_soc(program, &solution.x);
        let qcp_violation = most_violated_quadratic_constraint(program, &solution.x);
        let soc_amount = soc_violation
            .as_ref()
            .map_or(0.0, |(_, amount, _, _)| *amount);
        let qcp_amount = qcp_violation.as_ref().map_or(0.0, |(_, amount)| *amount);
        if soc_amount <= opts.conic.tolerance && qcp_amount <= opts.conic.tolerance {
            solution.solver = solver_name.to_string();
            solution.message = Some(format!(
                "cuts={}, tolerance={:.3e}",
                cut, opts.conic.tolerance
            ));
            return Ok(solution);
        }

        let cut_x = solution.x.clone();
        best = Some(solution);
        if qcp_amount >= soc_amount {
            let (row_idx, _) = qcp_violation.unwrap();
            let row = &program.quadratic_constraints[row_idx];
            let (coeffs, rhs) = quadratic_constraint_supporting_cut(row, &cut_x);
            relaxation.add_constraint(
                format!("{}__qcp_cut_{cut}", row.name),
                coeffs,
                RowSense::Le,
                rhs,
            )?;
        } else {
            let (cone_idx, _, norm_values, rhs_value) = soc_violation.unwrap();
            let cone = &program.second_order_cones[cone_idx];
            let norm = norm2(&norm_values);
            if norm <= 1e-12 {
                return Err(MathProgramError::Unsupported(format!(
                    "second-order cone `{}` is violated with near-zero norm and rhs {rhs_value}",
                    cone.name
                )));
            }
            let unit = norm_values
                .iter()
                .map(|value| value / norm)
                .collect::<Vec<_>>();
            let (coeffs, rhs) = soc_supporting_cut(cone, &unit);
            relaxation.add_constraint(
                format!("{}__soc_cut_{cut}", cone.name),
                coeffs,
                RowSense::Le,
                rhs,
            )?;
        }
    }

    let mut solution = best.unwrap_or(MathProgramSolution {
        status: MathProgramStatus::NumericalError,
        x: Vec::new(),
        objective: f64::NAN,
        solver: solver_name.to_string(),
        message: Some("no relaxation solve was attempted".to_string()),
    });
    solution.status = MathProgramStatus::IterLimit;
    solution.solver = solver_name.to_string();
    solution.message = Some(format!(
        "max_cuts={}, tolerance={:.3e}",
        opts.conic.max_cuts, opts.conic.tolerance
    ));
    Ok(solution)
}

fn run_external_math_program(
    python: &str,
    script: &str,
    method: &str,
    payload: &Value,
) -> Result<Value, MathProgramError> {
    let mut child = Command::new(python)
        .arg(script)
        .arg("--method")
        .arg(method)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            MathProgramError::External(format!("external math-program solver could not start: {e}"))
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload.to_string().as_bytes())
            .map_err(|e| MathProgramError::External(format!("external stdin failed: {e}")))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| MathProgramError::External(format!("external solver wait failed: {e}")))?;

    if out.status.code() != Some(0) {
        let code = out
            .status
            .code()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Ok(json!({
            "status": "numerical-error",
            "x": [],
            "objective": null,
            "message": format!("external solver exited with {code}: {stderr}"),
        }));
    }

    serde_json::from_slice(&out.stdout)
        .map_err(|e| MathProgramError::External(format!("external JSON parse failed: {e}")))
}

fn encode_lp_problem(lp: &LPProblem) -> Value {
    json!({
        "sense": lp.sense.as_str(),
        "c": &lp.c,
        "A_ub": &lp.a_ub,
        "b_ub": &lp.b_ub,
        "A_eq": &lp.a_eq,
        "b_eq": &lp.b_eq,
        "lb": &lp.lb,
        "ub": &lp.ub,
        "varNames": &lp.var_names,
        "conNames": &lp.con_names,
    })
}

fn encode_qp_problem(program: &MathProgram) -> Result<Value, MathProgramError> {
    let lp = program.to_linear_relaxation_lp_problem()?;
    Ok(json!({
        "sense": program.sense.to_lp().as_str(),
        "c": &lp.c,
        "quadratic": program.quadratic_objective.iter().map(|term| {
            json!({
                "i": term.var_i,
                "j": term.var_j,
                "coeff": term.coeff,
            })
        }).collect::<Vec<_>>(),
        "A_ub": &lp.a_ub,
        "b_ub": &lp.b_ub,
        "A_eq": &lp.a_eq,
        "b_eq": &lp.b_eq,
        "lb": &lp.lb,
        "ub": &lp.ub,
        "integerVars": program.variables.iter().map(|var| {
            Value::Bool(matches!(
                var.var_type,
                VariableType::Integer | VariableType::Binary
            ))
        }).collect::<Vec<_>>(),
        "varNames": &lp.var_names,
        "conNames": &lp.con_names,
    }))
}

fn encode_conic_problem(program: &MathProgram) -> Result<Value, MathProgramError> {
    let mut value = encode_qp_problem(program)?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "integerVars".to_string(),
            Value::Array(
                program
                    .variables
                    .iter()
                    .map(|var| {
                        Value::Bool(matches!(
                            var.var_type,
                            VariableType::Integer | VariableType::Binary
                        ))
                    })
                    .collect::<Vec<_>>(),
            ),
        );
        object.insert(
            "soc".to_string(),
            Value::Array(
                program
                    .second_order_cones
                    .iter()
                    .map(|cone| {
                        json!({
                            "name": cone.name,
                            "terms": cone.terms.iter().map(|term| {
                                json!({
                                    "coeffs": term.coeffs,
                                    "constant": term.constant,
                                })
                            }).collect::<Vec<_>>(),
                            "rhsCoeffs": cone.rhs_coeffs,
                            "rhsConstant": cone.rhs_constant,
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
        );
        object.insert(
            "quadraticConstraints".to_string(),
            Value::Array(
                program
                    .quadratic_constraints
                    .iter()
                    .map(|row| {
                        json!({
                            "name": row.name,
                            "quadratic": row.quadratic_terms.iter().map(|term| {
                                json!({
                                    "i": term.var_i,
                                    "j": term.var_j,
                                    "coeff": term.coeff,
                                })
                            }).collect::<Vec<_>>(),
                            "linear": row.linear_terms,
                            "sense": row.sense.as_str(),
                            "rhs": row.rhs,
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
        );
    }
    Ok(value)
}

fn encode_ipmip_problem(mip: &IPMIPProblem) -> Value {
    json!({
        "sense": mip.sense.as_str(),
        "c": &mip.c,
        "A": &mip.a,
        "b": &mip.b,
        "integerVars": &mip.integer_vars,
        "ub": &mip.ub,
        "varNames": &mip.var_names,
        "conNames": &mip.con_names,
    })
}

fn parse_external_status(raw: &Value) -> MathProgramStatus {
    match raw.get("status").and_then(Value::as_str) {
        Some("optimal") => MathProgramStatus::Optimal,
        Some("infeasible") => MathProgramStatus::Infeasible,
        Some("unbounded") => MathProgramStatus::Unbounded,
        Some("iter-limit") => MathProgramStatus::IterLimit,
        Some("time-limit") => MathProgramStatus::TimeLimit,
        _ => MathProgramStatus::NumericalError,
    }
}

fn from_lp_solution(sol: LPSolution) -> MathProgramSolution {
    MathProgramSolution {
        status: from_lp_status(sol.status),
        x: sol.x,
        objective: sol.objective,
        solver: sol.solver,
        message: sol.message,
    }
}

fn from_lp_status(status: LPStatus) -> MathProgramStatus {
    match status {
        LPStatus::Optimal => MathProgramStatus::Optimal,
        LPStatus::Infeasible => MathProgramStatus::Infeasible,
        LPStatus::Unbounded => MathProgramStatus::Unbounded,
        LPStatus::IterLimit => MathProgramStatus::IterLimit,
        LPStatus::NumericalError => MathProgramStatus::NumericalError,
    }
}

fn from_ipmip_status(status: IPMIPStatus) -> MathProgramStatus {
    match status {
        IPMIPStatus::Optimal => MathProgramStatus::Optimal,
        IPMIPStatus::Infeasible => MathProgramStatus::Infeasible,
        IPMIPStatus::Unbounded => MathProgramStatus::Unbounded,
        IPMIPStatus::MaxNodes | IPMIPStatus::TickLimit => MathProgramStatus::NodeLimit,
        IPMIPStatus::TimeLimit => MathProgramStatus::TimeLimit,
    }
}

#[derive(Clone, Debug)]
struct LinearExpansion {
    constant: f64,
    terms: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct SparseRow {
    coeffs: Vec<(usize, f64)>,
    rhs: f64,
    name: String,
}

#[derive(Clone, Debug)]
struct CompiledMip {
    problem: IPMIPProblem,
    expansions: Vec<LinearExpansion>,
}

impl CompiledMip {
    fn original_x(&self, canonical_x: &[f64]) -> Vec<f64> {
        self.expansions
            .iter()
            .map(|expansion| eval_expansion(expansion, canonical_x))
            .collect()
    }
}

fn canonical_mip_start(
    program: &MathProgram,
    compiled: &CompiledMip,
    start: &[f64],
) -> Result<Vec<f64>, MathProgramError> {
    if start.len() != program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "MIP start length {} does not match {} variables",
            start.len(),
            program.variables.len()
        )));
    }
    if start.iter().any(|v| !v.is_finite()) {
        return Err(MathProgramError::NonFinite(
            "MIP start contains a non-finite value".to_string(),
        ));
    }

    let mut canonical = vec![0.0; compiled.problem.c.len()];
    for (i, value) in start.iter().copied().enumerate() {
        set_expansion_start_value(&mut canonical, &compiled.expansions[i], value)?;
        if matches!(
            program.variables[i].var_type,
            VariableType::SemiContinuous | VariableType::SemiInteger
        ) {
            if let Some(active_idx) =
                compiled_var_index(compiled, &format!("{}__active", program.variables[i].name))
            {
                canonical[active_idx] = if value.abs() > 1e-9 { 1.0 } else { 0.0 };
            }
        }
    }
    Ok(canonical)
}

fn set_expansion_start_value(
    canonical: &mut [f64],
    expansion: &LinearExpansion,
    value: f64,
) -> Result<(), MathProgramError> {
    match expansion.terms.as_slice() {
        [(j, 1.0)] => {
            canonical[*j] = value - expansion.constant;
            Ok(())
        }
        [(j, -1.0)] => {
            canonical[*j] = expansion.constant - value;
            Ok(())
        }
        [(pos, 1.0), (neg, -1.0)] if expansion.constant.abs() <= 1e-12 => {
            canonical[*pos] = value.max(0.0);
            canonical[*neg] = (-value).max(0.0);
            Ok(())
        }
        _ => Err(MathProgramError::Unsupported(
            "MIP start cannot be mapped through this variable expansion".to_string(),
        )),
    }
}

fn compiled_var_index(compiled: &CompiledMip, name: &str) -> Option<usize> {
    compiled
        .problem
        .var_names
        .as_ref()?
        .iter()
        .position(|candidate| candidate == name)
}

fn compile_mip(program: &MathProgram) -> Result<CompiledMip, MathProgramError> {
    let mut names = Vec::new();
    let mut integer_vars = Vec::new();
    let mut ub = Vec::new();
    let mut expansions = Vec::new();
    let mut rows = Vec::new();

    for var in &program.variables {
        let expansion = compile_variable(var, &mut names, &mut integer_vars, &mut ub, &mut rows)?;
        expansions.push(expansion);
    }

    for row in &program.constraints {
        add_program_row(
            &mut rows,
            row.name.clone(),
            &expansions,
            &row.coeffs,
            row.sense,
            row.rhs,
        );
    }
    for indicator in &program.indicators {
        add_indicator_rows(program, &mut rows, &expansions, indicator)?;
    }
    for sos in &program.sos {
        add_sos_rows(
            program,
            &mut names,
            &mut integer_vars,
            &mut ub,
            &mut rows,
            &expansions,
            sos,
        )?;
    }
    for general in &program.general_constraints {
        add_general_constraint_rows(
            program,
            &mut names,
            &mut integer_vars,
            &mut ub,
            &mut rows,
            &expansions,
            general,
        )?;
    }
    let quadratic_objective_terms = add_quadratic_objective_rows(
        program,
        &mut names,
        &mut integer_vars,
        &mut ub,
        &mut rows,
        &expansions,
    )?;

    if rows.is_empty() {
        rows.push(SparseRow {
            coeffs: Vec::new(),
            rhs: 0.0,
            name: "__dummy_feasibility_row".to_string(),
        });
    }

    let n = names.len();
    let mut a = Vec::with_capacity(rows.len());
    let mut b = Vec::with_capacity(rows.len());
    let mut con_names = Vec::with_capacity(rows.len());
    for row in rows {
        a.push(dense_row(n, &row.coeffs));
        b.push(row.rhs);
        con_names.push(row.name);
    }

    let mut c = vec![0.0; n];
    for (i, var) in program.variables.iter().enumerate() {
        for &(j, coef) in &expansions[i].terms {
            c[j] += var.obj * coef;
        }
    }
    for (j, coef) in quadratic_objective_terms {
        c[j] += coef;
    }

    Ok(CompiledMip {
        problem: IPMIPProblem {
            sense: program.sense.to_lp(),
            c,
            a,
            b,
            integer_vars,
            ub: Some(ub),
            var_names: Some(names),
            con_names: Some(con_names),
            variable_nodes: None,
            constraint_nodes: None,
        },
        expansions,
    })
}

fn add_compiled_mip_cut(
    compiled: &mut CompiledMip,
    name: String,
    coeffs: Vec<(usize, f64)>,
    sense: RowSense,
    rhs: f64,
) {
    let mut rows = Vec::new();
    add_program_row(&mut rows, name, &compiled.expansions, &coeffs, sense, rhs);
    let n = compiled.problem.c.len();
    for row in rows {
        compiled.problem.a.push(dense_row(n, &row.coeffs));
        compiled.problem.b.push(row.rhs);
        if let Some(con_names) = compiled.problem.con_names.as_mut() {
            con_names.push(row.name);
        }
    }
}

fn compile_variable(
    var: &Variable,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
) -> Result<LinearExpansion, MathProgramError> {
    match var.var_type {
        VariableType::Continuous => compile_shifted_or_split(var, false, names, integer_vars, ub),
        VariableType::Integer => compile_shifted_or_split(var, true, names, integer_vars, ub),
        VariableType::Binary => {
            let j = push_canonical_var(&var.name, true, 1.0, names, integer_vars, ub);
            Ok(LinearExpansion {
                constant: 0.0,
                terms: vec![(j, 1.0)],
            })
        }
        VariableType::SemiContinuous | VariableType::SemiInteger => {
            let lower = var.lb.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "semi-continuous variable `{}` requires finite lb",
                    var.name
                ))
            })?;
            let upper = var.ub.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "semi-continuous variable `{}` requires finite ub",
                    var.name
                ))
            })?;
            if lower < 0.0 || upper < lower {
                return Err(MathProgramError::InvalidBound(format!(
                    "invalid semi-continuous bounds for `{}`",
                    var.name
                )));
            }
            let x = push_canonical_var(
                &var.name,
                var.var_type == VariableType::SemiInteger,
                upper,
                names,
                integer_vars,
                ub,
            );
            let z = push_canonical_var(
                &format!("{}__active", var.name),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            rows.push(SparseRow {
                coeffs: vec![(x, 1.0), (z, -upper)],
                rhs: 0.0,
                name: format!("{}__semi_upper", var.name),
            });
            rows.push(SparseRow {
                coeffs: vec![(x, -1.0), (z, lower)],
                rhs: 0.0,
                name: format!("{}__semi_lower", var.name),
            });
            Ok(LinearExpansion {
                constant: 0.0,
                terms: vec![(x, 1.0)],
            })
        }
    }
}

fn compile_shifted_or_split(
    var: &Variable,
    integer: bool,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
) -> Result<LinearExpansion, MathProgramError> {
    let (lb, upper) = normalized_bounds(var, integer)?;
    match (lb, upper) {
        (Some(lower), Some(upper)) => {
            if upper < lower {
                return Err(MathProgramError::InvalidBound(format!(
                    "upper bound below lower bound for `{}`",
                    var.name
                )));
            }
            let j = push_canonical_var(&var.name, integer, upper - lower, names, integer_vars, ub);
            Ok(LinearExpansion {
                constant: lower,
                terms: vec![(j, 1.0)],
            })
        }
        (Some(lower), None) => {
            let j = push_canonical_var(&var.name, integer, f64::INFINITY, names, integer_vars, ub);
            Ok(LinearExpansion {
                constant: lower,
                terms: vec![(j, 1.0)],
            })
        }
        (None, Some(upper)) => {
            let j = push_canonical_var(&var.name, integer, f64::INFINITY, names, integer_vars, ub);
            Ok(LinearExpansion {
                constant: upper,
                terms: vec![(j, -1.0)],
            })
        }
        (None, None) => {
            let pos = push_canonical_var(
                &format!("{}__pos", var.name),
                integer,
                f64::INFINITY,
                names,
                integer_vars,
                ub,
            );
            let neg = push_canonical_var(
                &format!("{}__neg", var.name),
                integer,
                f64::INFINITY,
                names,
                integer_vars,
                ub,
            );
            Ok(LinearExpansion {
                constant: 0.0,
                terms: vec![(pos, 1.0), (neg, -1.0)],
            })
        }
    }
}

fn normalized_bounds(
    var: &Variable,
    integer: bool,
) -> Result<(Option<f64>, Option<f64>), MathProgramError> {
    if !integer {
        return Ok((var.lb, var.ub));
    }
    let lb = var.lb.map(f64::ceil);
    let ub = var.ub.map(f64::floor);
    if let (Some(lower), Some(upper)) = (lb, ub) {
        if upper < lower {
            return Err(MathProgramError::InvalidBound(format!(
                "integer variable `{}` has no integer value inside its bounds",
                var.name
            )));
        }
    }
    Ok((lb, ub))
}

fn push_canonical_var(
    name: &str,
    integer: bool,
    upper: f64,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
) -> usize {
    names.push(name.to_string());
    integer_vars.push(integer);
    ub.push(upper);
    names.len() - 1
}

fn add_program_row(
    rows: &mut Vec<SparseRow>,
    name: String,
    expansions: &[LinearExpansion],
    coeffs: &[(usize, f64)],
    sense: RowSense,
    rhs: f64,
) {
    let (expanded, shifted_rhs) = expand_row(expansions, coeffs, rhs);
    match sense {
        RowSense::Le => rows.push(SparseRow {
            coeffs: expanded,
            rhs: shifted_rhs,
            name,
        }),
        RowSense::Ge => rows.push(SparseRow {
            coeffs: negate_sparse(&expanded),
            rhs: -shifted_rhs,
            name,
        }),
        RowSense::Eq => {
            rows.push(SparseRow {
                coeffs: expanded.clone(),
                rhs: shifted_rhs,
                name: format!("{name}__eq_le"),
            });
            rows.push(SparseRow {
                coeffs: negate_sparse(&expanded),
                rhs: -shifted_rhs,
                name: format!("{name}__eq_ge"),
            });
        }
    }
}

fn add_indicator_rows(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    indicator: &IndicatorConstraint,
) -> Result<(), MathProgramError> {
    match indicator.sense {
        RowSense::Le => add_indicator_le(
            program,
            rows,
            expansions,
            indicator,
            &indicator.coeffs,
            indicator.rhs,
        ),
        RowSense::Ge => {
            let coeffs = indicator
                .coeffs
                .iter()
                .map(|&(i, v)| (i, -v))
                .collect::<Vec<_>>();
            add_indicator_le(
                program,
                rows,
                expansions,
                indicator,
                &coeffs,
                -indicator.rhs,
            )
        }
        RowSense::Eq => {
            add_indicator_le(
                program,
                rows,
                expansions,
                indicator,
                &indicator.coeffs,
                indicator.rhs,
            )?;
            let coeffs = indicator
                .coeffs
                .iter()
                .map(|&(i, v)| (i, -v))
                .collect::<Vec<_>>();
            add_indicator_le(
                program,
                rows,
                expansions,
                indicator,
                &coeffs,
                -indicator.rhs,
            )
        }
    }
}

fn add_indicator_le(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    indicator: &IndicatorConstraint,
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> Result<(), MathProgramError> {
    let (_, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "indicator `{}` needs finite variable bounds for big-M lowering",
            indicator.name
        ))
    })?;
    let big_m = 0.0_f64.max(max_lhs - rhs);
    let mut lifted = coeffs.to_vec();
    if indicator.active_value {
        lifted.push((indicator.binary_var, big_m));
        add_program_row(
            rows,
            format!("{}__indicator", indicator.name),
            expansions,
            &lifted,
            RowSense::Le,
            rhs + big_m,
        );
    } else {
        lifted.push((indicator.binary_var, -big_m));
        add_program_row(
            rows,
            format!("{}__indicator", indicator.name),
            expansions,
            &lifted,
            RowSense::Le,
            rhs,
        );
    }
    Ok(())
}

fn add_sos_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    sos: &SOSConstraint,
) -> Result<(), MathProgramError> {
    let mut members = sos.members.clone();
    members.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    match sos.sos_type {
        SOSType::Sos1 => {
            let activators = members
                .iter()
                .enumerate()
                .map(|(k, _)| {
                    push_canonical_var(
                        &format!("{}__sos1_active_{k}", sos.name),
                        true,
                        1.0,
                        names,
                        integer_vars,
                        ub,
                    )
                })
                .collect::<Vec<_>>();
            rows.push(SparseRow {
                coeffs: activators.iter().map(|&z| (z, 1.0)).collect(),
                rhs: 1.0,
                name: format!("{}__sos1_at_most_one", sos.name),
            });
            for (k, &(var_idx, _)) in members.iter().enumerate() {
                add_zero_or_active_rows(
                    program,
                    rows,
                    expansions,
                    var_idx,
                    &[(activators[k], 1.0)],
                    &format!("{}__sos1_member_{k}", sos.name),
                )?;
            }
        }
        SOSType::Sos2 => {
            let intervals = (0..members.len() - 1)
                .map(|k| {
                    push_canonical_var(
                        &format!("{}__sos2_interval_{k}", sos.name),
                        true,
                        1.0,
                        names,
                        integer_vars,
                        ub,
                    )
                })
                .collect::<Vec<_>>();
            rows.push(SparseRow {
                coeffs: intervals.iter().map(|&z| (z, 1.0)).collect(),
                rhs: 1.0,
                name: format!("{}__sos2_one_interval", sos.name),
            });
            for (i, &(var_idx, _)) in members.iter().enumerate() {
                let mut adjacent = Vec::new();
                if i > 0 {
                    adjacent.push((intervals[i - 1], 1.0));
                }
                if i + 1 < members.len() {
                    adjacent.push((intervals[i], 1.0));
                }
                add_zero_or_active_rows(
                    program,
                    rows,
                    expansions,
                    var_idx,
                    &adjacent,
                    &format!("{}__sos2_member_{i}", sos.name),
                )?;
            }
        }
    }
    Ok(())
}

fn add_zero_or_active_rows(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    var_idx: usize,
    activator_sum: &[(usize, f64)],
    name: &str,
) -> Result<(), MathProgramError> {
    let (lower, upper) = variable_bounds(&program.variables[var_idx]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "variable `{}` requires finite bounds for activation lowering",
            program.variables[var_idx].name
        ))
    })?;
    let upper_terms = activator_sum
        .iter()
        .map(|&(idx, coef)| (idx, -upper * coef))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__upper"),
        expansions,
        &[(var_idx, 1.0)],
        &upper_terms,
        RowSense::Le,
        0.0,
    );
    let lower_terms = activator_sum
        .iter()
        .map(|&(idx, coef)| (idx, lower * coef))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__lower"),
        expansions,
        &[(var_idx, -1.0)],
        &lower_terms,
        RowSense::Le,
        0.0,
    );
    Ok(())
}

fn add_quadratic_objective_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
) -> Result<Vec<(usize, f64)>, MathProgramError> {
    let mut objective_terms = Vec::new();
    for (k, term) in program.quadratic_objective.iter().enumerate() {
        if term.var_i == term.var_j {
            let idx = canonical_binary_index(&expansions[term.var_i]).ok_or_else(|| {
                MathProgramError::Unsupported(format!(
                    "quadratic diagonal term {k} could not map `{}` to a canonical binary variable",
                    program.variables[term.var_i].name
                ))
            })?;
            objective_terms.push((idx, term.coeff));
            continue;
        }

        let product = push_canonical_var(
            &format!(
                "__quad_obj_{}_times_{}",
                program.variables[term.var_i].name, program.variables[term.var_j].name
            ),
            true,
            1.0,
            names,
            integer_vars,
            ub,
        );
        objective_terms.push((product, term.coeff));
        add_mixed_row(
            rows,
            format!("__quad_obj_{k}__product_le_left"),
            expansions,
            &[(term.var_i, -1.0)],
            &[(product, 1.0)],
            RowSense::Le,
            0.0,
        );
        add_mixed_row(
            rows,
            format!("__quad_obj_{k}__product_le_right"),
            expansions,
            &[(term.var_j, -1.0)],
            &[(product, 1.0)],
            RowSense::Le,
            0.0,
        );
        add_mixed_row(
            rows,
            format!("__quad_obj_{k}__product_ge_pair"),
            expansions,
            &[(term.var_i, 1.0), (term.var_j, 1.0)],
            &[(product, -1.0)],
            RowSense::Le,
            1.0,
        );
    }
    Ok(objective_terms)
}

fn add_general_constraint_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    general: &GeneralConstraint,
) -> Result<(), MathProgramError> {
    match general {
        GeneralConstraint::BinaryAnd {
            name,
            result_var,
            operands,
        } => {
            for &operand in operands {
                add_program_row(
                    rows,
                    format!("{name}__and_result_le_{operand}"),
                    expansions,
                    &[(*result_var, 1.0), (operand, -1.0)],
                    RowSense::Le,
                    0.0,
                );
            }
            let mut coeffs = operands.iter().map(|&idx| (idx, 1.0)).collect::<Vec<_>>();
            coeffs.push((*result_var, -1.0));
            add_program_row(
                rows,
                format!("{name}__and_result_ge_sum"),
                expansions,
                &coeffs,
                RowSense::Le,
                operands.len() as f64 - 1.0,
            );
        }
        GeneralConstraint::BinaryOr {
            name,
            result_var,
            operands,
        } => {
            for &operand in operands {
                add_program_row(
                    rows,
                    format!("{name}__or_result_ge_{operand}"),
                    expansions,
                    &[(operand, 1.0), (*result_var, -1.0)],
                    RowSense::Le,
                    0.0,
                );
            }
            let mut coeffs = vec![(*result_var, 1.0)];
            coeffs.extend(operands.iter().map(|&idx| (idx, -1.0)));
            add_program_row(
                rows,
                format!("{name}__or_result_le_sum"),
                expansions,
                &coeffs,
                RowSense::Le,
                0.0,
            );
        }
        GeneralConstraint::Abs {
            name,
            result_var,
            operand_var,
        } => {
            add_abs_rows(
                program,
                names,
                integer_vars,
                ub,
                rows,
                expansions,
                name,
                *result_var,
                *operand_var,
            )?;
        }
        GeneralConstraint::Max {
            name,
            result_var,
            operands,
        } => add_extreme_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *result_var,
            operands,
            true,
        )?,
        GeneralConstraint::Min {
            name,
            result_var,
            operands,
        } => add_extreme_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *result_var,
            operands,
            false,
        )?,
        GeneralConstraint::PiecewiseLinear {
            name,
            x_var,
            y_var,
            points,
        } => add_piecewise_linear_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *x_var,
            *y_var,
            points,
        )?,
        GeneralConstraint::AllDifferent { name, variables } => add_all_different_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
        )?,
        GeneralConstraint::AllowedAssignments {
            name,
            variables,
            tuples,
        } => add_allowed_assignment_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            tuples,
        )?,
        GeneralConstraint::NoOverlap { name, intervals } => {
            add_no_overlap_rows(names, integer_vars, ub, rows, expansions, name, intervals)?
        }
        GeneralConstraint::NoOverlap2D {
            name,
            x_intervals,
            y_intervals,
        } => add_no_overlap_2d_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            x_intervals,
            y_intervals,
        )?,
        GeneralConstraint::Cumulative {
            name,
            intervals,
            demands,
            capacity,
        } => add_cumulative_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            intervals,
            demands,
            *capacity,
        )?,
    }
    Ok(())
}

fn add_all_different_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
) -> Result<(), MathProgramError> {
    let mut value_literals = BTreeMap::<i64, Vec<usize>>::new();

    for &var_idx in variables {
        let var = &program.variables[var_idx];
        let (lower, upper) = integer_bounds(var).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "all-different variable `{}` requires finite integer bounds",
                var.name
            ))
        })?;
        let mut literals = Vec::new();
        for value in lower..=upper {
            let lit = push_canonical_var(
                &format!("{name}__{}__eq_{value}", var.name),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            literals.push((value, lit));
            value_literals.entry(value).or_default().push(lit);
        }

        let choose_coeffs = literals
            .iter()
            .map(|&(_, lit)| (lit, 1.0))
            .collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__{}__choose_one", var.name),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__{}__choose_one_ge", var.name),
        });

        let mut link_coeffs = expansions[var_idx].terms.clone();
        link_coeffs.extend(literals.iter().map(|&(value, lit)| (lit, -(value as f64))));
        let link_rhs = -expansions[var_idx].constant;
        rows.push(SparseRow {
            coeffs: combine_terms(&link_coeffs),
            rhs: link_rhs,
            name: format!("{name}__{}__link_value", var.name),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&combine_terms(&link_coeffs)),
            rhs: -link_rhs,
            name: format!("{name}__{}__link_value_ge", var.name),
        });
    }

    for (value, literals) in value_literals {
        if literals.len() > 1 {
            rows.push(SparseRow {
                coeffs: literals.iter().map(|&lit| (lit, 1.0)).collect(),
                rhs: 1.0,
                name: format!("{name}__value_{value}__at_most_one"),
            });
        }
    }

    Ok(())
}

fn add_allowed_assignment_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    tuples: &[Vec<i64>],
) -> Result<(), MathProgramError> {
    let selectors = tuples
        .iter()
        .enumerate()
        .map(|(k, _)| {
            push_canonical_var(
                &format!("{name}__tuple_{k}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();

    let choose_coeffs = selectors.iter().map(|&lit| (lit, 1.0)).collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: choose_coeffs.clone(),
        rhs: 1.0,
        name: format!("{name}__choose_one_tuple"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&choose_coeffs),
        rhs: -1.0,
        name: format!("{name}__choose_one_tuple_ge"),
    });

    for (col, &var_idx) in variables.iter().enumerate() {
        let mut link_coeffs = expansions[var_idx].terms.clone();
        link_coeffs.extend(
            tuples
                .iter()
                .zip(&selectors)
                .map(|(tuple, &lit)| (lit, -(tuple[col] as f64))),
        );
        let link_rhs = -expansions[var_idx].constant;
        let link_coeffs = combine_terms(&link_coeffs);
        rows.push(SparseRow {
            coeffs: link_coeffs.clone(),
            rhs: link_rhs,
            name: format!("{name}__var_{col}__link_tuple"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&link_coeffs),
            rhs: -link_rhs,
            name: format!("{name}__var_{col}__link_tuple_ge"),
        });
    }

    Ok(())
}

fn add_extreme_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    result_var: usize,
    operands: &[usize],
    is_max: bool,
) -> Result<(), MathProgramError> {
    let result_bounds = variable_bounds(&program.variables[result_var]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "extreme result `{}` requires finite bounds",
            program.variables[result_var].name
        ))
    })?;
    let selectors = operands
        .iter()
        .enumerate()
        .map(|(k, _)| {
            push_canonical_var(
                &format!("{name}__choice_{k}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: selectors.iter().map(|&z| (z, 1.0)).collect(),
        rhs: 1.0,
        name: format!("{name}__choose_one"),
    });

    for (k, &operand) in operands.iter().enumerate() {
        let operand_bounds = variable_bounds(&program.variables[operand]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "extreme operand `{}` requires finite bounds",
                program.variables[operand].name
            ))
        })?;
        if is_max {
            add_program_row(
                rows,
                format!("{name}__max_ge_{k}"),
                expansions,
                &[(operand, 1.0), (result_var, -1.0)],
                RowSense::Le,
                0.0,
            );
            let big_m = 0.0_f64.max(result_bounds.1 - operand_bounds.0);
            add_mixed_row(
                rows,
                format!("{name}__max_select_{k}"),
                expansions,
                &[(result_var, 1.0), (operand, -1.0)],
                &[(selectors[k], big_m)],
                RowSense::Le,
                big_m,
            );
        } else {
            add_program_row(
                rows,
                format!("{name}__min_le_{k}"),
                expansions,
                &[(result_var, 1.0), (operand, -1.0)],
                RowSense::Le,
                0.0,
            );
            let big_m = 0.0_f64.max(operand_bounds.1 - result_bounds.0);
            add_mixed_row(
                rows,
                format!("{name}__min_select_{k}"),
                expansions,
                &[(operand, 1.0), (result_var, -1.0)],
                &[(selectors[k], big_m)],
                RowSense::Le,
                big_m,
            );
        }
    }
    Ok(())
}

fn add_piecewise_linear_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    x_var: usize,
    y_var: usize,
    points: &[(f64, f64)],
) -> Result<(), MathProgramError> {
    let lambdas = points
        .iter()
        .enumerate()
        .map(|(k, _)| {
            push_canonical_var(
                &format!("{name}__lambda_{k}"),
                false,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: lambdas.iter().map(|&lambda| (lambda, 1.0)).collect(),
        rhs: 1.0,
        name: format!("{name}__lambda_sum"),
    });

    let x_terms = points
        .iter()
        .zip(&lambdas)
        .map(|(&(x, _), &lambda)| (lambda, -x))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__x_link"),
        expansions,
        &[(x_var, 1.0)],
        &x_terms,
        RowSense::Eq,
        0.0,
    );
    let y_terms = points
        .iter()
        .zip(&lambdas)
        .map(|(&(_, y), &lambda)| (lambda, -y))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__y_link"),
        expansions,
        &[(y_var, 1.0)],
        &y_terms,
        RowSense::Eq,
        0.0,
    );

    let intervals = (0..points.len() - 1)
        .map(|k| {
            push_canonical_var(
                &format!("{name}__segment_{k}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: intervals.iter().map(|&z| (z, 1.0)).collect(),
        rhs: 1.0,
        name: format!("{name}__segment_sum"),
    });
    for (i, &lambda) in lambdas.iter().enumerate() {
        let mut coeffs = vec![(lambda, 1.0)];
        if i > 0 {
            coeffs.push((intervals[i - 1], -1.0));
        }
        if i + 1 < points.len() {
            coeffs.push((intervals[i], -1.0));
        }
        rows.push(SparseRow {
            coeffs,
            rhs: 0.0,
            name: format!("{name}__lambda_adjacent_{i}"),
        });
    }
    Ok(())
}

fn add_no_overlap_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    intervals: &[IntervalTerm],
) -> Result<(), MathProgramError> {
    for (i, interval) in intervals.iter().enumerate() {
        add_interval_link_rows(
            ub,
            rows,
            expansions,
            &format!("{name}__interval_{i}"),
            interval,
        )?;
    }

    for i in 0..intervals.len() {
        for j in (i + 1)..intervals.len() {
            let order = push_canonical_var(
                &format!("{name}__order_{i}_before_{j}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            let left = &intervals[i];
            let right = &intervals[j];
            let mut presence_literals = Vec::new();
            if let Some(presence) = left.presence_var {
                presence_literals.push((presence, true));
            }
            if let Some(presence) = right.presence_var {
                presence_literals.push((presence, true));
            }
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__no_overlap_{i}_before_{j}"),
                &[(left.start_var, 1.0), (right.start_var, -1.0)],
                -left.duration,
                &presence_literals,
                &[(order, true)],
                ub,
            )?;
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__no_overlap_{j}_before_{i}"),
                &[(right.start_var, 1.0), (left.start_var, -1.0)],
                -right.duration,
                &presence_literals,
                &[(order, false)],
                ub,
            )?;
        }
    }
    Ok(())
}

fn rectangle_presence_literals(
    x_interval: &IntervalTerm,
    y_interval: &IntervalTerm,
) -> Vec<(usize, bool)> {
    let mut literals = Vec::new();
    if let Some(presence) = x_interval.presence_var {
        literals.push((presence, true));
    }
    if let Some(presence) = y_interval.presence_var {
        if !literals.iter().any(|&(idx, _)| idx == presence) {
            literals.push((presence, true));
        }
    }
    literals
}

fn add_no_overlap_2d_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    x_intervals: &[IntervalTerm],
    y_intervals: &[IntervalTerm],
) -> Result<(), MathProgramError> {
    for (i, (x_interval, y_interval)) in x_intervals.iter().zip(y_intervals).enumerate() {
        add_interval_link_rows(
            ub,
            rows,
            expansions,
            &format!("{name}__x_interval_{i}"),
            x_interval,
        )?;
        add_interval_link_rows(
            ub,
            rows,
            expansions,
            &format!("{name}__y_interval_{i}"),
            y_interval,
        )?;
    }

    for i in 0..x_intervals.len() {
        for j in (i + 1)..x_intervals.len() {
            let separators = [
                push_canonical_var(
                    &format!("{name}__rect_{i}_left_of_{j}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                ),
                push_canonical_var(
                    &format!("{name}__rect_{j}_left_of_{i}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                ),
                push_canonical_var(
                    &format!("{name}__rect_{i}_below_{j}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                ),
                push_canonical_var(
                    &format!("{name}__rect_{j}_below_{i}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                ),
            ];
            let mut active_literals = rectangle_presence_literals(&x_intervals[i], &y_intervals[i]);
            active_literals.extend(rectangle_presence_literals(
                &x_intervals[j],
                &y_intervals[j],
            ));
            active_literals.sort_unstable_by_key(|&(idx, _)| idx);
            active_literals.dedup_by_key(|literal| literal.0);

            let selector_terms = separators
                .iter()
                .map(|&selector| (selector, 1.0))
                .collect::<Vec<_>>();
            let presence_terms = active_literals
                .iter()
                .map(|&(presence, _)| (presence, -1.0))
                .collect::<Vec<_>>();
            add_mixed_row(
                rows,
                format!("{name}__rectangles_{i}_{j}__choose_separator"),
                expansions,
                &presence_terms,
                &selector_terms,
                RowSense::Ge,
                1.0 - active_literals.len() as f64,
            );

            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{i}_left_of_{j}__enforce"),
                &[
                    (x_intervals[i].start_var, 1.0),
                    (x_intervals[j].start_var, -1.0),
                ],
                -x_intervals[i].duration,
                &active_literals,
                &[(separators[0], true)],
                ub,
            )?;
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{j}_left_of_{i}__enforce"),
                &[
                    (x_intervals[j].start_var, 1.0),
                    (x_intervals[i].start_var, -1.0),
                ],
                -x_intervals[j].duration,
                &active_literals,
                &[(separators[1], true)],
                ub,
            )?;
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{i}_below_{j}__enforce"),
                &[
                    (y_intervals[i].start_var, 1.0),
                    (y_intervals[j].start_var, -1.0),
                ],
                -y_intervals[i].duration,
                &active_literals,
                &[(separators[2], true)],
                ub,
            )?;
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{j}_below_{i}__enforce"),
                &[
                    (y_intervals[j].start_var, 1.0),
                    (y_intervals[i].start_var, -1.0),
                ],
                -y_intervals[j].duration,
                &active_literals,
                &[(separators[3], true)],
                ub,
            )?;
        }
    }
    Ok(())
}

fn add_cumulative_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    intervals: &[IntervalTerm],
    demands: &[f64],
    capacity: f64,
) -> Result<(), MathProgramError> {
    let mut start_choices = Vec::new();
    let mut min_time = i64::MAX;
    let mut max_time = i64::MIN;

    for (i, interval) in intervals.iter().enumerate() {
        add_interval_link_rows(
            ub,
            rows,
            expansions,
            &format!("{name}__interval_{i}"),
            interval,
        )?;
        let (start_lb, start_ub) = integer_bounds(&program.variables[interval.start_var])
            .ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "cumulative interval {i} start requires finite integer bounds"
                ))
            })?;
        let duration = interval.duration.round() as i64;
        let choices = (start_lb..=start_ub)
            .map(|t| {
                push_canonical_var(
                    &format!("{name}__interval_{i}__starts_at_{t}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                )
            })
            .collect::<Vec<_>>();

        let choose_terms = choices
            .iter()
            .map(|&choice| (choice, 1.0))
            .collect::<Vec<_>>();
        if let Some(presence) = interval.presence_var {
            add_mixed_row(
                rows,
                format!("{name}__interval_{i}__choose_if_present"),
                expansions,
                &[(presence, -1.0)],
                &choose_terms,
                RowSense::Eq,
                0.0,
            );
        } else {
            add_mixed_row(
                rows,
                format!("{name}__interval_{i}__choose_one_start"),
                expansions,
                &[],
                &choose_terms,
                RowSense::Eq,
                1.0,
            );
        }

        let start_sum = (start_lb..=start_ub)
            .zip(&choices)
            .map(|(t, &choice)| (choice, t as f64))
            .collect::<Vec<_>>();
        let canonical_start_sum = start_sum
            .iter()
            .map(|&(idx, coef)| (idx, -coef))
            .collect::<Vec<_>>();
        if let Some(presence) = interval.presence_var {
            add_implied_mixed_le_row(
                rows,
                expansions,
                format!("{name}__interval_{i}__start_link_upper"),
                &[(interval.start_var, 1.0)],
                &canonical_start_sum,
                0.0,
                &[(presence, true)],
                &[],
                ub,
            )?;
            let reverse_start_sum = start_sum
                .iter()
                .map(|&(idx, coef)| (idx, coef))
                .collect::<Vec<_>>();
            add_implied_mixed_le_row(
                rows,
                expansions,
                format!("{name}__interval_{i}__start_link_lower"),
                &[(interval.start_var, -1.0)],
                &reverse_start_sum,
                0.0,
                &[(presence, true)],
                &[],
                ub,
            )?;
        } else {
            add_mixed_row(
                rows,
                format!("{name}__interval_{i}__start_link"),
                expansions,
                &[(interval.start_var, 1.0)],
                &canonical_start_sum,
                RowSense::Eq,
                0.0,
            );
        }

        min_time = min_time.min(start_lb);
        max_time = max_time.max(start_ub + duration);
        start_choices.push((start_lb, duration, choices));
    }

    for t in min_time..max_time {
        let mut coeffs = Vec::new();
        for (i, (start_lb, duration, choices)) in start_choices.iter().enumerate() {
            if demands[i].abs() <= 1e-12 {
                continue;
            }
            for (offset, &choice) in choices.iter().enumerate() {
                let start_time = *start_lb + offset as i64;
                if start_time <= t && t < start_time + *duration {
                    coeffs.push((choice, demands[i]));
                }
            }
        }
        if !coeffs.is_empty() {
            rows.push(SparseRow {
                coeffs: combine_terms(&coeffs),
                rhs: capacity,
                name: format!("{name}__capacity_at_{t}"),
            });
        }
    }
    Ok(())
}

fn add_interval_link_rows(
    ub: &[f64],
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    interval: &IntervalTerm,
) -> Result<(), MathProgramError> {
    let upper_terms = &[(interval.end_var, 1.0), (interval.start_var, -1.0)];
    if let Some(presence) = interval.presence_var {
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__end_after_start_upper"),
            upper_terms,
            interval.duration,
            &[(presence, true)],
            &[],
            ub,
        )?;
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__end_after_start_lower"),
            &[(interval.start_var, 1.0), (interval.end_var, -1.0)],
            -interval.duration,
            &[(presence, true)],
            &[],
            ub,
        )?;
    } else {
        add_program_row(
            rows,
            format!("{name}__end_after_start"),
            expansions,
            upper_terms,
            RowSense::Eq,
            interval.duration,
        );
    }
    Ok(())
}

fn add_implied_le_row(
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: String,
    coeffs: &[(usize, f64)],
    rhs: f64,
    program_literals: &[(usize, bool)],
    canonical_literals: &[(usize, bool)],
    ub: &[f64],
) -> Result<(), MathProgramError> {
    add_implied_mixed_le_row(
        rows,
        expansions,
        name,
        coeffs,
        &[],
        rhs,
        program_literals,
        canonical_literals,
        ub,
    )
}

fn add_implied_mixed_le_row(
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: String,
    coeffs: &[(usize, f64)],
    canonical_coeffs: &[(usize, f64)],
    rhs: f64,
    program_literals: &[(usize, bool)],
    canonical_literals: &[(usize, bool)],
    ub: &[f64],
) -> Result<(), MathProgramError> {
    let mut all_coeffs = coeffs.to_vec();
    let mut all_canonical_coeffs = canonical_coeffs.to_vec();
    let (expanded, shifted_rhs) = expand_row(expansions, coeffs, rhs);
    let mut lhs_bounds_terms = expanded.clone();
    lhs_bounds_terms.extend_from_slice(canonical_coeffs);
    let max_lhs = canonical_linear_upper_bound(&lhs_bounds_terms, ub).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!("{name} requires finite bounds for implication"))
    })?;
    let big_m = 0.0_f64.max(max_lhs - shifted_rhs);
    let mut implied_rhs = rhs;

    for &(literal, active) in program_literals {
        if active {
            all_coeffs.push((literal, big_m));
            implied_rhs += big_m;
        } else {
            all_coeffs.push((literal, -big_m));
        }
    }
    for &(literal, active) in canonical_literals {
        if active {
            all_canonical_coeffs.push((literal, big_m));
            implied_rhs += big_m;
        } else {
            all_canonical_coeffs.push((literal, -big_m));
        }
    }
    add_mixed_row(
        rows,
        name,
        expansions,
        &all_coeffs,
        &all_canonical_coeffs,
        RowSense::Le,
        implied_rhs,
    );
    Ok(())
}

fn add_abs_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    result_var: usize,
    operand_var: usize,
) -> Result<(), MathProgramError> {
    let (lower, upper) = variable_bounds(&program.variables[operand_var]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "abs `{name}` operand `{}` requires finite bounds",
            program.variables[operand_var].name
        ))
    })?;
    if lower >= 0.0 {
        add_program_row(
            rows,
            format!("{name}__abs_nonnegative"),
            expansions,
            &[(result_var, 1.0), (operand_var, -1.0)],
            RowSense::Eq,
            0.0,
        );
        return Ok(());
    }
    if upper <= 0.0 {
        add_program_row(
            rows,
            format!("{name}__abs_nonpositive"),
            expansions,
            &[(result_var, 1.0), (operand_var, 1.0)],
            RowSense::Eq,
            0.0,
        );
        return Ok(());
    }

    let z = push_canonical_var(
        &format!("{name}__abs_positive"),
        true,
        1.0,
        names,
        integer_vars,
        ub,
    );
    add_program_row(
        rows,
        format!("{name}__abs_ge_x"),
        expansions,
        &[(operand_var, 1.0), (result_var, -1.0)],
        RowSense::Le,
        0.0,
    );
    add_program_row(
        rows,
        format!("{name}__abs_ge_neg_x"),
        expansions,
        &[(operand_var, -1.0), (result_var, -1.0)],
        RowSense::Le,
        0.0,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_x_upper_branch"),
        expansions,
        &[(operand_var, 1.0)],
        &[(z, -upper)],
        RowSense::Le,
        0.0,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_x_lower_branch"),
        expansions,
        &[(operand_var, -1.0)],
        &[(z, -lower)],
        RowSense::Le,
        -lower,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_result_pos_branch"),
        expansions,
        &[(result_var, 1.0), (operand_var, -1.0)],
        &[(z, -2.0 * lower)],
        RowSense::Le,
        -2.0 * lower,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_result_neg_branch"),
        expansions,
        &[(result_var, 1.0), (operand_var, 1.0)],
        &[(z, -2.0 * upper)],
        RowSense::Le,
        0.0,
    );
    Ok(())
}

fn add_mixed_row(
    rows: &mut Vec<SparseRow>,
    name: String,
    expansions: &[LinearExpansion],
    coeffs: &[(usize, f64)],
    canonical_coeffs: &[(usize, f64)],
    sense: RowSense,
    rhs: f64,
) {
    let (mut expanded, shifted_rhs) = expand_row(expansions, coeffs, rhs);
    expanded.extend_from_slice(canonical_coeffs);
    match sense {
        RowSense::Le => rows.push(SparseRow {
            coeffs: combine_terms(&expanded),
            rhs: shifted_rhs,
            name,
        }),
        RowSense::Ge => rows.push(SparseRow {
            coeffs: combine_terms(&negate_sparse(&expanded)),
            rhs: -shifted_rhs,
            name,
        }),
        RowSense::Eq => {
            rows.push(SparseRow {
                coeffs: combine_terms(&expanded),
                rhs: shifted_rhs,
                name: format!("{name}__eq_le"),
            });
            rows.push(SparseRow {
                coeffs: combine_terms(&negate_sparse(&expanded)),
                rhs: -shifted_rhs,
                name: format!("{name}__eq_ge"),
            });
        }
    }
}

fn expand_row(
    expansions: &[LinearExpansion],
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> (Vec<(usize, f64)>, f64) {
    let mut constant = 0.0;
    let mut terms = BTreeMap::<usize, f64>::new();
    for &(var_idx, coef) in coeffs {
        let expansion = &expansions[var_idx];
        constant += coef * expansion.constant;
        for &(canon_idx, canon_coef) in &expansion.terms {
            *terms.entry(canon_idx).or_insert(0.0) += coef * canon_coef;
        }
    }
    let sparse = terms
        .into_iter()
        .filter(|(_, coef)| coef.abs() > 1e-12)
        .collect();
    (sparse, rhs - constant)
}

fn dense_row(n: usize, coeffs: &[(usize, f64)]) -> Vec<f64> {
    let mut row = vec![0.0; n];
    for &(i, value) in coeffs {
        row[i] += value;
    }
    row
}

fn scale_row(row: &[f64], scale: f64) -> Vec<f64> {
    row.iter().map(|v| scale * *v).collect()
}

fn scale_vec(values: &[f64], scale: f64) -> Vec<f64> {
    values.iter().map(|v| scale * *v).collect()
}

fn scale_matrix(matrix: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    matrix.iter().map(|row| scale_vec(row, scale)).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm2(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn eval_affine_term(term: &AffineTerm, x: &[f64]) -> f64 {
    term.constant
        + term
            .coeffs
            .iter()
            .map(|&(idx, coef)| coef * x[idx])
            .sum::<f64>()
}

fn scale_affine_term(term: &AffineTerm, scale: f64) -> AffineTerm {
    AffineTerm {
        coeffs: term
            .coeffs
            .iter()
            .map(|&(idx, coef)| (idx, scale * coef))
            .collect(),
        constant: scale * term.constant,
    }
}

fn add_affine_terms(left: &AffineTerm, right: &AffineTerm, right_scale: f64) -> AffineTerm {
    let mut coeffs = left.coeffs.clone();
    coeffs.extend(
        right
            .coeffs
            .iter()
            .map(|&(idx, coef)| (idx, right_scale * coef)),
    );
    AffineTerm {
        coeffs: combine_terms(&coeffs),
        constant: left.constant + right_scale * right.constant,
    }
}

fn eval_sparse_affine(coeffs: &[(usize, f64)], constant: f64, x: &[f64]) -> f64 {
    constant + coeffs.iter().map(|&(idx, coef)| coef * x[idx]).sum::<f64>()
}

fn eval_quadratic_terms(terms: &[QuadraticConstraintTerm], x: &[f64]) -> f64 {
    terms
        .iter()
        .map(|term| term.coeff * x[term.var_i] * x[term.var_j])
        .sum()
}

fn eval_quadratic_constraint_le(row: &QuadraticConstraint, x: &[f64]) -> f64 {
    let raw = eval_quadratic_terms(&row.quadratic_terms, x)
        + row
            .linear_terms
            .iter()
            .map(|&(idx, coef)| coef * x[idx])
            .sum::<f64>()
        - row.rhs;
    match row.sense {
        RowSense::Le => raw,
        RowSense::Ge => -raw,
        RowSense::Eq => raw.abs(),
    }
}

fn solution_max_violation(program: &MathProgram, x: &[f64], tol: f64) -> Option<f64> {
    if x.len() != program.variables.len() || x.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let semantic_tol = tol.max(1e-9);
    let mut max_violation: f64 = 0.0;
    for (var, &value) in program.variables.iter().zip(x) {
        max_violation = max_violation.max(variable_domain_violation(var, value));
    }
    for row in &program.constraints {
        let lhs = eval_sparse_affine(&row.coeffs, 0.0, x);
        max_violation = max_violation.max(row_sense_violation(lhs, row.sense, row.rhs));
    }
    for row in &program.quadratic_constraints {
        max_violation = max_violation.max(eval_quadratic_constraint_le(row, x).max(0.0));
    }
    for cone in &program.second_order_cones {
        let values = cone
            .terms
            .iter()
            .map(|term| eval_affine_term(term, x))
            .collect::<Vec<_>>();
        let rhs = eval_sparse_affine(&cone.rhs_coeffs, cone.rhs_constant, x);
        max_violation = max_violation.max((norm2(&values) - rhs).max(0.0));
    }
    for indicator in &program.indicators {
        if binary_truth(x[indicator.binary_var]) == indicator.active_value {
            let lhs = eval_sparse_affine(&indicator.coeffs, 0.0, x);
            max_violation =
                max_violation.max(row_sense_violation(lhs, indicator.sense, indicator.rhs));
        }
    }
    for sos in &program.sos {
        max_violation = max_violation.max(sos_violation(sos, x, semantic_tol));
    }
    for constraint in &program.general_constraints {
        max_violation =
            max_violation.max(general_constraint_violation(constraint, x, semantic_tol));
    }
    Some(max_violation)
}

fn row_sense_violation(lhs: f64, sense: RowSense, rhs: f64) -> f64 {
    match sense {
        RowSense::Le => (lhs - rhs).max(0.0),
        RowSense::Ge => (rhs - lhs).max(0.0),
        RowSense::Eq => (lhs - rhs).abs(),
    }
}

fn variable_domain_violation(var: &Variable, value: f64) -> f64 {
    let bound_violation = |lower: Option<f64>, upper: Option<f64>| {
        lower
            .map_or(0.0, |lb| (lb - value).max(0.0))
            .max(upper.map_or(0.0, |ub| (value - ub).max(0.0)))
    };
    match var.var_type {
        VariableType::Continuous => bound_violation(var.lb, var.ub),
        VariableType::Integer => bound_violation(var.lb, var.ub).max(integrality_violation(value)),
        VariableType::Binary => {
            bound_violation(Some(0.0), Some(1.0)).max(integrality_violation(value))
        }
        VariableType::SemiContinuous => semi_domain_violation(value, var.lb, var.ub, false),
        VariableType::SemiInteger => semi_domain_violation(value, var.lb, var.ub, true),
    }
}

fn semi_domain_violation(value: f64, lower: Option<f64>, upper: Option<f64>, integer: bool) -> f64 {
    let lb = lower.unwrap_or(0.0);
    let ub = upper.unwrap_or(f64::INFINITY);
    let zero_branch = value.abs();
    let interval_branch = (lb - value)
        .max(0.0)
        .max((value - ub).max(0.0))
        .max(if integer {
            integrality_violation(value)
        } else {
            0.0
        });
    zero_branch.min(interval_branch)
}

fn integrality_violation(value: f64) -> f64 {
    (value - value.round()).abs()
}

fn binary_truth(value: f64) -> bool {
    value >= 0.5
}

fn sos_violation(sos: &SOSConstraint, x: &[f64], tol: f64) -> f64 {
    let mut nonzero_positions = sos
        .members
        .iter()
        .enumerate()
        .filter_map(|(pos, &(idx, _))| (x[idx].abs() > tol).then_some(pos))
        .collect::<Vec<_>>();
    match sos.sos_type {
        SOSType::Sos1 => nonzero_positions.len().saturating_sub(1) as f64,
        SOSType::Sos2 => {
            if nonzero_positions.len() <= 1 {
                0.0
            } else {
                nonzero_positions.sort_unstable();
                let too_many = nonzero_positions.len().saturating_sub(2) as f64;
                let non_adjacent = nonzero_positions
                    .windows(2)
                    .any(|pair| pair[1] != pair[0] + 1);
                too_many.max(if non_adjacent { 1.0 } else { 0.0 })
            }
        }
    }
}

fn general_constraint_violation(constraint: &GeneralConstraint, x: &[f64], tol: f64) -> f64 {
    match constraint {
        GeneralConstraint::BinaryAnd {
            result_var,
            operands,
            ..
        } => {
            let expected = operands.iter().all(|&idx| binary_truth(x[idx])) as u8 as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::BinaryOr {
            result_var,
            operands,
            ..
        } => {
            let expected = operands.iter().any(|&idx| binary_truth(x[idx])) as u8 as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::Abs {
            result_var,
            operand_var,
            ..
        } => (x[*result_var] - x[*operand_var].abs()).abs(),
        GeneralConstraint::Max {
            result_var,
            operands,
            ..
        } => {
            let expected = operands
                .iter()
                .map(|&idx| x[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::Min {
            result_var,
            operands,
            ..
        } => {
            let expected = operands
                .iter()
                .map(|&idx| x[idx])
                .fold(f64::INFINITY, f64::min);
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::PiecewiseLinear {
            x_var,
            y_var,
            points,
            ..
        } => piecewise_violation(points, x[*x_var], x[*y_var], tol),
        GeneralConstraint::AllDifferent { variables, .. } => all_different_violation(variables, x),
        GeneralConstraint::AllowedAssignments {
            variables, tuples, ..
        } => allowed_assignments_violation(variables, tuples, x),
        GeneralConstraint::NoOverlap { intervals, .. } => no_overlap_violation(intervals, x, tol),
        GeneralConstraint::NoOverlap2D {
            x_intervals,
            y_intervals,
            ..
        } => no_overlap_2d_violation(x_intervals, y_intervals, x, tol),
        GeneralConstraint::Cumulative {
            intervals,
            demands,
            capacity,
            ..
        } => cumulative_violation(intervals, demands, *capacity, x, tol),
    }
}

fn piecewise_violation(points: &[(f64, f64)], x_value: f64, y_value: f64, tol: f64) -> f64 {
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x_value >= x0 - tol && x_value <= x1 + tol {
            let clamped_x = x_value.clamp(x0, x1);
            let alpha = (clamped_x - x0) / (x1 - x0);
            let expected_y = y0 + alpha * (y1 - y0);
            return (y_value - expected_y).abs();
        }
    }
    let x_range_violation = (points[0].0 - x_value)
        .max(0.0)
        .max((x_value - points[points.len() - 1].0).max(0.0));
    x_range_violation.max(1.0)
}

fn all_different_violation(variables: &[usize], x: &[f64]) -> f64 {
    let mut seen = Vec::new();
    for &idx in variables {
        let value = x[idx].round() as i64;
        if seen.contains(&value) {
            return 1.0;
        }
        seen.push(value);
    }
    0.0
}

fn allowed_assignments_violation(variables: &[usize], tuples: &[Vec<i64>], x: &[f64]) -> f64 {
    tuples
        .iter()
        .map(|tuple| {
            variables
                .iter()
                .zip(tuple)
                .map(|(&idx, &target)| (x[idx] - target as f64).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

fn rectangle_active(x_interval: &IntervalTerm, y_interval: &IntervalTerm, x: &[f64]) -> bool {
    interval_active(x_interval, x) && interval_active(y_interval, x)
}

fn interval_active(interval: &IntervalTerm, x: &[f64]) -> bool {
    interval
        .presence_var
        .map_or(true, |presence| binary_truth(x[presence]))
}

fn interval_end_violation(interval: &IntervalTerm, x: &[f64]) -> f64 {
    (x[interval.end_var] - x[interval.start_var] - interval.duration).abs()
}

fn no_overlap_violation(intervals: &[IntervalTerm], x: &[f64], tol: f64) -> f64 {
    let mut violation: f64 = intervals
        .iter()
        .filter(|interval| interval_active(interval, x))
        .map(|interval| interval_end_violation(interval, x))
        .fold(0.0_f64, f64::max);
    for i in 0..intervals.len() {
        if !interval_active(&intervals[i], x) {
            continue;
        }
        for j in (i + 1)..intervals.len() {
            if !interval_active(&intervals[j], x) {
                continue;
            }
            let left_start = x[intervals[i].start_var];
            let left_end = x[intervals[i].end_var];
            let right_start = x[intervals[j].start_var];
            let right_end = x[intervals[j].end_var];
            if left_end <= right_start + tol || right_end <= left_start + tol {
                continue;
            }
            violation = violation.max(left_end.min(right_end) - left_start.max(right_start));
        }
    }
    violation
}

fn no_overlap_2d_violation(
    x_intervals: &[IntervalTerm],
    y_intervals: &[IntervalTerm],
    x: &[f64],
    tol: f64,
) -> f64 {
    let x_end_violation = x_intervals
        .iter()
        .filter(|interval| interval_active(interval, x))
        .map(|interval| interval_end_violation(interval, x))
        .fold(0.0_f64, f64::max);
    let y_end_violation = y_intervals
        .iter()
        .filter(|interval| interval_active(interval, x))
        .map(|interval| interval_end_violation(interval, x))
        .fold(0.0_f64, f64::max);
    let mut violation = x_end_violation.max(y_end_violation);

    for i in 0..x_intervals.len() {
        if !rectangle_active(&x_intervals[i], &y_intervals[i], x) {
            continue;
        }
        for j in (i + 1)..x_intervals.len() {
            if !rectangle_active(&x_intervals[j], &y_intervals[j], x) {
                continue;
            }
            let x_left = x[x_intervals[i].start_var];
            let x_right = x[x_intervals[i].end_var];
            let x_other_left = x[x_intervals[j].start_var];
            let x_other_right = x[x_intervals[j].end_var];
            let y_bottom = x[y_intervals[i].start_var];
            let y_top = x[y_intervals[i].end_var];
            let y_other_bottom = x[y_intervals[j].start_var];
            let y_other_top = x[y_intervals[j].end_var];

            let separated = x_right <= x_other_left + tol
                || x_other_right <= x_left + tol
                || y_top <= y_other_bottom + tol
                || y_other_top <= y_bottom + tol;
            if separated {
                continue;
            }
            let x_overlap = x_right.min(x_other_right) - x_left.max(x_other_left);
            let y_overlap = y_top.min(y_other_top) - y_bottom.max(y_other_bottom);
            violation = violation.max(x_overlap.min(y_overlap));
        }
    }
    violation
}

fn cumulative_violation(
    intervals: &[IntervalTerm],
    demands: &[f64],
    capacity: f64,
    x: &[f64],
    tol: f64,
) -> f64 {
    let active = intervals
        .iter()
        .zip(demands)
        .filter(|(interval, _)| interval_active(interval, x))
        .collect::<Vec<_>>();
    let mut violation: f64 = active
        .iter()
        .map(|(interval, _)| interval_end_violation(interval, x))
        .fold(0.0_f64, f64::max);
    if active.is_empty() {
        return violation;
    }
    let min_start = active
        .iter()
        .map(|(interval, _)| x[interval.start_var].floor() as i64)
        .min()
        .unwrap();
    let max_end = active
        .iter()
        .map(|(interval, _)| x[interval.end_var].ceil() as i64)
        .max()
        .unwrap();
    for time in min_start..max_end {
        let time = time as f64;
        let load = active
            .iter()
            .filter(|(interval, _)| {
                x[interval.start_var] <= time + tol && time < x[interval.end_var] - tol
            })
            .map(|(_, &demand)| demand)
            .sum::<f64>();
        violation = violation.max((load - capacity).max(0.0));
    }
    violation
}

fn quadratic_constraint_gradient(row: &QuadraticConstraint, x: &[f64]) -> Vec<(usize, f64)> {
    let sign = match row.sense {
        RowSense::Le => 1.0,
        RowSense::Ge => -1.0,
        RowSense::Eq => 1.0,
    };
    let mut gradient = Vec::new();
    for term in &row.quadratic_terms {
        let coeff = sign * term.coeff;
        if term.var_i == term.var_j {
            gradient.push((term.var_i, 2.0 * coeff * x[term.var_i]));
        } else {
            gradient.push((term.var_i, coeff * x[term.var_j]));
            gradient.push((term.var_j, coeff * x[term.var_i]));
        }
    }
    gradient.extend(
        row.linear_terms
            .iter()
            .map(|&(idx, coef)| (idx, sign * coef)),
    );
    combine_terms(&gradient)
}

fn most_violated_soc(program: &MathProgram, x: &[f64]) -> Option<(usize, f64, Vec<f64>, f64)> {
    program
        .second_order_cones
        .iter()
        .enumerate()
        .filter_map(|(idx, cone)| {
            let values = cone
                .terms
                .iter()
                .map(|term| eval_affine_term(term, x))
                .collect::<Vec<_>>();
            let rhs = eval_sparse_affine(&cone.rhs_coeffs, cone.rhs_constant, x);
            let violation = norm2(&values) - rhs;
            (violation > 0.0).then_some((idx, violation, values, rhs))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

fn most_violated_quadratic_constraint(program: &MathProgram, x: &[f64]) -> Option<(usize, f64)> {
    program
        .quadratic_constraints
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            let violation = eval_quadratic_constraint_le(row, x);
            (violation > 0.0).then_some((idx, violation))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

fn soc_supporting_cut(cone: &SecondOrderConeConstraint, unit: &[f64]) -> (Vec<(usize, f64)>, f64) {
    let mut coeffs = Vec::new();
    let mut constant = 0.0;
    for (weight, term) in unit.iter().zip(&cone.terms) {
        constant += weight * term.constant;
        coeffs.extend(term.coeffs.iter().map(|&(idx, coef)| (idx, weight * coef)));
    }
    coeffs.extend(cone.rhs_coeffs.iter().map(|&(idx, coef)| (idx, -coef)));
    (combine_terms(&coeffs), cone.rhs_constant - constant)
}

fn quadratic_constraint_supporting_cut(
    row: &QuadraticConstraint,
    x: &[f64],
) -> (Vec<(usize, f64)>, f64) {
    let gradient = quadratic_constraint_gradient(row, x);
    let violation = eval_quadratic_constraint_le(row, x);
    let rhs = gradient
        .iter()
        .map(|&(idx, coef)| coef * x[idx])
        .sum::<f64>()
        - violation;
    (gradient, rhs)
}

fn quadratic_form(matrix: &[Vec<f64>], x: &[f64]) -> f64 {
    matrix.iter().zip(x).map(|(row, xi)| xi * dot(row, x)).sum()
}

fn min_symmetric_eigenvalue(mut matrix: Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 0.0;
    }
    for _ in 0..(100 * n * n).max(1) {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_offdiag = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let value = matrix[i][j].abs();
                if value > max_offdiag {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_offdiag <= 1e-10 {
            break;
        }
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = theta.cos();
        let s = theta.sin();
        for k in 0..n {
            if k != p && k != q {
                let akp = matrix[k][p];
                let akq = matrix[k][q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                matrix[k][p] = new_kp;
                matrix[p][k] = new_kp;
                matrix[k][q] = new_kq;
                matrix[q][k] = new_kq;
            }
        }
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
    }
    matrix
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .fold(f64::INFINITY, f64::min)
}

fn negate_sparse(coeffs: &[(usize, f64)]) -> Vec<(usize, f64)> {
    coeffs.iter().map(|&(i, value)| (i, -value)).collect()
}

fn combine_terms(coeffs: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut terms = BTreeMap::<usize, f64>::new();
    for &(idx, value) in coeffs {
        *terms.entry(idx).or_insert(0.0) += value;
    }
    terms
        .into_iter()
        .filter(|(_, value)| value.abs() > 1e-12)
        .collect()
}

fn canonical_binary_index(expansion: &LinearExpansion) -> Option<usize> {
    if expansion.constant.abs() <= 1e-12
        && expansion.terms.len() == 1
        && (expansion.terms[0].1 - 1.0).abs() <= 1e-12
    {
        Some(expansion.terms[0].0)
    } else {
        None
    }
}

fn canonical_linear_upper_bound(coeffs: &[(usize, f64)], ub: &[f64]) -> Option<f64> {
    let mut upper = 0.0;
    for &(idx, coef) in coeffs {
        if idx >= ub.len() {
            return None;
        }
        if coef > 0.0 {
            let var_ub = ub[idx];
            if !var_ub.is_finite() {
                return None;
            }
            upper += coef * var_ub;
        }
    }
    Some(upper)
}

fn variable_bounds(var: &Variable) -> Option<(f64, f64)> {
    match var.var_type {
        VariableType::Binary => Some((0.0, 1.0)),
        VariableType::SemiContinuous | VariableType::SemiInteger => Some((0.0, var.ub?)),
        VariableType::Continuous | VariableType::Integer => Some((var.lb?, var.ub?)),
    }
}

fn integer_bounds(var: &Variable) -> Option<(i64, i64)> {
    if !is_integer_time_var(var) {
        return None;
    }
    let (lower, upper) = variable_bounds(var)?;
    let lower = lower.ceil();
    let upper = upper.floor();
    if !lower.is_finite()
        || !upper.is_finite()
        || lower < i64::MIN as f64
        || upper > i64::MAX as f64
        || upper < lower
    {
        return None;
    }
    Some((lower as i64, upper as i64))
}

fn is_integer_time_var(var: &Variable) -> bool {
    matches!(var.var_type, VariableType::Binary | VariableType::Integer)
}

fn supports_native_nonlinear_var(var_type: VariableType) -> bool {
    matches!(
        var_type,
        VariableType::Continuous | VariableType::Integer | VariableType::Binary
    )
}

fn can_encode_direct_mixed_integer_nonlinear(program: &MathProgram) -> bool {
    program.indicators.is_empty()
        && program.sos.is_empty()
        && program.general_constraints.is_empty()
        && program
            .variables
            .iter()
            .all(|var| supports_native_nonlinear_var(var.var_type))
}

fn quadratic_objective_has_native_nonlinear_terms(program: &MathProgram) -> bool {
    program
        .quadratic_objective
        .iter()
        .any(|term| !is_binary_binary_objective_term(program, term))
}

fn is_binary_binary_objective_term(program: &MathProgram, term: &QuadraticObjectiveTerm) -> bool {
    program.variables[term.var_i].var_type == VariableType::Binary
        && program.variables[term.var_j].var_type == VariableType::Binary
}

fn quadratic_objective_epigraph_program(
    program: &MathProgram,
) -> Result<MathProgram, MathProgramError> {
    let nonlinear_terms = program
        .quadratic_objective
        .iter()
        .filter(|term| !is_binary_binary_objective_term(program, term))
        .cloned()
        .collect::<Vec<_>>();
    if nonlinear_terms.is_empty() {
        return Err(MathProgramError::Unsupported(
            "mixed-integer quadratic objective epigraph requires a non-binary quadratic term"
                .to_string(),
        ));
    }

    let (lower, upper) = quadratic_objective_interval_bounds(program, &nonlinear_terms)?;
    let mut transformed = program.clone();
    transformed.quadratic_objective = program
        .quadratic_objective
        .iter()
        .filter(|term| is_binary_binary_objective_term(program, term))
        .cloned()
        .collect();
    let epigraph = transformed.add_continuous_var(
        "__quadratic_objective_epigraph",
        1.0,
        Some(lower),
        Some(upper),
    )?;
    let sign = match program.sense {
        ObjectiveSense::Min => 1.0,
        ObjectiveSense::Max => -1.0,
    };
    transformed.add_quadratic_constraint(
        "__quadratic_objective_epigraph_row",
        nonlinear_terms
            .iter()
            .map(|term| (term.var_i, term.var_j, sign * term.coeff))
            .collect(),
        vec![(epigraph, -sign)],
        RowSense::Le,
        0.0,
    )?;
    Ok(transformed)
}

fn quadratic_objective_interval_bounds(
    program: &MathProgram,
    terms: &[QuadraticObjectiveTerm],
) -> Result<(f64, f64), MathProgramError> {
    let mut lower = 0.0;
    let mut upper = 0.0;
    for term in terms {
        let (term_lower, term_upper) = quadratic_objective_term_bounds(program, term)?;
        lower += term_lower;
        upper += term_upper;
    }
    Ok((lower, upper))
}

fn quadratic_objective_term_bounds(
    program: &MathProgram,
    term: &QuadraticObjectiveTerm,
) -> Result<(f64, f64), MathProgramError> {
    let (left_lower, left_upper) = finite_var_bounds(program, term.var_i)?;
    let (right_lower, right_upper) = finite_var_bounds(program, term.var_j)?;
    let mut products = if term.var_i == term.var_j {
        let mut values = vec![left_lower * left_lower, left_upper * left_upper];
        if left_lower <= 0.0 && left_upper >= 0.0 {
            values.push(0.0);
        }
        values
    } else {
        vec![
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ]
    };
    products.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let product_lower = products[0];
    let product_upper = products[products.len() - 1];
    if term.coeff >= 0.0 {
        Ok((term.coeff * product_lower, term.coeff * product_upper))
    } else {
        Ok((term.coeff * product_upper, term.coeff * product_lower))
    }
}

fn finite_var_bounds(
    program: &MathProgram,
    var_idx: usize,
) -> Result<(f64, f64), MathProgramError> {
    let var = &program.variables[var_idx];
    let lower = var.lb.ok_or_else(|| {
        MathProgramError::Unsupported(format!(
            "mixed-integer quadratic objective epigraph requires a finite lower bound for `{}`",
            var.name
        ))
    })?;
    let upper = var.ub.ok_or_else(|| {
        MathProgramError::Unsupported(format!(
            "mixed-integer quadratic objective epigraph requires a finite upper bound for `{}`",
            var.name
        ))
    })?;
    Ok((lower, upper))
}

fn is_integer_value(value: f64) -> bool {
    value.is_finite() && (value - value.round()).abs() <= 1e-9
}

fn linear_bounds(program: &MathProgram, coeffs: &[(usize, f64)]) -> Option<(f64, f64)> {
    let mut lo = 0.0;
    let mut hi = 0.0;
    for &(idx, coef) in coeffs {
        let (var_lo, var_hi) = variable_bounds(&program.variables[idx])?;
        if coef >= 0.0 {
            lo += coef * var_lo;
            hi += coef * var_hi;
        } else {
            lo += coef * var_hi;
            hi += coef * var_lo;
        }
    }
    Some((lo, hi))
}

fn eval_expansion(expansion: &LinearExpansion, canonical_x: &[f64]) -> f64 {
    expansion.constant
        + expansion
            .terms
            .iter()
            .map(|&(idx, coef)| coef * canonical_x.get(idx).copied().unwrap_or(0.0))
            .sum::<f64>()
}

fn objective_value(program: &MathProgram, x: &[f64]) -> f64 {
    let linear = program
        .variables
        .iter()
        .zip(x)
        .map(|(var, value)| var.obj * value)
        .sum::<f64>();
    let quadratic = program
        .quadratic_objective
        .iter()
        .map(|term| term.coeff * x[term.var_i] * x[term.var_j])
        .sum::<f64>();
    linear + quadratic
}

fn quadratic_gradient(program: &MathProgram, x: &[f64]) -> Vec<f64> {
    let mut gradient = program
        .variables
        .iter()
        .map(|var| var.obj)
        .collect::<Vec<_>>();
    for term in &program.quadratic_objective {
        if term.var_i == term.var_j {
            gradient[term.var_i] += 2.0 * term.coeff * x[term.var_i];
        } else {
            gradient[term.var_i] += term.coeff * x[term.var_j];
            gradient[term.var_j] += term.coeff * x[term.var_i];
        }
    }
    gradient
}

fn quadratic_hessian(program: &MathProgram) -> Vec<Vec<f64>> {
    let n = program.variables.len();
    let mut hessian = vec![vec![0.0; n]; n];
    for term in &program.quadratic_objective {
        if term.var_i == term.var_j {
            hessian[term.var_i][term.var_i] += 2.0 * term.coeff;
        } else {
            hessian[term.var_i][term.var_j] += term.coeff;
            hessian[term.var_j][term.var_i] += term.coeff;
        }
    }
    hessian
}

fn quadratic_terms_hessian(n: usize, terms: &[QuadraticConstraintTerm]) -> Vec<Vec<f64>> {
    let mut hessian = vec![vec![0.0; n]; n];
    for term in terms {
        if term.var_i == term.var_j {
            hessian[term.var_i][term.var_i] += 2.0 * term.coeff;
        } else {
            hessian[term.var_i][term.var_j] += term.coeff;
            hessian[term.var_j][term.var_i] += term.coeff;
        }
    }
    hessian
}

fn validate_variable(var: &Variable) -> Result<(), MathProgramError> {
    if !var.obj.is_finite() {
        return Err(MathProgramError::NonFinite(format!(
            "objective for variable `{}`",
            var.name
        )));
    }
    if var.lb.is_some_and(|v| !v.is_finite()) || var.ub.is_some_and(|v| !v.is_finite()) {
        return Err(MathProgramError::NonFinite(format!(
            "bounds for variable `{}`",
            var.name
        )));
    }
    if let (Some(lb), Some(ub)) = (var.lb, var.ub) {
        if ub < lb {
            return Err(MathProgramError::InvalidBound(format!(
                "upper bound below lower bound for `{}`",
                var.name
            )));
        }
    }
    match var.var_type {
        VariableType::Binary => {
            if var.lb.is_some_and(|lb| lb < 0.0) || var.ub.is_some_and(|ub| ub > 1.0) {
                return Err(MathProgramError::InvalidBound(format!(
                    "binary variable `{}` must live inside [0, 1]",
                    var.name
                )));
            }
        }
        VariableType::SemiContinuous | VariableType::SemiInteger => {
            let lb = var.lb.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "semi-continuous variable `{}` requires finite lb",
                    var.name
                ))
            })?;
            let ub = var.ub.ok_or_else(|| {
                MathProgramError::InvalidBound(format!(
                    "semi-continuous variable `{}` requires finite ub",
                    var.name
                ))
            })?;
            if lb < 0.0 || ub < lb {
                return Err(MathProgramError::InvalidBound(format!(
                    "invalid semi-continuous bounds for `{}`",
                    var.name
                )));
            }
        }
        VariableType::Continuous | VariableType::Integer => {}
    }
    Ok(())
}

fn validate_coeffs(n: usize, coeffs: &[(usize, f64)]) -> Result<(), MathProgramError> {
    for &(idx, value) in coeffs {
        if idx >= n {
            return Err(MathProgramError::BadIndex(format!(
                "coefficient variable index {idx} out of bounds for {n} variables"
            )));
        }
        if !value.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "coefficient for variable index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_linear_objective_args(
    n: usize,
    weight: f64,
    abs_tol: f64,
    rel_tol: f64,
    coeffs: &[(usize, f64)],
) -> Result<(), MathProgramError> {
    if !weight.is_finite() || weight.abs() <= 1e-12 {
        return Err(MathProgramError::NonFinite(
            "secondary objective weight must be finite and non-zero".to_string(),
        ));
    }
    if !abs_tol.is_finite() || abs_tol < 0.0 {
        return Err(MathProgramError::InvalidBound(format!(
            "secondary objective absolute tolerance must be finite and non-negative, got {abs_tol}"
        )));
    }
    if !rel_tol.is_finite() || rel_tol < 0.0 {
        return Err(MathProgramError::InvalidBound(format!(
            "secondary objective relative tolerance must be finite and non-negative, got {rel_tol}"
        )));
    }
    validate_coeffs(n, coeffs)
}

fn validate_sos_members(
    n: usize,
    sos_type: SOSType,
    members: &[(usize, f64)],
) -> Result<(), MathProgramError> {
    let min_len = match sos_type {
        SOSType::Sos1 => 1,
        SOSType::Sos2 => 2,
    };
    if members.len() < min_len {
        return Err(MathProgramError::Unsupported(format!(
            "{sos_type:?} requires at least {min_len} member(s)"
        )));
    }
    let mut weights = Vec::new();
    for &(idx, weight) in members {
        if idx >= n {
            return Err(MathProgramError::BadIndex(format!(
                "SOS member index {idx} out of bounds for {n} variables"
            )));
        }
        if !weight.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "SOS member weight for variable index {idx}"
            )));
        }
        if weights
            .iter()
            .any(|seen: &f64| (*seen - weight).abs() <= 1e-12)
        {
            return Err(MathProgramError::Unsupported(format!(
                "{sos_type:?} weights must be unique"
            )));
        }
        weights.push(weight);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6, "left={a}, right={b}");
    }

    #[test]
    fn continuous_model_accepts_ge_eq_and_bounds() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 3.0, Some(0.0), Some(3.0))
            .unwrap();
        let y = p.add_continuous_var("y", 2.0, Some(0.0), None).unwrap();
        p.add_constraint("demand", vec![(x, 1.0), (y, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        p.add_constraint("capacity", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 4.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.objective, 11.0);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.x[y], 1.0);
    }

    #[test]
    fn conflict_refiner_keeps_irreducible_linear_rows() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_continuous_var("x", 5.0, None, None).unwrap();
        let y = p.add_continuous_var("y", 0.0, Some(0.0), None).unwrap();
        p.add_constraint("x-at-least-two", vec![(x, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        p.add_constraint("x-at-most-one", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        p.add_constraint("redundant-y", vec![(y, 1.0)], RowSense::Ge, 0.0)
            .unwrap();

        let conflict = refine_math_program_conflict(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramConflictOptions::default(),
        )
        .unwrap();

        assert_eq!(conflict.status, MathProgramStatus::Infeasible);
        assert!(conflict.minimal);
        assert_eq!(conflict.items.len(), 2);
        let row_names = conflict
            .items
            .iter()
            .filter_map(|item| match item {
                MathProgramConflictItem::LinearConstraint { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(row_names.contains(&"x-at-least-two"));
        assert!(row_names.contains(&"x-at-most-one"));
        assert!(!row_names.contains(&"redundant-y"));
    }

    #[test]
    fn conflict_refiner_reports_variable_bound_conflicts() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_continuous_var("x", 3.0, Some(2.0), None).unwrap();
        p.add_constraint("x-at-most-one", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        p.add_constraint("redundant-floor", vec![(x, 1.0)], RowSense::Ge, 0.0)
            .unwrap();

        let conflict = refine_math_program_conflict(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramConflictOptions::default(),
        )
        .unwrap();

        assert_eq!(conflict.status, MathProgramStatus::Infeasible);
        assert_eq!(conflict.items.len(), 2);
        assert!(conflict.items.iter().any(|item| matches!(
            item,
            MathProgramConflictItem::VariableLowerBound { var, name } if *var == x && name == "x"
        )));
        assert!(conflict.items.iter().any(|item| matches!(
            item,
            MathProgramConflictItem::LinearConstraint { name, .. } if name == "x-at-most-one"
        )));
        assert!(!conflict.items.iter().any(|item| matches!(
            item,
            MathProgramConflictItem::LinearConstraint { name, .. } if name == "redundant-floor"
        )));
    }

    #[test]
    fn feasibility_relaxation_prefers_cheaper_bound_violation() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_continuous_var("x", 0.0, Some(2.0), None).unwrap();
        p.add_constraint("cap", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();

        let relaxation = solve_math_program_feas_relaxation(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramFeasRelaxOptions {
                linear_penalty: 10.0,
                bound_penalty: 1.0,
                ..MathProgramFeasRelaxOptions::default()
            },
        )
        .unwrap();

        assert_eq!(relaxation.status, MathProgramStatus::Optimal);
        assert_close(relaxation.violation_objective, 1.0);
        assert_close(relaxation.x[x], 1.0);
        assert_eq!(relaxation.violations.len(), 1);
        assert!(matches!(
            &relaxation.violations[0],
            MathProgramFeasRelaxViolation::VariableLowerBound {
                var,
                name,
                violation,
                penalty,
            } if *var == x
                && name == "x"
                && (*violation - 1.0).abs() <= 1e-6
                && (*penalty - 1.0).abs() <= 1e-6
        ));
    }

    #[test]
    fn hierarchical_objective_refines_primary_optimum() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("budget", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 4.0)
            .unwrap();
        p.add_secondary_objective("prefer-y", ObjectiveSense::Min, 10, 1.0, vec![(x, 1.0)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.objective, 4.0);
        assert_close(sol.x[x], 0.0);
        assert_close(sol.x[y], 4.0);
    }

    #[test]
    fn integer_model_maps_shifted_lower_bounds_back() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(-2.0), Some(3.0)).unwrap();
        let y = p
            .add_continuous_var("y", 0.5, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("budget", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 2.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.x[y], 0.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn mip_start_maps_from_original_variable_space() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(-2.0), Some(3.0)).unwrap();
        let y = p
            .add_continuous_var("y", 0.5, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("budget", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 2.0)
            .unwrap();

        let sol = solve_math_program(
            &p,
            &MathProgramSolveOptions {
                mip_start: Some(vec![2.0, 0.0]),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.x[y], 0.0);
        assert!(sol
            .message
            .as_deref()
            .is_some_and(|message| message.contains("incumbent_source=mip-start")));
    }

    #[test]
    fn indicator_constraint_lowers_with_big_m() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        p.add_constraint("force-active", vec![(b, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_indicator(
            "cap-when-active",
            b,
            true,
            vec![(x, 1.0)],
            RowSense::Le,
            2.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 2.0);
    }

    #[test]
    fn semi_continuous_variable_is_zero_or_inside_interval() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_semi_continuous_var("x", 1.0, 5.0, 10.0).unwrap();
        p.add_constraint("must-produce", vec![(x, 1.0)], RowSense::Ge, 1.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 5.0);
        assert_close(sol.objective, 5.0);
    }

    #[test]
    fn semi_integer_variable_is_zero_or_integer_inside_interval() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_semi_integer_var("x", 1.0, 3.0, 7.0).unwrap();
        p.add_constraint("must-produce", vec![(x, 1.0)], RowSense::Ge, 1.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn sos1_allows_at_most_one_nonzero_member() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p
            .add_continuous_var("x0", 5.0, Some(0.0), Some(1.0))
            .unwrap();
        let x1 = p
            .add_continuous_var("x1", 7.0, Some(0.0), Some(1.0))
            .unwrap();
        let x2 = p
            .add_continuous_var("x2", 3.0, Some(0.0), Some(1.0))
            .unwrap();
        p.add_sos1("choose-one", vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x0], 0.0);
        assert_close(sol.x[x1], 1.0);
        assert_close(sol.x[x2], 0.0);
        assert_close(sol.objective, 7.0);
    }

    #[test]
    fn sos2_requires_adjacent_nonzero_members() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p
            .add_continuous_var("x0", 6.0, Some(0.0), Some(1.0))
            .unwrap();
        let x1 = p
            .add_continuous_var("x1", 1.0, Some(0.0), Some(1.0))
            .unwrap();
        let x2 = p
            .add_continuous_var("x2", 6.0, Some(0.0), Some(1.0))
            .unwrap();
        p.add_constraint(
            "pick-two",
            vec![(x0, 1.0), (x1, 1.0), (x2, 1.0)],
            RowSense::Eq,
            2.0,
        )
        .unwrap();
        p.add_sos2("adjacent-pair", vec![(x0, 1.0), (x1, 2.0), (x2, 3.0)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x1], 1.0);
        assert_close(sol.objective, 7.0);
        assert!((sol.x[x0] - 1.0).abs() <= 1e-6 || (sol.x[x2] - 1.0).abs() <= 1e-6);
    }

    #[test]
    fn binary_and_or_general_constraints_are_exact() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 0.0).unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        let both = p.add_binary_var("both", 2.0).unwrap();
        let either = p.add_binary_var("either", 1.0).unwrap();
        p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-b-off", vec![(b, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_binary_and("both-active", both, vec![a, b]).unwrap();
        p.add_binary_or("either-active", either, vec![a, b])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[both], 0.0);
        assert_close(sol.x[either], 1.0);
        assert_close(sol.objective, 1.0);
    }

    #[test]
    fn abs_general_constraint_lowers_with_binary_split() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(-5.0), Some(4.0))
            .unwrap();
        let r = p
            .add_continuous_var("abs_x", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -3.0)
            .unwrap();
        p.add_abs("absolute-value", r, x).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], -3.0);
        assert_close(sol.x[r], 3.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn max_general_constraint_selects_largest_operand() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a = p
            .add_continuous_var("a", 0.0, Some(-2.0), Some(5.0))
            .unwrap();
        let b = p
            .add_continuous_var("b", 0.0, Some(-2.0), Some(5.0))
            .unwrap();
        let r = p
            .add_continuous_var("max_ab", 1.0, Some(-2.0), Some(5.0))
            .unwrap();
        p.add_constraint("fix-a", vec![(a, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        p.add_constraint("fix-b", vec![(b, 1.0)], RowSense::Eq, -1.0)
            .unwrap();
        p.add_max("maximum", r, vec![a, b]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[r], 2.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn min_general_constraint_selects_smallest_operand() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p
            .add_continuous_var("a", 0.0, Some(-2.0), Some(5.0))
            .unwrap();
        let b = p
            .add_continuous_var("b", 0.0, Some(-2.0), Some(5.0))
            .unwrap();
        let r = p
            .add_continuous_var("min_ab", 1.0, Some(-2.0), Some(5.0))
            .unwrap();
        p.add_constraint("fix-a", vec![(a, 1.0)], RowSense::Eq, 4.0)
            .unwrap();
        p.add_constraint("fix-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_min("minimum", r, vec![a, b]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[r], 1.0);
        assert_close(sol.objective, 1.0);
    }

    #[test]
    fn piecewise_linear_uses_adjacent_breakpoints() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 1.5)
            .unwrap();
        p.add_piecewise_linear("square-ish", x, y, vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 1.5);
        assert_close(sol.x[y], 2.5);
        assert_close(sol.objective, 2.5);
    }

    #[test]
    fn all_different_lowers_to_assignment_literals() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p
            .add_integer_var("x0", 100.0, Some(0.0), Some(2.0))
            .unwrap();
        let x1 = p.add_integer_var("x1", 10.0, Some(0.0), Some(2.0)).unwrap();
        let x2 = p.add_integer_var("x2", 1.0, Some(0.0), Some(2.0)).unwrap();
        p.add_all_different("permute", vec![x0, x1, x2]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x0], 2.0);
        assert_close(sol.x[x1], 1.0);
        assert_close(sol.x[x2], 0.0);
        assert_close(sol.objective, 210.0);
    }

    #[test]
    fn allowed_assignments_lowers_to_tuple_selectors() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 10.0, Some(0.0), Some(2.0)).unwrap();
        let y = p.add_integer_var("y", 1.0, Some(0.0), Some(2.0)).unwrap();
        p.add_allowed_assignments("allowed-pairs", vec![x, y], vec![vec![0, 2], vec![1, 1]])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 1.0);
        assert_close(sol.x[y], 1.0);
        assert_close(sol.objective, 11.0);
    }

    #[test]
    fn no_overlap_orders_fixed_size_intervals() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_start = p
            .add_integer_var("a_start", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let a_end = p
            .add_integer_var("a_end", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let b_start = p
            .add_integer_var("b_start", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        let b_end = p
            .add_integer_var("b_end", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_no_overlap(
            "single-machine",
            vec![
                MathProgram::interval(a_start, 3.0, a_end),
                MathProgram::interval(b_start, 2.0, b_end),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_start], 0.0);
        assert_close(sol.x[a_end], 3.0);
        assert_close(sol.x[b_start], 3.0);
        assert_close(sol.x[b_end], 5.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn no_overlap_2d_packs_rectangles_on_another_axis() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_x_start = p
            .add_integer_var("a_x_start", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let a_x_end = p
            .add_integer_var("a_x_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let a_y_start = p
            .add_integer_var("a_y_start", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let a_y_end = p
            .add_integer_var("a_y_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let b_x_start = p
            .add_integer_var("b_x_start", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let b_x_end = p
            .add_integer_var("b_x_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let b_y_start = p
            .add_integer_var("b_y_start", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        let b_y_end = p
            .add_integer_var("b_y_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();

        p.add_constraint("fix-a-x-start", vec![(a_x_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_constraint("fix-a-y-start", vec![(a_y_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_constraint("fix-b-x-start", vec![(b_x_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_no_overlap_2d(
            "packing",
            vec![
                MathProgram::interval(a_x_start, 2.0, a_x_end),
                MathProgram::interval(b_x_start, 2.0, b_x_end),
            ],
            vec![
                MathProgram::interval(a_y_start, 2.0, a_y_end),
                MathProgram::interval(b_y_start, 2.0, b_y_end),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_x_start], 0.0);
        assert_close(sol.x[a_x_end], 2.0);
        assert_close(sol.x[a_y_start], 0.0);
        assert_close(sol.x[a_y_end], 2.0);
        assert_close(sol.x[b_x_start], 0.0);
        assert_close(sol.x[b_x_end], 2.0);
        assert_close(sol.x[b_y_start], 2.0);
        assert_close(sol.x[b_y_end], 4.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn cumulative_blocks_over_capacity_overlap() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_start = p
            .add_integer_var("a_start", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let a_end = p
            .add_integer_var("a_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let b_start = p
            .add_integer_var("b_start", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        let b_end = p
            .add_integer_var("b_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_cumulative(
            "shared-resource",
            vec![
                MathProgram::interval(a_start, 2.0, a_end),
                MathProgram::interval(b_start, 2.0, b_end),
            ],
            vec![2.0, 2.0],
            3.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_start], 0.0);
        assert_close(sol.x[a_end], 2.0);
        assert_close(sol.x[b_start], 2.0);
        assert_close(sol.x[b_end], 4.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn solution_pool_enumerates_top_binary_assignments() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 4.0).unwrap();
        let b = p.add_binary_var("b", 2.0).unwrap();
        let c = p.add_binary_var("c", 1.0).unwrap();
        p.add_constraint(
            "choose-at-most-two",
            vec![(a, 1.0), (b, 1.0), (c, 1.0)],
            RowSense::Le,
            2.0,
        )
        .unwrap();

        let pool = solve_math_program_solution_pool(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramSolutionPoolOptions {
                max_solutions: 3,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(pool.solutions.len(), 3);
        assert!(!pool.exhausted);
        assert_close(pool.solutions[0].objective, 6.0);
        assert_close(pool.solutions[1].objective, 5.0);
        assert_close(pool.solutions[2].objective, 4.0);
        assert_eq!(pool.solutions[0].x, vec![1.0, 1.0, 0.0]);
        assert_eq!(pool.solutions[1].x, vec![1.0, 0.0, 1.0]);
        assert_eq!(pool.solutions[2].x, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn binary_quadratic_objective_lowers_to_product_variable() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 3.0).unwrap();
        let b = p.add_binary_var("b", 3.0).unwrap();
        p.add_quadratic_objective_term(a, b, -4.0).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a] + sol.x[b], 1.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn continuous_convex_quadratic_objective_uses_qp_solver() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", -4.0, Some(0.0), Some(5.0))
            .unwrap();
        p.add_quadratic_objective_term(x, x, 1.0).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.objective, -4.0);
        assert_eq!(sol.solver, "des-frank-wolfe-qp");
    }

    #[test]
    fn mixed_integer_quadratic_objective_uses_epigraph_cuts() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_integer_var("x", -4.0, Some(0.0), Some(5.0)).unwrap();
        p.add_quadratic_objective_term(x, x, 1.0).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.objective, -4.0);
        assert_eq!(sol.solver, "des-mip-convex-qp-cutting-plane");
    }

    #[test]
    fn convex_quadratic_constraint_lowers_with_supporting_cuts() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(20.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_quadratic_constraint(
            "epigraph-square",
            vec![(x, x, 1.0)],
            vec![(y, -1.0)],
            RowSense::Le,
            0.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.x[y], 9.0);
        assert_close(sol.objective, 9.0);
        assert_eq!(sol.solver, "des-convex-cutting-plane");
    }

    #[test]
    fn second_order_cone_lowers_with_supporting_cuts() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(0.0), Some(3.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let t = p
            .add_continuous_var("t", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
            .unwrap();
        p.add_second_order_cone(
            "norm-bound",
            vec![
                MathProgram::affine_term(vec![(x, 1.0)], 0.0),
                MathProgram::affine_term(vec![(y, 1.0)], 0.0),
            ],
            vec![(t, 1.0)],
            0.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.x[y], 4.0);
        assert_close(sol.x[t], 5.0);
        assert_close(sol.objective, 5.0);
        assert_eq!(sol.solver, "des-soc-cutting-plane");
    }

    #[test]
    fn rotated_second_order_cone_lowers_to_standard_soc() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let u = p
            .add_continuous_var("u", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let v = p
            .add_continuous_var("v", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        let z = p
            .add_continuous_var("z", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-u", vec![(u, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        p.add_constraint("fix-z", vec![(z, 1.0)], RowSense::Eq, 4.0)
            .unwrap();
        p.add_rotated_second_order_cone(
            "rotated-energy",
            MathProgram::affine_term(vec![(u, 1.0)], 0.0),
            MathProgram::affine_term(vec![(v, 1.0)], 0.0),
            vec![MathProgram::affine_term(vec![(z, 1.0)], 0.0)],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[u], 2.0);
        assert_close(sol.x[v], 4.0);
        assert_close(sol.x[z], 4.0);
        assert_close(sol.objective, 4.0);
        assert_eq!(sol.solver, "des-soc-cutting-plane");
    }

    #[test]
    fn mixed_integer_quadratic_constraint_uses_cutting_planes() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_integer_var("x", 0.0, Some(0.0), Some(5.0)).unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(20.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_quadratic_constraint(
            "integer-square",
            vec![(x, x, 1.0)],
            vec![(y, -1.0)],
            RowSense::Le,
            0.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.x[y], 9.0);
        assert_close(sol.objective, 9.0);
        assert_eq!(sol.solver, "des-mip-convex-cutting-plane");
    }

    #[test]
    fn mixed_integer_second_order_cone_uses_cutting_planes() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_integer_var("x", 0.0, Some(0.0), Some(3.0)).unwrap();
        let y = p.add_integer_var("y", 0.0, Some(0.0), Some(4.0)).unwrap();
        let t = p
            .add_continuous_var("t", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
            .unwrap();
        p.add_second_order_cone(
            "integer-norm",
            vec![
                MathProgram::affine_term(vec![(x, 1.0)], 0.0),
                MathProgram::affine_term(vec![(y, 1.0)], 0.0),
            ],
            vec![(t, 1.0)],
            0.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.x[y], 4.0);
        assert_close(sol.x[t], 5.0);
        assert_close(sol.objective, 5.0);
        assert_eq!(sol.solver, "des-mip-soc-cutting-plane");
    }

    #[test]
    fn cross_check_accepts_distinct_feasible_optima_with_same_objective() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(1.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(1.0))
            .unwrap();
        p.add_constraint("simplex", vec![(x, 1.0), (y, 1.0)], RowSense::Eq, 1.0)
            .unwrap();

        let internal = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![1.0, 0.0],
            objective: 1.0,
            solver: "internal".to_string(),
            message: None,
        };
        let external = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![0.0, 1.0],
            objective: 1.0,
            solver: "external".to_string(),
            message: None,
        };

        let report = compare_math_program_solutions(&p, internal, external, 1e-6);
        assert!(report.within_tolerance);
        assert_close(report.objective_abs_diff.unwrap(), 0.0);
        assert!(report.max_x_abs_diff.unwrap() > 0.9);
        assert_close(report.internal_max_violation.unwrap(), 0.0);
        assert_close(report.external_max_violation.unwrap(), 0.0);
    }
}
