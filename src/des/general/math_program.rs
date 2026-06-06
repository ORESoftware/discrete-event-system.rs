//! Solver-style math-programming facade over the in-house LP and IP/MIP solvers.
//!
//! The low-level solvers intentionally stay close to their TypeScript ports:
//! `LPProblem` accepts LP rows/bounds, and `IPMIPProblem` accepts non-negative
//! variables with `<=` rows. This module is the compatibility layer users expect
//! from tools such as OR-Tools, Gurobi, CPLEX, FICO Xpress, LINDO, SCIP, GLPK, and HiGHS:
//! named variables, `<=`/`>=`/`=`/range rows, objective constants,
//! continuous/integer/binary/semi-continuous domains, and indicator constraints.
//! The compiler lowers those features into the existing native solvers and keeps
//! enough metadata to map solutions back to the user's original variables.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::des::general::external_linear_cli::{
    solve_ipmip_with_external_cli, solve_lp_with_external_cli, ExternalLinearCliBranchRule,
    ExternalLinearCliKind, ExternalLinearCliMipSwitch, ExternalLinearCliModelFormat,
    ExternalLinearCliNodeSelection, ExternalLinearCliOptions, ExternalLinearCliPresolve,
    ExternalLinearCliSolution, ExternalLinearCliSolver, ExternalLinearCliStatus,
};
use crate::des::general::external_quadratic_reference::{
    solve_miqcp_with_external_reference, solve_miqp_with_external_reference,
    solve_misocp_with_external_reference, solve_qcp_with_external_reference,
    solve_qp_with_external_reference, solve_socp_with_external_reference,
    ExternalQuadraticReferenceOptions, ExternalQuadraticReferenceSolution,
    ExternalQuadraticReferenceSolver, ExternalQuadraticReferenceStatus,
};
use crate::des::general::ip_mip_des::{
    solve_ipmip_with_des, BranchOrCutConstraint, ConstraintKind, IPMIPProblem, IPMIPSolveOptions,
    IPMIPStatus, IPMIPTraceEvent, TraceAction,
};
use crate::des::general::lp::{
    analyze_lp_bound_sensitivity_internal, analyze_lp_objective_sensitivity_internal,
    analyze_lp_rhs_sensitivity_internal, solve_lp_external, solve_lp_internal,
    solve_lp_internal_ipm, ExternalSolverOptions, InternalInteriorPointOptions,
    InternalSimplexOptions, LPBoundSensitivityOptions, LPBoundSensitivityReport,
    LPInfeasibilityCertificate, LPObjectiveSensitivityOptions, LPObjectiveSensitivityReport,
    LPProblem, LPRhsSensitivityOptions, LPRhsSensitivityReport, LPSolution, LPStatus,
    Sense as LpSense,
};
use crate::des::general::lp_des::{solve_lp_via_des, DESSimplexOptions};
use crate::des::general::qp::{
    MixedIntegerQuadraticProgram, MixedIntegerQuadraticallyConstrainedProgram,
    MixedIntegerSecondOrderConeProgram, QuadraticConstraint as QpQuadraticConstraint,
    QuadraticProgram, QuadraticallyConstrainedProgram, SecondOrderCone as QpSecondOrderCone,
    SecondOrderConeProgram,
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

/// Linear row enforced only when every Boolean literal is satisfied.
#[derive(Clone, Debug, PartialEq)]
pub struct EnforcedLinearConstraint {
    pub name: String,
    pub literals: Vec<BoolLiteral>,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

/// Exact finite-domain reification of an integer linear threshold.
///
/// For `<=`, this means `literal <=> coeffs * x <= rhs`; because the row
/// activity is integer-valued, the false side is `coeffs * x >= rhs + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReifiedLinearConstraint {
    pub name: String,
    pub literal: BoolLiteral,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: RowSense,
    pub rhs: f64,
}

/// Boolean literal over a binary variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolLiteral {
    pub var: usize,
    pub value: bool,
}

/// Count bounds for one value in a global-cardinality constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalCardinalityCount {
    pub value: i64,
    pub min_count: Option<usize>,
    pub max_count: Option<usize>,
}

impl GlobalCardinalityCount {
    pub fn exact(value: i64, count: usize) -> Self {
        Self {
            value,
            min_count: Some(count),
            max_count: Some(count),
        }
    }

    pub fn range(value: i64, min_count: Option<usize>, max_count: Option<usize>) -> Self {
        Self {
            value,
            min_count,
            max_count,
        }
    }
}

/// Inclusive integer interval for a finite-domain linear expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearDomainInterval {
    pub lower: i64,
    pub upper: i64,
}

impl LinearDomainInterval {
    pub fn exact(value: i64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub fn range(lower: i64, upper: i64) -> Self {
        Self { lower, upper }
    }
}

/// Special ordered set type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SOSType {
    /// At most one member can be non-zero.
    Sos1,
    /// At most two non-zero members, and those members must be adjacent by weight.
    Sos2,
}

/// Exact linear norm general constraints supported by the native MIP lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormType {
    L0,
    L1,
    LInfinity,
}

/// Special ordered set constraint. Members are sorted by `weight` for SOS2.
#[derive(Clone, Debug, PartialEq)]
pub struct SOSConstraint {
    pub name: String,
    pub sos_type: SOSType,
    pub members: Vec<(usize, f64)>,
}

/// Fixed-size or variable-size interval used by CP-SAT-style scheduling constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalTerm {
    pub start_var: usize,
    /// Constant duration offset. For fixed-size intervals this is the full size.
    pub duration: f64,
    /// Optional variable duration term. When present, the interval size is
    /// `duration + duration_var`.
    pub duration_var: Option<usize>,
    pub end_var: usize,
    pub presence_var: Option<usize>,
}

/// A signed level-change event for a reservoir constraint.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirEvent {
    pub time_var: usize,
    pub demand: f64,
    pub active_var: Option<usize>,
}

/// A transition for a finite-state automaton constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutomatonTransition {
    pub tail: i64,
    pub label: i64,
    pub head: i64,
}

/// Directed arc literal for circuit and route constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitArc {
    pub tail: usize,
    pub head: usize,
    pub literal_var: usize,
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
    BinaryXor {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
    },
    BooleanAnd {
        name: String,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    },
    BooleanOr {
        name: String,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    },
    BooleanXor {
        name: String,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    },
    BinaryCardinality {
        name: String,
        operands: Vec<usize>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    },
    BooleanCardinality {
        name: String,
        literals: Vec<BoolLiteral>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    },
    BooleanCount {
        name: String,
        literals: Vec<BoolLiteral>,
        count_var: usize,
    },
    BooleanClause {
        name: String,
        literals: Vec<BoolLiteral>,
    },
    LinearDomain {
        name: String,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<LinearDomainInterval>,
    },
    EnforcedLinearDomain {
        name: String,
        enforcement: Vec<BoolLiteral>,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<LinearDomainInterval>,
    },
    MapDomain {
        name: String,
        var: usize,
        bool_vars: Vec<usize>,
        offset: i64,
    },
    IntegerProduct {
        name: String,
        target_var: usize,
        operands: Vec<usize>,
    },
    IntegerDivision {
        name: String,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
    },
    IntegerModulo {
        name: String,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
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
    Norm {
        name: String,
        result_var: usize,
        operands: Vec<usize>,
        norm_type: NormType,
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
    GlobalCardinality {
        name: String,
        variables: Vec<usize>,
        counts: Vec<GlobalCardinalityCount>,
    },
    ValueCount {
        name: String,
        variables: Vec<usize>,
        value: i64,
        count_var: usize,
    },
    AllowedAssignments {
        name: String,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    ForbiddenAssignments {
        name: String,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    EnforcedAllowedAssignments {
        name: String,
        enforcement: Vec<BoolLiteral>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    EnforcedForbiddenAssignments {
        name: String,
        enforcement: Vec<BoolLiteral>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    },
    BinPacking {
        name: String,
        item_bin_vars: Vec<usize>,
        load_vars: Vec<usize>,
        item_sizes: Vec<f64>,
    },
    Element {
        name: String,
        index_var: usize,
        target_var: usize,
        values: Vec<f64>,
    },
    VariableElement {
        name: String,
        index_var: usize,
        target_var: usize,
        variables: Vec<usize>,
    },
    Inverse {
        name: String,
        variables: Vec<usize>,
        inverse_variables: Vec<usize>,
    },
    Circuit {
        name: String,
        node_count: usize,
        arcs: Vec<CircuitArc>,
    },
    MultipleCircuit {
        name: String,
        node_count: usize,
        arcs: Vec<CircuitArc>,
    },
    Automaton {
        name: String,
        variables: Vec<usize>,
        starting_state: i64,
        final_states: Vec<i64>,
        transitions: Vec<AutomatonTransition>,
    },
    Alternative {
        name: String,
        master: IntervalTerm,
        alternatives: Vec<IntervalTerm>,
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
        demands: Vec<AffineTerm>,
        capacity: AffineTerm,
    },
    Reservoir {
        name: String,
        events: Vec<ReservoirEvent>,
        min_level: f64,
        max_level: f64,
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
    pub objective_offset: f64,
    pub variables: Vec<Variable>,
    pub quadratic_objective: Vec<QuadraticObjectiveTerm>,
    pub secondary_objectives: Vec<LinearObjective>,
    pub quadratic_constraints: Vec<QuadraticConstraint>,
    pub constraints: Vec<LinearConstraint>,
    pub lazy_constraints: Vec<LinearConstraint>,
    pub second_order_cones: Vec<SecondOrderConeConstraint>,
    pub indicators: Vec<IndicatorConstraint>,
    pub enforced_constraints: Vec<EnforcedLinearConstraint>,
    pub reified_constraints: Vec<ReifiedLinearConstraint>,
    pub sos: Vec<SOSConstraint>,
    pub general_constraints: Vec<GeneralConstraint>,
}

impl MathProgram {
    pub fn new(sense: ObjectiveSense) -> Self {
        MathProgram {
            sense,
            objective_offset: 0.0,
            variables: Vec::new(),
            quadratic_objective: Vec::new(),
            secondary_objectives: Vec::new(),
            quadratic_constraints: Vec::new(),
            constraints: Vec::new(),
            lazy_constraints: Vec::new(),
            second_order_cones: Vec::new(),
            indicators: Vec::new(),
            enforced_constraints: Vec::new(),
            reified_constraints: Vec::new(),
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

    pub fn set_objective_offset(&mut self, offset: f64) -> Result<(), MathProgramError> {
        validate_objective_offset(offset)?;
        self.objective_offset = offset;
        Ok(())
    }

    pub fn add_objective_offset(&mut self, offset: f64) -> Result<(), MathProgramError> {
        validate_objective_offset(offset)?;
        self.objective_offset += offset;
        validate_objective_offset(self.objective_offset)?;
        Ok(())
    }

    /// Add a source-level range row `lower <= coeffs*x <= upper`.
    ///
    /// Solvers such as CPLEX, Gurobi, LINDO, SCIP, GLPK, and HiGHS expose ranged
    /// rows as a modeling convenience. The facade stores them as ordinary one- or
    /// two-sided linear rows so all native and external backends see the same
    /// compiled model.
    pub fn add_range_constraint(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<Vec<usize>, MathProgramError> {
        let name = name.into();
        validate_range_bounds(&name, lower, upper)?;
        validate_coeffs(self.variables.len(), &coeffs)?;

        match (lower, upper) {
            (Some(lo), Some(hi)) if (lo - hi).abs() <= 1e-12 => self
                .add_constraint(name, coeffs, RowSense::Eq, lo)
                .map(|idx| vec![idx]),
            (Some(lo), Some(hi)) => {
                let lower_idx = self.add_constraint(
                    format!("{name}__range_lower"),
                    coeffs.clone(),
                    RowSense::Ge,
                    lo,
                )?;
                let upper_idx =
                    self.add_constraint(format!("{name}__range_upper"), coeffs, RowSense::Le, hi)?;
                Ok(vec![lower_idx, upper_idx])
            }
            (Some(lo), None) => self
                .add_constraint(name, coeffs, RowSense::Ge, lo)
                .map(|idx| vec![idx]),
            (None, Some(hi)) => self
                .add_constraint(name, coeffs, RowSense::Le, hi)
                .map(|idx| vec![idx]),
            (None, None) => unreachable!("validate_range_bounds rejects empty range rows"),
        }
    }

    pub fn add_lazy_constraint(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        let name = name.into();
        validate_coeffs(self.variables.len(), &coeffs)?;
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "lazy constraint rhs for `{name}`"
            )));
        }
        self.lazy_constraints.push(LinearConstraint {
            name,
            coeffs,
            sense,
            rhs,
        });
        Ok(self.lazy_constraints.len() - 1)
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

    pub fn add_enforced_constraint(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        coeffs: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_enforced_linear_args(&literals, &coeffs, rhs)?;
        self.enforced_constraints.push(EnforcedLinearConstraint {
            name: name.into(),
            literals,
            coeffs,
            sense,
            rhs,
        });
        Ok(self.enforced_constraints.len() - 1)
    }

    pub fn add_reified_le_constraint(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_reified_linear_constraint(name, literal, coeffs, RowSense::Le, rhs)
    }

    pub fn add_reified_ge_constraint(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_reified_linear_constraint(name, literal, coeffs, RowSense::Ge, rhs)
    }

    pub fn add_reified_eq_constraint(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        self.add_reified_linear_constraint(name, literal, coeffs, RowSense::Eq, rhs)
    }

    pub fn add_reified_linear_constraint(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        sense: RowSense,
        rhs: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_reified_linear_args(literal, &coeffs, sense, rhs)?;
        self.reified_constraints.push(ReifiedLinearConstraint {
            name: name.into(),
            literal,
            coeffs,
            sense,
            rhs,
        });
        Ok(self.reified_constraints.len() - 1)
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

    pub fn add_binary_xor(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_binary_general_args(result_var, &operands)?;
        self.general_constraints.push(GeneralConstraint::BinaryXor {
            name: name.into(),
            result_var,
            operands,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn bool_lit(var: usize) -> BoolLiteral {
        BoolLiteral { var, value: true }
    }

    pub fn not_lit(var: usize) -> BoolLiteral {
        BoolLiteral { var, value: false }
    }

    pub fn negated_lit(literal: BoolLiteral) -> BoolLiteral {
        BoolLiteral {
            var: literal.var,
            value: !literal.value,
        }
    }

    pub fn add_boolean_and(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_general_args("boolean-and", result_var, &literals)?;
        self.general_constraints
            .push(GeneralConstraint::BooleanAnd {
                name: name.into(),
                result_var,
                literals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_boolean_or(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_general_args("boolean-or", result_var, &literals)?;
        self.general_constraints.push(GeneralConstraint::BooleanOr {
            name: name.into(),
            result_var,
            literals,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_boolean_xor(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_general_args("boolean-xor", result_var, &literals)?;
        self.general_constraints
            .push(GeneralConstraint::BooleanXor {
                name: name.into(),
                result_var,
                literals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_binary_cardinality(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_binary_cardinality_args(&operands, min_count, max_count)?;
        self.general_constraints
            .push(GeneralConstraint::BinaryCardinality {
                name: name.into(),
                operands,
                min_count,
                max_count,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_at_most_k(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
        max_count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_binary_cardinality(name, operands, None, Some(max_count))
    }

    pub fn add_at_least_k(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
        min_count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_binary_cardinality(name, operands, Some(min_count), None)
    }

    pub fn add_exactly_k(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
        count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_binary_cardinality(name, operands, Some(count), Some(count))
    }

    pub fn add_at_most_one(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_at_most_k(name, operands, 1)
    }

    pub fn add_at_least_one(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_at_least_k(name, operands, 1)
    }

    pub fn add_exactly_one(
        &mut self,
        name: impl Into<String>,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_exactly_k(name, operands, 1)
    }

    pub fn add_boolean_cardinality(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_cardinality_args(&literals, min_count, max_count)?;
        self.general_constraints
            .push(GeneralConstraint::BooleanCardinality {
                name: name.into(),
                literals,
                min_count,
                max_count,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_literal_at_most_k(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        max_count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_boolean_cardinality(name, literals, None, Some(max_count))
    }

    pub fn add_literal_at_least_k(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        min_count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_boolean_cardinality(name, literals, Some(min_count), None)
    }

    pub fn add_literal_exactly_k(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        count: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_boolean_cardinality(name, literals, Some(count), Some(count))
    }

    pub fn add_literal_at_most_one(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_literal_at_most_k(name, literals, 1)
    }

    pub fn add_literal_at_least_one(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_literal_at_least_k(name, literals, 1)
    }

    pub fn add_literal_exactly_one(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_literal_exactly_k(name, literals, 1)
    }

    pub fn add_enforced_boolean_clause(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_enforced_boolean_cardinality_args(&enforcement, &literals, Some(1), None)?;
        let (coeffs, negated_count) = literal_cardinality_terms(&literals);
        self.add_enforced_constraint(
            name,
            enforcement,
            coeffs,
            RowSense::Ge,
            1.0 - negated_count as f64,
        )
    }

    pub fn add_enforced_boolean_or(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_enforced_boolean_clause(name, enforcement, literals)
    }

    pub fn add_enforced_boolean_and(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        let count = literals.len();
        let ids =
            self.add_enforced_boolean_cardinality(name, enforcement, literals, Some(count), None)?;
        ids.first().copied().ok_or_else(|| {
            MathProgramError::Unsupported("enforced boolean-and generated no rows".to_string())
        })
    }

    pub fn add_enforced_boolean_xor(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_enforced_boolean_cardinality_args(&enforcement, &literals, Some(1), None)?;
        let (coeffs, negated_count) = literal_cardinality_terms(&literals);
        let intervals = (1..=literals.len())
            .step_by(2)
            .map(|truth_count| {
                let shifted = truth_count as i64 - negated_count as i64;
                (shifted, shifted)
            })
            .collect::<Vec<_>>();
        self.add_enforced_linear_domain(name, enforcement, coeffs, intervals)
    }

    pub fn add_enforced_boolean_cardinality(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.validate_enforced_boolean_cardinality_args(
            &enforcement,
            &literals,
            min_count,
            max_count,
        )?;
        let name = name.into();
        let (coeffs, negated_count) = literal_cardinality_terms(&literals);
        let mut ids = Vec::new();
        if let Some(max_count) = max_count {
            let row_name = if min_count.is_some() {
                format!("{name}__at_most")
            } else {
                name.clone()
            };
            ids.push(self.add_enforced_constraint(
                row_name,
                enforcement.clone(),
                coeffs.clone(),
                RowSense::Le,
                max_count as f64 - negated_count as f64,
            )?);
        }
        if let Some(min_count) = min_count {
            let row_name = if max_count.is_some() {
                format!("{name}__at_least")
            } else {
                name
            };
            ids.push(self.add_enforced_constraint(
                row_name,
                enforcement,
                coeffs,
                RowSense::Ge,
                min_count as f64 - negated_count as f64,
            )?);
        }
        Ok(ids)
    }

    pub fn add_enforced_literal_at_most_k(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
        max_count: usize,
    ) -> Result<usize, MathProgramError> {
        let ids = self.add_enforced_boolean_cardinality(
            name,
            enforcement,
            literals,
            None,
            Some(max_count),
        )?;
        ids.first().copied().ok_or_else(|| {
            MathProgramError::Unsupported("enforced literal-at-most generated no rows".to_string())
        })
    }

    pub fn add_enforced_literal_at_least_k(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
        min_count: usize,
    ) -> Result<usize, MathProgramError> {
        let ids = self.add_enforced_boolean_cardinality(
            name,
            enforcement,
            literals,
            Some(min_count),
            None,
        )?;
        ids.first().copied().ok_or_else(|| {
            MathProgramError::Unsupported("enforced literal-at-least generated no rows".to_string())
        })
    }

    pub fn add_enforced_literal_exactly_k(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
        count: usize,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.add_enforced_boolean_cardinality(name, enforcement, literals, Some(count), Some(count))
    }

    pub fn add_enforced_literal_at_most_one(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_enforced_literal_at_most_k(name, enforcement, literals, 1)
    }

    pub fn add_enforced_literal_at_least_one(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.add_enforced_literal_at_least_k(name, enforcement, literals, 1)
    }

    pub fn add_enforced_literal_exactly_one(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        literals: Vec<BoolLiteral>,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.add_enforced_literal_exactly_k(name, enforcement, literals, 1)
    }

    pub fn add_boolean_count(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
        count_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_count_args(&literals, count_var)?;
        self.general_constraints
            .push(GeneralConstraint::BooleanCount {
                name: name.into(),
                literals,
                count_var,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_boolean_clause(
        &mut self,
        name: impl Into<String>,
        literals: Vec<BoolLiteral>,
    ) -> Result<usize, MathProgramError> {
        self.validate_boolean_clause_args(&literals)?;
        self.general_constraints
            .push(GeneralConstraint::BooleanClause {
                name: name.into(),
                literals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_literal_implication(
        &mut self,
        name: impl Into<String>,
        antecedent: BoolLiteral,
        consequent: BoolLiteral,
    ) -> Result<usize, MathProgramError> {
        self.add_boolean_clause(name, vec![Self::negated_lit(antecedent), consequent])
    }

    pub fn add_literal_equivalence(
        &mut self,
        name: impl Into<String>,
        left: BoolLiteral,
        right: BoolLiteral,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.validate_boolean_clause_args(&[left, right])?;
        let name = name.into();
        let forward = self.add_literal_implication(format!("{name}__forward"), left, right)?;
        let reverse = self.add_literal_implication(format!("{name}__reverse"), right, left)?;
        Ok(vec![forward, reverse])
    }

    pub fn add_binary_implication(
        &mut self,
        name: impl Into<String>,
        antecedent: usize,
        consequent: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_literal_implication(name, Self::bool_lit(antecedent), Self::bool_lit(consequent))
    }

    pub fn add_binary_equivalence(
        &mut self,
        name: impl Into<String>,
        left: usize,
        right: usize,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.add_literal_equivalence(name, Self::bool_lit(left), Self::bool_lit(right))
    }

    pub fn add_integer_product(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_integer_product_args(target_var, &operands)?;
        self.general_constraints
            .push(GeneralConstraint::IntegerProduct {
                name: name.into(),
                target_var,
                operands,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_multiplication_equality(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_integer_product(name, target_var, operands)
    }

    pub fn add_integer_division(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.validate_integer_binary_operation_args(
            "integer division",
            target_var,
            numerator_var,
            denominator_var,
            i64::checked_div,
        )?;
        self.general_constraints
            .push(GeneralConstraint::IntegerDivision {
                name: name.into(),
                target_var,
                numerator_var,
                denominator_var,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_division_equality(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_integer_division(name, target_var, numerator_var, denominator_var)
    }

    pub fn add_integer_modulo(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.validate_integer_binary_operation_args(
            "integer modulo",
            target_var,
            numerator_var,
            denominator_var,
            i64::checked_rem,
        )?;
        self.general_constraints
            .push(GeneralConstraint::IntegerModulo {
                name: name.into(),
                target_var,
                numerator_var,
                denominator_var,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_modulo_equality(
        &mut self,
        name: impl Into<String>,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.add_integer_modulo(name, target_var, numerator_var, denominator_var)
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

    pub fn add_norm(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
        norm_type: NormType,
    ) -> Result<usize, MathProgramError> {
        self.validate_norm_args(result_var, &operands, norm_type)?;
        self.general_constraints.push(GeneralConstraint::Norm {
            name: name.into(),
            result_var,
            operands,
            norm_type,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_l1_norm(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_norm(name, result_var, operands, NormType::L1)
    }

    pub fn add_l0_norm(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_norm(name, result_var, operands, NormType::L0)
    }

    pub fn add_l_infinity_norm(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.add_norm(name, result_var, operands, NormType::LInfinity)
    }

    /// Add the convex Euclidean norm epigraph `sqrt(sum_i operand_i^2) <= result`.
    ///
    /// This mirrors the common L2 norm modeling pattern used by commercial solvers:
    /// minimizing `result` makes the epigraph tight while preserving a convex
    /// second-order-cone formulation.
    pub fn add_l2_norm(
        &mut self,
        name: impl Into<String>,
        result_var: usize,
        operands: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        let name = name.into();
        let terms = operands
            .into_iter()
            .map(|var| AffineTerm {
                coeffs: vec![(var, 1.0)],
                constant: 0.0,
            })
            .collect::<Vec<_>>();
        let rhs_coeffs = vec![(result_var, 1.0)];
        self.validate_second_order_cone_args(&terms, &rhs_coeffs, 0.0)?;
        self.add_constraint(
            format!("{name}__result_nonnegative"),
            rhs_coeffs.clone(),
            RowSense::Ge,
            0.0,
        )?;
        self.second_order_cones.push(SecondOrderConeConstraint {
            name,
            terms,
            rhs_coeffs,
            rhs_constant: 0.0,
        });
        Ok(self.second_order_cones.len() - 1)
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

    /// Add a univariate function constraint `y = f(x)` by interpolating over
    /// explicit breakpoints and lowering to the same SOS2-style MIP encoding as
    /// [`Self::add_piecewise_linear`].
    pub fn add_univariate_piecewise_function<F>(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
        function: F,
    ) -> Result<usize, MathProgramError>
    where
        F: Fn(f64) -> f64,
    {
        let points =
            self.univariate_function_points("univariate function", &breakpoints, function)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = exp(x)` using explicit piecewise-linear breakpoints.
    pub fn add_exp_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points =
            self.univariate_function_points("exponential function", &breakpoints, f64::exp)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = ln(x)` using explicit piecewise-linear breakpoints.
    pub fn add_log_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points =
            self.univariate_function_points("logarithm function", &breakpoints, f64::ln)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = x^exponent` using explicit piecewise-linear breakpoints.
    pub fn add_power_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        exponent: f64,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        if !exponent.is_finite() {
            return Err(MathProgramError::NonFinite("power exponent".to_string()));
        }
        let points =
            self.univariate_function_points("power function", &breakpoints, |x| x.powf(exponent))?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = 1 / (1 + exp(-x))` using explicit piecewise-linear breakpoints.
    pub fn add_logistic_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points = self.univariate_function_points("logistic function", &breakpoints, |x| {
            1.0 / (1.0 + (-x).exp())
        })?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = sin(x)` using explicit piecewise-linear breakpoints.
    pub fn add_sin_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points = self.univariate_function_points("sine function", &breakpoints, f64::sin)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = cos(x)` using explicit piecewise-linear breakpoints.
    pub fn add_cos_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points = self.univariate_function_points("cosine function", &breakpoints, f64::cos)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
    }

    /// Add `y = tan(x)` using explicit piecewise-linear breakpoints.
    pub fn add_tan_function(
        &mut self,
        name: impl Into<String>,
        x_var: usize,
        y_var: usize,
        breakpoints: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        let points = self.univariate_function_points("tangent function", &breakpoints, f64::tan)?;
        self.add_piecewise_linear(name, x_var, y_var, points)
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

    pub fn add_global_cardinality(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        counts: Vec<GlobalCardinalityCount>,
    ) -> Result<usize, MathProgramError> {
        self.validate_global_cardinality_args(&variables, &counts)?;
        self.general_constraints
            .push(GeneralConstraint::GlobalCardinality {
                name: name.into(),
                variables,
                counts,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_value_count(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        value: i64,
        count_var: usize,
    ) -> Result<usize, MathProgramError> {
        self.validate_value_count_args(&variables, value, count_var)?;
        self.general_constraints
            .push(GeneralConstraint::ValueCount {
                name: name.into(),
                variables,
                value,
                count_var,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_linear_domain(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<(i64, i64)>,
    ) -> Result<usize, MathProgramError> {
        let intervals = intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();
        self.validate_linear_domain_args(&coeffs, &intervals)?;
        self.general_constraints
            .push(GeneralConstraint::LinearDomain {
                name: name.into(),
                coeffs,
                intervals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_enforced_linear_domain(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<(i64, i64)>,
    ) -> Result<usize, MathProgramError> {
        let intervals = intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();
        self.validate_enforced_linear_domain_args(&enforcement, &coeffs, &intervals)?;
        self.general_constraints
            .push(GeneralConstraint::EnforcedLinearDomain {
                name: name.into(),
                enforcement,
                coeffs,
                intervals,
            });
        Ok(self.general_constraints.len() - 1)
    }

    /// Reify finite-domain membership for an integer-valued linear expression:
    /// `literal <=> coeffs * x in intervals`.
    pub fn add_reified_linear_domain(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<(i64, i64)>,
    ) -> Result<Vec<usize>, MathProgramError> {
        let intervals = intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();
        self.validate_boolean_clause_args(&[literal])?;
        self.validate_linear_domain_args(&coeffs, &intervals)?;
        let complement_intervals =
            self.linear_not_in_domain_intervals("reified linear-domain", &coeffs, &intervals)?;
        let complement_intervals = complement_intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();

        let name = name.into();
        let true_idx = self.general_constraints.len();
        self.general_constraints
            .push(GeneralConstraint::EnforcedLinearDomain {
                name: format!("{name}__true"),
                enforcement: vec![literal],
                coeffs: coeffs.clone(),
                intervals,
            });
        let false_idx = self.general_constraints.len();
        self.general_constraints
            .push(GeneralConstraint::EnforcedLinearDomain {
                name: format!("{name}__false"),
                enforcement: vec![Self::negated_lit(literal)],
                coeffs,
                intervals: complement_intervals,
            });
        Ok(vec![true_idx, false_idx])
    }

    /// Reify finite-domain exclusion for an integer-valued linear expression:
    /// `literal <=> coeffs * x not in intervals`.
    pub fn add_reified_linear_not_in_domain(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        intervals: Vec<(i64, i64)>,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.add_reified_linear_domain(name, Self::negated_lit(literal), coeffs, intervals)
    }

    /// Reify a finite-domain linear disequality:
    /// `literal <=> coeffs * x != forbidden_value`.
    pub fn add_reified_linear_not_equal(
        &mut self,
        name: impl Into<String>,
        literal: BoolLiteral,
        coeffs: Vec<(usize, f64)>,
        forbidden_value: i64,
    ) -> Result<Vec<usize>, MathProgramError> {
        self.add_reified_linear_not_in_domain(
            name,
            literal,
            coeffs,
            vec![(forbidden_value, forbidden_value)],
        )
    }

    /// Restrict an integer-valued linear expression outside a finite union of
    /// forbidden integer intervals.
    ///
    /// This complements [`Self::add_linear_domain`]: the helper computes the
    /// finite integer expression bounds, removes the forbidden interval union,
    /// and lowers the remaining allowed intervals through exact finite-domain
    /// MIP rows.
    pub fn add_linear_not_in_domain(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        forbidden_intervals: Vec<(i64, i64)>,
    ) -> Result<usize, MathProgramError> {
        let forbidden_intervals = forbidden_intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();
        let intervals = self.linear_not_in_domain_intervals(
            "linear-not-in-domain",
            &coeffs,
            &forbidden_intervals,
        )?;
        self.add_linear_domain(name, coeffs, intervals)
    }

    /// Enforced variant of [`Self::add_linear_not_in_domain`].
    ///
    /// The domain-complement restriction is active only when every enforcement
    /// literal is true.
    pub fn add_enforced_linear_not_in_domain(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        coeffs: Vec<(usize, f64)>,
        forbidden_intervals: Vec<(i64, i64)>,
    ) -> Result<usize, MathProgramError> {
        if enforcement.is_empty() {
            return Err(MathProgramError::Unsupported(
                "enforced linear-not-in-domain constraints require at least one enforcement literal"
                    .to_string(),
            ));
        }
        self.validate_boolean_clause_args(&enforcement)?;
        let forbidden_intervals = forbidden_intervals
            .into_iter()
            .map(|(lower, upper)| LinearDomainInterval { lower, upper })
            .collect::<Vec<_>>();
        let intervals = self.linear_not_in_domain_intervals(
            "enforced linear-not-in-domain",
            &coeffs,
            &forbidden_intervals,
        )?;
        self.add_enforced_linear_domain(name, enforcement, coeffs, intervals)
    }

    /// Restrict an integer-valued linear expression away from one forbidden value.
    ///
    /// This is a CP-SAT-style convenience over
    /// [`Self::add_linear_not_in_domain`].
    pub fn add_linear_not_equal(
        &mut self,
        name: impl Into<String>,
        coeffs: Vec<(usize, f64)>,
        forbidden_value: i64,
    ) -> Result<usize, MathProgramError> {
        let intervals =
            self.linear_not_equal_intervals("linear-not-equal", &coeffs, forbidden_value)?;
        self.add_linear_domain(name, coeffs, intervals)
    }

    /// Enforced variant of [`Self::add_linear_not_equal`].
    ///
    /// The disequality is active only when every enforcement literal is true.
    pub fn add_enforced_linear_not_equal(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        coeffs: Vec<(usize, f64)>,
        forbidden_value: i64,
    ) -> Result<usize, MathProgramError> {
        if enforcement.is_empty() {
            return Err(MathProgramError::Unsupported(
                "enforced linear-not-equal constraints require at least one enforcement literal"
                    .to_string(),
            ));
        }
        self.validate_boolean_clause_args(&enforcement)?;
        let intervals =
            self.linear_not_equal_intervals("enforced linear-not-equal", &coeffs, forbidden_value)?;
        self.add_enforced_linear_domain(name, enforcement, coeffs, intervals)
    }

    /// Add the CP-SAT-style channeling constraint
    /// `bool_vars[i] <=> var == offset + i`.
    ///
    /// If `var` can take values outside the mapped range, every selector is
    /// forced to zero for those values. If `var` is bounded exactly to the
    /// mapped range, the selectors form a one-hot encoding.
    pub fn add_map_domain(
        &mut self,
        name: impl Into<String>,
        var: usize,
        bool_vars: Vec<usize>,
        offset: i64,
    ) -> Result<usize, MathProgramError> {
        self.validate_map_domain_args(var, &bool_vars, offset)?;
        self.general_constraints.push(GeneralConstraint::MapDomain {
            name: name.into(),
            var,
            bool_vars,
            offset,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_allowed_assignments(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    ) -> Result<usize, MathProgramError> {
        self.validate_table_assignments_args("allowed-assignments", &variables, &tuples)?;
        self.general_constraints
            .push(GeneralConstraint::AllowedAssignments {
                name: name.into(),
                variables,
                tuples,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_forbidden_assignments(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    ) -> Result<usize, MathProgramError> {
        self.validate_table_assignments_args("forbidden-assignments", &variables, &tuples)?;
        self.general_constraints
            .push(GeneralConstraint::ForbiddenAssignments {
                name: name.into(),
                variables,
                tuples,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_enforced_allowed_assignments(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    ) -> Result<usize, MathProgramError> {
        self.validate_enforced_table_assignments_args(
            "enforced-allowed-assignments",
            &enforcement,
            &variables,
            &tuples,
        )?;
        self.general_constraints
            .push(GeneralConstraint::EnforcedAllowedAssignments {
                name: name.into(),
                enforcement,
                variables,
                tuples,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_enforced_forbidden_assignments(
        &mut self,
        name: impl Into<String>,
        enforcement: Vec<BoolLiteral>,
        variables: Vec<usize>,
        tuples: Vec<Vec<i64>>,
    ) -> Result<usize, MathProgramError> {
        self.validate_enforced_table_assignments_args(
            "enforced-forbidden-assignments",
            &enforcement,
            &variables,
            &tuples,
        )?;
        self.general_constraints
            .push(GeneralConstraint::EnforcedForbiddenAssignments {
                name: name.into(),
                enforcement,
                variables,
                tuples,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_bin_packing(
        &mut self,
        name: impl Into<String>,
        item_bin_vars: Vec<usize>,
        load_vars: Vec<usize>,
        item_sizes: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        self.validate_bin_packing_args(&item_bin_vars, &load_vars, &item_sizes)?;
        self.general_constraints
            .push(GeneralConstraint::BinPacking {
                name: name.into(),
                item_bin_vars,
                load_vars,
                item_sizes,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_element(
        &mut self,
        name: impl Into<String>,
        index_var: usize,
        target_var: usize,
        values: Vec<f64>,
    ) -> Result<usize, MathProgramError> {
        self.validate_element_args(index_var, target_var, values.as_slice())?;
        self.general_constraints.push(GeneralConstraint::Element {
            name: name.into(),
            index_var,
            target_var,
            values,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_variable_element(
        &mut self,
        name: impl Into<String>,
        index_var: usize,
        target_var: usize,
        variables: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_variable_element_args(index_var, target_var, variables.as_slice())?;
        self.general_constraints
            .push(GeneralConstraint::VariableElement {
                name: name.into(),
                index_var,
                target_var,
                variables,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_inverse(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        inverse_variables: Vec<usize>,
    ) -> Result<usize, MathProgramError> {
        self.validate_inverse_args(&variables, &inverse_variables)?;
        self.general_constraints.push(GeneralConstraint::Inverse {
            name: name.into(),
            variables,
            inverse_variables,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_circuit(
        &mut self,
        name: impl Into<String>,
        node_count: usize,
        arcs: Vec<(usize, usize, usize)>,
    ) -> Result<usize, MathProgramError> {
        let arcs = arcs
            .into_iter()
            .map(|(tail, head, literal_var)| CircuitArc {
                tail,
                head,
                literal_var,
            })
            .collect::<Vec<_>>();
        self.validate_circuit_args(node_count, arcs.as_slice())?;
        self.general_constraints.push(GeneralConstraint::Circuit {
            name: name.into(),
            node_count,
            arcs,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_multiple_circuit(
        &mut self,
        name: impl Into<String>,
        node_count: usize,
        arcs: Vec<(usize, usize, usize)>,
    ) -> Result<usize, MathProgramError> {
        let arcs = arcs
            .into_iter()
            .map(|(tail, head, literal_var)| CircuitArc {
                tail,
                head,
                literal_var,
            })
            .collect::<Vec<_>>();
        self.validate_multiple_circuit_args(node_count, arcs.as_slice())?;
        self.general_constraints
            .push(GeneralConstraint::MultipleCircuit {
                name: name.into(),
                node_count,
                arcs,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn add_automaton(
        &mut self,
        name: impl Into<String>,
        variables: Vec<usize>,
        starting_state: i64,
        final_states: Vec<i64>,
        transitions: Vec<(i64, i64, i64)>,
    ) -> Result<usize, MathProgramError> {
        let transitions = transitions
            .into_iter()
            .map(|(tail, label, head)| AutomatonTransition { tail, label, head })
            .collect::<Vec<_>>();
        self.validate_automaton_args(
            &variables,
            starting_state,
            final_states.as_slice(),
            transitions.as_slice(),
        )?;
        self.general_constraints.push(GeneralConstraint::Automaton {
            name: name.into(),
            variables,
            starting_state,
            final_states,
            transitions,
        });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn interval(start_var: usize, duration: f64, end_var: usize) -> IntervalTerm {
        IntervalTerm {
            start_var,
            duration,
            duration_var: None,
            end_var,
            presence_var: None,
        }
    }

    pub fn variable_interval(
        start_var: usize,
        duration_var: usize,
        end_var: usize,
    ) -> IntervalTerm {
        IntervalTerm {
            start_var,
            duration: 0.0,
            duration_var: Some(duration_var),
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
            duration_var: None,
            end_var,
            presence_var: Some(presence_var),
        }
    }

    pub fn optional_variable_interval(
        start_var: usize,
        duration_var: usize,
        end_var: usize,
        presence_var: usize,
    ) -> IntervalTerm {
        IntervalTerm {
            start_var,
            duration: 0.0,
            duration_var: Some(duration_var),
            end_var,
            presence_var: Some(presence_var),
        }
    }

    pub fn add_alternative(
        &mut self,
        name: impl Into<String>,
        master: IntervalTerm,
        alternatives: Vec<IntervalTerm>,
    ) -> Result<usize, MathProgramError> {
        self.validate_alternative_args(&master, &alternatives)?;
        self.general_constraints
            .push(GeneralConstraint::Alternative {
                name: name.into(),
                master,
                alternatives,
            });
        Ok(self.general_constraints.len() - 1)
    }

    pub fn reservoir_event(time_var: usize, demand: f64) -> ReservoirEvent {
        ReservoirEvent {
            time_var,
            demand,
            active_var: None,
        }
    }

    pub fn optional_reservoir_event(
        time_var: usize,
        demand: f64,
        active_var: usize,
    ) -> ReservoirEvent {
        ReservoirEvent {
            time_var,
            demand,
            active_var: Some(active_var),
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

    pub fn add_reservoir(
        &mut self,
        name: impl Into<String>,
        events: Vec<ReservoirEvent>,
        min_level: f64,
        max_level: f64,
    ) -> Result<usize, MathProgramError> {
        self.validate_reservoir_args(&events, min_level, max_level)?;
        self.general_constraints.push(GeneralConstraint::Reservoir {
            name: name.into(),
            events,
            min_level,
            max_level,
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
        let demands = demands
            .into_iter()
            .map(|constant| AffineTerm {
                coeffs: Vec::new(),
                constant,
            })
            .collect::<Vec<_>>();
        let capacity = AffineTerm {
            coeffs: Vec::new(),
            constant: capacity,
        };
        self.add_cumulative_affine(name, intervals, demands, capacity)
    }

    pub fn add_cumulative_affine(
        &mut self,
        name: impl Into<String>,
        intervals: Vec<IntervalTerm>,
        demands: Vec<AffineTerm>,
        capacity: AffineTerm,
    ) -> Result<usize, MathProgramError> {
        self.validate_cumulative_args(&intervals, &demands, &capacity)?;
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
            || !self.enforced_constraints.is_empty()
            || !self.reified_constraints.is_empty()
            || !self.sos.is_empty()
            || !self.general_constraints.is_empty()
            || !self.lazy_constraints.is_empty()
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
        validate_objective_offset(self.objective_offset)?;
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
        for row in &self.lazy_constraints {
            validate_coeffs(self.variables.len(), &row.coeffs)?;
            if !row.rhs.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "lazy constraint rhs for `{}`",
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
        for enforced in &self.enforced_constraints {
            self.validate_enforced_linear_args(&enforced.literals, &enforced.coeffs, enforced.rhs)?;
        }
        for reified in &self.reified_constraints {
            self.validate_reified_linear_args(
                reified.literal,
                &reified.coeffs,
                reified.sense,
                reified.rhs,
            )?;
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
                }
                | GeneralConstraint::BinaryXor {
                    result_var,
                    operands,
                    ..
                } => self.validate_binary_general_args(*result_var, operands)?,
                GeneralConstraint::BooleanAnd {
                    result_var,
                    literals,
                    ..
                } => self.validate_boolean_general_args("boolean-and", *result_var, literals)?,
                GeneralConstraint::BooleanOr {
                    result_var,
                    literals,
                    ..
                } => self.validate_boolean_general_args("boolean-or", *result_var, literals)?,
                GeneralConstraint::BooleanXor {
                    result_var,
                    literals,
                    ..
                } => self.validate_boolean_general_args("boolean-xor", *result_var, literals)?,
                GeneralConstraint::BinaryCardinality {
                    operands,
                    min_count,
                    max_count,
                    ..
                } => {
                    self.validate_binary_cardinality_args(operands, *min_count, *max_count)?;
                }
                GeneralConstraint::BooleanCardinality {
                    literals,
                    min_count,
                    max_count,
                    ..
                } => {
                    self.validate_boolean_cardinality_args(literals, *min_count, *max_count)?;
                }
                GeneralConstraint::BooleanCount {
                    literals,
                    count_var,
                    ..
                } => self.validate_boolean_count_args(literals, *count_var)?,
                GeneralConstraint::BooleanClause { literals, .. } => {
                    self.validate_boolean_clause_args(literals)?;
                }
                GeneralConstraint::LinearDomain {
                    coeffs, intervals, ..
                } => {
                    self.validate_linear_domain_args(coeffs, intervals)?;
                }
                GeneralConstraint::EnforcedLinearDomain {
                    enforcement,
                    coeffs,
                    intervals,
                    ..
                } => {
                    self.validate_enforced_linear_domain_args(enforcement, coeffs, intervals)?;
                }
                GeneralConstraint::MapDomain {
                    var,
                    bool_vars,
                    offset,
                    ..
                } => self.validate_map_domain_args(*var, bool_vars, *offset)?,
                GeneralConstraint::IntegerProduct {
                    target_var,
                    operands,
                    ..
                } => self.validate_integer_product_args(*target_var, operands)?,
                GeneralConstraint::IntegerDivision {
                    target_var,
                    numerator_var,
                    denominator_var,
                    ..
                } => self.validate_integer_binary_operation_args(
                    "integer division",
                    *target_var,
                    *numerator_var,
                    *denominator_var,
                    i64::checked_div,
                )?,
                GeneralConstraint::IntegerModulo {
                    target_var,
                    numerator_var,
                    denominator_var,
                    ..
                } => self.validate_integer_binary_operation_args(
                    "integer modulo",
                    *target_var,
                    *numerator_var,
                    *denominator_var,
                    i64::checked_rem,
                )?,
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
                GeneralConstraint::Norm {
                    result_var,
                    operands,
                    norm_type,
                    ..
                } => self.validate_norm_args(*result_var, operands, *norm_type)?,
                GeneralConstraint::PiecewiseLinear {
                    x_var,
                    y_var,
                    points,
                    ..
                } => self.validate_piecewise_linear_args(*x_var, *y_var, points)?,
                GeneralConstraint::AllDifferent { variables, .. } => {
                    self.validate_all_different_args(variables)?
                }
                GeneralConstraint::GlobalCardinality {
                    variables, counts, ..
                } => self.validate_global_cardinality_args(variables, counts)?,
                GeneralConstraint::ValueCount {
                    variables,
                    value,
                    count_var,
                    ..
                } => self.validate_value_count_args(variables, *value, *count_var)?,
                GeneralConstraint::AllowedAssignments {
                    variables, tuples, ..
                } => {
                    self.validate_table_assignments_args("allowed-assignments", variables, tuples)?
                }
                GeneralConstraint::ForbiddenAssignments {
                    variables, tuples, ..
                } => self.validate_table_assignments_args(
                    "forbidden-assignments",
                    variables,
                    tuples,
                )?,
                GeneralConstraint::EnforcedAllowedAssignments {
                    enforcement,
                    variables,
                    tuples,
                    ..
                } => self.validate_enforced_table_assignments_args(
                    "enforced-allowed-assignments",
                    enforcement,
                    variables,
                    tuples,
                )?,
                GeneralConstraint::EnforcedForbiddenAssignments {
                    enforcement,
                    variables,
                    tuples,
                    ..
                } => self.validate_enforced_table_assignments_args(
                    "enforced-forbidden-assignments",
                    enforcement,
                    variables,
                    tuples,
                )?,
                GeneralConstraint::BinPacking {
                    item_bin_vars,
                    load_vars,
                    item_sizes,
                    ..
                } => self.validate_bin_packing_args(item_bin_vars, load_vars, item_sizes)?,
                GeneralConstraint::Element {
                    index_var,
                    target_var,
                    values,
                    ..
                } => self.validate_element_args(*index_var, *target_var, values)?,
                GeneralConstraint::VariableElement {
                    index_var,
                    target_var,
                    variables,
                    ..
                } => self.validate_variable_element_args(*index_var, *target_var, variables)?,
                GeneralConstraint::Inverse {
                    variables,
                    inverse_variables,
                    ..
                } => self.validate_inverse_args(variables, inverse_variables)?,
                GeneralConstraint::Circuit {
                    node_count, arcs, ..
                } => self.validate_circuit_args(*node_count, arcs)?,
                GeneralConstraint::MultipleCircuit {
                    node_count, arcs, ..
                } => self.validate_multiple_circuit_args(*node_count, arcs)?,
                GeneralConstraint::Automaton {
                    variables,
                    starting_state,
                    final_states,
                    transitions,
                    ..
                } => self.validate_automaton_args(
                    variables,
                    *starting_state,
                    final_states,
                    transitions,
                )?,
                GeneralConstraint::Alternative {
                    master,
                    alternatives,
                    ..
                } => self.validate_alternative_args(master, alternatives)?,
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
                } => self.validate_cumulative_args(intervals, demands, capacity)?,
                GeneralConstraint::Reservoir {
                    events,
                    min_level,
                    max_level,
                    ..
                } => self.validate_reservoir_args(events, *min_level, *max_level)?,
            }
        }
        Ok(())
    }

    fn validate_norm_args(
        &self,
        result_var: usize,
        operands: &[usize],
        norm_type: NormType,
    ) -> Result<(), MathProgramError> {
        let kind = match norm_type {
            NormType::L0 => "l0-norm",
            NormType::L1 => "l1-norm",
            NormType::LInfinity => "l-infinity-norm",
        };
        self.validate_extreme_general_args(kind, result_var, operands)?;
        if self.variables[result_var].lb.is_some_and(|lb| lb < 0.0) {
            return Err(MathProgramError::InvalidBound(format!(
                "{kind} result `{}` must have non-negative lower bound",
                self.variables[result_var].name
            )));
        }
        if norm_type == NormType::L0 {
            if !matches!(
                self.variables[result_var].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "l0-norm result `{}` must be binary or integer",
                    self.variables[result_var].name
                )));
            }
            let (result_lower, result_upper) = integer_bounds(&self.variables[result_var])
                .ok_or_else(|| {
                    MathProgramError::UnboundedBigM(format!(
                        "l0-norm result `{}` requires finite integer bounds",
                        self.variables[result_var].name
                    ))
                })?;
            if result_lower > operands.len() as i64 || result_upper < 0 {
                return Err(MathProgramError::InvalidBound(format!(
                    "l0-norm result `{}` bounds [{result_lower}, {result_upper}] cannot contain any count in [0, {}]",
                    self.variables[result_var].name,
                    operands.len()
                )));
            }

            let mut literal_count = 0usize;
            for &operand in operands {
                if !matches!(
                    self.variables[operand].var_type,
                    VariableType::Binary | VariableType::Integer
                ) {
                    return Err(MathProgramError::Unsupported(format!(
                        "l0-norm operand `{}` must be binary or integer for exact MIP lowering",
                        self.variables[operand].name
                    )));
                }
                let (lower, upper) = integer_bounds(&self.variables[operand]).ok_or_else(|| {
                    MathProgramError::UnboundedBigM(format!(
                        "l0-norm operand `{}` requires finite integer bounds",
                        self.variables[operand].name
                    ))
                })?;
                let domain_size = upper
                    .checked_sub(lower)
                    .and_then(|span| span.checked_add(1))
                    .ok_or_else(|| {
                        MathProgramError::Unsupported(format!(
                            "l0-norm operand `{}` has an oversized domain",
                            self.variables[operand].name
                        ))
                    })?;
                literal_count =
                    literal_count
                        .checked_add(domain_size as usize)
                        .ok_or_else(|| {
                            MathProgramError::Unsupported(
                                "l0-norm value literal count overflowed".to_string(),
                            )
                        })?;
                if literal_count > 4096 {
                    return Err(MathProgramError::Unsupported(format!(
                        "l0-norm exact MIP lowering is limited to 4096 value literals, got {literal_count}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_circuit_args(
        &self,
        node_count: usize,
        arcs: &[CircuitArc],
    ) -> Result<(), MathProgramError> {
        if node_count < 2 {
            return Err(MathProgramError::Unsupported(
                "circuit requires at least two nodes".to_string(),
            ));
        }
        if arcs.is_empty() {
            return Err(MathProgramError::Unsupported(
                "circuit requires at least one directed arc".to_string(),
            ));
        }
        if arcs.len() > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "circuit exact MIP lowering is limited to 4096 arcs, got {}",
                arcs.len()
            )));
        }

        let mut incoming = vec![0usize; node_count];
        let mut outgoing = vec![0usize; node_count];
        let mut seen_arcs = Vec::with_capacity(arcs.len());
        let mut seen_literals = Vec::with_capacity(arcs.len());

        for arc in arcs {
            if arc.tail >= node_count || arc.head >= node_count {
                return Err(MathProgramError::BadIndex(format!(
                    "circuit arc {} -> {} is outside node range [0, {})",
                    arc.tail, arc.head, node_count
                )));
            }
            if arc.tail == arc.head {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit arc {} -> {} is a self-loop; this Hamiltonian circuit lowering requires distinct endpoints",
                    arc.tail, arc.head
                )));
            }
            if arc.literal_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "circuit arc {} -> {} literal variable index {} out of bounds",
                    arc.tail, arc.head, arc.literal_var
                )));
            }
            let literal = &self.variables[arc.literal_var];
            if literal.var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit arc {} -> {} literal `{}` must be binary",
                    arc.tail, arc.head, literal.name
                )));
            }
            if seen_arcs.contains(&(arc.tail, arc.head)) {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit has duplicate directed arc {} -> {}",
                    arc.tail, arc.head
                )));
            }
            if seen_literals.contains(&arc.literal_var) {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit literal `{}` is used by more than one arc",
                    literal.name
                )));
            }
            seen_arcs.push((arc.tail, arc.head));
            seen_literals.push(arc.literal_var);
            outgoing[arc.tail] += 1;
            incoming[arc.head] += 1;
        }

        for node in 0..node_count {
            if outgoing[node] == 0 {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit node {node} has no outgoing candidate arcs"
                )));
            }
            if incoming[node] == 0 {
                return Err(MathProgramError::Unsupported(format!(
                    "circuit node {node} has no incoming candidate arcs"
                )));
            }
        }

        Ok(())
    }

    fn validate_multiple_circuit_args(
        &self,
        node_count: usize,
        arcs: &[CircuitArc],
    ) -> Result<(), MathProgramError> {
        if node_count < 2 {
            return Err(MathProgramError::Unsupported(
                "multiple-circuit requires at least two nodes".to_string(),
            ));
        }
        if arcs.is_empty() {
            return Err(MathProgramError::Unsupported(
                "multiple-circuit requires at least one directed arc".to_string(),
            ));
        }
        if arcs.len() > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "multiple-circuit exact MIP lowering is limited to 4096 arcs, got {}",
                arcs.len()
            )));
        }

        let mut incoming = vec![0usize; node_count];
        let mut outgoing = vec![0usize; node_count];
        let mut seen_arcs = Vec::with_capacity(arcs.len());
        let mut seen_literals = Vec::with_capacity(arcs.len());

        for arc in arcs {
            if arc.tail >= node_count || arc.head >= node_count {
                return Err(MathProgramError::BadIndex(format!(
                    "multiple-circuit arc {} -> {} is outside node range [0, {})",
                    arc.tail, arc.head, node_count
                )));
            }
            if arc.tail == 0 && arc.head == 0 {
                return Err(MathProgramError::Unsupported(
                    "multiple-circuit does not allow a depot self-loop".to_string(),
                ));
            }
            if arc.literal_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "multiple-circuit arc {} -> {} literal variable index {} out of bounds",
                    arc.tail, arc.head, arc.literal_var
                )));
            }
            let literal = &self.variables[arc.literal_var];
            if literal.var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "multiple-circuit arc {} -> {} literal `{}` must be binary",
                    arc.tail, arc.head, literal.name
                )));
            }
            if seen_arcs.contains(&(arc.tail, arc.head)) {
                return Err(MathProgramError::Unsupported(format!(
                    "multiple-circuit has duplicate directed arc {} -> {}",
                    arc.tail, arc.head
                )));
            }
            if seen_literals.contains(&arc.literal_var) {
                return Err(MathProgramError::Unsupported(format!(
                    "multiple-circuit literal `{}` is used by more than one arc",
                    literal.name
                )));
            }
            seen_arcs.push((arc.tail, arc.head));
            seen_literals.push(arc.literal_var);
            outgoing[arc.tail] += 1;
            incoming[arc.head] += 1;
        }

        for node in 1..node_count {
            if outgoing[node] == 0 {
                return Err(MathProgramError::Unsupported(format!(
                    "multiple-circuit node {node} has no outgoing candidate arcs"
                )));
            }
            if incoming[node] == 0 {
                return Err(MathProgramError::Unsupported(format!(
                    "multiple-circuit node {node} has no incoming candidate arcs"
                )));
            }
        }

        Ok(())
    }

    fn validate_reservoir_args(
        &self,
        events: &[ReservoirEvent],
        min_level: f64,
        max_level: f64,
    ) -> Result<(), MathProgramError> {
        if events.is_empty() {
            return Err(MathProgramError::Unsupported(
                "reservoir requires at least one event".to_string(),
            ));
        }
        if !min_level.is_finite() || !max_level.is_finite() || min_level > max_level {
            return Err(MathProgramError::InvalidBound(format!(
                "reservoir levels must be finite with min <= max, got [{min_level}, {max_level}]"
            )));
        }
        if min_level > 0.0 || max_level < 0.0 {
            return Err(MathProgramError::InvalidBound(format!(
                "reservoir initial level 0 must fit [{min_level}, {max_level}]"
            )));
        }

        let mut selector_count = 0usize;
        for (i, event) in events.iter().enumerate() {
            if event.time_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "reservoir event {i} time index {} out of bounds",
                    event.time_var
                )));
            }
            if !event.demand.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "reservoir event {i} demand"
                )));
            }
            let time_var = &self.variables[event.time_var];
            if !is_integer_time_var(time_var) {
                return Err(MathProgramError::Unsupported(format!(
                    "reservoir event {i} time `{}` must be binary or integer",
                    time_var.name
                )));
            }
            let (lower, upper) = integer_bounds(time_var).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "reservoir event {i} time `{}` requires finite integer bounds",
                    time_var.name
                ))
            })?;
            let domain_size = upper
                .checked_sub(lower)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!(
                        "reservoir event {i} time `{}` has an oversized domain",
                        time_var.name
                    ))
                })?;
            selector_count = selector_count
                .checked_add(domain_size as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported("reservoir selector count overflowed".to_string())
                })?;

            if let Some(active_var) = event.active_var {
                if active_var >= self.variables.len() {
                    return Err(MathProgramError::BadIndex(format!(
                        "reservoir event {i} active index {active_var} out of bounds"
                    )));
                }
                if self.variables[active_var].var_type != VariableType::Binary {
                    return Err(MathProgramError::Unsupported(format!(
                        "reservoir event {i} active literal `{}` must be binary",
                        self.variables[active_var].name
                    )));
                }
            }
        }

        if selector_count > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "reservoir exact MIP lowering is limited to 4096 time selectors, got {selector_count}"
            )));
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

    fn validate_global_cardinality_args(
        &self,
        variables: &[usize],
        counts: &[GlobalCardinalityCount],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "global-cardinality requires at least one variable".to_string(),
            ));
        }
        if counts.is_empty() {
            return Err(MathProgramError::Unsupported(
                "global-cardinality requires at least one counted value".to_string(),
            ));
        }

        let mut seen_values = BTreeSet::new();
        for count in counts {
            if count.min_count.is_none() && count.max_count.is_none() {
                return Err(MathProgramError::Unsupported(format!(
                    "global-cardinality value {} requires a min or max count",
                    count.value
                )));
            }
            if let (Some(min_count), Some(max_count)) = (count.min_count, count.max_count) {
                if min_count > max_count {
                    return Err(MathProgramError::Unsupported(format!(
                        "global-cardinality value {} minimum {min_count} exceeds maximum {max_count}",
                        count.value
                    )));
                }
            }
            if count
                .min_count
                .is_some_and(|min_count| min_count > variables.len())
                || count
                    .max_count
                    .is_some_and(|max_count| max_count > variables.len())
            {
                return Err(MathProgramError::Unsupported(format!(
                    "global-cardinality value {} count bound exceeds {} variables",
                    count.value,
                    variables.len()
                )));
            }
            if !seen_values.insert(count.value) {
                return Err(MathProgramError::Unsupported(format!(
                    "global-cardinality value {} is counted more than once",
                    count.value
                )));
            }
        }

        let mut literal_count = 0usize;
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "global-cardinality variable index {idx} out of bounds"
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "global-cardinality variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            let (lower, upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "global-cardinality variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
            let domain_size = upper
                .checked_sub(lower)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!(
                        "global-cardinality variable `{}` has an oversized domain",
                        self.variables[idx].name
                    ))
                })?;
            literal_count = literal_count
                .checked_add(domain_size as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported(
                        "global-cardinality value literal count overflowed".to_string(),
                    )
                })?;
            if literal_count > 4096 {
                return Err(MathProgramError::Unsupported(format!(
                    "global-cardinality exact MIP lowering is limited to 4096 value literals, got {literal_count}"
                )));
            }
        }
        Ok(())
    }

    fn validate_value_count_args(
        &self,
        variables: &[usize],
        _value: i64,
        count_var: usize,
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "value-count requires at least one variable".to_string(),
            ));
        }
        if count_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "value-count count variable index {count_var} out of bounds"
            )));
        }
        if !matches!(
            self.variables[count_var].var_type,
            VariableType::Binary | VariableType::Integer
        ) {
            return Err(MathProgramError::Unsupported(format!(
                "value-count count variable `{}` must be binary or integer",
                self.variables[count_var].name
            )));
        }
        let (count_lower, count_upper) =
            integer_bounds(&self.variables[count_var]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "value-count count variable `{}` requires finite integer bounds",
                    self.variables[count_var].name
                ))
            })?;
        if count_lower > variables.len() as i64 || count_upper < 0 {
            return Err(MathProgramError::InvalidBound(format!(
                "value-count count variable `{}` bounds [{count_lower}, {count_upper}] cannot contain any count in [0, {}]",
                self.variables[count_var].name,
                variables.len()
            )));
        }

        let mut literal_count = 0usize;
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "value-count variable index {idx} out of bounds"
                )));
            }
            if idx == count_var {
                return Err(MathProgramError::Unsupported(format!(
                    "value-count count variable `{}` must be distinct from counted variables",
                    self.variables[count_var].name
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "value-count variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            let (lower, upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "value-count variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
            let domain_size = upper
                .checked_sub(lower)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!(
                        "value-count variable `{}` has an oversized domain",
                        self.variables[idx].name
                    ))
                })?;
            literal_count = literal_count
                .checked_add(domain_size as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported(
                        "value-count value literal count overflowed".to_string(),
                    )
                })?;
            if literal_count > 4096 {
                return Err(MathProgramError::Unsupported(format!(
                    "value-count exact MIP lowering is limited to 4096 value literals, got {literal_count}"
                )));
            }
        }
        Ok(())
    }

    fn validate_linear_domain_args(
        &self,
        coeffs: &[(usize, f64)],
        intervals: &[LinearDomainInterval],
    ) -> Result<(), MathProgramError> {
        if coeffs.is_empty() {
            return Err(MathProgramError::Unsupported(
                "linear-domain constraints require at least one coefficient".to_string(),
            ));
        }
        if intervals.is_empty() {
            return Err(MathProgramError::Unsupported(
                "linear-domain constraints require at least one interval".to_string(),
            ));
        }
        if intervals.len() > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "linear-domain exact MIP lowering is limited to 4096 intervals, got {}",
                intervals.len()
            )));
        }
        validate_coeffs(self.variables.len(), coeffs)?;
        for interval in intervals {
            if interval.lower > interval.upper {
                return Err(MathProgramError::InvalidBound(format!(
                    "linear-domain interval [{}, {}] has lower bound above upper bound",
                    interval.lower, interval.upper
                )));
            }
        }
        for &(idx, coef) in coeffs {
            if !is_integer_value(coef) {
                return Err(MathProgramError::Unsupported(format!(
                    "linear-domain coefficient {coef} on `{}` must be integer-valued",
                    self.variables[idx].name
                )));
            }
            if integer_bounds(&self.variables[idx]).is_none() {
                return Err(MathProgramError::Unsupported(format!(
                    "linear-domain variable `{}` must be finite-domain integer or binary",
                    self.variables[idx].name
                )));
            }
        }
        linear_bounds(self, coeffs).ok_or_else(|| {
            MathProgramError::UnboundedBigM(
                "linear-domain constraint needs finite variable bounds for exact MIP lowering"
                    .to_string(),
            )
        })?;
        Ok(())
    }

    fn validate_enforced_linear_domain_args(
        &self,
        enforcement: &[BoolLiteral],
        coeffs: &[(usize, f64)],
        intervals: &[LinearDomainInterval],
    ) -> Result<(), MathProgramError> {
        if enforcement.is_empty() {
            return Err(MathProgramError::Unsupported(
                "enforced linear-domain constraints require at least one enforcement literal"
                    .to_string(),
            ));
        }
        self.validate_boolean_clause_args(enforcement)?;
        self.validate_linear_domain_args(coeffs, intervals)
    }

    fn linear_not_equal_intervals(
        &self,
        kind: &str,
        coeffs: &[(usize, f64)],
        forbidden_value: i64,
    ) -> Result<Vec<(i64, i64)>, MathProgramError> {
        self.linear_not_in_domain_intervals(
            kind,
            coeffs,
            &[LinearDomainInterval::exact(forbidden_value)],
        )
    }

    fn linear_not_in_domain_intervals(
        &self,
        kind: &str,
        coeffs: &[(usize, f64)],
        forbidden_intervals: &[LinearDomainInterval],
    ) -> Result<Vec<(i64, i64)>, MathProgramError> {
        if forbidden_intervals.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} constraints require at least one forbidden interval"
            )));
        }
        if forbidden_intervals.len() > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} exact MIP lowering is limited to 4096 forbidden intervals, got {}",
                forbidden_intervals.len()
            )));
        }
        for interval in forbidden_intervals {
            if interval.lower > interval.upper {
                return Err(MathProgramError::InvalidBound(format!(
                    "{kind} forbidden interval [{}, {}] has lower bound above upper bound",
                    interval.lower, interval.upper
                )));
            }
        }

        let (lower, upper) = self.integer_linear_expression_bounds(kind, coeffs)?;
        let forbidden_intervals = normalized_forbidden_intervals(forbidden_intervals, lower, upper);
        if forbidden_intervals.is_empty() {
            return Ok(vec![(lower, upper)]);
        }

        let mut intervals = Vec::new();
        let mut next_lower = lower;
        let mut covered_to_upper = false;
        for interval in forbidden_intervals {
            if next_lower < interval.lower {
                intervals.push((next_lower, interval.lower.saturating_sub(1)));
            }
            if interval.upper == i64::MAX {
                covered_to_upper = true;
                break;
            }
            next_lower = interval.upper + 1;
        }
        if !covered_to_upper && next_lower <= upper {
            intervals.push((next_lower, upper));
        }

        if intervals.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} excludes every integer value in its expression domain [{lower}, {upper}]"
            )));
        }
        if intervals.len() > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} exact MIP lowering is limited to 4096 allowed intervals after complementing, got {}",
                intervals.len()
            )));
        }
        Ok(intervals)
    }

    fn integer_linear_expression_bounds(
        &self,
        kind: &str,
        coeffs: &[(usize, f64)],
    ) -> Result<(i64, i64), MathProgramError> {
        if coeffs.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} constraints require at least one coefficient"
            )));
        }
        validate_coeffs(self.variables.len(), coeffs)?;
        let mut lower = 0.0;
        let mut upper = 0.0;
        for &(idx, coef) in coeffs {
            if !is_integer_value(coef) {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} coefficient {coef} on `{}` must be integer-valued",
                    self.variables[idx].name
                )));
            }
            let (var_lower, var_upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::Unsupported(format!(
                    "{kind} variable `{}` must be finite-domain integer or binary",
                    self.variables[idx].name
                ))
            })?;
            if coef >= 0.0 {
                lower += coef * var_lower as f64;
                upper += coef * var_upper as f64;
            } else {
                lower += coef * var_upper as f64;
                upper += coef * var_lower as f64;
            }
        }
        let lower = finite_integer_linear_bound(kind, "lower", lower)?;
        let upper = finite_integer_linear_bound(kind, "upper", upper)?;
        if lower > upper {
            return Err(MathProgramError::InvalidBound(format!(
                "{kind} expression lower bound {lower} is above upper bound {upper}"
            )));
        }
        Ok((lower, upper))
    }

    fn validate_map_domain_args(
        &self,
        var: usize,
        bool_vars: &[usize],
        offset: i64,
    ) -> Result<(), MathProgramError> {
        if var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "map-domain variable index {var} out of bounds"
            )));
        }
        if !matches!(
            self.variables[var].var_type,
            VariableType::Binary | VariableType::Integer
        ) {
            return Err(MathProgramError::Unsupported(format!(
                "map-domain variable `{}` must be binary or integer",
                self.variables[var].name
            )));
        }
        let (lower, upper) = integer_bounds(&self.variables[var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "map-domain variable `{}` requires finite integer bounds",
                self.variables[var].name
            ))
        })?;
        let domain_size = upper
            .checked_sub(lower)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                MathProgramError::Unsupported("map-domain source domain overflowed".to_string())
            })?;
        if domain_size > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "map-domain exact MIP lowering is limited to 512 source values, got {domain_size}"
            )));
        }
        if bool_vars.is_empty() {
            return Err(MathProgramError::Unsupported(
                "map-domain requires at least one selector variable".to_string(),
            ));
        }
        let mut seen = BTreeSet::new();
        for (pos, &bool_var) in bool_vars.iter().enumerate() {
            offset.checked_add(pos as i64).ok_or_else(|| {
                MathProgramError::Unsupported("map-domain target value overflowed".to_string())
            })?;
            if bool_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "map-domain selector index {bool_var} out of bounds"
                )));
            }
            if bool_var == var {
                return Err(MathProgramError::Unsupported(format!(
                    "map-domain selector `{}` must be distinct from the mapped variable",
                    self.variables[bool_var].name
                )));
            }
            if !seen.insert(bool_var) {
                return Err(MathProgramError::Unsupported(format!(
                    "map-domain does not support duplicate selector `{}`",
                    self.variables[bool_var].name
                )));
            }
            if self.variables[bool_var].var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "map-domain selector `{}` must be binary",
                    self.variables[bool_var].name
                )));
            }
        }
        Ok(())
    }

    fn validate_table_assignments_args(
        &self,
        kind: &str,
        variables: &[usize],
        tuples: &[Vec<i64>],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} requires at least one variable"
            )));
        }
        if tuples.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} requires at least one tuple"
            )));
        }
        if tuples.len() > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} exact MIP lowering is limited to 512 tuples, got {}",
                tuples.len(),
            )));
        }

        let mut bounds = Vec::with_capacity(variables.len());
        let mut value_literal_count = 0usize;
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "{kind} variable index {idx} out of bounds"
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            let (lower, upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "{kind} variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
            value_literal_count = value_literal_count
                .checked_add((upper - lower + 1) as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!("{kind} value literal count overflowed"))
                })?;
            bounds.push((lower, upper));
        }
        if value_literal_count > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} exact MIP lowering is limited to 4096 value literals, got {value_literal_count}"
            )));
        }

        for (row, tuple) in tuples.iter().enumerate() {
            if tuple.len() != variables.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} tuple {row} has length {}, expected {}",
                    tuple.len(),
                    variables.len()
                )));
            }
            for (col, &value) in tuple.iter().enumerate() {
                let (lower, upper) = bounds[col];
                if value < lower || value > upper {
                    return Err(MathProgramError::InvalidBound(format!(
                        "{kind} tuple {row} value {value} is outside bounds [{lower}, {upper}] for `{}`",
                        self.variables[variables[col]].name
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_enforced_table_assignments_args(
        &self,
        kind: &str,
        enforcement: &[BoolLiteral],
        variables: &[usize],
        tuples: &[Vec<i64>],
    ) -> Result<(), MathProgramError> {
        if enforcement.is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} requires at least one enforcement literal"
            )));
        }
        self.validate_boolean_clause_args(enforcement)?;
        self.validate_table_assignments_args(kind, variables, tuples)?;

        let mut seen_vars = BTreeSet::new();
        for &var in variables {
            if !seen_vars.insert(var) {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} does not support duplicate variable `{}`",
                    self.variables[var].name
                )));
            }
        }
        for literal in enforcement {
            if seen_vars.contains(&literal.var) {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} enforcement literal `{}` must be distinct from table variables",
                    self.variables[literal.var].name
                )));
            }
        }
        Ok(())
    }

    fn validate_bin_packing_args(
        &self,
        item_bin_vars: &[usize],
        load_vars: &[usize],
        item_sizes: &[f64],
    ) -> Result<(), MathProgramError> {
        if item_bin_vars.is_empty() {
            return Err(MathProgramError::Unsupported(
                "bin-packing requires at least one item".to_string(),
            ));
        }
        if load_vars.is_empty() {
            return Err(MathProgramError::Unsupported(
                "bin-packing requires at least one bin load variable".to_string(),
            ));
        }
        if item_bin_vars.len() != item_sizes.len() {
            return Err(MathProgramError::Unsupported(format!(
                "bin-packing requires one size per item, got {} items and {} sizes",
                item_bin_vars.len(),
                item_sizes.len()
            )));
        }

        for &load_var in load_vars {
            if load_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "bin-packing load variable index {load_var} out of bounds"
                )));
            }
        }

        let max_bin = i64::try_from(load_vars.len() - 1).map_err(|_| {
            MathProgramError::Unsupported(format!(
                "bin-packing bin count is too large: {} bins",
                load_vars.len()
            ))
        })?;
        let mut selector_count = 0usize;
        for (item, (&item_bin_var, &size)) in item_bin_vars.iter().zip(item_sizes).enumerate() {
            if !size.is_finite() || size < 0.0 {
                return Err(MathProgramError::InvalidBound(format!(
                    "bin-packing item {item} has invalid size {size}"
                )));
            }
            if item_bin_var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "bin-packing item {item} bin variable index {item_bin_var} out of bounds"
                )));
            }
            if !matches!(
                self.variables[item_bin_var].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "bin-packing item {item} variable `{}` must be binary or integer",
                    self.variables[item_bin_var].name
                )));
            }
            let (lower, upper) =
                integer_bounds(&self.variables[item_bin_var]).ok_or_else(|| {
                    MathProgramError::UnboundedBigM(format!(
                        "bin-packing item {item} variable `{}` requires finite integer bounds",
                        self.variables[item_bin_var].name
                    ))
                })?;
            if lower < 0 || upper > max_bin {
                return Err(MathProgramError::InvalidBound(format!(
                    "bin-packing item {item} variable `{}` bounds [{lower}, {upper}] must fit bin indices [0, {max_bin}]",
                    self.variables[item_bin_var].name
                )));
            }
            let domain_size = upper
                .checked_sub(lower)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| {
                    MathProgramError::Unsupported(format!(
                        "bin-packing item {item} domain size overflowed"
                    ))
                })?;
            selector_count = selector_count
                .checked_add(domain_size as usize)
                .ok_or_else(|| {
                    MathProgramError::Unsupported(
                        "bin-packing selector count overflowed".to_string(),
                    )
                })?;
            if selector_count > 4096 {
                return Err(MathProgramError::Unsupported(format!(
                    "bin-packing exact MIP lowering is limited to 4096 selectors, got {selector_count}"
                )));
            }
        }
        Ok(())
    }

    fn validate_element_args(
        &self,
        index_var: usize,
        target_var: usize,
        values: &[f64],
    ) -> Result<(), MathProgramError> {
        if values.is_empty() {
            return Err(MathProgramError::Unsupported(
                "element requires at least one value".to_string(),
            ));
        }
        if values.len() > i64::MAX as usize {
            return Err(MathProgramError::Unsupported(format!(
                "element value array is too large: {} values",
                values.len()
            )));
        }
        for (pos, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "element value at index {pos}"
                )));
            }
        }
        if index_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "element index variable {index_var} out of bounds"
            )));
        }
        if target_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "element target variable {target_var} out of bounds"
            )));
        }

        let index = &self.variables[index_var];
        if !matches!(index.var_type, VariableType::Binary | VariableType::Integer) {
            return Err(MathProgramError::Unsupported(format!(
                "element index variable `{}` must be binary or integer",
                index.name
            )));
        }
        let (lower, upper) = integer_bounds(index).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "element index variable `{}` requires finite integer bounds",
                index.name
            ))
        })?;
        let max_index = i64::try_from(values.len() - 1).map_err(|_| {
            MathProgramError::Unsupported(format!(
                "element value array is too large: {} values",
                values.len()
            ))
        })?;
        if lower < 0 || upper > max_index {
            return Err(MathProgramError::InvalidBound(format!(
                "element index variable `{}` bounds [{lower}, {upper}] must fit value indices [0, {max_index}]",
                index.name
            )));
        }
        let domain_size = upper
            .checked_sub(lower)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                MathProgramError::Unsupported(
                    "element index variable domain size overflowed".to_string(),
                )
            })?;
        if domain_size > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "element exact MIP lowering is limited to 512 index literals, got {domain_size}"
            )));
        }

        let target = &self.variables[target_var];
        for idx in lower..=upper {
            let value = values[idx as usize];
            if variable_domain_violation(target, value) > 1e-9 {
                return Err(MathProgramError::InvalidBound(format!(
                    "element value {value} at index {idx} is outside target variable `{}` domain",
                    target.name
                )));
            }
        }

        Ok(())
    }

    fn validate_variable_element_args(
        &self,
        index_var: usize,
        target_var: usize,
        variables: &[usize],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "variable-element requires at least one variable".to_string(),
            ));
        }
        if variables.len() > i64::MAX as usize {
            return Err(MathProgramError::Unsupported(format!(
                "variable-element array is too large: {} variables",
                variables.len()
            )));
        }
        if index_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "variable-element index variable {index_var} out of bounds"
            )));
        }
        if target_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "variable-element target variable {target_var} out of bounds"
            )));
        }
        for (pos, &var_idx) in variables.iter().enumerate() {
            if var_idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "variable-element source variable at index {pos} ({var_idx}) out of bounds"
                )));
            }
        }

        let index = &self.variables[index_var];
        if !matches!(index.var_type, VariableType::Binary | VariableType::Integer) {
            return Err(MathProgramError::Unsupported(format!(
                "variable-element index variable `{}` must be binary or integer",
                index.name
            )));
        }
        let (lower, upper) = integer_bounds(index).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "variable-element index variable `{}` requires finite integer bounds",
                index.name
            ))
        })?;
        let max_index = i64::try_from(variables.len() - 1).map_err(|_| {
            MathProgramError::Unsupported(format!(
                "variable-element array is too large: {} variables",
                variables.len()
            ))
        })?;
        if lower < 0 || upper > max_index {
            return Err(MathProgramError::InvalidBound(format!(
                "variable-element index variable `{}` bounds [{lower}, {upper}] must fit variable indices [0, {max_index}]",
                index.name
            )));
        }
        let domain_size = upper
            .checked_sub(lower)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                MathProgramError::Unsupported(
                    "variable-element index variable domain size overflowed".to_string(),
                )
            })?;
        if domain_size > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "variable-element exact MIP lowering is limited to 512 index literals, got {domain_size}"
            )));
        }

        let target = &self.variables[target_var];
        variable_bounds(target).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "variable-element target variable `{}` requires finite bounds",
                target.name
            ))
        })?;
        for idx in lower..=upper {
            let source = &self.variables[variables[idx as usize]];
            variable_bounds(source).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "variable-element source variable `{}` at index {idx} requires finite bounds",
                    source.name
                ))
            })?;
        }

        Ok(())
    }

    fn validate_inverse_args(
        &self,
        variables: &[usize],
        inverse_variables: &[usize],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "inverse requires at least one variable".to_string(),
            ));
        }
        if variables.len() != inverse_variables.len() {
            return Err(MathProgramError::Unsupported(format!(
                "inverse requires equally-sized variable arrays, got {} and {}",
                variables.len(),
                inverse_variables.len()
            )));
        }
        let literal_count = variables
            .len()
            .checked_mul(variables.len())
            .ok_or_else(|| {
                MathProgramError::Unsupported("inverse literal count overflowed".to_string())
            })?;
        if literal_count > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "inverse exact MIP lowering is limited to 4096 literals, got {literal_count}"
            )));
        }

        self.validate_inverse_side("inverse variable", variables)?;
        self.validate_inverse_side("inverse mirror variable", inverse_variables)?;
        Ok(())
    }

    fn validate_inverse_side(
        &self,
        kind: &str,
        variables: &[usize],
    ) -> Result<(), MathProgramError> {
        let mut seen = Vec::with_capacity(variables.len());
        let max_value = variables.len() as i64 - 1;
        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "{kind} index {idx} out of bounds"
                )));
            }
            if seen.contains(&idx) {
                return Err(MathProgramError::Unsupported(format!(
                    "inverse does not support duplicate {kind} `{}`",
                    self.variables[idx].name
                )));
            }
            seen.push(idx);
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            let (lower, upper) = integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "{kind} `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
            if lower < 0 || upper > max_value {
                return Err(MathProgramError::InvalidBound(format!(
                    "{kind} `{}` bounds [{lower}, {upper}] must fit inverse values [0, {max_value}]",
                    self.variables[idx].name
                )));
            }
        }
        Ok(())
    }

    fn validate_automaton_args(
        &self,
        variables: &[usize],
        starting_state: i64,
        final_states: &[i64],
        transitions: &[AutomatonTransition],
    ) -> Result<(), MathProgramError> {
        if variables.is_empty() {
            return Err(MathProgramError::Unsupported(
                "automaton requires at least one transition variable".to_string(),
            ));
        }
        if final_states.is_empty() {
            return Err(MathProgramError::Unsupported(
                "automaton requires at least one final state".to_string(),
            ));
        }
        if transitions.is_empty() {
            return Err(MathProgramError::Unsupported(
                "automaton requires at least one transition".to_string(),
            ));
        }
        if transitions.len() > 1024 {
            return Err(MathProgramError::Unsupported(format!(
                "automaton exact MIP lowering is limited to 1024 transitions, got {}",
                transitions.len()
            )));
        }

        for &idx in variables {
            if idx >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "automaton variable index {idx} out of bounds"
                )));
            }
            if !matches!(
                self.variables[idx].var_type,
                VariableType::Binary | VariableType::Integer
            ) {
                return Err(MathProgramError::Unsupported(format!(
                    "automaton variable `{}` must be binary or integer",
                    self.variables[idx].name
                )));
            }
            integer_bounds(&self.variables[idx]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "automaton variable `{}` requires finite integer bounds",
                    self.variables[idx].name
                ))
            })?;
        }

        let states = automaton_states(starting_state, final_states, transitions);
        let literal_count = (variables.len() + 1)
            .checked_mul(states.len())
            .and_then(|count| count.checked_add(variables.len() * transitions.len()))
            .ok_or_else(|| {
                MathProgramError::Unsupported(
                    "automaton exact MIP lowering literal count overflowed".to_string(),
                )
            })?;
        if literal_count > 4096 {
            return Err(MathProgramError::Unsupported(format!(
                "automaton exact MIP lowering is limited to 4096 literals, got {literal_count}"
            )));
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

    fn validate_boolean_general_args(
        &self,
        kind: &str,
        result_var: usize,
        literals: &[BoolLiteral],
    ) -> Result<(), MathProgramError> {
        if result_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{kind} result index {result_var} out of bounds"
            )));
        }
        if self.variables[result_var].var_type != VariableType::Binary {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} result `{}` must be binary",
                self.variables[result_var].name
            )));
        }
        self.validate_boolean_clause_args(literals)
    }

    fn validate_binary_cardinality_args(
        &self,
        operands: &[usize],
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<(), MathProgramError> {
        if operands.is_empty() {
            return Err(MathProgramError::Unsupported(
                "binary cardinality constraints require at least one operand".to_string(),
            ));
        }
        if min_count.is_none() && max_count.is_none() {
            return Err(MathProgramError::Unsupported(
                "binary cardinality constraints require at least one count bound".to_string(),
            ));
        }
        if let Some(min_count) = min_count {
            if min_count > operands.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "binary cardinality minimum {min_count} exceeds {} operands",
                    operands.len()
                )));
            }
        }
        if let Some(max_count) = max_count {
            if max_count > operands.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "binary cardinality maximum {max_count} exceeds {} operands",
                    operands.len()
                )));
            }
        }
        if let (Some(min_count), Some(max_count)) = (min_count, max_count) {
            if min_count > max_count {
                return Err(MathProgramError::Unsupported(format!(
                    "binary cardinality minimum {min_count} exceeds maximum {max_count}"
                )));
            }
        }
        for &operand in operands {
            if operand >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "binary cardinality operand index {operand} out of bounds"
                )));
            }
            if self.variables[operand].var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "binary cardinality operand `{}` must be binary",
                    self.variables[operand].name
                )));
            }
        }
        Ok(())
    }

    fn validate_boolean_clause_args(
        &self,
        literals: &[BoolLiteral],
    ) -> Result<(), MathProgramError> {
        if literals.is_empty() {
            return Err(MathProgramError::Unsupported(
                "boolean clauses require at least one literal".to_string(),
            ));
        }
        for literal in literals {
            if literal.var >= self.variables.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "boolean clause literal variable index {} out of bounds",
                    literal.var
                )));
            }
            if self.variables[literal.var].var_type != VariableType::Binary {
                return Err(MathProgramError::Unsupported(format!(
                    "boolean clause literal `{}` must be binary",
                    self.variables[literal.var].name
                )));
            }
        }
        Ok(())
    }

    fn validate_boolean_cardinality_args(
        &self,
        literals: &[BoolLiteral],
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<(), MathProgramError> {
        if literals.is_empty() {
            return Err(MathProgramError::Unsupported(
                "boolean cardinality constraints require at least one literal".to_string(),
            ));
        }
        if min_count.is_none() && max_count.is_none() {
            return Err(MathProgramError::Unsupported(
                "boolean cardinality constraints require at least one count bound".to_string(),
            ));
        }
        if let Some(min_count) = min_count {
            if min_count > literals.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "boolean cardinality minimum {min_count} exceeds {} literals",
                    literals.len()
                )));
            }
        }
        if let Some(max_count) = max_count {
            if max_count > literals.len() {
                return Err(MathProgramError::Unsupported(format!(
                    "boolean cardinality maximum {max_count} exceeds {} literals",
                    literals.len()
                )));
            }
        }
        if let (Some(min_count), Some(max_count)) = (min_count, max_count) {
            if min_count > max_count {
                return Err(MathProgramError::Unsupported(format!(
                    "boolean cardinality minimum {min_count} exceeds maximum {max_count}"
                )));
            }
        }
        self.validate_boolean_clause_args(literals)
    }

    fn validate_enforced_boolean_cardinality_args(
        &self,
        enforcement: &[BoolLiteral],
        literals: &[BoolLiteral],
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Result<(), MathProgramError> {
        if enforcement.is_empty() {
            return Err(MathProgramError::Unsupported(
                "enforced Boolean/cardinality constraints require at least one enforcement literal"
                    .to_string(),
            ));
        }
        self.validate_boolean_clause_args(enforcement)?;
        self.validate_boolean_cardinality_args(literals, min_count, max_count)
    }

    fn validate_boolean_count_args(
        &self,
        literals: &[BoolLiteral],
        count_var: usize,
    ) -> Result<(), MathProgramError> {
        self.validate_boolean_clause_args(literals)?;
        if count_var >= self.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "boolean count variable index {count_var} out of bounds"
            )));
        }
        if !matches!(
            self.variables[count_var].var_type,
            VariableType::Binary | VariableType::Integer
        ) {
            return Err(MathProgramError::Unsupported(format!(
                "boolean count variable `{}` must be binary or integer",
                self.variables[count_var].name
            )));
        }
        let (lower, upper) = integer_bounds(&self.variables[count_var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "boolean count variable `{}` requires finite integer bounds",
                self.variables[count_var].name
            ))
        })?;
        if lower > literals.len() as i64 || upper < 0 {
            return Err(MathProgramError::InvalidBound(format!(
                "boolean count variable `{}` bounds [{lower}, {upper}] cannot contain any count in [0, {}]",
                self.variables[count_var].name,
                literals.len()
            )));
        }
        Ok(())
    }

    fn validate_enforced_linear_args(
        &self,
        literals: &[BoolLiteral],
        coeffs: &[(usize, f64)],
        rhs: f64,
    ) -> Result<(), MathProgramError> {
        if literals.is_empty() {
            return Err(MathProgramError::Unsupported(
                "enforced linear constraints require at least one literal".to_string(),
            ));
        }
        self.validate_boolean_clause_args(literals)?;
        validate_coeffs(self.variables.len(), coeffs)?;
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite(
                "enforced linear rhs".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_reified_linear_args(
        &self,
        literal: BoolLiteral,
        coeffs: &[(usize, f64)],
        _sense: RowSense,
        rhs: f64,
    ) -> Result<(), MathProgramError> {
        self.validate_boolean_clause_args(&[literal])?;
        if coeffs.is_empty() {
            return Err(MathProgramError::Unsupported(
                "reified linear constraints require at least one coefficient".to_string(),
            ));
        }
        validate_coeffs(self.variables.len(), coeffs)?;
        if !rhs.is_finite() {
            return Err(MathProgramError::NonFinite(
                "reified linear rhs".to_string(),
            ));
        }
        if !is_integer_value(rhs) {
            return Err(MathProgramError::Unsupported(format!(
                "reified linear rhs {rhs} must be integer-valued for exact finite-domain lowering"
            )));
        }
        for &(idx, coef) in coeffs {
            if !is_integer_value(coef) {
                return Err(MathProgramError::Unsupported(format!(
                    "reified linear coefficient {coef} on `{}` must be integer-valued",
                    self.variables[idx].name
                )));
            }
            if integer_bounds(&self.variables[idx]).is_none() {
                return Err(MathProgramError::Unsupported(format!(
                    "reified linear variable `{}` must be finite-domain integer or binary",
                    self.variables[idx].name
                )));
            }
        }
        linear_bounds(self, coeffs).ok_or_else(|| {
            MathProgramError::UnboundedBigM(
                "reified linear constraint needs finite variable bounds for big-M lowering"
                    .to_string(),
            )
        })?;
        Ok(())
    }

    fn validate_integer_product_args(
        &self,
        target_var: usize,
        operands: &[usize],
    ) -> Result<(), MathProgramError> {
        integer_product_variables_and_tuples(self, target_var, operands).map(|_| ())
    }

    fn validate_integer_binary_operation_args(
        &self,
        kind: &str,
        target_var: usize,
        numerator_var: usize,
        denominator_var: usize,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<(), MathProgramError> {
        integer_binary_operation_variables_and_tuples(
            self,
            kind,
            target_var,
            numerator_var,
            denominator_var,
            operation,
        )
        .map(|_| ())
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

    fn univariate_function_points<F>(
        &self,
        kind: &str,
        breakpoints: &[f64],
        function: F,
    ) -> Result<Vec<(f64, f64)>, MathProgramError>
    where
        F: Fn(f64) -> f64,
    {
        if breakpoints.len() < 2 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} requires at least two breakpoints"
            )));
        }
        for &x in breakpoints {
            if !x.is_finite() {
                return Err(MathProgramError::NonFinite(format!("{kind} breakpoint")));
            }
        }
        for pair in breakpoints.windows(2) {
            if pair[1] <= pair[0] {
                return Err(MathProgramError::Unsupported(format!(
                    "{kind} breakpoints must be strictly increasing"
                )));
            }
        }
        breakpoints
            .iter()
            .map(|&x| {
                let y = function(x);
                if y.is_finite() {
                    Ok((x, y))
                } else {
                    Err(MathProgramError::NonFinite(format!(
                        "{kind} value at breakpoint {x}"
                    )))
                }
            })
            .collect()
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
            if let Some(duration_var) = interval.duration_var {
                if duration_var >= self.variables.len() {
                    return Err(MathProgramError::BadIndex(format!(
                        "{kind} interval {i} duration index {duration_var} out of bounds"
                    )));
                }
                let (duration_lb, _) =
                    variable_bounds(&self.variables[duration_var]).ok_or_else(|| {
                        MathProgramError::UnboundedBigM(format!(
                            "{kind} interval {i} duration `{}` requires finite bounds",
                            self.variables[duration_var].name
                        ))
                    })?;
                if duration_lb + interval.duration < -1e-9 {
                    return Err(MathProgramError::InvalidBound(format!(
                        "{kind} interval {i} duration `{}` can be negative",
                        self.variables[duration_var].name
                    )));
                }
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

    fn validate_alternative_args(
        &self,
        master: &IntervalTerm,
        alternatives: &[IntervalTerm],
    ) -> Result<(), MathProgramError> {
        self.validate_interval_args("alternative master", std::slice::from_ref(master))?;
        self.validate_interval_args("alternative", alternatives)?;
        let mut seen_presence = Vec::new();
        for (i, alternative) in alternatives.iter().enumerate() {
            let presence = alternative.presence_var.ok_or_else(|| {
                MathProgramError::Unsupported(format!("alternative interval {i} must be optional"))
            })?;
            if seen_presence.contains(&presence) {
                return Err(MathProgramError::Unsupported(format!(
                    "alternative interval {i} reuses presence variable `{}`",
                    self.variables[presence].name
                )));
            }
            seen_presence.push(presence);
        }
        Ok(())
    }

    fn validate_cumulative_args(
        &self,
        intervals: &[IntervalTerm],
        demands: &[AffineTerm],
        capacity: &AffineTerm,
    ) -> Result<(), MathProgramError> {
        self.validate_interval_args("cumulative", intervals)?;
        if intervals.len() != demands.len() {
            return Err(MathProgramError::Unsupported(format!(
                "cumulative requires one demand per interval, got {} intervals and {} demands",
                intervals.len(),
                demands.len()
            )));
        }
        self.validate_nonnegative_bounded_affine("cumulative capacity", capacity)?;
        for (i, demand) in demands.iter().enumerate() {
            self.validate_nonnegative_bounded_affine(&format!("cumulative demand {i}"), demand)?;
        }
        for (i, interval) in intervals.iter().enumerate() {
            if !is_integer_time_var(&self.variables[interval.start_var])
                || !is_integer_time_var(&self.variables[interval.end_var])
            {
                return Err(MathProgramError::Unsupported(format!(
                    "cumulative interval {i} start/end variables must be integer-time variables"
                )));
            }
            if !is_integer_value(interval.duration) {
                return Err(MathProgramError::Unsupported(format!(
                    "cumulative interval {i} fixed duration offset {} must be an integer",
                    interval.duration
                )));
            }
            if let Some(duration_var) = interval.duration_var {
                if !is_integer_time_var(&self.variables[duration_var]) {
                    return Err(MathProgramError::Unsupported(format!(
                        "cumulative interval {i} duration variable `{}` must be an integer-time variable",
                        self.variables[duration_var].name
                    )));
                }
                integer_bounds(&self.variables[duration_var]).ok_or_else(|| {
                    MathProgramError::UnboundedBigM(format!(
                        "cumulative interval {i} duration variable `{}` requires finite integer bounds",
                        self.variables[duration_var].name
                    ))
                })?;
            }
        }
        Ok(())
    }

    fn validate_nonnegative_bounded_affine(
        &self,
        kind: &str,
        term: &AffineTerm,
    ) -> Result<(), MathProgramError> {
        validate_coeffs(self.variables.len(), &term.coeffs)?;
        if !term.constant.is_finite() {
            return Err(MathProgramError::InvalidBound(format!(
                "{kind} constant must be finite, got {}",
                term.constant
            )));
        }
        let (lower, _upper) = affine_bounds(self, term).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "{kind} requires finite bounds for every referenced variable"
            ))
        })?;
        if lower < -1e-9 {
            return Err(MathProgramError::InvalidBound(format!(
                "{kind} can be negative over its variable bounds; lower bound is {lower}"
            )));
        }
        Ok(())
    }
}

/// Which LP backend to use for pure continuous models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProgramLpBackend {
    InternalSimplex,
    DESSimplex,
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
    pub lp_des: DESSimplexOptions,
    pub lp_ipm: InternalInteriorPointOptions,
    pub external_lp: ExternalSolverOptions,
    pub qp: MathProgramQpOptions,
    pub conic: MathProgramConicOptions,
    pub mip: IPMIPSolveOptions,
    /// Optional MIP start in the original math-program variable space.
    pub mip_start: Option<Vec<f64>>,
    /// Optional branching priorities in the original math-program variable space.
    /// Higher values branch before lower values in the native MIP backend. When
    /// set, these are mapped through the math-program compiler and override
    /// `mip.branch_priorities`, which remains available for canonical variables.
    pub branch_priorities: Option<Vec<i32>>,
}

impl Default for MathProgramSolveOptions {
    fn default() -> Self {
        MathProgramSolveOptions {
            lp_backend: MathProgramLpBackend::InternalSimplex,
            lp_simplex: InternalSimplexOptions::default(),
            lp_des: DESSimplexOptions::default(),
            lp_ipm: InternalInteriorPointOptions::default(),
            external_lp: ExternalSolverOptions::default(),
            qp: MathProgramQpOptions::default(),
            conic: MathProgramConicOptions::default(),
            mip: IPMIPSolveOptions::default(),
            mip_start: None,
            branch_priorities: None,
        }
    }
}

/// Options for an optional external-solver oracle.
///
/// Explicit local CLI methods such as `highs-cli`, `highs:cli`, `glpk-cli`,
/// `cbc-cli`, `clp-cli`, `soplex-cli`, `qsopt-ex-cli`, `lp-solve-cli`, and
/// `lindo-cli` use the Rust `external_linear_cli` adapter for LP/MIP models.
/// When `method` is omitted for linear LP/MIP models, the facade defaults to
/// the Rust-native `highs-cli` adapter. Python remains the fallback bridge for
/// explicit SciPy, OR-Tools, nonlinear oracles, and external API bindings.
/// Compatibility HiGHS, GLPK, SCIP, CBC, CLP, SoPlex, QSopt_ex, and lp_solve
/// method names use the Rust CLI adapter for LP/MIP models unless a Python
/// executable or script is explicitly provided.
/// The Python bridge remains a fallback when an explicit bridge is requested
/// but its optional Python solver stack is missing.
#[derive(Clone, Debug, Default)]
pub struct ExternalMathProgramOptions {
    /// Solver method, such as `highs`, `highs-cli`, `clp-cli`,
    /// `soplex-cli`, `qsopt-ex-cli`, `lp-solve-cli`, `ortools:GLOP`,
    /// `ortools:PDLP`, `ortools:SCIP`, `ortools:CP-SAT`, `glpk:default`,
    /// `gurobi:default`, `cplex:default`, `xpress:default`, or `lindo-cli`.
    pub method: Option<String>,
    /// Python executable. Defaults to `PYTHON_BIN`, then `PYTHON`, then `python3`.
    pub python: Option<String>,
    /// Script path. Defaults to `external-references/math-program/math_program_solve.py`.
    pub script: Option<String>,
    /// Optional MIP start in the original math-program variable space.
    pub mip_start: Option<Vec<f64>>,
    /// Optional external solver wall-clock limit in milliseconds.
    pub time_limit_ms: Option<f64>,
    /// Optional external MIP search node limit.
    pub node_limit: Option<usize>,
    /// Optional external relative MIP optimality gap.
    pub relative_gap: Option<f64>,
    /// Optional external absolute MIP optimality gap.
    pub absolute_gap: Option<f64>,
    /// Optional incumbent solution limit for MIP-style external solves.
    pub solution_limit: Option<u64>,
    /// Optional solution-pool target size for external CLI solves that expose it.
    pub solution_pool_size: Option<u64>,
    /// Optional incumbent objective target for external MIP-style solves.
    pub objective_limit: Option<f64>,
    /// Optional primal/row feasibility tolerance for external LP/MIP solves.
    pub primal_feasibility_tolerance: Option<f64>,
    /// Optional dual/reduced-cost feasibility tolerance for external LP/MIP solves.
    pub dual_feasibility_tolerance: Option<f64>,
    /// Optional integrality tolerance for external MIP solves.
    pub integer_feasibility_tolerance: Option<f64>,
    /// Optional worker thread count for external CLI/API solves that expose it.
    pub threads: Option<u32>,
    /// Optional random seed for external CLI/API solves that expose it.
    pub random_seed: Option<u64>,
    /// Optional presolve mode for external CLI solves that expose it.
    pub presolve: Option<ExternalLinearCliPresolve>,
    /// Optional cut-generation mode for external MIP CLI solves that expose it.
    pub cuts: Option<ExternalLinearCliMipSwitch>,
    /// Optional heuristic-search mode for external MIP CLI solves that expose it.
    pub heuristics: Option<ExternalLinearCliMipSwitch>,
    /// Optional branching rule for external MIP CLI solves that expose it.
    pub branch_rule: Option<ExternalLinearCliBranchRule>,
    /// Optional branching priorities in the original math-program variable space.
    pub branch_priorities: Option<Vec<i32>>,
    /// Optional node-selection rule for external MIP CLI solves that expose it.
    pub node_selection: Option<ExternalLinearCliNodeSelection>,
}

/// Facade status normalized across LP and IP/MIP solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProgramStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    IterLimit,
    NodeLimit,
    TimeLimit,
    NumericalError,
}

/// Solver-control feedback reported by an external backend.
///
/// These fields confirm that solver-style options such as solution limits,
/// objective targets, search strategy switches, and branch priorities survived
/// the facade-to-backend adapter path when the selected backend exposes such
/// diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MathProgramSolverControlFeedback {
    pub solution_limit: Option<u64>,
    pub solution_pool_size: Option<u64>,
    pub objective_limit: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub primal_feasibility_tolerance: Option<f64>,
    pub dual_feasibility_tolerance: Option<f64>,
    pub integer_feasibility_tolerance: Option<f64>,
    pub threads: Option<u32>,
    pub random_seed: Option<u64>,
    pub presolve: Option<String>,
    pub cuts: Option<String>,
    pub heuristics: Option<String>,
    pub branch_rule: Option<String>,
    pub node_selection: Option<String>,
    pub branch_priorities_accepted: Option<bool>,
    pub branch_priority_count: Option<usize>,
    pub mip_start_accepted: Option<bool>,
    pub mip_start_objective: Option<f64>,
}

/// Solution mapped back to the original variables.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramSolution {
    pub status: MathProgramStatus,
    pub x: Vec<f64>,
    pub objective: f64,
    /// Best known objective bound for MIP-style solves, in the original objective direction.
    pub best_bound: Option<f64>,
    /// Relative optimality gap for MIP-style solves when the backend reports one.
    pub mip_gap: Option<f64>,
    /// Number of explored MIP search nodes when the backend reports one.
    pub nodes_explored: Option<usize>,
    /// First MIP branch variable reported by the native backend, mapped back to
    /// the original facade variable name when possible.
    pub first_branch_variable: Option<String>,
    /// Version/build string reported by the solver backend, when available.
    pub solver_version: Option<String>,
    /// Algorithm iterations reported by the solver backend, when available.
    pub iterations: Option<usize>,
    /// Solver-control feedback reported by the backend, when available.
    pub control_feedback: Option<MathProgramSolverControlFeedback>,
    /// LP row shadow prices for the `LPProblem` inequality rows generated by the facade.
    pub dual_ub: Option<Vec<f64>>,
    /// LP row shadow prices for the `LPProblem` equality rows generated by the facade.
    pub dual_eq: Option<Vec<f64>>,
    /// LP bound reduced costs in the original objective direction.
    pub reduced_costs: Option<Vec<f64>>,
    /// LP basis status for original variables when reported by the backend.
    pub var_basis: Option<Vec<String>>,
    /// LP basis status for generated LP rows when reported by the backend.
    pub row_basis: Option<Vec<String>>,
    /// Primal improving ray for unbounded continuous linear models, in original variables.
    pub unbounded_ray: Option<Vec<f64>>,
    /// Farkas-style certificate for infeasible generated continuous LP models.
    pub infeasibility_certificate: Option<LPInfeasibilityCertificate>,
    pub solver: String,
    pub message: Option<String>,
}

/// LP row certificate mapped from generated LP row order back to a facade row.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramLpRowCertificate {
    /// Row index in the generated LP certificate vectors (`dual_ub + dual_eq`, row basis).
    pub generated_row: usize,
    /// Name of the modeled facade constraint.
    pub source_name: String,
    /// Index in `MathProgram::constraints`.
    pub source_index: usize,
    /// Sense of the modeled facade constraint.
    pub source_sense: RowSense,
    /// RHS of the modeled facade constraint.
    pub source_rhs: f64,
    /// Sense used in the generated LP row.
    pub generated_sense: RowSense,
    /// RHS used in the generated LP row.
    pub generated_rhs: f64,
    /// Multiplier applied to the source row to produce the generated LP row.
    pub multiplier: f64,
    /// Dual value in generated LP row coordinates, when reported by the backend.
    pub generated_dual: Option<f64>,
    /// Dual value mapped back to the source row coordinates, when reported by the backend.
    pub source_dual: Option<f64>,
    /// Row basis status in generated LP row order, when reported by the backend.
    pub basis: Option<String>,
}

/// LP variable certificate mapped from generated LP vectors back to a facade variable.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramLpVariableCertificate {
    /// Variable index in `MathProgram::variables`.
    pub variable: usize,
    /// Name of the modeled facade variable.
    pub name: String,
    /// Facade variable domain.
    pub var_type: VariableType,
    /// Objective coefficient in the modeled facade objective.
    pub objective_coefficient: f64,
    /// Lower bound in the modeled facade variable domain.
    pub lb: Option<f64>,
    /// Upper bound in the modeled facade variable domain.
    pub ub: Option<f64>,
    /// Primal value reported in `MathProgramSolution::x`, when present.
    pub value: Option<f64>,
    /// Reduced cost in the original objective direction, when reported by the backend.
    pub reduced_cost: Option<f64>,
    /// Variable basis status, when reported by the backend.
    pub basis: Option<String>,
}

impl MathProgramSolution {
    /// Map generated LP row duals/basis statuses back to named facade constraints.
    pub fn lp_row_certificates(
        &self,
        program: &MathProgram,
    ) -> Result<Vec<MathProgramLpRowCertificate>, MathProgramError> {
        map_math_program_lp_row_certificates(program, self)
    }

    /// Map LP reduced costs and variable basis statuses back to named facade variables.
    pub fn lp_variable_certificates(
        &self,
        program: &MathProgram,
    ) -> Result<Vec<MathProgramLpVariableCertificate>, MathProgramError> {
        map_math_program_lp_variable_certificates(program, self)
    }
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

/// Options for native deletion-filter assumption-core refinement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathProgramAssumptionCoreOptions {
    pub max_candidate_checks: usize,
}

impl Default for MathProgramAssumptionCoreOptions {
    fn default() -> Self {
        MathProgramAssumptionCoreOptions {
            max_candidate_checks: usize::MAX,
        }
    }
}

/// Minimal infeasible subset of Boolean assumptions for a math program.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramAssumptionCore {
    pub status: MathProgramStatus,
    pub assumptions: Vec<BoolLiteral>,
    pub subsystem: MathProgram,
    pub minimal: bool,
    pub solver: String,
    pub message: Option<String>,
}

/// Internal-vs-external check for the refined assumption-core subsystem.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramAssumptionCoreCrossCheck {
    pub internal: MathProgramAssumptionCore,
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

/// One exported-column term in the affine reconstruction of an original variable.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramExportTerm {
    pub exported_var: usize,
    pub exported_name: String,
    pub coefficient: f64,
}

/// Affine mapping from one original facade variable to exported LP/MPS columns.
///
/// For compiled MIP exports, free and shifted variables may map to generated
/// nonnegative columns. The original variable value is
/// `constant + sum(coefficient * exported_x[exported_var])`.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramExportVariableExpansion {
    pub original_var: usize,
    pub original_name: String,
    pub constant: f64,
    pub terms: Vec<MathProgramExportTerm>,
}

/// Origin of an exported LP/MPS row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProgramExportRowKind {
    LinearConstraint,
    QuadraticConstraint,
    LazyConstraint,
    Generated,
}

/// Metadata for one exported LP/MPS row.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramExportRowMapping {
    pub exported_row: usize,
    pub exported_name: String,
    pub source_kind: MathProgramExportRowKind,
    /// Index in the source row collection when available. For ordinary LP
    /// exports this is an index into `constraints` or `lazy_constraints`; for
    /// compiled models it is the generated row index.
    pub source_index: Option<usize>,
    pub source_name: String,
    pub sense: RowSense,
    pub rhs: f64,
    /// Multiplier between the source row activity and exported row activity.
    /// Generated compiled rows use `1.0`.
    pub multiplier: f64,
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

/// Compute LP objective-coefficient sensitivity ranges for a pure continuous
/// linear `MathProgram`.
pub fn analyze_math_program_objective_sensitivity(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    sensitivity_opts: &LPObjectiveSensitivityOptions,
) -> Result<LPObjectiveSensitivityReport, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty()
        || program.has_discrete_features()
        || program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
    {
        return Err(MathProgramError::Unsupported(
            "objective sensitivity requires a pure continuous single-objective linear model"
                .to_string(),
        ));
    }
    let lp = program.to_lp_problem()?;
    let mut report =
        analyze_lp_objective_sensitivity_internal(&lp, &solve_opts.lp_simplex, sensitivity_opts);
    if report.base_objective.is_finite() && program.objective_offset.abs() > 1e-12 {
        report.base_objective += program.objective_offset;
    }
    Ok(report)
}

/// Compute LP row-RHS sensitivity ranges for a pure continuous linear
/// `MathProgram`.
pub fn analyze_math_program_rhs_sensitivity(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    sensitivity_opts: &LPRhsSensitivityOptions,
) -> Result<LPRhsSensitivityReport, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty()
        || program.has_discrete_features()
        || program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
    {
        return Err(MathProgramError::Unsupported(
            "RHS sensitivity requires a pure continuous single-objective linear model".to_string(),
        ));
    }
    let lp = program.to_lp_problem()?;
    let mut report =
        analyze_lp_rhs_sensitivity_internal(&lp, &solve_opts.lp_simplex, sensitivity_opts);
    if report.base_objective.is_finite() && program.objective_offset.abs() > 1e-12 {
        report.base_objective += program.objective_offset;
    }
    Ok(report)
}

/// Compute LP variable-bound sensitivity ranges for a pure continuous linear
/// `MathProgram`.
pub fn analyze_math_program_bound_sensitivity(
    program: &MathProgram,
    solve_opts: &MathProgramSolveOptions,
    sensitivity_opts: &LPBoundSensitivityOptions,
) -> Result<LPBoundSensitivityReport, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty()
        || program.has_discrete_features()
        || program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
    {
        return Err(MathProgramError::Unsupported(
            "bound sensitivity requires a pure continuous single-objective linear model"
                .to_string(),
        ));
    }
    let lp = program.to_lp_problem()?;
    let mut report =
        analyze_lp_bound_sensitivity_internal(&lp, &solve_opts.lp_simplex, sensitivity_opts);
    if report.base_objective.is_finite() && program.objective_offset.abs() > 1e-12 {
        report.base_objective += program.objective_offset;
    }
    Ok(report)
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
        let mut solution = from_lp_solution(solve_lp_with_backend(&lp, opts));
        add_objective_offset_to_solution(&mut solution, program.objective_offset);
        return Ok(solution);
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
        solve_math_program_external_single_objective(stage_program, opts)
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
    let objective_offset = compiled_objective_offset(program, &compiled);
    let mip_opts = compiled_mip_options(program, &compiled, opts, true)?;
    let mip = solve_ipmip_with_des(compiled.problem.clone(), mip_opts);
    let x = compiled.original_x(&mip.x);
    let objective = objective_value(program, &x);
    let best_bound = original_mip_best_bound(mip.best_bound, objective_offset);
    let mip_gap = original_mip_gap(best_bound, objective);
    let first_branch_variable = first_mip_branch_variable_name(program, &compiled, &mip.trace);
    let incumbent_source = mip
        .incumbent_source
        .as_deref()
        .map(|source| format!(", incumbent_source={source}"))
        .unwrap_or_default();
    Ok(MathProgramSolution {
        status: from_ipmip_status(mip.status),
        x,
        objective,
        best_bound,
        mip_gap,
        nodes_explored: Some(mip.nodes_explored),
        first_branch_variable,
        solver_version: None,
        iterations: None,
        control_feedback: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
        solver: "des-ipmip".to_string(),
        message: Some(format!(
            "nodes={}, gap={:.3e}, lp_solves={}{}",
            mip.nodes_explored, mip.gap, mip.lp_solves, incumbent_source
        )),
    })
}

fn first_mip_branch_variable_name(
    program: &MathProgram,
    compiled: &CompiledMip,
    trace: &[IPMIPTraceEvent],
) -> Option<String> {
    let branch_var = trace
        .iter()
        .find(|event| event.action == TraceAction::Branch)
        .and_then(|event| event.branch_var)?;
    original_variable_name_for_compiled_var(program, compiled, branch_var).or_else(|| {
        compiled
            .problem
            .var_names
            .as_ref()
            .and_then(|names| names.get(branch_var).cloned())
    })
}

fn original_variable_name_for_compiled_var(
    program: &MathProgram,
    compiled: &CompiledMip,
    compiled_var: usize,
) -> Option<String> {
    for (idx, expansion) in compiled.expansions.iter().enumerate() {
        if expansion
            .terms
            .iter()
            .any(|&(term_var, _)| term_var == compiled_var)
        {
            return Some(program.variables[idx].name.clone());
        }
    }

    let canonical_name = compiled.problem.var_names.as_ref()?.get(compiled_var)?;
    for suffix in ["__active", "__pos", "__neg"] {
        if let Some(source_name) = canonical_name.strip_suffix(suffix) {
            if program
                .variables
                .iter()
                .any(|var| var.name.as_str() == source_name)
            {
                return Some(source_name.to_string());
            }
        }
    }
    program
        .variables
        .iter()
        .find(|var| var.name == *canonical_name)
        .map(|var| var.name.clone())
}

fn solve_mixed_integer_quadratic_objective(
    program: &MathProgram,
    opts: &MathProgramSolveOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    let original_vars = program.variables.len();
    let transformed = quadratic_objective_epigraph_program(program)?;
    let mut transformed_opts = opts.clone();
    if let Some(priorities) = &opts.branch_priorities {
        transformed_opts.branch_priorities = Some(extend_branch_priorities_for_added_variables(
            original_vars,
            transformed.variables.len(),
            priorities,
        )?);
    }
    let mut solution = solve_mixed_integer_conic(&transformed, &transformed_opts)?;
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
    let objective_offset = compiled_objective_offset(&relaxation, &compiled);
    let mip_opts = compiled_mip_options(&relaxation, &compiled, opts, false)?;
    let mut best = None;
    for cut in 0..=opts.conic.max_cuts {
        let mip = solve_ipmip_with_des(compiled.problem.clone(), mip_opts.clone());
        let x = compiled.original_x(&mip.x);
        let objective = objective_value(program, &x);
        let best_bound = original_mip_best_bound(mip.best_bound, objective_offset);
        let mip_gap = original_mip_gap(best_bound, objective);
        let mut solution = MathProgramSolution {
            status: from_ipmip_status(mip.status),
            x,
            objective,
            best_bound,
            mip_gap,
            nodes_explored: Some(mip.nodes_explored),
            first_branch_variable: first_mip_branch_variable_name(
                &relaxation,
                &compiled,
                &mip.trace,
            ),
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            unbounded_ray: None,
            infeasibility_certificate: None,
            solver: solver_name.to_string(),
            message: Some(format!(
                "cuts={}, nodes={}, gap={:.3e}, lp_solves={}",
                cut, mip.nodes_explored, mip.gap, mip.lp_solves
            )),
        };
        if solution.x.len() == program.variables.len() {
            solution.objective = objective_value(program, &solution.x);
        }
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
        best_bound: None,
        mip_gap: None,
        nodes_explored: None,
        first_branch_variable: None,
        solver_version: None,
        iterations: None,
        control_feedback: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
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

/// Solve the same model internally and with an optional external oracle.
///
/// Explicit local CLI methods route through the Rust `external_linear_cli`
/// adapter first. If the requested external stack is unavailable,
/// `external.status` is `NumericalError` and `within_tolerance` is false. The
/// internal solve is still returned so callers can keep validation runs
/// non-fatal on machines without every external solver installed.
pub fn cross_check_math_program_with_external(
    program: &MathProgram,
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    tol: f64,
) -> Result<MathProgramCrossCheck, MathProgramError> {
    let internal = solve_math_program(program, internal_opts)?;
    let external = solve_math_program_external(program, external_opts)?;
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
    let external = solve_math_program_external(&internal.subsystem, external_opts)?;
    let status_agree = internal.status == external.status;
    let within_tolerance = status_agree && internal.status == MathProgramStatus::Infeasible;
    Ok(MathProgramConflictCrossCheck {
        internal,
        external,
        status_agree,
        within_tolerance,
    })
}

pub fn find_math_program_assumption_unsat_core(
    program: &MathProgram,
    assumptions: &[BoolLiteral],
    solve_opts: &MathProgramSolveOptions,
    core_opts: &MathProgramAssumptionCoreOptions,
) -> Result<MathProgramAssumptionCore, MathProgramError> {
    program.validate()?;
    validate_math_program_assumptions(program, assumptions)?;

    let mut active = vec![true; assumptions.len()];
    let initial_subsystem = build_assumption_subsystem(program, assumptions, &active)?;
    let initial = solve_math_program(&initial_subsystem, solve_opts)?;
    if initial.status != MathProgramStatus::Infeasible {
        return Ok(MathProgramAssumptionCore {
            status: initial.status,
            assumptions: Vec::new(),
            subsystem: initial_subsystem,
            minimal: false,
            solver: "des-assumption-core".to_string(),
            message: Some(
                "assumptions do not make the model infeasible under feasibility probing"
                    .to_string(),
            ),
        });
    }

    let mut subsystem = initial_subsystem;
    let mut checks = 0usize;
    let mut limited = false;
    for i in 0..assumptions.len() {
        if checks >= core_opts.max_candidate_checks {
            limited = true;
            break;
        }
        active[i] = false;
        let trial = build_assumption_subsystem(program, assumptions, &active)?;
        let trial_solution = solve_math_program(&trial, solve_opts)?;
        checks += 1;
        if trial_solution.status == MathProgramStatus::Infeasible {
            subsystem = trial;
        } else {
            active[i] = true;
        }
    }

    let core_assumptions = active_assumptions(assumptions, &active);
    Ok(MathProgramAssumptionCore {
        status: MathProgramStatus::Infeasible,
        assumptions: core_assumptions,
        subsystem,
        minimal: !limited,
        solver: "des-assumption-core".to_string(),
        message: Some(format!(
            "assumptions={}, core={}, checks={}, minimal={}",
            assumptions.len(),
            active.iter().filter(|&&is_active| is_active).count(),
            checks,
            !limited
        )),
    })
}

pub fn cross_check_math_program_assumption_core_with_external(
    program: &MathProgram,
    assumptions: &[BoolLiteral],
    internal_opts: &MathProgramSolveOptions,
    external_opts: &ExternalMathProgramOptions,
    core_opts: &MathProgramAssumptionCoreOptions,
) -> Result<MathProgramAssumptionCoreCrossCheck, MathProgramError> {
    let internal =
        find_math_program_assumption_unsat_core(program, assumptions, internal_opts, core_opts)?;
    let external = solve_math_program_external(&internal.subsystem, external_opts)?;
    let status_agree = internal.status == external.status;
    let within_tolerance = status_agree && internal.status == MathProgramStatus::Infeasible;
    Ok(MathProgramAssumptionCoreCrossCheck {
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
    if let Some(priorities) = &solve_opts.branch_priorities {
        relaxed_solve_opts.branch_priorities = Some(extend_branch_priorities_for_added_variables(
            program.variables.len(),
            relaxed_program.variables.len(),
            priorities,
        )?);
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
    let external = solve_math_program_external(&internal.relaxed_program, external_opts)?;
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
    let original_vars = program.variables.len();
    solve_math_program_solution_pool_with(program, pool_opts, "des-solution-pool", |candidate| {
        let mut candidate_opts = solve_opts.clone();
        if let Some(priorities) = &solve_opts.branch_priorities {
            candidate_opts.branch_priorities = Some(extend_branch_priorities_for_added_variables(
                original_vars,
                candidate.variables.len(),
                priorities,
            )?);
        }
        solve_math_program(candidate, &candidate_opts)
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
        |candidate| solve_math_program_external(candidate, external_opts),
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

/// A CPLEX-LP text export suitable for local solver CLIs.
///
/// Continuous linear models are exported in the original named-variable space.
/// Discrete models are first compiled through the same exact MIP lowering used
/// by the native solver, so `variable_names` contains the exported compiled
/// columns, including any generated auxiliaries.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramCplexLpExport {
    pub text: String,
    pub variable_names: Vec<String>,
    pub constraint_names: Vec<String>,
    pub original_variable_expansions: Vec<MathProgramExportVariableExpansion>,
    pub row_mappings: Vec<MathProgramExportRowMapping>,
    pub is_mip: bool,
    pub original_variable_count: usize,
}

impl MathProgramCplexLpExport {
    pub fn original_x(&self, exported_x: &[f64]) -> Result<Vec<f64>, MathProgramError> {
        map_math_program_export_solution(&self.original_variable_expansions, exported_x)
    }
}

/// An MPS text export suitable for local solver CLIs.
///
/// Continuous linear models are exported in the original named-variable space.
/// Discrete models are first compiled through the same exact MIP lowering used
/// by the native solver, so `variable_names` contains the exported compiled
/// columns, including any generated auxiliaries.
#[derive(Clone, Debug, PartialEq)]
pub struct MathProgramMpsExport {
    pub text: String,
    pub variable_names: Vec<String>,
    pub constraint_names: Vec<String>,
    pub original_variable_expansions: Vec<MathProgramExportVariableExpansion>,
    pub row_mappings: Vec<MathProgramExportRowMapping>,
    pub is_mip: bool,
    pub original_variable_count: usize,
}

impl MathProgramMpsExport {
    pub fn original_x(&self, exported_x: &[f64]) -> Result<Vec<f64>, MathProgramError> {
        map_math_program_export_solution(&self.original_variable_expansions, exported_x)
    }
}

/// Export a linear, continuous-QP, or exactly lowered MIP facade model as CPLEX
/// LP text.
///
/// Continuous quadratic objectives use the CPLEX LP quadratic objective syntax.
/// Quadratic constraints, conic constraints, and hierarchical multi-objective
/// models should use the native solve APIs or MPS nonlinear sections instead.
pub fn export_math_program_cplex_lp(
    program: &MathProgram,
) -> Result<MathProgramCplexLpExport, MathProgramError> {
    let parts = math_program_linear_text_export_parts(program, "CPLEX LP export", true, false)?;
    let text = render_cplex_lp(
        parts.sense,
        &parts.objective,
        &parts.quadratic_objective,
        &parts.variable_names,
        &parts.rows,
        &parts.constraint_names,
        parts.lower.as_deref(),
        parts.upper.as_deref(),
        parts.integer_vars.as_deref(),
    )?;
    Ok(MathProgramCplexLpExport {
        text,
        variable_names: parts.variable_names,
        constraint_names: parts.constraint_names,
        original_variable_expansions: parts.original_variable_expansions,
        row_mappings: parts.row_mappings,
        is_mip: parts.is_mip,
        original_variable_count: parts.original_variable_count,
    })
}

/// Export a linear, continuous-QP, or exactly lowered MIP facade model as free
/// MPS text.
///
/// Continuous quadratic objectives and constraints use the extended `QUADOBJ`
/// and `QCMATRIX` sections. Conic constraints and hierarchical multi-objective
/// models should use the native solve APIs or solver-specific nonlinear file
/// formats instead.
pub fn export_math_program_mps(
    program: &MathProgram,
) -> Result<MathProgramMpsExport, MathProgramError> {
    let parts = math_program_linear_text_export_parts(program, "MPS export", true, true)?;
    let text = render_mps(
        parts.sense,
        &parts.objective,
        &parts.quadratic_objective,
        &parts.quadratic_constraints,
        &parts.variable_names,
        &parts.rows,
        &parts.constraint_names,
        parts.lower.as_deref(),
        parts.upper.as_deref(),
        parts.integer_vars.as_deref(),
    )?;
    Ok(MathProgramMpsExport {
        text,
        variable_names: parts.variable_names,
        constraint_names: parts.constraint_names,
        original_variable_expansions: parts.original_variable_expansions,
        row_mappings: parts.row_mappings,
        is_mip: parts.is_mip,
        original_variable_count: parts.original_variable_count,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct LinearTextExportParts {
    sense: LpSense,
    objective: Vec<f64>,
    quadratic_objective: Vec<QuadraticObjectiveTerm>,
    quadratic_constraints: Vec<MpsQuadraticConstraintExport>,
    variable_names: Vec<String>,
    rows: Vec<CplexLpExportRow>,
    constraint_names: Vec<String>,
    lower: Option<Vec<Option<f64>>>,
    upper: Option<Vec<Option<f64>>>,
    integer_vars: Option<Vec<bool>>,
    original_variable_expansions: Vec<MathProgramExportVariableExpansion>,
    row_mappings: Vec<MathProgramExportRowMapping>,
    is_mip: bool,
    original_variable_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct MpsQuadraticConstraintExport {
    row_name: String,
    terms: Vec<QuadraticConstraintTerm>,
}

fn math_program_linear_text_export_parts(
    program: &MathProgram,
    format_name: &str,
    allow_quadratic_objective: bool,
    allow_quadratic_constraints: bool,
) -> Result<LinearTextExportParts, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty() {
        return Err(MathProgramError::Unsupported(format!(
            "{format_name} does not support hierarchical multi-objective models"
        )));
    }
    if program.has_conic_constraints() {
        return Err(MathProgramError::Unsupported(format!(
            "{format_name} supports linear and exactly lowered MIP models only"
        )));
    }
    if program.has_quadratic_constraints() && !allow_quadratic_constraints {
        return Err(MathProgramError::Unsupported(format!(
            "{format_name} does not support quadratic constraints"
        )));
    }
    if program.has_quadratic_constraints() && program.has_discrete_features() {
        return Err(MathProgramError::Unsupported(format!(
            "{format_name} supports quadratic constraints only for pure continuous models"
        )));
    }

    if program.has_discrete_features() {
        let compiled = compile_mip(program)?;
        let variable_names = sanitize_cplex_lp_names(
            compiled.problem.var_names.as_deref(),
            compiled.problem.c.len(),
            "x",
        );
        let rows = compiled_mip_export_rows(&compiled.problem, variable_names.len())?;
        let raw_constraint_names = rows.iter().map(|row| row.name.clone()).collect::<Vec<_>>();
        let constraint_names =
            sanitize_cplex_lp_names(Some(&raw_constraint_names), rows.len(), "c");
        let compiled_upper = compiled.problem.ub.as_ref().map(|upper| {
            upper
                .iter()
                .copied()
                .map(|value| {
                    if value.is_infinite() && value.is_sign_positive() {
                        None
                    } else {
                        Some(value)
                    }
                })
                .collect::<Vec<_>>()
        });
        let original_variable_expansions =
            export_variable_expansions(program, &compiled.expansions, &variable_names)?;
        let row_mappings = export_row_mappings(&rows, &constraint_names)?;
        let objective_offset = compiled_objective_offset(program, &compiled);
        let mut parts = LinearTextExportParts {
            sense: compiled.problem.sense,
            objective: compiled.problem.c,
            quadratic_objective: Vec::new(),
            quadratic_constraints: Vec::new(),
            variable_names,
            rows,
            constraint_names,
            lower: None,
            upper: compiled_upper,
            integer_vars: Some(compiled.problem.integer_vars),
            original_variable_expansions,
            row_mappings,
            is_mip: true,
            original_variable_count: program.variables.len(),
        };
        add_objective_offset_export_column(&mut parts, objective_offset)?;
        return Ok(parts);
    }

    let quadratic_objective = if program.has_quadratic_objective() {
        if allow_quadratic_objective {
            program.quadratic_objective.clone()
        } else {
            return Err(MathProgramError::Unsupported(format!(
                "{format_name} does not support continuous quadratic objectives"
            )));
        }
    } else {
        Vec::new()
    };

    if program.has_quadratic_objective() && program.has_discrete_features() {
        return Err(MathProgramError::Unsupported(format!(
            "{format_name} supports quadratic objectives only for pure continuous models"
        )));
    }

    let raw_variable_names = program
        .variables
        .iter()
        .map(|var| var.name.clone())
        .collect::<Vec<_>>();
    let variable_names =
        sanitize_cplex_lp_names(Some(&raw_variable_names), program.variables.len(), "x");
    let rows = original_linear_export_rows(program, allow_quadratic_constraints)?;
    let raw_constraint_names = rows.iter().map(|row| row.name.clone()).collect::<Vec<_>>();
    let constraint_names = sanitize_cplex_lp_names(Some(&raw_constraint_names), rows.len(), "c");
    let row_mappings = export_row_mappings(&rows, &constraint_names)?;
    let quadratic_constraints =
        mps_quadratic_constraint_exports(program, &rows, &constraint_names)?;
    let lower = program
        .variables
        .iter()
        .map(|var| var.lb)
        .collect::<Vec<_>>();
    let upper = program
        .variables
        .iter()
        .map(|var| var.ub)
        .collect::<Vec<_>>();
    let objective = program
        .variables
        .iter()
        .map(|var| var.obj)
        .collect::<Vec<_>>();
    let original_variable_expansions =
        identity_export_variable_expansions(program, &variable_names);
    let mut parts = LinearTextExportParts {
        sense: program.sense.to_lp(),
        objective,
        quadratic_objective,
        quadratic_constraints,
        variable_names,
        rows,
        constraint_names,
        lower: Some(lower),
        upper: Some(upper),
        integer_vars: None,
        original_variable_expansions,
        row_mappings,
        is_mip: false,
        original_variable_count: program.variables.len(),
    };
    add_objective_offset_export_column(&mut parts, program.objective_offset)?;
    Ok(parts)
}

pub fn map_math_program_export_solution(
    expansions: &[MathProgramExportVariableExpansion],
    exported_x: &[f64],
) -> Result<Vec<f64>, MathProgramError> {
    let mut original = Vec::with_capacity(expansions.len());
    for expansion in expansions {
        let mut value = expansion.constant;
        for term in &expansion.terms {
            let exported_value = exported_x.get(term.exported_var).ok_or_else(|| {
                MathProgramError::BadIndex(format!(
                    "exported solution has {} values but `{}` references exported variable {}",
                    exported_x.len(),
                    expansion.original_name,
                    term.exported_var
                ))
            })?;
            value += term.coefficient * exported_value;
        }
        original.push(value);
    }
    Ok(original)
}

fn identity_export_variable_expansions(
    program: &MathProgram,
    variable_names: &[String],
) -> Vec<MathProgramExportVariableExpansion> {
    program
        .variables
        .iter()
        .enumerate()
        .map(|(idx, var)| MathProgramExportVariableExpansion {
            original_var: idx,
            original_name: var.name.clone(),
            constant: 0.0,
            terms: vec![MathProgramExportTerm {
                exported_var: idx,
                exported_name: variable_names[idx].clone(),
                coefficient: 1.0,
            }],
        })
        .collect()
}

fn export_variable_expansions(
    program: &MathProgram,
    expansions: &[LinearExpansion],
    variable_names: &[String],
) -> Result<Vec<MathProgramExportVariableExpansion>, MathProgramError> {
    if expansions.len() != program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "compiled expansion count {} does not match {} original variables",
            expansions.len(),
            program.variables.len()
        )));
    }
    expansions
        .iter()
        .enumerate()
        .map(|(original_var, expansion)| {
            let terms = expansion
                .terms
                .iter()
                .map(|&(exported_var, coefficient)| {
                    let exported_name = variable_names.get(exported_var).ok_or_else(|| {
                        MathProgramError::BadIndex(format!(
                            "compiled expansion for `{}` references missing exported variable {}",
                            program.variables[original_var].name, exported_var
                        ))
                    })?;
                    Ok(MathProgramExportTerm {
                        exported_var,
                        exported_name: exported_name.clone(),
                        coefficient,
                    })
                })
                .collect::<Result<Vec<_>, MathProgramError>>()?;
            Ok(MathProgramExportVariableExpansion {
                original_var,
                original_name: program.variables[original_var].name.clone(),
                constant: expansion.constant,
                terms,
            })
        })
        .collect()
}

fn export_row_mappings(
    rows: &[CplexLpExportRow],
    constraint_names: &[String],
) -> Result<Vec<MathProgramExportRowMapping>, MathProgramError> {
    if rows.len() != constraint_names.len() {
        return Err(MathProgramError::BadIndex(format!(
            "row mapping length {} does not match {} exported constraint names",
            rows.len(),
            constraint_names.len()
        )));
    }
    Ok(rows
        .iter()
        .zip(constraint_names)
        .enumerate()
        .map(
            |(exported_row, (row, exported_name))| MathProgramExportRowMapping {
                exported_row,
                exported_name: exported_name.clone(),
                source_kind: row.source_kind,
                source_index: row.source_index,
                source_name: row.source_name.clone(),
                sense: row.sense,
                rhs: row.rhs,
                multiplier: row.multiplier,
            },
        )
        .collect())
}

#[derive(Clone, Debug, PartialEq)]
struct CplexLpExportRow {
    name: String,
    coeffs: Vec<f64>,
    sense: RowSense,
    rhs: f64,
    source_kind: MathProgramExportRowKind,
    source_index: Option<usize>,
    source_name: String,
    multiplier: f64,
}

fn original_linear_export_rows(
    program: &MathProgram,
    include_quadratic_constraints: bool,
) -> Result<Vec<CplexLpExportRow>, MathProgramError> {
    let n = program.variables.len();
    let quadratic_count = if include_quadratic_constraints {
        program.quadratic_constraints.len()
    } else {
        0
    };
    let mut rows = Vec::with_capacity(
        program.constraints.len() + quadratic_count + program.lazy_constraints.len(),
    );
    for (idx, row) in program.constraints.iter().enumerate() {
        rows.push(CplexLpExportRow {
            name: row.name.clone(),
            coeffs: dense_row(n, &row.coeffs),
            sense: row.sense,
            rhs: row.rhs,
            source_kind: MathProgramExportRowKind::LinearConstraint,
            source_index: Some(idx),
            source_name: row.name.clone(),
            multiplier: 1.0,
        });
    }
    if include_quadratic_constraints {
        for (idx, row) in program.quadratic_constraints.iter().enumerate() {
            rows.push(CplexLpExportRow {
                name: row.name.clone(),
                coeffs: dense_row(n, &row.linear_terms),
                sense: row.sense,
                rhs: row.rhs,
                source_kind: MathProgramExportRowKind::QuadraticConstraint,
                source_index: Some(idx),
                source_name: row.name.clone(),
                multiplier: 1.0,
            });
        }
    }
    for (idx, row) in program.lazy_constraints.iter().enumerate() {
        rows.push(CplexLpExportRow {
            name: row.name.clone(),
            coeffs: dense_row(n, &row.coeffs),
            sense: row.sense,
            rhs: row.rhs,
            source_kind: MathProgramExportRowKind::LazyConstraint,
            source_index: Some(idx),
            source_name: row.name.clone(),
            multiplier: 1.0,
        });
    }
    Ok(rows)
}

fn compiled_mip_export_rows(
    problem: &IPMIPProblem,
    variable_count: usize,
) -> Result<Vec<CplexLpExportRow>, MathProgramError> {
    let mut rows = Vec::with_capacity(
        problem.a.len()
            + problem
                .lazy_constraints
                .as_ref()
                .map_or(0, |lazy_rows| lazy_rows.len()),
    );
    for (idx, (coeffs, rhs)) in problem.a.iter().zip(&problem.b).enumerate() {
        if coeffs.len() != variable_count {
            return Err(MathProgramError::BadIndex(format!(
                "compiled row {idx} has {} coefficients for {variable_count} variables",
                coeffs.len()
            )));
        }
        let name = problem
            .con_names
            .as_ref()
            .and_then(|names| names.get(idx))
            .cloned()
            .unwrap_or_else(|| format!("c{idx}"));
        rows.push(CplexLpExportRow {
            name: name.clone(),
            coeffs: coeffs.clone(),
            sense: RowSense::Le,
            rhs: *rhs,
            source_kind: MathProgramExportRowKind::Generated,
            source_index: Some(idx),
            source_name: name,
            multiplier: 1.0,
        });
    }
    if let Some(lazy_rows) = &problem.lazy_constraints {
        for (idx, lazy) in lazy_rows.iter().enumerate() {
            if lazy.coefs.len() != variable_count {
                return Err(MathProgramError::BadIndex(format!(
                    "compiled lazy row {idx} has {} coefficients for {variable_count} variables",
                    lazy.coefs.len()
                )));
            }
            rows.push(CplexLpExportRow {
                name: lazy.name.clone(),
                coeffs: lazy.coefs.clone(),
                sense: RowSense::Le,
                rhs: lazy.rhs,
                source_kind: MathProgramExportRowKind::LazyConstraint,
                source_index: Some(idx),
                source_name: lazy.name.clone(),
                multiplier: 1.0,
            });
        }
    }
    Ok(rows)
}

fn render_cplex_lp(
    sense: LpSense,
    objective: &[f64],
    quadratic_objective: &[QuadraticObjectiveTerm],
    variable_names: &[String],
    rows: &[CplexLpExportRow],
    constraint_names: &[String],
    lower: Option<&[Option<f64>]>,
    upper: Option<&[Option<f64>]>,
    integer_vars: Option<&[bool]>,
) -> Result<String, MathProgramError> {
    if objective.len() != variable_names.len() {
        return Err(MathProgramError::BadIndex(format!(
            "objective length {} does not match {} export variable names",
            objective.len(),
            variable_names.len()
        )));
    }
    if constraint_names.len() != rows.len() {
        return Err(MathProgramError::BadIndex(format!(
            "constraint name length {} does not match {} rows",
            constraint_names.len(),
            rows.len()
        )));
    }
    if let Some(lower) = lower {
        if lower.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "lower-bound length {} does not match {} export variable names",
                lower.len(),
                variable_names.len()
            )));
        }
    }
    if let Some(upper) = upper {
        if upper.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "upper-bound length {} does not match {} export variable names",
                upper.len(),
                variable_names.len()
            )));
        }
    }
    if let Some(integer_vars) = integer_vars {
        if integer_vars.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "integer marker length {} does not match {} export variable names",
                integer_vars.len(),
                variable_names.len()
            )));
        }
    }

    let mut out = String::new();
    out.push_str(match sense {
        LpSense::Max => "Maximize\n",
        LpSense::Min => "Minimize\n",
    });
    out.push_str(" obj: ");
    out.push_str(&cplex_lp_objective_expr(
        objective,
        quadratic_objective,
        variable_names,
    )?);
    out.push('\n');
    out.push_str("Subject To\n");
    if rows.is_empty() {
        out.push_str(" c0: ");
        out.push_str(&cplex_lp_zero_expr(variable_names));
        out.push_str(" <= 0\n");
    } else {
        for (row, name) in rows.iter().zip(constraint_names) {
            if row.coeffs.len() != variable_names.len() {
                return Err(MathProgramError::BadIndex(format!(
                    "row `{}` has {} coefficients for {} variables",
                    row.name,
                    row.coeffs.len(),
                    variable_names.len()
                )));
            }
            out.push(' ');
            out.push_str(name);
            out.push_str(": ");
            out.push_str(&cplex_lp_expr(&row.coeffs, variable_names)?);
            out.push(' ');
            out.push_str(row.sense.as_str());
            out.push(' ');
            out.push_str(&cplex_lp_number(row.rhs)?);
            out.push('\n');
        }
    }
    out.push_str("Bounds\n");
    for (idx, name) in variable_names.iter().enumerate() {
        let lo = lower.map_or(Some(0.0), |values| values[idx]);
        let hi = upper.and_then(|values| values[idx]);
        out.push_str(" ");
        match (lo, hi) {
            (None, None) => {
                out.push_str(name);
                out.push_str(" free");
            }
            (None, Some(hi)) => {
                out.push_str(name);
                out.push_str(" <= ");
                out.push_str(&cplex_lp_number(hi)?);
            }
            (Some(lo), None) => {
                out.push_str(&cplex_lp_number(lo)?);
                out.push_str(" <= ");
                out.push_str(name);
            }
            (Some(lo), Some(hi)) if (lo - hi).abs() <= 1e-12 => {
                out.push_str(name);
                out.push_str(" = ");
                out.push_str(&cplex_lp_number(lo)?);
            }
            (Some(lo), Some(hi)) => {
                out.push_str(&cplex_lp_number(lo)?);
                out.push_str(" <= ");
                out.push_str(name);
                out.push_str(" <= ");
                out.push_str(&cplex_lp_number(hi)?);
            }
        }
        out.push('\n');
    }
    if let Some(integer_vars) = integer_vars {
        let binary_names = integer_vars
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_integer)| {
                let hi = upper.and_then(|values| values[idx]);
                (is_integer && hi.is_some_and(|value| (value - 1.0).abs() <= 1e-12))
                    .then_some(variable_names[idx].as_str())
            })
            .collect::<Vec<_>>();
        let general_names = integer_vars
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_integer)| {
                let hi = upper.and_then(|values| values[idx]);
                (is_integer && !hi.is_some_and(|value| (value - 1.0).abs() <= 1e-12))
                    .then_some(variable_names[idx].as_str())
            })
            .collect::<Vec<_>>();
        if !binary_names.is_empty() {
            out.push_str("Binaries\n");
            for name in binary_names {
                out.push(' ');
                out.push_str(name);
                out.push('\n');
            }
        }
        if !general_names.is_empty() {
            out.push_str("Generals\n");
            for name in general_names {
                out.push(' ');
                out.push_str(name);
                out.push('\n');
            }
        }
    }
    out.push_str("End\n");
    Ok(out)
}

fn render_mps(
    sense: LpSense,
    objective: &[f64],
    quadratic_objective: &[QuadraticObjectiveTerm],
    quadratic_constraints: &[MpsQuadraticConstraintExport],
    variable_names: &[String],
    rows: &[CplexLpExportRow],
    constraint_names: &[String],
    lower: Option<&[Option<f64>]>,
    upper: Option<&[Option<f64>]>,
    integer_vars: Option<&[bool]>,
) -> Result<String, MathProgramError> {
    validate_linear_text_export_inputs(
        "MPS",
        objective,
        variable_names,
        rows,
        constraint_names,
        lower,
        upper,
        integer_vars,
    )?;

    let effective_rows = if rows.is_empty() {
        vec![CplexLpExportRow {
            name: "c0".to_string(),
            coeffs: vec![0.0; variable_names.len()],
            sense: RowSense::Le,
            rhs: 0.0,
            source_kind: MathProgramExportRowKind::Generated,
            source_index: None,
            source_name: "c0".to_string(),
            multiplier: 1.0,
        }]
    } else {
        rows.to_vec()
    };
    let effective_constraint_names = if rows.is_empty() {
        vec!["c0".to_string()]
    } else {
        constraint_names.to_vec()
    };

    let mut out = String::new();
    out.push_str("NAME          DES_MODEL\n");
    out.push_str("OBJSENSE\n");
    out.push_str(match sense {
        LpSense::Max => " MAX\n",
        LpSense::Min => " MIN\n",
    });
    out.push_str("ROWS\n");
    out.push_str(" N  OBJ\n");
    for (row, name) in effective_rows.iter().zip(&effective_constraint_names) {
        out.push(' ');
        out.push(mps_row_type(row.sense));
        out.push_str("  ");
        out.push_str(name);
        out.push('\n');
    }

    out.push_str("COLUMNS\n");
    let mut inside_integer = false;
    for (idx, name) in variable_names.iter().enumerate() {
        let is_integer = integer_vars.is_some_and(|values| values[idx]);
        if is_integer && !inside_integer {
            out.push_str("    MARK0000  'MARKER'                 'INTORG'\n");
            inside_integer = true;
        } else if !is_integer && inside_integer {
            out.push_str("    MARK0001  'MARKER'                 'INTEND'\n");
            inside_integer = false;
        }

        let mut entries = Vec::new();
        let obj = objective[idx];
        if !obj.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "MPS export objective coefficient for variable {idx}"
            )));
        }
        if obj.abs() > 1e-12 {
            entries.push(("OBJ".to_string(), obj));
        }
        for (row, row_name) in effective_rows.iter().zip(&effective_constraint_names) {
            let coef = row.coeffs[idx];
            if !coef.is_finite() {
                return Err(MathProgramError::NonFinite(format!(
                    "MPS export coefficient for variable {idx}"
                )));
            }
            if coef.abs() > 1e-12 {
                entries.push((row_name.clone(), coef));
            }
        }
        for chunk in entries.chunks(2) {
            out.push_str("    ");
            out.push_str(name);
            for (row_name, value) in chunk {
                out.push_str("  ");
                out.push_str(row_name);
                out.push_str("  ");
                out.push_str(&mps_number(*value)?);
            }
            out.push('\n');
        }
    }
    if inside_integer {
        out.push_str("    MARK0001  'MARKER'                 'INTEND'\n");
    }

    out.push_str("RHS\n");
    for chunk in effective_rows
        .iter()
        .zip(&effective_constraint_names)
        .collect::<Vec<_>>()
        .chunks(2)
    {
        out.push_str("    RHS1");
        for (row, name) in chunk {
            out.push_str("  ");
            out.push_str(name);
            out.push_str("  ");
            out.push_str(&mps_number(row.rhs)?);
        }
        out.push('\n');
    }

    out.push_str("BOUNDS\n");
    for (idx, name) in variable_names.iter().enumerate() {
        let lo = lower.map_or(Some(0.0), |values| values[idx]);
        let hi = upper.and_then(|values| values[idx]);
        append_mps_bound_rows(&mut out, name, lo, hi)?;
    }
    append_mps_quadratic_objective(&mut out, quadratic_objective, variable_names)?;
    append_mps_quadratic_constraints(&mut out, quadratic_constraints, variable_names)?;
    out.push_str("ENDATA\n");
    Ok(out)
}

fn validate_linear_text_export_inputs(
    format_name: &str,
    objective: &[f64],
    variable_names: &[String],
    rows: &[CplexLpExportRow],
    constraint_names: &[String],
    lower: Option<&[Option<f64>]>,
    upper: Option<&[Option<f64>]>,
    integer_vars: Option<&[bool]>,
) -> Result<(), MathProgramError> {
    if objective.len() != variable_names.len() {
        return Err(MathProgramError::BadIndex(format!(
            "{format_name} objective length {} does not match {} export variable names",
            objective.len(),
            variable_names.len()
        )));
    }
    if constraint_names.len() != rows.len() {
        return Err(MathProgramError::BadIndex(format!(
            "{format_name} constraint name length {} does not match {} rows",
            constraint_names.len(),
            rows.len()
        )));
    }
    for row in rows {
        if row.coeffs.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{format_name} row `{}` has {} coefficients for {} variables",
                row.name,
                row.coeffs.len(),
                variable_names.len()
            )));
        }
    }
    if let Some(lower) = lower {
        if lower.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{format_name} lower-bound length {} does not match {} export variable names",
                lower.len(),
                variable_names.len()
            )));
        }
    }
    if let Some(upper) = upper {
        if upper.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{format_name} upper-bound length {} does not match {} export variable names",
                upper.len(),
                variable_names.len()
            )));
        }
    }
    if let Some(integer_vars) = integer_vars {
        if integer_vars.len() != variable_names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{format_name} integer marker length {} does not match {} export variable names",
                integer_vars.len(),
                variable_names.len()
            )));
        }
    }
    Ok(())
}

fn mps_row_type(sense: RowSense) -> char {
    match sense {
        RowSense::Le => 'L',
        RowSense::Ge => 'G',
        RowSense::Eq => 'E',
    }
}

fn append_mps_bound_rows(
    out: &mut String,
    name: &str,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), MathProgramError> {
    match (lower, upper) {
        (None, None) => append_mps_bound_row(out, "FR", name, None)?,
        (None, Some(hi)) => {
            append_mps_bound_row(out, "MI", name, None)?;
            append_mps_bound_row(out, "UP", name, Some(hi))?;
        }
        (Some(lo), None) => {
            if lo.abs() > 1e-12 {
                append_mps_bound_row(out, "LO", name, Some(lo))?;
            }
        }
        (Some(lo), Some(hi)) if (lo - hi).abs() <= 1e-12 => {
            append_mps_bound_row(out, "FX", name, Some(lo))?;
        }
        (Some(lo), Some(hi)) => {
            if lo.abs() > 1e-12 {
                append_mps_bound_row(out, "LO", name, Some(lo))?;
            }
            append_mps_bound_row(out, "UP", name, Some(hi))?;
        }
    }
    Ok(())
}

fn add_objective_offset_export_column(
    parts: &mut LinearTextExportParts,
    offset: f64,
) -> Result<(), MathProgramError> {
    if offset.abs() <= 1e-12 {
        return Ok(());
    }
    validate_objective_offset(offset)?;
    let old_len = parts.variable_names.len();
    let mut used = parts
        .variable_names
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let name = unique_cplex_lp_name(
        sanitize_cplex_lp_name("__objective_offset", "x", old_len),
        &mut used,
    );
    parts.variable_names.push(name);
    parts.objective.push(offset);
    for row in &mut parts.rows {
        row.coeffs.push(0.0);
    }
    match parts.lower.as_mut() {
        Some(lower) => lower.push(Some(1.0)),
        None => {
            let mut lower = vec![Some(0.0); old_len];
            lower.push(Some(1.0));
            parts.lower = Some(lower);
        }
    }
    match parts.upper.as_mut() {
        Some(upper) => upper.push(Some(1.0)),
        None => {
            let mut upper = vec![None; old_len];
            upper.push(Some(1.0));
            parts.upper = Some(upper);
        }
    }
    if let Some(integer_vars) = parts.integer_vars.as_mut() {
        integer_vars.push(false);
    }
    Ok(())
}

fn append_mps_bound_row(
    out: &mut String,
    kind: &str,
    name: &str,
    value: Option<f64>,
) -> Result<(), MathProgramError> {
    out.push(' ');
    out.push_str(kind);
    out.push_str(" BND1  ");
    out.push_str(name);
    if let Some(value) = value {
        out.push_str("  ");
        out.push_str(&mps_number(value)?);
    }
    out.push('\n');
    Ok(())
}

fn append_mps_quadratic_objective(
    out: &mut String,
    terms: &[QuadraticObjectiveTerm],
    variable_names: &[String],
) -> Result<(), MathProgramError> {
    let entries = mps_quadratic_objective_entries(terms, variable_names.len())?;
    if entries.is_empty() {
        return Ok(());
    }
    out.push_str("QUADOBJ\n");
    for ((left, right), coefficient) in entries {
        out.push_str("    ");
        out.push_str(&variable_names[left]);
        out.push_str("  ");
        out.push_str(&variable_names[right]);
        out.push_str("  ");
        out.push_str(&mps_number(coefficient)?);
        out.push('\n');
    }
    Ok(())
}

fn mps_quadratic_objective_entries(
    terms: &[QuadraticObjectiveTerm],
    variable_count: usize,
) -> Result<BTreeMap<(usize, usize), f64>, MathProgramError> {
    let mut entries = BTreeMap::new();
    for (term_idx, term) in terms.iter().enumerate() {
        if term.var_i >= variable_count || term.var_j >= variable_count {
            return Err(MathProgramError::BadIndex(format!(
                "MPS export quadratic objective term {term_idx} references {} and {} with {variable_count} variables",
                term.var_i, term.var_j
            )));
        }
        if !term.coeff.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "MPS export quadratic objective coefficient for term {term_idx}"
            )));
        }
        let (left, right) = if term.var_i <= term.var_j {
            (term.var_i, term.var_j)
        } else {
            (term.var_j, term.var_i)
        };
        let coefficient = if left == right {
            2.0 * term.coeff
        } else {
            term.coeff
        };
        if !coefficient.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "MPS export quadratic objective coefficient for term {term_idx}"
            )));
        }
        if coefficient.abs() > 1e-12 {
            *entries.entry((left, right)).or_insert(0.0) += coefficient;
        }
    }
    entries.retain(|_, coefficient| coefficient.abs() > 1e-12);
    Ok(entries)
}

fn mps_quadratic_constraint_exports(
    program: &MathProgram,
    rows: &[CplexLpExportRow],
    constraint_names: &[String],
) -> Result<Vec<MpsQuadraticConstraintExport>, MathProgramError> {
    if rows.len() != constraint_names.len() {
        return Err(MathProgramError::BadIndex(format!(
            "MPS quadratic constraint export has {} rows but {} constraint names",
            rows.len(),
            constraint_names.len()
        )));
    }
    let mut exports = Vec::new();
    for (row, row_name) in rows.iter().zip(constraint_names) {
        if row.source_kind != MathProgramExportRowKind::QuadraticConstraint {
            continue;
        }
        let source_index = row.source_index.ok_or_else(|| {
            MathProgramError::BadIndex(format!(
                "quadratic export row `{}` is missing a source index",
                row.name
            ))
        })?;
        let quadratic = program
            .quadratic_constraints
            .get(source_index)
            .ok_or_else(|| {
                MathProgramError::BadIndex(format!(
                    "quadratic export row `{}` references missing source {source_index}",
                    row.name
                ))
            })?;
        exports.push(MpsQuadraticConstraintExport {
            row_name: row_name.clone(),
            terms: quadratic.quadratic_terms.clone(),
        });
    }
    Ok(exports)
}

fn append_mps_quadratic_constraints(
    out: &mut String,
    constraints: &[MpsQuadraticConstraintExport],
    variable_names: &[String],
) -> Result<(), MathProgramError> {
    for constraint in constraints {
        let entries = mps_quadratic_constraint_entries(&constraint.terms, variable_names.len())?;
        if entries.is_empty() {
            continue;
        }
        out.push_str("QCMATRIX   ");
        out.push_str(&constraint.row_name);
        out.push('\n');
        for ((left, right), coefficient) in entries {
            out.push_str("    ");
            out.push_str(&variable_names[left]);
            out.push_str("  ");
            out.push_str(&variable_names[right]);
            out.push_str("  ");
            out.push_str(&mps_number(coefficient)?);
            out.push('\n');
        }
    }
    Ok(())
}

fn mps_quadratic_constraint_entries(
    terms: &[QuadraticConstraintTerm],
    variable_count: usize,
) -> Result<BTreeMap<(usize, usize), f64>, MathProgramError> {
    let mut unordered = BTreeMap::new();
    for (term_idx, term) in terms.iter().enumerate() {
        if term.var_i >= variable_count || term.var_j >= variable_count {
            return Err(MathProgramError::BadIndex(format!(
                "MPS export quadratic constraint term {term_idx} references {} and {} with {variable_count} variables",
                term.var_i, term.var_j
            )));
        }
        if !term.coeff.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "MPS export quadratic constraint coefficient for term {term_idx}"
            )));
        }
        let (left, right) = if term.var_i <= term.var_j {
            (term.var_i, term.var_j)
        } else {
            (term.var_j, term.var_i)
        };
        if term.coeff.abs() > 1e-12 {
            *unordered.entry((left, right)).or_insert(0.0) += term.coeff;
        }
    }

    let mut entries = BTreeMap::new();
    for ((left, right), coefficient) in unordered {
        if coefficient.abs() <= 1e-12 {
            continue;
        }
        if left == right {
            entries.insert((left, right), coefficient);
        } else {
            let half = coefficient / 2.0;
            if !half.is_finite() {
                return Err(MathProgramError::NonFinite(
                    "MPS export quadratic constraint coefficient".to_string(),
                ));
            }
            entries.insert((left, right), half);
            entries.insert((right, left), half);
        }
    }
    Ok(entries)
}

fn mps_number(value: f64) -> Result<String, MathProgramError> {
    if !value.is_finite() {
        return Err(MathProgramError::NonFinite(format!(
            "MPS export cannot encode non-finite value {value}"
        )));
    }
    Ok(format!("{value:.17e}"))
}

fn cplex_lp_objective_expr(
    linear: &[f64],
    quadratic: &[QuadraticObjectiveTerm],
    names: &[String],
) -> Result<String, MathProgramError> {
    let linear_expr = cplex_lp_expr(linear, names)?;
    if quadratic.is_empty() {
        return Ok(linear_expr);
    }

    let quadratic_expr = cplex_lp_quadratic_expr(quadratic, names)?;
    if quadratic_expr.is_empty() {
        return Ok(linear_expr);
    }
    if linear.iter().any(|coef| coef.abs() > 1e-12) {
        Ok(format!("{linear_expr} + [ {quadratic_expr} ] / 2"))
    } else {
        Ok(format!("[ {quadratic_expr} ] / 2"))
    }
}

fn cplex_lp_quadratic_expr(
    terms: &[QuadraticObjectiveTerm],
    names: &[String],
) -> Result<String, MathProgramError> {
    let mut parts = Vec::new();
    for (term_idx, term) in terms.iter().enumerate() {
        if term.var_i >= names.len() || term.var_j >= names.len() {
            return Err(MathProgramError::BadIndex(format!(
                "LP export quadratic objective term {term_idx} references {} and {} with {} variables",
                term.var_i,
                term.var_j,
                names.len()
            )));
        }
        let doubled = 2.0 * term.coeff;
        if !doubled.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "LP export quadratic objective coefficient for term {term_idx}"
            )));
        }
        if doubled.abs() <= 1e-12 {
            continue;
        }

        let body =
            cplex_lp_quadratic_term_body(doubled.abs(), &names[term.var_i], &names[term.var_j])?;
        if parts.is_empty() {
            parts.push(if doubled < 0.0 {
                format!("- {body}")
            } else {
                body
            });
        } else if doubled < 0.0 {
            parts.push(format!("- {body}"));
        } else {
            parts.push(format!("+ {body}"));
        }
    }
    Ok(parts.join(" "))
}

fn cplex_lp_quadratic_term_body(
    coefficient: f64,
    left_name: &str,
    right_name: &str,
) -> Result<String, MathProgramError> {
    let product = format!("{left_name} * {right_name}");
    if (coefficient - 1.0).abs() <= 1e-12 {
        Ok(product)
    } else {
        Ok(format!("{} {product}", cplex_lp_number(coefficient)?))
    }
}

fn cplex_lp_expr(coeffs: &[f64], names: &[String]) -> Result<String, MathProgramError> {
    let mut parts = Vec::new();
    for (idx, &coef) in coeffs.iter().enumerate() {
        if !coef.is_finite() {
            return Err(MathProgramError::NonFinite(format!(
                "LP export coefficient for variable {idx}"
            )));
        }
        if coef.abs() <= 1e-12 {
            continue;
        }
        let body = if (coef.abs() - 1.0).abs() <= 1e-12 {
            names[idx].clone()
        } else {
            format!("{} {}", cplex_lp_number(coef.abs())?, names[idx])
        };
        if parts.is_empty() {
            parts.push(if coef < 0.0 {
                format!("- {body}")
            } else {
                body
            });
        } else if coef < 0.0 {
            parts.push(format!("- {body}"));
        } else {
            parts.push(format!("+ {body}"));
        }
    }
    if parts.is_empty() {
        Ok(cplex_lp_zero_expr(names))
    } else {
        Ok(parts.join(" "))
    }
}

fn cplex_lp_zero_expr(names: &[String]) -> String {
    names
        .first()
        .map(|name| format!("0 {name}"))
        .unwrap_or_else(|| "0".to_string())
}

fn cplex_lp_number(value: f64) -> Result<String, MathProgramError> {
    if !value.is_finite() {
        return Err(MathProgramError::NonFinite(format!(
            "LP export cannot encode non-finite value {value}"
        )));
    }
    Ok(format!("{value:.17e}"))
}

fn sanitize_cplex_lp_names(raw_names: Option<&[String]>, len: usize, prefix: &str) -> Vec<String> {
    let mut used = BTreeSet::new();
    (0..len)
        .map(|idx| {
            let raw = raw_names
                .and_then(|names| names.get(idx))
                .map(String::as_str)
                .unwrap_or("");
            let base = sanitize_cplex_lp_name(raw, prefix, idx);
            unique_cplex_lp_name(base, &mut used)
        })
        .collect()
}

fn sanitize_cplex_lp_name(raw: &str, prefix: &str, idx: usize) -> String {
    let mut name = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    while name.contains("__") {
        name = name.replace("__", "_");
    }
    name = name.trim_matches('_').to_string();
    if name.is_empty() {
        name = format!("{prefix}{idx}");
    }
    if name
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_digit() || ch == '.')
    {
        name = format!("{prefix}_{name}");
    }
    let lower = name.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "binary"
            | "binaries"
            | "bound"
            | "bounds"
            | "columns"
            | "end"
            | "endata"
            | "free"
            | "general"
            | "generals"
            | "max"
            | "maximize"
            | "min"
            | "minimize"
            | "name"
            | "obj"
            | "objsense"
            | "ranges"
            | "rhs"
            | "rows"
            | "subject"
            | "st"
            | "s.t."
    ) {
        name = format!("{prefix}_{name}");
    }
    name
}

fn unique_cplex_lp_name(base: String, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix loop must return")
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
        || !program.lazy_constraints.is_empty()
        || !program.general_constraints.is_empty()
    {
        return Err(MathProgramError::Unsupported(
            "conflict refinement currently supports linear rows and variable bounds only"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_math_program_assumptions(
    program: &MathProgram,
    assumptions: &[BoolLiteral],
) -> Result<(), MathProgramError> {
    if assumptions.is_empty() {
        return Err(MathProgramError::Unsupported(
            "assumption core refinement requires at least one Boolean assumption".to_string(),
        ));
    }
    program.validate_boolean_clause_args(assumptions)
}

fn build_assumption_subsystem(
    program: &MathProgram,
    assumptions: &[BoolLiteral],
    active: &[bool],
) -> Result<MathProgram, MathProgramError> {
    if active.len() != assumptions.len() {
        return Err(MathProgramError::BadIndex(format!(
            "assumption active mask length {} does not match {} assumptions",
            active.len(),
            assumptions.len()
        )));
    }

    let mut subsystem = program.clone();
    subsystem.sense = ObjectiveSense::Min;
    subsystem.objective_offset = 0.0;
    subsystem.quadratic_objective.clear();
    subsystem.secondary_objectives.clear();
    for var in &mut subsystem.variables {
        var.obj = 0.0;
    }

    for (idx, (&literal, &is_active)) in assumptions.iter().zip(active).enumerate() {
        if !is_active {
            continue;
        }
        let value = if literal.value { 1.0 } else { 0.0 };
        let suffix = if literal.value { "true" } else { "false" };
        subsystem.add_constraint(
            format!(
                "__assumption_{idx}_{}_is_{suffix}",
                program.variables[literal.var].name
            ),
            vec![(literal.var, 1.0)],
            RowSense::Eq,
            value,
        )?;
    }
    Ok(subsystem)
}

fn active_assumptions(assumptions: &[BoolLiteral], active: &[bool]) -> Vec<BoolLiteral> {
    assumptions
        .iter()
        .zip(active)
        .filter_map(|(&assumption, &is_active)| is_active.then_some(assumption))
        .collect()
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
        || !program.lazy_constraints.is_empty()
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

/// Solve a model through the optional external-solver oracle.
pub fn solve_math_program_external(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    program.validate()?;
    if !program.secondary_objectives.is_empty() {
        return solve_hierarchical_math_program_external(program, opts);
    }
    solve_math_program_external_single_objective(program, opts)
}

#[deprecated(
    note = "use solve_math_program_external; the default linear external path is the Rust CLI bridge"
)]
pub fn solve_math_program_external_scipy(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    solve_math_program_external(program, opts)
}

fn solve_math_program_external_single_objective(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
) -> Result<MathProgramSolution, MathProgramError> {
    program.validate()?;
    let method = opts.method.clone().unwrap_or_else(|| {
        if program.has_quadratic_objective()
            || program.has_conic_constraints()
            || program.has_quadratic_constraints()
        {
            if program.has_discrete_features() {
                return "rust".to_string();
            }
            "SLSQP".to_string()
        } else {
            "highs-cli".to_string()
        }
    });
    if let Some(solution) = solve_math_program_external_linear_cli(program, opts, &method)? {
        return Ok(solution);
    }
    if let Some(solution) =
        solve_math_program_external_quadratic_rust_reference(program, opts, &method)?
    {
        return Ok(solution);
    }
    let python = resolve_external_math_program_python(opts);
    let script = opts
        .script
        .clone()
        .unwrap_or_else(|| "external-references/math-program/math_program_solve.py".to_string());
    let external_options = encode_external_math_program_options(opts)?;

    let (payload, compiled) = if program.has_discrete_features()
        && (program.has_conic_constraints() || program.has_quadratic_constraints())
    {
        if !can_encode_direct_mixed_integer_nonlinear(program) {
            return Ok(MathProgramSolution {
                status: MathProgramStatus::NumericalError,
                x: Vec::new(),
                objective: f64::NAN,
                best_bound: None,
                mip_gap: None,
                nodes_explored: None,
                first_branch_variable: None,
                solver_version: None,
                iterations: None,
                control_feedback: None,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                unbounded_ray: None,
                infeasibility_certificate: None,
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
                "options": external_options.clone(),
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
                best_bound: None,
                mip_gap: None,
                nodes_explored: None,
                first_branch_variable: None,
                solver_version: None,
                iterations: None,
                control_feedback: None,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                unbounded_ray: None,
                infeasibility_certificate: None,
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
                "options": external_options.clone(),
            }),
            None,
        )
    } else if program.has_discrete_features() {
        let compiled = compile_mip(program)?;
        let mut mip_payload = encode_ipmip_problem(&compiled.problem);
        let mut mip_external_options = external_options.clone();
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
        if let Some(priorities) = &opts.branch_priorities {
            if let Some(object) = mip_external_options.as_object_mut() {
                object.insert(
                    "branchPriorities".to_string(),
                    Value::Array(
                        canonical_branch_priorities(program, &compiled, priorities)?
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
                "options": mip_external_options,
            }),
            Some(compiled),
        )
    } else if program.has_conic_constraints() || program.has_quadratic_constraints() {
        (
            json!({
                "kind": "conic",
                "conic": encode_conic_problem(program)?,
                "method": method,
                "options": external_options.clone(),
            }),
            None,
        )
    } else if program.has_quadratic_objective() {
        (
            json!({
                "kind": "qp",
                "qp": encode_qp_problem(program)?,
                "method": method,
                "options": external_options.clone(),
            }),
            None,
        )
    } else {
        (
            json!({
                "kind": "lp",
                "lp": encode_lp_problem(&program.to_lp_problem()?),
                "method": method,
                "options": external_options,
            }),
            None,
        )
    };

    let raw = run_external_math_program(&python, &script, &method, &payload)?;
    if should_fallback_highs_default_to_cli(program, &method, &raw, opts) {
        if let Some(solution) =
            solve_math_program_external_linear_cli(program, opts, "highs-cli:default")?
        {
            return Ok(solution);
        }
    }
    if should_fallback_glpk_default_to_cli(&method, &raw, opts) {
        if let Some(solution) =
            solve_math_program_external_linear_cli(program, opts, "glpk-cli:default")?
        {
            return Ok(solution);
        }
    }
    let status = parse_external_status(&raw);
    let raw_x = raw
        .get("x")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
        .unwrap_or_default();
    let x = match &compiled {
        Some(compiled) if raw_x.len() == compiled.problem.c.len() => compiled.original_x(&raw_x),
        _ => raw_x,
    };
    let objective = if x.len() == program.variables.len() {
        objective_value(program, &x)
    } else {
        raw.get("objective")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN)
    };
    let raw_best_bound = raw.get("bestBound").and_then(Value::as_f64);
    let best_bound = raw_best_bound.and_then(|bound| match &compiled {
        Some(compiled) => {
            original_mip_best_bound(bound, compiled_objective_offset(program, compiled))
        }
        None => finite_option(bound),
    });
    let raw_mip_gap = raw.get("mipGap").and_then(Value::as_f64);
    let mip_gap = original_mip_gap(best_bound, objective)
        .or_else(|| raw_mip_gap.and_then(|gap| gap.is_finite().then_some(gap.max(0.0))));

    let mut solution = MathProgramSolution {
        status,
        x,
        objective,
        best_bound,
        mip_gap,
        nodes_explored: raw
            .get("nodesExplored")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
        first_branch_variable: None,
        solver_version: raw
            .get("solverVersion")
            .and_then(Value::as_str)
            .map(str::to_string),
        iterations: raw
            .get("iterations")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
        control_feedback: parse_external_solver_control_feedback(&raw),
        dual_ub: parse_external_f64_array(&raw, "dualUB"),
        dual_eq: parse_external_f64_array(&raw, "dualEQ"),
        reduced_costs: parse_external_f64_array(&raw, "reducedCosts"),
        var_basis: parse_external_string_array(&raw, "varBasis"),
        row_basis: parse_external_string_array(&raw, "rowBasis"),
        unbounded_ray: parse_external_f64_array(&raw, "unboundedRay"),
        infeasibility_certificate: parse_external_lp_infeasibility_certificate(&raw),
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
    };
    infer_missing_external_lp_basis(program, &mut solution)?;
    Ok(solution)
}

fn resolve_external_math_program_python(opts: &ExternalMathProgramOptions) -> String {
    let python_bin = std::env::var("PYTHON_BIN").ok();
    let python = std::env::var("PYTHON").ok();
    resolve_external_math_program_python_from(opts, python_bin.as_deref(), python.as_deref())
}

fn resolve_external_math_program_python_from(
    opts: &ExternalMathProgramOptions,
    python_bin: Option<&str>,
    python: Option<&str>,
) -> String {
    opts.python
        .clone()
        .or_else(|| python_bin.map(str::to_string))
        .or_else(|| python.map(str::to_string))
        .unwrap_or_else(|| "python3".to_string())
}

fn should_fallback_highs_default_to_cli(
    program: &MathProgram,
    method: &str,
    raw: &Value,
    opts: &ExternalMathProgramOptions,
) -> bool {
    if opts.script.is_some()
        || !is_highs_default_method(method)
        || !program_supports_linear_cli_fallback(program)
    {
        return false;
    }
    if parse_external_status(raw) != MathProgramStatus::NumericalError {
        return false;
    }
    raw_message_mentions_missing_python_package(raw, &["scipy", "numpy"])
}

fn should_fallback_glpk_default_to_cli(
    method: &str,
    raw: &Value,
    opts: &ExternalMathProgramOptions,
) -> bool {
    if opts.script.is_some() || !is_glpk_default_method(method) {
        return false;
    }
    if parse_external_status(raw) != MathProgramStatus::NumericalError {
        return false;
    }
    raw_message_mentions_missing_python_package(raw, &["swiglpk"])
        || raw
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|message| message.to_ascii_lowercase().contains("glpk unavailable"))
}

fn raw_message_mentions_missing_python_package(raw: &Value, packages: &[&str]) -> bool {
    raw.get("message")
        .and_then(Value::as_str)
        .is_some_and(|message| {
            let message = message.to_ascii_lowercase();
            packages.iter().any(|package| {
                let package = package.to_ascii_lowercase();
                message.contains(&format!("no module named '{package}'"))
                    || message.contains(&format!("no module named \"{package}\""))
                    || message.contains(&format!("no module named {package}"))
            })
        })
}

fn is_highs_default_method(method: &str) -> bool {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    let normalized = normalized
        .strip_suffix(":default")
        .unwrap_or(&normalized)
        .to_string();
    matches!(normalized.as_str(), "highs" | "highs-ds" | "highs-ipm")
}

fn is_glpk_default_method(method: &str) -> bool {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "glpk" | "glpk:default" | "glpsol" | "glpsol:default"
    )
}

fn program_supports_linear_cli_fallback(program: &MathProgram) -> bool {
    if program.has_discrete_features() {
        !program.has_conic_constraints()
            && !program.has_quadratic_constraints()
            && !(program.has_quadratic_objective()
                && quadratic_objective_has_native_nonlinear_terms(program))
    } else {
        !program.has_conic_constraints()
            && !program.has_quadratic_constraints()
            && !program.has_quadratic_objective()
    }
}

fn solve_math_program_external_linear_cli(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
    method: &str,
) -> Result<Option<MathProgramSolution>, MathProgramError> {
    let Some(solver) = resolve_external_math_program_linear_cli_solver(program, opts, method)
    else {
        return Ok(None);
    };

    if program.has_discrete_features() {
        if program.has_conic_constraints()
            || program.has_quadratic_constraints()
            || (program.has_quadratic_objective()
                && quadratic_objective_has_native_nonlinear_terms(program))
        {
            return Ok(Some(math_program_external_cli_failure(
                method,
                "linear CLI adapters only accept linearly compiled LP/MIP facade models; use a nonlinear external method for QP/QCP/SOCP models"
                    .to_string(),
            )));
        }
        let kind = ExternalLinearCliKind::Mip;
        if !solver.supports_kind(kind) {
            return Ok(Some(math_program_external_cli_failure(
                method,
                format!(
                    "{} does not support {} models through the local CLI bridge",
                    solver.display_name(),
                    kind.as_str()
                ),
            )));
        }
        let compiled = compile_mip(program)?;
        let mut cli_opts = external_linear_cli_options_from_math(opts, solver)?;
        if let Some(start) = &opts.mip_start {
            cli_opts.mip_start = Some(canonical_mip_start(program, &compiled, start)?);
        }
        if let Some(priorities) = &opts.branch_priorities {
            cli_opts.branch_priorities =
                Some(canonical_branch_priorities(program, &compiled, priorities)?);
        }
        let cli_solution = solve_ipmip_with_external_cli(&compiled.problem, &cli_opts);
        return Ok(Some(math_program_solution_from_external_linear_cli(
            program,
            Some(&compiled),
            cli_solution,
        )));
    }

    if program.has_conic_constraints()
        || program.has_quadratic_constraints()
        || program.has_quadratic_objective()
    {
        return Ok(Some(math_program_external_cli_failure(
            method,
            "linear CLI adapters only accept LP/MIP models; use a nonlinear external method for QP/QCP/SOCP models"
                .to_string(),
        )));
    }
    let kind = ExternalLinearCliKind::Lp;
    if !solver.supports_kind(kind) {
        return Ok(Some(math_program_external_cli_failure(
            method,
            format!(
                "{} does not support {} models through the local CLI bridge",
                solver.display_name(),
                kind.as_str()
            ),
        )));
    }
    let lp = program.to_lp_problem()?;
    let cli_solution =
        solve_lp_with_external_cli(&lp, &external_linear_cli_options_from_math(opts, solver)?);
    Ok(Some(math_program_solution_from_external_linear_cli(
        program,
        None,
        cli_solution,
    )))
}

fn parse_external_math_program_linear_cli_method(method: &str) -> Option<ExternalLinearCliSolver> {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    let normalized = normalized
        .strip_suffix(":default")
        .unwrap_or(&normalized)
        .to_string();
    match normalized.as_str() {
        "highs-cli" | "highs:cli" => Some(ExternalLinearCliSolver::Highs),
        "glpk-cli" | "glpsol-cli" | "glpk:cli" | "glpsol:cli" => {
            Some(ExternalLinearCliSolver::Glpk)
        }
        "scip-cli" | "scip:cli" => Some(ExternalLinearCliSolver::Scip),
        "cbc-cli" | "coin-cbc-cli" | "coin-or-cbc-cli" | "cbc:cli" => {
            Some(ExternalLinearCliSolver::Cbc)
        }
        "clp-cli" | "clp:cli" => Some(ExternalLinearCliSolver::Clp),
        "soplex-cli" | "soplex:cli" => Some(ExternalLinearCliSolver::Soplex),
        "qsopt-ex-cli" | "qsopt:cli" | "qsopt-ex:cli" => Some(ExternalLinearCliSolver::QsoptEx),
        "lp-solve-cli" | "lpsolve-cli" | "lp-solve:cli" | "lpsolve:cli" => {
            Some(ExternalLinearCliSolver::LpSolve)
        }
        "gurobi-cli" | "gurobi-cl" | "gurobi:cli" => Some(ExternalLinearCliSolver::Gurobi),
        "cplex-cli" | "cplex:cli" => Some(ExternalLinearCliSolver::Cplex),
        "xpress-cli" | "optimizer-cli" | "xpress:cli" | "optimizer:cli" => {
            Some(ExternalLinearCliSolver::Xpress)
        }
        "lindo-cli" | "runlindo-cli" | "lindo:cli" | "runlindo:cli" => {
            Some(ExternalLinearCliSolver::Lindo)
        }
        _ => None,
    }
}

fn parse_external_math_program_default_linear_cli_method(
    method: &str,
) -> Option<ExternalLinearCliSolver> {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    let normalized = normalized
        .strip_suffix(":default")
        .unwrap_or(&normalized)
        .to_string();
    match normalized.as_str() {
        "scip" => Some(ExternalLinearCliSolver::Scip),
        "cbc" | "coin-cbc" | "coin-or-cbc" => Some(ExternalLinearCliSolver::Cbc),
        "clp" => Some(ExternalLinearCliSolver::Clp),
        "soplex" => Some(ExternalLinearCliSolver::Soplex),
        "qsopt" | "qsopt-ex" => Some(ExternalLinearCliSolver::QsoptEx),
        "lp-solve" | "lpsolve" => Some(ExternalLinearCliSolver::LpSolve),
        _ => None,
    }
}

fn resolve_external_math_program_linear_cli_solver(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
    method: &str,
) -> Option<ExternalLinearCliSolver> {
    if let Some(solver) = parse_external_math_program_linear_cli_method(method) {
        return Some(solver);
    }
    if opts.python.is_some()
        || opts.script.is_some()
        || !program_supports_linear_cli_fallback(program)
    {
        return None;
    }
    if is_highs_default_method(method) {
        return Some(ExternalLinearCliSolver::Highs);
    }
    if is_glpk_default_method(method) {
        return Some(ExternalLinearCliSolver::Glpk);
    }
    if let Some(solver) = parse_external_math_program_default_linear_cli_method(method) {
        return Some(solver);
    }
    None
}

fn external_linear_cli_options_from_math(
    opts: &ExternalMathProgramOptions,
    solver: ExternalLinearCliSolver,
) -> Result<ExternalLinearCliOptions, MathProgramError> {
    let _validated = encode_external_math_program_options(opts)?;
    Ok(ExternalLinearCliOptions {
        solver,
        model_format: if solver == ExternalLinearCliSolver::Glpk {
            ExternalLinearCliModelFormat::Mps
        } else {
            ExternalLinearCliModelFormat::CplexLp
        },
        time_limit_secs: opts.time_limit_ms.map(|ms| ms / 1000.0),
        node_limit: opts.node_limit,
        max_nodes: opts.node_limit.map(|limit| limit as u64),
        solution_limit: opts.solution_limit,
        solution_pool_size: opts.solution_pool_size,
        relative_gap: opts.relative_gap,
        absolute_gap: opts.absolute_gap,
        objective_limit: opts.objective_limit,
        primal_feasibility_tolerance: opts.primal_feasibility_tolerance,
        dual_feasibility_tolerance: opts.dual_feasibility_tolerance,
        integer_feasibility_tolerance: opts.integer_feasibility_tolerance,
        threads: opts.threads,
        random_seed: opts.random_seed,
        presolve: opts.presolve,
        cuts: opts.cuts,
        heuristics: opts.heuristics,
        branch_rule: opts.branch_rule,
        node_selection: opts.node_selection,
        python: opts.python.clone(),
        ..Default::default()
    })
}

fn math_program_solution_from_external_linear_cli(
    program: &MathProgram,
    compiled: Option<&CompiledMip>,
    cli: ExternalLinearCliSolution,
) -> MathProgramSolution {
    let status = from_external_linear_cli_status(cli.status);
    let x = match compiled {
        Some(compiled) if cli.x.len() == compiled.problem.c.len() => compiled.original_x(&cli.x),
        _ => cli.x.clone(),
    };
    let objective = if x.len() == program.variables.len() {
        objective_value(program, &x)
    } else {
        cli.objective.unwrap_or(f64::NAN)
    };
    let best_bound = cli.best_bound.and_then(|bound| match compiled {
        Some(compiled) => {
            original_mip_best_bound(bound, compiled_objective_offset(program, compiled))
        }
        None => finite_option(bound),
    });
    let mip_gap = original_mip_gap(best_bound, objective).or_else(|| {
        cli.mip_gap
            .and_then(|gap| gap.is_finite().then_some(gap.max(0.0)))
    });
    let nodes_explored = cli
        .nodes_explored
        .and_then(|nodes| usize::try_from(nodes).ok());
    let iterations = cli
        .iterations
        .and_then(|iterations| usize::try_from(iterations).ok());
    let control_feedback = math_program_control_feedback_from_external_linear_cli(&cli);

    MathProgramSolution {
        status,
        x,
        objective,
        best_bound,
        mip_gap,
        nodes_explored,
        first_branch_variable: None,
        solver_version: cli.solver_version,
        iterations,
        control_feedback,
        dual_ub: cli.dual_ub,
        dual_eq: cli.dual_eq,
        reduced_costs: cli.reduced_costs,
        var_basis: cli.var_basis,
        row_basis: cli.row_basis,
        unbounded_ray: None,
        infeasibility_certificate: None,
        solver: cli.solver,
        message: Some(cli.message),
    }
}

fn math_program_control_feedback_from_external_linear_cli(
    cli: &ExternalLinearCliSolution,
) -> Option<MathProgramSolverControlFeedback> {
    let feedback = MathProgramSolverControlFeedback {
        solution_limit: cli.solution_limit,
        solution_pool_size: cli.solution_pool_size,
        objective_limit: cli.objective_limit,
        absolute_gap: cli.absolute_gap,
        primal_feasibility_tolerance: cli.primal_feasibility_tolerance,
        dual_feasibility_tolerance: cli.dual_feasibility_tolerance,
        integer_feasibility_tolerance: cli.integer_feasibility_tolerance,
        threads: cli.threads,
        random_seed: cli.random_seed,
        presolve: cli.presolve.clone(),
        cuts: cli.cuts.clone(),
        heuristics: cli.heuristics.clone(),
        branch_rule: cli.branch_rule.clone(),
        node_selection: cli.node_selection.clone(),
        branch_priorities_accepted: cli.branch_priorities_accepted,
        branch_priority_count: cli
            .branch_priority_count
            .and_then(|count| usize::try_from(count).ok()),
        mip_start_accepted: cli.mip_start_accepted,
        mip_start_objective: cli.mip_start_objective,
    };
    (feedback != MathProgramSolverControlFeedback::default()).then_some(feedback)
}

fn from_external_linear_cli_status(status: ExternalLinearCliStatus) -> MathProgramStatus {
    match status {
        ExternalLinearCliStatus::Optimal => MathProgramStatus::Optimal,
        ExternalLinearCliStatus::Feasible => MathProgramStatus::Feasible,
        ExternalLinearCliStatus::Infeasible => MathProgramStatus::Infeasible,
        ExternalLinearCliStatus::Unbounded => MathProgramStatus::Unbounded,
        ExternalLinearCliStatus::Unavailable
        | ExternalLinearCliStatus::NumericalError
        | ExternalLinearCliStatus::Unknown => MathProgramStatus::NumericalError,
    }
}

fn math_program_external_cli_failure(method: &str, message: String) -> MathProgramSolution {
    MathProgramSolution {
        status: MathProgramStatus::NumericalError,
        x: Vec::new(),
        objective: f64::NAN,
        best_bound: None,
        mip_gap: None,
        nodes_explored: None,
        first_branch_variable: None,
        solver_version: None,
        iterations: None,
        control_feedback: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
        solver: method.to_string(),
        message: Some(message),
    }
}

fn solve_math_program_external_quadratic_rust_reference(
    program: &MathProgram,
    opts: &ExternalMathProgramOptions,
    method: &str,
) -> Result<Option<MathProgramSolution>, MathProgramError> {
    if opts.python.is_some()
        || opts.script.is_some()
        || !external_math_program_method_prefers_rust_quadratic_reference(method)
    {
        return Ok(None);
    }
    let has_nonlinear = program.has_quadratic_objective()
        || program.has_conic_constraints()
        || program.has_quadratic_constraints();
    if !has_nonlinear {
        return Ok(None);
    }
    let force_rust = external_math_program_method_is_rust_quadratic_reference(method);
    let unsupported = |message: String| {
        if force_rust {
            Some(math_program_external_cli_failure(method, message))
        } else {
            None
        }
    };
    let _validated = encode_external_math_program_options(opts)?;
    let has_discrete = program.has_discrete_features();
    if has_discrete && !can_encode_direct_mixed_integer_nonlinear(program) {
        return Ok(unsupported(
            "Rust mixed-integer nonlinear reference requires direct continuous/integer/binary variables without lowered discrete constraints"
                .to_string(),
        ));
    }

    let reference_opts = ExternalQuadraticReferenceOptions {
        solver: ExternalQuadraticReferenceSolver::RustInternal,
        max_enumerations: opts.node_limit,
    };

    if program.has_conic_constraints() {
        if program.has_quadratic_constraints() || program.has_quadratic_objective() {
            return Ok(unsupported(
                "Rust mixed-integer SOCP reference currently accepts linear objectives with SOC constraints only"
                    .to_string(),
            ));
        }
        let socp = math_program_to_second_order_cone_reference(program)?;
        let solution = if has_discrete {
            let problem = MixedIntegerSecondOrderConeProgram {
                socp,
                integer_vars: math_program_integer_var_mask(program),
            };
            solve_misocp_with_external_reference(&problem, &reference_opts)
        } else {
            solve_socp_with_external_reference(&socp, &reference_opts)
        };
        return Ok(Some(
            math_program_solution_from_external_quadratic_reference(program, solution),
        ));
    }

    if program.has_quadratic_constraints() {
        let qcp = math_program_to_quadratically_constrained_reference(program)?;
        let solution = if has_discrete {
            let problem = MixedIntegerQuadraticallyConstrainedProgram {
                qcp,
                integer_vars: math_program_integer_var_mask(program),
            };
            solve_miqcp_with_external_reference(&problem, &reference_opts)
        } else {
            solve_qcp_with_external_reference(&qcp, &reference_opts)
        };
        return Ok(Some(
            math_program_solution_from_external_quadratic_reference(program, solution),
        ));
    }

    if program.has_quadratic_objective() && quadratic_objective_has_native_nonlinear_terms(program)
    {
        let qp = math_program_to_quadratic_reference(program)?;
        let solution = if has_discrete {
            let problem = MixedIntegerQuadraticProgram {
                qp,
                integer_vars: math_program_integer_var_mask(program),
            };
            solve_miqp_with_external_reference(&problem, &reference_opts)
        } else {
            solve_qp_with_external_reference(&qp, &reference_opts)
        };
        return Ok(Some(
            math_program_solution_from_external_quadratic_reference(program, solution),
        ));
    }

    Ok(None)
}

fn external_math_program_method_prefers_rust_quadratic_reference(method: &str) -> bool {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "rust"
            | "rust-internal"
            | "rust:internal"
            | "auto"
            | "fallback"
            | "slsqp"
            | "scipy"
            | "scipy:slsqp"
            | "bounded-integer-qp-enumeration"
            | "bounded-integer-conic-enumeration"
    )
}

fn external_math_program_method_is_rust_quadratic_reference(method: &str) -> bool {
    let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "rust" | "rust-internal" | "rust:internal" | "auto" | "fallback"
    )
}

fn math_program_solution_from_external_quadratic_reference(
    program: &MathProgram,
    reference: ExternalQuadraticReferenceSolution,
) -> MathProgramSolution {
    let status = match reference.status {
        ExternalQuadraticReferenceStatus::Optimal => MathProgramStatus::Optimal,
        ExternalQuadraticReferenceStatus::Infeasible => MathProgramStatus::Infeasible,
        ExternalQuadraticReferenceStatus::Unbounded => MathProgramStatus::Unbounded,
        ExternalQuadraticReferenceStatus::NumericalError
        | ExternalQuadraticReferenceStatus::Unavailable => MathProgramStatus::NumericalError,
    };
    let objective = if reference.x.len() == program.variables.len() {
        objective_value(program, &reference.x)
    } else {
        reference.objective.unwrap_or(f64::NAN)
    };
    MathProgramSolution {
        status,
        x: reference.x,
        objective,
        best_bound: None,
        mip_gap: None,
        nodes_explored: reference
            .enumerated
            .and_then(|value| usize::try_from(value).ok()),
        first_branch_variable: None,
        solver_version: None,
        iterations: reference
            .iterations
            .and_then(|value| usize::try_from(value).ok()),
        control_feedback: None,
        dual_ub: reference.dual_ub,
        dual_eq: reference.dual_eq,
        reduced_costs: reference.reduced_gradient,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
        solver: reference.solver,
        message: Some(reference.message),
    }
}

fn math_program_to_quadratic_reference(
    program: &MathProgram,
) -> Result<QuadraticProgram, MathProgramError> {
    let lp = program.to_linear_relaxation_lp_problem()?;
    let minimize_sign = objective_minimize_sign(program.sense);
    Ok(QuadraticProgram {
        q: scale_matrix(&quadratic_hessian(program), minimize_sign),
        c: scale_vec(&lp.c, minimize_sign),
        a_ub: lp.a_ub,
        b_ub: lp.b_ub,
        a_eq: lp.a_eq,
        b_eq: lp.b_eq,
        lb: lp.lb,
        ub: lp.ub,
        var_names: lp.var_names,
    })
}

fn math_program_to_second_order_cone_reference(
    program: &MathProgram,
) -> Result<SecondOrderConeProgram, MathProgramError> {
    let lp = program.to_linear_relaxation_lp_problem()?;
    let minimize_sign = objective_minimize_sign(program.sense);
    let n = program.variables.len();
    Ok(SecondOrderConeProgram {
        c: scale_vec(&lp.c, minimize_sign),
        a_ub: lp.a_ub,
        b_ub: lp.b_ub,
        a_eq: lp.a_eq,
        b_eq: lp.b_eq,
        lb: lp.lb,
        ub: lp.ub,
        cones: program
            .second_order_cones
            .iter()
            .map(|cone| QpSecondOrderCone {
                a: cone
                    .terms
                    .iter()
                    .map(|term| dense_row(n, &term.coeffs))
                    .collect(),
                b: cone.terms.iter().map(|term| term.constant).collect(),
                c: dense_row(n, &cone.rhs_coeffs),
                d: cone.rhs_constant,
                name: Some(cone.name.clone()),
            })
            .collect(),
        var_names: lp.var_names,
    })
}

fn math_program_to_quadratically_constrained_reference(
    program: &MathProgram,
) -> Result<QuadraticallyConstrainedProgram, MathProgramError> {
    let qp = math_program_to_quadratic_reference(program)?;
    let n = program.variables.len();
    let mut constraints = Vec::new();
    for row in &program.quadratic_constraints {
        match row.sense {
            RowSense::Le => constraints.push(math_program_quadratic_constraint_reference(
                n, row, 1.0, None,
            )),
            RowSense::Ge => constraints.push(math_program_quadratic_constraint_reference(
                n,
                row,
                -1.0,
                Some("__ge".to_string()),
            )),
            RowSense::Eq => {
                constraints.push(math_program_quadratic_constraint_reference(
                    n, row, 1.0, None,
                ));
                constraints.push(math_program_quadratic_constraint_reference(
                    n,
                    row,
                    -1.0,
                    Some("__eq_lower".to_string()),
                ));
            }
        }
    }
    Ok(QuadraticallyConstrainedProgram {
        q: qp.q,
        c: qp.c,
        a_ub: qp.a_ub,
        b_ub: qp.b_ub,
        a_eq: qp.a_eq,
        b_eq: qp.b_eq,
        lb: qp.lb,
        ub: qp.ub,
        quadratic_constraints: constraints,
        var_names: qp.var_names,
    })
}

fn math_program_quadratic_constraint_reference(
    n: usize,
    row: &QuadraticConstraint,
    scale: f64,
    name_suffix: Option<String>,
) -> QpQuadraticConstraint {
    let mut q = vec![vec![0.0; n]; n];
    for term in &row.quadratic_terms {
        let coeff = scale * term.coeff;
        if term.var_i == term.var_j {
            q[term.var_i][term.var_i] += coeff;
        } else {
            q[term.var_i][term.var_j] += 0.5 * coeff;
            q[term.var_j][term.var_i] += 0.5 * coeff;
        }
    }
    QpQuadraticConstraint {
        q,
        c: scale_row(&dense_row(n, &row.linear_terms), scale),
        rhs: scale * row.rhs,
        name: Some(match name_suffix {
            Some(suffix) => format!("{}{}", row.name, suffix),
            None => row.name.clone(),
        }),
    }
}

fn math_program_integer_var_mask(program: &MathProgram) -> Vec<bool> {
    program
        .variables
        .iter()
        .map(|var| matches!(var.var_type, VariableType::Integer | VariableType::Binary))
        .collect()
}

fn objective_minimize_sign(sense: ObjectiveSense) -> f64 {
    match sense {
        ObjectiveSense::Min => 1.0,
        ObjectiveSense::Max => -1.0,
    }
}

fn solve_lp_with_backend(lp: &LPProblem, opts: &MathProgramSolveOptions) -> LPSolution {
    match opts.lp_backend {
        MathProgramLpBackend::InternalSimplex => solve_lp_internal(lp, &opts.lp_simplex),
        MathProgramLpBackend::DESSimplex => {
            let sol = solve_lp_via_des(lp, &opts.lp_des);
            LPSolution {
                status: sol.status,
                x: sol.x,
                objective: sol.objective,
                dual_ub: sol.dual_ub,
                dual_eq: sol.dual_eq,
                reduced_costs: sol.reduced_costs,
                var_basis: sol.var_basis,
                row_basis: sol.row_basis,
                unbounded_ray: None,
                infeasibility_certificate: None,
                iters: sol.iters,
                solver: sol.solver,
                elapsed_ms: sol.elapsed_ms,
                message: sol.message,
            }
        }
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
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: feasibility.iters,
            control_feedback: None,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            unbounded_ray: None,
            infeasibility_certificate: None,
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
                best_bound: None,
                mip_gap: None,
                nodes_explored: None,
                first_branch_variable: None,
                solver_version: None,
                iterations: linear.iters,
                control_feedback: None,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                unbounded_ray: None,
                infeasibility_certificate: None,
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
        best_bound: None,
        mip_gap: None,
        nodes_explored: None,
        first_branch_variable: None,
        solver_version: None,
        iterations: Some(iterations),
        control_feedback: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
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
        if solution.x.len() == program.variables.len() {
            solution.objective = objective_value(program, &solution.x);
        }
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
        best_bound: None,
        mip_gap: None,
        nodes_explored: None,
        first_branch_variable: None,
        solver_version: None,
        iterations: None,
        control_feedback: None,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
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

    let timeout_ms = math_program_external_timeout_ms();
    let (out, timed_out) = wait_for_math_program_external_output(child, timeout_ms)
        .map_err(MathProgramError::External)?;

    if out.status.code() != Some(0) {
        let code = out
            .status
            .code()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let stderr = if timed_out {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.trim().is_empty() {
                format!("external math-program solver timed out after {timeout_ms}ms")
            } else {
                format!("{stderr}; external math-program solver timed out after {timeout_ms}ms")
            }
        } else {
            String::from_utf8_lossy(&out.stderr).to_string()
        };
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

fn math_program_external_timeout_ms() -> u64 {
    std::env::var("MATH_PROGRAM_EXTERNAL_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_math_program_external_output(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => {
                return Err(format!(
                    "failed to poll external math-program solver: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("external solver wait failed: {err}"))
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
        "lazyConstraints": mip.lazy_constraints.as_ref().map(|rows| rows.iter().map(|row| {
            json!({
                "coefs": row.coefs,
                "rhs": row.rhs,
                "name": row.name,
            })
        }).collect::<Vec<_>>()),
    })
}

fn parse_external_status(raw: &Value) -> MathProgramStatus {
    match raw.get("status").and_then(Value::as_str) {
        Some("optimal") => MathProgramStatus::Optimal,
        Some("feasible") => MathProgramStatus::Feasible,
        Some("infeasible") => MathProgramStatus::Infeasible,
        Some("unbounded") => MathProgramStatus::Unbounded,
        Some("iter-limit") => MathProgramStatus::IterLimit,
        Some("node-limit") => MathProgramStatus::NodeLimit,
        Some("time-limit") => MathProgramStatus::TimeLimit,
        _ => MathProgramStatus::NumericalError,
    }
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn infer_missing_external_lp_basis(
    program: &MathProgram,
    solution: &mut MathProgramSolution,
) -> Result<(), MathProgramError> {
    if external_basis_is_uninformative_pdlp_status(solution) {
        solution.var_basis = None;
        solution.row_basis = None;
    }
    if solution.status != MathProgramStatus::Optimal
        || (solution.var_basis.is_some() && solution.row_basis.is_some())
        || program.has_discrete_features()
        || program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
    {
        return Ok(());
    }

    let lp = program.to_lp_problem()?;
    let Some(reduced_costs) = solution.reduced_costs.as_deref() else {
        return Ok(());
    };
    let dual_ub = solution.dual_ub.as_deref().unwrap_or(&[]);
    let dual_eq = solution.dual_eq.as_deref().unwrap_or(&[]);
    let (var_basis, row_basis) = infer_lp_basis_from_complementarity(
        &solution.x,
        lp.lb.as_deref(),
        lp.ub.as_deref(),
        lp.a_ub.as_deref().unwrap_or(&[]),
        lp.b_ub.as_deref().unwrap_or(&[]),
        dual_ub,
        lp.a_eq.as_deref().unwrap_or(&[]),
        lp.b_eq.as_deref().unwrap_or(&[]),
        dual_eq,
        reduced_costs,
    );
    if solution.var_basis.is_none() {
        solution.var_basis = var_basis;
    }
    if solution.row_basis.is_none() {
        solution.row_basis = row_basis;
    }
    Ok(())
}

fn external_basis_is_uninformative_pdlp_status(solution: &MathProgramSolution) -> bool {
    let solver = solution.solver.to_ascii_lowercase();
    if !solver.contains("pdlp") {
        return false;
    }
    let all_free = |values: &Option<Vec<String>>| {
        values.as_ref().is_some_and(|values| {
            !values.is_empty() && values.iter().all(|status| status == "free")
        })
    };
    all_free(&solution.var_basis) || all_free(&solution.row_basis)
}

fn infer_lp_basis_from_complementarity(
    x: &[f64],
    lower_bounds: Option<&[Option<f64>]>,
    upper_bounds: Option<&[Option<f64>]>,
    le_rows: &[Vec<f64>],
    le_rhs: &[f64],
    dual_ub: &[f64],
    eq_rows: &[Vec<f64>],
    eq_rhs: &[f64],
    dual_eq: &[f64],
    reduced_costs: &[f64],
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    const TOL: f64 = 1.0e-6;
    if x.len() != reduced_costs.len()
        || le_rows.len() != le_rhs.len()
        || le_rows.len() != dual_ub.len()
        || eq_rows.len() != eq_rhs.len()
        || eq_rows.len() != dual_eq.len()
    {
        return (None, None);
    }

    let mut var_basis = Vec::with_capacity(x.len());
    for (idx, (&value, &reduced_cost)) in x.iter().zip(reduced_costs).enumerate() {
        let lower = lp_lower_bound_at(lower_bounds, idx);
        let upper = lp_upper_bound_at(upper_bounds, idx);
        let at_lower = lower.is_some_and(|bound| (value - bound).abs() <= TOL);
        let at_upper = upper.is_some_and(|bound| (value - bound).abs() <= TOL);
        let fixed = lower
            .zip(upper)
            .is_some_and(|(lower, upper)| (lower - upper).abs() <= TOL);
        let status = if fixed && (at_lower || at_upper) {
            "fixed"
        } else if at_lower && reduced_cost.abs() > TOL {
            "at_lower"
        } else if at_upper && reduced_cost.abs() > TOL {
            "at_upper"
        } else if !at_lower && !at_upper && reduced_cost.abs() <= TOL {
            "basic"
        } else {
            return (None, None);
        };
        var_basis.push(status.to_string());
    }

    let mut row_basis = Vec::with_capacity(le_rows.len() + eq_rows.len());
    for ((row, &rhs), &dual) in le_rows.iter().zip(le_rhs).zip(dual_ub) {
        let slack = rhs - dot(row, x);
        if slack < -TOL {
            return (Some(var_basis), None);
        }
        if slack.abs() <= TOL {
            if dual.abs() > TOL {
                row_basis.push("at_upper".to_string());
            } else {
                return (Some(var_basis), None);
            }
        } else {
            row_basis.push("basic".to_string());
        }
    }
    for (row, &rhs) in eq_rows.iter().zip(eq_rhs) {
        if (dot(row, x) - rhs).abs() > TOL {
            return (Some(var_basis), None);
        }
        row_basis.push("fixed".to_string());
    }

    (Some(var_basis), Some(row_basis))
}

fn lp_lower_bound_at(bounds: Option<&[Option<f64>]>, index: usize) -> Option<f64> {
    bounds
        .and_then(|bounds| bounds.get(index).copied())
        .unwrap_or(Some(0.0))
}

fn lp_upper_bound_at(bounds: Option<&[Option<f64>]>, index: usize) -> Option<f64> {
    bounds
        .and_then(|bounds| bounds.get(index).copied())
        .unwrap_or(None)
}

fn encode_external_math_program_options(
    opts: &ExternalMathProgramOptions,
) -> Result<Value, MathProgramError> {
    let mut object = serde_json::Map::new();
    if let Some(time_limit_ms) = opts.time_limit_ms {
        if !time_limit_ms.is_finite() || time_limit_ms <= 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external time_limit_ms must be finite and positive".to_string(),
            ));
        }
        object.insert("timeLimitMs".to_string(), Value::from(time_limit_ms));
    }
    if let Some(node_limit) = opts.node_limit {
        if node_limit == 0 {
            return Err(MathProgramError::InvalidBound(
                "external node_limit must be positive".to_string(),
            ));
        }
        object.insert("nodeLimit".to_string(), Value::from(node_limit));
    }
    if let Some(relative_gap) = opts.relative_gap {
        if !relative_gap.is_finite() || relative_gap < 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external relative_gap must be finite and non-negative".to_string(),
            ));
        }
        object.insert("relativeGap".to_string(), Value::from(relative_gap));
    }
    if let Some(absolute_gap) = opts.absolute_gap {
        if !absolute_gap.is_finite() || absolute_gap < 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external absolute_gap must be finite and non-negative".to_string(),
            ));
        }
        object.insert("absoluteGap".to_string(), Value::from(absolute_gap));
    }
    if let Some(solution_limit) = opts.solution_limit {
        if solution_limit == 0 {
            return Err(MathProgramError::InvalidBound(
                "external solution_limit must be positive".to_string(),
            ));
        }
        object.insert("solutionLimit".to_string(), Value::from(solution_limit));
    }
    if let Some(solution_pool_size) = opts.solution_pool_size {
        if solution_pool_size == 0 {
            return Err(MathProgramError::InvalidBound(
                "external solution_pool_size must be positive".to_string(),
            ));
        }
        object.insert(
            "solutionPoolSize".to_string(),
            Value::from(solution_pool_size),
        );
    }
    if let Some(objective_limit) = opts.objective_limit {
        if !objective_limit.is_finite() {
            return Err(MathProgramError::InvalidBound(
                "external objective_limit must be finite".to_string(),
            ));
        }
        object.insert("objectiveLimit".to_string(), Value::from(objective_limit));
    }
    if let Some(tolerance) = opts.primal_feasibility_tolerance {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external primal_feasibility_tolerance must be finite and positive".to_string(),
            ));
        }
        object.insert(
            "primalFeasibilityTolerance".to_string(),
            Value::from(tolerance),
        );
    }
    if let Some(tolerance) = opts.dual_feasibility_tolerance {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external dual_feasibility_tolerance must be finite and positive".to_string(),
            ));
        }
        object.insert(
            "dualFeasibilityTolerance".to_string(),
            Value::from(tolerance),
        );
    }
    if let Some(tolerance) = opts.integer_feasibility_tolerance {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(MathProgramError::InvalidBound(
                "external integer_feasibility_tolerance must be finite and positive".to_string(),
            ));
        }
        object.insert(
            "integerFeasibilityTolerance".to_string(),
            Value::from(tolerance),
        );
    }
    if let Some(threads) = opts.threads {
        if threads == 0 {
            return Err(MathProgramError::InvalidBound(
                "external threads must be positive".to_string(),
            ));
        }
        object.insert("threads".to_string(), Value::from(threads));
    }
    if let Some(random_seed) = opts.random_seed {
        object.insert("randomSeed".to_string(), Value::from(random_seed));
    }
    if let Some(presolve) = opts.presolve {
        object.insert("presolve".to_string(), Value::from(presolve.as_str()));
    }
    if let Some(cuts) = opts.cuts {
        object.insert("cuts".to_string(), Value::from(cuts.as_str()));
    }
    if let Some(heuristics) = opts.heuristics {
        object.insert("heuristics".to_string(), Value::from(heuristics.as_str()));
    }
    if let Some(branch_rule) = opts.branch_rule {
        object.insert("branchRule".to_string(), Value::from(branch_rule.as_str()));
    }
    if let Some(node_selection) = opts.node_selection {
        object.insert(
            "nodeSelection".to_string(),
            Value::from(node_selection.as_str()),
        );
    }
    Ok(Value::Object(object))
}

fn compiled_objective_offset(program: &MathProgram, compiled: &CompiledMip) -> f64 {
    program.objective_offset
        + program
            .variables
            .iter()
            .zip(&compiled.expansions)
            .map(|(var, expansion)| var.obj * expansion.constant)
            .sum::<f64>()
}

fn original_mip_best_bound(best_bound: f64, objective_offset: f64) -> Option<f64> {
    finite_option(best_bound + objective_offset)
}

fn original_mip_gap(best_bound: Option<f64>, objective: f64) -> Option<f64> {
    best_bound.and_then(|bound| {
        objective
            .is_finite()
            .then_some((bound - objective).abs() / 1.0_f64.max(objective.abs()))
    })
}

fn parse_external_f64_array(raw: &Value, key: &str) -> Option<Vec<f64>> {
    raw.get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
}

fn parse_external_lp_infeasibility_certificate(raw: &Value) -> Option<LPInfeasibilityCertificate> {
    let certificate = raw.get("infeasibilityCertificate")?;
    Some(LPInfeasibilityCertificate {
        dual_ub: parse_external_f64_array(certificate, "dualUB")?,
        dual_eq: parse_external_f64_array(certificate, "dualEQ")?,
        lower_bound: parse_external_f64_array(certificate, "lowerBound")?,
        upper_bound: parse_external_f64_array(certificate, "upperBound")?,
        contradiction: certificate.get("contradiction").and_then(Value::as_f64)?,
    })
}

fn parse_external_solver_control_feedback(raw: &Value) -> Option<MathProgramSolverControlFeedback> {
    let feedback = MathProgramSolverControlFeedback {
        solution_limit: raw.get("solutionLimit").and_then(Value::as_u64),
        solution_pool_size: raw.get("solutionPoolSize").and_then(Value::as_u64),
        objective_limit: raw.get("objectiveLimit").and_then(Value::as_f64),
        absolute_gap: raw.get("absoluteGap").and_then(Value::as_f64),
        primal_feasibility_tolerance: raw
            .get("primalFeasibilityTolerance")
            .and_then(Value::as_f64),
        dual_feasibility_tolerance: raw.get("dualFeasibilityTolerance").and_then(Value::as_f64),
        integer_feasibility_tolerance: raw
            .get("integerFeasibilityTolerance")
            .and_then(Value::as_f64),
        threads: raw
            .get("threads")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        random_seed: raw.get("randomSeed").and_then(Value::as_u64),
        presolve: raw
            .get("presolve")
            .and_then(Value::as_str)
            .map(str::to_string),
        cuts: raw.get("cuts").and_then(Value::as_str).map(str::to_string),
        heuristics: raw
            .get("heuristics")
            .and_then(Value::as_str)
            .map(str::to_string),
        branch_rule: raw
            .get("branchRule")
            .and_then(Value::as_str)
            .map(str::to_string),
        node_selection: raw
            .get("nodeSelection")
            .and_then(Value::as_str)
            .map(str::to_string),
        branch_priorities_accepted: raw.get("branchPrioritiesAccepted").and_then(Value::as_bool),
        branch_priority_count: raw
            .get("branchPriorityCount")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
        mip_start_accepted: raw.get("mipStartAccepted").and_then(Value::as_bool),
        mip_start_objective: raw.get("mipStartObjective").and_then(Value::as_f64),
    };
    (feedback.solution_limit.is_some()
        || feedback.solution_pool_size.is_some()
        || feedback.objective_limit.is_some()
        || feedback.absolute_gap.is_some()
        || feedback.primal_feasibility_tolerance.is_some()
        || feedback.dual_feasibility_tolerance.is_some()
        || feedback.integer_feasibility_tolerance.is_some()
        || feedback.threads.is_some()
        || feedback.random_seed.is_some()
        || feedback.presolve.is_some()
        || feedback.cuts.is_some()
        || feedback.heuristics.is_some()
        || feedback.branch_rule.is_some()
        || feedback.node_selection.is_some()
        || feedback.branch_priorities_accepted.is_some()
        || feedback.branch_priority_count.is_some()
        || feedback.mip_start_accepted.is_some()
        || feedback.mip_start_objective.is_some())
    .then_some(feedback)
}

fn parse_external_string_array(raw: &Value, key: &str) -> Option<Vec<String>> {
    raw.get(key).and_then(Value::as_array).map(|items| {
        items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>()
    })
}

/// Map LP row certificate arrays (`dual_ub`, `dual_eq`, and `row_basis`) back to facade rows.
pub fn map_math_program_lp_row_certificates(
    program: &MathProgram,
    solution: &MathProgramSolution,
) -> Result<Vec<MathProgramLpRowCertificate>, MathProgramError> {
    validate_pure_linear_lp_certificates(program, "LP row certificates")?;

    let inequality_count = program
        .constraints
        .iter()
        .filter(|row| row.sense != RowSense::Eq)
        .count();
    let mut certificates = Vec::with_capacity(program.constraints.len());

    let mut inequality_index = 0usize;
    for (source_index, row) in program.constraints.iter().enumerate() {
        match row.sense {
            RowSense::Le => {
                let generated_row = inequality_index;
                certificates.push(MathProgramLpRowCertificate {
                    generated_row,
                    source_name: row.name.clone(),
                    source_index,
                    source_sense: row.sense,
                    source_rhs: row.rhs,
                    generated_sense: RowSense::Le,
                    generated_rhs: row.rhs,
                    multiplier: 1.0,
                    generated_dual: lp_solution_generated_dual(solution, generated_row, true),
                    source_dual: lp_solution_generated_dual(solution, generated_row, true),
                    basis: lp_solution_row_basis(solution, generated_row),
                });
                inequality_index += 1;
            }
            RowSense::Ge => {
                let generated_row = inequality_index;
                let generated_dual = lp_solution_generated_dual(solution, generated_row, true);
                certificates.push(MathProgramLpRowCertificate {
                    generated_row,
                    source_name: row.name.clone(),
                    source_index,
                    source_sense: row.sense,
                    source_rhs: row.rhs,
                    generated_sense: RowSense::Le,
                    generated_rhs: -row.rhs,
                    multiplier: -1.0,
                    generated_dual,
                    source_dual: generated_dual.map(|dual| -dual),
                    basis: lp_solution_row_basis(solution, generated_row),
                });
                inequality_index += 1;
            }
            RowSense::Eq => {}
        }
    }

    let mut equality_index = 0usize;
    for (source_index, row) in program.constraints.iter().enumerate() {
        if row.sense != RowSense::Eq {
            continue;
        }
        let generated_row = inequality_count + equality_index;
        certificates.push(MathProgramLpRowCertificate {
            generated_row,
            source_name: row.name.clone(),
            source_index,
            source_sense: row.sense,
            source_rhs: row.rhs,
            generated_sense: RowSense::Eq,
            generated_rhs: row.rhs,
            multiplier: 1.0,
            generated_dual: lp_solution_generated_dual(solution, equality_index, false),
            source_dual: lp_solution_generated_dual(solution, equality_index, false),
            basis: lp_solution_row_basis(solution, generated_row),
        });
        equality_index += 1;
    }

    Ok(certificates)
}

/// Map LP reduced-cost and variable-basis arrays back to facade variables.
pub fn map_math_program_lp_variable_certificates(
    program: &MathProgram,
    solution: &MathProgramSolution,
) -> Result<Vec<MathProgramLpVariableCertificate>, MathProgramError> {
    validate_pure_linear_lp_certificates(program, "LP variable certificates")?;
    Ok(program
        .variables
        .iter()
        .enumerate()
        .map(|(variable, var)| MathProgramLpVariableCertificate {
            variable,
            name: var.name.clone(),
            var_type: var.var_type,
            objective_coefficient: var.obj,
            lb: var.lb,
            ub: var.ub,
            value: solution.x.get(variable).copied(),
            reduced_cost: solution
                .reduced_costs
                .as_ref()
                .and_then(|values| values.get(variable).copied()),
            basis: solution
                .var_basis
                .as_ref()
                .and_then(|basis| basis.get(variable).cloned()),
        })
        .collect())
}

fn validate_pure_linear_lp_certificates(
    program: &MathProgram,
    label: &str,
) -> Result<(), MathProgramError> {
    program.validate()?;
    if program.has_discrete_features()
        || program.has_quadratic_objective()
        || program.has_quadratic_constraints()
        || program.has_conic_constraints()
    {
        return Err(MathProgramError::Unsupported(format!(
            "{label} require a pure continuous linear model"
        )));
    }
    Ok(())
}

fn lp_solution_generated_dual(
    solution: &MathProgramSolution,
    index: usize,
    inequality: bool,
) -> Option<f64> {
    if inequality {
        solution.dual_ub.as_ref()?.get(index).copied()
    } else {
        solution.dual_eq.as_ref()?.get(index).copied()
    }
}

fn lp_solution_row_basis(solution: &MathProgramSolution, generated_row: usize) -> Option<String> {
    solution.row_basis.as_ref()?.get(generated_row).cloned()
}

fn from_lp_solution(sol: LPSolution) -> MathProgramSolution {
    MathProgramSolution {
        status: from_lp_status(sol.status),
        x: sol.x,
        objective: sol.objective,
        best_bound: None,
        mip_gap: None,
        nodes_explored: None,
        first_branch_variable: None,
        solver_version: None,
        iterations: sol.iters,
        control_feedback: None,
        dual_ub: sol.dual_ub,
        dual_eq: sol.dual_eq,
        reduced_costs: sol.reduced_costs,
        var_basis: sol.var_basis,
        row_basis: sol.row_basis,
        unbounded_ray: sol.unbounded_ray,
        infeasibility_certificate: sol.infeasibility_certificate,
        solver: sol.solver,
        message: sol.message,
    }
}

fn add_objective_offset_to_solution(solution: &mut MathProgramSolution, offset: f64) {
    if offset.abs() > 1e-12 && solution.objective.is_finite() {
        solution.objective += offset;
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
        IPMIPStatus::GapLimit | IPMIPStatus::Cancelled => MathProgramStatus::IterLimit,
        IPMIPStatus::TimeLimit => MathProgramStatus::TimeLimit,
        IPMIPStatus::SolutionLimit
        | IPMIPStatus::ObjectiveLimit
        | IPMIPStatus::TakeNextIncumbent => MathProgramStatus::Feasible,
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

fn compiled_mip_options(
    program: &MathProgram,
    compiled: &CompiledMip,
    opts: &MathProgramSolveOptions,
    include_original_mip_start: bool,
) -> Result<IPMIPSolveOptions, MathProgramError> {
    let mut mip_opts = opts.mip.clone();
    if include_original_mip_start {
        if let Some(start) = &opts.mip_start {
            mip_opts.mip_start = Some(canonical_mip_start(program, compiled, start)?);
        }
    }
    if let Some(priorities) = &opts.branch_priorities {
        mip_opts.branch_priorities =
            Some(canonical_branch_priorities(program, compiled, priorities)?);
    } else if let Some(priorities) = mip_opts.branch_priorities.as_deref() {
        validate_canonical_branch_priorities(compiled, priorities)?;
    }
    Ok(mip_opts)
}

fn canonical_branch_priorities(
    program: &MathProgram,
    compiled: &CompiledMip,
    priorities: &[i32],
) -> Result<Vec<i32>, MathProgramError> {
    if priorities.len() != program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "branch priorities length {} does not match {} variables",
            priorities.len(),
            program.variables.len()
        )));
    }

    let mut canonical = vec![0; compiled.problem.c.len()];
    for (i, priority) in priorities.iter().copied().enumerate() {
        for &(j, _) in &compiled.expansions[i].terms {
            canonical[j] = priority;
        }
        if matches!(
            program.variables[i].var_type,
            VariableType::SemiContinuous | VariableType::SemiInteger
        ) {
            if let Some(active_idx) =
                compiled_var_index(compiled, &format!("{}__active", program.variables[i].name))
            {
                canonical[active_idx] = priority;
            }
        }
    }
    Ok(canonical)
}

fn validate_canonical_branch_priorities(
    compiled: &CompiledMip,
    priorities: &[i32],
) -> Result<(), MathProgramError> {
    if priorities.len() != compiled.problem.c.len() {
        return Err(MathProgramError::BadIndex(format!(
            "canonical branch priorities length {} does not match {} compiled variables",
            priorities.len(),
            compiled.problem.c.len()
        )));
    }
    Ok(())
}

fn extend_branch_priorities_for_added_variables(
    original_len: usize,
    target_len: usize,
    priorities: &[i32],
) -> Result<Vec<i32>, MathProgramError> {
    if priorities.len() != original_len {
        return Err(MathProgramError::BadIndex(format!(
            "branch priorities length {} does not match {} variables",
            priorities.len(),
            original_len
        )));
    }
    if target_len < original_len {
        return Err(MathProgramError::BadIndex(format!(
            "cannot map branch priorities from {original_len} variables to {target_len} variables"
        )));
    }
    let mut extended = priorities.to_vec();
    extended.resize(target_len, 0);
    Ok(extended)
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
    let mut lazy_rows = Vec::new();

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
    for row in &program.lazy_constraints {
        add_program_row(
            &mut lazy_rows,
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
    for enforced in &program.enforced_constraints {
        add_enforced_linear_rows(program, &mut rows, &expansions, enforced)?;
    }
    for reified in &program.reified_constraints {
        add_reified_linear_rows(
            program,
            &mut names,
            &mut integer_vars,
            &mut ub,
            &mut rows,
            &expansions,
            reified,
        )?;
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
    let lazy_constraints = lazy_rows
        .into_iter()
        .map(|row| BranchOrCutConstraint {
            coefs: dense_row(n, &row.coeffs),
            rhs: row.rhs,
            name: row.name,
            kind: ConstraintKind::Lazy,
        })
        .collect::<Vec<_>>();

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
            lazy_constraints: (!lazy_constraints.is_empty()).then_some(lazy_constraints),
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
    add_canonical_program_row(rows, name, expanded, sense, shifted_rhs);
}

fn add_program_row_with_canonical_terms(
    rows: &mut Vec<SparseRow>,
    name: String,
    expansions: &[LinearExpansion],
    coeffs: &[(usize, f64)],
    canonical_terms: &[(usize, f64)],
    sense: RowSense,
    rhs: f64,
) {
    let (mut expanded, shifted_rhs) = expand_row(expansions, coeffs, rhs);
    expanded.extend_from_slice(canonical_terms);
    add_canonical_program_row(rows, name, combine_terms(&expanded), sense, shifted_rhs);
}

fn add_canonical_program_row(
    rows: &mut Vec<SparseRow>,
    name: String,
    expanded: Vec<(usize, f64)>,
    sense: RowSense,
    shifted_rhs: f64,
) {
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

fn add_enforced_linear_rows(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    enforced: &EnforcedLinearConstraint,
) -> Result<(), MathProgramError> {
    match enforced.sense {
        RowSense::Le => add_enforced_linear_le(
            program,
            rows,
            expansions,
            enforced,
            &enforced.coeffs,
            enforced.rhs,
        ),
        RowSense::Ge => {
            let coeffs = enforced
                .coeffs
                .iter()
                .map(|&(i, v)| (i, -v))
                .collect::<Vec<_>>();
            add_enforced_linear_le(program, rows, expansions, enforced, &coeffs, -enforced.rhs)
        }
        RowSense::Eq => {
            add_enforced_linear_le(
                program,
                rows,
                expansions,
                enforced,
                &enforced.coeffs,
                enforced.rhs,
            )?;
            let coeffs = enforced
                .coeffs
                .iter()
                .map(|&(i, v)| (i, -v))
                .collect::<Vec<_>>();
            add_enforced_linear_le(program, rows, expansions, enforced, &coeffs, -enforced.rhs)
        }
    }
}

fn add_enforced_linear_le(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    enforced: &EnforcedLinearConstraint,
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> Result<(), MathProgramError> {
    let (_, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "enforced linear constraint `{}` needs finite variable bounds for big-M lowering",
            enforced.name
        ))
    })?;
    let big_m = 0.0_f64.max(max_lhs - rhs);
    let mut lifted = coeffs.to_vec();
    let mut shifted_rhs = rhs;
    for literal in &enforced.literals {
        if literal.value {
            lifted.push((literal.var, big_m));
            shifted_rhs += big_m;
        } else {
            lifted.push((literal.var, -big_m));
        }
    }
    add_program_row(
        rows,
        format!("{}__enforced", enforced.name),
        expansions,
        &lifted,
        RowSense::Le,
        shifted_rhs,
    );
    Ok(())
}

fn add_reified_linear_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    reified: &ReifiedLinearConstraint,
) -> Result<(), MathProgramError> {
    match reified.sense {
        RowSense::Le => add_reified_linear_le(
            program,
            rows,
            expansions,
            reified,
            &reified.coeffs,
            reified.rhs,
        ),
        RowSense::Ge => {
            let coeffs = reified
                .coeffs
                .iter()
                .map(|&(idx, value)| (idx, -value))
                .collect::<Vec<_>>();
            add_reified_linear_le(program, rows, expansions, reified, &coeffs, -reified.rhs)
        }
        RowSense::Eq => {
            add_reified_equality_rows(program, names, integer_vars, ub, rows, expansions, reified)
        }
    }
}

fn add_literal_implied_le(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: String,
    literal: BoolLiteral,
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> Result<(), MathProgramError> {
    let (_, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "reified linear constraint `{name}` needs finite variable bounds for big-M lowering"
        ))
    })?;
    let big_m = 0.0_f64.max(max_lhs - rhs);
    let mut lifted = coeffs.to_vec();
    if literal.value {
        lifted.push((literal.var, big_m));
        add_program_row(rows, name, expansions, &lifted, RowSense::Le, rhs + big_m);
    } else {
        lifted.push((literal.var, -big_m));
        add_program_row(rows, name, expansions, &lifted, RowSense::Le, rhs);
    }
    Ok(())
}

fn add_canonical_binary_implied_le(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: String,
    active_binary: usize,
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> Result<(), MathProgramError> {
    let (_, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "reified linear constraint `{name}` needs finite variable bounds for big-M lowering"
        ))
    })?;
    let big_m = 0.0_f64.max(max_lhs - rhs);
    add_program_row_with_canonical_terms(
        rows,
        name,
        expansions,
        coeffs,
        &[(active_binary, big_m)],
        RowSense::Le,
        rhs + big_m,
    );
    Ok(())
}

fn add_reified_equality_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    reified: &ReifiedLinearConstraint,
) -> Result<(), MathProgramError> {
    add_literal_implied_le(
        program,
        rows,
        expansions,
        format!("{}__reified_eq_true_le", reified.name),
        reified.literal,
        &reified.coeffs,
        reified.rhs,
    )?;
    let negated_coeffs = negate_sparse(&reified.coeffs);
    add_literal_implied_le(
        program,
        rows,
        expansions,
        format!("{}__reified_eq_true_ge", reified.name),
        reified.literal,
        &negated_coeffs,
        -reified.rhs,
    )?;

    let low = push_canonical_var(
        &format!("{}__eq_false_low", reified.name),
        true,
        1.0,
        names,
        integer_vars,
        ub,
    );
    let high = push_canonical_var(
        &format!("{}__eq_false_high", reified.name),
        true,
        1.0,
        names,
        integer_vars,
        ub,
    );
    let (literal_coef, rhs) = if reified.literal.value {
        (1.0, 1.0)
    } else {
        (-1.0, 0.0)
    };
    add_program_row_with_canonical_terms(
        rows,
        format!("{}__reified_eq_false_branch", reified.name),
        expansions,
        &[(reified.literal.var, literal_coef)],
        &[(low, 1.0), (high, 1.0)],
        RowSense::Eq,
        rhs,
    );
    add_canonical_binary_implied_le(
        program,
        rows,
        expansions,
        format!("{}__reified_eq_false_low", reified.name),
        low,
        &reified.coeffs,
        reified.rhs - 1.0,
    )?;
    add_canonical_binary_implied_le(
        program,
        rows,
        expansions,
        format!("{}__reified_eq_false_high", reified.name),
        high,
        &negated_coeffs,
        -reified.rhs - 1.0,
    )?;
    Ok(())
}

fn add_reified_linear_le(
    program: &MathProgram,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    reified: &ReifiedLinearConstraint,
    coeffs: &[(usize, f64)],
    rhs: f64,
) -> Result<(), MathProgramError> {
    let (min_lhs, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "reified linear constraint `{}` needs finite variable bounds for big-M lowering",
            reified.name
        ))
    })?;
    let true_big_m = 0.0_f64.max(max_lhs - rhs);
    let false_rhs = rhs + 1.0;
    let false_big_m = 0.0_f64.max(false_rhs - min_lhs);

    let mut true_coeffs = coeffs.to_vec();
    let mut true_rhs = rhs;
    let mut false_coeffs = coeffs.to_vec();
    let mut false_row_rhs = false_rhs;
    if reified.literal.value {
        true_coeffs.push((reified.literal.var, true_big_m));
        true_rhs += true_big_m;
        false_coeffs.push((reified.literal.var, false_big_m));
    } else {
        true_coeffs.push((reified.literal.var, -true_big_m));
        false_coeffs.push((reified.literal.var, -false_big_m));
        false_row_rhs -= false_big_m;
    }

    add_program_row(
        rows,
        format!("{}__reified_true", reified.name),
        expansions,
        &true_coeffs,
        RowSense::Le,
        true_rhs,
    );
    add_program_row(
        rows,
        format!("{}__reified_false", reified.name),
        expansions,
        &false_coeffs,
        RowSense::Ge,
        false_row_rhs,
    );
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
        GeneralConstraint::BinaryXor {
            name,
            result_var,
            operands,
        } => {
            let mut coeffs = operands.iter().map(|&idx| (idx, 1.0)).collect::<Vec<_>>();
            coeffs.push((*result_var, -1.0));
            if operands.len() == 1 {
                add_program_row(
                    rows,
                    format!("{name}__xor_parity"),
                    expansions,
                    &coeffs,
                    RowSense::Eq,
                    0.0,
                );
            } else {
                let quotient = push_canonical_var(
                    &format!("{name}__xor_quotient"),
                    true,
                    (operands.len() / 2) as f64,
                    names,
                    integer_vars,
                    ub,
                );
                add_mixed_row(
                    rows,
                    format!("{name}__xor_parity"),
                    expansions,
                    &coeffs,
                    &[(quotient, -2.0)],
                    RowSense::Eq,
                    0.0,
                );
            }
        }
        GeneralConstraint::BooleanAnd {
            name,
            result_var,
            literals,
        } => {
            for (pos, literal) in literals.iter().enumerate() {
                let (coeffs, rhs) = if literal.value {
                    (vec![(*result_var, 1.0), (literal.var, -1.0)], 0.0)
                } else {
                    (vec![(*result_var, 1.0), (literal.var, 1.0)], 1.0)
                };
                add_program_row(
                    rows,
                    format!("{name}__and_result_le_literal_{pos}"),
                    expansions,
                    &coeffs,
                    RowSense::Le,
                    rhs,
                );
            }
            let (mut coeffs, negated_count) = literal_cardinality_terms(literals);
            coeffs.push((*result_var, -1.0));
            add_program_row(
                rows,
                format!("{name}__and_result_ge_sum"),
                expansions,
                &coeffs,
                RowSense::Le,
                literals.len() as f64 - 1.0 - negated_count as f64,
            );
        }
        GeneralConstraint::BooleanOr {
            name,
            result_var,
            literals,
        } => {
            for (pos, literal) in literals.iter().enumerate() {
                let (coeffs, rhs) = if literal.value {
                    (vec![(literal.var, 1.0), (*result_var, -1.0)], 0.0)
                } else {
                    (vec![(literal.var, -1.0), (*result_var, -1.0)], -1.0)
                };
                add_program_row(
                    rows,
                    format!("{name}__or_result_ge_literal_{pos}"),
                    expansions,
                    &coeffs,
                    RowSense::Le,
                    rhs,
                );
            }
            let (literal_terms, negated_count) = literal_cardinality_terms(literals);
            let mut coeffs = vec![(*result_var, 1.0)];
            coeffs.extend(literal_terms.iter().map(|&(idx, coef)| (idx, -coef)));
            add_program_row(
                rows,
                format!("{name}__or_result_le_sum"),
                expansions,
                &coeffs,
                RowSense::Le,
                negated_count as f64,
            );
        }
        GeneralConstraint::BooleanXor {
            name,
            result_var,
            literals,
        } => {
            let (mut coeffs, negated_count) = literal_cardinality_terms(literals);
            coeffs.push((*result_var, -1.0));
            if literals.len() == 1 {
                add_program_row(
                    rows,
                    format!("{name}__xor_parity"),
                    expansions,
                    &coeffs,
                    RowSense::Eq,
                    -(negated_count as f64),
                );
            } else {
                let quotient = push_canonical_var(
                    &format!("{name}__xor_quotient"),
                    true,
                    (literals.len() / 2) as f64,
                    names,
                    integer_vars,
                    ub,
                );
                add_mixed_row(
                    rows,
                    format!("{name}__xor_parity"),
                    expansions,
                    &coeffs,
                    &[(quotient, -2.0)],
                    RowSense::Eq,
                    -(negated_count as f64),
                );
            }
        }
        GeneralConstraint::BinaryCardinality {
            name,
            operands,
            min_count,
            max_count,
        } => {
            let coeffs = operands.iter().map(|&idx| (idx, 1.0)).collect::<Vec<_>>();
            if let Some(max_count) = max_count {
                add_program_row(
                    rows,
                    format!("{name}__cardinality_at_most"),
                    expansions,
                    &coeffs,
                    RowSense::Le,
                    *max_count as f64,
                );
            }
            if let Some(min_count) = min_count {
                add_program_row(
                    rows,
                    format!("{name}__cardinality_at_least"),
                    expansions,
                    &coeffs,
                    RowSense::Ge,
                    *min_count as f64,
                );
            }
        }
        GeneralConstraint::BooleanCardinality {
            name,
            literals,
            min_count,
            max_count,
        } => {
            let (coeffs, negated_count) = literal_cardinality_terms(literals);
            if let Some(max_count) = max_count {
                add_program_row(
                    rows,
                    format!("{name}__literal_cardinality_at_most"),
                    expansions,
                    &coeffs,
                    RowSense::Le,
                    *max_count as f64 - negated_count as f64,
                );
            }
            if let Some(min_count) = min_count {
                add_program_row(
                    rows,
                    format!("{name}__literal_cardinality_at_least"),
                    expansions,
                    &coeffs,
                    RowSense::Ge,
                    *min_count as f64 - negated_count as f64,
                );
            }
        }
        GeneralConstraint::BooleanCount {
            name,
            literals,
            count_var,
        } => {
            let (literal_terms, negated_count) = literal_cardinality_terms(literals);
            let mut coeffs = vec![(*count_var, 1.0)];
            coeffs.extend(literal_terms.iter().map(|&(idx, coef)| (idx, -coef)));
            add_program_row(
                rows,
                format!("{name}__boolean_count"),
                expansions,
                &coeffs,
                RowSense::Eq,
                negated_count as f64,
            );
        }
        GeneralConstraint::BooleanClause { name, literals } => {
            let mut coeffs = Vec::with_capacity(literals.len());
            let mut negated_count = 0usize;
            for literal in literals {
                if literal.value {
                    coeffs.push((literal.var, 1.0));
                } else {
                    coeffs.push((literal.var, -1.0));
                    negated_count += 1;
                }
            }
            add_program_row(
                rows,
                format!("{name}__boolean_clause"),
                expansions,
                &coeffs,
                RowSense::Ge,
                1.0 - negated_count as f64,
            );
        }
        GeneralConstraint::IntegerProduct {
            name,
            target_var,
            operands,
        } => add_integer_product_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *target_var,
            operands,
        )?,
        GeneralConstraint::IntegerDivision {
            name,
            target_var,
            numerator_var,
            denominator_var,
        } => add_integer_binary_operation_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            "integer division",
            *target_var,
            *numerator_var,
            *denominator_var,
            i64::checked_div,
        )?,
        GeneralConstraint::IntegerModulo {
            name,
            target_var,
            numerator_var,
            denominator_var,
        } => add_integer_binary_operation_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            "integer modulo",
            *target_var,
            *numerator_var,
            *denominator_var,
            i64::checked_rem,
        )?,
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
        GeneralConstraint::Norm {
            name,
            result_var,
            operands,
            norm_type,
        } => add_norm_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *result_var,
            operands,
            *norm_type,
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
        GeneralConstraint::GlobalCardinality {
            name,
            variables,
            counts,
        } => add_global_cardinality_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            counts,
        )?,
        GeneralConstraint::ValueCount {
            name,
            variables,
            value,
            count_var,
        } => add_value_count_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            *value,
            *count_var,
        )?,
        GeneralConstraint::LinearDomain {
            name,
            coeffs,
            intervals,
        } => add_linear_domain_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            coeffs,
            intervals,
        )?,
        GeneralConstraint::EnforcedLinearDomain {
            name,
            enforcement,
            coeffs,
            intervals,
        } => add_enforced_linear_domain_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            enforcement,
            coeffs,
            intervals,
        )?,
        GeneralConstraint::MapDomain {
            name,
            var,
            bool_vars,
            offset,
        } => add_map_domain_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *var,
            bool_vars,
            *offset,
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
        GeneralConstraint::ForbiddenAssignments {
            name,
            variables,
            tuples,
        } => add_forbidden_assignment_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            tuples,
        )?,
        GeneralConstraint::EnforcedAllowedAssignments {
            name,
            enforcement,
            variables,
            tuples,
        } => add_enforced_table_assignment_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            enforcement,
            variables,
            tuples,
            false,
        )?,
        GeneralConstraint::EnforcedForbiddenAssignments {
            name,
            enforcement,
            variables,
            tuples,
        } => add_enforced_table_assignment_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            enforcement,
            variables,
            tuples,
            true,
        )?,
        GeneralConstraint::BinPacking {
            name,
            item_bin_vars,
            load_vars,
            item_sizes,
        } => add_bin_packing_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            item_bin_vars,
            load_vars,
            item_sizes,
        )?,
        GeneralConstraint::Element {
            name,
            index_var,
            target_var,
            values,
        } => add_element_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *index_var,
            *target_var,
            values,
        )?,
        GeneralConstraint::VariableElement {
            name,
            index_var,
            target_var,
            variables,
        } => add_variable_element_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *index_var,
            *target_var,
            variables,
        )?,
        GeneralConstraint::Inverse {
            name,
            variables,
            inverse_variables,
        } => add_inverse_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            inverse_variables,
        )?,
        GeneralConstraint::Circuit {
            name,
            node_count,
            arcs,
        } => add_circuit_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *node_count,
            arcs,
        )?,
        GeneralConstraint::MultipleCircuit {
            name,
            node_count,
            arcs,
        } => add_multiple_circuit_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            *node_count,
            arcs,
        )?,
        GeneralConstraint::Automaton {
            name,
            variables,
            starting_state,
            final_states,
            transitions,
        } => add_automaton_rows(
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            variables,
            *starting_state,
            final_states,
            transitions,
        )?,
        GeneralConstraint::Alternative {
            name,
            master,
            alternatives,
        } => add_alternative_rows(ub, rows, expansions, name, master, alternatives)?,
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
            capacity,
        )?,
        GeneralConstraint::Reservoir {
            name,
            events,
            min_level,
            max_level,
        } => add_reservoir_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            events,
            *min_level,
            *max_level,
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

fn add_global_cardinality_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    counts: &[GlobalCardinalityCount],
) -> Result<(), MathProgramError> {
    let counted_values = counts
        .iter()
        .map(|count| count.value)
        .collect::<BTreeSet<_>>();
    let mut value_literals = BTreeMap::<i64, Vec<usize>>::new();

    for &var_idx in variables {
        let var = &program.variables[var_idx];
        let (lower, upper) = integer_bounds(var).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "global-cardinality variable `{}` requires finite integer bounds",
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
            if counted_values.contains(&value) {
                value_literals.entry(value).or_default().push(lit);
            }
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
        let link_coeffs = combine_terms(&link_coeffs);
        rows.push(SparseRow {
            coeffs: link_coeffs.clone(),
            rhs: link_rhs,
            name: format!("{name}__{}__link_value", var.name),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&link_coeffs),
            rhs: -link_rhs,
            name: format!("{name}__{}__link_value_ge", var.name),
        });
    }

    for count in counts {
        let lits = value_literals
            .get(&count.value)
            .cloned()
            .unwrap_or_default();
        if let Some(max_count) = count.max_count {
            rows.push(SparseRow {
                coeffs: lits.iter().map(|&lit| (lit, 1.0)).collect(),
                rhs: max_count as f64,
                name: format!("{name}__value_{}__at_most", count.value),
            });
        }
        if let Some(min_count) = count.min_count {
            rows.push(SparseRow {
                coeffs: lits.iter().map(|&lit| (lit, -1.0)).collect(),
                rhs: -(min_count as f64),
                name: format!("{name}__value_{}__at_least", count.value),
            });
        }
    }

    Ok(())
}

fn add_value_count_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    value: i64,
    count_var: usize,
) -> Result<(), MathProgramError> {
    let mut counted_value_lits = Vec::new();

    for &var_idx in variables {
        let var = &program.variables[var_idx];
        let (lower, upper) = integer_bounds(var).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "value-count variable `{}` requires finite integer bounds",
                var.name
            ))
        })?;
        let mut literals = Vec::new();
        for candidate in lower..=upper {
            let lit = push_canonical_var(
                &format!("{name}__{}__eq_{candidate}", var.name),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            literals.push((candidate, lit));
            if candidate == value {
                counted_value_lits.push(lit);
            }
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
        link_coeffs.extend(
            literals
                .iter()
                .map(|&(candidate, lit)| (lit, -(candidate as f64))),
        );
        let link_rhs = -expansions[var_idx].constant;
        let link_coeffs = combine_terms(&link_coeffs);
        rows.push(SparseRow {
            coeffs: link_coeffs.clone(),
            rhs: link_rhs,
            name: format!("{name}__{}__link_value", var.name),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&link_coeffs),
            rhs: -link_rhs,
            name: format!("{name}__{}__link_value_ge", var.name),
        });
    }

    let mut count_link_coeffs = expansions[count_var].terms.clone();
    count_link_coeffs.extend(counted_value_lits.iter().map(|&lit| (lit, -1.0)));
    let count_link_rhs = -expansions[count_var].constant;
    let count_link_coeffs = combine_terms(&count_link_coeffs);
    rows.push(SparseRow {
        coeffs: count_link_coeffs.clone(),
        rhs: count_link_rhs,
        name: format!("{name}__count_link"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&count_link_coeffs),
        rhs: -count_link_rhs,
        name: format!("{name}__count_link_ge"),
    });

    Ok(())
}

fn add_linear_domain_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    coeffs: &[(usize, f64)],
    intervals: &[LinearDomainInterval],
) -> Result<(), MathProgramError> {
    let (min_lhs, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "linear-domain constraint `{name}` needs finite variable bounds for exact MIP lowering"
        ))
    })?;
    let selectors = intervals
        .iter()
        .enumerate()
        .map(|(idx, interval)| {
            push_canonical_var(
                &format!("{name}__domain_{}_{}_{idx}", interval.lower, interval.upper),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();
    let selector_sum = selectors
        .iter()
        .map(|&selector| (selector, 1.0))
        .collect::<Vec<_>>();
    add_program_row_with_canonical_terms(
        rows,
        format!("{name}__domain_select_one"),
        expansions,
        &[],
        &selector_sum,
        RowSense::Eq,
        1.0,
    );

    for (idx, (interval, &selector)) in intervals.iter().zip(&selectors).enumerate() {
        let lower = interval.lower as f64;
        let upper = interval.upper as f64;
        let upper_m = 0.0_f64.max(max_lhs - upper);
        add_program_row_with_canonical_terms(
            rows,
            format!("{name}__domain_{idx}_upper"),
            expansions,
            coeffs,
            &[(selector, upper_m)],
            RowSense::Le,
            upper + upper_m,
        );
        let lower_m = 0.0_f64.max(lower - min_lhs);
        add_program_row_with_canonical_terms(
            rows,
            format!("{name}__domain_{idx}_lower"),
            expansions,
            coeffs,
            &[(selector, -lower_m)],
            RowSense::Ge,
            lower - lower_m,
        );
    }
    Ok(())
}

fn add_enforced_linear_domain_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    enforcement: &[BoolLiteral],
    coeffs: &[(usize, f64)],
    intervals: &[LinearDomainInterval],
) -> Result<(), MathProgramError> {
    let (min_lhs, max_lhs) = linear_bounds(program, coeffs).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "enforced linear-domain constraint `{name}` needs finite variable bounds for exact MIP lowering"
        ))
    })?;
    let selectors = intervals
        .iter()
        .enumerate()
        .map(|(idx, interval)| {
            push_canonical_var(
                &format!(
                    "{name}__enforced_domain_{}_{}_{idx}",
                    interval.lower, interval.upper
                ),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            )
        })
        .collect::<Vec<_>>();
    let selector_sum = selectors
        .iter()
        .map(|&selector| (selector, 1.0))
        .collect::<Vec<_>>();
    add_program_row_with_canonical_terms(
        rows,
        format!("{name}__domain_select_at_most_one"),
        expansions,
        &[],
        &selector_sum,
        RowSense::Le,
        1.0,
    );

    let positive_literal_count = enforcement.iter().filter(|literal| literal.value).count() as f64;
    let activation_coeffs = enforcement
        .iter()
        .map(|literal| {
            let coeff = if literal.value { -1.0 } else { 1.0 };
            (literal.var, coeff)
        })
        .collect::<Vec<_>>();
    add_program_row_with_canonical_terms(
        rows,
        format!("{name}__domain_select_if_enforced"),
        expansions,
        &activation_coeffs,
        &selector_sum,
        RowSense::Ge,
        1.0 - positive_literal_count,
    );

    for (idx, (interval, &selector)) in intervals.iter().zip(&selectors).enumerate() {
        let lower = interval.lower as f64;
        let upper = interval.upper as f64;
        let upper_m = 0.0_f64.max(max_lhs - upper);
        add_program_row_with_canonical_terms(
            rows,
            format!("{name}__enforced_domain_{idx}_upper"),
            expansions,
            coeffs,
            &[(selector, upper_m)],
            RowSense::Le,
            upper + upper_m,
        );
        let lower_m = 0.0_f64.max(lower - min_lhs);
        add_program_row_with_canonical_terms(
            rows,
            format!("{name}__enforced_domain_{idx}_lower"),
            expansions,
            coeffs,
            &[(selector, -lower_m)],
            RowSense::Ge,
            lower - lower_m,
        );
    }
    Ok(())
}

fn add_map_domain_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    var: usize,
    bool_vars: &[usize],
    offset: i64,
) -> Result<(), MathProgramError> {
    let (variables, tuples) = map_domain_variables_and_tuples(program, var, bool_vars, offset)?;
    add_allowed_assignment_rows(
        names,
        integer_vars,
        ub,
        rows,
        expansions,
        name,
        &variables,
        &tuples,
    )
}

fn map_domain_variables_and_tuples(
    program: &MathProgram,
    var: usize,
    bool_vars: &[usize],
    offset: i64,
) -> Result<(Vec<usize>, Vec<Vec<i64>>), MathProgramError> {
    let (lower, upper) = integer_bounds(&program.variables[var]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "map-domain variable `{}` requires finite integer bounds",
            program.variables[var].name
        ))
    })?;
    let targets = (0..bool_vars.len())
        .map(|pos| {
            offset.checked_add(pos as i64).ok_or_else(|| {
                MathProgramError::Unsupported("map-domain target value overflowed".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut tuples = Vec::new();
    for value in lower..=upper {
        let mut tuple = Vec::with_capacity(1 + bool_vars.len());
        tuple.push(value);
        tuple.extend(
            targets
                .iter()
                .map(|&target| if value == target { 1 } else { 0 }),
        );
        tuples.push(tuple);
    }

    let mut variables = Vec::with_capacity(1 + bool_vars.len());
    variables.push(var);
    variables.extend_from_slice(bool_vars);
    Ok((variables, tuples))
}

fn add_integer_product_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    target_var: usize,
    operands: &[usize],
) -> Result<(), MathProgramError> {
    let (variables, tuples) = integer_product_variables_and_tuples(program, target_var, operands)?;
    add_allowed_assignment_rows(
        names,
        integer_vars,
        ub,
        rows,
        expansions,
        name,
        &variables,
        &tuples,
    )
}

fn integer_product_variables_and_tuples(
    program: &MathProgram,
    target_var: usize,
    operands: &[usize],
) -> Result<(Vec<usize>, Vec<Vec<i64>>), MathProgramError> {
    if operands.is_empty() {
        return Err(MathProgramError::Unsupported(
            "integer product requires at least one operand".to_string(),
        ));
    }
    if target_var >= program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "integer product target variable index {target_var} out of bounds"
        )));
    }
    if !matches!(
        program.variables[target_var].var_type,
        VariableType::Binary | VariableType::Integer
    ) {
        return Err(MathProgramError::Unsupported(format!(
            "integer product target `{}` must be binary or integer",
            program.variables[target_var].name
        )));
    }
    let (target_lb, target_ub) =
        integer_bounds(&program.variables[target_var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "integer product target `{}` requires finite integer bounds",
                program.variables[target_var].name
            ))
        })?;

    let mut unique_operands = Vec::new();
    let mut seen = BTreeMap::<usize, ()>::new();
    for &operand in operands {
        if operand >= program.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "integer product operand index {operand} out of bounds"
            )));
        }
        if operand == target_var {
            return Err(MathProgramError::Unsupported(format!(
                "integer product target `{}` must be distinct from its operands",
                program.variables[target_var].name
            )));
        }
        if !matches!(
            program.variables[operand].var_type,
            VariableType::Binary | VariableType::Integer
        ) {
            return Err(MathProgramError::Unsupported(format!(
                "integer product operand `{}` must be binary or integer",
                program.variables[operand].name
            )));
        }
        integer_bounds(&program.variables[operand]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "integer product operand `{}` requires finite integer bounds",
                program.variables[operand].name
            ))
        })?;
        if seen.insert(operand, ()).is_none() {
            unique_operands.push(operand);
        }
    }

    let mut tuple_count = 1usize;
    let mut domains = Vec::with_capacity(unique_operands.len());
    for &operand in &unique_operands {
        let (lower, upper) = integer_bounds(&program.variables[operand]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "integer product operand `{}` requires finite integer bounds",
                program.variables[operand].name
            ))
        })?;
        let domain_size = upper
            .checked_sub(lower)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                MathProgramError::Unsupported(format!(
                    "integer product operand `{}` has an oversized domain",
                    program.variables[operand].name
                ))
            })? as usize;
        tuple_count = tuple_count.checked_mul(domain_size).ok_or_else(|| {
            MathProgramError::Unsupported("integer product tuple count overflowed".to_string())
        })?;
        if tuple_count > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "integer product exact MIP lowering is limited to 512 operand assignments, got {tuple_count}"
            )));
        }
        domains.push((operand, lower, upper));
    }

    let mut tuples = Vec::new();
    let mut assignment = BTreeMap::<usize, i64>::new();
    enumerate_integer_product_tuples(
        &domains,
        0,
        operands,
        target_lb,
        target_ub,
        &mut assignment,
        &mut tuples,
    )?;

    let mut variables = unique_operands;
    variables.push(target_var);
    Ok((variables, tuples))
}

fn enumerate_integer_product_tuples(
    domains: &[(usize, i64, i64)],
    pos: usize,
    operands: &[usize],
    target_lb: i64,
    target_ub: i64,
    assignment: &mut BTreeMap<usize, i64>,
    tuples: &mut Vec<Vec<i64>>,
) -> Result<(), MathProgramError> {
    if pos == domains.len() {
        let mut product = 1i64;
        for &operand in operands {
            let value = assignment[&operand];
            product = product.checked_mul(value).ok_or_else(|| {
                MathProgramError::Unsupported(
                    "integer product value overflowed during exact lowering".to_string(),
                )
            })?;
        }
        if product >= target_lb && product <= target_ub {
            let mut tuple = domains
                .iter()
                .map(|(operand, _, _)| assignment[operand])
                .collect::<Vec<_>>();
            tuple.push(product);
            tuples.push(tuple);
        }
        return Ok(());
    }

    let (operand, lower, upper) = domains[pos];
    for value in lower..=upper {
        assignment.insert(operand, value);
        enumerate_integer_product_tuples(
            domains,
            pos + 1,
            operands,
            target_lb,
            target_ub,
            assignment,
            tuples,
        )?;
    }
    assignment.remove(&operand);
    Ok(())
}

fn add_integer_binary_operation_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    kind: &str,
    target_var: usize,
    numerator_var: usize,
    denominator_var: usize,
    operation: fn(i64, i64) -> Option<i64>,
) -> Result<(), MathProgramError> {
    let (variables, tuples) = integer_binary_operation_variables_and_tuples(
        program,
        kind,
        target_var,
        numerator_var,
        denominator_var,
        operation,
    )?;
    add_allowed_assignment_rows(
        names,
        integer_vars,
        ub,
        rows,
        expansions,
        name,
        &variables,
        &tuples,
    )
}

fn integer_binary_operation_variables_and_tuples(
    program: &MathProgram,
    kind: &str,
    target_var: usize,
    numerator_var: usize,
    denominator_var: usize,
    operation: fn(i64, i64) -> Option<i64>,
) -> Result<(Vec<usize>, Vec<Vec<i64>>), MathProgramError> {
    if target_var >= program.variables.len() {
        return Err(MathProgramError::BadIndex(format!(
            "{kind} target variable index {target_var} out of bounds"
        )));
    }
    if !matches!(
        program.variables[target_var].var_type,
        VariableType::Binary | VariableType::Integer
    ) {
        return Err(MathProgramError::Unsupported(format!(
            "{kind} target `{}` must be binary or integer",
            program.variables[target_var].name
        )));
    }
    let (target_lb, target_ub) =
        integer_bounds(&program.variables[target_var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "{kind} target `{}` requires finite integer bounds",
                program.variables[target_var].name
            ))
        })?;

    let mut unique_operands = Vec::new();
    let mut seen = BTreeMap::<usize, ()>::new();
    for &(role, operand) in &[
        ("numerator", numerator_var),
        ("denominator", denominator_var),
    ] {
        if operand >= program.variables.len() {
            return Err(MathProgramError::BadIndex(format!(
                "{kind} {role} variable index {operand} out of bounds"
            )));
        }
        if operand == target_var {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} target `{}` must be distinct from its operands",
                program.variables[target_var].name
            )));
        }
        if !matches!(
            program.variables[operand].var_type,
            VariableType::Binary | VariableType::Integer
        ) {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} {role} `{}` must be binary or integer",
                program.variables[operand].name
            )));
        }
        integer_bounds(&program.variables[operand]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "{kind} {role} `{}` requires finite integer bounds",
                program.variables[operand].name
            ))
        })?;
        if seen.insert(operand, ()).is_none() {
            unique_operands.push(operand);
        }
    }

    let mut tuple_count = 1usize;
    let mut domains = Vec::with_capacity(unique_operands.len());
    for &operand in &unique_operands {
        let (lower, upper) = integer_bounds(&program.variables[operand]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "{kind} operand `{}` requires finite integer bounds",
                program.variables[operand].name
            ))
        })?;
        let domain_size = upper
            .checked_sub(lower)
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| {
                MathProgramError::Unsupported(format!(
                    "{kind} operand `{}` has an oversized domain",
                    program.variables[operand].name
                ))
            })? as usize;
        tuple_count = tuple_count.checked_mul(domain_size).ok_or_else(|| {
            MathProgramError::Unsupported(format!("{kind} tuple count overflowed"))
        })?;
        if tuple_count > 512 {
            return Err(MathProgramError::Unsupported(format!(
                "{kind} exact MIP lowering is limited to 512 operand assignments, got {tuple_count}"
            )));
        }
        domains.push((operand, lower, upper));
    }

    let mut tuples = Vec::new();
    let mut assignment = BTreeMap::<usize, i64>::new();
    enumerate_integer_binary_operation_tuples(
        &domains,
        0,
        numerator_var,
        denominator_var,
        target_lb,
        target_ub,
        &mut assignment,
        &mut tuples,
        operation,
    );

    let mut variables = unique_operands;
    variables.push(target_var);
    Ok((variables, tuples))
}

fn enumerate_integer_binary_operation_tuples(
    domains: &[(usize, i64, i64)],
    pos: usize,
    numerator_var: usize,
    denominator_var: usize,
    target_lb: i64,
    target_ub: i64,
    assignment: &mut BTreeMap<usize, i64>,
    tuples: &mut Vec<Vec<i64>>,
    operation: fn(i64, i64) -> Option<i64>,
) {
    if pos == domains.len() {
        if let Some(result) = operation(assignment[&numerator_var], assignment[&denominator_var]) {
            if result >= target_lb && result <= target_ub {
                let mut tuple = domains
                    .iter()
                    .map(|(operand, _, _)| assignment[operand])
                    .collect::<Vec<_>>();
                tuple.push(result);
                tuples.push(tuple);
            }
        }
        return;
    }

    let (operand, lower, upper) = domains[pos];
    for value in lower..=upper {
        assignment.insert(operand, value);
        enumerate_integer_binary_operation_tuples(
            domains,
            pos + 1,
            numerator_var,
            denominator_var,
            target_lb,
            target_ub,
            assignment,
            tuples,
            operation,
        );
    }
    assignment.remove(&operand);
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

fn add_enforced_table_assignment_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    enforcement: &[BoolLiteral],
    variables: &[usize],
    tuples: &[Vec<i64>],
    forbidden: bool,
) -> Result<(), MathProgramError> {
    let (variables, expanded_tuples) =
        enforced_table_variables_and_tuples(program, enforcement, variables, tuples, forbidden)?;
    if expanded_tuples.is_empty() {
        rows.push(SparseRow {
            coeffs: Vec::new(),
            rhs: -1.0,
            name: format!("{name}__infeasible_enforced_table"),
        });
        return Ok(());
    }
    add_allowed_assignment_rows(
        names,
        integer_vars,
        ub,
        rows,
        expansions,
        name,
        &variables,
        &expanded_tuples,
    )
}

fn enforced_table_variables_and_tuples(
    program: &MathProgram,
    enforcement: &[BoolLiteral],
    variables: &[usize],
    tuples: &[Vec<i64>],
    forbidden: bool,
) -> Result<(Vec<usize>, Vec<Vec<i64>>), MathProgramError> {
    let mut enforcement_vars = Vec::new();
    let mut seen_enforcement = BTreeSet::new();
    for literal in enforcement {
        if seen_enforcement.insert(literal.var) {
            enforcement_vars.push(literal.var);
        }
    }

    let mut domains = Vec::with_capacity(variables.len());
    for &var in variables {
        let (lower, upper) = integer_bounds(&program.variables[var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "enforced table variable `{}` requires finite integer bounds",
                program.variables[var].name
            ))
        })?;
        domains.push((lower, upper));
    }

    let table = tuples.iter().cloned().collect::<BTreeSet<_>>();
    let mut expanded = Vec::new();
    let mut variable_values = Vec::with_capacity(variables.len());
    enumerate_enforced_table_variable_values(
        0,
        &domains,
        &mut variable_values,
        &table,
        enforcement,
        &enforcement_vars,
        forbidden,
        &mut expanded,
    )?;

    let mut expanded_vars = variables.to_vec();
    expanded_vars.extend(enforcement_vars);
    Ok((expanded_vars, expanded))
}

fn enumerate_enforced_table_variable_values(
    pos: usize,
    domains: &[(i64, i64)],
    variable_values: &mut Vec<i64>,
    table: &BTreeSet<Vec<i64>>,
    enforcement: &[BoolLiteral],
    enforcement_vars: &[usize],
    forbidden: bool,
    expanded: &mut Vec<Vec<i64>>,
) -> Result<(), MathProgramError> {
    if pos == domains.len() {
        let mut enforcement_values = Vec::with_capacity(enforcement_vars.len());
        enumerate_enforced_table_enforcement_values(
            0,
            variable_values,
            table,
            enforcement,
            enforcement_vars,
            &mut enforcement_values,
            forbidden,
            expanded,
        )?;
        return Ok(());
    }

    let (lower, upper) = domains[pos];
    for value in lower..=upper {
        variable_values.push(value);
        enumerate_enforced_table_variable_values(
            pos + 1,
            domains,
            variable_values,
            table,
            enforcement,
            enforcement_vars,
            forbidden,
            expanded,
        )?;
        variable_values.pop();
    }
    Ok(())
}

fn enumerate_enforced_table_enforcement_values(
    pos: usize,
    variable_values: &[i64],
    table: &BTreeSet<Vec<i64>>,
    enforcement: &[BoolLiteral],
    enforcement_vars: &[usize],
    enforcement_values: &mut Vec<i64>,
    forbidden: bool,
    expanded: &mut Vec<Vec<i64>>,
) -> Result<(), MathProgramError> {
    if pos == enforcement_vars.len() {
        let active = enforcement.iter().all(|literal| {
            let idx = enforcement_vars
                .iter()
                .position(|&var| var == literal.var)
                .expect("literal var listed");
            (enforcement_values[idx] == 1) == literal.value
        });
        let listed = table.contains(variable_values);
        let allowed = if forbidden {
            !(active && listed)
        } else {
            !active || listed
        };
        if allowed {
            if expanded.len() >= 512 {
                return Err(MathProgramError::Unsupported(
                    "enforced table exact MIP lowering is limited to 512 expanded tuples"
                        .to_string(),
                ));
            }
            let mut tuple = variable_values.to_vec();
            tuple.extend_from_slice(enforcement_values);
            expanded.push(tuple);
        }
        return Ok(());
    }

    for value in [0_i64, 1_i64] {
        enforcement_values.push(value);
        enumerate_enforced_table_enforcement_values(
            pos + 1,
            variable_values,
            table,
            enforcement,
            enforcement_vars,
            enforcement_values,
            forbidden,
            expanded,
        )?;
        enforcement_values.pop();
    }
    Ok(())
}

fn add_forbidden_assignment_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    tuples: &[Vec<i64>],
) -> Result<(), MathProgramError> {
    let mut value_literals_by_var = Vec::with_capacity(variables.len());

    for (col, &var_idx) in variables.iter().enumerate() {
        let var = &program.variables[var_idx];
        let (lower, upper) = integer_bounds(var).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "forbidden-assignments variable `{}` requires finite integer bounds",
                var.name
            ))
        })?;
        let mut literals = BTreeMap::new();
        for value in lower..=upper {
            let lit = push_canonical_var(
                &format!("{name}__var_{col}__eq_{value}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            literals.insert(value, lit);
        }

        let choose_coeffs = literals.values().map(|&lit| (lit, 1.0)).collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__var_{col}__choose_one_value"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__var_{col}__choose_one_value_ge"),
        });

        let mut link_coeffs = expansions[var_idx].terms.clone();
        link_coeffs.extend(literals.iter().map(|(&value, &lit)| (lit, -(value as f64))));
        let link_rhs = -expansions[var_idx].constant;
        let link_coeffs = combine_terms(&link_coeffs);
        rows.push(SparseRow {
            coeffs: link_coeffs.clone(),
            rhs: link_rhs,
            name: format!("{name}__var_{col}__link_value"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&link_coeffs),
            rhs: -link_rhs,
            name: format!("{name}__var_{col}__link_value_ge"),
        });

        value_literals_by_var.push(literals);
    }

    for (row, tuple) in tuples.iter().enumerate() {
        let coeffs = tuple
            .iter()
            .enumerate()
            .map(|(col, &value)| (value_literals_by_var[col][&value], 1.0))
            .collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs,
            rhs: variables.len().saturating_sub(1) as f64,
            name: format!("{name}__forbid_tuple_{row}"),
        });
    }

    Ok(())
}

fn add_inverse_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    inverse_variables: &[usize],
) -> Result<(), MathProgramError> {
    if variables.len() != inverse_variables.len() {
        return Err(MathProgramError::Unsupported(format!(
            "inverse requires equally-sized variable arrays, got {} and {}",
            variables.len(),
            inverse_variables.len()
        )));
    }
    let n = variables.len();
    let mut literals = Vec::with_capacity(n);
    for (i, &var_idx) in variables.iter().enumerate() {
        let mut row_literals = Vec::with_capacity(n);
        for j in 0..n {
            row_literals.push(push_canonical_var(
                &format!("{name}__x_{i}_eq_{j}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            ));
        }
        let choose_coeffs = row_literals
            .iter()
            .map(|&lit| (lit, 1.0))
            .collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__x_{i}__choose_one"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__x_{i}__choose_one_ge"),
        });

        let value_coeffs = row_literals
            .iter()
            .enumerate()
            .map(|(j, &lit)| (lit, -(j as f64)))
            .collect::<Vec<_>>();
        add_mixed_row(
            rows,
            format!("{name}__x_{i}__link_value"),
            expansions,
            &[(var_idx, 1.0)],
            &value_coeffs,
            RowSense::Eq,
            0.0,
        );
        literals.push(row_literals);
    }

    for (j, &inverse_var_idx) in inverse_variables.iter().enumerate() {
        let choose_coeffs = (0..n).map(|i| (literals[i][j], 1.0)).collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__inverse_{j}__choose_one"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__inverse_{j}__choose_one_ge"),
        });

        let value_coeffs = (0..n)
            .map(|i| (literals[i][j], -(i as f64)))
            .collect::<Vec<_>>();
        add_mixed_row(
            rows,
            format!("{name}__inverse_{j}__link_value"),
            expansions,
            &[(inverse_var_idx, 1.0)],
            &value_coeffs,
            RowSense::Eq,
            0.0,
        );
    }

    Ok(())
}

fn add_circuit_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    node_count: usize,
    arcs: &[CircuitArc],
) -> Result<(), MathProgramError> {
    if node_count < 2 {
        return Err(MathProgramError::Unsupported(
            "circuit requires at least two nodes".to_string(),
        ));
    }

    let mut outgoing = vec![Vec::<(usize, f64)>::new(); node_count];
    let mut incoming = vec![Vec::<(usize, f64)>::new(); node_count];
    for arc in arcs {
        if arc.tail >= node_count || arc.head >= node_count {
            return Err(MathProgramError::BadIndex(format!(
                "circuit arc {} -> {} is outside node range [0, {})",
                arc.tail, arc.head, node_count
            )));
        }
        outgoing[arc.tail].push((arc.literal_var, 1.0));
        incoming[arc.head].push((arc.literal_var, 1.0));
    }

    for node in 0..node_count {
        if outgoing[node].is_empty() || incoming[node].is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "circuit node {node} requires at least one incoming and outgoing arc"
            )));
        }
        add_program_row(
            rows,
            format!("{name}__node_{node}__out_degree"),
            expansions,
            &outgoing[node],
            RowSense::Eq,
            1.0,
        );
        add_program_row(
            rows,
            format!("{name}__node_{node}__in_degree"),
            expansions,
            &incoming[node],
            RowSense::Eq,
            1.0,
        );
    }

    let order_upper = node_count.saturating_sub(2) as f64;
    let mut order_vars = vec![None; node_count];
    for (node, slot) in order_vars.iter_mut().enumerate().skip(1) {
        *slot = Some(push_canonical_var(
            &format!("{name}__node_{node}__order"),
            true,
            order_upper,
            names,
            integer_vars,
            ub,
        ));
    }

    let big_m = (node_count - 1) as f64;
    let rhs = (node_count - 2) as f64;
    for arc in arcs {
        if arc.tail == 0 || arc.head == 0 {
            continue;
        }
        let tail_order = order_vars[arc.tail].expect("non-depot node has order variable");
        let head_order = order_vars[arc.head].expect("non-depot node has order variable");
        add_mixed_row(
            rows,
            format!("{name}__mtz_{}_{}", arc.tail, arc.head),
            expansions,
            &[(arc.literal_var, big_m)],
            &[(tail_order, 1.0), (head_order, -1.0)],
            RowSense::Le,
            rhs,
        );
    }

    Ok(())
}

fn add_multiple_circuit_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    node_count: usize,
    arcs: &[CircuitArc],
) -> Result<(), MathProgramError> {
    if node_count < 2 {
        return Err(MathProgramError::Unsupported(
            "multiple-circuit requires at least two nodes".to_string(),
        ));
    }

    let mut outgoing = vec![Vec::<(usize, f64)>::new(); node_count];
    let mut incoming = vec![Vec::<(usize, f64)>::new(); node_count];
    for arc in arcs {
        if arc.tail >= node_count || arc.head >= node_count {
            return Err(MathProgramError::BadIndex(format!(
                "multiple-circuit arc {} -> {} is outside node range [0, {})",
                arc.tail, arc.head, node_count
            )));
        }
        outgoing[arc.tail].push((arc.literal_var, 1.0));
        incoming[arc.head].push((arc.literal_var, 1.0));
    }

    for node in 1..node_count {
        if outgoing[node].is_empty() || incoming[node].is_empty() {
            return Err(MathProgramError::Unsupported(format!(
                "multiple-circuit node {node} requires at least one incoming and outgoing arc"
            )));
        }
        add_program_row(
            rows,
            format!("{name}__node_{node}__out_degree"),
            expansions,
            &outgoing[node],
            RowSense::Eq,
            1.0,
        );
        add_program_row(
            rows,
            format!("{name}__node_{node}__in_degree"),
            expansions,
            &incoming[node],
            RowSense::Eq,
            1.0,
        );
    }

    let mut depot_balance = outgoing[0].clone();
    depot_balance.extend(incoming[0].iter().map(|&(idx, coeff)| (idx, -coeff)));
    if !depot_balance.is_empty() {
        add_program_row(
            rows,
            format!("{name}__depot_balance"),
            expansions,
            &depot_balance,
            RowSense::Eq,
            0.0,
        );
    }

    let order_upper = node_count.saturating_sub(1) as f64;
    let mut order_vars = vec![None; node_count];
    for (node, slot) in order_vars.iter_mut().enumerate().skip(1) {
        *slot = Some(push_canonical_var(
            &format!("{name}__node_{node}__route_order"),
            true,
            order_upper,
            names,
            integer_vars,
            ub,
        ));
    }

    let big_m = node_count as f64;
    let rhs = (node_count - 1) as f64;
    for arc in arcs {
        if arc.tail == 0 || arc.head == 0 || arc.tail == arc.head {
            continue;
        }
        let tail_order = order_vars[arc.tail].expect("non-depot node has order variable");
        let head_order = order_vars[arc.head].expect("non-depot node has order variable");
        add_mixed_row(
            rows,
            format!("{name}__route_mtz_{}_{}", arc.tail, arc.head),
            expansions,
            &[(arc.literal_var, big_m)],
            &[(tail_order, 1.0), (head_order, -1.0)],
            RowSense::Le,
            rhs,
        );
    }

    Ok(())
}

fn add_norm_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    result_var: usize,
    operands: &[usize],
    norm_type: NormType,
) -> Result<(), MathProgramError> {
    if norm_type == NormType::L0 {
        return add_l0_norm_rows(
            program,
            names,
            integer_vars,
            ub,
            rows,
            expansions,
            name,
            result_var,
            operands,
        );
    }

    let result_bounds = variable_bounds(&program.variables[result_var]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "norm result `{}` requires finite bounds",
            program.variables[result_var].name
        ))
    })?;
    let abs_vars = operands
        .iter()
        .enumerate()
        .map(|(k, &operand)| {
            let (lower, upper) = variable_bounds(&program.variables[operand]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "norm operand `{}` requires finite bounds",
                    program.variables[operand].name
                ))
            })?;
            let abs_upper = lower.abs().max(upper.abs());
            let abs_integer = matches!(
                program.variables[operand].var_type,
                VariableType::Binary | VariableType::Integer | VariableType::SemiInteger
            );
            let abs_var = push_canonical_var(
                &format!("{name}__abs_operand_{k}"),
                abs_integer,
                abs_upper,
                names,
                integer_vars,
                ub,
            );
            add_abs_canonical_rows(
                program,
                names,
                integer_vars,
                ub,
                rows,
                expansions,
                &format!("{name}__abs_operand_{k}"),
                abs_var,
                operand,
            )?;
            Ok((abs_var, abs_upper))
        })
        .collect::<Result<Vec<_>, MathProgramError>>()?;

    match norm_type {
        NormType::L0 => unreachable!("l0 norm lowering returned before absolute-value rows"),
        NormType::L1 => {
            let abs_terms = abs_vars
                .iter()
                .map(|&(abs_var, _)| (abs_var, -1.0))
                .collect::<Vec<_>>();
            add_mixed_row(
                rows,
                format!("{name}__l1_sum"),
                expansions,
                &[(result_var, 1.0)],
                &abs_terms,
                RowSense::Eq,
                0.0,
            );
        }
        NormType::LInfinity => {
            let selectors = abs_vars
                .iter()
                .enumerate()
                .map(|(k, _)| {
                    push_canonical_var(
                        &format!("{name}__linf_choice_{k}"),
                        true,
                        1.0,
                        names,
                        integer_vars,
                        ub,
                    )
                })
                .collect::<Vec<_>>();
            let choose_coeffs = selectors.iter().map(|&z| (z, 1.0)).collect::<Vec<_>>();
            rows.push(SparseRow {
                coeffs: choose_coeffs.clone(),
                rhs: 1.0,
                name: format!("{name}__linf_choose_one"),
            });
            rows.push(SparseRow {
                coeffs: negate_sparse(&choose_coeffs),
                rhs: -1.0,
                name: format!("{name}__linf_choose_one_ge"),
            });
            for (k, &(abs_var, _)) in abs_vars.iter().enumerate() {
                add_mixed_row(
                    rows,
                    format!("{name}__linf_ge_{k}"),
                    expansions,
                    &[(result_var, -1.0)],
                    &[(abs_var, 1.0)],
                    RowSense::Le,
                    0.0,
                );
                let big_m = 0.0_f64.max(result_bounds.1);
                add_mixed_row(
                    rows,
                    format!("{name}__linf_select_{k}"),
                    expansions,
                    &[(result_var, 1.0)],
                    &[(abs_var, -1.0), (selectors[k], big_m)],
                    RowSense::Le,
                    big_m,
                );
            }
        }
    }
    Ok(())
}

fn add_l0_norm_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    result_var: usize,
    operands: &[usize],
) -> Result<(), MathProgramError> {
    let mut nonzero_literals = Vec::new();

    for (k, &operand) in operands.iter().enumerate() {
        let var = &program.variables[operand];
        let (lower, upper) = integer_bounds(var).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "l0-norm operand `{}` requires finite integer bounds",
                var.name
            ))
        })?;
        let mut literals = Vec::new();
        for value in lower..=upper {
            let lit = push_canonical_var(
                &format!("{name}__operand_{k}__eq_{value}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            literals.push((value, lit));
            if value != 0 {
                nonzero_literals.push(lit);
            }
        }

        let choose_coeffs = literals
            .iter()
            .map(|&(_, lit)| (lit, 1.0))
            .collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__operand_{k}__choose_one"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__operand_{k}__choose_one_ge"),
        });

        let mut link_coeffs = expansions[operand].terms.clone();
        link_coeffs.extend(literals.iter().map(|&(value, lit)| (lit, -(value as f64))));
        let link_rhs = -expansions[operand].constant;
        let link_coeffs = combine_terms(&link_coeffs);
        rows.push(SparseRow {
            coeffs: link_coeffs.clone(),
            rhs: link_rhs,
            name: format!("{name}__operand_{k}__link_value"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&link_coeffs),
            rhs: -link_rhs,
            name: format!("{name}__operand_{k}__link_value_ge"),
        });
    }

    let mut count_link_coeffs = expansions[result_var].terms.clone();
    count_link_coeffs.extend(nonzero_literals.iter().map(|&lit| (lit, -1.0)));
    let count_link_rhs = -expansions[result_var].constant;
    let count_link_coeffs = combine_terms(&count_link_coeffs);
    rows.push(SparseRow {
        coeffs: count_link_coeffs.clone(),
        rhs: count_link_rhs,
        name: format!("{name}__count_link"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&count_link_coeffs),
        rhs: -count_link_rhs,
        name: format!("{name}__count_link_ge"),
    });

    Ok(())
}

fn add_abs_canonical_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    abs_var: usize,
    operand_var: usize,
) -> Result<(), MathProgramError> {
    let (lower, upper) = variable_bounds(&program.variables[operand_var]).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "abs `{name}` operand `{}` requires finite bounds",
            program.variables[operand_var].name
        ))
    })?;
    if lower >= 0.0 {
        add_mixed_row(
            rows,
            format!("{name}__abs_nonnegative"),
            expansions,
            &[(operand_var, -1.0)],
            &[(abs_var, 1.0)],
            RowSense::Eq,
            0.0,
        );
        return Ok(());
    }
    if upper <= 0.0 {
        add_mixed_row(
            rows,
            format!("{name}__abs_nonpositive"),
            expansions,
            &[(operand_var, 1.0)],
            &[(abs_var, 1.0)],
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
    add_mixed_row(
        rows,
        format!("{name}__abs_ge_x"),
        expansions,
        &[(operand_var, 1.0)],
        &[(abs_var, -1.0)],
        RowSense::Le,
        0.0,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_ge_neg_x"),
        expansions,
        &[(operand_var, -1.0)],
        &[(abs_var, -1.0)],
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
        &[(operand_var, -1.0)],
        &[(abs_var, 1.0), (z, -2.0 * lower)],
        RowSense::Le,
        -2.0 * lower,
    );
    add_mixed_row(
        rows,
        format!("{name}__abs_result_neg_branch"),
        expansions,
        &[(operand_var, 1.0)],
        &[(abs_var, 1.0), (z, -2.0 * upper)],
        RowSense::Le,
        0.0,
    );
    Ok(())
}

fn add_bin_packing_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    item_bin_vars: &[usize],
    load_vars: &[usize],
    item_sizes: &[f64],
) -> Result<(), MathProgramError> {
    let mut load_terms = vec![Vec::<(usize, f64)>::new(); load_vars.len()];
    for (item, (&item_bin_var, &size)) in item_bin_vars.iter().zip(item_sizes).enumerate() {
        let (lower, upper) = integer_bounds(&program.variables[item_bin_var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "bin-packing item {item} variable `{}` requires finite integer bounds",
                program.variables[item_bin_var].name
            ))
        })?;
        let selectors = (lower..=upper)
            .map(|bin| {
                let lit = push_canonical_var(
                    &format!("{name}__item_{item}__bin_{bin}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                );
                (bin, lit)
            })
            .collect::<Vec<_>>();
        let choose_coeffs = selectors
            .iter()
            .map(|&(_, lit)| (lit, 1.0))
            .collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: choose_coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__item_{item}__choose_bin"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&choose_coeffs),
            rhs: -1.0,
            name: format!("{name}__item_{item}__choose_bin_ge"),
        });

        let item_bin_coeffs = selectors
            .iter()
            .map(|&(bin, lit)| (lit, -(bin as f64)))
            .collect::<Vec<_>>();
        add_mixed_row(
            rows,
            format!("{name}__item_{item}__link_bin"),
            expansions,
            &[(item_bin_var, 1.0)],
            &item_bin_coeffs,
            RowSense::Eq,
            0.0,
        );

        for (bin, lit) in selectors {
            load_terms[bin as usize].push((lit, -size));
        }
    }

    for (bin, &load_var) in load_vars.iter().enumerate() {
        add_mixed_row(
            rows,
            format!("{name}__bin_{bin}__load"),
            expansions,
            &[(load_var, 1.0)],
            &load_terms[bin],
            RowSense::Eq,
            0.0,
        );
    }

    Ok(())
}

fn add_element_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    index_var: usize,
    target_var: usize,
    values: &[f64],
) -> Result<(), MathProgramError> {
    if values.is_empty() {
        return Err(MathProgramError::Unsupported(
            "element requires at least one value".to_string(),
        ));
    }
    let index = &program.variables[index_var];
    let (lower, upper) = integer_bounds(index).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "element index variable `{}` requires finite integer bounds",
            index.name
        ))
    })?;
    let max_index = i64::try_from(values.len() - 1).map_err(|_| {
        MathProgramError::Unsupported(format!(
            "element value array is too large: {} values",
            values.len()
        ))
    })?;
    if lower < 0 || upper > max_index {
        return Err(MathProgramError::InvalidBound(format!(
            "element index variable `{}` bounds [{lower}, {upper}] must fit value indices [0, {max_index}]",
            index.name
        )));
    }

    let selectors = (lower..=upper)
        .map(|idx| {
            let lit = push_canonical_var(
                &format!("{name}__index_{idx}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            (idx, lit)
        })
        .collect::<Vec<_>>();
    let choose_coeffs = selectors
        .iter()
        .map(|&(_, lit)| (lit, 1.0))
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: choose_coeffs.clone(),
        rhs: 1.0,
        name: format!("{name}__choose_index"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&choose_coeffs),
        rhs: -1.0,
        name: format!("{name}__choose_index_ge"),
    });

    let index_coeffs = selectors
        .iter()
        .map(|&(idx, lit)| (lit, -(idx as f64)))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__link_index"),
        expansions,
        &[(index_var, 1.0)],
        &index_coeffs,
        RowSense::Eq,
        0.0,
    );

    let target_coeffs = selectors
        .iter()
        .map(|&(idx, lit)| (lit, -values[idx as usize]))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__link_target"),
        expansions,
        &[(target_var, 1.0)],
        &target_coeffs,
        RowSense::Eq,
        0.0,
    );

    Ok(())
}

fn add_variable_element_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    index_var: usize,
    target_var: usize,
    variables: &[usize],
) -> Result<(), MathProgramError> {
    if variables.is_empty() {
        return Err(MathProgramError::Unsupported(
            "variable-element requires at least one variable".to_string(),
        ));
    }
    let index = &program.variables[index_var];
    let (lower, upper) = integer_bounds(index).ok_or_else(|| {
        MathProgramError::UnboundedBigM(format!(
            "variable-element index variable `{}` requires finite integer bounds",
            index.name
        ))
    })?;
    let max_index = i64::try_from(variables.len() - 1).map_err(|_| {
        MathProgramError::Unsupported(format!(
            "variable-element array is too large: {} variables",
            variables.len()
        ))
    })?;
    if lower < 0 || upper > max_index {
        return Err(MathProgramError::InvalidBound(format!(
            "variable-element index variable `{}` bounds [{lower}, {upper}] must fit variable indices [0, {max_index}]",
            index.name
        )));
    }

    let selectors = (lower..=upper)
        .map(|idx| {
            let lit = push_canonical_var(
                &format!("{name}__index_{idx}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            );
            (idx, lit)
        })
        .collect::<Vec<_>>();
    let choose_coeffs = selectors
        .iter()
        .map(|&(_, lit)| (lit, 1.0))
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: choose_coeffs.clone(),
        rhs: 1.0,
        name: format!("{name}__choose_index"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&choose_coeffs),
        rhs: -1.0,
        name: format!("{name}__choose_index_ge"),
    });

    let index_coeffs = selectors
        .iter()
        .map(|&(idx, lit)| (lit, -(idx as f64)))
        .collect::<Vec<_>>();
    add_mixed_row(
        rows,
        format!("{name}__link_index"),
        expansions,
        &[(index_var, 1.0)],
        &index_coeffs,
        RowSense::Eq,
        0.0,
    );

    let (target_lb, target_ub) =
        variable_bounds(&program.variables[target_var]).ok_or_else(|| {
            MathProgramError::UnboundedBigM(format!(
                "variable-element target variable `{}` requires finite bounds",
                program.variables[target_var].name
            ))
        })?;
    for &(idx, lit) in &selectors {
        let source_var = variables[idx as usize];
        let (source_lb, source_ub) =
            variable_bounds(&program.variables[source_var]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "variable-element source variable `{}` at index {idx} requires finite bounds",
                    program.variables[source_var].name
                ))
            })?;
        let delta_lower = target_lb - source_ub;
        let delta_upper = target_ub - source_lb;
        add_mixed_row(
            rows,
            format!("{name}__index_{idx}__target_le_source"),
            expansions,
            &[(target_var, 1.0), (source_var, -1.0)],
            &[(lit, delta_upper)],
            RowSense::Le,
            delta_upper,
        );
        add_mixed_row(
            rows,
            format!("{name}__index_{idx}__target_ge_source"),
            expansions,
            &[(target_var, -1.0), (source_var, 1.0)],
            &[(lit, -delta_lower)],
            RowSense::Le,
            -delta_lower,
        );
    }

    Ok(())
}

fn add_automaton_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    variables: &[usize],
    starting_state: i64,
    final_states: &[i64],
    transitions: &[AutomatonTransition],
) -> Result<(), MathProgramError> {
    let states = automaton_states(starting_state, final_states, transitions);
    let state_index = states
        .iter()
        .enumerate()
        .map(|(idx, &state)| (state, idx))
        .collect::<BTreeMap<_, _>>();
    let mut state_literals = Vec::with_capacity(variables.len() + 1);
    for stage in 0..=variables.len() {
        let mut stage_literals = Vec::with_capacity(states.len());
        for &state in &states {
            stage_literals.push(push_canonical_var(
                &format!("{name}__stage_{stage}__state_{state}"),
                true,
                1.0,
                names,
                integer_vars,
                ub,
            ));
        }
        state_literals.push(stage_literals);
    }

    for (stage, literals) in state_literals.iter().enumerate() {
        let coeffs = literals.iter().map(|&lit| (lit, 1.0)).collect::<Vec<_>>();
        rows.push(SparseRow {
            coeffs: coeffs.clone(),
            rhs: 1.0,
            name: format!("{name}__stage_{stage}__one_state"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&coeffs),
            rhs: -1.0,
            name: format!("{name}__stage_{stage}__one_state_ge"),
        });
    }

    let start_idx = state_index[&starting_state];
    rows.push(SparseRow {
        coeffs: vec![(state_literals[0][start_idx], -1.0)],
        rhs: -1.0,
        name: format!("{name}__start_state"),
    });
    let final_coeffs = unique_i64(final_states)
        .iter()
        .map(|state| (state_literals[variables.len()][state_index[state]], -1.0))
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: final_coeffs,
        rhs: -1.0,
        name: format!("{name}__final_state"),
    });

    for (stage, &var_idx) in variables.iter().enumerate() {
        let arcs = transitions
            .iter()
            .enumerate()
            .map(|(transition_idx, _)| {
                push_canonical_var(
                    &format!("{name}__stage_{stage}__transition_{transition_idx}"),
                    true,
                    1.0,
                    names,
                    integer_vars,
                    ub,
                )
            })
            .collect::<Vec<_>>();

        for (state_pos, &state) in states.iter().enumerate() {
            let outgoing = transitions
                .iter()
                .enumerate()
                .filter_map(|(transition_idx, transition)| {
                    (transition.tail == state).then_some((arcs[transition_idx], 1.0))
                })
                .collect::<Vec<_>>();
            let mut outgoing_link = outgoing;
            outgoing_link.push((state_literals[stage][state_pos], -1.0));
            rows.push(SparseRow {
                coeffs: outgoing_link.clone(),
                rhs: 0.0,
                name: format!("{name}__stage_{stage}__state_{state}__outflow"),
            });
            rows.push(SparseRow {
                coeffs: negate_sparse(&outgoing_link),
                rhs: 0.0,
                name: format!("{name}__stage_{stage}__state_{state}__outflow_ge"),
            });

            let incoming = transitions
                .iter()
                .enumerate()
                .filter_map(|(transition_idx, transition)| {
                    (transition.head == state).then_some((arcs[transition_idx], 1.0))
                })
                .collect::<Vec<_>>();
            let mut incoming_link = incoming;
            incoming_link.push((state_literals[stage + 1][state_pos], -1.0));
            rows.push(SparseRow {
                coeffs: incoming_link.clone(),
                rhs: 0.0,
                name: format!("{name}__stage_{stage}__state_{state}__inflow"),
            });
            rows.push(SparseRow {
                coeffs: negate_sparse(&incoming_link),
                rhs: 0.0,
                name: format!("{name}__stage_{stage}__state_{state}__inflow_ge"),
            });
        }

        let canonical_value = transitions
            .iter()
            .zip(&arcs)
            .map(|(transition, &arc)| (arc, -(transition.label as f64)))
            .collect::<Vec<_>>();
        add_mixed_row(
            rows,
            format!("{name}__stage_{stage}__label"),
            expansions,
            &[(var_idx, 1.0)],
            &canonical_value,
            RowSense::Eq,
            0.0,
        );
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
    let choose_coeffs = selectors.iter().map(|&z| (z, 1.0)).collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: choose_coeffs.clone(),
        rhs: 1.0,
        name: format!("{name}__choose_one"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&choose_coeffs),
        rhs: -1.0,
        name: format!("{name}__choose_one_ge"),
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
    let lambda_sum = lambdas
        .iter()
        .map(|&lambda| (lambda, 1.0))
        .collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: lambda_sum.clone(),
        rhs: 1.0,
        name: format!("{name}__lambda_sum"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&lambda_sum),
        rhs: -1.0,
        name: format!("{name}__lambda_sum_ge"),
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
    let segment_sum = intervals.iter().map(|&z| (z, 1.0)).collect::<Vec<_>>();
    rows.push(SparseRow {
        coeffs: segment_sum.clone(),
        rhs: 1.0,
        name: format!("{name}__segment_sum"),
    });
    rows.push(SparseRow {
        coeffs: negate_sparse(&segment_sum),
        rhs: -1.0,
        name: format!("{name}__segment_sum_ge"),
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

fn add_alternative_rows(
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    master: &IntervalTerm,
    alternatives: &[IntervalTerm],
) -> Result<(), MathProgramError> {
    add_interval_link_rows(ub, rows, expansions, &format!("{name}__master"), master)?;
    for (i, alternative) in alternatives.iter().enumerate() {
        add_interval_link_rows(
            ub,
            rows,
            expansions,
            &format!("{name}__alternative_{i}"),
            alternative,
        )?;
    }

    let mut choose_coeffs = alternatives
        .iter()
        .map(|alternative| (alternative.presence_var.unwrap(), 1.0))
        .collect::<Vec<_>>();
    let rhs = if let Some(master_presence) = master.presence_var {
        choose_coeffs.push((master_presence, -1.0));
        0.0
    } else {
        1.0
    };
    add_program_row(
        rows,
        format!("{name}__choose_alternative"),
        expansions,
        &choose_coeffs,
        RowSense::Eq,
        rhs,
    );

    for (i, alternative) in alternatives.iter().enumerate() {
        let presence = alternative.presence_var.unwrap();
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__alternative_{i}__start_le_master"),
            &[(alternative.start_var, 1.0), (master.start_var, -1.0)],
            0.0,
            &[(presence, true)],
            &[],
            ub,
        )?;
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__alternative_{i}__master_le_start"),
            &[(master.start_var, 1.0), (alternative.start_var, -1.0)],
            0.0,
            &[(presence, true)],
            &[],
            ub,
        )?;
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__alternative_{i}__end_le_master"),
            &[(alternative.end_var, 1.0), (master.end_var, -1.0)],
            0.0,
            &[(presence, true)],
            &[],
            ub,
        )?;
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__alternative_{i}__master_le_end"),
            &[(master.end_var, 1.0), (alternative.end_var, -1.0)],
            0.0,
            &[(presence, true)],
            &[],
            ub,
        )?;
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
            let (left_before_right, left_before_right_rhs) =
                interval_precedence_row(left, right.start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__no_overlap_{i}_before_{j}"),
                &left_before_right,
                left_before_right_rhs,
                &presence_literals,
                &[(order, true)],
                ub,
            )?;
            let (right_before_left, right_before_left_rhs) =
                interval_precedence_row(right, left.start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__no_overlap_{j}_before_{i}"),
                &right_before_left,
                right_before_left_rhs,
                &presence_literals,
                &[(order, false)],
                ub,
            )?;
        }
    }
    Ok(())
}

fn interval_precedence_row(
    interval: &IntervalTerm,
    other_start_var: usize,
) -> (Vec<(usize, f64)>, f64) {
    let mut coeffs = vec![(interval.start_var, 1.0), (other_start_var, -1.0)];
    if let Some(duration_var) = interval.duration_var {
        coeffs.push((duration_var, 1.0));
    }
    (coeffs, -interval.duration)
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

            let (i_left_of_j, i_left_of_j_rhs) =
                interval_precedence_row(&x_intervals[i], x_intervals[j].start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{i}_left_of_{j}__enforce"),
                &i_left_of_j,
                i_left_of_j_rhs,
                &active_literals,
                &[(separators[0], true)],
                ub,
            )?;
            let (j_left_of_i, j_left_of_i_rhs) =
                interval_precedence_row(&x_intervals[j], x_intervals[i].start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{j}_left_of_{i}__enforce"),
                &j_left_of_i,
                j_left_of_i_rhs,
                &active_literals,
                &[(separators[1], true)],
                ub,
            )?;
            let (i_below_j, i_below_j_rhs) =
                interval_precedence_row(&y_intervals[i], y_intervals[j].start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{i}_below_{j}__enforce"),
                &i_below_j,
                i_below_j_rhs,
                &active_literals,
                &[(separators[2], true)],
                ub,
            )?;
            let (j_below_i, j_below_i_rhs) =
                interval_precedence_row(&y_intervals[j], y_intervals[i].start_var);
            add_implied_le_row(
                rows,
                expansions,
                format!("{name}__rect_{j}_below_{i}__enforce"),
                &j_below_i,
                j_below_i_rhs,
                &active_literals,
                &[(separators[3], true)],
                ub,
            )?;
        }
    }
    Ok(())
}

struct CumulativeChoice {
    start: i64,
    duration: i64,
    load_terms: Vec<(usize, f64)>,
}

fn add_binary_times_canonical_var_rows(
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    name: String,
    binary_var: usize,
    value_var: usize,
) -> Result<usize, MathProgramError> {
    let upper = ub.get(value_var).copied().ok_or_else(|| {
        MathProgramError::BadIndex(format!(
            "{name} references missing canonical variable {value_var}"
        ))
    })?;
    if !upper.is_finite() {
        return Err(MathProgramError::UnboundedBigM(format!(
            "{name} requires a finite upper bound for canonical variable {value_var}"
        )));
    }
    let product = push_canonical_var(
        &name,
        integer_vars.get(value_var).copied().ok_or_else(|| {
            MathProgramError::BadIndex(format!(
                "{name} references missing canonical variable {value_var}"
            ))
        })?,
        upper,
        names,
        integer_vars,
        ub,
    );
    rows.push(SparseRow {
        coeffs: vec![(product, 1.0), (value_var, -1.0)],
        rhs: 0.0,
        name: format!("{name}__upper_value"),
    });
    rows.push(SparseRow {
        coeffs: vec![(product, 1.0), (binary_var, -upper)],
        rhs: 0.0,
        name: format!("{name}__upper_active"),
    });
    rows.push(SparseRow {
        coeffs: vec![(value_var, 1.0), (product, -1.0), (binary_var, upper)],
        rhs: upper,
        name: format!("{name}__lower_active"),
    });
    Ok(product)
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
    demands: &[AffineTerm],
    capacity: &AffineTerm,
) -> Result<(), MathProgramError> {
    let mut interval_choices = Vec::new();
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
        let duration_offset = interval.duration.round() as i64;
        let duration_values = if let Some(duration_var) = interval.duration_var {
            let (duration_lb, duration_ub) = integer_bounds(&program.variables[duration_var])
                .ok_or_else(|| {
                    MathProgramError::UnboundedBigM(format!(
                        "cumulative interval {i} duration requires finite integer bounds"
                    ))
                })?;
            (duration_lb..=duration_ub)
                .map(|duration_value| (duration_value, duration_offset + duration_value))
                .collect::<Vec<_>>()
        } else {
            vec![(0, duration_offset)]
        };
        let mut choices = Vec::new();
        for start in start_lb..=start_ub {
            for &(duration_value, total_duration) in &duration_values {
                let choice_name = if interval.duration_var.is_some() {
                    format!("{name}__interval_{i}__starts_at_{start}__duration_{duration_value}")
                } else {
                    format!("{name}__interval_{i}__starts_at_{start}")
                };
                let choice = push_canonical_var(&choice_name, true, 1.0, names, integer_vars, ub);
                choices.push((start, duration_value, total_duration, choice));
            }
        }

        let choose_terms = choices
            .iter()
            .map(|&(_, _, _, choice)| (choice, 1.0))
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

        let start_sum = choices
            .iter()
            .map(|&(start, _, _, choice)| (choice, start as f64))
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

        if let Some(duration_var) = interval.duration_var {
            let duration_sum = choices
                .iter()
                .map(|&(_, duration_value, _, choice)| (choice, duration_value as f64))
                .collect::<Vec<_>>();
            let canonical_duration_sum = duration_sum
                .iter()
                .map(|&(idx, coef)| (idx, -coef))
                .collect::<Vec<_>>();
            if let Some(presence) = interval.presence_var {
                add_implied_mixed_le_row(
                    rows,
                    expansions,
                    format!("{name}__interval_{i}__duration_link_upper"),
                    &[(duration_var, 1.0)],
                    &canonical_duration_sum,
                    0.0,
                    &[(presence, true)],
                    &[],
                    ub,
                )?;
                let reverse_duration_sum = duration_sum
                    .iter()
                    .map(|&(idx, coef)| (idx, coef))
                    .collect::<Vec<_>>();
                add_implied_mixed_le_row(
                    rows,
                    expansions,
                    format!("{name}__interval_{i}__duration_link_lower"),
                    &[(duration_var, -1.0)],
                    &reverse_duration_sum,
                    0.0,
                    &[(presence, true)],
                    &[],
                    ub,
                )?;
            } else {
                add_mixed_row(
                    rows,
                    format!("{name}__interval_{i}__duration_link"),
                    expansions,
                    &[(duration_var, 1.0)],
                    &canonical_duration_sum,
                    RowSense::Eq,
                    0.0,
                );
            }
        }

        min_time = min_time.min(start_lb);
        if let Some(max_duration) = duration_values
            .iter()
            .map(|&(_, total_duration)| total_duration)
            .max()
        {
            max_time = max_time.max(start_ub + max_duration);
        }
        let (demand_coeffs, demand_constant) = expand_affine_term(expansions, &demands[i]);
        let mut cumulative_choices = Vec::with_capacity(choices.len());
        for &(start, _, total_duration, choice) in &choices {
            let mut load_terms = Vec::new();
            if demand_constant.abs() > 1e-12 {
                load_terms.push((choice, demand_constant));
            }
            for &(demand_var, coef) in &demand_coeffs {
                if coef.abs() <= 1e-12 {
                    continue;
                }
                let product = add_binary_times_canonical_var_rows(
                    names,
                    integer_vars,
                    ub,
                    rows,
                    format!("{name}__interval_{i}__choice_{choice}__demand_{demand_var}"),
                    choice,
                    demand_var,
                )?;
                load_terms.push((product, coef));
            }
            cumulative_choices.push(CumulativeChoice {
                start,
                duration: total_duration,
                load_terms,
            });
        }
        interval_choices.push(cumulative_choices);
    }

    let (capacity_coeffs, capacity_constant) = expand_affine_term(expansions, capacity);
    for t in min_time..max_time {
        let mut coeffs = Vec::new();
        for choices in &interval_choices {
            for choice in choices {
                if choice.start <= t && t < choice.start + choice.duration {
                    coeffs.extend(choice.load_terms.iter().copied());
                }
            }
        }
        if !coeffs.is_empty() {
            coeffs.extend(capacity_coeffs.iter().map(|&(idx, coef)| (idx, -coef)));
            rows.push(SparseRow {
                coeffs: combine_terms(&coeffs),
                rhs: capacity_constant,
                name: format!("{name}__capacity_at_{t}"),
            });
        }
    }
    Ok(())
}

fn add_reservoir_rows(
    program: &MathProgram,
    names: &mut Vec<String>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    rows: &mut Vec<SparseRow>,
    expansions: &[LinearExpansion],
    name: &str,
    events: &[ReservoirEvent],
    min_level: f64,
    max_level: f64,
) -> Result<(), MathProgramError> {
    let mut event_choices = Vec::with_capacity(events.len());
    let mut min_time = i64::MAX;
    let mut max_time = i64::MIN;

    for (i, event) in events.iter().enumerate() {
        let (time_lb, time_ub) =
            integer_bounds(&program.variables[event.time_var]).ok_or_else(|| {
                MathProgramError::UnboundedBigM(format!(
                    "reservoir event {i} time requires finite integer bounds"
                ))
            })?;
        let choices = (time_lb..=time_ub)
            .map(|time| {
                push_canonical_var(
                    &format!("{name}__event_{i}__at_{time}"),
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

        if let Some(active_var) = event.active_var {
            add_mixed_row(
                rows,
                format!("{name}__event_{i}__choose_if_active"),
                expansions,
                &[(active_var, -1.0)],
                &choose_terms,
                RowSense::Eq,
                0.0,
            );
        } else {
            add_mixed_row(
                rows,
                format!("{name}__event_{i}__choose_one_time"),
                expansions,
                &[],
                &choose_terms,
                RowSense::Eq,
                1.0,
            );
        }

        let time_sum = (time_lb..=time_ub)
            .zip(&choices)
            .map(|(time, &choice)| (choice, time as f64))
            .collect::<Vec<_>>();
        let canonical_time_sum = time_sum
            .iter()
            .map(|&(idx, coef)| (idx, -coef))
            .collect::<Vec<_>>();
        if let Some(active_var) = event.active_var {
            add_implied_mixed_le_row(
                rows,
                expansions,
                format!("{name}__event_{i}__time_link_upper"),
                &[(event.time_var, 1.0)],
                &canonical_time_sum,
                0.0,
                &[(active_var, true)],
                &[],
                ub,
            )?;
            let reverse_time_sum = time_sum
                .iter()
                .map(|&(idx, coef)| (idx, coef))
                .collect::<Vec<_>>();
            add_implied_mixed_le_row(
                rows,
                expansions,
                format!("{name}__event_{i}__time_link_lower"),
                &[(event.time_var, -1.0)],
                &reverse_time_sum,
                0.0,
                &[(active_var, true)],
                &[],
                ub,
            )?;
        } else {
            add_mixed_row(
                rows,
                format!("{name}__event_{i}__time_link"),
                expansions,
                &[(event.time_var, 1.0)],
                &canonical_time_sum,
                RowSense::Eq,
                0.0,
            );
        }

        min_time = min_time.min(time_lb);
        max_time = max_time.max(time_ub);
        event_choices.push((time_lb, event.demand, choices));
    }

    for time in min_time..=max_time {
        let mut coeffs = Vec::new();
        for (time_lb, demand, choices) in &event_choices {
            if demand.abs() <= 1e-12 {
                continue;
            }
            for (offset, &choice) in choices.iter().enumerate() {
                if *time_lb + offset as i64 <= time {
                    coeffs.push((choice, *demand));
                }
            }
        }
        if coeffs.is_empty() {
            continue;
        }
        let coeffs = combine_terms(&coeffs);
        rows.push(SparseRow {
            coeffs: coeffs.clone(),
            rhs: max_level,
            name: format!("{name}__level_max_at_{time}"),
        });
        rows.push(SparseRow {
            coeffs: negate_sparse(&coeffs),
            rhs: -min_level,
            name: format!("{name}__level_min_at_{time}"),
        });
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
    let mut upper_terms = vec![(interval.end_var, 1.0), (interval.start_var, -1.0)];
    if let Some(duration_var) = interval.duration_var {
        upper_terms.push((duration_var, -1.0));
    }
    let mut lower_terms = vec![(interval.start_var, 1.0), (interval.end_var, -1.0)];
    if let Some(duration_var) = interval.duration_var {
        lower_terms.push((duration_var, 1.0));
    }
    if let Some(presence) = interval.presence_var {
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__end_after_start_upper"),
            &upper_terms,
            interval.duration,
            &[(presence, true)],
            &[],
            ub,
        )?;
        add_implied_le_row(
            rows,
            expansions,
            format!("{name}__end_after_start_lower"),
            &lower_terms,
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
            &upper_terms,
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

fn expand_affine_term(
    expansions: &[LinearExpansion],
    term: &AffineTerm,
) -> (Vec<(usize, f64)>, f64) {
    let mut constant = term.constant;
    let mut terms = BTreeMap::<usize, f64>::new();
    for &(var_idx, coef) in &term.coeffs {
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
    (sparse, constant)
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
    for row in &program.lazy_constraints {
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
    for enforced in &program.enforced_constraints {
        if enforced
            .literals
            .iter()
            .all(|literal| binary_truth(x[literal.var]) == literal.value)
        {
            let lhs = eval_sparse_affine(&enforced.coeffs, 0.0, x);
            max_violation =
                max_violation.max(row_sense_violation(lhs, enforced.sense, enforced.rhs));
        }
    }
    for reified in &program.reified_constraints {
        max_violation = max_violation.max(reified_linear_violation(reified, x));
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

fn reified_linear_violation(reified: &ReifiedLinearConstraint, x: &[f64]) -> f64 {
    let lhs = eval_sparse_affine(&reified.coeffs, 0.0, x);
    let literal_true = binary_truth(x[reified.literal.var]) == reified.literal.value;
    if literal_true {
        return row_sense_violation(lhs, reified.sense, reified.rhs);
    }
    match reified.sense {
        RowSense::Le => (reified.rhs + 1.0 - lhs).max(0.0),
        RowSense::Ge => (lhs - (reified.rhs - 1.0)).max(0.0),
        RowSense::Eq => (1.0 - (lhs - reified.rhs).abs()).max(0.0),
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
        GeneralConstraint::BinaryXor {
            result_var,
            operands,
            ..
        } => {
            let expected =
                (operands.iter().filter(|&&idx| binary_truth(x[idx])).count() % 2) as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::BooleanAnd {
            result_var,
            literals,
            ..
        } => {
            let expected = literals
                .iter()
                .all(|literal| binary_truth(x[literal.var]) == literal.value)
                as u8 as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::BooleanOr {
            result_var,
            literals,
            ..
        } => {
            let expected = literals
                .iter()
                .any(|literal| binary_truth(x[literal.var]) == literal.value)
                as u8 as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::BooleanXor {
            result_var,
            literals,
            ..
        } => {
            let expected = (literals
                .iter()
                .filter(|literal| binary_truth(x[literal.var]) == literal.value)
                .count()
                % 2) as f64;
            (x[*result_var] - expected).abs()
        }
        GeneralConstraint::BinaryCardinality {
            operands,
            min_count,
            max_count,
            ..
        } => binary_cardinality_violation(operands, *min_count, *max_count, x),
        GeneralConstraint::BooleanCardinality {
            literals,
            min_count,
            max_count,
            ..
        } => boolean_cardinality_violation(literals, *min_count, *max_count, x),
        GeneralConstraint::BooleanCount {
            literals,
            count_var,
            ..
        } => boolean_count_violation(literals, *count_var, x),
        GeneralConstraint::BooleanClause { literals, .. } => boolean_clause_violation(literals, x),
        GeneralConstraint::LinearDomain {
            coeffs, intervals, ..
        } => linear_domain_violation(coeffs, intervals, x),
        GeneralConstraint::EnforcedLinearDomain {
            enforcement,
            coeffs,
            intervals,
            ..
        } => enforced_linear_domain_violation(enforcement, coeffs, intervals, x),
        GeneralConstraint::MapDomain {
            var,
            bool_vars,
            offset,
            ..
        } => map_domain_violation(*var, bool_vars, *offset, x),
        GeneralConstraint::IntegerProduct {
            target_var,
            operands,
            ..
        } => integer_product_violation(*target_var, operands, x),
        GeneralConstraint::IntegerDivision {
            target_var,
            numerator_var,
            denominator_var,
            ..
        } => integer_binary_operation_violation(
            *target_var,
            *numerator_var,
            *denominator_var,
            x,
            i64::checked_div,
        ),
        GeneralConstraint::IntegerModulo {
            target_var,
            numerator_var,
            denominator_var,
            ..
        } => integer_binary_operation_violation(
            *target_var,
            *numerator_var,
            *denominator_var,
            x,
            i64::checked_rem,
        ),
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
        GeneralConstraint::Norm {
            result_var,
            operands,
            norm_type,
            ..
        } => norm_violation(*result_var, operands, *norm_type, x),
        GeneralConstraint::PiecewiseLinear {
            x_var,
            y_var,
            points,
            ..
        } => piecewise_violation(points, x[*x_var], x[*y_var], tol),
        GeneralConstraint::AllDifferent { variables, .. } => all_different_violation(variables, x),
        GeneralConstraint::GlobalCardinality {
            variables, counts, ..
        } => global_cardinality_violation(variables, counts, x),
        GeneralConstraint::ValueCount {
            variables,
            value,
            count_var,
            ..
        } => value_count_violation(variables, *value, *count_var, x),
        GeneralConstraint::AllowedAssignments {
            variables, tuples, ..
        } => allowed_assignments_violation(variables, tuples, x),
        GeneralConstraint::ForbiddenAssignments {
            variables, tuples, ..
        } => forbidden_assignments_violation(variables, tuples, x),
        GeneralConstraint::EnforcedAllowedAssignments {
            enforcement,
            variables,
            tuples,
            ..
        } => enforced_table_violation(enforcement, variables, tuples, x, false),
        GeneralConstraint::EnforcedForbiddenAssignments {
            enforcement,
            variables,
            tuples,
            ..
        } => enforced_table_violation(enforcement, variables, tuples, x, true),
        GeneralConstraint::BinPacking {
            item_bin_vars,
            load_vars,
            item_sizes,
            ..
        } => bin_packing_violation(item_bin_vars, load_vars, item_sizes, x),
        GeneralConstraint::Element {
            index_var,
            target_var,
            values,
            ..
        } => element_violation(*index_var, *target_var, values, x),
        GeneralConstraint::VariableElement {
            index_var,
            target_var,
            variables,
            ..
        } => variable_element_violation(*index_var, *target_var, variables, x),
        GeneralConstraint::Inverse {
            variables,
            inverse_variables,
            ..
        } => inverse_violation(variables, inverse_variables, x),
        GeneralConstraint::Circuit {
            node_count, arcs, ..
        } => circuit_violation(*node_count, arcs, x),
        GeneralConstraint::MultipleCircuit {
            node_count, arcs, ..
        } => multiple_circuit_violation(*node_count, arcs, x),
        GeneralConstraint::Automaton {
            variables,
            starting_state,
            final_states,
            transitions,
            ..
        } => automaton_violation(variables, *starting_state, final_states, transitions, x),
        GeneralConstraint::Alternative {
            master,
            alternatives,
            ..
        } => alternative_violation(master, alternatives, x),
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
        } => cumulative_violation(intervals, demands, capacity, x, tol),
        GeneralConstraint::Reservoir {
            events,
            min_level,
            max_level,
            ..
        } => reservoir_violation(events, *min_level, *max_level, x),
    }
}

fn binary_cardinality_violation(
    operands: &[usize],
    min_count: Option<usize>,
    max_count: Option<usize>,
    x: &[f64],
) -> f64 {
    let count = operands.iter().filter(|&&idx| binary_truth(x[idx])).count() as f64;
    let lower_violation = min_count
        .map(|min_count| (min_count as f64 - count).max(0.0))
        .unwrap_or(0.0);
    let upper_violation = max_count
        .map(|max_count| (count - max_count as f64).max(0.0))
        .unwrap_or(0.0);
    lower_violation.max(upper_violation)
}

fn literal_cardinality_terms(literals: &[BoolLiteral]) -> (Vec<(usize, f64)>, usize) {
    let mut coeffs = Vec::with_capacity(literals.len());
    let mut negated_count = 0usize;
    for literal in literals {
        if literal.value {
            coeffs.push((literal.var, 1.0));
        } else {
            coeffs.push((literal.var, -1.0));
            negated_count += 1;
        }
    }
    (coeffs, negated_count)
}

fn boolean_cardinality_violation(
    literals: &[BoolLiteral],
    min_count: Option<usize>,
    max_count: Option<usize>,
    x: &[f64],
) -> f64 {
    let count = literals
        .iter()
        .filter(|literal| binary_truth(x[literal.var]) == literal.value)
        .count() as f64;
    let lower_violation = min_count
        .map(|min_count| (min_count as f64 - count).max(0.0))
        .unwrap_or(0.0);
    let upper_violation = max_count
        .map(|max_count| (count - max_count as f64).max(0.0))
        .unwrap_or(0.0);
    lower_violation.max(upper_violation)
}

fn boolean_count_violation(literals: &[BoolLiteral], count_var: usize, x: &[f64]) -> f64 {
    let count = literals
        .iter()
        .filter(|literal| binary_truth(x[literal.var]) == literal.value)
        .count() as f64;
    integrality_violation(x[count_var]).max((x[count_var] - count).abs())
}

fn boolean_clause_violation(literals: &[BoolLiteral], x: &[f64]) -> f64 {
    if literals
        .iter()
        .any(|literal| binary_truth(x[literal.var]) == literal.value)
    {
        0.0
    } else {
        1.0
    }
}

fn linear_domain_violation(
    coeffs: &[(usize, f64)],
    intervals: &[LinearDomainInterval],
    x: &[f64],
) -> f64 {
    let value = eval_sparse_affine(coeffs, 0.0, x);
    let domain_distance = intervals
        .iter()
        .map(|interval| {
            let lower = interval.lower as f64;
            let upper = interval.upper as f64;
            if value >= lower && value <= upper {
                0.0
            } else if value < lower {
                lower - value
            } else {
                value - upper
            }
        })
        .fold(f64::INFINITY, f64::min);
    integrality_violation(value).max(domain_distance)
}

fn enforced_linear_domain_violation(
    enforcement: &[BoolLiteral],
    coeffs: &[(usize, f64)],
    intervals: &[LinearDomainInterval],
    x: &[f64],
) -> f64 {
    let active = enforcement
        .iter()
        .all(|literal| binary_truth(x[literal.var]) == literal.value);
    if active {
        linear_domain_violation(coeffs, intervals, x)
    } else {
        0.0
    }
}

fn integer_product_violation(target_var: usize, operands: &[usize], x: &[f64]) -> f64 {
    let mut product = 1.0;
    let mut violation = integrality_violation(x[target_var]);
    for &operand in operands {
        violation = violation.max(integrality_violation(x[operand]));
        product *= x[operand].round();
    }
    violation.max((x[target_var] - product).abs())
}

fn integer_binary_operation_violation(
    target_var: usize,
    numerator_var: usize,
    denominator_var: usize,
    x: &[f64],
    operation: fn(i64, i64) -> Option<i64>,
) -> f64 {
    let violation = integrality_violation(x[target_var])
        .max(integrality_violation(x[numerator_var]))
        .max(integrality_violation(x[denominator_var]));
    let numerator = x[numerator_var].round() as i64;
    let denominator = x[denominator_var].round() as i64;
    let Some(expected) = operation(numerator, denominator) else {
        return violation.max(1.0);
    };
    violation.max((x[target_var] - expected as f64).abs())
}

fn norm_violation(result_var: usize, operands: &[usize], norm_type: NormType, x: &[f64]) -> f64 {
    if norm_type == NormType::L0 {
        let mut violation = integrality_violation(x[result_var]);
        let mut count = 0usize;
        for &operand in operands {
            violation = violation.max(integrality_violation(x[operand]));
            if x[operand].round() != 0.0 {
                count += 1;
            }
        }
        return violation.max((x[result_var] - count as f64).abs());
    }

    let expected = match norm_type {
        NormType::L0 => unreachable!("l0 norm violation returned before dense norm evaluation"),
        NormType::L1 => operands.iter().map(|&idx| x[idx].abs()).sum::<f64>(),
        NormType::LInfinity => operands
            .iter()
            .map(|&idx| x[idx].abs())
            .fold(0.0_f64, f64::max),
    };
    (x[result_var] - expected).abs()
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

fn global_cardinality_violation(
    variables: &[usize],
    counts: &[GlobalCardinalityCount],
    x: &[f64],
) -> f64 {
    let mut value_counts = BTreeMap::<i64, usize>::new();
    let mut violation: f64 = 0.0;
    for &idx in variables {
        let value = x[idx];
        violation = violation.max(integrality_violation(value));
        *value_counts.entry(value.round() as i64).or_insert(0) += 1;
    }

    for count in counts {
        let observed = *value_counts.get(&count.value).unwrap_or(&0) as f64;
        if let Some(min_count) = count.min_count {
            violation = violation.max((min_count as f64 - observed).max(0.0));
        }
        if let Some(max_count) = count.max_count {
            violation = violation.max((observed - max_count as f64).max(0.0));
        }
    }
    violation
}

fn value_count_violation(variables: &[usize], value: i64, count_var: usize, x: &[f64]) -> f64 {
    let mut observed = 0usize;
    let mut violation: f64 = integrality_violation(x[count_var]);
    for &idx in variables {
        let candidate = x[idx];
        violation = violation.max(integrality_violation(candidate));
        if candidate.round() as i64 == value {
            observed += 1;
        }
    }
    violation.max((x[count_var] - observed as f64).abs())
}

fn map_domain_violation(var: usize, bool_vars: &[usize], offset: i64, x: &[f64]) -> f64 {
    let value = x[var];
    let rounded = value.round() as i64;
    let mut violation = integrality_violation(value);
    for (pos, &bool_var) in bool_vars.iter().enumerate() {
        let Some(target) = offset.checked_add(pos as i64) else {
            return 1.0;
        };
        let expected = if rounded == target { 1.0 } else { 0.0 };
        violation = violation
            .max(integrality_violation(x[bool_var]))
            .max((x[bool_var] - expected).abs());
    }
    violation
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

fn forbidden_assignments_violation(variables: &[usize], tuples: &[Vec<i64>], x: &[f64]) -> f64 {
    let matches_forbidden = tuples.iter().any(|tuple| {
        variables
            .iter()
            .zip(tuple)
            .all(|(&idx, &target)| (x[idx] - target as f64).abs() <= 1e-6)
    });
    if matches_forbidden {
        1.0
    } else {
        0.0
    }
}

fn enforced_table_violation(
    enforcement: &[BoolLiteral],
    variables: &[usize],
    tuples: &[Vec<i64>],
    x: &[f64],
    forbidden: bool,
) -> f64 {
    let active = enforcement
        .iter()
        .all(|literal| binary_truth(x[literal.var]) == literal.value);
    if !active {
        return 0.0;
    }
    if forbidden {
        forbidden_assignments_violation(variables, tuples, x)
    } else {
        allowed_assignments_violation(variables, tuples, x)
    }
}

fn bin_packing_violation(
    item_bin_vars: &[usize],
    load_vars: &[usize],
    item_sizes: &[f64],
    x: &[f64],
) -> f64 {
    let mut violation: f64 = 0.0;
    let mut loads = vec![0.0; load_vars.len()];
    for (&item_bin_var, &size) in item_bin_vars.iter().zip(item_sizes) {
        let value = x[item_bin_var];
        let rounded = value.round();
        violation = violation.max((value - rounded).abs());
        if rounded < 0.0 || rounded >= load_vars.len() as f64 {
            violation = violation.max(1.0);
        } else {
            loads[rounded as usize] += size;
        }
    }
    for (bin, &load_var) in load_vars.iter().enumerate() {
        violation = violation.max((x[load_var] - loads[bin]).abs());
    }
    violation
}

fn element_violation(index_var: usize, target_var: usize, values: &[f64], x: &[f64]) -> f64 {
    let index_value = x[index_var];
    let rounded = index_value.round();
    let integrality = integrality_violation(index_value);
    if rounded < 0.0 || rounded >= values.len() as f64 {
        return integrality.max(1.0);
    }
    let expected = values[rounded as usize];
    integrality.max((x[target_var] - expected).abs())
}

fn variable_element_violation(
    index_var: usize,
    target_var: usize,
    variables: &[usize],
    x: &[f64],
) -> f64 {
    let index_value = x[index_var];
    let rounded = index_value.round();
    let integrality = integrality_violation(index_value);
    if rounded < 0.0 || rounded >= variables.len() as f64 {
        return integrality.max(1.0);
    }
    let source_var = variables[rounded as usize];
    integrality.max((x[target_var] - x[source_var]).abs())
}

fn inverse_violation(variables: &[usize], inverse_variables: &[usize], x: &[f64]) -> f64 {
    if variables.len() != inverse_variables.len() {
        return 1.0;
    }
    let n = variables.len();
    let mut violation: f64 = 0.0;
    let mut seen_values = vec![false; n];
    for (i, &var_idx) in variables.iter().enumerate() {
        let value = x[var_idx];
        let rounded = value.round();
        violation = violation.max(integrality_violation(value));
        if rounded < 0.0 || rounded >= n as f64 {
            return violation.max(1.0);
        }
        let j = rounded as usize;
        if seen_values[j] {
            violation = violation.max(1.0);
        }
        seen_values[j] = true;
        violation = violation.max((x[inverse_variables[j]] - i as f64).abs());
    }

    let mut seen_inverse_values = vec![false; n];
    for (j, &inverse_var_idx) in inverse_variables.iter().enumerate() {
        let value = x[inverse_var_idx];
        let rounded = value.round();
        violation = violation.max(integrality_violation(value));
        if rounded < 0.0 || rounded >= n as f64 {
            return violation.max(1.0);
        }
        let i = rounded as usize;
        if seen_inverse_values[i] {
            violation = violation.max(1.0);
        }
        seen_inverse_values[i] = true;
        violation = violation.max((x[variables[i]] - j as f64).abs());
    }

    violation
}

fn circuit_violation(node_count: usize, arcs: &[CircuitArc], x: &[f64]) -> f64 {
    if node_count < 2 {
        return 1.0;
    }

    let mut violation: f64 = 0.0;
    let mut incoming = vec![0usize; node_count];
    let mut outgoing = vec![0usize; node_count];
    let mut next = vec![None; node_count];

    for arc in arcs {
        if arc.tail >= node_count || arc.head >= node_count || arc.tail == arc.head {
            return 1.0;
        }
        let value = x[arc.literal_var];
        violation = violation.max(integrality_violation(value));
        if binary_truth(value) {
            outgoing[arc.tail] += 1;
            incoming[arc.head] += 1;
            if next[arc.tail].replace(arc.head).is_some() {
                violation = violation.max(1.0);
            }
        }
    }

    for node in 0..node_count {
        violation = violation
            .max((outgoing[node] as f64 - 1.0).abs())
            .max((incoming[node] as f64 - 1.0).abs());
    }
    if violation > 0.0 {
        return violation;
    }

    let mut seen = vec![false; node_count];
    let mut current = 0usize;
    for _ in 0..node_count {
        if seen[current] {
            return 1.0;
        }
        seen[current] = true;
        current = match next[current] {
            Some(head) => head,
            None => return 1.0,
        };
    }
    if current != 0 || seen.iter().any(|&visited| !visited) {
        1.0
    } else {
        0.0
    }
}

fn multiple_circuit_violation(node_count: usize, arcs: &[CircuitArc], x: &[f64]) -> f64 {
    if node_count < 2 {
        return 1.0;
    }

    let mut violation: f64 = 0.0;
    let mut incoming = vec![0usize; node_count];
    let mut outgoing = vec![0usize; node_count];
    let mut next = vec![None; node_count];

    for arc in arcs {
        if arc.tail >= node_count || arc.head >= node_count || (arc.tail == 0 && arc.head == 0) {
            return 1.0;
        }
        let value = x[arc.literal_var];
        violation = violation.max(integrality_violation(value));
        if binary_truth(value) {
            outgoing[arc.tail] += 1;
            incoming[arc.head] += 1;
            if arc.tail != arc.head && next[arc.tail].replace(arc.head).is_some() {
                violation = violation.max(1.0);
            }
        }
    }

    violation = violation.max((outgoing[0] as f64 - incoming[0] as f64).abs());
    for node in 1..node_count {
        violation = violation
            .max((outgoing[node] as f64 - 1.0).abs())
            .max((incoming[node] as f64 - 1.0).abs());
    }
    if violation > 0.0 {
        return violation;
    }

    for start in 1..node_count {
        let mut seen = vec![false; node_count];
        let mut current = start;
        loop {
            if current == 0 {
                break;
            }
            if seen[current] {
                return 1.0;
            }
            seen[current] = true;
            current = match next[current] {
                Some(head) => head,
                None => break,
            };
        }
    }

    0.0
}

fn automaton_violation(
    variables: &[usize],
    starting_state: i64,
    final_states: &[i64],
    transitions: &[AutomatonTransition],
    x: &[f64],
) -> f64 {
    let mut current_states = vec![starting_state];
    for &var_idx in variables {
        let label = x[var_idx].round() as i64;
        let mut next_states = transitions
            .iter()
            .filter_map(|transition| {
                (transition.label == label && current_states.contains(&transition.tail))
                    .then_some(transition.head)
            })
            .collect::<Vec<_>>();
        next_states.sort_unstable();
        next_states.dedup();
        if next_states.is_empty() {
            return 1.0;
        }
        current_states = next_states;
    }
    if current_states
        .iter()
        .any(|state| final_states.contains(state))
    {
        0.0
    } else {
        1.0
    }
}

fn automaton_states(
    starting_state: i64,
    final_states: &[i64],
    transitions: &[AutomatonTransition],
) -> Vec<i64> {
    let mut states = Vec::with_capacity(1 + final_states.len() + transitions.len() * 2);
    states.push(starting_state);
    states.extend_from_slice(final_states);
    for transition in transitions {
        states.push(transition.tail);
        states.push(transition.head);
    }
    unique_i64(&states)
}

fn unique_i64(values: &[i64]) -> Vec<i64> {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values
}

fn alternative_violation(master: &IntervalTerm, alternatives: &[IntervalTerm], x: &[f64]) -> f64 {
    let mut violation: f64 = 0.0;
    let master_presence = master.presence_var.map_or(1.0, |presence| {
        violation = violation.max(integrality_violation(x[presence]));
        x[presence]
    });
    if binary_truth(master_presence) {
        violation = violation.max(interval_end_violation(master, x));
    }

    let mut selected = 0.0;
    for alternative in alternatives {
        let Some(presence) = alternative.presence_var else {
            return 1.0;
        };
        let presence_value = x[presence];
        violation = violation.max(integrality_violation(presence_value));
        selected += presence_value;
        if binary_truth(presence_value) {
            violation = violation.max(interval_end_violation(alternative, x));
            violation = violation.max((x[alternative.start_var] - x[master.start_var]).abs());
            violation = violation.max((x[alternative.end_var] - x[master.end_var]).abs());
        }
    }

    violation.max((selected - master_presence).abs())
}

fn rectangle_active(x_interval: &IntervalTerm, y_interval: &IntervalTerm, x: &[f64]) -> bool {
    interval_active(x_interval, x) && interval_active(y_interval, x)
}

fn interval_active(interval: &IntervalTerm, x: &[f64]) -> bool {
    interval
        .presence_var
        .map_or(true, |presence| binary_truth(x[presence]))
}

fn interval_duration_value(interval: &IntervalTerm, x: &[f64]) -> f64 {
    interval.duration
        + interval
            .duration_var
            .map_or(0.0, |duration_var| x[duration_var])
}

fn interval_end_violation(interval: &IntervalTerm, x: &[f64]) -> f64 {
    (x[interval.end_var] - x[interval.start_var] - interval_duration_value(interval, x)).abs()
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
    demands: &[AffineTerm],
    capacity: &AffineTerm,
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
            .map(|(_, demand)| eval_affine_term(demand, x))
            .sum::<f64>();
        let capacity = eval_affine_term(capacity, x);
        violation = violation.max((load - capacity).max(0.0));
    }
    violation
}

fn reservoir_violation(
    events: &[ReservoirEvent],
    min_level: f64,
    max_level: f64,
    x: &[f64],
) -> f64 {
    let mut violation = (min_level - 0.0).max(0.0).max((0.0 - max_level).max(0.0));
    let mut active_events = Vec::new();

    for event in events {
        if let Some(active_var) = event.active_var {
            violation = violation.max(integrality_violation(x[active_var]));
            if !binary_truth(x[active_var]) {
                continue;
            }
        }
        let time_value = x[event.time_var];
        violation = violation.max(integrality_violation(time_value));
        active_events.push((time_value.round() as i64, event.demand));
    }

    if active_events.is_empty() {
        return violation;
    }

    let mut checkpoints = active_events
        .iter()
        .map(|&(time, _)| time)
        .collect::<Vec<_>>();
    checkpoints.sort_unstable();
    checkpoints.dedup();

    for checkpoint in checkpoints {
        let level = active_events
            .iter()
            .filter(|&&(time, _)| time <= checkpoint)
            .map(|&(_, demand)| demand)
            .sum::<f64>();
        violation = violation
            .max((level - max_level).max(0.0))
            .max((min_level - level).max(0.0));
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
        && program.enforced_constraints.is_empty()
        && program.reified_constraints.is_empty()
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

fn finite_integer_linear_bound(
    kind: &str,
    bound_name: &str,
    value: f64,
) -> Result<i64, MathProgramError> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(MathProgramError::InvalidBound(format!(
            "{kind} computed non-finite or out-of-range {bound_name} expression bound {value}"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-6 {
        return Err(MathProgramError::InvalidBound(format!(
            "{kind} computed non-integer {bound_name} expression bound {value}"
        )));
    }
    Ok(rounded as i64)
}

fn normalized_forbidden_intervals(
    intervals: &[LinearDomainInterval],
    domain_lower: i64,
    domain_upper: i64,
) -> Vec<LinearDomainInterval> {
    let mut clipped = intervals
        .iter()
        .filter_map(|interval| {
            let lower = interval.lower.max(domain_lower);
            let upper = interval.upper.min(domain_upper);
            (lower <= upper).then_some(LinearDomainInterval { lower, upper })
        })
        .collect::<Vec<_>>();
    clipped.sort_by_key(|interval| (interval.lower, interval.upper));

    let mut merged: Vec<LinearDomainInterval> = Vec::new();
    for interval in clipped {
        match merged.last_mut() {
            Some(last) if interval.lower <= last.upper.saturating_add(1) => {
                last.upper = last.upper.max(interval.upper);
            }
            _ => merged.push(interval),
        }
    }
    merged
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

fn affine_bounds(program: &MathProgram, term: &AffineTerm) -> Option<(f64, f64)> {
    let (lower, upper) = linear_bounds(program, &term.coeffs)?;
    let lower = lower + term.constant;
    let upper = upper + term.constant;
    if lower.is_finite() && upper.is_finite() {
        Some((lower, upper))
    } else {
        None
    }
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
    program.objective_offset + linear + quadratic
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

fn validate_objective_offset(offset: f64) -> Result<(), MathProgramError> {
    if offset.is_finite() {
        Ok(())
    } else {
        Err(MathProgramError::NonFinite("objective offset".to_string()))
    }
}

fn validate_range_bounds(
    name: &str,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), MathProgramError> {
    if lower.is_none() && upper.is_none() {
        return Err(MathProgramError::InvalidBound(format!(
            "range row `{name}` requires a lower or upper bound"
        )));
    }
    if lower.is_some_and(|value| !value.is_finite())
        || upper.is_some_and(|value| !value.is_finite())
    {
        return Err(MathProgramError::NonFinite(format!(
            "range row `{name}` bound"
        )));
    }
    if let (Some(lo), Some(hi)) = (lower, upper) {
        if lo > hi {
            return Err(MathProgramError::InvalidBound(format!(
                "range row `{name}` has lower bound {lo} above upper bound {hi}"
            )));
        }
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
    use crate::des::general::ip_mip_des::{
        BranchRule, ConcreteLpRelaxationAlgorithm, LpRelaxationAlgorithm, TraceAction,
    };
    use std::process::{Command, Stdio};

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6, "left={a}, right={b}");
    }

    #[test]
    fn math_program_external_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_math_program_external_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
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
        let dual_ub = sol.dual_ub.as_ref().expect("LP inequality duals");
        assert_close(dual_ub[0], 0.0);
        assert_close(dual_ub[1], 2.0);
        let row_certs = map_math_program_lp_row_certificates(&p, &sol).unwrap();
        assert_eq!(row_certs.len(), 2);
        assert_eq!(row_certs[0].source_name, "demand");
        assert_eq!(row_certs[0].source_sense, RowSense::Ge);
        assert_eq!(row_certs[0].generated_sense, RowSense::Le);
        assert_eq!(row_certs[0].generated_row, 0);
        assert_eq!(row_certs[0].multiplier, -1.0);
        assert_close(row_certs[0].generated_dual.unwrap(), dual_ub[0]);
        assert_close(row_certs[0].source_dual.unwrap(), -dual_ub[0]);
        assert_eq!(row_certs[1].source_name, "capacity");
        assert_eq!(row_certs[1].source_sense, RowSense::Le);
        assert_eq!(row_certs[1].generated_row, 1);
        assert_close(row_certs[1].source_dual.unwrap(), dual_ub[1]);
        let reduced_costs = sol.reduced_costs.as_ref().expect("LP reduced costs");
        assert_close(reduced_costs[x], 1.0);
        assert_close(reduced_costs[y], 0.0);
        let variable_certs = map_math_program_lp_variable_certificates(&p, &sol).unwrap();
        assert_eq!(variable_certs.len(), 2);
        assert_eq!(variable_certs[x].name, "x");
        assert_eq!(variable_certs[x].var_type, VariableType::Continuous);
        assert_eq!(variable_certs[x].lb, Some(0.0));
        assert_eq!(variable_certs[x].ub, Some(3.0));
        assert_close(variable_certs[x].value.unwrap(), sol.x[x]);
        assert_close(variable_certs[x].reduced_cost.unwrap(), reduced_costs[x]);
        assert_eq!(variable_certs[y].name, "y");
        assert_eq!(variable_certs[y].ub, None);
        assert_close(variable_certs[y].reduced_cost.unwrap(), reduced_costs[y]);

        let des_sol = solve_math_program(
            &p,
            &MathProgramSolveOptions {
                lp_backend: MathProgramLpBackend::DESSimplex,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(des_sol.status, MathProgramStatus::Optimal);
        assert_close(des_sol.objective, sol.objective);
        assert_close(des_sol.x[x], sol.x[x]);
        assert_close(des_sol.x[y], sol.x[y]);
        let des_dual_ub = des_sol.dual_ub.as_ref().expect("DES LP inequality duals");
        assert_close(des_dual_ub[0], dual_ub[0]);
        assert_close(des_dual_ub[1], dual_ub[1]);
        let des_reduced_costs = des_sol
            .reduced_costs
            .as_ref()
            .expect("DES LP reduced costs");
        assert_close(des_reduced_costs[x], reduced_costs[x]);
        assert_close(des_reduced_costs[y], reduced_costs[y]);
    }

    #[test]
    fn lp_row_certificates_preserve_generated_inequality_then_equality_order() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(3.0))
            .unwrap();
        p.add_constraint("floor", vec![(x, 1.0)], RowSense::Ge, 1.0)
            .unwrap();
        p.add_constraint("balance", vec![(x, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        p.add_constraint("cap", vec![(x, 1.0)], RowSense::Le, 3.0)
            .unwrap();

        let solution = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![2.0],
            objective: 2.0,
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: Some(vec![1.25, 2.5]),
            dual_eq: Some(vec![-3.0]),
            reduced_costs: None,
            var_basis: None,
            row_basis: Some(vec![
                "lower".to_string(),
                "upper".to_string(),
                "basic".to_string(),
            ]),
            unbounded_ray: None,
            infeasibility_certificate: None,
            solver: "synthetic".to_string(),
            message: None,
        };

        let rows = solution.lp_row_certificates(&p).unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.source_name.as_str())
                .collect::<Vec<_>>(),
            vec!["floor", "cap", "balance"]
        );
        assert_eq!(rows[0].source_index, 0);
        assert_eq!(rows[0].generated_row, 0);
        assert_eq!(rows[0].generated_rhs, -1.0);
        assert_eq!(rows[0].generated_dual, Some(1.25));
        assert_eq!(rows[0].source_dual, Some(-1.25));
        assert_eq!(rows[0].basis.as_deref(), Some("lower"));
        assert_eq!(rows[1].source_index, 2);
        assert_eq!(rows[1].generated_row, 1);
        assert_eq!(rows[1].generated_dual, Some(2.5));
        assert_eq!(rows[1].source_dual, Some(2.5));
        assert_eq!(rows[1].basis.as_deref(), Some("upper"));
        assert_eq!(rows[2].source_index, 1);
        assert_eq!(rows[2].generated_row, 2);
        assert_eq!(rows[2].generated_sense, RowSense::Eq);
        assert_eq!(rows[2].generated_dual, Some(-3.0));
        assert_eq!(rows[2].basis.as_deref(), Some("basic"));
    }

    #[test]
    fn lp_variable_certificates_preserve_names_bounds_and_basis_order() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let ship = p
            .add_continuous_var("ship amount", 1.5, Some(0.0), Some(4.0))
            .unwrap();
        let slack = p
            .add_continuous_var("service slack", 0.25, Some(-2.0), None)
            .unwrap();
        p.add_constraint("demand", vec![(ship, 1.0), (slack, 1.0)], RowSense::Eq, 3.0)
            .unwrap();

        let solution = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![3.0, 0.0],
            objective: 4.5,
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: None,
            dual_eq: Some(vec![1.5]),
            reduced_costs: Some(vec![0.0, -1.25]),
            var_basis: Some(vec!["basic".to_string(), "at_lower".to_string()]),
            row_basis: Some(vec!["fixed".to_string()]),
            unbounded_ray: None,
            infeasibility_certificate: None,
            solver: "synthetic".to_string(),
            message: None,
        };

        let vars = solution.lp_variable_certificates(&p).unwrap();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].variable, ship);
        assert_eq!(vars[0].name, "ship amount");
        assert_eq!(vars[0].objective_coefficient, 1.5);
        assert_eq!(vars[0].lb, Some(0.0));
        assert_eq!(vars[0].ub, Some(4.0));
        assert_eq!(vars[0].value, Some(3.0));
        assert_eq!(vars[0].reduced_cost, Some(0.0));
        assert_eq!(vars[0].basis.as_deref(), Some("basic"));
        assert_eq!(vars[1].variable, slack);
        assert_eq!(vars[1].name, "service slack");
        assert_eq!(vars[1].lb, Some(-2.0));
        assert_eq!(vars[1].ub, None);
        assert_eq!(vars[1].value, Some(0.0));
        assert_eq!(vars[1].reduced_cost, Some(-1.25));
        assert_eq!(vars[1].basis.as_deref(), Some("at_lower"));
    }

    #[test]
    fn external_lp_basis_inference_fills_missing_pdlp_style_basis() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_continuous_var("x", 3.0, Some(0.0), None).unwrap();
        let y = p.add_continuous_var("y", 4.0, Some(0.0), None).unwrap();
        p.add_constraint("c0", vec![(x, 1.0), (y, 2.0)], RowSense::Le, 14.0)
            .unwrap();
        p.add_constraint("c1", vec![(x, 3.0), (y, -1.0)], RowSense::Ge, 0.0)
            .unwrap();
        p.add_constraint("c2", vec![(x, 1.0), (y, -1.0)], RowSense::Le, 2.0)
            .unwrap();

        let mut solution = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![5.999999973406001, 3.999999971881202],
            objective: 33.99999980774281,
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: Some(vec![2.333333337179504, 0.0, 0.6666666702143926]),
            dual_eq: Some(Vec::new()),
            reduced_costs: Some(vec![0.0, 0.0]),
            var_basis: Some(vec!["free".to_string(), "free".to_string()]),
            row_basis: Some(vec![
                "free".to_string(),
                "free".to_string(),
                "free".to_string(),
            ]),
            unbounded_ray: None,
            infeasibility_certificate: None,
            solver: "ortools:pdlp".to_string(),
            message: None,
        };

        super::infer_missing_external_lp_basis(&p, &mut solution).unwrap();
        assert_eq!(
            solution.var_basis,
            Some(vec!["basic".to_string(), "basic".to_string()])
        );
        assert_eq!(
            solution.row_basis,
            Some(vec![
                "at_upper".to_string(),
                "basic".to_string(),
                "at_upper".to_string()
            ])
        );
    }

    #[test]
    fn external_lp_basis_inference_leaves_degenerate_rows_unreported() {
        let (var_basis, row_basis) = super::infer_lp_basis_from_complementarity(
            &[1.0],
            None,
            None,
            &[vec![1.0]],
            &[1.0],
            &[0.0],
            &[],
            &[],
            &[],
            &[0.0],
        );
        assert_eq!(var_basis, Some(vec!["basic".to_string()]));
        assert_eq!(row_basis, None);
    }

    #[test]
    fn range_rows_and_objective_offsets_are_facade_level_features() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 2.0, Some(0.0), Some(2.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        p.add_objective_offset(5.5).unwrap();
        let range_rows = p
            .add_range_constraint("throughput", vec![(x, 1.0), (y, 1.0)], Some(2.0), Some(3.0))
            .unwrap();

        assert_eq!(range_rows.len(), 2);
        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.x[y], 1.0);
        assert_close(sol.objective, 10.5);

        let export = export_math_program_cplex_lp(&p).unwrap();
        assert!(export.text.contains("objective_offset"));
        assert!(export.text.contains("objective_offset = 1."));
    }

    #[test]
    fn cplex_lp_export_preserves_continuous_linear_facade() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("profit x", 3.0, Some(0.0), Some(3.0))
            .unwrap();
        let y = p.add_continuous_var("2-y", 2.0, None, None).unwrap();
        p.add_constraint("demand >= row", vec![(x, 1.0), (y, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        p.add_constraint("balance=row", vec![(x, 1.0), (y, -1.0)], RowSense::Eq, 1.0)
            .unwrap();

        let export = export_math_program_cplex_lp(&p).unwrap();
        assert!(!export.is_mip);
        assert_eq!(export.original_variable_count, 2);
        assert_eq!(export.variable_names, vec!["profit_x", "x_2_y"]);
        assert_eq!(export.original_variable_expansions.len(), 2);
        assert_eq!(
            export.original_variable_expansions[x].original_name,
            "profit x"
        );
        assert_eq!(
            export.original_variable_expansions[x].terms[0].exported_name,
            "profit_x"
        );
        assert_eq!(export.original_x(&[2.0, -1.0]).unwrap(), vec![2.0, -1.0]);
        assert_eq!(export.row_mappings.len(), 2);
        assert_eq!(export.row_mappings[0].exported_name, "demand_row");
        assert_eq!(
            export.row_mappings[0].source_kind,
            MathProgramExportRowKind::LinearConstraint
        );
        assert_eq!(export.row_mappings[0].source_index, Some(0));
        assert_eq!(export.row_mappings[0].source_name, "demand >= row");
        assert_eq!(export.row_mappings[0].sense, RowSense::Ge);
        assert_eq!(export.row_mappings[1].exported_name, "balance_row");
        assert!(export.text.contains("Maximize\n"));
        assert!(export.text.contains("Subject To\n"));
        assert!(export.text.contains("Bounds\n"));
        assert!(export.text.contains("demand_row: profit_x + x_2_y >= 2."));
        assert!(export.text.contains("balance_row: profit_x - x_2_y = 1."));
        assert!(export.text.contains("x_2_y free"));
        assert!(export.text.ends_with("End\n"));
    }

    #[test]
    fn cplex_lp_export_preserves_continuous_quadratic_objective() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("qp x", -4.0, Some(0.0), Some(5.0))
            .unwrap();
        let y = p
            .add_continuous_var("qp y", 1.0, Some(-2.0), Some(3.0))
            .unwrap();
        p.add_quadratic_objective_term(x, x, 1.0).unwrap();
        p.add_quadratic_objective_term(x, y, -3.0).unwrap();
        p.add_constraint("qp-balance", vec![(x, 1.0), (y, -1.0)], RowSense::Le, 4.0)
            .unwrap();

        let export = export_math_program_cplex_lp(&p).unwrap();
        assert!(!export.is_mip);
        assert_eq!(export.original_variable_count, 2);
        assert_eq!(export.variable_names, vec!["qp_x", "qp_y"]);
        assert_eq!(export.original_variable_expansions.len(), 2);
        assert_eq!(export.row_mappings.len(), 1);
        assert_eq!(export.row_mappings[0].source_name, "qp-balance");
        assert!(export.text.contains("Minimize\n"));
        assert!(export.text.contains(
            "obj: - 4.00000000000000000e0 qp_x + qp_y + [ 2.00000000000000000e0 qp_x * qp_x - 6.00000000000000000e0 qp_x * qp_y ] / 2"
        ));
        assert!(export.text.contains("qp_balance: qp_x - qp_y <= 4."));
        assert!(export.text.contains("-2.00000000000000000e0 <= qp_y <= 3."));
        assert!(export.text.ends_with("End\n"));

        let mps_export = export_math_program_mps(&p).unwrap();
        assert!(!mps_export.is_mip);
        assert_eq!(mps_export.original_variable_count, 2);
        assert_eq!(mps_export.variable_names, vec!["qp_x", "qp_y"]);
        assert_eq!(mps_export.original_variable_expansions.len(), 2);
        assert_eq!(mps_export.row_mappings.len(), 1);
        assert!(mps_export.text.contains("OBJSENSE\n MIN\n"));
        assert!(mps_export.text.contains("QUADOBJ\n"));
        assert!(mps_export
            .text
            .contains("qp_x  qp_x  2.00000000000000000e0"));
        assert!(mps_export
            .text
            .contains("qp_x  qp_y  -3.00000000000000000e0"));
        assert!(mps_export.text.ends_with("ENDATA\n"));
    }

    #[test]
    fn mps_export_preserves_continuous_quadratic_constraints() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("qcp x", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let y = p
            .add_continuous_var("qcp y", 1.0, Some(0.0), Some(20.0))
            .unwrap();
        p.add_constraint("fix-qcp-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_quadratic_constraint(
            "qcp-epigraph-square",
            vec![(x, x, 1.0)],
            vec![(y, -1.0)],
            RowSense::Le,
            0.0,
        )
        .unwrap();

        let export = export_math_program_mps(&p).unwrap();
        assert!(!export.is_mip);
        assert_eq!(export.original_variable_count, 2);
        assert_eq!(export.variable_names, vec!["qcp_x", "qcp_y"]);
        assert_eq!(export.original_variable_expansions.len(), 2);
        assert_eq!(export.row_mappings.len(), 2);
        assert_eq!(export.row_mappings[0].exported_name, "fix_qcp_x");
        assert_eq!(
            export.row_mappings[0].source_kind,
            MathProgramExportRowKind::LinearConstraint
        );
        assert_eq!(export.row_mappings[1].exported_name, "qcp_epigraph_square");
        assert_eq!(
            export.row_mappings[1].source_kind,
            MathProgramExportRowKind::QuadraticConstraint
        );
        assert_eq!(export.row_mappings[1].source_index, Some(0));
        assert_eq!(export.row_mappings[1].source_name, "qcp-epigraph-square");
        assert!(export.text.contains("OBJSENSE\n MIN\n"));
        assert!(export
            .text
            .contains("ROWS\n N  OBJ\n E  fix_qcp_x\n L  qcp_epigraph_square\n"));
        assert!(export
            .text
            .contains("qcp_epigraph_square  -1.00000000000000000e0"));
        assert!(export.text.contains("QCMATRIX   qcp_epigraph_square\n"));
        assert!(export.text.contains("qcp_x  qcp_x  1.00000000000000000e0"));
        assert!(export.text.contains("BOUNDS\n"));
        assert!(export.text.ends_with("ENDATA\n"));
    }

    #[test]
    fn mps_quadratic_constraint_entries_split_off_diagonal_terms() {
        let terms = vec![
            QuadraticConstraintTerm {
                var_i: 0,
                var_j: 1,
                coeff: 3.0,
            },
            QuadraticConstraintTerm {
                var_i: 1,
                var_j: 0,
                coeff: 1.0,
            },
            QuadraticConstraintTerm {
                var_i: 1,
                var_j: 1,
                coeff: 2.0,
            },
        ];

        let entries = mps_quadratic_constraint_entries(&terms, 2).unwrap();
        assert_eq!(entries.get(&(0, 1)).copied(), Some(2.0));
        assert_eq!(entries.get(&(1, 0)).copied(), Some(2.0));
        assert_eq!(entries.get(&(1, 1)).copied(), Some(2.0));
    }

    #[test]
    fn cplex_lp_export_emits_compiled_mip_with_generated_columns() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let open = p.add_binary_var("open-a", 4.0).unwrap();
        let closed = p.add_binary_var("open b", 3.0).unwrap();
        let load = p
            .add_integer_var("load", 2.0, Some(0.0), Some(4.0))
            .unwrap();
        let reserve = p
            .add_integer_var("reserve", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let peak = p
            .add_continuous_var("peak", -1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_exactly_one("choose-one", vec![open, closed]).unwrap();
        p.add_indicator(
            "open-a-min-load",
            open,
            true,
            vec![(load, 1.0)],
            RowSense::Ge,
            3.0,
        )
        .unwrap();
        p.add_max("peak-load", peak, vec![load, reserve]).unwrap();

        let export = export_math_program_cplex_lp(&p).unwrap();
        assert!(export.is_mip);
        assert_eq!(export.original_variable_count, 5);
        assert!(export.variable_names.len() > export.original_variable_count);
        assert_eq!(export.original_variable_expansions.len(), 5);
        assert_eq!(export.original_variable_expansions[open].constant, 0.0);
        assert_eq!(export.original_variable_expansions[open].terms.len(), 1);
        assert_eq!(export.row_mappings.len(), export.constraint_names.len());
        assert!(export.row_mappings.iter().any(|row| {
            row.source_kind == MathProgramExportRowKind::Generated
                && row.source_name.contains("open-a-min-load__indicator")
                && row.exported_name.contains("open_a_min_load")
        }));
        assert!(export.text.contains("Maximize\n"));
        assert!(export.text.contains("Subject To\n"));
        assert!(export.text.contains("Bounds\n"));
        assert!(export.text.contains("Binaries\n"));
        assert!(export.text.contains("Generals\n"));
        assert!(export.text.contains("open_a"));
        assert!(export.text.contains("peak_load"));
        assert!(export.text.ends_with("End\n"));
    }

    #[test]
    fn mps_export_preserves_continuous_linear_facade() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("profit x", 3.0, Some(0.0), Some(3.0))
            .unwrap();
        let y = p.add_continuous_var("2-y", 2.0, None, None).unwrap();
        p.add_constraint("demand >= row", vec![(x, 1.0), (y, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        p.add_constraint("balance=row", vec![(x, 1.0), (y, -1.0)], RowSense::Eq, 1.0)
            .unwrap();

        let export = export_math_program_mps(&p).unwrap();
        assert!(!export.is_mip);
        assert_eq!(export.original_variable_count, 2);
        assert_eq!(export.variable_names, vec!["profit_x", "x_2_y"]);
        assert_eq!(export.original_variable_expansions.len(), 2);
        assert_eq!(export.original_x(&[2.0, -1.0]).unwrap(), vec![2.0, -1.0]);
        assert_eq!(export.row_mappings.len(), 2);
        assert_eq!(export.row_mappings[0].exported_name, "demand_row");
        assert_eq!(
            export.row_mappings[0].source_kind,
            MathProgramExportRowKind::LinearConstraint
        );
        assert_eq!(export.row_mappings[0].source_index, Some(0));
        assert_eq!(export.row_mappings[0].source_name, "demand >= row");
        assert_eq!(export.row_mappings[0].sense, RowSense::Ge);
        assert!(export.text.starts_with("NAME"));
        assert!(export.text.contains("OBJSENSE\n MAX\n"));
        assert!(export
            .text
            .contains("ROWS\n N  OBJ\n G  demand_row\n E  balance_row\n"));
        assert!(export.text.contains("COLUMNS\n"));
        assert!(export.text.contains("RHS\n"));
        assert!(export.text.contains("UP BND1  profit_x"));
        assert!(export.text.contains("FR BND1  x_2_y"));
        assert!(export.text.ends_with("ENDATA\n"));
    }

    #[test]
    fn mps_export_emits_compiled_mip_with_integer_markers() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let open = p.add_binary_var("open-a", 4.0).unwrap();
        let closed = p.add_binary_var("open b", 3.0).unwrap();
        let load = p
            .add_integer_var("load", 2.0, Some(0.0), Some(4.0))
            .unwrap();
        let reserve = p
            .add_integer_var("reserve", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let peak = p
            .add_continuous_var("peak", -1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_exactly_one("choose-one", vec![open, closed]).unwrap();
        p.add_indicator(
            "open-a-min-load",
            open,
            true,
            vec![(load, 1.0)],
            RowSense::Ge,
            3.0,
        )
        .unwrap();
        p.add_max("peak-load", peak, vec![load, reserve]).unwrap();

        let export = export_math_program_mps(&p).unwrap();
        assert!(export.is_mip);
        assert_eq!(export.original_variable_count, 5);
        assert!(export.variable_names.len() > export.original_variable_count);
        assert_eq!(export.original_variable_expansions.len(), 5);
        assert_eq!(export.row_mappings.len(), export.constraint_names.len());
        assert!(export.row_mappings.iter().any(|row| {
            row.source_kind == MathProgramExportRowKind::Generated
                && row.source_name.contains("open-a-min-load__indicator")
                && row.exported_name.contains("open_a_min_load")
        }));
        assert!(export.text.contains("OBJSENSE\n MAX\n"));
        assert!(export.text.contains("ROWS\n N  OBJ\n"));
        assert!(export.text.contains("'INTORG'"));
        assert!(export.text.contains("'INTEND'"));
        assert!(export.text.contains("BOUNDS\n"));
        assert!(export.text.contains("open_a"));
        assert!(export.text.contains("peak_load"));
        assert!(export.text.ends_with("ENDATA\n"));
    }

    #[test]
    fn linear_exports_map_generated_columns_back_to_original_variables() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let signed = p.add_integer_var("signed", 1.0, None, None).unwrap();
        let shifted = p
            .add_integer_var("shifted", 2.0, Some(3.0), Some(8.0))
            .unwrap();
        let choose = p.add_binary_var("choose", 0.0).unwrap();
        p.add_constraint(
            "cap",
            vec![(signed, 1.0), (shifted, 1.0)],
            RowSense::Le,
            6.0,
        )
        .unwrap();

        let lp_export = export_math_program_cplex_lp(&p).unwrap();
        let mps_export = export_math_program_mps(&p).unwrap();
        assert_eq!(
            lp_export.original_variable_expansions,
            mps_export.original_variable_expansions
        );

        let signed_map = &lp_export.original_variable_expansions[signed];
        assert_eq!(signed_map.original_name, "signed");
        assert_close(signed_map.constant, 0.0);
        assert_eq!(signed_map.terms.len(), 2);
        assert_eq!(signed_map.terms[0].coefficient, 1.0);
        assert_eq!(signed_map.terms[1].coefficient, -1.0);

        let shifted_map = &lp_export.original_variable_expansions[shifted];
        assert_eq!(shifted_map.original_name, "shifted");
        assert_close(shifted_map.constant, 3.0);
        assert_eq!(shifted_map.terms.len(), 1);
        assert_eq!(shifted_map.terms[0].coefficient, 1.0);

        let mut exported_x = vec![0.0; lp_export.variable_names.len()];
        for term in &signed_map.terms {
            exported_x[term.exported_var] = if term.coefficient > 0.0 { 4.0 } else { 2.0 };
        }
        exported_x[shifted_map.terms[0].exported_var] = 2.0;
        let original = lp_export.original_x(&exported_x).unwrap();
        assert_close(original[signed], 2.0);
        assert_close(original[shifted], 5.0);
        assert_close(original[choose], 0.0);
    }

    #[test]
    fn compiled_linear_exports_preserve_shifted_objective_offsets() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let shifted = p
            .add_integer_var("shifted", 2.0, Some(3.0), Some(8.0))
            .unwrap();
        p.add_constraint("fix-shifted", vec![(shifted, 1.0)], RowSense::Eq, 5.0)
            .unwrap();

        let parts = math_program_linear_text_export_parts(&p, "test export", false, false).unwrap();
        let offset_idx = parts
            .variable_names
            .iter()
            .position(|name| name == "objective_offset")
            .unwrap();
        assert_close(parts.objective[offset_idx], 6.0);
        assert_eq!(parts.lower.as_ref().unwrap()[offset_idx], Some(1.0));
        assert_eq!(parts.upper.as_ref().unwrap()[offset_idx], Some(1.0));

        let lp_export = export_math_program_cplex_lp(&p).unwrap();
        assert!(lp_export.is_mip);
        assert!(lp_export.text.contains("objective_offset = 1."));

        let mps_export = export_math_program_mps(&p).unwrap();
        assert!(mps_export.is_mip);
        assert!(mps_export.text.contains(" FX BND1  objective_offset  1"));
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
    fn assumption_core_drops_redundant_binary_literals() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a = p.add_binary_var("assume-a", 0.0).unwrap();
        let b = p.add_binary_var("assume-b", 0.0).unwrap();
        let noise = p.add_binary_var("irrelevant", 0.0).unwrap();
        p.add_constraint(
            "assumptions-at-most-one",
            vec![(a, 1.0), (b, 1.0)],
            RowSense::Le,
            1.0,
        )
        .unwrap();

        let assumptions = vec![
            MathProgram::bool_lit(a),
            MathProgram::bool_lit(b),
            MathProgram::not_lit(noise),
        ];
        let core = find_math_program_assumption_unsat_core(
            &p,
            &assumptions,
            &MathProgramSolveOptions::default(),
            &MathProgramAssumptionCoreOptions::default(),
        )
        .unwrap();

        assert_eq!(core.status, MathProgramStatus::Infeasible);
        assert!(core.minimal);
        assert_eq!(
            core.assumptions,
            vec![MathProgram::bool_lit(a), MathProgram::bool_lit(b)]
        );
        assert_eq!(
            core.subsystem
                .constraints
                .iter()
                .filter(|row| row.name.starts_with("__assumption_"))
                .count(),
            2
        );
        assert!(!core.assumptions.contains(&MathProgram::not_lit(noise)));
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
    fn hierarchical_objective_tolerance_allows_controlled_degradation() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let y = p
            .add_continuous_var("y", 0.0, Some(0.0), Some(10.0))
            .unwrap();
        let z = p
            .add_continuous_var("z", 0.0, Some(0.0), Some(10.0))
            .unwrap();
        p.add_constraint("capacity", vec![(y, 1.0), (z, 1.0)], RowSense::Le, 10.0)
            .unwrap();
        p.add_secondary_objective_with_tolerances(
            "prefer-y-with-degradation",
            ObjectiveSense::Max,
            10,
            1.0,
            1.0,
            0.1,
            vec![(y, 1.0)],
        )
        .unwrap();
        p.add_secondary_objective("prefer-z", ObjectiveSense::Max, 1, 1.0, vec![(z, 1.0)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.objective, 0.0);
        assert_close(sol.x[y], 8.0);
        assert_close(sol.x[z], 2.0);
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
            .is_some_and(|message| message.contains("incumbent_source=user-mip-start")));
    }

    #[test]
    fn branch_priorities_map_from_original_variable_space() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        p.add_binary_var("binary", 0.0).unwrap();
        p.add_integer_var("free", 0.0, None, None).unwrap();
        p.add_semi_integer_var("semi", 0.0, 3.0, 7.0).unwrap();

        let compiled = compile_mip(&p).unwrap();
        let priorities = canonical_branch_priorities(&p, &compiled, &[1, 7, 4]).unwrap();
        let names = compiled.problem.var_names.as_ref().unwrap();
        let var_idx = |name: &str| {
            names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap()
        };

        assert_eq!(priorities[var_idx("binary")], 1);
        assert_eq!(priorities[var_idx("free__pos")], 7);
        assert_eq!(priorities[var_idx("free__neg")], 7);
        assert_eq!(priorities[var_idx("semi")], 4);
        assert_eq!(priorities[var_idx("semi__active")], 4);
    }

    #[test]
    fn branch_priorities_reach_native_first_branch_rule() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let low = p.add_binary_var("low_priority", 1.0).unwrap();
        let high = p.add_binary_var("high_priority", 1.0).unwrap();
        p.add_constraint("cap_low", vec![(low, 1.0)], RowSense::Le, 0.5)
            .unwrap();
        p.add_constraint("cap_high", vec![(high, 1.0)], RowSense::Le, 0.5)
            .unwrap();

        let compiled = compile_mip(&p).unwrap();
        let high_compiled = compiled_var_index(&compiled, "high_priority").unwrap();
        let mip_opts = compiled_mip_options(
            &p,
            &compiled,
            &MathProgramSolveOptions {
                branch_priorities: Some(vec![0, 10]),
                mip: IPMIPSolveOptions {
                    branch_rule: Some(BranchRule::FirstFractional),
                    max_cut_rounds: Some(0),
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
            true,
        )
        .unwrap();

        let sol = solve_ipmip_with_des(compiled.problem, mip_opts);
        let first_branch = sol
            .trace
            .iter()
            .find(|event| event.action == TraceAction::Branch)
            .expect("branch event");

        assert_eq!(from_ipmip_status(sol.status), MathProgramStatus::Optimal);
        assert_eq!(first_branch.branch_var, Some(high_compiled));
    }

    #[test]
    fn branch_priorities_reject_bad_original_length() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        p.add_binary_var("x", 1.0).unwrap();
        p.add_binary_var("y", 1.0).unwrap();

        let err = solve_math_program(
            &p,
            &MathProgramSolveOptions {
                branch_priorities: Some(vec![10]),
                ..Default::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            MathProgramError::BadIndex(message)
                if message.contains("branch priorities length 1 does not match 2 variables")
        ));
    }

    #[test]
    fn lazy_constraint_cuts_integer_candidate() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_binary_var("x", 1.0).unwrap();
        let y = p.add_binary_var("y", 1.0).unwrap();
        p.add_lazy_constraint("at-most-one", vec![(x, 1.0), (y, 1.0)], RowSense::Le, 1.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();

        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.objective, 1.0);
        assert!(sol.x[x] + sol.x[y] <= 1.0 + 1e-6);
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
    fn inactive_value_indicator_lowers_with_big_m() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        let b = p.add_binary_var("b", -20.0).unwrap();
        p.add_indicator(
            "cap-when-false",
            b,
            false,
            vec![(x, 1.0)],
            RowSense::Le,
            2.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn enforced_linear_constraint_uses_literal_conjunction() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        let a = p.add_binary_var("a", 0.0).unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_enforced_constraint(
            "missed-literal-does-not-cap",
            vec![MathProgram::bool_lit(a), MathProgram::not_lit(b)],
            vec![(x, 1.0)],
            RowSense::Le,
            2.0,
        )
        .unwrap();
        p.add_enforced_constraint(
            "all-literals-cap",
            vec![MathProgram::bool_lit(a), MathProgram::bool_lit(b)],
            vec![(x, 1.0)],
            RowSense::Le,
            7.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 7.0);
    }

    #[test]
    fn reified_linear_constraint_links_literal_to_integer_threshold() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(5.0)).unwrap();
        let b = p.add_binary_var("b", 10.0).unwrap();
        p.add_reified_le_constraint("small-x", MathProgram::bool_lit(b), vec![(x, 1.0)], 2.0)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.objective, 12.0);

        let mut forced_large = MathProgram::new(ObjectiveSense::Max);
        let x = forced_large
            .add_integer_var("x", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        let b = forced_large.add_binary_var("b", 0.0).unwrap();
        forced_large
            .add_constraint("force-large", vec![(x, 1.0)], RowSense::Ge, 3.0)
            .unwrap();
        forced_large
            .add_reified_le_constraint("not-small-x", MathProgram::not_lit(b), vec![(x, 1.0)], 2.0)
            .unwrap();

        let sol = solve_math_program(&forced_large, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 5.0);

        let mut ge = MathProgram::new(ObjectiveSense::Max);
        let x = ge.add_integer_var("x", -1.0, Some(0.0), Some(5.0)).unwrap();
        let b = ge.add_binary_var("b", 10.0).unwrap();
        ge.add_reified_ge_constraint("large-x", MathProgram::bool_lit(b), vec![(x, 1.0)], 3.0)
            .unwrap();

        let sol = solve_math_program(&ge, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.objective, 7.0);
    }

    #[test]
    fn reified_linear_equality_links_literal_to_integer_row() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(5.0)).unwrap();
        let b = p.add_binary_var("b", 10.0).unwrap();
        p.add_reified_eq_constraint(
            "x-equals-three",
            MathProgram::bool_lit(b),
            vec![(x, 1.0)],
            3.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[x], 3.0);
        assert_close(sol.objective, 13.0);

        let mut forced_away = MathProgram::new(ObjectiveSense::Max);
        let x = forced_away
            .add_integer_var("x", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        let b = forced_away.add_binary_var("b", 10.0).unwrap();
        forced_away
            .add_constraint("force-away", vec![(x, 1.0)], RowSense::Ge, 4.0)
            .unwrap();
        forced_away
            .add_reified_eq_constraint(
                "not-x-equals-three",
                MathProgram::bool_lit(b),
                vec![(x, 1.0)],
                3.0,
            )
            .unwrap();

        let sol = solve_math_program(&forced_away, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[x], 5.0);

        let mut negated = MathProgram::new(ObjectiveSense::Max);
        let x = negated
            .add_integer_var("x", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let b = negated.add_binary_var("b", 1.0).unwrap();
        negated
            .add_constraint("force-two", vec![(x, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        negated
            .add_reified_eq_constraint(
                "not-b-iff-two",
                MathProgram::not_lit(b),
                vec![(x, 1.0)],
                2.0,
            )
            .unwrap();

        let sol = solve_math_program(&negated, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[x], 2.0);
    }

    #[test]
    fn reified_linear_domain_links_literal_to_interval_membership() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(5.0)).unwrap();
        let in_domain = p.add_binary_var("in-domain", 10.0).unwrap();
        let forced_x = p
            .add_integer_var("forced-x", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let forced_in_domain = p.add_binary_var("forced-in-domain", 7.0).unwrap();
        let outside_x = p
            .add_integer_var("outside-x", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        let outside_domain = p.add_binary_var("outside-domain", 10.0).unwrap();

        p.add_reified_linear_domain(
            "x-in-domain",
            MathProgram::bool_lit(in_domain),
            vec![(x, 1.0)],
            vec![(1, 2), (4, 4)],
        )
        .unwrap();
        p.add_constraint(
            "force-outside-domain",
            vec![(forced_x, 1.0)],
            RowSense::Eq,
            5.0,
        )
        .unwrap();
        p.add_reified_linear_domain(
            "forced-x-in-domain",
            MathProgram::bool_lit(forced_in_domain),
            vec![(forced_x, 1.0)],
            vec![(1, 2), (4, 4)],
        )
        .unwrap();
        p.add_constraint("outside-x-cap", vec![(outside_x, 1.0)], RowSense::Le, 4.0)
            .unwrap();
        p.add_reified_linear_not_in_domain(
            "outside-x-not-in-domain",
            MathProgram::bool_lit(outside_domain),
            vec![(outside_x, 1.0)],
            vec![(1, 3)],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 4.0);
        assert_close(sol.x[in_domain], 1.0);
        assert_close(sol.x[forced_x], 5.0);
        assert_close(sol.x[forced_in_domain], 0.0);
        assert_close(sol.x[outside_x], 4.0);
        assert_close(sol.x[outside_domain], 1.0);
        assert_close(sol.objective, 28.0);
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
    fn binary_xor_general_constraint_tracks_parity() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 0.0).unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        let c = p.add_binary_var("c", 0.0).unwrap();
        let odd = p.add_binary_var("odd", 1.0).unwrap();
        p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-b", vec![(b, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-c-off", vec![(c, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_binary_xor("odd-count", odd, vec![a, b, c]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[c], 0.0);
        assert_close(sol.x[odd], 0.0);
        assert_close(sol.objective, 0.0);
    }

    #[test]
    fn boolean_logic_general_constraints_accept_negated_literals() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 0.0).unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        let c = p.add_binary_var("c", 0.0).unwrap();
        let signed_and = p.add_binary_var("signed-and", 2.0).unwrap();
        let signed_or = p.add_binary_var("signed-or", 3.0).unwrap();
        let signed_xor = p.add_binary_var("signed-xor", 5.0).unwrap();
        p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-b-off", vec![(b, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_constraint("force-c", vec![(c, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_boolean_and(
            "signed-and-active",
            signed_and,
            vec![
                MathProgram::bool_lit(a),
                MathProgram::not_lit(b),
                MathProgram::bool_lit(c),
            ],
        )
        .unwrap();
        p.add_boolean_or(
            "signed-or-false",
            signed_or,
            vec![MathProgram::not_lit(a), MathProgram::bool_lit(b)],
        )
        .unwrap();
        p.add_boolean_xor(
            "signed-xor-odd",
            signed_xor,
            vec![
                MathProgram::bool_lit(a),
                MathProgram::not_lit(b),
                MathProgram::bool_lit(c),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[c], 1.0);
        assert_close(sol.x[signed_and], 1.0);
        assert_close(sol.x[signed_or], 0.0);
        assert_close(sol.x[signed_xor], 1.0);
        assert_close(sol.objective, 7.0);
    }

    #[test]
    fn signed_literal_implication_and_equivalence_lower_to_clauses() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 0.0).unwrap();
        let b = p.add_binary_var("b", 0.0).unwrap();
        let blocked = p.add_binary_var("blocked", 9.0).unwrap();
        let equivalent = p.add_binary_var("equivalent", 0.0).unwrap();
        p.add_constraint("force-a", vec![(a, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint("force-b-off", vec![(b, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_literal_implication(
            "not-b-implies-not-blocked",
            MathProgram::not_lit(b),
            MathProgram::not_lit(blocked),
        )
        .unwrap();
        p.add_literal_equivalence(
            "a-equivalent-equivalent",
            MathProgram::bool_lit(a),
            MathProgram::bool_lit(equivalent),
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[blocked], 0.0);
        assert_close(sol.x[equivalent], 1.0);
        assert_close(sol.objective, 0.0);
    }

    #[test]
    fn binary_cardinality_constraints_bound_selected_count() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 5.0).unwrap();
        let b = p.add_binary_var("b", 4.0).unwrap();
        let c = p.add_binary_var("c", 3.0).unwrap();
        let d = p.add_binary_var("d", 2.0).unwrap();
        p.add_at_most_one("at-most-one-ab", vec![a, b]).unwrap();
        p.add_at_least_one("at-least-one-cd", vec![c, d]).unwrap();
        p.add_exactly_k("exactly-two-total", vec![a, b, c, d], 2)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[c], 1.0);
        assert_close(sol.x[d], 0.0);
        assert_close(sol.objective, 8.0);
    }

    #[test]
    fn boolean_cardinality_counts_negated_literals() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 10.0).unwrap();
        let b = p.add_binary_var("b", -3.0).unwrap();
        let c = p.add_binary_var("c", -4.0).unwrap();
        let d = p.add_binary_var("d", 4.0).unwrap();
        p.add_literal_exactly_k(
            "signed-exactly-two",
            vec![
                MathProgram::bool_lit(a),
                MathProgram::not_lit(b),
                MathProgram::bool_lit(c),
                MathProgram::not_lit(d),
            ],
            2,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[c], 0.0);
        assert_close(sol.x[d], 1.0);
        assert_close(sol.objective, 14.0);
    }

    #[test]
    fn enforced_boolean_helpers_respect_active_and_inactive_gates() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let gate = p.add_binary_var("gate", 0.0).unwrap();
        let inactive_gate = p.add_binary_var("inactive-gate", 0.0).unwrap();
        let and_a = p.add_binary_var("and-a", 5.0).unwrap();
        let and_b = p.add_binary_var("and-b", 4.0).unwrap();
        let or_a = p.add_binary_var("or-a", 3.0).unwrap();
        let or_b = p.add_binary_var("or-b", 2.0).unwrap();
        let xor_a = p.add_binary_var("xor-a", 7.0).unwrap();
        let xor_b = p.add_binary_var("xor-b", 6.0).unwrap();
        let one_a = p.add_binary_var("one-a", 8.0).unwrap();
        let one_b = p.add_binary_var("one-b", 1.0).unwrap();
        let inactive_a = p.add_binary_var("inactive-a", 2.0).unwrap();
        let inactive_b = p.add_binary_var("inactive-b", 1.0).unwrap();

        p.add_constraint("force-gate", vec![(gate, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_constraint(
            "force-inactive-gate",
            vec![(inactive_gate, 1.0)],
            RowSense::Eq,
            0.0,
        )
        .unwrap();
        p.add_enforced_boolean_and(
            "active-and",
            vec![MathProgram::bool_lit(gate)],
            vec![MathProgram::bool_lit(and_a), MathProgram::not_lit(and_b)],
        )
        .unwrap();
        p.add_enforced_boolean_or(
            "active-or",
            vec![MathProgram::bool_lit(gate)],
            vec![MathProgram::bool_lit(or_a), MathProgram::bool_lit(or_b)],
        )
        .unwrap();
        p.add_enforced_boolean_xor(
            "active-xor",
            vec![MathProgram::bool_lit(gate)],
            vec![MathProgram::bool_lit(xor_a), MathProgram::bool_lit(xor_b)],
        )
        .unwrap();
        p.add_enforced_literal_exactly_one(
            "active-exactly-one",
            vec![MathProgram::bool_lit(gate)],
            vec![MathProgram::bool_lit(one_a), MathProgram::bool_lit(one_b)],
        )
        .unwrap();
        p.add_enforced_literal_at_most_one(
            "inactive-at-most-one",
            vec![MathProgram::bool_lit(inactive_gate)],
            vec![
                MathProgram::bool_lit(inactive_a),
                MathProgram::bool_lit(inactive_b),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[gate], 1.0);
        assert_close(sol.x[inactive_gate], 0.0);
        assert_close(sol.x[and_a], 1.0);
        assert_close(sol.x[and_b], 0.0);
        assert_close(sol.x[or_a], 1.0);
        assert_close(sol.x[or_b], 1.0);
        assert_close(sol.x[xor_a], 1.0);
        assert_close(sol.x[xor_b], 0.0);
        assert_close(sol.x[one_a], 1.0);
        assert_close(sol.x[one_b], 0.0);
        assert_close(sol.x[inactive_a], 1.0);
        assert_close(sol.x[inactive_b], 1.0);
        assert_close(sol.objective, 28.0);
    }

    #[test]
    fn boolean_count_links_signed_literals_to_count_variable() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 5.0).unwrap();
        let b = p.add_binary_var("b", -2.0).unwrap();
        let c = p.add_binary_var("c", 3.0).unwrap();
        let count = p
            .add_integer_var("signed-count", 4.0, Some(0.0), Some(3.0))
            .unwrap();
        p.add_boolean_count(
            "signed-count-link",
            vec![
                MathProgram::bool_lit(a),
                MathProgram::not_lit(b),
                MathProgram::bool_lit(c),
            ],
            count,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 0.0);
        assert_close(sol.x[c], 1.0);
        assert_close(sol.x[count], 3.0);
        assert_close(sol.objective, 20.0);
    }

    #[test]
    fn boolean_clause_allows_negated_literals() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let a = p.add_binary_var("a", 4.0).unwrap();
        let b = p.add_binary_var("b", 3.0).unwrap();
        let c = p.add_binary_var("c", 2.0).unwrap();
        p.add_binary_implication("a-implies-b", a, b).unwrap();
        p.add_binary_implication("b-implies-c", b, c).unwrap();
        p.add_boolean_clause(
            "choose-something",
            vec![
                MathProgram::bool_lit(a),
                MathProgram::bool_lit(b),
                MathProgram::bool_lit(c),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a], 1.0);
        assert_close(sol.x[b], 1.0);
        assert_close(sol.x[c], 1.0);
        assert_close(sol.objective, 9.0);
    }

    #[test]
    fn integer_product_matches_finite_domain_operands() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p.add_integer_var("x", 0.0, Some(0.0), Some(3.0)).unwrap();
        let y = p.add_integer_var("y", 0.0, Some(0.0), Some(3.0)).unwrap();
        let product = p
            .add_integer_var("product", 1.0, Some(0.0), Some(9.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_multiplication_equality("x-times-y", product, vec![x, y])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.x[y], 3.0);
        assert_close(sol.x[product], 6.0);
        assert_close(sol.objective, 6.0);
    }

    #[test]
    fn integer_division_and_modulo_use_truncating_semantics() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let numerator = p
            .add_integer_var("numerator", 0.0, Some(-8.0), Some(8.0))
            .unwrap();
        let denominator = p
            .add_integer_var("denominator", 0.0, Some(1.0), Some(4.0))
            .unwrap();
        let quotient = p
            .add_integer_var("quotient", 1.0, Some(-8.0), Some(8.0))
            .unwrap();
        let remainder = p
            .add_integer_var("remainder", 1.0, Some(-4.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-numerator", vec![(numerator, 1.0)], RowSense::Eq, -7.0)
            .unwrap();
        p.add_constraint(
            "fix-denominator",
            vec![(denominator, 1.0)],
            RowSense::Eq,
            3.0,
        )
        .unwrap();
        p.add_division_equality("divide", quotient, numerator, denominator)
            .unwrap();
        p.add_modulo_equality("modulo", remainder, numerator, denominator)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[numerator], -7.0);
        assert_close(sol.x[denominator], 3.0);
        assert_close(sol.x[quotient], -2.0);
        assert_close(sol.x[remainder], -1.0);
        assert_close(sol.objective, -3.0);
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
    fn max_general_constraint_stays_exact_when_result_is_rewarded() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
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
    fn l1_norm_general_constraint_sums_absolute_values() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let norm = p
            .add_continuous_var("norm", 1.0, Some(0.0), Some(8.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -2.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_l1_norm("l1", norm, vec![x, y]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[norm], 5.0);
        assert_close(sol.objective, 5.0);
    }

    #[test]
    fn l0_norm_counts_nonzero_integer_operands() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 10.0, Some(-1.0), Some(1.0)).unwrap();
        let y = p.add_integer_var("y", -5.0, Some(0.0), Some(1.0)).unwrap();
        let z = p.add_integer_var("z", 2.0, Some(0.0), Some(2.0)).unwrap();
        let norm = p.add_integer_var("l0", 4.0, Some(0.0), Some(3.0)).unwrap();
        p.add_l0_norm("l0-count", norm, vec![x, y, z]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 1.0);
        assert_close(sol.x[y], 0.0);
        assert_close(sol.x[z], 2.0);
        assert_close(sol.x[norm], 2.0);
        assert_close(sol.objective, 22.0);
    }

    #[test]
    fn l_infinity_norm_general_constraint_takes_largest_absolute_value() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let norm = p
            .add_continuous_var("norm", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -2.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_l_infinity_norm("linf", norm, vec![x, y]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[norm], 3.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn l_infinity_norm_stays_exact_when_result_is_rewarded() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p
            .add_continuous_var("x", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 0.0, Some(-4.0), Some(4.0))
            .unwrap();
        let norm = p
            .add_continuous_var("norm", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, -2.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_l_infinity_norm("linf", norm, vec![x, y]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[norm], 3.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn l2_norm_epigraph_is_tight_when_minimized() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let x = p
            .add_continuous_var("x", 0.0, Some(-5.0), Some(5.0))
            .unwrap();
        let y = p
            .add_continuous_var("y", 0.0, Some(-5.0), Some(5.0))
            .unwrap();
        let norm = p
            .add_continuous_var("norm", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        p.add_constraint("fix-x", vec![(x, 1.0)], RowSense::Eq, 3.0)
            .unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 4.0)
            .unwrap();
        p.add_l2_norm("l2", norm, vec![x, y]).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[norm], 5.0);
        assert_close(sol.objective, 5.0);
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
    fn univariate_function_helpers_lower_to_piecewise_linear() {
        let mut p = MathProgram::new(ObjectiveSense::Min);

        let square_x = p
            .add_continuous_var("square-x", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let square_y = p
            .add_continuous_var("square-y", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        p.add_constraint("fix-square-x", vec![(square_x, 1.0)], RowSense::Eq, 1.5)
            .unwrap();
        p.add_power_function(
            "square-function",
            square_x,
            square_y,
            2.0,
            vec![0.0, 1.0, 2.0],
        )
        .unwrap();

        let exp_x = p
            .add_continuous_var("exp-x", 0.0, Some(0.0), Some(1.0))
            .unwrap();
        let exp_y = p
            .add_continuous_var("exp-y", 1.0, Some(1.0), Some(3.0))
            .unwrap();
        p.add_constraint("fix-exp-x", vec![(exp_x, 1.0)], RowSense::Eq, 0.5)
            .unwrap();
        p.add_exp_function("exp-function", exp_x, exp_y, vec![0.0, 0.5, 1.0])
            .unwrap();

        let log_x = p
            .add_continuous_var("log-x", 0.0, Some(1.0), Some(4.0))
            .unwrap();
        let log_y = p
            .add_continuous_var("log-y", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        p.add_constraint("fix-log-x", vec![(log_x, 1.0)], RowSense::Eq, 2.0)
            .unwrap();
        p.add_log_function("log-function", log_x, log_y, vec![1.0, 2.0, 4.0])
            .unwrap();

        let logistic_x = p
            .add_continuous_var("logistic-x", 0.0, Some(-2.0), Some(2.0))
            .unwrap();
        let logistic_y = p
            .add_continuous_var("logistic-y", 1.0, Some(0.0), Some(1.0))
            .unwrap();
        p.add_constraint("fix-logistic-x", vec![(logistic_x, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_logistic_function(
            "logistic-function",
            logistic_x,
            logistic_y,
            vec![-2.0, 0.0, 2.0],
        )
        .unwrap();

        let trig_x = p
            .add_continuous_var("trig-x", 0.0, Some(-1.0), Some(1.0))
            .unwrap();
        let sin_y = p
            .add_continuous_var("sin-y", 1.0, Some(-1.0), Some(1.0))
            .unwrap();
        let cos_y = p
            .add_continuous_var("cos-y", 1.0, Some(-1.0), Some(1.0))
            .unwrap();
        let tan_y = p
            .add_continuous_var("tan-y", 1.0, Some(-2.0), Some(2.0))
            .unwrap();
        p.add_constraint("fix-trig-x", vec![(trig_x, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_sin_function("sin-function", trig_x, sin_y, vec![-1.0, 0.0, 1.0])
            .unwrap();
        p.add_cos_function("cos-function", trig_x, cos_y, vec![-1.0, 0.0, 1.0])
            .unwrap();
        p.add_tan_function("tan-function", trig_x, tan_y, vec![-1.0, 0.0, 1.0])
            .unwrap();

        let generic_x = p
            .add_continuous_var("generic-x", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let generic_y = p
            .add_continuous_var("generic-y", 1.0, Some(1.0), Some(5.0))
            .unwrap();
        p.add_constraint("fix-generic-x", vec![(generic_x, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_univariate_piecewise_function(
            "generic-function",
            generic_x,
            generic_y,
            vec![0.0, 1.0, 2.0],
            |x| x + 1.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[square_y], 2.5);
        assert_close(sol.x[exp_y], 0.5_f64.exp());
        assert_close(sol.x[log_y], 2.0_f64.ln());
        assert_close(sol.x[logistic_y], 0.5);
        assert_close(sol.x[sin_y], 0.0);
        assert_close(sol.x[cos_y], 1.0);
        assert_close(sol.x[tan_y], 0.0);
        assert_close(sol.x[generic_y], 2.0);
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
    fn global_cardinality_lowers_to_value_count_bounds() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p
            .add_integer_var("x0", 100.0, Some(0.0), Some(2.0))
            .unwrap();
        let x1 = p.add_integer_var("x1", 10.0, Some(0.0), Some(2.0)).unwrap();
        let x2 = p.add_integer_var("x2", 1.0, Some(0.0), Some(2.0)).unwrap();
        let x3 = p.add_integer_var("x3", 0.0, Some(0.0), Some(2.0)).unwrap();
        p.add_global_cardinality(
            "counts",
            vec![x0, x1, x2, x3],
            vec![
                GlobalCardinalityCount::exact(2, 1),
                GlobalCardinalityCount::exact(1, 2),
                GlobalCardinalityCount::range(0, None, Some(1)),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x0], 2.0);
        assert_close(sol.x[x1], 1.0);
        assert_close(sol.x[x2], 1.0);
        assert_close(sol.x[x3], 0.0);
        assert_close(sol.objective, 211.0);
    }

    #[test]
    fn value_count_lowers_to_count_variable() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p
            .add_integer_var("x0", 100.0, Some(0.0), Some(2.0))
            .unwrap();
        let x1 = p.add_integer_var("x1", 10.0, Some(0.0), Some(2.0)).unwrap();
        let x2 = p.add_integer_var("x2", 1.0, Some(0.0), Some(2.0)).unwrap();
        let count_one = p
            .add_integer_var("count-one", 50.0, Some(0.0), Some(2.0))
            .unwrap();
        p.add_value_count("count-ones", vec![x0, x1, x2], 1, count_one)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x0], 2.0);
        assert_close(sol.x[x1], 1.0);
        assert_close(sol.x[x2], 1.0);
        assert_close(sol.x[count_one], 2.0);
        assert_close(sol.objective, 311.0);
    }

    #[test]
    fn map_domain_links_integer_value_to_selectors() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let mode = p
            .add_integer_var("mode", 0.0, Some(5.0), Some(7.0))
            .unwrap();
        let is_five = p.add_binary_var("is-five", 1.0).unwrap();
        let is_six = p.add_binary_var("is-six", 10.0).unwrap();
        let is_seven = p.add_binary_var("is-seven", 2.0).unwrap();
        p.add_map_domain("mode-selectors", mode, vec![is_five, is_six, is_seven], 5)
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[mode], 6.0);
        assert_close(sol.x[is_five], 0.0);
        assert_close(sol.x[is_six], 1.0);
        assert_close(sol.x[is_seven], 0.0);
        assert_close(sol.objective, 10.0);

        let mut outside = MathProgram::new(ObjectiveSense::Max);
        let outside_mode = outside
            .add_integer_var("outside-mode", 0.0, Some(4.0), Some(7.0))
            .unwrap();
        let outside_five = outside.add_binary_var("outside-five", 1.0).unwrap();
        let outside_six = outside.add_binary_var("outside-six", 1.0).unwrap();
        let outside_seven = outside.add_binary_var("outside-seven", 1.0).unwrap();
        outside
            .add_constraint(
                "force-outside",
                vec![(outside_mode, 1.0)],
                RowSense::Eq,
                4.0,
            )
            .unwrap();
        outside
            .add_map_domain(
                "outside-selectors",
                outside_mode,
                vec![outside_five, outside_six, outside_seven],
                5,
            )
            .unwrap();

        let outside_sol =
            solve_math_program(&outside, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(outside_sol.status, MathProgramStatus::Optimal);
        assert_close(outside_sol.x[outside_mode], 4.0);
        assert_close(outside_sol.x[outside_five], 0.0);
        assert_close(outside_sol.x[outside_six], 0.0);
        assert_close(outside_sol.x[outside_seven], 0.0);
        assert_close(outside_sol.objective, 0.0);
    }

    #[test]
    fn linear_domain_restricts_integer_expression_to_interval_union() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(5.0)).unwrap();
        let y = p.add_integer_var("y", 1.0, Some(0.0), Some(5.0)).unwrap();
        p.add_constraint("fix-y", vec![(y, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_linear_domain("x-domain", vec![(x, 1.0), (y, 1.0)], vec![(1, 2), (4, 4)])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 4.0);
        assert_close(sol.x[y], 0.0);
        assert_close(sol.objective, 4.0);
    }

    #[test]
    fn enforced_linear_domain_only_applies_when_literals_are_active() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let active_gate = p.add_binary_var("active-gate", 0.0).unwrap();
        let active_blocker = p.add_binary_var("active-blocker", 0.0).unwrap();
        let active_x = p
            .add_integer_var("active-x", 10.0, Some(0.0), Some(5.0))
            .unwrap();
        let inactive_gate = p.add_binary_var("inactive-gate", 0.0).unwrap();
        let inactive_x = p
            .add_integer_var("inactive-x", 1.0, Some(0.0), Some(5.0))
            .unwrap();

        p.add_constraint(
            "force-active-gate",
            vec![(active_gate, 1.0)],
            RowSense::Eq,
            1.0,
        )
        .unwrap();
        p.add_constraint(
            "force-active-blocker",
            vec![(active_blocker, 1.0)],
            RowSense::Eq,
            0.0,
        )
        .unwrap();
        p.add_constraint(
            "force-inactive-gate",
            vec![(inactive_gate, 1.0)],
            RowSense::Eq,
            0.0,
        )
        .unwrap();

        p.add_enforced_linear_domain(
            "active-domain",
            vec![
                MathProgram::bool_lit(active_gate),
                MathProgram::not_lit(active_blocker),
            ],
            vec![(active_x, 1.0)],
            vec![(1, 2), (4, 4)],
        )
        .unwrap();
        p.add_enforced_linear_domain(
            "inactive-domain",
            vec![MathProgram::bool_lit(inactive_gate)],
            vec![(inactive_x, 1.0)],
            vec![(1, 2), (4, 4)],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[active_gate], 1.0);
        assert_close(sol.x[active_blocker], 0.0);
        assert_close(sol.x[active_x], 4.0);
        assert_close(sol.x[inactive_gate], 0.0);
        assert_close(sol.x[inactive_x], 5.0);
        assert_close(sol.objective, 45.0);
    }

    #[test]
    fn linear_not_equal_helpers_exclude_forbidden_integer_values() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(4.0)).unwrap();
        let active_gate = p.add_binary_var("active-gate", 0.0).unwrap();
        let active_y = p
            .add_integer_var("active-y", 1.0, Some(0.0), Some(4.0))
            .unwrap();
        let inactive_gate = p.add_binary_var("inactive-gate", 0.0).unwrap();
        let inactive_z = p
            .add_integer_var("inactive-z", 1.0, Some(0.0), Some(4.0))
            .unwrap();

        p.add_constraint("cap-x", vec![(x, 1.0)], RowSense::Le, 3.0)
            .unwrap();
        p.add_linear_not_equal("x-not-three", vec![(x, 1.0)], 3)
            .unwrap();
        p.add_constraint(
            "force-active-gate",
            vec![(active_gate, 1.0)],
            RowSense::Eq,
            1.0,
        )
        .unwrap();
        p.add_constraint("cap-active-y", vec![(active_y, 1.0)], RowSense::Le, 2.0)
            .unwrap();
        p.add_enforced_linear_not_equal(
            "active-y-not-two",
            vec![MathProgram::bool_lit(active_gate)],
            vec![(active_y, 1.0)],
            2,
        )
        .unwrap();
        p.add_constraint(
            "force-inactive-gate",
            vec![(inactive_gate, 1.0)],
            RowSense::Eq,
            0.0,
        )
        .unwrap();
        p.add_constraint("cap-inactive-z", vec![(inactive_z, 1.0)], RowSense::Le, 2.0)
            .unwrap();
        p.add_enforced_linear_not_equal(
            "inactive-z-not-two",
            vec![MathProgram::bool_lit(inactive_gate)],
            vec![(inactive_z, 1.0)],
            2,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 2.0);
        assert_close(sol.x[active_gate], 1.0);
        assert_close(sol.x[active_y], 1.0);
        assert_close(sol.x[inactive_gate], 0.0);
        assert_close(sol.x[inactive_z], 2.0);
        assert_close(sol.objective, 5.0);
    }

    #[test]
    fn linear_not_in_domain_helpers_exclude_forbidden_interval_unions() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_integer_var("x", 1.0, Some(0.0), Some(6.0)).unwrap();
        let active_gate = p.add_binary_var("active-gate", 0.0).unwrap();
        let active_y = p
            .add_integer_var("active-y", 1.0, Some(0.0), Some(6.0))
            .unwrap();
        let inactive_gate = p.add_binary_var("inactive-gate", 0.0).unwrap();
        let inactive_z = p
            .add_integer_var("inactive-z", 1.0, Some(0.0), Some(6.0))
            .unwrap();

        p.add_constraint("cap-x", vec![(x, 1.0)], RowSense::Le, 5.0)
            .unwrap();
        p.add_linear_not_in_domain("x-not-in-domain", vec![(x, 1.0)], vec![(2, 3), (5, 5)])
            .unwrap();
        p.add_constraint(
            "force-active-gate",
            vec![(active_gate, 1.0)],
            RowSense::Eq,
            1.0,
        )
        .unwrap();
        p.add_constraint("cap-active-y", vec![(active_y, 1.0)], RowSense::Le, 5.0)
            .unwrap();
        p.add_enforced_linear_not_in_domain(
            "active-y-not-in-domain",
            vec![MathProgram::bool_lit(active_gate)],
            vec![(active_y, 1.0)],
            vec![(1, 2), (4, 5)],
        )
        .unwrap();
        p.add_constraint(
            "force-inactive-gate",
            vec![(inactive_gate, 1.0)],
            RowSense::Eq,
            0.0,
        )
        .unwrap();
        p.add_constraint("cap-inactive-z", vec![(inactive_z, 1.0)], RowSense::Le, 5.0)
            .unwrap();
        p.add_enforced_linear_not_in_domain(
            "inactive-z-not-in-domain",
            vec![MathProgram::bool_lit(inactive_gate)],
            vec![(inactive_z, 1.0)],
            vec![(1, 2), (4, 5)],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 4.0);
        assert_close(sol.x[active_gate], 1.0);
        assert_close(sol.x[active_y], 3.0);
        assert_close(sol.x[inactive_gate], 0.0);
        assert_close(sol.x[inactive_z], 5.0);
        assert_close(sol.objective, 12.0);
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
    fn forbidden_assignments_excludes_disallowed_tuple() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_binary_var("x", 2.0).unwrap();
        let y = p.add_binary_var("y", 1.0).unwrap();
        p.add_forbidden_assignments("not-both", vec![x, y], vec![vec![1, 1]])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x], 1.0);
        assert_close(sol.x[y], 0.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn enforced_table_assignments_respect_enforcement_literals() {
        let mut active_allowed = MathProgram::new(ObjectiveSense::Max);
        let allow_gate = active_allowed.add_binary_var("allow-gate", 0.0).unwrap();
        let allow_block = active_allowed.add_binary_var("allow-block", 0.0).unwrap();
        let allow_x = active_allowed
            .add_integer_var("allow-x", 8.0, Some(0.0), Some(2.0))
            .unwrap();
        let allow_y = active_allowed
            .add_integer_var("allow-y", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        active_allowed
            .add_constraint(
                "force-allow-gate",
                vec![(allow_gate, 1.0)],
                RowSense::Eq,
                1.0,
            )
            .unwrap();
        active_allowed
            .add_constraint(
                "force-allow-block",
                vec![(allow_block, 1.0)],
                RowSense::Eq,
                0.0,
            )
            .unwrap();
        active_allowed
            .add_enforced_allowed_assignments(
                "active-allowed",
                vec![
                    MathProgram::bool_lit(allow_gate),
                    MathProgram::not_lit(allow_block),
                ],
                vec![allow_x, allow_y],
                vec![vec![0, 2], vec![2, 0]],
            )
            .unwrap();

        let active_allowed_sol =
            solve_math_program(&active_allowed, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(active_allowed_sol.status, MathProgramStatus::Optimal);
        assert_close(active_allowed_sol.x[allow_gate], 1.0);
        assert_close(active_allowed_sol.x[allow_block], 0.0);
        assert_close(active_allowed_sol.x[allow_x], 2.0);
        assert_close(active_allowed_sol.x[allow_y], 0.0);
        assert_close(active_allowed_sol.objective, 16.0);

        let mut active_forbidden = MathProgram::new(ObjectiveSense::Max);
        let forbid_gate = active_forbidden.add_binary_var("forbid-gate", 0.0).unwrap();
        let forbid_x = active_forbidden.add_binary_var("forbid-x", 5.0).unwrap();
        let forbid_y = active_forbidden.add_binary_var("forbid-y", 3.0).unwrap();
        active_forbidden
            .add_constraint(
                "force-forbid-gate",
                vec![(forbid_gate, 1.0)],
                RowSense::Eq,
                1.0,
            )
            .unwrap();
        active_forbidden
            .add_enforced_forbidden_assignments(
                "active-forbidden",
                vec![MathProgram::bool_lit(forbid_gate)],
                vec![forbid_x, forbid_y],
                vec![vec![1, 1]],
            )
            .unwrap();

        let active_forbidden_sol =
            solve_math_program(&active_forbidden, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(active_forbidden_sol.status, MathProgramStatus::Optimal);
        assert_close(active_forbidden_sol.x[forbid_gate], 1.0);
        assert_close(active_forbidden_sol.x[forbid_x], 1.0);
        assert_close(active_forbidden_sol.x[forbid_y], 0.0);
        assert_close(active_forbidden_sol.objective, 5.0);

        let mut inactive_forbidden = MathProgram::new(ObjectiveSense::Max);
        let inactive_gate = inactive_forbidden
            .add_binary_var("inactive-gate", 0.0)
            .unwrap();
        let inactive_x = inactive_forbidden
            .add_binary_var("inactive-x", 2.0)
            .unwrap();
        let inactive_y = inactive_forbidden
            .add_binary_var("inactive-y", 1.0)
            .unwrap();
        inactive_forbidden
            .add_constraint(
                "force-inactive-gate",
                vec![(inactive_gate, 1.0)],
                RowSense::Eq,
                0.0,
            )
            .unwrap();
        inactive_forbidden
            .add_enforced_forbidden_assignments(
                "inactive-forbidden",
                vec![MathProgram::bool_lit(inactive_gate)],
                vec![inactive_x, inactive_y],
                vec![vec![1, 1]],
            )
            .unwrap();

        let inactive_forbidden_sol =
            solve_math_program(&inactive_forbidden, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(inactive_forbidden_sol.status, MathProgramStatus::Optimal);
        assert_close(inactive_forbidden_sol.x[inactive_gate], 0.0);
        assert_close(inactive_forbidden_sol.x[inactive_x], 1.0);
        assert_close(inactive_forbidden_sol.x[inactive_y], 1.0);
        assert_close(inactive_forbidden_sol.objective, 3.0);
    }

    #[test]
    fn element_general_constraint_selects_value_by_index() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let index = p
            .add_integer_var("index", 0.0, Some(0.0), Some(3.0))
            .unwrap();
        let picked = p
            .add_integer_var("picked", 1.0, Some(0.0), Some(9.0))
            .unwrap();
        p.add_element("lookup", index, picked, vec![1.0, 7.0, 4.0, 9.0])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[index], 3.0);
        assert_close(sol.x[picked], 9.0);
        assert_close(sol.objective, 9.0);
    }

    #[test]
    fn variable_element_general_constraint_selects_variable_by_index() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let index = p
            .add_integer_var("index", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let a = p.add_integer_var("a", 0.0, Some(2.0), Some(2.0)).unwrap();
        let b = p.add_integer_var("b", 0.0, Some(8.0), Some(8.0)).unwrap();
        let c = p.add_integer_var("c", 0.0, Some(5.0), Some(5.0)).unwrap();
        let picked = p
            .add_integer_var("picked", 1.0, Some(0.0), Some(10.0))
            .unwrap();
        p.add_variable_element("variable-lookup", index, picked, vec![a, b, c])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[index], 1.0);
        assert_close(sol.x[picked], 8.0);
        assert_close(sol.objective, 8.0);
    }

    #[test]
    fn inverse_general_constraint_links_permutation_arrays() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x0 = p.add_integer_var("x0", 0.0, Some(0.0), Some(2.0)).unwrap();
        let x1 = p.add_integer_var("x1", 1.0, Some(0.0), Some(2.0)).unwrap();
        let x2 = p.add_integer_var("x2", 0.0, Some(0.0), Some(2.0)).unwrap();
        let y0 = p.add_integer_var("y0", 0.0, Some(0.0), Some(2.0)).unwrap();
        let y1 = p.add_integer_var("y1", 0.0, Some(0.0), Some(2.0)).unwrap();
        let y2 = p.add_integer_var("y2", 0.0, Some(0.0), Some(2.0)).unwrap();
        p.add_constraint("force-x0", vec![(x0, 1.0)], RowSense::Eq, 1.0)
            .unwrap();
        p.add_inverse("inverse-permutation", vec![x0, x1, x2], vec![y0, y1, y2])
            .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[x0], 1.0);
        assert_close(sol.x[x1], 2.0);
        assert_close(sol.x[x2], 0.0);
        assert_close(sol.x[y0], 2.0);
        assert_close(sol.x[y1], 0.0);
        assert_close(sol.x[y2], 1.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn circuit_general_constraint_selects_hamiltonian_cycle() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let mut arcs = Vec::new();
        let mut arc_vars = vec![vec![None; 4]; 4];
        for tail in 0..4 {
            for head in 0..4 {
                if tail == head {
                    continue;
                }
                let obj = match (tail, head) {
                    (0, 1) | (1, 2) | (2, 3) | (3, 0) => 10.0,
                    _ => 0.0,
                };
                let var = p.add_binary_var(format!("x_{tail}_{head}"), obj).unwrap();
                arcs.push((tail, head, var));
                arc_vars[tail][head] = Some(var);
            }
        }
        p.add_circuit("tour", 4, arcs).unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        for (tail, head) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            let var = arc_vars[tail][head].unwrap();
            assert_close(sol.x[var], 1.0);
        }
        assert_close(sol.objective, 40.0);
    }

    #[test]
    fn multiple_circuit_general_constraint_rejects_disconnected_subtours() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let depot_to_first = p.add_binary_var("x_0_1", 1.0).unwrap();
        let first_to_second = p.add_binary_var("x_1_2", 10.0).unwrap();
        let second_to_depot = p.add_binary_var("x_2_0", 1.0).unwrap();
        let second_to_first = p.add_binary_var("x_2_1", 10.0).unwrap();
        let skipped = p.add_binary_var("x_3_3", 0.0).unwrap();

        p.add_multiple_circuit(
            "routes",
            4,
            vec![
                (0, 1, depot_to_first),
                (1, 2, first_to_second),
                (2, 0, second_to_depot),
                (2, 1, second_to_first),
                (3, 3, skipped),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[depot_to_first], 1.0);
        assert_close(sol.x[first_to_second], 1.0);
        assert_close(sol.x[second_to_depot], 1.0);
        assert_close(sol.x[second_to_first], 0.0);
        assert_close(sol.x[skipped], 1.0);
        assert_close(sol.objective, 12.0);
    }

    #[test]
    fn alternative_interval_selects_one_mode() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let start = p
            .add_integer_var("task_start", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let size = p
            .add_integer_var("task_size", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let end = p
            .add_integer_var("task_end", 1.0, Some(0.0), Some(5.0))
            .unwrap();
        let slow_start = p
            .add_integer_var("slow_start", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let slow_end = p
            .add_integer_var("slow_end", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let fast_start = p
            .add_integer_var("fast_start", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let fast_end = p
            .add_integer_var("fast_end", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let slow_present = p.add_binary_var("slow_present", 0.0).unwrap();
        let fast_present = p.add_binary_var("fast_present", 0.0).unwrap();
        p.add_alternative(
            "choose-mode",
            MathProgram::variable_interval(start, size, end),
            vec![
                MathProgram::optional_interval(slow_start, 4.0, slow_end, slow_present),
                MathProgram::optional_interval(fast_start, 2.0, fast_end, fast_present),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[start], 0.0);
        assert_close(sol.x[size], 2.0);
        assert_close(sol.x[end], 2.0);
        assert_close(sol.x[slow_present], 0.0);
        assert_close(sol.x[fast_present], 1.0);
        assert_close(sol.x[fast_start], 0.0);
        assert_close(sol.x[fast_end], 2.0);
        assert_close(sol.objective, 2.0);
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
    fn no_overlap_orders_variable_size_intervals() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_start = p
            .add_integer_var("a_start", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let a_size = p
            .add_integer_var("a_size", 0.0, Some(2.0), Some(4.0))
            .unwrap();
        let a_end = p
            .add_integer_var("a_end", 0.0, Some(0.0), Some(8.0))
            .unwrap();
        let b_start = p
            .add_integer_var("b_start", 1.0, Some(0.0), Some(8.0))
            .unwrap();
        let b_end = p
            .add_integer_var("b_end", 0.0, Some(0.0), Some(8.0))
            .unwrap();

        p.add_constraint("fix-a-start", vec![(a_start, 1.0)], RowSense::Eq, 0.0)
            .unwrap();
        p.add_constraint(
            "force-a-size-through-end",
            vec![(a_end, 1.0)],
            RowSense::Ge,
            4.0,
        )
        .unwrap();
        p.add_no_overlap(
            "variable-machine",
            vec![
                MathProgram::variable_interval(a_start, a_size, a_end),
                MathProgram::interval(b_start, 2.0, b_end),
            ],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_start], 0.0);
        assert_close(sol.x[a_size], 4.0);
        assert_close(sol.x[a_end], 4.0);
        assert_close(sol.x[b_start], 4.0);
        assert_close(sol.x[b_end], 6.0);
        assert_close(sol.objective, 4.0);
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
    fn cumulative_orders_variable_size_interval() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_start = p
            .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let a_size = p
            .add_integer_var("a_size", 0.0, Some(1.0), Some(3.0))
            .unwrap();
        let a_end = p
            .add_integer_var("a_end", 0.0, Some(0.0), Some(3.0))
            .unwrap();
        let b_start = p
            .add_integer_var("b_start", 1.0, Some(0.0), Some(3.0))
            .unwrap();
        let b_end = p
            .add_integer_var("b_end", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        p.add_constraint("force-a-size", vec![(a_end, 1.0)], RowSense::Ge, 3.0)
            .unwrap();
        p.add_cumulative(
            "shared-resource",
            vec![
                MathProgram::variable_interval(a_start, a_size, a_end),
                MathProgram::interval(b_start, 2.0, b_end),
            ],
            vec![2.0, 2.0],
            3.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_start], 0.0);
        assert_close(sol.x[a_size], 3.0);
        assert_close(sol.x[a_end], 3.0);
        assert_close(sol.x[b_start], 3.0);
        assert_close(sol.x[b_end], 5.0);
        assert_close(sol.objective, 3.0);
    }

    #[test]
    fn cumulative_accepts_affine_demand_and_capacity() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let a_start = p
            .add_integer_var("a_start", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let a_end = p
            .add_integer_var("a_end", 0.0, Some(0.0), Some(2.0))
            .unwrap();
        let b_start = p
            .add_integer_var("b_start", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        let b_end = p
            .add_integer_var("b_end", 0.0, Some(0.0), Some(4.0))
            .unwrap();
        let a_demand = p
            .add_integer_var("a_demand", 0.0, Some(1.0), Some(2.0))
            .unwrap();
        let capacity = p
            .add_integer_var("capacity", 0.0, Some(3.0), Some(4.0))
            .unwrap();
        p.add_constraint("force-a-demand", vec![(a_demand, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        p.add_constraint("force-capacity", vec![(capacity, 1.0)], RowSense::Le, 3.0)
            .unwrap();
        p.add_cumulative_affine(
            "shared-resource",
            vec![
                MathProgram::interval(a_start, 2.0, a_end),
                MathProgram::interval(b_start, 2.0, b_end),
            ],
            vec![
                AffineTerm {
                    coeffs: vec![(a_demand, 1.0)],
                    constant: 0.0,
                },
                AffineTerm {
                    coeffs: Vec::new(),
                    constant: 2.0,
                },
            ],
            AffineTerm {
                coeffs: vec![(capacity, 1.0)],
                constant: 0.0,
            },
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[a_start], 0.0);
        assert_close(sol.x[a_end], 2.0);
        assert_close(sol.x[b_start], 2.0);
        assert_close(sol.x[b_end], 4.0);
        assert_close(sol.x[a_demand], 2.0);
        assert_close(sol.x[capacity], 3.0);
        assert_close(sol.objective, 2.0);
    }

    #[test]
    fn bin_packing_links_assignments_and_loads() {
        let mut p = MathProgram::new(ObjectiveSense::Min);
        let item0 = p
            .add_integer_var("item0_bin", 0.0, Some(0.0), Some(1.0))
            .unwrap();
        let item1 = p
            .add_integer_var("item1_bin", 0.0, Some(0.0), Some(1.0))
            .unwrap();
        let item2 = p
            .add_integer_var("item2_bin", 0.0, Some(0.0), Some(1.0))
            .unwrap();
        let load0 = p
            .add_integer_var("load0", 0.0, Some(0.0), Some(5.0))
            .unwrap();
        let load1 = p
            .add_integer_var("load1", 1.0, Some(0.0), Some(9.0))
            .unwrap();
        p.add_bin_packing(
            "packing",
            vec![item0, item1, item2],
            vec![load0, load1],
            vec![2.0, 3.0, 4.0],
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[item0], 0.0);
        assert_close(sol.x[item1], 0.0);
        assert_close(sol.x[item2], 1.0);
        assert_close(sol.x[load0], 5.0);
        assert_close(sol.x[load1], 4.0);
        assert_close(sol.objective, 4.0);
    }

    #[test]
    fn reservoir_general_constraint_enforces_prefix_level_bounds() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let supply_time = p
            .add_integer_var("supply_time", 1.0, Some(0.0), Some(2.0))
            .unwrap();
        let drain_time = p
            .add_integer_var("drain_time", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        p.add_reservoir(
            "tank",
            vec![
                MathProgram::reservoir_event(supply_time, 2.0),
                MathProgram::reservoir_event(drain_time, -2.0),
            ],
            0.0,
            2.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.x[supply_time], 0.0);
        assert_close(sol.x[drain_time], 0.0);
        assert_close(sol.objective, 0.0);
    }

    #[test]
    fn optional_reservoir_event_links_activity_and_level_bounds() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let required_time = p
            .add_integer_var("required_time", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let safe_time = p
            .add_integer_var("safe_time", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let overflow_time = p
            .add_integer_var("overflow_time", 0.0, Some(0.0), Some(0.0))
            .unwrap();
        let safe_active = p.add_binary_var("safe_active", 3.0).unwrap();
        let overflow_active = p.add_binary_var("overflow_active", 5.0).unwrap();
        p.add_reservoir(
            "optional-tank",
            vec![
                MathProgram::reservoir_event(required_time, 1.0),
                MathProgram::optional_reservoir_event(safe_time, 1.0, safe_active),
                MathProgram::optional_reservoir_event(overflow_time, 2.0, overflow_active),
            ],
            0.0,
            2.0,
        )
        .unwrap();

        let sol = solve_math_program(&p, &MathProgramSolveOptions::default()).unwrap();
        assert_eq!(sol.status, MathProgramStatus::Optimal);
        assert_close(sol.objective, 3.0);
        assert_close(sol.x[required_time], 0.0);
        assert_close(sol.x[safe_time], 0.0);
        assert_close(sol.x[overflow_time], 0.0);
        assert_close(sol.x[safe_active], 1.0);
        assert_close(sol.x[overflow_active], 0.0);
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
    fn solution_pool_enumerates_integer_and_semi_integer_domains() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let batches = p
            .add_integer_var("batches", 10.0, Some(0.0), Some(2.0))
            .unwrap();
        let lot = p.add_semi_integer_var("lot", 1.0, 2.0, 3.0).unwrap();
        p.add_constraint(
            "capacity",
            vec![(batches, 1.0), (lot, 1.0)],
            RowSense::Le,
            4.0,
        )
        .unwrap();

        let pool = solve_math_program_solution_pool(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramSolutionPoolOptions {
                max_solutions: 4,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(pool.solutions.len(), 4);
        assert!(!pool.exhausted);
        assert_close(pool.solutions[0].objective, 22.0);
        assert_close(pool.solutions[1].objective, 20.0);
        assert_close(pool.solutions[2].objective, 13.0);
        assert_close(pool.solutions[3].objective, 12.0);
        assert_eq!(pool.solutions[0].x, vec![2.0, 2.0]);
        assert_eq!(pool.solutions[1].x, vec![2.0, 0.0]);
        assert_eq!(pool.solutions[2].x, vec![1.0, 3.0]);
        assert_eq!(pool.solutions[3].x, vec![1.0, 2.0]);
    }

    #[test]
    fn solution_pool_respects_absolute_objective_gap_for_integer_domains() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let batches = p
            .add_integer_var("batches", 10.0, Some(0.0), Some(2.0))
            .unwrap();
        let lot = p.add_semi_integer_var("lot", 1.0, 2.0, 3.0).unwrap();
        p.add_constraint(
            "capacity",
            vec![(batches, 1.0), (lot, 1.0)],
            RowSense::Le,
            4.0,
        )
        .unwrap();

        let pool = solve_math_program_solution_pool(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramSolutionPoolOptions {
                max_solutions: 10,
                absolute_gap: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(pool.solutions.len(), 2);
        assert!(!pool.exhausted);
        assert!(pool
            .message
            .as_deref()
            .is_some_and(|message| message.contains("objective gap")));
        assert_close(pool.solutions[0].objective, 22.0);
        assert_close(pool.solutions[1].objective, 20.0);
        assert_eq!(pool.solutions[0].x, vec![2.0, 2.0]);
        assert_eq!(pool.solutions[1].x, vec![2.0, 0.0]);
    }

    #[test]
    fn solution_pool_respects_relative_objective_gap_for_integer_domains() {
        let mut p = MathProgram::new(ObjectiveSense::Max);
        let batches = p
            .add_integer_var("batches", 10.0, Some(0.0), Some(2.0))
            .unwrap();
        let lot = p.add_semi_integer_var("lot", 1.0, 2.0, 3.0).unwrap();
        p.add_constraint(
            "capacity",
            vec![(batches, 1.0), (lot, 1.0)],
            RowSense::Le,
            4.0,
        )
        .unwrap();

        let pool = solve_math_program_solution_pool(
            &p,
            &MathProgramSolveOptions::default(),
            &MathProgramSolutionPoolOptions {
                max_solutions: 10,
                relative_gap: Some(0.41),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(pool.solutions.len(), 3);
        assert!(!pool.exhausted);
        assert!(pool
            .message
            .as_deref()
            .is_some_and(|message| message.contains("objective gap")));
        assert_close(pool.solutions[0].objective, 22.0);
        assert_close(pool.solutions[1].objective, 20.0);
        assert_close(pool.solutions[2].objective, 13.0);
        assert_eq!(pool.solutions[0].x, vec![2.0, 2.0]);
        assert_eq!(pool.solutions[1].x, vec![2.0, 0.0]);
        assert_eq!(pool.solutions[2].x, vec![1.0, 3.0]);
    }

    #[test]
    fn external_math_program_cli_method_parser_keeps_api_methods_on_bridge() {
        assert_eq!(
            parse_external_math_program_linear_cli_method("highs:cli"),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            parse_external_math_program_linear_cli_method("highs-cli:default"),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            parse_external_math_program_linear_cli_method("qsopt_ex_cli"),
            Some(ExternalLinearCliSolver::QsoptEx)
        );
        assert_eq!(
            parse_external_math_program_linear_cli_method("lindo-cli"),
            Some(ExternalLinearCliSolver::Lindo)
        );
        assert_eq!(parse_external_math_program_linear_cli_method("highs"), None);
        assert_eq!(
            parse_external_math_program_linear_cli_method("glpk:default"),
            None
        );
        assert_eq!(
            parse_external_math_program_linear_cli_method("gurobi:default"),
            None
        );
        assert_eq!(
            parse_external_math_program_linear_cli_method("ortools:CP-SAT"),
            None
        );
    }

    #[test]
    fn external_math_program_rust_first_alias_resolver_prefers_cli_for_linear_models() {
        let mut lp = MathProgram::new(ObjectiveSense::Max);
        let x = lp.add_continuous_var("x", 1.0, Some(0.0), None).unwrap();
        lp.add_constraint("limit", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();

        assert_eq!(
            resolve_external_math_program_linear_cli_solver(
                &lp,
                &ExternalMathProgramOptions::default(),
                "highs"
            ),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(
                &lp,
                &ExternalMathProgramOptions::default(),
                "HiGHS-DS:default"
            ),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(
                &lp,
                &ExternalMathProgramOptions::default(),
                "glpk:default"
            ),
            Some(ExternalLinearCliSolver::Glpk)
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(
                &lp,
                &ExternalMathProgramOptions::default(),
                "glpsol"
            ),
            Some(ExternalLinearCliSolver::Glpk)
        );
        for (method, solver) in [
            ("scip", ExternalLinearCliSolver::Scip),
            ("cbc", ExternalLinearCliSolver::Cbc),
            ("coin-or-cbc:default", ExternalLinearCliSolver::Cbc),
            ("clp", ExternalLinearCliSolver::Clp),
            ("soplex", ExternalLinearCliSolver::Soplex),
            ("qsopt_ex", ExternalLinearCliSolver::QsoptEx),
            ("lp-solve", ExternalLinearCliSolver::LpSolve),
        ] {
            assert_eq!(
                resolve_external_math_program_linear_cli_solver(
                    &lp,
                    &ExternalMathProgramOptions::default(),
                    method
                ),
                Some(solver),
                "{method} should use the Rust CLI adapter by default"
            );
        }

        let explicit_python = ExternalMathProgramOptions {
            python: Some("python3".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &explicit_python, "highs"),
            None
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &explicit_python, "glpk"),
            None
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &explicit_python, "cbc"),
            None
        );

        let custom_script = ExternalMathProgramOptions {
            script: Some("custom_bridge.py".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &custom_script, "highs"),
            None
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &custom_script, "highs-cli"),
            Some(ExternalLinearCliSolver::Highs)
        );
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(&lp, &custom_script, "soplex"),
            None
        );

        let mut qp = MathProgram::new(ObjectiveSense::Min);
        let q = qp.add_continuous_var("q", 0.0, Some(0.0), None).unwrap();
        qp.add_quadratic_objective_term(q, q, 1.0).unwrap();
        assert_eq!(
            resolve_external_math_program_linear_cli_solver(
                &qp,
                &ExternalMathProgramOptions::default(),
                "highs"
            ),
            None
        );
    }

    #[test]
    fn external_math_program_default_continuous_qp_uses_rust_reference_without_python() {
        let mut qp = MathProgram::new(ObjectiveSense::Min);
        let x = qp.add_continuous_var("x", 0.0, Some(0.0), Some(5.0)).unwrap();
        let y = qp.add_continuous_var("y", 0.0, Some(0.0), Some(5.0)).unwrap();
        qp.add_quadratic_objective_term(x, x, 2.0).unwrap();
        qp.add_quadratic_objective_term(y, y, 2.0).unwrap();
        qp.add_linear_objective_term(x, -2.0).unwrap();
        qp.add_linear_objective_term(y, -4.0).unwrap();

        let solution =
            solve_math_program_external(&qp, &ExternalMathProgramOptions::default()).unwrap();

        assert_eq!(solution.status, MathProgramStatus::Optimal);
        assert_eq!(solution.solver, "rust:qp-active-set");
        assert_close(solution.x[x], 0.5);
        assert_close(solution.x[y], 1.0);
        assert_close(solution.objective, -2.5);
    }

    #[test]
    fn external_math_program_python_resolver_honors_python_bin_precedence() {
        assert_eq!(
            resolve_external_math_program_python_from(
                &ExternalMathProgramOptions::default(),
                Some("/tmp/python-bin"),
                Some("/tmp/python"),
            ),
            "/tmp/python-bin"
        );
        assert_eq!(
            resolve_external_math_program_python_from(
                &ExternalMathProgramOptions {
                    python: Some("/tmp/explicit-python".to_string()),
                    ..Default::default()
                },
                Some("/tmp/python-bin"),
                Some("/tmp/python"),
            ),
            "/tmp/explicit-python"
        );
        assert_eq!(
            resolve_external_math_program_python_from(
                &ExternalMathProgramOptions::default(),
                None,
                Some("/tmp/python"),
            ),
            "/tmp/python"
        );
        assert_eq!(
            resolve_external_math_program_python_from(
                &ExternalMathProgramOptions::default(),
                None,
                None,
            ),
            "python3"
        );
    }

    #[test]
    fn highs_default_cli_fallback_only_handles_missing_scipy_stack_for_linear_models() {
        let mut lp = MathProgram::new(ObjectiveSense::Max);
        let x = lp.add_continuous_var("x", 1.0, Some(0.0), None).unwrap();
        lp.add_constraint("limit", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        let missing_scipy = serde_json::json!({
            "status": "numerical-error",
            "message": "No module named 'scipy'",
        });
        assert!(should_fallback_highs_default_to_cli(
            &lp,
            "highs",
            &missing_scipy,
            &ExternalMathProgramOptions::default()
        ));
        assert!(should_fallback_highs_default_to_cli(
            &lp,
            "HiGHS-DS:default",
            &missing_scipy,
            &ExternalMathProgramOptions::default()
        ));

        let mut mip = MathProgram::new(ObjectiveSense::Max);
        let y = mip.add_binary_var("y", 1.0).unwrap();
        mip.add_constraint("pick", vec![(y, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        let missing_numpy = serde_json::json!({
            "status": "numerical-error",
            "message": "scipy milp unavailable: No module named 'numpy'",
        });
        assert!(should_fallback_highs_default_to_cli(
            &mip,
            "highs-ipm",
            &missing_numpy,
            &ExternalMathProgramOptions::default()
        ));

        let custom_script = ExternalMathProgramOptions {
            script: Some("custom_highs_bridge.py".to_string()),
            ..Default::default()
        };
        assert!(!should_fallback_highs_default_to_cli(
            &lp,
            "highs",
            &missing_scipy,
            &custom_script
        ));
        assert!(!should_fallback_highs_default_to_cli(
            &lp,
            "highs-cli:default",
            &missing_scipy,
            &ExternalMathProgramOptions::default()
        ));

        let solver_error = serde_json::json!({
            "status": "numerical-error",
            "message": "HiGHS failed while reading model",
        });
        assert!(!should_fallback_highs_default_to_cli(
            &lp,
            "highs",
            &solver_error,
            &ExternalMathProgramOptions::default()
        ));
        let infeasible = serde_json::json!({
            "status": "infeasible",
            "message": "No module named 'scipy'",
        });
        assert!(!should_fallback_highs_default_to_cli(
            &lp,
            "highs",
            &infeasible,
            &ExternalMathProgramOptions::default()
        ));

        let mut qp = MathProgram::new(ObjectiveSense::Min);
        let q = qp.add_continuous_var("q", 0.0, Some(0.0), None).unwrap();
        qp.add_quadratic_objective_term(q, q, 1.0).unwrap();
        assert!(!should_fallback_highs_default_to_cli(
            &qp,
            "highs",
            &missing_scipy,
            &ExternalMathProgramOptions::default()
        ));
    }

    #[test]
    fn glpk_default_cli_fallback_only_handles_missing_python_binding() {
        let missing = serde_json::json!({
            "status": "numerical-error",
            "message": "GLPK unavailable: No module named 'swiglpk'",
        });
        assert!(should_fallback_glpk_default_to_cli(
            "glpk:default",
            &missing,
            &ExternalMathProgramOptions::default()
        ));
        assert!(should_fallback_glpk_default_to_cli(
            " GLPK ",
            &missing,
            &ExternalMathProgramOptions::default()
        ));

        let custom_script = ExternalMathProgramOptions {
            script: Some("custom_glpk_bridge.py".to_string()),
            ..Default::default()
        };
        assert!(!should_fallback_glpk_default_to_cli(
            "glpk:default",
            &missing,
            &custom_script
        ));
        assert!(!should_fallback_glpk_default_to_cli(
            "glpk-cli:default",
            &missing,
            &ExternalMathProgramOptions::default()
        ));

        let solver_error = serde_json::json!({
            "status": "numerical-error",
            "message": "GLPK failed while reading model",
        });
        assert!(!should_fallback_glpk_default_to_cli(
            "glpk:default",
            &solver_error,
            &ExternalMathProgramOptions::default()
        ));
        let infeasible = serde_json::json!({
            "status": "infeasible",
            "message": "No module named 'swiglpk'",
        });
        assert!(!should_fallback_glpk_default_to_cli(
            "glpk:default",
            &infeasible,
            &ExternalMathProgramOptions::default()
        ));
    }

    #[test]
    fn external_default_linear_method_uses_native_highs_cli_when_available() {
        if std::process::Command::new("highs")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping MathProgram default external CLI test; highs is not installed");
            return;
        }

        let mut p = MathProgram::new(ObjectiveSense::Max);
        let x = p.add_continuous_var("x", 1.0, Some(0.0), None).unwrap();
        p.add_constraint("cap", vec![(x, 1.0)], RowSense::Le, 1.0)
            .unwrap();

        let solution =
            solve_math_program_external(&p, &ExternalMathProgramOptions::default()).unwrap();

        assert_eq!(solution.status, MathProgramStatus::Optimal);
        assert_eq!(solution.solver, "highs:cli");
        assert_close(solution.objective, 1.0);
        assert_close(solution.x[x], 1.0);
    }

    #[test]
    fn external_cli_facade_cross_checks_conflict_relaxation_pool_and_clp_lp() {
        if std::process::Command::new("highs")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping MathProgram external CLI facade test; highs is not installed");
            return;
        }

        let highs_cli = ExternalMathProgramOptions {
            method: Some("highs:cli".to_string()),
            time_limit_ms: Some(5_000.0),
            ..Default::default()
        };

        let mut conflict_model = MathProgram::new(ObjectiveSense::Min);
        let conflict_x = conflict_model
            .add_continuous_var("x", 0.0, None, None)
            .unwrap();
        let conflict_y = conflict_model
            .add_continuous_var("y", 0.0, Some(0.0), None)
            .unwrap();
        conflict_model
            .add_constraint("x-at-least-two", vec![(conflict_x, 1.0)], RowSense::Ge, 2.0)
            .unwrap();
        conflict_model
            .add_constraint("x-at-most-one", vec![(conflict_x, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        conflict_model
            .add_constraint("redundant-y", vec![(conflict_y, 1.0)], RowSense::Ge, 0.0)
            .unwrap();

        let conflict = cross_check_math_program_conflict_with_external(
            &conflict_model,
            &MathProgramSolveOptions::default(),
            &highs_cli,
            &MathProgramConflictOptions::default(),
        )
        .unwrap();
        assert_eq!(conflict.external.status, MathProgramStatus::Infeasible);
        assert!(conflict.within_tolerance);
        assert!(conflict.internal.minimal);
        assert_eq!(conflict.internal.items.len(), 2);

        let mut assumption_model = MathProgram::new(ObjectiveSense::Min);
        let assume_a = assumption_model.add_binary_var("assume-a", 0.0).unwrap();
        let assume_b = assumption_model.add_binary_var("assume-b", 0.0).unwrap();
        let assume_noise = assumption_model
            .add_binary_var("assume-noise", 0.0)
            .unwrap();
        assumption_model
            .add_constraint(
                "assumption-at-most-one",
                vec![(assume_a, 1.0), (assume_b, 1.0)],
                RowSense::Le,
                1.0,
            )
            .unwrap();
        let assumption_core = cross_check_math_program_assumption_core_with_external(
            &assumption_model,
            &[
                MathProgram::bool_lit(assume_a),
                MathProgram::bool_lit(assume_b),
                MathProgram::not_lit(assume_noise),
            ],
            &MathProgramSolveOptions::default(),
            &highs_cli,
            &MathProgramAssumptionCoreOptions::default(),
        )
        .unwrap();
        assert_eq!(
            assumption_core.external.status,
            MathProgramStatus::Infeasible
        );
        assert!(assumption_core.within_tolerance);
        assert_eq!(
            assumption_core.internal.assumptions,
            vec![
                MathProgram::bool_lit(assume_a),
                MathProgram::bool_lit(assume_b)
            ]
        );

        let mut relax_model = MathProgram::new(ObjectiveSense::Min);
        let relax_x = relax_model
            .add_continuous_var("x", 0.0, Some(2.0), None)
            .unwrap();
        relax_model
            .add_constraint("cap", vec![(relax_x, 1.0)], RowSense::Le, 1.0)
            .unwrap();
        let relaxation = cross_check_math_program_feas_relaxation_with_external(
            &relax_model,
            &MathProgramSolveOptions::default(),
            &highs_cli,
            &MathProgramFeasRelaxOptions {
                linear_penalty: 10.0,
                bound_penalty: 1.0,
                ..Default::default()
            },
            1e-7,
        )
        .unwrap();
        assert_eq!(relaxation.external.status, MathProgramStatus::Optimal);
        assert!(relaxation.within_tolerance);
        assert_close(relaxation.internal.violation_objective, 1.0);

        let mut pool_model = MathProgram::new(ObjectiveSense::Max);
        let pool_a = pool_model.add_binary_var("a", 4.0).unwrap();
        let pool_b = pool_model.add_binary_var("b", 2.0).unwrap();
        let pool_c = pool_model.add_binary_var("c", 1.0).unwrap();
        pool_model
            .add_constraint(
                "choose-at-most-two",
                vec![(pool_a, 1.0), (pool_b, 1.0), (pool_c, 1.0)],
                RowSense::Le,
                2.0,
            )
            .unwrap();
        let pool = cross_check_math_program_solution_pool_with_external(
            &pool_model,
            &MathProgramSolveOptions::default(),
            &highs_cli,
            &MathProgramSolutionPoolOptions {
                max_solutions: 3,
                ..Default::default()
            },
            1e-7,
        )
        .unwrap();
        assert!(pool.within_tolerance);
        assert!(pool.len_agree);
        assert_eq!(pool.external.solutions.len(), 3);

        if std::process::Command::new("clp")
            .arg("-version")
            .output()
            .is_ok()
        {
            let mut lp = MathProgram::new(ObjectiveSense::Max);
            let x = lp.add_continuous_var("x", 1.0, Some(0.0), None).unwrap();
            lp.add_constraint("cap", vec![(x, 1.0)], RowSense::Le, 1.0)
                .unwrap();
            let clp = cross_check_math_program_with_external(
                &lp,
                &MathProgramSolveOptions::default(),
                &ExternalMathProgramOptions {
                    method: Some("clp:cli".to_string()),
                    time_limit_ms: Some(5_000.0),
                    ..Default::default()
                },
                1e-7,
            )
            .unwrap();
            assert_eq!(clp.external.status, MathProgramStatus::Optimal);
            assert_eq!(clp.external.solver, "clp:cli");
            assert!(clp.within_tolerance);
            assert_close(clp.external.objective, 1.0);
        }
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
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            unbounded_ray: None,
            infeasibility_certificate: None,
            solver: "internal".to_string(),
            message: None,
        };
        let external = MathProgramSolution {
            status: MathProgramStatus::Optimal,
            x: vec![0.0, 1.0],
            objective: 1.0,
            best_bound: None,
            mip_gap: None,
            nodes_explored: None,
            first_branch_variable: None,
            solver_version: None,
            iterations: None,
            control_feedback: None,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            unbounded_ray: None,
            infeasibility_certificate: None,
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
