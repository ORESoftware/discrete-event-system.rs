//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/feasibility-pipeline.ts`
//! Rust target: `src/des/general/feasibility_pipeline.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/feasibility-pipeline.ts",
    "src/des/general/feasibility_pipeline.rs",
    &["RUST MIGRATION: target module src/des/general/feasibility_pipeline.rs.", "RUST MIGRATION: VariableKind, ConstraintSense, and ObjectiveSense become enums; all problem/candidate/evaluation/network/result interfaces become serde structs.", "RUST MIGRATION: CandidateToken, DomainCheckedToken, ConstraintCheckedToken, FeasibilityEvaluationToken, and pipeline stations become Token/Station trait impl structs.", "RUST MIGRATION: Record<string, number> coefficient/value maps become HashMap<String, f64>; Set<string> variable checks become HashSet<String>.", "RUST MIGRATION: runFeasibilityPipeline is graph-visible and should be a PureTransform entry struct; evaluateCandidate may stay a free function.", "RUST MIGRATION: Domain, constraint, and repair validation should return Result and avoid TS structural narrowing."],
    &["CANDIDATE_CHANNEL", "CONSTRAINT_CHANNEL", "CandidatePayload", "CandidateSolutionInput", "CandidateSourceStation", "CandidateToken", "ConstraintCheckedToken", "ConstraintCheckerStation", "ConstraintSense", "DOMAIN_CHANNEL", "DomainCheckedToken", "DomainCheckerStation", "EVALUATION_CHANNEL", "FeasibilityEvaluation", "FeasibilityEvaluationToken", "FeasibilityImprovementOptions", "FeasibilityPipelineEdge", "FeasibilityPipelineNetwork", "FeasibilityPipelineNode", "FeasibilityPipelineParams", "FeasibilityPipelineResult", "FeasibilitySinkStation", "FeasibilityViolation", "ImprovementStation", "LinearConstraint", "LinearObjective", "ObjectiveEvaluatorStation", "ObjectiveSense", "OptimizationVariable", "StructuredOptimizationProblem", "VariableKind", "evaluateCandidate", "runFeasibilityPipeline"],
);
