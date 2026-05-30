//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/soccer-rotation.ts`
//! Rust target: `src/des/general/soccer_rotation.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/soccer-rotation.ts",
    "src/des/general/soccer_rotation.rs",
    &["RUST MIGRATION: Target module `src/des/general/soccer_rotation.rs`.", "RUST MIGRATION: Convert problem, schedule, evaluation, LP/IP/MIP, POMDP feature, match-event, and aggregate interfaces to `serde` structs.", "RUST MIGRATION: Discriminated event/position/action concepts should become enums; use typed IDs/newtypes for players, periods, positions, and benches.", "RUST MIGRATION: Keep policy builders/solvers as free functions unless exposed as graph-visible transforms; LP/MIP vector translators should return `Result`.", "RUST MIGRATION: Replace map-like object keys and bench caches with `HashMap`/`HashSet`, inject RNG for randomized schedules/matches, and preserve deterministic ordering for reproducible reports."],
    &["AffinityBuilderOptions", "GoalEvent", "LPRelaxedScheduleResult", "MatchAggregate", "MatchResult", "MatchSimOptions", "MemorylessMDPResult", "Schedule", "ScheduleEvaluation", "SoccerIPMIPModel", "SoccerIPMIPPolicyOptions", "SoccerIPMIPPolicyResult", "SoccerPOMDPFeatureOptions", "SoccerPOMDPFeatureSummary", "SoccerPOMDPPeriodFeature", "SoccerProblem", "SubEvent", "buildSampleSoccerProblem", "buildSoccerIPMIP", "buildSoccerLP", "evaluateSchedule", "evaluateSoccerPOMDPFeatures", "policyGreedyHungarian", "policyIPMIPFeasible", "policyLPRelaxed", "policyMDPVI", "policyMDPVIMemoryless", "policyRandomSchedule", "runManyMatches", "scheduleFromSoccerIPMIPVector", "simulateMatchDES", "validateScheduleStructure", "welchT"],
);
