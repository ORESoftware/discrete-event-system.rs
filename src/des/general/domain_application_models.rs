//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/domain-application-models.ts`
//! Rust target: `src/des/general/domain_application_models.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/domain-application-models.ts",
    "src/des/general/domain_application_models.rs",
    &["RUST MIGRATION: target module src/des/general/domain_application_models.rs.", "RUST MIGRATION: DomainModelResult, DomainTrace, DomainEvaluation, scenario/plan/params/result types become serde structs; exported result aliases should become concrete type aliases.", "RUST MIGRATION: Generic DomainScenarioSource/CandidateGenerator/PlanEvaluator/ResultSink stations become generic structs implementing Station traits with explicit trait bounds.", "RUST MIGRATION: runDomainPipeline is graph-visible shared orchestration; port it as a reusable PureTransform/trait-backed pipeline, with each run* application as a thin PureTransform wrapper.", "RUST MIGRATION: Record-like plan fields map to HashMap<String, f64> where names are dynamic, and validation helpers return Result."],
    &["ActiveLearningParams", "ActiveLearningResult", "AdaptiveFuzzyControlParams", "AdaptiveFuzzyControlResult", "BuyerAwareDynamicPricingParams", "BuyerAwareDynamicPricingResult", "DecisionScienceParams", "DecisionScienceResult", "DomainEvaluation", "DomainModelResult", "DomainTrace", "EnergyParams", "EnergyResult", "FinancialControlParams", "FinancialControlResult", "LogisticsRoutingParams", "LogisticsRoutingResult", "ManufacturingParams", "ManufacturingResult", "OperationsParams", "OperationsResult", "RevenueManagementParams", "RevenueManagementResult", "SupplyChainParams", "SupplyChainResult", "runActiveLearningAcquisition", "runAdaptiveFuzzyControl", "runBottleneckProductionControl", "runBuyerAwareDynamicPricing", "runDynamicPricingRevenue", "runEnergyStorageDispatch", "runLogisticsRoutingHeuristics", "runPortfolioDrawdownControl", "runSupplyChainRiskPooling", "runVisualDecisionFrontier", "runWorkforceServiceOperations"],
);
