//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/factmachine-math.ts`
//! Rust target: `src/des/general/factmachine_math.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/factmachine-math.ts",
    "src/des/general/factmachine_math.rs",
    &["RUST MIGRATION: target module src/des/general/factmachine_math.rs.", "RUST MIGRATION: OptionPrices, BuyExecution, SellExecution, RecapResult, OptionAggregates, ReplayOrder, and ReplayResult become serde structs.", "RUST MIGRATION: All LMSR/PnL helpers are pure vanilla functions; keep them as free functions or wrap public API calls in small PureTransform structs only if graph-visible.", "RUST MIGRATION: Replace thrown numeric/domain checks with Result, and decide up front whether f64 is sufficient or a decimal crate is needed for production parity.", "RUST MIGRATION: ReadonlyArray inputs become slices, replayOrders consumes &[ReplayOrder], and aggregate state becomes an explicit mutable struct."],
    &["BuyExecution", "OptionAggregates", "OptionPrices", "RecapResult", "ReplayOrder", "ReplayResult", "SellExecution", "avgCostBasis", "bFromLiquidity", "buyExecution", "buyThenSellRoundTrip", "finalPnl", "lmsrCost", "maxPriceWithSlippage", "minPriceWithSlippage", "netPosition", "optionOnePrice", "optionPrices", "recapitalization", "replayOrders", "sellExecution", "sharesFromBudget", "unrealizedPnl"],
);
