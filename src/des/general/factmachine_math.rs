//! Port of `src/des/general/factmachine-math.ts` — a pure `f64` mirror of the
//! production `@factmachine/math/trading` LMSR / PnL trading math for a binary
//! market.
//!
//! The DES POMDP simulation needs LMSR price/cost/execution algebra that is
//! numerically equivalent to production (which runs in Decimal.js) but fast
//! enough to step millions of ticks per second, hence the `f64` mirror here.
//!
//! TS → Rust mapping:
//!   * `const LN2`               -> [`std::f64::consts::LN_2`]
//!   * `const BPS_BASE`          -> [`BPS_BASE`]
//!   * `class X extends PureTransform<I,O>` -> unit struct + `impl Transform<I,O>`
//!   * `args: { ... }` objects   -> named `*Input` structs
//!   * `ReplayResult extends OptionAggregates` -> flattened struct fields
//!   * `action: 'BUY' | 'SELL'`  -> [`OrderAction`] enum
//!   * `time?: number | null`    -> `Option<f64>`
//!   * `throw` on bad params     -> `panic!` (invariant) / `Result` (edge validation)
//!   * `Math.expm1`/`Math.log1p` -> `f64::exp_m1`/`f64::ln_1p`
//!   * the `@deprecated` free-function shims are dropped per the migration plan.

use std::cmp::Ordering;

use crate::des::shared::transform::Transform;

/// Basis-points denominator (10_000 = 100%).
pub const BPS_BASE: f64 = 10_000.0;

/// Canonical: convert a USDC liquidity amount `L` into the LMSR `b` parameter
/// via `b = L / ln(2)`. Errs if `L` is not finite or non-positive.
pub fn b_from_liquidity(l: f64) -> Result<f64, String> {
    if !l.is_finite() || l <= 0.0 {
        return Err("initialLiquidity must be a finite positive number".to_string());
    }
    Ok(l / std::f64::consts::LN_2)
}

// -----------------------------------------------------------------------------
// LMSR PRIMITIVES (binary market specialisation)
// -----------------------------------------------------------------------------

/// Bundled args for the LMSR price/cost primitives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmsrPriceInput {
    pub q_one: f64,
    pub q_two: f64,
    pub b: f64,
}

/// Production formula: `price_1 = 1 / (1 + exp((q2 − q1) / b))`. Always in (0, 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct OptionOnePrice;

impl Transform<LmsrPriceInput, f64> for OptionOnePrice {
    fn transform(&self, input: LmsrPriceInput) -> f64 {
        let LmsrPriceInput { q_one, q_two, b } = input;
        if b <= 0.0 {
            panic!("b must be > 0");
        }
        // Guard the exponent to prevent Infinity for huge |q2 − q1|/b: production
        // ranges keep |q2-q1|/b ≤ ~14, so we clamp at ±700 (where exp would
        // overflow IEEE 754 at ±710).
        let exponent = (q_two - q_one) / b;
        if exponent > 700.0 {
            return 0.0; // overwhelmingly P(option2) = 1
        }
        if exponent < -700.0 {
            return 1.0; // overwhelmingly P(option1) = 1
        }
        1.0 / (1.0 + exponent.exp())
    }
}

/// Production formula: `C(q) = b · ln(exp(q1/b) + exp(q2/b))`, log-sum-exp form.
#[derive(Clone, Copy, Debug, Default)]
pub struct LmsrCost;

impl Transform<LmsrPriceInput, f64> for LmsrCost {
    fn transform(&self, input: LmsrPriceInput) -> f64 {
        let LmsrPriceInput { q_one, q_two, b } = input;
        if b <= 0.0 {
            panic!("b must be > 0");
        }
        let max = if q_one >= q_two { q_one } else { q_two };
        let min = if q_one >= q_two { q_two } else { q_one };
        let exp_term = ((min - max) / b).exp();
        max + b * exp_term.ln_1p()
    }
}

/// Per-option marginal prices (sum to 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionPrices {
    pub option_one: f64,
    pub option_two: f64,
}

/// Compute both option prices from `(qOne, qTwo, b)`.
pub fn option_prices(q_one: f64, q_two: f64, b: f64) -> OptionPrices {
    let p1 = OptionOnePrice.transform(LmsrPriceInput { q_one, q_two, b });
    OptionPrices { option_one: p1, option_two: 1.0 - p1 }
}

// -----------------------------------------------------------------------------
// EXECUTION FORMULAS (budget-based buys, share-based sells)
// -----------------------------------------------------------------------------

/// Args for inverting the cost function to shares from a USDC budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharesFromBudgetInput {
    pub budget: f64,
    pub current_price: f64,
    pub b: f64,
}

/// Production formula `shares = b · ln(1 + (exp(budget/b) − 1) / currentPrice)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SharesFromBudget;

impl Transform<SharesFromBudgetInput, f64> for SharesFromBudget {
    fn transform(&self, input: SharesFromBudgetInput) -> f64 {
        let SharesFromBudgetInput { budget, current_price, b } = input;
        if b <= 0.0 {
            panic!("b must be > 0");
        }
        if current_price <= 0.0 {
            panic!("currentPrice must be > 0");
        }
        if budget <= 0.0 {
            return 0.0;
        }
        // exp_m1(x) = exp(x) - 1, accurate near 0; avoids cancellation when
        // budget << b (small trades).
        let expm1 = (budget / b).exp_m1();
        b * (expm1 / current_price).ln_1p()
    }
}

/// Result of a budget-driven buy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuyExecution {
    pub shares: f64,
    /// Amount net of fees.
    pub buy_amount: f64,
    /// `buy_amount / shares`.
    pub average_price: f64,
    pub fee_amount: f64,
    /// Mirrors prod: equals `shares` for a buy.
    pub reward: f64,
}

/// Args for [`BuyExecutor`]. `fee_bps`/`is_option_one` default to `0`/`true`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuyExecutionInput {
    pub amount: f64,
    pub option_one_shares: f64,
    pub option_two_shares: f64,
    pub b: f64,
    pub fee_bps: Option<f64>,
    pub is_option_one: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BuyExecutor;

impl Transform<BuyExecutionInput, BuyExecution> for BuyExecutor {
    fn transform(&self, args: BuyExecutionInput) -> BuyExecution {
        let fee_bps = args.fee_bps.unwrap_or(0.0);
        if fee_bps >= BPS_BASE {
            panic!("feeBps must be less than 10000 (100%)");
        }
        let is_option_one = args.is_option_one.unwrap_or(true);
        let price = OptionOnePrice.transform(LmsrPriceInput {
            q_one: args.option_one_shares,
            q_two: args.option_two_shares,
            b: args.b,
        });
        let fee_amount = (args.amount * fee_bps) / BPS_BASE;
        let buy_amount = args.amount - fee_amount;
        let side_current_price = if is_option_one { price } else { 1.0 - price };
        let shares = SharesFromBudget.transform(SharesFromBudgetInput {
            budget: buy_amount,
            current_price: side_current_price,
            b: args.b,
        });
        let average_price = if shares == 0.0 { 0.0 } else { buy_amount / shares };
        BuyExecution { shares, buy_amount, average_price, fee_amount, reward: shares }
    }
}

/// Result of a share-driven sell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SellExecution {
    pub usdc_out: f64,
    /// Gross USDC before fees.
    pub sell_amount: f64,
    /// `sell_amount / sharesOut`.
    pub average_price: f64,
    pub fee_amount: f64,
    /// Mirrors prod: equals `usdc_out` for a sell.
    pub reward: f64,
}

/// Args for [`SellExecutor`]. `fee_bps`/`is_option_one` default to `0`/`true`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SellExecutionInput {
    pub shares_out: f64,
    pub option_one_shares: f64,
    pub option_two_shares: f64,
    pub b: f64,
    pub fee_bps: Option<f64>,
    pub is_option_one: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SellExecutor;

impl Transform<SellExecutionInput, SellExecution> for SellExecutor {
    fn transform(&self, args: SellExecutionInput) -> SellExecution {
        let fee_bps = args.fee_bps.unwrap_or(0.0);
        if fee_bps >= BPS_BASE {
            panic!("feeBps must be less than 10000 (100%)");
        }
        let is_option_one = args.is_option_one.unwrap_or(true);
        let cost_before = LmsrCost.transform(LmsrPriceInput {
            q_one: args.option_one_shares,
            q_two: args.option_two_shares,
            b: args.b,
        });
        let new_q1 = if is_option_one {
            args.option_one_shares - args.shares_out
        } else {
            args.option_one_shares
        };
        let new_q2 = if is_option_one {
            args.option_two_shares
        } else {
            args.option_two_shares - args.shares_out
        };
        let cost_after = LmsrCost.transform(LmsrPriceInput { q_one: new_q1, q_two: new_q2, b: args.b });
        let sell_amount = cost_before - cost_after;
        let fee_amount = (sell_amount * fee_bps) / BPS_BASE;
        let usdc_out = sell_amount - fee_amount;
        let average_price = if args.shares_out == 0.0 { 0.0 } else { sell_amount / args.shares_out };
        SellExecution { usdc_out, sell_amount, average_price, fee_amount, reward: usdc_out }
    }
}

// -----------------------------------------------------------------------------
// SLIPPAGE (production: clamp to [0, 1])
// -----------------------------------------------------------------------------

pub fn max_price_with_slippage(price: f64, slippage_bps: f64) -> f64 {
    let factor = 1.0 + slippage_bps / BPS_BASE;
    (price * factor).min(1.0).max(0.0)
}

pub fn min_price_with_slippage(price: f64, slippage_bps: f64) -> f64 {
    let factor = 1.0 - slippage_bps / BPS_BASE;
    (price * factor).min(1.0).max(0.0)
}

// -----------------------------------------------------------------------------
// RECAPITALIZATION (add or remove liquidity)
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecapResult {
    pub new_option_one_shares: f64,
    pub new_option_two_shares: f64,
    pub new_b: f64,
    /// `|C(new) - C(old)|`; absolute USDC moved in/out.
    pub capital_delta: f64,
}

/// Args for [`Recapitalization`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecapitalizationInput {
    pub option_one_shares: f64,
    pub option_two_shares: f64,
    pub current_b: f64,
    pub new_b: f64,
}

/// `newQ = oldQ · newB / oldB` preserves prices because prices depend only on
/// `(q2 − q1)/b`, and `(q2 − q1)` and `b` scale together.
#[derive(Clone, Copy, Debug, Default)]
pub struct Recapitalization;

impl Transform<RecapitalizationInput, RecapResult> for Recapitalization {
    fn transform(&self, args: RecapitalizationInput) -> RecapResult {
        if args.current_b <= 0.0 {
            panic!("currentB must be > 0");
        }
        if args.new_b <= 0.0 {
            panic!("newB must be > 0");
        }
        if args.current_b == args.new_b {
            panic!("newB must differ from currentB");
        }
        let ratio = args.new_b / args.current_b;
        let new_option_one_shares = args.option_one_shares * ratio;
        let new_option_two_shares = args.option_two_shares * ratio;
        let cost_old = LmsrCost.transform(LmsrPriceInput {
            q_one: args.option_one_shares,
            q_two: args.option_two_shares,
            b: args.current_b,
        });
        let cost_new = LmsrCost.transform(LmsrPriceInput {
            q_one: new_option_one_shares,
            q_two: new_option_two_shares,
            b: args.new_b,
        });
        RecapResult {
            new_option_one_shares,
            new_option_two_shares,
            new_b: args.new_b,
            capital_delta: (cost_new - cost_old).abs(),
        }
    }
}

// -----------------------------------------------------------------------------
// PnL ACCOUNTING (weighted-average cost basis, mirrors production pnl.ts)
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OptionAggregates {
    pub total_shares_bought: f64,
    pub total_shares_sold: f64,
    pub total_buy_amount: f64,
    pub total_sell_proceeds: f64,
    pub realized_pnl: f64,
}

/// `avgCostBasis = (totalBuyAmount − totalSellProceeds + realizedPnl) / netPosition`.
/// Returns 0 when netPosition ≤ 0.
pub fn avg_cost_basis(s: &OptionAggregates) -> f64 {
    let net = s.total_shares_bought - s.total_shares_sold;
    if net <= 0.0 {
        return 0.0;
    }
    (s.total_buy_amount - s.total_sell_proceeds + s.realized_pnl) / net
}

/// `max(0, totalSharesBought − totalSharesSold)`.
pub fn net_position(s: &OptionAggregates) -> f64 {
    (s.total_shares_bought - s.total_shares_sold).max(0.0)
}

/// Args for [`unrealized_pnl`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnrealizedPnlInput {
    pub net_position: f64,
    pub current_price: f64,
    pub avg_cost_basis: f64,
}

pub fn unrealized_pnl(args: UnrealizedPnlInput) -> f64 {
    if args.net_position <= 0.0 {
        return 0.0;
    }
    args.net_position * (args.current_price - args.avg_cost_basis)
}

/// Args for [`final_pnl`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalPnlInput {
    pub total_sell_proceeds: f64,
    pub total_buy_amount: f64,
    pub net_position: f64,
    pub resolution_price: f64,
}

/// Final PnL after market resolution:
/// `proceeds + (resolutionPrice · netPos) − totalBuy`.
pub fn final_pnl(args: FinalPnlInput) -> f64 {
    args.total_sell_proceeds + args.resolution_price * args.net_position - args.total_buy_amount
}

/// Buy or sell side of a replayed order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderAction {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayOrder {
    pub action: OrderAction,
    pub shares: f64,
    pub usdc: f64,
    pub time: Option<f64>,
}

/// Flattened `ReplayResult extends OptionAggregates`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayResult {
    pub total_shares_bought: f64,
    pub total_shares_sold: f64,
    pub total_buy_amount: f64,
    pub total_sell_proceeds: f64,
    pub realized_pnl: f64,
    pub net_position: f64,
    pub avg_cost_basis: f64,
    pub total_orders: usize,
    pub total_volume: f64,
    pub last_time: Option<f64>,
}

/// Replay a sequence of orders using weighted-average cost accounting.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayOrders;

impl Transform<&[ReplayOrder], ReplayResult> for ReplayOrders {
    fn transform(&self, orders: &[ReplayOrder]) -> ReplayResult {
        let mut total_shares_bought = 0.0;
        let mut total_shares_sold = 0.0;
        let mut total_buy_amount = 0.0;
        let mut total_sell_proceeds = 0.0;
        let mut realized_pnl = 0.0;
        let mut total_volume = 0.0;
        let mut last_time: Option<f64> = None;

        // Stable sort by time, with nulls first (to mirror production behaviour).
        let mut sorted: Vec<ReplayOrder> = orders.to_vec();
        sorted.sort_by(|a, b| match (a.time, b.time) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        });

        for o in &sorted {
            if let Some(t) = o.time {
                last_time = Some(t);
            }
            match o.action {
                OrderAction::Buy => {
                    total_shares_bought += o.shares;
                    total_buy_amount += o.usdc;
                    total_volume += o.usdc.abs();
                }
                OrderAction::Sell => {
                    let acb = avg_cost_basis(&OptionAggregates {
                        total_shares_bought,
                        total_shares_sold,
                        total_buy_amount,
                        total_sell_proceeds,
                        realized_pnl,
                    });
                    realized_pnl += o.usdc - o.shares * acb;
                    total_shares_sold += o.shares;
                    total_sell_proceeds += o.usdc;
                    total_volume += o.usdc.abs();
                }
            }
        }

        let aggregates = OptionAggregates {
            total_shares_bought,
            total_shares_sold,
            total_buy_amount,
            total_sell_proceeds,
            realized_pnl,
        };
        ReplayResult {
            total_shares_bought,
            total_shares_sold,
            total_buy_amount,
            total_sell_proceeds,
            realized_pnl,
            net_position: (total_shares_bought - total_shares_sold).max(0.0),
            avg_cost_basis: avg_cost_basis(&aggregates),
            total_orders: orders.len(),
            total_volume,
            last_time,
        }
    }
}

// -----------------------------------------------------------------------------
// HIGHER-LEVEL HELPERS / INVARIANT CHECKS
// -----------------------------------------------------------------------------

/// Args for [`BuyThenSellRoundTrip`]. `fee_bps`/`is_option_one` default to `0`/`true`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuyThenSellRoundTripInput {
    pub amount: f64,
    pub option_one_shares: f64,
    pub option_two_shares: f64,
    pub b: f64,
    pub is_option_one: Option<bool>,
    pub fee_bps: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundTripResult {
    pub buy: BuyExecution,
    pub sell: SellExecution,
    pub net: f64,
}

/// Buy then immediately sell all shares just bought. Production tests that the
/// proceeds are ≤ the original buy amount (no-arb).
#[derive(Clone, Copy, Debug, Default)]
pub struct BuyThenSellRoundTrip;

impl Transform<BuyThenSellRoundTripInput, RoundTripResult> for BuyThenSellRoundTrip {
    fn transform(&self, args: BuyThenSellRoundTripInput) -> RoundTripResult {
        let is_option_one = args.is_option_one.unwrap_or(true);
        let buy = BuyExecutor.transform(BuyExecutionInput {
            amount: args.amount,
            option_one_shares: args.option_one_shares,
            option_two_shares: args.option_two_shares,
            b: args.b,
            fee_bps: args.fee_bps,
            is_option_one: args.is_option_one,
        });
        let new_q1 = if is_option_one {
            args.option_one_shares + buy.shares
        } else {
            args.option_one_shares
        };
        let new_q2 = if is_option_one {
            args.option_two_shares
        } else {
            args.option_two_shares + buy.shares
        };
        let sell = SellExecutor.transform(SellExecutionInput {
            shares_out: buy.shares,
            option_one_shares: new_q1,
            option_two_shares: new_q2,
            b: args.b,
            is_option_one: args.is_option_one,
            fee_bps: args.fee_bps,
        });
        RoundTripResult { buy, sell, net: sell.usdc_out - args.amount }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_prices_are_half_and_sum_to_one() {
        // Equal shares => fair 50/50 market.
        let p = option_prices(10.0, 10.0, 50.0);
        assert!((p.option_one - 0.5).abs() < 1e-12);
        assert!((p.option_one + p.option_two - 1.0).abs() < 1e-12);

        // b = L / ln(2).
        let b = b_from_liquidity(500.0).unwrap();
        assert!((b - 500.0 / std::f64::consts::LN_2).abs() < 1e-9);
        assert!(b_from_liquidity(0.0).is_err());
    }

    #[test]
    fn lmsr_cost_matches_closed_form_when_balanced() {
        // With q1 == q2, C(q) = q + b·ln(2).
        let b = 100.0;
        let cost = LmsrCost.transform(LmsrPriceInput { q_one: 7.0, q_two: 7.0, b });
        assert!((cost - (7.0 + b * std::f64::consts::LN_2)).abs() < 1e-9);
    }

    #[test]
    fn buy_then_sell_round_trip_is_non_profitable() {
        // No-arb: selling back everything you just bought never exceeds the
        // original outlay (net <= 0 within float tolerance).
        let rt = BuyThenSellRoundTrip.transform(BuyThenSellRoundTripInput {
            amount: 100.0,
            option_one_shares: 0.0,
            option_two_shares: 0.0,
            b: 721.35,
            is_option_one: Some(true),
            fee_bps: None,
        });
        assert!(rt.buy.shares > 0.0);
        assert!(rt.net <= 1e-6, "round-trip net should be non-positive, got {}", rt.net);
    }

    #[test]
    fn replay_orders_weighted_average_accounting() {
        let orders = [
            ReplayOrder { action: OrderAction::Buy, shares: 10.0, usdc: 5.0, time: Some(2.0) },
            ReplayOrder { action: OrderAction::Buy, shares: 10.0, usdc: 5.0, time: None },
            ReplayOrder { action: OrderAction::Sell, shares: 5.0, usdc: 4.0, time: Some(3.0) },
        ];
        let r = ReplayOrders.transform(&orders);
        assert_eq!(r.total_orders, 3);
        assert!((r.total_shares_bought - 20.0).abs() < 1e-12);
        assert!((r.net_position - 15.0).abs() < 1e-12);
        // nulls sort first, so the last processed time is 3.0.
        assert_eq!(r.last_time, Some(3.0));
        // avg cost basis before the sell = 10/20 = 0.5; realized = 4 - 5*0.5 = 1.5.
        assert!((r.realized_pnl - 1.5).abs() < 1e-12);
    }
}
