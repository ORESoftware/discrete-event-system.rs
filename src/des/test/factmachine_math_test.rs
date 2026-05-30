//! Port of src/des/test/factmachine-math-test.ts
//!
//! Focused unit tests for the float64 FactMachine LMSR market math layer.
//! The TS free functions map onto a mix of free fns and `Transform` structs;
//! invalid-input cases that the TS catches with `try/catch` map onto either an
//! `Err` (for the `Result`-returning `b_from_liquidity`) or `#[should_panic]`
//! tests (for the kernels that `panic!` on invariant violations).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::f64::consts::LN_2;

    use crate::des::general::factmachine_math::{
        avg_cost_basis, b_from_liquidity, final_pnl, max_price_with_slippage,
        min_price_with_slippage, net_position, option_prices, unrealized_pnl, BuyExecutionInput,
        BuyExecutor, BuyThenSellRoundTrip, BuyThenSellRoundTripInput, FinalPnlInput, LmsrCost,
        LmsrPriceInput, OptionAggregates, OptionOnePrice, OrderAction, Recapitalization,
        RecapitalizationInput, ReplayOrder, ReplayOrders, SellExecutionInput, SellExecutor,
        SharesFromBudget, SharesFromBudgetInput, UnrealizedPnlInput,
    };
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};
    use crate::des::shared::transform::Transform;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }
    fn close_tol(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }
    fn option_one_price(q1: f64, q2: f64, b: f64) -> f64 {
        OptionOnePrice.transform(LmsrPriceInput {
            q_one: q1,
            q_two: q2,
            b,
        })
    }
    fn lmsr_cost(q1: f64, q2: f64, b: f64) -> f64 {
        LmsrCost.transform(LmsrPriceInput {
            q_one: q1,
            q_two: q2,
            b,
        })
    }
    fn shares_from_budget(budget: f64, current_price: f64, b: f64) -> f64 {
        SharesFromBudget.transform(SharesFromBudgetInput {
            budget,
            current_price,
            b,
        })
    }

    // Group 1 — bFromLiquidity.
    #[test]
    fn b_from_liquidity_cases() {
        assert!(close(b_from_liquidity(LN_2).unwrap(), 1.0));
        assert!(close(b_from_liquidity(100.0).unwrap(), 100.0 / LN_2));
        assert!(b_from_liquidity(0.0).is_err());
        assert!(b_from_liquidity(-1.0).is_err());
        assert!(b_from_liquidity(f64::NAN).is_err());
    }

    // Group 2 — optionOnePrice / optionPrices.
    #[test]
    fn option_prices_values_and_symmetry() {
        assert!(close(option_one_price(50.0, 50.0, 100.0), 0.5));
        assert!(close(option_one_price(1000.0, 0.0, 100.0), 1.0 / (1.0 + (-10.0_f64).exp())));
        assert!(close(option_one_price(0.0, 1000.0, 100.0), 1.0 / (1.0 + 10.0_f64.exp())));

        let mut rng = SeededRandom::new(2);
        for _ in 0..10 {
            let q1 = rng.next_float() * 100.0;
            let q2 = rng.next_float() * 100.0;
            let b = 50.0;
            let p12 = option_prices(q1, q2, b);
            let p21 = option_prices(q2, q1, b);
            assert!(
                (p12.option_one - p21.option_two).abs() <= 1e-12,
                "symmetry p1(q1,q2)=p2(q2,q1)"
            );
        }
    }

    #[test]
    #[should_panic]
    fn option_one_price_b_zero_panics() {
        let _ = option_one_price(0.0, 0.0, 0.0);
    }

    // Group 3 — lmsrCost monotonicity.
    #[test]
    fn lmsr_cost_monotonic_and_symmetric() {
        let before = lmsr_cost(0.0, 0.0, 50.0);
        let after_y = lmsr_cost(10.0, 0.0, 50.0);
        let after_n = lmsr_cost(0.0, 10.0, 50.0);
        assert!(after_y > before);
        assert!(after_n > before);

        let mut rng = SeededRandom::new(3);
        for _ in 0..10 {
            let q1 = rng.next_float() * 100.0;
            let q2 = rng.next_float() * 100.0;
            assert!((lmsr_cost(q1, q2, 50.0) - lmsr_cost(q2, q1, 50.0)).abs() <= 1e-12);
        }
    }

    // Group 4 — buyExecution.
    #[test]
    fn buy_execution_values() {
        let r = BuyExecutor.transform(BuyExecutionInput {
            amount: 10.0,
            option_one_shares: 0.0,
            option_two_shares: 0.0,
            b: 100.0,
            fee_bps: None,
            is_option_one: None,
        });
        assert!(r.shares > 0.0, "shares = {}", r.shares);
        assert!(close(r.buy_amount, 10.0));
        assert!(close(r.fee_amount, 0.0));
        assert!(close(r.reward, r.shares));
        assert!(close(r.average_price, r.buy_amount / r.shares));

        let r2 = BuyExecutor.transform(BuyExecutionInput {
            amount: 10.0,
            option_one_shares: 0.0,
            option_two_shares: 0.0,
            b: 100.0,
            fee_bps: Some(100.0),
            is_option_one: None,
        });
        assert!(close(r2.fee_amount, 0.10));
        assert!(close(r2.buy_amount, 9.90));
    }

    #[test]
    #[should_panic]
    fn buy_execution_high_fee_panics() {
        let _ = BuyExecutor.transform(BuyExecutionInput {
            amount: 10.0,
            option_one_shares: 0.0,
            option_two_shares: 0.0,
            b: 100.0,
            fee_bps: Some(10000.0),
            is_option_one: None,
        });
    }

    // Group 5 — sellExecution.
    #[test]
    fn sell_execution_values() {
        let r = SellExecutor.transform(SellExecutionInput {
            shares_out: 5.0,
            option_one_shares: 50.0,
            option_two_shares: 30.0,
            b: 100.0,
            fee_bps: None,
            is_option_one: None,
        });
        assert!(r.sell_amount >= 0.0);
        assert!(close(r.usdc_out, r.sell_amount));
        assert!(close(r.reward, r.usdc_out));
        assert!(close(r.average_price, r.sell_amount / 5.0));

        let r2 = SellExecutor.transform(SellExecutionInput {
            shares_out: 5.0,
            option_one_shares: 50.0,
            option_two_shares: 30.0,
            b: 100.0,
            fee_bps: Some(50.0),
            is_option_one: None,
        });
        assert!(close(r2.usdc_out + r2.fee_amount, r2.sell_amount));
    }

    // Group 6 — buy-then-sell round-trip is non-positive.
    #[test]
    fn round_trip_never_profitable() {
        let mut rng = SeededRandom::new(6);
        for _ in 0..30 {
            let amount = 1.0 + rng.next_float() * 50.0;
            let q1 = rng.next_float() * 200.0;
            let q2 = rng.next_float() * 200.0;
            let rt = BuyThenSellRoundTrip.transform(BuyThenSellRoundTripInput {
                amount,
                option_one_shares: q1,
                option_two_shares: q2,
                b: 1000.0,
                is_option_one: None,
                fee_bps: None,
            });
            assert!(rt.net <= 1e-9, "round-trip net={}", rt.net);
        }
    }

    // Group 7 — recapitalization.
    #[test]
    fn recapitalization_preserves_prices() {
        let before = option_prices(50.0, 30.0, 100.0);
        let r = Recapitalization.transform(RecapitalizationInput {
            option_one_shares: 50.0,
            option_two_shares: 30.0,
            current_b: 100.0,
            new_b: 200.0,
        });
        let after = option_prices(r.new_option_one_shares, r.new_option_two_shares, r.new_b);
        assert!(close(after.option_one, before.option_one));
        assert!(close(after.option_two, before.option_two));
        assert!(r.capital_delta > 0.0, "Δ = {}", r.capital_delta);
        assert!(close(r.new_option_one_shares, 50.0 * 200.0 / 100.0));
    }

    #[test]
    #[should_panic]
    fn recapitalization_equal_b_panics() {
        let _ = Recapitalization.transform(RecapitalizationInput {
            option_one_shares: 0.0,
            option_two_shares: 0.0,
            current_b: 100.0,
            new_b: 100.0,
        });
    }

    // Group 8 — slippage helpers.
    #[test]
    fn slippage_helpers() {
        assert!(close(max_price_with_slippage(0.5, 0.0), 0.5));
        assert!(close(min_price_with_slippage(0.5, 0.0), 0.5));
        assert!(max_price_with_slippage(0.5, 100.0) >= 0.5);
        assert!(min_price_with_slippage(0.5, 100.0) <= 0.5);
        assert!(close(max_price_with_slippage(0.99, 5000.0), 1.0));
        assert!(close(min_price_with_slippage(0.01, 5000.0), 0.005));
    }

    // Group 9 — PnL helpers.
    #[test]
    fn pnl_helpers() {
        let acb = avg_cost_basis(&OptionAggregates {
            total_shares_bought: 10.0,
            total_shares_sold: 0.0,
            total_buy_amount: 4.0,
            total_sell_proceeds: 0.0,
            realized_pnl: 0.0,
        });
        assert!(close(acb, 0.4));

        assert!(close(
            net_position(&OptionAggregates {
                total_shares_bought: 10.0,
                total_shares_sold: 3.0,
                ..Default::default()
            }),
            7.0
        ));
        assert!(close(
            net_position(&OptionAggregates {
                total_shares_bought: 3.0,
                total_shares_sold: 10.0,
                ..Default::default()
            }),
            0.0
        ));

        assert!(close(
            final_pnl(FinalPnlInput {
                total_buy_amount: 4.0,
                total_sell_proceeds: 0.0,
                net_position: 7.0,
                resolution_price: 1.0,
            }),
            3.0
        ));

        assert!(close(
            unrealized_pnl(UnrealizedPnlInput {
                net_position: 7.0,
                current_price: 0.6,
                avg_cost_basis: 0.4,
            }),
            1.4
        ));
        assert!(close(
            unrealized_pnl(UnrealizedPnlInput {
                net_position: 0.0,
                current_price: 0.6,
                avg_cost_basis: 0.4,
            }),
            0.0
        ));
    }

    // Group 10 — replayOrders.
    #[test]
    fn replay_orders() {
        let orders = [
            ReplayOrder {
                action: OrderAction::Buy,
                shares: 10.0,
                usdc: 4.0,
                time: Some(1.0),
            },
            ReplayOrder {
                action: OrderAction::Sell,
                shares: 4.0,
                usdc: 2.0,
                time: Some(2.0),
            },
        ];
        let r = ReplayOrders.transform(&orders);
        assert!(close(r.total_shares_bought, 10.0));
        assert!(close(r.total_shares_sold, 4.0));
        assert!(close(r.total_buy_amount, 4.0));
        assert!(close(r.total_sell_proceeds, 2.0));
        assert!(close(r.net_position, 6.0));
        assert!(close(r.total_volume, 6.0));
        assert_eq!(r.last_time, Some(2.0));
        assert!(close_tol(r.realized_pnl, 0.4, 1e-12));
    }

    // Group 11 — sharesFromBudget edge cases.
    #[test]
    fn shares_from_budget_edges() {
        assert!(close(shares_from_budget(0.0, 0.5, 100.0), 0.0));
        assert!(close(shares_from_budget(-5.0, 0.5, 100.0), 0.0));
        let small = shares_from_budget(1e-6, 0.5, 100.0);
        assert!(close_tol(small, 1e-6 / 0.5, 1e-9));
    }

    #[test]
    #[should_panic]
    fn shares_from_budget_zero_price_panics() {
        let _ = shares_from_budget(10.0, 0.0, 100.0);
    }

    #[test]
    #[should_panic]
    fn shares_from_budget_zero_b_panics() {
        let _ = shares_from_budget(10.0, 0.5, 0.0);
    }
}
