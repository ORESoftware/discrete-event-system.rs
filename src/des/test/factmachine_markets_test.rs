//! Port of src/des/test/factmachine-markets-test.ts
//
// `main_factmachine_markets` (the prediction-market portfolio simulation) is now
// ported, so the TS `check()`/PASS-FAIL harness becomes real `#[test]` fns.

#[cfg(test)]
mod tests {
    use crate::des::main_factmachine_markets::{
        build_daily_market_caps, build_operator_mdp, daily_market_cap_for_day, default_config,
        run_portfolio, MarketKind, SchedulerPolicy,
    };

    // T1 — daily launch-cap schedule.
    #[test]
    fn daily_launch_cap_schedule_is_well_formed() {
        let caps = build_daily_market_caps(50.0, 2, 10, 42);
        assert_eq!(caps.len(), 50, "expected one cap per day");
        assert!(caps.iter().all(|&x| (2..=10).contains(&x)), "cap out of [2,10]: {caps:?}");
        assert!(caps.contains(&2), "schedule never hits the floor of 2");
        assert!(caps.contains(&10), "schedule never hits the ceiling of 10");

        // Past the end of the schedule the cap clamps to the last entry.
        let cfg = default_config();
        let last = *cfg.daily_market_caps.last().expect("non-empty cap schedule");
        assert_eq!(daily_market_cap_for_day(999_999, &cfg), last);
    }

    // T2 — MDP/POMDP portfolio run captures contract kinds and daily summaries.
    #[test]
    fn portfolio_run_breakdowns_are_consistent() {
        let mdp = build_operator_mdp();
        let cfg = default_config();
        let run = run_portfolio(SchedulerPolicy::GreedyBuzz, &cfg, &mdp);

        let closed = run.closed_markets.len() as i64;
        assert!(closed > 0, "no markets closed");

        // Each opened-per-day count obeys that day's cap (for in-horizon days).
        let horizon_days = (cfg.horizon_h / 24.0).ceil() as i64;
        assert!(
            run.daily.iter().filter(|d| d.day < horizon_days).all(|d| d.opened <= d.market_cap),
            "a day opened more markets than its cap allowed"
        );

        // Kind breakdown and daily summaries must each account for every closed
        // market exactly once.
        let total_by_kind: i64 = run.kind_breakdown.iter().map(|r| r.markets).sum();
        let total_by_day: i64 = run.daily.iter().map(|d| d.closed).sum();
        assert_eq!(total_by_kind, closed, "kind breakdown does not sum to closed markets");
        assert_eq!(total_by_day, closed, "daily summaries do not sum to closed markets");

        // The run should exercise both binary and scalar contract kinds.
        let kind_markets = |k: MarketKind| -> i64 {
            run.kind_breakdown.iter().filter(|r| r.kind == k).map(|r| r.markets).sum()
        };
        assert!(kind_markets(MarketKind::Binary) > 0, "run has no binary markets");
        assert!(kind_markets(MarketKind::Scalar) > 0, "run has no scalar markets");

        // The timeline must capture real betting activity at some point.
        assert!(
            run.timeline.iter().any(|x| x.bettors > 0 && x.trades > 0),
            "timeline never recorded both bettors and trades"
        );
    }
}
