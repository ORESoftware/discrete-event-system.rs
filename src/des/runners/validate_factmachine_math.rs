//! Port of `src/des/runners/validate-factmachine-math.ts`.
//!
//! Audits the FactMachine LMSR pricing math: Part A float64 invariants (prices
//! sum to 1, symmetry, monotonicity, buy/sell round-trips, recap, slippage),
//! Part B cross-validation vs the production `@factmachine/math` JS package, and
//! Part C the POMDP `LMSR` class conventions. Driver → [`run`].
//!
//! PORT NOTES:
//!   * The LMSR math (`b_from_liquidity`, `option_prices`, `lmsr_cost`,
//!     `buy_execution`, `sell_execution`, slippage, recap, round-trip, `LMSR`) is
//!     ported faithfully here so Part A / Part C invariants hold. Wire to
//!     `crate::des::general::factmachine_math` + `crate::des::main_factmachine::LMSR`
//!     when available.
//!   * `crate::des::general::prng::mulberry32`.
//!   * Part B `require('@factmachine/math/.../trading.js')` is a Node JS module
//!     load with no Rust analog → always SKIP (matches the TS "not loadable" path).

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// PRNG.
// =============================================================================

fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x6D2B_79F5);
        let mut t = (s ^ (s >> 15)).wrapping_mul(1 | s);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

// =============================================================================
// LMSR math (faithful).
// =============================================================================

fn b_from_liquidity(liquidity: f64) -> f64 {
    liquidity / (2.0_f64).ln()
}

fn option_prices(q1: f64, q2: f64, b: f64) -> (f64, f64) {
    // p1 = exp(q1/b) / (exp(q1/b)+exp(q2/b)) = sigmoid((q1-q2)/b); p2 = 1 - p1.
    let p1 = 1.0 / (1.0 + (-(q1 - q2) / b).exp());
    (p1, 1.0 - p1)
}

fn option_one_price(q1: f64, q2: f64, b: f64) -> f64 {
    option_prices(q1, q2, b).0
}

fn lmsr_cost(q1: f64, q2: f64, b: f64) -> f64 {
    let m = q1.max(q2);
    m + b * (((q1 - m) / b).exp() + ((q2 - m) / b).exp()).ln()
}

struct BuyResult {
    shares: f64,
    buy_amount: f64,
    fee_amount: f64,
    average_price: f64,
}

struct BuyInput {
    amount: f64,
    option_one_shares: f64,
    option_two_shares: f64,
    b: f64,
    fee_bps: f64,
    is_option_one: bool,
}

fn buy_execution(i: BuyInput) -> BuyResult {
    let fee_amount = i.amount * i.fee_bps / 10000.0;
    let buy_amount = i.amount - fee_amount;
    let e = (i.option_one_shares / i.b).exp() + (i.option_two_shares / i.b).exp();
    let factor = e * (buy_amount / i.b).exp();
    let shares = if i.is_option_one {
        i.b * (factor - (i.option_two_shares / i.b).exp()).ln() - i.option_one_shares
    } else {
        i.b * (factor - (i.option_one_shares / i.b).exp()).ln() - i.option_two_shares
    };
    let average_price = if shares != 0.0 { buy_amount / shares } else { 0.0 };
    BuyResult { shares, buy_amount, fee_amount, average_price }
}

struct SellResult {
    usdc_out: f64,
    fee_amount: f64,
    sell_amount: f64,
}

struct SellInput {
    shares_out: f64,
    option_one_shares: f64,
    option_two_shares: f64,
    b: f64,
    fee_bps: f64,
    is_option_one: bool,
}

fn sell_execution(i: SellInput) -> SellResult {
    let before = lmsr_cost(i.option_one_shares, i.option_two_shares, i.b);
    let after = if i.is_option_one {
        lmsr_cost(i.option_one_shares - i.shares_out, i.option_two_shares, i.b)
    } else {
        lmsr_cost(i.option_one_shares, i.option_two_shares - i.shares_out, i.b)
    };
    let sell_amount = before - after;
    let fee_amount = sell_amount * i.fee_bps / 10000.0;
    SellResult { usdc_out: sell_amount - fee_amount, fee_amount, sell_amount }
}

struct RoundTripResult {
    net: f64,
}

fn buy_then_sell_round_trip(amount: f64, q1: f64, q2: f64, b: f64, is_option_one: bool) -> RoundTripResult {
    let buy = buy_execution(BuyInput { amount, option_one_shares: q1, option_two_shares: q2, b, fee_bps: 0.0, is_option_one });
    let (nq1, nq2) = if is_option_one { (q1 + buy.shares, q2) } else { (q1, q2 + buy.shares) };
    let sell = sell_execution(SellInput { shares_out: buy.shares, option_one_shares: nq1, option_two_shares: nq2, b, fee_bps: 0.0, is_option_one });
    RoundTripResult { net: sell.usdc_out - amount }
}

struct RecapResult {
    new_option_one_shares: f64,
    new_option_two_shares: f64,
    new_b: f64,
    capital_delta: f64,
}

fn recapitalization(q1: f64, q2: f64, current_b: f64, new_b: f64) -> RecapResult {
    let scale = new_b / current_b;
    let n1 = q1 * scale;
    let n2 = q2 * scale;
    let capital_delta = lmsr_cost(n1, n2, new_b) - lmsr_cost(q1, q2, current_b);
    RecapResult { new_option_one_shares: n1, new_option_two_shares: n2, new_b, capital_delta }
}

fn max_price_with_slippage(price: f64, slippage_bps: f64) -> f64 {
    (price * (1.0 + slippage_bps / 10000.0)).min(1.0)
}

fn min_price_with_slippage(price: f64, slippage_bps: f64) -> f64 {
    (price * (1.0 - slippage_bps / 10000.0)).max(0.0)
}

/// POMDP-side LMSR class (`main-factmachine`'s `LMSR`).
struct Lmsr {
    b: f64,
    q1: f64,
    q2: f64,
}

impl Lmsr {
    fn new(liquidity: f64, num_options: f64, liquidity_is_b: bool) -> Self {
        let b = if liquidity_is_b { liquidity } else { liquidity / num_options.ln() };
        Lmsr { b, q1: 0.0, q2: 0.0 }
    }
    fn binary_prices(&self) -> (f64, f64) {
        option_prices(self.q1, self.q2, self.b)
    }
    fn buy(&mut self, amount: f64, is_yes: bool, fee_bps: f64) -> BuyResult {
        let r = buy_execution(BuyInput { amount, option_one_shares: self.q1, option_two_shares: self.q2, b: self.b, fee_bps, is_option_one: is_yes });
        if is_yes {
            self.q1 += r.shares;
        } else {
            self.q2 += r.shares;
        }
        r
    }
    fn sell(&mut self, shares: f64, is_yes: bool, fee_bps: f64) -> SellResult {
        let r = sell_execution(SellInput { shares_out: shares, option_one_shares: self.q1, option_two_shares: self.q2, b: self.b, fee_bps, is_option_one: is_yes });
        if is_yes {
            self.q1 -= shares;
        } else {
            self.q2 -= shares;
        }
        r
    }
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() { String::new() } else { format!("  — {}", detail) };
        println!("{}  {}{}", if ok { "  PASS" } else { "  FAIL" }, label, tail);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
    fn close(&mut self, label: &str, a: f64, b: f64, tol: f64) {
        let d = (a - b).abs();
        self.check(label, d <= tol, &format!("|{} − {}| = {:.2e}", a, b, d));
    }
}

/// `validate-factmachine-math.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\n=== PART A — invariants on float64 math layer ===\n");
    let mut rng = mulberry32(424242);
    const TRIALS: usize = 200;

    // A1 / A2.
    {
        let mut ok = true;
        let mut detail = String::new();
        let mut early = false;
        for t in 0..TRIALS {
            let q1 = rng() * 10_000.0;
            let q2 = rng() * 10_000.0;
            let liq = 500.0 + rng() * 100_000.0;
            let b = b_from_liquidity(liq);
            let p = option_prices(q1, q2, b);
            if (p.0 + p.1 - 1.0).abs() > 1e-12 {
                c.check("A1 prices sum to 1 (sample failure)", false, &format!("t={}, q1={}, q2={}, sum={}", t, q1, q2, p.0 + p.1));
                early = true;
                break;
            }
            if !(p.0 > 0.0 && p.0 < 1.0 && p.1 > 0.0 && p.1 < 1.0) {
                c.check("A2 prices strictly in (0, 1)", false, &format!("t={}, p1={}, p2={}", t, p.0, p.1));
                early = true;
                break;
            }
        }
        if !early {
            c.check(&format!("A1 prices sum to 1 across {} random trials", TRIALS), true, "");
            c.check(&format!("A2 prices strictly in (0, 1) across {} random trials", TRIALS), true, "");
        }
    }

    // A3 equal-shares.
    {
        let mut early = false;
        for t in 0..50 {
            let q = rng() * 5_000.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let p = option_prices(q, q, b);
            if (p.0 - 0.5).abs() > 1e-12 {
                c.check("A3 equal-shares ⇒ price = 0.5", false, &format!("t={}, q={}, p={}", t, q, p.0));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A3 equal-shares ⇒ price = 0.5 across 50 trials", true, "");
        }
    }

    // A4 symmetry.
    {
        let mut early = false;
        for t in 0..50 {
            let q1 = rng() * 5_000.0;
            let q2 = rng() * 5_000.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let p12 = option_prices(q1, q2, b);
            let p21 = option_prices(q2, q1, b);
            if (p12.0 - p21.1).abs() > 1e-12 {
                c.check("A4 symmetry under share swap", false, &format!("t={}", t));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A4 p₁(q1, q2) = p₂(q2, q1) across 50 trials", true, "");
        }
    }

    // A5 monotonicity.
    {
        let mut early = false;
        for t in 0..50 {
            let q1 = rng() * 1_000.0;
            let q2 = rng() * 1_000.0;
            let delta = 1.0 + rng() * 100.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            if option_one_price(q1 + delta, q2, b) < option_one_price(q1, q2, b) - 1e-12 {
                c.check("A5 more shares on option-1 raises price-1", false, &format!("t={}", t));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A5 more shares on option-1 raises price-1 across 50 trials", true, "");
        }
    }

    // A6-A9 buy invariants.
    {
        let mut early = false;
        for t in 0..50 {
            let amount = 0.01 + rng() * 100.0;
            let q1 = rng() * 1_000.0;
            let q2 = rng() * 1_000.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let fee = (rng() * 1000.0).floor();
            let is_one = rng() < 0.5;
            let r = buy_execution(BuyInput { amount, option_one_shares: q1, option_two_shares: q2, b, fee_bps: fee, is_option_one: is_one });
            if (r.buy_amount + r.fee_amount - amount).abs() > 1e-9 {
                c.check("A6 buy: fee + buyAmount = total", false, &format!("t={}", t));
                early = true;
                break;
            }
            if !(r.shares > 0.0) {
                c.check("A7 buy: amount>0 ⇒ shares>0", false, &format!("t={}", t));
                early = true;
                break;
            }
            if (r.average_price - r.buy_amount / r.shares).abs() > 1e-12 {
                c.check("A8 buy: averagePrice = buyAmount / shares", false, &format!("t={}", t));
                early = true;
                break;
            }
            if !(r.average_price < 1.0) {
                c.check("A9 buy: averagePrice < 1", false, &format!("t={}, ap={}", t, r.average_price));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A6 buy: fee + buyAmount = total (50 trials)", true, "");
            c.check("A7 buy: amount > 0 ⇒ shares > 0 (50 trials)", true, "");
            c.check("A8 buy: averagePrice = buyAmount / shares (50 trials)", true, "");
            c.check("A9 buy: averagePrice strictly < 1 (50 trials)", true, "");
        }
    }

    // A10 buy monotone in spending.
    {
        let mut early = false;
        for t in 0..30 {
            let a = 0.01 + rng() * 50.0;
            let b_amt = 0.01 + rng() * 50.0;
            let q1 = rng() * 100.0;
            let q2 = rng() * 100.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let small = a.min(b_amt);
            let large = a.max(b_amt);
            let rs = buy_execution(BuyInput { amount: small, option_one_shares: q1, option_two_shares: q2, b, fee_bps: 0.0, is_option_one: true });
            let rl = buy_execution(BuyInput { amount: large, option_one_shares: q1, option_two_shares: q2, b, fee_bps: 0.0, is_option_one: true });
            if rl.shares < rs.shares - 1e-12 {
                c.check("A10 buy: monotone in spending", false, &format!("t={}", t));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A10 buy: monotone in spending (30 trials)", true, "");
        }
    }

    // A11 sell fee balance.
    {
        let mut early = false;
        for t in 0..30 {
            let shares_out = 0.01 + rng() * 30.0;
            let q1 = shares_out + 50.0 + rng() * 100.0;
            let q2 = rng() * 100.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let fee = (rng() * 1000.0).floor();
            let r = sell_execution(SellInput { shares_out, option_one_shares: q1, option_two_shares: q2, b, fee_bps: fee, is_option_one: true });
            if (r.usdc_out + r.fee_amount - r.sell_amount).abs() > 1e-9 {
                c.check("A11 sell: usdcOut + fee = sellAmount", false, &format!("t={}", t));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A11 sell: usdcOut + fee = sellAmount (30 trials)", true, "");
        }
    }

    // A12 sell monotone.
    {
        let mut early = false;
        for t in 0..30 {
            let a = 0.01 + rng() * 30.0;
            let b_amt = 0.01 + rng() * 30.0;
            let q1 = a + b_amt + 50.0;
            let q2 = rng() * 100.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let small = a.min(b_amt);
            let large = a.max(b_amt);
            let rs = sell_execution(SellInput { shares_out: small, option_one_shares: q1, option_two_shares: q2, b, fee_bps: 0.0, is_option_one: true });
            let rl = sell_execution(SellInput { shares_out: large, option_one_shares: q1, option_two_shares: q2, b, fee_bps: 0.0, is_option_one: true });
            if rl.usdc_out < rs.usdc_out - 1e-12 {
                c.check("A12 sell: monotone in shares", false, &format!("t={}", t));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A12 sell: monotone in shares (30 trials)", true, "");
        }
    }

    // A13 round-trip no-arb.
    {
        let mut early = false;
        let mut max_loss = 0.0_f64;
        for t in 0..100 {
            let amount = 1.0 + rng() * 100.0;
            let q1 = rng() * 200.0;
            let q2 = rng() * 200.0;
            let b = b_from_liquidity(500.0 + rng() * 50_000.0);
            let is_one = rng() < 0.5;
            let rt = buy_then_sell_round_trip(amount, q1, q2, b, is_one);
            if rt.net > 1e-8 {
                c.check("A13 round-trip: sell-after-buy ≤ amount (no-arb)", false, &format!("t={}, net={:.2e}", t, rt.net));
                early = true;
                break;
            }
            if -rt.net > max_loss {
                max_loss = -rt.net;
            }
        }
        if !early {
            c.check(&format!("A13 round-trip ≤ buy amount across 100 trials  (max market-maker spread = {:.2e})", max_loss), true, "");
        }
    }

    // A14 recapitalisation preserves prices.
    {
        let mut early = false;
        for t in 0..30 {
            let q1 = rng() * 1_000.0;
            let q2 = rng() * 1_000.0;
            let liq_old = 500.0 + rng() * 50_000.0;
            let liq_new = liq_old * (1.1 + rng() * 2.0);
            let b_old = b_from_liquidity(liq_old);
            let b_new = b_from_liquidity(liq_new);
            let before = option_prices(q1, q2, b_old);
            let r = recapitalization(q1, q2, b_old, b_new);
            let after = option_prices(r.new_option_one_shares, r.new_option_two_shares, r.new_b);
            if (after.0 - before.0).abs() > 1e-10 {
                c.check("A14 recapitalisation preserves prices", false, &format!("t={}, |Δp|={:.2e}", t, (after.0 - before.0).abs()));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A14 recapitalisation preserves prices (30 trials)", true, "");
        }
    }

    // A15 slippage clamps.
    {
        let mut early = false;
        for t in 0..30 {
            let price = 0.01 + rng() * 0.98;
            let slip = (rng() * 2000.0).floor();
            let max = max_price_with_slippage(price, slip);
            let min = min_price_with_slippage(price, slip);
            if !((0.0..=1.0).contains(&max) && (0.0..=1.0).contains(&min)) {
                c.check("A15 slippage clamps to [0, 1]", false, &format!("t={}, max={}, min={}", t, max, min));
                early = true;
                break;
            }
            if !(max >= price - 1e-15 && min <= price + 1e-15) {
                c.check("A15 maxPrice ≥ price ≥ minPrice", false, &format!("t={}, price={}, max={}, min={}", t, price, max, min));
                early = true;
                break;
            }
        }
        if !early {
            c.check("A15 slippage clamps to [0, 1] and maxPrice ≥ price ≥ minPrice (30 trials)", true, "");
        }
    }

    // PART B — cross-validation vs production @factmachine/math.
    println!("\n=== PART B — cross-validation vs production @factmachine/math ===\n");
    let prod_trading_path = std::env::var("FACTMACHINE_TRADING_PATH")
        .unwrap_or_else(|_| "/Users/maca5/codes/factmachine/factmachine-monorepo/packages/math/dist/trading.js".to_string());
    // PORT NOTE: `require(...)` loads a JS package; no Rust analog → always SKIP.
    println!("  SKIP (production package not loadable at {})", prod_trading_path);
    println!("  Set FACTMACHINE_TRADING_PATH to override; or `pnpm -C ../.. build` in the monorepo.");

    // PART C — POMDP LMSR class uses production conventions.
    println!("\n=== PART C — POMDP LMSR class uses production conventions ===\n");
    {
        let mut m = Lmsr::new(50.0, 2.0, false);
        c.close("C1 default LMSR(50, 2) gives b = 50/ln(2)", m.b, 50.0 / (2.0_f64).ln(), 1e-12);
        let m_legacy = Lmsr::new(50.0, 2.0, true);
        c.close("C2 LMSR(50, 2, {liquidityIsB:true}) gives b = 50", m_legacy.b, 50.0, 1e-12);
        let p = m.binary_prices();
        c.close("C3 LMSR.binaryPrices() at q=0 returns 0.5/0.5", p.0, 0.5, 1e-12);
        let exec = m.buy(10.0, true, 0.0);
        c.check("C4 LMSR.buy(10, YES) returns positive shares", exec.shares > 0.0, &format!("shares={:.6}", exec.shares));
        let p_after = m.binary_prices();
        c.check("C5 P(YES) rose after buying YES", p_after.0 > 0.5, &format!("P(YES) before=0.5, after={:.6}", p_after.0));
        let shares = exec.shares;
        let sell_back = m.sell(shares, true, 0.0);
        c.check("C6 LMSR sell-back ≤ original amount (no-arb)", sell_back.usdc_out <= 10.0 + 1e-9, &format!("sellBack={:.6} ≤ 10", sell_back.usdc_out));
    }

    println!("\n{} checks: {} passed, {} failed", c.pass + c.fail, c.pass, c.fail);
    if c.fail > 0 {
        std::process::exit(1);
    }
}
