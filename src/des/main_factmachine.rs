//! Port of `src/des/main-factmachine.ts`.
//!
//! POMDP model of an LMSR opinion market (factmachine.com): a market maker,
//! noise traders, a Bayesian bettor over a θ-grid belief, and voter resolution.
//! Defines the model AND runs it (single run or multi-rep policy comparison).
//!
//! Reuses `general::belief` (`DiscreteBelief`, `brier_score`), `general::prng`
//! (`mulberry32`), `general::random_variables` (`sample_poisson`), and the
//! production-equivalent LMSR algebra in `general::factmachine_math`.
//!
//! PORT NOTES:
//!   * the TS stations all share ONE `mulberry32` closure, so RNG draws are
//!     interleaved deterministically. The orchestrator therefore owns a single
//!     `SeededRandom` and threads `&mut rng` through each phase rather than
//!     embedding it per station.
//!   * the `field-station.Station` base is inlined: stations are plain structs
//!     driven directly by `run_fact_machine`'s fixed 6-phase tick order.
//!   * `LMSR.recap()` mutates the (TS-`readonly`) `b` field, mirrored here with a
//!     plain mutable field.
//!   * the `ANIMATE=1` path needs `animation/scenes/factmachine-scene` +
//!     `animation/frame-recorder` (not ported); it is stubbed with a notice.

#![allow(dead_code)]

use crate::des::general::belief::{brier_score, BinaryOutcome, DiscreteBelief};
use crate::des::general::factmachine_math::{
    option_prices, BuyExecution, BuyExecutionInput, BuyExecutor, LmsrCost, LmsrPriceInput, OptionPrices,
    RecapResult, Recapitalization, RecapitalizationInput, SellExecution, SellExecutionInput, SellExecutor,
};
use crate::des::general::prng::mulberry32;
use crate::des::general::random_variables::sample_poisson;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// Enums (TS string unions).
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Random,
    Hold,
    Myopic,
    Qmdp,
    Oracle,
}
impl Policy {
    fn slug(self) -> &'static str {
        match self {
            Policy::Random => "random",
            Policy::Hold => "hold",
            Policy::Myopic => "myopic",
            Policy::Qmdp => "qmdp",
            Policy::Oracle => "oracle",
        }
    }
    fn from_slug(s: &str) -> Policy {
        match s {
            "random" => Policy::Random,
            "hold" => Policy::Hold,
            "myopic" => Policy::Myopic,
            "oracle" => Policy::Oracle,
            _ => Policy::Qmdp,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMode {
    Bernoulli,
    Majority,
}
impl ResolutionMode {
    fn slug(self) -> &'static str {
        match self {
            ResolutionMode::Bernoulli => "bernoulli",
            ResolutionMode::Majority => "majority",
        }
    }
    fn from_slug(s: &str) -> ResolutionMode {
        match s {
            "majority" => ResolutionMode::Majority,
            _ => ResolutionMode::Bernoulli,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MarketType {
    Binary,
    Scalar,
}
impl MarketType {
    fn slug(self) -> &'static str {
        match self {
            MarketType::Binary => "binary",
            MarketType::Scalar => "scalar",
        }
    }
    fn from_slug(s: &str) -> MarketType {
        match s {
            "scalar" => MarketType::Scalar,
            _ => MarketType::Binary,
        }
    }
}

#[derive(Clone)]
pub struct FactMachineParams {
    pub t: i64,
    pub n_voters: i64,
    pub k_noise: f64,
    pub informedness: f64,
    pub fee: f64,
    pub liquidity: f64,
    pub theta_bins: usize,
    pub true_theta: f64,
    pub seed: u32,
    pub policy: Policy,
    pub resolution_mode: ResolutionMode,
    pub market_type: MarketType,
    pub late_flip: bool,
    pub late_flip_multiplier: f64,
}

pub fn default_params() -> FactMachineParams {
    FactMachineParams {
        t: 24,
        n_voters: 51,
        k_noise: 20.0,
        informedness: 0.6,
        fee: 0.01,
        liquidity: 50.0,
        theta_bins: 21,
        true_theta: 0.65,
        seed: 1,
        policy: Policy::Qmdp,
        resolution_mode: ResolutionMode::Bernoulli,
        market_type: MarketType::Binary,
        late_flip: false,
        late_flip_multiplier: 10.0,
    }
}

/// Probability that YES wins, given θ and the resolution mode.
fn p_yes_wins(theta: f64, params: &FactMachineParams) -> f64 {
    if params.resolution_mode == ResolutionMode::Bernoulli {
        return theta;
    }
    let n = params.n_voters;
    let half = n / 2;
    let mut p = 0.0;
    let mut log_p = n as f64 * (1e-300_f64.max(1.0 - theta)).ln();
    let mut lcoef = 0.0;
    for k in 0..=n {
        if k > half {
            p += (lcoef + log_p).exp();
        }
        if k < n {
            lcoef += ((n - k) as f64).ln() - ((k + 1) as f64).ln();
            log_p += (1e-300_f64.max(theta)).ln() - (1e-300_f64.max(1.0 - theta)).ln();
        }
    }
    p.clamp(0.0, 1.0)
}

// -----------------------------------------------------------------------------
// LMSR market maker.
// -----------------------------------------------------------------------------

fn lmsr_cost(q0: f64, q1: f64, b: f64) -> f64 {
    LmsrCost.transform(LmsrPriceInput { q_one: q0, q_two: q1, b })
}

pub struct LMSR {
    pub q: Vec<f64>,
    pub b: f64,
    pub n: usize,
    pub liquidity: f64,
}
impl LMSR {
    pub fn new(liquidity: f64, n: usize, liquidity_is_b: bool) -> Self {
        if n < 2 {
            panic!("LMSR: need N ≥ 2 outcomes");
        }
        if !liquidity.is_finite() || liquidity <= 0.0 {
            panic!("LMSR: liquidity must be a finite positive number");
        }
        let b = if liquidity_is_b { liquidity } else { liquidity / (n as f64).ln() };
        LMSR { q: vec![0.0; n], b, n, liquidity }
    }
    pub fn prices(&self) -> Vec<f64> {
        let m = self.q.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / self.b;
        let exps: Vec<f64> = self.q.iter().map(|qi| (qi / self.b - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter().map(|e| e / s).collect()
    }
    pub fn price_yes(&self) -> f64 {
        self.prices()[0]
    }
    pub fn binary_prices(&self) -> OptionPrices {
        if self.n != 2 {
            panic!("LMSR.binary_prices() requires N = 2");
        }
        option_prices(self.q[0], self.q[1], self.b)
    }
    pub fn cost(&self, dq: &[f64]) -> f64 {
        if dq.len() != self.n {
            panic!("LMSR.cost: dq length {} ≠ N={}", dq.len(), self.n);
        }
        if self.n == 2 {
            return lmsr_cost(self.q[0] + dq[0], self.q[1] + dq[1], self.b) - lmsr_cost(self.q[0], self.q[1], self.b);
        }
        let m0 = self.q.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / self.b;
        let m1 = self.q.iter().zip(dq).map(|(qi, d)| qi + d).fold(f64::NEG_INFINITY, f64::max) / self.b;
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        for i in 0..self.n {
            s0 += (self.q[i] / self.b - m0).exp();
            s1 += ((self.q[i] + dq[i]) / self.b - m1).exp();
        }
        self.b * ((m1 + s1.ln()) - (m0 + s0.ln()))
    }
    pub fn trade(&mut self, dq: &[f64]) -> f64 {
        let c = self.cost(dq);
        for i in 0..self.n {
            self.q[i] += dq[i];
        }
        c
    }
    pub fn buy(&mut self, amount: f64, is_option_one: bool, fee_bps: f64) -> BuyExecution {
        if self.n != 2 {
            panic!("LMSR.buy() requires N = 2");
        }
        let exec = BuyExecutor.transform(BuyExecutionInput {
            amount,
            option_one_shares: self.q[0],
            option_two_shares: self.q[1],
            b: self.b,
            fee_bps: Some(fee_bps),
            is_option_one: Some(is_option_one),
        });
        if is_option_one {
            self.q[0] += exec.shares;
        } else {
            self.q[1] += exec.shares;
        }
        exec
    }
    pub fn sell(&mut self, shares_out: f64, is_option_one: bool, fee_bps: f64) -> SellExecution {
        if self.n != 2 {
            panic!("LMSR.sell() requires N = 2");
        }
        let exec = SellExecutor.transform(SellExecutionInput {
            shares_out,
            option_one_shares: self.q[0],
            option_two_shares: self.q[1],
            b: self.b,
            fee_bps: Some(fee_bps),
            is_option_one: Some(is_option_one),
        });
        if is_option_one {
            self.q[0] -= shares_out;
        } else {
            self.q[1] -= shares_out;
        }
        exec
    }
    pub fn recap(&mut self, new_liquidity: f64, liquidity_is_b: bool) -> f64 {
        if self.n != 2 {
            panic!("LMSR.recap() requires N = 2");
        }
        let new_b = if liquidity_is_b { new_liquidity } else { new_liquidity / (self.n as f64).ln() };
        let r: RecapResult = Recapitalization.transform(RecapitalizationInput {
            option_one_shares: self.q[0],
            option_two_shares: self.q[1],
            current_b: self.b,
            new_b,
        });
        self.q[0] = r.new_option_one_shares;
        self.q[1] = r.new_option_two_shares;
        self.b = r.new_b;
        r.capital_delta
    }
}

/// Convert USDC initial-liquidity `L` to LMSR `b` for an N-outcome market.
pub fn liquidity_to_b(l: f64, n: usize) -> f64 {
    if n == 2 {
        crate::des::general::factmachine_math::b_from_liquidity(l).unwrap()
    } else {
        l / (n as f64).ln()
    }
}

// -----------------------------------------------------------------------------
// Outcome-distribution helpers.
// -----------------------------------------------------------------------------

pub fn outcome_matrix(params: &FactMachineParams) -> Vec<Vec<f64>> {
    let k = params.theta_bins;
    let thetas: Vec<f64> = (0..k).map(|i| i as f64 / (k - 1) as f64).collect();
    if params.market_type == MarketType::Binary {
        return thetas
            .iter()
            .map(|&theta| {
                let p_yes = p_yes_wins(theta, params);
                vec![p_yes, 1.0 - p_yes]
            })
            .collect();
    }
    let n = params.n_voters;
    let mut matrix = Vec::new();
    for &theta in &thetas {
        let pmf = binomial_pmf_internal(n, theta);
        let mut row = vec![0.0; k];
        for kk in 0..=n {
            let x = kk as f64 / n as f64;
            let j = ((x * k as f64).floor() as usize).min(k - 1);
            row[j] += pmf[kk as usize];
        }
        matrix.push(row);
    }
    matrix
}

fn binomial_pmf_internal(n: i64, p: f64) -> Vec<f64> {
    let mut out = vec![0.0; (n + 1) as usize];
    if p <= 0.0 {
        out[0] = 1.0;
        return out;
    }
    if p >= 1.0 {
        out[n as usize] = 1.0;
        return out;
    }
    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    let mut log_coef = 0.0;
    out[0] = (n as f64 * log_q).exp();
    for k in 1..=n {
        log_coef += ((n - k + 1) as f64).ln() - (k as f64).ln();
        out[k as usize] = (log_coef + k as f64 * log_p + (n - k) as f64 * log_q).exp();
    }
    out
}

fn order_prob(theta: f64, informedness: f64) -> f64 {
    theta * informedness + 0.5 * (1.0 - informedness)
}

fn obs_likelihood(theta: f64, yes_orders: f64, total: f64, informedness: f64) -> f64 {
    let p = order_prob(theta, informedness);
    let log_l = yes_orders * (1e-300_f64.max(p)).ln() + (total - yes_orders) * (1e-300_f64.max(1.0 - p)).ln();
    log_l.exp()
}

// -----------------------------------------------------------------------------
// Action selection. Returns -1 for hold, else outcome index to buy.
// -----------------------------------------------------------------------------

fn pick_action(
    params: &FactMachineParams,
    belief: &DiscreteBelief<f64>,
    market: &LMSR,
    rng: &mut SeededRandom,
    tau: f64,
    outcomes: &[Vec<f64>],
) -> i32 {
    let n = market.n;
    let prices = market.prices();
    let fee = params.fee;
    match params.policy {
        Policy::Random => {
            let u = (rng.next_float() * (n + 1) as f64).floor() as i32;
            if (u as usize) < n {
                u
            } else {
                -1
            }
        }
        Policy::Hold => -1,
        Policy::Oracle => {
            let true_idx = (params.true_theta * (params.theta_bins - 1) as f64).round() as usize;
            let row = &outcomes[true_idx];
            let mut best_j = -1;
            let mut best_ev = fee;
            for j in 0..n {
                let ev = row[j] - prices[j];
                if ev > best_ev {
                    best_ev = ev;
                    best_j = j as i32;
                }
            }
            best_j
        }
        Policy::Myopic => {
            let mut best_j = -1;
            let mut best_ev = fee;
            for j in 0..n {
                let mut pj = 0.0;
                for i in 0..belief.weights.len() {
                    pj += belief.weights[i] * outcomes[i][j];
                }
                let ev = pj - prices[j];
                if ev > best_ev {
                    best_ev = ev;
                    best_j = j as i32;
                }
            }
            best_j
        }
        Policy::Qmdp => {
            let mut q = vec![0.0; n + 1];
            for i in 0..belief.weights.len() {
                let w = belief.weights[i];
                let mut best_ev_here = 0.0;
                for j in 0..n {
                    let ev = outcomes[i][j] - prices[j] - fee;
                    if ev > best_ev_here {
                        best_ev_here = ev;
                    }
                }
                let future = best_ev_here * tau;
                for j in 0..n {
                    let ev = outcomes[i][j] - prices[j] - fee;
                    q[j] += w * (ev.max(0.0) + future);
                }
                q[n] += w * future;
            }
            let mut best_j = -1;
            let mut best_q = q[n] + 1e-12;
            for j in 0..n {
                if q[j] > best_q {
                    best_q = q[j];
                    best_j = j as i32;
                }
            }
            best_j
        }
    }
}

// -----------------------------------------------------------------------------
// Stations (plain structs; orchestrated directly — see header PORT NOTE).
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Order {
    side: usize,
    is_yes: bool,
}

struct MarketStation {
    lmsr: LMSR,
    n_outcomes: usize,
    noise_queue: Vec<Order>,
    bettor_queue: Vec<Order>,
    last_noise_yes: i64,
    last_noise_total: i64,
    last_bettor_cost: f64,
    last_bettor_side: i32,
}
impl MarketStation {
    fn new(n_outcomes: usize, liquidity: f64) -> Self {
        MarketStation {
            lmsr: LMSR::new(liquidity, n_outcomes, false),
            n_outcomes,
            noise_queue: Vec::new(),
            bettor_queue: Vec::new(),
            last_noise_yes: 0,
            last_noise_total: 0,
            last_bettor_cost: 0.0,
            last_bettor_side: -1,
        }
    }
    fn enqueue_noise(&mut self, side: usize, is_yes: bool) {
        self.noise_queue.push(Order { side, is_yes });
    }
    fn enqueue_bettor(&mut self, side: usize) {
        self.bettor_queue.push(Order { side, is_yes: false });
    }
    fn settle_noise(&mut self) {
        let mut yes = 0;
        let mut dq = vec![0.0; self.n_outcomes];
        for o in &self.noise_queue {
            if o.is_yes {
                yes += 1;
            }
            dq[o.side] += 1.0;
        }
        if !self.noise_queue.is_empty() {
            self.lmsr.trade(&dq);
        }
        self.last_noise_yes = yes;
        self.last_noise_total = self.noise_queue.len() as i64;
        self.noise_queue.clear();
    }
    fn settle_bettor(&mut self) {
        if self.bettor_queue.is_empty() {
            self.last_bettor_cost = 0.0;
            self.last_bettor_side = -1;
            return;
        }
        let mut dq = vec![0.0; self.n_outcomes];
        let mut side = -1;
        for o in &self.bettor_queue {
            dq[o.side] += 1.0;
            side = o.side as i32;
        }
        self.last_bettor_cost = self.lmsr.trade(&dq);
        self.last_bettor_side = side;
        self.bettor_queue.clear();
    }
}

fn noise_run(params: &FactMachineParams, rng: &mut SeededRandom, market: &mut MarketStation, t: i64) {
    let mut k = params.k_noise;
    let mut q_signal = order_prob(params.true_theta, params.informedness);
    if params.late_flip && t == params.t - 2 {
        k = params.k_noise * params.late_flip_multiplier;
        q_signal = order_prob(1.0 - params.true_theta, params.informedness);
    }
    let total = (sample_poisson(rng, k).max(1.0)) as i64;
    let n_out = market.n_outcomes;
    let half = n_out / 2;
    let mut yes_count = 0i64;
    let mut no_count = 0i64;
    for _ in 0..total {
        let is_yes = rng.next_float() < q_signal;
        let side: usize;
        if is_yes {
            yes_count += 1;
        } else {
            no_count += 1;
        }
        if params.market_type == MarketType::Binary {
            side = if is_yes { 0 } else { 1 };
        } else if is_yes {
            side = half + ((yes_count - 1) as usize) % (n_out - half).max(1);
        } else {
            side = ((no_count - 1) as usize) % half.max(1);
        }
        market.enqueue_noise(side, is_yes);
    }
}

struct BettorStation {
    belief: DiscreteBelief<f64>,
    shares: Vec<f64>,
    cash: f64,
    fees_paid: f64,
    belief_mean: Vec<f64>,
    belief_var: Vec<f64>,
    belief_entropy: Vec<f64>,
}
impl BettorStation {
    fn new(params: &FactMachineParams, n_outcomes: usize) -> Self {
        let states: Vec<f64> = (0..params.theta_bins).map(|i| i as f64 / (params.theta_bins - 1) as f64).collect();
        let belief = DiscreteBelief::new(states, None);
        let belief_mean = vec![belief.mean()];
        let belief_var = vec![belief.variance()];
        let belief_entropy = vec![belief.entropy()];
        BettorStation {
            belief,
            shares: vec![0.0; n_outcomes],
            cash: 0.0,
            fees_paid: 0.0,
            belief_mean,
            belief_var,
            belief_entropy,
        }
    }
    fn run(
        &mut self,
        params: &FactMachineParams,
        rng: &mut SeededRandom,
        market: &mut MarketStation,
        outcomes: &[Vec<f64>],
        t: i64,
    ) {
        let yes = market.last_noise_yes;
        let total = market.last_noise_total;
        if total > 0 {
            let informedness = params.informedness;
            let yes_f = yes as f64;
            let total_f = total as f64;
            self.belief.update(|theta: &f64, _i| obs_likelihood(*theta, yes_f, total_f, informedness));
        }
        let tau = (params.t - t) as f64;
        let action = pick_action(params, &self.belief, &market.lmsr, rng, tau, outcomes);
        if action >= 0 {
            market.enqueue_bettor(action as usize);
        }
    }
    fn apply_settlement(&mut self, params: &FactMachineParams, market: &MarketStation) {
        if market.last_bettor_side >= 0 {
            self.shares[market.last_bettor_side as usize] += 1.0;
            self.cash -= market.last_bettor_cost;
            self.fees_paid += params.fee;
        }
        self.belief_mean.push(self.belief.mean());
        self.belief_var.push(self.belief.variance());
        self.belief_entropy.push(self.belief.entropy());
    }
}

struct ResolutionStation {
    outcome_idx: usize,
    vote_fraction: f64,
    payout: f64,
    fired: bool,
}
impl ResolutionStation {
    fn new() -> Self {
        ResolutionStation { outcome_idx: 0, vote_fraction: 0.0, payout: 0.0, fired: false }
    }
    fn run(
        &mut self,
        params: &FactMachineParams,
        rng: &mut SeededRandom,
        market: &MarketStation,
        bettor: &BettorStation,
        t: i64,
    ) {
        if t < params.t || self.fired {
            return;
        }
        self.fired = true;
        let mut yes_votes = 0i64;
        for _ in 0..params.n_voters {
            if rng.next_float() < params.true_theta {
                yes_votes += 1;
            }
        }
        self.vote_fraction = yes_votes as f64 / params.n_voters as f64;
        if params.market_type == MarketType::Binary {
            if params.resolution_mode == ResolutionMode::Majority {
                self.outcome_idx = if yes_votes as f64 > params.n_voters as f64 / 2.0 { 0 } else { 1 };
            } else {
                self.outcome_idx = if rng.next_float() < params.true_theta { 0 } else { 1 };
            }
        } else {
            self.outcome_idx =
                ((self.vote_fraction * market.n_outcomes as f64).floor() as usize).min(market.n_outcomes - 1);
        }
        self.payout = bettor.shares[self.outcome_idx];
    }
}

// -----------------------------------------------------------------------------
// Result + orchestration.
// -----------------------------------------------------------------------------

pub struct FactMachineResult {
    pub params: FactMachineParams,
    pub final_outcome_idx: usize,
    pub final_outcome: i32,
    pub final_theta: f64,
    pub final_vote_fraction: f64,
    pub shares: Vec<f64>,
    pub shares_yes: f64,
    pub shares_no: f64,
    pub trade_cost: f64,
    pub fees_paid: f64,
    pub payout: f64,
    pub pnl: f64,
    pub belief_mean: Vec<f64>,
    pub belief_var: Vec<f64>,
    pub belief_entropy: Vec<f64>,
    pub belief_snapshots: Vec<Vec<f64>>,
    pub price_history: Vec<Vec<f64>>,
    pub yes_orders_history: Vec<i64>,
    pub total_orders_history: Vec<i64>,
    pub action_history: Vec<i32>,
}

pub fn run_fact_machine(mut params: FactMachineParams) -> FactMachineResult {
    if params.market_type == MarketType::Scalar && params.resolution_mode != ResolutionMode::Majority {
        params.resolution_mode = ResolutionMode::Majority;
    }
    let mut rng = mulberry32(params.seed);
    let n_outcomes = if params.market_type == MarketType::Binary { 2 } else { params.theta_bins };
    let outcomes = outcome_matrix(&params);

    let mut market = MarketStation::new(n_outcomes, params.liquidity);
    let mut bettor = BettorStation::new(&params, n_outcomes);
    let mut resolver = ResolutionStation::new();

    let mut belief_snapshots: Vec<Vec<f64>> = vec![bettor.belief.as_array()];
    let mut price_history: Vec<Vec<f64>> = vec![market.lmsr.prices()];
    let mut yes_orders_history: Vec<i64> = Vec::new();
    let mut total_orders_history: Vec<i64> = Vec::new();
    let mut action_history: Vec<i32> = Vec::new();

    for t in 0..params.t {
        noise_run(&params, &mut rng, &mut market, t); // phase 1
        market.settle_noise(); // phase 2
        let census_prices = market.lmsr.prices(); // phase 3: snapshot for trace
        bettor.run(&params, &mut rng, &mut market, &outcomes, t); // phase 4
        market.settle_bettor(); // phase 5
        bettor.apply_settlement(&params, &market); // phase 6

        belief_snapshots.push(bettor.belief.as_array());
        price_history.push(census_prices);
        yes_orders_history.push(market.last_noise_yes);
        total_orders_history.push(market.last_noise_total);
        action_history.push(market.last_bettor_side);
    }

    resolver.run(&params, &mut rng, &market, &bettor, params.t);

    let shares = bettor.shares.clone();
    let shares_yes = shares[0];
    let shares_no = if n_outcomes >= 2 { shares[1] } else { 0.0 };
    let trade_cost = -bettor.cash;
    let pnl = resolver.payout + bettor.cash - bettor.fees_paid;
    FactMachineResult {
        final_outcome_idx: resolver.outcome_idx,
        final_outcome: if resolver.outcome_idx == 0 { 1 } else { 0 },
        final_theta: params.true_theta,
        final_vote_fraction: resolver.vote_fraction,
        shares,
        shares_yes,
        shares_no,
        trade_cost,
        fees_paid: bettor.fees_paid,
        payout: resolver.payout,
        pnl,
        belief_mean: bettor.belief_mean,
        belief_var: bettor.belief_var,
        belief_entropy: bettor.belief_entropy,
        belief_snapshots,
        price_history,
        yes_orders_history,
        total_orders_history,
        action_history,
        params,
    }
}

// -----------------------------------------------------------------------------
// CLI.
// -----------------------------------------------------------------------------

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn run() {
    let mut base = default_params();
    base.t = env_f64("T", 24.0) as i64;
    base.true_theta = env_f64("TRUE_THETA", 0.65);
    base.informedness = env_f64("INFORMEDNESS", 0.6);
    base.k_noise = env_f64("K_NOISE", 20.0);
    base.fee = env_f64("FEE", 0.01);
    base.liquidity = env_f64("LIQUIDITY", 50.0);
    base.theta_bins = env_f64("THETA_BINS", 21.0) as usize;
    base.seed = env_f64("SEED", 1.0) as u32;
    base.policy = Policy::from_slug(&env_str("POLICY", "qmdp"));
    base.resolution_mode = ResolutionMode::from_slug(&env_str("RESOLUTION", "bernoulli"));
    base.market_type = MarketType::from_slug(&env_str("MARKET", "binary"));
    base.late_flip = env_str("LATE_FLIP", "") == "1";
    base.late_flip_multiplier = env_f64("LATE_FLIP_MUL", 10.0);
    if base.market_type == MarketType::Scalar && base.resolution_mode != ResolutionMode::Majority {
        base.resolution_mode = ResolutionMode::Majority;
    }

    let reps = env_f64("N_REPS", 1.0) as i64;
    if reps == 1 {
        let r = run_fact_machine(base.clone());
        let n_outcomes = r.price_history[0].len();
        println!("# FactMachine POMDP single run");
        println!(
            "#   marketType={} ({} outcomes)  resolution={}  policy={}",
            base.market_type.slug(),
            n_outcomes,
            base.resolution_mode.slug(),
            base.policy.slug()
        );
        println!(
            "#   T={}  trueθ={}  informedness={}  K_noise={}  fee={}  liq={}",
            base.t, base.true_theta, base.informedness, base.k_noise, base.fee, base.liquidity
        );
        println!("#");
        if base.market_type == MarketType::Binary {
            println!("# t      P(YES)      E[θ]       Var[θ]    H(b)     yes/total");
            for t in 0..=base.t as usize {
                let yo = if t > 0 { r.yes_orders_history[t - 1] } else { 0 };
                let to = if t > 0 { r.total_orders_history[t - 1] } else { 0 };
                println!(
                    "# {:>2}  {:.4}  {:.4}  {:.5}  {:.3}  {}/{}",
                    t, r.price_history[t][0], r.belief_mean[t], r.belief_var[t], r.belief_entropy[t], yo, to
                );
            }
        } else {
            println!("# t   E[θ]  H(b)  market mode bin   |   peak market price");
            for t in 0..=base.t as usize {
                let ph = &r.price_history[t];
                let mut best_j = 0;
                for j in 1..ph.len() {
                    if ph[j] > ph[best_j] {
                        best_j = j;
                    }
                }
                let bin_center = (best_j as f64 + 0.5) / ph.len() as f64;
                println!(
                    "# {:>2}  {:.3}  {:.3}    bin {:>2} ≈ {:.2}     {:.3}",
                    t, r.belief_mean[t], r.belief_entropy[t], best_j, bin_center, ph[best_j]
                );
            }
        }
        println!("#");
        if base.market_type == MarketType::Binary {
            println!(
                "# RESOLUTION: vote fraction = {:.3}  outcome={}  shares: yes={}, no={}",
                r.final_vote_fraction,
                if r.final_outcome == 1 { "YES" } else { "NO" },
                fmt_int(r.shares_yes),
                fmt_int(r.shares_no)
            );
        } else {
            let win_bin = r.final_outcome_idx;
            let win_shares = r.shares[win_bin];
            let total_shares: f64 = r.shares.iter().sum();
            println!(
                "# RESOLUTION: vote fraction = {:.3}  → bin {} of {}",
                r.final_vote_fraction, win_bin, n_outcomes
            );
            let shares_str: Vec<String> = r.shares.iter().map(|s| fmt_int(*s)).collect();
            println!(
                "#   shares = [{}]   (winning bin holds {} of {} total)",
                shares_str.join(", "),
                fmt_int(win_shares),
                fmt_int(total_shares)
            );
        }
        println!("#   trade cost  = {:.4}", r.trade_cost);
        println!("#   fees paid   = {:.4}", r.fees_paid);
        println!("#   payout      = {:.4}", r.payout);
        println!("#   PnL         = {:.4}", r.pnl);
        println!(
            "#   final E[θ]  = {:.4}  (true {})",
            r.belief_mean[r.belief_mean.len() - 1],
            base.true_theta
        );

        if env_str("ANIMATE", "") == "1" {
            // PORT NOTE: factmachine animation scene + FrameRecorder not ported.
            render_animation_stub(&r, &base);
            println!("#   animation skipped (scene not ported — see PORT NOTE)");
        }
        return;
    }

    // Multi-rep summary across policies.
    println!(
        "# FactMachine POMDP — N_REPS={} per policy  (market={}, resolution={})",
        reps,
        base.market_type.slug(),
        base.resolution_mode.slug()
    );
    println!(
        "#   trueθ={}  T={}  informedness={}  fee={}\n",
        base.true_theta, base.t, base.informedness, base.fee
    );
    let policies = [Policy::Hold, Policy::Random, Policy::Myopic, Policy::Qmdp, Policy::Oracle];
    println!("# policy     mean PnL    sd PnL    win-rate   final-Brier   total shares   trades");
    for policy in policies {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut wins = 0.0;
        let mut brier = 0.0;
        let mut total_shares = 0.0;
        let mut trades = 0.0;
        for r in 0..reps {
            let mut params = base.clone();
            params.seed = (1000 + r) as u32;
            params.policy = policy;
            let out = run_fact_machine(params);
            sum += out.pnl;
            sum_sq += out.pnl * out.pnl;
            if out.pnl > 0.0 {
                wins += 1.0;
            }
            let last_mean = out.belief_mean[out.belief_mean.len() - 1];
            let y = if out.final_outcome == 1 { BinaryOutcome::One } else { BinaryOutcome::Zero };
            brier += brier_score(last_mean, y);
            let ts: f64 = out.shares.iter().sum();
            total_shares += ts;
            trades += ts;
        }
        let reps_f = reps as f64;
        let mean = sum / reps_f;
        let variance = (sum_sq / reps_f - mean * mean).max(0.0);
        let sd = variance.sqrt();
        println!(
            "# {:<8}  {:>9}  {:>8}  {:>8}    {:>8}      {:>7}    {}",
            policy.slug(),
            format!("{:.4}", mean),
            format!("{:.4}", sd),
            format!("{:.3}", wins / reps_f),
            format!("{:.4}", brier / reps_f),
            format!("{:.2}", total_shares / reps_f),
            format!("{:.2}", trades / reps_f)
        );
    }
}

fn fmt_int(x: f64) -> String {
    // Shares are whole-valued; mirror JS `String(number)` for integers.
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn render_animation_stub(_r: &FactMachineResult, _params: &FactMachineParams) {
    // PORT NOTE: animation/scenes/factmachine-scene and animation/frame-recorder
    // are not ported; the post-hoc HTML render is omitted.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lmsr_uniform_prices_at_init() {
        let m = LMSR::new(50.0, 2, false);
        let p = m.prices();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn outcome_matrix_binary_bernoulli() {
        let mut params = default_params();
        params.theta_bins = 5;
        params.market_type = MarketType::Binary;
        params.resolution_mode = ResolutionMode::Bernoulli;
        let m = outcome_matrix(&params);
        assert_eq!(m.len(), 5);
        for (i, row) in m.iter().enumerate() {
            let theta = i as f64 / 4.0;
            assert!((row[0] - theta).abs() < 1e-9);
            assert!((row[1] - (1.0 - theta)).abs() < 1e-9);
        }
    }

    #[test]
    fn run_is_deterministic() {
        let mut params = default_params();
        params.t = 8;
        params.seed = 42;
        let a = run_fact_machine(params.clone());
        let b = run_fact_machine(params);
        assert_eq!(a.final_outcome_idx, b.final_outcome_idx);
        assert!((a.pnl - b.pnl).abs() < 1e-12);
    }
}
