//! Port of `src/des/main-factmachine-markets.ts`.
//!
//! Platform-level multi-market scheduler above the single-market factmachine
//! model: which opinion markets to open, when, how, and for how long. Four
//! scheduler policies (fixed-daily / greedy-buzz / mdp-oracle / pomdp-belief)
//! are compared across participant-scale scenarios. Defines the model AND runs
//! it (`pub fn run`).
//!
//! Reuses `general::prng` (`mulberry32`/`SeededRandom`), `general::value_iteration`
//! (`MDPSpec`/`value_iteration`/`q_value`), `general::time_stepped_station`
//! (`TimeSteppedStation`), the LMSR algebra in `main_factmachine` (`LMSR`), and
//! `observability::logger` (`JsonValue`) for the JSON/HTML data artifact.
//!
//! PORT NOTES:
//!   * TS `rng: () => number` (a shared `mulberry32` closure) becomes a single
//!     owned `SeededRandom` threaded as `&mut` through each phase.
//!   * `SchedulerAction.kind: MarketKind | 'wait'` -> [`ActionKind`]; the market
//!     kinds proper -> [`MarketKind`]. String unions (category / verification /
//!     information mode / scheduler policy) -> enums with slug helpers.
//!   * `OperatorMDP` keeps a freshly-built [`MDPSpec`]; because `value_iteration`
//!     consumes its spec (boxed closures aren't `Clone`), the spec is rebuilt
//!     for value iteration, Q-value extraction, and storage.
//!   * `__dirname`-relative output path is simplified to a cwd-relative `out/`.
//!   * `Number.toLocaleString()` is reimplemented as thousands grouping
//!     ([`locale_int`]); `seed + 0x9e3779b9` etc. use `u32` wrapping (matching
//!     the TS `>>> 0` truncation inside `mulberry32`).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::prng::mulberry32;
use crate::des::general::time_stepped_station::TimeSteppedStation;
use crate::des::general::value_iteration::{q_value, value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::main_factmachine::LMSR;
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// =============================================================================
// Enums (TS string unions).
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    FixedDaily,
    GreedyBuzz,
    MdpOracle,
    PomdpBelief,
}
impl SchedulerPolicy {
    fn slug(self) -> &'static str {
        match self {
            SchedulerPolicy::FixedDaily => "fixed-daily",
            SchedulerPolicy::GreedyBuzz => "greedy-buzz",
            SchedulerPolicy::MdpOracle => "mdp-oracle",
            SchedulerPolicy::PomdpBelief => "pomdp-belief",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MarketKind {
    Binary,
    Scalar,
    Threshold,
}
impl MarketKind {
    fn slug(self) -> &'static str {
        match self {
            MarketKind::Binary => "binary",
            MarketKind::Scalar => "scalar",
            MarketKind::Threshold => "threshold",
        }
    }
    fn as_kind(self) -> ActionKind {
        match self {
            MarketKind::Binary => ActionKind::Binary,
            MarketKind::Scalar => ActionKind::Scalar,
            MarketKind::Threshold => ActionKind::Threshold,
        }
    }
}

/// `SchedulerAction.kind = MarketKind | 'wait'`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Wait,
    Binary,
    Scalar,
    Threshold,
}
impl ActionKind {
    fn slug(self) -> &'static str {
        match self {
            ActionKind::Wait => "wait",
            ActionKind::Binary => "binary",
            ActionKind::Scalar => "scalar",
            ActionKind::Threshold => "threshold",
        }
    }
    /// `action.kind as MarketKind` (only legal for non-wait actions).
    fn to_market(self) -> MarketKind {
        match self {
            ActionKind::Binary => MarketKind::Binary,
            ActionKind::Scalar => MarketKind::Scalar,
            ActionKind::Threshold => MarketKind::Threshold,
            ActionKind::Wait => panic!("ActionKind::Wait has no MarketKind"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Politics,
    Culture,
    Sports,
    Conspiracy,
    Breaking,
}
impl Category {
    fn slug(self) -> &'static str {
        match self {
            Category::Politics => "politics",
            Category::Culture => "culture",
            Category::Sports => "sports",
            Category::Conspiracy => "conspiracy",
            Category::Breaking => "breaking",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VerificationTier {
    Open,
    Basic,
    Proof,
}
impl VerificationTier {
    fn slug(self) -> &'static str {
        match self {
            VerificationTier::Open => "open",
            VerificationTier::Basic => "basic",
            VerificationTier::Proof => "proof",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InformationMode {
    PriceOnly,
    DelayedVotes,
    LiveVotes,
    DemographicSlices,
    MomentumSignals,
}
impl InformationMode {
    fn slug(self) -> &'static str {
        match self {
            InformationMode::PriceOnly => "price-only",
            InformationMode::DelayedVotes => "delayed-votes",
            InformationMode::LiveVotes => "live-votes",
            InformationMode::DemographicSlices => "demographic-slices",
            InformationMode::MomentumSignals => "momentum-signals",
        }
    }
}

// =============================================================================
// Data structures.
// =============================================================================

#[derive(Clone)]
pub struct CandidateTopic {
    pub id: i64,
    pub category: Category,
    pub created_at: f64,
    pub expires_at: f64,
    pub true_hotness: f64,
    pub ambiguity: f64,
    pub true_theta: f64,
    pub manipulation_risk: f64,
    pub news_cycle_intensity: f64,
    pub social_virality: f64,
    pub influencer_activity: f64,
    pub meme_momentum: f64,
    pub demographic_polarization: f64,
    pub opinion_event_coupling: f64,
    pub event_probability: f64,
    pub turnout_skew: f64,
    pub bot_pressure: f64,
    pub referral_elasticity: f64,
    pub observed_buzz: f64,
    pub observed_ambiguity: f64,
}

#[derive(Clone)]
pub struct SchedulerAction {
    pub label: String,
    pub kind: ActionKind,
    pub duration_h: f64,
    pub fee_rate: f64,
    pub liquidity_multiplier: f64,
    pub reward_multiplier: f64,
    pub verification: VerificationTier,
    pub information_mode: InformationMode,
    pub timing_decay: f64,
    pub threshold: Option<f64>,
    pub description: String,
}

#[derive(Clone)]
pub struct OpenMarket {
    pub id: i64,
    pub topic: CandidateTopic,
    pub action: SchedulerAction,
    pub open_at: f64,
    pub close_at: f64,
}

#[derive(Clone)]
pub struct ClosedMarket {
    pub id: i64,
    pub topic: CandidateTopic,
    pub kind: MarketKind,
    pub contract_label: String,
    pub duration_h: f64,
    pub open_at: f64,
    pub close_at: f64,
    pub fee_rate: f64,
    pub liquidity: f64,
    pub reward_multiplier: f64,
    pub verification: VerificationTier,
    pub information_mode: InformationMode,
    pub timing_decay: f64,
    pub threshold: Option<f64>,
    pub final_vote_fraction: f64,
    pub outcome_index: usize,
    pub votes: i64,
    pub suspected_sybil_votes: i64,
    pub avg_vote_time_fraction: f64,
    pub avg_timing_multiplier: f64,
    pub bettors: i64,
    pub trades: i64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub fee_revenue: f64,
    pub voter_points: f64,
    pub raffle_entries: i64,
    pub avg_prediction_error: f64,
    pub opinion_sampling_error: f64,
    pub prediction_brier_score: f64,
    pub external_outcome: i64,
    pub avg_trader_belief_error: f64,
    pub trader_belief_entropy: f64,
    pub herding_index: f64,
    pub price_opinion_gap: f64,
    pub market_maker_risk_bound: f64,
    pub fraud_pressure: f64,
    pub referral_adds: i64,
    pub churn_risk: f64,
    pub reward_inflation_pressure: f64,
    pub liquidity_utilization: f64,
    pub whale_trade_share: f64,
    pub trader_pnl: f64,
    pub lmsr_loss: f64,
}

#[derive(Clone)]
pub struct PortfolioConfig {
    pub scenario_label: String,
    pub horizon_h: f64,
    pub step_h: f64,
    pub max_concurrent: i64,
    pub min_daily_markets: i64,
    pub max_daily_markets: i64,
    pub daily_market_caps: Vec<i64>,
    pub seed: u32,
    pub liquidity: f64,
    pub fee_rate: f64,
    pub scalar_bins: usize,
    pub min_market_participants: f64,
}

#[derive(Clone)]
pub struct MarketKindAggregate {
    pub kind: MarketKind,
    pub markets: i64,
    pub votes: f64,
    pub bettors: f64,
    pub trades: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub fee_revenue: f64,
    pub voter_points: f64,
    pub suspected_sybil_votes: f64,
    pub avg_duration_h: f64,
    pub avg_liquidity: f64,
    pub avg_fee_rate: f64,
    pub avg_prediction_error: f64,
    pub avg_opinion_sampling_error: f64,
    pub avg_prediction_brier_score: f64,
    pub avg_trader_belief_error: f64,
    pub trader_belief_entropy: f64,
    pub herding_index: f64,
    pub price_opinion_gap: f64,
    pub fraud_pressure: f64,
    pub liquidity_utilization: f64,
    pub whale_trade_share: f64,
    pub platform_surplus: f64,
}

#[derive(Clone)]
pub struct DailySummary {
    pub day: i64,
    pub market_cap: i64,
    pub opened: i64,
    pub closed: i64,
    pub active_end: i64,
    pub queued_end: i64,
    pub votes: f64,
    pub bettors: f64,
    pub trades: f64,
    pub fee_revenue: f64,
    pub voter_points: f64,
    pub binary_closed: i64,
    pub scalar_closed: i64,
    pub threshold_closed: i64,
    pub avg_prediction_error: f64,
    pub avg_opinion_sampling_error: f64,
    pub avg_prediction_brier_score: f64,
    pub fraud_pressure: f64,
    pub herding_index: f64,
}

#[derive(Clone)]
pub struct PolicyAggregate {
    pub scenario_label: String,
    pub min_market_participants: f64,
    pub policy: SchedulerPolicy,
    pub markets_opened: i64,
    pub markets_closed: i64,
    pub binary_markets: i64,
    pub scalar_markets: i64,
    pub threshold_markets: i64,
    pub avg_duration_h: f64,
    pub avg_fee_rate: f64,
    pub avg_liquidity: f64,
    pub avg_reward_multiplier: f64,
    pub proof_markets: i64,
    pub avg_timing_decay: f64,
    pub votes: f64,
    pub suspected_sybil_votes: f64,
    pub avg_vote_time_fraction: f64,
    pub avg_timing_multiplier: f64,
    pub bettors: f64,
    pub trades: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub fee_revenue: f64,
    pub voter_points: f64,
    pub raffle_entries: f64,
    pub avg_prediction_error: f64,
    pub avg_opinion_sampling_error: f64,
    pub avg_prediction_brier_score: f64,
    pub avg_trader_belief_error: f64,
    pub trader_belief_entropy: f64,
    pub herding_index: f64,
    pub price_opinion_gap: f64,
    pub market_maker_risk_bound: f64,
    pub fraud_pressure: f64,
    pub referral_adds: f64,
    pub churn_risk: f64,
    pub reward_inflation_pressure: f64,
    pub liquidity_utilization: f64,
    pub whale_trade_share: f64,
    pub avg_news_cycle_intensity: f64,
    pub avg_social_virality: f64,
    pub avg_influencer_activity: f64,
    pub avg_demographic_polarization: f64,
    pub trader_pnl: f64,
    pub lmsr_loss: f64,
    pub platform_surplus: f64,
    pub engagement_score: f64,
    pub avg_belief_entropy: Option<f64>,
    pub avg_belief_error: Option<f64>,
}

#[derive(Clone)]
pub struct TimelineFrame {
    pub t: f64,
    pub day: i64,
    pub open: i64,
    pub closed: i64,
    pub queued: i64,
    pub votes: i64,
    pub bettors: i64,
    pub trades: i64,
    pub fees: f64,
    pub market_cap: i64,
    pub opened_today: i64,
    pub opened_total: i64,
}

#[derive(Clone)]
pub struct BeliefTraceEntry {
    pub t: f64,
    pub entropy: f64,
    pub expected_hotness: f64,
    pub error: f64,
}

#[derive(Clone)]
pub struct ActionCount {
    pub action: String,
    pub count: i64,
}

#[derive(Clone)]
pub struct PolicyRun {
    pub scenario_label: String,
    pub min_market_participants: f64,
    pub policy: SchedulerPolicy,
    pub aggregate: PolicyAggregate,
    pub kind_breakdown: Vec<MarketKindAggregate>,
    pub daily: Vec<DailySummary>,
    pub closed_markets: Vec<ClosedMarket>,
    pub action_counts: Vec<ActionCount>,
    pub timeline: Vec<TimelineFrame>,
    pub belief_trace: Option<Vec<BeliefTraceEntry>>,
}

/// Solved operator MDP. Keeps a freshly-built [`MDPSpec`] (see header PORT NOTE).
pub struct OperatorMDP {
    pub spec: MDPSpec,
    pub v: Vec<f64>,
    pub policy: Vec<i32>,
    pub q: Vec<Vec<f64>>,
    pub actions: Vec<SchedulerAction>,
    pub iterations: usize,
    pub final_delta: f64,
    pub gamma: f64,
}

// =============================================================================
// Category belief (POMDP scheduler).
// =============================================================================

#[derive(Clone)]
struct BeliefCell {
    hot_bin: i64,
    amb_bin: i64,
    prob: f64,
}

struct CategoryBelief {
    beta: HashMap<Category, (f64, f64)>,
}
impl CategoryBelief {
    fn new() -> Self {
        let mut beta = HashMap::new();
        beta.insert(Category::Politics, (2.0, 1.8));
        beta.insert(Category::Culture, (1.6, 2.0));
        beta.insert(Category::Sports, (1.5, 2.1));
        beta.insert(Category::Conspiracy, (1.8, 2.0));
        beta.insert(Category::Breaking, (2.2, 1.4));
        CategoryBelief { beta }
    }

    fn belief_for(&self, topic: &CandidateTopic) -> Vec<BeliefCell> {
        let (a, b) = self.beta[&topic.category];
        let prior_mean = a / (a + b);
        let hot_scores: Vec<f64> = (0..3)
            .map(|h| {
                let mid = hotness_midpoint(h);
                let obs = -(topic.observed_buzz - mid).powi(2) / 0.045;
                let prior = -(prior_mean - mid).powi(2) / 0.18;
                obs + prior
            })
            .collect();
        let amb_scores: Vec<f64> = (0..3)
            .map(|a| -(topic.observed_ambiguity - ambiguity_midpoint(a)).powi(2) / 0.055)
            .collect();
        let hot_p = softmax(&hot_scores);
        let amb_p = softmax(&amb_scores);
        let mut out = Vec::with_capacity(9);
        for h in 0..3 {
            for a in 0..3 {
                out.push(BeliefCell {
                    hot_bin: h,
                    amb_bin: a,
                    prob: hot_p[h as usize] * amb_p[a as usize],
                });
            }
        }
        out
    }

    fn observe_market(&mut self, market: &ClosedMarket) {
        let success = market.votes >= 180 || market.fee_revenue >= 45.0 || market.trades >= 90;
        let entry = self.beta.get_mut(&market.topic.category).unwrap();
        entry.0 += if success { 1.0 } else { 0.25 };
        entry.1 += if success { 0.25 } else { 1.0 };
    }
}

// =============================================================================
// Portfolio station.
// =============================================================================

struct FactMachinePortfolioStation<'a> {
    id: String,
    config: PortfolioConfig,
    scheduler: SchedulerPolicy,
    mdp: &'a OperatorMDP,
    rng: SeededRandom,
    pending: Vec<CandidateTopic>,
    active: Vec<OpenMarket>,
    closed: Vec<ClosedMarket>,
    action_counts: Vec<ActionCount>,
    timeline: Vec<TimelineFrame>,
    belief_trace: Vec<BeliefTraceEntry>,
    next_topic_id: i64,
    next_market_id: i64,
    fixed_next_open: f64,
    votes_so_far: i64,
    bettors_so_far: i64,
    trades_so_far: i64,
    fees_so_far: f64,
    opened_total: i64,
    opened_by_day: HashMap<i64, i64>,
    category_belief: CategoryBelief,
}

impl<'a> FactMachinePortfolioStation<'a> {
    fn new(
        config: PortfolioConfig,
        scheduler: SchedulerPolicy,
        mdp: &'a OperatorMDP,
        rng: SeededRandom,
    ) -> Self {
        FactMachinePortfolioStation {
            id: format!("factmachine-{}", scheduler.slug()),
            config,
            scheduler,
            mdp,
            rng,
            pending: Vec::new(),
            active: Vec::new(),
            closed: Vec::new(),
            action_counts: Vec::new(),
            timeline: Vec::new(),
            belief_trace: Vec::new(),
            next_topic_id: 0,
            next_market_id: 0,
            fixed_next_open: 0.0,
            votes_so_far: 0,
            bettors_so_far: 0,
            trades_so_far: 0,
            fees_so_far: 0.0,
            opened_total: 0,
            opened_by_day: HashMap::new(),
            category_belief: CategoryBelief::new(),
        }
    }

    fn to_run(&self) -> PolicyRun {
        let mut aggregate = aggregate_run(self.scheduler, &self.closed, &self.config);
        let belief_trace = if self.scheduler == SchedulerPolicy::PomdpBelief {
            Some(self.belief_trace.clone())
        } else {
            None
        };
        if let Some(bt) = &belief_trace {
            if !bt.is_empty() {
                aggregate.avg_belief_entropy =
                    Some(mean(&bt.iter().map(|x| x.entropy).collect::<Vec<_>>()));
                aggregate.avg_belief_error =
                    Some(mean(&bt.iter().map(|x| x.error).collect::<Vec<_>>()));
            }
        }
        let mut action_counts: Vec<ActionCount> = self.action_counts.clone();
        // stable sort by count descending (mirrors V8 Array.sort stability).
        action_counts.sort_by(|x, y| y.count.cmp(&x.count));
        PolicyRun {
            scenario_label: self.config.scenario_label.clone(),
            min_market_participants: self.config.min_market_participants,
            policy: self.scheduler,
            aggregate,
            kind_breakdown: aggregate_by_kind(&self.closed),
            daily: build_daily_summaries(&self.closed, &self.timeline, &self.config),
            closed_markets: self.closed.clone(),
            action_counts,
            timeline: self.timeline.clone(),
            belief_trace,
        }
    }

    fn emit_candidate_topics(&mut self, now: f64) {
        let base_rate_per_hour = 1.35;
        let lambda = base_rate_per_hour * self.config.step_h;
        let n = sample_poisson(lambda, &mut self.rng);
        for _ in 0..n {
            let id = self.next_topic_id;
            self.next_topic_id += 1;
            let topic = sample_topic(id, now, &mut self.rng);
            self.pending.push(topic);
        }
        self.pending.sort_by(|a, b| {
            b.observed_buzz
                .partial_cmp(&a.observed_buzz)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn close_markets(&mut self, now: f64) {
        let mut i = self.active.len();
        while i > 0 {
            i -= 1;
            if self.active[i].close_at > now + 1e-9 {
                continue;
            }
            let market = self.active.remove(i);
            let closed = simulate_market(&market, &self.config, &mut self.rng);
            self.votes_so_far += closed.votes;
            self.bettors_so_far += closed.bettors;
            self.trades_so_far += closed.trades;
            self.fees_so_far += closed.fee_revenue;
            self.category_belief.observe_market(&closed);
            self.closed.push(closed);
        }
    }

    fn expire_candidates(&mut self, now: f64) {
        self.pending.retain(|t| !(t.expires_at < now));
    }

    fn open_markets(&mut self, now: f64) {
        let day = day_index(now);
        let day_cap = daily_market_cap_for_day(day, &self.config);
        loop {
            let opened_today = *self.opened_by_day.get(&day).unwrap_or(&0);
            let can_open = self.active.len() < self.config.max_concurrent as usize
                && opened_today < day_cap
                && !self.pending.is_empty();
            if !can_open {
                break;
            }
            let idx = match self.choose_candidate(now) {
                Some(i) => i,
                None => return,
            };
            let candidate = self.pending[idx].clone();
            let action = self.choose_action(&candidate, now);
            bump(&mut self.action_counts, &action.label);
            if action.kind == ActionKind::Wait {
                return;
            }
            self.pending.remove(idx);
            let id = self.next_market_id;
            self.next_market_id += 1;
            let close_at = now + action.duration_h;
            self.active.push(OpenMarket {
                id,
                topic: candidate,
                action,
                open_at: now,
                close_at,
            });
            *self.opened_by_day.entry(day).or_insert(0) += 1;
            self.opened_total += 1;
            if self.scheduler == SchedulerPolicy::FixedDaily {
                self.fixed_next_open = now + 24.0;
            }
        }
    }

    fn choose_candidate(&self, now: f64) -> Option<usize> {
        if self.scheduler == SchedulerPolicy::FixedDaily && now + 1e-9 < self.fixed_next_open {
            return None;
        }
        if self.scheduler == SchedulerPolicy::FixedDaily && !self.active.is_empty() {
            return None;
        }
        if self.pending.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn choose_action(&mut self, topic: &CandidateTopic, now: f64) -> SchedulerAction {
        let fatigue_bin = fatigue_bin_for(self.active.len() as i64, self.config.max_concurrent);
        if self.scheduler == SchedulerPolicy::FixedDaily {
            return action_by(
                &self.mdp.actions,
                MarketKind::Binary,
                24.0,
                Some("baseline"),
            );
        }
        if self.scheduler == SchedulerPolicy::GreedyBuzz {
            if topic.observed_buzz < 0.38 && !self.active.is_empty() {
                return self.mdp.actions[0].clone();
            }
            if topic.observed_ambiguity > 0.72 {
                let duration_h = if topic.observed_buzz > 0.78 { 1.0 } else { 6.0 };
                return action_by(
                    &self.mdp.actions,
                    MarketKind::Scalar,
                    duration_h,
                    Some(if topic.observed_buzz > 0.78 {
                        "growth"
                    } else {
                        "deep"
                    }),
                );
            }
            if topic.observed_buzz > 0.74 && topic.observed_ambiguity > 0.42 {
                return action_by(
                    &self.mdp.actions,
                    MarketKind::Threshold,
                    1.0,
                    Some("over55"),
                );
            }
            let duration_h = if topic.observed_buzz > 0.76 {
                1.0
            } else if topic.observed_buzz > 0.55 {
                6.0
            } else {
                24.0
            };
            return action_by(
                &self.mdp.actions,
                MarketKind::Binary,
                duration_h,
                Some(if duration_h == 24.0 {
                    "baseline"
                } else {
                    "growth"
                }),
            );
        }
        if self.scheduler == SchedulerPolicy::MdpOracle {
            let s =
                encode_operator_state(bin3(topic.true_hotness), bin3(topic.ambiguity), fatigue_bin);
            return self.mdp.actions[self.mdp.policy[s] as usize].clone();
        }

        // pomdp-belief
        let belief = self.category_belief.belief_for(topic);
        let mut q_by_action = vec![0.0_f64; self.mdp.actions.len()];
        let mut expected_hotness = 0.0;
        let mut error = 0.0;
        for b in &belief {
            let s = encode_operator_state(b.hot_bin, b.amb_bin, fatigue_bin);
            expected_hotness += b.prob * hotness_midpoint(b.hot_bin);
            error += b.prob * (hotness_midpoint(b.hot_bin) - topic.true_hotness).abs();
            for a in 0..self.mdp.actions.len() {
                q_by_action[a] += b.prob * self.mdp.q[s][a];
            }
        }
        let best = arg_max(&q_by_action);
        let probs: Vec<f64> = belief.iter().map(|b| b.prob).collect();
        self.belief_trace.push(BeliefTraceEntry {
            t: now,
            entropy: entropy(&probs),
            expected_hotness,
            error,
        });
        self.mdp.actions[best].clone()
    }
}

impl<'a> TimeSteppedStation for FactMachinePortfolioStation<'a> {
    fn id(&self) -> &str {
        &self.id
    }

    fn run_time_step(&mut self, _step_size: f64, tick: f64) {
        let now = tick * self.config.step_h;
        let accepting_new_markets = now < self.config.horizon_h - 1e-9;
        if accepting_new_markets {
            self.emit_candidate_topics(now);
        }
        self.close_markets(now);
        self.expire_candidates(now);
        if accepting_new_markets {
            self.open_markets(now);
        }
        let day = day_index(now);
        self.timeline.push(TimelineFrame {
            t: now,
            day,
            open: self.active.len() as i64,
            closed: self.closed.len() as i64,
            queued: self.pending.len() as i64,
            votes: self.votes_so_far,
            bettors: self.bettors_so_far,
            trades: self.trades_so_far,
            fees: self.fees_so_far,
            market_cap: daily_market_cap_for_day(day, &self.config),
            opened_today: *self.opened_by_day.get(&day).unwrap_or(&0),
            opened_total: self.opened_total,
        });
    }
}

// =============================================================================
// Operator action recipes.
// =============================================================================

fn wait_action() -> SchedulerAction {
    SchedulerAction {
        label: "wait".to_string(),
        kind: ActionKind::Wait,
        duration_h: 0.0,
        fee_rate: 0.0,
        liquidity_multiplier: 1.0,
        reward_multiplier: 1.0,
        verification: VerificationTier::Basic,
        information_mode: InformationMode::PriceOnly,
        timing_decay: 1.0,
        threshold: None,
        description: "do not open a new market this step".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn market_action(
    label: &str,
    kind: MarketKind,
    duration_h: f64,
    fee_rate: f64,
    liquidity_multiplier: f64,
    reward_multiplier: f64,
    verification: VerificationTier,
    information_mode: InformationMode,
    timing_decay: f64,
    threshold: Option<f64>,
    description: &str,
) -> SchedulerAction {
    SchedulerAction {
        label: label.to_string(),
        kind: kind.as_kind(),
        duration_h,
        fee_rate,
        liquidity_multiplier,
        reward_multiplier,
        verification,
        information_mode,
        timing_decay,
        threshold,
        description: description.to_string(),
    }
}

fn operator_actions() -> Vec<SchedulerAction> {
    use InformationMode::*;
    use MarketKind::*;
    use VerificationTier::*;
    vec![
        wait_action(),
        market_action(
            "binary-baseline-24h",
            Binary,
            24.0,
            0.01,
            1.0,
            1.0,
            Basic,
            DelayedVotes,
            1.0,
            None,
            "baseline majority market",
        ),
        market_action(
            "binary-growth-15m",
            Binary,
            0.25,
            0.005,
            1.35,
            1.35,
            Open,
            MomentumSignals,
            1.45,
            None,
            "fast low-fee launch for hot topics",
        ),
        market_action(
            "binary-growth-1h",
            Binary,
            1.0,
            0.005,
            1.25,
            1.25,
            Open,
            LiveVotes,
            1.35,
            None,
            "one-hour majority market optimized for participation",
        ),
        market_action(
            "binary-surplus-6h",
            Binary,
            6.0,
            0.02,
            0.75,
            0.85,
            Basic,
            PriceOnly,
            0.85,
            None,
            "higher-fee majority market optimized for margin",
        ),
        market_action(
            "binary-proof-24h",
            Binary,
            24.0,
            0.01,
            1.15,
            1.05,
            Proof,
            DemographicSlices,
            1.05,
            None,
            "longer majority market with proof-of-personhood trust",
        ),
        market_action(
            "scalar-growth-1h",
            Scalar,
            1.0,
            0.005,
            1.35,
            1.25,
            Open,
            LiveVotes,
            1.25,
            None,
            "distribution market for ambiguous fast-moving topics",
        ),
        market_action(
            "scalar-deep-6h",
            Scalar,
            6.0,
            0.01,
            1.55,
            1.1,
            Basic,
            DemographicSlices,
            1.0,
            None,
            "deeper-liquidity distribution market",
        ),
        market_action(
            "scalar-proof-24h",
            Scalar,
            24.0,
            0.01,
            1.4,
            1.0,
            Proof,
            DemographicSlices,
            0.95,
            None,
            "long-form scalar sentiment read with verified voting",
        ),
        market_action(
            "over55-growth-1h",
            Threshold,
            1.0,
            0.005,
            1.25,
            1.2,
            Open,
            MomentumSignals,
            1.35,
            Some(0.55),
            "over/under 55% agree, optimized for rapid debate",
        ),
        market_action(
            "over60-surplus-6h",
            Threshold,
            6.0,
            0.02,
            0.9,
            0.9,
            Basic,
            PriceOnly,
            0.85,
            Some(0.60),
            "over/under 60% agree, optimized for fee capture",
        ),
        market_action(
            "over55-proof-24h",
            Threshold,
            24.0,
            0.01,
            1.25,
            1.05,
            Proof,
            DelayedVotes,
            1.1,
            Some(0.55),
            "verified over/under sentiment threshold",
        ),
    ]
}

/// Build a fresh [`MDPSpec`] for the operator MDP (rebuilt per use; see header).
fn build_operator_spec() -> MDPSpec {
    let actions = operator_actions();
    let n = actions.len();
    let outcomes_actions = actions.clone();
    let label_actions = actions;
    MDPSpec {
        num_states: 27,
        num_actions: Box::new(move |_s| n),
        outcomes: Box::new(move |s, a| operator_outcomes(s, &outcomes_actions[a])),
        is_terminal: None,
        terminal_reward: None,
        state_label: Some(Box::new(operator_state_label)),
        action_label: Some(Box::new(move |a| label_actions[a].label.clone())),
    }
}

pub fn build_operator_mdp() -> OperatorMDP {
    let gamma = 0.88;
    let actions = operator_actions();
    let n_actions = actions.len();
    let vi = value_iteration(
        build_operator_spec(),
        VIOptions {
            gamma,
            tol: 1e-8,
            max_iter: 10000,
            random_tie_break: false,
            ..Default::default()
        },
    );
    let q_spec = build_operator_spec();
    let num_states = q_spec.num_states;
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(num_states);
    for s in 0..num_states {
        let mut row = Vec::with_capacity(n_actions);
        for a in 0..n_actions {
            row.push(q_value(&q_spec, &vi.v, s, a, gamma));
        }
        q.push(row);
    }
    OperatorMDP {
        spec: build_operator_spec(),
        v: vi.v,
        policy: vi.policy,
        q,
        actions,
        iterations: vi.iterations,
        final_delta: vi.final_delta,
        gamma,
    }
}

fn operator_outcomes(s: usize, action: &SchedulerAction) -> Vec<Outcome> {
    let (hot_bin, amb_bin, fatigue_bin) = decode_operator_state(s);
    if action.kind == ActionKind::Wait {
        let next_fatigue = (fatigue_bin - 1).max(0);
        return vec![Outcome {
            prob: 1.0,
            reward: -1.5 + hot_bin as f64 * 0.2,
            next_state: encode_operator_state(hot_bin, amb_bin, next_fatigue),
        }];
    }
    let reward = expected_market_utility(hot_bin, amb_bin, fatigue_bin, action);
    let fatigue_up = (fatigue_bin + if action.duration_h >= 6.0 { 1 } else { 0 }).min(2);
    let fatigue_same = fatigue_bin;
    let fatigue_down = (fatigue_bin - 1).max(0);
    vec![
        Outcome {
            prob: 0.58,
            reward,
            next_state: encode_operator_state(hot_bin, amb_bin, fatigue_up),
        },
        Outcome {
            prob: 0.28,
            reward: reward * 0.88,
            next_state: encode_operator_state((hot_bin - 1).max(0), amb_bin, fatigue_same),
        },
        Outcome {
            prob: 0.14,
            reward: reward * 1.12,
            next_state: encode_operator_state((hot_bin + 1).min(2), amb_bin, fatigue_down),
        },
    ]
}

fn expected_market_utility(
    hot_bin: i64,
    amb_bin: i64,
    fatigue_bin: i64,
    action: &SchedulerAction,
) -> f64 {
    let hot = hotness_midpoint(hot_bin);
    let amb = ambiguity_midpoint(amb_bin);
    let reach = 1.0 - (-action.duration_h / 5.5).exp();
    let urgency = if action.duration_h <= 1.0 {
        1.22
    } else if action.duration_h <= 6.0 {
        1.06
    } else {
        0.92
    };
    let fatigue_penalty = 1.0 - fatigue_bin as f64 * 0.16;
    let kind_fit = match action.kind {
        ActionKind::Scalar => 0.70 + amb * 0.95,
        ActionKind::Threshold => 0.82 + amb * 0.18 + hot * 0.12,
        _ => 1.34 - amb * 0.42 + hot * 0.08,
    };
    let fee_drag = clampf(1.14 - action.fee_rate * 18.0, 0.66, 1.08);
    let liquidity_boost = 0.86 + action.liquidity_multiplier * 0.14;
    let reward_boost = 0.82 + action.reward_multiplier * 0.25;
    let verification_participation = verification_participation_multiplier(action.verification);
    let verification_trust = verification_trust_multiplier(action.verification);
    let information_engagement = information_engagement_multiplier(action.information_mode);
    let information_trust = information_trust_multiplier(action.information_mode);
    let timing_urgency = clampf(0.86 + action.timing_decay * 0.18, 0.78, 1.24);
    let votes = 45.0
        * hot
        * (0.45 + 2.4 * reach)
        * urgency
        * fatigue_penalty
        * kind_fit
        * reward_boost
        * verification_participation
        * information_engagement
        * timing_urgency;
    let traders = votes
        * (0.18 + 0.24 * hot + 0.08 * amb)
        * market_trader_fit(action.kind.to_market())
        * fee_drag
        * liquidity_boost;
    let avg_trade_size = (7.0 + 9.0 * hot) * (0.86 + action.liquidity_multiplier * 0.18);
    let fees = traders * avg_trade_size * action.fee_rate;
    let integrity_penalty = (hot * amb).powf(1.2)
        * (if action.duration_h >= 24.0 { 10.0 } else { 4.0 })
        * manipulation_multiplier(action.verification);
    let reward_cost = votes * (0.06 + action.reward_multiplier * 0.08);
    let threshold_penalty = if action.kind == ActionKind::Threshold {
        3.5 + (action.threshold.unwrap_or(0.55) - 0.55).max(0.0) * (1.0 - amb) * 8.0
    } else {
        0.0
    };
    let scalar_resolution_bonus = if action.kind == ActionKind::Scalar {
        8.5 * amb * (0.45 + hot)
    } else {
        0.0
    };
    let binary_clarity_bonus = if action.kind == ActionKind::Binary {
        7.0 * (1.0 - amb) * (0.55 + hot)
    } else {
        0.0
    };
    let threshold_headline_bonus = if action.kind == ActionKind::Threshold {
        4.2 * hot * (0.55 + amb)
    } else {
        0.0
    };
    let cascade_penalty =
        (hot * amb).powf(1.1) * information_herding_multiplier(action.information_mode) * 6.0;
    votes * 0.28
        + traders * 0.55
        + fees * 2.4
        + verification_trust * 3.0
        + information_trust * 2.0
        + scalar_resolution_bonus
        + binary_clarity_bonus
        + threshold_headline_bonus
        - reward_cost
        - integrity_penalty
        - threshold_penalty
        - cascade_penalty
        - fatigue_bin as f64 * 4.0
}

// =============================================================================
// Single-market simulation.
// =============================================================================

fn simulate_market(
    market: &OpenMarket,
    cfg: &PortfolioConfig,
    rng: &mut SeededRandom,
) -> ClosedMarket {
    let topic = &market.topic;
    let action = &market.action;
    let kind = action.kind.to_market();
    let n: usize = if kind == MarketKind::Scalar {
        cfg.scalar_bins
    } else {
        2
    };
    let population_scale = (cfg.min_market_participants / 1000.0).max(1.0);
    let liquidity = cfg.liquidity * action.liquidity_multiplier * population_scale.sqrt();
    let mut lmsr = LMSR::new(liquidity, n, false);
    let duration = action.duration_h;
    let reach = 1.0 - (-duration / 6.0).exp();
    let urgency = if duration <= 1.0 {
        1.22
    } else if duration <= 6.0 {
        1.05
    } else {
        0.9
    };
    let kind_vote_fit = match kind {
        MarketKind::Scalar => 0.92 + topic.ambiguity * 0.22,
        MarketKind::Threshold => {
            0.96 + threshold_drama(topic.true_theta, action.threshold.unwrap_or(0.55)) * 0.14
        }
        MarketKind::Binary => 1.06 - topic.ambiguity * 0.10,
    };
    let reward_boost = 0.82 + action.reward_multiplier * 0.25;
    let verification_participation = verification_participation_multiplier(action.verification);
    let information_engagement = information_engagement_multiplier(action.information_mode);
    let timing_urgency = clampf(0.86 + action.timing_decay * 0.18, 0.78, 1.24);
    let expected_votes = (55.0
        * topic.true_hotness
        * (0.35 + 2.5 * reach)
        * urgency
        * kind_vote_fit
        * reward_boost
        * verification_participation
        * information_engagement
        * timing_urgency
        * (0.82
            + topic.news_cycle_intensity * 0.22
            + topic.social_virality * 0.18
            + topic.referral_elasticity * 0.10))
        .max(6.0);
    let votes =
        (sample_poisson(expected_votes, rng) as f64).max(cfg.min_market_participants) as i64;
    let manip_mult = manipulation_multiplier(action.verification);
    let majority_direction = if topic.true_theta >= 0.5 { 1.0 } else { -1.0 };
    let manipulation_push = topic.manipulation_risk * manip_mult * majority_direction * 0.085;
    let turnout_push =
        topic.turnout_skew * topic.demographic_polarization * majority_direction * 0.045;
    let influencer_push =
        topic.influencer_activity * topic.meme_momentum * majority_direction * 0.025;
    let effective_theta = clampf(
        topic.true_theta + manipulation_push + turnout_push + influencer_push,
        0.03,
        0.97,
    );
    let suspected_sybil_votes = votes.min(sample_poisson(
        votes as f64
            * (topic.manipulation_risk * 0.12
                + topic.bot_pressure * 0.10
                + topic.influencer_activity * 0.03)
            * manip_mult
            * (1.0 + population_scale.log10() * 0.06),
        rng,
    ));
    let external_outcome: i64 = if rng.next_float() < topic.event_probability {
        1
    } else {
        0
    };
    let opinion_fact_gap = (topic.true_theta - topic.event_probability).abs();
    let resolution_confusion_rate = clampf(
        0.04 + opinion_fact_gap * 0.18
            + topic.ambiguity * 0.07
            + (if action.information_mode == InformationMode::PriceOnly {
                0.06
            } else {
                0.0
            })
            + (if action.information_mode == InformationMode::MomentumSignals {
                0.05
            } else {
                0.0
            })
            - (if action.information_mode == InformationMode::DemographicSlices {
                0.04
            } else {
                0.0
            }),
        0.02,
        0.34,
    );

    let mut yes_votes: i64 = 0;
    let mut bettor_count: i64 = 0;
    let mut trades: i64 = 0;
    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;
    let mut fee_revenue = 0.0;
    let mut voter_points = 0.0;
    let mut raffle_entries: i64 = 0;
    let mut prediction_error = 0.0;
    let mut vote_time_fraction = 0.0;
    let mut timing_multiplier_sum = 0.0;
    let mut trader_belief_error = 0.0;
    let mut trader_belief_entropy = 0.0;
    let mut herding_mass = 0.0;
    let mut whale_volume = 0.0;
    let mut trader_cash = 0.0;
    let mut trader_shares = vec![0.0_f64; n];
    let market_public_signal = clampf(
        effective_theta
            + normal(rng) * information_observation_noise(action.information_mode, topic.ambiguity),
        0.01,
        0.99,
    );

    for _ in 0..votes {
        let timing_exponent = clampf(
            1.05 + action.timing_decay * 0.62
                - information_wait_pressure(action.information_mode) * 0.32
                + action.reward_multiplier * 0.08,
            0.75,
            2.55,
        );
        let vote_time = duration * rng.next_float().powf(timing_exponent);
        vote_time_fraction += if duration > 0.0 {
            vote_time / duration
        } else {
            0.0
        };
        let vote_yes = rng.next_float() < effective_theta;
        if vote_yes {
            yes_votes += 1;
        }
        let voter_private_signal = clampf(
            effective_theta + normal(rng) * (0.11 + 0.09 * topic.ambiguity),
            0.01,
            0.99,
        );
        let voter_public_signal = clampf(
            market_public_signal
                + normal(rng)
                    * information_observation_noise(action.information_mode, topic.ambiguity)
                    * 0.55,
            0.01,
            0.99,
        );
        let voter_info_weight =
            information_signal_weight(action.information_mode) * (vote_time / duration.max(1e-9));
        let predicted_theta = clampf(
            voter_private_signal * (1.0 - voter_info_weight)
                + voter_public_signal * voter_info_weight,
            0.01,
            0.99,
        );
        let predicted_agree = if vote_yes {
            predicted_theta
        } else {
            1.0 - predicted_theta
        };
        let actual_agree_placeholder = if vote_yes {
            effective_theta
        } else {
            1.0 - effective_theta
        };
        let err = (predicted_agree - actual_agree_placeholder).abs();
        prediction_error += err;
        let timing_boost =
            1.0 + (-action.timing_decay * vote_time / (duration * 0.32).max(0.35)).exp();
        timing_multiplier_sum += timing_boost;
        let accuracy_points = if err <= 0.05 {
            18.0
        } else {
            (12.0 * (1.0 - err / 0.25)).max(0.0)
        };
        voter_points += accuracy_points * timing_boost * action.reward_multiplier;
        if err <= 0.20 {
            raffle_entries += 1;
        }

        let fee_drag = clampf(1.14 - action.fee_rate * 18.0, 0.66, 1.08);
        let liquidity_boost = 0.86 + action.liquidity_multiplier * 0.14;
        let social_trading_boost = 1.0
            + topic.social_virality * 0.16
            + topic.influencer_activity * 0.10
            + topic.meme_momentum * 0.10;
        let trade_prob = clampf(
            (0.14
                + topic.true_hotness * 0.28
                + topic.ambiguity * 0.08
                + (match kind {
                    MarketKind::Binary => 0.04,
                    MarketKind::Threshold => 0.02,
                    MarketKind::Scalar => -0.02,
                }))
                * fee_drag
                * liquidity_boost
                * social_trading_boost,
            0.04,
            0.68,
        );
        if rng.next_float() >= trade_prob {
            continue;
        }
        bettor_count += 1;
        let trades_as_prediction_market = rng.next_float() < resolution_confusion_rate;
        let trader_target = if trades_as_prediction_market {
            topic.event_probability
        } else {
            effective_theta
        };
        let private_signal = clampf(
            trader_target
                + normal(rng)
                    * (0.16 - 0.05 * topic.true_hotness
                        + (if trades_as_prediction_market {
                            0.04
                        } else {
                            0.0
                        })),
            0.01,
            0.99,
        );
        let public_signal = clampf(
            (if trades_as_prediction_market {
                topic.event_probability
            } else {
                market_public_signal
            }) + normal(rng)
                * information_observation_noise(action.information_mode, topic.ambiguity)
                * (if trades_as_prediction_market {
                    0.72
                } else {
                    0.45
                }),
            0.01,
            0.99,
        );
        let trader_info_weight = clampf(
            information_signal_weight(action.information_mode)
                + topic.influencer_activity * 0.08
                + topic.meme_momentum * 0.07,
            0.0,
            0.86,
        );
        let signal = clampf(
            private_signal * (1.0 - trader_info_weight) + public_signal * trader_info_weight,
            0.01,
            0.99,
        );
        trader_belief_error += (signal - effective_theta).abs();
        trader_belief_entropy += bernoulli_entropy(signal);
        herding_mass += (signal - private_signal).abs();
        let outcome: usize = match kind {
            MarketKind::Binary => {
                if signal >= 0.5 {
                    0
                } else {
                    1
                }
            }
            MarketKind::Threshold => {
                if signal >= action.threshold.unwrap_or(0.55) {
                    0
                } else {
                    1
                }
            }
            MarketKind::Scalar => {
                clampi((signal * n as f64).floor() as i64, 0, n as i64 - 1) as usize
            }
        };
        let whale_prob = clampf(
            0.012
                + topic.influencer_activity * 0.024
                + topic.social_virality * 0.014
                + population_scale.log10() * 0.006,
            0.008,
            0.08,
        );
        let is_whale = rng.next_float() < whale_prob;
        let whale_multiplier = if is_whale {
            6.0 + 18.0 * rng.next_float()
        } else {
            1.0
        };
        let base_budget = clampf(
            (4.0 + exp_sample(1.0 / (7.0 + 10.0 * topic.true_hotness), rng))
                * (0.92 + action.liquidity_multiplier * 0.10)
                * clampf(1.08 - action.fee_rate * 10.0, 0.78, 1.03),
            3.0,
            72.0,
        );
        let budget = clampf(
            base_budget * whale_multiplier,
            3.0,
            if is_whale { 900.0 } else { 72.0 },
        );
        let (buy_shares, _buy_cost, buy_fee) =
            buy_budget(&mut lmsr, outcome, budget, action.fee_rate);
        trades += 1;
        buy_volume += budget;
        fee_revenue += buy_fee;
        trader_cash -= _buy_cost + buy_fee;
        trader_shares[outcome] += buy_shares;
        if is_whale {
            whale_volume += budget;
        }

        if rng.next_float() < 0.26 + 0.10 * topic.manipulation_risk {
            let shares_out = trader_shares[outcome] * (0.25 + 0.35 * rng.next_float());
            if shares_out > 1e-9 {
                let (sell_gross, sell_fee) =
                    sell_shares(&mut lmsr, outcome, shares_out, action.fee_rate);
                trades += 1;
                sell_volume += sell_gross;
                fee_revenue += sell_fee;
                trader_cash += sell_gross - sell_fee;
                trader_shares[outcome] -= shares_out;
            }
        }
    }

    let final_vote_fraction = yes_votes as f64 / votes as f64;
    let outcome_index: usize = match kind {
        MarketKind::Binary => {
            if final_vote_fraction >= 0.5 {
                0
            } else {
                1
            }
        }
        MarketKind::Threshold => {
            if final_vote_fraction >= action.threshold.unwrap_or(0.55) {
                0
            } else {
                1
            }
        }
        MarketKind::Scalar => clampi(
            (final_vote_fraction * n as f64).floor() as i64,
            0,
            n as i64 - 1,
        ) as usize,
    };
    let payout = trader_shares[outcome_index];
    trader_cash += payout;
    let trader_pnl = trader_cash;
    let lmsr_loss = (payout - buy_volume + sell_volume).max(0.0);
    let final_prediction_error = if votes > 0 {
        prediction_error / votes as f64
    } else {
        0.0
    };
    let final_prices = lmsr.prices();
    let market_implied_vote_fraction = if kind == MarketKind::Scalar {
        final_prices
            .iter()
            .enumerate()
            .map(|(i, p)| *p * scalar_bin_midpoint(i, n))
            .sum::<f64>()
    } else {
        final_prices[0]
    };
    let observed_event = if kind == MarketKind::Scalar {
        final_vote_fraction
    } else if outcome_index == 0 {
        1.0
    } else {
        0.0
    };
    let price_opinion_gap = if kind == MarketKind::Scalar {
        (market_implied_vote_fraction - final_vote_fraction).abs()
    } else {
        (final_prices[0] - observed_event).abs()
    };
    let opinion_sampling_error = (final_vote_fraction - topic.true_theta).abs();
    let prediction_brier_score = (market_implied_vote_fraction - external_outcome as f64).powi(2);
    let referral_adds = (votes as f64
        * topic.referral_elasticity
        * (0.012 + topic.social_virality * 0.034 + topic.meme_momentum * 0.024)
        * information_engagement_multiplier(action.information_mode))
    .round() as i64;
    let fraud_pressure = if votes > 0 {
        suspected_sybil_votes as f64 / votes as f64
    } else {
        0.0
    };
    let reward_inflation_pressure = clampf(
        (voter_points / (votes as f64).max(1.0)) / 24.0
            + action.reward_multiplier * 0.10
            + population_scale.log10() * 0.025,
        0.0,
        1.0,
    );
    let liquidity_utilization = buy_volume / liquidity.max(1.0);
    let whale_trade_share = if buy_volume > 0.0 {
        whale_volume / buy_volume
    } else {
        0.0
    };
    let churn_risk = clampf(
        0.04 + reward_inflation_pressure * 0.20
            + fraud_pressure * 0.26
            + (price_opinion_gap - 0.25).max(0.0) * 0.20
            + topic.demographic_polarization * 0.08
            - topic.referral_elasticity * 0.05,
        0.0,
        1.0,
    );

    ClosedMarket {
        id: market.id,
        topic: topic.clone(),
        kind,
        contract_label: contract_label(action),
        duration_h: duration,
        open_at: market.open_at,
        close_at: market.close_at,
        fee_rate: action.fee_rate,
        liquidity,
        reward_multiplier: action.reward_multiplier,
        verification: action.verification,
        information_mode: action.information_mode,
        timing_decay: action.timing_decay,
        threshold: action.threshold,
        final_vote_fraction,
        outcome_index,
        votes,
        suspected_sybil_votes,
        avg_vote_time_fraction: if votes > 0 {
            vote_time_fraction / votes as f64
        } else {
            0.0
        },
        avg_timing_multiplier: if votes > 0 {
            timing_multiplier_sum / votes as f64
        } else {
            0.0
        },
        bettors: bettor_count,
        trades,
        buy_volume,
        sell_volume,
        fee_revenue,
        voter_points,
        raffle_entries,
        avg_prediction_error: final_prediction_error,
        opinion_sampling_error,
        prediction_brier_score,
        external_outcome,
        avg_trader_belief_error: if bettor_count > 0 {
            trader_belief_error / bettor_count as f64
        } else {
            0.0
        },
        trader_belief_entropy: if bettor_count > 0 {
            trader_belief_entropy / bettor_count as f64
        } else {
            0.0
        },
        herding_index: if bettor_count > 0 {
            clampf(herding_mass / (bettor_count as f64 * 0.5), 0.0, 1.0)
        } else {
            0.0
        },
        price_opinion_gap,
        market_maker_risk_bound: lmsr.b * (n as f64).ln(),
        fraud_pressure,
        referral_adds,
        churn_risk,
        reward_inflation_pressure,
        liquidity_utilization,
        whale_trade_share,
        trader_pnl,
        lmsr_loss,
    }
}

fn buy_budget(
    lmsr: &mut LMSR,
    outcome: usize,
    gross_budget: f64,
    fee_rate: f64,
) -> (f64, f64, f64) {
    let fee = gross_budget * fee_rate;
    let budget = gross_budget - fee;
    let prices = lmsr.prices();
    let p = prices[outcome].max(1e-9);
    let shares = lmsr.b * ((budget / lmsr.b).exp_m1() / p).ln_1p();
    let mut dq = vec![0.0_f64; lmsr.n];
    dq[outcome] = shares;
    let cost = lmsr.trade(&dq);
    (shares, cost, fee)
}

fn sell_shares(lmsr: &mut LMSR, outcome: usize, shares: f64, fee_rate: f64) -> (f64, f64) {
    let mut dq = vec![0.0_f64; lmsr.n];
    dq[outcome] = -shares;
    let gross = -lmsr.cost(&dq);
    lmsr.trade(&dq);
    (gross, gross * fee_rate)
}

// =============================================================================
// Topic sampling.
// =============================================================================

fn sample_topic(id: i64, now: f64, rng: &mut SeededRandom) -> CandidateTopic {
    let categories = [
        Category::Politics,
        Category::Culture,
        Category::Sports,
        Category::Conspiracy,
        Category::Breaking,
    ];
    let weights = [0.26, 0.22, 0.16, 0.16, 0.20];
    let category = categories[categorical(&weights, rng)];
    let category_hot_boost = match category {
        Category::Politics => 0.08,
        Category::Culture => -0.02,
        Category::Sports => -0.04,
        Category::Conspiracy => 0.02,
        Category::Breaking => 0.14,
    };
    let true_hotness = clampf(beta_like(rng, 2.0, 2.2) + category_hot_boost, 0.05, 0.98);
    let ambiguity = clampf(beta_like(rng, 2.1, 2.1), 0.05, 0.95);
    let lean = (if rng.next_float() < 0.5 { -1.0 } else { 1.0 })
        * (0.08 + 0.35 * (1.0 - ambiguity) * rng.next_float());
    let true_theta = clampf(0.5 + lean, 0.03, 0.97);
    let news_cycle_intensity = clampf(
        beta_like(
            rng,
            if category == Category::Breaking {
                3.2
            } else {
                2.1
            },
            2.0,
        ) + (if category == Category::Breaking {
            0.10
        } else {
            0.0
        }),
        0.02,
        1.0,
    );
    let social_virality = clampf(
        0.22 + 0.48 * true_hotness + 0.22 * news_cycle_intensity + normal(rng) * 0.13,
        0.0,
        1.0,
    );
    let influencer_activity = clampf(
        beta_like(rng, 1.8, 2.4)
            + (if category == Category::Politics || category == Category::Culture {
                0.08
            } else {
                0.0
            }),
        0.0,
        1.0,
    );
    let meme_momentum = clampf(
        0.18 + 0.44 * social_virality + 0.20 * ambiguity + normal(rng) * 0.12,
        0.0,
        1.0,
    );
    let demographic_polarization = clampf(
        0.12 + 0.55 * (1.0 - ambiguity)
            + (if category == Category::Politics || category == Category::Conspiracy {
                0.14
            } else {
                0.0
            })
            + normal(rng) * 0.10,
        0.0,
        1.0,
    );
    let coupling_prior = match category {
        Category::Politics => 0.48,
        Category::Culture => 0.32,
        Category::Sports => 0.78,
        Category::Conspiracy => 0.26,
        Category::Breaking => 0.66,
    };
    let opinion_event_coupling = clampf(
        coupling_prior + news_cycle_intensity * 0.12
            - ambiguity * 0.12
            - demographic_polarization * 0.16,
        0.12,
        0.92,
    );
    let event_probability = clampf(
        0.5 + (true_theta - 0.5) * opinion_event_coupling
            + normal(rng)
                * (0.12
                    + ambiguity * 0.10
                    + (if category == Category::Conspiracy {
                        0.08
                    } else {
                        0.0
                    })),
        0.03,
        0.97,
    );
    let turnout_skew = clampf(
        normal(rng).abs() * 0.18 + demographic_polarization * 0.24 + social_virality * 0.10,
        0.0,
        0.75,
    );
    let bot_pressure = clampf(
        0.03 + 0.20 * social_virality
            + 0.22 * influencer_activity
            + (if category == Category::Politics || category == Category::Conspiracy {
                0.08
            } else {
                0.0
            })
            + normal(rng) * 0.06,
        0.0,
        0.75,
    );
    let referral_elasticity = clampf(
        0.10 + 0.46 * meme_momentum + 0.22 * social_virality + normal(rng) * 0.10,
        0.0,
        1.0,
    );
    let manipulation_risk = clampf(
        0.06 + 0.18 * true_hotness * ambiguity
            + 0.16 * bot_pressure
            + 0.08 * demographic_polarization
            + (if category == Category::Politics || category == Category::Conspiracy {
                0.08
            } else {
                0.0
            }),
        0.0,
        0.68,
    );
    let observed_buzz = clampf(
        true_hotness
            + 0.18 * news_cycle_intensity
            + 0.16 * social_virality
            + 0.10 * influencer_activity
            + normal(rng) * 0.14,
        0.0,
        1.0,
    );
    let observed_ambiguity = clampf(ambiguity + normal(rng) * 0.16, 0.0, 1.0);
    CandidateTopic {
        id,
        category,
        created_at: now,
        expires_at: now + 5.5,
        true_hotness,
        ambiguity,
        true_theta,
        manipulation_risk,
        news_cycle_intensity,
        social_virality,
        influencer_activity,
        meme_momentum,
        demographic_polarization,
        opinion_event_coupling,
        event_probability,
        turnout_skew,
        bot_pressure,
        referral_elasticity,
        observed_buzz,
        observed_ambiguity,
    }
}

// =============================================================================
// Aggregation.
// =============================================================================

pub fn run_portfolio(
    policy: SchedulerPolicy,
    cfg: &PortfolioConfig,
    mdp: &OperatorMDP,
) -> PolicyRun {
    let rng = mulberry32(cfg.seed.wrapping_add(policy_seed(policy)));
    let mut station = FactMachinePortfolioStation::new(cfg.clone(), policy, mdp, rng);
    let ticks = (cfg.horizon_h / cfg.step_h).round() as i64;
    for tick in 0..=ticks {
        station.run_time_step(cfg.step_h, tick as f64);
    }
    while !station.active.is_empty() {
        let next_close = station
            .active
            .iter()
            .map(|m| m.close_at)
            .fold(f64::INFINITY, f64::min);
        station.run_time_step(cfg.step_h, (next_close / cfg.step_h).ceil());
    }
    station.to_run()
}

fn aggregate_run(
    policy: SchedulerPolicy,
    markets: &[ClosedMarket],
    cfg: &PortfolioConfig,
) -> PolicyAggregate {
    let markets_opened = markets.len() as i64;
    let binary_markets = markets
        .iter()
        .filter(|m| m.kind == MarketKind::Binary)
        .count() as i64;
    let scalar_markets = markets
        .iter()
        .filter(|m| m.kind == MarketKind::Scalar)
        .count() as i64;
    let threshold_markets = markets
        .iter()
        .filter(|m| m.kind == MarketKind::Threshold)
        .count() as i64;
    let fee_revenue: f64 = markets.iter().map(|m| m.fee_revenue).sum();
    let voter_points: f64 = markets.iter().map(|m| m.voter_points).sum();
    let lmsr_loss_total: f64 = markets.iter().map(|m| m.lmsr_loss).sum();
    let platform_surplus = fee_revenue - voter_points * 0.012 - lmsr_loss_total;
    let votes: f64 = markets.iter().map(|m| m.votes as f64).sum();
    let bettors: f64 = markets.iter().map(|m| m.bettors as f64).sum();
    PolicyAggregate {
        scenario_label: cfg.scenario_label.clone(),
        min_market_participants: cfg.min_market_participants,
        policy,
        markets_opened,
        markets_closed: markets.len() as i64,
        binary_markets,
        scalar_markets,
        threshold_markets,
        avg_duration_h: mean_iter(markets.iter().map(|m| m.duration_h)),
        avg_fee_rate: mean_iter(markets.iter().map(|m| m.fee_rate)),
        avg_liquidity: mean_iter(markets.iter().map(|m| m.liquidity)),
        avg_reward_multiplier: mean_iter(markets.iter().map(|m| m.reward_multiplier)),
        proof_markets: markets
            .iter()
            .filter(|m| m.verification == VerificationTier::Proof)
            .count() as i64,
        avg_timing_decay: mean_iter(markets.iter().map(|m| m.timing_decay)),
        votes,
        suspected_sybil_votes: markets.iter().map(|m| m.suspected_sybil_votes as f64).sum(),
        avg_vote_time_fraction: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.avg_vote_time_fraction, m.votes as f64)),
        ),
        avg_timing_multiplier: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.avg_timing_multiplier, m.votes as f64)),
        ),
        bettors,
        trades: markets.iter().map(|m| m.trades as f64).sum(),
        buy_volume: markets.iter().map(|m| m.buy_volume).sum(),
        sell_volume: markets.iter().map(|m| m.sell_volume).sum(),
        fee_revenue,
        voter_points,
        raffle_entries: markets.iter().map(|m| m.raffle_entries as f64).sum(),
        avg_prediction_error: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.avg_prediction_error, m.votes as f64)),
        ),
        avg_opinion_sampling_error: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.opinion_sampling_error, m.votes as f64)),
        ),
        avg_prediction_brier_score: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.prediction_brier_score, (m.trades as f64).max(1.0))),
        ),
        avg_trader_belief_error: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.avg_trader_belief_error, m.bettors as f64)),
        ),
        trader_belief_entropy: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.trader_belief_entropy, m.bettors as f64)),
        ),
        herding_index: weighted_mean_iter(
            markets.iter().map(|m| (m.herding_index, m.bettors as f64)),
        ),
        price_opinion_gap: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.price_opinion_gap, (m.trades as f64).max(1.0))),
        ),
        market_maker_risk_bound: markets.iter().map(|m| m.market_maker_risk_bound).sum(),
        fraud_pressure: weighted_mean_iter(
            markets.iter().map(|m| (m.fraud_pressure, m.votes as f64)),
        ),
        referral_adds: markets.iter().map(|m| m.referral_adds as f64).sum(),
        churn_risk: weighted_mean_iter(markets.iter().map(|m| (m.churn_risk, m.votes as f64))),
        reward_inflation_pressure: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.reward_inflation_pressure, m.votes as f64)),
        ),
        liquidity_utilization: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.liquidity_utilization, (m.trades as f64).max(1.0))),
        ),
        whale_trade_share: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.whale_trade_share, (m.trades as f64).max(1.0))),
        ),
        avg_news_cycle_intensity: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.topic.news_cycle_intensity, m.votes as f64)),
        ),
        avg_social_virality: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.topic.social_virality, m.votes as f64)),
        ),
        avg_influencer_activity: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.topic.influencer_activity, m.votes as f64)),
        ),
        avg_demographic_polarization: weighted_mean_iter(
            markets
                .iter()
                .map(|m| (m.topic.demographic_polarization, m.votes as f64)),
        ),
        trader_pnl: markets.iter().map(|m| m.trader_pnl).sum(),
        lmsr_loss: lmsr_loss_total,
        platform_surplus,
        engagement_score: votes + 1.8 * bettors + 0.35 * markets_opened as f64,
        avg_belief_entropy: None,
        avg_belief_error: None,
    }
}

fn aggregate_by_kind(markets: &[ClosedMarket]) -> Vec<MarketKindAggregate> {
    let kinds = [
        MarketKind::Binary,
        MarketKind::Scalar,
        MarketKind::Threshold,
    ];
    kinds
        .iter()
        .map(|&kind| {
            let subset: Vec<&ClosedMarket> = markets.iter().filter(|m| m.kind == kind).collect();
            let fee_revenue: f64 = subset.iter().map(|m| m.fee_revenue).sum();
            let voter_points: f64 = subset.iter().map(|m| m.voter_points).sum();
            let lmsr_loss_total: f64 = subset.iter().map(|m| m.lmsr_loss).sum();
            MarketKindAggregate {
                kind,
                markets: subset.len() as i64,
                votes: subset.iter().map(|m| m.votes as f64).sum(),
                bettors: subset.iter().map(|m| m.bettors as f64).sum(),
                trades: subset.iter().map(|m| m.trades as f64).sum(),
                buy_volume: subset.iter().map(|m| m.buy_volume).sum(),
                sell_volume: subset.iter().map(|m| m.sell_volume).sum(),
                fee_revenue,
                voter_points,
                suspected_sybil_votes: subset.iter().map(|m| m.suspected_sybil_votes as f64).sum(),
                avg_duration_h: mean_iter(subset.iter().map(|m| m.duration_h)),
                avg_liquidity: mean_iter(subset.iter().map(|m| m.liquidity)),
                avg_fee_rate: mean_iter(subset.iter().map(|m| m.fee_rate)),
                avg_prediction_error: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.avg_prediction_error, m.votes as f64)),
                ),
                avg_opinion_sampling_error: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.opinion_sampling_error, m.votes as f64)),
                ),
                avg_prediction_brier_score: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.prediction_brier_score, (m.trades as f64).max(1.0))),
                ),
                avg_trader_belief_error: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.avg_trader_belief_error, m.bettors as f64)),
                ),
                trader_belief_entropy: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.trader_belief_entropy, m.bettors as f64)),
                ),
                herding_index: weighted_mean_iter(
                    subset.iter().map(|m| (m.herding_index, m.bettors as f64)),
                ),
                price_opinion_gap: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.price_opinion_gap, (m.trades as f64).max(1.0))),
                ),
                fraud_pressure: weighted_mean_iter(
                    subset.iter().map(|m| (m.fraud_pressure, m.votes as f64)),
                ),
                liquidity_utilization: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.liquidity_utilization, (m.trades as f64).max(1.0))),
                ),
                whale_trade_share: weighted_mean_iter(
                    subset
                        .iter()
                        .map(|m| (m.whale_trade_share, (m.trades as f64).max(1.0))),
                ),
                platform_surplus: fee_revenue - voter_points * 0.012 - lmsr_loss_total,
            }
        })
        .collect()
}

fn build_daily_summaries(
    markets: &[ClosedMarket],
    timeline: &[TimelineFrame],
    cfg: &PortfolioConfig,
) -> Vec<DailySummary> {
    let horizon_days = (cfg.horizon_h / 24.0).ceil() as i64;
    let max_close_day = if markets.is_empty() {
        0
    } else {
        markets.iter().map(|m| day_index(m.close_at)).max().unwrap()
    };
    let max_timeline_day = if timeline.is_empty() {
        0
    } else {
        timeline.iter().map(|x| x.day).max().unwrap()
    };
    let days = horizon_days
        .max(max_close_day + 1)
        .max(max_timeline_day + 1);
    let mut last_timeline_by_day: HashMap<i64, TimelineFrame> = HashMap::new();
    for frame in timeline {
        last_timeline_by_day.insert(frame.day, frame.clone());
    }

    let mut summaries = Vec::new();
    for day in 0..days {
        let opened: Vec<&ClosedMarket> = markets
            .iter()
            .filter(|m| day_index(m.open_at) == day)
            .collect();
        let closed: Vec<&ClosedMarket> = markets
            .iter()
            .filter(|m| day_index(m.close_at) == day)
            .collect();
        let last_frame = last_timeline_by_day.get(&day);
        summaries.push(DailySummary {
            day,
            market_cap: daily_market_cap_for_day(day, cfg),
            opened: opened.len() as i64,
            closed: closed.len() as i64,
            active_end: last_frame.map(|f| f.open).unwrap_or(0),
            queued_end: last_frame.map(|f| f.queued).unwrap_or(0),
            votes: closed.iter().map(|m| m.votes as f64).sum(),
            bettors: closed.iter().map(|m| m.bettors as f64).sum(),
            trades: closed.iter().map(|m| m.trades as f64).sum(),
            fee_revenue: closed.iter().map(|m| m.fee_revenue).sum(),
            voter_points: closed.iter().map(|m| m.voter_points).sum(),
            binary_closed: closed
                .iter()
                .filter(|m| m.kind == MarketKind::Binary)
                .count() as i64,
            scalar_closed: closed
                .iter()
                .filter(|m| m.kind == MarketKind::Scalar)
                .count() as i64,
            threshold_closed: closed
                .iter()
                .filter(|m| m.kind == MarketKind::Threshold)
                .count() as i64,
            avg_prediction_error: weighted_mean_iter(
                closed
                    .iter()
                    .map(|m| (m.avg_prediction_error, m.votes as f64)),
            ),
            avg_opinion_sampling_error: weighted_mean_iter(
                closed
                    .iter()
                    .map(|m| (m.opinion_sampling_error, m.votes as f64)),
            ),
            avg_prediction_brier_score: weighted_mean_iter(
                closed
                    .iter()
                    .map(|m| (m.prediction_brier_score, (m.trades as f64).max(1.0))),
            ),
            fraud_pressure: weighted_mean_iter(
                closed.iter().map(|m| (m.fraud_pressure, m.votes as f64)),
            ),
            herding_index: weighted_mean_iter(
                closed.iter().map(|m| (m.herding_index, m.bettors as f64)),
            ),
        });
    }
    summaries
}

// =============================================================================
// Daily market caps + config.
// =============================================================================

pub fn build_daily_market_caps(
    horizon_days: f64,
    min_daily_markets: i64,
    max_daily_markets: i64,
    seed: u32,
) -> Vec<i64> {
    let days = (horizon_days.ceil() as i64).max(1);
    let lo = min_daily_markets.max(0);
    let hi = max_daily_markets.max(lo);
    if lo == hi {
        return vec![lo; days as usize];
    }
    let mut rng = mulberry32(seed.wrapping_add(0x9e37_79b9));
    let mut caps: Vec<i64> = Vec::new();
    for day in 0..days {
        let weekly_pulse =
            0.5 + 0.5 * (2.0 * std::f64::consts::PI * (day as f64 + (seed % 7) as f64) / 7.0).sin();
        let news_shock = rng.next_float();
        let weekend_drag = if day % 7 == 5 || day % 7 == 6 {
            -0.12
        } else {
            0.04
        };
        let level = clampf(
            0.52 * weekly_pulse + 0.40 * news_shock + weekend_drag,
            0.0,
            1.0,
        );
        caps.push((lo as f64 + (hi - lo) as f64 * level).round() as i64);
    }
    caps[0] = lo;
    if days > 1 {
        let idx = (days as f64 * 0.62).floor() as usize;
        caps[idx] = hi;
    }
    caps
}

pub fn daily_market_cap_for_day(day: i64, cfg: &PortfolioConfig) -> i64 {
    if cfg.daily_market_caps.is_empty() {
        return cfg.max_daily_markets;
    }
    let idx = clampi(day, 0, cfg.daily_market_caps.len() as i64 - 1) as usize;
    cfg.daily_market_caps[idx]
}

pub fn day_index(t_hours: f64) -> i64 {
    (t_hours / 24.0).floor().max(0.0) as i64
}

fn parse_daily_market_caps(raw: Option<&str>, horizon_days: i64) -> Option<Vec<i64>> {
    let raw = raw?;
    let mut parsed: Vec<i64> = raw
        .split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .map(|x| x.floor() as i64)
        .filter(|&x| x >= 0)
        .collect();
    if parsed.is_empty() {
        return None;
    }
    while (parsed.len() as i64) < horizon_days {
        parsed.push(*parsed.last().unwrap());
    }
    parsed.truncate(horizon_days.max(0) as usize);
    Some(parsed)
}

pub fn default_config() -> PortfolioConfig {
    let min_market_participants = env_f64("MIN_MARKET_PARTICIPANTS", 1000.0);
    let horizon_h = env_f64("HORIZON_H", 24.0 * 50.0);
    let horizon_days = (horizon_h / 24.0).ceil() as i64;
    let seed = env_f64("SEED", 42.0) as u32;
    let min_daily_markets = env_f64("MIN_DAILY_MARKETS", 2.0).floor().max(0.0) as i64;
    let max_daily_markets = env_f64("MAX_DAILY_MARKETS", 10.0)
        .floor()
        .max(min_daily_markets as f64) as i64;
    let daily_market_caps =
        parse_daily_market_caps(env_opt("DAILY_MARKET_CAPS").as_deref(), horizon_days)
            .unwrap_or_else(|| {
                build_daily_market_caps(
                    horizon_days as f64,
                    min_daily_markets,
                    max_daily_markets,
                    seed,
                )
            });
    PortfolioConfig {
        scenario_label: env_opt("SCENARIO_LABEL")
            .unwrap_or_else(|| format!("{} participants", locale_int(min_market_participants))),
        horizon_h,
        step_h: env_f64("STEP_H", 0.25),
        max_concurrent: env_f64("MAX_CONCURRENT", max_daily_markets as f64) as i64,
        min_daily_markets,
        max_daily_markets,
        daily_market_caps,
        seed,
        liquidity: env_f64("LIQUIDITY", 500.0),
        fee_rate: env_f64("FEE_RATE", 0.01),
        scalar_bins: env_f64("SCALAR_BINS", 7.0) as usize,
        min_market_participants,
    }
}

pub fn scenario_configs(base: &PortfolioConfig) -> Vec<PortfolioConfig> {
    let scales: Vec<f64> = match env_opt("PARTICIPANT_SCALES") {
        Some(raw) => raw
            .split(',')
            .filter_map(|x| x.trim().parse::<f64>().ok())
            .filter(|&x| x.is_finite() && x > 0.0)
            .collect(),
        None => vec![1000.0, 10000.0],
    };
    scales
        .into_iter()
        .map(|scale| {
            let mut cfg = base.clone();
            cfg.scenario_label = format!("{} participants", locale_int(scale));
            cfg.min_market_participants = scale;
            cfg
        })
        .collect()
}

// =============================================================================
// Driver.
// =============================================================================

pub fn run() {
    let cfg = default_config();
    let scenarios = scenario_configs(&cfg);
    let mdp = build_operator_mdp();
    let policies = [
        SchedulerPolicy::FixedDaily,
        SchedulerPolicy::GreedyBuzz,
        SchedulerPolicy::MdpOracle,
        SchedulerPolicy::PomdpBelief,
    ];
    let mut runs: Vec<PolicyRun> = Vec::new();
    for scenario in &scenarios {
        for &p in &policies {
            runs.push(run_portfolio(p, scenario, &mdp));
        }
    }

    // PORT NOTE: TS resolves out/ relative to __dirname; here it is cwd-relative.
    let _ = std::fs::create_dir_all("out");
    let html_path = "out/factmachine-markets.html";
    let json_path = "out/factmachine-markets-results.json";
    let _ = std::fs::write(html_path, build_html(&runs, &mdp, &cfg));
    let results_json = jobj(vec![
        ("config", cfg_json(&cfg)),
        ("mdp", summarize_operator_mdp_json(&mdp)),
        ("runs", jarr(runs.iter().map(policy_run_json).collect())),
    ]);
    let _ = std::fs::write(json_path, results_json.to_string_pretty(2));

    println!("# FactMachine multi-market simulation");
    println!(
        "#   horizon={}h, maxConcurrent={}, scenarios={}",
        num_str(cfg.horizon_h),
        cfg.max_concurrent,
        scenarios
            .iter()
            .map(|s| num_str(s.min_market_participants))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "#   operator MDP: {} states, {} actions, {} VI sweeps",
        mdp.spec.num_states,
        mdp.actions.len(),
        mdp.iterations
    );
    for r in &runs {
        let a = &r.aggregate;
        println!();
        println!("# {} / {}", r.scenario_label, r.policy.slug());
        println!(
            "#   markets {} ({} binary, {} scalar, {} over/under), avg duration {:.2}h",
            a.markets_closed,
            a.binary_markets,
            a.scalar_markets,
            a.threshold_markets,
            a.avg_duration_h
        );
        println!(
            "#   avg fee {:.2}%, avg liquidity ${:.0}, avg reward {:.2}x, timing decay {:.2}, proof markets {}",
            100.0 * a.avg_fee_rate,
            a.avg_liquidity,
            a.avg_reward_multiplier,
            a.avg_timing_decay,
            a.proof_markets
        );
        println!(
            "#   votes {}, suspect votes {}, bettors {}, trades {}, raffle entries {}",
            num_str(a.votes),
            num_str(a.suspected_sybil_votes),
            num_str(a.bettors),
            num_str(a.trades),
            num_str(a.raffle_entries)
        );
        println!(
            "#   user submodels: avg vote time {:.1}%, timing boost {:.2}x, trader belief error {:.1}%, herding {:.1}%",
            100.0 * a.avg_vote_time_fraction,
            a.avg_timing_multiplier,
            100.0 * a.avg_trader_belief_error,
            100.0 * a.herding_index
        );
        println!(
            "#   opinion vs prediction: opinion sampling error {:.1}%, prediction Brier {:.3}, price/opinion gap {:.1}%",
            100.0 * a.avg_opinion_sampling_error,
            a.avg_prediction_brier_score,
            100.0 * a.price_opinion_gap
        );
        println!(
            "#   fees ${:.2}, voter points {:.1}, surplus ${:.2}, engagement {:.1}",
            a.fee_revenue, a.voter_points, a.platform_surplus, a.engagement_score
        );
        if let Some(belief_error) = a.avg_belief_error {
            println!(
                "#   POMDP belief entropy {:.3}, hotness error {:.1}%",
                a.avg_belief_entropy.unwrap_or(0.0),
                100.0 * belief_error
            );
        }
    }
    println!();
    println!("# wrote {html_path}");
    println!("# wrote {json_path}");
}

// =============================================================================
// MDP state encoding + labels.
// =============================================================================

fn encode_operator_state(hot_bin: i64, amb_bin: i64, fatigue_bin: i64) -> usize {
    ((hot_bin * 3 + amb_bin) * 3 + fatigue_bin) as usize
}
fn decode_operator_state(s: usize) -> (i64, i64, i64) {
    let mut s = s as i64;
    let fatigue_bin = s % 3;
    s /= 3;
    let amb_bin = s % 3;
    s /= 3;
    (s, amb_bin, fatigue_bin)
}
fn operator_state_label(s: usize) -> String {
    let (hot_bin, amb_bin, fatigue_bin) = decode_operator_state(s);
    format!("hot={hot_bin}/amb={amb_bin}/fatigue={fatigue_bin}")
}
fn fatigue_bin_for(active: i64, cap: i64) -> i64 {
    let x = if cap <= 0 {
        1.0
    } else {
        active as f64 / cap as f64
    };
    if x < 0.34 {
        0
    } else if x < 0.67 {
        1
    } else {
        2
    }
}

fn action_by(
    actions: &[SchedulerAction],
    kind: MarketKind,
    duration_h: f64,
    label_hint: Option<&str>,
) -> SchedulerAction {
    let target = kind.as_kind();
    let hint_ok = |a: &SchedulerAction| match label_hint {
        Some(h) => a.label.contains(h),
        None => true,
    };
    actions
        .iter()
        .find(|a| a.kind == target && a.duration_h == duration_h && hint_ok(a))
        .or_else(|| {
            actions
                .iter()
                .find(|a| a.kind == target && a.duration_h == duration_h)
        })
        .cloned()
        .unwrap_or_else(|| actions[0].clone())
}

fn contract_label(action: &SchedulerAction) -> String {
    match action.kind {
        ActionKind::Threshold => format!(
            "over/under {}%",
            (action.threshold.unwrap_or(0.55) * 100.0).round() as i64
        ),
        ActionKind::Scalar => "scalar distribution".to_string(),
        ActionKind::Binary => "majority binary".to_string(),
        ActionKind::Wait => "wait".to_string(),
    }
}

fn scalar_bin_midpoint(i: usize, bins: usize) -> f64 {
    clampf((i as f64 + 0.5) / (bins as f64).max(1.0), 0.0, 1.0)
}

// =============================================================================
// Behavioral multipliers.
// =============================================================================

fn verification_participation_multiplier(tier: VerificationTier) -> f64 {
    match tier {
        VerificationTier::Proof => 0.84,
        VerificationTier::Basic => 0.96,
        VerificationTier::Open => 1.08,
    }
}
fn verification_trust_multiplier(tier: VerificationTier) -> f64 {
    match tier {
        VerificationTier::Proof => 1.18,
        VerificationTier::Basic => 1.0,
        VerificationTier::Open => 0.88,
    }
}
fn manipulation_multiplier(tier: VerificationTier) -> f64 {
    match tier {
        VerificationTier::Proof => 0.32,
        VerificationTier::Basic => 0.62,
        VerificationTier::Open => 1.0,
    }
}
fn market_trader_fit(kind: MarketKind) -> f64 {
    match kind {
        MarketKind::Binary => 1.08,
        MarketKind::Threshold => 1.14,
        MarketKind::Scalar => 0.88,
    }
}
fn threshold_drama(theta: f64, threshold: f64) -> f64 {
    (-(theta - threshold).powi(2) / 0.045).exp()
}
fn information_engagement_multiplier(mode: InformationMode) -> f64 {
    match mode {
        InformationMode::PriceOnly => 0.88,
        InformationMode::DelayedVotes => 0.98,
        InformationMode::LiveVotes => 1.12,
        InformationMode::DemographicSlices => 1.08,
        InformationMode::MomentumSignals => 1.18,
    }
}
fn information_trust_multiplier(mode: InformationMode) -> f64 {
    match mode {
        InformationMode::PriceOnly => 0.88,
        InformationMode::DelayedVotes => 1.04,
        InformationMode::LiveVotes => 0.95,
        InformationMode::DemographicSlices => 1.15,
        InformationMode::MomentumSignals => 0.92,
    }
}
fn information_herding_multiplier(mode: InformationMode) -> f64 {
    match mode {
        InformationMode::PriceOnly => 0.55,
        InformationMode::DelayedVotes => 0.72,
        InformationMode::LiveVotes => 1.15,
        InformationMode::DemographicSlices => 0.95,
        InformationMode::MomentumSignals => 1.35,
    }
}
fn information_signal_weight(mode: InformationMode) -> f64 {
    match mode {
        InformationMode::PriceOnly => 0.18,
        InformationMode::DelayedVotes => 0.34,
        InformationMode::LiveVotes => 0.54,
        InformationMode::DemographicSlices => 0.48,
        InformationMode::MomentumSignals => 0.62,
    }
}
fn information_wait_pressure(mode: InformationMode) -> f64 {
    match mode {
        InformationMode::PriceOnly => 0.15,
        InformationMode::DelayedVotes => 0.34,
        InformationMode::LiveVotes => 0.62,
        InformationMode::DemographicSlices => 0.48,
        InformationMode::MomentumSignals => 0.55,
    }
}
fn information_observation_noise(mode: InformationMode, ambiguity: f64) -> f64 {
    let base = match mode {
        InformationMode::PriceOnly => 0.16,
        InformationMode::DelayedVotes => 0.11,
        InformationMode::LiveVotes => 0.08,
        InformationMode::DemographicSlices => 0.07,
        InformationMode::MomentumSignals => 0.10,
    };
    base + ambiguity * 0.05
}

// =============================================================================
// Math + RNG helpers.
// =============================================================================

fn bernoulli_entropy(p: f64) -> f64 {
    let q = clampf(p, 1e-9, 1.0 - 1e-9);
    -(q * q.ln() + (1.0 - q) * (1.0 - q).ln())
}
fn bin3(x: f64) -> i64 {
    if x < 0.34 {
        0
    } else if x < 0.67 {
        1
    } else {
        2
    }
}
fn hotness_midpoint(bin: i64) -> f64 {
    match bin {
        0 => 0.20,
        1 => 0.52,
        2 => 0.84,
        _ => 0.52,
    }
}
fn ambiguity_midpoint(bin: i64) -> f64 {
    match bin {
        0 => 0.18,
        1 => 0.50,
        2 => 0.82,
        _ => 0.5,
    }
}
fn beta_like(rng: &mut SeededRandom, a: f64, b: f64) -> f64 {
    let x = -(rng.next_float().max(1e-12)).ln() / a;
    let y = -(rng.next_float().max(1e-12)).ln() / b;
    x / (x + y)
}
fn sample_poisson(lambda: f64, rng: &mut SeededRandom) -> i64 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0i64;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.next_float();
        if p <= l {
            break;
        }
    }
    k - 1
}
fn exp_sample(rate: f64, rng: &mut SeededRandom) -> f64 {
    -((1.0 - rng.next_float()).max(1e-12)).ln() / rate.max(1e-12)
}
fn normal(rng: &mut SeededRandom) -> f64 {
    let u = rng.next_float().max(1e-12);
    let v = rng.next_float().max(1e-12);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}
fn categorical(weights: &[f64], rng: &mut SeededRandom) -> usize {
    let total: f64 = weights.iter().sum();
    let mut u = rng.next_float() * total;
    for (i, &w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}
fn softmax(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|x| x / z).collect()
}
fn entropy(ps: &[f64]) -> f64 {
    -ps.iter()
        .fold(0.0, |s, &p| if p > 0.0 { s + p * p.ln() } else { s })
}
fn arg_max(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}
fn weighted_mean_iter(rows: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut w = 0.0;
    let mut acc = 0.0;
    for (value, weight) in rows {
        w += weight;
        acc += value * weight;
    }
    if w == 0.0 {
        0.0
    } else {
        acc / w
    }
}
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}
fn mean_iter(xs: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0usize;
    let mut acc = 0.0;
    for x in xs {
        n += 1;
        acc += x;
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f64
    }
}
fn clampf(x: f64, lo: f64, hi: f64) -> f64 {
    x.min(hi).max(lo)
}
fn clampi(x: i64, lo: i64, hi: i64) -> i64 {
    x.min(hi).max(lo)
}
fn bump(counts: &mut Vec<ActionCount>, key: &str) {
    if let Some(e) = counts.iter_mut().find(|c| c.action == key) {
        e.count += 1;
    } else {
        counts.push(ActionCount {
            action: key.to_string(),
            count: 1,
        });
    }
}
fn policy_seed(policy: SchedulerPolicy) -> u32 {
    match policy {
        SchedulerPolicy::FixedDaily => 10,
        SchedulerPolicy::GreedyBuzz => 20,
        SchedulerPolicy::MdpOracle => 30,
        SchedulerPolicy::PomdpBelief => 40,
    }
}

/// `Number.toLocaleString()` for integer-valued counts (thousands grouping).
fn locale_int(n: f64) -> String {
    let neg = n < 0.0;
    let digits = (n.abs().round() as i64).to_string();
    let len = digits.len();
    let mut out = String::new();
    for (idx, ch) in digits.chars().enumerate() {
        if idx > 0 && (len - idx).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if neg {
        format!("-{out}")
    } else {
        out
    }
}

/// JS `String(number)` for a finite f64 (drops trailing `.0` for integers).
fn num_str(x: f64) -> String {
    JsonValue::Number(x).to_string()
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}
fn env_opt(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

// =============================================================================
// JSON serialization (for the HTML data island + results.json).
// =============================================================================

fn jnum(x: f64) -> JsonValue {
    JsonValue::Number(x)
}
fn jstr(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}
fn jobj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}
fn jarr(items: Vec<JsonValue>) -> JsonValue {
    JsonValue::Array(items)
}

fn topic_json(t: &CandidateTopic) -> JsonValue {
    jobj(vec![
        ("id", jnum(t.id as f64)),
        ("category", jstr(t.category.slug())),
        ("createdAt", jnum(t.created_at)),
        ("expiresAt", jnum(t.expires_at)),
        ("trueHotness", jnum(t.true_hotness)),
        ("ambiguity", jnum(t.ambiguity)),
        ("trueTheta", jnum(t.true_theta)),
        ("manipulationRisk", jnum(t.manipulation_risk)),
        ("newsCycleIntensity", jnum(t.news_cycle_intensity)),
        ("socialVirality", jnum(t.social_virality)),
        ("influencerActivity", jnum(t.influencer_activity)),
        ("memeMomentum", jnum(t.meme_momentum)),
        ("demographicPolarization", jnum(t.demographic_polarization)),
        ("opinionEventCoupling", jnum(t.opinion_event_coupling)),
        ("eventProbability", jnum(t.event_probability)),
        ("turnoutSkew", jnum(t.turnout_skew)),
        ("botPressure", jnum(t.bot_pressure)),
        ("referralElasticity", jnum(t.referral_elasticity)),
        ("observedBuzz", jnum(t.observed_buzz)),
        ("observedAmbiguity", jnum(t.observed_ambiguity)),
    ])
}

fn action_json(a: &SchedulerAction) -> JsonValue {
    let mut entries = vec![
        ("label", jstr(&a.label)),
        ("kind", jstr(a.kind.slug())),
        ("durationH", jnum(a.duration_h)),
        ("feeRate", jnum(a.fee_rate)),
        ("liquidityMultiplier", jnum(a.liquidity_multiplier)),
        ("rewardMultiplier", jnum(a.reward_multiplier)),
        ("verification", jstr(a.verification.slug())),
        ("informationMode", jstr(a.information_mode.slug())),
        ("timingDecay", jnum(a.timing_decay)),
    ];
    if let Some(th) = a.threshold {
        entries.push(("threshold", jnum(th)));
    }
    entries.push(("description", jstr(&a.description)));
    jobj(entries)
}

fn closed_json(m: &ClosedMarket) -> JsonValue {
    let mut entries = vec![
        ("id", jnum(m.id as f64)),
        ("topic", topic_json(&m.topic)),
        ("kind", jstr(m.kind.slug())),
        ("contractLabel", jstr(&m.contract_label)),
        ("durationH", jnum(m.duration_h)),
        ("openAt", jnum(m.open_at)),
        ("closeAt", jnum(m.close_at)),
        ("feeRate", jnum(m.fee_rate)),
        ("liquidity", jnum(m.liquidity)),
        ("rewardMultiplier", jnum(m.reward_multiplier)),
        ("verification", jstr(m.verification.slug())),
        ("informationMode", jstr(m.information_mode.slug())),
        ("timingDecay", jnum(m.timing_decay)),
    ];
    if let Some(th) = m.threshold {
        entries.push(("threshold", jnum(th)));
    }
    entries.extend(vec![
        ("finalVoteFraction", jnum(m.final_vote_fraction)),
        ("outcomeIndex", jnum(m.outcome_index as f64)),
        ("votes", jnum(m.votes as f64)),
        ("suspectedSybilVotes", jnum(m.suspected_sybil_votes as f64)),
        ("avgVoteTimeFraction", jnum(m.avg_vote_time_fraction)),
        ("avgTimingMultiplier", jnum(m.avg_timing_multiplier)),
        ("bettors", jnum(m.bettors as f64)),
        ("trades", jnum(m.trades as f64)),
        ("buyVolume", jnum(m.buy_volume)),
        ("sellVolume", jnum(m.sell_volume)),
        ("feeRevenue", jnum(m.fee_revenue)),
        ("voterPoints", jnum(m.voter_points)),
        ("raffleEntries", jnum(m.raffle_entries as f64)),
        ("avgPredictionError", jnum(m.avg_prediction_error)),
        ("opinionSamplingError", jnum(m.opinion_sampling_error)),
        ("predictionBrierScore", jnum(m.prediction_brier_score)),
        ("externalOutcome", jnum(m.external_outcome as f64)),
        ("avgTraderBeliefError", jnum(m.avg_trader_belief_error)),
        ("traderBeliefEntropy", jnum(m.trader_belief_entropy)),
        ("herdingIndex", jnum(m.herding_index)),
        ("priceOpinionGap", jnum(m.price_opinion_gap)),
        ("marketMakerRiskBound", jnum(m.market_maker_risk_bound)),
        ("fraudPressure", jnum(m.fraud_pressure)),
        ("referralAdds", jnum(m.referral_adds as f64)),
        ("churnRisk", jnum(m.churn_risk)),
        ("rewardInflationPressure", jnum(m.reward_inflation_pressure)),
        ("liquidityUtilization", jnum(m.liquidity_utilization)),
        ("whaleTradeShare", jnum(m.whale_trade_share)),
        ("traderPnl", jnum(m.trader_pnl)),
        ("lmsrLoss", jnum(m.lmsr_loss)),
    ]);
    jobj(entries)
}

fn kind_agg_json(k: &MarketKindAggregate) -> JsonValue {
    jobj(vec![
        ("kind", jstr(k.kind.slug())),
        ("markets", jnum(k.markets as f64)),
        ("votes", jnum(k.votes)),
        ("bettors", jnum(k.bettors)),
        ("trades", jnum(k.trades)),
        ("buyVolume", jnum(k.buy_volume)),
        ("sellVolume", jnum(k.sell_volume)),
        ("feeRevenue", jnum(k.fee_revenue)),
        ("voterPoints", jnum(k.voter_points)),
        ("suspectedSybilVotes", jnum(k.suspected_sybil_votes)),
        ("avgDurationH", jnum(k.avg_duration_h)),
        ("avgLiquidity", jnum(k.avg_liquidity)),
        ("avgFeeRate", jnum(k.avg_fee_rate)),
        ("avgPredictionError", jnum(k.avg_prediction_error)),
        (
            "avgOpinionSamplingError",
            jnum(k.avg_opinion_sampling_error),
        ),
        (
            "avgPredictionBrierScore",
            jnum(k.avg_prediction_brier_score),
        ),
        ("avgTraderBeliefError", jnum(k.avg_trader_belief_error)),
        ("traderBeliefEntropy", jnum(k.trader_belief_entropy)),
        ("herdingIndex", jnum(k.herding_index)),
        ("priceOpinionGap", jnum(k.price_opinion_gap)),
        ("fraudPressure", jnum(k.fraud_pressure)),
        ("liquidityUtilization", jnum(k.liquidity_utilization)),
        ("whaleTradeShare", jnum(k.whale_trade_share)),
        ("platformSurplus", jnum(k.platform_surplus)),
    ])
}

fn daily_json(d: &DailySummary) -> JsonValue {
    jobj(vec![
        ("day", jnum(d.day as f64)),
        ("marketCap", jnum(d.market_cap as f64)),
        ("opened", jnum(d.opened as f64)),
        ("closed", jnum(d.closed as f64)),
        ("activeEnd", jnum(d.active_end as f64)),
        ("queuedEnd", jnum(d.queued_end as f64)),
        ("votes", jnum(d.votes)),
        ("bettors", jnum(d.bettors)),
        ("trades", jnum(d.trades)),
        ("feeRevenue", jnum(d.fee_revenue)),
        ("voterPoints", jnum(d.voter_points)),
        ("binaryClosed", jnum(d.binary_closed as f64)),
        ("scalarClosed", jnum(d.scalar_closed as f64)),
        ("thresholdClosed", jnum(d.threshold_closed as f64)),
        ("avgPredictionError", jnum(d.avg_prediction_error)),
        (
            "avgOpinionSamplingError",
            jnum(d.avg_opinion_sampling_error),
        ),
        (
            "avgPredictionBrierScore",
            jnum(d.avg_prediction_brier_score),
        ),
        ("fraudPressure", jnum(d.fraud_pressure)),
        ("herdingIndex", jnum(d.herding_index)),
    ])
}

fn aggregate_json(a: &PolicyAggregate) -> JsonValue {
    let mut entries = vec![
        ("scenarioLabel", jstr(&a.scenario_label)),
        ("minMarketParticipants", jnum(a.min_market_participants)),
        ("policy", jstr(a.policy.slug())),
        ("marketsOpened", jnum(a.markets_opened as f64)),
        ("marketsClosed", jnum(a.markets_closed as f64)),
        ("binaryMarkets", jnum(a.binary_markets as f64)),
        ("scalarMarkets", jnum(a.scalar_markets as f64)),
        ("thresholdMarkets", jnum(a.threshold_markets as f64)),
        ("avgDurationH", jnum(a.avg_duration_h)),
        ("avgFeeRate", jnum(a.avg_fee_rate)),
        ("avgLiquidity", jnum(a.avg_liquidity)),
        ("avgRewardMultiplier", jnum(a.avg_reward_multiplier)),
        ("proofMarkets", jnum(a.proof_markets as f64)),
        ("avgTimingDecay", jnum(a.avg_timing_decay)),
        ("votes", jnum(a.votes)),
        ("suspectedSybilVotes", jnum(a.suspected_sybil_votes)),
        ("avgVoteTimeFraction", jnum(a.avg_vote_time_fraction)),
        ("avgTimingMultiplier", jnum(a.avg_timing_multiplier)),
        ("bettors", jnum(a.bettors)),
        ("trades", jnum(a.trades)),
        ("buyVolume", jnum(a.buy_volume)),
        ("sellVolume", jnum(a.sell_volume)),
        ("feeRevenue", jnum(a.fee_revenue)),
        ("voterPoints", jnum(a.voter_points)),
        ("raffleEntries", jnum(a.raffle_entries)),
        ("avgPredictionError", jnum(a.avg_prediction_error)),
        (
            "avgOpinionSamplingError",
            jnum(a.avg_opinion_sampling_error),
        ),
        (
            "avgPredictionBrierScore",
            jnum(a.avg_prediction_brier_score),
        ),
        ("avgTraderBeliefError", jnum(a.avg_trader_belief_error)),
        ("traderBeliefEntropy", jnum(a.trader_belief_entropy)),
        ("herdingIndex", jnum(a.herding_index)),
        ("priceOpinionGap", jnum(a.price_opinion_gap)),
        ("marketMakerRiskBound", jnum(a.market_maker_risk_bound)),
        ("fraudPressure", jnum(a.fraud_pressure)),
        ("referralAdds", jnum(a.referral_adds)),
        ("churnRisk", jnum(a.churn_risk)),
        ("rewardInflationPressure", jnum(a.reward_inflation_pressure)),
        ("liquidityUtilization", jnum(a.liquidity_utilization)),
        ("whaleTradeShare", jnum(a.whale_trade_share)),
        ("avgNewsCycleIntensity", jnum(a.avg_news_cycle_intensity)),
        ("avgSocialVirality", jnum(a.avg_social_virality)),
        ("avgInfluencerActivity", jnum(a.avg_influencer_activity)),
        (
            "avgDemographicPolarization",
            jnum(a.avg_demographic_polarization),
        ),
        ("traderPnl", jnum(a.trader_pnl)),
        ("lmsrLoss", jnum(a.lmsr_loss)),
        ("platformSurplus", jnum(a.platform_surplus)),
        ("engagementScore", jnum(a.engagement_score)),
    ];
    if let Some(entropy) = a.avg_belief_entropy {
        entries.push(("avgBeliefEntropy", jnum(entropy)));
    }
    if let Some(err) = a.avg_belief_error {
        entries.push(("avgBeliefError", jnum(err)));
    }
    jobj(entries)
}

fn timeline_json(f: &TimelineFrame) -> JsonValue {
    jobj(vec![
        ("t", jnum(f.t)),
        ("day", jnum(f.day as f64)),
        ("open", jnum(f.open as f64)),
        ("closed", jnum(f.closed as f64)),
        ("queued", jnum(f.queued as f64)),
        ("votes", jnum(f.votes as f64)),
        ("bettors", jnum(f.bettors as f64)),
        ("trades", jnum(f.trades as f64)),
        ("fees", jnum(f.fees)),
        ("marketCap", jnum(f.market_cap as f64)),
        ("openedToday", jnum(f.opened_today as f64)),
        ("openedTotal", jnum(f.opened_total as f64)),
    ])
}

fn policy_run_json(r: &PolicyRun) -> JsonValue {
    let mut entries = vec![
        ("scenarioLabel", jstr(&r.scenario_label)),
        ("minMarketParticipants", jnum(r.min_market_participants)),
        ("policy", jstr(r.policy.slug())),
        ("aggregate", aggregate_json(&r.aggregate)),
        (
            "kindBreakdown",
            jarr(r.kind_breakdown.iter().map(kind_agg_json).collect()),
        ),
        ("daily", jarr(r.daily.iter().map(daily_json).collect())),
        (
            "closedMarkets",
            jarr(r.closed_markets.iter().map(closed_json).collect()),
        ),
        (
            "actionCounts",
            jarr(
                r.action_counts
                    .iter()
                    .map(|c| {
                        jobj(vec![
                            ("action", jstr(&c.action)),
                            ("count", jnum(c.count as f64)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "timeline",
            jarr(r.timeline.iter().map(timeline_json).collect()),
        ),
    ];
    if let Some(bt) = &r.belief_trace {
        entries.push((
            "beliefTrace",
            jarr(
                bt.iter()
                    .map(|x| {
                        jobj(vec![
                            ("t", jnum(x.t)),
                            ("entropy", jnum(x.entropy)),
                            ("expectedHotness", jnum(x.expected_hotness)),
                            ("error", jnum(x.error)),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    jobj(entries)
}

fn cfg_json(cfg: &PortfolioConfig) -> JsonValue {
    jobj(vec![
        ("scenarioLabel", jstr(&cfg.scenario_label)),
        ("horizonH", jnum(cfg.horizon_h)),
        ("stepH", jnum(cfg.step_h)),
        ("maxConcurrent", jnum(cfg.max_concurrent as f64)),
        ("minDailyMarkets", jnum(cfg.min_daily_markets as f64)),
        ("maxDailyMarkets", jnum(cfg.max_daily_markets as f64)),
        (
            "dailyMarketCaps",
            jarr(
                cfg.daily_market_caps
                    .iter()
                    .map(|&c| jnum(c as f64))
                    .collect(),
            ),
        ),
        ("seed", jnum(cfg.seed as f64)),
        ("liquidity", jnum(cfg.liquidity)),
        ("feeRate", jnum(cfg.fee_rate)),
        ("scalarBins", jnum(cfg.scalar_bins as f64)),
        ("minMarketParticipants", jnum(cfg.min_market_participants)),
    ])
}

fn summarize_operator_mdp_json(mdp: &OperatorMDP) -> JsonValue {
    let mut full_policy: Vec<JsonValue> = Vec::with_capacity(mdp.spec.num_states);
    for s in 0..mdp.spec.num_states {
        let mut qs = mdp.q[s].clone();
        qs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let (hot_bin, amb_bin, fatigue_bin) = decode_operator_state(s);
        let q_gap = if qs.len() >= 2 { qs[0] - qs[1] } else { 0.0 };
        full_policy.push(jobj(vec![
            ("state", jstr(&operator_state_label(s))),
            ("hotBin", jnum(hot_bin as f64)),
            ("ambBin", jnum(amb_bin as f64)),
            ("fatigueBin", jnum(fatigue_bin as f64)),
            ("action", jstr(&mdp.actions[mdp.policy[s] as usize].label)),
            ("value", jnum(mdp.v[s])),
            ("qGap", jnum(q_gap)),
            ("bestQ", jnum(qs.first().copied().unwrap_or(0.0))),
            ("secondQ", jnum(qs.get(1).copied().unwrap_or(0.0))),
        ]));
    }
    let sample_policy: Vec<JsonValue> = full_policy.iter().take(12).cloned().collect();
    jobj(vec![
        ("numStates", jnum(mdp.spec.num_states as f64)),
        (
            "actions",
            jarr(mdp.actions.iter().map(action_json).collect()),
        ),
        ("iterations", jnum(mdp.iterations as f64)),
        ("finalDelta", jnum(mdp.final_delta)),
        ("gamma", jnum(mdp.gamma)),
        ("fullPolicy", jarr(full_policy)),
        ("samplePolicy", jarr(sample_policy)),
    ])
}

// =============================================================================
// HTML report.
// =============================================================================

fn build_data_json(runs: &[PolicyRun], mdp: &OperatorMDP, cfg: &PortfolioConfig) -> String {
    let data = jobj(vec![
        ("runs", jarr(runs.iter().map(policy_run_json).collect())),
        ("mdp", summarize_operator_mdp_json(mdp)),
        ("cfg", cfg_json(cfg)),
    ]);
    // Mirror the TS `</script` / U+2028 / U+2029 escaping for safe inlining.
    data.to_string()
        .replace("</script", "<\\/script")
        .replace("</SCRIPT", "<\\/SCRIPT")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

fn build_html(runs: &[PolicyRun], mdp: &OperatorMDP, cfg: &PortfolioConfig) -> String {
    let data = build_data_json(runs, mdp, cfg);
    let mut s = String::new();
    s.push_str(HTML_HEAD_A);
    s.push_str(&num_str(cfg.horizon_h / 24.0));
    s.push_str(" days with daily launch capacity from ");
    s.push_str(&cfg.min_daily_markets.to_string());
    s.push_str(" to ");
    s.push_str(&cfg.max_daily_markets.to_string());
    s.push_str(HTML_HEAD_B);
    s.push_str(&mdp.spec.num_states.to_string());
    s.push_str(" operator states: topic hotness, topic ambiguity, and market-load fatigue. For each state it evaluates ");
    s.push_str(&mdp.actions.len().to_string());
    s.push_str(HTML_HEAD_C);
    s.push_str("<script type=\"application/json\" id=\"data\">");
    s.push_str(&data);
    s.push_str("</script>\n");
    s.push_str(HTML_JS_BODY);
    s
}

const HTML_HEAD_A: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FactMachine multi-market MDP/POMDP simulation</title>
<style>
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:#f5f5f7;color:#171717}
header{padding:16px 22px 10px;background:#fff;border-bottom:1px solid #ddd}
h1{font-size:19px;margin:0 0 4px}.sub{color:#666;font-size:13px;margin:0}
main{padding:16px 22px;max-width:1280px;margin:auto}
.controls,.time-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
select,button{padding:6px 8px;border:1px solid #bbb;border-radius:4px;background:white;font:inherit}
button{cursor:pointer}.grid{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin:12px 0}
.metric{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px}.metric .k{font-size:12px;color:#666}.metric .v{font-size:20px;font-weight:650;margin-top:2px}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin:12px 0;overflow:auto}.panel h3{font-size:14px;margin:0 0 10px}
.plot-grid{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:12px}.summary-grid{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:10px}
.summary-card{border:1px solid #e2e2e2;border-radius:6px;padding:10px;background:#fafafa;line-height:1.35}.summary-card h4{margin:0 0 6px;font-size:14px}.summary-card p{margin:5px 0;font-size:12px;color:#333}.summary-card b{color:#111}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #eee;padding:7px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
svg{width:100%;height:auto;display:block;background:#fff}.note{font-size:12px;color:#555;line-height:1.45}.pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eee;margin-right:4px}
input[type=range]{min-width:260px;flex:1}.readout{font-family:SF Mono,Menlo,Consolas,monospace;color:#333;font-size:12px;min-width:360px}
@media(max-width:860px){main{padding:12px}.grid{grid-template-columns:repeat(2,minmax(120px,1fr))}.summary-grid,.plot-grid{grid-template-columns:1fr}.readout{min-width:100%}}
</style>
</head>
<body>
<header>
  <h1>FactMachine multi-market MDP/POMDP simulation</h1>
  <p class="sub">Multiple opinion markets open/close over "##;

const HTML_HEAD_B: &str = r##". Betting is gated by voting; scalar, binary, and over/under contracts are compared across 1k and 10k participant-scale scenarios.</p>
</header>
<main>
  <div class="panel intro-panel">
    <h3>What This Page Is Attempting To Do</h3>
    <div class="summary-grid">
      <section class="summary-card">
        <h4>Goal</h4>
        <p>This is an operator simulation for FactMachine-style opinion markets. It asks which topics to open, when to open them, how long to run them, and whether each market should resolve as majority binary, scalar vote distribution, or over/under threshold.</p>
        <p>The output is not a claim that one policy is universally best; it is a controlled comparison of launch policies under the same synthetic participation, fraud, liquidity, timing, and information assumptions.</p>
      </section>
      <section class="summary-card">
        <h4>How MDP Works</h4>
        <p>The MDP has "##;

const HTML_HEAD_C: &str = r##" action recipes covering contract type, duration, fees, liquidity, rewards, verification, and information visibility.</p>
        <p>Value iteration estimates the long-run value of each action and chooses the recipe with the best expected future reward.</p>
      </section>
      <section class="summary-card">
        <h4>How POMDP Works</h4>
        <p>The POMDP version does not get to see true topic hotness or ambiguity. It observes noisy buzz and ambiguity signals, maintains a belief over possible states, and chooses actions by averaging MDP Q-values under that belief.</p>
        <p>After markets close, the belief updates from realized engagement, fees, and trades. This makes the POMDP closer to what a live operator would actually know.</p>
      </section>
      <section class="summary-card">
        <h4>Opinion vs Prediction</h4>
        <p>Opinion markets resolve to the final participant vote distribution. Prediction-market accuracy is tracked as a counterfactual Brier score against a latent external event outcome, because public opinion can diverge from what later turns out to be true.</p>
        <p>The dashboard now separates opinion sampling error from prediction Brier score so a market can be good at measuring opinion while still being a poor forecast of external reality.</p>
      </section>
      <section class="summary-card">
        <h4>Shortcomings</h4>
        <p>The model is synthetic. It assumes simple parametric behavior for voters, bettors, manipulation, liquidity response, and information exposure. Real behavior may have regime changes, coordinated campaigns, platform shocks, and feedback loops not captured here.</p>
      </section>
      <section class="summary-card">
        <h4>Missing Wildcards</h4>
        <p>Important unknowns include acquisition channel quality, real identity verification failure rates, creator/influencer incentives, legal or moderation constraints, market maker capital limits, off-platform coordination, news shocks, bot adaptation, and whether users understand that opinion markets do not necessarily resolve to factual truth.</p>
      </section>
    </div>
  </div>
  <div class="controls">
    <label>scale <select id="scenario"></select></label>
    <label>policy <select id="policy"></select></label>
    <label>contract metric <select id="contractMetric"></select></label>
    <label>scale metric <select id="scaleMetric"></select></label>
    <span id="badges"></span>
  </div>
  <div class="grid" id="metrics"></div>
  <div class="panel">
    <h3>Time-Step Simulation</h3>
    <div class="time-controls">
      <button id="play">Play</button>
      <label>speed <select id="playbackSpeed"></select></label>
      <button id="stepBack">Step -</button>
      <button id="stepForward">Step +</button>
      <input id="timeScrub" type="range" min="0" value="0" step="1">
      <span class="readout" id="timeReadout"></span>
    </div>
    <svg id="statePlot" viewBox="0 0 1160 360" aria-label="time step simulation state"></svg>
  </div>
  <div class="plot-grid">
    <div class="panel"><h3>Plot 1: Daily Market Capacity vs Opens</h3><svg id="dailyPlot" viewBox="0 0 580 320" aria-label="daily market capacity"></svg></div>
    <div class="panel"><h3>Plot 2: Cumulative Votes, Bettors, Fees</h3><svg id="throughputPlot" viewBox="0 0 580 320" aria-label="cumulative throughput"></svg></div>
    <div class="panel"><h3>Plot 3: Binary vs Scalar Contract Variables</h3><svg id="contractPlot" viewBox="0 0 580 320" aria-label="contract comparison"></svg></div>
    <div class="panel"><h3>Plot 4: 1k vs 10k Scale Comparison</h3><svg id="scalePlot" viewBox="0 0 580 320" aria-label="scale comparison"></svg></div>
  </div>
  <div class="panel"><h3>Executive Summary</h3><div id="execSummary"></div></div>
  <div class="panel"><h3>1k vs 10k Scale Comparison</h3><div id="scaleComparison"></div></div>
  <div class="panel"><h3>Nonlinearity Diagnostics</h3><div id="nonlinear"></div></div>
  <div class="panel"><h3>Policy Comparison</h3><div id="comparison"></div></div>
  <div class="panel"><h3>Decision Levers Learned</h3><div id="levers"></div></div>
  <div class="panel"><h3>Agent Model Decomposition</h3><div id="agents"></div></div>
  <div class="panel"><h3>Selected Policy Market Mix</h3><div id="mix"></div></div>
  <div class="panel"><h3>MDP/POMDP Notes</h3><div class="note" id="notes"></div></div>
</main>
"##;

const HTML_JS_BODY: &str = r##"<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const runs=DATA.runs;
const fmt=n=>Number(n||0).toLocaleString(undefined,{maximumFractionDigits:1});
const pct=n=>(100*Number(n||0)).toFixed(1)+'%';
const money=n=>'$'+fmt(n);
const actionMeta=Object.fromEntries(DATA.mdp.actions.map(a=>[a.label,a]));
const policySel=document.getElementById('policy');
const scenarioSel=document.getElementById('scenario');
const contractMetricSel=document.getElementById('contractMetric');
const scaleMetricSel=document.getElementById('scaleMetric');
const scrub=document.getElementById('timeScrub');
const playBtn=document.getElementById('play');
const speedSel=document.getElementById('playbackSpeed');
const stepBackBtn=document.getElementById('stepBack');
const stepForwardBtn=document.getElementById('stepForward');
const policies=[...new Set(runs.map(r=>r.policy))];
const scenarios=[...new Set(runs.map(r=>r.scenarioLabel))];
const contractMetrics=[
 ['markets','markets'],['votes','votes'],['bettors','bettors'],['trades','trades'],['feeRevenue','fee revenue'],['platformSurplus','surplus'],['avgPredictionError','voter prediction error'],['avgOpinionSamplingError','opinion sampling error'],['avgPredictionBrierScore','prediction Brier score'],['herdingIndex','herding'],['fraudPressure','fraud']
];
const scaleMetrics=[
 ['votes','votes'],['bettors','bettors'],['trades','trades'],['feeRevenue','fee revenue'],['platformSurplus','surplus'],['avgOpinionSamplingError','opinion sampling error'],['avgPredictionBrierScore','prediction Brier score'],['fraudPressure','fraud'],['herdingIndex','herding'],['liquidityUtilization','liquidity use'],['whaleTradeShare','whale share']
];
for(const s of scenarios){const o=document.createElement('option');o.value=s;o.textContent=s;o.selected=s.includes('10,000');scenarioSel.appendChild(o);}
for(const p of policies){const o=document.createElement('option');o.value=p;o.textContent=p;o.selected=p==='pomdp-belief';policySel.appendChild(o);}
for(const m of contractMetrics){const o=document.createElement('option');o.value=m[0];o.textContent=m[1];contractMetricSel.appendChild(o);}
for(const m of scaleMetrics){const o=document.createElement('option');o.value=m[0];o.textContent=m[1];scaleMetricSel.appendChild(o);}
for(const speed of [0.5,1,2,3,5,10]){const o=document.createElement('option');o.value=String(speed);o.textContent=speed+'x';o.selected=speed===2;speedSel.appendChild(o);}
scaleMetricSel.value='votes';
let frameIndex=0;
let playTimer=null;
policySel.addEventListener('change',()=>{frameIndex=0;render();});
scenarioSel.addEventListener('change',()=>{frameIndex=0;render();});
contractMetricSel.addEventListener('change',render);
scaleMetricSel.addEventListener('change',render);
speedSel.addEventListener('change',()=>{if(playTimer) startPlayback(); render();});
scrub.addEventListener('input',()=>{frameIndex=Number(scrub.value);render();});
stepBackBtn.addEventListener('click',()=>{const r=selected();frameIndex=Math.max(0,frameIndex-1);scrub.value=String(frameIndex);render();});
stepForwardBtn.addEventListener('click',()=>{const r=selected();frameIndex=Math.min(r.timeline.length-1,frameIndex+1);scrub.value=String(frameIndex);render();});
playBtn.addEventListener('click',()=>{if(playTimer) stopPlayback(); else startPlayback();});
function playbackDelayMs(){return Math.max(16,Math.round(260/Number(speedSel.value||1)));}
function advanceFrame(){const r=selected();frameIndex=(frameIndex+1)%Math.max(1,r.timeline.length);scrub.value=String(frameIndex);render();}
function stopPlayback(){if(playTimer) clearInterval(playTimer);playTimer=null;playBtn.textContent='Play';}
function startPlayback(){stopPlayback();playBtn.textContent='Pause';playTimer=setInterval(advanceFrame,playbackDelayMs());}
function currentRuns(){return runs.filter(r=>r.scenarioLabel===scenarioSel.value);}
function selected(){return runs.find(r=>r.policy===policySel.value&&r.scenarioLabel===scenarioSel.value)||currentRuns()[0]||runs[0];}
function render(){
 const r=selected(), a=r.aggregate;
 if(frameIndex>=r.timeline.length) frameIndex=Math.max(0,r.timeline.length-1);
 scrub.max=String(Math.max(0,r.timeline.length-1));
 scrub.value=String(frameIndex);
 const f=r.timeline[frameIndex]||r.timeline[0]||{t:0,day:0,open:0,closed:0,queued:0,votes:0,bettors:0,trades:0,fees:0,marketCap:0,openedToday:0,openedTotal:0};
 document.getElementById('badges').innerHTML='<span class="pill">'+r.scenarioLabel+'</span><span class="pill">'+a.marketsClosed+' closed markets</span><span class="pill">'+a.binaryMarkets+' binary</span><span class="pill">'+a.scalarMarkets+' scalar</span><span class="pill">'+a.thresholdMarkets+' over/under</span><span class="pill">day '+(f.day+1)+'</span>';
 document.getElementById('metrics').innerHTML=[
  ['Votes',fmt(a.votes)],['Bettors',fmt(a.bettors)],['Trades',fmt(a.trades)],['Fee revenue','$'+fmt(a.feeRevenue)],
  ['Voter points',fmt(a.voterPoints)],['Raffle entries',fmt(a.raffleEntries)],['Platform surplus','$'+fmt(a.platformSurplus)],['Avg pred error',pct(a.avgPredictionError)],
  ['Opinion sampling err',pct(a.avgOpinionSamplingError)],['Prediction Brier',fmt(a.avgPredictionBrierScore)],['Opinion/price gap',pct(a.priceOpinionGap)],['External forecast miss',fmt(Math.sqrt(a.avgPredictionBrierScore||0))],
  ['Avg fee',pct(a.avgFeeRate)],['Avg liquidity','$'+fmt(a.avgLiquidity)],['Avg reward',fmt(a.avgRewardMultiplier)+'x'],['Suspect votes',fmt(a.suspectedSybilVotes)],
  ['Avg vote time',pct(a.avgVoteTimeFraction)],['Timing boost',fmt(a.avgTimingMultiplier)+'x'],['Herding index',pct(a.herdingIndex)],['Fraud pressure',pct(a.fraudPressure)],
  ['Trader belief err',pct(a.avgTraderBeliefError)],['MM risk bound','$'+fmt(a.marketMakerRiskBound)],['Timing decay',fmt(a.avgTimingDecay)],
  ['Referrals',fmt(a.referralAdds)],['Churn risk',pct(a.churnRisk)],['Reward inflation',pct(a.rewardInflationPressure)],['Liq utilization',fmt(a.liquidityUtilization)+'x'],
  ['Whale share',pct(a.whaleTradeShare)],['News intensity',pct(a.avgNewsCycleIntensity)],['Social virality',pct(a.avgSocialVirality)],['Demo polarization',pct(a.avgDemographicPolarization)]
 ].map(([k,v])=>'<div class="metric"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');
 drawStatePlot(r,f);
 drawDailyPlot(r);
 drawThroughputPlot(r);
 drawContractPlot(r);
 drawScalePlot();
 drawExecutiveSummary();
 drawScaleComparison();
 drawNonlinearityDiagnostics();
 drawComparison();
 drawLevers(r);
 drawAgents(r);
 drawMix(r);
 drawNotes(r);
}
function drawStatePlot(r,f){
 const svg=document.getElementById('statePlot'); clear(svg);
 const W=1160,H=360,pad=42;
 rect(svg,0,0,W,H,'#fbfbfc','#ddd');
 text(svg,24,30,'day '+(f.day+1)+' / t='+f.t.toFixed(2)+'h','#111',16,'bold');
 text(svg,24,54,'cap '+f.marketCap+', opened today '+f.openedToday+', active '+f.open+', queued '+f.queued+', closed '+f.closed,'#555',12);
 document.getElementById('timeReadout').textContent='step '+frameIndex+' of '+Math.max(0,r.timeline.length-1)+' | day '+(f.day+1)+' | votes '+fmt(f.votes)+' | bettors '+fmt(f.bettors)+' | trades '+fmt(f.trades)+' | fees '+money(f.fees);
 const closed=marketsUntil(r,f.t);
 const kinds=['binary','scalar','threshold'];
 const kindColors={binary:'#2563eb',scalar:'#7c3aed',threshold:'#e11d48'};
 const kindCounts=Object.fromEntries(kinds.map(k=>[k,closed.filter(m=>m.kind===k).length]));
 const totalKind=Math.max(1,kinds.reduce((s,k)=>s+kindCounts[k],0));
 let x=24;
 for(const k of kinds){
   const w=280*(kindCounts[k]/totalKind);
   rect(svg,x,82,w,24,kindColors[k],null,0.88);
   text(svg,x+5,99,k+' '+kindCounts[k],'#fff',11,'bold');
   x+=w;
 }
 const bars=[
  ['active',f.open,Math.max(1,DATA.cfg.maxConcurrent),'#0f766e'],
  ['queued',f.queued,Math.max(1,...r.timeline.map(x=>x.queued)),'#f59e0b'],
  ['closed',f.closed,Math.max(1,r.aggregate.marketsClosed),'#334155'],
  ['votes',f.votes,Math.max(1,r.aggregate.votes),'#2563eb'],
  ['bettors',f.bettors,Math.max(1,r.aggregate.bettors),'#7c3aed'],
  ['fees',f.fees,Math.max(1,r.aggregate.feeRevenue),'#16a34a']
 ];
 const barX=24,barY=145,barW=106,barGap=22,barH=170;
 for(let i=0;i<bars.length;i++){
  const b=bars[i], h=barH*(b[1]/b[2]);
  rect(svg,barX+i*(barW+barGap),barY+barH-h,barW,h,b[3],null,0.9);
  rect(svg,barX+i*(barW+barGap),barY,barW,barH,'none','#ddd');
  text(svg,barX+i*(barW+barGap)+barW/2,barY+barH+20,b[0],'#444',12,'normal','middle');
  text(svg,barX+i*(barW+barGap)+barW/2,barY+barH-h-8,formatMetric(b[0]==='fees'?'feeRevenue':b[0],b[1]),b[3],12,'bold','middle');
 }
 const recent=closed.slice(-10).reverse();
 const listX=850;
 text(svg,listX,82,'latest closed markets','#111',13,'bold');
 for(let i=0;i<recent.length;i++){
  const m=recent[i], y=108+i*22;
  rect(svg,listX,y-13,10,10,kindColors[m.kind],null);
  text(svg,listX+18,y,'#'+m.id+' '+m.kind+' '+m.topic.category+' votes '+fmt(m.votes)+' fees '+money(m.feeRevenue),'#333',11);
 }
}
function drawDailyPlot(r){
 const svg=document.getElementById('dailyPlot'); clear(svg);
 const W=580,H=320,pad=38,days=r.daily.filter(d=>d.day<Math.ceil(DATA.cfg.horizonH/24));
 const maxY=Math.max(1,...days.map(d=>Math.max(d.marketCap,d.opened,d.binaryClosed+d.scalarClosed+d.thresholdClosed)));
 axes(svg,W,H,pad,'days','markets');
 const bw=(W-pad*1.6)/Math.max(1,days.length);
 for(const d of days){
  const x=pad+d.day*bw, y=scaleY(d.opened,maxY,H,pad);
  rect(svg,x+1,y,Math.max(1,bw-2),H-pad-y,'#38bdf8',null,0.55);
 }
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.marketCap,maxY,H,pad)]),'#0f172a',2);
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.scalarClosed,maxY,H,pad)]),'#7c3aed',2);
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.binaryClosed,maxY,H,pad)]),'#2563eb',2);
 const currentDay=(r.timeline[frameIndex]||{}).day||0;
 line(svg,pad+(currentDay+0.5)*bw,pad,pad+(currentDay+0.5)*bw,H-pad,'#ef4444');
 legend(svg,48,20,[['opened','#38bdf8'],['cap','#0f172a'],['scalar closed','#7c3aed'],['binary closed','#2563eb']]);
}
function drawThroughputPlot(r){
 const svg=document.getElementById('throughputPlot'); clear(svg);
 const W=580,H=320,pad=38,tl=r.timeline;
 const maxT=Math.max(1,...tl.map(x=>x.t)), maxY=Math.max(1,r.aggregate.votes,r.aggregate.bettors,r.aggregate.feeRevenue);
 axes(svg,W,H,pad,'hours','cumulative');
 const pts=(field,scale)=>tl.map(x=>[scaleX(x.t,maxT,W,pad),scaleY(x[field]*scale,maxY,H,pad)]);
 path(svg,pts('votes',1),'#2563eb',2);
 path(svg,pts('bettors',1),'#7c3aed',2);
 path(svg,pts('fees',1),'#16a34a',2);
 const f=tl[frameIndex]||tl[0];
 if(f) line(svg,scaleX(f.t,maxT,W,pad),pad,scaleX(f.t,maxT,W,pad),H-pad,'#ef4444');
 legend(svg,48,20,[['votes','#2563eb'],['bettors','#7c3aed'],['fee revenue','#16a34a']]);
}
function drawContractPlot(r){
 const svg=document.getElementById('contractPlot'); clear(svg);
 const W=580,H=320,pad=48,metric=contractMetricSel.value;
 const rows=r.kindBreakdown.filter(x=>x.kind!=='threshold'||x.markets>0);
 const maxY=Math.max(1,...rows.map(x=>Math.abs(x[metric]||0)));
 axes(svg,W,H,pad,'contract','value');
 const colors={binary:'#2563eb',scalar:'#7c3aed',threshold:'#e11d48'};
 const bw=(W-pad*2)/Math.max(1,rows.length)/1.6;
 for(let i=0;i<rows.length;i++){
  const row=rows[i], v=row[metric]||0, x=pad+30+i*((W-pad*2)/Math.max(1,rows.length)), h=(H-pad*2)*(Math.abs(v)/maxY);
  rect(svg,x,H-pad-h,bw,h,colors[row.kind],null,0.9);
  text(svg,x+bw/2,H-pad+18,row.kind,'#333',11,'normal','middle');
  text(svg,x+bw/2,H-pad-h-8,formatMetric(metric,v),colors[row.kind],11,'bold','middle');
 }
 text(svg,48,22,'selected metric: '+metricLabel(metric),'#111',12,'bold');
}
function drawScalePlot(){
 const svg=document.getElementById('scalePlot'); clear(svg);
 const W=580,H=320,pad=48,metric=scaleMetricSel.value;
 const byPolicy=policies.map(p=>runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants));
 const vals=[];
 for(const pair of byPolicy){for(const r of pair) vals.push(Math.abs(r.aggregate[metric]||0));}
 const maxY=Math.max(1,...vals);
 axes(svg,W,H,pad,'policy','value');
 const groupW=(W-pad*2)/Math.max(1,policies.length), bw=groupW*0.28;
 for(let i=0;i<byPolicy.length;i++){
  const pair=byPolicy[i];
  for(let j=0;j<pair.length;j++){
   const r=pair[j], v=r.aggregate[metric]||0, x=pad+i*groupW+groupW*0.22+j*bw*1.2, h=(H-pad*2)*(Math.abs(v)/maxY);
   rect(svg,x,H-pad-h,bw,h,j===0?'#60a5fa':'#1d4ed8',null,0.9);
   text(svg,x+bw/2,H-pad-h-8,formatMetric(metric,v),j===0?'#2563eb':'#1e3a8a',10,'bold','middle');
  }
  text(svg,pad+i*groupW+groupW/2,H-pad+18,policies[i].replace('-',' '),'#333',10,'normal','middle');
 }
 legend(svg,48,20,[[scenarios[0]||'low scale','#60a5fa'],[scenarios[scenarios.length-1]||'high scale','#1d4ed8']]);
}
function drawExecutiveSummary(){
 document.getElementById('execSummary').innerHTML='<div class="summary-grid">'+
 currentRuns().map(r=>'<section class="summary-card"><h4>'+r.policy+'</h4><p>'+policyRead(r)+'</p><p><b>Recipe:</b> '+recipeRead(r)+'</p><p><b>Tradeoff:</b> '+tradeoffRead(r)+'</p><p><b>Optimality:</b> '+optimalityRead(r)+'</p></section>').join('')+
 '</div>';
}
function drawScaleComparison(){
 const rows=[];
 for(const p of policies){
  const byScale=runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants);
  if(byScale.length<2) continue;
  const low=byScale[0].aggregate, high=byScale[byScale.length-1].aggregate;
  rows.push({policy:p,low,high,lowLabel:byScale[0].scenarioLabel,highLabel:byScale[byScale.length-1].scenarioLabel});
 }
 document.getElementById('scaleComparison').innerHTML='<table><thead><tr><th>policy</th><th>scale move</th><th>votes</th><th>bettors</th><th>fees</th><th>surplus</th><th>opinion err</th><th>prediction Brier</th><th>fraud</th><th>herding</th><th>liquidity use</th><th>whale share</th><th>executive read</th></tr></thead><tbody>'+
 rows.map(x=>'<tr><td>'+x.policy+'</td><td>'+x.lowLabel+' -> '+x.highLabel+'</td><td>'+fmt(x.low.votes)+' -> '+fmt(x.high.votes)+'</td><td>'+fmt(x.low.bettors)+' -> '+fmt(x.high.bettors)+'</td><td>$'+fmt(x.low.feeRevenue)+' -> $'+fmt(x.high.feeRevenue)+'</td><td>$'+fmt(x.low.platformSurplus)+' -> $'+fmt(x.high.platformSurplus)+'</td><td>'+pct(x.low.avgOpinionSamplingError)+' -> '+pct(x.high.avgOpinionSamplingError)+'</td><td>'+fmt(x.low.avgPredictionBrierScore)+' -> '+fmt(x.high.avgPredictionBrierScore)+'</td><td>'+pct(x.low.fraudPressure)+' -> '+pct(x.high.fraudPressure)+'</td><td>'+pct(x.low.herdingIndex)+' -> '+pct(x.high.herdingIndex)+'</td><td>'+fmt(x.low.liquidityUtilization)+'x -> '+fmt(x.high.liquidityUtilization)+'x</td><td>'+pct(x.low.whaleTradeShare)+' -> '+pct(x.high.whaleTradeShare)+'</td><td style="text-align:left">'+scaleRead(x.low,x.high)+'</td></tr>').join('')+
 '</tbody></table>';
}
function drawNonlinearityDiagnostics(){
 const policy=DATA.mdp.fullPolicy||[];
 const byKey=Object.fromEntries(policy.map(x=>[x.hotBin+'-'+x.ambBin+'-'+x.fatigueBin,x]));
 const switches=[];
 for(const s of policy){
  for(const axis of ['hotBin','ambBin','fatigueBin']){
   const n={hotBin:s.hotBin,ambBin:s.ambBin,fatigueBin:s.fatigueBin};
   n[axis]+=1;
   if(n[axis]>2) continue;
   const other=byKey[n.hotBin+'-'+n.ambBin+'-'+n.fatigueBin];
   if(other&&other.action!==s.action){
    switches.push({from:s,to:other,axis,gap:Math.min(s.qGap,other.qGap)});
   }
  }
 }
 switches.sort((a,b)=>a.gap-b.gap);
 const fragile=[...policy].sort((a,b)=>a.qGap-b.qGap).slice(0,6);
 const scaleRows=[];
 for(const p of policies){
  const rs=runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants);
  if(rs.length<2) continue;
  const low=rs[0].aggregate, high=rs[rs.length-1].aggregate;
  scaleRows.push({
   policy:p,
   fees:elasticity(low.feeRevenue,high.feeRevenue,rs[0].minMarketParticipants,rs[rs.length-1].minMarketParticipants),
   surplus:elasticity(Math.max(1,Math.abs(low.platformSurplus)),Math.max(1,Math.abs(high.platformSurplus)),rs[0].minMarketParticipants,rs[rs.length-1].minMarketParticipants),
   fraud:high.fraudPressure-low.fraudPressure,
   liq:high.liquidityUtilization-low.liquidityUtilization,
  });
 }
 document.getElementById('nonlinear').innerHTML=
  '<div class="summary-grid">'+
  '<section class="summary-card"><h4>Policy Switch Boundaries</h4><p>Adjacent MDP states where one-bin changes flip the operator action. These are discovered nonlinear thresholds in the learned value surface.</p><p>'+switches.slice(0,6).map(x=>x.axis+': '+stateShort(x.from)+' '+x.from.action+' -> '+stateShort(x.to)+' '+x.to.action+' (Q gap '+fmt(x.gap)+')').join('<br>')+'</p></section>'+
  '<section class="summary-card"><h4>Fragile Optima</h4><p>Small Q gaps mean small parameter changes can change the optimal policy.</p><p>'+fragile.map(x=>stateShort(x)+' -> '+x.action+' (Q gap '+fmt(x.qGap)+')').join('<br>')+'</p></section>'+
  '<section class="summary-card"><h4>Scale Elasticities</h4><p>Elasticity > 1 means superlinear scale effects; < 1 means sublinear.</p><p>'+scaleRows.map(x=>x.policy+': fees '+fmt(x.fees)+'x, surplus magnitude '+fmt(x.surplus)+'x, fraud \u0394 '+pct(x.fraud)+', liquidity-use \u0394 '+fmt(x.liq)+'x').join('<br>')+'</p></section>'+
  '<section class="summary-card"><h4>Where The POMDP Helps</h4><p>The POMDP discovers nonlinearities by tracking belief entropy/error and testing whether noisy observations cross an operator action boundary. The most important discoveries here are over/under becoming optimal after belief mass moves into hot/debatable states, and proof/delayed-vote recipes reducing fraud/herding enough to dominate at scale.</p></section>'+
  '</div>';
}
function drawComparison(){
 const sorted=[...currentRuns()].sort((a,b)=>b.aggregate.engagementScore-a.aggregate.engagementScore);
 document.getElementById('comparison').innerHTML='<table><thead><tr><th>policy</th><th>markets</th><th>binary</th><th>scalar</th><th>o/u</th><th>avg fee</th><th>votes</th><th>bettors</th><th>fees</th><th>surplus</th><th>engagement</th><th>opinion err</th><th>prediction Brier</th><th>vote time</th><th>herding</th><th>fraud</th><th>operator belief error</th></tr></thead><tbody>'+
 sorted.map(r=>{const a=r.aggregate;return '<tr><td>'+r.policy+'</td><td>'+a.marketsClosed+'</td><td>'+a.binaryMarkets+'</td><td>'+a.scalarMarkets+'</td><td>'+a.thresholdMarkets+'</td><td>'+pct(a.avgFeeRate)+'</td><td>'+fmt(a.votes)+'</td><td>'+fmt(a.bettors)+'</td><td>$'+fmt(a.feeRevenue)+'</td><td>$'+fmt(a.platformSurplus)+'</td><td>'+fmt(a.engagementScore)+'</td><td>'+pct(a.avgOpinionSamplingError)+'</td><td>'+fmt(a.avgPredictionBrierScore)+'</td><td>'+pct(a.avgVoteTimeFraction)+'</td><td>'+pct(a.herdingIndex)+'</td><td>'+pct(a.fraudPressure)+'</td><td>'+(a.avgBeliefError===undefined?'':pct(a.avgBeliefError))+'</td></tr>'}).join('')+
 '</tbody></table>';
}
function policyRead(r){
 const a=r.aggregate;
 if(r.policy==='fixed-daily') return 'Conservative launch baseline: simple to operate, easy to explain, and useful as the reference floor.';
 if(r.policy==='greedy-buzz') return 'Exploration and traffic-maximizing heuristic: chases observed buzz and tests many market designs.';
 if(r.policy==='mdp-oracle') return 'Upper-bound operator benchmark: solves the MDP as if latent topic quality were directly visible.';
 if(r.policy==='pomdp-belief') return 'Deployable learned operator: chooses actions from noisy buzz/ambiguity beliefs and updates after outcomes.';
 return 'Policy run with '+fmt(a.marketsClosed)+' closed markets.';
}
function recipeRead(r){
 const opened=r.actionCounts.filter(x=>x.action!=='wait');
 const top=opened[0];
 if(!top) return 'Mostly waits; no stable market-opening recipe emerged.';
 const meta=actionMeta[top.action]||{};
 const mix=opened.slice(0,3).map(x=>x.action+' x'+x.count).join(', ');
 return 'Top recipe: '+top.action+' ('+contractLabelJS(meta)+', '+(meta.durationH||0)+'h, '+pct(meta.feeRate)+', '+(meta.informationMode||'no info mode')+'). Mix: '+mix+'.';
}
function tradeoffRead(r){
 const a=r.aggregate;
 const engagementRank=rankOf(r,'engagementScore',true);
 const surplusRank=rankOf(r,'platformSurplus',true);
 const fraudRank=rankOf(r,'fraudPressure',false);
 const herdRank=rankOf(r,'herdingIndex',false);
 const voterCost=a.votes>0?a.voterPoints/a.votes:0;
 return 'Ranks #'+engagementRank+' engagement, #'+surplusRank+' surplus, #'+fraudRank+' fraud pressure, #'+herdRank+' herding. Reward cost is '+fmt(voterCost)+' points/voter; avg vote time is '+pct(a.avgVoteTimeFraction)+'.';
}
function optimalityRead(r){
 const a=r.aggregate;
 const tags=[];
 if(isBest(r,'engagementScore',true)) tags.push('best engagement/traffic');
 if(isBest(r,'feeRevenue',true)) tags.push('best gross fee revenue');
 if(isBest(r,'platformSurplus',true)) tags.push('best surplus control');
 if(isBest(r,'fraudPressure',false)) tags.push('lowest fraud pressure');
 if(isBest(r,'herdingIndex',false)) tags.push('lowest herding');
 if(isBest(r,'avgOpinionSamplingError',false)) tags.push('best opinion representativeness');
 if(isBest(r,'avgPredictionBrierScore',false)) tags.push('best counterfactual prediction score');
 if(isBest(r,'avgTraderBeliefError',false)) tags.push('best trader belief accuracy');
 if(r.aggregate.avgBeliefError!==undefined && isBestDefined(r,'avgBeliefError',false)) tags.push('best deployable operator belief read');
 if(tags.length===0) tags.push('benchmark or middle-ground comparison point');
 return tags.join('; ')+'.';
}
function rankOf(run,metric,higher){
 const sorted=[...currentRuns()].sort((x,y)=>higher?y.aggregate[metric]-x.aggregate[metric]:x.aggregate[metric]-y.aggregate[metric]);
 return sorted.findIndex(x=>x.policy===run.policy)+1;
}
function isBest(run,metric,higher){
 return rankOf(run,metric,higher)===1;
}
function isBestDefined(run,metric,higher){
 const filtered=currentRuns().filter(x=>x.aggregate[metric]!==undefined);
 const sorted=filtered.sort((x,y)=>higher?y.aggregate[metric]-x.aggregate[metric]:x.aggregate[metric]-y.aggregate[metric]);
 return sorted[0]&&sorted[0].policy===run.policy;
}
function scaleRead(low,high){
 const feeMultiple=high.feeRevenue/Math.max(1,low.feeRevenue);
 const fraudDelta=high.fraudPressure-low.fraudPressure;
 const liqDelta=high.liquidityUtilization-low.liquidityUtilization;
 const parts=['fees scale '+fmt(feeMultiple)+'x'];
 parts.push(fraudDelta>0.002?'fraud pressure rises':fraudDelta<-0.002?'fraud pressure improves':'fraud pressure is stable');
 parts.push(liqDelta>1?'liquidity gets stressed':liqDelta<-1?'liquidity headroom improves':'liquidity load is stable');
 if(high.avgOpinionSamplingError<low.avgOpinionSamplingError-0.002) parts.push('opinion sampling improves');
 if(high.avgPredictionBrierScore>low.avgPredictionBrierScore+0.02) parts.push('external prediction score worsens');
 if(high.whaleTradeShare>low.whaleTradeShare+0.02) parts.push('whale concentration increases');
 if(high.referralAdds>low.referralAdds*5) parts.push('referral loop becomes material');
 return parts.join('; ')+'.';
}
function elasticity(low,high,lowScale,highScale){
 if(low<=0||high<=0||lowScale<=0||highScale<=0||lowScale===highScale) return 0;
 return Math.log(high/low)/Math.log(highScale/lowScale);
}
function stateShort(s){return 'h'+s.hotBin+'/a'+s.ambBin+'/f'+s.fatigueBin;}
function drawLevers(r){
 const rows=r.actionCounts.slice(0,12);
 document.getElementById('levers').innerHTML='<table><thead><tr><th>action</th><th>count</th><th>contract</th><th>duration</th><th>fee</th><th>liquidity</th><th>rewards</th><th>verification</th><th>info</th><th>decay</th><th>description</th></tr></thead><tbody>'+
 rows.map(x=>{const a=actionMeta[x.action]||{};return '<tr><td>'+x.action+'</td><td>'+x.count+'</td><td>'+contractLabelJS(a)+'</td><td>'+(a.durationH||0)+'h</td><td>'+pct(a.feeRate)+'</td><td>'+fmt((DATA.cfg.liquidity||0)*(a.liquidityMultiplier||1))+'</td><td>'+fmt(a.rewardMultiplier||1)+'x</td><td>'+(a.verification||'')+'</td><td>'+(a.informationMode||'')+'</td><td>'+fmt(a.timingDecay||0)+'</td><td style="text-align:left">'+(a.description||'')+'</td></tr>'}).join('')+
 '</tbody></table>';
}
function drawAgents(r){
 const a=r.aggregate;
 const rows=[
  ['Platform operator','MDP/POMDP policy learner','Chooses topic timing, contract, duration, fee, liquidity, rewards, verification, and information reveal policy.','engagement '+fmt(a.engagementScore)+', surplus $'+fmt(a.platformSurplus)],
  ['Voters','Partial-observation timing/accuracy model','Must vote before betting; trade off early exponential multiplier against more public information later.','avg vote time '+pct(a.avgVoteTimeFraction)+', opinion sampling error '+pct(a.avgOpinionSamplingError)],
  ['Bettors','Partial-observation trader model','Observe noisy public/private signals, prices, and information mode; choose whether/what to buy after voting. Some fraction mistakenly trades as though the opinion market were an external prediction market.','bettors '+fmt(a.bettors)+', prediction Brier '+fmt(a.avgPredictionBrierScore)+', herding '+pct(a.herdingIndex)],
  ['Market maker','LMSR liquidity/risk model','Quotes binary, scalar, and over/under contracts with configurable b and bounded loss.','risk bound $'+fmt(a.marketMakerRiskBound)+', price gap '+pct(a.priceOpinionGap)],
  ['Trust & safety','Fraud-pressure response model','Verification tier reduces suspected Sybil vote pressure but also lowers participation.','suspect votes '+fmt(a.suspectedSybilVotes)+', fraud pressure '+pct(a.fraudPressure)]
 ];
 document.getElementById('agents').innerHTML='<table><thead><tr><th>layer</th><th>model</th><th>role in operator transition/reward</th><th>current selected policy signal</th></tr></thead><tbody>'+
 rows.map(row=>'<tr><td>'+row[0]+'</td><td>'+row[1]+'</td><td style="text-align:left">'+row[2]+'</td><td style="text-align:left">'+row[3]+'</td></tr>').join('')+
 '</tbody></table>';
}
function drawMix(r){
 const markets=[...r.closedMarkets].slice(-16).reverse();
 document.getElementById('mix').innerHTML='<table><thead><tr><th>market</th><th>cat</th><th>contract</th><th>dur</th><th>info</th><th>fee</th><th>liq</th><th>reward</th><th>verify</th><th>votes</th><th>bettors</th><th>vote time</th><th>herding</th><th>suspect</th><th>fees</th><th>opinion \u03b8</th><th>final vote</th><th>event p</th><th>event</th><th>opinion err</th><th>Brier</th></tr></thead><tbody>'+
 markets.map(m=>'<tr><td>#'+m.id+'</td><td>'+m.topic.category+'</td><td>'+m.contractLabel+'</td><td>'+m.durationH+'h</td><td>'+m.informationMode+'</td><td>'+pct(m.feeRate)+'</td><td>'+fmt(m.liquidity)+'</td><td>'+fmt(m.rewardMultiplier)+'x</td><td>'+m.verification+'</td><td>'+m.votes+'</td><td>'+m.bettors+'</td><td>'+pct(m.avgVoteTimeFraction)+'</td><td>'+pct(m.herdingIndex)+'</td><td>'+m.suspectedSybilVotes+'</td><td>$'+fmt(m.feeRevenue)+'</td><td>'+(100*m.topic.trueTheta).toFixed(0)+'%</td><td>'+(100*m.finalVoteFraction).toFixed(0)+'%</td><td>'+(100*m.topic.eventProbability).toFixed(0)+'%</td><td>'+(m.externalOutcome?'yes':'no')+'</td><td>'+pct(m.opinionSamplingError)+'</td><td>'+fmt(m.predictionBrierScore)+'</td></tr>').join('')+
 '</tbody></table>';
}
function drawNotes(r){
 const topActions=r.actionCounts.slice(0,5).map(x=>x.action+'='+x.count).join(', ');
 let s='Operator MDP: '+DATA.mdp.numStates+' states, '+DATA.mdp.actions.length+' action recipes, gamma='+DATA.mdp.gamma+'. Top scheduler actions for this policy: '+topActions+'. ';
 s+='Daily launch capacity is explicit in the state trace: min '+DATA.cfg.minDailyMarkets+', max '+DATA.cfg.maxDailyMarkets+', horizon '+(DATA.cfg.horizonH/24)+' days. ';
 s+='Each action recipe combines open/wait scheduling, contract type, duration, fee rate, LMSR liquidity, reward multiplier, verification tier, information visibility, and exponential timing decay. ';
 s+='Bettor/voter motives are not the top-level learned policy here; they are behavioral submodels that generate transition and reward signals for the operator policy. ';
 s+='Opinion accuracy is measured as sampling error against latent public opinion; prediction accuracy is measured as a counterfactual Brier score against a latent external event outcome. ';
 if(r.policy==='mdp-oracle') s+='The MDP oracle sees latent hotness and ambiguity before opening a market; this is a useful upper-bound, not a deployable assumption.';
 if(r.policy==='pomdp-belief') s+='The POMDP scheduler sees noisy buzz/ambiguity, maintains category-level beliefs, and updates them after market resolution. Its belief error/entropy are reported in the comparison table.';
 if(r.policy==='fixed-daily') s+='Fixed daily approximates the initial one-market-every-24h operating mode.';
 if(r.policy==='greedy-buzz') s+='Greedy buzz is a transparent heuristic baseline: open high observed-buzz topics and choose scalar when ambiguity looks high.';
 document.getElementById('notes').textContent=s;
}
function marketsUntil(r,t){return r.closedMarkets.filter(m=>m.closeAt<=t+1e-9)}
function metricLabel(metric){const all=contractMetrics.concat(scaleMetrics);const row=all.find(x=>x[0]===metric);return row?row[1]:metric}
function formatMetric(metric,value){if(['avgPredictionError','avgOpinionSamplingError','herdingIndex','fraudPressure','avgTraderBeliefError','whaleTradeShare'].includes(metric))return pct(value);if(['feeRevenue','platformSurplus','fees'].includes(metric))return money(value);return fmt(value)}
function clear(svg){while(svg.firstChild)svg.removeChild(svg.firstChild)}
function axes(svg,W,H,pad,xLabel,yLabel){line(svg,pad,H-pad,W-pad,H-pad,'#aaa');line(svg,pad,pad,pad,H-pad,'#aaa');text(svg,W-pad,H-10,xLabel,'#666',10,'normal','end');text(svg,10,pad,yLabel,'#666',10)}
function scaleX(x,max,W,pad){return pad+(W-pad*2)*(x/Math.max(1,max))}
function scaleY(y,max,H,pad){return H-pad-(H-pad*2)*(y/Math.max(1,max))}
function legend(svg,x,y,items){let dx=0;for(const it of items){rect(svg,x+dx,y-10,10,10,it[1],null);text(svg,x+dx+14,y,it[0],'#444',11);dx+=it[0].length*7+42}}
function line(svg,x1,y1,x2,y2,c){const e=document.createElementNS('http://www.w3.org/2000/svg','line');e.setAttribute('x1',String(x1));e.setAttribute('y1',String(y1));e.setAttribute('x2',String(x2));e.setAttribute('y2',String(y2));e.setAttribute('stroke',c);e.setAttribute('stroke-width','1.5');svg.appendChild(e)}
function rect(svg,x,y,w,h,fill,stroke,opacity){const e=document.createElementNS('http://www.w3.org/2000/svg','rect');e.setAttribute('x',String(x));e.setAttribute('y',String(y));e.setAttribute('width',String(Math.max(0,w)));e.setAttribute('height',String(Math.max(0,h)));e.setAttribute('fill',fill||'none');if(stroke)e.setAttribute('stroke',stroke);if(opacity!==undefined)e.setAttribute('opacity',String(opacity));svg.appendChild(e)}
function path(svg,pts,c,width){if(pts.length===0)return;const e=document.createElementNS('http://www.w3.org/2000/svg','path');e.setAttribute('d',pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' '));e.setAttribute('fill','none');e.setAttribute('stroke',c);e.setAttribute('stroke-width',String(width||2));svg.appendChild(e)}
function text(svg,x,y,s,c,size,weight,anchor){const e=document.createElementNS('http://www.w3.org/2000/svg','text');e.setAttribute('x',String(x));e.setAttribute('y',String(y));e.setAttribute('font-size',String(size||12));e.setAttribute('fill',c);if(weight)e.setAttribute('font-weight',weight);if(anchor)e.setAttribute('text-anchor',anchor);e.textContent=s;svg.appendChild(e)}
function contractLabelJS(a){if(a.kind==='threshold')return 'over/under '+Math.round((a.threshold||0.55)*100)+'%';if(a.kind==='scalar')return 'scalar distribution';if(a.kind==='binary')return 'majority binary';return 'wait'}
render();
</script>
</body>
</html>"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_state_encode_decode_roundtrips() {
        for s in 0..27 {
            let (hot, amb, fat) = decode_operator_state(s);
            assert_eq!(encode_operator_state(hot, amb, fat), s);
        }
    }

    #[test]
    fn build_daily_market_caps_endpoints() {
        let caps = build_daily_market_caps(10.0, 2, 10, 42);
        assert_eq!(caps.len(), 10);
        assert_eq!(caps[0], 2);
        assert_eq!(caps[(10.0_f64 * 0.62).floor() as usize], 10);
        for &c in &caps {
            assert!((2..=10).contains(&c));
        }
    }

    #[test]
    fn build_daily_market_caps_flat_when_equal() {
        let caps = build_daily_market_caps(5.0, 4, 4, 7);
        assert_eq!(caps, vec![4, 4, 4, 4, 4]);
    }

    #[test]
    fn day_index_floors_hours() {
        assert_eq!(day_index(0.0), 0);
        assert_eq!(day_index(23.9), 0);
        assert_eq!(day_index(24.0), 1);
        assert_eq!(day_index(49.0), 2);
        assert_eq!(day_index(-5.0), 0);
    }

    #[test]
    fn locale_int_groups_thousands() {
        assert_eq!(locale_int(1000.0), "1,000");
        assert_eq!(locale_int(10000.0), "10,000");
        assert_eq!(locale_int(999.0), "999");
    }

    #[test]
    fn operator_mdp_is_solvable_and_runs() {
        let mdp = build_operator_mdp();
        assert_eq!(mdp.spec.num_states, 27);
        assert_eq!(mdp.actions.len(), 12);
        assert_eq!(mdp.q.len(), 27);
        let mut cfg = default_config();
        cfg.horizon_h = 48.0;
        cfg.min_market_participants = 50.0;
        cfg.daily_market_caps =
            build_daily_market_caps(2.0, cfg.min_daily_markets, cfg.max_daily_markets, cfg.seed);
        let run = run_portfolio(SchedulerPolicy::PomdpBelief, &cfg, &mdp);
        assert!(!run.timeline.is_empty());
        assert!(run.belief_trace.is_some());
    }
}
