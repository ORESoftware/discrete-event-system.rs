//! Port of `src/des/general/collaborative-inference.ts` — sparse subjective
//! preference learning (ratings + pairwise comparisons to a global ranking)
//! modelled as a discrete-event station graph.
//!
//! A population knows only small overlapping subsets of a large option set.
//! Each respondent rates and ranks the subset they have experienced. The
//! station graph converts those local opinions into rating evidence and
//! pairwise preference evidence, then infers a global ranking with
//! empirical-Bayes shrinkage so lightly observed items do not jump to the top
//! on one lucky vote.
//!
//! Optional per-response `exposure_order` and `rating_ages` fields model the
//! temporal context of a judgment: an item tried later in a respondent's history
//! or rated at an older age can receive more item-specific credibility, while
//! all such multipliers remain capped. Presets cover programming languages,
//! movies, travel spots, books, songs, model-validation workflows, and learning
//! resources.
//!
//! ## DES mapping
//!
//!   * `RespondentSource` -> [`RespondentSourceStation`]
//!   * `SurveyEncoder` -> [`SurveyEncoderStation`]
//!   * `EvidenceAggregator` -> [`EvidenceAggregatorStation`]
//!   * `RankingInference` -> [`RankingInferenceStation`]
//!   * `InferenceResultSink` -> [`InferenceResultSinkStation`]
//!
//! ## TS to Rust mapping
//!
//!   * `CollaborativeInferenceScenario` becomes an enum; the many interfaces
//!     become structs. Public `*Params` keep `Option` fields and a private
//!     `NormalizedConfig` holds the resolved values.
//!   * Heavy `Map`/`Set` usage becomes `HashMap`/`HashSet`. Iteration order over
//!     a `HashMap` is unspecified, so where the TS relied on object key order
//!     for OUTPUT (the `invalidEvidence` diagnostic list and ranking fallback)
//!     this port iterates over sorted keys. The per-item aggregated sums are
//!     order-independent (each rating targets a distinct item and per-item
//!     accumulation happens in deterministic respondent order). (Flagged
//!     order divergence.)
//!   * `mulberry32(seed)` becomes an injected seeded `RandomSource`.
//!   * `Math.round` (round half toward +infinity) is reproduced via [`js_round`].
//!   * `throw new Error(...)` for invalid configuration becomes `panic!`.
//!   * The cross-station read `aggregator.survey.respondentsProcessed` is
//!     reproduced by handing the aggregator an `Rc<RefCell<SurveyEncoderStation>>`
//!     handle (the runner only borrows one station at a time, so the read never
//!     aliases).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{AnyToken, DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::ValidationCheck;
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

// =============================================================================
// Channels
// =============================================================================

const CH_RESPONDENT: &str = "respondent";
const CH_RATING_EVIDENCE: &str = "rating-evidence";
const CH_PAIRWISE_EVIDENCE: &str = "pairwise-evidence";
const CH_EVIDENCE_SNAPSHOT: &str = "evidence-snapshot";
const CH_RANKING: &str = "ranking";

// =============================================================================
// Public types
// =============================================================================

/// Built-in scenario preset selector. (TS `type CollaborativeInferenceScenario`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborativeInferenceScenario {
    ProgrammingLanguages,
    ModelValidation,
    LearningResources,
    Movies,
    TravelSpots,
    Books,
    Songs,
    Custom,
}

#[derive(Clone, Debug)]
pub struct CollaborativeInferenceItem {
    pub id: String,
    pub label: Option<String>,
    pub group: Option<String>,
    pub latent_utility: Option<f64>,
    pub exposure: Option<f64>,
    pub prior_score: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct CollaborativeInferenceResponse {
    pub id: Option<String>,
    pub item_ids: Option<Vec<String>>,
    pub ratings: Option<HashMap<String, f64>>,
    pub ranking: Option<Vec<String>>,
    pub exposure_order: Option<Vec<String>>,
    pub rating_ages: Option<HashMap<String, f64>>,
    pub age: Option<f64>,
    pub experience_years: Option<HashMap<String, f64>>,
    pub weight: Option<f64>,
    pub segment: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CollaborativeInferenceParams {
    pub scenario: Option<CollaborativeInferenceScenario>,
    pub items: Option<Vec<CollaborativeInferenceItem>>,
    pub responses: Option<Vec<CollaborativeInferenceResponse>>,
    pub respondent_count: Option<usize>,
    pub respondents: Option<usize>,
    pub min_items_per_respondent: Option<usize>,
    pub max_items_per_respondent: Option<usize>,
    pub respondents_per_tick: Option<usize>,
    pub rating_min: Option<f64>,
    pub rating_max: Option<f64>,
    pub noise_std: Option<f64>,
    pub seed: Option<u32>,
    pub rating_weight: Option<f64>,
    pub ranking_weight: Option<f64>,
    pub shrinkage: Option<f64>,
    pub top_k: Option<usize>,
    pub credibility_weighting: Option<bool>,
    pub credibility_passes: Option<usize>,
    pub min_credible_age: Option<f64>,
    pub reference_age: Option<f64>,
    pub reference_experience_years: Option<f64>,
    pub age_weight_strength: Option<f64>,
    pub experience_weight_strength: Option<f64>,
    pub exposure_order_weight_strength: Option<f64>,
    pub rating_age_weight_strength: Option<f64>,
    pub high_rated_breadth_strength: Option<f64>,
    pub high_rated_score_threshold: Option<f64>,
    pub min_high_rated_items: Option<usize>,
    pub max_credibility_multiplier: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CredibilityWeightSummary {
    pub enabled: bool,
    pub passes: usize,
    pub min_credible_age: f64,
    pub high_rated_score_threshold: f64,
    pub min_high_rated_items: usize,
    pub exposure_order_weight_strength: f64,
    pub rating_age_weight_strength: f64,
    pub mean_respondent_weight: f64,
    pub max_respondent_weight: f64,
    pub capped_experience_claims: usize,
    pub high_rated_bonus_respondents: usize,
}

#[derive(Clone, Debug)]
pub struct CollaborativeItemScore {
    pub rank: usize,
    pub item_id: String,
    pub label: String,
    pub group: Option<String>,
    pub score: f64,
    pub confidence: f64,
    pub uncertainty: f64,
    pub rating_mean: f64,
    pub rating_count: usize,
    pub comparison_count: f64,
    pub pairwise_win_rate: f64,
    pub support: f64,
}

#[derive(Clone, Debug)]
pub struct CollaborativeInferenceCoverage {
    pub items: usize,
    pub items_with_ratings: usize,
    pub items_with_comparisons: usize,
    pub min_ratings_per_item: f64,
    pub mean_ratings_per_item: f64,
    pub max_ratings_per_item: f64,
    pub min_comparisons_per_item: f64,
    pub mean_comparisons_per_item: f64,
    pub max_comparisons_per_item: f64,
}

#[derive(Clone, Debug)]
pub struct StationRoles {
    pub sources: Vec<String>,
    pub stations: Vec<String>,
    pub sinks: Vec<String>,
    pub movables: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct CollaborativeInferenceResult {
    pub scenario: CollaborativeInferenceScenario,
    pub scenario_label: String,
    pub synthetic: bool,
    pub respondent_count: usize,
    pub respondents_processed: usize,
    pub rating_evidence_count: usize,
    pub pairwise_evidence_count: usize,
    pub invalid_evidence: Vec<String>,
    pub credibility: CredibilityWeightSummary,
    pub coverage: CollaborativeInferenceCoverage,
    pub rankings: Vec<CollaborativeItemScore>,
    pub top: Vec<CollaborativeItemScore>,
    pub validation: Vec<ValidationCheck>,
    pub topology: StationGraphSummary,
    pub station_roles: StationRoles,
}

// =============================================================================
// Internal config types
// =============================================================================

struct ScenarioPreset {
    scenario: CollaborativeInferenceScenario,
    label: String,
    default_respondents: usize,
    min_items_per_respondent: usize,
    max_items_per_respondent: usize,
    rating_min: f64,
    rating_max: f64,
    noise_std: f64,
    items: Vec<CollaborativeInferenceItem>,
}

#[derive(Clone)]
struct CredibilityWeightingConfig {
    enabled: bool,
    passes: usize,
    min_credible_age: f64,
    reference_age: f64,
    reference_experience_years: f64,
    age_weight_strength: f64,
    experience_weight_strength: f64,
    exposure_order_weight_strength: f64,
    rating_age_weight_strength: f64,
    high_rated_breadth_strength: f64,
    high_rated_score_threshold: f64,
    min_high_rated_items: usize,
    max_multiplier: f64,
}

struct NormalizedConfig {
    scenario: CollaborativeInferenceScenario,
    scenario_label: String,
    items: Vec<CollaborativeInferenceItem>,
    item_by_id: HashMap<String, CollaborativeInferenceItem>,
    responses: Vec<CollaborativeInferenceResponse>,
    respondent_count: usize,
    min_items_per_respondent: usize,
    max_items_per_respondent: usize,
    respondents_per_tick: usize,
    rating_min: f64,
    rating_max: f64,
    noise_std: f64,
    seed: u32,
    rating_weight: f64,
    ranking_weight: f64,
    shrinkage: f64,
    top_k: usize,
    synthetic: bool,
    credibility: CredibilityWeightingConfig,
}

#[derive(Clone, Debug)]
struct ItemEvidenceStats {
    item_id: String,
    rating_sum: f64,
    rating_weight: f64,
    rating_count: usize,
    pairwise_wins: f64,
    pairwise_losses: f64,
}

struct RespondentWeightProfile {
    respondent_weight: f64,
    item_weights: HashMap<String, f64>,
    high_rated_item_count: usize,
    breadth_multiplier: f64,
    capped_experience_claims: usize,
}

// =============================================================================
// Tokens
// =============================================================================

struct RespondentToken {
    response: CollaborativeInferenceResponse,
}

struct RatingEvidenceToken {
    respondent_id: String,
    item_id: String,
    rating: f64,
    weight: f64,
}

struct PairwisePreferenceToken {
    respondent_id: String,
    winner_id: String,
    loser_id: String,
    weight: f64,
}

struct EvidenceSnapshotToken {
    item_stats: HashMap<String, ItemEvidenceStats>,
    respondents_processed: usize,
    rating_evidence_count: usize,
    pairwise_evidence_count: usize,
}

struct RankingToken {
    rankings: Vec<CollaborativeItemScore>,
}

// =============================================================================
// Stations
// =============================================================================

struct RespondentSourceStation {
    core: StationCore,
    responses: Vec<CollaborativeInferenceResponse>,
    respondents_per_tick: usize,
    emitted_count: usize,
}

impl RespondentSourceStation {
    fn new(
        id: impl Into<String>,
        responses: Vec<CollaborativeInferenceResponse>,
        respondents_per_tick: usize,
    ) -> Self {
        RespondentSourceStation {
            core: StationCore::new(id),
            responses,
            respondents_per_tick,
            emitted_count: 0,
        }
    }
}

impl DESStation for RespondentSourceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.emitted_count < self.responses.len()
    }
    fn run_time_step(&mut self) {
        let n = self
            .respondents_per_tick
            .min(self.responses.len() - self.emitted_count);
        for _ in 0..n {
            let response = self.responses[self.emitted_count].clone();
            let token: AnyToken = Rc::new(RespondentToken { response });
            self.core.emit(token, CH_RESPONDENT);
            self.emitted_count += 1;
        }
    }
}

struct SurveyEncoderStation {
    core: StationCore,
    valid_item_ids: HashSet<String>,
    rating_min: f64,
    rating_max: f64,
    credibility: CredibilityWeightingConfig,
    preliminary_scores: Option<HashMap<String, f64>>,
    respondents_processed: usize,
    rating_evidence_count: usize,
    pairwise_evidence_count: usize,
    respondent_weight_sum: f64,
    max_respondent_weight: f64,
    capped_experience_claims: usize,
    high_rated_bonus_respondents: usize,
    invalid_evidence: Vec<String>,
}

impl SurveyEncoderStation {
    fn new(
        id: impl Into<String>,
        valid_item_ids: HashSet<String>,
        rating_min: f64,
        rating_max: f64,
        credibility: CredibilityWeightingConfig,
        preliminary_scores: Option<HashMap<String, f64>>,
    ) -> Self {
        SurveyEncoderStation {
            core: StationCore::new(id),
            valid_item_ids,
            rating_min,
            rating_max,
            credibility,
            preliminary_scores,
            respondents_processed: 0,
            rating_evidence_count: 0,
            pairwise_evidence_count: 0,
            respondent_weight_sum: 0.0,
            max_respondent_weight: 0.0,
            capped_experience_claims: 0,
            high_rated_bonus_respondents: 0,
            invalid_evidence: Vec::new(),
        }
    }

    fn process(&mut self, response: &CollaborativeInferenceResponse) {
        let respondent_id = response
            .id
            .clone()
            .unwrap_or_else(|| format!("respondent-{}", self.respondents_processed));
        let mut seen: HashSet<String> = HashSet::new();
        if let Some(item_ids) = &response.item_ids {
            for item_id in item_ids {
                if self.valid_item_ids.contains(item_id) {
                    seen.insert(item_id.clone());
                } else {
                    self.invalid_evidence
                        .push(format!("{respondent_id}: unknown item {item_id}"));
                }
            }
        }
        if let Some(ratings) = &response.ratings {
            for item_id in sorted_keys(ratings) {
                if self.valid_item_ids.contains(&item_id) {
                    seen.insert(item_id);
                }
            }
        }
        if let Some(ranking) = &response.ranking {
            for item_id in ranking {
                if self.valid_item_ids.contains(item_id) {
                    seen.insert(item_id.clone());
                }
            }
        }
        if let Some(exposure_order) = &response.exposure_order {
            for item_id in exposure_order {
                if self.valid_item_ids.contains(item_id) {
                    seen.insert(item_id.clone());
                } else {
                    self.invalid_evidence.push(format!(
                        "{respondent_id}: exposure order references unknown item {item_id}"
                    ));
                }
            }
        }
        let profile = respondent_weight_profile(
            response,
            &seen,
            &self.credibility,
            self.preliminary_scores.as_ref(),
        );
        self.respondent_weight_sum += profile.respondent_weight;
        self.max_respondent_weight = self.max_respondent_weight.max(profile.respondent_weight);
        self.capped_experience_claims += profile.capped_experience_claims;
        if profile.breadth_multiplier > 1.0 {
            self.high_rated_bonus_respondents += 1;
        }

        if let Some(ratings) = &response.ratings {
            for item_id in sorted_keys(ratings) {
                let raw_rating = ratings[&item_id];
                if !self.valid_item_ids.contains(&item_id) {
                    self.invalid_evidence.push(format!(
                        "{respondent_id}: rating references unknown item {item_id}"
                    ));
                    continue;
                }
                if !raw_rating.is_finite()
                    || raw_rating < self.rating_min
                    || raw_rating > self.rating_max
                {
                    self.invalid_evidence.push(format!(
                        "{respondent_id}: rating for {item_id} outside [{}, {}]",
                        self.rating_min, self.rating_max
                    ));
                    continue;
                }
                seen.insert(item_id.clone());
                let weight = profile
                    .item_weights
                    .get(&item_id)
                    .copied()
                    .unwrap_or(profile.respondent_weight);
                let token: AnyToken = Rc::new(RatingEvidenceToken {
                    respondent_id: respondent_id.clone(),
                    item_id: item_id.clone(),
                    rating: raw_rating,
                    weight,
                });
                self.core.emit(token, CH_RATING_EVIDENCE);
                self.rating_evidence_count += 1;
            }
        }

        let ranking = self.valid_ranking(response, &seen, &respondent_id);
        for i in 0..ranking.len() {
            for j in (i + 1)..ranking.len() {
                let wi = profile
                    .item_weights
                    .get(&ranking[i])
                    .copied()
                    .unwrap_or(profile.respondent_weight);
                let wj = profile
                    .item_weights
                    .get(&ranking[j])
                    .copied()
                    .unwrap_or(profile.respondent_weight);
                let token: AnyToken = Rc::new(PairwisePreferenceToken {
                    respondent_id: respondent_id.clone(),
                    winner_id: ranking[i].clone(),
                    loser_id: ranking[j].clone(),
                    weight: (wi + wj) / 2.0,
                });
                self.core.emit(token, CH_PAIRWISE_EVIDENCE);
                self.pairwise_evidence_count += 1;
            }
        }
        self.respondents_processed += 1;
    }

    fn valid_ranking(
        &mut self,
        response: &CollaborativeInferenceResponse,
        seen: &HashSet<String>,
        respondent_id: &str,
    ) -> Vec<String> {
        let raw_ranking: Vec<String> = match &response.ranking {
            Some(r) if !r.is_empty() => r.clone(),
            _ => {
                let mut v: Vec<String> = seen.iter().cloned().collect();
                v.sort_by(|a, b| {
                    let ar = response
                        .ratings
                        .as_ref()
                        .and_then(|m| m.get(a))
                        .copied()
                        .unwrap_or(0.0);
                    let br = response
                        .ratings
                        .as_ref()
                        .and_then(|m| m.get(b))
                        .copied()
                        .unwrap_or(0.0);
                    match br.partial_cmp(&ar) {
                        Some(Ordering::Equal) | None => a.cmp(b),
                        Some(o) => o,
                    }
                });
                v
            }
        };
        let mut out: Vec<String> = Vec::new();
        let mut used: HashSet<String> = HashSet::new();
        for item_id in raw_ranking {
            if !self.valid_item_ids.contains(&item_id) {
                self.invalid_evidence.push(format!(
                    "{respondent_id}: ranking references unknown item {item_id}"
                ));
                continue;
            }
            if used.contains(&item_id) {
                continue;
            }
            used.insert(item_id.clone());
            out.push(item_id);
        }
        out
    }
}

impl DESStation for SurveyEncoderStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_RESPONDENT) > 0
    }
    fn run_time_step(&mut self) {
        let respondents = self.core_mut().drain::<RespondentToken>(CH_RESPONDENT);
        for token in &respondents {
            self.process(&token.response);
        }
    }
}

struct EvidenceAggregatorStation {
    core: StationCore,
    stats: HashMap<String, ItemEvidenceStats>,
    rating_evidence_count: usize,
    pairwise_evidence_count: usize,
    survey: Rc<RefCell<SurveyEncoderStation>>,
}

impl EvidenceAggregatorStation {
    fn new(
        id: impl Into<String>,
        item_ids: &[String],
        survey: Rc<RefCell<SurveyEncoderStation>>,
    ) -> Self {
        let mut stats: HashMap<String, ItemEvidenceStats> = HashMap::new();
        for item_id in item_ids {
            stats.insert(item_id.clone(), empty_stats(item_id));
        }
        EvidenceAggregatorStation {
            core: StationCore::new(id),
            stats,
            rating_evidence_count: 0,
            pairwise_evidence_count: 0,
            survey,
        }
    }
}

impl DESStation for EvidenceAggregatorStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_RATING_EVIDENCE) > 0
            || self.core.inbox_size(CH_PAIRWISE_EVIDENCE) > 0
    }
    fn run_time_step(&mut self) {
        let mut changed = false;
        let ratings = self
            .core_mut()
            .drain::<RatingEvidenceToken>(CH_RATING_EVIDENCE);
        for rating in &ratings {
            if let Some(s) = self.stats.get_mut(&rating.item_id) {
                s.rating_sum += rating.rating * rating.weight;
                s.rating_weight += rating.weight;
                s.rating_count += 1;
                self.rating_evidence_count += 1;
                changed = true;
            }
        }
        let comparisons = self
            .core_mut()
            .drain::<PairwisePreferenceToken>(CH_PAIRWISE_EVIDENCE);
        for cmp in &comparisons {
            if !self.stats.contains_key(&cmp.winner_id) || !self.stats.contains_key(&cmp.loser_id) {
                continue;
            }
            if let Some(winner) = self.stats.get_mut(&cmp.winner_id) {
                winner.pairwise_wins += cmp.weight;
            }
            if let Some(loser) = self.stats.get_mut(&cmp.loser_id) {
                loser.pairwise_losses += cmp.weight;
            }
            self.pairwise_evidence_count += 1;
            changed = true;
        }
        if changed {
            let item_stats = clone_stats(&self.stats);
            let respondents_processed = self.survey.borrow().respondents_processed;
            let token: AnyToken = Rc::new(EvidenceSnapshotToken {
                item_stats,
                respondents_processed,
                rating_evidence_count: self.rating_evidence_count,
                pairwise_evidence_count: self.pairwise_evidence_count,
            });
            self.core.emit(token, CH_EVIDENCE_SNAPSHOT);
        }
    }
}

struct RankingInferenceStation {
    core: StationCore,
    items: Vec<CollaborativeInferenceItem>,
    rating_min: f64,
    rating_max: f64,
    rating_weight: f64,
    ranking_weight: f64,
    shrinkage: f64,
}

impl RankingInferenceStation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: impl Into<String>,
        items: Vec<CollaborativeInferenceItem>,
        rating_min: f64,
        rating_max: f64,
        rating_weight: f64,
        ranking_weight: f64,
        shrinkage: f64,
    ) -> Self {
        RankingInferenceStation {
            core: StationCore::new(id),
            items,
            rating_min,
            rating_max,
            rating_weight,
            ranking_weight,
            shrinkage,
        }
    }

    fn rank(&self, snapshot: &EvidenceSnapshotToken) -> Vec<CollaborativeItemScore> {
        let width = self.rating_max - self.rating_min;
        let midpoint = (self.rating_min + self.rating_max) / 2.0;
        let mut rows: Vec<CollaborativeItemScore> = self
            .items
            .iter()
            .map(|item| {
                let s = snapshot
                    .item_stats
                    .get(&item.id)
                    .cloned()
                    .unwrap_or_else(|| empty_stats(&item.id));
                let rating_mean = if s.rating_weight > 0.0 {
                    s.rating_sum / s.rating_weight
                } else {
                    midpoint
                };
                let rating_score = clamp01((rating_mean - self.rating_min) / width);
                let comparisons = s.pairwise_wins + s.pairwise_losses;
                let pairwise_win_rate = if comparisons > 0.0 {
                    s.pairwise_wins / comparisons
                } else {
                    0.5
                };
                let pairwise_score =
                    (s.pairwise_wins + 0.5 * self.shrinkage) / (comparisons + self.shrinkage);
                let rating_confidence =
                    s.rating_count as f64 / (s.rating_count as f64 + self.shrinkage);
                let ranking_confidence = comparisons / (comparisons + self.shrinkage);
                let evidence_weight = self.rating_weight * rating_confidence
                    + self.ranking_weight * ranking_confidence;
                let prior_score = clamp01(item.prior_score.unwrap_or(0.5));
                let empirical = if evidence_weight > 0.0 {
                    (self.rating_weight * rating_confidence * rating_score
                        + self.ranking_weight * ranking_confidence * pairwise_score)
                        / evidence_weight
                } else {
                    prior_score
                };
                let support = s.rating_count as f64 + comparisons;
                let confidence = support / (support + self.shrinkage);
                let score = clamp01(confidence * empirical + (1.0 - confidence) * prior_score);
                let uncertainty =
                    ((score * (1.0 - score)).max(0.0) / (support + self.shrinkage).max(1.0)).sqrt();
                CollaborativeItemScore {
                    rank: 0,
                    item_id: item.id.clone(),
                    label: item.label.clone().unwrap_or_else(|| item.id.clone()),
                    group: item.group.clone(),
                    score,
                    confidence,
                    uncertainty,
                    rating_mean,
                    rating_count: s.rating_count,
                    comparison_count: comparisons,
                    pairwise_win_rate,
                    support,
                }
            })
            .collect();
        rows.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(Ordering::Equal) | None => match b.confidence.partial_cmp(&a.confidence) {
                Some(Ordering::Equal) | None => a.label.cmp(&b.label),
                Some(o) => o,
            },
            Some(o) => o,
        });
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank = i + 1;
        }
        rows
    }
}

impl DESStation for RankingInferenceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_EVIDENCE_SNAPSHOT) > 0
    }
    fn run_time_step(&mut self) {
        let snapshots = self
            .core_mut()
            .drain::<EvidenceSnapshotToken>(CH_EVIDENCE_SNAPSHOT);
        if snapshots.is_empty() {
            return;
        }
        let latest = &snapshots[snapshots.len() - 1];
        let rankings = self.rank(latest);
        let token: AnyToken = Rc::new(RankingToken { rankings });
        self.core.emit(token, CH_RANKING);
    }
}

struct InferenceResultSinkStation {
    core: StationCore,
    latest: Option<Rc<RankingToken>>,
}

impl InferenceResultSinkStation {
    fn new(id: impl Into<String>) -> Self {
        InferenceResultSinkStation {
            core: StationCore::new(id),
            latest: None,
        }
    }
}

impl DESStation for InferenceResultSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_RANKING) > 0
    }
    fn run_time_step(&mut self) {
        let rankings = self.core_mut().drain::<RankingToken>(CH_RANKING);
        if let Some(last) = rankings.last() {
            self.latest = Some(last.clone());
        }
    }
}

// =============================================================================
// Driver
// =============================================================================

/// Run the multi-pass collaborative-inference pipeline. (TS
/// `runCollaborativeInference`.)
pub fn run_collaborative_inference(
    params: CollaborativeInferenceParams,
) -> CollaborativeInferenceResult {
    let cfg = normalize_collaborative_inference_params(params);
    let passes = if cfg.credibility.enabled {
        cfg.credibility.passes
    } else {
        1
    };
    let mut preliminary_scores: Option<HashMap<String, f64>> = None;
    let mut final_run: Option<CollaborativeInferencePassResult> = None;
    for _pass in 0..passes {
        let run = run_collaborative_inference_pass(&cfg, preliminary_scores.as_ref());
        preliminary_scores = Some(
            run.rankings
                .iter()
                .map(|row| (row.item_id.clone(), row.score))
                .collect(),
        );
        final_run = Some(run);
    }
    let final_run = final_run.unwrap_or_else(|| panic!("collaborative-inference did not run"));
    pass_to_result(&cfg, &final_run, passes)
}

struct CollaborativeInferencePassResult {
    respondents_processed: usize,
    rating_evidence_count: usize,
    pairwise_evidence_count: usize,
    respondent_weight_sum: f64,
    max_respondent_weight: f64,
    capped_experience_claims: usize,
    high_rated_bonus_respondents: usize,
    invalid_evidence: Vec<String>,
    rankings: Vec<CollaborativeItemScore>,
    coverage: CollaborativeInferenceCoverage,
    validation: Vec<ValidationCheck>,
    topology: StationGraphSummary,
}

fn run_collaborative_inference_pass(
    cfg: &NormalizedConfig,
    preliminary_scores: Option<&HashMap<String, f64>>,
) -> CollaborativeInferencePassResult {
    let source = Rc::new(RefCell::new(RespondentSourceStation::new(
        "respondent-source",
        cfg.responses.clone(),
        cfg.respondents_per_tick,
    )));
    let valid_ids: HashSet<String> = cfg.items.iter().map(|i| i.id.clone()).collect();
    let survey = Rc::new(RefCell::new(SurveyEncoderStation::new(
        "survey-encoder",
        valid_ids,
        cfg.rating_min,
        cfg.rating_max,
        cfg.credibility.clone(),
        preliminary_scores.cloned(),
    )));
    let item_ids: Vec<String> = cfg.items.iter().map(|i| i.id.clone()).collect();
    let aggregator = Rc::new(RefCell::new(EvidenceAggregatorStation::new(
        "evidence-aggregator",
        &item_ids,
        survey.clone(),
    )));
    let ranker = Rc::new(RefCell::new(RankingInferenceStation::new(
        "ranking-inference",
        cfg.items.clone(),
        cfg.rating_min,
        cfg.rating_max,
        cfg.rating_weight,
        cfg.ranking_weight,
        cfg.shrinkage,
    )));
    let sink = Rc::new(RefCell::new(InferenceResultSinkStation::new(
        "inference-result-sink",
    )));

    source
        .borrow_mut()
        .core_mut()
        .pipe(survey.clone() as StationRef, CH_RESPONDENT, CH_RESPONDENT);
    survey.borrow_mut().core_mut().pipe(
        aggregator.clone() as StationRef,
        CH_RATING_EVIDENCE,
        CH_RATING_EVIDENCE,
    );
    survey.borrow_mut().core_mut().pipe(
        aggregator.clone() as StationRef,
        CH_PAIRWISE_EVIDENCE,
        CH_PAIRWISE_EVIDENCE,
    );
    aggregator.borrow_mut().core_mut().pipe(
        ranker.clone() as StationRef,
        CH_EVIDENCE_SNAPSHOT,
        CH_EVIDENCE_SNAPSHOT,
    );
    ranker
        .borrow_mut()
        .core_mut()
        .pipe(sink.clone() as StationRef, CH_RANKING, CH_RANKING);

    let max_ticks =
        (cfg.responses.len() as f64 / cfg.respondents_per_tick as f64).ceil() as usize + 5;
    run_iterative_des(
        vec![
            source.clone() as StationRef,
            survey.clone() as StationRef,
            aggregator.clone() as StationRef,
            ranker.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            run_validators: false,
            ..Default::default()
        },
    );

    let latest = sink.borrow().latest.clone();
    let rankings = latest
        .unwrap_or_else(|| panic!("collaborative-inference did not produce a ranking"))
        .rankings
        .clone();
    let coverage = coverage_summary(&aggregator.borrow().stats);
    let emitted_count = source.borrow().emitted_count;
    let (
        survey_processed,
        survey_rating,
        survey_pairwise,
        weight_sum,
        max_weight,
        capped,
        high_bonus,
        invalid,
    ) = {
        let s = survey.borrow();
        (
            s.respondents_processed,
            s.rating_evidence_count,
            s.pairwise_evidence_count,
            s.respondent_weight_sum,
            s.max_respondent_weight,
            s.capped_experience_claims,
            s.high_rated_bonus_respondents,
            s.invalid_evidence.clone(),
        )
    };
    let (agg_rating, agg_pairwise) = {
        let a = aggregator.borrow();
        (a.rating_evidence_count, a.pairwise_evidence_count)
    };

    let group = || Some("collaborative inference".to_string());
    let validation = vec![
        ValidationCheck {
            name: "respondent conservation".to_string(),
            group: group(),
            passed: emitted_count == survey_processed && survey_processed == cfg.responses.len(),
            observed: Some(survey_processed.to_string()),
            expected: Some(cfg.responses.len().to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "rating evidence conservation".to_string(),
            group: group(),
            passed: survey_rating == agg_rating,
            observed: Some(agg_rating.to_string()),
            expected: Some(survey_rating.to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "pairwise evidence conservation".to_string(),
            group: group(),
            passed: survey_pairwise == agg_pairwise,
            observed: Some(agg_pairwise.to_string()),
            expected: Some(survey_pairwise.to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "all items ranked".to_string(),
            group: group(),
            passed: rankings.len() == cfg.items.len(),
            observed: Some(rankings.len().to_string()),
            expected: Some(cfg.items.len().to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "scores are finite probabilities".to_string(),
            group: group(),
            passed: rankings
                .iter()
                .all(|r| r.score.is_finite() && r.score >= 0.0 && r.score <= 1.0),
            expected: Some("score in [0, 1] for every item".to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "coverage reaches every item".to_string(),
            group: group(),
            passed: coverage.items_with_ratings == cfg.items.len()
                && coverage.items_with_comparisons == cfg.items.len(),
            observed: Some(format!(
                "{}/{} rated, {}/{} compared",
                coverage.items_with_ratings,
                cfg.items.len(),
                coverage.items_with_comparisons,
                cfg.items.len()
            )),
            expected: Some("each item has rating and comparison evidence".to_string()),
            ..Default::default()
        },
        ValidationCheck {
            name: "no invalid evidence".to_string(),
            group: group(),
            passed: invalid.is_empty(),
            observed: Some(invalid.len().to_string()),
            expected: Some("0".to_string()),
            details: Some(
                invalid
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
        },
        ValidationCheck {
            name: "credibility weights are finite".to_string(),
            group: group(),
            passed: weight_sum.is_finite() && max_weight.is_finite(),
            observed: Some(format!(
                "mean={}, max={}",
                if survey_processed > 0 {
                    weight_sum / survey_processed as f64
                } else {
                    0.0
                },
                max_weight
            )),
            expected: Some("finite respondent weights".to_string()),
            ..Default::default()
        },
    ];

    let topology = build_topology();

    CollaborativeInferencePassResult {
        respondents_processed: survey_processed,
        rating_evidence_count: agg_rating,
        pairwise_evidence_count: agg_pairwise,
        respondent_weight_sum: weight_sum,
        max_respondent_weight: max_weight,
        capped_experience_claims: capped,
        high_rated_bonus_respondents: high_bonus,
        invalid_evidence: invalid,
        rankings,
        coverage,
        validation,
        topology,
    }
}

fn pass_to_result(
    cfg: &NormalizedConfig,
    pass: &CollaborativeInferencePassResult,
    passes: usize,
) -> CollaborativeInferenceResult {
    let top: Vec<CollaborativeItemScore> = pass.rankings.iter().take(cfg.top_k).cloned().collect();
    let invalid_evidence: Vec<String> = pass.invalid_evidence.iter().take(25).cloned().collect();
    let mean_respondent_weight = if pass.respondents_processed > 0 {
        pass.respondent_weight_sum / pass.respondents_processed as f64
    } else {
        0.0
    };
    CollaborativeInferenceResult {
        scenario: cfg.scenario,
        scenario_label: cfg.scenario_label.clone(),
        synthetic: cfg.synthetic,
        respondent_count: cfg.responses.len(),
        respondents_processed: pass.respondents_processed,
        rating_evidence_count: pass.rating_evidence_count,
        pairwise_evidence_count: pass.pairwise_evidence_count,
        invalid_evidence,
        credibility: CredibilityWeightSummary {
            enabled: cfg.credibility.enabled,
            passes,
            min_credible_age: cfg.credibility.min_credible_age,
            high_rated_score_threshold: cfg.credibility.high_rated_score_threshold,
            min_high_rated_items: cfg.credibility.min_high_rated_items,
            exposure_order_weight_strength: cfg.credibility.exposure_order_weight_strength,
            rating_age_weight_strength: cfg.credibility.rating_age_weight_strength,
            mean_respondent_weight,
            max_respondent_weight: pass.max_respondent_weight,
            capped_experience_claims: pass.capped_experience_claims,
            high_rated_bonus_respondents: pass.high_rated_bonus_respondents,
        },
        coverage: pass.coverage.clone(),
        rankings: pass.rankings.clone(),
        top,
        validation: pass.validation.clone(),
        topology: pass.topology.clone(),
        station_roles: StationRoles {
            sources: vec!["respondent-source".to_string()],
            stations: vec![
                "survey-encoder".to_string(),
                "evidence-aggregator".to_string(),
                "ranking-inference".to_string(),
            ],
            sinks: vec!["inference-result-sink".to_string()],
            movables: vec![
                "RespondentToken".to_string(),
                "RatingEvidenceToken".to_string(),
                "PairwisePreferenceToken".to_string(),
                "EvidenceSnapshotToken".to_string(),
                "RankingToken".to_string(),
            ],
        },
    }
}

fn build_topology() -> StationGraphSummary {
    let source = StationOrId::Id("respondent-source".to_string());
    let survey = StationOrId::Id("survey-encoder".to_string());
    let aggregator = StationOrId::Id("evidence-aggregator".to_string());
    let ranker = StationOrId::Id("ranking-inference".to_string());
    let sink = StationOrId::Id("inference-result-sink".to_string());
    let edges = vec![
        channel_edge(&source, CH_RESPONDENT, &survey, Some(CH_RESPONDENT)),
        channel_edge(
            &survey,
            CH_RATING_EVIDENCE,
            &aggregator,
            Some(CH_RATING_EVIDENCE),
        ),
        channel_edge(
            &survey,
            CH_PAIRWISE_EVIDENCE,
            &aggregator,
            Some(CH_PAIRWISE_EVIDENCE),
        ),
        channel_edge(
            &aggregator,
            CH_EVIDENCE_SNAPSHOT,
            &ranker,
            Some(CH_EVIDENCE_SNAPSHOT),
        ),
        channel_edge(&ranker, CH_RANKING, &sink, Some(CH_RANKING)),
    ];
    station_graph(
        &[source, survey, aggregator, ranker, sink],
        &[
            "RespondentToken".to_string(),
            "RatingEvidenceToken".to_string(),
            "PairwisePreferenceToken".to_string(),
            "EvidenceSnapshotToken".to_string(),
            "RankingToken".to_string(),
        ],
        &edges,
    )
}

// =============================================================================
// Normalization
// =============================================================================

fn normalize_collaborative_inference_params(
    params: CollaborativeInferenceParams,
) -> NormalizedConfig {
    let scenario = params
        .scenario
        .unwrap_or(CollaborativeInferenceScenario::ProgrammingLanguages);
    let preset = scenario_preset(scenario);
    let items_src = match &params.items {
        Some(it) if !it.is_empty() => it.clone(),
        _ => preset.items.clone(),
    };
    let items = normalize_items(&items_src);
    let mut item_by_id: HashMap<String, CollaborativeInferenceItem> = HashMap::new();
    for item in &items {
        item_by_id.insert(item.id.clone(), item.clone());
    }
    if item_by_id.len() != items.len() {
        panic!("collaborative-inference: item ids must be unique");
    }

    let respondent_count = params
        .respondent_count
        .or(params.respondents)
        .unwrap_or(preset.default_respondents);
    let min_items = params
        .min_items_per_respondent
        .unwrap_or(preset.min_items_per_respondent);
    let max_items = params
        .max_items_per_respondent
        .unwrap_or(preset.max_items_per_respondent);
    let rating_min = params.rating_min.unwrap_or(preset.rating_min);
    let rating_max = params.rating_max.unwrap_or(preset.rating_max);
    let seed = params.seed.unwrap_or(1);
    let credibility = normalize_credibility_config(&params);
    if rating_max <= rating_min {
        panic!("collaborative-inference: ratingMax must be greater than ratingMin");
    }
    if min_items < 1 || max_items < min_items {
        panic!(
            "collaborative-inference: require 1 <= minItemsPerRespondent <= maxItemsPerRespondent"
        );
    }

    let noise_std = params.noise_std.unwrap_or(preset.noise_std);
    let has_responses = params
        .responses
        .as_ref()
        .map(|r| !r.is_empty())
        .unwrap_or(false);
    let responses = if has_responses {
        params
            .responses
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, r)| normalize_response(r, i))
            .collect()
    } else {
        generate_synthetic_responses(
            &items,
            respondent_count,
            min_items,
            max_items,
            rating_min,
            rating_max,
            noise_std,
            seed,
            credibility.min_credible_age,
        )
    };

    NormalizedConfig {
        scenario,
        scenario_label: preset.label.clone(),
        items,
        item_by_id,
        respondent_count: responses.len(),
        responses,
        min_items_per_respondent: min_items,
        max_items_per_respondent: max_items,
        respondents_per_tick: params.respondents_per_tick.unwrap_or(100).max(1),
        rating_min,
        rating_max,
        noise_std,
        seed,
        rating_weight: params.rating_weight.unwrap_or(0.55).max(0.0),
        ranking_weight: params.ranking_weight.unwrap_or(0.45).max(0.0),
        shrinkage: params.shrinkage.unwrap_or(12.0).max(1e-9),
        top_k: params.top_k.unwrap_or(10).max(1),
        synthetic: !has_responses,
        credibility,
    }
}

fn normalize_items(items: &[CollaborativeInferenceItem]) -> Vec<CollaborativeInferenceItem> {
    if items.is_empty() {
        panic!("collaborative-inference: items must be non-empty");
    }
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let id = if item.id.is_empty() {
                format!("item-{}", i + 1)
            } else {
                item.id.clone()
            };
            CollaborativeInferenceItem {
                label: Some(item.label.clone().unwrap_or_else(|| id.clone())),
                group: item.group.clone(),
                latent_utility: Some(clamp01(item.latent_utility.unwrap_or(0.5))),
                exposure: Some(item.exposure.unwrap_or(1.0).max(0.0)),
                prior_score: item.prior_score.map(clamp01),
                id,
            }
        })
        .collect()
}

fn normalize_response(
    response: &CollaborativeInferenceResponse,
    index: usize,
) -> CollaborativeInferenceResponse {
    let rating_ids: Vec<String> = response
        .ratings
        .as_ref()
        .map(sorted_keys)
        .unwrap_or_default();
    let exposure_ids: Vec<String> = response
        .exposure_order
        .as_ref()
        .map(|r| unique(r))
        .unwrap_or_default();
    let item_ids: Vec<String> = match &response.item_ids {
        Some(ids) if !ids.is_empty() => ids.clone(),
        _ => match &response.ranking {
            Some(r) if !r.is_empty() => r.clone(),
            _ if !exposure_ids.is_empty() => exposure_ids.clone(),
            _ => rating_ids,
        },
    };
    CollaborativeInferenceResponse {
        id: Some(
            response
                .id
                .clone()
                .unwrap_or_else(|| format!("respondent-{}", index + 1)),
        ),
        item_ids: Some(unique(&item_ids)),
        ratings: response.ratings.clone(),
        ranking: response.ranking.as_ref().map(|r| unique(r)),
        exposure_order: response.exposure_order.as_ref().map(|r| unique(r)),
        rating_ages: response.rating_ages.clone(),
        age: response.age,
        experience_years: response.experience_years.clone(),
        weight: Some(finite_positive(response.weight, 1.0)),
        segment: response.segment.clone(),
    }
}

fn normalize_credibility_config(
    params: &CollaborativeInferenceParams,
) -> CredibilityWeightingConfig {
    CredibilityWeightingConfig {
        enabled: params.credibility_weighting.unwrap_or(true),
        passes: params.credibility_passes.unwrap_or(2).max(1),
        min_credible_age: params.min_credible_age.unwrap_or(15.0).max(0.0),
        reference_age: params.reference_age.unwrap_or(50.0).max(1.0),
        reference_experience_years: params.reference_experience_years.unwrap_or(15.0).max(1.0),
        age_weight_strength: params.age_weight_strength.unwrap_or(0.35).max(0.0),
        experience_weight_strength: params.experience_weight_strength.unwrap_or(0.60).max(0.0),
        exposure_order_weight_strength: params
            .exposure_order_weight_strength
            .unwrap_or(0.30)
            .max(0.0),
        rating_age_weight_strength: params.rating_age_weight_strength.unwrap_or(0.35).max(0.0),
        high_rated_breadth_strength: params.high_rated_breadth_strength.unwrap_or(0.40).max(0.0),
        high_rated_score_threshold: params
            .high_rated_score_threshold
            .unwrap_or(0.72)
            .clamp(0.0, 1.0),
        min_high_rated_items: params.min_high_rated_items.unwrap_or(2).max(1),
        max_multiplier: params.max_credibility_multiplier.unwrap_or(3.0).max(1.0),
    }
}

fn respondent_weight_profile(
    response: &CollaborativeInferenceResponse,
    seen_items: &HashSet<String>,
    cfg: &CredibilityWeightingConfig,
    preliminary_scores: Option<&HashMap<String, f64>>,
) -> RespondentWeightProfile {
    let explicit_weight = finite_positive(response.weight, 1.0);
    if !cfg.enabled {
        let mut item_weights = HashMap::new();
        for item_id in seen_items {
            item_weights.insert(item_id.clone(), explicit_weight);
        }
        return RespondentWeightProfile {
            respondent_weight: explicit_weight,
            item_weights,
            high_rated_item_count: 0,
            breadth_multiplier: 1.0,
            capped_experience_claims: 0,
        };
    }

    let age = finite_non_negative(response.age);
    let max_credible_years = match age {
        None => f64::INFINITY,
        Some(a) => (a - cfg.min_credible_age).max(0.0),
    };
    let age_multiplier = match age {
        None => 1.0,
        Some(a) => {
            1.0 + cfg.age_weight_strength
                * normalized_log(
                    (a - cfg.min_credible_age).max(0.0),
                    (cfg.reference_age - cfg.min_credible_age).max(1.0),
                )
        }
    };

    let high_rated_item_count = match preliminary_scores {
        Some(scores) => seen_items
            .iter()
            .filter(|item_id| {
                scores.get(*item_id).copied().unwrap_or(0.0) >= cfg.high_rated_score_threshold
            })
            .count(),
        None => 0,
    };
    let breadth_multiplier = if high_rated_item_count >= cfg.min_high_rated_items {
        1.0 + cfg.high_rated_breadth_strength
            * 2.0_f64.min(high_rated_item_count as f64 / cfg.min_high_rated_items as f64)
    } else {
        1.0
    };
    let exposure_positions = exposure_order_positions(response, seen_items);
    let exposure_count = exposure_positions.len();

    let base = cap_multiplier(
        explicit_weight * age_multiplier * breadth_multiplier,
        explicit_weight,
        cfg.max_multiplier,
    );
    let mut item_weights: HashMap<String, f64> = HashMap::new();
    let mut capped_experience_claims = 0usize;
    for item_id in seen_items {
        let raw_years = finite_non_negative(
            response
                .experience_years
                .as_ref()
                .and_then(|m| m.get(item_id).copied()),
        )
        .unwrap_or(0.0);
        let capped_years = raw_years.min(max_credible_years);
        if raw_years > capped_years + 1e-9 {
            capped_experience_claims += 1;
        }
        let experience_multiplier = 1.0
            + cfg.experience_weight_strength
                * normalized_log(capped_years, cfg.reference_experience_years);
        let exposure_multiplier = exposure_positions
            .get(item_id)
            .map(|&pos| exposure_order_multiplier(pos, exposure_count, cfg))
            .unwrap_or(1.0);
        let rating_age_multiplier = rating_age_multiplier(response, item_id, age, cfg);
        item_weights.insert(
            item_id.clone(),
            cap_multiplier(
                base * experience_multiplier * exposure_multiplier * rating_age_multiplier,
                explicit_weight,
                cfg.max_multiplier,
            ),
        );
    }

    RespondentWeightProfile {
        respondent_weight: base,
        item_weights,
        high_rated_item_count,
        breadth_multiplier,
        capped_experience_claims,
    }
}

fn exposure_order_positions(
    response: &CollaborativeInferenceResponse,
    seen_items: &HashSet<String>,
) -> HashMap<String, usize> {
    let mut positions = HashMap::new();
    if let Some(order) = &response.exposure_order {
        for item_id in order {
            if seen_items.contains(item_id) && !positions.contains_key(item_id) {
                let pos = positions.len();
                positions.insert(item_id.clone(), pos);
            }
        }
    }
    positions
}

fn exposure_order_multiplier(
    position: usize,
    exposure_count: usize,
    cfg: &CredibilityWeightingConfig,
) -> f64 {
    if exposure_count <= 1 || cfg.exposure_order_weight_strength <= 0.0 {
        return 1.0;
    }
    let maturity = position as f64 / (exposure_count - 1) as f64;
    (1.0 + cfg.exposure_order_weight_strength * (2.0 * maturity - 1.0)).max(0.05)
}

fn rating_age_multiplier(
    response: &CollaborativeInferenceResponse,
    item_id: &str,
    respondent_age: Option<f64>,
    cfg: &CredibilityWeightingConfig,
) -> f64 {
    if cfg.rating_age_weight_strength <= 0.0 {
        return 1.0;
    }
    let Some(raw_age) = response
        .rating_ages
        .as_ref()
        .and_then(|m| m.get(item_id).copied())
        .and_then(|v| finite_non_negative(Some(v)))
    else {
        return 1.0;
    };
    let age = match respondent_age {
        Some(current_age) => raw_age.min(current_age),
        None => raw_age,
    };
    1.0 + cfg.rating_age_weight_strength
        * normalized_log(
            (age - cfg.min_credible_age).max(0.0),
            (cfg.reference_age - cfg.min_credible_age).max(1.0),
        )
}

#[allow(clippy::too_many_arguments)]
fn generate_synthetic_responses(
    items: &[CollaborativeInferenceItem],
    respondent_count: usize,
    min_items: usize,
    max_items: usize,
    rating_min: f64,
    rating_max: f64,
    noise_std: f64,
    seed: u32,
    min_credible_age: f64,
) -> Vec<CollaborativeInferenceResponse> {
    let mut rng = mulberry32(seed);
    let mut out: Vec<CollaborativeInferenceResponse> = Vec::new();
    for r in 0..respondent_count {
        let k = random_int(&mut rng, min_items, max_items);
        let selected = weighted_sample_without_replacement(items, k, &mut rng);
        let age = synthetic_age(&mut rng);
        let max_credible_years = (age - min_credible_age).max(0.0);
        let personal_offset = normal01(&mut rng) * 0.12;
        let histories = synthetic_item_histories(&selected, age, min_credible_age, &mut rng);
        let exposure_order: Vec<String> = histories
            .iter()
            .map(|(item, _start_age, _rating_age)| item.id.clone())
            .collect();
        let mut ratings: HashMap<String, f64> = HashMap::new();
        let mut experience_years: HashMap<String, f64> = HashMap::new();
        let mut rating_ages: HashMap<String, f64> = HashMap::new();
        for (item, start_age, rating_age) in &histories {
            let latent = item.latent_utility.unwrap_or(0.5);
            let noisy_utility =
                clamp01(latent + personal_offset + normal01(&mut rng) * noise_std / 10.0);
            ratings.insert(
                item.id.clone(),
                round_to(rating_min + noisy_utility * (rating_max - rating_min), 2),
            );
            experience_years.insert(
                item.id.clone(),
                round_to((age - *start_age).max(0.0).min(max_credible_years), 1),
            );
            rating_ages.insert(item.id.clone(), round_to(*rating_age, 1));
        }
        let mut ranking: Vec<String> = selected.iter().map(|item| item.id.clone()).collect();
        ranking.sort_by(|a, b| {
            let ra = ratings.get(a).copied().unwrap_or(0.0);
            let rb = ratings.get(b).copied().unwrap_or(0.0);
            match rb.partial_cmp(&ra) {
                Some(Ordering::Equal) | None => a.cmp(b),
                Some(o) => o,
            }
        });
        out.push(CollaborativeInferenceResponse {
            id: Some(format!("respondent-{}", r + 1)),
            age: Some(age),
            item_ids: Some(selected.iter().map(|item| item.id.clone()).collect()),
            experience_years: Some(experience_years),
            ratings: Some(ratings),
            ranking: Some(ranking),
            exposure_order: Some(exposure_order),
            rating_ages: Some(rating_ages),
            weight: None,
            segment: None,
        });
    }
    out
}

fn synthetic_item_histories(
    selected: &[CollaborativeInferenceItem],
    age: f64,
    min_credible_age: f64,
    rng: &mut dyn RandomSource,
) -> Vec<(CollaborativeInferenceItem, f64, f64)> {
    let span = (age - min_credible_age).max(0.0);
    let mut histories: Vec<(CollaborativeInferenceItem, f64, f64)> = selected
        .iter()
        .map(|item| {
            let exposure = item.exposure.unwrap_or(1.0).max(0.0);
            let early_bias = 1.0 + 0.35 * exposure.min(4.0);
            let start_age = min_credible_age + span * rng.next_float().powf(early_bias);
            let assessment_gap = (age - start_age).max(0.0) * (0.25 + 0.65 * rng.next_float());
            let rating_age = (start_age + assessment_gap).min(age);
            (item.clone(), start_age, rating_age)
        })
        .collect();
    histories.sort_by(|a, b| match a.1.partial_cmp(&b.1) {
        Some(Ordering::Equal) | None => a.0.id.cmp(&b.0.id),
        Some(o) => o,
    });
    histories
}

// =============================================================================
// Scenario presets
// =============================================================================

fn scenario_preset(scenario: CollaborativeInferenceScenario) -> ScenarioPreset {
    match scenario {
        CollaborativeInferenceScenario::ProgrammingLanguages => programming_language_preset(),
        CollaborativeInferenceScenario::ModelValidation => model_validation_preset(),
        CollaborativeInferenceScenario::LearningResources => learning_resources_preset(),
        CollaborativeInferenceScenario::Movies => movies_preset(),
        CollaborativeInferenceScenario::TravelSpots => travel_spots_preset(),
        CollaborativeInferenceScenario::Books => books_preset(),
        CollaborativeInferenceScenario::Songs => songs_preset(),
        CollaborativeInferenceScenario::Custom => ScenarioPreset {
            scenario,
            label: "Custom collaborative inference scenario".to_string(),
            default_respondents: 200,
            min_items_per_respondent: 3,
            max_items_per_respondent: 5,
            rating_min: 1.0,
            rating_max: 10.0,
            noise_std: 1.0,
            items: vec![
                simple_item("option-a", "Option A", 0.58, 1.0),
                simple_item("option-b", "Option B", 0.50, 1.0),
                simple_item("option-c", "Option C", 0.42, 1.0),
            ],
        },
    }
}

fn simple_item(id: &str, label: &str, latent: f64, exposure: f64) -> CollaborativeInferenceItem {
    CollaborativeInferenceItem {
        id: id.to_string(),
        label: Some(label.to_string()),
        group: None,
        latent_utility: Some(latent),
        exposure: Some(exposure),
        prior_score: None,
    }
}

fn grouped_item(
    id: &str,
    label: &str,
    latent: f64,
    exposure: f64,
    group: &str,
) -> CollaborativeInferenceItem {
    CollaborativeInferenceItem {
        id: id.to_string(),
        label: Some(label.to_string()),
        group: Some(group.to_string()),
        latent_utility: Some(latent),
        exposure: Some(exposure),
        prior_score: None,
    }
}

fn programming_language_preset() -> ScenarioPreset {
    let names: [(&str, &str, f64, f64); 50] = [
        ("python", "Python", 0.88, 1.7),
        ("typescript", "TypeScript", 0.84, 1.45),
        ("rust", "Rust", 0.83, 0.75),
        ("go", "Go", 0.80, 1.05),
        ("kotlin", "Kotlin", 0.77, 0.70),
        ("swift", "Swift", 0.75, 0.66),
        ("javascript", "JavaScript", 0.74, 1.9),
        ("csharp", "C#", 0.73, 1.15),
        ("java", "Java", 0.70, 1.55),
        ("scala", "Scala", 0.69, 0.42),
        ("elixir", "Elixir", 0.68, 0.32),
        ("clojure", "Clojure", 0.67, 0.25),
        ("julia", "Julia", 0.66, 0.30),
        ("ruby", "Ruby", 0.65, 0.75),
        ("fsharp", "F#", 0.64, 0.20),
        ("haskell", "Haskell", 0.63, 0.22),
        ("php", "PHP", 0.61, 1.1),
        ("c", "C", 0.60, 1.05),
        ("cpp", "C++", 0.59, 1.15),
        ("r", "R", 0.58, 0.70),
        ("dart", "Dart", 0.57, 0.42),
        ("lua", "Lua", 0.56, 0.28),
        ("erlang", "Erlang", 0.55, 0.18),
        ("ocaml", "OCaml", 0.54, 0.16),
        ("zig", "Zig", 0.53, 0.24),
        ("nim", "Nim", 0.52, 0.12),
        ("perl", "Perl", 0.51, 0.25),
        ("shell", "Shell", 0.50, 1.25),
        ("sql", "SQL", 0.49, 1.35),
        ("matlab", "MATLAB", 0.48, 0.35),
        ("groovy", "Groovy", 0.47, 0.22),
        ("powershell", "PowerShell", 0.46, 0.65),
        ("objective-c", "Objective-C", 0.45, 0.20),
        ("visual-basic", "Visual Basic", 0.44, 0.25),
        ("fortran", "Fortran", 0.43, 0.12),
        ("cobol", "COBOL", 0.42, 0.10),
        ("delphi", "Delphi", 0.41, 0.12),
        ("smalltalk", "Smalltalk", 0.40, 0.08),
        ("elm", "Elm", 0.39, 0.12),
        ("reason", "ReasonML", 0.38, 0.08),
        ("solidity", "Solidity", 0.37, 0.24),
        ("apex", "Apex", 0.36, 0.15),
        ("abap", "ABAP", 0.35, 0.14),
        ("assembly", "Assembly", 0.34, 0.42),
        ("racket", "Racket", 0.33, 0.10),
        ("prolog", "Prolog", 0.32, 0.08),
        ("ada", "Ada", 0.31, 0.08),
        ("vba", "VBA", 0.30, 0.28),
        ("scratch", "Scratch", 0.29, 0.18),
        ("coffeescript", "CoffeeScript", 0.28, 0.10),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::ProgrammingLanguages,
        label: "Programming languages ranked from sparse developer experience".to_string(),
        default_respondents: 10000,
        min_items_per_respondent: 4,
        max_items_per_respondent: 5,
        rating_min: 1.0,
        rating_max: 10.0,
        noise_std: 1.1,
        items: names
            .iter()
            .map(|(id, label, latent, exposure)| {
                grouped_item(id, label, *latent, *exposure, "language")
            })
            .collect(),
    }
}

fn model_validation_preset() -> ScenarioPreset {
    let rows: [(&str, &str, f64, f64, &str); 12] = [
        (
            "des-station-graph",
            "DES station graph",
            0.86,
            1.2,
            "execution",
        ),
        (
            "fel-reference",
            "Future-event-list reference",
            0.78,
            0.9,
            "validation",
        ),
        (
            "monte-carlo",
            "Monte Carlo replication",
            0.76,
            1.1,
            "validation",
        ),
        (
            "analytical-baseline",
            "Analytical baseline",
            0.74,
            0.8,
            "validation",
        ),
        (
            "simpy-reference",
            "SimPy reference model",
            0.71,
            0.7,
            "external",
        ),
        (
            "ciw-reference",
            "Ciw queueing reference",
            0.68,
            0.5,
            "external",
        ),
        ("agent-based", "Agent-based model", 0.66, 0.8, "execution"),
        (
            "digital-twin",
            "Hybrid digital twin",
            0.64,
            0.45,
            "execution",
        ),
        (
            "neural-surrogate",
            "Neural surrogate",
            0.60,
            0.55,
            "approximation",
        ),
        (
            "spreadsheet",
            "Spreadsheet prototype",
            0.48,
            0.9,
            "baseline",
        ),
        ("ad-hoc-script", "Ad-hoc script", 0.38, 1.0, "baseline"),
        ("manual-review", "Manual review only", 0.30, 0.6, "baseline"),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::ModelValidation,
        label: "Model validation workflows ranked by external reviewers".to_string(),
        default_respondents: 800,
        min_items_per_respondent: 3,
        max_items_per_respondent: 5,
        rating_min: 1.0,
        rating_max: 7.0,
        noise_std: 0.9,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

fn learning_resources_preset() -> ScenarioPreset {
    let rows: [(&str, &str, f64, f64, &str); 10] = [
        ("worked-examples", "Worked examples", 0.84, 1.3, "practice"),
        (
            "interactive-notebooks",
            "Interactive notebooks",
            0.81,
            1.1,
            "practice",
        ),
        (
            "project-builds",
            "Small project builds",
            0.79,
            1.0,
            "practice",
        ),
        ("office-hours", "Office hours", 0.74, 0.8, "support"),
        (
            "visual-simulations",
            "Visual simulations",
            0.72,
            0.7,
            "exploration",
        ),
        ("short-videos", "Short videos", 0.68, 1.4, "content"),
        (
            "textbook-chapters",
            "Textbook chapters",
            0.62,
            1.1,
            "content",
        ),
        ("flashcards", "Flashcards", 0.55, 0.8, "review"),
        ("long-lectures", "Long lectures", 0.50, 1.0, "content"),
        ("discussion-board", "Discussion board", 0.46, 0.7, "support"),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::LearningResources,
        label: "Learning resources ranked from sparse student feedback".to_string(),
        default_respondents: 1200,
        min_items_per_respondent: 3,
        max_items_per_respondent: 4,
        rating_min: 1.0,
        rating_max: 5.0,
        noise_std: 0.8,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

fn movies_preset() -> ScenarioPreset {
    let rows = [
        ("seven-samurai", "Seven Samurai", 0.90, 0.32, "world-cinema"),
        ("the-godfather", "The Godfather", 0.89, 0.86, "crime"),
        ("casablanca", "Casablanca", 0.88, 0.70, "classic"),
        ("spirited-away", "Spirited Away", 0.87, 0.74, "animation"),
        (
            "in-the-mood-for-love",
            "In the Mood for Love",
            0.86,
            0.38,
            "world-cinema",
        ),
        ("parasite", "Parasite", 0.85, 0.76, "thriller"),
        (
            "mad-max-fury-road",
            "Mad Max: Fury Road",
            0.84,
            0.84,
            "action",
        ),
        (
            "into-the-spider-verse",
            "Into the Spider-Verse",
            0.83,
            0.78,
            "animation",
        ),
        ("the-matrix", "The Matrix", 0.82, 0.95, "sci-fi"),
        (
            "2001-space-odyssey",
            "2001: A Space Odyssey",
            0.81,
            0.58,
            "sci-fi",
        ),
        ("arrival", "Arrival", 0.80, 0.70, "sci-fi"),
        ("moonlight", "Moonlight", 0.79, 0.54, "drama"),
        (
            "portrait-lady-fire",
            "Portrait of a Lady on Fire",
            0.78,
            0.40,
            "drama",
        ),
        ("whiplash", "Whiplash", 0.77, 0.66, "drama"),
        ("get-out", "Get Out", 0.76, 0.78, "horror"),
        (
            "lord-of-the-rings",
            "The Lord of the Rings",
            0.75,
            1.05,
            "fantasy",
        ),
        ("alien", "Alien", 0.74, 0.82, "horror"),
        ("jaws", "Jaws", 0.73, 0.80, "thriller"),
        ("paddington-2", "Paddington 2", 0.72, 0.62, "family"),
        (
            "the-social-network",
            "The Social Network",
            0.71,
            0.68,
            "drama",
        ),
        (
            "the-princess-bride",
            "The Princess Bride",
            0.70,
            0.78,
            "comedy",
        ),
        ("toy-story", "Toy Story", 0.69, 0.92, "animation"),
        ("before-sunrise", "Before Sunrise", 0.68, 0.42, "romance"),
        (
            "blade-runner-2049",
            "Blade Runner 2049",
            0.67,
            0.56,
            "sci-fi",
        ),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::Movies,
        label: "Movies ranked from sparse watch histories and ratings".to_string(),
        default_respondents: 5000,
        min_items_per_respondent: 4,
        max_items_per_respondent: 7,
        rating_min: 1.0,
        rating_max: 10.0,
        noise_std: 1.0,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

fn travel_spots_preset() -> ScenarioPreset {
    let rows = [
        ("kyoto", "Kyoto", 0.88, 0.72, "city"),
        ("patagonia", "Patagonia", 0.87, 0.38, "nature"),
        (
            "new-zealand-south-island",
            "New Zealand South Island",
            0.86,
            0.45,
            "nature",
        ),
        (
            "iceland-ring-road",
            "Iceland Ring Road",
            0.84,
            0.58,
            "road-trip",
        ),
        ("tokyo", "Tokyo", 0.83, 0.92, "city"),
        ("rome", "Rome", 0.82, 0.90, "history"),
        ("paris", "Paris", 0.81, 1.10, "city"),
        ("machu-picchu", "Machu Picchu", 0.80, 0.54, "history"),
        ("banff", "Banff", 0.79, 0.56, "nature"),
        ("cape-town", "Cape Town", 0.78, 0.42, "city"),
        ("lisbon", "Lisbon", 0.77, 0.72, "city"),
        ("istanbul", "Istanbul", 0.76, 0.60, "history"),
        ("yosemite", "Yosemite", 0.75, 0.68, "nature"),
        ("costa-rica", "Costa Rica", 0.74, 0.64, "nature"),
        ("santorini", "Santorini", 0.73, 0.70, "coast"),
        ("amalfi-coast", "Amalfi Coast", 0.72, 0.62, "coast"),
        ("marrakesh", "Marrakesh", 0.71, 0.46, "culture"),
        ("vietnam-north", "Northern Vietnam", 0.70, 0.50, "culture"),
        ("galapagos", "Galapagos Islands", 0.69, 0.26, "nature"),
        ("barcelona", "Barcelona", 0.68, 0.88, "city"),
        ("prague", "Prague", 0.67, 0.76, "city"),
        ("sedona", "Sedona", 0.66, 0.42, "nature"),
        ("tasmania", "Tasmania", 0.65, 0.24, "nature"),
        (
            "alaska-inside-passage",
            "Alaska Inside Passage",
            0.64,
            0.32,
            "nature",
        ),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::TravelSpots,
        label: "Travel spots ranked from sparse trip histories and ratings".to_string(),
        default_respondents: 3500,
        min_items_per_respondent: 3,
        max_items_per_respondent: 6,
        rating_min: 1.0,
        rating_max: 10.0,
        noise_std: 1.15,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

fn books_preset() -> ScenarioPreset {
    let rows = [
        (
            "pride-and-prejudice",
            "Pride and Prejudice",
            0.88,
            0.82,
            "classic",
        ),
        ("beloved", "Beloved", 0.87, 0.50, "literary"),
        (
            "one-hundred-years-solitude",
            "One Hundred Years of Solitude",
            0.86,
            0.48,
            "literary",
        ),
        (
            "left-hand-darkness",
            "The Left Hand of Darkness",
            0.85,
            0.36,
            "sci-fi",
        ),
        ("dune", "Dune", 0.84, 0.86, "sci-fi"),
        (
            "lord-of-the-rings-book",
            "The Lord of the Rings",
            0.83,
            0.94,
            "fantasy",
        ),
        ("east-of-eden", "East of Eden", 0.82, 0.46, "literary"),
        ("kindred", "Kindred", 0.81, 0.44, "sci-fi"),
        ("the-dispossessed", "The Dispossessed", 0.80, 0.30, "sci-fi"),
        ("circe", "Circe", 0.79, 0.58, "fantasy"),
        ("educated", "Educated", 0.78, 0.68, "memoir"),
        (
            "godel-escher-bach",
            "Godel, Escher, Bach",
            0.77,
            0.34,
            "nonfiction",
        ),
        (
            "thinking-fast-and-slow",
            "Thinking, Fast and Slow",
            0.76,
            0.76,
            "nonfiction",
        ),
        (
            "remains-of-the-day",
            "The Remains of the Day",
            0.75,
            0.38,
            "literary",
        ),
        ("the-hobbit", "The Hobbit", 0.74, 0.98, "fantasy"),
        ("neuromancer", "Neuromancer", 0.73, 0.52, "sci-fi"),
        ("lonesome-dove", "Lonesome Dove", 0.72, 0.36, "western"),
        (
            "slaughterhouse-five",
            "Slaughterhouse-Five",
            0.71,
            0.62,
            "literary",
        ),
        ("the-overstory", "The Overstory", 0.70, 0.42, "literary"),
        ("sapiens", "Sapiens", 0.69, 0.88, "nonfiction"),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::Books,
        label: "Books ranked from sparse reading histories and ratings".to_string(),
        default_respondents: 4200,
        min_items_per_respondent: 3,
        max_items_per_respondent: 6,
        rating_min: 1.0,
        rating_max: 10.0,
        noise_std: 1.05,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

fn songs_preset() -> ScenarioPreset {
    let rows = [
        (
            "a-change-is-gonna-come",
            "A Change Is Gonna Come",
            0.89,
            0.58,
            "soul",
        ),
        ("bohemian-rhapsody", "Bohemian Rhapsody", 0.88, 1.02, "rock"),
        ("respect", "Respect", 0.87, 0.86, "soul"),
        ("purple-rain", "Purple Rain", 0.86, 0.84, "pop-rock"),
        (
            "like-a-rolling-stone",
            "Like a Rolling Stone",
            0.85,
            0.70,
            "rock",
        ),
        ("superstition", "Superstition", 0.84, 0.82, "funk"),
        ("whats-going-on", "What's Going On", 0.83, 0.64, "soul"),
        ("billie-jean", "Billie Jean", 0.82, 1.08, "pop"),
        ("fast-car", "Fast Car", 0.81, 0.70, "folk-pop"),
        ("hallelujah", "Hallelujah", 0.80, 0.74, "folk-rock"),
        ("dreams", "Dreams", 0.79, 0.84, "rock"),
        (
            "smells-like-teen-spirit",
            "Smells Like Teen Spirit",
            0.78,
            0.96,
            "rock",
        ),
        ("juicy", "Juicy", 0.77, 0.64, "hip-hop"),
        ("hey-ya", "Hey Ya!", 0.76, 0.92, "pop"),
        ("crazy-in-love", "Crazy in Love", 0.75, 0.90, "pop"),
        ("no-woman-no-cry", "No Woman No Cry", 0.74, 0.74, "reggae"),
        ("paper-planes", "Paper Planes", 0.73, 0.62, "pop"),
        ("mr-brightside", "Mr. Brightside", 0.72, 0.98, "rock"),
        ("bad-guy", "Bad Guy", 0.71, 0.76, "pop"),
        ("all-too-well", "All Too Well", 0.70, 0.72, "pop"),
    ];
    ScenarioPreset {
        scenario: CollaborativeInferenceScenario::Songs,
        label: "Songs ranked from sparse listening histories and ratings".to_string(),
        default_respondents: 6000,
        min_items_per_respondent: 4,
        max_items_per_respondent: 8,
        rating_min: 1.0,
        rating_max: 10.0,
        noise_std: 1.20,
        items: rows
            .iter()
            .map(|(id, label, latent, exposure, group)| {
                grouped_item(id, label, *latent, *exposure, group)
            })
            .collect(),
    }
}

// =============================================================================
// Free helpers
// =============================================================================

fn coverage_summary(stats: &HashMap<String, ItemEvidenceStats>) -> CollaborativeInferenceCoverage {
    let rows: Vec<&ItemEvidenceStats> = stats.values().collect();
    let rating_counts: Vec<f64> = rows.iter().map(|r| r.rating_count as f64).collect();
    let comparison_counts: Vec<f64> = rows
        .iter()
        .map(|r| r.pairwise_wins + r.pairwise_losses)
        .collect();
    CollaborativeInferenceCoverage {
        items: rows.len(),
        items_with_ratings: rating_counts.iter().filter(|&&v| v > 0.0).count(),
        items_with_comparisons: comparison_counts.iter().filter(|&&v| v > 0.0).count(),
        min_ratings_per_item: rating_counts.iter().copied().fold(f64::INFINITY, f64::min),
        mean_ratings_per_item: mean(&rating_counts),
        max_ratings_per_item: rating_counts
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        min_comparisons_per_item: comparison_counts
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        mean_comparisons_per_item: mean(&comparison_counts),
        max_comparisons_per_item: comparison_counts
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    }
}

fn clone_stats(stats: &HashMap<String, ItemEvidenceStats>) -> HashMap<String, ItemEvidenceStats> {
    stats.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

fn empty_stats(item_id: &str) -> ItemEvidenceStats {
    ItemEvidenceStats {
        item_id: item_id.to_string(),
        rating_sum: 0.0,
        rating_weight: 0.0,
        rating_count: 0,
        pairwise_wins: 0.0,
        pairwise_losses: 0.0,
    }
}

fn weighted_sample_without_replacement(
    items: &[CollaborativeInferenceItem],
    count: usize,
    rng: &mut dyn RandomSource,
) -> Vec<CollaborativeInferenceItem> {
    let mut pool: Vec<CollaborativeInferenceItem> = items.to_vec();
    let mut out: Vec<CollaborativeInferenceItem> = Vec::new();
    let n = count.min(pool.len());
    for _ in 0..n {
        let total: f64 = pool
            .iter()
            .map(|item| item.exposure.unwrap_or(1.0).max(0.0))
            .sum();
        let mut draw = rng.next_float()
            * (if total > 0.0 {
                total
            } else {
                pool.len() as f64
            });
        let mut idx = 0usize;
        while idx < pool.len() {
            draw -= if total > 0.0 {
                pool[idx].exposure.unwrap_or(1.0).max(0.0)
            } else {
                1.0
            };
            if draw <= 0.0 {
                break;
            }
            idx += 1;
        }
        let pick_idx = idx.min(pool.len() - 1);
        out.push(pool.remove(pick_idx));
    }
    out
}

fn normal01(rng: &mut dyn RandomSource) -> f64 {
    let u1 = 1e-12_f64.max(rng.next_float());
    let u2 = 1e-12_f64.max(rng.next_float());
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn random_int(rng: &mut dyn RandomSource, lo: usize, hi: usize) -> usize {
    lo + (rng.next_float() * ((hi - lo + 1) as f64)).floor() as usize
}

fn synthetic_age(rng: &mut dyn RandomSource) -> f64 {
    (18.0 + rng.next_float().powf(1.45) * 47.0).floor()
}

fn finite_positive(value: Option<f64>, fallback: f64) -> f64 {
    match value {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => fallback,
    }
}

fn finite_non_negative(value: Option<f64>) -> Option<f64> {
    match value {
        Some(v) if v.is_finite() && v >= 0.0 => Some(v),
        _ => None,
    }
}

fn normalized_log(value: f64, reference: f64) -> f64 {
    1.0_f64.min(value.max(0.0).ln_1p() / reference.max(1.0).ln_1p())
}

fn cap_multiplier(value: f64, base: f64, max_multiplier: f64) -> f64 {
    let cap = base.max(base * max_multiplier);
    value.min(cap).max(0.0)
}

fn clamp01(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.5;
    }
    x.clamp(0.0, 1.0)
}

/// JavaScript `Math.round` semantics (round half toward +infinity).
fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

fn round_to(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    js_round(x * m) / m
}

fn unique(values: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for v in values {
        if seen.insert(v.clone()) {
            out.push(v.clone());
        }
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sorted keys of a string-keyed map (gives a deterministic iteration order;
/// see module docs).
fn sorted_keys(map: &HashMap<String, f64>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Smoke tests on small scenarios verifying the pipeline ranks every item
    //! with finite probability scores and conserves respondents.

    use super::*;

    #[test]
    fn custom_scenario_ranks_all_items() {
        let params = CollaborativeInferenceParams {
            scenario: Some(CollaborativeInferenceScenario::Custom),
            respondent_count: Some(60),
            respondents_per_tick: Some(20),
            credibility_passes: Some(1),
            ..Default::default()
        };
        let result = run_collaborative_inference(params);
        assert_eq!(result.rankings.len(), 3);
        assert!(result
            .rankings
            .iter()
            .all(|r| r.score.is_finite() && r.score >= 0.0 && r.score <= 1.0));
        assert_eq!(result.coverage.items, 3);
        let mut ranks: Vec<usize> = result.rankings.iter().map(|r| r.rank).collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 2, 3]);
    }

    fn item(id: &str, latent: f64) -> CollaborativeInferenceItem {
        CollaborativeInferenceItem {
            id: id.to_string(),
            label: None,
            group: None,
            latent_utility: Some(latent),
            exposure: Some(1.0),
            prior_score: None,
        }
    }

    fn response(id: &str, ratings: &[(&str, f64)]) -> CollaborativeInferenceResponse {
        let mut m = HashMap::new();
        for (k, v) in ratings {
            m.insert(k.to_string(), *v);
        }
        CollaborativeInferenceResponse {
            id: Some(id.to_string()),
            ratings: Some(m),
            ..Default::default()
        }
    }

    fn response_with_history(
        id: &str,
        ratings: &[(&str, f64)],
        exposure_order: &[&str],
        rating_ages: &[(&str, f64)],
    ) -> CollaborativeInferenceResponse {
        let mut response = response(id, ratings);
        response.age = Some(36.0);
        response.exposure_order = Some(exposure_order.iter().map(|s| s.to_string()).collect());
        let mut ages = HashMap::new();
        for (item_id, age) in rating_ages {
            ages.insert(item_id.to_string(), *age);
        }
        response.rating_ages = Some(ages);
        response
    }

    fn score_for(result: &CollaborativeInferenceResult, item_id: &str) -> f64 {
        result
            .rankings
            .iter()
            .find(|row| row.item_id == item_id)
            .map(|row| row.score)
            .unwrap_or_else(|| panic!("missing score for {item_id}"))
    }

    #[test]
    fn explicit_responses_are_processed() {
        let params = CollaborativeInferenceParams {
            scenario: Some(CollaborativeInferenceScenario::Custom),
            items: Some(vec![item("a", 0.6), item("b", 0.5), item("c", 0.4)]),
            responses: Some(vec![
                response("r1", &[("a", 9.0), ("b", 5.0), ("c", 3.0)]),
                response("r2", &[("a", 8.0), ("b", 6.0), ("c", 4.0)]),
            ]),
            credibility_weighting: Some(false),
            ..Default::default()
        };
        let result = run_collaborative_inference(params);
        assert!(!result.synthetic);
        assert_eq!(result.respondents_processed, 2);
        assert_eq!(result.rankings.len(), 3);
        // 'a' is rated highest by both respondents, so it should rank first.
        assert_eq!(result.rankings[0].item_id, "a");
        assert!(result.invalid_evidence.is_empty());
    }

    #[test]
    fn later_and_older_item_history_increases_item_specific_credibility() {
        let make_result = |target_exposure: &[&str], target_ages: &[(&str, f64)]| {
            run_collaborative_inference(CollaborativeInferenceParams {
                scenario: Some(CollaborativeInferenceScenario::Custom),
                items: Some(vec![
                    item("rust", 0.6),
                    item("python", 0.6),
                    item("java", 0.6),
                ]),
                responses: Some(vec![
                    response_with_history(
                        "target",
                        &[("rust", 9.0), ("python", 5.0), ("java", 6.0)],
                        target_exposure,
                        target_ages,
                    ),
                    response("counter", &[("rust", 5.0), ("python", 8.0), ("java", 6.0)]),
                ]),
                credibility_passes: Some(1),
                age_weight_strength: Some(0.0),
                experience_weight_strength: Some(0.0),
                high_rated_breadth_strength: Some(0.0),
                exposure_order_weight_strength: Some(0.8),
                rating_age_weight_strength: Some(0.8),
                max_credibility_multiplier: Some(5.0),
                shrinkage: Some(2.0),
                ..Default::default()
            })
        };

        let rust_last = make_result(
            &["python", "java", "rust"],
            &[("python", 18.0), ("java", 25.0), ("rust", 36.0)],
        );
        let rust_first = make_result(
            &["rust", "python", "java"],
            &[("rust", 18.0), ("python", 25.0), ("java", 36.0)],
        );

        assert!(
            score_for(&rust_last, "rust") > score_for(&rust_first, "rust") + 0.03,
            "Rust should carry more evidence when it was learned and rated later"
        );
    }
}
