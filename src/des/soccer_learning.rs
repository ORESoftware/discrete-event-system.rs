//! Durable soccer self-play learning primitives.
//!
//! The soccer simulator owns the MDP/POMDP update mechanics. This module owns
//! the cross-run layer: outcome scoring, policy deltas, weighted merge, simple
//! evolutionary spawning, and a queue runner that keeps worker slots full.

use std::collections::{BTreeMap, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::des::general::soccer::{
    MatchConfig, MatchSummary, SoccerMatch, SoccerNeuralNetworkSnapshot, SoccerQEntry,
    SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningSummary,
    SoccerTacticalLearningWeights, SoccerTeamQPolicies, Team,
};

pub const SOCCER_LEARNING_FIXED_SCALE: i64 = 1_000_000;
pub const SOCCER_POLICY_STATUS_ACTIVE: &str = "active";
pub const SOCCER_POLICY_STATUS_ARCHIVED: &str = "archived";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoccerLearningPolicyEntryKind {
    Action,
    Target,
}

impl SoccerLearningPolicyEntryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Target => "target",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoccerLearningOutcome {
    Win,
    Draw,
    Loss,
}

impl SoccerLearningOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Draw => "draw",
            Self::Loss => "loss",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningTeamScore {
    pub team: Team,
    pub goals_for: u32,
    pub goals_against: u32,
    pub goal_diff: i32,
    pub outcome: SoccerLearningOutcome,
    pub merge_weight: f64,
    pub merge_weight_micros: i64,
    pub fitness: f64,
    pub fitness_micros: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningRunScore {
    pub home: SoccerLearningTeamScore,
    pub away: SoccerLearningTeamScore,
    pub match_fitness: f64,
    pub match_fitness_micros: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningPolicyDeltaEntry {
    pub team: Team,
    pub entry_kind: SoccerLearningPolicyEntryKind,
    pub state_hash: String,
    pub state_key: SoccerQStateKey,
    pub state_json: Value,
    pub action: String,
    pub target_fine_cell_id: i32,
    pub target_tactical_cell_id: i32,
    pub target_macro_cell_id: i32,
    pub target_root_cell_id: i32,
    pub before_value: f64,
    pub after_value: f64,
    pub value_delta: f64,
    pub before_value_micros: i64,
    pub after_value_micros: i64,
    pub value_delta_micros: i64,
    pub visit_delta: u32,
    pub merge_weight: f64,
    pub merge_weight_micros: i64,
    pub effective_visit_weight: f64,
    pub effective_visit_micros: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningPolicyDelta {
    pub entries: Vec<SoccerLearningPolicyDeltaEntry>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningCompletedGame {
    pub episode: usize,
    pub seed: u64,
    pub summary: MatchSummary,
    pub episode_summary: SoccerSelfPlayEpisodeSummary,
    pub tactical_summary: SoccerTacticalLearningSummary,
    pub starting_tactical_learning: SoccerTacticalLearningWeights,
    pub policies: SoccerTeamQPolicies,
    pub score: SoccerLearningRunScore,
    pub delta: SoccerLearningPolicyDelta,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningQueueRunnerConfig {
    pub games: usize,
    pub parallel_games: usize,
    pub base_seed: u32,
    pub match_config: MatchConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub neural_drain_timeout: Duration,
    pub options: SoccerQPolicyOptions,
    pub prune_action_entries_per_team: usize,
    pub prune_target_entries_per_team: usize,
    pub min_policy_visits: u32,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningQueueReport {
    pub completed_games: usize,
    pub failed_games: usize,
    pub elapsed_seconds: f64,
    pub total_home_goals: u32,
    pub total_away_goals: u32,
    pub tactical_summary: SoccerTacticalLearningSummary,
    pub final_policy_entries: usize,
    pub final_target_entries: usize,
    pub episode_summaries: Vec<SoccerSelfPlayEpisodeSummary>,
    pub final_policies: SoccerTeamQPolicies,
    pub latest_neural_network: Option<SoccerNeuralNetworkSnapshot>,
}

pub enum SoccerLearningQueueEvent<'a> {
    StartingBatch {
        next_episode: usize,
        match_config: &'a mut MatchConfig,
        policies: &'a mut SoccerTeamQPolicies,
        neural_network: &'a mut Option<SoccerNeuralNetworkSnapshot>,
    },
    CompletedGame {
        game: &'a SoccerLearningCompletedGame,
        merged_policies: &'a mut SoccerTeamQPolicies,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerEvolutionOptions {
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub crossover_rate: f64,
    pub exploration_rate: f64,
    pub exploration_scale: f64,
    pub elite_weight_floor: f64,
    pub population_size: usize,
    pub seed: u64,
}

impl Default for SoccerEvolutionOptions {
    fn default() -> Self {
        Self {
            mutation_rate: 0.04,
            mutation_scale: 0.22,
            crossover_rate: 0.45,
            exploration_rate: 0.06,
            exploration_scale: 0.55,
            elite_weight_floor: 0.05,
            population_size: 8,
            seed: 2026,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerTacticalLearningGenomeParent<'a> {
    pub summary: &'a SoccerTacticalLearningSummary,
    pub weights: &'a SoccerTacticalLearningWeights,
    pub fitness: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerPostgresPolicyRefreshCheck<'a> {
    pub current_policy_version_id: Option<&'a str>,
    pub current_generation: i32,
    pub current_updated_at_micros: i64,
    pub current_neural_network_present: bool,
    pub latest_policy_version_id: &'a str,
    pub latest_generation: i32,
    pub latest_updated_at_micros: i64,
    pub latest_neural_network_present: bool,
    pub local_tactical_evolved_since_pg_refresh: bool,
    pub postgres_tactical_learning_authoritative: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoccerPostgresPolicyRefreshDecision {
    pub refresh_policy: bool,
    pub apply_tactical_learning: bool,
    pub same_policy_version: bool,
    pub same_policy_newer_revision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoccerPostgresNewSimRefreshPlan {
    pub refresh_from_postgres: bool,
    pub flush_pending_policy_versions: bool,
    pub wait_for_async_policy_versions: bool,
}

pub fn soccer_postgres_policy_refresh_decision(
    check: SoccerPostgresPolicyRefreshCheck<'_>,
) -> SoccerPostgresPolicyRefreshDecision {
    let same_policy_version =
        check.current_policy_version_id == Some(check.latest_policy_version_id);
    let same_policy_newer_revision =
        same_policy_version && check.latest_updated_at_micros > check.current_updated_at_micros;
    let newer_generation = check.latest_generation > check.current_generation;
    let different_head_at_same_generation =
        check.latest_generation == check.current_generation && !same_policy_version;

    let refresh_policy = check.current_policy_version_id.is_none()
        || newer_generation
        || different_head_at_same_generation
        || same_policy_newer_revision
        || (same_policy_version
            && !check.current_neural_network_present
            && check.latest_neural_network_present);
    let apply_tactical_learning = check.current_policy_version_id.is_none()
        || newer_generation
        || different_head_at_same_generation
        || same_policy_newer_revision
        || (same_policy_version
            && (check.postgres_tactical_learning_authoritative
                || !check.local_tactical_evolved_since_pg_refresh));

    SoccerPostgresPolicyRefreshDecision {
        refresh_policy,
        apply_tactical_learning,
        same_policy_version,
        same_policy_newer_revision,
    }
}

pub fn soccer_should_refresh_postgres_for_new_sim(
    resume_artifact_present: bool,
    refresh_with_resume_artifact: bool,
) -> bool {
    !resume_artifact_present || refresh_with_resume_artifact
}

pub fn soccer_should_flush_postgres_policy_versions_for_new_sim(
    refresh_for_new_sims: bool,
    flush_policy_versions_before_new_sim: bool,
    pending_policy_versions: usize,
) -> bool {
    refresh_for_new_sims && flush_policy_versions_before_new_sim && pending_policy_versions > 0
}

pub fn soccer_postgres_new_sim_refresh_plan(
    resume_artifact_present: bool,
    refresh_with_resume_artifact: bool,
    flush_policy_versions_before_new_sim: bool,
    pending_policy_versions: usize,
    pending_async_policy_version_batches: usize,
) -> SoccerPostgresNewSimRefreshPlan {
    let refresh_from_postgres = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact_present,
        refresh_with_resume_artifact,
    );
    let flush_pending_policy_versions = soccer_should_flush_postgres_policy_versions_for_new_sim(
        refresh_from_postgres,
        flush_policy_versions_before_new_sim,
        pending_policy_versions,
    );
    let wait_for_async_policy_versions = refresh_from_postgres
        && flush_policy_versions_before_new_sim
        && pending_async_policy_version_batches > 0;

    SoccerPostgresNewSimRefreshPlan {
        refresh_from_postgres,
        flush_pending_policy_versions,
        wait_for_async_policy_versions,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PolicyEntryKey {
    team: &'static str,
    entry_kind: SoccerLearningPolicyEntryKind,
    state_hash: String,
    state_json: String,
    action: String,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

#[derive(Clone, Debug)]
struct EntryValue {
    state_key: SoccerQStateKey,
    action: String,
    value: f64,
    visits: u32,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

#[derive(Clone, Debug)]
struct MergeAccumulator {
    state_key: SoccerQStateKey,
    action: String,
    weighted_value_sum: f64,
    effective_visits: f64,
    display_visits: u32,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

pub fn soccer_learning_to_micros(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    (value * SOCCER_LEARNING_FIXED_SCALE as f64).round() as i64
}

pub fn soccer_learning_from_micros(value: i64) -> f64 {
    value as f64 / SOCCER_LEARNING_FIXED_SCALE as f64
}

pub fn soccer_policy_version_insert_status_after_active_head(
    requested_status: &'static str,
    parent_policy_version_id: Option<&str>,
    generation: i32,
    latest_active_policy_version_id: Option<&str>,
    latest_active_generation: Option<i32>,
) -> &'static str {
    if requested_status != SOCCER_POLICY_STATUS_ACTIVE {
        return requested_status;
    }
    let Some(latest_active_policy_version_id) = latest_active_policy_version_id else {
        return SOCCER_POLICY_STATUS_ACTIVE;
    };
    if parent_policy_version_id != Some(latest_active_policy_version_id) {
        return SOCCER_POLICY_STATUS_ARCHIVED;
    }
    if latest_active_generation.is_some_and(|active_generation| active_generation >= generation) {
        return SOCCER_POLICY_STATUS_ARCHIVED;
    }
    SOCCER_POLICY_STATUS_ACTIVE
}

pub fn soccer_team_label(team: Team) -> &'static str {
    match team {
        Team::Home => "home",
        Team::Away => "away",
    }
}

pub fn soccer_learning_run_score(summary: &MatchSummary) -> SoccerLearningRunScore {
    let home = soccer_learning_team_score(Team::Home, summary.score_home, summary.score_away);
    let away = soccer_learning_team_score(Team::Away, summary.score_away, summary.score_home);
    let match_fitness = (home.fitness + away.fitness) * 0.5;
    SoccerLearningRunScore {
        home,
        away,
        match_fitness,
        match_fitness_micros: soccer_learning_to_micros(match_fitness),
    }
}

pub fn soccer_learning_team_score(
    team: Team,
    goals_for: u32,
    goals_against: u32,
) -> SoccerLearningTeamScore {
    let goal_diff = goals_for as i32 - goals_against as i32;
    let outcome = if goal_diff > 0 {
        SoccerLearningOutcome::Win
    } else if goal_diff == 0 {
        SoccerLearningOutcome::Draw
    } else {
        SoccerLearningOutcome::Loss
    };

    let margin = goal_diff.unsigned_abs() as f64;
    let base_weight = match outcome {
        SoccerLearningOutcome::Win => 1.0 + 0.22 * margin.min(6.0),
        SoccerLearningOutcome::Draw => 0.55,
        SoccerLearningOutcome::Loss => 0.20 / (1.0 + 0.35 * margin.max(1.0)),
    };
    let attacking_signal = (goals_for as f64 * 0.04).min(0.18);
    let defensive_signal = (1.0 / (1.0 + goals_against as f64 * 0.10)).clamp(0.65, 1.0);
    let merge_weight = (base_weight * defensive_signal + attacking_signal).clamp(0.035, 2.5);
    let fitness = goal_diff as f64 + goals_for as f64 * 0.20 - goals_against as f64 * 0.12;

    SoccerLearningTeamScore {
        team,
        goals_for,
        goals_against,
        goal_diff,
        outcome,
        merge_weight,
        merge_weight_micros: soccer_learning_to_micros(merge_weight),
        fitness,
        fitness_micros: soccer_learning_to_micros(fitness),
    }
}

pub fn soccer_policy_delta_entries(
    before: &SoccerTeamQPolicies,
    after: &SoccerTeamQPolicies,
    score: &SoccerLearningRunScore,
) -> SoccerLearningPolicyDelta {
    let mut entries = Vec::with_capacity(
        after
            .home
            .q_values
            .len()
            .saturating_add(after.home.target_values.len())
            .saturating_add(after.away.q_values.len())
            .saturating_add(after.away.target_values.len()),
    );
    collect_team_policy_delta(
        Team::Home,
        &before.home,
        &after.home,
        score.home.merge_weight,
        &mut entries,
    );
    collect_team_policy_delta(
        Team::Away,
        &before.away,
        &after.away,
        score.away.merge_weight,
        &mut entries,
    );
    SoccerLearningPolicyDelta { entries }
}

pub fn merge_soccer_policy_deltas(
    base: &SoccerTeamQPolicies,
    deltas: &[SoccerLearningPolicyDelta],
    prior_weight: f64,
) -> Result<SoccerTeamQPolicies, String> {
    let mut action_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
    let mut target_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();

    seed_policy_accumulators(
        Team::Home,
        &base.home,
        prior_weight,
        &mut action_accumulators,
        &mut target_accumulators,
    );
    seed_policy_accumulators(
        Team::Away,
        &base.away,
        prior_weight,
        &mut action_accumulators,
        &mut target_accumulators,
    );

    for delta in deltas {
        for entry in &delta.entries {
            let accumulator = match entry.entry_kind {
                SoccerLearningPolicyEntryKind::Action => &mut action_accumulators,
                SoccerLearningPolicyEntryKind::Target => &mut target_accumulators,
            };
            let key = policy_delta_key(entry);
            let effective_visits = entry.effective_visit_weight.max(0.0);
            if effective_visits <= 0.0 {
                continue;
            }
            let item = accumulator.entry(key).or_insert_with(|| MergeAccumulator {
                state_key: entry.state_key.clone(),
                action: entry.action.clone(),
                weighted_value_sum: 0.0,
                effective_visits: 0.0,
                display_visits: 0,
                target_fine_cell_id: entry.target_fine_cell_id,
                target_tactical_cell_id: entry.target_tactical_cell_id,
                target_macro_cell_id: entry.target_macro_cell_id,
                target_root_cell_id: entry.target_root_cell_id,
            });
            item.weighted_value_sum += entry.after_value * effective_visits;
            item.effective_visits += effective_visits;
            item.display_visits = item
                .display_visits
                .saturating_add(effective_visits.round().max(1.0) as u32);
        }
    }

    build_policies_from_accumulators(
        base.home.options.clone(),
        base.away.options.clone(),
        action_accumulators,
        target_accumulators,
    )
}

pub fn evolve_soccer_team_policies(
    parents: &[(&SoccerTeamQPolicies, f64)],
    options: SoccerEvolutionOptions,
) -> Result<SoccerTeamQPolicies, String> {
    let population_size = options.population_size.max(1);
    let mut best = None::<(SoccerTeamQPolicies, f64)>;
    for candidate_index in 0..population_size {
        let mut candidate_options = options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(options.seed, candidate_index, 0x51f1_7ea9_8d12_0b1d);
        let candidate = evolve_soccer_team_policy_candidate(parents, candidate_options)?;
        let score = soccer_team_policy_search_score(&candidate);
        if best
            .as_ref()
            .map_or(true, |(_, best_score)| score > *best_score)
        {
            best = Some((candidate, score));
        }
    }
    best.map(|(policy, _)| policy)
        .ok_or_else(|| "at least one parent policy is required".to_string())
}

fn evolve_soccer_team_policy_candidate(
    parents: &[(&SoccerTeamQPolicies, f64)],
    options: SoccerEvolutionOptions,
) -> Result<SoccerTeamQPolicies, String> {
    let Some((first_parent, _)) = parents.first() else {
        return Err("at least one parent policy is required".to_string());
    };
    let mut action_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
    let mut target_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();

    for (policy, fitness) in parents {
        let weight = fitness.max(options.elite_weight_floor).max(0.0);
        seed_policy_accumulators(
            Team::Home,
            &policy.home,
            weight,
            &mut action_accumulators,
            &mut target_accumulators,
        );
        seed_policy_accumulators(
            Team::Away,
            &policy.away,
            weight,
            &mut action_accumulators,
            &mut target_accumulators,
        );
    }

    let mut rng = DeterministicRng::new(options.seed);
    let parent_action_maps = parents
        .iter()
        .map(|(policy, _)| policy_action_entry_map(policy))
        .collect::<Vec<_>>();
    let parent_target_maps = parents
        .iter()
        .map(|(policy, _)| policy_target_entry_map(policy))
        .collect::<Vec<_>>();
    crossover_accumulators(
        &mut action_accumulators,
        &parent_action_maps,
        &mut rng,
        options,
    );
    crossover_accumulators(
        &mut target_accumulators,
        &parent_target_maps,
        &mut rng,
        options,
    );
    explore_accumulators(&mut action_accumulators, &mut rng, options);
    explore_accumulators(&mut target_accumulators, &mut rng, options);
    mutate_accumulators(&mut action_accumulators, &mut rng, options);
    mutate_accumulators(&mut target_accumulators, &mut rng, options);

    build_policies_from_accumulators(
        first_parent.home.options.clone(),
        first_parent.away.options.clone(),
        action_accumulators,
        target_accumulators,
    )
}

fn soccer_team_policy_search_score(policy: &SoccerTeamQPolicies) -> f64 {
    let (home_score, home_weight) = soccer_q_policy_search_score(&policy.home);
    let (away_score, away_weight) = soccer_q_policy_search_score(&policy.away);
    let weight = home_weight + away_weight;
    if weight <= 0.0 {
        0.0
    } else {
        (home_score + away_score) / weight
    }
}

fn soccer_q_policy_search_score(policy: &SoccerQPolicy) -> (f64, f64) {
    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;
    for entry in policy.entries() {
        let weight = f64::from(entry.visits.max(1)).sqrt();
        weighted_score += entry.value.clamp(-120.0, 120.0) * weight;
        total_weight += weight;
    }
    for entry in policy.target_entries() {
        let weight = f64::from(entry.visits.max(1)).sqrt();
        weighted_score += entry.value.clamp(-120.0, 120.0) * weight;
        total_weight += weight;
    }
    (weighted_score, total_weight)
}

pub fn evolve_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    parents: &[(&SoccerTacticalLearningSummary, f64)],
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let Some(weighted_summary) = weighted_tactical_evolution_summary(parents, options) else {
        return base.clone();
    };
    let search_options =
        adapt_soccer_evolution_options_for_tactical_search(options, &weighted_summary);
    let population_size = search_options.population_size.max(1);
    let mut best = base.clone();
    let mut best_score = soccer_tactical_weight_search_score(&best, &weighted_summary);
    search_soccer_tactical_strategy_candidates(base, &weighted_summary, |candidate| {
        keep_best_soccer_tactical_candidate(
            &mut best,
            &mut best_score,
            candidate,
            &weighted_summary,
        );
    });
    for candidate_index in 0..population_size {
        let mut candidate_options = search_options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(search_options.seed, candidate_index, 0x7ac7_1ca1_5eed_c0de);
        let candidate =
            evolve_soccer_tactical_learning_candidate(base, &weighted_summary, candidate_options);
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
}

pub fn evolve_soccer_tactical_learning_weights_from_genomes(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let summary_parents = parents
        .iter()
        .map(|parent| (parent.summary, parent.fitness))
        .collect::<Vec<_>>();
    let Some(weighted_summary) = weighted_tactical_evolution_summary(&summary_parents, options)
    else {
        return base.clone();
    };
    let search_options =
        adapt_soccer_evolution_options_for_tactical_search(options, &weighted_summary);
    let population_size = search_options.population_size.max(1);
    let mut best = base.clone();
    let mut best_score = soccer_tactical_weight_search_score(&best, &weighted_summary);
    search_soccer_tactical_strategy_candidates(base, &weighted_summary, |candidate| {
        keep_best_soccer_tactical_candidate(
            &mut best,
            &mut best_score,
            candidate,
            &weighted_summary,
        );
    });

    for parent in parents {
        if !parent.fitness.is_finite() {
            continue;
        }
        let candidate = clamp_soccer_tactical_learning_weights(parent.weights);
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }

    search_soccer_tactical_genome_blend_candidates(
        base,
        parents,
        search_options,
        &weighted_summary,
        |candidate| {
            keep_best_soccer_tactical_candidate(
                &mut best,
                &mut best_score,
                candidate,
                &weighted_summary,
            );
        },
    );

    for candidate_index in 0..population_size {
        let mut candidate_options = search_options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(search_options.seed, candidate_index, 0x7ac7_1ca1_5eed_c0de);
        let mut rng = DeterministicRng::new(candidate_options.seed ^ 0x671e_d1ce_5eed_ba5e);
        let candidate_base =
            crossover_soccer_tactical_learning_weights(base, parents, search_options, &mut rng);
        let candidate = evolve_soccer_tactical_learning_candidate(
            &candidate_base,
            &weighted_summary,
            candidate_options,
        );
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
}

fn search_soccer_tactical_genome_blend_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    weighted_summary: &SoccerTacticalLearningSummary,
    mut visit: F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let blend_enabled =
        parents.len() >= 2 && (options.population_size > 1 || options.crossover_rate > 0.0);
    if !blend_enabled {
        return;
    }

    let Some(centroid) = weighted_soccer_tactical_genome_centroid(parents, options) else {
        return;
    };
    visit(centroid.clone());

    let pressure = soccer_tactical_search_pressure(weighted_summary);
    if pressure <= 1e-12 {
        return;
    }
    let scale = 1.0 + pressure * 0.75;
    visit(extrapolate_soccer_tactical_learning_weights(
        base, &centroid, scale,
    ));
    search_soccer_tactical_genome_archetype_candidates(
        &centroid,
        weighted_summary,
        pressure,
        &mut visit,
    );
    search_soccer_tactical_gp_program_candidates(
        base,
        &centroid,
        weighted_summary,
        pressure,
        &mut visit,
    );
    search_soccer_tactical_mutated_gp_program_candidates(
        base,
        &centroid,
        weighted_summary,
        pressure,
        options,
        &mut visit,
    );
}

fn search_soccer_tactical_genome_archetype_candidates<F>(
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_pressure =
        (attack_width_gap * 0.42 + attack_flank_gap * 0.42 + attack_spacing_gap * 0.16)
            .clamp(0.0, 1.0);
    let defense_pressure = (defense_contract_gap * 0.50
        + defense_spacing_gap * 0.18
        + defense_ball_gap * 0.16
        + press_gap * 0.16)
        .clamp(0.0, 1.0);

    if attack_pressure > 1e-12 {
        let mut flank_switch = centroid.clone();
        flank_switch.attack_width_delta_weight += attack_width_gap * (0.16 + pressure * 0.10);
        flank_switch.attack_flank_lane_weight += attack_flank_gap * (0.26 + pressure * 0.16);
        flank_switch.attack_spacing_delta_weight +=
            attack_spacing_gap * (0.10 + attack_pressure * 0.08);
        flank_switch.attack_spacing_score_weight += attack_spacing_gap * 0.05;
        visit(clamp_soccer_tactical_learning_weights(&flank_switch));

        let mut support_spread = centroid.clone();
        support_spread.attack_spacing_delta_weight += attack_spacing_gap * (0.22 + pressure * 0.08);
        support_spread.attack_spacing_score_weight += attack_spacing_gap * 0.10;
        support_spread.attack_width_delta_weight += attack_width_gap * 0.10;
        support_spread.attack_flank_lane_weight += attack_flank_gap * 0.14;
        visit(clamp_soccer_tactical_learning_weights(&support_spread));
    }

    if defense_pressure > 1e-12 {
        let mut compact_block = centroid.clone();
        compact_block.defense_contract_delta_weight +=
            defense_contract_gap * (0.26 + pressure * 0.14);
        compact_block.defense_compactness_score_weight +=
            defense_contract_gap * (0.18 + defense_pressure * 0.08);
        compact_block.defense_spacing_delta_weight += defense_spacing_gap * 0.08;
        compact_block.defense_ball_depth_score_weight +=
            defense_ball_gap * (0.08 + defense_pressure * 0.04);
        visit(clamp_soccer_tactical_learning_weights(&compact_block));

        let mut press_contract = centroid.clone();
        press_contract.defense_contract_delta_weight +=
            defense_contract_gap * (0.18 + press_gap * 0.10);
        press_contract.defender_midfielder_press_weight +=
            press_gap * (0.09 + defense_pressure * 0.04);
        press_contract.midfielder_press_weight += press_gap * (0.08 + defense_pressure * 0.03);
        press_contract.defense_ball_depth_score_weight += defense_ball_gap * 0.05;
        visit(clamp_soccer_tactical_learning_weights(&press_contract));
    }

    if attack_pressure > 1e-12 && defense_pressure > 1e-12 {
        let mut two_phase_shape = centroid.clone();
        two_phase_shape.attack_width_delta_weight += attack_width_gap * 0.13;
        two_phase_shape.attack_flank_lane_weight += attack_flank_gap * (0.20 + pressure * 0.10);
        two_phase_shape.attack_spacing_delta_weight += attack_spacing_gap * 0.08;
        two_phase_shape.defense_contract_delta_weight +=
            defense_contract_gap * (0.21 + pressure * 0.11);
        two_phase_shape.defense_compactness_score_weight += defense_contract_gap * 0.14;
        two_phase_shape.defense_spacing_delta_weight += defense_spacing_gap * 0.06;
        visit(clamp_soccer_tactical_learning_weights(&two_phase_shape));
    }
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpGene {
    AttackSpacingDelta,
    AttackSpacingScore,
    AttackWidthDelta,
    AttackWidthScore,
    AttackFlankLane,
    DefenseSpacingDelta,
    DefenseSpacingScore,
    DefenseContractDelta,
    DefenseCompactnessScore,
    DefenseBallDepthScore,
    DefenderMidfielderPress,
    MidfielderPress,
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpSignal {
    AttackWidthGap,
    AttackFlankGap,
    AttackSpacingGap,
    DefenseContractGap,
    DefenseSpacingGap,
    DefenseBallGap,
    PressGap,
    Pressure,
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpExpr {
    Signal(SoccerTacticalGpSignal),
    Product(SoccerTacticalGpSignal, SoccerTacticalGpSignal),
    Blend(SoccerTacticalGpSignal, SoccerTacticalGpSignal, f64),
    Max(SoccerTacticalGpSignal, SoccerTacticalGpSignal),
}

#[derive(Clone, Copy, Debug)]
struct SoccerTacticalGpInstruction {
    gene: SoccerTacticalGpGene,
    expr: SoccerTacticalGpExpr,
    scale: f64,
}

const ATTACK_SOCCER_TACTICAL_GP_GENES: [SoccerTacticalGpGene; 5] = [
    SoccerTacticalGpGene::AttackSpacingDelta,
    SoccerTacticalGpGene::AttackSpacingScore,
    SoccerTacticalGpGene::AttackWidthDelta,
    SoccerTacticalGpGene::AttackWidthScore,
    SoccerTacticalGpGene::AttackFlankLane,
];

const DEFENSE_SOCCER_TACTICAL_GP_GENES: [SoccerTacticalGpGene; 7] = [
    SoccerTacticalGpGene::DefenseSpacingDelta,
    SoccerTacticalGpGene::DefenseSpacingScore,
    SoccerTacticalGpGene::DefenseContractDelta,
    SoccerTacticalGpGene::DefenseCompactnessScore,
    SoccerTacticalGpGene::DefenseBallDepthScore,
    SoccerTacticalGpGene::DefenderMidfielderPress,
    SoccerTacticalGpGene::MidfielderPress,
];

const ATTACK_SOCCER_TACTICAL_GP_SIGNALS: [SoccerTacticalGpSignal; 4] = [
    SoccerTacticalGpSignal::AttackWidthGap,
    SoccerTacticalGpSignal::AttackFlankGap,
    SoccerTacticalGpSignal::AttackSpacingGap,
    SoccerTacticalGpSignal::Pressure,
];

const DEFENSE_SOCCER_TACTICAL_GP_SIGNALS: [SoccerTacticalGpSignal; 5] = [
    SoccerTacticalGpSignal::DefenseContractGap,
    SoccerTacticalGpSignal::DefenseSpacingGap,
    SoccerTacticalGpSignal::DefenseBallGap,
    SoccerTacticalGpSignal::PressGap,
    SoccerTacticalGpSignal::Pressure,
];

const WIDE_FLANK_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.55,
        ),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
        scale: 0.10,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::AttackFlankGap,
            SoccerTacticalGpSignal::AttackWidthGap,
        ),
        scale: 0.42,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackSpacingGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.24,
    },
];

const SUPPORT_SPACING_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackSpacingGap,
            SoccerTacticalGpSignal::AttackWidthGap,
            0.70,
        ),
        scale: 0.32,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackSpacingGap),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.18,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackFlankGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.20,
    },
];

const COMPACT_BLOCK_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::DefenseBallGap,
            0.70,
        ),
        scale: 0.44,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseCompactnessScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseSpacingDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseSpacingGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseBallDepthScore,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseBallGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.20,
    },
];

const PRESS_FUNNEL_GP_PROGRAM: [SoccerTacticalGpInstruction; 5] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::PressGap,
        ),
        scale: 0.28,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseSpacingScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseSpacingGap),
        scale: 0.08,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenderMidfielderPress,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::PressGap,
            SoccerTacticalGpSignal::DefenseContractGap,
            0.65,
        ),
        scale: 0.16,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::MidfielderPress,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::PressGap),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseBallDepthScore,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::DefenseBallGap,
            SoccerTacticalGpSignal::PressGap,
            0.60,
        ),
        scale: 0.12,
    },
];

const TWO_PHASE_GP_PROGRAM: [SoccerTacticalGpInstruction; 5] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.50,
        ),
        scale: 0.20,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackFlankGap),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackSpacingGap),
        scale: 0.10,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
        scale: 0.34,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseCompactnessScore,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.22,
    },
];

fn search_soccer_tactical_gp_program_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    if pressure <= 1e-12 {
        return;
    }

    let programs: [&[SoccerTacticalGpInstruction]; 5] = [
        &WIDE_FLANK_GP_PROGRAM,
        &SUPPORT_SPACING_GP_PROGRAM,
        &COMPACT_BLOCK_GP_PROGRAM,
        &PRESS_FUNNEL_GP_PROGRAM,
        &TWO_PHASE_GP_PROGRAM,
    ];
    for seed in [base, centroid] {
        for program in programs {
            let mut candidate = seed.clone();
            for instruction in program {
                apply_soccer_tactical_gp_instruction(
                    &mut candidate,
                    *instruction,
                    weighted_summary,
                    pressure,
                );
            }
            visit(clamp_soccer_tactical_learning_weights(&candidate));
        }
    }
}

fn search_soccer_tactical_mutated_gp_program_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    options: SoccerEvolutionOptions,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    if pressure <= 1e-12 {
        return;
    }
    let search_enabled = options.population_size > 1
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0
        || options.crossover_rate > 0.0
        || options.exploration_rate > 0.0
        || options.exploration_scale > 0.0;
    if !search_enabled {
        return;
    }

    let candidate_count = options.population_size.max(2).min(64);
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    let scale_boost = 1.0
        + options.mutation_scale.max(0.0).min(1.5) * 0.35
        + options.exploration_scale.max(0.0).min(1.5) * 0.25;
    for candidate_index in 0..candidate_count {
        let mut rng = DeterministicRng::new(candidate_seed(
            options.seed,
            candidate_index,
            0x67a5_1c9d_7ac7_1ca1,
        ));
        let mut candidate = if candidate_index % 2 == 0 || rng.next_f64() < crossover_rate {
            centroid.clone()
        } else {
            base.clone()
        };
        seed_soccer_tactical_gp_mutation_bias(
            &mut candidate,
            weighted_summary,
            pressure,
            candidate_index,
            scale_boost,
        );

        let program_len = 3 + rng.next_usize(4);
        for _ in 0..program_len {
            let instruction = random_soccer_tactical_gp_instruction(
                weighted_summary,
                pressure,
                scale_boost,
                &mut rng,
            );
            apply_soccer_tactical_gp_instruction(
                &mut candidate,
                instruction,
                weighted_summary,
                pressure,
            );
        }
        visit(clamp_soccer_tactical_learning_weights(&candidate));
    }
}

fn seed_soccer_tactical_gp_mutation_bias(
    weights: &mut SoccerTacticalLearningWeights,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    candidate_index: usize,
    scale_boost: f64,
) {
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    if candidate_index % 3 == 0 {
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackWidthDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
                scale: (0.10 + attack_width_gap * 0.16) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackFlankLane,
                expr: SoccerTacticalGpExpr::Blend(
                    SoccerTacticalGpSignal::AttackFlankGap,
                    SoccerTacticalGpSignal::AttackWidthGap,
                    0.70,
                ),
                scale: (0.12 + attack_flank_gap * 0.18) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenseContractDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
                scale: (0.12 + defense_contract_gap * 0.18) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenseCompactnessScore,
                expr: SoccerTacticalGpExpr::Product(
                    SoccerTacticalGpSignal::DefenseContractGap,
                    SoccerTacticalGpSignal::Pressure,
                ),
                scale: (0.08 + defense_contract_gap * 0.12) * scale_boost,
            },
            summary,
            pressure,
        );
    } else if candidate_index % 3 == 1 {
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackWidthDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
                scale: (0.10 + attack_width_gap * 0.16) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenderMidfielderPress,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::PressGap),
                scale: (0.06 + press_gap * 0.10) * scale_boost,
            },
            summary,
            pressure,
        );
    }
}

fn random_soccer_tactical_gp_instruction(
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    scale_boost: f64,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpInstruction {
    let gene = random_soccer_tactical_gp_gene(summary, rng);
    let attack_gene = soccer_tactical_gp_gene_is_attack(gene);
    let expr = random_soccer_tactical_gp_expr(attack_gene, rng);
    let scale = (0.05 + rng.next_f64() * 0.34) * (0.65 + pressure * 0.75) * scale_boost;
    SoccerTacticalGpInstruction { gene, expr, scale }
}

fn random_soccer_tactical_gp_gene(
    summary: &SoccerTacticalLearningSummary,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpGene {
    let attack_pressure = ((1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0) * 0.35
        + (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0) * 0.45
        + (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0) * 0.20)
        .max(1e-12);
    let defense_pressure = ((1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0) * 0.48
        + (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0) * 0.18
        + (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0) * 0.17
        + (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0) * 0.17)
        .max(1e-12);
    let attack_probability = attack_pressure / (attack_pressure + defense_pressure);
    if rng.next_f64() < attack_probability {
        ATTACK_SOCCER_TACTICAL_GP_GENES[rng.next_usize(ATTACK_SOCCER_TACTICAL_GP_GENES.len())]
    } else {
        DEFENSE_SOCCER_TACTICAL_GP_GENES[rng.next_usize(DEFENSE_SOCCER_TACTICAL_GP_GENES.len())]
    }
}

fn random_soccer_tactical_gp_expr(
    attack_gene: bool,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpExpr {
    let first = random_soccer_tactical_gp_signal(attack_gene, rng);
    match rng.next_usize(4) {
        0 => SoccerTacticalGpExpr::Signal(first),
        1 => {
            SoccerTacticalGpExpr::Product(first, random_soccer_tactical_gp_signal(attack_gene, rng))
        }
        2 => SoccerTacticalGpExpr::Blend(
            first,
            random_soccer_tactical_gp_signal(!attack_gene, rng),
            0.25 + rng.next_f64() * 0.50,
        ),
        _ => SoccerTacticalGpExpr::Max(first, random_soccer_tactical_gp_signal(!attack_gene, rng)),
    }
}

fn random_soccer_tactical_gp_signal(
    attack_signal: bool,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpSignal {
    if attack_signal {
        ATTACK_SOCCER_TACTICAL_GP_SIGNALS[rng.next_usize(ATTACK_SOCCER_TACTICAL_GP_SIGNALS.len())]
    } else {
        DEFENSE_SOCCER_TACTICAL_GP_SIGNALS[rng.next_usize(DEFENSE_SOCCER_TACTICAL_GP_SIGNALS.len())]
    }
}

fn soccer_tactical_gp_gene_is_attack(gene: SoccerTacticalGpGene) -> bool {
    matches!(
        gene,
        SoccerTacticalGpGene::AttackSpacingDelta
            | SoccerTacticalGpGene::AttackSpacingScore
            | SoccerTacticalGpGene::AttackWidthDelta
            | SoccerTacticalGpGene::AttackWidthScore
            | SoccerTacticalGpGene::AttackFlankLane
    )
}

fn apply_soccer_tactical_gp_instruction(
    weights: &mut SoccerTacticalLearningWeights,
    instruction: SoccerTacticalGpInstruction,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) {
    let signal = soccer_tactical_gp_expr_value(instruction.expr, summary, pressure);
    if signal <= 0.0 || !signal.is_finite() {
        return;
    }
    adjust_soccer_tactical_gp_gene(weights, instruction.gene, instruction.scale * signal);
}

fn soccer_tactical_gp_expr_value(
    expr: SoccerTacticalGpExpr,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) -> f64 {
    match expr {
        SoccerTacticalGpExpr::Signal(signal) => {
            soccer_tactical_gp_signal_value(signal, summary, pressure)
        }
        SoccerTacticalGpExpr::Product(left, right) => {
            soccer_tactical_gp_signal_value(left, summary, pressure)
                * soccer_tactical_gp_signal_value(right, summary, pressure)
        }
        SoccerTacticalGpExpr::Blend(left, right, mix) => {
            let mix = mix.clamp(0.0, 1.0);
            soccer_tactical_gp_signal_value(left, summary, pressure) * mix
                + soccer_tactical_gp_signal_value(right, summary, pressure) * (1.0 - mix)
        }
        SoccerTacticalGpExpr::Max(left, right) => {
            soccer_tactical_gp_signal_value(left, summary, pressure)
                .max(soccer_tactical_gp_signal_value(right, summary, pressure))
        }
    }
    .clamp(0.0, 1.0)
}

fn soccer_tactical_gp_signal_value(
    signal: SoccerTacticalGpSignal,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) -> f64 {
    match signal {
        SoccerTacticalGpSignal::AttackWidthGap => 1.0 - summary.mean_attack_width_score,
        SoccerTacticalGpSignal::AttackFlankGap => 1.0 - summary.mean_attack_flank_lane_score,
        SoccerTacticalGpSignal::AttackSpacingGap => 1.0 - summary.mean_attack_spacing_score,
        SoccerTacticalGpSignal::DefenseContractGap => 1.0 - summary.mean_defense_contract_score,
        SoccerTacticalGpSignal::DefenseSpacingGap => 1.0 - summary.mean_defense_spacing_score,
        SoccerTacticalGpSignal::DefenseBallGap => 1.0 - summary.mean_defense_ball_gap_score,
        SoccerTacticalGpSignal::PressGap => 1.0 - summary.mean_defense_role_press_score,
        SoccerTacticalGpSignal::Pressure => pressure,
    }
    .clamp(0.0, 1.0)
}

fn adjust_soccer_tactical_gp_gene(
    weights: &mut SoccerTacticalLearningWeights,
    gene: SoccerTacticalGpGene,
    delta: f64,
) {
    match gene {
        SoccerTacticalGpGene::AttackSpacingDelta => {
            weights.attack_spacing_delta_weight += delta;
        }
        SoccerTacticalGpGene::AttackSpacingScore => {
            weights.attack_spacing_score_weight += delta;
        }
        SoccerTacticalGpGene::AttackWidthDelta => {
            weights.attack_width_delta_weight += delta;
        }
        SoccerTacticalGpGene::AttackWidthScore => {
            weights.attack_width_score_weight += delta;
        }
        SoccerTacticalGpGene::AttackFlankLane => {
            weights.attack_flank_lane_weight += delta;
        }
        SoccerTacticalGpGene::DefenseSpacingDelta => {
            weights.defense_spacing_delta_weight += delta;
        }
        SoccerTacticalGpGene::DefenseSpacingScore => {
            weights.defense_spacing_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenseContractDelta => {
            weights.defense_contract_delta_weight += delta;
        }
        SoccerTacticalGpGene::DefenseCompactnessScore => {
            weights.defense_compactness_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenseBallDepthScore => {
            weights.defense_ball_depth_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenderMidfielderPress => {
            weights.defender_midfielder_press_weight += delta;
        }
        SoccerTacticalGpGene::MidfielderPress => {
            weights.midfielder_press_weight += delta;
        }
    }
}

fn keep_best_soccer_tactical_candidate(
    best: &mut SoccerTacticalLearningWeights,
    best_score: &mut f64,
    candidate: SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
) {
    let candidate = clamp_soccer_tactical_learning_weights(&candidate);
    let score = soccer_tactical_weight_search_score(&candidate, weighted_summary);
    if score > *best_score {
        *best_score = score;
        *best = candidate;
    }
}

fn search_soccer_tactical_strategy_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    mut visit: F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_pressure =
        (attack_width_gap * 0.42 + attack_flank_gap * 0.43 + attack_spacing_gap * 0.15)
            .clamp(0.0, 1.0);
    let defense_pressure = (defense_contract_gap * 0.50
        + defense_spacing_gap * 0.18
        + defense_ball_gap * 0.16
        + press_gap * 0.16)
        .clamp(0.0, 1.0);
    let shape_pressure = attack_pressure.max(defense_pressure);

    if attack_pressure > 1e-12 {
        let mut wide_flank = base.clone();
        wide_flank.attack_width_delta_weight += attack_width_gap * 0.24 + attack_pressure * 0.10;
        wide_flank.attack_width_score_weight += attack_width_gap * 0.07;
        wide_flank.attack_flank_lane_weight += attack_flank_gap * 0.30 + attack_pressure * 0.12;
        wide_flank.attack_spacing_delta_weight += attack_spacing_gap * 0.08;
        visit(wide_flank);

        let mut flank_overload = base.clone();
        flank_overload.attack_width_delta_weight +=
            attack_width_gap * 0.30 + attack_flank_gap * 0.10;
        flank_overload.attack_width_score_weight += attack_width_gap * 0.09;
        flank_overload.attack_flank_lane_weight += attack_flank_gap * 0.38 + attack_pressure * 0.18;
        flank_overload.attack_spacing_delta_weight +=
            attack_spacing_gap * 0.12 + attack_pressure * 0.04;
        flank_overload.attack_spacing_score_weight += attack_spacing_gap * 0.04;
        visit(flank_overload);

        let mut spread_support = base.clone();
        spread_support.attack_spacing_delta_weight +=
            attack_spacing_gap * 0.20 + attack_pressure * 0.06;
        spread_support.attack_spacing_score_weight += attack_spacing_gap * 0.10;
        spread_support.attack_width_delta_weight += attack_width_gap * 0.16;
        spread_support.attack_flank_lane_weight += attack_flank_gap * 0.20;
        visit(spread_support);
    }

    if defense_pressure > 1e-12 {
        let mut compact_defense = base.clone();
        compact_defense.defense_contract_delta_weight +=
            defense_contract_gap * 0.30 + defense_pressure * 0.12;
        compact_defense.defense_compactness_score_weight += defense_contract_gap * 0.18;
        compact_defense.defense_spacing_delta_weight += defense_spacing_gap * 0.08;
        compact_defense.defense_ball_depth_score_weight += defense_ball_gap * 0.06;
        compact_defense.defender_midfielder_press_weight += press_gap * 0.04;
        compact_defense.midfielder_press_weight += press_gap * 0.035;
        visit(compact_defense);

        let mut deep_compact_block = base.clone();
        deep_compact_block.defense_contract_delta_weight +=
            defense_contract_gap * 0.34 + defense_pressure * 0.14;
        deep_compact_block.defense_compactness_score_weight += defense_contract_gap * 0.22;
        deep_compact_block.defense_spacing_delta_weight += defense_spacing_gap * 0.07;
        deep_compact_block.defense_spacing_score_weight += defense_spacing_gap * 0.04;
        deep_compact_block.defense_ball_depth_score_weight +=
            defense_ball_gap * 0.12 + defense_pressure * 0.04;
        visit(deep_compact_block);

        let mut press_contract = base.clone();
        press_contract.defense_contract_delta_weight +=
            defense_contract_gap * 0.24 + press_gap * 0.08;
        press_contract.defense_compactness_score_weight += defense_contract_gap * 0.14;
        press_contract.defender_midfielder_press_weight +=
            press_gap * 0.08 + defense_pressure * 0.025;
        press_contract.midfielder_press_weight += press_gap * 0.07 + defense_pressure * 0.020;
        press_contract.defense_ball_depth_score_weight += defense_ball_gap * 0.05;
        visit(press_contract);
    }

    if shape_pressure > 1e-12 {
        let mut balanced_shape = base.clone();
        balanced_shape.attack_width_delta_weight +=
            attack_width_gap * 0.22 + attack_pressure * 0.08;
        balanced_shape.attack_width_score_weight += attack_width_gap * 0.05;
        balanced_shape.attack_flank_lane_weight += attack_flank_gap * 0.27 + attack_pressure * 0.10;
        balanced_shape.attack_spacing_delta_weight += attack_spacing_gap * 0.07 * shape_pressure;
        balanced_shape.defense_contract_delta_weight +=
            defense_contract_gap * 0.27 + defense_pressure * 0.10;
        balanced_shape.defense_compactness_score_weight += defense_contract_gap * 0.16;
        balanced_shape.defense_spacing_delta_weight += defense_spacing_gap * 0.06 * shape_pressure;
        balanced_shape.defense_ball_depth_score_weight += defense_ball_gap * 0.05 * shape_pressure;
        visit(balanced_shape);
    }
}

pub fn soccer_tactical_search_pressure(summary: &SoccerTacticalLearningSummary) -> f64 {
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    (attack_width_gap * 0.22
        + attack_flank_gap * 0.28
        + attack_spacing_gap * 0.08
        + defense_contract_gap * 0.26
        + defense_spacing_gap * 0.06
        + defense_ball_gap * 0.05
        + press_gap * 0.05)
        .clamp(0.0, 1.0)
}

pub fn adapt_soccer_evolution_options_for_tactical_search(
    mut options: SoccerEvolutionOptions,
    summary: &SoccerTacticalLearningSummary,
) -> SoccerEvolutionOptions {
    let pressure = soccer_tactical_search_pressure(summary);
    if pressure <= 1e-12 {
        return options;
    }
    let search_enabled = options.population_size > 1
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0
        || options.crossover_rate > 0.0
        || options.exploration_rate > 0.0
        || options.exploration_scale > 0.0;
    if !search_enabled {
        return options;
    }

    options.mutation_rate = (options.mutation_rate + pressure * 0.015).clamp(0.0, 0.35);
    options.mutation_scale = (options.mutation_scale * (1.0 + pressure * 0.45))
        .max(options.mutation_scale)
        .min(0.85);
    if options.exploration_rate > 0.0 || options.exploration_scale > 0.0 {
        options.exploration_rate = (options.exploration_rate + pressure * 0.08).clamp(0.0, 0.4);
        options.exploration_scale = (options.exploration_scale * (1.0 + pressure * 0.55))
            .max(options.exploration_scale)
            .min(1.2);
    }
    if options.population_size > 1 {
        let extra_candidates = ((options.population_size as f64) * pressure).ceil() as usize;
        let cap = options.population_size.saturating_mul(2).min(64);
        options.population_size = options
            .population_size
            .saturating_add(extra_candidates)
            .min(cap)
            .max(2);
    }
    options
}

fn weighted_tactical_evolution_summary(
    parents: &[(&SoccerTacticalLearningSummary, f64)],
    options: SoccerEvolutionOptions,
) -> Option<SoccerTacticalLearningSummary> {
    if parents.is_empty() {
        return None;
    }
    let mut weighted_summary = SoccerTacticalLearningSummary::default();
    let mut total_weight = 0.0;
    for (summary, fitness) in parents {
        let weight = fitness.max(options.elite_weight_floor).max(0.0);
        if weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        weighted_summary.mean_attack_width_score += summary.mean_attack_width_score * weight;
        weighted_summary.mean_attack_flank_lane_score +=
            summary.mean_attack_flank_lane_score * weight;
        weighted_summary.mean_attack_spacing_score += summary.mean_attack_spacing_score * weight;
        weighted_summary.mean_defense_contract_score +=
            summary.mean_defense_contract_score * weight;
        weighted_summary.mean_defense_spacing_score += summary.mean_defense_spacing_score * weight;
        weighted_summary.mean_defense_ball_gap_score +=
            summary.mean_defense_ball_gap_score * weight;
        weighted_summary.mean_defense_role_press_score +=
            summary.mean_defense_role_press_score * weight;
    }
    if total_weight <= 0.0 {
        return None;
    }
    weighted_summary.mean_attack_width_score /= total_weight;
    weighted_summary.mean_attack_flank_lane_score /= total_weight;
    weighted_summary.mean_attack_spacing_score /= total_weight;
    weighted_summary.mean_defense_contract_score /= total_weight;
    weighted_summary.mean_defense_spacing_score /= total_weight;
    weighted_summary.mean_defense_ball_gap_score /= total_weight;
    weighted_summary.mean_defense_role_press_score /= total_weight;
    Some(weighted_summary)
}

fn weighted_soccer_tactical_genome_centroid(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
) -> Option<SoccerTacticalLearningWeights> {
    let mut centroid = SoccerTacticalLearningWeights::default();
    centroid.attack_spacing_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_spacing_delta_weight
        })?;
    centroid.attack_spacing_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_spacing_score_weight
        })?;
    centroid.attack_width_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_width_delta_weight
        })?;
    centroid.attack_width_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_width_score_weight
        })?;
    centroid.attack_flank_lane_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_flank_lane_weight
        })?;
    centroid.defense_spacing_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_spacing_delta_weight
        })?;
    centroid.defense_spacing_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_spacing_score_weight
        })?;
    centroid.defense_contract_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_contract_delta_weight
        })?;
    centroid.defense_compactness_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_compactness_score_weight
        })?;
    centroid.defense_ball_depth_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_ball_depth_score_weight
        })?;
    centroid.defense_endline_soft_penalty_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_endline_soft_penalty_weight
        })?;
    centroid.defense_endline_hard_penalty_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_endline_hard_penalty_weight
        })?;
    centroid.defender_midfielder_press_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defender_midfielder_press_weight
        })?;
    centroid.midfielder_press_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.midfielder_press_weight
        })?;
    centroid.formation_lp_alignment_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.formation_lp_alignment_weight
        })?;

    Some(clamp_soccer_tactical_learning_weights(&centroid))
}

fn weighted_soccer_tactical_genome_gene<F>(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    getter: F,
) -> Option<f64>
where
    F: Fn(&SoccerTacticalLearningWeights) -> f64,
{
    let mut weighted_gene = 0.0;
    let mut total_weight = 0.0;
    for parent in parents {
        let value = getter(parent.weights);
        if !value.is_finite() {
            continue;
        }
        let weight = tactical_genome_parent_weight(parent.fitness, options);
        if weight <= 0.0 {
            continue;
        }
        weighted_gene += value * weight;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        Some(weighted_gene / total_weight)
    } else {
        None
    }
}

fn extrapolate_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    target: &SoccerTacticalLearningWeights,
    scale: f64,
) -> SoccerTacticalLearningWeights {
    let scale = scale.max(0.0);
    let mut extrapolated = base.clone();
    extrapolated.attack_spacing_delta_weight = extrapolate_soccer_tactical_weight(
        base.attack_spacing_delta_weight,
        target.attack_spacing_delta_weight,
        scale,
    );
    extrapolated.attack_spacing_score_weight = extrapolate_soccer_tactical_weight(
        base.attack_spacing_score_weight,
        target.attack_spacing_score_weight,
        scale,
    );
    extrapolated.attack_width_delta_weight = extrapolate_soccer_tactical_weight(
        base.attack_width_delta_weight,
        target.attack_width_delta_weight,
        scale,
    );
    extrapolated.attack_width_score_weight = extrapolate_soccer_tactical_weight(
        base.attack_width_score_weight,
        target.attack_width_score_weight,
        scale,
    );
    extrapolated.attack_flank_lane_weight = extrapolate_soccer_tactical_weight(
        base.attack_flank_lane_weight,
        target.attack_flank_lane_weight,
        scale,
    );
    extrapolated.defense_spacing_delta_weight = extrapolate_soccer_tactical_weight(
        base.defense_spacing_delta_weight,
        target.defense_spacing_delta_weight,
        scale,
    );
    extrapolated.defense_spacing_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_spacing_score_weight,
        target.defense_spacing_score_weight,
        scale,
    );
    extrapolated.defense_contract_delta_weight = extrapolate_soccer_tactical_weight(
        base.defense_contract_delta_weight,
        target.defense_contract_delta_weight,
        scale,
    );
    extrapolated.defense_compactness_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_compactness_score_weight,
        target.defense_compactness_score_weight,
        scale,
    );
    extrapolated.defense_ball_depth_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_ball_depth_score_weight,
        target.defense_ball_depth_score_weight,
        scale,
    );
    extrapolated.defense_endline_soft_penalty_weight = extrapolate_soccer_tactical_weight(
        base.defense_endline_soft_penalty_weight,
        target.defense_endline_soft_penalty_weight,
        scale,
    );
    extrapolated.defense_endline_hard_penalty_weight = extrapolate_soccer_tactical_weight(
        base.defense_endline_hard_penalty_weight,
        target.defense_endline_hard_penalty_weight,
        scale,
    );
    extrapolated.defender_midfielder_press_weight = extrapolate_soccer_tactical_weight(
        base.defender_midfielder_press_weight,
        target.defender_midfielder_press_weight,
        scale,
    );
    extrapolated.midfielder_press_weight = extrapolate_soccer_tactical_weight(
        base.midfielder_press_weight,
        target.midfielder_press_weight,
        scale,
    );
    extrapolated.formation_lp_alignment_weight = extrapolate_soccer_tactical_weight(
        base.formation_lp_alignment_weight,
        target.formation_lp_alignment_weight,
        scale,
    );
    clamp_soccer_tactical_learning_weights(&extrapolated)
}

fn extrapolate_soccer_tactical_weight(base: f64, target: f64, scale: f64) -> f64 {
    if base.is_finite() && target.is_finite() {
        base + (target - base) * scale
    } else {
        base
    }
}

fn evolve_soccer_tactical_learning_candidate(
    base: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let mut rng = DeterministicRng::new(options.seed ^ 0x5eed_f00d_cafe_babe);
    let mut evolved = base.clone();
    let mutation_scale = options.mutation_scale.max(0.0) * 0.22;
    let exploration_rate = options.exploration_rate.clamp(0.0, 1.0);
    let exploration_scale = options.exploration_scale.max(0.0) * 0.18;

    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    evolve_weight(
        &mut evolved.attack_spacing_delta_weight,
        attack_spacing_gap * 0.050,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.8,
    );
    evolve_weight(
        &mut evolved.attack_width_delta_weight,
        attack_width_gap * 0.070,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.2,
    );
    evolve_weight(
        &mut evolved.attack_flank_lane_weight,
        attack_flank_gap * 0.085,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.2,
    );
    evolve_weight(
        &mut evolved.defense_contract_delta_weight,
        defense_contract_gap * 0.080,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.4,
    );
    evolve_weight(
        &mut evolved.defense_compactness_score_weight,
        defense_contract_gap * 0.052,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.defense_spacing_delta_weight,
        defense_spacing_gap * 0.040,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.8,
    );
    evolve_weight(
        &mut evolved.defense_ball_depth_score_weight,
        defense_ball_gap * 0.050,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.defender_midfielder_press_weight,
        press_gap * 0.034,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.6,
    );
    evolve_weight(
        &mut evolved.midfielder_press_weight,
        press_gap * 0.030,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.6,
    );

    evolved
}

fn crossover_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
) -> SoccerTacticalLearningWeights {
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    if parents.len() < 2 || crossover_rate <= 0.0 {
        return base.clone();
    }
    let mut child = base.clone();
    crossover_tactical_weight_gene(
        &mut child.attack_spacing_delta_weight,
        parents,
        options,
        rng,
        1.8,
        |weights| weights.attack_spacing_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_spacing_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.attack_spacing_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_width_delta_weight,
        parents,
        options,
        rng,
        2.2,
        |weights| weights.attack_width_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_width_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.attack_width_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_flank_lane_weight,
        parents,
        options,
        rng,
        2.2,
        |weights| weights.attack_flank_lane_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_spacing_delta_weight,
        parents,
        options,
        rng,
        1.8,
        |weights| weights.defense_spacing_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_spacing_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.defense_spacing_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_contract_delta_weight,
        parents,
        options,
        rng,
        2.4,
        |weights| weights.defense_contract_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_compactness_score_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.defense_compactness_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_ball_depth_score_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.defense_ball_depth_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_endline_soft_penalty_weight,
        parents,
        options,
        rng,
        5.0,
        |weights| weights.defense_endline_soft_penalty_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_endline_hard_penalty_weight,
        parents,
        options,
        rng,
        5.0,
        |weights| weights.defense_endline_hard_penalty_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defender_midfielder_press_weight,
        parents,
        options,
        rng,
        1.6,
        |weights| weights.defender_midfielder_press_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.midfielder_press_weight,
        parents,
        options,
        rng,
        1.6,
        |weights| weights.midfielder_press_weight,
    );
    child
}

fn crossover_tactical_weight_gene<F>(
    value: &mut f64,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
    max_value: f64,
    getter: F,
) where
    F: Fn(&SoccerTacticalLearningWeights) -> f64,
{
    if rng.next_f64() > options.crossover_rate.clamp(0.0, 1.0) {
        return;
    }
    let Some(parent_index) = select_tactical_genome_parent(parents, options, rng) else {
        return;
    };
    let parent_value = getter(parents[parent_index].weights);
    if parent_value.is_finite() {
        *value = parent_value.max(0.0).min(max_value.max(0.0));
    }
}

fn select_tactical_genome_parent(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
) -> Option<usize> {
    let total_weight = parents
        .iter()
        .map(|parent| tactical_genome_parent_weight(parent.fitness, options))
        .sum::<f64>();
    if total_weight <= 0.0 {
        return None;
    }
    let mut cursor = rng.next_f64() * total_weight;
    for (index, parent) in parents.iter().enumerate() {
        let weight = tactical_genome_parent_weight(parent.fitness, options);
        if weight <= 0.0 {
            continue;
        }
        if cursor <= weight {
            return Some(index);
        }
        cursor -= weight;
    }
    parents
        .iter()
        .enumerate()
        .rfind(|(_, parent)| tactical_genome_parent_weight(parent.fitness, options) > 0.0)
        .map(|(index, _)| index)
}

fn tactical_genome_parent_weight(fitness: f64, options: SoccerEvolutionOptions) -> f64 {
    if fitness.is_finite() {
        fitness.max(options.elite_weight_floor).max(0.0)
    } else {
        0.0
    }
}

fn clamp_soccer_tactical_learning_weights(
    weights: &SoccerTacticalLearningWeights,
) -> SoccerTacticalLearningWeights {
    let mut clamped = weights.clone();
    clamped.attack_spacing_delta_weight = clamped.attack_spacing_delta_weight.max(0.0).min(1.8);
    clamped.attack_spacing_score_weight = clamped.attack_spacing_score_weight.max(0.0).min(1.2);
    clamped.attack_width_delta_weight = clamped.attack_width_delta_weight.max(0.0).min(2.2);
    clamped.attack_width_score_weight = clamped.attack_width_score_weight.max(0.0).min(1.2);
    clamped.attack_flank_lane_weight = clamped.attack_flank_lane_weight.max(0.0).min(2.2);
    clamped.defense_spacing_delta_weight = clamped.defense_spacing_delta_weight.max(0.0).min(1.8);
    clamped.defense_spacing_score_weight = clamped.defense_spacing_score_weight.max(0.0).min(1.2);
    clamped.defense_contract_delta_weight = clamped.defense_contract_delta_weight.max(0.0).min(2.4);
    clamped.defense_compactness_score_weight =
        clamped.defense_compactness_score_weight.max(0.0).min(2.0);
    clamped.defense_ball_depth_score_weight =
        clamped.defense_ball_depth_score_weight.max(0.0).min(2.0);
    clamped.defense_endline_soft_penalty_weight = clamped
        .defense_endline_soft_penalty_weight
        .max(0.0)
        .min(5.0);
    clamped.defense_endline_hard_penalty_weight = clamped
        .defense_endline_hard_penalty_weight
        .max(0.0)
        .min(5.0);
    clamped.defender_midfielder_press_weight =
        clamped.defender_midfielder_press_weight.max(0.0).min(1.6);
    clamped.midfielder_press_weight = clamped.midfielder_press_weight.max(0.0).min(1.6);
    clamped.formation_lp_alignment_weight =
        clamped.formation_lp_alignment_weight.max(-5.0).min(5.0);
    clamped
}

fn soccer_tactical_weight_search_score(
    weights: &SoccerTacticalLearningWeights,
    summary: &SoccerTacticalLearningSummary,
) -> f64 {
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    weights.attack_width_delta_weight * attack_width_gap * 1.15
        + weights.attack_width_score_weight * attack_width_gap * 0.45
        + weights.attack_flank_lane_weight * attack_flank_gap * 1.35
        + weights.attack_spacing_delta_weight * attack_spacing_gap * 0.85
        + weights.attack_spacing_score_weight * attack_spacing_gap * 0.35
        + weights.defense_contract_delta_weight * defense_contract_gap * 1.35
        + weights.defense_compactness_score_weight * defense_contract_gap * 1.05
        + weights.defense_spacing_delta_weight * defense_spacing_gap * 0.80
        + weights.defense_spacing_score_weight * defense_spacing_gap * 0.35
        + weights.defense_ball_depth_score_weight * defense_ball_gap * 0.70
        + weights.defender_midfielder_press_weight * press_gap * 0.40
        + weights.midfielder_press_weight * press_gap * 0.35
}

pub fn run_soccer_learning_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: SoccerTeamQPolicies,
    neural_drain_timeout: Duration,
) -> Result<SoccerLearningCompletedGame, String> {
    run_soccer_learning_game_from_snapshot(
        episode,
        config,
        Arc::new(starting_policies),
        None,
        neural_drain_timeout,
    )
}

fn run_soccer_learning_game_from_snapshot(
    episode: usize,
    mut config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    neural_drain_timeout: Duration,
) -> Result<SoccerLearningCompletedGame, String> {
    let started = Instant::now();
    config.seed = config.seed.wrapping_add(episode as u32);
    let seed = config.seed as u64;
    let starting_tactical_learning = config.tactical_learning.clone();
    let total_ticks = config.total_ticks();
    let mut sim =
        SoccerMatch::default_11v11(config).with_team_policies((*starting_policies).clone());
    if let Some(snapshot) = initial_neural_network.as_ref() {
        sim.set_neural_network_snapshot((**snapshot).clone())?;
    }

    for _ in 0..total_ticks {
        sim.run_time_step();
    }

    sim.drain_neural_learning(neural_drain_timeout);
    let policies = sim
        .team_policies()
        .cloned()
        .ok_or_else(|| "soccer learning produced no team policies".to_string())?;
    let mut artifact = sim.team_policy_artifact();
    let neural_network = artifact.learning.neural_network.take();
    let summary = artifact.summary.clone();
    let score = soccer_learning_run_score(&summary);
    let delta = soccer_policy_delta_entries(starting_policies.as_ref(), &policies, &score);
    let episode_summary = SoccerSelfPlayEpisodeSummary {
        episode,
        seed,
        summary,
        transitions: artifact.learning.total_transitions,
        home_policy_entries: artifact.home_entries.len(),
        home_policy_target_entries: artifact.home_target_entries.len(),
        away_policy_entries: artifact.away_entries.len(),
        away_policy_target_entries: artifact.away_target_entries.len(),
    };

    Ok(SoccerLearningCompletedGame {
        episode,
        seed,
        summary: episode_summary.summary.clone(),
        episode_summary,
        tactical_summary: artifact.tactical_summary,
        starting_tactical_learning,
        policies,
        score,
        delta,
        neural_network,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn run_soccer_learning_queue(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
) -> Result<SoccerLearningQueueReport, String> {
    run_soccer_learning_queue_with_observer(config, initial_policies, |_, _| Ok(()))
}

struct SoccerLearningQueueTask {
    episode: usize,
    match_config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    neural_drain_timeout: Duration,
}

pub fn run_soccer_learning_queue_with_observer<F>(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
    mut on_completed_game: F,
) -> Result<SoccerLearningQueueReport, String>
where
    F: FnMut(&SoccerLearningCompletedGame, &SoccerTeamQPolicies) -> Result<(), String>,
{
    run_soccer_learning_queue_with_events(config, initial_policies, |event| {
        if let SoccerLearningQueueEvent::CompletedGame {
            game,
            merged_policies,
        } = event
        {
            on_completed_game(game, merged_policies)?;
        }
        Ok(())
    })
}

pub fn run_soccer_learning_queue_with_events<F>(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
    mut on_event: F,
) -> Result<SoccerLearningQueueReport, String>
where
    F: for<'a> FnMut(SoccerLearningQueueEvent<'a>) -> Result<(), String>,
{
    let started = Instant::now();
    let mut config = config;
    let parallel_games = config.parallel_games.clamp(1, 100);
    let (task_tx, task_rx) = mpsc::sync_channel::<SoccerLearningQueueTask>(parallel_games);
    let task_rx = Arc::new(Mutex::new(task_rx));
    let (result_tx, result_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(parallel_games);
    for _ in 0..parallel_games {
        let task_rx = Arc::clone(&task_rx);
        let result_tx = result_tx.clone();
        workers.push(thread::spawn(move || loop {
            let task = {
                let receiver = task_rx
                    .lock()
                    .expect("soccer learning queue task receiver poisoned");
                receiver.recv()
            };
            let Ok(task) = task else {
                break;
            };
            let result = run_soccer_learning_game_from_snapshot(
                task.episode,
                task.match_config,
                task.starting_policies,
                task.initial_neural_network,
                task.neural_drain_timeout,
            );
            let _ = result_tx.send((task.episode, result));
        }));
    }
    drop(result_tx);

    let mut active = 0usize;
    let mut next_episode = 0usize;
    let mut completed_games = 0usize;
    let mut failed_games = 0usize;
    let mut policies = initial_policies;
    let mut episode_summaries = Vec::with_capacity(config.games);
    let mut tactical_summary = SoccerTacticalLearningSummary::default();
    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut latest_neural_network = config.initial_neural_network.clone();
    let mut first_error = None::<String>;

    while completed_games + failed_games < config.games && first_error.is_none() {
        if active < parallel_games && next_episode < config.games {
            while active < parallel_games && next_episode < config.games {
                if let Err(err) = on_event(SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    match_config: &mut config.match_config,
                    policies: &mut policies,
                    neural_network: &mut latest_neural_network,
                }) {
                    first_error = Some(err);
                    break;
                }
                let starting_policies = Arc::new(policies.clone());
                let starting_neural_network = latest_neural_network.clone().map(Arc::new);
                let episode = next_episode;
                let mut match_config = config.match_config.clone();
                match_config.seed = config.base_seed;
                let neural_drain_timeout = config.neural_drain_timeout;
                if let Err(err) = task_tx.send(SoccerLearningQueueTask {
                    episode,
                    match_config,
                    starting_policies: Arc::clone(&starting_policies),
                    initial_neural_network: starting_neural_network.as_ref().map(Arc::clone),
                    neural_drain_timeout,
                }) {
                    first_error = Some(format!("soccer learning queue task send failed: {err}"));
                    break;
                }
                active += 1;
                next_episode += 1;
            }
        }

        if first_error.is_some() {
            break;
        }

        let game_result = match result_rx.recv() {
            Ok((_, game_result)) => game_result,
            Err(err) => {
                first_error = Some(format!(
                    "soccer learning queue worker channel closed: {err}"
                ));
                break;
            }
        };
        active = active.saturating_sub(1);

        match game_result {
            Ok(mut game) => {
                let merged = match merge_soccer_policy_deltas(
                    &policies,
                    std::slice::from_ref(&game.delta),
                    1.0,
                ) {
                    Ok(merged) => merged,
                    Err(err) => {
                        first_error = Some(err);
                        break;
                    }
                };
                policies = merged;
                policies.prune(
                    config.prune_action_entries_per_team,
                    config.prune_target_entries_per_team,
                    config.min_policy_visits,
                );
                if let Err(err) = on_event(SoccerLearningQueueEvent::CompletedGame {
                    game: &game,
                    merged_policies: &mut policies,
                }) {
                    first_error = Some(err);
                    break;
                }
                if let Some(snapshot) = game.neural_network.take() {
                    latest_neural_network = Some(snapshot);
                }
                total_home_goals = total_home_goals.saturating_add(game.summary.score_home);
                total_away_goals = total_away_goals.saturating_add(game.summary.score_away);
                tactical_summary.merge(&game.tactical_summary);
                episode_summaries.push(game.episode_summary);
                completed_games += 1;
            }
            Err(_) => {
                failed_games += 1;
            }
        }
    }

    drop(task_tx);
    for worker in workers {
        if worker.join().is_err() && first_error.is_none() {
            first_error = Some("soccer learning queue worker thread panicked".to_string());
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }

    episode_summaries.sort_by_key(|summary| summary.episode);
    let final_policy_entries = policies.home.entries().len() + policies.away.entries().len();
    let final_target_entries =
        policies.home.target_entries().len() + policies.away.target_entries().len();

    Ok(SoccerLearningQueueReport {
        completed_games,
        failed_games,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        total_home_goals,
        total_away_goals,
        tactical_summary,
        final_policy_entries,
        final_target_entries,
        episode_summaries,
        final_policies: policies,
        latest_neural_network,
    })
}

pub fn soccer_self_play_artifact_from_queue_report(
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    report: &SoccerLearningQueueReport,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
        tactical_learning: config.tactical_learning.clone(),
        tactical_summary: report.tactical_summary.clone(),
        config,
        options,
        episodes: report.episode_summaries.clone(),
        home_entries: report.final_policies.home.entries(),
        home_target_entries: report.final_policies.home.target_entries(),
        away_entries: report.final_policies.away.entries(),
        away_target_entries: report.final_policies.away.target_entries(),
    }
}

fn collect_team_policy_delta(
    team: Team,
    before: &SoccerQPolicy,
    after: &SoccerQPolicy,
    merge_weight: f64,
    entries: &mut Vec<SoccerLearningPolicyDeltaEntry>,
) {
    let before_actions = entry_map(
        team,
        SoccerLearningPolicyEntryKind::Action,
        before.entries().into_iter().map(action_entry_value),
    );
    for entry in after.entries() {
        let value = action_entry_value(entry);
        let key = policy_entry_key(team, SoccerLearningPolicyEntryKind::Action, &value);
        let before_value = before_actions.get(&key);
        push_delta_entry(
            team,
            SoccerLearningPolicyEntryKind::Action,
            value,
            before_value,
            merge_weight,
            entries,
        );
    }

    let before_targets = entry_map(
        team,
        SoccerLearningPolicyEntryKind::Target,
        before.target_entries().into_iter().map(target_entry_value),
    );
    for entry in after.target_entries() {
        let value = target_entry_value(entry);
        let key = policy_entry_key(team, SoccerLearningPolicyEntryKind::Target, &value);
        let before_value = before_targets.get(&key);
        push_delta_entry(
            team,
            SoccerLearningPolicyEntryKind::Target,
            value,
            before_value,
            merge_weight,
            entries,
        );
    }
}

fn push_delta_entry(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    value: EntryValue,
    before_value: Option<&EntryValue>,
    merge_weight: f64,
    entries: &mut Vec<SoccerLearningPolicyDeltaEntry>,
) {
    let before_visits = before_value.map(|entry| entry.visits).unwrap_or(0);
    if value.visits <= before_visits {
        return;
    }
    let visit_delta = value.visits - before_visits;
    let before_q = before_value.map(|entry| entry.value).unwrap_or(0.0);
    let state_json = serde_json::to_value(&value.state_key)
        .unwrap_or_else(|_| Value::Object(Default::default()));
    let state_hash = state_hash(&state_json);
    let effective_visit_weight = f64::from(visit_delta) * merge_weight.max(0.0);
    entries.push(SoccerLearningPolicyDeltaEntry {
        team,
        entry_kind,
        state_hash,
        state_key: value.state_key,
        state_json,
        action: value.action,
        target_fine_cell_id: value.target_fine_cell_id,
        target_tactical_cell_id: value.target_tactical_cell_id,
        target_macro_cell_id: value.target_macro_cell_id,
        target_root_cell_id: value.target_root_cell_id,
        before_value: before_q,
        after_value: value.value,
        value_delta: value.value - before_q,
        before_value_micros: soccer_learning_to_micros(before_q),
        after_value_micros: soccer_learning_to_micros(value.value),
        value_delta_micros: soccer_learning_to_micros(value.value - before_q),
        visit_delta,
        merge_weight,
        merge_weight_micros: soccer_learning_to_micros(merge_weight),
        effective_visit_weight,
        effective_visit_micros: soccer_learning_to_micros(effective_visit_weight),
    });
}

fn entry_map(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    mut entries: impl Iterator<Item = EntryValue>,
) -> HashMap<PolicyEntryKey, EntryValue> {
    let (lower, upper) = entries.size_hint();
    let mut map = HashMap::with_capacity(upper.unwrap_or(lower));
    for entry in entries.by_ref() {
        map.insert(policy_entry_key(team, entry_kind, &entry), entry);
    }
    map
}

fn action_entry_value(entry: SoccerQEntry) -> EntryValue {
    EntryValue {
        state_key: entry.state,
        action: entry.action,
        value: entry.value,
        visits: entry.visits,
        target_fine_cell_id: -1,
        target_tactical_cell_id: -1,
        target_macro_cell_id: -1,
        target_root_cell_id: -1,
    }
}

fn target_entry_value(entry: SoccerQTargetEntry) -> EntryValue {
    EntryValue {
        state_key: entry.state,
        action: entry.action,
        value: entry.value,
        visits: entry.visits,
        target_fine_cell_id: entry.target_fine_cell_id as i32,
        target_tactical_cell_id: entry.target_tactical_cell_id as i32,
        target_macro_cell_id: entry.target_macro_cell_id as i32,
        target_root_cell_id: entry.target_root_cell_id as i32,
    }
}

fn policy_entry_key(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    value: &EntryValue,
) -> PolicyEntryKey {
    let state_json = serde_json::to_string(&value.state_key).unwrap_or_default();
    let state_hash = fnv1a_hex(state_json.as_bytes());
    PolicyEntryKey {
        team: soccer_team_label(team),
        entry_kind,
        state_hash,
        state_json,
        action: value.action.clone(),
        target_fine_cell_id: value.target_fine_cell_id,
        target_tactical_cell_id: value.target_tactical_cell_id,
        target_macro_cell_id: value.target_macro_cell_id,
        target_root_cell_id: value.target_root_cell_id,
    }
}

fn policy_delta_key(entry: &SoccerLearningPolicyDeltaEntry) -> PolicyEntryKey {
    let state_json = serde_json::to_string(&entry.state_key).unwrap_or_default();
    PolicyEntryKey {
        team: soccer_team_label(entry.team),
        entry_kind: entry.entry_kind,
        state_hash: entry.state_hash.clone(),
        state_json,
        action: entry.action.clone(),
        target_fine_cell_id: entry.target_fine_cell_id,
        target_tactical_cell_id: entry.target_tactical_cell_id,
        target_macro_cell_id: entry.target_macro_cell_id,
        target_root_cell_id: entry.target_root_cell_id,
    }
}

fn seed_policy_accumulators(
    team: Team,
    policy: &SoccerQPolicy,
    prior_weight: f64,
    action_accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    target_accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
) {
    for entry in policy.entries().into_iter().map(action_entry_value) {
        seed_entry_accumulator(
            team,
            SoccerLearningPolicyEntryKind::Action,
            entry,
            prior_weight,
            action_accumulators,
        );
    }
    for entry in policy.target_entries().into_iter().map(target_entry_value) {
        seed_entry_accumulator(
            team,
            SoccerLearningPolicyEntryKind::Target,
            entry,
            prior_weight,
            target_accumulators,
        );
    }
}

fn seed_entry_accumulator(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    entry: EntryValue,
    prior_weight: f64,
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
) {
    let effective_visits = f64::from(entry.visits.max(1)) * prior_weight.max(0.0);
    if effective_visits <= 0.0 {
        return;
    }
    let key = policy_entry_key(team, entry_kind, &entry);
    let item = accumulators.entry(key).or_insert_with(|| MergeAccumulator {
        state_key: entry.state_key.clone(),
        action: entry.action.clone(),
        weighted_value_sum: 0.0,
        effective_visits: 0.0,
        display_visits: 0,
        target_fine_cell_id: entry.target_fine_cell_id,
        target_tactical_cell_id: entry.target_tactical_cell_id,
        target_macro_cell_id: entry.target_macro_cell_id,
        target_root_cell_id: entry.target_root_cell_id,
    });
    item.weighted_value_sum += entry.value * effective_visits;
    item.effective_visits += effective_visits;
    item.display_visits = item.display_visits.saturating_add(entry.visits.max(1));
}

fn policy_action_entry_map(policy: &SoccerTeamQPolicies) -> HashMap<PolicyEntryKey, EntryValue> {
    let mut map = entry_map(
        Team::Home,
        SoccerLearningPolicyEntryKind::Action,
        policy.home.entries().into_iter().map(action_entry_value),
    );
    map.extend(entry_map(
        Team::Away,
        SoccerLearningPolicyEntryKind::Action,
        policy.away.entries().into_iter().map(action_entry_value),
    ));
    map
}

fn policy_target_entry_map(policy: &SoccerTeamQPolicies) -> HashMap<PolicyEntryKey, EntryValue> {
    let mut map = entry_map(
        Team::Home,
        SoccerLearningPolicyEntryKind::Target,
        policy
            .home
            .target_entries()
            .into_iter()
            .map(target_entry_value),
    );
    map.extend(entry_map(
        Team::Away,
        SoccerLearningPolicyEntryKind::Target,
        policy
            .away
            .target_entries()
            .into_iter()
            .map(target_entry_value),
    ));
    map
}

fn crossover_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    parent_maps: &[HashMap<PolicyEntryKey, EntryValue>],
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    if parent_maps.len() < 2 || crossover_rate <= 0.0 {
        return;
    }
    for (key, accumulator) in accumulators.iter_mut() {
        if rng.next_f64() > crossover_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let start = rng.next_usize(parent_maps.len());
        for offset in 0..parent_maps.len() {
            let index = (start + offset) % parent_maps.len();
            let Some(parent_value) = parent_maps[index].get(key) else {
                continue;
            };
            accumulator.weighted_value_sum =
                parent_value.value.clamp(-120.0, 120.0) * accumulator.effective_visits;
            accumulator.display_visits = accumulator.display_visits.max(parent_value.visits.max(1));
            break;
        }
    }
}

fn explore_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let exploration_rate = options.exploration_rate.clamp(0.0, 1.0);
    let exploration_scale = options.exploration_scale.max(0.0);
    if exploration_rate <= 0.0 || exploration_scale <= 0.0 {
        return;
    }
    for accumulator in accumulators.values_mut() {
        if rng.next_f64() > exploration_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let current = accumulator.weighted_value_sum / accumulator.effective_visits;
        let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let magnitude = (0.25 + rng.next_f64() * 0.75) * exploration_scale;
        accumulator.weighted_value_sum =
            (current + sign * magnitude).clamp(-120.0, 120.0) * accumulator.effective_visits;
    }
}

fn evolve_weight(
    weight: &mut f64,
    directed_delta: f64,
    mutation_scale: f64,
    exploration_rate: f64,
    exploration_scale: f64,
    rng: &mut DeterministicRng,
    max_value: f64,
) {
    let mutation = if mutation_scale > 0.0 {
        (rng.next_f64() * 2.0 - 1.0) * mutation_scale
    } else {
        0.0
    };
    let exploration = if exploration_scale > 0.0 && rng.next_f64() < exploration_rate {
        (rng.next_f64() * 2.0 - 1.0) * exploration_scale
    } else {
        0.0
    };
    *weight = (*weight + directed_delta + mutation + exploration)
        .max(0.0)
        .min(max_value.max(0.0));
}

fn mutate_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let mutation_rate = options.mutation_rate.clamp(0.0, 1.0);
    let mutation_scale = options.mutation_scale.max(0.0);
    for accumulator in accumulators.values_mut() {
        if rng.next_f64() > mutation_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let current = accumulator.weighted_value_sum / accumulator.effective_visits;
        let perturbation = (rng.next_f64() * 2.0 - 1.0) * mutation_scale;
        accumulator.weighted_value_sum =
            (current + perturbation).clamp(-120.0, 120.0) * accumulator.effective_visits;
    }
}

fn state_hash(state_json: &Value) -> String {
    let raw = serde_json::to_string(state_json).unwrap_or_default();
    fnv1a_hex(raw.as_bytes())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn candidate_seed(base_seed: u64, candidate_index: usize, salt: u64) -> u64 {
    if candidate_index == 0 {
        return base_seed;
    }
    let mut mixed = base_seed ^ salt;
    mixed ^= (candidate_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^ (mixed >> 29)
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.state;
        z ^= z >> 33;
        z = z.wrapping_mul(0xff51afd7ed558ccd);
        z ^= z >> 33;
        z = z.wrapping_mul(0xc4ceb9fe1a85ec53);
        z ^ (z >> 33)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, upper_bound: usize) -> usize {
        if upper_bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper_bound
        }
    }
}

fn build_policies_from_accumulators(
    home_options: SoccerQPolicyOptions,
    away_options: SoccerQPolicyOptions,
    action_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
    target_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
) -> Result<SoccerTeamQPolicies, String> {
    let action_capacity = action_accumulators.len();
    let target_capacity = target_accumulators.len();
    let mut home_entries = Vec::with_capacity(action_capacity.saturating_add(1) / 2);
    let mut away_entries = Vec::with_capacity(action_capacity / 2);
    let mut home_targets = Vec::with_capacity(target_capacity.saturating_add(1) / 2);
    let mut away_targets = Vec::with_capacity(target_capacity / 2);

    for (key, accumulator) in action_accumulators {
        if accumulator.effective_visits <= 0.0 {
            continue;
        }
        let entry = SoccerQEntry {
            state: accumulator.state_key,
            action: accumulator.action,
            value: accumulator.weighted_value_sum / accumulator.effective_visits,
            visits: accumulator.display_visits.max(1),
        };
        match key.team {
            "home" => home_entries.push(entry),
            "away" => away_entries.push(entry),
            _ => {}
        }
    }

    for (key, accumulator) in target_accumulators {
        if accumulator.effective_visits <= 0.0 {
            continue;
        }
        let entry = SoccerQTargetEntry {
            state: accumulator.state_key,
            action: accumulator.action,
            target_fine_cell_id: accumulator.target_fine_cell_id.max(0) as usize,
            target_tactical_cell_id: accumulator.target_tactical_cell_id.max(0) as usize,
            target_macro_cell_id: accumulator.target_macro_cell_id.max(0) as usize,
            target_root_cell_id: accumulator.target_root_cell_id.max(0) as usize,
            value: accumulator.weighted_value_sum / accumulator.effective_visits,
            visits: accumulator.display_visits.max(1),
        };
        match key.team {
            "home" => home_targets.push(entry),
            "away" => away_targets.push(entry),
            _ => {}
        }
    }

    Ok(SoccerTeamQPolicies {
        home: SoccerQPolicy::from_entries_with_targets(home_options, &home_entries, &home_targets)?,
        away: SoccerQPolicy::from_entries_with_targets(away_options, &away_entries, &away_targets)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::soccer::{PlayerRole, TacticalPhase};

    fn test_state() -> SoccerQStateKey {
        serde_json::from_value(serde_json::json!({
            "phase": "Kickoff",
            "role": "Midfielder",
            "possessionRelative": 1,
            "ballZoneX": 2,
            "ballZoneY": 3,
            "scoreDiffBucket": 0,
            "hasBall": true,
            "visibleBall": true,
            "shotLaneOpen": false,
            "visiblePassOptionsBin": 2,
            "ballDistanceBin": 1,
            "yardsToGoalBin": 5,
            "pressureBin": 1,
            "openSpaceBin": 3
        }))
        .expect("test state")
    }

    fn policy_with_action(value: f64, visits: u32) -> SoccerTeamQPolicies {
        let state = test_state();
        let entries = vec![SoccerQEntry {
            state,
            action: "pass".to_string(),
            value,
            visits,
        }];
        SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries(SoccerQPolicyOptions::default(), &entries)
                .expect("home policy"),
            away: SoccerQPolicy::new(SoccerQPolicyOptions::default()),
        }
    }

    #[test]
    fn losing_team_delta_is_weighted_below_winning_team_delta() {
        let score = soccer_learning_run_score(&MatchSummary {
            score_home: 3,
            score_away: 4,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });

        assert_eq!(score.home.outcome, SoccerLearningOutcome::Loss);
        assert_eq!(score.away.outcome, SoccerLearningOutcome::Win);
        assert!(score.home.merge_weight < score.away.merge_weight);
        assert!(score.home.merge_weight < 0.30);
    }

    #[test]
    fn extracts_only_new_visit_deltas() {
        let before = policy_with_action(1.0, 2);
        let after = policy_with_action(3.0, 5);
        let score = soccer_learning_run_score(&MatchSummary {
            score_home: 2,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });
        let delta = soccer_policy_delta_entries(&before, &after, &score);

        assert_eq!(delta.entries.len(), 1);
        assert_eq!(delta.entries[0].visit_delta, 3);
        assert_eq!(delta.entries[0].before_value, 1.0);
        assert_eq!(delta.entries[0].after_value, 3.0);
    }

    #[test]
    fn merge_prefers_outcome_weighted_winner_values() {
        let base = policy_with_action(0.0, 1);
        let mut winning = soccer_policy_delta_entries(
            &base,
            &policy_with_action(4.0, 4),
            &soccer_learning_run_score(&MatchSummary {
                score_home: 2,
                score_away: 0,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            }),
        );
        let mut losing = soccer_policy_delta_entries(
            &base,
            &policy_with_action(-4.0, 4),
            &soccer_learning_run_score(&MatchSummary {
                score_home: 0,
                score_away: 2,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            }),
        );
        winning.entries.append(&mut losing.entries);

        let merged = merge_soccer_policy_deltas(&base, &[winning], 0.0).expect("merged policy");
        let value = merged.home.entries()[0].value;

        assert!(value > 2.0, "winning signal should dominate, got {value}");
    }

    #[test]
    fn evolution_blends_elite_parent_weights() {
        let low = policy_with_action(0.0, 1);
        let high = policy_with_action(10.0, 1);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 7,
        };

        let evolved =
            evolve_soccer_team_policies(&[(&low, 1.0), (&high, 3.0)], options).expect("evolved");
        let value = evolved.home.entries()[0].value;

        assert!((value - 7.5).abs() < 1e-9, "weighted value was {value}");
    }

    #[test]
    fn evolution_crossover_samples_parent_values() {
        let left = policy_with_action(-4.0, 1);
        let right = policy_with_action(4.0, 1);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 11,
        };

        let evolved =
            evolve_soccer_team_policies(&[(&left, 1.0), (&right, 1.0)], options).expect("evolved");
        let value = evolved.home.entries()[0].value;

        assert!(
            (value - -4.0).abs() < 1e-9 || (value - 4.0).abs() < 1e-9,
            "crossover should inherit a parent value, got {value}"
        );
    }

    #[test]
    fn evolution_population_search_keeps_best_policy_candidate() {
        let low = policy_with_action(0.5, 4);
        let high = policy_with_action(3.0, 4);
        let single_options = SoccerEvolutionOptions {
            mutation_rate: 1.0,
            mutation_scale: 0.35,
            crossover_rate: 1.0,
            exploration_rate: 1.0,
            exploration_scale: 0.45,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 41,
        };
        let population_options = SoccerEvolutionOptions {
            population_size: 8,
            ..single_options
        };

        let single = evolve_soccer_team_policies(&[(&low, 1.0), (&high, 2.0)], single_options)
            .expect("single candidate");
        let population =
            evolve_soccer_team_policies(&[(&low, 1.0), (&high, 2.0)], population_options)
                .expect("population candidate");

        assert!(
            soccer_team_policy_search_score(&population)
                >= soccer_team_policy_search_score(&single) - 1e-12
        );
    }

    #[test]
    fn default_evolution_options_keep_genetic_search_broad() {
        let options = SoccerEvolutionOptions::default();

        assert!(options.population_size >= 8);
        assert!(options.mutation_rate >= 0.04);
        assert!(options.mutation_scale >= 0.20);
        assert!(options.crossover_rate >= 0.40);
        assert!(options.exploration_rate >= 0.05);
        assert!(options.exploration_scale >= 0.50);
    }

    #[test]
    fn pending_policy_stays_active_only_when_latest_head_matches_parent() {
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                Some("parent"),
                Some(11),
            ),
            SOCCER_POLICY_STATUS_ACTIVE
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                Some("newer-head"),
                Some(12),
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                Some("parent"),
                Some(12),
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                "archived",
                Some("parent"),
                12,
                Some("newer-head"),
                Some(14),
            ),
            "archived"
        );
    }

    #[test]
    fn postgres_policy_refresh_picks_up_first_or_newer_active_weights() {
        let first = soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
            current_policy_version_id: None,
            current_generation: 0,
            current_updated_at_micros: 0,
            current_neural_network_present: false,
            latest_policy_version_id: "v1",
            latest_generation: 4,
            latest_updated_at_micros: 100,
            latest_neural_network_present: true,
            local_tactical_evolved_since_pg_refresh: false,
            postgres_tactical_learning_authoritative: false,
        });
        assert!(first.refresh_policy);
        assert!(first.apply_tactical_learning);

        let newer_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_neural_network_present: true,
                latest_policy_version_id: "v2",
                latest_generation: 5,
                latest_updated_at_micros: 125,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(newer_head.refresh_policy);
        assert!(newer_head.apply_tactical_learning);
        assert!(!newer_head.same_policy_version);
    }

    #[test]
    fn postgres_policy_refresh_picks_up_same_head_revision_and_neural_snapshot() {
        let revised_same_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_neural_network_present: true,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 125,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(revised_same_head.refresh_policy);
        assert!(revised_same_head.apply_tactical_learning);
        assert!(revised_same_head.same_policy_newer_revision);

        let neural_arrived =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_neural_network_present: false,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(neural_arrived.refresh_policy);
        assert!(!neural_arrived.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_picks_up_durable_pending_evolved_head() {
        let durable_pending_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("local-mutation-v2"),
                current_generation: 5,
                current_updated_at_micros: 0,
                current_neural_network_present: true,
                latest_policy_version_id: "local-mutation-v2",
                latest_generation: 5,
                latest_updated_at_micros: 250,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(durable_pending_head.refresh_policy);
        assert!(durable_pending_head.apply_tactical_learning);
        assert!(durable_pending_head.same_policy_version);
        assert!(durable_pending_head.same_policy_newer_revision);
    }

    #[test]
    fn postgres_policy_refresh_preserves_local_tactical_search_without_new_pg_head() {
        let unchanged = soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
            current_policy_version_id: Some("v1"),
            current_generation: 4,
            current_updated_at_micros: 100,
            current_neural_network_present: true,
            latest_policy_version_id: "v1",
            latest_generation: 4,
            latest_updated_at_micros: 100,
            latest_neural_network_present: true,
            local_tactical_evolved_since_pg_refresh: true,
            postgres_tactical_learning_authoritative: false,
        });
        assert!(!unchanged.refresh_policy);
        assert!(!unchanged.apply_tactical_learning);

        let no_local_search =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                local_tactical_evolved_since_pg_refresh: false,
                ..SoccerPostgresPolicyRefreshCheck {
                    current_policy_version_id: Some("v1"),
                    current_generation: 4,
                    current_updated_at_micros: 100,
                    current_neural_network_present: true,
                    latest_policy_version_id: "v1",
                    latest_generation: 4,
                    latest_updated_at_micros: 100,
                    latest_neural_network_present: true,
                    local_tactical_evolved_since_pg_refresh: true,
                    postgres_tactical_learning_authoritative: false,
                }
            });
        assert!(!no_local_search.refresh_policy);
        assert!(no_local_search.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_can_make_postgres_tactical_weights_authoritative() {
        let unchanged_head_with_local_search =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_neural_network_present: true,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(!unchanged_head_with_local_search.refresh_policy);
        assert!(unchanged_head_with_local_search.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_does_not_reapply_older_active_tactical_weights() {
        let pending_local_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("pending-v2"),
                current_generation: 5,
                current_updated_at_micros: 0,
                current_neural_network_present: true,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 200,
                latest_neural_network_present: true,
                local_tactical_evolved_since_pg_refresh: false,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(!pending_local_head.refresh_policy);
        assert!(!pending_local_head.apply_tactical_learning);
        assert!(!pending_local_head.same_policy_version);
    }

    #[test]
    fn postgres_refresh_for_new_sim_can_include_resume_artifacts() {
        assert!(soccer_should_refresh_postgres_for_new_sim(false, false));
        assert!(soccer_should_refresh_postgres_for_new_sim(false, true));
        assert!(!soccer_should_refresh_postgres_for_new_sim(true, false));
        assert!(soccer_should_refresh_postgres_for_new_sim(true, true));
    }

    #[test]
    fn postgres_policy_version_flush_for_new_sim_requires_pending_refreshable_head() {
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            false, true, 1
        ));
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, false, 1
        ));
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, true, 0
        ));
        assert!(soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, true, 2
        ));
    }

    #[test]
    fn postgres_new_sim_refresh_plan_makes_evolved_heads_durable_before_reads() {
        let plan = soccer_postgres_new_sim_refresh_plan(false, false, true, 2, 1);

        assert!(plan.refresh_from_postgres);
        assert!(plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn postgres_new_sim_refresh_plan_waits_for_async_heads_without_local_buffer() {
        let plan = soccer_postgres_new_sim_refresh_plan(false, false, true, 0, 2);

        assert!(plan.refresh_from_postgres);
        assert!(!plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn postgres_new_sim_refresh_plan_respects_resume_and_flush_opt_outs() {
        let resume_plan = soccer_postgres_new_sim_refresh_plan(true, true, true, 1, 0);
        assert!(resume_plan.refresh_from_postgres);
        assert!(resume_plan.flush_pending_policy_versions);
        assert!(!resume_plan.wait_for_async_policy_versions);

        let stale_resume_plan = soccer_postgres_new_sim_refresh_plan(true, false, true, 1, 1);
        assert!(!stale_resume_plan.refresh_from_postgres);
        assert!(!stale_resume_plan.flush_pending_policy_versions);
        assert!(!stale_resume_plan.wait_for_async_policy_versions);

        let no_flush_plan = soccer_postgres_new_sim_refresh_plan(false, false, false, 1, 1);
        assert!(no_flush_plan.refresh_from_postgres);
        assert!(!no_flush_plan.flush_pending_policy_versions);
        assert!(!no_flush_plan.wait_for_async_policy_versions);
    }

    #[test]
    fn tactical_search_pressure_prioritizes_flanks_and_defensive_contraction() {
        let healthy = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.92,
            mean_attack_flank_lane_score: 0.88,
            mean_attack_spacing_score: 0.86,
            mean_defense_contract_score: 0.90,
            mean_defense_spacing_score: 0.84,
            mean_defense_ball_gap_score: 0.86,
            mean_defense_role_press_score: 0.82,
            ..Default::default()
        };
        let poor_shape = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.24,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.52,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.46,
            mean_defense_ball_gap_score: 0.56,
            mean_defense_role_press_score: 0.58,
            ..Default::default()
        };

        assert!(
            soccer_tactical_search_pressure(&poor_shape)
                > soccer_tactical_search_pressure(&healthy) + 0.45
        );
    }

    #[test]
    fn tactical_search_adapts_evolution_budget_without_enabling_disabled_search() {
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.18,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let base = SoccerEvolutionOptions::default();
        let adapted = adapt_soccer_evolution_options_for_tactical_search(base, &summary);

        assert!(adapted.population_size > base.population_size);
        assert!(adapted.mutation_rate > base.mutation_rate);
        assert!(adapted.mutation_scale > base.mutation_scale);
        assert!(adapted.exploration_rate > base.exploration_rate);
        assert!(adapted.exploration_scale > base.exploration_scale);
        assert!(adapted.population_size <= base.population_size * 2);

        let disabled = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 99,
        };
        assert_eq!(
            adapt_soccer_evolution_options_for_tactical_search(disabled, &summary).population_size,
            disabled.population_size
        );
        assert_eq!(
            adapt_soccer_evolution_options_for_tactical_search(disabled, &summary).mutation_rate,
            disabled.mutation_rate
        );
    }

    #[test]
    fn tactical_weight_evolution_pushes_flank_and_contract_search() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.20,
            mean_attack_flank_lane_score: 0.18,
            mean_attack_spacing_score: 0.35,
            mean_defense_contract_score: 0.22,
            mean_defense_spacing_score: 0.44,
            mean_defense_ball_gap_score: 0.50,
            mean_defense_role_press_score: 0.40,
            ..Default::default()
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 29,
        };

        let evolved = evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], options);

        assert!(evolved.attack_width_delta_weight > base.attack_width_delta_weight);
        assert!(evolved.attack_flank_lane_weight > base.attack_flank_lane_weight);
        assert!(evolved.defense_contract_delta_weight > base.defense_contract_delta_weight);
        assert!(evolved.defense_compactness_score_weight > base.defense_compactness_score_weight);
    }

    #[test]
    fn tactical_strategy_candidates_coordinate_flanks_and_contraction() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.20,
            mean_attack_flank_lane_score: 0.18,
            mean_attack_spacing_score: 0.35,
            mean_defense_contract_score: 0.22,
            mean_defense_spacing_score: 0.44,
            mean_defense_ball_gap_score: 0.50,
            mean_defense_role_press_score: 0.40,
            ..Default::default()
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 30,
        };

        let evolved = evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], options);

        assert!(evolved.attack_width_delta_weight >= base.attack_width_delta_weight + 0.18);
        assert!(evolved.attack_flank_lane_weight >= base.attack_flank_lane_weight + 0.24);
        assert!(evolved.defense_contract_delta_weight >= base.defense_contract_delta_weight + 0.22);
        assert!(
            evolved.defense_compactness_score_weight
                >= base.defense_compactness_score_weight + 0.10
        );
    }

    #[test]
    fn tactical_strategy_candidate_families_search_more_shape_space() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut candidates = Vec::new();
        search_soccer_tactical_strategy_candidates(&base, &summary, |candidate| {
            candidates.push(candidate);
        });

        assert!(candidates.len() >= 7);
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight >= base.attack_flank_lane_weight + 0.40
                && candidate.attack_spacing_delta_weight > base.attack_spacing_delta_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight >= base.defense_contract_delta_weight + 0.35
                && candidate.defense_ball_depth_score_weight
                    >= base.defense_ball_depth_score_weight + 0.09
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defender_midfielder_press_weight
                >= base.defender_midfielder_press_weight + 0.06
                && candidate.midfielder_press_weight >= base.midfielder_press_weight + 0.05
        }));
    }

    #[test]
    fn tactical_population_search_keeps_best_weight_candidate() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let single_options = SoccerEvolutionOptions {
            mutation_rate: 1.0,
            mutation_scale: 0.40,
            crossover_rate: 0.0,
            exploration_rate: 1.0,
            exploration_scale: 0.55,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 73,
        };
        let population_options = SoccerEvolutionOptions {
            population_size: 10,
            ..single_options
        };

        let single =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], single_options);
        let population =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], population_options);
        let weighted_summary =
            weighted_tactical_evolution_summary(&[(&summary, 1.0)], single_options)
                .expect("weighted summary");

        assert!(
            soccer_tactical_weight_search_score(&population, &weighted_summary)
                >= soccer_tactical_weight_search_score(&single, &weighted_summary) - 1e-12
        );
    }

    #[test]
    fn tactical_genome_search_crosses_over_elite_weight_sets() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut elite = base.clone();
        elite.attack_width_delta_weight = 1.15;
        elite.attack_flank_lane_weight = 1.35;
        elite.defense_contract_delta_weight = 1.45;
        elite.defense_compactness_score_weight = 1.05;
        let mut alternate = base.clone();
        alternate.attack_spacing_delta_weight = 1.10;
        alternate.defense_ball_depth_score_weight = 0.95;
        alternate.defender_midfielder_press_weight = 0.72;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 6,
            seed: 37,
        };

        let summary_only =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 10.0)], options);
        let genome = evolve_soccer_tactical_learning_weights_from_genomes(
            &base,
            &[
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &elite,
                    fitness: 10.0,
                },
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &alternate,
                    fitness: 1.0,
                },
            ],
            options,
        );
        let weighted_summary =
            weighted_tactical_evolution_summary(&[(&summary, 10.0), (&summary, 1.0)], options)
                .expect("weighted summary");

        assert!(genome.attack_flank_lane_weight >= elite.attack_flank_lane_weight);
        assert!(genome.defense_contract_delta_weight >= elite.defense_contract_delta_weight);
        assert!(
            soccer_tactical_weight_search_score(&genome, &weighted_summary)
                > soccer_tactical_weight_search_score(&summary_only, &weighted_summary)
        );
    }

    #[test]
    fn tactical_genome_blend_candidates_search_centroid_and_extrapolation() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut left = base.clone();
        left.attack_width_delta_weight = 0.90;
        left.attack_flank_lane_weight = 0.80;
        left.defense_contract_delta_weight = 0.95;
        let mut right = base.clone();
        right.attack_width_delta_weight = 1.70;
        right.attack_flank_lane_weight = 1.60;
        right.defense_contract_delta_weight = 1.75;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 43,
        };
        let parents = [
            SoccerTacticalLearningGenomeParent {
                summary: &summary,
                weights: &left,
                fitness: 1.0,
            },
            SoccerTacticalLearningGenomeParent {
                summary: &summary,
                weights: &right,
                fitness: 3.0,
            },
        ];
        let weighted_summary = weighted_tactical_evolution_summary(
            &parents
                .iter()
                .map(|parent| (parent.summary, parent.fitness))
                .collect::<Vec<_>>(),
            options,
        )
        .expect("weighted summary");
        let mut candidates = Vec::new();

        search_soccer_tactical_genome_blend_candidates(
            &base,
            &parents,
            options,
            &weighted_summary,
            |candidate| candidates.push(candidate),
        );

        assert!(candidates.len() >= 7);
        assert!((candidates[0].attack_width_delta_weight - 1.50).abs() < 1e-12);
        assert!((candidates[0].attack_flank_lane_weight - 1.40).abs() < 1e-12);
        assert!((candidates[0].defense_contract_delta_weight - 1.55).abs() < 1e-12);
        assert!(candidates[1].attack_width_delta_weight > candidates[0].attack_width_delta_weight);
        assert!(candidates[1].attack_flank_lane_weight > candidates[0].attack_flank_lane_weight);
        assert!(
            candidates[1].defense_contract_delta_weight
                > candidates[0].defense_contract_delta_weight
        );
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.attack_flank_lane_weight > candidates[0].attack_flank_lane_weight
                && candidate.attack_spacing_delta_weight > candidates[0].attack_spacing_delta_weight
        }));
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.defense_contract_delta_weight > candidates[0].defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > candidates[0].defense_compactness_score_weight
        }));
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.attack_flank_lane_weight > candidates[0].attack_flank_lane_weight
                && candidate.defense_contract_delta_weight
                    > candidates[0].defense_contract_delta_weight
        }));
    }

    #[test]
    fn tactical_gp_program_candidates_expand_flank_and_compact_search() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.14,
            mean_attack_flank_lane_score: 0.11,
            mean_attack_spacing_score: 0.26,
            mean_defense_contract_score: 0.13,
            mean_defense_spacing_score: 0.34,
            mean_defense_ball_gap_score: 0.40,
            mean_defense_role_press_score: 0.36,
            ..Default::default()
        };
        let pressure = soccer_tactical_search_pressure(&summary);
        let mut candidates = Vec::new();

        search_soccer_tactical_gp_program_candidates(
            &base,
            &base,
            &summary,
            pressure,
            &mut |candidate| candidates.push(candidate),
        );

        assert_eq!(candidates.len(), 10);
        assert!(candidates.iter().all(|candidate| {
            candidate.attack_flank_lane_weight <= 2.2
                && candidate.defense_contract_delta_weight <= 2.4
                && candidate.defender_midfielder_press_weight <= 1.6
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_delta_weight > base.attack_width_delta_weight
                && candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > base.defense_compactness_score_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            soccer_tactical_weight_search_score(candidate, &summary)
                > soccer_tactical_weight_search_score(&base, &summary)
        }));
    }

    #[test]
    fn tactical_mutated_gp_candidates_search_seeded_extra_shape_space() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.12,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.24,
            mean_defense_contract_score: 0.11,
            mean_defense_spacing_score: 0.31,
            mean_defense_ball_gap_score: 0.38,
            mean_defense_role_press_score: 0.34,
            ..Default::default()
        };
        let pressure = soccer_tactical_search_pressure(&summary);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.10,
            mutation_scale: 0.45,
            crossover_rate: 1.0,
            exploration_rate: 0.20,
            exploration_scale: 0.90,
            elite_weight_floor: 0.0,
            population_size: 12,
            seed: 59,
        };
        let mut candidates = Vec::new();

        search_soccer_tactical_mutated_gp_program_candidates(
            &base,
            &base,
            &summary,
            pressure,
            options,
            &mut |candidate| candidates.push(candidate),
        );

        assert_eq!(candidates.len(), options.population_size);
        assert!(candidates.iter().all(|candidate| {
            candidate.attack_flank_lane_weight <= 2.2
                && candidate.defense_contract_delta_weight <= 2.4
                && candidate.defender_midfielder_press_weight <= 1.6
                && candidate.midfielder_press_weight <= 1.6
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_delta_weight > base.attack_width_delta_weight
                && candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > base.defense_compactness_score_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
                && candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defender_midfielder_press_weight > base.defender_midfielder_press_weight
                || candidate.midfielder_press_weight > base.midfielder_press_weight
        }));
    }

    #[test]
    fn tactical_genome_search_can_extrapolate_beyond_elite_parents() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.12,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.28,
            mean_defense_contract_score: 0.14,
            mean_defense_spacing_score: 0.32,
            mean_defense_ball_gap_score: 0.42,
            mean_defense_role_press_score: 0.35,
            ..Default::default()
        };
        let mut left = base.clone();
        left.attack_width_delta_weight = 0.90;
        left.attack_flank_lane_weight = 0.85;
        left.defense_contract_delta_weight = 0.95;
        let mut right = base.clone();
        right.attack_width_delta_weight = 1.20;
        right.attack_flank_lane_weight = 1.15;
        right.defense_contract_delta_weight = 1.25;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 47,
        };

        let evolved = evolve_soccer_tactical_learning_weights_from_genomes(
            &base,
            &[
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &left,
                    fitness: 1.0,
                },
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &right,
                    fitness: 3.0,
                },
            ],
            options,
        );

        assert!(evolved.attack_width_delta_weight > right.attack_width_delta_weight);
        assert!(evolved.attack_flank_lane_weight > right.attack_flank_lane_weight);
        assert!(evolved.defense_contract_delta_weight > right.defense_contract_delta_weight);
    }

    #[test]
    fn queue_runner_keeps_policy_output_available() {
        let options = SoccerQPolicyOptions::default();
        let report = run_soccer_learning_queue(
            SoccerLearningQueueRunnerConfig {
                games: 2,
                parallel_games: 2,
                base_seed: 1888,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 2);
        assert_eq!(report.failed_games, 0);
        assert_eq!(report.episode_summaries.len(), 2);
        assert_eq!(
            report
                .episode_summaries
                .iter()
                .map(|summary| summary.seed)
                .collect::<Vec<_>>(),
            vec![1888, 1889]
        );
        assert!(report.tactical_summary.total_transitions > 0);
        assert!(report.tactical_summary.shape_transitions > 0);
    }

    #[test]
    fn queue_starting_batch_can_update_match_config() {
        let options = SoccerQPolicyOptions::default();
        let mut saw_starting_batch = false;
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 1999,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch { match_config, .. } = event {
                    match_config.tactical_learning.attack_flank_lane_weight = 0.91;
                    match_config.tactical_learning.defense_contract_delta_weight = 0.73;
                    saw_starting_batch = true;
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert!(saw_starting_batch);
        assert_eq!(report.completed_games, 1);
        assert_eq!(report.failed_games, 0);
    }

    #[test]
    fn queue_completed_game_carries_episode_starting_tactical_weights() {
        let options = SoccerQPolicyOptions::default();
        let mut completed_weights = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 2,
                parallel_games: 1,
                base_seed: 2007,
                match_config: MatchConfig {
                    duration_seconds: 0.0,
                    half_duration_seconds: 0.0,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                match event {
                    SoccerLearningQueueEvent::StartingBatch {
                        next_episode,
                        match_config,
                        ..
                    } => {
                        match_config.tactical_learning.attack_flank_lane_weight =
                            0.90 + next_episode as f64 * 0.05;
                        match_config.tactical_learning.defense_contract_delta_weight =
                            0.70 + next_episode as f64 * 0.03;
                    }
                    SoccerLearningQueueEvent::CompletedGame { game, .. } => {
                        completed_weights.push((
                            game.episode,
                            game.starting_tactical_learning.attack_flank_lane_weight,
                            game.starting_tactical_learning
                                .defense_contract_delta_weight,
                        ));
                    }
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 2);
        assert_eq!(report.failed_games, 0);
        assert_eq!(completed_weights.len(), 2);
        completed_weights.sort_by_key(|(episode, _, _)| *episode);
        assert_eq!(completed_weights[0].0, 0);
        assert!((completed_weights[0].1 - 0.90).abs() < 1e-12);
        assert!((completed_weights[0].2 - 0.70).abs() < 1e-12);
        assert_eq!(completed_weights[1].0, 1);
        assert!((completed_weights[1].1 - 0.95).abs() < 1e-12);
        assert!((completed_weights[1].2 - 0.73).abs() < 1e-12);
    }

    #[test]
    fn queue_starting_batch_fires_before_each_enqueued_game() {
        let options = SoccerQPolicyOptions::default();
        let mut start_episodes = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 3,
                parallel_games: 3,
                base_seed: 2010,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch { next_episode, .. } = event {
                    start_episodes.push(next_episode);
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 3);
        assert_eq!(report.failed_games, 0);
        assert_eq!(start_episodes, vec![0, 1, 2]);
    }

    #[test]
    fn queue_starting_batch_replaces_policy_for_next_game() {
        let options = SoccerQPolicyOptions::default();
        let refreshed = policy_with_action(12.0, 7);
        let mut refreshed_episodes = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2031,
                match_config: MatchConfig {
                    duration_seconds: 0.0,
                    half_duration_seconds: 0.0,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            policy_with_action(-2.0, 1),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    policies,
                    ..
                } = event
                {
                    refreshed_episodes.push(next_episode);
                    *policies = refreshed.clone();
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(refreshed_episodes, vec![0]);
        assert_eq!(report.completed_games, 1);
        assert_eq!(report.failed_games, 0);
        let entries = report.final_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 12.0).abs() < 1e-12);
        assert_eq!(entries[0].visits, 7);
    }

    #[test]
    fn queue_completed_game_can_replace_merged_policy() {
        let options = SoccerQPolicyOptions::default();
        let replacement = policy_with_action(8.0, 3);
        let mut replaced = false;
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2021,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::CompletedGame {
                    merged_policies, ..
                } = event
                {
                    *merged_policies = replacement.clone();
                    replaced = true;
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert!(replaced);
        let entries = report.final_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 8.0).abs() < 1e-12);
        assert_eq!(entries[0].visits, 3);
    }

    #[test]
    fn test_state_uses_expected_public_variants() {
        let state = test_state();
        assert_eq!(state.phase, TacticalPhase::Kickoff);
        assert_eq!(state.role, PlayerRole::Midfielder);
    }
}
