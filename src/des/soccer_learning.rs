//! Durable soccer self-play learning primitives.
//!
//! The soccer simulator owns the MDP/POMDP update mechanics. This module owns
//! the cross-run layer: outcome scoring, policy deltas, weighted merge, simple
//! evolutionary spawning, and a queue runner that keeps worker slots full.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::des::general::soccer::{
    MatchConfig, MatchSummary, SoccerMatch, SoccerNeuralNetworkSnapshot, SoccerQEntry,
    SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningSummary,
    SoccerTeamQPolicies, Team,
};

pub const SOCCER_LEARNING_FIXED_SCALE: i64 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerEvolutionOptions {
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub elite_weight_floor: f64,
    pub seed: u64,
}

impl Default for SoccerEvolutionOptions {
    fn default() -> Self {
        Self {
            mutation_rate: 0.025,
            mutation_scale: 0.18,
            elite_weight_floor: 0.05,
            seed: 2026,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    let mut entries = Vec::new();
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
    mutate_accumulators(&mut action_accumulators, &mut rng, options);
    mutate_accumulators(&mut target_accumulators, &mut rng, options);

    build_policies_from_accumulators(
        first_parent.home.options.clone(),
        first_parent.away.options.clone(),
        action_accumulators,
        target_accumulators,
    )
}

pub fn run_soccer_learning_game(
    episode: usize,
    mut config: MatchConfig,
    starting_policies: SoccerTeamQPolicies,
    neural_drain_timeout: Duration,
) -> Result<SoccerLearningCompletedGame, String> {
    let started = Instant::now();
    config.seed = config.seed.wrapping_add(episode as u32);
    let seed = config.seed as u64;
    let total_ticks = config.total_ticks();
    let mut sim = SoccerMatch::default_11v11(config).with_team_policies(starting_policies.clone());

    for _ in 0..total_ticks {
        sim.run_time_step();
    }

    sim.drain_neural_learning(neural_drain_timeout);
    let policies = sim
        .team_policies()
        .cloned()
        .ok_or_else(|| "soccer learning produced no team policies".to_string())?;
    let artifact = sim.team_policy_artifact();
    let summary = artifact.summary.clone();
    let score = soccer_learning_run_score(&summary);
    let delta = soccer_policy_delta_entries(&starting_policies, &policies, &score);
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
        policies,
        score,
        delta,
        neural_network: artifact.learning.neural_network.clone(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn run_soccer_learning_queue(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
) -> Result<SoccerLearningQueueReport, String> {
    run_soccer_learning_queue_with_observer(config, initial_policies, |_, _| Ok(()))
}

pub fn run_soccer_learning_queue_with_observer<F>(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
    mut on_completed_game: F,
) -> Result<SoccerLearningQueueReport, String>
where
    F: FnMut(&SoccerLearningCompletedGame, &SoccerTeamQPolicies) -> Result<(), String>,
{
    let started = Instant::now();
    let parallel_games = config.parallel_games.clamp(1, 100);
    let (tx, rx) = mpsc::channel();
    let mut active = 0usize;
    let mut next_episode = 0usize;
    let mut completed_games = 0usize;
    let mut failed_games = 0usize;
    let mut policies = initial_policies;
    let mut episode_summaries = Vec::new();
    let mut tactical_summary = SoccerTacticalLearningSummary::default();
    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut latest_neural_network = None::<SoccerNeuralNetworkSnapshot>;

    while completed_games + failed_games < config.games {
        while active < parallel_games && next_episode < config.games {
            let tx = tx.clone();
            let episode = next_episode;
            let starting_policies = policies.clone();
            let mut match_config = config.match_config.clone();
            match_config.seed = config.base_seed;
            let neural_drain_timeout = config.neural_drain_timeout;
            thread::spawn(move || {
                let result = run_soccer_learning_game(
                    episode,
                    match_config,
                    starting_policies,
                    neural_drain_timeout,
                );
                let _ = tx.send((episode, result));
            });
            active += 1;
            next_episode += 1;
        }

        let (_, game_result) = rx
            .recv()
            .map_err(|err| format!("soccer learning queue worker channel closed: {err}"))?;
        active = active.saturating_sub(1);

        match game_result {
            Ok(game) => {
                let merged = merge_soccer_policy_deltas(&policies, &[game.delta.clone()], 1.0)?;
                policies = merged;
                policies.prune(
                    config.prune_action_entries_per_team,
                    config.prune_target_entries_per_team,
                    config.min_policy_visits,
                );
                on_completed_game(&game, &policies)?;
                if let Some(snapshot) = game.neural_network.clone() {
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
    entries: impl Iterator<Item = EntryValue>,
) -> BTreeMap<PolicyEntryKey, EntryValue> {
    entries
        .map(|entry| (policy_entry_key(team, entry_kind, &entry), entry))
        .collect()
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
    accumulators.insert(
        key,
        MergeAccumulator {
            state_key: entry.state_key,
            action: entry.action,
            weighted_value_sum: entry.value * effective_visits,
            effective_visits,
            display_visits: entry.visits.max(1),
            target_fine_cell_id: entry.target_fine_cell_id,
            target_tactical_cell_id: entry.target_tactical_cell_id,
            target_macro_cell_id: entry.target_macro_cell_id,
            target_root_cell_id: entry.target_root_cell_id,
        },
    );
}

fn build_policies_from_accumulators(
    home_options: SoccerQPolicyOptions,
    away_options: SoccerQPolicyOptions,
    action_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
    target_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
) -> Result<SoccerTeamQPolicies, String> {
    let mut home_entries = Vec::new();
    let mut away_entries = Vec::new();
    let mut home_targets = Vec::new();
    let mut away_targets = Vec::new();

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
        assert!(report.tactical_summary.total_transitions > 0);
        assert!(report.tactical_summary.shape_transitions > 0);
    }

    #[test]
    fn test_state_uses_expected_public_variants() {
        let state = test_state();
        assert_eq!(state.phase, TacticalPhase::Kickoff);
        assert_eq!(state.role, PlayerRole::Midfielder);
    }
}
