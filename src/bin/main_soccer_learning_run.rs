//! Run accelerated soccer self-play and persist learned MDP/POMDP policies.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    soccer_moment_records_from_jsonl, soccer_moment_records_to_learning_dataset, MatchConfig,
    SoccerMatch, SoccerMomentWindow, SoccerNeuralLearningBackend, SoccerNeuralLearningConfig,
    SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningSummary, SoccerTacticalLearningWeights, SoccerTeamPolicyArtifact,
    SoccerTeamQPolicies,
};
use des_engine::des::soccer_learning::{
    soccer_learning_run_score, soccer_policy_delta_entries, SoccerLearningCompletedGame,
};
use des_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use serde::Serialize;

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|_| {
        invalid_data(format!("{name} must be an unsigned integer, got {value:?}")).into()
    })
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{name} must be a u32, got {value:?}")).into())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{name} must be a finite number, got {value:?}")))?;
    if !parsed.is_finite() {
        return Err(invalid_data(format!("{name} must be finite, got {value:?}")).into());
    }
    Ok(parsed)
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(invalid_data(format!("{name} must be a boolean, got {value:?}")).into()),
    }
}

fn env_f64_alias(primary: &str, alias: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_f64(primary, default)
    } else {
        env_f64(alias, default)
    }
}

fn env_usize_alias(primary: &str, alias: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_usize(primary, default)
    } else {
        env_usize(alias, default)
    }
}

fn env_bool_alias(primary: &str, alias: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_bool(primary, default)
    } else {
        env_bool(alias, default)
    }
}

fn neural_backend_label(backend: SoccerNeuralLearningBackend) -> &'static str {
    match backend {
        SoccerNeuralLearningBackend::Inline => "inline",
        SoccerNeuralLearningBackend::Threaded => "threaded",
    }
}

fn env_neural_learning_backend(
    default: SoccerNeuralLearningBackend,
) -> Result<SoccerNeuralLearningBackend, Box<dyn Error>> {
    let Some(value) =
        env_value("SOCCER_NEURAL_LEARNING_BACKEND").or_else(|| env_value("SOCCER_NEURAL_BACKEND"))
    else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "threaded" | "thread" | "worker" => Ok(SoccerNeuralLearningBackend::Threaded),
        "inline" | "sync" => Ok(SoccerNeuralLearningBackend::Inline),
        _ => Err(invalid_data(format!(
            "SOCCER_NEURAL_LEARNING_BACKEND must be inline or threaded, got {value:?}"
        ))
        .into()),
    }
}

fn env_neural_learning_config() -> Result<SoccerNeuralLearningConfig, Box<dyn Error>> {
    let default = SoccerNeuralLearningConfig::default();
    Ok(SoccerNeuralLearningConfig {
        enabled: env_bool_alias(
            "SOCCER_NEURAL_LEARNING_ENABLED",
            "SOCCER_NEURAL_LEARNING",
            default.enabled,
        )?,
        backend: env_neural_learning_backend(default.backend)?,
        learning_rate: env_f64_alias(
            "SOCCER_NEURAL_LEARNING_RATE",
            "SOCCER_NEURAL_RATE",
            default.learning_rate,
        )?,
        batch_size: env_usize_alias(
            "SOCCER_NEURAL_BATCH_SIZE",
            "SOCCER_NEURAL_LEARNING_BATCH_SIZE",
            default.batch_size,
        )?,
        train_every_ticks: env_usize_alias(
            "SOCCER_NEURAL_TRAIN_EVERY_TICKS",
            "SOCCER_NEURAL_LEARNING_TRAIN_EVERY_TICKS",
            default.train_every_ticks,
        )?,
        max_batches_per_tick: env_usize_alias(
            "SOCCER_NEURAL_MAX_BATCHES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_MAX_BATCHES_PER_TICK",
            default.max_batches_per_tick,
        )?,
        hidden_units: env_usize_alias(
            "SOCCER_NEURAL_HIDDEN_UNITS",
            "SOCCER_NEURAL_LEARNING_HIDDEN_UNITS",
            default.hidden_units,
        )?,
        target_scale: env_f64_alias(
            "SOCCER_NEURAL_TARGET_SCALE",
            "SOCCER_NEURAL_LEARNING_TARGET_SCALE",
            default.target_scale,
        )?,
        max_pending_batches: env_usize_alias(
            "SOCCER_NEURAL_MAX_PENDING_BATCHES",
            "SOCCER_NEURAL_LEARNING_MAX_PENDING_BATCHES",
            default.max_pending_batches,
        )?,
        replay_capacity: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_CAPACITY",
            "SOCCER_NEURAL_LEARNING_REPLAY_CAPACITY",
            default.replay_capacity,
        )?,
        replay_samples_per_tick: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_REPLAY_SAMPLES_PER_TICK",
            default.replay_samples_per_tick,
        )?,
        target_clip: env_f64_alias(
            "SOCCER_NEURAL_TARGET_CLIP",
            "SOCCER_NEURAL_LEARNING_TARGET_CLIP",
            default.target_clip,
        )?,
    })
}

fn q_entry_order(a: &SoccerQEntry, b: &SoccerQEntry) -> std::cmp::Ordering {
    b.visits.cmp(&a.visits).then_with(|| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn target_entry_order(a: &SoccerQTargetEntry, b: &SoccerQTargetEntry) -> std::cmp::Ordering {
    b.visits.cmp(&a.visits).then_with(|| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn compact_training_artifact_for_export(
    artifact: &SoccerSelfPlayTrainingArtifact,
    max_entries_per_policy: usize,
) -> SoccerSelfPlayTrainingArtifact {
    let mut export = artifact.clone();
    if max_entries_per_policy == 0 {
        return export;
    }
    if export.home_entries.len() > max_entries_per_policy {
        export.home_entries.sort_by(q_entry_order);
        export.home_entries.truncate(max_entries_per_policy);
    }
    if export.away_entries.len() > max_entries_per_policy {
        export.away_entries.sort_by(q_entry_order);
        export.away_entries.truncate(max_entries_per_policy);
    }
    if export.home_target_entries.len() > max_entries_per_policy {
        export.home_target_entries.sort_by(target_entry_order);
        export.home_target_entries.truncate(max_entries_per_policy);
    }
    if export.away_target_entries.len() > max_entries_per_policy {
        export.away_target_entries.sort_by(target_entry_order);
        export.away_target_entries.truncate(max_entries_per_policy);
    }
    export
}

fn env_tactical_learning_weights() -> Result<SoccerTacticalLearningWeights, Box<dyn Error>> {
    let default = SoccerTacticalLearningWeights::default();
    Ok(SoccerTacticalLearningWeights {
        attack_spacing_delta_weight: env_f64(
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            default.attack_spacing_delta_weight,
        )?,
        attack_spacing_score_weight: env_f64(
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            default.attack_spacing_score_weight,
        )?,
        attack_width_delta_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            default.attack_width_delta_weight,
        )?,
        attack_width_score_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            default.attack_width_score_weight,
        )?,
        attack_flank_lane_weight: env_f64(
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            default.attack_flank_lane_weight,
        )?,
        defense_spacing_delta_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            default.defense_spacing_delta_weight,
        )?,
        defense_spacing_score_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            default.defense_spacing_score_weight,
        )?,
        defense_contract_delta_weight: env_f64(
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            default.defense_contract_delta_weight,
        )?,
        defense_compactness_score_weight: env_f64(
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            default.defense_compactness_score_weight,
        )?,
    })
}

fn artifact_minutes_label(minutes: f64) -> String {
    if (minutes - minutes.round()).abs() < 1e-9 {
        format!("{:.0}", minutes)
    } else {
        format!("{:.2}", minutes).replace('.', "p")
    }
}

fn default_artifact_path_in_run_dir(
    run_dir: &Path,
    games: usize,
    minutes: f64,
    shard_index: usize,
    shard_count: usize,
) -> PathBuf {
    let minutes = artifact_minutes_label(minutes);
    let file_name = if shard_count > 1 {
        format!(
            "final-policy-shard-{}-of-{}-{}x{}.json",
            shard_index, shard_count, games, minutes
        )
    } else {
        "final-policy.json".to_string()
    };
    run_dir.join(file_name)
}

fn validate_run_settings(
    games: usize,
    halves: usize,
    half_minutes: f64,
    minutes: f64,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    parallel_games: usize,
    shard_seed_stride: u32,
    options: &SoccerQPolicyOptions,
    tactical_learning: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    if games == 0 {
        return Err(invalid_data("SOCCER_GAMES must be at least 1").into());
    }
    if !(1..=8).contains(&halves) {
        return Err(invalid_data("SOCCER_HALVES must be between 1 and 8").into());
    }
    if !half_minutes.is_finite() || half_minutes <= 0.0 || half_minutes > 120.0 {
        return Err(invalid_data("SOCCER_HALF_MINUTES must be finite and in (0, 120]").into());
    }
    if !minutes.is_finite() || minutes <= 0.0 || minutes > 24.0 * 60.0 {
        return Err(invalid_data("SOCCER_MINUTES must be finite and in (0, 1440]").into());
    }
    if env_value("SOCCER_MINUTES").is_some() {
        let expected = half_minutes * halves as f64;
        if (minutes - expected).abs() > 1e-6 {
            return Err(invalid_data(format!(
                "SOCCER_MINUTES ({minutes}) must equal SOCCER_HALF_MINUTES * SOCCER_HALVES ({expected})"
            ))
            .into());
        }
    }
    if !period_break_recovery_seconds.is_finite()
        || !(0.0..=60.0 * 60.0).contains(&period_break_recovery_seconds)
    {
        return Err(invalid_data(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS must be finite and in [0, 3600]",
        )
        .into());
    }
    if !dt_seconds.is_finite() || !(0.01..=5.0).contains(&dt_seconds) {
        return Err(invalid_data("SOCCER_DT_SECONDS must be finite and in [0.01, 5.0]").into());
    }
    if learning_interval_ticks == 0 {
        return Err(invalid_data("SOCCER_LEARNING_INTERVAL_TICKS must be at least 1").into());
    }
    if !(1..=100).contains(&parallel_games) {
        return Err(invalid_data("SOCCER_PARALLEL_GAMES must be between 1 and 100").into());
    }
    if shard_seed_stride == 0 {
        return Err(invalid_data("SOCCER_SHARD_SEED_STRIDE must be at least 1").into());
    }
    validate_soccer_q_policy_options_for_runner(options)?;
    validate_tactical_learning_weights(tactical_learning)?;
    Ok(())
}

fn validate_soccer_q_policy_options_for_runner(
    options: &SoccerQPolicyOptions,
) -> Result<(), Box<dyn Error>> {
    if !options.alpha.is_finite() || !(0.0..=1.0).contains(&options.alpha) {
        return Err(invalid_data("SOCCER_ALPHA must be finite and in [0, 1]").into());
    }
    if !options.gamma.is_finite() || !(0.0..=1.0).contains(&options.gamma) {
        return Err(invalid_data("SOCCER_GAMMA must be finite and in [0, 1]").into());
    }
    Ok(())
}

fn validate_tactical_learning_weights(
    tactical_learning: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    let weights = [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            tactical_learning.attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            tactical_learning.attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            tactical_learning.attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            tactical_learning.attack_width_score_weight,
        ),
        (
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            tactical_learning.attack_flank_lane_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            tactical_learning.defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            tactical_learning.defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            tactical_learning.defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            tactical_learning.defense_compactness_score_weight,
        ),
    ];
    for (name, value) in weights {
        if !value.is_finite() {
            return Err(invalid_data(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn write_episode_log(
    file: &mut std::fs::File,
    episode: &SoccerSelfPlayEpisodeSummary,
) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *file, episode)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}
fn action_summary(entries: &[SoccerQEntry]) -> Vec<(String, u64, f64)> {
    let mut by_action: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    for entry in entries {
        let visits = u64::from(entry.visits.max(1));
        let item = by_action.entry(entry.action.clone()).or_insert((0, 0.0));
        item.0 += visits;
        item.1 += entry.value * visits as f64;
    }
    let mut summary = by_action
        .into_iter()
        .map(|(action, (visits, weighted_value))| {
            let mean_value = if visits == 0 {
                0.0
            } else {
                weighted_value / visits as f64
            };
            (action, visits, mean_value)
        })
        .collect::<Vec<_>>();
    summary.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    summary
}

#[derive(Debug)]
struct CompletedGame {
    episode_summary: SoccerSelfPlayEpisodeSummary,
    artifact: SoccerTeamPolicyArtifact,
    policies: SoccerTeamQPolicies,
    elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GameManifestEntry {
    episode: usize,
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_kind: Option<String>,
    score_home: u32,
    score_away: u32,
    ticks: u64,
    simulated_seconds: f64,
    transitions: usize,
    home_policy_entries: usize,
    away_policy_entries: usize,
    home_policy_target_entries: usize,
    away_policy_target_entries: usize,
    elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunManifest {
    run_id: String,
    run_dir: String,
    game_dir: String,
    games: usize,
    parallel_games: usize,
    shard_index: usize,
    shard_count: usize,
    base_seed: u32,
    effective_seed: u32,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: String,
    learned_params_path: String,
    checkpoint_artifact_path: String,
    episode_log_path: String,
    checkpoint_interval_games: usize,
    artifact_max_entries_per_policy: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
    postgres_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_experiment_slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_experiment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_last_policy_version_id: Option<String>,
    postgres_persisted_games: usize,
    write_game_artifacts: bool,
    game_artifact_mode: String,
    game_artifacts: Vec<GameManifestEntry>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ActionSummaryEntry {
    action: String,
    visits: u64,
    mean_q: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactGameArtifact {
    artifact_kind: String,
    episode: usize,
    seed: u64,
    config: MatchConfig,
    summary: des_engine::des::general::soccer::MatchSummary,
    tactical_summary: SoccerTacticalLearningSummary,
    transitions: usize,
    home_policy_entries: usize,
    away_policy_entries: usize,
    home_policy_target_entries: usize,
    away_policy_target_entries: usize,
    elapsed_seconds: f64,
    home_action_summary: Vec<ActionSummaryEntry>,
    away_action_summary: Vec<ActionSummaryEntry>,
}

fn io_error(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::Other, message.into())
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, value)?;
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        let _ = fs::File::open(parent).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

fn default_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("soccer-learning-{seconds}")
}

fn policies_from_self_play_artifact(
    artifact: &SoccerSelfPlayTrainingArtifact,
) -> Result<SoccerTeamQPolicies, String> {
    Ok(SoccerTeamQPolicies {
        home: SoccerQPolicy::from_entries_with_targets(
            artifact.options.clone(),
            &artifact.home_entries,
            &artifact.home_target_entries,
        )?,
        away: SoccerQPolicy::from_entries_with_targets(
            artifact.options.clone(),
            &artifact.away_entries,
            &artifact.away_target_entries,
        )?,
    })
}

fn load_initial_policies(
    path: Option<&str>,
    options: SoccerQPolicyOptions,
) -> Result<SoccerTeamQPolicies, Box<dyn Error>> {
    let Some(path) = path else {
        return Ok(SoccerTeamQPolicies::new(options));
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(artifact) = serde_json::from_str::<SoccerSelfPlayTrainingArtifact>(&raw) {
        return policies_from_self_play_artifact(&artifact).map_err(|err| invalid_data(err).into());
    }
    if let Ok(params) = serde_json::from_str::<SoccerSelfPlayLearnedParams>(&raw) {
        return SoccerTeamQPolicies::from_learned_params(&params)
            .map_err(|err| invalid_data(err).into());
    }
    if let Ok(artifact) = serde_json::from_str::<SoccerTeamPolicyArtifact>(&raw) {
        return SoccerTeamQPolicies::from_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    Err(invalid_data(format!(
        "resume artifact {path} is neither a self-play, learned-params, nor team-policy artifact"
    ))
    .into())
}

fn run_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: SoccerTeamQPolicies,
    adversarial_moment_windows: Vec<SoccerMomentWindow>,
    print_progress: bool,
) -> Result<CompletedGame, String> {
    let started = Instant::now();
    let total_ticks = config.total_ticks();
    let episode_seed = config.seed as u64;
    let progress_interval = (total_ticks / 9).max(1);
    let mut sim = SoccerMatch::default_11v11(config).with_team_policies(starting_policies);
    for window in adversarial_moment_windows {
        sim.remember_adversarial_moment_window(window);
    }

    for tick_idx in 0..total_ticks {
        sim.run_time_step();
        let completed_ticks = tick_idx + 1;
        if print_progress
            && (completed_ticks == total_ticks || completed_ticks % progress_interval == 0)
        {
            println!(
                "progress_game={} seed={} ticks={}/{}",
                episode + 1,
                episode_seed,
                completed_ticks,
                total_ticks
            );
            let _ = std::io::stdout().flush();
        }
    }

    let artifact = sim.team_policy_artifact();
    let policies = SoccerTeamQPolicies::from_artifact(&artifact)?;
    let episode_summary = SoccerSelfPlayEpisodeSummary {
        episode,
        seed: episode_seed,
        summary: artifact.summary.clone(),
        transitions: artifact.learning.total_transitions,
        home_policy_entries: artifact.home_entries.len(),
        home_policy_target_entries: artifact.home_target_entries.len(),
        away_policy_entries: artifact.away_entries.len(),
        away_policy_target_entries: artifact.away_target_entries.len(),
    };

    Ok(CompletedGame {
        episode_summary,
        artifact,
        policies,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn merge_policy_delta(dst: &mut SoccerQPolicy, before: &SoccerQPolicy, after: &SoccerQPolicy) {
    for (key, after_visits) in &after.visits {
        let before_visits = before.visits.get(key).copied().unwrap_or(0);
        if *after_visits <= before_visits {
            continue;
        }
        let delta_visits = *after_visits - before_visits;
        let after_value = after.q_values.get(key).copied().unwrap_or(0.0);
        let dst_visits = dst.visits.get(key).copied().unwrap_or(0);
        let dst_value = dst.q_values.get(key).copied().unwrap_or(0.0);
        let merged_visits = dst_visits.saturating_add(delta_visits);
        let merged_value = if merged_visits == 0 {
            after_value
        } else {
            (dst_value * f64::from(dst_visits) + after_value * f64::from(delta_visits))
                / f64::from(merged_visits)
        };
        dst.q_values.insert(key.clone(), merged_value);
        dst.visits.insert(key.clone(), merged_visits);
    }

    for (key, after_visits) in &after.target_visits {
        let before_visits = before.target_visits.get(key).copied().unwrap_or(0);
        if *after_visits <= before_visits {
            continue;
        }
        let delta_visits = *after_visits - before_visits;
        let after_value = after.target_values.get(key).copied().unwrap_or(0.0);
        let dst_visits = dst.target_visits.get(key).copied().unwrap_or(0);
        let dst_value = dst.target_values.get(key).copied().unwrap_or(0.0);
        let merged_visits = dst_visits.saturating_add(delta_visits);
        let merged_value = if merged_visits == 0 {
            after_value
        } else {
            (dst_value * f64::from(dst_visits) + after_value * f64::from(delta_visits))
                / f64::from(merged_visits)
        };
        dst.target_values.insert(key.clone(), merged_value);
        dst.target_visits.insert(key.clone(), merged_visits);
    }
}

fn merge_team_policy_delta(
    dst: &mut SoccerTeamQPolicies,
    before: &SoccerTeamQPolicies,
    after: &SoccerTeamQPolicies,
) {
    merge_policy_delta(&mut dst.home, &before.home, &after.home);
    merge_policy_delta(&mut dst.away, &before.away, &after.away);
}

fn soccer_learning_completed_game_from_completed(
    game: &CompletedGame,
    starting_policies: &SoccerTeamQPolicies,
) -> SoccerLearningCompletedGame {
    let summary = game.episode_summary.summary.clone();
    let score = soccer_learning_run_score(&summary);
    let delta = soccer_policy_delta_entries(starting_policies, &game.policies, &score);
    SoccerLearningCompletedGame {
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        summary,
        episode_summary: game.episode_summary.clone(),
        tactical_summary: game.artifact.tactical_summary.clone(),
        policies: game.policies.clone(),
        score,
        delta,
        elapsed_seconds: game.elapsed_seconds,
    }
}

fn soccer_learning_pg_version_label(run_id: &str, shard_index: usize, episode: usize) -> String {
    let suffix = format!("-s{:03}-e{:06}", shard_index, episode + 1);
    let max_prefix_len = 160usize.saturating_sub(suffix.len());
    let mut prefix = run_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-'))
        .take(max_prefix_len)
        .collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("run");
    }
    format!("{prefix}{suffix}")
}

fn soccer_learning_pg_runner_id(run_id: &str, shard_index: usize, shard_count: usize) -> String {
    let suffix = format!("-shard-{}-of-{}", shard_index, shard_count);
    let max_prefix_len = 200usize.saturating_sub(suffix.len());
    let mut prefix = run_id.chars().take(max_prefix_len).collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("soccer-self-play");
    }
    format!("{prefix}{suffix}")
}

fn print_completed_game(game: &CompletedGame) {
    let stats = &game.episode_summary.summary.stats;
    println!(
        "completed_game={} seed={} score={}-{} shots={} on_target={} blocks={} pass_completion={}/{} interceptions={} elapsed={:.2}s",
        game.episode_summary.episode + 1,
        game.episode_summary.seed,
        game.episode_summary.summary.score_home,
        game.episode_summary.summary.score_away,
        stats.shots_home + stats.shots_away,
        stats.shots_on_target_home + stats.shots_on_target_away,
        stats.shot_blocks_home + stats.shot_blocks_away,
        stats.passes_completed_home + stats.passes_completed_away,
        stats.passes_attempted_home + stats.passes_attempted_away,
        stats.interceptions_home + stats.interceptions_away,
        game.elapsed_seconds,
    );
    let _ = std::io::stdout().flush();
}

fn game_manifest_entry(game: &CompletedGame, artifact_path: Option<String>) -> GameManifestEntry {
    GameManifestEntry {
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        artifact_path,
        artifact_kind: None,
        score_home: game.episode_summary.summary.score_home,
        score_away: game.episode_summary.summary.score_away,
        ticks: game.episode_summary.summary.ticks,
        simulated_seconds: game.episode_summary.summary.simulated_seconds,
        transitions: game.episode_summary.transitions,
        home_policy_entries: game.episode_summary.home_policy_entries,
        away_policy_entries: game.episode_summary.away_policy_entries,
        home_policy_target_entries: game.episode_summary.home_policy_target_entries,
        away_policy_target_entries: game.episode_summary.away_policy_target_entries,
        elapsed_seconds: game.elapsed_seconds,
    }
}

fn action_summary_entries(entries: &[SoccerQEntry]) -> Vec<ActionSummaryEntry> {
    action_summary(entries)
        .into_iter()
        .take(12)
        .map(|(action, visits, mean_q)| ActionSummaryEntry {
            action,
            visits,
            mean_q,
        })
        .collect()
}

fn compact_game_artifact(game: &CompletedGame) -> CompactGameArtifact {
    CompactGameArtifact {
        artifact_kind: "soccer-compact-game-summary".to_string(),
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        config: game.artifact.config.clone(),
        summary: game.episode_summary.summary.clone(),
        tactical_summary: game.artifact.tactical_summary.clone(),
        transitions: game.episode_summary.transitions,
        home_policy_entries: game.episode_summary.home_policy_entries,
        away_policy_entries: game.episode_summary.away_policy_entries,
        home_policy_target_entries: game.episode_summary.home_policy_target_entries,
        away_policy_target_entries: game.episode_summary.away_policy_target_entries,
        elapsed_seconds: game.elapsed_seconds,
        home_action_summary: action_summary_entries(&game.artifact.home_entries),
        away_action_summary: action_summary_entries(&game.artifact.away_entries),
    }
}

fn self_play_artifact_from_policies(
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    tactical_summary: SoccerTacticalLearningSummary,
    episode_summaries: Vec<SoccerSelfPlayEpisodeSummary>,
    policies: &SoccerTeamQPolicies,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
        tactical_learning: config.tactical_learning.clone(),
        tactical_summary,
        config,
        options,
        episodes: episode_summaries,
        home_entries: policies.home.entries(),
        home_target_entries: policies.home.target_entries(),
        away_entries: policies.away.entries(),
        away_target_entries: policies.away.target_entries(),
    }
}

fn run_manifest(
    run_id: &str,
    run_dir: &Path,
    game_dir: &Path,
    games: usize,
    parallel_games: usize,
    shard_index: usize,
    shard_count: usize,
    base_seed: u32,
    effective_seed: u32,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: &Path,
    learned_params_path: &Path,
    checkpoint_artifact_path: &Path,
    episode_log_path: &Path,
    checkpoint_interval_games: usize,
    artifact_max_entries_per_policy: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
    postgres_enabled: bool,
    postgres_experiment_slug: Option<String>,
    postgres_experiment_id: Option<String>,
    postgres_last_policy_version_id: Option<String>,
    postgres_persisted_games: usize,
    write_game_artifacts: bool,
    game_artifact_mode: &str,
    game_artifacts: Vec<GameManifestEntry>,
) -> RunManifest {
    RunManifest {
        run_id: run_id.to_string(),
        run_dir: run_dir.display().to_string(),
        game_dir: game_dir.display().to_string(),
        games,
        parallel_games,
        shard_index,
        shard_count,
        base_seed,
        effective_seed,
        config,
        options,
        final_artifact_path: final_artifact_path.display().to_string(),
        learned_params_path: learned_params_path.display().to_string(),
        checkpoint_artifact_path: checkpoint_artifact_path.display().to_string(),
        episode_log_path: episode_log_path.display().to_string(),
        checkpoint_interval_games,
        artifact_max_entries_per_policy,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_path,
        moment_replay_records,
        moment_replay_transitions,
        moment_replay_passes,
        moment_replay_reward_scale,
        postgres_enabled,
        postgres_experiment_slug,
        postgres_experiment_id,
        postgres_last_policy_version_id,
        postgres_persisted_games,
        write_game_artifacts,
        game_artifact_mode: game_artifact_mode.to_string(),
        game_artifacts,
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let games = env_usize("SOCCER_GAMES", 100)?;
    let halves = env_usize("SOCCER_HALVES", 2)?;
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0)?;
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64)?;
    let effective_minutes = minutes;
    let period_break_recovery_seconds = env_f64_alias(
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
        "SOCCER_HALFTIME_RECOVERY_SECONDS",
        900.0,
    )?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
    let parallel_games = env_usize("SOCCER_PARALLEL_GAMES", 1)?;
    let checkpoint_interval_games = env_usize("SOCCER_CHECKPOINT_INTERVAL_GAMES", 10)?;
    let artifact_max_entries_per_policy =
        env_usize("SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY", 10_000)?;
    let max_policy_entries_per_team = env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?;
    let max_policy_target_entries_per_team = env_usize(
        "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
        max_policy_entries_per_team,
    )?;
    let min_policy_visits = env_u32("SOCCER_MIN_POLICY_VISITS", 0)?;
    let moment_replay_path = env_value("SOCCER_MOMENT_REPLAY_PATH");
    let moment_replay_limit = env_usize("SOCCER_MOMENT_REPLAY_LIMIT", 0)?;
    let moment_replay_passes = env_usize("SOCCER_MOMENT_REPLAY_PASSES", 1)?;
    let moment_replay_reward_scale = env_f64("SOCCER_MOMENT_REPLAY_REWARD_SCALE", 1.0)?;
    let write_game_artifacts = env_bool("SOCCER_WRITE_GAME_ARTIFACTS", true)?;
    let write_final_artifacts = env_bool("SOCCER_WRITE_FINAL_ARTIFACTS", true)?;
    let write_checkpoint_artifacts =
        env_bool("SOCCER_WRITE_CHECKPOINT_ARTIFACTS", write_final_artifacts)?;
    let game_artifact_mode = env_value("SOCCER_GAME_ARTIFACT_MODE")
        .unwrap_or_else(|| "summary".to_string())
        .to_ascii_lowercase();
    if !matches!(game_artifact_mode.as_str(), "summary" | "full") {
        return Err(invalid_data("SOCCER_GAME_ARTIFACT_MODE must be summary or full").into());
    }
    if moment_replay_passes == 0 {
        return Err(invalid_data("SOCCER_MOMENT_REPLAY_PASSES must be at least 1").into());
    }
    let learning_logging_enabled = env_bool("SOCCER_LEARNING_LOGGING", false)?;
    let shard_index = env_usize("SOCCER_SHARD_INDEX", 0)?;
    let shard_count = env_usize("SOCCER_SHARD_COUNT", 1)?;
    if shard_count == 0 {
        return Err(invalid_data("SOCCER_SHARD_COUNT must be at least 1").into());
    }
    if shard_index >= shard_count {
        return Err(invalid_data("SOCCER_SHARD_INDEX must be less than SOCCER_SHARD_COUNT").into());
    }
    let seed = env_u32("SOCCER_SEED", 2026)?;
    let shard_seed_stride = env_u32("SOCCER_SHARD_SEED_STRIDE", 1_000_000)?;
    let effective_seed = seed.wrapping_add((shard_index as u32).wrapping_mul(shard_seed_stride));
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20)?,
        gamma: env_f64("SOCCER_GAMMA", 0.96)?,
    };
    let tactical_learning = env_tactical_learning_weights()?;
    validate_run_settings(
        games,
        halves,
        half_minutes,
        minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        parallel_games,
        shard_seed_stride,
        &options,
        &tactical_learning,
    )?;
    let half_duration_seconds = half_minutes * 60.0;
    let halftime_fatigue_recovery = if half_duration_seconds > 0.0 {
        (period_break_recovery_seconds / half_duration_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let neural_learning = env_neural_learning_config()?;
    let default_config = MatchConfig::default();
    let adversarial_embedding_exploitation_enabled = env_bool_alias(
        "SOCCER_ADVERSARIAL_EMBEDDING_EXPLOITATION_ENABLED",
        "SOCCER_ADVERSARIAL_EMBEDDINGS",
        default_config.adversarial_embedding_exploitation_enabled,
    )?;
    let adversarial_embedding_memory_limit = env_usize_alias(
        "SOCCER_ADVERSARIAL_EMBEDDING_MEMORY_LIMIT",
        "SOCCER_ADVERSARIAL_MOMENT_MEMORY_LIMIT",
        default_config.adversarial_embedding_memory_limit,
    )?;
    let config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        halves: halves as u8,
        half_duration_seconds,
        halftime_fatigue_recovery,
        period_count: halves,
        period_break_recovery_seconds,
        learning_enabled: true,
        learning_logging_enabled,
        learning_interval_ticks,
        tactical_learning: tactical_learning.clone(),
        neural_learning: neural_learning.clone(),
        adversarial_embedding_exploitation_enabled,
        adversarial_embedding_memory_limit,
        max_human_players: 0,
        seed: effective_seed,
        ..default_config
    };
    let run_id = env_value("SOCCER_RUN_ID").unwrap_or_else(default_run_id);
    let run_dir = PathBuf::from(
        env_value("SOCCER_RUN_DIR").unwrap_or_else(|| format!("out/soccer-learning-runs/{run_id}")),
    );
    let game_dir = run_dir.join("games");
    let final_artifact_path = env_value("SOCCER_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            default_artifact_path_in_run_dir(&run_dir, games, minutes, shard_index, shard_count)
        });
    let learned_params_path = env_value("SOCCER_LEARNED_PARAMS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("learned-params.json"));
    let checkpoint_artifact_path = env_value("SOCCER_CHECKPOINT_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("checkpoint-policy.json"));
    let episode_log_path = env_value("SOCCER_EPISODE_LOG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("episodes.jsonl"));
    if let Some(parent) = episode_log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut episode_log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&episode_log_path)?;
    let manifest_path = run_dir.join("manifest.json");
    let resume_artifact =
        env_value("SOCCER_RESUME_ARTIFACT").or_else(|| env_value("SOCCER_RESUME_ARTIFACT_PATH"));
    let mut pg_store = SoccerLearningPgStore::connect_from_env().map_err(invalid_data)?;
    let mut pg_experiment_slug = None::<String>;
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_last_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;
    let mut pg_persisted_games = 0usize;
    let mut policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let experiment_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id = store
            .ensure_experiment(&experiment_slug, &experiment_name, &config)
            .map_err(invalid_data)?;
        if resume_artifact.is_none() {
            if let Some(version) = store
                .load_latest_active_policy(&experiment_id, options.clone(), options.clone())
                .map_err(invalid_data)?
            {
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={}",
                    experiment_slug, version.id, version.generation
                );
                policies = version.policies;
                pg_base_policy_version_id = Some(version.id.clone());
                pg_last_policy_version_id = Some(version.id);
                pg_generation = version.generation;
            }
        }
        println!(
            "postgres_enabled experiment={} experiment_id={} base_policy_version={} generation={}",
            experiment_slug,
            experiment_id,
            pg_base_policy_version_id.as_deref().unwrap_or("none"),
            pg_generation
        );
        pg_experiment_slug = Some(experiment_slug);
        pg_experiment_id = Some(experiment_id);
    } else {
        println!("postgres_enabled=false");
    }
    let mut moment_replay_records = 0usize;
    let mut moment_replay_transitions = 0usize;
    let mut adversarial_moment_windows = Vec::new();
    if let Some(path) = moment_replay_path.as_deref() {
        let raw = fs::read_to_string(path)?;
        let mut records = soccer_moment_records_from_jsonl(&raw).map_err(invalid_data)?;
        if moment_replay_limit > 0 && records.len() > moment_replay_limit {
            records = records.split_off(records.len() - moment_replay_limit);
        }
        let replay_dataset = soccer_moment_records_to_learning_dataset(
            &records,
            config.clone(),
            moment_replay_reward_scale,
        )
        .map_err(invalid_data)?;
        moment_replay_records = records.len();
        moment_replay_transitions = replay_dataset.transitions.len();
        adversarial_moment_windows = records
            .iter()
            .map(|record| record.window.clone())
            .collect::<Vec<_>>();
        for _ in 0..moment_replay_passes {
            policies.train_adversarial(&replay_dataset.transitions);
        }
    }

    println!(
        "soccer_self_play_start run_id={} games={} parallel_games={} minutes={:.1} halves={} half_minutes={:.1} period_break_recovery_seconds={:.1} dt={:.3}s learning_interval_ticks={} ticks_per_game={} shard={}/{} base_seed={} effective_seed={} logging_transitions={} game_artifact_mode={} checkpoint_interval_games={} artifact_max_entries_per_policy={} max_policy_entries_per_team={} max_policy_target_entries_per_team={} min_policy_visits={} moment_replay_records={} moment_replay_transitions={} moment_replay_passes={} moment_replay_reward_scale={:.3}",
        run_id,
        games,
        parallel_games,
        effective_minutes,
        halves,
        half_minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        config.total_ticks(),
        shard_index,
        shard_count,
        seed,
        effective_seed,
        learning_logging_enabled,
        game_artifact_mode,
        checkpoint_interval_games,
        artifact_max_entries_per_policy,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_records,
        moment_replay_transitions,
        if moment_replay_path.is_some() { moment_replay_passes } else { 0 },
        moment_replay_reward_scale
    );
    println!("artifact={}", final_artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    println!("manifest={}", manifest_path.display());
    println!("episode_log={}", episode_log_path.display());
    println!("write_final_artifacts={write_final_artifacts}");
    println!("write_checkpoint_artifacts={write_checkpoint_artifacts}");
    if checkpoint_interval_games == 0 || !write_checkpoint_artifacts {
        println!("checkpoint_artifact=disabled");
    } else {
        println!("checkpoint_artifact={}", checkpoint_artifact_path.display());
        println!("checkpoint_interval_games={}", checkpoint_interval_games);
    }
    if let Some(path) = &resume_artifact {
        println!("resume_artifact={path}");
    }
    if let Some(path) = &moment_replay_path {
        println!(
            "moment_replay path={} records={} transitions={} passes={} reward_scale={:.3}",
            path,
            moment_replay_records,
            moment_replay_transitions,
            moment_replay_passes,
            moment_replay_reward_scale
        );
    }
    println!(
        "tactical_learning attack_spacing_delta={:.3} attack_spacing_score={:.3} attack_width_delta={:.3} attack_width_score={:.3} attack_flank_lane={:.3} defense_spacing_delta={:.3} defense_spacing_score={:.3} defense_contract_delta={:.3} defense_compactness_score={:.3}",
        tactical_learning.attack_spacing_delta_weight,
        tactical_learning.attack_spacing_score_weight,
        tactical_learning.attack_width_delta_weight,
        tactical_learning.attack_width_score_weight,
        tactical_learning.attack_flank_lane_weight,
        tactical_learning.defense_spacing_delta_weight,
        tactical_learning.defense_spacing_score_weight,
        tactical_learning.defense_contract_delta_weight,
        tactical_learning.defense_compactness_score_weight,
    );
    println!(
        "neural_learning enabled={} backend={} learning_rate={:.5} batch_size={} train_every_ticks={} max_batches_per_tick={} hidden_units={} target_scale={:.3} max_pending_batches={} replay_capacity={} replay_samples_per_tick={} target_clip={:.3}",
        neural_learning.enabled,
        neural_backend_label(neural_learning.backend),
        neural_learning.learning_rate,
        neural_learning.batch_size,
        neural_learning.train_every_ticks,
        neural_learning.max_batches_per_tick,
        neural_learning.hidden_units,
        neural_learning.target_scale,
        neural_learning.max_pending_batches,
        neural_learning.replay_capacity,
        neural_learning.replay_samples_per_tick,
        neural_learning.target_clip,
    );
    println!(
        "adversarial_embedding enabled={} memory_limit={} preloaded_windows={}",
        adversarial_embedding_exploitation_enabled,
        adversarial_embedding_memory_limit,
        adversarial_moment_windows.len(),
    );
    println!("game seed score shots on_target passes_completed/pass_attempted interceptions");

    let started = Instant::now();
    let mut episode_summaries = Vec::new();
    let mut manifest_games = Vec::new();
    let mut tactical_summary = SoccerTacticalLearningSummary::default();
    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut total_shots = 0u32;
    let mut total_on_target = 0u32;
    let mut total_pass_attempts = 0u32;
    let mut total_pass_completions = 0u32;
    let mut total_interceptions = 0u32;
    let mut next_episode = 0usize;
    let mut last_checkpoint_episode = 0usize;

    while next_episode < games {
        let batch_size = parallel_games.min(games - next_episode);
        let batch_start_episode = next_episode;
        let batch_start_policies = policies.clone();
        println!(
            "starting_batch episodes={}..{} parallel_games={}",
            batch_start_episode + 1,
            batch_start_episode + batch_size,
            batch_size
        );

        let mut handles = Vec::new();
        for offset in 0..batch_size {
            let episode = batch_start_episode + offset;
            let mut episode_config = config.clone();
            episode_config.seed = effective_seed.wrapping_add(episode as u32);
            let starting_policies = batch_start_policies.clone();
            let adversarial_moment_windows = adversarial_moment_windows.clone();
            let print_progress = true;
            handles.push(thread::spawn(move || {
                run_game(
                    episode,
                    episode_config,
                    starting_policies,
                    adversarial_moment_windows,
                    print_progress,
                )
            }));
        }

        let mut completed_games = Vec::new();
        for handle in handles {
            let game = handle
                .join()
                .map_err(|_| io_error("soccer learning worker thread panicked"))?
                .map_err(invalid_data)?;
            completed_games.push(game);
        }
        completed_games.sort_by_key(|game| game.episode_summary.episode);

        let merge_deltas = completed_games.len() > 1;
        let pg_batch_base_policy_version_id = pg_base_policy_version_id.clone();
        for game in completed_games {
            let completed_learning_game =
                soccer_learning_completed_game_from_completed(&game, &batch_start_policies);
            if merge_deltas {
                merge_team_policy_delta(&mut policies, &batch_start_policies, &game.policies);
            } else {
                policies = game.policies.clone();
            }
            if let (Some(store), Some(experiment_id)) =
                (pg_store.as_mut(), pg_experiment_id.as_deref())
            {
                let next_generation = pg_generation.saturating_add(1);
                let version_label = soccer_learning_pg_version_label(
                    &run_id,
                    shard_index,
                    game.episode_summary.episode,
                );
                let output_policy_version_id = store
                    .insert_policy_version(
                        experiment_id,
                        pg_batch_base_policy_version_id.as_deref(),
                        next_generation,
                        &version_label,
                        "merge",
                        "active",
                        &config,
                        options.clone(),
                        options.clone(),
                        &policies,
                        completed_learning_game.score.match_fitness,
                    )
                    .map_err(invalid_data)?;
                let runner_id = soccer_learning_pg_runner_id(&run_id, shard_index, shard_count);
                let run_row_id = store
                    .insert_completed_run(
                        experiment_id,
                        &runner_id,
                        pg_batch_base_policy_version_id.as_deref(),
                        Some(&output_policy_version_id),
                        &completed_learning_game,
                    )
                    .map_err(invalid_data)?;
                println!(
                    "postgres_persisted_game episode={} shard={}/{} run_id={} policy_version={} generation={}",
                    game.episode_summary.episode + 1,
                    shard_index,
                    shard_count,
                    run_row_id,
                    output_policy_version_id,
                    next_generation
                );
                pg_base_policy_version_id = Some(output_policy_version_id.clone());
                pg_last_policy_version_id = Some(output_policy_version_id);
                pg_generation = next_generation;
                pg_persisted_games += 1;
            }
            print_completed_game(&game);

            let (game_artifact_path, game_artifact_kind) = if write_game_artifacts {
                let path = game_dir.join(format!(
                    "game-{:06}-seed-{}.json",
                    game.episode_summary.episode + 1,
                    game.episode_summary.seed
                ));
                if game_artifact_mode == "full" {
                    write_json(&path, &game.artifact)?;
                    (
                        Some(path.display().to_string()),
                        Some("soccer-full-team-policy-artifact".to_string()),
                    )
                } else {
                    let compact = compact_game_artifact(&game);
                    write_json(&path, &compact)?;
                    (
                        Some(path.display().to_string()),
                        Some(compact.artifact_kind.clone()),
                    )
                }
            } else {
                (None, None)
            };

            let stats = &game.episode_summary.summary.stats;
            total_home_goals += game.episode_summary.summary.score_home;
            total_away_goals += game.episode_summary.summary.score_away;
            total_shots += stats.shots_home + stats.shots_away;
            total_on_target += stats.shots_on_target_home + stats.shots_on_target_away;
            total_pass_attempts += stats.passes_attempted_home + stats.passes_attempted_away;
            total_pass_completions += stats.passes_completed_home + stats.passes_completed_away;
            total_interceptions += stats.interceptions_home + stats.interceptions_away;
            tactical_summary.merge(&game.artifact.tactical_summary);

            let mut manifest_entry = game_manifest_entry(&game, game_artifact_path);
            manifest_entry.artifact_kind = game_artifact_kind;
            manifest_games.push(manifest_entry);
            write_episode_log(&mut episode_log, &game.episode_summary)?;
            episode_summaries.push(game.episode_summary);
        }

        let prune_summary = policies.prune(
            max_policy_entries_per_team,
            max_policy_target_entries_per_team,
            min_policy_visits,
        );
        if prune_summary.removed_entries() > 0 {
            println!(
                "policy_pruned games_completed={} removed={} home_actions={}->{} away_actions={}->{} home_targets={}->{} away_targets={}->{}",
                episode_summaries.len(),
                prune_summary.removed_entries(),
                prune_summary.home.action_entries_before,
                prune_summary.home.action_entries_after,
                prune_summary.away.action_entries_before,
                prune_summary.away.action_entries_after,
                prune_summary.home.target_entries_before,
                prune_summary.home.target_entries_after,
                prune_summary.away.target_entries_before,
                prune_summary.away.target_entries_after,
            );
            let _ = std::io::stdout().flush();
        }

        next_episode += batch_size;

        let should_checkpoint = write_checkpoint_artifacts
            && checkpoint_interval_games > 0
            && (next_episode >= games
                || next_episode.saturating_sub(last_checkpoint_episode)
                    >= checkpoint_interval_games);
        if should_checkpoint {
            let checkpoint_artifact = self_play_artifact_from_policies(
                config.clone(),
                options.clone(),
                tactical_summary.clone(),
                episode_summaries.clone(),
                &policies,
            );
            let checkpoint_export = compact_training_artifact_for_export(
                &checkpoint_artifact,
                artifact_max_entries_per_policy,
            );
            write_json(&checkpoint_artifact_path, &checkpoint_export)?;
            let checkpoint_params =
                SoccerSelfPlayLearnedParams::from_training_artifact(&checkpoint_artifact);
            write_json(&learned_params_path, &checkpoint_params)?;
            let checkpoint_manifest = run_manifest(
                &run_id,
                &run_dir,
                &game_dir,
                games,
                parallel_games,
                shard_index,
                shard_count,
                seed,
                effective_seed,
                config.clone(),
                options.clone(),
                &final_artifact_path,
                &learned_params_path,
                &checkpoint_artifact_path,
                &episode_log_path,
                checkpoint_interval_games,
                artifact_max_entries_per_policy,
                max_policy_entries_per_team,
                max_policy_target_entries_per_team,
                min_policy_visits,
                moment_replay_path.clone(),
                moment_replay_records,
                moment_replay_transitions,
                if moment_replay_path.is_some() {
                    moment_replay_passes
                } else {
                    0
                },
                moment_replay_reward_scale,
                pg_store.is_some(),
                pg_experiment_slug.clone(),
                pg_experiment_id.clone(),
                pg_last_policy_version_id.clone(),
                pg_persisted_games,
                write_game_artifacts,
                &game_artifact_mode,
                manifest_games.clone(),
            );
            write_json(&manifest_path, &checkpoint_manifest)?;
            last_checkpoint_episode = next_episode;
            println!(
                "checkpoint games_completed={} artifact={} manifest={}",
                episode_summaries.len(),
                checkpoint_artifact_path.display(),
                manifest_path.display()
            );
            let _ = std::io::stdout().flush();
        }
    }

    let artifact = self_play_artifact_from_policies(
        config.clone(),
        options.clone(),
        tactical_summary,
        episode_summaries,
        &policies,
    );
    if write_final_artifacts {
        let final_export =
            compact_training_artifact_for_export(&artifact, artifact_max_entries_per_policy);
        write_json(&final_artifact_path, &final_export)?;
        let learned_params = SoccerSelfPlayLearnedParams::from_training_artifact(&artifact);
        write_json(&learned_params_path, &learned_params)?;

        let manifest = run_manifest(
            &run_id,
            &run_dir,
            &game_dir,
            games,
            parallel_games,
            shard_index,
            shard_count,
            seed,
            effective_seed,
            config.clone(),
            options,
            &final_artifact_path,
            &learned_params_path,
            &checkpoint_artifact_path,
            &episode_log_path,
            checkpoint_interval_games,
            artifact_max_entries_per_policy,
            max_policy_entries_per_team,
            max_policy_target_entries_per_team,
            min_policy_visits,
            moment_replay_path.clone(),
            moment_replay_records,
            moment_replay_transitions,
            if moment_replay_path.is_some() {
                moment_replay_passes
            } else {
                0
            },
            moment_replay_reward_scale,
            pg_store.is_some(),
            pg_experiment_slug.clone(),
            pg_experiment_id.clone(),
            pg_last_policy_version_id.clone(),
            pg_persisted_games,
            write_game_artifacts,
            &game_artifact_mode,
            manifest_games,
        );
        write_json(&manifest_path, &manifest)?;
    } else {
        println!("final_artifacts=disabled");
    }

    let elapsed = started.elapsed();
    let game_count = games.max(1) as f64;
    println!(
        "soccer_self_play_done games={} minutes={:.1} halves={} half_minutes={:.1} dt={:.3}s ticks_per_game={} elapsed={:.2?}",
        games,
        effective_minutes,
        halves,
        half_minutes,
        dt_seconds,
        config.total_ticks(),
        elapsed
    );
    println!("artifact={}", final_artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    println!("manifest={}", manifest_path.display());
    println!("episode_log={}", episode_log_path.display());
    println!(
        "aggregate goals_per_game={:.2} home_goals={} away_goals={} shots_per_game={:.2} on_target_rate={:.2} pass_completion={:.2} interceptions_per_game={:.2}",
        (total_home_goals + total_away_goals) as f64 / game_count,
        total_home_goals,
        total_away_goals,
        total_shots as f64 / game_count,
        if total_shots == 0 { 0.0 } else { total_on_target as f64 / total_shots as f64 },
        if total_pass_attempts == 0 { 0.0 } else { total_pass_completions as f64 / total_pass_attempts as f64 },
        total_interceptions as f64 / game_count,
    );
    println!(
        "tactical_summary shape_transitions={} attack={} defense={} attack_width_score={:.3} attack_flank_lane={:.3} defense_contract_score={:.3} defense_contract_delta_yards={:.3} mean_tactical_reward={:.3}",
        artifact.tactical_summary.shape_transitions,
        artifact.tactical_summary.attack_transitions,
        artifact.tactical_summary.defense_transitions,
        artifact.tactical_summary.mean_attack_width_score,
        artifact.tactical_summary.mean_attack_flank_lane_score,
        artifact.tactical_summary.mean_defense_contract_score,
        artifact.tactical_summary.mean_defense_contract_delta_yards,
        artifact.tactical_summary.mean_tactical_reward,
    );

    println!("home learned actions action visits mean_q");
    for (action, visits, mean_q) in action_summary(&artifact.home_entries).into_iter().take(8) {
        println!("{action} {visits} {mean_q:.3}");
    }
    println!("away learned actions action visits mean_q");
    for (action, visits, mean_q) in action_summary(&artifact.away_entries).into_iter().take(8) {
        println!("{action} {visits} {mean_q:.3}");
    }
    println!(
        "policy_entries home={} away={} target_entries home={} away={}",
        artifact.home_entries.len(),
        artifact.away_entries.len(),
        artifact.home_target_entries.len(),
        artifact.away_target_entries.len()
    );

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("soccer learning run failed: {error}");
        std::process::exit(1);
    }
}
