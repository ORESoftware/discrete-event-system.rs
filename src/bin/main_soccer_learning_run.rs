//! Run accelerated soccer self-play and persist learned MDP/POMDP policies.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    soccer_moment_records_from_jsonl, soccer_moment_records_to_learning_dataset, MatchConfig,
    SoccerMatch, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerSelfPlayEpisodeSummary,
    SoccerSelfPlayTrainingArtifact, SoccerTeamPolicyArtifact, SoccerTeamQPolicies,
};
use serde::Serialize;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u8(name: &str, default: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        })
        .unwrap_or(default)
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
    games: usize,
    parallel_games: usize,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: String,
    checkpoint_artifact_path: String,
    checkpoint_interval_games: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
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
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = fs::File::create(&tmp_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, value)?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
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
    if let Ok(artifact) = serde_json::from_str::<SoccerTeamPolicyArtifact>(&raw) {
        return SoccerTeamQPolicies::from_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    Err(invalid_data(format!(
        "resume artifact {path} is neither a self-play nor team-policy artifact"
    ))
    .into())
}

fn run_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: SoccerTeamQPolicies,
    print_progress: bool,
) -> Result<CompletedGame, String> {
    let started = Instant::now();
    let total_ticks = config.total_ticks();
    let episode_seed = config.seed as u64;
    let progress_interval = (total_ticks / 9).max(1);
    let mut sim = SoccerMatch::default_11v11(config).with_team_policies(starting_policies);

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
    episode_summaries: Vec<SoccerSelfPlayEpisodeSummary>,
    policies: &SoccerTeamQPolicies,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
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
    games: usize,
    parallel_games: usize,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: &Path,
    checkpoint_artifact_path: &Path,
    checkpoint_interval_games: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
    write_game_artifacts: bool,
    game_artifact_mode: &str,
    game_artifacts: Vec<GameManifestEntry>,
) -> RunManifest {
    RunManifest {
        run_id: run_id.to_string(),
        run_dir: run_dir.display().to_string(),
        games,
        parallel_games,
        config,
        options,
        final_artifact_path: final_artifact_path.display().to_string(),
        checkpoint_artifact_path: checkpoint_artifact_path.display().to_string(),
        checkpoint_interval_games,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_path,
        moment_replay_records,
        moment_replay_transitions,
        moment_replay_passes,
        moment_replay_reward_scale,
        write_game_artifacts,
        game_artifact_mode: game_artifact_mode.to_string(),
        game_artifacts,
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let games = env_usize("SOCCER_GAMES", 100);
    let minutes = env_f64("SOCCER_MINUTES", 90.0);
    let halves = env_u8("SOCCER_HALVES", 2).max(1);
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", minutes / f64::from(halves));
    let effective_minutes = half_minutes * f64::from(halves);
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2);
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4).max(1);
    let parallel_games = env_usize("SOCCER_PARALLEL_GAMES", 1).max(1);
    let checkpoint_interval_games = env_usize("SOCCER_CHECKPOINT_INTERVAL_GAMES", parallel_games);
    let max_policy_entries_per_team = env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0);
    let max_policy_target_entries_per_team = env_usize(
        "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
        max_policy_entries_per_team,
    );
    let min_policy_visits = env_u32("SOCCER_MIN_POLICY_VISITS", 0);
    let moment_replay_path = std::env::var("SOCCER_MOMENT_REPLAY_PATH")
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty());
    let moment_replay_limit = env_usize("SOCCER_MOMENT_REPLAY_LIMIT", 0);
    let moment_replay_passes = env_usize("SOCCER_MOMENT_REPLAY_PASSES", 1).max(1);
    let moment_replay_reward_scale = env_f64("SOCCER_MOMENT_REPLAY_REWARD_SCALE", 1.0);
    let write_game_artifacts = env_bool("SOCCER_WRITE_GAME_ARTIFACTS", true);
    let game_artifact_mode = std::env::var("SOCCER_GAME_ARTIFACT_MODE")
        .unwrap_or_else(|_| "summary".to_string())
        .trim()
        .to_ascii_lowercase();
    let learning_logging_enabled = env_bool("SOCCER_LEARNING_LOGGING", false);
    let halftime_fatigue_recovery = env_f64("SOCCER_HALFTIME_FATIGUE_RECOVERY", 0.18);
    let seed = std::env::var("SOCCER_SEED")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2026);
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20),
        gamma: env_f64("SOCCER_GAMMA", 0.96),
    };
    let config = MatchConfig {
        dt_seconds,
        duration_seconds: effective_minutes * 60.0,
        halves,
        half_duration_seconds: half_minutes * 60.0,
        halftime_fatigue_recovery,
        learning_enabled: true,
        learning_logging_enabled,
        learning_interval_ticks,
        max_human_players: 0,
        seed,
        ..MatchConfig::default()
    };

    let run_id = std::env::var("SOCCER_RUN_ID").unwrap_or_else(|_| default_run_id());
    let run_dir = PathBuf::from(
        std::env::var("SOCCER_RUN_DIR")
            .unwrap_or_else(|_| format!("out/soccer-learning-runs/{run_id}")),
    );
    let game_dir = run_dir.join("games");
    let final_artifact_path = PathBuf::from(
        std::env::var("SOCCER_ARTIFACT_PATH")
            .unwrap_or_else(|_| run_dir.join("final-policy.json").display().to_string()),
    );
    let checkpoint_artifact_path = PathBuf::from(
        std::env::var("SOCCER_CHECKPOINT_ARTIFACT_PATH")
            .unwrap_or_else(|_| run_dir.join("checkpoint-policy.json").display().to_string()),
    );
    let manifest_path = run_dir.join("manifest.json");
    let resume_artifact = std::env::var("SOCCER_RESUME_ARTIFACT").ok();
    let mut policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    let mut moment_replay_records = 0usize;
    let mut moment_replay_transitions = 0usize;
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
        for _ in 0..moment_replay_passes {
            policies.train_adversarial(&replay_dataset.transitions);
        }
    }

    println!(
        "soccer_self_play_start run_id={} games={} parallel_games={} minutes={:.1} halves={} half_minutes={:.1} dt={:.3}s learning_interval_ticks={} ticks_per_game={} logging_transitions={} game_artifact_mode={} checkpoint_interval_games={} max_policy_entries_per_team={} max_policy_target_entries_per_team={} min_policy_visits={} moment_replay_records={} moment_replay_transitions={} moment_replay_passes={} moment_replay_reward_scale={:.3}",
        run_id,
        games,
        parallel_games,
        effective_minutes,
        halves,
        half_minutes,
        dt_seconds,
        learning_interval_ticks,
        config.total_ticks(),
        learning_logging_enabled,
        game_artifact_mode,
        checkpoint_interval_games,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_records,
        moment_replay_transitions,
        if moment_replay_path.is_some() { moment_replay_passes } else { 0 },
        moment_replay_reward_scale
    );
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

    let started = Instant::now();
    let mut episode_summaries = Vec::new();
    let mut manifest_games = Vec::new();
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
            episode_config.seed = seed.wrapping_add(episode as u32);
            let starting_policies = batch_start_policies.clone();
            let print_progress = true;
            handles.push(thread::spawn(move || {
                run_game(episode, episode_config, starting_policies, print_progress)
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
        for game in completed_games {
            if merge_deltas {
                merge_team_policy_delta(&mut policies, &batch_start_policies, &game.policies);
            } else {
                policies = game.policies.clone();
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

            let mut manifest_entry = game_manifest_entry(&game, game_artifact_path);
            manifest_entry.artifact_kind = game_artifact_kind;
            manifest_games.push(manifest_entry);
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

        let should_checkpoint = checkpoint_interval_games > 0
            && (next_episode >= games
                || next_episode.saturating_sub(last_checkpoint_episode)
                    >= checkpoint_interval_games);
        if should_checkpoint {
            let checkpoint_artifact = self_play_artifact_from_policies(
                config.clone(),
                options.clone(),
                episode_summaries.clone(),
                &policies,
            );
            write_json(&checkpoint_artifact_path, &checkpoint_artifact)?;
            let checkpoint_manifest = run_manifest(
                &run_id,
                &run_dir,
                games,
                parallel_games,
                config.clone(),
                options.clone(),
                &final_artifact_path,
                &checkpoint_artifact_path,
                checkpoint_interval_games,
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
        episode_summaries,
        &policies,
    );
    write_json(&final_artifact_path, &artifact)?;

    let manifest = run_manifest(
        &run_id,
        &run_dir,
        games,
        parallel_games,
        config.clone(),
        options,
        &final_artifact_path,
        &checkpoint_artifact_path,
        checkpoint_interval_games,
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
        write_game_artifacts,
        &game_artifact_mode,
        manifest_games,
    );
    write_json(&manifest_path, &manifest)?;

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
    println!("manifest={}", manifest_path.display());
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
