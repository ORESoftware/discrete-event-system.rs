//! Run accelerated soccer self-play and summarize learned MDP/POMDP policies.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

use des_engine::des::general::soccer::{
    soccer_self_play_default_learned_params_path,
    train_soccer_team_policies_from_self_play_with_initial_policies_progress_and_checkpoints,
    MatchConfig, SoccerQEntry, SoccerQPolicyOptions, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningWeights, SoccerTeamQPolicies,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_f64_alias(primary: &str, alias: &str, default: f64) -> f64 {
    std::env::var(primary)
        .or_else(|_| std::env::var(alias))
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn create_parent_dir(path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
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

fn write_training_artifact(
    path: &str,
    artifact: &SoccerSelfPlayTrainingArtifact,
    max_entries_per_policy: usize,
) -> bool {
    create_parent_dir(path);
    let export = compact_training_artifact_for_export(artifact, max_entries_per_policy);
    let json = match serde_json::to_string_pretty(&export) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("failed to serialize self-play artifact {path}: {e}");
            return false;
        }
    };
    match std::fs::write(path, json) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("failed to write self-play artifact {path}: {e}");
            false
        }
    }
}

fn write_learned_params(path: &str, artifact: &SoccerSelfPlayTrainingArtifact) -> bool {
    create_parent_dir(path);
    let params = SoccerSelfPlayLearnedParams::from_training_artifact(artifact);
    let json = match serde_json::to_string_pretty(&params) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("failed to serialize learned params {path}: {e}");
            return false;
        }
    };
    match std::fs::write(path, json) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("failed to write learned params {path}: {e}");
            false
        }
    }
}

fn env_tactical_learning_weights() -> SoccerTacticalLearningWeights {
    let default = SoccerTacticalLearningWeights::default();
    SoccerTacticalLearningWeights {
        attack_spacing_delta_weight: env_f64(
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            default.attack_spacing_delta_weight,
        ),
        attack_spacing_score_weight: env_f64(
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            default.attack_spacing_score_weight,
        ),
        attack_width_delta_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            default.attack_width_delta_weight,
        ),
        attack_width_score_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            default.attack_width_score_weight,
        ),
        attack_flank_lane_weight: env_f64(
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            default.attack_flank_lane_weight,
        ),
        defense_spacing_delta_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            default.defense_spacing_delta_weight,
        ),
        defense_spacing_score_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            default.defense_spacing_score_weight,
        ),
        defense_contract_delta_weight: env_f64(
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            default.defense_contract_delta_weight,
        ),
        defense_compactness_score_weight: env_f64(
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            default.defense_compactness_score_weight,
        ),
    }
}

fn artifact_minutes_label(minutes: f64) -> String {
    if (minutes - minutes.round()).abs() < 1e-9 {
        format!("{:.0}", minutes)
    } else {
        format!("{:.2}", minutes).replace('.', "p")
    }
}

fn default_artifact_path(
    games: usize,
    minutes: f64,
    shard_index: usize,
    shard_count: usize,
) -> String {
    if shard_count > 1 {
        format!(
            "out/soccer-mdp-pomdp-self-play-shard-{}-of-{}-{}x{}.json",
            shard_index,
            shard_count,
            games,
            artifact_minutes_label(minutes)
        )
    } else {
        format!(
            "out/soccer-mdp-pomdp-self-play-{}x{}.json",
            games,
            artifact_minutes_label(minutes)
        )
    }
}

fn load_initial_policies(
    resume_path: Option<&str>,
    options: &SoccerQPolicyOptions,
) -> SoccerTeamQPolicies {
    let Some(path) = resume_path else {
        return SoccerTeamQPolicies::new(options.clone());
    };
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("failed to read SOCCER_RESUME_ARTIFACT_PATH={path}: {e}");
            std::process::exit(2);
        }
    };
    if let Ok(artifact) = serde_json::from_str::<SoccerSelfPlayTrainingArtifact>(&raw) {
        return match SoccerTeamQPolicies::from_self_play_artifact(&artifact) {
            Ok(policies) => policies,
            Err(e) => {
                eprintln!("failed to restore policies from {path}: {e}");
                std::process::exit(2);
            }
        };
    }
    let params = match serde_json::from_str::<SoccerSelfPlayLearnedParams>(&raw) {
        Ok(params) => params,
        Err(e) => {
            eprintln!("failed to parse self-play artifact or learned params {path}: {e}");
            std::process::exit(2);
        }
    };
    match SoccerTeamQPolicies::from_learned_params(&params) {
        Ok(policies) => policies,
        Err(e) => {
            eprintln!("failed to restore policies from {path}: {e}");
            std::process::exit(2);
        }
    }
}

fn write_episode_log(log: &mut Option<std::fs::File>, episode: &SoccerSelfPlayEpisodeSummary) {
    let Some(file) = log.as_mut() else {
        return;
    };
    if serde_json::to_writer(&mut *file, episode).is_ok() {
        let _ = writeln!(file);
        let _ = file.flush();
    }
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

fn main() {
    let games = env_usize("SOCCER_GAMES", 100);
    let halves = env_usize("SOCCER_HALVES", 2).max(1);
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0);
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64);
    let period_break_recovery_seconds = env_f64_alias(
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
        "SOCCER_HALFTIME_RECOVERY_SECONDS",
        900.0,
    );
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2);
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4).max(1);
    let shard_index = env_usize("SOCCER_SHARD_INDEX", 0);
    let shard_count = env_usize("SOCCER_SHARD_COUNT", 1).max(1);
    if shard_index >= shard_count {
        eprintln!("SOCCER_SHARD_INDEX must be less than SOCCER_SHARD_COUNT");
        std::process::exit(2);
    }
    let seed = env_u32("SOCCER_SEED", 2026);
    let shard_seed_stride = env_u32("SOCCER_SHARD_SEED_STRIDE", 1_000_000);
    let effective_seed = seed.wrapping_add((shard_index as u32).wrapping_mul(shard_seed_stride));
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20),
        gamma: env_f64("SOCCER_GAMMA", 0.96),
    };
    let tactical_learning = env_tactical_learning_weights();
    let config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        period_count: halves,
        period_break_recovery_seconds,
        learning_enabled: true,
        learning_logging_enabled: false,
        learning_interval_ticks,
        tactical_learning: tactical_learning.clone(),
        max_human_players: 0,
        seed: effective_seed,
        ..MatchConfig::default()
    };
    let artifact_path = std::env::var("SOCCER_ARTIFACT_PATH")
        .unwrap_or_else(|_| default_artifact_path(games, minutes, shard_index, shard_count));
    let learned_params_path = std::env::var("SOCCER_LEARNED_PARAMS_PATH")
        .unwrap_or_else(|_| soccer_self_play_default_learned_params_path(&artifact_path));
    let checkpoint_interval_games = env_usize("SOCCER_CHECKPOINT_INTERVAL_GAMES", 10);
    let artifact_max_entries_per_policy =
        env_usize("SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY", 10_000);
    let checkpoint_artifact_path = std::env::var("SOCCER_CHECKPOINT_ARTIFACT_PATH")
        .ok()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| format!("{artifact_path}.checkpoint.json"));
    let episode_log_path = std::env::var("SOCCER_EPISODE_LOG_PATH")
        .unwrap_or_else(|_| format!("{artifact_path}.episodes.jsonl"));
    create_parent_dir(&episode_log_path);
    let mut episode_log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&episode_log_path)
        .ok();
    let resume_path = std::env::var("SOCCER_RESUME_ARTIFACT_PATH").ok();
    let initial_policies = load_initial_policies(resume_path.as_deref(), &options);

    let started = Instant::now();
    let artifact =
        train_soccer_team_policies_from_self_play_with_initial_policies_progress_and_checkpoints(
            config.clone(),
            games,
            options,
            initial_policies,
            |episode| {
                let stats = &episode.summary.stats;
                println!(
                "completed_game={} seed={} score={}-{} shots={} on_target={} pass_completion={}/{} interceptions={}",
                episode.episode + 1,
                episode.seed,
                episode.summary.score_home,
                episode.summary.score_away,
                stats.shots_home + stats.shots_away,
                stats.shots_on_target_home + stats.shots_on_target_away,
                stats.passes_completed_home + stats.passes_completed_away,
                stats.passes_attempted_home + stats.passes_attempted_away,
                stats.interceptions_home + stats.interceptions_away,
            );
                write_episode_log(&mut episode_log, episode);
                let _ = std::io::stdout().flush();
            },
            |episode, seed, completed_ticks, total_ticks| {
                println!(
                    "progress_game={} seed={} ticks={}/{}",
                    episode + 1,
                    seed,
                    completed_ticks,
                    total_ticks
                );
                let _ = std::io::stdout().flush();
            },
            |checkpoint| {
                let completed_games = checkpoint.episodes.len();
                if checkpoint_interval_games > 0
                    && completed_games % checkpoint_interval_games == 0
                    && write_training_artifact(
                        &checkpoint_artifact_path,
                        checkpoint,
                        artifact_max_entries_per_policy,
                    )
                {
                    println!(
                        "checkpoint_game={} artifact={}",
                        completed_games, checkpoint_artifact_path
                    );
                    let _ = std::io::stdout().flush();
                }
            },
        );
    let elapsed = started.elapsed();

    let _ = write_training_artifact(&artifact_path, &artifact, artifact_max_entries_per_policy);
    let _ = write_learned_params(&learned_params_path, &artifact);

    println!(
        "soccer_self_play games={} halves={} half_minutes={:.1} minutes={:.1} dt={:.3}s learning_interval_ticks={} ticks_per_game={} shard={}/{} base_seed={} effective_seed={} elapsed={:.2?}",
        games,
        halves,
        minutes / halves as f64,
        minutes,
        dt_seconds,
        learning_interval_ticks,
        config.total_ticks(),
        shard_index,
        shard_count,
        seed,
        effective_seed,
        elapsed
    );
    println!("artifact={}", artifact_path);
    println!("learned_params={}", learned_params_path);
    if checkpoint_interval_games == 0 {
        println!("checkpoint_artifact=disabled");
    } else {
        println!("checkpoint_artifact={}", checkpoint_artifact_path);
        println!("checkpoint_interval_games={}", checkpoint_interval_games);
    }
    if artifact_max_entries_per_policy == 0 {
        println!("artifact_max_entries_per_policy=unbounded");
    } else {
        println!(
            "artifact_max_entries_per_policy={}",
            artifact_max_entries_per_policy
        );
    }
    println!("episode_log={}", episode_log_path);
    println!(
        "resume_artifact={}",
        resume_path.as_deref().unwrap_or("fresh")
    );
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
    println!("game seed score shots on_target passes_completed/pass_attempted interceptions");

    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut total_shots = 0u32;
    let mut total_on_target = 0u32;
    let mut total_pass_attempts = 0u32;
    let mut total_pass_completions = 0u32;
    let mut total_interceptions = 0u32;
    for episode in &artifact.episodes {
        let stats = &episode.summary.stats;
        let shots = stats.shots_home + stats.shots_away;
        let on_target = stats.shots_on_target_home + stats.shots_on_target_away;
        let pass_attempts = stats.passes_attempted_home + stats.passes_attempted_away;
        let pass_completions = stats.passes_completed_home + stats.passes_completed_away;
        let interceptions = stats.interceptions_home + stats.interceptions_away;
        total_home_goals += episode.summary.score_home;
        total_away_goals += episode.summary.score_away;
        total_shots += shots;
        total_on_target += on_target;
        total_pass_attempts += pass_attempts;
        total_pass_completions += pass_completions;
        total_interceptions += interceptions;
        println!(
            "{} {} {}-{} {} {} {}/{} {}",
            episode.episode + 1,
            episode.seed,
            episode.summary.score_home,
            episode.summary.score_away,
            shots,
            on_target,
            pass_completions,
            pass_attempts,
            interceptions
        );
    }

    let game_count = games.max(1) as f64;
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
}
