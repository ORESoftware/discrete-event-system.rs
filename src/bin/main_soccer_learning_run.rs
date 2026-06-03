//! Run accelerated soccer self-play and summarize learned MDP/POMDP policies.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use des_engine::des::general::soccer::{
    train_soccer_team_policies_from_self_play_with_progress, MatchConfig, SoccerQEntry,
    SoccerQPolicyOptions,
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
    let games = env_usize("SOCCER_GAMES", 10);
    let minutes = env_f64("SOCCER_MINUTES", 90.0);
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2);
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4).max(1);
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
        duration_seconds: minutes * 60.0,
        learning_enabled: true,
        learning_logging_enabled: false,
        learning_interval_ticks,
        max_human_players: 0,
        seed,
        ..MatchConfig::default()
    };

    let started = Instant::now();
    let artifact = train_soccer_team_policies_from_self_play_with_progress(
        config.clone(),
        games,
        options,
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
    );
    let elapsed = started.elapsed();

    let artifact_path = std::env::var("SOCCER_ARTIFACT_PATH")
        .unwrap_or_else(|_| "out/soccer-mdp-pomdp-self-play-10x90.json".to_string());
    if let Some(parent) = std::path::Path::new(&artifact_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&artifact) {
        let _ = std::fs::write(&artifact_path, json);
    }

    println!(
        "soccer_self_play games={} minutes={:.1} dt={:.3}s learning_interval_ticks={} ticks_per_game={} elapsed={:.2?}",
        games,
        minutes,
        dt_seconds,
        learning_interval_ticks,
        config.total_ticks(),
        elapsed
    );
    println!("artifact={}", artifact_path);
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
