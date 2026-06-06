//! Postgres persistence for soccer self-play learning.
//!
//! The canonical table contract lives in `remote/libs/pg-defs/schema/schema.sql`.
//! This module is a small Rust adapter over that contract for queue runners.

use native_tls::TlsConnector;
use postgres::types::ToSql;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};
use std::fmt::Write as _;
use uuid::Uuid;

use crate::des::general::soccer::{
    MatchConfig, SoccerNeuralNetworkSnapshot, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions,
    SoccerQStateKey, SoccerQTargetEntry, SoccerSetPlayTrainingArtifact,
    SoccerTacticalLearningWeights, SoccerTeamQPolicies, Team,
};
use crate::des::soccer_learning::{
    soccer_learning_from_micros, soccer_learning_to_micros, soccer_team_label,
    SoccerLearningCompletedGame, SoccerLearningPolicyDeltaEntry, SoccerLearningPolicyEntryKind,
};

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyVersion {
    pub id: String,
    pub generation: i32,
    pub updated_at_micros: i64,
    pub policies: SoccerTeamQPolicies,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub tactical_learning: Option<SoccerTacticalLearningWeights>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyMetadata {
    pub id: String,
    pub generation: i32,
    pub updated_at_micros: i64,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub tactical_learning: Option<SoccerTacticalLearningWeights>,
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerLearningPgCompletedRunInsert<'a> {
    pub base_policy_version_id: Option<&'a str>,
    pub output_policy_version_id: Option<&'a str>,
    pub game: &'a SoccerLearningCompletedGame,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SoccerLearningPgCompletedRunRetentionPrune {
    pub deleted_runs: u64,
    pub deleted_delta_rows: u64,
}

const POSTGRES_MAX_QUERY_PARAMETERS: usize = 65_535;
const SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT: usize = 22;
const SOCCER_RUN_DELTA_PARAMETER_COUNT: usize = 16;
const SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT: usize = 9;
const SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT: usize = 13;

const SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE: usize = 1024;
const SOCCER_RUN_DELTA_INSERT_BATCH_SIZE: usize = 1024;
const SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE: usize = 512;
const SOCCER_POLICY_RETENTION_BRANCH_TIP: &str = "branch_tip";

const _: () = {
    assert!(
        SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_RUN_DELTA_INSERT_BATCH_SIZE * SOCCER_RUN_DELTA_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
};

fn postgres_insert_sql_buffer(prefix: &str, rows: usize, parameters_per_row: usize) -> String {
    let estimated_tuple_bytes = parameters_per_row.saturating_mul(8).saturating_add(24);
    let mut sql = String::with_capacity(
        prefix
            .len()
            .saturating_add(rows.saturating_mul(estimated_tuple_bytes)),
    );
    sql.push_str(prefix);
    sql
}

fn soccer_policy_version_metrics(
    fitness: f64,
    neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    tactical_learning: Option<&SoccerTacticalLearningWeights>,
) -> Result<Value, String> {
    let mut metrics = json!({
        "fitness": fitness,
        "learningComponents": {
            "mdpPolicyEntries": true,
            "pomdpStateKeyFeatures": true,
            "mpcFormationLpWeights": tactical_learning.is_some(),
            "lpAlignmentWeight": tactical_learning
                .map(|weights| weights.formation_lp_alignment_weight)
                .unwrap_or(0.0),
            "neuralNetworkSnapshot": neural_network.is_some()
        }
    });
    if let Some(neural_network) = neural_network {
        validate_soccer_neural_network_snapshot_for_pg(neural_network)?;
        metrics["neuralNetwork"] = serde_json::to_value(neural_network)
            .map_err(|err| format!("serialize soccer neural network snapshot: {err}"))?;
    }
    if let Some(tactical_learning) = tactical_learning {
        validate_soccer_tactical_learning_weights_for_pg(tactical_learning)?;
        metrics["tacticalLearning"] = serde_json::to_value(tactical_learning)
            .map_err(|err| format!("serialize soccer tactical learning weights: {err}"))?;
    }
    Ok(metrics)
}

fn soccer_policy_version_metrics_with_retention(
    mut metrics: Value,
    policy_version_id: &str,
    branch_key: &str,
    retention_kind: &str,
    full_entries_retained: bool,
) -> Value {
    metrics["postgresRetention"] = json!({
        "policyVersionId": policy_version_id,
        "branchKey": branch_key,
        "retentionKind": retention_kind,
        "fullEntriesRetained": full_entries_retained,
    });
    metrics
}

fn soccer_policy_version_neural_network_from_metrics(
    metrics: &Value,
) -> Result<Option<SoccerNeuralNetworkSnapshot>, String> {
    let Some(neural_network) = metrics.get("neuralNetwork") else {
        return Ok(None);
    };
    let snapshot = serde_json::from_value::<SoccerNeuralNetworkSnapshot>(neural_network.clone())
        .map_err(|err| format!("decode soccer neural network snapshot: {err}"))?;
    validate_soccer_neural_network_snapshot_for_pg(&snapshot)?;
    Ok(Some(snapshot))
}

fn soccer_policy_version_tactical_learning_from_values(
    config: &Value,
    metrics: &Value,
) -> Result<Option<SoccerTacticalLearningWeights>, String> {
    let tactical_learning = metrics
        .get("tacticalLearning")
        .or_else(|| config.get("tacticalLearning"));
    let Some(tactical_learning) = tactical_learning else {
        return Ok(None);
    };
    let weights =
        serde_json::from_value::<SoccerTacticalLearningWeights>(tactical_learning.clone())
            .map_err(|err| format!("decode soccer tactical learning weights: {err}"))?;
    validate_soccer_tactical_learning_weights_for_pg(&weights)?;
    Ok(Some(weights))
}

fn soccer_policy_version_retains_full_entries(status: &str) -> bool {
    !matches!(status, "archived" | "rejected")
}

fn validate_soccer_tactical_learning_weights_for_pg(
    weights: &SoccerTacticalLearningWeights,
) -> Result<(), String> {
    for (name, value) in [
        (
            "attackSpacingDeltaWeight",
            weights.attack_spacing_delta_weight,
        ),
        (
            "attackSpacingScoreWeight",
            weights.attack_spacing_score_weight,
        ),
        ("attackWidthDeltaWeight", weights.attack_width_delta_weight),
        ("attackWidthScoreWeight", weights.attack_width_score_weight),
        ("attackFlankLaneWeight", weights.attack_flank_lane_weight),
        (
            "defenseSpacingDeltaWeight",
            weights.defense_spacing_delta_weight,
        ),
        (
            "defenseSpacingScoreWeight",
            weights.defense_spacing_score_weight,
        ),
        (
            "defenseContractDeltaWeight",
            weights.defense_contract_delta_weight,
        ),
        (
            "defenseCompactnessScoreWeight",
            weights.defense_compactness_score_weight,
        ),
        (
            "defenseBallDepthScoreWeight",
            weights.defense_ball_depth_score_weight,
        ),
        (
            "defenseEndlineSoftPenaltyWeight",
            weights.defense_endline_soft_penalty_weight,
        ),
        (
            "defenseEndlineHardPenaltyWeight",
            weights.defense_endline_hard_penalty_weight,
        ),
        (
            "defenderMidfielderPressWeight",
            weights.defender_midfielder_press_weight,
        ),
        ("midfielderPressWeight", weights.midfielder_press_weight),
        (
            "formationLpAlignmentWeight",
            weights.formation_lp_alignment_weight,
        ),
    ] {
        if !value.is_finite() {
            return Err(format!(
                "{name} must be finite in soccer tactical learning weights"
            ));
        }
    }
    Ok(())
}

fn validate_soccer_neural_network_snapshot_for_pg(
    snapshot: &SoccerNeuralNetworkSnapshot,
) -> Result<(), String> {
    if snapshot.input_dim == 0 {
        return Err("soccer neural network snapshot input_dim must be positive".to_string());
    }
    if snapshot.output_dim == 0 {
        return Err("soccer neural network snapshot output_dim must be positive".to_string());
    }
    if snapshot.layers.is_empty() {
        return Err("soccer neural network snapshot must contain layers".to_string());
    }
    if !snapshot.l2_norm.is_finite() {
        return Err("soccer neural network snapshot l2_norm must be finite".to_string());
    }
    let mut expected_input_dim = snapshot.input_dim;
    let mut parameter_count = 0usize;
    for (layer_idx, layer) in snapshot.layers.iter().enumerate() {
        let activation = layer.activation.to_ascii_lowercase();
        if !matches!(activation.as_str(), "linear" | "sigmoid" | "tanh" | "relu") {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has unsupported activation {:?}",
                layer.activation
            ));
        }
        let output_dim = layer.biases.len();
        if output_dim == 0 {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has no biases"
            ));
        }
        if layer.weights.len() != output_dim {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has {} weight rows for {} biases",
                layer.weights.len(),
                output_dim
            ));
        }
        for (row_idx, row) in layer.weights.iter().enumerate() {
            if row.len() != expected_input_dim {
                return Err(format!(
                    "soccer neural network snapshot layer {layer_idx} row {row_idx} has input dim {}, expected {}",
                    row.len(),
                    expected_input_dim
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "soccer neural network snapshot layer {layer_idx} row {row_idx} has non-finite weights"
                ));
            }
            parameter_count = parameter_count.saturating_add(row.len());
        }
        if layer.biases.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has non-finite biases"
            ));
        }
        parameter_count = parameter_count.saturating_add(layer.biases.len());
        expected_input_dim = output_dim;
    }
    if expected_input_dim != snapshot.output_dim {
        return Err(format!(
            "soccer neural network snapshot final output dim {expected_input_dim} does not match declared {}",
            snapshot.output_dim
        ));
    }
    if snapshot.parameter_count != 0 && snapshot.parameter_count != parameter_count {
        return Err(format!(
            "soccer neural network snapshot parameter_count {} does not match decoded {}",
            snapshot.parameter_count, parameter_count
        ));
    }
    Ok(())
}

pub struct SoccerLearningPgStore {
    client: Client,
}

impl SoccerLearningPgStore {
    pub fn connect(database_url: &str) -> Result<Self, String> {
        let mut tls_builder = TlsConnector::builder();
        if !soccer_learning_pg_should_verify_certificates(database_url) {
            // Match libpq sslmode=require/prefer semantics: encrypt the wire,
            // but do not require an RDS CA bundle in minimal runner images.
            tls_builder.danger_accept_invalid_certs(true);
            tls_builder.danger_accept_invalid_hostnames(true);
        }
        let tls = tls_builder
            .build()
            .map_err(|err| format!("build soccer learning postgres tls connector: {err}"))?;
        let client = Client::connect(database_url, MakeTlsConnector::new(tls))
            .map_err(|err| format!("connect soccer learning postgres: {err}"))?;
        Ok(Self { client })
    }

    pub fn connect_from_env() -> Result<Option<Self>, String> {
        let Some(database_url) = soccer_learning_database_url() else {
            return Ok(None);
        };
        Self::connect(&database_url).map(Some)
    }

    pub fn ensure_experiment(
        &mut self,
        slug: &str,
        display_name: &str,
        config: &MatchConfig,
    ) -> Result<String, String> {
        if let Some(row) = self
            .client
            .query_opt(
                r#"
                select id::text
                from des_soccer_learning_experiments
                where slug = $1 and is_soft_deleted = false
                limit 1
                "#,
                &[&slug],
            )
            .map_err(|err| format!("select soccer learning experiment: {err}"))?
        {
            return Ok(row.get(0));
        }

        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let row = self
            .client
            .query_one(
                r#"
                insert into des_soccer_learning_experiments
                  (slug, display_name, config, meta_data)
                values
                  ($1, $2, $3, '{}'::jsonb)
                returning id::text
                "#,
                &[&slug, &display_name, &config_json],
            )
            .map_err(|err| format!("insert soccer learning experiment: {err}"))?;
        Ok(row.get(0))
    }

    pub fn load_latest_active_policy(
        &mut self,
        experiment_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        let Some(metadata) = self.load_latest_active_policy_metadata(experiment_id)? else {
            return Ok(None);
        };
        let policies = self.load_policy_entries(&metadata.id, home_options, away_options)?;
        Ok(Some(SoccerLearningPgPolicyVersion {
            id: metadata.id,
            generation: metadata.generation,
            updated_at_micros: metadata.updated_at_micros,
            policies,
            neural_network: metadata.neural_network,
            tactical_learning: metadata.tactical_learning,
        }))
    }

    pub fn load_latest_active_policy_metadata(
        &mut self,
        experiment_id: &str,
    ) -> Result<Option<SoccerLearningPgPolicyMetadata>, String> {
        let Some(row) = self
            .client
            .query_opt(
                r#"
                select
                  id::text,
                  generation,
                  metrics,
                  config,
                  coalesce((extract(epoch from updated_at) * 1000000)::bigint, 0)
                from des_soccer_learning_policy_versions
                where experiment_id = $1::text::uuid and status = 'active'
                order by generation desc, updated_at desc, id desc
                limit 1
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("select latest soccer policy version metadata: {err}"))?
        else {
            return Ok(None);
        };
        let id: String = row.get(0);
        let generation: i32 = row.get(1);
        let metrics: Value = row.get(2);
        let config: Value = row.get(3);
        let updated_at_micros: i64 = row.get(4);
        let neural_network = soccer_policy_version_neural_network_from_metrics(&metrics)?;
        let tactical_learning =
            soccer_policy_version_tactical_learning_from_values(&config, &metrics)?;
        Ok(Some(SoccerLearningPgPolicyMetadata {
            id,
            generation,
            updated_at_micros,
            neural_network,
            tactical_learning,
        }))
    }

    pub fn insert_policy_version(
        &mut self,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
    ) -> Result<String, String> {
        let policy_version_id = Uuid::new_v4().to_string();
        self.insert_policy_version_with_id(
            &policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
        )?;
        Ok(policy_version_id)
    }

    pub fn insert_policy_version_with_id(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
    ) -> Result<(), String> {
        self.insert_policy_version_with_id_inner(
            policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
            None,
        )
    }

    pub fn insert_policy_version_with_id_and_neural_network(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
        neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    ) -> Result<(), String> {
        self.insert_policy_version_with_id_inner(
            policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
            neural_network,
        )
    }

    fn insert_policy_version_with_id_inner(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
        neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    ) -> Result<(), String> {
        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let options_json = json!({
            "home": home_options,
            "away": away_options,
        });
        let lineage = parent_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let base_metrics = soccer_policy_version_metrics(
            fitness,
            neural_network,
            Some(&config.tactical_learning),
        )?;
        let retention_kind = SOCCER_POLICY_RETENTION_BRANCH_TIP;
        let full_entries_retained = soccer_policy_version_retains_full_entries(status);
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(fitness);
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer policy version transaction: {err}"))?;
        ensure_soccer_learning_policy_retention_columns(&mut tx)?;
        let branch_key = soccer_policy_branch_key_for_insert(
            &mut tx,
            parent_policy_version_id,
            policy_version_id,
        )?;
        let metrics = soccer_policy_version_metrics_with_retention(
            base_metrics,
            policy_version_id,
            &branch_key,
            retention_kind,
            full_entries_retained,
        );

        if status == "active" {
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::text::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let inserted = tx
            .execute(
                r#"
            insert into des_soccer_learning_policy_versions
              (
                id,
                experiment_id,
                parent_policy_version_id,
                generation,
                version_label,
                source_kind,
                status,
                options,
                config,
                lineage,
                metrics,
                entry_count,
                target_entry_count,
                visit_count,
                fitness_micros,
                branch_key,
                retention_kind,
                full_entries_retained
              )
            values
              (
                $1::text::uuid,
                $2::text::uuid,
                $3::text::uuid,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16::text::uuid,
                $17,
                $18
              )
            "#,
                &[
                    &policy_version_id,
                    &experiment_id,
                    &parent_policy_version_id,
                    &generation,
                    &version_label,
                    &source_kind,
                    &status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                    &branch_key,
                    &retention_kind,
                    &full_entries_retained,
                ],
            )
            .map_err(|err| format!("insert soccer policy version: {err}"))?;
        if inserted != 1 {
            return Err(format!(
                "insert soccer policy version inserted {inserted} rows for policy version {policy_version_id}"
            ));
        }

        if full_entries_retained {
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Home,
                &policies.home,
                None,
            )?;
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Away,
                &policies.away,
                None,
            )?;
            let _ = prune_old_policy_entries_for_branch_tip(
                &mut tx,
                experiment_id,
                &branch_key,
                policy_version_id,
            )?;
        }

        tx.commit()
            .map_err(|err| format!("commit soccer policy version: {err}"))?;
        Ok(())
    }

    pub fn insert_completed_run(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        output_policy_version_id: Option<&str>,
        game: &SoccerLearningCompletedGame,
    ) -> Result<String, String> {
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer run transaction: {err}"))?;
        let run_id = insert_completed_run_in_transaction(
            &mut tx,
            experiment_id,
            runner_id,
            base_policy_version_id,
            output_policy_version_id,
            game,
        )?;
        tx.commit()
            .map_err(|err| format!("commit soccer learning run: {err}"))?;
        Ok(run_id)
    }

    pub fn insert_completed_runs(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        runs: &[SoccerLearningPgCompletedRunInsert<'_>],
    ) -> Result<Vec<String>, String> {
        if runs.is_empty() {
            return Ok(Vec::new());
        }

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer run batch transaction: {err}"))?;
        let mut run_ids = Vec::with_capacity(runs.len());
        for chunk in runs.chunks(SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE) {
            let chunk_run_ids = insert_completed_run_headers_in_transaction(
                &mut tx,
                experiment_id,
                runner_id,
                chunk,
            )?;
            insert_completed_run_delta_rows_in_transaction(&mut tx, &chunk_run_ids, chunk)?;
            run_ids.extend(chunk_run_ids);
        }
        tx.commit()
            .map_err(|err| format!("commit soccer learning run batch: {err}"))?;
        Ok(run_ids)
    }

    pub fn prune_completed_runs_for_experiment(
        &mut self,
        experiment_id: &str,
        keep_latest_runs: usize,
    ) -> Result<SoccerLearningPgCompletedRunRetentionPrune, String> {
        let Some(keep_latest_runs) = soccer_completed_run_retention_limit(keep_latest_runs) else {
            return Ok(SoccerLearningPgCompletedRunRetentionPrune::default());
        };

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer completed-run retention transaction: {err}"))?;
        let deleted_delta_rows = tx
            .execute(
                r#"
                with stale_runs as (
                  select id
                  from (
                    select
                      id,
                      row_number() over (order by created_at desc, id desc) as keep_rank
                    from des_soccer_learning_runs
                    where experiment_id = $1::text::uuid
                  ) ranked
                  where keep_rank > $2
                )
                delete from des_soccer_learning_run_deltas deltas
                using stale_runs
                where deltas.run_id = stale_runs.id
                "#,
                &[&experiment_id, &keep_latest_runs],
            )
            .map_err(|err| format!("prune soccer completed-run deltas: {err}"))?;
        let deleted_runs = tx
            .execute(
                r#"
                with stale_runs as (
                  select id
                  from (
                    select
                      id,
                      row_number() over (order by created_at desc, id desc) as keep_rank
                    from des_soccer_learning_runs
                    where experiment_id = $1::text::uuid
                  ) ranked
                  where keep_rank > $2
                )
                delete from des_soccer_learning_runs runs
                using stale_runs
                where runs.id = stale_runs.id
                "#,
                &[&experiment_id, &keep_latest_runs],
            )
            .map_err(|err| format!("prune soccer completed runs: {err}"))?;
        tx.commit()
            .map_err(|err| format!("commit soccer completed-run retention: {err}"))?;

        Ok(SoccerLearningPgCompletedRunRetentionPrune {
            deleted_runs,
            deleted_delta_rows,
        })
    }

    pub fn insert_set_play_training_artifact(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        status: &str,
        artifact: &SoccerSetPlayTrainingArtifact,
        elapsed_seconds: f64,
    ) -> Result<(String, String), String> {
        let policies = SoccerTeamQPolicies {
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
        };
        let config_json = serde_json::to_value(&artifact.config)
            .map_err(|err| format!("serialize set-play config: {err}"))?;
        let options_json = json!({
            "home": &artifact.options,
            "away": &artifact.options,
        });
        let lineage = base_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let neural = json!({
            "enabled": artifact.learning.neural_learning_enabled,
            "backend": artifact.learning.neural_learning_backend,
            "trainingSteps": artifact.learning.neural_learning_training_steps,
            "samples": artifact.learning.neural_learning_samples,
            "pendingBatches": artifact.learning.neural_learning_pending_batches,
            "droppedBatches": artifact.learning.neural_learning_dropped_batches,
            "replaySamples": artifact.learning.neural_learning_replay_samples,
            "replayCapacity": artifact.learning.neural_learning_replay_capacity,
            "parameterCount": artifact.learning.neural_learning_parameter_count,
            "targetClip": artifact.learning.neural_learning_target_clip,
            "lastLoss": artifact.learning.neural_learning_last_loss,
            "averageLoss": artifact.learning.neural_learning_average_loss,
        });
        let base_metrics = json!({
            "fitness": artifact.goal_rate,
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
            "neural": neural,
        });
        let summary_json = json!({
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
        });
        let stats_json = json!({
            "learning": &artifact.learning,
            "neural": base_metrics["neural"].clone(),
            "episodes": &artifact.episodes,
        });
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(artifact.goal_rate);
        let score_home = checked_i32(if artifact.team == Team::Home {
            artifact.goals
        } else {
            0
        });
        let score_away = checked_i32(if artifact.team == Team::Away {
            artifact.goals
        } else {
            0
        });
        let home_goal_diff = if artifact.team == Team::Home {
            score_home
        } else {
            -score_away
        };
        let away_goal_diff = -home_goal_diff;
        let trained_team_scored = artifact.goals > 0;
        let (home_outcome, away_outcome) = match (artifact.team, trained_team_scored) {
            (Team::Home, true) => ("win", "loss"),
            (Team::Away, true) => ("loss", "win"),
            _ => ("draw", "draw"),
        };
        let trained_merge_weight = soccer_learning_to_micros(1.0 + artifact.goal_rate);
        let defending_merge_weight = soccer_learning_to_micros((1.0 - artifact.goal_rate) * 0.5);
        let (home_merge_weight_micros, away_merge_weight_micros) = match artifact.team {
            Team::Home => (trained_merge_weight, defending_merge_weight),
            Team::Away => (defending_merge_weight, trained_merge_weight),
        };
        let duration_ticks = checked_i64(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.ticks)
                .sum::<u64>(),
        );
        let simulated_seconds = artifact
            .episodes
            .iter()
            .map(|episode| episode.simulated_seconds)
            .sum::<f64>();
        let simulated_seconds_micros = soccer_learning_to_micros(simulated_seconds);
        let elapsed_millis = (elapsed_seconds.max(0.0) * 1000.0).round() as i64;
        let transitions = checked_i32(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.policy_updates)
                .sum::<u64>(),
        );

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer set-play training transaction: {err}"))?;
        ensure_soccer_learning_policy_retention_columns(&mut tx)?;
        ensure_soccer_learning_set_play_tables(&mut tx)?;
        let policy_version_id = Uuid::new_v4().to_string();
        let retention_kind = SOCCER_POLICY_RETENTION_BRANCH_TIP;
        let full_entries_retained = soccer_policy_version_retains_full_entries(status);
        let branch_key = soccer_policy_branch_key_for_insert(
            &mut tx,
            base_policy_version_id,
            &policy_version_id,
        )?;
        let metrics = soccer_policy_version_metrics_with_retention(
            base_metrics,
            &policy_version_id,
            &branch_key,
            retention_kind,
            full_entries_retained,
        );

        if status == "active" {
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::text::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let inserted_policy_rows = tx
            .execute(
                r#"
                insert into des_soccer_learning_policy_versions
                  (
                    id,
                    experiment_id,
                    parent_policy_version_id,
                    generation,
                    version_label,
                    source_kind,
                    status,
                    options,
                    config,
                    lineage,
                    metrics,
                    entry_count,
                    target_entry_count,
                    visit_count,
                    fitness_micros,
                    branch_key,
                    retention_kind,
                    full_entries_retained
                  )
                values
                  (
                    $1::text::uuid,
                    $2::text::uuid,
                    $3::text::uuid,
                    $4,
                    $5,
                    'replay',
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15::text::uuid,
                    $16,
                    $17
                  )
                "#,
                &[
                    &policy_version_id,
                    &experiment_id,
                    &base_policy_version_id,
                    &generation,
                    &version_label,
                    &status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                    &branch_key,
                    &retention_kind,
                    &full_entries_retained,
                ],
            )
            .map_err(|err| format!("insert soccer set-play policy version: {err}"))?;
        if inserted_policy_rows != 1 {
            return Err(format!(
                "insert soccer set-play policy version inserted {inserted_policy_rows} rows for policy version {policy_version_id}"
            ));
        }

        let run_row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_runs
                  (
                    experiment_id,
                    base_policy_version_id,
                    output_policy_version_id,
                    runner_id,
                    seed,
                    episode_index,
                    status,
                    score_home,
                    score_away,
                    home_goal_diff,
                    away_goal_diff,
                    home_outcome,
                    away_outcome,
                    home_merge_weight_micros,
                    away_merge_weight_micros,
                    fitness_micros,
                    duration_ticks,
                    simulated_seconds_micros,
                    elapsed_millis,
                    transitions,
                    summary,
                    stats
                  )
                values
                  (
                    $1::text::uuid,
                    $2::text::uuid,
                    $3::text::uuid,
                    $4,
                    $5,
                    0,
                    'completed',
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16,
                    $17,
                    $18,
                    $19,
                    $20
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &base_policy_version_id,
                    &policy_version_id,
                    &runner_id,
                    &(artifact.config.seed as i64),
                    &score_home,
                    &score_away,
                    &home_goal_diff,
                    &away_goal_diff,
                    &home_outcome,
                    &away_outcome,
                    &home_merge_weight_micros,
                    &away_merge_weight_micros,
                    &fitness_micros,
                    &duration_ticks,
                    &simulated_seconds_micros,
                    &elapsed_millis,
                    &transitions,
                    &summary_json,
                    &stats_json,
                ],
            )
            .map_err(|err| format!("insert soccer set-play learning run: {err}"))?;
        let run_id: String = run_row.get(0);

        if full_entries_retained {
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Home,
                &policies.home,
                Some(&run_id),
            )?;
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Away,
                &policies.away,
                Some(&run_id),
            )?;
            let _ = prune_old_policy_entries_for_branch_tip(
                &mut tx,
                experiment_id,
                &branch_key,
                &policy_version_id,
            )?;
        }
        insert_normalized_set_play_training_records(
            &mut tx,
            &run_id,
            &policy_version_id,
            artifact,
        )?;

        tx.commit()
            .map_err(|err| format!("commit soccer set-play training transaction: {err}"))?;
        Ok((policy_version_id, run_id))
    }

    fn load_policy_entries(
        &mut self,
        policy_version_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
    ) -> Result<SoccerTeamQPolicies, String> {
        let rows = self
            .client
            .query(
                r#"
                select
                  team,
                  entry_kind,
                  state_key,
                  action,
                  target_fine_cell_id,
                  target_tactical_cell_id,
                  target_macro_cell_id,
                  target_root_cell_id,
                  value_micros,
                  visits
                from des_soccer_learning_policy_entries
                where policy_version_id = $1::text::uuid
                order by team, entry_kind, state_hash, action
                "#,
                &[&policy_version_id],
            )
            .map_err(|err| format!("select soccer policy entries: {err}"))?;
        let mut home_entries = Vec::new();
        let mut away_entries = Vec::new();
        let mut home_targets = Vec::new();
        let mut away_targets = Vec::new();

        for row in rows {
            let team: String = row.get(0);
            let entry_kind: String = row.get(1);
            let state_key_json: Value = row.get(2);
            let state: SoccerQStateKey = serde_json::from_value(state_key_json)
                .map_err(|err| format!("decode soccer policy state key: {err}"))?;
            let action: String = row.get(3);
            let target_fine_cell_id: i32 = row.get(4);
            let target_tactical_cell_id: i32 = row.get(5);
            let target_macro_cell_id: i32 = row.get(6);
            let target_root_cell_id: i32 = row.get(7);
            let value_micros: i64 = row.get(8);
            let visits_i32: i32 = row.get(9);
            let visits = visits_i32.max(0) as u32;
            let value = soccer_learning_from_micros(value_micros);
            match (team.as_str(), entry_kind.as_str()) {
                ("home", "action") => home_entries.push(SoccerQEntry {
                    state,
                    action,
                    value,
                    visits,
                }),
                ("away", "action") => away_entries.push(SoccerQEntry {
                    state,
                    action,
                    value,
                    visits,
                }),
                ("home", "target") => home_targets.push(SoccerQTargetEntry {
                    state,
                    action,
                    target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                    target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                    target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                    target_root_cell_id: target_root_cell_id.max(0) as usize,
                    value,
                    visits,
                }),
                ("away", "target") => away_targets.push(SoccerQTargetEntry {
                    state,
                    action,
                    target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                    target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                    target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                    target_root_cell_id: target_root_cell_id.max(0) as usize,
                    value,
                    visits,
                }),
                _ => {}
            }
        }

        Ok(SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                home_options,
                &home_entries,
                &home_targets,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                away_options,
                &away_entries,
                &away_targets,
            )?,
        })
    }
}

fn insert_completed_run_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    runner_id: &str,
    base_policy_version_id: Option<&str>,
    output_policy_version_id: Option<&str>,
    game: &SoccerLearningCompletedGame,
) -> Result<String, String> {
    let summary_json =
        serde_json::to_value(&game.summary).map_err(|err| format!("serialize summary: {err}"))?;
    let stats_json = serde_json::to_value(&game.summary.stats)
        .map_err(|err| format!("serialize stats: {err}"))?;
    let seed = checked_i64(game.seed);
    let episode_index = checked_i32(game.episode);
    let score_home = checked_i32(game.summary.score_home);
    let score_away = checked_i32(game.summary.score_away);
    let home_goal_diff = game.score.home.goal_diff;
    let away_goal_diff = game.score.away.goal_diff;
    let home_outcome = game.score.home.outcome.as_str();
    let away_outcome = game.score.away.outcome.as_str();
    let home_merge_weight_micros = game.score.home.merge_weight_micros;
    let away_merge_weight_micros = game.score.away.merge_weight_micros;
    let duration_ticks = checked_i64(game.summary.ticks);
    let simulated_seconds_micros = soccer_learning_to_micros(game.summary.simulated_seconds);
    let elapsed_millis = (game.elapsed_seconds * 1000.0).round().max(0.0) as i64;
    let transitions = checked_i32(game.episode_summary.transitions);
    let row = tx
        .query_one(
            r#"
            insert into des_soccer_learning_runs
              (
                experiment_id,
                base_policy_version_id,
                output_policy_version_id,
                runner_id,
                seed,
                episode_index,
                status,
                score_home,
                score_away,
                home_goal_diff,
                away_goal_diff,
                home_outcome,
                away_outcome,
                home_merge_weight_micros,
                away_merge_weight_micros,
                fitness_micros,
                duration_ticks,
                simulated_seconds_micros,
                elapsed_millis,
                transitions,
                summary,
                stats
              )
            values
              (
                $1::text::uuid,
                $2::text::uuid,
                $3::text::uuid,
                $4,
                $5,
                $6,
                'completed',
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                $20,
                $21
              )
            returning id::text
            "#,
            &[
                &experiment_id,
                &base_policy_version_id,
                &output_policy_version_id,
                &runner_id,
                &seed,
                &episode_index,
                &score_home,
                &score_away,
                &home_goal_diff,
                &away_goal_diff,
                &home_outcome,
                &away_outcome,
                &home_merge_weight_micros,
                &away_merge_weight_micros,
                &game.score.match_fitness_micros,
                &duration_ticks,
                &simulated_seconds_micros,
                &elapsed_millis,
                &transitions,
                &summary_json,
                &stats_json,
            ],
        )
        .map_err(|err| format!("insert soccer learning run: {err}"))?;
    let run_id: String = row.get(0);

    insert_run_delta_rows(tx, &run_id, &game.delta.entries)?;

    Ok(run_id)
}

#[derive(Clone, Debug)]
struct SoccerCompletedRunHeaderInsert<'a> {
    run_id: String,
    base_policy_version_id: Option<&'a str>,
    output_policy_version_id: Option<&'a str>,
    seed: i64,
    episode_index: i32,
    score_home: i32,
    score_away: i32,
    home_goal_diff: i32,
    away_goal_diff: i32,
    home_outcome: &'static str,
    away_outcome: &'static str,
    home_merge_weight_micros: i64,
    away_merge_weight_micros: i64,
    fitness_micros: i64,
    duration_ticks: i64,
    simulated_seconds_micros: i64,
    elapsed_millis: i64,
    transitions: i32,
    summary_json: Value,
    stats_json: Value,
}

fn completed_run_header_insert<'a>(
    run: &SoccerLearningPgCompletedRunInsert<'a>,
) -> Result<SoccerCompletedRunHeaderInsert<'a>, String> {
    let game = run.game;
    let summary_json =
        serde_json::to_value(&game.summary).map_err(|err| format!("serialize summary: {err}"))?;
    let stats_json = serde_json::to_value(&game.summary.stats)
        .map_err(|err| format!("serialize stats: {err}"))?;
    Ok(SoccerCompletedRunHeaderInsert {
        run_id: Uuid::new_v4().to_string(),
        base_policy_version_id: run.base_policy_version_id,
        output_policy_version_id: run.output_policy_version_id,
        seed: checked_i64(game.seed),
        episode_index: checked_i32(game.episode),
        score_home: checked_i32(game.summary.score_home),
        score_away: checked_i32(game.summary.score_away),
        home_goal_diff: game.score.home.goal_diff,
        away_goal_diff: game.score.away.goal_diff,
        home_outcome: game.score.home.outcome.as_str(),
        away_outcome: game.score.away.outcome.as_str(),
        home_merge_weight_micros: game.score.home.merge_weight_micros,
        away_merge_weight_micros: game.score.away.merge_weight_micros,
        fitness_micros: game.score.match_fitness_micros,
        duration_ticks: checked_i64(game.summary.ticks),
        simulated_seconds_micros: soccer_learning_to_micros(game.summary.simulated_seconds),
        elapsed_millis: (game.elapsed_seconds * 1000.0).round().max(0.0) as i64,
        transitions: checked_i32(game.episode_summary.transitions),
        summary_json,
        stats_json,
    })
}

fn insert_completed_run_headers_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    runner_id: &str,
    runs: &[SoccerLearningPgCompletedRunInsert<'_>],
) -> Result<Vec<String>, String> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }

    let batch_rows = runs
        .iter()
        .map(completed_run_header_insert)
        .collect::<Result<Vec<_>, _>>()?;
    let sql_prefix = r#"
        insert into des_soccer_learning_runs
          (
            id,
            experiment_id,
            base_policy_version_id,
            output_policy_version_id,
            runner_id,
            seed,
            episode_index,
            status,
            score_home,
            score_away,
            home_goal_diff,
            away_goal_diff,
            home_outcome,
            away_outcome,
            home_merge_weight_micros,
            away_merge_weight_micros,
            fitness_micros,
            duration_ticks,
            simulated_seconds_micros,
            elapsed_millis,
            transitions,
            summary,
            stats
          )
        values
        "#;
    let mut sql = postgres_insert_sql_buffer(
        sql_prefix,
        batch_rows.len(),
        SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT,
    );
    let mut params: Vec<&(dyn ToSql + Sync)> =
        Vec::with_capacity(batch_rows.len() * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT);
    for (idx, row) in batch_rows.iter().enumerate() {
        if idx > 0 {
            sql.push_str(", ");
        }
        append_completed_run_header_value_tuple(
            &mut sql,
            idx * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT + 1,
        );
        params.push(&row.run_id);
        params.push(&experiment_id);
        params.push(&row.base_policy_version_id);
        params.push(&row.output_policy_version_id);
        params.push(&runner_id);
        params.push(&row.seed);
        params.push(&row.episode_index);
        params.push(&row.score_home);
        params.push(&row.score_away);
        params.push(&row.home_goal_diff);
        params.push(&row.away_goal_diff);
        params.push(&row.home_outcome);
        params.push(&row.away_outcome);
        params.push(&row.home_merge_weight_micros);
        params.push(&row.away_merge_weight_micros);
        params.push(&row.fitness_micros);
        params.push(&row.duration_ticks);
        params.push(&row.simulated_seconds_micros);
        params.push(&row.elapsed_millis);
        params.push(&row.transitions);
        params.push(&row.summary_json);
        params.push(&row.stats_json);
    }
    let inserted = tx
        .execute(&sql, &params)
        .map_err(|err| format!("insert soccer learning run header batch: {err}"))?;
    if inserted as usize != batch_rows.len() {
        return Err(format!(
            "insert soccer learning run header batch inserted {inserted} rows for {} inputs",
            batch_rows.len()
        ));
    }
    Ok(batch_rows.into_iter().map(|row| row.run_id).collect())
}

#[derive(Clone, Copy, Debug)]
struct SoccerRunDeltaBatchEntryInsert<'a> {
    run_index: usize,
    delta: &'a SoccerLearningPolicyDeltaEntry,
    team: &'static str,
    entry_kind: &'static str,
    visit_delta: i32,
}

fn soccer_run_delta_batch_entry_insert(
    run_index: usize,
    delta: &SoccerLearningPolicyDeltaEntry,
) -> SoccerRunDeltaBatchEntryInsert<'_> {
    SoccerRunDeltaBatchEntryInsert {
        run_index,
        delta,
        team: soccer_team_label(delta.team),
        entry_kind: delta.entry_kind.as_str(),
        visit_delta: checked_i32(delta.visit_delta),
    }
}

fn insert_run_delta_rows(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    rows: &[SoccerLearningPolicyDeltaEntry],
) -> Result<(), String> {
    let run_ids = [run_id.to_string()];
    let mut batch_rows = Vec::with_capacity(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE);
    for delta in rows {
        batch_rows.push(soccer_run_delta_batch_entry_insert(0, delta));
        if batch_rows.len() == SOCCER_RUN_DELTA_INSERT_BATCH_SIZE {
            insert_run_delta_batch_rows(tx, &run_ids, &batch_rows)?;
            batch_rows.clear();
        }
    }
    if !batch_rows.is_empty() {
        insert_run_delta_batch_rows(tx, &run_ids, &batch_rows)?;
    }
    Ok(())
}

fn insert_completed_run_delta_rows_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    run_ids: &[String],
    runs: &[SoccerLearningPgCompletedRunInsert<'_>],
) -> Result<(), String> {
    if run_ids.len() != runs.len() {
        return Err(format!(
            "insert soccer learning run delta batch got {} run ids for {} runs",
            run_ids.len(),
            runs.len()
        ));
    }
    let mut batch_rows = Vec::with_capacity(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE);
    for (run_index, run) in runs.iter().enumerate() {
        for delta in &run.game.delta.entries {
            batch_rows.push(soccer_run_delta_batch_entry_insert(run_index, delta));
            if batch_rows.len() == SOCCER_RUN_DELTA_INSERT_BATCH_SIZE {
                insert_run_delta_batch_rows(tx, run_ids, &batch_rows)?;
                batch_rows.clear();
            }
        }
    }
    if !batch_rows.is_empty() {
        insert_run_delta_batch_rows(tx, run_ids, &batch_rows)?;
    }
    Ok(())
}

fn insert_run_delta_batch_rows(
    tx: &mut postgres::Transaction<'_>,
    run_ids: &[String],
    rows: &[SoccerRunDeltaBatchEntryInsert<'_>],
) -> Result<(), String> {
    for chunk in rows.chunks(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_run_deltas
              (
                run_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                target_fine_cell_id,
                target_tactical_cell_id,
                target_macro_cell_id,
                target_root_cell_id,
                before_value_micros,
                after_value_micros,
                value_delta_micros,
                visit_delta,
                merge_weight_micros,
                effective_visit_micros
              )
            values
            "#;
        let mut sql =
            postgres_insert_sql_buffer(sql_prefix, chunk.len(), SOCCER_RUN_DELTA_PARAMETER_COUNT);
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_RUN_DELTA_PARAMETER_COUNT);
        for (idx, batch_row) in chunk.iter().enumerate() {
            let delta = batch_row.delta;
            let run_id = run_ids.get(batch_row.run_index).ok_or_else(|| {
                format!(
                    "insert soccer learning run delta batch has row for missing run index {}",
                    batch_row.run_index
                )
            })?;
            if idx > 0 {
                sql.push_str(", ");
            }
            append_run_delta_value_tuple(&mut sql, idx * SOCCER_RUN_DELTA_PARAMETER_COUNT + 1);
            params.push(run_id);
            params.push(&batch_row.team);
            params.push(&batch_row.entry_kind);
            params.push(&delta.state_hash);
            params.push(&delta.state_json);
            params.push(&delta.action);
            params.push(&delta.target_fine_cell_id);
            params.push(&delta.target_tactical_cell_id);
            params.push(&delta.target_macro_cell_id);
            params.push(&delta.target_root_cell_id);
            params.push(&delta.before_value_micros);
            params.push(&delta.after_value_micros);
            params.push(&delta.value_delta_micros);
            params.push(&batch_row.visit_delta);
            params.push(&delta.merge_weight_micros);
            params.push(&delta.effective_visit_micros);
        }
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer learning run delta batch: {err}"))?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SoccerPolicyActionEntryInsert {
    state_json: Value,
    state_hash: String,
    action: String,
    value_micros: i64,
    visits: i32,
}

#[derive(Clone, Debug)]
struct SoccerPolicyTargetEntryInsert {
    state_json: Value,
    state_hash: String,
    action: String,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
    value_micros: i64,
    visits: i32,
}

fn insert_policy_entries_for_team(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team: Team,
    policy: &SoccerQPolicy,
    source_run_id: Option<&str>,
) -> Result<(), String> {
    let team_label = soccer_team_label(team);
    let mut action_rows = Vec::new();
    for entry in policy.entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer action state key: {err}"))?;
        action_rows.push(SoccerPolicyActionEntryInsert {
            state_hash: state_hash(&state_json),
            state_json,
            action: entry.action,
            value_micros: soccer_learning_to_micros(entry.value),
            visits: checked_i32(entry.visits),
        });
    }
    insert_policy_action_entry_rows(
        tx,
        policy_version_id,
        &team_label,
        source_run_id,
        &action_rows,
    )?;

    let mut target_rows = Vec::new();
    for entry in policy.target_entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer target state key: {err}"))?;
        target_rows.push(SoccerPolicyTargetEntryInsert {
            state_hash: state_hash(&state_json),
            state_json,
            action: entry.action,
            target_fine_cell_id: checked_i32(entry.target_fine_cell_id),
            target_tactical_cell_id: checked_i32(entry.target_tactical_cell_id),
            target_macro_cell_id: checked_i32(entry.target_macro_cell_id),
            target_root_cell_id: checked_i32(entry.target_root_cell_id),
            value_micros: soccer_learning_to_micros(entry.value),
            visits: checked_i32(entry.visits),
        });
    }
    insert_policy_target_entry_rows(
        tx,
        policy_version_id,
        &team_label,
        source_run_id,
        &target_rows,
    )
}

fn insert_policy_action_entry_rows(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team_label: &str,
    source_run_id: Option<&str>,
    rows: &[SoccerPolicyActionEntryInsert],
) -> Result<(), String> {
    let entry_kind = SoccerLearningPolicyEntryKind::Action.as_str();
    for chunk in rows.chunks(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                value_micros,
                visits,
                source_run_id
              )
            values
            "#;
        let mut sql = postgres_insert_sql_buffer(
            sql_prefix,
            chunk.len(),
            SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT,
        );
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT);
        for (idx, row) in chunk.iter().enumerate() {
            if idx > 0 {
                sql.push_str(", ");
            }
            append_policy_entry_value_tuple(
                &mut sql,
                idx * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT + 1,
                false,
            );
            params.push(&policy_version_id);
            params.push(&team_label);
            params.push(&entry_kind);
            params.push(&row.state_hash);
            params.push(&row.state_json);
            params.push(&row.action);
            params.push(&row.value_micros);
            params.push(&row.visits);
            params.push(&source_run_id);
        }
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer policy action entry batch: {err}"))?;
    }
    Ok(())
}

fn insert_policy_target_entry_rows(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team_label: &str,
    source_run_id: Option<&str>,
    rows: &[SoccerPolicyTargetEntryInsert],
) -> Result<(), String> {
    let entry_kind = SoccerLearningPolicyEntryKind::Target.as_str();
    for chunk in rows.chunks(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                target_fine_cell_id,
                target_tactical_cell_id,
                target_macro_cell_id,
                target_root_cell_id,
                value_micros,
                visits,
                source_run_id
              )
            values
            "#;
        let mut sql = postgres_insert_sql_buffer(
            sql_prefix,
            chunk.len(),
            SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT,
        );
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT);
        for (idx, row) in chunk.iter().enumerate() {
            if idx > 0 {
                sql.push_str(", ");
            }
            append_policy_entry_value_tuple(
                &mut sql,
                idx * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT + 1,
                true,
            );
            params.push(&policy_version_id);
            params.push(&team_label);
            params.push(&entry_kind);
            params.push(&row.state_hash);
            params.push(&row.state_json);
            params.push(&row.action);
            params.push(&row.target_fine_cell_id);
            params.push(&row.target_tactical_cell_id);
            params.push(&row.target_macro_cell_id);
            params.push(&row.target_root_cell_id);
            params.push(&row.value_micros);
            params.push(&row.visits);
            params.push(&source_run_id);
        }
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer policy target entry batch: {err}"))?;
    }
    Ok(())
}

fn append_completed_run_header_value_tuple(sql: &mut String, first_param: usize) {
    write!(
        sql,
        "(${}::text::uuid, ${}::text::uuid, ${}::text::uuid, ${}::text::uuid, ${}, ${}, ${}, 'completed', ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
        first_param,
        first_param + 1,
        first_param + 2,
        first_param + 3,
        first_param + 4,
        first_param + 5,
        first_param + 6,
        first_param + 7,
        first_param + 8,
        first_param + 9,
        first_param + 10,
        first_param + 11,
        first_param + 12,
        first_param + 13,
        first_param + 14,
        first_param + 15,
        first_param + 16,
        first_param + 17,
        first_param + 18,
        first_param + 19,
        first_param + 20,
        first_param + 21
    )
    .expect("write completed run header tuple");
}

fn append_run_delta_value_tuple(sql: &mut String, first_param: usize) {
    write!(
        sql,
        "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
        first_param,
        first_param + 1,
        first_param + 2,
        first_param + 3,
        first_param + 4,
        first_param + 5,
        first_param + 6,
        first_param + 7,
        first_param + 8,
        first_param + 9,
        first_param + 10,
        first_param + 11,
        first_param + 12,
        first_param + 13,
        first_param + 14,
        first_param + 15
    )
    .expect("write run delta tuple");
}

fn append_policy_entry_value_tuple(
    sql: &mut String,
    first_param: usize,
    include_target_cells: bool,
) {
    if include_target_cells {
        write!(
            sql,
            "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}::text::uuid)",
            first_param,
            first_param + 1,
            first_param + 2,
            first_param + 3,
            first_param + 4,
            first_param + 5,
            first_param + 6,
            first_param + 7,
            first_param + 8,
            first_param + 9,
            first_param + 10,
            first_param + 11,
            first_param + 12
        )
        .expect("write target policy entry tuple");
    } else {
        write!(
            sql,
            "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}::text::uuid)",
            first_param,
            first_param + 1,
            first_param + 2,
            first_param + 3,
            first_param + 4,
            first_param + 5,
            first_param + 6,
            first_param + 7,
            first_param + 8
        )
        .expect("write action policy entry tuple");
    }
}

fn ensure_soccer_learning_policy_retention_columns(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        alter table des_soccer_learning_policy_versions
          add column if not exists branch_key uuid,
          add column if not exists retention_kind varchar(32) default 'branch_tip' not null,
          add column if not exists full_entries_retained boolean default true not null,
          add column if not exists full_entries_pruned_at timestamptz;

        with recursive policy_lineage as (
          select
            id,
            id as root_branch_key
          from des_soccer_learning_policy_versions
          where parent_policy_version_id is null
          union all
          select
            child.id,
            parent.root_branch_key
          from des_soccer_learning_policy_versions child
          join policy_lineage parent
            on child.parent_policy_version_id = parent.id
        )
        update des_soccer_learning_policy_versions versions
        set branch_key = policy_lineage.root_branch_key
        from policy_lineage
        where versions.id = policy_lineage.id
          and (
            versions.branch_key is null
            or (
              versions.branch_key = versions.id
              and versions.parent_policy_version_id is not null
            )
          );

        alter table des_soccer_learning_policy_versions
          alter column branch_key set not null;

        do $$
        begin
          if not exists (
            select 1
            from pg_constraint
            where conname = 'des_soccer_learning_policy_versions_retention_kind_chk'
          ) then
            alter table des_soccer_learning_policy_versions
              add constraint des_soccer_learning_policy_versions_retention_kind_chk
              check (retention_kind in ('branch_tip', 'retain_all', 'metadata_only'));
          end if;
        end $$;

        create index if not exists des_soccer_learning_policy_versions_branch_tip_idx
          on des_soccer_learning_policy_versions (
            experiment_id,
            branch_key,
            generation desc,
            updated_at desc
          )
          where full_entries_retained = true;
        "#,
    )
    .map_err(|err| format!("ensure soccer policy retention columns: {err}"))?;
    Ok(())
}

fn soccer_policy_branch_key_for_insert(
    tx: &mut postgres::Transaction<'_>,
    parent_policy_version_id: Option<&str>,
    policy_version_id: &str,
) -> Result<String, String> {
    if let Some(parent_policy_version_id) = parent_policy_version_id {
        if let Some(row) = tx
            .query_opt(
                r#"
                select branch_key::text
                from des_soccer_learning_policy_versions
                where id = $1::text::uuid
                limit 1
                "#,
                &[&parent_policy_version_id],
            )
            .map_err(|err| format!("select parent soccer policy branch key: {err}"))?
        {
            return Ok(row.get(0));
        }
    }
    Ok(policy_version_id.to_string())
}

fn soccer_completed_run_retention_limit(keep_latest_runs: usize) -> Option<i64> {
    if keep_latest_runs == 0 {
        None
    } else {
        Some(keep_latest_runs.min(i64::MAX as usize) as i64)
    }
}

fn prune_old_policy_entries_for_branch_tip(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    branch_key: &str,
    policy_version_id: &str,
) -> Result<u64, String> {
    tx.execute(
        r#"
        with pruned_versions as (
          update des_soccer_learning_policy_versions
          set
            full_entries_retained = false,
            full_entries_pruned_at = now(),
            updated_at = now(),
            metrics = jsonb_set(
              metrics,
              '{postgresRetention}',
              coalesce(metrics->'postgresRetention', '{}'::jsonb)
                || jsonb_build_object(
                  'fullEntriesRetained', false,
                  'prunedByPolicyVersionId', $3,
                  'retentionKind', 'branch_tip'
                ),
              true
            )
          where experiment_id = $1::text::uuid
            and branch_key = $2::text::uuid
            and id <> $3::text::uuid
            and full_entries_retained = true
          returning id
        )
        delete from des_soccer_learning_policy_entries entries
        using pruned_versions
        where entries.policy_version_id = pruned_versions.id
        "#,
        &[&experiment_id, &branch_key, &policy_version_id],
    )
    .map_err(|err| format!("prune old soccer policy branch entries: {err}"))
}

fn ensure_soccer_learning_set_play_tables(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        create table if not exists des_soccer_learning_set_play_runs (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          primary_restart varchar(40) not null,
          team varchar(8) not null,
          spot_x_micros bigint not null,
          spot_y_micros bigint not null,
          duration_seconds_micros bigint not null,
          episode_count integer not null,
          goals integer not null,
          goal_rate_micros bigint not null,
          first_window_goal_rate_micros bigint not null,
          last_window_goal_rate_micros bigint not null,
          goal_rate_delta_micros bigint not null,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_set_play_runs_restart_chk
            check (primary_restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_runs_team_chk
            check (team in ('home', 'away')),
          constraint des_soccer_learning_set_play_runs_duration_chk
            check (duration_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_runs_episode_chk
            check (episode_count >= 0),
          constraint des_soccer_learning_set_play_runs_goals_chk
            check (goals >= 0),
          constraint des_soccer_learning_set_play_runs_goal_rate_chk
            check (goal_rate_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_set_play_restart_mix (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          ordinal integer not null,
          restart varchar(40) not null,
          primary key (run_id, ordinal),
          constraint des_soccer_learning_set_play_restart_mix_ordinal_chk
            check (ordinal >= 0),
          constraint des_soccer_learning_set_play_restart_mix_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick'))
        );

        create table if not exists des_soccer_learning_set_play_episode_metrics (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          episode_index integer not null,
          seed bigint not null,
          restart varchar(40) not null,
          routine varchar(80),
          scored boolean not null,
          score_delta_for_team integer not null,
          ticks bigint not null,
          simulated_seconds_micros bigint not null,
          policy_updates bigint not null,
          home_policy_entries integer not null,
          home_policy_target_entries integer not null,
          away_policy_entries integer not null,
          away_policy_target_entries integer not null,
          neural_training_steps integer not null,
          neural_samples bigint not null,
          neural_replay_samples integer not null,
          neural_last_loss_micros bigint,
          cumulative_goals integer not null,
          goal_rate_so_far_micros bigint not null,
          primary key (run_id, episode_index),
          constraint des_soccer_learning_set_play_episode_idx_chk
            check (episode_index >= 0),
          constraint des_soccer_learning_set_play_episode_seed_chk
            check (seed >= 0),
          constraint des_soccer_learning_set_play_episode_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_episode_ticks_chk
            check (ticks >= 0),
          constraint des_soccer_learning_set_play_episode_seconds_chk
            check (simulated_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_episode_policy_updates_chk
            check (policy_updates >= 0),
          constraint des_soccer_learning_set_play_episode_entries_chk
            check (
              home_policy_entries >= 0
              and home_policy_target_entries >= 0
              and away_policy_entries >= 0
              and away_policy_target_entries >= 0
            ),
          constraint des_soccer_learning_set_play_episode_neural_chk
            check (
              neural_training_steps >= 0
              and neural_samples >= 0
              and neural_replay_samples >= 0
            ),
          constraint des_soccer_learning_set_play_episode_goals_chk
            check (cumulative_goals >= 0),
          constraint des_soccer_learning_set_play_episode_goal_rate_chk
            check (goal_rate_so_far_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_neural_run_metrics (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          enabled boolean not null,
          backend varchar(32) not null,
          training_steps integer not null,
          samples bigint not null,
          pending_batches integer not null,
          dropped_batches integer not null,
          replay_samples integer not null,
          replay_capacity integer not null,
          parameter_count integer not null,
          target_clip_micros bigint not null,
          last_loss_micros bigint,
          average_loss_micros bigint,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_neural_run_backend_chk
            check (backend in ('inline', 'threaded')),
          constraint des_soccer_learning_neural_run_counts_chk
            check (
              training_steps >= 0
              and samples >= 0
              and pending_batches >= 0
              and dropped_batches >= 0
              and replay_samples >= 0
              and replay_capacity >= 0
              and parameter_count >= 0
            )
        );

        create index if not exists des_soccer_learning_set_play_episode_restart_idx
          on des_soccer_learning_set_play_episode_metrics (restart, scored, episode_index);

        create index if not exists des_soccer_learning_neural_run_steps_idx
          on des_soccer_learning_neural_run_metrics (training_steps desc, samples desc);
        "#,
    )
    .map_err(|err| format!("ensure soccer set-play learning tables: {err}"))?;
    Ok(())
}

fn insert_normalized_set_play_training_records(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    policy_version_id: &str,
    artifact: &SoccerSetPlayTrainingArtifact,
) -> Result<(), String> {
    let team = soccer_team_label(artifact.team);
    let primary_restart = artifact.restart.as_label();
    let spot_x_micros = soccer_learning_to_micros(artifact.spot.x);
    let spot_y_micros = soccer_learning_to_micros(artifact.spot.y);
    let duration_seconds_micros = soccer_learning_to_micros(artifact.duration_seconds);
    let episode_count = checked_i32(artifact.episodes.len());
    let goals = checked_i32(artifact.goals);
    let goal_rate_micros = soccer_learning_to_micros(artifact.goal_rate);
    let first_window_goal_rate_micros = soccer_learning_to_micros(artifact.first_window_goal_rate);
    let last_window_goal_rate_micros = soccer_learning_to_micros(artifact.last_window_goal_rate);
    let goal_rate_delta_micros = soccer_learning_to_micros(artifact.goal_rate_delta);

    tx.execute(
        r#"
        insert into des_soccer_learning_set_play_runs
          (
            run_id,
            policy_version_id,
            primary_restart,
            team,
            spot_x_micros,
            spot_y_micros,
            duration_seconds_micros,
            episode_count,
            goals,
            goal_rate_micros,
            first_window_goal_rate_micros,
            last_window_goal_rate_micros,
            goal_rate_delta_micros
          )
        values
          ($1::text::uuid, $2::text::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &primary_restart,
            &team,
            &spot_x_micros,
            &spot_y_micros,
            &duration_seconds_micros,
            &episode_count,
            &goals,
            &goal_rate_micros,
            &first_window_goal_rate_micros,
            &last_window_goal_rate_micros,
            &goal_rate_delta_micros,
        ],
    )
    .map_err(|err| format!("insert soccer set-play run metrics: {err}"))?;

    for (ordinal, restart) in artifact.restarts.iter().enumerate() {
        let ordinal = checked_i32(ordinal);
        let restart_label = restart.as_label();
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_restart_mix
              (run_id, ordinal, restart)
            values
              ($1::text::uuid, $2, $3)
            "#,
            &[&run_id, &ordinal, &restart_label],
        )
        .map_err(|err| format!("insert soccer set-play restart mix: {err}"))?;
    }

    for episode in &artifact.episodes {
        let episode_index = checked_i32(episode.episode);
        let seed = checked_i64(episode.seed);
        let restart = episode.restart.as_label();
        let routine = episode
            .routine
            .map(|routine| routine.as_label().to_string());
        let ticks = checked_i64(episode.ticks);
        let simulated_seconds_micros = soccer_learning_to_micros(episode.simulated_seconds);
        let policy_updates = checked_i64(episode.policy_updates);
        let home_policy_entries = checked_i32(episode.home_policy_entries);
        let home_policy_target_entries = checked_i32(episode.home_policy_target_entries);
        let away_policy_entries = checked_i32(episode.away_policy_entries);
        let away_policy_target_entries = checked_i32(episode.away_policy_target_entries);
        let neural_training_steps = checked_i32(episode.neural_training_steps);
        let neural_samples = checked_i64(episode.neural_samples as u64);
        let neural_replay_samples = checked_i32(episode.neural_replay_samples);
        let neural_last_loss_micros = episode.neural_last_loss.map(soccer_learning_to_micros);
        let cumulative_goals = checked_i32(episode.cumulative_goals);
        let goal_rate_so_far_micros = soccer_learning_to_micros(episode.goal_rate_so_far);
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_episode_metrics
              (
                run_id,
                episode_index,
                seed,
                restart,
                routine,
                scored,
                score_delta_for_team,
                ticks,
                simulated_seconds_micros,
                policy_updates,
                home_policy_entries,
                home_policy_target_entries,
                away_policy_entries,
                away_policy_target_entries,
                neural_training_steps,
                neural_samples,
                neural_replay_samples,
                neural_last_loss_micros,
                cumulative_goals,
                goal_rate_so_far_micros
              )
            values
              (
                $1::text::uuid,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                $20
              )
            "#,
            &[
                &run_id,
                &episode_index,
                &seed,
                &restart,
                &routine,
                &episode.scored,
                &episode.score_delta_for_team,
                &ticks,
                &simulated_seconds_micros,
                &policy_updates,
                &home_policy_entries,
                &home_policy_target_entries,
                &away_policy_entries,
                &away_policy_target_entries,
                &neural_training_steps,
                &neural_samples,
                &neural_replay_samples,
                &neural_last_loss_micros,
                &cumulative_goals,
                &goal_rate_so_far_micros,
            ],
        )
        .map_err(|err| format!("insert soccer set-play episode metrics: {err}"))?;
    }

    let enabled = artifact.learning.neural_learning_enabled;
    let backend = artifact.learning.neural_learning_backend.as_str();
    let training_steps = checked_i32(artifact.learning.neural_learning_training_steps);
    let samples = checked_i64(artifact.learning.neural_learning_samples as u64);
    let pending_batches = checked_i32(artifact.learning.neural_learning_pending_batches);
    let dropped_batches = checked_i32(artifact.learning.neural_learning_dropped_batches);
    let replay_samples = checked_i32(artifact.learning.neural_learning_replay_samples);
    let replay_capacity = checked_i32(artifact.learning.neural_learning_replay_capacity);
    let parameter_count = checked_i32(artifact.learning.neural_learning_parameter_count);
    let target_clip_micros =
        soccer_learning_to_micros(artifact.learning.neural_learning_target_clip);
    let last_loss_micros = artifact
        .learning
        .neural_learning_last_loss
        .map(soccer_learning_to_micros);
    let average_loss_micros = artifact
        .learning
        .neural_learning_average_loss
        .map(soccer_learning_to_micros);
    tx.execute(
        r#"
        insert into des_soccer_learning_neural_run_metrics
          (
            run_id,
            policy_version_id,
            enabled,
            backend,
            training_steps,
            samples,
            pending_batches,
            dropped_batches,
            replay_samples,
            replay_capacity,
            parameter_count,
            target_clip_micros,
            last_loss_micros,
            average_loss_micros
          )
        values
          ($1::text::uuid, $2::text::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &enabled,
            &backend,
            &training_steps,
            &samples,
            &pending_batches,
            &dropped_batches,
            &replay_samples,
            &replay_capacity,
            &parameter_count,
            &target_clip_micros,
            &last_loss_micros,
            &average_loss_micros,
        ],
    )
    .map_err(|err| format!("insert soccer neural run metrics: {err}"))?;

    Ok(())
}

fn soccer_learning_database_url() -> Option<String> {
    [
        "SOCCER_DATABASE_URL",
        "AGENT_TASKS_RDS_DATABASE_URL",
        "RDS_DATABASE_URL",
        "DATABASE_URL",
        "PG_DATABASE_URL",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn soccer_learning_pg_should_verify_certificates(database_url: &str) -> bool {
    soccer_learning_pg_sslmode(database_url).is_some_and(|sslmode| {
        sslmode.eq_ignore_ascii_case("verify-ca") || sslmode.eq_ignore_ascii_case("verify-full")
    })
}

fn soccer_learning_pg_sslmode(database_url: &str) -> Option<&str> {
    let query = database_url.split_once('?')?.1;
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        key.eq_ignore_ascii_case("sslmode").then_some(value)
    })
}

fn state_hash(state_json: &Value) -> String {
    let raw = serde_json::to_string(state_json).unwrap_or_default();
    let mut hash = 0xcbf29ce484222325u64;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn checked_i32(value: impl TryInto<i64>) -> i32 {
    let value = value.try_into().unwrap_or(i64::MAX);
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn checked_i64(value: impl TryInto<u64>) -> i64 {
    let value = value.try_into().unwrap_or(u64::MAX);
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_neural_snapshot() -> SoccerNeuralNetworkSnapshot {
        SoccerNeuralNetworkSnapshot {
            input_dim: 2,
            output_dim: 1,
            parameter_count: 3,
            l2_norm: 0.5,
            layers: vec![crate::des::general::soccer::SoccerNeuralLayerSnapshot {
                activation: "linear".to_string(),
                weights: vec![vec![0.25, -0.25]],
                biases: vec![0.125],
            }],
        }
    }

    #[test]
    fn soccer_learning_pg_sslmode_parses_query_param() {
        assert_eq!(
            soccer_learning_pg_sslmode("postgres://u:p@host/db?sslmode=require"),
            Some("require")
        );
        assert_eq!(
            soccer_learning_pg_sslmode(
                "postgres://u:p@host/db?connect_timeout=5&sslmode=verify-full"
            ),
            Some("verify-full")
        );
        assert_eq!(soccer_learning_pg_sslmode("postgres://u:p@host/db"), None);
    }

    #[test]
    fn soccer_learning_pg_only_verifies_explicit_verify_modes() {
        assert!(!soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db"
        ));
        assert!(!soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=require"
        ));
        assert!(soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=verify-ca"
        ));
        assert!(soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=VERIFY-FULL"
        ));
    }

    #[test]
    fn policy_version_metrics_round_trip_neural_snapshot() {
        let snapshot = tiny_neural_snapshot();
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = 0.37;
        let metrics =
            soccer_policy_version_metrics(1.25, Some(&snapshot), Some(&tactical_learning))
                .expect("metrics");
        assert_eq!(metrics["fitness"], json!(1.25));
        assert_eq!(
            metrics["learningComponents"]["mdpPolicyEntries"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["pomdpStateKeyFeatures"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["mpcFormationLpWeights"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["lpAlignmentWeight"],
            json!(0.37)
        );
        assert_eq!(
            metrics["learningComponents"]["neuralNetworkSnapshot"],
            json!(true)
        );

        let decoded =
            soccer_policy_version_neural_network_from_metrics(&metrics).expect("decode snapshot");
        let decoded = decoded.expect("snapshot present");
        assert_eq!(decoded.parameter_count, snapshot.parameter_count);
        assert_eq!(decoded.layers[0].weights, snapshot.layers[0].weights);
        assert_eq!(decoded.layers[0].biases, snapshot.layers[0].biases);

        let decoded_weights =
            soccer_policy_version_tactical_learning_from_values(&json!({}), &metrics)
                .expect("decode tactical learning")
                .expect("tactical learning present");
        assert_eq!(
            decoded_weights.attack_flank_lane_weight,
            tactical_learning.attack_flank_lane_weight
        );
        assert_eq!(
            decoded_weights.defense_contract_delta_weight,
            tactical_learning.defense_contract_delta_weight
        );
        assert_eq!(
            decoded_weights.formation_lp_alignment_weight,
            tactical_learning.formation_lp_alignment_weight
        );
    }

    #[test]
    fn policy_version_tactical_learning_falls_back_to_config_lp_weight() {
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = 0.42;
        let config = json!({
            "tacticalLearning": serde_json::to_value(&tactical_learning).expect("weights json")
        });
        let decoded = soccer_policy_version_tactical_learning_from_values(&config, &json!({}))
            .expect("decode tactical learning")
            .expect("tactical learning present");

        assert_eq!(decoded.formation_lp_alignment_weight, 0.42);
    }

    #[test]
    fn postgres_rejects_non_finite_lp_alignment_weight() {
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = f64::INFINITY;

        assert!(
            validate_soccer_tactical_learning_weights_for_pg(&tactical_learning)
                .expect_err("non-finite LP alignment weight should be rejected")
                .contains("formationLpAlignmentWeight")
        );
    }

    #[test]
    fn postgres_rejects_malformed_neural_snapshot_metrics() {
        let metrics = json!({
            "neuralNetwork": {
                "inputDim": 2,
                "outputDim": 1,
                "parameterCount": 2,
                "l2Norm": 0.4,
                "layers": [{
                    "activation": "linear",
                    "weights": [[0.25]],
                    "biases": [0.1]
                }]
            }
        });

        assert!(soccer_policy_version_neural_network_from_metrics(&metrics)
            .expect_err("malformed neural snapshot should be rejected")
            .contains("expected 2"));
    }

    #[test]
    fn postgres_policy_state_json_preserves_mdp_pomdp_and_lp_bins() {
        let state_json = json!({
            "phase": "HomeBuildUp",
            "role": "Defender",
            "possessionRelative": -1,
            "ballZoneX": 3,
            "ballZoneY": 4,
            "playerTacticalCellId": 77,
            "scoreDiffBucket": 0,
            "teamBrainPressBin": 4,
            "teamBrainRiskBin": 2,
            "teamCentroidBallDistanceBin": 3,
            "teamSpreadBin": 2,
            "formationLpGuidance": true,
            "formationLpMoveBin": 3,
            "formationLpPairErrorBin": 2,
            "formationLpPressureBin": 4,
            "formationLpSpeedMatchBin": 1,
            "hasBall": false,
            "visibleBall": true,
            "shotLaneOpen": false,
            "visiblePassOptionsBin": 2,
            "ballDistanceBin": 1,
            "yardsToGoalBin": 4,
            "pressureBin": 3,
            "openSpaceBin": 2
        });
        let state: SoccerQStateKey =
            serde_json::from_value(state_json).expect("state key decodes with defaults");
        let persisted_state_json =
            serde_json::to_value(&state).expect("state key serializes for postgres jsonb");

        assert_eq!(persisted_state_json["playerTacticalCellId"], json!(77));
        assert_eq!(persisted_state_json["teamBrainPressBin"], json!(4));
        assert_eq!(
            persisted_state_json["teamCentroidBallDistanceBin"],
            json!(3)
        );
        assert_eq!(persisted_state_json["formationLpGuidance"], json!(true));
        assert_eq!(persisted_state_json["formationLpMoveBin"], json!(3));
        assert_eq!(persisted_state_json["formationLpPairErrorBin"], json!(2));
        assert_eq!(persisted_state_json["formationLpPressureBin"], json!(4));
        assert_eq!(persisted_state_json["formationLpSpeedMatchBin"], json!(1));
        let hash = state_hash(&persisted_state_json);
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn policy_version_metrics_record_branch_tip_retention() {
        let metrics = soccer_policy_version_metrics_with_retention(
            json!({ "fitness": 2.5 }),
            "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
            SOCCER_POLICY_RETENTION_BRANCH_TIP,
            true,
        );

        assert_eq!(
            metrics["postgresRetention"]["policyVersionId"],
            json!("aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa")
        );
        assert_eq!(
            metrics["postgresRetention"]["branchKey"],
            json!("bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb")
        );
        assert_eq!(
            metrics["postgresRetention"]["retentionKind"],
            json!("branch_tip")
        );
        assert_eq!(
            metrics["postgresRetention"]["fullEntriesRetained"],
            json!(true)
        );
    }

    #[test]
    fn policy_version_retention_skips_archived_and_rejected_entries() {
        assert!(soccer_policy_version_retains_full_entries("active"));
        assert!(soccer_policy_version_retains_full_entries("candidate"));
        assert!(!soccer_policy_version_retains_full_entries("archived"));
        assert!(!soccer_policy_version_retains_full_entries("rejected"));
    }

    #[test]
    fn completed_run_retention_zero_disables_pruning() {
        assert_eq!(soccer_completed_run_retention_limit(0), None);
    }

    #[test]
    fn completed_run_retention_limit_is_bounded_for_postgres() {
        assert_eq!(soccer_completed_run_retention_limit(1), Some(1));
        assert_eq!(
            soccer_completed_run_retention_limit(usize::MAX),
            Some(i64::MAX)
        );
    }

    #[test]
    fn policy_entry_batch_placeholders_preserve_uuid_casts_and_offsets() {
        let mut completed_run_sql = String::new();
        append_completed_run_header_value_tuple(&mut completed_run_sql, 1);
        completed_run_sql.push_str(", ");
        append_completed_run_header_value_tuple(&mut completed_run_sql, 23);
        assert_eq!(
            completed_run_sql,
            "($1::text::uuid, $2::text::uuid, $3::text::uuid, $4::text::uuid, $5, $6, $7, 'completed', $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22), ($23::text::uuid, $24::text::uuid, $25::text::uuid, $26::text::uuid, $27, $28, $29, 'completed', $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44)"
        );

        let mut delta_sql = String::new();
        append_run_delta_value_tuple(&mut delta_sql, 1);
        delta_sql.push_str(", ");
        append_run_delta_value_tuple(&mut delta_sql, 17);
        assert_eq!(
            delta_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16), ($17::text::uuid, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32)"
        );

        let mut action_sql = String::new();
        append_policy_entry_value_tuple(&mut action_sql, 1, false);
        action_sql.push_str(", ");
        append_policy_entry_value_tuple(&mut action_sql, 10, false);
        assert_eq!(
            action_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9::text::uuid), ($10::text::uuid, $11, $12, $13, $14, $15, $16, $17, $18::text::uuid)"
        );

        let mut target_sql = String::new();
        append_policy_entry_value_tuple(&mut target_sql, 1, true);
        target_sql.push_str(", ");
        append_policy_entry_value_tuple(&mut target_sql, 14, true);
        assert_eq!(
            target_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::text::uuid), ($14::text::uuid, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26::text::uuid)"
        );
    }

    #[test]
    fn soccer_learning_pg_batch_sizes_stay_under_postgres_parameter_limit() {
        assert!(
            SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_RUN_DELTA_INSERT_BATCH_SIZE * SOCCER_RUN_DELTA_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
    }
}
