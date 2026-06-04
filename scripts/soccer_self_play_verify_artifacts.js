#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function readRunEnv(runDir) {
  const envPath = path.join(runDir, "run.env");
  const env = {};
  if (!fs.existsSync(envPath)) {
    return env;
  }
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const idx = line.indexOf("=");
    if (idx > 0) {
      env[line.slice(0, idx)] = line.slice(idx + 1);
    }
  }
  return env;
}

function latestRunDir() {
  const root = path.join("out", "soccer-self-play");
  if (!fs.existsSync(root)) {
    return null;
  }
  let best = null;
  for (const name of fs.readdirSync(root)) {
    const runDir = path.join(root, name);
    const envPath = path.join(runDir, "run.env");
    if (!fs.existsSync(envPath)) {
      continue;
    }
    const mtimeMs = fs.statSync(envPath).mtimeMs;
    if (!best || mtimeMs > best.mtimeMs) {
      best = { runDir, mtimeMs };
    }
  }
  return best ? best.runDir : null;
}

function numeric(value, fallback = NaN) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function entryStats(entries) {
  let visits = 0;
  let nonZeroValues = 0;
  for (const entry of entries) {
    visits += Number(entry.visits || 0);
    if (Number(entry.value || 0) !== 0) {
      nonZeroValues += 1;
    }
  }
  return { count: entries.length, visits, nonZeroValues };
}

function verifyTacticalWeights(artifact, runEnv, errors) {
  const tactical = artifact.tacticalLearning || (artifact.config && artifact.config.tacticalLearning) || {};
  const checks = [
    ["attackWidthDeltaWeight", "attack_width_delta_weight", 0.5],
    ["attackFlankLaneWeight", "attack_flank_lane_weight", 0.25],
    ["defenseContractDeltaWeight", "defense_contract_delta_weight", 0.4],
    ["defenseCompactnessScoreWeight", "defense_compactness_score_weight", 0.12],
  ];
  const observed = {};
  for (const [jsonName, envName, minimum] of checks) {
    const value = numeric(tactical[jsonName]);
    observed[jsonName] = value;
    if (!Number.isFinite(value)) {
      errors.push(`missing tacticalLearning.${jsonName}`);
      continue;
    }
    const expected = numeric(runEnv[envName]);
    if (Number.isFinite(expected) && Math.abs(value - expected) > 1e-9) {
      errors.push(`expected tacticalLearning.${jsonName}=${expected}, got ${value}`);
    } else if (!Number.isFinite(expected) && value < minimum) {
      errors.push(`expected tacticalLearning.${jsonName} >= ${minimum}, got ${value}`);
    }
  }
  return observed;
}

function verifyArtifact(artifactPath, runEnv, shardName, kind) {
  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  const errors = [];
  const periodCount = numeric(artifact.config && artifact.config.periodCount);
  const durationSeconds = numeric(artifact.config && artifact.config.durationSeconds);
  const dtSeconds = numeric(artifact.config && artifact.config.dtSeconds);
  const expectedPeriodCount = numeric(runEnv.halves, 2);
  const expectedMinutes = numeric(runEnv.minutes);
  const expectedDurationSeconds = Number.isFinite(expectedMinutes)
    ? expectedMinutes * 60
    : numeric(runEnv.half_minutes, 45) * expectedPeriodCount * 60;
  const expectedDtSeconds = numeric(runEnv.dt_seconds);
  const expectedGames = numeric(runEnv.games);

  if (periodCount !== expectedPeriodCount) {
    errors.push(`expected periodCount=${expectedPeriodCount}, got ${periodCount}`);
  }
  if (Math.abs(durationSeconds - expectedDurationSeconds) > 1e-9) {
    errors.push(
      `expected durationSeconds=${expectedDurationSeconds}, got ${durationSeconds}`,
    );
  }
  if (Number.isFinite(expectedDtSeconds) && Math.abs(dtSeconds - expectedDtSeconds) > 1e-9) {
    errors.push(`expected dtSeconds=${expectedDtSeconds}, got ${dtSeconds}`);
  }
  if (!Array.isArray(artifact.episodes) || artifact.episodes.length < 1) {
    errors.push("expected at least one completed episode");
  }
  if (Number.isFinite(expectedGames) && artifact.episodes.length > expectedGames) {
    errors.push(`episodes ${artifact.episodes.length} exceeds expected games ${expectedGames}`);
  }

  const home = entryStats(Array.isArray(artifact.homeEntries) ? artifact.homeEntries : []);
  const away = entryStats(Array.isArray(artifact.awayEntries) ? artifact.awayEntries : []);
  const homeTarget = Array.isArray(artifact.homeTargetEntries) ? artifact.homeTargetEntries.length : 0;
  const awayTarget = Array.isArray(artifact.awayTargetEntries) ? artifact.awayTargetEntries.length : 0;

  if (home.count === 0 || away.count === 0) {
    errors.push("expected non-empty home and away policy entries");
  }
  if (homeTarget === 0 || awayTarget === 0) {
    errors.push("expected non-empty home and away target policy entries");
  }
  if (home.visits === 0 || away.visits === 0) {
    errors.push("expected policy entries with visits");
  }
  if (home.nonZeroValues === 0 || away.nonZeroValues === 0) {
    errors.push("expected non-zero learned policy values");
  }
  const tactical = verifyTacticalWeights(artifact, runEnv, errors);

  const summary = {
    shard: shardName,
    kind,
    artifact: artifactPath,
    episodes: artifact.episodes.length,
    periodCount,
    durationSeconds,
    dtSeconds,
    homeEntries: home.count,
    homeVisits: home.visits,
    awayEntries: away.count,
    awayVisits: away.visits,
    homeTargetEntries: homeTarget,
    awayTargetEntries: awayTarget,
    attackWidthDeltaWeight: tactical.attackWidthDeltaWeight,
    attackFlankLaneWeight: tactical.attackFlankLaneWeight,
    defenseContractDeltaWeight: tactical.defenseContractDeltaWeight,
    defenseCompactnessScoreWeight: tactical.defenseCompactnessScoreWeight,
  };

  return { summary, errors };
}

const runDir = process.argv[2] || latestRunDir();
if (!runDir) {
  console.error("usage: scripts/soccer_self_play_verify_artifacts.js <run-dir>");
  console.error("no run.env found under out/soccer-self-play");
  process.exit(2);
}
if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
  console.error(`run directory not found: ${runDir}`);
  process.exit(2);
}

const runEnv = readRunEnv(runDir);
const shardDirs = fs
  .readdirSync(runDir)
  .filter((name) => /^shard-\d+-of-\d+$/.test(name))
  .sort()
  .map((name) => path.join(runDir, name));

if (shardDirs.length === 0) {
  console.error(`no shard directories found in ${runDir}`);
  process.exit(1);
}

let verified = 0;
let failed = 0;
console.log(`run_dir=${runDir}`);

for (const shardDir of shardDirs) {
  const shardName = path.basename(shardDir);
  const artifacts = [
    ["final", path.join(shardDir, "artifact.json")],
    ["checkpoint", path.join(shardDir, "checkpoint-policy.json")],
  ].filter(([, artifactPath]) => fs.existsSync(artifactPath));

  if (artifacts.length === 0) {
    console.log(`${shardName} artifacts=missing`);
    continue;
  }

  for (const [kind, artifactPath] of artifacts) {
    const { summary, errors } = verifyArtifact(artifactPath, runEnv, shardName, kind);
    if (errors.length > 0) {
      failed += 1;
      console.error(`${shardName} ${kind} invalid: ${errors.join("; ")}`);
      continue;
    }
    verified += 1;
    console.log(
      `${summary.shard} ${summary.kind} episodes=${summary.episodes} periodCount=${summary.periodCount} durationSeconds=${summary.durationSeconds} dtSeconds=${summary.dtSeconds} homeEntries=${summary.homeEntries} homeVisits=${summary.homeVisits} awayEntries=${summary.awayEntries} awayVisits=${summary.awayVisits} homeTargetEntries=${summary.homeTargetEntries} awayTargetEntries=${summary.awayTargetEntries}`,
    );
    console.log(
      `${summary.shard} ${summary.kind} tactical attackWidthDelta=${summary.attackWidthDeltaWeight} attackFlankLane=${summary.attackFlankLaneWeight} defenseContractDelta=${summary.defenseContractDeltaWeight} defenseCompactness=${summary.defenseCompactnessScoreWeight}`,
    );
  }
}

if (failed > 0) {
  process.exit(1);
}
if (verified === 0) {
  console.error("no checkpoint or final artifacts available to verify");
  process.exit(1);
}
