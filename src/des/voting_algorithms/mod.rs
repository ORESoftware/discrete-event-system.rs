//! Voting-system models and a self-contained animated lab.
//!
//! The reusable behavior lives in Rust: scenario construction, threshold math,
//! ranked ballot transfers, pairwise contests, score aggregation, and wagered
//! magnitude accounting. The HTML shell only animates the already-computed trace.

use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

pub const VOTING_LAB_REL_PATH: &str = "voting-lab.html";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VotingLab {
    pub title: String,
    pub subtitle: String,
    pub total_vote_records: u32,
    pub requirements: Vec<Requirement>,
    pub systems: Vec<VotingSystem>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Requirement {
    pub label: String,
    pub value: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VotingSystem {
    pub id: String,
    pub label: String,
    pub short_label: String,
    pub family: String,
    pub description: String,
    pub scenario_fit: String,
    pub caveats: Vec<String>,
    pub cases: Vec<ElectionCase>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub id: String,
    pub name: String,
    pub short_name: String,
    pub color: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ElectionCase {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub scenario: String,
    pub edge_focus: Option<String>,
    pub total_vote_records: u32,
    pub counted_votes: u32,
    pub threshold_label: Option<String>,
    pub winner_id: Option<String>,
    pub winner_name: Option<String>,
    pub outcome: String,
    pub candidates: Vec<Candidate>,
    pub caveats: Vec<String>,
    pub betting_lines: Vec<BettingLine>,
    pub frames: Vec<ElectionFrame>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BettingLine {
    pub id: String,
    pub label: String,
    pub line_kind: String,
    pub market: String,
    pub line: f64,
    pub actual: f64,
    pub over_or_favorite: String,
    pub under_or_underdog: String,
    pub covered_by: String,
    pub margin: f64,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ElectionFrame {
    pub title: String,
    pub detail: String,
    pub round: usize,
    pub totals: Vec<TallyBar>,
    pub packets: Vec<VotePacket>,
    pub transfers: Vec<Transfer>,
    pub pairwise: Vec<PairwiseCell>,
    pub active: Vec<String>,
    pub eliminated: Vec<String>,
    pub exhausted: u32,
    pub threshold: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TallyBar {
    pub candidate_id: String,
    pub label: String,
    pub color: String,
    pub value: f64,
    pub percent: f64,
    pub status: String,
    pub detail: String,
    pub secondary_value: Option<f64>,
    pub secondary_label: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VotePacket {
    pub label: String,
    pub count: u32,
    pub from: Option<String>,
    pub to: Option<String>,
    pub color: String,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Transfer {
    pub from: String,
    pub to: Option<String>,
    pub count: u32,
    pub label: String,
    pub color: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairwiseCell {
    pub row: String,
    pub col: String,
    pub row_votes: u32,
    pub col_votes: u32,
    pub margin: i32,
    pub winner: Option<String>,
}

#[derive(Clone, Debug)]
struct VoteGroup {
    label: String,
    count: u32,
    ranking: Vec<String>,
    approvals: BTreeSet<String>,
    scores: BTreeMap<String, u32>,
    wagers: BTreeMap<String, i32>,
}

impl VoteGroup {
    fn new(label: &str, count: u32, ranking: &[&str]) -> Self {
        VoteGroup {
            label: label.to_string(),
            count,
            ranking: ranking.iter().map(|s| s.to_string()).collect(),
            approvals: BTreeSet::new(),
            scores: BTreeMap::new(),
            wagers: BTreeMap::new(),
        }
    }

    fn approvals(mut self, ids: &[&str]) -> Self {
        self.approvals = ids.iter().map(|s| s.to_string()).collect();
        self
    }

    fn scores(mut self, pairs: &[(&str, u32)]) -> Self {
        self.scores = pairs
            .iter()
            .map(|(id, score)| ((*id).to_string(), *score))
            .collect();
        self
    }

    fn wagers(mut self, pairs: &[(&str, i32)]) -> Self {
        self.wagers = pairs
            .iter()
            .map(|(id, wager)| ((*id).to_string(), *wager))
            .collect();
        self
    }
}

pub fn voting_lab_payload() -> VotingLab {
    let systems = vec![
        usacc_supermajority_system(),
        simple_majority_system(),
        unanimity_system(),
        plurality_system(),
        approval_system(),
        score_system(),
        ranked_choice_system(),
        borda_system(),
        condorcet_system(),
        wagered_magnitude_system(),
    ];
    VotingLab {
        title: "Voting Algorithm Lab".to_string(),
        subtitle: "Ten 1,000-record civic scenarios with animated tallies, transfers, supermajorities, and failure modes.".to_string(),
        total_vote_records: 1000,
        requirements: vec![
            Requirement {
                label: "USACC panel".to_string(),
                value: "15 seats / 12 required".to_string(),
                detail: "Model a 1,000-person eligible reviewer pool, seat 15 after conflict screening, and require an 80% supermajority for a finding.".to_string(),
            },
            Requirement {
                label: "Signal hygiene".to_string(),
                value: "verified, de-duped, auditable".to_string(),
                detail: "Public support, reviewer votes, and escrow signals are kept separate so a funding surge cannot masquerade as adjudication.".to_string(),
            },
            Requirement {
                label: "Partial observability".to_string(),
                value: "process action, not guilt oracle".to_string(),
                detail: "Several edge cases show when a vote should trigger audit, supplementation, recusal, or closure rather than a final factual conclusion.".to_string(),
            },
        ],
        systems,
    }
}

pub fn voting_lab_page_html() -> String {
    let json = serde_json::to_string_pretty(&voting_lab_payload()).unwrap_or_else(|_| "{}".into());
    let escaped = json
        .replace("</", "<\\/")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    include_str!("voting_lab.html").replace("__VOTING_LAB_PAYLOAD__", &escaped)
}

pub fn write_voting_lab_html(out_root: impl AsRef<Path>) -> io::Result<PathBuf> {
    let path = out_root.as_ref().join(VOTING_LAB_REL_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, voting_lab_page_html())?;
    Ok(path)
}

fn system(
    id: &str,
    label: &str,
    short_label: &str,
    family: &str,
    description: &str,
    scenario_fit: &str,
    caveats: &[&str],
    cases: Vec<ElectionCase>,
) -> VotingSystem {
    VotingSystem {
        id: id.to_string(),
        label: label.to_string(),
        short_label: short_label.to_string(),
        family: family.to_string(),
        description: description.to_string(),
        scenario_fit: scenario_fit.to_string(),
        caveats: caveats.iter().map(|s| s.to_string()).collect(),
        cases,
    }
}

fn election_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    candidates: Vec<Candidate>,
    counted_votes: u32,
    threshold_label: Option<String>,
    winner_id: Option<String>,
    outcome: String,
    caveats: &[&str],
    frames: Vec<ElectionFrame>,
) -> ElectionCase {
    let winner_name = winner_id.as_ref().and_then(|id| {
        candidates
            .iter()
            .find(|c| c.id == *id)
            .map(|c| c.name.clone())
    });
    ElectionCase {
        id: id.to_string(),
        label: label.to_string(),
        kind: kind.to_string(),
        scenario: scenario.to_string(),
        edge_focus: edge_focus.map(str::to_string),
        total_vote_records: 1000,
        counted_votes,
        threshold_label,
        winner_id,
        winner_name,
        outcome,
        candidates,
        caveats: caveats.iter().map(|s| s.to_string()).collect(),
        betting_lines: Vec::new(),
        frames,
    }
}

fn frame(
    title: &str,
    detail: &str,
    round: usize,
    totals: Vec<TallyBar>,
    threshold: Option<f64>,
    notes: &[&str],
) -> ElectionFrame {
    ElectionFrame {
        title: title.to_string(),
        detail: detail.to_string(),
        round,
        totals,
        packets: Vec::new(),
        transfers: Vec::new(),
        pairwise: Vec::new(),
        active: Vec::new(),
        eliminated: Vec::new(),
        exhausted: 0,
        threshold,
        notes: notes.iter().map(|s| s.to_string()).collect(),
    }
}

fn candidate(id: &str, name: &str, short: &str, color: &str) -> Candidate {
    Candidate {
        id: id.to_string(),
        name: name.to_string(),
        short_name: short.to_string(),
        color: color.to_string(),
    }
}

fn civic_candidates() -> Vec<Candidate> {
    vec![
        candidate("avery", "Avery", "Av", "#1f77b4"),
        candidate("blair", "Blair", "Bl", "#2ca02c"),
        candidate("casey", "Casey", "Ca", "#d95f02"),
        candidate("devon", "Devon", "De", "#7b61ff"),
        candidate("emery", "Emery", "Em", "#e11d48"),
    ]
}

fn proposal_candidates() -> Vec<Candidate> {
    vec![
        candidate("yes", "Yes", "Yes", "#0f9f6e"),
        candidate("no", "No", "No", "#d43f3a"),
    ]
}

fn unanimity_candidates() -> Vec<Candidate> {
    vec![
        candidate("consent", "Consent", "OK", "#0f9f6e"),
        candidate("block", "Block", "Blk", "#d43f3a"),
        candidate("stand_aside", "Stand aside", "SA", "#f59f00"),
    ]
}

fn court_candidates() -> Vec<Candidate> {
    vec![
        candidate("finding", "Finding supported", "12+", "#0f766e"),
        candidate("not_proven", "Not proven", "NP", "#dc2626"),
        candidate("reserve", "Eligible reserve", "Res", "#64748b"),
        candidate("recused", "Recused", "Rec", "#f59e0b"),
    ]
}

fn candidate_color(candidates: &[Candidate], id: &str) -> String {
    candidates
        .iter()
        .find(|c| c.id == id)
        .map(|c| c.color.clone())
        .unwrap_or_else(|| "#64748b".to_string())
}

fn candidate_label(candidates: &[Candidate], id: &str) -> String {
    candidates
        .iter()
        .find(|c| c.id == id)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| id.to_string())
}

fn total_count(groups: &[VoteGroup]) -> u32 {
    groups.iter().map(|g| g.count).sum()
}

fn assert_thousand(groups: &[VoteGroup]) {
    assert_eq!(total_count(groups), 1000, "scenario must model 1,000 vote records");
}

fn first_choice_tally(groups: &[VoteGroup]) -> BTreeMap<String, u32> {
    let mut totals = BTreeMap::new();
    for group in groups {
        if let Some(id) = group.ranking.first() {
            *totals.entry(id.clone()).or_insert(0) += group.count;
        }
    }
    totals
}

fn approval_tally(groups: &[VoteGroup]) -> BTreeMap<String, u32> {
    let mut totals = BTreeMap::new();
    for group in groups {
        for id in &group.approvals {
            *totals.entry(id.clone()).or_insert(0) += group.count;
        }
    }
    totals
}

fn score_tally(groups: &[VoteGroup]) -> BTreeMap<String, u32> {
    let mut totals = BTreeMap::new();
    for group in groups {
        for (id, score) in &group.scores {
            *totals.entry(id.clone()).or_insert(0) += group.count * *score;
        }
    }
    totals
}

fn wager_tally(groups: &[VoteGroup]) -> BTreeMap<String, i32> {
    let mut totals = BTreeMap::new();
    for group in groups {
        for (id, wager) in &group.wagers {
            *totals.entry(id.clone()).or_insert(0) += group.count as i32 * *wager;
        }
    }
    totals
}

fn sorted_winner(totals: &BTreeMap<String, u32>) -> Option<String> {
    totals
        .iter()
        .max_by(|(aid, av), (bid, bv)| match av.cmp(bv) {
            Ordering::Equal => bid.cmp(aid),
            ord => ord,
        })
        .map(|(id, _)| id.clone())
}

fn sorted_i32_winner(totals: &BTreeMap<String, i32>) -> Option<String> {
    totals
        .iter()
        .max_by(|(aid, av), (bid, bv)| match av.cmp(bv) {
            Ordering::Equal => bid.cmp(aid),
            ord => ord,
        })
        .map(|(id, _)| id.clone())
}

fn bars_from_u32(
    candidates: &[Candidate],
    totals: &BTreeMap<String, u32>,
    denominator: f64,
    winner_id: Option<&str>,
    eliminated: &BTreeSet<String>,
    units: &str,
) -> Vec<TallyBar> {
    let mut bars: Vec<TallyBar> = candidates
        .iter()
        .filter_map(|c| {
            let value = *totals.get(&c.id).unwrap_or(&0) as f64;
            if value == 0.0 && !totals.contains_key(&c.id) {
                return None;
            }
            let status = if Some(c.id.as_str()) == winner_id {
                "winner"
            } else if eliminated.contains(&c.id) {
                "eliminated"
            } else {
                "active"
            };
            Some(TallyBar {
                candidate_id: c.id.clone(),
                label: c.name.clone(),
                color: c.color.clone(),
                value,
                percent: if denominator > 0.0 {
                    100.0 * value / denominator
                } else {
                    0.0
                },
                status: status.to_string(),
                detail: format!("{} {}", fmt_num(value), units),
                secondary_value: None,
                secondary_label: None,
            })
        })
        .collect();
    bars.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    bars
}

fn bars_from_i32(
    candidates: &[Candidate],
    totals: &BTreeMap<String, i32>,
    denominator: f64,
    winner_id: Option<&str>,
    units: &str,
) -> Vec<TallyBar> {
    let mut bars: Vec<TallyBar> = candidates
        .iter()
        .filter_map(|c| {
            let value = *totals.get(&c.id).unwrap_or(&0) as f64;
            if value == 0.0 && !totals.contains_key(&c.id) {
                return None;
            }
            Some(TallyBar {
                candidate_id: c.id.clone(),
                label: c.name.clone(),
                color: c.color.clone(),
                value,
                percent: if denominator > 0.0 {
                    100.0 * value.abs() / denominator
                } else {
                    0.0
                },
                status: if Some(c.id.as_str()) == winner_id {
                    "winner".to_string()
                } else {
                    "active".to_string()
                },
                detail: format!("{} {}", fmt_num(value), units),
                secondary_value: None,
                secondary_label: None,
            })
        })
        .collect();
    bars.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    bars
}

fn packets_from_first_choices(candidates: &[Candidate], groups: &[VoteGroup]) -> Vec<VotePacket> {
    groups
        .iter()
        .map(|g| {
            let to = g.ranking.first().cloned();
            VotePacket {
                label: g.label.clone(),
                count: g.count,
                from: None,
                to: to.clone(),
                color: to
                    .as_deref()
                    .map(|id| candidate_color(candidates, id))
                    .unwrap_or_else(|| "#64748b".to_string()),
                note: to
                    .as_deref()
                    .map(|id| format!("first choice: {}", candidate_label(candidates, id)))
                    .unwrap_or_else(|| "no continuing choice".to_string()),
            }
        })
        .collect()
}

fn fmt_num(value: f64) -> String {
    if value.is_finite() && (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.2}")
    }
}

fn top_two(totals: &BTreeMap<String, u32>) -> Vec<String> {
    let mut rows: Vec<(String, u32)> = totals.iter().map(|(k, v)| (k.clone(), *v)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.into_iter().take(2).map(|(id, _)| id).collect()
}

fn usacc_supermajority_system() -> VotingSystem {
    system(
        "usacc-supermajority",
        "USACC 12-of-15 Supermajority Panel",
        "USACC 12/15",
        "court / supermajority",
        "A 1,000-person eligible pool is screened, a 15-person panel is seated, and a finding needs 12 votes.",
        "Anti-corruption admission or trial gates where legitimacy depends on a high bar, random rotation, conflicts review, and sealed votes.",
        &[
            "A supermajority vote is a gate, not an automated finding of truth.",
            "Pool eligibility, conflict screening, panel replacement, and audit logs matter as much as the threshold.",
            "Public support and escrow signals should route process actions; they should not count as panel votes.",
        ],
        vec![
            usacc_panel_case(
                "clean-12-of-15",
                "Clean 12-of-15 finding",
                "normal",
                "Randomized panel reaches the 80% finding threshold.",
                None,
                12,
                3,
                0,
                985,
                &[
                    "The threshold is exactly met: 12 finding-supported votes out of 15 seated panelists.",
                    "The other 985 eligible reviewers remain in reserve for rotation and backfill.",
                ],
            ),
            usacc_panel_case(
                "holdout-11-of-15",
                "Holdout edge",
                "edge",
                "A strong 11-of-15 majority still fails the 12-vote rule.",
                Some("One extra holdout blocks a finding even though support is 73.3% of the panel."),
                11,
                4,
                0,
                985,
                &[
                    "This is intentional for trial-like gates: one or two dissenters do not block, but four do.",
                    "The UI keeps this distinct from unanimity so the design can tune false-positive and false-negative risk.",
                ],
            ),
            usacc_panel_case(
                "recusal-refill",
                "Recusal + refill edge",
                "edge",
                "Two conflicted panelists are removed and replaced from the reserve pool before the sealed vote is counted.",
                Some("Conflict screening changes who counts without changing the 12-of-15 threshold."),
                12,
                3,
                2,
                983,
                &[
                    "Recusals should be visible in the audit trail, but recused votes never enter the verdict denominator.",
                    "Replacement draws need their own randomness and conflict checks to avoid panel shopping.",
                ],
            ),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn usacc_panel_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    finding: u32,
    not_proven: u32,
    recused: u32,
    reserve: u32,
    caveats: &[&str],
) -> ElectionCase {
    let candidates = court_candidates();
    let mut pool = BTreeMap::new();
    pool.insert("reserve".to_string(), 1000);
    let mut seated = BTreeMap::new();
    seated.insert("finding".to_string(), finding);
    seated.insert("not_proven".to_string(), not_proven);
    if recused > 0 {
        seated.insert("recused".to_string(), recused);
        seated.insert("reserve".to_string(), reserve);
    } else {
        seated.insert("reserve".to_string(), reserve);
    }

    let mut final_votes = BTreeMap::new();
    final_votes.insert("finding".to_string(), finding);
    final_votes.insert("not_proven".to_string(), not_proven);

    let winner = if finding >= 12 {
        Some("finding".to_string())
    } else {
        Some("not_proven".to_string())
    };
    let mut frames = vec![
        frame(
            "Eligible reviewer pool",
            "All 1,000 records start as eligible reviewers before random draw, conflicts review, and stage separation.",
            0,
            bars_from_u32(&candidates, &pool, 1000.0, None, &BTreeSet::new(), "eligible records"),
            None,
            &["The pool is intentionally much larger than the seated panel to reduce capture and repeat-player effects."],
        ),
        frame(
            if recused > 0 { "Draw, recuse, and refill" } else { "Random 15-person panel seated" },
            if recused > 0 {
                "The first draw exposes conflicts; recused panelists are replaced from reserve before voting."
            } else {
                "Fifteen panelists are seated and the remaining records stay outside the counted vote."
            },
            1,
            bars_from_u32(&candidates, &seated, 1000.0, None, &BTreeSet::new(), "records"),
            None,
            if recused > 0 {
                &["Recusal is a process event, not a vote outcome.", "Backfill should be auditable and randomized inside the qualified pool."]
            } else {
                &["The denominator for the finding is the seated panel, not the full pool."]
            },
        ),
        frame(
            "Sealed panel vote",
            "The final finding threshold is 12 of 15. Falling below it produces a not-proven result.",
            2,
            bars_from_u32(
                &candidates,
                &final_votes,
                15.0,
                winner.as_deref(),
                &BTreeSet::new(),
                "panel votes",
            ),
            Some(12.0),
            caveats,
        ),
    ];
    frames[0].packets = vec![VotePacket {
        label: "eligible pool".to_string(),
        count: 1000,
        from: None,
        to: Some("reserve".to_string()),
        color: candidate_color(&candidates, "reserve"),
        note: "qualified records before random panel draw".to_string(),
    }];
    frames[1].packets = vec![
        VotePacket {
            label: "seated panel".to_string(),
            count: 15,
            from: Some("reserve".to_string()),
            to: Some("finding".to_string()),
            color: candidate_color(&candidates, "finding"),
            note: "15 counted seats after screening".to_string(),
        },
        VotePacket {
            label: "not seated".to_string(),
            count: reserve,
            from: Some("reserve".to_string()),
            to: Some("reserve".to_string()),
            color: candidate_color(&candidates, "reserve"),
            note: "eligible reserve remains available for later rotation".to_string(),
        },
    ];
    if recused > 0 {
        frames[1].transfers = vec![Transfer {
            from: "recused".to_string(),
            to: Some("reserve".to_string()),
            count: recused,
            label: "conflict backfill".to_string(),
            color: candidate_color(&candidates, "recused"),
        }];
    }
    let mut case = election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        15,
        Some("12 of 15 required".to_string()),
        winner.clone(),
        if winner.as_deref() == Some("finding") {
            format!("Finding supported with {finding}/15 panel votes.")
        } else {
            format!("Not proven: {finding}/15 is below the 12-vote threshold.")
        },
        caveats,
        frames,
    )
}

fn simple_majority_system() -> VotingSystem {
    system(
        "simple-majority",
        "Simple Majority",
        "Majority",
        "binary threshold",
        "Each of the 1,000 vote records chooses yes or no. More than half wins; an exact tie has no winner.",
        "Referenda, governance proposals, and low-stakes binary decisions where a bare majority is acceptable.",
        &[
            "The exact threshold must be explicit: greater than half, at least half, or half plus one.",
            "Abstentions, spoiled ballots, and recount rules should be specified before voting starts.",
        ],
        vec![
            majority_case(
                "proposal-passes",
                "Proposal passes",
                "normal",
                "A policy proposal clears the half-plus-one threshold.",
                None,
                534,
                466,
                &["A bare majority is decisive here because the model has only two valid choices."],
            ),
            majority_case(
                "exact-tie",
                "Exact tie edge",
                "edge",
                "Five hundred yes and five hundred no records create a deadlock.",
                Some("Tie-breaking must be external to the tally rule."),
                500,
                500,
                &[
                    "A deterministic tie-breaker can be legitimate for scheduling, but not for high-stakes factual findings.",
                    "For civic decisions, a tie usually means the status quo remains or the vote repeats.",
                ],
            ),
        ],
    )
}

fn majority_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    yes: u32,
    no: u32,
    caveats: &[&str],
) -> ElectionCase {
    let candidates = proposal_candidates();
    let mut totals = BTreeMap::new();
    totals.insert("yes".to_string(), yes);
    totals.insert("no".to_string(), no);
    let winner = if yes > no {
        Some("yes".to_string())
    } else if no > yes {
        Some("no".to_string())
    } else {
        None
    };
    let mut frames = vec![
        frame(
            "Collect binary vote records",
            "Each record contributes exactly one yes/no vote.",
            0,
            bars_from_u32(&candidates, &totals, 1000.0, None, &BTreeSet::new(), "votes"),
            Some(501.0),
            &["The reference denominator is 1,000 valid vote records."],
        ),
        frame(
            "Apply majority threshold",
            "The winning side must exceed half of the valid votes.",
            1,
            bars_from_u32(
                &candidates,
                &totals,
                1000.0,
                winner.as_deref(),
                &BTreeSet::new(),
                "votes",
            ),
            Some(501.0),
            caveats,
        ),
    ];
    frames[0].packets = vec![
        VotePacket {
            label: "yes records".to_string(),
            count: yes,
            from: None,
            to: Some("yes".to_string()),
            color: candidate_color(&candidates, "yes"),
            note: "proposal support".to_string(),
        },
        VotePacket {
            label: "no records".to_string(),
            count: no,
            from: None,
            to: Some("no".to_string()),
            color: candidate_color(&candidates, "no"),
            note: "proposal opposition".to_string(),
        },
    ];
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        Some("501 of 1000 required".to_string()),
        winner.clone(),
        match winner.as_deref() {
            Some("yes") => format!("Proposal passes {yes}-{no}."),
            Some("no") => format!("Proposal fails {no}-{yes}."),
            _ => "No winner: the vote is tied 500-500.".to_string(),
        },
        caveats,
        frames,
    )
}

fn unanimity_system() -> VotingSystem {
    system(
        "unanimity",
        "Unanimous Consent",
        "Unanimity",
        "consensus / veto",
        "A proposal carries only when every counted record consents or, under the chosen rule, stands aside without blocking.",
        "Tiny committees, consent agendas, and settings where protecting each participant from being overridden is more important than throughput.",
        &[
            "The system must define whether abstentions, silence, or stand-asides count as consent.",
            "One bad-faith veto can create denial-of-service, so scope and good-standing rules matter.",
        ],
        vec![
            unanimity_case(
                "all-consent",
                "All consent",
                "normal",
                "Every one of the 1,000 records consents.",
                None,
                1000,
                0,
                0,
                &["This is the clean case: no block, no ambiguity, proposal carries."],
            ),
            unanimity_case(
                "single-block",
                "Single blocker edge",
                "edge",
                "One blocker is enough to stop the proposal.",
                Some("A single veto has full stopping power."),
                999,
                1,
                0,
                &[
                    "This protects minorities, but it can also let one actor halt the institution.",
                    "A process rule should distinguish principled blocks from irrelevant or abusive blocks.",
                ],
            ),
            unanimity_case(
                "stand-aside",
                "Stand-aside edge",
                "edge",
                "Forty records stand aside; none block, so the proposal carries under a consent-with-stand-asides rule.",
                Some("Stand-asides lower enthusiasm without vetoing."),
                960,
                0,
                40,
                &[
                    "Stand-asides should be logged because a proposal with many stand-asides may need follow-up.",
                    "This variant avoids making abstention equivalent to veto.",
                ],
            ),
        ],
    )
}

fn unanimity_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    consent: u32,
    block: u32,
    stand_aside: u32,
    caveats: &[&str],
) -> ElectionCase {
    let candidates = unanimity_candidates();
    let mut totals = BTreeMap::new();
    totals.insert("consent".to_string(), consent);
    totals.insert("block".to_string(), block);
    totals.insert("stand_aside".to_string(), stand_aside);
    let winner = if block == 0 {
        Some("consent".to_string())
    } else {
        Some("block".to_string())
    };
    let mut frames = vec![
        frame(
            "Check every record",
            "Unanimity is a veto scan: the first valid block changes the result.",
            0,
            bars_from_u32(&candidates, &totals, 1000.0, None, &BTreeSet::new(), "records"),
            Some(0.0),
            &["The threshold line here means zero blockers, not a minimum support count."],
        ),
        frame(
            "Apply no-block rule",
            "The proposal carries only when the block count is zero.",
            1,
            bars_from_u32(
                &candidates,
                &totals,
                1000.0,
                winner.as_deref(),
                &BTreeSet::new(),
                "records",
            ),
            Some(0.0),
            caveats,
        ),
    ];
    frames[0].packets = vec![
        VotePacket {
            label: "consent".to_string(),
            count: consent,
            from: None,
            to: Some("consent".to_string()),
            color: candidate_color(&candidates, "consent"),
            note: "affirmative consent records".to_string(),
        },
        VotePacket {
            label: "block".to_string(),
            count: block,
            from: None,
            to: Some("block".to_string()),
            color: candidate_color(&candidates, "block"),
            note: "blocking records".to_string(),
        },
        VotePacket {
            label: "stand aside".to_string(),
            count: stand_aside,
            from: None,
            to: Some("stand_aside".to_string()),
            color: candidate_color(&candidates, "stand_aside"),
            note: "non-blocking abstentions".to_string(),
        },
    ];
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        Some("0 blockers required".to_string()),
        winner.clone(),
        if block == 0 {
            format!("Proposal carries: {consent} consent, {stand_aside} stand aside, 0 blocks.")
        } else {
            format!("Proposal blocked by {block} record(s).")
        },
        caveats,
        frames,
    )
}

fn plurality_system() -> VotingSystem {
    system(
        "plurality",
        "Plurality / First Past the Post",
        "Plurality",
        "single-mark",
        "Only the first preference counts. The largest pile wins, even without a majority.",
        "Fast, familiar elections where simplicity matters more than measuring fallback preference.",
        &[
            "A plurality winner can be opposed by most voters.",
            "Similar candidates can split a coalition and change the winner.",
            "Ties and recount triggers need external rules.",
        ],
        vec![
            plurality_case(
                "city-council",
                "Four-way council race",
                "normal",
                "Avery wins the largest first-choice pile with 34.2% of the vote.",
                None,
                vec![
                    VoteGroup::new("Avery first", 342, &["avery", "blair", "casey", "devon"]),
                    VoteGroup::new("Blair first", 318, &["blair", "casey", "avery", "devon"]),
                    VoteGroup::new("Casey first", 190, &["casey", "blair", "avery", "devon"]),
                    VoteGroup::new("Devon first", 150, &["devon", "casey", "blair", "avery"]),
                ],
                &["The winner has the biggest pile, but not majority consent."],
            ),
            plurality_case(
                "spoiler-split",
                "Spoiler split edge",
                "edge",
                "Avery wins while Blair would beat Avery head-to-head because Casey splits the anti-Avery coalition.",
                Some("Plurality sees only first choices and ignores the majority's fallback preference."),
                vec![
                    VoteGroup::new("Avery bloc", 360, &["avery", "blair", "casey"]),
                    VoteGroup::new("Blair bloc", 330, &["blair", "casey", "avery"]),
                    VoteGroup::new("Casey bloc", 310, &["casey", "blair", "avery"]),
                ],
                &[
                    "Blair is preferred over Avery by the Blair and Casey blocs: 640 to 360.",
                    "This is the classic spoiler/split-vote failure mode.",
                ],
            ),
        ],
    )
}

fn plurality_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let candidates = civic_candidates();
    let totals = first_choice_tally(&groups);
    let winner = sorted_winner(&totals);
    let mut frames = vec![
        frame(
            "Sort first choices",
            "Each ballot goes to its first listed active candidate.",
            0,
            bars_from_u32(&candidates, &totals, 1000.0, None, &BTreeSet::new(), "first-choice votes"),
            None,
            &["Only the top name on each record is used."],
        ),
        frame(
            "Largest pile wins",
            "No majority threshold is applied.",
            1,
            bars_from_u32(
                &candidates,
                &totals,
                1000.0,
                winner.as_deref(),
                &BTreeSet::new(),
                "first-choice votes",
            ),
            None,
            caveats,
        ),
    ];
    frames[0].packets = packets_from_first_choices(&candidates, &groups);
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        None,
        winner.clone(),
        winner
            .as_deref()
            .map(|id| format!("{} wins by plurality.", candidate_label(&candidates, id)))
            .unwrap_or_else(|| "No plurality winner.".to_string()),
        caveats,
        frames,
    )
}

fn approval_system() -> VotingSystem {
    system(
        "approval",
        "Approval Voting",
        "Approval",
        "multi-mark",
        "Each record can approve any number of acceptable candidates. Highest approval count wins.",
        "Screening, committee selection, and broad-acceptability elections where voters can name all tolerable outcomes.",
        &[
            "The ballot must explain whether approving more candidates can harm a favorite.",
            "Bullet voting can collapse approval into plurality-like behavior.",
            "A tie can be common when a consensus slate is strong.",
        ],
        vec![
            approval_case(
                "broad-acceptability",
                "Broad acceptability",
                "normal",
                "Blair wins because every bloc except the pure Devon bloc treats Blair as acceptable.",
                None,
                vec![
                    VoteGroup::new("Avery-aligned", 340, &["avery"]).approvals(&["avery", "blair"]),
                    VoteGroup::new("Blair-aligned", 260, &["blair"]).approvals(&["blair", "avery", "casey"]),
                    VoteGroup::new("Casey-aligned", 220, &["casey"]).approvals(&["casey", "blair"]),
                    VoteGroup::new("Devon-aligned", 180, &["devon"]).approvals(&["devon", "casey"]),
                ],
                &["Approval can reward the candidate most groups can live with."],
            ),
            approval_case(
                "bullet-voting",
                "Bullet voting edge",
                "edge",
                "The same first-choice blocs approve only their favorite, making the result behave like plurality.",
                Some("Strategic bullet voting erases the extra information approval was meant to collect."),
                vec![
                    VoteGroup::new("Avery bullets", 340, &["avery"]).approvals(&["avery"]),
                    VoteGroup::new("Blair bullets", 260, &["blair"]).approvals(&["blair"]),
                    VoteGroup::new("Casey bullets", 220, &["casey"]).approvals(&["casey"]),
                    VoteGroup::new("Devon bullets", 180, &["devon"]).approvals(&["devon"]),
                ],
                &[
                    "The interface should make strategic incentives visible before users rely on the tally.",
                    "In high-stakes settings, approval may need education, audits, and clear tie rules.",
                ],
            ),
        ],
    )
}

fn approval_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let candidates = civic_candidates();
    let totals = approval_tally(&groups);
    let winner = sorted_winner(&totals);
    let mut frames = vec![
        frame(
            "Collect all approvals",
            "A single record can add one approval to several candidates.",
            0,
            bars_from_u32(&candidates, &totals, 1000.0, None, &BTreeSet::new(), "approvals"),
            None,
            &["Approval totals can exceed 1,000 because each voter can approve multiple candidates."],
        ),
        frame(
            "Highest approval count wins",
            "The most broadly acceptable candidate rises to the top.",
            1,
            bars_from_u32(
                &candidates,
                &totals,
                1000.0,
                winner.as_deref(),
                &BTreeSet::new(),
                "approvals",
            ),
            None,
            caveats,
        ),
    ];
    frames[0].packets = groups
        .iter()
        .map(|g| VotePacket {
            label: g.label.clone(),
            count: g.count,
            from: None,
            to: g.approvals.iter().next().cloned(),
            color: g
                .approvals
                .iter()
                .next()
                .map(|id| candidate_color(&candidates, id))
                .unwrap_or_else(|| "#64748b".to_string()),
            note: format!("approves {} candidate(s)", g.approvals.len()),
        })
        .collect();
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        None,
        winner.clone(),
        winner
            .as_deref()
            .map(|id| format!("{} wins on approval breadth.", candidate_label(&candidates, id)))
            .unwrap_or_else(|| "No approval winner.".to_string()),
        caveats,
        frames,
    )
}

fn score_system() -> VotingSystem {
    system(
        "score",
        "Score / Range Voting",
        "Score",
        "cardinal rating",
        "Every record rates each candidate from 0 to 5. The highest average rating wins.",
        "Product councils, participatory budgeting screens, and preference surveys where magnitude matters.",
        &[
            "Scale interpretation is fragile: one voter's 4 may mean another voter's 2.",
            "Strategic max/min scoring can dominate sincere middle scores.",
            "Missing ratings need a rule: zero, neutral, or invalid ballot.",
        ],
        vec![
            score_case(
                "consensus-score",
                "Consensus rating",
                "normal",
                "Blair wins by high average support across blocs, even when not everyone ranks Blair first.",
                None,
                vec![
                    VoteGroup::new("Avery fans", 320, &["avery", "blair", "casey", "devon"])
                        .scores(&[("avery", 5), ("blair", 3), ("casey", 1), ("devon", 0)]),
                    VoteGroup::new("Blair fans", 260, &["blair", "casey", "avery", "devon"])
                        .scores(&[("avery", 2), ("blair", 5), ("casey", 3), ("devon", 1)]),
                    VoteGroup::new("Casey fans", 220, &["casey", "blair", "avery", "devon"])
                        .scores(&[("avery", 1), ("blair", 4), ("casey", 5), ("devon", 1)]),
                    VoteGroup::new("Devon fans", 200, &["devon", "blair", "casey", "avery"])
                        .scores(&[("avery", 0), ("blair", 4), ("casey", 2), ("devon", 5)]),
                ],
                &["Score voting can surface a consensus option without requiring ranked transfers."],
            ),
            score_case(
                "max-min-strategy",
                "Max/min edge",
                "edge",
                "A polarized bloc uses 5 for its favorite and 0 for everyone else, compressing the middle.",
                Some("Strategic extremes can overpower sincere moderate ratings."),
                vec![
                    VoteGroup::new("Avery maximizers", 380, &["avery"]).scores(&[
                        ("avery", 5),
                        ("blair", 0),
                        ("casey", 0),
                        ("devon", 0),
                    ]),
                    VoteGroup::new("Consensus raters", 320, &["blair"]).scores(&[
                        ("avery", 2),
                        ("blair", 4),
                        ("casey", 3),
                        ("devon", 2),
                    ]),
                    VoteGroup::new("Casey raters", 180, &["casey"]).scores(&[
                        ("avery", 1),
                        ("blair", 4),
                        ("casey", 5),
                        ("devon", 1),
                    ]),
                    VoteGroup::new("Devon raters", 120, &["devon"]).scores(&[
                        ("avery", 1),
                        ("blair", 3),
                        ("casey", 2),
                        ("devon", 5),
                    ]),
                ],
                &[
                    "Score systems need a policy for normalization, missing values, and obvious ballot compression.",
                    "The tally is mathematically simple; the human scale semantics are the hard part.",
                ],
            ),
        ],
    )
}

fn score_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let candidates = civic_candidates();
    let totals = score_tally(&groups);
    let winner = sorted_winner(&totals);
    let mut bars = score_bars(&candidates, &totals, winner.as_deref());
    let mut preview = bars.clone();
    for bar in &mut preview {
        bar.status = "active".to_string();
    }
    let frames = vec![
        frame(
            "Aggregate 0-5 ratings",
            "Scores add across all 1,000 records; the display shows average rating per candidate.",
            0,
            preview,
            Some(5.0),
            &["The maximum possible total is 5,000 points per candidate."],
        ),
        frame(
            "Highest average wins",
            "The top average rating becomes the score-voting winner.",
            1,
            bars,
            Some(5.0),
            caveats,
        ),
    ];
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        Some("highest average score".to_string()),
        winner.clone(),
        winner
            .as_deref()
            .map(|id| format!("{} wins by average score.", candidate_label(&candidates, id)))
            .unwrap_or_else(|| "No score winner.".to_string()),
        caveats,
        frames,
    )
}

fn score_bars(
    candidates: &[Candidate],
    totals: &BTreeMap<String, u32>,
    winner_id: Option<&str>,
) -> Vec<TallyBar> {
    let mut bars: Vec<TallyBar> = candidates
        .iter()
        .filter_map(|c| {
            let total = *totals.get(&c.id).unwrap_or(&0);
            if total == 0 && !totals.contains_key(&c.id) {
                return None;
            }
            let avg = total as f64 / 1000.0;
            Some(TallyBar {
                candidate_id: c.id.clone(),
                label: c.name.clone(),
                color: c.color.clone(),
                value: avg,
                percent: 100.0 * avg / 5.0,
                status: if Some(c.id.as_str()) == winner_id {
                    "winner".to_string()
                } else {
                    "active".to_string()
                },
                detail: format!("{avg:.2}/5 avg ({total} pts)"),
                secondary_value: Some(total as f64),
                secondary_label: Some("raw points".to_string()),
            })
        })
        .collect();
    bars.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    bars
}

fn ranked_choice_system() -> VotingSystem {
    system(
        "ranked-choice",
        "Ranked-Choice / Instant Runoff",
        "RCV",
        "ranked elimination",
        "Ballots rank candidates. If nobody has a majority of continuing ballots, the lowest active candidate is eliminated and their ballots transfer.",
        "Single-winner elections where fallback preferences matter and the winner should clear a majority of continuing ballots.",
        &[
            "Incomplete rankings can exhaust, lowering the continuing-vote denominator.",
            "Ties for last place need a pre-published tie-breaker.",
            "The tabulation should never mutate the original ballot records; transfers are derived views.",
        ],
        vec![
            ranked_choice_case(
                "transfer-majority",
                "Transfer majority",
                "normal",
                "No candidate starts with a majority; Devon and Blair are eliminated and Casey wins through transfers.",
                None,
                vec![
                    VoteGroup::new("Avery first", 340, &["avery", "blair", "casey", "devon"]),
                    VoteGroup::new("Blair first", 260, &["blair", "casey", "avery", "devon"]),
                    VoteGroup::new("Casey first", 230, &["casey", "blair", "avery", "devon"]),
                    VoteGroup::new("Devon first", 170, &["devon", "casey", "blair", "avery"]),
                ],
                &["The winner emerges from ranked fallback support, not first-choice plurality alone."],
            ),
            ranked_choice_case(
                "exhausted-ballots",
                "Exhausted ballot edge",
                "edge",
                "Some records rank only one candidate; when that candidate is eliminated, the records leave the continuing denominator.",
                Some("Exhausted ballots can lower the majority threshold in later rounds."),
                vec![
                    VoteGroup::new("Avery-only", 410, &["avery"]),
                    VoteGroup::new("Blair-to-Casey", 300, &["blair", "casey"]),
                    VoteGroup::new("Casey-to-Blair", 200, &["casey", "blair"]),
                    VoteGroup::new("Devon-only", 90, &["devon"]),
                ],
                &[
                    "The page shows exhausted records separately so the lower denominator is not mistaken for disappearing voters.",
                    "Ballot instructions should encourage full rankings when exhaustion risk matters.",
                ],
            ),
            ranked_choice_case(
                "last-place-tie",
                "Last-place tie edge",
                "edge",
                "Casey and Devon tie for last in round one; the deterministic tie-breaker decides who leaves first.",
                Some("A last-place tie can steer the transfer path."),
                vec![
                    VoteGroup::new("Avery first", 350, &["avery", "blair", "casey", "devon"]),
                    VoteGroup::new("Blair first", 250, &["blair", "avery", "casey", "devon"]),
                    VoteGroup::new("Casey first", 200, &["casey", "blair", "avery", "devon"]),
                    VoteGroup::new("Devon first", 200, &["devon", "casey", "blair", "avery"]),
                ],
                &[
                    "The implementation uses candidate id order as a deterministic demo tie-breaker.",
                    "A production election would need a legally approved tie procedure.",
                ],
            ),
        ],
    )
}

fn ranked_choice_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let candidates = civic_candidates();
    let (winner, mut frames) = irv_trace(&candidates, &groups);
    if let Some(last) = frames.last_mut() {
        last.notes.extend(caveats.iter().map(|s| s.to_string()));
    }
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        Some("majority of continuing ballots".to_string()),
        winner.clone(),
        winner
            .as_deref()
            .map(|id| format!("{} wins after ranked transfers.", candidate_label(&civic_candidates(), id)))
            .unwrap_or_else(|| "No ranked-choice winner.".to_string()),
        caveats,
        frames,
    )
}

fn irv_trace(candidates: &[Candidate], groups: &[VoteGroup]) -> (Option<String>, Vec<ElectionFrame>) {
    let mut active: BTreeSet<String> = candidate_ids_from_groups(groups).into_iter().collect();
    let mut eliminated: BTreeSet<String> = BTreeSet::new();
    let mut frames = Vec::new();
    let mut round = 1usize;

    loop {
        let (totals, exhausted) = active_tally(groups, &active);
        let continuing: u32 = totals.values().sum();
        let threshold = continuing as f64 / 2.0;
        let winner = totals
            .iter()
            .find(|(_, count)| **count as f64 > threshold)
            .map(|(id, _)| id.clone());
        let mut bars = bars_from_u32(
            candidates,
            &totals,
            continuing.max(1) as f64,
            winner.as_deref(),
            &eliminated,
            "continuing votes",
        );
        for bar in &mut bars {
            bar.secondary_value = Some(*totals.get(&bar.candidate_id).unwrap_or(&0) as f64);
            bar.secondary_label = Some("round tally".to_string());
        }
        let mut f = frame(
            &format!("Round {round}: count continuing ballots"),
            &format!(
                "{continuing} continuing records, {exhausted} exhausted; majority requires more than {:.0}.",
                threshold.floor()
            ),
            round,
            bars,
            Some(threshold + 0.0001),
            &[],
        );
        f.active = active.iter().cloned().collect();
        f.eliminated = eliminated.iter().cloned().collect();
        f.exhausted = exhausted;
        if round == 1 {
            f.packets = packets_from_first_choices(candidates, groups);
        }
        if let Some(id) = winner {
            f.notes.push(format!(
                "{} has a majority of continuing ballots.",
                candidate_label(candidates, &id)
            ));
            frames.push(f);
            return (Some(id), frames);
        }

        let min_votes = active
            .iter()
            .map(|id| *totals.get(id).unwrap_or(&0))
            .min()
            .unwrap_or(0);
        let tied: Vec<String> = active
            .iter()
            .filter(|id| *totals.get(*id).unwrap_or(&0) == min_votes)
            .cloned()
            .collect();
        let eliminated_id = tied.first().cloned();
        if let Some(id) = eliminated_id {
            if tied.len() > 1 {
                f.notes.push(format!(
                    "Last-place tie among {}; demo tie-break eliminates {}.",
                    tied.iter()
                        .map(|x| candidate_label(candidates, x))
                        .collect::<Vec<_>>()
                        .join(", "),
                    candidate_label(candidates, &id)
                ));
            } else {
                f.notes.push(format!("{} is lowest and is eliminated.", candidate_label(candidates, &id)));
            }
            let mut next_active = active.clone();
            next_active.remove(&id);
            f.transfers = transfer_packets(candidates, groups, &id, &active, &next_active);
            frames.push(f);
            active = next_active;
            eliminated.insert(id);
        } else {
            frames.push(f);
            return (None, frames);
        }

        if active.len() <= 1 {
            let winner = active.iter().next().cloned();
            return (winner, frames);
        }
        round += 1;
        if round > 12 {
            return (None, frames);
        }
    }
}

fn candidate_ids_from_groups(groups: &[VoteGroup]) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for group in groups {
        for id in &group.ranking {
            ids.insert(id.clone());
        }
        for id in &group.approvals {
            ids.insert(id.clone());
        }
        for id in group.scores.keys() {
            ids.insert(id.clone());
        }
        for id in group.wagers.keys() {
            ids.insert(id.clone());
        }
    }
    ids.into_iter().collect()
}

fn first_active_choice(ranking: &[String], active: &BTreeSet<String>) -> Option<String> {
    ranking.iter().find(|id| active.contains(*id)).cloned()
}

fn active_tally(
    groups: &[VoteGroup],
    active: &BTreeSet<String>,
) -> (BTreeMap<String, u32>, u32) {
    let mut totals = BTreeMap::new();
    let mut exhausted = 0;
    for group in groups {
        if let Some(id) = first_active_choice(&group.ranking, active) {
            *totals.entry(id).or_insert(0) += group.count;
        } else {
            exhausted += group.count;
        }
    }
    (totals, exhausted)
}

fn transfer_packets(
    candidates: &[Candidate],
    groups: &[VoteGroup],
    eliminated_id: &str,
    active_before: &BTreeSet<String>,
    active_after: &BTreeSet<String>,
) -> Vec<Transfer> {
    let mut totals: BTreeMap<Option<String>, u32> = BTreeMap::new();
    for group in groups {
        if first_active_choice(&group.ranking, active_before).as_deref() == Some(eliminated_id) {
            let next = first_active_choice(&group.ranking, active_after);
            *totals.entry(next).or_insert(0) += group.count;
        }
    }
    totals
        .into_iter()
        .map(|(to, count)| Transfer {
            from: eliminated_id.to_string(),
            to: to.clone(),
            count,
            label: to
                .as_deref()
                .map(|id| format!("to {}", candidate_label(candidates, id)))
                .unwrap_or_else(|| "exhausted".to_string()),
            color: to
                .as_deref()
                .map(|id| candidate_color(candidates, id))
                .unwrap_or_else(|| "#94a3b8".to_string()),
        })
        .collect()
}

fn borda_system() -> VotingSystem {
    system(
        "borda",
        "Borda Count",
        "Borda",
        "ranked scoring",
        "Rank positions convert to points. With N candidates, first gets N-1 points, second N-2, and so on.",
        "Prioritization and committee-style rankings where full preference order should matter.",
        &[
            "Borda is sensitive to clones and nomination strategy.",
            "Truncated rankings need a points rule for unranked candidates.",
            "It can elect a compromise candidate without pairwise majority support.",
        ],
        vec![
            borda_case(
                "full-ranking",
                "Full ranking",
                "normal",
                "A compromise candidate accumulates steady second-place points and wins.",
                None,
                civic_candidates(),
                vec![
                    VoteGroup::new("Avery bloc", 310, &["avery", "blair", "casey", "devon"]),
                    VoteGroup::new("Blair bloc", 260, &["blair", "casey", "avery", "devon"]),
                    VoteGroup::new("Casey bloc", 240, &["casey", "blair", "devon", "avery"]),
                    VoteGroup::new("Devon bloc", 190, &["devon", "blair", "casey", "avery"]),
                ],
                &["The middle-ranked candidate can win by collecting points from many blocs."],
            ),
            borda_case(
                "clone-candidate",
                "Clone edge",
                "edge",
                "A similar Avery-Prime entrant changes the point field and drags points away from the original Avery coalition.",
                Some("Adding a near-clone can alter the outcome even when voter sentiment barely changes."),
                vec![
                    candidate("avery", "Avery", "Av", "#1f77b4"),
                    candidate("avery_prime", "Avery Prime", "A+", "#38bdf8"),
                    candidate("blair", "Blair", "Bl", "#2ca02c"),
                    candidate("casey", "Casey", "Ca", "#d95f02"),
                ],
                vec![
                    VoteGroup::new("Avery loyalists", 300, &["avery", "avery_prime", "blair", "casey"]),
                    VoteGroup::new("Avery Prime loyalists", 120, &["avery_prime", "avery", "blair", "casey"]),
                    VoteGroup::new("Blair coalition", 330, &["blair", "casey", "avery", "avery_prime"]),
                    VoteGroup::new("Casey coalition", 250, &["casey", "blair", "avery_prime", "avery"]),
                ],
                &[
                    "Clone sensitivity is an agenda-design edge case, not a bug in arithmetic.",
                    "Candidate qualification rules are part of the voting system.",
                ],
            ),
        ],
    )
}

fn borda_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    candidates: Vec<Candidate>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let totals = borda_tally(&candidates, &groups);
    let winner = sorted_winner(&totals);
    let max_per_ballot = candidates.len().saturating_sub(1) as f64;
    let mut frames = vec![
        frame(
            "Convert ranks to points",
            "Each ballot distributes descending points across its ranked candidates.",
            0,
            bars_from_u32(&candidates, &totals, 1000.0 * max_per_ballot, None, &BTreeSet::new(), "points"),
            None,
            &["Unranked candidates receive zero points in this demo."],
        ),
        frame(
            "Sum Borda points",
            "The candidate with the highest total point score wins.",
            1,
            bars_from_u32(
                &candidates,
                &totals,
                1000.0 * max_per_ballot,
                winner.as_deref(),
                &BTreeSet::new(),
                "points",
            ),
            None,
            caveats,
        ),
    ];
    frames[0].packets = packets_from_first_choices(&candidates, &groups);
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates,
        1000,
        Some("highest Borda score".to_string()),
        winner.clone(),
        winner
            .as_deref()
            .map(|id| format!("{} wins by Borda points.", candidate_label(&frames_candidates_hint(), id)))
            .unwrap_or_else(|| "No Borda winner.".to_string()),
        caveats,
        frames,
    )
}

fn frames_candidates_hint() -> Vec<Candidate> {
    let mut c = civic_candidates();
    c.push(candidate("avery_prime", "Avery Prime", "A+", "#38bdf8"));
    c
}

fn borda_tally(candidates: &[Candidate], groups: &[VoteGroup]) -> BTreeMap<String, u32> {
    let n = candidates.len();
    let mut totals = BTreeMap::new();
    for group in groups {
        for (rank, id) in group.ranking.iter().enumerate() {
            if candidates.iter().any(|c| c.id == *id) && rank < n {
                let points = (n - rank - 1) as u32;
                *totals.entry(id.clone()).or_insert(0) += group.count * points;
            }
        }
    }
    totals
}

fn condorcet_system() -> VotingSystem {
    system(
        "condorcet",
        "Condorcet / Copeland Fallback",
        "Condorcet",
        "pairwise majority",
        "Every candidate is compared head-to-head. A Condorcet winner beats every other candidate; otherwise Copeland scores pairwise wins minus losses.",
        "High-legitimacy ranked decisions where pairwise majority preference is a first-class criterion.",
        &[
            "A Condorcet cycle can exist: A beats B, B beats C, and C beats A.",
            "The fallback rule must be specified before the election.",
            "Pairwise math is auditable but harder to explain than a single pile count.",
        ],
        vec![
            condorcet_case(
                "pairwise-winner",
                "Pairwise winner",
                "normal",
                "Blair beats each other candidate head-to-head.",
                None,
                civic_candidates(),
                vec![
                    VoteGroup::new("Avery bloc", 360, &["avery", "blair", "casey", "devon"]),
                    VoteGroup::new("Blair bloc", 300, &["blair", "casey", "avery", "devon"]),
                    VoteGroup::new("Casey bloc", 200, &["casey", "blair", "avery", "devon"]),
                    VoteGroup::new("Devon bloc", 140, &["devon", "casey", "blair", "avery"]),
                ],
                &["A pairwise winner is strong evidence of broad legitimacy."],
            ),
            condorcet_case(
                "cycle",
                "Cycle edge",
                "edge",
                "The electorate forms a rock-paper-scissors loop: Avery beats Blair, Blair beats Casey, Casey beats Avery.",
                Some("There is no candidate that beats every other candidate."),
                vec![
                    candidate("avery", "Avery", "Av", "#1f77b4"),
                    candidate("blair", "Blair", "Bl", "#2ca02c"),
                    candidate("casey", "Casey", "Ca", "#d95f02"),
                ],
                vec![
                    VoteGroup::new("A > B > C", 334, &["avery", "blair", "casey"]),
                    VoteGroup::new("B > C > A", 333, &["blair", "casey", "avery"]),
                    VoteGroup::new("C > A > B", 333, &["casey", "avery", "blair"]),
                ],
                &[
                    "The fallback exposes a governance choice: break the cycle, request deliberation, or rerun.",
                    "For court-like gates, a cycle is a signal to slow down, not force a verdict.",
                ],
            ),
        ],
    )
}

fn condorcet_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    candidates: Vec<Candidate>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let result = pairwise_result(&candidates, &groups);
    let winner = result.condorcet_winner.clone().or(result.copeland_winner.clone());
    let mut frames = vec![
        frame(
            "Build pairwise matrix",
            "Each pair of candidates is tested as a one-on-one majority contest.",
            0,
            pairwise_bars(&candidates, &result, None),
            None,
            &["The matrix records margins for every head-to-head race."],
        ),
        frame(
            if result.condorcet_winner.is_some() { "Condorcet winner found" } else { "No Condorcet winner" },
            if result.condorcet_winner.is_some() {
                "One candidate beats every other candidate head-to-head."
            } else {
                "The pairwise graph cycles, so the fallback score is shown without pretending there is a clean majority winner."
            },
            1,
            pairwise_bars(&candidates, &result, winner.as_deref()),
            None,
            caveats,
        ),
    ];
    for f in &mut frames {
        f.pairwise = result.cells.clone();
    }
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates.clone(),
        1000,
        Some("beats every rival head-to-head".to_string()),
        winner.clone(),
        match result.condorcet_winner.as_deref() {
            Some(id) => format!("{} is the Condorcet winner.", candidate_label(&candidates, id)),
            None => match result.copeland_winner.as_deref() {
                Some(id) => format!("No Condorcet winner; {} leads the Copeland fallback.", candidate_label(&candidates, id)),
                None => "No Condorcet winner and Copeland fallback is tied.".to_string(),
            },
        },
        caveats,
        frames,
    )
}

#[derive(Clone, Debug)]
struct PairwiseResult {
    cells: Vec<PairwiseCell>,
    wins: BTreeMap<String, i32>,
    losses: BTreeMap<String, i32>,
    condorcet_winner: Option<String>,
    copeland_winner: Option<String>,
}

fn pairwise_result(candidates: &[Candidate], groups: &[VoteGroup]) -> PairwiseResult {
    let mut cells = Vec::new();
    let mut wins: BTreeMap<String, i32> = BTreeMap::new();
    let mut losses: BTreeMap<String, i32> = BTreeMap::new();
    for row in candidates {
        wins.entry(row.id.clone()).or_insert(0);
        losses.entry(row.id.clone()).or_insert(0);
    }
    for i in 0..candidates.len() {
        for j in i + 1..candidates.len() {
            let a = &candidates[i].id;
            let b = &candidates[j].id;
            let (a_votes, b_votes) = pairwise_votes(groups, a, b);
            let winner = if a_votes > b_votes {
                *wins.entry(a.clone()).or_insert(0) += 1;
                *losses.entry(b.clone()).or_insert(0) += 1;
                Some(a.clone())
            } else if b_votes > a_votes {
                *wins.entry(b.clone()).or_insert(0) += 1;
                *losses.entry(a.clone()).or_insert(0) += 1;
                Some(b.clone())
            } else {
                None
            };
            cells.push(PairwiseCell {
                row: a.clone(),
                col: b.clone(),
                row_votes: a_votes,
                col_votes: b_votes,
                margin: a_votes as i32 - b_votes as i32,
                winner,
            });
        }
    }
    let needed = candidates.len().saturating_sub(1) as i32;
    let condorcet_winner = wins
        .iter()
        .find(|(_, count)| **count == needed)
        .map(|(id, _)| id.clone());
    let mut copeland: Vec<(String, i32)> = candidates
        .iter()
        .map(|c| {
            let score = *wins.get(&c.id).unwrap_or(&0) - *losses.get(&c.id).unwrap_or(&0);
            (c.id.clone(), score)
        })
        .collect();
    copeland.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let copeland_winner = if copeland.len() > 1 && copeland[0].1 == copeland[1].1 {
        None
    } else {
        copeland.first().map(|(id, _)| id.clone())
    };
    PairwiseResult {
        cells,
        wins,
        losses,
        condorcet_winner,
        copeland_winner,
    }
}

fn pairwise_votes(groups: &[VoteGroup], a: &str, b: &str) -> (u32, u32) {
    let mut a_votes = 0;
    let mut b_votes = 0;
    for group in groups {
        let apos = rank_pos(&group.ranking, a);
        let bpos = rank_pos(&group.ranking, b);
        if apos < bpos {
            a_votes += group.count;
        } else if bpos < apos {
            b_votes += group.count;
        }
    }
    (a_votes, b_votes)
}

fn rank_pos(ranking: &[String], id: &str) -> usize {
    ranking
        .iter()
        .position(|x| x == id)
        .unwrap_or(usize::MAX / 2)
}

fn pairwise_bars(
    candidates: &[Candidate],
    result: &PairwiseResult,
    winner_id: Option<&str>,
) -> Vec<TallyBar> {
    let max_wins = candidates.len().saturating_sub(1).max(1) as f64;
    let mut bars: Vec<TallyBar> = candidates
        .iter()
        .map(|c| {
            let wins = *result.wins.get(&c.id).unwrap_or(&0);
            let losses = *result.losses.get(&c.id).unwrap_or(&0);
            let score = wins - losses;
            TallyBar {
                candidate_id: c.id.clone(),
                label: c.name.clone(),
                color: c.color.clone(),
                value: wins as f64,
                percent: 100.0 * wins as f64 / max_wins,
                status: if Some(c.id.as_str()) == winner_id {
                    "winner".to_string()
                } else {
                    "active".to_string()
                },
                detail: format!("{wins} pairwise wins, {losses} losses, Copeland {score:+}"),
                secondary_value: Some(score as f64),
                secondary_label: Some("Copeland".to_string()),
            }
        })
        .collect();
    bars.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.secondary_value.partial_cmp(&a.secondary_value).unwrap_or(Ordering::Equal))
            .then_with(|| a.label.cmp(&b.label))
    });
    bars
}

fn wagered_magnitude_system() -> VotingSystem {
    system(
        "wagered-magnitude",
        "Wagered Magnitude Voting",
        "Wager",
        "dollar-weighted signal",
        "All 1,000 records vote, but each record carries a dollar-backed magnitude. Raw dollars can be counted directly or compared against capped/square-root damped voice.",
        "Escrow-backed public support, prediction-style conviction signals, or funding readiness gates where intensity is meaningful but capture risk is high.",
        &[
            "Dollar magnitude is not democratic equality; it is an intensity or funding signal.",
            "Raw dollar weighting can let a tiny wealthy bloc dominate a large public majority.",
            "A court-like system should keep wagered support separate from reviewer verdict votes.",
        ],
        vec![
            wager_case(
                "high-intensity-minority",
                "High-intensity minority",
                "normal",
                "A smaller Avery bloc wins the raw dollar-weighted tally by attaching larger dollar amounts to each record.",
                None,
                vec![
                    VoteGroup::new("Avery high-intensity", 260, &["avery"]).wagers(&[("avery", 8)]),
                    VoteGroup::new("Blair broad-support", 520, &["blair"]).wagers(&[("blair", 3)]),
                    VoteGroup::new("Casey medium", 220, &["casey"]).wagers(&[("casey", 4)]),
                ],
                &[
                    "This is appropriate only if dollars represent the thing being allocated, such as escrow readiness.",
                    "The same 1,000 people are counted; the difference is how much magnitude each record carries.",
                ],
            ),
            wager_case(
                "whale-dominance",
                "Whale dominance edge",
                "edge",
                "Twenty high-dollar records beat 980 low-dollar records under raw weighting.",
                Some("Dollar-backed voting can convert wealth concentration into governance control."),
                vec![
                    VoteGroup::new("20 high-dollar Avery sponsors", 20, &["avery"]).wagers(&[("avery", 250)]),
                    VoteGroup::new("980 low-dollar Blair supporters", 980, &["blair"]).wagers(&[("blair", 3)]),
                ],
                &[
                    "Raw dollars pick Avery; square-root damping would pick Blair in the comparison frame.",
                    "Caps, matching funds, per-person limits, or quadratic costs are policy choices, not UI details.",
                ],
            ),
        ],
    )
}

fn wager_case(
    id: &str,
    label: &str,
    kind: &str,
    scenario: &str,
    edge_focus: Option<&str>,
    groups: Vec<VoteGroup>,
    caveats: &[&str],
) -> ElectionCase {
    assert_thousand(&groups);
    let candidates = civic_candidates();
    let raw = wager_tally(&groups);
    let raw_winner = sorted_i32_winner(&raw);
    let raw_denominator: f64 = raw.values().map(|v| (*v as f64).abs()).sum::<f64>().max(1.0);
    let quadratic = quadratic_voice_tally(&groups);
    let quadratic_winner = sorted_f64_winner(&quadratic);
    let quadratic_denominator: f64 = quadratic.values().map(|v| v.abs()).sum::<f64>().max(1.0);
    let mut raw_bars = bars_from_i32(
        &candidates,
        &raw,
        raw_denominator,
        raw_winner.as_deref(),
        "raw dollar-voice",
    );
    for bar in &mut raw_bars {
        if let Some(q) = quadratic.get(&bar.candidate_id) {
            bar.secondary_value = Some(*q);
            bar.secondary_label = Some("sqrt-damped voice".to_string());
        }
    }
    let mut frames = vec![
        frame(
            "Collect 1,000 wagered records",
            "Each vote record names a candidate and attaches a dollar-backed magnitude.",
            0,
            bars_from_i32(&candidates, &raw, raw_denominator, None, "raw dollar-voice"),
            None,
            &["The count remains 1,000; the dollar field changes the weight of each record."],
        ),
        frame(
            "Raw dollar-weighted tally",
            "The direct rule sums dollars behind each candidate.",
            1,
            raw_bars,
            None,
            caveats,
        ),
        frame(
            "Square-root damped comparison",
            "A quadratic-style comparison converts each record's dollars to sqrt(dollars) voice before aggregation.",
            2,
            bars_from_f64(
                &candidates,
                &quadratic,
                quadratic_denominator,
                quadratic_winner.as_deref(),
                "sqrt voice",
            ),
            None,
            &[
                "This frame is a policy comparison, not the raw-rule winner.",
                "Damping preserves intensity while limiting wealth dominance.",
            ],
        ),
    ];
    frames[0].packets = groups
        .iter()
        .map(|g| {
            let to = g.wagers.keys().next().cloned();
            let dollars = g.wagers.values().next().copied().unwrap_or(0);
            VotePacket {
                label: g.label.clone(),
                count: g.count,
                from: None,
                to: to.clone(),
                color: to
                    .as_deref()
                    .map(|id| candidate_color(&candidates, id))
                    .unwrap_or_else(|| "#64748b".to_string()),
                note: format!("{} records x ${dollars}", g.count),
            }
        })
        .collect();
    election_case(
        id,
        label,
        kind,
        scenario,
        edge_focus,
        candidates.clone(),
        1000,
        Some("highest raw dollar-weighted total".to_string()),
        raw_winner.clone(),
        raw_winner
            .as_deref()
            .map(|id| {
                let q = quadratic_winner
                    .as_deref()
                    .map(|qid| candidate_label(&candidates, qid))
                    .unwrap_or_else(|| "none".to_string());
                format!(
                    "{} wins under raw dollars; square-root comparison winner: {q}.",
                    candidate_label(&candidates, id)
                )
            })
            .unwrap_or_else(|| "No wagered-magnitude winner.".to_string()),
        caveats,
        frames,
    );
    case.betting_lines = wager_betting_lines(&candidates, &raw, &quadratic);
    case
}

fn quadratic_voice_tally(groups: &[VoteGroup]) -> BTreeMap<String, f64> {
    let mut totals = BTreeMap::new();
    for group in groups {
        for (id, wager) in &group.wagers {
            let sign = if *wager < 0 { -1.0 } else { 1.0 };
            let voice = sign * (*wager as f64).abs().sqrt() * group.count as f64;
            *totals.entry(id.clone()).or_insert(0.0) += voice;
        }
    }
    totals
}

fn sorted_f64_winner(totals: &BTreeMap<String, f64>) -> Option<String> {
    totals
        .iter()
        .max_by(|(aid, av), (bid, bv)| {
            av.partial_cmp(bv)
                .unwrap_or(Ordering::Equal)
                .then_with(|| bid.cmp(aid))
        })
        .map(|(id, _)| id.clone())
}

fn bars_from_f64(
    candidates: &[Candidate],
    totals: &BTreeMap<String, f64>,
    denominator: f64,
    winner_id: Option<&str>,
    units: &str,
) -> Vec<TallyBar> {
    let mut bars: Vec<TallyBar> = candidates
        .iter()
        .filter_map(|c| {
            let value = *totals.get(&c.id).unwrap_or(&0.0);
            if value == 0.0 && !totals.contains_key(&c.id) {
                return None;
            }
            Some(TallyBar {
                candidate_id: c.id.clone(),
                label: c.name.clone(),
                color: c.color.clone(),
                value,
                percent: if denominator > 0.0 {
                    100.0 * value.abs() / denominator
                } else {
                    0.0
                },
                status: if Some(c.id.as_str()) == winner_id {
                    "winner".to_string()
                } else {
                    "active".to_string()
                },
                detail: format!("{} {}", fmt_num(value), units),
                secondary_value: None,
                secondary_label: None,
            })
        })
        .collect();
    bars.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    bars
}
