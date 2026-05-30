//! Port of `src/des/mdp/usacc-mdp.ts`.
//!
//! The USACC (US Anti-Corruption Court) court-case MDP: states, actions,
//! transitions and rewards. We model the MDP version (the visible state IS the
//! ground truth the policy acts on), per the project spec at
//! https://oresoftware.github.io/us-anti-corruption-court-project/mdp.
//!
//! State factors and their domain sizes are: case_stage in {SUB,VAL,ADM,TRI}
//! (4), evidence_strength in {LO,MED,HI} (3), corroboration in
//! {NONE,SINGLE,MULTI} (3), manipulation_risk in {LO,MED,HI} (3), conflict_risk
//! in {LO,HI} (2), funding_status in {UNFUNDED,ESCROWED,ACTIVE,EXHAUSTED} (4).
//! That gives 4*3*3*3*2*4 = 864 non-terminal states plus 3 terminal states
//! (ACCEPTED, CLOSED, EXHAUSTED) for 867 total.
//!
//! States are encoded as a single sequential integer:
//! id(s) = ((((stage*3 + ev)*3 + corr)*3 + man)*2 + conf)*4 + fund, with the
//! terminals at ACCEPTED = 864, CLOSED = 865, EXHAUSTED = 866.
//!
//! Each (s, a) yields a list of (s', p, r) outcomes whose probabilities sum to
//! 1; r is the immediate reward of the (s, a, s') triple. Per-action costs are
//! small negatives; terminal rewards are large signed values. ACCEPTED reward =
//! 50*(Q - 0.5) and CLOSED reward = 50*(0.5 - Q) where Q = ev + corr - man -
//! 1.5*conf; EXHAUSTED reward = -150.
//!
//! Probabilities are encoded as integer percents and divided by 10000 at the
//! very end so the floats round identically in TS, Python and Rust. The factor
//! string-literal unions become Rust enums; the `CaseState` itself stores the
//! factors as small ints (0..N) exactly as the TS does.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::shared::capabilities::RandomSource;

// -----------------------------------------------------------------------------
// State factor enumerations (string-literal unions in the TS source).
// -----------------------------------------------------------------------------

pub const STAGES: [&str; 4] = ["SUB", "VAL", "ADM", "TRI"];
pub const EVIDENCE: [&str; 3] = ["LO", "MED", "HI"];
pub const CORROBORATION: [&str; 3] = ["NONE", "SINGLE", "MULTI"];
pub const MANIPULATION: [&str; 3] = ["LO", "MED", "HI"];
pub const CONFLICT: [&str; 2] = ["LO", "HI"];
pub const FUNDING: [&str; 4] = ["UNFUNDED", "ESCROWED", "ACTIVE", "EXHAUSTED"];

/// `type Stage = typeof STAGES[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Sub,
    Val,
    Adm,
    Tri,
}

/// `type Evidence = typeof EVIDENCE[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    Lo,
    Med,
    Hi,
}

/// `type Corroboration = typeof CORROBORATION[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corroboration {
    None,
    Single,
    Multi,
}

/// `type Manipulation = typeof MANIPULATION[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manipulation {
    Lo,
    Med,
    Hi,
}

/// `type Conflict = typeof CONFLICT[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conflict {
    Lo,
    Hi,
}

/// `type Funding = typeof FUNDING[number]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Funding {
    Unfunded,
    Escrowed,
    Active,
    Exhausted,
}

/// A non-terminal case state. Each factor is stored as a small integer index
/// (0..N), matching the TS `interface CaseState` where every field is a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseState {
    pub stage: u8,         // 0..3
    pub evidence: u8,      // 0..2
    pub corroboration: u8, // 0..2
    pub manipulation: u8,  // 0..2
    pub conflict: u8,      // 0..1
    pub funding: u8,       // 0..3
}

pub const N_STATES: usize = 4 * 3 * 3 * 3 * 2 * 4 + 3; // 867
pub const ACCEPTED: usize = 864;
pub const CLOSED: usize = 865;
pub const EXHAUSTED: usize = 866;

pub const ACTIONS: [&str; 8] = [
    "request_more_evidence",
    "verify_identity",
    "normalize_record",
    "assign_reviewers",
    "hold_for_audit",
    "escalate_to_next_stage",
    "release_escrow",
    "reject_or_close",
];
pub const N_ACTIONS: usize = ACTIONS.len();

pub const FUND_UNFUNDED: u8 = 0;
pub const FUND_ESCROWED: u8 = 1;
pub const FUND_ACTIVE: u8 = 2;
pub const FUND_EXHAUSTED: u8 = 3;

/// `const ACTIONS (as const) + type Action`. Carries the per-action cost / draw
/// tables as `match`-based associated functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    RequestMoreEvidence,
    VerifyIdentity,
    NormalizeRecord,
    AssignReviewers,
    HoldForAudit,
    EscalateToNextStage,
    ReleaseEscrow,
    RejectOrClose,
}

impl Action {
    /// `ACTIONS[index]`. Panics on an out-of-range index (invariant violation).
    pub fn from_index(index: usize) -> Action {
        match index {
            0 => Action::RequestMoreEvidence,
            1 => Action::VerifyIdentity,
            2 => Action::NormalizeRecord,
            3 => Action::AssignReviewers,
            4 => Action::HoldForAudit,
            5 => Action::EscalateToNextStage,
            6 => Action::ReleaseEscrow,
            7 => Action::RejectOrClose,
            _ => panic!("Action::from_index: out-of-range action index {index}"),
        }
    }

    /// `ACTION_COST[a]` — the per-action small negative cost (used as the
    /// per-step reward of non-terminal transitions).
    pub fn cost(self) -> f64 {
        match self {
            Action::RequestMoreEvidence => -2.0,
            Action::VerifyIdentity => -2.0,
            Action::NormalizeRecord => -1.0,
            Action::AssignReviewers => -3.0,
            Action::HoldForAudit => -5.0,
            Action::EscalateToNextStage => -2.0,
            Action::ReleaseEscrow => -1.0,
            Action::RejectOrClose => 0.0,
        }
    }

    /// `drawPctPerAction[a]` — the integer-percent funding-draw probability.
    pub fn draw_pct(self) -> u32 {
        match self {
            Action::RequestMoreEvidence => 25,
            Action::VerifyIdentity => 25,
            Action::NormalizeRecord => 10,
            Action::AssignReviewers => 30,
            Action::HoldForAudit => 50,
            Action::EscalateToNextStage => 25,
            Action::ReleaseEscrow => 0,
            Action::RejectOrClose => 0,
        }
    }
}

// -----------------------------------------------------------------------------
// State encoding / decoding.
// -----------------------------------------------------------------------------

pub fn encode(s: &CaseState) -> usize {
    ((((s.stage as usize * 3 + s.evidence as usize) * 3 + s.corroboration as usize) * 3
        + s.manipulation as usize)
        * 2
        + s.conflict as usize)
        * 4
        + s.funding as usize
}

pub fn decode(id: usize) -> Option<CaseState> {
    if id >= 864 {
        return None;
    }
    let mut id = id;
    let funding = (id % 4) as u8;
    id /= 4;
    let conflict = (id % 2) as u8;
    id /= 2;
    let manipulation = (id % 3) as u8;
    id /= 3;
    let corroboration = (id % 3) as u8;
    id /= 3;
    let evidence = (id % 3) as u8;
    id /= 3;
    let stage = id as u8;
    Some(CaseState {
        stage,
        evidence,
        corroboration,
        manipulation,
        conflict,
        funding,
    })
}

pub fn is_terminal(id: usize) -> bool {
    id >= 864
}

// -----------------------------------------------------------------------------
// Reward model.
// -----------------------------------------------------------------------------

/// Quality score Q in [-3.5, +4]: how genuinely strong / clean / honest the
/// case is, given its visible factors. High Q means escalating to ACCEPTED is
/// good and closing is bad; low Q means the reverse.
pub fn quality(s: &CaseState) -> f64 {
    s.evidence as f64 + s.corroboration as f64 - s.manipulation as f64 - 1.5 * s.conflict as f64
}

pub fn terminal_reward(id: usize) -> f64 {
    if id == ACCEPTED {
        return 0.0; // populated at transition time
    }
    if id == CLOSED {
        return 0.0; // populated at transition time
    }
    if id == EXHAUSTED {
        return -150.0;
    }
    0.0
}

/// Reward of arriving at ACCEPTED from a (presumably non-terminal) state `s`.
pub fn reward_of_accept(s: &CaseState) -> f64 {
    50.0 * (quality(s) - 0.5)
}

/// Reward of arriving at CLOSED from state `s`.
pub fn reward_of_close(s: &CaseState) -> f64 {
    50.0 * (0.5 - quality(s))
}

// -----------------------------------------------------------------------------
// Transition model.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    pub next_state: usize, // state id (0..866)
    pub prob: f64,
    pub reward: f64,
}

/// Local `type Edge` (built via spread/`Partial`/`Omit` in TS). Here it is a
/// `Copy` struct populated with struct-update syntax (`Edge { ..stay }`).
#[derive(Clone, Copy)]
struct Edge {
    next_stage: u8,
    next_ev: u8,
    next_corr: u8,
    next_man: u8,
    next_conf: u8,
    next_fund: u8,
    pct: u32, // probability in integer percent (sum to 100)
    extra_reward: f64,
    target_terminal: Option<usize>,
}

/// Build the outcome list for `(s, a)`. Pure function: the same input always
/// produces the same list, and the probabilities sum to 1.
pub fn outcomes(state_id: usize, action: usize) -> Vec<Outcome> {
    if is_terminal(state_id) {
        // Terminal states absorb.
        return vec![Outcome {
            next_state: state_id,
            prob: 1.0,
            reward: 0.0,
        }];
    }
    let s = decode(state_id).expect("decode of a non-terminal state id");
    let a = Action::from_index(action);
    let cost = a.cost();

    let stay = Edge {
        next_stage: s.stage,
        next_ev: s.evidence,
        next_corr: s.corroboration,
        next_man: s.manipulation,
        next_conf: s.conflict,
        next_fund: s.funding,
        pct: 0,
        extra_reward: 0.0,
        target_terminal: None,
    };

    // `funded` is computed but unused in the TS source; retained for parity.
    let _funded = s.funding > FUND_UNFUNDED;

    let mut edges: Vec<Edge> = Vec::new();

    match a {
        Action::RequestMoreEvidence => {
            // 60% evidence +1, 30% nothing, 10% reveal manipulation +1.
            let ev_up = (s.evidence + 1).min(2);
            let man_up = (s.manipulation + 1).min(2);
            edges.push(Edge {
                next_ev: ev_up,
                pct: 60,
                ..stay
            });
            edges.push(Edge { pct: 30, ..stay });
            edges.push(Edge {
                next_man: man_up,
                pct: 10,
                ..stay
            });
        }
        Action::VerifyIdentity => {
            // 50% corroboration +1, 30% nothing, 20% manipulation -1 (if > LO).
            let corr_up = (s.corroboration + 1).min(2);
            let man_dn = s.manipulation.saturating_sub(1);
            edges.push(Edge {
                next_corr: corr_up,
                pct: 50,
                ..stay
            });
            edges.push(Edge { pct: 30, ..stay });
            edges.push(Edge {
                next_man: man_dn,
                pct: 20,
                ..stay
            });
        }
        Action::NormalizeRecord => {
            // 30% evidence +1, 40% conflict resolves to LO, 30% nothing.
            let ev_up = (s.evidence + 1).min(2);
            edges.push(Edge {
                next_ev: ev_up,
                pct: 30,
                ..stay
            });
            edges.push(Edge {
                next_conf: 0,
                pct: 40,
                ..stay
            });
            edges.push(Edge { pct: 30, ..stay });
        }
        Action::AssignReviewers => {
            // 60% conflict resolves to LO, 30% evidence +1, 10% nothing.
            let ev_up = (s.evidence + 1).min(2);
            edges.push(Edge {
                next_conf: 0,
                pct: 60,
                ..stay
            });
            edges.push(Edge {
                next_ev: ev_up,
                pct: 30,
                ..stay
            });
            edges.push(Edge { pct: 10, ..stay });
        }
        Action::HoldForAudit => {
            // 70% manipulation collapses to LO, 20% evidence +1, 10% nothing.
            let ev_up = (s.evidence + 1).min(2);
            edges.push(Edge {
                next_man: 0,
                pct: 70,
                ..stay
            });
            edges.push(Edge {
                next_ev: ev_up,
                pct: 20,
                ..stay
            });
            edges.push(Edge { pct: 10, ..stay });
        }
        Action::EscalateToNextStage => {
            if s.stage == 3 {
                // From TRI, escalation = ACCEPTED.
                return vec![Outcome {
                    next_state: ACCEPTED,
                    prob: 1.0,
                    reward: cost + reward_of_accept(&s),
                }];
            }
            // 80% stage advances, 20% fails (conflict surfaces) — stage stays,
            // conflict goes HI.
            edges.push(Edge {
                next_stage: s.stage + 1,
                pct: 80,
                ..stay
            });
            edges.push(Edge {
                next_conf: 1,
                pct: 20,
                ..stay
            });
        }
        Action::ReleaseEscrow => {
            // funding advances by 1 (capped at ACTIVE). 100% deterministic.
            let fund_next = (s.funding + 1).min(FUND_ACTIVE);
            edges.push(Edge {
                next_fund: fund_next,
                pct: 100,
                ..stay
            });
        }
        Action::RejectOrClose => {
            return vec![Outcome {
                next_state: CLOSED,
                prob: 1.0,
                reward: cost + reward_of_close(&s),
            }];
        }
    }

    let draw_pct = a.draw_pct();

    // Build the final outcome list. For each base edge, fork it into
    // (no-draw, with-draw). Funding decrease: ACTIVE -> ESCROWED -> UNFUNDED ->
    // EXHAUSTED.
    let mut out: Vec<Outcome> = Vec::new();
    for e in &edges {
        let base_prob = (e.pct * (100 - draw_pct)) as f64 / 10000.0;
        let draw_prob = (e.pct * draw_pct) as f64 / 10000.0;
        if base_prob > 0.0 {
            let s_next = CaseState {
                stage: e.next_stage,
                evidence: e.next_ev,
                corroboration: e.next_corr,
                manipulation: e.next_man,
                conflict: e.next_conf,
                funding: e.next_fund,
            };
            out.push(Outcome {
                next_state: encode(&s_next),
                prob: base_prob,
                reward: cost,
            });
        }
        if draw_prob > 0.0 {
            let f_after_draw = e.next_fund as i32 - 1;
            if f_after_draw < FUND_UNFUNDED as i32 {
                // Funding underflow -> EXHAUSTED.
                out.push(Outcome {
                    next_state: EXHAUSTED,
                    prob: draw_prob,
                    reward: cost - 150.0,
                });
            } else {
                let s_next = CaseState {
                    stage: e.next_stage,
                    evidence: e.next_ev,
                    corroboration: e.next_corr,
                    manipulation: e.next_man,
                    conflict: e.next_conf,
                    funding: f_after_draw as u8,
                };
                out.push(Outcome {
                    next_state: encode(&s_next),
                    prob: draw_prob,
                    reward: cost,
                });
            }
        }
    }

    // Coalesce duplicates (same nextState gets prob summed), preserving the
    // insertion order of the TS `Map`.
    let mut index_of: HashMap<usize, usize> = HashMap::new();
    let mut coalesced: Vec<Outcome> = Vec::new();
    for o in out {
        if let Some(&idx) = index_of.get(&o.next_state) {
            coalesced[idx].prob += o.prob;
        } else {
            index_of.insert(o.next_state, coalesced.len());
            coalesced.push(o);
        }
    }

    // Sanity: probabilities sum to ~1.
    let mut p = 0.0;
    for o in &coalesced {
        p += o.prob;
    }
    if (p - 1.0).abs() > 1e-9 {
        panic!("outcomes({state_id}, {action}) probability sum {p} != 1");
    }
    coalesced
}

// -----------------------------------------------------------------------------
// Initial state distribution: how cases enter the system.
// -----------------------------------------------------------------------------

/// Returns the starting state for a freshly-filed case under random "real
/// world" conditions. Most cases enter at SUB stage with messy partial info.
///
/// The TS `rng: () => number` becomes an injected `RandomSource`. The exact
/// number and order of `next_float()` draws is preserved (each call advances
/// the RNG state), including the short-circuit on the corroboration /
/// manipulation conditionals.
pub fn sample_initial_state(rng: &mut dyn RandomSource) -> CaseState {
    let ev_roll = rng.next_float();
    let evidence = if ev_roll < 0.5 {
        0
    } else if ev_roll < 0.85 {
        1
    } else {
        2
    };
    let corroboration = if rng.next_float() < 0.6 {
        0
    } else if rng.next_float() < 0.5 {
        1
    } else {
        2
    };
    let manipulation = if rng.next_float() < 0.5 {
        0
    } else if rng.next_float() < 0.6 {
        1
    } else {
        2
    };
    let conflict = if rng.next_float() < 0.7 { 0 } else { 1 };
    let funding = if rng.next_float() < 0.5 {
        FUND_UNFUNDED
    } else {
        FUND_ESCROWED
    };
    CaseState {
        stage: 0,
        evidence,
        corroboration,
        manipulation,
        conflict,
        funding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn encode_decode_roundtrip() {
        for id in 0..864usize {
            let s = decode(id).expect("non-terminal decodes");
            assert_eq!(encode(&s), id, "roundtrip failed for id {id}");
        }
        assert!(decode(864).is_none());
        assert!(decode(866).is_none());
    }

    #[test]
    fn terminal_helpers() {
        assert!(!is_terminal(0));
        assert!(!is_terminal(863));
        assert!(is_terminal(864));
        assert!(is_terminal(866));
        assert_eq!(terminal_reward(EXHAUSTED), -150.0);
        assert_eq!(terminal_reward(ACCEPTED), 0.0);
    }

    #[test]
    fn outcomes_probabilities_sum_to_one() {
        // Exercises every (state, action) pair, including the panic path.
        for s in 0..N_STATES {
            for a in 0..N_ACTIONS {
                let ol = outcomes(s, a);
                let p: f64 = ol.iter().map(|o| o.prob).sum();
                assert!((p - 1.0).abs() < 1e-9, "state {s} action {a} sum {p}");
                if is_terminal(s) {
                    assert_eq!(ol.len(), 1);
                    assert_eq!(ol[0].next_state, s);
                }
            }
        }
    }

    #[test]
    fn escalate_from_tri_accepts() {
        let s = CaseState {
            stage: 3,
            evidence: 2,
            corroboration: 2,
            manipulation: 0,
            conflict: 0,
            funding: 2,
        };
        let id = encode(&s);
        let ol = outcomes(id, 5); // escalate_to_next_stage
        assert_eq!(ol.len(), 1);
        assert_eq!(ol[0].next_state, ACCEPTED);
    }

    #[test]
    fn sample_initial_state_is_deterministic_under_seed() {
        let mut a = SeededRandom::new(99);
        let mut b = SeededRandom::new(99);
        for _ in 0..50 {
            assert_eq!(sample_initial_state(&mut a), sample_initial_state(&mut b));
        }
        let s = sample_initial_state(&mut SeededRandom::new(1));
        assert_eq!(s.stage, 0);
        assert!(s.evidence <= 2 && s.corroboration <= 2 && s.manipulation <= 2);
        assert!(s.conflict <= 1 && s.funding <= FUND_ESCROWED);
    }
}
