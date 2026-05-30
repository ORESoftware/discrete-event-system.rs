//! Port of `src/des/general/des-base/stateful-token.ts` — lineage-tracked
//! tokens with an optional state machine, plus a tracking registry.
//!
//! ## Rust shape (faithful translation)
//!
//!   * `type TokenStateMode` → [`TokenStateMode`] enum.
//!   * `interface TokenLineage` / `TokenStateTransition<S>` →
//!     `#[derive(Clone)]` structs with `Option` fields.
//!   * `interface StatefulToken<S> extends Token` → the [`StatefulToken<S>`]
//!     data struct (the TS interface was a plain record; there is no `Token`
//!     marker trait in the Rust framework — tokens travel as `Rc<dyn Any>`).
//!   * The `make*/transition/spawn/childLineage/isStatefulToken` free functions
//!     → free functions; the options objects → small param structs.
//!   * `class PayloadStatefulToken<S,P>` → struct embedding the base token plus
//!     a typed `payload`.
//!   * `class StatefulTokenRegistry` (which stored `StatefulToken<any>`) → a
//!     struct keyed by token id. Heterogeneous storage uses the object-safe
//!     [`TrackedToken`] view trait (`Box<dyn TrackedToken>`), since Rust cannot
//!     store differing `StatefulToken<S>` directly.
//!
//! FLAGGED simplifications vs. TS (no unported deps involved):
//!   * `childLineage` / parent params accept a borrowed [`TokenLineage`] rather
//!     than a whole parent token — only the lineage is read in TS.
//!   * `transition_token` mutates `&mut self` and returns nothing (TS returned
//!     the same handle for chaining).
//!   * `is_stateful_token` downcasts a `&dyn Any` to a concrete
//!     `StatefulToken<S>` (Rust has no structural narrowing); it does not match
//!     the `PayloadStatefulToken` wrapper, unlike the TS structural guard.

use std::any::Any;
use std::collections::HashMap;

/// `type TokenStateMode = 'stateless' | 'stateful'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenStateMode {
    Stateless,
    Stateful,
}

/// Provenance / parentage of a token within a run.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLineage {
    pub token_id: String,
    pub parent_token_id: Option<String>,
    pub root_token_id: String,
    pub causation_token_id: Option<String>,
    pub generation: usize,
}

/// One recorded state-machine transition for a token.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenStateTransition<S = String> {
    pub tick: f64,
    pub station_id: String,
    pub from: Option<S>,
    pub to: S,
    pub event: String,
    pub detail: Option<String>,
}

/// Lineage-tracked token with an optional state machine (the TS interface as a
/// concrete record).
#[derive(Clone, Debug, PartialEq)]
pub struct StatefulToken<S = String> {
    pub kind: String,
    pub lineage: TokenLineage,
    pub state_mode: TokenStateMode,
    pub current_state: Option<S>,
    pub state_history: Option<Vec<TokenStateTransition<S>>>,
}

/// Object-safe view used by the registry to store tokens of differing `S`.
pub trait TrackedToken {
    fn kind(&self) -> &str;
    fn lineage(&self) -> &TokenLineage;
    fn state_mode(&self) -> TokenStateMode;
    fn state_history_len(&self) -> usize;
}

impl<S: 'static> TrackedToken for StatefulToken<S> {
    fn kind(&self) -> &str {
        &self.kind
    }
    fn lineage(&self) -> &TokenLineage {
        &self.lineage
    }
    fn state_mode(&self) -> TokenStateMode {
        self.state_mode
    }
    fn state_history_len(&self) -> usize {
        self.state_history.as_ref().map_or(0, |h| h.len())
    }
}

/// Aggregate stats produced by [`StatefulTokenRegistry::snapshot`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StatefulTokenRegistryStats {
    pub created: u64,
    pub stateful: u64,
    pub stateless: u64,
    pub state_transitions: u64,
    pub max_generation: usize,
    pub by_kind: HashMap<String, u64>,
}

/// Options for [`make_stateful_token`].
pub struct MakeStatefulTokenOpts<S> {
    pub kind: String,
    pub token_id: String,
    pub initial_state: S,
    pub tick: f64,
    pub station_id: String,
    pub event: Option<String>,
    pub detail: Option<String>,
}

/// Options for [`make_stateless_token`].
pub struct MakeStatelessTokenOpts {
    pub kind: String,
    pub token_id: String,
    pub parent: Option<TokenLineage>,
    pub causation_token_id: Option<String>,
}

/// Options for [`spawn_stateful_child_token`].
pub struct SpawnStatefulChildTokenOpts<S> {
    pub kind: String,
    pub token_id: String,
    pub initial_state: S,
    pub tick: f64,
    pub station_id: String,
    pub event: Option<String>,
    pub detail: Option<String>,
}

/// Options for [`transition_token`].
pub struct TransitionTokenOpts {
    pub tick: f64,
    pub station_id: String,
    pub event: String,
    pub detail: Option<String>,
}

/// Build a root stateful token with an initial recorded transition.
pub fn make_stateful_token<S: Clone>(opts: MakeStatefulTokenOpts<S>) -> StatefulToken<S> {
    let state = opts.initial_state;
    StatefulToken {
        kind: opts.kind,
        lineage: TokenLineage {
            token_id: opts.token_id.clone(),
            parent_token_id: None,
            root_token_id: opts.token_id,
            causation_token_id: None,
            generation: 0,
        },
        state_mode: TokenStateMode::Stateful,
        current_state: Some(state.clone()),
        state_history: Some(vec![TokenStateTransition {
            tick: opts.tick,
            station_id: opts.station_id,
            from: None,
            to: state,
            event: opts.event.unwrap_or_else(|| "created".to_string()),
            detail: opts.detail,
        }]),
    }
}

/// Build a stateless token (no state machine), optionally as a child.
pub fn make_stateless_token<S>(opts: MakeStatelessTokenOpts) -> StatefulToken<S> {
    StatefulToken {
        kind: opts.kind,
        lineage: child_lineage(opts.token_id, opts.parent.as_ref(), opts.causation_token_id),
        state_mode: TokenStateMode::Stateless,
        current_state: None,
        state_history: None,
    }
}

/// Spawn a stateful child token whose lineage descends from `parent`.
pub fn spawn_stateful_child_token<S: Clone>(
    parent: &TokenLineage,
    opts: SpawnStatefulChildTokenOpts<S>,
) -> StatefulToken<S> {
    let state = opts.initial_state;
    StatefulToken {
        kind: opts.kind,
        lineage: child_lineage(opts.token_id, Some(parent), Some(parent.token_id.clone())),
        state_mode: TokenStateMode::Stateful,
        current_state: Some(state.clone()),
        state_history: Some(vec![TokenStateTransition {
            tick: opts.tick,
            station_id: opts.station_id,
            from: None,
            to: state,
            event: opts.event.unwrap_or_else(|| "spawned".to_string()),
            detail: opts.detail,
        }]),
    }
}

/// Apply a state transition in place. No-op for stateless tokens.
pub fn transition_token<S: Clone>(token: &mut StatefulToken<S>, next_state: S, opts: TransitionTokenOpts) {
    if token.state_mode != TokenStateMode::Stateful {
        return;
    }
    let from = token.current_state.clone();
    token.current_state = Some(next_state.clone());
    let history = token.state_history.get_or_insert_with(Vec::new);
    history.push(TokenStateTransition {
        tick: opts.tick,
        station_id: opts.station_id,
        from,
        to: next_state,
        event: opts.event,
        detail: opts.detail,
    });
}

/// Compute the lineage for a (possibly child) token.
pub fn child_lineage(
    token_id: String,
    parent: Option<&TokenLineage>,
    causation_token_id: Option<String>,
) -> TokenLineage {
    match parent {
        None => TokenLineage {
            token_id: token_id.clone(),
            parent_token_id: None,
            root_token_id: token_id,
            causation_token_id,
            generation: 0,
        },
        Some(p) => TokenLineage {
            token_id,
            parent_token_id: Some(p.token_id.clone()),
            root_token_id: p.root_token_id.clone(),
            causation_token_id,
            generation: p.generation + 1,
        },
    }
}

/// Best-effort analogue of the TS `isStatefulToken` type guard: succeeds when
/// `t` is a concrete `StatefulToken<S>`.
pub fn is_stateful_token<S: 'static>(t: &dyn Any) -> bool {
    t.is::<StatefulToken<S>>()
}

/// Options for [`PayloadStatefulToken::new`].
pub struct PayloadStatefulTokenOpts<S, P> {
    pub kind: String,
    pub token_id: String,
    pub payload: P,
    pub initial_state: S,
    pub tick: f64,
    pub station_id: String,
    pub event: Option<String>,
    pub detail: Option<String>,
    pub parent: Option<TokenLineage>,
    pub causation_token_id: Option<String>,
    pub state_mode: Option<TokenStateMode>,
}

/// A stateful token carrying a typed `payload`.
#[derive(Clone, Debug, PartialEq)]
pub struct PayloadStatefulToken<S = String, P = ()> {
    pub base: StatefulToken<S>,
    pub payload: P,
}

impl<S: Clone, P> PayloadStatefulToken<S, P> {
    pub fn new(opts: PayloadStatefulTokenOpts<S, P>) -> Self {
        let PayloadStatefulTokenOpts {
            kind,
            token_id,
            payload,
            initial_state,
            tick,
            station_id,
            event,
            detail,
            parent,
            causation_token_id,
            state_mode,
        } = opts;

        let base = if state_mode == Some(TokenStateMode::Stateless) {
            make_stateless_token::<S>(MakeStatelessTokenOpts { kind, token_id, parent, causation_token_id })
        } else if let Some(parent_lineage) = parent {
            spawn_stateful_child_token(
                &parent_lineage,
                SpawnStatefulChildTokenOpts { kind, token_id, initial_state, tick, station_id, event, detail },
            )
        } else {
            make_stateful_token(MakeStatefulTokenOpts {
                kind,
                token_id,
                initial_state,
                tick,
                station_id,
                event,
                detail,
            })
        };

        PayloadStatefulToken { base, payload }
    }
}

/// Tracks tokens by id and produces aggregate statistics.
#[derive(Default)]
pub struct StatefulTokenRegistry {
    tokens: HashMap<String, Box<dyn TrackedToken>>,
    by_kind: HashMap<String, u64>,
    created: u64,
    stateful: u64,
    stateless: u64,
    max_generation: usize,
}

impl StatefulTokenRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a token. Re-tracking an existing id updates the stored handle
    /// without re-counting (faithful to the TS dedup-by-id behaviour).
    pub fn track<T: TrackedToken + 'static>(&mut self, t: T) {
        let id = t.lineage().token_id.clone();
        let kind = t.kind().to_string();
        let mode = t.state_mode();
        let generation = t.lineage().generation;
        match self.tokens.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // Re-tracking an existing id updates the stored handle without
                // re-counting (faithful to the TS dedup-by-id behaviour).
                e.insert(Box::new(t));
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(Box::new(t));
                self.created += 1;
                *self.by_kind.entry(kind).or_insert(0) += 1;
                if mode == TokenStateMode::Stateful {
                    self.stateful += 1;
                } else {
                    self.stateless += 1;
                }
                self.max_generation = self.max_generation.max(generation);
            }
        }
    }

    pub fn snapshot(&self) -> StatefulTokenRegistryStats {
        let mut state_transitions = 0u64;
        for t in self.tokens.values() {
            state_transitions += t.state_history_len() as u64;
        }
        StatefulTokenRegistryStats {
            created: self.created,
            stateful: self.stateful,
            stateless: self.stateless,
            state_transitions,
            max_generation: self.max_generation,
            by_kind: self.by_kind.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stateful_token_records_transitions() {
        let mut tok = make_stateful_token(MakeStatefulTokenOpts {
            kind: "job".to_string(),
            token_id: "t1".to_string(),
            initial_state: "queued".to_string(),
            tick: 0.0,
            station_id: "src".to_string(),
            event: None,
            detail: None,
        });
        assert_eq!(tok.current_state.as_deref(), Some("queued"));
        assert_eq!(tok.state_history.as_ref().unwrap().len(), 1);

        transition_token(
            &mut tok,
            "running".to_string(),
            TransitionTokenOpts { tick: 1.0, station_id: "proc".to_string(), event: "start".to_string(), detail: None },
        );
        assert_eq!(tok.current_state.as_deref(), Some("running"));
        let history = tok.state_history.as_ref().unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].from.as_deref(), Some("queued"));
        assert_eq!(history[1].to, "running");
    }

    #[test]
    fn child_lineage_increments_generation() {
        let root = make_stateful_token(MakeStatefulTokenOpts {
            kind: "root".to_string(),
            token_id: "r".to_string(),
            initial_state: "a".to_string(),
            tick: 0.0,
            station_id: "s".to_string(),
            event: None,
            detail: None,
        });
        let child = spawn_stateful_child_token(
            &root.lineage,
            SpawnStatefulChildTokenOpts {
                kind: "child".to_string(),
                token_id: "c".to_string(),
                initial_state: "a".to_string(),
                tick: 1.0,
                station_id: "s".to_string(),
                event: None,
                detail: None,
            },
        );
        assert_eq!(child.lineage.generation, 1);
        assert_eq!(child.lineage.parent_token_id.as_deref(), Some("r"));
        assert_eq!(child.lineage.root_token_id, "r");
        assert_eq!(child.lineage.causation_token_id.as_deref(), Some("r"));
    }

    #[test]
    fn registry_snapshot_counts() {
        let mut reg = StatefulTokenRegistry::new();
        let a = make_stateful_token(MakeStatefulTokenOpts {
            kind: "job".to_string(),
            token_id: "a".to_string(),
            initial_state: "x".to_string(),
            tick: 0.0,
            station_id: "s".to_string(),
            event: None,
            detail: None,
        });
        let b = make_stateless_token::<String>(MakeStatelessTokenOpts {
            kind: "signal".to_string(),
            token_id: "b".to_string(),
            parent: Some(a.lineage.clone()),
            causation_token_id: None,
        });
        reg.track(a.clone());
        reg.track(a); // re-track same id: no double count
        reg.track(b);

        let stats = reg.snapshot();
        assert_eq!(stats.created, 2);
        assert_eq!(stats.stateful, 1);
        assert_eq!(stats.stateless, 1);
        assert_eq!(stats.state_transitions, 1);
        assert_eq!(stats.max_generation, 1);
        assert_eq!(stats.by_kind.get("job"), Some(&1));
        assert_eq!(stats.by_kind.get("signal"), Some(&1));
    }
}
