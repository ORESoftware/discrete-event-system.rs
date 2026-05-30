//! Port of `src/des/general/do-audit.ts` — module `des::general::do_audit`.
//!
//! Debug invariant check: the total registered-entity buffer size must stay
//! constant across audits. The first audit records a baseline; every later
//! audit compares against the previous total and panics (an invariant
//! violation, the TS `throw makeError(...)`) when they disagree.
//!
//! Conversion notes from the TS source:
//!   * The TS module-level mutable state `first` / `previousTotal` is held in a
//!     `thread_local!` cell rather than as file-level `static mut` (matching the
//!     way `entity_registration` mirrors the global `reg` singleton).
//!   * The TS pulled the process-wide `reg` singleton directly; here we call the
//!     `entity_registration` free-function facade (`get_all_*`).
//!   * `throw makeError('totals are not equal', ...)` becomes a `panic!`.

#![allow(dead_code)]

use std::cell::Cell;

use crate::des::general::entity_registration::{
    get_all_decision_nodes, get_all_processors, get_all_sinks, EntityHandle,
};
use crate::des::general::general::make_error;

thread_local! {
    static FIRST: Cell<bool> = const { Cell::new(true) };
    static PREVIOUS_TOTAL: Cell<i64> = const { Cell::new(0) };
}

/// Per-entity audit snapshot — the TS `entity.doAudit()` return shape. Only the
/// `total_size` field is consulted here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntityAuditResult {
    pub total_size: i64,
}

/// Entities that can report their buffered-item totals to the auditor. The TS
/// `EntityProcessor` / `EntitySink` / `ProbabilityDecisionEntity` all expose a
/// `doAudit()` method returning `{ totalSize }`.
pub trait Auditable {
    fn do_audit(&self) -> EntityAuditResult;
}

// PORT NOTE: the registry currently hands out type-erased `Rc<RefCell<dyn
// HasId>>` handles (see `entity_registration`), and `HasId` does not expose
// `do_audit()`. The concrete processor/sink/decision entity types that
// implement `Auditable` are not ported yet, so this helper contributes 0 for
// each handle. Once those entities exist, downcast each handle to `&dyn
// Auditable` and sum `do_audit().total_size`.
fn audit_handle(_handle: &EntityHandle) -> i64 {
    0
}

/// Run one audit pass. Panics if the total registered-entity buffer size has
/// changed since the previous audit (after the first, baseline-establishing
/// call).
pub fn do_audit() {
    let mut total: i64 = 0;

    for v in get_all_processors() {
        total += audit_handle(&v);
    }
    for v in get_all_sinks() {
        total += audit_handle(&v);
    }
    for v in get_all_decision_nodes() {
        total += audit_handle(&v);
    }

    let first = FIRST.with(|f| f.get());
    if !first {
        let previous_total = PREVIOUS_TOTAL.with(|p| p.get());
        if previous_total != total {
            panic!(
                "{}",
                make_error(&format!(
                    "totals are not equal (previous={previous_total}, current={total})"
                ))
            );
        }
    }

    PREVIOUS_TOTAL.with(|p| p.set(total));
    FIRST.with(|f| f.set(false));
}

/// Reset the auditor's baseline (test helper; the TS relied on fresh module
/// state per process).
pub fn reset_audit() {
    FIRST.with(|f| f.set(true));
    PREVIOUS_TOTAL.with(|p| p.set(0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_is_stable_across_calls() {
        reset_audit();
        // First call establishes the baseline; subsequent calls must agree.
        do_audit();
        do_audit();
        do_audit();
    }

    #[test]
    fn auditable_trait_reports_total_size() {
        struct Proc {
            size: i64,
        }
        impl Auditable for Proc {
            fn do_audit(&self) -> EntityAuditResult {
                EntityAuditResult {
                    total_size: self.size,
                }
            }
        }
        let p = Proc { size: 7 };
        assert_eq!(p.do_audit().total_size, 7);
    }
}
