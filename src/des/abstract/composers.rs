//! Canonical use path: `crate::des::r#abstract::composers::*`
//!
//! Port of `src/des/abstract/composers.ts` — reusable behaviour "mixins"
//! composed into stationary entities. (`abstract` is a Rust keyword, hence
//! `r#abstract`; the file stays `composers.rs`.)
//!
//! The TS `DoesFanOut<V>` held a `HasManyOutputConnections<any,any>` and routed a
//! moving entity to the first downstream target that accepts it. The unused
//! generic `<V>` is dropped; the held node is a trait object
//! `Rc<RefCell<dyn HasManyOutputConnections>>`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::r#abstract::interfaces::HasManyOutputConnections;

/// Result of a fan-out attempt (TS `{accepted: boolean}`).
#[derive(Clone, Copy, Debug, Default)]
pub struct FanOutResult {
    pub accepted: bool,
}

/// `class DoesFanOut` — delegate a node's outbound routing.
pub struct DoesFanOut {
    pub entity: Rc<RefCell<dyn HasManyOutputConnections>>,
}

impl DoesFanOut {
    /// TS constructor took `{entity}`; here it is the field directly (no `opts` twin).
    pub fn new(entity: Rc<RefCell<dyn HasManyOutputConnections>>) -> Self {
        DoesFanOut { entity }
    }

    /// `doFanOut(ame)` — offer `ame` to each outbound target in turn, handing it
    /// off to the first that accepts.
    ///
    /// Borrows are taken and released step-by-step (snapshot the connection list,
    /// then the target handle, then borrow the target mutably) so no two
    /// `RefCell` borrows overlap.
    pub fn do_fan_out(&self, ame: Rc<RefCell<dyn MovingEntity>>) -> FanOutResult {
        let mut accepted = false;

        // Snapshot the outbound connections (Rc clones), releasing the borrow of
        // `self.entity` before touching any target.
        let connections = self.entity.borrow().get_out_connections();

        for conn in connections {
            // Resolve and release the connection borrow before touching the target.
            let target = conn.borrow().get_target();
            let target = match target {
                Some(t) => t,
                None => {
                    eprintln!("warning: could not find target.");
                    continue;
                }
            };

            accepted = target.borrow_mut().accept_item(ame.clone());
            if accepted {
                target.borrow_mut().take_item(ame.clone());
                break;
            }
        }

        FanOutResult { accepted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::r#abstract::interfaces::{HasOutput, HasInput};

    // A trivial node with no outbound connections: fan-out should report "not accepted".
    struct EmptyNode;
    impl HasOutput for EmptyNode {
        fn id(&self) -> String {
            "empty".into()
        }
        fn add_out_connection(
            &mut self,
            _target: Rc<RefCell<dyn HasInput>>,
        ) -> Option<Rc<RefCell<crate::des::r#abstract::r#abstract::EntityConnection>>> {
            None
        }
        fn do_setup_after_input_conn(&mut self) -> bool {
            true
        }
        fn notify_targets(&mut self) {}
        fn do_setup_after_output_conn(&mut self) -> bool {
            true
        }
    }
    impl HasManyOutputConnections for EmptyNode {
        fn get_out_connections(
            &self,
        ) -> Vec<Rc<RefCell<crate::des::r#abstract::r#abstract::EntityConnection>>> {
            Vec::new()
        }
    }

    #[test]
    fn fan_out_with_no_targets_is_not_accepted() {
        let node: Rc<RefCell<dyn HasManyOutputConnections>> =
            Rc::new(RefCell::new(EmptyNode));
        let fan = DoesFanOut::new(node);
        let me = crate::des::entity_moving::moving::BasicMovingEntity::new();
        let me: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(me));
        let res = fan.do_fan_out(me);
        assert!(!res.accepted);
    }
}
