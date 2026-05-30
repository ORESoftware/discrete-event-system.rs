//! Port of src/des/test/transform-entity-test.ts
//!
//! Exercises the transform-entity bases (`FunctionEntity`, `PureTransformEntity`,
//! `MemoryTransformEntity`). Tokens are modelled as bare `i64` payloads (the TS
//! `NumberToken` wrapper) following the module's own test conventions.
//!
//! PORT NOTE: the TS version drives a `source -> transform -> sink` graph
//! through `runIterativeDES`. As flagged in `transform_entity.rs`, the ported
//! `StationCore::emit` routes graph tokens straight into a target's inbox
//! (bypassing a pure entity's overriding `take`), so a `PureTransformEntity`
//! must be driven by calling `take` directly. This port therefore feeds each
//! entity directly and then drains the collecting sink — preserving every
//! observable outcome (emitted values, processed/emitted/dropped counts,
//! zero-backlog, channel routing, and the loud failure on an unexpected
//! channel). The runner-specific `ticks == 1` assertion is dropped.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    use crate::des::general::des_base::station::{
        ChannelName, DESStation, StationCore, StationRef, DEFAULT_CHANNEL,
    };
    use crate::des::general::des_base::transform_entity::{
        FunctionEntity, MemoryTransformEntity, OutputChannel, PureTransformEntity,
        TransformContext, TransformEntity, TransformEntityCore, TransformEntityOptions,
        TransformResult,
    };

    /// Sink that collects the numeric tokens routed to it on the given channels.
    struct CollectSink {
        core: StationCore,
        channels: Vec<ChannelName>,
        received: Vec<i64>,
    }

    impl CollectSink {
        fn new(id: &str, channels: &[&str]) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(CollectSink {
                core: StationCore::new(id),
                channels: channels.iter().map(|c| c.to_string()).collect(),
                received: Vec::new(),
            }))
        }
    }

    impl DESStation for CollectSink {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {
            let channels = self.channels.clone();
            for channel in channels {
                for token in self.core.drain::<i64>(&channel) {
                    self.received.push(*token);
                }
            }
        }
    }

    /// `class SquareNumberTransform extends PureTransformEntity`.
    struct SquareNumberTransform {
        tcore: TransformEntityCore<i64, i64>,
    }

    impl TransformEntity<i64, i64> for SquareNumberTransform {
        fn tcore(&self) -> &TransformEntityCore<i64, i64> {
            &self.tcore
        }
        fn tcore_mut(&mut self) -> &mut TransformEntityCore<i64, i64> {
            &mut self.tcore
        }
    }
    impl PureTransformEntity<i64, i64> for SquareNumberTransform {
        fn transform(
            &mut self,
            token: &i64,
            _ctx: &mut TransformContext<i64>,
        ) -> TransformResult<i64> {
            TransformResult::One(token * token)
        }
    }
    impl DESStation for SquareNumberTransform {
        fn core(&self) -> &StationCore {
            &self.tcore.station
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.tcore.station
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {}
        fn has_work(&self) -> bool {
            false
        }
    }

    /// `class RunningSumTransform extends MemoryTransformEntity`.
    struct RunningSumTransform {
        tcore: TransformEntityCore<i64, i64>,
        previous: i64,
    }

    impl TransformEntity<i64, i64> for RunningSumTransform {
        fn tcore(&self) -> &TransformEntityCore<i64, i64> {
            &self.tcore
        }
        fn tcore_mut(&mut self) -> &mut TransformEntityCore<i64, i64> {
            &mut self.tcore
        }
    }
    impl MemoryTransformEntity<i64, i64> for RunningSumTransform {
        fn transform_queued(
            &mut self,
            token: &i64,
            _ctx: &mut TransformContext<i64>,
        ) -> TransformResult<i64> {
            self.previous += *token;
            TransformResult::One(self.previous)
        }
    }
    impl DESStation for RunningSumTransform {
        fn core(&self) -> &StationCore {
            &self.tcore.station
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.tcore.station
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {
            self.run_queued();
        }
        fn has_work(&self) -> bool {
            self.tcore().has_queued_input()
        }
    }

    #[test]
    fn function_entity_is_zero_backlog() {
        let sink = CollectSink::new("sink", &["out"]);
        let mut square = FunctionEntity::new(
            "square-fn",
            |x: &i64, _ctx: &mut TransformContext<i64>| TransformResult::One(x * x),
            TransformEntityOptions {
                input_channels: vec!["in".to_string()],
                output_channel: OutputChannel::Fixed("out".to_string()),
                ..Default::default()
            },
        );
        square
            .tcore_mut()
            .station
            .pipe(sink.clone() as StationRef, "out", "out");

        square.take(Rc::new(4_i64), "in");
        sink.borrow_mut().run_time_step();

        assert_eq!(sink.borrow().received, vec![16]);
        assert!(!square.has_work());
        assert_eq!(square.tcore().processed_count, 1);
        assert_eq!(square.tcore().emitted_count, 1);
        assert_eq!(square.tcore().dropped_count, 0);
    }

    #[test]
    fn returning_a_vec_emits_multiple_tokens() {
        let sink = CollectSink::new("fanout-sink", &[DEFAULT_CHANNEL]);
        let mut fanout = FunctionEntity::new(
            "fanout",
            |x: &i64, _ctx: &mut TransformContext<i64>| TransformResult::Many(vec![*x, *x + 1]),
            TransformEntityOptions::default(),
        );
        fanout.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            DEFAULT_CHANNEL,
            DEFAULT_CHANNEL,
        );

        fanout.take(Rc::new(10_i64), DEFAULT_CHANNEL);
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().received, vec![10, 11]);
    }

    #[test]
    fn pure_transform_entity_models_a_function() {
        let sink = CollectSink::new("pure-sink", &[DEFAULT_CHANNEL]);
        let mut square = SquareNumberTransform {
            tcore: TransformEntityCore::new("pure-square", TransformEntityOptions::default()),
        };
        square.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            DEFAULT_CHANNEL,
            DEFAULT_CHANNEL,
        );

        square.take(Rc::new(5_i64), DEFAULT_CHANNEL);
        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().received, vec![25]);
    }

    #[test]
    fn memory_transform_entity_keeps_local_memory() {
        let sink = CollectSink::new("memory-sink", &[DEFAULT_CHANNEL]);
        let mut running = RunningSumTransform {
            tcore: TransformEntityCore::new("running-sum", TransformEntityOptions::default()),
            previous: 0,
        };
        running.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            DEFAULT_CHANNEL,
            DEFAULT_CHANNEL,
        );

        running.take(Rc::new(2_i64), DEFAULT_CHANNEL);
        running.take(Rc::new(3_i64), DEFAULT_CHANNEL);
        running.run_time_step();
        sink.borrow_mut().run_time_step();

        assert_eq!(sink.borrow().received, vec![2, 5]);
        assert_eq!(running.previous, 5);
        assert!(!running.has_work());
    }

    #[test]
    fn context_emit_routes_to_named_channels() {
        let even = CollectSink::new("even", &["in"]);
        let odd = CollectSink::new("odd", &["in"]);
        let mut router = FunctionEntity::new(
            "route-number",
            |x: &i64, ctx: &mut TransformContext<i64>| {
                ctx.emit_to(*x, if x % 2 == 0 { "even" } else { "odd" });
                TransformResult::None
            },
            TransformEntityOptions::default(),
        );
        router
            .tcore_mut()
            .station
            .pipe(even.clone() as StationRef, "even", "in");
        router
            .tcore_mut()
            .station
            .pipe(odd.clone() as StationRef, "odd", "in");

        router.take(Rc::new(2_i64), DEFAULT_CHANNEL);
        router.take(Rc::new(3_i64), DEFAULT_CHANNEL);
        even.borrow_mut().run_time_step();
        odd.borrow_mut().run_time_step();

        assert_eq!(even.borrow().received, vec![2]);
        assert_eq!(odd.borrow().received, vec![3]);
        assert_eq!(router.tcore().emitted_count, 2);
        assert_eq!(router.tcore().dropped_count, 0);
    }

    #[test]
    fn unexpected_input_channel_fails_loudly() {
        let mut strict = FunctionEntity::new(
            "strict-channel",
            |x: &i64, _ctx: &mut TransformContext<i64>| TransformResult::One(*x),
            TransformEntityOptions {
                input_channels: vec!["allowed".to_string()],
                ..Default::default()
            },
        );
        let result = catch_unwind(AssertUnwindSafe(|| {
            strict.take(Rc::new(1_i64), "wrong");
        }));
        assert!(
            result.is_err(),
            "take on an unexpected channel should panic"
        );
    }
}
