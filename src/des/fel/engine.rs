//! A classic **Future Event List** (FEL) discrete-event engine.
//!
//! This is the *next-event time-advance* paradigm: the simulation clock jumps
//! directly from one scheduled event to the next (no fixed time step). Events
//! live in a time-ordered priority queue (the FEL); processing an event may
//! schedule further events. Between events nothing happens, so the clock can
//! leap over long idle stretches in a single step.
//!
//! Contrast with the engine's existing **time-stepped** entity network
//! (`entity_source` / `entity_processing` / `entity_sink`), which advances every
//! station by a fixed Δt of real seconds each tick regardless of whether
//! anything is happening. See [`crate::des::fel::compare`] for a head-to-head.
//!
//! Generic over a user `World` state. Events are `FnOnce(&mut Engine<World>)`
//! closures, so an event can both mutate the world and schedule successors. The
//! event is removed from the FEL before it runs, so there is no aliasing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

/// Invalid simulation-clock input. Rejecting these values preserves the engine's
/// fundamental invariant that logical time is finite and monotonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationTimeError {
    NonFinite,
    BeforeCurrentTime,
}

impl fmt::Display for SimulationTimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationTimeError::NonFinite => formatter.write_str("simulation time must be finite"),
            SimulationTimeError::BeforeCurrentTime => {
                formatter.write_str("simulation time cannot precede the current clock")
            }
        }
    }
}

impl std::error::Error for SimulationTimeError {}

/// Failure to admit an event into the future-event list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    Time(SimulationTimeError),
    SequenceExhausted,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::Time(error) => error.fmt(formatter),
            ScheduleError::SequenceExhausted => {
                formatter.write_str("event insertion sequence is exhausted")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

impl From<SimulationTimeError> for ScheduleError {
    fn from(error: SimulationTimeError) -> Self {
        Self::Time(error)
    }
}

/// One scheduled future event: run `action` at simulated time `time`.
/// `seq` is a monotonic insertion counter giving FIFO order among events with
/// identical timestamps (so the schedule is fully deterministic).
struct Scheduled<W> {
    time: f64,
    seq: u64,
    action: Box<dyn FnOnce(&mut Engine<W>)>,
}

impl<W> PartialEq for Scheduled<W> {
    fn eq(&self, other: &Self) -> bool {
        self.time.total_cmp(&other.time) == Ordering::Equal && self.seq == other.seq
    }
}
impl<W> Eq for Scheduled<W> {}

impl<W> Ord for Scheduled<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap, but we want the EARLIEST event to pop
        // first, so invert: smaller time => "greater". Ties break on the lower
        // sequence number (FIFO).
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl<W> PartialOrd for Scheduled<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The FEL engine: a simulated clock, the event queue, and the user world.
pub struct Engine<W> {
    clock: f64,
    seq: u64,
    fel: BinaryHeap<Scheduled<W>>,
    processed: u64,
    /// User-defined simulation state, mutated by event handlers.
    pub world: W,
}

impl<W> Engine<W> {
    pub fn new(world: W) -> Self {
        Engine {
            clock: 0.0,
            seq: 0,
            fel: BinaryHeap::new(),
            processed: 0,
            world,
        }
    }

    /// Current simulated time.
    pub fn now(&self) -> f64 {
        self.clock
    }

    /// Number of events processed so far (the FEL work metric).
    pub fn events_processed(&self) -> u64 {
        self.processed
    }

    /// Number of events still queued.
    pub fn pending(&self) -> usize {
        self.fel.len()
    }

    fn checked_future_time(&self, time: f64) -> Result<f64, SimulationTimeError> {
        match (time.is_finite(), time >= self.clock) {
            (false, false) | (false, true) => Err(SimulationTimeError::NonFinite),
            (true, false) => Err(SimulationTimeError::BeforeCurrentTime),
            // IEEE -0.0 and +0.0 represent the same logical instant. Canonicalizing
            // them also makes timestamp equality agree with FIFO tie-breaking.
            (true, true) if time == 0.0 => Ok(0.0),
            (true, true) => Ok(time),
        }
    }

    /// Checked form of [`Engine::schedule_at`]. The queue is unchanged on failure.
    pub fn try_schedule_at(
        &mut self,
        time: f64,
        action: impl FnOnce(&mut Engine<W>) + 'static,
    ) -> Result<(), ScheduleError> {
        let time = self.checked_future_time(time)?;
        let next_seq = self
            .seq
            .checked_add(1)
            .ok_or(ScheduleError::SequenceExhausted)?;
        let seq = self.seq;
        self.seq = next_seq;
        self.fel.push(Scheduled {
            time,
            seq,
            action: Box::new(action),
        });
        Ok(())
    }

    /// Schedule `action` to fire at absolute simulated time `time`.
    ///
    /// # Panics
    ///
    /// Panics if `time` is non-finite, precedes the current clock, or the deterministic
    /// insertion sequence is exhausted. Use [`Engine::try_schedule_at`] at an input boundary.
    #[track_caller]
    pub fn schedule_at(&mut self, time: f64, action: impl FnOnce(&mut Engine<W>) + 'static) {
        let now = self.clock;
        if let Err(error) = self.try_schedule_at(time, action) {
            panic!("cannot schedule event at {time:?} from clock {now:?}: {error}");
        }
    }

    /// Checked form of [`Engine::schedule_after`]. Negative finite delays retain the
    /// historical behavior of scheduling immediately; non-finite values are rejected.
    pub fn try_schedule_after(
        &mut self,
        delay: f64,
        action: impl FnOnce(&mut Engine<W>) + 'static,
    ) -> Result<(), ScheduleError> {
        if !delay.is_finite() {
            return Err(SimulationTimeError::NonFinite.into());
        }
        let time = self.clock + if delay > 0.0 { delay } else { 0.0 };
        self.try_schedule_at(time, action)
    }

    /// Schedule `action` to fire `delay` time units from now (clamped to ≥ 0).
    ///
    /// # Panics
    ///
    /// Panics for a non-finite delay, an overflowing target time, or an exhausted
    /// insertion sequence. Use [`Engine::try_schedule_after`] at an input boundary.
    #[track_caller]
    pub fn schedule_after(&mut self, delay: f64, action: impl FnOnce(&mut Engine<W>) + 'static) {
        let now = self.clock;
        if let Err(error) = self.try_schedule_after(delay, action) {
            panic!("cannot schedule event after {delay:?} from clock {now:?}: {error}");
        }
    }

    /// Checked form of [`Engine::run_until`]. No event is processed on an invalid horizon.
    pub fn try_run_until(&mut self, end: f64) -> Result<(), SimulationTimeError> {
        let end = self.checked_future_time(end)?;
        while let Some(next) = self.fel.peek() {
            if next.time > end {
                break;
            }
            let ev = self.fel.pop().expect("peeked event must pop");
            debug_assert!(ev.time >= self.clock);
            self.clock = ev.time;
            self.processed += 1;
            (ev.action)(self);
        }
        if self.clock < end {
            self.clock = end;
        }
        Ok(())
    }

    /// Process events in time order until the FEL is empty or the next event is
    /// past `end`. The clock is advanced to `end` on return so callers can close
    /// any time-weighted accumulators over the final interval.
    ///
    /// # Panics
    ///
    /// Panics if `end` is non-finite or precedes the current clock. Use
    /// [`Engine::try_run_until`] when the horizon is supplied by an external caller.
    #[track_caller]
    pub fn run_until(&mut self, end: f64) {
        let now = self.clock;
        if let Err(error) = self.try_run_until(end) {
            panic!("cannot run to horizon {end:?} from clock {now:?}: {error}");
        }
    }

    /// Process every queued event until the FEL drains.
    pub fn run(&mut self) {
        while let Some(ev) = self.fel.pop() {
            debug_assert!(ev.time >= self.clock);
            self.clock = ev.time;
            self.processed += 1;
            (ev.action)(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_fire_in_time_order_not_insertion_order() {
        let mut eng: Engine<Vec<f64>> = Engine::new(Vec::new());
        // Insert out of order; they must come out sorted by time.
        eng.schedule_at(3.0, |e| e.world.push(e.now()));
        eng.schedule_at(1.0, |e| e.world.push(e.now()));
        eng.schedule_at(2.0, |e| e.world.push(e.now()));
        eng.run();
        assert_eq!(eng.world, vec![1.0, 2.0, 3.0]);
        assert_eq!(eng.events_processed(), 3);
        assert_eq!(eng.now(), 3.0);
    }

    #[test]
    fn same_time_events_are_fifo() {
        let mut eng: Engine<Vec<u32>> = Engine::new(Vec::new());
        eng.schedule_at(1.0, |e| e.world.push(1));
        eng.schedule_at(1.0, |e| e.world.push(2));
        eng.schedule_at(1.0, |e| e.world.push(3));
        eng.run();
        assert_eq!(eng.world, vec![1, 2, 3]);
    }

    #[test]
    fn events_can_schedule_successors() {
        // A self-rescheduling "tick" that fires 5 times at unit spacing.
        let mut eng: Engine<u32> = Engine::new(0);
        fn tick(e: &mut Engine<u32>) {
            e.world += 1;
            if e.world < 5 {
                e.schedule_after(1.0, tick);
            }
        }
        eng.schedule_after(1.0, tick);
        eng.run();
        assert_eq!(eng.world, 5);
        assert_eq!(eng.now(), 5.0);
    }

    #[test]
    fn run_until_stops_at_horizon_and_advances_clock() {
        let mut eng: Engine<Vec<f64>> = Engine::new(Vec::new());
        eng.schedule_at(1.0, |e| e.world.push(e.now()));
        eng.schedule_at(5.0, |e| e.world.push(e.now())); // beyond horizon
        eng.run_until(3.0);
        assert_eq!(eng.world, vec![1.0]); // event at 5.0 not processed
        assert_eq!(eng.now(), 3.0); // clock parked at the horizon
        assert_eq!(eng.pending(), 1); // the 5.0 event is still queued
    }

    #[test]
    fn invalid_time_inputs_fail_without_mutating_clock_or_queue() {
        let mut eng: Engine<()> = Engine::new(());

        assert_eq!(
            eng.try_schedule_at(f64::NAN, |_| {}),
            Err(ScheduleError::Time(SimulationTimeError::NonFinite))
        );
        assert_eq!(
            eng.try_schedule_at(f64::INFINITY, |_| {}),
            Err(ScheduleError::Time(SimulationTimeError::NonFinite))
        );
        assert_eq!(
            eng.try_schedule_after(f64::NEG_INFINITY, |_| {}),
            Err(ScheduleError::Time(SimulationTimeError::NonFinite))
        );
        assert_eq!(eng.pending(), 0);
        assert_eq!(eng.now(), 0.0);

        eng.try_run_until(2.0).expect("finite future horizon");
        assert_eq!(
            eng.try_schedule_at(1.0, |_| {}),
            Err(ScheduleError::Time(SimulationTimeError::BeforeCurrentTime))
        );
        assert_eq!(
            eng.try_run_until(1.0),
            Err(SimulationTimeError::BeforeCurrentTime)
        );
        assert_eq!(eng.pending(), 0);
        assert_eq!(eng.now(), 2.0);

        eng.seq = u64::MAX;
        assert_eq!(
            eng.try_schedule_at(2.0, |_| {}),
            Err(ScheduleError::SequenceExhausted)
        );
        assert_eq!(eng.pending(), 0);
    }

    #[test]
    fn negative_finite_delay_remains_an_immediate_event() {
        let mut eng: Engine<Vec<f64>> = Engine::new(Vec::new());
        eng.try_schedule_after(-1.0, |engine| engine.world.push(engine.now()))
            .expect("negative finite delay clamps to now");
        eng.run_until(0.0);
        assert_eq!(eng.world, vec![0.0]);
    }

    #[test]
    fn bounded_scheduler_model_checks_all_four_event_traces() {
        const TIMES: [f64; 4] = [-0.0, 0.0, 1.0, 2.0];
        const EVENT_COUNT: usize = 4;
        let trace_count = TIMES.len().pow(EVENT_COUNT as u32);

        for encoded_trace in 0..trace_count {
            let mut remaining = encoded_trace;
            let mut expected = Vec::with_capacity(EVENT_COUNT);
            let mut eng: Engine<Vec<(f64, usize)>> = Engine::new(Vec::new());

            for event_id in 0..EVENT_COUNT {
                let raw_time = TIMES[remaining % TIMES.len()];
                remaining /= TIMES.len();
                let canonical_time = if raw_time == 0.0 { 0.0 } else { raw_time };
                expected.push((canonical_time, event_id));
                eng.try_schedule_at(raw_time, move |engine| {
                    engine.world.push((engine.now(), event_id));
                })
                .expect("bounded time is admissible");
            }

            expected.sort_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            eng.run();

            assert_eq!(eng.world, expected, "trace {encoded_trace} misordered");
            assert!(eng
                .world
                .windows(2)
                .all(|window| window[0].0 <= window[1].0));
            assert_eq!(eng.events_processed(), EVENT_COUNT as u64);
            assert_eq!(eng.pending(), 0);
        }
    }
}
