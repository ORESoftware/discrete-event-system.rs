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
        self.time == other.time && self.seq == other.seq
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

    /// Schedule `action` to fire at absolute simulated time `time`.
    pub fn schedule_at(&mut self, time: f64, action: impl FnOnce(&mut Engine<W>) + 'static) {
        let seq = self.seq;
        self.seq += 1;
        self.fel.push(Scheduled {
            time,
            seq,
            action: Box::new(action),
        });
    }

    /// Schedule `action` to fire `delay` time units from now (clamped to ≥ 0).
    pub fn schedule_after(&mut self, delay: f64, action: impl FnOnce(&mut Engine<W>) + 'static) {
        let time = self.clock + if delay > 0.0 { delay } else { 0.0 };
        self.schedule_at(time, action);
    }

    /// Process events in time order until the FEL is empty or the next event is
    /// past `end`. The clock is advanced to `end` on return so callers can close
    /// any time-weighted accumulators over the final interval.
    pub fn run_until(&mut self, end: f64) {
        while let Some(next) = self.fel.peek() {
            if next.time > end {
                break;
            }
            let ev = self.fel.pop().expect("peeked event must pop");
            self.clock = ev.time;
            self.processed += 1;
            (ev.action)(self);
        }
        if self.clock < end {
            self.clock = end;
        }
    }

    /// Process every queued event until the FEL drains.
    pub fn run(&mut self) {
        while let Some(ev) = self.fel.pop() {
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
}
