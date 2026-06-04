//! **Future Event List (FEL)** discrete-event simulation — a new, additive
//! engine alongside the existing time-stepped entity network.
//!
//! The engine has three "runnable" shapes (see [`crate::des::streaming`] for the
//! taxonomy). The existing `entity_source`/`entity_processing`/`entity_sink`
//! network is a **time-stepped DES**: it advances every station by a fixed Δt of
//! real seconds each tick. This module adds the classic **next-event** style:
//!
//! * [`engine`] — a generic FEL scheduler (a time-ordered event queue; the clock
//!   jumps from event to event, skipping idle time).
//! * [`mm1`] — an exact, event-driven M/M/1 queue built on the engine.
//! * [`time_stepped_mm1`] — the *same* M/M/1 built on the existing time-stepped
//!   entity engine (calls it, does not modify it).
//! * [`compare`] — runs both against the closed-form analytical M/M/1 and reports
//!   accuracy and work performed.
//! * [`elevator`] — a next-event elevator bank under a shared LOOK policy (a
//!   FEL model that does *not* touch the existing time-stepped elevator), plus
//!   canonical MDP/POMDP formulations of elevator dispatch that run on the
//!   first-class model citizens.
//!
//! Nothing here modifies the existing engine; the comparison wraps it through
//! its public API only.
//!
//! ## When FEL wins, and why time-stepping is still the platform default
//!
//! For a **pure discrete** queueing model (M/M/1), the FEL is strictly better:
//! it is *exact* (no Δt bias) and does far less work, because it jumps straight
//! to the next event and skips all the idle time the fixed-step engine must tick
//! through. The [`compare`] harness demonstrates exactly this.
//!
//! But this platform targets **mixed discrete/continuous** systems — control
//! loops, ODE plants, filters, multirate samplers — and there the fixed-time-step
//! engine is the better executive:
//!
//! * **Continuous dynamics have no "next event".** An ODE/SDE plant evolves at
//!   every instant; an integrator (RK4) needs to advance on a regular grid. There
//!   is nothing for a pure FEL to "jump" to between samples.
//! * **Feedback couples continuous state to discrete decisions.** A controller
//!   reads the plant, computes an action, and writes it back every sample period.
//!   A fixed step makes that sample-and-hold loop trivial; an FEL would have to
//!   schedule a synthetic event every Δt anyway — losing its only advantage.
//! * **Multirate is just more time steps.** Blocks at different rates land on
//!   their own sample hits within one stepped clock, which is awkward to express
//!   as an event queue.
//!
//! So the rule the [`crate::des::exec`] seam encodes is: route **pure discrete /
//! event-sparse** workloads to a next-event engine, and route **continuous,
//! feedback, or mixed** workloads to the time-stepped / [`crate::des::hybrid`]
//! executive (which adds RK4 integration, multirate discrete updates, and
//! zero-crossing event detection on top of a stepped clock). This module exists
//! to make that trade-off measurable rather than asserted.

pub mod compare;
pub mod elevator;
pub mod engine;
pub mod mm1;
pub mod time_stepped_mm1;
