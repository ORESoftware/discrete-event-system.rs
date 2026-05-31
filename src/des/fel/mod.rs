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
//!
//! Nothing here modifies the existing engine; the comparison wraps it through
//! its public API only.

pub mod compare;
pub mod engine;
pub mod mm1;
pub mod time_stepped_mm1;
