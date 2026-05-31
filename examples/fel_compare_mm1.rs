//! Run the FEL-vs-time-step M/M/1 comparison and print the report.
//!
//! ```sh
//! cargo run --example fel_compare_mm1
//! ```
//!
//! This drives [`des_engine::des::fel::compare::run`], which simulates the same
//! M/M/1 queue two ways — an exact next-event (FEL) engine and the existing
//! fixed-time-step entity network at several Δt — and prints each estimate next
//! to the closed-form analytical truth plus the relative work each did. It makes
//! concrete *why* the FEL is the right tool for pure discrete queues (exact, far
//! less work) while the time-stepped engine remains the platform default for
//! mixed discrete/continuous and control workloads (see the `des::fel` and
//! `des::exec` module docs).

fn main() {
    des_engine::des::fel::compare::run();
}
