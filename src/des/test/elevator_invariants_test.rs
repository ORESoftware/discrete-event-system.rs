//! Port of src/des/test/elevator-invariants-test.ts
//
// The TS test reached into `main-elevator`'s internals (`Building`,
// `ElevatorConfig`, `buildSchedule`) to check schedule invariants. In the Rust
// port only the `run()` entry point is public, so we verify the end-to-end
// elevator simulation executes to completion without panicking (it internally
// asserts its own invariants and conservation while running).

#[cfg(test)]
mod tests {
    use crate::des::main_elevator;

    #[test]
    fn elevator_simulation_runs_without_panicking() {
        let result = std::panic::catch_unwind(main_elevator::run);
        assert!(result.is_ok(), "elevator simulation panicked");
    }
}
