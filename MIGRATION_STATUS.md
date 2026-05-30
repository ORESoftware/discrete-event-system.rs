# Migration Status

Generated on 2026-05-29 from the TypeScript repository at
`/Users/alexandermills/codes/ores/discrete-event-system`.

## Coverage

- TypeScript source files mapped: 386
- Rust library scaffolds from TS headers: 233
- Rust binary scaffolds: 95
- Rust integration-test scaffolds: 58

## Manually Ported Core Modules

- `src/core.rs`
- `src/migration.rs`
- `src/numeric.rs`
- `src/des/abstract/abstract.rs`
- `src/des/abstract/interfaces.rs`
- `src/des/general/general.rs`
- `src/des/general/des_base/transform_entity.rs`
- `src/des/general/des_base/argmax.rs`
- `src/des/general/des_base/fixed_point.rs`
- `src/des/general/des_base/preconditions.rs`
- `src/des/general/des_base/runner.rs`
- `src/des/general/des_base/station.rs`
- `src/des/general/des_base/validation.rs`
- `src/des/general/ode.rs`
- `src/des/general/network_flow.rs` (max-flow slice)
- `src/des/general/prng.rs`
- `src/des/general/quadrature.rs`
- `src/des/general/random_variables.rs`
- `src/des/general/root.rs`
- `src/des/general/shortest_path_des.rs`
- `src/des/entity_moving/moving.rs`
- `src/des/entity_processing/per_individual_processor.rs`
- `src/des/entity_queue/queue.rs`
- `src/des/entity_routing/output_routing_policy.rs`
- `src/des/entity_decision/decision.rs`
- `src/des/entity_decision/probability_decision.rs`
- `src/des/entity_decision/binary_decision.rs`
- `src/des/animation/types.rs`
- `src/des/animation/html_player.rs`
- `src/des/signals/signal_value.rs`
- `src/bin/main_build_site.rs` (sample HTML-generation CLI only)

## Behavioral Rust Ports

The following mapped TypeScript tests now contain behavioral Rust assertions
instead of scaffold-only registration checks:

- `tests/argmax_tiebreak_test.rs` covers the pure tie-break utility sections
  from `src/des/test/argmax-tiebreak-test.ts`.
- `tests/calculus_test.rs` covers the pure quadrature and ODE solver sections
  from `src/des/test/calculus-test.ts`; expression parsing and PDE/station
  sections remain pending their own mapped modules.
- `tests/output_routing_policy_test.rs` covers router policy behavior plus the
  `PerIndividualProcessor` routing groups from
  `src/des/test/output-routing-policy-test.ts`.
- `tests/preconditions_test.rs` covers the low-level `Preconditions.*` utility
  section from `src/des/test/preconditions-test.ts`; model-specific sections
  remain pending those mapped model ports.
- `tests/random_variables_test.rs` covers
  `src/des/test/random-variables-test.ts`.
- `tests/shortest_path_test.rs` covers
  `src/des/test/shortest-path-test.ts`; edge weights and finite distances are
  now exact `DesDecimal` values, with unreachable nodes represented as `None`.
- `tests/validation_test.rs` covers the validator factory primitives,
  `DESStation` validator wiring, `runIterativeDES` aggregation, finalization
  hooks, and fixed-point external-reference sections from
  `src/des/test/validation-test.ts`; algorithm-specific intrinsic validator
  sections remain pending those mapped model ports.
- `tests/network_flow_test.rs` covers the textbook max-flow / min-cut DES
  optimization slice from `src/des/test/network-flow-test.ts`; traffic-flow,
  smart-traffic, modular traffic, and stochastic-flow sections remain pending
  their own mapped modules.
- `tests/numeric_test.rs` covers the Rust numeric policy layer added for the
  migration: exact `Decimal` arithmetic, exact rational arithmetic,
  non-finite float rejection, and compensated f64 summation.
- `tests/probability_decision_test.rs` covers exact decimal probability
  validation and routing for `src/des/entity-decision/probability-decision.ts`.
- `tests/animation_test.rs` covers the migrated animation schema, self-contained
  HTML player generation, script/title escaping, chart/player markers, and
  multi-variant animation-set embedding from
  `src/des/test/animation-test.ts`.

## Numeric Policy

- Use `DesDecimal` / `rust_decimal::Decimal` for base-10 model-state values
  that are compared, serialized, accumulated, or exposed in DES results.
- Use `DesRational` / `num_rational::BigRational` for exact fractional values
  where repeated arithmetic must preserve fractions instead of approximating
  them.
- Keep `f64` for continuous numerical algorithms, geometry, random samples,
  and crate/library boundaries; cross into/out of exact math through
  `src/numeric.rs`.
- The max-flow port now uses `DesDecimal` for capacities, flows, residuals,
  bottlenecks, total max-flow, and min-cut capacity while preserving `f64` for
  node coordinates only.
- `PerIndividualProcessor` service durations and remaining processing time use
  `DesDecimal`, so DES time-step math stays exact.
- `shortest_path_des` uses `DesDecimal` for edge weights, wave distances, heap
  priorities, traces, and finite result distances; coordinates and random graph
  sampling stay at the `f64` boundary.
- `ProbabilityDecisionOpts` stores probabilities as `DesDecimal`; only the RNG
  sample crosses from `f64` into decimal at routing time.

## Runtime Parity Notes

- Rust can now generate a self-contained simulation HTML file through
  `src/des/animation/html_player.rs`; `src/bin/main_build_site.rs` writes a
  sample animation to a requested path.
- The full TypeScript site builder that regenerates every historical animation
  and report remains pending in Rust.
- External solver parity still lives in the TypeScript runner suite. The Rust
  `src/bin/compare_external_fel_models.rs` binary compiles and exits
  successfully, but it is still a scaffold and does not yet execute external
  adapters or compare outputs.

## Verification

Base PATH does not include `cargo`, `rustc`, or `rustfmt`, so verification was
run through a temporary Nix toolchain:

- `nix-shell -p cargo rustc rustfmt --run 'cargo fmt --check'`
- `nix-shell -p cargo rustc rustfmt --run 'cargo check --all-targets'`
- `nix-shell -p cargo rustc rustfmt --run 'cargo test --all-targets'`
- `nix-shell -p cargo rustc rustfmt --run 'cargo run --bin main_build_site -- /tmp/des_rs_simulation_demo.html'`

All commands pass as of 2026-05-30. The suite now includes a mix of scaffold
registration tests and the behavioral Rust ports listed above; remaining
scaffold tests still need module-by-module behavioral ports from the TypeScript
suite.

The TypeScript source baseline at
`/Users/alexandermills/codes/ores/discrete-event-system` was verified on
2026-05-30 for the modules touched in the Rust migration pass:

- `npm run build`
- `npm run test-calculus`
- `npm run test-output-routing`
- `npm run test-preconditions`
- `npm run test-shortest-path`
- `npm run test-argmax-tiebreak`
- `npx ts-node src/des/test/random-variables-test.ts`
- `npm run test-validation`
- `npm run test-network-flow`
- `node dist/des/test/animation-test.js`
- `npm run compare-external-fel`
- `npm run validate-external-fel-models`
- `npm run validate-smart-traffic-external`
- `npm run test-external-modules`

The full TypeScript `npm run test-*` sweep was last run successfully on
2026-05-30: 45/45 package test scripts passed.
