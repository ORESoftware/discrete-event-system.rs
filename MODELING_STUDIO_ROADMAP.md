# Modeling Studio Roadmap

Goal: make this crate an open-source modeling environment in the spirit of
PyDy, OpenMDAO, Simulink, and Modelica while keeping the Rust engine embeddable
and inspectable.

Reference points:

- Simulink: graphical block diagrams for algorithms and physical systems,
  simulation, model-based design, verification, and code generation.
  See <https://www.mathworks.com/help/simulink/modeling.html>.
- Modelica: object-oriented, equation-based, acausal, multi-domain modeling for
  cyber-physical systems. See <https://modelica.org/language/> and the language
  specification at <https://specification.modelica.org/master/>.
- PyDy: symbolic multibody model specification, equation generation, numerical
  integration, visualization, benchmarking, and publication workflows. See
  <https://pydy.readthedocs.io/en/stable/>.
- OpenMDAO: component/group model composition, drivers, solvers, derivatives,
  case recording, optimization metadata, and N2 model visualization. See
  <https://openmdao.org/newdocs/versions/latest/>.

## What We Already Have

- Simulation kernels: DES, hybrid stepping, ODE/PDE/math blocks, optimization,
  control, MDP/POMDP/RL, stochastic models, queueing and domain simulations.
- Block semantics: `math_blocks` has sources, sums, gains, integrators,
  derivatives, filters, comparators, expressions, and Laplacian blocks.
- Visual rendering: `VisualBlockSpec` renders block diagrams into animation
  shapes; calculus blocks now render semantic glyphs.
- Studio runtime: a flat visual graph of blocks, each block containing one or
  more executable runtime ops.
- Studio authoring: serializable graph specs, JSON Schema, Rust code generation,
  palette metadata, a generated browser editor, and a generated workbench.
- Studio analysis: component summaries, sparse N2 dependency data, execution
  order, executive selection, design-variable/objective/constraint metadata,
  and driver-style parameter sweeps.
- Acausal equation runtime: `des::acausal` accepts JSON variables/equations,
  runs alias/connect elimination, dependency-sorts algebraic assignments,
  rejects algebraic loops, simulates explicit ODE systems with Euler/RK4, and
  exposes a UI workbench descriptor for equation/model authoring.
- Artifacts: model runs can produce uniform `RunArtifact`s with frames, charts,
  controls, summaries, and validation.

## Gaps To Close

- UI authoring: richer drag interactions, typed wiring, validation overlays,
  scoped signal logging, reusable subsystem masks, and model diffing.
- Simulink-style simulation: feedback loops with delay/state breakpoints,
  algebraic-loop detection, subsystem masking, buses, scopes, logging, solver
  configuration, parameter sweeps, and code export.
- PyDy-style mechanics: reference frames, points/particles/rigid bodies,
  symbolic Kane/Lagrange equation assembly, generated numeric functions,
  mechanics examples, and 2D/3D visualization of bodies and joints.
- OpenMDAO-style analysis: explicit components/groups, nonlinear/linear solver
  configuration, derivative approximation/checking, case recording, DOE and
  optimizer drivers, and nested N2 views.
- Modelica-style modeling: components/classes, typed physical connectors,
  acausal connect equations, units/domains, DAE compilation, equation sorting,
  initialization, events, and library packaging.
- Interoperability: import/export for JSON, Modelica-like text, FMI/FMU, SBML or
  domain-specific adapters where useful.
- Engineering workflow: model diffing, version metadata, test harnesses,
  requirements/validation blocks, reproducible reports, and CI-friendly runs.

## First Implemented Slice

- `studio::spec` defines a serializable `StudioModelSpec`.
- `studio::spec::studio_palette()` exposes UI palette metadata and inspector
  fields.
- `studio::spec::compile_model_spec()` validates and compiles saved diagrams
  into the existing Studio runtime.
- `hybrid::spec` defines a typed `HybridModelSpec` for continuous, discrete, and
  event-driven block diagrams.
- `schemars` derives JSON Schema from the Rust graph-spec types, so editor JSON,
  saved files, and runtime validation share one contract.
- Studio and Hybrid citizens now run saved JSON graph specs directly and include
  generated Rust runner source in their result documents.
- `model::authoring` carries the cross-cutting metadata for hierarchy, variants,
  physical domains/connectors/equations, solver policy, statecharts, FMI intent,
  requirements/V&V, tooling, and Rust codegen.
- Studio nodes now retain stable palette `kind` metadata for UI/tooling.
- A terminal `Probe` op gives sink blocks true `1 -> 0` port semantics.
- `studio::analysis` exports component/N2/executive metadata for the UI and API.
- `studio::sweep` runs OpenMDAO-style design-variable sweeps and records
  objectives/constraints.
- `studio::ui` renders `out/studio/workbench.html`, a self-contained workbench
  with palette, inspector, canvas, N2 view, JSON view, local run, and sweep
  driver.
- `studio::players` writes catalog-ready Studio run, N2, and sweep-driver HTML
  players through the uniform `RunArtifact` renderer.
- `acausal::compile_acausal_model()` adds the first ModelingToolkit-style
  structural pass: alias elimination, algebraic ordering, missing-equation
  checks, and generated simulation traces.
- `acausal::acausal_workbench_descriptor()` exposes UI tabs, palette items, and
  a starter damped mass-spring model for an editor shell.

## Next Slices

1. Make the existing editor and workbench share one UI bundle, then add direct
   canvas wiring, block dragging, validation overlays, and signal scopes.
2. Solver depth: wire adaptive RK45/backward-Euler selection into the hybrid
   executive and add an algebraic solver path instead of only rejecting loops.
3. PyDy mechanics slice: add Mass, Spring, Damper, RevoluteJoint, Body, and
   Kane/Lagrange equation export for small mechanical systems.
4. OpenMDAO driver slice: add DOE grids, finite-difference derivative checks,
   case recording, and optimizer-driver adapters over the sweep API.
5. Units and typed ports: scalar, vector, bus, event, and physical connector
   compatibility checks backed by the authoring metadata.
6. Scopes and logging: add Scope, ToWorkspace, assertions, and signal history
   selection per block/port.
7. Acausal components: introduce physical connector equations and compile small
   Modelica-like networks into equations/DAEs.
8. FMI/FMU: package Rust-backed co-simulation/model-exchange exports and import
   FMU manifests into graph blocks.
9. V&V/tooling: make requirements, coverage, test harnesses, signal-inspector
   comparisons, variant manager, and dependency analyzer executable workflows.
10. Symbolic transforms: residual DAE support, Jacobian/sparsity generation,
   tearing, initialization problems, and code export.
