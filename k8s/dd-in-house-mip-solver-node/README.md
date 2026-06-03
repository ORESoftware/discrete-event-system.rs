# dd-in-house-mip-solver-node

Kubernetes bundle for the distributed in-house LP/MIP/IP solver node.

The same Rust image runs as either a master or slave. Role is deterministic at
boot through `MIP_SOLVER_NODE_ROLE`, so pods never elect or switch roles at
runtime:

- `dd-in-house-mip-solver-node-master` exposes HTTP solve/session APIs and
  publishes branch-and-bound subproblem jobs.
- `dd-in-house-mip-solver-node-slave` consumes JetStream work, solves
  subproblems with the in-house Rust LP/MIP/IP path, publishes results, and is
  scaled by KEDA from consumer lag.

The NATS names in these manifests mirror the generated Rust constants from
`remote/libs/nats/subject-defs/generated/rust`:

- `MIP_SOLVER_JOBS_SUBJECT`: `dd.remote.mip_solver.jobs`
- `MIP_SOLVER_RESULTS_SUBJECT`: `dd.remote.mip_solver.results`
- `MIP_SOLVER_EVENTS_SUBJECT`: `dd.remote.mip_solver.events`
- `DD_REMOTE_MIP_SOLVER_STREAM_NAME`: `DD_REMOTE_MIP_SOLVER`
- `MIP_SOLVER_WORKERS_QUEUE_GROUP`: `dd-in-house-mip-solver-node-workers`

KEDA watches the durable JetStream consumer `dd-in-house-mip-solver-node-workers`
on stream `DD_REMOTE_MIP_SOLVER`. When pending subproblems grow, KEDA scales the
slave Deployment. New slave pods boot with `MIP_SOLVER_NODE_ROLE=slave`, attach
to the same consumer, and begin receiving pending work without changing NATS
subjects.

GPU support is opt-in by overlaying GPU resource limits, for example
`nvidia.com/gpu: '1'`, on the slave Deployment. The base manifest leaves GPU
resources unset so it can run on CPU-only clusters while the server still probes
GPU availability when devices are present.

