# Migration Manifest

Generated from TypeScript `RUST MIGRATION` headers.

- TypeScript files mapped: 386
- Library modules: 233
- Binaries: 95
- Integration tests: 58

| TypeScript source | Rust target | Kind | Top-level declarations |
| --- | --- | --- | --- |
| `src/des/abstract/abstract.ts` | `src/des/abstract/abstract.rs` | lib | `AbstractBidirectionalEntity`, `Entity`, `EntityConnection`, `EntityObserver`, `HasNumericValue`, `IsSerializable`, `Serializable`, `StationaryEntity`, `TimeStepOpts` |
| `src/des/abstract/composers.ts` | `src/des/abstract/composers.rs` | lib | `DoesFanOut` |
| `src/des/abstract/interfaces.ts` | `src/des/abstract/interfaces.rs` | lib | `EntityGraphData`, `EventNames`, `HasEntityValidation`, `HasId`, `HasInput`, `HasInternalQueue`, `HasManyInputConnections`, `HasManyOutputConnections`, `HasOutput`, `HasSingleInputConnection`, `HasSingleOutputConnection`, `IsObservable` |
| `src/des/abstract/test.ts` | `src/des/abstract/test.rs` | lib |  |
| `src/des/animation/frame-recorder.ts` | `src/des/animation/frame_recorder.rs` | lib | `FrameRecorder`, `FrameRecorderOpts`, `readAnimation` |
| `src/des/animation/html-player.ts` | `src/des/animation/html_player.rs` | lib | `AnimationSetOptions`, `AnimationVariant`, `buildHTML`, `buildHTMLSet` |
| `src/des/animation/render.ts` | `src/des/animation/render.rs` | lib |  |
| `src/des/animation/run-report.ts` | `src/des/animation/run_report.rs` | lib | `CatalogEntry`, `CatalogSection`, `IndexEntry`, `IndexGroup`, `MetricRow`, `ReportSection`, `RunReportPage`, `SimulationIndexPage` |
| `src/des/animation/scenes/calculus-scene.ts` | `src/des/animation/scenes/calculus_scene.rs` | lib | `POISSON_H`, `POISSON_W`, `STAGE_H`, `STAGE_W`, `buildField1DChart`, `buildField1DFrame`, `buildPoissonFrame` |
| `src/des/animation/scenes/computer-network-scene.ts` | `src/des/animation/scenes/computer_network_scene.rs` | lib | `COMPUTER_NETWORK_STAGE_H`, `COMPUTER_NETWORK_STAGE_W`, `buildComputerNetworkAnimation` |
| `src/des/animation/scenes/contact-seir-scene.ts` | `src/des/animation/scenes/contact_seir_scene.rs` | lib | `PersonView`, `STAGE_H`, `STAGE_W`, `buildContactChart`, `buildContactFrame`, `layoutGrid` |
| `src/des/animation/scenes/dc-motor-scene.ts` | `src/des/animation/scenes/dc_motor_scene.rs` | lib | `DcMotorScene`, `DcMotorSceneOpts`, `MOTOR_STAGE_H`, `MOTOR_STAGE_W` |
| `src/des/animation/scenes/elevator-scene.ts` | `src/des/animation/scenes/elevator_scene.rs` | lib | `STAGE_H`, `STAGE_W`, `buildElevatorChart`, `buildElevatorFrame` |
| `src/des/animation/scenes/factmachine-scene.ts` | `src/des/animation/scenes/factmachine_scene.rs` | lib | `ArchitectureFrameArgs`, `STAGE_H`, `STAGE_W`, `buildFactMachineCharts`, `buildFactMachineFrame` |
| `src/des/animation/scenes/genetic-tsp-scene.ts` | `src/des/animation/scenes/genetic_tsp_scene.rs` | lib | `ArchitectureFrameArgs`, `STAGE_H`, `STAGE_W`, `buildGeneticTSPCharts`, `buildGeneticTSPFrame` |
| `src/des/animation/scenes/incremental-lp-scene.ts` | `src/des/animation/scenes/incremental_lp_scene.rs` | lib | `STAGE_H`, `STAGE_W`, `buildIncrementalLPCharts`, `buildIncrementalLPFrame` |
| `src/des/animation/scenes/neural-network-scene.ts` | `src/des/animation/scenes/neural_network_scene.rs` | lib | `NEURAL_STAGE_H`, `NEURAL_STAGE_W`, `buildNeuralOdeAnimation`, `buildNeuralQCorridorAnimation`, `buildNeuralXorAnimation` |
| `src/des/animation/scenes/newsvendor-scene.ts` | `src/des/animation/scenes/newsvendor_scene.rs` | lib | `NewsvendorFrameData`, `STAGE_H`, `STAGE_W`, `buildNewsvendorChart`, `buildNewsvendorFrame` |
| `src/des/animation/scenes/obs-ctrl-scene.ts` | `src/des/animation/scenes/obs_ctrl_scene.rs` | lib | `OC_STAGE_H`, `OC_STAGE_W`, `ObsCtrlScene`, `StoryStep` |
| `src/des/animation/scenes/shortest-path-scene.ts` | `src/des/animation/scenes/shortest_path_scene.rs` | lib | `STAGE_H`, `STAGE_W`, `buildShortestPathCharts`, `buildShortestPathFrame` |
| `src/des/animation/scenes/soccer-ipmip-solver-scene.ts` | `src/des/animation/scenes/soccer_ipmip_solver_scene.rs` | lib | `SOCCER_IPMIP_SOLVER_H`, `SOCCER_IPMIP_SOLVER_W`, `SOLVER_FRAMES_PER_EVENT`, `buildSoccerIPMIPSolverCharts`, `buildSoccerIPMIPSolverFrame`, `soccerIPMIPSolverFrameCount` |
| `src/des/animation/scenes/soccer-scene.ts` | `src/des/animation/scenes/soccer_scene.rs` | lib | `STAGE_H`, `STAGE_W`, `SoccerFrameInput`, `buildSoccerCharts`, `buildSoccerFrame` |
| `src/des/animation/scenes/temp-control-scene.ts` | `src/des/animation/scenes/temp_control_scene.rs` | lib | `STAGE_H`, `STAGE_W`, `buildTempControlAnimation`, `buildTempControlFrame` |
| `src/des/animation/scenes/two-disease-scene.ts` | `src/des/animation/scenes/two_disease_scene.rs` | lib | `COLORS`, `CompartmentCounts`, `STAGE_H`, `STAGE_W`, `buildBars`, `buildCompartmentChart`, `buildFrame` |
| `src/des/animation/scenes/warehouse-track3t-scene.ts` | `src/des/animation/scenes/warehouse_track3t_scene.rs` | lib | `WAREHOUSE_TRACK3T_STAGE_H`, `WAREHOUSE_TRACK3T_STAGE_W`, `buildWarehouseComparisonCharts`, `buildWarehouseComparisonFrame`, `warehouseComparisonFrameCount`, `warehouseComparisonFrameTime` |
| `src/des/animation/scenes/wind-mppt-scene.ts` | `src/des/animation/scenes/wind_mppt_scene.rs` | lib | `WIND_STAGE_H`, `WIND_STAGE_W`, `WindMpptScene`, `WindSceneOpts` |
| `src/des/animation/types.ts` | `src/des/animation/types.rs` | lib | `Animation`, `ChartSeries`, `ChartSpec`, `CircleShape`, `Frame`, `LineShape`, `PathShape`, `RectShape`, `Shape`, `TextShape` |
| `src/des/child.ts` | `src/des/child.rs` | lib |  |
| `src/des/entity-conn.ts/conn.ts` | `src/des/entity_conn_ts/conn.rs` | lib | `ConnectionOpts` |
| `src/des/entity-decision/binary-decision.ts` | `src/des/entity_decision/binary_decision.rs` | lib | `BinaryDecisionEntity`, `DecisionEntityGraph` |
| `src/des/entity-decision/decision.ts` | `src/des/entity_decision/decision.rs` | lib | `DecisionEntity`, `DecisionEntityGraph` |
| `src/des/entity-decision/probability-decision.ts` | `src/des/entity_decision/probability_decision.rs` | lib | `DecisionEntityGraph`, `ProbabilityDecisionEntity` |
| `src/des/entity-moving/moving.ts` | `src/des/entity_moving/moving.rs` | lib | `AbstractMovingEntity`, `BasicMovingEntity`, `BasicQuantityMovingEntity`, `ProcessableMovingEntity`, `ProcessingTimeValue` |
| `src/des/entity-processing/per-individual-processor.ts` | `src/des/entity_processing/per_individual_processor.rs` | lib | `PerIndividualProcessor`, `PerIndividualProcessorOpts` |
| `src/des/entity-processing/processing.ts` | `src/des/entity_processing/processing.rs` | lib | `EntityProcessor`, `ProcessorEntityGraphData`, `isProcessor` |
| `src/des/entity-processing/value-adder.ts` | `src/des/entity_processing/value_adder.rs` | lib | `EntityNumericProcessor`, `GraphData`, `isProcessor` |
| `src/des/entity-queue/queue.ts` | `src/des/entity_queue/queue.rs` | lib | `QueueEntity`, `QueueEntityGraphData` |
| `src/des/entity-routing/entity-splitter.ts` | `src/des/entity_routing/entity_splitter.rs` | lib | `DecisionEntityGraph`, `EntitySplitter` |
| `src/des/entity-routing/output-routing-policy.ts` | `src/des/entity_routing/output_routing_policy.rs` | lib | `HasOutputRoutingPolicy`, `OutputConnectionRouter`, `OutputRoutingPolicy` |
| `src/des/entity-sink/generic-sink.ts` | `src/des/entity_sink/generic_sink.rs` | lib | `GenericEntitySink` |
| `src/des/entity-sink/sink.ts` | `src/des/entity_sink/sink.rs` | lib | `AbstractSinkEntity`, `EntitySink` |
| `src/des/entity-source/source.ts` | `src/des/entity_source/source.rs` | lib | `AbstractSourceEntity`, `DefiniteFiniteSource`, `EntitySource` |
| `src/des/entity-travel/time-delay.ts` | `src/des/entity_travel/time_delay.rs` | lib | `DelayTimeStepOpts`, `TimeDelayEntityGraphData`, `TimeDelayOrTravelEntity` |
| `src/des/general/actor-critic-gridworld.ts` | `src/des/general/actor_critic_gridworld.rs` | lib | `ActorCriticResult`, `ActorCriticTrainOpts`, `runActorCriticGridworld` |
| `src/des/general/adapters/adapter-utils.ts` | `src/des/general/adapters/adapter_utils.rs` | lib | `csvCell`, `csvRow`, `defaultFramesPath`, `framesPath`, `jsonCsvCell`, `jsonCsvRow`, `numberPair`, `optionalNumberPair`, `validationLine`, `writeCsvLines` |
| `src/des/general/adapters/advanced-optimization-control-adapter.ts` | `src/des/general/adapters/advanced_optimization_control_adapter.rs` | lib |  |
| `src/des/general/adapters/classical-optimization-adapter.ts` | `src/des/general/adapters/classical_optimization_adapter.rs` | lib |  |
| `src/des/general/adapters/collaborative-inference-adapter.ts` | `src/des/general/adapters/collaborative_inference_adapter.rs` | lib |  |
| `src/des/general/adapters/computer-network-adapter.ts` | `src/des/general/adapters/computer_network_adapter.rs` | lib |  |
| `src/des/general/adapters/domain-application-adapter.ts` | `src/des/general/adapters/domain_application_adapter.rs` | lib |  |
| `src/des/general/adapters/feasibility-pipeline-adapter.ts` | `src/des/general/adapters/feasibility_pipeline_adapter.rs` | lib |  |
| `src/des/general/adapters/internal-solver-network-adapter.ts` | `src/des/general/adapters/internal_solver_network_adapter.rs` | lib |  |
| `src/des/general/adapters/learning-optimization-adapter.ts` | `src/des/general/adapters/learning_optimization_adapter.rs` | lib |  |
| `src/des/general/adapters/math-blocks-adapter.ts` | `src/des/general/adapters/math_blocks_adapter.rs` | lib |  |
| `src/des/general/adapters/mdp-adjacent-adapters.ts` | `src/des/general/adapters/mdp_adjacent_adapters.rs` | lib |  |
| `src/des/general/adapters/milp-bnb-adapter.ts` | `src/des/general/adapters/milp_bnb_adapter.rs` | lib |  |
| `src/des/general/adapters/multistage-sddp-adapter.ts` | `src/des/general/adapters/multistage_sddp_adapter.rs` | lib |  |
| `src/des/general/adapters/network-flow-adapter.ts` | `src/des/general/adapters/network_flow_adapter.rs` | lib |  |
| `src/des/general/adapters/network-flow-adapters.ts` | `src/des/general/adapters/network_flow_adapters.rs` | lib |  |
| `src/des/general/adapters/neural-network-adapters.ts` | `src/des/general/adapters/neural_network_adapters.rs` | lib |  |
| `src/des/general/adapters/nonlinear-forecasting-adapter.ts` | `src/des/general/adapters/nonlinear_forecasting_adapter.rs` | lib |  |
| `src/des/general/adapters/nonlinear-optimization-adapter.ts` | `src/des/general/adapters/nonlinear_optimization_adapter.rs` | lib |  |
| `src/des/general/adapters/optimal-control-adapters.ts` | `src/des/general/adapters/optimal_control_adapters.rs` | lib |  |
| `src/des/general/adapters/shortest-path-adapter.ts` | `src/des/general/adapters/shortest_path_adapter.rs` | lib |  |
| `src/des/general/adapters/signal-transforms-adapter.ts` | `src/des/general/adapters/signal_transforms_adapter.rs` | lib |  |
| `src/des/general/adapters/simulated-annealing-adapter.ts` | `src/des/general/adapters/simulated_annealing_adapter.rs` | lib |  |
| `src/des/general/adapters/statistical-optimization-adapter.ts` | `src/des/general/adapters/statistical_optimization_adapter.rs` | lib |  |
| `src/des/general/adapters/stochastic-flow-mdp-adapter.ts` | `src/des/general/adapters/stochastic_flow_mdp_adapter.rs` | lib |  |
| `src/des/general/adapters/stochastic-optimization-adapters.ts` | `src/des/general/adapters/stochastic_optimization_adapters.rs` | lib |  |
| `src/des/general/adapters/temp-control-adapter.ts` | `src/des/general/adapters/temp_control_adapter.rs` | lib |  |
| `src/des/general/advanced-control-models.ts` | `src/des/general/advanced_control_models.rs` | lib | `HInfinityRobustControlParams`, `HInfinityRobustControlResult`, `PursuitEvasionGameParams`, `PursuitEvasionGameResult`, `advancedControlChannels`, `runHInfinityRobustControl`, `runPursuitEvasionGame` |
| `src/des/general/advanced-optimization-models.ts` | `src/des/general/advanced_optimization_models.rs` | lib | `AntColonyTSPParams`, `AntColonyTSPResult`, `ContinuousObjectiveName`, `MapColoringCSPParams`, `MapColoringCSPResult`, `MaxSATParams`, `MaxSATResult`, `ParetoPortfolioParams`, `ParetoPortfolioPoint`, `ParetoPortfolioResult`, `ParticleSwarmParams`, `ParticleSwarmResult`, `Point2`, `PortfolioAsset`, `SDPMaxCutParams`, `SDPMaxCutResult`, `WeightedEdge`, `paretoFrontIsNondominated`, `runAntColonyTSP`, `runMapColoringCSP`, `runMaxSATLocalSearch`, `runParetoPortfolio`, `runParticleSwarm`, `runSDPMaxCutRelaxation` |
| `src/des/general/belief.ts` | `src/des/general/belief.rs` | lib | `DiscreteBelief`, `brierScore`, `klDivergence` |
| `src/des/general/blackjack.ts` | `src/des/general/blackjack.rs` | lib | `Blackjack`, `BlackjackResult`, `BlackjackTrainOpts`, `runBlackjackMC` |
| `src/des/general/cartesian-state-space.ts` | `src/des/general/cartesian_state_space.rs` | lib | `CartesianDimension`, `CartesianStateSpace`, `CoordinateMDPSpec`, `CoordinateTransition`, `coordinateMDPToSpec` |
| `src/des/general/classical-optimization-models.ts` | `src/des/general/classical_optimization_models.rs` | lib | `AssignmentParams`, `AssignmentResult`, `FlowShopJob`, `FlowShopNEHParams`, `FlowShopNEHResult`, `JobOperation`, `JobShopDispatchParams`, `JobShopDispatchResult`, `JobShopJob`, `QPProjectedGradientParams`, `QPProjectedGradientResult`, `ScheduledOperation`, `VRPCustomer`, `VRPRoute`, `VRPSavingsParams`, `VRPSavingsResult`, `runAuctionAssignment`, `runFlowShopNEH`, `runHungarianAssignment`, `runJobShopDispatch`, `runQPCoordinateDescent`, `runQPProjectedGradient`, `runVRPNearestNeighbor`, `runVRPSavings` |
| `src/des/general/collaborative-inference.ts` | `src/des/general/collaborative_inference.rs` | lib | `CollaborativeInferenceCoverage`, `CollaborativeInferenceItem`, `CollaborativeInferenceParams`, `CollaborativeInferenceResponse`, `CollaborativeInferenceResult`, `CollaborativeInferenceScenario`, `CollaborativeItemScore`, `CredibilityWeightSummary`, `runCollaborativeInference` |
| `src/des/general/computer-network.ts` | `src/des/general/computer_network.rs` | lib | `ComputerNetworkProblem`, `ComputerNetworkResult`, `ComputerNetworkStation`, `NetworkBottleneckReport`, `NetworkDelayStation`, `NetworkFlowSpec`, `NetworkFlowStats`, `NetworkHostStation`, `NetworkLinkSpec`, `NetworkLinkStation`, `NetworkLinkStats`, `NetworkNodeKind`, `NetworkNodeSpec`, `NetworkNodeStation`, `NetworkNodeStats`, `NetworkPacket`, `NetworkPacketSnapshot`, `NetworkProtocol`, `NetworkRouterStation`, `NetworkRoutingMetric`, `NetworkStation`, `NetworkSwitchStation`, `NetworkTimeSample`, `PacketDropReason`, `buildBottleneckComputerNetworkProblem`, `buildDefaultComputerNetworkProblem`, `runComputerNetworkSimulation`, `validateComputerNetworkProblem` |
| `src/des/general/control-systems/dc-motor.ts` | `src/des/general/control_systems/dc_motor.rs` | lib | `DcMotorChannels`, `DcMotorDynamics`, `DcMotorParams`, `DcMotorPlantOpts`, `DcMotorPlantStation`, `DcMotorSinkStation`, `LoadProfile`, `LoadSegment`, `MotorStateToken`, `SpeedPiVoltageController`, `SpeedPiVoltageOpts`, `SpeedReferenceSegment`, `VoltageToken` |
| `src/des/general/control-systems/empirical-control.ts` | `src/des/general/control_systems/empirical_control.rs` | lib | `BeliefTracker`, `ControllabilityGramian`, `DegreeKind`, `DegreeReportSinkStation`, `DegreeReportToken`, `DiscreteLinearSystem`, `DiscreteSystemSourceStation`, `DiscreteSystemToken`, `EmpiricalChannels`, `GramianDegree`, `LtiDegreeEvaluatorStation`, `MdpControllabilityDegree`, `MdpDegreeEvaluatorStation`, `MdpDegreeSourceStation`, `MdpDegreeToken`, `MinEnergyController`, `MonteCarloControllability`, `MonteCarloControllabilityResult`, `MonteCarloDistinguishability`, `MonteCarloObservability`, `MonteCarloObservabilityResult`, `Mulberry32`, `ObservabilityGramian`, `PomdpDegreeEvaluatorStation`, `PomdpDegreeSourceStation`, `PomdpDegreeToken`, `PomdpObservabilityResult` |
| `src/des/general/control-systems/linear-algebra.ts` | `src/des/general/control_systems/linear_algebra.rs` | lib | `LinAlg`, `Mat`, `MatrixInverse`, `MatrixRank`, `SymmetricEigen`, `Vec` |
| `src/des/general/control-systems/numerical-solvers.ts` | `src/des/general/control_systems/numerical_solvers.rs` | lib | `FixedStepIntegrator`, `ForwardEulerIntegrator`, `OdeSystem`, `RungeKutta4Integrator` |
| `src/des/general/control-systems/observability-controllability.ts` | `src/des/general/control_systems/observability_controllability.rs` | lib | `ControllabilityEvaluatorStation`, `EvaluationKind`, `EvaluationSinkStation`, `EvaluationToken`, `MarkovDecisionProcess`, `MdpControllabilityEvaluatorStation`, `MdpSourceStation`, `MdpSpec`, `MdpToken`, `ObsCtrlChannels`, `ObservabilityEvaluatorStation`, `PartiallyObservableProcess`, `PomdpObservabilityEvaluatorStation`, `PomdpSourceStation`, `PomdpSpec`, `PomdpToken`, `StateSpaceModel`, `StateSpaceSourceStation`, `StateSpaceSpec`, `StateSpaceToken` |
| `src/des/general/control-systems/sde-learning.ts` | `src/des/general/control_systems/sde_learning.rs` | lib | `DenoisingDiffusionModel`, `DiffusionOptions`, `EnkfOptions`, `EnsembleKalmanFilter`, `EnsembleKalmanFilterStation`, `GbmFamily`, `MleFitResult`, `Mlp`, `OuFamily`, `ParametricSdeFamily`, `SdeMaximumLikelihoodEstimator` |
| `src/des/general/control-systems/stochastic-sde.ts` | `src/des/general/control_systems/stochastic_sde.rs` | lib | `EulerMaruyamaIntegrator`, `GeometricBrownianMotion`, `OrnsteinUhlenbeck`, `SdeChannels`, `SdeEstimateSinkStation`, `SdeEstimateToken`, `SdeObservationToken`, `SdePlantOptions`, `SdePlantStation`, `SdeStateToken`, `SdeSystem`, `StochasticDcMotor`, `StochasticDcMotorSpec` |
| `src/des/general/control-systems/wind-mppt.ts` | `src/des/general/control_systems/wind_mppt.rs` | lib | `GenTorqueToken`, `OptimalTorqueMpptController`, `RotorDynamics`, `SpeedPiMpptController`, `SpeedPiMpptOpts`, `TurbineStateToken`, `WindMpptChannels`, `WindMpptSinkStation`, `WindProfile`, `WindProfileSegment`, `WindTurbineAeroOpts`, `WindTurbineAerodynamics`, `WindTurbinePlantOpts`, `WindTurbinePlantStation` |
| `src/des/general/des-base/actor-critic.ts` | `src/des/general/des_base/actor_critic.rs` | lib | `ActorCriticOptions`, `TabularActorCritic` |
| `src/des/general/des-base/advanced-optimization.ts` | `src/des/general/des_base/advanced_optimization.rs` | lib | `ConstraintAssignmentToken`, `ConstraintSatisfactionSearchStation`, `ConstraintSearchNode`, `GraphWalkToken`, `NumericSwarmOptimizerStation`, `NumericSwarmOptions`, `NumericSwarmParticle`, `OptimizationCandidateToken`, `OptimizationTraceRow`, `ParetoArchiveRow`, `ParetoArchiveStation`, `ParetoCandidateToken`, `PheromoneGraphOptions`, `PheromoneGraphSearchStation`, `SourceDrivenConstraintSatisfactionSearchStation`, `UnitVectorRelaxationOptions`, `UnitVectorRelaxationStation`, `UnitVectorRelaxationTraceRow`, `dominates`, `gram`, `normalize`, `vectorDot` |
| `src/des/general/des-base/adversarial-control.ts` | `src/des/general/des_base/adversarial_control.rs` | lib | `CH_CONTROL`, `CH_DISTURBANCE`, `CH_OBSERVATION`, `ClosedLoopGameRunOptions`, `ClosedLoopGameTraceRow`, `ClosedLoopPlantOptions`, `ClosedLoopPlantStation`, `ControlMoveToken`, `DisturbanceMoveToken`, `DisturbancePolicyStation`, `FeedbackPolicyStation`, `StateObservationToken`, `runClosedLoopGame`, `wireClosedLoopGame` |
| `src/des/general/des-base/argmax.ts` | `src/des/general/des_base/argmax.rs` | lib | `ARGMAX_EPS_DEFAULT`, `allArgMaxTies`, `argMaxWithTieBreak`, `chooseRandomTied`, `scanArgMaxTieBreak` |
| `src/des/general/des-base/belief-state.ts` | `src/des/general/des_base/belief_state.rs` | lib | `ActionObservationToken`, `BeliefStateStation`, `BeliefToken`, `POMDPCore` |
| `src/des/general/des-base/composite-station.ts` | `src/des/general/des_base/composite_station.rs` | lib | `CompositeDESStation`, `CompositeStationSnapshot` |
| `src/des/general/des-base/control-blocks.ts` | `src/des/general/des_base/control_blocks.rs` | lib | `ClosedLoopOpts`, `ClosedLoopResult`, `ControllerBlock`, `EstimatorBlock`, `PlantBlock`, `VectorSignal`, `runClosedLoop` |
| `src/des/general/des-base/controller.ts` | `src/des/general/des_base/controller.rs` | lib | `ControlToken`, `ControllerStation`, `ObservationToken` |
| `src/des/general/des-base/cut-pool.ts` | `src/des/general/des_base/cut_pool.rs` | lib | `AffineCut`, `AffineCutPool`, `CutEnvelopeSense` |
| `src/des/general/des-base/environment.ts` | `src/des/general/des_base/environment.rs` | lib | `EnvironmentStation`, `EnvironmentStationOptions`, `PureEnvironment` |
| `src/des/general/des-base/episode-accounting.ts` | `src/des/general/des_base/episode_accounting.rs` | lib | `EpisodeAccounting`, `EpisodeSummary`, `VectorEpisodeAccounting`, `VectorEpisodeSummary` |
| `src/des/general/des-base/finite-horizon-dp.ts` | `src/des/general/des_base/finite_horizon_dp.rs` | lib | `DPOptions`, `DPOutcome`, `FiniteHorizonDPStation` |
| `src/des/general/des-base/fixed-point.ts` | `src/des/general/des_base/fixed_point.rs` | lib | `FixedPointIterationStation`, `FixedPointOptions` |
| `src/des/general/des-base/index.ts` | `src/des/general/des_base/mod.rs` | lib |  |
| `src/des/general/des-base/learning-optimization.ts` | `src/des/general/des_base/learning_optimization.rs` | lib | `CandidateEvaluatorStation`, `CandidateSourceStation`, `CandidateToken`, `EvaluatedCandidateToken`, `GradientEvaluation`, `GradientOptimizerOptions`, `GradientOptimizerStation`, `GradientStepToken`, `GradientTraceSinkStation`, `IncumbentSinkStation`, `IncumbentToken`, `LatestTokenSinkStation`, `MiniBatchStation`, `SingleTokenSourceStation`, `StationGraphSummary`, `VectorBatchToken`, `VectorSampleSourceStation`, `VectorSampleToken`, `channelEdge`, `cloneMatrix`, `dot`, `emptyStationGraph`, `nonEmptyArray`, `norm2`, `runStateLoopPipeline`, `sigmoid`, `softmax`, `stateLoopTopology`, `stationGraph`, `zeros` |
| `src/des/general/des-base/linear-vfa.ts` | `src/des/general/des_base/linear_vfa.rs` | lib | `LinearVFAOptions`, `LinearVFAStation` |
| `src/des/general/des-base/lqr-controller.ts` | `src/des/general/des_base/lqr_controller.rs` | lib | `LQRController`, `LQRSpec`, `Mat`, `Vec` |
| `src/des/general/des-base/model-topology.ts` | `src/des/general/des_base/model_topology.rs` | lib | `StationGraphTopology`, `stationGraphTopology` |
| `src/des/general/des-base/monte-carlo-rl.ts` | `src/des/general/des_base/monte_carlo_rl.rs` | lib | `MonteCarloAgent`, `MonteCarloOptions` |
| `src/des/general/des-base/multi-agent.ts` | `src/des/general/des_base/multi_agent.rs` | lib | `JointEnvStation`, `JointEnvironment`, `MultiAgentSystem`, `MultiAgentSystemOpts` |
| `src/des/general/des-base/neural-network.ts` | `src/des/general/des_base/neural_network.rs` | lib | `NeuralInferenceToken`, `NeuralNetworkLike`, `NeuralNetworkStation`, `NeuralPredictionToken`, `NeuralSnapshotToken`, `NeuralTrainingResultToken`, `NumericVector`, `SupervisedNeuralNetworkStation`, `SupervisedNeuralNetworkStationOptions`, `SupervisedSampleToken`, `TrainableNeuralNetwork` |
| `src/des/general/des-base/policy-gradient-agent.ts` | `src/des/general/des_base/policy_gradient_agent.rs` | lib | `PolicyGradientAgent`, `PolicyUpdateStation`, `RolloutEntry` |
| `src/des/general/des-base/population-optimizer.ts` | `src/des/general/des_base/population_optimizer.rs` | lib | `POPULATION_INITIAL_CHANNEL`, `POPULATION_RESULT_CHANNEL`, `PopulationInitialToken`, `PopulationOptimizer`, `PopulationResultSnapshot`, `PopulationResultToken`, `PopulationSinkStation`, `PopulationSourceStation` |
| `src/des/general/des-base/preconditions.ts` | `src/des/general/des_base/preconditions.rs` | lib | `PreconditionError`, `allFinite`, `arrNonNegative`, `check`, `equal`, `finite`, `inRange`, `integer`, `integerInRange`, `lengthEq`, `magnitudeLeq`, `nonEmpty`, `nonNegative`, `notDivByZero`, `positive`, `positiveDefiniteCholesky`, `positiveSemidefiniteDiag`, `probabilityVector`, `rectangularMatrix`, `squareMatrix`, `symmetricMatrix` |
| `src/des/general/des-base/rl-agent.ts` | `src/des/general/des_base/rl_agent.rs` | lib | `RLAgentStation` |
| `src/des/general/des-base/rl-tokens.ts` | `src/des/general/des_base/rl_tokens.rs` | lib | `ActionToken`, `ResumeToken`, `StateToken`, `TrainTriggerToken`, `TransitionToken` |
| `src/des/general/des-base/runner.ts` | `src/des/general/des_base/runner.rs` | lib | `DESResultStation`, `IterativeDESParticipant`, `IterativeRunOptions`, `IterativeRunSummary`, `assertNoValidationFailures`, `failedValidationChecks`, `runIterativeDES`, `runResultStation`, `validationFailureNames` |
| `src/des/general/des-base/semi-mdp.ts` | `src/des/general/des_base/semi_mdp.rs` | lib | `Option`, `SemiMDPAgentStation`, `SemiMDPOptions` |
| `src/des/general/des-base/single-state-optimizer.ts` | `src/des/general/des_base/single_state_optimizer.rs` | lib | `SINGLE_STATE_INITIAL_CHANNEL`, `SINGLE_STATE_RESULT_CHANNEL`, `SingleStateInitialToken`, `SingleStateOptimizer`, `SingleStateResultSnapshot`, `SingleStateResultToken`, `SingleStateSinkStation`, `SingleStateSourceStation` |
| `src/des/general/des-base/smart-movable.ts` | `src/des/general/des_base/smart_movable.rs` | lib | `SmartMovable` |
| `src/des/general/des-base/stateful-token.ts` | `src/des/general/des_base/stateful_token.rs` | lib | `PayloadStatefulToken`, `StatefulToken`, `StatefulTokenRegistry`, `StatefulTokenRegistryStats`, `TokenLineage`, `TokenStateMode`, `TokenStateTransition`, `childLineage`, `isStatefulToken`, `makeStatefulToken`, `makeStatelessToken`, `spawnStatefulChildToken`, `transitionToken` |
| `src/des/general/des-base/station.ts` | `src/des/general/des_base/station.rs` | lib | `ChannelName`, `DEFAULT_CHANNEL`, `DESRunLoopEntity`, `DESStation`, `HasRunTimeStep`, `Token` |
| `src/des/general/des-base/transform-entity.ts` | `src/des/general/des_base/transform_entity.rs` | lib | `FunctionEntity`, `MemoryTransformEntity`, `PureTransform`, `PureTransformEntity`, `TransformContext`, `TransformEntity`, `TransformEntityOptions`, `TransformFunction`, `TransformResult` |
| `src/des/general/des-base/tree-search.ts` | `src/des/general/des_base/tree_search.rs` | lib | `NodeEvaluation`, `SearchObjective`, `TreeSearchStation` |
| `src/des/general/des-base/validation.ts` | `src/des/general/des_base/validation.rs` | lib | `ValidationCheck`, `Validator`, `boundValidator`, `externalReferenceValidator`, `formatValidationReport`, `groundTruthValidator`, `intrinsicCheck`, `monotonicityValidator`, `numericValidator`, `runValidators` |
| `src/des/general/des-base/visual-block.ts` | `src/des/general/des_base/visual_block.rs` | lib | `VisualBlock`, `VisualBlockConnectionOptions`, `VisualBlockConnectionSpec`, `VisualBlockLayout`, `VisualBlockMember`, `VisualBlockOptions`, `VisualBlockPort`, `VisualBlockPortSpec`, `VisualBlockRenderContext`, `VisualBlockRenderable`, `VisualBlockRole`, `VisualBlockSpec`, `VisualBlockStyle`, `VisualPortDirection`, `VisualPortInput`, `VisualPortOptions`, `isVisualBlock`, `renderVisualBlockSpec`, `renderVisualBlocks`, `visualBlockSpecs` |
| `src/des/general/des-lp-bridge.ts` | `src/des/general/des_lp_bridge.rs` | lib | `MDPLPSolution`, `RollingHorizonStep`, `buildMDPLP`, `lpRollingHorizon`, `solveLPThenSimulate`, `solveMDPAsLP` |
| `src/des/general/des-registry.ts` | `src/des/general/des_registry.rs` | lib | `RunFromSpecOptions`, `getModel`, `listModels`, `registerModel` |
| `src/des/general/des-spec.ts` | `src/des/general/des_spec.rs` | lib | `DESModelMetadata`, `DESModelRegistration`, `DESModelSpec`, `DESRunSummary`, `DESRuntimeConfig`, `ParamSchema`, `ValidationResult`, `validate` |
| `src/des/general/dispatch.ts` | `src/des/general/dispatch.rs` | lib | `DispatchPolicy`, `DispatchProblem`, `DispatchResult`, `DispatchState`, `EvaluationResult`, `FluidLPPolicyResult`, `MCTSPolicyOptions`, `MDPVIPolicyOptions`, `MDPVIPolicyResult`, `buildDispatchFluidLP`, `evaluatePolicy`, `policyFluidLP`, `policyMCTS`, `policyMDPVI`, `policyRandom`, `policyRoundRobin`, `policySECT`, `policyShortestQueue`, `simulateDispatch`, `welchT` |
| `src/des/general/do-audit.ts` | `src/des/general/do_audit.rs` | lib | `doAudit` |
| `src/des/general/domain-application-models.ts` | `src/des/general/domain_application_models.rs` | lib | `ActiveLearningParams`, `ActiveLearningResult`, `AdaptiveFuzzyControlParams`, `AdaptiveFuzzyControlResult`, `BuyerAwareDynamicPricingParams`, `BuyerAwareDynamicPricingResult`, `DecisionScienceParams`, `DecisionScienceResult`, `DomainEvaluation`, `DomainModelResult`, `DomainTrace`, `EnergyParams`, `EnergyResult`, `FinancialControlParams`, `FinancialControlResult`, `LogisticsRoutingParams`, `LogisticsRoutingResult`, `ManufacturingParams`, `ManufacturingResult`, `OperationsParams`, `OperationsResult`, `RevenueManagementParams`, `RevenueManagementResult`, `SupplyChainParams`, `SupplyChainResult`, `runActiveLearningAcquisition`, `runAdaptiveFuzzyControl`, `runBottleneckProductionControl`, `runBuyerAwareDynamicPricing`, `runDynamicPricingRevenue`, `runEnergyStorageDispatch`, `runLogisticsRoutingHeuristics`, `runPortfolioDrawdownControl`, `runSupplyChainRiskPooling`, `runVisualDecisionFrontier`, `runWorkforceServiceOperations` |
| `src/des/general/double-integrator-lqr.ts` | `src/des/general/double_integrator_lqr.rs` | lib | `DoubleIntegratorOpts`, `DoubleIntegratorResult`, `runDoubleIntegratorLQR` |
| `src/des/general/entity-registration.ts` | `src/des/general/entity_registration.rs` | lib | `reg` |
| `src/des/general/equation-to-stations.ts` | `src/des/general/equation_to_stations.rs` | lib | `BC`, `Field1DBuild`, `Field1DScheme`, `Field1DSpec`, `Field2DScheme`, `ODEScheme`, `ODESystemSpec`, `Poisson2DResult`, `Poisson2DSpec`, `buildField1D`, `buildODESystem`, `solvePoisson2D`, `thomas` |
| `src/des/general/expr.ts` | `src/des/general/expr.rs` | lib | `BinNode`, `Env`, `Expr`, `FuncName`, `FuncNode`, `NumNode`, `ONE`, `UnaryNeg`, `VarNode`, `ZERO`, `add`, `diff`, `div`, `evaluate`, `fn`, `mul`, `neg`, `num`, `numericalDerivative`, `numericalGradient`, `parse`, `pow`, `richardsonDerivative`, `simplify`, `stringify`, `sub`, `toFunction`, `v` |
| `src/des/general/factmachine-math.ts` | `src/des/general/factmachine_math.rs` | lib | `BuyExecution`, `OptionAggregates`, `OptionPrices`, `RecapResult`, `ReplayOrder`, `ReplayResult`, `SellExecution`, `avgCostBasis`, `bFromLiquidity`, `buyExecution`, `buyThenSellRoundTrip`, `finalPnl`, `lmsrCost`, `maxPriceWithSlippage`, `minPriceWithSlippage`, `netPosition`, `optionOnePrice`, `optionPrices`, `recapitalization`, `replayOrders`, `sellExecution`, `sharesFromBudget`, `unrealizedPnl` |
| `src/des/general/factory-floor-track3t.ts` | `src/des/general/factory_floor_track3t.rs` | lib | `BASELINE_WAREHOUSE_SCENARIO`, `StationDefinition`, `TRACK3T_ARCHIVE_GROUNDING`, `TRACK3T_WAREHOUSE_SCENARIO`, `WarehouseAction`, `WarehouseComparisonResult`, `WarehouseDecisionState`, `WarehouseForklift`, `WarehouseJobSummary`, `WarehouseLayout`, `WarehouseMetrics`, `WarehouseObservation`, `WarehousePOMDPModel`, `WarehousePallet`, `WarehousePlanner`, `WarehouseQMDPSolver`, `WarehouseScenarioConfig`, `WarehouseScenarioResult`, `WarehouseSimulationOptions`, `WarehouseSink`, `WarehouseSource`, `WarehouseStation`, `WarehouseStationKind`, `WarehouseStepTrace`, `beliefByStation`, `buildWarehouseFloor`, `buildWarehousePOMDP`, `defaultWarehouseLayout`, `initialWarehouseBelief`, `runWarehouseComparison`, `simulateWarehouseScenario`, `summarizeWarehouseComparison`, `travelMinutes` |
| `src/des/general/feasibility-pipeline.ts` | `src/des/general/feasibility_pipeline.rs` | lib | `CANDIDATE_CHANNEL`, `CONSTRAINT_CHANNEL`, `CandidatePayload`, `CandidateSolutionInput`, `CandidateSourceStation`, `CandidateToken`, `ConstraintCheckedToken`, `ConstraintCheckerStation`, `ConstraintSense`, `DOMAIN_CHANNEL`, `DomainCheckedToken`, `DomainCheckerStation`, `EVALUATION_CHANNEL`, `FeasibilityEvaluation`, `FeasibilityEvaluationToken`, `FeasibilityImprovementOptions`, `FeasibilityPipelineEdge`, `FeasibilityPipelineNetwork`, `FeasibilityPipelineNode`, `FeasibilityPipelineParams`, `FeasibilityPipelineResult`, `FeasibilitySinkStation`, `FeasibilityViolation`, `ImprovementStation`, `LinearConstraint`, `LinearObjective`, `ObjectiveEvaluatorStation`, `ObjectiveSense`, `OptimizationVariable`, `StructuredOptimizationProblem`, `VariableKind`, `evaluateCandidate`, `runFeasibilityPipeline` |
| `src/des/general/feedback-linearization.ts` | `src/des/general/feedback_linearization.rs` | lib | `FeedbackLinearizationOpts`, `FeedbackLinearizationResult`, `PendulumParams`, `runFeedbackLinearization` |
| `src/des/general/field-station.ts` | `src/des/general/field_station.rs` | lib | `Census`, `FieldSimulation`, `FieldSimulationOptions`, `FieldSimulationResult`, `FieldStation`, `FieldUpdater`, `Station` |
| `src/des/general/four-rooms.ts` | `src/des/general/four_rooms.rs` | lib | `FourRoomsEnv`, `FourRoomsOpts`, `FourRoomsResult`, `FourRoomsTrainOpts`, `buildFourRoomsOptions`, `runFourRoomsSMDP` |
| `src/des/general/ga-des.ts` | `src/des/general/ga_des.rs` | lib | `GADESResult`, `TSPGAOptimizer`, `TSPGAOptions`, `runTSPGADES` |
| `src/des/general/general.ts` | `src/des/general/general.rs` | lib | `DESMap`, `DESSet`, `HasComputedProperties`, `bgn`, `deJSON`, `fisherYatesShuffle`, `getReasonableU`, `getReasonableUNative`, `getShortUUID`, `getSortedHistogram`, `getSortedHistogram_NEW`, `getSortedTimeHistogram`, `makeError`, `sendRaw` |
| `src/des/general/genetic-tsp.ts` | `src/des/general/genetic_tsp.rs` | lib | `GAPerformanceStats`, `GASolverOptions`, `GASolverResult`, `GenerationInfo`, `TSPInstance`, `Tour`, `buildPentagonTSP`, `buildRandomTSP`, `checkPrecedence`, `heldKarpExact`, `inversionMutate`, `isPermutation`, `oneTreeLowerBound`, `orderCrossover`, `repairPrecedence`, `runGeneticTSP`, `swapMutate`, `tourLength`, `tournamentSelect`, `twoOptImprove` |
| `src/des/general/grid-localization-pomdp.ts` | `src/des/general/grid_localization_pomdp.rs` | lib | `GridLocalizationAction`, `GridLocalizationActionKind`, `GridLocalizationObservation`, `GridLocalizationParams`, `GridLocalizationResult`, `GridLocalizationTraceRow`, `buildGridLocalizationPOMDP`, `runGridLocalizationPOMDP` |
| `src/des/general/hungarian.ts` | `src/des/general/hungarian.rs` | lib | `AssignmentDirection`, `AssignmentResult`, `hungarian` |
| `src/des/general/incremental-lp.ts` | `src/des/general/incremental_lp.rs` | lib | `IncrementalLP`, `IncrementalLPInit`, `LPEvent`, `LPSnapshot`, `PivotEvent` |
| `src/des/general/internal-solver-network.ts` | `src/des/general/internal_solver_network.rs` | lib | `InternalSolverKind`, `InternalSolverRunParams`, `InternalSolverRunResult`, `KnapsackDPStation`, `KnapsackParams`, `KnapsackSAStation`, `ObservableTSPGAOptimizer`, `ObservableTSPSAOptimizer`, `SOLUTION_CHANNEL`, `STOP_CHANNEL`, `ShortestPathSolverParams`, `ShortestPathSolverStation`, `SnapshotProvider`, `SolutionSinkStation`, `SolverNetworkDescription`, `SolverNetworkEdge`, `SolverNetworkNode`, `SolverProgressPayload`, `SolverSolutionToken`, `StopSignalToken`, `TSPHeldKarpStation`, `TSPSolverParams`, `WallClockCheckerStation`, `runInternalSolverNetwork` |
| `src/des/general/inventory-dp.ts` | `src/des/general/inventory_dp.rs` | lib | `InventoryDPResult`, `InventoryDPStation`, `InventoryProblem`, `simulateInventory`, `solveInventoryDP` |
| `src/des/general/ip-mip-des.ts` | `src/des/general/ip_mip_des.rs` | lib | `BranchAndCutSolverStation`, `BranchOrCutConstraint`, `ConcreteLPRelaxationAlgorithm`, `IPMIPPerformanceStats`, `IPMIPProblem`, `IPMIPProblemFeatures`, `IPMIPSolution`, `IPMIPSolveOptions`, `IPMIPSolverTechniquePlan`, `IPMIPTraceEvent`, `LPRelaxationAlgorithm`, `LPRelaxationStation`, `SolverTokenStats`, `SolverTopologyNode`, `analyzeIPMIPProblem`, `buildBinaryKnapsackIP`, `buildIPMIPSolverTechniquePlan`, `buildSmallMixedIP`, `solveIPMIPWithDES`, `validateIPMIPProblem` |
| `src/des/general/iterative-learning-control.ts` | `src/des/general/iterative_learning_control.rs` | lib | `ILCReferenceKind`, `ILCTrialSummary`, `IterativeLearningControlParams`, `IterativeLearningControlResult`, `runIterativeLearningControl` |
| `src/des/general/kalman-filter.ts` | `src/des/general/kalman_filter.rs` | lib | `KalmanFilterBlock`, `RadarTrackingOpts`, `RadarTrackingResult`, `runRadarTracking` |
| `src/des/general/learning-optimization-models.ts` | `src/des/general/learning_optimization_models.rs` | lib | `BackpropMLPParams`, `GradientTrainingResult`, `LinearRegressionParams`, `LinearRegressionResult`, `LogisticRegressionSGDParams`, `RidgeRegressionParams`, `SupervisedSample`, `multiclassAccuracy`, `runBackpropMLPClassifier`, `runLinearRegressionLS`, `runLogisticRegressionSGD`, `runRidgeRegressionLS` |
| `src/des/general/lp-des.ts` | `src/des/general/lp_des.rs` | lib | `DESSimplexOptions`, `DESSimplexSolution`, `DESSimplexTrace`, `solveLPViaDES` |
| `src/des/general/lp.ts` | `src/des/general/lp.rs` | lib | `ExternalSolverOptions`, `InternalSimplexOptions`, `LPProblem`, `LPSolution`, `LPStatus`, `lpToString`, `solveLP`, `solveLPExternal`, `solveLPInternal` |
| `src/des/general/math-blocks.ts` | `src/des/general/math_blocks.rs` | lib | `BlockGraphEdge`, `BlockGraphNode`, `BlockModelLogger`, `ComparatorBlock`, `ComparatorOp`, `ConstantSourceBlock`, `DerivativeBlock`, `ExpressionBlock`, `ExpressionSourceBlock`, `FirstOrderFilterBlock`, `FunctionSourceBlock`, `GainBlock`, `Heat1DBlockParams`, `Heat1DBlockResult`, `Heat1DTraceRow`, `IntegratorBlock`, `IntegratorMethod`, `Laplacian1DBlock`, `LogicBlock`, `LogicOp`, `MATH_IN`, `MATH_OUT`, `MathBlock`, `MathBlockOptions`, `MathBlockRunResult`, `MathSample`, `MathSignal`, `ODEBlockSystemParams`, `ODEBlockSystemResult`, `ODEStateSpec`, `ODETraceRow`, `ProductBlock`, `SaturationBlock`, `SinkBlock`, `SubtractBlock`, `SumBlock`, `runHeat1DBlockGrid`, `runMathBlockDiagram`, `runODEBlockSystem` |
| `src/des/general/math-equation-input.ts` | `src/des/general/math_equation_input.rs` | lib | `EquationInputFormat`, `EquationProblemKind`, `MathEquationInputParams`, `MathEquationNetwork`, `MathEquationResult`, `latexToExpression`, `normalizeMathEquationProblem`, `runMathEquationProblem` |
| `src/des/general/max-flow.ts` | `src/des/general/max_flow.rs` | lib | `MaxFlowEdge`, `MaxFlowProblem`, `MaxFlowResult`, `MaxFlowStation`, `MaxFlowTraceEntry`, `buildTextbookMaxFlowProblem`, `solveMaxFlow`, `validateMaxFlowProblem` |
| `src/des/general/mcts.ts` | `src/des/general/mcts.rs` | lib | `MCTSEnv`, `MCTSOptions`, `MCTSStation`, `mcts` |
| `src/des/general/milp-bnb.ts` | `src/des/general/milp_bnb.rs` | lib | `FacilityLocationProblem`, `MILPProblem`, `MILPSolution`, `MILPSolveOptions`, `NodeEvent`, `buildFacilityLocationMILP`, `buildKnapsackMILP`, `solveMILP` |
| `src/des/general/mountain-car.ts` | `src/des/general/mountain_car.rs` | lib | `MountainCarOpts`, `MountainCarResult`, `MountainCarTrainOpts`, `runMountainCar` |
| `src/des/general/mpc-double-integrator.ts` | `src/des/general/mpc_double_integrator.rs` | lib | `MPCDoubleIntOpts`, `MPCDoubleIntResult`, `runMPCDoubleIntegrator` |
| `src/des/general/mrac.ts` | `src/des/general/mrac.rs` | lib | `MRACOpts`, `MRACResult`, `runMRAC` |
| `src/des/general/multistage-stochastic.ts` | `src/des/general/multistage_stochastic.rs` | lib | `DemandOutcome`, `ExactTreeNodeResult`, `MultiStageInventoryProblem`, `MultiStageRunResult`, `SDDPIterationTrace`, `SDDPOptions`, `SDDPResult`, `SDDPStation`, `StageDecision`, `buildDefaultMultiStageInventoryProblem`, `evaluatePolicyExact`, `expectedStageValue`, `runMultiStageInventoryDemo`, `solveExactScenarioTree`, `solveMultiStageSDDP`, `solveStageDecision`, `validateMultiStageProblem` |
| `src/des/general/network-flow.ts` | `src/des/general/network_flow.rs` | lib | `AugmentingPathToken`, `CarToken`, `FlowEdge`, `FlowEdgeResult`, `MaxFlowMinCut`, `MaxFlowOptimizationStation`, `MaxFlowParams`, `MaxFlowResult`, `MaxFlowTraceRow`, `OptimizationLogger`, `TrafficCarSnapshot`, `TrafficCellStation`, `TrafficCellStats`, `TrafficGridStation`, `TrafficLane`, `TrafficNetwork`, `TrafficNode`, `TrafficNodeKind`, `TrafficParams`, `TrafficResult`, `TrafficScheduledTrip`, `TrafficSignal`, `TrafficSignalPhase`, `TrafficSink`, `TrafficSource`, `TrafficTraceRow`, `buildFiveIntersectionTrafficNetwork`, `runMaxFlow`, `runTrafficFlow` |
| `src/des/general/network-mutex.ts` | `src/des/general/network_mutex.rs` | lib | `LockGrantToken`, `LockReleaseToken`, `LockRequestToken`, `MUTEX_DONE_CHANNEL`, `MUTEX_GRANT_CHANNEL`, `MUTEX_RELEASE_CHANNEL`, `MUTEX_REQUEST_CHANNEL`, `MUTEX_WORK_CHANNEL`, `MutexChildState`, `MutexChildToken`, `MutexCompletionSinkStation`, `MutexSourceSpec`, `MutexWorkItem`, `MutexWorkSourceStation`, `MutexWorkState`, `NetworkMutexLockServiceOpts`, `NetworkMutexLockServiceStation`, `NetworkMutexLockStats`, `NetworkMutexSimulationOpts`, `NetworkMutexSimulationResult`, `NetworkMutexTraceEvent`, `NetworkMutexWorkerOpts`, `NetworkMutexWorkerStation`, `NetworkMutexWorkerStats`, `buildNetworkMutexStations`, `runNetworkMutexSimulation` |
| `src/des/general/neural-network.ts` | `src/des/general/neural_network.rs` | lib | `ActivationName`, `DenseLayerConfig`, `FeedForwardNetwork`, `NeuralODEOptions`, `NeuralODESolutionToken`, `NeuralODESolveToken`, `NeuralODESolverName`, `NeuralODESolverStation`, `NeuralPredictionSink`, `NeuralQLearningAgent`, `NeuralQLearningOptions`, `NeuralQLearningResult`, `StateEncoder`, `SupervisedDatasetSource`, `SupervisedNeuralNetDESResult`, `SupervisedSample`, `XOR_DATASET`, `XorNeuralNetOptions`, `oneHotEncoder`, `runNeuralQLearningDES`, `runSupervisedNeuralNetDES`, `runXorNeuralNetDES`, `solveNeuralODE` |
| `src/des/general/nonlinear-forecasting-model.ts` | `src/des/general/nonlinear_forecasting_model.rs` | lib | `FineTuneTraceRow`, `ForecastProjectionPoint`, `LatentBeliefPoint`, `LatentBeliefTrace`, `MDPDiscoveryStep`, `NonlinearMDPPOMDPForecastParams`, `NonlinearMDPPOMDPForecastResult`, `TunedEquation`, `VariableDiscoveryResult`, `runNonlinearMDPPOMDPForecast` |
| `src/des/general/nonlinear-optimization-models.ts` | `src/des/general/nonlinear_optimization_models.rs` | lib | `CurveFitPoint`, `NonlinearLeastSquaresParams`, `NonlinearLeastSquaresResult`, `NonlinearTopology`, `UnconstrainedOptParams`, `UnconstrainedOptResult`, `runBFGSRosenbrock`, `runGaussNewtonCurveFit`, `runLevenbergMarquardtCurveFit`, `runNewtonRosenbrock` |
| `src/des/general/ode.ts` | `src/des/general/ode.rs` | lib | `Jac`, `ODETrace`, `RHS`, `backwardEuler`, `euler`, `rk2Heun`, `rk4`, `rk45`, `secondOrderToFirstOrder` |
| `src/des/general/optim.ts` | `src/des/general/optim.rs` | lib | `OptimOptions`, `OptimResult`, `autoGradient`, `bfgs`, `gradientDescent`, `newtonOptim` |
| `src/des/general/pomdp.ts` | `src/des/general/pomdp.rs` | lib | `BeliefActionValue`, `BeliefLookaheadLeaf`, `BeliefLookaheadOptions`, `BeliefLookaheadSolver`, `MDPVIOptions`, `MDPVIResult`, `MostLikelyStateSolver`, `POMDPExactResult`, `POMDPSpec`, `QMDPSolver`, `beliefUpdate`, `expectedBeliefReward`, `mdpValueIteration`, `observationDistribution`, `pomdpExactFiniteHorizon` |
| `src/des/general/pontryagin-bang-bang.ts` | `src/des/general/pontryagin_bang_bang.rs` | lib | `PontryaginOpts`, `PontryaginResult`, `optimalTimeDoubleIntegrator`, `runPontryaginBangBang` |
| `src/des/general/ppo-des.ts` | `src/des/general/ppo_des.rs` | lib | `PPOClipUpdateStation`, `PPODESResult`, `PPOUpdateOptions`, `TabularPPOAgent`, `runPPODES` |
| `src/des/general/prng.ts` | `src/des/general/prng.rs` | lib | `mulberry32`, `withSeed` |
| `src/des/general/qlearning-des.ts` | `src/des/general/qlearning_des.rs` | lib | `QLearningAgent`, `QLearningOptions`, `QLearningResult`, `runQLearningDES` |
| `src/des/general/quadrature.ts` | `src/des/general/quadrature.rs` | lib | `QuadResult`, `adaptiveSimpson`, `gaussLegendre`, `monteCarlo`, `monteCarloND`, `simpson`, `trapezoidal` |
| `src/des/general/random-variables.ts` | `src/des/general/random_variables.rs` | lib | `bernoulliPMF`, `binomialPMF`, `competingRisks`, `discreteConvolve`, `discreteConvolveMany`, `discreteConvolveSelf`, `discretisePDF`, `meanFromPMF`, `normalizePMF`, `pmfTotalMass`, `poissonBinomialPMF`, `sampleCategorical`, `sampleExponential`, `sampleFromPMF`, `sampleGamma`, `samplePoisson`, `varianceFromPMF` |
| `src/des/general/rl-environments.ts` | `src/des/general/rl_environments.rs` | lib | `Corridor`, `Environment`, `GridWorld`, `evalPolicy` |
| `src/des/general/rl-learning-models.ts` | `src/des/general/rl_learning_models.rs` | lib | `ExpectedSarsaGridParams`, `ExpectedSarsaGridResult`, `PolicyGradientCorridorParams`, `PolicyGradientCorridorResult`, `RLTopology`, `runExpectedSarsaGridworld`, `runPolicyGradientCorridor` |
| `src/des/general/root.ts` | `src/des/general/root.rs` | lib | `RootResult`, `bisection`, `newton`, `secant` |
| `src/des/general/run.ts` | `src/des/general/run.rs` | lib |  |
| `src/des/general/sa-des.ts` | `src/des/general/sa_des.rs` | lib | `CoolingSchedule`, `SADESResult`, `TSPHillClimber`, `TSPSAOptimizer`, `TSPSAOptions`, `runTSPHillClimberDES`, `runTSPSADES`, `temperatureAt` |
| `src/des/general/shortest-path-des.ts` | `src/des/general/shortest_path_des.rs` | lib | `BellmanFordOptions`, `Edge`, `Graph`, `SPResult`, `buildRandomGraph`, `buildSmallChainGraph`, `reconstructPath`, `shortestPathBellmanFordDES`, `shortestPathDijkstraDES` |
| `src/des/general/signal-transforms.ts` | `src/des/general/signal_transforms.rs` | lib | `ComplexPoint`, `ComplexPointInput`, `ComplexValue`, `FourierTransformParams`, `LaplaceTransformParams`, `QuadratureRule`, `TransformContributionRecord`, `TransformEntityFrameworkSummary`, `TransformKind`, `TransformOutputPoint`, `TransformRunResult`, `TransformSampleRecord`, `ZTransformParams`, `formatComplex`, `runFourierTransform`, `runLaplaceTransform`, `runZTransform` |
| `src/des/general/simulated-annealing.ts` | `src/des/general/simulated_annealing.rs` | lib | `CoolingSchedule`, `KnapsackInstance`, `SAProblem`, `SAResult`, `SASolverOptions`, `SATickEvent`, `buildKnapsackSAProblem`, `buildTSPSAProblem`, `runSimulatedAnnealing`, `temperatureAt` |
| `src/des/general/sliding-mode-control.ts` | `src/des/general/sliding_mode_control.rs` | lib | `SlidingModeOpts`, `SlidingModeResult`, `runSlidingMode` |
| `src/des/general/smart-traffic-flow.ts` | `src/des/general/smart_traffic_flow.rs` | lib | `SmartTrafficAccident`, `SmartTrafficCar`, `SmartTrafficCarSnapshot`, `SmartTrafficCellStation`, `SmartTrafficCellStats`, `SmartTrafficExecutionStats`, `SmartTrafficFaultMode`, `SmartTrafficParams`, `SmartTrafficResult`, `SmartTrafficTraceRow`, `SmartTrafficWorldStation`, `runSmartTrafficFlow` |
| `src/des/general/soccer-rotation.ts` | `src/des/general/soccer_rotation.rs` | lib | `AffinityBuilderOptions`, `GoalEvent`, `LPRelaxedScheduleResult`, `MatchAggregate`, `MatchResult`, `MatchSimOptions`, `MemorylessMDPResult`, `Schedule`, `ScheduleEvaluation`, `SoccerIPMIPModel`, `SoccerIPMIPPolicyOptions`, `SoccerIPMIPPolicyResult`, `SoccerPOMDPFeatureOptions`, `SoccerPOMDPFeatureSummary`, `SoccerPOMDPPeriodFeature`, `SoccerProblem`, `SubEvent`, `buildSampleSoccerProblem`, `buildSoccerIPMIP`, `buildSoccerLP`, `evaluateSchedule`, `evaluateSoccerPOMDPFeatures`, `policyGreedyHungarian`, `policyIPMIPFeasible`, `policyLPRelaxed`, `policyMDPVI`, `policyMDPVIMemoryless`, `policyRandomSchedule`, `runManyMatches`, `scheduleFromSoccerIPMIPVector`, `simulateMatchDES`, `validateScheduleStructure`, `welchT` |
| `src/des/general/stag-hunt.ts` | `src/des/general/stag_hunt.rs` | lib | `StagHuntOpts`, `StagHuntResult`, `runStagHunt` |
| `src/des/general/statistical-optimization.ts` | `src/des/general/statistical_optimization.rs` | lib | `AdaptiveAlternative`, `AdaptiveSimOptParams`, `AdaptiveSimOptResult`, `AdaptiveSimulationOptimizerStation`, `AdaptiveTraceRow`, `AlternativeStats`, `CapacityExpansionSDDPStation`, `DemandRange`, `DemandScenario`, `DemandSpec`, `DistributionFamily`, `DistributionFitParams`, `DistributionFitResult`, `DistributionFitStation`, `EmpiricalPoint`, `FitMethod`, `FittedDistribution`, `OptimizationLogger`, `RiskCandidateResult`, `RiskCapacityParams`, `RiskCapacityResult`, `RiskCapacityStation`, `SDDPIteration`, `SDDPParams`, `SDDPResult`, `buildDemandScenarios`, `capacityProfit`, `fitDistribution`, `runAdaptiveSimOpt`, `runCapacityExpansionSDDP`, `runDistributionFit`, `runRiskCapacity`, `sampleDemandVector`, `sampleFittedDistribution` |
| `src/des/general/stochastic-flow-mdp.ts` | `src/des/general/stochastic_flow_mdp.rs` | lib | `FlowMDPAction`, `FlowMDPDecision`, `FlowMDPSimStep`, `FlowMDPState`, `StochasticFlowEdge`, `StochasticFlowMDPProblem`, `StochasticFlowMDPResult`, `StochasticFlowMDPStation`, `buildDefaultStochasticFlowMDPProblem`, `simulateStochasticFlowPolicy`, `solveStochasticFlowMDP`, `validateStochasticFlowMDPProblem` |
| `src/des/general/stochastic-lp.ts` | `src/des/general/stochastic_lp.rs` | lib | `BendersIteration`, `BendersOpts`, `SLPProblem`, `SLPSolveResult`, `Scenario`, `UniformDemandSpec`, `buildProductionSLP`, `buildProductionScenarios`, `mulberry32`, `solveProductionClosedForm`, `solveSLPBenders`, `solveSLPMonolithic`, `solveSubproblemWithDuals` |
| `src/des/general/temp-control.ts` | `src/des/general/temp_control.rs` | lib | `BangBangController`, `ControllerSpec`, `ControllerState`, `DEFAULT_HOUSE`, `DEFAULT_OUTDOOR`, `FuzzyController`, `HouseParams`, `MdpMpcController`, `OutdoorPattern`, `PIDController`, `RunResult`, `SimConfig`, `TempControllerBase`, `TempObs`, `TickRecord`, `controllerStep`, `fuzzyDeltaController`, `houseStep`, `makeTempController`, `mdpMPCController`, `mulberry32`, `runTempControl`, `trueOutdoorTemp` |
| `src/des/general/tiger-pomdp.ts` | `src/des/general/tiger_pomdp.rs` | lib | `ACT_LISTEN`, `ACT_OPEN_LEFT`, `ACT_OPEN_RIGHT`, `OBS_HEAR_LEFT`, `OBS_HEAR_RIGHT`, `OneStepLookAheadStation`, `QMDPStation`, `TIGER_LEFT`, `TIGER_RIGHT`, `TigerOpts`, `TigerSimOpts`, `TigerSimResult`, `buildTigerSpec`, `simulateTiger` |
| `src/des/general/time-accrued.ts` | `src/des/general/time_accrued.rs` | lib | `bumpTimeAccruedByTimeStep`, `getStepSize`, `getTimeAccrued`, `setStepSize` |
| `src/des/general/time-stepped-station.ts` | `src/des/general/time_stepped_station.rs` | lib | `BidirectionalTimeSteppedStation`, `BufferedTimeSteppedStation`, `RoutedTimeSteppedStation`, `SynchronousDataflowConnection`, `SynchronousDataflowStation`, `TimeSteppedStation` |
| `src/des/general/traffic-flow.ts` | `src/des/general/traffic_flow.rs` | lib | `IntersectionStation`, `RoadLinkStation`, `SignalAxis`, `TrafficCar`, `TrafficCarSnapshot`, `TrafficGridStation`, `TrafficLinkSpec`, `TrafficLinkStats`, `TrafficNodeSpec`, `TrafficProblem`, `TrafficSimulationResult`, `TrafficSourceSpec`, `TrafficTimeSample`, `buildDefaultTrafficProblem`, `buildTrafficMaxFlowProblem`, `runTrafficSimulation`, `validateTrafficProblem` |
| `src/des/general/universal-model-spec.ts` | `src/des/general/universal_model_spec.rs` | lib | `UniversalDESModelSpec`, `UniversalDESNetworkSpec`, `UniversalEndpointSpec`, `UniversalGraphEdge`, `UniversalInputFormat`, `UniversalMathCondition`, `UniversalMathEquation`, `UniversalMathParameter`, `UniversalMathSpec`, `UniversalMathVariable`, `UniversalModelKind`, `UniversalMovingEntity`, `UniversalNormalizedMath`, `UniversalNumericsSpec`, `UniversalOriginalInput`, `UniversalPortRef`, `UniversalSolverSpec`, `UniversalStationaryEntity`, `assertUniversalDESModelSpec`, `isUniversalDESModelSpec`, `universalFromMathEquationResult`, `universalToDESModelSpec`, `universalToMathEquationInput`, `validateUniversalDESModelSpec` |
| `src/des/general/value-iteration.ts` | `src/des/general/value_iteration.rs` | lib | `MDPSpec`, `Outcome`, `VIOptions`, `VIResult`, `ValueIterationStation`, `qValue`, `qValuesAll`, `valueIteration` |
| `src/des/http-server/index.ts` | `src/des/http_server/mod.rs` | lib | `getHTTPServer` |
| `src/des/main-backpropagation.ts` | `src/bin/main_backpropagation.rs` | bin | `BackpropResult`, `InitialWeights`, `initWeights`, `runBackprop` |
| `src/des/main-build-site.ts` | `src/bin/main_build_site.rs` | bin |  |
| `src/des/main-calculus.ts` | `src/bin/main_calculus.rs` | bin |  |
| `src/des/main-computer-network.ts` | `src/bin/main_computer_network.rs` | bin |  |
| `src/des/main-contact-seir.ts` | `src/bin/main_contact_seir.rs` | bin | `ContactSEIRParams`, `ContactSEIRResult`, `Kernel`, `Person`, `State`, `runContactSEIR` |
| `src/des/main-convolution.ts` | `src/bin/main_convolution.rs` | bin | `ConvolutionResult`, `runConvolution` |
| `src/des/main-court-mdp.ts` | `src/bin/main_court_mdp.rs` | bin | `AlwaysEscalatePolicy`, `CourtMDPConfig`, `CourtMDPResult`, `NaiveThresholdPolicy`, `OptimalPolicy`, `Policy`, `RejectAllPolicy`, `runCourtSim` |
| `src/des/main-dc-motor-anim.ts` | `src/bin/main_dc_motor_anim.rs` | bin |  |
| `src/des/main-dc-motor.ts` | `src/bin/main_dc_motor.rs` | bin |  |
| `src/des/main-dispatch-combo.ts` | `src/bin/main_dispatch_combo.rs` | bin |  |
| `src/des/main-electric-circuit.ts` | `src/bin/main_electric_circuit.rs` | bin | `RLCConfig`, `RLCResult`, `runRLC` |
| `src/des/main-elevator-highrise.ts` | `src/bin/main_elevator_highrise.rs` | bin | `DECISION_AUTHORITIES`, `DecisionAuthority`, `HIGHRISE_POLICIES`, `HighriseAggregates`, `HighriseBuilding`, `HighriseElevatorConfig`, `HighriseElevatorResult`, `HighrisePassengerSnapshot`, `HighrisePolicy`, `MDPDispatchTuningSummary`, `MDPObservability`, `MDPRunDiagnostics`, `MarginalComparison`, `buildHighriseSchedule`, `runHighriseElevators` |
| `src/des/main-elevator.ts` | `src/bin/main_elevator.rs` | bin | `Building`, `ElevatorConfig`, `ElevatorResult`, `buildSchedule`, `runElevator` |
| `src/des/main-empirical-control-report.ts` | `src/bin/main_empirical_control_report.rs` | bin |  |
| `src/des/main-empirical-control.ts` | `src/bin/main_empirical_control.rs` | bin |  |
| `src/des/main-epidemic-improved.ts` | `src/bin/main_epidemic_improved.rs` | bin |  |
| `src/des/main-epidemic.ts` | `src/bin/main_epidemic.rs` | bin | `MovingEntity` |
| `src/des/main-factmachine-markets.ts` | `src/bin/main_factmachine_markets.rs` | bin | `ClosedMarket`, `DailySummary`, `MarketKind`, `MarketKindAggregate`, `OperatorMDP`, `PolicyAggregate`, `PolicyRun`, `PortfolioConfig`, `SchedulerPolicy`, `buildDailyMarketCaps`, `buildOperatorMDP`, `dailyMarketCapForDay`, `dayIndex`, `defaultConfig`, `runPortfolio`, `scenarioConfigs` |
| `src/des/main-factmachine.ts` | `src/bin/main_factmachine.rs` | bin | `Bettor`, `FactMachineParams`, `FactMachineResult`, `LMSR`, `LMSROptions`, `defaultParams`, `liquidityToB`, `outcomeMatrix`, `runFactMachine` |
| `src/des/main-factory-floor-track3t.ts` | `src/bin/main_factory_floor_track3t.rs` | bin |  |
| `src/des/main-fibonacci-recursion.ts` | `src/bin/main_fibonacci_recursion.rs` | bin | `MovingEntity` |
| `src/des/main-from-json.ts` | `src/bin/main_from_json.rs` | bin |  |
| `src/des/main-genetic-tsp.ts` | `src/bin/main_genetic_tsp.rs` | bin |  |
| `src/des/main-hazard-function-survival-analysis.ts` | `src/bin/main_hazard_function_survival_analysis.rs` | bin |  |
| `src/des/main-incremental-lp.ts` | `src/bin/main_incremental_lp.rs` | bin |  |
| `src/des/main-inventory-mdp.ts` | `src/bin/main_inventory_mdp.rs` | bin | `InventoryParams`, `PolicyStructure`, `detectPolicyStructure`, `inventoryMDPSpec`, `simulateInventoryMDP` |
| `src/des/main-ip-mip-des.ts` | `src/bin/main_ip_mip_des.rs` | bin |  |
| `src/des/main-knapsack-problem.ts` | `src/bin/main_knapsack_problem.rs` | bin |  |
| `src/des/main-lp-des.ts` | `src/bin/main_lp_des.rs` | bin |  |
| `src/des/main-lp-factory.ts` | `src/bin/main_lp_factory.rs` | bin |  |
| `src/des/main-markov.ts` | `src/bin/main_markov.rs` | bin | `MovingEntity` |
| `src/des/main-mdp-lp.ts` | `src/bin/main_mdp_lp.rs` | bin |  |
| `src/des/main-milp-bnb.ts` | `src/bin/main_milp_bnb.rs` | bin |  |
| `src/des/main-monte-carlo-sim.ts` | `src/bin/main_monte_carlo_sim.rs` | bin |  |
| `src/des/main-network-mutex.ts` | `src/bin/main_network_mutex.rs` | bin |  |
| `src/des/main-neural-net.ts` | `src/bin/main_neural_net.rs` | bin |  |
| `src/des/main-newsvendor.ts` | `src/bin/main_newsvendor.rs` | bin | `DemandDist`, `NewsvendorParams`, `analyticalOptimalQ`, `bruteSearchOptimalQ`, `cdfFromPMF`, `demandPoissonPMF`, `demandUniformPMF`, `expectedProfit`, `mdpOptimalQ`, `meanFromDemand`, `newsvendorMDPSpec`, `profit`, `sampleDemand`, `simulate` |
| `src/des/main-observability-controllability-anim.ts` | `src/bin/main_observability_controllability_anim.rs` | bin |  |
| `src/des/main-observability-controllability.ts` | `src/bin/main_observability_controllability.rs` | bin |  |
| `src/des/main-optimization-as-des.ts` | `src/bin/main_optimization_as_des.rs` | bin |  |
| `src/des/main-plumbing-flow.ts` | `src/bin/main_plumbing_flow.rs` | bin |  |
| `src/des/main-shortest-path-algo.ts` | `src/bin/main_shortest_path_algo.rs` | bin |  |
| `src/des/main-shortest-path.ts` | `src/bin/main_shortest_path.rs` | bin |  |
| `src/des/main-signal-processing.ts` | `src/bin/main_signal_processing.rs` | bin |  |
| `src/des/main-simulated-annealing.ts` | `src/bin/main_simulated_annealing.rs` | bin |  |
| `src/des/main-snowball.ts` | `src/bin/main_snowball.rs` | bin |  |
| `src/des/main-soccer-rotation.ts` | `src/bin/main_soccer_rotation.rs` | bin |  |
| `src/des/main-stochastic-flow-mdp.ts` | `src/bin/main_stochastic_flow_mdp.rs` | bin |  |
| `src/des/main-stochastic-lp.ts` | `src/bin/main_stochastic_lp.rs` | bin |  |
| `src/des/main-stochastic-sde-report.ts` | `src/bin/main_stochastic_sde_report.rs` | bin |  |
| `src/des/main-stochastic-sde.ts` | `src/bin/main_stochastic_sde.rs` | bin |  |
| `src/des/main-temp-control-anim.ts` | `src/bin/main_temp_control_anim.rs` | bin |  |
| `src/des/main-temp-control.ts` | `src/bin/main_temp_control.rs` | bin |  |
| `src/des/main-traffic.ts` | `src/bin/main_traffic.rs` | bin |  |
| `src/des/main-two-disease.ts` | `src/bin/main_two_disease.rs` | bin | `CompartmentId`, `TwoDiseaseParams`, `TwoDiseaseResult`, `TwoDiseaseTrace`, `runTwoDisease` |
| `src/des/main-wind-mppt-anim.ts` | `src/bin/main_wind_mppt_anim.rs` | bin |  |
| `src/des/main-wind-mppt.ts` | `src/bin/main_wind_mppt.rs` | bin |  |
| `src/des/main.ts` | `src/bin/main.rs` | bin | `MovingEntity` |
| `src/des/max-flow.ts` | `src/des/max_flow.rs` | lib |  |
| `src/des/mdp/usacc-mdp.ts` | `src/des/mdp/usacc_mdp.rs` | lib | `ACCEPTED`, `ACTIONS`, `Action`, `CLOSED`, `CONFLICT`, `CORROBORATION`, `CaseState`, `Conflict`, `Corroboration`, `EVIDENCE`, `EXHAUSTED`, `Evidence`, `FUNDING`, `FUND_ACTIVE`, `FUND_ESCROWED`, `FUND_EXHAUSTED`, `FUND_UNFUNDED`, `Funding`, `MANIPULATION`, `Manipulation`, `N_ACTIONS`, `N_STATES`, `Outcome`, `STAGES`, `Stage`, `decode`, `encode`, `isTerminal`, `outcomes`, `quality`, `rewardOfAccept`, `rewardOfClose`, `sampleInitialState`, `terminalReward` |
| `src/des/mdp/value-iteration.ts` | `src/des/mdp/value_iteration.rs` | lib | `VIOptions`, `VIResult`, `buildTransitionTable`, `valueIteration` |
| `src/des/observability/logger.ts` | `src/des/observability/logger.rs` | lib | `BaseEvent`, `JsonlLogger`, `LogLevel`, `readEvents` |
| `src/des/observability/validate-epidemic.ts` | `src/des/observability/validate_epidemic.rs` | lib |  |
| `src/des/observers/program-observer.ts` | `src/des/observers/program_observer.rs` | lib | `ProgramObserver` |
| `src/des/parent.ts` | `src/des/parent.rs` | lib |  |
| `src/des/program.ts` | `src/des/program.rs` | lib | `getEntities` |
| `src/des/random-variables/generate.ts` | `src/des/random_variables/generate.rs` | lib | `runExponential`, `runUniform` |
| `src/des/random-variables/half.ts` | `src/des/random_variables/half.rs` | lib |  |
| `src/des/random-variables/rv.ts` | `src/des/random_variables/rv.rs` | lib | `BernoulliRandomVariable`, `ExponentialRandomVariable`, `ExponentialRandomVariable2`, `ExponentialRandomVariable3`, `PoissonRandomVariable`, `RandomVariable`, `UniformRandomVariable`, `UniformRandomVariable2` |
| `src/des/reference/compare-epidemic.ts` | `src/des/reference/compare_epidemic.rs` | lib |  |
| `src/des/reference/main-epidemic-fel.ts` | `src/des/reference/main_epidemic_fel.rs` | lib |  |
| `src/des/runners/compare-elevator-dispatch.ts` | `src/bin/compare_elevator_dispatch.rs` | bin |  |
| `src/des/runners/compare-external-fel-models.ts` | `src/bin/compare_external_fel_models.rs` | bin |  |
| `src/des/runners/compare-traffic-engines.ts` | `src/bin/compare_traffic_engines.rs` | bin |  |
| `src/des/runners/difference-runner.ts` | `src/des/runners/difference_runner.rs` | lib | `SteadyState`, `analyticalSteadyState`, `maxStableStep`, `runDifferenceOnce` |
| `src/des/runners/external-modules.ts` | `src/des/runners/external_modules.rs` | lib | `COMPUTER_NETWORK_FEL_REFERENCE_ID`, `COMPUTER_NETWORK_REFERENCE_ID`, `IP_MIP_REFERENCE_ID`, `NEURAL_NETWORK_REFERENCE_ID`, `TRAFFIC_CIW_REFERENCE_ID`, `TRAFFIC_FEL_REFERENCE_ID`, `TRAFFIC_SIMPY_REFERENCE_ID`, `TRAFFIC_SUMO_REFERENCE_ID`, `registerBuiltInExternalModules` |
| `src/des/runners/external-program.ts` | `src/des/runners/external_program.rs` | lib | `ExternalInterpreterSpec`, `ExternalModuleContext`, `ExternalModuleKind`, `ExternalModuleParams`, `ExternalParamValue`, `ExternalProgramModule`, `ExternalProgramResult`, `getExternalModule`, `listExternalModules`, `registerExternalModule`, `repoRootFromRunner`, `resolveExternalScript`, `runExternalModule`, `runExternalProgram`, `runPythonReference` |
| `src/des/runners/fel-runner.ts` | `src/des/runners/fel_runner.rs` | lib | `runFelOnce` |
| `src/des/runners/framework-runner.ts` | `src/des/runners/framework_runner.rs` | lib | `runFrameworkOnce` |
| `src/des/runners/gillespie-runner.ts` | `src/des/runners/gillespie_runner.rs` | lib | `runGillespieOnce` |
| `src/des/runners/ode-runner.ts` | `src/des/runners/ode_runner.rs` | lib | `runOdeOnce` |
| `src/des/runners/per-individual-runner.ts` | `src/des/runners/per_individual_runner.rs` | lib | `runPerIndividualOnce` |
| `src/des/runners/per-individual-vs-fel.ts` | `src/bin/per_individual_vs_fel.rs` | bin |  |
| `src/des/runners/replicate.ts` | `src/bin/replicate.rs` | bin |  |
| `src/des/runners/run-external-module.ts` | `src/bin/run_external_module.rs` | bin |  |
| `src/des/runners/shared.ts` | `src/des/runners/shared.rs` | lib | `TRANSITION_MATRIX_COLS`, `TRANSITION_MATRIX_ROWS`, `TransitionCountMap`, `TransitionCounter`, `TransitionTables`, `analyticalTransitionTables`, `averageRecord`, `buildTransitionTables`, `compartmentPopulations`, `meanResidence`, `updatePeaks`, `zeroCompartmentRecord` |
| `src/des/runners/stats.ts` | `src/des/runners/stats.rs` | lib | `WelchResult`, `mean`, `sampleVariance`, `stddev`, `welch` |
| `src/des/runners/steady-state.ts` | `src/bin/steady_state.rs` | bin |  |
| `src/des/runners/stepsize-sweep.ts` | `src/bin/stepsize_sweep.rs` | bin |  |
| `src/des/runners/types.ts` | `src/des/runners/types.rs` | lib | `COMPARTMENT_GROUPS`, `COMPARTMENT_ORDER`, `DEFAULT_CONFIG`, `DEFAULT_RESIDENCE`, `EDGES`, `Kernel`, `RunOpts`, `RunResult`, `SimConfig`, `buildSuccessors` |
| `src/des/runners/validate-backpropagation.ts` | `src/bin/validate_backpropagation.rs` | bin |  |
| `src/des/runners/validate-calculus.ts` | `src/bin/validate_calculus.rs` | bin |  |
| `src/des/runners/validate-computer-network.ts` | `src/bin/validate_computer_network.rs` | bin |  |
| `src/des/runners/validate-contact-vs-meanfield.ts` | `src/bin/validate_contact_vs_meanfield.rs` | bin |  |
| `src/des/runners/validate-convolution.ts` | `src/bin/validate_convolution.rs` | bin |  |
| `src/des/runners/validate-court-mdp.ts` | `src/bin/validate_court_mdp.rs` | bin |  |
| `src/des/runners/validate-dispatch.ts` | `src/bin/validate_dispatch.rs` | bin |  |
| `src/des/runners/validate-electric-circuit.ts` | `src/bin/validate_electric_circuit.rs` | bin |  |
| `src/des/runners/validate-elevator.ts` | `src/bin/validate_elevator.rs` | bin |  |
| `src/des/runners/validate-external-fel-models.ts` | `src/bin/validate_external_fel_models.rs` | bin |  |
| `src/des/runners/validate-factmachine-math.ts` | `src/bin/validate_factmachine_math.rs` | bin |  |
| `src/des/runners/validate-factmachine.ts` | `src/bin/validate_factmachine.rs` | bin |  |
| `src/des/runners/validate-genetic-tsp.ts` | `src/bin/validate_genetic_tsp.rs` | bin |  |
| `src/des/runners/validate-incremental-lp.ts` | `src/bin/validate_incremental_lp.rs` | bin |  |
| `src/des/runners/validate-ip-mip-external.ts` | `src/bin/validate_ip_mip_external.rs` | bin |  |
| `src/des/runners/validate-lp.ts` | `src/bin/validate_lp.rs` | bin |  |
| `src/des/runners/validate-milp-bnb.ts` | `src/bin/validate_milp_bnb.rs` | bin |  |
| `src/des/runners/validate-neural-network.ts` | `src/bin/validate_neural_network.rs` | bin |  |
| `src/des/runners/validate-newsvendor.ts` | `src/bin/validate_newsvendor.rs` | bin |  |
| `src/des/runners/validate-optimization-as-des.ts` | `src/bin/validate_optimization_as_des.rs` | bin |  |
| `src/des/runners/validate-references.ts` | `src/bin/validate_references.rs` | bin |  |
| `src/des/runners/validate-shortest-path.ts` | `src/bin/validate_shortest_path.rs` | bin |  |
| `src/des/runners/validate-simulated-annealing.ts` | `src/bin/validate_simulated_annealing.rs` | bin |  |
| `src/des/runners/validate-smart-traffic-external.ts` | `src/bin/validate_smart_traffic_external.rs` | bin |  |
| `src/des/runners/validate-soccer.ts` | `src/bin/validate_soccer.rs` | bin |  |
| `src/des/runners/validate-stochastic-lp.ts` | `src/bin/validate_stochastic_lp.rs` | bin |  |
| `src/des/runners/validate-temp-control.ts` | `src/bin/validate_temp_control.rs` | bin |  |
| `src/des/runners/validate-two-disease.ts` | `src/bin/validate_two_disease.rs` | bin |  |
| `src/des/runners/validate-with-externals.ts` | `src/bin/validate_with_externals.rs` | bin |  |
| `src/des/signals/abstract.ts` | `src/des/signals/abstract.rs` | lib | `SignalEntity`, `SignalEntityGraphData`, `SignalMarker`, `SignalTimeStepOpts` |
| `src/des/signals/adder.ts` | `src/des/signals/adder.rs` | lib | `Adder`, `IntegratorTimeStepOpts` |
| `src/des/signals/differential.ts` | `src/des/signals/differential.rs` | lib | `DifferentialTimeStepOpts`, `Differentiator` |
| `src/des/signals/incrementer.ts` | `src/des/signals/incrementer.rs` | lib | `IncrementorTimeStepOpts`, `SignalIncrementor` |
| `src/des/signals/integral.ts` | `src/des/signals/integral.rs` | lib | `Integrator`, `IntegratorTimeStepOpts` |
| `src/des/signals/multi-directional-signal-entity.ts` | `src/des/signals/multi_directional_signal_entity.rs` | lib | `MultiDirectionalSignalEntity` |
| `src/des/signals/mux.ts` | `src/des/signals/mux.rs` | lib | `Multiplexer`, `MultiplexerTimeStepOpts` |
| `src/des/signals/signal-value.ts` | `src/des/signals/signal_value.rs` | lib | `AbstractSignalValue`, `SignalValue`, `SignalValueUnity`, `SignalValueZero` |
| `src/des/signals/single-direction-signal-entity.ts` | `src/des/signals/single_direction_signal_entity.rs` | lib | `SingleInManyOutSignalEntity` |
| `src/des/test/advanced-optimization-control-test.ts` | `tests/advanced_optimization_control_test.rs` | test |  |
| `src/des/test/animation-test.ts` | `tests/animation_test.rs` | test |  |
| `src/des/test/argmax-tiebreak-test.ts` | `tests/argmax_tiebreak_test.rs` | test |  |
| `src/des/test/calculus-test.ts` | `tests/calculus_test.rs` | test |  |
| `src/des/test/classical-optimization-test.ts` | `tests/classical_optimization_test.rs` | test |  |
| `src/des/test/collaborative-inference-test.ts` | `tests/collaborative_inference_test.rs` | test |  |
| `src/des/test/computer-network-test.ts` | `tests/computer_network_test.rs` | test |  |
| `src/des/test/dc-motor-test.ts` | `tests/dc_motor_test.rs` | test |  |
| `src/des/test/dispatch-test.ts` | `tests/dispatch_test.rs` | test |  |
| `src/des/test/domain-application-test.ts` | `tests/domain_application_test.rs` | test |  |
| `src/des/test/elevator-invariants-test.ts` | `tests/elevator_invariants_test.rs` | test |  |
| `src/des/test/empirical-control-test.ts` | `tests/empirical_control_test.rs` | test |  |
| `src/des/test/external-module-test.ts` | `tests/external_module_test.rs` | test |  |
| `src/des/test/factmachine-markets-test.ts` | `tests/factmachine_markets_test.rs` | test |  |
| `src/des/test/factmachine-math-test.ts` | `tests/factmachine_math_test.rs` | test |  |
| `src/des/test/factory-floor-track3t-test.ts` | `tests/factory_floor_track3t_test.rs` | test |  |
| `src/des/test/feasibility-pipeline-test.ts` | `tests/feasibility_pipeline_test.rs` | test |  |
| `src/des/test/float-bias-test.ts` | `tests/float_bias_test.rs` | test |  |
| `src/des/test/genetic-tsp-test.ts` | `tests/genetic_tsp_test.rs` | test |  |
| `src/des/test/incremental-lp-test.ts` | `tests/incremental_lp_test.rs` | test |  |
| `src/des/test/internal-solver-network-test.ts` | `tests/internal_solver_network_test.rs` | test |  |
| `src/des/test/ip-mip-des-test.ts` | `tests/ip_mip_des_test.rs` | test |  |
| `src/des/test/iterator-test.ts` | `tests/iterator_test.rs` | test |  |
| `src/des/test/learning-optimization-test.ts` | `tests/learning_optimization_test.rs` | test |  |
| `src/des/test/lp-test.ts` | `tests/lp_test.rs` | test |  |
| `src/des/test/math-blocks-test.ts` | `tests/math_blocks_test.rs` | test |  |
| `src/des/test/mdp-adjacent-test.ts` | `tests/mdp_adjacent_test.rs` | test |  |
| `src/des/test/milp-bnb-test.ts` | `tests/milp_bnb_test.rs` | test |  |
| `src/des/test/multistage-stochastic-test.ts` | `tests/multistage_stochastic_test.rs` | test |  |
| `src/des/test/network-flow-test.ts` | `tests/network_flow_test.rs` | test |  |
| `src/des/test/network-mutex-test.ts` | `tests/network_mutex_test.rs` | test |  |
| `src/des/test/neural-animation-test.ts` | `tests/neural_animation_test.rs` | test |  |
| `src/des/test/neural-network-test.ts` | `tests/neural_network_test.rs` | test |  |
| `src/des/test/newsvendor-test.ts` | `tests/newsvendor_test.rs` | test |  |
| `src/des/test/nonlinear-forecasting-test.ts` | `tests/nonlinear_forecasting_test.rs` | test |  |
| `src/des/test/nonlinear-optimization-test.ts` | `tests/nonlinear_optimization_test.rs` | test |  |
| `src/des/test/observability-controllability-test.ts` | `tests/observability_controllability_test.rs` | test |  |
| `src/des/test/optimal-control-test.ts` | `tests/optimal_control_test.rs` | test |  |
| `src/des/test/optimization-as-des-test.ts` | `tests/optimization_as_des_test.rs` | test |  |
| `src/des/test/output-routing-policy-test.ts` | `tests/output_routing_policy_test.rs` | test |  |
| `src/des/test/preconditions-test.ts` | `tests/preconditions_test.rs` | test |  |
| `src/des/test/queue-bias-test.ts` | `tests/queue_bias_test.rs` | test |  |
| `src/des/test/random-variables-test.ts` | `tests/random_variables_test.rs` | test |  |
| `src/des/test/shortest-path-test.ts` | `tests/shortest_path_test.rs` | test |  |
| `src/des/test/signal-transforms-test.ts` | `tests/signal_transforms_test.rs` | test |  |
| `src/des/test/simulated-annealing-test.ts` | `tests/simulated_annealing_test.rs` | test |  |
| `src/des/test/soccer-test.ts` | `tests/soccer_test.rs` | test |  |
| `src/des/test/statistical-optimization-test.ts` | `tests/statistical_optimization_test.rs` | test |  |
| `src/des/test/stochastic-lp-test.ts` | `tests/stochastic_lp_test.rs` | test |  |
| `src/des/test/stochastic-sde-test.ts` | `tests/stochastic_sde_test.rs` | test |  |
| `src/des/test/temp-control-test.ts` | `tests/temp_control_test.rs` | test |  |
| `src/des/test/test.ts` | `tests/test.rs` | test |  |
| `src/des/test/transform-entity-test.ts` | `tests/transform_entity_test.rs` | test |  |
| `src/des/test/ts-test.ts` | `tests/ts_test.rs` | test |  |
| `src/des/test/universal-model-spec-test.ts` | `tests/universal_model_spec_test.rs` | test |  |
| `src/des/test/validation-test.ts` | `tests/validation_test.rs` | test |  |
| `src/des/test/visual-block-test.ts` | `tests/visual_block_test.rs` | test |  |
| `src/des/test/wind-mppt-test.ts` | `tests/wind_mppt_test.rs` | test |  |
| `src/des/visual/visual-node.ts` | `src/des/visual/visual_node.rs` | lib | `ManyInManyOut`, `OneInManyOut`, `OneInOneOut`, `VisualConnection`, `VisualNode`, `VisualNodeEvents`, `VisualNodeObserver`, `ZeroInManyOut`, `ZeroOutManyIn` |
| `src/des/ws-server/ws-server.ts` | `src/des/ws_server/ws_server.rs` | lib | `getWebsocketServer`, `wss` |
