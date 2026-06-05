//! Port of src/des/test/factory-floor-track3t-test.ts
//!
//! Tests the warehouse MDP/POMDP comparison (`general/factory-floor-track3t`):
//! POMDP state-space / action-set cardinality identities, the sharper Track3t
//! observation model, the end-to-end comparison metrics (cycle time, throughput,
//! search misses, shipping error rate, belief entropy, trace), and the rendered
//! Track3t animation scene.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::factory_floor_track3t::{
        baseline_warehouse_scenario, build_warehouse_pomdp, default_warehouse_layout,
        run_warehouse_comparison, track3t_warehouse_scenario, WarehouseSimulationOptions,
    };
    use crate::des::model::track3t_warehouse::build_track3t_animation;

    #[test]
    fn pomdp_state_space_includes_terminal_state() {
        let layout = default_warehouse_layout();
        let dest = layout
            .stations
            .iter()
            .position(|s| s.id == "shipping")
            .expect("shipping station");
        let base_model = build_warehouse_pomdp(&layout, &baseline_warehouse_scenario(), dest);
        let n = layout.stations.len();
        assert_eq!(
            base_model.states.len(),
            n * n * 2 + 1,
            "states={}",
            base_model.states.len()
        );
    }

    #[test]
    fn action_set_drives_to_every_stationary_entity() {
        let layout = default_warehouse_layout();
        let dest = layout
            .stations
            .iter()
            .position(|s| s.id == "shipping")
            .expect("shipping station");
        let base_model = build_warehouse_pomdp(&layout, &baseline_warehouse_scenario(), dest);
        assert_eq!(base_model.actions.len(), layout.stations.len());
    }

    #[test]
    fn track3t_observation_model_is_sharper_than_baseline() {
        let track = track3t_warehouse_scenario();
        let base = baseline_warehouse_scenario();
        assert!(
            track.location_accuracy > base.location_accuracy,
            "{} > {}",
            track.location_accuracy,
            base.location_accuracy
        );
    }

    #[test]
    fn both_models_use_same_hidden_state_cardinality() {
        let layout = default_warehouse_layout();
        let dest = layout
            .stations
            .iter()
            .position(|s| s.id == "shipping")
            .expect("shipping station");
        let base_model = build_warehouse_pomdp(&layout, &baseline_warehouse_scenario(), dest);
        let track_model = build_warehouse_pomdp(&layout, &track3t_warehouse_scenario(), dest);
        assert_eq!(track_model.states.len(), base_model.states.len());
    }

    #[test]
    fn track3t_comparison_improves_every_metric() {
        let result = run_warehouse_comparison(WarehouseSimulationOptions {
            jobs: Some(120),
            seed: Some(7),
            record_trace: Some(true),
            ..Default::default()
        });
        let b = &result.baseline.metrics;
        let t = &result.track3t.metrics;

        // Track3t improves mean cycle time.
        assert!(
            t.mean_cycle_time < b.mean_cycle_time,
            "baseline={}, track3t={}",
            b.mean_cycle_time,
            t.mean_cycle_time
        );
        // Track3t improves throughput.
        assert!(
            t.throughput_per_hour > b.throughput_per_hour,
            "baseline={}, track3t={}",
            b.throughput_per_hour,
            t.throughput_per_hour
        );
        // Track3t reduces search misses.
        assert!(
            t.mean_search_misses_per_job < b.mean_search_misses_per_job,
            "baseline={}, track3t={}",
            b.mean_search_misses_per_job,
            t.mean_search_misses_per_job
        );
        // Track3t reduces shipping error rate.
        assert!(
            t.shipping_error_rate < b.shipping_error_rate,
            "baseline={}, track3t={}",
            b.shipping_error_rate,
            t.shipping_error_rate
        );
        // Larger sample keeps nonzero residual Track3t errors.
        assert!(
            t.shipping_errors > 0 && result.deltas.error_reduction_pct < 100.0,
            "track3t errors={}, reduction={}",
            t.shipping_errors,
            result.deltas.error_reduction_pct
        );
        // Track3t keeps a lower belief entropy.
        assert!(
            t.mean_belief_entropy < b.mean_belief_entropy,
            "baseline={}, track3t={}",
            b.mean_belief_entropy,
            t.mean_belief_entropy
        );
        // Comparison trace records both scenarios.
        assert!(
            !result.baseline.trace.is_empty() && !result.track3t.trace.is_empty(),
            "frames={}/{}",
            result.baseline.trace.len(),
            result.track3t.trace.len()
        );
    }

    #[test]
    fn track3t_animation_scene_builds_frames_and_charts() {
        let result = run_warehouse_comparison(WarehouseSimulationOptions {
            jobs: Some(8),
            seed: Some(7),
            record_trace: Some(true),
            ..Default::default()
        });
        let animation = build_track3t_animation(&result, 2, Some(24));

        assert!(!animation.frames.is_empty(), "no animation frames");
        assert!(
            !animation.frames[0].shapes.is_empty(),
            "first animation frame has no shapes"
        );
        assert_eq!(animation.charts.len(), 2);
    }
}
