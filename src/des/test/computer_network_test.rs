//! Port of src/des/test/computer-network-test.ts
//
// The `general/computer_network` model is now ported, so this exercises it
// directly (the TS variant also drove it through the des-registry `runFromSpec`,
// which is covered separately by the registry's own tests).

#[cfg(test)]
mod tests {
    use crate::des::general::computer_network::{
        build_bottleneck_computer_network_problem, run_computer_network_simulation,
        validate_computer_network_problem,
    };

    #[test]
    fn bottleneck_problem_validates() {
        let problem = build_bottleneck_computer_network_problem();
        assert!(validate_computer_network_problem(&problem).is_ok());
    }

    #[test]
    fn bottleneck_simulation_conserves_packets() {
        let problem = build_bottleneck_computer_network_problem();
        let result = run_computer_network_simulation(&problem);
        // Every generated packet is delivered, dropped, or still in flight.
        let conserved =
            result.delivered_packets + result.dropped_packets + result.active_packets;
        assert!(
            (result.generated_packets - conserved).abs() < 0.5,
            "packet conservation violated: generated={} conserved={}",
            result.generated_packets,
            conserved
        );
        assert!(result.generated_packets > 0.0, "source generated no packets");
        assert_eq!(result.node_stats.len(), 5);
    }
}
