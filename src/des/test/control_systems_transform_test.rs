#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::control_systems::lagrange::{
        generalized_acceleration, lagrange_to_state_space, LagrangeSecondOrderSystem,
    };
    use crate::des::general::control_systems::transforms::{
        engineering_core_trio, transform_descriptor, ControlAnalysisDomain, TransformKind,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    #[test]
    fn transform_descriptors_unify_engineering_core() {
        assert_eq!(
            engineering_core_trio(),
            [
                TransformKind::Fourier,
                TransformKind::Laplace,
                TransformKind::Z
            ]
        );
        let laplace = transform_descriptor(TransformKind::Laplace);
        assert_eq!(laplace.domain, ControlAnalysisDomain::ContinuousTime);
        assert!(laplace.control_use.contains("transfer functions"));

        let z = transform_descriptor(TransformKind::Z);
        assert_eq!(z.domain, ControlAnalysisDomain::DiscreteTime);
        assert!(z.diagonalizes.contains("difference"));

        let wavelet = transform_descriptor(TransformKind::Wavelet);
        assert_eq!(wavelet.domain, ControlAnalysisDomain::TimeFrequency);
    }

    #[test]
    fn lagrange_second_order_system_becomes_state_space() {
        let system = LagrangeSecondOrderSystem {
            mass: vec![vec![2.0]],
            damping: vec![vec![1.0]],
            stiffness: vec![vec![8.0]],
            input: vec![vec![1.0]],
            force_bias: Some(vec![4.0]),
        };

        let ss = lagrange_to_state_space(&system);
        assert_eq!(ss.a.len(), 2);
        assert!(close(ss.a[0][1], 1.0, 1e-12));
        assert!(close(ss.a[1][0], -4.0, 1e-12));
        assert!(close(ss.a[1][1], -0.5, 1e-12));
        assert!(close(ss.b[0][0], 0.0, 1e-12));
        assert!(close(ss.b[1][0], 0.5, 1e-12));
        assert!(close(ss.bias[1], 2.0, 1e-12));

        let qdd = generalized_acceleration(&system, &[1.0], &[2.0], &[6.0]);
        assert!(close(qdd[0], 0.0, 1e-12));
    }
}
