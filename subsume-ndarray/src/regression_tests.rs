//! Regression tests for performance benchmarks.
//!
//! These tests track performance metrics over time to detect regressions.
//! They should be run in CI/CD and fail if performance degrades significantly.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use subsume_ndarray::NdarrayBox;
    use subsume_core::Box;
    use std::time::Instant;

    /// Baseline performance for volume computation (in microseconds).
    /// Update this if legitimate optimizations improve performance.
    const BASELINE_VOLUME_TIME_US: u128 = 10;

    /// Baseline performance for containment probability (in microseconds).
    const BASELINE_CONTAINMENT_TIME_US: u128 = 15;

    /// Maximum allowed regression factor (e.g., 2.0 = 2x slower is acceptable).
    const MAX_REGRESSION_FACTOR: f64 = 2.0;

    #[test]
    fn test_volume_performance_regression() {
        let box_ = NdarrayBox::new(
            array![0.0, 0.0, 0.0],
            array![1.0, 1.0, 1.0],
            1.0,
        ).unwrap();

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = box_.volume(1.0).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() / 1000;

        let regression_factor = avg_time_us as f64 / BASELINE_VOLUME_TIME_US as f64;
        
        assert!(
            regression_factor <= MAX_REGRESSION_FACTOR,
            "Volume computation regression: {:.2}x slower than baseline ({} us vs {} us)",
            regression_factor,
            avg_time_us,
            BASELINE_VOLUME_TIME_US
        );
    }

    #[test]
    fn test_containment_performance_regression() {
        let box_a = NdarrayBox::new(
            array![0.0, 0.0, 0.0],
            array![1.0, 1.0, 1.0],
            1.0,
        ).unwrap();

        let box_b = NdarrayBox::new(
            array![0.2, 0.2, 0.2],
            array![0.8, 0.8, 0.8],
            1.0,
        ).unwrap();

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = box_a.containment_prob(&box_b, 1.0).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() / 1000;

        let regression_factor = avg_time_us as f64 / BASELINE_CONTAINMENT_TIME_US as f64;
        
        assert!(
            regression_factor <= MAX_REGRESSION_FACTOR,
            "Containment computation regression: {:.2}x slower than baseline ({} us vs {} us)",
            regression_factor,
            avg_time_us,
            BASELINE_CONTAINMENT_TIME_US
        );
    }

    /// Test that mathematical properties still hold (regression test for correctness).
    #[test]
    fn test_containment_reflexivity_regression() {
        let box_ = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();

        // A box should always contain itself
        let prob = box_.containment_prob(&box_, 1.0).unwrap();
        assert!(
            (prob - 1.0).abs() < 1e-5,
            "Containment reflexivity regression: P(A ⊆ A) = {}, expected 1.0",
            prob
        );
    }

    /// Test that volume is non-negative (regression test for correctness).
    #[test]
    fn test_volume_non_negative_regression() {
        let box_ = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();

        let volume = box_.volume(1.0).unwrap();
        assert!(
            volume >= 0.0,
            "Volume non-negativity regression: volume = {}, expected >= 0.0",
            volume
        );
    }
}

