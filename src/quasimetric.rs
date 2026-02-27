//! Interval Quasimetric Embeddings (IQE).
//!
//! IQE models asymmetric reachability by representing points as boxes (intervals).
//! The distance d(u, v) is the non-negative "gap" that must be bridged to reach v from u.

/// Interval Quasimetric Embedding (IQE).
///
/// IQE models asymmetric reachability by representing points as boxes (intervals).
/// The distance d(u, v) is the non-negative "gap" that must be bridged to reach v from u.
#[derive(Debug, Clone)]
pub struct IntervalEmbedding {
    /// Lower bounds for each dimension.
    pub min: Vec<f64>,
    /// Upper bounds for each dimension.
    pub max: Vec<f64>,
}

impl IntervalEmbedding {
    /// Create a new interval embedding.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Self {
        debug_assert_eq!(min.len(), max.len(), "Dimension mismatch");
        Self { min, max }
    }

    /// Compute the reachability from self to target.
    ///
    /// This is the "Soft-Interval" distance: how much of the target interval
    /// lies outside our current interval.
    pub fn reachability(&self, target: &Self, alpha: f64) -> f64 {
        let dim = self.min.len();
        if dim == 0 {
            return 0.0;
        }

        let mut total_dist = 0.0;
        let mut max_dist = 0.0;

        for i in 0..dim {
            // Gap logic: how much of the target interval lies *outside* our interval.
            // This is 0 if `self` contains `target` in this dimension.
            //
            // If target extends left of us: gap = self.min - target.min
            let left_gap = (self.min[i] - target.min[i]).max(0.0);
            let right_gap = (target.max[i] - self.max[i]).max(0.0);
            let gap = left_gap + right_gap;

            total_dist += gap;
            if gap > max_dist {
                max_dist = gap;
            }
        }

        // 2026 Bleed: Weighted combination of Mean and Max gaps
        // This ensures the triangle inequality holds while maintaining sensitivity
        // to individual dimension "bottlenecks".
        alpha * (total_dist / dim as f64) + (1.0 - alpha) * max_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reachability_containment() {
        let u = IntervalEmbedding::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        let v = IntervalEmbedding::new(vec![2.0, 2.0], vec![8.0, 8.0]);

        // u contains v, so reachability u -> v should be 0
        assert_eq!(u.reachability(&v, 0.5), 0.0);

        // v does not contain u, so reachability v -> u should be > 0
        assert!(v.reachability(&u, 0.5) > 0.0);
    }

    #[test]
    fn test_reachability_triangle_inequality() {
        let u = IntervalEmbedding::new(vec![0.0], vec![1.0]);
        let v = IntervalEmbedding::new(vec![2.0], vec![3.0]);
        let w = IntervalEmbedding::new(vec![4.0], vec![5.0]);

        let uv = u.reachability(&v, 1.0);
        let vw = v.reachability(&w, 1.0);
        let uw = u.reachability(&w, 1.0);

        // d(u, w) <= d(u, v) + d(v, w)
        assert!(uw <= uv + vw + 1e-6);
    }

    #[test]
    fn test_reachability_self_is_zero() {
        let u = IntervalEmbedding::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
        let d = u.reachability(&u, 0.5);
        assert!(
            d.abs() < 1e-10,
            "reachability to self should be 0, got {d}"
        );
    }

    #[test]
    fn test_reachability_asymmetric() {
        // u contains v, so u->v = 0, but v->u > 0
        let u = IntervalEmbedding::new(vec![0.0], vec![10.0]);
        let v = IntervalEmbedding::new(vec![3.0], vec![7.0]);
        let d_uv = u.reachability(&v, 0.5);
        let d_vu = v.reachability(&u, 0.5);
        assert!(d_uv.abs() < 1e-10, "u contains v: d(u,v) should be 0, got {d_uv}");
        assert!(d_vu > 0.0, "v does not contain u: d(v,u) should be > 0, got {d_vu}");
    }

    #[test]
    fn test_reachability_zero_dim() {
        let u = IntervalEmbedding::new(vec![], vec![]);
        let v = IntervalEmbedding::new(vec![], vec![]);
        let d = u.reachability(&v, 0.5);
        assert!(d.abs() < 1e-10, "zero-dim reachability should be 0, got {d}");
    }

    #[test]
    fn test_reachability_alpha_extremes() {
        let u = IntervalEmbedding::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let v = IntervalEmbedding::new(vec![2.0, 5.0], vec![3.0, 6.0]);
        // alpha=1 => pure mean; alpha=0 => pure max
        let d_mean = u.reachability(&v, 1.0);
        let d_max = u.reachability(&v, 0.0);
        // gaps: dim0: left=0, right=max(0,3-1)=2, gap=2
        //        dim1: left=0, right=max(0,6-1)=5, gap=5
        // total=7, max=5, dim=2
        // alpha=1: 1.0*(7/2) + 0.0*5 = 3.5
        // alpha=0: 0.0*(7/2) + 1.0*5 = 5.0
        assert!((d_mean - 3.5).abs() < 1e-10, "alpha=1 should be mean gap = 3.5, got {d_mean}");
        assert!((d_max - 5.0).abs() < 1e-10, "alpha=0 should be max gap = 5.0, got {d_max}");
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_reachability_non_negative(
            min1 in prop::collection::vec(-10.0f64..10.0, 1..16),
            max1 in prop::collection::vec(-10.0f64..10.0, 1..16),
            min2 in prop::collection::vec(-10.0f64..10.0, 1..16),
            max2 in prop::collection::vec(-10.0f64..10.0, 1..16),
            alpha in 0.0f64..1.0
        ) {
            let n = min1.len().min(max1.len()).min(min2.len()).min(max2.len());
            // Ensure min <= max
            let u_min: Vec<f64> = min1[..n].iter().zip(&max1[..n]).map(|(&a, &b)| a.min(b)).collect();
            let u_max: Vec<f64> = min1[..n].iter().zip(&max1[..n]).map(|(&a, &b)| a.max(b)).collect();
            let v_min: Vec<f64> = min2[..n].iter().zip(&max2[..n]).map(|(&a, &b)| a.min(b)).collect();
            let v_max: Vec<f64> = min2[..n].iter().zip(&max2[..n]).map(|(&a, &b)| a.max(b)).collect();

            let u = IntervalEmbedding::new(u_min, u_max);
            let v = IntervalEmbedding::new(v_min, v_max);

            let d = u.reachability(&v, alpha);
            prop_assert!(d >= 0.0);
        }

        #[test]
        fn prop_reachability_triangle_inequality(
            min1 in prop::collection::vec(-10.0f64..10.0, 1..8),
            max1 in prop::collection::vec(-10.0f64..10.0, 1..8),
            min2 in prop::collection::vec(-10.0f64..10.0, 1..8),
            max2 in prop::collection::vec(-10.0f64..10.0, 1..8),
            min3 in prop::collection::vec(-10.0f64..10.0, 1..8),
            max3 in prop::collection::vec(-10.0f64..10.0, 1..8),
            alpha in 0.0f64..1.0
        ) {
            let n = min1.len().min(max1.len()).min(min2.len()).min(max2.len()).min(min3.len()).min(max3.len());
            let u_min: Vec<f64> = min1[..n].iter().zip(&max1[..n]).map(|(&a, &b)| a.min(b)).collect();
            let u_max: Vec<f64> = min1[..n].iter().zip(&max1[..n]).map(|(&a, &b)| a.max(b)).collect();
            let v_min: Vec<f64> = min2[..n].iter().zip(&max2[..n]).map(|(&a, &b)| a.min(b)).collect();
            let v_max: Vec<f64> = min2[..n].iter().zip(&max2[..n]).map(|(&a, &b)| a.max(b)).collect();
            let w_min: Vec<f64> = min3[..n].iter().zip(&max3[..n]).map(|(&a, &b)| a.min(b)).collect();
            let w_max: Vec<f64> = min3[..n].iter().zip(&max3[..n]).map(|(&a, &b)| a.max(b)).collect();

            let u = IntervalEmbedding::new(u_min, u_max);
            let v = IntervalEmbedding::new(v_min, v_max);
            let w = IntervalEmbedding::new(w_min, w_max);

            let uv = u.reachability(&v, alpha);
            let vw = v.reachability(&w, alpha);
            let uw = u.reachability(&w, alpha);

            // d(u, w) <= d(u, v) + d(v, w)
            prop_assert!(uw <= uv + vw + 1e-5);
        }
    }
}
