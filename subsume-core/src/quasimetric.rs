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
