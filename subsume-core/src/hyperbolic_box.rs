//! Hyperbolic Box Embeddings.
//!
//! Combines box embeddings (intervals) with hyperbolic geometry to model 
//! massive hierarchies with high capacity and reachability constraints.

use crate::hyperbolic::{Curvature, HyperbolicPoint, PoincareBallPoint};

/// A Box in Hyperbolic space (specifically the Poincaré Ball).
/// 
/// Defined by a center and a set of offsets (half-widths) in the tangent space, 
/// then mapped to the manifold.
pub struct HyperbolicBox {
    /// Center point of the box in the ambient coordinates (Poincaré ball model).
    pub center: Vec<f64>,
    /// Per-dimension offsets (half-widths) in the tangent-space parameterization.
    pub offsets: Vec<f64>,
}

impl HyperbolicBox {
    /// Create a new hyperbolic box.
    pub fn new(center: Vec<f64>, offsets: Vec<f64>) -> Self {
        Self { center, offsets }
    }

    /// Compute the reachability between two hyperbolic boxes.
    /// 
    /// This uses the hyperbolic distance between centers (Poincaré ball),
    /// adjusted by a simple per-dimension offset penalty.
    #[must_use]
    pub fn reachability(&self, target: &Self) -> f64 {
        // Keep this self-contained: avoid depending on external hyperbolic crates here.
        // If centers are out-of-ball, project them.
        let p1 = PoincareBallPoint::new_projected(self.center.clone(), Curvature::default());
        let p2 = PoincareBallPoint::new_projected(target.center.clone(), Curvature::default());
        let dist = p1.distance(&p2).unwrap_or(0.0);
        
        // 2026 Bleed: Subsumption in Hyperbolic Space.
        // We penalize based on how much the target box exceeds the 
        // "shadow" of the source box in the hyperbolic manifold.
        let mut penalty = 0.0;
        for i in 0..self.offsets.len() {
            let gap = (target.offsets[i] - self.offsets[i]).max(0.0);
            penalty += gap;
        }
        
        dist + penalty
    }
}
