//! Ndarray implementation of the Octagon trait.
//!
//! Octagons are axis-aligned polytopes with additional diagonal constraints
//! on adjacent dimension pairs (i, i+1). They are strictly more expressive
//! than boxes while remaining closed under intersection.
//!
//! # References
//!
//! - Charpenay & Schockaert (IJCAI 2024, arXiv:2401.16270),
//!   "Capturing Knowledge Graphs and Rules with Octagon Embeddings"

use crate::octagon::{DiagonalBounds, Octagon, OctagonError};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Diagonal bounds for a single adjacent dimension pair (i, i+1).
///
/// Constrains `x_i + x_{i+1}` and `x_i - x_{i+1}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdarrayDiagBounds {
    /// Lower bound on x_i + x_{i+1}.
    pub sum_min: f32,
    /// Upper bound on x_i + x_{i+1}.
    pub sum_max: f32,
    /// Lower bound on x_i - x_{i+1}.
    pub diff_min: f32,
    /// Upper bound on x_i - x_{i+1}.
    pub diff_max: f32,
}

impl DiagonalBounds for NdarrayDiagBounds {
    type Scalar = f32;

    fn sum_min(&self) -> f32 {
        self.sum_min
    }
    fn sum_max(&self) -> f32 {
        self.sum_max
    }
    fn diff_min(&self) -> f32 {
        self.diff_min
    }
    fn diff_max(&self) -> f32 {
        self.diff_max
    }
}

/// An octagon embedding implemented using `ndarray::Array1<f32>`.
///
/// Each octagon is defined by axis-aligned bounds (min/max per dimension)
/// plus diagonal bounds on adjacent dimension pairs.
#[derive(Debug, Clone)]
pub struct NdarrayOctagon {
    /// Minimum axis-aligned bound per dimension.
    axis_min: Array1<f32>,
    /// Maximum axis-aligned bound per dimension.
    axis_max: Array1<f32>,
    /// Diagonal bounds for adjacent pairs (i, i+1). Length = dim - 1.
    diag_bounds: Vec<NdarrayDiagBounds>,
}

impl NdarrayOctagon {
    /// Create a new `NdarrayOctagon` with validation.
    ///
    /// # Errors
    ///
    /// - [`OctagonError::DimensionMismatch`] if axis_min/axis_max differ in length.
    /// - [`OctagonError::InvalidAxisBounds`] if min > max for any dimension.
    /// - [`OctagonError::InvalidDiagonalBounds`] if sum_min > sum_max or diff_min > diff_max.
    pub fn new(
        axis_min: Array1<f32>,
        axis_max: Array1<f32>,
        diag_bounds: Vec<NdarrayDiagBounds>,
    ) -> Result<Self, OctagonError> {
        let d = axis_min.len();
        if axis_max.len() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: axis_max.len(),
            });
        }

        // Validate axis bounds.
        for i in 0..d {
            if axis_min[i] > axis_max[i] {
                return Err(OctagonError::InvalidAxisBounds {
                    dim: i,
                    min: axis_min[i] as f64,
                    max: axis_max[i] as f64,
                });
            }
        }

        // Validate diagonal bounds count.
        let expected_pairs = if d == 0 { 0 } else { d - 1 };
        if diag_bounds.len() != expected_pairs {
            return Err(OctagonError::DimensionMismatch {
                expected: expected_pairs,
                actual: diag_bounds.len(),
            });
        }

        // Validate diagonal bounds.
        for (k, db) in diag_bounds.iter().enumerate() {
            if db.sum_min > db.sum_max {
                return Err(OctagonError::InvalidDiagonalBounds {
                    dim_i: k,
                    dim_j: k + 1,
                    kind: "sum".to_string(),
                    lo: db.sum_min as f64,
                    hi: db.sum_max as f64,
                });
            }
            if db.diff_min > db.diff_max {
                return Err(OctagonError::InvalidDiagonalBounds {
                    dim_i: k,
                    dim_j: k + 1,
                    kind: "diff".to_string(),
                    lo: db.diff_min as f64,
                    hi: db.diff_max as f64,
                });
            }
        }

        Ok(Self {
            axis_min,
            axis_max,
            diag_bounds,
        })
    }

    /// Create an octagon from just axis bounds (vacuous diagonal constraints).
    ///
    /// This produces an octagon equivalent to a box: diagonal constraints are
    /// set to the loosest values implied by the axis bounds.
    pub fn from_box_bounds(
        axis_min: Array1<f32>,
        axis_max: Array1<f32>,
    ) -> Result<Self, OctagonError> {
        let d = axis_min.len();
        if axis_max.len() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: axis_max.len(),
            });
        }
        for i in 0..d {
            if axis_min[i] > axis_max[i] {
                return Err(OctagonError::InvalidAxisBounds {
                    dim: i,
                    min: axis_min[i] as f64,
                    max: axis_max[i] as f64,
                });
            }
        }

        let mut diag_bounds = Vec::with_capacity(if d == 0 { 0 } else { d - 1 });
        for k in 0..d.saturating_sub(1) {
            diag_bounds.push(NdarrayDiagBounds {
                sum_min: axis_min[k] + axis_min[k + 1],
                sum_max: axis_max[k] + axis_max[k + 1],
                diff_min: axis_min[k] - axis_max[k + 1],
                diff_max: axis_max[k] - axis_min[k + 1],
            });
        }

        Ok(Self {
            axis_min,
            axis_max,
            diag_bounds,
        })
    }
}

/// Numerically stable sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl Octagon for NdarrayOctagon {
    type Scalar = f32;
    type Vector = Array1<f32>;
    type Diag = NdarrayDiagBounds;

    fn axis_min(&self) -> &Array1<f32> {
        &self.axis_min
    }

    fn axis_max(&self) -> &Array1<f32> {
        &self.axis_max
    }

    fn dim(&self) -> usize {
        self.axis_min.len()
    }

    fn diagonal_bounds(&self, pair_index: usize) -> Option<&NdarrayDiagBounds> {
        self.diag_bounds.get(pair_index)
    }

    fn contains(&self, point: &[f32]) -> Result<bool, OctagonError> {
        let d = self.dim();
        if point.len() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: point.len(),
            });
        }

        // Check axis-aligned bounds.
        for (i, &val) in point.iter().enumerate().take(d) {
            if val < self.axis_min[i] || val > self.axis_max[i] {
                return Ok(false);
            }
        }

        // Check diagonal bounds.
        for (k, db) in self.diag_bounds.iter().enumerate() {
            let sum = point[k] + point[k + 1];
            let diff = point[k] - point[k + 1];
            if sum < db.sum_min || sum > db.sum_max || diff < db.diff_min || diff > db.diff_max {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn intersection(&self, other: &Self) -> Result<Self, OctagonError> {
        let d = self.dim();
        if other.dim() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: other.dim(),
            });
        }

        // Componentwise tighten axis bounds.
        let new_min = Array1::from_vec(
            (0..d)
                .map(|i| self.axis_min[i].max(other.axis_min[i]))
                .collect(),
        );
        let new_max = Array1::from_vec(
            (0..d)
                .map(|i| self.axis_max[i].min(other.axis_max[i]))
                .collect(),
        );

        // Check for empty intersection in axis bounds.
        for i in 0..d {
            if new_min[i] > new_max[i] {
                return Err(OctagonError::Empty);
            }
        }

        // Componentwise tighten diagonal bounds.
        let mut new_diag = Vec::with_capacity(self.diag_bounds.len());
        for (k, (a, b)) in self
            .diag_bounds
            .iter()
            .zip(other.diag_bounds.iter())
            .enumerate()
        {
            let sum_min = a.sum_min.max(b.sum_min);
            let sum_max = a.sum_max.min(b.sum_max);
            let diff_min = a.diff_min.max(b.diff_min);
            let diff_max = a.diff_max.min(b.diff_max);

            if sum_min > sum_max {
                return Err(OctagonError::InvalidDiagonalBounds {
                    dim_i: k,
                    dim_j: k + 1,
                    kind: "sum".to_string(),
                    lo: sum_min as f64,
                    hi: sum_max as f64,
                });
            }
            if diff_min > diff_max {
                return Err(OctagonError::InvalidDiagonalBounds {
                    dim_i: k,
                    dim_j: k + 1,
                    kind: "diff".to_string(),
                    lo: diff_min as f64,
                    hi: diff_max as f64,
                });
            }

            new_diag.push(NdarrayDiagBounds {
                sum_min,
                sum_max,
                diff_min,
                diff_max,
            });
        }

        Ok(Self {
            axis_min: new_min,
            axis_max: new_max,
            diag_bounds: new_diag,
        })
    }

    fn volume(&self) -> Result<f32, OctagonError> {
        let d = self.dim();
        if d == 0 {
            return Ok(0.0);
        }

        // Compute bounding box volume.
        let mut box_vol = 1.0f32;
        for i in 0..d {
            let side = self.axis_max[i] - self.axis_min[i];
            if side <= 0.0 {
                return Ok(0.0);
            }
            box_vol *= side;
        }

        if d == 1 {
            // 1D: octagon = interval, no diagonal constraints.
            return Ok(box_vol);
        }

        if d == 2 {
            // 2D: exact octagon area via vertex enumeration.
            return self.volume_2d();
        }

        // Higher dimensions: Monte Carlo approximation.
        self.volume_monte_carlo(box_vol)
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: f32,
    ) -> Result<f32, OctagonError> {
        let d = self.dim();
        if other.dim() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: other.dim(),
            });
        }

        // Product of per-constraint sigmoid scores.
        // For axis bounds: self should contain other => self.min <= other.min, other.max <= self.max.
        let mut product = 1.0f32;

        for i in 0..d {
            // other.min >= self.min (positive margin = good)
            let margin_lo = other.axis_min[i] - self.axis_min[i];
            product *= sigmoid(margin_lo / temperature);
            // self.max >= other.max (positive margin = good)
            let margin_hi = self.axis_max[i] - other.axis_max[i];
            product *= sigmoid(margin_hi / temperature);
        }

        // For diagonal bounds: self's constraints are looser than other's.
        for (a, b) in self.diag_bounds.iter().zip(other.diag_bounds.iter()) {
            // self.sum_min <= other.sum_min
            let m1 = b.sum_min - a.sum_min;
            product *= sigmoid(m1 / temperature);
            // self.sum_max >= other.sum_max
            let m2 = a.sum_max - b.sum_max;
            product *= sigmoid(m2 / temperature);
            // self.diff_min <= other.diff_min
            let m3 = b.diff_min - a.diff_min;
            product *= sigmoid(m3 / temperature);
            // self.diff_max >= other.diff_max
            let m4 = a.diff_max - b.diff_max;
            product *= sigmoid(m4 / temperature);
        }

        Ok(product)
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: f32,
    ) -> Result<f32, OctagonError> {
        let d = self.dim();
        if other.dim() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: other.dim(),
            });
        }

        // Product of per-constraint overlap sigmoids.
        let mut product = 1.0f32;

        for i in 0..d {
            // Overlap in dimension i: self.max >= other.min AND other.max >= self.min.
            let margin_1 = self.axis_max[i] - other.axis_min[i];
            let margin_2 = other.axis_max[i] - self.axis_min[i];
            product *= sigmoid(margin_1 / temperature);
            product *= sigmoid(margin_2 / temperature);
        }

        for (a, b) in self.diag_bounds.iter().zip(other.diag_bounds.iter()) {
            // Overlap in sum constraint: a.sum_max >= b.sum_min AND b.sum_max >= a.sum_min.
            product *= sigmoid((a.sum_max - b.sum_min) / temperature);
            product *= sigmoid((b.sum_max - a.sum_min) / temperature);
            // Overlap in diff constraint.
            product *= sigmoid((a.diff_max - b.diff_min) / temperature);
            product *= sigmoid((b.diff_max - a.diff_min) / temperature);
        }

        Ok(product)
    }

    fn to_bounding_box_bounds(&self) -> (Array1<f32>, Array1<f32>) {
        (self.axis_min.clone(), self.axis_max.clone())
    }
}

impl NdarrayOctagon {
    /// Exact 2D octagon area via the Sutherland-Hodgman approach:
    /// intersect the box with each half-plane from diagonal constraints.
    fn volume_2d(&self) -> Result<f32, OctagonError> {
        // Start with the bounding box as a polygon.
        let x_min = self.axis_min[0];
        let x_max = self.axis_max[0];
        let y_min = self.axis_min[1];
        let y_max = self.axis_max[1];

        let mut polygon: Vec<[f32; 2]> = vec![
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ];

        if let Some(db) = self.diag_bounds.first() {
            // Clip by: x + y >= sum_min  =>  x + y - sum_min >= 0
            polygon = clip_polygon(&polygon, |p| p[0] + p[1] - db.sum_min);
            // Clip by: x + y <= sum_max  =>  sum_max - x - y >= 0
            polygon = clip_polygon(&polygon, |p| db.sum_max - p[0] - p[1]);
            // Clip by: x - y >= diff_min  =>  x - y - diff_min >= 0
            polygon = clip_polygon(&polygon, |p| p[0] - p[1] - db.diff_min);
            // Clip by: x - y <= diff_max  =>  diff_max - x + y >= 0
            polygon = clip_polygon(&polygon, |p| db.diff_max - p[0] + p[1]);
        }

        if polygon.len() < 3 {
            return Ok(0.0);
        }

        // Shoelace formula for polygon area.
        let area = polygon_area(&polygon);
        Ok(area)
    }

    /// Monte Carlo volume estimate for d > 2.
    fn volume_monte_carlo(&self, box_vol: f32) -> Result<f32, OctagonError> {
        let d = self.dim();
        // Use a deterministic quasi-random sequence for reproducibility.
        let n_samples = 10_000;
        let mut inside = 0u32;

        // Simple LCG for reproducible sampling within the bounding box.
        let mut rng_state = 0x12345678u64;
        for _ in 0..n_samples {
            let mut point = vec![0.0f32; d];
            let mut all_inside = true;

            for (i, p) in point.iter_mut().enumerate().take(d) {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                *p = self.axis_min[i] + u * (self.axis_max[i] - self.axis_min[i]);
            }

            // Check diagonal constraints.
            for (k, db) in self.diag_bounds.iter().enumerate() {
                let sum = point[k] + point[k + 1];
                let diff = point[k] - point[k + 1];
                if sum < db.sum_min || sum > db.sum_max || diff < db.diff_min || diff > db.diff_max
                {
                    all_inside = false;
                    break;
                }
            }

            if all_inside {
                inside += 1;
            }
        }

        Ok(box_vol * (inside as f32 / n_samples as f32))
    }
}

/// Clip a convex polygon by a half-plane defined by `f(p) >= 0`.
fn clip_polygon(polygon: &[[f32; 2]], f: impl Fn(&[f32; 2]) -> f32) -> Vec<[f32; 2]> {
    if polygon.is_empty() {
        return vec![];
    }

    let mut output = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let current = &polygon[i];
        let next = &polygon[(i + 1) % n];
        let f_current = f(current);
        let f_next = f(next);

        if f_current >= 0.0 {
            output.push(*current);
            if f_next < 0.0 {
                // Edge exits: compute intersection.
                let t = f_current / (f_current - f_next);
                output.push([
                    current[0] + t * (next[0] - current[0]),
                    current[1] + t * (next[1] - current[1]),
                ]);
            }
        } else if f_next >= 0.0 {
            // Edge enters: compute intersection.
            let t = f_current / (f_current - f_next);
            output.push([
                current[0] + t * (next[0] - current[0]),
                current[1] + t * (next[1] - current[1]),
            ]);
        }
    }

    output
}

/// Shoelace formula for the area of a simple polygon.
fn polygon_area(vertices: &[[f32; 2]]) -> f32 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    area.abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    // ---- Construction ----

    #[test]
    fn new_valid_2d() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.5,
                sum_max: 3.5,
                diff_min: -1.5,
                diff_max: 1.5,
            }],
        );
        assert!(oct.is_ok());
    }

    #[test]
    fn new_rejects_axis_min_gt_max() {
        let result = NdarrayOctagon::new(
            array![2.0, 0.0],
            array![1.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 4.0,
                diff_min: -2.0,
                diff_max: 2.0,
            }],
        );
        assert!(matches!(result, Err(OctagonError::InvalidAxisBounds { .. })));
    }

    #[test]
    fn new_rejects_dim_mismatch() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0],
            vec![],
        );
        assert!(matches!(result, Err(OctagonError::DimensionMismatch { .. })));
    }

    #[test]
    fn new_rejects_wrong_diag_count() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![], // Should have 1 diagonal bounds entry.
        );
        assert!(matches!(result, Err(OctagonError::DimensionMismatch { .. })));
    }

    #[test]
    fn new_rejects_invalid_diag_bounds() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![NdarrayDiagBounds {
                sum_min: 5.0, // > sum_max
                sum_max: 1.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        );
        assert!(matches!(
            result,
            Err(OctagonError::InvalidDiagonalBounds { .. })
        ));
    }

    #[test]
    fn from_box_bounds_produces_vacuous_diag() {
        let oct = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        assert_eq!(oct.diag_bounds.len(), 1);
        // Vacuous: sum_min = 0+0=0, sum_max = 1+1=2
        assert!((oct.diag_bounds[0].sum_min - 0.0).abs() < 1e-6);
        assert!((oct.diag_bounds[0].sum_max - 2.0).abs() < 1e-6);
        // diff_min = 0-1=-1, diff_max = 1-0=1
        assert!((oct.diag_bounds[0].diff_min - (-1.0)).abs() < 1e-6);
        assert!((oct.diag_bounds[0].diff_max - 1.0).abs() < 1e-6);
    }

    // ---- Contains ----

    #[test]
    fn contains_center_point() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 4.0,
                diff_min: -2.0,
                diff_max: 2.0,
            }],
        )
        .unwrap();
        assert!(oct.contains(&[1.0, 1.0]).unwrap());
    }

    #[test]
    fn contains_rejects_outside_axis() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 2.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        )
        .unwrap();
        assert!(!oct.contains(&[1.5, 0.5]).unwrap());
    }

    #[test]
    fn contains_rejects_outside_diagonal() {
        // Point is inside the box but violates diagonal constraint.
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 1.0,
                sum_max: 3.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        )
        .unwrap();
        // (0.1, 0.1): sum=0.2 < sum_min=1.0
        assert!(!oct.contains(&[0.1, 0.1]).unwrap());
    }

    #[test]
    fn contains_dim_mismatch() {
        let oct = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        assert!(matches!(
            oct.contains(&[0.5]),
            Err(OctagonError::DimensionMismatch { .. })
        ));
    }

    // ---- Intersection ----

    #[test]
    fn intersection_identical_octagons() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.5,
                sum_max: 3.5,
                diff_min: -1.5,
                diff_max: 1.5,
            }],
        )
        .unwrap();
        let inter = oct.intersection(&oct).unwrap();
        // Should be the same octagon.
        for i in 0..2 {
            assert!((inter.axis_min[i] - oct.axis_min[i]).abs() < 1e-6);
            assert!((inter.axis_max[i] - oct.axis_max[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn intersection_tightens_bounds() {
        let a = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![4.0, 4.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 8.0,
                diff_min: -4.0,
                diff_max: 4.0,
            }],
        )
        .unwrap();
        let b = NdarrayOctagon::new(
            array![1.0, 1.0],
            array![3.0, 3.0],
            vec![NdarrayDiagBounds {
                sum_min: 2.5,
                sum_max: 5.5,
                diff_min: -1.5,
                diff_max: 1.5,
            }],
        )
        .unwrap();
        let inter = a.intersection(&b).unwrap();
        assert!((inter.axis_min[0] - 1.0).abs() < 1e-6);
        assert!((inter.axis_max[0] - 3.0).abs() < 1e-6);
        assert!((inter.diag_bounds[0].sum_min - 2.5).abs() < 1e-6);
    }

    #[test]
    fn intersection_empty_returns_error() {
        let a = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        let b = NdarrayOctagon::from_box_bounds(array![2.0, 2.0], array![3.0, 3.0]).unwrap();
        assert!(a.intersection(&b).is_err());
    }

    #[test]
    fn intersection_dim_mismatch() {
        let a = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        let b = NdarrayOctagon::from_box_bounds(array![0.0], array![1.0]).unwrap();
        assert!(matches!(
            a.intersection(&b),
            Err(OctagonError::DimensionMismatch { .. })
        ));
    }

    // ---- Volume ----

    #[test]
    fn volume_box_equivalent() {
        // Octagon with vacuous diagonal = box volume.
        let oct = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![2.0, 3.0]).unwrap();
        let vol = oct.volume().unwrap();
        assert!((vol - 6.0).abs() < 1e-4, "box-equivalent volume: expected 6, got {vol}");
    }

    #[test]
    fn volume_2d_octagon_smaller_than_box() {
        // Tight diagonal constraints should produce smaller volume than the box.
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 1.0,
                sum_max: 3.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        )
        .unwrap();
        let vol = oct.volume().unwrap();
        assert!(vol > 0.0, "non-degenerate octagon should have positive volume");
        assert!(vol < 4.0, "octagon should be smaller than bounding box (area 4), got {vol}");
    }

    #[test]
    fn volume_1d() {
        let oct = NdarrayOctagon::new(array![1.0], array![5.0], vec![]).unwrap();
        let vol = oct.volume().unwrap();
        assert!((vol - 4.0).abs() < 1e-6, "1D volume: expected 4.0, got {vol}");
    }

    #[test]
    fn volume_zero_width_dimension() {
        let oct = NdarrayOctagon::from_box_bounds(array![1.0, 0.0], array![1.0, 2.0]).unwrap();
        let vol = oct.volume().unwrap();
        assert_eq!(vol, 0.0, "zero-width dimension should give zero volume");
    }

    // ---- Containment probability ----

    #[test]
    fn containment_prob_identical_is_boundary() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 2.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.5,
                sum_max: 3.5,
                diff_min: -1.5,
                diff_max: 1.5,
            }],
        )
        .unwrap();
        let p = oct.containment_prob(&oct, 0.1).unwrap();
        // All margins are 0 => sigmoid(0) = 0.5, product of 0.5^(2*2 + 4*1) = 0.5^8
        let expected = 0.5f32.powi(8);
        assert!(
            (p - expected).abs() < 0.01,
            "self-containment should be 0.5^8 = {expected}, got {p}"
        );
    }

    #[test]
    fn containment_prob_wider_contains_narrower() {
        let wide = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![4.0, 4.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 8.0,
                diff_min: -4.0,
                diff_max: 4.0,
            }],
        )
        .unwrap();
        let narrow = NdarrayOctagon::new(
            array![1.0, 1.0],
            array![3.0, 3.0],
            vec![NdarrayDiagBounds {
                sum_min: 2.5,
                sum_max: 5.5,
                diff_min: -1.5,
                diff_max: 1.5,
            }],
        )
        .unwrap();
        let p = wide.containment_prob(&narrow, 0.1).unwrap();
        assert!(
            p > 0.9,
            "wide octagon should contain narrow with high prob, got {p}"
        );
    }

    // ---- Overlap probability ----

    #[test]
    fn overlap_prob_identical_is_high() {
        let oct = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![2.0, 2.0]).unwrap();
        let p = oct.overlap_prob(&oct, 0.1).unwrap();
        assert!(p > 0.99, "identical octagons should have very high overlap, got {p}");
    }

    #[test]
    fn overlap_prob_disjoint_is_low() {
        let a = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        let b = NdarrayOctagon::from_box_bounds(array![5.0, 5.0], array![6.0, 6.0]).unwrap();
        let p = a.overlap_prob(&b, 0.1).unwrap();
        assert!(p < 0.01, "disjoint octagons should have very low overlap, got {p}");
    }

    // ---- Bounding box ----

    #[test]
    fn to_bounding_box_drops_diag() {
        let oct = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![2.0, 3.0],
            vec![NdarrayDiagBounds {
                sum_min: 1.0,
                sum_max: 4.0,
                diff_min: -2.0,
                diff_max: 2.0,
            }],
        )
        .unwrap();
        let (bb_min, bb_max) = oct.to_bounding_box_bounds();
        assert_eq!(bb_min, array![0.0, 0.0]);
        assert_eq!(bb_max, array![2.0, 3.0]);
    }

    // ---- Property tests ----

    fn arb_diag_bounds() -> impl Strategy<Value = NdarrayDiagBounds> {
        (-10.0f32..10.0, 0.1f32..5.0, -10.0f32..10.0, 0.1f32..5.0).prop_map(
            |(sum_base, sum_half, diff_base, diff_half)| NdarrayDiagBounds {
                sum_min: sum_base - sum_half,
                sum_max: sum_base + sum_half,
                diff_min: diff_base - diff_half,
                diff_max: diff_base + diff_half,
            },
        )
    }

    fn arb_octagon_2d() -> impl Strategy<Value = NdarrayOctagon> {
        (
            -5.0f32..5.0,
            0.1f32..5.0,
            -5.0f32..5.0,
            0.1f32..5.0,
            arb_diag_bounds(),
        )
            .prop_map(|(x_base, x_half, y_base, y_half, db)| {
                NdarrayOctagon::new(
                    array![x_base - x_half, y_base - y_half],
                    array![x_base + x_half, y_base + y_half],
                    vec![db],
                )
                .unwrap()
            })
    }

    proptest! {
        #[test]
        fn prop_self_containment(oct in arb_octagon_2d()) {
            // containment_prob(self, self) should be consistent (all margins=0 => sigmoid^n).
            let p = oct.containment_prob(&oct, 0.1).unwrap();
            // 2D: 2*2 axis constraints + 4*1 diag constraints = 8 => 0.5^8
            let expected = 0.5f32.powi(8);
            prop_assert!(
                (p - expected).abs() < 0.02,
                "self-containment should be ~{expected}, got {p}"
            );
        }

        #[test]
        fn prop_intersection_idempotent(oct in arb_octagon_2d()) {
            let inter = oct.intersection(&oct).unwrap();
            for i in 0..2 {
                prop_assert!((inter.axis_min[i] - oct.axis_min[i]).abs() < 1e-5);
                prop_assert!((inter.axis_max[i] - oct.axis_max[i]).abs() < 1e-5);
            }
            prop_assert!((inter.diag_bounds[0].sum_min - oct.diag_bounds[0].sum_min).abs() < 1e-5);
            prop_assert!((inter.diag_bounds[0].sum_max - oct.diag_bounds[0].sum_max).abs() < 1e-5);
        }

        #[test]
        fn prop_volume_nonnegative(oct in arb_octagon_2d()) {
            let vol = oct.volume().unwrap();
            prop_assert!(vol >= 0.0, "volume must be non-negative, got {vol}");
        }

        #[test]
        fn prop_containment_implies_overlap(
            (a, b) in (arb_octagon_2d(), arb_octagon_2d()),
        ) {
            let cont = a.containment_prob(&b, 0.5).unwrap();
            let overlap = a.overlap_prob(&b, 0.5).unwrap();
            // If containment is high, overlap should be at least as high.
            if cont > 0.9 {
                prop_assert!(overlap > 0.5,
                    "high containment ({cont}) should imply overlap ({overlap})");
            }
        }

        #[test]
        fn prop_octagon_degrades_to_box(
            x_base in -5.0f32..5.0,
            x_half in 0.1f32..5.0,
            y_base in -5.0f32..5.0,
            y_half in 0.1f32..5.0,
        ) {
            let oct = NdarrayOctagon::from_box_bounds(
                array![x_base - x_half, y_base - y_half],
                array![x_base + x_half, y_base + y_half],
            ).unwrap();
            let vol = oct.volume().unwrap();
            let box_vol = (2.0 * x_half) * (2.0 * y_half);
            prop_assert!(
                (vol - box_vol).abs() < 0.01 * box_vol.max(1e-6),
                "box-equivalent octagon volume {vol} should match box volume {box_vol}"
            );
        }
    }
}
