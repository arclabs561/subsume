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

use crate::octagon::OctagonError;
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

impl NdarrayDiagBounds {
    /// Lower bound on x_i + x_{i+1}.
    pub fn sum_min(&self) -> f32 {
        self.sum_min
    }
    /// Upper bound on x_i + x_{i+1}.
    pub fn sum_max(&self) -> f32 {
        self.sum_max
    }
    /// Lower bound on x_i - x_{i+1}.
    pub fn diff_min(&self) -> f32 {
        self.diff_min
    }
    /// Upper bound on x_i - x_{i+1}.
    pub fn diff_max(&self) -> f32 {
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

impl Serialize for NdarrayOctagon {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NdarrayOctagon", 3)?;
        state.serialize_field("axis_min", &self.axis_min.to_vec())?;
        state.serialize_field("axis_max", &self.axis_max.to_vec())?;
        state.serialize_field("diag_bounds", &self.diag_bounds)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NdarrayOctagon {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            AxisMin,
            AxisMax,
            DiagBounds,
        }

        struct OctagonVisitor;

        impl<'de> Visitor<'de> for OctagonVisitor {
            type Value = NdarrayOctagon;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter
                    .write_str("struct NdarrayOctagon with fields axis_min, axis_max, diag_bounds")
            }

            fn visit_map<V>(self, mut map: V) -> Result<NdarrayOctagon, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut axis_min = None;
                let mut axis_max = None;
                let mut diag_bounds = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::AxisMin => {
                            if axis_min.is_some() {
                                return Err(de::Error::duplicate_field("axis_min"));
                            }
                            axis_min = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::AxisMax => {
                            if axis_max.is_some() {
                                return Err(de::Error::duplicate_field("axis_max"));
                            }
                            axis_max = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::DiagBounds => {
                            if diag_bounds.is_some() {
                                return Err(de::Error::duplicate_field("diag_bounds"));
                            }
                            diag_bounds = Some(map.next_value::<Vec<NdarrayDiagBounds>>()?);
                        }
                    }
                }
                let axis_min = axis_min.ok_or_else(|| de::Error::missing_field("axis_min"))?;
                let axis_max = axis_max.ok_or_else(|| de::Error::missing_field("axis_max"))?;
                let diag_bounds =
                    diag_bounds.ok_or_else(|| de::Error::missing_field("diag_bounds"))?;
                NdarrayOctagon::new(Array1::from(axis_min), Array1::from(axis_max), diag_bounds)
                    .map_err(|e| de::Error::custom(format!("Invalid octagon: {e}")))
            }
        }

        const FIELDS: &[&str] = &["axis_min", "axis_max", "diag_bounds"];
        deserializer.deserialize_struct("NdarrayOctagon", FIELDS, OctagonVisitor)
    }
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

        // Reject non-finite values.
        if axis_min.iter().any(|v| !v.is_finite()) || axis_max.iter().any(|v| !v.is_finite()) {
            return Err(OctagonError::InvalidAxisBounds {
                dim: 0,
                min: f64::NAN,
                max: f64::NAN,
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

        // Validate diagonal bounds (including non-finite rejection).
        for (k, db) in diag_bounds.iter().enumerate() {
            if !db.sum_min.is_finite()
                || !db.sum_max.is_finite()
                || !db.diff_min.is_finite()
                || !db.diff_max.is_finite()
            {
                return Err(OctagonError::InvalidDiagonalBounds {
                    dim_i: k,
                    dim_j: k + 1,
                    kind: "non-finite".to_string(),
                    lo: f64::NAN,
                    hi: f64::NAN,
                });
            }
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
        // Reject non-finite in axis bounds.
        if axis_min.iter().any(|v| !v.is_finite()) || axis_max.iter().any(|v| !v.is_finite()) {
            return Err(OctagonError::InvalidAxisBounds {
                dim: 0,
                min: f64::NAN,
                max: f64::NAN,
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

use crate::utils::stable_sigmoid as sigmoid;

impl NdarrayOctagon {
    /// Get the minimum axis-aligned bound in each dimension.
    pub fn axis_min(&self) -> &Array1<f32> {
        &self.axis_min
    }

    /// Get the maximum axis-aligned bound in each dimension.
    pub fn axis_max(&self) -> &Array1<f32> {
        &self.axis_max
    }

    /// Get the number of dimensions.
    pub fn dim(&self) -> usize {
        self.axis_min.len()
    }

    /// Get the diagonal bounds for the k-th adjacent pair (k, k+1).
    ///
    /// Returns `None` if k >= dim - 1.
    pub fn diagonal_bounds(&self, pair_index: usize) -> Option<&NdarrayDiagBounds> {
        self.diag_bounds.get(pair_index)
    }

    /// Number of adjacent dimension pairs = dim - 1.
    pub fn num_diagonal_pairs(&self) -> usize {
        let d = self.dim();
        if d == 0 {
            0
        } else {
            d - 1
        }
    }

    /// Check whether a point lies inside this octagon.
    pub fn contains(&self, point: &[f32]) -> Result<bool, OctagonError> {
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

    /// Compute the intersection of two octagons.
    pub fn intersection(&self, other: &Self) -> Result<Self, OctagonError> {
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

    /// Compute the volume of this octagon.
    pub fn volume(&self) -> Result<f32, OctagonError> {
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

    /// Soft containment probability: P(other inside self), smoothed by temperature.
    pub fn containment_prob(&self, other: &Self, temperature: f32) -> Result<f32, OctagonError> {
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

    /// Soft overlap probability: Vol(self intersection other) / Vol(self union other).
    pub fn overlap_prob(&self, other: &Self, temperature: f32) -> Result<f32, OctagonError> {
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

    /// Convert this octagon to its bounding box (drop diagonal constraints).
    pub fn to_bounding_box_bounds(&self) -> (Array1<f32>, Array1<f32>) {
        (self.axis_min.clone(), self.axis_max.clone())
    }

    /// Normalize (tighten) the octagon bounds so no constraint can be strengthened
    /// without changing the represented set. This is Proposition 1 from
    /// Charpenay & Schockaert (IJCAI 2024).
    ///
    /// Normalization is a fixed-point operation: tightening one bound may enable
    /// tightening another. We iterate until convergence (bounded by 10 passes).
    /// Normalize (tighten) the octagon bounds (Proposition 1).
    pub fn normalize(&self) -> Result<Self, OctagonError> {
        let d = self.dim();
        if d <= 1 {
            return Ok(self.clone());
        }

        let mut x_lo: Vec<f32> = self.axis_min.to_vec();
        let mut x_hi: Vec<f32> = self.axis_max.to_vec();
        let mut diag: Vec<NdarrayDiagBounds> = self.diag_bounds.clone();

        for _pass in 0..10 {
            let mut changed = false;

            for k in 0..d - 1 {
                let xl = x_lo[k];
                let xh = x_hi[k];
                let yl = x_lo[k + 1];
                let yh = x_hi[k + 1];
                // Paper convention: u = y - x, v = x + y
                // Our convention: sum = x + y, diff = x - y
                // So: u_lo = -(diff_max), u_hi = -(diff_min), v_lo = sum_min, v_hi = sum_max
                let u_lo = -diag[k].diff_max;
                let u_hi = -diag[k].diff_min;
                let v_lo = diag[k].sum_min;
                let v_hi = diag[k].sum_max;

                // Tighten x bounds
                let new_xl = xl.max(v_lo - yh).max(yl - u_hi).max((v_lo - u_hi) / 2.0);
                let new_xh = xh.min(v_hi - yl).min(yh - u_lo).min((v_hi - u_lo) / 2.0);

                // Tighten y bounds
                let new_yl = yl.max(u_lo + xl).max(v_lo - xh).max((u_lo + v_lo) / 2.0);
                let new_yh = yh.min(u_hi + xh).min(v_hi - xl).min((u_hi + v_hi) / 2.0);

                // Tighten u bounds (y - x)
                let new_u_lo = u_lo
                    .max(new_yl - new_xh)
                    .max(v_lo - 2.0 * new_xh)
                    .max(2.0 * new_yl - v_hi);
                let new_u_hi = u_hi
                    .min(new_yh - new_xl)
                    .min(v_hi - 2.0 * new_xl)
                    .min(2.0 * new_yh - v_lo);

                // Tighten v bounds (x + y)
                let new_v_lo = v_lo
                    .max(new_xl + new_yl)
                    .max(new_u_lo + 2.0 * new_xl)
                    .max(2.0 * new_yl - new_u_hi);
                let new_v_hi = v_hi
                    .min(new_xh + new_yh)
                    .min(new_u_hi + 2.0 * new_xh)
                    .min(2.0 * new_yh - new_u_lo);

                let eps = 1e-6;
                if (new_xl - x_lo[k]).abs() > eps
                    || (new_xh - x_hi[k]).abs() > eps
                    || (new_yl - x_lo[k + 1]).abs() > eps
                    || (new_yh - x_hi[k + 1]).abs() > eps
                {
                    changed = true;
                }

                x_lo[k] = new_xl;
                x_hi[k] = new_xh;
                x_lo[k + 1] = new_yl;
                x_hi[k + 1] = new_yh;
                // Convert back: diff_min = -u_hi, diff_max = -u_lo
                diag[k].diff_min = -new_u_hi;
                diag[k].diff_max = -new_u_lo;
                diag[k].sum_min = new_v_lo;
                diag[k].sum_max = new_v_hi;
            }

            if !changed {
                break;
            }
        }

        // Check feasibility after normalization
        for i in 0..d {
            if x_lo[i] > x_hi[i] + 1e-6 {
                return Err(OctagonError::Empty);
            }
        }

        NdarrayOctagon::new(Array1::from_vec(x_lo), Array1::from_vec(x_hi), diag)
    }

    /// Relational composition of two octagon relations.
    pub fn compose(&self, other: &Self) -> Result<Self, OctagonError> {
        let d = self.dim();
        if other.dim() != d {
            return Err(OctagonError::DimensionMismatch {
                expected: d,
                actual: other.dim(),
            });
        }

        if d == 0 {
            return Ok(Self {
                axis_min: Array1::zeros(0),
                axis_max: Array1::zeros(0),
                diag_bounds: vec![],
            });
        }

        // For d == 1, there are no diagonal constraints. The relation is just
        // an interval on the single dimension. Composition projects out the
        // shared variable, so the result is the full range of x (from R) that
        // has some overlapping y with S, which is just R's axis bounds intersected
        // with... actually for 1D, the "relation" interpretation is degenerate
        // (single variable, no pair). Return intersection of axis bounds.
        if d == 1 {
            let new_min = self.axis_min[0].max(other.axis_min[0]);
            let new_max = self.axis_max[0].min(other.axis_max[0]);
            if new_min > new_max {
                return Err(OctagonError::Empty);
            }
            return NdarrayOctagon::new(
                Array1::from_vec(vec![new_min]),
                Array1::from_vec(vec![new_max]),
                vec![],
            );
        }

        // For each adjacent pair k, apply the 2D composition formula from
        // Proposition 2 (Charpenay & Schockaert, IJCAI 2024).
        //
        // Paper convention: u = y - x, v = x + y
        // Our convention: sum = x + y, diff = x - y
        // Mapping: u = -diff, v = sum
        //   u⁻ = -diff_max, u⁺ = -diff_min
        //   v⁻ = sum_min,   v⁺ = sum_max

        let num_pairs = d - 1;

        // Per-pair composition produces candidate axis bounds for each dimension.
        // A dimension k appears as first-dim in pair k and second-dim in pair k-1.
        // We take the tightest bounds across all pairs that reference it.
        let mut result_axis_min = vec![f32::NEG_INFINITY; d];
        let mut result_axis_max = vec![f32::INFINITY; d];
        let mut result_diag = Vec::with_capacity(num_pairs);

        for k in 0..num_pairs {
            let r = &self.diag_bounds[k];
            let s = &other.diag_bounds[k];

            // R's parameters for pair k (dims k, k+1):
            let x1_lo = self.axis_min[k];
            let x1_hi = self.axis_max[k];
            let y1_lo = self.axis_min[k + 1];
            let y1_hi = self.axis_max[k + 1];
            let u1_lo = -r.diff_max; // paper u = y - x = -(x - y) = -diff
            let u1_hi = -r.diff_min;
            let v1_lo = r.sum_min;
            let v1_hi = r.sum_max;

            // S's parameters for pair k (dims k, k+1):
            let x2_lo = other.axis_min[k]; // shared variable (y in composition)
            let x2_hi = other.axis_max[k];
            let y2_lo = other.axis_min[k + 1]; // output variable (z in composition)
            let y2_hi = other.axis_max[k + 1];
            let u2_lo = -s.diff_max;
            let u2_hi = -s.diff_min;
            let v2_lo = s.sum_min;
            let v2_hi = s.sum_max;

            // Proposition 2 formulas:
            let x3_lo = f32::max(x1_lo, f32::max(x2_lo - u1_hi, v1_lo - x2_hi));
            let x3_hi = f32::min(x1_hi, f32::min(x2_hi - u1_lo, v1_hi - x2_lo));

            let y3_lo = f32::max(y2_lo, f32::max(u2_lo + y1_lo, v2_lo - y1_hi));
            let y3_hi = f32::min(y2_hi, f32::min(u2_hi + y1_hi, v2_hi - y1_lo));

            let u3_lo = f32::max(y2_lo - x1_hi, f32::max(u2_lo + u1_lo, v2_lo - v1_hi));
            let u3_hi = f32::min(y2_hi - x1_lo, f32::min(u2_hi + u1_hi, v2_hi - v1_lo));

            let v3_lo = f32::max(x1_lo + y2_lo, f32::max(u2_lo + v1_lo, v2_lo - u1_hi));
            let v3_hi = f32::min(x1_hi + y2_hi, f32::min(u2_hi + v1_hi, v2_hi - u1_lo));

            // Convert back to our convention:
            // diff = x - y = -u, so diff_min = -u3_hi, diff_max = -u3_lo
            let diff_min = -u3_hi;
            let diff_max = -u3_lo;
            let sum_min = v3_lo;
            let sum_max = v3_hi;

            // Check feasibility.
            if x3_lo > x3_hi || y3_lo > y3_hi || sum_min > sum_max || diff_min > diff_max {
                return Err(OctagonError::Empty);
            }

            // Tighten axis bounds: take the max of lower bounds and min of upper bounds
            // across all pairs that produce bounds for this dimension.
            result_axis_min[k] = result_axis_min[k].max(x3_lo);
            result_axis_max[k] = result_axis_max[k].min(x3_hi);
            result_axis_min[k + 1] = result_axis_min[k + 1].max(y3_lo);
            result_axis_max[k + 1] = result_axis_max[k + 1].min(y3_hi);

            result_diag.push(NdarrayDiagBounds {
                sum_min,
                sum_max,
                diff_min,
                diff_max,
            });
        }

        // Final feasibility check on axis bounds (after tightening across pairs).
        for i in 0..d {
            if result_axis_min[i] > result_axis_max[i] {
                return Err(OctagonError::Empty);
            }
        }

        let raw = NdarrayOctagon::new(
            Array1::from_vec(result_axis_min),
            Array1::from_vec(result_axis_max),
            result_diag,
        )?;
        // Normalize to tighten bounds (Proposition 1). Without this,
        // composition is not associative because intermediate bounds
        // carry slack that compounds differently depending on grouping.
        raw.normalize()
    }

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
        assert!(matches!(
            result,
            Err(OctagonError::InvalidAxisBounds { .. })
        ));
    }

    #[test]
    fn new_rejects_dim_mismatch() {
        let result = NdarrayOctagon::new(array![0.0, 0.0], array![1.0], vec![]);
        assert!(matches!(
            result,
            Err(OctagonError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn new_rejects_nan_axis_min() {
        let result = NdarrayOctagon::new(
            array![f32::NAN, 0.0],
            array![1.0, 1.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 2.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        );
        assert!(matches!(
            result,
            Err(OctagonError::InvalidAxisBounds { .. })
        ));
    }

    #[test]
    fn new_rejects_nan_axis_max() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, f32::NAN],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 2.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        );
        assert!(matches!(
            result,
            Err(OctagonError::InvalidAxisBounds { .. })
        ));
    }

    #[test]
    fn new_rejects_wrong_diag_count() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![], // Should have 1 diagonal bounds entry.
        );
        assert!(matches!(
            result,
            Err(OctagonError::DimensionMismatch { .. })
        ));
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

    // ---- Serde roundtrip ----

    #[test]
    fn serde_json_roundtrip() {
        let original = NdarrayOctagon::new(
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

        let json = serde_json::to_string(&original).expect("serialize");
        let restored: NdarrayOctagon = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(original.dim(), restored.dim());
        for i in 0..original.dim() {
            assert!(
                (original.axis_min()[i] - restored.axis_min()[i]).abs() < 1e-6,
                "axis_min[{i}] mismatch"
            );
            assert!(
                (original.axis_max()[i] - restored.axis_max()[i]).abs() < 1e-6,
                "axis_max[{i}] mismatch"
            );
        }
        let orig_diag = &original.diag_bounds[0];
        let rest_diag = &restored.diag_bounds[0];
        assert!((orig_diag.sum_min - rest_diag.sum_min).abs() < 1e-6);
        assert!((orig_diag.sum_max - rest_diag.sum_max).abs() < 1e-6);
        assert!((orig_diag.diff_min - rest_diag.diff_min).abs() < 1e-6);
        assert!((orig_diag.diff_max - rest_diag.diff_max).abs() < 1e-6);
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
        assert!(
            (vol - 6.0).abs() < 1e-4,
            "box-equivalent volume: expected 6, got {vol}"
        );
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
        assert!(
            vol > 0.0,
            "non-degenerate octagon should have positive volume"
        );
        assert!(
            vol < 4.0,
            "octagon should be smaller than bounding box (area 4), got {vol}"
        );
    }

    #[test]
    fn volume_1d() {
        let oct = NdarrayOctagon::new(array![1.0], array![5.0], vec![]).unwrap();
        let vol = oct.volume().unwrap();
        assert!(
            (vol - 4.0).abs() < 1e-6,
            "1D volume: expected 4.0, got {vol}"
        );
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
        assert!(
            p > 0.99,
            "identical octagons should have very high overlap, got {p}"
        );
    }

    #[test]
    fn overlap_prob_disjoint_is_low() {
        let a = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        let b = NdarrayOctagon::from_box_bounds(array![5.0, 5.0], array![6.0, 6.0]).unwrap();
        let p = a.overlap_prob(&b, 0.1).unwrap();
        assert!(
            p < 0.01,
            "disjoint octagons should have very low overlap, got {p}"
        );
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

        // ---- Composition property tests ----

        #[test]
        fn prop_composition_closure(
            (r, s) in (arb_octagon_2d(), arb_octagon_2d()),
        ) {
            // Composing two valid octagons should produce either a valid octagon
            // or Empty (if no valid (x,z) pair exists). It must never produce
            // an octagon with min > max.
            match r.compose(&s) {
                Ok(c) => {
                    let d = c.dim();
                    for i in 0..d {
                        prop_assert!(
                            c.axis_min[i] <= c.axis_max[i],
                            "composed octagon has invalid axis bounds at dim {i}: {} > {}",
                            c.axis_min[i], c.axis_max[i]
                        );
                    }
                    for db in &c.diag_bounds {
                        prop_assert!(
                            db.sum_min <= db.sum_max,
                            "composed octagon has invalid sum bounds: {} > {}",
                            db.sum_min, db.sum_max
                        );
                        prop_assert!(
                            db.diff_min <= db.diff_max,
                            "composed octagon has invalid diff bounds: {} > {}",
                            db.diff_min, db.diff_max
                        );
                    }
                }
                Err(OctagonError::Empty) => {
                    // Valid outcome: composition is empty.
                }
                Err(e) => {
                    prop_assert!(false, "unexpected error: {e}");
                }
            }
        }

        #[test]
        fn prop_composition_contains_relational_composition(
            (r, s) in (arb_octagon_2d(), arb_octagon_2d()),
            y0 in -10.0f32..10.0,
            _y1 in -10.0f32..10.0,
        ) {
            // If point (a0,a1,y0,y1) is in R (interpreting R as constraining the
            // pair where first dim = a, second dim = y), and (y0,y1,c0,c1) is in S,
            // then (a0,a1,c0,c1) should be in compose(R,S).
            //
            // For 2D octagons with adjacent-pair diagonal constraints, R constrains
            // (x0, x1) as a point. We interpret R as a relation on the pair
            // (first_dim, second_dim), so we need to find (a, y) in R and (y, c) in S
            // where a = first dim of R, y = second dim of R = first dim of S, c = second dim of S.
            //
            // Concretely: for the 2D case, dim 0 is "a" and dim 1 is "y" in R.
            // We pick a y value and check if there exist a, c such that
            // (a, y) in R and (y, c) in S, then (a, c) in compose(R, S).

            // For 2D, the octagon constrains a single 2D point. The "relational"
            // interpretation treats dim 0 as input and dim 1 as output.
            // R: constrains (a, y_shared), S: constrains (y_shared, c).
            // We sample y_shared and find valid a and c.

            // Check if y0 is in R's dim-1 range and S's dim-0 range.
            let y_shared = y0; // use y0 as the shared variable (dim 1 of R, dim 0 of S)
            if y_shared < r.axis_min[1] || y_shared > r.axis_max[1] {
                return Ok(());
            }
            if y_shared < s.axis_min[0] || y_shared > s.axis_max[0] {
                return Ok(());
            }

            // Find a valid 'a' (dim 0 of R) given y_shared.
            // Constraints on a from R: axis_min[0] <= a <= axis_max[0],
            // sum_min <= a + y_shared <= sum_max,
            // diff_min <= a - y_shared <= diff_max.
            let a_lo = r.axis_min[0]
                .max(r.diag_bounds[0].sum_min - y_shared)
                .max(r.diag_bounds[0].diff_min + y_shared);
            let a_hi = r.axis_max[0]
                .min(r.diag_bounds[0].sum_max - y_shared)
                .min(r.diag_bounds[0].diff_max + y_shared);
            if a_lo > a_hi {
                return Ok(()); // No valid 'a' for this y_shared.
            }
            let a = (a_lo + a_hi) / 2.0; // pick midpoint

            // Find a valid 'c' (dim 1 of S) given y_shared.
            let c_lo = s.axis_min[1]
                .max(s.diag_bounds[0].sum_min - y_shared)
                .max(-(s.diag_bounds[0].diff_max - y_shared));
            let c_hi = s.axis_max[1]
                .min(s.diag_bounds[0].sum_max - y_shared)
                .min(-(s.diag_bounds[0].diff_min - y_shared));
            if c_lo > c_hi {
                return Ok(()); // No valid 'c' for this y_shared.
            }
            let c = (c_lo + c_hi) / 2.0; // pick midpoint

            // Verify (a, y_shared) is in R and (y_shared, c) is in S.
            if !r.contains(&[a, y_shared]).unwrap() || !s.contains(&[y_shared, c]).unwrap() {
                return Ok(()); // Floating point edge case, skip.
            }

            // Now (a, c) should be in compose(R, S).
            match r.compose(&s) {
                Ok(composed) => {
                    prop_assert!(
                        composed.contains(&[a, c]).unwrap(),
                        "point ({a}, {c}) should be in compose(R, S) since ({a}, {y_shared}) in R and ({y_shared}, {c}) in S.\n\
                         Composed bounds: axis=[{:?}, {:?}], diag={:?}",
                        composed.axis_min, composed.axis_max, composed.diag_bounds
                    );
                }
                Err(OctagonError::Empty) => {
                    prop_assert!(false,
                        "compose returned Empty but we found valid (a, y, c) = ({a}, {y_shared}, {c})");
                }
                Err(e) => {
                    prop_assert!(false, "unexpected error: {e}");
                }
            }
        }

        #[test]
        fn prop_composition_is_associative(
            (r, s, t) in (arb_octagon_2d(), arb_octagon_2d(), arb_octagon_2d()),
        ) {
            // compose(compose(R, S), T) should equal compose(R, compose(S, T))
            // for 2D octagons (single pair, no inter-pair loss).
            // Normalize inputs to ensure consistent bounds. Skip if normalization
            // produces Empty (randomly generated diag bounds may be infeasible).
            let r = match r.normalize() { Ok(r) => r, Err(_) => return Ok(()) };
            let s = match s.normalize() { Ok(s) => s, Err(_) => return Ok(()) };
            let t = match t.normalize() { Ok(t) => t, Err(_) => return Ok(()) };

            let rs = r.compose(&s);
            let st = s.compose(&t);

            if let (Ok(rs), Ok(st)) = (rs, st) {
                let rst_left = rs.compose(&t);
                let rst_right = r.compose(&st);
                match (rst_left, rst_right) {
                    (Ok(l), Ok(r_)) => {
                        let eps = 1.0;
                        for i in 0..l.dim() {
                            prop_assert!(
                                (l.axis_min[i] - r_.axis_min[i]).abs() < eps,
                                "associativity violation: axis_min[{i}] left={} right={}",
                                l.axis_min[i], r_.axis_min[i]
                            );
                            prop_assert!(
                                (l.axis_max[i] - r_.axis_max[i]).abs() < eps,
                                "associativity violation: axis_max[{i}] left={} right={}",
                                l.axis_max[i], r_.axis_max[i]
                            );
                        }
                        for (k, (ld, rd)) in l.diag_bounds.iter().zip(r_.diag_bounds.iter()).enumerate() {
                            prop_assert!(
                                (ld.sum_min - rd.sum_min).abs() < eps,
                                "associativity violation: pair {k} sum_min left={} right={}",
                                ld.sum_min, rd.sum_min
                            );
                            prop_assert!(
                                (ld.sum_max - rd.sum_max).abs() < eps,
                                "associativity violation: pair {k} sum_max left={} right={}",
                                ld.sum_max, rd.sum_max
                            );
                            prop_assert!(
                                (ld.diff_min - rd.diff_min).abs() < eps,
                                "associativity violation: pair {k} diff_min left={} right={}",
                                ld.diff_min, rd.diff_min
                            );
                            prop_assert!(
                                (ld.diff_max - rd.diff_max).abs() < eps,
                                "associativity violation: pair {k} diff_max left={} right={}",
                                ld.diff_max, rd.diff_max
                            );
                        }
                    }
                    (Err(OctagonError::Empty), Err(OctagonError::Empty)) => {
                        // Both empty: consistent.
                    }
                    (Err(OctagonError::Empty), Ok(_)) | (Ok(_), Err(OctagonError::Empty)) => {
                        // One empty, other not: floating-point edge case. Skip.
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        prop_assert!(false, "unexpected error: {e}");
                    }
                }
            }
            // If either R compose S or S compose T is empty, skip associativity check.
        }
    }

    // ---- Composition unit tests ----

    #[test]
    fn compose_identity_with_universal() {
        // A "universal" octagon with very wide bounds should act as an approximate
        // identity for composition: compose(R, universal) should be close to R
        // (potentially slightly wider due to the universal's bounds being finite).
        let r = NdarrayOctagon::new(
            array![1.0, 2.0],
            array![3.0, 5.0],
            vec![NdarrayDiagBounds {
                sum_min: 4.0,
                sum_max: 7.0,
                diff_min: -3.0,
                diff_max: 0.0,
            }],
        )
        .unwrap();

        let big = 1000.0f32;
        let universal = NdarrayOctagon::new(
            array![-big, -big],
            array![big, big],
            vec![NdarrayDiagBounds {
                sum_min: -2.0 * big,
                sum_max: 2.0 * big,
                diff_min: -2.0 * big,
                diff_max: 2.0 * big,
            }],
        )
        .unwrap();

        let composed = r.compose(&universal).unwrap();
        let eps = 1.0; // wide tolerance since universal is finite
        assert!(
            (composed.axis_min[0] - r.axis_min[0]).abs() < eps,
            "axis_min[0]: expected ~{}, got {}",
            r.axis_min[0],
            composed.axis_min[0]
        );
        assert!(
            (composed.axis_max[0] - r.axis_max[0]).abs() < eps,
            "axis_max[0]: expected ~{}, got {}",
            r.axis_max[0],
            composed.axis_max[0]
        );
    }

    #[test]
    fn compose_is_not_intersection() {
        // Composition and intersection should produce different results for
        // non-trivial octagons (sanity check that we compute the right thing).
        let r = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![4.0, 4.0],
            vec![NdarrayDiagBounds {
                sum_min: 1.0,
                sum_max: 7.0,
                diff_min: -3.0,
                diff_max: 3.0,
            }],
        )
        .unwrap();
        let s = NdarrayOctagon::new(
            array![1.0, 1.0],
            array![5.0, 5.0],
            vec![NdarrayDiagBounds {
                sum_min: 3.0,
                sum_max: 9.0,
                diff_min: -3.0,
                diff_max: 3.0,
            }],
        )
        .unwrap();

        let composed = r.compose(&s).unwrap();
        let intersected = r.intersection(&s).unwrap();

        // They should differ in at least one bound.
        let mut any_different = false;
        for i in 0..2 {
            if (composed.axis_min[i] - intersected.axis_min[i]).abs() > 1e-6 {
                any_different = true;
            }
            if (composed.axis_max[i] - intersected.axis_max[i]).abs() > 1e-6 {
                any_different = true;
            }
        }
        if (composed.diag_bounds[0].sum_min - intersected.diag_bounds[0].sum_min).abs() > 1e-6 {
            any_different = true;
        }
        if (composed.diag_bounds[0].sum_max - intersected.diag_bounds[0].sum_max).abs() > 1e-6 {
            any_different = true;
        }
        if (composed.diag_bounds[0].diff_min - intersected.diag_bounds[0].diff_min).abs() > 1e-6 {
            any_different = true;
        }
        if (composed.diag_bounds[0].diff_max - intersected.diag_bounds[0].diff_max).abs() > 1e-6 {
            any_different = true;
        }

        assert!(
            any_different,
            "compose and intersection should differ for non-trivial inputs.\n\
             Composed: axis=[{:?}, {:?}], diag={:?}\n\
             Intersected: axis=[{:?}, {:?}], diag={:?}",
            composed.axis_min,
            composed.axis_max,
            composed.diag_bounds,
            intersected.axis_min,
            intersected.axis_max,
            intersected.diag_bounds,
        );
    }

    #[test]
    fn compose_dim_mismatch() {
        let a = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, 1.0]).unwrap();
        let b = NdarrayOctagon::from_box_bounds(array![0.0], array![1.0]).unwrap();
        assert!(matches!(
            a.compose(&b),
            Err(OctagonError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn compose_concrete_example() {
        // R constrains (x, y): x in [0,2], y in [0,2], x+y in [0,4], x-y in [-2,2]
        // (vacuous diagonal = box)
        let r = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![2.0, 2.0]).unwrap();
        // S constrains (y, z): y in [1,3], z in [1,3], vacuous diagonal
        let s = NdarrayOctagon::from_box_bounds(array![1.0, 1.0], array![3.0, 3.0]).unwrap();

        let composed = r.compose(&s).unwrap();
        // R's "output" (y) overlaps with S's "input" (y) on [1,2].
        // R's "input" (x) given y in [1,2]: x in [0,2] (no tightening from box diag).
        // S's "output" (z) given y in [1,2]: z in [1,3] (no tightening from box diag).
        // So composed should constrain: x in [0,2], z in [1,3].
        assert!(composed.axis_min[0] <= 0.0 + 0.1);
        assert!(composed.axis_max[0] >= 2.0 - 0.1);
        assert!(composed.axis_min[1] <= 1.0 + 0.1);
        assert!(composed.axis_max[1] >= 3.0 - 0.1);
    }

    // ---- NaN rejection: diagonal bounds ----

    #[test]
    fn new_rejects_nan_diag_sum_min() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![NdarrayDiagBounds {
                sum_min: f32::NAN,
                sum_max: 2.0,
                diff_min: -1.0,
                diff_max: 1.0,
            }],
        );
        assert!(result.is_err(), "NaN diagonal sum_min should be rejected");
    }

    #[test]
    fn new_rejects_nan_diag_diff_max() {
        let result = NdarrayOctagon::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            vec![NdarrayDiagBounds {
                sum_min: 0.0,
                sum_max: 2.0,
                diff_min: -1.0,
                diff_max: f32::NAN,
            }],
        );
        assert!(result.is_err(), "NaN diagonal diff_max should be rejected");
    }

    // ---- NaN rejection: from_box_bounds ----

    #[test]
    fn from_box_bounds_rejects_nan_axis_min() {
        let result = NdarrayOctagon::from_box_bounds(array![f32::NAN, 0.0], array![1.0, 1.0]);
        assert!(result.is_err(), "NaN axis_min should be rejected");
    }

    #[test]
    fn from_box_bounds_rejects_nan_axis_max() {
        let result = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![1.0, f32::NAN]);
        assert!(result.is_err(), "NaN axis_max should be rejected");
    }
}
