//! Ndarray implementation of GumbelBox trait.

use crate::ndarray_box::NdarrayBox;
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use subsume_core::{
    gumbel_membership_prob, map_gumbel_to_bounds, sample_gumbel, Box, BoxError, GumbelBox,
};

/// A Gumbel box embedding implemented using `ndarray::Array1<f32>`.
#[derive(Debug, Clone)]
pub struct NdarrayGumbelBox {
    inner: NdarrayBox,
}

impl Serialize for NdarrayGumbelBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NdarrayGumbelBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = NdarrayBox::deserialize(deserializer)?;
        Ok(Self { inner })
    }
}

impl NdarrayGumbelBox {
    /// Create a new NdarrayGumbelBox.
    pub fn new(min: Array1<f32>, max: Array1<f32>, temperature: f32) -> Result<Self, BoxError> {
        Ok(Self {
            inner: NdarrayBox::new(min, max, temperature)?,
        })
    }
}

impl Box for NdarrayGumbelBox {
    type Scalar = f32;
    type Vector = Array1<f32>;

    fn min(&self) -> &Self::Vector {
        self.inner.min()
    }

    fn max(&self) -> &Self::Vector {
        self.inner.max()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn volume(&self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
        self.inner.volume(temperature)
    }

    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.intersection(&other.inner)?,
        })
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        self.inner.containment_prob(&other.inner, temperature)
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        self.inner.overlap_prob(&other.inner, temperature)
    }

    fn union(&self, other: &Self) -> Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.union(&other.inner)?,
        })
    }

    fn center(&self) -> Result<Self::Vector, BoxError> {
        self.inner.center()
    }

    fn distance(&self, other: &Self) -> Result<Self::Scalar, BoxError> {
        self.inner.distance(&other.inner)
    }
}

impl GumbelBox for NdarrayGumbelBox {
    fn temperature(&self) -> Self::Scalar {
        self.inner.temperature
    }

    fn membership_probability(&self, point: &Self::Vector) -> Result<Self::Scalar, BoxError> {
        if point.len() != self.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: point.len(),
            });
        }

        // P(point ∈ box) = ∏ P(min[i] <= point[i] <= max[i])
        // Using numerically stable Gumbel-Softmax probability calculation
        let temp = self.temperature();
        let mut prob = 1.0;

        for (i, &coord) in point.iter().enumerate() {
            let dim_prob = gumbel_membership_prob(coord, self.min()[i], self.max()[i], temp);
            prob *= dim_prob;
        }

        Ok(prob)
    }

    fn sample(&self) -> Self::Vector {
        use ndarray::Array1;

        // Use LCG for pseudo-random sampling to avoid rand dependency conflicts.
        // Note: LCG has known limitations (correlation, period) but is sufficient
        // for non-cryptographic sampling in embeddings.
        let temp = self.temperature();
        let dim = self.dim();

        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: u64 = 1u64 << 32;

        let mut sampled = Vec::with_capacity(dim);
        for i in 0..dim {
            seed = (A.wrapping_mul(seed).wrapping_add(C)) % M;
            let u = (seed as f32) / (M as f32);
            let gumbel = sample_gumbel(u, 1e-7);
            let value = map_gumbel_to_bounds(gumbel, self.min()[i], self.max()[i], temp);
            sampled.push(value);
        }

        Array1::from(sampled)
    }
}
