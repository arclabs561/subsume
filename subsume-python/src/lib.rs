//! Python bindings for subsume geometric KG embeddings.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use subsume::ndarray_backend::NdarrayBox;
use subsume::trainer::{BoxEmbeddingTrainer, TrainingConfig};
use subsume::Box as BoxTrait;

/// Python wrapper around `subsume::ndarray_backend::NdarrayBox`.
#[pyclass(name = "NdarrayBox")]
#[derive(Clone)]
struct PyNdarrayBox {
    inner: NdarrayBox,
}

#[pymethods]
impl PyNdarrayBox {
    /// Create a new box embedding from min/max bounds and temperature.
    #[new]
    fn new(min: Vec<f32>, max: Vec<f32>, temperature: f32) -> PyResult<Self> {
        let min = ndarray::Array1::from(min);
        let max = ndarray::Array1::from(max);
        let inner = NdarrayBox::new(min, max, temperature)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Compute the volume of this box at the given temperature.
    fn volume(&self, temperature: f32) -> PyResult<f32> {
        self.inner
            .volume(temperature)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute containment probability P(other inside self).
    fn containment_prob(&self, other: &PyNdarrayBox, temperature: f32) -> PyResult<f32> {
        self.inner
            .containment_prob(&other.inner, temperature)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Number of dimensions.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return min bounds as a numpy array.
    fn min_array<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let min = self.inner.min().clone();
        Ok(min.into_pyarray(py).into())
    }

    /// Return max bounds as a numpy array.
    fn max_array<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let max = self.inner.max().clone();
        Ok(max.into_pyarray(py).into())
    }
}

/// Python wrapper around `subsume::trainer::BoxEmbeddingTrainer`.
#[pyclass(name = "BoxEmbeddingTrainer")]
struct PyBoxEmbeddingTrainer {
    inner: BoxEmbeddingTrainer,
}

#[pymethods]
impl PyBoxEmbeddingTrainer {
    /// Create a new trainer with given learning rate and embedding dimension.
    #[new]
    fn new(learning_rate: f32, dim: usize) -> Self {
        let config = TrainingConfig {
            learning_rate,
            ..TrainingConfig::default()
        };
        Self {
            inner: BoxEmbeddingTrainer::new(config, dim),
        }
    }

    /// Run one training step over the given triples.
    ///
    /// Each triple is `(head_id, relation_id, tail_id)`.
    /// Returns the average loss.
    fn train_step(&mut self, triples: Vec<(usize, usize, usize)>) -> PyResult<f32> {
        self.inner
            .train_step(&triples)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the learned box for an entity, if it exists.
    fn get_box(&self, entity_id: usize) -> PyResult<Option<PyNdarrayBox>> {
        Ok(self
            .inner
            .get_box(entity_id)
            .map(|b| PyNdarrayBox { inner: b }))
    }

    /// Export all embeddings as `(entity_ids, min_bounds_2d, max_bounds_2d)`.
    ///
    /// `min_bounds_2d` and `max_bounds_2d` are numpy arrays of shape `(n_entities, dim)`.
    #[allow(clippy::type_complexity)]
    fn export_embeddings<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<usize>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
        let (ids, mins_flat, maxs_flat) = self.inner.export_embeddings();
        let n = ids.len();
        let dim = self.inner.dim;

        let mins_nd = ndarray::Array2::from_shape_vec((n, dim), mins_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let maxs_nd = ndarray::Array2::from_shape_vec((n, dim), maxs_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok((ids, mins_nd.into_pyarray(py).into(), maxs_nd.into_pyarray(py).into()))
    }

    /// Serialize trainer state to JSON.
    fn save_checkpoint(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deserialize trainer state from JSON.
    #[staticmethod]
    fn load_checkpoint(json: &str) -> PyResult<Self> {
        let inner: BoxEmbeddingTrainer = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Compute containment probability between two boxes specified by flat coordinate vectors.
///
/// P(box_b inside box_a) using the given temperature.
#[pyfunction]
fn containment_probability(
    min_a: Vec<f32>,
    max_a: Vec<f32>,
    min_b: Vec<f32>,
    max_b: Vec<f32>,
    temperature: f32,
) -> PyResult<f32> {
    let a = NdarrayBox::new(
        ndarray::Array1::from(min_a),
        ndarray::Array1::from(max_a),
        temperature,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let b = NdarrayBox::new(
        ndarray::Array1::from(min_b),
        ndarray::Array1::from(max_b),
        temperature,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    a.containment_prob(&b, temperature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Python module for subsume geometric KG embeddings.
#[pymodule]
fn subsume_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdarrayBox>()?;
    m.add_class::<PyBoxEmbeddingTrainer>()?;
    m.add_function(wrap_pyfunction!(containment_probability, m)?)?;
    Ok(())
}
