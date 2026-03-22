//! Python bindings for subsume geometric KG embeddings.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use subsume::dataset::{Dataset, Triple};
use subsume::ndarray_backend::{NdarrayBox, NdarrayCone, NdarrayGumbelBox};
use subsume::trainer::{
    BoxEmbeddingTrainer, ConeEmbeddingTrainer, EvaluationResults, TrainingConfig, TrainingResult,
};
use subsume::Box as BoxTrait;

/// Python wrapper around `subsume::ndarray_backend::NdarrayBox`.
#[pyclass(name = "BoxEmbedding")]
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

    /// Compute the volume of this box.
    fn volume(&self) -> PyResult<f32> {
        self.inner
            .volume()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute containment probability P(other inside self).
    fn containment_prob(&self, other: &PyNdarrayBox) -> PyResult<f32> {
        self.inner
            .containment_prob(&other.inner)
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

    fn __repr__(&self) -> String {
        let vol = self.inner.volume().unwrap_or(0.0);
        format!("BoxEmbedding(dim={}, volume={:.4})", self.inner.dim(), vol)
    }
}

/// Python wrapper around `subsume::ndarray_backend::NdarrayGumbelBox`.
#[pyclass(name = "GumbelBoxEmbedding")]
#[derive(Clone)]
struct PyNdarrayGumbelBox {
    inner: NdarrayGumbelBox,
}

#[pymethods]
impl PyNdarrayGumbelBox {
    /// Create a new Gumbel box embedding from min/max bounds and temperature.
    #[new]
    fn new(min: Vec<f32>, max: Vec<f32>, temperature: f32) -> PyResult<Self> {
        let inner = NdarrayGumbelBox::new(
            ndarray::Array1::from(min),
            ndarray::Array1::from(max),
            temperature,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Compute the Gumbel-softplus volume of this box.
    fn volume(&self) -> PyResult<f32> {
        self.inner
            .volume()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute containment probability P(other inside self).
    fn containment_prob(&self, other: &PyNdarrayGumbelBox) -> PyResult<f32> {
        self.inner
            .containment_prob(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute overlap probability between this box and another.
    fn overlap_prob(&self, other: &PyNdarrayGumbelBox) -> PyResult<f32> {
        self.inner
            .overlap_prob(&other.inner)
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

    /// Get the temperature parameter.
    fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    /// Compute the membership probability for a point.
    fn membership_probability(&self, point: Vec<f32>) -> PyResult<f32> {
        self.inner
            .membership_probability(&ndarray::Array1::from(point))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let vol = self.inner.volume().unwrap_or(0.0);
        format!(
            "GumbelBoxEmbedding(dim={}, temperature={}, volume={:.4})",
            self.inner.dim(),
            self.inner.temperature(),
            vol
        )
    }
}

/// Accept either a scalar float or a list of floats for cone apertures.
#[derive(FromPyObject)]
enum PyApertures {
    Scalar(f32),
    List(Vec<f32>),
}

/// Python wrapper around `subsume::ndarray_backend::NdarrayCone`.
#[pyclass(name = "ConeEmbedding")]
#[derive(Clone)]
struct PyNdarrayCone {
    inner: NdarrayCone,
}

#[pymethods]
impl PyNdarrayCone {
    /// Create a new cone embedding from axis angles and apertures.
    ///
    /// ``apertures`` may be a list of per-dimension values or a single float
    /// that is broadcast to all dimensions.
    #[new]
    fn new(axes: Vec<f32>, apertures: PyApertures) -> PyResult<Self> {
        let aperture_vec = match apertures {
            PyApertures::Scalar(v) => vec![v; axes.len()],
            PyApertures::List(v) => v,
        };
        let inner = NdarrayCone::new(
            ndarray::Array1::from(axes),
            ndarray::Array1::from(aperture_vec),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of dimensions.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return axis angles as a numpy array.
    fn axes<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let axes = self.inner.axes().clone();
        Ok(axes.into_pyarray(py).into())
    }

    /// Return apertures as a numpy array.
    fn apertures<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let apertures = self.inner.apertures().clone();
        Ok(apertures.into_pyarray(py).into())
    }

    /// Compute the ConE distance between self (query) and another cone (entity).
    fn cone_distance(&self, other: &PyNdarrayCone, cen: f32) -> PyResult<f32> {
        self.inner
            .cone_distance(&other.inner, cen)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("ConeEmbedding(dim={})", self.inner.dim())
    }
}

/// Training configuration for box embeddings.
///
/// All fields have sensible defaults. Pass only the fields you want to change::
///
///     config = subsumer.TrainingConfig(learning_rate=0.01, dim=16, epochs=200)
///     trainer = subsumer.BoxEmbeddingTrainer.from_config(config)
#[pyclass(name = "TrainingConfig")]
#[derive(Clone)]
struct PyTrainingConfig {
    inner: TrainingConfig,
    #[pyo3(get)]
    dim: usize,
    #[pyo3(get)]
    learning_rate: f32,
    #[pyo3(get)]
    epochs: usize,
    #[pyo3(get)]
    batch_size: usize,
    #[pyo3(get)]
    margin: f32,
    #[pyo3(get)]
    negative_samples: usize,
    #[pyo3(get)]
    negative_weight: f32,
    #[pyo3(get)]
    regularization: f32,
    #[pyo3(get)]
    gumbel_beta: f32,
    #[pyo3(get)]
    gumbel_beta_final: f32,
    #[pyo3(get)]
    warmup_epochs: usize,
    #[pyo3(get)]
    early_stopping_patience: Option<usize>,
    #[pyo3(get)]
    symmetric_loss: bool,
}

#[pymethods]
impl PyTrainingConfig {
    /// Create a training configuration.
    ///
    /// Args:
    ///     dim: Embedding dimension (required).
    ///     learning_rate: Optimizer step size (default: 0.001).
    ///     epochs: Number of training epochs for fit() (default: 100).
    ///     batch_size: Triples per batch (default: 512).
    ///     margin: Negative margin (default: 1.0).
    ///     negative_samples: Negatives per positive (default: 1).
    ///     negative_weight: Weight on negative loss term (default: 1.0).
    ///     regularization: Volume regularization weight (default: 0.0001).
    ///     gumbel_beta: Softplus steepness, start (default: 10.0).
    ///     gumbel_beta_final: Softplus steepness, end after annealing (default: 50.0).
    ///     warmup_epochs: LR warmup epochs (default: 10).
    ///     early_stopping_patience: Stop after N epochs without improvement (default: 10, None to disable).
    ///     symmetric_loss: Use symmetric min(P(A|B), P(B|A)) loss (default: False).
    #[new]
    #[pyo3(signature = (dim, learning_rate=0.001, epochs=100, batch_size=512, margin=1.0, negative_samples=1, negative_weight=1.0, regularization=0.0001, gumbel_beta=10.0, gumbel_beta_final=50.0, warmup_epochs=10, early_stopping_patience=Some(10), symmetric_loss=false))]
    fn new(
        dim: usize,
        learning_rate: f32,
        epochs: usize,
        batch_size: usize,
        margin: f32,
        negative_samples: usize,
        negative_weight: f32,
        regularization: f32,
        gumbel_beta: f32,
        gumbel_beta_final: f32,
        warmup_epochs: usize,
        early_stopping_patience: Option<usize>,
        symmetric_loss: bool,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim must be > 0"));
        }
        if learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be > 0",
            ));
        }
        if epochs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "epochs must be > 0",
            ));
        }
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be > 0",
            ));
        }
        let inner = TrainingConfig {
            learning_rate,
            epochs,
            batch_size,
            margin,
            negative_samples,
            negative_weight,
            regularization,
            gumbel_beta,
            gumbel_beta_final,
            warmup_epochs,
            early_stopping_patience,
            symmetric_loss,
            ..TrainingConfig::default()
        };
        Ok(Self {
            inner,
            dim,
            learning_rate,
            epochs,
            batch_size,
            margin,
            negative_samples,
            negative_weight,
            regularization,
            gumbel_beta,
            gumbel_beta_final,
            warmup_epochs,
            early_stopping_patience,
            symmetric_loss,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingConfig(dim={}, lr={}, epochs={}, batch_size={}, margin={}, neg_samples={}, neg_weight={}, reg={}, gumbel_beta={}, gumbel_beta_final={}, warmup={}, patience={:?}, symmetric={})",
            self.dim,
            self.learning_rate,
            self.epochs,
            self.batch_size,
            self.margin,
            self.negative_samples,
            self.negative_weight,
            self.regularization,
            self.gumbel_beta,
            self.gumbel_beta_final,
            self.warmup_epochs,
            self.early_stopping_patience,
            self.symmetric_loss,
        )
    }
}

/// Python wrapper around `subsume::trainer::BoxEmbeddingTrainer`.
///
/// Triple convention: ``(head_id, relation_id, tail_id)`` where the head box
/// is trained to **contain** the tail box.  For a hypernym relation like
/// "is-a", use ``(parent_id, rel_id, child_id)`` -- e.g.
/// ``(animal_id, hypernym_id, dog_id)`` means the animal box contains the dog
/// box.
///
/// Standard KG files list triples as ``(subject, predicate, object)``.  For
/// directed relations like hypernym/subclass, this often means
/// ``(child, hypernym, parent)`` which is the **reverse** of what the trainer
/// expects. Use :meth:`load_dataset` with ``reverse=True`` or swap head/tail
/// before calling :meth:`train_step` / :meth:`fit`.
#[pyclass(name = "BoxEmbeddingTrainer")]
struct PyBoxEmbeddingTrainer {
    inner: BoxEmbeddingTrainer,
    /// Entity names by ID (populated by from_triples/load_dataset).
    entity_names: Option<Vec<String>>,
    /// Relation names by ID.
    relation_names: Option<Vec<String>>,
}

#[pymethods]
impl PyBoxEmbeddingTrainer {
    /// Create a new trainer with given learning rate and embedding dimension.
    ///
    /// For full configuration control, use ``from_config`` instead.
    #[new]
    fn new(learning_rate: f32, dim: usize) -> Self {
        let config = TrainingConfig {
            learning_rate,
            ..TrainingConfig::default()
        };
        Self {
            inner: BoxEmbeddingTrainer::new(config, dim),
            entity_names: None,
            relation_names: None,
        }
    }

    /// Create a trainer from a TrainingConfig.
    #[staticmethod]
    fn from_config(config: &PyTrainingConfig) -> Self {
        Self {
            inner: BoxEmbeddingTrainer::new(config.inner.clone(), config.dim),
            entity_names: None,
            relation_names: None,
        }
    }

    /// Create a trainer from string triples, handling interning automatically.
    ///
    /// Args:
    ///     triples: List of (head, relation, tail) string triples.
    ///     config: Training configuration (optional, uses defaults if None).
    ///     reverse: If True, swap head and tail (use for "child hypernym parent" data).
    ///
    /// Returns:
    ///     Tuple of (trainer, train_triple_ids) where train_triple_ids is a
    ///     list of (head_id, relation_id, tail_id) tuples ready for train_step/fit.
    #[staticmethod]
    #[pyo3(signature = (triples, config=None, reverse=false))]
    fn from_triples(
        triples: Vec<(String, String, String)>,
        config: Option<&PyTrainingConfig>,
        reverse: bool,
    ) -> PyResult<(Self, Vec<(usize, usize, usize)>)> {
        let string_triples: Vec<Triple> = triples
            .into_iter()
            .map(|(h, r, t)| {
                if reverse {
                    Triple::new(t, r, h)
                } else {
                    Triple::new(h, r, t)
                }
            })
            .collect();

        let dataset = Dataset::new(string_triples, vec![], vec![]);
        let interned = dataset.into_interned();

        let train_ids: Vec<(usize, usize, usize)> = interned
            .train
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();

        let entity_names: Vec<String> = (0..interned.entities.len())
            .map(|i| interned.entities.get(i).unwrap_or("?").to_string())
            .collect();
        let relation_names: Vec<String> = (0..interned.relations.len())
            .map(|i| interned.relations.get(i).unwrap_or("?").to_string())
            .collect();

        let (cfg, dim) = if let Some(c) = config {
            (c.inner.clone(), c.dim)
        } else {
            (TrainingConfig::default(), 16)
        };

        let trainer = Self {
            inner: BoxEmbeddingTrainer::new(cfg, dim),
            entity_names: Some(entity_names),
            relation_names: Some(relation_names),
        };

        Ok((trainer, train_ids))
    }

    /// Run one training step over the given triples.
    ///
    /// Each triple is ``(head_id, relation_id, tail_id)`` where the head box
    /// should contain the tail box. Returns the average loss.
    fn train_step(&mut self, triples: Vec<(usize, usize, usize)>) -> PyResult<f32> {
        self.inner
            .train_step(&triples)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Train for multiple epochs with optional validation and early stopping.
    ///
    /// Args:
    ///     train_triples: List of (head_id, relation_id, tail_id) tuples.
    ///     val_triples: Optional validation triples for MRR tracking.
    ///     num_entities: Total entity count (required if val_triples provided).
    ///
    /// Returns:
    ///     Dict with keys: mrr, hits_at_1, hits_at_3, hits_at_10, mean_rank,
    ///     loss_history, validation_mrr_history, best_epoch.
    #[pyo3(signature = (train_triples, val_triples=None, num_entities=None))]
    fn fit(
        &mut self,
        train_triples: Vec<(usize, usize, usize)>,
        val_triples: Option<Vec<(usize, usize, usize)>>,
        num_entities: Option<usize>,
    ) -> PyResult<PyObject> {
        use subsume::dataset::{TripleIds, Vocab};

        let val_data = if let Some(vt) = &val_triples {
            let val_triple_ids: Vec<TripleIds> = vt
                .iter()
                .map(|&(h, r, t)| TripleIds {
                    head: h,
                    relation: r,
                    tail: t,
                })
                .collect();
            let n = num_entities.unwrap_or_else(|| {
                let max_id = train_triples
                    .iter()
                    .chain(vt.iter())
                    .flat_map(|&(h, _, t)| [h, t])
                    .max()
                    .unwrap_or(0);
                max_id + 1
            });
            let mut vocab = Vocab::default();
            for i in 0..n {
                vocab.intern(i.to_string());
            }
            Some((val_triple_ids, vocab))
        } else {
            None
        };

        let validation = val_data
            .as_ref()
            .map(|(triples, vocab)| (triples.as_slice(), vocab));

        let result = self
            .inner
            .fit(&train_triples, validation, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        training_result_to_dict(result)
    }

    /// Evaluate on test triples, returning MRR, Hits@K, Mean Rank.
    ///
    /// Args:
    ///     test_triples: List of (head_id, relation_id, tail_id) tuples.
    ///     num_entities: Total entity count (for ranking against all entities).
    ///         If None, auto-computed as max(entity_id) + 1 from triples and
    ///         trained boxes.
    ///
    /// Returns:
    ///     Dict with keys: mrr, head_mrr, tail_mrr, hits_at_1, hits_at_3,
    ///     hits_at_10, mean_rank.
    #[pyo3(signature = (test_triples, num_entities=None))]
    fn evaluate(
        &self,
        test_triples: Vec<(usize, usize, usize)>,
        num_entities: Option<usize>,
    ) -> PyResult<PyObject> {
        use subsume::dataset::{TripleIds, Vocab};

        let triple_ids: Vec<TripleIds> = test_triples
            .iter()
            .map(|&(h, r, t)| TripleIds {
                head: h,
                relation: r,
                tail: t,
            })
            .collect();

        let n = num_entities.unwrap_or_else(|| {
            let max_from_triples = test_triples
                .iter()
                .flat_map(|&(h, _, t)| [h, t])
                .max()
                .unwrap_or(0);
            let max_from_boxes = self.inner.boxes.keys().copied().max().unwrap_or(0);
            max_from_triples.max(max_from_boxes) + 1
        });

        let mut vocab = Vocab::default();
        for i in 0..n {
            vocab.intern(i.to_string());
        }

        let results = self
            .inner
            .evaluate(&triple_ids, &vocab, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        eval_results_to_dict(results)
    }

    /// Get the learned box for an entity, if it exists.
    fn get_box(&self, entity_id: usize) -> PyResult<Option<PyNdarrayBox>> {
        Ok(self
            .inner
            .get_box(entity_id)
            .map(|b| PyNdarrayBox { inner: b }))
    }

    /// Export all embeddings as ``(entity_ids, min_bounds_2d, max_bounds_2d)``.
    ///
    /// ``min_bounds_2d`` and ``max_bounds_2d`` are numpy arrays of shape ``(n_entities, dim)``.
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

        Ok((
            ids,
            mins_nd.into_pyarray(py).into(),
            maxs_nd.into_pyarray(py).into(),
        ))
    }

    /// Get entity name for an ID (if loaded via from_triples or load_dataset).
    fn entity_name(&self, entity_id: usize) -> Option<String> {
        self.entity_names
            .as_ref()
            .and_then(|names| names.get(entity_id).cloned())
    }

    /// Get all entity names (if loaded via from_triples or load_dataset).
    #[getter]
    fn get_entity_names(&self) -> Option<Vec<String>> {
        self.entity_names.clone()
    }

    /// Get all relation names (if loaded via from_triples or load_dataset).
    #[getter]
    fn get_relation_names(&self) -> Option<Vec<String>> {
        self.relation_names.clone()
    }

    /// Score a single (head, relation, tail) triple.
    ///
    /// Returns the containment probability P(tail inside translated_head).
    fn predict(&self, head_id: usize, _relation_id: usize, tail_id: usize) -> PyResult<f32> {
        let box_h = self.inner.get_box(head_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("no box for entity {head_id}"))
        })?;
        let box_t = self.inner.get_box(tail_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("no box for entity {tail_id}"))
        })?;
        box_h
            .containment_prob(&box_t)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Score all entities as tails for a (head, relation) query.
    ///
    /// Returns a numpy array of shape (num_entities,) with containment probabilities.
    /// Entities without learned embeddings get score 0.0.
    fn score_tails<'py>(
        &self,
        py: Python<'py>,
        head_id: usize,
        _relation_id: usize,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let box_h = self.inner.get_box(head_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("no box for entity {head_id}"))
        })?;
        let max_id = self.inner.boxes.keys().copied().max().unwrap_or(0);
        let mut scores = vec![0.0f32; max_id + 1];
        for (&eid, tb) in &self.inner.boxes {
            if let Ok(nb) = tb.to_ndarray_box() {
                scores[eid] = box_h.containment_prob(&nb).unwrap_or(0.0);
            }
        }
        let arr = ndarray::Array1::from(scores);
        Ok(arr.into_pyarray(py).into())
    }

    fn __repr__(&self) -> String {
        format!(
            "BoxEmbeddingTrainer(dim={}, n_entities={})",
            self.inner.dim,
            self.inner.boxes.len()
        )
    }

    /// Serialize trainer state to JSON (includes entity/relation names).
    fn save_checkpoint(&self) -> PyResult<String> {
        let trainer_json = serde_json::to_value(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut checkpoint = serde_json::Map::new();
        checkpoint.insert("trainer".into(), trainer_json);
        if let Some(ref names) = self.entity_names {
            checkpoint.insert(
                "entity_names".into(),
                serde_json::to_value(names)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            );
        }
        if let Some(ref names) = self.relation_names {
            checkpoint.insert(
                "relation_names".into(),
                serde_json::to_value(names)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            );
        }

        serde_json::to_string(&checkpoint)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deserialize trainer state from JSON.
    ///
    /// Accepts both the new envelope format (with entity_names) and the
    /// legacy format (bare trainer JSON) for backward compatibility.
    #[staticmethod]
    fn load_checkpoint(json: &str) -> PyResult<Self> {
        // Try new envelope format first.
        if let Ok(map) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(json) {
            if let Some(trainer_val) = map.get("trainer") {
                let inner: BoxEmbeddingTrainer = serde_json::from_value(trainer_val.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let entity_names: Option<Vec<String>> = map
                    .get("entity_names")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let relation_names: Option<Vec<String>> = map
                    .get("relation_names")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                return Ok(Self {
                    inner,
                    entity_names,
                    relation_names,
                });
            }
        }

        // Fall back to legacy bare-trainer format.
        let inner: BoxEmbeddingTrainer = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            entity_names: None,
            relation_names: None,
        })
    }
}

/// Python wrapper around `subsume::trainer::ConeEmbeddingTrainer`.
///
/// Same triple convention as `BoxEmbeddingTrainer`: head **contains** tail.
/// Each entity is a cone (axis angles + apertures) instead of a box.
/// Cones are better suited for DAG/partial-order relationships.
#[pyclass(name = "ConeEmbeddingTrainer")]
struct PyConeEmbeddingTrainer {
    inner: ConeEmbeddingTrainer,
    /// Entity names by ID (populated by from_triples).
    entity_names: Option<Vec<String>>,
    /// Relation names by ID.
    relation_names: Option<Vec<String>>,
}

#[pymethods]
impl PyConeEmbeddingTrainer {
    /// Create a new cone trainer with given learning rate and embedding dimension.
    ///
    /// For full configuration control, use ``from_config`` instead.
    #[new]
    fn new(learning_rate: f32, dim: usize) -> Self {
        let config = TrainingConfig {
            learning_rate,
            ..TrainingConfig::default()
        };
        Self {
            inner: ConeEmbeddingTrainer::new(config, dim, None),
            entity_names: None,
            relation_names: None,
        }
    }

    /// Create a cone trainer from a TrainingConfig.
    #[staticmethod]
    fn from_config(config: &PyTrainingConfig) -> Self {
        Self {
            inner: ConeEmbeddingTrainer::new(config.inner.clone(), config.dim, None),
            entity_names: None,
            relation_names: None,
        }
    }

    /// Create a cone trainer from string triples, handling interning automatically.
    ///
    /// Args:
    ///     triples: List of (head, relation, tail) string triples.
    ///     config: Training configuration (optional, uses defaults if None).
    ///     reverse: If True, swap head and tail (use for "child hypernym parent" data).
    ///
    /// Returns:
    ///     Tuple of (trainer, train_triple_ids) where train_triple_ids is a
    ///     list of (head_id, relation_id, tail_id) tuples ready for train_step/fit.
    #[staticmethod]
    #[pyo3(signature = (triples, config=None, reverse=false))]
    fn from_triples(
        triples: Vec<(String, String, String)>,
        config: Option<&PyTrainingConfig>,
        reverse: bool,
    ) -> PyResult<(Self, Vec<(usize, usize, usize)>)> {
        let string_triples: Vec<Triple> = triples
            .into_iter()
            .map(|(h, r, t)| {
                if reverse {
                    Triple::new(t, r, h)
                } else {
                    Triple::new(h, r, t)
                }
            })
            .collect();

        let dataset = Dataset::new(string_triples, vec![], vec![]);
        let interned = dataset.into_interned();

        let train_ids: Vec<(usize, usize, usize)> = interned
            .train
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();

        let entity_names: Vec<String> = (0..interned.entities.len())
            .map(|i| interned.entities.get(i).unwrap_or("?").to_string())
            .collect();
        let relation_names: Vec<String> = (0..interned.relations.len())
            .map(|i| interned.relations.get(i).unwrap_or("?").to_string())
            .collect();

        let (cfg, dim) = if let Some(c) = config {
            (c.inner.clone(), c.dim)
        } else {
            (TrainingConfig::default(), 16)
        };

        let trainer = Self {
            inner: ConeEmbeddingTrainer::new(cfg, dim, None),
            entity_names: Some(entity_names),
            relation_names: Some(relation_names),
        };

        Ok((trainer, train_ids))
    }

    /// Run one training step over the given triples.
    ///
    /// Each triple is ``(head_id, relation_id, tail_id)`` where the head cone
    /// should contain the tail cone. Returns the average loss.
    fn train_step(&mut self, triples: Vec<(usize, usize, usize)>) -> PyResult<f32> {
        self.inner
            .train_step_batch(&triples)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Train for multiple epochs with optional validation and early stopping.
    ///
    /// Args:
    ///     train_triples: List of (head_id, relation_id, tail_id) tuples.
    ///     val_triples: Optional validation triples for MRR tracking.
    ///     num_entities: Total entity count (required if val_triples provided).
    ///
    /// Returns:
    ///     Dict with keys: mrr, hits_at_1, hits_at_3, hits_at_10, mean_rank,
    ///     loss_history, validation_mrr_history, best_epoch.
    #[pyo3(signature = (train_triples, val_triples=None, num_entities=None))]
    fn fit(
        &mut self,
        train_triples: Vec<(usize, usize, usize)>,
        val_triples: Option<Vec<(usize, usize, usize)>>,
        num_entities: Option<usize>,
    ) -> PyResult<PyObject> {
        use subsume::dataset::{TripleIds, Vocab};

        let val_data = if let Some(vt) = &val_triples {
            let val_triple_ids: Vec<TripleIds> = vt
                .iter()
                .map(|&(h, r, t)| TripleIds {
                    head: h,
                    relation: r,
                    tail: t,
                })
                .collect();
            let n = num_entities.unwrap_or_else(|| {
                let max_id = train_triples
                    .iter()
                    .chain(vt.iter())
                    .flat_map(|&(h, _, t)| [h, t])
                    .max()
                    .unwrap_or(0);
                max_id + 1
            });
            let mut vocab = Vocab::default();
            for i in 0..n {
                vocab.intern(i.to_string());
            }
            Some((val_triple_ids, vocab))
        } else {
            None
        };

        let validation = val_data
            .as_ref()
            .map(|(triples, vocab)| (triples.as_slice(), vocab));

        let result = self
            .inner
            .fit(&train_triples, validation, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        training_result_to_dict(result)
    }

    /// Export all cone embeddings as ``(entity_ids, axes_2d, apertures_2d)``.
    ///
    /// ``axes_2d`` and ``apertures_2d`` are numpy arrays of shape ``(n_entities, dim)``.
    #[allow(clippy::type_complexity)]
    fn export_embeddings<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<usize>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
        let (ids, axes_flat, apertures_flat) = self.inner.export_embeddings();
        let n = ids.len();
        let dim = self.inner.dim;

        let axes_nd = ndarray::Array2::from_shape_vec((n, dim), axes_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let apertures_nd = ndarray::Array2::from_shape_vec((n, dim), apertures_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok((
            ids,
            axes_nd.into_pyarray(py).into(),
            apertures_nd.into_pyarray(py).into(),
        ))
    }

    /// Get entity name for an ID (if loaded via from_triples).
    fn entity_name(&self, entity_id: usize) -> Option<String> {
        self.entity_names
            .as_ref()
            .and_then(|names| names.get(entity_id).cloned())
    }

    /// Get all entity names (if loaded via from_triples).
    #[getter]
    fn get_entity_names(&self) -> Option<Vec<String>> {
        self.entity_names.clone()
    }

    /// Get all relation names (if loaded via from_triples).
    #[getter]
    fn get_relation_names(&self) -> Option<Vec<String>> {
        self.relation_names.clone()
    }

    /// Evaluate on test triples, returning MRR, Hits@K, Mean Rank.
    #[pyo3(signature = (test_triples, num_entities=None))]
    fn evaluate(
        &self,
        test_triples: Vec<(usize, usize, usize)>,
        num_entities: Option<usize>,
    ) -> PyResult<PyObject> {
        use subsume::dataset::{TripleIds, Vocab};

        let triple_ids: Vec<TripleIds> = test_triples
            .iter()
            .map(|&(h, r, t)| TripleIds {
                head: h,
                relation: r,
                tail: t,
            })
            .collect();

        let n = num_entities.unwrap_or_else(|| {
            let max_from_triples = test_triples
                .iter()
                .flat_map(|&(h, _, t)| [h, t])
                .max()
                .unwrap_or(0);
            let max_from_cones = self.inner.cones.keys().copied().max().unwrap_or(0);
            max_from_triples.max(max_from_cones) + 1
        });

        let mut vocab = Vocab::default();
        for i in 0..n {
            vocab.intern(i.to_string());
        }

        let results = self
            .inner
            .evaluate(&triple_ids, &vocab, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        eval_results_to_dict(results)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConeEmbeddingTrainer(dim={}, n_entities={})",
            self.inner.dim,
            self.inner.cones.len()
        )
    }

    /// Serialize trainer state to JSON (includes entity/relation names).
    fn save_checkpoint(&self) -> PyResult<String> {
        let trainer_json = serde_json::to_value(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut checkpoint = serde_json::Map::new();
        checkpoint.insert("trainer".into(), trainer_json);
        if let Some(ref names) = self.entity_names {
            checkpoint.insert(
                "entity_names".into(),
                serde_json::to_value(names)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            );
        }
        if let Some(ref names) = self.relation_names {
            checkpoint.insert(
                "relation_names".into(),
                serde_json::to_value(names)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            );
        }

        serde_json::to_string(&checkpoint)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Deserialize trainer state from JSON.
    ///
    /// Accepts both the new envelope format (with entity_names) and the
    /// legacy format (bare trainer JSON) for backward compatibility.
    #[staticmethod]
    fn load_checkpoint(json: &str) -> PyResult<Self> {
        // Try new envelope format first.
        if let Ok(map) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(json) {
            if let Some(trainer_val) = map.get("trainer") {
                let inner: ConeEmbeddingTrainer = serde_json::from_value(trainer_val.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let entity_names: Option<Vec<String>> = map
                    .get("entity_names")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let relation_names: Option<Vec<String>> = map
                    .get("relation_names")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                return Ok(Self {
                    inner,
                    entity_names,
                    relation_names,
                });
            }
        }

        // Fall back to legacy bare-trainer format.
        let inner: ConeEmbeddingTrainer = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            entity_names: None,
            relation_names: None,
        })
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

    a.containment_prob(&b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute distance from a point (vector) to a box.
///
/// Returns 0.0 if the point is inside the box. Otherwise returns the
/// Euclidean distance to the nearest point on the box boundary.
#[pyfunction]
fn vector_to_box_distance(
    point: Vec<f32>,
    min_bounds: Vec<f32>,
    max_bounds: Vec<f32>,
    temperature: f32,
) -> PyResult<f32> {
    let box_ = NdarrayBox::new(
        ndarray::Array1::from(min_bounds),
        ndarray::Array1::from(max_bounds),
        temperature,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    subsume::ndarray_backend::distance::vector_to_box_distance(&ndarray::Array1::from(point), &box_)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute boundary distance between an outer and inner box.
///
/// Returns the minimum gap between the inner box and the outer box's
/// boundary, or None if the inner box is not fully contained in the outer.
#[pyfunction]
fn boundary_distance(
    outer_min: Vec<f32>,
    outer_max: Vec<f32>,
    inner_min: Vec<f32>,
    inner_max: Vec<f32>,
    temperature: f32,
) -> PyResult<Option<f32>> {
    let outer = NdarrayBox::new(
        ndarray::Array1::from(outer_min),
        ndarray::Array1::from(outer_max),
        temperature,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let inner = NdarrayBox::new(
        ndarray::Array1::from(inner_min),
        ndarray::Array1::from(inner_max),
        temperature,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    subsume::ndarray_backend::distance::boundary_distance(&outer, &inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute fuzzy containment score between two cones.
///
/// Returns a score in [0, 1] using sigmoid: ``1 / (1 + exp(gamma * distance))``.
/// Higher scores mean stronger containment of ``entity`` within ``query``.
///
/// Args:
///     query_axes: Axis angles for the query cone.
///     query_apertures: Apertures for the query cone.
///     entity_axes: Axis angles for the entity cone.
///     entity_apertures: Apertures for the entity cone.
///     cen: Center weight for ConE distance (default: 0.02).
///     gamma: Sigmoid steepness (default: 10.0).
#[pyfunction]
#[pyo3(signature = (query_axes, query_apertures, entity_axes, entity_apertures, cen=0.02, gamma=10.0))]
fn cone_containment_score(
    query_axes: Vec<f32>,
    query_apertures: Vec<f32>,
    entity_axes: Vec<f32>,
    entity_apertures: Vec<f32>,
    cen: f32,
    gamma: f32,
) -> PyResult<f32> {
    let query = NdarrayCone::new(
        ndarray::Array1::from(query_axes),
        ndarray::Array1::from(query_apertures),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let entity = NdarrayCone::new(
        ndarray::Array1::from(entity_axes),
        ndarray::Array1::from(entity_apertures),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    subsume::cone_query::cone_containment_score(&query, &entity, cen, gamma)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Load a KG dataset from a directory containing train.txt, valid.txt, test.txt.
///
/// Each file has one triple per line: ``head\trelation\ttail`` (tab or space separated).
///
/// Args:
///     path: Directory containing the dataset files.
///     reverse: If True, swap head and tail in all triples. Use this when your
///         data has ``(child, hypernym, parent)`` ordering but you want the
///         trainer to learn ``parent contains child``.
///
/// Returns:
///     Tuple of (trainer_triples, val_triples, test_triples, entity_names, relation_names)
///     where triples are lists of (head_id, relation_id, tail_id).
#[pyfunction]
#[pyo3(signature = (path, reverse=false))]
fn load_dataset(
    path: &str,
    reverse: bool,
) -> PyResult<(
    Vec<(usize, usize, usize)>,
    Vec<(usize, usize, usize)>,
    Vec<(usize, usize, usize)>,
    Vec<String>,
    Vec<String>,
)> {
    let dataset = subsume::dataset::load_dataset(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dataset = if reverse {
        Dataset::new(
            dataset
                .train
                .into_iter()
                .map(|t| Triple::new(t.tail, t.relation, t.head))
                .collect(),
            dataset
                .valid
                .into_iter()
                .map(|t| Triple::new(t.tail, t.relation, t.head))
                .collect(),
            dataset
                .test
                .into_iter()
                .map(|t| Triple::new(t.tail, t.relation, t.head))
                .collect(),
        )
    } else {
        dataset
    };

    let interned = dataset.into_interned();

    let to_tuples = |triples: &[subsume::dataset::TripleIds]| -> Vec<(usize, usize, usize)> {
        triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect()
    };

    let train = to_tuples(&interned.train);
    let valid = to_tuples(&interned.valid);
    let test = to_tuples(&interned.test);

    let entity_names: Vec<String> = (0..interned.entities.len())
        .map(|i| interned.entities.get(i).unwrap_or("?").to_string())
        .collect();
    let relation_names: Vec<String> = (0..interned.relations.len())
        .map(|i| interned.relations.get(i).unwrap_or("?").to_string())
        .collect();

    Ok((train, valid, test, entity_names, relation_names))
}

/// Convert EvaluationResults to a Python dict.
fn eval_results_to_dict(results: EvaluationResults) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("mrr", results.mrr)?;
        dict.set_item("head_mrr", results.head_mrr)?;
        dict.set_item("tail_mrr", results.tail_mrr)?;
        dict.set_item("hits_at_1", results.hits_at_1)?;
        dict.set_item("hits_at_3", results.hits_at_3)?;
        dict.set_item("hits_at_10", results.hits_at_10)?;
        dict.set_item("mean_rank", results.mean_rank)?;
        Ok(dict.into())
    })
}

/// Convert TrainingResult to a Python dict.
fn training_result_to_dict(result: TrainingResult) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("mrr", result.final_results.mrr)?;
        dict.set_item("hits_at_1", result.final_results.hits_at_1)?;
        dict.set_item("hits_at_3", result.final_results.hits_at_3)?;
        dict.set_item("hits_at_10", result.final_results.hits_at_10)?;
        dict.set_item("mean_rank", result.final_results.mean_rank)?;
        dict.set_item("loss_history", result.loss_history)?;
        dict.set_item("validation_mrr_history", result.validation_mrr_history)?;
        dict.set_item("best_epoch", result.best_epoch)?;
        Ok(dict.into())
    })
}

/// Geometric region embeddings for knowledge graph subsumption.
///
/// Python bindings for the ``subsume`` Rust crate.
///
/// Triple convention: head box **contains** tail box. For hypernym/subclass
/// relations where data is ``(child, hypernym, parent)``, pass ``reverse=True``
/// to ``load_dataset`` or ``from_triples``.
///
/// Quick start::
///
///     import subsumer
///
///     # From string triples (head contains tail)
///     triples = [("animal", "hypernym", "dog"), ("animal", "hypernym", "cat")]
///     trainer, ids = subsumer.BoxEmbeddingTrainer.from_triples(triples)
///     result = trainer.fit(ids)
///     print(f"MRR: {result['mrr']:.3f}")
///
///     # From files (WN18RR format)
///     train, val, test, ents, rels = subsumer.load_dataset("data/wn18rr")
///     config = subsumer.TrainingConfig(dim=32, epochs=50, learning_rate=0.01)
///     trainer = subsumer.BoxEmbeddingTrainer.from_config(config)
///     result = trainer.fit(train, val_triples=val, num_entities=len(ents))
#[pymodule]
fn subsumer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyNdarrayBox>()?;
    m.add_class::<PyNdarrayGumbelBox>()?;
    m.add_class::<PyNdarrayCone>()?;
    m.add_class::<PyBoxEmbeddingTrainer>()?;
    m.add_class::<PyConeEmbeddingTrainer>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_function(wrap_pyfunction!(containment_probability, m)?)?;
    m.add_function(wrap_pyfunction!(vector_to_box_distance, m)?)?;
    m.add_function(wrap_pyfunction!(boundary_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cone_containment_score, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    Ok(())
}
