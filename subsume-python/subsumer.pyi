"""Type stubs for subsumer Python bindings (PyPI package: subsumer).

Geometric region embeddings for knowledge graph subsumption.

Triple convention: head box **contains** tail box. For hypernym/subclass
relations where data is (child, hypernym, parent), pass ``reverse=True``
to ``load_dataset`` or ``from_triples``.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

__version__: str

class NdarrayBox:
    """Axis-aligned box embedding with softplus-based containment."""

    def __init__(self, min: List[float], max: List[float], temperature: float) -> None:
        """Create a box embedding from min/max bounds and temperature.

        Args:
            min: Lower bounds for each dimension.
            max: Upper bounds for each dimension.
            temperature: Softplus temperature for containment probabilities.

        Raises:
            ValueError: If min and max have different lengths.
        """
        ...

    def volume(self) -> float:
        """Compute the volume of this box.

        Returns:
            Product of side lengths across all dimensions.
        """
        ...

    def containment_prob(self, other: "NdarrayBox") -> float:
        """Compute containment probability P(other inside self).

        Args:
            other: Another box embedding.

        Returns:
            Probability in [0.0, 1.0].
        """
        ...

    def dim(self) -> int:
        """Return the number of dimensions.

        Returns:
            Dimensionality of the box.
        """
        ...

    def min_array(self) -> npt.NDArray[np.float32]:
        """Return min bounds as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

    def max_array(self) -> npt.NDArray[np.float32]:
        """Return max bounds as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

class NdarrayGumbelBox:
    """Gumbel box embedding with softplus-based intersection volumes."""

    def __init__(self, min: List[float], max: List[float], temperature: float) -> None:
        """Create a Gumbel box embedding from min/max bounds and temperature.

        Args:
            min: Lower bounds for each dimension.
            max: Upper bounds for each dimension.
            temperature: Softplus temperature for volume computation.

        Raises:
            ValueError: If min and max have different lengths.
        """
        ...

    def volume(self) -> float:
        """Compute the Gumbel-softplus volume of this box.

        Returns:
            Gumbel-softplus volume.
        """
        ...

    def containment_prob(self, other: "NdarrayGumbelBox") -> float:
        """Compute containment probability P(other inside self).

        Args:
            other: Another Gumbel box embedding.

        Returns:
            Probability in [0.0, 1.0].
        """
        ...

    def overlap_prob(self, other: "NdarrayGumbelBox") -> float:
        """Compute overlap probability between this box and another.

        Args:
            other: Another Gumbel box embedding.

        Returns:
            Overlap probability in [0.0, 1.0].
        """
        ...

    def dim(self) -> int:
        """Return the number of dimensions.

        Returns:
            Dimensionality of the box.
        """
        ...

    def min_array(self) -> npt.NDArray[np.float32]:
        """Return min bounds as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

    def max_array(self) -> npt.NDArray[np.float32]:
        """Return max bounds as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

    def temperature(self) -> float:
        """Get the temperature parameter.

        Returns:
            Softplus temperature value.
        """
        ...

    def membership_probability(self, point: List[float]) -> float:
        """Compute the membership probability for a point.

        Args:
            point: Coordinates of the point (length must equal dim).

        Returns:
            Membership probability in [0.0, 1.0].
        """
        ...

class NdarrayCone:
    """Cone embedding for ConE-style knowledge graph reasoning."""

    def __init__(self, axes: List[float], apertures: Union[float, List[float]]) -> None:
        """Create a cone embedding from axis angles and apertures.

        Args:
            axes: Axis angle for each dimension.
            apertures: Per-dimension aperture values, or a single float
                broadcast to all dimensions.

        Raises:
            ValueError: If axes and apertures have incompatible lengths.
        """
        ...

    def dim(self) -> int:
        """Return the number of dimensions.

        Returns:
            Dimensionality of the cone.
        """
        ...

    def axes(self) -> npt.NDArray[np.float32]:
        """Return axis angles as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

    def apertures(self) -> npt.NDArray[np.float32]:
        """Return apertures as a numpy array.

        Returns:
            1-D numpy array of shape (dim,).
        """
        ...

    def cone_distance(self, other: "NdarrayCone", cen: float) -> float:
        """Compute the ConE distance between self (query) and other (entity).

        Args:
            other: Another cone embedding (entity cone).
            cen: ConE distance parameter.

        Returns:
            Distance value (lower means closer relation).
        """
        ...

class TrainingConfig:
    """Training configuration for box embeddings.

    All fields have defaults. Pass only the fields you want to change::

        config = subsumer.TrainingConfig(dim=32, learning_rate=0.01, epochs=200)
    """

    dim: int
    learning_rate: float
    epochs: int
    batch_size: int
    margin: float
    negative_samples: int
    negative_weight: float
    regularization: float
    gumbel_beta: float
    gumbel_beta_final: float
    warmup_epochs: int
    early_stopping_patience: Optional[int]
    symmetric_loss: bool

    def __init__(
        self,
        dim: int,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 512,
        margin: float = 1.0,
        negative_samples: int = 1,
        negative_weight: float = 1.0,
        regularization: float = 0.0001,
        gumbel_beta: float = 10.0,
        gumbel_beta_final: float = 50.0,
        warmup_epochs: int = 10,
        early_stopping_patience: Optional[int] = 10,
        symmetric_loss: bool = False,
    ) -> None:
        """Create a training configuration.

        Args:
            dim: Embedding dimension.
            learning_rate: Optimizer step size. Default: 0.001.
            epochs: Number of training epochs. Default: 100.
            batch_size: Triples per batch. Default: 512.
            margin: Negative margin for contrastive loss. Default: 1.0.
            negative_samples: Negatives per positive triple. Default: 1.
            negative_weight: Weight on the negative loss term. Default: 1.0.
            regularization: Volume regularization weight. Default: 0.0001.
            gumbel_beta: Softplus steepness at start. Default: 10.0.
            gumbel_beta_final: Softplus steepness after annealing. Default: 50.0.
            warmup_epochs: Learning rate warmup epochs. Default: 10.
            early_stopping_patience: Stop after N epochs without improvement.
                Default: 10. None to disable.
            symmetric_loss: Use symmetric min(P(A|B), P(B|A)) loss.
                Default: False.
        """
        ...

class BoxEmbeddingTrainer:
    """Trainer for Gumbel box embeddings on knowledge graph triples.

    Triple convention: (head_id, relation_id, tail_id) where the head box
    is trained to contain the tail box. For hypernym relations like "is-a",
    use (parent_id, rel_id, child_id).

    Standard KG files often list (child, hypernym, parent), which is the
    reverse. Use ``from_triples(..., reverse=True)`` or
    ``load_dataset(..., reverse=True)`` to swap head/tail automatically.
    """

    entity_names: Optional[List[str]]
    """Entity names indexed by ID, if loaded via from_triples or load_dataset."""

    def __init__(self, learning_rate: float, dim: int) -> None:
        """Create a trainer with default config and the given learning rate and dimension.

        Args:
            learning_rate: Optimizer step size.
            dim: Embedding dimension.
        """
        ...

    @staticmethod
    def from_config(config: TrainingConfig) -> "BoxEmbeddingTrainer":
        """Create a trainer from a TrainingConfig.

        Args:
            config: Training configuration with dim and hyperparameters.

        Returns:
            A new BoxEmbeddingTrainer.
        """
        ...

    @staticmethod
    def from_triples(
        triples: List[Tuple[str, str, str]],
        config: Optional[TrainingConfig] = None,
        reverse: bool = False,
    ) -> Tuple["BoxEmbeddingTrainer", List[Tuple[int, int, int]]]:
        """Create a trainer from string triples, handling interning automatically.

        Args:
            triples: List of (head, relation, tail) string triples.
            config: Training configuration. Uses defaults (dim=16) if None.
            reverse: If True, swap head and tail. Use for
                (child, hypernym, parent) data.

        Returns:
            Tuple of (trainer, train_triple_ids) where train_triple_ids is
            a list of (head_id, relation_id, tail_id) integer tuples.
        """
        ...

    def train_step(self, triples: List[Tuple[int, int, int]]) -> float:
        """Run one training step over the given triples.

        Each triple is (head_id, relation_id, tail_id) where the head box
        should contain the tail box.

        Args:
            triples: List of integer triple IDs.

        Returns:
            Average loss for this step.
        """
        ...

    def fit(
        self,
        train_triples: List[Tuple[int, int, int]],
        val_triples: Optional[List[Tuple[int, int, int]]] = None,
        num_entities: Optional[int] = None,
    ) -> Dict[str, object]:
        """Train for multiple epochs with optional validation and early stopping.

        Args:
            train_triples: List of (head_id, relation_id, tail_id) tuples.
            val_triples: Optional validation triples for MRR tracking.
            num_entities: Total entity count. Required if val_triples provided.
                Auto-computed as max(entity_id) + 1 if None.

        Returns:
            Dict with keys: mrr, hits_at_1, hits_at_3, hits_at_10,
            mean_rank, loss_history, validation_mrr_history, best_epoch.
        """
        ...

    def evaluate(
        self,
        test_triples: List[Tuple[int, int, int]],
        num_entities: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on test triples.

        Args:
            test_triples: List of (head_id, relation_id, tail_id) tuples.
            num_entities: Total entity count. Auto-computed if None.

        Returns:
            Dict with keys: mrr, head_mrr, tail_mrr, hits_at_1,
            hits_at_3, hits_at_10, mean_rank.
        """
        ...

    def get_box(self, entity_id: int) -> Optional[NdarrayBox]:
        """Get the learned box for an entity.

        Args:
            entity_id: Integer entity ID.

        Returns:
            NdarrayBox if the entity has a learned embedding, None otherwise.
        """
        ...

    def export_embeddings(
        self,
    ) -> Tuple[List[int], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Export all learned embeddings as numpy arrays.

        Returns:
            Tuple of (entity_ids, min_bounds, max_bounds) where
            min_bounds and max_bounds have shape (n_entities, dim).
        """
        ...

    def entity_name(self, entity_id: int) -> Optional[str]:
        """Get entity name for an ID.

        Only available if the trainer was created via from_triples or
        load_dataset.

        Args:
            entity_id: Integer entity ID.

        Returns:
            Entity name string, or None if not available.
        """
        ...

    def save_checkpoint(self) -> str:
        """Serialize trainer state to JSON.

        Includes entity and relation names if available.

        Returns:
            JSON string of the trainer checkpoint.
        """
        ...

    @staticmethod
    def load_checkpoint(json: str) -> "BoxEmbeddingTrainer":
        """Deserialize trainer state from JSON.

        Accepts both the envelope format (with entity_names) and the
        legacy format (bare trainer JSON).

        Args:
            json: JSON string from save_checkpoint.

        Returns:
            Restored BoxEmbeddingTrainer.
        """
        ...

class ConeEmbeddingTrainer:
    """Trainer for ConE cone embeddings on knowledge graph triples.

    Same triple convention as BoxEmbeddingTrainer: (head_id, relation_id, tail_id)
    where the head cone is trained to contain the tail cone. Each entity is
    represented as a cone (axis angles + apertures) instead of a box.
    Cones are better suited for DAG/partial-order relationships.
    """

    entity_names: Optional[List[str]]
    """Entity names indexed by ID, if loaded via from_triples."""

    def __init__(self, learning_rate: float, dim: int) -> None:
        """Create a trainer with default config and the given learning rate and dimension.

        Args:
            learning_rate: Optimizer step size.
            dim: Embedding dimension.
        """
        ...

    @staticmethod
    def from_config(config: TrainingConfig) -> "ConeEmbeddingTrainer":
        """Create a trainer from a TrainingConfig.

        Args:
            config: Training configuration with dim and hyperparameters.

        Returns:
            A new ConeEmbeddingTrainer.
        """
        ...

    @staticmethod
    def from_triples(
        triples: List[Tuple[str, str, str]],
        config: Optional[TrainingConfig] = None,
        reverse: bool = False,
    ) -> Tuple["ConeEmbeddingTrainer", List[Tuple[int, int, int]]]:
        """Create a trainer from string triples, handling interning automatically.

        Args:
            triples: List of (head, relation, tail) string triples.
            config: Training configuration. Uses defaults (dim=16) if None.
            reverse: If True, swap head and tail. Use for
                (child, hypernym, parent) data.

        Returns:
            Tuple of (trainer, train_triple_ids) where train_triple_ids is
            a list of (head_id, relation_id, tail_id) integer tuples.
        """
        ...

    def train_step(self, triples: List[Tuple[int, int, int]]) -> float:
        """Run one training step over the given triples.

        Args:
            triples: List of integer triple IDs.

        Returns:
            Average loss for this step.
        """
        ...

    def fit(
        self,
        train_triples: List[Tuple[int, int, int]],
        val_triples: Optional[List[Tuple[int, int, int]]] = None,
        num_entities: Optional[int] = None,
    ) -> Dict[str, object]:
        """Train for multiple epochs with optional validation and early stopping.

        Args:
            train_triples: List of (head_id, relation_id, tail_id) tuples.
            val_triples: Optional validation triples for MRR tracking.
            num_entities: Total entity count. Required if val_triples provided.
                Auto-computed as max(entity_id) + 1 if None.

        Returns:
            Dict with keys: mrr, hits_at_1, hits_at_3, hits_at_10,
            mean_rank, loss_history, validation_mrr_history, best_epoch.
        """
        ...

    def export_embeddings(
        self,
    ) -> Tuple[List[int], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Export all cone embeddings as numpy arrays.

        Returns:
            Tuple of (entity_ids, axes, apertures) where
            axes and apertures have shape (n_entities, dim).
        """
        ...

    def entity_name(self, entity_id: int) -> Optional[str]:
        """Get entity name for an ID.

        Only available if the trainer was created via from_triples.

        Args:
            entity_id: Integer entity ID.

        Returns:
            Entity name string, or None if not available.
        """
        ...

    def save_checkpoint(self) -> str:
        """Serialize trainer state to JSON.

        Includes entity and relation names if available.

        Returns:
            JSON string of the trainer checkpoint.
        """
        ...

    @staticmethod
    def load_checkpoint(json: str) -> "ConeEmbeddingTrainer":
        """Deserialize trainer state from JSON.

        Accepts both the envelope format (with entity_names) and the
        legacy format (bare trainer JSON).

        Args:
            json: JSON string from save_checkpoint.

        Returns:
            Restored ConeEmbeddingTrainer.
        """
        ...

def containment_probability(
    min_a: List[float],
    max_a: List[float],
    min_b: List[float],
    max_b: List[float],
    temperature: float,
) -> float:
    """Compute containment probability between two boxes from flat coordinate vectors.

    P(box_b inside box_a) using the given temperature.

    Args:
        min_a: Lower bounds of box A.
        max_a: Upper bounds of box A.
        min_b: Lower bounds of box B.
        max_b: Upper bounds of box B.
        temperature: Softplus temperature.

    Returns:
        Containment probability in [0.0, 1.0].
    """
    ...

def load_dataset(
    path: str,
    reverse: bool = False,
) -> Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]],
    List[str],
    List[str],
]:
    """Load a KG dataset from a directory with train.txt, valid.txt, test.txt.

    Each file has one triple per line, tab- or space-separated:
    ``head  relation  tail``.

    Args:
        path: Directory containing the dataset files.
        reverse: If True, swap head and tail in all triples. Use when
            data has (child, hypernym, parent) ordering but you want the
            trainer to learn parent-contains-child.

    Returns:
        Tuple of (train_triples, val_triples, test_triples, entity_names,
        relation_names). Triples are lists of (head_id, relation_id, tail_id)
        integer tuples. entity_names and relation_names are indexed by ID.
    """
    ...
