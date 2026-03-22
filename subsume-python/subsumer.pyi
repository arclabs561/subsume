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

class BoxEmbedding:
    """Axis-aligned box embedding with softplus-based containment."""

    def __init__(self, min: List[float], max: List[float], temperature: float) -> None: ...
    def volume(self) -> float: ...
    def containment_prob(self, other: "BoxEmbedding") -> float: ...
    def dim(self) -> int: ...
    def min_array(self) -> npt.NDArray[np.float32]: ...
    def max_array(self) -> npt.NDArray[np.float32]: ...
    def __repr__(self) -> str: ...

class GumbelBoxEmbedding:
    """Gumbel box embedding with softplus-based intersection volumes."""

    def __init__(self, min: List[float], max: List[float], temperature: float) -> None: ...
    def volume(self) -> float: ...
    def containment_prob(self, other: "GumbelBoxEmbedding") -> float: ...
    def overlap_prob(self, other: "GumbelBoxEmbedding") -> float: ...
    def dim(self) -> int: ...
    def min_array(self) -> npt.NDArray[np.float32]: ...
    def max_array(self) -> npt.NDArray[np.float32]: ...
    def temperature(self) -> float: ...
    def membership_probability(self, point: List[float]) -> float: ...
    def __repr__(self) -> str: ...

class ConeEmbedding:
    """Cone embedding for ConE-style knowledge graph reasoning."""

    def __init__(self, axes: List[float], apertures: Union[float, List[float]]) -> None: ...
    def dim(self) -> int: ...
    def axes(self) -> npt.NDArray[np.float32]: ...
    def apertures(self) -> npt.NDArray[np.float32]: ...
    def cone_distance(self, other: "ConeEmbedding", cen: float) -> float: ...
    def __repr__(self) -> str: ...

class TrainingConfig:
    """Training configuration for box/cone embeddings.

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
    ) -> None: ...
    def __repr__(self) -> str: ...

class BoxEmbeddingTrainer:
    """Trainer for Gumbel box embeddings on knowledge graph triples.

    Triple convention: (head_id, relation_id, tail_id) where the head box
    is trained to contain the tail box.
    """

    entity_names: Optional[List[str]]
    relation_names: Optional[List[str]]

    def __init__(self, learning_rate: float, dim: int) -> None: ...

    @staticmethod
    def from_config(config: TrainingConfig) -> "BoxEmbeddingTrainer": ...

    @staticmethod
    def from_triples(
        triples: List[Tuple[str, str, str]],
        config: Optional[TrainingConfig] = None,
        reverse: bool = False,
    ) -> Tuple["BoxEmbeddingTrainer", List[Tuple[int, int, int]]]: ...

    def train_step(self, triples: List[Tuple[int, int, int]]) -> float: ...

    def fit(
        self,
        train_triples: List[Tuple[int, int, int]],
        val_triples: Optional[List[Tuple[int, int, int]]] = None,
        num_entities: Optional[int] = None,
    ) -> Dict[str, object]: ...

    def evaluate(
        self,
        test_triples: List[Tuple[int, int, int]],
        num_entities: Optional[int] = None,
    ) -> Dict[str, float]: ...

    def predict(self, head_id: int, relation_id: int, tail_id: int) -> float:
        """Score a single (head, relation, tail) triple.

        Returns containment probability P(tail inside head).
        """
        ...

    def score_tails(self, head_id: int, relation_id: int) -> npt.NDArray[np.float32]:
        """Score all entities as tails for a (head, relation) query.

        Returns 1-D array of containment probabilities.
        """
        ...

    def get_box(self, entity_id: int) -> Optional[BoxEmbedding]: ...

    def export_embeddings(
        self,
    ) -> Tuple[List[int], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

    def entity_name(self, entity_id: int) -> Optional[str]: ...
    def save_checkpoint(self) -> str: ...

    @staticmethod
    def load_checkpoint(json: str) -> "BoxEmbeddingTrainer": ...

    def __repr__(self) -> str: ...

class ConeEmbeddingTrainer:
    """Trainer for ConE cone embeddings on knowledge graph triples."""

    entity_names: Optional[List[str]]
    relation_names: Optional[List[str]]

    def __init__(self, learning_rate: float, dim: int) -> None: ...

    @staticmethod
    def from_config(config: TrainingConfig) -> "ConeEmbeddingTrainer": ...

    @staticmethod
    def from_triples(
        triples: List[Tuple[str, str, str]],
        config: Optional[TrainingConfig] = None,
        reverse: bool = False,
    ) -> Tuple["ConeEmbeddingTrainer", List[Tuple[int, int, int]]]: ...

    def train_step(self, triples: List[Tuple[int, int, int]]) -> float: ...

    def fit(
        self,
        train_triples: List[Tuple[int, int, int]],
        val_triples: Optional[List[Tuple[int, int, int]]] = None,
        num_entities: Optional[int] = None,
    ) -> Dict[str, object]: ...

    def evaluate(
        self,
        test_triples: List[Tuple[int, int, int]],
        num_entities: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on test triples.

        Returns dict with mrr, head_mrr, tail_mrr, hits_at_1/3/10, mean_rank.
        """
        ...

    def export_embeddings(
        self,
    ) -> Tuple[List[int], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

    def entity_name(self, entity_id: int) -> Optional[str]: ...
    def save_checkpoint(self) -> str: ...

    @staticmethod
    def load_checkpoint(json: str) -> "ConeEmbeddingTrainer": ...

    def __repr__(self) -> str: ...

def containment_probability(
    min_a: List[float],
    max_a: List[float],
    min_b: List[float],
    max_b: List[float],
    temperature: float,
) -> float:
    """Compute containment probability between two boxes from flat coordinate vectors."""
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
    """Load a KG dataset from a directory with train.txt, valid.txt, test.txt."""
    ...

def load_el_axioms(path: str) -> Dict[str, object]:
    """Load EL++ normalized axioms from a TSV file.

    Returns a dict with keys: nf1, nf2, nf3, nf4, ri6, ri7, disj,
    num_classes, num_roles, total_axioms.
    """
    ...

def el_inclusion_loss(
    center_a: List[float],
    offset_a: List[float],
    center_b: List[float],
    offset_b: List[float],
    margin: float = 0.0,
) -> float:
    """Compute EL++ inclusion loss: how much box A fails to be contained in box B."""
    ...

def el_intersection_loss(
    center_c1: List[float],
    offset_c1: List[float],
    center_c2: List[float],
    offset_c2: List[float],
    center_d: List[float],
    offset_d: List[float],
    margin: float = 0.0,
) -> float:
    """Compute NF1 intersection loss: C1 AND C2 should be contained in D."""
    ...
