# subsume

[![crates.io](https://img.shields.io/crates/v/subsume.svg)](https://crates.io/crates/subsume)
[![Documentation](https://docs.rs/subsume/badge.svg)](https://docs.rs/subsume)
[![CI](https://github.com/arclabs561/subsume/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/subsume/actions/workflows/ci.yml)

Geometric region embeddings in Rust for entailment and set containment:
boxes, cones, octagons, Gaussians, hyperbolic intervals, and sheaves.

![Box embedding concepts](docs/box_concepts.png)

*(a) Containment: nested boxes encode taxonomic is-a relationships. (b) Gumbel soft boundary: temperature controls membership sharpness. (c) Octagon: diagonal constraints cut corners for tighter volume bounds.*

For background on why region embeddings work and how the geometries relate, see [Why Regions, Not Points](https://attobop.net/posts/region-embeddings/).

## Geometries

15 geometry types: boxes (hard and Gumbel), cones, octagons, Gaussians (TaxoBell),
hyperbolic intervals, spherical, density matrices, sheaf networks, balls, spherical
caps, subspaces, ellipsoids, TransBox, and annular sectors. Each implements
containment probability, volume, intersection, and distance. CPU backend via
ndarray, GPU via candle (`features = ["candle-backend"]` or `["cuda"]`) or burn
(`features = ["burn-ndarray"]` or `["burn-wgpu"]`).

Scoring: Query2Box distance, fuzzy t-norms for logical queries, EL++ ontology
losses (Box2EL/TransBox). Training: CPU trainer with analytical gradients or
GPU trainer with AdamW autograd. Evaluation: MRR, Hits@k, Mean Rank (filtered).

## Usage

```toml
[dependencies]
subsume = { version = "0.12", features = ["ndarray-backend"] }
ndarray = "0.16"
```

```rust
use subsume::ndarray_backend::NdarrayBox;
use ndarray::array;

// Box A: [0,0,0] to [1,1,1] (general concept)
let premise = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;

// Box B: [0.2,0.2,0.2] to [0.8,0.8,0.8] (specific, inside A)
let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

// Containment probability: P(B inside A)
let p = premise.containment_prob(&hypothesis)?;
assert!(p > 0.9);
```

### Training (Rust)

```rust,ignore
use subsume::{BoxEmbeddingTrainer, TrainingConfig, Dataset};
use subsume::dataset::load_dataset;
use std::path::Path;

let dataset = load_dataset(Path::new("data/wn18rr"))?;
let interned = dataset.into_interned();
let train: Vec<_> = interned.train.iter().map(|t| (t.head, t.relation, t.tail)).collect();

let config = TrainingConfig { learning_rate: 0.01, epochs: 50, ..Default::default() };
let mut trainer = BoxEmbeddingTrainer::new(config, 32);
let result = trainer.fit(&train, None, None)?;
println!("MRR: {:.3}", result.final_results.mrr);
```

### Training (GPU via candle)

```rust,ignore
use subsume::CandleBoxTrainer;
use candle_core::Device;

let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
let trainer = CandleBoxTrainer::new(num_entities, num_relations, 200, 10.0, &device)?
    .with_inside_weight(0.02)   // BoxE-style center attraction
    .with_vol_reg(0.0001);      // prevent trivial solution

let losses = trainer.fit(&train_triples, 500, 0.001, 512, 9.0, 128, 1.0)?;
let (mrr, h1, h3, h10, mr) = trainer.evaluate(&test_triples, &all_triples)?;
```

### Training (Python)

```bash
pip install subsumer
```

```python
import subsumer

triples = [("animal", "hypernym", "dog"), ("animal", "hypernym", "cat"), ...]
config = subsumer.TrainingConfig(dim=32, epochs=50, learning_rate=0.01)
trainer, ids = subsumer.BoxEmbeddingTrainer.from_triples(triples, config=config)
result = trainer.fit(ids)
print(f"MRR: {result['mrr']:.3f}")
```

Triple convention: head box **contains** tail box. For datasets where triples
are `(child, hypernym, parent)`, pass `reverse=True` to `from_triples` or
`load_dataset`.

## Examples

```bash
cargo run --example box_training             # train box embeddings on a 25-entity taxonomy
cargo run --example dataset_training --release # full pipeline: WN18RR data, train, evaluate
cargo run --example el_training              # EL++ ontology embedding
```

27 examples total covering all geometries, training modes, and query types.
See [`examples/README.md`](examples/README.md) for the full list.

## Tests

```bash
cargo test -p subsume
```

Unit, property, and doc tests covering:

- Box geometry: intersection, union, containment, overlap, distance, volume, truncation
- Gumbel boxes: membership probability, temperature edge cases, Bessel volume
- Cones: angular containment, negation closure, aperture bounds
- Octagon: intersection closure, containment, Sutherland-Hodgman volume
- Fuzzy: t-norm/t-conorm commutativity, associativity, De Morgan duality
- Gaussian boxes, EL++ ontology losses, sheaf networks, hyperbolic geometry
- Training: MRR, Hits@k, Mean Rank, negative sampling (uniform, Bernoulli), AMSGrad

## Choosing a geometry

| Geometry | When to use it | ¬? | Key tradeoff |
|---|---|---|---|
| NdarrayBox / NdarrayGumbelBox | Containment hierarchies, each dimension independent | No | Simple, fast; Gumbel adds dense gradients where hard boxes have zero gradient |
| Cone | Multi-hop queries requiring negation (FOL with ¬) | Yes | Closed under complement; angular parameterization harder to initialize |
| Octagon | Rule-aware KG completion; tighter containment than boxes | No | Diagonal constraints cut box corners; more parameters per entity |
| Gaussian | Taxonomy expansion with uncertainty (TaxoBell) | No | KL = asymmetric containment; Bhattacharyya = symmetric overlap |
| Hyperbolic | Tree-like hierarchies with exponential branching | No | Low-dim capacity; numerical care near Poincare ball boundary |
| Ball | Spherical containment (SpherE + RegD); GPU via burn | No | Simpler than boxes; fewer parameters; analytical gradients available |
| SphericalCap | Directional containment; cone angle as uncertainty | No | Novel parameterization; may need more epochs than boxes |
| Subspace | Conjunction/disjunction/negation via projection | Yes | Closed under set ops; finite-diff gradients slow at high dim |
| Ellipsoid | Full-covariance containment via Cholesky; anisotropic | No | Expressiveness at cost of O(d²) parameters; numerical care near degenerate shapes |
| TransBox | EL++-closed ontology embedding (Yang et al., 2024) | No | Designed for DL semantics; requires box parameterization |
| AnnularSector | Angular position + spread; rotation uncertainty (Zhu & Zeng, 2025) | No | Best empirical MRR on small hierarchies; novel parameterization |

## Why regions instead of points?

Point embeddings (TransE, RotatE, ComplEx) represent entities as vectors. They work
well for link prediction -- RotatE hits 0.476 MRR on WN18RR, BoxE hits 0.451.
For standard triple scoring, points are simpler and equally accurate.

Regions become necessary when the task requires structure that points cannot encode.
The core operation is **containment probability**:

$$P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}$$

If B fits inside A, $P = 1$. If disjoint, $P = 0$. This is the scoring
function used for evaluation (`containment_prob`).

| What you need | Points | Regions |
|---|---|---|
| Containment (A ⊆ B) | No -- points have no interior | Box nesting = subsumption |
| Volume = generality | No -- points have no size | Large box = broad concept |
| Intersection (A ∧ B) | No set operations | Box ∩ Box = another box |
| Negation (¬A) | No complement | Cone complement = another cone |
| Uncertainty per dimension | No | Gaussian sigma |

Three tasks where point embeddings structurally fail:

1. **Ontology completion (EL++)**: "Dog is-a Animal" requires representing one concept's
   extension as a subset of another's. Points have no containment. Box2EL, TransBox, and
   DELE use boxes for this and outperform point baselines on Gene Ontology, GALEN, and
   Anatomy.

2. **Logical query answering (∧, ∨, ¬)**: multi-hop KG queries with conjunction,
   disjunction, and negation need set operations. ConE handles all three (MRR 52.9 on
   FB15k EPFO+negation queries vs Query2Box's 41.0 and BetaE's 44.6). Points cannot
   attempt negation queries at all.

3. **Taxonomy expansion**: inserting a new concept at the right depth requires knowing
   both what it is (similarity) and how general it is (volume). TaxoBell uses Gaussian
   boxes where KL divergence gives asymmetric parent-child containment for free.

If your task is link prediction or entity similarity, use RotatE. If you need
containment, set operations, or volume, you need regions.

On general knowledge graphs without explicit DL semantics, foundation models now
match or exceed geometric methods. The geometric advantage is strongest in
ontology-specific settings where containment and DL axioms are the ground truth.

See [`docs/SUBSUMPTION_HISTORY.md`](docs/SUBSUMPTION_HISTORY.md) for the research
history of geometric subsumption embeddings, from hard boxes through Gumbel, cones, and beyond.

## Why Gumbel boxes?

![Gumbel gradient landscape](docs/gumbel_robustness.png)

*(a) Membership probability at a box boundary: hard boxes have a discontinuous step, Gumbel boxes have smooth sigmoids controlled by temperature. (b) Gradient magnitude: hard boxes produce zero gradient everywhere except the exact boundary (gray regions), while Gumbel boxes provide gradients throughout the space.*

Gumbel boxes model coordinates as Gumbel random variables, creating soft boundaries
that provide dense gradients throughout training. Hard boxes create flat regions where
gradients vanish; Gumbel boxes solve this *local identifiability problem*
(Dasgupta et al., 2020). Lower temperature (small beta) gives crisper boundaries with
sharper gradients; higher temperature gives broader gradients that reach further from
the boundary but sacrifice containment precision.

## Training convergence

![Training convergence](docs/training_convergence.png)

*25-entity taxonomy learned over 300 epochs. Left: total violation drops 3 orders of magnitude. Right: containment probabilities converge to 1.0 at different rates depending on hierarchy depth. Reproduce: `cargo run --example box_training` or `uv run scripts/plot_training.py`.*

## Embedding export

`BoxEmbeddingTrainer::export_embeddings()` returns flat f32 vectors suitable for
safetensors, numpy (via reshape), and vector databases:

```rust,ignore
let (ids, mins, maxs) = trainer.export_embeddings();
// mins/maxs are flat Vec<f32> of length n_entities * dim
// Reshape to (n_entities, dim) for numpy/safetensors
```

Checkpoint save/load via serde:

```rust,ignore
let json = serde_json::to_string(&trainer)?;
let restored: BoxEmbeddingTrainer = serde_json::from_str(&json)?;
```

## Integration patterns

Convert from petgraph (when `petgraph` feature is enabled):

```rust,ignore
use subsume::petgraph_adapter::from_graph;
let dataset = from_graph(&my_digraph);
```

Convert from polars (no dependency needed, user-side code):

```rust,ignore
use subsume::dataset::Triple;
let triples: Vec<Triple> = df.column("head")?.str()?
    .into_iter()
    .zip(df.column("relation")?.str()?)
    .zip(df.column("tail")?.str()?)
    .filter_map(|((h, r), t)| Some(Triple::new(h?, r?, t?)))
    .collect();
let dataset = Dataset::new(triples, vec![], vec![]);
```

## EL++ ontology completion benchmarks

Per-normal-form results on Box2EL benchmark datasets (Jackermeier et al., 2023),
evaluated by center L2 distance ranking (matching Box2EL protocol).
subsume: dim=200, 5000 epochs, single run, default hyperparameters.
Box2EL/TransBox: 5000 epochs, best of 10 runs (Table 7, TransBox WWW 2025).

| Dataset | NF type | subsume MRR | subsume H@1 | subsume H@10 |
|---|---|---|---|---|
| GALEN (23K) | NF1: C1 ⊓ C2 ⊑ D | 0.051 | 0.015 | 0.096 |
| GALEN | NF2: C ⊑ D | **0.137** | 0.039 | **0.335** |
| GALEN | NF3: C ⊑ ∃r.D | **0.320** | **0.229** | **0.476** |
| GALEN | NF4: ∃r.C ⊑ D | 0.002 | 0.001 | 0.002 |
| GO (46K) | NF1 | **0.216** | **0.124** | **0.392** |
| GO | NF2 | 0.061 | 0.024 | 0.130 |
| GO | NF3 | **0.371** | **0.292** | **0.507** |
| GO | NF4 | 0.044 | 0.002 | 0.161 |
| ANATOMY (106K) | NF1 | 0.066 | 0.047 | 0.100 |
| ANATOMY | NF2 | 0.093 | 0.055 | 0.160 |
| ANATOMY | NF3 | **0.208** | **0.154** | **0.311** |
| ANATOMY | NF4 | 0.000 | 0.000 | 0.000 |

NF3 (existential restrictions) is consistently the strongest result, with
MRR 0.21-0.37 across all three datasets. GO NF1 (conjunction) reaches MRR
0.216 using Gumbel soft intersection with beta annealing.

Key techniques:
- Gumbel soft intersection for NF1 with beta annealing (0.3 -> 2.0)
- Center attraction fallback (weight 0.5) for degenerate intersections
- Box2EL-style bump translations and dual-direction NF3 negative sampling
- GCI0 deductive closure filtering for negative sampling (DELE, Mashkova et al. 2024)
- L2-normalized embedding initialization (consistent bump scale)
- Cosine LR with 10% floor, validation-based checkpointing
- Disjointness (DISJ) training loss

Reproduce (burn backend, Metal GPU):
```sh
DIM=200 EPOCHS=5000 DATASET=GALEN \
  cargo run --features "burn-ndarray,burn-wgpu" --example el_benchmark_burn --release
```

Reproduce (candle backend, CPU/CUDA):
```sh
BACKEND=candle EPOCHS=5000 \
  cargo run --features candle-backend --example el_benchmark --release -- data/GALEN
```

Full experiment log with all ablations: `experiments/el_log.md`.

## GPU training

The `CandleBoxTrainer` supports CPU, CUDA, and Metal via the candle backend:

```bash
# CPU
cargo run --features candle-backend --example wn18rr_candle --release

# CUDA GPU
cargo run --features cuda --example wn18rr_candle --release
```

Configure via environment variables:

```bash
DIM=200 EPOCHS=500 LR=0.001 NEG=128 BATCH=512 MARGIN=9.0 \
  ADV_TEMP=1.0 INSIDE_W=0.02 VOL_REG=0.0001 BOUNDS_EVERY=50 \
  cargo run --features cuda --example wn18rr_candle --release
```

The burn ball trainer runs on ndarray (CPU) or wgpu (GPU/AMD/WebGPU):

```bash
# CPU via burn-ndarray
DIM=64 EPOCHS=300 LR=0.01 BATCH=512 NEG=10 \
  cargo run --features burn-ndarray --example wn18rr_ball_burn --release
```

See `examples/README.md` for all available examples.

## References

- Nickel & Kiela (2017). "Poincare Embeddings for Learning Hierarchical Representations"
- Vilnis et al. (2018). "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Li et al. (2019). "Smoothing the Geometry of Probabilistic Box Embeddings" (ICLR 2019)
- Sun et al. (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (self-adversarial negative sampling)
- Abboud et al. (2020). "BoxE: A Box Embedding Model for Knowledge Base Completion"
- Dasgupta et al. (2020). "Improving Local Identifiability in Probabilistic Box Embeddings"
- Ren et al. (2020). "Query2Box: Reasoning over Knowledge Graphs using Box Embeddings"
- Hansen & Ghrist (2019). "Toward a Spectral Theory of Cellular Sheaves"
- Bodnar et al. (2022). "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"
- Boratko et al. (2021). "Box Embeddings: An open-source library for representation learning using geometric structures" (EMNLP Demo)
- Chen et al. (2021). "Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning" (BEUrRE, ACL 2021)
- Gebhart, Hansen & Schrater (2021). "Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding"
- Zhang et al. (2021). "ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs"
- Chen et al. (2022). "Fuzzy Logic Based Logical Query Answering on Knowledge Graphs"
- Jackermeier et al. (2023). "Dual Box Embeddings for the Description Logic EL++"
- Yang, Chen & Sattler (2024). "TransBox: EL++-closed Ontology Embedding"
- Bourgaux et al. (2024). "Knowledge Base Embeddings: Semantics and Theoretical Properties" (KR 2024)
- Charpenay & Schockaert (2024). "Capturing Knowledge Graphs and Rules with Octagon Embeddings"
- Lacerda et al. (2024). "Strong Faithfulness for ELH Ontology Embeddings" (TGDK 2024)
- Huang et al. (2023). "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs"
- Mashkova et al. (2024). "DELE: Deductive EL++ Embeddings for Knowledge Base Completion"
- Yang & Chen (2025). "Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"
- Mishra et al. (2026). "TaxoBell: Gaussian Box Embeddings for Self-Supervised Taxonomy Expansion" (WWW '26)

## License

MIT OR Apache-2.0
