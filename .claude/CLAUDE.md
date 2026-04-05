# subsume — Project Alignment & Memory
# Last updated: 2026-04-05 (session 3)

## What This Project Is

**subsume** = geometric region embeddings where containment = subsumption.
Entities are regions (boxes, balls, cones, ellipsoids, subspaces, sectors, etc.), not points. If region A contains region B, then A subsumes B (A is more general).

**tranz** (sibling crate at `../tranz`) = point embeddings (TransE, RotatE, ComplEx, DistMult). For link prediction. Different task, different crate.

subsume handles: ontology completion, taxonomy expansion, logical query answering.
tranz handles: link prediction, relation extraction, knowledge base completion.

---

## Architecture Decisions

### Backend: Burn is the goal, Candle is the placeholder

- **Current**: ndarray (CPU inference) + candle (GPU training) for old geometry types (boxes, cones)
- **New geometry types**: All use backend-agnostic `Vec<f32>` for inference; burn for training
- **Why burn**: 14.7k stars, wgpu (AMD/Intel/WebGPU) + CUDA + ndarray via feature flags, one implementation compiles anywhere. Candle is Meta's inference-only project, wrong fit for research.
- **Migration strategy**: Don't rewrite old code. New trainers use burn. Old trainers stay on candle.
- **Burn 0.20 API quirks** (already solved in burn_ball_trainer.rs):
  - `NdArray` not `Ndarray` (camelCase)
  - `AdamConfig::new().init::<B, M>()` returns `OptimizerAdaptor<Adam, M, B>`
  - `GradientsParams::from_grads(grads, &model)` before `optim.step(lr as f64, model, grads)`
  - `optim.step(...)` returns `M` (not `(M, State)`)
  - `select(0, ids)` consumes the tensor — clone first if needed again
  - Repeat-interleave needs index tensor: build `repeat_ids` with `(0..bs).flat_map(|i| iter::repeat(i).take(n_neg))`
  - Feature flags: `burn-ndarray`, `burn-wgpu`, `burn-tch` in Cargo.toml

### Design Philosophy

- **Implement published methods first, invent second.** Novel contributions: spherical caps, full-covariance ellipsoids. Everything else is faithful paper implementation.
- **Backend-agnostic first.** New geometry types: `Vec<f32>` params. No feature flags.
- **Every trainer needs eval integration tests.** See critical bugs below.
- **No candle implementations for new types.** Burn only.
- **Shared infrastructure**: `trainer/trainer_utils.rs` has `AdamState` (persistent across epochs) and `self_adversarial_weights`. Use these instead of per-trainer HashMap state.
- **sigmoid**: Use `crate::utils::stable_sigmoid`. Do NOT add local copies.

---

## Critical Bugs Found This Session (never repeat)

### 1. Loss sign flip (found in all 6 trainers)
`margin - pos_score + neg_score` is **wrong**. Correct: `margin + pos_score - neg_score`.
- This minimizes when pos_prob is LARGE and neg_prob is SMALL — the correct direction.
- Was invisible to unit tests but caught by `train_and_evaluate_synthetic`.

### 2. Adam state reset each epoch
All trainers were creating `HashMap::new()` inside `train_epoch`. This loses momentum every epoch.
- Fixed: `adam: AdamState` field on each trainer struct, initialized in `new()`, persists across calls.

### 3. Burn trainer loss formula doubled pos_loss
`pos_loss + ranking_loss` where `ranking_loss` already contained `pos_loss`. Correct: just `ranking_loss`.

### 4. rotate_vector wrong in non-3D (spherical_cap.rs)
The generalized rotation formula for d≠3 was computing wrong `w` vector. Fixed with proper Gram-Schmidt.
- 3D: use exact Rodrigues formula `v*cos(a) + (axis×v)*sin(a) + axis*(axis·v)*(1-cos(a))`
- General d: decompose into parallel + perpendicular, find w orthogonal to v_perp_hat via Gram-Schmidt

### 5. Subspace intersection missed linear combinations
Single-vector projection threshold `> 0.999` only finds basis vectors that are directly in B, not linear combinations. Fixed with alternating projection (10 iterations of P_B * P_A).

### 6. Annular midpoint wraparound
`(theta_start + theta_end) / 2` fails near 0/2π boundary. Fixed with circular mean: `atan2(sin(s)+sin(e), cos(s)+cos(e))`.

### 7. Ellipsoid Cholesky near-zero division
`solve_lower` divided by diagonal without guarding zero. Fixed with `if diag.abs() < 1e-8 { x[i] = 0.0 }`.

---

## Complete Geometry Inventory (705 tests)

### Pre-existing (9 types, unchanged)

| Geometry | File | Training |
|---|---|---|
| Hard Boxes | `ndarray_backend/ndarray_box.rs` | ✅ CPU+GPU |
| Gumbel Boxes | `ndarray_backend/ndarray_gumbel.rs` | ✅ CPU+GPU |
| Cones | `ndarray_backend/ndarray_cone.rs` | ✅ CPU+GPU |
| Octagons | `ndarray_backend/ndarray_octagon.rs` | ❌ inference only |
| Gaussian Boxes (TaxoBell) | `gaussian.rs` | ✅ MLP encoder |
| Density Matrices | `density.rs` | ✅ density_el |
| Hyperbolic | `hyperbolic.rs` | ❌ |
| Spherical | `spherical.rs` | ❌ |
| Sheaf Networks | `sheaf.rs` | ❌ |

### New this session (6 geometry types + 7 trainers)

| Geometry | File | Tests | Notes |
|---|---|---|---|
| **Ball** (SpherE + RegD) | `ball.rs` | 56 | d+1 params, `‖cA-cB‖+rA ≤ rB` |
| **Spherical Cap** | `spherical_cap.rs` | 43 | NOVEL: `angle(cA,cB)+θA ≤ θB` |
| **Subspace** | `subspace.rs` | 31 | Conjunction/disjunction/negation native |
| **Ellipsoid** | `ellipsoid.rs` | 27 | NOVEL: full-covariance via Cholesky |
| **TransBox** | `transbox.rs` | 31 | EL++-closed, Yang et al. 2024 |
| **Annular Sectors** | `annular.rs` | 28 | Zhu & Zeng 2025, rotation + uncertainty |
| **Ball Trainer** | `trainer/ball_trainer.rs` | 8 | Analytical grads, multi-neg, self-adversarial |
| **SphericalCap Trainer** | `trainer/spherical_cap_trainer.rs` | 6 | Analytical grads, multi-neg |
| **Subspace Trainer** | `trainer/subspace_trainer.rs` | 6 | Finite-diff grads, multi-neg |
| **Ellipsoid Trainer** | `trainer/ellipsoid_trainer.rs` | 6 | Finite-diff grads, multi-neg |
| **TransBox Trainer** | `trainer/transbox_trainer.rs` | 6 | Finite-diff grads, multi-neg |
| **Annular Trainer** | `trainer/annular_trainer.rs` | 6 | Finite-diff grads, multi-neg |
| **Burn Ball Trainer** | `trainer/burn_ball_trainer.rs` | 5 | Batched tensor ops + autodiff (burn 0.20) |
| **Trainer Utils** | `trainer/trainer_utils.rs` | 7 | `AdamState`, `self_adversarial_weights` |

### Shared infrastructure added

- `trainer/trainer_utils.rs`: `AdamState` (persistent m/v/step across epochs), `self_adversarial_weights` (RotatE-style)
- `trainer/evaluation.rs`: Added `evaluate_link_prediction_generic` — works with any scoring function, used by all 6 new trainers
- `crate::utils::stable_sigmoid`: Was already there. Removed 3 private copies from ball/cap/ellipsoid modules.

---

## Benchmark Results

### Geometry comparison: WordNet subset (30 triples, 32 entities, 100 epochs, dim=32)
Run: `cargo run --example geometry_comparison --release`

| Geometry | MRR | H@10 | MeanRank |
|---|---|---|---|
| **AnnularSector** | **0.745** | **1.000** | **2.0** |
| TransBox | 0.687 | 0.867 | 4.6 |
| Ball | 0.404 | 0.950 | 3.7 |
| Ellipsoid | 0.401 | 0.983 | 3.0 |
| SphericalCap | 0.336 | 0.950 | 4.4 |
| Subspace | 0.105 | 0.333 | 16.0 |

Random baseline: MRR ≈ 0.031. All geometries well above random.

### WN18RR: Ball trainer (40K entities, 86K triples)
Run: `DIM=32 EPOCHS=50 LR=0.01 NEG=10 K=2 cargo run --example wn18rr_ball --release`

- Per-triple SGD: MRR 0.049, H@10 0.111, MR 9222 (dim=32, 50 epochs)
- Published SpherE target: MRR 0.453 (SIGIR 2024)
- Per-triple SGD is too slow for WN18RR at scale (>15 min/epoch with neg=20)

### WN18RR: Burn batch trainer (session 3 results)
Run: `DIM=200 EPOCHS=300 LR=0.005 BATCH=512 NEG=20 ADV_TEMP=1.0 INFONCE=1 cargo run --features burn-ndarray --example wn18rr_ball_burn --release`

| Config | MRR | H@10 | MR |
|---|---|---|---|
| dim=32, 50ep, no InfoNCE | 0.049 | 0.111 | 9222 |
| dim=200, 100ep, InfoNCE | 0.136 | 0.409 | 5245 |
| dim=200, 300ep, InfoNCE+adv | **0.148** | **0.427** | **4642** |
| SpherE target (point embeddings) | 0.453 | 0.568 | 2779 |

- H@10 is 75% of SpherE target; MRR gap is architectural (containment scoring vs point distance)
- Per-relation: rel 1 MRR=0.32/H@10=0.90; rel 0 (_hypernym, 40% of test) drags at MRR=0.03
- Blog post confirms: geometric methods retain advantage in ontology settings, not general link prediction
- **WN18RR is not subsume's benchmark** — EL++ ontology completion is the competitive niche

---

## What Needs to Happen Next (prioritized)

### 1. ~~Type-constrained negative sampling~~ — DONE (session 3)
Implemented in `negative_sampling.rs`. Bernoulli + per-relation entity pools. Integrated into ball_trainer and burn_ball_trainer. Improved MRR from 0.050 to 0.148 (with InfoNCE + higher dim).

### 2. Migrate remaining trainers to burn (MEDIUM — architecture goal)
Pattern established in `burn_ball_trainer.rs` with `Param<Tensor>` model, dual translations, type-constrained neg sampling. Next: `burn_spherical_cap_trainer.rs`, then others. Each should have a batched `compute_batch_loss` like the ball trainer.

### 3. WN18RR for other geometries (LOW — not practical with finite-diff)
Per-triple SGD with finite-diff gradients is too slow for WN18RR (86K triples, 40K entities). Only ball (analytical gradients) and burn ball (autodiff) are practical. Other geometries need burn migration first.

### 4. EL++ benchmark improvements (MEDIUM — competitive niche)
subsume wins NF3 (existential) on GALEN/ANATOMY. Improving NF1/NF2 and matching Box2EL on GO would strengthen the story. Focus here rather than WN18RR link prediction.

### 5. Subspace analytical gradients (LOW)
Stiefel manifold Riemannian gradient needed for practical Subspace training. Research task.

---

## Testing Conventions

Every trainer must have:
1. `train_and_evaluate_synthetic` test: trains on hierarchy, verifies `MRR > 0.3` and `mean_rank <= 3.0`
2. `gradients_are_finite` test: checks all gradient fields are finite
3. `train_epoch_runs` test: basic smoke test that loss is finite

Tolerances: 1e-4 for unit tests, 1e-3 for proptests (f32 accumulation).

Full suite: `cargo test --lib` = 707 tests (715 with `--features burn-ndarray`).

---

## Trainer Pattern

```rust
// 1. Add adam: AdamState to struct
pub struct MyTrainer { rng: StdRng, adam: AdamState }

// 2. init_embeddings uses rng
// 3. train_epoch:
//    - Shuffle triples
//    - For each triple: sample n_neg = config.negative_samples negatives
//    - weights = self_adversarial_weights(&neg_scores, config.adversarial_temperature)
//    - weighted gradient accumulation
//    - (bias1, bias2) = self.adam.tick()
//    - self.adam.apply(key, &mut param, grad, lr, bias1, bias2)
// 4. evaluate() delegates to evaluate_link_prediction_generic

// 5. train_and_evaluate_synthetic test:
//    - 4 entities, 2 relations, hierarchy structure
//    - 50 epochs, lr=0.05
//    - assert MRR > 0.3, mean_rank <= 3.0
```

---

## Data

- `data/WN18RR/{train,valid,test}.txt` — full WN18RR (not git-tracked, `data/` in .gitignore)
- `data/go_subset/go_normalized.tsv` — small GO EL++ subset (committed, 134 lines)
- `data/GALEN/`, `data/GO/`, `data/ANATOMY/` — EL++ benchmark datasets (not committed)

---

## Shared Infra Files

| File | What |
|---|---|
| `trainer/trainer_utils.rs` | `AdamState`, `self_adversarial_weights` |
| `trainer/negative_sampling.rs` | `RelationCardinality`, `RelationEntityPools`, `compute_relation_entity_pools`, `sample_excluding` |
| `trainer/evaluation.rs` | `evaluate_link_prediction_generic`, `FilteredTripleIndexIds` |
| `trainer/mod.rs` | All re-exports, `CpuBoxTrainingConfig` (has margin, neg_samples, adv_temp, batch_size, lr, sigmoid_k, bernoulli_sampling) |
| `utils.rs` | `stable_sigmoid`, `softplus`, `stable_logsumexp` — use these, don't copy |
| `dataset.rs` | `Triple`, `TripleIds`, `Vocab`, `load_dataset` (via lattix) |
| `metrics.rs` | `mean_reciprocal_rank`, `hits_at_k`, `mean_rank` (via lattix) |

---

## Key Examples

| Example | What |
|---|---|
| `geometry_comparison` | All 6 new geometries on WordNet subset, comparison table (FAST: ~0.4s) |
| `wn18rr_ball` | Ball trainer on WN18RR, per-triple SGD, env vars: DIM/EPOCHS/LR/NEG/ADV_TEMP/MARGIN |
| `wn18rr_ball_burn` | Burn batched ball trainer on WN18RR, env vars: DIM/EPOCHS/LR/NEG/BATCH/MARGIN |
| `wn18rr_training` | Box trainer on WN18RR (reference, CPU) |
| `wn18rr_candle` | Box trainer on WN18RR (reference, GPU) |
| `el_benchmark` | EL++ (GALEN/GO/Anatomy) box2el benchmark |
| `dataset_training` | Self-contained WordNet subset with BoxEmbeddingTrainer |

---

## Sibling Crate: tranz

Point embeddings at `../tranz`. Uses `candle` + `lattix::kge`. Link prediction, not subsumption. Shared dataset format (TSV triples). Different enough that there's no code overlap.
