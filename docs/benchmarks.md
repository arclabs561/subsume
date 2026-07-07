# Benchmarks

This page records benchmark claims that are too detailed for the README.

## EL++ Ontology Completion

Per-normal-form results on Box2EL benchmark datasets (Jackermeier et al., 2023),
evaluated by center L2 distance ranking to match the Box2EL protocol.

- `subsume`: dim=200, 5000 epochs, single run, default hyperparameters.
- Box2EL/TransBox: 5000 epochs, best of 10 runs from TransBox WWW 2025, Table 7.

| Dataset | NF type | subsume MRR | subsume H@1 | subsume H@10 |
| --- | --- | --- | --- | --- |
| GALEN (23K) | NF1: C1 and C2 subset D | 0.051 | 0.015 | 0.096 |
| GALEN | NF2: C subset D | 0.137 | 0.039 | 0.335 |
| GALEN | NF3: C subset exists r.D | 0.320 | 0.229 | 0.476 |
| GALEN | NF4: exists r.C subset D | 0.002 | 0.001 | 0.002 |
| GO (46K) | NF1 | 0.216 | 0.124 | 0.392 |
| GO | NF2 | 0.061 | 0.024 | 0.130 |
| GO | NF3 | 0.371 | 0.292 | 0.507 |
| GO | NF4 | 0.044 | 0.002 | 0.161 |
| ANATOMY (106K) | NF1 | 0.066 | 0.047 | 0.100 |
| ANATOMY | NF2 | 0.093 | 0.055 | 0.160 |
| ANATOMY | NF3 | 0.208 | 0.154 | 0.311 |
| ANATOMY | NF4 | 0.000 | 0.000 | 0.000 |

NF3 existential restrictions are the strongest recorded result, with MRR
0.21-0.37 across all three datasets. GO NF1 reaches MRR 0.216 using Gumbel soft
intersection with beta annealing.

Techniques used in the recorded run:

- Gumbel soft intersection for NF1 with beta annealing from 0.3 to 2.0.
- Center attraction fallback for degenerate intersections.
- Box2EL-style bump translations and dual-direction NF3 negative sampling.
- GCI0 deductive closure filtering for negative sampling.
- L2-normalized embedding initialization.
- Cosine learning rate with 10 percent floor and validation checkpointing.
- Disjointness training loss.

Reproduce with the Burn backend:

```sh
DIM=200 EPOCHS=5000 DATASET=GALEN \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```

## GPU Training

The Burn trainers run on ndarray CPU or wgpu GPU backends:

```bash
DIM=64 EPOCHS=300 LR=0.01 BATCH=512 NEG=10 \
  cargo run --features burn-ndarray --example wn18rr_ball_burn --release

DIM=64 EPOCHS=300 LR=0.01 BATCH=512 NEG=10 \
  cargo run --features burn-wgpu --example wn18rr_ball_burn --release
```

The full example map is in [`../examples/README.md`](../examples/README.md).
