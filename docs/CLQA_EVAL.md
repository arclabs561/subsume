# CLQA Evaluation

`subsume::clqa` contains the reusable candidate-pool and box-readout pieces.
The full real-ontology comparison stays in the data-gated `el_clqa_galen`
example because it depends on dataset files, Heyting conformal utilities, and a
Tranz point-embedding baseline.

Run the standard harness:

```sh
scripts/run_clqa_eval.sh box
```

Modes:

| Mode | What it runs |
|---|---|
| `box` | Trains the Burn box model and skips the TransE baseline. |
| `full` | Trains the Burn box model and the Tranz TransE baseline. |
| `symbolic` | Runs direct-frontier retrieval diagnostics without model training. |
| `learned` | Runs direct-frontier retrieval plus the learned graph-feature ranker. |

The script defaults to `BACKEND=wgpu` and writes
`target/clqa-eval/GALEN-<mode>-wgpu.csv`. Set `BACKEND=ndarray` for the CPU
fallback. The example exits successfully with a message when the dataset is not
present.

Common overrides:

```sh
DATASET=GALEN DIM=200 EPOCHS=500 QUERIES=300 scripts/run_clqa_eval.sh box
BACKEND=ndarray scripts/run_clqa_eval.sh symbolic
METRICS_CSV=target/clqa-eval/galen-full.csv scripts/run_clqa_eval.sh full
```

The CSV includes dataset, model, hyperparameter, retrieval, and conformal rows.
It is the stable artifact for comparing runs; trained embedding export remains a
caller-owned concern. `BoxEmbeddingTrainer::export_embeddings()` exposes raw
values, but the crate does not yet define a multi-file embedding manifest.

Boundary:

- `heyting` owns conformal answer-set construction.
- `subsume` owns region geometry, Burn training, and CLQA candidate/readout code.
- `tranz` supplies the point-embedding baseline for comparison.
- Deprecated in-crate fuzzy/cone-query helpers remain compatibility shims until
  the next breaking release.
