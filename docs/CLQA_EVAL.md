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
LEARNED_EPOCHS=20 QUERIES=50 scripts/run_clqa_eval.sh learned
METRICS_CSV=target/clqa-eval/galen-full.csv scripts/run_clqa_eval.sh full
```

The CSV includes dataset, model, hyperparameter, retrieval, and conformal rows.
It is the stable artifact for comparing runs; trained embedding export remains a
caller-owned concern. `BoxEmbeddingTrainer::export_embeddings()` exposes raw
values, but the crate does not yet define a multi-file embedding manifest.

Learned-ranker controls are forwarded with the `LEARNED_*` prefix:
`LEARNED_EXTRA_HOPS`, `LEARNED_EPOCHS`, `LEARNED_LR`, `LEARNED_L2`,
`LEARNED_REPEATS`, `LEARNED_SPLIT_SEED`, and `LEARNED_CASE_LIMIT`.

## Data

The example reads `data/GALEN/train.tsv`. To reproduce the benchmark input,
convert the GALEN prediction files from the
[Box²EL v1.0.0 bundle](https://github.com/KRR-Oxford/BoxSquaredEL/tree/v1.0.0):

```sh
mkdir -p data
curl --fail --location \
  https://raw.githubusercontent.com/KRR-Oxford/BoxSquaredEL/v1.0.0/data.zip \
  --output data/boxsquaredel-v1.0.0.zip
unzip data/boxsquaredel-v1.0.0.zip 'data/GALEN/prediction/*' \
  -d data/boxsquaredel-v1.0.0
uv run scripts/convert_box2el.py \
  data/boxsquaredel-v1.0.0/data/GALEN/prediction data/GALEN
```

The archive's SHA-256 is
`fcdfe52ac6ea84777ab513341bf4f255f0cde0e000b30ade5c47352aedc42ba8`.
The converted training file contains 67,562 axioms. The converter also writes
validation and test files; this query diagnostic builds its ontology and query
splits from the training file alone.

## Learned answer sets

The learned ranker fits on half the queries. The remaining queries are split
into calibration and test sets using the same seeded ordering. Retrieval rules,
ranker weights and the score definition stay fixed during calibration.

A query is **supported** when its candidate pool contains at least one target
deepest common ancestor. Only supported calibration queries contribute a score.
The three score choices are the gap from the best candidate score, that gap
divided by the candidate score range, and the target's zero-based rank. Missing
targets are retrieval failures; they are not assigned an artificial score.

The report separates:

- **Pool recall:** the fraction of test queries with a target in the pool.
- **Conditional coverage:** the fraction of supported test queries whose answer
  set contains a target. It is undefined when no test query is supported.
- **End-to-end coverage:** the fraction of all test queries whose answer set
  contains a target. It equals pool recall times conditional coverage when the
  latter is defined, because every answer set is a subset of its pool.

All three events mean finding at least one target, not every common ancestor.
The calibration count is the number of supported calibration queries. If none
are supported, the example returns the full candidate pool and labels the
result an uncalibrated fallback. A finite-sample rank beyond that count also
returns the full pool; with nonempty calibration data this is a conservative
conformal threshold. Neither case can recover a target absent from the pool.

The report counts empty pools and nonempty pools missing a target separately.
Invalid scores stop that calibration run. Repeated-run coverage summaries include
both calibrated sets and full-pool fallbacks, with counts for each. Threshold
means and standard deviations use only finite thresholds; the CSV records
unbounded thresholds with a flag instead of an infinite numeric value.

Split conformal coverage applies within the supported population only if its
calibration and future scores are exchangeable. Support depends on the unknown
target, so it is an evaluation condition, not information available for routing
a new query. Queries drawn from one fixed ontology provide an empirical
diagnostic; a seeded split does not establish deployment exchangeability.
Candidate-retrieval studies likewise need an explicit assumption about target
inclusion; see the [molecular retrieval setup, §2.1](https://www.biorxiv.org/content/10.64898/2026.03.12.711424v1.full#sec-3).

Boundary:

- `statskit` owns scalar conformal rank selection; `heyting` provides query
  adapters, and this example constructs sets for its rank and gap scores.
- `subsume` owns region geometry, Burn training, and CLQA candidate/readout code.
- `tranz` supplies the point-embedding baseline for comparison.
- Query-layer truth algebras and conformal sets live in `heyting`; `subsume`
  no longer exports the legacy fuzzy/cone-query helpers.
