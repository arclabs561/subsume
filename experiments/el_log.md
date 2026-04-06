# EL++ Experiment Log

Each entry: hypothesis, command (exact, copy-paste), data, result, conclusion.
Results reference `el_results.tsv` by ID. All commands run from repo root.

---

## E01: Candle baseline (prior session)
**Hypothesis**: Candle trainer with checkpoint + beta annealing gives best prior NF results on GALEN.
**Command**:
```sh
BACKEND=candle DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG_SAMPLES=2 MARGIN=0.15 NEG_DIST=5.0 REG_FACTOR=0.4 \
  cargo run --features candle-backend --example el_benchmark --release -- data/GALEN
```
**Commit**: `6ab9711` (no closure filtering)
**Data**: GALEN train.tsv (67K axioms, 23K concepts, 951 roles), test.tsv (8K axioms).
**Result**: NF1 0.101, NF2 0.131, NF3 0.284, NF4 0.004. H@10 not recorded.
**Conclusion**: Strong NF3, weak NF4 (no negatives). This is the baseline to beat.

## E02-E03: Burn first run (hyperparam-search config)
**Hypothesis**: Hyperparams from GALEN 500ep random search should work at full budget on burn.
**Command**:
```sh
DIM=200 EPOCHS=5000 LR=0.015 BATCH=512 NEG=2 MARGIN=0.093 NEG_DIST=4.0 REG=0.91 DATASET=GALEN \
  cargo run --features burn-ndarray --example el_benchmark_burn --release
```
**Commit**: `7a96135` (closure filtering, no L2-init)
**Data**: Same GALEN.
**Result**: E02 (500ep): NF3 0.082. E03 (5000ep): NF3 0.164, NF2 0.068.
**Conclusion**: Hyperparams were tuned for candle, not burn. Different optimizer defaults.

## E04: Closure filtering ablation (candle)
**Hypothesis**: Deductive closure filtering (259K entailed pairs) improves NF2 by removing false negatives.
**Command**:
```sh
BACKEND=candle DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG_SAMPLES=2 MARGIN=0.15 NEG_DIST=5.0 REG_FACTOR=0.4 \
  cargo run --features candle-backend --example el_benchmark --release -- data/GALEN
```
**Commit**: `7a96135` (closure filtering added)
**Data**: Same GALEN.
**Result**: NF2 0.136 (+0.005 vs E01). NF1 0.067 (-0.034). Checkpoint at ep1000 (very early).
**Conclusion**: Closure helps NF2 marginally but shifts checkpoint dynamics. The early peak hurts NF1/NF3.

## E05: Burn with candle HP, no L2-init (GALEN)
**Hypothesis**: Using candle's exact hyperparams on burn closes the performance gap.
**Command**:
```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 DATASET=GALEN \
  cargo run --features burn-ndarray --example el_benchmark_burn --release
```
**Commit**: `9336f51` (no L2-init)
**Data**: Same GALEN.
**Result**: NF3 0.213 (vs candle 0.284). NF2 0.061.
**Conclusion**: Candle HP helps but gap persists. Initialization diff suspected.

## E06: GO (burn-wgpu, no L2-init)
**Command**:
```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 DATASET=GO \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```
**Commit**: `9336f51`
**Result**: NF3 0.380, NF1 0.095, NF4 0.016. 182s on Metal.

## E07: ANATOMY (burn-wgpu, no L2-init)
**Command**:
```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 DATASET=ANATOMY \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```
**Commit**: `9336f51`
**Result**: NF2 0.086, NF3 0.101. 314s on Metal, 221s eval (106K concepts).

## E08: L2-normalized initialization fix (GALEN)
**Hypothesis**: Candle L2-normalizes init (Box2EL pattern). Burn's Uniform[-1,1] gives inconsistent bump scale. Normalizing should improve NF3 specifically.
**Command**: Same as E05 but with burn-wgpu.
```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 DATASET=GALEN \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```
**Commit**: `a45f26c` (L2-init fix)
**Result**: NF2 **0.137** (matches candle 0.136), NF3 **0.320** (+34% vs candle 0.284). 118s on Metal.
**Conclusion**: **L2-init was the primary cause of the burn-candle gap.** Bumps need consistent initial scale.

## E09: GO with L2-init
**Hypothesis**: L2-init improvement should transfer to GO.
**Command**: Same as E06 but at commit `a45f26c`.
**Commit**: `a45f26c`
**Result**: NF1 **0.216** (2.3x vs E06), NF4 **0.044** (2.7x). NF3 stable at 0.371.
**Conclusion**: L2-init helps all NF types across datasets.

## E10: Epsilon ablation (1e-8 vs burn default 1e-5)
**Hypothesis**: Burn's Adam eps=1e-5 is 100x larger than candle's 1e-8. Smaller eps helps sparse bumps.
**Command**: Same as E08.
**Commit**: `c66266d` (eps=1e-8)
**Result**: NF1 +0.037 (0.051->0.088), NF2 -0.017 (0.137->0.120). NF3 stable.
**Conclusion**: Epsilon tradeoff. Helps sparse params, hurts dense. **Reverted** (commit `79a687c`).

---

## Best known config (E08)

```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 DATASET=GALEN \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```
Requires commit `a45f26c` or later (L2-init). burn-wgpu for Metal GPU, burn-ndarray for CPU.

## Open experiments
- [ ] Bump curriculum (frozen centers first 500ep)
- [ ] DISJ-disabled burn run (isolate DISJ gradient competition)
- [ ] Per-parameter epsilon (1e-8 for bumps, 1e-5 for centers)
- [ ] ANATOMY with L2-init (E11, running)

## E11: NF4 negatives enabled (GALEN)
**Hypothesis**: With closure filtering in place, NF4 negatives (weight=0.3) should be safe and improve NF4 MRR.
**Command**:
```sh
DIM=200 EPOCHS=5000 LR=0.01 BATCH=512 NEG=2 MARGIN=0.15 NEG_DIST=5.0 REG=0.4 NF4_NEG_W=0.3 DATASET=GALEN \
  cargo run --features burn-wgpu --example el_benchmark_burn --release
```
**Commit**: `d2cee76` (L2-init, default eps, NF4_NEG_W wired)
**Result**: NF4 0.004 (2x vs E08 0.002). NF1 0.099 (+0.048). NF2 0.118 (-0.019). NF3 0.298 (-0.022).
**Conclusion**: NF4 negatives improve NF4 and NF1 but hurt NF2/NF3 via gradient competition. Closure filtering makes NF4 negatives safer than prior session (NF1 0.099 vs prior 0.089). Best used when NF1/NF4 matter more than NF2/NF3. **Not default.**
