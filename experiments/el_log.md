# EL++ Experiment Log

Each entry has: hypothesis, method, result, conclusion.
Results reference `el_results.tsv` by ID (E01, E02, ...).

---

## E01: Candle baseline (prior session)
**Hypothesis**: Candle trainer with checkpoint + beta annealing gives best prior NF results on GALEN.
**Method**: `BACKEND=candle ... cargo run --features candle-backend --example el_benchmark --release -- data/GALEN`
**Data**: GALEN train (67K axioms, 23K concepts, 951 roles), test (8K axioms).
**Result**: NF1 0.101, NF2 0.131, NF3 0.284, NF4 0.004. H@10 not recorded.
**Conclusion**: Strong NF3, weak NF4 (no negatives). This is the baseline to beat.

## E02-E03: Burn first run (hyperparam-search config)
**Hypothesis**: Hyperparams from GALEN 500ep random search (lr=0.015, margin=0.093, neg_dist=4.0, reg=0.91) should work at full budget.
**Method**: `cargo run --features burn-ndarray --example el_benchmark_burn --release`
**Result**: E03 (5000ep): NF3 0.164, NF2 0.068. Well below candle.
**Conclusion**: Hyperparams were tuned for candle, not burn. Need candle's original HP for apples-to-apples.

## E04: Closure filtering ablation (candle)
**Hypothesis**: Deductive closure filtering (259K entailed pairs) improves NF2 by removing false negatives.
**Method**: Same as E01 but with closure filtering enabled (commit 7a96135).
**Result**: NF2 0.136 (+0.005 vs E01). NF1 0.067 (-0.034), NF3 0.238 (-0.046). Checkpoint at ep1000 (much earlier than E01).
**Conclusion**: Closure helps NF2 marginally but shifts checkpoint dynamics. The early checkpoint hurts NF1/NF3 which need more epochs to develop.

## E05-E07: Burn with candle HP (no L2-init)
**Hypothesis**: Using candle's exact hyperparams on burn closes the gap.
**Method**: Candle HP (lr=0.01, margin=0.15, neg_dist=5.0, reg=0.4) on burn-ndarray and burn-wgpu.
**Result**: E05 GALEN: NF3 0.213 (vs candle 0.284). E06 GO: NF3 0.380. E07 ANATOMY: NF3 0.101.
**Conclusion**: Candle HP helps (NF3 0.213 vs 0.164) but gap persists. Initialization or optimizer diff suspected.

## E08: L2-normalized initialization fix
**Hypothesis**: Candle L2-normalizes init (Box2EL pattern). Burn's Uniform[-1,1] gives inconsistent bump scale. Normalizing should improve NF3.
**Method**: L2-normalize each embedding row at init (commit a45f26c). Same HP as E05.
**Result**: NF2 0.137 (matches candle E04 0.136), NF3 **0.320** (exceeds candle 0.284 by 34%).
**Conclusion**: **L2-init was the primary cause of the burn-candle gap.** Bumps need consistent initial scale for the existential encoding mechanism. This is the most impactful single fix.

## E09: GO with L2-init
**Hypothesis**: L2-init fix should also improve GO results.
**Result**: NF1 0.216 (was 0.095, 2.3x), NF4 0.044 (was 0.016, 2.7x). NF3 stable at 0.371.
**Conclusion**: L2-init helps all NF types across datasets. GO benefits most for NF1 (more intersection axioms in GO's structure).

## E10: Epsilon ablation (1e-8 vs 1e-5)
**Hypothesis**: Burn's default eps=1e-5 is 100x larger than candle's 1e-8. Smaller epsilon should help sparse bump updates (NF3/NF1).
**Method**: Added `.with_epsilon(1e-8)` to AdamConfig (commit c66266d).
**Result**: NF1 +0.037 (0.051->0.088), NF2 -0.017 (0.137->0.120). NF3 stable.
**Conclusion**: Epsilon tradeoff: helps sparse params (NF1 bumps) but destabilizes dense params (NF2 centers). **Reverted** -- net NF2+NF3 better with default 1e-5.

---

## Best known config (as of E08)

```
trainer: burn-wgpu (Metal)
init: L2-normalized (unit sphere)
optimizer: Adam (burn default, eps=1e-5, beta1=0.9, beta2=0.999)
HP: lr=0.01, margin=0.15, neg_dist=5.0, reg=0.4, batch=512, neg=2
epochs: 5000 with validation checkpoint (every 500ep, best NF2+NF3)
closure: GCI0 transitive (BFS)
```

## Open experiments
- [ ] Bump curriculum (frozen centers first 500ep)
- [ ] DISJ-disabled burn run (isolate DISJ gradient competition)
- [ ] Per-parameter epsilon (1e-8 for bumps, 1e-5 for centers)
- [ ] ANATOMY with L2-init (running)
