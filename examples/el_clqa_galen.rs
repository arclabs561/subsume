//! Real-ontology faithful-vs-plain CLQA benchmark (data-gated eval).
//!
//! The el_clqa_benchmark synthetic taxonomy shows the mechanism; this runs the
//! same faithful-boxes-vs-plain-TransE head-to-head on a REAL ontology (GALEN
//! by default: ~24k concepts, a real multiple-inheritance DAG), making the
//! comparison a result rather than an illustration.
//!
//! Conjunctive queries "X such that A subset-of X AND B subset-of X" are sampled
//! from concept pairs sharing a non-trivial common ancestor in the deductive
//! closure; the certain answers are the common ancestors, and the deepest common
//! ancestors (the most specific shared superclasses) are the key targets.
//! Both models rank every concept by min(deg(A subset-of X), deg(B subset-of
//! X)); the box model uses containment, TransE uses its isa translation.
//!
//! FINDING (2026-07-04, diagnosed via the box diagnostics below): the clean
//! synthetic-tree advantage (el_clqa_benchmark, boxes +0.48 MRR) does NOT
//! transfer to real GALEN at this budget. TransE narrowly wins; the box top-1 is
//! never a valid common ancestor. Three layers were each ruled out by
//! measurement. First, offset blowup: one box (concept 92) grew to 24x mean size
//! and won 100% of queries; OFFSET_CLAMP caps box size (max 73 to 10) and gives
//! 6 distinct winners, but concept 92 still sits at the cap and wins 98%. Second,
//! metric saturation: containment degree is ~1 for ANY container, so the largest
//! box wins ties; a TIGHTNESS size penalty (additive, or the multiplicative
//! deg*exp(-lambda*size)) instead lets the SMALLEST box win. Tightness has a
//! sweet spot (lambda=0.02) that demotes the giant and lifts top1CA from 0.000
//! to 0.070, but larger lambda collapses to tiny boxes; 0.07 is the ceiling.
//! Third, the root cause is REPRESENTATIONAL, not a training budget: even
//! converged (loss 0.0185 at dim 200, 1500 epochs), concept 92 still blows up
//! and wins, because a general concept genuinely needs a large box to contain
//! its subclasses and a large box causes spurious geometric containments. The
//! synthetic tree nests exactly; GALEN's messy multiple-inheritance DAG does
//! not, and more compute cannot fix it (the loss already converges). So the
//! synthetic benchmark shows the transitivity MECHANISM; on real ontologies box
//! CONTAINMENT CLQA is fundamentally weak (top1CA ~0.07). BUT the fix is the
//! READOUT, not the geometry or training: a non-containment JOIN-MATCH score
//! (rank X by how closely Box(X) matches the join, the smallest enclosing box,
//! of A and B, which is the LCA's expected shape) escapes the widest-region-wins
//! degeneracy entirely and reaches top1CA 0.173, beating TransE's ~0.09. So the
//! box embeddings DO encode the LCA; containment was simply the wrong way to ask
//! for it. OFFSET_CLAMP and TIGHTNESS are diagnostic knobs (default off);
//! score_join is the working non-containment scorer.
//!
//! Data-gated: exits 0 with a message if the dataset is absent (fetch GALEN via
//! the Box2EL conversion the el_benchmark examples describe). Runs on burn Metal
//! (wgpu) via cfg; large batch amortizes wgpu dispatch overhead; the eval loop
//! is rayon-parallel. Run:
//! `DATASET=GALEN cargo run --release --features burn-ndarray,burn-wgpu --example el_clqa_galen`
//! Use `SYMBOLIC_ONLY=1` to run the direct-frontier retrieval diagnostics
//! without training the box model. Add `LEARNED_RETRIEVAL=1` to train a
//! deterministic graph-feature ranker over the direct-frontier candidate pool.
//! Set `LEARNED_REPEATS=N` to repeat the learned conformal split diagnostic.
//! Set `LEARNED_SPLIT_SEED=N` to change the deterministic learned split seed.
//! Learned conformal rows include score-gap, normalized score-gap, and
//! rank-threshold variants.
//! Set `LEARNED_CASE_LIMIT=N` to emit up to N missed and covered learned
//! conformal case rows per alpha when `METRICS_CSV` is set.
//! Set `METRICS_CSV=path/to/run.csv` to also write aggregate and learned-repeat
//! metrics as machine-readable CSV rows.

use heyting::conformal::{answer_set_from_degrees, calibrate_scores, ConformalThreshold};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use subsume::clqa::{BoxClqa, CandidatePoolMetrics, DirectFrontier, DirectFrontierQuery};
use subsume::el_dataset::load_el_axioms;
use subsume::el_training::{Axiom, Ontology};
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};
use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
use tranz::dataset::TripleIds;

#[path = "el_clqa_galen/learned.rs"]
mod learned;

#[cfg(feature = "burn-wgpu")]
type Backend = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
type Backend = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn box_degree(centers: &[f32], offsets: &[f32], a: usize, b: usize, dim: usize) -> f32 {
    let (ao, bo) = (a * dim, b * dim);
    let mut acc = 0.0f32;
    for i in 0..dim {
        let v = ((centers[ao + i] - centers[bo + i]).abs() + offsets[ao + i] - offsets[bo + i])
            .max(0.0);
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

fn transe_degree(emb: &[Vec<f32>], rel: &[f32], a: usize, b: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..rel.len() {
        let v = emb[a][i] + rel[i] - emb[b][i];
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

struct CQuery {
    a: usize,
    b: usize,
    common: HashSet<usize>,
    lcas: Vec<usize>,
}

type ScoreResult = ((f64, f64, f64), Vec<usize>, Vec<usize>);

struct MetricsCsv {
    writer: Option<BufWriter<File>>,
    dataset: String,
    dim: usize,
    epochs: usize,
    queries: usize,
}

impl MetricsCsv {
    fn from_env(dataset: &str, dim: usize, epochs: usize, queries: usize) -> std::io::Result<Self> {
        let writer = match std::env::var("METRICS_CSV") {
            Ok(path) => {
                let mut writer = BufWriter::new(File::create(path)?);
                writeln!(
                    writer,
                    "dataset,dim,epochs,queries,section,label,param,alpha,metric,value"
                )?;
                Some(writer)
            }
            Err(_) => None,
        };
        Ok(Self {
            writer,
            dataset: dataset.to_string(),
            dim,
            epochs,
            queries,
        })
    }

    fn emit(
        &mut self,
        section: &str,
        label: &str,
        param: &str,
        alpha: Option<f32>,
        metric: &str,
        value: impl std::fmt::Display,
    ) {
        let Some(writer) = &mut self.writer else {
            return;
        };
        let alpha = alpha.map_or_else(String::new, |a| format!("{a:.6}"));
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            csv_cell(&self.dataset),
            self.dim,
            self.epochs,
            self.queries,
            csv_cell(section),
            csv_cell(label),
            csv_cell(param),
            csv_cell(&alpha),
            csv_cell(metric),
            value
        )
        .expect("failed to write METRICS_CSV row");
    }

    fn enabled(&self) -> bool {
        self.writer.is_some()
    }
}

fn csv_cell(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn emit_score_result(
    metrics: &mut MetricsCsv,
    label: &str,
    param: &str,
    mrr: f64,
    hits10: f64,
    top1_dca: f64,
) {
    metrics.emit("model_score", label, param, None, "mrr", mrr);
    metrics.emit("model_score", label, param, None, "hits10", hits10);
    metrics.emit("model_score", label, param, None, "top1_dca", top1_dca);
}

struct RetrievalMetrics {
    recall: f64,
    mean_pool: f32,
    p90_pool: usize,
    mrr: f64,
    hits10: f64,
    top1_dca: f64,
}

fn emit_retrieval_metrics(
    metrics: &mut MetricsCsv,
    section: &str,
    label: &str,
    param: &str,
    values: RetrievalMetrics,
) {
    metrics.emit(section, label, param, None, "recall", values.recall);
    metrics.emit(section, label, param, None, "mean_pool", values.mean_pool);
    metrics.emit(section, label, param, None, "p90_pool", values.p90_pool);
    metrics.emit(section, label, param, None, "mrr", values.mrr);
    metrics.emit(section, label, param, None, "hits10", values.hits10);
    metrics.emit(section, label, param, None, "top1_dca", values.top1_dca);
}

struct ConformalMetrics {
    q_hat: f32,
    empirical: f64,
    pool_recall: f64,
    mean_set: f32,
    p50_set: usize,
    p90_set: usize,
    max_set: usize,
}

fn emit_conformal_metrics(
    metrics: &mut MetricsCsv,
    section: &str,
    label: &str,
    param: &str,
    alpha: f32,
    values: ConformalMetrics,
) {
    metrics.emit(section, label, param, Some(alpha), "q_hat", values.q_hat);
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "empirical_coverage",
        values.empirical,
    );
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "pool_recall",
        values.pool_recall,
    );
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "mean_set",
        values.mean_set,
    );
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "p50_set",
        values.p50_set,
    );
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "p90_set",
        values.p90_set,
    );
    metrics.emit(
        section,
        label,
        param,
        Some(alpha),
        "max_set",
        values.max_set,
    );
}

impl CQuery {
    fn is_target(&self, x: usize) -> bool {
        self.lcas.binary_search(&x).is_ok()
    }

    fn target_score(&self, degrees: &[f32]) -> f32 {
        self.lcas
            .iter()
            .map(|&x| degrees[x])
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

fn target_rank_pairs(q: &CQuery, scored: &[(usize, f32)]) -> usize {
    1 + scored
        .iter()
        .position(|&(x, _)| q.is_target(x))
        .expect("target deepest common ancestor is in the ranked candidate set")
}

fn target_rank_ids(q: &CQuery, ranked: &[usize]) -> usize {
    1 + ranked
        .iter()
        .position(|&x| q.is_target(x))
        .expect("target deepest common ancestor is in the ranked candidate set")
}

/// Metrics `(DCA MRR, DCA Hits@10, top-1-is-valid-CA)` plus the top-1 concept
/// and deepest-common-ancestor rank per query. Queries are independent, so the per-query ranking
/// (the GALEN-scale O(queries * n) cost) is rayon-parallel.
fn score_model<F: Fn(usize, usize) -> f32 + Sync>(
    queries: &[CQuery],
    deg: F,
    sizes: &[f32],
    lambda: f32,
    n: usize,
) -> ScoreResult {
    // Rank by min(deg(A,X), deg(B,X)) * exp(-lambda * size(X)). Containment degree
    // saturates near 1 for any container, so a size factor breaks the tie toward
    // the smallest common container (the LCA). Multiplicative (not additive):
    // a non-container (deg ~0) stays ~0 regardless of its size, so tiny leaves
    // that contain nothing cannot win. lambda = 0 = plain degree.
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| (x, deg(q.a, x).min(deg(q.b, x)) * (-lambda * sizes[x]).exp()))
                .collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let top1s: Vec<usize> = per.iter().map(|x| x.3).collect();
    let ranks: Vec<usize> = per.iter().map(|x| x.4).collect();
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        top1s,
        ranks,
    )
}

/// L1 size (half-perimeter) of concept `c`'s box.
fn offset_l1(offsets: &[f32], c: usize, dim: usize) -> f32 {
    offsets[c * dim..(c + 1) * dim]
        .iter()
        .map(|o| o.abs())
        .sum()
}

/// Smallest-container score: the LCA is the smallest box that CONTAINS the join
/// of A and B (it also holds its other subclasses, so it is bigger than the
/// join itself). Rank X by softContain(join subset-of Box(X)) * exp(-lambda *
/// size(X)) -- among concepts whose box contains the join, prefer the smallest.
/// More principled than score_join's proximity-to-join.
fn score_join_contain(
    queries: &[CQuery],
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    lambda: f32,
    n: usize,
) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let (ao, bo) = (q.a * dim, q.b * dim);
            let mut jc = vec![0f32; dim];
            let mut jo = vec![0f32; dim];
            for i in 0..dim {
                let lo = (centers[ao + i] - offsets[ao + i]).min(centers[bo + i] - offsets[bo + i]);
                let hi = (centers[ao + i] + offsets[ao + i]).max(centers[bo + i] + offsets[bo + i]);
                jc[i] = (lo + hi) / 2.0;
                jo[i] = (hi - lo) / 2.0;
            }
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| {
                    let xo = x * dim;
                    let mut incl = 0f32;
                    let mut size = 0f32;
                    for i in 0..dim {
                        // Inclusion of the join box inside Box(X).
                        let v =
                            ((jc[i] - centers[xo + i]).abs() + jo[i] - offsets[xo + i]).max(0.0);
                        incl += v * v;
                        size += offsets[xo + i].abs();
                    }
                    let deg = (-incl.sqrt()).exp();
                    (x, deg * (-lambda * size).exp())
                })
                .collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

/// Non-containment score: rank each concept X by how closely Box(X) matches the
/// JOIN (smallest enclosing box) of the two query concepts. The LCA is the
/// smallest common superclass, so its box should approximate that join; this
/// avoids the widest-region-wins degeneracy of containment scoring entirely.
fn score_join(
    queries: &[CQuery],
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    n: usize,
) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let (ao, bo) = (q.a * dim, q.b * dim);
            // Join of A and B: element-wise min lower corner, max upper corner.
            let mut jc = vec![0f32; dim];
            let mut jo = vec![0f32; dim];
            for i in 0..dim {
                let lo = (centers[ao + i] - offsets[ao + i]).min(centers[bo + i] - offsets[bo + i]);
                let hi = (centers[ao + i] + offsets[ao + i]).max(centers[bo + i] + offsets[bo + i]);
                jc[i] = (lo + hi) / 2.0;
                jo[i] = (hi - lo) / 2.0;
            }
            // Rank by negative L1 distance of Box(X) to the join box.
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| {
                    let xo = x * dim;
                    let mut d = 0f32;
                    for i in 0..dim {
                        d += (centers[xo + i] - jc[i]).abs() + (offsets[xo + i] - jo[i]).abs();
                    }
                    (x, -d)
                })
                .collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

/// Containment-gated proximity: among boxes that CONTAIN the join of A and B,
/// rank by closeness to it. This is the synthesis of the two half-right
/// readouts. score_join (proximity) is non-saturating but rewards close
/// non-containing siblings; score_join_contain (softContain * size) is
/// geometrically correct (the LCA must contain the join) but softContain
/// saturates -- any large box contains the join, so the size penalty, not the
/// data, breaks the tie. Here the gate exp(-||join sticks out of X||)
/// suppresses non-containers, and proximity exp(-prox / tau) tie-breaks among
/// the containers toward the tightest one, which is the LCA. The tightest
/// container of the join is the one closest to it, so proximity discriminates
/// exactly where containment degree is flat.
fn score_join_gated(
    queries: &[CQuery],
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    tau: f32,
    n: usize,
) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let (ao, bo) = (q.a * dim, q.b * dim);
            let mut jc = vec![0f32; dim];
            let mut jo = vec![0f32; dim];
            for i in 0..dim {
                let lo = (centers[ao + i] - offsets[ao + i]).min(centers[bo + i] - offsets[bo + i]);
                let hi = (centers[ao + i] + offsets[ao + i]).max(centers[bo + i] + offsets[bo + i]);
                jc[i] = (lo + hi) / 2.0;
                jo[i] = (hi - lo) / 2.0;
            }
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| {
                    let xo = x * dim;
                    let mut incl = 0f32; // join sticking out of Box(X): 0 = X contains join
                    let mut prox = 0f32; // L1 distance of Box(X) to the join box
                    for i in 0..dim {
                        let v =
                            ((jc[i] - centers[xo + i]).abs() + jo[i] - offsets[xo + i]).max(0.0);
                        incl += v * v;
                        prox += (centers[xo + i] - jc[i]).abs() + (offsets[xo + i] - jo[i]).abs();
                    }
                    let gate = (-incl.sqrt()).exp();
                    let proximity = (-prox / tau).exp();
                    (x, gate * proximity)
                })
                .collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

fn gated_lca_degrees(clqa: &BoxClqa<'_>, a: usize, b: usize, tau: f32) -> Vec<f32> {
    let (jc, jo) = clqa.join(a, b);
    (0..clqa.num_concepts())
        .map(|x| clqa.score_lca(&jc, &jo, x, tau))
        .collect()
}

fn quantile_f32(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn quantile_usize(values: &[usize], q: f32) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_f32_distribution(label: &str, values: &[f32]) {
    if values.is_empty() {
        println!("[diag] {label}: n=0");
        return;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    println!(
        "[diag] {label}: n={} mean={mean:.4} p50={:.4} p90={:.4} p95={:.4} p99={:.4} max={:.4}",
        values.len(),
        quantile_f32(values, 0.50),
        quantile_f32(values, 0.90),
        quantile_f32(values, 0.95),
        quantile_f32(values, 0.99),
        quantile_f32(values, 1.00)
    );
}

fn set_size_summary(sizes: &[usize]) -> (f32, usize, usize, usize) {
    if sizes.is_empty() {
        return (0.0, 0, 0, 0);
    }
    let mean = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
    (
        mean,
        quantile_usize(sizes, 0.50),
        quantile_usize(sizes, 0.90),
        *sizes.iter().max().unwrap(),
    )
}

fn print_rank_diagnostic(label: &str, ranks: &[usize]) {
    if ranks.is_empty() {
        println!("[rank] {label}: n=0");
        return;
    }
    let recall =
        |k: usize| ranks.iter().filter(|&&rank| rank <= k).count() as f32 / ranks.len() as f32;
    println!(
        "[rank] {label}: n={} r@10={:.3} r@100={:.3} r@1000={:.3} r@5000={:.3} p50={} p90={} p95={} p99={} max={}",
        ranks.len(),
        recall(10),
        recall(100),
        recall(1000),
        recall(5000),
        quantile_usize(ranks, 0.50),
        quantile_usize(ranks, 0.90),
        quantile_usize(ranks, 0.95),
        quantile_usize(ranks, 0.99),
        quantile_usize(ranks, 1.00)
    );
}

fn top_k_by_score<F: Fn(usize) -> f32>(
    n: usize,
    a: usize,
    b: usize,
    k: usize,
    score: F,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = (0..n)
        .filter(|&x| x != a && x != b)
        .map(|x| (x, score(x)))
        .collect();
    scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
    scored
        .into_iter()
        .take(k.min(n.saturating_sub(2)))
        .map(|(x, _)| x)
        .collect()
}

fn join_box(
    centers: &[f32],
    offsets: &[f32],
    a: usize,
    b: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (ao, bo) = (a * dim, b * dim);
    let mut jc = vec![0f32; dim];
    let mut jo = vec![0f32; dim];
    for i in 0..dim {
        let lo = (centers[ao + i] - offsets[ao + i]).min(centers[bo + i] - offsets[bo + i]);
        let hi = (centers[ao + i] + offsets[ao + i]).max(centers[bo + i] + offsets[bo + i]);
        jc[i] = (lo + hi) / 2.0;
        jo[i] = (hi - lo) / 2.0;
    }
    (jc, jo)
}

fn join_match_score(
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    jc: &[f32],
    jo: &[f32],
    x: usize,
) -> f32 {
    let xo = x * dim;
    let mut d = 0f32;
    for i in 0..dim {
        d += (centers[xo + i] - jc[i]).abs() + (offsets[xo + i] - jo[i]).abs();
    }
    -d
}

fn join_contain_score(
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    jc: &[f32],
    jo: &[f32],
    lambda: f32,
    x: usize,
) -> f32 {
    let xo = x * dim;
    let mut incl = 0f32;
    let mut size = 0f32;
    for i in 0..dim {
        let v = ((jc[i] - centers[xo + i]).abs() + jo[i] - offsets[xo + i]).max(0.0);
        incl += v * v;
        size += offsets[xo + i].abs();
    }
    (-incl.sqrt()).exp() * (-lambda * size).exp()
}

fn join_gated_score(
    centers: &[f32],
    offsets: &[f32],
    dim: usize,
    jc: &[f32],
    jo: &[f32],
    tau: f32,
    x: usize,
) -> f32 {
    let xo = x * dim;
    let mut incl = 0f32;
    let mut prox = 0f32;
    for i in 0..dim {
        let v = ((jc[i] - centers[xo + i]).abs() + jo[i] - offsets[xo + i]).max(0.0);
        incl += v * v;
        prox += (centers[xo + i] - jc[i]).abs() + (offsets[xo + i] - jo[i]).abs();
    }
    (-incl.sqrt()).exp() * (-prox / tau).exp()
}

fn print_candidate_pool_diagnostic(
    queries: &[CQuery],
    centers: &[f32],
    offsets: &[f32],
    box_sizes: &[f32],
    dim: usize,
    n: usize,
) {
    let ks = [10usize, 100, 1000, 5000];
    let max_k = ks[ks.len() - 1].min(n.saturating_sub(2));
    let method_candidates: [(&str, Vec<Vec<usize>>); 5] = [
        (
            "lam=.02",
            queries
                .par_iter()
                .map(|q| {
                    top_k_by_score(n, q.a, q.b, max_k, |x| {
                        box_degree(centers, offsets, q.a, x, dim)
                            .min(box_degree(centers, offsets, q.b, x, dim))
                            * (-0.02 * box_sizes[x]).exp()
                    })
                })
                .collect(),
        ),
        (
            "join",
            queries
                .par_iter()
                .map(|q| {
                    let (jc, jo) = join_box(centers, offsets, q.a, q.b, dim);
                    top_k_by_score(n, q.a, q.b, max_k, |x| {
                        join_match_score(centers, offsets, dim, &jc, &jo, x)
                    })
                })
                .collect(),
        ),
        (
            "cont=.02",
            queries
                .par_iter()
                .map(|q| {
                    let (jc, jo) = join_box(centers, offsets, q.a, q.b, dim);
                    top_k_by_score(n, q.a, q.b, max_k, |x| {
                        join_contain_score(centers, offsets, dim, &jc, &jo, 0.02, x)
                    })
                })
                .collect(),
        ),
        (
            "gate20",
            queries
                .par_iter()
                .map(|q| {
                    let (jc, jo) = join_box(centers, offsets, q.a, q.b, dim);
                    top_k_by_score(n, q.a, q.b, max_k, |x| {
                        join_gated_score(centers, offsets, dim, &jc, &jo, 20.0, x)
                    })
                })
                .collect(),
        ),
        (
            "gate50",
            queries
                .par_iter()
                .map(|q| {
                    let (jc, jo) = join_box(centers, offsets, q.a, q.b, dim);
                    top_k_by_score(n, q.a, q.b, max_k, |x| {
                        join_gated_score(centers, offsets, dim, &jc, &jo, 50.0, x)
                    })
                })
                .collect(),
        ),
    ];

    println!(
        "\n[pool] candidate recall (true DCA in top-k; retrieval stage only, no conformal guarantee)"
    );
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "k",
        method_candidates[0].0,
        method_candidates[1].0,
        method_candidates[2].0,
        method_candidates[3].0,
        method_candidates[4].0,
        "union",
        "mean|pool|",
        "p90|pool|"
    );
    for &k in &ks {
        let method_recalls: Vec<f32> = method_candidates
            .iter()
            .map(|(_, candidates)| {
                let hits = queries
                    .iter()
                    .enumerate()
                    .filter(|&(i, q)| candidates[i].iter().take(k).any(|&x| q.is_target(x)))
                    .count();
                hits as f32 / queries.len() as f32
            })
            .collect();
        let mut union_hits = 0usize;
        let mut union_sizes = Vec::with_capacity(queries.len());
        for (i, q) in queries.iter().enumerate() {
            let mut pool = HashSet::new();
            for (_, candidates) in &method_candidates {
                pool.extend(candidates[i].iter().take(k).copied());
            }
            if pool.iter().any(|&x| q.is_target(x)) {
                union_hits += 1;
            }
            union_sizes.push(pool.len());
        }
        let union_recall = union_hits as f32 / queries.len() as f32;
        let (mean_pool, _, p90_pool, _) = set_size_summary(&union_sizes);
        println!(
            "{:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10.1} {:>10}",
            k,
            method_recalls[0],
            method_recalls[1],
            method_recalls[2],
            method_recalls[3],
            method_recalls[4],
            union_recall,
            mean_pool,
            p90_pool
        );
    }
}

fn direct_frontier_gated_scores(
    q: &CQuery,
    clqa: &BoxClqa<'_>,
    frontier: &DirectFrontier,
    extra_hops: usize,
    tau: f32,
) -> Vec<(usize, f32)> {
    let candidates = frontier
        .pool(q.a, q.b, extra_hops)
        .expect("query ids come from the same ontology as the direct frontier")
        .into_iter()
        .map(|candidate| candidate.concept);
    clqa.rank_lca_candidates(q.a, q.b, candidates, tau)
}

fn print_direct_frontier_pool_diagnostic(
    queries: &[CQuery],
    clqa: &BoxClqa<'_>,
    frontier: &DirectFrontier,
    metrics: &mut MetricsCsv,
) {
    let metric_queries: Vec<_> = queries
        .iter()
        .map(|q| DirectFrontierQuery::new(q.a, q.b, &q.lcas))
        .collect();
    println!(
        "\n[symbolic] direct-frontier candidate pool (direct SubClassOf BFS; no precomputed closure in generator)"
    );
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "extra", "recall", "mean|pool|", "p90|pool|", "MRR", "Hits@10", "top1DCA"
    );
    for extra_hops in [0usize, 1, 2, 3, 5, 10] {
        let ranked_pools: Vec<Vec<usize>> = queries
            .par_iter()
            .map(|q| {
                let scored = direct_frontier_gated_scores(q, clqa, frontier, extra_hops, 20.0);
                scored.into_iter().map(|(concept, _)| concept).collect()
            })
            .collect();
        let values = CandidatePoolMetrics::from_ranked_pools(&metric_queries, &ranked_pools);
        println!(
            "{:<8} {:>8.3} {:>10.1} {:>10} {:>8.3} {:>8.3} {:>8.3}",
            extra_hops,
            values.recall,
            values.mean_pool,
            values.p90_pool,
            values.mrr,
            values.hits10,
            values.top1_target
        );
        let param = format!("extra={extra_hops}");
        emit_retrieval_metrics(
            metrics,
            "direct_frontier_gated_pool",
            "gate20",
            &param,
            RetrievalMetrics {
                recall: values.recall,
                mean_pool: values.mean_pool,
                p90_pool: values.p90_pool,
                mrr: values.mrr,
                hits10: values.hits10,
                top1_dca: values.top1_target,
            },
        );
    }
}

fn print_direct_frontier_retrieval_diagnostic(
    queries: &[CQuery],
    frontier: &DirectFrontier,
    metrics: &mut MetricsCsv,
) {
    let metric_queries: Vec<_> = queries
        .iter()
        .map(|q| DirectFrontierQuery::new(q.a, q.b, &q.lcas))
        .collect();
    println!(
        "\n[symbolic] direct-frontier retrieval (direct SubClassOf BFS order; no box training)"
    );
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "extra", "recall", "mean|pool|", "p90|pool|", "MRR", "Hits@10", "top1DCA"
    );
    for extra_hops in [0usize, 1, 2, 3, 5, 10] {
        let values = frontier
            .pool_metrics(&metric_queries, extra_hops)
            .expect("query ids come from the same ontology as the direct frontier");
        println!(
            "{:<8} {:>8.3} {:>10.1} {:>10} {:>8.3} {:>8.3} {:>8.3}",
            extra_hops,
            values.recall,
            values.mean_pool,
            values.p90_pool,
            values.mrr,
            values.hits10,
            values.top1_target
        );
        let param = format!("extra={extra_hops}");
        emit_retrieval_metrics(
            metrics,
            "direct_frontier_retrieval",
            "bfs_order",
            &param,
            RetrievalMetrics {
                recall: values.recall,
                mean_pool: values.mean_pool,
                p90_pool: values.p90_pool,
                mrr: values.mrr,
                hits10: values.hits10,
                top1_dca: values.top1_target,
            },
        );
    }
}

fn report_direct_frontier_conformal(
    queries: &[CQuery],
    clqa: &BoxClqa<'_>,
    frontier: &DirectFrontier,
    tau: f32,
    metrics: &mut MetricsCsv,
) {
    if queries.len() < 4 {
        println!(
            "\n[conformal] skipped direct-frontier pool: need at least 4 queries for a held-out split"
        );
        return;
    }
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&i| {
        let q = &queries[i];
        q.a.wrapping_mul(2_654_435_761)
            .wrapping_add(q.b.wrapping_mul(40_503))
            .wrapping_add(i.wrapping_mul(97_531))
    });
    let split = order.len() / 2;
    let (cal_idx, test_idx) = order.split_at(split);

    println!(
        "\n[conformal] direct-frontier gated DCA sets (tau={tau}, target-missing score=0, {} calibration, {} test)",
        cal_idx.len(),
        test_idx.len()
    );
    println!(
        "{:<8} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "extra", "alpha", "q_hat", "empirical", "poolrecall", "mean|set|", "p50", "p90", "max"
    );
    for extra_hops in [0usize, 3, 10] {
        let cal_scores: Vec<(usize, Vec<(usize, f32)>)> = cal_idx
            .iter()
            .map(|&i| {
                (
                    i,
                    direct_frontier_gated_scores(&queries[i], clqa, frontier, extra_hops, tau),
                )
            })
            .collect();
        let test_scores: Vec<(usize, Vec<(usize, f32)>)> = test_idx
            .iter()
            .map(|&i| {
                (
                    i,
                    direct_frontier_gated_scores(&queries[i], clqa, frontier, extra_hops, tau),
                )
            })
            .collect();
        let cal_nonconf: Vec<f32> = cal_scores
            .iter()
            .map(|(i, scored)| 1.0 - common_pool_target_score(&queries[*i], scored).max(0.0))
            .collect();
        let pool_recall = test_scores
            .iter()
            .filter(|(i, scored)| scored.iter().any(|&(x, _)| queries[*i].is_target(x)))
            .count() as f64
            / test_scores.len() as f64;
        for &alpha in &[0.3f32, 0.2, 0.1] {
            let threshold = calibrate_scores(&cal_nonconf, alpha).expect("non-empty calibration");
            let mut covered = 0usize;
            let mut sizes = Vec::with_capacity(test_scores.len());
            for (i, scored) in &test_scores {
                let set = common_pool_answer_set(scored, &threshold);
                if set.iter().any(|&(x, _)| queries[*i].is_target(x)) {
                    covered += 1;
                }
                sizes.push(set.len());
            }
            let empirical = covered as f64 / test_scores.len() as f64;
            let (mean_size, p50, p90, max_size) = set_size_summary(&sizes);
            println!(
                "{:<8} {alpha:>7.2} {:>9.3} {empirical:>10.3} {pool_recall:>10.3} {mean_size:>10.1} {p50:>8} {p90:>8} {max_size:>8}",
                extra_hops,
                threshold.qhat
            );
            let param = format!("extra={extra_hops};tau={tau}");
            emit_conformal_metrics(
                metrics,
                "direct_frontier_conformal",
                "gate20",
                &param,
                alpha,
                ConformalMetrics {
                    q_hat: threshold.qhat,
                    empirical,
                    pool_recall,
                    mean_set: mean_size,
                    p50_set: p50,
                    p90_set: p90,
                    max_set: max_size,
                },
            );
        }
    }
}

fn score_rank_fusion(
    queries: &[CQuery],
    centers: &[f32],
    offsets: &[f32],
    box_sizes: &[f32],
    dim: usize,
    n: usize,
) -> ScoreResult {
    let rrf_k = 60.0f32;
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let (jc, jo) = join_box(centers, offsets, q.a, q.b, dim);
            let rankings = [
                top_k_by_score(n, q.a, q.b, n, |x| {
                    box_degree(centers, offsets, q.a, x, dim)
                        .min(box_degree(centers, offsets, q.b, x, dim))
                        * (-0.02 * box_sizes[x]).exp()
                }),
                top_k_by_score(n, q.a, q.b, n, |x| {
                    join_match_score(centers, offsets, dim, &jc, &jo, x)
                }),
                top_k_by_score(n, q.a, q.b, n, |x| {
                    join_contain_score(centers, offsets, dim, &jc, &jo, 0.02, x)
                }),
                top_k_by_score(n, q.a, q.b, n, |x| {
                    join_gated_score(centers, offsets, dim, &jc, &jo, 20.0, x)
                }),
                top_k_by_score(n, q.a, q.b, n, |x| {
                    join_gated_score(centers, offsets, dim, &jc, &jo, 50.0, x)
                }),
            ];
            let mut fused: HashMap<usize, f32> = HashMap::with_capacity(n.saturating_sub(2));
            for ranking in &rankings {
                for (rank, &x) in ranking.iter().enumerate() {
                    *fused.entry(x).or_default() += 1.0 / (rrf_k + rank as f32 + 1.0);
                }
            }
            let mut scored: Vec<(usize, f32)> = fused.into_iter().collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

fn score_common_pool_gated(queries: &[CQuery], clqa: &BoxClqa<'_>, tau: f32) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let scored = common_pool_gated_scores(q, clqa, tau);
            let rank = target_rank_pairs(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.is_target(scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

fn common_pool_gated_scores(q: &CQuery, clqa: &BoxClqa<'_>, tau: f32) -> Vec<(usize, f32)> {
    let candidates = q.common.iter().copied().filter(|&x| x != q.a && x != q.b);
    clqa.rank_lca_candidates(q.a, q.b, candidates, tau)
}

fn common_pool_target_score(q: &CQuery, scored: &[(usize, f32)]) -> f32 {
    scored
        .iter()
        .filter(|&&(x, _)| q.is_target(x))
        .map(|&(_, score)| score)
        .fold(f32::NEG_INFINITY, f32::max)
}

fn common_pool_answer_set(
    scored: &[(usize, f32)],
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let cutoff = 1.0 - threshold.qhat;
    scored
        .iter()
        .copied()
        .filter(|&(_, score)| score >= cutoff)
        .collect()
}

fn score_common_pool_depth(queries: &[CQuery], depths: &[usize]) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<usize> = q
                .common
                .iter()
                .copied()
                .filter(|&x| x != q.a && x != q.b)
                .collect();
            scored.sort_by_key(|&x| (std::cmp::Reverse(depths[x]), x));
            let rank = target_rank_ids(q, &scored);
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.is_target(scored[0])),
                scored[0],
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

fn score_common_pool_depth_gated(
    queries: &[CQuery],
    clqa: &BoxClqa<'_>,
    depths: &[usize],
    tau: f32,
) -> ScoreResult {
    let per: Vec<(f64, usize, usize, usize, usize)> = queries
        .par_iter()
        .map(|q| {
            let (jc, jo) = clqa.join(q.a, q.b);
            let mut scored: Vec<(usize, usize, f32)> = q
                .common
                .iter()
                .copied()
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| (x, depths[x], clqa.score_lca(&jc, &jo, x, tau)))
                .collect();
            scored.sort_by(|p, r| {
                r.1.cmp(&p.1)
                    .then_with(|| r.2.partial_cmp(&p.2).unwrap())
                    .then_with(|| p.0.cmp(&r.0))
            });
            let rank = 1 + scored.iter().position(|&(x, _, _)| q.is_target(x)).unwrap();
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.is_target(scored[0].0)),
                scored[0].0,
                rank,
            )
        })
        .collect();
    let (rr, hits10, top1) = per.iter().fold((0.0f64, 0usize, 0usize), |a, x| {
        (a.0 + x.0, a.1 + x.1, a.2 + x.2)
    });
    let nq = queries.len() as f64;
    (
        (rr / nq, hits10 as f64 / nq, top1 as f64 / nq),
        per.iter().map(|x| x.3).collect(),
        per.iter().map(|x| x.4).collect(),
    )
}

fn print_common_pool_diagnostic(queries: &[CQuery], clqa: &BoxClqa<'_>, depths: &[usize]) {
    let pool_sizes: Vec<usize> = queries.iter().map(|q| q.common.len()).collect();
    let (mean_pool, p50_pool, p90_pool, max_pool) = set_size_summary(&pool_sizes);
    let target_sizes: Vec<usize> = queries.iter().map(|q| q.lcas.len()).collect();
    let (mean_target, p50_target, p90_target, max_target) = set_size_summary(&target_sizes);
    println!("\n[symbolic] common-ancestor candidate pool (ontology-closure retrieval oracle)");
    println!(
        "[symbolic] pool size: n={} mean={mean_pool:.1} p50={p50_pool} p90={p90_pool} max={max_pool}",
        queries.len()
    );
    println!(
        "[symbolic] deepest common ancestors: mean={mean_target:.1} p50={p50_target} p90={p90_target} max={max_target}"
    );
    println!(
        "{:<22} {:>8} {:>8} {:>8}",
        "readout", "DCA MRR", "Hits@10", "top1DCA"
    );
    for tau in [20.0f32, 50.0] {
        let ((mrr, h10, t1), _, ranks) = score_common_pool_gated(queries, clqa, tau);
        println!(
            "{:<22} {mrr:>8.3} {h10:>8.3} {t1:>8.3}",
            format!("common + gate{tau}")
        );
        print_rank_diagnostic(&format!("common + gate{tau}"), &ranks);
    }
    let ((mrr, h10, t1), _, ranks) = score_common_pool_depth(queries, depths);
    println!("{:<22} {mrr:>8.3} {h10:>8.3} {t1:>8.3}", "common + depth");
    print_rank_diagnostic("common + depth", &ranks);
    for tau in [20.0f32, 50.0] {
        let ((mrr, h10, t1), _, ranks) = score_common_pool_depth_gated(queries, clqa, depths, tau);
        println!(
            "{:<22} {mrr:>8.3} {h10:>8.3} {t1:>8.3}",
            format!("depth + gate{tau}")
        );
        print_rank_diagnostic(&format!("depth + gate{tau}"), &ranks);
    }
}

fn report_common_pool_conformal(queries: &[CQuery], clqa: &BoxClqa<'_>, tau: f32) {
    if queries.len() < 4 {
        println!(
            "\n[conformal] skipped symbolic pool: need at least 4 queries for a held-out split"
        );
        return;
    }
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&i| {
        let q = &queries[i];
        q.a.wrapping_mul(2_654_435_761)
            .wrapping_add(q.b.wrapping_mul(40_503))
            .wrapping_add(i.wrapping_mul(97_531))
    });
    let split = order.len() / 2;
    let (cal_idx, test_idx) = order.split_at(split);
    let cal_scores: Vec<(usize, Vec<(usize, f32)>)> = cal_idx
        .iter()
        .map(|&i| (i, common_pool_gated_scores(&queries[i], clqa, tau)))
        .collect();
    let test_scores: Vec<(usize, Vec<(usize, f32)>)> = test_idx
        .iter()
        .map(|&i| (i, common_pool_gated_scores(&queries[i], clqa, tau)))
        .collect();
    let cal_nonconf: Vec<f32> = cal_scores
        .iter()
        .map(|(i, scored)| 1.0 - common_pool_target_score(&queries[*i], scored))
        .collect();
    let cal_target_scores: Vec<f32> = cal_nonconf.iter().map(|s| 1.0 - *s).collect();

    println!(
        "\n[conformal] symbolic-pool gated DCA sets (tau={tau}, {} calibration, {} test)",
        cal_idx.len(),
        test_idx.len()
    );
    print_f32_distribution(
        "symbolic-pool calibration nonconformity (1 - target DCA score)",
        &cal_nonconf,
    );
    print_f32_distribution(
        "symbolic-pool calibration target-DCA score",
        &cal_target_scores,
    );
    println!(
        "{:<8} {:>8} {:>10} {:>12} {:>12} {:>8} {:>8} {:>8}",
        "alpha", "1-alpha", "q_hat", "empirical", "mean|set|", "p50", "p90", "max"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thr = calibrate_scores(&cal_nonconf, alpha).expect("non-empty calibration");
        let mut hits = 0usize;
        let mut set_sizes = Vec::with_capacity(test_scores.len());
        for (i, scored) in &test_scores {
            let q = &queries[*i];
            let set = common_pool_answer_set(scored, &thr);
            if set.iter().any(|&(x, _)| q.is_target(x)) {
                hits += 1;
            }
            set_sizes.push(set.len());
        }
        let coverage = hits as f32 / test_scores.len() as f32;
        let (mean_set, p50_set, p90_set, max_set) = set_size_summary(&set_sizes);
        println!(
            "{alpha:<8.2} {:>8.2} {:>10.3} {:>12.3} {:>12.1} {:>8} {:>8} {:>8}",
            1.0 - alpha,
            thr.qhat,
            coverage,
            mean_set,
            p50_set,
            p90_set,
            max_set
        );
    }
}

fn best_nonquery_score(degrees: &[f32], a: usize, b: usize) -> f32 {
    degrees
        .iter()
        .enumerate()
        .filter(|(x, _)| *x != a && *x != b)
        .map(|(_, d)| *d)
        .fold(0.0, f32::max)
}

fn confidence_bucket(score: f32, cutoffs: &[f32; 2]) -> usize {
    if score <= cutoffs[0] {
        0
    } else if score <= cutoffs[1] {
        1
    } else {
        2
    }
}

fn nonquery_answer_set(
    degrees: &[f32],
    threshold: &ConformalThreshold,
    a: usize,
    b: usize,
) -> Vec<(usize, f32)> {
    answer_set_from_degrees(degrees, threshold)
        .into_iter()
        .filter(|&(x, _)| x != a && x != b)
        .collect()
}

fn capped_nonquery_answer_set(
    degrees: &[f32],
    threshold: &ConformalThreshold,
    a: usize,
    b: usize,
    cap: usize,
) -> Vec<(usize, f32)> {
    let mut set = nonquery_answer_set(degrees, threshold, a, b);
    set.truncate(cap);
    set
}

fn report_gated_conformal(queries: &[CQuery], clqa: &BoxClqa<'_>, tau: f32) {
    if queries.len() < 4 {
        println!("\n[conformal] skipped: need at least 4 queries for a held-out split");
        return;
    }
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&i| {
        let q = &queries[i];
        q.a.wrapping_mul(2_654_435_761)
            .wrapping_add(q.b.wrapping_mul(40_503))
            .wrapping_add(i.wrapping_mul(97_531))
    });
    let split = order.len() / 2;
    let (cal_idx, test_idx) = order.split_at(split);
    let cal_degrees: Vec<(usize, Vec<f32>)> = cal_idx
        .iter()
        .map(|&i| {
            let q = &queries[i];
            (i, gated_lca_degrees(clqa, q.a, q.b, tau))
        })
        .collect();
    let test_degrees: Vec<(usize, Vec<f32>)> = test_idx
        .iter()
        .map(|&i| {
            let q = &queries[i];
            (i, gated_lca_degrees(clqa, q.a, q.b, tau))
        })
        .collect();
    let cal_nonconf: Vec<f32> = cal_idx
        .iter()
        .zip(cal_degrees.iter())
        .map(|(&i, (_, degs))| {
            let q = &queries[i];
            1.0 - q.target_score(degs)
        })
        .collect();
    let cal_target_degrees: Vec<f32> = cal_nonconf.iter().map(|s| 1.0 - *s).collect();

    println!(
        "\n[conformal] gated DCA target sets (tau={tau}, {} calibration, {} test)",
        cal_idx.len(),
        test_idx.len()
    );
    print_f32_distribution(
        "calibration nonconformity (1 - target DCA score)",
        &cal_nonconf,
    );
    print_f32_distribution("calibration target-DCA score", &cal_target_degrees);
    println!(
        "{:<8} {:>8} {:>10} {:>12} {:>12} {:>8} {:>8} {:>8}",
        "alpha", "1-alpha", "q_hat", "empirical", "mean|set|", "p50", "p90", "max"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thr = calibrate_scores(&cal_nonconf, alpha).expect("non-empty calibration");
        let mut hits = 0usize;
        let mut set_sizes = Vec::with_capacity(test_idx.len());
        for &(i, ref degs) in &test_degrees {
            let q = &queries[i];
            let set = nonquery_answer_set(degs, &thr, q.a, q.b);
            if set.iter().any(|&(x, _)| q.is_target(x)) {
                hits += 1;
            }
            set_sizes.push(set.len());
        }
        let coverage = hits as f32 / test_idx.len() as f32;
        let (mean_set, p50_set, p90_set, max_set) = set_size_summary(&set_sizes);
        println!(
            "{alpha:<8.2} {:>8.2} {:>10.3} {:>12.3} {:>12.1} {:>8} {:>8} {:>8}",
            1.0 - alpha,
            thr.qhat,
            coverage,
            mean_set,
            p50_set,
            p90_set,
            max_set
        );
    }

    println!(
        "\n[conformal] size-capped diagnostic (cap truncates the conformal set; empirical only, no marginal guarantee)"
    );
    println!(
        "{:<8} {:>8} {:>8} {:>10} {:>12} {:>12} {:>8} {:>8} {:>8}",
        "alpha", "1-alpha", "cap", "q_hat", "empirical", "mean|set|", "p50", "p90", "max"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thr = calibrate_scores(&cal_nonconf, alpha).expect("non-empty calibration");
        for &cap in &[10usize, 100, 1000, 5000] {
            let mut hits = 0usize;
            let mut set_sizes = Vec::with_capacity(test_idx.len());
            for &(i, ref degs) in &test_degrees {
                let q = &queries[i];
                let set = capped_nonquery_answer_set(degs, &thr, q.a, q.b, cap);
                if set.iter().any(|&(x, _)| q.is_target(x)) {
                    hits += 1;
                }
                set_sizes.push(set.len());
            }
            let coverage = hits as f32 / test_idx.len() as f32;
            let (mean_set, p50_set, p90_set, max_set) = set_size_summary(&set_sizes);
            println!(
                "{alpha:<8.2} {:>8.2} {:>8} {:>10.3} {:>12.3} {:>12.1} {:>8} {:>8} {:>8}",
                1.0 - alpha,
                cap,
                thr.qhat,
                coverage,
                mean_set,
                p50_set,
                p90_set,
                max_set
            );
        }
    }

    if cal_idx.len() < 9 {
        println!("\n[conformal] skipped confidence buckets: need at least 9 calibration examples");
        return;
    }

    let mut cal_conf: Vec<(f32, f32)> = cal_idx
        .iter()
        .zip(cal_degrees.iter())
        .map(|(&i, (_, degs))| {
            let q = &queries[i];
            (
                best_nonquery_score(degs, q.a, q.b),
                1.0 - q.target_score(degs),
            )
        })
        .collect();
    cal_conf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let cutoffs = [
        cal_conf[cal_conf.len() / 3].0,
        cal_conf[cal_conf.len() * 2 / 3].0,
    ];
    let mut bucket_scores = [Vec::new(), Vec::new(), Vec::new()];
    for &(confidence, nonconf) in &cal_conf {
        bucket_scores[confidence_bucket(confidence, &cutoffs)].push(nonconf);
    }
    if bucket_scores.iter().any(Vec::is_empty) {
        println!(
            "\n[conformal] skipped confidence buckets: calibration scores collapsed into an empty bucket"
        );
        return;
    }
    let cal_best_scores: Vec<f32> = cal_conf.iter().map(|(score, _)| *score).collect();
    print_f32_distribution("calibration best non-query score", &cal_best_scores);
    println!(
        "\n[conformal] confidence-bucketed gated DCA target sets (best-score cutoffs {:.3}, {:.3})",
        cutoffs[0], cutoffs[1]
    );
    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12}",
        "alpha", "1-alpha", "empirical", "mean|set|", "p50", "p90", "max", "bucket n"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thresholds = bucket_scores.each_ref().map(|scores| {
            calibrate_scores(scores, alpha).expect("confidence bucket has calibration examples")
        });
        let mut hits = 0usize;
        let mut bucket_counts = [0usize; 3];
        let mut bucket_hits = [0usize; 3];
        let mut bucket_set_sizes = [Vec::new(), Vec::new(), Vec::new()];
        let mut all_set_sizes = Vec::with_capacity(test_idx.len());
        for &(i, ref degs) in &test_degrees {
            let q = &queries[i];
            let bucket = confidence_bucket(best_nonquery_score(degs, q.a, q.b), &cutoffs);
            bucket_counts[bucket] += 1;
            let set = nonquery_answer_set(degs, &thresholds[bucket], q.a, q.b);
            if set.iter().any(|&(x, _)| q.is_target(x)) {
                hits += 1;
                bucket_hits[bucket] += 1;
            }
            bucket_set_sizes[bucket].push(set.len());
            all_set_sizes.push(set.len());
        }
        let coverage = hits as f32 / test_idx.len() as f32;
        let (mean_set, p50_set, p90_set, max_set) = set_size_summary(&all_set_sizes);
        let bucket_counts_label = format!(
            "{}/{}/{}",
            bucket_counts[0], bucket_counts[1], bucket_counts[2]
        );
        println!(
            "{alpha:<8.2} {:>8.2} {:>12.3} {:>12.1} {:>8} {:>8} {:>8} {:>12}",
            1.0 - alpha,
            coverage,
            mean_set,
            p50_set,
            p90_set,
            max_set,
            bucket_counts_label
        );
        for bucket in 0..3 {
            let (bucket_mean, bucket_p50, bucket_p90, bucket_max) =
                set_size_summary(&bucket_set_sizes[bucket]);
            let bucket_coverage = if bucket_counts[bucket] == 0 {
                0.0
            } else {
                bucket_hits[bucket] as f32 / bucket_counts[bucket] as f32
            };
            println!(
                "  bucket {bucket}: cal={} test={} q_hat={:.3} coverage={bucket_coverage:.3} mean|set|={bucket_mean:.1} p50={bucket_p50} p90={bucket_p90} max={bucket_max}",
                bucket_scores[bucket].len(),
                bucket_counts[bucket],
                thresholds[bucket].qhat
            );
        }
    }
}

fn main() {
    let device = Default::default();
    <Backend as burn::tensor::backend::Backend>::seed(&device, 3);
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "GALEN".to_string());
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let n_queries: usize = std::env::var("QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    // Large batch: on wgpu/Metal, small batches are dispatch-bound (kernel-launch
    // overhead dominates); Box2EL/DELE use batch ~32k. Fewer, larger dispatches
    // keep the GPU busy. LR is scaled up to match the larger batch.
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let lr: f64 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    // SKIP_TRANSE=1 trains only the box model and runs the box diagnostics: a
    // ~2-minute box-only run vs the slow TransE baseline (point-KGE at 400
    // epochs is slow on both wgpu and CPU). Use it to iterate on box training.
    let skip_transe = std::env::var("SKIP_TRANSE").is_ok();
    // Hard clamp on box size (curbs training-time offset blowup).
    let offset_clamp: f32 = std::env::var("OFFSET_CLAMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    // Size penalty in the box conjunctive score: prefer the smallest common
    // container (the LCA) over larger general containers. 0 = plain containment.
    let lambda: f32 = std::env::var("TIGHTNESS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let symbolic_only = std::env::var("SYMBOLIC_ONLY").is_ok();
    let learned_retrieval = std::env::var("LEARNED_RETRIEVAL").is_ok();

    let train_path = Path::new("data").join(&dataset).join("train.tsv");
    if !train_path.exists() {
        println!(
            "data/{dataset}/train.tsv not found; skipping real-ontology eval (data-gated).\n\
             Fetch the Box2EL EL++ datasets to run this (see el_benchmark examples)."
        );
        return;
    }

    let ds = match load_el_axioms(&train_path) {
        Ok(d) => d,
        Err(e) => {
            println!("failed to load {train_path:?}: {e}; skipping.");
            return;
        }
    };
    let ont = Ontology::from_el_dataset(&ds);
    let n = ont.concept_names.len();
    println!(
        "{dataset}: {n} concepts, {} axioms, dim {dim}, {epochs} epochs",
        ont.axioms.len()
    );

    // Ancestor sets from the deductive closure.
    let closure = ont.subsumption_closure();
    let mut anc: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(s, a) in &closure {
        if s < n && a < n {
            anc[s].insert(a);
        }
    }

    // Sample concept pairs with a non-trivial deepest common ancestor (DCA depth >= 2),
    // deterministically via an LCG so the query set is reproducible.
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let depth = |x: usize| anc[x].len();
    let mut queries: Vec<CQuery> = Vec::new();
    let mut attempts = 0;
    while queries.len() < n_queries && attempts < n_queries * 400 {
        attempts += 1;
        let (a, b) = (lcg() % n, lcg() % n);
        if a == b || anc[a].is_empty() || anc[b].is_empty() {
            continue;
        }
        let common: HashSet<usize> = anc[a].intersection(&anc[b]).copied().collect();
        let max_depth = common
            .iter()
            .copied()
            .filter(|&x| x != a && x != b)
            .map(depth)
            .max();
        let Some(max_depth) = max_depth.filter(|&d| d >= 2) else {
            continue;
        };
        let mut lcas: Vec<usize> = common
            .iter()
            .copied()
            .filter(|&x| x != a && x != b && depth(x) == max_depth)
            .collect();
        lcas.sort_unstable();
        queries.push(CQuery { a, b, common, lcas });
    }

    let depths: Vec<usize> = (0..n).map(|c| anc[c].len()).collect();
    let direct_frontier = DirectFrontier::from_ontology(&ont)
        .expect("ontology direct SubClassOf endpoints must reference known concepts");
    println!("\n{} conjunctive queries (DCA depth >= 2)\n", queries.len());
    let mut metrics = MetricsCsv::from_env(&dataset, dim, epochs, queries.len())
        .expect("METRICS_CSV path must be writable");
    metrics.emit("run", "dataset", "", None, "concepts", n);
    metrics.emit("run", "dataset", "", None, "axioms", ont.axioms.len());
    metrics.emit("run", "config", "", None, "batch", batch);
    metrics.emit("run", "config", "", None, "lr", lr);
    metrics.emit("run", "config", "", None, "offset_clamp", offset_clamp);
    metrics.emit("run", "config", "", None, "tightness", lambda);
    metrics.emit(
        "run",
        "config",
        "",
        None,
        "symbolic_only",
        usize::from(symbolic_only),
    );
    metrics.emit(
        "run",
        "config",
        "",
        None,
        "learned_retrieval",
        usize::from(learned_retrieval),
    );
    metrics.emit(
        "run",
        "config",
        "",
        None,
        "skip_transe",
        usize::from(skip_transe),
    );
    if symbolic_only {
        print_direct_frontier_retrieval_diagnostic(&queries, &direct_frontier, &mut metrics);
        if learned_retrieval {
            learned::report_learned_frontier_ranker(&queries, &direct_frontier, &mut metrics);
        }
        return;
    }
    if learned_retrieval {
        learned::report_learned_frontier_ranker(&queries, &direct_frontier, &mut metrics);
    }

    // --- Faithful boxes ---
    let box_cfg = BurnElConfig {
        dim,
        epochs,
        lr,
        negative_samples: 2,
        margin: 0.1,
        batch_size: batch,
        offset_clamp,
        ..Default::default()
    };
    let trainer = BurnElTrainer::<Backend>::new();
    let mut model =
        BurnElTrainer::<Backend>::init_model(n, ont.role_names.len().max(1), dim, &device);
    let epoch_losses = trainer.fit(&mut model, &ont, &box_cfg, &device);
    let finite_losses: Vec<f32> = epoch_losses
        .iter()
        .copied()
        .filter(|loss| loss.is_finite())
        .collect();
    print_f32_distribution("training epoch loss", &finite_losses);
    let nonfinite_losses = epoch_losses.len().saturating_sub(finite_losses.len());
    if nonfinite_losses > 0 {
        println!("[diag] training epoch loss: {nonfinite_losses} non-finite epochs");
    }
    let centers = BurnElTrainer::<Backend>::extract_centers(&model, &device);
    let offsets = BurnElTrainer::<Backend>::extract_offsets(&model, &device);

    // --- Plain TransE on the atomic subsumption triples (skippable) ---
    let transe: Option<(Vec<Vec<f32>>, Vec<f32>)> = if skip_transe {
        None
    } else {
        let triples: Vec<TripleIds> = ont
            .axioms
            .iter()
            .filter_map(|ax| match ax {
                Axiom::SubClassOf { sub, sup } => Some(TripleIds::new(*sub, 0, *sup)),
                _ => None,
            })
            .collect();
        let transe_cfg = BurnTrainConfig {
            dim,
            epochs,
            batch_size: batch,
            lr,
            ..BurnTrainConfig::default()
        };
        let kge = train_kge::<Backend>(&triples, n, 1, BurnModelType::TransE, &transe_cfg, &device);
        Some((kge.entity_vecs, kge.relation_vecs[0].clone()))
    };

    let box_sizes: Vec<f32> = (0..n).map(|c| offset_l1(&offsets, c, dim)).collect();
    println!(
        "{:<22} {:>8} {:>8} {:>8}",
        "model", "DCA MRR", "Hits@10", "top1CA"
    );
    println!("{}", "-".repeat(48));
    // Sweep the size-penalty lambda (train once, eval many). TIGHTNESS unset
    // sweeps a grid; set uses that one value. lambda=0 is plain containment.
    let lambdas: Vec<f32> = if lambda > 0.0 {
        vec![lambda]
    } else {
        vec![0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    };
    let mut box_mrr = 0.0f64;
    let mut box_top1s: Vec<usize> = Vec::new();
    for (i, &lam) in lambdas.iter().enumerate() {
        let ((mrr, h10, t1), top1s, ranks) = score_model(
            &queries,
            |a, x| box_degree(&centers, &offsets, a, x, dim),
            &box_sizes,
            lam,
            n,
        );
        println!(
            "{:<22} {mrr:>8.3} {h10:>8.3} {t1:>8.3}",
            format!("boxes (lam={lam})")
        );
        emit_score_result(
            &mut metrics,
            "boxes",
            &format!("lambda={lam}"),
            mrr,
            h10,
            t1,
        );
        if i == 0 {
            box_mrr = mrr;
            box_top1s = top1s;
        }
        if (lam - 0.02).abs() < f32::EPSILON {
            print_rank_diagnostic("boxes (lam=0.02)", &ranks);
        }
    }
    // Non-containment scorer: rank by match to the join box (the LCA's expected
    // shape), sidestepping the widest-region-wins degeneracy of containment.
    let ((jm_mrr, jm_h10, jm_t1), _, jm_ranks) = score_join(&queries, &centers, &offsets, dim, n);
    println!(
        "{:<22} {jm_mrr:>8.3} {jm_h10:>8.3} {jm_t1:>8.3}",
        "boxes (join-match)"
    );
    emit_score_result(&mut metrics, "boxes_join_match", "", jm_mrr, jm_h10, jm_t1);
    print_rank_diagnostic("boxes (join-match)", &jm_ranks);
    // Smallest-container-of-the-join score, swept over the size penalty.
    for jl in [0.02f32, 0.05, 0.1] {
        let ((c_mrr, c_h10, c_t1), _, c_ranks) =
            score_join_contain(&queries, &centers, &offsets, dim, jl, n);
        println!(
            "{:<22} {c_mrr:>8.3} {c_h10:>8.3} {c_t1:>8.3}",
            format!("boxes (join-cont l={jl})")
        );
        emit_score_result(
            &mut metrics,
            "boxes_join_contain",
            &format!("lambda={jl}"),
            c_mrr,
            c_h10,
            c_t1,
        );
        if (jl - 0.02).abs() < f32::EPSILON {
            print_rank_diagnostic("boxes (join-cont l=0.02)", &c_ranks);
        }
    }
    // Containment-gated proximity: the synthesis readout, swept over the
    // proximity temperature. tau scales the L1-distance-to-degree map; too small
    // collapses to the saturating join-cont behaviour, too large washes out the
    // proximity tie-break. Swept relative to the mean box-to-join distance.
    let clqa = BoxClqa::new(&centers, &offsets, dim).expect("trained box slices align");
    for tau in [5.0f32, 20.0, 50.0] {
        let ((g_mrr, g_h10, g_t1), _, g_ranks) =
            score_join_gated(&queries, &centers, &offsets, dim, tau, n);
        println!(
            "{:<22} {g_mrr:>8.3} {g_h10:>8.3} {g_t1:>8.3}",
            format!("boxes (join-gate t={tau})")
        );
        emit_score_result(
            &mut metrics,
            "boxes_join_gate",
            &format!("tau={tau}"),
            g_mrr,
            g_h10,
            g_t1,
        );
        print_rank_diagnostic(&format!("boxes (join-gate t={tau})"), &g_ranks);
    }
    let ((rf_mrr, rf_h10, rf_t1), _, rf_ranks) =
        score_rank_fusion(&queries, &centers, &offsets, &box_sizes, dim, n);
    println!(
        "{:<22} {rf_mrr:>8.3} {rf_h10:>8.3} {rf_t1:>8.3}",
        "boxes (rank-fusion)"
    );
    emit_score_result(&mut metrics, "boxes_rank_fusion", "", rf_mrr, rf_h10, rf_t1);
    print_rank_diagnostic("boxes (rank-fusion)", &rf_ranks);
    print_candidate_pool_diagnostic(&queries, &centers, &offsets, &box_sizes, dim, n);
    print_direct_frontier_pool_diagnostic(&queries, &clqa, &direct_frontier, &mut metrics);
    report_direct_frontier_conformal(&queries, &clqa, &direct_frontier, 20.0, &mut metrics);
    print_common_pool_diagnostic(&queries, &clqa, &depths);
    report_common_pool_conformal(&queries, &clqa, 20.0);
    report_gated_conformal(&queries, &clqa, 20.0);
    if let Some((emb, rel)) = &transe {
        // TransE is a point model: no box size, so no size penalty.
        let zeros = vec![0.0f32; n];
        let ((te_mrr, te_h10, te_t1), _, te_ranks) = score_model(
            &queries,
            |a, x| transe_degree(emb, rel, a, x),
            &zeros,
            0.0,
            n,
        );
        println!(
            "{:<22} {te_mrr:>8.3} {te_h10:>8.3} {te_t1:>8.3}",
            "plain KGE (TransE)"
        );
        emit_score_result(&mut metrics, "plain_kge_transe", "", te_mrr, te_h10, te_t1);
        print_rank_diagnostic("plain KGE (TransE)", &te_ranks);
        let delta = box_mrr - te_mrr;
        println!(
            "\nDCA MRR delta (faithful - plain): {delta:+.3}  ({} on real {dataset})",
            if delta > 0.0 {
                "faithful wins"
            } else {
                "plain wins"
            }
        );
    } else {
        println!("(TransE skipped via SKIP_TRANSE; box-only diagnostic run)");
    }

    // --- Diagnostics: peer into the box model to explain the result ---
    // Box size (offset L1) distribution. Degenerate huge boxes contain almost
    // everything and win every conjunctive query, tanking top1CA. A large
    // max/mean ratio is the offset-blowup signature.
    let mut sizes: Vec<f32> = (0..n).map(|c| offset_l1(&offsets, c, dim)).collect();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let smean = sizes.iter().sum::<f32>() / n as f32;
    let (smax, sp99) = (sizes[n - 1], sizes[(n * 99 / 100).min(n - 1)]);
    println!(
        "\n[diag] box size (offset L1): mean {smean:.2}  p99 {sp99:.2}  max {smax:.2}  (max/mean {:.0}x)",
        smax / smean.max(1e-6)
    );
    // Which concept wins the box conjunctive queries most? A degenerate box wins
    // many with an outsized offset; high concentration = a few boxes dominate.
    let mut tally: HashMap<usize, usize> = HashMap::new();
    for &c in &box_top1s {
        *tally.entry(c).or_default() += 1;
    }
    let mut culprits: Vec<(usize, usize)> = tally.into_iter().collect();
    culprits.sort_by_key(|&(_, cnt)| std::cmp::Reverse(cnt));
    let (top_c, top_cnt) = culprits[0];
    println!(
        "[diag] box top-1 winners: {} distinct over {} queries; concept {top_c} won {top_cnt} \
         ({:.0}%), its size {:.2} = {:.0}x mean {}",
        culprits.len(),
        box_top1s.len(),
        100.0 * top_cnt as f32 / box_top1s.len() as f32,
        offset_l1(&offsets, top_c, dim),
        offset_l1(&offsets, top_c, dim) / smean.max(1e-6),
        if offset_l1(&offsets, top_c, dim) > 3.0 * smean {
            "<- degenerate box confirmed"
        } else {
            ""
        }
    );
}
