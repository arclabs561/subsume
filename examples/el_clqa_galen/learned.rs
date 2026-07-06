use super::{
    emit_conformal_metrics, emit_retrieval_metrics, print_f32_distribution, set_size_summary,
    CQuery, ConformalMetrics, MetricsCsv, RetrievalMetrics,
};
use heyting::conformal::calibrate_scores;
use rayon::prelude::*;
use subsume::clqa::{DirectFrontier, FrontierCandidate};

const FRONTIER_FEATURES: usize = 5;

#[derive(Clone, Copy)]
struct LearnedRankerConfig {
    extra_hops: usize,
    epochs: usize,
    lr: f32,
    l2: f32,
    repeats: usize,
}

fn query_split_indices(queries: &[CQuery]) -> (Vec<usize>, Vec<usize>) {
    let order = query_order_indices(queries);
    let split = order.len() / 2;
    (order[..split].to_vec(), order[split..].to_vec())
}

fn query_order_indices(queries: &[CQuery]) -> Vec<usize> {
    query_order_indices_with_salt(queries, 0)
}

fn query_order_indices_with_salt(queries: &[CQuery], salt: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&i| {
        let q = &queries[i];
        q.a.wrapping_mul(2_654_435_761)
            .wrapping_add(q.b.wrapping_mul(40_503))
            .wrapping_add(i.wrapping_mul(97_531))
    });
    if !order.is_empty() && salt > 0 {
        let len = order.len();
        let shift = salt.wrapping_mul(len / 3 + 1) % len;
        order.rotate_left(shift);
    }
    order
}

fn frontier_features(
    candidate: &FrontierCandidate,
    frontier: &DirectFrontier,
) -> [f32; FRONTIER_FEATURES] {
    let parent_count = frontier
        .parents(candidate.concept)
        .expect("frontier candidate concept came from the same frontier")
        .len() as f32;
    let imbalance = candidate
        .frontier_depth
        .saturating_mul(2)
        .abs_diff(candidate.path_len) as f32;
    [
        1.0,
        candidate.frontier_depth as f32,
        candidate.path_len as f32,
        imbalance,
        parent_count.ln_1p(),
    ]
}

fn dot_frontier(weights: &[f32; FRONTIER_FEATURES], features: &[f32; FRONTIER_FEATURES]) -> f32 {
    weights.iter().zip(features).map(|(w, f)| w * f).sum()
}

fn sigmoid_neg_margin(margin: f32) -> f32 {
    if margin >= 0.0 {
        let e = (-margin).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + margin.exp())
    }
}

fn pairwise_logistic_loss(margin: f32) -> f32 {
    if margin >= 0.0 {
        (-margin).exp().ln_1p()
    } else {
        -margin + margin.exp().ln_1p()
    }
}

fn train_frontier_ranker(
    queries: &[CQuery],
    train_idx: &[usize],
    frontier: &DirectFrontier,
    extra_hops: usize,
    epochs: usize,
    lr: f32,
    l2: f32,
) -> ([f32; FRONTIER_FEATURES], Vec<f32>, usize) {
    let mut weights = [0.0f32; FRONTIER_FEATURES];
    let mut losses = Vec::with_capacity(epochs);
    let mut total_pairs = 0usize;
    for _ in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_pairs = 0usize;
        for &i in train_idx {
            let q = &queries[i];
            let pool = frontier
                .pool(q.a, q.b, extra_hops)
                .expect("query ids come from the same ontology as the direct frontier");
            let rows: Vec<(bool, [f32; FRONTIER_FEATURES])> = pool
                .iter()
                .map(|candidate| {
                    (
                        q.is_target(candidate.concept),
                        frontier_features(candidate, frontier),
                    )
                })
                .collect();
            let positives: Vec<usize> = rows
                .iter()
                .enumerate()
                .filter_map(|(j, (is_target, _))| is_target.then_some(j))
                .collect();
            let negatives: Vec<usize> = rows
                .iter()
                .enumerate()
                .filter_map(|(j, (is_target, _))| (!is_target).then_some(j))
                .collect();
            for &p in &positives {
                for &neg in &negatives {
                    let mut diff = [0.0f32; FRONTIER_FEATURES];
                    for (f, value) in diff.iter_mut().enumerate() {
                        *value = rows[p].1[f] - rows[neg].1[f];
                    }
                    let margin = dot_frontier(&weights, &diff);
                    let scale = sigmoid_neg_margin(margin);
                    for (weight, delta) in weights.iter_mut().zip(diff) {
                        *weight += lr * scale * delta;
                    }
                    epoch_loss += pairwise_logistic_loss(margin);
                    epoch_pairs += 1;
                }
            }
        }
        if l2 > 0.0 {
            for weight in &mut weights[1..] {
                *weight *= (1.0 - lr * l2).max(0.0);
            }
        }
        if epoch_pairs > 0 {
            losses.push(epoch_loss / epoch_pairs as f32);
            total_pairs = epoch_pairs;
        } else {
            losses.push(0.0);
        }
    }
    (weights, losses, total_pairs)
}

fn evaluate_frontier_ranker(
    queries: &[CQuery],
    idx: &[usize],
    frontier: &DirectFrontier,
    extra_hops: usize,
    weights: Option<&[f32; FRONTIER_FEATURES]>,
) -> RetrievalMetrics {
    let per: Vec<(bool, usize, Option<usize>, bool)> = idx
        .par_iter()
        .map(|&i| {
            let q = &queries[i];
            let pool = frontier
                .pool(q.a, q.b, extra_hops)
                .expect("query ids come from the same ontology as the direct frontier");
            let mut ranked: Vec<(usize, f32, usize, usize, usize)> = pool
                .iter()
                .map(|candidate| {
                    let score = weights.map_or(0.0, |w| {
                        dot_frontier(w, &frontier_features(candidate, frontier))
                    });
                    (
                        candidate.concept,
                        score,
                        candidate.frontier_depth,
                        candidate.path_len,
                        candidate.concept,
                    )
                })
                .collect();
            if weights.is_some() {
                ranked.sort_by(|p, r| {
                    r.1.total_cmp(&p.1)
                        .then_with(|| p.2.cmp(&r.2))
                        .then_with(|| p.3.cmp(&r.3))
                        .then_with(|| p.4.cmp(&r.4))
                });
            }
            let hit = ranked.iter().any(|&(x, _, _, _, _)| q.is_target(x));
            let rank = ranked
                .iter()
                .position(|&(x, _, _, _, _)| q.is_target(x))
                .map(|j| j + 1);
            let top1 = ranked
                .first()
                .is_some_and(|&(x, _, _, _, _)| q.is_target(x));
            (hit, ranked.len(), rank, top1)
        })
        .collect();
    let hits = per.iter().filter(|x| x.0).count();
    let pool_sizes: Vec<usize> = per.iter().map(|x| x.1).collect();
    let (mean_pool, _, p90_pool, _) = set_size_summary(&pool_sizes);
    let mrr = per
        .iter()
        .filter_map(|x| x.2.map(|rank| 1.0 / rank as f64))
        .sum::<f64>()
        / idx.len() as f64;
    let h10 = per
        .iter()
        .filter(|x| x.2.is_some_and(|rank| rank <= 10))
        .count() as f64
        / idx.len() as f64;
    let top1 = per.iter().filter(|x| x.3).count() as f64 / idx.len() as f64;
    RetrievalMetrics {
        recall: hits as f64 / idx.len() as f64,
        mean_pool,
        p90_pool,
        mrr,
        hits10: h10,
        top1_dca: top1,
    }
}

fn scored_frontier_pool(
    q: &CQuery,
    frontier: &DirectFrontier,
    extra_hops: usize,
    weights: &[f32; FRONTIER_FEATURES],
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32, usize, usize, usize)> = frontier
        .pool(q.a, q.b, extra_hops)
        .expect("query ids come from the same ontology as the direct frontier")
        .iter()
        .map(|candidate| {
            (
                candidate.concept,
                dot_frontier(weights, &frontier_features(candidate, frontier)),
                candidate.frontier_depth,
                candidate.path_len,
                candidate.concept,
            )
        })
        .collect();
    scored.sort_by(|p, r| {
        r.1.total_cmp(&p.1)
            .then_with(|| p.2.cmp(&r.2))
            .then_with(|| p.3.cmp(&r.3))
            .then_with(|| p.4.cmp(&r.4))
    });
    scored
        .into_iter()
        .map(|(x, score, _, _, _)| (x, score))
        .collect()
}

fn learned_nonconformity(q: &CQuery, scored: &[(usize, f32)]) -> f32 {
    let Some(best_score) = scored.first().map(|&(_, score)| score) else {
        return f32::INFINITY;
    };
    let Some(target_score) = scored
        .iter()
        .filter(|&&(x, _)| q.is_target(x))
        .map(|&(_, score)| score)
        .max_by(f32::total_cmp)
    else {
        let worst_score = scored.last().map_or(best_score, |&(_, score)| score);
        return (best_score - worst_score).abs() + 1.0;
    };
    best_score - target_score
}

fn learned_answer_set(scored: &[(usize, f32)], qhat: f32) -> Vec<(usize, f32)> {
    let Some(best_score) = scored.first().map(|&(_, score)| score) else {
        return Vec::new();
    };
    let cutoff = best_score - qhat;
    scored
        .iter()
        .copied()
        .filter(|&(_, score)| score >= cutoff)
        .collect()
}

struct LearnedConformalResult {
    alpha: f32,
    q_hat: f32,
    empirical: f64,
    pool_recall: f64,
    mean_set: f32,
    p50_set: usize,
    p90_set: usize,
    max_set: usize,
}

fn learned_conformal_results(
    queries: &[CQuery],
    frontier: &DirectFrontier,
    extra_hops: usize,
    weights: &[f32; FRONTIER_FEATURES],
    cal_idx: &[usize],
    test_idx: &[usize],
) -> (Vec<f32>, Vec<LearnedConformalResult>) {
    let cal_scores: Vec<(usize, Vec<(usize, f32)>)> = cal_idx
        .iter()
        .map(|&i| {
            (
                i,
                scored_frontier_pool(&queries[i], frontier, extra_hops, weights),
            )
        })
        .collect();
    let test_scores: Vec<(usize, Vec<(usize, f32)>)> = test_idx
        .iter()
        .map(|&i| {
            (
                i,
                scored_frontier_pool(&queries[i], frontier, extra_hops, weights),
            )
        })
        .collect();
    let cal_nonconf: Vec<f32> = cal_scores
        .iter()
        .map(|(i, scored)| learned_nonconformity(&queries[*i], scored))
        .collect();
    let pool_recall = test_scores
        .iter()
        .filter(|(i, scored)| scored.iter().any(|&(x, _)| queries[*i].is_target(x)))
        .count() as f64
        / test_scores.len() as f64;
    let results = [0.3f32, 0.2, 0.1]
        .into_iter()
        .map(|alpha| {
            let threshold = calibrate_scores(&cal_nonconf, alpha).expect("non-empty calibration");
            let mut covered = 0usize;
            let mut sizes = Vec::with_capacity(test_scores.len());
            for (i, scored) in &test_scores {
                let set = learned_answer_set(scored, threshold.qhat);
                if set.iter().any(|&(x, _)| queries[*i].is_target(x)) {
                    covered += 1;
                }
                sizes.push(set.len());
            }
            let empirical = covered as f64 / test_scores.len() as f64;
            let (mean_set, p50_set, p90_set, max_set) = set_size_summary(&sizes);
            LearnedConformalResult {
                alpha,
                q_hat: threshold.qhat,
                empirical,
                pool_recall,
                mean_set,
                p50_set,
                p90_set,
                max_set,
            }
        })
        .collect();
    (cal_nonconf, results)
}

fn report_learned_conformal(
    queries: &[CQuery],
    frontier: &DirectFrontier,
    extra_hops: usize,
    weights: &[f32; FRONTIER_FEATURES],
    param: &str,
    metrics: &mut MetricsCsv,
) {
    if queries.len() < 8 {
        println!("\n[learned conformal] skipped: need at least 8 queries");
        return;
    }
    let order = query_order_indices(queries);
    let train_end = order.len() / 2;
    let cal_end = train_end + (order.len() - train_end) / 2;
    let cal_idx = &order[train_end..cal_end];
    let test_idx = &order[cal_end..];
    if cal_idx.is_empty() || test_idx.is_empty() {
        println!("\n[learned conformal] skipped: empty calibration or test split");
        return;
    }
    let (cal_nonconf, results) =
        learned_conformal_results(queries, frontier, extra_hops, weights, cal_idx, test_idx);
    println!(
        "\n[learned conformal] direct-frontier ranker sets (extra={extra_hops}, {} calibration, {} test)",
        cal_idx.len(),
        test_idx.len()
    );
    print_f32_distribution(
        "learned ranker calibration nonconformity (best - target)",
        &cal_nonconf,
    );
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "alpha", "1-alpha", "q_hat", "empirical", "poolrecall", "mean|set|", "p50", "p90", "max"
    );
    for result in results {
        println!(
            "{:<8.2} {:>8.2} {:>10.3} {:>10.3} {:>10.3} {:>10.1} {:>8} {:>8} {:>8}",
            result.alpha,
            1.0 - result.alpha,
            result.q_hat,
            result.empirical,
            result.pool_recall,
            result.mean_set,
            result.p50_set,
            result.p90_set,
            result.max_set
        );
        emit_conformal_metrics(
            metrics,
            "learned_frontier_conformal",
            "pairwise_linear",
            param,
            result.alpha,
            ConformalMetrics {
                q_hat: result.q_hat,
                empirical: result.empirical,
                pool_recall: result.pool_recall,
                mean_set: result.mean_set,
                p50_set: result.p50_set,
                p90_set: result.p90_set,
                max_set: result.max_set,
            },
        );
    }
}

fn report_repeated_learned_conformal(
    queries: &[CQuery],
    frontier: &DirectFrontier,
    config: LearnedRankerConfig,
    metrics: &mut MetricsCsv,
) {
    if config.repeats <= 1 || queries.len() < 8 {
        return;
    }
    let mut rows: Vec<Vec<LearnedConformalResult>> = Vec::with_capacity(config.repeats);
    for repeat in 0..config.repeats {
        let order = query_order_indices_with_salt(queries, repeat + 1);
        let train_end = order.len() / 2;
        let cal_end = train_end + (order.len() - train_end) / 2;
        let train_idx = &order[..train_end];
        let cal_idx = &order[train_end..cal_end];
        let test_idx = &order[cal_end..];
        if train_idx.is_empty() || cal_idx.is_empty() || test_idx.is_empty() {
            continue;
        }
        let (weights, _, _) = train_frontier_ranker(
            queries,
            train_idx,
            frontier,
            config.extra_hops,
            config.epochs,
            config.lr,
            config.l2,
        );
        let (_, results) = learned_conformal_results(
            queries,
            frontier,
            config.extra_hops,
            &weights,
            cal_idx,
            test_idx,
        );
        rows.push(results);
    }
    if rows.is_empty() {
        println!("\n[learned conformal repeats] skipped: empty repeated splits");
        return;
    }
    println!(
        "\n[learned conformal repeats] {}/{} deterministic train/cal/test splits (extra={extra_hops}, epochs={epochs})",
        rows.len(),
        config.repeats,
        extra_hops = config.extra_hops,
        epochs = config.epochs
    );
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "alpha", "cov_mean", "cov_min", "cov_max", "set_mean", "recall_min"
    );
    for alpha in [0.3f32, 0.2, 0.1] {
        let mut coverages = Vec::new();
        let mut mean_sets = Vec::new();
        let mut recalls = Vec::new();
        for repeat_results in &rows {
            if let Some(result) = repeat_results.iter().find(|result| result.alpha == alpha) {
                coverages.push(result.empirical as f32);
                mean_sets.push(result.mean_set);
                recalls.push(result.pool_recall as f32);
            }
        }
        if coverages.is_empty() {
            continue;
        }
        let coverage_mean = coverages.iter().sum::<f32>() / coverages.len() as f32;
        let set_mean = mean_sets.iter().sum::<f32>() / mean_sets.len() as f32;
        let coverage_min = coverages.iter().copied().fold(f32::INFINITY, f32::min);
        let coverage_max = coverages.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let recall_min = recalls.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "{alpha:<8.2} {coverage_mean:>10.3} {coverage_min:>10.3} {coverage_max:>10.3} {set_mean:>10.1} {recall_min:>10.3}"
        );
        let param = format!(
            "extra={};epochs={};lr={};l2={};repeats={}",
            config.extra_hops, config.epochs, config.lr, config.l2, config.repeats
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "coverage_mean",
            coverage_mean,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "coverage_min",
            coverage_min,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "coverage_max",
            coverage_max,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "mean_set_mean",
            set_mean,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "pool_recall_min",
            recall_min,
        );
    }
}

pub(super) fn report_learned_frontier_ranker(
    queries: &[CQuery],
    frontier: &DirectFrontier,
    metrics: &mut MetricsCsv,
) {
    if queries.len() < 4 {
        println!("\n[learned] skipped direct-frontier ranker: need at least 4 queries");
        return;
    }
    let extra_hops: usize = std::env::var("LEARNED_EXTRA_HOPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let epochs: usize = std::env::var("LEARNED_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let lr: f32 = std::env::var("LEARNED_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.03);
    let l2: f32 = std::env::var("LEARNED_L2")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-4);
    let repeats: usize = std::env::var("LEARNED_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let (train_idx, test_idx) = query_split_indices(queries);
    let (weights, losses, pairs_per_epoch) =
        train_frontier_ranker(queries, &train_idx, frontier, extra_hops, epochs, lr, l2);
    println!(
        "\n[learned] direct-frontier pairwise ranker (graph features only; extra={extra_hops}, train={}, test={}, epochs={epochs}, lr={lr}, l2={l2})",
        train_idx.len(),
        test_idx.len()
    );
    println!(
        "[learned] feature order: bias, frontier_depth, path_len, imbalance, ln1p(parent_count)"
    );
    println!("[learned] weights: {weights:?}");
    println!("[learned] pairwise training pairs per epoch: {pairs_per_epoch}");
    print_f32_distribution("learned frontier ranker loss", &losses);
    println!(
        "{:<18} {:<14} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "split", "order", "recall", "mean|pool|", "p90|pool|", "MRR", "Hits@10", "top1DCA"
    );
    let rows = [
        (
            "train",
            "bfs_order",
            evaluate_frontier_ranker(queries, &train_idx, frontier, extra_hops, None),
        ),
        (
            "train",
            "pairwise_linear",
            evaluate_frontier_ranker(queries, &train_idx, frontier, extra_hops, Some(&weights)),
        ),
        (
            "test",
            "bfs_order",
            evaluate_frontier_ranker(queries, &test_idx, frontier, extra_hops, None),
        ),
        (
            "test",
            "pairwise_linear",
            evaluate_frontier_ranker(queries, &test_idx, frontier, extra_hops, Some(&weights)),
        ),
    ];
    let param = format!(
        "extra={extra_hops};epochs={epochs};lr={lr};l2={l2};train={};test={}",
        train_idx.len(),
        test_idx.len()
    );
    for (split, label, values) in rows {
        println!(
            "{split:<18} {label:<14} {:>8.3} {:>10.1} {:>10} {:>8.3} {:>8.3} {:>8.3}",
            values.recall,
            values.mean_pool,
            values.p90_pool,
            values.mrr,
            values.hits10,
            values.top1_dca
        );
        emit_retrieval_metrics(
            metrics,
            "learned_frontier_ranker",
            &format!("{split}_{label}"),
            &param,
            values,
        );
    }
    report_learned_conformal(queries, frontier, extra_hops, &weights, &param, metrics);
    report_repeated_learned_conformal(
        queries,
        frontier,
        LearnedRankerConfig {
            extra_hops,
            epochs,
            lr,
            l2,
            repeats,
        },
        metrics,
    );
}
