use super::{
    emit_retrieval_metrics, print_f32_distribution, set_size_summary, CQuery, MetricsCsv,
    RetrievalMetrics,
};
use rayon::prelude::*;
use subsume::clqa::{DirectFrontier, FrontierCandidate};

const FRONTIER_FEATURES: usize = 5;

fn query_split_indices(queries: &[CQuery]) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&i| {
        let q = &queries[i];
        q.a.wrapping_mul(2_654_435_761)
            .wrapping_add(q.b.wrapping_mul(40_503))
            .wrapping_add(i.wrapping_mul(97_531))
    });
    let split = order.len() / 2;
    (order[..split].to_vec(), order[split..].to_vec())
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
}
