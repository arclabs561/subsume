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

#[derive(Clone, Copy, Debug)]
struct CandidateScore {
    concept: usize,
    score: f32,
    frontier_depth: usize,
    path_len: usize,
    parent_count: usize,
}

struct ConformalCase<'a> {
    query_index: usize,
    query: &'a CQuery,
    scored: &'a [CandidateScore],
    set: &'a [CandidateScore],
}

struct LearnedCaseSource<'a> {
    queries: &'a [CQuery],
    frontier: &'a DirectFrontier,
    extra_hops: usize,
    weights: &'a [f32; FRONTIER_FEATURES],
    test_idx: &'a [usize],
    results: &'a [LearnedConformalResult],
    base_param: &'a str,
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

fn scored_frontier_pool_details(
    q: &CQuery,
    frontier: &DirectFrontier,
    extra_hops: usize,
    weights: &[f32; FRONTIER_FEATURES],
) -> Vec<CandidateScore> {
    let mut scored: Vec<CandidateScore> = frontier
        .pool(q.a, q.b, extra_hops)
        .expect("query ids come from the same ontology as the direct frontier")
        .iter()
        .map(|candidate| {
            let parent_count = frontier
                .parents(candidate.concept)
                .expect("frontier candidate concept came from the same frontier")
                .len();
            CandidateScore {
                concept: candidate.concept,
                score: dot_frontier(weights, &frontier_features(candidate, frontier)),
                frontier_depth: candidate.frontier_depth,
                path_len: candidate.path_len,
                parent_count,
            }
        })
        .collect();
    scored.sort_by(|p, r| {
        r.score
            .total_cmp(&p.score)
            .then_with(|| p.frontier_depth.cmp(&r.frontier_depth))
            .then_with(|| p.path_len.cmp(&r.path_len))
            .then_with(|| p.concept.cmp(&r.concept))
    });
    scored
}

fn learned_nonconformity(q: &CQuery, scored: &[CandidateScore]) -> f32 {
    let Some(best_score) = scored.first().map(|candidate| candidate.score) else {
        return f32::INFINITY;
    };
    let Some(target_score) = scored
        .iter()
        .filter(|candidate| q.is_target(candidate.concept))
        .map(|candidate| candidate.score)
        .max_by(f32::total_cmp)
    else {
        let worst_score = scored
            .last()
            .map_or(best_score, |candidate| candidate.score);
        return (best_score - worst_score).abs() + 1.0;
    };
    best_score - target_score
}

fn learned_answer_set(scored: &[CandidateScore], qhat: f32) -> Vec<CandidateScore> {
    let Some(best_score) = scored.first().map(|candidate| candidate.score) else {
        return Vec::new();
    };
    let cutoff = best_score - qhat;
    scored
        .iter()
        .copied()
        .filter(|candidate| candidate.score >= cutoff)
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
    let cal_scores: Vec<(usize, Vec<CandidateScore>)> = cal_idx
        .iter()
        .map(|&i| {
            (
                i,
                scored_frontier_pool_details(&queries[i], frontier, extra_hops, weights),
            )
        })
        .collect();
    let test_scores: Vec<(usize, Vec<CandidateScore>)> = test_idx
        .iter()
        .map(|&i| {
            (
                i,
                scored_frontier_pool_details(&queries[i], frontier, extra_hops, weights),
            )
        })
        .collect();
    let cal_nonconf: Vec<f32> = cal_scores
        .iter()
        .map(|(i, scored)| learned_nonconformity(&queries[*i], scored))
        .collect();
    let pool_recall = test_scores
        .iter()
        .filter(|(i, scored)| {
            scored
                .iter()
                .any(|candidate| queries[*i].is_target(candidate.concept))
        })
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
                if set
                    .iter()
                    .any(|candidate| queries[*i].is_target(candidate.concept))
                {
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

fn learned_case_limit() -> usize {
    std::env::var("LEARNED_CASE_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

fn learned_conformal_param(base_param: &str, cal_len: usize, test_len: usize) -> String {
    format!("{base_param};cal={cal_len};conformal_test={test_len}")
}

fn repeat_param(
    config: LearnedRankerConfig,
    repeat: usize,
    ranker_train_len: usize,
    cal_len: usize,
    conformal_test_len: usize,
) -> String {
    format!(
        "extra={};epochs={};lr={};l2={};split=rotated_hash;repeat={};ranker_train={};cal={};conformal_test={}",
        config.extra_hops,
        config.epochs,
        config.lr,
        config.l2,
        repeat,
        ranker_train_len,
        cal_len,
        conformal_test_len
    )
}

fn repeat_summary_param(config: LearnedRankerConfig, effective_repeats: usize) -> String {
    format!(
        "extra={};epochs={};lr={};l2={};split=rotated_hash;repeats={};effective_repeats={}",
        config.extra_hops, config.epochs, config.lr, config.l2, config.repeats, effective_repeats
    )
}

fn mean_sample_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / (values.len() - 1) as f32;
    (mean, variance.sqrt())
}

fn emit_optional_usize(
    metrics: &mut MetricsCsv,
    label: &str,
    param: &str,
    alpha: f32,
    metric: &str,
    value: Option<usize>,
) {
    if let Some(value) = value {
        metrics.emit(
            "learned_frontier_conformal_case",
            label,
            param,
            Some(alpha),
            metric,
            value,
        );
    }
}

fn emit_optional_f32(
    metrics: &mut MetricsCsv,
    label: &str,
    param: &str,
    alpha: f32,
    metric: &str,
    value: Option<f32>,
) {
    if let Some(value) = value {
        metrics.emit(
            "learned_frontier_conformal_case",
            label,
            param,
            Some(alpha),
            metric,
            value,
        );
    }
}

fn emit_case_metric(
    metrics: &mut MetricsCsv,
    label: &str,
    param: &str,
    alpha: f32,
    metric: &str,
    value: impl std::fmt::Display,
) {
    metrics.emit(
        "learned_frontier_conformal_case",
        label,
        param,
        Some(alpha),
        metric,
        value,
    );
}

fn emit_one_conformal_case(
    metrics: &mut MetricsCsv,
    label: &str,
    base_param: &str,
    alpha: f32,
    case: ConformalCase<'_>,
) {
    let param = format!(
        "{base_param};query={query_index};a={};b={}",
        case.query.a,
        case.query.b,
        query_index = case.query_index
    );
    let pool_hit = case
        .scored
        .iter()
        .any(|candidate| case.query.is_target(candidate.concept));
    let covered = case
        .set
        .iter()
        .any(|candidate| case.query.is_target(candidate.concept));
    let best_target = case
        .scored
        .iter()
        .enumerate()
        .find(|(_, candidate)| case.query.is_target(candidate.concept));
    let top = case.scored.first();
    let target_recovered = case
        .set
        .iter()
        .filter(|candidate| case.query.is_target(candidate.concept))
        .count();
    let target_set_recall = if case.query.lcas.is_empty() {
        0.0
    } else {
        target_recovered as f32 / case.query.lcas.len() as f32
    };
    let score_gap = top
        .zip(best_target)
        .map(|(top, (_, target))| top.score - target.score);

    emit_case_metric(
        metrics,
        label,
        &param,
        alpha,
        "covered",
        usize::from(covered),
    );
    emit_case_metric(
        metrics,
        label,
        &param,
        alpha,
        "pool_hit",
        usize::from(pool_hit),
    );
    emit_case_metric(
        metrics,
        label,
        &param,
        alpha,
        "pool_size",
        case.scored.len(),
    );
    emit_case_metric(metrics, label, &param, alpha, "set_size", case.set.len());
    emit_case_metric(
        metrics,
        label,
        &param,
        alpha,
        "target_count",
        case.query.lcas.len(),
    );
    emit_case_metric(
        metrics,
        label,
        &param,
        alpha,
        "target_set_recall",
        target_set_recall,
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "target_rank",
        best_target.map(|(rank, _)| rank + 1),
    );
    emit_optional_f32(metrics, label, &param, alpha, "target_score_gap", score_gap);
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "top_frontier_depth",
        top.map(|candidate| candidate.frontier_depth),
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "target_frontier_depth",
        best_target.map(|(_, candidate)| candidate.frontier_depth),
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "top_path_len",
        top.map(|candidate| candidate.path_len),
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "target_path_len",
        best_target.map(|(_, candidate)| candidate.path_len),
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "top_parent_count",
        top.map(|candidate| candidate.parent_count),
    );
    emit_optional_usize(
        metrics,
        label,
        &param,
        alpha,
        "target_parent_count",
        best_target.map(|(_, candidate)| candidate.parent_count),
    );
}

fn emit_learned_case_metrics(source: LearnedCaseSource<'_>, metrics: &mut MetricsCsv) {
    if !metrics.enabled() {
        return;
    }
    let limit = learned_case_limit();
    if limit == 0 {
        return;
    }
    let covered_limit = limit;
    let scored: Vec<(usize, Vec<CandidateScore>)> = source
        .test_idx
        .iter()
        .map(|&i| {
            (
                i,
                scored_frontier_pool_details(
                    &source.queries[i],
                    source.frontier,
                    source.extra_hops,
                    source.weights,
                ),
            )
        })
        .collect();
    for result in source.results {
        let mut missed = 0usize;
        let mut covered = 0usize;
        for (query_index, scored) in &scored {
            let query = &source.queries[*query_index];
            let set = learned_answer_set(scored, result.q_hat);
            let is_covered = set
                .iter()
                .any(|candidate| query.is_target(candidate.concept));
            if !is_covered && missed < limit {
                emit_one_conformal_case(
                    metrics,
                    "missed",
                    source.base_param,
                    result.alpha,
                    ConformalCase {
                        query_index: *query_index,
                        query,
                        scored,
                        set: &set,
                    },
                );
                missed += 1;
            } else if is_covered && covered < covered_limit {
                emit_one_conformal_case(
                    metrics,
                    "covered",
                    source.base_param,
                    result.alpha,
                    ConformalCase {
                        query_index: *query_index,
                        query,
                        scored,
                        set: &set,
                    },
                );
                covered += 1;
            }
            if missed >= limit && covered >= covered_limit {
                break;
            }
        }
    }
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
    let conformal_param = learned_conformal_param(param, cal_idx.len(), test_idx.len());
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
    emit_learned_case_metrics(
        LearnedCaseSource {
            queries,
            frontier,
            extra_hops,
            weights,
            test_idx,
            results: &results,
            base_param: &conformal_param,
        },
        metrics,
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
            &conformal_param,
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
        let repeat_row_param = repeat_param(
            config,
            repeat + 1,
            train_idx.len(),
            cal_idx.len(),
            test_idx.len(),
        );
        for result in &results {
            emit_conformal_metrics(
                metrics,
                "learned_frontier_conformal_repeat",
                "pairwise_linear",
                &repeat_row_param,
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
        let mut qhats = Vec::new();
        for repeat_results in &rows {
            if let Some(result) = repeat_results.iter().find(|result| result.alpha == alpha) {
                qhats.push(result.q_hat);
            }
        }
        let (coverage_mean, coverage_std) = mean_sample_std(&coverages);
        let (set_mean, set_std) = mean_sample_std(&mean_sets);
        let (recall_mean, recall_std) = mean_sample_std(&recalls);
        let (qhat_mean, qhat_std) = mean_sample_std(&qhats);
        let coverage_min = coverages.iter().copied().fold(f32::INFINITY, f32::min);
        let coverage_max = coverages.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let recall_min = recalls.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "{alpha:<8.2} {coverage_mean:>10.3} {coverage_min:>10.3} {coverage_max:>10.3} {set_mean:>10.1} {recall_min:>10.3}"
        );
        let param = repeat_summary_param(config, rows.len());
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
            "coverage_sample_std",
            coverage_std,
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
            "mean_set_sample_std",
            set_std,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "pool_recall_mean",
            recall_mean,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "pool_recall_min",
            recall_min,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "pool_recall_sample_std",
            recall_std,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "q_hat_mean",
            qhat_mean,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "q_hat_sample_std",
            qhat_std,
        );
        metrics.emit(
            "learned_frontier_conformal_repeats",
            "pairwise_linear",
            &param,
            Some(alpha),
            "split_count",
            coverages.len(),
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
        "extra={extra_hops};epochs={epochs};lr={lr};l2={l2};ranker_train={};ranker_test={}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn query(a: usize, b: usize) -> CQuery {
        CQuery {
            a,
            b,
            common: HashSet::new(),
            lcas: vec![0],
        }
    }

    #[test]
    fn query_split_indices_partition_queries() {
        let queries: Vec<CQuery> = (0..12).map(|i| query(i, i + 100)).collect();
        let (train, test) = query_split_indices(&queries);
        assert_eq!(train.len(), 6);
        assert_eq!(test.len(), 6);

        let mut all = train.clone();
        all.extend(test.iter().copied());
        all.sort_unstable();
        assert_eq!(all, (0..queries.len()).collect::<Vec<_>>());

        let train_set: HashSet<usize> = train.into_iter().collect();
        assert!(test.iter().all(|i| !train_set.contains(i)));
    }

    #[test]
    fn salted_query_orders_keep_membership_and_change_order() {
        let queries: Vec<CQuery> = (0..12).map(|i| query(i, i + 100)).collect();
        let base = query_order_indices_with_salt(&queries, 0);
        let salted = query_order_indices_with_salt(&queries, 1);
        assert_ne!(base, salted);

        for order in [base, salted, query_order_indices_with_salt(&queries, 2)] {
            let mut sorted = order;
            sorted.sort_unstable();
            assert_eq!(sorted, (0..queries.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn learned_answer_set_grows_with_nonconformity_threshold() {
        let scored = vec![
            CandidateScore {
                concept: 1,
                score: 3.0,
                frontier_depth: 1,
                path_len: 2,
                parent_count: 1,
            },
            CandidateScore {
                concept: 2,
                score: 2.5,
                frontier_depth: 1,
                path_len: 2,
                parent_count: 1,
            },
            CandidateScore {
                concept: 3,
                score: 1.0,
                frontier_depth: 2,
                path_len: 4,
                parent_count: 2,
            },
        ];

        assert_eq!(
            learned_answer_set(&scored, 0.0)
                .iter()
                .map(|candidate| candidate.concept)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            learned_answer_set(&scored, 0.5)
                .iter()
                .map(|candidate| candidate.concept)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn learned_conformal_param_keeps_ranker_and_conformal_splits_separate() {
        let param =
            learned_conformal_param("extra=10;epochs=20;ranker_train=6;ranker_test=6", 3, 3);
        assert_eq!(
            param,
            "extra=10;epochs=20;ranker_train=6;ranker_test=6;cal=3;conformal_test=3"
        );
    }

    #[test]
    fn repeat_params_record_effective_split_shape() {
        let config = LearnedRankerConfig {
            extra_hops: 10,
            epochs: 20,
            lr: 0.03,
            l2: 1e-4,
            repeats: 5,
        };
        assert_eq!(
            repeat_param(config, 2, 6, 3, 3),
            "extra=10;epochs=20;lr=0.03;l2=0.0001;split=rotated_hash;repeat=2;ranker_train=6;cal=3;conformal_test=3"
        );
        assert_eq!(
            repeat_summary_param(config, 4),
            "extra=10;epochs=20;lr=0.03;l2=0.0001;split=rotated_hash;repeats=5;effective_repeats=4"
        );
    }

    #[test]
    fn mean_sample_std_reports_repeat_dispersion() {
        assert_eq!(mean_sample_std(&[]), (0.0, 0.0));
        assert_eq!(mean_sample_std(&[2.0]), (2.0, 0.0));

        let (mean, std) = mean_sample_std(&[1.0, 3.0, 5.0]);
        assert_eq!(mean, 3.0);
        assert!((std - 2.0).abs() < 1e-6);
    }
}
