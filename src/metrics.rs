//! Rank-based evaluation metrics for link prediction and knowledge graph tasks.
//!
//! Ranks are 1-indexed.

/// Mean Reciprocal Rank: average of `1/rank` over all ranks.
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64
}

/// Hits@k: fraction of ranks that are `<= k`.
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().filter(|&&r| r <= k).count() as f64 / ranks.len() as f64
}

/// Mean Rank: arithmetic mean of all ranks. Lower is better.
pub fn mean_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
}

/// Adjusted Mean Rank (AMR): `mean_rank / expected_random_mean_rank`.
///
/// The expected mean rank under a uniform random model is `(num_entities + 1) / 2`.
/// AMR < 1.0 means better than random; AMR = 1.0 means random performance.
pub fn adjusted_mean_rank(ranks: &[usize], num_entities: usize) -> f64 {
    if ranks.is_empty() || num_entities == 0 {
        return 0.0;
    }
    let mr = mean_rank(ranks);
    let expected = (num_entities as f64 + 1.0) / 2.0;
    mr / expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrr_basic() {
        let ranks = vec![1, 2, 4];
        let mrr = mean_reciprocal_rank(&ranks);
        // (1/1 + 1/2 + 1/4) / 3 = 1.75 / 3 ≈ 0.5833
        assert!((mrr - 0.5833).abs() < 0.001);
    }

    #[test]
    fn hits_at_k_basic() {
        let ranks = vec![1, 2, 5, 10, 20];
        assert!((hits_at_k(&ranks, 10) - 0.8).abs() < 1e-9);
        assert!((hits_at_k(&ranks, 1) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn mean_rank_basic() {
        let ranks = vec![1, 3, 5];
        assert!((mean_rank(&ranks) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn adjusted_mean_rank_basic() {
        // 100 entities, expected MR = 50.5
        let ranks = vec![1, 1, 1]; // MR = 1.0
        let amr = adjusted_mean_rank(&ranks, 100);
        assert!((amr - 1.0 / 50.5).abs() < 1e-9);
    }

    #[test]
    fn empty_ranks() {
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
        assert_eq!(hits_at_k(&[], 10), 0.0);
        assert_eq!(mean_rank(&[]), 0.0);
        assert_eq!(adjusted_mean_rank(&[], 100), 0.0);
    }
}
