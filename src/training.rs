//! Rank-based evaluation metrics for link prediction and knowledge graph tasks.

/// Rank-based evaluation metrics for link prediction and knowledge graph tasks.
///
/// These metrics are the standard evaluation suite for knowledge graph embedding
/// quality on downstream tasks like knowledge graph completion, where we need to
/// rank candidate entities.
pub mod metrics {
    /// Compute Mean Reciprocal Rank (MRR) for a set of queries.
    ///
    /// MRR averages the reciprocal of the rank at which the first correct answer
    /// appears across all queries. It gives exponentially more weight to top
    /// rankings: rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1.
    ///
    /// # Parameters
    ///
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers.
    ///   Rank 0 is treated as "not found" and contributes 0 to the sum.
    ///
    /// # Returns
    ///
    /// MRR value in \[0, 1\], where higher is better. Returns 0.0 for empty input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::training::metrics::mean_reciprocal_rank;
    ///
    /// let ranks = vec![1, 3, 2, 5]; // First correct answer at positions 1, 3, 2, 5
    /// let mrr = mean_reciprocal_rank(ranks.iter().copied());
    /// // (1/1 + 1/3 + 1/2 + 1/5) / 4 ≈ 0.5083
    /// assert!(mrr > 0.5 && mrr < 0.52);
    /// ```
    pub fn mean_reciprocal_rank<I>(ranks: I) -> f32
    where
        I: Iterator<Item = usize>,
    {
        let mut sum = 0.0;
        let mut count = 0;

        for rank in ranks {
            if rank > 0 {
                sum += 1.0 / (rank as f32);
            }
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Compute Hits@K metric: fraction of queries where the correct answer appears in top K.
    ///
    /// # Parameters
    ///
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers.
    ///   Rank 0 is treated as "not found" (always a miss).
    /// - `k`: Top-K threshold
    ///
    /// # Returns
    ///
    /// Hits@K value in \[0, 1\], where higher is better. Returns 0.0 for empty input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::training::metrics::hits_at_k;
    ///
    /// let ranks = vec![1, 3, 2, 5, 10];
    /// let hits_3 = hits_at_k(ranks.iter().copied(), 3);
    /// assert_eq!(hits_3, 0.6); // 3 out of 5 queries have rank <= 3
    /// ```
    pub fn hits_at_k<I>(ranks: I, k: usize) -> f32
    where
        I: Iterator<Item = usize>,
    {
        let mut hits = 0;
        let mut count = 0;

        for rank in ranks {
            if rank > 0 && rank <= k {
                hits += 1;
            }
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            hits as f32 / count as f32
        }
    }

    /// Compute Mean Rank (MR): average rank of correct answers.
    ///
    /// # Parameters
    ///
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers.
    ///   Rank 0 contributes 0 to the sum but still counts toward the denominator.
    ///
    /// # Returns
    ///
    /// Mean rank value in \[0, inf), where lower is better. Returns 0.0 for empty input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::training::metrics::mean_rank;
    ///
    /// let ranks = vec![1, 3, 2, 5];
    /// let mr = mean_rank(ranks.iter().copied());
    /// assert_eq!(mr, 2.75); // (1 + 3 + 2 + 5) / 4
    /// ```
    pub fn mean_rank<I>(ranks: I) -> f32
    where
        I: Iterator<Item = usize>,
    {
        let mut sum = 0.0;
        let mut count = 0;

        for rank in ranks {
            if rank > 0 {
                sum += rank as f32;
            }
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Compute Normalized Discounted Cumulative Gain (nDCG) for ranking.
    ///
    /// nDCG measures ranking quality by considering both relevance and position.
    /// Higher positions contribute more to the score via logarithmic discounting.
    ///
    /// # Parameters
    ///
    /// - `relevance_scores`: Iterator of relevance scores for items in ranked order
    /// - `ideal_relevance`: Iterator of ideal relevance scores (sorted descending)
    ///
    /// # Returns
    ///
    /// nDCG value in \[0, 1\], where higher is better. Returns 0.0 when ideal DCG is zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::training::metrics::ndcg;
    ///
    /// let ranked = vec![0.9, 0.5, 0.8, 0.2]; // Actual ranking
    /// let ideal = vec![0.9, 0.8, 0.5, 0.2];  // Ideal ranking
    /// let score = ndcg(ranked.iter().copied(), ideal.iter().copied());
    /// assert!(score > 0.9); // Good ranking quality
    /// ```
    pub fn ndcg<I, J>(relevance_scores: I, ideal_relevance: J) -> f32
    where
        I: Iterator<Item = f32>,
        J: Iterator<Item = f32>,
    {
        let mut dcg = 0.0;
        let mut idcg = 0.0;
        let mut position = 1;

        let mut relevance_iter = relevance_scores.peekable();
        let mut ideal_iter = ideal_relevance.peekable();

        while relevance_iter.peek().is_some() && ideal_iter.peek().is_some() {
            let rel = relevance_iter.next().unwrap_or(0.0);
            let ideal_rel = ideal_iter.next().unwrap_or(0.0);

            // DCG: sum of (relevance / log2(position + 1))
            dcg += rel / ((position + 1) as f32).log2();
            idcg += ideal_rel / ((position + 1) as f32).log2();

            position += 1;
        }

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_reciprocal_rank() {
        let ranks = [1, 3, 2, 5];
        let mrr = metrics::mean_reciprocal_rank(ranks.iter().copied());
        // (1/1 + 1/3 + 1/2 + 1/5) / 4 ≈ 0.5083
        assert!((mrr - 0.5083).abs() < 1e-3);
    }

    #[test]
    fn test_hits_at_k() {
        let ranks = [1, 3, 2, 5, 10];
        let hits_3 = metrics::hits_at_k(ranks.iter().copied(), 3);
        assert_eq!(hits_3, 0.6);
    }

    #[test]
    fn test_mean_rank() {
        let ranks = [1, 3, 2, 5];
        let mr = metrics::mean_rank(ranks.iter().copied());
        assert_eq!(mr, 2.75);
    }

    #[test]
    fn test_ndcg() {
        let ranked = [0.9, 0.5, 0.8, 0.2];
        let ideal = [0.9, 0.8, 0.5, 0.2];
        let score = metrics::ndcg(ranked.iter().copied(), ideal.iter().copied());
        assert!(score > 0.9);
    }

    #[test]
    fn test_empty_metrics() {
        let empty_mrr = metrics::mean_reciprocal_rank(std::iter::empty());
        assert_eq!(empty_mrr, 0.0);

        let empty_hits = metrics::hits_at_k(std::iter::empty(), 10);
        assert_eq!(empty_hits, 0.0);

        let empty_mr = metrics::mean_rank(std::iter::empty());
        assert_eq!(empty_mr, 0.0);
    }

    #[test]
    fn test_edge_cases_zero_rank() {
        // Rank of 0 should be handled gracefully
        let ranks = [0, 1, 2, 0, 3];
        let mrr = metrics::mean_reciprocal_rank(ranks.iter().copied());
        // 0 contributes nothing to sum but counts: (0 + 1/1 + 1/2 + 0 + 1/3) / 5
        assert!(mrr > 0.0 && mrr < 1.0);
    }

    #[test]
    fn test_mrr_single_element() {
        let mrr = metrics::mean_reciprocal_rank([1].iter().copied());
        assert_eq!(mrr, 1.0);

        let mrr5 = metrics::mean_reciprocal_rank([5].iter().copied());
        assert!((mrr5 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_mrr_all_perfect() {
        let mrr = metrics::mean_reciprocal_rank([1, 1, 1, 1].iter().copied());
        assert_eq!(mrr, 1.0);
    }

    #[test]
    fn test_mrr_all_zero_rank() {
        let mrr = metrics::mean_reciprocal_rank([0, 0, 0, 0].iter().copied());
        assert_eq!(mrr, 0.0);
    }

    #[test]
    fn test_mean_rank_with_zero_ranks() {
        // ranks = [0, 2, 4, 0] => sum = 0 + 2 + 4 + 0 = 6, count = 4, mean = 1.5
        let mr = metrics::mean_rank([0, 2, 4, 0].iter().copied());
        assert!((mr - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_rank_single_element() {
        let mr = metrics::mean_rank([7].iter().copied());
        assert_eq!(mr, 7.0);
    }

    #[test]
    fn test_hits_at_k_all_hit() {
        let h = metrics::hits_at_k([1, 2, 3].iter().copied(), 3);
        assert_eq!(h, 1.0);
    }

    #[test]
    fn test_hits_at_k_none_hit() {
        let h = metrics::hits_at_k([4, 5, 6].iter().copied(), 3);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_hits_at_k_with_zero_rank() {
        let h = metrics::hits_at_k([0, 1, 0, 2].iter().copied(), 10);
        assert_eq!(h, 0.5);
    }

    #[test]
    fn test_hits_at_1() {
        let h = metrics::hits_at_k([1, 2, 1, 3].iter().copied(), 1);
        assert_eq!(h, 0.5);
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let ideal = [3.0, 2.0, 1.0, 0.0];
        let score = metrics::ndcg(ideal.iter().copied(), ideal.iter().copied());
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_reversed_ranking() {
        let ranked = [0.0, 1.0, 2.0, 3.0];
        let ideal = [3.0, 2.0, 1.0, 0.0];
        let score = metrics::ndcg(ranked.iter().copied(), ideal.iter().copied());
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_ndcg_all_zero_relevance() {
        let ranked = [0.0, 0.0, 0.0];
        let ideal = [0.0, 0.0, 0.0];
        let score = metrics::ndcg(ranked.iter().copied(), ideal.iter().copied());
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_ndcg_single_element() {
        let score = metrics::ndcg([1.0].iter().copied(), [1.0].iter().copied());
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_exact_computation() {
        // Two items: ranked [1.0, 3.0], ideal [3.0, 1.0]
        // DCG  = 1.0/log2(2) + 3.0/log2(3) = 1.0 + 1.8928 = 2.8928
        // iDCG = 3.0/log2(2) + 1.0/log2(3) = 3.0 + 0.6309 = 3.6309
        // nDCG = 2.8928 / 3.6309 = 0.7967
        let score = metrics::ndcg([1.0, 3.0].iter().copied(), [3.0, 1.0].iter().copied());
        assert!((score - 0.7967).abs() < 0.01);
    }
}
