//! Training quality metrics, diagnostics, and evaluation utilities for box embeddings.

/// Rank-based evaluation metrics for link prediction and knowledge graph tasks.
///
/// These metrics are essential for evaluating box embedding quality on downstream tasks
/// like knowledge graph completion, where we need to rank candidate entities.
pub mod metrics {

    /// Compute Mean Reciprocal Rank (MRR) for a set of queries.
    ///
    /// MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is the rank of the first correct answer
    /// for query i.
    ///
    /// # Parameters
    ///
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers
    ///
    /// # Returns
    ///
    /// MRR value in [0, 1], where higher is better
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume_core::training::metrics::mean_reciprocal_rank;
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

    /// Compute Hits@K metric: fraction of queries where correct answer appears in top K.
    ///
    /// # Parameters
    ///
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers
    /// - `k`: Top-K threshold
    ///
    /// # Returns
    ///
    /// Hits@K value in [0, 1], where higher is better
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume_core::training::metrics::hits_at_k;
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
    /// - `ranks`: Iterator of ranks (1-indexed) for correct answers
    ///
    /// # Returns
    ///
    /// Mean rank value in [1, ∞), where lower is better
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume_core::training::metrics::mean_rank;
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
    /// Higher positions contribute more to the score.
    ///
    /// # Parameters
    ///
    /// - `relevance_scores`: Iterator of relevance scores for items in ranked order
    /// - `ideal_relevance`: Iterator of ideal relevance scores (sorted descending)
    ///
    /// # Returns
    ///
    /// nDCG value in [0, 1], where higher is better
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume_core::training::metrics::ndcg;
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

/// Training diagnostics for monitoring box embedding training quality.
pub mod diagnostics {
    use std::collections::VecDeque;

    /// Training statistics tracked over a window of recent iterations.
    #[derive(Debug, Clone)]
    pub struct TrainingStats {
        /// Loss values (windowed)
        losses: VecDeque<f32>,
        /// Volume statistics (windowed)
        volumes: VecDeque<f32>,
        /// Gradient norms (windowed)
        gradient_norms: VecDeque<f32>,
        /// Window size for statistics
        window_size: usize,
    }

    impl TrainingStats {
        /// Create new training statistics tracker.
        ///
        /// # Parameters
        ///
        /// - `window_size`: Number of recent iterations to track
        pub fn new(window_size: usize) -> Self {
            Self {
                losses: VecDeque::with_capacity(window_size),
                volumes: VecDeque::with_capacity(window_size),
                gradient_norms: VecDeque::with_capacity(window_size),
                window_size,
            }
        }

        /// Record a training step.
        pub fn record(&mut self, loss: f32, avg_volume: f32, gradient_norm: f32) {
            self.losses.push_back(loss);
            self.volumes.push_back(avg_volume);
            self.gradient_norms.push_back(gradient_norm);
            
            if self.losses.len() > self.window_size {
                self.losses.pop_front();
            }
            if self.volumes.len() > self.window_size {
                self.volumes.pop_front();
            }
            if self.gradient_norms.len() > self.window_size {
                self.gradient_norms.pop_front();
            }
        }

        /// Check if training has converged based on loss stability.
        ///
        /// # Parameters
        ///
        /// - `tolerance`: Maximum allowed change in loss over window
        /// - `min_iterations`: Minimum iterations before checking convergence
        ///
        /// # Returns
        ///
        /// `true` if loss has stabilized (converged)
        pub fn is_converged(&self, tolerance: f32, min_iterations: usize) -> bool {
            if self.losses.len() < min_iterations {
                return false;
            }
            
            let min_loss = self.losses.iter().copied().fold(f32::INFINITY, f32::min);
            let max_loss = self.losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            (max_loss - min_loss) <= tolerance
        }

        /// Detect gradient explosion (gradients too large).
        ///
        /// # Parameters
        ///
        /// - `threshold`: Maximum allowed gradient norm
        ///
        /// # Returns
        ///
        /// `true` if gradients are exploding
        pub fn is_gradient_exploding(&self, threshold: f32) -> bool {
            self.gradient_norms.iter().any(|&norm| norm > threshold)
        }

        /// Detect gradient vanishing (gradients too small).
        ///
        /// # Parameters
        ///
        /// - `threshold`: Minimum expected gradient norm
        ///
        /// # Returns
        ///
        /// `true` if gradients are vanishing
        pub fn is_gradient_vanishing(&self, threshold: f32) -> bool {
            self.gradient_norms.iter().all(|&norm| norm < threshold)
        }

        /// Detect volume collapse (all boxes becoming too small).
        ///
        /// # Parameters
        ///
        /// - `min_volume`: Minimum expected average volume
        ///
        /// # Returns
        ///
        /// `true` if volumes have collapsed
        pub fn is_volume_collapsed(&self, min_volume: f32) -> bool {
            self.volumes.iter().all(|&vol| vol < min_volume)
        }

        /// Get current loss statistics.
        pub fn loss_stats(&self) -> Option<(f32, f32, f32)> {
            if self.losses.is_empty() {
                return None;
            }
            
            let mean = self.losses.iter().sum::<f32>() / self.losses.len() as f32;
            let min = self.losses.iter().copied().fold(f32::INFINITY, f32::min);
            let max = self.losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            Some((mean, min, max))
        }

        /// Get current volume statistics.
        pub fn volume_stats(&self) -> Option<(f32, f32, f32)> {
            if self.volumes.is_empty() {
                return None;
            }
            
            let mean = self.volumes.iter().sum::<f32>() / self.volumes.len() as f32;
            let min = self.volumes.iter().copied().fold(f32::INFINITY, f32::min);
            let max = self.volumes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            Some((mean, min, max))
        }

        /// Get current gradient statistics.
        pub fn gradient_stats(&self) -> Option<(f32, f32, f32)> {
            if self.gradient_norms.is_empty() {
                return None;
            }
            
            let mean = self.gradient_norms.iter().sum::<f32>() / self.gradient_norms.len() as f32;
            let min = self.gradient_norms.iter().copied().fold(f32::INFINITY, f32::min);
            let max = self.gradient_norms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            Some((mean, min, max))
        }
    }

    /// Loss component breakdown for multi-component loss functions.
    ///
    /// Box embedding training often uses multiple loss components (containment loss,
    /// regularization loss, etc.). This tracks each component separately.
    #[derive(Debug, Clone)]
    pub struct LossComponents {
        /// Containment/overlap loss component
        pub containment_loss: f32,
        /// Volume regularization loss component
        pub regularization_loss: f32,
        /// Additional constraint loss (e.g., logical constraints)
        pub constraint_loss: f32,
    }

    impl LossComponents {
        /// Create new loss components tracker.
        pub fn new(
            containment_loss: f32,
            regularization_loss: f32,
            constraint_loss: f32,
        ) -> Self {
            Self {
                containment_loss,
                regularization_loss,
                constraint_loss,
            }
        }

        /// Total loss (sum of all components).
        pub fn total(&self) -> f32 {
            self.containment_loss + self.regularization_loss + self.constraint_loss
        }

        /// Check if loss components are imbalanced.
        ///
        /// Returns `true` if any component dominates (>80% of total).
        pub fn is_imbalanced(&self) -> bool {
            let total = self.total();
            if total == 0.0 {
                return false;
            }
            
            let containment_ratio = self.containment_loss / total;
            let reg_ratio = self.regularization_loss / total;
            let constraint_ratio = self.constraint_loss / total;
            
            containment_ratio > 0.8 || reg_ratio > 0.8 || constraint_ratio > 0.8
        }

        /// Get the dominant component (if any).
        pub fn dominant_component(&self) -> Option<&'static str> {
            let total = self.total();
            if total == 0.0 {
                return None;
            }
            
            let containment_ratio = self.containment_loss / total;
            let reg_ratio = self.regularization_loss / total;
            let constraint_ratio = self.constraint_loss / total;
            
            if containment_ratio > 0.8 {
                Some("containment")
            } else if reg_ratio > 0.8 {
                Some("regularization")
            } else if constraint_ratio > 0.8 {
                Some("constraint")
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_reciprocal_rank() {
        let ranks = vec![1, 3, 2, 5];
        let mrr = metrics::mean_reciprocal_rank(ranks.iter().copied());
        // (1/1 + 1/3 + 1/2 + 1/5) / 4 = (1 + 0.3333 + 0.5 + 0.2) / 4 ≈ 0.5083
        assert!((mrr - 0.5083).abs() < 1e-3);
    }

    #[test]
    fn test_hits_at_k() {
        let ranks = vec![1, 3, 2, 5, 10];
        let hits_3 = metrics::hits_at_k(ranks.iter().copied(), 3);
        assert_eq!(hits_3, 0.6);
    }

    #[test]
    fn test_mean_rank() {
        let ranks = vec![1, 3, 2, 5];
        let mr = metrics::mean_rank(ranks.iter().copied());
        assert_eq!(mr, 2.75);
    }

    #[test]
    fn test_ndcg() {
        let ranked = vec![0.9, 0.5, 0.8, 0.2];
        let ideal = vec![0.9, 0.8, 0.5, 0.2];
        let score = metrics::ndcg(ranked.iter().copied(), ideal.iter().copied());
        assert!(score > 0.9);
    }

    #[test]
    fn test_training_stats() {
        let mut stats = diagnostics::TrainingStats::new(10);
        
        // Record some steps
        for i in 0..5 {
            stats.record(1.0 - i as f32 * 0.1, 0.5, 0.1);
        }
        
        assert!(!stats.is_converged(0.01, 5));
        assert!(!stats.is_gradient_exploding(100.0));
        assert!(!stats.is_volume_collapsed(0.01));
        
        let loss_stats = stats.loss_stats().unwrap();
        assert!(loss_stats.0 > 0.0); // Mean loss
    }

    #[test]
    fn test_loss_components() {
        // Check that containment is dominant (0.81 / 1.0 = 0.81, which is > 0.8 threshold)
        let components2 = diagnostics::LossComponents::new(0.81, 0.1, 0.09);
        assert!(components2.is_imbalanced());
        assert_eq!(components2.dominant_component(), Some("containment"));
        
        let balanced = diagnostics::LossComponents::new(0.4, 0.3, 0.3);
        assert!(!balanced.is_imbalanced());
        assert_eq!(balanced.dominant_component(), None);
    }
}

