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

/// Embedding quality assessment utilities for box embeddings.
pub mod quality {

    /// Volume distribution statistics for analyzing box embedding quality.
    #[derive(Debug, Clone)]
    pub struct VolumeDistribution {
        /// Minimum volume
        pub min: f32,
        /// Maximum volume
        pub max: f32,
        /// Mean volume
        pub mean: f32,
        /// Median volume
        pub median: f32,
        /// Standard deviation of volumes
        pub std_dev: f32,
        /// Coefficient of variation (std_dev / mean)
        pub cv: f32,
    }

    impl VolumeDistribution {
        /// Compute volume distribution statistics from a collection of volumes.
        pub fn from_volumes<I>(volumes: I) -> Self
        where
            I: Iterator<Item = f32>,
        {
            let mut vols: Vec<f32> = volumes.collect();
            if vols.is_empty() {
                return Self {
                    min: 0.0,
                    max: 0.0,
                    mean: 0.0,
                    median: 0.0,
                    std_dev: 0.0,
                    cv: 0.0,
                };
            }

            vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let min = vols[0];
            let max = vols[vols.len() - 1];
            let mean = vols.iter().sum::<f32>() / vols.len() as f32;
            
            let median = if vols.len() % 2 == 0 {
                (vols[vols.len() / 2 - 1] + vols[vols.len() / 2]) / 2.0
            } else {
                vols[vols.len() / 2]
            };
            
            let variance = vols.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f32>() / vols.len() as f32;
            let std_dev = variance.sqrt();
            
            let cv = if mean > 0.0 {
                std_dev / mean
            } else {
                0.0
            };

            Self {
                min,
                max,
                mean,
                median,
                std_dev,
                cv,
            }
        }

        /// Check if volume distribution indicates collapse (all volumes too small).
        pub fn is_collapsed(&self, threshold: f32) -> bool {
            self.max < threshold
        }

        /// Check if volume distribution is degenerate (all volumes identical).
        pub fn is_degenerate(&self, tolerance: f32) -> bool {
            (self.max - self.min) < tolerance
        }

        /// Check if volume distribution has useful hierarchy (high variance).
        pub fn has_hierarchy(&self, min_cv: f32) -> bool {
            self.cv >= min_cv
        }
    }

    /// Containment accuracy statistics.
    #[derive(Debug, Clone)]
    pub struct ContainmentAccuracy {
        /// True positives (correctly predicted containments)
        pub true_positives: usize,
        /// False positives (incorrectly predicted containments)
        pub false_positives: usize,
        /// True negatives (correctly predicted non-containments)
        pub true_negatives: usize,
        /// False negatives (missed containments)
        pub false_negatives: usize,
    }

    impl ContainmentAccuracy {
        /// Create new containment accuracy tracker.
        pub fn new() -> Self {
            Self {
                true_positives: 0,
                false_positives: 0,
                true_negatives: 0,
                false_negatives: 0,
            }
        }

        /// Record a containment prediction.
        pub fn record(&mut self, predicted: bool, actual: bool) {
            match (predicted, actual) {
                (true, true) => self.true_positives += 1,
                (true, false) => self.false_positives += 1,
                (false, true) => self.false_negatives += 1,
                (false, false) => self.true_negatives += 1,
            }
        }

        /// Precision: TP / (TP + FP)
        pub fn precision(&self) -> f32 {
            let total_positive = self.true_positives + self.false_positives;
            if total_positive == 0 {
                0.0
            } else {
                self.true_positives as f32 / total_positive as f32
            }
        }

        /// Recall: TP / (TP + FN)
        pub fn recall(&self) -> f32 {
            let total_actual_positive = self.true_positives + self.false_negatives;
            if total_actual_positive == 0 {
                0.0
            } else {
                self.true_positives as f32 / total_actual_positive as f32
            }
        }

        /// F1 score: 2 * (precision * recall) / (precision + recall)
        pub fn f1(&self) -> f32 {
            let prec = self.precision();
            let rec = self.recall();
            if prec + rec == 0.0 {
                0.0
            } else {
                2.0 * prec * rec / (prec + rec)
            }
        }

        /// Accuracy: (TP + TN) / (TP + FP + TN + FN)
        pub fn accuracy(&self) -> f32 {
            let total = self.true_positives + self.false_positives
                + self.true_negatives + self.false_negatives;
            if total == 0 {
                0.0
            } else {
                (self.true_positives + self.true_negatives) as f32 / total as f32
            }
        }
    }

    impl Default for ContainmentAccuracy {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Intersection topology statistics for analyzing box relationships.
    #[derive(Debug, Clone)]
    pub struct IntersectionTopology {
        /// Total number of box pairs analyzed
        pub total_pairs: usize,
        /// Number of pairs with non-empty intersection
        pub intersecting_pairs: usize,
        /// Number of pairs with containment relationship
        pub containment_pairs: usize,
        /// Number of pairs that are disjoint
        pub disjoint_pairs: usize,
    }

    impl IntersectionTopology {
        /// Create new intersection topology tracker.
        pub fn new() -> Self {
            Self {
                total_pairs: 0,
                intersecting_pairs: 0,
                containment_pairs: 0,
                disjoint_pairs: 0,
            }
        }

        /// Record an intersection relationship.
        pub fn record_intersection(&mut self, has_intersection: bool) {
            self.total_pairs += 1;
            if has_intersection {
                self.intersecting_pairs += 1;
            } else {
                self.disjoint_pairs += 1;
            }
        }

        /// Record a containment relationship.
        pub fn record_containment(&mut self, has_containment: bool) {
            if has_containment {
                self.containment_pairs += 1;
            }
        }

        /// Fraction of pairs that intersect.
        pub fn intersection_rate(&self) -> f32 {
            if self.total_pairs == 0 {
                0.0
            } else {
                self.intersecting_pairs as f32 / self.total_pairs as f32
            }
        }

        /// Fraction of pairs with containment.
        pub fn containment_rate(&self) -> f32 {
            if self.total_pairs == 0 {
                0.0
            } else {
                self.containment_pairs as f32 / self.total_pairs as f32
            }
        }

        /// Fraction of pairs that are disjoint.
        pub fn disjoint_rate(&self) -> f32 {
            if self.total_pairs == 0 {
                0.0
            } else {
                self.disjoint_pairs as f32 / self.total_pairs as f32
            }
        }
    }

    impl Default for IntersectionTopology {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Calibration metrics for probabilistic box embeddings.
pub mod calibration {
    /// Expected Calibration Error (ECE) for box embedding predictions.
    ///
    /// Measures how well predicted probabilities match empirical frequencies.
    /// Lower ECE indicates better calibration.
    ///
    /// # Parameters
    ///
    /// - `predictions`: Iterator of predicted probabilities
    /// - `actuals`: Iterator of actual binary outcomes (true/false)
    /// - `n_bins`: Number of bins for probability discretization
    ///
    /// # Returns
    ///
    /// ECE value in [0, 1], where lower is better
    pub fn expected_calibration_error<I, J>(
        predictions: I,
        actuals: J,
        n_bins: usize,
    ) -> f32
    where
        I: Iterator<Item = f32>,
        J: Iterator<Item = bool>,
    {
        let mut bins: Vec<Vec<(f32, bool)>> = vec![Vec::new(); n_bins];
        
        // Collect predictions and actuals into bins
        for (pred, actual) in predictions.zip(actuals) {
            let pred_clamped = pred.clamp(0.0, 1.0);
            let bin_idx = ((pred_clamped * n_bins as f32) as usize).min(n_bins - 1);
            bins[bin_idx].push((pred_clamped, actual));
        }
        
        let mut ece = 0.0;
        let mut total_samples = 0;
        
        for bin in bins.iter() {
            if bin.is_empty() {
                continue;
            }
            
            let bin_size = bin.len();
            total_samples += bin_size;
            
            // Average predicted probability in this bin
            let avg_pred = bin.iter().map(|(p, _)| p).sum::<f32>() / bin_size as f32;
            
            // Empirical accuracy in this bin
            let empirical_acc = bin.iter().filter(|(_, a)| *a).count() as f32 / bin_size as f32;
            
            // Weighted absolute difference
            ece += (avg_pred - empirical_acc).abs() * bin_size as f32;
        }
        
        if total_samples > 0 {
            ece / total_samples as f32
        } else {
            0.0
        }
    }

    /// Brier score for probabilistic predictions.
    ///
    /// Measures the mean squared difference between predicted probabilities
    /// and actual binary outcomes. Lower is better.
    ///
    /// # Parameters
    ///
    /// - `predictions`: Iterator of predicted probabilities
    /// - `actuals`: Iterator of actual binary outcomes (true/false)
    ///
    /// # Returns
    ///
    /// Brier score in [0, 1], where lower is better
    pub fn brier_score<I, J>(predictions: I, actuals: J) -> f32
    where
        I: Iterator<Item = f32>,
        J: Iterator<Item = bool>,
    {
        let mut sum_squared_error = 0.0;
        let mut count = 0;
        
        for (pred, actual) in predictions.zip(actuals) {
            let pred_clamped = pred.clamp(0.0, 1.0);
            let actual_val = if actual { 1.0 } else { 0.0 };
            sum_squared_error += (pred_clamped - actual_val).powi(2);
            count += 1;
        }
        
        if count > 0 {
            sum_squared_error / count as f32
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

    #[test]
    fn test_volume_distribution() {
        let volumes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let dist = quality::VolumeDistribution::from_volumes(volumes.iter().copied());
        
        assert!((dist.min - 0.1).abs() < 1e-6);
        assert!((dist.max - 1.0).abs() < 1e-6);
        assert!((dist.mean - 0.55).abs() < 1e-6);
        assert!(!dist.is_collapsed(0.05));
        assert!(!dist.is_degenerate(0.01));
        assert!(dist.has_hierarchy(0.1));
    }

    #[test]
    fn test_containment_accuracy() {
        let mut acc = quality::ContainmentAccuracy::new();
        
        // Record some predictions
        acc.record(true, true);   // TP
        acc.record(true, true);   // TP
        acc.record(true, false);  // FP
        acc.record(false, true); // FN
        acc.record(false, false); // TN
        acc.record(false, false); // TN
        
        assert_eq!(acc.true_positives, 2);
        assert_eq!(acc.false_positives, 1);
        assert_eq!(acc.false_negatives, 1);
        assert_eq!(acc.true_negatives, 2);
        
        // Precision = 2 / (2 + 1) = 2/3 ≈ 0.667
        assert!((acc.precision() - 0.6667).abs() < 1e-3);
        // Recall = 2 / (2 + 1) = 2/3 ≈ 0.667
        assert!((acc.recall() - 0.6667).abs() < 1e-3);
        // Accuracy = (2 + 2) / 6 = 4/6 ≈ 0.667
        assert!((acc.accuracy() - 0.6667).abs() < 1e-3);
    }

    #[test]
    fn test_expected_calibration_error() {
        // Well-calibrated predictions
        let preds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let actuals = vec![false, false, false, false, true, true, true, true, true];
        let ece = calibration::expected_calibration_error(
            preds.iter().copied(),
            actuals.iter().copied(),
            10,
        );
        // Should be relatively low for well-calibrated predictions
        assert!(ece < 0.3);
    }

    #[test]
    fn test_brier_score() {
        // Perfect predictions
        let preds = vec![0.0, 0.0, 1.0, 1.0];
        let actuals = vec![false, false, true, true];
        let brier = calibration::brier_score(preds.iter().copied(), actuals.iter().copied());
        assert_eq!(brier, 0.0);
        
        // Random predictions (should have higher Brier score)
        let preds2 = vec![0.5, 0.5, 0.5, 0.5];
        let actuals2 = vec![false, true, false, true];
        let brier2 = calibration::brier_score(preds2.iter().copied(), actuals2.iter().copied());
        assert!(brier2 > 0.0);
    }

    #[test]
    fn test_intersection_topology() {
        let mut topology = quality::IntersectionTopology::new();
        
        topology.record_intersection(true);   // Intersecting
        topology.record_intersection(true);   // Intersecting
        topology.record_intersection(false);  // Disjoint
        topology.record_containment(true);    // Has containment
        topology.record_containment(false);   // No containment
        
        assert_eq!(topology.total_pairs, 3);
        assert_eq!(topology.intersecting_pairs, 2);
        assert_eq!(topology.disjoint_pairs, 1);
        assert_eq!(topology.containment_pairs, 1);
        
        assert!((topology.intersection_rate() - 0.6667).abs() < 1e-3);
        assert!((topology.containment_rate() - 0.3333).abs() < 1e-3);
        assert!((topology.disjoint_rate() - 0.3333).abs() < 1e-3);
    }
}

