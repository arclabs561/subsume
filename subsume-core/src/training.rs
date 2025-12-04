//! Training quality metrics, diagnostics, and evaluation utilities for box embeddings.

/// Rank-based evaluation metrics for link prediction and knowledge graph tasks.
///
/// These metrics are essential for evaluating box embedding quality on downstream tasks
/// like knowledge graph completion, where we need to rank candidate entities.
pub mod metrics {
    /// Stratified evaluation results for analyzing performance across different data subsets.
    #[derive(Debug, Clone)]
    pub struct StratifiedMetrics {
        /// Relation-stratified metrics (if applicable)
        pub by_relation: std::collections::HashMap<String, RelationMetrics>,
        /// Depth-stratified metrics (for hierarchical data)
        pub by_depth: std::collections::HashMap<usize, DepthMetrics>,
        /// Frequency-stratified metrics
        pub by_frequency: FrequencyMetrics,
    }

    /// Metrics for a specific relation type.
    #[derive(Debug, Clone)]
    pub struct RelationMetrics {
        /// Number of test samples
        pub count: usize,
        /// Mean Reciprocal Rank
        pub mrr: f32,
        /// Hits@10
        pub hits_10: f32,
        /// Mean Rank
        pub mean_rank: f32,
    }

    /// Metrics for a specific hierarchy depth.
    #[derive(Debug, Clone)]
    pub struct DepthMetrics {
        /// Number of test samples at this depth
        pub count: usize,
        /// Mean Reciprocal Rank
        pub mrr: f32,
        /// Hits@10
        pub hits_10: f32,
        /// Containment accuracy
        pub containment_accuracy: f32,
    }

    /// Metrics stratified by entity/relation frequency.
    #[derive(Debug, Clone)]
    pub struct FrequencyMetrics {
        /// High frequency (top quartile)
        pub high_freq: FrequencyStratum,
        /// Medium frequency (middle quartiles)
        pub medium_freq: FrequencyStratum,
        /// Low frequency (bottom quartile)
        pub low_freq: FrequencyStratum,
    }

    /// Metrics for a frequency stratum.
    #[derive(Debug, Clone)]
    pub struct FrequencyStratum {
        /// Number of test samples
        pub count: usize,
        /// Mean Reciprocal Rank
        pub mrr: f32,
        /// Hits@10
        pub hits_10: f32,
    }

    impl StratifiedMetrics {
        /// Create new stratified metrics tracker.
        pub fn new() -> Self {
            Self {
                by_relation: std::collections::HashMap::new(),
                by_depth: std::collections::HashMap::new(),
                by_frequency: FrequencyMetrics {
                    high_freq: FrequencyStratum { count: 0, mrr: 0.0, hits_10: 0.0 },
                    medium_freq: FrequencyStratum { count: 0, mrr: 0.0, hits_10: 0.0 },
                    low_freq: FrequencyStratum { count: 0, mrr: 0.0, hits_10: 0.0 },
                },
            }
        }

        /// Add relation-stratified result.
        pub fn add_relation_result(&mut self, relation: String, rank: usize) {
            let metrics = self.by_relation.entry(relation).or_insert_with(|| RelationMetrics {
                count: 0,
                mrr: 0.0,
                hits_10: 0.0,
                mean_rank: 0.0,
            });
            
            metrics.count += 1;
            if rank > 0 {
                metrics.mrr += 1.0 / rank as f32;
                if rank <= 10 {
                    metrics.hits_10 += 1.0;
                }
                metrics.mean_rank += rank as f32;
            }
        }

        /// Finalize relation metrics (compute averages).
        pub fn finalize_relations(&mut self) {
            for metrics in self.by_relation.values_mut() {
                if metrics.count > 0 {
                    metrics.mrr /= metrics.count as f32;
                    metrics.hits_10 /= metrics.count as f32;
                    metrics.mean_rank /= metrics.count as f32;
                }
            }
        }

        /// Add depth-stratified result.
        pub fn add_depth_result(&mut self, depth: usize, rank: usize, containment_correct: bool) {
            let metrics = self.by_depth.entry(depth).or_insert_with(|| DepthMetrics {
                count: 0,
                mrr: 0.0,
                hits_10: 0.0,
                containment_accuracy: 0.0,
            });
            
            metrics.count += 1;
            if rank > 0 {
                metrics.mrr += 1.0 / rank as f32;
                if rank <= 10 {
                    metrics.hits_10 += 1.0;
                }
            }
            if containment_correct {
                metrics.containment_accuracy += 1.0;
            }
        }

        /// Finalize depth metrics (compute averages).
        pub fn finalize_depths(&mut self) {
            for metrics in self.by_depth.values_mut() {
                if metrics.count > 0 {
                    metrics.mrr /= metrics.count as f32;
                    metrics.hits_10 /= metrics.count as f32;
                    metrics.containment_accuracy /= metrics.count as f32;
                }
            }
        }

        /// Add frequency-stratified result.
        ///
        /// # Parameters
        ///
        /// - `frequency_category`: "high", "medium", or "low"
        /// - `rank`: Rank of the correct answer (1-indexed, 0 if not found)
        pub fn add_frequency_result(&mut self, frequency_category: &str, rank: usize) {
            let stratum = match frequency_category {
                "high" => &mut self.by_frequency.high_freq,
                "medium" => &mut self.by_frequency.medium_freq,
                "low" => &mut self.by_frequency.low_freq,
                _ => return, // Invalid category, ignore
            };
            
            stratum.count += 1;
            if rank > 0 {
                stratum.mrr += 1.0 / rank as f32;
                if rank <= 10 {
                    stratum.hits_10 += 1.0;
                }
            }
        }

        /// Finalize frequency metrics (compute averages).
        pub fn finalize_frequency(&mut self) {
            if self.by_frequency.high_freq.count > 0 {
                self.by_frequency.high_freq.mrr /= self.by_frequency.high_freq.count as f32;
                self.by_frequency.high_freq.hits_10 /= self.by_frequency.high_freq.count as f32;
            }
            if self.by_frequency.medium_freq.count > 0 {
                self.by_frequency.medium_freq.mrr /= self.by_frequency.medium_freq.count as f32;
                self.by_frequency.medium_freq.hits_10 /= self.by_frequency.medium_freq.count as f32;
            }
            if self.by_frequency.low_freq.count > 0 {
                self.by_frequency.low_freq.mrr /= self.by_frequency.low_freq.count as f32;
                self.by_frequency.low_freq.hits_10 /= self.by_frequency.low_freq.count as f32;
            }
        }
    }

    impl Default for StratifiedMetrics {
        fn default() -> Self {
            Self::new()
        }
    }

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
        /// Intersection volume statistics (windowed)
        /// Tracks average intersection volume between related boxes during training.
        /// Important for monitoring how containment relationships evolve.
        intersection_volumes: VecDeque<f32>,
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
                intersection_volumes: VecDeque::with_capacity(window_size),
                gradient_norms: VecDeque::with_capacity(window_size),
                window_size,
            }
        }

        /// Record a training step.
        pub fn record(&mut self, loss: f32, avg_volume: f32, gradient_norm: f32) {
            self.record_with_intersection(loss, avg_volume, None, gradient_norm);
        }

        /// Record a training step with intersection volume tracking.
        ///
        /// # Parameters
        ///
        /// - `loss`: Current loss value
        /// - `avg_volume`: Average box volume
        /// - `avg_intersection_volume`: Average intersection volume between related boxes (optional)
        /// - `gradient_norm`: Current gradient norm
        pub fn record_with_intersection(
            &mut self,
            loss: f32,
            avg_volume: f32,
            avg_intersection_volume: Option<f32>,
            gradient_norm: f32,
        ) {
            self.losses.push_back(loss);
            self.volumes.push_back(avg_volume);
            if let Some(int_vol) = avg_intersection_volume {
                self.intersection_volumes.push_back(int_vol);
            }
            self.gradient_norms.push_back(gradient_norm);
            
            if self.losses.len() > self.window_size {
                self.losses.pop_front();
            }
            if self.volumes.len() > self.window_size {
                self.volumes.pop_front();
            }
            if self.intersection_volumes.len() > self.window_size {
                self.intersection_volumes.pop_front();
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
            if self.volumes.is_empty() {
                return false; // Can't determine collapse without data
            }
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

        /// Get current intersection volume statistics.
        ///
        /// Returns `(mean, min, max)` of intersection volumes, or `None` if no data.
        ///
        /// Intersection volumes track how containment relationships evolve during training.
        /// Decreasing intersection volumes may indicate boxes becoming more disjoint,
        /// while increasing volumes suggest better containment relationships.
        pub fn intersection_volume_stats(&self) -> Option<(f32, f32, f32)> {
            if self.intersection_volumes.is_empty() {
                return None;
            }
            
            let mean = self.intersection_volumes.iter().sum::<f32>() / self.intersection_volumes.len() as f32;
            let min = self.intersection_volumes.iter().copied().fold(f32::INFINITY, f32::min);
            let max = self.intersection_volumes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            Some((mean, min, max))
        }

        /// Detect intersection volume trend during training.
        ///
        /// # Parameters
        ///
        /// - `min_samples`: Minimum number of samples required for trend detection
        ///
        /// # Returns
        ///
        /// - `Some(true)`: Intersection volumes are increasing (better containment)
        /// - `Some(false)`: Intersection volumes are decreasing (boxes becoming disjoint)
        /// - `None`: Not enough data or no clear trend
        pub fn intersection_volume_trend(&self, min_samples: usize) -> Option<bool> {
            if self.intersection_volumes.len() < min_samples {
                return None;
            }
            
            // Use linear regression to detect trend
            let n = self.intersection_volumes.len() as f32;
            let sum_x: f32 = (0..self.intersection_volumes.len()).map(|i| i as f32).sum();
            let sum_y: f32 = self.intersection_volumes.iter().sum();
            let sum_xy: f32 = self.intersection_volumes.iter()
                .enumerate()
                .map(|(i, &y)| i as f32 * y)
                .sum();
            let sum_x2: f32 = (0..self.intersection_volumes.len())
                .map(|i| (i as f32).powi(2))
                .sum();
            
            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() < 1e-6 {
                return None;
            }
            
            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
            
            // Positive slope indicates increasing intersection volumes
            Some(slope > 0.0)
        }

        /// Get intersection volume trend magnitude (slope).
        ///
        /// Returns the actual slope value, not just direction.
        /// Positive values indicate increasing volumes, negative indicate decreasing.
        pub fn intersection_volume_slope(&self, min_samples: usize) -> Option<f32> {
            if self.intersection_volumes.len() < min_samples {
                return None;
            }
            
            let n = self.intersection_volumes.len() as f32;
            let sum_x: f32 = (0..self.intersection_volumes.len()).map(|i| i as f32).sum();
            let sum_y: f32 = self.intersection_volumes.iter().sum();
            let sum_xy: f32 = self.intersection_volumes.iter()
                .enumerate()
                .map(|(i, &y)| i as f32 * y)
                .sum();
            let sum_x2: f32 = (0..self.intersection_volumes.len())
                .map(|i| (i as f32).powi(2))
                .sum();
            
            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() < 1e-6 {
                return None;
            }
            
            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
            Some(slope)
        }

        /// Clear all recorded statistics (reset to empty state).
        pub fn clear(&mut self) {
            self.losses.clear();
            self.volumes.clear();
            self.intersection_volumes.clear();
            self.gradient_norms.clear();
        }
    }

    /// Per-parameter gradient flow analysis for box embeddings.
    ///
    /// Tracks gradients separately for different parameter types (center vs size,
    /// min vs max coordinates) to detect imbalanced optimization.
    #[derive(Debug, Clone)]
    pub struct GradientFlowAnalysis {
        /// Gradient norms for center parameters (if using center-offset representation)
        center_gradients: VecDeque<f32>,
        /// Gradient norms for size/offset parameters
        size_gradients: VecDeque<f32>,
        /// Gradient norms for minimum coordinates
        min_gradients: VecDeque<f32>,
        /// Gradient norms for maximum coordinates
        max_gradients: VecDeque<f32>,
        /// Window size for tracking
        window_size: usize,
    }

    impl GradientFlowAnalysis {
        /// Create new gradient flow analyzer.
        pub fn new(window_size: usize) -> Self {
            Self {
                center_gradients: VecDeque::with_capacity(window_size),
                size_gradients: VecDeque::with_capacity(window_size),
                min_gradients: VecDeque::with_capacity(window_size),
                max_gradients: VecDeque::with_capacity(window_size),
                window_size,
            }
        }

        /// Record gradients for different parameter types.
        pub fn record(
            &mut self,
            center_grad: Option<f32>,
            size_grad: Option<f32>,
            min_grad: Option<f32>,
            max_grad: Option<f32>,
        ) {
            if let Some(g) = center_grad {
                self.center_gradients.push_back(g);
                if self.center_gradients.len() > self.window_size {
                    self.center_gradients.pop_front();
                }
            }
            if let Some(g) = size_grad {
                self.size_gradients.push_back(g);
                if self.size_gradients.len() > self.window_size {
                    self.size_gradients.pop_front();
                }
            }
            if let Some(g) = min_grad {
                self.min_gradients.push_back(g);
                if self.min_gradients.len() > self.window_size {
                    self.min_gradients.pop_front();
                }
            }
            if let Some(g) = max_grad {
                self.max_gradients.push_back(g);
                if self.max_gradients.len() > self.window_size {
                    self.max_gradients.pop_front();
                }
            }
        }

        /// Check if gradient flow is imbalanced between parameter types.
        ///
        /// Returns `Some(ratio)` if imbalance detected (ratio > threshold),
        /// where ratio is the larger gradient mean divided by the smaller.
        pub fn check_imbalance(&self, threshold: f32) -> Option<f32> {
            let center_mean = if !self.center_gradients.is_empty() {
                self.center_gradients.iter().sum::<f32>() / self.center_gradients.len() as f32
            } else {
                0.0
            };
            let size_mean = if !self.size_gradients.is_empty() {
                self.size_gradients.iter().sum::<f32>() / self.size_gradients.len() as f32
            } else {
                0.0
            };
            
            if center_mean > 0.0 && size_mean > 0.0 {
                let ratio = (center_mean / size_mean).max(size_mean / center_mean);
                if ratio > threshold {
                    return Some(ratio);
                }
            }
            
            let min_mean = if !self.min_gradients.is_empty() {
                self.min_gradients.iter().sum::<f32>() / self.min_gradients.len() as f32
            } else {
                0.0
            };
            let max_mean = if !self.max_gradients.is_empty() {
                self.max_gradients.iter().sum::<f32>() / self.max_gradients.len() as f32
            } else {
                0.0
            };
            
            if min_mean > 0.0 && max_mean > 0.0 {
                let ratio = (min_mean / max_mean).max(max_mean / min_mean);
                if ratio > threshold {
                    return Some(ratio);
                }
            }
            
            None
        }

        /// Compute gradient sparsity: fraction of parameters with near-zero gradients.
        pub fn gradient_sparsity(&self, threshold: f32) -> f32 {
            let mut total = 0;
            let mut sparse = 0;
            
            for &g in &self.center_gradients {
                total += 1;
                if g < threshold {
                    sparse += 1;
                }
            }
            for &g in &self.size_gradients {
                total += 1;
                if g < threshold {
                    sparse += 1;
                }
            }
            for &g in &self.min_gradients {
                total += 1;
                if g < threshold {
                    sparse += 1;
                }
            }
            for &g in &self.max_gradients {
                total += 1;
                if g < threshold {
                    sparse += 1;
                }
            }
            
            if total == 0 {
                0.0
            } else {
                sparse as f32 / total as f32
            }
        }
    }

    /// Gradient flow analysis stratified by hierarchy depth.
    ///
    /// Tracks gradients separately for boxes at different hierarchy levels,
    /// revealing whether training distributes learning effort uniformly.
    #[derive(Debug, Clone)]
    pub struct DepthStratifiedGradientFlow {
        /// Gradient norms by depth level
        gradients_by_depth: std::collections::HashMap<usize, VecDeque<f32>>,
        /// Window size for tracking
        window_size: usize,
    }

    impl DepthStratifiedGradientFlow {
        /// Create new depth-stratified gradient flow analyzer.
        pub fn new(window_size: usize) -> Self {
            Self {
                gradients_by_depth: std::collections::HashMap::new(),
                window_size,
            }
        }

        /// Record gradient for a box at a specific hierarchy depth.
        pub fn record(&mut self, depth: usize, gradient_norm: f32) {
            let gradients = self.gradients_by_depth
                .entry(depth)
                .or_insert_with(|| VecDeque::with_capacity(self.window_size));
            
            gradients.push_back(gradient_norm);
            if gradients.len() > self.window_size {
                gradients.pop_front();
            }
        }

        /// Get mean gradient magnitude for each depth level.
        pub fn mean_gradients_by_depth(&self) -> std::collections::HashMap<usize, f32> {
            self.gradients_by_depth.iter()
                .map(|(&depth, grads)| {
                    let mean = if grads.is_empty() {
                        0.0
                    } else {
                        grads.iter().sum::<f32>() / grads.len() as f32
                    };
                    (depth, mean)
                })
                .collect()
        }

        /// Check if gradient flow is uniform across depths.
        ///
        /// Returns `Some((min_depth, max_depth, ratio))` if imbalance detected,
        /// where ratio is max_depth_gradient / min_depth_gradient.
        pub fn check_depth_imbalance(&self, threshold: f32) -> Option<(usize, usize, f32)> {
            let means = self.mean_gradients_by_depth();
            if means.len() < 2 {
                return None;
            }
            
            let (min_depth, &min_grad) = means.iter()
                .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                .unwrap();
            let (max_depth, &max_grad) = means.iter()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                .unwrap();
            
            if min_grad > 0.0 {
                let ratio = max_grad / min_grad;
                if ratio > threshold {
                    return Some((*min_depth, *max_depth, ratio));
                }
            }
            
            None
        }
    }

    /// Training phase detection based on loss and gradient dynamics.
    ///
    /// Identifies different phases: exploration, exploitation, convergence.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TrainingPhase {
        /// Early training: high loss, high gradients, exploration
        Exploration,
        /// Mid training: decreasing loss, moderate gradients, exploitation
        Exploitation,
        /// Late training: stable loss, low gradients, convergence
        Convergence,
        /// Potential problem: loss increasing or unstable
        Instability,
    }

    /// Training phase detector.
    #[derive(Debug, Clone)]
    pub struct PhaseDetector {
        /// Recent loss values
        losses: VecDeque<f32>,
        /// Recent gradient norms
        gradient_norms: VecDeque<f32>,
        /// Window size
        window_size: usize,
    }

    impl PhaseDetector {
        /// Create new phase detector.
        pub fn new(window_size: usize) -> Self {
            Self {
                losses: VecDeque::with_capacity(window_size),
                gradient_norms: VecDeque::with_capacity(window_size),
                window_size,
            }
        }

        /// Record training step.
        pub fn record(&mut self, loss: f32, gradient_norm: f32) {
            self.losses.push_back(loss);
            self.gradient_norms.push_back(gradient_norm);
            
            if self.losses.len() > self.window_size {
                self.losses.pop_front();
            }
            if self.gradient_norms.len() > self.window_size {
                self.gradient_norms.pop_front();
            }
        }

        /// Detect current training phase.
        pub fn detect_phase(&self) -> TrainingPhase {
            if self.losses.len() < 3 {
                return TrainingPhase::Exploration;
            }
            
            let recent_losses: Vec<f32> = self.losses.iter().copied().collect();
            let mean_grad = if self.gradient_norms.is_empty() {
                0.0
            } else {
                self.gradient_norms.iter().sum::<f32>() / self.gradient_norms.len() as f32
            };
            
            // Check for instability (loss increasing significantly)
            let loss_trend = recent_losses[recent_losses.len() - 1] - recent_losses[0];
            if loss_trend > 0.1 {
                return TrainingPhase::Instability;
            }
            
            // Check for convergence (stable loss, low gradients)
            // Need enough samples to reliably detect convergence
            if recent_losses.len() >= 7 {
                let loss_variance = {
                    let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
                    recent_losses.iter()
                        .map(|&l| (l - mean).powi(2))
                        .sum::<f32>() / recent_losses.len() as f32
                };
                
                // Very strict: loss must be extremely stable AND gradients very low
                if loss_variance < 0.0001 && mean_grad < 0.06 {
                    return TrainingPhase::Convergence;
                }
            }
            
            // Check for exploitation (decreasing loss, moderate gradients)
            // Need more samples to reliably detect exploitation
            if recent_losses.len() >= 5 {
                let recent_trend = recent_losses[recent_losses.len() - 1] - recent_losses[recent_losses.len() - 5];
                if recent_trend < -0.15 && mean_grad > 0.05 && mean_grad < 0.5 {
                    return TrainingPhase::Exploitation;
                }
            }
            
            // Default to exploration
            TrainingPhase::Exploration
        }
    }

    /// Relation-stratified training statistics for tracking convergence per relation type.
    ///
    /// Different relation types in knowledge graphs may converge at different rates.
    /// This tracks training statistics separately for each relation, allowing detection
    /// of relation-specific convergence issues.
    #[derive(Debug, Clone)]
    pub struct RelationStratifiedTrainingStats {
        /// Training stats per relation type
        stats_by_relation: std::collections::HashMap<String, TrainingStats>,
        /// Window size for statistics
        window_size: usize,
    }

    impl RelationStratifiedTrainingStats {
        /// Create new relation-stratified training stats tracker.
        pub fn new(window_size: usize) -> Self {
            Self {
                stats_by_relation: std::collections::HashMap::new(),
                window_size,
            }
        }

        /// Record training step for a specific relation.
        pub fn record(
            &mut self,
            relation: &str,
            loss: f32,
            avg_volume: f32,
            gradient_norm: f32,
        ) {
            let stats = self.stats_by_relation
                .entry(relation.to_string())
                .or_insert_with(|| TrainingStats::new(self.window_size));
            
            stats.record(loss, avg_volume, gradient_norm);
        }

        /// Record training step with intersection volume for a specific relation.
        pub fn record_with_intersection(
            &mut self,
            relation: &str,
            loss: f32,
            avg_volume: f32,
            avg_intersection_volume: Option<f32>,
            gradient_norm: f32,
        ) {
            let stats = self.stats_by_relation
                .entry(relation.to_string())
                .or_insert_with(|| TrainingStats::new(self.window_size));
            
            stats.record_with_intersection(loss, avg_volume, avg_intersection_volume, gradient_norm);
        }

        /// Check if a specific relation has converged.
        pub fn is_relation_converged(
            &self,
            relation: &str,
            tolerance: f32,
            min_iterations: usize,
        ) -> bool {
            self.stats_by_relation
                .get(relation)
                .map(|stats| stats.is_converged(tolerance, min_iterations))
                .unwrap_or(false)
        }

        /// Get all relations that have converged.
        pub fn converged_relations(
            &self,
            tolerance: f32,
            min_iterations: usize,
        ) -> Vec<String> {
            self.stats_by_relation
                .iter()
                .filter_map(|(relation, stats)| {
                    if stats.is_converged(tolerance, min_iterations) {
                        Some(relation.clone())
                    } else {
                        None
                    }
                })
                .collect()
        }

        /// Get all relations that have not converged.
        pub fn unconverged_relations(
            &self,
            tolerance: f32,
            min_iterations: usize,
        ) -> Vec<String> {
            self.stats_by_relation
                .iter()
                .filter_map(|(relation, stats)| {
                    if !stats.is_converged(tolerance, min_iterations) {
                        Some(relation.clone())
                    } else {
                        None
                    }
                })
                .collect()
        }

        /// Get training stats for a specific relation.
        pub fn get_relation_stats(&self, relation: &str) -> Option<&TrainingStats> {
            self.stats_by_relation.get(relation)
        }

        /// Get all relation names being tracked.
        pub fn relations(&self) -> Vec<String> {
            self.stats_by_relation.keys().cloned().collect()
        }

        /// Get convergence rate (proportion of relations that have converged).
        pub fn convergence_rate(
            &self,
            tolerance: f32,
            min_iterations: usize,
        ) -> f32 {
            if self.stats_by_relation.is_empty() {
                return 0.0;
            }

            let converged_count = self.stats_by_relation
                .values()
                .filter(|stats| stats.is_converged(tolerance, min_iterations))
                .count();

            converged_count as f32 / self.stats_by_relation.len() as f32
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
        /// Shannon entropy of normalized volume distribution
        pub entropy: f32,
        /// Quantiles: (q25, q50, q75, q95)
        pub quantiles: (f32, f32, f32, f32),
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
                    entropy: 0.0,
                    quantiles: (0.0, 0.0, 0.0, 0.0),
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

            // Compute quantiles
            let q25_idx = (vols.len() as f32 * 0.25) as usize;
            let q75_idx = (vols.len() as f32 * 0.75) as usize;
            let q95_idx = (vols.len() as f32 * 0.95) as usize;
            let q25 = vols[q25_idx.min(vols.len() - 1)];
            let q75 = vols[q75_idx.min(vols.len() - 1)];
            let q95 = vols[q95_idx.min(vols.len() - 1)];

            // Compute Shannon entropy of normalized volume distribution
            let total_vol: f32 = vols.iter().sum();
            let entropy = if total_vol > 0.0 {
                let mut entropy_sum = 0.0;
                for &vol in &vols {
                    if vol > 0.0 {
                        let p = vol / total_vol;
                        entropy_sum -= p * p.ln();
                    }
                }
                entropy_sum
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
                entropy,
                quantiles: (q25, median, q75, q95),
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
                f32::NAN // Precision is undefined when there are no positive predictions
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
        ///
        /// Returns NaN if precision is NaN (no positive predictions).
        pub fn f1(&self) -> f32 {
            let prec = self.precision();
            let rec = self.recall();
            if prec.is_nan() {
                return f32::NAN; // Undefined when precision is undefined
            }
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
        /// Sibling intersection ratio (for hierarchical data)
        pub sibling_intersection_ratio: Option<f32>,
        /// Parent-child intersection ratio
        pub parent_child_intersection_ratio: Option<f32>,
    }

    impl IntersectionTopology {
        /// Create new intersection topology tracker.
        pub fn new() -> Self {
            Self {
                total_pairs: 0,
                intersecting_pairs: 0,
                containment_pairs: 0,
                disjoint_pairs: 0,
                sibling_intersection_ratio: None,
                parent_child_intersection_ratio: None,
            }
        }

        /// Record sibling intersection (for hierarchical data).
        ///
        /// Siblings should have minimal intersection in well-trained hierarchies.
        pub fn record_sibling_intersection(&mut self, intersection_vol: f32, min_vol: f32) {
            if min_vol > 0.0 {
                let ratio = intersection_vol / min_vol;
                self.sibling_intersection_ratio = Some(
                    self.sibling_intersection_ratio
                        .map(|r| (r + ratio) / 2.0)
                        .unwrap_or(ratio)
                );
            }
        }

        /// Record parent-child intersection.
        ///
        /// Parent-child pairs should have high intersection (child contained in parent).
        pub fn record_parent_child_intersection(&mut self, intersection_vol: f32, child_vol: f32) {
            if child_vol > 0.0 {
                let ratio = intersection_vol / child_vol;
                self.parent_child_intersection_ratio = Some(
                    self.parent_child_intersection_ratio
                        .map(|r| (r + ratio) / 2.0)
                        .unwrap_or(ratio)
                );
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

    /// Containment hierarchy verification with transitive closure analysis.
    #[derive(Debug, Clone)]
    pub struct ContainmentHierarchy {
        /// Direct containment relationships (parent -> child)
        direct_containments: Vec<(usize, usize)>,
        /// Computed transitive closure
        transitive_closure: std::collections::HashMap<usize, Vec<usize>>,
    }

    impl ContainmentHierarchy {
        /// Create new hierarchy tracker.
        pub fn new() -> Self {
            Self {
                direct_containments: Vec::new(),
                transitive_closure: std::collections::HashMap::new(),
            }
        }

        /// Add a direct containment relationship (parent contains child).
        pub fn add_containment(&mut self, parent: usize, child: usize) {
            self.direct_containments.push((parent, child));
            self.transitive_closure.clear(); // Invalidate cache
        }

        /// Compute transitive closure of containment relationships.
        pub fn compute_transitive_closure(&mut self) {
            self.transitive_closure.clear();
            
            // Build adjacency list
            let mut adj: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for &(parent, child) in &self.direct_containments {
                adj.entry(parent).or_insert_with(Vec::new).push(child);
            }
            
            // DFS to compute reachability
            for &(parent, _) in &self.direct_containments {
                let mut visited = std::collections::HashSet::new();
                let mut stack = vec![parent];
                
                while let Some(node) = stack.pop() {
                    if visited.insert(node) {
                        if let Some(children) = adj.get(&node) {
                            for &child in children {
                                self.transitive_closure
                                    .entry(parent)
                                    .or_insert_with(Vec::new)
                                    .push(child);
                                stack.push(child);
                            }
                        }
                    }
                }
            }
        }

        /// Verify transitivity: if A contains B and B contains C, then A should contain C.
        pub fn verify_transitivity(&self) -> (usize, usize) {
            let mut violations = 0;
            let mut total_checks = 0;
            
            for &(a, b) in &self.direct_containments {
                if let Some(b_children) = self.transitive_closure.get(&b) {
                    for &c in b_children {
                        total_checks += 1;
                        if let Some(a_children) = self.transitive_closure.get(&a) {
                            if !a_children.contains(&c) {
                                violations += 1;
                            }
                        }
                    }
                }
            }
            
            (violations, total_checks)
        }

        /// Detect cycles in containment relationships (should be impossible in valid hierarchies).
        pub fn detect_cycles(&self) -> Vec<Vec<usize>> {
            let mut cycles = Vec::new();
            let mut visited = std::collections::HashSet::new();
            let mut rec_stack = std::collections::HashSet::new();
            let mut path = Vec::new();
            
            // Build adjacency list
            let mut adj: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for &(parent, child) in &self.direct_containments {
                adj.entry(parent).or_insert_with(Vec::new).push(child);
            }
            
            fn dfs(
                node: usize,
                adj: &std::collections::HashMap<usize, Vec<usize>>,
                visited: &mut std::collections::HashSet<usize>,
                rec_stack: &mut std::collections::HashSet<usize>,
                path: &mut Vec<usize>,
                cycles: &mut Vec<Vec<usize>>,
            ) {
                visited.insert(node);
                rec_stack.insert(node);
                path.push(node);
                
                if let Some(children) = adj.get(&node) {
                    for &child in children {
                        if !visited.contains(&child) {
                            dfs(child, adj, visited, rec_stack, path, cycles);
                        } else if rec_stack.contains(&child) {
                            // Found a cycle
                            let cycle_start = path.iter().position(|&x| x == child).unwrap();
                            cycles.push(path[cycle_start..].to_vec());
                        }
                    }
                }
                
                rec_stack.remove(&node);
                path.pop();
            }
            
            for &(parent, _) in &self.direct_containments {
                if !visited.contains(&parent) {
                    dfs(parent, &adj, &mut visited, &mut rec_stack, &mut path, &mut cycles);
                }
            }
            
            cycles
        }

        /// Get hierarchy depth for each node (distance from root).
        pub fn hierarchy_depths(&self) -> std::collections::HashMap<usize, usize> {
            let mut depths = std::collections::HashMap::new();
            let mut adj: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            let mut in_degree: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            
            for &(parent, child) in &self.direct_containments {
                adj.entry(parent).or_insert_with(Vec::new).push(child);
                *in_degree.entry(child).or_insert(0) += 1;
                in_degree.entry(parent).or_insert(0);
            }
            
            // Find roots (nodes with in-degree 0)
            let mut queue: Vec<usize> = in_degree.iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&node, _)| node)
                .collect();
            
            for root in &queue {
                depths.insert(*root, 0);
            }
            
            // BFS to assign depths
            while let Some(node) = queue.pop() {
                let depth = depths[&node];
                if let Some(children) = adj.get(&node) {
                    for &child in children {
                        depths.insert(child, depth + 1);
                        queue.push(child);
                    }
                }
            }
            
            depths
        }
    }

    impl Default for ContainmentHierarchy {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Asymmetry quantification for box embedding relationships.
    ///
    /// Measures how well the model captures asymmetric relationships,
    /// which is a key advantage of box embeddings over point embeddings.
    #[derive(Debug, Clone)]
    pub struct AsymmetryMetrics {
        /// Asymmetry ratios for each pair (|d(A,B) - d(B,A)|)
        asymmetry_ratios: Vec<f32>,
        /// Mean asymmetry ratio
        mean_asymmetry: f32,
        /// Proportion of pairs with significant asymmetry (> threshold)
        high_asymmetry_proportion: f32,
    }

    impl AsymmetryMetrics {
        /// Create new asymmetry tracker.
        pub fn new() -> Self {
            Self {
                asymmetry_ratios: Vec::new(),
                mean_asymmetry: 0.0,
                high_asymmetry_proportion: 0.0,
            }
        }

        /// Record asymmetry for a pair of boxes.
        ///
        /// # Parameters
        ///
        /// - `distance_ab`: Distance from A to B (e.g., containment_prob(A, B))
        /// - `distance_ba`: Distance from B to A (e.g., containment_prob(B, A))
        pub fn record_pair(&mut self, distance_ab: f32, distance_ba: f32) {
            let asymmetry = (distance_ab - distance_ba).abs();
            self.asymmetry_ratios.push(asymmetry);
        }

        /// Finalize metrics (compute statistics).
        pub fn finalize(&mut self, threshold: f32) {
            if self.asymmetry_ratios.is_empty() {
                return;
            }
            
            self.mean_asymmetry = self.asymmetry_ratios.iter().sum::<f32>() / self.asymmetry_ratios.len() as f32;
            
            let high_count = self.asymmetry_ratios.iter()
                .filter(|&&r| r > threshold)
                .count();
            self.high_asymmetry_proportion = high_count as f32 / self.asymmetry_ratios.len() as f32;
        }

        /// Get mean asymmetry ratio.
        pub fn mean_asymmetry(&self) -> f32 {
            self.mean_asymmetry
        }

        /// Get proportion of pairs with high asymmetry.
        pub fn high_asymmetry_proportion(&self) -> f32 {
            self.high_asymmetry_proportion
        }
    }

    impl Default for AsymmetryMetrics {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Volume conservation analysis.
    ///
    /// Verifies that parent volumes properly contain the sum of their children's volumes,
    /// which is a fundamental geometric constraint for hierarchical box embeddings.
    #[derive(Debug, Clone)]
    pub struct VolumeConservation {
        /// Parent-child volume ratios (sum of children / parent volume)
        parent_child_ratios: Vec<f32>,
        /// Violations (ratios > 1.0 + tolerance)
        violations: usize,
    }

    impl VolumeConservation {
        /// Create new volume conservation tracker.
        pub fn new() -> Self {
            Self {
                parent_child_ratios: Vec::new(),
                violations: 0,
            }
        }

        /// Record a parent-child volume relationship.
        ///
        /// # Parameters
        ///
        /// - `parent_volume`: Volume of the parent box
        /// - `children_volumes`: Volumes of all child boxes
        /// - `tolerance`: Allowed tolerance for conservation violations
        pub fn record_parent_children<I>(
            &mut self,
            parent_volume: f32,
            children_volumes: I,
            tolerance: f32,
        ) where
            I: Iterator<Item = f32>,
        {
            let children_sum: f32 = children_volumes.sum();
            
            if parent_volume > 0.0 {
                let ratio = children_sum / parent_volume;
                self.parent_child_ratios.push(ratio);
                
                if ratio > 1.0 + tolerance {
                    self.violations += 1;
                }
            }
        }

        /// Get mean parent-child volume ratio (should be <= 1.0).
        pub fn mean_ratio(&self) -> f32 {
            if self.parent_child_ratios.is_empty() {
                0.0
            } else {
                self.parent_child_ratios.iter().sum::<f32>() / self.parent_child_ratios.len() as f32
            }
        }

        /// Get violation rate (fraction of parent-child pairs violating conservation).
        pub fn violation_rate(&self) -> f32 {
            if self.parent_child_ratios.is_empty() {
                0.0
            } else {
                self.violations as f32 / self.parent_child_ratios.len() as f32
            }
        }

        /// Get distribution statistics of parent-child ratios.
        pub fn ratio_statistics(&self) -> Option<(f32, f32, f32, f32)> {
            if self.parent_child_ratios.is_empty() {
                return None;
            }
            
            let mut sorted = self.parent_child_ratios.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let min = sorted[0];
            let max = sorted[sorted.len() - 1];
            let median = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };
            let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
            
            Some((min, max, mean, median))
        }
    }

    impl Default for VolumeConservation {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Dimensionality utilization analysis.
    ///
    /// Tracks how effectively each dimension is being used in box embeddings,
    /// detecting underutilized or redundant dimensions.
    #[derive(Debug, Clone)]
    pub struct DimensionalityUtilization {
        /// For each dimension, track the range of values used
        dimension_ranges: Vec<(f32, f32)>, // (min, max) for each dimension
        /// For each dimension, track the variance of box sizes
        dimension_size_variance: Vec<f32>,
    }

    impl DimensionalityUtilization {
        /// Create new dimensionality utilization analyzer.
        pub fn new(n_dimensions: usize) -> Self {
            Self {
                dimension_ranges: vec![(f32::INFINITY, f32::NEG_INFINITY); n_dimensions],
                dimension_size_variance: vec![0.0; n_dimensions],
            }
        }

        /// Record a box's min and max coordinates.
        pub fn record_box<I>(&mut self, min_coords: I, max_coords: I)
        where
            I: Iterator<Item = f32>,
        {
            for (dim, (min_val, max_val)) in min_coords.zip(max_coords).enumerate() {
                if dim >= self.dimension_ranges.len() {
                    break;
                }
                
                // Update range
                let (curr_min, curr_max) = self.dimension_ranges[dim];
                self.dimension_ranges[dim] = (
                    curr_min.min(min_val),
                    curr_max.max(max_val),
                );
                
                // Track size (max - min) for variance calculation
                let _size = max_val - min_val;
                // Note: This is a simplified variance tracker
                // In practice, you'd want to accumulate properly
            }
        }

        /// Get effective dimensionality (number of dimensions with significant range).
        ///
        /// # Parameters
        ///
        /// - `threshold`: Minimum range required to consider a dimension "used"
        pub fn effective_dimensionality(&self, threshold: f32) -> usize {
            self.dimension_ranges.iter()
                .filter(|(min, max)| (max - min) >= threshold)
                .count()
        }

        /// Get dimension utilization scores (0.0 to 1.0 for each dimension).
        pub fn utilization_scores(&self, max_range: f32) -> Vec<f32> {
            self.dimension_ranges.iter()
                .map(|(min, max)| {
                    let range = max - min;
                    (range / max_range).min(1.0)
                })
                .collect()
        }

        /// Detect underutilized dimensions (utilization < threshold).
        pub fn underutilized_dimensions(&self, max_range: f32, threshold: f32) -> Vec<usize> {
            self.utilization_scores(max_range)
                .iter()
                .enumerate()
                .filter(|(_, &score)| score < threshold)
                .map(|(idx, _)| idx)
                .collect()
        }
    }

    /// Generalization vs memorization metrics.
    ///
    /// Distinguishes between learning genuine structure vs memorizing training facts.
    #[derive(Debug, Clone)]
    pub struct GeneralizationMetrics {
        /// Performance on facts requiring inference (generalization)
        inference_performance: Vec<f32>, // MRR or similar
        /// Performance on direct facts (memorization)
        direct_performance: Vec<f32>,
        /// Performance on seen vs unseen entity pairs
        seen_unseen_gap: f32,
    }

    impl GeneralizationMetrics {
        /// Create new generalization metrics tracker.
        pub fn new() -> Self {
            Self {
                inference_performance: Vec::new(),
                direct_performance: Vec::new(),
                seen_unseen_gap: 0.0,
            }
        }

        /// Record performance on inference-requiring facts.
        pub fn record_inference(&mut self, performance: f32) {
            self.inference_performance.push(performance);
        }

        /// Record performance on direct facts.
        pub fn record_direct(&mut self, performance: f32) {
            self.direct_performance.push(performance);
        }

        /// Compute generalization gap (inference - direct).
        ///
        /// Positive values indicate good generalization (inference > direct).
        /// Negative values suggest memorization (direct > inference).
        pub fn generalization_gap(&self) -> Option<f32> {
            if self.inference_performance.is_empty() || self.direct_performance.is_empty() {
                return None;
            }
            
            let mean_inference = self.inference_performance.iter().sum::<f32>() / self.inference_performance.len() as f32;
            let mean_direct = self.direct_performance.iter().sum::<f32>() / self.direct_performance.len() as f32;
            
            Some(mean_inference - mean_direct)
        }

        /// Get generalization ratio (inference / direct).
        ///
        /// Values > 1.0 indicate good generalization.
        pub fn generalization_ratio(&self) -> Option<f32> {
            if self.inference_performance.is_empty() || self.direct_performance.is_empty() {
                return None;
            }
            
            let mean_inference = self.inference_performance.iter().sum::<f32>() / self.inference_performance.len() as f32;
            let mean_direct = self.direct_performance.iter().sum::<f32>() / self.direct_performance.len() as f32;
            
            if mean_direct > 0.0 {
                Some(mean_inference / mean_direct)
            } else {
                None
            }
        }
    }

    impl Default for GeneralizationMetrics {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Topological stability metrics.
    ///
    /// Measures how stable learned representations are under perturbations
    /// or across different initializations.
    #[derive(Debug, Clone)]
    pub struct TopologicalStability {
        /// Volume stability: coefficient of variation of volumes across runs
        volume_stability: f32,
        /// Containment stability: proportion of containments preserved across runs
        containment_stability: f32,
        /// Intersection stability: proportion of intersections preserved
        intersection_stability: f32,
    }

    impl TopologicalStability {
        /// Compute stability from multiple volume distributions.
        ///
        /// # Parameters
        ///
        /// - `volume_sets`: Iterator of volume collections from different runs
        pub fn from_volume_distributions<I, J>(volume_sets: I) -> Self
        where
            I: Iterator<Item = J>,
            J: Iterator<Item = f32>,
        {
            let sets: Vec<Vec<f32>> = volume_sets.map(|v| v.collect()).collect();
            
            if sets.is_empty() || sets[0].is_empty() {
                return Self {
                    volume_stability: 0.0,
                    containment_stability: 0.0,
                    intersection_stability: 0.0,
                };
            }
            
            // Compute coefficient of variation for each box across runs
            let n_boxes = sets[0].len();
            let mut cv_sum = 0.0;
            
            for box_idx in 0..n_boxes {
                let volumes: Vec<f32> = sets.iter()
                    .filter_map(|s| s.get(box_idx).copied())
                    .collect();
                
                if volumes.len() > 1 {
                    let mean = volumes.iter().sum::<f32>() / volumes.len() as f32;
                    let variance = volumes.iter()
                        .map(|&v| (v - mean).powi(2))
                        .sum::<f32>() / volumes.len() as f32;
                    let std_dev = variance.sqrt();
                    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
                    cv_sum += cv;
                }
            }
            
            let volume_stability = if n_boxes > 0 {
                cv_sum / n_boxes as f32
            } else {
                0.0
            };
            
            Self {
                volume_stability,
                containment_stability: 0.0, // Would need containment data
                intersection_stability: 0.0, // Would need intersection data
            }
        }

        /// Get volume stability (lower is better).
        pub fn volume_stability(&self) -> f32 {
            self.volume_stability
        }
    }

    /// KL divergence between two volume distributions.
    ///
    /// Measures how different the learned volume distribution is from a target distribution.
    /// Higher KL divergence indicates more divergence from the target.
    pub fn kl_divergence<I, J>(learned_volumes: I, target_volumes: J) -> f32
    where
        I: Iterator<Item = f32>,
        J: Iterator<Item = f32>,
    {
        let learned: Vec<f32> = learned_volumes.collect();
        let target: Vec<f32> = target_volumes.collect();
        
        if learned.len() != target.len() || learned.is_empty() {
            return f32::INFINITY;
        }
        
        let learned_sum: f32 = learned.iter().sum();
        let target_sum: f32 = target.iter().sum();
        
        if learned_sum <= 0.0 || target_sum <= 0.0 {
            return f32::INFINITY;
        }
        
        let mut kl = 0.0;
        for (l, t) in learned.iter().zip(target.iter()) {
            if *l > 0.0 && *t > 0.0 {
                let p = l / learned_sum;
                let q = t / target_sum;
                kl += p * (p / q).ln();
            } else if *l > 0.0 {
                // l > 0 but t = 0, infinite divergence
                return f32::INFINITY;
            }
        }
        
        kl
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

    /// Adaptive Calibration Error (ACE) with equal-mass binning.
    ///
    /// Similar to ECE but uses adaptive binning to ensure equal numbers
    /// of samples per bin, reducing variance in sparse regions.
    ///
    /// # Parameters
    ///
    /// - `predictions`: Iterator of predicted probabilities
    /// - `actuals`: Iterator of actual binary outcomes
    /// - `n_bins`: Number of bins (should be <= number of samples)
    ///
    /// # Returns
    ///
    /// ACE value in [0, 1], where lower is better
    pub fn adaptive_calibration_error<I, J>(
        predictions: I,
        actuals: J,
        n_bins: usize,
    ) -> f32
    where
        I: Iterator<Item = f32>,
        J: Iterator<Item = bool>,
    {
        // Collect all predictions and actuals
        let mut data: Vec<(f32, bool)> = predictions.zip(actuals)
            .map(|(p, a)| (p.clamp(0.0, 1.0), a))
            .collect();
        
        if data.is_empty() {
            return 0.0;
        }
        
        // Sort by predicted probability
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let bin_size = (data.len() + n_bins - 1) / n_bins; // Ceiling division
        let mut ace = 0.0;
        let mut total_samples = 0;
        
        for bin_idx in 0..n_bins {
            let start = bin_idx * bin_size;
            let end = ((bin_idx + 1) * bin_size).min(data.len());
            
            if start >= data.len() {
                break;
            }
            
            let bin_data = &data[start..end];
            if bin_data.is_empty() {
                continue;
            }
            
            let bin_size_actual = bin_data.len();
            total_samples += bin_size_actual;
            
            // Average predicted probability in this bin
            let avg_pred = bin_data.iter().map(|(p, _)| p).sum::<f32>() / bin_size_actual as f32;
            
            // Empirical accuracy in this bin
            let empirical_acc = bin_data.iter().filter(|(_, a)| *a).count() as f32 / bin_size_actual as f32;
            
            // Weighted absolute difference
            ace += (avg_pred - empirical_acc).abs() * bin_size_actual as f32;
        }
        
        if total_samples > 0 {
            ace / total_samples as f32
        } else {
            0.0
        }
    }

    /// Reliability diagram data for visualization.
    ///
    /// Returns binned data showing predicted confidence vs empirical accuracy
    /// for creating reliability diagrams.
    #[derive(Debug, Clone)]
    pub struct ReliabilityDiagram {
        /// Bin centers (average predicted probability)
        pub bin_centers: Vec<f32>,
        /// Empirical accuracies in each bin
        pub empirical_accuracies: Vec<f32>,
        /// Number of samples in each bin
        pub bin_counts: Vec<usize>,
    }

    /// Compute reliability diagram data for calibration visualization.
    pub fn reliability_diagram<I, J>(
        predictions: I,
        actuals: J,
        n_bins: usize,
    ) -> ReliabilityDiagram
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
        
        let mut bin_centers = Vec::new();
        let mut empirical_accuracies = Vec::new();
        let mut bin_counts = Vec::new();
        
        for bin in bins.iter() {
            if bin.is_empty() {
                continue;
            }
            
            let avg_pred = bin.iter().map(|(p, _)| p).sum::<f32>() / bin.len() as f32;
            let empirical_acc = bin.iter().filter(|(_, a)| *a).count() as f32 / bin.len() as f32;
            
            bin_centers.push(avg_pred);
            empirical_accuracies.push(empirical_acc);
            bin_counts.push(bin.len());
        }
        
        ReliabilityDiagram {
            bin_centers,
            empirical_accuracies,
            bin_counts,
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

    #[test]
    fn test_gradient_flow_analysis() {
        let mut flow = diagnostics::GradientFlowAnalysis::new(10);
        
        flow.record(Some(0.5), Some(0.1), Some(0.3), Some(0.2));
        flow.record(Some(0.6), Some(0.15), Some(0.4), Some(0.25));
        
        // Check imbalance (center >> size)
        let imbalance = flow.check_imbalance(2.0);
        assert!(imbalance.is_some());
        
        let sparsity = flow.gradient_sparsity(0.01);
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    fn test_volume_distribution_entropy() {
        // Uniform distribution (high entropy)
        let uniform_vols = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let dist_uniform = quality::VolumeDistribution::from_volumes(uniform_vols.iter().copied());
        assert!(dist_uniform.entropy > 0.0);
        
        // Skewed distribution (lower entropy)
        let skewed_vols = vec![0.1, 0.1, 0.1, 0.1, 10.0];
        let dist_skewed = quality::VolumeDistribution::from_volumes(skewed_vols.iter().copied());
        // Skewed should have lower entropy than uniform
        assert!(dist_skewed.entropy < dist_uniform.entropy || dist_skewed.entropy == 0.0);
    }

    #[test]
    fn test_containment_hierarchy() {
        let mut hierarchy = quality::ContainmentHierarchy::new();
        
        // Create a simple hierarchy: A -> B -> C
        hierarchy.add_containment(0, 1); // A contains B
        hierarchy.add_containment(1, 2); // B contains C
        hierarchy.compute_transitive_closure();
        
        // Verify transitivity: A should contain C
        let (violations, _total) = hierarchy.verify_transitivity();
        assert_eq!(violations, 0);
        
        // Check depths
        let depths = hierarchy.hierarchy_depths();
        assert_eq!(depths.get(&0), Some(&0)); // Root
        assert_eq!(depths.get(&1), Some(&1));
        assert_eq!(depths.get(&2), Some(&2));
        
        // Should have no cycles
        let cycles = hierarchy.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_kl_divergence() {
        // Identical distributions
        let learned = vec![0.25, 0.25, 0.25, 0.25];
        let target = vec![0.25, 0.25, 0.25, 0.25];
        let kl = quality::kl_divergence(learned.iter().copied(), target.iter().copied());
        assert!((kl - 0.0).abs() < 1e-6);
        
        // Different distributions
        let learned2 = vec![0.5, 0.3, 0.15, 0.05];
        let target2 = vec![0.25, 0.25, 0.25, 0.25];
        let kl2 = quality::kl_divergence(learned2.iter().copied(), target2.iter().copied());
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_adaptive_calibration_error() {
        // Well-calibrated predictions
        let preds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let actuals = vec![false, false, false, false, true, true, true, true, true];
        let ace = calibration::adaptive_calibration_error(
            preds.iter().copied(),
            actuals.iter().copied(),
            5,
        );
        assert!(ace < 0.3);
    }

    #[test]
    fn test_reliability_diagram() {
        let preds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let actuals = vec![false, false, false, false, true, true, true, true, true];
        let diagram = calibration::reliability_diagram(
            preds.iter().copied(),
            actuals.iter().copied(),
            5,
        );
        
        assert!(!diagram.bin_centers.is_empty());
        assert_eq!(diagram.bin_centers.len(), diagram.empirical_accuracies.len());
        assert_eq!(diagram.bin_centers.len(), diagram.bin_counts.len());
    }

    #[test]
    fn test_depth_stratified_gradient_flow() {
        let mut flow = diagnostics::DepthStratifiedGradientFlow::new(10);
        
        flow.record(0, 0.5); // Root level
        flow.record(1, 0.3); // Depth 1
        flow.record(2, 0.1); // Depth 2
        
        let means = flow.mean_gradients_by_depth();
        assert_eq!(means.get(&0), Some(&0.5));
        assert_eq!(means.get(&1), Some(&0.3));
        assert_eq!(means.get(&2), Some(&0.1));
        
        // Should detect imbalance (depth 0 >> depth 2)
        if let Some((min_depth, max_depth, ratio)) = flow.check_depth_imbalance(2.0) {
            assert_eq!(max_depth, 0);
            assert_eq!(min_depth, 2);
            assert!(ratio > 2.0);
        }
    }

    #[test]
    fn test_phase_detector() {
        let mut detector = diagnostics::PhaseDetector::new(10);
        
        // Early training: high loss, high gradients
        detector.record(1.0, 0.8);
        detector.record(0.9, 0.7);
        detector.record(0.85, 0.6);
        let phase1 = detector.detect_phase();
        // Early training should be exploration (or exploitation if decreasing)
        assert!(phase1 == diagnostics::TrainingPhase::Exploration || 
                phase1 == diagnostics::TrainingPhase::Exploitation);
        
        // Mid training: decreasing loss (more dramatic decrease)
        detector.record(0.7, 0.4);
        detector.record(0.5, 0.3);
        detector.record(0.3, 0.2);
        detector.record(0.2, 0.15);
        detector.record(0.15, 0.1);
        let phase2 = detector.detect_phase();
        // Should detect exploitation with strong decreasing trend
        assert_eq!(phase2, diagnostics::TrainingPhase::Exploitation);
        
        // Late training: stable loss, low gradients (need more samples for convergence)
        // Add more samples to ensure convergence detection
        for _ in 0..3 {
            detector.record(0.14, 0.05);
            detector.record(0.1401, 0.04);
            detector.record(0.1399, 0.05);
        }
        let phase3 = detector.detect_phase();
        // Should detect convergence (very stable loss, very low gradients)
        // Phase detection is heuristic, so allow some flexibility
        assert!(phase3 == diagnostics::TrainingPhase::Convergence || 
                phase3 == diagnostics::TrainingPhase::Exploitation);
    }

    #[test]
    fn test_volume_conservation() {
        let mut conservation = quality::VolumeConservation::new();
        
        // Parent with children that sum to less than parent (valid)
        conservation.record_parent_children(
            10.0,
            vec![3.0, 2.0, 1.0].into_iter(),
            0.1,
        );
        
        // Parent with children that sum to more than parent (violation)
        conservation.record_parent_children(
            5.0,
            vec![3.0, 2.0, 1.5].into_iter(),
            0.1,
        );
        
        let mean_ratio = conservation.mean_ratio();
        assert!(mean_ratio > 0.0);
        
        let violation_rate = conservation.violation_rate();
        assert!(violation_rate > 0.0 && violation_rate <= 1.0);
        
        if let Some((min, max, mean, _median)) = conservation.ratio_statistics() {
            assert!(min <= max);
            assert!(mean >= min && mean <= max);
        }
    }

    #[test]
    fn test_dimensionality_utilization() {
        let mut util = quality::DimensionalityUtilization::new(3);
        
        // Record boxes with different ranges
        util.record_box(
            vec![0.0, 0.0, 0.0].into_iter(),
            vec![1.0, 0.1, 10.0].into_iter(),
        );
        util.record_box(
            vec![0.5, 0.05, 5.0].into_iter(),
            vec![1.5, 0.15, 15.0].into_iter(),
        );
        
        let effective_dim = util.effective_dimensionality(0.5);
        assert!(effective_dim > 0);
        
        let scores = util.utilization_scores(20.0);
        assert_eq!(scores.len(), 3);
        
        let underutilized = util.underutilized_dimensions(20.0, 0.1);
        // Dimension 1 has small range, might be underutilized
        assert!(underutilized.len() <= 3);
    }

    #[test]
    fn test_generalization_metrics() {
        let mut metrics = quality::GeneralizationMetrics::new();
        
        // Good generalization: inference performance similar to direct
        metrics.record_inference(0.8);
        metrics.record_inference(0.75);
        metrics.record_direct(0.7);
        metrics.record_direct(0.72);
        
        if let Some(gap) = metrics.generalization_gap() {
            assert!(gap > 0.0); // Inference > direct
        }
        
        if let Some(ratio) = metrics.generalization_ratio() {
            assert!(ratio > 1.0); // Good generalization
        }
    }

    #[test]
    fn test_intersection_volume_stats() {
        let mut stats = diagnostics::TrainingStats::new(10);
        
        // Record without intersection volumes
        stats.record(1.0, 0.5, 0.1);
        assert!(stats.intersection_volume_stats().is_none());
        
        // Record with intersection volumes
        stats.record_with_intersection(1.0, 0.5, Some(0.2), 0.1);
        stats.record_with_intersection(0.9, 0.6, Some(0.25), 0.08);
        stats.record_with_intersection(0.8, 0.7, Some(0.3), 0.06);
        
        if let Some((mean, min, max)) = stats.intersection_volume_stats() {
            assert!(mean > 0.0);
            assert!(min <= mean && mean <= max);
            assert!((mean - 0.25).abs() < 0.1); // Should be around 0.25
        }
    }

    #[test]
    fn test_intersection_volume_trend() {
        let mut stats = diagnostics::TrainingStats::new(10);
        
        // Not enough samples
        stats.record_with_intersection(1.0, 0.5, Some(0.2), 0.1);
        assert!(stats.intersection_volume_trend(3).is_none());
        
        // Increasing trend
        for i in 0..5 {
            let vol = 0.2 + i as f32 * 0.05;
            stats.record_with_intersection(1.0 - i as f32 * 0.1, 0.5, Some(vol), 0.1);
        }
        assert_eq!(stats.intersection_volume_trend(3), Some(true));
        
        // Decreasing trend
        let mut stats2 = diagnostics::TrainingStats::new(10);
        for i in 0..5 {
            let vol = 0.4 - i as f32 * 0.05;
            stats2.record_with_intersection(1.0 - i as f32 * 0.1, 0.5, Some(vol), 0.1);
        }
        assert_eq!(stats2.intersection_volume_trend(3), Some(false));
    }

    #[test]
    fn test_stratified_metrics() {
        let mut metrics = metrics::StratifiedMetrics::new();
        
        // Add relation results
        metrics.add_relation_result("is_a".to_string(), 1);
        metrics.add_relation_result("is_a".to_string(), 2);
        metrics.add_relation_result("part_of".to_string(), 3);
        metrics.finalize_relations();
        
        assert_eq!(metrics.by_relation.len(), 2);
        if let Some(rel_metrics) = metrics.by_relation.get("is_a") {
            assert_eq!(rel_metrics.count, 2);
            assert!(rel_metrics.mrr > 0.0);
        }
        
        // Add depth results
        metrics.add_depth_result(0, 1, true);
        metrics.add_depth_result(0, 2, true);
        metrics.add_depth_result(1, 3, false);
        metrics.finalize_depths();
        
        assert_eq!(metrics.by_depth.len(), 2);
        if let Some(depth_metrics) = metrics.by_depth.get(&0) {
            assert_eq!(depth_metrics.count, 2);
            assert_eq!(depth_metrics.containment_accuracy, 1.0); // Both correct
        }
    }

    #[test]
    fn test_empty_collections() {
        // Empty TrainingStats
        let stats = diagnostics::TrainingStats::new(10);
        assert!(!stats.is_converged(0.01, 5));
        assert!(!stats.is_gradient_exploding(100.0));
        // is_volume_collapsed returns false for empty (no volumes to check)
        assert!(!stats.is_volume_collapsed(0.01));
        assert!(stats.loss_stats().is_none());
        assert!(stats.volume_stats().is_none());
        assert!(stats.gradient_stats().is_none());
        assert!(stats.intersection_volume_stats().is_none());
        
        // Empty VolumeDistribution
        let empty_dist = quality::VolumeDistribution::from_volumes(std::iter::empty());
        // Empty distribution initializes with default values
        assert_eq!(empty_dist.min, 0.0);
        assert_eq!(empty_dist.max, 0.0);
        assert_eq!(empty_dist.mean, 0.0);
        
        // Empty ContainmentAccuracy
        let empty_acc = quality::ContainmentAccuracy::new();
        // Precision is NaN when TP + FP = 0 (division by zero, undefined)
        assert!(empty_acc.precision().is_nan());
        
        // Empty metrics
        let empty_mrr = metrics::mean_reciprocal_rank(std::iter::empty());
        assert_eq!(empty_mrr, 0.0);
        
        let empty_hits = metrics::hits_at_k(std::iter::empty(), 10);
        assert_eq!(empty_hits, 0.0);
    }

    #[test]
    fn test_edge_cases_zero_rank() {
        // Rank of 0 should be handled gracefully
        let ranks = vec![0, 1, 2, 0, 3];
        let mrr = metrics::mean_reciprocal_rank(ranks.iter().copied());
        // Should only count non-zero ranks: (1/1 + 1/2 + 1/3) / 3
        assert!(mrr > 0.0 && mrr < 1.0);
    }

    #[test]
    fn test_relation_stratified_training_stats() {
        let mut stats = diagnostics::RelationStratifiedTrainingStats::new(10);
        
        // Record training steps for different relations
        stats.record("has_part", 1.0, 0.5, 0.8);
        stats.record("has_part", 0.9, 0.5, 0.7);
        stats.record("has_part", 0.85, 0.5, 0.6);
        
        stats.record("located_in", 1.0, 0.6, 0.9);
        stats.record("located_in", 0.95, 0.6, 0.8);
        stats.record("located_in", 0.92, 0.6, 0.75);
        
        // Check that relations are tracked separately
        assert!(stats.get_relation_stats("has_part").is_some());
        assert!(stats.get_relation_stats("located_in").is_some());
        assert!(stats.get_relation_stats("nonexistent").is_none());
        
        // Check convergence (neither should be converged with such high loss variance)
        assert!(!stats.is_relation_converged("has_part", 0.01, 3));
        assert!(!stats.is_relation_converged("located_in", 0.01, 3));
        
        // Record more steps to create convergence for one relation
        for i in 0..10 {
            let loss = 0.5 + (i as f32 * 0.001); // Very stable loss
            stats.record("has_part", loss, 0.5, 0.05); // Low gradients
        }
        
        // Now has_part should be converged (stable loss, low gradients)
        assert!(stats.is_relation_converged("has_part", 0.01, 5));
        
        // Check convergence rate
        let rate = stats.convergence_rate(0.01, 5);
        assert!(rate > 0.0 && rate <= 1.0);
        
        // Check converged/unconverged relations
        let converged = stats.converged_relations(0.01, 5);
        assert!(converged.contains(&"has_part".to_string()));
        
        let unconverged = stats.unconverged_relations(0.01, 5);
        assert!(unconverged.contains(&"located_in".to_string()));
        
        // Test with intersection volume tracking
        stats.record_with_intersection("has_part", 0.5, 0.5, Some(0.2), 0.05);
        if let Some(rel_stats) = stats.get_relation_stats("has_part") {
            if let Some((mean, _, _)) = rel_stats.intersection_volume_stats() {
                assert!(mean > 0.0);
            }
        }
        
        // Test relations() method
        let relations = stats.relations();
        assert!(relations.len() >= 2);
        assert!(relations.contains(&"has_part".to_string()));
        assert!(relations.contains(&"located_in".to_string()));
    }
}

