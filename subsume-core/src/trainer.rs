//! Training utilities for box embeddings.
//!
//! Provides training loop infrastructure, negative sampling, and evaluation.
//!
//! # Research Background
//!
//! The training approach follows established practices from knowledge graph embedding literature:
//! - **Negative sampling**: Bordes et al. (2013) - TransE, adapted for box embeddings
//! - **Margin-based ranking loss**: Used in BoxE (Boratko et al., 2020) and many KG embedding methods
//! - **Evaluation metrics**: Standard link prediction metrics from Bordes et al. (2013) and subsequent work
//!
//! **Key Papers**:
//! - Bordes et al. (2013): "Translating Embeddings for Modeling Multi-relational Data" (TransE)
//!   - Introduces negative sampling and margin-based ranking loss for knowledge graphs
//! - Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
//!   - Adapts TransE-style training to box embeddings with translational bumps
//! - Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
//!   - Foundational work on box embeddings for knowledge graphs
//!
//! # Intuitive Overview
//!
//! Training box embeddings for knowledge graphs is like teaching the model to arrange boxes
//! in space so that geometric relationships (containment, overlap) match logical relationships
//! (subsumption, relatedness) in the knowledge graph.
//!
//! ## Paradigm Problem: Teaching the Model Hierarchical Relationships
//!
//! **The task**: Given a knowledge graph with triples like (dog, is_a, mammal) and
//! (mammal, is_a, animal), teach the model to arrange boxes so that geometric containment
//! matches logical relationships.
//!
//! **The training process** (step-by-step):
//!
//! 1. **Positive examples**: True facts like (Paris, located_in, France)
//!    - **Goal**: Teach that the France box should contain the Paris box (or be related
//!      via relation-specific transformations in BoxE)
//!    - **Method**: Compute containment probability, maximize it (high score = good)
//!
//! 2. **Negative examples**: Corrupted facts like (Paris, located_in, Germany)
//!    - **Goal**: Teach that the Germany box should NOT contain the Paris box
//!    - **Method**: Compute containment probability, minimize it (low score = good)
//!    - **Corruption strategy**: Replace tail with random entity (CorruptTail), or head
//!      (CorruptHead), or both (CorruptBoth)
//!
//! 3. **Loss function**: Measures how well the current box arrangement matches these constraints
//!    - Margin-based ranking loss: positive score should exceed negative score by a margin
//!    - If positive score = 0.9 and negative score = 0.1 with margin = 1.0, loss = 0 (good)
//!    - If positive score = 0.6 and negative score = 0.5 with margin = 1.0, loss > 0 (bad)
//!
//! 4. **Optimization**: Adjusts box positions/sizes to minimize loss
//!    - Gradient descent updates box coordinates
//!    - Over many iterations, boxes arrange themselves to satisfy constraints
//!
//! **Why negative sampling matters**: Without negative examples, the model could learn the
//! trivial solution: make all boxes contain everything. This satisfies all positive examples
//! but doesn't learn anything useful. Negative examples force discrimination—the model
//! must learn which containments are true and which are false.
//!
//! **Research foundation**: This training approach follows **Bordes et al. (2013)** for TransE,
//! adapted for box embeddings by **Boratko et al. (2020)**. The margin-based ranking loss and
//! negative sampling strategies are standard practice in knowledge graph embedding literature.

use crate::box_trait::BoxError;
use crate::dataset::Triple;
use crate::training::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank};
#[cfg(feature = "rand")]
use rand::Rng;
use std::collections::HashMap;
#[cfg(feature = "rand")]
use std::collections::HashSet;

/// Negative sampling strategy for training.
///
/// # Research Background
///
/// Negative sampling was introduced in knowledge graph embedding by **Bordes et al. (2013)**
/// for TransE and has become standard practice. The choice of corruption strategy significantly
/// affects what the model learns to distinguish.
///
/// **Reference**: Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
///
/// # Intuitive Explanation
///
/// Negative sampling creates "false facts" to contrast with true facts during training.
/// Different strategies work better for different types of knowledge graphs:
///
/// - **CorruptTail**: Replace the tail entity (e.g., (Paris, located_in, ?) → (Paris, located_in, Germany))
///   - Best for: Most knowledge graphs where relations are directional
///   - Why: Head entity often determines what tail makes sense
///
/// - **CorruptHead**: Replace the head entity (e.g., (?, located_in, France) → (Tokyo, located_in, France))
///   - Best for: Relations where tail constrains head (e.g., "part_of")
///   - Why: Some relations work better when we fix the "container" and vary the "contained"
///
/// - **CorruptBoth**: Replace both entities
///   - Best for: Symmetric relations (e.g., "sibling_of", "married_to")
///   - Why: These relations don't have clear head/tail directionality
///
/// - **Uniform**: Randomly corrupt either head or tail
///   - Best for: Balanced datasets or when relation directionality is unclear
///   - Why: Provides diverse negative examples
///
/// **The key insight**: The strategy affects what the model learns to distinguish. CorruptTail
/// teaches "given a head and relation, which tails are plausible?" while CorruptHead teaches
/// "given a tail and relation, which heads are plausible?"
#[derive(Debug, Clone)]
pub enum NegativeSamplingStrategy {
    /// Uniform random sampling (corrupts head or tail randomly)
    Uniform,
    /// Corrupt head entity (fix tail, vary head)
    CorruptHead,
    /// Corrupt tail entity (fix head, vary tail) - most common
    CorruptTail,
    /// Corrupt both (for symmetric relations)
    CorruptBoth,
}

/// Training configuration.
///
/// # Research Background
///
/// Hyperparameter ranges are based on empirical findings from:
/// - **BoxE paper** (Boratko et al., 2020): Learning rates, batch sizes, regularization
/// - **Gumbel-Box papers** (Dasgupta et al., 2020): Temperature scheduling for Gumbel boxes
/// - **Knowledge graph embedding literature**: Standard practices for negative sampling, margins
///
/// **Key References**:
/// - Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
/// - Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS)
///
/// # Intuitive Guide to Hyperparameters
///
/// These parameters control how the model learns box embeddings from knowledge graphs.
/// Understanding what each does helps you tune for your specific dataset.
///
/// ## Core Learning Parameters
///
/// - **`learning_rate`**: How big steps the optimizer takes
///   - Too high: Model overshoots optimal box positions (unstable training)
///   - Too low: Model learns very slowly (wastes compute)
///   - Sweet spot: Usually 1e-3 to 5e-4 for box embeddings
///
/// - **`batch_size`**: How many triples processed together
///   - Larger: More stable gradients, faster training, but needs more memory
///   - Smaller: More noisy gradients, slower training, but less memory
///   - Sweet spot: 512-2048 for most knowledge graphs
///
/// - **`epochs`**: How many times to see the training data
///   - Too few: Model doesn't learn enough (underfitting)
///   - Too many: Model memorizes training data (overfitting)
///   - Use early stopping to find the right number automatically
///
/// ## Negative Sampling Parameters
///
/// - **`negative_samples`**: How many false facts per true fact
///   - More negatives: Model learns better discrimination, but slower training
///   - Fewer negatives: Faster training, but may not learn fine distinctions
///   - Common: 1-5 negatives per positive
///
/// - **`negative_strategy`**: Which part of triple to corrupt (see [`NegativeSamplingStrategy`])
///
/// ## Regularization Parameters
///
/// - **`regularization_weight`**: Penalty for boxes being too large
///   - Prevents boxes from growing unbounded (which would make everything contain everything)
///   - Higher: Tighter, more specific boxes
///   - Lower: Larger, more general boxes
///   - Common: 0.01-0.1
///
/// - **`weight_decay`**: L2 regularization on box parameters
///   - Prevents overfitting by keeping box coordinates small
///   - Higher: Stronger regularization (simpler model)
///   - Lower: Less regularization (more complex model)
///
/// ## Box-Specific Parameters
///
/// - **`temperature`**: Controls "softness" of Gumbel box boundaries
///   - Lower (0.1-0.5): Sharp boundaries, more like hard boxes
///   - Higher (1.0-2.0): Soft boundaries, smoother gradients
///   - Can be scheduled: Start high, decrease during training
///
/// - **`margin`**: Minimum score difference between positive and negative triples
///   - Higher: Forces stronger separation (better discrimination)
///   - Lower: Allows closer scores (may be easier to optimize)
///   - Common: 0.5-2.0
///
/// ## Training Control
///
/// - **`early_stopping_patience`**: Stop if validation doesn't improve for N epochs
///   - Prevents overfitting automatically
///   - None: Train for all epochs (may overfit)
///   - Some(10): Stop if no improvement for 10 epochs
///
/// # Mathematical Relationships
///
/// The total loss combines multiple terms:
///
/// \[
/// L_{\text{total}} = L_{\text{ranking}} + \lambda_{\text{reg}} \cdot L_{\text{volume}} + \lambda_{\text{wd}} \cdot ||\theta||^2
/// \]
///
/// where:
/// - \(L_{\text{ranking}}\) is the margin-based ranking loss
/// - \(L_{\text{volume}}\) is volume regularization (penalizes large boxes)
/// - \(||\theta||^2\) is L2 regularization on parameters
/// - \(\lambda_{\text{reg}}\) is `regularization_weight`
/// - \(\lambda_{\text{wd}}\) is `weight_decay`
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate (default: 1e-3, paper range: 1e-3 to 5e-4)
    pub learning_rate: f32,
    /// Number of training epochs (default: 100)
    pub epochs: usize,
    /// Batch size (default: 512, paper range: 512-4096)
    pub batch_size: usize,
    /// Number of negative samples per positive (default: 1)
    pub negative_samples: usize,
    /// Negative sampling strategy (default: CorruptTail)
    pub negative_strategy: NegativeSamplingStrategy,
    /// Regularization weight (default: 0.01)
    pub regularization_weight: f32,
    /// Temperature for Gumbel boxes (default: 1.0)
    pub temperature: f32,
    /// Weight decay for AdamW (default: 1e-5, paper range: 1e-5 to 1e-3)
    pub weight_decay: f32,
    /// Margin for ranking loss (default: 1.0)
    pub margin: f32,
    /// Early stopping patience (default: Some(10))
    pub early_stopping_patience: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3, // Paper default: 1e-3 to 5e-4
            epochs: 100,
            batch_size: 512, // Paper range: 512-4096
            negative_samples: 1,
            negative_strategy: NegativeSamplingStrategy::CorruptTail,
            regularization_weight: 0.01,
            temperature: 1.0,
            weight_decay: 1e-5,                // Paper range: 1e-5 to 1e-3
            margin: 1.0,                       // Margin for ranking loss
            early_stopping_patience: Some(10), // Early stopping after 10 epochs without improvement
        }
    }
}

/// Evaluation results for link prediction.
#[derive(Debug, Clone)]
pub struct EvaluationResults {
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Hits@1
    pub hits_at_1: f32,
    /// Hits@3
    pub hits_at_3: f32,
    /// Hits@10
    pub hits_at_10: f32,
    /// Mean Rank
    pub mean_rank: f32,
}

/// Generate negative samples for a triple.
///
/// # Intuitive Explanation
///
/// Creates "false facts" by corrupting a true triple. For example, if the positive triple is
/// (Paris, located_in, France), negative samples might be:
/// - (Paris, located_in, Germany) - same head/relation, different tail
/// - (Tokyo, located_in, France) - same relation/tail, different head
///
/// **Why this works**: The model learns to assign high scores to true facts and low scores
/// to false facts. By seeing many negative examples, it learns to distinguish plausible from
/// implausible relationships.
///
/// **Example**: If we always corrupt the tail, the model learns "given Paris and 'located_in',
/// which countries make sense?" It learns that France is plausible but Germany is not (for
/// this specific fact).
///
/// # Arguments
///
/// * `triple` - The positive triple (true fact) to corrupt
/// * `entities` - Set of all entities (candidates for corruption)
/// * `strategy` - Which part of the triple to corrupt
/// * `n` - Number of negative samples to generate
///
/// # Returns
///
/// Vector of negative triples (corrupted versions of the positive triple)
#[cfg(feature = "rand")]
pub fn generate_negative_samples(
    triple: &Triple,
    entities: &HashSet<String>,
    strategy: &NegativeSamplingStrategy,
    n: usize,
) -> Vec<Triple> {
    let entities_vec: Vec<&String> = entities.iter().collect();
    let mut negatives = Vec::new();

    let mut rng = rand::rng();

    for _ in 0..n {
        let negative = match strategy {
            NegativeSamplingStrategy::Uniform => {
                // Randomly corrupt either head or tail
                if rng.random::<bool>() {
                    Triple {
                        head: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
                        relation: triple.relation.clone(),
                        tail: triple.tail.clone(),
                    }
                } else {
                    Triple {
                        head: triple.head.clone(),
                        relation: triple.relation.clone(),
                        tail: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
                    }
                }
            }
            NegativeSamplingStrategy::CorruptHead => Triple {
                head: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
                relation: triple.relation.clone(),
                tail: triple.tail.clone(),
            },
            NegativeSamplingStrategy::CorruptTail => Triple {
                head: triple.head.clone(),
                relation: triple.relation.clone(),
                tail: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
            },
            NegativeSamplingStrategy::CorruptBoth => Triple {
                head: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
                relation: triple.relation.clone(),
                tail: entities_vec[rng.random_range(0..entities_vec.len())].clone(),
            },
        };

        // Ensure negative is different from positive
        if negative != *triple {
            negatives.push(negative);
        }
    }

    negatives
}

/// Evaluate link prediction performance.
///
/// # Research Background
///
/// Link prediction evaluation follows the standard protocol established by **Bordes et al. (2013)**
/// for TransE and used consistently across knowledge graph embedding literature. The metrics
/// (MRR, Hits@K, Mean Rank) are standard benchmarks for knowledge graph completion.
///
/// **Reference**: Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
///
/// # Intuitive Explanation
///
/// Link prediction is the core task: given (head, relation, ?), predict which tail entity
/// completes the triple. This function evaluates how well the model does this.
///
/// **The process**:
/// 1. For each test triple (e.g., (Paris, located_in, ?))
/// 2. Score all possible tail entities using containment probability
/// 3. Rank them by score (highest = most likely)
/// 4. Check where the correct answer (France) appears in the ranking
///
/// **Metrics computed**:
/// - **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct answers
///   - If correct answer is rank 1 → 1/1 = 1.0 (perfect)
///   - If correct answer is rank 5 → 1/5 = 0.2
///   - Higher is better (range: 0 to 1)
///
/// - **Hits@K**: Fraction of queries where correct answer is in top K
///   - Hits@10 = 0.8 means 80% of queries have correct answer in top 10
///   - Higher is better (range: 0 to 1)
///
/// - **Mean Rank**: Average position of correct answers
///   - Lower is better (best = 1.0, worst = number of entities)
///
/// **Why this matters**: These metrics tell you if the model learned meaningful geometric
/// relationships. High MRR means boxes are arranged so containment probabilities match
/// knowledge graph structure.
///
/// # Arguments
///
/// * `test_triples` - Test set triples (held-out true facts)
/// * `entity_boxes` - Map from entity ID to box embedding
/// * `relation_boxes` - Map from relation ID to box embedding (optional, for relation-specific boxes)
///
/// # Returns
///
/// Evaluation results with MRR, Hits@K, Mean Rank
///
/// # Note
///
/// This function requires `B::Scalar = f32`. For other scalar types, use backend-specific evaluation functions.
pub fn evaluate_link_prediction<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    _relation_boxes: Option<&HashMap<String, B>>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    let mut ranks = Vec::new();

    for triple in test_triples {
        // Get head and relation boxes
        let head_box = entity_boxes
            .get(&triple.head)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.head)))?;

        // For simplicity, use head box directly (can be extended for relation-specific boxes)
        let query_box = head_box;

        // Score all entities
        let mut scores: Vec<(String, f32)> = entity_boxes
            .iter()
            .map(|(entity, box_)| {
                let score = query_box.containment_prob(box_, 1.0)?;
                Ok((entity.clone(), score))
            })
            .collect::<Result<_, crate::BoxError>>()?;

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find rank of correct tail
        let rank = scores
            .iter()
            .position(|(entity, _)| entity == &triple.tail)
            .map(|pos| pos + 1)
            .unwrap_or(usize::MAX);

        ranks.push(rank);
    }

    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let hits_at_1 = hits_at_k(ranks.iter().copied(), 1);
    let hits_at_3 = hits_at_k(ranks.iter().copied(), 3);
    let hits_at_10 = hits_at_k(ranks.iter().copied(), 10);
    let mean_rank = mean_rank(ranks.iter().copied());

    Ok(EvaluationResults {
        mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank,
    })
}

/// Training result with metrics and history.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final evaluation results
    pub final_results: EvaluationResults,
    /// Training loss history
    pub loss_history: Vec<f32>,
    /// Validation MRR history
    pub validation_mrr_history: Vec<f32>,
    /// Best epoch (based on validation MRR)
    pub best_epoch: usize,
    /// Total training time (if tracked)
    pub training_time_seconds: Option<f64>,
}

/// Hyperparameter search configuration.
#[derive(Debug, Clone)]
pub struct HyperparameterSearch {
    /// Learning rates to try
    pub learning_rates: Vec<f32>,
    /// Batch sizes to try
    pub batch_sizes: Vec<usize>,
    /// Embedding dimensions to try
    pub dimensions: Vec<usize>,
    /// Regularization weights to try
    pub regularization_weights: Vec<f32>,
    /// Number of trials per combination
    pub trials_per_config: usize,
}

impl Default for HyperparameterSearch {
    fn default() -> Self {
        Self {
            learning_rates: vec![1e-3, 5e-4, 1e-4],
            batch_sizes: vec![512, 1024, 2048],
            dimensions: vec![50, 100, 200],
            regularization_weights: vec![1e-5, 1e-4, 1e-3],
            trials_per_config: 1,
        }
    }
}

/// Log training results to file or stdout.
pub fn log_training_result(result: &TrainingResult, path: Option<&str>) -> Result<(), BoxError> {
    let output = format!(
        "Training Results\n\
         ===============\n\
         Final MRR: {:.4}\n\
         Final Hits@1: {:.4}\n\
         Final Hits@3: {:.4}\n\
         Final Hits@10: {:.4}\n\
         Final Mean Rank: {:.2}\n\
         Best Epoch: {}\n\
         Training Time: {:.2}s\n",
        result.final_results.mrr,
        result.final_results.hits_at_1,
        result.final_results.hits_at_3,
        result.final_results.hits_at_10,
        result.final_results.mean_rank,
        result.best_epoch,
        result.training_time_seconds.unwrap_or(0.0)
    );

    if let Some(p) = path {
        std::fs::write(p, output).map_err(|e| BoxError::Internal(e.to_string()))?;
    } else {
        println!("{}", output);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_negative_samples() {
        let triple = Triple {
            head: "e1".to_string(),
            relation: "r1".to_string(),
            tail: "e2".to_string(),
        };

        let entities: HashSet<String> = ["e1", "e2", "e3", "e4"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let negatives = generate_negative_samples(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptTail,
            5,
        );

        // May generate fewer than 5 if some negatives match the positive
        // With only 4 entities and CorruptTail, at most 3 unique negatives (e1, e3, e4)
        assert!(
            negatives.len() >= 1,
            "Expected at least 1 negative, got {}",
            negatives.len()
        );
        for neg in negatives {
            assert_eq!(neg.head, "e1");
            assert_eq!(neg.relation, "r1");
            assert_ne!(neg.tail, "e2"); // Should be different from positive
        }
    }

    #[test]
    fn test_generate_negative_samples_all_strategies() {
        let triple = Triple {
            head: "e1".to_string(),
            relation: "r1".to_string(),
            tail: "e2".to_string(),
        };

        let entities: HashSet<String> = ["e1", "e2", "e3", "e4", "e5"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Test all strategies
        for strategy in [
            NegativeSamplingStrategy::Uniform,
            NegativeSamplingStrategy::CorruptHead,
            NegativeSamplingStrategy::CorruptTail,
            NegativeSamplingStrategy::CorruptBoth,
        ] {
            let negatives = generate_negative_samples(&triple, &entities, &strategy, 10);
            assert!(
                !negatives.is_empty(),
                "Strategy {:?} should generate negatives",
                strategy
            );

            for neg in &negatives {
                assert_ne!(neg, &triple, "Negative should differ from positive");
            }
        }
    }

    #[test]
    fn test_log_training_result() {
        let result = TrainingResult {
            final_results: EvaluationResults {
                mrr: 0.5,
                hits_at_1: 0.3,
                hits_at_3: 0.4,
                hits_at_10: 0.6,
                mean_rank: 5.5,
            },
            loss_history: vec![1.0, 0.8, 0.6],
            validation_mrr_history: vec![0.3, 0.4, 0.5],
            best_epoch: 2,
            training_time_seconds: Some(10.5),
        };

        // Test stdout logging (should not panic)
        log_training_result(&result, None).unwrap();

        // Test file logging
        let temp_file = std::env::temp_dir().join("test_training_result.txt");
        log_training_result(&result, Some(temp_file.to_str().unwrap())).unwrap();

        // Verify file was created and contains expected content
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("Training Results"));
        assert!(content.contains("0.5000")); // MRR
        assert!(content.contains("2")); // Best epoch

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_hyperparameter_search_default() {
        let search = HyperparameterSearch::default();
        assert!(!search.learning_rates.is_empty());
        assert!(!search.batch_sizes.is_empty());
        assert!(!search.dimensions.is_empty());
        assert!(!search.regularization_weights.is_empty());
        assert!(search.trials_per_config > 0);
    }

    #[test]
    #[allow(unused_variables)] // empty_boxes documents the test structure
    fn test_evaluate_link_prediction_basic() {
        // This test requires a backend implementation
        // We'll test the logic with a mock, but full integration test should be in backend tests
        // For now, just verify the function signature and error handling

        // Test with empty triples
        let _empty_boxes: HashMap<String, ()> = HashMap::new();
        // Can't actually call evaluate_link_prediction without a Box implementation
        // This test documents the need for integration tests in backend modules
        // Full integration tests are in subsume-ndarray/src/trainer_integration_tests.rs
    }
}
