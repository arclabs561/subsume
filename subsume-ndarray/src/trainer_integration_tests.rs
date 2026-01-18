//! Integration tests for trainer module with ndarray backend.

#![allow(clippy::useless_vec)] // vec! is clearer in tests
#![allow(clippy::needless_range_loop)] // Indexing is intentional in tests

#[cfg(test)]
mod tests {
    use crate::NdarrayBox;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::{HashMap, HashSet};
    use subsume_core::dataset::Triple;
    use subsume_core::trainer::{
        evaluate_link_prediction, generate_negative_samples, generate_negative_samples_from_pool_with_rng, log_training_result,
        EvaluationResults, HyperparameterSearch, NegativeSamplingStrategy, TrainingConfig,
        TrainingResult,
    };

    #[test]
    fn test_evaluate_link_prediction() {
        // Create a simple hierarchy: animal -> mammal -> dog
        let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();

        // Animal box (large, contains everything)
        let animal = NdarrayBox::new(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![10.0, 10.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("animal".to_string(), animal);

        // Mammal box (medium, contained in animal)
        let mammal = NdarrayBox::new(
            Array1::from_vec(vec![2.0, 2.0]),
            Array1::from_vec(vec![8.0, 8.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("mammal".to_string(), mammal);

        // Dog box (small, contained in mammal)
        let dog = NdarrayBox::new(
            Array1::from_vec(vec![4.0, 4.0]),
            Array1::from_vec(vec![6.0, 6.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("dog".to_string(), dog);

        // Bird box (disjoint from mammal)
        let bird = NdarrayBox::new(
            Array1::from_vec(vec![12.0, 12.0]),
            Array1::from_vec(vec![15.0, 15.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("bird".to_string(), bird);

        // Test triples: animal -> mammal should rank mammal highly
        let test_triples = vec![
            Triple {
                head: "animal".to_string(),
                relation: "is_a".to_string(),
                tail: "mammal".to_string(),
            },
            Triple {
                head: "mammal".to_string(),
                relation: "is_a".to_string(),
                tail: "dog".to_string(),
            },
        ];

        let results =
            evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None).unwrap();

        // Animal -> mammal should have high rank (mammal should be near top)
        // Since mammal is contained in animal, it should rank highly
        assert!(results.mrr > 0.0, "MRR should be positive");
        assert!(results.hits_at_10 > 0.0, "Hits@10 should be positive");

        // All metrics should be in valid ranges
        assert!(results.mrr >= 0.0 && results.mrr <= 1.0);
        assert!(results.hits_at_1 >= 0.0 && results.hits_at_1 <= 1.0);
        assert!(results.hits_at_3 >= 0.0 && results.hits_at_3 <= 1.0);
        assert!(results.hits_at_10 >= 0.0 && results.hits_at_10 <= 1.0);
        assert!(results.mean_rank >= 1.0);
    }

    #[test]
    fn test_evaluate_link_prediction_with_negative_samples() {
        // Test that negative sampling works with evaluate_link_prediction
        let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();

        let e1 = NdarrayBox::new(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![2.0, 2.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("e1".to_string(), e1);

        let e2 = NdarrayBox::new(
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![3.0, 3.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("e2".to_string(), e2);

        let e3 = NdarrayBox::new(
            Array1::from_vec(vec![10.0, 10.0]),
            Array1::from_vec(vec![12.0, 12.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("e3".to_string(), e3);

        let test_triples = vec![Triple {
            head: "e1".to_string(),
            relation: "r1".to_string(),
            tail: "e2".to_string(),
        }];

        let results =
            evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None).unwrap();

        // e1 -> e2 should rank e2 highly (they overlap)
        assert!(results.mrr > 0.0);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.epochs > 0);
        assert!(config.batch_size > 0);
        assert!(config.negative_samples > 0);
    }

    #[test]
    fn test_hyperparameter_search() {
        let search = HyperparameterSearch::default();
        assert!(!search.learning_rates.is_empty());
        assert!(!search.batch_sizes.is_empty());
        assert!(!search.dimensions.is_empty());
        assert!(!search.regularization_weights.is_empty());

        // Verify all learning rates are positive
        for lr in &search.learning_rates {
            assert!(*lr > 0.0);
        }

        // Verify all batch sizes are positive
        for bs in &search.batch_sizes {
            assert!(*bs > 0);
        }
    }

    #[test]
    fn test_log_training_result_integration() {
        let result = TrainingResult {
            final_results: EvaluationResults {
                mrr: 0.75,
                hits_at_1: 0.5,
                hits_at_3: 0.6,
                hits_at_10: 0.8,
                mean_rank: 3.5,
            },
            loss_history: vec![1.0, 0.8, 0.6, 0.4],
            validation_mrr_history: vec![0.3, 0.5, 0.7, 0.75],
            best_epoch: 3,
            training_time_seconds: Some(42.5),
        };

        // Test that logging doesn't panic
        log_training_result(&result, None).unwrap();
    }

    #[test]
    fn test_generate_negative_samples_integration() {
        let triple = Triple {
            head: "animal".to_string(),
            relation: "is_a".to_string(),
            tail: "mammal".to_string(),
        };

        let entities: HashSet<String> = ["animal", "mammal", "dog", "cat", "bird"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Test all strategies produce valid negatives
        for strategy in [
            NegativeSamplingStrategy::Uniform,
            NegativeSamplingStrategy::CorruptHead,
            NegativeSamplingStrategy::CorruptTail,
            NegativeSamplingStrategy::CorruptBoth,
        ] {
            let negatives = generate_negative_samples(&triple, &entities, &strategy, 10);

            // Should generate some negatives
            assert!(
                !negatives.is_empty(),
                "Strategy {:?} should generate negatives",
                strategy
            );

            // All negatives should differ from positive
            for neg in &negatives {
                assert_ne!(neg, &triple);
                // All entities should be in the entity set
                assert!(entities.contains(&neg.head));
                assert!(entities.contains(&neg.tail));
            }
        }
    }

    #[test]
    fn test_generate_negative_samples_from_pool_is_deterministic() {
        let triple = Triple {
            head: "animal".to_string(),
            relation: "is_a".to_string(),
            tail: "mammal".to_string(),
        };

        // Pool is intentionally small: this simulates “hard negatives from neighborhood”.
        let pool: Vec<String> = ["dog", "cat", "bird"].iter().map(|s| s.to_string()).collect();

        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);

        let n1 = generate_negative_samples_from_pool_with_rng(
            &triple,
            &pool,
            &NegativeSamplingStrategy::CorruptTail,
            10,
            &mut rng1,
        );
        let n2 = generate_negative_samples_from_pool_with_rng(
            &triple,
            &pool,
            &NegativeSamplingStrategy::CorruptTail,
            10,
            &mut rng2,
        );

        assert_eq!(n1, n2, "seeded RNG should yield reproducible negatives");
        for neg in n1 {
            assert_eq!(neg.head, "animal");
            assert_eq!(neg.relation, "is_a");
            assert!(pool.contains(&neg.tail));
            assert_ne!(neg.tail, "mammal");
        }
    }

    #[test]
    fn test_evaluate_link_prediction_empty_triples() {
        let entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
        let test_triples: Vec<Triple> = vec![];

        let results =
            evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None).unwrap();

        // Empty triples should give zero metrics
        assert_eq!(results.mrr, 0.0);
        assert_eq!(results.hits_at_1, 0.0);
        assert_eq!(results.hits_at_3, 0.0);
        assert_eq!(results.hits_at_10, 0.0);
        assert_eq!(results.mean_rank, 0.0);
    }

    #[test]
    fn test_evaluate_link_prediction_missing_entity() {
        let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
        let e1 = NdarrayBox::new(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 1.0]),
            1.0,
        )
        .unwrap();
        entity_boxes.insert("e1".to_string(), e1);

        // Try to evaluate with missing entity
        let test_triples = vec![Triple {
            head: "missing".to_string(),
            relation: "r1".to_string(),
            tail: "e1".to_string(),
        }];

        let result = evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None);
        assert!(result.is_err(), "Should error on missing entity");
    }
}
