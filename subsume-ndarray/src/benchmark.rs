//! Automated benchmark runner with result saving and comparison.
//!
//! Provides infrastructure for running comprehensive benchmarks across
//! different configurations, optimizers, and hyperparameters.

use crate::evaluation::{EvaluationConfig, EvaluationMetrics};
use crate::optimizer::{Adam, AdamW, SGD};
use ndarray::Array1;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Base evaluation config
    pub eval_config: EvaluationConfig,
    /// Learning rates to test
    pub learning_rates: Vec<f32>,
    /// Weight decays to test (for AdamW)
    pub weight_decays: Vec<f32>,
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Number of runs per configuration (for statistical significance)
    pub num_runs: usize,
    /// Whether to generate plots
    pub generate_plots: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            eval_config: EvaluationConfig::default(),
            learning_rates: vec![1e-3, 5e-4, 1e-4],
            weight_decays: vec![1e-2, 1e-3, 1e-4],
            batch_sizes: vec![16, 32, 64],
            num_runs: 3,
            generate_plots: true,
        }
    }
}

/// Benchmark result for a single configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Configuration name
    pub config_name: String,
    /// Optimizer name
    pub optimizer: String,
    /// Hyperparameters
    pub hyperparams: HashMap<String, f32>,
    /// Metrics (averaged over runs)
    pub metrics: EvaluationMetrics,
    /// Standard deviation across runs
    pub std_dev: Option<EvaluationMetrics>,
}

/// Comprehensive benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Output directory
    pub output_dir: String,
}

impl BenchmarkSuite {
    /// Create new benchmark suite.
    pub fn new(output_dir: String) -> Self {
        Self {
            results: Vec::new(),
            output_dir,
        }
    }

    /// Run comprehensive benchmarks.
    pub fn run_benchmarks(
        &mut self,
        config: &BenchmarkConfig,
    ) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        let output_dir = self.output_dir.clone();
        let output_path = Path::new(&output_dir);
        fs::create_dir_all(output_path)?;

        println!("Running comprehensive benchmarks...");
        println!("Output directory: {}\n", output_dir);

        // Benchmark different optimizers
        self.benchmark_optimizers(config)?;

        // Benchmark learning rates
        self.benchmark_learning_rates(config)?;

        // Benchmark batch sizes
        self.benchmark_batch_sizes(config)?;

        // Save all results
        self.save_results(output_path)?;

        // Generate comparison plots
        if config.generate_plots {
            self.generate_comparison_plots(output_path)?;
        }

        println!(
            "\nBenchmarks complete! Results saved to {}",
            self.output_dir
        );
        Ok(())
    }

    /// Benchmark different optimizers.
    fn benchmark_optimizers(
        &mut self,
        config: &BenchmarkConfig,
    ) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        println!("Benchmarking optimizers...");

        // Test each optimizer separately to avoid trait object issues
        println!("  Testing Adam...");
        let mut all_metrics_adam = Vec::new();
        for run in 0..config.num_runs {
            let mut optimizer = Adam::new(config.eval_config.learning_rate);
            let metrics =
                run_single_benchmark_with_config(&mut optimizer, "Adam", &config.eval_config)?;
            all_metrics_adam.push(metrics);
            println!(
                "    Run {}: MRR = {:.4}",
                run + 1,
                all_metrics_adam.last().unwrap().final_mrr
            );
        }
        let avg_adam = average_metrics(&all_metrics_adam);
        let std_dev_adam = if config.num_runs > 1 {
            Some(compute_std_dev(&all_metrics_adam, &avg_adam))
        } else {
            None
        };
        self.results.push(BenchmarkResult {
            config_name: "Adam_default".to_string(),
            optimizer: "Adam".to_string(),
            hyperparams: {
                let mut h = HashMap::new();
                h.insert(
                    "learning_rate".to_string(),
                    config.eval_config.learning_rate,
                );
                h
            },
            metrics: avg_adam,
            std_dev: std_dev_adam,
        });

        println!("  Testing AdamW...");
        let mut all_metrics_adamw = Vec::new();
        for run in 0..config.num_runs {
            let mut optimizer = AdamW::new(
                config.eval_config.learning_rate,
                config.eval_config.weight_decay,
            );
            let metrics =
                run_single_benchmark_with_config(&mut optimizer, "AdamW", &config.eval_config)?;
            all_metrics_adamw.push(metrics);
            println!(
                "    Run {}: MRR = {:.4}",
                run + 1,
                all_metrics_adamw.last().unwrap().final_mrr
            );
        }
        let avg_adamw = average_metrics(&all_metrics_adamw);
        let std_dev_adamw = if config.num_runs > 1 {
            Some(compute_std_dev(&all_metrics_adamw, &avg_adamw))
        } else {
            None
        };
        self.results.push(BenchmarkResult {
            config_name: "AdamW_default".to_string(),
            optimizer: "AdamW".to_string(),
            hyperparams: {
                let mut h = HashMap::new();
                h.insert(
                    "learning_rate".to_string(),
                    config.eval_config.learning_rate,
                );
                h.insert("weight_decay".to_string(), config.eval_config.weight_decay);
                h
            },
            metrics: avg_adamw,
            std_dev: std_dev_adamw,
        });

        println!("  Testing SGD...");
        let mut all_metrics_sgd = Vec::new();
        for run in 0..config.num_runs {
            let mut optimizer = SGD::new(config.eval_config.learning_rate);
            let metrics =
                run_single_benchmark_with_config(&mut optimizer, "SGD", &config.eval_config)?;
            all_metrics_sgd.push(metrics);
            println!(
                "    Run {}: MRR = {:.4}",
                run + 1,
                all_metrics_sgd.last().unwrap().final_mrr
            );
        }
        let avg_sgd = average_metrics(&all_metrics_sgd);
        let std_dev_sgd = if config.num_runs > 1 {
            Some(compute_std_dev(&all_metrics_sgd, &avg_sgd))
        } else {
            None
        };
        self.results.push(BenchmarkResult {
            config_name: "SGD_default".to_string(),
            optimizer: "SGD".to_string(),
            hyperparams: {
                let mut h = HashMap::new();
                h.insert(
                    "learning_rate".to_string(),
                    config.eval_config.learning_rate,
                );
                h
            },
            metrics: avg_sgd,
            std_dev: std_dev_sgd,
        });

        Ok(())
    }

    /// Benchmark different learning rates.
    fn benchmark_learning_rates(
        &mut self,
        config: &BenchmarkConfig,
    ) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        println!("\nBenchmarking learning rates...");

        for lr in &config.learning_rates {
            println!("  Testing learning rate = {}...", lr);
            let mut optimizer = AdamW::new(*lr, config.eval_config.weight_decay);
            let metrics =
                run_single_benchmark_with_config(&mut optimizer, "AdamW", &config.eval_config)?;

            let result = BenchmarkResult {
                config_name: format!("AdamW_lr_{}", lr),
                optimizer: "AdamW".to_string(),
                hyperparams: {
                    let mut h = HashMap::new();
                    h.insert("learning_rate".to_string(), *lr);
                    h.insert("weight_decay".to_string(), config.eval_config.weight_decay);
                    h
                },
                metrics,
                std_dev: None,
            };

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark different batch sizes.
    fn benchmark_batch_sizes(
        &mut self,
        config: &BenchmarkConfig,
    ) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        println!("\nBenchmarking batch sizes...");

        for batch_size in &config.batch_sizes {
            println!("  Testing batch size = {}...", batch_size);
            let mut eval_config = config.eval_config.clone();
            eval_config.batch_size = *batch_size;

            let mut optimizer = AdamW::new(eval_config.learning_rate, eval_config.weight_decay);
            let metrics = run_single_benchmark_with_config(&mut optimizer, "AdamW", &eval_config)?;

            let result = BenchmarkResult {
                config_name: format!("AdamW_batch_{}", batch_size),
                optimizer: "AdamW".to_string(),
                hyperparams: {
                    let mut h = HashMap::new();
                    h.insert("learning_rate".to_string(), eval_config.learning_rate);
                    h.insert("weight_decay".to_string(), eval_config.weight_decay);
                    h.insert("batch_size".to_string(), *batch_size as f32);
                    h
                },
                metrics,
                std_dev: None,
            };

            self.results.push(result);
        }

        Ok(())
    }

    /// Save all benchmark results to JSON.
    fn save_results(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let results_path = output_path.join("benchmark_results.json");
        let mut json_results = serde_json::Map::new();

        for result in &self.results {
            let mut result_json = serde_json::Map::new();
            result_json.insert("optimizer".to_string(), serde_json::json!(result.optimizer));
            result_json.insert(
                "final_mrr".to_string(),
                serde_json::json!(result.metrics.final_mrr),
            );
            result_json.insert(
                "final_hits_at_1".to_string(),
                serde_json::json!(result.metrics.final_hits_at_1),
            );
            result_json.insert(
                "final_hits_at_10".to_string(),
                serde_json::json!(result.metrics.final_hits_at_10),
            );
            result_json.insert(
                "total_training_time".to_string(),
                serde_json::json!(result.metrics.total_training_time),
            );
            result_json.insert(
                "hyperparams".to_string(),
                serde_json::json!(result.hyperparams),
            );
            result_json.insert(
                "loss_history".to_string(),
                serde_json::json!(result.metrics.loss_history),
            );
            result_json.insert(
                "validation_mrr".to_string(),
                serde_json::json!(result.metrics.validation_mrr),
            );
            result_json.insert(
                "epoch_times".to_string(),
                serde_json::json!(result.metrics.epoch_times),
            );

            json_results.insert(
                result.config_name.clone(),
                serde_json::Value::Object(result_json),
            );
        }

        fs::write(results_path, serde_json::to_string_pretty(&json_results)?)?;
        Ok(())
    }

    /// Generate comparison plots.
    fn generate_comparison_plots(
        &self,
        output_path: &Path,
    ) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        #[cfg(feature = "plotting")]
        {
            use crate::evaluation::plotting;

            // Optimizer comparison
            let mut comparison = crate::evaluation::OptimizerComparison::new();
            for result in &self.results {
                if result.config_name.contains("_default") {
                    comparison.add_result(result.optimizer.clone(), result.metrics.clone());
                }
            }
            let comparison_plot = output_path.join("optimizer_comparison.png");
            plotting::plot_optimizer_comparison(&comparison, &comparison_plot)?;
            println!("  Saved optimizer comparison plot to {:?}", comparison_plot);

            // Learning rate comparison
            let lr_results: Vec<_> = self
                .results
                .iter()
                .filter(|r| r.config_name.starts_with("AdamW_lr_"))
                .collect();
            if !lr_results.is_empty() {
                // Create a simple comparison plot for learning rates
                println!("  Learning rate comparison plot (implement in plotting module)");
            }
        }
        #[cfg(not(feature = "plotting"))]
        {
            let _ = output_path; // Suppress unused variable warning when plotting is disabled
        }
        Ok(())
    }
}

/// Trait for optimizers in benchmarks.
pub trait OptimizerTrait {
    /// Update parameters given gradients.
    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>);
    /// Reset optimizer state (e.g., momentum buffers).
    fn reset(&mut self);
    /// Update the learning rate.
    fn set_learning_rate(&mut self, lr: f32);
}

impl OptimizerTrait for Adam {
    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        Adam::update(self, name, param, grad);
    }
    fn reset(&mut self) {
        Adam::reset(self);
    }
    fn set_learning_rate(&mut self, lr: f32) {
        Adam::set_learning_rate(self, lr);
    }
}

impl OptimizerTrait for AdamW {
    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        AdamW::update(self, name, param, grad);
    }
    fn reset(&mut self) {
        AdamW::reset(self);
    }
    fn set_learning_rate(&mut self, lr: f32) {
        AdamW::set_learning_rate(self, lr);
    }
}

impl OptimizerTrait for SGD {
    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        SGD::update(self, name, param, grad);
    }
    fn reset(&mut self) {
        SGD::reset(self);
    }
    fn set_learning_rate(&mut self, lr: f32) {
        SGD::set_learning_rate(self, lr);
    }
}

/// Run a single benchmark with custom config.
fn run_single_benchmark_with_config(
    optimizer: &mut dyn OptimizerTrait,
    _optimizer_name: &str,
    config: &EvaluationConfig,
) -> Result<EvaluationMetrics, std::boxed::Box<dyn std::error::Error>> {
    use crate::NdarrayBox;
    use std::collections::{HashMap, HashSet};
    use subsume_core::dataset::Triple;
    use subsume_core::trainer::evaluate_link_prediction;
    use subsume_core::Box;

    // Create synthetic dataset
    let train_triples = vec![
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

    let test_triples = vec![Triple {
        head: "animal".to_string(),
        relation: "is_a".to_string(),
        tail: "bird".to_string(),
    }];

    let mut entities = HashSet::new();
    for triple in train_triples.iter().chain(test_triples.iter()) {
        entities.insert(triple.head.clone());
        entities.insert(triple.tail.clone());
    }

    // Initialize embeddings
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        let offset = idx as f32 * 2.0;
        let box_ = NdarrayBox::new(
            Array1::from_vec(vec![offset, offset]),
            Array1::from_vec(vec![offset + 1.5, offset + 1.5]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    let mut metrics = EvaluationMetrics::new();
    let total_start = Instant::now();

    // Training loop
    for _epoch in 0..config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_triples.chunks(config.batch_size) {
            for triple in batch {
                // Compute loss first (before any mutable borrows)
                let (pos_score, tail_min, tail_max, head_key) =
                    if let (Some(head_box), Some(tail_box)) = (
                        entity_boxes.get(&triple.head),
                        entity_boxes.get(&triple.tail),
                    ) {
                        let score = head_box.containment_prob(tail_box, 1.0)?;
                        let t_min = tail_box.min().to_owned();
                        let t_max = tail_box.max().to_owned();
                        (score, t_min, t_max, triple.head.clone())
                    } else {
                        continue;
                    };

                let loss = 1.0 - pos_score;
                epoch_loss += loss;

                // Now get mutable reference for gradient update
                // Safe because we already verified the key exists above
                let head_box_mut = entity_boxes.get_mut(&head_key).ok_or_else(|| {
                    std::boxed::Box::<dyn std::error::Error>::from(format!(
                        "Entity {} disappeared from map (concurrent modification?)",
                        head_key
                    ))
                })?;
                let head_min_ref = head_box_mut.min();
                let mut head_min = head_min_ref.to_owned();
                let grad_min_vec: Vec<f32> = head_min
                    .iter()
                    .zip(tail_min.iter())
                    .map(|(h, t)| loss * (t - h))
                    .collect();
                let grad_min = Array1::from_vec(grad_min_vec);
                optimizer.update(
                    &format!("{}_min", triple.head),
                    &mut head_min,
                    grad_min.view(),
                );

                let head_max_ref = head_box_mut.max();
                let mut head_max = head_max_ref.to_owned();
                let grad_max_vec: Vec<f32> = head_max
                    .iter()
                    .zip(tail_max.iter())
                    .map(|(h, t)| loss * (t - h))
                    .collect();
                let grad_max = Array1::from_vec(grad_max_vec);
                optimizer.update(
                    &format!("{}_max", triple.head),
                    &mut head_max,
                    grad_max.view(),
                );

                *head_box_mut = NdarrayBox::new(head_min, head_max, 1.0)?;
            }
            batch_count += 1;
        }

        let avg_loss = epoch_loss / (batch_count as f32).max(1.0);
        let epoch_time = epoch_start.elapsed().as_secs_f64();

        // Evaluate
        let eval_results =
            evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None)?;
        let mrr = eval_results.mrr;

        metrics.record_epoch(avg_loss, mrr, epoch_time);
    }

    // Final evaluation
    let final_results = evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None)?;
    metrics.final_mrr = final_results.mrr;
    metrics.final_hits_at_1 = final_results.hits_at_1;
    metrics.final_hits_at_10 = final_results.hits_at_10;
    metrics.total_training_time = total_start.elapsed().as_secs_f64();

    Ok(metrics)
}

/// Average metrics across multiple runs.
fn average_metrics(metrics_list: &[EvaluationMetrics]) -> EvaluationMetrics {
    if metrics_list.is_empty() {
        return EvaluationMetrics::default();
    }

    let n = metrics_list.len() as f32;
    let mut avg = EvaluationMetrics::default();

    // Average loss history (assuming same length)
    if !metrics_list[0].loss_history.is_empty() {
        let len = metrics_list[0].loss_history.len();
        avg.loss_history = (0..len)
            .map(|i| {
                metrics_list
                    .iter()
                    .map(|m| m.loss_history.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / n
            })
            .collect();
    }

    // Average validation MRR
    if !metrics_list[0].validation_mrr.is_empty() {
        let len = metrics_list[0].validation_mrr.len();
        avg.validation_mrr = (0..len)
            .map(|i| {
                metrics_list
                    .iter()
                    .map(|m| m.validation_mrr.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / n
            })
            .collect();
    }

    // Average epoch times
    if !metrics_list[0].epoch_times.is_empty() {
        let len = metrics_list[0].epoch_times.len();
        avg.epoch_times = (0..len)
            .map(|i| {
                metrics_list
                    .iter()
                    .map(|m| m.epoch_times.get(i).copied().unwrap_or(0.0))
                    .sum::<f64>()
                    / n as f64
            })
            .collect();
    }

    // Average final metrics
    avg.final_mrr = metrics_list.iter().map(|m| m.final_mrr).sum::<f32>() / n;
    avg.final_hits_at_1 = metrics_list.iter().map(|m| m.final_hits_at_1).sum::<f32>() / n;
    avg.final_hits_at_10 = metrics_list.iter().map(|m| m.final_hits_at_10).sum::<f32>() / n;
    avg.total_training_time = metrics_list
        .iter()
        .map(|m| m.total_training_time)
        .sum::<f64>()
        / n as f64;

    avg
}

/// Compute standard deviation of metrics.
fn compute_std_dev(
    metrics_list: &[EvaluationMetrics],
    avg: &EvaluationMetrics,
) -> EvaluationMetrics {
    if metrics_list.len() <= 1 {
        return EvaluationMetrics::default();
    }

    let n = metrics_list.len() as f32;
    let mut std_dev = EvaluationMetrics::default();

    // Standard deviation of final metrics
    let mrr_variance = metrics_list
        .iter()
        .map(|m| (m.final_mrr - avg.final_mrr).powi(2))
        .sum::<f32>()
        / (n - 1.0);
    std_dev.final_mrr = mrr_variance.sqrt();

    let hits1_variance = metrics_list
        .iter()
        .map(|m| (m.final_hits_at_1 - avg.final_hits_at_1).powi(2))
        .sum::<f32>()
        / (n - 1.0);
    std_dev.final_hits_at_1 = hits1_variance.sqrt();

    let hits10_variance = metrics_list
        .iter()
        .map(|m| (m.final_hits_at_10 - avg.final_hits_at_10).powi(2))
        .sum::<f32>()
        / (n - 1.0);
    std_dev.final_hits_at_10 = hits10_variance.sqrt();

    std_dev
}
