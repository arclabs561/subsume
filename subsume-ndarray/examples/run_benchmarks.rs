//! Automated benchmark runner example.
//!
//! This example demonstrates running comprehensive benchmarks across
//! different optimizers, learning rates, and batch sizes.

use subsume_ndarray::benchmark::{BenchmarkConfig, BenchmarkSuite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automated Benchmark Runner");
    println!("==========================\n");

    // Configure benchmarks
    let mut config = BenchmarkConfig::default();
    config.eval_config.epochs = 20; // Shorter for demo
    config.num_runs = 2; // Fewer runs for demo
    config.generate_plots = cfg!(feature = "plotting");

    // Create benchmark suite
    let mut suite = BenchmarkSuite::new("benchmark_results".to_string());

    // Run all benchmarks
    suite.run_benchmarks(&config)?;

    // Print summary
    println!("\n=== Benchmark Summary ===");
    for result in &suite.results {
        println!(
            "{}: MRR = {:.4} ± {:.4}, Time = {:.2}s",
            result.config_name,
            result.metrics.final_mrr,
            result.std_dev.as_ref().map(|s| s.final_mrr).unwrap_or(0.0),
            result.metrics.total_training_time
        );
    }

    Ok(())
}
