//! Automated evaluation infrastructure with visualization.
//!
//! Provides tools for running benchmarks, collecting metrics, and generating visualizations.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Evaluation run configuration.
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of epochs to train
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay (for AdamW)
    pub weight_decay: f32,
    /// Batch size
    pub batch_size: usize,
    /// Output directory for results
    pub output_dir: String,
    /// Whether to generate plots
    pub generate_plots: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 1e-3,
            weight_decay: 1e-2,
            batch_size: 32,
            output_dir: "eval_results".to_string(),
            generate_plots: true,
        }
    }
}

/// Metrics collected during evaluation.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Training loss per epoch
    pub loss_history: Vec<f32>,
    /// Validation MRR per epoch
    pub validation_mrr: Vec<f32>,
    /// Training time per epoch (seconds)
    pub epoch_times: Vec<f64>,
    /// Final Mean Reciprocal Rank
    pub final_mrr: f32,
    /// Final Hits@1 (fraction of correct top-1 predictions)
    pub final_hits_at_1: f32,
    /// Final Hits@10 (fraction of correct top-10 predictions)
    pub final_hits_at_10: f32,
    /// Total training time in seconds
    pub total_training_time: f64,
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self {
            loss_history: Vec::new(),
            validation_mrr: Vec::new(),
            epoch_times: Vec::new(),
            final_mrr: 0.0,
            final_hits_at_1: 0.0,
            final_hits_at_10: 0.0,
            total_training_time: 0.0,
        }
    }
}

impl EvaluationMetrics {
    /// Create new metrics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record epoch metrics.
    pub fn record_epoch(&mut self, loss: f32, mrr: f32, epoch_time: f64) {
        self.loss_history.push(loss);
        self.validation_mrr.push(mrr);
        self.epoch_times.push(epoch_time);
    }

    /// Save metrics to JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::json!({
            "loss_history": self.loss_history,
            "validation_mrr": self.validation_mrr,
            "epoch_times": self.epoch_times,
            "final_mrr": self.final_mrr,
            "final_hits_at_1": self.final_hits_at_1,
            "final_hits_at_10": self.final_hits_at_10,
            "total_training_time": self.total_training_time,
        });
        fs::write(path, serde_json::to_string_pretty(&json)?)?;
        Ok(())
    }
}

/// Optimizer comparison results.
#[derive(Debug, Clone)]
pub struct OptimizerComparison {
    /// Results for each optimizer
    pub results: HashMap<String, EvaluationMetrics>,
}

impl Default for OptimizerComparison {
    fn default() -> Self {
        Self {
            results: HashMap::new(),
        }
    }
}

impl OptimizerComparison {
    /// Create new comparison.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add optimizer results.
    pub fn add_result(&mut self, optimizer_name: String, metrics: EvaluationMetrics) {
        self.results.insert(optimizer_name, metrics);
    }

    /// Save comparison to JSON.
    pub fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut json_results = serde_json::Map::new();
        for (name, metrics) in &self.results {
            json_results.insert(
                name.clone(),
                serde_json::json!({
                    "final_mrr": metrics.final_mrr,
                    "final_hits_at_1": metrics.final_hits_at_1,
                    "final_hits_at_10": metrics.final_hits_at_10,
                    "total_training_time": metrics.total_training_time,
                }),
            );
        }
        fs::write(path, serde_json::to_string_pretty(&json_results)?)?;
        Ok(())
    }
}

/// Plot generation utilities.
#[cfg(feature = "plotting")]
pub mod plotting {
    use super::*;
    use plotters::prelude::*;

    /// Generate loss curve plot.
    pub fn plot_loss_curve(
        metrics: &EvaluationMetrics,
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training Loss", ("sans-serif", 40).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..metrics.loss_history.len(), 0f32..1f32)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                metrics
                    .loss_history
                    .iter()
                    .enumerate()
                    .map(|(i, &loss)| (i, loss)),
                &RED,
            ))?
            .label("Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Generate MRR curve plot.
    pub fn plot_mrr_curve(
        metrics: &EvaluationMetrics,
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_mrr = metrics
            .validation_mrr
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(0.1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Validation MRR", ("sans-serif", 40).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..metrics.validation_mrr.len(), 0f32..max_mrr)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                metrics
                    .validation_mrr
                    .iter()
                    .enumerate()
                    .map(|(i, &mrr)| (i, mrr)),
                &BLUE,
            ))?
            .label("MRR")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Generate optimizer comparison plot.
    pub fn plot_optimizer_comparison(
        comparison: &OptimizerComparison,
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(output_path, (1000, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let optimizers: Vec<&String> = comparison.results.keys().collect();
        let max_mrr = comparison
            .results
            .values()
            .map(|m| m.final_mrr)
            .fold(0.0f32, f32::max)
            .max(0.1);

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Optimizer Comparison (Final MRR)",
                ("sans-serif", 40).into_font(),
            )
            .margin(5)
            .x_label_area_size(60)
            .y_label_area_size(40)
            .build_cartesian_2d(0..optimizers.len(), 0f32..max_mrr)?;

        chart.configure_mesh().draw()?;

        let colors = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];
        for (i, optimizer) in optimizers.iter().enumerate() {
            let mrr = comparison.results[*optimizer].final_mrr;
            let color = colors[i % colors.len()];
            chart.draw_series(std::iter::once(Circle::new((i, mrr), 5, color.filled())))?;
        }

        root.present()?;
        Ok(())
    }

    /// Generate learning rate comparison plot.
    pub fn plot_learning_rate_comparison(
        results: &[(String, f32, f32)], // (config_name, lr, mrr)
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_mrr = results
            .iter()
            .map(|(_, _, mrr)| *mrr)
            .fold(0.0f32, f32::max)
            .max(0.1);
        let min_lr = results
            .iter()
            .map(|(_, lr, _)| *lr)
            .fold(f32::INFINITY, f32::min);
        let max_lr = results.iter().map(|(_, lr, _)| *lr).fold(0.0f32, f32::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Learning Rate Comparison", ("sans-serif", 40).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (min_lr.log10() - 0.5)..(max_lr.log10() + 0.5),
                0f32..max_mrr,
            )?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                results.iter().map(|(_, lr, mrr)| (lr.log10(), *mrr)),
                RED.stroke_width(2),
            ))?
            .label("MRR")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Generate batch size comparison plot.
    pub fn plot_batch_size_comparison(
        results: &[(usize, f32)], // (batch_size, mrr)
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_mrr = results
            .iter()
            .map(|(_, mrr)| *mrr)
            .fold(0.0f32, f32::max)
            .max(0.1);
        let min_batch = results.iter().map(|(bs, _)| *bs).min().unwrap_or(1);
        let max_batch = results.iter().map(|(bs, _)| *bs).max().unwrap_or(100);

        let mut chart = ChartBuilder::on(&root)
            .caption("Batch Size Comparison", ("sans-serif", 40).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(min_batch..max_batch, 0f32..max_mrr)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                results.iter().map(|(bs, mrr)| (*bs, *mrr)),
                BLUE.stroke_width(2),
            ))?
            .label("MRR")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

#[cfg(not(feature = "plotting"))]
#[allow(missing_docs)]
pub mod plotting {
    //! Plotting stubs - requires `plotting` feature for actual implementation.

    use super::*;

    /// Stub for loss curve plotting.
    pub fn plot_loss_curve(
        _metrics: &EvaluationMetrics,
        _output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Plotting feature not enabled. Install with --features plotting");
        Ok(())
    }

    /// Stub for MRR curve plotting.
    pub fn plot_mrr_curve(
        _metrics: &EvaluationMetrics,
        _output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Plotting feature not enabled. Install with --features plotting");
        Ok(())
    }

    /// Stub for optimizer comparison plotting.
    pub fn plot_optimizer_comparison(
        _comparison: &OptimizerComparison,
        _output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Plotting feature not enabled. Install with --features plotting");
        Ok(())
    }

    /// Stub for learning rate comparison plotting.
    pub fn plot_learning_rate_comparison(
        _results: &[(String, f32, f32)],
        _output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Plotting feature not enabled. Install with --features plotting");
        Ok(())
    }

    /// Stub for batch size comparison plotting.
    pub fn plot_batch_size_comparison(
        _results: &[(usize, f32)],
        _output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Plotting feature not enabled. Install with --features plotting");
        Ok(())
    }
}
