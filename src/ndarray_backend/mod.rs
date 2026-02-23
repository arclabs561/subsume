pub mod distance;
pub mod evaluation;
pub mod ndarray_box;
pub mod ndarray_gumbel;
pub mod optimizer;
pub mod scheduler;

pub use distance::{boundary_distance, depth_distance, vector_to_box_distance};
pub use evaluation::{EvaluationConfig, EvaluationMetrics, OptimizerComparison};
pub use ndarray_box::NdarrayBox;
pub use ndarray_gumbel::NdarrayGumbelBox;
pub use optimizer::{Adam, AdamW, SGD};
