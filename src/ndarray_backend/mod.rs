pub mod distance;
pub mod evaluation;
pub mod ndarray_box;
pub mod ndarray_cone;
pub mod ndarray_gumbel;
pub mod ndarray_octagon;
pub mod optimizer;
pub mod scheduler;

pub use distance::{boundary_distance, depth_distance, query2box_distance, vector_to_box_distance};
pub use evaluation::{EvaluationConfig, EvaluationMetrics, OptimizerComparison};
pub use ndarray_box::NdarrayBox;
pub use ndarray_cone::NdarrayCone;
pub use ndarray_gumbel::NdarrayGumbelBox;
pub use ndarray_octagon::{NdarrayDiagBounds, NdarrayOctagon};
pub use optimizer::{Adam, AdamW, SGD};
