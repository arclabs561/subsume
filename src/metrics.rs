//! Rank-based evaluation metrics for link prediction and knowledge graph tasks.
//!
//! Re-exports from [`lattix::metrics`]. Ranks are 1-indexed.

pub use lattix::metrics::{adjusted_mean_rank, hits_at_k, mean_rank, mean_reciprocal_rank};
