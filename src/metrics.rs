//! Rank-based evaluation metrics for link prediction.
//!
//! All metrics are provided by [`lattix::kge`] and re-exported here.
//! Ranks are 1-indexed (rank 1 = best possible).

pub use lattix::kge::{
    adjusted_mean_rank, hits_at_k, mean_rank, mean_reciprocal_rank, realistic_rank,
};
