//! Candle (GPU) backend for box embeddings.
//!
//! Provides differentiable box and Gumbel box embeddings using Candle's
//! autograd. Training is done via [`CandleBoxTrainer`](crate::trainer::CandleBoxTrainer).
//!
//! # Missing geometries
//!
//! The Candle backend currently only supports **Box** and **GumbelBox**.
//! The following geometries are available only via the ndarray (CPU) backend:
//!
//! - **Cone** — Candle autograd for angular representations requires careful
//!   handling of periodic boundary conditions (`sin`/`cos` gradients).
//! - **Octagon** — Diagonal constraints are expensive to differentiate;
//!   the CPU implementation uses polygon clipping which doesn't translate
//!   directly to tensor ops.
//! - **Gaussian** — Complex number representations need custom Candle kernels.
//!
//! If you need GPU training for these geometries, consider contributing
//! implementations to the `candle_backend` module.

pub mod candle_box;
pub mod candle_gumbel;
pub mod distance;

pub use candle_box::CandleBox;
pub use candle_gumbel::CandleGumbelBox;
pub use distance::{boundary_distance, vector_to_box_distance};
