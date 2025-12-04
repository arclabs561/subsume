//! # subsume-candle
//!
//! Candle implementation of `subsume-core` traits for box embeddings.
//!
//! This crate provides `CandleBox` and `CandleGumbelBox` types that implement
//! the `Box` and `GumbelBox` traits using `candle_core::Tensor`.

use subsume_core::BoxError;

impl From<candle_core::Error> for BoxError {
    fn from(err: candle_core::Error) -> Self {
        BoxError::Internal(err.to_string())
    }
}
//!
//! # Example
//!
//! ```rust,ignore
//! use subsume_candle::CandleBox;
//! use subsume_core::Box;
//! use candle_core::{Device, Tensor};
//!
//! let device = Device::Cpu;
//! let min = Tensor::new(&[0.0f32, 0.0, 0.0], &device)?;
//! let max = Tensor::new(&[1.0f32, 1.0, 1.0], &device)?;
//!
//! let box_a = CandleBox::new(min, max, 1.0)?;
//! let volume = box_a.volume(1.0)?;
//! ```

#![warn(missing_docs)]

mod candle_box;
mod candle_gumbel;

pub use candle_box::CandleBox;
pub use candle_gumbel::CandleGumbelBox;

