//! Example: Serializing and deserializing Candle box embeddings.
//!
//! This example demonstrates how to save and load box embeddings using serde,
//! which is useful for model persistence, checkpointing, and sharing embeddings.

use candle_core::Device;
use serde_json;
use subsume_candle::{CandleBox, CandleGumbelBox};
use subsume_core::{Box as BoxTrait, GumbelBox};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Candle Serialization Example");
    println!("============================\n");

    let device = Device::Cpu;

    // Create a standard box
    let original_box = CandleBox::new(
        candle_core::Tensor::new(&[0.0f32, 1.0, 2.0], &device)?,
        candle_core::Tensor::new(&[1.0f32, 2.0, 3.0], &device)?,
        1.5,
    )?;

    println!("1. Serializing standard box to JSON:");
    let json = serde_json::to_string_pretty(&original_box)?;
    println!("{}", json);

    // Deserialize it back
    let deserialized_box: CandleBox = serde_json::from_str(&json)?;
    println!("\n2. Deserialized box:");
    println!("   Dimensions: {}", deserialized_box.dim());
    println!("   Volume: {:.3}", deserialized_box.volume(1.0)?);

    // Verify they're equivalent
    assert_eq!(original_box.dim(), deserialized_box.dim());
    let vol_orig = original_box.volume(1.0)?;
    let vol_deser = deserialized_box.volume(1.0)?;
    assert!((vol_orig - vol_deser).abs() < 1e-5, "Volumes should match");
    println!("\n3. ✓ Round-trip serialization successful!");

    // Serialize a Gumbel box
    let gumbel_box = CandleGumbelBox::new(
        candle_core::Tensor::new(&[0.0f32, 0.0], &device)?,
        candle_core::Tensor::new(&[1.0f32, 1.0], &device)?,
        0.5,
    )?;

    println!("\n4. Serializing Gumbel box:");
    let gumbel_json = serde_json::to_string_pretty(&gumbel_box)?;
    println!("{}", gumbel_json);

    let deserialized_gumbel: CandleGumbelBox = serde_json::from_str(&gumbel_json)?;
    assert!((gumbel_box.temperature() - deserialized_gumbel.temperature()).abs() < 1e-5);
    println!("   ✓ Gumbel box round-trip successful!");

    println!("\n5. Use cases:");
    println!("   - Model checkpointing: save embeddings during training");
    println!("   - Model sharing: serialize trained embeddings for deployment");
    println!("   - Caching: store precomputed embeddings to disk");
    println!("   - API responses: send embeddings over HTTP as JSON");

    Ok(())
}
