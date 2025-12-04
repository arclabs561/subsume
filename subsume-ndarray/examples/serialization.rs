//! Example: Serializing and deserializing box embeddings.
//!
//! This example demonstrates how to save and load box embeddings using serde,
//! which is useful for model persistence, checkpointing, and sharing embeddings.

use ndarray::array;
use serde_json;
use subsume_core::{Box as BoxTrait, GumbelBox};
use subsume_ndarray::{NdarrayBox, NdarrayGumbelBox};

fn main() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
    println!("Serialization Example");
    println!("====================");
    
    // Create a standard box
    let original_box = NdarrayBox::new(
        array![0.0, 1.0, 2.0],
        array![1.0, 2.0, 3.0],
        1.5,
    )?;
    
    println!("\n1. Serializing standard box to JSON:");
    let json = serde_json::to_string_pretty(&original_box)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;
    println!("{}", json);
    
    // Deserialize it back
    let deserialized_box: NdarrayBox = serde_json::from_str(&json)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;
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
    let gumbel_box = NdarrayGumbelBox::new(
        array![0.0, 0.0],
        array![1.0, 1.0],
        0.5,
    )?;
    
    println!("\n4. Serializing Gumbel box:");
    let gumbel_json = serde_json::to_string_pretty(&gumbel_box)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;
    println!("{}", gumbel_json);
    
    let deserialized_gumbel: NdarrayGumbelBox = serde_json::from_str(&gumbel_json)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;
    assert_eq!(gumbel_box.temperature(), deserialized_gumbel.temperature());
    println!("   ✓ Gumbel box round-trip successful!");
    
    // Example: Save to file (commented out, but shows the pattern)
    // use std::fs;
    // fs::write("box_embedding.json", json)?;
    // let loaded: NdarrayBox = serde_json::from_str(&fs::read_to_string("box_embedding.json")?)?;
    
    println!("\n5. Use cases:");
    println!("   - Model checkpointing: save embeddings during training");
    println!("   - Model sharing: serialize trained embeddings for deployment");
    println!("   - Caching: store precomputed embeddings to disk");
    println!("   - API responses: send embeddings over HTTP as JSON");
    
    Ok(())
}

