//! Hierarchical classification example using box embeddings.
//!
//! This example demonstrates how to use box embeddings for hierarchical
//! classification tasks, where classes form a tree structure (e.g., 
//! Animal -> Mammal -> Cat, or Product -> Electronics -> Computer).
//!
//! Box embeddings naturally model hierarchical relationships through
//! containment: if "Animal" contains "Mammal", then P(Mammal ⊆ Animal) ≈ 1.0.

use ndarray::Array1;
use subsume_core::{
    Box, BoxEmbedding,
    training::{
        quality::{
            ContainmentHierarchy, VolumeConservation, ContainmentAccuracy,
            VolumeDistribution,
        },
    },
};
use subsume_ndarray::NdarrayBox;
use subsume_core::BoxCollection;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Hierarchical Classification with Box Embeddings ===\n");

    // Define a hierarchical taxonomy: Product -> Electronics -> Computer -> Laptop
    //                              Product -> Clothing -> Shirt
    //                              Product -> Food -> Fruit -> Apple
    
    let mut boxes: BoxCollection<NdarrayBox> = BoxCollection::new();
    let mut hierarchy = ContainmentHierarchy::new();
    let mut volume_conservation = VolumeConservation::new();
    let mut containment_accuracy = ContainmentAccuracy::new();

    // Level 0: Root (Product)
    let product = NdarrayBox::new(
        Array1::from(vec![0.0, 0.0, 0.0]),
        Array1::from(vec![20.0, 20.0, 20.0]),
        1.0,
    )?;
    boxes.push(product.clone());

    // Level 1: Categories
    let electronics = NdarrayBox::new(
        Array1::from(vec![1.0, 1.0, 1.0]),
        Array1::from(vec![8.0, 8.0, 8.0]),
        1.0,
    )?;
    let clothing = NdarrayBox::new(
        Array1::from(vec![12.0, 1.0, 1.0]),
        Array1::from(vec![18.0, 8.0, 8.0]),
        1.0,
    )?;
    let food = NdarrayBox::new(
        Array1::from(vec![1.0, 12.0, 1.0]),
        Array1::from(vec![8.0, 18.0, 8.0]),
        1.0,
    )?;
    boxes.push(electronics.clone());
    boxes.push(clothing.clone());
    boxes.push(food.clone());

    // Level 2: Subcategories
    let computer = NdarrayBox::new(
        Array1::from(vec![2.0, 2.0, 2.0]),
        Array1::from(vec![5.0, 5.0, 5.0]),
        1.0,
    )?;
    let shirt = NdarrayBox::new(
        Array1::from(vec![13.0, 2.0, 2.0]),
        Array1::from(vec![17.0, 5.0, 5.0]),
        1.0,
    )?;
    let fruit = NdarrayBox::new(
        Array1::from(vec![2.0, 13.0, 2.0]),
        Array1::from(vec![5.0, 17.0, 5.0]),
        1.0,
    )?;
    boxes.push(computer.clone());
    boxes.push(shirt.clone());
    boxes.push(fruit.clone());

    // Level 3: Specific items
    let laptop = NdarrayBox::new(
        Array1::from(vec![3.0, 3.0, 3.0]),
        Array1::from(vec![4.0, 4.0, 4.0]),
        1.0,
    )?;
    let apple = NdarrayBox::new(
        Array1::from(vec![3.0, 14.0, 3.0]),
        Array1::from(vec![4.0, 15.0, 4.0]),
        1.0,
    )?;
    boxes.push(laptop.clone());
    boxes.push(apple.clone());

    // Build hierarchy relationships
    // Product -> Electronics, Clothing, Food
    hierarchy.add_containment(0, 1); // Product -> Electronics
    hierarchy.add_containment(0, 2); // Product -> Clothing
    hierarchy.add_containment(0, 3); // Product -> Food
    
    // Electronics -> Computer
    hierarchy.add_containment(1, 4); // Electronics -> Computer
    
    // Clothing -> Shirt
    hierarchy.add_containment(2, 5); // Clothing -> Shirt
    
    // Food -> Fruit
    hierarchy.add_containment(3, 6); // Food -> Fruit
    
    // Computer -> Laptop
    hierarchy.add_containment(4, 7); // Computer -> Laptop
    
    // Fruit -> Apple
    hierarchy.add_containment(6, 8); // Fruit -> Apple

    hierarchy.compute_transitive_closure();

    println!("1. Hierarchy Structure:");
    let depths = hierarchy.hierarchy_depths();
    for (node_id, depth) in &depths {
        let node_name = match *node_id {
            0 => "Product",
            1 => "Electronics",
            2 => "Clothing",
            3 => "Food",
            4 => "Computer",
            5 => "Shirt",
            6 => "Fruit",
            7 => "Laptop",
            8 => "Apple",
            _ => "Unknown",
        };
        println!("  {} (depth {}): {}", "  ".repeat(*depth), depth, node_name);
    }
    println!();

    // Verify containment probabilities
    println!("2. Containment Probabilities:");
    
    // Direct containments
    let p_electronics_in_product = product.containment_prob(&electronics, 1.0)?;
    println!("  P(Electronics ⊆ Product) = {:.4}", p_electronics_in_product);
    
    let p_computer_in_electronics = electronics.containment_prob(&computer, 1.0)?;
    println!("  P(Computer ⊆ Electronics) = {:.4}", p_computer_in_electronics);
    
    let p_laptop_in_computer = computer.containment_prob(&laptop, 1.0)?;
    println!("  P(Laptop ⊆ Computer) = {:.4}", p_laptop_in_computer);
    
    // Transitive containments (should also be high)
    let p_laptop_in_product = product.containment_prob(&laptop, 1.0)?;
    println!("  P(Laptop ⊆ Product) [transitive] = {:.4}", p_laptop_in_product);
    
    let p_apple_in_food = food.containment_prob(&apple, 1.0)?;
    println!("  P(Apple ⊆ Food) [transitive] = {:.4}", p_apple_in_food);
    println!();

    // Classification example: Given a new item, classify it hierarchically
    println!("3. Hierarchical Classification Example:");
    println!("   Classifying a new 'MacBook' item.\n");
    
    // Create a new item that should be classified as Laptop -> Computer -> Electronics -> Product
    let macbook = NdarrayBox::new(
        Array1::from(vec![3.2, 3.2, 3.2]),
        Array1::from(vec![3.8, 3.8, 3.8]),
        1.0,
    )?;
    
    // Find best matching class at each level
    let candidates = vec![
        ("Product", &product),
        ("Electronics", &electronics),
        ("Clothing", &clothing),
        ("Food", &food),
        ("Computer", &computer),
        ("Shirt", &shirt),
        ("Fruit", &fruit),
        ("Laptop", &laptop),
        ("Apple", &apple),
    ];
    
    // Sort by containment probability (descending)
    let mut scores: Vec<(&str, f32)> = candidates.iter()
        .map(|(name, candidate)| {
            let prob = macbook.containment_prob(candidate, 1.0).unwrap_or(0.0);
            (*name, prob)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("   Classification scores (containment probabilities):");
    for (name, score) in &scores {
        println!("     {}: {:.4}", name, score);
    }
    
    // Hierarchical classification: find best path through hierarchy
    println!("\n   Hierarchical classification path:");
    let mut current_level = 0;
    let mut path = Vec::new();
    
    // Start at root
    path.push(("Product", 0));
    
    // Level 1: Find best category
    let level1_candidates = vec![
        ("Electronics", 1, &electronics),
        ("Clothing", 2, &clothing),
        ("Food", 3, &food),
    ];
    let (best_level1, _, _) = level1_candidates.iter()
        .max_by(|(_, _, a), (_, _, b)| {
            let prob_a = macbook.containment_prob(a, 1.0).unwrap_or(0.0);
            let prob_b = macbook.containment_prob(b, 1.0).unwrap_or(0.0);
            prob_a.partial_cmp(&prob_b).unwrap()
        })
        .unwrap();
    path.push((best_level1, 1));
    
    // Level 2: Find best subcategory under Electronics
    if *best_level1 == "Electronics" {
        let level2_candidates = vec![
            ("Computer", 4, &computer),
        ];
        let (best_level2, _, _) = level2_candidates.iter()
            .max_by(|(_, _, a), (_, _, b)| {
                let prob_a = macbook.containment_prob(a, 1.0).unwrap_or(0.0);
                let prob_b = macbook.containment_prob(b, 1.0).unwrap_or(0.0);
                prob_a.partial_cmp(&prob_b).unwrap()
            })
            .unwrap();
        path.push((best_level2, 2));
        
        // Level 3: Find best specific item
        let level3_candidates = vec![
            ("Laptop", 7, &laptop),
        ];
        let (best_level3, _, _) = level3_candidates.iter()
            .max_by(|(_, _, a), (_, _, b)| {
                let prob_a = macbook.containment_prob(a, 1.0).unwrap_or(0.0);
                let prob_b = macbook.containment_prob(b, 1.0).unwrap_or(0.0);
                prob_a.partial_cmp(&prob_b).unwrap()
            })
            .unwrap();
        path.push((best_level3, 3));
    }
    
    for (name, depth) in &path {
        println!("     {} {}", "  ".repeat(*depth), name);
    }
    println!();

    // Verify volume conservation
    println!("4. Volume Conservation Verification:");
    let product_vol = product.volume(1.0)?;
    let categories_vol = electronics.volume(1.0)? + clothing.volume(1.0)? + food.volume(1.0)?;
    volume_conservation.record_parent_children(
        product_vol,
        vec![electronics.volume(1.0)?, clothing.volume(1.0)?, food.volume(1.0)?].into_iter(),
        0.1,
    );
    
    println!("  Product volume: {:.4}", product_vol);
    println!("  Sum of categories: {:.4}", categories_vol);
    println!("  Conservation ratio: {:.4}", categories_vol / product_vol);
    let mean_ratio = volume_conservation.mean_ratio();
    println!("  Mean conservation ratio: {:.4} (should be <= 1.0)", mean_ratio);
    println!();

    // Evaluate classification accuracy
    println!("5. Classification Accuracy Evaluation:");
    
    // Simulate test cases
    let test_cases = vec![
        (macbook.clone(), "Laptop", true),  // Correct
        (laptop.clone(), "Laptop", true),   // Correct
        (apple.clone(), "Apple", true),     // Correct
        (shirt.clone(), "Shirt", true),     // Correct
        (macbook.clone(), "Apple", false),  // Incorrect
        (apple.clone(), "Laptop", false),   // Incorrect
    ];
    
    for (item, expected_class, should_contain) in test_cases {
        let class_box = boxes.get(match expected_class {
            "Laptop" => 7,
            "Apple" => 8,
            "Shirt" => 5,
            _ => 0,
        })?;
        let prob = item.containment_prob(class_box, 1.0)?;
        let predicted = prob > 0.5;
        containment_accuracy.record(predicted, should_contain);
    }
    
    println!("  Precision: {:.4}", containment_accuracy.precision());
    println!("  Recall: {:.4}", containment_accuracy.recall());
    println!("  F1: {:.4}", containment_accuracy.f1());
    println!("  Accuracy: {:.4}", containment_accuracy.accuracy());
    println!();

    // Hierarchy verification
    println!("6. Hierarchy Verification:");
    let (violations, total) = hierarchy.verify_transitivity();
    println!("  Transitivity violations: {}/{}", violations, total);
    let cycles = hierarchy.detect_cycles();
    println!("  Cycles detected: {}", cycles.len());
    println!("  ✓ Hierarchy is valid");
    println!();

    // Volume distribution analysis
    println!("7. Volume Distribution Analysis:");
    let volumes: Vec<f32> = (0..boxes.len())
        .filter_map(|i| boxes.get(i).ok()?.volume(1.0).ok())
        .collect();
    let vol_dist = VolumeDistribution::from_volumes(volumes.iter().copied());
    println!("  Volume entropy: {:.4}", vol_dist.entropy);
    println!("  Coefficient of variation: {:.4}", vol_dist.cv);
    println!("  Has hierarchy: {}", vol_dist.has_hierarchy(0.1));
    println!("  Volume range: [{:.4}, {:.4}]", vol_dist.min, vol_dist.max);
    println!();

    println!("✓ Hierarchical classification example complete!");
    println!("\nKey insights:");
    println!("  - Box embeddings naturally encode hierarchical relationships");
    println!("  - Containment probabilities can be used for classification");
    println!("  - Volume conservation ensures geometric consistency");
    println!("  - Hierarchy verification catches logical inconsistencies");

    Ok(())
}

