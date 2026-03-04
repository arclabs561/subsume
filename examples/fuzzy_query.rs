//! Fuzzy logical query answering with t-norms and t-conorms.
//!
//! FuzzQE (Chen et al., AAAI 2022) replaces crisp set operations with fuzzy
//! logic operators for answering complex queries over knowledge graphs.
//! Membership scores in [0,1] are combined with t-norms (intersection),
//! t-conorms (union), and negation.
//!
//! This example demonstrates:
//! 1. The three standard t-norm families (Min, Product, Lukasiewicz)
//! 2. Fuzzy query answering on a small knowledge graph
//! 3. De Morgan duality: negating an intersection = union of negations
//! 4. How t-norm choice affects answer rankings
//!
//! Reference: Chen, Hu, Sun, Leskovec (AAAI 2022), "Fuzzy Logic Based Logical
//! Query Answering on Knowledge Graphs"
//!
//! Run: cargo run -p subsume --example fuzzy_query
//!
//! Related examples:
//! - `query2box`: compositional query answering with box intersection (geometric approach)
//! - `octagon_demo`: octagon embeddings with diagonal constraints (geometric + relational)
//! - `containment_hierarchy`: box containment and overlap (basic geometry)

use subsume::fuzzy::{fuzzy_negation, TConorm, TNorm};

/// Simulate fuzzy membership scores for entities in a predicate.
/// In practice these come from a trained embedding model; here they're hand-set.
struct FuzzyKG {
    /// Entity names.
    entities: Vec<&'static str>,
    /// is_mammal[i]: degree to which entity i is a mammal.
    is_mammal: Vec<f32>,
    /// is_aquatic[i]: degree to which entity i is aquatic.
    is_aquatic: Vec<f32>,
    /// is_endangered[i]: degree to which entity i is endangered.
    is_endangered: Vec<f32>,
}

fn build_kg() -> FuzzyKG {
    FuzzyKG {
        entities: vec![
            "dolphin", "whale", "shark", "salmon", "eagle", "tiger", "panda",
        ],
        is_mammal: vec![0.95, 0.98, 0.05, 0.02, 0.01, 0.97, 0.99],
        is_aquatic: vec![0.90, 0.92, 0.99, 0.95, 0.05, 0.10, 0.02],
        is_endangered: vec![0.60, 0.85, 0.70, 0.30, 0.75, 0.90, 0.95],
    }
}

fn main() {
    println!("=== Fuzzy Query Answering with T-norms ===\n");

    let kg = build_kg();
    let norms = [TNorm::Min, TNorm::Product, TNorm::Lukasiewicz];

    // --- Query 1: aquatic AND mammal (aquatic mammals) ---

    println!("Query 1: aquatic AND mammal  (t-norm = intersection)\n");
    println!(
        "  {:>10} | {:>7} {:>7} | {:>7} {:>7} {:>7}",
        "entity", "mammal", "aquatic", "Min", "Product", "Lukasz"
    );
    println!(
        "  {:-<10}-+-{:-<7}-{:-<7}-+-{:-<7}-{:-<7}-{:-<7}",
        "", "", "", "", "", ""
    );

    for (i, &name) in kg.entities.iter().enumerate() {
        let m = kg.is_mammal[i];
        let a = kg.is_aquatic[i];
        let scores: Vec<f32> = norms.iter().map(|t| t.apply(m, a)).collect();
        println!(
            "  {:>10} | {:>7.3} {:>7.3} | {:>7.3} {:>7.3} {:>7.3}",
            name, m, a, scores[0], scores[1], scores[2]
        );
    }
    println!();

    // --- Query 2: aquatic OR endangered (t-conorm = union) ---

    println!("Query 2: aquatic OR endangered  (t-conorm = union)\n");
    println!(
        "  {:>10} | {:>7} {:>7} | {:>7} {:>7} {:>7}",
        "entity", "aquatic", "endgrd", "Max", "Prob", "Lukasz"
    );
    println!(
        "  {:-<10}-+-{:-<7}-{:-<7}-+-{:-<7}-{:-<7}-{:-<7}",
        "", "", "", "", "", ""
    );

    for (i, &name) in kg.entities.iter().enumerate() {
        let a = kg.is_aquatic[i];
        let e = kg.is_endangered[i];
        let conorms = [TConorm::Max, TConorm::Probabilistic, TConorm::Lukasiewicz];
        let scores: Vec<f32> = conorms.iter().map(|s| s.apply(a, e)).collect();
        println!(
            "  {:>10} | {:>7.3} {:>7.3} | {:>7.3} {:>7.3} {:>7.3}",
            name, a, e, scores[0], scores[1], scores[2]
        );
    }
    println!();

    // --- Query 3: NOT mammal AND aquatic (non-mammal aquatic = fish/sharks) ---

    println!("Query 3: NOT mammal AND aquatic  (negation + t-norm)\n");
    println!(
        "  {:>10} | {:>7} {:>7} {:>7} | {:>7}",
        "entity", "mammal", "NOT_m", "aquatic", "result"
    );
    println!(
        "  {:-<10}-+-{:-<7}-{:-<7}-{:-<7}-+-{:-<7}",
        "", "", "", "", ""
    );

    let tnorm = TNorm::Product; // Use product t-norm
    for (i, &name) in kg.entities.iter().enumerate() {
        let m = kg.is_mammal[i];
        let not_m = fuzzy_negation(m);
        let a = kg.is_aquatic[i];
        let result = tnorm.apply(not_m, a);
        println!(
            "  {:>10} | {:>7.3} {:>7.3} {:>7.3} | {:>7.3}",
            name, m, not_m, a, result
        );
    }
    println!();

    // --- De Morgan duality verification ---

    println!("De Morgan: neg(T(a,b)) = S(neg(a), neg(b))\n");
    let a = 0.7;
    let b = 0.4;
    for t in &norms {
        let s = t.dual();
        let lhs = fuzzy_negation(t.apply(a, b));
        let rhs = s.apply(fuzzy_negation(a), fuzzy_negation(b));
        println!(
            "  {:?}: neg(T({a},{b})) = {lhs:.6},  S(neg({a}),neg({b})) = {rhs:.6},  match={}",
            t,
            (lhs - rhs).abs() < 1e-6
        );
    }
    println!();

    // --- Ranking comparison ---

    println!("Ranking: top-3 aquatic mammals by each t-norm\n");
    for t in &norms {
        let mut ranked: Vec<(&str, f32)> = kg
            .entities
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, t.apply(kg.is_mammal[i], kg.is_aquatic[i])))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top3: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(n, s)| format!("{n}({s:.3})"))
            .collect();
        println!("  {:?}: {}", t, top3.join(", "));
    }

    println!("\n--- Summary ---\n");
    println!("  Min t-norm: conservative, score = weakest link");
    println!("  Product t-norm: balanced, penalizes low scores multiplicatively");
    println!("  Lukasiewicz t-norm: strictest, requires both inputs high (additive threshold)");
    println!("  All three agree on top entity (dolphin/whale) but differ on cutoff sharpness");
}
