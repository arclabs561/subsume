use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

use ndarray::Array1;
use std::collections::HashMap;
use subsume_core::dataset::Triple;
use subsume_core::trainer::FilteredTripleIndex;
use subsume_ndarray::NdarrayBox;

fn make_box_1d(min: f32, max: f32) -> NdarrayBox {
    NdarrayBox::new(Array1::from_vec(vec![min]), Array1::from_vec(vec![max]), 1.0)
        .expect("valid box")
}

#[test]
fn filtered_link_prediction_does_not_allocate_per_candidate() {
    // Goal: a regression guardrail for “resource consumption”.
    //
    // The filtered ranking path iterates over all entity boxes and scores each candidate.
    // That *must* be allocation-flat (no per-candidate heap work), otherwise performance
    // collapses at large entity counts.

    // Small-but-nontrivial entity universe.
    let n_entities = 2_000usize;
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::with_capacity(n_entities);
    for i in 0..n_entities {
        // A simple 1D containment ladder: [i, i+1]
        entity_boxes.insert(format!("e{i}"), make_box_1d(i as f32, i as f32 + 1.0));
    }

    // One relation box is enough for this test.
    let mut relation_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    relation_boxes.insert("r0".to_string(), make_box_1d(0.0, 1.0));

    // One test triple, plus a small train index (for filtering).
    let test_triples = vec![Triple {
        head: "e10".to_string(),
        relation: "r0".to_string(),
        tail: "e11".to_string(),
    }];

    let train_triples = vec![Triple {
        head: "e10".to_string(),
        relation: "r0".to_string(),
        tail: "e12".to_string(),
    }];
    let filter = FilteredTripleIndex::from_owned_triples(train_triples);

    // Warm up once to avoid first-use effects (e.g., thread-local init).
    let _ = subsume_core::trainer::evaluate_link_prediction_filtered(
        &test_triples,
        &entity_boxes,
        Some(&relation_boxes),
        &filter,
    )
    .expect("evaluation should succeed");

    // Measure allocations *during* the evaluation call.
    let reg = Region::new(GLOBAL);
    let _ = subsume_core::trainer::evaluate_link_prediction_filtered(
        &test_triples,
        &entity_boxes,
        Some(&relation_boxes),
        &filter,
    )
    .expect("evaluation should succeed");
    let s = reg.change();

    // We expect only a small constant number of allocations here (e.g., result vectors).
    // If this starts scaling with n_entities, we’ll see allocation counts explode.
    assert!(
        s.allocations <= 50,
        "too many allocations during filtered evaluation: {s:?}"
    );
}

