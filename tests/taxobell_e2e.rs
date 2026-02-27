//! End-to-end integration test for the TaxoBell pipeline.
//!
//! Builds a small taxonomy inline, creates Gaussian boxes for each concept,
//! computes the combined TaxoBell loss, and verifies that the math is
//! consistent with the expected hierarchy.

use std::collections::HashMap;
use std::io::Write;
use tempfile::tempdir;

use subsume::gaussian::{
    bhattacharyya_coefficient, kl_divergence, GaussianBox,
};
use subsume::taxonomy::TaxonomyDataset;
use subsume::taxobell::{TaxoBellConfig, TaxoBellLoss};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a small taxonomy on disk and load it.
///
/// Taxonomy:
///   Animal (0)
///     Dog (1)
///       Poodle (3)
///       Labrador (4)
///     Cat (2)
///       Siamese (5)
///     Bird (6)
fn build_taxonomy() -> TaxonomyDataset {
    let dir = tempdir().expect("failed to create temp dir");

    let terms_path = dir.path().join("test.terms");
    let taxo_path = dir.path().join("test.taxo");

    let mut terms_file = std::fs::File::create(&terms_path).unwrap();
    write!(
        terms_file,
        "0\tanimal\n1\tdog\n2\tcat\n3\tpoodle\n4\tlabrador\n5\tsiamese\n6\tbird\n"
    )
    .unwrap();

    let mut taxo_file = std::fs::File::create(&taxo_path).unwrap();
    // parent_id  child_id
    write!(
        taxo_file,
        "0\t1\n0\t2\n0\t6\n1\t3\n1\t4\n2\t5\n"
    )
    .unwrap();

    // Keep the tempdir alive by leaking it (tests are short-lived).
    let ds = TaxonomyDataset::load(&terms_path, &taxo_path, None).unwrap();
    // Prevent cleanup until after we've loaded.
    std::mem::forget(dir);
    ds
}

/// Assign fixed Gaussian boxes to each concept.
///
/// Strategy: parent boxes are wide (large sigma) and centered at the origin;
/// children are narrower and centered near the parent. Distant concepts
/// (e.g., Bird vs Siamese) get separated centers.
fn build_boxes() -> HashMap<&'static str, GaussianBox> {
    let dim = 8;
    let mut boxes = HashMap::new();

    // Root: wide box centered at origin.
    boxes.insert("animal", GaussianBox::new(vec![0.0; dim], vec![4.0; dim]).unwrap());

    // Children of Animal: narrower, slightly offset.
    boxes.insert("dog",  GaussianBox::new(vec![-1.0; dim], vec![2.0; dim]).unwrap());
    boxes.insert("cat",  GaussianBox::new(vec![1.0; dim],  vec![2.0; dim]).unwrap());
    boxes.insert("bird", GaussianBox::new(vec![6.0; dim],  vec![1.5; dim]).unwrap());

    // Children of Dog: even narrower, close to dog's center.
    boxes.insert("poodle",   GaussianBox::new(vec![-1.2; dim], vec![0.8; dim]).unwrap());
    boxes.insert("labrador", GaussianBox::new(vec![-0.8; dim], vec![0.8; dim]).unwrap());

    // Child of Cat.
    boxes.insert("siamese", GaussianBox::new(vec![1.1; dim], vec![0.7; dim]).unwrap());

    boxes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn taxonomy_loads_and_converts_to_triples() {
    let ds = build_taxonomy();
    assert_eq!(ds.num_nodes(), 7, "7 concepts in the taxonomy");
    assert_eq!(ds.num_edges(), 6, "6 parent->child edges");

    let triples = ds.to_triples();
    assert_eq!(triples.len(), 6);

    // Every triple should have relation "hypernym".
    for t in &triples {
        assert_eq!(t.relation, "hypernym");
    }
}

#[test]
fn taxonomy_split_covers_all_edges() {
    let ds = build_taxonomy();
    let (train, val, test) = ds.split(0.6, 0.2, 42);
    assert_eq!(
        train.len() + val.len() + test.len(),
        6,
        "split must cover all edges"
    );
}

#[test]
fn combined_loss_is_finite_and_positive() {
    let ds = build_taxonomy();
    let boxes = build_boxes();
    let (train_edges, _, _) = ds.split(0.8, 0.1, 42);

    let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());

    // Build positive pairs (child, parent) from training edges.
    let positives: Vec<(&GaussianBox, &GaussianBox)> = train_edges
        .iter()
        .filter_map(|&(parent_id, child_id)| {
            let parent_name = &ds.nodes[ds.node_index[&parent_id]].name;
            let child_name = &ds.nodes[ds.node_index[&child_id]].name;
            Some((
                boxes.get(child_name.as_str())?,
                boxes.get(parent_name.as_str())?,
            ))
        })
        .collect();

    // For simplicity, use a single symmetric triplet: dog (anchor),
    // poodle (positive sibling of dog's subtree), bird (negative).
    let negatives = vec![(
        boxes.get("dog").unwrap(),
        boxes.get("poodle").unwrap(),
        boxes.get("bird").unwrap(),
    )];

    let all: Vec<&GaussianBox> = boxes.values().collect();

    let result = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();

    assert!(result.total.is_finite(), "total loss must be finite");
    assert!(result.total >= 0.0, "total loss must be non-negative, got {}", result.total);
    assert!(result.l_asym.is_finite(), "l_asym must be finite");
    assert!(result.l_reg.is_finite(), "l_reg must be finite");
    assert!(result.l_clip.is_finite(), "l_clip must be finite");
}

#[test]
fn loss_decreases_when_children_fit_inside_parents() {
    let dim = 8;
    let loss_fn = TaxoBellLoss::new(TaxoBellConfig {
        alpha: 0.0,   // disable symmetric to isolate containment
        beta: 1.0,
        gamma: 0.0,
        delta: 0.0,
        ..TaxoBellConfig::default()
    });

    let parent = GaussianBox::new(vec![0.0; dim], vec![3.0; dim]).unwrap();

    // "Before": child is far away and wide -- bad containment.
    let child_before = GaussianBox::new(vec![5.0; dim], vec![4.0; dim]).unwrap();

    // "After": child moved inside parent and narrowed -- good containment.
    let child_after = GaussianBox::new(vec![0.0; dim], vec![1.0; dim]).unwrap();

    let loss_before = loss_fn
        .combined_loss(&[(&child_before, &parent)], &[], &[&child_before, &parent])
        .unwrap()
        .total;

    let loss_after = loss_fn
        .combined_loss(&[(&child_after, &parent)], &[], &[&child_after, &parent])
        .unwrap()
        .total;

    assert!(
        loss_after < loss_before,
        "loss should decrease when child fits inside parent: {loss_after} < {loss_before}"
    );
}

#[test]
fn kl_child_parent_less_than_parent_child() {
    let boxes = build_boxes();

    let pairs: Vec<(&str, &str)> = vec![
        ("poodle", "dog"),
        ("labrador", "dog"),
        ("siamese", "cat"),
        ("dog", "animal"),
        ("cat", "animal"),
        ("bird", "animal"),
    ];

    for (child_name, parent_name) in &pairs {
        let child = boxes.get(child_name).unwrap();
        let parent = boxes.get(parent_name).unwrap();

        let kl_cp = kl_divergence(child, parent).unwrap();
        let kl_pc = kl_divergence(parent, child).unwrap();

        assert!(
            kl_cp < kl_pc,
            "D_KL({child_name}||{parent_name}) = {kl_cp} should be < \
             D_KL({parent_name}||{child_name}) = {kl_pc}"
        );
    }
}

#[test]
fn bc_siblings_greater_than_bc_distant() {
    let boxes = build_boxes();

    // Siblings: poodle and labrador (both children of dog).
    let bc_siblings = bhattacharyya_coefficient(
        boxes.get("poodle").unwrap(),
        boxes.get("labrador").unwrap(),
    )
    .unwrap();

    // Distant: poodle and bird (different branches entirely).
    let bc_distant = bhattacharyya_coefficient(
        boxes.get("poodle").unwrap(),
        boxes.get("bird").unwrap(),
    )
    .unwrap();

    assert!(
        bc_siblings > bc_distant,
        "BC(poodle, labrador) = {bc_siblings} should be > BC(poodle, bird) = {bc_distant}"
    );

    // Also check cat-branch siblings vs distant.
    // Siamese is the only cat child, so compare cat vs siamese overlap
    // against cat vs bird overlap.
    let bc_cat_siamese = bhattacharyya_coefficient(
        boxes.get("cat").unwrap(),
        boxes.get("siamese").unwrap(),
    )
    .unwrap();

    let bc_cat_bird = bhattacharyya_coefficient(
        boxes.get("cat").unwrap(),
        boxes.get("bird").unwrap(),
    )
    .unwrap();

    assert!(
        bc_cat_siamese > bc_cat_bird,
        "BC(cat, siamese) = {bc_cat_siamese} should be > BC(cat, bird) = {bc_cat_bird}"
    );
}

#[test]
fn from_center_offset_roundtrip_in_pipeline() {
    // Verify that from_center_offset produces valid boxes that work through
    // the full loss pipeline.
    let dim = 4;
    let parent = GaussianBox::from_center_offset(vec![0.0; dim], vec![2.0; dim]).unwrap();
    let child = GaussianBox::from_center_offset(vec![0.1; dim], vec![0.0; dim]).unwrap();
    let negative = GaussianBox::from_center_offset(vec![10.0; dim], vec![1.0; dim]).unwrap();

    let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());

    let result = loss_fn
        .combined_loss(
            &[(&child, &parent)],
            &[(&child, &parent, &negative)],
            &[&parent, &child, &negative],
        )
        .unwrap();

    assert!(result.total.is_finite(), "loss must be finite with from_center_offset boxes");
    assert!(result.total >= 0.0);
}

#[test]
fn full_pipeline_smoke_test() {
    // Wire it all together: taxonomy -> triples -> boxes -> loss.
    let ds = build_taxonomy();
    let triples = ds.to_triples();
    let (train_edges, val_edges, test_edges) = ds.split(0.7, 0.15, 99);
    let boxes = build_boxes();

    // Verify split is non-empty in each partition.
    assert!(!train_edges.is_empty(), "train split should be non-empty");
    assert!(!val_edges.is_empty() || !test_edges.is_empty(),
        "at least one of val/test should be non-empty for 6 edges");

    // Build positive pairs from training edges.
    let positives: Vec<(&GaussianBox, &GaussianBox)> = train_edges
        .iter()
        .filter_map(|&(pid, cid)| {
            let pname = &ds.nodes[ds.node_index[&pid]].name;
            let cname = &ds.nodes[ds.node_index[&cid]].name;
            Some((boxes.get(cname.as_str())?, boxes.get(pname.as_str())?))
        })
        .collect();

    // Build a few negative triples.
    let negatives = vec![
        (
            boxes.get("dog").unwrap(),
            boxes.get("cat").unwrap(),
            boxes.get("bird").unwrap(),
        ),
    ];

    let all: Vec<&GaussianBox> = boxes.values().collect();

    let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
    let result = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();

    // Verify the total decomposes correctly.
    let cfg = &loss_fn.config;
    let expected_total = cfg.alpha * result.l_sym
        + cfg.beta * result.l_asym
        + cfg.gamma * result.l_reg
        + cfg.delta * result.l_clip;

    assert!(
        (result.total - expected_total).abs() < 1e-5,
        "total={} != expected={expected_total}",
        result.total
    );

    // Verify triples are well-formed.
    for t in &triples {
        assert!(!t.head.is_empty());
        assert!(!t.tail.is_empty());
        assert_eq!(t.relation, "hypernym");
    }
}
