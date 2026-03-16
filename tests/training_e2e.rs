//! End-to-end integration tests for training pipelines.
//!
//! Covers box training (direct coordinate updates), cone training
//! (ConeEmbeddingTrainer), EL++ ontology training, dataset loading
//! roundtrip, and the full dataset-to-evaluation pipeline.

#![cfg(feature = "ndarray-backend")]

use std::collections::{HashMap, HashSet};
use std::io::Write;
use tempfile::tempdir;

use ndarray::Array1;
use subsume::dataset::load_dataset;
use subsume::metrics::{hits_at_k, mean_reciprocal_rank};
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;
use subsume::{ConeEmbeddingTrainer, TrainingConfig};

// ---------------------------------------------------------------------------
// 1. Box training convergence
// ---------------------------------------------------------------------------

/// Train box embeddings on a 10-entity taxonomy via direct coordinate updates.
/// Verify: violation decreases, containment > 0.5, volume ordering holds.
#[test]
fn box_training_convergence() {
    // Taxonomy (3 levels, 10 entities):
    //   entity
    //     animal
    //       dog, cat, bird
    //     vehicle
    //       car, truck
    //     plant
    //       oak, rose
    let containment_pairs: Vec<(&str, &str)> = vec![
        ("entity", "animal"),
        ("entity", "vehicle"),
        ("entity", "plant"),
        ("animal", "dog"),
        ("animal", "cat"),
        ("animal", "bird"),
        ("vehicle", "car"),
        ("vehicle", "truck"),
        ("plant", "oak"),
        ("plant", "rose"),
    ];

    let mut entity_set = HashSet::new();
    for (h, t) in &containment_pairs {
        entity_set.insert(*h);
        entity_set.insert(*t);
    }
    let entity_names: Vec<&str> = {
        let mut v: Vec<&str> = entity_set.iter().copied().collect();
        v.sort();
        v
    };

    // Build parent/children/depth/sibling structures
    let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut parent_of: HashMap<&str, &str> = HashMap::new();
    for &(head, tail) in &containment_pairs {
        children_of.entry(head).or_default().push(tail);
        parent_of.insert(tail, head);
    }

    let mut sibling_idx: HashMap<&str, usize> = HashMap::new();
    for children in children_of.values() {
        for (i, child) in children.iter().enumerate() {
            sibling_idx.insert(child, i);
        }
    }

    let mut depth: HashMap<&str, usize> = HashMap::new();
    for &name in &entity_names {
        let mut d = 0;
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            d += 1;
            cur = p;
        }
        depth.insert(name, d);
    }

    // Initialize boxes (hierarchy-aware)
    let dim = 8;
    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();
    for &name in &entity_names {
        let d = depth[name];
        let half = match d {
            0 => 5.0,
            1 => 3.0,
            2 => 1.5,
            _ => 0.4,
        };
        let mut center = vec![0.0f32; dim];
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            let si = sibling_idx.get(cur).copied().unwrap_or(0);
            let sep_dim = depth[cur] % dim;
            center[sep_dim] += (si as f32) * 2.5;
            cur = p;
        }
        let min_arr = Array1::from_vec(center.iter().map(|c| c - half).collect());
        let max_arr = Array1::from_vec(center.iter().map(|c| c + half).collect());
        boxes.insert(name, (min_arr, max_arr));
    }

    // Build negative (sibling) pairs
    let mut negative_pairs: Vec<(&str, &str)> = Vec::new();
    for children in children_of.values() {
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                negative_pairs.push((children[i], children[j]));
                negative_pairs.push((children[j], children[i]));
            }
        }
    }

    let heads: HashSet<&str> = containment_pairs.iter().map(|(h, _)| *h).collect();
    let leaves: Vec<&str> = entity_names
        .iter()
        .copied()
        .filter(|n| !heads.contains(n))
        .collect();

    let lr = 0.05;
    let neg_lr = 0.04;
    let shrink_lr = 0.002;
    let parent_shrink_lr = 0.03;
    let epochs = 100;

    let mut first_violation = 0.0f32;
    let mut last_violation = 0.0f32;

    for epoch in 0..epochs {
        let mut total_violation = 0.0f32;

        // Pass 1: positive containment
        for &(head, tail) in &containment_pairs {
            let (tail_min, tail_max) = boxes[tail].clone();
            let (head_min, head_max) = boxes.get_mut(head).unwrap();
            for d in 0..dim {
                let margin = 0.05;
                if head_min[d] > tail_min[d] - margin {
                    let v = head_min[d] - (tail_min[d] - margin);
                    head_min[d] -= lr * v;
                    total_violation += v.abs();
                }
                if head_max[d] < tail_max[d] + margin {
                    let v = (tail_max[d] + margin) - head_max[d];
                    head_max[d] += lr * v;
                    total_violation += v.abs();
                }
            }
        }

        // Pass 2: parent regularization
        for (parent, children) in &children_of {
            let mut child_min = vec![f32::MAX; dim];
            let mut child_max = vec![f32::MIN; dim];
            for &child in children {
                let (cmin, cmax) = &boxes[child];
                for d in 0..dim {
                    child_min[d] = child_min[d].min(cmin[d]);
                    child_max[d] = child_max[d].max(cmax[d]);
                }
            }
            let margin = 0.1;
            let (pmin, pmax) = boxes.get_mut(parent).unwrap();
            for d in 0..dim {
                let target_min = child_min[d] - margin;
                let target_max = child_max[d] + margin;
                if pmin[d] < target_min {
                    pmin[d] += parent_shrink_lr * (target_min - pmin[d]);
                }
                if pmax[d] > target_max {
                    pmax[d] -= parent_shrink_lr * (pmax[d] - target_max);
                }
            }
        }

        // Pass 3: negative separation
        for &(a_name, b_name) in &negative_pairs {
            let (b_min_r, b_max_r) = boxes[b_name].clone();
            let (a_min_r, a_max_r) = &boxes[a_name];
            let mut best_dim: Option<usize> = None;
            let mut best_gap = f32::MAX;
            for d in 0..dim {
                if a_min_r[d] <= b_min_r[d] && a_max_r[d] >= b_max_r[d] {
                    let gap = (b_min_r[d] - a_min_r[d]).min(a_max_r[d] - b_max_r[d]);
                    if gap < best_gap {
                        best_gap = gap;
                        best_dim = Some(d);
                    }
                }
            }
            if let Some(d) = best_dim {
                let (a_min, a_max) = boxes.get_mut(a_name).unwrap();
                let gap_min = b_min_r[d] - a_min[d];
                let gap_max = a_max[d] - b_max_r[d];
                if gap_min <= gap_max {
                    a_min[d] += neg_lr * (gap_min + 0.3);
                } else {
                    a_max[d] -= neg_lr * (gap_max + 0.3);
                }
                total_violation += best_gap;
            }
        }

        // Pass 4: leaf shrinkage
        for &leaf in &leaves {
            let (lmin, lmax) = boxes.get_mut(leaf).unwrap();
            for d in 0..dim {
                let center = (lmin[d] + lmax[d]) * 0.5;
                lmin[d] += shrink_lr * (center - lmin[d]);
                lmax[d] -= shrink_lr * (lmax[d] - center);
            }
        }

        // Enforce min < max
        for (_, (bmin, bmax)) in boxes.iter_mut() {
            for d in 0..dim {
                if bmin[d] >= bmax[d] {
                    let mid = (bmin[d] + bmax[d]) * 0.5;
                    bmin[d] = mid - 0.01;
                    bmax[d] = mid + 0.01;
                }
            }
        }

        if epoch == 0 {
            first_violation = total_violation;
        }
        if epoch == epochs - 1 {
            last_violation = total_violation;
        }
    }

    // Check 1: violation decreases
    assert!(
        last_violation < first_violation,
        "total violation should decrease: first={first_violation}, last={last_violation}"
    );

    // Build NdarrayBox instances for evaluation
    let entity_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .map(|(&name, (min, max))| {
            let b = NdarrayBox::new(min.clone(), max.clone(), 1.0).unwrap();
            (name, b)
        })
        .collect();

    // Check 2: all parent-child containment probabilities > 0.5
    for &(parent, child) in &containment_pairs {
        let pb = &entity_boxes[parent];
        let cb = &entity_boxes[child];
        let p = pb.containment_prob(cb, 1.0).unwrap();
        assert!(
            p > 0.5,
            "containment_prob({parent}, {child}) = {p}, expected > 0.5"
        );
    }

    // Check 3: volume ordering -- parent volume > child volume for >= 80% of pairs
    let mut volume_ok = 0;
    for &(parent, child) in &containment_pairs {
        let pv = entity_boxes[parent].volume(1.0).unwrap();
        let cv = entity_boxes[child].volume(1.0).unwrap();
        if pv > cv {
            volume_ok += 1;
        }
    }
    let ratio = volume_ok as f32 / containment_pairs.len() as f32;
    assert!(
        ratio >= 0.8,
        "volume ordering holds for {volume_ok}/{} pairs ({ratio:.0}%), expected >= 80%",
        containment_pairs.len()
    );
}

// ---------------------------------------------------------------------------
// 2. Cone training convergence
// ---------------------------------------------------------------------------

/// Train cone embeddings on a 10-entity taxonomy using ConeEmbeddingTrainer.
/// Verify: final loss < initial loss, parent apertures > child apertures on average.
#[test]
fn cone_training_convergence() {
    // Taxonomy:
    //   entity(0) > animal(1), vehicle(2), plant(3)
    //   animal(1) > dog(4), cat(5), bird(6)
    //   vehicle(2) > car(7), truck(8)
    //   plant(3) > oak(9)
    let positive_pairs: Vec<(usize, usize)> = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 7),
        (2, 8),
        (3, 9),
    ];

    let negative_pairs: Vec<(usize, usize)> = vec![
        (4, 0), // dog does NOT subsume entity
        (7, 0), // car does NOT subsume entity
        (1, 2), // animal does NOT subsume vehicle
        (2, 1), // vehicle does NOT subsume animal
        (4, 5), // dog does NOT subsume cat
        (5, 4), // cat does NOT subsume dog
        (7, 8), // car does NOT subsume truck
        (8, 7), // truck does NOT subsume car
        (4, 1), // dog does NOT subsume animal
        (6, 1), // bird does NOT subsume animal
        (7, 2), // car does NOT subsume vehicle
    ];

    let n_entities = 10;
    let dim = 16;
    let warmup_epochs = 30;
    let joint_epochs = 70;
    let total_epochs = warmup_epochs + joint_epochs;

    let config = TrainingConfig {
        learning_rate: 0.02,
        temperature: 1.0,
        margin: 1.0,
        regularization: 0.0,
        negative_weight: 0.5,
        ..Default::default()
    };

    let mut trainer = ConeEmbeddingTrainer::new(config, dim, None);
    for id in 0..n_entities {
        trainer.ensure_entity(id);
    }

    let mut first_joint_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for epoch in 0..total_epochs {
        let mut epoch_loss = 0.0;
        let mut n_pairs = 0;

        for &(head, tail) in &positive_pairs {
            epoch_loss += trainer.train_step(head, tail, true);
            n_pairs += 1;
        }

        if epoch >= warmup_epochs {
            for &(head, tail) in &negative_pairs {
                epoch_loss += trainer.train_step(head, tail, false);
                n_pairs += 1;
            }
        }

        let avg = epoch_loss / n_pairs as f32;
        // Record the first joint-phase loss (when negatives kick in).
        if epoch == warmup_epochs {
            first_joint_loss = avg;
        }
        if epoch == total_epochs - 1 {
            last_loss = avg;
        }
    }

    // Check 1: loss decreases during joint training (warmup-only loss is
    // artificially low because it lacks negative pairs).
    assert!(
        last_loss < first_joint_loss,
        "cone loss should decrease during joint phase: first_joint={first_joint_loss}, last={last_loss}"
    );

    // Check 2: parent apertures > child apertures on average
    //   Parents: 0 (entity), 1 (animal), 2 (vehicle), 3 (plant)
    //   Children: 4..9
    let parent_ids = [0usize, 1, 2, 3];
    let child_ids = [4usize, 5, 6, 7, 8, 9];

    let avg_parent_aper: f32 = parent_ids
        .iter()
        .map(|&id| trainer.cones[&id].mean_aperture())
        .sum::<f32>()
        / parent_ids.len() as f32;

    let avg_child_aper: f32 = child_ids
        .iter()
        .map(|&id| trainer.cones[&id].mean_aperture())
        .sum::<f32>()
        / child_ids.len() as f32;

    assert!(
        avg_parent_aper > avg_child_aper,
        "avg parent aperture ({avg_parent_aper:.4}) should be > avg child aperture ({avg_child_aper:.4})"
    );
}

// ---------------------------------------------------------------------------
// 3. EL++ training convergence
// ---------------------------------------------------------------------------

/// Train EL++ box embeddings on a mini ontology and evaluate subsumption.
/// Verify: loss decreases, Hits@10 > 0.5.
#[test]
fn el_training_convergence() {
    use subsume::{evaluate_subsumption, train_el_embeddings, ElTrainingConfig, Ontology};

    let ontology_text = "\
SubClassOf Dog Animal
SubClassOf Cat Animal
SubClassOf Eagle Bird
SubClassOf Bird Animal
SubClassOf Animal LivingThing
Disjoint Dog Cat
Disjoint Bird Dog
Existential eats Plant Animal
";

    let ontology = Ontology::parse(ontology_text.as_bytes()).expect("parse ontology");
    assert_eq!(ontology.num_concepts(), 7); // Dog, Cat, Eagle, Bird, Animal, LivingThing, Plant
    assert_eq!(ontology.num_roles(), 1); // eats

    let config = ElTrainingConfig {
        dim: 20,
        epochs: 200,
        learning_rate: 0.005,
        margin: 0.05,
        negative_samples: 3,
        warmup_epochs: 10,
        log_interval: 0, // suppress output
        seed: 42,
        ..Default::default()
    };

    let result = train_el_embeddings(&ontology, &config);

    // Check 1: loss decreases
    let first_loss = result.epoch_losses[0];
    let last_loss = *result.epoch_losses.last().unwrap();
    assert!(
        last_loss < first_loss,
        "EL++ loss should decrease: first={first_loss}, last={last_loss}"
    );

    // Check 2: Hits@10 > 0.5
    let (_hits1, hits10, _mrr) = evaluate_subsumption(&result, &ontology.axioms);
    assert!(hits10 > 0.5, "EL++ Hits@10 = {hits10}, expected > 0.5");
}

// ---------------------------------------------------------------------------
// 4. Dataset roundtrip
// ---------------------------------------------------------------------------

/// Write triples to temp files, load with load_dataset, verify counts match.
#[test]
fn dataset_roundtrip() {
    let dir = tempdir().expect("create temp dir");

    let train_data = "dog\t_hypernym\tanimal\ncat\t_hypernym\tanimal\neagle\t_hypernym\tbird\n";
    let valid_data = "bird\t_hypernym\tanimal\n";
    let test_data = "salmon\t_hypernym\tfish\noak\t_hypernym\tplant\n";

    let mut f = std::fs::File::create(dir.path().join("train.txt")).unwrap();
    write!(f, "{train_data}").unwrap();
    let mut f = std::fs::File::create(dir.path().join("valid.txt")).unwrap();
    write!(f, "{valid_data}").unwrap();
    let mut f = std::fs::File::create(dir.path().join("test.txt")).unwrap();
    write!(f, "{test_data}").unwrap();

    let dataset = load_dataset(dir.path()).expect("load_dataset");
    let stats = dataset.stats();

    assert_eq!(stats.num_train, 3, "train triples");
    assert_eq!(stats.num_valid, 1, "valid triples");
    assert_eq!(stats.num_test, 2, "test triples");

    // 8 unique entities: dog, cat, eagle, bird, animal, salmon, fish, oak, plant
    assert_eq!(stats.num_entities, 9, "unique entities");

    // 1 unique relation: _hypernym
    assert_eq!(stats.num_relations, 1, "unique relations");

    // Spot-check a training triple
    assert_eq!(dataset.train[0].head, "dog");
    assert_eq!(dataset.train[0].relation, "_hypernym");
    assert_eq!(dataset.train[0].tail, "animal");
}

// ---------------------------------------------------------------------------
// 5. Full pipeline: load dataset, train box embeddings, evaluate MRR
// ---------------------------------------------------------------------------

/// End-to-end: dataset -> box training -> link prediction MRR > 0.0.
#[test]
fn full_pipeline_mrr() {
    let dir = tempdir().expect("create temp dir");

    let train_data = "\
dog\t_hypernym\tcanine
canine\t_hypernym\tmammal
mammal\t_hypernym\tanimal
cat\t_hypernym\tfeline
feline\t_hypernym\tmammal
eagle\t_hypernym\tbird
bird\t_hypernym\tanimal
salmon\t_hypernym\tfish
fish\t_hypernym\tanimal
animal\t_hypernym\tentity";

    let valid_data = "\
dog\t_hypernym\tmammal
eagle\t_hypernym\tanimal";

    let test_data = "\
cat\t_hypernym\tmammal
salmon\t_hypernym\tanimal";

    let mut f = std::fs::File::create(dir.path().join("train.txt")).unwrap();
    write!(f, "{train_data}").unwrap();
    let mut f = std::fs::File::create(dir.path().join("valid.txt")).unwrap();
    write!(f, "{valid_data}").unwrap();
    let mut f = std::fs::File::create(dir.path().join("test.txt")).unwrap();
    write!(f, "{test_data}").unwrap();

    let dataset = load_dataset(dir.path()).expect("load_dataset");

    // Build hierarchy from training triples (child _hypernym parent)
    let mut parent_of: HashMap<&str, &str> = HashMap::new();
    let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();
    for t in &dataset.train {
        parent_of.insert(t.head.as_str(), t.tail.as_str());
        children_of
            .entry(t.tail.as_str())
            .or_default()
            .push(t.head.as_str());
    }

    let entity_set = dataset.entities();
    let mut entity_names: Vec<&str> = entity_set.iter().map(|s| s.as_str()).collect();
    entity_names.sort();

    // Compute depth
    let mut depth: HashMap<&str, usize> = HashMap::new();
    for &name in &entity_names {
        let mut d = 0;
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            d += 1;
            cur = p;
        }
        depth.insert(name, d);
    }

    let mut sibling_idx: HashMap<&str, usize> = HashMap::new();
    for children in children_of.values() {
        for (i, child) in children.iter().enumerate() {
            sibling_idx.insert(child, i);
        }
    }

    // Initialize boxes
    let dim = 8;
    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();
    for &name in &entity_names {
        let d = depth.get(name).copied().unwrap_or(0);
        let half = match d {
            0 => 5.0,
            1 => 3.5,
            2 => 2.0,
            3 => 1.0,
            _ => 0.4,
        };
        let mut center = vec![0.0f32; dim];
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            let si = sibling_idx.get(cur).copied().unwrap_or(0);
            let sep_dim = depth.get(cur).copied().unwrap_or(0) % dim;
            center[sep_dim] += (si as f32) * 2.5;
            cur = p;
        }
        let min_arr = Array1::from_vec(center.iter().map(|c| c - half).collect());
        let max_arr = Array1::from_vec(center.iter().map(|c| c + half).collect());
        boxes.insert(name, (min_arr, max_arr));
    }

    let containment_pairs: Vec<(&str, &str)> = dataset
        .train
        .iter()
        .map(|t| (t.tail.as_str(), t.head.as_str()))
        .collect();

    let mut negative_pairs: Vec<(&str, &str)> = Vec::new();
    for children in children_of.values() {
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                negative_pairs.push((children[i], children[j]));
                negative_pairs.push((children[j], children[i]));
            }
        }
    }

    let head_set: HashSet<&str> = containment_pairs.iter().map(|(h, _)| *h).collect();
    let leaves: Vec<&str> = entity_names
        .iter()
        .copied()
        .filter(|n| !head_set.contains(n))
        .collect();

    // Train
    let lr = 0.05;
    let neg_lr = 0.04;
    let shrink_lr = 0.002;
    let parent_shrink_lr = 0.03;
    let epochs = 100;

    for _epoch in 0..epochs {
        // Pass 1: positive containment
        for &(head, tail) in &containment_pairs {
            let (tail_min, tail_max) = boxes[tail].clone();
            let (head_min, head_max) = boxes.get_mut(head).unwrap();
            for d in 0..dim {
                let margin = 0.05;
                if head_min[d] > tail_min[d] - margin {
                    let v = head_min[d] - (tail_min[d] - margin);
                    head_min[d] -= lr * v;
                }
                if head_max[d] < tail_max[d] + margin {
                    let v = (tail_max[d] + margin) - head_max[d];
                    head_max[d] += lr * v;
                }
            }
        }

        // Pass 2: parent regularization
        for (parent, children) in &children_of {
            let mut child_min = vec![f32::MAX; dim];
            let mut child_max = vec![f32::MIN; dim];
            for &child in children {
                let (cmin, cmax) = &boxes[child];
                for d in 0..dim {
                    child_min[d] = child_min[d].min(cmin[d]);
                    child_max[d] = child_max[d].max(cmax[d]);
                }
            }
            let margin = 0.1;
            let (pmin, pmax) = boxes.get_mut(parent).unwrap();
            for d in 0..dim {
                let target_min = child_min[d] - margin;
                let target_max = child_max[d] + margin;
                if pmin[d] < target_min {
                    pmin[d] += parent_shrink_lr * (target_min - pmin[d]);
                }
                if pmax[d] > target_max {
                    pmax[d] -= parent_shrink_lr * (pmax[d] - target_max);
                }
            }
        }

        // Pass 3: negative separation
        for &(a_name, b_name) in &negative_pairs {
            let (b_min_r, b_max_r) = boxes[b_name].clone();
            let (a_min_r, a_max_r) = &boxes[a_name];
            let mut best_dim: Option<usize> = None;
            let mut best_gap = f32::MAX;
            for d in 0..dim {
                if a_min_r[d] <= b_min_r[d] && a_max_r[d] >= b_max_r[d] {
                    let gap = (b_min_r[d] - a_min_r[d]).min(a_max_r[d] - b_max_r[d]);
                    if gap < best_gap {
                        best_gap = gap;
                        best_dim = Some(d);
                    }
                }
            }
            if let Some(d) = best_dim {
                let (a_min, a_max) = boxes.get_mut(a_name).unwrap();
                let gap_min = b_min_r[d] - a_min[d];
                let gap_max = a_max[d] - b_max_r[d];
                if gap_min <= gap_max {
                    a_min[d] += neg_lr * (gap_min + 0.3);
                } else {
                    a_max[d] -= neg_lr * (gap_max + 0.3);
                }
            }
        }

        // Pass 4: leaf shrinkage
        for &leaf in &leaves {
            let (lmin, lmax) = boxes.get_mut(leaf).unwrap();
            for d in 0..dim {
                let center = (lmin[d] + lmax[d]) * 0.5;
                lmin[d] += shrink_lr * (center - lmin[d]);
                lmax[d] -= shrink_lr * (lmax[d] - center);
            }
        }

        // Enforce min < max
        for (_, (bmin, bmax)) in boxes.iter_mut() {
            for d in 0..dim {
                if bmin[d] >= bmax[d] {
                    let mid = (bmin[d] + bmax[d]) * 0.5;
                    bmin[d] = mid - 0.01;
                    bmax[d] = mid + 0.01;
                }
            }
        }
    }

    // Convert to NdarrayBox for evaluation
    let ndarray_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .filter_map(|(name, (min, max))| {
            NdarrayBox::new(min.clone(), max.clone(), 1.0)
                .ok()
                .map(|b| (*name, b))
        })
        .collect();

    // Evaluate on test set: rank the true parent among all entities
    let mut ranks = Vec::new();
    for triple in &dataset.test {
        let child_name = triple.head.as_str();
        let true_parent = triple.tail.as_str();

        let Some(child_b) = ndarray_boxes.get(child_name) else {
            continue;
        };

        let mut scores: Vec<(&str, f32)> = entity_names
            .iter()
            .filter_map(|&ename| {
                let parent_b = ndarray_boxes.get(ename)?;
                let p = parent_b.containment_prob(child_b, 1.0).ok()?;
                Some((ename, p))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rank = scores
            .iter()
            .position(|(name, _)| *name == true_parent)
            .map(|pos| pos + 1)
            .unwrap_or(0);

        ranks.push(rank);
    }

    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let h10 = hits_at_k(ranks.iter().copied(), 10);

    assert!(
        mrr > 0.0,
        "MRR should be > 0.0 (got {mrr}), pipeline produces non-degenerate results"
    );
    assert!(h10 > 0.0, "Hits@10 should be > 0.0 (got {h10})");
}
