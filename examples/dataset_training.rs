//! End-to-end box embedding training on a real-format knowledge graph dataset.
//!
//! Demonstrates the full pipeline: load a dataset in WN18RR/FB15k-237 format,
//! train box embeddings with direct coordinate updates, and evaluate with
//! standard link prediction metrics (MRR, Hits@1, Hits@10).
//!
//! The dataset is a 60-triple subset of WordNet hypernym relations, embedded
//! inline so the example is self-contained. The format matches WN18RR:
//! tab-separated (head, relation, tail) triples in train/valid/test splits.
//!
//! Training uses four passes per epoch (same approach as `box_training`):
//!   1. Positive: expand parent boxes to contain children
//!   2. Parent regularization: tighten parents to children's extent
//!   3. Negative: break containment between siblings/unrelated pairs
//!   4. Leaf shrinkage: contract leaf boxes for volume differentiation
//!
//! References:
//! - Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
//! - Abboud et al. (2020), "BoxE: A Box Embedding Model for Knowledge Base Completion"
//! - Dettmers et al. (2018), "Convolutional 2D Knowledge Graph Embeddings" (WN18RR)
//!
//! Run: cargo run -p subsume --example dataset_training --release
//!
//! Related examples:
//! - `box_training`: hand-placed taxonomy with same training approach
//! - `el_training`: EL++ ontology embeddings with role composition
//! - `cone_training`: cone embeddings with angular containment

use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use subsume::dataset::load_dataset;
use subsume::ndarray_backend::NdarrayBox;
use subsume::training::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank};
use subsume::Box as BoxTrait;

/// WordNet hypernym triples (subset). Format: head \t relation \t tail.
/// These represent real is-a relationships from WordNet's noun hierarchy.
const TRAIN_DATA: &str = "\
dog.n.01\t_hypernym\tcanine.n.02
canine.n.02\t_hypernym\tcarnivore.n.01
carnivore.n.01\t_hypernym\tplacental.n.01
placental.n.01\t_hypernym\tmammal.n.01
mammal.n.01\t_hypernym\tvertebrate.n.01
vertebrate.n.01\t_hypernym\tchordate.n.01
chordate.n.01\t_hypernym\tanimal.n.01
animal.n.01\t_hypernym\torganism.n.01
organism.n.01\t_hypernym\tentity.n.01
cat.n.01\t_hypernym\tfeline.n.01
feline.n.01\t_hypernym\tcarnivore.n.01
wolf.n.01\t_hypernym\tcanine.n.02
fox.n.01\t_hypernym\tcanine.n.02
lion.n.01\t_hypernym\tfeline.n.01
tiger.n.01\t_hypernym\tfeline.n.01
horse.n.01\t_hypernym\tequine.n.01
equine.n.01\t_hypernym\tplacental.n.01
eagle.n.01\t_hypernym\tbird_of_prey.n.01
bird_of_prey.n.01\t_hypernym\tbird.n.01
bird.n.01\t_hypernym\tvertebrate.n.01
sparrow.n.01\t_hypernym\tpasserine.n.01
passerine.n.01\t_hypernym\tbird.n.01
salmon.n.01\t_hypernym\tfish.n.01
fish.n.01\t_hypernym\tvertebrate.n.01
trout.n.01\t_hypernym\tfish.n.01
oak.n.01\t_hypernym\ttree.n.01
tree.n.01\t_hypernym\tplant.n.02
plant.n.02\t_hypernym\torganism.n.01
pine.n.01\t_hypernym\ttree.n.01
rose.n.01\t_hypernym\tflower.n.01
flower.n.01\t_hypernym\tplant.n.02
tulip.n.01\t_hypernym\tflower.n.01
car.n.01\t_hypernym\tvehicle.n.01
vehicle.n.01\t_hypernym\tartifact.n.01
artifact.n.01\t_hypernym\tentity.n.01
truck.n.01\t_hypernym\tvehicle.n.01
bicycle.n.01\t_hypernym\tvehicle.n.01
whale.n.01\t_hypernym\tplacental.n.01
dolphin.n.01\t_hypernym\tplacental.n.01
snake.n.01\t_hypernym\treptile.n.01
reptile.n.01\t_hypernym\tvertebrate.n.01
lizard.n.01\t_hypernym\treptile.n.01
penguin.n.01\t_hypernym\tbird.n.01
bat.n.01\t_hypernym\tplacental.n.01
spider.n.01\t_hypernym\tarthropod.n.01
arthropod.n.01\t_hypernym\tanimal.n.01";

const VALID_DATA: &str = "\
wolf.n.01\t_hypernym\tcarnivore.n.01
horse.n.01\t_hypernym\tmammal.n.01
eagle.n.01\t_hypernym\tbird.n.01
oak.n.01\t_hypernym\tplant.n.02
car.n.01\t_hypernym\tartifact.n.01
snake.n.01\t_hypernym\tvertebrate.n.01
spider.n.01\t_hypernym\tanimal.n.01";

const TEST_DATA: &str = "\
fox.n.01\t_hypernym\tcarnivore.n.01
lion.n.01\t_hypernym\tcarnivore.n.01
tiger.n.01\t_hypernym\tcarnivore.n.01
trout.n.01\t_hypernym\tvertebrate.n.01
pine.n.01\t_hypernym\tplant.n.02
tulip.n.01\t_hypernym\tplant.n.02
truck.n.01\t_hypernym\tartifact.n.01";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dataset-Driven Box Embedding Training ===\n");

    // --- Step 1: Write dataset to temp files and load ---
    let dir = tempfile::tempdir()?;
    let mut train_file = std::fs::File::create(dir.path().join("train.txt"))?;
    let mut valid_file = std::fs::File::create(dir.path().join("valid.txt"))?;
    let mut test_file = std::fs::File::create(dir.path().join("test.txt"))?;
    write!(train_file, "{TRAIN_DATA}")?;
    write!(valid_file, "{VALID_DATA}")?;
    write!(test_file, "{TEST_DATA}")?;

    let dataset = load_dataset(dir.path())?;
    let stats = dataset.stats();
    println!(
        "Dataset: {} entities, {} relations, {} train / {} valid / {} test triples",
        stats.num_entities, stats.num_relations, stats.num_train, stats.num_valid, stats.num_test
    );

    // --- Step 2: Build hierarchy from training triples ---
    // Extract parent-child relationships (child _hypernym parent)
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

    // Compute depth (root = 0)
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

    // Sibling indices for spatial separation
    let mut sibling_idx: HashMap<&str, usize> = HashMap::new();
    for children in children_of.values() {
        for (i, child) in children.iter().enumerate() {
            sibling_idx.insert(child, i);
        }
    }

    // --- Step 3: Initialize box embeddings (hierarchy-aware) ---
    let dim = 12;
    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();
    for &name in &entity_names {
        let d = depth.get(name).copied().unwrap_or(0);
        let half = match d {
            0 => 6.0,
            1 => 4.0,
            2 => 2.5,
            3 => 1.5,
            4 => 0.8,
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

    // Build containment pairs and negative (sibling) pairs
    let containment_pairs: Vec<(&str, &str)> = dataset
        .train
        .iter()
        .map(|t| (t.tail.as_str(), t.head.as_str()))
        .collect(); // (parent, child)

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

    // --- Step 4: Train ---
    let lr = 0.05;
    let neg_lr = 0.04;
    let shrink_lr = 0.002;
    let parent_shrink_lr = 0.03;
    let epochs = 200;

    println!(
        "\nTraining {} epochs (dim={}, {} containment pairs, {} negative pairs)...\n",
        epochs,
        dim,
        containment_pairs.len(),
        negative_pairs.len()
    );

    for epoch in 0..epochs {
        let mut total_violation = 0.0f32;

        // Pass 1: positive containment -- expand parent to contain child
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

        // Pass 2: parent regularization -- tighten parents to children's extent
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

        // Pass 3: negative separation -- break containment between siblings
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

        // Pass 4: leaf shrinkage -- contract leaves toward center
        for &leaf in &leaves {
            let (lmin, lmax) = boxes.get_mut(leaf).unwrap();
            let center: Vec<f32> = lmin
                .iter()
                .zip(lmax.iter())
                .map(|(&a, &b)| (a + b) / 2.0)
                .collect();
            for d in 0..dim {
                lmin[d] += shrink_lr * (center[d] - lmin[d]);
                lmax[d] -= shrink_lr * (lmax[d] - center[d]);
            }
        }

        if epoch % 40 == 0 || epoch == epochs - 1 {
            println!(
                "  epoch {:>3}: total_violation = {:.4}",
                epoch, total_violation
            );
        }
    }

    // --- Step 5: Evaluate on test set ---
    println!("\n--- Evaluation (test set) ---\n");

    // Convert to NdarrayBox for trait-based evaluation
    let ndarray_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .filter_map(|(name, (min, max))| {
            NdarrayBox::new(min.clone(), max.clone(), 1.0)
                .ok()
                .map(|b| (*name, b))
        })
        .collect();

    let mut ranks = Vec::new();
    let n_entities = entity_names.len();

    for triple in &dataset.test {
        let child_name = triple.head.as_str();
        let true_parent = triple.tail.as_str();

        let Some(child_b) = ndarray_boxes.get(child_name) else {
            continue;
        };

        // Score all entities as potential parents
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
        println!(
            "  {} -> {}: rank {} / {}",
            child_name, true_parent, rank, n_entities
        );
    }

    // Compute metrics
    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let h1 = hits_at_k(ranks.iter().copied(), 1);
    let h3 = hits_at_k(ranks.iter().copied(), 3);
    let h10 = hits_at_k(ranks.iter().copied(), 10);
    let mr = mean_rank(ranks.iter().copied());

    println!("\n--- Link Prediction Metrics ---\n");
    println!("  MRR:       {mrr:.4}");
    println!("  Hits@1:    {h1:.4}");
    println!("  Hits@3:    {h3:.4}");
    println!("  Hits@10:   {h10:.4}");
    println!("  Mean Rank: {mr:.1}");
    println!(
        "\n  ({} test triples, {} candidate entities)",
        ranks.len(),
        n_entities
    );

    Ok(())
}
