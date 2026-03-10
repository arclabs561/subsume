//! Training box embeddings on a 20+ entity taxonomy.
//!
//! This example trains axis-aligned box embeddings to represent hierarchical
//! relationships through geometric containment. A box A containing box B
//! means "B is-a A" (e.g., dog is-a mammal).
//!
//! Taxonomy (3 levels, 25 entities):
//!   entity
//!     animal
//!       mammal: dog, cat, whale, bat
//!       bird: eagle, sparrow, penguin
//!       fish: salmon, tuna
//!     plant
//!       tree: oak, pine
//!       flower: rose, tulip
//!     vehicle
//!       car, truck, bicycle
//!
//! The training uses direct coordinate updates with three passes per epoch:
//!   1. Positive: expand head boxes to contain their children (push min down, max up).
//!   2. Negative: for sibling and cross-branch pairs, shrink boxes to break
//!      full containment on at least some dimensions.
//!   3. Shrinkage: gently contract leaf boxes toward their center so they
//!      develop varied, tighter volumes.
//!
//! This is a simplified approach; production systems would use backpropagation
//! through the containment probability.
//!
//! Reference: Vilnis et al. (2018), "Probabilistic Embedding of Knowledge
//! Graphs with Box Lattice Measures"
//!
//! Run: cargo run -p subsume --example box_training

use ndarray::Array1;
use std::collections::HashMap;
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Box Embedding Training (25 entities, direct coordinate updates) ===\n");

    // --- Define taxonomy as (head, tail) containment pairs ---
    // head should contain tail (head is more general).
    let containment_pairs: Vec<(&str, &str)> = vec![
        // Level 0 -> 1
        ("entity", "animal"),
        ("entity", "plant"),
        ("entity", "vehicle"),
        // Level 1 -> 2
        ("animal", "mammal"),
        ("animal", "bird"),
        ("animal", "fish"),
        ("plant", "tree"),
        ("plant", "flower"),
        // Level 2 -> 3
        ("mammal", "dog"),
        ("mammal", "cat"),
        ("mammal", "whale"),
        ("mammal", "bat"),
        ("bird", "eagle"),
        ("bird", "sparrow"),
        ("bird", "penguin"),
        ("fish", "salmon"),
        ("fish", "tuna"),
        ("tree", "oak"),
        ("tree", "pine"),
        ("flower", "rose"),
        ("flower", "tulip"),
        ("vehicle", "car"),
        ("vehicle", "truck"),
        ("vehicle", "bicycle"),
    ];

    // Collect all entity names
    let mut entity_set = std::collections::HashSet::new();
    for (h, t) in &containment_pairs {
        entity_set.insert(*h);
        entity_set.insert(*t);
    }
    let entity_names: Vec<&str> = {
        let mut v: Vec<&str> = entity_set.iter().copied().collect();
        v.sort();
        v
    };
    let n_entities = entity_names.len();

    println!("Entities: {}", n_entities);
    println!("Containment pairs: {}\n", containment_pairs.len());

    // --- Build taxonomy structure ---
    let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut parent_of: HashMap<&str, &str> = HashMap::new();
    for &(head, tail) in &containment_pairs {
        children_of.entry(head).or_default().push(tail);
        parent_of.insert(tail, head);
    }

    // Sibling index: position among siblings under the same parent.
    let mut sibling_idx: HashMap<&str, usize> = HashMap::new();
    for children in children_of.values() {
        for (i, child) in children.iter().enumerate() {
            sibling_idx.insert(child, i);
        }
    }

    // Depth for each entity (root = 0).
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

    // --- Initialize box embeddings ---
    //
    // Hierarchy-aware initialization: assign each entity a position based
    // on its branch in the taxonomy. Siblings get different offsets on a
    // dedicated "separation dimension" so they start non-overlapping.
    // Parents start large enough to cover their children's initial positions.
    let dim = 8;

    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();
    for &name in &entity_names {
        let d = depth[name];
        let half = match d {
            0 => 5.0, // root: very large
            1 => 3.0, // level 1: large
            2 => 1.5, // level 2: medium
            _ => 0.4, // leaves: small
        };
        // Build center: walk up the tree, accumulating sibling offsets.
        // Each level uses a different dimension for separation.
        let mut center = vec![0.0f32; dim];
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            let si = sibling_idx.get(cur).copied().unwrap_or(0);
            let sep_dim = depth[cur] % dim; // which dim to separate on
            center[sep_dim] += (si as f32) * 2.5;
            cur = p;
        }
        let min_arr = Array1::from_vec(center.iter().map(|c| c - half).collect());
        let max_arr = Array1::from_vec(center.iter().map(|c| c + half).collect());
        boxes.insert(name, (min_arr, max_arr));
    }

    // --- Build negative (non-containment) pairs ---
    //
    // Sibling pairs: children of the same parent should not contain each other.
    let mut negative_pairs: Vec<(&str, &str)> = Vec::new();
    for children in children_of.values() {
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                negative_pairs.push((children[i], children[j]));
                negative_pairs.push((children[j], children[i]));
            }
        }
    }

    let lr = 0.05;
    let neg_lr = 0.04;
    let shrink_lr = 0.002;
    let parent_shrink_lr = 0.03;
    let epochs = 300;

    // --- Training loop ---
    //
    // Three passes per epoch:
    //   1. Positive: expand head to contain tail (as before).
    //   2. Negative: for sibling/cross-branch pairs, push boxes apart so
    //      A does not fully contain B.
    //   3. Shrinkage: gently shrink leaf boxes toward their center so they
    //      don't all end up with identical volume.
    println!(
        "Training for {} epochs (dim={}, lr={}, neg_lr={})...\n",
        epochs, dim, lr, neg_lr
    );

    // Identify leaf entities (those that never appear as a head).
    let heads: std::collections::HashSet<&str> =
        containment_pairs.iter().map(|(h, _)| *h).collect();
    let leaves: Vec<&str> = entity_names
        .iter()
        .copied()
        .filter(|n| !heads.contains(n))
        .collect();

    for epoch in 0..epochs {
        let mut total_violation = 0.0f32;

        // Pass 1: positive containment -- expand head to contain tail.
        for &(head, tail) in &containment_pairs {
            let (tail_min, tail_max) = boxes[tail].clone();
            let (head_min, head_max) = boxes.get_mut(head).unwrap();

            for d in 0..dim {
                let margin = 0.05;
                if head_min[d] > tail_min[d] - margin {
                    let violation = head_min[d] - (tail_min[d] - margin);
                    head_min[d] -= lr * violation;
                    total_violation += violation.abs();
                }
                if head_max[d] < tail_max[d] + margin {
                    let violation = (tail_max[d] + margin) - head_max[d];
                    head_max[d] += lr * violation;
                    total_violation += violation.abs();
                }
            }
        }

        // Pass 2: parent regularization -- tighten each parent's boundaries
        // toward the actual extent of its children (plus margin). This
        // prevents parents from growing far beyond what they need, which
        // is the main cause of cross-branch contamination.
        for (parent, children) in &children_of {
            let mut child_min = vec![f32::MAX; dim];
            let mut child_max = vec![f32::MIN; dim];
            for &child in children {
                let (cmin, cmax) = &boxes[child];
                for d in 0..dim {
                    if cmin[d] < child_min[d] {
                        child_min[d] = cmin[d];
                    }
                    if cmax[d] > child_max[d] {
                        child_max[d] = cmax[d];
                    }
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

        // Pass 3: negative separation -- for sibling pairs (a, b) where a
        // should NOT contain b, push a's boundary inward on the single
        // dimension where b is closest to escaping. We shrink a (not shift
        // b) to avoid pushing b out of its actual parent.
        for &(a_name, b_name) in &negative_pairs {
            let (b_min_r, b_max_r) = boxes[b_name].clone();

            // Find dimension where a covers b with smallest gap.
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
                // Push the closer boundary of a past b to break coverage.
                if gap_min <= gap_max {
                    // Push a_min above b_min.
                    a_min[d] += neg_lr * (gap_min + 0.3);
                } else {
                    // Push a_max below b_max.
                    a_max[d] -= neg_lr * (gap_max + 0.3);
                }
                total_violation += best_gap;
            }
        }

        // Pass 4: shrink leaf boxes slightly toward their center.
        // This prevents all leaves from having identical volume and
        // encourages tighter, more specific representations.
        for &leaf in &leaves {
            let (leaf_min, leaf_max) = boxes.get_mut(leaf).unwrap();
            for d in 0..dim {
                let center = (leaf_min[d] + leaf_max[d]) * 0.5;
                leaf_min[d] += shrink_lr * (center - leaf_min[d]);
                leaf_max[d] -= shrink_lr * (leaf_max[d] - center);
            }
        }

        // Re-enforce min < max invariant.
        for (_name, (bmin, bmax)) in boxes.iter_mut() {
            for d in 0..dim {
                if bmin[d] >= bmax[d] {
                    let mid = (bmin[d] + bmax[d]) * 0.5;
                    bmin[d] = mid - 0.01;
                    bmax[d] = mid + 0.01;
                }
            }
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!(
                "  Epoch {:>4}: total_violation = {:.4}",
                epoch, total_violation
            );
        }
    }

    // Build NdarrayBox instances for evaluation
    let entity_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .map(|(&name, (min, max))| {
            let b = NdarrayBox::new(min.clone(), max.clone(), 1.0)
                .expect("box construction should succeed after training");
            (name, b)
        })
        .collect();

    // --- Evaluate learned boxes ---
    println!("\n--- Learned Box Volumes (larger = more general) ---\n");

    let mut vol_pairs: Vec<(&str, f32)> = entity_boxes
        .iter()
        .map(|(&name, b)| (name, b.volume(1.0).unwrap_or(0.0)))
        .collect();
    vol_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, vol) in &vol_pairs {
        println!("  {:>12}: volume = {:.6e}", name, vol);
    }

    // --- Containment checks ---
    println!("\n--- Containment Checks ---\n");

    let checks: Vec<(&str, &str, &str, bool)> = vec![
        // Positives (should have high containment probability)
        ("entity > animal", "entity", "animal", true),
        ("entity > vehicle", "entity", "vehicle", true),
        ("animal > mammal", "animal", "mammal", true),
        ("animal > bird", "animal", "bird", true),
        ("mammal > dog", "mammal", "dog", true),
        ("mammal > cat", "mammal", "cat", true),
        ("bird > eagle", "bird", "eagle", true),
        ("fish > salmon", "fish", "salmon", true),
        ("plant > tree", "plant", "tree", true),
        ("tree > oak", "tree", "oak", true),
        ("flower > rose", "flower", "rose", true),
        ("vehicle > car", "vehicle", "car", true),
        // Negatives (should have low containment probability)
        ("dog > animal (reverse)", "dog", "animal", false),
        ("cat > dog (sibling)", "cat", "dog", false),
        ("animal > vehicle (cross)", "animal", "vehicle", false),
    ];

    let mut correct = 0;
    let total = checks.len();
    for (label, head, tail, expect_high) in &checks {
        let hb = &entity_boxes[head];
        let tb = &entity_boxes[tail];
        let p = hb.containment_prob(tb, 1.0)?;
        let ok = if *expect_high { p > 0.5 } else { p < 0.5 };
        let status = if ok { "OK" } else { "FAIL" };
        println!("  [{:>4}] {:<30} P = {:.3}", status, label, p);
        if ok {
            correct += 1;
        }
    }

    println!(
        "\nHierarchy accuracy: {}/{} ({:.0}%)",
        correct,
        total,
        100.0 * correct as f32 / total as f32
    );

    println!("\nNotes:");
    println!("  - This uses direct coordinate updates, not backpropagation");
    println!("  - Negative separation pushes sibling/cross-branch boxes apart");
    println!("  - Leaf shrinkage produces varied volumes (more specific = smaller)");
    println!("  - Volume ordering (general > specific) emerges from containment constraints");

    // See scripts/plot_box_concept.py for a visualization of box containment geometry.

    Ok(())
}
