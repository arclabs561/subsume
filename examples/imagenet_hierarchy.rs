//! Box embeddings on the Tiny ImageNet class hierarchy (WordNet synsets).
//!
//! Trains axis-aligned box embeddings on the WordNet hypernym graph underlying
//! Tiny ImageNet's 200 classes. Each class is a WordNet synset with a known
//! hypernym chain up to `entity`. This creates a real-world hierarchy with
//! 60 leaf classes and ~80 internal nodes (shared ancestors), totaling ~140
//! entities across animals, artifacts, food, and geological formations.
//!
//! The hierarchy is embedded inline as tab-separated (child, `_hypernym`,
//! parent) triples -- same format as `dataset_training`. Training uses
//! direct coordinate updates with four passes per epoch:
//!   1. Positive: expand parent boxes to contain children
//!   2. Parent regularization: tighten parents to children's extent
//!   3. Negative: break containment between siblings/unrelated pairs
//!   4. Leaf shrinkage: contract leaf boxes for volume differentiation
//!
//! Evaluation:
//! - Held-out hypernym triples scored by MRR and Hits@k
//! - Volume-depth correlation: deeper (more specific) concepts should have
//!   smaller boxes, yielding a negative Spearman rank correlation
//!
//! Reference: Deng et al. (2009), "ImageNet: A Large-Scale Hierarchical
//! Image Database"
//!
//! Run: cargo run -p subsume --example imagenet_hierarchy --release
//!
//! Related examples:
//! - `box_training`: smaller hand-placed taxonomy with same training approach
//! - `dataset_training`: training on real-format knowledge graph datasets
//! - `cone_training`: cone embeddings with angular containment

use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use subsume::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank};
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

/// Tiny ImageNet WordNet hypernym triples (60 leaf classes, ~140 total entities).
/// Format: child \t _hypernym \t parent.
///
/// Covers four domains: animals (vertebrates, invertebrates), artifacts
/// (vehicles, instruments, structures, clothing), food (fruit, vegetables,
/// dishes, beverages), and geological formations.
const TRAIN_DATA: &str = "\
goldfish\t_hypernym\tbony_fish
bony_fish\t_hypernym\tfish
fish\t_hypernym\tvertebrate
vertebrate\t_hypernym\tchordate
chordate\t_hypernym\tanimal
animal\t_hypernym\torganism
organism\t_hypernym\tentity
European_fire_salamander\t_hypernym\tsalamander
salamander\t_hypernym\tamphibian
amphibian\t_hypernym\tvertebrate
bullfrog\t_hypernym\tfrog
frog\t_hypernym\tamphibian
tailed_frog\t_hypernym\tfrog
American_alligator\t_hypernym\tcrocodilian
crocodilian\t_hypernym\treptile
reptile\t_hypernym\tvertebrate
boa_constrictor\t_hypernym\tsnake
snake\t_hypernym\treptile
trilobite\t_hypernym\tarthropod
arthropod\t_hypernym\tinvertebrate
invertebrate\t_hypernym\tanimal
scorpion\t_hypernym\tarachnid
arachnid\t_hypernym\tarthropod
black_widow\t_hypernym\tspider
spider\t_hypernym\tarachnid
goose\t_hypernym\taquatic_bird
aquatic_bird\t_hypernym\tbird
bird\t_hypernym\tvertebrate
koala\t_hypernym\tmarsupial
marsupial\t_hypernym\tmammal
mammal\t_hypernym\tvertebrate
jellyfish\t_hypernym\tcnidarian
cnidarian\t_hypernym\tinvertebrate
brain_coral\t_hypernym\tcoral
coral\t_hypernym\tcnidarian
slug\t_hypernym\tgastropod
gastropod\t_hypernym\tmollusk
mollusk\t_hypernym\tinvertebrate
sea_slug\t_hypernym\tgastropod
king_penguin\t_hypernym\tpenguin
penguin\t_hypernym\tseabird
seabird\t_hypernym\tbird
albatross\t_hypernym\tseabird
dugong\t_hypernym\tsea_cow
sea_cow\t_hypernym\taquatic_mammal
aquatic_mammal\t_hypernym\tmammal
Chihuahua\t_hypernym\ttoy_dog
toy_dog\t_hypernym\tdog
dog\t_hypernym\tcanine
canine\t_hypernym\tcarnivore
carnivore\t_hypernym\tplacental
placental\t_hypernym\tmammal
Yorkshire_terrier\t_hypernym\tterrier
terrier\t_hypernym\tdog
golden_retriever\t_hypernym\tretriever
retriever\t_hypernym\tsporting_dog
sporting_dog\t_hypernym\tdog
Labrador_retriever\t_hypernym\tretriever
standard_poodle\t_hypernym\tpoodle
poodle\t_hypernym\tdog
tabby_cat\t_hypernym\tdomestic_cat
domestic_cat\t_hypernym\tcat
cat\t_hypernym\tfeline
feline\t_hypernym\tcarnivore
Persian_cat\t_hypernym\tdomestic_cat
Egyptian_cat\t_hypernym\tdomestic_cat
cougar\t_hypernym\tbig_cat
big_cat\t_hypernym\tcat
lion\t_hypernym\tbig_cat
brown_bear\t_hypernym\tbear
bear\t_hypernym\tcarnivore
grasshopper\t_hypernym\tinsect
insect\t_hypernym\tarthropod
walking_stick\t_hypernym\tinsect
cockroach\t_hypernym\tinsect
mantis\t_hypernym\tinsect
dragonfly\t_hypernym\tinsect
monarch_butterfly\t_hypernym\tbutterfly
butterfly\t_hypernym\tinsect
sulphur_butterfly\t_hypernym\tbutterfly
sea_cucumber\t_hypernym\techinoderm
echinoderm\t_hypernym\tinvertebrate
guinea_pig\t_hypernym\trodent
rodent\t_hypernym\tplacental
pig\t_hypernym\tswine
swine\t_hypernym\teven_toed_ungulate
even_toed_ungulate\t_hypernym\tplacental
ox\t_hypernym\tbovid
bovid\t_hypernym\teven_toed_ungulate
bison\t_hypernym\tbovid
bighorn_sheep\t_hypernym\tbovid
gazelle\t_hypernym\tbovid
Arabian_camel\t_hypernym\tcamelid
camelid\t_hypernym\teven_toed_ungulate
orangutan\t_hypernym\tgreat_ape
great_ape\t_hypernym\tprimate
primate\t_hypernym\tplacental
chimpanzee\t_hypernym\tgreat_ape
baboon\t_hypernym\tOld_World_monkey
Old_World_monkey\t_hypernym\tprimate
African_elephant\t_hypernym\telephant
elephant\t_hypernym\tproboscidean
proboscidean\t_hypernym\tplacental
lesser_panda\t_hypernym\tprocyonid
procyonid\t_hypernym\tcarnivore
abacus\t_hypernym\tcalculator
calculator\t_hypernym\tmachine
machine\t_hypernym\tdevice
device\t_hypernym\tartifact
artifact\t_hypernym\tentity
backpack\t_hypernym\tbag
bag\t_hypernym\tcontainer
container\t_hypernym\tartifact
barn\t_hypernym\tfarm_building
farm_building\t_hypernym\tbuilding
building\t_hypernym\tstructure
structure\t_hypernym\tartifact
bikini\t_hypernym\tswimsuit
swimsuit\t_hypernym\tclothing
clothing\t_hypernym\tcovering
covering\t_hypernym\tartifact
cannon\t_hypernym\tartillery
artillery\t_hypernym\tweapon
weapon\t_hypernym\tinstrument
instrument\t_hypernym\tartifact
convertible\t_hypernym\tcar
car\t_hypernym\tmotor_vehicle
motor_vehicle\t_hypernym\tvehicle
vehicle\t_hypernym\tconveyance
conveyance\t_hypernym\tartifact
crane\t_hypernym\tlifting_device
lifting_device\t_hypernym\tdevice
dam\t_hypernym\tbarrier
barrier\t_hypernym\tstructure
dining_table\t_hypernym\ttable
table\t_hypernym\tfurniture
furniture\t_hypernym\tfurnishing
furnishing\t_hypernym\tartifact
flagpole\t_hypernym\tpole
pole\t_hypernym\tsupport
support\t_hypernym\tartifact
fountain\t_hypernym\tstructure
gasmask\t_hypernym\tmask
mask\t_hypernym\tprotective_covering
protective_covering\t_hypernym\tcovering
go_kart\t_hypernym\twheeled_vehicle
wheeled_vehicle\t_hypernym\tvehicle
gondola\t_hypernym\tboat
boat\t_hypernym\tvessel
vessel\t_hypernym\tconveyance
hourglass\t_hypernym\ttimepiece
timepiece\t_hypernym\tinstrument
kimono\t_hypernym\tgarment
garment\t_hypernym\tclothing
lampshade\t_hypernym\tshade
shade\t_hypernym\tprotective_covering
magnetic_compass\t_hypernym\tcompass
compass\t_hypernym\tinstrument
maypole\t_hypernym\tpole
miniskirt\t_hypernym\tskirt
skirt\t_hypernym\tgarment
moving_van\t_hypernym\tvan
van\t_hypernym\tmotor_vehicle
obelisk\t_hypernym\tpillar
pillar\t_hypernym\tstructure
parking_meter\t_hypernym\tmeter
meter\t_hypernym\tinstrument
police_van\t_hypernym\tvan
projectile\t_hypernym\tweapon
refrigerator\t_hypernym\twhite_goods
white_goods\t_hypernym\tappliance
appliance\t_hypernym\tdevice
rocking_chair\t_hypernym\tchair
chair\t_hypernym\tfurniture
school_bus\t_hypernym\tbus
bus\t_hypernym\tmotor_vehicle
sewing_machine\t_hypernym\tmachine
snorkel\t_hypernym\tbreathing_device
breathing_device\t_hypernym\tdevice
spider_web\t_hypernym\tweb
web\t_hypernym\tobject
object\t_hypernym\tentity
sports_car\t_hypernym\tcar
steel_arch_bridge\t_hypernym\tbridge
bridge\t_hypernym\tstructure
stopwatch\t_hypernym\twatch
watch\t_hypernym\ttimepiece
sunglasses\t_hypernym\tspectacles
spectacles\t_hypernym\toptical_instrument
optical_instrument\t_hypernym\tinstrument
suspension_bridge\t_hypernym\tbridge
swimming_trunks\t_hypernym\tswimsuit
teddy_bear\t_hypernym\tplaything
plaything\t_hypernym\tartifact
tractor\t_hypernym\tself_propelled_vehicle
self_propelled_vehicle\t_hypernym\tvehicle
triumphal_arch\t_hypernym\tarch
arch\t_hypernym\tstructure
trolleybus\t_hypernym\tbus
vestment\t_hypernym\tgarment
volleyball\t_hypernym\tball
ball\t_hypernym\tsports_equipment
sports_equipment\t_hypernym\tequipment
equipment\t_hypernym\tartifact
plate\t_hypernym\tdish
dish\t_hypernym\tcontainer
guacamole\t_hypernym\tdip
dip\t_hypernym\tcondiment
condiment\t_hypernym\tfood
food\t_hypernym\tsubstance
substance\t_hypernym\tentity
ice_cream\t_hypernym\tfrozen_dessert
frozen_dessert\t_hypernym\tdessert
dessert\t_hypernym\tfood
ice_lolly\t_hypernym\tfrozen_dessert
pretzel\t_hypernym\tbread
bread\t_hypernym\tbaked_goods
baked_goods\t_hypernym\tfood
mashed_potato\t_hypernym\tpotato
potato\t_hypernym\tvegetable
vegetable\t_hypernym\tfood
cauliflower\t_hypernym\tcrucifer
crucifer\t_hypernym\tvegetable
bell_pepper\t_hypernym\tpepper
pepper\t_hypernym\tvegetable
mushroom\t_hypernym\tfungus
fungus\t_hypernym\torganism
orange\t_hypernym\tcitrus
citrus\t_hypernym\tfruit
fruit\t_hypernym\tfood
lemon\t_hypernym\tcitrus
banana\t_hypernym\tfruit
pomegranate\t_hypernym\tfruit
meat_loaf\t_hypernym\tloaf
loaf\t_hypernym\tfood
pizza\t_hypernym\tdish
espresso\t_hypernym\tcoffee
coffee\t_hypernym\tbeverage
beverage\t_hypernym\tfood
punch\t_hypernym\tbeverage
alp\t_hypernym\tmountain
mountain\t_hypernym\tnatural_elevation
natural_elevation\t_hypernym\tgeological_formation
geological_formation\t_hypernym\tnatural_object
natural_object\t_hypernym\tentity
cliff\t_hypernym\tgeological_formation
coral_reef\t_hypernym\treef
reef\t_hypernym\tgeological_formation
lakeside\t_hypernym\tshore
shore\t_hypernym\tgeological_formation
seashore\t_hypernym\tshore";

/// Held-out triples: transitive hypernym relations (child is-a ancestor
/// skipping one or more intermediate nodes). Tests generalization beyond
/// direct edges.
const TEST_DATA: &str = "\
goldfish\t_hypernym\tvertebrate
bullfrog\t_hypernym\tvertebrate
American_alligator\t_hypernym\tvertebrate
scorpion\t_hypernym\tinvertebrate
black_widow\t_hypernym\tarthropod
goose\t_hypernym\tvertebrate
koala\t_hypernym\tvertebrate
jellyfish\t_hypernym\tanimal
slug\t_hypernym\tinvertebrate
Chihuahua\t_hypernym\tcarnivore
golden_retriever\t_hypernym\tdog
tabby_cat\t_hypernym\tcarnivore
lion\t_hypernym\tcarnivore
grasshopper\t_hypernym\tarthropod
monarch_butterfly\t_hypernym\tarthropod
guinea_pig\t_hypernym\tmammal
ox\t_hypernym\tplacental
orangutan\t_hypernym\tplacental
African_elephant\t_hypernym\tmammal
convertible\t_hypernym\tvehicle
school_bus\t_hypernym\tvehicle
stopwatch\t_hypernym\tinstrument
sunglasses\t_hypernym\tinstrument
barn\t_hypernym\tartifact
bikini\t_hypernym\tartifact
refrigerator\t_hypernym\tartifact
ice_cream\t_hypernym\tfood
pretzel\t_hypernym\tfood
orange\t_hypernym\tfood
espresso\t_hypernym\tfood
alp\t_hypernym\tgeological_formation
coral_reef\t_hypernym\tgeological_formation";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tiny ImageNet Hierarchy: Box Embedding Training ===\n");

    // --- Step 1: Parse triples into parent-child relationships ---
    let mut parent_of: HashMap<&str, &str> = HashMap::new();
    let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut entity_set: HashSet<&str> = HashSet::new();

    for line in TRAIN_DATA.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let (child, _rel, parent) = (parts[0], parts[1], parts[2]);
        parent_of.insert(child, parent);
        children_of.entry(parent).or_default().push(child);
        entity_set.insert(child);
        entity_set.insert(parent);
    }

    let mut entity_names: Vec<&str> = entity_set.iter().copied().collect();
    entity_names.sort();
    let n_entities = entity_names.len();

    // Containment pairs: (parent, child) -- parent should contain child.
    let containment_pairs: Vec<(&str, &str)> = parent_of
        .iter()
        .map(|(&child, &parent)| (parent, child))
        .collect();

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

    let max_depth = depth.values().copied().max().unwrap_or(0);

    // Sibling indices for spatial separation
    let mut sibling_idx: HashMap<&str, usize> = HashMap::new();
    for children in children_of.values() {
        for (i, child) in children.iter().enumerate() {
            sibling_idx.insert(child, i);
        }
    }

    // Identify leaves (never appear as a parent)
    let parents: HashSet<&str> = containment_pairs.iter().map(|(h, _)| *h).collect();
    let leaves: Vec<&str> = entity_names
        .iter()
        .copied()
        .filter(|n| !parents.contains(n))
        .collect();

    println!(
        "Hierarchy: {} entities ({} leaves, {} internal), depth {}, {} edges\n",
        n_entities,
        leaves.len(),
        n_entities - leaves.len(),
        max_depth,
        containment_pairs.len()
    );

    // --- Step 2: Initialize box embeddings (hierarchy-aware) ---
    let dim = 16;
    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();

    for &name in &entity_names {
        let d = depth[name];
        // Half-width shrinks with depth so parents start larger than children.
        let half = 6.0 * (0.55_f32).powi(d as i32);

        // Build center by walking up the tree; each level offsets on a
        // different dimension so siblings start spatially separated.
        let mut center = vec![0.0f32; dim];
        let mut cur = name;
        while let Some(&p) = parent_of.get(cur) {
            let si = sibling_idx.get(cur).copied().unwrap_or(0);
            let sep_dim = depth[cur] % dim;
            center[sep_dim] += (si as f32) * 2.0;
            cur = p;
        }

        let min_arr = Array1::from_vec(center.iter().map(|c| c - half).collect());
        let max_arr = Array1::from_vec(center.iter().map(|c| c + half).collect());
        boxes.insert(name, (min_arr, max_arr));
    }

    // --- Step 3: Build negative pairs (siblings) ---
    let mut negative_pairs: Vec<(&str, &str)> = Vec::new();
    for children in children_of.values() {
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                negative_pairs.push((children[i], children[j]));
                negative_pairs.push((children[j], children[i]));
            }
        }
    }

    // --- Step 4: Train ---
    let lr = 0.06;
    let neg_lr = 0.04;
    let shrink_lr = 0.003;
    let parent_shrink_lr = 0.025;
    let epochs = 400;

    println!(
        "Training {} epochs (dim={}, lr={}, {} containment, {} negative pairs)...\n",
        epochs,
        dim,
        lr,
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
        // Scale shrinkage by 1/(1+depth) so deeper leaves shrink less aggressively,
        // preventing collapse to zero width at max depth.
        for &leaf in &leaves {
            let d_depth = depth[leaf];
            let effective_lr = shrink_lr / (1.0 + d_depth as f32);
            let (lmin, lmax) = boxes.get_mut(leaf).unwrap();
            for d in 0..dim {
                let center = (lmin[d] + lmax[d]) * 0.5;
                lmin[d] += effective_lr * (center - lmin[d]);
                lmax[d] -= effective_lr * (lmax[d] - center);
            }
        }

        // Re-enforce min < max invariant
        for (_, (bmin, bmax)) in boxes.iter_mut() {
            for d in 0..dim {
                if bmin[d] >= bmax[d] {
                    let mid = (bmin[d] + bmax[d]) * 0.5;
                    bmin[d] = mid - 0.01;
                    bmax[d] = mid + 0.01;
                }
            }
        }

        if epoch % 80 == 0 || epoch == epochs - 1 {
            println!(
                "  epoch {:>3}: total_violation = {:.4}",
                epoch, total_violation
            );
        }
    }

    // --- Step 5: Build NdarrayBox instances ---
    let ndarray_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .filter_map(|(name, (min, max))| {
            NdarrayBox::new(min.clone(), max.clone(), 1.0)
                .ok()
                .map(|b| (*name, b))
        })
        .collect();

    // --- Step 6: Evaluate on held-out transitive hypernym triples ---
    println!("\n--- Evaluation (held-out transitive hypernym triples) ---\n");

    let mut ranks = Vec::new();
    for line in TEST_DATA.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let (child_name, _rel, true_parent) = (parts[0], parts[1], parts[2]);

        let Some(child_b) = ndarray_boxes.get(child_name) else {
            continue;
        };

        // Score all entities as potential parents by containment probability.
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
            .unwrap_or(n_entities);

        ranks.push(rank);
    }

    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let h1 = hits_at_k(ranks.iter().copied(), 1);
    let h3 = hits_at_k(ranks.iter().copied(), 3);
    let h10 = hits_at_k(ranks.iter().copied(), 10);
    let mr = mean_rank(ranks.iter().copied());

    println!("  MRR:       {mrr:.4}");
    println!("  Hits@1:    {h1:.4}");
    println!("  Hits@3:    {h3:.4}");
    println!("  Hits@10:   {h10:.4}");
    println!("  Mean Rank: {mr:.1}");
    println!(
        "  ({} test triples, {} candidate entities)",
        ranks.len(),
        n_entities
    );

    // --- Step 7: Volume-depth correlation ---
    //
    // Deeper (more specific) concepts should have smaller boxes. Compute
    // log-volume for each entity and report Spearman rank correlation with
    // depth.
    println!("\n--- Volume-Depth Correlation ---\n");

    let mut vol_depth: Vec<(&str, usize, f64)> = entity_names
        .iter()
        .filter_map(|&name| {
            let b = ndarray_boxes.get(name)?;
            let v = b.volume(1.0).unwrap_or(0.0) as f64;
            let lv = if v > 0.0 { v.ln() } else { f64::NEG_INFINITY };
            Some((name, depth[name], lv))
        })
        .collect();

    // Print a sample: show average log-volume per depth level.
    let mut depth_vols: HashMap<usize, Vec<f64>> = HashMap::new();
    for &(_name, d, lv) in &vol_depth {
        if lv.is_finite() {
            depth_vols.entry(d).or_default().push(lv);
        }
    }
    let mut depth_levels: Vec<usize> = depth_vols.keys().copied().collect();
    depth_levels.sort();

    println!("  {:>5}  {:>6}  {:>12}", "depth", "count", "avg_log_vol");
    println!("  {:->5}  {:->6}  {:->12}", "", "", "");
    for &d in &depth_levels {
        let vs = &depth_vols[&d];
        let avg: f64 = vs.iter().sum::<f64>() / vs.len() as f64;
        println!("  {:>5}  {:>6}  {:>12.2}", d, vs.len(), avg);
    }

    // Spearman rank correlation between depth and log-volume.
    let valid: Vec<(f64, f64)> = vol_depth
        .iter()
        .filter(|(_, _, lv)| lv.is_finite())
        .map(|(_, d, lv)| (*d as f64, *lv))
        .collect();
    let spearman = spearman_correlation(&valid);
    println!(
        "\n  Spearman(depth, log_volume) = {:.4}  (n={})",
        spearman,
        valid.len()
    );
    if spearman < -0.3 {
        println!("  -> Negative correlation confirms: specific concepts have smaller boxes.");
    } else {
        println!("  -> Weak/positive correlation: volume ordering did not fully emerge.");
    }

    // Show top-5 largest and smallest volume entities for sanity check.
    println!("\n--- Volume Extremes ---\n");
    vol_depth.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Largest boxes (most general):");
    for &(name, d, lv) in vol_depth.iter().take(5) {
        println!("    {:<28} depth={} log_vol={:.2}", name, d, lv);
    }
    println!("  Smallest boxes (most specific):");
    for &(name, d, lv) in vol_depth.iter().rev().take(5) {
        println!("    {:<28} depth={} log_vol={:.2}", name, d, lv);
    }

    println!("\n  {} total entities trained", n_entities);

    Ok(())
}

/// Spearman rank correlation between paired observations.
/// Returns a value in [-1, 1]. Negative means inverse relationship.
fn spearman_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 3 {
        return 0.0;
    }

    // Rank each variable independently (average ranks for ties).
    let rank_x = compute_ranks(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let rank_y = compute_ranks(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());

    // Pearson correlation on ranks.
    let n_f = n as f64;
    let mean_rx: f64 = rank_x.iter().sum::<f64>() / n_f;
    let mean_ry: f64 = rank_y.iter().sum::<f64>() / n_f;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = rank_x[i] - mean_rx;
        let dy = rank_y[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Assign ranks to values (1-based). Ties get average rank.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        // Average rank for tied group [i, j)
        let avg_rank = (i + j + 1) as f64 / 2.0; // +1 for 1-based
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}
