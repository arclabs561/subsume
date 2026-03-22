//! End-to-end box embedding training on a real-format knowledge graph dataset.
//!
//! Demonstrates the full pipeline using [`BoxEmbeddingTrainer`]:
//!   1. Load a dataset in WN18RR/FB15k-237 format
//!   2. Intern to integer IDs for efficient training
//!   3. Train box embeddings with `BoxEmbeddingTrainer::fit` (epochs, early stopping, LR warmup)
//!   4. Evaluate with standard link prediction metrics (MRR, Hits@1, Hits@10)
//!   5. Save and reload a training checkpoint via serde
//!
//! The dataset is a 60-triple subset of WordNet hypernym relations, embedded
//! inline so the example is self-contained. The format matches WN18RR:
//! tab-separated (head, relation, tail) triples in train/valid/test splits.
//!
//! References:
//! - Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
//! - Abboud et al. (2020), "BoxE: A Box Embedding Model for Knowledge Base Completion"
//! - Dettmers et al. (2018), "Convolutional 2D Knowledge Graph Embeddings" (WN18RR)
//!
//! Run: cargo run -p subsume --example dataset_training --release
//!
//! Related examples:
//! - `box_training`: hand-placed taxonomy with direct coordinate updates
//! - `el_training`: EL++ ontology embeddings with role composition
//! - `cone_training`: cone embeddings with angular containment

use std::io::Write;
use subsume::dataset::load_dataset;
use subsume::trainer::{BoxEmbeddingTrainer, FilteredTripleIndexIds};
use subsume::TrainingConfig;

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
    println!("=== Dataset-Driven Box Embedding Training (BoxEmbeddingTrainer) ===\n");

    // --- Step 1: Write dataset to temp files and load ---
    let dir = tempfile::tempdir()?;
    let mut train_file = std::fs::File::create(dir.path().join("train.txt"))?;
    let mut valid_file = std::fs::File::create(dir.path().join("valid.txt"))?;
    let mut test_file = std::fs::File::create(dir.path().join("test.txt"))?;
    write!(train_file, "{TRAIN_DATA}")?;
    write!(valid_file, "{VALID_DATA}")?;
    write!(test_file, "{TEST_DATA}")?;

    let dataset = load_dataset(dir.path())?;
    println!(
        "Dataset: {} entities, {} relations, {} train / {} valid / {} test triples",
        dataset.entities().len(),
        dataset.relations().len(),
        dataset.train.len(),
        dataset.valid.len(),
        dataset.test.len()
    );

    // --- Step 2: Intern to integer IDs ---
    let interned = dataset.into_interned();
    let train_triples: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();

    // Build filtered index for evaluation (excludes known-true triples).
    let filter = FilteredTripleIndexIds::from_dataset(&interned);

    // --- Step 3: Configure and create trainer ---
    let config = TrainingConfig {
        learning_rate: 0.02,
        epochs: 200,
        margin: 0.1,
        regularization: 0.0001,
        negative_weight: 1.0,
        early_stopping_patience: Some(30),
        warmup_epochs: 10,
        gumbel_beta: 10.0,
        gumbel_beta_final: 50.0,
        ..Default::default()
    };
    let dim = 12;
    let mut trainer = BoxEmbeddingTrainer::new(config, dim);

    println!(
        "\nTraining up to 200 epochs (dim={}, {} train triples, early stopping patience=30)...\n",
        dim,
        train_triples.len()
    );

    // --- Step 4: Train with fit() -- handles epochs, LR warmup, early stopping ---
    let result = trainer.fit(
        &train_triples,
        Some((&interned.valid, &interned.entities)),
        Some(&filter),
    )?;

    let actual_epochs = result.loss_history.len();
    println!(
        "  Trained {} epochs (best epoch: {})",
        actual_epochs, result.best_epoch
    );
    if let (Some(&first), Some(&last)) = (result.loss_history.first(), result.loss_history.last()) {
        println!("  Loss: {:.4} -> {:.4}", first, last);
    }
    if let (Some(&first), Some(&last)) = (
        result.validation_mrr_history.first(),
        result.validation_mrr_history.last(),
    ) {
        println!("  Val MRR: {:.4} -> {:.4}", first, last);
    }

    // --- Step 5: Evaluate on test set ---
    println!("\n--- Evaluation (test set, filtered) ---\n");

    let test_results = trainer.evaluate(&interned.test, &interned.entities, Some(&filter))?;

    println!("  MRR:       {:.4}", test_results.mrr);
    println!("  Hits@1:    {:.4}", test_results.hits_at_1);
    println!("  Hits@3:    {:.4}", test_results.hits_at_3);
    println!("  Hits@10:   {:.4}", test_results.hits_at_10);
    println!("  Mean Rank: {:.1}", test_results.mean_rank);
    println!(
        "\n  ({} test triples, {} entities, filtered ranking)",
        interned.test.len(),
        interned.entities.len()
    );

    // --- Step 6: Show per-relation breakdown ---
    if !test_results.per_relation.is_empty() {
        println!("\n--- Per-Relation Breakdown ---\n");
        for pr in &test_results.per_relation {
            let rel_name = interned
                .relations
                .get(pr.relation.parse::<usize>().unwrap_or(0))
                .unwrap_or(&pr.relation);
            println!(
                "  {}: MRR={:.4}, Hits@10={:.4} ({} triples)",
                rel_name, pr.mrr, pr.hits_at_10, pr.count
            );
        }
    }

    // --- Step 7: Demonstrate checkpoint save/load ---
    let checkpoint = serde_json::to_string(&trainer)?;
    println!(
        "\nCheckpoint: {} bytes (serialized {} entity boxes + optimizer state)",
        checkpoint.len(),
        trainer.boxes.len()
    );

    let restored: BoxEmbeddingTrainer = serde_json::from_str(&checkpoint)?;
    assert_eq!(restored.boxes.len(), trainer.boxes.len());
    println!(
        "Checkpoint round-trip: OK ({} entities restored)",
        restored.boxes.len()
    );

    Ok(())
}
