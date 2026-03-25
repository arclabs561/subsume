//! Train box embeddings on a WordNet subset and save the checkpoint.
//!
//! Produces `pretrained/wordnet_subset.json` -- a serialized [`BoxEmbeddingTrainer`]
//! containing entity and relation box embeddings trained on 47 hypernym triples.
//!
//! Run: cargo run -p subsume --example save_checkpoint --release

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
    println!("=== Training Box Embeddings on WordNet Subset ===\n");

    // --- Load dataset from inline triples ---
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

    let interned = dataset.into_interned();
    let train_triples: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();
    let filter = FilteredTripleIndexIds::from_dataset(&interned);

    // --- Configure and train ---
    let config = TrainingConfig {
        learning_rate: 0.02,
        epochs: 500,
        margin: 0.1,
        regularization: 0.0001,
        negative_weight: 1.0,
        early_stopping_patience: Some(50),
        warmup_epochs: 10,
        softplus_beta: 10.0,
        softplus_beta_final: 50.0,
        ..Default::default()
    };
    let dim = 16;
    let mut trainer = BoxEmbeddingTrainer::new(config, dim);

    println!(
        "\nTraining for up to 500 epochs (dim={}, {} train triples)...\n",
        dim,
        train_triples.len()
    );

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

    // --- Evaluate on test set (filtered) ---
    println!("\n--- Test Set Evaluation (filtered) ---\n");

    let test_results = trainer.evaluate(&interned.test, &interned.entities, Some(&filter))?;

    println!("  MRR:       {:.4}", test_results.mrr);
    println!("  Hits@1:    {:.4}", test_results.hits_at_1);
    println!("  Hits@3:    {:.4}", test_results.hits_at_3);
    println!("  Hits@10:   {:.4}", test_results.hits_at_10);
    println!("  Mean Rank: {:.1}", test_results.mean_rank);

    // --- Save checkpoint ---
    let checkpoint_path = std::path::Path::new("pretrained/wordnet_subset.json");
    std::fs::create_dir_all(checkpoint_path.parent().unwrap())?;

    let checkpoint = serde_json::to_string_pretty(&trainer)?;
    std::fs::write(checkpoint_path, &checkpoint)?;

    println!("\n--- Checkpoint ---\n");
    println!("  Saved to: {}", checkpoint_path.display());
    println!(
        "  Size: {} bytes ({:.1} KB)",
        checkpoint.len(),
        checkpoint.len() as f64 / 1024.0
    );
    println!("  Entities: {}", trainer.boxes.len());

    // --- Verify round-trip ---
    let reloaded: BoxEmbeddingTrainer = serde_json::from_str(&checkpoint)?;
    assert_eq!(reloaded.boxes.len(), trainer.boxes.len());
    println!("  Round-trip: OK");

    Ok(())
}
