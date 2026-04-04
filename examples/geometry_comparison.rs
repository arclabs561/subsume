//! Geometry comparison harness on a WordNet hypernym subset.
//!
//! Trains all 6 new geometry types on the same WordNet hypernym dataset
//! (60 triples, 11 relations, ~30 entities) and reports MRR / Hits@10 / Mean Rank
//! side-by-side for comparison.
//!
//! This is a sanity-check / regression harness, not a full benchmark.
//! For full WN18RR benchmarks, see wn18rr_ball.rs and wn18rr_ball_burn.rs.
//!
//! Run: cargo run -p subsume --example geometry_comparison --release

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;
use subsume::dataset::{Triple, TripleIds, Vocab};
use subsume::trainer::CpuBoxTrainingConfig;

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
bear.n.01\t_hypernym\tcarnivore.n.01
whale.n.01\t_hypernym\tmammal.n.01
dolphin.n.01\t_hypernym\tplacental.n.01
eagle.n.01\t_hypernym\tbird.n.01
bird.n.01\t_hypernym\tvertebrate.n.01
snake.n.01\t_hypernym\treptile.n.01
reptile.n.01\t_hypernym\tvertebrate.n.01
frog.n.01\t_hypernym\tamphibian.n.01
amphibian.n.01\t_hypernym\tvertebrate.n.01
salmon.n.01\t_hypernym\tfish.n.01
fish.n.01\t_hypernym\tvertebrate.n.01
rose.n.01\t_hypernym\tflower.n.01
flower.n.01\t_hypernym\tplant.n.01";

fn parse_triples(data: &str) -> (Vec<Triple>, Vocab, Vocab) {
    let mut entity_vocab = Vocab::default();
    let mut relation_vocab = Vocab::default();
    let triples: Vec<Triple> = data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            let head = parts[0].to_string();
            let relation = parts[1].to_string();
            let tail = parts[2].to_string();
            entity_vocab.intern(head.clone());
            entity_vocab.intern(tail.clone());
            relation_vocab.intern(relation.clone());
            Triple {
                head,
                relation,
                tail,
            }
        })
        .collect();
    (triples, entity_vocab, relation_vocab)
}

fn build_maps(triples: &[Triple]) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut emap: HashMap<String, usize> = HashMap::new();
    let mut rmap: HashMap<String, usize> = HashMap::new();
    let mut eid = 0usize;
    let mut rid = 0usize;
    for t in triples {
        emap.entry(t.head.clone()).or_insert_with(|| {
            let id = eid;
            eid += 1;
            id
        });
        emap.entry(t.tail.clone()).or_insert_with(|| {
            let id = eid;
            eid += 1;
            id
        });
        rmap.entry(t.relation.clone()).or_insert_with(|| {
            let id = rid;
            rid += 1;
            id
        });
    }
    (emap, rmap)
}

fn build_test_ids(
    triples: &[Triple],
    emap: &HashMap<String, usize>,
    rmap: &HashMap<String, usize>,
) -> Vec<TripleIds> {
    triples
        .iter()
        .filter_map(|t| {
            Some(TripleIds {
                head: *emap.get(&t.head)?,
                relation: *rmap.get(&t.relation)?,
                tail: *emap.get(&t.tail)?,
            })
        })
        .collect()
}

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    mrr: f32,
    hits10: f32,
    mean_rank: f32,
    train_secs: f64,
}

fn run_ball(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::ball_trainer::BallTrainer;
    let mut trainer = BallTrainer::new(42);
    let (mut entities, mut relations) = trainer.init_embeddings(emap.len(), rmap.len(), 32);
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.05,
        margin: 1.0,
        negative_samples: 5,
        ..Default::default()
    };
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut entities, &mut relations, triples, &config, emap, rmap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&entities, &relations, test_ids, None);
    BenchResult {
        name: "Ball (SpherE+RegD)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn run_spherical_cap(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::spherical_cap_trainer::SphericalCapTrainer;
    let mut trainer = SphericalCapTrainer::new(42);
    let (mut entities, mut relations) = trainer.init_embeddings(emap.len(), rmap.len(), 32);
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.05,
        margin: 1.0,
        negative_samples: 5,
        ..Default::default()
    };
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut entities, &mut relations, triples, &config, emap, rmap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&entities, &relations, test_ids, None);
    BenchResult {
        name: "SphericalCap (novel)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn run_subspace(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    _rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::subspace_trainer::SubspaceTrainer;
    let mut trainer = SubspaceTrainer::new(42);
    let mut entities = trainer.init_embeddings(emap.len(), 8, 2);
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.01,
        margin: 1.0,
        negative_samples: 5,
        ..Default::default()
    };
    // Subspace trainer uses entity_to_idx only (no relations in scoring)
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut entities, triples, &config, emap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&entities, test_ids, None);
    BenchResult {
        name: "Subspace (logical ops)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn run_ellipsoid(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    _rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::ellipsoid_trainer::EllipsoidTrainer;
    let mut trainer = EllipsoidTrainer::new(42);
    let mut entities = trainer.init_embeddings(emap.len(), 4);
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.01,
        margin: 1.0,
        negative_samples: 5,
        ..Default::default()
    };
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut entities, triples, &config, emap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&entities, test_ids, None);
    BenchResult {
        name: "Ellipsoid (full-cov)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn run_transbox(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::transbox_trainer::TransBoxTrainer;
    let mut trainer = TransBoxTrainer::new(42);
    let (mut concepts, mut roles) = trainer.init_embeddings(emap.len(), rmap.len(), 32);
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.05,
        margin: 0.5,
        negative_samples: 5,
        ..Default::default()
    };
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut concepts, &mut roles, triples, &config, emap, rmap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&concepts, &roles, test_ids, None);
    BenchResult {
        name: "TransBox (EL++-closed)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn run_annular(
    triples: &[Triple],
    test_ids: &[TripleIds],
    emap: &HashMap<String, usize>,
    rmap: &HashMap<String, usize>,
) -> BenchResult {
    use subsume::trainer::annular_trainer::AnnularTrainer;
    let mut trainer = AnnularTrainer::new(42);
    let (mut entities, mut relations) = trainer.init_embeddings(emap.len(), rmap.len());
    let config = CpuBoxTrainingConfig {
        learning_rate: 0.05,
        margin: 1.0,
        negative_samples: 5,
        ..Default::default()
    };
    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = trainer.train_epoch(&mut entities, &mut relations, triples, &config, emap, rmap);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let results = trainer.evaluate(&entities, &relations, test_ids, None);
    BenchResult {
        name: "AnnularSector (Zhu 2025)",
        mrr: results.mrr,
        hits10: results.hits_at_10,
        mean_rank: results.mean_rank,
        train_secs: elapsed,
    }
}

fn main() {
    println!("=== Geometry Comparison: WordNet Hypernym Subset ===\n");
    println!("Dataset: 30 training triples, ~32 entities, 1 relation");
    println!("Training: 100 epochs, dim=32, lr=0.05, neg=5\n");

    let (triples, _, _) = parse_triples(TRAIN_DATA);
    let (emap, rmap) = build_maps(&triples);
    let test_ids = build_test_ids(&triples, &emap, &rmap);

    println!("Entities: {}, Relations: {}", emap.len(), rmap.len());
    println!("Test triples: {}\n", test_ids.len());
    println!(
        "{:<28} {:>8} {:>8} {:>10} {:>8}",
        "Geometry", "MRR", "H@10", "MeanRank", "Secs"
    );
    println!("{}", "-".repeat(68));

    let runners: Vec<Box<dyn Fn() -> BenchResult>> = vec![
        Box::new(|| run_ball(&triples, &test_ids, &emap, &rmap)),
        Box::new(|| run_spherical_cap(&triples, &test_ids, &emap, &rmap)),
        Box::new(|| run_subspace(&triples, &test_ids, &emap, &rmap)),
        Box::new(|| run_ellipsoid(&triples, &test_ids, &emap, &rmap)),
        Box::new(|| run_transbox(&triples, &test_ids, &emap, &rmap)),
        Box::new(|| run_annular(&triples, &test_ids, &emap, &rmap)),
    ];

    let mut all_results: Vec<BenchResult> = Vec::new();
    for runner in &runners {
        let result = runner();
        println!(
            "{:<28} {:>8.4} {:>8.4} {:>10.1} {:>8.2}",
            result.name, result.mrr, result.hits10, result.mean_rank, result.train_secs
        );
        std::io::stdout().flush().ok();
        all_results.push(result);
    }

    println!("{}", "-".repeat(68));

    // Find best
    let best = all_results
        .iter()
        .max_by(|a, b| a.mrr.partial_cmp(&b.mrr).unwrap())
        .unwrap();
    println!("\nBest MRR: {} ({:.4})", best.name, best.mrr);

    println!(
        "\nNote: Random baseline MRR ≈ {:.4} ({} entities)",
        1.0 / emap.len() as f32,
        emap.len()
    );
}
