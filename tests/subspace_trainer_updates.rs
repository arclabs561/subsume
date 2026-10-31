//! Regression test: SubspaceTrainer must update entity bases.
use std::collections::HashMap;
use subsume::dataset::Triple;
use subsume::trainer::{CpuBoxTrainingConfig, SubspaceTrainer};

fn triple(h: &str, r: &str, t: &str) -> Triple {
    Triple {
        head: h.to_string(),
        relation: r.to_string(),
        tail: t.to_string(),
    }
}

/// subspace_trainer.rs:283-306 applies Adam to `&mut entities[i].basis_mut()[j][i]`,
/// but Subspace::basis_mut (subspace.rs:202-214) returns a *copy* ("After
/// modification, use set_basis"), so every update is written into a temporary
/// and dropped: 50 epochs leave every entity bit-identical while the reported
/// loss is positive.
#[test]
fn subspace_trainer_updates_entities() {
    let mut tr = SubspaceTrainer::new(42);
    let mut ents = tr.init_embeddings(4, 4, 2);
    let before: Vec<Vec<Vec<f32>>> = ents
        .iter()
        .map(|s| (0..s.rank()).map(|j| s.basis_vector(j)).collect())
        .collect();
    let triples = vec![
        triple("e0", "r0", "e1"),
        triple("e2", "r0", "e3"),
        triple("e0", "r1", "e2"),
    ];
    let map: HashMap<String, usize> = (0..4).map(|i| (format!("e{i}"), i)).collect();
    let cfg = CpuBoxTrainingConfig {
        learning_rate: 0.05,
        margin: 1.0,
        ..Default::default()
    };
    let mut last = 0.0;
    for _ in 0..50 {
        last = tr.train_epoch(&mut ents, &triples, &cfg, &map);
    }
    let after: Vec<Vec<Vec<f32>>> = ents
        .iter()
        .map(|s| (0..s.rank()).map(|j| s.basis_vector(j)).collect())
        .collect();
    assert!(last > 0.0, "loss {last}");
    assert_ne!(
        before, after,
        "no parameter changed in 50 epochs (final loss {last})"
    );
}
