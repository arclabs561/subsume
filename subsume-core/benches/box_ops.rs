//! Benchmarks for box embedding operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use subsume_core::{center_offset_to_min_max, min_max_to_center_offset};

fn bench_coordinate_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_transforms");

    for dim in [32, 64, 128, 256] {
        // Create sample min/max coordinates
        let min: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let max: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.01).collect();
        let mut center_out = vec![0.0f32; dim];
        let mut offset_out = vec![0.0f32; dim];

        group.bench_with_input(
            BenchmarkId::new("min_max_to_center_offset", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    min_max_to_center_offset(
                        black_box(&min),
                        black_box(&max),
                        &mut center_out,
                        &mut offset_out,
                    )
                })
            },
        );

        // Create sample center/offset coordinates
        let center: Vec<f32> = (0..dim).map(|i| 0.25 + i as f32 * 0.01).collect();
        let offset: Vec<f32> = (0..dim).map(|_| 0.25).collect();
        let mut min_out = vec![0.0f32; dim];
        let mut max_out = vec![0.0f32; dim];

        group.bench_with_input(
            BenchmarkId::new("center_offset_to_min_max", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    center_offset_to_min_max(
                        black_box(&center),
                        black_box(&offset),
                        &mut min_out,
                        &mut max_out,
                    )
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "rand")]
fn bench_negative_sampling(c: &mut Criterion) {
    use rand::SeedableRng;
    use subsume_core::trainer::{
        generate_negative_samples_from_sorted_pool_with_rng, generate_negative_samples_with_rng,
        NegativeSamplingStrategy, SortedEntityPool,
    };
    use subsume_core::Triple;
    use std::collections::HashSet;

    let mut group = c.benchmark_group("negative_sampling");

    // A moderately large entity universe to make pool construction visible.
    let n_entities = 50_000usize;
    let entities: HashSet<String> = (0..n_entities).map(|i| format!("e{i:06}")).collect();
    let triple = Triple {
        head: "e000001".to_string(),
        relation: "r".to_string(),
        tail: "e000002".to_string(),
    };
    let strategy = NegativeSamplingStrategy::Uniform;

    // Build once for the "reuse pool" variant.
    let pool = SortedEntityPool::new(&entities);

    for n in [1usize, 4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("build_pool_each_call", n), &n, |b, &n| {
            b.iter(|| {
                // Deterministic RNG per-iteration to keep work comparable.
                let mut rng = rand::rngs::StdRng::seed_from_u64(123);
                let out = generate_negative_samples_with_rng(
                    black_box(&triple),
                    black_box(&entities),
                    black_box(&strategy),
                    black_box(n),
                    &mut rng,
                );
                black_box(out.len())
            })
        });

        group.bench_with_input(BenchmarkId::new("reuse_sorted_pool", n), &n, |b, &n| {
            b.iter(|| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(123);
                let out = generate_negative_samples_from_sorted_pool_with_rng(
                    black_box(&triple),
                    black_box(&pool),
                    black_box(&strategy),
                    black_box(n),
                    &mut rng,
                );
                black_box(out.len())
            })
        });
    }

    group.finish();
}

#[cfg(not(feature = "rand"))]
fn bench_negative_sampling(_: &mut Criterion) {}

fn bench_link_prediction_ranking(c: &mut Criterion) {
    let mut group = c.benchmark_group("link_prediction_ranking");

    // Benchmark the ranking algorithm in isolation (score list → rank),
    // since `subsume-core` is backend-agnostic and doesn't ship a concrete `Box` type.
    for n in [1_000usize, 10_000, 100_000] {
        let tail_idx = n / 2;

        // Deterministic score pattern with a few ties.
        let scores: Vec<f32> = (0..n)
            .map(|i| ((i % 997) as f32) / 997.0)
            .collect();
        let tail_score = scores[tail_idx];

        group.bench_with_input(BenchmarkId::new("sort", n), &n, |b, &_n| {
            b.iter(|| {
                let mut items: Vec<(usize, f32)> = (0..n).map(|i| (i, scores[i])).collect();
                items.sort_by(|a, b| {
                    // Desc score, then asc id for determinism.
                    b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0))
                });
                let rank = items
                    .iter()
                    .position(|(i, _)| *i == tail_idx)
                    .map(|pos| pos + 1)
                    .unwrap_or(usize::MAX);
                black_box(rank)
            })
        });

        group.bench_with_input(BenchmarkId::new("linear", n), &n, |b, &_n| {
            b.iter(|| {
                let mut better = 0usize;
                let mut tie_before = 0usize;
                for i in 0..n {
                    if i == tail_idx {
                        continue;
                    }
                    let s = scores[i];
                    if s > tail_score {
                        better += 1;
                    } else if s == tail_score && i < tail_idx {
                        tie_before += 1;
                    }
                }
                black_box(better + tie_before + 1)
            })
        });
    }

    group.finish();
}

fn bench_trainer_kernels(c: &mut Criterion) {
    use subsume_core::trainer::compute_pair_loss;
    use subsume_core::{AMSGradState, TrainableBox};

    let mut group = c.benchmark_group("trainer_kernels");

    for dim in [32usize, 64, 128, 256] {
        let mu_a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        let mu_b: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.001).collect();
        let delta_a: Vec<f32> = vec![0.1f32.ln(); dim];
        let delta_b: Vec<f32> = vec![0.1f32.ln(); dim];
        let box_a = TrainableBox::new(mu_a, delta_a);
        let box_b = TrainableBox::new(mu_b, delta_b);
        let cfg = subsume_core::TrainingConfig::default();

        group.bench_with_input(BenchmarkId::new("compute_pair_loss_pos", dim), &dim, |b, _| {
            b.iter(|| compute_pair_loss(black_box(&box_a), black_box(&box_b), true, black_box(&cfg)))
        });

        group.bench_with_input(BenchmarkId::new("compute_pair_loss_neg", dim), &dim, |b, _| {
            b.iter(|| compute_pair_loss(black_box(&box_a), black_box(&box_b), false, black_box(&cfg)))
        });

        let mut box_u = box_a.clone();
        let mut state = AMSGradState::new(dim, 1e-3);
        let grad_mu: Vec<f32> = vec![0.01; dim];
        let grad_delta: Vec<f32> = vec![0.01; dim];

        group.bench_with_input(BenchmarkId::new("update_amsgrad", dim), &dim, |b, _| {
            b.iter(|| {
                box_u.update_amsgrad(black_box(&grad_mu), black_box(&grad_delta), black_box(&mut state));
                black_box(&box_u);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_coordinate_transforms,
    bench_negative_sampling,
    bench_link_prediction_ranking,
    bench_trainer_kernels
);
criterion_main!(benches);
