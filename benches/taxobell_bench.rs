#![allow(missing_docs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use subsume::gaussian::GaussianBox;
use subsume::taxobell::{TaxoBellConfig, TaxoBellLoss};

fn make_box(dim: usize, offset: f32) -> GaussianBox {
    let mu: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + offset).collect();
    let sigma: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.05).collect();
    GaussianBox::new(mu, sigma).unwrap()
}

fn bench_symmetric_loss(c: &mut Criterion) {
    let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
    let mut group = c.benchmark_group("symmetric_loss");

    for dim in [8, 64] {
        let anchor = make_box(dim, 0.0);
        let positive = make_box(dim, 0.1);
        let negative = make_box(dim, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| {
                loss_fn.symmetric_loss(
                    black_box(&anchor),
                    black_box(&positive),
                    black_box(&negative),
                )
            });
        });
    }
    group.finish();
}

fn bench_combined_loss(c: &mut Criterion) {
    let config = TaxoBellConfig::default();
    let loss_fn = TaxoBellLoss::new(config);

    let dim = 8;
    let n_pos = 100;
    let n_neg = 100;
    let n_boxes = 50;

    // Pre-create boxes.
    let boxes: Vec<GaussianBox> = (0..n_boxes)
        .map(|i| make_box(dim, i as f32 * 0.3))
        .collect();

    // Build positive pairs: (child, parent) cycling through boxes.
    let positives: Vec<(&GaussianBox, &GaussianBox)> = (0..n_pos)
        .map(|i| (&boxes[i % n_boxes], &boxes[(i + 1) % n_boxes]))
        .collect();

    // Build negative triples: (anchor, positive, negative) cycling.
    let negatives: Vec<(&GaussianBox, &GaussianBox, &GaussianBox)> = (0..n_neg)
        .map(|i| {
            (
                &boxes[i % n_boxes],
                &boxes[(i + 1) % n_boxes],
                &boxes[(i + 2) % n_boxes],
            )
        })
        .collect();

    let all_refs: Vec<&GaussianBox> = boxes.iter().collect();

    c.bench_function("combined_loss/100pos_100neg_50boxes_dim8", |bench| {
        bench.iter(|| {
            loss_fn.combined_loss(
                black_box(&positives),
                black_box(&negatives),
                black_box(&all_refs),
            )
        });
    });
}

criterion_group!(benches, bench_symmetric_loss, bench_combined_loss);
criterion_main!(benches);
