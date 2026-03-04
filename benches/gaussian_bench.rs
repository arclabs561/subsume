use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use subsume::gaussian::{bhattacharyya_coefficient, kl_divergence, GaussianBox};

fn make_pair(dim: usize) -> (GaussianBox, GaussianBox) {
    let a = GaussianBox::new(vec![0.5; dim], vec![1.0; dim]).unwrap();
    let b = GaussianBox::new(vec![-0.5; dim], vec![2.0; dim]).unwrap();
    (a, b)
}

fn bench_kl_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_divergence");
    for dim in [8, 64, 256] {
        let (a, b) = make_pair(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| kl_divergence(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_bhattacharyya_coefficient(c: &mut Criterion) {
    let mut group = c.benchmark_group("bhattacharyya_coefficient");
    for dim in [8, 64, 256] {
        let (a, b) = make_pair(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| bhattacharyya_coefficient(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_from_center_offset(c: &mut Criterion) {
    let center = vec![1.0f32; 64];
    let offset = vec![0.5f32; 64];
    c.bench_function("from_center_offset/64", |bench| {
        bench.iter(|| {
            GaussianBox::from_center_offset(black_box(center.clone()), black_box(offset.clone()))
        });
    });
}

criterion_group!(
    benches,
    bench_kl_divergence,
    bench_bhattacharyya_coefficient,
    bench_from_center_offset
);
criterion_main!(benches);
