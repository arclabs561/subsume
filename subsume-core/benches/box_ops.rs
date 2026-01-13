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

criterion_group!(benches, bench_coordinate_transforms);
criterion_main!(benches);
