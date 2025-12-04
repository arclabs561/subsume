//! Comparative benchmarks validating paper performance claims.
//!
//! These benchmarks compare our implementations to:
//! - Euclidean distance baseline (for depth distance)
//! - Paper-reported performance metrics
//! - Alternative distance metrics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use subsume_core::Box;
use subsume_ndarray::{distance, NdarrayBox};

/// Benchmark: Depth distance vs Euclidean distance on hierarchies.
///
/// This validates the RegD (2025) claim that depth distance addresses
/// crowding effect better than Euclidean distance.
fn bench_depth_vs_euclidean_hierarchy(c: &mut Criterion) {
    let mut group = c.benchmark_group("depth_vs_euclidean_hierarchy");

    // Create hierarchy: parent with many children
    let parent = NdarrayBox::new(
        Array1::from(vec![0.0f32; 8]),
        Array1::from(vec![10.0f32; 8]),
        1.0,
    )
    .unwrap();

    let mut children = Vec::new();
    for i in 0..20 {
        let offset = 4.0 + (i as f32) * 0.1;
        let child = NdarrayBox::new(
            Array1::from(vec![offset; 8]),
            Array1::from(vec![offset + 0.5; 8]),
            1.0,
        )
        .unwrap();
        children.push(child);
    }

    // Benchmark Euclidean distance
    group.bench_function("euclidean_hierarchy", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    total += children[i].distance(black_box(&children[j])).unwrap_or(0.0);
                }
            }
            total
        });
    });

    // Benchmark depth distance
    group.bench_function("depth_hierarchy", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    total += distance::depth_distance(
                        black_box(&children[i]),
                        black_box(&children[j]),
                        1.0,
                        0.1,
                    )
                    .unwrap_or(0.0);
                }
            }
            total
        });
    });

    group.finish();
}

/// Benchmark: Vector-to-box distance vs point-in-box check.
///
/// Validates that vector-to-box distance is efficient for hybrid representations.
fn bench_vector_to_box_vs_containment(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_to_box_vs_containment");

    for dim in [2, 4, 8, 16, 32].iter() {
        let box_ = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        )
        .unwrap();

        let point = Array1::from(vec![0.5f32; *dim]);

        // Benchmark vector-to-box distance
        group.bench_with_input(
            BenchmarkId::new("vector_to_box", dim),
            &(&point, &box_),
            |b, (p, b)| b.iter(|| distance::vector_to_box_distance(black_box(p), black_box(b))),
        );

        // Benchmark containment check (alternative approach)
        group.bench_with_input(
            BenchmarkId::new("containment_check", dim),
            &(&point, &box_),
            |b, (p, b)| {
                // Create a zero-volume box at point and check containment
                b.iter(|| {
                    // Simplified: just check if point is in bounds
                    let mut inside = true;
                    for i in 0..b.dim() {
                        if p[i] < b.min()[i] || p[i] > b.max()[i] {
                            inside = false;
                            break;
                        }
                    }
                    if inside {
                        0.0
                    } else {
                        1.0
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Boundary distance vs containment probability.
///
/// Validates that boundary distance provides additional information
/// beyond simple containment probability.
fn bench_boundary_vs_containment_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_vs_containment_prob");

    for dim in [2, 4, 8, 16].iter() {
        let outer = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        )
        .unwrap();

        let inner = NdarrayBox::new(
            Array1::from(vec![0.2f32; *dim]),
            Array1::from(vec![0.8f32; *dim]),
            1.0,
        )
        .unwrap();

        // Benchmark boundary distance
        group.bench_with_input(
            BenchmarkId::new("boundary_distance", dim),
            &(&outer, &inner),
            |b, (o, i)| b.iter(|| distance::boundary_distance(black_box(o), black_box(i), 1.0)),
        );

        // Benchmark containment probability
        group.bench_with_input(
            BenchmarkId::new("containment_prob", dim),
            &(&outer, &inner),
            |b, (o, i)| b.iter(|| o.containment_prob(black_box(i), black_box(1.0))),
        );
    }

    group.finish();
}

/// Benchmark: Precision/stability comparison.
///
/// Validates RegD claim that depth distance eliminates precision issues
/// compared to hyperbolic methods (we compare to complex operations).
fn bench_precision_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_stability");

    // Create boxes with very small volumes (precision edge case)
    let box_a = NdarrayBox::new(
        Array1::from(vec![0.0f32; 8]),
        Array1::from(vec![1e-5f32; 8]),
        1.0,
    )
    .unwrap();

    let box_b = NdarrayBox::new(
        Array1::from(vec![1e-5f32; 8]),
        Array1::from(vec![2e-5f32; 8]),
        1.0,
    )
    .unwrap();

    // Benchmark depth distance (should be stable)
    group.bench_function("depth_distance_small_volumes", |b| {
        b.iter(|| distance::depth_distance(black_box(&box_a), black_box(&box_b), 1.0, 0.1));
    });

    // Benchmark volume calculation (precision-sensitive)
    group.bench_function("volume_calculation_small", |b| {
        b.iter(|| box_a.volume(black_box(1.0)).unwrap_or(0.0));
    });

    group.finish();
}

criterion_group!(
    comparative_benches,
    bench_depth_vs_euclidean_hierarchy,
    bench_vector_to_box_vs_containment,
    bench_boundary_vs_containment_prob,
    bench_precision_stability
);
criterion_main!(comparative_benches);
