//! Benchmarks for box embedding operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array1;
use subsume_core::{Box, GumbelBox, BoxEmbedding};
use subsume_ndarray::{NdarrayBox, NdarrayGumbelBox};

fn bench_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume");
    
    for dim in [2, 4, 8, 16, 32, 64].iter() {
        let min = Array1::from(vec![0.0f32; *dim]);
        let max = Array1::from(vec![1.0f32; *dim]);
        let box_ = NdarrayBox::new(min, max, 1.0).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &box_,
            |b, box_| {
                b.iter(|| box_.volume(black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let box_a = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            Array1::from(vec![0.5f32; *dim]),
            Array1::from(vec![1.5f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&box_a, &box_b),
            |bench, (a, other)| {
                bench.iter(|| a.intersection(black_box(other)))
            },
        );
    }
    
    group.finish();
}

fn bench_containment_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("containment_prob");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let premise = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let hypothesis = NdarrayBox::new(
            Array1::from(vec![0.2f32; *dim]),
            Array1::from(vec![0.8f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&premise, &hypothesis),
            |b, (p, h)| {
                b.iter(|| p.containment_prob(black_box(h), black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_overlap_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap_prob");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let box_a = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            Array1::from(vec![0.5f32; *dim]),
            Array1::from(vec![1.5f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&box_a, &box_b),
            |bench, (a, other)| {
                bench.iter(|| a.overlap_prob(black_box(other), black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_gumbel_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_sample");
    
    for dim in [2, 4, 8, 16, 32, 64].iter() {
        let gumbel_box = NdarrayGumbelBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &gumbel_box,
            |bench, box_| {
                bench.iter(|| {
                    let sample = box_.sample();
                    black_box(sample)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_gumbel_membership(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_membership");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let gumbel_box = NdarrayGumbelBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let point = Array1::from(vec![0.5f32; *dim]);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&gumbel_box, &point),
            |bench, (box_, pt)| {
                bench.iter(|| {
                    let prob = box_.membership_probability(pt).unwrap();
                    black_box(prob)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_containment_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("containment_matrix");
    
    for size in [5, 10, 20, 50, 100].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &collection,
            |b, coll| {
                b.iter(|| coll.containment_matrix(black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_containing_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("containing_boxes");
    
    for size in [10, 50, 100, 500].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        let query = NdarrayBox::new(
            Array1::from(vec![0.25f32; 3]),
            Array1::from(vec![0.75f32; 3]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&collection, &query),
            |b, (coll, q)| {
                b.iter(|| coll.containing_boxes(black_box(q), black_box(0.5), black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_contained_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("contained_boxes");
    
    for size in [10, 50, 100, 500].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        let query = NdarrayBox::new(
            Array1::from(vec![0.0f32; 3]),
            Array1::from(vec![1.0f32; 3]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&collection, &query),
            |b, (coll, q)| {
                b.iter(|| coll.contained_boxes(black_box(q), black_box(0.5), black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("union");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let box_a = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            Array1::from(vec![0.5f32; *dim]),
            Array1::from(vec![1.5f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&box_a, &box_b),
            |b, (a, other)| {
                b.iter(|| a.union(black_box(other)))
            },
        );
    }
    
    group.finish();
}

fn bench_center(c: &mut Criterion) {
    let mut group = c.benchmark_group("center");
    
    for dim in [2, 4, 8, 16, 32, 64, 128].iter() {
        let box_ = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &box_,
            |b, box_| {
                b.iter(|| box_.center())
            },
        );
    }
    
    group.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    
    for dim in [2, 4, 8, 16, 32].iter() {
        let box_a = NdarrayBox::new(
            Array1::from(vec![0.0f32; *dim]),
            Array1::from(vec![1.0f32; *dim]),
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            Array1::from(vec![2.0f32; *dim]),
            Array1::from(vec![3.0f32; *dim]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&box_a, &box_b),
            |b, (a, other)| {
                b.iter(|| a.distance(black_box(other)))
            },
        );
    }
    
    group.finish();
}

fn bench_overlap_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap_matrix");
    
    for size in [5, 10, 20, 50, 100].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &collection,
            |b, coll| {
                b.iter(|| coll.overlap_matrix(black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_overlapping_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlapping_boxes");
    
    for size in [10, 50, 100, 500].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        let query = NdarrayBox::new(
            Array1::from(vec![0.25f32; 3]),
            Array1::from(vec![0.75f32; 3]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&collection, &query),
            |b, (coll, q)| {
                b.iter(|| coll.overlapping_boxes(black_box(q), black_box(0.1), black_box(1.0)))
            },
        );
    }
    
    group.finish();
}

fn bench_nearest_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_boxes");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.1;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        let query = NdarrayBox::new(
            Array1::from(vec![0.25f32; 3]),
            Array1::from(vec![0.75f32; 3]),
            1.0,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&collection, &query),
            |b, (coll, q)| {
                b.iter(|| coll.nearest_boxes(black_box(q), black_box(10)))
            },
        );
    }
    
    group.finish();
}

fn bench_bounding_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounding_box");
    
    for size in [5, 10, 20, 50, 100, 500].iter() {
        let mut collection = subsume_core::BoxCollection::new();
        
        for i in 0..*size {
            let offset = (i as f32) * 0.01;
            let box_ = NdarrayBox::new(
                Array1::from(vec![offset; 3]),
                Array1::from(vec![offset + 0.5; 3]),
                1.0,
            ).unwrap();
            collection.push(box_);
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &collection,
            |b, coll| {
                b.iter(|| coll.bounding_box())
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_volume,
    bench_intersection,
    bench_containment_prob,
    bench_overlap_prob,
    bench_gumbel_sample,
    bench_gumbel_membership,
    bench_containment_matrix,
    bench_containing_boxes,
    bench_contained_boxes,
    bench_union,
    bench_center,
    bench_distance,
    bench_overlap_matrix,
    bench_overlapping_boxes,
    bench_nearest_boxes,
    bench_bounding_box
);
criterion_main!(benches);

