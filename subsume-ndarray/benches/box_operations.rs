//! Benchmarks for box embedding operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array1;
use subsume_core::{Box, GumbelBox};
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

criterion_group!(
    benches,
    bench_volume,
    bench_intersection,
    bench_containment_prob,
    bench_overlap_prob,
    bench_gumbel_sample,
    bench_gumbel_membership
);
criterion_main!(benches);

