use criterion::{black_box, criterion_group, criterion_main, Criterion};
use subsume::el;

fn make_vecs(dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let center_a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
    let offset_a: Vec<f32> = vec![0.5; dim];
    let center_b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let offset_b: Vec<f32> = vec![1.0; dim];
    (center_a, offset_a, center_b, offset_b)
}

fn bench_inclusion_loss(c: &mut Criterion) {
    for dim in [8, 64, 256] {
        let (ca, oa, cb, ob) = make_vecs(dim);
        c.bench_function(&format!("el_inclusion_loss/dim={dim}"), |b| {
            b.iter(|| {
                el::el_inclusion_loss(
                    black_box(&ca),
                    black_box(&oa),
                    black_box(&cb),
                    black_box(&ob),
                    black_box(0.1),
                )
            })
        });
    }
}

fn bench_compose_roles(c: &mut Criterion) {
    let dim = 64;
    let (cr, or, cs, os) = make_vecs(dim);
    let mut c_out = vec![0.0f32; dim];
    let mut o_out = vec![0.0f32; dim];
    c.bench_function("compose_roles/dim=64", |b| {
        b.iter(|| {
            el::compose_roles(
                black_box(&cr),
                black_box(&or),
                black_box(&cs),
                black_box(&os),
                black_box(&mut c_out),
                black_box(&mut o_out),
            )
        })
    });
}

fn bench_existential_box(c: &mut Criterion) {
    let dim = 64;
    let (rc, ro, fc, fo) = make_vecs(dim);
    let mut c_out = vec![0.0f32; dim];
    let mut o_out = vec![0.0f32; dim];
    c.bench_function("existential_box/dim=64", |b| {
        b.iter(|| {
            el::existential_box(
                black_box(&rc),
                black_box(&ro),
                black_box(&fc),
                black_box(&fo),
                black_box(&mut c_out),
                black_box(&mut o_out),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_inclusion_loss,
    bench_compose_roles,
    bench_existential_box
);
criterion_main!(benches);
