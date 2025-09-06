use criterion::{Criterion, criterion_group, criterion_main};
use faster_poisson::{Poisson2D, Poisson3D, PoissonBridson2D, PoissonND};

fn poisson2d_benchmark(c: &mut Criterion) {
    c.bench_function("2D Parental (5 x 5)", |b| {
        b.iter(|| {
            Poisson2D::new()
                .use_dims([5.0; 2])
                .use_seed(Some(0xDEADBEEF))
                .run()
        })
    });
    c.bench_function("2D Bridson (5 x 5)", |b| {
        b.iter(|| {
            PoissonBridson2D::new()
                .use_dims([5.0; 2])
                .use_seed(Some(0xDEADBEEF))
                .run()
        })
    });
    c.bench_function("3D (2 x 2 x 2)", |b| {
        b.iter(|| {
            Poisson3D::new()
                .use_dims([2.0; 3])
                .use_seed(Some(0xDEADBEEF))
                .run()
        })
    });
    c.bench_function("4D (0.5 x 0.5 x 0.5 x 0.5)", |b| {
        b.iter(|| {
            PoissonND::<4>::new()
                .use_dims([0.5; 4])
                .use_seed(Some(0xDEADBEEF))
                .run()
        })
    });
}

criterion_group!(benches, poisson2d_benchmark);
criterion_main!(benches);
