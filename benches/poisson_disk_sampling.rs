use criterion::{Criterion, criterion_group, criterion_main};
use faster_poisson::{Poisson2D, Poisson3D, PoissonBridson2D, PoissonDart2D, PoissonND};

fn poisson2d_benchmark(c: &mut Criterion) {
    c.bench_function("2D Parental (5 x 5)", |b| {
        b.iter(|| Poisson2D::new().dims([5.0; 2]).seed(Some(0xDEADBEEF)).run())
    });
    c.bench_function("2D Bridson (5 x 5)", |b| {
        b.iter(|| {
            PoissonBridson2D::new()
                .dims([5.0; 2])
                .seed(Some(0xDEADBEEF))
                .run()
        })
    });
    c.bench_function("2D Dart (5 x 5)", |b| {
        b.iter(|| {
            PoissonDart2D::new()
                .dims([5.0; 2])
                .seed(Some(0xDEADBEEF))
                .run()
        })
    });
    c.bench_function("3D (2 x 2 x 2)", |b| {
        b.iter(|| Poisson3D::new().dims([2.0; 3]).seed(Some(0xDEADBEEF)).run())
    });
    c.bench_function("4D (0.5 x 0.5 x 0.5 x 0.5)", |b| {
        b.iter(|| {
            PoissonND::<4>::new()
                .dims([0.5; 4])
                .seed(Some(0xDEADBEEF))
                .run()
        })
    });
}

criterion_group!(benches, poisson2d_benchmark);
criterion_main!(benches);
