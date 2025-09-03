use criterion::{Criterion, criterion_group, criterion_main};
use faster_poisson::{ParentalNbhdSampler, Poisson, StandardNbhdSampler};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

fn poisson2d_benchmark(c: &mut Criterion) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(0xDEADBEEF);
    c.bench_function("2D Standard", |b| {
        b.iter(|| {
            Poisson::<2, StandardNbhdSampler<2>>::new()
                .set_dims([5.0; 2])
                .set_seed(Some(rng.next_u64()))
                .run()
        })
    });
    c.bench_function("3D Standard", |b| {
        b.iter(|| {
            Poisson::<3, StandardNbhdSampler<3>>::new()
                .set_dims([5.0; 3])
                .set_seed(Some(rng.next_u64()))
                .run()
        })
    });
    c.bench_function("2D Parental", |b| {
        b.iter(|| {
            Poisson::<2, ParentalNbhdSampler<2>>::new()
                .set_dims([5.0; 2])
                .set_seed(Some(rng.next_u64()))
                .run()
        })
    });
}

criterion_group!(benches, poisson2d_benchmark);
criterion_main!(benches);
