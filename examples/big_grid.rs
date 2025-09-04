use std::time::Instant;

use faster_poisson::{ParentalNbhdSampler, Poisson};

fn main() {
    let dims = [100.0, 100.0];
    let radius = 0.1;
    let attempts = 20;
    let cdf_exp = 1.0;
    println!(
        "Sampling points from a {}x{} grid with minimum distance {radius} (attempts={attempts}, cdf_exp={cdf_exp})...",
        dims[0], dims[1]
    );
    let poisson = Poisson::<2, ParentalNbhdSampler<2>>::new()
        .use_dims(dims)
        .use_radius(radius)
        .use_attempts(attempts)
        .use_cdf_exp(cdf_exp);
    let start = Instant::now();
    let samples = poisson.run();
    let elapsed = start.elapsed();
    println!("Generated {} points in {:?}.", samples.len(), elapsed);
}
