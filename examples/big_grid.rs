use faster_poisson::Poisson2D;
use std::time::Instant;

fn main() {
    let dims = [100.0, 100.0];
    let radius = 0.1;
    println!(
        "Sampling points from a(n) {} by {} grid with minimum distance {radius}...",
        dims[0], dims[1]
    );
    let poisson = Poisson2D::new()
        .use_dims(dims)
        .use_radius(radius)
        .use_seed(Some(0xDEADBEEF));
    let start = Instant::now();
    let samples = poisson.run();
    let elapsed = start.elapsed();
    println!("Generated {} points in {:?}.", samples.len(), elapsed);
}
