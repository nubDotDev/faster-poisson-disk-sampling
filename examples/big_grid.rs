use faster_poisson::Poisson2D;
use std::time::Instant;

fn main() {
    let dims = [125.0, 125.0];
    let radius = 0.1;
    println!(
        "Sampling points from a(n) {} by {} grid with minimum distance {radius}...",
        dims[0], dims[1]
    );
    let poisson = Poisson2D::new()
        .dims(dims)
        .radius(radius)
        .seed(Some(0xDEADBEEF));
    let start = Instant::now();
    let samples = poisson.run();
    let elapsed = start.elapsed();
    println!("Generated {} points in {:?}.", samples.len(), elapsed);
}
