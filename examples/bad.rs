use faster_poisson::PoissonRegular2D;

fn main() {
    let poisson = PoissonRegular2D::new();
    let samples = poisson.run();
    println!("Regular samples:");
    println!("{:?}", samples);
}
