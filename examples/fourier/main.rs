use faster_poisson::{
    Poisson2D, PoissonBridson2D, PoissonDart2D, PoissonND, PoissonNaiveND, PoissonRegular2D,
    fourier, plot_2d,
};

fn main() {
    let _ = std::fs::create_dir_all("examples/fourier/images");
    let dims = [10.0; 2];
    let pixels_per_unit = 100;
    let brightness = 3.0;

    let poisson = Poisson2D::new().dims(dims).seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/parental_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/parental_plot.html"),
    );
    println!(
        "Calculated Fourier transform for parental sampler ({} samples).",
        samples.len()
    );

    let poisson = PoissonNaiveND::new()
        .dims(dims)
        .attempts(1000)
        .seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/naive_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/naive_plot.html"),
    );
    println!(
        "Calculated Fourier transform for naive sampler ({} samples).",
        samples.len()
    );

    let poisson = PoissonBridson2D::new().dims(dims).seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/bridson_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/bridson_plot.html"),
    );
    println!(
        "Calculated Fourier transform for Bridson sampler ({} samples).",
        samples.len()
    );

    let poisson = PoissonDart2D::new().dims(dims).seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/dart_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/dart_plot.html"),
    );
    println!(
        "Calculated Fourier transform for dart sampler ({} samples).",
        samples.len()
    );

    let poisson = PoissonRegular2D::new().dims(dims);
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/regular_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/regular_plot.html"),
    );
    println!(
        "Calculated Fourier transform for regular sampler ({} samples).",
        samples.len()
    );

    let poisson = PoissonND::<2>::new()
        .dims(dims)
        .radius_fn(Some(|p| {
            0.1 + 0.1 * p[0].sin().abs() + 0.1 * p[1].sin().abs()
        }))
        .seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        pixels_per_unit,
        brightness,
        String::from("examples/fourier/images/dynamic_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/dynamic_plot.html"),
    );
    println!(
        "Calculated Fourier transform for dynamic radius sampler ({} samples).",
        samples.len()
    );
}
