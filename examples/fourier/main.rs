use faster_poisson::{
    Poisson2D, PoissonBridson2D, PoissonDart2D, PoissonRegular2D, fourier, plot_2d,
};

fn main() {
    let _ = std::fs::create_dir_all("examples/fourier/images");
    let dims = [10.0; 2];

    let poisson = Poisson2D::new()
        .use_dims(dims)
        .use_radius(0.1)
        .use_seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        100,
        3.0,
        String::from("examples/fourier/images/parental_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/parental_plot.html"),
    );
    println!("Calculated Fourier transform for parental sampler.");

    let poisson = PoissonBridson2D::new()
        .use_dims(dims)
        .use_radius(0.1)
        .use_seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        100,
        3.0,
        String::from("examples/fourier/images/bridson_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/bridson_plot.html"),
    );
    println!("Calculated Fourier transform for Bridson sampler.");

    let poisson = PoissonDart2D::new()
        .use_dims(dims)
        .use_radius(0.1)
        .use_seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        100,
        3.0,
        String::from("examples/fourier/images/dart_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/dart_plot.html"),
    );
    println!("Calculated Fourier transform for dart sampler.");

    let poisson = PoissonRegular2D::new().use_dims(dims).use_radius(0.1);
    let samples = poisson.run();
    fourier(
        &samples,
        dims,
        100,
        3.0,
        String::from("examples/fourier/images/regular_fourier.png"),
    );
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/fourier/images/regular_plot.html"),
    );
    println!("Calculated Fourier transform for regular sampler.");
}
