use faster_poisson::{Poisson2D, Poisson3D, plot_2d, plot_3d};

fn main() {
    let _ = std::fs::create_dir_all("examples/plot/images");

    let poisson = Poisson2D::new()
        .dims([5.0; 2])
        .seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    plot_2d(
        &samples,
        [1000, 1000],
        String::from("examples/plot/images/plot_2d.html"),
    );
    println!(
        "Plotted {} samples in examples/plot/images/plot_2d.html",
        samples.len()
    );

    let poisson = Poisson3D::new().seed(Some(0xDEADBEEF));
    let samples = poisson.run();
    plot_3d(
        &samples,
        [1000, 1000],
        String::from("examples/plot/images/plot_3d.html"),
    );
    println!(
        "Plotted {} samples in examples/plot/images/plot_23.html",
        samples.len()
    );
}
