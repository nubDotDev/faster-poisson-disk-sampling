use faster_poisson::{Poisson2D, Poisson3D};

#[cfg(feature = "plotly")]
fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("examples/plot/plots")?;

    let poisson = Poisson2D::new()
        .use_dims([5.0; 2])
        .use_seed(Some(0xDEADBEEF));
    let samples = poisson.run_and_plot(String::from("examples/plot/plots/plot_2d.html"), 200);
    println!(
        "Plotted {} samples in examples/plot/plots/plot_2d.html",
        samples.len()
    );

    let poisson = Poisson3D::new().use_seed(Some(0xDEADBEEF));
    let samples = poisson.run_and_plot(String::from("examples/plot/plots/plot_3d.html"));
    println!(
        "Plotted {} samples in examples/plot/plots/plot_23.html",
        samples.len()
    );

    Ok(())
}
