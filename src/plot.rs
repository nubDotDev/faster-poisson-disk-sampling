use crate::Point;
use plotly::{
    Layout, Plot, Scatter, Scatter3D,
    common::{Marker, Mode},
    layout::{Axis, LayoutScene},
};

/// Plots 2D `samples` on a grid of pixel dimensions `dims` as an HTML file at `path`.
///
/// ### Example
/// ```
/// use faster_poisson::{Poisson2D, plot_2d};
///
/// let samples = Poisson2D::new().run();
/// plot_2d(&samples, [1000, 1000], String::from("plot_2d.html"));
/// ```
pub fn plot_2d(samples: &Vec<Point<2>>, dims: [usize; 2], path: String) {
    let (x, y): (Vec<_>, Vec<_>) = samples.clone().into_iter().map(|[x, y]| (x, y)).unzip();

    let mut plot = Plot::new();
    let layout = Layout::new()
        .x_axis(Axis::new().visible(false))
        .y_axis(Axis::new().scale_anchor("x").visible(false))
        .width(dims[0])
        .height(dims[1]);
    let trace = Scatter::new(x, y)
        .mode(Mode::Markers)
        .marker(Marker::new().color("black"));
    plot.set_layout(layout);
    plot.add_trace(trace);
    plot.write_html(path);
}

/// Plots 3D `samples` on a grid of pixel dimensions `dims` as an HTML file at `path`.
///
/// ### Example
/// ```
/// use faster_poisson::{Poisson3D, plot_3d};
///
/// let samples = Poisson3D::new().run();
/// plot_3d(&samples, [1000, 1000], String::from("plot_3d.html"));
/// ```
pub fn plot_3d(samples: &Vec<Point<3>>, dims: [usize; 2], path: String) {
    let (x, y, z): (Vec<_>, Vec<_>, Vec<_>) =
        samples.clone().into_iter().map(|[a, b, c]| (a, b, c)).fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut xs, mut ys, mut zs), (x, y, z)| {
                xs.push(x);
                ys.push(y);
                zs.push(z);
                (xs, ys, zs)
            },
        );

    let mut plot = Plot::new();
    let layout = Layout::new()
        .scene(
            LayoutScene::new()
                .x_axis(Axis::new().title("").show_tick_labels(false))
                .y_axis(Axis::new().title("").show_tick_labels(false))
                .z_axis(Axis::new().title("").show_tick_labels(false)),
        )
        .width(dims[0])
        .height(dims[1]);
    let trace = Scatter3D::new(x, y, z)
        .mode(Mode::Markers)
        .marker(Marker::new().size(3).color("black"));
    plot.set_layout(layout);
    plot.add_trace(trace);
    plot.write_html(path);
}
