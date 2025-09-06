use crate::{Params, Point, Poisson, Sampler};
use plotly::{
    Layout, Plot, Scatter, Scatter3D,
    common::{Marker, Mode},
    layout::{Axis, LayoutScene},
};

impl<P, S> Poisson<2, P, S>
where
    P: Params<2>,
    S: Sampler<2>,
{
    pub fn run_and_plot(&self, path: String, pixels_per_unit: u32) -> Vec<Point<2>> {
        let samples = self.run();
        let (x, y): (Vec<_>, Vec<_>) = samples.clone().into_iter().map(|[x, y]| (x, y)).unzip();

        let mut plot = Plot::new();
        let layout = Layout::new()
            .x_axis(
                Axis::new()
                    .range(vec![0.0, self.params.dims[0]])
                    .visible(false),
            )
            .y_axis(
                Axis::new()
                    .range(vec![0.0, self.params.dims[1]])
                    .visible(false),
            )
            .width((self.params.dims[0] * pixels_per_unit as f64) as usize)
            .height((self.params.dims[1] * pixels_per_unit as f64) as usize);
        let trace = Scatter::new(x, y)
            .mode(Mode::Markers)
            .marker(Marker::new().color("black"));
        plot.set_layout(layout);
        plot.add_trace(trace);
        plot.write_html(path);

        samples
    }
}

impl<P, S> Poisson<3, P, S>
where
    P: Params<3>,
    S: Sampler<3>,
{
    pub fn run_and_plot(&self, path: String) -> Vec<Point<3>> {
        let samples = self.run();
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
        let layout = Layout::new().scene(
            LayoutScene::new()
                .x_axis(
                    Axis::new()
                        .range(vec![0.0, self.params.dims[0]])
                        .visible(false),
                )
                .y_axis(
                    Axis::new()
                        .range(vec![0.0, self.params.dims[1]])
                        .visible(false),
                )
                .z_axis(
                    Axis::new()
                        .range(vec![0.0, self.params.dims[2]])
                        .visible(false),
                ),
        );
        let trace = Scatter3D::new(x, y, z)
            .mode(Mode::Markers)
            .marker(Marker::new().size(3).color("black"));
        plot.set_layout(layout);
        plot.add_trace(trace);
        plot.write_html(path);

        samples
    }
}
