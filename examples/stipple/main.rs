use faster_poisson::stipple;

fn main() {
    stipple(
        String::from("examples/stipple/images/cat.png"),
        faster_poisson::StippleMode::Shade,
        1,
        1.5,
        5.0,
        String::from("examples/stipple/images/cat_stipple.png"),
    );
}
