use std::sync::Arc;

use image::{GrayImage, Luma};
use imageproc::drawing::draw_filled_circle_mut;

use crate::Poisson2D;

/// - `Highlight` puts white dots on a black background.
/// - `Shade` puts black dots on a white background.
pub enum StippleMode {
    Highlight,
    Shade,
}

/// Transforms the image at `src_path` into a stippled black and white image saved to `path`.
/// Each dot has radius `dot_radius`
/// and the minimum distance between points linearly interpolates between `min_radius` and `max_radius`.
pub fn stipple(
    src_path: String,
    mode: StippleMode,
    dot_radius: i32,
    min_radius: f64,
    max_radius: f64,
    path: String,
) {
    let img = image::open(src_path).unwrap();
    let gray_img = img.to_luma8();
    let img_dims = gray_img.dimensions();
    let poisson = Poisson2D::new()
        .dims([img_dims.0 as f64, img_dims.1 as f64])
        .radius(min_radius);

    match mode {
        StippleMode::Shade => {
            let poisson = poisson.radius_fn(Some(Arc::new(move |p| {
                min_radius
                    + (max_radius - min_radius)
                        * (gray_img.get_pixel(p[0].floor() as u32, p[1].floor() as u32)[0] as f64)
                        / 255.0
            })));
            let mut stipple_img = GrayImage::new(img_dims.0, img_dims.1);
            for px in stipple_img.iter_mut() {
                *px = 255;
            }
            for p in poisson.iter() {
                // stipple_img.put_pixel(p[0].floor() as u32, p[1].floor() as u32, Luma([0]));
                draw_filled_circle_mut(
                    &mut stipple_img,
                    (p[0].floor() as i32, p[1].floor() as i32),
                    dot_radius,
                    Luma([0]),
                );
            }
            let _ = stipple_img.save(path);
        }
        StippleMode::Highlight => {
            let poisson = poisson.radius_fn(Some(Arc::new(move |p| {
                min_radius
                    + (max_radius - min_radius)
                        * (255.0
                            - gray_img.get_pixel(p[0].floor() as u32, p[1].floor() as u32)[0]
                                as f64)
                        / 255.0
            })));
            let mut stipple_img = GrayImage::new(img_dims.0, img_dims.1);
            for p in poisson.iter() {
                // stipple_img.put_pixel(p[0].floor() as u32, p[1].floor() as u32, Luma([255]));
                draw_filled_circle_mut(
                    &mut stipple_img,
                    (p[0].floor() as i32, p[1].floor() as i32),
                    dot_radius,
                    Luma([255]),
                );
            }
            let _ = stipple_img.save(path);
        }
    };
}
