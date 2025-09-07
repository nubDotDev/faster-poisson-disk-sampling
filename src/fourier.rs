use crate::Point;
use image::{GrayImage, Luma};
use rustfft::{
    FftPlanner,
    num_complex::Complex64,
    num_traits::{One, Zero},
};

pub fn fourier(
    samples: &Vec<Point<2>>,
    dims: [f64; 2],
    pixels_per_unit: usize,
    brightness: f64,
    path: String,
) {
    // The magnitude of the DFT has 4-fold rotational symmetry, so compute the first quadrant.
    let quad_dims = dims.map(|x| (x * pixels_per_unit as f64).ceil() as usize);
    let mut quad = vec![vec![Complex64::zero(); quad_dims[0]]; quad_dims[1]];

    // The image is twice the size of the quadrant minus 1 for the center pixel.
    let img_dims = quad_dims.map(|x| 2 * x - 1);
    let mut image = GrayImage::new(img_dims[0] as u32, img_dims[1] as u32);

    // Turn the samples into binary pixels.
    for sample in samples {
        quad[((quad_dims[1] as f64 * sample[1] / dims[1]).floor() as usize)
            .min(quad_dims[1] - 1)][((quad_dims[0] as f64 * sample[0] / dims[0]).floor()
            as usize)
            .min(quad_dims[0] - 1)] = Complex64::one().scale(brightness);
    }

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let row_fft = planner.plan_fft_forward(quad_dims[0]);
    let col_fft = planner.plan_fft_forward(quad_dims[1]);

    // Take the DFTs of the rows.
    for row in quad.iter_mut() {
        row_fft.process(row);
    }

    // Take the transpose and reflect.
    let mut quad_t = vec![vec![Complex64::zero(); quad_dims[1]]; img_dims[0]];
    for i in 0..quad_dims[0] {
        for j in 0..quad_dims[1] {
            quad_t[quad_dims[0] - 1 + i][j] = quad[j][i];
            quad_t[quad_dims[0] - 1 - i][j] = quad[j][i].conj(); // F(-x) = F^*(x)
        }
    }

    // Take the DFTs of the columns.
    for col in quad_t.iter_mut() {
        col_fft.process(col);
    }

    // Create the image.
    let scale = 1.0 / ((quad_dims[0] * quad_dims[1]) as f64).sqrt();
    for i in 0..img_dims[0] {
        for j in 0..quad_dims[1] {
            let gray = [(255.0 * quad_t[i][j].norm() * scale).round() as u8];
            image.put_pixel(i as u32, (quad_dims[1] + j - 1) as u32, Luma(gray));
            image.put_pixel(
                (img_dims[0] - i - 1) as u32,
                (quad_dims[1] - j - 1) as u32,
                Luma(gray),
            );
        }
    }

    let _ = image.save(path);
}
