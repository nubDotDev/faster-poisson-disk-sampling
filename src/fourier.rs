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
    // Turn the samples into binary pixels.
    let image_dims = dims.map(|x| (x * pixels_per_unit as f64).ceil() as usize);
    let mut pixels = vec![vec![Complex64::zero(); image_dims[0]]; image_dims[1]];
    for sample in samples {
        pixels[((image_dims[1] as f64 * sample[1] / dims[1]).floor() as usize)
            .min(image_dims[1] - 1)][((image_dims[0] as f64 * sample[0] / dims[0]).floor()
            as usize)
            .min(image_dims[0] - 1)] = Complex64::one().scale(brightness);
    }

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let row_fft = planner.plan_fft_forward(image_dims[0]);
    let col_fft = planner.plan_fft_forward(image_dims[1]);

    // Compute the DFTs of the rows.
    for row in pixels.iter_mut() {
        row_fft.process(row);
    }

    // Take the transpose.
    let mut pixels_t = vec![vec![Complex64::zero(); image_dims[1]]; image_dims[0]];
    for i in 0..image_dims[0] {
        for j in 0..image_dims[1] {
            pixels_t[i][j] = pixels[j][i];
        }
    }

    // Compute the DFTs of the columns.
    for col in pixels_t.iter_mut() {
        col_fft.process(col);
    }

    // Create the image.
    let mut image = GrayImage::new(image_dims[0] as u32, image_dims[1] as u32);
    let scale = 1.0 / ((image_dims[0] * image_dims[1]) as f64).sqrt();
    let nyquist = [image_dims[0] / 2, image_dims[1] / 2];
    for i in 0..image_dims[0] {
        for j in 0..image_dims[1] {
            let gray = [(255.0 * pixels_t[i][j].norm() * scale).round() as u8];
            let x = if i >= nyquist[0] {
                i - nyquist[0]
            } else {
                i + nyquist[0]
            } as u32;
            let y = if j >= nyquist[1] {
                j - nyquist[1]
            } else {
                j + nyquist[1]
            } as u32;
            image.put_pixel(x, y, Luma(gray));
        }
    }

    let _ = image.save(path);
}
