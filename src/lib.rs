#![doc = include_str!("../README.md")]

pub mod bridson;
pub mod common;
pub mod dart;
pub mod naive;
pub mod regular;

#[cfg(feature = "fourier")]
mod fourier;
#[cfg(feature = "fourier")]
pub use crate::fourier::fourier;

#[cfg(feature = "plotly")]
mod plot;
#[cfg(feature = "plotly")]
pub use crate::plot::{plot_2d, plot_3d};

use crate::{bridson::*, common::*, dart::*, naive::*, regular::*};
use derive_more::with_trait::DerefMut;
use rand::{Rng, SeedableRng};
use std::iter;

/// Generates Poisson disk samples.
#[derive(Default)]
pub struct Poisson<const N: usize, S>
where
    S: Sampler<N>,
{
    params: S::Params,
    sampler: S,
}

impl<const N: usize, S> Poisson<N, S>
where
    S: Sampler<N>,
{
    /// Creates a new instance with the default parameters (`dims = [1.0; N]`, `radius = 0.1`)
    /// and sampler settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an iterator that lazily generates points.
    /// Note that this does not consume the instance, so you can reuse it.
    pub fn iter(&self) -> impl Iterator<Item = Point<N>> {
        let params = self.params;
        let mut grid = self.params.grid();
        let mut state = self.sampler.new_state(&self.params, &grid);
        iter::from_fn(move || self.sampler.sample(&params, &mut grid, &mut state))
    }

    /// Equivalent to `self.iter().collect()`.
    pub fn run(&self) -> Vec<Point<N>> {
        self.iter().collect()
    }

    /// Specify the dimensions of the grid from which to sample points.
    pub fn dims(mut self, dims: Point<N>) -> Self {
        self.params.dims = dims;
        self
    }

    /// Specify the minimum distance between points.
    pub fn radius(mut self, radius: f64) -> Self {
        self.params.radius = radius;
        self
    }
}

impl<const N: usize, S> Poisson<N, S>
where
    S: Sampler<N> + DerefMut<Target: HasRandom>,
{
    /// This has slightly different meaning depending on which sampler you use.
    /// - For `ParentalSampler` and `BridsonSampler`, this is the number of points to try from an annulus centered around a previous sample.
    /// - For `DartSampler`,  this is the number of points to try from each grid cell of side length `radius / N.sqrt()`.
    /// - For `NaiveSampler`, this is the number of failed attempts before giving up.
    pub fn attempts(mut self, attempts: usize) -> Self {
        self.sampler.get_random_mut().attempts = attempts;
        self
    }

    /// Specify the seed for the PRNG.
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.sampler.get_random_mut().seed = seed;
        self
    }
}

impl<const N: usize, R, S> Poisson<N, S>
where
    R: Rng + SeedableRng,
    S: Sampler<N> + DerefMut<Target = BridsonSamplerBase<N, R>>,
{
    /// For `ParentalSampler` and `BridsonSampler`, this specifies the bias toward or away from the center of the annulus.
    /// `None` and smaller values bias the center and larger values bias the outside.
    /// Set this to 2 for a uniform distribution.
    /// Non-`None` values must be positive.
    /// The number of points generated tends to increase as this decreases,
    /// with the most points being achieved by `None` (effectively the limit as `cdf_exp` goes to 0).
    pub fn cdf_exp(mut self, cdf_exp: Option<f64>) -> Self {
        self.sampler.cdf_exp = cdf_exp;
        self
    }
}

pub type Poisson2D = Poisson<2, ParentalSampler2D>;
pub type PoissonBridson2D = Poisson<2, BridsonSampler2D>;
pub type PoissonDart2D = Poisson<2, DartSampler2D>;
pub type PoissonNaive2D = Poisson<2, NaiveSamplerND<2>>;
pub type PoissonRegular2D = Poisson<2, RegularSamplerND<2>>;
pub type Poisson3D = Poisson<3, BridsonSampler3D>;
pub type PoissonDart3D = Poisson<3, DartSampler3D>;
pub type PoissonNaive3D = Poisson<3, NaiveSamplerND<3>>;
pub type PoissonRegular3D<const N: usize> = Poisson<3, RegularSamplerND<3>>;
pub type PoissonND<const N: usize> = Poisson<N, BridsonSamplerND<N>>;
pub type PoissonDartND<const N: usize> = Poisson<N, DartSamplerND<N>>;
pub type PoissonNaiveND<const N: usize> = Poisson<N, NaiveSamplerND<N>>;
pub type PoissonRegularND<const N: usize> = Poisson<N, RegularSamplerND<N>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::array;

    fn dist<const N: usize>(v1: &[f64; N], v2: &[f64; N]) -> f64 {
        array::from_fn::<f64, N, _>(|i| (v1[i] - v2[i]).powi(2))
            .iter()
            .sum::<f64>()
            .sqrt()
    }

    fn len_and_distance<const N: usize, S>(poisson: &Poisson<N, S>)
    where
        S: Sampler<N>,
    {
        let samples = poisson.run();

        // Check that the number of samples is at least half the orthogonal packing.
        let ortho = (poisson.params.dims.iter().product::<f64>()
            / (2.0 * poisson.params.radius).powi(N as i32)) as usize;
        assert!(
            samples.len() >= ortho,
            "Expected at least {ortho} samples but got {}",
            samples.len()
        );

        // Naive pairwise distance check.
        for (i, s1) in samples.iter().enumerate() {
            for s2 in &samples[i + 1..] {
                if s1 == s2 {
                    continue;
                }
                let d = dist(s1, s2);
                assert!(
                    d >= poisson.params.radius - std::f64::EPSILON,
                    "Invalid pair: {:?} {:?} (distance = {d}).",
                    s1,
                    s2
                );
            }
        }
    }

    #[test]
    fn test_2d_parental() {
        let poisson = Poisson::<2, ParentalSampler2D>::new()
            .dims([5.0; 2])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_2d_bridson() {
        let poisson = Poisson::<2, BridsonSampler2D>::new()
            .dims([5.0; 2])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_2d_dart() {
        let poisson = Poisson::<2, DartSampler2D>::new()
            .dims([5.0; 2])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_3d() {
        let poisson = Poisson::<3, BridsonSampler3D>::new()
            .dims([2.0; 3])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_3d_dart() {
        let poisson = Poisson::<3, DartSampler3D>::new()
            .dims([2.0; 3])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_4d() {
        let poisson = Poisson::<4, BridsonSamplerND<4>>::new()
            .dims([0.5; 4])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_4d_dart() {
        let poisson = Poisson::<4, DartSamplerND<4>>::new()
            .dims([0.5; 4])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_regular() {
        let poisson = Poisson::<3, RegularSamplerND<3>>::new();
        len_and_distance(&poisson);
    }

    #[test]
    fn test_naive() {
        let poisson = Poisson::<3, NaiveSamplerND<3>>::new()
            .dims([2.0; 3])
            .seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }
}
