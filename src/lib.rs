mod bridson;
mod common;
mod dart;
mod regular;

#[cfg(feature = "fourier")]
mod fourier;
#[cfg(feature = "fourier")]
pub use crate::fourier::fourier;

#[cfg(feature = "plotly")]
mod plot;
#[cfg(feature = "plotly")]
pub use crate::plot::{plot_2d, plot_3d};

use crate::{bridson::*, common::*, dart::*, regular::*};
use derive_more::with_trait::DerefMut;
use rand::{Rng, SeedableRng};
use std::iter;

#[derive(Default)]
pub struct Poisson<const N: usize, P, S>
where
    P: Params<N>,
    S: Sampler<N>,
{
    params: P,
    sampler: S,
}

impl<const N: usize, P, S> Poisson<N, P, S>
where
    P: Params<N>,
    S: Sampler<N>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iter(&self) -> impl Iterator<Item = Point<N>> {
        let mut grid = self.params.grid();
        let mut state = self.sampler.new_state(&self.params, &grid);
        iter::from_fn(move || self.sampler.sample(&self.params, &mut grid, &mut state))
    }

    pub fn run(&self) -> Vec<Point<N>> {
        self.iter().collect()
    }

    pub fn use_dims(mut self, dims: Point<N>) -> Self {
        self.params.dims = dims;
        self
    }

    pub fn use_radius(mut self, radius: f64) -> Self {
        self.params.radius = radius;
        self
    }
}

impl<const N: usize, P, R, S> Poisson<N, P, S>
where
    P: Params<N>,
    R: Rng + SeedableRng,
    S: Sampler<N> + DerefMut<Target = BridsonSamplerBase<N, R>>,
{
    pub fn use_attempts(mut self, attempts: usize) -> Self {
        self.sampler.attempts = attempts;
        self
    }

    pub fn use_cdf_exp(mut self, cdf_exp: f64) -> Self {
        self.sampler.cdf_exp = cdf_exp;
        self
    }

    pub fn use_seed(mut self, seed: Option<u64>) -> Self {
        self.sampler.random.seed = seed;
        self
    }
}

impl<const N: usize, P, R> Poisson<N, P, DartSamplerND<N, R>>
where
    P: Params<N>,
    R: Rng + SeedableRng,
{
    pub fn use_attempts(mut self, attempts: usize) -> Self {
        self.sampler.attempts = attempts;
        self
    }

    pub fn use_seed(mut self, seed: Option<u64>) -> Self {
        self.sampler.random.seed = seed;
        self
    }
}

pub type Poisson2D = Poisson<2, Params2D, ParentalSampler2D>;
pub type PoissonBridson2D = Poisson<2, Params2D, BridsonSampler2D>;
pub type PoissonDart2D = Poisson<2, Params2D, DartSamplerND<2>>;
pub type PoissonRegular2D = Poisson<2, Params2D, RegularSamplerND<2>>;
pub type Poisson3D = Poisson<3, Params3D, BridsonSampler3D>;
pub type PoissonDart3D = Poisson<3, Params3D, DartSamplerND<3>>;
pub type PoissonRegular3D<const N: usize> = Poisson<3, Params3D, RegularSamplerND<3>>;
pub type PoissonND<const N: usize> = Poisson<N, ParamsND<N>, BridsonSamplerND<N>>;
pub type PoissonRegularND<const N: usize> = Poisson<N, ParamsND<N>, RegularSamplerND<N>>;
pub type PoissonDartND<const N: usize> = Poisson<N, ParamsND<N>, DartSamplerND<N>>;

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

    fn len_and_distance<const N: usize, P, S>(poisson: &Poisson<N, P, S>)
    where
        P: Params<N>,
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
        let poisson = Poisson::<2, Params2D, ParentalSampler2D>::new()
            .use_dims([5.0; 2])
            .use_seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_2d_bridson() {
        let poisson = Poisson::<2, Params2D, BridsonSampler2D>::new()
            .use_dims([5.0; 2])
            .use_seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_3d() {
        let poisson = Poisson::<3, Params3D, BridsonSampler3D>::new()
            .use_dims([2.0; 3])
            .use_seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_4d() {
        let poisson = Poisson::<4, ParamsND<4>, BridsonSamplerND<4>>::new()
            .use_dims([0.5; 4])
            .use_seed(Some(0xDEADBEEF));
        len_and_distance(&poisson);
    }

    #[test]
    fn test_dart() {
        let poisson = Poisson::<3, Params3D, DartSamplerND<3>>::new();
        len_and_distance(&poisson);
    }

    #[test]
    fn test_regular() {
        let poisson = Poisson::<3, Params3D, RegularSamplerND<3>>::new();
        len_and_distance(&poisson);
    }
}
