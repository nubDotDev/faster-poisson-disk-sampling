use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::marker::PhantomData;

mod iter;
pub use iter::PoissonIter;
use iter::{PoissonIterImpl, PoissonIterInner};

mod nbhd;
pub use nbhd::{NbhdSampler, ParentalNbhdSampler, StandardNbhdSampler};

type Float = f64;
type Point<const N: usize> = [Float; N];
type NDIdx<const N: usize> = [usize; N];

pub struct Poisson<
    const N: usize,
    S: NbhdSampler<N> = StandardNbhdSampler<N>,
    R: Rng + SeedableRng = Xoshiro256StarStar,
> {
    dims: Point<N>,
    radius: Float,
    attempts: usize,
    cdf_exp: Float,
    initial_sample: Option<Point<N>>,

    _nbhd_sampler: PhantomData<S>,

    _rng: PhantomData<R>,
    seed: Option<u64>,
}

pub type Poisson2D = Poisson<2>;
pub type Poisson3D = Poisson<3>;

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> Poisson<N, S, R>
where
    PoissonIterInner<N, S, R>: PoissonIterImpl<N>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iter(&self) -> PoissonIter<N, S, R> {
        PoissonIter::new(self)
    }

    pub fn run(&self) -> Vec<Point<N>> {
        self.iter().collect()
    }

    pub fn use_dims(mut self, dims: Point<N>) -> Self {
        self.dims = dims;
        self
    }

    pub fn use_radius(mut self, radius: Float) -> Self {
        self.radius = radius;
        self
    }

    pub fn use_attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts;
        self
    }

    pub fn use_cdf_exp(mut self, cdf_exp: Float) -> Self {
        self.cdf_exp = cdf_exp;
        self
    }

    pub fn use_initial_sample(mut self, initial_sample: Option<Point<N>>) -> Self {
        self.initial_sample = initial_sample;
        self
    }

    pub fn use_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> Clone for Poisson<N, S, R> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> Default for Poisson<N, S, R> {
    fn default() -> Self {
        Poisson {
            dims: [1.0; N],
            radius: 0.1,
            attempts: 30,
            cdf_exp: 2.0,
            initial_sample: None,

            _nbhd_sampler: Default::default(),

            seed: None,
            _rng: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonempty() {
        assert!(Poisson2D::new().use_seed(Some(0xDEADBEEF)).run().len() > 0);
        assert!(Poisson3D::new().use_seed(Some(0xDEADBEEF)).run().len() > 0);
    }

    fn dist<const N: usize>(v1: &[Float; N], v2: &[Float; N]) -> Float {
        std::array::from_fn::<Float, N, _>(|i| (v1[i] - v2[i]).powi(2))
            .iter()
            .sum::<Float>()
            .sqrt()
    }

    #[test]
    fn closeness2d() {
        let poisson2d = Poisson2D::new().use_seed(Some(0xDEADBEEF));
        let samples2d = poisson2d.run();
        for (i, s1) in samples2d.iter().enumerate() {
            for s2 in &samples2d[i + 1..] {
                if s1 == s2 {
                    continue;
                }
                assert!(dist(s1, s2) >= poisson2d.radius);
            }
        }
    }

    #[test]
    fn closeness3d() {
        let poisson3d = Poisson3D::new().use_seed(Some(0xDEADBEEF));
        let samples3d = poisson3d.run();
        for (i, s1) in samples3d.iter().enumerate() {
            for s2 in &samples3d[i + 1..] {
                if s1 == s2 {
                    continue;
                }
                assert!(dist(s1, s2) >= poisson3d.radius, "{:?} {:?}", s1, s2);
            }
        }
    }
}
