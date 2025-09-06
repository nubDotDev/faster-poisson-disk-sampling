mod bridson;
mod regular;

#[cfg(feature = "plotly")]
mod plot;

use crate::{
    bridson::{
        BridsonSampler2D, BridsonSampler3D, BridsonSamplerBase, BridsonSamplerND, ParentalSampler2D,
    },
    regular::RegularSampler,
};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng};
use std::{array, f64::consts::SQRT_2, iter};

type Point<const N: usize> = [f64; N];
type Idx<const N: usize> = [usize; N];

pub struct GridBase<const N: usize> {
    cell_len: f64,
    grid_dims: Idx<N>,
    cells: Vec<Option<usize>>,
    samples: Vec<Point<N>>,
}

pub trait Grid<const N: usize>:
    Deref<Target = GridBase<N>> + DerefMut<Target = GridBase<N>>
{
    fn ndidx_to_idx(&self, ndidx: &Idx<N>) -> usize;
    fn point_to_ndidx(&self, p: &Point<N>) -> Idx<N>;

    #[inline(always)]
    fn point_to_idx(&self, p: &Point<N>) -> usize {
        self.ndidx_to_idx(&self.point_to_ndidx(p))
    }

    fn add_point(&mut self, p: &Point<N>) -> usize {
        let grid_idx = self.point_to_idx(p);
        let sample_idx = self.samples.len();
        self.cells[grid_idx] = Some(sample_idx);
        self.samples.push(*p);
        return sample_idx;
    }
}

#[derive(Deref, DerefMut)]
pub struct Grid2D(GridBase<2>);

#[derive(Deref, DerefMut)]
pub struct Grid3D(GridBase<3>);

#[derive(Deref, DerefMut)]
pub struct GridND<const N: usize>(GridBase<N>);

impl Grid<2> for Grid2D {
    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<2>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1]
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<2>) -> Idx<2> {
        [
            (p[0] / self.cell_len).floor() as usize,
            (p[1] / self.cell_len).floor() as usize,
        ]
    }
}

impl Grid<3> for Grid3D {
    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<3>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1] + self.grid_dims[0] * self.grid_dims[1] * ndidx[2]
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<3>) -> Idx<3> {
        [
            (p[0] / self.cell_len).floor() as usize,
            (p[1] / self.cell_len).floor() as usize,
            (p[2] / self.cell_len).floor() as usize,
        ]
    }
}

impl<const N: usize> Grid<N> for GridND<N> {
    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<N>) -> usize {
        let mut res = 0;
        let mut mul = 1;
        for i in 0..N {
            res += ndidx[i] * mul;
            mul *= self.grid_dims[i];
        }
        res
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<N>) -> Idx<N> {
        p.map(|x| (x / self.cell_len).floor() as usize)
    }
}

pub struct ParamsBase<const N: usize> {
    dims: Point<N>,
    radius: f64,
}

pub trait Params<const N: usize>:
    Default + Deref<Target = ParamsBase<N>> + DerefMut<Target = ParamsBase<N>>
{
    type Grid: Grid<N>;

    fn grid(&self) -> Self::Grid;
    fn is_sample_valid(&self, p: &Point<N>, grid: &Self::Grid) -> bool;
}

#[derive(Deref, DerefMut)]
pub struct Params2D(ParamsBase<2>);

#[derive(Deref, DerefMut)]
pub struct Params3D(ParamsBase<3>);

#[derive(Deref, DerefMut)]
pub struct ParamsND<const N: usize>(ParamsBase<N>);

impl Default for Params2D {
    fn default() -> Self {
        Params2D(ParamsBase {
            dims: [1.0, 1.0],
            radius: 0.1,
        })
    }
}

impl Default for Params3D {
    fn default() -> Self {
        Params3D(ParamsBase {
            dims: [1.0, 1.0, 1.0],
            radius: 0.1,
        })
    }
}

impl<const N: usize> Default for ParamsND<N> {
    fn default() -> Self {
        ParamsND(ParamsBase {
            dims: [1.0; N],
            radius: 0.1,
        })
    }
}

impl Params<2> for Params2D {
    type Grid = Grid2D;

    fn grid(&self) -> Grid2D {
        let cell_len = self.radius / SQRT_2;
        let grid_dims = [
            (self.dims[0] / cell_len).ceil() as usize,
            (self.dims[1] / cell_len).ceil() as usize,
        ];
        let grid_len = grid_dims[0] * grid_dims[1];
        let mut samples = Vec::new();
        samples.reserve(grid_len);
        Grid2D(GridBase {
            cell_len,
            grid_dims,
            cells: vec![None; grid_len],
            samples,
        })
    }

    fn is_sample_valid(&self, p: &Point<2>, grid: &Grid2D) -> bool {
        if !(0.0..=self.dims[0]).contains(&p[0]) || !(0.0..=self.dims[1]).contains(&p[1]) {
            return false;
        }

        const FRACT_SQRT_2: f64 = SQRT_2 - 1.0;
        const ONE_MINUS_FRACT_SQRT_2: f64 = 2.0 - SQRT_2;

        let ndidx = grid.point_to_ndidx(p);
        let p_rems = [
            (p[0] / grid.cell_len).fract(),
            (p[1] / grid.cell_len).fract(),
        ];
        let ranges = [
            ndidx[0].saturating_sub(1 + ((p_rems[0] <= FRACT_SQRT_2) as usize))
                ..=(ndidx[0] + 1 + ((p_rems[0] >= ONE_MINUS_FRACT_SQRT_2) as usize))
                    .min(grid.grid_dims[0] - 1),
            ndidx[1].saturating_sub(1 + ((p_rems[1] <= FRACT_SQRT_2) as usize))
                ..=(ndidx[1] + 1 + ((p_rems[1] >= ONE_MINUS_FRACT_SQRT_2) as usize))
                    .min(grid.grid_dims[1] - 1),
        ];

        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                match grid.cells[grid.ndidx_to_idx(&[i, j])] {
                    None => (),
                    Some(sample_idx) => {
                        let sample = &grid.samples[sample_idx];
                        let d = [sample[0] - p[0], sample[1] - p[1]];
                        if d[0] * d[0] + d[1] * d[1] < self.radius * self.radius {
                            return false;
                        }
                    }
                };
            }
        }

        return true;
    }
}

impl Params<3> for Params3D {
    type Grid = Grid3D;

    fn grid(&self) -> Grid3D {
        let cell_len = self.radius / 3.0f64.sqrt();
        let grid_dims = [
            (self.dims[0] / cell_len).ceil() as usize,
            (self.dims[1] / cell_len).ceil() as usize,
            (self.dims[2] / cell_len).ceil() as usize,
        ];
        let grid_len = grid_dims[0] * grid_dims[1] * grid_dims[2];
        let mut samples = Vec::new();
        samples.reserve(grid_len);
        Grid3D(GridBase {
            cell_len,
            grid_dims,
            cells: vec![None; grid_len],
            samples,
        })
    }

    fn is_sample_valid(&self, p: &Point<3>, grid: &Grid3D) -> bool {
        if !(0.0..=self.dims[0]).contains(&p[0])
            || !(0.0..=self.dims[1]).contains(&p[1])
            || !(0.0..=self.dims[2]).contains(&p[2])
        {
            return false;
        }

        let fract_sqrt_3: f64 = 3.0f64.sqrt() - 1.0;
        let one_minus_fract_sqrt_3: f64 = 2.0 - 3.0f64.sqrt();

        let ndidx = grid.point_to_ndidx(p);
        let p_rems = [
            (p[0] / grid.cell_len).fract(),
            (p[1] / grid.cell_len).fract(),
            (p[2] / grid.cell_len).fract(),
        ];
        let ranges = [
            ndidx[0].saturating_sub(1 + ((p_rems[0] <= fract_sqrt_3) as usize))
                ..=(ndidx[0] + 1 + ((p_rems[0] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[0] - 1),
            ndidx[1].saturating_sub(1 + ((p_rems[1] <= fract_sqrt_3) as usize))
                ..=(ndidx[1] + 1 + ((p_rems[1] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[1] - 1),
            ndidx[2].saturating_sub(1 + ((p_rems[2] <= fract_sqrt_3) as usize))
                ..=(ndidx[2] + 1 + ((p_rems[2] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[2] - 1),
        ];

        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                for k in ranges[2].clone() {
                    match grid.cells[grid.ndidx_to_idx(&[i, j, k])] {
                        None => (),
                        Some(sample_idx) => {
                            let sample = &grid.samples[sample_idx];
                            let d = [sample[0] - p[0], sample[1] - p[1]];
                            if d[0] * d[0] + d[1] * d[1] < self.radius * self.radius {
                                return false;
                            }
                        }
                    };
                }
            }
        }

        return true;
    }
}

impl<const N: usize> Params<N> for ParamsND<N> {
    type Grid = GridND<N>;

    fn grid(&self) -> GridND<N> {
        let cell_len = self.radius / (N as f64).sqrt();
        let grid_dims = self.dims.map(|x| (x / cell_len).ceil() as usize);
        let grid_len = grid_dims.iter().product();
        let mut samples = Vec::new();
        samples.reserve(grid_len);
        GridND(GridBase {
            cell_len,
            grid_dims,
            cells: vec![None; grid_len],
            samples,
        })
    }

    fn is_sample_valid(&self, p: &Point<N>, grid: &GridND<N>) -> bool {
        for i in 0..N {
            if !(0.0..=self.dims[i]).contains(&p[i]) {
                return false;
            }
        }

        let ndidx = grid.point_to_ndidx(p);
        let radius_quo = N.isqrt();
        let radius_rem = (N as f64).sqrt().fract();
        let p_rems = p.map(|x| (x / grid.cell_len).fract());
        let loidx: [usize; N] = array::from_fn(|i| {
            ndidx[i].saturating_sub(radius_quo + ((p_rems[i] <= radius_rem) as usize))
        });
        let hiidx: [usize; N] = array::from_fn(|i| {
            (ndidx[i] + radius_quo + ((p_rems[i] >= grid.cell_len - radius_rem) as usize))
                .min(grid.grid_dims[i] - 1)
        });

        let mut curr = loidx;
        'out: loop {
            match grid.cells[grid.ndidx_to_idx(&curr)] {
                None => (),
                Some(sample_idx) => {
                    let sample = &grid.samples[sample_idx];
                    let mut dist_sq = 0.0;
                    for i in 0..N {
                        let d = sample[i] - p[i];
                        dist_sq += d * d;
                    }
                    if dist_sq < self.radius * self.radius {
                        return false;
                    }
                }
            };

            for i in 0..N - 1 {
                if curr[i] < hiidx[i] {
                    curr[i] += 1;
                    continue 'out;
                }
                curr[i] = loidx[i];
            }
            if curr[N - 1] < hiidx[N - 1] {
                curr[N - 1] += 1;
                continue;
            }
            break;
        }

        return true;
    }
}

pub trait State<const N: usize, S>
where
    S: Sampler<N>,
{
    fn new<P>(sampler: &S, params: &P, grid: &P::Grid) -> Self
    where
        P: Params<N>;
}

pub trait Sampler<const N: usize>: Default {
    type State: State<N, Self>;

    fn new() -> Self {
        Self::default()
    }

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut Self::State,
    ) -> Option<Point<N>>
    where
        P: Params<N>;
}

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
        let mut state = S::State::new(&self.sampler, &self.params, &grid);
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

pub type Poisson2D = Poisson<2, Params2D, ParentalSampler2D>;
pub type PoissonBridson2D = Poisson<2, Params2D, BridsonSampler2D>;
pub type PoissonRegular2D = Poisson<2, Params2D, RegularSampler<2>>;
pub type Poisson3D = Poisson<3, Params3D, BridsonSampler3D>;
pub type PoissonRegular3D<const N: usize> = Poisson<3, Params3D, RegularSampler<3>>;
pub type PoissonND<const N: usize> = Poisson<N, ParamsND<N>, BridsonSamplerND<N>>;
pub type PoissonRegularND<const N: usize> = Poisson<N, ParamsND<N>, RegularSampler<N>>;

#[cfg(test)]
mod tests {
    use crate::bridson::BridsonSampler2D;

    use super::*;

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
    fn test_regular() {
        let poisson = Poisson::<3, Params3D, RegularSampler<3>>::new();
        len_and_distance(&poisson);
    }
}
