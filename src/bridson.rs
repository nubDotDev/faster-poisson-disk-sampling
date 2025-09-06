use super::Point;
use crate::{Grid, PoissonParams, PoissonSampler, State};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::{array, f64::consts::TAU, iter, marker::PhantomData};

struct ActiveSample {
    idx: usize,
    parent_idx: Option<usize>,
}

pub struct BridsonState<R>
where
    R: Rng + SeedableRng,
{
    active: Vec<usize>,
    rng: R,
}

pub struct ParentalState<R>
where
    R: Rng + SeedableRng,
{
    active: Vec<ActiveSample>,
    rng: R,
}

impl<const N: usize, B, R> State<B> for BridsonState<R>
where
    B: Deref<Target = BridsonSamplerBase<N, R>>,
    R: Rng + SeedableRng,
{
    fn new(sampler: &B) -> Self {
        BridsonState {
            active: Vec::new(),
            rng: match sampler.random.seed {
                None => R::from_os_rng(),
                Some(seed) => R::seed_from_u64(seed),
            },
        }
    }
}

impl<R> State<ParentalSampler2D<R>> for ParentalState<R>
where
    R: Rng + SeedableRng,
{
    fn new(sampler: &ParentalSampler2D<R>) -> Self {
        ParentalState {
            active: Vec::new(),
            rng: match sampler.random.seed {
                None => R::from_os_rng(),
                Some(seed) => R::seed_from_u64(seed),
            },
        }
    }
}

pub(crate) struct RandomSampler<R>
where
    R: Rng + SeedableRng,
{
    pub(crate) seed: Option<u64>,
    _rng: PhantomData<R>,
}

impl<R> Default for RandomSampler<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        RandomSampler {
            seed: None,
            _rng: Default::default(),
        }
    }
}

pub struct BridsonSamplerBase<const N: usize, R>
where
    R: Rng + SeedableRng,
{
    pub(crate) attempts: usize,
    pub(crate) cdf_exp: f64,
    pub(crate) random: RandomSampler<R>,
}

#[derive(Deref, DerefMut)]
pub struct BridsonSampler2D<R = Xoshiro256StarStar>(BridsonSamplerBase<2, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct BridsonSampler3D<R = Xoshiro256StarStar>(BridsonSamplerBase<3, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct BridsonSamplerND<const N: usize, R = Xoshiro256StarStar>(BridsonSamplerBase<N, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct ParentalSampler2D<R = Xoshiro256StarStar>(BridsonSamplerBase<2, R>)
where
    R: Rng + SeedableRng;

impl<R> Default for BridsonSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSampler2D(BridsonSamplerBase {
            attempts: 30,
            cdf_exp: 2.0,
            random: RandomSampler::default(),
        })
    }
}

impl<R> Default for BridsonSampler3D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSampler3D(BridsonSamplerBase {
            attempts: 30,
            cdf_exp: 2.0,
            random: RandomSampler::default(),
        })
    }
}

impl<const N: usize, R> Default for BridsonSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSamplerND(BridsonSamplerBase {
            attempts: 30,
            cdf_exp: 2.0,
            random: RandomSampler::default(),
        })
    }
}

impl<R> Default for ParentalSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        ParentalSampler2D(BridsonSamplerBase {
            attempts: 21,
            cdf_exp: 1.0,
            random: RandomSampler::default(),
        })
    }
}

impl<R> PoissonSampler<2> for BridsonSampler2D<R>
where
    R: Rng + SeedableRng,
{
    type State = BridsonState<R>;

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut BridsonState<R>,
    ) -> Option<Point<2>>
    where
        P: PoissonParams<2>,
    {
        if grid.samples.len() == 0 {
            let p = [
                state.rng.random_range(0.0..=params.dims[0]),
                state.rng.random_range(0.0..=params.dims[1]),
            ];
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p: &[f64; 2] = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = 2.0
                    * params.radius
                    * state
                        .rng
                        .random_range(0.5f64.powf(self.cdf_exp)..=1.0)
                        .powf(1.0 / self.cdf_exp);
                let theta = state.rng.random_range(0.0..TAU);
                Some([p[0] + s * theta.cos(), p[1] + s * theta.sin()])
            })
            .take(self.attempts)
            .find(|cand| params.is_sample_valid(&cand, grid));

            match p_opt {
                None => {
                    state.active.swap_remove(active_idx);
                }
                Some(p) => {
                    state.active.push(grid.add_point(&p));
                    return p_opt;
                }
            }
        }

        None
    }
}

impl<R> PoissonSampler<3> for BridsonSampler3D<R>
where
    R: Rng + SeedableRng,
{
    type State = BridsonState<R>;

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut BridsonState<R>,
    ) -> Option<Point<3>>
    where
        P: PoissonParams<3>,
    {
        if grid.samples.len() == 0 {
            let p = [
                state.rng.random_range(0.0..=params.dims[0]),
                state.rng.random_range(0.0..=params.dims[1]),
                state.rng.random_range(0.0..=params.dims[2]),
            ];
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = 2.0
                    * params.radius
                    * state
                        .rng
                        .random_range(0.5f64.powf(self.cdf_exp)..=1.0)
                        .powf(1.0 / self.cdf_exp);
                let v: Point<3> = [
                    state.rng.random_range(-1.0..=1.0),
                    state.rng.random_range(-1.0..=1.0),
                    state.rng.random_range(-1.0..=1.0),
                ];
                let scale = s / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                Some([
                    p[0] + scale * v[0],
                    p[1] + scale * v[1],
                    p[2] + scale * v[2],
                ])
            })
            .take(self.attempts)
            .find(|cand| params.is_sample_valid(&cand, grid));

            match p_opt {
                None => {
                    state.active.swap_remove(active_idx);
                }
                Some(p) => {
                    state.active.push(grid.add_point(&p));
                    return p_opt;
                }
            }
        }

        None
    }
}

impl<const N: usize, R> PoissonSampler<N> for BridsonSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    type State = BridsonState<R>;

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut BridsonState<R>,
    ) -> Option<Point<N>>
    where
        P: PoissonParams<N>,
    {
        if grid.samples.len() == 0 {
            let p = params.dims.map(|x| state.rng.random_range(0.0..=x));
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = 2.0
                    * params.radius
                    * state
                        .rng
                        .random_range(0.5f64.powf(self.cdf_exp)..=1.0)
                        .powf(1.0 / self.cdf_exp);
                let v: Point<N> = array::from_fn(|_| state.rng.random_range(-1.0..=1.0));
                let scale = s / v.iter().map(|x| x * x).sum::<f64>().sqrt();
                Some(array::from_fn(|i| p[i] + scale * v[i]))
            })
            .take(self.attempts)
            .find(|cand| params.is_sample_valid(&cand, grid));

            match p_opt {
                None => {
                    state.active.swap_remove(active_idx);
                }
                Some(p) => {
                    state.active.push(grid.add_point(&p));
                    return p_opt;
                }
            }
        }

        None
    }
}

impl<R> PoissonSampler<2> for ParentalSampler2D<R>
where
    R: Rng + SeedableRng,
{
    type State = ParentalState<R>;

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut ParentalState<R>,
    ) -> Option<Point<2>>
    where
        P: PoissonParams<2>,
    {
        if grid.samples.len() == 0 {
            let p = params.dims.map(|x| state.rng.random_range(0.0..=x));
            state.active.push(ActiveSample {
                idx: grid.add_point(&p),
                parent_idx: None,
            });
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample = &state.active[active_idx];
            let p = &grid.samples[sample.idx];

            let theta_range = match sample.parent_idx {
                None => 0.0..TAU,
                Some(parent_idx) => {
                    let parent = &grid.samples[parent_idx];
                    let d = [
                        (parent[0] - p[0]) / params.radius,
                        (parent[1] - p[1]) / params.radius,
                    ];
                    let dist2 = d[0] * d[0] + d[1] * d[1];
                    let dist = dist2.sqrt();
                    let alpha = d[1].atan2(d[0]);
                    let outer = ((dist2 + 3.0) / (4.0 * dist)).acos();
                    let inner = (0.5 * dist).acos();
                    let beta = outer.min(inner);
                    alpha + beta..alpha + std::f64::consts::TAU - beta
                }
            };

            let p_opt = iter::from_fn(|| {
                let s = 2.0
                    * params.radius
                    * state
                        .rng
                        .random_range(0.5f64.powf(self.cdf_exp)..=1.0)
                        .powf(1.0 / self.cdf_exp);
                let theta = state.rng.random_range(theta_range.clone());
                Some([p[0] + s * theta.cos(), p[1] + s * theta.sin()])
            })
            .take(self.attempts)
            .find(|cand| params.is_sample_valid(&cand, grid));

            match p_opt {
                None => {
                    state.active.swap_remove(active_idx);
                }
                Some(p) => {
                    state.active.push(ActiveSample {
                        idx: grid.add_point(&p),
                        parent_idx: Some(sample.idx),
                    });
                    return p_opt;
                }
            }
        }

        None
    }
}
