//! Improves upon the algorithm from Robert Bridson's
//! [Fast Poisson Disk Sampling in Arbitrary Dimensions](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf)
//!
//! Instead of uniformly sampling the annulus centered around a previous point,
//! we bias the center of the annulus.
//! When uniformly sampling a circle, the CDF of the distance from the center is x<sup>2</sup>.
//! Here, you can decrease this exponent or even set it to `None`,
//! representing the limit as the exponent goes to 0.
//! Restricted to the annulus of inner radius 1/2,
//! the limiting CDF of the distance from the center is log<sub>2</sub>(x) + 1.
//!
//! [`ParentalSampler2D`] makes the additional optimization of
//! removing the slice of the annulus that is guaranteed to be covered by the parent of the base point
//! (i.e., the point around which an annulus was sampled to generate the current point).
//!
//! `attempts` defaults to 16 in [`BridsonSampler2D`], [`BridsonSampler3D`], and [`BridsonSamplerND`].
//!
//! `attempts` defaults to 14 in [`ParentalSampler2D`].
//!
//! `cdf_exp` defaults to `None` in all of the above.

use super::Point;
use crate::{
    Grid, Params, Sampler,
    common::{
        Grid2D, Grid3D, GridND, HasRandom, Params2D, Params3D, ParamsND, Random, RandomState,
    },
};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::{array, f64::consts::TAU, iter, marker::PhantomData};

pub struct ActiveSample {
    idx: usize,
    parent_idx: Option<usize>,
}

pub struct BridsonSamplerBase<const N: usize, R>
where
    R: Rng + SeedableRng,
{
    pub(crate) cdf_exp: Option<f64>,
    pub(crate) random: Random,
    _rng: PhantomData<R>,
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

impl<const N: usize, R> HasRandom for BridsonSamplerBase<N, R>
where
    R: Rng + SeedableRng,
{
    fn get_random(&self) -> &Random {
        &self.random
    }

    fn get_random_mut(&mut self) -> &mut Random {
        &mut self.random
    }
}

impl<const N: usize, R> Default for BridsonSamplerBase<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSamplerBase {
            cdf_exp: None,
            random: Random::new(16),
            _rng: Default::default(),
        }
    }
}

impl<R> Default for BridsonSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSampler2D(BridsonSamplerBase::default())
    }
}

impl<R> Default for BridsonSampler3D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSampler3D(BridsonSamplerBase::default())
    }
}

impl<const N: usize, R> Default for BridsonSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        BridsonSamplerND(BridsonSamplerBase::default())
    }
}

impl<R> Default for ParentalSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        ParentalSampler2D(BridsonSamplerBase {
            cdf_exp: None,
            random: Random::new(14),
            _rng: Default::default(),
        })
    }
}

impl<R> Sampler<2> for BridsonSampler2D<R>
where
    R: Rng + SeedableRng,
{
    type Params = Params2D;
    type State = RandomState<R>;

    fn new_state(&self, _params: &Params2D, grid: &Grid2D) -> Self::State {
        RandomState::new(self.deref(), _params, grid)
    }

    fn sample(
        &self,
        params: &Params2D,
        grid: &mut Grid2D,
        state: &mut RandomState<R>,
    ) -> Option<Point<2>> {
        if grid.samples.len() == 0 {
            let p = [
                state.rng.random_range(0.0..params.dims[0]),
                state.rng.random_range(0.0..params.dims[1]),
            ];
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p: &[f64; 2] = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = match self.cdf_exp {
                    None => params.radius * 2.0f64.powf(state.rng.random_range(0.0..=1.0)),
                    Some(cdf_exp) => {
                        2.0 * params.radius
                            * state
                                .rng
                                .random_range(0.5f64.powf(cdf_exp)..=1.0)
                                .powf(1.0 / cdf_exp)
                    }
                };
                let theta = state.rng.random_range(0.0..TAU);
                Some([p[0] + s * theta.cos(), p[1] + s * theta.sin()])
            })
            .take(self.random.attempts)
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

impl<R> Sampler<3> for BridsonSampler3D<R>
where
    R: Rng + SeedableRng,
{
    type Params = Params3D;
    type State = RandomState<R>;

    fn new_state(&self, _params: &Params3D, grid: &Grid3D) -> Self::State {
        RandomState::new(self.deref(), _params, grid)
    }

    fn sample(
        &self,
        params: &Params3D,
        grid: &mut Grid3D,
        state: &mut RandomState<R>,
    ) -> Option<Point<3>> {
        if grid.samples.len() == 0 {
            let p = [
                state.rng.random_range(0.0..params.dims[0]),
                state.rng.random_range(0.0..params.dims[1]),
                state.rng.random_range(0.0..params.dims[2]),
            ];
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = match self.cdf_exp {
                    None => params.radius * 2.0f64.powf(state.rng.random_range(0.0..=1.0)),
                    Some(cdf_exp) => {
                        2.0 * params.radius
                            * state
                                .rng
                                .random_range(0.5f64.powf(cdf_exp)..=1.0)
                                .powf(1.0 / cdf_exp)
                    }
                };
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
            .take(self.random.attempts)
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

impl<const N: usize, R> Sampler<N> for BridsonSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    type Params = ParamsND<N>;
    type State = RandomState<R>;

    fn new_state(&self, _params: &ParamsND<N>, grid: &GridND<N>) -> Self::State {
        RandomState::new(self.deref(), _params, grid)
    }

    fn sample(
        &self,
        params: &ParamsND<N>,
        grid: &mut GridND<N>,
        state: &mut RandomState<R>,
    ) -> Option<Point<N>> {
        if grid.samples.len() == 0 {
            let p = params.dims.map(|x| state.rng.random_range(0.0..x));
            state.active.push(grid.add_point(&p));
            return Some(p);
        }

        while !state.active.is_empty() {
            let active_idx = state.rng.random_range(0..state.active.len());
            let sample_idx = state.active[active_idx];
            let p = &grid.samples[sample_idx];
            let p_opt = iter::from_fn(|| {
                let s = match self.cdf_exp {
                    None => params.radius * 2.0f64.powf(state.rng.random_range(0.0..=1.0)),
                    Some(cdf_exp) => {
                        2.0 * params.radius
                            * state
                                .rng
                                .random_range(0.5f64.powf(cdf_exp)..=1.0)
                                .powf(1.0 / cdf_exp)
                    }
                };
                let v: Point<N> = array::from_fn(|_| state.rng.random_range(-1.0..=1.0));
                let scale = s / v.iter().map(|x| x * x).sum::<f64>().sqrt();
                Some(array::from_fn(|i| p[i] + scale * v[i]))
            })
            .take(self.random.attempts)
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

impl<R> Sampler<2> for ParentalSampler2D<R>
where
    R: Rng + SeedableRng,
{
    type Params = Params2D;
    type State = RandomState<R, ActiveSample>;

    fn new_state(&self, _params: &Params2D, grid: &Grid2D) -> Self::State {
        RandomState::new(self.deref(), _params, grid)
    }

    fn sample(
        &self,
        params: &Params2D,
        grid: &mut Grid2D,
        state: &mut Self::State,
    ) -> Option<Point<2>> {
        if grid.samples.len() == 0 {
            let p = [
                state.rng.random_range(0.0..=params.dims[0]),
                state.rng.random_range(0.0..=params.dims[1]),
            ];
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
                    alpha + beta..alpha + TAU - beta
                }
            };

            let p_opt = iter::from_fn(|| {
                let s = match self.cdf_exp {
                    None => params.radius * 2.0f64.powf(state.rng.random_range(0.0..=1.0)),
                    Some(cdf_exp) => {
                        2.0 * params.radius
                            * state
                                .rng
                                .random_range(0.5f64.powf(cdf_exp)..=1.0)
                                .powf(1.0 / cdf_exp)
                    }
                };
                let theta = state.rng.random_range(theta_range.clone());
                Some([p[0] + s * theta.cos(), p[1] + s * theta.sin()])
            })
            .take(self.random.attempts)
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
