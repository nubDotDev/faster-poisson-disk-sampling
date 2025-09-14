//! Implements an improved dart throwing algorithm.
//!
//! The grid cells are shuffled randomly and a fixed number of attempts are made to place a point in each one.
//!
//! `attempts` defaults to 6.

use crate::common::{
    Grid, Grid2D, Grid3D, GridImpl, GridND, Idx, ND, Params2D, Params3D, ParamsImpl, ParamsND,
    RandomSampler, RandomSpec, RandomState, Sampler, ThreeD, TwoD,
};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256StarStar;
use std::array;

fn new_state<const N: usize, R, T>(
    sampler: &RandomSampler<R, Dart>,
    grid: &Grid<N, T>,
) -> RandomState<R>
where
    R: Rng + SeedableRng,
{
    let mut rng = match sampler.random.seed {
        None => R::from_os_rng(),
        Some(seed) => R::seed_from_u64(seed),
    };
    let mut active: Vec<usize> = (0..grid.cells.len()).collect();
    active.shuffle(&mut rng);
    RandomState { active, rng }
}

pub struct Dart;

pub type DartSampler<R = Xoshiro256StarStar> = RandomSampler<R, Dart>;

impl<R> Default for RandomSampler<R, Dart>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        RandomSampler {
            random: RandomSpec::new(6),
            _rng: Default::default(),
            _t: Default::default(),
        }
    }
}

impl<R> Sampler<2, TwoD> for DartSampler<R>
where
    R: Rng + SeedableRng,
{
    type State = RandomState<R>;

    fn new_state(&self, _params: &Params2D, grid: &Grid2D) -> Self::State {
        new_state(self, grid)
    }

    fn sample(
        &self,
        params: &Params2D,
        grid: &mut Grid2D,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<2>> {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx = grid.idx_to_ndidx(idx);
                    let ranges = [
                        ndidx[0] as f64 * grid.cell_len..(ndidx[0] + 1) as f64 * grid.cell_len,
                        ndidx[1] as f64 * grid.cell_len..(ndidx[1] + 1) as f64 * grid.cell_len,
                    ];
                    for _ in 0..self.random.attempts {
                        let cand = [
                            state.rng.random_range(ranges[0].clone()),
                            state.rng.random_range(ranges[1].clone()),
                        ];
                        if params.is_sample_valid(&cand, grid) {
                            grid.add_point(&cand);
                            return Some(cand);
                        }
                    }
                }
            }
        }
    }
}

impl<R> Sampler<3, ThreeD> for DartSampler<R>
where
    R: Rng + SeedableRng,
{
    type State = RandomState<R>;

    fn new_state(&self, _params: &Params3D, grid: &Grid3D) -> Self::State {
        new_state(self, grid)
    }

    fn sample(
        &self,
        params: &Params3D,
        grid: &mut Grid3D,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<3>> {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx = grid.idx_to_ndidx(idx);
                    let ranges = [
                        ndidx[0] as f64 * grid.cell_len..(ndidx[0] + 1) as f64 * grid.cell_len,
                        ndidx[1] as f64 * grid.cell_len..(ndidx[1] + 1) as f64 * grid.cell_len,
                        ndidx[2] as f64 * grid.cell_len..(ndidx[2] + 1) as f64 * grid.cell_len,
                    ];
                    for _ in 0..self.random.attempts {
                        let cand = [
                            state.rng.random_range(ranges[0].clone()),
                            state.rng.random_range(ranges[1].clone()),
                            state.rng.random_range(ranges[2].clone()),
                        ];
                        if params.is_sample_valid(&cand, grid) {
                            grid.add_point(&cand);
                            return Some(cand);
                        }
                    }
                }
            }
        }
    }
}

impl<const N: usize, R> Sampler<N, ND> for DartSampler<R>
where
    R: Rng + SeedableRng,
{
    type State = RandomState<R>;

    fn new_state(&self, _params: &ParamsND<N>, grid: &GridND<N>) -> Self::State {
        new_state(self, grid)
    }

    fn sample(
        &self,
        params: &ParamsND<N>,
        grid: &mut GridND<N>,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<N>> {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx: Idx<N> = grid.idx_to_ndidx(idx);
                    let ranges =
                        ndidx.map(|x| x as f64 * grid.cell_len..(x + 1) as f64 * grid.cell_len);
                    for _ in 0..self.random.attempts {
                        let cand = array::from_fn(|i| state.rng.random_range(ranges[i].clone()));
                        if params.is_sample_valid(&cand, grid) {
                            grid.add_point(&cand);
                            return Some(cand);
                        }
                    }
                }
            }
        }
    }
}
