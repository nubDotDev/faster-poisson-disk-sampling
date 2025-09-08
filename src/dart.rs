use crate::common::{
    Grid, Grid2D, Grid3D, GridND, Idx, Params, Params2D, Params3D, ParamsND, RandomSamplerBase,
    Sampler,
};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256StarStar;
use std::array;

pub struct DartState<R>
where
    R: Rng + SeedableRng,
{
    active: Vec<usize>,
    rng: R,
}

#[derive(Deref, DerefMut)]
pub struct DartSampler2D<R = Xoshiro256StarStar>(RandomSamplerBase<2, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct DartSampler3D<R = Xoshiro256StarStar>(RandomSamplerBase<3, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct DartSamplerND<const N: usize, R = Xoshiro256StarStar>(RandomSamplerBase<N, R>)
where
    R: Rng + SeedableRng;

impl<R> Default for DartSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSampler2D(RandomSamplerBase::default())
    }
}

impl<R> Default for DartSampler3D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSampler3D(RandomSamplerBase::default())
    }
}

impl<const N: usize, R> Default for DartSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSamplerND(RandomSamplerBase::default())
    }
}

fn new_state<const N: usize, P, R>(
    sampler: &impl Deref<Target = RandomSamplerBase<N, R>>,
    _params: &P,
    grid: &P::Grid,
) -> DartState<R>
where
    P: Params<N>,
    R: Rng + SeedableRng,
{
    let mut rng = match sampler.random.seed {
        None => R::from_os_rng(),
        Some(seed) => R::seed_from_u64(seed),
    };
    let mut active: Vec<usize> = (0..grid.cells.len()).collect();
    active.shuffle(&mut rng);
    DartState { active, rng }
}

impl<R> Sampler<2> for DartSampler2D<R>
where
    R: Rng + SeedableRng,
{
    type Params = Params2D;
    type State = DartState<R>;

    fn new_state(&self, _params: &Params2D, grid: &Grid2D) -> Self::State {
        new_state(self, _params, grid)
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

impl<R> Sampler<3> for DartSampler3D<R>
where
    R: Rng + SeedableRng,
{
    type Params = Params3D;
    type State = DartState<R>;

    fn new_state(&self, _params: &Params3D, grid: &Grid3D) -> Self::State {
        new_state(self, _params, grid)
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

impl<const N: usize, R> Sampler<N> for DartSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    type Params = ParamsND<N>;
    type State = DartState<R>;

    fn new_state(&self, _params: &ParamsND<N>, grid: &GridND<N>) -> Self::State {
        new_state(self, _params, grid)
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
