use crate::common::{Grid, HasRandom, Idx, Params, Random, Sampler};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256StarStar;
use std::{array, marker::PhantomData};

pub struct DartState<R>
where
    R: Rng + SeedableRng,
{
    active: Vec<usize>,
    rng: R,
}

pub struct DartSamplerBase<const N: usize, R>
where
    R: Rng + SeedableRng,
{
    pub(crate) random: Random,

    _rng: PhantomData<R>,
}

#[derive(Deref, DerefMut)]
pub struct DartSampler2D<R = Xoshiro256StarStar>(DartSamplerBase<2, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct DartSampler3D<R = Xoshiro256StarStar>(DartSamplerBase<3, R>)
where
    R: Rng + SeedableRng;

#[derive(Deref, DerefMut)]
pub struct DartSamplerND<const N: usize, R = Xoshiro256StarStar>(DartSamplerBase<N, R>)
where
    R: Rng + SeedableRng;

impl<const N: usize, R> HasRandom for DartSamplerBase<N, R>
where
    R: Rng + SeedableRng,
{
    fn get_random_mut(&mut self) -> &mut Random {
        &mut self.random
    }
}

impl<const N: usize, R> Default for DartSamplerBase<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSamplerBase {
            random: Random::new(6),
            _rng: Default::default(),
        }
    }
}

impl<R> Default for DartSampler2D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSampler2D(DartSamplerBase::default())
    }
}

impl<R> Default for DartSampler3D<R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSampler3D(DartSamplerBase::default())
    }
}

impl<const N: usize, R> Default for DartSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSamplerND(DartSamplerBase::default())
    }
}

fn new_state<const N: usize, P, R>(
    sampler: &impl Deref<Target = DartSamplerBase<N, R>>,
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
    type State = DartState<R>;

    fn new_state<P>(&self, _params: &P, grid: &P::Grid) -> Self::State
    where
        P: crate::common::Params<2>,
    {
        new_state(self, _params, grid)
    }

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<2>>
    where
        P: crate::common::Params<2>,
    {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx = [idx % grid.grid_dims[0], idx / grid.grid_dims[0]];
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
    type State = DartState<R>;

    fn new_state<P>(&self, _params: &P, grid: &P::Grid) -> Self::State
    where
        P: crate::common::Params<3>,
    {
        new_state(self, _params, grid)
    }

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<3>>
    where
        P: crate::common::Params<3>,
    {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx = [
                        idx % grid.grid_dims[0],
                        (idx / grid.grid_dims[0]) % grid.grid_dims[1],
                        idx / (grid.grid_dims[0] * grid.grid_dims[1]),
                    ];
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
    type State = DartState<R>;

    fn new_state<P>(&self, _params: &P, grid: &P::Grid) -> Self::State
    where
        P: crate::common::Params<N>,
    {
        new_state(self, _params, grid)
    }

    fn sample<P>(
        &self,
        params: &P,
        grid: &mut P::Grid,
        state: &mut Self::State,
    ) -> Option<crate::common::Point<N>>
    where
        P: crate::common::Params<N>,
    {
        loop {
            match state.active.pop() {
                None => return None,
                Some(idx) => {
                    let ndidx: Idx<N> = {
                        let mut idx = idx;
                        array::from_fn(|i| {
                            let ret = idx % grid.grid_dims[i];
                            idx /= grid.grid_dims[i];
                            ret
                        })
                    };
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
