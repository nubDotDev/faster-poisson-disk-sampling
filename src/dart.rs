// TODO: Specialize for 2D and 3D.

use crate::common::{Grid, Idx, Random, Sampler};
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

pub struct DartSamplerND<const N: usize, R = Xoshiro256StarStar>
where
    R: Rng + SeedableRng,
{
    pub(crate) attempts: usize,
    pub(crate) random: Random<R>,
}

impl<const N: usize, R> Default for DartSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        DartSamplerND {
            attempts: 6,
            random: Random::default(),
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
        let mut rng = match self.random.seed {
            None => R::from_os_rng(),
            Some(seed) => R::seed_from_u64(seed),
        };
        let mut active: Vec<usize> = (0..grid.cells.len()).collect();
        active.shuffle(&mut rng);
        DartState { active, rng }
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
                    for _ in 0..self.attempts {
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
