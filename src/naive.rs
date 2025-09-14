//! Implements a slightly optimized naive dart throwing algorithm.
//!
//! We divide the grid into cells of length radius/sqrt(N)
//! and mark each cell as inactive when a point is placed inside it.
//! The algorithm only tries to place points in active cells.
//! The resulting distribution is equivalent to true naive dart throwing.
//! This algorithm is very slow and should not be used in production.

use crate::common::{
    GridImpl, GridND, ND, ParamsImpl, ParamsND, Point, RandomSampler, RandomSpec, RandomState,
    Sampler,
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::array;

pub struct Naive;

pub type NaiveSampler<R = Xoshiro256StarStar> = RandomSampler<R, Naive>;

impl<R> Default for RandomSampler<R, Naive>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        RandomSampler {
            random: RandomSpec::new(100),
            _rng: Default::default(),
            _t: Default::default(),
        }
    }
}

impl<const N: usize, R> Sampler<N, ND> for NaiveSampler<R>
where
    R: Rng + SeedableRng,
{
    type State = RandomState<R>;

    fn new_state(&self, _params: &ParamsND<N>, grid: &GridND<N>) -> Self::State {
        let mut state = RandomState::new::<N>(self);
        state.active.extend(0..grid.cells.len());
        state
    }

    fn sample(
        &self,
        params: &ParamsND<N>,
        grid: &mut GridND<N>,
        state: &mut Self::State,
    ) -> Option<Point<N>> {
        for _ in 0..self.random.attempts {
            let (active_idx, ndidx, cell_dims) = 'out: loop {
                let active_idx = state.rng.random_range(0..state.active.len());
                let ndidx = grid.idx_to_ndidx(state.active[active_idx]);
                let mut cell_dims: Point<N> = array::from_fn(|_| grid.cell_len);

                // If the chosen grid cell is along a far edge,
                // then repick with probability proportional to its missing area.
                for i in 0..N {
                    if ndidx[i] == grid.grid_dims[i] - 1 {
                        cell_dims[i] = params.dims[i] % grid.cell_len;
                        if state.rng.random_range(0.0..grid.cell_len) > cell_dims[i] {
                            continue 'out;
                        }
                    }
                }

                break (active_idx, ndidx, cell_dims);
            };

            let cand = array::from_fn(|i| {
                ndidx[i] as f64 * grid.cell_len + state.rng.random_range(0.0..cell_dims[i])
            });
            if params.is_sample_valid(&cand, grid) {
                grid.add_point(&cand);
                state.active.swap_remove(active_idx);
                return Some(cand);
            }
        }

        None
    }
}
