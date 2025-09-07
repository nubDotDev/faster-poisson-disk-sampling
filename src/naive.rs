use crate::common::{Grid, Params, Point, Random, RandomSamplerBase, RandomState, Sampler};
use derive_more::with_trait::{Deref, DerefMut};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::array;

#[derive(Deref, DerefMut)]
pub struct NaiveSamplerND<const N: usize, R = Xoshiro256StarStar>(RandomSamplerBase<N, R>)
where
    R: Rng + SeedableRng;

impl<const N: usize, R> Default for NaiveSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        NaiveSamplerND(RandomSamplerBase {
            random: Random::new(100),
            _rng: Default::default(),
        })
    }
}

impl<const N: usize, R> Sampler<N> for NaiveSamplerND<N, R>
where
    R: Rng + SeedableRng,
{
    type State = RandomState<R>;

    fn new_state<P>(&self, _params: &P, grid: &P::Grid) -> Self::State
    where
        P: crate::common::Params<N>,
    {
        let mut state = RandomState::new(self.deref(), _params, grid);
        state.active.extend(0..grid.cells.len());
        state
    }

    fn sample<P>(&self, params: &P, grid: &mut P::Grid, state: &mut Self::State) -> Option<Point<N>>
    where
        P: Params<N>,
    {
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
                ndidx[i] as f64 * grid.cell_len + state.rng.random_range(0.0..=cell_dims[i])
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
