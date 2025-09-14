//! Generates a regular grid of points with exactly `radius` between each of them.

use crate::{
    Point, Sampler,
    common::{GridND, ND, ParamsND},
};
use std::iter;

pub struct RegularState<const N: usize>(Box<dyn Iterator<Item = Point<N>>>);

#[derive(Default)]
pub struct RegularSampler<const N: usize> {}

impl<const N: usize> Sampler<N, ND> for RegularSampler<N> {
    type State = RegularState<N>;

    fn new_state(&self, params: &ParamsND<N>, _grid: &GridND<N>) -> Self::State {
        let radius = params.radius;
        let hiidx = params.dims.map(|x| (x / params.radius).floor() as usize);
        let mut curr = [0; N];
        RegularState(Box::new(iter::once([0.0; N]).chain(iter::from_fn(
            move || {
                for i in 0..N {
                    if curr[i] < hiidx[i] {
                        curr[i] += 1;
                        break;
                    }
                    if i == N - 1 {
                        return None;
                    }
                    curr[i] = 0;
                }
                Some(curr.map(|x| x as f64 * radius))
            },
        ))))
    }

    fn sample(
        &self,
        _params: &ParamsND<N>,
        _grid: &mut GridND<N>,
        state: &mut RegularState<N>,
    ) -> Option<crate::Point<N>> {
        state.0.next()
    }
}
