use crate::{Point, Sampler, common::Params};
use std::iter;

pub struct RegularState<const N: usize>(Box<dyn Iterator<Item = Point<N>>>);

#[derive(Default)]
pub struct RegularSamplerND<const N: usize> {}

impl<const N: usize> Sampler<N> for RegularSamplerND<N> {
    type State = RegularState<N>;

    fn new_state<P>(&self, params: &P, _grid: &P::Grid) -> Self::State
    where
        P: Params<N>,
    {
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

    fn sample<P>(
        &self,
        _params: &P,
        _grid: &mut P::Grid,
        state: &mut RegularState<N>,
    ) -> Option<crate::Point<N>>
    where
        P: crate::Params<N>,
    {
        state.0.next()
    }
}
