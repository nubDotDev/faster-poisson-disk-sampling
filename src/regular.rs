use crate::{Point, Sampler, State};
use std::iter;

pub struct RegularState<const N: usize>(Box<dyn Iterator<Item = Point<N>>>);

impl<const N: usize> State<N, RegularSampler<N>> for RegularState<N> {
    fn new<P>(_sampler: &RegularSampler<N>, params: &P, _grid: &P::Grid) -> Self
    where
        P: crate::Params<N>,
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
}

#[derive(Default)]
pub struct RegularSampler<const N: usize> {}

impl<const N: usize> Sampler<N> for RegularSampler<N> {
    type State = RegularState<N>;

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
