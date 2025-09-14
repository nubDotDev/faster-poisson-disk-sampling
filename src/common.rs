use rand::{Rng, SeedableRng};
use std::{array, f64::consts::SQRT_2, marker::PhantomData};

pub(crate) type Point<const N: usize> = [f64; N];
pub(crate) type Idx<const N: usize> = [usize; N];

#[derive(Default, Clone, Copy)]
pub struct TwoD;
#[derive(Default, Clone, Copy)]
pub struct ThreeD;
#[derive(Default, Clone, Copy)]
pub struct ND;

pub struct Grid<const N: usize, T> {
    pub(crate) cell_len: f64,
    pub(crate) grid_dims: Idx<N>,
    pub(crate) cells: Vec<Option<usize>>,
    pub(crate) samples: Vec<Point<N>>,
    _t: PhantomData<T>,
}

pub type Grid2D = Grid<2, TwoD>;
pub type Grid3D = Grid<3, ThreeD>;
pub type GridND<const N: usize> = Grid<N, ND>;

pub trait GridImpl<const N: usize> {
    fn idx_to_ndidx(&self, idx: usize) -> Idx<N>;
    fn ndidx_to_idx(&self, ndidx: &Idx<N>) -> usize;
    fn point_to_ndidx(&self, p: &Point<N>) -> Idx<N>;
}

impl<const N: usize, T> Grid<N, T>
where
    Self: GridImpl<N>,
{
    #[inline(always)]
    pub fn point_to_idx(&self, p: &Point<N>) -> usize {
        self.ndidx_to_idx(&self.point_to_ndidx(p))
    }

    pub fn add_point(&mut self, p: &Point<N>) -> usize {
        let grid_idx = self.point_to_idx(p);
        let sample_idx = self.samples.len();
        self.cells[grid_idx] = Some(sample_idx);
        self.samples.push(*p);
        return sample_idx;
    }
}

impl GridImpl<2> for Grid2D {
    #[inline(always)]
    fn idx_to_ndidx(&self, idx: usize) -> Idx<2> {
        [idx % self.grid_dims[0], idx / self.grid_dims[0]]
    }

    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<2>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1]
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<2>) -> Idx<2> {
        [
            (p[0] / self.cell_len).floor() as usize,
            (p[1] / self.cell_len).floor() as usize,
        ]
    }
}

impl GridImpl<3> for Grid3D {
    #[inline(always)]
    fn idx_to_ndidx(&self, idx: usize) -> Idx<3> {
        [
            idx % self.grid_dims[0],
            (idx / self.grid_dims[0]) % self.grid_dims[1],
            idx / (self.grid_dims[0] * self.grid_dims[1]),
        ]
    }

    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<3>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1] + self.grid_dims[0] * self.grid_dims[1] * ndidx[2]
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<3>) -> Idx<3> {
        [
            (p[0] / self.cell_len).floor() as usize,
            (p[1] / self.cell_len).floor() as usize,
            (p[2] / self.cell_len).floor() as usize,
        ]
    }
}

impl<const N: usize> GridImpl<N> for GridND<N> {
    #[inline(always)]
    fn idx_to_ndidx(&self, idx: usize) -> Idx<N> {
        let mut idx = idx;
        array::from_fn(|i| {
            let ret = idx % self.grid_dims[i];
            idx /= self.grid_dims[i];
            ret
        })
    }

    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &Idx<N>) -> usize {
        let mut res = 0;
        let mut mul = 1;
        for i in 0..N {
            res += ndidx[i] * mul;
            mul *= self.grid_dims[i];
        }
        res
    }

    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<N>) -> Idx<N> {
        p.map(|x| (x / self.cell_len).floor() as usize)
    }
}

#[derive(Clone, Copy)]
pub struct Params<const N: usize, T> {
    pub(crate) dims: Point<N>,
    pub(crate) radius: f64,
    _t: PhantomData<T>,
}

pub(crate) type Params2D = Params<2, TwoD>;
pub(crate) type Params3D = Params<3, ThreeD>;
pub(crate) type ParamsND<const N: usize> = Params<N, ND>;

pub trait ParamsImpl<const N: usize, T> {
    fn grid(&self) -> Grid<N, T>;
    fn is_sample_valid(&self, p: &Point<N>, grid: &Grid<N, T>) -> bool;
}

impl<const N: usize, T> Default for Params<N, T> {
    fn default() -> Self {
        Params {
            dims: [1.0; N],
            radius: 0.1,
            _t: Default::default(),
        }
    }
}

impl ParamsImpl<2, TwoD> for Params2D {
    fn grid(&self) -> Grid2D {
        let cell_len = self.radius / SQRT_2;
        let grid_dims = [
            (self.dims[0] / cell_len).ceil() as usize,
            (self.dims[1] / cell_len).ceil() as usize,
        ];
        Grid2D {
            cell_len,
            grid_dims,
            cells: vec![None; grid_dims[0] * grid_dims[1]],
            samples: Vec::new(),
            _t: Default::default(),
        }
    }

    fn is_sample_valid(&self, p: &Point<2>, grid: &Grid2D) -> bool {
        if !(0.0..self.dims[0]).contains(&p[0]) || !(0.0..self.dims[1]).contains(&p[1]) {
            return false;
        }

        const FRACT_SQRT_2: f64 = SQRT_2 - 1.0;
        const ONE_MINUS_FRACT_SQRT_2: f64 = 2.0 - SQRT_2;

        let ndidx = grid.point_to_ndidx(p);
        let p_rems = [
            (p[0] / grid.cell_len).fract(),
            (p[1] / grid.cell_len).fract(),
        ];
        let ranges = [
            ndidx[0].saturating_sub(1 + ((p_rems[0] <= FRACT_SQRT_2) as usize))
                ..=(ndidx[0] + 1 + ((p_rems[0] >= ONE_MINUS_FRACT_SQRT_2) as usize))
                    .min(grid.grid_dims[0] - 1),
            ndidx[1].saturating_sub(1 + ((p_rems[1] <= FRACT_SQRT_2) as usize))
                ..=(ndidx[1] + 1 + ((p_rems[1] >= ONE_MINUS_FRACT_SQRT_2) as usize))
                    .min(grid.grid_dims[1] - 1),
        ];

        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                match grid.cells[grid.ndidx_to_idx(&[i, j])] {
                    None => (),
                    Some(sample_idx) => {
                        let sample = &grid.samples[sample_idx];
                        let d = [sample[0] - p[0], sample[1] - p[1]];
                        if d[0] * d[0] + d[1] * d[1] < self.radius * self.radius {
                            return false;
                        }
                    }
                };
            }
        }

        return true;
    }
}

impl ParamsImpl<3, ThreeD> for Params3D {
    fn grid(&self) -> Grid3D {
        let cell_len = self.radius / 3.0f64.sqrt();
        let grid_dims = [
            (self.dims[0] / cell_len).ceil() as usize,
            (self.dims[1] / cell_len).ceil() as usize,
            (self.dims[2] / cell_len).ceil() as usize,
        ];
        Grid3D {
            cell_len,
            grid_dims,
            cells: vec![None; grid_dims[0] * grid_dims[1] * grid_dims[2]],
            samples: Vec::new(),
            _t: Default::default(),
        }
    }

    fn is_sample_valid(&self, p: &Point<3>, grid: &Grid3D) -> bool {
        if !(0.0..self.dims[0]).contains(&p[0])
            || !(0.0..self.dims[1]).contains(&p[1])
            || !(0.0..self.dims[2]).contains(&p[2])
        {
            return false;
        }

        let fract_sqrt_3: f64 = 3.0f64.sqrt() - 1.0;
        let one_minus_fract_sqrt_3: f64 = 2.0 - 3.0f64.sqrt();

        let ndidx = grid.point_to_ndidx(p);
        let p_rems = [
            (p[0] / grid.cell_len).fract(),
            (p[1] / grid.cell_len).fract(),
            (p[2] / grid.cell_len).fract(),
        ];
        let ranges = [
            ndidx[0].saturating_sub(1 + ((p_rems[0] <= fract_sqrt_3) as usize))
                ..=(ndidx[0] + 1 + ((p_rems[0] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[0] - 1),
            ndidx[1].saturating_sub(1 + ((p_rems[1] <= fract_sqrt_3) as usize))
                ..=(ndidx[1] + 1 + ((p_rems[1] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[1] - 1),
            ndidx[2].saturating_sub(1 + ((p_rems[2] <= fract_sqrt_3) as usize))
                ..=(ndidx[2] + 1 + ((p_rems[2] >= one_minus_fract_sqrt_3) as usize))
                    .min(grid.grid_dims[2] - 1),
        ];

        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                for k in ranges[2].clone() {
                    match grid.cells[grid.ndidx_to_idx(&[i, j, k])] {
                        None => (),
                        Some(sample_idx) => {
                            let sample = &grid.samples[sample_idx];
                            let d = [sample[0] - p[0], sample[1] - p[1]];
                            if d[0] * d[0] + d[1] * d[1] < self.radius * self.radius {
                                return false;
                            }
                        }
                    };
                }
            }
        }

        return true;
    }
}

impl<const N: usize> ParamsImpl<N, ND> for ParamsND<N> {
    fn grid(&self) -> GridND<N> {
        let cell_len = self.radius / (N as f64).sqrt();
        let grid_dims = self.dims.map(|x| (x / cell_len).ceil() as usize);
        GridND {
            cell_len,
            grid_dims,
            cells: vec![None; grid_dims.iter().product()],
            samples: Vec::new(),
            _t: Default::default(),
        }
    }

    fn is_sample_valid(&self, p: &Point<N>, grid: &GridND<N>) -> bool {
        for i in 0..N {
            if !(0.0..self.dims[i]).contains(&p[i]) {
                return false;
            }
        }

        let ndidx = grid.point_to_ndidx(p);
        let radius_quo = N.isqrt();
        let radius_rem = (N as f64).sqrt().fract();
        let p_rems = p.map(|x| (x / grid.cell_len).fract());
        let loidx: [usize; N] = array::from_fn(|i| {
            ndidx[i].saturating_sub(radius_quo + ((p_rems[i] <= radius_rem) as usize))
        });
        let hiidx: [usize; N] = array::from_fn(|i| {
            (ndidx[i] + radius_quo + ((p_rems[i] >= grid.cell_len - radius_rem) as usize))
                .min(grid.grid_dims[i] - 1)
        });

        let mut curr = loidx;
        'out: loop {
            match grid.cells[grid.ndidx_to_idx(&curr)] {
                None => (),
                Some(sample_idx) => {
                    let sample = &grid.samples[sample_idx];
                    let mut dist_sq = 0.0;
                    for i in 0..N {
                        let d = sample[i] - p[i];
                        dist_sq += d * d;
                    }
                    if dist_sq < self.radius * self.radius {
                        return false;
                    }
                }
            };

            for i in 0..N - 1 {
                if curr[i] < hiidx[i] {
                    curr[i] += 1;
                    continue 'out;
                }
                curr[i] = loidx[i];
            }
            if curr[N - 1] < hiidx[N - 1] {
                curr[N - 1] += 1;
                continue;
            }
            break;
        }

        return true;
    }
}

pub trait Sampler<const N: usize, T>: Default {
    type State;

    fn new_state(&self, params: &Params<N, T>, grid: &Grid<N, T>) -> Self::State;

    fn sample(
        &self,
        params: &Params<N, T>,
        grid: &mut Grid<N, T>,
        state: &mut Self::State,
    ) -> Option<Point<N>>;
}

pub struct RandomSpec {
    pub(crate) attempts: usize,
    pub(crate) seed: Option<u64>,
}

impl RandomSpec {
    pub(crate) fn new(attempts: usize) -> Self {
        RandomSpec {
            attempts,
            seed: None,
        }
    }
}

pub struct RandomSampler<R, T>
where
    R: Rng + SeedableRng,
{
    pub(crate) random: RandomSpec,
    pub(crate) _rng: PhantomData<R>,
    pub(crate) _t: PhantomData<T>,
}

pub trait HasRandom {
    fn get_random(&self) -> &RandomSpec;
    fn get_random_mut(&mut self) -> &mut RandomSpec;
}

impl<R, T> HasRandom for RandomSampler<R, T>
where
    R: Rng + SeedableRng,
{
    fn get_random(&self) -> &RandomSpec {
        &self.random
    }

    fn get_random_mut(&mut self) -> &mut RandomSpec {
        &mut self.random
    }
}

pub struct RandomState<R, T = usize>
where
    R: Rng + SeedableRng,
{
    pub(crate) active: Vec<T>,
    pub(crate) rng: R,
}

impl<R, T> RandomState<R, T>
where
    R: Rng + SeedableRng,
{
    pub fn new(sampler: &impl HasRandom) -> Self
    where
        R: Rng + SeedableRng,
    {
        RandomState {
            active: Vec::new(),
            rng: match sampler.get_random().seed {
                None => R::from_os_rng(),
                Some(seed) => R::seed_from_u64(seed),
            },
        }
    }
}
