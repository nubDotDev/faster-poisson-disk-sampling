use super::{Float, NDIdx, Point, Poisson};
use crate::NbhdSampler;
use rand::{Rng, SeedableRng};

pub struct ActiveSample {
    pub(crate) idx: usize,
    pub(crate) parent_idx: Option<usize>,
}

pub struct PoissonIterInner<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> {
    pub(crate) poisson: Poisson<N, S, R>,

    pub(crate) r2: Float,
    pub(crate) inv_cell_len: Float,
    pub(crate) grid_dims: [usize; N],

    pub(crate) samples: Vec<Point<N>>,
    pub(crate) grid: Vec<Option<usize>>,
    pub(crate) active: Vec<ActiveSample>,
}

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> PoissonIterInner<N, S, R>
where
    Self: PoissonIterImpl<N>,
{
    #[inline(always)]
    fn point_to_ndidx(&self, p: &Point<N>) -> NDIdx<N> {
        p.map(|x| (x * self.inv_cell_len).floor() as usize)
    }

    #[inline(always)]
    fn point_to_idx(&self, p: &Point<N>) -> usize {
        PoissonIterImpl::ndidx_to_idx(self, &self.point_to_ndidx(p))
    }

    fn add_point(&mut self, p: &Point<N>, parent_idx: Option<usize>) {
        let idx = self.samples.len();
        let grid_idx = self.point_to_idx(p);
        self.samples.push(*p);
        self.grid[grid_idx] = Some(idx);
        self.active.push(ActiveSample { idx, parent_idx });
    }
}

pub trait PoissonIterImpl<const N: usize> {
    fn ndidx_to_idx(&self, ndidx: &NDIdx<N>) -> usize;
    fn is_sample_valid(&self, p: &Point<N>) -> bool;
}

impl<S: NbhdSampler<2>, R: Rng + SeedableRng> PoissonIterImpl<2> for PoissonIterInner<2, S, R> {
    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &NDIdx<2>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1]
    }

    fn is_sample_valid(&self, p: &Point<2>) -> bool {
        for i in 0..2 {
            if p[i] > self.poisson.dims[i] || p[i] < 0.0 {
                return false;
            }
        }

        let ndidx = self.point_to_ndidx(p);
        let mut buff = [0; 2];
        let off = p.map(|x| ((x * self.inv_cell_len) % 1.0).round() as usize);
        for i in
            ndidx[0].saturating_sub(2 - off[0])..=(ndidx[0] + 1 + off[0]).min(self.grid_dims[0] - 1)
        {
            buff[0] = i;
            for j in ndidx[1].saturating_sub(2 - off[1])
                ..=(ndidx[1] + 1 + off[1]).min(self.grid_dims[1] - 1)
            {
                buff[1] = j;
                match self.grid[self.ndidx_to_idx(&buff)] {
                    None => continue,
                    Some(neighbor_sample_idx) => {
                        let ns = self.samples[neighbor_sample_idx];
                        let d = [p[0] - ns[0], p[1] - ns[1]];
                        if d[0] * d[0] + d[1] * d[1] < self.r2 {
                            return false;
                        }
                        continue;
                    }
                }
            }
        }

        return true;
    }
}

impl<S: NbhdSampler<3>, R: Rng + SeedableRng> PoissonIterImpl<3> for PoissonIterInner<3, S, R> {
    #[inline(always)]
    fn ndidx_to_idx(&self, ndidx: &NDIdx<3>) -> usize {
        ndidx[0] + self.grid_dims[0] * ndidx[1] + self.grid_dims[0] * self.grid_dims[1] * ndidx[2]
    }

    fn is_sample_valid(&self, p: &Point<3>) -> bool {
        for i in 0..3 {
            if p[i] > self.poisson.dims[i] || p[i] < 0.0 {
                return false;
            }
        }

        let ndidx = self.point_to_ndidx(p);
        let mut buff = [0; 3];
        for i in ndidx[0].saturating_sub(2)..=(ndidx[0] + 2).min(self.grid_dims[0] - 1) {
            buff[0] = i;
            for j in ndidx[1].saturating_sub(2)..=(ndidx[1] + 2).min(self.grid_dims[1] - 1) {
                buff[1] = j;
                for k in ndidx[2].saturating_sub(2)..=(ndidx[2] + 2).min(self.grid_dims[2] - 1) {
                    buff[2] = k;
                    match self.grid[self.ndidx_to_idx(&buff)] {
                        None => continue,
                        Some(neighbor_sample_idx) => {
                            let ns = self.samples[neighbor_sample_idx];
                            let d = [p[0] - ns[0], p[1] - ns[1], p[2] - ns[2]];
                            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < self.r2 {
                                return false;
                            }
                            continue;
                        }
                    }
                }
            }
        }

        return true;
    }
}

pub struct PoissonIter<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> {
    inner: PoissonIterInner<N, S, R>,
    rng: R,
}

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> PoissonIter<N, S, R>
where
    PoissonIterInner<N, S, R>: PoissonIterImpl<N>,
{
    pub(crate) fn new(poisson: &Poisson<N, S, R>) -> Self {
        let inv_cell_len = (poisson.dims.len() as f64).sqrt() / poisson.radius;
        let grid_dims = poisson.dims.map(|x| (x * inv_cell_len).ceil() as usize);

        let grid_len = grid_dims.iter().product();
        let grid = vec![None; grid_len];
        let mut samples = Vec::new();
        samples.reserve(grid_len);
        let mut active = Vec::new();
        active.reserve(grid_len);

        PoissonIter {
            inner: PoissonIterInner {
                poisson: poisson.clone(),

                r2: poisson.radius * poisson.radius,
                inv_cell_len,
                grid_dims,

                samples,
                grid,
                active,
            },

            rng: match poisson.seed {
                None => R::from_os_rng(),
                Some(seed) => R::seed_from_u64(seed),
            },
        }
    }

    fn initial_sample(&mut self) -> Point<N> {
        let p = match self.inner.poisson.initial_sample {
            None => self
                .inner
                .poisson
                .dims
                .map(|x| self.rng.random_range(0.0..=x)),
            Some(initial_sample) => initial_sample,
        };
        self.inner.add_point(&p, None);
        p
    }

    fn sample(&mut self) -> Option<Point<N>> {
        let active_idx = self.rng.random_range(0..self.inner.active.len());
        let p_opt;
        let sample_idx;
        {
            let mut it;
            {
                let sample = &self.inner.active[active_idx];
                sample_idx = sample.idx;
                it = S::sample_nbhd(sample, &self.inner, &mut self.rng)
                    .take(self.inner.poisson.attempts);
            }
            p_opt = loop {
                let next = it.next();
                match next {
                    None => break None,
                    Some(cand) => {
                        if self.inner.is_sample_valid(&cand) {
                            break next;
                        }
                    }
                }
            };
        }
        match p_opt {
            None => {
                self.inner.active.swap_remove(active_idx);
                return None;
            }
            Some(p) => {
                self.inner.add_point(&p, Some(sample_idx));
                return Some(p);
            }
        }
    }
}

impl<const N: usize, S: NbhdSampler<N>, R: Rng + SeedableRng> Iterator for PoissonIter<N, S, R>
where
    PoissonIterInner<N, S, R>: PoissonIterImpl<N>,
{
    type Item = Point<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.samples.is_empty() {
            return Some(self.initial_sample());
        }
        while !self.inner.active.is_empty() {
            let p = self.sample();
            match p {
                None => continue,
                Some(_) => return p,
            }
        }
        return None;
    }
}
