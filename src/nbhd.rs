use super::{Point, Poisson};
use crate::iter::ActiveSample;
use rand::{Rng, SeedableRng};

// TODO: Make this an iterator
pub trait NbhdSampler<const N: usize> {
    fn sample_nbhd<S: NbhdSampler<N>, R: Rng + SeedableRng>(
        sample: &ActiveSample,
        samples: &Vec<Point<N>>,
        poisson: &Poisson<N, S, R>,
        rng: &mut R,
    ) -> Point<N>;
}

pub struct StandardNbhdSampler<const N: usize>;
pub struct ParentalNbhdSampler<const N: usize>;

impl NbhdSampler<2> for StandardNbhdSampler<2> {
    fn sample_nbhd<S: NbhdSampler<2>, R: Rng + SeedableRng>(
        sample: &ActiveSample,
        samples: &Vec<Point<2>>,
        poisson: &Poisson<2, S, R>,
        rng: &mut R,
    ) -> Point<2> {
        let p = &samples[sample.idx];
        let s = 2.0
            * poisson.radius
            * rng
                .random_range(0.5f64.powf(poisson.cdf_exp)..=1.0)
                .powf(1.0 / poisson.cdf_exp);
        let theta = rng.random_range(0.0..std::f64::consts::TAU);
        return [p[0] + s * theta.cos(), p[1] + s * theta.sin()];
    }
}

impl NbhdSampler<3> for StandardNbhdSampler<3> {
    fn sample_nbhd<S: NbhdSampler<3>, R: Rng + SeedableRng>(
        sample: &ActiveSample,
        samples: &Vec<Point<3>>,
        poisson: &Poisson<3, S, R>,
        rng: &mut R,
    ) -> Point<3> {
        let p = &samples[sample.idx];
        let s = 2.0
            * poisson.radius
            * rng
                .random_range(0.5f64.powf(poisson.cdf_exp)..=1.0)
                .powf(1.0 / poisson.cdf_exp);
        let v: Point<3> = [
            rng.random_range(0.0..=1.0),
            rng.random_range(0.0..=1.0),
            rng.random_range(0.0..=1.0),
        ];
        let scale = s / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        return [
            p[0] + scale * v[0],
            p[1] + scale * v[1],
            p[2] + scale * v[2],
        ];
    }
}

impl NbhdSampler<2> for ParentalNbhdSampler<2> {
    fn sample_nbhd<S: NbhdSampler<2>, R: Rng + SeedableRng>(
        sample: &ActiveSample,
        samples: &Vec<Point<2>>,
        poisson: &Poisson<2, S, R>,
        rng: &mut R,
    ) -> Point<2> {
        let p = &samples[sample.idx];
        let s = 2.0
            * poisson.radius
            * rng
                .random_range(0.5f64.powf(poisson.cdf_exp)..=1.0)
                .powf(1.0 / poisson.cdf_exp);

        let theta = match sample.parent_idx {
            None => rng.random_range(0.0..std::f64::consts::TAU),
            Some(parent_idx) => {
                let parent = &samples[parent_idx];
                let d = [
                    (parent[0] - p[0]) / poisson.radius,
                    (parent[1] - p[1]) / poisson.radius,
                ];
                let dist2 = d[0] * d[0] + d[1] * d[1];
                let dist = dist2.sqrt();
                let alpha = d[1].atan2(d[0]);
                let outer = ((dist2 + 3.0) / (4.0 * dist)).acos();
                let inner = (dist / 2.0).acos();
                let beta = outer.min(inner);
                println!("{}", beta);
                rng.random_range((alpha + beta)..(alpha + std::f64::consts::TAU - beta))
            }
        };
        return [p[0] + s * theta.cos(), p[1] + s * theta.sin()];
    }
}
