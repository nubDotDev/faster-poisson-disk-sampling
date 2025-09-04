use faster_poisson::{ParentalNbhdSampler, Poisson, StandardNbhdSampler};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

fn main() {
    let n = 100;
    let num_attempts = (18..=30).step_by(3);
    let cdf_exps = [0.5, 1.0, 2.0];
    let dims = [6.0, 6.0];
    let radius = 0.1;

    let nf64 = n as f64;
    let mut rng = Xoshiro256StarStar::seed_from_u64(0xDEADBEEF);

    println!(
        "Averages taken over {n} trials on a(n) {} by {} grid with minimum distance {radius}...",
        dims[0], dims[1]
    );
    for attempts in num_attempts {
        println!("With {attempts} attempts:");
        for cdf_exp in cdf_exps {
            println!("  With CDF exponent {cdf_exp}:");
            let mut standard_avg = 0.0;
            let mut parental_avg = 0.0;
            for _ in 0..n {
                standard_avg += Poisson::<2, StandardNbhdSampler<2>>::new()
                    .use_dims(dims)
                    .use_radius(radius)
                    .use_attempts(attempts)
                    .use_cdf_exp(cdf_exp)
                    .use_seed(Some(rng.next_u64()))
                    .run()
                    .len() as f64;
                parental_avg += Poisson::<2, ParentalNbhdSampler<2>>::new()
                    .use_dims(dims)
                    .use_radius(radius)
                    .use_attempts(attempts)
                    .use_cdf_exp(cdf_exp)
                    .use_seed(Some(rng.next_u64()))
                    .run()
                    .len() as f64;
            }
            standard_avg /= nf64;
            parental_avg /= nf64;
            println!("    Standard: {standard_avg}");
            println!("    Parental: {parental_avg}");
        }
    }
}
