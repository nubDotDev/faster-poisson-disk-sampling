use faster_poisson::{Poisson2D, PoissonBridson2D};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

struct Result {
    avg_len: f64,
    attempts: usize,
    cdf_exp: f64,
    sampler: String,
}

fn main() {
    let n = 100;
    let num_attempts = (18..=30).step_by(3);
    let cdf_exps = [0.5, 1.0, 2.0];
    let dims = [6.0, 6.0];
    let radius = 0.1;

    let nf64 = n as f64;
    let mut rng = Xoshiro256StarStar::seed_from_u64(0xDEADBEEF);

    let mut results = Vec::<Result>::new();

    println!(
        "Averages taken over {n} trials on a(n) {} by {} grid with minimum distance {radius}...",
        dims[0], dims[1]
    );
    for attempts in num_attempts {
        println!("  With {attempts} attempts:");
        for cdf_exp in cdf_exps {
            println!("    With CDF exponent {cdf_exp}:");
            let mut bridson_avg = 0.0;
            let mut parental_avg = 0.0;
            for _ in 0..n {
                bridson_avg += PoissonBridson2D::new()
                    .dims(dims)
                    .radius(radius)
                    .attempts(attempts)
                    .cdf_exp(Some(cdf_exp))
                    .seed(Some(rng.next_u64()))
                    .run()
                    .len() as f64;
                parental_avg += Poisson2D::new()
                    .dims(dims)
                    .radius(radius)
                    .attempts(attempts)
                    .cdf_exp(Some(cdf_exp))
                    .seed(Some(rng.next_u64()))
                    .run()
                    .len() as f64;
            }
            bridson_avg /= nf64;
            parental_avg /= nf64;
            println!("      Standard: {bridson_avg}");
            println!("      Parental: {parental_avg}");
            results.push(Result {
                avg_len: bridson_avg,
                attempts,
                cdf_exp,
                sampler: String::from("Bridson"),
            });
            results.push(Result {
                avg_len: parental_avg,
                attempts,
                cdf_exp,
                sampler: String::from("Parental"),
            });
        }
    }

    results.sort_by(|a, b| f64::total_cmp(&a.avg_len, &b.avg_len));
    println!("\n\nRanking:");
    for (i, result) in results.iter().rev().enumerate() {
        println!(
            "  {i:2}) {:8}  attempts={}  cdf_exp={:<3} | {}",
            result.sampler, result.attempts, result.cdf_exp, result.avg_len
        );
    }
}
