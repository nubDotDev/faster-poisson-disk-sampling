# faster_poisson

This is a library that implements various methods of Poisson disk sampling.

Each method returns a set of points where no two are less than a fixed distance from each other.

## Usage

The easiest way to use this library is with the [`Poisson2D`], [`Poisson3D`], and [`PoissonND`] types.
You can generate points all at once with the `run` method, or lazily with the `iter` method.
To change parameters like the grid dimensions or minimum distance, [`Poisson`] uses a fluent interface pattern.

```rust
use faster_poisson::{Poisson2D, Poisson3D, PoissonND};

// Sample points from a 6 x 4 grid with minimum distance 0.5.
let poisson_2d = Poisson2D::new().dims([6.0, 4.0]).radius(0.5);
let samples_2d = poisson_2d.run();

// The default side length of the grid is 1.0 and the default radius is 0.1.
let poisson_3d = Poisson3D::new();
let samples_3d = poisson_3d.run();

// For dimensions higher than 3, use PoissonND.
let poisson_4d = PoissonND::<4>::new();
// Points are generated lazily, so this is fast.
let samples_4d_100: Vec<[f64; 4]> = poisson_4d.iter().take(100).collect();
```

For smaller 2D grids (< 1,000,000 points) it is probably faster to use [`PoissonDart2D`].

## Features

- `plotly`

    Use the [plotly](https://docs.rs/plotly/latest/plotly/) library to plot distributions with [`plot_2d`] and [`plot_3d`].

- `fourier`

    Use the [image](https://docs.rs/image/latest/image/) and [rustfft](https://docs.rs/rustfft/latest/rustfft/) libraries to generate 2D periodograms of distributions with [`fourier`].

    <img src="https://raw.githubusercontent.com/nubDotDev/faster-poisson-disk-sampling/refs/heads/main/assets/parental_fourier.png" width=250 style="border: 2px solid white;"/>
