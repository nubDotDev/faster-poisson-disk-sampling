from dataclasses import dataclass
import itertools

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Iterable


@dataclass
class SampleAttempt:
    parent: int | None
    sample: np.ndarray | None
    is_valid: bool


class PoissonDiskSampler:
    def __init__(
        self,
        dims: np.ndarray,
        r: float,
        k: int,
        cdf_exp: float = 2,
        enable_logging: bool = False,
    ):
        self.dims = dims
        self.d = len(dims)
        self.r = r
        self.k = k
        self.cdf_exp = cdf_exp
        self.enable_logging = enable_logging

        self.cell_dim: float = r / np.sqrt(self.d)
        self.grid_dims: np.ndarray = np.ceil(dims / self.cell_dim).astype(int)
        self.grid: np.ndarray = -np.ones(self.grid_dims, dtype=int)
        self.active: list[int] = []
        self.samples: list[np.ndarray] = []
        self.log: list[SampleAttempt] = []

    def _maybe_log(self, entry: SampleAttempt):
        if self.enable_logging:
            self.log.append(entry)

    def _point_to_indices(self, p: np.ndarray) -> np.ndarray:
        return (p // self.cell_dim).astype(int)

    def _add(self, p: np.ndarray):
        sample_idx = len(self.samples)
        self.active.append(sample_idx)
        self.samples.append(p)
        self.grid[*self._point_to_indices(p)] = sample_idx

    def _initial_sample(self) -> None:
        p = np.random.rand(self.d) * self.dims
        self._add(p)
        self._maybe_log(SampleAttempt(parent=None, sample=p, is_valid=True))

    def _sample_annulus(self, p: np.ndarray) -> np.ndarray:
        a = 0.5**self.cdf_exp
        s = (np.random.rand() * (1 - a) + a) ** (1 / self.cdf_exp)
        v = np.random.rand(self.d) * 2 - 1
        return (2 * self.r * s / np.linalg.norm(v)) * v + p

    def _neighbors_iter(self, p: np.ndarray) -> Iterable[np.ndarray]:
        grid_idx = self._point_to_indices(p)
        for delta in itertools.product(range(-2, 3), repeat=self.d):
            neighbor_grid_idx = grid_idx + delta
            if ((neighbor_grid_idx >= self.grid_dims) | (neighbor_grid_idx < 0)).any():
                continue
            neighbor_sample_idx = self.grid[*neighbor_grid_idx]
            if neighbor_sample_idx == -1:
                continue
            yield neighbor_sample_idx

    def _is_sample_valid(self, p: np.ndarray) -> bool:
        if ((p > self.dims) | (p < 0)).any():
            return False
        for neighbor_sample_idx in self._neighbors_iter(p):
            if np.linalg.norm(p - self.samples[neighbor_sample_idx]) < self.r:
                return False
        return True

    def _sample(self) -> int:
        active_idx = np.random.randint(len(self.active))
        sample_idx = self.active[active_idx]
        p = self.samples[sample_idx]
        for i in range(self.k):
            cand = self._sample_annulus(p)
            if self._is_sample_valid(cand):
                self._add(cand)
                self._maybe_log(
                    SampleAttempt(parent=sample_idx, sample=cand, is_valid=True)
                )
                return i + 1
            self._maybe_log(
                SampleAttempt(parent=sample_idx, sample=cand, is_valid=False)
            )
        self.active[active_idx] = self.active[-1]
        self.active.pop()
        return 0

    def run(self, silent: bool = True) -> None:
        self._initial_sample()
        if not silent:
            print(f"Samples: {len(self.samples)}, Active: {len(self.active)}", end="\r")
        while self.active:
            self._sample()
            if not silent:
                print(
                    f"Samples: {len(self.samples)}, Active: {len(self.active)}",
                    end="\r",
                )
        if not silent:
            print()

    def setup_plot(self) -> None:
        self.fig = plt.figure()
        if self.d == 2:
            self.ax = self.fig.add_subplot()
            self.ax.set_aspect("equal")
            self.ax.set_xlim(0, self.dims[0])
            self.ax.set_ylim(0, self.dims[1])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        elif self.d == 3:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_xlim(0, self.dims[0])
            self.ax.set_ylim(0, self.dims[1])
            self.ax.set_zlim(0, self.dims[2])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])
        else:
            raise ValueError("Only 2D or 3D samples are supported for plotting.")

    def plot(self) -> None:
        self.setup_plot()
        self.ax.scatter(*self.samples[0], color="green", s=30)
        if len(self.samples) > 1:
            self.ax.scatter(*np.array(self.samples[1:]).T, s=15)

        # import matplotlib.patches as mpatches

        # for p in self.samples:
        #     circle = mpatches.Circle(
        #         (p[0], p[1]), radius=1, fill=False, edgecolor="black", linewidth=0.7
        #     )
        #     self.ax.add_patch(circle)

    def _require_logging(f: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not self.enable_logging:
                raise ValueError(f"{f.__name__} requires enable_logging to be true.")
            return f(self, *args, **kwargs)

        return wrapper

    @_require_logging
    def plot_log(self, pause: float = 0.1) -> None:
        self.setup_plot()
        for entry in self.log:
            if entry.parent is None:
                self.ax.scatter(*entry.sample, color="green", s=30)
                plt.pause(pause)
            else:
                x = self.ax.scatter(*entry.sample, color="red", s=15)
                plt.pause(pause)
                x.remove()
                if entry.is_valid:
                    self.ax.scatter(*entry.sample, color="blue", s=15)

    @_require_logging
    def sample_count_history(self) -> list[int]:
        history = []
        curr = 0
        for entry in self.log:
            if entry.is_valid:
                curr += 1
            history.append(curr)
        return history

    def _nearest_neighbor_distance(self, p) -> float:
        return min(
            filter(
                lambda x: x != 0,
                map(
                    lambda neighbor_sample_idx: np.linalg.norm(
                        p - self.samples[neighbor_sample_idx]
                    ),
                    self._neighbors_iter(p),
                ),
            ),
            default=np.inf,
        )

    def nearest_neighbors_distances(self) -> list[float]:
        return list(map(self._nearest_neighbor_distance, self.samples))


dims = np.array([10, 10])
pds = PoissonDiskSampler(dims, 1, 30, enable_logging=True)
pds.run()
pds1 = PoissonDiskSampler(dims, 1, 30, cdf_exp=0.1, enable_logging=True)
pds1.run()

# pds.plot()
# pds.plot_log(pause=0.1)
# plt.show()

# plt.figure()
# plt.plot(pds.sample_count_history())
# plt.plot(pds1.sample_count_history())
# plt.xlabel("Sample Attempt")
# plt.ylabel("Number of Valid Samples")
# plt.title("Sample Count History")
# plt.show()


plt.figure()
plt.hist(pds.nearest_neighbors_distances())
plt.hist(pds1.nearest_neighbors_distances())
plt.xlabel("Distance to nearest neighbor")
plt.show()

# Test average length of pds.samples over 50 trials for different cdf_exp values
# cdf_exp_values = [0.01, 0.1, 1, 2, 5]
# trials = 10
# results = {}
# dims = np.array([10, 10, 10])
# r = 1
# k = 15
# for cdf_exp in cdf_exp_values:
#     sample_counts = []
#     for _ in range(trials):
#         pds = PoissonDiskSampler(dims, r, k, cdf_exp=cdf_exp)
#         pds.run(silent=True)
#         sample_counts.append(len(pds.samples))
#     avg = np.mean(sample_counts)
#     results[cdf_exp] = avg
#     print(f"cdf_exp={cdf_exp}: average samples = {avg:.2f}")
