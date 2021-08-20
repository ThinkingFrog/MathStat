import statistics
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as scs
from matplotlib.patches import Ellipse


class DistrManager:
    def __init__(self, sizes: List[int], rhos: List[float], times: int) -> None:
        self._count = 5
        self._sizes = sizes
        self._rhos = rhos
        self._times = times

    def _generate_normal(self, size: int, rho: float) -> np.ndarray:
        arr = scs.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size)
        arr.sort()

        return arr

    def _generate_normal_mixed(self, size: int) -> np.ndarray:
        arr = 0.9 * scs.multivariate_normal.rvs(
            [0, 0], [[1, 0.9], [0.9, 1]], size
        ) + 0.1 * scs.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
        arr.sort()

        return arr

    def _quadrant_coeff(self, x, y) -> float:
        size = len(x)
        x_med = np.median(x)
        y_med = np.median(y)
        n = [0, 0, 0, 0]

        for i in range(size):
            if x[i] >= x_med and y[i] >= y_med:
                n[0] += 1
            elif x[i] < x_med and y[i] >= y_med:
                n[1] += 1
            elif x[i] < x_med:
                n[2] += 1
            else:
                n[3] += 1

        return (n[0] + n[2] - n[1] - n[3]) / size

    def get_coeff_stats(
        self, distribution: str, size: int, rho: float = 0
    ) -> Tuple[List, List, List]:
        pearson, spearman, quadrant = list(), list(), list()

        for _ in range(self._times):
            if distribution == "Normal":
                distr = self._generate_normal(size, rho)
            if distribution == "Mixed":
                distr = self._generate_normal_mixed(size)

            x, y = distr[:, 0], distr[:, 1]
            pearson.append(scs.pearsonr(x, y)[0])
            spearman.append(scs.spearmanr(x, y)[0])
            quadrant.append(self._quadrant_coeff(x, y))

        mean = [
            np.around(np.median(pearson), decimals=4),
            np.around(np.median(spearman), decimals=4),
            np.around(np.median(quadrant), decimals=4),
        ]
        sq_mean = [
            np.around(np.median(list(map(lambda x: x ** 2, pearson))), decimals=4),
            np.around(np.median(list(map(lambda x: x ** 2, spearman))), decimals=4),
            np.around(np.median(list(map(lambda x: x ** 2, quadrant))), decimals=4),
        ]
        disp = [
            np.around(statistics.variance(pearson), decimals=4),
            np.around(statistics.variance(spearman), decimals=4),
            np.around(statistics.variance(quadrant), decimals=4),
        ]

        return mean, sq_mean, disp

    def build_ellipse(self, x, y, ax) -> None:
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        rad_x = np.sqrt(1 + pearson)
        rad_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0), width=rad_x * 2, height=rad_y * 2, facecolor="none", edgecolor="red"
        )

        scale_x = np.sqrt(cov[0, 0]) * 3
        mean_x = np.mean(x)

        scale_y = np.sqrt(cov[1, 1]) * 3
        mean_y = np.mean(y)

        transform = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transform + ax.transData)
        ax.add_patch(ellipse)

    def draw(self, size: int) -> None:
        fig, ax = plt.subplots(1, 3)
        titles = [f"rho = {rho}" for rho in self._rhos]

        for i in range(len(self._rhos)):
            sample = self._generate_normal(size, self._rhos[i])
            x, y = sample[:, 0], sample[:, 1]
            self.build_ellipse(x, y, ax[i])
            ax[i].grid()
            ax[i].scatter(x, y, s=5)
            ax[i].set_title(titles[i])
        plt.suptitle(f"Size {size}")
        plt.show()
