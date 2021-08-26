import math
from typing import Tuple

import numpy as np
import scipy.stats as scs
from tabulate import tabulate


class DistrManager:
    def __init__(self, alpha: float = 0.05) -> None:
        self._p = 1 - alpha

    def get_probability(
        self, distr: np.ndarray, limits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        p_array = np.array([])
        n_array = np.array([])

        for idx in range(-1, len(limits)):
            previous_cdf = 0 if idx == -1 else scs.norm.cdf(limits[idx])
            current_cdf = 1 if idx == len(limits) - 1 else scs.norm.cdf(limits[idx + 1])
            p_array = np.append(p_array, current_cdf - previous_cdf)

            if idx == -1:
                n_array = np.append(n_array, len(distr[distr <= limits[0]]))
            elif idx == len(limits) - 1:
                n_array = np.append(n_array, len(distr[distr >= limits[-1]]))
            else:
                n_array = np.append(
                    n_array,
                    len(distr[(distr <= limits[idx + 1]) & (distr >= limits[idx])]),
                )

        return n_array, p_array

    def print_table(
        self,
        n_array: np.ndarray,
        p_array: np.ndarray,
        limits: np.ndarray,
        size: int,
        format: str = "latex",
    ) -> None:
        result = np.divide(
            np.multiply((n_array - size * p_array), (n_array - size * p_array)),
            p_array * size,
        )
        rows = list()

        for idx in range(len(n_array)):
            if idx == 0:
                boarders = [-np.inf, np.around(limits[0], decimals=8)]
            elif idx == len(n_array) - 1:
                boarders = [np.around(limits[-1], decimals=8), np.inf]
            else:
                boarders = [
                    np.around(limits[idx - 1], decimals=8),
                    np.around(limits[idx], decimals=8),
                ]

            rows.append(
                [
                    idx + 1,
                    boarders,
                    n_array[idx],
                    np.around(p_array[idx], decimals=8),
                    np.around(p_array[idx] * size, decimals=8),
                    np.around(n_array[idx] - size * p_array[idx], decimals=8),
                    np.around(result[idx], decimals=8),
                ]
            )

        rows.append(
            [
                len(n_array) + 1,
                "-",
                np.sum(n_array),
                np.around(np.sum(p_array), decimals=8),
                np.around(np.sum(p_array * size), decimals=8),
                np.around(np.sum(n_array - size * p_array), decimals=8),
                np.around(np.sum(result), decimals=8),
            ]
        )

        print("\n", tabulate(rows, tablefmt=format), "\n")

    def generate(self, distr_name: str, size: int) -> None:
        if distr_name == "Normal":
            distr = np.random.normal(0, 1, size=100)
        elif distr_name == "Laplace":
            distr = scs.laplace.rvs(size=20, scale=1 / math.sqrt(2), loc=0)
        elif distr_name == "Uniform":
            distr = scs.uniform.rvs(size=20, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
        else:
            raise ValueError(f"Unexpected distribution: {distr_name}")

        k = math.ceil(1.72 * size ** (1 / 3))

        print(f"\n{distr_name} distribution:")
        print(f"mu = {np.around(np.mean(distr), decimals=8)}")
        print(f"sigma = {np.around(np.std(distr), decimals=8)}")
        print(f"k = {k}")
        print(f"chi_2 = {scs.chi2.ppf(self._p, k - 1)}")

        limits = np.linspace(-1.1, 1.1, num=k - 1)
        n_array, p_array = self.get_probability(distr, limits)
        self.print_table(n_array, p_array, limits, size)
