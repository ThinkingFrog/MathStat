from typing import List, Tuple
import numpy as np
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.distributions.empirical_distribution import ECDF


class DistrManager:
    def __init__(self, sizes = List[int], koeffs = List[int]) -> None:
        self._count = 5
        self._sizes = sizes
        self._koeffs = koeffs

    def generate_distr(self, scs_distr: str, size: int) -> Tuple[List, List, List, np.linspace]:
        left, right = -4, 4

        if scs_distr == "Normal":
            arr = scs.norm.rvs(size=size)
            arr.sort()
            x = np.linspace(left, right, 1000)
            pdf = scs.norm.pdf(x)
            cdf = scs.norm.cdf(x)

        if scs_distr == "Cauchy":
            arr = scs.cauchy.rvs(size=size)
            arr.sort()
            x = np.linspace(left, right, 1000)
            pdf = scs.cauchy.pdf(x)
            cdf = scs.cauchy.cdf(x)

        if scs_distr == "Laplace":
            arr = scs.laplace.rvs(size=size, scale=1 / math.sqrt(2), loc=0)
            arr.sort()
            x = np.linspace(left, right, 1000)
            pdf = scs.laplace.pdf(x, loc=0, scale=1 / math.sqrt(2))
            cdf = scs.laplace.cdf(x, loc=0, scale=1 / math.sqrt(2))

        if scs_distr == "Poisson":
            arr = scs.poisson.rvs(10, size=size)
            arr.sort()
            left, right = 6, 14
            x = np.linspace(left, right, 1000)
            pdf = scs.poisson(10).pmf(x)
            cdf = scs.poisson(10).cdf(x)

        if scs_distr == "Uniform":
            arr = scs.uniform.rvs(size=size, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
            arr.sort()
            x = np.linspace(left, right, 1000)
            pdf = scs.uniform.pdf(x, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
            cdf = scs.uniform.cdf(x, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
        
        return arr, pdf, cdf, x, left, right

    def draw_graphics(self, distr_list: List[str]):
        sb.set_style('whitegrid')

        for distr in distr_list:
            figures, axs = plt.subplots(ncols=3, figsize=(15, 5))

            for idx, size in enumerate(self._sizes):
                sample, pdf, cdf, x, left, right = self.generate_distr(distr, size)
                ecdf = ECDF(sample)

                axs[idx].plot(x, cdf, color='red', label='cdf')
                axs[idx].plot(x, ecdf(x), color='blue', label='ecdf')
                axs[idx].legend(loc='lower right')
                axs[idx].set(xlabel='x', ylabel='F(x)')
                axs[idx].set_title(f"n = {size}")
            figures.suptitle(f"{distr} distribution")
            plt.show()

    def draw_kde(self, distr_list: List[str]):
        sb.set_style('whitegrid')
        
        for distr in distr_list:
            for idx, size in enumerate(self._sizes):
                figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
                sample, pdf, cdf, x, left, right = self.generate_distr(distr, size)

                for idx, koeff in enumerate(self._koeffs):
                    axs[idx].plot(x, pdf, color="red", label="pdf")
                    sb.kdeplot(data=sample, bw_method="silverman", bw_adjust=koeff, ax=axs[idx],
                                fill=True, linewidth=0, label="kde")
                    axs[idx].legend(loc="upper right")
                    axs[idx].set(xlabel="x", ylabel="f(x)")
                    axs[idx].set_xlim([left, right])
                    axs[idx].set_title("h = " + str(koeff))
                figures.suptitle(f"{distr} KDE n = {size}")
                plt.show()
