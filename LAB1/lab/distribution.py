import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs


class CustomDistr:
    def __init__(self, scs_distr: str, size: int) -> None:
        self._distr_title = scs_distr

        if scs_distr == "Normal":
            self._dens = scs.norm()
            self._hist = scs.norm.rvs(size=size)
        if scs_distr == "Cauchy":
            self._dens = scs.cauchy()
            self._hist = scs.cauchy.rvs(size=size)
        if scs_distr == "Laplace":
            self._dens = scs.laplace(scale=1 / math.sqrt(2), loc=0)
            self._hist = scs.laplace.rvs(size=size, scale=1 / math.sqrt(2), loc=0)
        if scs_distr == "Poisson":
            self._dens = scs.poisson(10)
            self._hist = scs.poisson.rvs(10, size=size)
        if scs_distr == "Uniform":
            self._dens = scs.uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3))
            self._hist = scs.uniform.rvs(
                size=size, loc=-math.sqrt(3), scale=2 * math.sqrt(3)
            )

    def save(self, imgpath: Path) -> None:
        fig, ax = plt.subplots(1, 1)

        ax.hist(self._hist, density=True, histtype="stepfilled")
        if self._distr_title == "Poisson":
            x = np.arange(self._dens.ppf(0.01), self._dens.ppf(0.99))
        else:
            x = np.linspace(self._dens.ppf(0.01), self._dens.ppf(0.99), 100)

        if self._distr_title == "Poisson":
            ax.plot(x, self._dens.pmf(x), "r")
        else:
            ax.plot(x, self._dens.pdf(x), "r")

        ax.set_xlabel(f"{len(self._hist)}")
        ax.set_ylabel("density")
        ax.set_title(self._distr_title)
        plt.grid()

        try:
            plt.savefig(imgpath)
        except PermissionError:
            pass
