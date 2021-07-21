import numpy as np
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
import seaborn as sb


class CustomDistr:
    def __init__(self, scs_distr: str, size: int) -> None:
        self._distr_title = scs_distr
        self._size = size

    def _generate_distr(self):
        if self._distr_title == "Normal":
            self._arr = scs.norm.rvs(size=self._size)
            self._arr.sort()

        if self._distr_title == "Cauchy":
            self._arr = scs.cauchy.rvs(size=self._size)
            self._arr.sort()

        if self._distr_title == "Laplace":
            self._arr = scs.laplace.rvs(size=self._size, scale=1 / math.sqrt(2), loc=0)
            self._arr.sort()
 
        if self._distr_title == "Poisson":
            self._arr = scs.poisson.rvs(10, size=self._size)
            self._arr.sort()

        if self._distr_title == "Uniform":
            self._arr = scs.uniform.rvs(size=self._size, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
            self._arr.sort()

    def __str__(self):
        return self._distr_title
