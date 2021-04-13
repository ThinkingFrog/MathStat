import numpy as np
import scipy.stats as scs
import math
from typing import List, Tuple

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


    def z_tr(self) -> float:
        r = int(len(self._arr) / 4)
        sum = 0
        for i in range(r + 1, len(self._arr) - r + 1):
            sum += self._arr[i]
        return (1 / (len(self._arr) - 2 * r)) * sum


    def count_ave_stats(self, times) -> Tuple[List[int], List[int]]:
        list_mean = list()
        list_median = list()
        list_zr = list()
        list_zq = list()
        list_ztr = list()

        E_list = list()
        D_list = list()
        E_plus_sqrt_D = list()
        E_minus_sqrt_D = list()

        for idx in range(times):
            self._generate_distr()

            list_mean.append(np.mean(self._arr))
            list_median.append(np.median(self._arr))
            list_zr.append((self._arr[0] + self._arr[-1]) / 2)
            list_zq.append((self._arr[math.ceil(len(self._arr) * 0.25)] + self._arr[math.ceil(len(self._arr) * 0.75)]) / 2)
            list_ztr.append(self.z_tr())
        
        for item in [list_mean, list_median, list_zr, list_zq, list_ztr]:
            E_list.append(round(np.mean(item), 6))
            D_list.append(round(np.std(item) ** 2, 6))
            E_plus_sqrt_D.append(round(np.mean(item) + math.sqrt(np.std(item) ** 2), 6))
            E_minus_sqrt_D.append(round(np.mean(item) - math.sqrt(np.std(item) ** 2), 6))

        return E_list, D_list, E_plus_sqrt_D, E_minus_sqrt_D