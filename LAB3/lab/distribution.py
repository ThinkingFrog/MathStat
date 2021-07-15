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

    def mustache(self):
        q_1, q_3 = np.quantile(self._arr, [0.25, 0.75])
        return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)

    def count_emissions(self):
        x1, x2 = self.mustache()
        filtered = [x for x in self._arr if x > x2 or x < x1]
        return len(filtered)

    def draw(self):
        sb.set_theme(style="whitegrid")
        sb.boxplot(data=self._arr, palette='Set1', orient='h')
        sb.despine(offset=10)
        
        plt.xlabel("x")
        plt.ylabel("n")
        plt.title(self._distr_title)
        plt.show()
        
        # plt.savefig("image/" + self._distr_title + str(self._size) + ".png")

    def emission_share(self, times):
        count = 0
        for idx in range(times):
            self._generate_distr()
            count += self.count_emissions()
        share = count / (self._size * times)
        print(f"{self._distr_title} of size {self._size}: Emission share is {share}")

    def boxplot(self, times):
        self._generate_distr()

        self.emission_share(times)

        self._generate_distr()
        self.draw()
