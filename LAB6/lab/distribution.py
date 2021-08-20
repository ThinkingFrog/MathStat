from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from scipy.optimize import minimize


class DistrManager:
    def __init__(
        self, left_border: int = -1.8, right_border: int = 2, step: int = 0.2
    ) -> None:
        self._left_border = left_border
        self._right_border = right_border
        self._step = step

    def get_range(self) -> np.ndarray:
        return np.arange(self._left_border, self._right_border, self._step)

    def eval_x(self, x: float) -> float:
        return 2 + 2 * x

    def get_relation(self, x: List[float]) -> List[float]:
        return [self.eval_x(x_i) + scs.norm.rvs(0, 1) for x_i in x]

    def mess_relation(self, y: List[float]) -> List[float]:
        print("\nAdd error to relation\n")
        y[0] += 10
        y[-1] -= 10
        return y

    def _get_mnk_params(self, x: float, y: float) -> Tuple[float, float]:
        beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (
            np.mean(x ** 2) - np.mean(x) ** 2
        )
        beta_0 = np.mean(y) - beta_1 * np.mean(x)
        return beta_0, beta_1

    def mnk(self, x: float, y: float) -> List[float]:
        beta_0, beta_1 = self._get_mnk_params(x, y)
        print(f"MNK:\t beta_0 = {beta_0}\t beta_1 = {beta_1}")
        return [beta_0 + beta_1 * element for element in x]

    def _minimize_mnm(self, x_0: Tuple[float, float], x: float, y: float) -> float:
        return sum(abs(y[idx] - x_0[0] - x_0[1] * x_val) for idx, x_val in enumerate(x))

    def _get_mnm_params(self, x: float, y: float) -> Tuple[float, float]:
        beta_0, beta_1 = self._get_mnk_params(x, y)
        minimized = minimize(
            self._minimize_mnm, [beta_0, beta_1], args=(x, y), method="SLSQP"
        )
        return minimized.x[0], minimized.x[1]

    def mnm(self, x: float, y: float) -> List[float]:
        beta_0, beta_1 = self._get_mnm_params(x, y)
        print(f"MNM:\t beta_0 = {beta_0}\t beta_1 = {beta_1}")
        return [beta_0 + beta_1 * element for element in x]

    def draw(self, x: float, y: float, name: str) -> None:
        y_mnk = self.mnk(x, y)
        y_mnm = self.mnm(x, y)

        mnk_dist = sum((self.eval_x(x)[i] - y_mnk[i]) ** 2 for i in range(len(y)))
        mnm_dist = sum(abs(self.eval_x(x)[i] - y_mnm[i]) for i in range(len(y)))
        print(f"MNK distance = {mnk_dist}\t MNM distance = {mnm_dist}")

        plt.plot(x, self.eval_x(x), color="red", label="Ideal")
        plt.plot(x, y_mnk, color="green", label="MNK")
        plt.plot(x, y_mnm, color="orange", label="MNM")
        plt.scatter(x, y, c="blue", label="Sample")
        plt.xlim([self._left_border, self._right_border])
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.show()
