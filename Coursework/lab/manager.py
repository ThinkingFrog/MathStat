from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from lab.lib.tolsolvty import tolsolvty


class SignalManager:
    _signal_length: int
    _subarray_size: int

    def __init__(self) -> None:
        self._signal_length = 1024
        self._subarray_size = 9

    def _interpolate(
        self, x: Tuple[float, float], y: Tuple[float, float], x_val: float
    ) -> float:
        return y[0] + (x_val - x[0]) / (x[1] - x[0]) * (y[1] - y[0])

    def _get_level_idx(self, val: float, averages: np.ndarray) -> int:
        if val < averages[0]:
            return 0

        if val > averages[-1]:
            return len(averages) - 2

        for idx in range(len(averages)):
            if val > averages[idx] and val < averages[idx + 1]:
                return idx

    def read_signal(self, dirpath: Path, signal_part: int) -> np.ndarray:
        for idx, data_file in enumerate(dirpath.iterdir()):
            if idx == signal_part:
                return self.read_file(data_file)

    def read_file(self, filepath: Path) -> np.ndarray:
        with filepath.open("r") as file:
            data = np.zeros((self._signal_length, self._subarray_size))
            for line_num, line in enumerate(file):
                if line_num == 0:
                    stop_pos = int(line.split()[-1].strip())
                    continue
                if line_num > self._signal_length:
                    break

                data[
                    (self._signal_length + line_num - stop_pos) % self._signal_length
                ] = list(map(lambda x: float(x.strip()), line.split()[1:]))
        return data

    def read_dir(self, dirpath: Path) -> np.ndarray:
        data = np.zeros(
            (len(list(dirpath.glob("*.txt"))), self._signal_length, self._subarray_size)
        )
        for idx, data_file in enumerate(dirpath.glob("*.txt")):
            data[idx] = self.read_file(data_file)
        return data

    def get_average(self, data_array: np.ndarray) -> np.ndarray:
        data_array = np.mean(data_array, axis=2)
        return np.mean(data_array, axis=0)

    def interpolation(self, data, dc, constants):
        result = np.zeros(len(data))

        for idx in range(len(data)):
            id = self._get_level_idx(data[idx], constants[:, idx])
            result[idx] = self._interpolate(
                [constants[id, idx], constants[id + 1, idx]],
                [dc[id], dc[id + 1]],
                data[idx],
            )

        return result

    def plot(self, data_arrays: List[np.ndarray]) -> None:
        for data in data_arrays:
            plt.plot(range(data.size), data)

        plt.show()
