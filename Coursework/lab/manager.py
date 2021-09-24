from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SignalManager:
    _signal_length: int
    _signal_part: int

    def __init__(self) -> None:
        self._signal_length = 1024
        self._signal_part = 17

    def _interpolate(
        self, x: Tuple[float, float], y: Tuple[float, float], x_val: float
    ) -> float:
        return y[0] + (x_val - x[0]) / (x[1] - x[0]) * (y[1] - y[0])

    def read_signal(self, dirpath: Path) -> List[float]:
        for idx, data_file in enumerate(dirpath.iterdir()):
            if idx == self._signal_part:
                return self.read_file(data_file)

    def read_file(self, filepath: Path) -> List[List[float]]:
        signal: List[List[float]] = list()
        with filepath.open("r") as file:
            for line_num, line in enumerate(file):
                if line_num == 0:
                    stop_pos = int(line.split()[-1].strip())
                    continue
                if line_num > self._signal_length:
                    break

                signal.append(list(map(lambda x: float(x.strip()), line.split()[1:])))
        return signal

    def get_average(self, data_array: List[List[List[List[float]]]]) -> np.ndarray:
        ave_arrays = list()

        for data in data_array:
            averages = list()
            average_by_line = [[] for _ in range(self._signal_length)]

            for file_data in data:
                for line_idx, line_data in enumerate(file_data):
                    average_by_line[line_idx].append(np.mean(line_data))

            for ave in average_by_line:
                averages.append(np.mean(ave))
            ave_arrays.append(np.asarray(averages))

        return np.asarray(ave_arrays)

    def draw_average(self, data_array: List[List[List[List[float]]]]) -> None:
        for data in data_array:
            x, y = list(), list()
            average_by_line = [[] for _ in range(self._signal_length)]

            for file_data in data:
                for line_idx, line_data in enumerate(file_data):
                    average_by_line[line_idx].append(np.mean(line_data))

            for idx, ave in enumerate(average_by_line):
                x.append(np.mean(ave))
                y.append(idx)
            plt.xlabel("Measuring number")
            plt.ylabel("Average signal value")
            plt.plot(y, x)

        plt.show()

    def draw_average_with_signal(
        self, signal_data: List[List[float]], data_array: List[List[List[List[float]]]]
    ) -> None:
        signal_average_data = [np.mean(subarr) for subarr in signal_data]

        plt.xlabel("Measuring number")
        plt.ylabel("Average signal value")
        plt.plot(range(len(signal_average_data)), signal_average_data)

        self.draw_average(data_array)

    def _get_level_idx(self, val: float, averages: np.ndarray) -> int:
        if val < averages[0]:
            return 0

        if val > averages[-1]:
            return len(averages) - 2

        for idx in range(len(averages)):
            if val > averages[idx] and val < averages[idx + 1]:
                return idx

    def draw_signal_interpolation(
        self,
        signal_data: List[List[float]],
        averages: np.ndarray,
        levels: List[float],
    ) -> None:
        signal_interpolation = list()
        signal_average_data = [np.mean(subarr) for subarr in signal_data]

        for idx in range(len(signal_average_data)):
            level_idx = self._get_level_idx(signal_average_data[idx], averages[:, idx])
            signal_interpolation.append(
                self._interpolate(
                    [averages[level_idx, idx], averages[level_idx + 1, idx]],
                    [levels[level_idx], levels[level_idx + 1]],
                    signal_average_data[idx],
                )
            )

        plt.plot(range(len(signal_interpolation)), signal_interpolation)
        plt.show()

    def get_scaled(self, signal_data: List[List[float]]) -> List[float]:
        signal_average_data = [np.mean(subarr) for subarr in signal_data]
        up_bound = max(signal_average_data)
        lo_bound = min(signal_average_data)

        signal_scale_data = list()
        for val in signal_average_data:
            if val > 0:
                signal_scale_data.append(val / up_bound)
            elif val < 0:
                signal_scale_data.append(-val / lo_bound)

        return signal_scale_data

    def draw_scale(self, signal_data: List[List[float]]) -> None:
        signal_scale_data = self.get_scaled(signal_data)

        plt.plot(range(len(signal_scale_data)), signal_scale_data)
        plt.show()
