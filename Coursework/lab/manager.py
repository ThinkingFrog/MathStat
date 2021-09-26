import math
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

    def get_interpolated(self, data, levels, levels_data) -> np.ndarray:
        interpolated = np.zeros(len(data))

        for idx in range(len(data)):
            level_idx = self._get_level_idx(data[idx], levels_data[:, idx])
            interpolated[idx] = self._interpolate(
                [levels_data[level_idx, idx], levels_data[level_idx + 1, idx]],
                [levels[level_idx], levels[level_idx + 1]],
                data[idx],
            )

        return interpolated

    def get_sin(self, data):
        sin_data = np.divide(data, max(abs(max(data)), abs(min(data))))
        sin_data = np.add(sin_data, 1)
        sin_data = np.divide(sin_data, max(sin_data))
        return sin_data

    def get_partition(self, data):
        elements = [[], []]

        if data[0] < data[1]:
            elements[0].append(0)
        else:
            elements[1].append(0)

        for idx in range(1, len(data)):
            if data[idx] >= data[idx - 1]:
                elements[0].append(idx)
            else:
                elements[1].append(idx)

        return elements

    def get_asin_amp(self, bin_val, ids):
        dy = 0.015
        di = 1 / 2
        A2_bot = np.zeros((len(ids[1]), 3))
        A2_top = np.zeros((len(ids[1]), 3))
        B2_bot = np.zeros((len(ids[1]), 1))
        B2_top = np.zeros((len(ids[1]), 1))

        A1_bot = np.zeros((len(ids[0]), 3))
        A1_top = np.zeros((len(ids[0]), 3))
        B1_bot = np.zeros((len(ids[0]), 1))
        B1_top = np.zeros((len(ids[0]), 1))

        count = 0

        for i in range(len(ids[0])):
            if i != 0 and ids[0][i] - ids[0][i - 1] > 2:
                count += 1

            A1_bot[i, 0] = ids[0][i] - di + 1
            A1_bot[i, 1] = 1
            A1_bot[i, 2] = count
            B1_bot[i, 0] = bin_val[ids[0][i]] - dy * abs(bin_val[ids[0][i]])

            A1_top[i, 0] = ids[0][i] + di + 1
            A1_top[i, 1] = 1
            A1_top[i, 2] = count
            B1_top[i, 0] = bin_val[ids[0][i]] + dy * abs(bin_val[ids[0][i]])

        count = 0

        for i in range(len(ids[1])):
            if i != 0 and ids[1][i] - ids[1][i - 1] > 2:
                count += 1

            A2_bot[i, 0] = ids[1][i] - di
            A2_bot[i, 1] = 1
            A2_bot[i, 2] = count
            B2_bot[i, 0] = bin_val[ids[1][i]] - dy * abs(bin_val[ids[1][i]])

            A2_top[i, 0] = ids[1][i] + di
            A2_top[i, 1] = 1
            A2_top[i, 2] = count
            B2_top[i, 0] = bin_val[ids[1][i]] + dy * abs(bin_val[ids[1][i]])

        [tolmax, argmax, envs, ccode] = tolsolvty(A1_bot, A1_top, B1_bot, B1_top)
        a1 = argmax[0]
        b1 = argmax[1]
        [tolmax, argmax, envs, ccode] = tolsolvty(A2_bot, A2_top, B2_bot, B2_top)
        a2 = argmax[0]
        b2 = argmax[1]
        y = abs((b1 * a2 - b2 * a1) / (a2 - a1))

        return [y, a1, b1, a2, b2]

    def set_amplitude(self, data: np.ndarray, coefficient: float) -> np.ndarray:
        new_data = np.divide(data, max(abs(max(data)), abs(min(data))))
        return np.multiply(new_data, coefficient)

    def plot(
        self,
        data_arrays: List[np.ndarray],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        for data in data_arrays:
            plt.plot(range(data.size), data)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_signal_with_lines(
        self,
        signal_data: np.ndarray,
        a1: float,
        b1: float,
        a2: float,
        b2: float,
        num_of_dots: int,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        x = np.zeros(num_of_dots)
        y1 = np.zeros(num_of_dots)
        y2 = np.zeros(num_of_dots)
        for i in range(num_of_dots):
            x[i] = i
            y1[i] = a1 * x[i] + b1
            y2[i] = a2 * x[i] + b2

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y1, "g")
        plt.plot(x, y2, "r")
        self.plot([signal_data], title)

    def plot_time_delta(
        self,
        data: np.ndarray,
        num_of_dots: int,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        t_i = np.zeros(num_of_dots)
        for idx in range(num_of_dots):
            t_i[idx] = (data[idx + 1] - data[idx]) / (2 * math.pi)
        t_i = np.add(t_i, abs(min(t_i)))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlabel("time(ps)")
        plt.scatter(t_i, [1 for _ in range(num_of_dots)], marker=".")
        plt.show()

    def hist(
        self,
        data: np.ndarray,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        y = [el / (2 * math.pi) for el in data]

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlabel("time")
        plt.hist(y, bins=16)
        plt.show()
