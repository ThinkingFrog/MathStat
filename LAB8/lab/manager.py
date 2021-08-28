import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class SignalManager:
    _signal: np.ndarray
    _start: List
    _finish: List
    _types: List
    _zones: List
    _zone_types: List
    _signal_data: List[List]
    _signal_length: int

    def __init__(self) -> None:
        self._signal_length = 1024

    def _data(self, zones: List) -> List[List]:
        return [
            [self._signal[idx] for idx in range(borders[0], borders[1])]
            for borders in zones
        ]

    def _bin(self) -> int:
        return math.ceil(1.72 * (len(self._signal) ** (1 / 3)))

    def _inter(self, data: np.ndarray) -> float:
        sum = 0.0
        mean_signal = np.empty(data.shape[0])
        for i in range(len(data)):
            mean_signal[i] = np.mean(data[i])
        mean = np.mean(mean_signal)

        for i in range(len(mean_signal)):
            sum += (mean_signal[i] - mean) ** 2
        sum /= data.shape[0]

        return len(data) * sum

    def _intra(self, data: np.ndarray) -> float:
        result = 0.0
        for i in range(data.shape[0]):
            mean = np.mean(data[i])
            sum = 0.0
            for j in range(data.shape[1]):
                sum += (data[i][j] - mean) ** 2
            sum /= data.shape[0]
            result += sum

        return result / data.shape[0]

    def _fisher(self, signal_part: np.ndarray, k: int) -> float:
        data = np.reshape(signal_part, (k, math.ceil(signal_part.size / k)))
        f = self._inter(data) / self._intra(data)
        print("k = " + str(k))
        print("F = " + str(f))
        return f

    def _k(self, num: int) -> int:
        i = 4
        while num % i != 0:
            i += 1
        return i

    def read_signal(self, filepath: Path) -> None:
        with filepath.open("r") as file:
            data = np.asarray(
                [
                    [
                        float(el)
                        for el in line.replace("[", "").replace("]", "").split(", ")
                    ]
                    for line in file
                ]
            )

        data = np.reshape(data, (data.shape[1] // self._signal_length, self._signal_length))
        self._signal = data[1]

    def save_areas(self) -> None:
        hist = plt.hist(self._signal, bins=self._bin())

        x = [hist[0][idx] for idx in range(self._bin())]
        x_sorted = sorted(x)
        start_y = [hist[1][idx] for idx in range(self._bin())]
        finish_y = [hist[1][idx + 1] for idx in range(self._bin())]
        types = [0 for _ in range(self._bin())]

        for idx in range(self._bin()):
            if x[idx] == x_sorted[len(x) - 1]:
                types[idx] = "background"
            elif x[idx] == x_sorted[len(x) - 2]:
                types[idx] = "signal"
            else:
                types[idx] = "переход"

        self._start = start_y
        self._finish = finish_y
        self._types = types

    def save_zones(self) -> None:
        signal_types = [0 for _ in range(len(self._signal))]
        zones, zones_type = list(), list()

        for i in range(len(self._signal)):
            for j in range(len(self._types)):
                if (self._signal[i] >= self._start[j]) and (
                    self._signal[i] <= self._finish[j]
                ):
                    signal_types[i] = self._types[j]

        currentType = signal_types[0]
        self._start = 0

        for idx in range(len(signal_types)):
            if currentType != signal_types[idx]:
                zones_type.append(currentType)
                zones.append([self._start, idx])
                self._start = idx
                currentType = signal_types[idx]

        if currentType != zones_type[-1]:
            zones_type.append(currentType)
            zones.append([self._start, len(self._signal) - 1])

        self._zones = zones
        self._zone_types = zones_type
        self._signal_data = self._data(zones)

    def print_params(self) -> None:
        fishers = list()
        for i in range(len(self._zones)):
            start = self._zones[i][0]
            finish = self._zones[i][1]
            k = self._k(finish - start)

            while k == finish - start:
                finish += 1
                k = self._k(finish - start)

            fishers.append(self._fisher(self._signal[start:finish], k))

        print(f"Zones: {self._zones}")

    def draw_signal(self) -> None:
        plt.title("Signal")
        plt.plot(range(len(self._signal)), self._signal, "blue")
        plt.grid()
        plt.show()

    def draw_hist(self) -> None:
        plt.hist(self._signal, bins=self._bin(), color="blue")
        plt.grid()
        plt.title("Signal histogram")
        plt.show()

    def draw_areas(self) -> None:
        plt.title("Area types plot")
        plt.ylim([-0.5, 0])

        for idx, data in enumerate(self._zones):
            if self._zone_types[idx] == "background":
                color = "yellow"
            elif self._zone_types[idx] == "signal":
                color = "red"
            else:
                color = "green"
            plt.plot(
                [el for el in range(data[0], data[1])],
                self._signal_data[idx],
                color=color,
                label=self._zone_types[idx],
            )

        plt.grid()
        plt.legend()
        plt.show()
