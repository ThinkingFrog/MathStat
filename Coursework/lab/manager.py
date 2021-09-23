import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class SignalManager:
    _signal_length: int
    _signal_part: int

    def __init__(self) -> None:
        self._signal_length = 1024
        self._signal_part = 17

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
        x, y = list(), list()

        for idx, subarr in enumerate(signal_data):
            x.append(np.mean(subarr))
            y.append(idx)

        plt.xlabel("Measuring number")
        plt.ylabel("Average signal value")
        plt.plot(y, x)

        self.draw_average(data_array)

    def draw_signal_interpolation(self, signal_data: List[List[float]]) -> None:
        pass
