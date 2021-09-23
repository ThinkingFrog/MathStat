import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class SignalManager:
    _signal_length: int

    def __init__(self) -> None:
        self._signal_length = 1024

    def read_signal(self, filepath: Path) -> List[float]:
        signal: List[float] = list()
        with filepath.open("r") as file:
            for line_num, line in enumerate(file):
                if line_num == 0:
                    stop_pos = int(line.split()[-1].strip())
                    continue
                if line_num > self._signal_length:
                    break

                signal += list(map(lambda x: float(x.strip()), line.split()[1:]))
        return signal
