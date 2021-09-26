from typing import List
from pathlib import Path

import numpy as np

from lab.manager import SignalManager


def main():
    manager = SignalManager()

    data_dir = Path("data")

    signal_path = data_dir / "Sin_100MHz"
    signal_part = 17
    level0_path = data_dir / "ZeroLine"
    level1_path = data_dir / "-0_5V"
    level2_path = data_dir / "-0_25V"
    level3_path = data_dir / "+0_25V"
    level4_path = data_dir / "+0_5V"

    signal_data = manager.read_signal(signal_path, signal_part)
    level0_data = manager.read_dir(level0_path)
    level1_data = manager.read_dir(level1_path)
    level2_data = manager.read_dir(level2_path)
    level3_data = manager.read_dir(level3_path)
    level4_data = manager.read_dir(level4_path)

    signal_data = np.mean(signal_data, axis=1)
    level0_data = manager.get_average(level0_data)
    level1_data = manager.get_average(level1_data)
    level2_data = manager.get_average(level2_data)
    level3_data = manager.get_average(level3_data)
    level4_data = manager.get_average(level4_data)

    manager.plot([level0_data, level1_data, level2_data, level3_data, level4_data])

    manager.plot(
        [signal_data, level0_data, level1_data, level2_data, level3_data, level4_data]
    )

    levels = [-0.5, -0.25, 0, 0.25, 0.5]
    consts = np.array([level1_data, level2_data, level0_data, level3_data, level4_data])
    signal_data = manager.interpolation(signal_data, levels, consts)
    manager.plot([signal_data])
