from typing import List
from pathlib import Path

from numpy import sign

from lab.manager import SignalManager


def main():
    manager = SignalManager()

    data_dir = Path("data")

    signal_path = data_dir / "Sin_100MHz"
    level0_path = data_dir / "ZeroLine"
    level1_path = data_dir / "-0_5V"
    level2_path = data_dir / "-0_25V"
    level3_path = data_dir / "+0_5V"
    level4_path = data_dir / "+0_25V"

    signal_data: List[List[float]] = manager.read_signal(signal_path)
    level0_data: List[List[List[float]]] = list()
    level1_data: List[List[List[float]]] = list()
    level2_data: List[List[List[float]]] = list()
    level3_data: List[List[List[float]]] = list()
    level4_data: List[List[List[float]]] = list()

    for data_path, data_array in zip(
        [
            level0_path,
            level1_path,
            level2_path,
            level3_path,
            level4_path,
        ],
        [level0_data, level1_data, level2_data, level3_data, level4_data],
    ):
        for data_file in data_path.iterdir():
            data_array.append(manager.read_file(data_file))

    manager.draw_average(
        [level0_data, level1_data, level2_data, level3_data, level4_data]
    )

    manager.draw_average_with_signal(
        signal_data, [level0_data, level1_data, level2_data, level3_data, level4_data]
    )

    manager.draw_signal_interpolation(None)
