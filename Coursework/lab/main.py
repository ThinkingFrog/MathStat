from typing import List
from pathlib import Path

import numpy as np

from lab.manager import SignalManager


def main():
    manager = SignalManager()

    data_dir = Path("data")

    signal_path = data_dir / "Sin_100MHz"
    signal_part = 7
    level0_path = data_dir / "ZeroLine"
    level1_path = data_dir / "-0_5V"
    level2_path = data_dir / "-0_25V"
    level3_path = data_dir / "+0_25V"
    level4_path = data_dir / "+0_5V"

    signal_data = manager.read_signal(signal_path, signal_part)
    all_signals_data = manager.read_dir(signal_path)
    level0_data = manager.read_dir(level0_path)
    level1_data = manager.read_dir(level1_path)
    level2_data = manager.read_dir(level2_path)
    level3_data = manager.read_dir(level3_path)
    level4_data = manager.read_dir(level4_path)

    signal_data = np.mean(signal_data, axis=1)
    all_signals_data = manager.get_average(all_signals_data)
    level0_data = manager.get_average(level0_data)
    level1_data = manager.get_average(level1_data)
    level2_data = manager.get_average(level2_data)
    level3_data = manager.get_average(level3_data)
    level4_data = manager.get_average(level4_data)

    manager.plot(
        [level0_data, level1_data, level2_data, level3_data, level4_data],
        title="Amplitudes",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    manager.plot(
        [signal_data, level0_data, level1_data, level2_data, level3_data, level4_data],
        title="Amplitudes with signal",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    levels = [-0.5, -0.25, 0, 0.25, 0.5]
    levels_data = np.array(
        [level1_data, level2_data, level0_data, level3_data, level4_data]
    )
    manager.plot_regression_coeffs(
        levels,
        [
            level1_data[0],
            level2_data[0],
            level0_data[0],
            level3_data[0],
            level4_data[0],
        ],
    )
    signal_data = manager.get_interpolated(signal_data, levels, levels_data)
    all_signals_data = manager.get_interpolated(all_signals_data, levels, levels_data)
    manager.plot(
        [signal_data],
        title="Interpolated signal",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    sin_data = manager.get_sin(signal_data)
    all_sin_data = manager.get_sin(all_signals_data)
    manager.plot(
        [sin_data],
        title="Interpolated signal on [0, 1] segment",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    arcsin_data = np.arcsin(sin_data)
    manager.plot(
        [arcsin_data],
        title="Signal arcsinus",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    elements = manager.get_partition(arcsin_data)
    [ampl, a1, b1, a2, b2] = manager.get_asin_amp(arcsin_data, elements)
    manager.plot_signal_with_lines(
        arcsin_data,
        a1,
        b1,
        a2,
        b2,
        50,
        title="Signal amplitude",
        xlabel="Measurement",
        ylabel="Amplitude value",
    )

    signal_data = manager.set_amplitude(sin_data, ampl)
    all_signals_data = manager.set_amplitude(all_sin_data, ampl)

    manager.plot_time_delta(signal_data, 40, title="Time periods", xlabel="Time")
    manager.hist(all_signals_data, title="Time histogram", xlabel="Time (ns)")
