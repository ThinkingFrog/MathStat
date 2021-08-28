from pathlib import Path

from lab.manager import SignalManager


def main():
    manager = SignalManager()

    manager.read_signal(Path("wave_ampl.txt"))
    manager.draw_signal()

    manager.save_areas()
    manager.save_zones()

    manager.draw_hist()
    manager.draw_areas()

    manager.print_params()
