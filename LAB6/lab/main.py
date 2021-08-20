import numpy as np
from lab.distribution import DistrManager


def main():
    manager = DistrManager()

    x = manager.get_range()
    y = manager.get_relation(x)
    manager.draw(x, y, "Распределение без возмущения")
    y[0] += 10
    y[-1] -= 10
    manager.draw(x, y, "Распределение с возмущением")
