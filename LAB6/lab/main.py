from lab.distribution import DistrManager


def main():
    manager = DistrManager()

    x = manager.get_range()
    y = manager.get_relation(x)
    manager.draw(x, y, "Distribution")
    y = manager.mess_relation(y)
    manager.draw(x, y, "Distribution with error")
