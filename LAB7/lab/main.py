from lab.distribution import DistrManager


def main():
    manager = DistrManager()

    manager.generate("Normal", 100)
    manager.generate("Laplace", 20)
    manager.generate("Uniform", 20)
