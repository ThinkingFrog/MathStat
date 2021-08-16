from lab.distribution import DistrManager


def main():
    sizes = [20, 60, 100]
    koeffs = [0.5, 1, 2]
    distr_list = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]

    manager = DistrManager(sizes, koeffs)
    manager.draw_graphics(distr_list)
    manager.draw_kde(distr_list)
