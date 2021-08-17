from lab.distribution import DistrManager


def main():
    sizes = [20, 60, 100]
    rhos = [0, 0.5, 0.9]
    times = 1000

    manager = DistrManager(sizes, rhos, times)

    for size in sizes:
        for rho in rhos:
            mean, sq_mean, disp = manager.get_coeff_stats('Normal', size, rho)
            print(f"Normal\t Size = {size}\t Rho = {rho}\t Mean = {mean}\t Squares mean = {sq_mean}\t Dispersion = {disp}")

        mean, sq_mean, disp = manager.get_coeff_stats('Mixed', size, rho)
        print(f"Mixed\t Size = {size}\t Mean = {mean}\t Squares mean = {sq_mean}\t Dispersion = {disp}")

        manager.draw(size)
