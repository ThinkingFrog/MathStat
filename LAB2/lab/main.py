from lab.distribution import CustomDistr

def main():
    for distr in ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]:
        print(distr)
        for size in [10, 100, 1000]:
            obj = CustomDistr(distr, size)
            E, D, E_plus_sqrt_D, E_minus_sqrt_D = obj.count_ave_stats(1000)

            print(f"size {size} & & & & &\\\\\\hline")
            print("$E(z)$ &", end=' ')
            print(' & '.join(map(str, E)), '\\\\\\hline')
            print("$D(z)$ &", end=' ')
            print(' & '.join(map(str, D)), '\\\\\\hline')
            print("$E + \sqrt D$ &", end=' ')
            print(' & '.join(map(str, E_plus_sqrt_D)), '\\\\\\hline')
            print("$E - \sqrt D$ &", end=' ')
            print(' & '.join(map(str, E_minus_sqrt_D)), '\\\\\\hline')
