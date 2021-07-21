from lab.distribution import CustomDistr

def main():
    for distr in ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]:
        for size in [20, 100]:
            obj = CustomDistr(distr, size)
