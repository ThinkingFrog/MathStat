from lab.distribution import CustomDistr

def main():
    for distr in ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]:
        for size in [10, 100, 1000]:
            obj = CustomDistr(distr, size)
            E, D = obj.count_ave_stats(1000)

            print(f"\n{str(obj)} with size {size}")
            print(f"E:\n{E}\n")
            print(f"D:\n{D}\n")
