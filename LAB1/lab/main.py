from pathlib import Path

from lab.distribution import CustomDistr


def main():
    imgdir = Path("result")
    for distr in ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]:
        for size in [10, 50, 1000]:
            obj = CustomDistr(distr, size)

            Path(imgdir / distr).mkdir(parents=True, exist_ok=True)
            obj.save(imgdir / distr / f"size_{size}.png")
