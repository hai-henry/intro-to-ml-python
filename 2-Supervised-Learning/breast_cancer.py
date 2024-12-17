import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np


def main():
    cancer = load_breast_cancer()

    print("Cancer keys: \n", cancer.keys())
    print("Cancer dataset shape: ", cancer.data.shape)

    zipped = zip(map(str, cancer.target_names), map(int, np.bincount(cancer.target)))
    print("Sample counts per class: \n", {n: v for n, v in zipped})

    print("Cancer dataset features: \n", cancer.feature_names)


if __name__ == "__main__":
    main()
