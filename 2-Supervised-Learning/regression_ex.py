import mglearn
import matplotlib.pyplot as plt


def main():
    X, y = mglearn.datasets.make_wave(n_samples=40)

    plt.plot(X, y, "o")
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()


if __name__ == "__main__":
    main()
