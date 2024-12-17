import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    cancer = load_breast_cancer()

    print("Cancer keys: \n", cancer.keys())
    print("Cancer dataset shape: ", cancer.data.shape)

    zipped = zip(map(str, cancer.target_names), map(int, np.bincount(cancer.target)))
    print("Sample counts per class: \n", {n: v for n, v in zipped})

    print("Cancer dataset features: \n", cancer.feature_names)

    # Split data set into training and test (75% train, 25% test)
    # Stratify to maintain proportions as dataset when splitting
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66
    )

    training_accuracy = []
    test_accuracy = []

    # k neighbors 1 to 10
    neighbors_settings = range(1, 11)

    for n_neighbors in neighbors_settings:
        # Build model and fit model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)

        # Record training accuracy
        training_accuracy.append(clf.score(X_train, y_train))

        # Record testing accuracy
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="Testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
