"""
Not meant to be optimal, just function split be section in the book.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def cancer_info():
    """
    For viewing keys, features, samples by class.
    """
    cancer = load_breast_cancer()

    print("Cancer keys: \n", cancer.keys())
    print("Cancer dataset shape: ", cancer.data.shape)

    zipped = zip(map(str, cancer.target_names), map(int, np.bincount(cancer.target)))
    print("Sample counts per class: \n", {n: v for n, v in zipped})

    print("Cancer dataset features: \n", cancer.feature_names)


def knn_accuracy():
    """
    For seeing KNeighborsClassifier model accuracy.
    """
    cancer = load_breast_cancer()

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


def linear_logistic():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )

    logreg = LogisticRegression().fit(X_train, y_train)
    print(f"Training set score: {logreg.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg.score(X_test,y_test):.3f}")

    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print(f"Training set score: {logreg100.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg100.score(X_test,y_test):.3f}")

    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print(f"Training set score: {logreg001.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg001.score(X_test,y_test):.3f}")

    plt.plot(logreg.coef_.T, "o", label="C=1")
    plt.plot(logreg100.coef_.T, "^", label="C=100")
    plt.plot(logreg001.coef_.T, "v", label="C=0.001")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient maginitude")
    plt.legend()
    plt.show()


def main():
    linear_logistic()


if __name__ == "__main__":
    main()
